#!/usr/bin/env python
"""Generate MolmoAct2 depth-token annotations from recorded ZED depth.

``lib/molmoact2/experiments/scripts/generate_depth_annotation.py`` estimates depth
from RGB with Depth-Anything-V2 and tokenizes it with the released depth VQ-VAE.
This script keeps that pipeline (VQ-VAE, 10x10 code grid, RGB-driven adaptive
buffer update) but replaces the Depth-Anything estimate with the measured ZED
depth stored in ``observation.images.head_depth`` (uint16, mm, registered to the
left camera).

The VQ-VAE was trained on Depth-Anything output, which is *relative inverse
depth* (near = bright) min-max normalized per frame. The ZED map is converted to
the same convention: invalid pixels (0) are masked, depth is inverted
(1/m), normalized per frame with robust percentiles, holes are inpainted, and
the result is quantized to uint8 before tokenization.

Output is a depth companion dataset that MolmoAct2's LeRobot loader finds under
``${LEROBOT_DEPTH_DATA_ROOT}/<repo_id>``. It carries no visual features, so the
loader does not decode any video when it reads depth codes.

Use ``--compare-depth-anything N`` to tokenize N sampled frames with both
Depth-Anything and ZED and report how close they are (rank correlation of the
depth maps, VQ-VAE reconstruction error, code agreement) before committing to a
full run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from common import load_upstream_depth_annotation

DEPTH_KEY = "observation.images.head_depth"
RGB_KEY = "observation.images.head_left"
KEEP_COLUMNS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]


def decode_depth_mm(cell) -> np.ndarray:
    data = cell["bytes"] if isinstance(cell, dict) else cell
    depth = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("failed to decode depth PNG")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def zed_to_relative_inverse_depth(
    depth_mm: np.ndarray,
    *,
    min_mm: float,
    max_mm: float,
    lo_pct: float,
    hi_pct: float,
) -> tuple[np.ndarray, float]:
    """Return a uint8 relative inverse-depth image (near = 255) and the invalid fraction."""
    depth = depth_mm.astype(np.float32)
    valid = (depth >= min_mm) & (depth <= max_mm) & np.isfinite(depth)
    invalid_frac = float(1.0 - valid.mean())
    if valid.sum() < 64:
        return np.zeros(depth.shape, np.uint8), invalid_frac
    inv = np.zeros_like(depth)
    inv[valid] = 1000.0 / depth[valid]
    lo, hi = np.percentile(inv[valid], [lo_pct, hi_pct])
    scaled = np.clip((inv - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    vis = (scaled * 255.0).astype(np.uint8)
    if not valid.all():
        vis = cv2.inpaint(vis, (~valid).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return vis, invalid_frac


@torch.no_grad()
def tokenize(vae, vis_u8: list[np.ndarray], size: int, device: torch.device, batch: int = 64) -> list[np.ndarray]:
    codes: list[np.ndarray] = []
    for start in range(0, len(vis_u8), batch):
        chunk = [cv2.resize(v, (size, size), interpolation=cv2.INTER_AREA) for v in vis_u8[start : start + batch]]
        x = torch.from_numpy(np.stack(chunk)).float().div(255.0).sub(0.5).div(0.5)[:, None].to(device)
        grids = vae(img=x, return_indices=True)
        codes.extend(g.cpu().numpy().astype(np.int16) for g in grids)
    return codes


def buffer_codes(up, code_grids, rgb_resized, grid: int, threshold: float):
    """Same adaptive buffer update as the upstream ``process_episode_frames``."""
    token_buffer = code_grids[0].copy()
    depth_codes, buffers, masks = [], [], []
    for idx, code_grid in enumerate(code_grids):
        if idx == 0:
            updated = np.ones((grid, grid), dtype=bool)
        else:
            sim = up.cosine_similarity_patches(rgb_resized[idx - 1], rgb_resized[idx], 32)
            updated = sim < threshold
            token_buffer[updated] = code_grid[updated]
        depth_codes.append(code_grid.reshape(-1).astype(np.int16))
        buffers.append(token_buffer.reshape(-1).astype(np.int16))
        masks.append(updated.reshape(-1))
    return depth_codes, buffers, masks


def write_companion_meta(view_root: Path, out_root: Path, args, grid_size: int) -> None:
    shutil.copytree(view_root / "meta", out_root / "meta", dirs_exist_ok=True)
    info = json.loads((view_root / "meta" / "info.json").read_text())
    features = {k: v for k, v in info["features"].items() if k in KEEP_COLUMNS}
    for name, dtype, desc in (
        ("depth_codes", "int16", "Per-frame VQ-VAE depth codes from ZED depth."),
        ("buffer_codes", "int16", "Depth token buffer state after RGB-driven patch updates."),
        ("depth_updated_mask", "bool", "Mask of depth patches updated in the buffer."),
    ):
        features[name] = {"dtype": dtype, "shape": [grid_size], "names": None, "description": desc}
    info["features"] = features
    info["depth_vae"] = {
        "vae_image_size": args.vae_image_size,
        "patch_size": 32,
        "grid_h": args.vae_image_size // 32,
        "grid_w": args.vae_image_size // 32,
        "num_tokens": 128,
        "cosine_threshold": args.threshold,
        "depth_encoder": "zed",
        "depth_source_key": DEPTH_KEY,
        "zed_valid_mm": [args.min_mm, args.max_mm],
        "zed_percentiles": [args.lo_pct, args.hi_pct],
        "image_key": RGB_KEY,
    }
    (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    stats_path = out_root / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats_path.write_text(json.dumps({k: v for k, v in stats.items() if k in KEEP_COLUMNS}, indent=4) + "\n")


def compare_with_depth_anything(up, vae, source: Path, args, device, out_dir: Path) -> None:
    from scipy.stats import spearmanr

    depth_model = up.DepthAnythingV2().to(device).eval()
    depth_model.load_state_dict(torch.load(up.resolve_checkpoint(None, up.DEFAULT_DEPTH_CKPT), map_location="cpu"))
    info = json.loads((source / "meta" / "info.json").read_text())
    video_meta = up.load_episode_metadata(source, RGB_KEY)
    files = sorted((source / "data").glob("chunk-*/*.parquet"))
    rng = np.random.default_rng(0)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for parquet_path in rng.choice(files, size=min(args.compare_depth_anything, len(files)), replace=False):
        table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", DEPTH_KEY])
        r = int(rng.integers(table.num_rows))
        ep = int(table.column("episode_index")[r].as_py())
        fr = int(table.column("frame_index")[r].as_py())
        meta = video_meta[ep]
        frames = up.decode_video_frames(up.video_path_for_episode(source, info, RGB_KEY, meta), meta["from_timestamp"], fr + 1)
        bgr = frames[fr]
        zed_vis, invalid = zed_to_relative_inverse_depth(
            decode_depth_mm(table.column(DEPTH_KEY)[r].as_py()),
            min_mm=args.min_mm, max_mm=args.max_mm, lo_pct=args.lo_pct, hi_pct=args.hi_pct,
        )
        tensor, (h, w) = up.depth_image_to_tensor(bgr, args.depth_input_size, device)
        with torch.no_grad():
            da = depth_model(tensor[None])[0]
            da = torch.nn.functional.interpolate(da[None, None], (h, w), mode="bilinear", align_corners=True)[0, 0].cpu().numpy()
        da_vis = ((da - da.min()) / (da.max() - da.min() + 1e-8) * 255).astype(np.uint8)
        zed_codes, da_codes = tokenize(vae, [zed_vis, da_vis], args.vae_image_size, device)
        with torch.no_grad():
            recon = []
            for codes in (zed_codes, da_codes):
                img = vae.decode(torch.from_numpy(codes.astype(np.int64))[None].to(device))
                recon.append(((img[0, 0].clamp(-1, 1) * 0.5 + 0.5) * 255).cpu().numpy())
        small = lambda v: cv2.resize(v, (args.vae_image_size,) * 2, interpolation=cv2.INTER_AREA).astype(np.float32)
        rho = spearmanr(small(zed_vis).ravel(), small(da_vis).ravel()).correlation
        rows.append({
            "episode": ep, "frame": fr, "zed_invalid_frac": invalid, "spearman_zed_vs_da": float(rho),
            "recon_mae_zed": float(np.abs(recon[0] - small(zed_vis)).mean() / 255),
            "recon_mae_da": float(np.abs(recon[1] - small(da_vis)).mean() / 255),
            "code_agreement": float((zed_codes == da_codes).mean()),
        })
        panel = np.concatenate([
            cv2.resize(bgr, (320, 180)),
            cv2.applyColorMap(cv2.resize(zed_vis, (320, 180)), cv2.COLORMAP_INFERNO),
            cv2.applyColorMap(cv2.resize(da_vis, (320, 180)), cv2.COLORMAP_INFERNO),
            cv2.applyColorMap(cv2.resize(recon[0].astype(np.uint8), (320, 180)), cv2.COLORMAP_INFERNO),
        ], axis=1)
        cv2.imwrite(str(out_dir / f"ep{ep:03d}_fr{fr:05d}.png"), panel)
        print(json.dumps(rows[-1]))
    keys = [k for k in rows[0] if k not in {"episode", "frame"}]
    summary = {k: float(np.mean([row[k] for row in rows])) for k in keys}
    (out_dir / "summary.json").write_text(json.dumps({"mean": summary, "frames": rows}, indent=2))
    print("mean:", json.dumps(summary))
    print(f"panels (RGB | ZED | Depth-Anything | VQ-VAE recon of ZED codes) in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Curated dataset with observation.images.head_depth.")
    parser.add_argument("view", type=Path, help="Training view built by prepare_dataset.py (same row layout).")
    parser.add_argument("output", type=Path, help="Depth companion root, i.e. ${LEROBOT_DEPTH_DATA_ROOT}/<repo_id>.")
    parser.add_argument("--vae-image-size", type=int, default=320)
    parser.add_argument("--threshold", type=float, default=0.996, help="RGB patch cosine threshold for updates.")
    parser.add_argument("--depth-input-size", type=int, default=518, help="Only for --compare-depth-anything.")
    parser.add_argument("--min-mm", type=float, default=150.0)
    parser.add_argument("--max-mm", type=float, default=5000.0)
    parser.add_argument("--lo-pct", type=float, default=1.0)
    parser.add_argument("--hi-pct", type=float, default=99.0)
    parser.add_argument("--compare-depth-anything", type=int, default=0, metavar="N")
    parser.add_argument("--compare-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    view = args.view.expanduser().resolve()
    out_root = args.output.expanduser().resolve()
    up = load_upstream_depth_annotation()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = args.vae_image_size // 32
    vae = up.VQVAE(image_size=args.vae_image_size).to(device).eval()
    state = torch.load(up.resolve_checkpoint(None, up.DEFAULT_VAE_CKPT), map_location="cpu")["weights"]
    vae.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})

    if args.compare_depth_anything:
        compare_with_depth_anything(up, vae, source, args, device, args.compare_dir or out_root.with_name(out_root.name + "_compare"))
        return

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    write_companion_meta(view, out_root, args, grid * grid)

    info = json.loads((source / "meta" / "info.json").read_text())
    video_meta = up.load_episode_metadata(source, RGB_KEY)
    files = sorted((source / "data").glob("chunk-*/*.parquet"))
    invalid_fracs = []
    for i, parquet_path in enumerate(files, 1):
        rel = parquet_path.relative_to(source)
        out_path = out_root / rel
        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(files)}] skip {rel}")
            continue
        view_table = pq.read_table(view / rel, columns=KEEP_COLUMNS)
        src_table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", DEPTH_KEY])
        if src_table.num_rows != view_table.num_rows:
            raise RuntimeError(f"{rel}: row count differs between source and view")
        episodes = src_table.column("episode_index").to_numpy()
        frames = src_table.column("frame_index").to_numpy()
        depth_cells = src_table.column(DEPTH_KEY).to_pylist()
        n = src_table.num_rows
        out_codes = [None] * n
        out_buffers = [None] * n
        out_masks = [None] * n
        for ep in np.unique(episodes):
            rows = np.flatnonzero(episodes == ep)
            rows = rows[np.argsort(frames[rows])]
            meta = video_meta[int(ep)]
            bgr = up.decode_video_frames(
                up.video_path_for_episode(source, info, RGB_KEY, meta), meta["from_timestamp"], len(rows)
            )
            rgb_resized = [cv2.resize(b, (args.vae_image_size,) * 2, interpolation=cv2.INTER_LINEAR) for b in bgr]
            vis = []
            for r in rows:
                v, invalid = zed_to_relative_inverse_depth(
                    decode_depth_mm(depth_cells[r]),
                    min_mm=args.min_mm, max_mm=args.max_mm, lo_pct=args.lo_pct, hi_pct=args.hi_pct,
                )
                vis.append(v)
                invalid_fracs.append(invalid)
            grids = tokenize(vae, vis, args.vae_image_size, device)
            codes, buffers, masks = buffer_codes(up, grids, rgb_resized, grid, args.threshold)
            for j, r in enumerate(rows):
                out_codes[r] = codes[j].tolist()
                out_buffers[r] = buffers[j].tolist()
                out_masks[r] = masks[j].tolist()
        table = view_table
        table = table.append_column("depth_codes", pa.array(out_codes, type=pa.list_(pa.int16())))
        table = table.append_column("buffer_codes", pa.array(out_buffers, type=pa.list_(pa.int16())))
        table = table.append_column("depth_updated_mask", pa.array(out_masks, type=pa.list_(pa.bool_())))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)
        print(f"[{i}/{len(files)}] wrote {rel} rows={n} mean_invalid={np.mean(invalid_fracs[-n:]):.3f}", flush=True)
    print(f"done: {out_root}")


if __name__ == "__main__":
    main()
