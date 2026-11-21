#!/usr/bin/env python
"""Mine context-matched pairs for the bbox-action binding loss.

Adapts the context-matched pair search of ListenVLA (arXiv:2610.16313, Sec. III-C1)
from audio to the target bbox drawn on the front image. For an anchor frame i we
look for a donor frame j from *another* episode of the *same task* whose
non-bbox context matches i but whose target bbox and demonstrated actions differ:

- robot state: every joint within ``--joint-tol`` rad and both grippers within
  ``--gripper-tol`` (ListenVLA matches end-effector pose; joint space is used here
  because both arms are 7-DoF and the state is recorded in joint space),
- motion history: RMS difference of the relative joint trajectory over the
  preceding ``--history`` steps below ``--history-tol``, and the same
  moving/stationary flag,
- images: 8x8 average-pooled RGB fingerprints of the raw (bbox-free) front and
  wrist frames within ``--fingerprint-tol`` RMS,
- bbox: centers at least ``--min-bbox-shift`` px apart,
- actions: mean absolute difference of the first ``--steps`` normalized actions
  at least ``--min-action-diff``.

Among accepted donors the closest ``--top-k`` by context distance D are kept,
with confidence c = exp(-D) and weight
w = c * clip(bbox_center_shift / --bbox-shift-scale, 0, 1)
(the bbox analogue of ListenVLA's acoustic-separation factor, Eq. 5).

Output: ``<view>/meta/binding_pairs.parquet``, one row per (anchor, donor) with
the donor bbox and its raw (unnormalized) first ``--steps`` actions, so the
training loader can draw the donor bbox on the anchor image and build the
target A_i - A_j with the run's own normalization.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

FINGERPRINT_CAMERAS = ("observation.images.head_left", "observation.images.wrist_right")


def load_frames(view: Path) -> dict[str, np.ndarray]:
    cols = ["episode_index", "frame_index", "index", "task_index", "observation.state", "action", "target_bbox"]
    tables = [pq.read_table(p, columns=cols) for p in sorted((view / "data").glob("chunk-*/*.parquet"))]
    table = pa.concat_tables(tables)
    out = {c: table.column(c).to_numpy() for c in ("episode_index", "frame_index", "index", "task_index")}
    for c in ("observation.state", "action", "target_bbox"):
        out[c] = np.stack(table.column(c).to_numpy(zero_copy_only=False)).astype(np.float32)
    order = np.lexsort((out["frame_index"], out["episode_index"]))
    return {k: v[order] for k, v in out.items()}


def episode_video_meta(view: Path, key: str) -> dict[int, tuple[Path, float]]:
    info = json.loads((view / "meta" / "info.json").read_text())
    pattern = info["video_path"]
    meta = {}
    for p in sorted((view / "meta" / "episodes").glob("chunk-*/*.parquet")):
        t = pq.read_table(p).to_pydict()
        for i, ep in enumerate(t["episode_index"]):
            path = view / pattern.format(
                video_key=key,
                chunk_index=int(t[f"videos/{key}/chunk_index"][i]),
                file_index=int(t[f"videos/{key}/file_index"][i]),
            )
            meta[int(ep)] = (path, float(t[f"videos/{key}/from_timestamp"][i]))
    return meta


def fingerprints_for_episode(path: Path, from_ts: float, n: int) -> np.ndarray:
    out = np.zeros((n, 8 * 8 * 3), dtype=np.float32)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        target = int(from_ts / stream.time_base)
        if from_ts > 0:
            container.seek(target, stream=stream)
        k = 0
        for frame in container.decode(video=0):
            if frame.pts is not None and frame.pts < target:
                continue
            small = frame.reformat(width=64, height=36, format="rgb24").to_ndarray()
            out[k] = cv2.resize(small, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
            k += 1
            if k >= n:
                break
    if k != n:
        raise RuntimeError(f"{path}: expected {n} frames from t={from_ts}, decoded {k}")
    return out


def compute_fingerprints(view: Path, frames: dict[str, np.ndarray], workers: int) -> dict[str, np.ndarray]:
    episodes = frames["episode_index"]
    result = {}
    for key in FINGERPRINT_CAMERAS:
        meta = episode_video_meta(view, key)
        fp = np.zeros((len(episodes), 192), dtype=np.float32)
        uniq = np.unique(episodes)

        def job(ep):
            rows = np.flatnonzero(episodes == ep)
            path, ts = meta[int(ep)]
            return rows, fingerprints_for_episode(path, ts, len(rows))

        with ThreadPoolExecutor(workers) as pool:
            for rows, values in pool.map(job, uniq):
                fp[rows] = values
        result[key] = fp
        print(f"fingerprints: {key} done", flush=True)
    return result


def normalize_q01_q99(x: np.ndarray, stats: dict) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return np.clip(2.0 * (x - q01) / np.maximum(q99 - q01, 1e-6) - 1.0, -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("view", type=Path, help="Training view built by prepare_dataset.py.")
    parser.add_argument("--steps", type=int, default=8, help="Action steps supervised by the binding loss.")
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--joint-tol", type=float, default=0.05, help="rad, per joint")
    parser.add_argument("--gripper-tol", type=float, default=0.004, help="state units, per gripper")
    parser.add_argument("--history-tol", type=float, default=0.02, help="rad, RMS of relative joint trajectory")
    parser.add_argument("--moving-threshold", type=float, default=0.002, help="rad/step mean joint speed")
    parser.add_argument("--fingerprint-tol", type=float, default=0.20)
    parser.add_argument("--min-bbox-shift", type=float, default=20.0, help="px between bbox centers")
    parser.add_argument("--bbox-shift-scale", type=float, default=80.0, help="px for full weight")
    parser.add_argument("--min-action-diff", type=float, default=0.02, help="normalized units")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    view = args.view.expanduser().resolve()
    info = json.loads((view / "meta" / "info.json").read_text())
    stats = json.loads((view / "meta" / "stats.json").read_text())
    names = info["features"]["observation.state"]["names"]
    gripper = np.array(["gripper" in n for n in names])

    f = load_frames(view)
    n = len(f["episode_index"])
    ep = f["episode_index"]
    state = f["observation.state"]
    action = f["action"]
    bbox = f["target_bbox"]
    centers = np.stack([(bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2], axis=1)
    action_n = normalize_q01_q99(action, stats["action"])

    # Position of each row within its episode, and episode lengths.
    starts = np.r_[0, np.flatnonzero(np.diff(ep)) + 1]
    lengths = np.diff(np.r_[starts, n])
    ep_start = np.repeat(starts, lengths)
    pos = np.arange(n) - ep_start
    ep_len = np.repeat(lengths, lengths)

    # Rows with a full action chunk and a full motion history inside the episode.
    eligible = (pos >= args.history) & (pos + args.steps <= ep_len)
    rows = np.flatnonzero(eligible)

    chunk = np.stack([action_n[rows + s] for s in range(args.steps)], axis=1)  # (m, steps, D)
    raw_chunk = np.stack([action[rows + s] for s in range(args.steps)], axis=1)
    hist = np.stack([state[rows - h] - state[rows] for h in range(1, args.history + 1)], axis=1)
    hist_joint = hist[..., ~gripper]
    speed = np.abs(state[rows] - state[rows - 1])[:, ~gripper].mean(axis=1)
    moving = speed > args.moving_threshold

    fps = compute_fingerprints(view, f, args.workers)
    fp = {k: v[rows] for k, v in fps.items()}

    tol = np.where(gripper, args.gripper_tol, args.joint_tol).astype(np.float32)
    tree = cKDTree(state[rows] / tol)
    out = {k: [] for k in (
        "anchor_index", "anchor_episode", "anchor_frame", "donor_index", "donor_episode", "donor_frame",
        "donor_bbox", "donor_action", "anchor_bbox", "bbox_shift_px", "action_diff", "context_distance",
        "confidence", "weight",
    )}
    n_with_pair = 0
    for a in range(len(rows)):
        cand = np.asarray(tree.query_ball_point(state[rows[a]] / tol, r=1.0, p=np.inf))
        if cand.size == 0:
            continue
        r_a = rows[a]
        cand = cand[(ep[rows[cand]] != ep[r_a]) & (f["task_index"][rows[cand]] == f["task_index"][r_a])]
        cand = cand[moving[cand] == moving[a]]
        if cand.size == 0:
            continue
        hist_rms = np.sqrt(((hist_joint[cand] - hist_joint[a]) ** 2).mean(axis=(1, 2)))
        cand, hist_rms = cand[hist_rms <= args.history_tol], hist_rms[hist_rms <= args.history_tol]
        if cand.size == 0:
            continue
        fp_d = np.stack([np.sqrt(((fp[k][cand] - fp[k][a]) ** 2).mean(axis=1)) for k in fp], axis=1)
        keep = (fp_d <= args.fingerprint_tol).all(axis=1)
        cand, hist_rms, fp_d = cand[keep], hist_rms[keep], fp_d[keep]
        if cand.size == 0:
            continue
        shift = np.linalg.norm(centers[rows[cand]] - centers[r_a], axis=1)
        adiff = np.abs(chunk[cand] - chunk[a]).mean(axis=(1, 2))
        keep = (shift >= args.min_bbox_shift) & (adiff >= args.min_action_diff)
        cand, hist_rms, fp_d, shift, adiff = cand[keep], hist_rms[keep], fp_d[keep], shift[keep], adiff[keep]
        if cand.size == 0:
            continue
        state_d = np.abs((state[rows[cand]] - state[r_a]) / tol).mean(axis=1)
        dist = (state_d + hist_rms / args.history_tol) / 2.0 + fp_d.mean(axis=1) / args.fingerprint_tol
        best = np.argsort(dist)[: args.top_k]
        n_with_pair += 1
        for b in best:
            d = rows[cand[b]]
            conf = float(np.exp(-dist[b]))
            out["anchor_index"].append(int(f["index"][r_a]))
            out["anchor_episode"].append(int(ep[r_a]))
            out["anchor_frame"].append(int(f["frame_index"][r_a]))
            out["donor_index"].append(int(f["index"][d]))
            out["donor_episode"].append(int(ep[d]))
            out["donor_frame"].append(int(f["frame_index"][d]))
            out["donor_bbox"].append(bbox[d].tolist())
            out["donor_action"].append(raw_chunk[cand[b]].reshape(-1).tolist())
            out["anchor_bbox"].append(bbox[r_a].tolist())
            out["bbox_shift_px"].append(float(shift[b]))
            out["action_diff"].append(float(adiff[b]))
            out["context_distance"].append(float(dist[b]))
            out["confidence"].append(conf)
            out["weight"].append(conf * float(np.clip(shift[b] / args.bbox_shift_scale, 0.0, 1.0)))

    table = pa.table({
        **{k: pa.array(v) for k, v in out.items() if k not in {"donor_bbox", "donor_action", "anchor_bbox"}},
        "donor_bbox": pa.array(out["donor_bbox"], type=pa.list_(pa.float32(), 4)),
        "anchor_bbox": pa.array(out["anchor_bbox"], type=pa.list_(pa.float32(), 4)),
        "donor_action": pa.array(out["donor_action"], type=pa.list_(pa.float32())),
    })
    meta = {
        "steps": args.steps,
        "action_dim": int(action.shape[1]),
        "eligible_anchors": int(len(rows)),
        "anchors_with_pair": int(n_with_pair),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    table = table.replace_schema_metadata({"binding": json.dumps(meta)})
    output = args.output or view / "meta" / "binding_pairs.parquet"
    pq.write_table(table, output)
    print(json.dumps(meta, indent=2))
    if table.num_rows:
        for c in ("bbox_shift_px", "action_diff", "confidence", "weight"):
            v = table.column(c).to_numpy()
            print(f"{c}: mean={v.mean():.4f} p10={np.percentile(v, 10):.4f} p90={np.percentile(v, 90):.4f}")
        per_ep = np.bincount(np.asarray(out["anchor_episode"]))
        print(f"anchors per episode (pairs rows): min={per_ep[per_ep > 0].min()} max={per_ep.max()}")
    print(f"wrote {output} rows={table.num_rows}")


if __name__ == "__main__":
    main()
