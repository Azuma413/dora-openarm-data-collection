#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["rerun-sdk>=0.23", "pyarrow", "numpy", "pyyaml", "pillow"]
# ///
"""Play back a recorded episode in the Rerun viewer.

    scripts/view-episode.py data/webxr-pedestal/dataset            # list episodes
    scripts/view-episode.py data/webxr-pedestal/dataset 2          # view episode 2
    scripts/view-episode.py data/webxr-pedestal/dataset 2 --save x.rrd
"""

import argparse
import pathlib

import numpy as np
import pyarrow.parquet as pq
import rerun as rr
import yaml
from PIL import Image

JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]


def list_episodes(dataset: pathlib.Path) -> None:
    meta = yaml.safe_load((dataset / "metadata.yaml").read_text())
    for ep in meta.get("episodes", []):
        d = dataset / "episodes" / str(ep["id"])
        obs = d / "obs" / "arms" / "right" / "state.parquet"
        n, sec = 0, 0.0
        if obs.exists():
            ts = pq.read_table(obs, columns=["timestamp"])["timestamp"]
            ts = ts.cast("int64").to_numpy()
            n, sec = len(ts), (ts[-1] - ts[0]) / 1e9 if len(ts) else 0.0
        cams = sorted(p.name for p in (d / "cameras").glob("*")) if (d / "cameras").exists() else []
        mark = "OK  " if ep.get("success") else "FAIL"
        print(f"{ep['id']:>4} {mark} task={ep.get('task_index')} {sec:6.1f}s {n:5d} rows  cameras={cams}")


def log_arms(d: pathlib.Path) -> None:
    for kind in ("obs", "action"):
        for side_dir in sorted((d / kind / "arms").glob("*")):
            for f in side_dir.glob("*.parquet"):
                t = pq.read_table(f)
                ts = t["timestamp"].cast("int64").to_numpy()
                for field in ("qpos", "qvel", "qtorque"):
                    if field not in t.column_names:
                        continue
                    vals = np.array(t[field].to_pylist(), dtype=np.float32)
                    for j in range(vals.shape[1]):
                        name = JOINTS[j] if j < len(JOINTS) else f"j{j + 1}"
                        rr.send_columns(
                            f"{kind}/{side_dir.name}/{field}/{name}",
                            indexes=[rr.TimeColumn("time", timestamp=ts.astype("datetime64[ns]"))],
                            columns=rr.Scalars.columns(scalars=vals[:, j]),
                        )


def episode_depth_range(images: list[pathlib.Path]) -> tuple[float, float]:
    """One color range for the whole episode, from up to 20 sampled frames.

    Without it Rerun fits the colormap to each frame's own min/max, so the
    colors of a still background shift from frame to frame.
    """
    step = max(1, len(images) // 20)
    lows, highs = [], []
    for img in images[::step]:
        v = np.asarray(Image.open(img), dtype=np.uint16)
        v = v[v > 0]
        if v.size:
            lows.append(np.percentile(v, 1))
            highs.append(np.percentile(v, 99))
    if not lows:
        return (0.0, 3000.0)
    return (float(min(lows)), float(max(highs)))


def log_cameras(d: pathlib.Path) -> None:
    for cam in sorted((d / "cameras").glob("*")):
        images = sorted(p for p in cam.iterdir() if p.stem.isdigit())
        depth_range = episode_depth_range(images) if "depth" in cam.name else None
        if depth_range:
            print(f"{cam.name}: color range {depth_range[0]:.0f}-{depth_range[1]:.0f} mm")
        for img in images:
            if not img.stem.isdigit():
                continue
            rr.set_time("time", timestamp=np.datetime64(int(img.stem), "ns"))
            if "depth" in cam.name:
                # 16-bit PNG in mm (0 = no depth); as a plain image it is
                # almost black, so log it as a depth image with a colormap.
                depth = np.asarray(Image.open(img), dtype=np.uint16)
                rr.log(
                    f"cameras/{cam.name}",
                    rr.DepthImage(
                        depth,
                        meter=1000.0,
                        colormap="turbo",
                        depth_range=depth_range,
                    ),
                )
            else:
                rr.log(f"cameras/{cam.name}", rr.EncodedImage(path=img))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=pathlib.Path, help="e.g. data/webxr-pedestal/dataset")
    p.add_argument("episode", nargs="?", help="episode id; omit to list episodes")
    p.add_argument("--save", type=pathlib.Path, help="write an .rrd file instead of opening the viewer")
    args = p.parse_args()

    if args.episode is None:
        list_episodes(args.dataset)
        return
    d = args.dataset / "episodes" / args.episode
    if not d.is_dir():
        raise SystemExit(f"no such episode: {d}")

    rr.init(f"openarm-{args.dataset.parent.name}-{args.episode}", spawn=args.save is None)
    if args.save:
        rr.save(args.save)
    log_arms(d)
    if (d / "cameras").exists():
        log_cameras(d)


if __name__ == "__main__":
    main()
