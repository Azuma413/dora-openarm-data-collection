#!/usr/bin/env python
"""Decode the training view once into low-resolution frame arrays for the small flow policy.

Writes to OUT:
  head_left.npy    uint8 (N, 144, 256, 3)  RGB, raw (no bbox drawn)
  wrist_right.npy  uint8 (N, 160, 256, 3)
  lowdim.npz       state, action (N, 16) float32; episode_index, frame_index (N,) int64;
                   bbox (N, 4) float32 in head_left 1280x720 px; seg_end (N,) int64 = first frame
                   of the next target segment in the same episode (or a large value)
Row n is dataset index n of the LeRobot view.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMS = {"observation.images.head_left": ("head_left", 144, 256), "observation.images.wrist_right": ("wrist_right", 160, 256)}


def decode_job(args):
    path, from_ts, start, length, out_file, h, w, n_total = args
    arr = np.load(out_file, mmap_mode="r+")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        target = int(round(from_ts / stream.time_base))
        if from_ts > 0:
            container.seek(target, stream=stream)
        k = 0
        for frame in container.decode(video=0):
            if frame.pts is not None and frame.pts < target - 1:
                continue
            arr[start + k] = frame.reformat(width=w, height=h, format="rgb24", interpolation="AREA").to_ndarray()
            k += 1
            if k >= length:
                break
    arr.flush()
    if k != length:
        raise RuntimeError(f"{path}: decoded {k}/{length} frames from t={from_ts}")
    return start, length


def main():
    p = argparse.ArgumentParser()
    p.add_argument("view", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    view, out = a.view.resolve(), a.out
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((view / "meta" / "info.json").read_text())
    n = int(info["total_frames"])

    cols = ["index", "episode_index", "frame_index", "observation.state", "action", "target_bbox"]
    t = pa.concat_tables([pq.read_table(f, columns=cols) for f in sorted((view / "data").glob("chunk-*/*.parquet"))])
    idx = t.column("index").to_numpy()
    if not np.array_equal(np.sort(idx), np.arange(n)):
        raise SystemExit("dataset index is not 0..N-1")
    order = np.argsort(idx)
    ep = t.column("episode_index").to_numpy()[order]
    fr = t.column("frame_index").to_numpy()[order]
    stack = lambda c: np.stack(t.column(c).to_numpy(zero_copy_only=False))[order].astype(np.float32)
    seg_end = np.full(n, 1 << 40, dtype=np.int64)
    with (view / "meta" / "target_bbox_segments.csv").open() as fh:
        switches = {}
        for row in csv.DictReader(fh):
            switches.setdefault(int(row["episode_index"]), []).append(int(row["frame_end"]))
    for e, ends in switches.items():
        for s in sorted(ends)[:-1]:
            sel = (ep == e) & (fr < s)
            seg_end[sel] = np.minimum(seg_end[sel], s)
    np.savez(out / "lowdim.npz", state=stack("observation.state"), action=stack("action"), bbox=stack("target_bbox"),
             episode_index=ep, frame_index=fr, seg_end=seg_end)

    meta = pq.read_table(sorted((view / "meta" / "episodes").glob("chunk-*/*.parquet"))[0]).to_pandas()
    for m in sorted((view / "meta" / "episodes").glob("chunk-*/*.parquet"))[1:]:
        meta = meta._append(pq.read_table(m).to_pandas())
    jobs = []
    for key, (name, h, w) in CAMS.items():
        f = out / f"{name}.npy"
        np.lib.format.open_memmap(f, mode="w+", dtype=np.uint8, shape=(n, h, w, 3)).flush()
        for _, r in meta.iterrows():
            path = view / info["video_path"].format(video_key=key, chunk_index=int(r[f"videos/{key}/chunk_index"]),
                                                    file_index=int(r[f"videos/{key}/file_index"]))
            jobs.append((path, float(r[f"videos/{key}/from_timestamp"]), int(r["dataset_from_index"]), int(r["length"]),
                         f, h, w, n))
    done = 0
    with ProcessPoolExecutor(a.workers) as pool:
        for start, length in pool.map(decode_job, jobs):
            done += 1
            if done % 20 == 0:
                print(f"decoded {done}/{len(jobs)} episode-videos", flush=True)
    print(f"done: {out} N={n}", flush=True)


if __name__ == "__main__":
    main()
