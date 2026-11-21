#!/usr/bin/env python
"""Build the MolmoAct2 training view of a curated OpenArm LeRobot v3 dataset.

The source dataset (for example ``~/openarm-datasets/webxr-pedestal-zed-lerobot-curated``)
records four RGB videos plus the ZED depth map as an ``image`` column. MolmoAct2's
LeRobot loader decodes every video/image feature a dataset declares on every
sample, so this script writes a slimmer dataset next to it that

- keeps only the cameras the policy uses (``head_left`` for the front view with
  the target bbox drawn on it, ``wrist_right``); the videos are symlinked, not
  re-encoded,
- drops the large ``observation.images.head_depth`` column (the ZED depth is
  turned into MolmoAct2 depth tokens by ``generate_zed_depth_annotation.py``,
  which reads it from the source dataset), and
- adds ``target_bbox`` = ``[x_min, y_min, x_max, y_max]`` in ``head_left`` pixels
  from ``meta/target_bbox_frames.csv`` so the training loader can draw it.

Row order and parquet file layout are identical to the source, so frame indices,
episode metadata and video timestamps stay valid.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_CAMERAS = ("observation.images.head_left", "observation.images.wrist_right")
DEPTH_KEY = "observation.images.head_depth"
BBOX_KEY = "target_bbox"
BBOX_NAMES = ["x_min", "y_min", "x_max", "y_max"]


def load_bboxes(csv_path: Path) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    boxes: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            key = (int(row["episode_index"]), int(row["frame_index"]))
            boxes[key] = tuple(float(row[name]) for name in BBOX_NAMES)  # type: ignore[assignment]
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Curated LeRobot v3 dataset directory.")
    parser.add_argument("output", type=Path, help="Output dataset directory (its name is the repo id).")
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src: Path = args.source.expanduser().resolve()
    dst: Path = args.output.expanduser().resolve()
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite to rebuild it.")
        shutil.rmtree(dst)

    info = json.loads((src / "meta" / "info.json").read_text())
    features = info["features"]
    visual = [k for k, v in features.items() if v.get("dtype") in {"video", "image"}]
    missing = [k for k in args.cameras if k not in features]
    if missing:
        raise SystemExit(f"cameras {missing} not in source features {visual}")
    dropped = [k for k in visual if k not in args.cameras]

    boxes = load_bboxes(src / "meta" / "target_bbox_frames.csv")

    # meta/: copy everything, then rewrite info.json and stats.json.
    shutil.copytree(src / "meta", dst / "meta")
    new_features = {k: v for k, v in features.items() if k not in dropped}
    new_features[BBOX_KEY] = {"dtype": "float32", "shape": [4], "names": BBOX_NAMES}
    info["features"] = new_features
    info["molmoact2_view"] = {
        "source": str(src),
        "cameras": list(args.cameras),
        "dropped_features": dropped,
        "bbox_feature": BBOX_KEY,
        "bbox_camera": "observation.images.head_left",
    }
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    stats_path = dst / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats = {k: v for k, v in stats.items() if k not in dropped}
        stats_path.write_text(json.dumps(stats, indent=4) + "\n")

    # videos/: symlink only the kept video keys.
    for key in args.cameras:
        if features[key].get("dtype") != "video":
            continue
        (dst / "videos").mkdir(parents=True, exist_ok=True)
        (dst / "videos" / key).symlink_to(src / "videos" / key, target_is_directory=True)

    # data/: same files and rows, without the dropped image columns, plus target_bbox.
    n_rows = 0
    n_missing = 0
    for parquet_path in sorted((src / "data").glob("chunk-*/*.parquet")):
        rel = parquet_path.relative_to(src)
        schema_names = pq.read_schema(parquet_path).names
        columns = [c for c in schema_names if c not in dropped]
        table = pq.read_table(parquet_path, columns=columns)
        episodes = table.column("episode_index").to_numpy()
        frames = table.column("frame_index").to_numpy()
        bbox = np.full((table.num_rows, 4), np.nan, dtype=np.float32)
        for i, (ep, fr) in enumerate(zip(episodes, frames)):
            box = boxes.get((int(ep), int(fr)))
            if box is not None:
                bbox[i] = box
        n_missing += int(np.isnan(bbox[:, 0]).sum())
        flat = pa.array(bbox.reshape(-1), type=pa.float32())
        table = table.append_column(BBOX_KEY, pa.FixedSizeListArray.from_arrays(flat, 4))
        # The source schema metadata lists the HF features of every source column; it no
        # longer matches after dropping columns, so let the loader use info.json instead.
        table = table.replace_schema_metadata(None)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)
        n_rows += table.num_rows

    if n_rows != int(info.get("total_frames", n_rows)):
        raise SystemExit(f"wrote {n_rows} rows but info.json says total_frames={info['total_frames']}")
    if n_missing:
        raise SystemExit(f"{n_missing} frames have no bbox in target_bbox_frames.csv")
    print(f"wrote {dst} rows={n_rows} cameras={args.cameras} dropped={dropped}")


if __name__ == "__main__":
    main()
