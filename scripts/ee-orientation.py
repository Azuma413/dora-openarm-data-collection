#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["openarm-control>=0.6.0", "pyarrow", "numpy", "scipy"]
# ///
"""Print an arm's end-effector orientation at one moment of a recorded episode.

    scripts/ee-orientation.py data/webxr-pedestal-zed/dataset 3 12.4
    scripts/ee-orientation.py data/webxr-pedestal-zed/dataset 3 12.4 --side right

The time is seconds from the first frame of the camera (head_left by default),
the same clock as the video in scripts/review-episodes.py, so pause the video
there on the frame whose hand orientation you want and pass its time.

The joint positions nearest that time (obs = measured, or --source action =
commanded) go through the same forward kinematics as dora-openarm-ik, so the
printed quaternion is in the IK target frame. Pass it to the WebXR node as
FIXED_ORIENTATION_LEFT (or --fixed-orientation-left) to hold the hand there.
"""

import argparse
import pathlib

import numpy as np
import pyarrow.parquet as pq
from openarm_control import Kinematics, register_common_args, setup_from_args
from scipy.spatial.transform import Rotation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=pathlib.Path, help="Dataset directory (holds episodes/)")
    ap.add_argument("episode", help="Episode id, e.g. 3")
    ap.add_argument("time", type=float, help="Seconds from the camera's first frame")
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--source", choices=("obs", "action"), default="obs")
    ap.add_argument("--camera", default="head_left")
    # --xml etc., so the model matches the IK node's if it is given one.
    register_common_args(ap)
    args = ap.parse_args()

    ep_dir = args.dataset / "episodes" / args.episode
    frames = sorted((ep_dir / "cameras" / args.camera).glob("*.jpeg"), key=lambda f: int(f.stem))
    if not frames:
        ap.error(f"no {args.camera} frames in {ep_dir}")
    target_ns = int(frames[0].stem) + round(args.time * 1e9)

    table = pq.read_table(ep_dir / args.source / "arms" / args.side / "state.parquet")
    stamps = table.column("timestamp").cast("int64").to_numpy()
    i = int(np.argmin(np.abs(stamps - target_ns)))
    qpos = np.asarray(table.column("qpos")[i].as_py(), dtype=np.float32)

    kin = Kinematics(setup_from_args(args))
    pose = kin.fk(args.side, qpos)
    qw, qx, qy, qz = (float(v) for v in pose[3:7])
    rotation = Rotation.from_quat([qx, qy, qz, qw])
    tool_z = rotation.apply([0.0, 0.0, 1.0])

    print(f"episode {args.episode}  t={args.time:.2f}s  ({args.source} row {i}, "
          f"{(stamps[i] - target_ns) / 1e6:+.0f} ms from the frame)")
    print(f"qpos      : {np.array2string(qpos, precision=3, separator=', ')}")
    print(f"position  : {np.array2string(pose[:3], precision=3, separator=', ')} m")
    print(f"euler xyz : {np.array2string(rotation.as_euler('xyz', degrees=True), precision=1, separator=', ')} deg")
    print(f"EE z axis : {np.array2string(tool_z, precision=3, separator=', ')}")
    value = f"{qw:.4f},{qx:.4f},{qy:.4f},{qz:.4f}"
    print()
    print(f"FIXED_ORIENTATION_{args.side.upper()}: \"{value}\"   # qw,qx,qy,qz")


if __name__ == "__main__":
    main()
