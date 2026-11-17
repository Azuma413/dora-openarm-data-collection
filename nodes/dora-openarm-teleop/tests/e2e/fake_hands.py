#!/usr/bin/env python3
"""Scripted stand-in for the webxr node: two hands and their grips, per tick.

Timeline (seconds from the first tick), right hand only; the left hand stays
still with its grip released the whole time:
  0.0 - 1.0  still, grip released (the clutch's settle wait, then armed)
  1.0 -      grip squeezed (engage)
  1.5 - 1.5+MOVE_S  hand moves +MOVE_X along x and turns ROT_DEG about the
             arm_origin z axis, both at a constant rate
  then       still, grip still squeezed
"""

import os
import time

import numpy as np
import pyarrow as pa
from dora import Node

from dora_openarm_teleop import arrow, frames, model_info

MOVE_X = float(os.environ.get("MOVE_X", "0.10"))
MOVE_S = float(os.environ.get("MOVE_S", "1.0"))
ROT_DEG = float(os.environ.get("ROT_DEG", "0"))


def turned(quat: np.ndarray, degrees: float) -> np.ndarray:
    """quat (w, x, y, z) turned about the frame's z axis, applied on the left."""
    half = np.radians(degrees) / 2.0
    return frames.quat_multiply(np.array([np.cos(half), 0.0, 0.0, np.sin(half)]), quat)


def main() -> None:
    home = model_info.home_ee_poses("demo")
    node = Node()
    start = None
    for event in node:
        if event["type"] != "INPUT" or event["id"] != "tick":
            continue
        now = time.monotonic()
        start = now if start is None else start
        t = now - start
        right = home["right"].copy()
        progress = float(np.clip((t - 1.5) / MOVE_S, 0.0, 1.0))
        right[0] += MOVE_X * progress
        right[3:7] = turned(right[3:7], ROT_DEG * progress)
        node.send_output("pose_right", arrow.pose_array(np.append(right, 0.0)))
        node.send_output("pose_left", arrow.pose_array(np.append(home["left"], 0.0)))
        node.send_output("grip_right", pa.array([1.0 if t >= 1.0 else 0.0], type=pa.float32()))
        node.send_output("grip_left", pa.array([0.0], type=pa.float32()))


if __name__ == "__main__":
    main()
