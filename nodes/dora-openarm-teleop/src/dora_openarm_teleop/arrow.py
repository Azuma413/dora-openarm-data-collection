"""Arrow message helpers shared by the nodes (docs/teleop.md, Stages)."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import pyarrow as pa

POSE_TYPE = pa.struct({"pose": pa.list_(pa.float32())})
QPOS_TYPE = pa.struct({"qpos": pa.list_(pa.float32())})
STATE_TYPE = pa.struct(
    {
        "qpos": pa.list_(pa.float32()),
        "qvel": pa.list_(pa.float32()),
        "qtorque": pa.list_(pa.float32()),
    }
)


def pose_array(pose: np.ndarray) -> pa.Array:
    """POSE8: [{"pose": [x, y, z, qw, qx, qy, qz, gripper]}]."""
    return pa.array([{"pose": np.asarray(pose, dtype=np.float32)}], type=POSE_TYPE)


def qpos_array(qpos: np.ndarray) -> pa.Array:
    """QPOS8 / QPOS16: [{"qpos": [...]}]."""
    return pa.array([{"qpos": np.asarray(qpos, dtype=np.float32)}], type=QPOS_TYPE)


def state_array(qpos: np.ndarray, qvel: np.ndarray, qtorque: np.ndarray) -> pa.Array:
    """STATE8: [{"qpos": [...], "qvel": [...], "qtorque": [...]}]."""
    return pa.array(
        [
            {
                "qpos": np.asarray(qpos, dtype=np.float32),
                "qvel": np.asarray(qvel, dtype=np.float32),
                "qtorque": np.asarray(qtorque, dtype=np.float32),
            }
        ],
        type=STATE_TYPE,
    )


def string_array(text: str) -> pa.Array:
    """STRING: ["text"]."""
    return pa.array([text], type=pa.utf8())


def json_array(obj: Any) -> pa.Array:
    """STRING holding compact JSON."""
    return string_array(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))


def values(value: pa.Array, key: str) -> np.ndarray:
    """Read `key` from a length-1 struct array, or take a flat numeric array as-is.

    Same contract as the Enactic nodes' `extract_values`.
    """
    if pa.types.is_struct(value.type):
        value = value.field(key)[0].values
    return np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float32)


def scalar(value: pa.Array) -> float:
    """First element of a numeric or boolean array as a float."""
    return float(value[0].as_py())


def text(value: pa.Array) -> str:
    """First element of a string array."""
    return str(value[0].as_py())


def now_ns() -> int:
    """Timestamp convention for metadata["timestamp"] (UNIX ns)."""
    return time.time_ns()


def start_epoch(metadata: Mapping[str, Any] | None) -> int | None:
    """The follower's start_epoch from a state's metadata (dora-openarm), if it has one."""
    value = dict(metadata or {}).get("start_epoch")
    return value if isinstance(value, int) and not isinstance(value, bool) else None
