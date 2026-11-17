"""Quaternion and pose math in numpy (docs/teleop.md, Clutch).

Conventions, matching MuJoCo and the POSE8 message:

- quaternions are ``[w, x, y, z]`` (Hamilton product, active rotations);
- rotation matrices act on column vectors (``v_world = R @ v_local``);
- a pose7 is ``[x, y, z, qw, qx, qy, qz]``: the frame's origin and orientation
  expressed in a parent frame.

Every function broadcasts over leading dimensions (``(..., 4)``, ``(..., 3)``),
so the same code serves the per-tick clutch and vectorised tests.
"""

from __future__ import annotations

import numpy as np

# Below this norm a quaternion carries no usable rotation (float32 inputs that
# were zeroed or underflowed); callers must reject rather than invent one.
_MIN_NORM = 1e-9


def _as_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion must have 4 components, got shape {q.shape}")
    return q


def canonicalize(q: np.ndarray) -> np.ndarray:
    """Pick the representative with ``w >= 0`` (q and -q are the same rotation).

    Downstream consumers (IK, logs, tests) then see one stable sign.
    """
    q = _as_quat(q)
    return np.where(q[..., :1] < 0.0, -q, q)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Unit quaternion with ``w >= 0``. Raises ValueError on zero or non-finite input."""
    q = _as_quat(q)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(q)) or np.any(norm < _MIN_NORM):
        raise ValueError("cannot normalise a zero or non-finite quaternion")
    return canonicalize(q / norm)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate; the inverse rotation for unit quaternions."""
    q = _as_quat(q)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Multiplicative inverse, valid for non-unit quaternions too."""
    q = _as_quat(q)
    return quat_conjugate(q) / np.sum(q * q, axis=-1, keepdims=True)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ⊗ b``: rotate by ``b`` first, then by ``a``."""
    a = _as_quat(a)
    b = _as_quat(b)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` by unit quaternion ``q``."""
    q = _as_quat(q)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., :1]
    u = q[..., 1:]
    # v' = v + 2w(u×v) + 2u×(u×v): avoids building the matrix per call.
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix of a (normalised here) quaternion."""
    w, x, y, z = np.moveaxis(quat_normalize(q), -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """Quaternion (``w >= 0``) of a rotation matrix.

    Shepperd's method: derive the quaternion from whichever of w, x, y, z is
    largest, so no branch divides by a small number.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must be 3x3, got shape {m.shape}")
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    candidates = np.stack(
        [
            np.stack([1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01], -1),
            np.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], -1),
            np.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], -1),
            np.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], -1),
        ],
        axis=-2,
    )
    # Row k of `candidates` is 4*q_k*q; its k-th entry is 4*q_k^2.
    best = np.argmax(np.stack([candidates[..., k, k] for k in range(4)], -1), axis=-1)
    q = np.take_along_axis(candidates, best[..., None, None], axis=-2)[..., 0, :]
    return quat_normalize(q)


def quat_from_axis_angle(axis: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    """Rotation of ``angle`` rad about ``axis`` (need not be unit length)."""
    axis = np.asarray(axis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)[..., None]
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    if np.any(norm < _MIN_NORM):
        raise ValueError("rotation axis must be non-zero")
    half = 0.5 * angle
    return canonicalize(np.concatenate([np.cos(half), np.sin(half) * axis / norm], axis=-1))


def quat_angle(q: np.ndarray) -> np.ndarray:
    """Rotation angle in ``[0, pi]`` of a unit quaternion."""
    q = quat_normalize(q)
    return 2.0 * np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), q[..., 0])


def quat_delta(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """World-frame rotation ``d`` with ``d ⊗ q_from = q_to``.

    This is the clutch's hand rotation delta ``R_hand · R0⁻¹``: applied on the
    left of the held orientation, it turns the target about world axes the way
    the hand turned, independent of how the gripper is oriented.
    """
    return quat_normalize(quat_multiply(q_to, quat_conjugate(q_from)))


def quat_slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Shortest-arc spherical interpolation from ``a`` (t=0) to ``b`` (t=1)."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    # Nearly parallel: sin(theta) -> 0, fall back to a normalised lerp.
    near = sin_theta < 1e-6
    safe = np.where(near, 1.0, sin_theta)
    wa = np.where(near, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    wb = np.where(near, t, np.sin(t * theta) / safe)
    return quat_normalize(wa * a + wb * b)


def pose_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a ∘ b`` for pose7s: ``b`` expressed in ``a``'s frame, mapped to ``a``'s parent."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    position = a[..., :3] + quat_rotate(a[..., 3:7], b[..., :3])
    orientation = quat_normalize(quat_multiply(a[..., 3:7], b[..., 3:7]))
    return np.concatenate([position, orientation], axis=-1)


def pose_inverse(a: np.ndarray) -> np.ndarray:
    """Inverse of a pose7 (parent expressed in the pose's own frame)."""
    a = np.asarray(a, dtype=np.float64)
    inv_q = quat_conjugate(quat_normalize(a[..., 3:7]))
    return np.concatenate([-quat_rotate(inv_q, a[..., :3]), canonicalize(inv_q)], axis=-1)


def pose_relative(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``target`` (world pose7) expressed in the frame of ``origin`` (world pose7)."""
    return pose_compose(pose_inverse(origin), target)
