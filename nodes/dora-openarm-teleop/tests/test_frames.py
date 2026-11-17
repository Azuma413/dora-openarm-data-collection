"""Quaternion/pose helpers against scipy's Rotation on random samples."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from dora_openarm_teleop import frames

N = 500


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def _random_quats(rng: np.random.Generator, n: int = N) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _rot(q: np.ndarray) -> Rotation:
    return Rotation.from_quat(q, scalar_first=True)


def _same_rotation(a: np.ndarray, b: np.ndarray, atol: float = 1e-9) -> bool:
    """q and -q are the same rotation: compare |<a, b>| with 1."""
    return bool(np.all(np.abs(np.abs(np.sum(a * b, axis=-1)) - 1.0) < atol))


def test_normalize_is_unit_and_canonical(rng: np.random.Generator) -> None:
    raw = rng.normal(size=(N, 4)) * rng.uniform(0.1, 10.0, size=(N, 1))
    q = frames.quat_normalize(raw)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-12)
    assert np.all(q[:, 0] >= 0.0)
    assert _same_rotation(q, raw / np.linalg.norm(raw, axis=1, keepdims=True))


@pytest.mark.parametrize("bad", [[0, 0, 0, 0], [np.nan, 0, 0, 1], [np.inf, 0, 0, 0]])
def test_normalize_rejects_degenerate(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        frames.quat_normalize(np.array(bad, dtype=float))


def test_multiply_matches_scipy(rng: np.random.Generator) -> None:
    a = _random_quats(rng)
    b = _random_quats(rng)
    expected = (_rot(a) * _rot(b)).as_quat(scalar_first=True)
    assert _same_rotation(frames.quat_multiply(a, b), expected)


def test_conjugate_and_inverse_match_scipy(rng: np.random.Generator) -> None:
    q = _random_quats(rng)
    expected = _rot(q).inv().as_quat(scalar_first=True)
    assert _same_rotation(frames.quat_conjugate(q), expected)
    scaled = q * 3.0
    np.testing.assert_allclose(
        frames.quat_multiply(scaled, frames.quat_inverse(scaled)),
        np.tile([1.0, 0, 0, 0], (N, 1)),
        atol=1e-12,
    )


def test_rotate_matches_scipy(rng: np.random.Generator) -> None:
    q = _random_quats(rng)
    v = rng.normal(size=(N, 3))
    np.testing.assert_allclose(frames.quat_rotate(q, v), _rot(q).apply(v), atol=1e-12)


def test_matrix_round_trip_matches_scipy(rng: np.random.Generator) -> None:
    q = _random_quats(rng)
    matrices = frames.quat_to_matrix(q)
    np.testing.assert_allclose(matrices, _rot(q).as_matrix(), atol=1e-12)
    back = frames.matrix_to_quat(matrices)
    assert np.all(back[:, 0] >= 0.0)
    assert _same_rotation(back, q)


def test_matrix_to_quat_near_180_degrees() -> None:
    # Rotations by pi make w ~ 0: the branch Shepperd's method exists for.
    axes = np.eye(3)
    q = frames.quat_from_axis_angle(axes, np.full(3, np.pi))
    back = frames.matrix_to_quat(_rot(q).as_matrix())
    assert _same_rotation(back, q)


def test_axis_angle_and_angle_match_scipy(rng: np.random.Generator) -> None:
    axis = rng.normal(size=(N, 3))
    angle = rng.uniform(-np.pi, np.pi, size=N)
    q = frames.quat_from_axis_angle(axis, angle)
    rotvec = axis / np.linalg.norm(axis, axis=1, keepdims=True) * angle[:, None]
    assert _same_rotation(q, Rotation.from_rotvec(rotvec).as_quat(scalar_first=True))
    np.testing.assert_allclose(frames.quat_angle(q), np.abs(angle), atol=1e-9)


def test_delta_is_world_frame(rng: np.random.Generator) -> None:
    q_from = _random_quats(rng)
    q_to = _random_quats(rng)
    d = frames.quat_delta(q_from, q_to)
    expected = (_rot(q_to) * _rot(q_from).inv()).as_quat(scalar_first=True)
    assert _same_rotation(d, expected)
    assert _same_rotation(frames.quat_multiply(d, q_from), q_to)


def test_slerp_matches_scipy(rng: np.random.Generator) -> None:
    for _ in range(50):
        a, b = _random_quats(rng, 2)
        t = rng.uniform(0.0, 1.0)
        expected = Slerp([0.0, 1.0], _rot(np.stack([a, b])))([t]).as_quat(scalar_first=True)
        assert _same_rotation(frames.quat_slerp(a, b, t), expected[0], atol=1e-9)
    q = _random_quats(rng, 1)[0]
    assert _same_rotation(frames.quat_slerp(q, q, 0.3), q)


def test_pose_compose_inverse_relative(rng: np.random.Generator) -> None:
    a = np.concatenate([rng.normal(size=(N, 3)), _random_quats(rng)], axis=1)
    b = np.concatenate([rng.normal(size=(N, 3)), _random_quats(rng)], axis=1)
    ab = frames.pose_compose(a, b)
    np.testing.assert_allclose(ab[:, :3], a[:, :3] + _rot(a[:, 3:]).apply(b[:, :3]), atol=1e-12)
    assert _same_rotation(ab[:, 3:], (_rot(a[:, 3:]) * _rot(b[:, 3:])).as_quat(scalar_first=True))
    identity = frames.pose_compose(a, frames.pose_inverse(a))
    np.testing.assert_allclose(identity[:, :3], 0.0, atol=1e-12)
    np.testing.assert_allclose(identity[:, 3], 1.0, atol=1e-12)
    rel = frames.pose_relative(a, ab)
    np.testing.assert_allclose(rel[:, :3], b[:, :3], atol=1e-12)
    assert _same_rotation(rel[:, 3:], b[:, 3:])
