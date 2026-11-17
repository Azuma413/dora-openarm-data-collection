"""MJCF-derived facts (docs/teleop.md, Clutch) checked against the contract and MuJoCo."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from dora_openarm_teleop import frames, model_info
from dora_openarm_teleop.model_info import SIDES

# DESIGN §3.1: the only place these numbers may appear is a test.
DESIGN_HOME = {
    "right": [0.216, -0.1535, -0.22, 0.7071, 0.0, -0.7071, 0.0],
    "left": [0.216, 0.1535, -0.22, 0.7071, 0.0, -0.7071, 0.0],
}


def _same_rotation(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    return abs(abs(float(np.dot(a, b))) - 1.0) < atol


def _site_pose_in_origin(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    """Independent FK readout: origin⁻¹ ∘ site via MuJoCo's own quaternion helpers."""
    origin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_origin")
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee_control_point")
    poses = []
    for index in (origin, site):
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[index])
        poses.append(np.concatenate([data.site_xpos[index], quat]))
    return frames.pose_relative(poses[0], poses[1])


def test_home_poses_match_design() -> None:
    home = model_info.home_ee_poses()
    for side in SIDES:
        expected = np.array(DESIGN_HOME[side])
        np.testing.assert_allclose(home[side][:3], expected[:3], atol=1e-3)
        assert _same_rotation(home[side][3:], expected[3:], atol=1e-3)


def test_home_poses_are_copies() -> None:
    model_info.home_ee_poses()["right"][0] = 99.0
    assert model_info.home_ee_poses()["right"][0] != 99.0


def test_table_top_in_arm_origin() -> None:
    assert model_info.table_top_z_arm_origin() == pytest.approx(-0.335, abs=1e-6)
    # The task scenes attach the same cell, so the clamp holds in all of them.
    assert model_info.table_top_z_arm_origin("demo") == pytest.approx(-0.335, abs=1e-6)


def test_shoulders_lie_on_joint1_axis() -> None:
    model = model_info.load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    origin = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_origin")]
    shoulders = model_info.shoulder_positions()
    for side, sign in (("right", -1.0), ("left", 1.0)):
        np.testing.assert_allclose(shoulders[side], [0.0, sign * 0.1535, 0.0], atol=1e-6)
        for name in ("joint1", "joint2", "joint3"):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_{name}")
            offset = shoulders[side] - (data.xanchor[joint] - origin)
            distance = np.linalg.norm(np.cross(offset, data.xaxis[joint]))
            assert distance < 1e-6, f"{side} {name} axis misses the shoulder by {distance} m"


def test_gripper_ranges_per_side() -> None:
    ranges = model_info.gripper_ranges()
    assert ranges["right"] == pytest.approx((-0.7854, 0.0), abs=1e-6)
    assert ranges["left"] == pytest.approx((0.0, 0.7854), abs=1e-6)


def test_fingertip_offset_is_lowest_closed_hull_point() -> None:
    for side in SIDES:
        offset = model_info.fingertip_offset(side)
        points = model_info.fingertip_points(side, 0.0)
        assert offset[2] == pytest.approx(points[:, 2].min())
        # About 17 cm beyond the wrist, near the gripper's centre line.
        assert -0.18 < offset[2] < -0.16
        assert np.all(np.abs(offset[:2]) < 0.02)


@pytest.mark.parametrize("fraction", [0.0, 0.3, 1.0])
def test_fingertip_points_match_mujoco_fk(fraction: float) -> None:
    """The posable hulls reproduce MuJoCo's own FK for open fingers and a bent wrist."""
    model = model_info.load_model()
    rng = np.random.default_rng(7)
    ranges = model_info.gripper_ranges()
    for side in SIDES:
        low, high = ranges[side]
        gripper = (
            low + fraction * (high - low) if side == "left" else high + fraction * (low - high)
        )
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        for i in range(5, 8):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{i}")
            data.qpos[model.jnt_qposadr[joint]] = rng.uniform(-0.5, 0.5)
        for name in ("finger_joint1", "finger_joint2"):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_{name}")
            data.qpos[model.jnt_qposadr[joint]] = gripper
        mujoco.mj_kinematics(model, data)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee_control_point")
        site_rot = data.site_xmat[site].reshape(3, 3)
        expected = []
        for hull_body in ("inner", "outer"):
            body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_ee_{hull_body}_finger"
            )
            for geom in np.flatnonzero(model.geom_bodyid == body):
                if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                    continue
                mesh = model.geom_dataid[geom]
                start = model.mesh_vertadr[mesh]
                vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
                world = data.geom_xpos[geom] + vertices @ data.geom_xmat[geom].reshape(3, 3).T
                expected.append((world - data.site_xpos[site]) @ site_rot)
        expected = np.concatenate(expected)
        points = model_info.fingertip_points(side, gripper)
        # A convex hull has the same support function as the full vertex set:
        # compare the extreme extent along random directions.
        directions = rng.normal(size=(64, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        np.testing.assert_allclose(
            (points @ directions.T).min(axis=0),
            (expected @ directions.T).min(axis=0),
            atol=1e-9,
        )
        assert len(points) < len(expected)


def test_fk_at_home_equals_home_poses() -> None:
    home_qpos = np.zeros(16)
    home_qpos[3] = home_qpos[11] = np.pi / 2
    poses = model_info.ee_poses_from_qpos(home_qpos)
    home = model_info.home_ee_poses()
    for side in SIDES:
        np.testing.assert_allclose(poses[side][:3], home[side][:3], atol=1e-6)
        assert _same_rotation(poses[side][3:], home[side][3:], atol=1e-9)


def test_fk_matches_arms_only_model() -> None:
    """The cell scene FK agrees with the arms-only bimanual model the IK loads."""
    rng = np.random.default_rng(3)
    model = mujoco.MjModel.from_xml_path(str(model_info.bimanual_xml()))
    for _ in range(5):
        qpos16 = rng.uniform(-0.6, 0.6, size=16)
        qpos16[3] = rng.uniform(0.2, 2.0)
        qpos16[11] = rng.uniform(0.2, 2.0)
        poses = model_info.ee_poses_from_qpos(qpos16)
        data = mujoco.MjData(model)
        for offset, side in ((0, "right"), (8, "left")):
            for i in range(7):
                joint = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{i + 1}"
                )
                data.qpos[model.jnt_qposadr[joint]] = qpos16[offset + i]
        mujoco.mj_kinematics(model, data)
        for side in SIDES:
            expected = _site_pose_in_origin(model, data, side)
            np.testing.assert_allclose(poses[side][:3], expected[:3], atol=1e-9)
            assert _same_rotation(poses[side][3:], expected[3:], atol=1e-9)


@pytest.mark.parametrize("bad", [np.zeros(15), np.full(16, np.nan)])
def test_fk_rejects_malformed(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        model_info.ee_poses_from_qpos(bad)


# ---------------------------------------------------------------------------
# The scene registry: the scenes openarm_mujoco ships


@pytest.mark.parametrize("name", sorted(model_info.SCENES))
def test_every_registered_scene_resolves_to_an_installed_file(name: str) -> None:
    assert model_info.scene_xml(name).is_file()


def test_an_unknown_scene_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown scene"):
        model_info.scene_xml("compress")


def test_a_finger_geom_shape_the_hull_cannot_read_is_an_error_not_a_gap() -> None:
    """A silently dropped finger is a clamp that lets the real one through the table.

    The hull handles the two shapes the two shipped jaws are made of, meshes and boxes.
    A third shape on a finger has to stop the build, not quietly leave that geom out.
    """
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="b">'
        '<geom name="ball" type="sphere" size="0.01"/>'
        '<geom name="brick" type="box" size="0.01 0.02 0.03"/>'
        "</body></worldbody></mujoco>"
    )
    brick = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "brick")
    corners = model_info._geom_points(model, brick)
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(np.abs(corners).max(axis=0), [0.01, 0.02, 0.03])
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    with pytest.raises(ValueError, match="fingertip hull handles meshes and boxes"):
        model_info._geom_points(model, ball)
