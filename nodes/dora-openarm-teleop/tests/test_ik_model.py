"""IK model with the gate's joint ranges (GATE-1 root fix, dora_openarm_teleop.ik_model)."""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
import pytest

from dora_openarm_teleop import frames, model_info
from dora_openarm_teleop.gate import (
    ARM_DOF,
    DEFAULT_LIMITS_PATH,
    ENGAGED,
    IK_LIMIT,
    SIDES,
    ArmGate,
    load_limits,
    profile_params,
)
from dora_openarm_teleop.ik_model import (
    arm_joint_name,
    ik_base_xml,
    narrowed_ranges,
    write_ik_config,
    write_ik_model,
)

MARGIN = profile_params("real").limit_margin
LIMITS = load_limits()


@pytest.fixture(scope="module")
def ik_xml(tmp_path_factory):
    return write_ik_model(DEFAULT_LIMITS_PATH, MARGIN, tmp_path_factory.mktemp("run") / "ik.xml")


def _gate_range(side: str, joint: int) -> tuple[float, float]:
    arm = ArmGate(side, LIMITS[side], profile_params("real"))
    return float(arm.lower[joint]), float(arm.upper[joint])


def test_model_compiles_from_anywhere_with_the_gate_ranges(ik_xml, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # asset paths must not depend on the working directory
    model = mujoco.MjModel.from_xml_path(str(ik_xml))
    base = mujoco.MjModel.from_xml_path(str(ik_base_xml()))
    narrowed = set()
    for side in SIDES:
        for j in range(ARM_DOF):
            name = arm_joint_name(side, j)
            joint = model.joint(name)
            assert model.jnt_limited[joint.id]
            # Exactly the range the gate clips commands to: an IK solution that respects
            # it is never clipped, so the gate never sees an "ik_limit".
            np.testing.assert_allclose(joint.range, _gate_range(side, j), atol=1e-12)
            lo, hi = base.joint(name).range
            assert lo <= joint.range[0] < joint.range[1] <= hi
            narrowed.add(model.joint(name).id)
    # Right J1 3.49 -> 1.95: the gap the review measured.
    assert model.joint("openarm_right_joint1").range[1] == pytest.approx(1.95)
    for jid in range(model.njnt):
        if jid not in narrowed:  # fingers, lifter: untouched
            np.testing.assert_array_equal(model.jnt_range[jid], base.jnt_range[jid])


def test_model_is_the_ik_default_scene_otherwise_unchanged(ik_xml):
    openarm_config = pytest.importorskip("openarm_control.config")
    # dora-openarm-ik loads openarm_control's default when no --xml is given; the copy
    # must start from that scene (with its "home" keyframe), not the arms-only file.
    assert str(ik_base_xml()) == str(openarm_config._DEFAULT_XML)
    model = mujoco.MjModel.from_xml_path(str(ik_xml))
    base = mujoco.MjModel.from_xml_path(str(ik_base_xml()))
    assert (model.nq, model.nv, model.nbody, model.nsite, model.nkey) == (
        base.nq,
        base.nv,
        base.nbody,
        base.nsite,
        base.nkey,
    )
    np.testing.assert_array_equal(model.key_qpos, base.key_qpos)
    for field in ("body_pos", "body_quat", "jnt_axis", "jnt_pos", "site_pos", "site_quat"):
        np.testing.assert_array_equal(getattr(model, field), getattr(base, field), err_msg=field)
    rng = np.random.default_rng(5)
    for _ in range(5):
        q = base.key_qpos[0] + rng.uniform(-0.4, 0.4, base.nq)
        poses = []
        for m in (model, base):
            data = mujoco.MjData(m)
            data.qpos[:] = q
            mujoco.mj_kinematics(m, data)
            poses.append(np.concatenate([data.site_xpos.ravel(), data.site_xmat.ravel()]))
        np.testing.assert_allclose(poses[0], poses[1], atol=1e-12)


def test_margin_is_validated(tmp_path):
    arms = ik_base_xml().parent.parent / "openarm_bimanual.xml"
    with pytest.raises(ValueError):
        narrowed_ranges(DEFAULT_LIMITS_PATH, -0.01, arms)
    with pytest.raises(ValueError):
        narrowed_ranges(DEFAULT_LIMITS_PATH, 1.0, arms)  # J6 is only +-0.785
    with pytest.raises(ValueError):
        write_ik_model(DEFAULT_LIMITS_PATH, float("nan"), tmp_path / "ik.xml")


def _ik_args(xml=None, config=None, tick_hz: float = 200.0) -> argparse.Namespace:
    """dora-openarm-ik's own argument parsing, with the dataflow's IK args."""
    control = pytest.importorskip("openarm_control")
    from dora_openarm_teleop.ik_node import IK_ARGS

    parser = argparse.ArgumentParser()
    control.register_common_args(parser)
    control.register_ik_args(parser)
    argv = [*IK_ARGS.split(), "--tick-hz", f"{tick_hz:g}"]
    if xml is not None:
        argv += ["--xml", str(xml)]
    if config is not None:
        argv += ["--config", str(config)]
    return parser.parse_args(argv)


def _ik(xml, config=None, tick_hz: float = 200.0):
    control = pytest.importorskip("openarm_control")
    args = _ik_args(xml, config, tick_hz)
    return control.Kinematics(control.setup_from_args(args), control.ik_params_from_args(args))


def _raise_right_hand(kin, ticks: int = 600):
    """Move the right target 60 cm up over 1.5 s to (0.10, -0.15, 0.40), above the shoulder."""
    home = model_info.home_ee_poses()
    goal = np.array([0.10, -0.15, 0.40])
    for k in range(ticks):
        pose = home["right"].copy()
        pose[:3] += min(k / 300, 1.0) * (goal - home["right"][:3])
        kin.set_target("right", pose.astype(np.float32))
        kin.set_gripper("right", 0.0)
        kin.set_target("left", home["left"].astype(np.float32))
        kin.set_gripper("left", 0.0)
        q = kin.solve()
        if q is not None:
            yield q


def test_real_ik_honours_the_narrowed_ranges(ik_xml):
    """One real openarm_control IK run through the written model (dataflow IK args)."""
    lower, upper = _gate_range("right", 0)
    # The stock model lets the IK take J1 well past the gate's range for this target.
    stock = max(float(q[0]) for q in _raise_right_hand(_ik(None)))
    assert stock > upper + 0.2

    gate = ArmGate("right", LIMITS["right"], profile_params("real"))
    ranges = np.array([_gate_range(side, j) for side in SIDES for j in range(ARM_DOF)])
    j1 = []
    t = 0.0
    for q in _raise_right_hand(_ik(ik_xml)):
        arms = np.concatenate([q[:ARM_DOF], q[8 : 8 + ARM_DOF]]).astype(np.float64)
        assert np.all(arms >= ranges[:, 0] - 1e-6) and np.all(arms <= ranges[:, 1] + 1e-6)
        j1.append(float(q[0]))
        t += 0.005
        gate.on_ik(q[:8], t)
        gate.on_state(gate.command if gate.command is not None else q[:8], t)
        gate.step(t)
        assert gate.reason != IK_LIMIT
    # The IK brakes into the limit instead of stopping short of it.
    assert max(j1) > upper - 0.1
    assert lower < max(j1) <= upper + 1e-6


# -- joint velocity caps (P3) ---------------------------------------------------------


def _gate_caps(profile: str, **overrides) -> dict[str, float]:
    """MJCF joint name -> the gate's own J1-J7 velocity cap (rad/s)."""
    caps = {}
    for side in SIDES:
        arm = ArmGate(side, LIMITS[side], profile_params(profile, **overrides))
        for j in range(ARM_DOF):
            caps[arm_joint_name(side, j)] = float(arm.velocity[j])
    return caps


@pytest.mark.parametrize(
    ("profile", "scale"), [("sim", None), ("real", None), ("real", 0.3), ("sim", 1.0)]
)
def test_ik_config_is_nine_tenths_of_the_gate_caps_as_openarm_control_reads_it(
    tmp_path, profile, scale
):
    control = pytest.importorskip("openarm_control")
    path = write_ik_config(DEFAULT_LIMITS_PATH, profile, scale, tmp_path / "run" / "ik_config.yaml")
    assert path.is_absolute() and path.is_file()
    # Parsed by openarm_control itself: --config only counts with --limit-velocity.
    params = control.ik_params_from_args(_ik_args(config=path))
    expected = _gate_caps(profile, velocity_scale=scale)
    assert params.velocity_limits is not None
    assert params.velocity_limits.keys() == expected.keys()
    for name, cap in expected.items():
        assert params.velocity_limits[name] == pytest.approx(0.9 * cap, rel=1e-12), name
    # The sim J1 cap the e2e run saw the stock IK exceed: 0.9 * 0.8 * 2.0.
    if (profile, scale) == ("sim", None):
        assert params.velocity_limits["openarm_right_joint1"] == pytest.approx(1.44)
    # Without --config the IK uses its built-in caps, faster than the written ones.
    stock = control.ik_params_from_args(_ik_args()).velocity_limits
    assert stock["openarm_right_joint1"] > params.velocity_limits["openarm_right_joint1"]
    # Without --limit-velocity openarm_control ignores the file altogether.
    unlimited = _ik_args(config=path)
    unlimited.limit_velocity = False
    assert control.ik_params_from_args(unlimited).velocity_limits is None


def test_ik_config_arguments_are_validated(tmp_path):
    with pytest.raises(ValueError, match="profile"):
        write_ik_config(DEFAULT_LIMITS_PATH, "moon", None, tmp_path / "c.yaml")
    for bad in (0.0, -0.5, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="velocity_scale"):
            write_ik_config(DEFAULT_LIMITS_PATH, "real", bad, tmp_path / "c.yaml")
    assert not (tmp_path / "c.yaml").exists()


def _far_targets():
    """Both hands thrown far from home at once: up above the shoulders, wrists twisted."""
    home = model_info.home_ee_poses()
    goals = {}
    for side, y in (("right", -0.15), ("left", 0.15)):
        pose = home[side].copy()
        pose[:3] = (0.10, y, 0.40)
        twist = np.array([np.cos(1.0), 0.0, 0.0, np.sin(1.0)])  # 2 rad about z
        pose[3:7] = frames.quat_multiply(twist, pose[3:7])
        goals[side] = pose.astype(np.float32)
    return goals


def _drive_through_gate(kin, profile: str, tick_hz: float, seconds: float = 1.0):
    """Solve once per tick towards far targets and stream every solution through the gate
    (real profile ramps, measured = command). Returns the peak joint speed of the IK per
    joint name, the largest |gate command - IK| while engaged, and the gate states seen."""
    dt = 1.0 / tick_hz
    goals = _far_targets()
    gates = {side: ArmGate(side, LIMITS[side], profile_params(profile)) for side in SIDES}
    peak = dict.fromkeys(_gate_caps(profile), 0.0)
    worst_lag = 0.0
    seen = set()
    prev = None
    t = 0.0
    for _ in range(round(seconds * tick_hz)):
        for side in SIDES:
            kin.set_target(side, goals[side])
            kin.set_gripper(side, 0.0)
        q = kin.solve()
        assert q is not None
        q = q.astype(np.float64)
        t += dt
        for i, side in enumerate(SIDES):
            arm_q = q[8 * i : 8 * i + 8]
            gate = gates[side]
            if gate.command is None:
                gate.on_state(arm_q, t)  # the arm starts where the IK starts
            gate.on_ik(arm_q, t)
            gate.on_state(gate.command, t)
            out = gate.step(t).astype(np.float64)
            seen.add(f"{gate.state}:{gate.reason}")
            if gate.state == ENGAGED:
                worst_lag = max(worst_lag, float(np.max(np.abs(out[:ARM_DOF] - arm_q[:ARM_DOF]))))
            if prev is not None:
                speed = np.abs(arm_q - prev[8 * i : 8 * i + 8]) / dt
                for j in range(ARM_DOF):
                    name = arm_joint_name(side, j)
                    peak[name] = max(peak[name], float(speed[j]))
        prev = q
    return peak, worst_lag, seen


@pytest.mark.parametrize("profile", ["sim", "real"])
def test_real_ik_never_outruns_the_gate_with_the_written_config(ik_xml, tmp_path, profile):
    """P3 with a real openarm_control solve at the dataflow's tick rate.

    A far target makes the IK move as fast as it may. With the stock caps it moves J1 at
    2.0 rad/s against the gate's 1.6 (sim) / 1.0 (real): the gate clips every tick, falls
    behind and holds on a jump. With write_ik_config every joint stays at or below 0.9 x
    the gate cap per solve, the caps are actually reached, and the gate streams the IK
    unchanged.
    """
    tick_hz = 250.0  # the leader tick of dataflow.yaml (dora/timer/millis/4)
    gate_caps = _gate_caps(profile)
    float32_slack = 1e-5 * tick_hz  # the IK publishes float32 joint angles

    stock_peak, stock_lag, stock_seen = _drive_through_gate(
        _ik(ik_xml, tick_hz=tick_hz), profile, tick_hz
    )
    assert stock_peak["openarm_right_joint1"] > gate_caps["openarm_right_joint1"] + 0.3
    assert stock_lag > 0.05 or "holding:jump" in stock_seen

    config = write_ik_config(DEFAULT_LIMITS_PATH, profile, None, tmp_path / "ik_config.yaml")
    peak, lag, seen = _drive_through_gate(_ik(ik_xml, config, tick_hz), profile, tick_hz)
    for name, cap in gate_caps.items():
        assert peak[name] <= 0.9 * cap + float32_slack, (name, peak[name], 0.9 * cap)
    # The limits bind (this is a fast reach, not a slow one that never tested them).
    for name in ("openarm_right_joint1", "openarm_left_joint1", "openarm_right_joint4"):
        assert peak[name] >= 0.9 * gate_caps[name] - float32_slack, (name, peak[name])
    assert seen == {f"{ENGAGED}:"}
    assert lag < 1e-5
