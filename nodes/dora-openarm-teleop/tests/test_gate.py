"""Joint-space safety gate (docs/teleop.md, Gate): invariants, transitions, node wiring."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_teleop import arrow, gate_node
from dora_openarm_teleop.gate import (
    ARM_DOF,
    DEFAULT_LIMITS_PATH,
    DOF,
    ENGAGED,
    FAULT,
    HOLDING,
    HOMING,
    IK_LIMIT,
    JUMP,
    NAN_IK,
    NAN_STATE,
    NO_IK,
    NO_STATE,
    NOT_TRACKING,
    PROFILES,
    SHAPE_IK,
    SIDES,
    STALE_IK,
    STALE_STATE,
    WAITING,
    ArmGate,
    Gate,
    GateParams,
    load_limits,
    load_tracking_tol,
    profile_params,
)

LIMITS = load_limits()
HOME = LIMITS["right"].home.copy()
TICK = 0.005
EPS = 1e-6  # float32 output rounding


def make_arm(profile: str = "sim", side: str = "right", **overrides) -> ArmGate:
    return ArmGate(side, LIMITS[side], profile_params(profile, **overrides))


def feed(g: ArmGate, t: float, ik=None, state=None) -> np.ndarray | None:
    """Deliver fresh samples at time t, then tick."""
    if state is not None:
        g.on_state(np.asarray(state), t)
    if ik is not None:
        g.on_ik(np.asarray(ik), t)
    return g.step(t)


def engaged_arm(profile: str = "sim", pose=HOME, **overrides) -> tuple[ArmGate, float]:
    g = make_arm(profile, **overrides)
    feed(g, 0.0, ik=pose, state=pose)
    assert g.state == ENGAGED
    return g, 0.0


def offset(pose, joint: int, delta: float) -> np.ndarray:
    q = np.array(pose, dtype=np.float64)
    q[joint] += delta
    return q


# -- limits file ---------------------------------------------------------------


def test_limits_file_is_inside_the_mjcf_ranges():
    mujoco = pytest.importorskip("mujoco")
    from dora_openarm_teleop.model_info import bimanual_xml

    model = mujoco.MjModel.from_xml_path(str(bimanual_xml()))
    for side in SIDES:
        limits = LIMITS[side]
        for j in range(ARM_DOF):
            lo, hi = model.joint(f"openarm_{side}_joint{j + 1}").range
            assert lo <= limits.lower[j] < limits.upper[j] <= hi, (side, j)
        lo, hi = model.actuator(f"{side}_finger1_ctrl").ctrlrange
        # The gripper range is the MJCF range itself (DESIGN §3.1).
        assert (limits.lower[7], limits.upper[7]) == pytest.approx((lo, hi), abs=1e-6)


def test_limits_file_is_inside_the_driver_config():
    openarm_driver = pytest.importorskip("openarm_driver")
    config = openarm_driver.Config("openarm_pedestal")
    velocity = np.asarray(config.get_joint_velocity_limits(), dtype=float)
    for side in SIDES:
        driver = np.asarray(config.get_joint_limits(f"{side}_arm"), dtype=float)
        assert np.all(LIMITS[side].lower >= driver[:, 0])
        assert np.all(LIMITS[side].upper <= driver[:, 1])
    np.testing.assert_allclose(LIMITS["right"].velocity, velocity)


def test_home_is_inside_the_margins_and_caps_match_the_driver_yaml():
    for side in SIDES:
        g = make_arm(side=side)
        assert np.all(g.lower <= LIMITS[side].home) and np.all(LIMITS[side].home <= g.upper)
    # openarm_driver openarm_pedestal.yaml:56-63, duplicated in the limits file.
    np.testing.assert_allclose(LIMITS["left"].velocity, [2.0, 2.0, 3.3, 3.3, 6.3, 6.3, 6.3, 20.0])


def test_params_are_validated():
    with pytest.raises(ValueError):
        GateParams(engage_tol=0.3, jump_tol=0.2)
    with pytest.raises(ValueError):
        GateParams(velocity_scale=1.5)
    with pytest.raises(ValueError):
        GateParams(velocity_scale=0.5, homing_scale=0.6)
    with pytest.raises(ValueError):
        profile_params("moon")


def test_profiles_scale_the_caps():
    sim, real = make_arm("sim"), make_arm("real")
    np.testing.assert_allclose(sim.velocity, 0.8 * LIMITS["right"].velocity)
    np.testing.assert_allclose(real.velocity, 0.5 * LIMITS["right"].velocity)
    assert sim.params.reset_mode == "snap" and real.params.reset_mode == "ramp"


# -- randomized invariants -------------------------------------------------------


class Scenario:
    """Random IK/feedback streams against one Gate, checking the invariants every tick."""

    def __init__(self, seed: int, profile: str) -> None:
        self.rng = np.random.default_rng(seed)
        self.params = profile_params(profile)
        self.gate = Gate(LIMITS, self.params)
        self.t = 0.0
        self.ik = {s: LIMITS[s].home.copy() for s in SIDES}
        self.ik_pause_until = dict.fromkeys(SIDES, 0.0)
        self.state_pause_until = dict.fromkeys(SIDES, 0.0)
        self.stuck_until = dict.fromkeys(SIDES, 0.0)
        self.stuck_at: dict[str, np.ndarray | None] = dict.fromkeys(SIDES)
        self.next_state = dict.fromkeys(SIDES, 0.0)
        self.prev: dict[str, np.ndarray | None] = dict.fromkeys(SIDES)
        self.prev_t = 0.0
        self.snapped = dict.fromkeys(SIDES, False)
        self.nan_latched = dict.fromkeys(SIDES, False)
        self.last_ik_t: dict[str, float | None] = dict.fromkeys(SIDES)
        self.last_state_t: dict[str, float | None] = dict.fromkeys(SIDES)
        self.seen: set[str] = set()

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self._advance_clock()
            for side in SIDES:
                self._drive(side)
            self._maybe_reset()
            out = self.gate.step(self.t)
            for side in SIDES:
                self._check(side, out[side])
            self.prev_t = self.t

    def _advance_clock(self) -> None:
        r = self.rng.random()
        if r < 0.01:
            self.t += self.rng.uniform(0.05, 0.4)  # process stall
        elif r < 0.015:
            self.t -= 0.001  # clock going backwards must not move the arm
        else:
            self.t += self.rng.uniform(0.001, 0.012)

    def _drive(self, side: str) -> None:
        rng, arm = self.rng, self.gate.arms[side]
        limits = LIMITS[side]
        if self.t >= self.state_pause_until[side] and self.t >= self.next_state[side]:
            if self.t >= self.stuck_until[side]:
                self.stuck_at[side] = None
            base = self.stuck_at[side]
            if base is None:
                base = arm.command if arm.command is not None else limits.home
            sag = rng.normal(0.0, 0.02, DOF)
            arm.on_state(base + sag, self.t)
            self.last_state_t[side] = self.t
            self.next_state[side] = self.t + 0.01
            if rng.random() < 0.004:
                self.state_pause_until[side] = self.t + rng.uniform(0.02, 0.3)
            elif rng.random() < 0.006:
                # The arm stops following: feedback keeps arriving, from the same pose.
                self.stuck_at[side] = np.asarray(base, dtype=float).copy()
                self.stuck_until[side] = self.t + rng.uniform(0.1, 0.6)
        if self.t < self.ik_pause_until[side]:
            return
        r = rng.random()
        ik = self.ik[side]
        if r < 0.01:  # IK teleport, sometimes well outside the limits
            ik[:] = rng.uniform(limits.lower - 0.5, limits.upper + 0.5)
        elif r < 0.03:  # come back close to the command so re-engaging is exercised
            if arm.command is not None:
                ik[:] = arm.command + rng.uniform(-0.08, 0.08, DOF)
        elif r < 0.035:
            self.ik_pause_until[side] = self.t + rng.uniform(0.02, 0.3)
            return
        elif r < 0.04:  # oscillation around the current target
            ik[:] = ik + 0.25 * np.sin(2 * np.pi * rng.uniform(1, 20) * self.t)
        else:  # fast random walk, often faster than the caps
            ik[:] = ik + rng.normal(0.0, 0.01, DOF)
        sample = ik.copy()
        r = rng.random()
        if r < 0.002:
            sample[rng.integers(DOF)] = rng.choice([np.nan, np.inf, -np.inf])
            self.nan_latched[side] = True
        elif r < 0.003:
            sample = sample[:5]
        arm.on_ik(sample, self.t)
        if sample.shape == (DOF,) and np.all(np.isfinite(sample)):
            self.last_ik_t[side] = self.t
        else:
            self.nan_latched[side] = True

    def _maybe_reset(self) -> None:
        r = self.rng.random()
        if r < 0.004:
            self.gate.on_control_reset(self.t)
            self.nan_latched = dict.fromkeys(SIDES, False)
            self._mark_snap()
        elif r < 0.006:
            pose = np.concatenate(
                [self.rng.uniform(LIMITS[s].lower, LIMITS[s].upper) for s in SIDES]
            )
            self.gate.on_reset_position(pose, self.t)
            self._mark_snap()

    def _mark_snap(self) -> None:
        if self.params.reset_mode == "snap":
            self.snapped = dict.fromkeys(SIDES, True)

    def _check(self, side: str, out: np.ndarray | None) -> None:
        arm = self.gate.arms[side]
        self.seen.add(f"{arm.state}:{arm.reason}")
        if out is None:
            assert self.last_state_t[side] is None
            return
        assert out.dtype == np.float32 and out.shape == (DOF,)
        assert np.all(np.isfinite(out))
        assert np.all(out >= arm.lower - EPS) and np.all(out <= arm.upper + EPS)
        prev = self.prev[side]
        if prev is not None and not self.snapped[side]:
            dt = np.clip(self.t - self.prev_t, 0.0, self.params.max_dt)
            step = np.abs(out.astype(np.float64) - prev)
            assert np.all(step <= arm.velocity * dt + EPS), (step, arm.velocity * dt)
            if arm.state != ENGAGED and arm.reason != HOMING:
                np.testing.assert_array_equal(out, prev)
            if arm.state != ENGAGED and not arm.homing and np.any(step > 0.0):
                # The approach ramp moved an unengaged command: it may not walk the arm
                # away from where the arm actually is (a stuck arm stops it).
                follow = np.max(np.abs(arm.command[:ARM_DOF] - arm.measured.value[:ARM_DOF]))
                bound = self.params.tracking_tol + np.max(arm.home_velocity) * self.params.max_dt
                assert follow <= bound, (follow, bound)
        if arm.state == ENGAGED:
            assert self.t - self.last_ik_t[side] <= self.params.stale_s + 1e-12
            assert self.t - self.last_state_t[side] <= self.params.stale_s + 1e-12
            # Never engaged on an IK solved far past the limits (GATE-1).
            raw = arm.ik.value[:ARM_DOF]
            beyond = np.abs(raw - np.clip(raw, arm.lower[:ARM_DOF], arm.upper[:ARM_DOF]))
            assert np.all(beyond <= self.params.jump_tol + 1e-12), beyond
        if self.nan_latched[side]:
            assert arm.state == FAULT
        self.prev[side] = out.astype(np.float64)
        self.snapped[side] = False


@pytest.mark.parametrize("profile", ["sim", "real"])
def test_randomized_streams_keep_the_invariants(profile):
    seen: set[str] = set()
    for seed in range(40):
        scenario = Scenario(seed, profile)
        scenario.run(500)
        seen |= scenario.seen
    # The random streams must actually reach the interesting states.
    for label in (
        f"{ENGAGED}:",
        f"{HOLDING}:{JUMP}",
        f"{HOLDING}:{IK_LIMIT}",
        f"{HOLDING}:{STALE_IK}",
        f"{HOLDING}:{NOT_TRACKING}",
        f"{HOLDING}:{HOMING}",  # the approach ramp; the reset ramp is real-only
    ):
        assert label in seen, sorted(seen)
    assert f"{HOLDING}:{STALE_STATE}" in seen
    assert any(s.startswith(f"{FAULT}:") for s in seen)
    if profile == "real":
        assert f"{HOLDING}:{HOMING}" in seen


# -- transitions -------------------------------------------------------------------


def test_nothing_is_sent_before_feedback():
    g = make_arm()
    g.on_ik(HOME, 0.0)
    assert g.step(0.0) is None
    assert (g.state, g.reason) == (HOLDING, NO_STATE)
    out = feed(g, TICK, state=HOME)
    np.testing.assert_allclose(out, HOME, atol=EPS)


def test_first_command_is_the_measurement_clipped_into_the_margins():
    g = make_arm()
    measured = offset(HOME, 3, -HOME[3])  # joint4 at its hard limit 0.0
    out = feed(g, 0.0, state=measured)
    assert out[3] == pytest.approx(0.05, abs=EPS)
    assert (g.state, g.reason) == (HOLDING, NO_IK)


def test_engages_only_when_ik_is_close_to_the_command():
    g = make_arm()
    feed(g, 0.0, ik=offset(HOME, 1, 0.15), state=HOME)
    assert (g.state, g.reason) == (HOLDING, WAITING)  # dt 0: the ramp cannot move yet
    out = feed(g, TICK, ik=offset(HOME, 1, 0.15), state=HOME)
    assert (g.state, g.reason) == (HOLDING, HOMING)  # 0.15 rad away: ramped, not refused
    assert out[1] > HOME[1]
    feed(g, 2 * TICK, ik=offset(HOME, 1, 0.09), state=HOME)
    assert g.state == ENGAGED


def test_first_engage_checks_that_the_arm_is_following_the_command():
    g = make_arm()
    feed(g, 0.0, state=HOME)
    # The arm was pushed away after the command was seeded; IK still equals the command.
    feed(g, TICK, ik=HOME, state=offset(HOME, 2, 0.3))
    assert (g.state, g.reason) == (HOLDING, NOT_TRACKING)
    # Still refused while it is only most of the way back (tracking_tol 0.25 rad).
    feed(g, 2 * TICK, ik=HOME, state=offset(HOME, 2, 0.26))
    assert (g.state, g.reason) == (HOLDING, NOT_TRACKING)
    # 0.16 rad is sag, not a stuck arm: the old rule (raw IK within engage_tol of the
    # measurement) could never engage on it.
    feed(g, 3 * TICK, ik=HOME, state=offset(HOME, 2, 0.16))
    assert g.state == ENGAGED


def test_tracking_is_rate_limited_to_the_scaled_cap():
    g, t = engaged_arm()
    target = offset(HOME, 0, 0.15)
    v = 0.8 * 2.0
    prev = HOME.copy()
    for _ in range(10):
        t += TICK
        out = feed(g, t, ik=target, state=HOME).astype(np.float64)
        assert out[0] - prev[0] == pytest.approx(v * TICK, abs=EPS)
        prev = out
    for _ in range(20):
        t += TICK
        out = feed(g, t, ik=target, state=HOME)
    assert out[0] == pytest.approx(target[0], abs=EPS)
    assert g.state == ENGAGED


def test_jump_holds_then_reengages_near_the_command_despite_sag():
    g, t = engaged_arm()
    command = g.command.copy()
    t += TICK
    out = feed(g, t, ik=offset(HOME, 2, 0.5), state=HOME)
    assert (g.state, g.reason) == (HOLDING, JUMP)
    np.testing.assert_array_equal(out, command.astype(np.float32))
    sagged = offset(HOME, 1, -0.15)  # measured pose sags away from the command
    t += TICK
    out = feed(g, t, ik=offset(command, 2, 0.15), state=sagged)
    # Back inside jump_tol: the gate closes the last 0.15 rad itself instead of waiting
    # for the operator to bring the IK within engage_tol of the command.
    assert (g.state, g.reason) == (HOLDING, HOMING)
    assert not g.homing  # not a reset ramp: an episode start must not wait on this
    assert out[2] > command[2]
    t += TICK
    feed(g, t, ik=offset(command, 2, 0.05), state=sagged)
    assert g.state == ENGAGED


def test_ik_outrunning_the_cap_disengages_instead_of_lagging_without_bound():
    """The openarm_driver crash scenario: IK at 2x the cap on J1 (hazard_sim [A])."""
    g, t = engaged_arm("real")
    ik = HOME.copy()
    max_gap = 0.0
    for _ in range(400):
        t += TICK
        ik[0] += 4.0 * TICK
        out = feed(g, t, ik=ik, state=g.command)
        if g.state != ENGAGED:
            break
        max_gap = max(max_gap, ik[0] - float(out[0]))
    assert (g.state, g.reason) == (HOLDING, JUMP)
    # Gap never grows past jump_tol plus one IK step: far below the driver's 1.0 rad.
    assert max_gap <= 0.2 + 4.0 * TICK
    held = g.command.copy()
    for _ in range(50):
        t += TICK
        ik[0] += 4.0 * TICK
        feed(g, t, ik=ik, state=held)
    np.testing.assert_array_equal(g.command, held)


def test_stale_ik_holds_and_recovers_on_the_command():
    g, t = engaged_arm()
    last_ik = t
    while t - last_ik <= 0.05:
        t += TICK
        feed(g, t, state=HOME)
    assert (g.state, g.reason) == (HOLDING, STALE_IK)
    assert t - last_ik <= 0.05 + TICK + 1e-9  # detected within one tick of stale_s
    t += TICK
    feed(g, t, ik=HOME, state=offset(HOME, 1, -0.15))  # sag does not block this
    assert g.state == ENGAGED


def test_stale_feedback_requires_the_arm_to_be_following_again():
    g, t = engaged_arm()
    for _ in range(12):
        t += TICK
        feed(g, t, ik=HOME)
    assert (g.state, g.reason) == (HOLDING, STALE_STATE)
    t += TICK
    feed(g, t, ik=HOME, state=offset(HOME, 0, 0.3))  # arm is not where we left it
    assert (g.state, g.reason) == (HOLDING, NOT_TRACKING)
    t += TICK
    feed(g, t, ik=HOME, state=offset(HOME, 0, 0.05))
    assert g.state == ENGAGED


# -- a sagging arm (GATE-2) ---------------------------------------------------------

# The worst steady-state joint error measured on the real model (pedestal gains, no
# gravity feed-forward): 158 mrad reaching low, 154 mrad with the arm extended.
SAG = np.array([0.07, 0.05, 0.02, 0.16, 0.0, 0.05, 0.0, 0.0])


def test_a_sag_settled_start_ramps_up_to_the_ik_and_engages():
    """The arm has settled a sag below its command and the IK solves the nominal pose.

    Comparing the IK with the measurement (engage_tol 0.1 rad) can never be satisfied at
    0.16 rad of droop, and neither can comparing it with the command, which is seeded
    from that same sagged measurement. The gate ramps the command over instead.
    """
    g = make_arm("sim")
    settled = HOME - SAG
    t = 0.0
    feed(g, t, state=settled, ik=HOME)  # command seeded a sag below the IK
    assert (g.state, g.reason) == (HOLDING, WAITING)  # dt 0: nothing to ramp with yet
    prev = g.command.copy()
    reasons = set()
    while True:
        t += TICK
        # The arm keeps its droop as the command rises (PD, no gravity feed-forward).
        out = feed(g, t, ik=HOME, state=g.command - SAG).astype(np.float64)
        if g.state == ENGAGED:
            break
        assert np.all(np.abs(out - prev) <= g.home_velocity * TICK + EPS)
        assert not g.homing  # an episode start must not wait on the approach ramp
        reasons.add(g.reason)
        prev = out
        assert t < 1.0, "never engaged on a sagged arm"
    assert reasons == {HOMING}
    assert t == pytest.approx(0.085, abs=2 * TICK)  # (0.16 - 0.1) / (0.25 x 3.3 rad/s)
    for _ in range(20):  # and it stays engaged while the sag stays
        t += TICK
        feed(g, t, ik=HOME, state=g.command - SAG)
        assert g.state == ENGAGED


def test_a_stuck_arm_holds_with_not_tracking_until_it_follows_again():
    """An arm that stops moving (brake, e-stop, obstacle, a hand holding it).

    The gate keeps streaming while it is engaged - the command leads the measurement by
    the servo lag on every fast move, so a tracking test there would disengage mid-reach -
    but it refuses to engage again on an arm that is 0.3 rad from its command.
    """
    g, t = engaged_arm("sim")
    stuck = g.command.copy()
    ik = HOME.copy()
    for _ in range(60):  # the operator keeps moving; the arm does not
        t += TICK
        ik[0] += 1.0 * TICK
        feed(g, t, ik=ik, state=stuck)
    assert g.state == ENGAGED
    assert float(g.command[0] - stuck[0]) == pytest.approx(0.3, abs=0.02)
    for _ in range(12):  # an IK dropout disengages it
        t += TICK
        feed(g, t, state=stuck)
    assert (g.state, g.reason) == (HOLDING, STALE_IK)
    held = g.command.copy()
    for _ in range(5):
        t += TICK
        # The IK comes back exactly on the command: the old rule re-engaged here, on an
        # arm 0.3 rad away, and handed the driver that error as a step.
        feed(g, t, ik=held, state=stuck)
        assert (g.state, g.reason) == (HOLDING, NOT_TRACKING)
        np.testing.assert_array_equal(g.command, held)  # and the ramp stays put
    t += TICK  # the brake releases, the arm catches up to a sag away from the command
    feed(g, t, ik=held, state=held - SAG)
    assert g.state == ENGAGED


def test_the_approach_ramp_stops_when_the_arm_stops_following():
    """The ramp cannot run away: an IK that stays 0.15 rad ahead of the command forever
    pulls it at most tracking_tol away from an arm that never moves."""
    g = make_arm("sim")
    frozen = HOME.copy()
    feed(g, 0.0, state=frozen)
    t, worst, reasons = 0.0, 0.0, []
    for _ in range(400):
        t += TICK
        feed(g, t, ik=offset(g.command, 0, 0.15), state=frozen)
        assert g.state != ENGAGED
        worst = max(worst, float(np.max(np.abs(g.command[:ARM_DOF] - frozen[:ARM_DOF]))))
        reasons.append(g.reason)
    step = float(np.max(g.home_velocity)) * TICK
    assert HOMING in reasons  # it did ramp
    assert reasons[-1] == NOT_TRACKING and reasons[-100:] == [NOT_TRACKING] * 100
    assert g.params.tracking_tol - step <= worst <= g.params.tracking_tol + step


def test_the_approach_ramp_never_follows_an_ik_past_jump_tol():
    """Far and moving: the gate holds the command, it does not walk after the IK."""
    g = make_arm("sim")
    feed(g, 0.0, state=HOME)
    seed = g.command.copy()
    ik = offset(HOME, 0, 0.5)
    t = 0.0
    for _ in range(200):
        t += TICK
        ik[0] += 0.3 * TICK
        feed(g, t, ik=ik, state=HOME)
        assert (g.state, g.reason) == (HOLDING, WAITING)
    np.testing.assert_array_equal(g.command, seed)


@pytest.mark.parametrize(("profile", "scale"), [("sim", 0.8), ("real", 0.5)])
def test_the_approach_ramp_moves_at_the_homing_rate(profile, scale):
    """Unsteered motion runs at homing_scale, below the profile's engaged caps."""
    g = make_arm(profile)
    feed(g, 0.0, state=HOME)
    t, prev, steps = 0.0, g.command.copy(), []
    while g.state != ENGAGED:
        t += TICK
        out = feed(g, t, ik=offset(HOME, 0, 0.15), state=HOME).astype(np.float64)
        if g.reason == HOMING:
            steps.append(float(np.max(np.abs(out - prev))))
        prev = out
        assert t < 1.0
    cap = float(LIMITS["right"].velocity[0])
    assert max(steps) == pytest.approx(0.25 * cap * TICK, abs=EPS)
    assert max(steps) < scale * cap * TICK  # slower than the gate follows the operator


def test_the_real_profile_tolerates_more_droop_than_the_sim_one():
    # 0.27 rad: past the sim tolerance (0.25), inside the real one (0.30).
    for profile, engages in (("sim", False), ("real", True)):
        g = make_arm(profile)
        feed(g, 0.0, state=HOME)
        feed(g, TICK, ik=HOME, state=offset(HOME, 3, -0.27))
        assert (g.state == ENGAGED) is engages
        assert g.params.tracking_tol == (0.25 if profile == "sim" else 0.30)


def test_tracking_tol_comes_from_the_limits_file(tmp_path):
    tolerances = load_tracking_tol()
    assert tolerances == {"sim": 0.25, "real": 0.30}
    assert {name: p.tracking_tol for name, p in PROFILES.items()} == tolerances
    # Past the worst sag measured on the real model, and past jump_tol so the approach
    # ramp can always close its whole window before the tracking test stops it.
    assert all(tol > 0.158 and tol > PROFILES["sim"].jump_tol for tol in tolerances.values())
    with pytest.raises(ValueError):
        GateParams(jump_tol=0.2, tracking_tol=0.2)
    text = DEFAULT_LIMITS_PATH.read_text(encoding="utf-8")
    for broken in ("gate_tracking_tol:", "  real: 0.30"):
        path = tmp_path / f"{broken.strip(': ')}.yaml"
        path.write_text(text.replace(broken, "# removed"), encoding="utf-8")
        with pytest.raises(ValueError, match="gate_tracking_tol"):
            load_tracking_tol(path)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_ik_faults_until_control_reset(bad):
    g, t = engaged_arm()
    t += TICK
    feed(g, t, ik=offset(HOME, 0, 0.05), state=HOME)
    held = g.command.copy()
    g.on_ik(offset(HOME, 4, bad), t)
    assert (g.state, g.reason) == (FAULT, NAN_IK)
    for _ in range(20):
        t += TICK
        out = feed(g, t, ik=HOME, state=HOME)
        assert np.all(np.isfinite(out))
        np.testing.assert_array_equal(out, held.astype(np.float32))
        assert g.state == FAULT
    g.on_control_reset(t)
    t += TICK
    feed(g, t, ik=HOME, state=HOME)
    assert g.state == ENGAGED


def test_non_finite_state_and_bad_shapes_fault():
    g, t = engaged_arm()
    g.on_state(offset(HOME, 3, np.nan), t)
    assert (g.state, g.reason) == (FAULT, NAN_STATE)
    g2, t2 = engaged_arm()
    g2.on_ik(HOME[:7], t2)
    assert (g2.state, g2.reason) == (FAULT, SHAPE_IK)
    assert g2.step(t2 + TICK) is not None and g2.state == FAULT


def test_commands_are_clamped_inside_the_limits_with_margin():
    g, t = engaged_arm()
    upper = LIMITS["right"].upper
    lower = LIMITS["right"].lower
    ik = HOME.copy()
    for _ in range(400):
        t += TICK
        # joint4 towards and past its upper limit, joint6 past its lower limit, both
        # stopping 0.15 rad beyond the margin (inside jump_tol, so the gate clamps).
        ik[3] = min(ik[3] + 0.004, upper[3] - 0.05 + 0.15)
        ik[5] = max(ik[5] - 0.004, lower[5] + 0.05 - 0.15)
        ik[7] -= 0.004  # gripper opening past the MJCF range
        out = feed(g, t, ik=ik, state=g.command)
    assert g.state == ENGAGED
    assert out[3] == pytest.approx(upper[3] - 0.05, abs=EPS)
    assert out[5] == pytest.approx(lower[5] + 0.05, abs=EPS)
    assert out[7] == pytest.approx(lower[7], abs=EPS)  # gripper has no margin
    # Closing reaches the closed stop exactly (0 for the right gripper).
    ik[7] = 0.2
    for _ in range(40):
        t += TICK
        out = feed(g, t, ik=ik, state=g.command)
    assert out[7] == pytest.approx(0.0, abs=EPS)


def test_ik_ramping_far_past_a_limit_holds_with_ik_limit():
    """GATE-1: the IK plans J1 past the gate's range (MJCF allows 3.49, the gate 2.0).

    Clamping J1 and streaming the IK's other joints put the real EE 271 mm from the
    target; the gate must hold instead of staying engaged on a clipped value.
    """
    g, t = engaged_arm("real", pose=offset(HOME, 0, 1.9))
    upper = g.upper[0]
    ik = g.command.copy()
    worst_engaged_gap = 0.0
    for _ in range(400):
        t += TICK
        ik[0] = min(3.1, ik[0] + 1.0 * TICK)  # 1 rad/s ramp past the limit
        out = feed(g, t, ik=ik, state=g.command)
        if g.state == ENGAGED:
            worst_engaged_gap = max(worst_engaged_gap, ik[0] - float(out[0]))
    assert (g.state, g.reason) == (HOLDING, IK_LIMIT)
    assert float(g.command[0]) == pytest.approx(upper, abs=EPS)
    # Never engaged with the raw IK more than jump_tol (plus one IK step) from the command.
    assert worst_engaged_gap <= 0.2 + 1.0 * TICK + EPS
    # Back inside the range but still away from the command: an ordinary jump hold.
    t += TICK
    feed(g, t, ik=offset(g.command, 0, -0.3), state=g.command)
    assert (g.state, g.reason) == (HOLDING, JUMP)
    # Re-engaging uses the jump rule: raw IK within engage_tol of the command.
    t += TICK
    feed(g, t, ik=offset(g.command, 0, -0.05), state=offset(g.command, 1, -0.15))
    assert g.state == ENGAGED


def test_jump_is_measured_on_the_raw_ik_not_the_clipped_one():
    upper = make_arm().upper[0]
    g, t = engaged_arm(pose=offset(HOME, 0, upper - 0.15))
    # Clipped, this IK is 0.15 rad from the command (< jump_tol); raw it is 0.3 rad away.
    t += TICK
    out = feed(g, t, ik=offset(HOME, 0, upper + 0.15), state=g.command)
    assert (g.state, g.reason) == (HOLDING, JUMP)
    np.testing.assert_allclose(out[0], upper - 0.15, atol=EPS)


def test_engaging_compares_the_raw_ik_with_the_command():
    upper = make_arm().upper[0]
    at_limit = offset(HOME, 0, upper)
    g = make_arm()
    feed(g, 0.0, state=at_limit)
    # The clipped IK equals the command exactly, but the raw IK is 0.15 rad past it.
    feed(g, TICK, ik=offset(HOME, 0, upper + 0.15), state=at_limit)
    assert (g.state, g.reason) == (HOLDING, WAITING)
    feed(g, 2 * TICK, ik=offset(HOME, 0, upper + 0.05), state=at_limit)
    assert g.state == ENGAGED


def test_gripper_jumps_are_rate_limited_but_do_not_disengage():
    g, t = engaged_arm()
    ik = offset(HOME, 7, -0.78)
    t += TICK
    out = feed(g, t, ik=ik, state=HOME)
    assert g.state == ENGAGED
    assert out[7] == pytest.approx(-0.8 * 20.0 * TICK, abs=EPS)


def test_dt_is_clamped_after_a_long_gap():
    g, t = engaged_arm()
    target = offset(HOME, 0, 0.15)
    t += 1.0  # one-second stall of the gate itself; fresh samples arrive with the tick
    out = feed(g, t, ik=target, state=HOME)
    assert g.state == ENGAGED
    assert out[0] - HOME[0] == pytest.approx(0.8 * 2.0 * 0.05, abs=EPS)


def test_snap_reset_position_sets_the_command_and_restarts_engagement():
    g, t = engaged_arm()
    pose = offset(HOME, 0, 0.1)
    for _ in range(30):
        t += TICK
        feed(g, t, ik=pose, state=g.command)
    reset = offset(HOME, 6, 0.3)
    g.on_reset_position(reset, t)
    t += TICK
    out = feed(g, t, ik=pose, state=pose)  # stale IK solution from before the reset
    np.testing.assert_allclose(out, reset, atol=EPS)
    # The command teleported with the plant; until an observation from after the reset
    # arrives the arm is 0.3 rad from it, which is not "following" (tracking_tol 0.25).
    assert (g.state, g.reason) == (HOLDING, NOT_TRACKING)
    t += TICK
    feed(g, t, ik=reset, state=pose)  # sim observation not updated yet
    assert g.state == HOLDING
    t += TICK
    feed(g, t, ik=reset, state=reset)
    assert g.state == ENGAGED


def test_snap_control_reset_uses_the_last_reset_position():
    gate = Gate(LIMITS, profile_params("sim"))
    custom = {s: offset(LIMITS[s].home, 0, 0.2 if s == "right" else -0.2) for s in SIDES}
    gate.on_reset_position(np.concatenate([custom[s] for s in SIDES]), 0.0)
    for side in SIDES:
        gate.on_state(side, LIMITS[side].home, 0.0)
    gate.step(0.0)
    gate.on_control_reset(TICK)
    out = gate.step(TICK)
    for side in SIDES:
        np.testing.assert_allclose(out[side], custom[side], atol=EPS)


def test_ramp_reset_homes_at_the_homing_rate_then_waits():
    start = offset(HOME, 0, 0.5)
    g, t = engaged_arm("real", pose=start)
    g.on_control_reset(t)
    v_home = 0.25 * LIMITS["right"].velocity
    prev = g.command.copy()
    ticks = 0
    while True:
        t += TICK
        ticks += 1
        out = feed(g, t, ik=start, state=g.command).astype(np.float64)
        assert np.all(np.abs(out - prev) <= v_home * TICK + EPS)
        prev = out
        if g.reason != HOMING:
            break
        assert ticks < 1000
    assert ticks == pytest.approx(0.5 / (0.5 * TICK), abs=2)
    np.testing.assert_allclose(g.command, HOME, atol=EPS)
    assert (g.state, g.reason) == (HOLDING, WAITING)  # IK still at the old pose
    t += TICK
    feed(g, t, ik=HOME, state=HOME)
    assert g.state == ENGAGED


def test_ramp_homing_pauses_without_feedback():
    g, t = engaged_arm("real", pose=offset(HOME, 0, 0.5))
    g.on_control_reset(t)
    t += TICK
    feed(g, t, state=g.command)
    held = g.command.copy()
    for _ in range(20):
        t += TICK
        g.step(t)
    assert (g.state, g.reason) == (HOLDING, STALE_STATE)
    assert np.all(np.abs(g.command - held) <= 0.5 * 0.05 + EPS)  # at most until stale


def test_ramp_homing_is_a_straight_line_in_joint_space():
    start = HOME + np.array([0.4, 0.3, -0.5, 0.2, 0.9, 0.0, -0.6, -0.3])
    g, t = engaged_arm("real", pose=start)
    g.on_control_reset(t)
    direction = start - HOME
    fractions = []
    while g.reason == HOMING or not fractions:
        t += TICK
        feed(g, t, state=g.command)
        remaining = g.command - HOME
        fraction = remaining[0] / direction[0]
        np.testing.assert_allclose(remaining, fraction * direction, atol=1e-9)
        fractions.append(fraction)
        assert len(fractions) < 2000
    assert fractions[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(fractions) <= 0.0)


@pytest.mark.parametrize(("profile", "scale"), [("sim", 0.8), ("real", 0.5)])
def test_sustained_motion_never_exceeds_the_profile_caps(profile, scale):
    g, t = engaged_arm(profile)
    caps = scale * LIMITS["right"].velocity
    peak = np.zeros(DOF)
    prev = g.command.copy()
    direction = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
    for _ in range(40):
        t += TICK
        # Keep the IK just inside engage_tol of the command, so it always pulls at full speed.
        ik = g.command + 0.09 * direction
        out = feed(g, t, ik=ik, state=g.command).astype(np.float64)
        assert g.state == ENGAGED
        peak = np.maximum(peak, np.abs(out - prev) / TICK)
        prev = out
    np.testing.assert_allclose(peak[:ARM_DOF], caps[:ARM_DOF], rtol=1e-4)
    assert np.all(peak <= caps + EPS / TICK)


def test_state_event_only_on_change():
    gate = Gate(LIMITS, profile_params("sim"))
    first = gate.state_event()
    assert first == {s: {"state": HOLDING, "reason": NO_STATE, "homing": False} for s in SIDES}
    gate.step(0.0)
    assert gate.state_event() is None
    for side in SIDES:
        gate.on_state(side, LIMITS[side].home, 0.0)
        gate.on_ik(side, LIMITS[side].home, 0.0)
    gate.step(0.0)
    assert gate.state_event() == {
        s: {"state": ENGAGED, "reason": "", "homing": False} for s in SIDES
    }
    gate.step(TICK)
    assert gate.state_event() is None


def _homing_flags(gate: Gate) -> dict[str, bool]:
    return {side: entry["homing"] for side, entry in gate.status().items()}


def test_homing_flag_stays_true_until_the_ramp_arrives_through_stalls_and_faults():
    """DATA-4 residual: a feedback stall mid-ramp turns the reason into stale_state (and a
    fault into fault), but the arm has not arrived and resumes homing by itself. The
    episode node blocks a start on the flag, so it must stay true until the ramp ends."""
    gate = Gate(LIMITS, profile_params("real"))
    start = {s: offset(LIMITS[s].home, 0, 0.5 if s == "right" else -0.5) for s in SIDES}
    t = 0.0
    for side in SIDES:
        gate.on_state(side, start[side], t)
        gate.on_ik(side, start[side], t)
    gate.step(t)
    assert gate.state_event() == {
        s: {"state": ENGAGED, "reason": "", "homing": False} for s in SIDES
    }

    def tick(fresh=SIDES):
        nonlocal t
        t += TICK
        for side in fresh:
            gate.on_state(side, gate.arms[side].command, t)
        gate.step(t)
        return gate.state_event()

    gate.on_control_reset(t)
    event = gate.state_event()
    assert event == {s: {"state": HOLDING, "reason": HOMING, "homing": True} for s in SIDES}
    for _ in range(40):
        assert tick() is None
    # The right feedback stalls mid-ramp: the reason changes, the flag does not.
    events = [tick(fresh=("left",)) for _ in range(20)]
    stalled = [e for e in events if e is not None]
    assert stalled[-1]["right"] == {"state": HOLDING, "reason": STALE_STATE, "homing": True}
    assert _homing_flags(gate) == {"right": True, "left": True}
    held = gate.arms["right"].command.copy()
    # A fault latched mid-ramp: still not home.
    gate.on_state("left", offset(gate.arms["left"].command, 3, np.nan), t)
    assert gate.status()["left"] == {"state": FAULT, "reason": NAN_STATE, "homing": True}
    tick(fresh=("right",))
    assert gate.arms["right"].reason == HOMING and gate.arms["right"].homing
    assert not np.array_equal(gate.arms["right"].command, held)  # the ramp resumed
    # The right arm arrives: the flag drops together with the reason.
    while gate.arms["right"].reason == HOMING:
        tick(fresh=("right",))
        assert t < 5.0
    np.testing.assert_allclose(gate.arms["right"].command, LIMITS["right"].home, atol=EPS)
    # No IK since t=0: after the ramp the arm waits for a fresh one.
    assert gate.status()["right"] == {"state": HOLDING, "reason": STALE_IK, "homing": False}
    assert gate.status()["left"]["homing"]  # still faulted mid-ramp
    # control=reset clears the fault and restarts the left ramp, which then completes.
    gate.on_control_reset(t)
    while gate.arms["left"].homing:
        tick()
        assert t < 10.0
    assert _homing_flags(gate) == {"right": False, "left": False}


def test_a_reset_during_a_stall_publishes_the_homing_flag():
    """The reason stays stale_state across the reset, so only the flag makes it an event."""
    gate = Gate(LIMITS, profile_params("real"))
    for side in SIDES:
        gate.on_state(side, LIMITS[side].home, 0.0)
    t = 0.0
    while gate.arms["right"].reason != STALE_STATE:
        t += TICK
        gate.step(t)
    gate.state_event()
    gate.on_control_reset(t)
    t += TICK
    gate.step(t)
    event = gate.state_event()
    assert event is not None
    assert event["right"] == {"state": HOLDING, "reason": STALE_STATE, "homing": True}


def test_snap_resets_never_report_homing():
    gate = Gate(LIMITS, profile_params("sim"))
    for side in SIDES:
        gate.on_state(side, offset(LIMITS[side].home, 0, 0.3), 0.0)
    gate.step(0.0)
    gate.on_control_reset(0.0)
    gate.on_reset_position(np.concatenate([LIMITS[s].home for s in SIDES]), 0.0)
    gate.step(TICK)
    assert _homing_flags(gate) == {"right": False, "left": False}


# -- node ------------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def make_node(argv: list[str] | None = None):
    args = gate_node.parse_args(argv or [])
    gate = Gate(load_limits(args.limits), gate_node.params_from_args(args))
    sent: list[tuple[str, pa.Array, dict]] = []
    clock = FakeClock()
    node = gate_node.GateNode(
        gate, lambda i, v, m: sent.append((i, v, m)), clock=clock, log=lambda _: None
    )
    return node, sent, clock


def event(input_id: str, value: pa.Array) -> dict:
    return {"type": "INPUT", "id": input_id, "value": value, "metadata": {}}


def tick() -> dict:
    return event("tick", pa.array([], type=pa.null()))


def outputs(sent, name):
    return [v for i, v, _ in sent if i == name]


def test_node_sends_commands_on_ticks_once_feedback_arrived():
    node, sent, clock = make_node()
    home = LIMITS["right"].home
    events = [
        tick(),
        event("state_right", arrow.state_array(home, np.zeros(8), np.zeros(8))),
        event("state_left", pa.array(LIMITS["left"].home, type=pa.float32())),  # flat form
        event("position_right", arrow.qpos_array(home)),
        event("position_left", arrow.qpos_array(LIMITS["left"].home)),
        tick(),
        tick(),
    ]
    for e in events:
        clock.t += 0.001
        assert node.handle(e)
    assert len(outputs(sent, "cmd_right")) == 2 and len(outputs(sent, "cmd_left")) == 2
    _, value, metadata = next(s for s in sent if s[0] == "cmd_right")
    assert value.type == arrow.QPOS_TYPE
    np.testing.assert_allclose(arrow.values(value, "qpos"), home, atol=EPS)
    assert isinstance(metadata["timestamp"], int)
    states = [json.loads(arrow.text(v)) for v in outputs(sent, "gate_state")]
    assert states[0]["right"] == {"state": HOLDING, "reason": NO_STATE, "homing": False}
    assert states[-1] == {s: {"state": ENGAGED, "reason": "", "homing": False} for s in SIDES}
    assert len(states) == len({json.dumps(s, sort_keys=True) for s in states})


def test_node_quit_and_stop_end_the_loop():
    node, sent, _ = make_node()
    home = LIMITS["right"].home
    stream = [
        event("state_right", arrow.qpos_array(home)),
        event("control", arrow.string_array("quit")),
        tick(),
    ]
    gate_node.run(iter(stream), node)
    assert outputs(sent, "cmd_right") == []  # the tick after quit was never handled
    node2, sent2, _ = make_node()
    gate_node.run(iter([{"type": "STOP"}, tick()]), node2)
    assert sent2 == []


def test_node_reset_position_and_control_reset():
    node, sent, clock = make_node(["--profile", "sim"])
    homes = {s: LIMITS[s].home for s in SIDES}
    for side in SIDES:
        node.handle(event(f"state_{side}", arrow.qpos_array(homes[side])))
    reset = np.concatenate([offset(homes["right"], 0, 0.1), offset(homes["left"], 0, -0.1)])
    node.handle(event("reset_position", arrow.qpos_array(reset)))
    clock.t += TICK
    node.handle(tick())
    np.testing.assert_allclose(
        arrow.values(outputs(sent, "cmd_right")[-1], "qpos"), reset[:8], atol=EPS
    )
    np.testing.assert_allclose(
        arrow.values(outputs(sent, "cmd_left")[-1], "qpos"), reset[8:], atol=EPS
    )
    node.handle(event("position_right", arrow.qpos_array(offset(homes["right"], 0, np.nan))))
    assert node.gate.arms["right"].state == FAULT
    node.handle(event("control", arrow.string_array("reset")))
    assert node.gate.arms["right"].state == HOLDING


def test_node_publishes_the_homing_flag_and_logs_a_paused_ramp():
    args = gate_node.parse_args(["--profile", "real"])
    gate = Gate(load_limits(args.limits), gate_node.params_from_args(args))
    sent: list[tuple[str, pa.Array, dict]] = []
    lines: list[str] = []
    clock = FakeClock()
    node = gate_node.GateNode(gate, lambda i, v, m: sent.append((i, v, m)), clock, lines.append)
    for side in SIDES:
        node.handle(event(f"state_{side}", arrow.qpos_array(offset(LIMITS[side].home, 0, 0.2))))
    node.handle(event("control", arrow.string_array("reset")))
    state = json.loads(arrow.text(outputs(sent, "gate_state")[-1]))
    assert state["right"] == {"state": HOLDING, "reason": HOMING, "homing": True}
    assert lines[-1] == "[dora-openarm-gate] right=holding(homing)  left=holding(homing)"
    clock.t += 1.0  # no feedback since: the ramp pauses
    node.handle(tick())
    state = json.loads(arrow.text(outputs(sent, "gate_state")[-1]))
    assert state["right"] == {"state": HOLDING, "reason": STALE_STATE, "homing": True}
    assert (
        lines[-1]
        == "[dora-openarm-gate] right=holding(stale_state) homing  left=holding(stale_state) homing"
    )


def test_node_malformed_input_faults_instead_of_crashing():
    node, _, _ = make_node()
    assert node.handle(event("position_right", pa.array(["not", "numbers"])))
    assert (node.gate.arms["right"].state, node.gate.arms["right"].reason) == (FAULT, SHAPE_IK)
    assert node.handle(event("control", pa.array([], type=pa.utf8())))  # empty: ignored


def test_node_profile_arguments():
    real = gate_node.params_from_args(gate_node.parse_args(["--profile", "real"]))
    assert (real.velocity_scale, real.reset_mode) == (0.5, "ramp")
    tuned = gate_node.params_from_args(
        gate_node.parse_args(["--profile", "real", "--velocity-scale", "0.3", "--jump-tol", "0.15"])
    )
    assert (tuned.velocity_scale, tuned.jump_tol, tuned.engage_tol) == (0.3, 0.15, 0.1)


def test_node_tracking_tol_argument(tmp_path):
    def params(argv):
        return gate_node.params_from_args(gate_node.parse_args(argv, environ={"ROBOT": "mujoco"}))

    assert params([]).tracking_tol == 0.25
    assert params(["--profile", "real"]).tracking_tol == 0.30
    assert params(["--tracking-tol", "0.4"]).tracking_tol == 0.4
    # The limits file the node is given wins over the profile's copy of the value.
    limits = tmp_path / "limits.yaml"
    text = DEFAULT_LIMITS_PATH.read_text(encoding="utf-8")
    limits.write_text(text.replace("sim: 0.25", "sim: 0.22"), encoding="utf-8")
    assert params(["--limits", str(limits)]).tracking_tol == 0.22
    assert params(["--limits", str(limits), "--tracking-tol", "0.4"]).tracking_tol == 0.4
    # A file that is not there at all leaves the profile's value: resolving the args must
    # keep working (oqt run does it before writing the run directory).
    assert params(["--limits", str(tmp_path / "gone.yaml")]).tracking_tol == 0.25


def test_node_publishes_not_tracking_without_the_homing_flag():
    node, sent, clock = make_node()
    home = LIMITS["right"].home
    node.handle(event("state_right", arrow.qpos_array(home)))
    node.handle(event("state_left", arrow.qpos_array(LIMITS["left"].home)))
    clock.t += TICK
    node.handle(event("state_right", arrow.qpos_array(offset(home, 3, -0.4))))
    node.handle(event("position_right", arrow.qpos_array(home)))
    node.handle(tick())
    state = json.loads(arrow.text(outputs(sent, "gate_state")[-1]))
    assert state["right"] == {"state": HOLDING, "reason": NOT_TRACKING, "homing": False}


@pytest.mark.parametrize(
    ("robot", "profile"),
    [
        ("mujoco", "sim"),
        ("dummy", "sim"),
        ("pedestal", "real"),
        ("cell", "real"),
        ("pedestal-zed", "real"),
        # Unset or unknown: the slower real profile, never the sim's speed on a real arm.
        (None, "real"),
        ("some-new-robot", "real"),
    ],
)
def test_auto_profile_follows_robot(robot, profile):
    environ = {} if robot is None else {"ROBOT": robot}
    assert gate_node.parse_args([], environ=environ).profile == profile


def test_an_explicit_profile_wins_over_robot():
    assert (
        gate_node.parse_args(["--profile", "real"], environ={"ROBOT": "mujoco"}).profile == "real"
    )


# -- follower restarts (start_epoch) --------------------------------------------

START = offset(HOME, 3, 0.2)  # where Start Arm leaves a Cell arm, relative to home


def test_a_new_start_epoch_reseeds_the_command_from_the_arm():
    g = make_arm()
    g.on_state(HOME, 0.0, 1)
    g.on_ik(HOME, 0.0)
    g.step(0.0)
    assert g.state == ENGAGED
    # Pause Arm, Start Arm: the arm is at its start pose, the IK still at the old hold.
    g.on_state(START, 0.1, 2)
    assert g.state == HOLDING and g.reason == WAITING
    np.testing.assert_allclose(g.command, g._clip(START))
    g.on_ik(START, 0.1)
    out = g.step(0.1)
    np.testing.assert_allclose(out, g._clip(START), atol=EPS)
    assert g.state == ENGAGED


def test_the_same_epoch_does_not_reseed():
    g, _ = engaged_arm()
    g.on_state(HOME, 0.0, 1)
    g.on_state(START, 0.004, 1)  # e.g. a shove: the tracking test's business, not a restart
    g.on_state(START, 0.008, None)
    np.testing.assert_allclose(g.command, g._clip(HOME))


def test_node_stamps_commands_with_the_followers_start_epoch():
    node, sent, _clock = make_node()
    home = LIMITS["right"].home
    state = arrow.state_array(home, np.zeros(8), np.zeros(8))
    node.handle({**event("state_right", state), "metadata": {"start_epoch": 3}})
    node.handle(
        event("state_left", arrow.state_array(LIMITS["left"].home, np.zeros(8), np.zeros(8)))
    )
    node.handle(tick())
    stamps = {i: m.get("start_epoch") for i, _, m in sent if i.startswith("cmd_")}
    assert stamps == {"cmd_right": 3, "cmd_left": None}
