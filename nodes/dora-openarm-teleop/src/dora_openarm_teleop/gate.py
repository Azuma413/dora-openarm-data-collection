"""Joint-space safety gate between the IK and the arm (docs/teleop.md, Gate).

The gate owns the command `c` that is actually sent to the arm (and recorded as the
action). It only ever moves `c` towards the IK solution at a bounded joint velocity, and
stops moving it whenever the IK or the arm feedback cannot be trusted. That keeps the
command continuous, which is exactly what openarm_driver's JointDeltaPosChecker needs:
it raises (and kills the dora-openarm node) once a target is more than 1.0 rad away
from its last, velocity-clamped command (see docs/teleop.md, Gate).

Jumps are measured on the raw IK solution, before it is clipped into the limits: an IK
joint more than jump_tol past [lower + margin, upper - margin] holds with reason
"ik_limit" instead of streaming a clamped joint next to joints solved for another
configuration. dora_openarm_teleop.ik_model gives the IK these same ranges so it does not plan past them,
and IK_VELOCITY_FRACTION of the gate's velocity caps so the gate never clips its motion.

Engaging compares the IK with the command `c`, never with the measured pose: a real arm
(and, with the pedestal motor profile, the simulated one) sags away from its command
because the driver sends MIT kp/kd with tau = 0 and nothing feeds gravity forward
(openarm_driver driver.py:209-218). The measured steady-state joint error is 72 mrad at
home and 154-158 mrad with the arm extended, so "the raw IK is within engage_tol of the
measured pose" was unreachable at those poses and the gate held forever. What the
measurement is still needed for is the other question - is the arm following at all? -
and that is a much looser test: max|measured - c| < tracking_tol (0.25 rad sim, 0.30 rad
real, joint_limits.yaml). A stuck, disabled or held arm holds with reason
"not_tracking" instead of being handed a target it is nowhere near.

An IK further than engage_tol from `c` but no further than jump_tol no longer refuses to
engage either: the gate ramps `c` towards it at homing_scale (reason "homing", the same
slow unsteered motion as the reset ramp) until it is inside engage_tol. That is what lets
a sagged start converge, since `c` is seeded from the sagged measurement and the IK
therefore starts a sag away from it. The ramp cannot run away: it stops as soon as the IK
is more than jump_tol from `c` (a moving IK outruns it), and the tracking test bounds it
to tracking_tol from where the arm actually is, so a frozen arm stops it after tracking_tol.

Invariants, for every arm and every pair of consecutive outputs:
  * the output is finite,
  * the output is inside [lower + margin, upper - margin],
  * |c_k - c_{k-1}| <= v * clip(t_k - t_{k-1}, 0, max_dt) per joint,
    except across a reset in "snap" mode (the sim teleports the plant too).

Pure logic: numpy and pyyaml only, so the node, the unit tests and
scripts/hazard_shadow.py (which runs inside the openarm_driver venv) share this code.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

SIDES: tuple[str, ...] = ("right", "left")
ARM_DOF = 7
DOF = 8  # joint1..joint7, gripper

ENGAGED = "engaged"
HOLDING = "holding"
FAULT = "fault"

# Reasons published in gate_state. Holding: why the command is frozen. Fault: what
# was rejected; only control=reset clears it.
NO_STATE = "no_state"  # no arm feedback yet, nothing is sent
NO_IK = "no_ik"
STALE_IK = "stale_ik"
STALE_STATE = "stale_state"
WAITING = "waiting"  # IK further than jump_tol from the command, nothing to ramp towards
JUMP = "jump"  # IK moved more than jump_tol away from the command
IK_LIMIT = "ik_limit"  # IK solved a joint more than jump_tol past the gate's range
NOT_TRACKING = "not_tracking"  # the arm is not following the command (stuck, off, held)
HOMING = "homing"  # ramping to the reset pose (reset_mode "ramp") or towards the IK
NAN_IK = "nan_ik"
NAN_STATE = "nan_state"
NAN_RESET = "nan_reset"
SHAPE_IK = "shape_ik"
SHAPE_STATE = "shape_state"
SHAPE_RESET = "shape_reset"

ResetMode = Literal["snap", "ramp"]

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_LIMITS_PATH = PACKAGE_DIR / "joint_limits.yaml"

# Homing ends when every joint is this close to the reset pose.
_HOME_REACHED_RAD = 1e-6

# The IK's own J1-J7 velocity caps as a fraction of the gate's (dora_openarm_teleop.ik_model.write_ik_config).
# The IK moves at most cap / tick_hz per solve and the gate allows v * dt per tick; the 10 %
# leaves room for a gate tick that comes early, so the gate never has to clip a solution
# the IK reached at full speed and the command does not trail it.
IK_VELOCITY_FRACTION = 0.9


@dataclass(frozen=True)
class GateParams:
    """Tunables. Defaults are the sim profile."""

    velocity_scale: float = 0.8  # fraction of joint_velocity_limits
    engage_tol: float = 0.1  # rad, every arm joint
    jump_tol: float = 0.2  # rad, any arm joint
    # How far the measured pose may be from the command and still count as following it
    # (rad, every arm joint). Bigger than the worst sag of a PD-only arm without gravity
    # feed-forward, and the bound on how far the pre-engage ramp can pull the command
    # away from an arm that is not moving. TRACKING_TOL_KEY of the limits file.
    tracking_tol: float = 0.25
    stale_s: float = 0.05  # IK or feedback older than this freezes the command
    limit_margin: float = 0.05  # rad inside the joint limits (J1-J7)
    gripper_margin: float = 0.0  # rad inside the gripper limits
    max_dt: float = 0.05  # s, cap on the measured tick period used for rate limiting
    # What a reset does to the command: "snap" sets it to the reset pose at once (the sim
    # has teleported the arm there); "ramp" drives it there at homing_scale (a real arm
    # cannot teleport, and a jump would trip the driver's delta check).
    reset_mode: ResetMode = "snap"
    homing_scale: float = 0.25  # fraction of joint_velocity_limits while homing

    def __post_init__(self) -> None:
        if not 0.0 < self.velocity_scale <= 1.0:
            raise ValueError("velocity_scale must be in (0, 1]")
        if not 0.0 < self.homing_scale <= self.velocity_scale:
            # Keeps the step invariant a single bound: v * dt.
            raise ValueError("homing_scale must be in (0, velocity_scale]")
        if not 0.0 < self.engage_tol < self.jump_tol:
            raise ValueError("need 0 < engage_tol < jump_tol")
        if not self.jump_tol < self.tracking_tol:
            # The ramp closes a gap of up to jump_tol; a tracking_tol below that would
            # stop it half way and never let a sagged arm engage.
            raise ValueError("need jump_tol < tracking_tol")
        if self.stale_s <= 0.0 or self.max_dt <= 0.0:
            raise ValueError("stale_s and max_dt must be positive")
        if self.limit_margin < 0.0 or self.gripper_margin < 0.0:
            raise ValueError("margins must be non-negative")
        if self.reset_mode not in ("snap", "ramp"):
            raise ValueError(f"unknown reset_mode {self.reset_mode!r}")


# tracking_tol mirrors TRACKING_TOL_KEY of DEFAULT_LIMITS_PATH (a unit test pins them
# together); the node reads it from its own --limits file, so another limits file wins.
PROFILES: dict[str, GateParams] = {
    "sim": GateParams(velocity_scale=0.8, reset_mode="snap", tracking_tol=0.25),
    "real": GateParams(velocity_scale=0.5, reset_mode="ramp", tracking_tol=0.30),
}

# ROBOT values (modules/robot-$ROBOT.yaml) that drive no real motor. Every other
# value, an unset ROBOT and a robot module added later included, gets the slower
# "real" profile: a sim run made too slow is an annoyance, a real arm run at the sim's
# speed is not.
SIM_ROBOTS = frozenset({"mujoco", "dummy"})


def profile_for_robot(robot: str | None) -> str:
    """The gate profile for a dataflow's ROBOT ("auto")."""
    return "sim" if (robot or "").strip() in SIM_ROBOTS else "real"


def resolve_profile(profile: str, environ: Any) -> str:
    """``profile`` itself, or for "auto" the one ROBOT in ``environ`` calls for."""
    if profile == "auto":
        return profile_for_robot(environ.get("ROBOT"))
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r} (known: auto, {', '.join(PROFILES)})")
    return profile


def profile_params(profile: str, **overrides: Any) -> GateParams:
    """Profile defaults with the non-None overrides applied."""
    try:
        base = PROFILES[profile]
    except KeyError:
        raise ValueError(f"unknown profile {profile!r} (known: {', '.join(PROFILES)})") from None
    changes = {k: v for k, v in overrides.items() if v is not None}
    return dataclasses.replace(base, **changes)


@dataclass(frozen=True)
class ArmLimits:
    """Hard limits of one arm, before margins."""

    lower: np.ndarray  # (8,)
    upper: np.ndarray  # (8,)
    velocity: np.ndarray  # (8,) rad/s, unscaled
    home: np.ndarray  # (8,)


def load_limits(path: str | Path = DEFAULT_LIMITS_PATH) -> dict[str, ArmLimits]:
    """Read joint_limits.yaml (this package's, unless another path is given)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    velocity = _vector(data["joint_velocity_limits"], "joint_velocity_limits")
    if np.any(velocity <= 0.0):
        raise ValueError("joint_velocity_limits must be positive")
    limits = {}
    for side in SIDES:
        ranges = np.asarray(data["joint_limits"][side], dtype=np.float64)
        if ranges.shape != (DOF, 2) or not np.all(np.isfinite(ranges)):
            raise ValueError(f"joint_limits.{side} must be {DOF} finite [min, max] pairs")
        if np.any(ranges[:, 0] >= ranges[:, 1]):
            raise ValueError(f"joint_limits.{side} has min >= max")
        limits[side] = ArmLimits(
            lower=ranges[:, 0].copy(),
            upper=ranges[:, 1].copy(),
            velocity=velocity,
            home=_vector(data["home"][side], f"home.{side}"),
        )
    return limits


TRACKING_TOL_KEY = "gate_tracking_tol"


def load_tracking_tol(path: str | Path = DEFAULT_LIMITS_PATH) -> dict[str, float]:
    """Per-profile tracking_tol from the limits YAML (TRACKING_TOL_KEY).

    Every profile in PROFILES must have a positive finite entry: a limits file without
    one would silently leave the gate on a tolerance that does not belong to it.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError as error:  # a missing or unreadable --limits is a bad gate setting
        raise ValueError(f"{path}: {error.strerror or error}") from None
    values = (data or {}).get(TRACKING_TOL_KEY)
    if not isinstance(values, dict):
        values = {}
    tolerances = {}
    for profile in PROFILES:
        try:
            tol = float(values[profile])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{path}: {TRACKING_TOL_KEY}.{profile} must be a number") from None
        if not (math.isfinite(tol) and tol > 0.0):
            raise ValueError(f"{path}: {TRACKING_TOL_KEY}.{profile} must be positive")
        tolerances[profile] = tol
    return tolerances


def ik_velocity_caps(limits: dict[str, ArmLimits], velocity_scale: float) -> np.ndarray:
    """J1..J7 joint velocity caps (rad/s) for the IK: IK_VELOCITY_FRACTION of the gate's.

    The gate moves each joint at velocity_scale * joint_velocity_limits. openarm_control
    takes one list for both arms, so each entry is the smaller of the two sides. The
    gripper is not in it: the IK passes the gripper through without limiting it.
    """
    if not (math.isfinite(velocity_scale) and 0.0 < velocity_scale <= 1.0):
        raise ValueError("velocity_scale must be in (0, 1]")
    gate_caps = velocity_scale * np.min([limits[side].velocity for side in SIDES], axis=0)
    return IK_VELOCITY_FRACTION * gate_caps[:ARM_DOF]


def _vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (DOF,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be {DOF} finite numbers")
    return array


@dataclass
class _Input:
    """Last accepted sample of a stream and when it arrived."""

    value: np.ndarray | None = None
    time: float | None = None

    def age(self, now: float) -> float:
        return np.inf if self.time is None else now - self.time


class ArmGate:
    """Gate for one arm. Feed it with on_* and call step() on every tick."""

    def __init__(self, side: str, limits: ArmLimits, params: GateParams) -> None:
        self.side = side
        self.limits = limits
        self.params = params
        margin = np.full(DOF, self.params.limit_margin)
        margin[ARM_DOF] = self.params.gripper_margin
        self.lower = self.limits.lower + margin
        self.upper = self.limits.upper - margin
        if np.any(self.lower > self.upper):
            raise ValueError(f"{self.side}: margins leave an empty joint range")
        self.velocity = self.params.velocity_scale * self.limits.velocity
        self.home_velocity = self.params.homing_scale * self.limits.velocity
        self.reset_pose = self._clip(self.limits.home)
        self.command: np.ndarray | None = None
        self.ik = _Input()
        self.measured = _Input()
        # The follower's start_epoch of the latest state (None: never told).
        self.epoch: int | None = None
        self.engaged = False
        self.fault: str | None = None
        self.hold_reason = WAITING
        self.homing = False
        self.last_step: float | None = None
        self.state = HOLDING
        self.reason = NO_STATE

    # -- inputs ---------------------------------------------------------------

    def on_ik(self, qpos: np.ndarray, now: float) -> None:
        """IK solution (QPOS8)."""
        q = self._accept(qpos, SHAPE_IK, NAN_IK)
        if q is not None:
            self.ik = _Input(q, now)

    def on_state(self, qpos: np.ndarray, now: float, epoch: int | None = None) -> None:
        """Measured joint positions (qpos of STATE8), and the follower's start_epoch.

        A new start_epoch means the follower restarted; the command is re-seeded from
        the arm, as on the first state.
        """
        q = self._accept(qpos, SHAPE_STATE, NAN_STATE)
        if q is None:
            return
        self.measured = _Input(q, now)
        restarted = epoch is not None and self.epoch is not None and epoch != self.epoch
        if epoch is not None:
            self.epoch = epoch
        if self.command is None or restarted:
            # A measurement outside the margins moves the first command at most by the
            # margin (if the arm is inside its hard limits at all).
            self.command = self._clip(q)
        if restarted:
            # The follower restarted (Pause Arm, Start Arm) and moved to its start pose:
            # the old command is somewhere the arm is not. Start over from the arm.
            self.engaged = False
            self.homing = False
            self.hold_reason = WAITING
            if not self.fault:
                self._set(HOLDING, WAITING)

    def on_reset_position(self, qpos: np.ndarray, now: float) -> None:
        """The plant was reset to this pose (sim/reset_position, this arm's 8 values)."""
        q = self._accept(qpos, SHAPE_RESET, NAN_RESET)
        if q is None:
            return
        self.reset_pose = self._clip(q)
        self._restart()

    def on_control_reset(self, now: float) -> None:
        """control=reset: clear a fault and go back to the reset pose."""
        self.fault = None
        self._restart()

    # -- tick -----------------------------------------------------------------

    def step(self, now: float) -> np.ndarray | None:
        """Advance the command by one tick and return it (None before any feedback)."""
        if self.command is None:
            self._set(FAULT if self.fault else HOLDING, self.fault or NO_STATE)
            return None
        dt = (
            0.0
            if self.last_step is None
            else float(np.clip(now - self.last_step, 0.0, self.params.max_dt))
        )
        self.last_step = now
        self._advance(now, dt)
        return self.command.astype(np.float32)

    def _advance(self, now: float, dt: float) -> None:
        if self.fault:
            self._hold(FAULT, self.fault)
            return
        if self.measured.age(now) > self.params.stale_s:
            self._hold(HOLDING, STALE_STATE)
            return
        if self.homing:
            # Nobody is steering this motion, so it goes slowly and in a straight line in
            # joint space (all joints arrive together) to keep the swept volume predictable.
            self._move_straight_towards(self.reset_pose, self.home_velocity * dt)
            if np.all(np.abs(self.reset_pose - self.command) <= _HOME_REACHED_RAD):
                self.homing = False
            self._hold(HOLDING, HOMING)
            return
        if self.ik.age(now) > self.params.stale_s:
            self._hold(HOLDING, NO_IK if self.ik.time is None else STALE_IK)
            return

        # Measure on the raw IK, clip only the motion target. The IK integrates its own
        # joint state, so when it solves a joint past the gate's range its other joints
        # are solved for that out-of-range value: clamping the one joint and streaming the
        # rest would put the arm somewhere else entirely (GATE-1). Measured on the clipped
        # value, such a solution always looks close to the command, which sits at the limit.
        raw = self.ik.value
        target = self._clip(raw)
        gap = np.max(np.abs(raw[:ARM_DOF] - self.command[:ARM_DOF]))
        beyond = np.max(np.abs(raw[:ARM_DOF] - target[:ARM_DOF]))
        if beyond > self.params.jump_tol:
            # The command is inside the range, so this also means gap > jump_tol: it is
            # a jump, told apart only so the HUD can say why. Re-engaging follows the jump
            # rule (raw IK back within engage_tol of the command).
            if self.engaged:
                self.hold_reason = JUMP
            self._hold(HOLDING, IK_LIMIT)
            return
        if self.engaged and gap > self.params.jump_tol:
            self.hold_reason = JUMP
            self._hold(HOLDING, JUMP)
            return
        if not self.engaged and not self._approach(target, gap, dt):
            return
        self._move_towards(target, self.velocity * dt)
        self._set(ENGAGED, "")

    def _approach(self, target: np.ndarray, gap: float, dt: float) -> bool:
        """Engage on this IK, or close the gap to it. True once engaged.

        The command is only ever compared with the IK; the measurement answers the
        separate question of whether the arm is following the command at all, which a
        sagging arm does (tracking_tol is well past the worst sag) and a stuck, disabled
        or held one does not. Engaging a stuck arm would hand the driver a target the arm
        is nowhere near, and ramping towards the IK would only wind that error up further,
        so the gate stops the ramp there too.
        """
        follow = float(np.max(np.abs(self.measured.value[:ARM_DOF] - self.command[:ARM_DOF])))
        if follow >= self.params.tracking_tol:
            self._hold(HOLDING, NOT_TRACKING)
            return False
        if gap >= self.params.engage_tol:
            # Too far to engage on, close enough to trust: walk the command over at the
            # homing rate (nobody is steering this motion either). A moving IK that
            # outruns the ramp leaves the jump_tol window and stops it.
            moved = gap <= self.params.jump_tol and self._move_straight_towards(
                target, self.home_velocity * dt
            )
            self._hold(HOLDING, HOMING if moved else self.hold_reason)
            return False
        self.engaged = True
        self.hold_reason = WAITING
        return True

    # -- helpers --------------------------------------------------------------

    def _accept(self, qpos: np.ndarray, shape_reason: str, nan_reason: str) -> np.ndarray | None:
        """Validated float64 copy, or None after latching a fault."""
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.shape != (DOF,):
            self._latch_fault(shape_reason)
            return None
        if not np.all(np.isfinite(q)):
            self._latch_fault(nan_reason)
            return None
        return q.copy()

    def _latch_fault(self, reason: str) -> None:
        if self.fault is None:
            self.fault = reason
        self.engaged = False
        self._set(FAULT, self.fault)

    def _restart(self) -> None:
        self.engaged = False
        self.hold_reason = WAITING
        if self.params.reset_mode == "ramp":
            self.homing = True
        elif self.command is not None:
            self.homing = False
            self.command = self.reset_pose.copy()
        # Publish the new status right away instead of on the next tick.
        if self.fault:
            self._set(FAULT, self.fault)
        elif self.command is None:
            self._set(HOLDING, NO_STATE)
        else:
            self._set(HOLDING, HOMING if self.homing else WAITING)

    def _move_towards(self, target: np.ndarray, max_step: np.ndarray) -> None:
        step = np.clip(target - self.command, -max_step, max_step)
        # The final clip cannot lengthen the step: the previous command is in range.
        self.command = self._clip(self.command + step)

    def _move_straight_towards(self, target: np.ndarray, max_step: np.ndarray) -> bool:
        """Move on the straight line to target, capped per joint. False if already there."""
        delta = target - self.command
        distance = np.abs(delta)
        moving = distance > 0.0
        if not np.any(moving):
            return False
        scale = min(1.0, float(np.min(max_step[moving] / distance[moving])))
        self.command = self._clip(self.command + scale * delta)
        return scale > 0.0

    def _clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=np.float64), self.lower, self.upper)

    def _hold(self, state: str, reason: str) -> None:
        self.engaged = False
        self._set(state, reason)

    def _set(self, state: str, reason: str) -> None:
        self.state = state
        self.reason = reason


class Gate:
    """Both arms plus the gate_state change detection."""

    def __init__(self, limits: dict[str, ArmLimits], params: GateParams) -> None:
        self.params = params
        self.arms = {side: ArmGate(side, limits[side], params) for side in SIDES}
        self._published: dict[str, dict[str, Any]] | None = None

    def on_ik(self, side: str, qpos: np.ndarray, now: float) -> None:
        self.arms[side].on_ik(qpos, now)

    def on_state(self, side: str, qpos: np.ndarray, now: float, epoch: int | None = None) -> None:
        self.arms[side].on_state(qpos, now, epoch)

    def on_reset_position(self, qpos16: np.ndarray, now: float) -> None:
        """QPOS16, right 8 then left 8."""
        q = np.asarray(qpos16, dtype=np.float64).reshape(-1)
        if q.shape != (2 * DOF,):
            for arm in self.arms.values():
                arm.on_reset_position(np.empty(0), now)
            return
        for i, side in enumerate(SIDES):
            self.arms[side].on_reset_position(q[i * DOF : (i + 1) * DOF], now)

    def on_control_reset(self, now: float) -> None:
        for arm in self.arms.values():
            arm.on_control_reset(now)

    def step(self, now: float) -> dict[str, np.ndarray | None]:
        return {side: arm.step(now) for side, arm in self.arms.items()}

    def status(self) -> dict[str, dict[str, Any]]:
        """The gate_state payload: per side state, reason and homing.

        ``homing`` is true from a reset (control=reset, or reset_position in ramp mode)
        until the ramp has reached the reset pose, whatever the reason says meanwhile: a
        feedback stall pauses the ramp with reason "stale_state" and a fault latched
        mid-ramp shows as "fault", but the arm has not arrived and resumes the move on
        its own once the feedback is back (DATA-4). The pre-engage ramp reports reason
        "homing" as well but leaves the flag false: it is not going to the reset pose and
        it ends by itself within jump_tol, so it must not block an episode start.
        """
        return {
            side: {"state": arm.state, "reason": arm.reason, "homing": bool(arm.homing)}
            for side, arm in self.arms.items()
        }

    def state_event(self) -> dict[str, dict[str, Any]] | None:
        """The gate_state payload if it changed since the last call, else None."""
        current = self.status()
        if current == self._published:
            return None
        self._published = current
        return current
