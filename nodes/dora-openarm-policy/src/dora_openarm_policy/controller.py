"""What the policy node sends to the gate on each tick (no torch, no dora: testable).

Modes:

* ``passthrough`` -- the IK's output goes to the gate unchanged (the node forwards
  it as it arrives; :meth:`Controller.step` returns None). The clutch runs the left
  arm's intro and holds both arms this way, exactly as in data collection.
* ``running`` -- the policy drives both arms. Each chunk is 30 absolute joint
  targets at 30 fps, step 0 at the time its observation was taken. On every tick
  the target is the chunk interpolated at ``now``; a new chunk is cross-faded in
  over ``blend_s`` so the target never jumps, and the first one over
  ``start_blend_s`` from where the gate's command was when Start was pressed.
  Until the first chunk arrives the arms hold there.
* ``stopped`` -- hold the gate's command as it was when Stop was pressed (not the
  last target, which the gate may still be catching up with).
* ``returning`` -- ease in joint space from the gate's command to the IK's current
  output (where the clutch holds the arms), at ``return_speed`` rad/s for the
  joint that moves most, then back to passthrough. The gate sees no jump at the
  hand-over because the two agree by then.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIDES = ("right", "left")
DOF = 8  # 7 joints + gripper per arm

GRIPPER_R, GRIPPER_L = DOF - 1, 2 * DOF - 1  # in the 16-vector

PASSTHROUGH = "passthrough"
RUNNING = "running"
STOPPED = "stopped"
RETURNING = "returning"


@dataclass(frozen=True)
class ControllerParams:
    fps: float = 30.0
    blend_s: float = 0.15
    start_blend_s: float = 0.3
    return_speed: float = 0.3  # rad/s for the joint that moves most
    min_return_s: float = 0.5
    # Keep both grippers where the clutch had them at Start (right 60% open, left
    # closed on the patch), whatever the chunk says. They were fixed in every
    # demonstration, so their normalization range is zero and a policy's gripper
    # output is not to be trusted (MolmoAct2 returns ~0 for the right one).
    hold_grippers: bool = True


def _split(q16: np.ndarray) -> dict[str, np.ndarray]:
    return {"right": q16[:DOF].copy(), "left": q16[DOF:].copy()}


def _join(sides: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([sides["right"], sides["left"]])


def _smooth(a: float) -> float:
    a = min(max(a, 0.0), 1.0)
    return a * a * (3.0 - 2.0 * a)


class Controller:
    def __init__(self, params: ControllerParams | None = None) -> None:
        self.params = params or ControllerParams()
        self.mode = PASSTHROUGH
        self.ik: dict[str, np.ndarray] = {}
        self.cmd: dict[str, np.ndarray] = {}
        self.run_id = 0
        self.reason = ""
        # running
        self._chunk: np.ndarray | None = None  # (H, 16)
        self._chunk_t = 0.0
        self._chunk_arrived = 0.0
        self._blend_from: np.ndarray | None = None
        self._blend_t = 0.0
        self._blend_s = 0.0
        self._hold: np.ndarray | None = None
        self._grippers: np.ndarray | None = None
        self._last: np.ndarray | None = None
        # returning
        self._ret_from: np.ndarray | None = None
        self._ret_t = 0.0
        self._ret_s = 0.0
        self.after_return: str | None = None
        # Playback speed of the chunks: 0.5 runs a 1 s chunk over 2 s (same path,
        # half the speed). Changed live with set_speed.
        self.speed = 1.0

    # -- inputs -------------------------------------------------------------

    def on_ik(self, side: str, q: np.ndarray) -> None:
        self.ik[side] = np.asarray(q, np.float64)[:DOF].copy()

    def on_cmd(self, side: str, q: np.ndarray) -> None:
        self.cmd[side] = np.asarray(q, np.float64)[:DOF].copy()

    def _command16(self) -> np.ndarray | None:
        if all(s in self.cmd for s in SIDES):
            return _join(self.cmd)
        return self._last.copy() if self._last is not None else None

    # -- mode changes -------------------------------------------------------

    def start(self, now: float) -> str | None:
        """Start (or resume) the policy. Returns why it can't, or None."""
        here = self._command16()
        if here is None:
            return "the gate has not sent a command yet"
        self.run_id += 1
        self.mode = RUNNING
        self.reason = ""
        self._chunk = None
        self._hold = here
        self._grippers = here[[GRIPPER_R, GRIPPER_L]].copy()
        self._blend_from = None
        self._last = here.copy()
        return None

    def stop(self, now: float, reason: str = "stop") -> None:
        """Hold where the gate's command is now."""
        if self.mode == PASSTHROUGH:
            return
        here = self._command16()
        self.run_id += 1
        self.mode = STOPPED
        self.reason = reason
        self._chunk = None
        self._hold = here if here is not None else self._last
        self.after_return = None

    def go_back(self, now: float, then: str | None = None) -> str | None:
        """Ease back to the IK's output, then pass it through. ``then`` is kept for
        the caller (e.g. a clutch command to send once back)."""
        if not all(s in self.ik for s in SIDES):
            return "the IK has not sent a position yet"
        if self.mode == PASSTHROUGH:
            self.after_return = then
            return None
        here = self._command16()
        if here is None:
            self.mode = PASSTHROUGH
            self.after_return = then
            return None
        self.run_id += 1
        self.mode = RETURNING
        self.reason = ""
        self._chunk = None
        self._ret_from = here
        self._ret_t = now
        target = _join(self.ik)
        move = float(np.max(np.abs(target[: DOF - 1] - here[: DOF - 1])))
        move = max(move, float(np.max(np.abs(target[DOF : 2 * DOF - 1] - here[DOF : 2 * DOF - 1]))))
        self._ret_s = max(self.params.min_return_s, move / self.params.return_speed)
        self.after_return = then
        return None

    def to_passthrough(self) -> None:
        self.run_id += 1
        self.mode = PASSTHROUGH
        self._chunk = None

    def on_chunk(self, chunk: np.ndarray, t_obs: float, run_id: int, now: float) -> bool:
        """A chunk inferred from the observation taken at ``t_obs``. False if stale."""
        if self.mode != RUNNING or run_id != self.run_id:
            return False
        chunk = np.array(chunk, np.float64)
        if self.params.hold_grippers and self._grippers is not None:
            chunk[:, GRIPPER_R] = self._grippers[0]
            chunk[:, GRIPPER_L] = self._grippers[1]
        current = self._running_target(now)
        first = self._chunk is None
        self._chunk, self._chunk_t = chunk, t_obs
        self._chunk_arrived = now
        self._blend_from = current
        self._blend_t = now
        self._blend_s = self.params.start_blend_s if first else self.params.blend_s
        return True

    # -- output -------------------------------------------------------------

    def _index(self, now: float) -> float:
        """Where in the current chunk ``now`` is, in (fractional) steps."""
        return (now - self._chunk_t) * self.params.fps * self.speed

    def set_speed(self, now: float, speed: float) -> None:
        """Change the playback speed without moving the target: the current chunk's
        clock is re-anchored so that ``now`` stays at the same step."""
        speed = min(max(float(speed), 0.05), 1.0)
        if self._chunk is not None:
            f = self._index(now)
            self._chunk_t = now - f / (self.params.fps * speed)
        self.speed = speed

    def _chunk_at(self, now: float) -> np.ndarray:
        assert self._chunk is not None
        f = self._index(now)
        last = len(self._chunk) - 1
        f = min(max(f, 0.0), float(last))
        i = int(np.floor(f))
        j = min(i + 1, last)
        a = f - i
        return (1.0 - a) * self._chunk[i] + a * self._chunk[j]

    def _running_target(self, now: float) -> np.ndarray:
        if self._chunk is None:
            assert self._hold is not None
            return self._hold.copy()
        target = self._chunk_at(now)
        if self._blend_from is not None and self._blend_s > 0:
            a = _smooth((now - self._blend_t) / self._blend_s)
            if a >= 1.0:
                self._blend_from = None
            else:
                target = (1.0 - a) * self._blend_from + a * target
        return target

    def chunk_left_s(self, now: float) -> float:
        """Seconds of the current chunk not yet executed (0 without one)."""
        if self._chunk is None:
            return 0.0
        left_steps = (len(self._chunk) - 1) - self._index(now)
        return max(0.0, left_steps / (self.params.fps * self.speed))

    def chunk_age_s(self, now: float) -> float | None:
        """Seconds since the current chunk arrived (None without one)."""
        return None if self._chunk is None else now - self._chunk_arrived

    def step(self, now: float) -> dict[str, np.ndarray] | None:
        """This tick's targets per side, or None in passthrough."""
        if self.mode == PASSTHROUGH:
            return None
        if self.mode == RUNNING:
            out = self._running_target(now)
        elif self.mode == STOPPED:
            if self._hold is None:
                return None
            out = self._hold.copy()
        else:  # RETURNING
            assert self._ret_from is not None
            target = _join(self.ik)
            a = _smooth((now - self._ret_t) / self._ret_s)
            out = (1.0 - a) * self._ret_from + a * target
            if a >= 1.0:
                self.to_passthrough()
                self._last = out.copy()
                return _split(out)
        self._last = out.copy()
        return _split(out)
