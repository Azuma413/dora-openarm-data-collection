"""Why the arms look stopped while the policy runs: once a second, per arm.

A stall is one of two things, and they need different fixes:

* the policy holds still: its own target barely moves ("policy holding");
* the policy moves on but the arm does not follow: the target moves, the arm's
  measured speed is ~0 and the gap between the gate's command and the measured
  joints grows, usually with a high joint torque ("blocked by contact?").

Numbers per arm, joints 1-7 only (grippers are held fixed):

  target_speed   how far the policy's target moved over the last second / 1 s
  arm_speed      max |measured joint velocity|
  tracking       max |gate command - measured position|, and which joint
  gate_lag       max |policy target - gate command| (the gate's speed caps)
  torque         max |measured joint torque|, and which joint
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

JOINTS = slice(0, 7)
WINDOW_S = 1.0
# Below these the policy's target / the arm counts as not moving.
STILL_TARGET = 0.02  # rad/s
STILL_ARM = 0.02  # rad/s
# A command the arm is this far from, while not moving, looks like contact.
BLOCKED_TRACKING = 0.06  # rad


@dataclass
class ArmSample:
    target_speed: float
    arm_speed: float
    tracking: float
    tracking_joint: int
    gate_lag: float
    torque: float
    torque_joint: int
    verdict: str

    def text(self) -> str:
        return (
            f"target {self.target_speed:.2f} rad/s, arm {self.arm_speed:.2f} rad/s, "
            f"cmd-meas {self.tracking:.3f} rad (J{self.tracking_joint + 1}), "
            f"gate lag {self.gate_lag:.3f} rad, torque {self.torque:.1f} Nm (J{self.torque_joint + 1})"
            + (f" -> {self.verdict}" if self.verdict else "")
        )

    def as_dict(self) -> dict:
        return {
            "target_speed": round(self.target_speed, 3),
            "arm_speed": round(self.arm_speed, 3),
            "tracking": round(self.tracking, 3),
            "tracking_joint": self.tracking_joint + 1,
            "gate_lag": round(self.gate_lag, 3),
            "torque": round(self.torque, 2),
            "torque_joint": self.torque_joint + 1,
            "verdict": self.verdict,
        }


def verdict(target_speed: float, arm_speed: float, tracking: float) -> str:
    if target_speed < STILL_TARGET:
        return "policy holding"
    if arm_speed < STILL_ARM and tracking >= BLOCKED_TRACKING:
        return "blocked by contact?"
    return ""


class StallMonitor:
    """Fed every tick while the policy runs; ``sample`` once a second."""

    def __init__(self) -> None:
        self._targets: dict[str, collections.deque] = {}

    def reset(self) -> None:
        self._targets.clear()

    def on_target(self, side: str, now: float, target: np.ndarray) -> None:
        history = self._targets.setdefault(side, collections.deque())
        history.append((now, np.asarray(target, np.float64)[JOINTS].copy()))
        while history and now - history[0][0] > WINDOW_S + 0.5:
            history.popleft()

    def sample(
        self,
        side: str,
        now: float,
        command: np.ndarray | None,
        measured: np.ndarray | None,
        velocity: np.ndarray | None,
        torque: np.ndarray | None,
    ) -> ArmSample | None:
        history = self._targets.get(side)
        if not history or command is None or measured is None:
            return None
        latest_t, latest = history[-1]
        old_t, old = history[0]
        for t, q in history:  # the sample closest to one window back
            if latest_t - t <= WINDOW_S:
                old_t, old = t, q
                break
        span = max(latest_t - old_t, 1e-3)
        target_speed = float(np.max(np.abs(latest - old)) / span) if latest_t > old_t else 0.0
        command = np.asarray(command, np.float64)[JOINTS]
        measured = np.asarray(measured, np.float64)[JOINTS]
        gap = np.abs(command - measured)
        arm_speed = (
            float(np.max(np.abs(np.asarray(velocity)[JOINTS]))) if velocity is not None else float("nan")
        )
        torques = np.abs(np.asarray(torque, np.float64)[JOINTS]) if torque is not None else np.zeros(7)
        tracking = float(gap.max())
        return ArmSample(
            target_speed=target_speed,
            arm_speed=arm_speed,
            tracking=tracking,
            tracking_joint=int(gap.argmax()),
            gate_lag=float(np.max(np.abs(latest - command))),
            torque=float(torques.max()),
            torque_joint=int(torques.argmax()),
            verdict=verdict(target_speed, arm_speed if np.isfinite(arm_speed) else 0.0, tracking),
        )
