"""dora-openarm-gate: joint-space safety gate node (docs/teleop.md, Gate).

Inputs:  position_right/left (QPOS8, ik), state_right/left (STATE8 or QPOS8, arm feedback),
         tick (timer), reset_position (QPOS16), control (STRING: reset / quit)
Outputs: cmd_right/left (QPOS8, every tick once that arm has feedback),
         gate_state (JSON {side: {state, reason, homing}}, on change)

The logic lives in dora_openarm_teleop.gate; this file only maps dora events onto it, so the loop can be
driven by a fake event iterator in tests.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from dora import Node

from dora_openarm_teleop import arrow
from dora_openarm_teleop.gate import (
    DEFAULT_LIMITS_PATH,
    HOMING,
    PROFILES,
    SIDES,
    TRACKING_TOL_KEY,
    Gate,
    GateParams,
    load_limits,
    load_tracking_tol,
    profile_params,
    resolve_profile,
)

Sender = Callable[[str, pa.Array, dict[str, Any]], None]

_IK_INPUTS = {f"position_{side}": side for side in SIDES}
_STATE_INPUTS = {f"state_{side}": side for side in SIDES}


def parse_args(
    argv: list[str] | None = None, environ: Mapping[str, str] | None = None
) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.profile = resolve_profile(args.profile, os.environ if environ is None else environ)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dora-openarm-gate", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile",
        choices=["auto", *sorted(PROFILES)],
        default="auto",
        help="auto: sim for ROBOT=mujoco|dummy, real for anything else (default)",
    )
    parser.add_argument(
        "--limits",
        type=Path,
        default=DEFAULT_LIMITS_PATH,
        help=f"joint limits YAML (default: {DEFAULT_LIMITS_PATH})",
    )
    # None means "take the profile's value".
    parser.add_argument("--velocity-scale", type=float, help="sim 0.8, real 0.5")
    parser.add_argument("--engage-tol", type=float, help="rad (default 0.1)")
    parser.add_argument("--jump-tol", type=float, help="rad (default 0.2)")
    parser.add_argument(
        "--tracking-tol",
        type=float,
        help="rad, how far the measurement may be from the command and still engage "
        f"(default: {TRACKING_TOL_KEY} of --limits, sim 0.25 / real 0.30)",
    )
    parser.add_argument("--stale-s", type=float, help="s (default 0.05)")
    parser.add_argument("--limit-margin", type=float, help="rad (default 0.05)")
    parser.add_argument("--gripper-margin", type=float, help="rad (default 0)")
    parser.add_argument(
        "--reset-mode", choices=["snap", "ramp"], help="sim snap (teleport), real ramp (home)"
    )
    parser.add_argument("--homing-scale", type=float, help="fraction of the caps (default 0.25)")
    return parser


def params_from_args(args: argparse.Namespace) -> GateParams:
    """The gate's parameters, with the tracking tolerance read from the limits file.

    A limits file with no usable tolerance for the profile raises ValueError, like the
    parameter checks themselves, instead of falling back to another file's value: scripts/check-dataflow
    reports it as a bad gate setting. A file that is not there at all keeps the
    profile's value, because resolving args must not depend on the file existing yet;
    load_limits names the missing path a moment later when the node really starts.
    """
    tracking_tol = args.tracking_tol
    if tracking_tol is None and Path(args.limits).is_file():
        tracking_tol = load_tracking_tol(args.limits)[args.profile]
    return profile_params(
        args.profile,
        velocity_scale=args.velocity_scale,
        engage_tol=args.engage_tol,
        jump_tol=args.jump_tol,
        tracking_tol=tracking_tol,
        stale_s=args.stale_s,
        limit_margin=args.limit_margin,
        gripper_margin=args.gripper_margin,
        reset_mode=args.reset_mode,
        homing_scale=args.homing_scale,
    )


def _qpos(value: pa.Array) -> np.ndarray:
    """qpos of a struct or a flat array; anything unreadable becomes an empty array,
    which the gate rejects as a shape fault instead of crashing the node."""
    try:
        return arrow.values(value, "qpos")
    except Exception:  # noqa: BLE001 - malformed input must not kill the gate
        return np.empty(0, dtype=np.float32)


# Motor statuses that still mean the qpos is a fresh reading (openarm_can's names).
# Anything else -- SILENT (no reply: the driver repeats the last value it got),
# COMMUNICATION_LOST, OVERCURRENT and the other faults, UNKNOWN -- does not.
_LIVE_MOTOR_STATUSES = frozenset({"ENABLED", "DISABLED"})


def state_faults(value: pa.Array) -> list[str]:
    """Why a STATE8 cannot be trusted as feedback (empty when it can).

    dora-openarm keeps publishing state when the motors stop answering or the CAN
    link goes down: the qpos is the last reply, stamped as new. Taking it would keep
    the gate engaged on an arm it cannot see. A flat QPOS8, or a struct without
    motor_status / bus, has nothing to say and is trusted as before.
    """
    try:
        if not pa.types.is_struct(value.type) or len(value) == 0:
            return []
        row = value[0].as_py() or {}
    except Exception:  # noqa: BLE001 - malformed input is the qpos check's job
        return []
    faults = [
        f"J{index + 1}={status}"
        for index, status in enumerate(row.get("motor_status") or [])
        if status not in _LIVE_MOTOR_STATUSES
    ]
    bus = row.get("bus")
    if isinstance(bus, Mapping) and bus.get("carrier") is False:
        faults.append("bus=no_carrier")
    return faults


class GateNode:
    """Maps dora events onto a Gate."""

    def __init__(
        self,
        gate: Gate,
        send: Sender,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.gate = gate
        self.send = send
        self.clock = clock
        self.log = log or (lambda line: print(line, file=sys.stderr, flush=True))
        # Last reported faults per side, so a fault is logged when it starts and ends.
        self._state_faults: dict[str, list[str]] = {}

    def handle(self, event: Mapping[str, Any]) -> bool:
        """Process one event. Returns False when the node should exit."""
        kind = event.get("type")
        if kind == "STOP":
            return False
        if kind != "INPUT":
            return True
        now = self.clock()
        input_id = event["id"]
        value = event["value"]
        keep_running = True
        if input_id == "tick":
            self._send_commands(now)
        elif input_id in _IK_INPUTS:
            self.gate.on_ik(_IK_INPUTS[input_id], _qpos(value), now)
        elif input_id in _STATE_INPUTS:
            side = _STATE_INPUTS[input_id]
            faults = state_faults(value)
            if faults != self._state_faults.get(side, []):
                self.log(
                    f"[dora-openarm-gate] {side}: ignoring arm state ({', '.join(faults)})"
                    if faults
                    else f"[dora-openarm-gate] {side}: arm state is live again"
                )
                self._state_faults[side] = faults
            if not faults:
                # A faulty state is dropped as if it never came: the feedback goes
                # stale and the gate holds (stale_state) instead of steering blind.
                epoch = arrow.start_epoch(event.get("metadata"))
                self.gate.on_state(side, _qpos(value), now, epoch)
        elif input_id == "reset_position":
            self.gate.on_reset_position(_qpos(value), now)
        elif input_id == "control":
            keep_running = self._control(value, now)
        self._publish_state()
        return keep_running

    def _control(self, value: pa.Array, now: float) -> bool:
        try:
            command = arrow.text(value).strip()
        except Exception:  # noqa: BLE001
            return True
        if command == "quit":
            return False
        if command == "reset":
            self.gate.on_control_reset(now)
        return True

    def _send_commands(self, now: float) -> None:
        for side, command in self.gate.step(now).items():
            if command is not None:
                metadata: dict[str, Any] = {"timestamp": arrow.now_ns()}
                epoch = self.gate.arms[side].epoch
                if epoch is not None:
                    # dora-openarm drops a command addressed to an earlier start.
                    metadata["start_epoch"] = epoch
                self.send(f"cmd_{side}", arrow.qpos_array(command), metadata)

    def _publish_state(self) -> None:
        event = self.gate.state_event()
        if event is None:
            return
        self.send("gate_state", arrow.json_array(event), {"timestamp": arrow.now_ns()})
        summary = "  ".join(
            f"{side}={s['state']}"
            + (f"({s['reason']})" if s["reason"] else "")
            # A ramp paused by a stall or a fault is still unfinished.
            + (" homing" if s["homing"] and s["reason"] != HOMING else "")
            for side, s in event.items()
        )
        self.log(f"[dora-openarm-gate] {summary}")


def run(events: Iterable[Mapping[str, Any]], node: GateNode) -> None:
    """Drive the node until quit, STOP or the end of the event stream."""
    for event in events:
        if event is None or not node.handle(event):
            break


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    params = params_from_args(args)
    gate = Gate(load_limits(args.limits), params)
    print(
        f"[dora-openarm-gate] profile={args.profile} limits={args.limits} {params}",
        file=sys.stderr,
        flush=True,
    )

    dora_node = Node()
    run(dora_node, GateNode(gate, dora_node.send_output))


if __name__ == "__main__":
    main()
