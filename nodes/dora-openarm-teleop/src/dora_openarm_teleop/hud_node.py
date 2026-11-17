"""dora-openarm-teleop-hud: the headset HUD text (see hud.py).

Inputs:
  tick           - timer; the HUD is (re)sent from here, at most every 0.2 s
  command        - STRING from the data collection UI (start/success/fail/cancel/quit)
  clutch_state   - STRING(JSON) from dora-openarm-clutch
  gate_state     - STRING(JSON) from dora-openarm-gate

Output:
  hud            - STRING(JSON) for the webxr node's `hud` input
"""

from __future__ import annotations

import time

import pyarrow as pa

from dora_openarm_teleop import arrow
from dora_openarm_teleop.hud import Hud


def handle(hud: Hud, event: dict, now: float) -> str | None:
    """Apply one INPUT event; return a HUD message to send, if any."""
    input_id = event["id"]
    try:
        text = arrow.text(event["value"])
    except Exception:  # noqa: BLE001 - a malformed input must not stop the HUD
        return None
    if input_id == "command":
        hud.on_command(text.strip(), dict(event.get("metadata") or {}), now)
    elif input_id == "clutch_state":
        hud.on_clutch_state(text)
    elif input_id == "gate_state":
        hud.on_gate_state(text)
    return hud.message(now) if input_id == "tick" or hud.haptic else None


def main() -> None:
    from dora import Node

    node = Node()
    hud = Hud()
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "tick":
            message = hud.message(time.monotonic())
        else:
            message = handle(hud, event, time.monotonic())
        if message is not None:
            node.send_output("hud", pa.array([message]), {"timestamp": arrow.now_ns()})
        if hud.quitting:
            break


if __name__ == "__main__":
    main()
