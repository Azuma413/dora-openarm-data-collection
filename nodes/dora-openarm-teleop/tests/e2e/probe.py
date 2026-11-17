#!/usr/bin/env python3
"""Records what the pipeline produced; writes JSON to $OUT on STOP."""

import json
import os
import time

from dora import Node

from dora_openarm_teleop import arrow


def main() -> None:
    node = Node()
    log: dict[str, list] = {}
    start = time.monotonic()
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        t = time.monotonic() - start
        name = event["id"]
        if name in ("clutch_state", "gate_state", "hud"):
            value = arrow.text(event["value"])
        else:
            key = "pose" if name.startswith("target") else "qpos"
            value = [float(v) for v in arrow.values(event["value"], key)]
        log.setdefault(name, []).append([t, value])
    with open(os.environ["OUT"], "w", encoding="utf-8") as f:
        json.dump(log, f)


if __name__ == "__main__":
    main()
