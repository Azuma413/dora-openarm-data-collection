# openarm-quest-teleop addition to the vendored dora-openarm-webxr tests
# (Apache-2.0, like the rest of this directory).

"""Dora node for the browser test: records webxr's outputs and drives its inputs.

Every webxr output wired to this node is appended to $LOG as one JSON line,
``{"t": unix seconds, "id": output, "value": [...]}``. Commands are read from
$CMD, one JSON object per line, on every tick: ``{"hud": {...}}`` goes out on
the ``hud`` output as a JSON string, ``{"scene_state": {...}}`` on
``scene_state`` (the sim's own output for the 3D view, PATCHES.md 15, stood in
for here) and ``{"control": "quit"}`` on ``control``.

``{"scene_state_binary": [byte, ...]}`` sends the same input in the sim's other
encoding -- a uint8 array rather than a utf8 string -- which is what
``--scene-state-format binary`` publishes, and what ``oqt run sim`` now asks
for by default. The two shapes leave this node differently, and everything
after it, node and page alike, has to take both.

After a quit the node gives webxr a few seconds and exits, so ``dora run`` ends
once webxr has gone too.
"""

import json
import os
import time

import dora
import pyarrow as pa

LOG = os.environ["LOG"]
CMD = os.environ["CMD"]

# Long enough for webxr to close its peers and stop uvicorn.
QUIT_GRACE_S = 5.0


def _values(value: pa.Array) -> list:
    if isinstance(value, pa.StructArray):
        return [list(value.field(name)[0].as_py()) for name in value.type.names]
    return value.to_pylist()


def _read_commands(offset: int) -> tuple[list[dict], int]:
    try:
        with open(CMD, encoding="utf-8") as commands:
            commands.seek(offset)
            lines = commands.readlines()
            offset = commands.tell()
    except FileNotFoundError:
        return [], offset
    return [json.loads(line) for line in lines if line.strip()], offset


def main() -> None:
    """Record and drive until a quit has had time to take effect."""
    node = dora.Node()
    offset = 0
    quit_at = None
    with open(LOG, "a", encoding="utf-8") as log:

        def record(input_id: str, value) -> None:
            log.write(json.dumps({"t": time.time(), "id": input_id, "value": value}))
            log.write("\n")
            log.flush()

        for event in node:
            if event["type"] != "INPUT":
                continue
            if event["id"] != "tick":
                record(event["id"], _values(event["value"]))
                continue
            if quit_at is not None:
                if time.time() - quit_at > QUIT_GRACE_S:
                    break
                continue
            commands, offset = _read_commands(offset)
            for command in commands:
                if "hud" in command:
                    node.send_output("hud", pa.array([json.dumps(command["hud"])]))
                if "scene_state" in command:
                    node.send_output(
                        "scene_state", pa.array([json.dumps(command["scene_state"])])
                    )
                if "scene_state_binary" in command:
                    # As the sim's binary encoding leaves it, and as the
                    # camera JPEGs arrive: raw bytes in a uint8 array.
                    node.send_output(
                        "scene_state",
                        pa.array(command["scene_state_binary"], type=pa.uint8()),
                    )
                if "control" in command:
                    node.send_output("control", pa.array([command["control"]]))
                    if command["control"] == "quit":
                        quit_at = time.time()
                record("driver-sent", command)


if __name__ == "__main__":
    main()
