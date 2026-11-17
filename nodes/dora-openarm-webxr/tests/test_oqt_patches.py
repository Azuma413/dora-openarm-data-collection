# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# openarm-quest-teleop: portions derived from enactic/dora-openarm-webxr
# tests/test_main.py and tests/test_webrtc.py (Apache-2.0).
#
# Tests for the openarm-quest-teleop patches (PATCHES.md): the hud and
# control inputs, the One Euro environment variables, the HUD replay on
# reconnect, and a syntax check of the page's modules.

import asyncio
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import time
import tomllib
import types

import pyarrow as pa
import pytest
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from dora_openarm_webxr import main, video, webrtc

STATIC = pathlib.Path(main.__file__).parent / "static"
# openarm-quest-teleop's config/, when this tree is vendored there.
PROJECT_CONFIG = pathlib.Path(main.__file__).resolve().parents[4] / "config"


class _FakeTransport:
    def __init__(self, running=True):
        self.hud = []
        self.closed = False
        self.running = running

    def send_hud(self, payload):
        self.hud.append(payload)

    def send_control(self, payload):
        pass

    async def close(self):
        self.closed = True


class _FakeNode:
    """Replays events the way dora.Node hands them out, then says stop.

    ``gate`` holds the events back until it returns True, so a test can
    wait for the Web server to be up before the quit arrives.
    """

    def __init__(self, events, gate=None):
        self._events = list(events)
        self._gate = gate
        self.consumed = []
        self.outputs = []

    def is_empty(self):
        # Always "empty", so every event goes through the to_thread path
        # the node takes when its queue is idle.
        return True

    def next(self, timeout=None):
        if self._gate is not None and not self._gate():
            time.sleep(timeout or 0.01)
            return {"type": "INPUT", "id": "not-an-input-of-this-node"}
        if not self._events:
            return None
        event = self._events.pop(0)
        self.consumed.append(event)
        return event

    def send_output(self, output_id, value, metadata=None):
        self.outputs.append((output_id, value))


def _input(input_id, value):
    return {"type": "INPUT", "id": input_id, "value": value, "metadata": {}}


# hud input


def test_parse_hud_design_example():
    # The HUD JSON of DESIGN.md section 10.3, as the episode node sends it.
    text = json.dumps(
        {
            "type": "hud",
            "lines": ["REC 00:12  ep 3", "Pick up the cube.", "L:HOLD  R:FOLLOW"],
            "recording": True,
            "haptic": "right",
        }
    )
    assert main._parse_hud(pa.array([text])) == {
        "type": "hud",
        "lines": ["REC 00:12  ep 3", "Pick up the cube.", "L:HOLD  R:FOLLOW"],
        "recording": True,
        "haptic": "right",
    }


def test_parse_hud_is_the_whole_state():
    # No lines clears the text: every message is the whole HUD.
    message = main._parse_hud(pa.array(['{"recording": false}']))
    assert message == {"type": "hud", "lines": [], "recording": False, "haptic": None}


def test_parse_hud_cuts_to_the_panel():
    lines = [f"line {index} " + "x" * 200 for index in range(20)]
    message = main._parse_hud(pa.array([json.dumps({"lines": lines})]))
    assert len(message["lines"]) == main._HUD_MAX_LINES
    assert all(len(line) == main._HUD_MAX_LINE_LENGTH for line in message["lines"])


def test_parse_hud_coerces_rather_than_drops():
    message = main._parse_hud(
        pa.array([json.dumps({"lines": "one\ntwo", "recording": 1, "haptic": "up"})])
    )
    assert message["lines"] == ["one", "two"]
    assert message["recording"] is True
    # An unknown haptic target buzzes nothing rather than failing the HUD.
    assert message["haptic"] is None


@pytest.mark.parametrize("haptic", ["right", "left", "both", None])
def test_parse_hud_haptic_targets(haptic):
    message = main._parse_hud(pa.array([json.dumps({"lines": [], "haptic": haptic})]))
    assert message["haptic"] == haptic


@pytest.mark.parametrize(
    "value",
    [
        pa.array(["not json"]),
        pa.array(['["a", "list"]']),
        pa.array([], type=pa.string()),
        pa.array([1.0], type=pa.float32()),
    ],
)
def test_parse_hud_rejects_malformed(value, capsys):
    assert main._parse_hud(value) is None
    assert "hud:" in capsys.readouterr().out


def test_parse_hud_accepts_other_string_encodings():
    text = '{"lines": ["a"]}'
    assert main._parse_hud(pa.array([text], type=pa.large_string()))["lines"] == ["a"]
    assert main._parse_hud(pa.array([text.encode()], type=pa.binary()))["lines"] == [
        "a"
    ]
    assert main._parse_hud(text)["lines"] == ["a"]


def test_on_hud_forwards_to_transport(monkeypatch):
    transport = _FakeTransport()
    monkeypatch.setattr(main, "webrtc_server", transport)
    main._on_hud(pa.array(['{"lines": ["ready"]}']))
    main._on_hud(pa.array(["broken"]))
    assert transport.hud == [
        {"type": "hud", "lines": ["ready"], "recording": False, "haptic": None}
    ]


# control input


@pytest.mark.parametrize(
    ("value", "command"),
    [
        (pa.array(["quit"]), "quit"),
        (pa.array([" quit\n"]), "quit"),
        (pa.array(["reset"]), "reset"),
        (pa.array(["quit"], type=pa.large_string()), "quit"),
        (pa.array([], type=pa.string()), None),
        (pa.array([None], type=pa.string()), None),
    ],
)
def test_control_command(value, command):
    assert main._control_command(value) == command


def _run_dora_loop(monkeypatch, events, transport=None):
    fake = _FakeNode(events)
    transport = transport or _FakeTransport()
    monkeypatch.setattr(main, "node", fake)
    monkeypatch.setattr(main, "webrtc_server", transport)
    monkeypatch.setattr(main, "server", None)
    monkeypatch.setattr(main, "_running", True)
    monkeypatch.setattr(main, "_state", main._ConnectionState())
    asyncio.run(asyncio.wait_for(main._main_dora(), 5.0))
    return fake, transport


def test_control_quit_ends_the_dora_loop(monkeypatch):
    after_quit = _input("hud", pa.array(['{"lines": ["too late"]}']))
    fake, transport = _run_dora_loop(
        monkeypatch,
        [
            _input("hud", pa.array(['{"lines": ["idle"]}'])),
            # Not this node's command: the loop carries on.
            _input("control", pa.array(["reset"])),
            _input("tick", pa.array([], type=pa.null())),
            _input("control", pa.array(["quit"])),
            after_quit,
        ],
    )
    assert [message["lines"] for message in transport.hud] == [["idle"]]
    assert after_quit not in fake.consumed
    # The loop's finally ran: the rest of the node is told to stop.
    assert not main._running


def test_quit_on_control_still_sends_quit_command(monkeypatch):
    # --quit-button's command=quit goes out however the loop ends, the
    # control input included, so a quitter downstream still hears it.
    monkeypatch.setattr(main, "_QUIT_BUTTONS", ("b",))
    fake, _transport = _run_dora_loop(
        monkeypatch, [_input("control", pa.array(["quit"]))]
    )
    assert [output_id for output_id, _value in fake.outputs] == ["command"]


def test_control_quit_stops_web_server_and_peers(monkeypatch):
    # The hosted mode end to end, with a real uvicorn and a real WebRTC
    # server: quit on the control input must close the peers and bring
    # the Web server down, so the process can exit.
    monkeypatch.setattr(main, "_running", True)
    monkeypatch.setattr(main, "server", None)
    monkeypatch.setattr(
        main,
        "args",
        types.SimpleNamespace(
            host="127.0.0.1", port=0, tls_key_file=None, tls_certificate_file=None
        ),
    )
    monkeypatch.setattr(main, "_state", main._ConnectionState())
    transport = webrtc.WebRTCServer(
        on_frame=lambda payload: None, on_session_start=None
    )
    closed = []
    original_close = transport.close

    async def close():
        closed.append(True)
        await original_close()

    transport.close = close
    monkeypatch.setattr(main, "webrtc_server", transport)
    fake = _FakeNode(
        [_input("control", pa.array(["quit"]))],
        gate=lambda: main.server is not None and main.server.started,
    )
    monkeypatch.setattr(main, "node", fake)

    started = time.monotonic()
    asyncio.run(asyncio.wait_for(main._main_hosted(), 10.0))
    assert closed == [True]
    assert main.server.should_exit
    assert time.monotonic() - started < 10.0


# One Euro parameters and ICE servers from the environment


async def _no_session():
    pass


@pytest.fixture
def main_globals(monkeypatch, tmp_path):
    # main() configures module globals; put every one of them back.
    for name in (
        "args",
        "node",
        "_MAX_LINEAR_SPEED",
        "_MAX_ANGULAR_SPEED",
        "_POSE_TIMEOUT",
        "_ONE_EURO_MIN_CUTOFF",
        "_ONE_EURO_BETA",
        "_CALIBRATION_ENABLED",
        "_ICE_SERVERS",
        "_NECK_PIVOT_FILE",
        "_QUIT_BUTTONS",
        "_FRAME_OFFSET_CELL",
        "_NECK_PIVOT_OFFSET",
        "_SMOOTHER_RESET_GAP",
        "_PIVOT_MODE",
        "_PAIRING_TOKEN",
        "_quest_log",
    ):
        monkeypatch.setattr(main, name, getattr(main, name))
    monkeypatch.setattr(video, "_view_configuration", video.view_configuration())
    for name in (
        "ONE_EURO_MIN_CUTOFF",
        "ONE_EURO_BETA",
        "ICE_SERVERS",
        "VIEW_CONFIGURATION_FILE",
        "CALIBRATION",
        "QUIT_BUTTON",
        "OFFER",
        "ANSWER_PORT",
        "TLS_CERTIFICATE_FILE",
        "TLS_KEY_FILE",
        "SMOOTHER_RESET_GAP",
        "PIVOT_MODE",
        "PAIRING_TOKEN",
        "QUEST_LOG",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NECK_PIVOT_FILE", str(tmp_path / "neck_pivot.yaml"))
    # Stop main() at the point it would join the dataflow.
    monkeypatch.setattr(main.dora, "Node", lambda: object())
    monkeypatch.setattr(main, "_main_async", _no_session)
    monkeypatch.setattr(
        sys, "argv", ["dora-openarm-webxr", "--offer", "v=0", "--answer-port", "1"]
    )


def test_one_euro_defaults(main_globals):
    main.main()
    assert main._ONE_EURO_MIN_CUTOFF == 2.0
    assert main._ONE_EURO_BETA == 0.04
    smoother = main._ConnectionState().smoothers["right"]
    assert (smoother.min_cutoff, smoother.beta, smoother.d_cutoff) == (2.0, 0.04, 1.5)


def test_one_euro_from_environment(main_globals, monkeypatch):
    monkeypatch.setenv("ONE_EURO_MIN_CUTOFF", "5.5")
    monkeypatch.setenv("ONE_EURO_BETA", "0")
    main.main()
    for smoother in main._ConnectionState().smoothers.values():
        assert smoother.min_cutoff == 5.5
        assert smoother.beta == 0.0


def test_one_euro_command_line_wins(main_globals, monkeypatch):
    monkeypatch.setenv("ONE_EURO_MIN_CUTOFF", "5.5")
    sys.argv += ["--one-euro-min-cutoff", "3", "--one-euro-beta", "0.1"]
    main.main()
    assert (main._ONE_EURO_MIN_CUTOFF, main._ONE_EURO_BETA) == (3.0, 0.1)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ONE_EURO_MIN_CUTOFF", "0"),
        ("ONE_EURO_MIN_CUTOFF", "-1"),
        ("ONE_EURO_MIN_CUTOFF", "nan"),
        ("ONE_EURO_MIN_CUTOFF", "fast"),
        ("ONE_EURO_BETA", "-0.1"),
        ("ONE_EURO_BETA", "inf"),
    ],
)
def test_one_euro_rejects_bad_environment(
    main_globals, monkeypatch, capsys, name, value
):
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit):
        main.main()
    assert "one-euro" in capsys.readouterr().err


def test_ice_servers_empty_list_means_host_candidates_only(main_globals, monkeypatch):
    # What DESIGN.md asks the dataflow to set on a LAN: no STUN at all,
    # not the public default an unset variable gives.
    monkeypatch.setenv("ICE_SERVERS", "[]")
    main.main()
    assert main._ICE_SERVERS == []
    captured = []
    real_configuration = webrtc.RTCConfiguration

    def capture(iceServers):  # aiortc's own keyword
        captured.append(iceServers)
        return real_configuration(iceServers=iceServers)

    monkeypatch.setattr(webrtc, "RTCConfiguration", capture)
    server = webrtc.WebRTCServer(
        on_frame=None, on_session_start=None, ice_servers=main._ICE_SERVERS
    )

    async def create_and_close():
        await server._create_peer().close()

    asyncio.run(create_and_close())
    assert captured == [[]]


# Pairing token, smoother reset gap and pivot mode (PATCHES.md 8, 12, 13)


def test_session_options_defaults(main_globals):
    main.main()
    assert main._PAIRING_TOKEN is None
    assert main._SMOOTHER_RESET_GAP == 0.1
    assert main._PIVOT_MODE == "live"


def test_session_options_from_environment(main_globals, monkeypatch):
    # The token as oqt reads it from tls/pairing_token, newline and all.
    monkeypatch.setenv("PAIRING_TOKEN", "  AbC-123_x\n")
    monkeypatch.setenv("SMOOTHER_RESET_GAP", "0.25")
    monkeypatch.setenv("PIVOT_MODE", "latched")
    main.main()
    assert main._PAIRING_TOKEN == "AbC-123_x"
    assert main._SMOOTHER_RESET_GAP == 0.25
    assert main._PIVOT_MODE == "latched"


def test_empty_pairing_token_means_unset(main_globals, monkeypatch):
    monkeypatch.setenv("PAIRING_TOKEN", " ")
    main.main()
    assert main._PAIRING_TOKEN is None


def test_the_headset_log_comes_from_the_environment(
    main_globals, monkeypatch, tmp_path, capsys
):
    # `oqt run` sets QUEST_LOG to runs/<name>/quest.log; without it the node
    # has no sink at all and the page is told to keep its buffer.
    main.main()
    assert main._quest_log is None
    assert "quest log: off" in capsys.readouterr().out

    path = tmp_path / "runs" / "session" / "quest.log"
    monkeypatch.setenv("QUEST_LOG", str(path))
    main.main()
    assert main._quest_log is not None and main._quest_log.path == path
    # The directory is made at startup, but the file only when there is
    # something to write: a node nobody connected to leaves nothing behind.
    assert path.parent.is_dir() and not path.exists()
    assert str(path) in capsys.readouterr().out


def test_a_headset_log_that_cannot_be_opened_is_not_fatal(
    main_globals, monkeypatch, tmp_path
):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    monkeypatch.setenv("QUEST_LOG", str(blocked / "quest.log"))
    # Teleoperation does not depend on the log; the node says so and runs.
    main.main()
    assert main._quest_log is None


@pytest.mark.parametrize(
    ("name", "value", "option"),
    [
        ("SMOOTHER_RESET_GAP", "0", "smoother-reset-gap"),
        ("SMOOTHER_RESET_GAP", "-0.1", "smoother-reset-gap"),
        ("SMOOTHER_RESET_GAP", "nan", "smoother-reset-gap"),
        ("SMOOTHER_RESET_GAP", "soon", "smoother-reset-gap"),
        ("PIVOT_MODE", "fixed", "PIVOT_MODE"),
    ],
)
def test_session_options_reject_bad_environment(
    main_globals, monkeypatch, capsys, name, value, option
):
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit):
        main.main()
    assert option in capsys.readouterr().err


# HUD replay on reconnect


async def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError
        await asyncio.sleep(0.02)


def test_hud_reaches_browsers_and_is_replayed_without_haptic(monkeypatch):
    # The HUD goes to the active browser (PATCHES.md 9), and a browser that
    # becomes active later gets the last one replayed.
    monkeypatch.setattr(webrtc, "ICE_SERVERS", [])
    video.reset()

    async def browser(server):
        messages = []
        channels = []
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        pc.createDataChannel("xr", ordered=False, maxRetransmits=0)

        @pc.on("datachannel")
        def on_datachannel(channel):
            channels.append(channel)

            @channel.on("message")
            def on_message(message):
                messages.append(json.loads(message))

        for _ in range(webrtc.VIDEO_TRANSCEIVERS):
            pc.addTransceiver("video", direction="recvonly")
        await pc.setLocalDescription(await pc.createOffer())
        answer = await server.answer(pc.localDescription.sdp)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))
        await _wait_for(lambda: messages)

        def start():
            channels[0].send(json.dumps({"type": "session-start"}))

        return pc, messages, start

    async def scenario():
        server = webrtc.WebRTCServer(on_frame=None, on_session_start=lambda: None)
        first, first_messages, start_first = await browser(server)
        start_first()
        await _wait_for(lambda: server._active_peer is not None)
        hud = {"type": "hud", "lines": ["REC"], "recording": True, "haptic": "both"}
        server.send_hud(hud)
        await _wait_for(lambda: len(first_messages) >= 2)
        assert first_messages[1] == hud

        # A page that reconnects gets the HUD once its session starts, but
        # not the pulse: that event is over.
        second, second_messages, start_second = await browser(server)
        await first.close()
        await _wait_for(lambda: server._active_peer is None)
        assert [message["type"] for message in second_messages] == ["configuration"]
        start_second()
        await _wait_for(lambda: len(second_messages) >= 2)
        assert [message["type"] for message in second_messages] == [
            "configuration",
            "hud",
        ]
        assert second_messages[1] == {**hud, "haptic": None}
        await second.close()
        await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 20.0))
    video.reset()


# openarm-quest-teleop view configurations


@pytest.mark.parametrize(
    ("name", "eyes", "mode", "lock"),
    [
        (
            "view_mono.yaml",
            ["right", "main", "arm_left", "arm_right"],
            "immersive-ar",
            "room",
        ),
        (
            "view_stereo.yaml",
            ["left", "right", "main", "arm_left", "arm_right"],
            "immersive-vr",
            "head",
        ),
    ],
)
def test_project_view_configurations(monkeypatch, name, eyes, mode, lock):
    path = PROJECT_CONFIG / name
    if not path.exists():
        pytest.skip(f"not vendored into openarm-quest-teleop: {path}")
    monkeypatch.setattr(video, "_view_configuration", video.view_configuration())
    video.configure(types.SimpleNamespace(view_configuration_file=path))
    configuration = video.view_configuration()
    # A file that fails to parse leaves the default in place silently.
    assert configuration is not video.DEFAULT_VIEW_CONFIGURATION
    assert video.eyes() == eyes
    assert configuration["session"]["mode"] == mode
    assert configuration["panel"]["lock"] == lock
    assert len(configuration["pose"]["frame_offset"]) == 3
    assert len(configuration["pose"]["neck_pivot_offset"]) == 3
    if name == "view_stereo.yaml":
        # The sim head cameras: fovy 52 at 1280x720, parallel, no offsets.
        assert configuration["camera"] == {"horizontal_fov": 81.9, "vertical_fov": 52}
        assert configuration["stereo"] == {"vertical_align": 0, "convergence": 0}


# The page


def test_everything_the_page_imports_is_packaged():
    """Patch 15: the 3D view imports the vendored three.js out of static/three/.

    setuptools ships only what package-data lists, and each glob is one level
    deep, so `*.js` does not carry static/three/. A wheel built without the
    `three/*` entry serves a page whose imports 404 -- which nothing running
    from the source tree can notice, because the editable install this
    repository uses sees the files on disk either way.
    """
    package = pathlib.Path(main.__file__).parent
    pyproject = package.parents[1] / "pyproject.toml"
    patterns = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"][
        "setuptools"
    ]["package-data"]["dora_openarm_webxr.static"]
    shipped = {path.resolve() for pattern in patterns for path in STATIC.glob(pattern)}
    missing = sorted(
        str(path.relative_to(package))
        for path in STATIC.rglob("*")
        if path.is_file() and path.resolve() not in shipped
    )
    assert not missing, f"not in package-data: {missing}"
    # Named, because these are the files the page fails on and the licence that
    # has to travel with them.
    assert (STATIC / "three" / "three.module.min.js").resolve() in shipped
    assert (STATIC / "three" / "LICENSE").resolve() in shipped


def test_the_vendored_three_js_is_the_one_its_provenance_claims():
    """Patch 15 vendors a third-party library into the page, so three things have to
    agree: the files on disk, the per-file SHA-256 in static/three/README.md, and the
    repository's NOTICE (version, licence, location). An update that drops new files
    in and forgets one of the other two is what this catches -- nobody notices from
    the page, which works either way.
    """
    directory = STATIC / "three"
    readme = (directory / "README.md").read_text(encoding="utf-8")
    recorded = {
        name: digest
        for digest, name in re.findall(
            r"^([0-9a-f]{64})\s+(\S+)$", readme, re.MULTILINE
        )
    }
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "README.md"
    }
    assert actual == recorded, "static/three/README.md does not describe what is there"

    version = re.search(r"npm integrity|Version: \*\*([0-9.]+)\*\*", readme)
    assert version and version.group(1), "static/three/README.md records no version"
    notice = PROJECT_CONFIG.parent / "NOTICE"
    if not notice.is_file():
        pytest.skip("not vendored in openarm-quest-teleop")
    text = notice.read_text(encoding="utf-8")
    assert "three.js" in text and "MIT License" in text
    assert f"three@{version.group(1)}" in text, (
        f"NOTICE does not name the vendored three {version.group(1)}"
    )
    assert "static/three/" in text


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_static_modules_parse(tmp_path):
    # Checked as .mjs: `node --check` on a .js file with import statements
    # reports success without parsing it (Node 24).
    failures = []
    for source in sorted(STATIC.glob("*.js")):
        copy = tmp_path / f"{source.stem}.mjs"
        copy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        result = subprocess.run(
            ["node", "--check", str(copy)], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            failures.append(f"{source.name}: {result.stderr}")
    assert not failures, "\n".join(failures)


def test_page_has_no_default_stun_server():
    # Patch 1: the page offers with no ICE servers unless ?ice= names some.
    source = (STATIC / "connection.js").read_text(encoding="utf-8")
    assert "stun.cloudflare.com" not in source
    assert 'getAll("ice")' in source
