# openarm-quest-teleop addition to the vendored dora-openarm-webxr tests
# (Apache-2.0, like the rest of this directory).
#
# Tests for PATCHES.md 8-11: the pairing token on POST /offer, one session
# at a time with an explicit takeover, the HUD going to the active peer
# only, peers that cannot be answered or never connect being dropped, and
# the page ending sessions the node will not serve.

import asyncio
import json
import pathlib
import shutil
import subprocess
import textwrap
import time

import pytest
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from dora_openarm_webxr import main, video, webrtc

STATIC = pathlib.Path(main.__file__).parent / "static"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(webrtc, "ICE_SERVERS", [])
    video.reset()
    yield
    video.reset()


async def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError
        await asyncio.sleep(0.02)


# POST /offer and the pairing token (PATCHES.md 8, 10)


class _AnsweringTransport:
    """Stands in for WebRTCServer: records offers, answers or raises."""

    def __init__(self, error=None):
        self.offers = []
        self.error = error

    async def answer(self, sdp):
        self.offers.append(sdp)
        if self.error is not None:
            raise self.error
        return "answer-sdp"


async def _post_offer(body):
    """POST ``body`` (bytes) to the node's /offer through its ASGI app."""
    requests = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        return requests.pop(0) if requests else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/offer",
        "raw_path": b"/offer",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"host", b"node")],
        "client": ("192.0.2.7", 50000),
        "server": ("127.0.0.1", 8443),
    }
    await main.app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    payload = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return status, json.loads(payload)


def _post(body):
    if not isinstance(body, bytes):
        body = json.dumps(body).encode()
    return asyncio.run(_post_offer(body))


@pytest.fixture
def paired(monkeypatch):
    transport = _AnsweringTransport()
    monkeypatch.setattr(main, "webrtc_server", transport)
    monkeypatch.setattr(main, "_PAIRING_TOKEN", "s3cret-token", raising=False)
    return transport


@pytest.mark.parametrize(
    "body",
    [
        {"sdp": "v=0"},
        {"sdp": "v=0", "token": "wrong"},
        {"sdp": "v=0", "token": "s3cret-token-but-longer"},
        {"sdp": "v=0", "token": None},
        {"sdp": "v=0", "token": ["s3cret-token"]},
        {"sdp": "v=0", "token": "s3crét"},
        b"not json",
        b"\xff\xfe",
        ["s3cret-token"],
    ],
)
def test_offer_without_the_pairing_token_is_refused(paired, body, capsys):
    # SEC-1: refused before any peer exists, whatever the body looks like.
    status, payload = _post(body)
    assert status == 403
    assert "pairing token" in payload["error"]
    assert paired.offers == []
    assert "refused an offer from 192.0.2.7" in capsys.readouterr().out


def test_offer_with_the_pairing_token_is_answered(paired):
    status, payload = _post({"sdp": "v=0 offer", "token": "s3cret-token"})
    assert (status, payload) == (200, {"sdp": "answer-sdp", "type": "answer"})
    assert paired.offers == ["v=0 offer"]


def test_unset_pairing_token_accepts_and_warns_once(monkeypatch, capsys):
    transport = _AnsweringTransport()
    monkeypatch.setattr(main, "webrtc_server", transport)
    monkeypatch.setattr(main, "_PAIRING_TOKEN", None, raising=False)
    monkeypatch.setattr(main, "_warned_unpaired", False, raising=False)
    assert _post({"sdp": "one"})[0] == 200
    assert _post({"sdp": "two", "token": "anything"})[0] == 200
    assert transport.offers == ["one", "two"]
    assert capsys.readouterr().out.count("PAIRING_TOKEN is not set") == 1


@pytest.mark.parametrize(
    "body",
    [
        {"token": "s3cret-token"},
        {"sdp": 42, "token": "s3cret-token"},
    ],
)
def test_malformed_offer_is_a_400(paired, body):
    status, _payload = _post(body)
    assert status == 400
    assert paired.offers == []


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (lambda: ValueError("SDP does not have 'v=0' line"), 400),
        (lambda: AssertionError(), 400),
        (lambda: KeyError("setup"), 400),
        (lambda: webrtc.PendingPeerLimitError("4 peers are still negotiating"), 503),
    ],
)
def test_answer_errors_map_to_client_errors(monkeypatch, error, status):
    # NET-2: aiortc reports a bad SDP with all sorts of exceptions; none of
    # them is the node's fault, and none may become a 500.
    monkeypatch.setattr(main, "webrtc_server", _AnsweringTransport(error()))
    monkeypatch.setattr(main, "_PAIRING_TOKEN", "t", raising=False)
    assert _post({"sdp": "v=0", "token": "t"})[0] == status


# One session at a time, and the HUD to the active peer (PATCHES.md 9)


class _Browser:
    """An in-process stand-in for connection.js, like the upstream tests'."""

    def __init__(self):
        self.messages = []
        self.control = None
        self.control_closed = False
        self.sequence = 0

    async def connect(self, server):
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.xr = self.pc.createDataChannel("xr", ordered=False, maxRetransmits=0)

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            if channel.label != "control":
                return
            self.control = channel

            @channel.on("message")
            def on_message(message):
                self.messages.append(json.loads(message))

            @channel.on("close")
            def on_close():
                self.control_closed = True

        for _ in range(webrtc.VIDEO_TRANSCEIVERS):
            self.pc.addTransceiver("video", direction="recvonly")
        await self.pc.setLocalDescription(await self.pc.createOffer())
        answer = await server.answer(self.pc.localDescription.sdp)
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer, type="answer")
        )
        await _wait_for(lambda: self.types() == ["configuration"])
        await _wait_for(lambda: self.xr.readyState == "open")

    def types(self):
        return [message["type"] for message in self.messages]

    def start(self):
        self.control.send(json.dumps({"type": "session-start"}))

    def frame(self, who):
        if self.xr.readyState != "open":
            return
        self.sequence += 1
        self.xr.send(
            json.dumps({"type": "frame", "sequence": self.sequence, "who": who})
        )


class _Node:
    def __init__(self):
        self.frames = []
        self.events = []

    def server(self, **options):
        return webrtc.WebRTCServer(
            on_frame=lambda payload: self.frames.append(payload["who"]),
            on_session_start=lambda: self.events.append("start"),
            on_session_end=lambda: self.events.append("end"),
            **options,
        )


HUD = {
    "type": "hud",
    "lines": ["REC 00:03 ep 1", "R:FOLLOW"],
    "recording": True,
    "haptic": "right",
}


async def _send_frames(browser, who, stop):
    while not stop.is_set():
        browser.frame(who)
        await asyncio.sleep(0.02)


def test_session_start_is_refused_while_the_active_peer_sends_frames():
    # SEC-1: a second client's session-start must neither take the robot
    # nor see the operator's HUD while the operator's frames flow.
    asyncio.run(asyncio.wait_for(_run_refused(), 30.0))


async def _run_refused():
    node = _Node()
    server = node.server()
    operator, other = _Browser(), _Browser()
    stop = asyncio.Event()
    try:
        await operator.connect(server)
        operator.start()
        await _wait_for(lambda: node.events == ["start"])
        sender = asyncio.ensure_future(_send_frames(operator, "operator", stop))
        await _wait_for(lambda: node.frames)

        await other.connect(server)
        server.send_hud(HUD)
        await _wait_for(lambda: "hud" in operator.types())
        assert operator.messages[-1] == HUD

        other.start()
        await _wait_for(lambda: "session-refused" in other.types())
        assert other.messages[-1] == {
            "type": "session-refused",
            "reason": "another headset is active",
        }
        node.frames.clear()
        for _ in range(10):
            other.frame("other")
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)
        assert set(node.frames) == {"operator"}
        assert node.events == ["start"]
        # The refused page never got the operator's HUD, now or on refusal.
        assert other.types() == ["configuration", "session-refused"]
        assert not operator.control_closed
        stop.set()
        await sender
    finally:
        stop.set()
        await operator.pc.close()
        await other.pc.close()
        await server.close()


def test_idle_active_peer_is_taken_over_and_told():
    asyncio.run(asyncio.wait_for(_run_takeover(), 30.0))


async def _run_takeover():
    node = _Node()
    server = node.server()
    operator, other = _Browser(), _Browser()
    try:
        await operator.connect(server)
        operator.start()
        await _wait_for(lambda: node.events == ["start"])
        operator.frame("operator")
        await _wait_for(lambda: node.frames)
        server.send_hud(HUD)
        await _wait_for(lambda: "hud" in operator.types())

        # The operator's headset goes quiet (asleep, its connection only
        # "disconnected"); after TAKEOVER_IDLE another may take over.
        await other.connect(server)
        await asyncio.sleep(1.2)
        other.start()
        await _wait_for(lambda: node.events == ["start", "end", "start"])
        # Frames the old page still sends before it is closed go nowhere.
        node.frames.clear()
        operator.frame("operator")
        other.frame("other")
        await _wait_for(lambda: node.frames)
        await asyncio.sleep(0.1)
        assert node.frames == ["other"]
        await _wait_for(lambda: "session-ended" in operator.types())
        assert operator.messages[-1] == {
            "type": "session-ended",
            "reason": "taken-over",
        }
        # The newcomer gets the HUD the session left off with, minus the
        # pulse, which belonged to an event before it arrived.
        await _wait_for(lambda: "hud" in other.types())
        assert other.messages[-1] == {**HUD, "haptic": None}
        # The old peer is closed, so its page leaves immersive mode.
        await _wait_for(lambda: operator.control_closed, timeout=5.0)
        # Its late close changes nothing for the session that replaced it.
        assert node.events == ["start", "end", "start"]
    finally:
        await operator.pc.close()
        await other.pc.close()
        await server.close()


class _StandInChannel:
    readyState = "open"

    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(json.loads(message))


class _StandInPeer:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _ClockedServer:
    """A WebRTCServer on a fake clock, with stand-in peers and channels."""

    START = json.dumps({"type": "session-start"})
    FRAME = json.dumps({"type": "frame", "who": "operator"})

    def __init__(self):
        self.node = _Node()
        self.server = self.node.server()
        self.clock = [100.0]
        self.server._clock = lambda: self.clock[0]
        self.peers = {}

    def peer(self, name):
        pc = _StandInPeer()
        self.server._pcs.add(pc)
        self.server._controls[pc] = _StandInChannel()
        self.peers[name] = pc
        return pc

    def start(self, name):
        self.server._handle_control_message(self.START, self.peers[name])

    def frame(self, name):
        self.server._handle_frame_message(self.FRAME, self.peers[name])

    def told(self, name):
        return [m["type"] for m in self.server._controls[self.peers[name]].sent]


def test_session_that_has_not_sent_a_frame_yet_is_not_taken_over():
    # SEC-1 residual (P4): the takeover idle runs from the later of the
    # active peer's session-start and its last frame, so the window between
    # a session starting and its first frame -- a Quest session that is not
    # "visible" yet -- is not open to another peer.
    async def scenario():
        rig = _ClockedServer()
        operator, intruder = rig.peer("operator"), rig.peer("intruder")
        rig.start("operator")
        for elapsed in (0.0, 0.5, webrtc.TAKEOVER_IDLE):
            rig.clock[0] = 100.0 + elapsed
            rig.start("intruder")
            assert rig.server._active_peer is operator, elapsed
        assert rig.told("intruder") == ["session-refused"] * 3
        assert rig.told("operator") == []
        assert rig.node.events == ["start"]
        rig.frame("intruder")
        assert rig.node.frames == []
        await asyncio.sleep(webrtc.TAKEOVER_CLOSE_GRACE + 0.1)
        assert not operator.closed and not intruder.closed
        # The refused attempts do not count as the operator's use: past
        # TAKEOVER_IDLE since its start, with no frame, the next one wins.
        rig.clock[0] = 100.0 + webrtc.TAKEOVER_IDLE + 0.01
        rig.start("intruder")
        assert rig.server._active_peer is intruder
        await rig.server.close()

    asyncio.run(scenario())


def test_session_idle_since_its_start_and_last_frame_is_taken_over():
    # The sleeping-headset recovery still works: once the active peer has
    # neither started its session nor sent a frame for TAKEOVER_IDLE, a
    # session-start takes over, frames or not.
    async def scenario():
        rig = _ClockedServer()
        operator, intruder = rig.peer("operator"), rig.peer("intruder")
        rig.start("operator")
        rig.clock[0] += webrtc.TAKEOVER_IDLE + 0.01
        rig.start("intruder")
        assert rig.server._active_peer is intruder
        assert rig.node.events == ["start", "end", "start"]
        assert rig.told("operator") == ["session-ended"]
        assert rig.told("intruder") == []
        await _wait_for(lambda: operator.closed, timeout=5.0)
        assert not intruder.closed

        # A frame late in the window restarts the idle time: 0.9 s after
        # the start and 0.6 s before the next attempt, it is still in use.
        rig.clock[0] += 0.9
        rig.frame("intruder")
        rig.clock[0] += 0.6
        latecomer = rig.peer("latecomer")
        rig.start("latecomer")
        assert rig.server._active_peer is intruder
        assert rig.told("latecomer") == ["session-refused"]
        rig.clock[0] += webrtc.TAKEOVER_IDLE - 0.6 + 0.01
        rig.start("latecomer")
        assert rig.server._active_peer is latecomer
        await rig.server.close()

    asyncio.run(scenario())


def test_session_start_is_refused_before_the_active_peer_sends_a_frame():
    # The same with real aiortc peers (reverify/sec1/hijack_now.py (d)): a
    # session that has started but not sent a frame is refused to another
    # peer until it has been idle for takeover_idle.
    asyncio.run(asyncio.wait_for(_run_frameless_refusal(), 30.0))


async def _run_frameless_refusal():
    node = _Node()
    server = node.server(takeover_idle=0.5)
    operator, other = _Browser(), _Browser()
    try:
        await operator.connect(server)
        await other.connect(server)
        operator.start()
        await _wait_for(lambda: node.events == ["start"])
        other.start()
        await _wait_for(lambda: "session-refused" in other.types())
        assert node.events == ["start"]
        assert operator.types() == ["configuration"]

        # Idle for longer than takeover_idle since its start: taken over.
        await asyncio.sleep(0.7)
        other.start()
        await _wait_for(lambda: node.events == ["start", "end", "start"])
        await _wait_for(lambda: "session-ended" in operator.types())
        other.frame("other")
        await _wait_for(lambda: node.frames == ["other"])
    finally:
        await operator.pc.close()
        await other.pc.close()
        await server.close()


def test_hud_and_calibration_results_go_to_the_active_peer_only():
    asyncio.run(asyncio.wait_for(_run_hud_routing(), 30.0))


async def _run_hud_routing():
    node = _Node()
    server = node.server()
    idle, active = _Browser(), _Browser()
    try:
        await idle.connect(server)
        await active.connect(server)
        # With no session yet, nobody is told anything, but the HUD is kept.
        server.send_hud(HUD)
        await asyncio.sleep(0.2)
        assert idle.types() == active.types() == ["configuration"]

        active.start()
        await _wait_for(lambda: "hud" in active.types())
        assert active.messages[-1] == {**HUD, "haptic": None}
        server.send_hud({**HUD, "lines": ["READY"]})
        server.send_control({"type": "calibration-result", "accepted": False})
        await _wait_for(lambda: "calibration-result" in active.types())
        assert active.types() == ["configuration", "hud", "hud", "calibration-result"]
        await asyncio.sleep(0.2)
        assert idle.types() == ["configuration"]
    finally:
        await idle.pc.close()
        await active.pc.close()
        await server.close()


# Peers that cannot be answered or never connect (PATCHES.md 10)


async def _offer_sdp():
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    pc.createDataChannel("xr")
    # As many video slots as the page offers: the node adds one track per
    # camera it serves, and aiortc cannot answer a track the offer has no slot for.
    for _ in range(webrtc.VIDEO_TRANSCEIVERS):
        pc.addTransceiver("video", direction="recvonly")
    await pc.setLocalDescription(await pc.createOffer())
    return pc, pc.localDescription.sdp


def test_unanswerable_offer_leaves_no_peer():
    async def scenario():
        server = webrtc.WebRTCServer(on_frame=None, on_session_start=None)
        for sdp in ("v=0\r\n", "", "hello"):
            with pytest.raises(Exception):  # noqa: B017 -- aiortc's choice
                await server.answer(sdp)
        assert not server._pcs
        await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 20.0))


def test_peer_that_never_connects_is_closed():
    # The browser posts an offer and goes away before applying the answer.
    async def scenario():
        server = webrtc.WebRTCServer(
            on_frame=None, on_session_start=None, connect_timeout=0.5
        )
        browser, sdp = await _offer_sdp()
        try:
            await server.answer(sdp)
            assert len(server._pcs) == 1
            await browser.close()
            await _wait_for(lambda: not server._pcs, timeout=5.0)
        finally:
            await browser.close()
            await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 20.0))


def test_pending_peers_are_capped():
    async def scenario():
        server = webrtc.WebRTCServer(
            on_frame=None, on_session_start=None, max_pending_peers=2
        )
        browsers = []
        try:
            for _ in range(2):
                browser, sdp = await _offer_sdp()
                browsers.append(browser)
                await server.answer(sdp)
            browser, sdp = await _offer_sdp()
            browsers.append(browser)
            with pytest.raises(webrtc.PendingPeerLimitError):
                await server.answer(sdp)
            assert len(server._pcs) == 2
        finally:
            for browser in browsers:
                await browser.close()
            await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 20.0))


def test_connected_peer_does_not_count_as_pending():
    # The cap must not lock out the next reload of a headset that is
    # connected and working.
    async def scenario():
        server = webrtc.WebRTCServer(
            on_frame=None, on_session_start=lambda: None, max_pending_peers=1
        )
        first, second = _Browser(), _Browser()
        try:
            await first.connect(server)
            await _wait_for(lambda: not server._pending)
            await second.connect(server)
        finally:
            await first.pc.close()
            await second.pc.close()
            await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 20.0))


# The page (PATCHES.md 8, 9, 11), run in Node.js against stand-in modules

_STUB_MODULES = {
    "connection.js": """
        export const log = [];
        export const connections = [];
        export async function connect() {
          const handlers = {};
          const c = {
            configuration: { view: "none", session: { mode: "immersive-ar" }, panel: {} },
            calibration: { enabled: false }, eyes: [], tracks: {},
            sendFrame() {},
            sendControl(m) { log.push(`control:${m.type}`); },
            onCalibrationResult() {},
            onHud(h) { handlers.hud = h; },
            onSessionNotice(h) { handlers.notice = h; },
            onClose(h) { handlers.close = h; },
            close() { log.push("close"); },
            lose() { log.push("lost"); handlers.close && handlers.close(); },
            hud(m) { handlers.hud(m); },
            notice(m) { handlers.notice(m); },
          };
          connections.push(c);
          log.push("connected");
          return c;
        }
    """,
    "hud.js": """
        export const hudLines = [];
        export function createHudPanel() {
          return { set(m) { hudLines.push(m.lines || []); }, attach() {}, render() {} };
        }
    """,
    "instructions.js": "export function createInstructionPanel() { return {}; }",
    "panel.js": """
        export function createCameraPanel() { return null; }
        export function createCameraPanels() { return null; }
    """,
    "stereo.js": "export function createStereoPanel() { return null; }",
}

_PAGE_PRELUDE = """
    let clickHandler = null;
    const status = { innerText: "" };
    globalThis.document = {
      getElementById(id) {
        return id === "start" ? { addEventListener(_, h) { clickHandler = h; } } : status;
      },
      createElement() { return { getContext() { return {}; } }; },
    };
    globalThis.XRWebGLLayer = class {};
    globalThis.alert = () => {};
    const sessions = [];
    let resolveSession = null;
    function makeSession() {
      const listeners = {};
      const session = {
        ended: false, visibilityState: "visible", inputSources: [],
        addEventListener(type, handler) { listeners[type] = handler; },
        updateRenderState() {},
        requestReferenceSpace() { return Promise.resolve({ addEventListener() {} }); },
        requestAnimationFrame() {},
        end() {
          session.ended = true;
          setTimeout(() => listeners.end && listeners.end(), 0);
          return Promise.resolve();
        },
      };
      sessions.push(session);
      return session;
    }
    Object.defineProperty(globalThis, "navigator", { configurable: true, value: {
      xr: {
        isSessionSupported() { return Promise.resolve(true); },
        requestSession() { return new Promise((resolve) => { resolveSession = resolve; }); },
      },
    }});
    const tick = () => new Promise((resolve) => setTimeout(resolve, 20));
    const conn = await import("./connection.js");
    const { hudLines } = await import("./hud.js");
    await import("./ar.js");
    await tick();
"""


def _run_page(tmp_path, scenario):
    for name, source in _STUB_MODULES.items():
        (tmp_path / name).write_text(textwrap.dedent(source), encoding="utf-8")
    shutil.copy(STATIC / "ar.js", tmp_path / "ar.js")
    (tmp_path / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    script = tmp_path / "scenario.mjs"
    script.write_text(
        textwrap.dedent(_PAGE_PRELUDE) + textwrap.dedent(scenario), encoding="utf-8"
    )
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")


@needs_node
def test_page_ends_a_session_that_resolves_after_the_connection_was_lost(tmp_path):
    # JS-3: the node goes away while requestSession is pending (the Quest's
    # permission prompt). The session that then resolves must be ended,
    # a new connection opened, and the stale HUD taken down.
    result = _run_page(
        tmp_path,
        """
        const first = conn.connections[0];
        first.hud({ type: "hud", lines: ["READY ep 0"], haptic: null });
        clickHandler();
        await tick();
        first.lose();
        await tick();
        const hudAfterLoss = hudLines.at(-1);
        resolveSession(makeSession());
        await tick(); await tick();
        console.log(JSON.stringify({
          ended: sessions[0].ended,
          sessionStartSent: conn.log.filter((l) => l === "control:session-start").length,
          reconnected: conn.connections.length,
          hudAfterLoss,
          status: status.innerText,
        }));
        """,
    )
    assert result["ended"] is True
    assert result["sessionStartSent"] == 0
    assert result["reconnected"] == 2
    assert result["hudAfterLoss"] == []
    # The reason outlives the reconnect that replaces the connection state.
    assert result["status"] == (
        "Lost the node while starting the session.\nConnected. Press Start."
    )


@needs_node
@pytest.mark.parametrize(
    ("notice", "expected"),
    [
        (
            {"type": "session-refused", "reason": "another headset is active"},
            "The node refused the session: another headset is active.",
        ),
        (
            {"type": "session-ended", "reason": "taken-over"},
            "Another headset took over the session.",
        ),
    ],
)
def test_page_ends_its_session_when_the_node_will_not_serve_it(
    tmp_path, notice, expected
):
    # SEC-1: the page leaves immersive mode and keeps the reason on the
    # 2D page through the reconnect that follows.
    result = _run_page(
        tmp_path,
        f"""
        const first = conn.connections[0];
        clickHandler();
        await tick();
        resolveSession(makeSession());
        await tick();
        first.notice({json.dumps(notice)});
        await tick(); await tick();
        console.log(JSON.stringify({{
          ended: sessions[0].ended,
          reconnected: conn.connections.length,
          status: status.innerText,
        }}));
        """,
    )
    assert result["ended"] is True
    assert result["reconnected"] == 2
    assert result["status"].startswith(expected)
    assert "Connected. Press Start." in result["status"]


_CONNECTION_PRELUDE = """
    const storage = new Map();
    const posted = [];
    let status = 200;
    globalThis.window = {
      location: { hash: "", search: "" },
      sessionStorage: {
        getItem: (key) => (storage.has(key) ? storage.get(key) : null),
        setItem: (key, value) => storage.set(key, String(value)),
      },
    };
    globalThis.fetch = async (url, options) => {
      posted.push(JSON.parse(options.body));
      return { ok: status === 200, status, json: async () => ({ sdp: "answer" }) };
    };
    class Channel {
      constructor(label) { this.label = label; this.listeners = {}; this.readyState = "open"; }
      addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
      send() {}
    }
    globalThis.RTCPeerConnection = class {
      constructor() { this.listeners = {}; this.iceGatheringState = "complete"; }
      addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
      createDataChannel(label) { return new Channel(label); }
      addTransceiver() {}
      async createOffer() { return { type: "offer", sdp: "offer" }; }
      async setLocalDescription(description) { this.localDescription = description; }
      async setRemoteDescription() {
        const control = new Channel("control");
        for (const handler of this.listeners.datachannel || []) handler({ channel: control });
        const configuration = { type: "configuration", view_configuration: {}, calibration: false, eyes: [] };
        for (const handler of control.listeners.message || []) handler({ data: JSON.stringify(configuration) });
      }
      close() {}
    };
    const { connect } = await import("./connection.js");
    async function attempt() {
      try { await connect(); return "ok"; } catch (error) { return error.message; }
    }
"""


@needs_node
def test_page_sends_the_pairing_token_from_the_fragment_and_keeps_it(tmp_path):
    shutil.copy(STATIC / "connection.js", tmp_path / "connection.js")
    (tmp_path / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    script = tmp_path / "scenario.mjs"
    script.write_text(
        textwrap.dedent(_CONNECTION_PRELUDE)
        + textwrap.dedent(
            """
            window.location.hash = "#k=AbC-123_x";
            const withHash = await attempt();
            window.location.hash = "";  // a reload that lost the fragment
            const reloaded = await attempt();
            status = 403;
            const refused = await attempt();
            console.log(JSON.stringify({ withHash, reloaded, refused, posted }));
            """
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["withHash"] == outcome["reloaded"] == "ok"
    assert [body.get("token") for body in outcome["posted"]] == ["AbC-123_x"] * 3
    assert all(body["sdp"] == "offer" for body in outcome["posted"])
    assert "403" in outcome["refused"] and "pairing token" in outcome["refused"]
