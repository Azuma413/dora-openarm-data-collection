# openarm-quest-teleop addition to the vendored dora-openarm-webxr tests
# (Apache-2.0, like the rest of this directory).
#
# The node half of the 3D scene (PATCHES.md, docs/DESIGN.md X2): the
# ``scene_state`` input, the unreliable ``state`` data channel it is
# forwarded on, the active-peer-only rule, drop-on-backpressure, and what
# the configuration message says about the exported glTF.
#
# The peers here are real aiortc ones over a loopback SCTP association, so
# a rate, a byte count and a latency measured here are measured through the
# same stack a headset talks to.

import asyncio
import gzip
import json
import pathlib
import random
import statistics
import struct
import time

import pyarrow as pa
import pytest
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from dora_openarm_webxr import main, video, webrtc


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(webrtc, "ICE_SERVERS", [])
    # main() resolves this from the view configuration, and a test that ran
    # main() with a 2D view would otherwise leave it False for everything
    # after it. The tests that are ABOUT the gate set it themselves.
    monkeypatch.setattr(main, "_FORWARD_SCENE_STATE", True, raising=False)
    video.reset()
    yield
    video.reset()


async def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError
        await asyncio.sleep(0.005)


class _Browser:
    """An in-process stand-in for the page: the ``xr``, ``control`` and ``state`` channels."""

    def __init__(self):
        self.messages = []  # decoded control messages
        self.states = []  # (arrival monotonic, payload) off the state channel
        self.labels = []  # channel labels, in the order the node offered them
        self.control = None
        self.state = None
        self.pc = None
        self.sequence = 0

    async def connect(self, server, expect_state=True):
        """Connect and wait for the configuration, and for ``state`` if offered.

        ``expect_state=False`` is a view that draws no 3D scene: the node
        offers no such channel there, so waiting for one would only time out.
        """
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.xr = self.pc.createDataChannel("xr", ordered=False, maxRetransmits=0)

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            self.labels.append(channel.label)
            if channel.label == "control":
                self.control = channel

                @channel.on("message")
                def on_control(message):
                    self.messages.append(json.loads(message))
            elif channel.label == "state":
                self.state = channel

                @channel.on("message")
                def on_state(message):
                    self.states.append((time.monotonic(), message))

        for _ in range(webrtc.VIDEO_TRANSCEIVERS):
            self.pc.addTransceiver("video", direction="recvonly")
        await self.pc.setLocalDescription(await self.pc.createOffer())
        answer = await server.answer(self.pc.localDescription.sdp)
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer, type="answer")
        )
        await _wait_for(
            lambda: self.messages and (self.state is not None or not expect_state)
        )
        await _wait_for(lambda: self.xr.readyState == "open")

    def types(self):
        return [message["type"] for message in self.messages]

    def start(self):
        self.control.send(json.dumps({"type": "session-start"}))

    def frame(self, who):
        self.sequence += 1
        self.xr.send(
            json.dumps({"type": "frame", "sequence": self.sequence, "who": who})
        )

    async def close(self):
        if self.pc is not None:
            await self.pc.close()


class _Node:
    """A server with the callbacks the real node passes, and what they saw."""

    def __init__(self):
        self.frames = []
        self.sessions = []

    def server(self, **options):
        return webrtc.WebRTCServer(
            on_frame=lambda payload: self.frames.append(payload.get("who")),
            on_session_start=lambda: self.sessions.append("start"),
            on_session_end=lambda: self.sessions.append("end"),
            **options,
        )


# A scene state the size of the compress cell's: 191 joint values of JSON.
def _state_message(sequence, sent_at=0.0, nq=191):
    return json.dumps(
        {
            "type": "state",
            "v": 1,
            "seq": sequence,
            "t": round(sent_at, 6),
            "flags": 0,
            "qpos": [
                round(-0.5 + 0.0011 * (index + sequence), 4) for index in range(nq)
            ],
        },
        separators=(",", ":"),
    )


async def _push(server, sequences, gap=0.005):
    """Send those scenes the way the sim does: paced, not in one burst.

    The channel holds one frame's worth (STATE_BUFFER_LIMIT) and drops what
    will not fit, so a burst of whole scenes is dropped by design; every
    test that is not about the dropping paces its sends.
    """
    results = []
    for sequence in sequences:
        results.append(server.send_state(_state_message(sequence)))
        await asyncio.sleep(gap)
    return results


# The ``state`` channel itself (X2)


def test_the_node_opens_one_unreliable_state_channel_after_control():
    # The page is offered ``control`` first and ``state`` second: a page --
    # or an upstream test -- that takes the first channel it is handed as
    # the control one was right before this node opened a second of its
    # own, and must stay right. The scene's channel is unordered and never
    # retransmitted, like the browser's own ``xr``.
    async def scenario():
        node = _Node()
        server = node.server()
        page = _Browser()
        try:
            await page.connect(server)
            assert page.labels == ["control", "state"]
            assert page.state.ordered is False
            assert page.state.maxRetransmits == 0
            assert page.state.maxPacketLifeTime is None
            # And it is a channel of its own: the page's "xr" is untouched.
            assert page.xr.label == "xr"
            assert page.xr.readyState == "open"
        finally:
            await page.close()
            await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 30.0))


def test_no_scene_state_is_sent_while_no_session_is_active():
    # A page that is connected but not teleoperating is not drawing the
    # scene either, so nothing goes out -- and that is not a drop.
    async def scenario():
        node = _Node()
        server = node.server()
        page = _Browser()
        try:
            await page.connect(server)
            for sequence in range(10):
                assert server.send_state(_state_message(sequence)) is False
            await asyncio.sleep(0.2)
            assert page.states == []
            assert (server.state_sent, server.state_dropped) == (0, 0)

            # The moment a session starts, the same call lands.
            page.start()
            await _wait_for(lambda: server._active_peer is not None)
            assert server.send_state(_state_message(10)) is True
            await _wait_for(lambda: len(page.states) == 1)
            assert server.state_sent == 1
        finally:
            await page.close()
            await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 30.0))


def test_scene_state_goes_to_the_active_headset_only():
    # Two headsets are connected; only the one driving the robot is shown
    # the scene it is driving.
    async def scenario():
        node = _Node()
        # Shortened so the takeover below does not have to wait out the
        # real idle window; the rule under test is not the window.
        server = node.server(takeover_idle=0.05)
        active, idle = _Browser(), _Browser()
        try:
            await active.connect(server)
            await idle.connect(server)
            active.start()
            await _wait_for(lambda: server._active_peer is not None)
            assert await _push(server, range(5)) == [True] * 5
            await _wait_for(lambda: len(active.states) == 5)
            await asyncio.sleep(0.2)
            assert idle.states == []
            assert server.state_sent == 5

            # A takeover moves the stream with the session: the headset
            # that is now driving is the one shown the scene.
            idle.start()
            await _wait_for(lambda: "session-ended" in active.types())
            drawn = len(active.states)
            assert await _push(server, range(5, 9)) == [True] * 4
            await _wait_for(lambda: len(idle.states) == 4)
            assert len(active.states) == drawn
        finally:
            await active.close()
            await idle.close()
            await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 30.0))


def test_the_state_stream_holds_its_rate_bytes_and_latency():
    # 72 Hz for a second through a real peer: every message arrives, each
    # is the size the codec says, and the node-to-page latency stays in
    # the millisecond range rather than growing with the stream.
    measured = {}

    async def scenario():
        node = _Node()
        server = node.server()
        page = _Browser()
        try:
            await page.connect(server)
            page.start()
            await _wait_for(lambda: server._active_peer is not None)
            hz, seconds = 72.0, 1.0
            count = int(hz * seconds)
            sent_at = {}
            started = time.monotonic()
            for sequence in range(count):
                due = started + sequence / hz
                delay = due - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                sent_at[sequence] = time.monotonic()
                server.send_state(_state_message(sequence))
            elapsed = time.monotonic() - started
            await _wait_for(lambda: len(page.states) >= count, timeout=5.0)

            sizes = {len(payload.encode()) for _, payload in page.states}
            latencies = [
                (arrival - sent_at[json.loads(payload)["seq"]]) * 1e3
                for arrival, payload in page.states
            ]
            measured.update(
                count=len(page.states),
                hz=len(page.states) / elapsed,
                bytes=max(sizes),
                latency_median_ms=statistics.median(latencies),
                latency_max_ms=max(latencies),
                dropped=server.state_dropped,
            )
            assert server.state_sent == count
            assert server.state_dropped == 0
            # Nothing is lost on a loopback link that keeps up.
            assert len(page.states) == count
            assert [json.loads(payload)["seq"] for _, payload in page.states] == list(
                range(count)
            )
            assert 70.0 <= measured["hz"] <= 74.0
            # The compress cell's order of magnitude: 191 joint values of
            # JSON, the 1.4 kB the design budgets for (the exact size of a
            # real message is measured against the sim in
            # oqt tests/unit/test_sim_scene_state.py).
            assert 1000 <= measured["bytes"] <= 1600
            # A queue would show as a latency that climbs with the stream.
            assert measured["latency_median_ms"] < 20.0
            assert latencies[-1] < 50.0
        finally:
            await page.close()
            await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 60.0))
    print(f"scene_state through a real peer: {measured}")


def test_a_slow_link_drops_new_scenes_instead_of_queueing_them():
    # The point of the unreliable channel: when the send buffer is already
    # holding a frame's worth, the newest scene is thrown away rather than
    # queued behind an older one. Measured against a real peer by shrinking
    # the limit to nothing, which is what a link that cannot keep up looks
    # like from here.
    async def scenario():
        node = _Node()
        server = node.server(state_buffer_limit=0)
        page = _Browser()
        try:
            await page.connect(server)
            page.start()
            await _wait_for(lambda: server._active_peer is not None)
            # Back to back with no awaits: the buffer cannot drain between
            # sends, so everything after the first is dropped.
            results = [
                server.send_state(_state_message(sequence)) for sequence in range(50)
            ]
            assert results[0] is True
            assert results.count(True) < 10, (
                "a full buffer must not accept the whole burst"
            )
            assert server.state_dropped == results.count(False)
            assert server.state_sent == results.count(True)
            await _wait_for(lambda: len(page.states) == results.count(True))
            # What did arrive is a subset of what was sent, in order, and
            # never more than was accepted: nothing queued up behind.
            arrived = [json.loads(payload)["seq"] for _, payload in page.states]
            assert arrived == sorted(arrived)
            assert len(arrived) == results.count(True)

            # Once the buffer drains the next scene goes out: a slow link
            # loses frames, it does not lose the stream.
            await _wait_for(
                lambda: server._states[server._active_peer].bufferedAmount == 0
            )
            assert server.send_state(_state_message(999)) is True
            await _wait_for(lambda: json.loads(page.states[-1][1])["seq"] == 999)
        finally:
            await page.close()
            await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 30.0))


def test_the_scene_stream_leaves_the_pose_and_hud_paths_alone():
    # The page's own ``xr`` channel still carries poses to the node, and
    # the HUD still arrives on ``control``, while scenes are flowing.
    async def scenario():
        node = _Node()
        server = node.server()
        page = _Browser()
        try:
            await page.connect(server)
            page.start()
            await _wait_for(lambda: server._active_peer is not None)
            for sequence in range(20):
                server.send_state(_state_message(sequence))
                page.frame("operator")
                await asyncio.sleep(0.005)
            server.send_hud({"type": "hud", "lines": ["REC"], "recording": True})
            await _wait_for(lambda: len(node.frames) == 20)
            await _wait_for(lambda: "hud" in page.types())
            await _wait_for(lambda: len(page.states) == 20)
            assert node.frames == ["operator"] * 20
            assert page.types() == ["configuration", "hud"]
        finally:
            await page.close()
            await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 30.0))


# What the configuration message says about the exported glTF (X1/X2)


def _glb(extras, payload=b"\0" * 64):
    """A binary glTF whose JSON chunk carries ``extras`` on its one scene."""
    document = json.dumps({"scenes": [{"extras": extras}], "scene": 0}).encode()
    document += b" " * (-len(document) % 4)
    chunk = struct.pack("<II", len(document), 0x4E4F534A) + document
    body = payload + b"\0" * (-len(payload) % 4)
    binary = struct.pack("<II", len(body), 0x004E4942) + body
    header = struct.pack("<III", 0x46546C67, 2, 12 + len(chunk) + len(binary))
    return header + chunk + binary


def test_the_configuration_carries_the_glb_hash_and_body_order(tmp_path):
    path = tmp_path / "scene.glb"
    path.write_bytes(
        _glb({"nq": 191, "bodies": ["world", "arm_right_link1", "compress"]})
    )
    scene = main._configure_scene(path)
    assert scene["url"] == "/scene/scene.glb"
    assert scene["nq"] == 191
    assert scene["bodies"] == ["world", "arm_right_link1", "compress"]
    assert scene["bytes"] == path.stat().st_size
    # The hash is of the file, so a stale cached asset cannot pass for it.
    assert len(scene["hash"]) == 32
    path.write_bytes(_glb({"nq": 191, "bodies": ["world"]}))
    assert main._configure_scene(path)["hash"] != scene["hash"]


def test_a_missing_or_unreadable_glb_leaves_the_page_on_the_panel(tmp_path, capsys):
    assert main._configure_scene(None) is None
    assert main._configure_scene(tmp_path / "nothing.glb") is None
    assert "no 3D scene" in capsys.readouterr().out
    # A file that is not a glTF still serves: the page is told the hash and
    # decides for itself, and the node never loads the meshes to find out.
    plain = tmp_path / "scene.glb"
    plain.write_bytes(b"not a glTF at all")
    scene = main._configure_scene(plain)
    assert scene is not None and "nq" not in scene and "bodies" not in scene
    assert "cannot read the scene extras" in capsys.readouterr().out


def test_the_scene_block_rides_the_configuration_message_to_the_page():
    scene = {
        "url": "/scene/scene.glb",
        "hash": "abc",
        "bytes": 12,
        "nq": 19,
        "bodies": ["world"],
    }

    async def scenario(server, expected):
        page = _Browser()
        try:
            await page.connect(server)
            assert page.messages[0].get("scene") == expected
        finally:
            await page.close()
            await server.close()

    node = _Node()
    asyncio.run(asyncio.wait_for(scenario(node.server(scene=scene), scene), 30.0))
    # Without a scene the configuration is exactly what it is today.
    asyncio.run(asyncio.wait_for(scenario(node.server(), None), 30.0))


def test_the_glb_is_served_once_by_name_with_its_hash_as_the_etag(
    tmp_path, monkeypatch
):
    path = tmp_path / "scene.glb"
    path.write_bytes(_glb({"nq": 3, "bodies": ["world"]}))
    scene = main._configure_scene(path)
    monkeypatch.setattr(main, "_SCENE", scene)
    monkeypatch.setattr(main, "_SCENE_GLB", str(path))
    status, headers, body = asyncio.run(_get("/scene/scene.glb"))
    assert status == 200
    assert headers[b"content-type"] == b"model/gltf-binary"
    assert headers[b"etag"] == f'"{scene["hash"]}"'.encode()
    assert body == path.read_bytes()

    monkeypatch.setattr(main, "_SCENE", None)
    monkeypatch.setattr(main, "_SCENE_GLB", None)
    status, _headers, body = asyncio.run(_get("/scene/scene.glb"))
    assert status == 404
    assert json.loads(body)["error"] == "this node has no 3D scene"


# Compressed on the wire (the headset fetches this over Wi-Fi)


# A real export, the one tests/web draws: 11,520 bytes of glTF, which is the
# same float32 vertex and index data a 4.7 MB cell is made of.
FIXTURE_GLB = pathlib.Path(__file__).parent / "web" / "fixture" / "arm.glb"


def _serve(monkeypatch, path):
    """Put the node behind one scene file, the way main() does at startup."""
    scene = main._configure_scene(path)
    monkeypatch.setattr(main, "_SCENE", scene)
    monkeypatch.setattr(main, "_SCENE_GLB", str(path))
    return scene


def test_the_glb_goes_over_the_wire_gzipped(monkeypatch):
    # The asset is the first-load cost in the headset, and over Wi-Fi it is the
    # bytes that cost, not the disk.
    scene = _serve(monkeypatch, FIXTURE_GLB)
    original = FIXTURE_GLB.read_bytes()
    status, headers, body = asyncio.run(
        _get("/scene/scene.glb", {"accept-encoding": "gzip, deflate, br"})
    )
    assert status == 200
    assert headers[b"content-encoding"] == b"gzip"
    assert headers[b"content-type"] == b"model/gltf-binary"
    # What a cache between here and the headset needs to know: these bytes are
    # one of two representations of this URL.
    assert headers[b"vary"] == b"Accept-Encoding"
    # The number that matters: fewer bytes on the wire, and Content-Length says
    # so, so the headset's own Resource Timing sees the saving.
    assert len(body) < len(original)
    assert headers[b"content-length"] == str(len(body)).encode()
    # And they are still the file: the page's glTF loader gets the same bytes
    # back after the browser has inflated them.
    assert gzip.decompress(body) == original
    print(
        f"scene.glb on the wire: {len(original)} -> {len(body)} bytes "
        f"({len(body) / len(original) * 100:.1f}%)"
    )
    # Different bytes, so a different entity tag (RFC 9110 8.8.3), or a cache
    # would hand the compressed copy to a client that refused it.
    assert headers[b"etag"] == f'"{scene["hash"]}-gzip"'.encode()


def test_a_client_that_will_not_inflate_gets_the_file_itself(monkeypatch):
    scene = _serve(monkeypatch, FIXTURE_GLB)
    original = FIXTURE_GLB.read_bytes()
    for accept in (None, "identity", "gzip;q=0", "*;q=0", "br"):
        status, headers, body = asyncio.run(
            _get(
                "/scene/scene.glb",
                None if accept is None else {"accept-encoding": accept},
            )
        )
        assert status == 200, accept
        assert b"content-encoding" not in headers, accept
        assert body == original, accept
        assert headers[b"etag"] == f'"{scene["hash"]}"'.encode(), accept
        assert headers[b"vary"] == b"Accept-Encoding", accept
    # A wildcard, or a gzip behind a wildcard that refuses everything else, is
    # still a yes.
    for accept in ("*", "br;q=1.0, gzip;q=0.5", "*;q=0, gzip"):
        _status, headers, _body = asyncio.run(
            _get("/scene/scene.glb", {"accept-encoding": accept})
        )
        assert headers[b"content-encoding"] == b"gzip", accept


def test_a_cached_asset_is_revalidated_instead_of_sent_again(monkeypatch):
    # Past Cache-Control's hour the headset revalidates rather than trusting
    # its copy; without this that costs the whole asset a second time.
    _serve(monkeypatch, FIXTURE_GLB)
    for accept in ({"accept-encoding": "gzip"}, {}):
        _status, headers, _body = asyncio.run(_get("/scene/scene.glb", accept))
        etag = headers[b"etag"].decode()
        status, again, body = asyncio.run(
            _get("/scene/scene.glb", {**accept, "if-none-match": etag})
        )
        assert status == 304, accept
        assert body == b""
        assert again[b"etag"] == etag.encode()
        assert again[b"cache-control"] == b"public, max-age=3600"
        assert again[b"vary"] == b"Accept-Encoding"
        # A stale tag (another run's scene) is a download, not a 304.
        stale = asyncio.run(
            _get("/scene/scene.glb", {**accept, "if-none-match": '"0" * 32'})
        )
        assert stale[0] == 200


def test_the_compressed_copy_is_a_function_of_the_file_alone(monkeypatch, tmp_path):
    # mtime=0: two nodes serving the same export serve the same bytes, the way
    # the exporter itself is byte-deterministic. A clock in there would make
    # every relaunch a cache miss for a headset that already has the asset.
    path = tmp_path / "scene.glb"
    path.write_bytes(FIXTURE_GLB.read_bytes())
    first = main._compress_scene(path.read_bytes(), "abc")
    time.sleep(1.1)
    path.touch()
    second = main._compress_scene(path.read_bytes(), "abc")
    assert first["body"] == second["body"]


def test_an_asset_that_does_not_shrink_is_served_as_it_is(monkeypatch, tmp_path):
    # Not every scene is float32 meshes: an export whose payload is already
    # compressed would come out of gzip bigger than it went in.
    path = tmp_path / "scene.glb"
    path.write_bytes(_glb({"nq": 3}, payload=random.Random(7).randbytes(1 << 16)))
    _serve(monkeypatch, path)
    assert main._SCENE_GZIP is None
    status, headers, body = asyncio.run(
        _get("/scene/scene.glb", {"accept-encoding": "gzip"})
    )
    assert status == 200
    assert b"content-encoding" not in headers
    assert body == path.read_bytes()


def test_a_scene_swapped_under_the_endpoint_is_not_served_the_old_compression(
    monkeypatch, tmp_path
):
    # The compressed copy is keyed by the hash it was made from. Nothing in the
    # node swaps a scene today -- it is read once at startup -- but serving one
    # file's bytes under another's hash is the failure the page cannot detect,
    # because the hash it checks would be the right one.
    _serve(monkeypatch, FIXTURE_GLB)
    other = tmp_path / "other.glb"
    other.write_bytes(_glb({"nq": 5, "bodies": ["world"]}))
    monkeypatch.setattr(main, "_SCENE", {**main._SCENE, "hash": "not the same file"})
    monkeypatch.setattr(main, "_SCENE_GLB", str(other))
    status, headers, body = asyncio.run(
        _get("/scene/scene.glb", {"accept-encoding": "gzip"})
    )
    assert status == 200
    assert b"content-encoding" not in headers
    assert body == other.read_bytes()


async def _get(path, headers=None):
    """GET ``path`` through the node's ASGI app; returns (status, headers, body)."""
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    request_headers = [(b"host", b"node")]
    for name, value in (headers or {}).items():
        request_headers.append((name.lower().encode(), value.encode()))

    await main.app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": request_headers,
            "client": ("192.0.2.7", 50000),
            "server": ("127.0.0.1", 8443),
        },
        receive,
        send,
    )
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return start["status"], dict(start["headers"]), body


# The dora input (X2)


class _Forwarder:
    """Stands in for WebRTCServer in _on_scene_state."""

    def __init__(self, accept=True):
        self.payloads = []
        self.state_dropped = 0
        self.state_bytes = 0
        self.state_buffered_peak = 0
        self.state_buffer_limit = webrtc.STATE_BUFFER_LIMIT
        self._accept = accept

    def send_state(self, payload):
        self.payloads.append(payload)
        if self._accept:
            self.state_bytes += (
                len(payload.encode()) if isinstance(payload, str) else len(payload)
            )
        return self._accept


@pytest.mark.parametrize(
    "value",
    [
        '{"type":"state"}',
        pa.array(['{"type":"state"}']),
        pa.array([b'{"type":"state"}'], type=pa.binary()),
    ],
)
def test_a_json_scene_state_is_forwarded_as_the_sim_sent_it(value, monkeypatch):
    forwarder = _Forwarder()
    monkeypatch.setattr(main, "webrtc_server", forwarder)
    main._on_scene_state(value)
    assert forwarder.payloads == ['{"type":"state"}']


def test_a_binary_scene_state_is_forwarded_as_bytes(monkeypatch):
    forwarder = _Forwarder()
    monkeypatch.setattr(main, "webrtc_server", forwarder)
    payload = struct.pack("<IBBHdHH", 0x4F515433, 1, 5, 42, 12.3456, 2, 0) + b"\0" * 8
    main._on_scene_state(pa.array(list(payload), type=pa.uint8()))
    assert forwarder.payloads == [payload]
    assert isinstance(forwarder.payloads[0], bytes)


@pytest.mark.parametrize(
    "value",
    [None, "", pa.array([], type=pa.string()), pa.array([1.5], type=pa.float64()), 17],
)
def test_a_malformed_scene_state_is_dropped_and_said_once(value, monkeypatch, capsys):
    monkeypatch.setattr(main, "_warned_scene_state", False, raising=False)
    forwarder = _Forwarder()
    monkeypatch.setattr(main, "webrtc_server", forwarder)
    main._on_scene_state(value)
    main._on_scene_state(value)
    assert forwarder.payloads == []
    # Said once, not once a frame.
    assert capsys.readouterr().out.count("scene_state:") == 1


def test_an_absurdly_long_scene_state_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(main, "_warned_scene_state", False, raising=False)
    forwarder = _Forwarder()
    monkeypatch.setattr(main, "webrtc_server", forwarder)
    main._on_scene_state("x" * (main._SCENE_STATE_MAX_BYTES + 1))
    assert forwarder.payloads == []
    assert "dropping a" in capsys.readouterr().out


def _reset_state_report(monkeypatch):
    for name, value in (
        ("_state_forwarded", 0),
        ("_state_reported", 0),
        ("_state_reported_at", 0.0),
        ("_state_bytes_reported", 0),
    ):
        monkeypatch.setattr(main, name, value, raising=False)


def test_a_link_that_keeps_dropping_says_so_at_most_once_a_batch(monkeypatch, capsys):
    _reset_state_report(monkeypatch)
    forwarder = _Forwarder()
    monkeypatch.setattr(main, "webrtc_server", forwarder)
    for index in range(main._STATE_REPORT_EVERY):
        forwarder.state_dropped = index  # every other frame, say
        forwarder.state_buffered_peak = 6000
        main._on_scene_state('{"type":"state"}')
    out = capsys.readouterr().out
    assert out.count("scene_state:") == 1
    assert len(forwarder.payloads) == main._STATE_REPORT_EVERY


def test_the_drop_report_says_what_it_measured_not_what_it_guessed(monkeypatch, capsys):
    """The old line read "N scenes did not fit the headset link and were dropped".

    That is a verdict on the radio, and the test behind it is
    ``bufferedAmount > 4096 bytes`` -- about 70 ms of backlog at 72 Hz, which
    the node's own event loop trips as readily as a slow link. The 2026-09-16
    session was read as "the link cannot carry the scene" on the strength of
    it. The line now names the limit it compared against and how deep the
    buffer actually got, so the two can be told apart.
    """
    _reset_state_report(monkeypatch)
    forwarder = _Forwarder()
    monkeypatch.setattr(main, "webrtc_server", forwarder)
    for _ in range(main._STATE_REPORT_EVERY):
        forwarder.state_dropped = 78
        forwarder.state_buffered_peak = 6132
        main._on_scene_state('{"type":"state"}')
    line = next(t for t in capsys.readouterr().out.splitlines() if "scene_state:" in t)
    assert "did not fit" not in line
    assert f"{main._STATE_REPORT_EVERY} forwarded" in line
    assert "78 dropped with more than 4096 B already queued (peak 6132 B)" in line
    # And it says the rate and the size, so a real capacity problem is
    # arithmetic rather than a guess.
    assert "Hz)" in line and "B each" in line


def test_a_healthy_stream_says_so_too(monkeypatch, capsys):
    _reset_state_report(monkeypatch)
    forwarder = _Forwarder()
    monkeypatch.setattr(main, "webrtc_server", forwarder)
    for _ in range(main._STATE_REPORT_EVERY):
        main._on_scene_state('{"type":"state"}')
    line = next(t for t in capsys.readouterr().out.splitlines() if "scene_state:" in t)
    assert "none dropped" in line


# The stream is not sent to a view that draws no 3D scene (the LOST finding)


def test_no_state_channel_is_offered_to_a_view_that_draws_no_scene():
    """The 2026-09-16 session ran `--view mono` with the 72 Hz stream on.

    ``connection.js`` only registers a handler for the ``state`` channel from a
    3D view, so every one of those messages was received, decoded and thrown
    away -- while its packets still took a contention cycle each on the
    half-duplex 2.4 GHz channel the operator's hand poses came back over.
    Measured on the host during that run: 72 of 557 tx packets a second and
    0.77 Mbit/s, against a clutch freshness window 7.2 frames wide.
    """

    async def scenario():
        node = _Node()
        server = node.server(state_channel=False)
        page = _Browser()
        try:
            await page.connect(server, expect_state=False)
            # The page is offered ``control`` and nothing else.
            assert page.labels == ["control"]
            assert page.state is None
            page.start()
            await _wait_for(lambda: server._active_peer is not None)
            # And nothing goes out, nor is counted as a drop: there is
            # nothing to be behind on.
            assert server.send_state(_state_message(1)) is False
            await asyncio.sleep(0.2)
            assert page.states == []
            assert (server.state_sent, server.state_dropped) == (0, 0)
            # The rest of the session is untouched: poses still arrive.
            page.frame("operator")
            await _wait_for(lambda: node.frames == ["operator"])
        finally:
            await page.close()
            await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 30.0))


def test_a_browser_that_waits_for_the_state_channel_is_not_left_waiting():
    # `_Browser.connect` waits for the configuration and the state channel;
    # with the channel gone the configuration must still arrive, which is what
    # the page actually needs to draw itself.
    async def scenario():
        node = _Node()
        server = node.server(state_channel=False)
        page = _Browser()
        try:
            await page.connect(server, expect_state=False)
            assert page.types() == ["configuration"]
        finally:
            await page.close()
            await server.close()

    asyncio.run(asyncio.wait_for(scenario(), 30.0))


def test_the_node_forwards_nothing_when_the_view_draws_no_scene(monkeypatch, capsys):
    # And it is dropped where it arrives, without being parsed: the point is
    # to spend nothing at all on it.
    monkeypatch.setattr(main, "_FORWARD_SCENE_STATE", False, raising=False)
    monkeypatch.setattr(main, "_scene_state_unwanted", 0, raising=False)
    monkeypatch.setattr(main, "_warned_scene_state_unwanted", False, raising=False)
    forwarder = _Forwarder()
    monkeypatch.setattr(main, "webrtc_server", forwarder)
    for _ in range(100):
        main._on_scene_state('{"type":"state"}')
    assert forwarder.payloads == []
    assert main._scene_state_unwanted == 100
    # Said once, with what to do about it -- not once a frame.
    out = capsys.readouterr().out
    assert out.count("scene_state:") == 1
    assert "draws no 3D scene" in out and "--scene-state-hz 0" in out


@pytest.mark.parametrize(
    "view, draws",
    [
        ("scene", True),
        ("scene+panel", True),
        ("mono", False),
        ("stereo", False),
        ("none", False),
    ],
)
def test_which_views_draw_the_scene(monkeypatch, view, draws):
    # The page decides the same thing in ar.js SCENE_VIEWS; these two lists
    # are the contract, and the node must not open a channel the page will
    # never read from.
    monkeypatch.setattr(video, "_view_configuration", {"view": view})
    assert video.draws_scene() is draws


def test_the_page_modules_are_served_next_to_the_scene_route():
    # The route is declared before the static mount, which matches every
    # path: without that order /scene/scene.glb would 404 as a static file.
    static = pathlib.Path(main.__file__).parent / "static"
    assert (static / "connection.js").exists()
    paths = [getattr(route, "path", None) for route in main.app.routes]
    mount = next(
        index
        for index, route in enumerate(main.app.routes)
        if route.__class__.__name__ == "Mount"
    )
    assert paths.index("/scene/scene.glb") < mount
