# openarm-quest-teleop addition to the vendored dora-openarm-webxr tests
# (Apache-2.0, like the rest of this directory).
#
# The headset's own log (T3): POST /log and the control channel, the
# pairing token, the caps that keep a mad page from taking the node down,
# the redaction that keeps the token out of the file, and the rotation.

import asyncio
import json
import pathlib
import time

import pytest
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from dora_openarm_webxr import main, video, webrtc

TOKEN = "s3cret-token-0123456789"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(webrtc, "ICE_SERVERS", [])
    monkeypatch.setattr(main, "_PAIRING_TOKEN", TOKEN, raising=False)
    monkeypatch.setattr(main, "_quest_log", None, raising=False)
    # The request limiter is module state shared by every test.
    monkeypatch.setattr(main, "_log_requests", main._PeerLimiter(), raising=False)
    video.reset()
    yield
    video.reset()


class _Clock:
    """A monotonic clock the tests move by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def sink(tmp_path, monkeypatch):
    """A log file at tmp_path/quest.log, with its clock under our control."""
    clock = _Clock()
    log = main._QuestLog(
        tmp_path / "quest.log",
        clock=clock,
        now=lambda: "2026-09-17T00:00:00.000+09:00",
    )
    log.clock = clock
    monkeypatch.setattr(main, "_quest_log", log, raising=False)
    return log


def lines(path):
    """Every record in the file, parsed."""
    if not pathlib.Path(path).exists():
        return []
    return [
        json.loads(line)
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


# ---------------------------------------------------------------------------
# POST /log, with no WebRTC anywhere


async def _post_log(
    body,
    *,
    headers=None,
    chunks=None,
    client=("192.0.2.7", 50000),
    path="/log",
    cut=False,
):
    """POST ``body`` to /log through the node's ASGI app, as uvicorn would.

    ``cut`` leaves the body unfinished, so the next receive() is the
    disconnect a browser being torn down sends.
    """
    pieces = chunks if chunks is not None else [body]
    requests = [
        {
            "type": "http.request",
            "body": piece,
            "more_body": cut or index < len(pieces) - 1,
        }
        for index, piece in enumerate(pieces)
    ]
    sent = []

    async def receive():
        return requests.pop(0) if requests else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    default = [
        (b"content-type", b"application/json"),
        (b"host", b"node"),
        (b"content-length", str(len(body)).encode()),
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": default if headers is None else headers,
        "client": client,
        "server": ("127.0.0.1", 8443),
    }
    await main.app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    payload = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return status, payload


def post(body, **kwargs):
    if not isinstance(body, bytes):
        body = json.dumps(body).encode()
    return asyncio.run(_post_log(body, **kwargs))


def envelope(*records, token=TOKEN, session="head-1", **extra):
    body = {"session": session, "records": list(records), **extra}
    if token is not None:
        body["token"] = token
    return body


def test_a_post_with_the_token_is_written_as_one_json_line(sink):
    status, payload = post(
        envelope({"level": "error", "event": "uncaught-error", "message": "boom"})
    )
    assert (status, payload) == (204, b"")
    (record,) = lines(sink.path)
    assert record["level"] == "error"
    assert record["event"] == "uncaught-error"
    assert record["message"] == "boom"
    # What the node knows and the page cannot forge.
    assert record["session"] == "head-1"
    assert record["peer"] == "192.0.2.7"
    assert record["via"] == "http"
    assert record["seq"] == 1
    assert record["received"] == "2026-09-17T00:00:00.000+09:00"


@pytest.mark.parametrize(
    "body",
    [
        {"session": "x", "records": []},
        {"session": "x", "records": [], "token": "wrong"},
        {"session": "x", "records": [], "token": None},
        b"not json at all",
        b"\xff\xfe",
    ],
)
def test_a_post_without_the_pairing_token_is_refused(sink, body, capsys):
    # The page's log endpoint is reachable by anything that reaches the port,
    # so it is paired exactly like POST /offer.
    status, payload = post(body)
    assert status == 403
    assert b"pairing token" in payload
    assert lines(sink.path) == []
    assert "refused a log post from 192.0.2.7" in capsys.readouterr().out


def test_every_line_says_which_page_build_and_which_batch_it_came_from(sink):
    """An operator greps one line out and sends it; it has to stand alone.

    The page's own envelope counter is the half the node cannot supply: a gap
    in it means an envelope never arrived, which no counter of this node's can
    show.
    """
    status, _ = post(
        envelope(
            {"level": "error", "event": "uncaught-error", "message": "boom"},
            build="0.6.0-page",
            node="0.6.0+abcdef0123456789",
            page="7",
            reason="pagehide",
            v=1,
        )
    )
    assert status == 204
    (record,) = lines(sink.path)
    assert record["build"] == "0.6.0-page"
    assert record["node"] == "0.6.0+abcdef0123456789"
    assert record["page"] == "7"
    assert record["flush"] == "pagehide"
    assert record["v"] == 1
    # And a page that sends none of them still writes a usable line.
    post(envelope({"event": "plain"}))
    assert "build" not in lines(sink.path)[-1]


def test_what_the_page_itself_dropped_is_a_line_of_its_own(sink):
    """The page's ring is bounded too, and a full one is a page in trouble.

    Without this the hole would read as the node losing records, which is a
    different fault with a different fix.
    """
    post(envelope({"event": "after-the-hole"}, dropped=42))
    first, second = lines(sink.path)
    assert first["event"] == "page-dropped" and first["count"] == 42
    assert first["level"] == "warn"
    assert second["event"] == "after-the-hole"
    assert sink.page_dropped == 42
    # ... and it is not counted as one of the node's own drops.
    assert sink.dropped == 0
    assert "42 dropped by the page" in sink.summary()


def test_both_names_of_the_endpoint_answer(sink):
    # The page posts to /log; its browser harness calls the same sink
    # /telemetry. A page that reached neither would back off in silence.
    for path in ("/log", "/telemetry"):
        scope_post = asyncio.run(
            _post_log(json.dumps(envelope({"event": path})).encode(), path=path)
        )
        assert scope_post == (204, b""), path
    assert [record["event"] for record in lines(sink.path)] == ["/log", "/telemetry"]


def test_a_post_cut_off_mid_upload_is_not_a_crash(sink, capsys):
    # A page being torn down is exactly when this endpoint is used, and a
    # browser can go away between its headers and its body.
    body = json.dumps(envelope({"event": "x"})).encode()
    status, _ = asyncio.run(_post_log(body, chunks=[body[:10]], cut=True))
    assert status == 400
    assert "was cut off" in capsys.readouterr().out
    assert lines(sink.path) == []
    # ... and the node still takes the next one.
    assert post(envelope({"event": "after"}))[0] == 204


def test_a_body_that_is_not_an_envelope_is_a_400_and_writes_nothing(sink):
    for body in ({"token": TOKEN}, {"token": TOKEN, "records": "everything"}):
        assert post(body)[0] == 400
    assert lines(sink.path) == []


def test_a_node_with_no_log_file_says_so_instead_of_dropping_the_report(monkeypatch):
    # 503, not 204: the page has to know its buffer is still its own.
    monkeypatch.setattr(main, "_quest_log", None, raising=False)
    status, payload = post(envelope({"event": "x"}))
    assert status == 503
    assert b"--quest-log" in payload


def test_a_ten_megabyte_body_is_refused_without_being_written(sink):
    big = json.dumps(envelope({"event": "x", "message": "A" * (10 << 20)})).encode()
    assert len(big) > 10 << 20
    status, payload = post(big)
    assert status == 413
    assert b"at most" in payload
    assert lines(sink.path) == []


def test_an_oversized_body_is_refused_even_with_a_lying_content_length(sink):
    # The bound is on what was actually read, not on what the sender claimed.
    big = json.dumps(envelope({"event": "x", "message": "A" * (1 << 20)})).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", b"12")]
    chunks = [big[i : i + 65536] for i in range(0, len(big), 65536)]
    assert asyncio.run(_post_log(big, headers=headers, chunks=chunks))[0] == 413
    assert lines(sink.path) == []


def test_a_flood_of_posts_is_rate_limited_per_peer(sink):
    refused = [post(envelope({"event": "flood"}))[0] for _ in range(60)]
    assert 204 in refused and 429 in refused
    assert refused.count(204) <= main.QUEST_LOG_REQUEST_BURST + 1
    # The other headset in the room is not locked out by the first one.
    assert post(envelope({"event": "other"}), client=("192.0.2.8", 40000))[0] == 204
    assert lines(sink.path)[-1]["peer"] == "192.0.2.8"


def test_a_flood_of_records_is_capped_and_the_file_says_so(sink):
    # A page in a console.error loop: the batch cap first, then the record
    # rate. What matters is that the file admits what it lost.
    for _ in range(40):
        post(envelope(*({"event": "spin", "i": i} for i in range(200))))
    written = lines(sink.path)
    assert sink.dropped > 0
    dropped = [record for record in written if record["event"] == "quest-log-dropped"]
    assert dropped, "the file must say that records were thrown away"
    assert dropped[0]["count"] >= 136  # 200 - QUEST_LOG_MAX_RECORDS
    # The sequence numbers are the node's own and never repeat.
    numbers = [record["seq"] for record in written]
    assert numbers == sorted(set(numbers))


def test_small_batches_in_a_loop_are_capped_by_the_record_rate(sink):
    """The batch cap only bounds ONE envelope.

    A page whose console.error fires every frame posts ten records at a time
    for as long as it lives, and what has to hold then is the rate, not the
    batch. The clock is frozen here, so the bucket only has its burst.
    """
    for _ in range(100):
        sink.accept(
            {"session": "s", "records": [{"event": "spin"}] * 10}, peer="p", via="http"
        )
    assert sink.written <= main.QUEST_LOG_RECORD_BURST
    assert sink.dropped >= 1000 - main.QUEST_LOG_RECORD_BURST
    # A minute later the bucket has refilled, and the first record through
    # says how much was lost while it was empty.
    sink.clock.now += 60.0
    sink.accept({"session": "s", "records": [{"event": "after"}]}, peer="p", via="http")
    note = [
        record for record in lines(sink.path) if record["event"] == "quest-log-dropped"
    ][-1]
    assert note["count"] >= 1000 - main.QUEST_LOG_RECORD_BURST
    assert lines(sink.path)[-1]["event"] == "after"


# ---------------------------------------------------------------------------
# Redaction: the pairing token must never reach the file


def test_the_pairing_token_is_never_written_however_it_arrives(sink):
    post(
        envelope(
            {
                "event": "uncaught-error",
                "message": f"failed to fetch https://node:8443/ar.js#k={TOKEN}",
                "url": f"https://node:8443/?k={TOKEN}#k={TOKEN}",
                "fragment": f"#k={TOKEN}",
                "stack": f"at connect (https://node:8443/connection.js:66:13)\nk={TOKEN}",
                "nested": {"deep": [f"token={TOKEN}"]},
            }
        )
    )
    text = sink.path.read_text(encoding="utf-8")
    assert TOKEN not in text
    # ... and the body of the record is still worth reading.
    (record,) = lines(sink.path)
    assert "https://node:8443/ar.js" in record["message"]
    assert "connection.js:66:13" in record["stack"]
    assert record["url"] == "https://node:8443/?<redacted>#<redacted>"
    assert record["fragment"] == "#k=<redacted>"


def test_redaction_is_linear_in_the_length_of_a_word(sink):
    """The redaction runs on the event loop that answers the operator's hands.

    A page in trouble sends long unbroken words -- a base64 blob, a stack of a
    minified bundle, one character repeated -- and a URL pattern that can
    backtrack turns a single 60 kB word into MINUTES of that loop. One POST may
    carry 64 kB, so this is reachable from the page.
    """
    started = time.perf_counter()
    assert post(envelope({"event": "blob", "message": "A" * 60_000}))[0] == 204
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, f"redacting one 64 kB word took {elapsed:.1f} s"
    assert lines(sink.path)[-1]["message"].startswith("A")


def test_a_token_this_node_does_not_know_is_redacted_too(sink):
    # The page logs the body it posted to /offer, and that body carries a
    # token -- which on a page that reached the wrong node is not this node's
    # to recognise. Names, not values: "key" is left alone because that is
    # also what a keyboard event calls the letter that was pressed.
    post(
        envelope(
            {
                "event": "offer",
                "msg": 'posted {"sdp": "v=0", "token": "someone-elses-token"}',
                "data": {"pressed": '{"key": "ArrowUp"}'},
            }
        )
    )
    (record,) = lines(sink.path)
    assert "someone-elses-token" not in json.dumps(record)
    assert '"token": "<redacted>"' in record["msg"]
    assert record["data"]["pressed"] == '{"key": "ArrowUp"}'


def test_the_token_the_page_authenticated_with_is_not_written_either(sink):
    post(envelope({"event": "x"}))
    (record,) = lines(sink.path)
    assert "token" not in record


def test_a_record_cannot_overwrite_what_the_node_knows(sink):
    post(
        envelope(
            {
                "event": "x",
                "peer": "somebody-else",
                "via": "control",
                "received": "1999-01-01",
                "seq": 9999,
                "session": "another",
            },
            session="head-1",
        )
    )
    (record,) = lines(sink.path)
    assert record["peer"] == "192.0.2.7"
    assert record["via"] == "http"
    assert record["seq"] == 1
    assert record["session"] == "head-1"
    assert record["received"] == "2026-09-17T00:00:00.000+09:00"


# ---------------------------------------------------------------------------
# A record the node cannot take at face value


def test_a_record_is_cut_down_rather_than_written_whole(sink):
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": "bottom"}}}}}}}
    post(
        envelope(
            {
                "event": "E" * 500,
                "message": "M" * 10000,
                "list": list(range(1000)),
                "deep": deep,
                "huge": 10**400,
                "level": "SHOUT",
            }
        )
    )
    (record,) = lines(sink.path)
    assert len(record["event"]) <= main.QUEST_LOG_MAX_KEY + 1
    assert len(record["message"]) <= main.QUEST_LOG_MAX_STRING + 1
    assert len(record["list"]) == main.QUEST_LOG_MAX_ITEMS
    assert isinstance(record["huge"], str)
    assert record["level"] == "info"  # an unknown level is not a level
    assert json.dumps(record)  # whatever came in, the line is still JSON


@pytest.mark.parametrize(
    "records",
    [
        ["a string, not an object"],
        [None, 42],
        [{"event": "kept"}, "rubbish"],
    ],
)
def test_a_record_that_is_not_an_object_is_dropped_not_fatal(sink, records):
    assert post(envelope(*records))[0] == 204
    assert sink.dropped >= 1
    assert all(record["event"] != "rubbish" for record in lines(sink.path))


def test_nothing_the_page_can_send_raises(sink):
    # Everything json.loads can produce, in the places a record has.
    for value in ({}, [], "", 0, 1.5, True, None, float("1e400"), {"": {"": {}}}):
        assert post(envelope({"event": "x", "value": value}))[0] == 204
    assert sink.written >= 9


# ---------------------------------------------------------------------------
# The file itself


def test_the_file_is_capped_and_rotated_once(tmp_path, monkeypatch):
    log = main._QuestLog(tmp_path / "quest.log", max_bytes=4096, now=lambda: "t")
    monkeypatch.setattr(main, "_quest_log", log, raising=False)
    for index in range(200):
        log.accept(
            {
                "session": "s",
                "records": [{"event": "fill", "message": "x" * 200, "i": index}],
            },
            peer="p",
            via="http",
        )
    assert log.rotations >= 1
    assert log.path.stat().st_size <= 4096
    rotated = log.path.with_name("quest.log.1")
    assert rotated.is_file() and rotated.stat().st_size <= 4096
    # Only one rotation: a run costs at most twice the cap, never a pile.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["quest.log", "quest.log.1"]
    # And the newest records are the ones that survived.
    assert lines(log.path)[-1]["i"] == 199
    log.close()


def test_a_file_that_cannot_be_written_does_not_stop_the_node(tmp_path, capsys):
    log = main._QuestLog(tmp_path / "nope" / "quest.log", now=lambda: "t")
    (tmp_path / "nope").write_text("a file where the directory should be")
    assert log.accept({"records": [{"event": "x"}]}, peer="p", via="http") == 0
    assert log.write_failures == 1
    assert "cannot write" in capsys.readouterr().err
    # ... and it is not retried on every record either.
    assert log.accept({"records": [{"event": "x"}]}, peer="p", via="http") == 0
    assert log.write_failures == 1


def test_errors_are_mirrored_to_stderr_where_the_operator_already_looks(sink, capsys):
    post(
        envelope(
            {"level": "error", "event": "uncaught-error", "message": "TypeError: x"},
            {"level": "warn", "event": "slow-frame", "message": "18 ms"},
        )
    )
    mirrored = capsys.readouterr().err
    assert "headset error [head-1] uncaught-error: TypeError: x" in mirrored
    assert "slow-frame" not in mirrored
    assert sink.errors == 1 and sink.warnings == 1


def test_the_mirror_reads_the_pages_own_field_names(sink, capsys):
    # static/telemetry.js writes `msg`, and coalesces a repeat into `n` rather
    # than sending it 72 times a second; the count is the whole story of how
    # bad it is, so the mirrored line says it.
    post(
        envelope(
            {
                "level": "error",
                "event": "window",
                "msg": "TypeError: hud is not a function",
                "n": 360,
                "record": 12,
            }
        )
    )
    assert "TypeError: hud is not a function (x360)" in capsys.readouterr().err
    # The page's own per-record counter is written through untouched: a gap in
    # it is a record the page's buffer dropped.
    assert lines(sink.path)[-1]["record"] == 12


def test_the_stderr_mirror_is_rate_limited_but_the_file_is_not(sink, capsys):
    for _ in range(3):
        post(envelope(*({"level": "error", "event": "spin"} for _ in range(20))))
    mirrored = capsys.readouterr().err
    assert mirrored.count("headset error") <= main.QUEST_LOG_MIRROR_BURST + 1
    assert "further headset errors are only in" in mirrored
    assert sink.errors == 60
    assert sum(1 for record in lines(sink.path) if record["event"] == "spin") == 60


def test_the_summary_names_what_the_headset_reported(sink, capsys):
    post(
        envelope({"level": "error", "event": "boom"}, {"level": "warn", "event": "hm"})
    )
    sink.close()
    said = capsys.readouterr().out
    assert "quest.log: 2 records, 1 errors, 1 warnings" in said
    assert str(sink.path) in said


# ---------------------------------------------------------------------------
# The control channel, with real aiortc peers


class _Browser:
    """An in-process stand-in for the page, as the session tests use."""

    def __init__(self):
        self.control = None
        self.configuration = None

    async def connect(self, server):
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.xr = self.pc.createDataChannel("xr", ordered=False, maxRetransmits=0)
        ready = asyncio.get_running_loop().create_future()

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            if channel.label != "control":
                return
            self.control = channel

            @channel.on("message")
            def on_message(message):
                if not ready.done():
                    self.configuration = json.loads(message)
                    ready.set_result(self.configuration)

        for _ in range(webrtc.VIDEO_TRANSCEIVERS):
            self.pc.addTransceiver("video", direction="recvonly")
        await self.pc.setLocalDescription(await self.pc.createOffer())
        answer = await server.answer(self.pc.localDescription.sdp)
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer, type="answer")
        )
        await asyncio.wait_for(ready, 10.0)

    def log(self, **envelope_fields):
        self.control.send(json.dumps({"type": "log", **envelope_fields}))


async def _wait_for(predicate, timeout=10.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise TimeoutError
        await asyncio.sleep(0.02)


def _server(**options):
    return webrtc.WebRTCServer(
        on_frame=lambda payload: None,
        on_session_start=lambda: None,
        on_log=main._on_quest_log,
        **options,
    )


def test_two_headsets_log_over_the_control_channel_and_stay_apart(sink):
    asyncio.run(asyncio.wait_for(_run_two_headsets(sink), 60.0))


async def _run_two_headsets(sink):
    server = _server()
    first, second = _Browser(), _Browser()
    try:
        await first.connect(server)
        await second.connect(server)
        first.log(
            session="head-1", records=[{"level": "error", "event": "glb-load-failed"}]
        )
        second.log(
            session="head-2", records=[{"event": "xr-session-end", "reason": "quit"}]
        )
        await _wait_for(lambda: len(lines(sink.path)) == 2)
        by_session = {record["session"]: record for record in lines(sink.path)}
        assert set(by_session) == {"head-1", "head-2"}
        assert by_session["head-1"]["event"] == "glb-load-failed"
        assert by_session["head-2"]["reason"] == "quit"
        # Every record says which transport carried it and which peer sent it;
        # two headsets in one file are worth nothing if they cannot be told apart.
        assert {record["via"] for record in by_session.values()} == {"control"}
        assert len({record["peer"] for record in by_session.values()}) == 2
        assert all(record["peer"].startswith("peer-") for record in by_session.values())
    finally:
        await first.pc.close()
        await second.pc.close()
        await server.close()


def test_a_malformed_control_message_costs_the_record_not_the_session(sink):
    asyncio.run(asyncio.wait_for(_run_malformed(sink), 60.0))


async def _run_malformed(sink):
    server = _server()
    browser = _Browser()
    try:
        await browser.connect(server)
        browser.control.send("{not json")
        browser.log(records="not a list")
        browser.log(session="head-1", records=[[], {"event": "still-here"}])
        await _wait_for(
            lambda: any(r["event"] == "still-here" for r in lines(sink.path))
        )
        # The two envelopes that were not envelopes wrote nothing at all; the
        # record that was not a record cost itself and is admitted to.
        assert [record["event"] for record in lines(sink.path)] == [
            "quest-log-dropped",
            "still-here",
        ]
        # The channel is still open and the next record still arrives.
        browser.log(session="head-1", records=[{"event": "after"}])
        await _wait_for(lambda: any(r["event"] == "after" for r in lines(sink.path)))
    finally:
        await browser.pc.close()
        await server.close()


def test_the_control_channel_needs_no_token_but_its_records_are_still_cleaned(sink):
    asyncio.run(asyncio.wait_for(_run_control_redaction(sink), 60.0))


async def _run_control_redaction(sink):
    server = _server()
    browser = _Browser()
    try:
        await browser.connect(server)
        # No "token" key at all: the peer exists because an offer carried one.
        browser.log(
            session="head-1",
            records=[{"event": "reload", "url": f"https://node:8443/#k={TOKEN}"}],
        )
        await _wait_for(lambda: lines(sink.path))
        (record,) = lines(sink.path)
        assert record["url"] == "https://node:8443/#<redacted>"
        assert TOKEN not in sink.path.read_text(encoding="utf-8")
    finally:
        await browser.pc.close()
        await server.close()


# ---------------------------------------------------------------------------
# Which build of the page this node serves


def test_the_build_changes_when_the_served_page_does(tmp_path):
    # A page out of the headset's HTTP cache is a whole class of reports that
    # otherwise reads as a node gone mad, and the version alone never moves
    # while the page is being worked on.
    (tmp_path / "ar.js").write_text("const a = 1;")
    first = main._build_identity(tmp_path)
    assert first == main._build_identity(tmp_path)
    (tmp_path / "ar.js").write_text("const a = 2;")
    assert main._build_identity(tmp_path) != first
    # A renamed file is a different build too, not the same bytes.
    (tmp_path / "ar.js").rename(tmp_path / "ar2.js")
    assert main._build_identity(tmp_path) != main._build_identity(tmp_path / "nothing")
    assert main._build_identity()  # the node's own static directory


def test_the_page_is_told_which_build_it_should_be(sink):
    asyncio.run(asyncio.wait_for(_run_build(sink), 60.0))


async def _run_build(sink):
    server = _server(build="9.9.9+abcdef")
    browser = _Browser()
    try:
        await browser.connect(server)
        assert browser.configuration["type"] == "configuration"
        assert browser.configuration["build"] == "9.9.9+abcdef"
    finally:
        await browser.pc.close()
        await server.close()


def test_a_node_with_no_sink_ignores_control_log_messages(monkeypatch):
    monkeypatch.setattr(main, "_quest_log", None, raising=False)
    # No sink is the default; this must be a no-op, not an exception.
    main._on_quest_log({"records": [{"event": "x"}]}, "peer-1")
