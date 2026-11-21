"""The browser UI: head_left live with the bbox drawn by dragging, buttons, status.

Standard library only (ThreadingHTTPServer). The handlers never touch the robot:
they put requests on a queue the dora loop takes on its next tick, and read the
status the loop publishes.

    GET  /            the page
    GET  /head.mjpg   head_left as it arrives (the camera's own JPEG, not re-encoded)
    GET  /snapshot.jpg  head_left as it was after the intro or after a run (the bbox is drawn on it)
    GET  /view.jpg    what the network sees: head with the bbox drawn, over wrist_right
    GET  /status      JSON
    POST /bbox        {"bbox": [x1, y1, x2, y2]} normalized to the image (0..1), or null
    POST /action      {"action": "start" | "stop" | "voice_stop" | "proceed" | "reset" | "test" | "snapshot"}
    POST /voice       an utterance, audio/wav (the browser detects and records speech)
    POST /text        {"text": ...}, as if it had been said
    POST /speech_done {"id": n}, the browser finished playing reply n
    POST /assistant/reset   start the dialogue over
    POST /speed       {"speed": 0.05..1}, the chunks' playback speed, applied at once
    POST /vad_stats   what the browser's speech detection sees (logged every 10 s)
    GET  /tts/<n>.mp3 spoken reply n
"""

from __future__ import annotations

import collections
import json
import queue
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

ACTIONS = frozenset({"start", "stop", "voice_stop", "proceed", "reset", "test", "snapshot"})
# An utterance longer than this (16 kHz mono 16-bit WAV, ~60 s) is refused.
MAX_VOICE_BYTES = 2_000_000


class Shared:
    """State shared between the dora loop, the inference thread and the handlers."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame = threading.Condition(self.lock)
        self.head_jpeg: bytes | None = None
        self.head_seq = 0
        self.view_jpeg: bytes | None = None
        # head_left frozen at a moment (after the intro, after a run) to draw the bbox on.
        self.snapshot_jpeg: bytes | None = None
        self.status: dict[str, Any] = {}
        self.requests: queue.Queue[tuple[str, Any]] = queue.Queue()
        # The voice assistant's inbox, status and spoken replies (id -> mp3).
        self.assistant_inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.assistant_status: dict[str, Any] = {}
        self.tts: collections.OrderedDict[int, bytes] = collections.OrderedDict()
        self.closed = False

    def put_head(self, jpeg: bytes) -> None:
        with self.frame:
            self.head_jpeg = jpeg
            self.head_seq += 1
            self.frame.notify_all()

    def close(self) -> None:
        with self.frame:
            self.closed = True
            self.frame.notify_all()


def _page() -> bytes:
    return resources.files("dora_openarm_policy").joinpath("index.html").read_bytes()


def make_handler(shared: Shared):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, status: int = 200) -> None:
            self._send(json.dumps(obj).encode(), "application/json", status)

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/":
                self._send(_page(), "text/html; charset=utf-8")
            elif path == "/status":
                with shared.lock:
                    status = dict(shared.status)
                self._json(status)
            elif path == "/view.jpg":
                with shared.lock:
                    jpeg = shared.view_jpeg
                if jpeg is None:
                    self._send(b"", "text/plain", HTTPStatus.NO_CONTENT)
                else:
                    self._send(jpeg, "image/jpeg")
            elif path == "/snapshot.jpg":
                with shared.lock:
                    jpeg = shared.snapshot_jpeg
                if jpeg is None:
                    self._send(b"", "text/plain", HTTPStatus.NO_CONTENT)
                else:
                    self._send(jpeg, "image/jpeg")
            elif path.startswith("/tts/") and path.endswith(".mp3"):
                try:
                    speech_id = int(path[len("/tts/") : -len(".mp3")])
                except ValueError:
                    speech_id = -1
                with shared.lock:
                    audio = shared.tts.get(speech_id)
                if audio is None:
                    self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                else:
                    self._send(audio, "audio/mpeg")
            elif path == "/head.mjpg":
                self._stream()
            else:
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with shared.frame:
                        shared.frame.wait_for(lambda: shared.closed or shared.head_seq != seen, timeout=2.0)  # noqa: B023
                        if shared.closed:
                            return
                        if shared.head_seq == seen or shared.head_jpeg is None:
                            continue
                        seen, jpeg = shared.head_seq, shared.head_jpeg
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            path = self.path.split("?")[0]
            if path == "/voice":
                if not 0 < length <= MAX_VOICE_BYTES:
                    self.rfile.read(length)
                    self._json({"error": "bad audio size"}, HTTPStatus.BAD_REQUEST)
                    return
                shared.assistant_inbox.put(("voice", self.rfile.read(length)))
                self._json({"ok": True})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/text":
                text = str(body.get("text") or "").strip()
                if text:
                    shared.assistant_inbox.put(("text", text[:500]))
                self._json({"ok": True})
            elif path == "/speech_done":
                try:
                    shared.assistant_inbox.put(("speech_done", int(body.get("id"))))
                except (TypeError, ValueError):
                    self._json({"error": "bad id"}, HTTPStatus.BAD_REQUEST)
                    return
                self._json({"ok": True})
            elif path == "/vad_stats":
                if isinstance(body, dict):
                    shared.requests.put(("vad_stats", body))
                self._json({"ok": True})
            elif path == "/speed":
                try:
                    speed = float(body.get("speed"))
                except (TypeError, ValueError):
                    self._json({"error": "speed must be a number"}, HTTPStatus.BAD_REQUEST)
                    return
                if not 0.05 <= speed <= 1.0:
                    self._json({"error": "speed must be 0.05..1"}, HTTPStatus.BAD_REQUEST)
                    return
                shared.requests.put(("speed", speed))
                self._json({"ok": True})
            elif path == "/assistant/reset":
                shared.assistant_inbox.put(("reset", None))
                self._json({"ok": True})
            elif path == "/bbox":
                bbox = body.get("bbox")
                if bbox is not None:
                    try:
                        x1, y1, x2, y2 = (float(v) for v in bbox)
                    except (TypeError, ValueError):
                        self._json({"error": "bbox must be 4 numbers"}, HTTPStatus.BAD_REQUEST)
                        return
                    x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
                    y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
                    if x2 - x1 < 0.005 or y2 - y1 < 0.005:
                        self._json({"error": "bbox too small"}, HTTPStatus.BAD_REQUEST)
                        return
                    bbox = [x1, y1, x2, y2]
                shared.requests.put(("bbox", bbox))
                self._json({"ok": True})
            elif path == "/action":
                action = body.get("action")
                if action not in ACTIONS:
                    self._json({"error": f"unknown action {action!r}"}, HTTPStatus.BAD_REQUEST)
                    return
                shared.requests.put(("action", action))
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return Handler


def serve(shared: Shared, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(shared))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="policy-ui", daemon=True).start()
    return server
