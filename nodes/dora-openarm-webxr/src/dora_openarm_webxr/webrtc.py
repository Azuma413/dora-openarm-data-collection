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

"""WebXR teleoperation over WebRTC.

Everything the VR device and the node say to each other rides one peer
connection, so the page can be served by anyone -- this node over HTTPS
during development, or a hosted service in production. That is the whole
point: WebRTC authenticates itself with a self-signed certificate and an
SDP fingerprint, so a node reached this way needs no certificate of its
own and no HTTPS server.

Two data channels, split by what they can afford to lose:

* ``xr`` -- opened by the browser, **unordered and never retransmitted**.
  Carries the ``frame`` messages at the WebXR animation frame rate
  (72-120 Hz on a Quest 3). Only the newest pose is worth anything, so a
  lost frame is dropped rather than retransmitted; the next one
  supersedes it. Frames carry a ``sequence`` so the receiver can drop the
  stale ones an unordered channel occasionally delivers late.
* ``control`` -- opened by this node, reliable and ordered. Carries the
  things that must not be lost: the node pushes the view configuration
  and the calibration flag once on open, the browser sends
  ``session-start``, and the node sends back what came of a calibration
  run and the operator's HUD (openarm-quest-teleop patch). Pushing the
  configuration is what lets a page this node never served still know
  how to draw itself.
* ``state`` -- opened by this node, **unordered and never
  retransmitted** (openarm-quest-teleop patch), and **only for a view
  that draws the 3D scene** (``video.draws_scene``). Carries the
  simulated scene the page draws in 3D over passthrough, one message per
  sim-clock instant at the headset's display rate. Same trade as ``xr``
  in the other direction: only the newest scene is worth anything, so a
  message that does not fit is dropped rather than queued, and the next
  one is 14 ms away. It is a separate channel rather than the browser's
  ``xr`` one, which is page-to-node by contract and sequence-numbered for
  that one direction, and separate from ``control``, whose reliability
  would turn a slow link into a growing backlog of stale poses. In a view
  that draws no scene the channel is never created: the page would
  register no handler for it, and the messages would only spend the
  headset's airtime against the hand poses coming back.

One session at a time (openarm-quest-teleop patch): a ``session-start``
while another peer is active and still sending frames is answered with
``session-refused``; otherwise the new peer takes over, and the old one
is told ``session-ended`` (reason ``taken-over``) and closed. The HUD
and calibration results go only to the active peer.

Camera frames leave on WebRTC video tracks, one per eye, in the order
:func:`.video.eyes` gives. The eye order rides the ``control`` channel's
configuration message rather than being implied by track order, because
the mono view sends one track and the browser must not have to guess
which eye it is. Both tracks are stamped from one clock so the receiver
can line the eyes up; two clocks would let the eyes drift apart, which
in a headset is worse than either eye being late.
"""

import asyncio
import fractions
import json
import time

import av
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import VideoStreamTrack

from . import video

# RTP video clock; pts for outgoing frames are expressed in this rate.
_CLOCK_RATE = 90_000

# Public so the front-end and the tests agree on one number: the browser
# always negotiates this many recvonly video transceivers, whatever the
# view draws, because the one-shot signaling here cannot renegotiate
# later and the browser does not know the view when it makes the offer.
VIDEO_TRANSCEIVERS = 5

_STUN_URL = "stun:stun.cloudflare.com:3478"

# The ICE servers a peer is built with when the caller passes none. A
# public STUN server is enough when a direct path exists; on one LAN
# the host candidates alone connect. Behind a symmetric NAT or a
# UDP-blocking firewall only a TURN relay connects, and TURN needs
# credentials, so it cannot live in a default: the signaling service
# mints short-lived ones per session and hands them in through
# ``ice_servers``.
ICE_SERVERS = [RTCIceServer(urls=[_STUN_URL])]

# openarm-quest-teleop patch: how long the active peer must have been idle
# -- no session-start and no frame from it -- before another peer's
# session-start takes the session over, in seconds. The idle time runs from
# the later of the two, so a session that has just started is protected
# before its first frame too (a Quest session can take a while to become
# "visible" and send one). While frames flow, a second headset (or anything
# else holding the pairing token) is refused: taking over mid-motion would
# feed the clutch a different hand inside its freshness window. A peer that
# has been quiet this long -- a sleeping headset whose connection is only
# "disconnected" -- must not lock its own operator out, and by then every
# downstream freshness check has already let go of it.
TAKEOVER_IDLE = 1.0

# How long a taken-over peer is kept after it was told so, in seconds.
# Closing at once would abort the SCTP association while the
# session-ended message may still be waiting for its acknowledgement.
TAKEOVER_CLOSE_GRACE = 0.5

# A peer that has not connected this long after its answer is closed, in
# seconds. Chromium offers carry mDNS host candidates and no
# end-of-candidates, so an offer abandoned by its browser (a reload or a
# sleeping headset mid-negotiation) otherwise keeps aioice polling
# forever.
PEER_CONNECT_TIMEOUT = 15.0

# At most this many peers may be negotiating at once; further offers are
# refused (HTTP 503) instead of building peers that may never connect.
MAX_PENDING_PEERS = 4

# The reasons the page shows when its session is refused or ended.
SESSION_REFUSED_REASON = "another headset is active"
TAKEN_OVER_REASON = "taken-over"

# openarm-quest-teleop patch: how many BYTES of scene state may be waiting in
# the ``state`` channel's send buffer before the next message is dropped
# instead. SCTP has no way to say "drop the one you are holding", so this is
# what keeps a slow link showing a late scene rather than an ever older one:
# above the limit the newest message is thrown away, and the one after it goes
# out as soon as the buffer drains. Five frames of the largest scene measured
# (the compress cell: 191 joint values, 788 bytes of the binary encoding), or
# three of the same scene's JSON -- so an ordinary frame is never dropped,
# while a link that has fallen a frame or two behind stops accumulating.
#
# Read it as a LATENCY guard, not as the link's capacity. 4096 bytes is about
# 70 ms of backlog at 72 Hz of binary, and the node's own event loop being late
# for a couple of frames trips it exactly as a slow radio does. That is why
# `WebRTCServer` also keeps `state_buffered_peak`, and why the node's log line
# says how deep the buffer got rather than claiming the scene "did not fit".
STATE_BUFFER_LIMIT = 4096


class PendingPeerLimitError(RuntimeError):
    """Too many peers are still negotiating, so an offer was not answered."""


def parse_ice_servers(text: str) -> list[RTCIceServer]:
    """Parse ICE servers from JSON in the browser's own form.

    The JSON is the ``iceServers`` list an ``RTCPeerConnection`` takes
    in the browser: objects with ``urls`` (one URL or a list) and, for
    TURN, ``username`` and ``credential``. The signaling service already
    sends its browsers exactly this, so it can hand this node the very
    same list -- short-lived TURN credentials included -- with no
    translation on either side.

    Raises ValueError -- or TypeError when the JSON is not a list at
    all -- so a caller can turn a malformed configuration into a
    startup error instead of a peer that quietly cannot connect.
    """
    try:
        servers = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"not JSON: {error}") from error
    if not isinstance(servers, list):
        raise TypeError(f"not a list: {servers!r}")
    parsed = []
    for server in servers:
        if not isinstance(server, dict) or "urls" not in server:
            raise ValueError(f'not an object with "urls": {server!r}')
        parsed.append(
            RTCIceServer(
                urls=server["urls"],
                username=server.get("username"),
                credential=server.get("credential"),
            )
        )
    return parsed


class _SharedClock:
    """One timeline for every eye on a connection.

    Each eye is encoded on its own track, and tracks are paced
    independently, so nothing else keeps them together. Stamping both
    from one clock gives the receiver what it needs to present them as
    one moment.
    """

    def __init__(self) -> None:
        """Start unset; the first frame of either eye fixes the origin."""
        self._t0: float | None = None

    def timestamp(self) -> int:
        """Return the current time in RTP clock ticks since the origin."""
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        return int((now - self._t0) * _CLOCK_RATE)


class _EyeVideoTrack(VideoStreamTrack):
    """Video track that decodes one eye's JPEG frames on demand."""

    def __init__(self, eye: str, clock: _SharedClock) -> None:
        """Attach to an eye before its first frame, sharing ``clock``."""
        super().__init__()
        self._eye = eye
        self._clock = clock
        self._seen_sequence = 0
        # The MJPEG decoder is stateful, so each track owns one.
        self._decoder = av.CodecContext.create("mjpeg", "r")
        self._warned_decode = False

    async def recv(self) -> av.VideoFrame:
        """Return the eye's next camera frame, stamped from the shared clock."""
        while True:
            jpeg, self._seen_sequence = await video.wait_next(
                self._eye, self._seen_sequence
            )
            try:
                frames = await asyncio.to_thread(self._decoder.decode, av.Packet(jpeg))
            except av.error.FFmpegError:
                if not self._warned_decode:
                    self._warned_decode = True
                    print(
                        f"WARNING: the {self._eye} camera input is not decodable JPEG",
                        flush=True,
                    )
                continue
            if frames:
                break
        frame = frames[-1]
        # Wall-clock pts instead of VideoStreamTrack's fixed 30 fps pacing:
        # frames should leave when the camera produces them, at its rate.
        frame.pts = self._clock.timestamp()
        frame.time_base = fractions.Fraction(1, _CLOCK_RATE)
        return frame


class WebRTCServer:
    """The robot half of the WebXR peer connection.

    Owns the peers, the data channels and the video tracks. It knows
    nothing about poses: frame payloads go straight to the ``on_frame``
    callback, which is where ``main`` turns them into dora outputs.
    """

    def __init__(
        self,
        on_frame,
        on_session_start,
        calibration_enabled: bool = False,
        ice_servers: list[RTCIceServer] | None = None,
        on_session_end=None,
        on_log=None,
        takeover_idle: float = TAKEOVER_IDLE,
        connect_timeout: float = PEER_CONNECT_TIMEOUT,
        max_pending_peers: int = MAX_PENDING_PEERS,
        scene: dict | None = None,
        state_buffer_limit: int = STATE_BUFFER_LIMIT,
        state_channel: bool = True,
        build: str | None = None,
    ) -> None:
        """Prepare a server; no peer exists until an offer is answered.

        ``ice_servers`` is what every peer is built with; None means the
        default public STUN server. The signaling service passes its own
        list here when a direct path cannot be counted on, since a TURN
        relay only works with the short-lived credentials it mints.

        ``takeover_idle``, ``connect_timeout`` and ``max_pending_peers``
        (openarm-quest-teleop patch) default to the module constants of
        the same meaning; tests shorten them.

        ``scene`` (openarm-quest-teleop patch) is what the configuration
        message says about the 3D scene -- the exported glTF's URL, its
        hash and the model it was exported from -- or None when there is
        no scene to draw, in which case the configuration carries no
        ``scene`` key at all and the page stays on the camera panel.

        ``state_channel`` (openarm-quest-teleop patch) is whether to offer
        the unreliable ``state`` channel at all. False for a view whose
        page never reads one (``video.draws_scene``): the channel is then
        never created, :meth:`send_state` forwards nothing and counts
        nothing, and the headset's link carries only what it draws.

        ``build`` (openarm-quest-teleop patch) is what the node says it is
        in the configuration message -- its version and the hash of the page
        it serves -- so a page kept in a headset's HTTP cache can notice that
        it is not the page this node serves, and say so in its own log. None
        leaves the key out.

        ``on_log`` (openarm-quest-teleop patch) is called with (payload,
        peer name) for every ``{"type": "log"}`` message on any peer's
        ``control`` channel -- the headset's own error log, coming back the
        reliable way while there is still a peer to carry it. None ignores
        them, which is what a node with no log file does.
        """
        self._on_frame = on_frame
        self._on_session_start = on_session_start
        self._on_session_end = on_session_end
        self._on_log = on_log
        self._build = build
        self._active_peer = None
        self._calibration_enabled = calibration_enabled
        self._ice_servers = ice_servers
        self._pcs: set = set()
        # Each peer's open control channel, so a message can go to one
        # peer rather than to every browser (openarm-quest-teleop patch).
        self._controls: dict = {}
        # The same for the unreliable ``state`` channel, plus what the
        # configuration says about the scene it carries.
        self._states: dict = {}
        self._scene = scene
        # Public: the node's log line names it, because the number is what
        # the drop test actually measures.
        self.state_buffer_limit = state_buffer_limit
        self._state_channel = state_channel
        # Scene states sent to a headset and dropped for want of room, for
        # the node's own log: a link that cannot keep up should say so.
        self.state_sent = 0
        self.state_dropped = 0
        # Bytes actually handed to the channel, and the deepest the send
        # buffer was ever seen (openarm-quest-teleop patch). The drop count
        # alone cannot tell a link that is genuinely too slow from an event
        # loop that was busy for one frame; the peak against the limit can.
        self.state_bytes = 0
        self.state_buffered_peak = 0
        self._running = True
        # The last HUD sent, without its haptic, repeated to a peer when it
        # becomes the active one: the HUD is sent on change, so a page that
        # reconnects would otherwise show nothing until the next one.
        self._hud: dict | None = None
        # openarm-quest-teleop patch: session takeover and peer hygiene.
        self._takeover_idle = takeover_idle
        self._connect_timeout = connect_timeout
        self._max_pending_peers = max_pending_peers
        # When the active peer's session started and when it last delivered
        # a frame, on _clock; None with no active peer, and the frame time
        # also until its first frame. A takeover needs takeover_idle since
        # the later of the two.
        self._session_started_at: float | None = None
        self._last_frame_at: float | None = None
        self._clock = time.monotonic
        # Peers that have not reached "connected" yet.
        self._pending: set = set()
        # A short name per peer for the log records it forwards
        # (openarm-quest-teleop patch). The peer's address is not reachable
        # from here without digging into aiortc's ICE internals, and two
        # headsets are a supported configuration, so the log needs something
        # that tells one from the other. It is also what a page cannot forge.
        self._peer_names: dict = {}
        self._peers_seen = 0
        # Watchdogs and delayed closes, kept so they are not garbage
        # collected mid-flight and can be cancelled on close.
        self._tasks: set = set()

    @property
    def running(self) -> bool:
        """Whether the node should keep serving.

        True until :meth:`stop`. In WebRTC-only mode it also turns False
        when the one peer goes away, since with no HTTP server no other
        browser can ever take its place.
        """
        return self._running

    def stop(self) -> None:
        """Stop serving and close every peer."""
        self._running = False

    async def close(self) -> None:
        """Close every peer connection."""
        for task in list(self._tasks):
            task.cancel()
        self._end_session(self._active_peer)
        pcs = list(self._pcs)
        self._pcs.clear()
        self._controls.clear()
        self._states.clear()
        self._pending.clear()
        self._peer_names.clear()
        for pc in pcs:
            await _close_peer(pc)

    def send_control(self, payload: dict) -> None:
        """Send a message to the active browser on ``control``.

        Used for what came of a calibration run: the operator cannot see
        the node's output while wearing a headset. Only the active peer
        is told (openarm-quest-teleop patch): the run came from its
        frames, and another page must not show state it did not produce.
        """
        self._send(self._active_peer, payload)

    def send_state(self, payload) -> bool:
        """Send one scene state to the active headset, or drop it.

        openarm-quest-teleop patch. ``payload`` is passed to the channel
        as it came from the sim node -- ``str`` for the JSON encoding,
        ``bytes`` for the binary one -- so the node never parses a
        message it only forwards.

        Nothing is sent, and False reported, when there is no active
        session: a page that is connected but not teleoperating is not
        drawing this either, and a second headset must not be shown the
        scene it is not driving. That case is not counted as a drop --
        with nobody watching there is nothing to be behind on.

        A message offered while more than ``state_buffer_limit`` BYTES are
        still waiting in the channel's send buffer is counted and dropped.
        That is the whole point of an unreliable channel: a headset the
        link cannot keep up with sees a frame or two go missing, not an
        ever older scene arriving on time. It is a LATENCY guard, not a
        statement about the link's capacity -- the node's own event loop
        being late for one frame trips it just as surely -- which is why
        the peak backlog is kept alongside the count.

        Nothing is sent and nothing counted when this view opens no state
        channel at all (``state_channel=False``).
        """
        channel = self._states.get(self._active_peer) if self._active_peer else None
        if channel is None or channel.readyState != "open":
            return False
        buffered = channel.bufferedAmount
        self.state_buffered_peak = max(self.state_buffered_peak, buffered)
        if buffered > self.state_buffer_limit:
            self.state_dropped += 1
            return False
        channel.send(payload)
        self.state_sent += 1
        self.state_bytes += (
            len(payload.encode("utf-8")) if isinstance(payload, str) else len(payload)
        )
        return True

    def send_hud(self, payload: dict) -> None:
        """Send a HUD message now, and to the next peer to become active.

        The haptic is only sent now: a pulse replayed on reconnect would
        buzz for an event that happened before the operator was back.
        """
        self._hud = {**payload, "haptic": None}
        self.send_control(payload)

    async def answer(self, offer_sdp: str) -> str:
        """Answer one offer and return the bare answer SDP.

        openarm-quest-teleop patch: raises :class:`PendingPeerLimitError`
        without building a peer when too many are still negotiating. A
        peer whose offer cannot be answered is closed and forgotten before
        the error propagates -- aiortc raises all sorts of exceptions on a
        malformed SDP -- and one that is answered but never connects is
        closed after ``connect_timeout``.
        """
        if len(self._pending) >= self._max_pending_peers:
            raise PendingPeerLimitError(
                f"{len(self._pending)} WebRTC peers are still negotiating"
            )
        pc = self._create_peer()
        try:
            answer_sdp = await self._answer(
                pc, RTCSessionDescription(sdp=offer_sdp, type="offer")
            )
        except BaseException:
            await self._discard_peer(pc)
            raise
        self._spawn(self._close_if_unconnected(pc, self._connect_timeout))
        return answer_sdp

    async def negotiate_oneshot(
        self,
        offer_sdp: str,
        answer_host: str,
        answer_port: int,
        connect_timeout: float = 60.0,
    ) -> None:
        """Answer a single offer handed in at startup, with no HTTP server.

        The offer arrives out of band (a command-line argument or an
        environment variable) as the bare SDP -- its type is always
        ``offer`` -- and the answer goes back as the bare answer SDP
        written to the TCP socket the caller is listening on. This is the
        WebRTC-only mode: another service hosts the page and brokers
        signaling, and this node just runs the peer for its lifetime.

        After sending the answer, waits up to ``connect_timeout`` seconds
        for the peer to connect. If it never does, the peer is closed and
        this raises :class:`RuntimeError`, so a stranded node exits
        instead of holding a dead connection forever.
        """
        pc = self._create_peer()
        established = asyncio.get_running_loop().create_future()

        @pc.on("connectionstatechange")
        def on_connectionstatechange() -> None:
            if pc.connectionState == "connected":
                if not established.done():
                    established.set_result(True)
            elif pc.connectionState in ("failed", "closed"):
                if not established.done():
                    established.set_result(False)
                # No HTTP server means no replacement browser, ever.
                self._running = False

        answer_sdp = await self._answer(
            pc, RTCSessionDescription(sdp=offer_sdp, type="offer")
        )

        _reader, writer = await asyncio.open_connection(answer_host, answer_port)
        writer.write(answer_sdp.encode("utf-8"))
        await writer.drain()
        writer.write_eof()
        writer.close()
        await writer.wait_closed()

        try:
            connected = await asyncio.wait_for(established, connect_timeout)
        except TimeoutError:
            connected = False
        if not connected:
            await self._discard_peer(pc)
            raise RuntimeError(
                f"no WebRTC connection within {connect_timeout:g}s of the answer"
            )

    async def _answer(self, pc: RTCPeerConnection, offer: RTCSessionDescription) -> str:
        """Consume an offer and return the answer SDP.

        The video tracks are attached between setting the remote and the
        local description, so they bind to the transceivers the browser
        offered rather than adding new ones the browser never asked for.

        aiortc's ``setLocalDescription`` waits for ICE gathering to
        finish, so the returned SDP already carries every candidate:
        signaling is a single exchange with no trickle.
        """
        await pc.setRemoteDescription(offer)
        clock = _SharedClock()
        for eye in video.eyes():
            pc.addTrack(_EyeVideoTrack(eye, clock))
        await pc.setLocalDescription(await pc.createAnswer())
        return pc.localDescription.sdp

    def _create_peer(self) -> RTCPeerConnection:
        """Build a peer with its data channels wired up."""
        ice_servers = self._ice_servers
        if ice_servers is None:
            ice_servers = ICE_SERVERS
        configuration = RTCConfiguration(iceServers=list(ice_servers))
        pc = RTCPeerConnection(configuration)
        self._pcs.add(pc)
        self._pending.add(pc)
        self._peers_seen += 1
        self._peer_names[pc] = f"peer-{self._peers_seen}"

        # Opened by this node, like the page's own configuration: a page
        # this node never served has no other way to learn it.
        #
        # Created FIRST, and before the scene's channel below: the stream
        # id, and with it the order the browser is told about the
        # channels, follows creation order, and a page (or a test) that
        # takes the first channel it is offered as the control one was
        # right before this node opened a second channel of its own.
        control = pc.createDataChannel("control")

        @control.on("open")
        def on_control_open() -> None:
            self._controls[pc] = control
            control.send(
                json.dumps(
                    {
                        "type": "configuration",
                        "view_configuration": video.view_configuration(),
                        "calibration": self._calibration_enabled,
                        # Which eye each video track carries, in track
                        # order. The mono view sends one track, so the
                        # browser cannot infer the eye from the count.
                        "eyes": video.eyes(),
                        # openarm-quest-teleop patch: the 3D scene, or
                        # nothing at all when there is none to draw.
                        **({"scene": self._scene} if self._scene else {}),
                        # openarm-quest-teleop patch: which build of the page
                        # this node serves, for a page that came out of a
                        # cache instead.
                        **({"build": self._build} if self._build else {}),
                    }
                )
            )
            # No HUD here (openarm-quest-teleop patch): it describes the
            # active session, so it is replayed when this peer starts one.

        @control.on("close")
        def on_control_close() -> None:
            self._controls.pop(pc, None)
            self._end_session(pc)

        @control.on("message")
        def on_control_message(message: object) -> None:
            self._handle_control_message(message, pc)

        # openarm-quest-teleop patch: the 3D scene's own channel, opened
        # here because the node is the one that sends on it. Unordered
        # and never retransmitted, like the browser's ``xr``: a scene
        # that could not go out is stale before it could be resent.
        # Opened after ``control`` so it is the second channel the page
        # hears about, never the first (see the comment there).
        #
        # Not opened at all when this view draws no 3D scene: the page
        # would register no handler for it, so every message would cross
        # the headset's link to be decoded and thrown away, and on the
        # half-duplex radio that costs the hand poses their airtime.
        if self._state_channel:
            state = pc.createDataChannel("state", ordered=False, maxRetransmits=0)

            @state.on("open")
            def on_state_open() -> None:
                self._states[pc] = state

            @state.on("close")
            def on_state_close() -> None:
                self._states.pop(pc, None)

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            if channel.label != "xr":
                return

            @channel.on("message")
            def on_message(message: object) -> None:
                self._handle_frame_message(message, pc)

            @channel.on("close")
            def on_close() -> None:
                self._end_session(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            # A browser that leaves without closing -- a reload, a
            # sleeping headset -- is only ever noticed here. Left alone,
            # the peer's video encoders would keep running for nobody
            # until the node exits, and every reload would stack another
            # one on. Only the terminal states count: "disconnected" is
            # also what a brief network blip looks like, and that can
            # still recover.
            if pc.connectionState == "connected":
                self._pending.discard(pc)
            elif pc.connectionState in ("failed", "closed"):
                await self._discard_peer(pc)

        return pc

    async def _discard_peer(self, pc) -> None:
        """Forget a peer, ending its session if it had one, and close it."""
        self._end_session(pc)
        self._pcs.discard(pc)
        self._pending.discard(pc)
        self._controls.pop(pc, None)
        self._states.pop(pc, None)
        self._peer_names.pop(pc, None)
        await _close_peer(pc)

    def _spawn(self, coroutine) -> None:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _close_if_unconnected(self, pc, timeout: float) -> None:
        # openarm-quest-teleop patch (NET-2): the watchdog for a peer whose
        # browser went away before ICE finished.
        await asyncio.sleep(timeout)
        if pc in self._pcs and pc in self._pending:
            print(
                f"closing a WebRTC peer that did not connect within {timeout:g}s",
                flush=True,
            )
            await self._discard_peer(pc)

    async def _close_later(self, pc, delay: float) -> None:
        await asyncio.sleep(delay)
        if pc in self._pcs:
            await self._discard_peer(pc)

    def _send(self, pc, payload: dict) -> None:
        """Send ``payload`` on ``pc``'s control channel, if it is open."""
        channel = self._controls.get(pc) if pc is not None else None
        if channel is not None and channel.readyState == "open":
            channel.send(json.dumps(payload))

    def _end_session(self, pc) -> None:
        if pc is not None and pc is self._active_peer:
            self._active_peer = None
            self._session_started_at = None
            self._last_frame_at = None
            if self._on_session_end is not None:
                self._on_session_end()

    def peer_name(self, pc) -> str:
        """Return the short name this peer is known by in the headset's log."""
        return self._peer_names.get(pc, "peer-?")

    def _handle_control_message(self, message: object, pc) -> None:
        payload = _decode(message)
        if payload is None:
            return
        kind = payload.get("type")
        if kind == "session-start":
            self._start_session(pc)
        elif kind == "log" and self._on_log is not None:
            # openarm-quest-teleop patch: the headset's own log, riding the
            # reliable channel while there is one. Guarded because this runs
            # in aiortc's callback: a record the sink chokes on must cost
            # that record, not the session.
            try:
                self._on_log(payload, self.peer_name(pc))
            except Exception as error:  # noqa: BLE001
                print(
                    f"WARNING: a headset log message was not recorded: {error!r}",
                    flush=True,
                )

    def _start_session(self, pc) -> None:
        """Make ``pc`` the active peer, unless another one is still in use.

        openarm-quest-teleop patch: upstream let whichever peer last sent
        session-start become active without a word to the one it
        displaced, whose page kept showing a session and a HUD that no
        longer reached the robot. Another peer is in use until it has been
        idle for ``takeover_idle``: the time since the later of its
        session-start and its last frame. Counting only frames let a
        session that had not sent its first one yet be taken over at once.
        """
        active = self._active_peer
        now = self._clock()
        if active is not None and active is not pc:
            used = [self._session_started_at, self._last_frame_at]
            last_used = max((at for at in used if at is not None), default=None)
            if last_used is not None and now - last_used <= self._takeover_idle:
                print(
                    "refused a session-start: another headset is active",
                    flush=True,
                )
                self._send(
                    pc, {"type": "session-refused", "reason": SESSION_REFUSED_REASON}
                )
                return
            print("a new session took over from an idle headset", flush=True)
            self._end_session(active)
            self._send(active, {"type": "session-ended", "reason": TAKEN_OVER_REASON})
            self._spawn(self._close_later(active, TAKEOVER_CLOSE_GRACE))
        self._active_peer = pc
        self._session_started_at = now
        self._last_frame_at = None
        self._on_session_start()
        if self._hud is not None:
            self._send(pc, self._hud)

    def _handle_frame_message(self, message: object, pc) -> None:
        if pc is not self._active_peer:
            return
        payload = _decode(message)
        if payload is None:
            return
        if payload.get("type") == "frame":
            self._last_frame_at = self._clock()
            self._on_frame(payload)


async def _close_peer(pc) -> None:
    """Close a peer, reporting rather than raising a failure.

    aiortc can fail to close a peer whose offer it only half applied
    (openarm-quest-teleop patch), and one such peer must not stop the
    others from being closed or the answer error from reaching the caller.
    """
    try:
        await pc.close()
    except Exception as error:  # noqa: BLE001
        print(f"WARNING: closing a WebRTC peer failed: {error!r}", flush=True)


def _decode(message: object) -> dict | None:
    """Parse one channel message, or None if it is not a JSON object.

    A malformed payload is ignored rather than fatal, so a single bad
    message cannot take the transport down mid-session.
    """
    if not isinstance(message, str):
        return None
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
