# Local patches to dora-openarm-webxr

Against upstream `52cb5bb` (see `UPSTREAM.md`). Patches 1-17 were written for
openarm-quest-teleop and copied here unchanged; their text still names that
project's `oqt run`, which in this repository is the dataflow plus
`scripts/check-dataflow` (see `docs/teleop.md`). Patches 18-19 were added in
this repository. Every changed spot carries a comment saying why;
new code is marked "openarm-quest-teleop patch". The page files were formatted
with Biome 2.5.13 and the Python with Ruff, as upstream's pre-commit does.

## 1. No ICE servers by default on the page

`connection.js` built every peer with `stun:stun.cloudflare.com:3478` and waits
for ICE gathering to finish before posting the offer. On a LAN the host
candidates alone connect, and with no route to the STUN server Chromium held the
offer back: in the browser test the unpatched page posted its offer about 40 s
after loading. The page now builds its peer with `iceServers: []`, unless its URL
names servers with `?ice=stun:host:port` (comma-separated or repeated; they
become one `{urls: [...]}` entry), and `connect({ iceServers })` overrides both
for a page that brokers its own TURN credentials. The node's own default is
unchanged (public STUN when `ICE_SERVERS` is unset, as upstream's tests expect);
the dataflow sets `ICE_SERVERS='[]'`, which the `--ice-servers` help and the
README now recommend for a LAN. `connect()` also closes its peer when the
attempt fails, since the page now reconnects (patch 2).
Files: `src/dora_openarm_webxr/static/connection.js`, `src/dora_openarm_webxr/main.py` (help text), `README.md`.

## 2. Start reconnects after a session ended

Ending a session closes the connection, and upstream connected once per page
load, so a second Start opened a session that sent nothing. `ar.js` now opens
the connection in `open()` (panels, calibration panel and handlers rebuilt for
each connection) behind `ensureConnected()`, which reuses an open connection or
one being opened. `onSessionEnd` reconnects straight away, so the next Start
finds an open connection and requests the session within the click's user
activation; if there is no connection yet, Start waits for it first (quick with
no ICE servers, but a slow node could outlast the activation, in which case the
status says so and the next Start works). The Start button is wired on load, so
it also retries a node that was not up when the page loaded, and a close from a
connection that has already been replaced is ignored. The empty `<pre id="log">`
under Start now shows the connection state ("Connected. Press Start.", "Lost the
node...", "Cannot reach the node (...)").
Files: `src/dora_openarm_webxr/static/ar.js`, `src/dora_openarm_webxr/static/connection.js`.

## 3. No frames while not visible, no emulated poses, pause after a recenter

`onFrame` sends nothing while `session.visibilityState !== "visible"`: under
the system menu (`visible-blurred`) the operator is not teleoperating, and when
`hidden` the poses are stale. The node's `POSE_TIMEOUT` then holds the hands
and the clutch goes stale. A viewer pose with `emulatedPosition` is not sent
(every hand position is made relative to it, so the node holds both hands). A
controller whose aim pose has `emulatedPosition` (IMU only, out of the cameras'
sight) sends no pose, trigger or grip, so the node stops that hand and the clutch
releases it. Its A/B/X/Y buttons and thumbstick are still sent, because they
move nothing and the operator may need them to end an episode with that hand
out of view. Not in DESIGN.md's list but in the same spirit: the `local`
reference space's `reset` event (recenter) pauses frames for 300 ms, longer than
the clutch's 100 ms freshness and the pose timeout the generated flows set, so a recenter
releases the clutch instead of jumping the target by the origin shift.
Files: `src/dora_openarm_webxr/static/ar.js`.

## 4. Any number of controllers, no hands

The `session.inputSources.length < 2` gate is gone: whatever controllers exist
send their pose, trigger, grip, buttons and joystick. Upstream sent nothing
below two, so one controller asleep froze both arms and the episode buttons. Input
sources without a gamepad or with a `hand` (hand tracking, or hands tracked
alongside controllers on Quest 3) are skipped. Hands have no trigger to drive a
gripper, and a hand could overwrite the controller of the same handedness.
Files: `src/dora_openarm_webxr/static/ar.js`.

## 5. HUD input, head-locked text panel and haptics

New node input `hud`: a utf8 JSON object `{"lines": [str], "recording": bool,
"haptic": "right"|"left"|"both"|null}` (DESIGN.md 10.3). `main._parse_hud`
validates it into `{type: "hud", lines, recording, haptic}`: at most 5 lines of
80 characters, an unknown `haptic` becomes null, and a malformed message is
reported and dropped. `WebRTCServer.send_hud` sends it on the `control` channel
and keeps a copy with `haptic: null`, which is sent again later (since patch 9
only the active session's browser gets the HUD, and the copy is sent when a
browser's session starts rather than when its control channel opens). The HUD is sent on
change, so a reconnected page would otherwise be blank, and a replayed pulse
would buzz for an old event. Each message is the whole HUD: no `lines` clears
it. On the page, `connection.js` routes `hud` messages to `onHud` (keeping one
that arrives before the handler is set), and the new `hud.js` draws the lines on
a 1024x384 canvas texture on a head-locked quad (0.5 m wide at 1.1 m, 0.24 m
above the middle of the view), modelled on `instructions.js`. The first line is
bold, and the panel gets a red frame and red first line while `recording`. It is
drawn after the camera and instruction panels with alpha blending in both
`immersive-ar` and `immersive-vr`, clears the frame itself when nothing else
drew, and hides when there are no lines. `haptic` pulses the named controllers
with `gamepad.hapticActuators[0].pulse(0.6, 80)`, falling back to
`vibrationActuator.playEffect("dual-rumble")`, and skips controllers that have
neither.
Files: `src/dora_openarm_webxr/main.py`, `src/dora_openarm_webxr/webrtc.py`, `src/dora_openarm_webxr/static/hud.js` (new), `src/dora_openarm_webxr/static/connection.js`, `src/dora_openarm_webxr/static/ar.js`, `README.md`.

## 6. Control input quits the node

New node input `control` (utf8). `"quit"` (surrounding whitespace ignored)
breaks the dora loop the same way `STOP` does, so the existing shutdown path
runs: the `finally` sends `command=quit` if `--quit-button` is set and stops
uvicorn, then `_main_hosted` closes every peer (the page sees its control channel
close and ends its session) and waits for uvicorn to finish. Other commands
(`reset`) are ignored, since this node has nothing to reset.
Files: `src/dora_openarm_webxr/main.py`, `README.md`.

## 7. One Euro parameters from the environment

`--one-euro-min-cutoff` / `ONE_EURO_MIN_CUTOFF` (default 2.0 Hz) and
`--one-euro-beta` / `ONE_EURO_BETA` (default 0.04) replace the hard-coded values
in `_ConnectionState`. `d_cutoff` stays 1.5. The defaults are passed as strings,
so argparse converts the environment value and a malformed one is a usage error.
A cutoff that is not finite and positive, or a beta that is not finite and
non-negative, is refused at startup.
Files: `src/dora_openarm_webxr/main.py`, `README.md`.

## 8. Pairing token on `POST /offer`

The node listens on every interface and upstream answered any `POST /offer`, so
anything that reached the port could start a session and, as the active peer,
drive the robot. `PAIRING_TOKEN` (environment only, so the secret never shows in
the process list; surrounding whitespace is dropped and an empty value means
unset) makes `/offer` require the JSON body `{"sdp": ..., "token": ...}` with a
token equal under `secrets.compare_digest` (compared as UTF-8 bytes, so an odd
token is a refusal rather than a `TypeError`). A missing, wrong or unreadable
token gets HTTP 403 before any peer is built, and the node logs the refusal with
the client address. Unset, every offer is accepted as upstream did, and the first
one logs a warning. `connection.js` reads the token from the page URL's fragment,
`.../#k=<token>`, which never leaves the browser in a request or a Referer, keeps
it in `sessionStorage` so a reload that lost the fragment still pairs, and sends
it with the offer; a 403, 400 or 503 reaches the status line with what it means.
A page that passes its own `signal` to `connect()` is unaffected.
Files: `src/dora_openarm_webxr/main.py`, `src/dora_openarm_webxr/static/connection.js`, `README.md`.

## 9. One session at a time: refusal, explicit takeover, HUD to the active peer

Upstream made whichever peer last sent `session-start` the active one without a
word: the displaced page kept its immersive session and its HUD (sent to every
browser) while its frames were dropped, and the newcomer's first frame could land
inside the clutch's 100 ms freshness window, which turned the distance between two
hands into an arm jump. `WebRTCServer` now refuses a `session-start` while another
peer is active and has started its session or delivered a frame within
`TAKEOVER_IDLE` (1.0 s), i.e. while `now - max(session_started_at, last_frame_at)
<= TAKEOVER_IDLE`. Counting the session start as use matters: a session that has
not sent its first frame yet (a Quest session not "visible" yet) was otherwise open
to takeover at once. The newcomer gets `{type: "session-refused", reason: "another
headset is active"}` and nothing else changes; refused attempts do not count as the
active peer's use. Otherwise the newcomer takes over: the old session is ended
(`on_session_end`), the old peer is sent `{type: "session-ended", reason:
"taken-over"}` and is closed `TAKEOVER_CLOSE_GRACE` (0.5 s) later, so the reliable
message is not lost to the SCTP abort that closing sends. The idle limit keeps a
sleeping headset, whose connection is only "disconnected", from locking its own
operator out (a peer idle for more than a second since both its session start and
its last frame can be taken over), and it means a takeover always follows a second
without a session start or a frame, by which time every downstream freshness check
has let go. `send_hud` and
`send_control` (calibration results) go to the active peer only, and the last HUD
(without its haptic) is replayed when a peer becomes active. On the page,
`connection.js` hands both messages to `onSessionNotice`, and `ar.js` ends the XR
session and keeps the reason ("The node refused the session: another headset is
active." or "Another headset took over the session.") above the status line
through the reconnect that follows, until a session starts again.
Files: `src/dora_openarm_webxr/webrtc.py`, `src/dora_openarm_webxr/static/connection.js`, `src/dora_openarm_webxr/static/ar.js`, `README.md`.

## 10. Peers that cannot be answered or never connect

`answer()` registered its peer before applying the offer and nothing removed it
when aiortc raised, and a Chromium offer abandoned before its checks ran (mDNS
host candidates, no end-of-candidates) kept aioice polling for as long as the node
ran. A peer whose offer cannot be answered is now closed and forgotten before the
error propagates; aiortc reports a malformed SDP with `ValueError`, `KeyError`,
`AssertionError`, `AttributeError`, `IndexError`, `RuntimeError` and more (found
by answering truncated offers), and can fail to close such a peer, which is logged
rather than raised. A peer that has not reached "connected" `PEER_CONNECT_TIMEOUT`
(15 s) after its answer is closed. At most `MAX_PENDING_PEERS` (4) peers may be
negotiating; `answer()` raises `PendingPeerLimitError` for more, without building
a peer, and connected peers do not count. `/offer` answers 400 to a body that is
not an object with an `sdp` string and to any error answering the offer, and 503
to the pending limit. `WebRTCServer()` takes `takeover_idle`, `connect_timeout`
and `max_pending_peers`, defaulting to the module constants, for tests.
Files: `src/dora_openarm_webxr/webrtc.py`, `src/dora_openarm_webxr/main.py`.

## 11. A session that resolves after the connection was lost

`requestSession` can take seconds on a Quest (the first-use permission prompt,
the session starting up). A connection lost in that window ran its close handler
with no session to end, so the session that then resolved sent nothing and never
ended, under a HUD still showing the last status. `onStart` now keeps the
connection it requested the session on, and if that is gone or replaced when the
session resolves, ends the session at once, reconnects (its end listener is not
attached yet, so nothing else would) and keeps "Lost the node while starting the
session." on the page. The close handler clears the HUD, and when it ends a
running session keeps "Lost the node during the session." the same way.
Files: `src/dora_openarm_webxr/static/ar.js`.

## 12. Filter reset after input gaps

A hand whose pose, trigger or head pose went missing had its One Euro filter
suspended, which keeps the pose from before the gap, so when tracking returned the
published pose slid toward the hand at up to `MAX_LINEAR_SPEED`. A clutch that let
go during the gap re-anchored on that slide and carried the rest of it into the
arm: up to 25 cm with the hand still. `--smoother-reset-gap` /
`SMOOTHER_RESET_GAP` (seconds, default 0.1, the clutch's freshness window; must be
finite and positive) now separates the two cases. A tick where a hand cannot be
published resets its filter if the hand was last received more than the gap ago,
and suspends it otherwise. A tick that publishes resets the filter first if the
frame it uses was received more than the gap after the frame the hand was last
published from, which catches frames that stopped while ticks kept publishing the
last one, a calibration run, and stalled ticks. The pose timeout clearing the
frame and a session end reset as well, and so does a frame that arrives more than
`POSE_TIMEOUT` after the previous accepted one, whether or not a tick ran in
between to clear the frame: the clutch times its grip gap sample to sample, so
with 5 ms ticks a stall of 100-105 ms disarmed it while webxr, which no tick told
of the timeout, kept sliding the filter (hand moved 8 cm during the stall, then
still: 14-66 mm of target drift after the clutch engaged again, depending on how
soon the grip closed; 0 mm with this reset). The generated flows keep `POSE_TIMEOUT <= fresh_s <
SMOOTHER_RESET_GAP` (0.09, 0.1, 0.12; `docs/DESIGN.md` 2.1): any stall long enough to
disarm the clutch then also resets the filter, including stalls between 0.1 and
0.12 s, which the reset gap alone suspends. `POSE_TIMEOUT` sits 10 ms below the
clutch's window because webxr times frame arrivals and the clutch grip deliveries:
with both at 0.1 s, a stall webxr saw as 99 ms reached the clutch as 101 ms, which
disarmed it without a reset (66-70 mm slides with 4 ms of delivery jitter); stalls of
90-100 ms that the clutch stays engaged through now reset as well, which only steps
the target to where the hand is. A hand counts as received in a frame that
carries its pose and trigger together with a head pose, timed on the
`perf_counter` clock of the pose timeout. Shorter gaps of a hand missing from
frames that keep arriving still suspend: the clutch stays engaged across them. The flows
`oqt run` generates set 0.12 s, a little above the clutch's 0.1 s, so that
webxr and the clutch, which time the same gap on clocks a few ms apart, never
disagree in the direction that matters for patch 13 (`docs/DESIGN.md` 2.1).
Files: `src/dora_openarm_webxr/main.py`, `README.md`.

## 13. Latched neck pivot

Hand positions were the controller minus the neck pivot of every frame, so with
the grip held and the controller still in the room, leaning 15 cm moved the
target 15 cm. `--pivot-mode` / `PIVOT_MODE` is `live` (default, upstream) or
`latched`. Latched subtracts a pivot position latched from the first head pose of
a session, and again from the first head pose after a gap in head poses longer
than `SMOOTHER_RESET_GAP`. Such a gap stops both hands' poses (no head pose, no
hand pose) or their grips (no frames at all), so a clutch with that freshness
window has released both arms before the pivot moves, and patch 12 resets the
filters with it, so nothing slides. A recenter is such a gap, since the page
pauses frames for 300 ms. The orientation is unchanged, and the neck pivot offset
applies as it was when the pivot latched.
Files: `src/dora_openarm_webxr/main.py`, `README.md`.

## 14. The 3D scene: a `scene_state` input on its own unreliable channel

The page drew a flat panel of a simulated camera; to draw the cell as real
objects over passthrough it needs the poses, not pixels. The node gains a
`scene_state` input (queue 1) carrying every joint value of the model the sim
loaded — a utf8 JSON string, or a uint8 array with the sim's
`--scene-state-format binary` — and forwards each message untouched to the
**active** peer only, on a data channel this node opens called `state`,
`ordered=False, maxRetransmits=0`. It is a separate channel from the browser's
own `xr`: that one carries poses the other way, and only the newest scene is
worth anything in this one. `control` is still created first, so it stays the
first channel the page is offered.

**The channel is only opened for a view that draws the scene.** `video.draws_scene()`
(view in `scene` / `scene+panel`, kept in step with ar.js `SCENE_VIEWS`) decides
at startup, and the node says which of the three cases it is in one line. On
`mono` / `stereo` / `none` the page registers no handler for `state` at all, so
`WebRTCServer(state_channel=False)` never calls `createDataChannel("state")`,
`send_state()` finds no channel and sends and counts nothing, and a `scene_state`
input is dropped where it arrives — before parsing — with one warning naming the
rate and what to do about it. This is belt and braces with oqt's own side
(`--scene-state-hz` follows `--view`, and `validate` refuses the pairing): a
hand-written flow cannot spend the link either. Measured against the real node
and a real page: a `mono` flow fed 1999 messages of 1457 B forwarded 0, and the
page decoded 0.

Backpressure is a drop, not a queue: SCTP cannot be told to discard what it is
holding, so a message offered while more than `STATE_BUFFER_LIMIT` (4096 bytes)
is still waiting is thrown away and counted. That limit is a **latency guard,
not a capacity verdict** — about 70 ms of backlog, five frames of the binary
encoding of the largest scene measured (788 B) or three of its JSON. The node's
own event loop trips it as readily as a slow radio does, so the report now says
what it measured rather than blaming the link: how many were forwarded, over
what wall time and at what rate, the average bytes each, and — only when any
were dropped — the byte limit they were compared against and the peak backlog
reached. (The old line, "N of the last 4096 scenes did not fit the headset link
and were dropped", read as a verdict on the radio, and the 4096 in it was the
batch size, not the limit it tested.) A healthy stream says so too:
`scene_state: 4096 forwarded in 61.5 s (66.6 Hz), 1445 B each; none dropped`.
A headset on a slow link sees a frame go missing rather than an ever older
scene arriving on time. Nothing is sent, and nothing counted, while no session
is active. A message over 1 MB, or of any other shape, is dropped with one line
rather than taking teleoperation down.

`--scene-glb` / `SCENE_GLB` names the exported glTF of that same scene. It is
served at `/scene/scene.glb` (one file by name, declared before the static
mount, since the run directory it sits in also holds the pairing token), and
the configuration message gains a `scene` block with its URL, its blake2b hash
— what lets the page refuse a stale cached copy — its size, and the `nq` and
body list read from the export's own JSON chunk. No file, or an unreadable one,
means no `scene` key at all and a page that keeps to the camera panel.

**The asset goes over the wire gzipped.** `_configure_scene` already read the
whole export to hash it, so it compresses it there too (level 6, `mtime=0`) and
keeps the result in memory: once at node startup, never per request and not
through a middleware, because the file cannot change under a running node and
this node shares its host with the simulator -- per-request deflate would come
out of the loop that answers the operator's hands. `_scene_endpoint` picks the
representation from `Accept-Encoding` (honouring `gzip;q=0`, preferring an
explicit `gzip` over `*`) and answers with `Content-Encoding: gzip`,
`Vary: Accept-Encoding` and a **distinct strong ETag** `"<hash>-gzip"` -- RFC
9110 8.8.3: different bytes, different entity tag, or a shared cache hands the
compressed copy to a client that refused it. A client that will not inflate
still gets the same `FileResponse` as before. `If-None-Match` is handled with
weak comparison, so the revalidation after `max-age=3600` is a 304 rather than
megabytes again. Compression is keyed by the hash, so a scene swapped in under
the endpoint can never be served the previous file's bytes, and a file that
fails to compress or does not shrink prints one line and is served plain --
never a black headset. Measured: the demo cell 4,716,792 -> 2,022,637 B
(42.9 %), the compress cell 3,930,400 -> 1,639,480 B (41.7 %, byte for byte the
`gzip_bytes` the exporter's own manifest predicted); level 9 buys 545 B
(0.03 %) for a third more CPU and level 1 gives back 4.0 % to save 26 ms. The
whole of `_configure_scene` (read + blake2b + gzip) costs 75 ms, once. The page
needs no change: the browser inflates transparently and `GLTFLoader` sees the
same bytes. One known gap: a ranged request that also accepts gzip now gets a
full 200 rather than a 206, because the gzip path is a plain `Response`. No
browser range-requests a glTF, and the identity path still supports ranges.
Files: `src/dora_openarm_webxr/main.py`, `src/dora_openarm_webxr/webrtc.py`.

## 15. The 3D view: the scene drawn around the operator, over passthrough

The page drew one flat panel of a simulated camera. `view: scene` and
`view: scene+panel` draw the cell itself instead -- the arms, the table, the
care receiver, whatever the sim loaded -- as real objects at real size around
the operator, over passthrough in `immersive-ar`. `mono`, `stereo` and `none`
are untouched and load none of this: `ar.js` imports `scene3d.js` with a
dynamic `import()` only for a 3D view, so a camera view fetches no three.js.

**three.js is vendored** under `static/three/` (0.179.1, MIT, byte-identical
build files, two example modules with their import specifiers rewritten; see
`static/three/README.md` for the provenance, the hashes and the NOTICE entry).
A CDN is no use on a LAN with no route to the Internet.

**The asset** is the glTF the node serves at `/scene/scene.glb` (patch 14),
fetched with the node's blake2b hash in the query string so a headset that
cached an earlier run's scene fetches this one instead. If the export's own
joint count disagrees with the `nq` the node read out of the file it is
serving, the page is looking at a different file and refuses it rather than
applying every joint value to the wrong body.

**The kinematics are MuJoCo's own** (`mjkin.js`): the exporter writes one glTF
node per body with its rest transform and its joints' axes, anchors and qpos
addresses, so `qpos` alone puts every body -- free objects, the compress
laminate and the Fin-Ray fingers included -- exactly where MuJoCo has it. The
page's answer agrees with `mj_kinematics` to 14 nm over the test trajectory.
`scene-state.js` decodes the sim's JSON and its binary layout alike, drops a
message with one unusable value rather than drawing half a pose, and drops one
the unordered channel delivered late (sequences compared modulo 2^16, as the
pose channel's receiver does).

**Where the robot stands** (`align.js`) is the part the clutch decides. The
node's pose mapping is `p_arm = R (p_local - pivot) + frame_offset`, so the
only rotation under which a hand pushed forward pushes the gripper forward is
`R^T`: the robot faces the same way the operator does, and they stand behind
it. The rule is

> the sim's `arm_origin` site lands `scene.exo.distance` m in front of the
> anchor and `lateral` m to its right, at the height that puts the table top at
> `scene.exo.table_height` above the operator's own floor, with the robot's
> heading turned by `scene.exo.yaw` from the anchor's

where the anchor is the head pose **latched at the same moments the node
latches its neck pivot**: the first frame of a session, and the first frame
after a gap in head poses longer than `SMOOTHER_RESET_GAP` (patch 13) -- a
recenter, the system menu, a headset taken off. Latching the two together is
the point: by the time the anchor moves, the clutch has released both arms, so
the robot can never move under a hand that is commanding it. Walking about
does not move it. `placement: ego` is the exact inverse of the node's mapping
instead, putting the robot's chest where the operator's is; it is honest only
while the clutch is disengaged (the clutch is relative, so after a re-anchor
the gripper is not where the hand is), which is why it is not the default.
Both draw a **gripper ghost** at each controller's aim pose, turned by the
node's own `CONTROLLER_TO_EE`, solid while that arm is following and faint
while it is not -- the only honest answer to "which way will the gripper hold
if I squeeze now".

**The nudge**: the left thumbstick moves the scene forward/back and sideways,
the right one turns and raises it, so the operator can line the virtual robot
up with the real table it is drawn over. A stick only nudges while its own
arm's clutch is disengaged (the state message's flags say), moves at 0.3 m/s
and 0.6 rad/s outside a 0.2 deadzone, is clamped to 1 m / 0.5 m / 90 degrees,
and survives a re-anchor: it is a correction to the rule, not to one anchor.

**The panels.** three.js owns the session's render state, because on a headset
it takes the `XRProjectionLayer` path and `session.renderState.baseLayer` is
then null -- the hand-written quad renderers cannot draw in that session at
all. So in a 3D view the HUD is a mesh with a `CanvasTexture` over the canvas
`hud.js` already lays its text out on (one layout, two consumers: `HudContent`
is shared, and the WebGL panel now uploads by revision), and `scene+panel`
draws the head camera as a mesh with a `VideoTexture` on the same track. The
camera panel is depth-sorted against the robot now, where the 2D panel was
always in front -- which is why its distance is a rule and not a taste: an
opaque quad in the scene's own depth buffer hides everything beyond it, so it
has to sit past the far edge of the cell table, since everything the robot can
reach stands on that table. `config/view_scene_panel.yaml` carries the number
and the measurements behind it; quote it from there rather than from here. The
calibration instructions stay 2D: a 3D view says so rather than crashing on the
missing base layer.

**Never a black headset** (X5). No `scene` from the node, an asset that is not
the node's, no WebGL, or the whole view failing to start: the page falls back
to the camera panel (`scene+panel`) or to passthrough and the HUD alone
(`scene`), with one notice. The notice goes on the HUD panel, in amber on its
own row, because that is the only place the operator can read it with the
headset on, and on the page's `#log` for whoever is holding the laptop. A
state stream that stops leaves the robot frozen where it was, saying so; a
`state` channel that closes changes nothing else; and the node keeps getting
exactly the frames it got before, from the same `local` space, paused by the
same rules. A 3D view is the only thing that asks for a `local-floor`
reference space, as an optional feature, and works out the floor from the head
height when the runtime has none.

Measured in the emulated Quest 3 on this host, with the compress cell's own
export (94,828 triangles, 69 meshes, 51 moving bodies, 1.4 MB): **0.7 ms
median / 2.6 ms p95** per frame of page work, **1.8 / 4.1 ms at 4x CPU
throttling**, against the 13.9 ms a 72 Hz frame has. Against the real flow
(`oqt run sim --view scene`, the exporter's own `runs/<name>/scene.glb`): 0.7 ms
median / 3.0 p95 for `demo` (96,306 triangles), 1.5 / 4.6 for `compress`
(82,938 triangles, 179 moving bodies), and **11.9 ms median / 20.9 p95 for
`scene+panel`** -- the camera panel's `VideoTexture` upload, not the robot, is
the expensive part of that view here.

What none of this measures is the GPU. On this host the emulated headset
rasterises through SwiftShader, and a real export costs it enough that the
page's XR frame rate falls from 56 Hz (a camera view) to **8-10 Hz** in a 3D
view, with the page's own CPU work still at 0.7 ms. That is a property of
software rendering, not of the page, but it has a consequence worth knowing:
at that frame rate the clutch's `--stale-ms 100` holds the arm, so
teleoperation in a 3D view cannot be demonstrated on this host without raising
it (and `SMOOTHER_RESET_GAP` with it). Whether a Quest 3 holds 72 Hz with
96k triangles per eye is the headset question this leaves open.

The real-headset boundary is logged rather than inferred. The page records
when the glTF has built, when three.js has attached its XR renderer, and the
first rendered frame's floor space, anchor, world transform, camera-panel
visibility and WebGL error. The camera panel also retries muted video playback
after the operator presses Start; if Quest Browser rejects that attempt it is
reported in `quest.log` instead of leaving an unexplained empty rectangle.
That logging found a real Quest Browser 149 startup race: when `local-floor`
was not in `XRSession.enabledFeatures`, two starts rejected its reference-space
request immediately and one left the promise pending until the session ended.
The page now asks for `local-floor` only when the session says it was granted;
otherwise it selects `local` directly, so an optional floor cannot hold the
entire three.js renderer before its first frame.

Three test files still described the two-transceiver, head-camera-only page
after that change, and were brought in line with it on 2026-09-22.
`test_video.py` now expects every view to carry the overview and both arm
cameras after its head cameras. `test_oqt_sessions.py` offers
`VIDEO_TRANSCEIVERS` video slots, because aiortc cannot answer a track the
offer has no slot for (`ValueError: None is not in list` in `and_direction`).
Its stub `panel.js` also exports `createCameraPanels`, which `ar.js` now
imports. No shipped code changed.

`window.__oqtPage` is a read-only test seam: WebXR draws into the session's
own framebuffer, so a test outside the headset has no other way to see what
the view did. `scene3d.js`'s `captureEyes` renders the same scene with the
same view and projection matrices into an offscreen target and reads it back,
which is where the eye images in `tests/web/` come from.
Files: `src/dora_openarm_webxr/static/scene3d.js`, `static/align.js`,
`static/mjkin.js`, `static/scene-state.js`, `static/three/*` (vendored),
`static/ar.js`, `static/connection.js`, `static/hud.js`,
`static/video-source.js`, `config/view_scene.yaml`,
`config/view_scene_panel.yaml`.

## 16. One line a minute saying what became of the page's frames

A session that teleoperates badly gave no numbers at all: the node said nothing
about the frames it received, so "the arm keeps stopping" could not be told from
"the link is dropping frames" without adding instrumentation after the fact.
`_FrameStats` / `_report_frames` now summarise every minute, called from
`_publish_poses` so **the line arrives on the pose tick whether or not a frame
did** — a headset that stopped sending is exactly the case worth reporting.

```
poses: 4320 in 60.0 s (72.0 Hz); gap p50 13.9 p90 14.2 p99 16.1 max 18 ms, 0 over 100 ms; dropped none
poses: 3551 in 60.0 s (59.2 Hz); gap p50 16.7 p90 17.6 p99 18.8 max 199 ms, 2 over 100 ms; dropped 2 timed out
poses: 145 in 3.0 s (48.3 Hz); gap p50 16.6 p90 17.2 p99 17.9 max 1133 ms, 1 over 100 ms; dropped 11 right untracked, 67 left untracked
```

Accepted frames and their rate; inter-arrival gap p50/p90/p99/max — **the
median and the tail separately**, because the failure shape is a healthy p50
with a 251 ms max, which a mean hides; how many gaps exceeded the downstream
freshness window; and per-reason drops the node can actually see: frames
refused by the `sequence <= last_sequence` check (previously silent), frames
with no head pose, hands the page left out because it saw the controller as
untracked (counted per side), and how many times the cached frame aged past
`--pose-timeout` (once per dropout, not per tick). The gap clock is broken on
session start and end, so a reconnect is not reported as one huge gap.

**The gap threshold is the clutch's own number, not one webxr picked.**
`--pose-gap-ms` / `POSE_GAP_MS` (default 100) changes no behaviour; it only
labels this line. `oqt run` builds the clutch node before webxr and passes
`clutch_fresh_s(clutch)` through as `POSE_GAP_MS`, so "N over 100 ms" means "N
times the clutch could have let go", and it follows an overridden `--stale-ms`.
A nonsensical value falls back to 100 with one line rather than failing a run.

Two of the five interesting drop reasons **cannot** be separated from this side
and are not guessed at: `visibility` (the page deliberately sending nothing
under the system menu or with the headset off) reaches the node as an ordinary
gap, indistinguishable from link loss, and telling "I never sent it" from "I
sent it and it was lost" needs a change in `static/ar.js`. The `_FrameStats`
docstring says so.

Defect found by running it rather than reading it: the first live line was
`poses: 1 in 3026528.4 s; (0.0 Hz)` — the machine's uptime, because the window
started at 0.0 on a `perf_counter` clock and was therefore due on the first
frame. `__init__` now defaults `now` to `time.perf_counter()` and `_main_async`
restarts the window when the node starts serving.
Files: `src/dora_openarm_webxr/main.py`.

## 17. The page sends its own log back to the node

When something goes wrong on the Quest, the page knew about it and nobody
could hear: debugging it meant a cable, `adb` and `chrome://inspect`, by which
time the failure was an hour old and the operator had taken the headset off.
The page now records what it knows and delivers it to the node by itself.
`static/telemetry.js` is the whole page half; its module header is the design
document and this is the contract.

**Loaded before everything, imported by nothing.** `index.html` loads
`telemetry.js` as its own `<script type="module">` ahead of `ar.js`, and it
publishes itself as `window.__oqtTelemetry`. `ar.js`, `connection.js` and
`scene3d.js` read that global through `?.`, so a page whose telemetry module
failed to fetch works exactly as it did before, and so do the Node.js
stand-ins `tests/test_oqt_sessions.py` runs `ar.js` against — telemetry may
never be the reason teleoperation stops. Ahead of even that, `index.html`
carries a nine-line **classic** script that catches `error` and
`unhandledrejection` into a bounded array: a module cannot catch an `ar.js`
that will not parse or will not fetch, and that is exactly the failure that
leaves a blank page. `telemetry.js` drains the array at install and sets it to
null, which turns the early handlers off so nothing is recorded twice.

**What is captured.** Uncaught errors with their stack; unhandled rejections;
`console.error` and `console.warn` (and `console.log` only with
`?telemetry=verbose`); resource loads that failed; WebGL context lost and
restored, watched on both canvases the page can make (the panel session's and
the 3D view's); the scene glTF failing to load, with the URL and the HTTP
status; the XR session starting — with the user agent, `enabledFeatures`, the
blend mode, the interaction mode, the device pixel ratio and the page's origin
— and ending, with the node's own reason when it gave one; XR and document
visibility changes; input sources added and removed; the head or a controller
going untracked and coming back, **as edges only**; every WebRTC connection,
ICE and data-channel state change and close, and an offer the node refused
with its status; and once a minute the page's **own** frame rate and pose-send
interval as p50/p95/p99/max, which is the other end of the node's own gap line
from patch 16 — together they tell "the page never sent it" from "the link
lost it", which patch 16's docstring says it cannot do alone.

Every envelope carries the page's **build identity** — the
`dora-openarm-webxr` version the page files came from, `const BUILD` in
`telemetry.js`, pinned to `pyproject.toml` by a test — and the node's own
build from its configuration message (its version plus a hash of the files it
serves). A version that is not this page's is an **error** record saying the
headset is showing a cached page and to reload it, which the node mirrors to
its stderr. The hash half is the node's to compare, since the page cannot hash
itself without fetching all of itself again; the page only carries it, in
every envelope, so the node can see which of its builds this page came from.
A headset running last week's bundle against this week's node is otherwise a
whole class of reports that reads as a node gone mad.

**The transport is the point of the feature**, because the case worth
debugging is usually the case where WebRTC is broken. In order: the reliable
`control` channel when it is open, as `{"type": "log", ...}` (a node that does
not know the type ignores it, so this is safe against one without the sink);
an ordinary `POST log` to the node's own HTTPS server otherwise, same origin
as `POST offer` and carrying the same pairing token in the body, which works
with **no WebRTC at all**; and `navigator.sendBeacon` on `pagehide` and on
going hidden, which outlives the page and is also why the token goes in the
body rather than a header. A beacon the browser will not queue, and records
the sink refused, are stashed in `sessionStorage` and sent by the next page
load under their **original** session id. A sink that refuses is backed off
from, 1 s doubling to 30 s, and nothing is thrown away for it: a non-2xx puts
the records back in the buffer, which is what a node started without
`--quest-log` (503) or an older node (405, the static mount) gets.

**The buffer keeps the first records as well as the last.** 256 records; when
it is full the one evicted is the oldest *after* the first 32, so a crash
cascade can never push out its own cause. It is module state, tied to neither
the XR session nor the connection, so it survives a session end and every
reconnect — and the flush that happens when the control channel opens is how
the reason the *previous* connection died reaches the node. Records with the
same level, source, message and top stack frame coalesce into one with `n` and
`last`: an error thrown every frame arrives as `n: 360`, not as 72 messages a
second. An envelope too big to send is split rather than dropped, and the
envelope numbers still run without a gap.

**Nothing that could carry the pairing token is logged.** `location.hash` is
never read into a record; the exact token is replaced with `[token]` wherever
it appears, and anything shaped like `#k=`, `?k=`, `token=` or `"token":` with
`[redacted]`, in the message, the stack and every string of `data` — including
a token this page does not know the value of. The one record that says where
the page came from names the origin and the path only.

**Nothing runs on the XR animation loop.** `ar.js` calls `telemetry.frame()`
once a frame (two subtractions, two stores, two counters) and
`telemetry.tracking()` three times (a compare that returns unless the state
actually changed); everything else — building an envelope, encoding it,
sending it — happens on a `setTimeout`, which keeps running while an immersive
session presents. Measured in the emulated Quest 3 on this host: **63-70 ns
per XR frame** for all five hooks together, against the 13.9 ms a 72 Hz frame
has. The A/B of the 3D view's own per-frame cost with telemetry on and off
cannot resolve that and says so: Chromium quantises `performance.now()` to
100 µs and the mean over ~240 frames wanders by tens of microseconds either
way on a shared machine (runs have come out 60 µs *faster* with telemetry on),
so the browser test bounds that comparison at 200 µs as a regression guard and
takes the real number from the direct measurement. A record costs about 2 µs
to make and a repeat about 2 µs; one envelope of 64 records costs about 3 µs
to build and encode.

**The wire, as the node reads it.** The envelope is
`{type: "log", v: 1, session, build, node?, page, dropped, reason, records[]}`
on the control channel, and the same object plus `token` as `POST /log`,
answered 204 when stored. Each record is
`{record, t, mono, xr?, level, event, msg, stack?, n?, last?, data?}`. The
field names follow what the node writes: it gives each line its own
`received`, `session`, `peer`, `via` and `seq` and keeps its own when a name
collides (`main.py` `_line`), so the page's counter is `record` — a gap in it
is a record the page's buffer had to drop — and what would naturally be called
`source` is `event`, the name of that column in the node's line. Strings are
capped at 2048 by the node, so the page cuts stacks at 2000 rather than
spending the headset's link on bytes that will not survive; a batch is 64
records because that is the node's own cap.

A context the page drops **on purpose** is not a failure. `scene3d.js`
`close()` calls `forceContextLoss()` to give the browser its context back, so
`close()` says `telemetry.contextClosing(canvas)` first and that loss is
recorded as `info` ("context released") rather than `error`. Without it every
clean `--view scene` session end left an error in the file, `oqt run` ended
every good run by reporting one, and the operator would have learnt to ignore
the one level that matters.

Two known gaps. A page the node refuses to pair with has no token, so it
cannot post its log either — the reason it could not pair is the one thing
this cannot deliver, and the page says it on `#log` instead. And a beacon's
records can arrive twice if the browser queued one the node had already taken;
`(session, record)` is the pair to dedupe on.

### 17.1 The node's side: the sink

The node keeps the other end. `QUEST_LOG` (or `--quest-log PATH`) names one
file; unset means there is no sink at all, and then `POST /log` answers **503**
so the page *keeps buffering* rather than discarding what it has. `oqt run`
sets it to `runs/<name>/quest.log`, for the real cell as well as the sim — it
is the same page on both.

**Two ways in, one writer.** `POST /log` (and `/telemetry`, the same handler)
is checked against the same pairing token as `POST /offer` and answers `204`
with no body, because a `sendBeacon` cannot read one. Refusals: `403` (wrong or
missing token), `400` (not an envelope, or a body cut off mid-upload), `413`
(over 64 KiB), `429` (this peer is posting too often) and the `503` above. The
body is read by `_read_bounded`, which streams and stops accumulating past the
cap whether the `Content-Length` was honest or not, and drains up to 16 MiB so
an honest oversized POST still *receives* its 413 instead of a reset socket.
The other way in is the reliable `control` channel: `webrtc.py` routes a
`{"type": "log"}` message to the same sink with a per-peer name (`peer-1`,
`peer-2`, assigned when the peer is made). No token is asked for there — the
peer exists only because an offer carried one.

**The line.** One JSON object per line. The node writes `received` (local
ISO 8601), `level`, `event`, `session`, `peer`, `via` (`http` or `control`) and
`seq` (its own counter), and copies `build`, `node`, `page`, `flush` (the
envelope's `reason`) and `v` onto **every** line the envelope carries, so a
single line grepped out and mailed still says which page build and which batch
it belongs to. A record can never overwrite any of those, and a top-level
`token` key is never written at all. The file is opened lazily — a node nobody
connected to leaves nothing behind — and flushed per line.

**Nothing the page sends is trusted.** `_clean` truncates strings at 2048
characters, keys at 64, lists and objects at 64 items, nesting at 6, turns a
400-digit integer or a non-finite float into a string, and forces an unknown
`level` to `info`; a record that is not an object is dropped and counted, never
fatal. `_redact` runs over every string at every depth, unconditionally: each
URL loses its query and fragment (keeping scheme, host, path and a stack
frame's `line:col`), the configured token is replaced wherever else it appears,
and a bare `#k=` / `?token=` or a quoted `"token": "..."` is blanked — which is
how a token belonging to *another* node is caught, since only its shape can be
recognised, not its value.

**Bounded in every direction.** Per peer, 20 requests burst and 5 a second,
refused with 429 *before* the body is read; per node, 400 records burst and 200
a second, and at most 64 records taken from one envelope. The file is capped at
4 MiB and rotated **once** to `quest.log.1`, so a run costs at most twice the
cap and the newest records are the ones kept. Everything thrown away is counted
and admitted in the file itself — `quest-log-dropped` for the node's own
losses, `page-dropped` for a non-zero envelope `dropped`, which is the page's
buffer and a different fault — so a gap in `seq` is never a mystery. An
`OSError` on write is said once, counted and retried no sooner than 5 s later:
a full disk costs the log, not teleoperation.

**One error where the operator already looks.** A `level: error` record also
prints a line to the node's stderr, which dora relays into
`runs/<name>/dora_run.log`. Rate limited to 10 burst and 0.5 a second with a
single "further headset errors are only in `<path>`" note, so a cascade cannot
drown the robot's own log. It reads `msg` (what `telemetry.js` writes) or
`message`, and appends `(xN)` when the page coalesced a repeat.

**Which build the page should be.** `_build_identity()` is the package version
plus a blake2b of every file in `static/` — the page *this node serves now*.
Printed at startup as `page build: 0.6.0+<hash>` and pushed to every page as
`build` in the configuration message, which is the half `telemetry.js` compares
against its own constant.

At shutdown the node prints `quest.log: N records, E errors, W warnings, D
dropped by the page (<path>)`. A real bug found on the way: the URL pattern in
`_redact` backtracked on a long unbroken run of letters — a base64 blob, or one
character repeated, which is exactly what a page in trouble sends — so a single
60 kB word cost minutes of the event loop that answers the operator's hands.
Fixed with a lookbehind and possessive quantifiers, and pinned by a test that
requires a 60 kB word to be written in under 5 s.

**Measured.** On this host, the whole sink costs about **23 µs** for one record
(redact, clean, JSON, write, flush) and **18 µs** a record in a full envelope of
64; the node's own 200 records/s cap is therefore about **0.45 % of one core**,
and that is the most a headset can ask for. `_redact` is linear in the length
of a word: 1 KiB 14 µs, 16 KiB 211 µs, 64 KiB 680 µs. A 2-minute `--view scene`
session that goes right writes **25 lines, 9.5 KiB** (~4.8 KiB a minute, ~279
KiB an hour against the 4 MiB cap).

Files: `src/dora_openarm_webxr/static/telemetry.js` (new),
`static/index.html`, `static/ar.js`, `static/connection.js`,
`static/scene3d.js`, `src/dora_openarm_webxr/main.py`,
`src/dora_openarm_webxr/webrtc.py`.

## 18. A headset wired over USB relays through a local TURN server

Adopted from guchio123/dora-openarm-data-collection `6676c7c`. With the Quest
on USB, `scripts/webxr-usb/start.sh` reverse-forwards TCP 8443 (the page) and
3479 (a loopback-only coturn) with `adb reverse`, so the page is opened as
`https://localhost:8443/#k=<token>`. `adb reverse` carries TCP only and there
is no UDP path to the node, so a page on `localhost` with no `?ice=` offers
with `iceTransportPolicy: "relay"` through `turn:localhost:3479?transport=tcp`
(`rtcConfiguration()` in `connection.js`). Everywhere else patch 1 still
holds: no ICE servers, and `?ice=` or `connect({ iceServers })` win over the
USB default. The relay is TCP, so a lost segment delays every later frame
instead of dropping one; prefer the LAN when it is good.
Test: `tests/web/connection_units.test.mjs` (Node.js alone).

## 19. Pairing token from a file

`PAIRING_TOKEN_FILE` names a file holding the token (patch 8), so a dataflow
that is committed and shared names a path instead of carrying the secret in
its `env:`. `scripts/webxr-setup` writes it (mode 0600) next to the TLS files.
`PAIRING_TOKEN` still wins when both are set. A `PAIRING_TOKEN_FILE` that is
set but missing or empty stops the node: the operator asked for pairing, and
silently accepting every offer instead would put the arms in reach of anyone
on the network. Test: `tests/test_pairing_token_file.py`.

## 20. The argument parser is a function

`main()` built its `argparse` parser inline. `build_parser()` now returns it
and `main()` calls it, with no change to any option. That lets
`scripts/check-dataflow` (`dora_openarm_teleop.check`) read this node's
`args:` and `env:` from a dataflow exactly as the node will, abbreviations and
environment defaults included, so the timing chain it checks
(`POSE_TIMEOUT <= clutch fresh_s < SMOOTHER_RESET_GAP`) is never computed from
a second, drifting copy of the option list.

## Tests

`tests/test_oqt_patches.py` (pytest, part of `uv run pytest
third_party/dora-openarm-webxr/tests`) covers HUD parsing and forwarding, control
commands, the dora loop ending on `quit` (also with `--quit-button`), a hosted
run with a real uvicorn and WebRTC server stopping on `quit`, the One Euro,
`ICE_SERVERS='[]'`, `PAIRING_TOKEN`, `SMOOTHER_RESET_GAP` and `PIVOT_MODE`
environment handling through `main()`, the HUD replay on a second aiortc peer, and
a `node --check` of every page module. The modules are checked as `.mjs` copies,
because Node 24's `node --check` on a `.js` file with `import` statements reports
success without parsing it. `tests/test_oqt_sessions.py` covers patches 8-11:
`/offer` through the ASGI app (403 for every kind of missing or wrong token before
the transport is called, 200 with it, the one warning when unset, 400 and 503),
refusal, takeover and HUD routing between real aiortc peers, the takeover idle
timed from the later of the session start and the last frame (on a fake clock and
between real peers: refused before the first frame, taken over after the idle), the
peer cleanup and cap, and `ar.js` and `connection.js` run in Node.js against
stand-ins. `tests/test_oqt_quest_log.py` covers the sink half of patch 17: a post with the
token written as one JSON line and five kinds of post without it refused; a body
that is not an envelope, and one cut off mid-upload; a 10 MiB body and one with
a lying `Content-Length` refused 413 with nothing written; both names of the
endpoint; a node with no sink answering 503 rather than dropping the report; a
flood of posts limited per peer without locking out a second headset, and a
flood of records capped with the file saying so; two real aiortc peers logging
over the control channel and kept apart, a malformed control message costing
the record and not the session, and control records cleaned although no token
was asked for; the pairing token never written however it arrives (including
one this node does not know, and the one the page authenticated with); the
redaction linear in the length of a word (60 kB in under 5 s — it took over
100 s before the ReDoS fix); the file capped and rotated once; a write that
fails not stopping the node; errors mirrored to stderr, rate limited, reading
either `msg` or `message`; a record cut down rather than written whole, and one
that is not an object dropped rather than fatal; every line saying which page
build and which batch it came from; what the *page* dropped kept separate from
what the node dropped; and the shutdown summary.
`tests/test_oqt_pose_gaps.py` covers patches 12, 13 and 16 by
replaying 72 Hz frames and 200 Hz ticks on a fake clock, including a 105 ms
stall with `POSE_TIMEOUT` 0.1 and `SMOOTHER_RESET_GAP` 0.12 and a 101 ms stall
that no tick saw time out. For patch 16 it pins the minute line: a burst of 9
lost frames keeps p50 at 13.9 ms while max passes 125 and the line says "1 over
100 ms"; a frame delivered late is counted rather than silently dropped; a hand
the page saw as untracked is named per side; a frame with no head pose is
counted; the node's own pose timeout is counted once per dropout; the line still
arrives when the headset is what stopped; the gap clock does not span a session;
and the first window starts when the node does (checked by re-breaking it — the
test fails with the old 0.0 default). `tests/test_oqt_scene_state.py` covers
patch 14 between real aiortc peers: **no `state` channel offered to a view that
draws no scene** (real aiortc peers over loopback SCTP), a browser waiting for
that channel not being left waiting, the node forwarding nothing in that case,
which of the five views draw the scene, and the drop report saying what it
measured rather than what it guessed (a healthy stream, a link that keeps
dropping, and at most one line a batch); plus the channel order and its
reliability settings, nothing sent
while no session is active, the active peer only (through a takeover), 72 Hz
for a second with every message arriving (measured: 1.5 kB a frame, 0.11 ms
median node-to-page latency, no drops), a full send buffer dropping the newest
scene and recovering when it drains, the pose and HUD paths still working while
scenes flow, and the `scene` block, the glTF route and the input shapes. Six of
its tests cover the gzip representation over the committed 11,520 B fixture
export: the headers and the compressed body, a client that will not inflate
getting the file itself, a cached asset revalidating to 304 for either
representation, the compressed copy being a function of the file alone
(`mtime=0`), an asset that does not shrink being served as it is, and a scene
swapped under the endpoint not being served the old compression. The browser
end of it is in `tests/web/webxr_page.test.mjs`, which reads the page's own
Resource Timing for the very fetch that produced the triangles and asserts
`transferSize < decodedBodySize` with `triangles > 0`.
Patch 17 has two tests of its own in `tests/web/`, plus two checks against the
real node. `telemetry_units.test.mjs` (Node.js alone, about a second) builds a
fake browser -- window, document, console, navigator, sessionStorage, fetch,
sendBeacon -- and imports `static/telemetry.js` against it, once per case with
a cache-busting query, so the module installs itself exactly as it does on a
page: it pins the page's `BUILD` to `pyproject.toml`'s version, and checks the
record shape, the console hook printing before it records and not recursing,
`console.log` staying out unless `?telemetry=verbose` asks for it, 500 copies
of one error becoming `n: 500`, a buffer of 1000 distinct records keeping
`distinct 0..31` **and** `distinct 999` (and counting the 744 it dropped), a
token that appears three ways in a message coming out `[token]`/`[redacted]`
and never itself, the control channel being preferred, the HTTPS POST carrying
the token to `log` on the page's own origin with no fragment in the URL,
a sink answering 405 twice leaving the records buffered until it answers 204,
a node build that is not this page's coming out as an error record saying the
headset is showing a cached page,
`pagehide` and going hidden beaconing, a beacon the browser will not take
being carried to the next page load **under its original session id**, a
session end flushing early, the buffer surviving a session end and a
reconnect, the minute report's percentiles over a 72 Hz stream with one 200 ms
stall, tracking recording only the edges, a context the page *releases* itself
being `info` while one it *loses* stays `error`, an over-large envelope being
split with no gap in the envelope numbers, `enabled: false` leaving no global
at all, and a record made of circular and unprintable arguments still going
out.
It also measures the cost: about 2 µs a record, 2 µs a repeat, 3 µs an
envelope, 0.15 µs for a whole frame's hooks.
`telemetry.test.mjs` (Playwright + IWER, its own HTTP server answering
`POST /log` as the node's sink does, and a peer in the page standing in
for the node) does what a fake cannot: a real uncaught error with its stack, a
real rejected promise, a real `console.error` that still prints, 120 copies of
one error arriving as `n: 59` in two records, a real WebGL context loss
(`renderer.forceContextLoss()`), a real glTF that 404s arriving with the URL
and the status, a real session end with its reason, a real `pagehide` beacon
after navigating away, **the blank page** (an `ar.js` served with a syntax
error: nothing of the page ran, nothing drew, there is no connection, and the
parse error still reaches the node over HTTPS through the inline handler in
`index.html`), and **the WebRTC-down case**: with
`RTCPeerConnection` made to throw, the page's log arrives over HTTPS with the
pairing token and the records naming why there is no channel. It also measures
the frame cost both ways -- the 3D view's own per-frame mean with telemetry on
and off, which is noise at this resolution and is checked only as a 200 µs
regression guard, and the hooks directly at 63-70 ns per XR frame.
`tests/web/` is an end-to-end browser test
(Playwright Chromium + IWER + `dora run`) of patches 1-6, 8, 9 and 17, and with
`VIEW=scene` or `VIEW=scene_panel` of patch 15 as well: the node serves its
glTF, the page draws it, and a `scene_state` dora input puts the robot's wrist
where MuJoCo says, to 0.000 mm. `SCENE_GLB` may name a run's own
`runs/<name>/scene.glb` rather than the committed fixture; `tests/web/scene_truth.py`
then writes the ground truth for that asset (MuJoCo's `mj_kinematics` on the model
the manifest names) and the check uses the deepest body a joint moves, whatever
this scene calls it. Patch 15 also has two tests of its own, which
need neither dora nor a fixed port: `tests/web/scene_units.test.mjs` (Node.js
alone) checks the page's kinematics against MuJoCo's own over a recorded
trajectory, the placement against the pose maths in main.py that it inverts,
the thumbstick nudge and the state codec; `tests/web/scene3d.test.mjs`
(Playwright + IWER, with a peer inside the page standing in for the node)
checks the whole view in a real XRSession -- the asset and its hash, the
placement rule, the re-anchoring, the nudge and its clutch lock-out, the HUD,
the frame budget under CPU throttling, the fallbacks, and what each eye
actually drew. See `tests/web/README.md`.
Four upstream tests were changed with the behaviour they check, each
marked in a comment: `test_invalid_target_suspends_and_recovers` uses a 50 ms
gap instead of 9 s (patch 12), `test_pose_timeout_uses_accepted_frame_receive_time`
expects the timeout to reset the filter (patch 12),
`test_calibration_result_broadcast` starts a session first and
`test_only_active_session_can_publish_or_end` expects the refusal and the
takeover (patch 9). The other upstream tests are unchanged and pass.
