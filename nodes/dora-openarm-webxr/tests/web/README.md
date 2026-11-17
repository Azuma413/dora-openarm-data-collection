# Browser tests of the WebXR page (openarm-quest-teleop)

Five tests, in the order they are worth running:

| File | Needs | Covers |
| --- | --- | --- |
| `telemetry_units.test.mjs` | Node.js | the page log's rules: buffer, coalescing, redaction, backoff (PATCHES.md 17) |
| `scene_units.test.mjs` | Node.js | the 3D view's maths: kinematics, placement, nudge, state codec (PATCHES.md 15) |
| `telemetry.test.mjs` | + Playwright, IWER | the page log in a real browser, including the WebRTC-down case (PATCHES.md 17) |
| `scene3d.test.mjs` | + Playwright, IWER | the 3D view in a real XRSession, with a peer in the page standing in for the node |
| `webxr_page.test.mjs` | + `dora run`, port 18443 | the real node and the real page together (patches 1-6, 8, 9, 17, and 15 with `VIEW=scene`) |

`scene_truth.py` is not a test: it writes the ground truth (`trajectory.json`)
the 3D checks compare the drawn robot against, for any `scene.glb` the
exporter produced. `webxr_page.test.mjs` runs it by itself when `SCENE_GLB`
names something other than the committed fixture.

## `webxr_page.test.mjs`

Runs the real node and the real page together, with no headset:

- `dora run` starts `dora-openarm-webxr` (HTTPS on 127.0.0.1:18443, a throwaway
  self-signed certificate, `ICE_SERVERS='[]'`, `config/view_mono.yaml`, a random
  `PAIRING_TOKEN` and `PIVOT_MODE=latched`, as `oqt run` sets them) and
  `driver.py`, a small dora node that records every webxr output it is wired to
  and sends `hud` / `control` / `scene_state` inputs when the test asks
  (`scene_state` sends the JSON encoding, `scene_state_binary` the uint8 array
  the sim sends by default).
- Playwright's headless Chromium opens the page with Meta's
  [IWER](https://github.com/meta-quest/immersive-web-emulation-runtime)
  (`iwer` on npm) installed as `navigator.xr`, emulating a Quest 3 with Touch
  Plus controllers. It opens the page as `https://127.0.0.1:18443/#k=<token>`.
- The test clicks Start, moves the emulated controllers and presses buttons
  through IWER, and checks what reaches the driver. Some checks open further
  pages (each its own tab) to try the token and a second headset.

What it checks, one line each (PATCHES.md numbers):

| Check | Patch |
| --- | --- |
| The page builds its peer with `iceServers: []`; `?ice=stun:...` overrides it | 1 |
| `POST /offer` without the token or with a wrong one gets 403, garbage SDP with the token 400; a page opened without `#k=` says 403 and why | 8, 10 |
| A reload after the `#k=` fragment was removed from the URL still pairs (sessionStorage) | 8 |
| Start opens a session; the node publishes `status=ready` | - |
| `pose_right` (8 finite values), `grip_right=1`, `button_a=true` arrive; moving the controller 20 cm forward moves `pose_right` x | - |
| A second page's Start while the first sends frames gets `session-refused`: its session ends and it says why, no second `status=ready`, no HUD for it, the first page's poses keep flowing | 9 |
| A `hud` input reaches the page as `{type: "hud"}` and pulses the right controller once (0.6, 80 ms); screenshot in the run directory | 5 |
| No frames while `visibilityState` is `visible-blurred`; frames again when `visible` | 3 |
| With the left controller disconnected, the right pose, grip and buttons still flow | 4 |
| An `emulatedPosition` right pose drops `pose_right`/`grip_right`/`trigger_right`, keeps `button_a` | 3 |
| Hand input sources publish no pose, grip or buttons | 4 |
| Frames pause for 250 ms-1 s after a recenter (`reset` event) | 3 |
| An uncaught error reaches the node as a `{type: "log"}` envelope on the reliable channel, is written to `runs/<name>/quest.log`, carries no pairing token, and the poses keep flowing | 17 |
| With `RTCPeerConnection` broken in its tab, a page's log reaches the node's HTTPS server (`POST /log`, 204) and lands in `quest.log` as `via=http`; a refusal leaves the records buffered | 17 |
| After the session ends, Start works again without a reload; nothing is sent in between | 2 |
| The reconnected page gets the last HUD again, with `haptic: null` and no second pulse | 5, 9 |
| With the first page silent (visible-blurred) for 1.3 s, a second page's Start takes over: `status=ready`, its poses flow, and the first page gets one `session-ended`, leaves its session and says "Another headset took over the session." | 9 |
| `control=quit` makes `dora run` exit 0, and every page ends its session | 6 |
| No uncaught page errors | - |

With `VIEW=scene` or `VIEW=scene_panel`, four more (the node is given
`SCENE_GLB` and a `scene_state` input):

| Check | Patch |
| --- | --- |
| The configuration carries a `scene` block with `/scene/scene.glb` and its hash, and the page loads it and reports its triangles | 15 |
| The asset arrives gzipped and still parses, read off the page's own Resource Timing: fewer bytes transferred than decoded (a real cell export is 4,716,792 → 2,022,637 B; the committed fixture 11,520 → 1,780 B) | 14 |
| A `scene_state` input puts the robot's wrist where MuJoCo says it is, through the page's own placement (0.000 mm) | 15 |
| The binary encoding drives the same robot: 68 bytes of binary put the wrist 0.000 mm from MuJoCo's answer (the same frame as JSON is 122 bytes) | 14, 15 |

The body the last check watches is the tracked body the ground truth moves
furthest, not the first in the list: the committed fixture deliberately tracks a
static table as well, so that a page that moved one would be caught.

Run it from anywhere (needs Node.js 20+, `openssl`, and the repository venv from
`uv sync`; it uses `dora run`, so nothing else may hold port 18443):

```bash
third_party/dora-openarm-webxr/tests/web/run.sh
```

`run.sh` installs `playwright` and `iwer` into `node_modules/` here on first
use and downloads Playwright's Chromium (about 115 MB, into
`~/.cache/ms-playwright`). A run takes about 25 s and prints `PASS`/`FAIL` per
check; the exit code is non-zero if any check fails. The run directory (dataflow,
`dora_run.log`, `events.jsonl`, `console.log`, `hud.png`) is kept on failure or
with `KEEP_RUN=1`, under `$TMPDIR`. `VIEW` picks the view configuration:
`mono` (default, `immersive-ar`, one camera panel), `stereo`
(`immersive-vr`, head-locked stereo panel), `scene` (the 3D view) or
`scene_panel` (3D and the camera panel). All four pass. In a scene view the
node is started with `SCENE_GLB` pointing at `fixture/arm.glb`, or at whatever
`SCENE_GLB` in the environment names -- a run's own `runs/<name>/scene.glb`,
for instance:

```bash
SCENE_GLB=$PWD/runs/<name>/scene.glb VIEW=scene \
  third_party/dora-openarm-webxr/tests/web/run.sh
```

The committed `fixture/trajectory.json` names the fixture's own bodies and its
own `nq`, so another asset needs its own ground truth: the test writes one with
`scene_truth.py` (MuJoCo's `mj_kinematics` on the model the export's manifest
names) into the run directory, and `SCENE_TRUTH` can point at one made earlier.
The robot check then uses the deepest body a joint value moves, whatever this
scene calls it -- `openarm_right_ee_outer_finger` for `demo`, a Fin-Ray segment
for `compress`.

Two checks count frames over a 700 ms window to see that the page stopped
sending and started again. What counts as "sending" is measured from this run
rather than fixed: on a host with no GPU the emulator rasterises a real 96k
export through SwiftShader and the page manages 8-10 XR frames a second, about
a sixth of a camera view's rate. Those checks care whether frames stop, not how
fast they come.

To see what the unpatched upstream page gets wrong, point the node at an
upstream checkout. `FORCE_EMPTY_ICE=1` keeps its STUN default from delaying the
offer (about 40 s on a machine with no route to the STUN server), so the later
checks run:

```bash
FORCE_EMPTY_ICE=1 PYTHONPATH=/path/to/dora-openarm-webxr/src \
  node third_party/dora-openarm-webxr/tests/web/webxr_page.test.mjs
```

At upstream 52cb5bb that fails 14 of the checks above: every patch check except
the reload one, which a node that asks for no token passes trivially.

Limits of the emulation:

- IWER flags every pose `emulatedPosition: true`, so the test reports tracked
  poses as `false` itself. Whether Quest Browser reports `false` for tracked Touch
  Plus controllers has to be checked on the headset (see the repository README).
- IWER has no user-activation rule, so the time limit on `requestSession` after
  a click is not exercised; neither are real haptics, passthrough or the HUD's
  readability.
- No camera frames are sent, so the video track stays empty.

## `telemetry.test.mjs` -- the page's own log (PATCHES.md 17)

The page in a real Chromium under IWER, with no dora and no fixed port: a
local HTTP server serves the page's files **and answers `POST /log` the
way the node's sink will**, and a second `RTCPeerConnection` in the same tab
answers the page's offer and opens the `control` channel. Both transports are
therefore real, and the test can see what arrived on each.

```bash
node third_party/dora-openarm-webxr/tests/web/telemetry.test.mjs
```

| Check |
| --- |
| A session start says what the headset is: user agent, `enabledFeatures`, blend mode, and the page's build identity |
| A real uncaught error reaches the node on the reliable channel, with its stack |
| A real rejected promise reaches the node |
| A real `console.error` reaches the node **and still prints** |
| 120 copies of one error arrive as one record counting them, not as 120 records |
| The session's end reaches the node, with a reason |
| Closing the 3D view gives its context back **quietly** -- `info`, not an error, so a clean `--view scene` run reports none |
| A real WebGL context loss on a **live** view (`renderer.forceContextLoss()`) reaches the node as an error |
| **With `RTCPeerConnection` unavailable, the log arrives over HTTPS** with the pairing token, and says why there was no channel |
| A page whose own `ar.js` will not parse -- nothing ran, nothing drew, no connection -- still reports the syntax error over HTTPS |
| A page navigated away beacons what it had not sent, and the sink gets it |
| A glTF that 404s arrives with the URL and the status, and the page falls back |
| The 3D view's own per-frame cost does not move with telemetry on |
| The per-frame hooks cost 63 ns a frame, measured in the page |

`SETTLE` (default 4000) is how long each timing sample runs; `HEADED` and
`KEEP_RUN` work as in the other tests. The run directory keeps `sink.jsonl`,
every envelope the HTTPS sink received.

## `telemetry_units.test.mjs` -- the page log's rules (PATCHES.md 17)

Node.js alone, about a second. Each case builds a fake browser on `globalThis`
and imports `static/telemetry.js` against it with a cache-busting query, so
the module installs itself exactly as it does on a page -- which is how a
sink that refuses, a buffer that overflows, storage that will not take a
value, and a page that goes away can all be made to happen on demand.

```bash
node third_party/dora-openarm-webxr/tests/web/telemetry_units.test.mjs
```

It also pins the page's `BUILD` constant to `pyproject.toml`'s version (that
constant is how a headset showing a page cached from an older run gives itself
away), and prints what a record, a repeat, a frame's hooks and an envelope
cost.

## `scene3d.test.mjs` -- the 3D view (PATCHES.md 15)

The same page, in a real `XRSession` under IWER, but with no dora and no fixed
port: a second `RTCPeerConnection` **in the same tab** answers the page's offer
the way the node's aiortc server does, opens the `control` and `state`
channels, pushes a view configuration and a `scene` block, and plays a recorded
MuJoCo trajectory at 72 Hz. `connection.js`, `ar.js` and `scene3d.js` are all
the real thing; only the network between them is not.

```bash
node third_party/dora-openarm-webxr/tests/web/scene3d.test.mjs
```

| Check |
| --- |
| The page loads the node's asset, with the node's hash in the query string so a cached one is not used |
| Start opens a session, the scene anchors, and it is drawn in `local-floor` |
| The placement puts `arm_origin` where the rule says, from the anchor the page latched |
| The robot follows the state stream: the wrist lands where MuJoCo puts it for that `qpos` |
| A free object follows the pose in the message, over the same body's joint values |
| A message delivered late (an older sequence) is dropped, not drawn |
| The thumbstick nudges the scene, keeps to its limit, and the rule still holds with the nudge in it |
| A thumbstick whose own arm is following the hand (the clutch flag) does not nudge at all |
| Walking about does not re-anchor; a recenter does, once, and the robot is in front of the operator again |
| The HUD is drawn, as a mesh, in the 3D view |
| Both eyes draw the scene over passthrough, and see it from different places |
| A stalled state stream freezes the robot, says so on the HUD, and keeps drawing |
| A `state` channel that closes stops neither the drawing nor the poses going to the node |
| The page's own work stays inside the 13.9 ms a 72 Hz frame has, at 1x and at `THROTTLE`x CPU throttling |
| A node with no scene falls back to passthrough with a notice, and the session still runs |
| An asset whose joint count is not the node's is refused, with a notice |
| `scene+panel` draws the camera panel as a mesh as well |
| `immersive-vr` draws the same scene on its own floor |
| No uncaught page errors |

Environment: `KEEP_RUN`, `THROTTLE` (default 4), `CAPTURE` (eye capture size,
default 512), `HEADED`, and `SCENE_GLB` to add one more check against a scene
the exporter really wrote (`SCENE_ARM_ORIGIN`, `SCENE_TABLE_Z` describe it).

Each run writes PNGs of what **each eye actually drew** into its run directory,
read back from an offscreen render with the session's own view and projection
matrices -- `scene-ar-{left,right}.png` as the compositor gets them (everything
the scene does not cover is transparent, which is where passthrough shows) and
`-over-room.png` with a flat grey room composited behind, plus the stereo pair,
the stalled view, `scene-panel-*` and `scene-vr-*`. Look at them: no check can
tell you the robot is the right way round.

## `scene_units.test.mjs` -- the 3D view's maths

Node.js alone, about a second:

```bash
node third_party/dora-openarm-webxr/tests/web/scene_units.test.mjs
```

It loads `fixture/arm.glb` with the page's own three.js and GLTFLoader and
replays `fixture/trajectory.json`, which `make_fixture.py` wrote from MuJoCo's
own `mj_kinematics`: the page's answer has to agree to 1e-6 m and 1e-6 in the
quaternion (it agrees to 14 nm, the limit of the float32 the state channel
carries). It also checks the exo placement against the rule and the ego
placement against main.py's own pose mapping, which it inverts; the nudge's
deadzone, rate and limits; how a scene configuration with missing or malformed
fields is read; and the state codec in both encodings, including the messages
it has to drop.

`fixture/arm.glb` and `fixture/trajectory.json` are committed, so neither test
needs MuJoCo. Re-run `make_fixture.py` (with the repository venv) only when the
exporter's node layout changes.
