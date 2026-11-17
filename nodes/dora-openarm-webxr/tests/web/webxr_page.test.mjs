// openarm-quest-teleop addition to the vendored dora-openarm-webxr tests
// (Apache-2.0, like the rest of this directory).
//
// Browser test of the patched page against the node.
//
// Runs `dora run` with the webxr node and driver.py, opens the page in
// Playwright's headless Chromium with Meta's IWER emulating a Quest 3 and
// Touch Plus controllers, and checks what the node publishes while the
// emulated operator moves, presses buttons, loses a controller, recenters,
// leaves and restarts the session, and while a second page tries to take
// the session over. See README.md in this directory.
//
// Environment:
//   OQT_ROOT   repository root holding .venv and config/ (default: four
//              directories up from this file)
//   WEB_PORT   HTTPS port of the node (default 18443)
//   KEEP_RUN   keep the run directory (logs, screenshots) even on success
//   VIEW       which config/view_<VIEW>.yaml the node is started with: mono
//              (default, immersive-ar), stereo (immersive-vr), scene (the 3D
//              view, PATCHES.md 15) or scene_panel (3D and the camera panel)
//   SCENE_GLB  the exported scene a 3D view draws (default: the fixture the
//              scene3d test uses; a run's own runs/<name>/scene.glb works too)
//   SCENE_TRUTH  the trajectory the drawn robot is checked against (default:
//              the fixture's, or one scene_truth.py writes for SCENE_GLB)
//   PYTHONPATH passed on to the node, e.g. an upstream checkout's src to
//              see which checks the unpatched page fails
//   FORCE_EMPTY_ICE  build the page's peers with no ICE servers whatever the
//              page asks for; with an upstream page, whose STUN default can
//              hold the offer back for tens of seconds offline, this lets
//              the later checks run

import { spawn, spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import https from "node:https";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(
  process.env.OQT_ROOT || path.join(HERE, "../../../.."),
);
const PORT = Number(process.env.WEB_PORT || 18443);
const URL = `https://127.0.0.1:${PORT}/`;
// The node refuses offers without this (PAIRING_TOKEN, patch 8); the page
// reads it from the URL fragment the way oqt prints it.
const TOKEN = crypto.randomBytes(24).toString("base64url");
const PAGE_URL = `${URL}#k=${TOKEN}`;
const VENV_BIN = path.join(ROOT, ".venv", "bin");
const IWER = path.join(HERE, "node_modules", "iwer", "build", "iwer.min.js");
const FORCE_EMPTY_ICE = Boolean(process.env.FORCE_EMPTY_ICE);
const VIEW = process.env.VIEW || "mono";
const VIEW_FILE = path.join(ROOT, "config", `view_${VIEW}.yaml`);
// The 3D views (PATCHES.md 15). They need an exported scene to draw, and the
// node then gains a `scene` block and a `scene_state` input; the fixture the
// scene3d test uses stands in for a run's own scene.glb, which SCENE_GLB can
// name instead.
const SCENE_VIEW = VIEW.startsWith("scene");
const FIXTURE_GLB = path.join(HERE, "fixture", "arm.glb");
const SCENE_GLB = process.env.SCENE_GLB || FIXTURE_GLB;

const RUN = fs.mkdtempSync(path.join(os.tmpdir(), "oqt-webxr-test-"));
const LOG = path.join(RUN, "events.jsonl");
const CMD = path.join(RUN, "commands.jsonl");
// Where the node writes what the headset reported (patch 17), as `oqt run`
// points it at the run directory.
const QUEST_LOG = path.join(RUN, "quest.log");
// Where MuJoCo puts each body for a given qpos: what the robot the page drew
// is checked against. The committed trajectory belongs to the committed
// fixture and names its bodies, so a real run's scene.glb needs its own --
// scene_truth.py writes one for any asset the exporter produced, and
// SCENE_TRUTH names one made earlier.
const TRAJECTORY =
  process.env.SCENE_TRUTH ||
  (SCENE_GLB === FIXTURE_GLB
    ? path.join(HERE, "fixture", "trajectory.json")
    : path.join(RUN, "trajectory.json"));

function ensureTrajectory() {
  if (fs.existsSync(TRAJECTORY)) {
    return;
  }
  const result = spawnSync(
    path.join(VENV_BIN, "python"),
    [path.join(HERE, "scene_truth.py"), SCENE_GLB, "-o", TRAJECTORY],
    { encoding: "utf8" },
  );
  if (result.status !== 0) {
    throw new Error(
      `scene_truth.py failed for ${SCENE_GLB}: ${result.stderr || result.stdout}`,
    );
  }
  console.log(result.stdout.trim());
}

// The webxr outputs the driver records.
const RECORDED = [
  "pose_right",
  "pose_left",
  "trigger_right",
  "trigger_left",
  "grip_right",
  "grip_left",
  "button_a",
  "button_b",
  "button_x",
  "button_y",
  "status",
  "vr_receive_times",
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const nowS = () => Date.now() / 1000;

const results = [];

function expect(condition, detail) {
  if (!condition) {
    throw new Error(detail);
  }
  return detail;
}

// One named check. A failure is recorded and the next section still runs,
// so a run against the unpatched page lists everything it gets wrong.
async function section(name, body) {
  try {
    const detail = await body();
    results.push({ name, ok: true });
    console.log(`PASS ${name}${detail ? ` -- ${detail}` : ""}`);
  } catch (error) {
    results.push({ name, ok: false });
    console.log(`FAIL ${name} -- ${error.message}`);
  }
}

async function waitFor(what, predicate, timeout = 15000, interval = 100) {
  const deadline = Date.now() + timeout;
  for (;;) {
    const value = await predicate();
    if (value) {
      return value;
    }
    if (Date.now() > deadline) {
      throw new Error(`timed out waiting for ${what}`);
    }
    await sleep(interval);
  }
}

function readEvents() {
  if (!fs.existsSync(LOG)) {
    return [];
  }
  return fs
    .readFileSync(LOG, "utf8")
    .split("\n")
    .filter((line) => line.trim())
    .flatMap((line) => {
      try {
        return [JSON.parse(line)];
      } catch {
        return []; // the line being written
      }
    });
}

function eventsSince(t, id) {
  return readEvents().filter((e) => e.t >= t && (!id || e.id === id));
}

// What the headset reported, as the node wrote it down (patch 17).
function questLogText() {
  return fs.existsSync(QUEST_LOG) ? fs.readFileSync(QUEST_LOG, "utf8") : "";
}

function questLog() {
  return questLogText()
    .split("\n")
    .filter((line) => line.trim())
    .flatMap((line) => {
      try {
        return [JSON.parse(line)];
      } catch {
        return []; // the line being written
      }
    });
}

// Counts of each output over the next `ms`, after letting `settle` pass.
async function countOver(ms, settle = 400) {
  await sleep(settle);
  const t0 = nowS();
  await sleep(ms);
  const counts = {};
  for (const event of eventsSince(t0)) {
    counts[event.id] = (counts[event.id] || 0) + 1;
  }
  return (id) => counts[id] || 0;
}

// How many frames in a 700 ms window count as "the page is sending". Measured
// from this run rather than fixed, because it depends on what the page has to
// draw: a 3D view of a real 96k-triangle export, rendered by SwiftShader in
// headless Chromium with no GPU, manages about a sixth of the frame rate the
// camera-panel views do. The checks that use it care whether frames stop and
// start again, not how fast they come, and a headset's GPU is not this one.
let frameFloor = 10;

function command(object) {
  fs.appendFileSync(CMD, `${JSON.stringify(object)}\n`);
}

function httpsUp() {
  return new Promise((resolve) => {
    const request = https.get(
      URL,
      { rejectUnauthorized: false },
      (response) => {
        response.resume();
        resolve(response.statusCode === 200);
      },
    );
    request.on("error", () => resolve(false));
    request.setTimeout(1000, () => {
      request.destroy();
      resolve(false);
    });
  });
}

// POST /offer the way connection.js does, with whatever body.
function postOffer(body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const request = https.request(
      `${URL}offer`,
      {
        method: "POST",
        rejectUnauthorized: false,
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(data),
        },
      },
      (response) => {
        response.resume();
        resolve(response.statusCode);
      },
    );
    request.on("error", reject);
    request.end(data);
  });
}

function writeDataflow() {
  const tls = path.join(RUN, "tls");
  fs.mkdirSync(tls);
  const openssl = spawnSync(
    "openssl",
    [
      "req",
      "-x509",
      "-newkey",
      "rsa:2048",
      "-nodes",
      "-days",
      "1",
      "-keyout",
      path.join(tls, "server.key"),
      "-out",
      path.join(tls, "server.crt"),
      "-subj",
      "/CN=127.0.0.1",
      "-addext",
      "subjectAltName=IP:127.0.0.1",
    ],
    { encoding: "utf8" },
  );
  if (openssl.status !== 0) {
    throw new Error(`openssl failed: ${openssl.stderr}`);
  }
  const outputs = [
    ...RECORDED,
    "pose_reference",
    "joystick_x_right",
    "joystick_y_right",
    "joystick_x_left",
    "joystick_y_left",
  ];
  const yaml = `nodes:
  - id: webxr
    path: dora-openarm-webxr
    env:
      PORT: "${PORT}"
      HOST: "127.0.0.1"
      TLS_CERTIFICATE_FILE: "${path.join(tls, "server.crt")}"
      TLS_KEY_FILE: "${path.join(tls, "server.key")}"
      ICE_SERVERS: "[]"
      POSE_TIMEOUT: "0.2"
      PAIRING_TOKEN: "${TOKEN}"
      QUEST_LOG: "${QUEST_LOG}"
      PIVOT_MODE: "latched"
      NECK_PIVOT_FILE: "${path.join(RUN, "neck_pivot.yaml")}"
      VIEW_CONFIGURATION_FILE: "${VIEW_FILE}"
${SCENE_VIEW ? `      SCENE_GLB: "${SCENE_GLB}"\n` : ""}    inputs:
      tick: dora/timer/millis/5
      hud: driver/hud
      control: driver/control
${SCENE_VIEW ? "      scene_state: {source: driver/scene_state, queue_size: 1}\n" : ""}    outputs:
${outputs.map((output) => `      - ${output}`).join("\n")}
  - id: driver
    path: ${path.join(HERE, "driver.py")}
    env:
      LOG: "${LOG}"
      CMD: "${CMD}"
    inputs:
      tick: dora/timer/millis/20
${RECORDED.map((output) => `      ${output}: {source: webxr/${output}, queue_size: 1000}`).join("\n")}
    outputs:
      - hud
      - control
${SCENE_VIEW ? "      - scene_state\n" : ""}`;
  const file = path.join(RUN, "dataflow.yaml");
  fs.writeFileSync(file, yaml);
  return file;
}

// Runs in the page before its modules. IWER stands in for the headset, and
// spies record what the page builds its peer with, what the node says on
// the control channel, and which haptic pulses the page asks for.
function installEmulator(forceEmptyIce) {
  const device = new IWER.XRDevice(IWER.metaQuest3);
  device.installRuntime({ forceInstall: true });
  window.__xr = device;
  window.__peerConfigurations = [];
  window.__controlMessages = [];
  window.__sentControl = [];
  window.__pulses = [];
  window.__emulated = { right: false };

  const NativePeer = window.RTCPeerConnection;
  // biome-ignore lint/complexity/useArrowFunction: called with `new`.
  window.RTCPeerConnection = function (configuration, ...rest) {
    window.__peerConfigurations.push(JSON.parse(JSON.stringify(configuration)));
    const pc = new NativePeer(
      forceEmptyIce ? { ...configuration, iceServers: [] } : configuration,
      ...rest,
    );
    pc.addEventListener("datachannel", (event) => {
      if (event.channel.label === "control") {
        event.channel.addEventListener("message", (message) => {
          try {
            window.__controlMessages.push(JSON.parse(message.data));
          } catch {}
        });
        // The other direction too (patch 17): what this page put on the
        // reliable channel, so a test outside the page can see that its log
        // really went to the node and not just into a buffer.
        const send = event.channel.send.bind(event.channel);
        event.channel.send = (text) => {
          try {
            window.__sentControl.push(JSON.parse(text));
          } catch {}
          return send(text);
        };
      }
    });
    return pc;
  };
  window.RTCPeerConnection.prototype = NativePeer.prototype;

  // IWER reports every XRFrame.getPose result as emulatedPosition: true
  // (all of its spaces are flagged emulated), which the patched page rightly
  // refuses. Report tracked poses as a headset does, and make the right
  // controller's aim pose look IMU-only on demand.
  const getPose = IWER.XRFrame.prototype.getPose;
  IWER.XRFrame.prototype.getPose = function (space, base) {
    const pose = getPose.call(this, space, base);
    if (!pose) {
      return pose;
    }
    const right = device.controllers.right;
    const emulated =
      window.__emulated.right &&
      right !== undefined &&
      space === right.inputSource.targetRaySpace;
    return { transform: pose.transform, emulatedPosition: emulated };
  };

  const actuator =
    device.controllers.right.inputSource.gamepad.hapticActuators[0];
  const prototype = Object.getPrototypeOf(actuator);
  const pulse = prototype.pulse;
  prototype.pulse = function (value, duration) {
    window.__pulses.push({ value, duration });
    return pulse.call(this, value, duration);
  };
}

async function run(page, doraExited, openPage) {
  const sessionOn = (target) =>
    target.evaluate(() => Boolean(window.__xr.activeSession));
  const inSession = () => sessionOn(page);
  const configurationsOn = (target) =>
    target.evaluate(
      () =>
        window.__controlMessages.filter((m) => m.type === "configuration")
          .length,
    );
  const configurations = () => configurationsOn(page);
  const controlTypes = (target) =>
    target.evaluate(() => window.__controlMessages.map((m) => m.type));
  // Pages opened besides the first, so the quit check can see them leave.
  const others = [];

  await section("page connects with no ICE servers (patch 1)", async () => {
    await waitFor("the configuration message", configurations, 20000);
    const [first] = await page.evaluate(() => window.__peerConfigurations);
    return expect(
      JSON.stringify(first.iceServers) === "[]",
      `iceServers=${JSON.stringify(first.iceServers)}`,
    );
  });

  await section("?ice= names the page's ICE servers (patch 1)", async () => {
    const other = await openPage(
      `${URL}?ice=stun:192.0.2.1:3478,stun:192.0.2.2:3478#k=${TOKEN}`,
    );
    const configuration = await waitFor("a peer on the second page", () =>
      other.evaluate(() => window.__peerConfigurations[0]),
    );
    await other.close();
    return expect(
      JSON.stringify(configuration.iceServers) ===
        '[{"urls":["stun:192.0.2.1:3478","stun:192.0.2.2:3478"]}]',
      `iceServers=${JSON.stringify(configuration.iceServers)}`,
    );
  });

  await section(
    "an offer without the pairing token gets 403 (patch 8)",
    async () => {
      const missing = await postOffer({ sdp: "v=0\r\n" });
      const wrong = await postOffer({ sdp: "v=0\r\n", token: `${TOKEN}x` });
      // With the token, the same garbage SDP is a 400, not a 500 (patch 10).
      const malformed = await postOffer({ sdp: "v=0\r\n", token: TOKEN });
      const unpaired = await openPage(URL);
      const status = await waitFor("the unpaired page's status", async () => {
        const text = await unpaired.locator("#log").innerText();
        return text.includes("Cannot reach") && text;
      });
      await unpaired.close();
      return expect(
        missing === 403 &&
          wrong === 403 &&
          malformed === 400 &&
          status.includes("403") &&
          status.includes("pairing token"),
        `no token=${missing} wrong token=${wrong} garbage with token=${malformed}; ` +
          `page without #k: ${JSON.stringify(status)}`,
      );
    },
  );

  await section(
    "a reload that lost the #k fragment still pairs (patch 8)",
    async () => {
      const other = await openPage(PAGE_URL);
      await waitFor("the configuration", () => configurationsOn(other));
      await other.evaluate(() =>
        history.replaceState(null, "", location.pathname),
      );
      await other.reload();
      await waitFor("the configuration after the reload", () =>
        configurationsOn(other),
      );
      const url = other.url();
      await other.close();
      return expect(!url.includes("#k="), `reloaded ${url}`);
    },
  );

  await section("Start opens a session and the node says ready", async () => {
    const t0 = nowS();
    await page.click("#start");
    await waitFor("a session", inSession);
    await waitFor("status ready", () => eventsSince(t0, "status").length > 0);
    // What this page can manage with nothing in its way, for the checks that
    // ask whether frames stopped (see frameFloor).
    const baseline = (await countOver(700))("vr_receive_times");
    frameFloor = Math.max(2, Math.floor(baseline / 2));
    return expect(
      // A camera view manages 40+; the heaviest 3D view on a host with no GPU
      // manages 5. Below 3 the page is not really sending and the later checks
      // would be measuring noise.
      baseline > 2,
      `${baseline} frames in 700 ms; "sending" is more than ${frameFloor}`,
    );
  });

  if (SCENE_VIEW) {
    // The 3D view, end to end through the real node (PATCHES.md 15): its
    // `scene` block, the glTF it serves, and a scene state travelling from a
    // dora input to the robot the headset draws. tests/web/scene3d.test.mjs
    // covers the view itself in detail; this is the wiring.
    await section(
      "the node serves the scene and the page draws it",
      async () => {
        const scene = await page.evaluate(
          () =>
            window.__controlMessages.find((m) => m.type === "configuration")
              .scene,
        );
        const stats = await waitFor("the 3D view", () =>
          page.evaluate(() => window.__oqtPage.scene?.stats),
        );
        const floor = await page.evaluate(
          () => window.__oqtPage.scene.floorSpace,
        );
        return expect(
          scene?.url === "/scene/scene.glb" &&
            typeof scene.hash === "string" &&
            stats.triangles > 0,
          `scene ${JSON.stringify(scene)}; ${stats.triangles} triangles, ` +
            `${stats.movingBodies} moving bodies, drawn in ${floor}`,
        );
      },
    );

    // The asset is the first-load cost in the headset, and over Wi-Fi it is
    // the bytes that cost. The node gzips it; this is the browser's own
    // accounting of what actually crossed the link, for the very fetch that
    // produced the triangles above.
    await section("the scene arrives gzipped and still loads", async () => {
      const timing = await waitFor("the scene fetch to be timed", () =>
        page.evaluate(() => {
          const entry = performance
            .getEntriesByType("resource")
            .find((e) => new URL(e.name).pathname === "/scene/scene.glb");
          // decodedBodySize is 0 until the response is complete.
          return entry?.decodedBodySize ? entry.toJSON() : null;
        }),
      );
      const stats = await page.evaluate(() => window.__oqtPage.scene?.stats);
      const share = (
        (100 * timing.transferSize) /
        timing.decodedBodySize
      ).toFixed(1);
      return expect(
        timing.encodedBodySize > 0 &&
          timing.encodedBodySize < timing.decodedBodySize &&
          timing.transferSize < timing.decodedBodySize &&
          stats.triangles > 0,
        `${timing.decodedBodySize} B of glTF arrived as ${timing.encodedBodySize} B encoded / ${timing.transferSize} B transferred (${share}%), and the page parsed ${stats.triangles} triangles out of it`,
      );
    });

    await section(
      "a scene_state input moves the robot in the headset",
      async () => {
        const trajectory = JSON.parse(fs.readFileSync(TRAJECTORY, "utf8"));
        const frame = trajectory.frames[7];
        // The tracked body a joint value moves FURTHEST, whatever this asset
        // calls it: an error anywhere in the chain above it lands there, and
        // the check below can only see a body that travels. Picked by what the
        // ground truth says it does rather than by its place in the list: the
        // committed fixture deliberately tracks a static table as well (so a
        // page that moved one would be caught), and scene_truth.py lists the
        // deepest body first, which for a real arm is the wrist but for a
        // hinge at the origin can be a body that only ever turns.
        const travel = (name) => {
          const poses = trajectory.frames.map((f) => f.bodies[name]);
          return Math.max(
            ...poses.flatMap((a) =>
              poses.map((b) =>
                Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]),
              ),
            ),
          );
        };
        const body = trajectory.tracked.reduce((best, name) =>
          travel(name) > travel(best) ? name : best,
        );
        const before = await page.evaluate(
          (name) => window.__oqtPage.scene.bodyPosition(name),
          body,
        );
        command({
          scene_state: {
            type: "state",
            v: 1,
            seq: 1,
            t: 0.1,
            flags: 0,
            qpos: frame.qpos,
          },
        });
        const after = await waitFor("the robot to move", async () => {
          const at = await page.evaluate(
            (name) => window.__oqtPage.scene.bodyPosition(name),
            body,
          );
          return at &&
            Math.hypot(...at.map((value, i) => value - before[i])) > 0.005
            ? at
            : null;
        });
        // Where MuJoCo puts that body for that qpos, through the page's own
        // placement: the whole chain, from a dora input to a pose in the room.
        const placement = await page.evaluate(
          () => window.__oqtPage.scene.placement,
        );
        const truth = frame.bodies[body];
        const want = [
          placement[0] * truth[0] +
            placement[4] * truth[1] +
            placement[8] * truth[2] +
            placement[12],
          placement[1] * truth[0] +
            placement[5] * truth[1] +
            placement[9] * truth[2] +
            placement[13],
          placement[2] * truth[0] +
            placement[6] * truth[1] +
            placement[10] * truth[2] +
            placement[14],
        ];
        const error = Math.hypot(...after.map((value, i) => value - want[i]));
        return expect(
          error < 1e-4,
          `${body} landed ${(error * 1000).toFixed(3)} mm from ${trajectory.truth || "MuJoCo"}'s ` +
            `answer, ${(Math.hypot(...after.map((v, i) => v - before[i])) * 100).toFixed(1)} cm ` +
            "from where it was",
        );
      },
    );

    await section("the binary encoding drives the same robot", async () => {
      // `oqt run sim` asks for `--scene-state-format binary` by default: 24 +
      // 4*nq bytes whatever the pose (788 on the compress cell) against
      // JSON's 1,341 while it moves, on the same half-duplex radio the
      // operator's hand poses come back over. That default is only safe if
      // the whole path takes it, so the same kind of frame goes out again as
      // bytes and has to land in the same place.
      const trajectory = JSON.parse(fs.readFileSync(TRAJECTORY, "utf8"));
      const frame = trajectory.frames[3];
      // The same body the check above picked, and for the same reason:
      // `tracked[0]` is the fixture's deliberately static table.
      const travel = (name) => {
        const poses = trajectory.frames.map((f) => f.bodies[name]);
        return Math.max(
          ...poses.flatMap((a) =>
            poses.map((b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])),
          ),
        );
      };
      const body = trajectory.tracked.reduce((best, name) =>
        travel(name) > travel(best) ? name : best,
      );
      const before = await page.evaluate(
        (name) => window.__oqtPage.scene.bodyPosition(name),
        body,
      );
      // oqt.sim.scene_state.encode_binary: u32 "OQT3", u8 version, u8 flags,
      // u16 seq, f64 sim time, u16 nq, u16 nobj, four bytes of padding, then
      // nq float32 joint values.
      const nq = frame.qpos.length;
      const buffer = new ArrayBuffer(24 + 4 * nq);
      const view = new DataView(buffer);
      view.setUint32(0, 0x4f515433, true);
      view.setUint8(4, 1);
      view.setUint8(5, 0);
      view.setUint16(6, 2, true);
      view.setFloat64(8, 0.2, true);
      view.setUint16(16, nq, true);
      view.setUint16(18, 0, true);
      frame.qpos.forEach((value, index) => {
        view.setFloat32(24 + 4 * index, value, true);
      });
      command({ scene_state_binary: Array.from(new Uint8Array(buffer)) });
      const after = await waitFor("the robot to move again", async () => {
        const at = await page.evaluate(
          (name) => window.__oqtPage.scene.bodyPosition(name),
          body,
        );
        return at &&
          Math.hypot(...at.map((value, i) => value - before[i])) > 0.001
          ? at
          : null;
      });
      const placement = await page.evaluate(
        () => window.__oqtPage.scene.placement,
      );
      const truth = frame.bodies[body];
      const want = [0, 1, 2].map(
        (axis) =>
          placement[axis] * truth[0] +
          placement[4 + axis] * truth[1] +
          placement[8 + axis] * truth[2] +
          placement[12 + axis],
      );
      const error = Math.hypot(...after.map((value, i) => value - want[i]));
      const asJson = JSON.stringify({
        type: "state",
        v: 1,
        seq: 2,
        t: 0.2,
        flags: 0,
        qpos: frame.qpos.map((value) => Math.round(value * 1e4) / 1e4),
      }).length;
      return expect(
        error < 1e-3,
        `${buffer.byteLength} bytes of binary put ${body} ${(error * 1000).toFixed(3)} mm ` +
          `from MuJoCo's answer (the same frame as JSON is ${asJson} bytes)`,
      );
    });
  }

  let firstPose = null;
  await section(
    "node publishes pose_right, grip_right and button_a",
    async () => {
      await page.evaluate(() => {
        const { right, left } = window.__xr.controllers;
        right.position.set(0.2, 1.3, -0.4);
        left.position.set(-0.2, 1.3, -0.4);
        right.updateButtonValue("squeeze", 1);
        right.updateButtonValue("trigger", 0.5);
        right.updateButtonValue("a-button", 1);
        left.updateButtonValue("x-button", 1);
      });
      const t0 = nowS();
      await waitFor("pose_right, grip_right=1 and button_a=true", () => {
        const events = eventsSince(t0);
        return (
          events.some((e) => e.id === "pose_right") &&
          events.some((e) => e.id === "button_a" && e.value[0] === true) &&
          events.some(
            (e) => e.id === "grip_right" && Math.abs(e.value[0] - 1) < 1e-6,
          )
        );
      });
      firstPose = eventsSince(t0, "pose_right").at(-1).value[0];
      return expect(
        firstPose.length === 8 && firstPose.every(Number.isFinite),
        `pose_right=[${firstPose.map((v) => v.toFixed(3)).join(", ")}]`,
      );
    },
  );

  await section(
    "moving the controller 20 cm forward moves pose_right x",
    async () => {
      await page.evaluate(() =>
        window.__xr.controllers.right.position.set(0.2, 1.3, -0.6),
      );
      const t0 = nowS();
      // Robot x is WebXR -z; the One Euro filter and 1 m/s cap take a moment.
      const pose = await waitFor("pose_right to move forward", () => {
        const poses = eventsSince(t0, "pose_right");
        const last = poses.at(-1);
        return last && last.value[0][0] > firstPose[0] + 0.15 && last.value[0];
      });
      return `x ${firstPose[0].toFixed(3)} -> ${pose[0].toFixed(3)}`;
    },
  );

  await section(
    "a second page cannot take over while the first sends frames (patch 9)",
    async () => {
      const other = await openPage(PAGE_URL);
      await waitFor("the second page's configuration", () =>
        configurationsOn(other),
      );
      const t0 = nowS();
      await other.click("#start");
      await waitFor("session-refused on the second page", async () =>
        (await controlTypes(other)).includes("session-refused"),
      );
      await waitFor(
        "the refused page to end its session",
        async () => !(await sessionOn(other)),
      );
      const status = await waitFor("the refused page's status", async () => {
        const text = await other.locator("#log").innerText();
        return text.includes("Press Start") && text;
      });
      const count = await countOver(600, 0);
      const types = await controlTypes(other);
      const first = await inSession();
      await other.close();
      return expect(
        first &&
          eventsSince(t0, "status").length === 0 &&
          count("pose_right") > 10 &&
          !types.includes("hud") &&
          status.includes("another headset is active"),
        `first page in session=${first} ` +
          `status outputs=${eventsSince(t0, "status").length} ` +
          `pose_right=${count("pose_right")} second page got ${JSON.stringify(types)}; ` +
          `its status: ${JSON.stringify(status)}`,
      );
    },
  );

  await section(
    "hud input reaches the page and pulses the right controller (patch 5)",
    async () => {
      const hud = {
        lines: ["REC 00:12  ep 3", "Pick up the cube.", "L:HOLD  R:FOLLOW"],
        recording: true,
        haptic: "right",
      };
      command({ hud });
      const message = await waitFor("a hud message", () =>
        page.evaluate(() =>
          window.__controlMessages.find((m) => m.type === "hud"),
        ),
      );
      expect(
        JSON.stringify(message) === JSON.stringify({ type: "hud", ...hud }),
        `message=${JSON.stringify(message)}`,
      );
      const pulses = await waitFor("a haptic pulse", () =>
        page.evaluate(() => window.__pulses.length > 0 && window.__pulses),
      );
      await sleep(300);
      await page.screenshot({ path: path.join(RUN, "hud.png") });
      return expect(
        JSON.stringify(pulses) === '[{"value":0.6,"duration":80}]',
        `pulses=${JSON.stringify(pulses)}`,
      );
    },
  );

  await section(
    "no frames while the session is visible-blurred (patch 3)",
    async () => {
      await page.evaluate(() =>
        window.__xr.updateVisibilityState("visible-blurred"),
      );
      const blurred = await countOver(700);
      await page.evaluate(() => window.__xr.updateVisibilityState("visible"));
      const visible = await countOver(700);
      return expect(
        blurred("vr_receive_times") === 0 &&
          visible("vr_receive_times") > frameFloor,
        `frames blurred=${blurred("vr_receive_times")} visible=${visible("vr_receive_times")} ` +
          `(more than ${frameFloor} is "sending")`,
      );
    },
  );

  await section(
    "one controller still sends its pose, grip and buttons (patch 4)",
    async () => {
      await page.evaluate(() => {
        window.__xr.controllers.left.connected = false;
      });
      const count = await countOver(700);
      await page.evaluate(() => {
        window.__xr.controllers.left.connected = true;
      });
      return expect(
        count("pose_right") > frameFloor &&
          count("grip_right") > 5 &&
          count("button_a") > 5 &&
          count("pose_left") === 0 &&
          count("button_x") === 0,
        `pose_right=${count("pose_right")} grip_right=${count("grip_right")} ` +
          `button_a=${count("button_a")} pose_left=${count("pose_left")} ` +
          `button_x=${count("button_x")}`,
      );
    },
  );

  await section(
    "emulatedPosition drops pose/trigger/grip, keeps buttons (patch 3)",
    async () => {
      await page.evaluate(() => {
        window.__emulated.right = true;
      });
      const count = await countOver(700);
      await page.evaluate(() => {
        window.__emulated.right = false;
      });
      return expect(
        count("pose_right") === 0 &&
          count("grip_right") === 0 &&
          count("trigger_right") === 0 &&
          count("button_a") > 5 &&
          count("pose_left") > 10,
        `pose_right=${count("pose_right")} grip_right=${count("grip_right")} ` +
          `trigger_right=${count("trigger_right")} button_a=${count("button_a")} ` +
          `pose_left=${count("pose_left")}`,
      );
    },
  );

  await section("hand input sources are ignored (patch 4)", async () => {
    await page.evaluate(() => {
      window.__xr.primaryInputMode = "hand";
    });
    const count = await countOver(700);
    await page.evaluate(() => {
      window.__xr.primaryInputMode = "controller";
    });
    await countOver(0, 400);
    return expect(
      count("vr_receive_times") > frameFloor &&
        count("pose_right") === 0 &&
        count("grip_right") === 0 &&
        count("button_a") === 0,
      `frames=${count("vr_receive_times")} pose_right=${count("pose_right")} ` +
        `grip_right=${count("grip_right")} button_a=${count("button_a")}`,
    );
  });

  await section(
    "frames pause for > 250 ms after a recenter (reset event)",
    async () => {
      const t0 = nowS();
      await sleep(300);
      const resetAt = nowS();
      await page.evaluate(() => window.__xr.recenter());
      await sleep(1200);
      const times = eventsSince(t0, "vr_receive_times").map((e) => e.t);
      let gap = 0;
      for (let i = 1; i < times.length; i++) {
        if (times[i] >= resetAt) {
          gap = Math.max(gap, times[i] - times[i - 1]);
        }
      }
      return expect(
        gap > 0.25 && gap < 1.0,
        `largest gap ${(gap * 1000).toFixed(0)} ms`,
      );
    },
  );

  await section(
    "Start after a session end works without a reload (patch 2)",
    async () => {
      await page.evaluate(() => window.__xr.activeSession.end());
      await waitFor(
        "the page to reconnect",
        async () => !(await inSession()) && (await configurations()) >= 2,
        20000,
      );
      const idle = await countOver(500, 200);
      const t0 = nowS();
      await page.click("#start");
      await waitFor("a second session", inSession);
      await waitFor(
        "a second status ready",
        () => eventsSince(t0, "status").length > 0,
      );
      await waitFor(
        "poses again",
        () => eventsSince(t0, "pose_right").length > 10,
      );
      return expect(
        idle("vr_receive_times") === 0,
        `frames between sessions=${idle("vr_receive_times")}`,
      );
    },
  );

  await section(
    "the reconnected page gets the last HUD again, without haptic",
    async () => {
      const huds = await page.evaluate(() =>
        window.__controlMessages.filter((m) => m.type === "hud"),
      );
      const pulses = await page.evaluate(() => window.__pulses.length);
      await page.screenshot({
        path: path.join(RUN, "hud_after_reconnect.png"),
      });
      return expect(
        huds.length === 2 &&
          huds[1].haptic === null &&
          huds[1].lines.length === 3 &&
          pulses === 1,
        `hud messages=${huds.length} last=${JSON.stringify(huds.at(-1))} pulses=${pulses}`,
      );
    },
  );

  await section(
    "an idle session is taken over and its page is told why (patch 9)",
    async () => {
      const other = await openPage(PAGE_URL);
      others.push(other);
      await waitFor("the second page's configuration", () =>
        configurationsOn(other),
      );
      // The first headset stops sending (its system menu is up) for longer
      // than the node's 1 s takeover idle.
      await page.evaluate(() =>
        window.__xr.updateVisibilityState("visible-blurred"),
      );
      await sleep(1300);
      const t0 = nowS();
      await other.click("#start");
      await waitFor("a session on the second page", () => sessionOn(other));
      await waitFor("status ready", () => eventsSince(t0, "status").length > 0);
      await waitFor(
        "the first page to leave its session",
        async () => !(await inSession()),
      );
      const status = await waitFor("the first page's status", async () => {
        const text = await page.locator("#log").innerText();
        return text.includes("Connected") && text;
      });
      await waitFor(
        "poses from the second page",
        () => eventsSince(t0, "pose_right").length > 10,
      );
      await page.evaluate(() => window.__xr.updateVisibilityState("visible"));
      const ended = (await controlTypes(page)).filter(
        (type) => type === "session-ended",
      ).length;
      return expect(
        ended === 1 &&
          status.startsWith("Another headset took over the session."),
        `session-ended messages=${ended}; first page: ${JSON.stringify(status)}`,
      );
    },
  );

  // Patch 17: the page's own log, against the real node. The details of the
  // buffer and the record shape are in tests/web/telemetry_units.test.mjs and
  // the browser end of it in tests/web/telemetry.test.mjs; what only this
  // test can show is that both transports really reach this node -- the
  // reliable channel a node that does not know the message type still
  // tolerates, and the HTTPS route it has to serve.
  await section(
    "the page's log reaches the node's channel (patch 17)",
    async () => {
      const t0 = nowS();
      // A real uncaught error, from the page's own task queue.
      await page.evaluate(() => {
        setTimeout(() => {
          throw new TypeError("a deliberate error from the browser test");
        }, 0);
      });
      const envelope = await waitFor("a telemetry envelope on the wire", () =>
        page.evaluate(() =>
          window.__sentControl
            .filter((message) => message.type === "log")
            .find((message) =>
              message.records.some((record) =>
                record.msg.includes("a deliberate error from the browser test"),
              ),
            ),
        ),
      );
      // Teleoperation is what must not notice: the poses have to keep
      // flowing across the envelope.
      await waitFor(
        "poses after the envelope",
        () => eventsSince(t0, "pose_right").length > 5,
      );
      // ...and the node has to have written it down.
      const line = await waitFor("the record in the node's quest.log", () =>
        questLog().find((entry) =>
          String(entry.msg || "").includes(
            "a deliberate error from the browser test",
          ),
        ),
      );
      return expect(
        envelope.v === 1 &&
          /^[0-9a-f]{16}$/.test(envelope.session) &&
          envelope.build.length > 0 &&
          !JSON.stringify(envelope).includes(TOKEN) &&
          line.level === "error" &&
          line.event === "window" &&
          line.via === "control" &&
          line.session === envelope.session &&
          !questLogText().includes(TOKEN),
        `build ${envelope.build}, session ${envelope.session}; ` +
          `quest.log has ${line.level}/${line.event} from ${line.peer} via ${line.via}, ` +
          "and no pairing token anywhere in it",
      );
    },
  );

  await section(
    "the node's own build reaches the page's log (patch 17)",
    async () => {
      // The node hashes the page files it serves and says so; the page
      // compares that with the build baked into the copy the headset is
      // actually running, so a cached page gives itself away.
      const build = await page.evaluate(
        () =>
          window.__controlMessages.find((m) => m.type === "configuration")
            ?.build,
      );
      const line = await waitFor("the build record", () =>
        questLog().find((entry) =>
          String(entry.msg || "").startsWith("the node is "),
        ),
      );
      return expect(
        typeof build === "string" &&
          build.includes("+") &&
          line.msg.includes(build),
        `the node serves ${build}; the page said ${JSON.stringify(line.msg)}`,
      );
    },
  );

  await section(
    "with WebRTC gone the log reaches the node over HTTPS (patch 17)",
    async () => {
      // The case the feature exists for: WebRTC itself is broken, so there
      // is no data channel to say anything on -- but the page is paired and
      // the node's HTTPS server is right there. Its own tab, so the session
      // under test is untouched.
      const orphan = await openPage(PAGE_URL, () => {
        // A peer that cannot be built at all: connect() throws before any
        // channel exists, which is what a headset with a blocked UDP path or
        // a browser refusing WebRTC looks like to this page.
        window.RTCPeerConnection = function () {
          throw new Error("WebRTC is not available on this device");
        };
      });
      others.push(orphan);
      await waitFor("the orphan page to give up on WebRTC", async () => {
        const text = await orphan.locator("#log").innerText();
        return text.includes("Cannot reach the node") && text;
      });
      await orphan.evaluate(() => {
        window.__oqtTelemetry.error("page", "no channel left to say this on");
      });
      const posted = await waitFor(
        "the node's own log to show the POST",
        () => {
          const log = fs.existsSync(path.join(RUN, "dora_run.log"))
            ? fs.readFileSync(path.join(RUN, "dora_run.log"), "utf8")
            : "";
          const line = log
            .split("\n")
            .reverse()
            .find((entry) => entry.includes('"POST /log HTTP/1.1"'));
          return line || null;
        },
        30000,
        250,
      );
      const status = /"POST \/log HTTP\/1\.1" (\d+)/.exec(posted);
      const line = await waitFor("the record in the node's quest.log", () =>
        questLog().find((entry) =>
          String(entry.msg || "").includes("no channel left to say this on"),
        ),
      );
      const buffered = await orphan.evaluate(
        () => window.__oqtTelemetry.stats().buffered,
      );
      return expect(
        status !== null &&
          status[1] === "204" &&
          line.via === "http" &&
          buffered === 0 &&
          !questLogText().includes(TOKEN),
        `${posted.trim().slice(-46)}; quest.log has it via ${line.via} ` +
          `from ${line.peer}, and the page's buffer is ${buffered}`,
      );
    },
  );

  await section(
    "control=quit ends the node and the page ends its session (patch 6)",
    async () => {
      command({ control: "quit" });
      const exit = await waitFor("dora run to exit", doraExited, 30000, 200);
      expect(exit.code === 0, `dora run exit ${JSON.stringify(exit)}`);
      await waitFor(
        "every session to end",
        async () => {
          for (const target of [page, ...others]) {
            if (await sessionOn(target)) {
              return false;
            }
          }
          return true;
        },
        15000,
      );
      return `dora run exit ${JSON.stringify(exit)}; page: ${await page.locator("#log").innerText()}`;
    },
  );
}

async function main() {
  if (!fs.existsSync(VIEW_FILE)) {
    throw new Error(`no view configuration ${VIEW_FILE}`);
  }
  if (SCENE_VIEW) {
    if (!fs.existsSync(SCENE_GLB)) {
      throw new Error(`no scene asset ${SCENE_GLB}`);
    }
    // Before dora starts: a missing ground truth is a mistake in the run, not
    // a failing check, and it takes a few seconds to compute.
    ensureTrajectory();
  }
  const dataflow = writeDataflow();
  const doraLog = fs.openSync(path.join(RUN, "dora_run.log"), "w");
  const dora = spawn(
    path.join(VENV_BIN, "dora"),
    ["run", dataflow, "--stop-after", "240s"],
    {
      cwd: RUN,
      env: {
        ...process.env,
        PATH: `${VENV_BIN}:${process.env.PATH}`,
        VIRTUAL_ENV: path.join(ROOT, ".venv"),
      },
      stdio: ["ignore", doraLog, doraLog],
    },
  );
  let doraExit = null;
  dora.on("exit", (code, signal) => {
    doraExit = { code, signal };
  });

  let browser = null;
  const consoleLines = [];
  try {
    await waitFor(
      "the node's HTTPS server",
      async () => {
        if (doraExit) {
          throw new Error(`dora run exited early: ${JSON.stringify(doraExit)}`);
        }
        return httpsUp();
      },
      90000,
      250,
    );
    browser = await chromium.launch({
      args: ["--ignore-certificate-errors", "--enable-unsafe-swiftshader"],
    });
    const context = await browser.newContext({ ignoreHTTPSErrors: true });
    let opened = 0;
    // Every page gets the emulator and has its console kept; each new page
    // is its own tab, so it starts with an empty sessionStorage.
    // `extra` runs after the emulator and before the page's own modules, for
    // a tab that has to start out broken in some particular way.
    const openPage = async (url, extra = null) => {
      const name = opened === 0 ? "" : `[page ${opened}] `;
      opened += 1;
      const target = await context.newPage();
      target.on("console", (message) =>
        consoleLines.push(`${name}${message.type()}: ${message.text()}`),
      );
      target.on("pageerror", (error) =>
        consoleLines.push(`pageerror: ${name}${error.message}`),
      );
      await target.addInitScript({ path: IWER });
      await target.addInitScript(installEmulator, FORCE_EMPTY_ICE);
      if (extra) {
        await target.addInitScript(extra);
      }
      await target.goto(url);
      return target;
    };
    const page = await openPage(PAGE_URL);

    await run(page, () => doraExit, openPage);

    await section("no uncaught page errors", async () => {
      const errors = consoleLines.filter(
        (line) =>
          line.startsWith("pageerror") &&
          // The patch 17 check throws one on purpose, to watch it come back
          // from the page as a telemetry record.
          !line.includes("a deliberate error from the browser test"),
      );
      return expect(errors.length === 0, errors.join(" | "));
    });
  } catch (error) {
    results.push({ name: "setup", ok: false });
    console.log(`FAIL setup -- ${error.stack || error}`);
  } finally {
    fs.writeFileSync(path.join(RUN, "console.log"), consoleLines.join("\n"));
    if (browser) {
      await browser.close();
    }
    if (doraExit === null) {
      dora.kill("SIGINT");
      await waitFor(
        "dora run to stop",
        () => doraExit !== null,
        30000,
        200,
      ).catch(() => dora.kill("SIGKILL"));
    }
  }

  const failed = results.filter((result) => !result.ok);
  console.log(
    `\n${results.length - failed.length} passed, ${failed.length} failed`,
  );
  if (process.env.KEEP_RUN || failed.length > 0) {
    console.log(`run directory: ${RUN}`);
  } else {
    fs.rmSync(RUN, { recursive: true, force: true });
  }
  process.exit(failed.length === 0 ? 0 : 1);
}

main();
