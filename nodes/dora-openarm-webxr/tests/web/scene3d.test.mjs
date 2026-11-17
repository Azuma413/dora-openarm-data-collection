// openarm-quest-teleop addition to the vendored dora-openarm-webxr tests
// (Apache-2.0, like the rest of this directory).
//
// Browser test of the 3D view (PATCHES.md 15), in a real XRSession.
//
// The page is the one the node serves, byte for byte, opened in Playwright's
// Chromium with Meta's IWER emulating a Quest 3. What stands in for the node
// is a second RTCPeerConnection in the same tab: it answers the page's offer,
// opens the "control" and "state" channels the node opens, pushes a view
// configuration, and plays a recorded MuJoCo trajectory at 72 Hz. So
// connection.js, ar.js and scene3d.js are all the real thing, and only the
// network between them is not.
//
//   node third_party/dora-openarm-webxr/tests/web/scene3d.test.mjs
//
// Needs Node.js 20+ and this directory's node_modules (run.sh installs them).
// It starts no dora and takes no fixed port, so it can run next to anything.
//
// Environment:
//   KEEP_RUN     keep the run directory (eye captures, console log)
//   THROTTLE     CPU throttling factor for the frame budget check (default 4,
//                the prototype's stand-in for a Quest 3 CPU against this host)
//   SCENE_GLB    also run the asset checks against a real exported scene,
//                e.g. runs/<name>/scene.glb
//   CAPTURE      pixels per side of the eye captures (default 512)
//   HEADED       run Chromium headed (for looking at it yourself)

import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";
import { chromium } from "playwright";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.join(HERE, "../../src/dora_openarm_webxr/static");
const FIXTURE = path.join(HERE, "fixture");
const IWER = path.join(HERE, "node_modules", "iwer", "build", "iwer.min.js");
const RUN = fs.mkdtempSync(path.join(os.tmpdir(), "oqt-scene3d-test-"));
const THROTTLE = Number(process.env.THROTTLE || 4);
// How big the eye captures are written, for looking at them closely.
const CAPTURE = Number(process.env.CAPTURE || 512);

const trajectory = JSON.parse(
  fs.readFileSync(path.join(FIXTURE, "trajectory.json"), "utf8"),
);

// Where the fixture's own arm_origin is: the fixture has no such site, so the
// test names one, exactly as the node names the cell's.
const ARM_ORIGIN = [0, 0, 0.9];
// What the view configuration file says: where the robot stands.
const SCENE = {
  placement: "exo",
  arm_origin_world: ARM_ORIGIN,
  table_top_world_z: 0.52,
  exo: { distance: 1.15, table_height: 0.95, yaw: 0 },
  objects: ["cube"],
};
// What the node works out for itself and sends beside the view configuration
// (main.py _configure_scene): the asset it serves and what it read from it.
const NODE_SCENE = {
  url: "/fixture/arm.glb",
  hash: "fixturehash",
  bytes: 11520,
  nq: trajectory.nq,
  bodies: trajectory.tracked,
};
// table_height -> the height the arm origin itself is put at.
const EXPECTED_HEIGHT = 0.95 + ARM_ORIGIN[2] - SCENE.table_top_world_z;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const results = [];

function expect(condition, detail) {
  if (!condition) {
    throw new Error(detail);
  }
  return detail;
}

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

// Every URL the page asked the server for, so the test can see what it
// fetched the asset as.
const requestedUrls = [];

// The page's own files, the fixture, and whatever SCENE_GLB names.
function serve() {
  const types = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
    ".glb": "model/gltf-binary",
  };
  const roots = { "/static/": STATIC, "/fixture/": FIXTURE };
  const server = http.createServer((request, response) => {
    requestedUrls.push(request.url);
    const url = new URL(request.url, "http://127.0.0.1");
    let file = null;
    if (url.pathname === "/" || url.pathname === "/index.html") {
      file = path.join(STATIC, "index.html");
    } else if (url.pathname === "/scene.glb" && process.env.SCENE_GLB) {
      file = path.resolve(process.env.SCENE_GLB);
    } else {
      for (const [prefix, root] of Object.entries(roots)) {
        if (url.pathname.startsWith(prefix)) {
          const wanted = path.normalize(url.pathname.slice(prefix.length));
          const resolved = path.join(root, wanted);
          if (resolved.startsWith(root)) {
            file = resolved;
          }
        }
      }
      // The page's own modules are served next to it, as the node serves them.
      if (file === null) {
        const resolved = path.join(STATIC, path.normalize(url.pathname));
        if (resolved.startsWith(STATIC)) {
          file = resolved;
        }
      }
    }
    if (
      file === null ||
      !fs.existsSync(file) ||
      fs.statSync(file).isDirectory()
    ) {
      response.writeHead(404).end("not found");
      return;
    }
    const body = fs.readFileSync(file);
    response.writeHead(200, {
      "Content-Type": types[path.extname(file)] || "application/octet-stream",
      "Content-Length": body.length,
    });
    response.end(body);
  });
  return server;
}

// Runs in the page before its own modules: IWER for the headset, and a peer
// that answers the page's offer the way the node's aiortc server does.
function installEmulator({ configuration, scene, eyes, video }) {
  const device = new IWER.XRDevice(IWER.metaQuest3);
  device.installRuntime({ forceInstall: true });
  window.__xr = device;
  device.position.set(0, 1.6, 0);

  // IWER flags every pose emulatedPosition, which the patched page rightly
  // refuses to send; a headset reports tracked poses as not emulated.
  const getPose = IWER.XRFrame.prototype.getPose;
  IWER.XRFrame.prototype.getPose = function (space, base) {
    const pose = getPose.call(this, space, base);
    return pose ? { transform: pose.transform, emulatedPosition: false } : pose;
  };
  const getViewerPose = IWER.XRFrame.prototype.getViewerPose;
  IWER.XRFrame.prototype.getViewerPose = function (base) {
    const pose = getViewerPose.call(this, base);
    if (!pose) {
      return pose;
    }
    // Keep the views and the transform, drop the emulated flag.
    return new Proxy(pose, {
      get(target, property) {
        if (property === "emulatedPosition") {
          return false;
        }
        const value = target[property];
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
  };

  const node = {
    frames: 0,
    lastFrame: null,
    control: null,
    state: null,
    received: [],
    stateSequence: 0,
    sent: 0,
  };
  window.__node = node;

  function gathered(pc) {
    if (pc.iceGatheringState === "complete") {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      pc.addEventListener("icegatheringstatechange", () => {
        if (pc.iceGatheringState === "complete") {
          resolve();
        }
      });
    });
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = String(input);
    if (!url.endsWith("offer")) {
      return realFetch(input, init);
    }
    const offer = JSON.parse(init.body);
    const pc = new RTCPeerConnection({ iceServers: [] });
    node.pc = pc;
    const control = pc.createDataChannel("control");
    node.control = control;
    control.addEventListener("open", () => {
      control.send(
        JSON.stringify({
          type: "configuration",
          view_configuration: configuration,
          ...(scene ? { scene } : {}),
          calibration: false,
          eyes,
        }),
      );
    });
    control.addEventListener("message", (message) => {
      node.received.push(JSON.parse(message.data));
    });
    // The node's own unreliable channel for the simulation's state.
    const state = pc.createDataChannel("state", {
      ordered: false,
      maxRetransmits: 0,
    });
    node.state = state;
    pc.addEventListener("datachannel", (event) => {
      if (event.channel.label !== "xr") {
        return;
      }
      event.channel.addEventListener("message", (message) => {
        node.frames += 1;
        node.lastFrame = JSON.parse(message.data);
      });
    });
    await pc.setRemoteDescription({ type: "offer", sdp: offer.sdp });
    if (video) {
      // A head camera, so the camera panel has something to show. Drawn
      // once; the track keeps sending it.
      const canvas = document.createElement("canvas");
      canvas.width = 640;
      canvas.height = 360;
      const context = canvas.getContext("2d");
      const paint = () => {
        context.fillStyle = "#1c3f6e";
        context.fillRect(0, 0, 640, 360);
        context.fillStyle = "#ffd166";
        context.fillRect(40 + ((Date.now() / 40) % 400), 120, 160, 120);
        requestAnimationFrame(paint);
      };
      paint();
      const stream = canvas.captureStream(30);
      // Bound between the remote and the local description, so it binds to a
      // transceiver the page already offered, as aiortc does.
      pc.addTrack(stream.getVideoTracks()[0], stream);
      node.videoTrack = stream.getVideoTracks()[0];
    }
    await pc.setLocalDescription(await pc.createAnswer());
    await gathered(pc);
    return new Response(JSON.stringify({ sdp: pc.localDescription.sdp }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  // Play a recorded trajectory on the state channel at the sim's rate.
  window.__play = (frames, { hz = 72, flags = 0, loop = true } = {}) => {
    window.__stop();
    let at = 0;
    node.playing = setInterval(() => {
      if (node.state.readyState !== "open") {
        return;
      }
      const frame = frames[at % frames.length];
      at += 1;
      if (!loop && at > frames.length) {
        window.__stop();
        return;
      }
      node.stateSequence += 1;
      node.sent += 1;
      node.state.send(
        JSON.stringify({
          type: "state",
          v: 1,
          seq: node.stateSequence,
          t: at / hz,
          flags,
          qpos: frame,
        }),
      );
    }, 1000 / hz);
  };
  window.__stop = () => {
    if (node.playing) {
      clearInterval(node.playing);
      node.playing = null;
    }
  };
}

// A minimal PNG writer, so the eye captures can be looked at.
function writePng(file, width, height, pixels) {
  const stride = width * 4;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    // WebGL reads bottom-up; PNG rows run top-down.
    const source = (height - 1 - y) * stride;
    raw[y * (stride + 1)] = 0;
    pixels.copy(raw, y * (stride + 1) + 1, source, source + stride);
  }
  const chunk = (type, data) => {
    const length = Buffer.alloc(4);
    length.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(body) >>> 0);
    return Buffer.concat([length, body, crc]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 6; // RGBA
  fs.writeFileSync(
    file,
    Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      chunk("IHDR", header),
      chunk("IDAT", zlib.deflateSync(raw)),
      chunk("IEND", Buffer.alloc(0)),
    ]),
  );
}

const CRC_TABLE = (() => {
  const table = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c;
  }
  return table;
})();

function crc32(buffer) {
  let c = 0xffffffff;
  for (let i = 0; i < buffer.length; i++) {
    c = CRC_TABLE[(c ^ buffer[i]) & 0xff] ^ (c >>> 8);
  }
  return c ^ 0xffffffff;
}

// Capture what each eye drew this frame and write it out, over a flat grey
// "room" as well, since in immersive-ar everything the scene does not cover
// is transparent and the compositor fills it with the passthrough cameras.
async function captureEyes(page, name, { over = null } = {}) {
  const views = await page.evaluate(
    async (size) => {
      const shots = await window.__oqtPage.scene.captureEyes(size);
      const encode = (values) => {
        const bytes = new Uint8Array(values);
        let text = "";
        for (let i = 0; i < bytes.length; i += 0x8000) {
          text += String.fromCharCode.apply(
            null,
            bytes.subarray(i, i + 0x8000),
          );
        }
        return btoa(text);
      };
      return shots.map((shot) => ({
        eye: shot.eye,
        width: shot.width,
        height: shot.height,
        data: encode(shot.pixels),
      }));
    },
    { width: CAPTURE, height: CAPTURE },
  );
  const written = [];
  for (const view of views) {
    const pixels = Buffer.from(view.data, "base64");
    const file = path.join(RUN, `${name}-${view.eye}.png`);
    writePng(file, view.width, view.height, pixels);
    written.push(file);
    if (over) {
      const composited = Buffer.from(pixels);
      for (let i = 0; i < composited.length; i += 4) {
        const alpha = composited[i + 3] / 255;
        for (let c = 0; c < 3; c++) {
          composited[i + c] = Math.round(
            composited[i + c] * alpha + over[c] * (1 - alpha),
          );
        }
        composited[i + 3] = 255;
      }
      const overFile = path.join(RUN, `${name}-${view.eye}-over-room.png`);
      writePng(overFile, view.width, view.height, composited);
      written.push(overFile);
    }
  }
  return { views, written };
}

// Send one state message as the node would, numbered after the last one so
// the page does not drop it as a late delivery.
function sendState(page, payload) {
  return page.evaluate((message) => {
    window.__node.stateSequence += 1;
    window.__node.state.send(
      JSON.stringify({
        type: "state",
        v: 1,
        seq: window.__node.stateSequence,
        t: 0,
        flags: 0,
        ...message,
      }),
    );
    return window.__node.stateSequence;
  }, payload);
}

// How much of a capture the scene actually covers, and how bright it is: a
// black headset is the one outcome the fallbacks exist to prevent. The
// fixture is one small arm on one small table 1.15 m away, so it covers a few
// per cent of an eye; the thresholds below are about "something real is
// drawn", not about how big the robot is.
const COVERS_SOMETHING = 0.01;

function coverage(pixels) {
  let drawn = 0;
  let lit = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i + 3] > 8) {
      drawn += 1;
      if (pixels[i] + pixels[i + 1] + pixels[i + 2] > 60) {
        lit += 1;
      }
    }
  }
  const total = pixels.length / 4;
  return { drawn: drawn / total, lit: lit / total };
}

async function main() {
  const server = serve();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  const url = `http://127.0.0.1:${port}/`;
  const consoleLines = [];
  const browser = await chromium.launch({
    headless: !process.env.HEADED,
    args: ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"],
  });
  const context = await browser.newContext({
    viewport: { width: 1200, height: 800 },
  });
  let opened = 0;
  const openPage = async (options) => {
    const name = opened === 0 ? "" : `[page ${opened}] `;
    opened += 1;
    const page = await context.newPage();
    page.on("console", (message) =>
      consoleLines.push(`${name}${message.type()}: ${message.text()}`),
    );
    page.on("pageerror", (error) =>
      consoleLines.push(`pageerror: ${name}${error.stack || error.message}`),
    );
    await page.addInitScript({ path: IWER });
    await page.addInitScript(installEmulator, options);
    await page.goto(url);
    return page;
  };

  try {
    await run(openPage, consoleLines);
  } catch (error) {
    results.push({ name: "setup", ok: false });
    console.log(`FAIL setup -- ${error.stack || error}`);
  } finally {
    fs.writeFileSync(path.join(RUN, "console.log"), consoleLines.join("\n"));
    await browser.close();
    server.close();
  }

  const failed = results.filter((result) => !result.ok);
  console.log(
    `\n${results.length - failed.length} passed, ${failed.length} failed`,
  );
  if (process.env.KEEP_RUN || failed.length > 0) {
    console.log(`run directory: ${RUN}`);
  } else {
    console.log(`captures: ${RUN}`);
  }
  process.exit(failed.length === 0 ? 0 : 1);
}

async function run(openPage, consoleLines) {
  const view = {
    view: "scene",
    session: { mode: "immersive-ar" },
    scene: SCENE,
    pose: {
      frame_offset: [-0.1, 0, 0.23],
      neck_pivot_offset: [0, -0.075, 0.08],
    },
  };
  const page = await openPage({
    configuration: view,
    scene: NODE_SCENE,
    eyes: [],
    video: false,
  });
  const qpos = trajectory.frames.map((frame) => frame.qpos);

  await section("the page loads the node's scene asset", async () => {
    await waitFor("the 3D view", () =>
      page.evaluate(() => Boolean(window.__oqtPage?.scene)),
    );
    const stats = await page.evaluate(() => window.__oqtPage.scene.stats);
    const settings = await page.evaluate(() => window.__oqtPage.scene.settings);
    return expect(
      stats.triangles > 0 &&
        stats.movingBodies === 4 &&
        settings.mode === "exo",
      `${stats.triangles} triangles, ${stats.meshes} meshes, ` +
        `${stats.movingBodies} moving bodies, placement ${settings.mode}`,
    );
  });

  await section("the asset is fetched with the node's hash on it", async () => {
    // A headset that cached a previous run's asset at the same path has to
    // fetch this one instead of drawing the wrong robot.
    const asset = requestedUrls.filter((requested) =>
      requested.includes("arm.glb"),
    );
    return expect(
      asset.length > 0 &&
        asset.every((requested) => requested.includes("h=fixturehash")),
      `asset requests: ${JSON.stringify(asset)}`,
    );
  });

  await section("Start opens a session and the scene is drawn", async () => {
    await page.click("#start");
    await waitFor("a session", () =>
      page.evaluate(() => Boolean(window.__xr.activeSession)),
    );
    await waitFor("frames at the node", () =>
      page.evaluate(() => window.__node.frames > 10),
    );
    const anchored = await waitFor("the scene to anchor", () =>
      page.evaluate(() => window.__oqtPage.scene.state.anchor),
    );
    const floor = await page.evaluate(() => window.__oqtPage.scene.floorSpace);
    return expect(
      Number.isFinite(anchored.yaw) && floor.length > 0,
      `anchored at [${anchored.position.x.toFixed(2)}, ${anchored.position.y.toFixed(2)}, ${anchored.position.z.toFixed(2)}], ` +
        `yaw ${anchored.yaw.toFixed(3)}, drawn in ${floor}`,
    );
  });

  // Where the placement rule says arm_origin should be, from the anchor the
  // page latched, and how far it actually is from there.
  async function ruleError(target) {
    const placement = await target.evaluate(() => ({
      matrix: window.__oqtPage.scene.placement,
      anchor: window.__oqtPage.scene.state.anchor,
      floor: window.__oqtPage.scene.floorSpace,
      settings: window.__oqtPage.scene.settings,
      nudge: window.__oqtPage.scene.state.nudge,
    }));
    const placed = applyMatrix(placement.matrix, ARM_ORIGIN);
    const { anchor, settings, nudge } = placement;
    const floorY =
      placement.floor === "local-floor" ? 0 : anchor.position.y - 1.6;
    const forward = settings.distance + nudge.forward;
    const lateral = settings.lateral + nudge.lateral;
    const want = [
      anchor.position.x +
        Math.cos(anchor.yaw) * lateral -
        Math.sin(anchor.yaw) * forward,
      floorY + EXPECTED_HEIGHT + nudge.height,
      anchor.position.z -
        Math.sin(anchor.yaw) * lateral -
        Math.cos(anchor.yaw) * forward,
    ];
    return {
      placed,
      want,
      error: Math.hypot(
        placed[0] - want[0],
        placed[1] - want[1],
        placed[2] - want[2],
      ),
    };
  }

  await section(
    "the placement puts arm_origin where the rule says",
    async () => {
      const { placed, want, error } = await ruleError(page);
      return expect(
        error < 1e-4,
        `arm_origin at [${placed.map((v) => v.toFixed(3)).join(", ")}], ` +
          `rule says [${want.map((v) => v.toFixed(3)).join(", ")}] (${(error * 1000).toFixed(2)} mm)`,
      );
    },
  );

  await section("the robot follows the sim's state stream", async () => {
    // Nothing yet: the scene stands at the pose it was exported in and says
    // so rather than pretending to be the robot.
    await waitFor("the no-state notice", () =>
      page.evaluate(() => window.__oqtPage.scene.state.notice),
    );
    const idle = await page.evaluate(() =>
      window.__oqtPage.scene.bodyPosition("wrist"),
    );

    await page.evaluate((frames) => window.__play(frames), qpos);
    await waitFor("state messages", () =>
      page.evaluate(() => window.__oqtPage.scene.state.sequence > 5),
    );
    // Follow one exact frame: stop the stream on a known one and compare the
    // wrist against MuJoCo's own answer for that qpos, through the placement.
    await page.evaluate(() => window.__stop());
    const frame = trajectory.frames[7];
    await sendState(page, { qpos: frame.qpos });
    await sleep(200);
    const got = await page.evaluate(() =>
      window.__oqtPage.scene.bodyPosition("wrist"),
    );
    const matrix = await page.evaluate(() => window.__oqtPage.scene.placement);
    const want = applyMatrix(matrix, frame.bodies.wrist.slice(0, 3));
    const error = Math.hypot(
      got[0] - want[0],
      got[1] - want[1],
      got[2] - want[2],
    );
    const moved = Math.hypot(
      got[0] - idle[0],
      got[1] - idle[1],
      got[2] - idle[2],
    );
    // And the notice goes away once the state is flowing again.
    await page.evaluate((frames) => window.__play(frames), qpos);
    await waitFor("the notice to clear", () =>
      page.evaluate(() => window.__oqtPage.scene.state.notice === ""),
    );
    return expect(
      error < 1e-4 && moved > 0.05,
      `wrist ${(error * 1000).toFixed(3)} mm from MuJoCo's answer, ` +
        `${(moved * 100).toFixed(1)} cm from the rest pose`,
    );
  });

  await section(
    "a free object follows its own pose in the message",
    async () => {
      await page.evaluate(() => window.__stop());
      const turn = Math.SQRT1_2;
      const pose = [0.55, -0.15, 0.71, turn, 0, turn, 0];
      // The qpos in the same message puts the cube somewhere else: the poses
      // the sim sends for its free objects are the ones that must win.
      await sendState(page, { qpos: trajectory.frames[0].qpos, obj: pose });
      await sleep(200);
      const got = await page.evaluate(() =>
        window.__oqtPage.scene.bodyPosition("cube"),
      );
      const matrix = await page.evaluate(
        () => window.__oqtPage.scene.placement,
      );
      const want = applyMatrix(matrix, pose.slice(0, 3));
      const error = Math.hypot(
        got[0] - want[0],
        got[1] - want[1],
        got[2] - want[2],
      );
      await page.evaluate((frames) => window.__play(frames), qpos);
      return expect(
        error < 1e-4,
        `cube ${(error * 1000).toFixed(3)} mm from its pose`,
      );
    },
  );

  await section("a message delivered late is not drawn", async () => {
    await page.evaluate(() => window.__stop());
    const sequence = await sendState(page, { qpos: trajectory.frames[3].qpos });
    await sleep(150);
    const current = await page.evaluate(() =>
      window.__oqtPage.scene.bodyPosition("wrist"),
    );
    // An older sequence, with a very different pose: it must be dropped, as
    // an unordered channel occasionally delivers one late.
    await page.evaluate(
      ([values, late]) =>
        window.__node.state.send(
          JSON.stringify({ type: "state", v: 1, seq: late, qpos: values }),
        ),
      [trajectory.frames[9].qpos, sequence - 10],
    );
    await sleep(150);
    const after = await page.evaluate(() =>
      window.__oqtPage.scene.bodyPosition("wrist"),
    );
    const moved = Math.hypot(
      after[0] - current[0],
      after[1] - current[1],
      after[2] - current[2],
    );
    await page.evaluate((frames) => window.__play(frames), qpos);
    return expect(moved < 1e-6, `the stale message moved the wrist ${moved} m`);
  });

  await section(
    "the thumbstick nudges the scene, within its limit",
    async () => {
      const before = await page.evaluate(
        () => window.__oqtPage.scene.placement,
      );
      await page.evaluate(() => {
        window.__xr.controllers.left.updateAxes("thumbstick", 0, -1);
      });
      await sleep(500);
      await page.evaluate(() => {
        window.__xr.controllers.left.updateAxes("thumbstick", 0, 0);
      });
      await sleep(120);
      const nudge = await page.evaluate(
        () => window.__oqtPage.scene.state.nudge,
      );
      const after = await page.evaluate(() => window.__oqtPage.scene.placement);
      const from = applyMatrix(before, ARM_ORIGIN);
      const to = applyMatrix(after, ARM_ORIGIN);
      const moved = Math.hypot(
        to[0] - from[0],
        to[1] - from[1],
        to[2] - from[2],
      );
      expect(
        nudge.forward > 0.05 && Math.abs(moved - nudge.forward) < 1e-4,
        `nudge ${JSON.stringify(nudge)} moved the scene ${moved.toFixed(4)} m`,
      );
      // Held over, it keeps moving and stays inside its limit rather than
      // walking the robot out of the room (the limit itself is checked to the
      // last millimetre in scene_units.test.mjs).
      await page.evaluate(() => {
        window.__xr.controllers.left.updateAxes("thumbstick", 0, -1);
      });
      await sleep(1500);
      await page.evaluate(() => {
        window.__xr.controllers.left.updateAxes("thumbstick", 0, 0);
      });
      await sleep(120);
      const held = await page.evaluate(
        () => window.__oqtPage.scene.state.nudge,
      );
      // And the placement still follows the rule, nudge included.
      const { error } = await ruleError(page);
      return expect(
        held.forward > nudge.forward &&
          held.forward <= 1.0 + 1e-9 &&
          error < 1e-4,
        `nudged ${nudge.forward.toFixed(3)} m, then held to ${held.forward.toFixed(3)} m; ` +
          `the rule still holds to ${(error * 1000).toFixed(2)} mm`,
      );
    },
  );

  await section("an engaged clutch locks the nudge out", async () => {
    const before = await page.evaluate(
      () => window.__oqtPage.scene.state.nudge,
    );
    // Flag bit 1 is the left clutch: that hand is driving an arm now.
    await page.evaluate(
      ([frames]) => {
        window.__stop();
        window.__play(frames, { flags: 2 });
      },
      [qpos],
    );
    await waitFor("the clutch flag", () =>
      page.evaluate(() => window.__oqtPage.scene.state.flags.clutchLeft),
    );
    await page.evaluate(() => {
      window.__xr.controllers.left.updateAxes("thumbstick", 1, -1);
    });
    await sleep(600);
    const after = await page.evaluate(() => window.__oqtPage.scene.state.nudge);
    await page.evaluate(() => {
      window.__xr.controllers.left.updateAxes("thumbstick", 0, 0);
    });
    await page.evaluate((frames) => window.__play(frames), qpos);
    return expect(
      after.forward === before.forward && after.lateral === before.lateral,
      `nudge ${JSON.stringify(before)} -> ${JSON.stringify(after)} with the clutch engaged`,
    );
  });

  await section(
    "a recenter re-anchors the scene to the new head pose",
    async () => {
      const before = await page.evaluate(() => ({
        anchor: window.__oqtPage.scene.state.anchor,
        count: window.__oqtPage.scene.state.reanchors,
      }));
      // Walking about does not move the robot: the anchor is latched.
      await page.evaluate(() => window.__xr.position.set(0.5, 1.6, -0.3));
      await sleep(600);
      const walked = await page.evaluate(
        () => window.__oqtPage.scene.state.reanchors,
      );
      expect(
        walked === before.count,
        `the scene re-anchored ${walked - before.count} times while walking`,
      );
      // A recenter is a gap in the frames the node sees, so it latches a new
      // pivot -- and the scene has to move with it.
      await page.evaluate(() => window.__xr.recenter());
      await waitFor(
        "a re-anchor",
        () =>
          page.evaluate(
            (count) => window.__oqtPage.scene.state.reanchors > count,
            walked,
          ),
        8000,
        100,
      ).catch(() => {});
      const after = await page.evaluate(() => ({
        anchor: window.__oqtPage.scene.state.anchor,
        count: window.__oqtPage.scene.state.reanchors,
      }));
      // The robot has to end up in front of the operator again, not behind
      // them: a recenter moves the origin the scene was placed against.
      const { error } = await ruleError(page);
      return expect(
        after.count > before.count && error < 1e-4,
        `re-anchored ${after.count - before.count} time(s) on the recenter ` +
          `(and ${walked - before.count} while walking); the rule holds to ` +
          `${(error * 1000).toFixed(2)} mm at the new anchor`,
      );
    },
  );

  await section("the HUD is drawn in the 3D view", async () => {
    await page.evaluate(() => {
      window.__node.control.send(
        JSON.stringify({
          type: "hud",
          lines: ["REC 00:12  ep 3", "Peel the compress.", "L:HOLD  R:FOLLOW"],
          recording: true,
          haptic: null,
        }),
      );
    });
    await waitFor("the HUD text", () =>
      page.evaluate(() => window.__oqtPage.hud.lines.length === 3),
    );
    await sleep(200);
    const drawn = await page.evaluate(
      () => window.__oqtPage.scene.scene.getObjectByName("hud").visible,
    );
    return expect(drawn === true, `hud mesh visible=${drawn}`);
  });

  await section("both eyes draw the scene over passthrough", async () => {
    const { views, written } = await captureEyes(page, "scene-ar", {
      over: [110, 112, 116],
    });
    const covers = views.map((viewShot) =>
      coverage(Buffer.from(viewShot.data, "base64")),
    );
    const eyes = views.map((viewShot) => viewShot.eye).join(",");
    return expect(
      views.length === 2 &&
        covers.every(
          (cover) => cover.drawn > COVERS_SOMETHING && cover.lit > 0.005,
        ) &&
        covers.every((cover) => cover.drawn < 0.98),
      `eyes ${eyes}; covered ${covers.map((c) => `${(c.drawn * 100).toFixed(1)}%`).join(" / ")} ` +
        `(lit ${covers.map((c) => `${(c.lit * 100).toFixed(1)}%`).join(" / ")}); ` +
        `${written.length} images in the run directory`,
    );
  });

  await section(
    "the two eyes see the scene from different places",
    async () => {
      const { views } = await captureEyes(page, "scene-stereo");
      const left = Buffer.from(views[0].data, "base64");
      const right = Buffer.from(views[1].data, "base64");
      let different = 0;
      for (let i = 0; i < left.length; i += 4) {
        if (Math.abs(left[i] - right[i]) > 6) {
          different += 1;
        }
      }
      const fraction = different / (left.length / 4);
      return expect(
        fraction > 0.002,
        `${(fraction * 100).toFixed(2)}% of pixels differ between the eyes`,
      );
    },
  );

  await section(
    "a stalled state stream freezes rather than blanks",
    async () => {
      await page.evaluate(() => window.__stop());
      const notice = await waitFor("the stale notice", () =>
        page.evaluate(() => window.__oqtPage.scene.state.notice),
      );
      // Read after the stream stopped, so this is the pose the robot is held
      // at rather than one it was passing through.
      const before = await page.evaluate(() =>
        window.__oqtPage.scene.bodyPosition("wrist"),
      );
      const frames = await page.evaluate(
        () => window.__oqtPage.scene.state.frames,
      );
      await sleep(500);
      const after = await page.evaluate(() =>
        window.__oqtPage.scene.bodyPosition("wrist"),
      );
      const drewMore = await page.evaluate(
        (was) => window.__oqtPage.scene.state.frames - was,
        frames,
      );
      const { views } = await captureEyes(page, "scene-stalled");
      const cover = coverage(Buffer.from(views[0].data, "base64"));
      // The same line reaches the operator inside the headset (the HUD panel)
      // and whoever is looking at the page.
      const hud = await page.evaluate(() => window.__oqtPage.sceneNotice);
      const onHud = await page.evaluate(
        () => window.__oqtPage.scene.scene.getObjectByName("hud").visible,
      );
      await page.evaluate((played) => window.__play(played), qpos);
      const held = Math.hypot(...after.map((value, i) => value - before[i]));
      return expect(
        notice.includes("stopped") &&
          hud === notice &&
          onHud === true &&
          drewMore > 10 &&
          cover.drawn > COVERS_SOMETHING &&
          held < 1e-6,
        `notice ${JSON.stringify(notice)}; ${drewMore} frames drawn while stalled, ` +
          `the robot held to ${held} m, ${(cover.drawn * 100).toFixed(1)}% of the eye still covered`,
      );
    },
  );

  await section(
    "the state channel closing does not stop the view",
    async () => {
      await page.evaluate(() => {
        window.__stop();
        window.__node.state.close();
      });
      await sleep(600);
      const frames = await page.evaluate(
        () => window.__oqtPage.scene.state.frames,
      );
      await sleep(400);
      const later = await page.evaluate(
        () => window.__oqtPage.scene.state.frames,
      );
      const nodeFrames = await page.evaluate(() => window.__node.frames);
      await sleep(300);
      const nodeLater = await page.evaluate(() => window.__node.frames);
      const { views } = await captureEyes(page, "scene-channel-gone");
      const cover = coverage(Buffer.from(views[0].data, "base64"));
      return expect(
        later > frames + 5 &&
          nodeLater > nodeFrames + 5 &&
          cover.drawn > COVERS_SOMETHING,
        `${later - frames} frames drawn and ${nodeLater - nodeFrames} poses sent ` +
          `after the channel closed; ${(cover.drawn * 100).toFixed(1)}% of the eye covered`,
      );
    },
  );

  await section(`frame budget at ${THROTTLE}x CPU throttling`, async () => {
    const cdp = await page.context().newCDPSession(page);
    const measure = async (rate) => {
      await cdp.send("Emulation.setCPUThrottlingRate", { rate });
      await page.evaluate(() => {
        window.__oqtPage.scene.state.frameAt = 0;
      });
      await sleep(3000);
      return page.evaluate(() => window.__oqtPage.scene.timings());
    };
    const plain = await measure(1);
    const throttled = await measure(THROTTLE);
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: 1 });
    // 13.9 ms is the whole budget at 72 Hz; the page's own work has to leave
    // room for the compositor.
    return expect(
      plain.median < 13.9 && throttled.median < 13.9,
      `1x: ${plain.median.toFixed(2)} ms median, ${plain.p95.toFixed(2)} p95 (${plain.count} frames); ` +
        `${THROTTLE}x: ${throttled.median.toFixed(2)} ms median, ${throttled.p95.toFixed(2)} p95 (${throttled.count} frames)`,
    );
  });

  // A second page, with a node that has no scene to give: the fallback.
  await section(
    "no asset falls back to passthrough with a notice",
    async () => {
      const bare = await openPage({
        configuration: { view: "scene", session: { mode: "immersive-ar" } },
        eyes: [],
        video: false,
      });
      await waitFor("the page to connect", () =>
        bare.evaluate(
          () => window.__node.received.length >= 0 && window.__oqtPage,
        ),
      );
      await waitFor("a notice", () =>
        bare.evaluate(() => window.__oqtPage.sceneNotice),
      );
      const noScene = await bare.evaluate(
        () => window.__oqtPage.scene === null,
      );
      await bare.click("#start");
      await waitFor("a session", () =>
        bare.evaluate(() => Boolean(window.__xr.activeSession)),
      );
      await waitFor("frames at the node", () =>
        bare.evaluate(() => window.__node.frames > 10),
      );
      const notice = await bare.evaluate(() => window.__oqtPage.sceneNotice);
      const log = await bare.locator("#log").innerText();
      // In the headset too: with no 3D view there is no mesh to draw it on,
      // so it goes on the HUD panel the 2D views draw through their shader.
      const hud = await bare.evaluate(() => window.__oqtPage.hud);
      await bare.close();

      // The same failure in "scene+panel", where there IS a camera panel left.
      const withPanel = await openPage({
        configuration: {
          view: "scene+panel",
          session: { mode: "immersive-ar" },
          panel: { distance: 1.6, width: 1.2 },
        },
        eyes: ["right"],
        video: true,
      });
      await waitFor("a notice", () =>
        withPanel.evaluate(() => window.__oqtPage.sceneNotice),
      );
      const panelNotice = await withPanel.evaluate(
        () => window.__oqtPage.sceneNotice,
      );
      await withPanel.close();
      return expect(
        noScene &&
          notice.includes("3D scene") &&
          log.includes("3D scene") &&
          hud.notice === notice &&
          // "scene" renders no camera at all, so it must not send the operator
          // looking for a camera panel that does not exist; "scene+panel" has one.
          !notice.includes("camera view") &&
          notice.includes("passthrough") &&
          panelNotice.includes("camera view"),
        `scene=${noScene ? "null" : "built"}; notice ${JSON.stringify(notice)}; ` +
          `on the HUD panel: ${JSON.stringify(hud.notice)}; ` +
          `scene+panel says ${JSON.stringify(panelNotice)}`,
      );
    },
  );

  await section("an asset that is not the node's is refused", async () => {
    // A headset that cached an earlier run's scene at the same path would
    // apply every joint value to the wrong body. The node says how many joint
    // values its own export has; a file with another number is not it.
    const stale = await openPage({
      configuration: {
        view: "scene",
        session: { mode: "immersive-ar" },
        scene: SCENE,
      },
      scene: { ...NODE_SCENE, nq: NODE_SCENE.nq + 3 },
      eyes: [],
      video: false,
    });
    const notice = await waitFor("a notice", () =>
      stale.evaluate(() => window.__oqtPage.sceneNotice),
    );
    const noScene = await stale.evaluate(() => window.__oqtPage.scene === null);
    await stale.click("#start");
    await waitFor("a session", () =>
      stale.evaluate(() => Boolean(window.__xr.activeSession)),
    );
    await waitFor("frames at the node", () =>
      stale.evaluate(() => window.__node.frames > 10),
    );
    await stale.close();
    return expect(
      noScene && notice.includes("joint values"),
      `scene=${noScene ? "null" : "built"}; notice ${JSON.stringify(notice)}`,
    );
  });

  await section("scene+panel draws the camera panel as well", async () => {
    const panelPage = await openPage({
      configuration: {
        view: "scene+panel",
        session: { mode: "immersive-ar" },
        scene: SCENE,
        panel: { lock: "room", distance: 1.6, width: 1.2 },
      },
      scene: NODE_SCENE,
      eyes: ["right"],
      video: true,
    });
    await waitFor("the 3D view", () =>
      panelPage.evaluate(() => Boolean(window.__oqtPage?.scene)),
    );
    await panelPage.click("#start");
    await waitFor("a session", () =>
      panelPage.evaluate(() => Boolean(window.__xr.activeSession)),
    );
    await panelPage.evaluate((frames) => window.__play(frames), qpos);
    const visible = await waitFor(
      "the camera panel",
      () =>
        panelPage.evaluate(
          () =>
            window.__oqtPage.scene.scene.getObjectByName("camera_panel")
              ?.visible,
        ),
      20000,
    );
    const { views, written } = await captureEyes(panelPage, "scene-panel", {
      over: [110, 112, 116],
    });
    const cover = coverage(Buffer.from(views[0].data, "base64"));
    await panelPage.close();
    return expect(
      visible === true && cover.drawn > 0.05,
      `panel visible, ${(cover.drawn * 100).toFixed(1)}% of the eye covered, ` +
        `${written.length} images written`,
    );
  });

  await section(
    "immersive-vr draws the same scene on its own floor",
    async () => {
      const vrPage = await openPage({
        configuration: {
          view: "scene",
          session: { mode: "immersive-vr" },
          scene: SCENE,
        },
        scene: NODE_SCENE,
        eyes: [],
        video: false,
      });
      await waitFor("the 3D view", () =>
        vrPage.evaluate(() => Boolean(window.__oqtPage?.scene)),
      );
      await vrPage.click("#start");
      await waitFor("a session", () =>
        vrPage.evaluate(() => Boolean(window.__xr.activeSession)),
      );
      await vrPage.evaluate((frames) => window.__play(frames), qpos);
      await sleep(500);
      const { views, written } = await captureEyes(vrPage, "scene-vr");
      const cover = coverage(Buffer.from(views[0].data, "base64"));
      const ground = await vrPage.evaluate(
        () =>
          window.__oqtPage.scene.scene.getObjectByName("room_ground").visible,
      );
      await vrPage.close();
      return expect(
        ground === true && cover.drawn > 0.9,
        `ground drawn=${ground}, ${(cover.drawn * 100).toFixed(1)}% of the eye covered, ` +
          `${written.length} images written`,
      );
    },
  );

  // The fixture is a toy. A run against a scene the exporter really wrote --
  // SCENE_GLB=runs/<name>/scene.glb -- is what says whether the operator's
  // own cell loads, draws and fits in the frame budget.
  if (process.env.SCENE_GLB) {
    await section("a real exported scene loads and draws", async () => {
      const armOrigin = (process.env.SCENE_ARM_ORIGIN || "0.185,0,1.34")
        .split(",")
        .map(Number);
      const realPage = await openPage({
        configuration: {
          view: "scene",
          session: { mode: "immersive-ar" },
          scene: {
            ...SCENE,
            arm_origin_world: armOrigin,
            table_top_world_z: Number(process.env.SCENE_TABLE_Z || 1.005),
            objects: [],
          },
        },
        scene: { url: "/scene.glb" },
        eyes: [],
        video: false,
      });
      await waitFor(
        "the 3D view",
        () => realPage.evaluate(() => Boolean(window.__oqtPage?.scene)),
        60000,
      );
      const stats = await realPage.evaluate(() => window.__oqtPage.scene.stats);
      await realPage.click("#start");
      await waitFor("a session", () =>
        realPage.evaluate(() => Boolean(window.__xr.activeSession)),
      );
      await realPage.evaluate(() => {
        window.__node.control.send(
          JSON.stringify({
            type: "hud",
            lines: [
              "REC 00:31  ep 7",
              "Peel the compress.",
              "L:FOLLOW  R:FOLLOW",
            ],
            recording: true,
            haptic: null,
          }),
        );
      });
      await sleep(2500);
      const timings = await realPage.evaluate(() =>
        window.__oqtPage.scene.timings(),
      );
      const cdp = await realPage.context().newCDPSession(realPage);
      await cdp.send("Emulation.setCPUThrottlingRate", { rate: THROTTLE });
      await realPage.evaluate(() => {
        window.__oqtPage.scene.state.frameAt = 0;
      });
      await sleep(3000);
      const throttled = await realPage.evaluate(() =>
        window.__oqtPage.scene.timings(),
      );
      await cdp.send("Emulation.setCPUThrottlingRate", { rate: 1 });
      const { views, written } = await captureEyes(realPage, "real-scene", {
        over: [110, 112, 116],
      });
      const cover = coverage(Buffer.from(views[0].data, "base64"));
      await realPage.close();
      return expect(
        stats.triangles > 1000 &&
          cover.drawn > COVERS_SOMETHING &&
          timings.median < 13.9,
        `${stats.triangles} triangles, ${stats.meshes} meshes, ` +
          `${stats.movingBodies} moving bodies; ` +
          `1x: ${timings.median.toFixed(2)} ms median / ${timings.p95.toFixed(2)} p95, ` +
          `${THROTTLE}x: ${throttled.median.toFixed(2)} / ${throttled.p95.toFixed(2)}; ` +
          `${(cover.drawn * 100).toFixed(1)}% of the eye; ${written.length} images`,
      );
    });
  }

  await section("no uncaught page errors", async () => {
    const errors = consoleLines.filter((line) => line.startsWith("pageerror"));
    return expect(errors.length === 0, errors.join(" | "));
  });
}

function applyMatrix(elements, point) {
  const [x, y, z] = point;
  return [
    elements[0] * x + elements[4] * y + elements[8] * z + elements[12],
    elements[1] * x + elements[5] * y + elements[9] * z + elements[13],
    elements[2] * x + elements[6] * y + elements[10] * z + elements[14],
  ];
}

main();
