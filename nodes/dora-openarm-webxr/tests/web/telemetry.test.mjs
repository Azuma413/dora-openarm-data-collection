// openarm-quest-teleop addition to the vendored dora-openarm-webxr tests
// (Apache-2.0, like the rest of this directory).
//
// The page's own log, in a real browser (PATCHES.md 17).
//
//   node third_party/dora-openarm-webxr/tests/web/telemetry.test.mjs
//
// Needs Node.js 20+, Playwright's Chromium and IWER (`run.sh` installs both).
// No dora and no fixed port: a local HTTP server serves the page's own files
// and answers `POST /log` the way the node's sink does, and a second
// `RTCPeerConnection` in the same tab answers the page's offer the way the
// node's aiortc server does, opening the `control` channel. `telemetry.js`,
// `ar.js`, `connection.js` and `scene3d.js` are all the real thing; only the
// node behind them is not.
//
// This is the half that a Node.js fake cannot do: a real uncaught error, a
// real rejected promise, a real WebGL context loss, a real glTF that 404s, a
// real XR session ending, a real `sendBeacon` as the page goes away, and the
// real WebRTC-down case -- the node refusing the offer, which is when this
// feature has to work and the only transport left is HTTPS. It also measures
// what the log costs the XR frame, with telemetry on and off.
//
// tests/web/telemetry_units.test.mjs is the other half: the buffer, the
// redaction, the backoff.
//
// Environment: KEEP_RUN, HEADED, SETTLE (ms of frames per timing sample).

import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.join(HERE, "../../src/dora_openarm_webxr/static");
const FIXTURE = path.join(HERE, "fixture");
const IWER = path.join(HERE, "node_modules", "iwer", "build", "iwer.min.js");
const RUN = fs.mkdtempSync(path.join(os.tmpdir(), "oqt-telemetry-test-"));
const SETTLE = Number(process.env.SETTLE || 4000);

const trajectory = JSON.parse(
  fs.readFileSync(path.join(FIXTURE, "trajectory.json"), "utf8"),
);

// The fixture scene, as the node describes it in its configuration message.
const NODE_SCENE = {
  url: "/fixture/arm.glb",
  hash: "fixturehash",
  nq: trajectory.nq,
  objects: trajectory.objects || [],
};

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
    console.log(`FAIL ${name} -- ${error.stack || error.message}`);
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

// Every envelope the sink received, in arrival order, with what carried it.
const sink = [];

// When set, the server hands out an ar.js that will not parse: the blank-page
// case, where the page's whole module graph is dead and only the inline
// handler in index.html is left to notice.
let breakAr = false;

// The page's own files, the fixture, and the node's log sink.
function serve() {
  const types = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
    ".glb": "model/gltf-binary",
  };
  return http.createServer((request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    if (request.method === "POST" && url.pathname === "/log") {
      // The node's sink (T3), as this page needs it to behave: read the
      // envelope, answer 2xx. A refusal is checked in the unit test.
      const chunks = [];
      request.on("data", (chunk) => chunks.push(chunk));
      request.on("end", () => {
        try {
          sink.push({
            via: "https",
            at: Date.now(),
            body: JSON.parse(Buffer.concat(chunks).toString("utf8")),
          });
          response.writeHead(204).end();
        } catch (error) {
          response.writeHead(400).end(String(error));
        }
      });
      return;
    }
    if (breakAr && url.pathname === "/ar.js") {
      const broken = 'import { nothing } from "./does-not-exist.js"; this is(';
      response.writeHead(200, {
        "Content-Type": "text/javascript",
        "Cache-Control": "no-store",
        "Content-Length": broken.length,
      });
      response.end(broken);
      return;
    }
    let file = null;
    if (url.pathname === "/" || url.pathname === "/index.html") {
      file = path.join(STATIC, "index.html");
    } else if (url.pathname.startsWith("/fixture/")) {
      const wanted = path.normalize(url.pathname.slice("/fixture/".length));
      const resolved = path.join(FIXTURE, wanted);
      file = resolved.startsWith(FIXTURE) ? resolved : null;
    } else {
      const resolved = path.join(STATIC, path.normalize(url.pathname));
      file = resolved.startsWith(STATIC) ? resolved : null;
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
      // So the deliberately broken ar.js below is not served out of, or left
      // in, the browser's memory cache.
      "Cache-Control": "no-store",
    });
    response.end(body);
  });
}

// Runs in the page before its own modules: IWER for the headset, and a peer
// that answers the page's offer the way the node's aiortc server does. The
// peer can be told to refuse, which is the case this feature exists for.
function installEmulator({ configuration, scene }) {
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

  // Every envelope the page put on the reliable channel, and every offer it
  // made, so the test can see both transports from inside the page.
  const node = {
    control: null,
    pc: null,
    received: [],
    telemetry: [],
    offers: 0,
    refuse: false,
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
      // Everything else -- the page's modules, the glTF, and the telemetry
      // POST -- goes to the server for real.
      return realFetch(input, init);
    }
    node.offers += 1;
    if (node.refuse) {
      // The node is there but will not pair: no peer, no data channel, no
      // WebRTC at all. Exactly what patch 8 answers an unpaired page.
      return new Response("forbidden", { status: 403 });
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
          eyes: [],
        }),
      );
    });
    control.addEventListener("message", (message) => {
      const payload = JSON.parse(message.data);
      node.received.push(payload);
      if (payload.type === "log") {
        node.telemetry.push(payload);
      }
    });
    pc.createDataChannel("state", { ordered: false, maxRetransmits: 0 });
    await pc.setRemoteDescription({ type: "offer", sdp: offer.sdp });
    await pc.setLocalDescription(await pc.createAnswer());
    await gathered(pc);
    return new Response(JSON.stringify({ sdp: pc.localDescription.sdp }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
}

// Every record the node has seen on either transport, newest last.
function allRecords() {
  return sink.flatMap((entry) =>
    entry.body.records.map((record) => ({ ...record, via: entry.via })),
  );
}

async function controlRecords(page) {
  const envelopes = await page.evaluate(() => window.__node.telemetry);
  return envelopes.flatMap((envelope) =>
    envelope.records.map((record) => ({ ...record, via: "control", envelope })),
  );
}

// Everything that reached the node by either route. Which one carried a
// given record is not the page's decision to make on any one occasion -- it
// uses whichever is up -- so a check that only wants "it arrived" looks here.
async function delivered(page) {
  return [
    ...(await controlRecords(page)),
    ...sink.map((entry) =>
      entry.body.records.map((record) => ({
        ...record,
        via: "https",
        envelope: entry.body,
      })),
    ),
  ].flat();
}

async function main() {
  const server = serve();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  const base = `http://127.0.0.1:${port}/`;
  // The pairing token the page finds in its fragment and the sink expects in
  // the POST body. It must never appear in a record.
  const TOKEN = "unit-test-pairing-token-9f3a";
  const consoleLines = [];
  const browser = await chromium.launch({
    headless: !process.env.HEADED,
    args: ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"],
  });
  const context = await browser.newContext({
    viewport: { width: 1000, height: 700 },
  });
  let opened = 0;
  const openPage = async (options = {}) => {
    const name = opened === 0 ? "" : `[page ${opened}] `;
    opened += 1;
    const page = await context.newPage();
    page.on("console", (message) =>
      consoleLines.push(`${name}${message.type()}: ${message.text()}`),
    );
    page.on("pageerror", (error) =>
      consoleLines.push(`pageerror: ${name}${error.message}`),
    );
    await page.addInitScript({ path: IWER });
    if (options.disableTelemetry) {
      await page.addInitScript(() => {
        window.__oqtTelemetryConfig = { enabled: false };
      });
    }
    await page.addInitScript(installEmulator, {
      configuration: options.configuration || {
        view: "scene",
        session: { mode: "immersive-ar" },
        scene: { exo: { distance: 0.55, table_height: 0.95 } },
      },
      scene: options.scene === undefined ? NODE_SCENE : options.scene,
    });
    await page.goto(`${base}${options.query || ""}#k=${TOKEN}`);
    return page;
  };

  try {
    await run(openPage, base, TOKEN);
  } catch (error) {
    results.push({ name: "setup", ok: false });
    console.log(`FAIL setup -- ${error.stack || error}`);
  } finally {
    fs.writeFileSync(path.join(RUN, "console.log"), consoleLines.join("\n"));
    fs.writeFileSync(
      path.join(RUN, "sink.jsonl"),
      sink.map((entry) => JSON.stringify(entry)).join("\n"),
    );
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
    fs.rmSync(RUN, { recursive: true, force: true });
  }
  process.exit(failed.length === 0 ? 0 : 1);
}

async function run(openPage, base, token) {
  const page = await openPage();
  await waitFor("the page to connect", () =>
    page.evaluate(() => window.__node.received.length > 0),
  );

  await section(
    "the page introduces itself when a session starts",
    async () => {
      await page.click("#start");
      await waitFor("a session", () =>
        page.evaluate(() => Boolean(window.__xr.activeSession)),
      );
      const records = await waitFor("the env record", async () => {
        const found = (await controlRecords(page)).filter(
          (record) => record.event === "env",
        );
        return found.length > 0 ? found : null;
      });
      const [env] = records;
      const envelope = env.envelope;
      return expect(
        env.msg === "xr session started" &&
          /Chrome|Chromium|Mozilla/.test(env.data.ua) &&
          env.data.mode === "immersive-ar" &&
          env.data.view === "scene" &&
          typeof env.data.blend === "string" &&
          envelope.type === "log" &&
          envelope.v === 1 &&
          /^[0-9a-f]{16}$/.test(envelope.session) &&
          envelope.build.length > 0 &&
          !JSON.stringify(envelope).includes(token),
        `build ${envelope.build}, features "${env.data.features}", blend ${env.data.blend}, ` +
          `${env.data.ua.slice(0, 48)}…`,
      );
    },
  );

  await section(
    "an uncaught error reaches the node, with its stack",
    async () => {
      // Thrown from a timer, so it is a real uncaught error on the page's own
      // task queue rather than an exception Playwright would catch.
      await page.evaluate(() => {
        setTimeout(() => {
          throw new TypeError("the gripper ghost has no matrix");
        }, 0);
      });
      const record = await waitFor("the error to arrive", async () =>
        (await controlRecords(page)).find((entry) =>
          entry.msg.includes("the gripper ghost has no matrix"),
        ),
      );
      return expect(
        record.level === "error" &&
          record.event === "window" &&
          record.stack.includes("TypeError") &&
          typeof record.mono === "number" &&
          typeof record.t === "number",
        `${record.event}/${record.level} "${record.msg}" at ${record.data && record.data.file}`,
      );
    },
  );

  await section("a rejected promise reaches the node", async () => {
    await page.evaluate(() => {
      Promise.reject(new Error("the state channel went away"));
    });
    const record = await waitFor("the rejection to arrive", async () =>
      (await controlRecords(page)).find(
        (entry) =>
          entry.event === "promise" &&
          entry.msg.includes("the state channel went away"),
      ),
    );
    return expect(record.level === "error", `${record.level}: ${record.msg}`);
  });

  await section("console.error reaches the node and still prints", async () => {
    const printed = await page.evaluate(() => {
      const seen = [];
      const original = console.error;
      console.error = (...args) => {
        seen.push(args.join(" "));
        original.apply(console, args);
      };
      console.error("cannot pulse the controller: no actuator");
      console.error = original;
      return seen;
    });
    const record = await waitFor("the console line to arrive", async () =>
      (await controlRecords(page)).find(
        (entry) =>
          entry.event === "console" &&
          entry.msg.includes("cannot pulse the controller"),
      ),
    );
    return expect(
      record.level === "error" && printed.length === 1,
      `printed ${printed.length}, recorded "${record.msg}"`,
    );
  });

  await section(
    "an error every frame is counted, not sent every frame",
    async () => {
      const before = (await controlRecords(page)).length;
      // 120 of the same error as fast as the page will take them, which is
      // what a broken render loop does.
      await page.evaluate(async () => {
        for (let i = 0; i < 120; i += 1) {
          console.error("the same broken frame");
          await new Promise((resolve) => setTimeout(resolve, 2));
        }
      });
      const record = await waitFor("the counted record", async () =>
        (await controlRecords(page)).find(
          (entry) => entry.msg.includes("the same broken frame") && entry.n > 1,
        ),
      );
      const after = (await controlRecords(page)).length;
      return expect(
        record.n > 20 && after - before < 20,
        `n=${record.n} in ${after - before} new records`,
      );
    },
  );

  await section(
    "the session's end reaches the node, with a reason",
    async () => {
      await page.evaluate(() => window.__xr.activeSession.end());
      const record = await waitFor("the session end to arrive", async () =>
        (await delivered(page)).find((entry) =>
          entry.msg.startsWith("xr session ended"),
        ),
      );
      return expect(
        record.event === "xr" && record.msg.length > "xr session ended".length,
        `${record.msg} (by ${record.via})`,
      );
    },
  );

  // The session that just ended closed its 3D view, and closing one gives its
  // WebGL context back on purpose (scene3d.js `close()`). That is
  // housekeeping and must NOT be an error: otherwise every clean session end
  // puts one in the file and `oqt run` reports one after every good run.
  await section("closing a view gives its context back quietly", async () => {
    const record = await waitFor("the released context to arrive", async () =>
      (await delivered(page)).find(
        (entry) => entry.event === "webgl" && entry.msg.includes("released"),
      ),
    );
    const errors = (await delivered(page)).filter(
      (entry) => entry.event === "webgl" && entry.level === "error",
    );
    return expect(
      record.level === "info" && errors.length === 0,
      `${record.level}: ${record.msg} (by ${record.via}); ` +
        `webgl errors so far: ${errors.length}`,
    );
  });

  // Last on this page: losing the context stops the view drawing, so nothing
  // after it would be measuring the page the operator has.
  await section("a WebGL context lost reaches the node", async () => {
    // A LIVE view, so the loss below is one the page did not ask for -- the
    // failure that turns the headset black, not the teardown above.
    await page.click("#start");
    await waitFor("a session again", () =>
      page.evaluate(() => Boolean(window.__xr.activeSession)),
    );
    const canvas = await waitFor("a fresh 3D view", () =>
      page.evaluate(() =>
        window.__oqtPage.scene
          ? window.__oqtPage.scene.renderer.domElement.id || true
          : false,
      ),
    );
    await page.evaluate(() => {
      // The real thing: three.js asks WEBGL_lose_context to drop it, and the
      // canvas fires webglcontextlost exactly as a headset under memory
      // pressure would.
      window.__oqtPage.scene.renderer.forceContextLoss();
    });
    const record = await waitFor("the context loss to arrive", async () =>
      (await delivered(page)).find(
        (entry) => entry.event === "webgl" && entry.level === "error",
      ),
    );
    return expect(
      record.level === "error" &&
        record.msg.includes("scene3d") &&
        record.msg.includes("lost"),
      `${record.level}: ${record.msg} (by ${record.via}); fresh view=${canvas}`,
    );
  });

  await section(
    "with the node refusing the offer, the log arrives over HTTPS",
    async () => {
      const before = sink.length;
      // The case worth debugging: no data channel at all. The page keeps
      // trying to connect and keeps failing, which is all the operator sees.
      await page.evaluate(() => {
        window.__node.refuse = true;
        if (window.__node.pc) {
          window.__node.pc.close();
        }
      });
      await page.evaluate(() => {
        setTimeout(() => {
          throw new Error("nothing but HTTPS is left");
        }, 0);
      });
      await page.click("#start").catch(() => {});
      const envelope = await waitFor(
        "an envelope at the HTTPS sink",
        () =>
          sink
            .slice(before)
            .find((entry) =>
              entry.body.records.some((record) =>
                record.msg.includes("nothing but HTTPS is left"),
              ),
            ),
        20000,
      );
      const body = envelope.body;
      return expect(
        envelope.via === "https" &&
          body.token === token &&
          body.type === "log" &&
          !JSON.stringify(body.records).includes(token) &&
          body.records.some((record) => record.event === "webrtc"),
        `${body.records.length} records with the pairing token, reason "${body.reason}": ` +
          body.records
            .map((record) => `${record.event}/${record.msg.slice(0, 40)}`)
            .join(" | "),
      );
    },
  );

  await section(
    "a page that goes away beacons what it had not sent",
    async () => {
      const leaving = await openPage();
      await waitFor("the second page to connect", () =>
        leaving.evaluate(() => window.__node.received.length > 0),
      );
      const session = await leaving.evaluate(
        () => window.__oqtTelemetry.stats().session,
      );
      await leaving.evaluate(() => {
        // Recorded and not yet due: the ordinary flush is seconds away.
        window.__oqtTelemetry.error("page", "the last thing the page knew");
      });
      // A real pagehide: the page navigates away, and the browser sends the
      // beacon on its behalf.
      await leaving.goto("about:blank");
      const envelope = await waitFor("the beacon at the sink", () =>
        sink.find(
          (entry) =>
            entry.body.session === session &&
            entry.body.records.some((record) =>
              record.msg.includes("the last thing the page knew"),
            ),
        ),
      );
      await leaving.close();
      return expect(
        envelope.body.reason === "pagehide" && envelope.via === "https",
        `reason ${envelope.body.reason}, ${envelope.body.records.length} records`,
      );
    },
  );

  await section(
    "a page whose own ar.js will not parse still reports it",
    async () => {
      // The blank page: no module of the page's ever ran, so there is no
      // connection, no session and nothing on screen -- and the operator has
      // nothing to describe but "it does not work". Only the nine lines of
      // classic script in index.html see this, and telemetry.js drains them.
      breakAr = true;
      const before = sink.length;
      const broken = await openPage();
      const envelope = await waitFor("the parse error at the sink", () =>
        sink
          .slice(before)
          .find((entry) =>
            entry.body.records.some(
              (record) =>
                record.event === "window" &&
                /ar\.js|does-not-exist/.test(
                  `${record.msg} ${JSON.stringify(record.data || {})}`,
                ),
            ),
          ),
      );
      const started = await broken.evaluate(
        () => window.__node.offers > 0 || Boolean(window.__oqtPage),
      );
      await broken.close();
      breakAr = false;
      return expect(
        envelope.via === "https" && started === false,
        `${envelope.body.records.map((record) => record.msg).join(" | ")} ` +
          `-- with no connection of any kind (page modules ran: ${started})`,
      );
    },
  );

  await section(
    "a scene asset that 404s arrives with the URL that failed",
    async () => {
      const broken = await openPage({
        scene: { ...NODE_SCENE, url: "/fixture/no-such-scene.glb" },
      });
      const record = await waitFor("the asset error", async () => {
        const records = await controlRecords(broken);
        return records.find((entry) => entry.event === "asset");
      });
      const status = await broken.locator("#log").innerText();
      await broken.close();
      return expect(
        record.level === "error" &&
          record.data.url.includes("no-such-scene.glb") &&
          status.includes("3D scene unavailable"),
        `${record.msg} (url ${record.data.url}); page says ${JSON.stringify(status.split("\n").pop())}`,
      );
    },
  );

  await section("the log costs the XR frame nothing measurable", async () => {
    // The same view, the same asset, the same emulated headset, once with the
    // page log installed and once without it. What is compared is the page's
    // own per-frame work, which is what a 72 Hz headset lives on.
    const measure = async (options) => {
      const target = await openPage(options);
      await target.click("#start");
      await waitFor("a session", () =>
        target.evaluate(() => Boolean(window.__xr.activeSession)),
      );
      await waitFor("the 3D view", () =>
        target.evaluate(() => Boolean(window.__oqtPage.scene)),
      );
      await sleep(SETTLE);
      const timings = await target.evaluate(() =>
        window.__oqtPage.scene.timings(),
      );
      const frames = await target.evaluate(
        () => window.__oqtPage.scene.state.frames,
      );
      await target.close();
      return { ...timings, frames };
    };
    const off = await measure({ disableTelemetry: true });
    const on = await measure({});
    const shift = on.mean - off.mean;
    return expect(
      // What this check can and cannot say. Chromium quantises
      // performance.now() to 100 us, so one frame's cost is only ever a
      // multiple of that, and on a shared machine the mean over a few
      // hundred frames still wanders by tens of microseconds in either
      // direction -- runs of this have come out 60 us *faster* with
      // telemetry on. So the bound here is a noise bound: anything that
      // moved the frame by a fifth of a millisecond would be a real
      // regression, and the actual cost is measured directly in the next
      // check, three orders of magnitude below what this one could see.
      Math.abs(shift) < 0.2 &&
        Math.abs(on.median - off.median) <= 0.2 &&
        on.count > 50 &&
        off.count > 50,
      `mean ${off.mean.toFixed(3)} -> ${on.mean.toFixed(3)} ms ` +
        `(${shift >= 0 ? "+" : ""}${(shift * 1000).toFixed(0)} us/frame, ` +
        `inside this measurement's own noise), ` +
        `median ${off.median.toFixed(1)} -> ${on.median.toFixed(1)}, ` +
        `p95 ${off.p95.toFixed(1)} -> ${on.p95.toFixed(1)} ms, ` +
        `over ${off.frames} and ${on.frames} frames`,
    );
  });

  await section("what the per-frame hooks themselves cost", async () => {
    // The frame comparison above can only see what a 100 us clock lets it
    // see. This is the hooks measured directly, in the same browser, over
    // enough calls that the clock's resolution does not matter.
    const cost = await page.evaluate(() => {
      const telemetry = window.__oqtTelemetry;
      const rounds = 200000;
      // What ar.js calls for one XR frame: the frame, the pose it sent, and
      // the tracking state of the head and both controllers.
      const start = performance.now();
      for (let i = 0; i < rounds; i += 1) {
        const time = i * 13.9;
        telemetry.frame(time);
        telemetry.pose(time);
        telemetry.tracking("head", true);
        telemetry.tracking("right", true);
        telemetry.tracking("left", true);
      }
      const perFrame = ((performance.now() - start) * 1e6) / rounds;
      return { perFrame, rounds };
    });
    return expect(
      cost.perFrame < 500,
      `${cost.perFrame.toFixed(0)} ns per XR frame over ${cost.rounds} frames, ` +
        `against the 13,900,000 ns a 72 Hz frame has`,
    );
  });

  await section("the node saw both transports carry the log", async () => {
    const overHttps = allRecords().length;
    const overControl = (await controlRecords(page)).length;
    return expect(
      overHttps > 0 && overControl > 0,
      `${overControl} records on the control channel, ${overHttps} over HTTPS ` +
        `in ${sink.length} envelopes`,
    );
  });
}

main();
