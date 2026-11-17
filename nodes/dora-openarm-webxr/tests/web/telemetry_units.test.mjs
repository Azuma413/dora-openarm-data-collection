// openarm-quest-teleop addition to the vendored dora-openarm-webxr tests
// (Apache-2.0, like the rest of this directory).
//
// The page log's own rules, in Node.js alone (PATCHES.md 17). No browser, no
// dora, about a second:
//
//   node third_party/dora-openarm-webxr/tests/web/telemetry_units.test.mjs
//
// Each case builds a fresh fake browser on globalThis -- window, document,
// console, navigator, sessionStorage, fetch, sendBeacon -- and then imports
// static/telemetry.js with a cache-busting query, so the module installs
// itself on that fake exactly as it installs itself on a real page. What is
// checked here is what a browser test cannot make happen on demand: a buffer
// that overflows, a sink that refuses, a token that must not leak, a page
// that goes away.
//
// tests/web/telemetry.test.mjs is the other half: the same module in a real
// Chromium, against a real XR session and a real WebRTC peer.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.join(HERE, "../../src/dora_openarm_webxr/static");
const MODULE = pathToFileURL(path.join(STATIC, "telemetry.js")).href;
const ROOT = path.join(HERE, "../..");

// The real console, captured before any case replaces globalThis.console
// with its own fake (which is the whole point of the console hook check).
const say = console.log.bind(console);

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
    say(`PASS ${name}${detail ? ` -- ${detail}` : ""}`);
  } catch (error) {
    results.push({ name, ok: false });
    say(`FAIL ${name} -- ${error.stack || error.message}`);
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// A browser, as far as telemetry.js is concerned. Everything it touches is
// here, so a case can make the node refuse, the storage throw, or the page go
// away, none of which a real browser does on request.
let instance = 0;

// One network for every case. Each page's module instance keeps its timers
// running after its case is over (they are unref'd, not cancelled), so a
// later flush from an earlier page would otherwise land in the current
// page's fetch. Envelopes carry their session id; that is what routes them.
const pages = new Map();

function routed(kind, url, text, options) {
  let session = null;
  try {
    session = JSON.parse(text).session;
  } catch {
    session = null;
  }
  const page = pages.get(session);
  if (!page) {
    // An envelope from a page no longer under test: accepted and dropped.
    return { ok: true };
  }
  return page[kind](url, text, options);
}

globalThis.fetch = async (url, options) => {
  const answer = routed("post", url, options.body, options);
  if (answer instanceof Error) {
    throw answer;
  }
  return answer;
};

async function fakePage({
  hash = "",
  search = "",
  config = null,
  storage = new Map(),
  respond = () => ({ ok: true }),
  beacon = () => true,
} = {}) {
  const listeners = new Map();
  const posts = [];
  const beacons = [];
  const printed = [];
  const emit = (type, event) => {
    for (const handler of listeners.get(type) || []) {
      handler(event);
    }
  };
  const target = {
    addEventListener(type, handler) {
      if (!listeners.has(type)) {
        listeners.set(type, []);
      }
      listeners.get(type).push(handler);
    },
    removeEventListener() {},
  };
  globalThis.addEventListener = target.addEventListener;
  globalThis.removeEventListener = target.removeEventListener;
  globalThis.window = globalThis;
  globalThis.location = {
    href: `https://node.invalid/page.html${search}${hash}`,
    hash,
    search,
    origin: "https://node.invalid",
    pathname: "/page.html",
  };
  globalThis.sessionStorage = {
    getItem: (key) => (storage.has(key) ? storage.get(key) : null),
    setItem: (key, value) => storage.set(key, String(value)),
    removeItem: (key) => storage.delete(key),
  };
  globalThis.document = {
    visibilityState: "visible",
    addEventListener: target.addEventListener,
  };
  globalThis.console = {
    error: (...args) => printed.push(`error ${args.join(" ")}`),
    warn: (...args) => printed.push(`warn ${args.join(" ")}`),
    log: (...args) => printed.push(`log ${args.join(" ")}`),
  };
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      userAgent: "Mozilla/5.0 (X11; Linux x86_64) Quest/test",
      hardwareConcurrency: 8,
      sendBeacon: (url, body) => routed("beacon", url, String(body.__text)),
    },
  });
  globalThis.Blob = class {
    constructor(parts, options) {
      this.__text = parts.join("");
      this.type = options ? options.type : "";
    }
  };
  globalThis.__oqtEarly = [];
  globalThis.__oqtTelemetry = undefined;
  globalThis.__oqtTelemetryConfig = config || undefined;

  instance += 1;
  const module = await import(`${MODULE}?case=${instance}`);
  const telemetry = globalThis.__oqtTelemetry || module.telemetry;
  const page = {
    telemetry,
    module,
    posts,
    beacons,
    printed,
    storage,
    emit,
    listeners,
    post(url, text, options) {
      const body = JSON.parse(text);
      posts.push({ url, body, options });
      return respond(posts.length, body);
    },
    beacon(url, text) {
      beacons.push({ url, body: text });
      return beacon();
    },
    setVisibility(state) {
      globalThis.document.visibilityState = state;
      emit("visibilitychange", {});
    },
  };
  pages.set(telemetry.stats().session, page);
  return page;
}

async function main() {
  await section("it installs itself and publishes one session id", async () => {
    const page = await fakePage();
    const stats = page.telemetry.stats();
    return expect(
      /^[0-9a-f]{16}$/.test(stats.session) && stats.build.length > 0,
      `session=${stats.session} build=${stats.build}`,
    );
  });

  await section(
    "the page's build identity is the node's own version",
    async () => {
      // A headset showing a page cached from an older run is invisible unless
      // the page says which build it is. This is the pin that keeps the
      // constant honest.
      const pyproject = fs.readFileSync(
        path.join(ROOT, "pyproject.toml"),
        "utf8",
      );
      const version = /^version = "([^"]+)"/m.exec(pyproject);
      const source = fs.readFileSync(path.join(STATIC, "telemetry.js"), "utf8");
      const build = /^const BUILD = "([^"]+)";/m.exec(source);
      return expect(
        version && build && version[1] === build[1],
        `pyproject ${version && version[1]} vs page ${build && build[1]}`,
      );
    },
  );

  await section("a page out of the headset's cache says so", async () => {
    // The node hashes the files it serves and pushes that build; the page
    // knows the build it was itself built from. A version that is not this
    // page's means the headset is running something the node did not serve,
    // which is a whole class of reports that otherwise reads as a node gone
    // mad -- so it is an error, which the node mirrors to its own stderr.
    const current = await fakePage();
    const build = current.telemetry.stats().build;
    current.telemetry.nodeBuild(`${build}+1234abcd`);
    const matched = current.telemetry.peek();

    const stale = await fakePage();
    stale.telemetry.nodeBuild("9.9.9+deadbeef");
    const [mismatch] = stale.telemetry.peek();
    const sent = [];
    stale.telemetry.attach((text) => {
      sent.push(JSON.parse(text));
      return true;
    });
    await sleep(20);
    return expect(
      matched.length === 1 &&
        matched[0].level === "info" &&
        mismatch.level === "error" &&
        mismatch.msg.includes("cached page") &&
        mismatch.data.node === "9.9.9+deadbeef" &&
        // Every envelope carries what the node said it was, so the node can
        // tell which of its own builds the page came from.
        sent[0].node === "9.9.9+deadbeef" &&
        sent[0].build === build,
      `matched: ${matched[0].msg}; stale: ${mismatch.msg}`,
    );
  });

  await section("an uncaught error becomes one record", async () => {
    const page = await fakePage();
    const error = new TypeError("cannot read x of null");
    page.emit("error", {
      message: "Uncaught TypeError: cannot read x of null",
      filename: "https://node.invalid/ar.js",
      lineno: 640,
      colno: 12,
      error,
    });
    const [entry] = page.telemetry.peek();
    return expect(
      entry.level === "error" &&
        entry.event === "window" &&
        entry.msg.includes("cannot read x of null") &&
        entry.stack.includes("TypeError") &&
        entry.data.file === "ar.js" &&
        entry.data.line === 640 &&
        typeof entry.t === "number" &&
        typeof entry.mono === "number" &&
        entry.record === 1,
      JSON.stringify({ ...entry, stack: entry.stack.slice(0, 40) }),
    );
  });

  await section("a rejected promise becomes one record", async () => {
    const page = await fakePage();
    page.emit("unhandledrejection", { reason: new Error("no scene") });
    const [entry] = page.telemetry.peek();
    return expect(
      entry.event === "promise" &&
        entry.level === "error" &&
        entry.msg.includes("no scene"),
      entry.msg,
    );
  });

  await section(
    "console.error is recorded, printed, and cannot recurse",
    async () => {
      const page = await fakePage();
      // The hook itself logs: nothing here may call console.* on the delivery
      // path, but if anything ever does, it must pass straight through
      // instead of making a record that makes a record.
      console.error("first", new Error("boom"));
      console.warn("second");
      console.log("third");
      const records = page.telemetry.peek();
      const levels = records.map((entry) => `${entry.level}:${entry.event}`);
      return expect(
        page.printed.length === 3 &&
          page.printed[0].startsWith("error first") &&
          levels.join(" ") === "error:console warn:console" &&
          records[0].stack.includes("Error: boom"),
        `printed=${page.printed.length} records=${levels.join(" ")}`,
      );
    },
  );

  await section("console.log is kept out unless asked for", async () => {
    const quiet = await fakePage();
    console.log("noise");
    const verbose = await fakePage({ search: "?telemetry=verbose" });
    console.log("noise");
    return expect(
      quiet.telemetry.peek().length === 0 &&
        verbose.telemetry.peek().length === 1 &&
        verbose.telemetry.peek()[0].level === "debug",
      `quiet=${quiet.telemetry.peek().length} verbose=${verbose.telemetry.peek().length}`,
    );
  });

  await section("the same error 500 times is one record counting", async () => {
    const page = await fakePage();
    for (let i = 0; i < 500; i += 1) {
      page.emit("error", {
        message: "Uncaught TypeError: per-frame",
        filename: "ar.js",
        lineno: 1,
        colno: 1,
        error: null,
      });
    }
    const records = page.telemetry.peek();
    return expect(
      records.length === 1 && records[0].n === 500 && records[0].last > 0,
      `${records.length} records, n=${records[0] && records[0].n}`,
    );
  });

  await section(
    "a full buffer keeps the FIRST records as well as the last",
    async () => {
      const page = await fakePage();
      // Every record distinct, so none of them coalesce: 1000 into a buffer
      // of 256.
      for (let i = 0; i < 1000; i += 1) {
        page.telemetry.error("page", `distinct ${i}`);
      }
      const records = page.telemetry.peek();
      const first = records[0].msg;
      const head = records.slice(0, 32).map((entry) => entry.msg);
      const last = records[records.length - 1].msg;
      const stats = page.telemetry.stats();
      return expect(
        records.length === 256 &&
          first === "distinct 0" &&
          head[31] === "distinct 31" &&
          last === "distinct 999" &&
          stats.dropped === 744,
        `${records.length} records, ${first} .. ${head[31]} .. ${last}, dropped ${stats.dropped}`,
      );
    },
  );

  await section("nothing that could carry the token is logged", async () => {
    const token = "s3cr3t-pairing-token";
    const page = await fakePage({ hash: `#k=${token}` });
    page.telemetry.error(
      "page",
      // Three ways the token gets into a message: in the page's own URL, as
      // a query parameter, and bare in some library's error text.
      `failed for https://node.invalid/page.html#k=${token}, ?k=${token}, ` +
        `and the token is ${token}`,
      {
        stack: `at fetch (https://node.invalid/offer?token=${token}:1:1)`,
        data: {
          body: `{"sdp":"x","token":"${token}"}`,
          // A token this page does not know the value of: the shape alone
          // has to be enough.
          where: "#k=someone-elses-token",
        },
      },
    );
    const [entry] = page.telemetry.peek();
    const text = JSON.stringify(entry);
    return expect(
      !text.includes(token) &&
        !text.includes("someone-elses-token") &&
        text.includes("[token]") &&
        text.includes("[redacted]"),
      text.slice(0, 260),
    );
  });

  await section(
    "the control channel is preferred, and carries the envelope",
    async () => {
      const page = await fakePage();
      const sent = [];
      page.telemetry.error("page", "before the channel");
      page.telemetry.attach((text) => {
        sent.push(JSON.parse(text));
        return true;
      });
      await sleep(30);
      const envelope = sent[0];
      return expect(
        sent.length === 1 &&
          envelope.type === "log" &&
          envelope.v === 1 &&
          envelope.page === "1" &&
          envelope.reason === "attach" &&
          envelope.records.some((r) => r.msg === "before the channel") &&
          page.posts.length === 0 &&
          page.telemetry.stats().buffered === 0,
        `${sent.length} envelopes, ${envelope.records.length} records, ${page.posts.length} posts`,
      );
    },
  );

  await section(
    "with no channel it posts to the node over HTTPS, with the token",
    async () => {
      const page = await fakePage({
        hash: "#k=abc123",
        config: { limits: { flushMs: 20, errorFlushMs: 20 } },
      });
      page.telemetry.error("page", "no webrtc at all");
      await sleep(120);
      const post = page.posts[0];
      return expect(
        page.posts.length === 1 &&
          post.url === "https://node.invalid/log" &&
          !post.url.includes("k=") &&
          post.body.token === "abc123" &&
          post.body.records[0].msg === "no webrtc at all" &&
          post.options.headers["Content-Type"] === "application/json" &&
          page.telemetry.stats().buffered === 0,
        `${post.url} token=${post.body.token}`,
      );
    },
  );

  await section("a sink that refuses keeps the records", async () => {
    const page = await fakePage({
      config: {
        limits: {
          flushMs: 20,
          errorFlushMs: 20,
          backoffMs: 20,
          maxBackoffMs: 40,
        },
      },
      // 405 is what an unpatched node answers: the static mount has no
      // /log route, so this is also the real first-run case.
      respond: (n) => (n < 3 ? { ok: false, status: 405 } : { ok: true }),
    });
    page.telemetry.error("page", "the node has no sink yet");
    await sleep(60);
    const heldBack = page.telemetry.stats().buffered;
    await sleep(300);
    const stats = page.telemetry.stats();
    return expect(
      heldBack === 1 &&
        page.posts.length >= 3 &&
        stats.buffered === 0 &&
        stats.counters.refused === 2 &&
        stats.counters.https === 1,
      `held ${heldBack}, ${page.posts.length} posts, ${JSON.stringify(stats.counters)}`,
    );
  });

  await section("a page going away beacons what is left", async () => {
    const page = await fakePage({ respond: () => new Error("no route") });
    page.telemetry.error("page", "the last thing it knew");
    page.emit("pagehide", {});
    const beacon = JSON.parse(page.beacons[0].body);
    return expect(
      page.beacons.length === 1 &&
        page.beacons[0].url === "https://node.invalid/log" &&
        beacon.reason === "pagehide" &&
        beacon.records[0].msg === "the last thing it knew" &&
        page.telemetry.stats().buffered === 0,
      `${page.beacons.length} beacons, reason ${beacon.reason}`,
    );
  });

  await section("going hidden beacons too, and says so", async () => {
    const page = await fakePage();
    page.telemetry.error("page", "before the headset came off");
    page.setVisibility("hidden");
    const beacon = JSON.parse(page.beacons[0].body);
    return expect(
      page.beacons.length === 1 &&
        beacon.reason === "visibility" &&
        beacon.records.some((r) => r.msg === "page hidden"),
      `${page.beacons.length} beacons: ${beacon.records.map((r) => r.msg).join(" | ")}`,
    );
  });

  await section(
    "a beacon the browser will not take is carried to the next page load",
    async () => {
      const storage = new Map();
      const first = await fakePage({ storage, beacon: () => false });
      first.telemetry.error("page", "the reason the page reloaded");
      first.emit("pagehide", {});
      const stashed = storage.get("dora-openarm-webxr-telemetry");
      // The page comes back: same tab, same sessionStorage.
      const second = await fakePage({
        storage,
        config: { limits: { flushMs: 20 } },
      });
      const sent = [];
      second.telemetry.attach((text) => {
        sent.push(JSON.parse(text));
        return true;
      });
      await sleep(60);
      const carried = sent.find((envelope) =>
        envelope.records.some((r) => r.msg === "the reason the page reloaded"),
      );
      return expect(
        Boolean(stashed) &&
          Boolean(carried) &&
          carried.session === first.telemetry.stats().session &&
          carried.session !== second.telemetry.stats().session &&
          !storage.has("dora-openarm-webxr-telemetry"),
        `carried ${carried && carried.records.length} records under the old session id`,
      );
    },
  );

  await section("a session end flushes without waiting", async () => {
    const page = await fakePage({ config: { limits: { flushMs: 60000 } } });
    const sent = [];
    page.telemetry.attach((text) => {
      sent.push(JSON.parse(text));
      return true;
    });
    page.telemetry.info("page", "mid session");
    page.telemetry.sessionEnded("taken over");
    await sleep(400);
    const records = sent.flatMap((envelope) => envelope.records);
    return expect(
      records.some((r) => r.msg.includes("xr session ended: taken over")) &&
        records.some((r) => r.msg === "mid session"),
      records.map((r) => r.msg).join(" | "),
    );
  });

  await section(
    "the buffer survives a session end and a reconnect",
    async () => {
      const page = await fakePage({
        config: { limits: { flushMs: 60000, errorFlushMs: 60000 } },
        respond: () => new Error("the node is gone"),
      });
      const sent = [];
      let open = true;
      page.telemetry.attach((text) => (open ? (sent.push(text), true) : false));
      open = false;
      page.telemetry.detach("the node went away");
      page.telemetry.error("page", "while there was no node");
      page.telemetry.sessionEnded("the operator took the headset off");
      const kept = page.telemetry.peek().map((entry) => entry.msg);
      // A new connection, as ar.js makes after every session.
      open = true;
      page.telemetry.attach((text) => (open ? (sent.push(text), true) : false));
      await sleep(50);
      const delivered = sent
        .map((text) => JSON.parse(text))
        .flatMap((envelope) => envelope.records.map((r) => r.msg));
      return expect(
        kept.some((msg) => msg === "while there was no node") &&
          delivered.some((msg) => msg === "while there was no node") &&
          delivered.some((msg) => msg.includes("xr session ended")),
        `kept ${kept.length}, delivered ${delivered.length}`,
      );
    },
  );

  await section("the minute report is percentiles, not a mean", async () => {
    const page = await fakePage();
    // 72 Hz with one 200 ms stall: the shape a mean hides.
    let t = 1000;
    for (let i = 0; i < 400; i += 1) {
      t += i === 200 ? 200 : 13.9;
      page.telemetry.frame(t);
      page.telemetry.pose(t);
    }
    page.telemetry.report();
    const entry = page.telemetry.peek().find((r) => r.event === "perf");
    return expect(
      entry &&
        entry.data.frames === 400 &&
        entry.data.frame_p50 === 13.9 &&
        entry.data.frame_max === 200 &&
        entry.data.poses === 400 &&
        entry.msg.includes("p50 13.9"),
      entry ? entry.msg : "no perf record",
    );
  });

  await section("tracking records the edges, not every frame", async () => {
    const page = await fakePage();
    page.telemetry.tracking("right", true); // the first frame: not news
    for (let i = 0; i < 100; i += 1) {
      page.telemetry.tracking("right", true);
    }
    for (let i = 0; i < 100; i += 1) {
      page.telemetry.tracking("right", false);
    }
    page.telemetry.tracking("right", true);
    const records = page.telemetry.peek();
    return expect(
      records.length === 2 &&
        records[0].level === "warn" &&
        records[0].msg.includes("right untracked") &&
        records[1].msg.includes("right tracked again"),
      records.map((r) => `${r.level}:${r.msg}`).join(" | "),
    );
  });

  // A context the page drops on purpose is housekeeping; one it loses is the
  // failure that turns the headset black. Without the distinction, EVERY
  // clean `--view scene` session end leaves an error in the file, `oqt run`
  // reports one after every good run, and the level stops meaning anything.
  await section(
    "a context the page releases itself is not an error",
    async () => {
      const page = await fakePage();
      const canvases = [];
      const makeCanvas = () => {
        const handlers = new Map();
        const canvas = {
          addEventListener(type, handler) {
            if (!handlers.has(type)) {
              handlers.set(type, []);
            }
            handlers.get(type).push(handler);
          },
          lose(event = {}) {
            for (const handler of handlers.get("webglcontextlost") || []) {
              handler(event);
            }
          },
        };
        canvases.push(canvas);
        return canvas;
      };

      // One the page really lost: still an error, still flushed at once.
      const lost = makeCanvas();
      page.telemetry.context(lost, "scene3d");
      lost.lose({ statusMessage: "" });

      // One the view is closing, as scene3d.js `close()` does.
      const released = makeCanvas();
      page.telemetry.context(released, "scene3d");
      page.telemetry.contextClosing(released);
      released.lose({ statusMessage: "" });

      const records = page.telemetry.peek();
      const errors = records.filter((r) => r.level === "error");
      return expect(
        records.length === 2 &&
          errors.length === 1 &&
          errors[0].msg.includes("WebGL context lost") &&
          records[1].level === "info" &&
          records[1].msg.includes("WebGL context released"),
        records.map((r) => `${r.level}:${r.msg}`).join(" | "),
      );
    },
  );

  await section(
    "an envelope too big to send is split, not dropped",
    async () => {
      const page = await fakePage({
        config: { limits: { bytes: 900, flushMs: 10, errorFlushMs: 10 } },
      });
      for (let i = 0; i < 40; i += 1) {
        page.telemetry.error("page", `a message with some length to it ${i}`);
      }
      const sent = [];
      page.telemetry.attach((text) => {
        sent.push(text);
        return true;
      });
      await sleep(200);
      const sizes = sent.map((text) => text.length);
      const envelopes = sent.map((text) => JSON.parse(text));
      const carried = envelopes.map((envelope) => envelope.records.length);
      // Every one of the 40 arrives, each envelope inside the limit, and the
      // envelope numbers run 1, 2, 3... with no gap: a split must not look to
      // the node like an envelope that was lost.
      const arrived = new Set(
        envelopes.flatMap((envelope) =>
          envelope.records
            .filter((entry) => entry.msg.startsWith("a message"))
            .map((entry) => entry.record),
        ),
      );
      return expect(
        sent.length > 1 &&
          sizes.every((size) => size <= 900) &&
          arrived.size === 40 &&
          envelopes.map((e) => e.page).join(",") ===
            envelopes.map((_, i) => i + 1).join(","),
        `${sent.length} envelopes of ${sizes.join("/")} bytes carrying ${carried.join("+")} records, ${arrived.size} of the 40 messages`,
      );
    },
  );

  await section("a disabled page has no telemetry at all", async () => {
    const page = await fakePage({ config: { enabled: false } });
    console.error("nobody is listening");
    return expect(
      globalThis.__oqtTelemetry === undefined &&
        page.telemetry.peek().length === 0 &&
        page.printed.length === 1,
      `global=${globalThis.__oqtTelemetry} printed=${page.printed.length}`,
    );
  });

  await section("recording never throws, whatever it is given", async () => {
    const page = await fakePage();
    const circular = {};
    circular.self = circular;
    page.telemetry.error("page", { toString: () => "odd" }, { data: circular });
    page.telemetry.error("page", "ok", {
      data: { n: Number.NaN, u: undefined },
    });
    page.emit("error", {});
    page.emit("unhandledrejection", {});
    const records = page.telemetry.peek();
    return expect(
      records.length >= 3 && JSON.stringify(records).length > 0,
      `${records.length} records survived`,
    );
  });

  await section("what one record costs the page", async () => {
    const page = await fakePage({ config: { limits: { flushMs: 60000 } } });
    const rounds = 20000;
    // A distinct message each time: the worst case, where nothing coalesces
    // and the buffer is full and evicting.
    let start = process.hrtime.bigint();
    for (let i = 0; i < rounds; i += 1) {
      page.telemetry.error("page", `distinct message ${i}`);
    }
    const perRecord = Number(process.hrtime.bigint() - start) / rounds / 1000;
    // The repeat path, which is what a per-frame error actually takes.
    start = process.hrtime.bigint();
    for (let i = 0; i < rounds; i += 1) {
      page.telemetry.error("page", "the same message");
    }
    const perRepeat = Number(process.hrtime.bigint() - start) / rounds / 1000;
    // The per-frame hooks, which run on the XR animation loop.
    start = process.hrtime.bigint();
    for (let i = 0; i < rounds; i += 1) {
      page.telemetry.frame(i * 13.9);
      page.telemetry.pose(i * 13.9);
      page.telemetry.tracking("right", true);
      page.telemetry.tracking("left", true);
      page.telemetry.tracking("head", true);
    }
    const perFrame = Number(process.hrtime.bigint() - start) / rounds / 1000;
    // One envelope of 64 records, built and encoded.
    start = process.hrtime.bigint();
    const batches = 200;
    for (let i = 0; i < batches; i += 1) {
      page.telemetry.flush("timer");
    }
    const perBatch = Number(process.hrtime.bigint() - start) / batches / 1000;
    return expect(
      perFrame < 20 && perRecord < 200,
      `record ${perRecord.toFixed(2)} us, repeat ${perRepeat.toFixed(2)} us, ` +
        `frame hooks ${perFrame.toFixed(2)} us, batch ${perBatch.toFixed(1)} us`,
    );
  });

  const failed = results.filter((result) => !result.ok);
  say(`\n${results.length - failed.length} passed, ${failed.length} failed`);
  process.exit(failed.length === 0 ? 0 : 1);
}

main();
