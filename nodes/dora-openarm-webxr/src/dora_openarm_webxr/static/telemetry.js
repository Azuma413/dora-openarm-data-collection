// Copyright 2026 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The page's own log, sent back to the node (openarm-quest-teleop patch 17).
//
// Debugging a Quest through developer mode means a cable, `adb` and
// chrome://inspect, at which point the failure is an hour old. Everything
// this page knows about a failure is recorded here and delivered to the node
// by itself, so `runs/<name>/quest.log` has it before anyone asks.
//
// This module is loaded by `index.html` **before** `ar.js`, as its own
// `<script type="module">`, and publishes itself as `window.__oqtTelemetry`.
// It is not imported: `ar.js`, `connection.js` and `scene3d.js` read the
// global and call it through `?.`, so a page whose telemetry module failed to
// fetch, or the Node.js stand-ins the unit tests run `ar.js` against, work
// exactly as before. Telemetry may never be the reason teleoperation stops.
//
// ## The design points, and why each is what it is
//
// **Hooking console.error/warn must not swallow them or recurse.** The
// original function is called FIRST and unconditionally, before anything here
// runs, so an exception in this module cannot lose a line the operator would
// otherwise have seen in a remote debugger. A re-entry flag makes a
// console.error raised *inside* the hook (or by anything the hook calls) pass
// straight through. Nothing on the delivery path ever calls console.*: a sink
// that is refusing envelopes would otherwise log about it, which would make a
// record, which would schedule a flush.
//
// **The buffer keeps the FIRST records as well as the last.** A crash
// cascade -- one broken frame, then the same error 72 times a second -- must
// not push out the cause. `buffer` holds at most HEAD + TAIL records; when it
// is full the record evicted is the oldest one *after* the first HEAD, so the
// first HEAD records of the session survive anything that follows them. It is
// a module-level array, tied to neither the XR session nor the connection, so
// it survives a session end and every reconnect; `pagehide` stashes whatever
// is still undelivered in sessionStorage, and the next page load sends it.
//
// **Sending never runs on the XR animation loop.** `record()` only formats
// and buffers; every delivery happens from a `setTimeout` (5 s, or 250 ms
// after an error so a cascade coalesces into one envelope), from the control
// channel opening, or from `pagehide`. Window timers keep running while an
// immersive session presents, which `requestAnimationFrame` on the window
// does not.
//
// **Duplicates are counted, not repeated.** Records with the same level,
// source, message and top stack frame coalesce into one record with `n` and
// `last`, per envelope. A per-frame error becomes one record saying `n: 360`,
// not 360 messages a second.
//
// **The per-frame hooks are three compares and a store.** `frame()` and
// `pose()` write one interval into a fixed Float64Array ring and bump a
// counter -- no allocation, no `performance.now()` (the XR frame time is on
// the same clock and is already in hand), no percentile work until the
// once-a-minute report. `tracking()` returns on the first compare unless the
// state actually changed. Measured in the harness: the page's XR frame time
// does not move (tests/web/telemetry.test.mjs).
//
// **Nothing that could carry the pairing token is logged.** `location.hash`
// is never read into a record, the exact token is replaced with `[token]`
// wherever it appears, and anything shaped like `#k=`, `?k=`, `token=` is
// replaced with `[redacted]` -- in the message, in the stack and in every
// string of `data`. The one record that names where the page came from names
// the origin and the path, never the fragment.
//
// **Transport, which is the point of the feature.** The case worth debugging
// is usually the one where WebRTC is broken, so the reliable `control`
// channel is only the *first* choice. With no channel the envelope goes as an
// ordinary `POST log` to the node's own HTTPS server, same origin as
// `POST offer` and carrying the same pairing token in the body, which works
// with no WebRTC at all; on `pagehide` and on going hidden it goes by
// `navigator.sendBeacon`, which outlives the page -- and which cannot set a
// header, hence the token in the body. A sink that refuses is backed off
// from, 1 s doubling to 30 s, and the records stay buffered: a node with no
// `--quest-log` answers 503 and must not cost the page its log.
//
// The wire contract (envelope and record shape, the routes, what the node
// should expect) is in PATCHES.md 17.

// The build this page was served from. It must equal the version in
// pyproject.toml: it is how a headset showing a cached page from an older run
// gives itself away, since the node knows its own version and this is the
// page's. tests/web/telemetry_units.test.mjs pins the two together.
//
// The node knows more than this -- it hashes the files it serves and pushes
// `build` in its configuration message -- so `nodeBuild()` below compares the
// two and says so in the log when they disagree. The page cannot hash itself
// cheaply, so a cached page of the *same* version is the node's to notice.
const BUILD = "0.6.0";

const VERSION = 1;

// Where the node takes the log: `{"type": "log", ...}` on the reliable
// channel, and `POST log` on its own HTTPS server (main.py `_log_endpoint`,
// webrtc.py `_handle_control_message`). Both names are the node's.
const MESSAGE_TYPE = "log";
const ROUTE = "log";

const LIMITS = {
  // Never evicted: the first records of the session, which hold the cause of
  // whatever cascade fills the rest.
  head: 32,
  // The most recent records, oldest evicted first.
  tail: 224,
  // Records per envelope, and bytes per envelope (sendBeacon's own limit is
  // 64 KiB, and a batch that does not fit is split rather than dropped).
  batch: 64,
  bytes: 60000,
  // How long a quiet page waits before delivering, and how long an error
  // waits so that the cascade behind it coalesces into the same envelope.
  flushMs: 5000,
  errorFlushMs: 250,
  // The report of what the page's own frames cost.
  reportMs: 60000,
  // Interval samples kept per series: a minute at 72 Hz is 4320.
  samples: 8192,
  // Distinct messages tracked for coalescing within one envelope.
  keys: 512,
  // HTTPS backoff while the sink refuses.
  backoffMs: 1000,
  maxBackoffMs: 30000,
  msg: 512,
  // The node truncates any string past 2048 bytes, so there is no point
  // spending the headset's link on more of a stack than it will keep.
  stack: 2000,
};

// Where connection.js keeps the pairing token for this tab. Read here, never
// written: connection.js owns the pairing contract (patch 8), this module
// only needs the token to authenticate its POST, and reads the fragment
// itself so that a failure before connection.js has run still posts.
const PAIRING_TOKEN_KEY = "dora-openarm-webxr-pairing-token";

// Where an envelope the page could not deliver before it went away waits for
// the next page load, and how many are kept.
const CARRIED_KEY = "dora-openarm-webxr-telemetry";
const CARRIED = 2;

const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };

function hex(bytes) {
  const random = globalThis.crypto;
  if (random && typeof random.getRandomValues === "function") {
    const values = new Uint8Array(bytes);
    random.getRandomValues(values);
    return Array.from(values, (value) =>
      value.toString(16).padStart(2, "0"),
    ).join("");
  }
  let out = "";
  for (let index = 0; index < bytes; index += 1) {
    out += Math.floor(Math.random() * 256)
      .toString(16)
      .padStart(2, "0");
  }
  return out;
}

// Percentiles of the first `count` entries of a ring, on a copy, so the ring
// itself is never reordered under the frame loop.
function quantiles(ring, count) {
  if (count === 0) {
    return null;
  }
  const sorted = Array.prototype.slice
    .call(ring.subarray(0, count))
    .sort((a, b) => a - b);
  const at = (quantile) =>
    Math.round(sorted[Math.min(count - 1, Math.floor(quantile * count))] * 10) /
    10;
  return {
    p50: at(0.5),
    p95: at(0.95),
    p99: at(0.99),
    max: Math.round(sorted[count - 1] * 10) / 10,
  };
}

// One value of a console argument, small enough to log and never throwing.
function describe(value) {
  if (typeof value === "string") {
    return value;
  }
  if (value instanceof Error) {
    return `${value.name}: ${value.message}`;
  }
  if (value === null || value === undefined || typeof value !== "object") {
    return String(value);
  }
  try {
    const text = JSON.stringify(value);
    return text === undefined ? String(value) : text.slice(0, 200);
  } catch {
    try {
      return String(value);
    } catch {
      return "[unprintable]";
    }
  }
}

export function createTelemetry(config = {}) {
  const enabled = config.enabled !== false;
  const limits = { ...LIMITS, ...(config.limits || {}) };
  const sessionId = config.sessionId || hex(8);

  // Everything the page has recorded and not yet delivered. A plain array,
  // because the eviction rule is "the oldest that is not one of the first
  // HEAD", which a ring cannot express and a splice can.
  const buffer = [];
  // Records that went out in the envelope currently in flight over HTTPS, so
  // a refusal can put them back rather than lose them.
  let inflight = null;
  // An envelope stashed by a previous page load's pagehide, resent as it was.
  const carried = [];
  let dropped = 0;
  let seq = 0;
  let pageSeq = 0;
  // Same level, source, message and top frame within this envelope: counted,
  // not repeated. Cleared at every flush.
  const keys = new Map();

  let sender = null;
  let installed = false;
  let flushTimer = null;
  let flushAt = Infinity;
  let reportTimer = null;
  let backoff = limits.backoffMs;
  let posting = false;
  let token = null;
  // What the node last said it is, so every envelope carries it and the node
  // can see which of its builds the page actually came from.
  let nodeBuildString = null;
  const counters = { control: 0, https: 0, beacon: 0, refused: 0, failed: 0 };
  // Canvases whose context the page is dropping on purpose (see
  // `contextClosing`). Weak, so marking one never keeps a dead view alive.
  const releasing = new WeakSet();

  const frameIntervals = new Float64Array(limits.samples);
  const poseIntervals = new Float64Array(limits.samples);
  const rates = {
    frames: 0,
    frameAt: 0,
    lastFrame: null,
    poses: 0,
    poseAt: 0,
    lastPose: null,
    since: now(),
  };
  // The XR frame time of the newest frame, stamped on records made inside
  // one. Compared against the page clock at record time rather than tracked
  // per frame, so a frame costs nothing extra for it.
  let xrTime = null;
  const tracked = Object.create(null);

  // The original console, captured before anything is replaced, and the only
  // console this module ever calls (it never does, except through these).
  const console0 = {};
  let insideConsole = false;

  let verbose = Boolean(config.verbose);

  function now() {
    return typeof performance !== "undefined" && performance.now
      ? performance.now()
      : Date.now();
  }

  // --- the pairing token, for redaction and for the POST -------------------

  function readToken() {
    let found = null;
    try {
      const hash = globalThis.location ? globalThis.location.hash || "" : "";
      found = new URLSearchParams(hash.replace(/^#/, "")).get("k");
    } catch {
      found = null;
    }
    if (!found) {
      try {
        found = globalThis.sessionStorage
          ? globalThis.sessionStorage.getItem(PAIRING_TOKEN_KEY)
          : null;
      } catch {
        found = null;
      }
    }
    if (found) {
      token = found;
    }
    return token;
  }

  // Nothing that could carry the pairing token may leave this page in a
  // record: the exact token wherever it appears, and anything shaped like a
  // token in a URL or a JSON body, whether or not this page knows the value.
  function redact(text) {
    let out = String(text);
    if (token) {
      out = out.split(token).join("[token]");
    }
    out = out.replace(
      /([#?&](?:k|token|pairing[-_]?token|key)=)[^&\s"'\\]*/gi,
      "$1[redacted]",
    );
    out = out.replace(
      /("(?:token|k|pairing_token)"\s*:\s*")[^"]*"/gi,
      '$1[redacted]"',
    );
    return out;
  }

  function clean(text, cap) {
    const out = redact(text);
    return out.length > cap
      ? `${out.slice(0, cap)}…(+${out.length - cap})`
      : out;
  }

  // Flat, small, JSON-safe, and redacted like everything else.
  function cleanData(data) {
    if (!data || typeof data !== "object") {
      return undefined;
    }
    const out = {};
    let count = 0;
    for (const key of Object.keys(data)) {
      if (count >= 16) {
        break;
      }
      const value = data[key];
      if (value === null || value === undefined) {
        continue;
      }
      if (typeof value === "number") {
        out[key] = Number.isFinite(value) ? value : String(value);
      } else if (typeof value === "boolean") {
        out[key] = value;
      } else if (Array.isArray(value)) {
        out[key] = clean(value.map(describe).join(","), 200);
      } else {
        out[key] = clean(describe(value), 200);
      }
      count += 1;
    }
    return count > 0 ? out : undefined;
  }

  // --- the buffer ----------------------------------------------------------

  function push(record) {
    while (buffer.length >= limits.head + limits.tail) {
      // The oldest record that is not one of the first HEAD: the cause of a
      // cascade is never pushed out by the cascade.
      buffer.splice(limits.head, 1);
      dropped += 1;
    }
    buffer.push(record);
  }

  function record(level, source, message, extra = {}) {
    if (!enabled) {
      return null;
    }
    try {
      if (LEVELS[level] === undefined) {
        level = "info";
      }
      if (level === "debug" && !verbose) {
        return null;
      }
      // An Error handed in as the message reads as "TypeError: ..." rather
      // than "[object Object]", and brings its stack if none was given.
      const msg = clean(
        typeof message === "string" ? message : describe(message),
        limits.msg,
      );
      const inherited =
        !extra.stack && message instanceof Error ? message.stack : extra.stack;
      const stack = inherited ? clean(inherited, limits.stack) : undefined;
      const at = now();
      // Same thing again: count it instead of sending it again. The key is
      // built before the record, so a per-frame error costs one map lookup.
      const key = `${level}\u0000${source}\u0000${msg}\u0000${
        stack ? stack.slice(0, 120) : ""
      }`;
      const seen = keys.get(key);
      if (seen) {
        seen.n = (seen.n || 1) + 1;
        seen.last = Math.round(at * 10) / 10;
        if (level === "error" || level === "warn") {
          schedule(limits.errorFlushMs);
        }
        return seen;
      }
      seq += 1;
      const entry = {
        // The node writes each record as a line of its own and adds
        // `received`, `session`, `peer`, `via` and its own `seq`, keeping its
        // fields when a record's name collides (main.py `_line`). So the
        // page's own counter is `record` -- a gap in it is a record the
        // page's buffer had to drop -- and what would have been `source` is
        // `event`, which is the name the node's line gives that column.
        record: seq,
        t: Date.now(),
        mono: Math.round(at * 10) / 10,
        level,
        event: source,
        msg,
      };
      // The XR frame time, when this record was made inside a frame. The two
      // clocks are the same one, so a frame that is still current is a
      // comparison rather than a per-frame timestamp.
      if (xrTime !== null && at - xrTime < 100) {
        entry.xr = Math.round(xrTime * 10) / 10;
      }
      if (stack) {
        entry.stack = stack;
      }
      const data = cleanData(extra.data);
      if (data) {
        entry.data = data;
      }
      push(entry);
      if (keys.size < limits.keys) {
        keys.set(key, entry);
      }
      schedule(
        level === "error" || level === "warn"
          ? limits.errorFlushMs
          : limits.flushMs,
      );
      return entry;
    } catch {
      // A telemetry module that throws is worse than one that misses a line.
      return null;
    }
  }

  // --- delivery ------------------------------------------------------------

  function schedule(delay) {
    if (!enabled) {
      return;
    }
    const at = now() + delay;
    if (flushTimer !== null && flushAt <= at) {
      return;
    }
    if (flushTimer !== null) {
      clearTimeout(flushTimer);
    }
    flushAt = at;
    flushTimer = setTimeout(() => {
      flushTimer = null;
      flushAt = Infinity;
      flush("timer");
    }, delay);
    // Never hold a Node.js process open for a flush (the unit tests).
    if (flushTimer && typeof flushTimer.unref === "function") {
      flushTimer.unref();
    }
  }

  function envelope(records, reason, droppedCount) {
    pageSeq += 1;
    return {
      type: MESSAGE_TYPE,
      v: VERSION,
      session: sessionId,
      build: BUILD,
      ...(nodeBuildString ? { node: nodeBuildString } : {}),
      // A string, because that is what the node keeps on each line it writes
      // (main.py `accept`); a gap in it means an envelope never arrived.
      page: String(pageSeq),
      dropped: droppedCount,
      reason,
      records,
    };
  }

  // Take the oldest records, splitting the batch until its JSON fits.
  function take(reason) {
    const records = buffer.splice(0, limits.batch);
    if (records.length === 0) {
      return null;
    }
    const droppedCount = dropped;
    dropped = 0;
    let body = envelope(records, reason, droppedCount);
    let text = stringify(body);
    while (text.length > limits.bytes && body.records.length > 1) {
      const half = Math.max(1, Math.floor(body.records.length / 2));
      const keep = body.records.slice(0, half);
      buffer.unshift(...body.records.slice(half));
      // The split envelope keeps this envelope's number rather than taking a
      // new one; the remainder goes out as the next one.
      pageSeq -= 1;
      body = envelope(keep, reason, droppedCount);
      text = stringify(body);
    }
    return { body, text, records: body.records, dropped: droppedCount };
  }

  function stringify(value) {
    try {
      return JSON.stringify(value);
    } catch {
      return JSON.stringify({
        type: MESSAGE_TYPE,
        v: VERSION,
        session: sessionId,
        build: BUILD,
        page: pageSeq,
        dropped: 0,
        reason: "unencodable",
        records: [],
      });
    }
  }

  // Put an undelivered batch back at the front, oldest first, and let the
  // eviction rule decide what has to go if it no longer fits.
  function restore(batch, droppedCount) {
    buffer.unshift(...batch);
    dropped += droppedCount;
    while (buffer.length > limits.head + limits.tail) {
      buffer.splice(limits.head, 1);
      dropped += 1;
    }
  }

  function postUrl() {
    try {
      // Resolved against the page, like connection.js resolves "offer", and
      // resolving drops the fragment -- the pairing token never appears in a
      // URL this module builds.
      return new URL(ROUTE, globalThis.location.href).toString();
    } catch {
      return ROUTE;
    }
  }

  function flush(reason = "timer", { beacon = false } = {}) {
    if (!enabled) {
      return false;
    }
    try {
      // Whatever a previous page load could not deliver goes first: it is the
      // older news, and it is usually why there was a new page load. It keeps
      // its own session id, so the node still groups it with the run it
      // belongs to.
      if (carried.length > 0 && !posting) {
        const text = carried[0];
        const outcome = deliver(text, beacon, () => {
          carried.unshift(text);
          saveCarried();
        });
        if (outcome !== "no") {
          carried.shift();
          saveCarried();
        }
      }
      if (posting && !beacon) {
        return false;
      }
      if (buffer.length === 0) {
        return true;
      }
      keys.clear();
      const batch = take(reason);
      if (!batch) {
        return true;
      }
      const outcome = deliver(batch.text, beacon, () =>
        restoreText(batch.text),
      );
      if (outcome === "no") {
        // Nothing took it, so it goes back where it was: a sink that is down
        // is exactly when the records matter most.
        restore(batch.records, batch.dropped);
        return false;
      }
      // A backlog bigger than one envelope -- a page that has been offline,
      // or records too big to travel together -- goes on emptying rather
      // than waiting for the next record to schedule a flush.
      if (outcome === "sent" && (buffer.length > 0 || carried.length > 0)) {
        schedule(0);
      }
      return true;
    } catch {
      return false;
    }
  }

  // One envelope, by whichever transport can take it.
  //
  //   "sent"     the page may forget the records: the reliable channel took
  //              them, or the browser took the beacon.
  //   "inflight" an HTTPS POST is on its way; `onFail` puts them back if the
  //              node refuses it.
  //   "no"       nothing could take it.
  function deliver(text, beacon, onFail) {
    if (!beacon && sender) {
      try {
        if (sender(text) === true) {
          counters.control += 1;
          return "sent";
        }
      } catch {
        // A channel that throws is a channel that is closing.
      }
    }
    const url = postUrl();
    let body = text;
    const authenticated = readToken();
    if (authenticated) {
      try {
        body = stringify({ ...JSON.parse(text), token: authenticated });
      } catch {
        body = text;
      }
    }
    if (beacon) {
      return sendBeacon(url, body) ? "sent" : "no";
    }
    return post(url, body, onFail) ? "inflight" : "no";
  }

  function sendBeacon(url, body) {
    try {
      const navigator0 = globalThis.navigator;
      if (!navigator0 || typeof navigator0.sendBeacon !== "function") {
        return false;
      }
      const blob =
        typeof Blob === "function"
          ? new Blob([body], { type: "application/json" })
          : body;
      // true means the browser has taken responsibility for sending it, which
      // is as close to delivered as an unloading page can get.
      if (navigator0.sendBeacon(url, blob) === true) {
        counters.beacon += 1;
        return true;
      }
    } catch {
      // A beacon that cannot be queued is one more reason to keep the records.
    }
    return false;
  }

  // Never awaited by anything on the page: the answer only decides whether
  // the records are forgotten or put back.
  function post(url, body, onFail) {
    const fetch0 = globalThis.fetch;
    if (typeof fetch0 !== "function") {
      return false;
    }
    posting = true;
    inflight = body;
    let request;
    try {
      request = fetch0(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        // Telemetry is not worth a credential or a cache entry.
        credentials: "omit",
        cache: "no-store",
        keepalive: true,
      });
    } catch {
      posting = false;
      inflight = null;
      counters.failed += 1;
      backOff();
      return false;
    }
    const settled = (ok, counter) => {
      posting = false;
      inflight = null;
      if (ok) {
        counters.https += 1;
        backoff = limits.backoffMs;
        if (buffer.length > 0 || carried.length > 0) {
          schedule(0);
        }
        return;
      }
      counters[counter] += 1;
      try {
        onFail();
      } catch {
        // Nothing to put back.
      }
      backOff();
    };
    Promise.resolve(request).then(
      (response) => settled(Boolean(response && response.ok), "refused"),
      () => settled(false, "failed"),
    );
    // Not delivered yet; flush() has already taken the records out, and
    // `onFail` puts them back if the node says no.
    return true;
  }

  function restoreText(text) {
    try {
      const body = JSON.parse(text);
      restore(body.records || [], body.dropped || 0);
    } catch {
      // Nothing to put back.
    }
  }

  function backOff() {
    backoff = Math.min(limits.maxBackoffMs, backoff * 2);
    schedule(backoff);
  }

  function saveCarried() {
    try {
      const storage = globalThis.sessionStorage;
      if (!storage) {
        return;
      }
      while (carried.length > CARRIED) {
        // The newest goes: the oldest envelope holds the first errors, and
        // those are the ones a cascade would otherwise bury.
        carried.pop();
      }
      if (carried.length === 0) {
        storage.removeItem(CARRIED_KEY);
      } else {
        storage.setItem(CARRIED_KEY, JSON.stringify(carried));
      }
    } catch {
      // Storage can be refused (a private window, blocked site data); the
      // records are simply not carried over a reload.
    }
  }

  // Whatever a previous page load could not deliver before it went away.
  function loadCarried() {
    try {
      const storage = globalThis.sessionStorage;
      const text = storage ? storage.getItem(CARRIED_KEY) : null;
      if (!text) {
        return;
      }
      const stored = JSON.parse(text);
      if (Array.isArray(stored)) {
        for (const entry of stored.slice(0, CARRIED)) {
          if (typeof entry === "string") {
            carried.push(entry);
          }
        }
      }
    } catch {
      // Nothing carried over.
    }
  }

  // The last chance: whatever is still here when the page goes away is
  // beaconed, and what the beacon would not take is stashed for the next load
  // -- a headset put down on a node that is not answering still gives its log
  // up when the page comes back.
  function onHide(reason) {
    flush(reason, { beacon: true });
    while (buffer.length > 0) {
      keys.clear();
      const batch = take(reason);
      if (!batch) {
        break;
      }
      carried.push(batch.text);
      if (carried.length >= CARRIED) {
        break;
      }
    }
    saveCarried();
  }

  // --- the hooks -----------------------------------------------------------

  function hookConsole() {
    const target = globalThis.console;
    if (!target) {
      return;
    }
    for (const level of ["error", "warn", "log"]) {
      if (typeof target[level] !== "function") {
        continue;
      }
      const original = target[level];
      console0[level] = original;
      target[level] = function (...args) {
        // First and unconditionally: hooking a console must never swallow it,
        // whatever happens below.
        original.apply(this, args);
        if (insideConsole) {
          return;
        }
        insideConsole = true;
        try {
          const error = args.find((value) => value instanceof Error);
          record(
            level === "log" ? "debug" : level,
            "console",
            args.map(describe).join(" "),
            { stack: error ? error.stack : undefined },
          );
        } catch {
          // Never worth breaking a console call over.
        } finally {
          insideConsole = false;
        }
      };
    }
  }

  function onError(event) {
    try {
      const target = event.target;
      if (target && target !== globalThis && (target.src || target.href)) {
        // A resource that would not load: a script, a texture, the scene.
        record(
          "error",
          "asset",
          `${target.tagName || "resource"} failed to load`,
          {
            data: { url: String(target.src || target.href) },
          },
        );
        return;
      }
      const error = event.error;
      record("error", "window", event.message || describe(error), {
        stack: error && error.stack ? error.stack : undefined,
        data: {
          file: basename(event.filename),
          line: event.lineno,
          column: event.colno,
        },
      });
    } catch {
      // Never throw out of an error handler.
    }
  }

  // What the inline handler in index.html caught before this module existed:
  // an ar.js that would not parse, a module that would not fetch. Nothing
  // else can see those, because they happen before any module runs.
  function drainEarly() {
    try {
      const early = globalThis.__oqtEarly;
      if (!Array.isArray(early)) {
        return;
      }
      for (const entry of early) {
        record("error", entry.url ? "asset" : "window", entry.message, {
          stack: entry.stack,
          data: {
            file: basename(entry.file),
            line: entry.line,
            url: entry.url,
          },
        });
      }
      // Handing over: the inline handlers stop collecting, so nothing is
      // recorded twice and the array cannot grow behind this module's back.
      globalThis.__oqtEarly = null;
    } catch {
      // Never worth failing the install over.
    }
  }

  function onRejection(event) {
    try {
      const reason = event.reason;
      record("error", "promise", `unhandled rejection: ${describe(reason)}`, {
        stack: reason && reason.stack ? reason.stack : undefined,
      });
    } catch {
      // Never throw out of an error handler.
    }
  }

  function basename(url) {
    if (!url) {
      return undefined;
    }
    const text = String(url).split(/[?#]/)[0];
    const parts = text.split("/");
    return parts[parts.length - 1] || text;
  }

  // The once-a-minute line: what the page's own frames and pose sends cost,
  // as percentiles, because the failure shape is a healthy median with a long
  // tail and a mean hides it.
  function report() {
    const elapsed = (now() - rates.since) / 1000;
    if (elapsed <= 0) {
      return;
    }
    // One fewer interval than frames, and never more than the ring holds.
    const frames = quantiles(
      frameIntervals,
      Math.min(Math.max(rates.frames - 1, 0), limits.samples),
    );
    const poses = quantiles(
      poseIntervals,
      Math.min(Math.max(rates.poses - 1, 0), limits.samples),
    );
    if (rates.frames > 0 || rates.poses > 0 || buffer.length > 0) {
      const data = {
        seconds: Math.round(elapsed * 10) / 10,
        frames: rates.frames,
        frame_hz: Math.round((10 * rates.frames) / elapsed) / 10,
        poses: rates.poses,
        pose_hz: Math.round((10 * rates.poses) / elapsed) / 10,
        records: seq,
        dropped_records: dropped,
        sent_control: counters.control,
        sent_https: counters.https,
        sent_beacon: counters.beacon,
        refused: counters.refused + counters.failed,
      };
      if (frames) {
        data.frame_p50 = frames.p50;
        data.frame_p95 = frames.p95;
        data.frame_p99 = frames.p99;
        data.frame_max = frames.max;
      }
      if (poses) {
        data.pose_p50 = poses.p50;
        data.pose_p95 = poses.p95;
        data.pose_p99 = poses.p99;
        data.pose_max = poses.max;
      }
      record(
        "info",
        "perf",
        `frames ${rates.frames} at ${data.frame_hz} Hz` +
          (frames
            ? `, interval p50 ${frames.p50} p95 ${frames.p95} p99 ${frames.p99} max ${frames.max} ms`
            : "") +
          `; poses ${rates.poses} at ${data.pose_hz} Hz` +
          (poses
            ? `, interval p50 ${poses.p50} p95 ${poses.p95} p99 ${poses.p99} max ${poses.max} ms`
            : ""),
        { data },
      );
    }
    rates.frames = 0;
    rates.frameAt = 0;
    rates.lastFrame = null;
    rates.poses = 0;
    rates.poseAt = 0;
    rates.lastPose = null;
    rates.since = now();
  }

  const telemetry = {
    get sessionId() {
      return sessionId;
    },
    get build() {
      return BUILD;
    },
    get enabled() {
      return enabled;
    },

    install() {
      if (installed || !enabled) {
        return telemetry;
      }
      installed = true;
      readToken();
      loadCarried();
      try {
        const query = globalThis.location
          ? globalThis.location.search || ""
          : "";
        if (new URLSearchParams(query).get("telemetry") === "verbose") {
          verbose = true;
        }
      } catch {
        // No query string to read.
      }
      hookConsole();
      drainEarly();
      const target = globalThis.addEventListener ? globalThis : null;
      if (target) {
        // Capturing, so a resource that would not load is caught too: those
        // error events do not bubble.
        target.addEventListener("error", onError, true);
        target.addEventListener("unhandledrejection", onRejection);
        target.addEventListener("pagehide", () => onHide("pagehide"));
      }
      const document0 = globalThis.document;
      if (document0 && document0.addEventListener) {
        document0.addEventListener("visibilitychange", () => {
          const state = document0.visibilityState;
          record("info", "page", `page ${state}`);
          if (state === "hidden") {
            onHide("visibility");
          }
        });
      }
      reportTimer = setInterval(report, limits.reportMs);
      if (reportTimer && typeof reportTimer.unref === "function") {
        reportTimer.unref();
      }
      if (carried.length > 0) {
        schedule(limits.errorFlushMs);
      }
      return telemetry;
    },

    record,
    error: (source, message, extra) => record("error", source, message, extra),
    warn: (source, message, extra) => record("warn", source, message, extra),
    info: (source, message, extra) => record("info", source, message, extra),
    debug: (source, message, extra) => record("debug", source, message, extra),

    // The reliable channel is open: prefer it from now on, and send whatever
    // has been waiting for it.
    attach(send) {
      sender = typeof send === "function" ? send : null;
      readToken();
      backoff = limits.backoffMs;
      record("info", "webrtc", "control channel open");
      flush("attach");
      return telemetry;
    },

    // What the node says it is, from its configuration message: its version
    // and a hash of the very files it serves the page from. A page that came
    // out of the headset's HTTP cache instead of off this node is a whole
    // class of reports that otherwise reads as a node gone mad, so a version
    // that is not this page's is an error record, mirrored to the node's own
    // stderr where the operator is already looking. The hash half is the
    // node's to check against what it serves -- the page cannot hash itself
    // without fetching all of itself again -- so it is only carried.
    nodeBuild(build) {
      if (typeof build !== "string" || !build || build === nodeBuildString) {
        return telemetry;
      }
      nodeBuildString = build;
      const version = build.split("+")[0];
      if (version === BUILD) {
        record("info", "env", `the node is ${build}, this page is ${BUILD}`);
      } else {
        record(
          "error",
          "env",
          `this page is build ${BUILD} but the node is ${version}: the ` +
            "headset is showing a cached page, so reload it",
          { data: { page: BUILD, node: build } },
        );
      }
      return telemetry;
    },

    detach(reason) {
      sender = null;
      record(
        "warn",
        "webrtc",
        `control channel gone${reason ? `: ${reason}` : ""}`,
      );
      // Straight to HTTPS: the case this feature exists for.
      schedule(limits.errorFlushMs);
      return telemetry;
    },

    // An XR session started: what the headset is, what the runtime gave it,
    // and every state change worth knowing afterwards.
    session(session, extra = {}) {
      try {
        const navigator0 = globalThis.navigator || {};
        const data = {
          // Both builds on the one record a human reads first.
          build: BUILD,
          node: nodeBuildString,
          mode: extra.mode,
          view: extra.view,
          blend: session.environmentBlendMode,
          interaction: session.interactionMode,
          visibility: session.visibilityState,
          features: session.enabledFeatures
            ? Array.from(session.enabledFeatures).join(" ")
            : undefined,
          ua: navigator0.userAgent,
          cores: navigator0.hardwareConcurrency,
          dpr: globalThis.devicePixelRatio,
          // Where the page came from, without its fragment: a page served by
          // one node while its frames go to another is worth seeing, and the
          // pairing token is never part of this.
          origin: globalThis.location ? globalThis.location.origin : undefined,
          path: globalThis.location ? globalThis.location.pathname : undefined,
        };
        record("info", "env", "xr session started", { data });
        // Registered before the page's own end handler, so `endReason` still
        // reads whatever the node said about why this session is ending.
        session.addEventListener("end", () => {
          telemetry.sessionEnded(
            (extra.endReason ? extra.endReason() : "") ||
              "ended by the operator or the runtime",
          );
        });
        session.addEventListener("visibilitychange", () => {
          record("info", "xr", `session ${session.visibilityState}`);
        });
        session.addEventListener("inputsourceschange", (event) => {
          const describeSources = (list) =>
            Array.from(list || [], (source) =>
              [
                source.handedness,
                source.hand ? "hand" : "controller",
                (source.profiles || [])[0] || "",
              ]
                .filter(Boolean)
                .join(":"),
            ).join(" ");
          if (event.added && event.added.length) {
            record(
              "info",
              "xr",
              `input added: ${describeSources(event.added)}`,
            );
          }
          if (event.removed && event.removed.length) {
            record(
              "warn",
              "xr",
              `input removed: ${describeSources(event.removed)}`,
            );
          }
        });
      } catch (error) {
        record(
          "warn",
          "telemetry",
          `cannot watch the session: ${describe(error)}`,
        );
      }
      return telemetry;
    },

    sessionEnded(reason) {
      record("info", "xr", `xr session ended${reason ? `: ${reason}` : ""}`);
      // A session that ended is a moment the operator may take the headset
      // off, so do not wait out the ordinary 5 s.
      schedule(limits.errorFlushMs);
      for (const key of Object.keys(tracked)) {
        delete tracked[key];
      }
      return telemetry;
    },

    // One WebGL context, watched for the loss that turns the headset black.
    // `canvas` is the element three.js or the quad renderers draw into.
    context(canvas, label) {
      try {
        if (!canvas || typeof canvas.addEventListener !== "function") {
          return telemetry;
        }
        canvas.addEventListener("webglcontextlost", (event) => {
          // A view being closed frees its own context on purpose
          // (scene3d.js `close()` calls forceContextLoss, because a browser
          // keeps only a handful of contexts and this page builds one per
          // connection). That is housekeeping, not a failure, and recording
          // it as an error would put one in EVERY clean --view scene run --
          // which would teach the operator to ignore the level that matters.
          if (releasing.has(canvas)) {
            record("info", "webgl", `WebGL context released (${label})`);
            return;
          }
          record("error", "webgl", `WebGL context lost (${label})`, {
            data: { restorable: Boolean(event && event.statusMessage === "") },
          });
          // Nothing will be drawn again until it is restored, so this one
          // does not wait for the timer.
          schedule(0);
        });
        canvas.addEventListener("webglcontextrestored", () => {
          record("info", "webgl", `WebGL context restored (${label})`);
        });
      } catch {
        // A canvas that cannot be watched is not worth failing over.
      }
      return telemetry;
    },

    // The page is about to drop this canvas's context on purpose. Said just
    // before `forceContextLoss()`, so the loss that follows is recorded as
    // the housekeeping it is rather than as the failure it looks like.
    contextClosing(canvas) {
      try {
        if (canvas) {
          releasing.add(canvas);
        }
      } catch {
        // A canvas that cannot be marked is only a mislabelled record.
      }
      return telemetry;
    },

    // --- the per-frame hooks, which must stay this cheap --------------------

    // One XR frame was drawn. `time` is the frame's own timestamp, on the
    // same clock as performance.now().
    frame(time) {
      if (rates.lastFrame !== null) {
        frameIntervals[rates.frameAt] = time - rates.lastFrame;
        rates.frameAt = (rates.frameAt + 1) % limits.samples;
      }
      rates.lastFrame = time;
      rates.frames += 1;
      xrTime = time;
    },

    // One pose message went to the node.
    pose(time) {
      if (rates.lastPose !== null) {
        poseIntervals[rates.poseAt] = time - rates.lastPose;
        rates.poseAt = (rates.poseAt + 1) % limits.samples;
      }
      rates.lastPose = time;
      rates.poses += 1;
    },

    // Whether the runtime is tracking something (the head, a controller).
    // Called every frame; records only the edges, so a controller put down
    // and picked up is two lines, not two a frame.
    tracking(which, ok) {
      if (tracked[which] === ok) {
        return;
      }
      const first = tracked[which] === undefined;
      tracked[which] = ok;
      if (first && ok) {
        return;
      }
      record(
        ok ? "info" : "warn",
        "xr",
        ok
          ? `${which} tracked again`
          : `${which} untracked (pose emulated or absent)`,
      );
    },

    flush,

    // Test seam. Nothing on the page reads this.
    stats() {
      return {
        session: sessionId,
        build: BUILD,
        buffered: buffer.length,
        carried: carried.length,
        dropped,
        seq,
        page: pageSeq,
        counters: { ...counters },
        frames: rates.frames,
        poses: rates.poses,
        attached: sender !== null,
        inflight: inflight !== null,
        verbose,
      };
    },
    // Test seam: the buffered records, without taking them.
    peek() {
      return buffer.slice();
    },
    report,
    console0,
  };

  return telemetry;
}

// Installed on import, because this module is loaded before every other one:
// an error while `ar.js` or `connection.js` is still being evaluated has to
// be caught too. The page reaches it as `window.__oqtTelemetry`; nothing
// imports it, so a page that could not fetch this file still runs.
const configured =
  typeof globalThis !== "undefined" && globalThis.__oqtTelemetryConfig
    ? globalThis.__oqtTelemetryConfig
    : {};

export const telemetry = createTelemetry(configured);

if (
  typeof window !== "undefined" &&
  telemetry.enabled &&
  !window.__oqtTelemetry
) {
  window.__oqtTelemetry = telemetry;
  telemetry.install();
}
