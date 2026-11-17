// Addition to the vendored dora-openarm-webxr tests (Apache-2.0, like the
// rest of this directory).
//
// Which ICE configuration the page offers with (PATCHES.md 1 and 18), in
// Node.js alone:
//
//   node nodes/dora-openarm-webxr/tests/web/connection_units.test.mjs

import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MODULE = pathToFileURL(
  path.join(HERE, "../../src/dora_openarm_webxr/static/connection.js"),
).href;

const { rtcConfiguration } = await import(MODULE);

function at(url) {
  const parsed = new URL(url);
  return { hostname: parsed.hostname, search: parsed.search };
}

// LAN: no ICE servers at all (patch 1).
assert.deepEqual(rtcConfiguration(undefined, at("https://spark.local:8443/")), {
  iceServers: [],
});

// USB through adb reverse: relay only, through the PC's TURN (patch 18).
const usb = rtcConfiguration(undefined, at("https://localhost:8443/"));
assert.equal(usb.iceTransportPolicy, "relay");
assert.deepEqual(usb.iceServers[0].urls, ["turn:localhost:3479?transport=tcp"]);

// ?ice= wins over the USB default, also on localhost.
assert.deepEqual(
  rtcConfiguration(undefined, at("https://localhost:8443/?ice=stun:h:3478")),
  { iceServers: [{ urls: ["stun:h:3478"] }] },
);

// An explicit connect({ iceServers }) wins over both.
assert.deepEqual(rtcConfiguration([], at("https://localhost:8443/")), {
  iceServers: [],
});

console.log("connection_units: 4 cases passed");
