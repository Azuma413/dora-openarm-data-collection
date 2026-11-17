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

// The browser half of the WebRTC connection to the node.
//
// Everything the headset and the node say to each other rides one peer
// connection, so this page does not have to be served by the node. That
// is the point: WebRTC needs no certificate from whoever answers, so the
// node can sit on a LAN with no HTTPS server while the page comes from
// somewhere that already has a certificate.
//
// Three data channels, split by what they can afford to lose. "xr" is
// unordered and never retransmitted, and carries the frame messages at
// the animation frame rate: only the newest pose is worth anything, so a
// lost frame is dropped rather than retransmitted. Each frame carries a
// "sequence" so the node can drop the stale ones an unordered channel
// occasionally delivers late. "control" is reliable and carries what must
// not be lost: the node's configuration push, session-start, the select
// and squeeze events, what came of a calibration run, the HUD, and the
// node refusing or ending a session.
//
// "state" (openarm-quest-teleop patch 15) is the node's own unreliable
// channel, carrying the simulation's state to the 3D view at the sim's
// frame rate. It is opened by the node, not here, and it goes only to the
// headset whose session is active; the "xr" channel is not reused for it,
// because that one is the page's, sequence-numbered for one direction, and
// mixing the two would make the takeover rules harder to be sure of. Like
// "xr" it is dropped rather than retransmitted: the next state is 14 ms away.

// No ICE servers by default (openarm-quest-teleop patch). The headset and
// the node share a LAN, where the host candidates alone connect; a STUN
// server only adds a round trip to the Internet before the offer can go,
// and on an offline LAN the browser waits out its gathering timeout for
// one that never answers. The node needs the same, ICE_SERVERS='[]'.
//
// A page opened somewhere a direct path cannot be counted on names its
// servers in the query string instead, since the offer is made before the
// node can say anything: ?ice=stun:host:3478, comma-separated or repeated
// for several. connect({ iceServers }) takes precedence over both, for a
// page that brokers its own signaling and TURN credentials.

// The page's own log (openarm-quest-teleop patch 17). Read from the global
// rather than imported: telemetry.js is loaded by index.html before any
// module, and nothing here may depend on it being there -- a page whose
// telemetry module failed to fetch, and the Node.js stand-ins the unit tests
// run this file against, simply have none.
const telemetry = globalThis.__oqtTelemetry || null;

function iceServersFromQuery(search) {
  const urls = new URLSearchParams(search)
    .getAll("ice")
    .flatMap((value) => value.split(","))
    .map((url) => url.trim())
    .filter((url) => url.length > 0);
  return urls.length > 0 ? [{ urls: urls }] : [];
}

// A headset wired over USB (patch 18). The page is opened as
// https://localhost:8443 through `adb reverse`, which forwards TCP only, so
// no UDP path to the node exists: relay everything through the TURN server
// that scripts/webxr-usb/start.sh runs on the PC (also reverse-forwarded).
// Adopted from guchio123/dora-openarm-data-collection 6676c7c. ?ice= still
// wins, so a page on localhost can be pointed elsewhere.
const USB_TURN = {
  iceServers: [
    {
      urls: ["turn:localhost:3479?transport=tcp"],
      username: "webxr",
      credential: "webxr",
    },
  ],
  iceTransportPolicy: "relay",
};

export function rtcConfiguration(iceServers, location) {
  if (iceServers !== undefined) {
    return { iceServers: iceServers };
  }
  const fromQuery = iceServersFromQuery(location.search);
  if (fromQuery.length === 0 && location.hostname === "localhost") {
    return USB_TURN;
  }
  return { iceServers: fromQuery };
}

// The node decides how many eyes it draws, but this page has to make the
// offer before it can be told, and the signaling here is a single
// exchange with no renegotiation. So always offer this many and let the
// node leave the ones it does not use inactive.
const VIDEO_TRANSCEIVERS = 5;

// Where the pairing token is kept for this tab (openarm-quest-teleop
// patch).
const PAIRING_TOKEN_KEY = "dora-openarm-webxr-pairing-token";

// The pairing token the node asks every offer for (openarm-quest-teleop
// patch), from the URL's fragment, ".../#k=<token>": a fragment never
// leaves the browser in a request or a Referer. It is kept in
// sessionStorage, so a reload of a URL that lost its fragment still
// pairs. Null when there is none; a node started without a token does not
// ask for one.
function pairingToken(hash) {
  const fromHash = new URLSearchParams(hash.replace(/^#/, "")).get("k");
  try {
    if (fromHash) {
      window.sessionStorage.setItem(PAIRING_TOKEN_KEY, fromHash);
      return fromHash;
    }
    return window.sessionStorage.getItem(PAIRING_TOKEN_KEY);
  } catch {
    // Storage can be refused (a private window, blocked site data); the
    // fragment alone still works.
    return fromHash;
  }
}

// What the operator is told when the node turns an offer down.
const SIGNALING_ERRORS = {
  400: "the node could not read this page's offer",
  403: "the node refused this page: open the address with its #k=... pairing token",
  503: "the node has too many connections still starting; try again shortly",
};

// Signaling for the node's own HTTPS server: one POST, because both
// sides gather all their ICE candidates before exchanging anything.
async function postOffer(sdp) {
  const offer = { sdp: sdp };
  const token = pairingToken(window.location.hash);
  if (token) {
    offer.token = token;
  }
  const response = await fetch("offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(offer),
  });
  if (!response.ok) {
    const reason = SIGNALING_ERRORS[response.status];
    // The node would not answer this page's offer, so there is no data
    // channel to say so on: this one reaches the node over HTTPS, which is
    // the transport that still works here (patch 17).
    telemetry?.error(
      "webrtc",
      `the node refused the offer: ${response.status}`,
      {
        data: { status: response.status, reason },
      },
    );
    throw new Error(
      `signaling failed: ${response.status}${reason ? ` (${reason})` : ""}`,
    );
  }
  const answer = await response.json();
  return answer.sdp;
}

// Resolves once the browser has finished gathering; there is no
// promise-based API for this, so bridge the event.
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

// Connect to the node and resolve once it has said how to draw itself.
//
// `signal` takes the offer SDP and resolves with the answer SDP, so a
// differently hosted page can broker signaling its own way without
// changing anything else here.
export async function connect({
  signal = postOffer,
  iceServers = undefined,
} = {}) {
  const pc = new RTCPeerConnection(
    rtcConfiguration(iceServers, window.location),
  );
  const received = [];
  const handlers = {
    calibrationResult: null,
    close: null,
    hud: null,
    sessionNotice: null,
    state: null,
  };
  let control = null;
  let sequence = 0;
  let closed = false;
  let pendingHud = null;
  let pendingSessionNotice = null;

  const xr = pc.createDataChannel("xr", {
    ordered: false,
    maxRetransmits: 0,
  });
  // The pose channel closing is the robot going deaf, and it is silent
  // otherwise: nothing else on this page would notice (patch 17).
  xr.addEventListener("close", () => {
    telemetry?.warn("webrtc", "the pose channel closed");
  });
  xr.addEventListener("error", (event) => {
    telemetry?.error(
      "webrtc",
      `the pose channel failed: ${event?.error?.message || "unknown"}`,
    );
  });

  for (let index = 0; index < VIDEO_TRANSCEIVERS; index++) {
    pc.addTransceiver("video", { direction: "recvonly" });
  }

  pc.addEventListener("track", (event) => {
    // Keyed by mid, because the eye a track carries is decided by
    // negotiation order and the node names that order in its
    // configuration message.
    received.push({ mid: Number(event.transceiver.mid), track: event.track });
  });

  const configured = new Promise((resolve, reject) => {
    // The node going away is noticed in two ways, and either must end
    // the session. The reliable channel closes almost instantly when
    // the node shuts down cleanly; the connection state only reaches
    // "failed" after an ICE timeout, long after a node that died
    // without a word. The channel close is the fast path, the state
    // change the last resort.
    function lost() {
      reject(new Error("the connection to the node was lost"));
      if (!closed && handlers.close) {
        closed = true;
        handlers.close();
      }
    }
    pc.addEventListener("datachannel", (event) => {
      if (event.channel.label === "state") {
        // The simulation's state for the 3D view (patch 15). Binary is the
        // encoding this is meant to end up in, so ask for it as a buffer
        // rather than a Blob, which would have to be read asynchronously and
        // would arrive a frame late.
        const channel = event.channel;
        channel.binaryType = "arraybuffer";
        channel.addEventListener("close", () => {
          telemetry?.warn("webrtc", "the scene state channel closed");
        });
        channel.addEventListener("message", (message) => {
          // Dropped rather than kept when nothing is drawing it: unlike the
          // HUD, a state message is worthless a frame later.
          if (handlers.state) {
            handlers.state(message.data);
          }
        });
        return;
      }
      if (event.channel.label !== "control") {
        return;
      }
      control = event.channel;
      // The reliable channel is the page log's first choice of transport
      // (patch 17): attaching flushes whatever was buffered while there was
      // none, which is how the reason a previous connection died reaches the
      // node as soon as a new one is up.
      const attachTelemetry = () => {
        telemetry?.attach((text) => {
          if (control && control.readyState === "open") {
            control.send(text);
            return true;
          }
          return false;
        });
      };
      if (control.readyState === "open") {
        attachTelemetry();
      }
      control.addEventListener("open", attachTelemetry);
      control.addEventListener("close", () => {
        telemetry?.detach("closed");
      });
      control.addEventListener("close", lost);
      control.addEventListener("message", (message) => {
        let payload;
        try {
          payload = JSON.parse(message.data);
        } catch (error) {
          console.error("cannot read a message from the node: " + error);
          return;
        }
        if (payload.type === "configuration") {
          resolve(payload);
        } else if (payload.type === "calibration-result") {
          if (handlers.calibrationResult) {
            handlers.calibrationResult(payload);
          }
        } else if (payload.type === "hud") {
          // The node repeats its last HUD right after the configuration,
          // which can land before the caller has set a handler: keep the
          // newest one for it rather than dropping it.
          if (handlers.hud) {
            handlers.hud(payload);
          } else {
            pendingHud = payload;
          }
        } else if (
          payload.type === "session-refused" ||
          payload.type === "session-ended"
        ) {
          // The node turned this page's session-start down, or another
          // headset took the session over (openarm-quest-teleop patch).
          if (handlers.sessionNotice) {
            handlers.sessionNotice(payload);
          } else {
            pendingSessionNotice = payload;
          }
        }
      });
    });
    // Every state change the link makes, so a session that went quiet can be
    // told from one whose ICE never connected (patch 17). None of these
    // changes any behaviour; the two that end the session are below.
    pc.addEventListener("iceconnectionstatechange", () => {
      const state = pc.iceConnectionState;
      telemetry?.record(
        state === "failed"
          ? "error"
          : state === "disconnected"
            ? "warn"
            : "info",
        "webrtc",
        `ice ${state}`,
      );
    });
    pc.addEventListener("connectionstatechange", () => {
      const state = pc.connectionState;
      telemetry?.record(
        state === "failed"
          ? "error"
          : state === "disconnected"
            ? "warn"
            : "info",
        "webrtc",
        `connection ${state}`,
      );
      if (pc.connectionState === "failed" || pc.connectionState === "closed") {
        lost();
      }
    });
  });

  let payload;
  try {
    await pc.setLocalDescription(await pc.createOffer());
    await gathered(pc);
    const answer = await signal(pc.localDescription.sdp);
    await pc.setRemoteDescription({ type: "answer", sdp: answer });

    payload = await configured;
  } catch (error) {
    // The page now reconnects after every session, so a failed attempt
    // must not leave its peer behind (openarm-quest-teleop patch).
    closed = true;
    pc.close();
    throw error;
  }
  // The node names the eyes in track order, so a lone track in the mono
  // view is never mistaken for the left eye.
  received.sort((a, b) => a.mid - b.mid);
  const tracks = {};
  payload.eyes.forEach((eye, index) => {
    if (received[index]) {
      tracks[eye] = received[index].track;
    }
  });

  return {
    configuration: payload.view_configuration,
    // What the node says about the 3D scene it can serve (patch 15): the
    // exported glTF's URL, its hash, and the model facts read from it. Null
    // when the node has no scene, which is what keeps the page on the camera
    // panel. It rides beside the view configuration rather than inside it
    // because the node works it out at startup from --scene-glb, while the
    // view configuration is a file somebody wrote.
    scene: payload.scene || null,
    // What the node says it is: its version and a hash of the page files it
    // serves (patch 17). Straight to the log, which is the only place it
    // means anything -- a page out of the headset's cache says so there.
    build: typeof payload.build === "string" ? payload.build : null,
    calibration: { enabled: payload.calibration === true },
    eyes: payload.eyes,
    tracks: tracks,
    // Frames are numbered here so the node can drop the ones this
    // unordered channel delivers late.
    sendFrame(message) {
      if (xr.readyState !== "open") {
        return;
      }
      sequence += 1;
      xr.send(JSON.stringify({ ...message, sequence }));
    },
    sendControl(message) {
      if (control && control.readyState === "open") {
        control.send(JSON.stringify(message));
      }
    },
    onCalibrationResult(handler) {
      handlers.calibrationResult = handler;
    },
    // The node's status text for the operator, and which controller to
    // buzz.
    onHud(handler) {
      handlers.hud = handler;
      if (pendingHud) {
        handler(pendingHud);
        pendingHud = null;
      }
    },
    // The simulation's state for the 3D view, as it arrives: a JSON string
    // or an ArrayBuffer, decoded by scene-state.js. A node that opens no
    // "state" channel simply never calls this.
    onState(handler) {
      handlers.state = handler;
    },
    // {type: "session-refused" | "session-ended", reason}: the node will
    // not take this page's frames, so its session should end.
    onSessionNotice(handler) {
      handlers.sessionNotice = handler;
      if (pendingSessionNotice) {
        handler(pendingSessionNotice);
        pendingSessionNotice = null;
      }
    },
    onClose(handler) {
      handlers.close = handler;
    },
    close() {
      closed = true;
      pc.close();
    },
  };
}
