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

import { connect } from "./connection.js";
import { createHudPanel } from "./hud.js";
import { createInstructionPanel } from "./instructions.js";
import { createCameraPanel, createCameraPanels } from "./panel.js";
import { createStereoPanel } from "./stereo.js";

// The page's own log, sent back to the node by itself (openarm-quest-teleop
// patch 17): debugging a headset through developer mode means a cable and
// chrome://inspect, by which time the failure is an hour old.
//
// Read from the global rather than imported. index.html loads telemetry.js as
// its own module before this one, so it is installed before anything here can
// fail, and nothing here may depend on it: a page whose telemetry module
// failed to fetch -- and the Node.js stand-ins the unit tests run this file
// against -- get `null` and every call below is skipped.
const telemetry = globalThis.__oqtTelemetry || null;

// Used only when the node's view configuration cannot be read.
const FALLBACK_CONFIGURATION = {
  view: "mono",
  session: { mode: "immersive-ar" },
  panel: { lock: "room", distance: 1.3, width: 1.5 },
};

// The controller pulse a HUD message asks for: strong enough to feel
// through a squeezed grip, short enough not to read as a fault.
const HAPTIC = { intensity: 0.6, duration: 80 };

// How long no frame is sent after the local space resets, in
// milliseconds. A recenter moves the origin under hands the node is
// following; a gap longer than the clutch's 100 ms freshness and the
// 200 ms pose timeout makes both let go instead of jumping.
const RESET_PAUSE = 300;

// The views that draw the sim as real objects around the operator
// (openarm-quest-teleop patch 15), and whether each still shows the head
// camera. Everything else -- "mono", "stereo", "none" -- is the camera panel
// as it always was, and never loads three.js or scene3d.js at all.
const SCENE_VIEWS = {
  scene: { panel: false },
  "scene+panel": { panel: true },
};

if (navigator.xr) {
  let connection = null;
  // The connection being opened, so a Start pressed meanwhile waits for
  // it instead of opening a second one.
  let connecting = null;
  let starting = false;
  let runningSession = null;
  let configuration = null;
  let cameraPanel = null;
  let auxiliaryPanels = null;
  let instructionPanel = null;
  // The 3D view, when the node asked for one and it could be built
  // (openarm-quest-teleop patch 15). Null in every other view, and null again
  // when a 3D view could not be built, which is what makes the page fall back
  // to the camera panel or to passthrough alone.
  let sceneView = null;
  // Outlives connections: the node repeats its last HUD to each new session.
  const hudPanel = createHudPanel();
  // Why the node refused or ended the last session, if it said
  // (openarm-quest-teleop patch). Kept above the connection state until a
  // session starts again, because the reconnect that follows the session's
  // end would otherwise overwrite it before the operator is out of the
  // headset to read it.
  let notice = "";
  // What the 3D view could not do, if anything (openarm-quest-teleop patch
  // 14). Shown in the headset on the HUD panel, and here for whoever is
  // looking at the page.
  let sceneNotice = "";
  let statusText = "";

  // The operator's only feedback outside the headset: the log element
  // under the Start button, which upstream leaves empty.
  function showStatus(message) {
    statusText = message;
    const element = document.getElementById("log");
    if (element) {
      const lines = [notice, message, sceneNotice].filter((line) => line);
      element.innerText = lines.join("\n");
    }
  }

  // A line for the operator about the 3D view, on the HUD panel where they
  // can read it inside the headset, and on the page as well.
  function showSceneNotice(text) {
    sceneNotice = text || "";
    if (hudPanel.setNotice) {
      hudPanel.setNotice(sceneNotice);
    }
    showStatus(statusText);
  }

  // The node pushes its configuration as soon as the connection opens,
  // so this page needs no route of its own to read it: whoever serves
  // this file, the node still decides how it draws itself. The session
  // can only start once this has resolved, so the view and the panels
  // are always set by then.
  //
  // Opened on page load and again after every session end, since ending
  // a session closes the connection (openarm-quest-teleop patch: upstream
  // connected once per page load, so a second Start needed a reload).
  function open() {
    showStatus("Connecting to the node…");
    return connect().then(async (opened) => {
      connection = opened;
      configuration = opened.configuration || FALLBACK_CONFIGURATION;
      // Which build of this page the node serves, against the one the
      // headset is actually running (patch 17). A cached page reads as a
      // node gone mad otherwise.
      telemetry?.nodeBuild(opened.build);

      // A connection lost while idle leaves the last one's video behind.
      if (cameraPanel) {
        cameraPanel.close();
      }
      auxiliaryPanels?.close();
      if (sceneView) {
        sceneView.close();
      }
      cameraPanel = null;
      auxiliaryPanels = null;
      instructionPanel = null;
      sceneView = null;
      showSceneNotice("");

      // The 3D view, if the node asked for one: the sim drawn as real
      // objects around the operator (patch 15). Loaded here rather than at
      // the session, so three.js and the scene asset are in hand before
      // Start and the session opens straight away -- and so that a node that
      // cannot supply them is found out while the operator is still looking
      // at the page.
      const wants3d = SCENE_VIEWS[configuration.view];
      if (wants3d) {
        sceneView = await openSceneView(opened, wants3d.panel);
      }

      // How many images: the default "mono" draws one, "stereo" draws one
      // per eye, and "none" shows no camera at all. Where they hang is a
      // separate question, answered by "panel: lock" in the configuration.
      //
      // A 3D view draws its own camera panel, because three.js owns the
      // session's render state there and these quad renderers have no base
      // layer to draw into. So they are built only when there is no 3D view:
      // either the node asked for none, or one was asked for and could not be
      // built, and then "scene+panel" falls back to the mono panel and
      // "scene" to passthrough and the HUD alone (patch 15, X5).
      if (sceneView) {
        // Drawn by scene3d.js.
      } else if (configuration.view === "stereo") {
        cameraPanel = createStereoPanel(configuration, opened.tracks);
      } else if (wants3d ? wants3d.panel : configuration.view !== "none") {
        cameraPanel = createCameraPanel(configuration, opened.tracks.right);
      }
      if (!sceneView) {
        auxiliaryPanels = createCameraPanels(configuration, opened.tracks);
      }

      // Only a node started with --calibration wants the instructions, and
      // only it acts on the Y button, so the panel that asks for a head
      // turn is never in front of an operator whose Y button means
      // something else. Read after the view, since the panel has to know
      // whether a camera view is drawing the frame or it is the only one
      // there. It is a quad renderer too, so a 3D view does without it and
      // says so rather than crashing on the missing base layer.
      if (opened.calibration.enabled && sceneView) {
        showSceneNotice(
          "Calibration needs a 2D view; run it with --view mono.",
        );
      } else if (opened.calibration.enabled) {
        instructionPanel = createInstructionPanel({
          clears: cameraPanel === null,
        });
      }

      opened.onCalibrationResult((message) => {
        if (instructionPanel) {
          instructionPanel.setResult(message);
        }
      });

      // A node restarted since the last connection may have nothing to
      // say yet, so its predecessor's text must not stay up.
      hudPanel.set({ lines: [] });
      opened.onHud(onHud);

      // The node takes one session at a time (openarm-quest-teleop
      // patch): it refuses a Start while another headset is sending, and
      // tells a headset whose idle session another one took over. Either
      // way nothing this page sends reaches the robot any more, so the
      // session ends, and the reason stays on the page for the operator
      // who takes the headset off to find out why.
      opened.onSessionNotice((message) => {
        if (connection !== opened) {
          return;
        }
        telemetry?.warn("xr", `the node said ${message.type}`, {
          data: { reason: message.reason },
        });
        if (message.type === "session-refused") {
          notice = `The node refused the session: ${message.reason || "another headset is active"}.`;
        } else if (message.reason === "taken-over") {
          notice = "Another headset took over the session.";
        } else {
          notice = `The node ended the session (${message.reason}).`;
        }
        showStatus("Press Start to try again.");
        hudPanel.set({ lines: [] });
        if (runningSession) {
          // onSessionEnd closes this connection and opens a fresh one.
          runningSession.end();
        }
      });

      // Losing the connection ends the session: an operator left inside a
      // frozen view has no way to tell that nothing is reaching the robot.
      opened.onClose(() => {
        // A late close of a connection already replaced must not drop
        // the one that replaced it.
        if (connection !== opened) {
          return;
        }
        connection = null;
        telemetry?.error("webrtc", "lost the node", {
          data: { inSession: runningSession !== null },
        });
        // A HUD left up over a dead connection would keep asserting a
        // state (READY, FOLLOW) nobody is producing (openarm-quest-teleop
        // patch).
        hudPanel.set({ lines: [] });
        if (runningSession) {
          // Kept as a notice: ending the session reconnects, and the
          // connection state would otherwise replace the reason.
          notice = "Lost the node during the session.";
          showStatus("Press Start to reconnect.");
          runningSession.end();
          runningSession = null;
        } else {
          showStatus("Lost the node. Press Start to reconnect.");
        }
      });

      showStatus("Connected. Press Start.");
      return configuration;
    });
  }

  // Build the 3D view for this connection, or return null with a notice
  // saying why there is none (openarm-quest-teleop patch 15). scene3d.js and
  // three.js are imported here and nowhere else, so a 2D view never fetches
  // them.
  async function openSceneView(opened, panel) {
    try {
      const { createScene3d } = await import("./scene3d.js");
      const view = await createScene3d({
        // Two sources, one block: the asset and the model facts come from the
        // node, which read them out of the export it is serving, and where the
        // robot stands comes from the view configuration file, which is where
        // whoever set the room up can change it.
        configuration: {
          ...configuration,
          scene: { ...(configuration.scene || {}), ...(opened.scene || {}) },
        },
        tracks: opened.tracks,
        // The same canvas the HUD panel lays its text out on, drawn as a
        // texture here instead of through its own shader.
        hud: hudPanel.content,
        panel,
        onNotice: showSceneNotice,
      });
      // A node with no state channel never calls this, and the scene then
      // stands at the pose the asset was exported in and says so.
      if (opened.onState) {
        opened.onState((data) => view.setState(data));
      }
      return view;
    } catch (error) {
      // The asset it could not build from, and the error object rather than
      // its text: the telemetry console hook reads a stack off an Error
      // argument (patch 17), and which file this was is the whole question
      // when a headset draws nothing.
      console.error(
        `cannot build the 3D view (asset ${opened.scene ? opened.scene.url : "none"}):`,
        error,
      );
      // What is left to look at depends on the view: "scene+panel" falls back
      // to the head camera, "scene" renders no camera at all (its sim runs with
      // `--cameras=`), so telling that operator to look at a camera view would
      // send them hunting for a panel that does not exist.
      showSceneNotice(
        `3D scene unavailable (${error.message}); ${
          panel
            ? "showing the camera view instead."
            : "this view has no camera panel: passthrough and the HUD only."
        }`,
      );
      return null;
    }
  }

  // Resolve with the configuration once there is a connection, opening
  // one only if there is none and none is on its way.
  function ensureConnected() {
    if (connection) {
      return Promise.resolve(configuration);
    }
    if (!connecting) {
      connecting = open().finally(() => {
        connecting = null;
      });
    }
    return connecting;
  }

  function onHud(message) {
    hudPanel.set(message);
    if (message.haptic && runningSession) {
      pulse(runningSession, message.haptic);
    }
  }

  // Buzz the controllers ``which`` names: "right", "left" or "both".
  // hapticActuators is what Quest Browser implements for WebXR gamepads;
  // vibrationActuator is the standard Gamepad API's, for browsers that
  // only have that. Neither is guaranteed, so a missing one is skipped.
  function pulse(session, which) {
    for (const source of session.inputSources) {
      const gamepad = source.gamepad;
      if (!gamepad || (which !== "both" && source.handedness !== which)) {
        continue;
      }
      try {
        const actuator = gamepad.hapticActuators?.[0];
        if (actuator && typeof actuator.pulse === "function") {
          Promise.resolve(
            actuator.pulse(HAPTIC.intensity, HAPTIC.duration),
          ).catch(() => {});
        } else if (gamepad.vibrationActuator?.playEffect) {
          gamepad.vibrationActuator
            .playEffect("dual-rumble", {
              duration: HAPTIC.duration,
              strongMagnitude: HAPTIC.intensity,
            })
            .catch(() => {});
        }
      } catch (error) {
        console.error(`cannot pulse the controller: ${error}`);
      }
    }
  }

  function sendControl(message) {
    if (connection) {
      connection.sendControl(message);
    }
  }

  function log(message) {
    // document.getElementById("log").innerText += `${message}\n`;
    // sendControl({type: "log", message: `${message}`});
  }
  function onSessionEnd(event) {
    log("ended");
    // The session's end is recorded by telemetry's own listener, registered
    // in onSessionStart before this one and reading the same `notice` for
    // the reason (patch 17).
    if (cameraPanel) {
      cameraPanel.close();
      cameraPanel = null;
    }
    auxiliaryPanels?.close();
    auxiliaryPanels = null;
    if (sceneView) {
      // Stopped, not closed: the connection is about to be replaced, and
      // whatever replaces it builds its own view.
      sceneView.stop();
    }
    runningSession = null;
    if (connection) {
      connection.close();
      connection = null;
    }
    // Reconnect now rather than on the next Start: a session can only be
    // requested close to the click, and a connection that is already
    // open lets Start go straight to it. A node that is gone fails here,
    // and Start tries again.
    ensureConnected().catch((error) => {
      showStatus(
        `Cannot reach the node (${error.message}). Press Start to retry.`,
      );
    });
  }
  function onSelectStart(event) {
    const response = {
      type: "select-start",
      buttons: event.inputSource.gamepad.buttons,
      axes: event.inputSource.gamepad.axes,
    };
    sendControl(response);
  }
  function onSelect(event) {
    const response = {
      type: "select",
      buttons: event.inputSource.gamepad.buttons,
      axes: event.inputSource.gamepad.axes,
    };
    sendControl(response);
  }
  function onSelectEnd(event) {
    const response = {
      type: "select-end",
      buttons: event.inputSource.gamepad.buttons,
      axes: event.inputSource.gamepad.axes,
    };
    sendControl(response);
  }
  function onSqueezeStart(event) {
    const response = {
      type: "squeeze-start",
      buttons: event.inputSource.gamepad.buttons,
      axes: event.inputSource.gamepad.axes,
    };
    sendControl(response);
  }
  function onSqueeze(event) {
    const response = {
      type: "squeeze",
      buttons: event.inputSource.gamepad.buttons,
      axes: event.inputSource.gamepad.axes,
    };
    sendControl(response);
  }
  function onSqueezeEnd(event) {
    const response = {
      type: "squeeze-end",
      buttons: event.inputSource.gamepad.buttons,
      axes: event.inputSource.gamepad.axes,
    };
    sendControl(response);
  }
  function sendFrameResponse(response) {
    if (instructionPanel) {
      // An absent button is a released one, which is how the node reads
      // it too, so a controller that falls asleep or goes missing ends
      // the run on the panel just as it ends it there.
      instructionPanel.setPressed(response.button_y === true);
    }
    // Frames go on the unreliable channel, which numbers them: only the
    // newest pose is worth anything, so a lost one is dropped rather
    // than retransmitted behind the frames that follow it.
    if (connection) {
      connection.sendFrame(response);
      // One store into a ring and a counter (patch 17): the interval between
      // pose sends is what the node's own gap line measures from the other
      // end, and having both is how "the page never sent it" is told from
      // "the link lost it". Nothing here sends anything.
      telemetry?.pose(response.time);
    }
  }
  // Send one frame to the node, and say whether it carried a head pose: the
  // node latches its neck pivot from those, and the 3D view re-anchors the
  // robot on exactly the gaps between them (patch 13, patch 15).
  function sendFrame(session, space, time, frame) {
    const response = {
      type: "frame",
      time: time,
    };
    // Read before the controller check below, so a consumer driving something
    // from head motion keeps tracking while the controllers are off or asleep.
    // The hand poses are made relative to this pose by the node, so when it is
    // missing the node drops them and only the head keeps flowing.
    const viewerPose = frame.getViewerPose(space);
    // Only the edges are recorded, never the state each frame: a head the
    // runtime stopped tracking is why the node holds both hands, and it is
    // invisible from the node's end (patch 17).
    telemetry?.tracking(
      "head",
      Boolean(viewerPose) && !viewerPose.emulatedPosition,
    );
    // An emulated position is the runtime guessing (tracking lost, IMU
    // only), and every hand position is made relative to this one, so a
    // guessed head is not sent and the node holds both hands.
    if (viewerPose && !viewerPose.emulatedPosition) {
      const transform = viewerPose.transform;
      response.pose_reference = {
        x: transform.position.x,
        y: transform.position.y,
        z: transform.position.z,
        qx: transform.orientation.x,
        qy: transform.orientation.y,
        qz: transform.orientation.z,
        qw: transform.orientation.w,
      };
    }
    // Whatever controllers are there are sent, one or two: upstream sent
    // nothing below two, so one controller asleep froze both arms and
    // the episode buttons with them. Hands (hand tracking, or a Quest 3
    // tracking both) are not controllers: they have no trigger to hold a
    // gripper with, and would overwrite the controller of the same hand.
    for (const source of session.inputSources) {
      if (source.handedness === "none" || source.hand || !source.gamepad) {
        continue;
      }
      const suffix = `_${source.handedness}`;
      // The target ray space is the OpenXR aim pose: its -Z points where
      // the controller points. The node maps the gripper onto that axis
      // directly, so it must not be the grip pose, whose -Z runs along the
      // handle and would turn the handle's tilt into the gripper's.
      const pose = frame.getPose(source.targetRaySpace, space);
      // A controller out of the cameras' sight reports an emulated
      // position that drifts. Its pose, trigger and grip are left out,
      // so the node holds that hand and the clutch lets go of it; its
      // buttons still go, since they move nothing and the operator may
      // need them to end an episode with the hand out of view.
      const tracked = Boolean(pose) && !pose.emulatedPosition;
      telemetry?.tracking(source.handedness, tracked);
      if (tracked) {
        response[`pose${suffix}`] = {
          x: pose.transform.position.x,
          y: pose.transform.position.y,
          z: pose.transform.position.z,
          qx: pose.transform.orientation.x,
          qy: pose.transform.orientation.y,
          qz: pose.transform.orientation.z,
          qw: pose.transform.orientation.w,
        };
      }
      const gamepad = source.gamepad;
      if (gamepad) {
        if (
          source.profiles.includes("pico-4u") ||
          source.profiles.includes("meta-quest-touch-plus") ||
          source.profiles.includes("oculus-touch-v3")
        ) {
          const trigger = gamepad.buttons[0];
          if (tracked) {
            response[`trigger${suffix}`] = trigger.value;
          }

          // The squeeze sits next to the trigger under the xr-standard
          // mapping. Sent as its 0..1 value rather than a pressed flag, so
          // that a consumer can pick its own threshold.
          const grip = gamepad.buttons[1];
          if (grip && tracked) {
            response[`grip${suffix}`] = grip.value;
          }
          if (source.handedness === "right") {
            const a = gamepad.buttons[4];
            const b = gamepad.buttons[5];
            response.button_a = a.pressed;
            response.button_b = b.pressed;
          } else {
            const x = gamepad.buttons[4];
            const y = gamepad.buttons[5];
            response.button_x = x.pressed;
            response.button_y = y.pressed;
          }
        }
        // Send the whole array and let the node pick the pair it wants.
        // Testing the axes for truthiness, as this once did, meant the
        // joystick was never sent from a Quest at all: its Touch controllers
        // have no touchpad and report axes[0..1] as a constant 0, so the
        // check could never pass. A centred thumbstick reads exactly 0 too,
        // which would have gated out the resting position even on a device
        // that does have a touchpad.
        if (gamepad.axes.length >= 2) {
          response[`joystick${suffix}`] = Array.from(gamepad.axes);
        }
      }
    }
    sendFrameResponse(response);
    return response.pose_reference !== undefined;
  }
  function onSessionStart(session) {
    runningSession = session;
    // A new attempt: the reason the last one ended no longer applies.
    notice = "";
    // TODO: Add device information
    const response = {
      type: "session-start",
    };
    sendControl(response);

    // What this headset is and what the runtime gave it, once per session
    // (patch 17): the user agent, the enabled features, the blend mode. A
    // Quest Browser that quietly refused a feature is otherwise invisible,
    // and so is a headset running a page cached from an older run. The
    // session's own end, visibility and input-source changes are watched
    // from here too.
    telemetry?.session(session, {
      mode: configuration ? configuration.session.mode : null,
      view: configuration ? configuration.view : null,
      endReason: () => notice,
    });

    session.addEventListener("end", onSessionEnd);

    session.addEventListener("selectstart", onSelectStart);
    session.addEventListener("select", onSelect);
    session.addEventListener("selectend", onSelectEnd);

    session.addEventListener("squeezestart", onSqueezeStart);
    session.addEventListener("squeeze", onSqueeze);
    session.addEventListener("squeezeend", onSqueezeEnd);

    if (sceneView) {
      startSceneSession(session);
    } else {
      startPanelSession(session);
    }
  }

  // The 3D view takes the session over: three.js makes the layer the headset
  // draws into (an XRProjectionLayer where the browser has one, which is why
  // the quad renderers cannot share the session) and runs the frame loop.
  // The node still gets exactly the frames it got before, from the same
  // "local" space, paused by the same rules.
  function startSceneSession(session) {
    // Only the "local" space here: the node's frames are read in it, as they
    // always were, and the scene keeps its own floor-relative space for
    // drawing (scene3d.js), which is a different question.
    session
      .requestReferenceSpace("local")
      .then((localSpace) => {
        let pausedUntil = 0;
        localSpace.addEventListener("reset", () => {
          pausedUntil = performance.now() + RESET_PAUSE;
        });
        return sceneView.start(session, {
          onFrame(time, frame) {
            // As in the panel loop: a store and a counter, no sending (patch
            // 17). Counted before the visibility gate, because a frame drawn
            // while nothing is sent is exactly the difference worth seeing.
            telemetry?.frame(time);
            if (
              session.visibilityState !== "visible" ||
              performance.now() < pausedUntil
            ) {
              return false;
            }
            return sendFrame(session, localSpace, time, frame);
          },
        });
      })
      .catch((error) => {
        // Whatever went wrong, the operator must not be left in a session
        // that draws nothing: hand it to the panel renderers, which need
        // only a base layer of their own, and say what happened.
        console.error("cannot start the 3D view:", error);
        sceneView.stop();
        sceneView = null;
        showSceneNotice(`3D scene stopped (${error.message}).`);
        startPanelSession(session);
      });
  }

  function startPanelSession(session) {
    // The render state is needed to use immersive AR even when nothing
    // is drawn into it. The camera views draw their panel into it.
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl", { xrCompatible: true });
    // A context lost here is a headset that draws nothing from now on, and
    // the page has no other way to say so (patch 17).
    telemetry?.context(canvas, "panel");
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
    if (cameraPanel) {
      cameraPanel.attach(gl);
    }
    auxiliaryPanels?.attach(gl);
    if (instructionPanel) {
      instructionPanel.attach(gl);
    }
    hudPanel.attach(gl);

    Promise.all([
      // The hand poses and the head pose are both read in the
      // world-fixed local space, and the node makes the hand positions
      // relative to the head pose without inheriting the head rotation.
      // The panel hangs in whichever space "panel: lock" names: the
      // room is the local space, the head is the viewer space.
      session.requestReferenceSpace("viewer"),
      session.requestReferenceSpace("local"),
    ])
      .then(([viewerSpace, localSpace]) => {
        const panel = configuration.panel || {};
        const panelSpace = panel.lock === "head" ? viewerSpace : localSpace;
        let pausedUntil = 0;
        localSpace.addEventListener("reset", () => {
          pausedUntil = performance.now() + RESET_PAUSE;
        });
        function onFrame(time, frame) {
          log("sources: " + session.inputSources.length);
          // One subtraction, one store and a counter (patch 17). Sending is
          // never done from here: telemetry batches on its own timer.
          telemetry?.frame(time);
          // Nothing is sent while the session is not "visible": under the
          // system menu ("visible-blurred") the poses are not the
          // operator's to give, and with the headset off ("hidden") they
          // are stale. The node's pose timeout then holds the arms.
          if (
            session.visibilityState === "visible" &&
            performance.now() >= pausedUntil
          ) {
            sendFrame(session, localSpace, time, frame);
          }
          if (cameraPanel) {
            cameraPanel.render(session, panelSpace, frame);
            auxiliaryPanels?.render(session, panelSpace, frame, false);
          } else if (auxiliaryPanels?.length) {
            auxiliaryPanels.render(session, panelSpace, frame, true);
          }
          // After the camera view, so the text is over the image rather
          // than cleared away with the rest of the frame, and always in
          // the viewer space so it stays readable through a head turn.
          if (instructionPanel) {
            instructionPanel.render(session, viewerSpace, frame);
          }
          hudPanel.render(session, viewerSpace, frame, {
            clears:
              cameraPanel === null &&
              !auxiliaryPanels?.length &&
              instructionPanel === null,
          });
          session.requestAnimationFrame(onFrame);
        }
        session.requestAnimationFrame(onFrame);
      })
      .catch((error) => {
        telemetry?.error("xr", `cannot start the session's panels: ${error}`, {
          stack: error && error.stack,
        });
        alert(error);
      });
  }
  // The session mode is configured with the head camera view so that
  // passthrough can be turned off without changing this file.
  //
  // With a connection already open the session is requested within the
  // click, as WebXR requires. Without one it is requested once the
  // connection opens, which on a LAN with no ICE servers is well inside
  // the time a click stays valid for; if it is not, the status says so
  // and the next Start finds the connection open.
  function onStart() {
    if (runningSession || starting) {
      return;
    }
    starting = true;
    // The connection the session is requested on (openarm-quest-teleop
    // patch). requestSession can take seconds on a Quest -- the first-use
    // permission prompt, the session starting up -- and a connection lost
    // in that window has already run its close handler, with no session
    // to end yet. Starting the session anyway would leave the operator in
    // one that sends nothing and never ends.
    let startedWith = null;
    ensureConnected()
      .then((configuration) => {
        startedWith = connection;
        // A 3D view asks for a floor-relative space as well (patch 15): an
        // immersive session comes with "viewer" and "local" only, and the
        // scene's heights are measured from the operator's own floor. It is
        // optional, so a runtime without one still gets its session, and the
        // scene then works the floor out from the head height instead. The
        // camera views ask for nothing extra, exactly as before.
        if (sceneView) {
          return navigator.xr.requestSession(configuration.session.mode, {
            optionalFeatures: ["local-floor"],
          });
        }
        return navigator.xr.requestSession(configuration.session.mode);
      })
      .then((session) => {
        if (connection === null || connection !== startedWith) {
          // The end listener is not attached yet, so nothing reconnects
          // on its own: do it here, as onSessionEnd would.
          try {
            Promise.resolve(session.end()).catch(() => {});
          } catch (error) {
            console.error(`cannot end the session: ${error}`);
          }
          notice = "Lost the node while starting the session.";
          showStatus("Press Start to retry.");
          ensureConnected().catch((error) => {
            showStatus(
              `Cannot reach the node (${error.message}). Press Start to retry.`,
            );
          });
          return;
        }
        onSessionStart(session);
      })
      .catch((error) => {
        telemetry?.error("xr", `cannot start a session: ${error.message}`, {
          stack: error.stack,
        });
        showStatus(`Cannot start (${error.message}). Press Start to retry.`);
      })
      .finally(() => {
        starting = false;
      });
  }

  // Test seam (openarm-quest-teleop patch 15). tests/web/scene3d.test.mjs
  // drives this page from outside the headset, and the 3D view's state --
  // where it put the robot, what it could not do -- is not visible anywhere
  // else: WebXR draws into the session's own framebuffer, not the page.
  // Read-only, and nothing on the page reads it.
  if (typeof window !== "undefined") {
    window.__oqtPage = {
      get view() {
        return configuration ? configuration.view : null;
      },
      get scene() {
        return sceneView;
      },
      get sceneNotice() {
        return sceneNotice;
      },
      get hud() {
        return {
          ...hudPanel.state,
          // What the panel is actually showing the operator, the page's own
          // line included: the 2D views draw it through their shader, where a
          // test outside the headset cannot read the pixels at all.
          notice: hudPanel.content ? hudPanel.content.notice : "",
        };
      },
    };
  }

  // WebXR requires explicit user interaction on start. We use button
  // click here. Wired before the first connection settles, so that Start
  // also retries a node that was not up when the page loaded.
  document.getElementById("start").addEventListener("click", onStart);
  ensureConnected()
    .then((configuration) => {
      const mode = configuration.session.mode;
      return navigator.xr.isSessionSupported(mode).then((isSupported) => {
        if (!isSupported) {
          showStatus(`This browser does not support ${mode}.`);
        }
      });
    })
    .catch((error) => {
      console.error("cannot connect to the node:", error);
      showStatus(
        `Cannot reach the node (${error.message}). Press Start to retry.`,
      );
    });
} else {
  // No WebXR at all: not a headset, or a browser that will not give it. The
  // page can do nothing, and the operator will say "it does not work", so
  // this is exactly the case the log has to carry (patch 17).
  globalThis.__oqtTelemetry?.error(
    "env",
    "this browser has no navigator.xr, so nothing on this page will run",
    { data: { ua: navigator.userAgent } },
  );
}
