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

// openarm-quest-teleop addition (PATCHES.md 15): the simulated cell drawn as
// real objects around the operator, over passthrough.
//
// What the node sends and what this draws:
//
//   configuration.scene   once, on the reliable channel: which asset to load,
//                         where the robot stands and what the sim's own frame
//                         is (see readSceneConfiguration below)
//   "state" channel       one message per sim frame: qpos, free-object poses
//                         and the clutch / recording / fault flags
//
// three.js owns the session's render state here, because on a headset it
// takes the XRProjectionLayer path and `session.renderState.baseLayer` is
// then null -- the hand-written quad renderers (panel.js, stereo.js, hud.js)
// cannot draw in that session at all. So in a 3D view the camera panel and
// the HUD are meshes in this scene instead, drawn from the same video track
// and the same HUD canvas as before. The 2D views are untouched: ar.js only
// loads this file when the node asks for a 3D one.
//
// Nothing here is allowed to leave the operator with a black headset. Every
// failure -- no asset, a refused asset, no state, a stalled stream -- ends in
// a notice on the HUD and whatever else can still be drawn.

import {
  advanceNudge,
  CONTROLLER_TO_EE,
  DEFAULT_ARM_ORIGIN_WORLD,
  DEFAULT_EXO,
  DEFAULT_EYE_HEIGHT,
  DEFAULT_FRAME_OFFSET,
  DEFAULT_NECK_PIVOT_OFFSET,
  emptyNudge,
  headingYaw,
  neckPivot,
  placementMatrix,
} from "./align.js";
import { PANEL as HUD_PANEL } from "./hud.js";
import { Kinematics } from "./mjkin.js";
import { decodeSceneState, isNewer, STALE_AFTER_MS } from "./scene-state.js";
import { GLTFLoader } from "./three/GLTFLoader.js";
import {
  AmbientLight,
  BoxGeometry,
  Camera,
  CanvasTexture,
  Color,
  DirectionalLight,
  Group,
  LinearFilter,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Quaternion,
  Scene,
  SRGBColorSpace,
  Vector3,
  VideoTexture,
  WebGLRenderer,
  WebGLRenderTarget,
} from "./three/three.module.min.js";
import { createVideoElement } from "./video-source.js";

// The node re-latches its neck pivot after a gap in head poses longer than
// its SMOOTHER_RESET_GAP (main.py, PATCHES.md 13). The scene re-anchors on
// exactly the same gap, so the robot never moves under a hand that is
// commanding it: by the time the pivot moves, the clutch has let both arms go.
const REANCHOR_GAP_MS = 100;

// How long a session runs with no state at all before the operator is told.
// Longer than STALE_AFTER_MS, because at the start of a session the node has
// a dataflow to get going first.
const NO_STATE_AFTER_MS = 1500;

const GHOST = { length: 0.12, width: 0.04, height: 0.02 };

// How many frames of timing are kept: eight seconds at 72 Hz, which is a
// median and a p95 worth having and nothing that grows.
const FRAME_SAMPLES = 600;

export const NOTICES = {
  noAsset: "3D scene: no asset from the node; showing passthrough only.",
  staleAsset: "3D scene: the cached asset is stale; showing passthrough only.",
  noState: "3D scene: no state from the sim; the robot is at its rest pose.",
  staleState: "3D scene: the sim's state stream stopped; the robot is frozen.",
};

// Read the node's `configuration.scene`, filling in what an older node does
// not send. Every field is optional on purpose: a page that refuses to draw
// because one number is missing is worse than one that draws the robot a
// little off and says so.
export function readSceneConfiguration(configuration) {
  const scene = configuration?.scene || {};
  const exo = scene.exo || {};
  const pose = configuration?.pose || {};
  return {
    // `url` is what the node calls it; `glb` is accepted as well, for a view
    // configuration file that names an asset the node does not serve.
    url: firstString(scene.url, scene.glb),
    hash: typeof scene.hash === "string" ? scene.hash : null,
    // What the export says the model is, when the node passed it on: the page
    // only reports a disagreement, because the joint addresses it applies come
    // from the asset itself.
    nq: typeof scene.nq === "number" ? scene.nq : null,
    mode: scene.placement === "ego" ? "ego" : "exo",
    armOriginWorld: triple(scene.arm_origin_world, DEFAULT_ARM_ORIGIN_WORLD),
    frameOffset: triple(
      pose.frame_offset ?? scene.frame_offset,
      DEFAULT_FRAME_OFFSET,
    ),
    neckPivotOffset: triple(pose.neck_pivot_offset, DEFAULT_NECK_PIVOT_OFFSET),
    distance: number(exo.distance, DEFAULT_EXO.distance),
    lateral: number(exo.lateral, DEFAULT_EXO.lateral),
    // The node may give the height either as the arm origin's own height or
    // as the table top's, which is what a person setting it up can measure.
    // The two differ by however far the site sits above the table in the sim.
    height: number(
      exo.height,
      number(exo.table_height, null) !== null
        ? exo.table_height +
            (number(scene.arm_origin_world?.[2], DEFAULT_ARM_ORIGIN_WORLD[2]) -
              number(scene.table_top_world_z, 1.005))
        : DEFAULT_EXO.height,
    ),
    yaw: number(exo.yaw, DEFAULT_EXO.yaw),
    objects: Array.isArray(scene.objects) ? scene.objects.map(String) : [],
  };
}

export async function floorReferenceSpaceType(session) {
  // Quest Browser exposes the features it actually granted. Asking it for an
  // ungranted optional local-floor space usually rejects, but on the real
  // Quest 3 it has also remained pending forever (1 of 3 measured starts),
  // which prevents three.js from ever attaching its XR renderer. Older
  // runtimes without enabledFeatures keep the defensive request-and-fallback.
  const granted = session.enabledFeatures;
  if (granted && !Array.from(granted).includes("local-floor")) {
    return "local";
  }
  try {
    await session.requestReferenceSpace("local-floor");
    return "local-floor";
  } catch {
    return "local";
  }
}

function number(value, fallback) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function firstString(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }
  return null;
}

function triple(value, fallback) {
  if (
    Array.isArray(value) &&
    value.length >= 3 &&
    value.every(Number.isFinite)
  ) {
    return [value[0], value[1], value[2]];
  }
  return fallback;
}

// Build the 3D view. Rejects when there is nothing to draw (no asset, a
// refused one, or no WebGL), which is ar.js's signal to fall back.
export async function createScene3d({
  configuration,
  tracks = {},
  hud = null,
  panel = false,
  onNotice = () => {},
} = {}) {
  const settings = readSceneConfiguration(configuration);
  if (!settings.url) {
    throw new Error("the node sent no scene asset");
  }
  const sessionMode = configuration?.session?.mode || "immersive-ar";
  const passthrough = sessionMode === "immersive-ar";

  const canvas = document.createElement("canvas");
  // A WebGL context lost is the one failure that turns the headset black with
  // nothing on the page to show for it, and this view builds a context per
  // connection, so it is also the one this page is most likely to cause
  // (openarm-quest-teleop patch 17).
  globalThis.__oqtTelemetry?.context(canvas, "scene3d");
  const renderer = new WebGLRenderer({
    canvas,
    antialias: true,
    // The compositor blends what is not drawn with the room's own image, so
    // in immersive-ar everything this does not cover has to stay transparent.
    alpha: true,
  });
  renderer.xr.enabled = true;
  renderer.setClearColor(new Color(0x0d1014), passthrough ? 0 : 1);

  const scene = new Scene();
  scene.background = null;
  // One group carries the whole sim world, so the placement is one matrix and
  // a re-anchor is one assignment.
  const world = new Group();
  world.matrixAutoUpdate = false;
  scene.add(world);

  const gltf = await loadAsset(settings);
  world.add(gltf.scene);
  const extras =
    gltf.parser?.json?.scenes?.[gltf.parser?.json?.scene || 0]?.extras || {};
  const kinematics = new Kinematics(gltf.scene);
  // The rest pose the asset was exported at, so the robot stands in a pose
  // the model can actually take while no state has arrived yet.
  if (Array.isArray(extras.qpos0)) {
    kinematics.apply(Float32Array.from(extras.qpos0));
  }
  // The node read `nq` out of the file it is serving. A different number here
  // means this page is drawing a different file -- a copy cached from an
  // earlier run -- and every joint value would land in the wrong place.
  if (settings.nq !== null && settings.nq !== kinematics.qposLength) {
    throw new Error(
      `the asset has ${kinematics.qposLength} joint values, the node's has ${settings.nq}`,
    );
  }
  globalThis.__oqtTelemetry?.info("scene", "3D scene built", {
    data: {
      url: settings.url,
      nq: kinematics.qposLength,
      panel: Boolean(panel),
      objects: settings.objects.length,
    },
  });

  // Light it from above and in front, the way the cell's own room is lit.
  // Ambient carries most of it: a single hard key on a grey robot over
  // passthrough reads as a cut-out.
  scene.add(new AmbientLight(0xffffff, 1.6));
  const key = new DirectionalLight(0xffffff, 2.2);
  key.position.set(1.2, 2.4, 1.0);
  scene.add(key);
  const fill = new DirectionalLight(0xbfd4ff, 0.8);
  fill.position.set(-1.5, 1.0, -1.2);
  scene.add(fill);

  // A floor, but only where there is no real one to see: in immersive-ar the
  // operator's own floor is right there and a drawn one would hide it.
  const ground = new Mesh(
    new PlaneGeometry(8, 8),
    new MeshStandardMaterial({ color: new Color(0x20242a), roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.name = "room_ground";
  ground.visible = !passthrough;
  scene.add(ground);

  // Where the operator's hands are, drawn as the gripper the node turns them
  // into. The clutch is a relative mapping, so after a re-anchor the gripper
  // is not where the hand is; these markers are the only honest answer to
  // "which way will the gripper be holding if I squeeze now".
  const ghosts = {
    right: handGhost(0x4ec9ff),
    left: handGhost(0xffc14e),
  };
  scene.add(ghosts.right, ghosts.left);

  const hudMesh = hud ? canvasPanel(hud.canvas, HUD_PANEL.width) : null;
  if (hudMesh) {
    hudMesh.name = "hud";
    hudMesh.visible = false;
    scene.add(hudMesh);
  }
  let hudRevision = -1;

  const cameraPanel =
    panel && tracks.right ? videoPanel(tracks.right, configuration) : null;
  const auxiliaryPanels = (configuration?.auxiliary_panels || [])
    .map((item) =>
      tracks[item.track]
        ? videoPanel(tracks[item.track], { ...configuration, panel: item })
        : null,
    )
    .filter(Boolean);
  const cameraPanels = [cameraPanel, ...auxiliaryPanels].filter(Boolean);
  for (const item of cameraPanels) {
    scene.add(item.mesh);
  }

  // A plain camera: three.js replaces it with the session's own pair while
  // presenting, and it is what the mirror render below uses out of session.
  const camera = new PerspectiveCamera(70, 1, 0.05, 40);
  scene.add(camera);

  const state = {
    anchor: null,
    floorY: 0,
    nudge: emptyNudge(),
    flags: {
      clutchRight: false,
      clutchLeft: false,
      recording: false,
      gateFault: false,
    },
    sequence: null,
    lastStateAt: null,
    startedAt: null,
    notice: "",
    frames: 0,
    reanchors: 0,
    frameMs: new Float32Array(FRAME_SAMPLES),
    frameAt: 0,
  };
  let pending = null;
  let previousSentAt = null;
  let previousFrameAt = null;
  let session = null;
  let renderSpace = null;
  let hooks = null;
  let capture = null;
  // Which reference space the scene is drawn in, settled when the session
  // starts: "local-floor" puts y = 0 on the operator's own floor, "local"
  // leaves the floor to be guessed from the head height at the anchor.
  let floorType = "local";

  function say(text) {
    if (state.notice === text) {
      return;
    }
    state.notice = text;
    if (hud) {
      hud.setNotice(text);
    }
    onNotice(text);
  }

  function reanchor(viewerPose) {
    const transform = viewerPose.transform;
    state.floorY =
      floorType === "local-floor"
        ? 0
        : transform.position.y - DEFAULT_EYE_HEIGHT;
    state.anchor = {
      position: {
        x: transform.position.x,
        y: transform.position.y,
        z: transform.position.z,
      },
      yaw: headingYaw(transform.orientation),
      pivot: neckPivot(
        transform.position,
        transform.orientation,
        settings.neckPivotOffset,
      ),
    };
    state.reanchors += 1;
    place();
    for (const item of cameraPanels.filter((candidate) => candidate.lock === "room")) {
      // The room-locked panel hangs where the operator was looking when the
      // scene anchored, so the two never disagree about what "in front" is.
      const [side, up] = item.center;
      item.mesh.position.set(
        state.anchor.position.x -
          Math.sin(state.anchor.yaw) * item.distance +
          Math.cos(state.anchor.yaw) * side,
        state.anchor.position.y + up,
        state.anchor.position.z -
          Math.cos(state.anchor.yaw) * item.distance -
          Math.sin(state.anchor.yaw) * side,
      );
      item.mesh.quaternion.setFromAxisAngle(UP, state.anchor.yaw);
    }
  }

  function place() {
    world.matrix.copy(
      placementMatrix({
        mode: settings.mode,
        armOriginWorld: settings.armOriginWorld,
        frameOffset: settings.frameOffset,
        anchor: state.anchor,
        distance: settings.distance,
        lateral: settings.lateral,
        height: state.floorY + settings.height,
        yaw: settings.yaw,
        nudge: state.nudge,
      }),
    );
    world.matrixWorldNeedsUpdate = true;
  }

  // One message off the "state" channel. Kept rather than applied, because
  // the channel can deliver several between two frames and only the newest
  // one is worth drawing.
  function setState(data) {
    const message = decodeSceneState(data);
    if (!message) {
      return false;
    }
    if (!isNewer(message.sequence, state.sequence)) {
      // An unordered channel delivers the occasional message late; drawing it
      // would step the robot backwards.
      return false;
    }
    state.sequence = message.sequence;
    state.lastStateAt = performance.now();
    state.flags = message.flags;
    pending = message;
    return true;
  }

  function applyPending() {
    if (!pending) {
      return;
    }
    const message = pending;
    pending = null;
    if (message.qpos) {
      kinematics.apply(message.qpos);
    }
    if (message.objects && settings.objects.length > 0) {
      for (let i = 0; i < settings.objects.length; i++) {
        const at = i * 7;
        if (at + 7 > message.objects.length) {
          break;
        }
        kinematics.setBodyPose(
          settings.objects[i],
          message.objects.subarray(at, at + 7),
        );
      }
    }
    if (message.bodies) {
      for (const [name, pose] of message.bodies) {
        kinematics.setBodyPose(name, pose);
      }
    }
  }

  function updateGhosts(frame) {
    for (const handedness of ["right", "left"]) {
      ghosts[handedness].visible = false;
    }
    if (!session) {
      return;
    }
    for (const source of session.inputSources) {
      const ghost = ghosts[source.handedness];
      if (!ghost || source.hand || !source.gamepad) {
        continue;
      }
      // The target ray space is the aim pose the node maps onto the gripper.
      const pose = frame.getPose(source.targetRaySpace, renderSpace);
      if (!pose) {
        continue;
      }
      ghost.visible = true;
      ghost.position.set(
        pose.transform.position.x,
        pose.transform.position.y,
        pose.transform.position.z,
      );
      scratchQuaternion.set(
        pose.transform.orientation.x,
        pose.transform.orientation.y,
        pose.transform.orientation.z,
        pose.transform.orientation.w,
      );
      ghost.quaternion.copy(scratchQuaternion).multiply(CONTROLLER_TO_EE);
      // Solid while that arm is following the hand, faint while it is not,
      // so the marker never claims to be driving something it is not.
      const engaged =
        source.handedness === "right"
          ? state.flags.clutchRight
          : state.flags.clutchLeft;
      ghost.material.opacity = engaged ? 0.85 : 0.3;
    }
  }

  function updateNudge(seconds) {
    if (!session) {
      return;
    }
    const axes = { forward: 0, lateral: 0, height: 0, yaw: 0 };
    for (const source of session.inputSources) {
      if (source.hand || !source.gamepad || source.handedness === "none") {
        continue;
      }
      const sticks = source.gamepad.axes;
      // The xr-standard mapping reserves the first axis pair for a touchpad
      // and the second for the thumbstick, which is the rule main.py follows
      // when it publishes them.
      const x = sticks.length >= 4 ? sticks[2] : sticks[0];
      const y = sticks.length >= 4 ? sticks[3] : sticks[1];
      // A thumbstick only nudges while its own arm is not following it:
      // moving the robot under an engaged clutch would move the target.
      if (source.handedness === "left" && !state.flags.clutchLeft) {
        axes.lateral = x;
        axes.forward = -y;
      } else if (source.handedness === "right" && !state.flags.clutchRight) {
        axes.yaw = -x;
        axes.height = -y;
      }
    }
    if (advanceNudge(state.nudge, axes, seconds)) {
      place();
    }
  }

  function updateHud(viewerPose) {
    if (!hudMesh || !hud) {
      return;
    }
    const revision = hud.paint();
    if (revision !== hudRevision) {
      hudMesh.material.map.needsUpdate = true;
      hudRevision = revision;
    }
    hudMesh.visible = !hud.empty;
    if (!hudMesh.visible) {
      return;
    }
    // Head-locked, like the quad the 2D views draw: the operator has to be
    // able to read it wherever they are looking.
    placeInView(hudMesh, viewerPose, HUD_PANEL.distance, HUD_PANEL.center);
  }

  function onXrFrame(time, frame) {
    if (!frame) {
      return;
    }
    const now = performance.now();
    const seconds =
      previousFrameAt === null ? 0 : (now - previousFrameAt) / 1000;
    previousFrameAt = now;
    state.frames += 1;

    // The node's frame first: its latency is what the robot follows, and
    // nothing drawn here is worth delaying it for.
    const sent = hooks.onFrame(time, frame) === true;
    const viewerPose = frame.getViewerPose(renderSpace);

    if (viewerPose) {
      if (state.anchor === null) {
        reanchor(viewerPose);
      } else if (
        sent &&
        previousSentAt !== null &&
        now - previousSentAt > REANCHOR_GAP_MS
      ) {
        // The same gap that makes the node latch a new pivot.
        reanchor(viewerPose);
      }
    }
    if (sent) {
      previousSentAt = now;
    }

    updateNudge(seconds);
    applyPending();
    updateGhosts(frame);
    if (viewerPose) {
      updateHud(viewerPose);
      for (const item of cameraPanels) item.update(viewerPose);
    }
    checkState(now);

    renderer.render(scene, camera);
    if (state.frames === 1) {
      const gl = renderer.getContext();
      globalThis.__oqtTelemetry?.info("scene", "first 3D frame rendered", {
        data: {
          floor: floorType,
          anchor: state.anchor,
          panel: cameraPanel
            ? {
                present: true,
                visible: cameraPanel.mesh.visible,
                position: cameraPanel.mesh.position.toArray(),
              }
            : { present: false },
          world: world.matrix.elements,
          gl_error: gl.getError(),
        },
      });
    }
    // What this frame cost the page's own thread, which is the number that
    // decides whether the operator sees 72 Hz. Kept in a ring of the last
    // FRAME_SAMPLES frames: enough for a median and a p95 over eight
    // seconds, and nothing to grow.
    state.frameMs[state.frameAt % FRAME_SAMPLES] = performance.now() - now;
    state.frameAt += 1;
    if (capture) {
      const request = capture;
      capture = null;
      request.resolve(renderViews(viewerPose, request.size));
    }
  }

  function checkState(now) {
    if (state.lastStateAt === null) {
      if (
        state.startedAt !== null &&
        now - state.startedAt > NO_STATE_AFTER_MS
      ) {
        say(NOTICES.noState);
      }
      return;
    }
    if (now - state.lastStateAt > STALE_AFTER_MS) {
      say(NOTICES.staleState);
    } else if (
      state.notice === NOTICES.staleState ||
      state.notice === NOTICES.noState
    ) {
      say("");
    }
  }

  // Render what each eye saw into an offscreen target and read it back, so a
  // test outside the headset can look at the same pixels the compositor got.
  // Never called by the page itself.
  function renderViews(viewerPose, size) {
    if (!viewerPose) {
      return [];
    }
    const width = size?.width || 640;
    const height = size?.height || 640;
    const target = new WebGLRenderTarget(width, height);
    const mirror = new Camera();
    mirror.matrixAutoUpdate = false;
    const wasPresenting = renderer.xr.enabled;
    renderer.xr.enabled = false;
    const out = [];
    for (const view of viewerPose.views) {
      mirror.projectionMatrix.fromArray(view.projectionMatrix);
      mirror.projectionMatrixInverse.copy(mirror.projectionMatrix).invert();
      mirror.matrix.fromArray(view.transform.matrix);
      mirror.matrixWorld.fromArray(view.transform.matrix);
      renderer.setRenderTarget(target);
      renderer.clear();
      renderer.render(scene, mirror);
      const pixels = new Uint8Array(width * height * 4);
      renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
      out.push({ eye: view.eye, width, height, pixels: Array.from(pixels) });
    }
    renderer.setRenderTarget(null);
    renderer.xr.enabled = wasPresenting;
    target.dispose();
    return out;
  }

  return {
    settings,
    stats: assetStats(gltf.scene, kinematics),
    state,
    setState,
    // ar.js calls this once the session is running and the reference spaces
    // it needs for the node's frames are in hand.
    async start(startedSession, sessionHooks) {
      session = startedSession;
      hooks = sessionHooks;
      state.startedAt = performance.now();
      // The scene is drawn in a floor-relative space when the runtime has
      // one, so "the table is 0.95 m up" means the operator's own floor.
      // Without one the floor is guessed from the head height, once, at the
      // anchor, which is why the height is configurable.
      floorType = await floorReferenceSpaceType(session);
      renderer.xr.setReferenceSpaceType(floorType);
      // Maximum foveation: on a Quest this is what the compositor renders the
      // edges of the eye buffers at a lower resolution for, and the operator
      // is looking at the middle.
      renderer.xr.setFoveation(1.0);
      await renderer.xr.setSession(session);
      renderSpace = renderer.xr.getReferenceSpace();
      for (const item of cameraPanels) item.start();
      renderer.setAnimationLoop(onXrFrame);
      globalThis.__oqtTelemetry?.info("scene", "3D XR renderer started", {
        data: {
          floor: floorType,
          foveation: renderer.xr.getFoveation(),
          panel: Boolean(cameraPanel),
          auxiliary_panels: auxiliaryPanels.length,
          base_layer: Boolean(session.renderState.baseLayer),
          layers: session.renderState.layers?.length || 0,
        },
      });
    },
    stop() {
      renderer.setAnimationLoop(null);
      session = null;
      hooks = null;
      renderSpace = null;
      previousSentAt = null;
      state.anchor = null;
      state.startedAt = null;
      state.lastStateAt = null;
      state.sequence = null;
    },
    close() {
      renderer.setAnimationLoop(null);
      for (const item of cameraPanels) item.close();
      renderer.dispose();
      // A browser keeps only a handful of WebGL contexts alive, and this page
      // builds a view per connection -- one per session. Without this, an
      // operator who started and ended enough sessions would eventually find
      // the 3D view falling back because no context was left to give it.
      //
      // Said first, so the loss this is about to cause is recorded as the
      // housekeeping it is. Otherwise every clean `--view scene` session end
      // leaves an error in the headset's log, `oqt run` ends every good run
      // by reporting one, and the operator learns to ignore the level that
      // matters (openarm-quest-teleop patch 17).
      globalThis.__oqtTelemetry?.contextClosing(canvas);
      renderer.forceContextLoss();
      session = null;
    },
    // What one frame of this view costs the page's thread: the median and
    // the p95 are what a 72 Hz headset lives or dies by, so they are worth
    // being able to read from a running session rather than guessing.
    timings() {
      const count = Math.min(state.frameAt, FRAME_SAMPLES);
      if (count === 0) {
        return { count: 0 };
      }
      const sorted = Array.from(state.frameMs.subarray(0, count)).sort(
        (a, b) => a - b,
      );
      const at = (quantile) =>
        sorted[Math.min(count - 1, Math.floor(quantile * count))];
      return {
        count,
        median: at(0.5),
        p95: at(0.95),
        max: sorted[count - 1],
        mean: sorted.reduce((a, b) => a + b, 0) / count,
      };
    },
    // Where a body of the sim ended up in the operator's room, for a test
    // that cannot see the headset's display.
    bodyPosition(name) {
      const entry = kinematics.byName.get(name);
      if (!entry) {
        return null;
      }
      scene.updateMatrixWorld(true);
      const matrix = entry.object.matrixWorld.elements;
      return [matrix[12], matrix[13], matrix[14]];
    },
    // Test seam: ask for the next frame's eye images. Resolves with one
    // {eye, width, height, pixels} per view.
    captureEyes(size) {
      return new Promise((resolve) => {
        capture = { size, resolve };
      });
    },
    get placement() {
      return world.matrix.elements.slice();
    },
    get floorSpace() {
      return floorType;
    },
    renderer,
    scene,
    world,
  };
}

const UP = new Vector3(0, 1, 0);
const scratchQuaternion = new Quaternion();
const scratchPosition = new Vector3();
const scratchOffset = new Vector3();

async function loadAsset(settings) {
  // The hash the node computed goes in the query string, so a headset that
  // cached a previous run's asset at the same path fetches this one instead
  // of drawing the wrong robot.
  const url = settings.hash
    ? `${settings.url}${settings.url.includes("?") ? "&" : "?"}h=${encodeURIComponent(settings.hash)}`
    : settings.url;
  let gltf;
  try {
    gltf = await new GLTFLoader().loadAsync(url);
  } catch (error) {
    // A scene that would not load is a headset showing passthrough and a
    // notice, and from the node's end it looks like a page that simply never
    // drew anything. Say which file, and what the fetch made of it (patch
    // 17): a 404 from a run whose scene.glb was cleaned up and a glTF the
    // parser choked on need different answers. Rethrown unchanged.
    globalThis.__oqtTelemetry?.error(
      "asset",
      `the scene asset would not load: ${error && error.message}`,
      {
        stack: error && error.stack,
        data: {
          url,
          status: error && error.target ? error.target.status : undefined,
          nq: settings.nq,
        },
      },
    );
    throw error;
  }
  const scenes = gltf.parser?.json?.scenes || [];
  const extras = scenes[gltf.parser?.json?.scene || 0]?.extras || {};
  // An asset that says which build it is and disagrees with the node is the
  // one case where drawing nothing is right: it would be a different robot.
  if (settings.hash && extras.hash && extras.hash !== settings.hash) {
    throw new Error(
      `the asset is ${extras.hash}, the node asked for ${settings.hash}`,
    );
  }
  return gltf;
}

function assetStats(root, kinematics) {
  let triangles = 0;
  let meshes = 0;
  let nodes = 0;
  root.traverse((object) => {
    nodes += 1;
    if (!object.isMesh) {
      return;
    }
    meshes += 1;
    const index = object.geometry.index;
    triangles += index
      ? index.count / 3
      : object.geometry.attributes.position.count / 3;
  });
  return { triangles, meshes, nodes, movingBodies: kinematics.bodies.length };
}

function handGhost(color) {
  const ghost = new Mesh(
    new BoxGeometry(GHOST.width, GHOST.height, GHOST.length),
    new MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.3,
      depthWrite: false,
    }),
  );
  ghost.visible = false;
  return ghost;
}

function canvasPanel(canvas, width) {
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.minFilter = LinearFilter;
  const height = (width * canvas.height) / canvas.width;
  const mesh = new Mesh(
    new PlaneGeometry(width, height),
    new MeshBasicMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    }),
  );
  // Over the robot rather than inside it: the HUD is the node talking to the
  // operator, not part of the scene.
  mesh.renderOrder = 10;
  return mesh;
}

function videoPanel(track, configuration) {
  const settings = configuration?.panel || {};
  const width = number(settings.width, 1.5);
  const distance = number(settings.distance, 1.3);
  const video = createVideoElement(track);
  const texture = new VideoTexture(video);
  texture.colorSpace = SRGBColorSpace;
  const mesh = new Mesh(
    new PlaneGeometry(width, (width * 9) / 16),
    new MeshBasicMaterial({ map: texture }),
  );
  mesh.name = "camera_panel";
  mesh.visible = false;
  let shape = 0;
  return {
    mesh,
    lock: settings.lock === "head" ? "head" : "room",
    distance,
    center: Array.isArray(settings.center) ? settings.center : [0, 0],
    start() {
      // This runs after the operator pressed Start. Quest Browser can reject
      // the eager play() done while the page is connecting even though the
      // track is muted; retry inside the session's user-initiated path and
      // make a second failure visible in quest.log.
      video.play().catch((error) => {
        globalThis.__oqtTelemetry?.warn(
          "video",
          `3D camera panel would not play: ${error.message}`,
          { stack: error.stack },
        );
      });
    },
    update(viewerPose) {
      if (video.readyState < 2 || video.videoWidth === 0) {
        // No frame yet: an empty panel would hide the robot behind it.
        return;
      }
      mesh.visible = true;
      const aspect = video.videoWidth / video.videoHeight;
      if (aspect !== shape) {
        // The height follows the image so it is never stretched.
        shape = aspect;
        mesh.geometry.dispose();
        mesh.geometry = new PlaneGeometry(width, width / aspect);
      }
      if (settings.lock === "head") {
        const center = Array.isArray(settings.center) ? settings.center : [0, 0];
        placeInView(mesh, viewerPose, distance, center[1] || 0, center[0] || 0);
      }
    },
    close() {
      video.pause();
      video.srcObject = null;
    },
  };
}

// Hang a panel in front of the head, `up` metres above the middle of the
// view, the way the 2D HUD's shader places its quad.
function placeInView(mesh, viewerPose, distance, up, side = 0) {
  const transform = viewerPose.transform;
  scratchQuaternion.set(
    transform.orientation.x,
    transform.orientation.y,
    transform.orientation.z,
    transform.orientation.w,
  );
  scratchPosition.set(
    transform.position.x,
    transform.position.y,
    transform.position.z,
  );
  mesh.quaternion.copy(scratchQuaternion);
  mesh.position
    .copy(
      scratchOffset.set(side, up, -distance).applyQuaternion(scratchQuaternion),
    )
    .add(scratchPosition);
}
