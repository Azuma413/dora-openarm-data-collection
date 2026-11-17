// openarm-quest-teleop addition to the vendored dora-openarm-webxr tests
// (Apache-2.0, like the rest of this directory).
//
// The parts of the 3D page that can be checked without a headset: the
// kinematics against MuJoCo's own, the placement rule against the pose maths
// in main.py that it inverts, the thumbstick nudge, and the state codec.
//
//   node third_party/dora-openarm-webxr/tests/web/scene_units.test.mjs
//
// Needs Node.js 20+ and nothing else -- no browser, no dora, no MuJoCo: the
// ground truth is fixture/trajectory.json, written by make_fixture.py from
// MuJoCo's own mj_kinematics.

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.join(HERE, "../../src/dora_openarm_webxr/static");

const { Matrix4, Quaternion, Vector3 } = await import(
  `${STATIC}/three/three.module.min.js`
);
const { GLTFLoader } = await import(`${STATIC}/three/GLTFLoader.js`);
const { Kinematics } = await import(`${STATIC}/mjkin.js`);
const align = await import(`${STATIC}/align.js`);
const { floorReferenceSpaceType, readSceneConfiguration } = await import(
  `${STATIC}/scene3d.js`
);
const { FLAG, decodeSceneState, encodeBinary, isNewer } = await import(
  `${STATIC}/scene-state.js`
);

const results = [];

function section(name, body) {
  try {
    const detail = body();
    results.push({ name, ok: true });
    console.log(`PASS ${name}${detail ? ` -- ${detail}` : ""}`);
  } catch (error) {
    results.push({ name, ok: false });
    console.log(`FAIL ${name} -- ${error.message}`);
  }
}

async function asyncSection(name, body) {
  try {
    const detail = await body();
    results.push({ name, ok: true });
    console.log(`PASS ${name}${detail ? ` -- ${detail}` : ""}`);
  } catch (error) {
    results.push({ name, ok: false });
    console.log(`FAIL ${name} -- ${error.message}`);
  }
}

// main.py's own mapping, written out here so the placement is checked against
// the thing it inverts rather than against itself:
//   p_arm = R (p_local - pivot) + frame_offset,  R = [[0,0,-1],[-1,0,0],[0,1,0]]
function robotFromXr(local, pivot, frameOffset) {
  const v = [local.x - pivot.x, local.y - pivot.y, local.z - pivot.z];
  return [
    -v[2] + frameOffset[0],
    -v[0] + frameOffset[1],
    v[1] + frameOffset[2],
  ];
}

const trajectory = JSON.parse(
  fs.readFileSync(path.join(HERE, "fixture/trajectory.json"), "utf8"),
);

async function loadFixture() {
  const bytes = fs.readFileSync(path.join(HERE, "fixture/arm.glb"));
  const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  const gltf = await new Promise((resolve, reject) =>
    new GLTFLoader().parse(buffer, "", resolve, reject),
  );
  return gltf;
}

const gltf = await loadFixture();
const kinematics = new Kinematics(gltf.scene);

section("the page's kinematics agree with MuJoCo's", () => {
  let worstPosition = 0;
  let worstRotation = 0;
  const position = new Vector3();
  const quaternion = new Quaternion();
  const scale = new Vector3();
  for (const frame of trajectory.frames) {
    // Sent as float32, which is what the state channel carries.
    assert.equal(kinematics.apply(Float32Array.from(frame.qpos)), true);
    gltf.scene.updateMatrixWorld(true);
    for (const name of trajectory.tracked) {
      const node = gltf.scene.getObjectByName(name);
      node.matrixWorld.decompose(position, quaternion, scale);
      const truth = frame.bodies[name];
      worstPosition = Math.max(
        worstPosition,
        position.distanceTo(new Vector3(truth[0], truth[1], truth[2])),
      );
      // MuJoCo writes quaternions w first, and a quaternion and its negative
      // are the same rotation. Compared component by component rather than
      // through Quaternion.angleTo, whose acos turns a 1e-9 dot product error
      // into a 5e-5 rad "angle" and would measure the test, not the page.
      worstRotation = Math.max(
        worstRotation,
        quaternionDistance(quaternion, [
          truth[4],
          truth[5],
          truth[6],
          truth[3],
        ]),
      );
    }
  }
  assert.ok(worstPosition < 1e-6, `worst position error ${worstPosition} m`);
  assert.ok(worstRotation < 1e-6, `worst rotation error ${worstRotation}`);
  return `${trajectory.frames.length} frames x ${trajectory.tracked.length} bodies, worst ${(worstPosition * 1e9).toFixed(1)} nm / ${(worstRotation * 1e9).toFixed(1)} nano-quaternion`;
});

// How far apart two rotations are, as the larger of |q - e| and |q + e|'s
// complement: a quaternion and its negative are the same rotation.
function quaternionDistance(quaternion, expected) {
  let same = 0;
  let flipped = 0;
  const values = [quaternion.x, quaternion.y, quaternion.z, quaternion.w];
  for (let i = 0; i < 4; i++) {
    same = Math.max(same, Math.abs(values[i] - expected[i]));
    flipped = Math.max(flipped, Math.abs(values[i] + expected[i]));
  }
  return Math.min(same, flipped);
}

section("a qpos too short for the model moves nothing", () => {
  const frame = trajectory.frames[5];
  assert.equal(kinematics.apply(Float32Array.from(frame.qpos)), true);
  gltf.scene.updateMatrixWorld(true);
  const wrist = gltf.scene.getObjectByName("wrist");
  const before = wrist.matrixWorld.elements.slice();
  assert.equal(
    kinematics.apply(Float32Array.from(frame.qpos.slice(0, 5))),
    false,
  );
  assert.equal(kinematics.apply(null), false);
  gltf.scene.updateMatrixWorld(true);
  assert.deepEqual(wrist.matrixWorld.elements.slice(), before);
  return `refused ${5} of ${trajectory.nq} values`;
});

section("a body pose from the sim puts a free body where it says", () => {
  const pose = [0.31, -0.22, 0.75, 0.5, 0.5, 0.5, 0.5];
  assert.equal(kinematics.setBodyPose("cube", pose), true);
  gltf.scene.updateMatrixWorld(true);
  const cube = gltf.scene.getObjectByName("cube");
  const position = new Vector3();
  const quaternion = new Quaternion();
  cube.matrixWorld.decompose(position, quaternion, new Vector3());
  assert.ok(position.distanceTo(new Vector3(0.31, -0.22, 0.75)) < 1e-6);
  assert.ok(quaternionDistance(quaternion, [0.5, 0.5, 0.5, 0.5]) < 1e-6);
  assert.equal(kinematics.setBodyPose("no_such_body", pose), false);
  assert.equal(kinematics.setBodyPose("cube", [1, 2, 3]), false);
  return "cube placed, unknown body and short pose refused";
});

section("exo puts arm_origin where the rule says", () => {
  const armOrigin = [0.185, 0, 1.34];
  for (const [yaw, distance, height, lateral] of [
    [0, 1.15, 1.285, 0],
    [Math.PI / 2, 0.8, 1.0, 0.25],
    [-2.3, 1.5, 1.4, -0.4],
  ]) {
    const anchor = { position: { x: 0.4, y: 1.62, z: -0.7 }, yaw };
    const matrix = align.placementMatrix({
      mode: "exo",
      armOriginWorld: armOrigin,
      anchor,
      distance,
      height,
      lateral,
    });
    const placed = new Vector3(...armOrigin).applyMatrix4(matrix);
    const want = new Vector3(
      anchor.position.x + Math.cos(yaw) * lateral - Math.sin(yaw) * distance,
      height,
      anchor.position.z - Math.sin(yaw) * lateral - Math.cos(yaw) * distance,
    );
    assert.ok(
      placed.distanceTo(want) < 1e-6,
      `arm_origin at ${placed.toArray()} not ${want.toArray()}`,
    );
    // ... and the robot faces the way the operator does, which is the only
    // heading under which a hand pushed forward pushes the gripper forward.
    const forward = new Vector3(1, 0, 0)
      .applyMatrix4(new Matrix4().extractRotation(matrix))
      .normalize();
    const facing = new Vector3(-Math.sin(yaw), 0, -Math.cos(yaw));
    assert.ok(
      forward.distanceTo(facing) < 1e-6,
      `robot faces ${forward.toArray()} not ${facing.toArray()}`,
    );
  }
  return "3 anchors, arm_origin and heading exact";
});

section("ego is the exact inverse of main.py's pose mapping", () => {
  const armOrigin = [0.185, 0, 1.34];
  const frameOffset = [-0.1, 0.0, 0.23];
  const pivot = new Vector3(0.12, 1.42, -0.33);
  const matrix = align.placementMatrix({
    mode: "ego",
    armOriginWorld: armOrigin,
    frameOffset,
    anchor: { pivot },
  });
  let worst = 0;
  for (const point of [armOrigin, [0.4, 0.2, 1.1], [-0.2, -0.35, 1.6]]) {
    const local = new Vector3(...point).applyMatrix4(matrix);
    // What the node would publish for a hand held at that spot in the room.
    const arm = robotFromXr(local, pivot, frameOffset);
    // What the sim itself calls that point, in the arm_origin frame.
    const truth = [
      point[0] - armOrigin[0],
      point[1] - armOrigin[1],
      point[2] - armOrigin[2],
    ];
    for (let i = 0; i < 3; i++) {
      worst = Math.max(worst, Math.abs(arm[i] - truth[i]));
    }
  }
  assert.ok(worst < 1e-6, `worst disagreement ${worst} m`);
  return `3 points, worst ${(worst * 1e9).toFixed(1)} nm`;
});

section("the heading comes from where the operator faces", () => {
  const cases = [
    [0, 0, 0, 1, 0],
    [0, Math.sin(Math.PI / 4), 0, Math.cos(Math.PI / 4), Math.PI / 2],
    [0, Math.sin(-1.1), 0, Math.cos(-1.1), -2.2],
  ];
  for (const [x, y, z, w, want] of cases) {
    const got = align.headingYaw({ x, y, z, w });
    assert.ok(
      Math.abs(Math.atan2(Math.sin(got - want), Math.cos(got - want))) < 1e-6,
      `yaw ${got} not ${want}`,
    );
  }
  // Looking straight up, with the head also turned: the forward axis has
  // nothing left in the floor plane, and the up axis has to stand in.
  const turn = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), 0.7);
  const pitch = new Quaternion().setFromAxisAngle(
    new Vector3(1, 0, 0),
    Math.PI / 2,
  );
  const looking = turn.clone().multiply(pitch);
  const up = align.headingYaw(looking);
  assert.ok(Math.abs(up - 0.7) < 1e-6, `yaw looking up ${up} not 0.7`);
  return "4 orientations including straight up";
});

section("the thumbstick nudge has a deadzone, a rate and a limit", () => {
  const nudge = align.emptyNudge();
  // Inside the deadzone: nothing moves, however long it is held.
  assert.equal(align.advanceNudge(nudge, { forward: 0.19 }, 1.0), false);
  assert.deepEqual(nudge, align.emptyNudge());
  // Fully over moves at the rate.
  assert.equal(align.advanceNudge(nudge, { forward: 1 }, 0.1), true);
  assert.ok(
    Math.abs(nudge.forward - align.NUDGE.rate.forward * 0.1) < 1e-9,
    `moved ${nudge.forward} m in 0.1 s`,
  );
  // Half over moves less than half, because the deadzone is taken out.
  const before = nudge.forward;
  align.advanceNudge(nudge, { forward: 0.5 }, 0.1);
  const half = nudge.forward - before;
  assert.ok(half > 0 && half < align.NUDGE.rate.forward * 0.05);
  // Held over, it stops at the limit rather than walking the robot away.
  for (let i = 0; i < 200; i++) {
    align.advanceNudge(
      nudge,
      { forward: 1, lateral: -1, yaw: 1, height: 1 },
      0.1,
    );
  }
  assert.equal(nudge.forward, align.NUDGE.limit.forward);
  assert.equal(nudge.lateral, -align.NUDGE.limit.lateral);
  assert.equal(nudge.yaw, align.NUDGE.limit.yaw);
  assert.equal(nudge.height, align.NUDGE.limit.height);
  // A frame that took a long time (a stall) does not teleport the scene.
  const jump = align.emptyNudge();
  align.advanceNudge(jump, { forward: 1 }, 30);
  assert.ok(jump.forward <= align.NUDGE.rate.forward * 0.25 + 1e-9);
  return `limits ${align.NUDGE.limit.forward} m / ${align.NUDGE.limit.yaw.toFixed(2)} rad`;
});

section("the nudge moves the placement by what it says", () => {
  const armOrigin = [0.185, 0, 1.34];
  const anchor = { position: { x: 0, y: 1.6, z: 0 }, yaw: Math.PI / 2 };
  const options = {
    mode: "exo",
    armOriginWorld: armOrigin,
    anchor,
    distance: 1,
    height: 1.2,
  };
  const plain = new Vector3(...armOrigin).applyMatrix4(
    align.placementMatrix(options),
  );
  const nudged = new Vector3(...armOrigin).applyMatrix4(
    align.placementMatrix({
      ...options,
      nudge: { forward: 0.2, lateral: 0.1, height: 0.05, yaw: 0 },
    }),
  );
  // Facing +x (yaw 90 degrees): forward is -x... in WebXR axes, forward at
  // yaw 90 degrees is (-1, 0, 0) and right is (0, 0, -1).
  const moved = nudged.clone().sub(plain);
  assert.ok(Math.abs(moved.x - -0.2) < 1e-6, `forward moved x by ${moved.x}`);
  assert.ok(Math.abs(moved.y - 0.05) < 1e-6, `height moved y by ${moved.y}`);
  assert.ok(Math.abs(moved.z - -0.1) < 1e-6, `lateral moved z by ${moved.z}`);
  return "forward, lateral and height in the anchor's own axes";
});

section("the node's scene configuration is read, with defaults", () => {
  const full = readSceneConfiguration({
    scene: {
      // `url`, `hash` and `nq` are the node's own (main.py _configure_scene);
      // the rest comes from the view configuration file.
      url: "/scene/scene.glb",
      hash: "abc123",
      nq: 191,
      placement: "ego",
      arm_origin_world: [0.185, 0, 1.34],
      table_top_world_z: 1.005,
      exo: { distance: 1.3, table_height: 0.8, yaw: 0.2 },
      objects: ["compress", "liner"],
    },
    pose: { frame_offset: [-0.1, 0, 0.23] },
  });
  assert.equal(full.url, "/scene/scene.glb");
  assert.equal(full.hash, "abc123");
  assert.equal(full.nq, 191);
  assert.equal(full.mode, "ego");
  assert.deepEqual(full.frameOffset, [-0.1, 0, 0.23]);
  assert.equal(full.distance, 1.3);
  // A table height is turned into the arm origin's own height, because that
  // is the point the placement rule puts down.
  assert.ok(
    Math.abs(full.height - (0.8 + 1.34 - 1.005)) < 1e-9,
    `height ${full.height}`,
  );
  assert.deepEqual(full.objects, ["compress", "liner"]);

  // A node that says nothing but the asset still gets a sensible scene, and
  // one that says nothing at all gets no URL, which is what makes ar.js fall
  // back rather than draw a robot in the wrong place.
  const bare = readSceneConfiguration({ scene: { url: "s.glb" } });
  assert.equal(bare.mode, "exo");
  assert.equal(bare.nq, null);
  assert.deepEqual(bare.armOriginWorld, align.DEFAULT_ARM_ORIGIN_WORLD);
  assert.equal(bare.height, align.DEFAULT_EXO.height);
  assert.deepEqual(bare.objects, []);
  assert.equal(readSceneConfiguration({}).url, null);
  assert.equal(readSceneConfiguration(undefined).url, null);
  // A view configuration file may name the asset itself, as `glb`.
  assert.equal(
    readSceneConfiguration({ scene: { glb: "own.glb" } }).url,
    "own.glb",
  );
  // Nonsense where a number belongs is ignored rather than drawn.
  const bad = readSceneConfiguration({
    scene: { url: "s.glb", exo: { distance: "far" }, arm_origin_world: [1, 2] },
  });
  assert.equal(bad.distance, align.DEFAULT_EXO.distance);
  assert.deepEqual(bad.armOriginWorld, align.DEFAULT_ARM_ORIGIN_WORLD);
  return "full, bare, empty and malformed configurations";
});

section("a state message decodes from JSON and from bytes alike", () => {
  const qpos = [0.1, -0.2, 0.3, 0.4];
  const objects = [0.5, 0.6, 0.7, 1, 0, 0, 0];
  const flags = FLAG.clutchRight | FLAG.recording;
  // The JSON is what oqt.sim.scene_state.encode_json writes, `v` and all.
  const json = decodeSceneState(
    JSON.stringify({
      type: "state",
      v: 1,
      seq: 7,
      t: 1.5,
      flags,
      qpos,
      obj: objects,
    }),
  );
  const binary = decodeSceneState(
    encodeBinary({ sequence: 7, simTime: 1.5, qpos, objects, flags }),
  );
  for (const message of [json, binary]) {
    assert.equal(message.sequence, 7);
    assert.ok(Math.abs(message.simTime - 1.5) < 1e-9);
    assert.equal(message.flags.clutchRight, true);
    assert.equal(message.flags.clutchLeft, false);
    assert.equal(message.flags.recording, true);
    assert.equal(message.flags.gateFault, false);
    assert.equal(message.qpos.length, 4);
    assert.ok(Math.abs(message.qpos[2] - 0.3) < 1e-6);
    assert.equal(message.objects.length, 7);
  }
  // A Uint8Array view, which is what some transports hand over.
  const bytes = new Uint8Array(encodeBinary({ sequence: 9, qpos }));
  assert.equal(decodeSceneState(bytes).sequence, 9);
  return "JSON, ArrayBuffer and a typed-array view";
});

section("a state message the page cannot trust is dropped", () => {
  assert.equal(decodeSceneState("not json"), null);
  assert.equal(
    decodeSceneState(JSON.stringify({ type: "hud", lines: [] })),
    null,
  );
  assert.equal(decodeSceneState(JSON.stringify({ hello: "world" })), null);
  // One unusable value poisons the message: half a qpos draws an arm in a
  // pose the robot is not in.
  assert.equal(
    decodeSceneState(JSON.stringify({ type: "state", qpos: [1, null, 3] })),
    null,
  );
  assert.equal(
    decodeSceneState(JSON.stringify({ type: "state", qpos: [1, "x", 3] })),
    null,
  );
  // Bytes that are not ours, and a truncated message.
  assert.equal(decodeSceneState(new ArrayBuffer(8)), null);
  const short = encodeBinary({ qpos: [1, 2, 3] }).slice(0, 28);
  assert.equal(decodeSceneState(short), null);
  // A node that sends no type, or a version the page has not seen, is still
  // read: those are exactly the fields an older or newer node changes.
  const untyped = decodeSceneState(JSON.stringify({ seq: 2, qpos: [1] }));
  assert.equal(untyped.sequence, 2);
  const future = decodeSceneState(
    JSON.stringify({ type: "state", v: 9, seq: 3, qpos: [1] }),
  );
  assert.equal(future.version, 9);
  return "garbage dropped, missing version tolerated";
});

section("a state message delivered late is not drawn", () => {
  assert.equal(isNewer(1, null), true);
  assert.equal(isNewer(5, 4), true);
  assert.equal(isNewer(4, 5), false);
  assert.equal(isNewer(5, 5), false);
  // The sequence wraps at 16 bits, on the wire and in a JSON counter that has
  // been running for a quarter of an hour at 72 Hz.
  assert.equal(isNewer(2, 65534), true);
  assert.equal(isNewer(65534, 2), false);
  return "wrap handled in both directions";
});

await asyncSection("an ungranted local-floor cannot hold 3D startup", async () => {
  let requests = 0;
  const denied = {
    enabledFeatures: ["local", "viewer"],
    requestReferenceSpace() {
      requests += 1;
      return new Promise(() => {});
    },
  };
  const deniedResult = await Promise.race([
    floorReferenceSpaceType(denied),
    new Promise((resolve) => setTimeout(() => resolve("timeout"), 50)),
  ]);
  assert.equal(deniedResult, "local");
  assert.equal(requests, 0, "the ungranted reference space was still requested");

  const granted = {
    enabledFeatures: ["local", "local-floor"],
    async requestReferenceSpace(type) {
      requests += 1;
      assert.equal(type, "local-floor");
      return { type };
    },
  };
  assert.equal(await floorReferenceSpaceType(granted), "local-floor");
  assert.equal(requests, 1);
  return "ungranted skipped, granted requested once";
});

const failed = results.filter((result) => !result.ok);
console.log(
  `\n${results.length - failed.length} passed, ${failed.length} failed`,
);
process.exit(failed.length === 0 ? 0 : 1);
