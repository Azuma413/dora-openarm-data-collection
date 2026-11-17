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

// openarm-quest-teleop addition (PATCHES.md 15): where the simulated cell
// stands in the operator's room, and which way it faces.
//
// THE PLACEMENT RULE
//
// The node already owns one mapping between the two frames. main.py turns a
// controller pose into an arm_origin-frame target with
//
//   p_arm = R * (p_local - pivot) + frame_offset
//   R = [[0,0,-1],[-1,0,0],[0,1,0]]      (main.py _ROBOT_ROTATION_MATRIX:
//                                         WebXR x right / y up / z back ->
//                                         robot x forward / y left / z up)
//
// so the only rotation under which the robot moves the way the operator's
// hand moves is R^T, `ROBOT_TO_XR` below. It is not a choice: turn the robot
// by anything else and pushing a hand forward no longer pushes the gripper
// forward. That is why the default `exo` placement leaves the robot facing
// the same way as the operator (they stand behind it, looking over its
// shoulder) instead of turning it around to face them, which would look
// better and control mirrored.
//
// What is left is where to put it, and the clutch decides that too: it is a
// relative mapping (`p = P + s * (p_hand - p0)`), so nothing can keep the
// rendered gripper in the operator's hand once the clutch has re-anchored.
// The page therefore does not try. It puts the cell at a fixed, room-stable
// place in front of the operator:
//
//   the sim's arm_origin site lands `distance` m in front of the anchor and
//   `lateral` m to its right, at `height` m above the floor, and the robot
//   faces the anchor's heading turned by `yaw`
//
// where the anchor is the operator's head pose LATCHED at the same moments
// the node latches its neck pivot (session start, and after any gap in head
// poses longer than SMOOTHER_RESET_GAP -- see PATCHES.md 13). Latching the
// two together is the point: re-anchoring while the clutch holds an arm
// would move the robot under a hand that is commanding it.
//
// `ego` is the other placement: the exact inverse of the mapping above, which
// puts the robot's chest where the operator's is, so they wear it. It is
// exact only while the clutch is disengaged, for the reason above, so it is
// an option rather than the default.
//
// On top of either, `nudge` is the operator's own correction from the
// thumbsticks, for lining the virtual robot up with the real table it is
// drawn over. It is in the anchor's own axes, so it means the same thing
// whichever way the operator was facing when the scene anchored, and it
// survives a re-anchor: it is a correction to the rule, not to one anchor.

import { Matrix4, Quaternion, Vector3 } from "./three/three.module.min.js";

// R^T as a three.js matrix: robot (x forward, y left, z up) -> WebXR
// (x right, y up, z back).
// biome-ignore format: one row of the matrix per line.
export const ROBOT_TO_XR = new Matrix4().set(
  0, -1, 0, 0,
  0, 0, 1, 0,
  -1, 0, 0, 0,
  0, 0, 0, 1,
);

// main.py's own defaults, for a node that does not send its own values:
// _FRAME_OFFSET_CELL and _NECK_PIVOT_OFFSET.
export const DEFAULT_FRAME_OFFSET = [-0.085, 0, -0.14];
export const DEFAULT_NECK_PIVOT_OFFSET = [0.0, -0.075, 0.08];

// Where the cell stands when the node says nothing else. The arm_origin site
// of the v2 cell is 0.335 m above the table top, so an arm origin 1.285 m up
// puts the table at 0.95 m, about the height of the bench the real cell sits
// on. 1.15 m in front leaves room to step in without walking into the table.
export const DEFAULT_EXO = {
  distance: 1.15,
  lateral: 0.0,
  height: 1.285,
  yaw: 0.0,
};

// Where the arm_origin site is in the sim's world, when the node does not
// say (the v2 cell's `arm_origin`).
export const DEFAULT_ARM_ORIGIN_WORLD = [0.185, 0, 1.34];

// A standing adult's eyes, used only to guess the floor when the runtime has
// no local-floor space to give (WebXR's own default for `local`).
export const DEFAULT_EYE_HEIGHT = 1.6;

// How far the operator may nudge the scene, and how fast a thumbstick held
// fully over moves it. Small enough that a stick brushed by accident is a
// correction rather than a relocation, and bounded so the robot can never be
// pushed out of the room or behind the operator.
export const NUDGE = {
  deadzone: 0.2,
  rate: { forward: 0.3, lateral: 0.3, height: 0.2, yaw: 0.6 },
  limit: { forward: 1.0, lateral: 1.0, height: 0.5, yaw: Math.PI / 2 },
};

const scratchVector = new Vector3();
const scratchQuaternion = new Quaternion();
const scratchMatrix = new Matrix4();
const FORWARD = new Vector3(0, 0, -1);
const UP = new Vector3(0, 1, 0);

export function emptyNudge() {
  return { forward: 0, lateral: 0, height: 0, yaw: 0 };
}

// The neck pivot of a head pose, in the reference space, exactly as main.py's
// `_neck_pivot` computes it: the offset is turned into world axes, so "behind
// the face" follows where the head faces.
export function neckPivot(
  position,
  orientation,
  offset = DEFAULT_NECK_PIVOT_OFFSET,
) {
  scratchQuaternion.set(
    orientation.x,
    orientation.y,
    orientation.z,
    orientation.w,
  );
  return new Vector3(offset[0], offset[1], offset[2])
    .applyQuaternion(scratchQuaternion)
    .add(scratchVector.set(position.x, position.y, position.z));
}

// Which way a head pose faces, as a turn about the room's up axis.
//
// The head's own forward (-Z) is used, projected on the floor. Looking
// straight up or down leaves nothing to project, so the head's up axis stands
// in: it points backwards then, which is where the -Z came from.
export function headingYaw(orientation) {
  scratchQuaternion.set(
    orientation.x,
    orientation.y,
    orientation.z,
    orientation.w,
  );
  const forward = scratchVector
    .copy(FORWARD)
    .applyQuaternion(scratchQuaternion);
  let x = forward.x;
  let z = forward.z;
  const pitched = forward.y;
  if (Math.hypot(x, z) < 0.1) {
    // Looking up: the head's up axis points behind the operator, and the
    // other way round when looking down. Either way its horizontal part is
    // the one axis still in the floor plane.
    const up = scratchVector.copy(UP).applyQuaternion(scratchQuaternion);
    const sign = pitched > 0 ? -1 : 1;
    x = sign * up.x;
    z = sign * up.z;
  }
  return Math.atan2(-x, -z);
}

// Move a nudge by one frame of thumbstick, and keep it inside its limits.
// `axes` is {forward, lateral, height, yaw}, each -1..1 as the stick reads.
export function advanceNudge(nudge, axes, seconds) {
  const step = Math.min(Math.max(seconds, 0), 0.25);
  let moved = false;
  for (const name of ["forward", "lateral", "height", "yaw"]) {
    const raw = axes[name] || 0;
    if (Math.abs(raw) < NUDGE.deadzone) {
      continue;
    }
    // Take the deadzone out rather than stepping over it, so the slowest
    // movement the stick can ask for is a slow one.
    const scale = (Math.abs(raw) - NUDGE.deadzone) / (1 - NUDGE.deadzone);
    const limit = NUDGE.limit[name];
    const target =
      nudge[name] + Math.sign(raw) * scale * NUDGE.rate[name] * step;
    const clamped = Math.min(Math.max(target, -limit), limit);
    if (clamped !== nudge[name]) {
      nudge[name] = clamped;
      moved = true;
    }
  }
  return moved;
}

// The matrix that takes a point in the sim's world into the WebXR reference
// space the scene is drawn in. See the rule at the top of this file.
//
//   exo: M = T(origin) Ry(anchorYaw + yaw + nudge.yaw) R^T T(-armOriginWorld)
//   ego: M = T(pivot)  R^T T(-armOriginWorld - frameOffset), nudged the same
//
// `anchor` is {position, yaw} for exo and {pivot} for ego; both come from the
// head pose latched with the node's neck pivot.
export function placementMatrix({
  mode = "exo",
  armOriginWorld = DEFAULT_ARM_ORIGIN_WORLD,
  frameOffset = DEFAULT_FRAME_OFFSET,
  anchor = null,
  distance = DEFAULT_EXO.distance,
  lateral = DEFAULT_EXO.lateral,
  height = DEFAULT_EXO.height,
  yaw = DEFAULT_EXO.yaw,
  nudge = null,
} = {}) {
  const offset = nudge || emptyNudge();
  const origin = new Vector3(
    armOriginWorld[0],
    armOriginWorld[1],
    armOriginWorld[2],
  );
  const matrix = new Matrix4();
  if (mode === "raw") {
    // The axis remap alone, for comparing a render against MuJoCo's own.
    return matrix.copy(ROBOT_TO_XR);
  }

  if (mode === "ego") {
    // p_local = pivot + R^T (p_world - arm_origin - frame_offset): the exact
    // inverse of main.py's mapping, so a sim point and the hand commanding it
    // coincide while the clutch is disengaged.
    const pivot = anchor?.pivot
      ? new Vector3(anchor.pivot.x, anchor.pivot.y, anchor.pivot.z)
      : new Vector3(0, DEFAULT_EYE_HEIGHT, 0);
    const shift = origin
      .clone()
      .add(new Vector3(frameOffset[0], frameOffset[1], frameOffset[2]))
      .applyMatrix4(ROBOT_TO_XR)
      .multiplyScalar(-1)
      .add(pivot);
    // The nudge is in the room's axes here: ego has no anchor heading of its
    // own, because main.py's mapping never applies the head rotation either.
    shift.x += offset.lateral;
    shift.y += offset.height;
    shift.z -= offset.forward;
    matrix.copy(ROBOT_TO_XR).setPosition(shift);
    if (offset.yaw !== 0) {
      turnAbout(matrix, shift, offset.yaw);
    }
    return matrix;
  }

  const anchorPosition = anchor?.position || {
    x: 0,
    y: DEFAULT_EYE_HEIGHT,
    z: 0,
  };
  const anchorYaw = anchor?.yaw || 0;
  const heading = anchorYaw + yaw + offset.yaw;
  const forward = distance + offset.forward;
  const side = lateral + offset.lateral;
  // Where the arm_origin site lands: `forward` metres along the anchor's own
  // heading, `side` metres to its right, at an absolute height above the
  // floor (the operator's eye height is not part of it -- the robot stands on
  // the floor of the room, not on the operator).
  const place = new Vector3(
    anchorPosition.x +
      Math.cos(anchorYaw) * side -
      Math.sin(anchorYaw) * forward,
    height + offset.height,
    anchorPosition.z -
      Math.sin(anchorYaw) * side -
      Math.cos(anchorYaw) * forward,
  );
  matrix.multiplyMatrices(scratchMatrix.makeRotationY(heading), ROBOT_TO_XR);
  // T(place) M T(-armOrigin): the site itself lands on `place`, whatever the
  // model puts around it.
  const shifted = origin
    .clone()
    .applyMatrix4(matrix)
    .multiplyScalar(-1)
    .add(place);
  matrix.setPosition(shifted);
  return matrix;
}

// Turn an already-placed matrix about the vertical axis through `pivot`.
function turnAbout(matrix, pivot, angle) {
  const spin = new Matrix4().makeRotationY(angle);
  const back = new Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z);
  const there = new Matrix4().makeTranslation(pivot.x, pivot.y, pivot.z);
  matrix.premultiply(back).premultiply(spin).premultiply(there);
}

// The turn that puts a gripper marker in the operator's hand. main.py maps
// the controller's aim orientation to the end effector with
// R * q_controller * Rz(90 deg); drawing the same 90 degree turn on the aim
// pose in WebXR axes shows the orientation the gripper will take.
export const CONTROLLER_TO_EE = new Quaternion().setFromAxisAngle(
  new Vector3(0, 0, 1),
  Math.PI / 2,
);
