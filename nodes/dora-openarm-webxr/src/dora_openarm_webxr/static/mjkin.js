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

// openarm-quest-teleop addition (PATCHES.md 15): MuJoCo's own forward
// kinematics, applied to the glTF node tree the scene exporter writes.
//
// The exporter writes one glTF node per MuJoCo body, carrying the body's rest
// transform (body_pos / body_quat) and, in `extras.joints`, the joints that
// move it. Reproducing mj_kinematics on that tree is what lets the node send
// qpos alone -- 19 floats for the cell -- instead of a pose per body:
//
//   hinge: local = T(body_pos) R(body_quat) T(p) R(axis, q - q0) T(-p)
//   slide: local = T(body_pos) R(body_quat) T(axis * (q - q0))
//   free:  local = T(qpos[a..a+2]) R(qpos[a+3..a+6])   (a child of the world)
//   ball:  local = T(body_pos) R(body_quat) T(p) R(qpos[a..a+3]) T(-p)
//
// Everything is written straight into `object.matrix` with matrixAutoUpdate
// off, so a frame costs one matrix chain per moving body and no quaternion
// bookkeeping inside three.js. Measured at 0.05 ms median for the 127-node
// compress scene (0.2 ms at 4x CPU throttling); see tests/web/README.md.

import { Matrix4, Quaternion, Vector3 } from "./three/three.module.min.js";

const scratchQuaternion = new Quaternion();
const scratchMatrix = new Matrix4();
const scratchPivot = new Matrix4();
const scratchUnpivot = new Matrix4();
const scratchAxis = new Vector3();
const ONE = new Vector3(1, 1, 1);

export class Kinematics {
  // `root` is the loaded glTF scene. Bodies with no joint never move, so they
  // are left to three.js and cost nothing here.
  constructor(root) {
    this.root = root;
    this.bodies = [];
    this.byName = new Map();
    // The largest qpos address any joint reads, so a state message that is
    // too short for this model can be refused rather than read past its end.
    this.qposLength = 0;
    root.traverse((object) => {
      const entry = { object, name: object.name, joints: null };
      this.byName.set(object.name, entry);
      const joints = object.userData?.joints;
      if (!Array.isArray(joints) || joints.length === 0) {
        return;
      }
      entry.rest = new Matrix4().compose(
        object.position,
        object.quaternion,
        object.scale,
      );
      entry.joints = joints.map((joint) => {
        const axis = joint.axis || [0, 0, 1];
        const position = joint.pos || [0, 0, 0];
        const width = { free: 7, ball: 4 }[joint.type] || 1;
        this.qposLength = Math.max(this.qposLength, joint.qposadr + width);
        return {
          type: joint.type,
          axis: new Vector3(axis[0], axis[1], axis[2]),
          pos: new Vector3(position[0], position[1], position[2]),
          qposadr: joint.qposadr,
          ref: joint.ref || 0,
          // A joint anchored at the body origin needs no pivot shift, which
          // is every joint of the OpenArm model: skipping the two extra
          // matrix multiplies is most of the per-body cost.
          atOrigin: position[0] === 0 && position[1] === 0 && position[2] === 0,
        };
      });
      object.matrixAutoUpdate = false;
      this.bodies.push(entry);
    });
  }

  // qpos: the model's nq values, in the model's own order. A shorter array is
  // refused whole rather than applied in part, because a half-applied qpos
  // draws an arm in a pose the robot is not in.
  apply(qpos) {
    if (!qpos || qpos.length < this.qposLength) {
      return false;
    }
    for (let i = 0; i < this.bodies.length; i++) {
      const entry = this.bodies[i];
      const matrix = entry.object.matrix;
      matrix.copy(entry.rest);
      for (let j = 0; j < entry.joints.length; j++) {
        const joint = entry.joints[j];
        if (joint.type === "hinge") {
          scratchQuaternion.setFromAxisAngle(
            joint.axis,
            qpos[joint.qposadr] - joint.ref,
          );
          scratchMatrix.makeRotationFromQuaternion(scratchQuaternion);
          if (joint.atOrigin) {
            matrix.multiply(scratchMatrix);
          } else {
            scratchPivot.makeTranslation(joint.pos.x, joint.pos.y, joint.pos.z);
            scratchUnpivot.makeTranslation(
              -joint.pos.x,
              -joint.pos.y,
              -joint.pos.z,
            );
            matrix
              .multiply(scratchPivot)
              .multiply(scratchMatrix)
              .multiply(scratchUnpivot);
          }
        } else if (joint.type === "slide") {
          scratchAxis
            .copy(joint.axis)
            .multiplyScalar(qpos[joint.qposadr] - joint.ref);
          scratchMatrix.makeTranslation(
            scratchAxis.x,
            scratchAxis.y,
            scratchAxis.z,
          );
          matrix.multiply(scratchMatrix);
        } else if (joint.type === "free") {
          const a = joint.qposadr;
          // MuJoCo stores quaternions w first, three.js w last.
          scratchQuaternion.set(
            qpos[a + 4],
            qpos[a + 5],
            qpos[a + 6],
            qpos[a + 3],
          );
          matrix.compose(
            scratchAxis.set(qpos[a], qpos[a + 1], qpos[a + 2]),
            scratchQuaternion,
            ONE,
          );
        } else if (joint.type === "ball") {
          const a = joint.qposadr;
          scratchQuaternion.set(qpos[a + 1], qpos[a + 2], qpos[a + 3], qpos[a]);
          scratchMatrix.makeRotationFromQuaternion(scratchQuaternion);
          scratchPivot.makeTranslation(joint.pos.x, joint.pos.y, joint.pos.z);
          scratchUnpivot.makeTranslation(
            -joint.pos.x,
            -joint.pos.y,
            -joint.pos.z,
          );
          matrix
            .multiply(scratchPivot)
            .multiply(scratchMatrix)
            .multiply(scratchUnpivot);
        }
      }
      entry.object.matrixWorldNeedsUpdate = true;
    }
    return true;
  }

  // Put one body where the sim says it is, for a body the page is given a
  // pose for instead of joint values (a free object, or a body whose degrees
  // of freedom the qpos stream leaves out). `pose` is the MuJoCo world pose
  // [x, y, z, qw, qx, qy, qz]; the body must be a child of the world, which
  // is where the exporter puts every free-jointed body.
  setBodyPose(name, pose) {
    const entry = this.byName.get(name);
    if (!entry || !pose || pose.length < 7) {
      return false;
    }
    const object = entry.object;
    object.matrixAutoUpdate = false;
    scratchQuaternion.set(pose[4], pose[5], pose[6], pose[3]);
    object.matrix.compose(
      scratchAxis.set(pose[0], pose[1], pose[2]),
      scratchQuaternion,
      ONE,
    );
    object.matrixWorldNeedsUpdate = true;
    return true;
  }
}
