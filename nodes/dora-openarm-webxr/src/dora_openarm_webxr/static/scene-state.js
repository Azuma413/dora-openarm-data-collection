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

// openarm-quest-teleop addition (PATCHES.md 15): the sim's scene state, as it
// arrives on the "state" data channel.
//
// One message is one instant of the simulation, sampled on the sim clock
// (oqt.sim.scene_state writes both encodings; this reads them):
//
//   JSON   {"type": "state", "v": 1, "seq": 4242, "t": 12.3456, "flags": 5,
//           "qpos": [...], "obj": [x,y,z,qw,qx,qy,qz, ...],
//           "bodies": {"compress_plate_00": [x,y,z,qw,qx,qy,qz], ...}}
//
//   binary u32 magic "OQT3" | u8 version | u8 flags | u16 seq | f64 sim time
//          | u16 nq | u16 nobj | f32[nq] qpos | f32[nobj * 7] object poses
//
// The page needs only `qpos`: the exporter writes every joint of the model,
// free joints included, into the glTF tree, so mjkin.js can put every body
// where MuJoCo has it from the joint values alone (19 floats for the cell).
// `obj` and `bodies` are for whatever the joint values cannot say -- a body
// the sim drives directly, or a deformable the page is given the result of.
// All three are optional, and a message carrying none of them is still worth
// decoding for its flags.
//
// Which names `obj` is in is decided once, in the configuration message
// (`scene.objects`), not per frame.
//
// This file is deliberately the only place that knows the wire format. The
// node starts with JSON; the binary layout above is the drop-in for when a
// deformable liner makes the difference (a 20x20 flex grid is 4.8 kB binary
// against 12 kB of JSON), and both decode here.

const MAGIC = 0x4f515433;
export const BINARY_HEADER_BYTES = 24;

// How long the page keeps drawing the last state before it says the stream
// has stopped. Five frames at 72 Hz: long enough that a dropped message on an
// unreliable channel is invisible, short enough that the operator learns the
// robot is not being reported any more well inside a reach.
export const STALE_AFTER_MS = 400;

export const FLAG = {
  clutchRight: 1 << 0,
  clutchLeft: 1 << 1,
  recording: 1 << 2,
  gateFault: 1 << 3,
};

// Decode one message from the channel, or null if it is not one of ours.
// Never throws: a malformed message on an unreliable channel is dropped, not
// a reason to stop drawing.
export function decodeSceneState(data) {
  try {
    if (typeof data === "string") {
      return fromJson(JSON.parse(data));
    }
    if (data instanceof ArrayBuffer) {
      return fromBinary(data);
    }
    if (ArrayBuffer.isView(data)) {
      return fromBinary(
        data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
      );
    }
  } catch (error) {
    console.error(`cannot read a scene state message: ${error}`);
  }
  return null;
}

function fromJson(payload) {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  // A node that labels its messages is taken at its word; one that does not
  // is accepted if it looks like a state message, because the version and the
  // type are exactly the fields an older node may not send yet.
  if (payload.type !== undefined && payload.type !== "state") {
    return null;
  }
  const qpos = numbers(payload.qpos);
  const objects = numbers(payload.obj ?? payload.objects);
  const bodies = poseMap(payload.bodies ?? payload.body);
  if (
    qpos === null &&
    objects === null &&
    bodies === null &&
    payload.flags === undefined
  ) {
    return null;
  }
  return {
    // The node calls it `v`; `version` is taken too, so a message written by
    // hand or by an older encoder still reads.
    version: Number(payload.v ?? payload.version ?? 1),
    sequence: Number(payload.seq ?? payload.sequence ?? 0) >>> 0,
    simTime: Number(payload.t ?? payload.sim_time ?? 0),
    flags: flags(Number(payload.flags || 0)),
    qpos,
    objects,
    bodies,
  };
}

function fromBinary(buffer) {
  if (buffer.byteLength < BINARY_HEADER_BYTES) {
    return null;
  }
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC) {
    return null;
  }
  const nq = view.getUint16(16, true);
  const nobj = view.getUint16(18, true);
  const values = nq + nobj * 7;
  if (buffer.byteLength < BINARY_HEADER_BYTES + values * 4) {
    return null;
  }
  // Subarrays of the message's own buffer: nothing is copied, and the
  // renderer reads them in place.
  const floats = new Float32Array(buffer, BINARY_HEADER_BYTES, values);
  return {
    version: view.getUint8(4),
    sequence: view.getUint16(6, true),
    simTime: view.getFloat64(8, true),
    flags: flags(view.getUint8(5)),
    qpos: nq > 0 ? floats.subarray(0, nq) : null,
    objects: nobj > 0 ? floats.subarray(nq) : null,
    bodies: null,
  };
}

function flags(bits) {
  return {
    raw: bits,
    clutchRight: (bits & FLAG.clutchRight) !== 0,
    clutchLeft: (bits & FLAG.clutchLeft) !== 0,
    recording: (bits & FLAG.recording) !== 0,
    gateFault: (bits & FLAG.gateFault) !== 0,
  };
}

function numbers(value) {
  if (!Array.isArray(value) || value.length === 0) {
    return null;
  }
  const out = new Float32Array(value.length);
  for (let i = 0; i < value.length; i++) {
    // Only a real number will do. Number("") and Number(null) are both 0,
    // which would quietly draw a joint at zero rather than refusing the
    // message: one bad value means the whole message is untrustworthy,
    // because half a qpos is an arm in a pose the robot is not in.
    if (typeof value[i] !== "number" || !Number.isFinite(value[i])) {
      return null;
    }
    out[i] = value[i];
  }
  return out;
}

function poseMap(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const out = new Map();
  for (const [name, pose] of Object.entries(value)) {
    const values = numbers(pose);
    if (values && values.length >= 7) {
      out.set(name, values);
    }
  }
  return out.size > 0 ? out : null;
}

// Whether `sequence` is newer than `previous`, on a channel that is allowed
// to deliver late. Compared modulo 2^16, which is both what the binary header
// carries and what a JSON counter that has been running for a quarter of an
// hour at 72 Hz looks like once it passes 65535.
export function isNewer(sequence, previous) {
  if (previous === null || previous === undefined) {
    return true;
  }
  const difference = (sequence - previous) & 0xffff;
  return difference !== 0 && difference < 0x8000;
}

// The binary encoding, kept next to its decoder so the two cannot drift.
// Used by the tests, and by any node that would rather send bytes.
export function encodeBinary({
  sequence = 0,
  simTime = 0,
  qpos = [],
  objects = [],
  flags: bits = 0,
  version = 1,
}) {
  const nq = qpos.length;
  const nobj = Math.floor(objects.length / 7);
  const buffer = new ArrayBuffer(BINARY_HEADER_BYTES + (nq + nobj * 7) * 4);
  const view = new DataView(buffer);
  view.setUint32(0, MAGIC, true);
  view.setUint8(4, version);
  view.setUint8(5, bits);
  view.setUint16(6, sequence & 0xffff, true);
  view.setFloat64(8, simTime, true);
  view.setUint16(16, nq, true);
  view.setUint16(18, nobj, true);
  const floats = new Float32Array(buffer, BINARY_HEADER_BYTES);
  floats.set(qpos, 0);
  floats.set(objects.slice(0, nobj * 7), nq);
  return buffer;
}
