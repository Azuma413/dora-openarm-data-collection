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

// openarm-quest-teleop patch: portions derived from enactic/dora-openarm-webxr
// instructions.js (Apache-2.0).
//
// The node's status text for the operator: recording or not, the episode,
// the task, what each arm is doing. The operator is wearing the headset,
// so this is the only place they can read it.
//
// Drawn like the calibration instructions: text on a 2D canvas, uploaded
// as a texture onto a head-locked quad. It hangs above the middle of the
// view, where a head camera looking down at the table shows background,
// and above the instruction panel, which sits below the middle. Drawn in
// immersive-ar and immersive-vr alike, over whatever the camera view drew.

const VERTEX_SHADER = `
attribute vec2 a_corner;
uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec2 u_half_extent;
uniform vec2 u_center;
uniform float u_distance;
varying vec2 v_uv;
void main() {
  gl_Position = u_projection * u_view *
    vec4(a_corner * u_half_extent + u_center, -u_distance, 1.0);
  // Flipped because the canvas starts at its top row and texture
  // coordinates start at the bottom.
  v_uv = vec2((a_corner.x + 1.0) * 0.5, (1.0 - a_corner.y) * 0.5);
}
`;

const FRAGMENT_SHADER = `
precision mediump float;
uniform sampler2D u_texture;
varying vec2 v_uv;
void main() {
  gl_FragColor = texture2D(u_texture, v_uv);
}
`;

// The canvas the text is drawn on, in pixels: room for MAX_LINES lines.
export const CANVAS = { width: 1024, height: 384 };

// Where the panel hangs, in meters, in the operator's own view: a little
// above the middle, about 26 by 10 degrees, closer than the camera panel
// so it is never behind it.
export const PANEL = { distance: 1.1, width: 0.5, center: 0.24 };

// The node cuts messages to this many lines too; more would not fit.
const MAX_LINES = 5;

// Left margin and line pitch on the canvas, in pixels.
const MARGIN = 40;
const LINE_HEIGHT = 68;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

// Shorten text with an ellipsis until it fits, rather than letting it run
// off the panel where the operator cannot see it was cut.
function fit(context, text, width) {
  if (context.measureText(text).width <= width) {
    return text;
  }
  let shortened = text;
  while (
    shortened.length > 0 &&
    context.measureText(`${shortened}…`).width > width
  ) {
    shortened = shortened.slice(0, -1);
  }
  return `${shortened}…`;
}

// The text of the HUD and the canvas it is laid out on, with no WebGL of its
// own (openarm-quest-teleop patch 15). The panel below draws this canvas
// through its own shader; the 3D view draws the same canvas as a texture on a
// three.js mesh, because three.js owns the render state there and the
// session's base layer the shader needs is null. One layout, two consumers,
// so the HUD reads the same either way.
class HudContent {
  #canvas = null;
  #context = null;
  #lines = [];
  #recording = false;
  #notice = "";
  // Bumped by every change and compared by each consumer against what it last
  // uploaded, so the text is laid out on a change rather than on every frame
  // and neither consumer can miss one.
  #revision = 1;
  #painted = 0;

  constructor() {
    this.#canvas = document.createElement("canvas");
    this.#canvas.width = CANVAS.width;
    this.#canvas.height = CANVAS.height;
    this.#context = this.#canvas.getContext("2d");
  }

  get canvas() {
    return this.#canvas;
  }

  get revision() {
    return this.#revision;
  }

  // The whole HUD, as the node sent it: a message replaces the last one,
  // and no lines hides the panel.
  set(message) {
    const lines = Array.isArray(message.lines) ? message.lines : [];
    this.#lines = lines.slice(0, MAX_LINES).map((line) => String(line));
    this.#recording = message.recording === true;
    this.#revision += 1;
  }

  // The page's own line, kept apart from the node's so a HUD update cannot
  // take it down: what the 3D view could not do (no scene asset, a stalled
  // state stream, no WebGL) has to reach the operator inside the headset, and
  // this panel is already there. "" clears it.
  setNotice(text) {
    const notice = text ? String(text) : "";
    if (notice === this.#notice) {
      return;
    }
    this.#notice = notice;
    this.#revision += 1;
  }

  // What is shown, for tests that cannot read the headset's display.
  get state() {
    return { lines: [...this.#lines], recording: this.#recording };
  }

  get notice() {
    return this.#notice;
  }

  get empty() {
    return this.#lines.length === 0 && this.#notice === "";
  }

  // Lay the text out if it has changed, and return the revision now on the
  // canvas. Cheap to call every frame.
  paint() {
    if (this.#painted === this.#revision) {
      return this.#revision;
    }
    const context = this.#context;
    const { width, height } = CANVAS;
    // A notice takes the last row, so the node's lines never sit on top of it.
    const room = this.#notice ? MAX_LINES - 1 : MAX_LINES;
    const lines = this.#lines.slice(0, room);

    context.clearRect(0, 0, width, height);
    // Dark enough to read white text over passthrough or a bright image.
    context.fillStyle = "rgba(0, 0, 0, 0.72)";
    context.fillRect(0, 0, width, height);
    if (this.#recording) {
      // A red frame reads as "recording" from the corner of the eye, which
      // is where the panel is while the operator looks at the task.
      context.strokeStyle = "#ff3b30";
      context.lineWidth = 16;
      context.strokeRect(8, 8, width - 16, height - 16);
    }

    context.textAlign = "left";
    context.textBaseline = "middle";
    lines.forEach((line, index) => {
      const first = index === 0;
      context.font = first ? "bold 54px sans-serif" : "46px sans-serif";
      context.fillStyle = first && this.#recording ? "#ff6b61" : "#ffffff";
      const y = MARGIN + LINE_HEIGHT / 2 + index * LINE_HEIGHT;
      context.fillText(fit(context, line, width - 2 * MARGIN), MARGIN, y);
    });
    if (this.#notice) {
      // Amber, and always on the bottom row wherever the node's lines end,
      // so the operator learns to look in one place for it.
      context.font = "40px sans-serif";
      context.fillStyle = "#ffcc55";
      const y = MARGIN + LINE_HEIGHT / 2 + (MAX_LINES - 1) * LINE_HEIGHT;
      context.fillText(
        fit(context, this.#notice, width - 2 * MARGIN),
        MARGIN,
        y,
      );
    }
    this.#painted = this.#revision;
    return this.#revision;
  }
}

class HudPanel {
  #gl = null;
  #program = null;
  #buffer = null;
  #attributes = {};
  #uniforms = {};
  #texture = null;
  #content = null;
  // The revision uploaded into the texture, or -1 for a context whose
  // texture is still empty.
  #uploaded = -1;

  constructor(content) {
    this.#content = content;
  }

  // The canvas and the text on it, shared with the 3D view.
  get content() {
    return this.#content;
  }

  set(message) {
    this.#content.set(message);
  }

  setNotice(text) {
    this.#content.setNotice(text);
  }

  get state() {
    return this.#content.state;
  }

  // Called once per WebXR session, with the session's WebGL context. A new
  // session brings a new context, so everything GL is made again.
  attach(gl) {
    this.#gl = gl;
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    this.#program = program;

    this.#attributes.corner = gl.getAttribLocation(program, "a_corner");
    for (const name of [
      "u_projection",
      "u_view",
      "u_half_extent",
      "u_center",
      "u_distance",
      "u_texture",
    ]) {
      this.#uniforms[name] = gl.getUniformLocation(program, name);
    }

    this.#buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.#buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW,
    );

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Not a power of two, so no mipmaps and no repeating.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.#texture = texture;
    // The vertical flip is done in the vertex shader instead.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    // The texture of the new context is empty, whatever was drawn before.
    this.#uploaded = -1;
  }

  #upload() {
    const revision = this.#content.paint();
    if (revision === this.#uploaded) {
      return;
    }
    const gl = this.#gl;
    gl.bindTexture(gl.TEXTURE_2D, this.#texture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      this.#content.canvas,
    );
    this.#uploaded = revision;
  }

  // ``space`` is the viewer space, so the text stays in view through any
  // head turn. ``clears`` is set when nothing else drew this frame, since
  // the last frame would otherwise stay on the display.
  render(session, space, frame, { clears = false } = {}) {
    if (!this.#gl) {
      return;
    }
    const pose = frame.getViewerPose(space);
    if (!pose) {
      return;
    }
    const gl = this.#gl;
    const layer = session.renderState.baseLayer;

    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    if (clears) {
      // Transparent so passthrough shows around the panel.
      gl.clearColor(0.0, 0.0, 0.0, 0.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    }
    if (this.#content.empty) {
      return;
    }
    gl.disable(gl.DEPTH_TEST);
    // Over whatever the camera view drew, rather than in place of it.
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    this.#upload();

    const halfExtent = [
      PANEL.width / 2.0,
      ((PANEL.width / 2.0) * CANVAS.height) / CANVAS.width,
    ];

    gl.useProgram(this.#program);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.#buffer);
    gl.enableVertexAttribArray(this.#attributes.corner);
    gl.vertexAttribPointer(this.#attributes.corner, 2, gl.FLOAT, false, 0, 0);
    gl.uniform2f(this.#uniforms.u_half_extent, halfExtent[0], halfExtent[1]);
    gl.uniform2f(this.#uniforms.u_center, 0.0, PANEL.center);
    gl.uniform1f(this.#uniforms.u_distance, PANEL.distance);
    gl.uniform1i(this.#uniforms.u_texture, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.#texture);

    for (const view of pose.views) {
      const viewport = layer.getViewport(view);
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      gl.uniformMatrix4fv(
        this.#uniforms.u_projection,
        false,
        view.projectionMatrix,
      );
      gl.uniformMatrix4fv(
        this.#uniforms.u_view,
        false,
        view.transform.inverse.matrix,
      );
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    gl.disable(gl.BLEND);
  }
}

export function createHudPanel() {
  return new HudPanel(new HudContent());
}
