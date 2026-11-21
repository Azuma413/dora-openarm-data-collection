# dora-openarm-policy

Runs the small flow-matching policy (`training/small_flow`) on OpenArm, between
the IK and the safety gate, with a browser UI to draw the target bbox on
head_left. Used by `LEADER=policy` (`modules/leader-policy.yaml`).

```
clutch --target--> ik --position--> policy --position--> gate --cmd--> robot
  ^                                   |  ^                 |
  +---------- clutch_control ---------+  +----- cmd -------+
```

## Setup and start

```bash
scripts/policy-setup     # once: .venv-policy with CUDA torch (the main .venv has no torch)
scripts/policy           # LEADER=policy ROBOT=pedestal-zed; the arms start moving
```

Then open http://localhost:8010/ (the data collection UI stays on :8000).

## Flow

1. **Intro.** Until Start is pressed, the node passes the IK's output through, so
   the clutch does exactly what it does in data collection: the left arm turns
   the finger on the right down and opens; press **湿布を受け取った** after
   putting the patch in; it closes and turns the finger on the left down. The
   right arm holds its start pose, gripper 60% open.
2. **bbox.** Drag on head_left. It is sent as a fraction of the image and used
   in head_left 1280x720 pixels, as in training. It can be redrawn at any
   time; the next inference uses it.
3. **Test inference** (optional) runs the policy once without moving and reports
   how far the first target is from the arms and how far the chunk goes.
4. **Start.** The policy drives both arms: a 30-step chunk (1 s at 30 fps), re-planned
   every `EXECUTE_STEPS` (10 = 0.33 s). Each new chunk is cross-faded in over
   0.15 s and the first over 0.3 s, so the gate never sees a jump.
5. **Stop** (button, Space or Esc) holds the gate's current command. **Start** again
   resumes from there.
6. **初期姿勢に戻して初期動作をやり直す** eases both arms in joint space back to
   where the clutch holds them (0.3 rad/s), hands them back to the clutch, and
   sends it `cancel`: both arms go to the start pose and the intro runs again.
   The data collection UI's success/fail/cancel do the same while the policy has
   the arms.

The policy stops by itself (holds) when a motor is not driving its arm, when
either gate side stops following (`holding`), when a camera image is older than
0.5 s or the arm state older than 0.25 s, when inference fails, or when no chunk
arrives within 3 s.

A motor is not driving its arm when it stops answering (`SILENT`), reports a
fault (`OVERCURRENT`, ...), or is `DISABLED` while the others are `ENABLED`. The
last one is a motor that lost power for a moment and restarted. Its reading is
still fresh, but it holds no torque until the arm is stopped and started again.
The gate holds the arm in all three cases (`stale_state`), and the UI names the
motor.

## Voice dialogue

Press **🎤 音声オン** in the UI (the browser asks for the microphone; the click also
lets it play the replies). Speak or type:

1. **Where.** "左肩に貼って". The VLM sees head_left (the snapshot after the intro)
   with the presets drawn as numbered boxes. It asks follow-up questions aloud
   until the place is unique ("肩の上の方ですか、下の方ですか"), then picks the closest
   preset. The bbox is set and the policy starts as soon as the reply has been
   played (or once the intro is done, if it is not yet).
2. **Corrections.** While the policy runs, **any voice stops the robot at once**.
   The browser detects speech locally, from the level against the room's noise,
   and sends `voice_stop` before anything goes to OpenAI. The utterance is then
   transcribed, and the VLM picks one of:
   - **adjust** ("もっと上", "少し右"): move the bbox by whole steps (`step_px`), its
     center kept within the presets' extent plus `margin_px`, reply, restart.
   - **resume** ("続けて"): restart as is.
   - **relocate**: another preset, restart.
   - **stop** or **ask**: stay stopped and reply.
   An unintelligible utterance after a voice stop keeps the robot stopped
   ("止まりました。指示をどうぞ。").

Detection pauses while a reply plays, so the robot does not stop itself. A
sound counts as speech when it is louder than the noise floor (the 20th
percentile of the last 3 s, so it follows the motors) by **感度** dB *and*
louder than **最小音量** (default -58 dBFS, the white line on the meter). Raise
either if the motors stop the robot; lower them if a voice is missed. The
sliders are remembered per browser. An utterance whose peak is below -46 dBFS
is taken for noise: never transcribed, and if it was what stopped the robot,
the policy resumes by itself. Left and right are the patient's own; the VLM turns them into image
directions from the person's orientation in the picture.

| Variable | Default | |
|---|---|---|
| `OPENAI_API_KEY` | in `.env` at the repository root (see `.env.example`) | without it the dialogue is off and the UI says so |
| `VLM_MODEL` | `gpt-6.2-sol` | Responses API, JSON schema output, image input |
| `STT_MODEL` | `gpt-4o-transcribe` | |
| `TTS_MODEL` / `TTS_VOICE` | `gpt-4o-mini-tts` / `alloy` | |
| `PATCH_PRESETS` | `configs/patch_presets.yaml` | |

### Presets

`configs/patch_presets.yaml` lists the target bboxes the VLM may choose from
(1280x720 px), with `step_px` and `margin_px` for corrections, and `view`: which
side of the patient head_left sees (`back`: their left is the image's left, as
on the robot; `front`: mirrored). The VLM is told it rather than guessing.

The current file is an evenly spaced 3x3 grid over the curated dataset's 103
annotated targets (`meta/target_bbox_segments.csv`): centers from the 10th to
the 90th percentile of the targets' centers, 75 px apart. Every bbox is the
targets' median size (118x155 px), so the policy gets a trained size. Boxes
that size overlap at 75 px spacing, so the UI and the VLM see the presets as
non-overlapping 75x75 px cells (`cell_px`) numbered in the middle. `margin_px`
70 lets voice corrections reach the targets' whole spread.

```bash
.venv/bin/python scripts/make-patch-presets datasets/webxr-pedestal-zed-lerobot-curated \
    configs/patch_presets.yaml --regular 3x3 --margin-px 70 --preview presets.jpg
```

`--regular COLSxROWS` is that grid. `--grid COLSxROWS` instead splits the
targets at quantiles of their centers, and `-k N` clusters them with k-means
(the targets are spread over the upper back rather than clustered, so both
give overlapping boxes). The
dataset can also be a training view with a `target_bbox` column. Rename the
presets to body parts after looking at the preview.

## MolmoAct2

`POLICY=molmoact2` (in the environment or `.env`) runs the fine-tuned MolmoAct2
instead of the small flow policy:

```bash
POLICY=molmoact2 scripts/policy
# another checkpoint
POLICY=molmoact2 MOLMOACT2_CHECKPOINT=training/molmoact2/checkpoints/<run>/stepN-merged scripts/policy
```

- The merged checkpoint (`stepN-merged/`: `config.yaml` and one fp32 `model*.pt`)
  is loaded by the release code's LeRobot policy (`lib/molmoact2/experiments`,
  a git submodule that `scripts/policy-setup` fetches).
- The weights are materialized in bf16 directly on the GPU, and the fp32 file is
  memory-mapped, not read into RAM: 11 GB of VRAM, about 10 s to load. Inference
  runs under bf16 autocast, as training (`amp_bf16`) did.
- Inputs, as in training: head_left with the bbox drawn on the full 1280x720
  image (green, 4 px, `molmoact2_patches.draw_bbox`) before the model resizes it,
  then wrist_right, the 16 measured joints, and the instruction exactly as
  trained, not normalized: "Apply the compress patch at the location marked by
  the green box.". Normalization tag `openarm_bimanual_bbox`.
- Style `robot_action` (the action expert straight from the VLM, no depth tokens
  generated). Training used it at weight 1.0 next to `robot_depth_action`.
  About 0.42 s per chunk on an RTX 4090 (0.55 s inside the dataflow).
- The grippers stay where the clutch had them at Start, for both policies. The
  demonstrations held them fixed, so their normalization range is zero, and
  MolmoAct2 returns about 0 for the right gripper instead of its -0.47.

MolmoAct2's consecutive chunks disagree more than the small flow policy's, so
with `POLICY=molmoact2` a new chunk is cross-faded in over 0.4 s (`BLEND_S`)
instead of 0.15 s. Measured offline on recorded frames, at the deployed timing
(chunk 16 steps old on arrival, re-planned every 10), the target changes by
0.16 rad (median, p90 0.35) when a chunk takes over, against 0.11 rad for the
small flow policy. Fresh noise for every chunk made that 0.24 rad, so the flow's
starting noise is the same for every inference (`POLICY_SEED`, default 0;
`random` for fresh noise). Do not re-plan less often (`EXECUTE_STEPS=20`): with
~0.55 s per inference the next chunk would arrive after the current one has run
out. If the gate still holds with `jump`, try `BLEND_S=0.6`.

The chunks can also be played slower than the 30 fps they were trained at: the
**実行速度** slider (and `POLICY_SPEED`) stretches each chunk in time, so the arm
follows the same path at a fraction of the speed. Re-planning still happens
every `EXECUTE_STEPS` / 30 s of wall time, and a changed speed takes effect
without moving the target. MolmoAct2 starts at 50%.

## Inputs to the network

Taken from `training/small_flow/small_flow.py` itself (imported by path), so the
model, the bbox drawing and the normalization are the trained ones:

- head_left: the ZED's JPEG decoded and converted BGR -> RGB (raw `bgra8` is
  also converted), resized to 256x144, bbox drawn on the resized image.
- wrist_right: RGB, 256x160.
- state: measured joints, right 8 then left 8.
- bbox: x1, y1, x2, y2 in head_left 1280x720 px.

The UI shows what the network sees (head with the bbox, wrist) at training
resolution.

## Settings (environment)

| Variable | Default | |
|---|---|---|
| `POLICY_CHECKPOINT` | `training/small_flow/runs/flow40k_v3/step20000.pt` | |
| `SMALL_FLOW_PY` | `training/small_flow/small_flow.py` | model code |
| `EXECUTE_STEPS` | 10 | chunk steps run before re-planning (1..29) |
| `FLOW_STEPS` | 10 | Euler steps per sample |
| `RETURN_SPEED` | 0.3 | rad/s when easing back to the clutch |
| `POLICY_UI_PORT` | 8010 | UI port (bound to 127.0.0.1) |
| `POLICY` | `small_flow` | or `molmoact2` |
| `MOLMOACT2_CHECKPOINT` | `training/molmoact2/checkpoints/molmoact2-think-openarm-bbox-lora-h200/step2000-merged` | |
| `BLEND_S` | 0.15 (MolmoAct2 0.4) | seconds over which a new chunk is cross-faded in |
| `POLICY_SPEED` | 1 (MolmoAct2 0.5) | playback speed of the chunks; the UI's **実行速度** slider changes it live |
| `POLICY_SEED` | 0 | the flow's starting noise, the same for every chunk; `random` for fresh noise |

The node reads `.env` at the repository root when it starts. A variable already
set in the environment wins over `.env`, except `POLICY` and
`MOLMOACT2_CHECKPOINT`, which the module passes empty when they are unset so
that `.env` can choose them.

## Tests

```bash
.venv-policy/bin/python -m pytest nodes/dora-openarm-policy/tests
```
