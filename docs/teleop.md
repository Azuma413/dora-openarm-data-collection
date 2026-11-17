# WebXR teleoperation (`LEADER=webxr`)

How a Meta Quest moves the arms, why each stage exists, and where its numbers
come from. The operator's steps are in the [README](../README.md) (Japanese);
the dataflow rules are in [design.md](design.md).

```
webxr --pose/grip--> clutch --target--> ik --position--> gate --cmd--> robot
 (Quest page)          |  ^                                ^  |
                       |  +----------- state_* ------------+  +-- start_epoch
                       +--clutch_state--> hud <--gate_state
```

All of it is [`modules/leader-webxr.yaml`](../modules/leader-webxr.yaml). The
nodes other than `webxr` are the [`dora-openarm-teleop`](../nodes/dora-openarm-teleop)
package. They are ported from openarm-quest-teleop, a MuJoCo and Quest 3
teleoperation project, where they were developed against measured headset
sessions.

## Controls

| Input | Does |
|---|---|
| **Grip** (middle finger), held | That arm follows the hand's *motion* since the squeeze |
| Grip released | That arm holds where it is |
| **Trigger** (index finger) | That gripper: pulled closes |
| **A** | Idle: start an episode. Recording: save it as a success |
| **B** | Recording: save it as a failure. Idle: **hold 2 s** to quit |

The arm follows the hand's displacement, not its absolute position. So you can
let go, move your hand somewhere comfortable, and squeeze again: the arm doesn't
jump (a "clutch", like lifting a mouse off the desk). Turning or leaning your
head doesn't move an engaged arm (`PIVOT_MODE=latched`).

Quit needs a 2 s hold because B is also the failure button. An operator who
fails an episode and keeps B down for a moment no longer ends the session.

## Stages

### webxr

[`nodes/dora-openarm-webxr`](../nodes/dora-openarm-webxr) is enactic's node
with the patches listed in its [`PATCHES.md`](../nodes/dora-openarm-webxr/PATCHES.md):

- Pose smoothing with a reset after gaps, latched neck pivot.
- The HUD panel and haptics.
- A pairing token in the URL, so that only your headset can drive the robot.
- A log of the page's own errors (`out/quest.log`).
- USB relaying (patch 18).

### Clutch (`dora-openarm-clutch`)

- **Engage.** An arm is armed when its grip has been released and the hand has
  been still for 300 ms (< 0.05 m/s, < 0.5 rad/s). Then the grip engages it at
  0.7 squeeze and releases it at 0.4. The hysteresis means a half-held grip
  doesn't chatter.
- **Gaps.** A hand pose older than 100 ms makes the arm hold. If the input comes
  back within 500 ms, within 20 mm and 5° of where it left, the arm resumes
  without a re-grip. Wi-Fi drops are usually this short.
- **Clamps.** Every tick it clamps the target to the arm's reach, keeps it on its
  own side, and keeps the fingertips above the table.
  - The table is the MuJoCo scene's when `ROBOT` is `mujoco` or `dummy`.
  - On a real robot, give the measured table height as `TABLE_Z`, in meters, in
    the `arm_origin` frame. Without `TABLE_Z` the table clamp is **off**, and
    the node says so at startup. Clamping against the simulated table's height
    would be wrong in either direction.
- **Where it starts.** The first `state_*` of each arm, and the first after each
  follower restart, sets that arm's hold to where the arm actually is, locked
  until the grip is released.
  - The follower's `start_epoch` metadata tells a restart (Pause Arm, then
    Start Arm) apart.
  - Without this, the IK would start at the scene's home pose. The Cell's start
    pose is 0.2 rad away from it in J4 (`openarm_driver` config), which is
    exactly the gate's `jump_tol`, so the gate would never engage.
  - Later states are ignored, because holding the measured pose would make the
    hold sag with the arm.

### IK (`dora-openarm-teleop-ik`)

`dora-openarm-teleop-ik` starts enactic's `dora-openarm-ik` with the following:

- **No per-solve frame-error limits.**
  - The noguchi fork passes
    `--frame-position-error-limit 0.001 --frame-orientation-error-limit 0.004`.
    That moves the target at most 1 mm and 0.004 rad per solve, which at 250
    solves/s caps the hand at 0.25 m/s and the wrist at 1 rad/s (57°/s). That
    is the lag operators feel.
  - Speed is bounded where it belongs: in joint space, by the gate.
- **The gate's joint ranges.** A copy of the MJCF with each J1-J7 range
  narrowed to the gate's range minus its margin, so the IK doesn't plan
  configurations that the gate then refuses.
- **Joint speeds at 0.9× the gate's caps** (`--config`). The IK's built-in caps
  are faster than the gate's. The gate would then clip every tick of a fast
  reach, and the arm would lag the IK's plan.
- **`--tick-hz` equal to the leader tick.** The IK integrates its velocity caps
  over 1/tick-hz. If it says 200 while ticks come at 250, the arm moves 25 %
  faster than the caps say. A unit test caught exactly this.
- **Pinned to the fast cores** on a big.LITTLE CPU, read from `cpu_capacity`.
  On a homogeneous CPU it doesn't pin.

The generated MJCF and config are cached under `~/.cache/dora-openarm-teleop/`
and keyed by their settings.

### Gate (`dora-openarm-gate`)

The last stage before the robot. Every tick, each arm's command moves towards
the IK's solution at no more than the joint speed caps:

| Profile | `velocity_scale` | Used when |
|---|---|---|
| sim | 0.8 × `joint_velocity_limits` | `ROBOT` is `mujoco` or `dummy` |
| real | 0.5 × `joint_velocity_limits` (J1 1.0 rad/s, J5-J7 3.15 rad/s) | any other `ROBOT` |

`joint_velocity_limits` comes from
[`joint_limits.yaml`](../nodes/dora-openarm-teleop/src/dora_openarm_teleop/joint_limits.yaml),
which also gives each joint range and its source. The ranges are the narrowest
of openarm_driver's pedestal config and the MJCF. **The Cell configs are not
checked against them.**

The gate holds, and says why, when any of these happens:

- the IK is older than 50 ms;
- the robot's state is older than 50 ms;
- the IK jumps more than 0.2 rad from the command;
- the IK solves a joint past its range;
- the arm doesn't follow the command (more than 0.30 rad off on the real
  robot, 0.25 rad in sim).

It engages by ramping the command towards the IK. It compares the IK with the
command, never with the measured pose: on the real robot the driver sends no
gravity torque, and the arm sags 0.07-0.16 rad below its command.

It seeds the command from the measured pose, at startup and on each new
`start_epoch`, and stamps that epoch on its commands. `dora-openarm` drops a
command addressed to an earlier start.

### HUD (`dora-openarm-teleop-hud`)

Shows in the headset:

- `READY` / `REC mm:ss ep N`;
- each arm's clutch: `FOLLOW` (engaged), `HOLD`, `REGRIP` (locked: release
  and squeeze again), `LATE` (input gap);
- any arm the gate is holding, and why.

It also buzzes: right controller on start/success/fail, left on cancel, both on
quit.

## Timing

```
0.03 s <= POSE_TIMEOUT (0.09) <= clutch stale (0.1) < SMOOTHER_RESET_GAP (0.12)
clutch stale (0.1) < resume gap (0.5)
```

- If webxr kept sending a pose after the clutch has dropped it, a stale hand
  would come back as fresh.
- If webxr's smoother didn't reset after a gap the clutch noticed, the first
  pose back would be smoothed towards the old one, and the arm would slide on
  resume.
- Why 0.09 s and not 0.1 s: webxr times frame arrivals and the clutch times
  grip deliveries, a few ms of dora jitter apart. In openarm-quest-teleop's
  measured headset sessions, with both at 0.1 s, 73 of 793 stalls re-engaged
  on a 66-70 mm slide; at 0.09 s none of 827 did.

## Checks

[`scripts/dora-envsubst`](../scripts/dora-envsubst) runs the same check as
[`scripts/check-dataflow`](../scripts/check-dataflow) before `run`, `start` and
`build`, and refuses to start a dataflow whose teleop nodes disagree on:

- the timing chain above, and `PIVOT_MODE=latched`;
- the tick rates of the leader, the clutch and the IK;
- the IK's input queue size (≥ 2, so a late IK doesn't pair this tick's right
  with last tick's left);
- the gate and the IK's profile, ranges, margin and velocity scale;
- whether the gate has the robot's state.

`SKIP_TELEOP_CHECK=1` skips it.

## Measured

`nodes/dora-openarm-teleop/tests/e2e` (`uv run pytest -m e2e -s`, about 15 s):

- **Setup.** Scripted hands drive the real clutch, IK, gate, HUD and
  dora-openarm-mujoco under dora. The right hand moves 0.10 m and turns 45° in
  0.5 s (0.2 m/s, 90°/s).
- **Measured.** The gate's command, through forward kinematics, against the
  clutch target. The figures below are from one run on a DGX Spark.

| IK arguments | Worst position error | Worst angle error | Settles after the hand stops |
|---|---|---|---|
| this module | 1.9 mm | 0.8° | 6 ms |
| plus the fork's error limits | 1.2 mm | 23.3° | 525 ms |

The test fails if the error limits stop making a difference, and if this
module's tracking gets worse than 5 mm, 3° or 0.1 s.

## Headset setup

```bash
scripts/webxr-setup          # once: tls/ with a local CA, the certificate and the pairing token
```

It prints the URLs to open in the Quest browser, `https://<host>:8443/#k=<token>`,
and the certificates' SHA-256 fingerprints. The Quest browser warns about the
certificate the first time; proceed past the warning. The token is the only thing that keeps others on the
network from driving the robot: don't share the URL.

- `--rotate-token` makes every saved URL stop working.
- `--force` makes new certificates, e.g. after the host's IP changed.

Over USB, `scripts/webxr-usb/start.sh` reverses ports 8443 and 3479 with adb
and runs a TURN server on 3479 (`sudo apt install coturn`). The page opened as
`https://localhost:8443/#k=<token>` relays through it on its own, by patch 18
in the webxr node. **This path isn't tested with a headset yet.**

## What was taken from the noguchi fork

From [guchio123/dora-openarm-data-collection](https://github.com/guchio123/dora-openarm-data-collection):

**Kept:**

- the Japanese README's operating procedure;
- `metadata.yaml` with tasks and `TASK`;
- the UI and recorder changes;
- the pedestal/ZED robot modules and `view-episode.py`;
- the followers' `--align-trigger gripper`. After each Start Arm, pull each
  trigger once: the follower then accepts the gate's commands. The gate starts
  them at the arm, so they align at once.
- `start_epoch`, now set by the gate from the follower's own state;
- the USB relay.

**Not taken:**

- **The IK error limits** (see [IK](#ik-dora-openarm-teleop-ik)).
- **`pose.mode: relative` in `view-webxr.yaml`.** Re-expressing the hands in
  the headset frame every frame moves an engaged arm whenever the head turns.
  The clutch plus a latched pivot gives the same "move from where my hand is"
  without that.
