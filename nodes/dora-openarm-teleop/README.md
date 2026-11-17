# dora-openarm-teleop

The stages between the WebXR node and the robot in `modules/leader-webxr.yaml`:

| Command | Node |
|---|---|
| `dora-openarm-clutch` | Grip clutch: an arm follows the hand's motion only while its grip is squeezed |
| `dora-openarm-teleop-ik` | Starts `dora-openarm-ik` with the gate's ranges and speeds, no per-solve error limits, on the fast cores |
| `dora-openarm-gate` | Joint-space safety gate: bounded speed, jump/range/tracking checks |
| `dora-openarm-teleop-hud` | The headset HUD and haptics |
| `openarm-teleop-check` | Checks a dataflow's teleop couplings (`scripts/check-dataflow`) |
| `openarm-teleop-tls` | TLS certificates and the pairing token (`scripts/webxr-setup`) |

See [docs/teleop.md](../../docs/teleop.md) for what each one does and why.

```bash
uv run pytest nodes/dora-openarm-teleop/tests            # unit tests, ~10 s
uv run pytest nodes/dora-openarm-teleop/tests -m e2e -s  # real dora + MuJoCo, ~15 s
```

Ported from openarm-quest-teleop (the clutch, the gate, the IK model, the
frames and the TLS helpers), with the scene list cut down to
`openarm_mujoco`'s bundled scenes.
