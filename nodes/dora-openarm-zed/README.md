# dora-openarm-zed

dora-rs node to capture a Stereolabs ZED stereo camera, e.g. the ZED Mini on
the head of the OpenArm Cell, and compute depth with the ZED SDK.

One `grab()` gives the rectified left and right images and the depth of the
same exposure, so the outputs of a tick always belong together. The depth is
computed on the GPU by the ZED SDK and is aligned with the left image.

## Setup

The ZED SDK needs an NVIDIA GPU with CUDA.

1. Install the ZED SDK 5.3 for your Ubuntu and CUDA version from
   <https://www.stereolabs.com/developers/release>, e.g. for Ubuntu 24.04 with
   CUDA 13:

   ```bash
   wget -O ZED_SDK.run https://download.stereolabs.com/zedsdk/5.3/cu13/ubuntu24
   chmod +x ZED_SDK.run
   ./ZED_SDK.run -- silent skip_python
   ```

   `skip_python` skips the SDK's own Python API installer: this project
   installs the matching `pyzed` wheel with `uv sync --locked`.

2. Plug the camera into a USB 3 port and check it with `ZED_Explorer`. The SDK
   downloads the camera's factory calibration to
   `/usr/local/zed/settings/SN<serial>.conf` on first use, so the first run
   needs Internet access.

3. The first run in a `NEURAL*` depth mode optimizes the neural network for
   the GPU, which takes several minutes. Run `ZED_Depth_Viewer` once in that
   mode, or start this node once, before collecting data.

`pyzed` must have the same major.minor as the installed ZED SDK. To use
another SDK version, change the `pyzed` version and wheel URL in the
repository's `pyproject.toml`, then run `uv lock` and `uv sync --locked`.

## Inputs

- `tick`: Grabs and sends one frame. The node stops when it is closed.
  `grab()` waits for the camera's next frame, so a tick faster than `FPS` only
  makes the node wait.

## Outputs

| Output | Encoding | Content |
|---|---|---|
| `image_left` | JPEG | Rectified left image |
| `image_right` | JPEG | Rectified right image |
| `depth` | 16-bit PNG | Depth in mm, aligned with `image_left`. 0 means no depth |
| `depth_preview` | JPEG | Colorized depth for monitoring (near: red, far: blue, none: black) |

All outputs carry `encoding`, `width`, `height` and `capture_timestamp` (ns,
the camera's exposure time) in their metadata. `depth` also carries the left
intrinsics `fx`, `fy`, `cx`, `cy` (pixels), `baseline` (mm), `depth_unit`
(`mm`), `camera_model` and `serial_number`.

To read a recorded depth image:

```python
import cv2

depth_mm = cv2.imread("cameras/head_depth/<timestamp>.png", cv2.IMREAD_UNCHANGED)
# uint16, shape (720, 1280). 0 is no depth.
```

A pixel `(u, v)` with depth `z` (mm) is at
`x = (u - cx) * z / fx`, `y = (v - cy) * z / fy` in the left camera frame.

## Configuration

Each option is an environment variable or a command line argument.

| Variable | Argument | Default | |
|---|---|---|---|
| `SERIAL_NUMBER` | `--serial-number` | `0` | Camera serial number; 0 opens the first camera |
| `SVO_FILE` | `--svo-file` | | Replay an SVO recording instead of a live camera |
| `RESOLUTION` | `--resolution` | `HD720` | `HD2K`, `HD1080`, `HD720` or `VGA` |
| `FPS` | `--fps` | `30` | Capture frame rate |
| `DEPTH_MODE` | `--depth-mode` | `NEURAL` | `PERFORMANCE`, `QUALITY`, `ULTRA`, `NEURAL_LIGHT`, `NEURAL`, `NEURAL_PLUS` |
| `DEPTH_MINIMUM` | `--depth-minimum` | `100` | Nearest depth in mm |
| `DEPTH_MAXIMUM` | `--depth-maximum` | `3000` | Farthest depth in mm |
| `CONFIDENCE_THRESHOLD` | `--confidence-threshold` | `95` | Drop depth below this confidence (1-100) |
| `TEXTURE_CONFIDENCE_THRESHOLD` | `--texture-confidence-threshold` | `100` | Drop depth on textureless areas (1-100) |
| `FILL_MODE` | `--fill-mode` | off | Fill holes in depth. Filled values are guesses |
| `JPEG_QUALITY` | `--jpeg-quality` | `90` | JPEG quality of the images |
| `PNG_COMPRESSION` | `--png-compression` | `1` | PNG compression of depth (0-9) |
| `PREVIEW_EVERY` | `--preview-every` | `3` | Send `depth_preview` every N ticks; 0 disables it |
| `PREVIEW_MAXIMUM` | `--preview-maximum` | `1500` | Depth in mm drawn as the farthest color in `depth_preview`. Doesn't change `depth` |
| `PREVIEW_WIDTH` | `--preview-width` | `640` | Width of `depth_preview` |

Depth stabilization is off so that each recorded depth image depends only on
its own frame.
