# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""dora-rs node to capture a ZED stereo camera and compute depth.

One `grab()` gives the rectified left and right images and the depth map of
the same exposure, so the three outputs of a tick always belong together.

Inputs:
  - tick: Grabs and sends one frame. The node stops when it is closed.

Outputs:
  - image_left:    Rectified left image, JPEG.
  - image_right:   Rectified right image, JPEG.
  - depth:         Depth aligned with image_left, 16-bit PNG in millimeters.
                   0 means no depth (occluded, out of range or low
                   confidence).
  - depth_preview: Colorized depth for monitoring, JPEG. Sent every
                   PREVIEW_EVERY ticks; not meant to be recorded.

All outputs carry "encoding", "width", "height" and "capture_timestamp" (ns,
the camera's exposure time). `depth` also carries the left camera intrinsics
("fx", "fy", "cx", "cy" in pixels), "baseline" (mm) and "depth_unit" ("mm"),
so a depth image can be turned into a point cloud without the ZED SDK.
"""

import argparse
import concurrent.futures
import os
import sys

import cv2
import numpy as np
import pyarrow as pa
from dora import Node

try:
    import pyzed.sl as sl
except ImportError as error:  # pragma: no cover - depends on the machine
    raise SystemExit(
        "pyzed is not available. Install the ZED SDK whose major.minor matches "
        "pyzed in pyproject.toml; see nodes/dora-openarm-zed/README.md."
        f"\n({error})"
    ) from error


RESOLUTIONS = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "VGA": sl.RESOLUTION.VGA,
}

DEPTH_MODES = {
    "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
    "QUALITY": sl.DEPTH_MODE.QUALITY,
    "ULTRA": sl.DEPTH_MODE.ULTRA,
    "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,
    "NEURAL": sl.DEPTH_MODE.NEURAL,
    "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS,
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments, defaulting to environment variables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--serial-number",
        default=int(os.getenv("SERIAL_NUMBER", 0)),
        help="Camera serial number; 0 opens the first camera (default: 0)",
        type=int,
    )
    parser.add_argument(
        "--svo-file",
        default=os.getenv("SVO_FILE"),
        help="Replay an SVO recording instead of a live camera",
    )
    parser.add_argument(
        "--resolution",
        choices=RESOLUTIONS,
        default=os.getenv("RESOLUTION", "HD720"),
        help="Capture resolution per eye (default: HD720, 1280x720)",
    )
    parser.add_argument(
        "--fps",
        default=int(os.getenv("FPS", 30)),
        help="Capture frame rate (default: 30)",
        type=int,
    )
    parser.add_argument(
        "--depth-mode",
        choices=DEPTH_MODES,
        default=os.getenv("DEPTH_MODE", "NEURAL"),
        help="ZED depth mode (default: NEURAL)",
    )
    parser.add_argument(
        "--depth-minimum",
        default=int(os.getenv("DEPTH_MINIMUM", 100)),
        help="Nearest depth in mm (default: 100, the ZED Mini's limit)",
        type=int,
    )
    parser.add_argument(
        "--depth-maximum",
        default=int(os.getenv("DEPTH_MAXIMUM", 3000)),
        help="Farthest depth in mm (default: 3000)",
        type=int,
    )
    parser.add_argument(
        "--confidence-threshold",
        default=int(os.getenv("CONFIDENCE_THRESHOLD", 95)),
        help="Drop depth below this confidence, 1-100 (default: 95)",
        type=int,
    )
    parser.add_argument(
        "--texture-confidence-threshold",
        default=int(os.getenv("TEXTURE_CONFIDENCE_THRESHOLD", 100)),
        help="Drop depth on textureless areas, 1-100 (default: 100, keep all)",
        type=int,
    )
    parser.add_argument(
        "--fill-mode",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("FILL_MODE", False),
        help="Fill holes in depth. Filled values are guesses (default: off)",
    )
    parser.add_argument(
        "--jpeg-quality",
        default=int(os.getenv("JPEG_QUALITY", 90)),
        help="JPEG quality of the color images, 0-100 (default: 90)",
        type=int,
    )
    parser.add_argument(
        "--png-compression",
        default=int(os.getenv("PNG_COMPRESSION", 1)),
        help="PNG compression of depth, 0-9. Higher is smaller and slower (default: 1)",
        type=int,
    )
    parser.add_argument(
        "--preview-every",
        default=int(os.getenv("PREVIEW_EVERY", 3)),
        help="Send depth_preview every N ticks; 0 disables it (default: 3)",
        type=int,
    )
    parser.add_argument(
        "--preview-maximum",
        default=int(os.getenv("PREVIEW_MAXIMUM", 1500)),
        help="Depth in mm drawn as the farthest color in depth_preview; "
        "farther is clipped. Doesn't change depth (default: 1500)",
        type=int,
    )
    parser.add_argument(
        "--preview-width",
        default=int(os.getenv("PREVIEW_WIDTH", 640)),
        help="Width of depth_preview in pixels (default: 640)",
        type=int,
    )
    return parser.parse_args()


def open_camera(args: argparse.Namespace) -> "sl.Camera":
    """Open the camera with depth computation enabled."""
    init = sl.InitParameters()
    if args.svo_file:
        init.set_from_svo_file(args.svo_file)
        # Pace the replay like a live camera so ticks don't outrun it.
        init.svo_real_time_mode = True
    elif args.serial_number:
        init.set_from_serial_number(args.serial_number)
    init.camera_resolution = RESOLUTIONS[args.resolution]
    init.camera_fps = args.fps
    init.depth_mode = DEPTH_MODES[args.depth_mode]
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.depth_minimum_distance = args.depth_minimum
    init.depth_maximum_distance = args.depth_maximum
    # Recorded frames should not depend on the previous ones.
    init.depth_stabilization = 0

    camera = sl.Camera()
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"cannot open the ZED camera: {status}")
    return camera


def camera_parameters(camera: "sl.Camera") -> dict:
    """Return the rectified left intrinsics and the baseline as metadata."""
    info = camera.get_camera_information()
    calibration = info.camera_configuration.calibration_parameters
    left = calibration.left_cam
    return {
        "camera_model": str(info.camera_model),
        "serial_number": int(info.serial_number),
        "fx": float(left.fx),
        "fy": float(left.fy),
        "cx": float(left.cx),
        "cy": float(left.cy),
        # In coordinate_units, i.e. mm.
        "baseline": float(calibration.get_camera_baseline()),
    }


def encode_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """JPEG-encode an image; returns a 1-D uint8 array."""
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return buffer.ravel()


def encode_png(depth: np.ndarray, compression: int) -> np.ndarray:
    """PNG-encode a 16-bit depth map; returns a 1-D uint8 array."""
    success, buffer = cv2.imencode(
        ".png", depth, [cv2.IMWRITE_PNG_COMPRESSION, compression]
    )
    if not success:
        raise RuntimeError("cv2.imencode(.png) failed")
    return buffer.ravel()


def colorize_depth(
    depth: np.ndarray, minimum: int, maximum: int, width: int
) -> np.ndarray:
    """Map depth in mm to a color image; near is red, far is blue, none is black."""
    height = round(depth.shape[0] * width / depth.shape[1])
    # Nearest keeps 0 (no depth) from bleeding into valid pixels.
    small = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    scale = 255.0 / max(maximum - minimum, 1)
    normalized = np.clip((small.astype(np.float32) - minimum) * scale, 0, 255)
    colored = cv2.applyColorMap((255 - normalized).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[small == 0] = 0
    return colored


def image_metadata(image: np.ndarray, encoding: str, timestamp: int) -> dict:
    """Return the metadata common to all outputs."""
    return {
        "encoding": encoding,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "primitive": "image",
        "capture_timestamp": timestamp,
    }


def main():
    """Grab ZED frames on each tick and send images and depth."""
    args = parse_args()
    camera = open_camera(args)
    parameters = camera_parameters(camera)
    print(f"ZED camera opened: {parameters}", flush=True)

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = args.confidence_threshold
    runtime.texture_confidence_threshold = args.texture_confidence_threshold
    runtime.enable_fill_mode = args.fill_mode

    left = sl.Mat()
    right = sl.Mat()
    depth = sl.Mat()

    node = Node()
    # cv2 releases the GIL while encoding, so the four encodes run in
    # parallel and fit in one frame at 30 fps.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    frame_count = 0
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] != "tick":
                continue
            # grab() paces us at the camera's FPS. A slightly faster tick, e.g.
            # 33 ms (30.3 Hz) against 30 fps, would pile up ticks until dora
            # drops them with a warning, so drop the extra ones here.
            if any(pending["type"] == "STOP" for pending in node.drain() or []):
                break

            status = camera.grab(runtime)
            if status != sl.ERROR_CODE.SUCCESS:
                if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    print("End of the SVO file", flush=True)
                    break
                print(f"ZED grab failed: {status}", file=sys.stderr, flush=True)
                continue

            timestamp = camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            camera.retrieve_image(left, sl.VIEW.LEFT_BGR)
            camera.retrieve_image(right, sl.VIEW.RIGHT_BGR)
            camera.retrieve_measure(depth, sl.MEASURE.DEPTH_U16_MM)
            # Views into the SDK's buffers; they stay valid until the next
            # retrieve, and we finish encoding before that.
            left_image = left.get_data()
            right_image = right.get_data()
            depth_image = depth.get_data()

            futures = {
                "image_left": executor.submit(
                    encode_jpeg, left_image, args.jpeg_quality
                ),
                "image_right": executor.submit(
                    encode_jpeg, right_image, args.jpeg_quality
                ),
                "depth": executor.submit(encode_png, depth_image, args.png_compression),
            }
            send_preview = args.preview_every > 0 and (
                frame_count % args.preview_every == 0
            )
            if send_preview:
                futures["depth_preview"] = executor.submit(
                    lambda: encode_jpeg(
                        colorize_depth(
                            depth_image,
                            args.depth_minimum,
                            args.preview_maximum,
                            args.preview_width,
                        ),
                        80,
                    )
                )
            frame_count += 1

            node.send_output(
                "image_left",
                pa.array(futures["image_left"].result()),
                image_metadata(left_image, "jpeg", timestamp),
            )
            node.send_output(
                "image_right",
                pa.array(futures["image_right"].result()),
                image_metadata(right_image, "jpeg", timestamp),
            )
            node.send_output(
                "depth",
                pa.array(futures["depth"].result()),
                {
                    **image_metadata(depth_image, "png", timestamp),
                    **parameters,
                    "depth_unit": "mm",
                },
            )
            if send_preview:
                preview = futures["depth_preview"].result()
                node.send_output(
                    "depth_preview",
                    pa.array(preview),
                    {
                        "encoding": "jpeg",
                        "width": args.preview_width,
                        "height": round(
                            depth_image.shape[0]
                            * args.preview_width
                            / depth_image.shape[1]
                        ),
                        "primitive": "image",
                        "capture_timestamp": timestamp,
                    },
                )
    finally:
        executor.shutdown()
        camera.close()


if __name__ == "__main__":
    main()
