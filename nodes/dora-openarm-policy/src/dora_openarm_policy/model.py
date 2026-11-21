"""Load the small flow policy from training/small_flow and run it on camera frames.

The model code is imported from the training script itself (``small_flow.py``), so
the network, the bbox drawing and the normalization are the ones it was trained
with, not a copy that can drift.

Inputs, as in training (training/small_flow/small_flow.py, build_cache.py):

* head_left: RGB. The ZED node sends JPEG (cv2 decodes to BGR, from the ZED's
  BGRA), so it is converted to RGB here. ``predict`` resizes it to 256x144 and
  draws the bbox on the resized image, as the cache-based training did.
* wrist_right: RGB, resized to 256x160.
* state: 16 joint positions, right 8 then left 8.
* bbox: x1, y1, x2, y2 in head_left 1280x720 pixels.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BBOX_REF = (1280.0, 720.0)


def load_small_flow_module(path: Path):
    spec = importlib.util.spec_from_file_location("small_flow", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["small_flow"] = module
    spec.loader.exec_module(module)
    return module


def decode_image(
    data: bytes | np.ndarray, encoding: str, width: int | None, height: int | None
) -> np.ndarray:
    """A camera message as HxWx3 uint8 RGB."""
    encoding = (encoding or "").lower()
    buf = (
        np.frombuffer(data, np.uint8) if isinstance(data, (bytes, bytearray)) else np.asarray(data, np.uint8)
    )
    if encoding in ("jpeg", "jpg", "png"):
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode a {encoding} image")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if width is None or height is None:
        raise ValueError(f"raw {encoding} image without width/height")
    channels = {"bgra8": 4, "rgba8": 4, "bgr8": 3, "rgb8": 3}.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported encoding {encoding!r}")
    image = buf.reshape(int(height), int(width), channels)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if encoding == "bgr8":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.copy()


class Policy:
    def __init__(self, checkpoint: Path, small_flow_py: Path, device: str = "cuda", steps: int = 10) -> None:
        import torch

        self.torch = torch
        self.sf = load_small_flow_module(small_flow_py)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.inner = self.sf.SmallFlowPolicy.load(str(checkpoint), device=device)
        self.steps = steps
        self.horizon = int(self.sf.HORIZON)
        ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        self.step_trained = ck.get("step")
        del ck

    def warm_up(self) -> float:
        head = np.zeros((720, 1280, 3), np.uint8)
        wrist = np.zeros((600, 960, 3), np.uint8)
        state = np.zeros(16, np.float32)
        bbox = np.array([500, 300, 700, 450], np.float32)
        for _ in range(3):
            t0 = time.monotonic()
            self.predict(head, wrist, state, bbox)
        return time.monotonic() - t0

    def predict(
        self,
        head_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state16: np.ndarray,
        bbox_1280: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        """``seed``: the flow's starting noise; the same seed gives the same chunk for
        the same observation (None: fresh noise)."""
        if seed is not None:
            self.torch.manual_seed(seed)
        chunk = self.inner.predict(
            head_rgb,
            wrist_rgb,
            np.asarray(state16, np.float32),
            np.asarray(bbox_1280, np.float32),
            steps=self.steps,
        )
        self.torch.cuda.synchronize()
        return np.asarray(chunk, np.float64)

    def model_view(self, head_rgb: np.ndarray, wrist_rgb: np.ndarray, bbox_1280: np.ndarray) -> np.ndarray:
        """What the network sees (head with the bbox drawn over wrist), as BGR for JPEG."""
        hh, hw = self.sf.HEAD_HW
        wh, ww = self.sf.WRIST_HW
        head = cv2.resize(head_rgb, (hw, hh), interpolation=cv2.INTER_AREA)
        head = self.sf.draw_bbox(head, np.asarray(bbox_1280, np.float32))
        wrist = cv2.resize(wrist_rgb, (ww, wh), interpolation=cv2.INTER_AREA)
        width = max(hw, ww)
        pad = lambda im: cv2.copyMakeBorder(im, 0, 0, 0, width - im.shape[1], cv2.BORDER_CONSTANT)
        view = np.vstack([pad(head), pad(wrist)])
        return cv2.cvtColor(view, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------------------------
# MolmoAct2 (training/molmoact2): the release code's LeRobot policy on a merged checkpoint.

# Exactly what training used (molmoact2_patches.task_instruction with the default green
# overlay): passed through as is, no normalization.
MOLMOACT2_INSTRUCTION = "Apply the compress patch at the location marked by the green box."
MOLMOACT2_NORM_TAG = "openarm_bimanual_bbox"
# molmoact2_patches.draw_bbox: green, 4 px at 1280 px width, anti-aliased, on the
# original-resolution head_left image before any resizing.
MOLMOACT2_BBOX_COLOR = (0, 255, 0)
MOLMOACT2_BBOX_THICKNESS_AT_1280 = 4.0


def draw_bbox_molmoact2(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(image).copy()
    if bbox is None or not np.all(np.isfinite(bbox)):
        return out
    h, w = out.shape[:2]
    sx, sy = w / BBOX_REF[0], h / BBOX_REF[1]
    # Written as in molmoact2_patches.draw_bbox, on purpose.
    thickness = max(1, int(round(MOLMOACT2_BBOX_THICKNESS_AT_1280 * w / BBOX_REF[0])))  # noqa: RUF046
    p1 = (int(round(bbox[0] * sx)), int(round(bbox[1] * sy)))  # noqa: RUF046
    p2 = (int(round(bbox[2] * sx)), int(round(bbox[3] * sy)))  # noqa: RUF046
    cv2.rectangle(out, p1, p2, MOLMOACT2_BBOX_COLOR, thickness, lineType=cv2.LINE_AA)
    return out


def _load_into_bf16(checkpoint_dir, model) -> None:
    """Replaces olmo's load_model_state for one merged checkpoint: the fp32 state dict
    is memory-mapped (not read into RAM, 21 GB) and copied into bf16 parameters on the
    GPU tensor by tensor (11 GB instead of 21 GB of VRAM). Buffers stay fp32."""
    import torch

    for p in model.parameters():
        if p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    path = Path(checkpoint_dir) / "model.pt"
    if not path.exists():
        candidates = sorted(Path(checkpoint_dir).glob("model*.pt"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"{checkpoint_dir}: expected one model*.pt, found {candidates}")
        path = candidates[0]
    state = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state)
    del state


class MolmoAct2Runner:
    """Same interface as Policy: predict(head, wrist, state16, bbox) -> (30, 16)."""

    def __init__(
        self, checkpoint: Path, experiments: Path, device: str = "cuda", steps: int | None = None
    ) -> None:
        import torch

        self.torch = torch
        for p in (experiments, experiments / "lerobot" / "src"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        os.environ.setdefault("MOLMO_DATA_DIR", str(Path.home() / ".cache" / "molmo"))
        from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
        from olmo.train import checkpointer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        original = checkpointer.load_model_state
        checkpointer.load_model_state = lambda d, model, cfg=None: _load_into_bf16(d, model)
        # The model is built on the meta device and then materialized with to_empty:
        # make its parameters bf16 while they are still meta, so the GPU never holds
        # the fp32 copy (22 GB peak otherwise).
        original_to_empty = torch.nn.Module.to_empty

        def to_empty_bf16(module, *args, **kwargs):
            for p in module.parameters():
                if p.is_meta and p.dtype == torch.float32:
                    p.data = p.data.to(torch.bfloat16)
            return original_to_empty(module, *args, **kwargs)

        torch.nn.Module.to_empty = to_empty_bf16
        try:
            config = MolmoAct2Config(
                checkpoint_path=str(checkpoint),
                device=device,
                norm_tag=MOLMOACT2_NORM_TAG,
                num_steps=steps,
                enable_depth_reasoning=False,
                enable_inference_cuda_graph=False,
            )
            self.inner = MolmoAct2Policy(config)
        finally:
            checkpointer.load_model_state = original
            torch.nn.Module.to_empty = original_to_empty
        self.horizon = int(self.inner._handles.action_horizon)
        self.step_trained = Path(checkpoint).name
        self.steps = steps

    def warm_up(self) -> float:
        head = np.zeros((720, 1280, 3), np.uint8)
        wrist = np.zeros((600, 960, 3), np.uint8)
        state = np.zeros(16, np.float32)
        bbox = np.array([500, 300, 700, 450], np.float32)
        for _ in range(2):
            t0 = time.monotonic()
            self.predict(head, wrist, state, bbox)
        return time.monotonic() - t0

    def predict(
        self,
        head_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state16: np.ndarray,
        bbox_1280: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        torch = self.torch
        generator = None if seed is None else torch.Generator(device="cuda").manual_seed(seed)
        head = draw_bbox_molmoact2(head_rgb, np.asarray(bbox_1280, np.float32))
        observation = {
            # Insertion order is the camera order: head_left, then wrist_right (as trained).
            "observation.images.head_left": head,
            "observation.images.wrist_right": np.ascontiguousarray(wrist_rgb),
            "observation.state": np.asarray(state16, np.float32),
            "task": MOLMOACT2_INSTRUCTION,
        }
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            actions, _, _ = self.inner.generate_action_chunk_from_observations(
                observation, norm_tag=MOLMOACT2_NORM_TAG, num_steps=self.steps, generator=generator
            )
        torch.cuda.synchronize()
        chunk = actions.float().cpu().numpy() if hasattr(actions, "cpu") else np.asarray(actions, np.float32)
        return np.asarray(chunk, np.float64).reshape(-1, chunk.shape[-1])[:, :16]

    def model_view(self, head_rgb: np.ndarray, wrist_rgb: np.ndarray, bbox_1280: np.ndarray) -> np.ndarray:
        head = draw_bbox_molmoact2(head_rgb, np.asarray(bbox_1280, np.float32))
        head = cv2.resize(head, (320, 180), interpolation=cv2.INTER_AREA)
        wrist = cv2.resize(wrist_rgb, (320, 200), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(np.vstack([head, wrist]), cv2.COLOR_RGB2BGR)
