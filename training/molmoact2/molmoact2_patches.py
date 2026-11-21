"""Runtime patches that adapt the MolmoAct2 release trainer to OpenArm bbox-conditioned
depth fine-tuning. Nothing under ``lib/molmoact2`` is edited; ``train.py`` imports this
module before the trainer builds its datasets and model.

Patches
-------
1. **bbox overlay** (``LeRobotDatasetWrapper._extract_image``): the target bbox stored
   in the ``target_bbox`` column is drawn on the front camera (``head_left``) frame
   before resizing and image preprocessing, so augmentation moves it with the image.
2. **depth supervision in continuous mode** (``LeRobotDatasetWrapper.get``): the
   release wrapper drops the depth-token string when ``--action_format=continuous``
   (the default, and the README's depth fine-tuning recipe), so depth reasoning would
   get no teacher forcing and no CE loss. The depth string is restored as the
   response. ``MolmoAct2._get_encoder_attention_mask`` additionally masks the EOS
   that now ends the response, matching inference where the action expert runs right
   after the depth span.
3. **bbox-action binding loss** (ListenVLA, arXiv:2610.16313, Eqs. 4-6, with the
   audio history replaced by the bbox drawn on the front image):
   - the loader picks a context-matched donor ``j`` for the sampled frame ``i``
     (``mine_binding_pairs.py``) and re-draws the donor's bbox on frame ``i``'s raw
     front image; both versions go through the same image preprocessing with the same
     RNG state, so they differ only in the drawn bbox;
   - after the regular forward/backward of each optimizer step, both versions are run
     through the policy at the pure-noise endpoint ``t = 0`` with a shared Gaussian
     sample, and the velocity difference is regressed onto the normalized
     demonstration difference ``A_i - A_j`` over the first ``steps`` actions,
     weighted by the detached pair weight ``w_ij``. The loss is averaged over all rows
     of the micro-batch (rows without a donor contribute zero), and its backward runs
     separately so peak memory does not grow by the binding branch.

Configuration (environment variables, read in every process):
  MOLMOACT2_BINDING_WEIGHT   lambda for the binding loss (default 0.5, i.e. the
                             2:1 flow:binding ratio of ListenVLA Eq. 6); 0 disables it
  MOLMOACT2_BINDING_PROB     probability of attaching a donor when the frame has one (1.0)
  MOLMOACT2_BINDING_PAIRS    path to binding_pairs.parquet (default: <dataset>/meta/)
  MOLMOACT2_BBOX_CAMERA      camera key that gets the overlay (observation.images.head_left)
  MOLMOACT2_BBOX_COLOR       R,G,B of the overlay (0,255,0)
  MOLMOACT2_BBOX_THICKNESS   line width in px at 1280 px image width (4)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent.parent / "lib" / "molmoact2" / "experiments"
for _p in (EXPERIMENTS, EXPERIMENTS / "lerobot" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from olmo.data import dataset as olmo_dataset  # noqa: E402
from olmo.data import lerobot_wrapper as lw  # noqa: E402
from olmo.extra_tokens import build_discrete_depth_string  # noqa: E402
from olmo.models.molmoact2.molmoact2 import MolmoAct2  # noqa: E402
from olmo.preprocessing import multimodal_collator as mmc  # noqa: E402
from olmo.train import trainer as olmo_trainer  # noqa: E402

log = logging.getLogger(__name__)

BBOX_KEY = "target_bbox"
BBOX_REF_SIZE = (1280.0, 720.0)  # head_left pixel frame the bbox annotations live in
BINDING_KEYS = ("_binding_images", "_binding_valid", "_binding_delta", "_binding_mask", "_binding_weight")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def binding_weight() -> float:
    return _env_float("MOLMOACT2_BINDING_WEIGHT", 0.5)


def bbox_camera() -> str:
    return os.environ.get("MOLMOACT2_BBOX_CAMERA", "observation.images.head_left")


# ---------------------------------------------------------------------------
# 1. bbox overlay
# ---------------------------------------------------------------------------


def draw_bbox(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Draw ``bbox`` (x_min, y_min, x_max, y_max in 1280x720 px) on an HWC uint8 RGB image."""
    out = np.ascontiguousarray(image).copy()
    if bbox is None or not np.all(np.isfinite(bbox)):
        return out
    h, w = out.shape[:2]
    sx, sy = w / BBOX_REF_SIZE[0], h / BBOX_REF_SIZE[1]
    color = tuple(int(c) for c in os.environ.get("MOLMOACT2_BBOX_COLOR", "0,255,0").split(","))
    thickness = max(1, int(round(_env_float("MOLMOACT2_BBOX_THICKNESS", 4.0) * w / BBOX_REF_SIZE[0])))
    p1 = (int(round(bbox[0] * sx)), int(round(bbox[1] * sy)))
    p2 = (int(round(bbox[2] * sx)), int(round(bbox[3] * sy)))
    cv2.rectangle(out, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    return out


def _bbox_from_example(example: Dict[str, Any]) -> Optional[np.ndarray]:
    value = example.get(BBOX_KEY)
    if value is None:
        return None
    arr = lw._to_numpy(value).astype(np.float32).reshape(-1, 4)
    return arr[-1]


def _patched_extract_image(self, example, keys_to_use):
    available_keys = [key for key in example if key.startswith("observation.images")]
    if not keys_to_use:
        raise ValueError("No camera keys available in the LeRobot dataset")
    missing_keys = [key for key in keys_to_use if key not in example]
    if missing_keys:
        raise ValueError(
            f"Missing required camera keys {missing_keys} for repo '{self._get_repo_id()}'. "
            f"Available keys: {available_keys}"
        )
    bbox = _bbox_from_example(example)
    front = bbox_camera()
    self._binding_front = None
    all_images: List[np.ndarray] = []
    for key in keys_to_use:
        frames = lw._collect_image_frames(example[key])
        if key == front and bbox is not None:
            # n_obs_steps=1: the last frame is the current observation.
            self._binding_front = (len(all_images) + len(frames) - 1, frames[-1].copy())
            frames = [draw_bbox(f, bbox) for f in frames]
        all_images.extend(frames)
    if not all_images:
        raise ValueError("Failed to extract any camera images from LeRobot example")
    return all_images[0] if len(all_images) == 1 else all_images


# ---------------------------------------------------------------------------
# 2 + 3a. depth response restore and binding counterfactual in the loader
# ---------------------------------------------------------------------------


class _PairTable:
    """Donor candidates per anchor (episode, frame), loaded lazily per worker."""

    def __init__(self, path: Path):
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        meta = json.loads((table.schema.metadata or {}).get(b"binding", b"{}"))
        self.steps = int(meta.get("steps", 8))
        self.action_dim = int(meta.get("action_dim", 16))
        cols = table.to_pydict()
        self.by_anchor: Dict[Tuple[int, int], List[Tuple[np.ndarray, np.ndarray, float]]] = {}
        for i in range(table.num_rows):
            key = (int(cols["anchor_episode"][i]), int(cols["anchor_frame"][i]))
            self.by_anchor.setdefault(key, []).append((
                np.asarray(cols["donor_bbox"][i], np.float32),
                np.asarray(cols["donor_action"][i], np.float32).reshape(self.steps, self.action_dim),
                float(cols["weight"][i]),
            ))
        log.info("binding pairs: %d anchors, %d rows from %s", len(self.by_anchor), table.num_rows, path)

    def sample(self, key, rng) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        donors = self.by_anchor.get(key)
        if not donors:
            return None
        weights = np.asarray([d[2] for d in donors], np.float64)
        if weights.sum() <= 0:
            return None
        idx = rng.choice(len(donors), p=weights / weights.sum())
        return donors[int(idx)]


def _pair_table(self) -> Optional[_PairTable]:
    if getattr(self, "_binding_pairs_loaded", False):
        return self._binding_pairs
    self._binding_pairs_loaded = True
    self._binding_pairs = None
    path = os.environ.get("MOLMOACT2_BINDING_PAIRS")
    if not path:
        root = self._get_dataset_root_path()
        path = str(root / "meta" / "binding_pairs.parquet") if root is not None else ""
    if path and Path(path).exists():
        self._binding_pairs = _PairTable(Path(path))
    elif binding_weight() > 0:
        log.warning("binding loss enabled but no pair table at %r; binding is off for %s", path, self._get_repo_id())
    return self._binding_pairs


_orig_get = lw.LeRobotDatasetWrapper.get
_orig_getstate = lw.LeRobotDatasetWrapper.__getstate__


def _patched_getstate(self):
    state = _orig_getstate(self)
    for key in ("_binding_pairs", "_binding_pairs_loaded", "_binding_front"):
        state.pop(key, None)
    return state


def _patched_get(self, item, rng, **kwargs):
    out = _orig_get(self, item, rng, **kwargs)

    # Restore the depth-token response that the release wrapper drops in continuous mode.
    codes = out.get("depth_buffer_codes")
    if self.action_format == "continuous" and codes is not None and "messages" in out:
        depth_string = build_discrete_depth_string(codes, num_depth_tokens=self.num_depth_tokens)
        out["messages"] = {**out["messages"], "prompt_only": False, "answers": depth_string}

    front = getattr(self, "_binding_front", None)
    self._binding_front = None
    meta = out.get("metadata") or {}
    if (
        binding_weight() <= 0
        or front is None
        or "action" not in out
        or str(self.split).lower() != "train"
        or rng.random() >= _env_float("MOLMOACT2_BINDING_PROB", 1.0)
    ):
        return out
    table = _pair_table(self)
    if table is None:
        return out
    donor = table.sample((int(meta.get("episode_index", -1)), int(meta.get("frame_index", -1))), rng)
    if donor is None:
        return out
    donor_bbox, donor_action_raw, weight = donor

    repo_id = self._get_repo_id()
    donor_action = donor_action_raw
    if self.robot_processor is not None:
        donor_action = np.asarray(self.robot_processor.normalize_action(donor_action_raw, repo_id), np.float32)
    action = np.asarray(out["action"], np.float32)  # (max_horizon, max_dim), normalized + padded
    steps, real_dim = donor_action.shape
    delta = np.zeros_like(action)
    mask = np.zeros(action.shape, dtype=bool)
    delta[:steps, :real_dim] = action[:steps, :real_dim] - donor_action
    mask[:steps, :real_dim] = True
    horizon_pad = out.get("action_horizon_is_pad")
    if horizon_pad is not None:
        mask[:steps] &= ~np.asarray(horizon_pad[:steps], bool)[:, None]
    dim_pad = out.get("action_dim_is_pad")
    if dim_pad is not None:
        mask &= ~np.asarray(dim_pad, bool)[None, :]

    images = out["image"] if isinstance(out["image"], list) else [out["image"]]
    position, raw_front = front
    cf_front = draw_bbox(raw_front, donor_bbox)
    resize = lw._parse_image_resize(os.environ.get("LEROBOT_IMAGE_RESIZE"))
    if resize is not None:
        cf_front = lw._resize_image_array(cf_front, resize)
    cf_images = list(images)
    cf_images[position] = cf_front
    out["_binding"] = {
        "image": cf_images if isinstance(out["image"], list) else cf_images[0],
        "delta": delta,
        "mask": mask,
        "weight": float(weight),
    }
    return out


def _patched_deterministic_get(self, idx, epoch=0):
    rng = np.random.RandomState((self.seed * 195172 + idx + len(self.dataset) * epoch) % (2**32 - 1))
    if idx >= len(self.dataset):
        item = dict(self.dataset.get(0, rng))
        metadata = item.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else ({} if metadata is None else None)
        if metadata is not None:
            metadata["valid"] = False
            item["metadata"] = metadata
    else:
        item = dict(self.dataset.get(idx, rng))
    metadata = item.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else ({} if metadata is None else None)
    if metadata is not None and self.dataset_name is not None and "dataset_name" not in metadata:
        metadata["dataset_name"] = self.dataset_name
    if metadata is not None:
        item["metadata"] = metadata
    if self.weighting:
        item["weight"] = self.weighting
    binding = item.pop("_binding", None)
    if not self.preprocessor:
        return item
    cf_item = None
    if binding is not None:
        cf_item = copy.deepcopy(item)
        cf_item["image"] = binding["image"]
    state = rng.get_state()
    out = self.preprocessor(item, rng, **self.preprocessor_kwargs)
    is_lerobot = "actions" in out or "action" in item or "depth_buffer_codes" in out
    if cf_item is not None:
        cf_rng = np.random.RandomState()
        cf_rng.set_state(state)
        cf_out = self.preprocessor(cf_item, cf_rng, **self.preprocessor_kwargs)
        same_layout = (
            np.array_equal(np.asarray(out["input_tokens"]), np.asarray(cf_out["input_tokens"]))
            and np.asarray(out["images"]).shape == np.asarray(cf_out["images"]).shape
        )
        if same_layout:
            out["_binding_images"] = cf_out["images"]
            out["_binding_valid"] = np.array(True)
            out["_binding_delta"] = binding["delta"].astype(np.float32)
            out["_binding_mask"] = binding["mask"]
            out["_binding_weight"] = np.float32(binding["weight"])
            return out
        log.warning("binding counterfactual changed the token layout; dropping it for this sample")
    if is_lerobot and "images" in out and "actions" in out:
        actions = np.asarray(out["actions"])
        out["_binding_images"] = out["images"]
        out["_binding_valid"] = np.array(False)
        out["_binding_delta"] = np.zeros(actions.shape[-2:], np.float32)
        out["_binding_mask"] = np.zeros(actions.shape[-2:], bool)
        out["_binding_weight"] = np.float32(0.0)
    return out


# ---------------------------------------------------------------------------
# 3b. collate the binding side channel
# ---------------------------------------------------------------------------

_orig_collate_call = mmc.MMCollator.__call__


def _patched_collate_call(self, batch):
    has_binding = all("_binding_valid" in ex for ex in batch)
    side = [{k: ex.pop(k, None) for k in BINDING_KEYS} for ex in batch]
    out = _orig_collate_call(self, batch)
    if not has_binding or "images" not in out:
        return out
    spec = self.shapes_to_pad_to.get("images")
    pad_value = 0 if spec is not None and spec.dtype == np.uint8 else -1
    if spec is not None and isinstance(spec, mmc.VariablePaddingSpec):
        pad = None
    else:
        pad = self.pad
    out["_binding_images"] = mmc._collate(
        [s["_binding_images"] for s in side],
        spec.shape if spec is not None else None,
        dtype=spec.dtype if spec is not None else None,
        pad=pad,
        pad_value=pad_value,
        allow_truncate=False,
    )
    if out["_binding_images"].shape != out["images"].shape:
        raise RuntimeError(
            f"binding images {tuple(out['_binding_images'].shape)} != images {tuple(out['images'].shape)}"
        )
    out["_binding_valid"] = torch.as_tensor(np.stack([np.asarray(s["_binding_valid"]) for s in side]), dtype=torch.bool)
    out["_binding_delta"] = torch.as_tensor(np.stack([s["_binding_delta"] for s in side]), dtype=torch.float32)
    out["_binding_mask"] = torch.as_tensor(np.stack([s["_binding_mask"] for s in side]), dtype=torch.bool)
    out["_binding_weight"] = torch.as_tensor(np.stack([s["_binding_weight"] for s in side]), dtype=torch.float32)
    return out


# ---------------------------------------------------------------------------
# 2b + 3c. model: EOS mask and the t=0 velocity path used by the binding loss
# ---------------------------------------------------------------------------

_orig_encoder_mask = MolmoAct2._get_encoder_attention_mask
_orig_forward = MolmoAct2.forward

_BACKBONE_KEYS = (
    "input_ids", "input_embeddings", "attention_mask", "attention_bias", "response_mask", "subsegment_ids",
    "position_ids", "labels", "loss_masks", "images", "image_masks", "token_pooling", "low_res_token_pooling",
    "num_images", "multimodal_type", "num_image_starts", "response_logits_only", "past_key_values", "use_cache",
    "last_logits_only", "append_last_valid_logits",
)


def _patched_encoder_mask(self, input_ids, attention_mask):
    mask = _orig_encoder_mask(self, input_ids, attention_mask)
    if (
        mask is not None
        and input_ids is not None
        and self.config.action_format == "continuous"
        and getattr(self.config, "enable_depth_reasoning", False)
        and getattr(self, "_eos_token_id", None) is not None
    ):
        mask = mask & (input_ids != self._eos_token_id)
    return mask


def _patched_forward(self, *args, binding_noise=None, binding_action_horizon_is_pad=None,
                     binding_action_dim_is_pad=None, **kwargs):
    if binding_noise is None:
        return _orig_forward(self, *args, **kwargs)
    if args:
        raise TypeError("binding forward expects keyword arguments only")
    action_expert = self._require_action_expert()
    states = kwargs.pop("states", None)
    forward_kwargs = {k: kwargs[k] for k in _BACKBONE_KEYS if k in kwargs}
    forward_kwargs.setdefault("response_logits_only", False)
    base_output, _, layer_kv_states = self._run_backbone(
        collect_layer_hidden_states=False,
        collect_layer_kv_states=True,
        collect_layer_input_states=False,
        output_hidden_states=False,
        **forward_kwargs,
    )
    input_ids = forward_kwargs["input_ids"]
    encoder_attention_mask = self._get_encoder_attention_mask(input_ids, forward_kwargs.get("attention_mask"))
    kv = self._select_layer_kv_states(layer_kv_states)
    depth_gate, depth_mask = self._depth_gate_from_condition(
        input_ids=input_ids, encoder_attention_mask=encoder_attention_mask,
        layer_states=None, layer_kv_states=kv,
    )
    kv = self._apply_depth_gate_to_layer_kv_states(kv, depth_mask, depth_gate)
    _, kv = self._maybe_detach_action_expert_condition(None, kv)
    batch = binding_noise.shape[0]
    rows = torch.arange(batch, device=binding_noise.device)
    chunk_mask = self._chunk_attention_mask(
        encoder_attention_mask, forward_kwargs.get("subsegment_ids"), rows, torch.zeros_like(rows)
    )
    mask_dims = lambda x: self._mask_action_dim_tensor(
        x, action_dim_is_pad=binding_action_dim_is_pad, enabled=self.config.mask_action_dim_padding
    )
    noise = mask_dims(binding_noise)
    action_mask = None if binding_action_horizon_is_pad is None else ~binding_action_horizon_is_pad.to(torch.bool)
    velocity = action_expert(
        noise,
        torch.zeros(batch, device=noise.device, dtype=noise.dtype),
        encoder_kv_states=kv,
        encoder_attention_mask=chunk_mask,
        action_attention_mask=action_mask,
        state_embeddings=states,
    )
    internal = dict(base_output.internal or {})
    internal["binding_velocity"] = mask_dims(velocity)
    return base_output._replace(internal=internal)


# ---------------------------------------------------------------------------
# 3d. trainer: binding forward/backward after the regular micro-batches
# ---------------------------------------------------------------------------

_orig_train_batch = olmo_trainer.Trainer.train_batch
_orig_train_step = olmo_trainer.Trainer.train_step
_MODEL_EXCLUDE = {
    "metadata", "loss_masks", "labels", "actions", "action_horizon_is_pad", "action_is_pad", "action_dim_is_pad",
    "packed_batch_idx", "packed_example_ids", "packed_action_chunk_is_valid", "packed_num_chunks",
    "packed_action_chunk_cap", "packed_action_chunk_overflow", "data_worker_state",
} | set(olmo_trainer._DEPTH_SUPERVISION_BATCH_KEYS)


def _binding_loss(self, micro_batch) -> Optional[Tuple[torch.Tensor, Dict[str, float]]]:
    valid = micro_batch.get("_binding_valid")
    if valid is None or not bool(valid.any()):
        return None
    rows = valid.nonzero(as_tuple=False).flatten()
    batch_rows = int(valid.shape[0])
    model_inputs = {}
    for key, value in micro_batch.items():
        if key in _MODEL_EXCLUDE or key.startswith("_") or not isinstance(value, torch.Tensor):
            continue
        if value.ndim == 0 or value.shape[0] != batch_rows:
            model_inputs[key] = value
            continue
        picked = value.index_select(0, rows)
        if key == "images":
            cf = micro_batch["_binding_images"].index_select(0, rows)
            model_inputs[key] = torch.cat([picked, cf], dim=0)
        else:
            model_inputs[key] = torch.cat([picked, picked], dim=0)
    actions = micro_batch["actions"]
    n = rows.numel()
    noise = torch.randn((n, *actions.shape[1:]), device=actions.device, dtype=actions.dtype)
    horizon_pad = micro_batch.get("action_horizon_is_pad")
    dim_pad = micro_batch.get("action_dim_is_pad")
    pick2 = lambda t: None if t is None else torch.cat([t.index_select(0, rows)] * 2, dim=0)
    response_mask = micro_batch["loss_masks"].index_select(0, rows) > 0
    with torch.autocast("cuda", dtype=self.cfg.autocast_precision):
        out = self.fsdp_model(
            **model_inputs,
            response_mask=torch.cat([response_mask, response_mask], dim=0),
            response_logits_only=self.cfg.response_logits_only,
            binding_noise=torch.cat([noise, noise], dim=0),
            binding_action_horizon_is_pad=pick2(horizon_pad),
            binding_action_dim_is_pad=pick2(dim_pad),
        )
    velocity = out.internal["binding_velocity"].float()
    delta_v = velocity[:n] - velocity[n:]
    target = micro_batch["_binding_delta"].index_select(0, rows).float()
    mask = micro_batch["_binding_mask"].index_select(0, rows).float()
    weight = micro_batch["_binding_weight"].index_select(0, rows).float().detach()
    per_pair = ((delta_v - target) ** 2 * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
    loss = (weight * per_pair).sum() / max(1, batch_rows)
    # Diagnostics: how much the policy currently reacts to the bbox swap vs. the demonstrated difference.
    with torch.no_grad():
        resp = ((delta_v * mask).abs().sum() / mask.sum().clamp_min(1.0)).item()
        tgt = ((target * mask).abs().sum() / mask.sum().clamp_min(1.0)).item()
        cos = torch.nn.functional.cosine_similarity((delta_v * mask).flatten(1), (target * mask).flatten(1), dim=1)
    stats = {
        "binding/pair_loss": float(per_pair.mean().item()),
        "binding/pairs": float(n),
        "binding/rows": float(batch_rows),
        "binding/response_abs": resp,
        "binding/target_abs": tgt,
        "binding/cosine": float(cos.mean().item()),
    }
    return loss, stats


def _patched_train_batch(self, batch, compute_metrics):
    lam = binding_weight()
    micro_batches = self.split_batch(batch) if lam > 0 and "_binding_valid" in batch else []
    total = _orig_train_batch(self, batch, compute_metrics)
    self._binding_stats = {}
    if not micro_batches:
        return total
    sums: Dict[str, float] = {}
    loss_sum = 0.0
    for micro_batch in micro_batches:
        result = _binding_loss(self, micro_batch)
        if result is None:
            continue
        loss, stats = result
        scaled = lam * loss / len(micro_batches)
        scaled.backward()
        total = total + scaled.detach()
        loss_sum += float(loss.detach().item()) / len(micro_batches)
        for k, v in stats.items():
            sums[k] = sums.get(k, 0.0) + (v if k in {"binding/pairs", "binding/rows"} else v / len(micro_batches))
    self._binding_stats = {"binding/loss": loss_sum, **sums}
    return total


def _patched_train_step(self, batch, compute_metrics=True):
    metrics = _orig_train_step(self, batch, compute_metrics)
    stats = getattr(self, "_binding_stats", None)
    if stats and metrics is not None:
        metrics.update({f"train/{k}": v for k, v in stats.items()})
    return metrics


# ---------------------------------------------------------------------------


def apply() -> None:
    if getattr(lw.LeRobotDatasetWrapper, "_openarm_patched", False):
        return
    lw.LeRobotDatasetWrapper._extract_image = _patched_extract_image
    lw.LeRobotDatasetWrapper.get = _patched_get
    lw.LeRobotDatasetWrapper.__getstate__ = _patched_getstate
    lw.LeRobotDatasetWrapper._openarm_patched = True
    olmo_dataset.DeterministicDataset.get = _patched_deterministic_get
    mmc.MMCollator.__call__ = _patched_collate_call
    MolmoAct2._get_encoder_attention_mask = _patched_encoder_mask
    MolmoAct2.forward = _patched_forward
    olmo_trainer.Trainer.train_batch = _patched_train_batch
    olmo_trainer.Trainer.train_step = _patched_train_step
