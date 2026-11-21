#!/usr/bin/env python
"""Small flow-matching policy for the OpenArm compress-patch task (fallback to MolmoAct2).

Observation: head_left RGB with the target bbox drawn on it (green, same convention as the
MolmoAct2 run), wrist_right RGB, 16-D joint state, and the bbox coordinates themselves.
Action: 30-step chunk of the 16-D absolute joint targets, normalised to [-1, 1] with q01/q99.

Architecture: two ImageNet ResNet18 trunks -> layer4 feature maps as tokens (+ state and bbox
tokens) -> transformer encoder; 30 action tokens -> transformer decoder (cross-attention to the
observation, adaLN-style time conditioning) -> velocity. Rectified flow: x_t = (1-t) noise +
t action, target v = action - noise; Euler integration at inference.

Target switch (reposition episodes): chunk steps at or after the next target segment's first
frame are masked out of the loss, so a frame showing box A is never supervised with motion to B.

    python small_flow.py train --cache ~/openarm-datasets/small-flow-cache --out runs/x
    policy = SmallFlowPolicy.load("runs/x/latest.pt"); actions = policy.predict(head, wrist, state, bbox)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

HORIZON = 30
ACT_DIM = 16
HEAD_HW = (144, 256)
WRIST_HW = (160, 256)
BBOX_REF = (1280.0, 720.0)
BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS_AT_1280 = 4.0
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def draw_bbox(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Same drawing rule as molmoact2_patches.draw_bbox (thickness scaled from 4 px at 1280 wide)."""
    out = np.ascontiguousarray(image).copy()
    if bbox is None or not np.all(np.isfinite(bbox)):
        return out
    h, w = out.shape[:2]
    sx, sy = w / BBOX_REF[0], h / BBOX_REF[1]
    thickness = max(1, int(round(BBOX_THICKNESS_AT_1280 * w / BBOX_REF[0])))
    p1 = (int(round(bbox[0] * sx)), int(round(bbox[1] * sy)))
    p2 = (int(round(bbox[2] * sx)), int(round(bbox[3] * sy)))
    cv2.rectangle(out, p1, p2, BBOX_COLOR, thickness, lineType=cv2.LINE_AA)
    return out


def bbox_features(bbox: np.ndarray) -> np.ndarray:
    """(…, 4) px -> normalised [cx, cy, w, h] in [0, 1]."""
    b = np.asarray(bbox, np.float32)
    cx = (b[..., 0] + b[..., 2]) / 2 / BBOX_REF[0]
    cy = (b[..., 1] + b[..., 3]) / 2 / BBOX_REF[1]
    w = (b[..., 2] - b[..., 0]) / BBOX_REF[0]
    h = (b[..., 3] - b[..., 1]) / BBOX_REF[1]
    return np.stack([cx, cy, w, h], -1)


class Normalizer:
    def __init__(self, q01, q99):
        self.lo = np.asarray(q01, np.float32)
        self.hi = np.asarray(q99, np.float32)

    def norm(self, x):
        # Clip to [-1, 1] like the MolmoAct2 robot processor (gripper action dims have q01 == q99).
        return np.clip(2 * (x - self.lo) / np.maximum(self.hi - self.lo, 1e-6) - 1, -1.0, 1.0).astype(np.float32)

    def unnorm(self, y):
        return ((np.asarray(y, np.float32) + 1) / 2 * np.maximum(self.hi - self.lo, 1e-6) + self.lo).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# data


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, cache: Path, indices: np.ndarray, state_norm, action_norm, augment: bool):
        self.head = np.load(cache / "head_left.npy", mmap_mode="r")
        self.wrist = np.load(cache / "wrist_right.npy", mmap_mode="r")
        low = np.load(cache / "lowdim.npz")
        self.state, self.action, self.bbox = low["state"], low["action"], low["bbox"]
        self.ep, self.fr, self.seg_end = low["episode_index"], low["frame_index"], low["seg_end"]
        n = len(self.ep)
        starts = np.r_[0, np.flatnonzero(np.diff(self.ep)) + 1]
        lengths = np.diff(np.r_[starts, n])
        self.ep_end = np.repeat(starts + lengths, lengths)  # exclusive global end index of episode
        self.indices = indices
        self.sn, self.an = state_norm, action_norm
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def _shift(self, img, rng, pad=8):
        if not self.augment:
            return img
        h, w = img.shape[:2]
        p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        y, x = rng.integers(0, 2 * pad + 1, 2)
        return p[y:y + h, x:x + w]

    def __getitem__(self, k):
        i = int(self.indices[k])
        rng = np.random.default_rng()
        head = self._shift(draw_bbox(self.head[i], self.bbox[i]), rng)
        wrist = self._shift(np.array(self.wrist[i]), rng)
        end = int(self.ep_end[i])
        steps = np.arange(i, i + HORIZON)
        valid = steps < end
        # Mask chunk steps that already belong to the next target segment.
        valid &= (self.fr[i] + np.arange(HORIZON)) < self.seg_end[i]
        idx = np.minimum(steps, end - 1)
        act = self.an.norm(self.action[idx])
        if self.augment:
            jitter = rng.uniform(0.9, 1.1, 3).astype(np.float32)
            head = np.clip(head.astype(np.float32) * jitter, 0, 255).astype(np.uint8)
            wrist = np.clip(wrist.astype(np.float32) * jitter, 0, 255).astype(np.uint8)
        return {
            "head": torch.from_numpy(np.ascontiguousarray(head)).permute(2, 0, 1),
            "wrist": torch.from_numpy(np.ascontiguousarray(wrist)).permute(2, 0, 1),
            "state": torch.from_numpy(self.sn.norm(self.state[i])),
            "bbox": torch.from_numpy(bbox_features(self.bbox[i])),
            "action": torch.from_numpy(act),
            "valid": torch.from_numpy(valid),
        }


# ---------------------------------------------------------------------------------------------
# model


def sinusoidal(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000 * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


class ResNetTokens(nn.Module):
    def __init__(self, d, hw):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        gh, gw = math.ceil(hw[0] / 32), math.ceil(hw[1] / 32)
        self.proj = nn.Linear(512, d)
        self.pos = nn.Parameter(torch.randn(1, gh * gw, d) * 0.02)

    def forward(self, img_u8):
        x = img_u8.float() / 255.0
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        f = self.trunk(x)  # B,512,gh,gw
        tok = self.proj(f.flatten(2).transpose(1, 2))
        return tok + self.pos[:, : tok.shape[1]]


class AdaLNDecoderLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d, elementwise_affine=False) for _ in range(3))
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.mod = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, mem, temb):
        s1, b1, s2, b2, s3, b3 = self.mod(temb)[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + s2) + b2
        x = x + self.ca(h, mem, mem, need_weights=False)[0]
        h = self.n3(x) * (1 + s3) + b3
        return x + self.ff(h)


class FlowPolicy(nn.Module):
    def __init__(self, d=384, heads=8, enc_layers=4, dec_layers=6):
        super().__init__()
        self.head_enc = ResNetTokens(d, HEAD_HW)
        self.wrist_enc = ResNetTokens(d, WRIST_HW)
        self.state_proj = nn.Sequential(nn.Linear(ACT_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.bbox_proj = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.randn(4, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, enc_layers)
        self.act_in = nn.Linear(ACT_DIM, d)
        self.act_pos = nn.Parameter(torch.randn(1, HORIZON, d) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.decoder = nn.ModuleList([AdaLNDecoderLayer(d, heads) for _ in range(dec_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.act_out = nn.Linear(d, ACT_DIM)
        self.d = d

    def encode(self, head, wrist, state, bbox):
        h = self.head_enc(head) + self.type_emb[0]
        w = self.wrist_enc(wrist) + self.type_emb[1]
        s = self.state_proj(state)[:, None] + self.type_emb[2]
        b = self.bbox_proj(bbox)[:, None] + self.type_emb[3]
        return self.encoder(torch.cat([h, w, s, b], 1))

    def velocity(self, mem, x_t, t):
        temb = self.t_mlp(sinusoidal(t, self.d))
        x = self.act_in(x_t) + self.act_pos
        for layer in self.decoder:
            x = layer(x, mem, temb)
        return self.act_out(self.out_norm(x))

    @torch.no_grad()
    def sample(self, mem, steps=10, noise=None):
        b = mem.shape[0]
        x = torch.randn(b, HORIZON, ACT_DIM, device=mem.device) if noise is None else noise
        for k in range(steps):
            t = torch.full((b,), k / steps, device=mem.device)
            x = x + self.velocity(mem, x, t) / steps
        return x


def flow_loss(model, batch, generator=None):
    mem = model.encode(batch["head"], batch["wrist"], batch["state"], batch["bbox"])
    a = batch["action"]
    b = a.shape[0]
    noise = torch.randn(a.shape, device=a.device, dtype=a.dtype, generator=generator)
    # t=0 is pure noise in this convention; Beta(1, 1.5) puts more weight near the noisy end
    # (same as MolmoAct2's sampler), where the coarse mode (which target) is decided.
    if generator is None:
        t = torch.distributions.Beta(1.0, 1.5).sample((b,)).to(a.device) * 0.999
    else:  # deterministic t for evaluation (inverse-CDF of Beta(1, 1.5): 1 - (1-u)^(1/1.5))
        u = torch.rand(b, device=a.device, generator=generator)
        t = (1 - (1 - u) ** (1 / 1.5)) * 0.999
    x_t = (1 - t)[:, None, None] * noise + t[:, None, None] * a
    v = model.velocity(mem, x_t, t)
    m = batch["valid"].float()[..., None]
    return (((v - (a - noise)) ** 2) * m).sum() / (m.sum() * ACT_DIM).clamp_min(1.0)


# ---------------------------------------------------------------------------------------------
# inference wrapper


class SmallFlowPolicy:
    """Load a checkpoint and predict a 30-step absolute joint chunk from raw camera frames."""

    def __init__(self, model, state_norm, action_norm, device):
        self.model, self.sn, self.an, self.device = model.eval(), state_norm, action_norm, device

    @classmethod
    def load(cls, path, device="cuda"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = FlowPolicy(**ck["model_args"])
        model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
        return cls(model.to(device), Normalizer(*ck["state_norm"]), Normalizer(*ck["action_norm"]), device)

    @torch.no_grad()
    def predict(self, head_left_rgb, wrist_right_rgb, state, bbox_xyxy_1280, steps=10):
        """head_left_rgb: HxWx3 uint8 RGB (any size, 16:9); wrist_right_rgb: HxWx3 uint8 RGB;
        state: (16,) joint state; bbox_xyxy_1280: target box in head_left 1280x720 px.
        Returns (30, 16) absolute joint targets."""
        head = cv2.resize(head_left_rgb, (HEAD_HW[1], HEAD_HW[0]), interpolation=cv2.INTER_AREA)
        head = draw_bbox(head, np.asarray(bbox_xyxy_1280, np.float32))
        wrist = cv2.resize(wrist_right_rgb, (WRIST_HW[1], WRIST_HW[0]), interpolation=cv2.INTER_AREA)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mem = self.model.encode(
                t(head).permute(2, 0, 1)[None], t(wrist).permute(2, 0, 1)[None],
                t(self.sn.norm(np.asarray(state, np.float32)))[None], t(bbox_features(np.asarray(bbox_xyxy_1280)))[None],
            )
            a = self.model.sample(mem.float(), steps=steps)
        return self.an.unnorm(a[0].float().cpu().numpy())


# ---------------------------------------------------------------------------------------------
# training


def evaluate(model, val_loader, device, max_batches=1000):
    model.eval()
    fl, mse, n = 0.0, 0.0, 0
    g = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for bi, batch in enumerate(val_loader):
            if bi >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            fl += float(flow_loss(model, batch, generator=g))
            mem = model.encode(batch["head"], batch["wrist"], batch["state"], batch["bbox"])
            noise = torch.randn(batch["action"].shape, device=device, generator=g)
            pred = model.sample(mem.float(), noise=noise)
            m = batch["valid"].float()[..., None]
            mse += float((((pred - batch["action"]) ** 2) * m).sum() / (m.sum() * ACT_DIM))
            n += 1
    model.train()
    return fl / max(n, 1), mse / max(n, 1)


@torch.no_grad()
def bbox_response(model, ds, device, n=128):
    """Swap the drawn bbox + bbox token of val frames for a bbox from a different frame and
    measure how much the sampled chunk changes (normalised units), with shared noise."""
    model.eval()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    other = rng.permutation(idx)
    items = [ds[int(k)] for k in idx]
    swapped = []
    for k, o in zip(idx, other):
        i, j = int(ds.indices[k]), int(ds.indices[o])
        head = draw_bbox(ds.head[i], ds.bbox[j])
        swapped.append((torch.from_numpy(np.ascontiguousarray(head)).permute(2, 0, 1), torch.from_numpy(bbox_features(ds.bbox[j]))))
    st = lambda key: torch.stack([it[key] for it in items]).to(device)
    noise = torch.randn(len(idx), HORIZON, ACT_DIM, device=device, generator=torch.Generator(device=device).manual_seed(0))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a1 = model.sample(model.encode(st("head"), st("wrist"), st("state"), st("bbox")).float(), noise=noise)
        a2 = model.sample(model.encode(torch.stack([s[0] for s in swapped]).to(device), st("wrist"), st("state"),
                                       torch.stack([s[1] for s in swapped]).to(device)).float(), noise=noise)
    shift = np.linalg.norm(bbox_features(ds.bbox[ds.indices[idx]])[:, :2] - bbox_features(ds.bbox[ds.indices[other]])[:, :2], axis=1)
    resp = (a1 - a2).abs().mean(dim=(1, 2)).cpu().numpy()
    model.train()
    far = shift > 0.05
    return float(resp[far].mean()) if far.any() else float("nan"), float(resp[~far].mean()) if (~far).any() else float("nan")


def train(a):
    device = torch.device("cuda")
    cache = a.cache.expanduser()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    stats = json.loads((Path(a.view).expanduser() / "meta" / "stats.json").read_text())
    sn = Normalizer(stats["observation.state"]["q01"], stats["observation.state"]["q99"])
    an = Normalizer(stats["action"]["q01"], stats["action"]["q99"])
    low = np.load(cache / "lowdim.npz")
    ep = low["episode_index"]
    val_eps = set(int(e) for e in a.val_episodes.split(",")) if a.val_episodes else set()
    all_idx = np.arange(len(ep))
    tr_idx = all_idx[~np.isin(ep, list(val_eps))]
    va_idx = all_idx[np.isin(ep, list(val_eps))]
    tr = ChunkDataset(cache, tr_idx, sn, an, augment=True)
    va = ChunkDataset(cache, va_idx, sn, an, augment=False) if len(va_idx) else None
    sampler = torch.utils.data.RandomSampler(tr, replacement=True, num_samples=a.steps * a.batch)
    loader = torch.utils.data.DataLoader(tr, a.batch, sampler=sampler, num_workers=a.workers, pin_memory=True,
                                         drop_last=True, persistent_workers=True, prefetch_factor=4)
    if va is not None:  # fixed evaluation subset (every 8th val frame), same for every checkpoint
        va_eval = ChunkDataset(cache, va_idx[::8], sn, an, augment=False)
        val_loader = torch.utils.data.DataLoader(va_eval, a.batch, shuffle=False, num_workers=4)
    else:
        val_loader = None

    model_args = dict(d=a.d, heads=8, enc_layers=a.enc_layers, dec_layers=a.dec_layers)
    model = FlowPolicy(**model_args).to(device)
    ema = FlowPolicy(**model_args).to(device)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    backbone = [p for n, p in model.named_parameters() if ".trunk." in n]
    rest = [p for n, p in model.named_parameters() if ".trunk." not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": a.lr * 0.1}, {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M train={len(tr_idx)} val={len(va_idx)}", flush=True)
    log = (out / "log.jsonl").open("a")
    t0 = time.time()
    skipped = 0
    model.train()
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow_loss(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            sched.step()
            if skipped > 50:
                raise RuntimeError(f"too many non-finite steps ({skipped}) at step {step}")
            continue
        opt.step()
        sched.step()
        with torch.no_grad():
            decay = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        if step % 50 == 0:
            rec = {"step": step, "loss": float(loss), "grad_norm": float(gn), "lr": sched.get_last_lr()[1], "skipped": skipped,
                   "sec_per_step": (time.time() - t0) / step}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            rec = {"step": step, "type": "eval"}
            if val_loader is not None:
                rec["val_flow_loss"], rec["val_chunk_mse"] = evaluate(ema, val_loader, device)
                rec["bbox_resp_far"], rec["bbox_resp_near"] = bbox_response(ema, va, device)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            ck = {"model": model.state_dict(), "ema": ema.state_dict(), "model_args": model_args, "step": step,
                  "state_norm": (sn.lo.tolist(), sn.hi.tolist()), "action_norm": (an.lo.tolist(), an.hi.tolist()),
                  "camera_hw": {"head_left": HEAD_HW, "wrist_right": WRIST_HW}}
            torch.save(ck, out / f"step{step}.pt")
            torch.save(ck, out / "latest.pt")
        if step >= a.steps:
            break
    print("done", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--view", default="~/openarm-datasets/molmoact2/webxr-pedestal-zed-lerobot-curated-molmoact2")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--d", type=int, default=384)
    t.add_argument("--enc_layers", type=int, default=4)
    t.add_argument("--dec_layers", type=int, default=6)
    t.add_argument("--ema", type=float, default=0.999)
    t.add_argument("--workers", type=int, default=12)
    t.add_argument("--eval_every", type=int, default=2000)
    t.add_argument("--val_episodes", default="")
    a = p.parse_args()
    if a.cmd == "train":
        train(a)


if __name__ == "__main__":
    main()
