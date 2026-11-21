"""Patch target presets (configs/patch_presets.yaml): load, move within range, draw."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Preset:
    id: int
    name: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in image_size pixels


@dataclass(frozen=True)
class Presets:
    image_size: tuple[int, int]  # width, height
    step_px: float
    margin_px: float
    items: tuple[Preset, ...]
    # Which side of the patient head_left sees: "back" (the patient's left is the
    # image's left) or "front" (mirrored).
    view: str = "back"
    # Size of the grid cell each preset is shown as (w, h px), centered on its bbox;
    # None shows the bbox itself. The policy always gets the bbox: the cells only
    # keep a grid of trained-size boxes, which overlap, readable.
    cell: tuple[float, float] | None = None

    @classmethod
    def load(cls, path: Path) -> Presets:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        items = tuple(
            Preset(int(p["id"]), str(p.get("name", f"P{p['id']}")), tuple(float(v) for v in p["bbox"]))
            for p in data.get("presets") or []
        )
        if not items:
            raise ValueError(f"{path}: no presets")
        ids = [p.id for p in items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate preset ids")
        width, height = data.get("image_size", [1280, 720])
        view = str(data.get("view", "back")).strip().lower()
        if view not in ("back", "front"):
            raise ValueError(f"{path}: view must be back or front, not {view!r}")
        cell = data.get("cell_px")
        return cls(
            (int(width), int(height)),
            float(data.get("step_px", 30)),
            float(data.get("margin_px", 40)),
            items,
            view,
            None if cell is None else (float(cell[0]), float(cell[1])),
        )

    def shown(self, preset: Preset) -> tuple[float, float, float, float]:
        """What is drawn for a preset: its grid cell, or its bbox without one."""
        if self.cell is None:
            return preset.bbox
        cx, cy = (preset.bbox[0] + preset.bbox[2]) / 2, (preset.bbox[1] + preset.bbox[3]) / 2
        w, h = self.cell
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def sides_text(self) -> str:
        """How the patient's left and right map onto the image, for the VLM."""
        if self.view == "back":
            return (
                "カメラは貼られる人を背中側から見ています。人の左は画像の左側、人の右は画像の右側に写ります"
                "（鏡像ではありません）。"
            )
        return "カメラは貼られる人を正面から見ています。人の左は画像の右側、人の右は画像の左側に写ります。"

    def get(self, preset_id: int) -> Preset | None:
        return next((p for p in self.items if p.id == preset_id), None)

    def normalized(self, bbox) -> list[float]:
        w, h = self.image_size
        return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

    def pixels(self, bbox_normalized) -> list[float]:
        w, h = self.image_size
        b = bbox_normalized
        return [b[0] * w, b[1] * h, b[2] * w, b[3] * h]

    def center_range(self) -> tuple[float, float, float, float]:
        """Where a bbox center may go: the presets' centers' extent plus the margin."""
        cx = [(p.bbox[0] + p.bbox[2]) / 2 for p in self.items]
        cy = [(p.bbox[1] + p.bbox[3]) / 2 for p in self.items]
        m = self.margin_px
        return min(cx) - m, min(cy) - m, max(cx) + m, max(cy) + m

    def shift(self, bbox_px, dx_steps: float, dy_steps: float) -> tuple[list[float], bool]:
        """bbox moved by whole steps (+x right, +y down), center kept in range.
        Returns (bbox, clamped)."""
        x1, y1, x2, y2 = (float(v) for v in bbox_px)
        cx, cy = (x1 + x2) / 2 + dx_steps * self.step_px, (y1 + y2) / 2 + dy_steps * self.step_px
        lo_x, lo_y, hi_x, hi_y = self.center_range()
        ncx, ncy = min(max(cx, lo_x), hi_x), min(max(cy, lo_y), hi_y)
        hw, hh = (x2 - x1) / 2, (y2 - y1) / 2
        return [ncx - hw, ncy - hh, ncx + hw, ncy + hh], (ncx, ncy) != (cx, cy)

    def as_status(self) -> list[dict]:
        return [
            {
                "id": p.id,
                "name": p.name,
                "bbox": self.normalized(p.bbox),
                "shown": self.normalized(self.shown(p)),
            }
            for p in self.items
        ]


def annotate(jpeg: bytes, presets: Presets, current_px=None) -> bytes:
    """The head image with the presets as numbered boxes, or numbered grid cells (number
    in the middle) when the presets have a cell size, and the current bbox in green."""
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot decode the head image")
    h, w = image.shape[:2]
    sx, sy = w / presets.image_size[0], h / presets.image_size[1]

    def box(b, color, thickness):
        p1 = (round(b[0] * sx), round(b[1] * sy))
        p2 = (round(b[2] * sx), round(b[3] * sy))
        cv2.rectangle(image, p1, p2, color, thickness, cv2.LINE_AA)
        return p1

    for p in presets.items:
        shown = presets.shown(p)
        x, y = box(shown, (0, 200, 255), 2)
        label = str(p.id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        if presets.cell is not None:  # a grid: the number in the middle of its cell
            x = round((shown[0] + shown[2]) / 2 * sx) - tw // 2 - 4
            y = round((shown[1] + shown[3]) / 2 * sy) + th // 2 + 4
        cv2.rectangle(image, (x, y - th - 8), (x + tw + 8, y), (0, 200, 255), -1)
        cv2.putText(image, label, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    if current_px is not None:
        box(current_px, (0, 255, 0), 3)
    ok, out = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("cannot encode the annotated image")
    return out.tobytes()
