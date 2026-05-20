from __future__ import annotations

import colorsys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
RECOLOR_DIR = BASE_DIR / "recolor_runtime"
RECOLOR_OUTPUT_DIR = RECOLOR_DIR / "outputs"
RECOLOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_hex_color(hex_color: str) -> tuple[float, float, float]:
    c = (hex_color or "").strip().lstrip("#")
    if len(c) != 6:
        raise ValueError("target_hex 必须是 6 位十六进制颜色，如 #FF5500")
    try:
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
    except Exception as exc:
        raise ValueError("target_hex 格式错误") from exc
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h, s, v


def _build_soft_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int, feather_px: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    if x1 <= x0 or y1 <= y0:
        return mask
    mask[y0:y1, x0:x1] = 1.0
    if feather_px <= 0:
        return mask

    yy, xx = np.mgrid[0:h, 0:w]
    dx = np.maximum(np.maximum(x0 - xx, 0), np.maximum(xx - (x1 - 1), 0)).astype(np.float32)
    dy = np.maximum(np.maximum(y0 - yy, 0), np.maximum(yy - (y1 - 1), 0)).astype(np.float32)
    dist = np.sqrt(dx * dx + dy * dy)
    outer = np.clip(1.0 - (dist / float(feather_px)), 0.0, 1.0)
    return np.maximum(mask, outer)


def recolor_region(
    file_bytes: bytes,
    suffix: str,
    target_hex: str,
    x_ratio: float,
    y_ratio: float,
    w_ratio: float,
    h_ratio: float,
    strength: float = 0.8,
    feather_ratio: float = 0.02,
) -> dict:
    target_h, target_s, _ = _parse_hex_color(target_hex)

    # load image from bytes safely
    from io import BytesIO

    img = Image.open(BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    h, w, _ = arr.shape

    x_ratio = float(np.clip(x_ratio, 0.0, 1.0))
    y_ratio = float(np.clip(y_ratio, 0.0, 1.0))
    w_ratio = float(np.clip(w_ratio, 0.01, 1.0))
    h_ratio = float(np.clip(h_ratio, 0.01, 1.0))
    strength = float(np.clip(strength, 0.0, 1.0))
    feather_ratio = float(np.clip(feather_ratio, 0.0, 0.25))

    x0 = int(round(x_ratio * w))
    y0 = int(round(y_ratio * h))
    bw = int(round(w_ratio * w))
    bh = int(round(h_ratio * h))
    x1 = min(w, x0 + bw)
    y1 = min(h, y0 + bh)
    feather_px = int(round(min(w, h) * feather_ratio))

    mask = _build_soft_mask(h, w, x0, y0, x1, y1, feather_px)
    alpha = mask * strength

    rgb_flat = arr.reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(*px) for px in rgb_flat], dtype=np.float32).reshape(h, w, 3)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    hue_diff = ((target_h - hue + 0.5) % 1.0) - 0.5
    new_hue = (hue + hue_diff * alpha) % 1.0
    new_sat = np.clip(sat * (1.0 - alpha) + (target_s + (1.0 - target_s) * 0.15) * alpha, 0.0, 1.0)

    hsv_new = np.stack([new_hue, new_sat, val], axis=-1).reshape(-1, 3)
    rgb_new = np.array([colorsys.hsv_to_rgb(*px) for px in hsv_new], dtype=np.float32).reshape(h, w, 3)

    out = np.clip(rgb_new * 255.0, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(out, mode="RGB")

    out_id = uuid.uuid4().hex
    out_path = RECOLOR_OUTPUT_DIR / f"{out_id}.jpg"
    out_img.save(out_path, format="JPEG", quality=92)

    return {
        "job_id": out_id,
        "recolored_url": f"/recolor-static/outputs/{out_path.name}",
        "bbox": {"x": x0, "y": y0, "w": max(1, x1 - x0), "h": max(1, y1 - y0)},
    }
