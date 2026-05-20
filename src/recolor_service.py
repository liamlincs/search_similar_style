from __future__ import annotations

import colorsys
import base64
import json
import uuid
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter

BASE_DIR = Path(__file__).resolve().parent
RECOLOR_DIR = BASE_DIR / "recolor_runtime"
RECOLOR_OUTPUT_DIR = RECOLOR_DIR / "outputs"
RECOLOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/images/generations"


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


def _apply_hsv_retarget(
    arr: np.ndarray,
    target_hex: str,
    mask: np.ndarray,
    strength: float,
    sat_boost: float = 0.15,
) -> np.ndarray:
    target_h, target_s, _ = _parse_hex_color(target_hex)
    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0) * float(np.clip(strength, 0.0, 1.0))

    h, w, _ = arr.shape
    rgb_flat = arr.reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(*px) for px in rgb_flat], dtype=np.float32).reshape(h, w, 3)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    hue_diff = ((target_h - hue + 0.5) % 1.0) - 0.5
    new_hue = (hue + hue_diff * alpha) % 1.0
    new_sat = np.clip(sat * (1.0 - alpha) + (target_s + (1.0 - target_s) * sat_boost) * alpha, 0.0, 1.0)

    hsv_new = np.stack([new_hue, new_sat, val], axis=-1).reshape(-1, 3)
    rgb_new = np.array([colorsys.hsv_to_rgb(*px) for px in hsv_new], dtype=np.float32).reshape(h, w, 3)
    return np.clip(rgb_new, 0.0, 1.0)


def _auto_subject_mask_from_image(img_rgb: Image.Image) -> np.ndarray:
    arr = np.array(img_rgb).astype(np.float32) / 255.0
    rgb_flat = arr.reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(*px) for px in rgb_flat], dtype=np.float32).reshape(arr.shape[0], arr.shape[1], 3)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # 粗分离主体：优先保留有颜色/有明暗信息的区域，抑制低饱和背景。
    base = ((sat > 0.12) & (val > 0.10)).astype(np.float32)

    # 中心先验：商品通常在画面中部，降低边缘背景误检概率。
    h, w = base.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    nx = (xx - cx) / max(1.0, w * 0.55)
    ny = (yy - cy) / max(1.0, h * 0.55)
    center_prior = np.exp(-(nx * nx + ny * ny))
    m = np.clip(base * (0.55 + 0.9 * center_prior), 0.0, 1.0)
    m = (m > 0.22).astype(np.uint8) * 255

    m_img = Image.fromarray(m, mode="L")
    # 简单形态学平滑，降低噪点并连通主体。
    m_img = (
        m_img
        .filter(ImageFilter.MaxFilter(5))
        .filter(ImageFilter.MinFilter(3))
        .filter(ImageFilter.GaussianBlur(2))
    )
    return np.array(m_img).astype(np.float32) / 255.0


def _blend_with_original_background(
    original_rgb: np.ndarray,
    edited_rgb: np.ndarray,
    subject_mask: np.ndarray,
    edge_soften_px: int = 2,
) -> np.ndarray:
    mask_u8 = np.clip(subject_mask * 255.0, 0, 255).astype(np.uint8)
    if edge_soften_px > 0:
        m_img = Image.fromarray(mask_u8, mode="L").filter(ImageFilter.GaussianBlur(edge_soften_px))
        alpha = np.array(m_img).astype(np.float32) / 255.0
    else:
        alpha = mask_u8.astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    return np.clip(edited_rgb * alpha + original_rgb * (1.0 - alpha), 0.0, 1.0)


def _apply_luma_tint(
    arr: np.ndarray,
    target_hex: str,
    mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    # 对低饱和主体更稳定：保持原图明暗，仅把色相/色调锚定到目标色，避免 HSV 路径发粉发脏。
    c = (target_hex or "").strip().lstrip("#")
    tr = int(c[0:2], 16) / 255.0
    tg = int(c[2:4], 16) / 255.0
    tb = int(c[4:6], 16) / 255.0
    t = np.array([tr, tg, tb], dtype=np.float32)
    y_w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    t_luma = float(np.dot(t, y_w))
    t_luma = max(t_luma, 1e-5)

    luma = np.tensordot(arr, y_w, axes=([2], [0]))  # HxW
    recolored = (luma[..., None] / t_luma) * t[None, None, :]
    recolored = np.clip(recolored, 0.0, 1.0)

    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0) * float(np.clip(strength, 0.0, 1.0))
    out = arr * (1.0 - alpha[..., None]) + recolored * alpha[..., None]
    return np.clip(out, 0.0, 1.0)


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
    auto_mask: bool = False,
) -> dict:
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

    if auto_mask:
        mask = _auto_subject_mask_from_image(img)
        if feather_px > 0:
            m_img = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L").filter(
                ImageFilter.GaussianBlur(max(1, feather_px))
            )
            mask = np.array(m_img).astype(np.float32) / 255.0
    else:
        mask = _build_soft_mask(h, w, x0, y0, x1, y1, feather_px)
    if auto_mask:
        rgb_new = _apply_luma_tint(arr, target_hex=target_hex, mask=mask, strength=strength)
    else:
        rgb_new = _apply_hsv_retarget(arr, target_hex=target_hex, mask=mask, strength=strength, sat_boost=0.15)
    if auto_mask:
        # 自动模式锁背景：主体外回填原图，避免背景脏色/彩点。
        rgb_new = _blend_with_original_background(
            original_rgb=arr,
            edited_rgb=rgb_new,
            subject_mask=(mask > 0.18).astype(np.float32),
            edge_soften_px=max(1, feather_px // 2),
        )
    out = np.clip(rgb_new * 255.0, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(out, mode="RGB")

    out_id = uuid.uuid4().hex
    out_path = RECOLOR_OUTPUT_DIR / f"{out_id}.jpg"
    out_img.save(out_path, format="JPEG", quality=92)

    return {
        "job_id": out_id,
        "recolored_url": f"/recolor-static/outputs/{out_path.name}",
        "bbox": {"x": x0, "y": y0, "w": max(1, x1 - x0), "h": max(1, y1 - y0)},
        "mask_mode": "auto_subject" if auto_mask else "manual_bbox",
    }


def recolor_region_ai(
    file_bytes: bytes,
    suffix: str,
    api_key: str,
    model: str,
    target_hex: str,
    x_ratio: float,
    y_ratio: float,
    w_ratio: float,
    h_ratio: float,
    strength: float = 0.7,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int | None = None,
    cfg_scale: float | None = None,
    num_inference_steps: int | None = None,
    postprocess: bool = True,
    image2: str | None = None,
    image3: str | None = None,
) -> dict:
    if not api_key:
        raise ValueError("缺少 SILICONFLOW_API_KEY，无法调用 AI 改色")

    from io import BytesIO

    img = Image.open(BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size

    x_ratio = float(np.clip(x_ratio, 0.0, 1.0))
    y_ratio = float(np.clip(y_ratio, 0.0, 1.0))
    w_ratio = float(np.clip(w_ratio, 0.01, 1.0))
    h_ratio = float(np.clip(h_ratio, 0.01, 1.0))
    strength = float(np.clip(strength, 0.0, 1.0))

    x0 = int(round(x_ratio * w))
    y0 = int(round(y_ratio * h))
    bw = int(round(w_ratio * w))
    bh = int(round(h_ratio * h))
    x1 = min(w, x0 + bw)
    y1 = min(h, y0 + bh)

    full_image_mode = x_ratio <= 0.001 and y_ratio <= 0.001 and w_ratio >= 0.999 and h_ratio >= 0.999
    if full_image_mode and postprocess:
        auto_mask = _auto_subject_mask_from_image(img)
        mask = (auto_mask > 0.25).astype(np.uint8)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1

    # 生成可视蒙版图（白色=需要改色，黑色=保留）
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_buf = BytesIO()
    mask_img.save(mask_buf, format="PNG")
    mask_b64 = base64.b64encode(mask_buf.getvalue()).decode("ascii")

    src_buf = BytesIO()
    img.save(src_buf, format="PNG")
    src_b64 = base64.b64encode(src_buf.getvalue()).decode("ascii")

    final_prompt = (
        prompt.strip()
        or f"将白色蒙版区域改为 #{target_hex.upper()}，保持纹理、光影和细节一致；非蒙版区域保持不变。"
    )

    payload: dict = {
        "model": model,
        "prompt": final_prompt,
        "image": f"data:image/png;base64,{src_b64}",
    }
    # 原生模式(关闭后处理)下，整图不传蒙版，更贴近 playground 的直出行为。
    if not (full_image_mode and not postprocess):
        payload["image2"] = f"data:image/png;base64,{mask_b64}"
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["seed"] = int(seed)
    if cfg_scale is not None:
        payload["cfg_scale"] = float(np.clip(cfg_scale, 1.0, 32.0))
    if num_inference_steps is not None:
        payload["num_inference_steps"] = int(np.clip(num_inference_steps, 1, 100))
    if image2:
        payload["image2"] = image2
    if image3:
        payload["image3"] = image3

    req = urllib.request.Request(
        SILICONFLOW_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
    except Exception as exc:
        raise ValueError(f"调用 SiliconFlow 失败: {exc}") from exc

    images = data.get("images") or []
    if not images or not isinstance(images[0], dict) or not images[0].get("url"):
        raise ValueError(f"SiliconFlow 未返回图片链接: {data}")
    result_url = str(images[0]["url"])

    try:
        with urllib.request.urlopen(result_url, timeout=120) as img_resp:
            result_bytes = img_resp.read()
        out_img = Image.open(BytesIO(result_bytes))
        out_img = ImageOps.exif_transpose(out_img).convert("RGB")
        # 第三方模型可能返回与输入不同分辨率，后续融合需要与原图同尺寸。
        if out_img.size != (w, h):
            out_img = out_img.resize((w, h), resample=Image.BICUBIC)
    except Exception as exc:
        raise ValueError(f"下载 AI 改色结果失败: {exc}") from exc

    if postprocess:
        # AI 模型可能出现目标色漂移（例如粉色偏紫），增加一步数值后校色，确保更接近 target_hex。
        out_arr = np.array(out_img).astype(np.float32) / 255.0
        src_arr = np.array(img).astype(np.float32) / 255.0
        if full_image_mode:
            post_mask = _auto_subject_mask_from_image(out_img)
        else:
            post_mask = _build_soft_mask(
                h=out_arr.shape[0],
                w=out_arr.shape[1],
                x0=int(round(x_ratio * out_arr.shape[1])),
                y0=int(round(y_ratio * out_arr.shape[0])),
                x1=min(out_arr.shape[1], int(round((x_ratio + w_ratio) * out_arr.shape[1]))),
                y1=min(out_arr.shape[0], int(round((y_ratio + h_ratio) * out_arr.shape[0]))),
                feather_px=int(round(min(out_arr.shape[1], out_arr.shape[0]) * 0.01)),
            )
        corrected = _apply_hsv_retarget(
            out_arr,
            target_hex=target_hex,
            mask=post_mask,
            strength=max(0.75, min(1.0, strength + 0.2)),
            sat_boost=0.05,
        )
        # 锁定背景：主体外区域回填原图，避免背景出现彩色噪点。
        corrected = _blend_with_original_background(
            original_rgb=src_arr,
            edited_rgb=corrected,
            subject_mask=post_mask if full_image_mode else (post_mask > 0.02).astype(np.float32),
            edge_soften_px=2,
        )
        out_img = Image.fromarray(np.clip(corrected * 255.0, 0, 255).astype(np.uint8), mode="RGB")

    out_id = uuid.uuid4().hex
    out_path = RECOLOR_OUTPUT_DIR / f"{out_id}.jpg"
    out_img.save(out_path, format="JPEG", quality=92)

    return {
        "job_id": out_id,
        "recolored_url": f"/recolor-static/outputs/{out_path.name}",
        "bbox": {"x": x0, "y": y0, "w": max(1, x1 - x0), "h": max(1, y1 - y0)},
        "mode": "ai",
        "used_params": {
            "provider": "siliconflow",
            "model": model,
            "prompt": final_prompt,
            "mask_mode": ("auto_subject" if (full_image_mode and postprocess) else ("full_or_manual" if full_image_mode else "manual_bbox")),
            "postprocess": bool(postprocess),
            "negative_prompt": negative_prompt,
            "seed": seed,
            "cfg_scale": payload.get("cfg_scale"),
            "num_inference_steps": payload.get("num_inference_steps"),
            "strength_hint": strength,
        },
    }
