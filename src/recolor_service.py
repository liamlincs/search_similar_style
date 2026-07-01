from __future__ import annotations

import colorsys
import base64
import json
import uuid
import urllib.request
import urllib.error
import os
import logging
from pathlib import Path
from collections import deque

import numpy as np
from PIL import Image, ImageOps, ImageFilter

BASE_DIR = Path(__file__).resolve().parent
RECOLOR_DIR = BASE_DIR / "recolor_runtime"
RECOLOR_OUTPUT_DIR = RECOLOR_DIR / "outputs"
RECOLOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RECOLOR_ARK_INPUT_DIR = RECOLOR_DIR / "ark_inputs"
RECOLOR_ARK_INPUT_DIR.mkdir(parents=True, exist_ok=True)

ARK_IMAGE_GENERATION_URL = os.getenv(
    "ARK_IMAGE_GENERATION_URL",
    "https://ark.cn-beijing.volces.com/api/v3/images/generations",
)
_REMBG_SESSION = None
_REMBG_MODEL = os.getenv("REMBG_MODEL", "u2netp").strip() or "u2netp"
_RECOLOR_MAX_SIDE = int(os.getenv("RECOLOR_MAX_SIDE", "1600"))
_ARK_INPUT_MAX_SIDE = int(os.getenv("ARK_INPUT_MAX_SIDE", "1024"))
_ARK_INPUT_JPEG_QUALITY = int(os.getenv("ARK_INPUT_JPEG_QUALITY", "88"))
_ARK_USE_DATA_URL_INPUTS = os.getenv("ARK_USE_DATA_URL_INPUTS", "").strip().lower() in {"1", "true", "yes"}


def _truncate_for_log(value: str, limit: int = 2000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _summarize_image_input(value: str) -> str:
    text = str(value or "")
    if text.startswith("data:"):
        header = text.split(",", 1)[0]
        return f"{header},<base64 {max(0, len(text) - len(header) - 1)} chars>"
    return _truncate_for_log(text, 500)


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


def _auto_subject_mask_from_image(img_rgb: Image.Image) -> tuple[np.ndarray, str]:
    global _REMBG_SESSION
    # 优先使用 rembg(U2Net) 抠图；无依赖时再回退到规则分割。
    try:
        if _REMBG_SESSION is None:
            from rembg import new_session
            _REMBG_SESSION = new_session(_REMBG_MODEL)
        from rembg import remove
        m_img = remove(
            img_rgb.convert("RGB"),
            session=_REMBG_SESSION,
            only_mask=True,
            post_process_mask=True,
        )
        if not isinstance(m_img, Image.Image):
            m_img = Image.fromarray(np.array(m_img).astype(np.uint8), mode="L")
        m_img = m_img.convert("L").filter(ImageFilter.GaussianBlur(1.5))
        return np.array(m_img).astype(np.float32) / 255.0, "rembg_u2net"
    except Exception:
        pass

    # 更稳的自动主体分割：
    # 1) 以边缘像素建模背景色；2) 计算“偏离背景”得分；3) 保留与中心连通的主体区域。
    src = img_rgb.convert("RGB")
    ow, oh = src.size
    max_side = max(ow, oh)
    if max_side > 512:
        scale = 512.0 / max_side
        sw, sh = max(64, int(round(ow * scale))), max(64, int(round(oh * scale)))
        small = src.resize((sw, sh), resample=Image.BILINEAR)
    else:
        sw, sh = ow, oh
        small = src

    arr = np.array(small).astype(np.float32) / 255.0  # HxWx3
    h, w, _ = arr.shape

    b = max(2, int(round(min(h, w) * 0.08)))
    top = arr[:b, :, :]
    bottom = arr[-b:, :, :]
    left = arr[:, :b, :]
    right = arr[:, -b:, :]
    border = np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )
    bg_mu = border.mean(axis=0)
    bg_sigma = border.std(axis=0) + 1e-4

    z = (arr - bg_mu[None, None, :]) / bg_sigma[None, None, :]
    color_dist = np.sqrt(np.sum(z * z, axis=2))

    yy, xx = np.mgrid[0:h, 0:w]
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    nx = (xx - cx) / max(1.0, w * 0.48)
    ny = (yy - cy) / max(1.0, h * 0.48)
    center_prior = np.exp(-(nx * nx + ny * ny))

    score = color_dist * (0.65 + 0.9 * center_prior)
    score_t = float(np.percentile(score, 72))
    binary = score > max(0.6, score_t)

    # 仅保留与中心连通的前景，清除零散背景斑点。
    sx, sy = int(round(cx)), int(round(cy))
    if not binary[sy, sx]:
        ys, xs = np.where(binary)
        if len(xs) > 0:
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            i = int(np.argmin(d2))
            sx, sy = int(xs[i]), int(ys[i])
        else:
            sx, sy = int(round(w * 0.5)), int(round(h * 0.5))

    visited = np.zeros((h, w), dtype=np.uint8)
    fg = np.zeros((h, w), dtype=np.uint8)
    q: deque[tuple[int, int]] = deque()
    if binary[sy, sx]:
        q.append((sy, sx))
        visited[sy, sx] = 1
    while q:
        y, x = q.popleft()
        fg[y, x] = 1
        for ny_, nx_ in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny_ < 0 or ny_ >= h or nx_ < 0 or nx_ >= w:
                continue
            if visited[ny_, nx_] or not binary[ny_, nx_]:
                continue
            visited[ny_, nx_] = 1
            q.append((ny_, nx_))

    # 平滑边缘并回到原图大小
    m_img = Image.fromarray((fg * 255).astype(np.uint8), mode="L")
    m_img = m_img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2))
    if (w, h) != (ow, oh):
        m_img = m_img.resize((ow, oh), resample=Image.BILINEAR)
    return np.array(m_img).astype(np.float32) / 255.0, "fallback_rule"


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


def _downscale_if_needed(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = float(max_side) / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), resample=Image.LANCZOS)


def _apply_fabric_recolor(
    arr: np.ndarray,
    target_hex: str,
    mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    # 布料场景更自然的换色：
    # 1) 锁定目标 hue；2) 保留原明暗纹理；3) 在高亮区压饱和，避免荧光感。
    target_h, target_s, _target_v = _parse_hex_color(target_hex)
    h, w, _ = arr.shape
    rgb_flat = arr.reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(*px) for px in rgb_flat], dtype=np.float32).reshape(h, w, 3)
    orig_h = hsv[:, :, 0]
    orig_s = hsv[:, :, 1]
    orig_v = hsv[:, :, 2]

    s = float(np.clip(strength, 0.0, 1.0))

    # 高亮区域（接近白）降低目标饱和度，防止“荧光塑料感”。
    bright_dampen = 1.0 - 0.55 * np.clip((orig_v - 0.7) / 0.3, 0.0, 1.0)
    # 强度越高，目标色在饱和度中的权重越大；强度低时更偏保守。
    target_s_weight = 0.45 + 0.55 * s
    tgt_s_local = ((1.0 - target_s_weight) * orig_s + target_s_weight * target_s) * bright_dampen
    # 适度给高强度增加色彩浓度，但限制上限避免荧光。
    chroma_gain = 0.85 + 0.35 * s
    tgt_s_local = np.clip(tgt_s_local * chroma_gain, 0.0, 0.96)

    # 仅轻微拉亮/压暗，主要沿用原 V 保纹理。
    tgt_v_local = np.clip(orig_v * (0.97 - 0.02 * s) + (0.03 + 0.02 * s), 0.0, 1.0)

    recolor_hsv = np.stack(
        [
            np.full_like(orig_h, target_h, dtype=np.float32),
            tgt_s_local.astype(np.float32),
            tgt_v_local.astype(np.float32),
        ],
        axis=-1,
    ).reshape(-1, 3)
    recolored = np.array([colorsys.hsv_to_rgb(*px) for px in recolor_hsv], dtype=np.float32).reshape(h, w, 3)

    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0) * s
    # 感知增强：高强度时更明显，低强度时更柔和。
    alpha = np.clip(np.power(alpha, 0.85) * (0.88 + 0.12 * s), 0.0, 1.0)
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
    img = _downscale_if_needed(img, _RECOLOR_MAX_SIDE)
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

    mask_backend = "manual_bbox"
    if auto_mask:
        mask, mask_backend = _auto_subject_mask_from_image(img)
        if feather_px > 0:
            m_img = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L").filter(
                ImageFilter.GaussianBlur(max(1, feather_px))
            )
            mask = np.array(m_img).astype(np.float32) / 255.0
    else:
        mask = _build_soft_mask(h, w, x0, y0, x1, y1, feather_px)
    if auto_mask:
        rgb_new = _apply_fabric_recolor(arr, target_hex=target_hex, mask=mask, strength=strength)
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
        "mask_backend": mask_backend,
    }


def _crop_data_url_image(
    image_data_url: str | None,
    x_ratio: float | None,
    y_ratio: float | None,
    w_ratio: float | None,
    h_ratio: float | None,
) -> str | None:
    if not image_data_url:
        return None
    if x_ratio is None or y_ratio is None or w_ratio is None or h_ratio is None:
        return image_data_url
    if "," not in image_data_url:
        return image_data_url

    from io import BytesIO

    try:
        _header, b64 = image_data_url.split(",", 1)
        raw = base64.b64decode(b64)
        img = Image.open(BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        iw, ih = img.size
        x = float(np.clip(x_ratio, 0.0, 1.0))
        y = float(np.clip(y_ratio, 0.0, 1.0))
        cw = float(np.clip(w_ratio, 0.01, 1.0))
        ch = float(np.clip(h_ratio, 0.01, 1.0))
        x0 = int(round(x * iw))
        y0 = int(round(y * ih))
        x1 = min(iw, max(x0 + 1, int(round((x + cw) * iw))))
        y1 = min(ih, max(y0 + 1, int(round((y + ch) * ih))))
        if x1 <= x0 or y1 <= y0:
            return image_data_url
        cropped = img.crop((x0, y0, x1, y1))
        out = BytesIO()
        cropped.save(out, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(out.getvalue()).decode('ascii')}"
    except Exception:
        return image_data_url


def _data_url_to_bytes(data_url: str) -> tuple[bytes, str]:
    if "," not in data_url:
        raise ValueError("invalid data url")
    header, b64 = data_url.split(",", 1)
    ext = "png"
    if "image/jpeg" in header or "image/jpg" in header:
        ext = "jpg"
    elif "image/webp" in header:
        ext = "webp"
    return base64.b64decode(b64), ext


def _write_ark_input_bytes(raw: bytes, ext: str, public_base_url: str) -> str:
    from io import BytesIO

    try:
        img = Image.open(BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = _downscale_if_needed(img, _ARK_INPUT_MAX_SIDE)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=int(np.clip(_ARK_INPUT_JPEG_QUALITY, 50, 95)), optimize=True)
        raw = buf.getvalue()
        safe_ext = "jpg"
    except Exception:
        safe_ext = ext.strip(".").lower() or "jpg"
        if safe_ext == "jpeg":
            safe_ext = "jpg"
    name = f"{uuid.uuid4().hex}.{safe_ext}"
    out_path = RECOLOR_ARK_INPUT_DIR / name
    out_path.write_bytes(raw)
    logging.info("ark input file written path=%s bytes=%d", out_path.name, len(raw))
    if not public_base_url:
        return ""
    return f"{public_base_url.rstrip('/')}/recolor-static/ark_inputs/{name}"


def _write_ark_input_data_url(data_url: str, public_base_url: str) -> str:
    raw, ext = _data_url_to_bytes(data_url)
    return _write_ark_input_bytes(raw, ext, public_base_url)


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
    cfg: float | None = None,
    num_inference_steps: int | None = None,
    postprocess: bool = True,
    image2: str | None = None,
    image3: str | None = None,
    image2_crop_x: float | None = None,
    image2_crop_y: float | None = None,
    image2_crop_w: float | None = None,
    image2_crop_h: float | None = None,
    size: str = "2K",
    watermark: bool = False,
    output_format: str = "png",
    sequential_image_generation: str = "disabled",
    public_base_url: str = "",
) -> dict:
    if not api_key:
        raise ValueError("缺少 ARK_API_KEY，无法调用融合预览")

    from io import BytesIO

    img = Image.open(BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = _downscale_if_needed(img, _RECOLOR_MAX_SIDE)
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
    src_buf = BytesIO()
    ark_src_img = _downscale_if_needed(img, _ARK_INPUT_MAX_SIDE)
    ark_src_img.save(src_buf, format="JPEG", quality=int(np.clip(_ARK_INPUT_JPEG_QUALITY, 50, 95)), optimize=True)
    src_b64 = base64.b64encode(src_buf.getvalue()).decode("ascii")

    has_reference_images = bool(image2 or image3)
    final_prompt = prompt.strip() or (
        f"把图1的衣服改为 #{target_hex.upper()}，保持主体、材质、光影和背景自然。"
    )

    src_data_url = f"data:image/jpeg;base64,{src_b64}"
    if public_base_url and not _ARK_USE_DATA_URL_INPUTS:
        image_inputs = [_write_ark_input_data_url(src_data_url, public_base_url)]
    else:
        image_inputs = [src_data_url]
    if image2:
        image_inputs.append(_write_ark_input_data_url(image2, public_base_url) if public_base_url and not _ARK_USE_DATA_URL_INPUTS and image2.startswith("data:") else image2)
    if image3:
        image_inputs.append(_write_ark_input_data_url(image3, public_base_url) if public_base_url and not _ARK_USE_DATA_URL_INPUTS and image3.startswith("data:") else image3)

    payload: dict = {
        "model": model,
        "prompt": final_prompt,
        "image": image_inputs,
        "sequential_image_generation": sequential_image_generation,
        "size": size,
        "output_format": output_format,
        "watermark": bool(watermark),
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    logging.info(
        "ark image request model=%s size=%s output_format=%s watermark=%s sequential=%s image_count=%d prompt=%s images=%s",
        model,
        size,
        output_format,
        bool(watermark),
        sequential_image_generation,
        len(image_inputs),
        _truncate_for_log(final_prompt, 1000),
        [_summarize_image_input(item) for item in image_inputs],
    )

    req = urllib.request.Request(
        ARK_IMAGE_GENERATION_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = getattr(resp, "status", None) or getattr(resp, "code", None)
            body = resp.read().decode("utf-8")
            logging.info("ark image response status=%s body=%s", status, _truncate_for_log(body, 4000))
            data = json.loads(body)
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        logging.error(
            "ark image http error status=%s reason=%s body=%s",
            getattr(exc, "code", ""),
            getattr(exc, "reason", ""),
            _truncate_for_log(err_body, 4000),
        )
        raise ValueError(f"调用火山方舟失败: HTTP {getattr(exc, 'code', '')} {_truncate_for_log(err_body, 1000)}") from exc
    except Exception as exc:
        logging.exception("ark image request failed")
        raise ValueError(f"调用火山方舟失败: {exc}") from exc

    images = data.get("data") or data.get("images") or []
    if not images or not isinstance(images[0], dict) or not images[0].get("url"):
        logging.error("ark image missing url response=%s", _truncate_for_log(json.dumps(data, ensure_ascii=False), 4000))
        raise ValueError(f"火山方舟未返回图片链接: {data}")
    result_url = str(images[0]["url"])
    logging.info("ark image result url=%s", _truncate_for_log(result_url, 1000))

    try:
        with urllib.request.urlopen(result_url, timeout=120) as img_resp:
            result_bytes = img_resp.read()
        out_img = Image.open(BytesIO(result_bytes))
        out_img = ImageOps.exif_transpose(out_img).convert("RGB")
        # 第三方模型可能返回与输入不同分辨率，后续融合需要与原图同尺寸。
        if out_img.size != (w, h):
            out_img = out_img.resize((w, h), resample=Image.BICUBIC)
    except Exception as exc:
        logging.exception("ark image result download failed url=%s", _truncate_for_log(result_url, 1000))
        raise ValueError(f"下载预览结果失败: {exc}") from exc

    if postprocess and not has_reference_images:
        # AI 模型可能出现目标色漂移（例如粉色偏紫），增加一步数值后校色，确保更接近 target_hex。
        out_arr = np.array(out_img).astype(np.float32) / 255.0
        src_arr = np.array(img).astype(np.float32) / 255.0
        if full_image_mode:
            post_mask, _post_backend = _auto_subject_mask_from_image(out_img)
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
            "provider": "volcengine_ark",
            "model": model,
            "prompt": final_prompt,
            "task_mode": "reference_generate" if has_reference_images else "recolor_edit",
            "mask_mode": "none_multi_image" if has_reference_images else ("auto_subject" if (full_image_mode and postprocess) else ("full_or_manual" if full_image_mode else "manual_bbox")),
            "mask_backend": "",
            "postprocess": bool(postprocess and not has_reference_images),
            "negative_prompt": negative_prompt,
            "seed": seed,
            "cfg": payload.get("cfg"),
            "num_inference_steps": payload.get("num_inference_steps"),
            "strength_hint": strength,
            "has_image2": bool(image2),
            "has_image3": bool(image3),
            "has_target_mask": False,
            "image2_crop": None,
        },
    }
