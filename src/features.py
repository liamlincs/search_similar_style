import math
from typing import Tuple

import numpy as np
from PIL import Image


def _rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)


def _hsv_hist(arr: np.ndarray, bins: Tuple[int, int, int] = (12, 4, 4)) -> np.ndarray:
    rgb = arr.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff > 1e-6

    r_mask = (mx == r) & mask
    g_mask = (mx == g) & mask
    b_mask = (mx == b) & mask

    h[r_mask] = ((g[r_mask] - b[r_mask]) / diff[r_mask]) % 6
    h[g_mask] = (b[g_mask] - r[g_mask]) / diff[g_mask] + 2
    h[b_mask] = (r[b_mask] - g[b_mask]) / diff[b_mask] + 4
    h = h / 6.0

    s = np.zeros_like(mx)
    nz = mx > 1e-6
    s[nz] = diff[nz] / mx[nz]
    v = mx

    hb, sb, vb = bins
    hq = np.clip((h * hb).astype(np.int32), 0, hb - 1)
    sq = np.clip((s * sb).astype(np.int32), 0, sb - 1)
    vq = np.clip((v * vb).astype(np.int32), 0, vb - 1)

    idx = hq * (sb * vb) + sq * vb + vq
    hist = np.bincount(idx.ravel(), minlength=hb * sb * vb).astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def _lbp_hist(gray: np.ndarray, bins: int = 16) -> np.ndarray:
    # 8-neighbor LBP then compressed to 16 bins
    g = gray
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)

    neighbors = [
        g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:],
        g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1],
        g[2:, :-2], g[1:-1, :-2],
    ]

    for i, nb in enumerate(neighbors):
        code |= ((nb >= c).astype(np.uint8) << i)

    compressed = (code.astype(np.int32) * bins) // 256
    hist = np.bincount(compressed.ravel(), minlength=bins).astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def _edge_orientation_hist(gray: np.ndarray, bins: int = 18) -> np.ndarray:
    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    gx = gx[1:-1, :]
    gy = gy[:, 1:-1]

    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + math.pi) / (2.0 * math.pi)
    q = np.clip((ang * bins).astype(np.int32), 0, bins - 1)

    hist = np.bincount(q.ravel(), weights=mag.ravel(), minlength=bins).astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def _structure_descriptor(gray: np.ndarray, size: Tuple[int, int] = (24, 24)) -> np.ndarray:
    img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L").resize(size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr.ravel()


def extract_feature(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = _rgb_to_gray(rgb)

    f1 = _hsv_hist(rgb, bins=(12, 4, 4))
    f2 = _lbp_hist(gray, bins=16)
    f3 = _edge_orientation_hist(gray, bins=18)
    f4 = _structure_descriptor(gray, size=(24, 24))

    feat = np.concatenate([0.8 * f1, 1.0 * f2, 1.0 * f3, 0.5 * f4]).astype(np.float32)
    norm = np.linalg.norm(feat) + 1e-8
    return feat / norm


def extract_garment_color_feature(image: Image.Image, bins: Tuple[int, int, int] = (12, 4, 4)) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    # Garment region mask: ignore near-white background and the top plate-number strip.
    mask_bg = np.any(rgb < 240, axis=-1)
    mask_top = np.ones(mask_bg.shape, dtype=bool)
    cut = min(24, mask_bg.shape[0])
    mask_top[:cut, :] = False
    mask = mask_bg & mask_top

    if mask.sum() < 32:
        return _hsv_hist(rgb, bins=bins)

    sel = rgb[mask]
    sel_img = sel.reshape(-1, 1, 3)
    return _hsv_hist(sel_img, bins=bins)


def extract_stripe_feature(
    image: Image.Image,
    fft_keep: int = 24,
    resize_hw: Tuple[int, int] = (192, 192),
) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB").resize((resize_hw[1], resize_hw[0]), Image.BILINEAR), dtype=np.uint8)
    gray = _rgb_to_gray(rgb)

    # Keep foreground and suppress top banner noise.
    fg = np.any(rgb < 240, axis=-1).astype(np.float32)
    cut = min(24, fg.shape[0])
    fg[:cut, :] = 0.0

    if float(fg.sum()) < 64:
        fg = np.ones_like(fg, dtype=np.float32)

    denom_y = np.clip(fg.sum(axis=1), 1.0, None)
    denom_x = np.clip(fg.sum(axis=0), 1.0, None)
    proj_y = (gray * fg).sum(axis=1) / denom_y
    proj_x = (gray * fg).sum(axis=0) / denom_x

    def _fft_mag(sig: np.ndarray, k: int) -> np.ndarray:
        s = sig.astype(np.float32)
        s = s - s.mean()
        spec = np.abs(np.fft.rfft(s))
        # remove DC
        spec = spec[1 : 1 + k]
        if spec.shape[0] < k:
            spec = np.pad(spec, (0, k - spec.shape[0]))
        spec = spec.astype(np.float32)
        spec /= (spec.sum() + 1e-8)
        return spec

    fy = _fft_mag(proj_y, fft_keep)
    fx = _fft_mag(proj_x, fft_keep)
    feat = np.concatenate([fy, fx]).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-8)
    return feat
