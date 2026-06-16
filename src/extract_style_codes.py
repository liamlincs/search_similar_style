import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

STYLE_RE = re.compile(r"([A-Za-z0-9_-]+#)")
SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")
DEFAULT_CONFIG = Path("config/search_config.json")
OCR_ENGINE = None
OCR_IMPORT_ERROR: Exception | None = None
try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_ENGINE = RapidOCR()
except Exception as exc:
    OCR_IMPORT_ERROR = exc


def collect_images(base: Path, pattern: str, exts: list[str]) -> list[Path]:
    allow = {e.lower().lstrip(".") for e in exts}
    out = []
    for p in sorted(base.glob(pattern)):
        if not p.is_file():
            continue
        ext = p.suffix.lower().lstrip(".")
        if ext in allow:
            out.append(p)
    if out:
        return out
    # fallback: if pattern misses files, scan directory and keep only allowed image extensions
    for p in sorted(base.glob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower().lstrip(".")
        if ext in allow and p not in out:
            out.append(p)
    return out


def build_header_crops(img_path: Path) -> list[Image.Image]:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    boxes = [
        (0, 0, int(w * 0.65), int(h * 0.20)),
        (0, 0, int(w * 0.80), int(h * 0.25)),
        (0, 0, int(w * 1.00), int(h * 0.30)),
    ]
    return [img.crop(b) for b in boxes]


def _red_label_roi(img: Image.Image) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    mask = (r > 160) & (g < 150) & (b < 150) & ((r - g) > 35) & ((r - b) > 20)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad_x = max(4, int((x1 - x0 + 1) * 0.04))
    pad_y = max(4, int((y1 - y0 + 1) * 0.10))

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(img.width - 1, x1 + pad_x)
    y1 = min(img.height - 1, y1 + pad_y)
    return img.crop((x0, y0, x1 + 1, y1 + 1))


def _prep_for_ocr(img: Image.Image) -> list[Image.Image]:
    roi = _red_label_roi(img)
    gray = ImageOps.grayscale(roi)
    gray = ImageEnhance.Contrast(gray).enhance(2.5)
    gray = gray.resize((gray.width * 3, gray.height * 3), Image.BICUBIC)
    gray = ImageOps.autocontrast(gray)

    variants = [gray]
    for th in (120, 140, 160, 180):
        bw = gray.point(lambda p, t=th: 255 if p > t else 0, mode="1").convert("L")
        variants.append(bw)
    return variants


def _run_rapidocr(img: Image.Image) -> str:
    if OCR_ENGINE is None:
        raise RuntimeError(
            "rapidocr_onnxruntime is not installed; install it first to use OCR style-code extraction"
        ) from OCR_IMPORT_ERROR
    arr = np.asarray(img.convert("RGB"))
    result, _ = OCR_ENGINE(arr)
    if not result:
        return ""
    # sort by top-left y then x
    rows = []
    for item in result:
        box, text, score = item
        if not text:
            continue
        x = min(pt[0] for pt in box)
        y = min(pt[1] for pt in box)
        rows.append((y, x, str(text), float(score)))
    rows.sort(key=lambda t: (t[0], t[1]))
    return "\n".join([r[2] for r in rows])


def _run_tesseract(img: Image.Image, tesseract_bin: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
        img.save(f.name, format="PNG")
        cmd = [
            tesseract_bin,
            f.name,
            "stdout",
            "--psm",
            "7",
            "-c",
            "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-#",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return (proc.stdout or "").strip()


def _extract_code(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = text.replace("\r", "\n")
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]

    for ln in lines:
        m = STYLE_RE.search(ln)
        if m:
            code = m.group(1).upper()
            if re.fullmatch(r"[A-Z0-9_-]+#", code):
                return code

    compact = re.sub(r"\s+", "", cleaned)
    m = STYLE_RE.search(compact)
    if m:
        code = m.group(1).upper()
        if re.fullmatch(r"[A-Z0-9_-]+#", code):
            return code
    return None


def try_extract_code_from_image(header_crop_rgb: Image.Image, tesseract_bin: Optional[str]) -> Optional[str]:
    for i, v in enumerate(_prep_for_ocr(header_crop_rgb), start=1):
        raw = _run_rapidocr(v)
        logging.info("ocr raw(v%d/rapidocr): %s", i, raw.replace("\n", " ")[:240])
        code = _extract_code(raw)
        if code:
            return code

        if tesseract_bin:
            raw_t = _run_tesseract(v, tesseract_bin)
            logging.info("ocr raw(v%d/tesseract): %s", i, raw_t.replace("\n", " ")[:240])
            code_t = _extract_code(raw_t)
            if code_t:
                return code_t
    return None


def code_to_filename_prefix(code: str) -> str:
    core = code[:-1] if code.endswith("#") else code
    core = SAFE_RE.sub("_", core).strip("_")
    return core if core else "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(description="提取左上角款号并重命名为 款号_000.png（本地OCR）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--standard-dir", type=Path, default=Path("data/standard_samples"))
    parser.add_argument("--pattern", type=str, default="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = {}
    if args.config.exists():
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        path_cfg = cfg.get("paths", {})
        if args.standard_dir == Path("data/standard_samples") and path_cfg.get("standard_dir"):
            args.standard_dir = Path(path_cfg["standard_dir"])

    tesseract_bin = shutil.which("tesseract")
    if tesseract_bin:
        logging.info("tesseract found: %s", tesseract_bin)
    else:
        logging.info("tesseract not found; using rapidocr only")

    exts = cfg.get("paths", {}).get("image_exts", ["png", "jpg", "jpeg"]) if args.config.exists() else ["png", "jpg", "jpeg"]
    files = collect_images(args.standard_dir, args.pattern, exts)
    if not files:
        raise RuntimeError(f"no standard images found in {args.standard_dir}")

    plan = []
    skipped = []
    seq = defaultdict(int)

    for p in files:
        try:
            code = None
            for i, crop in enumerate(build_header_crops(p), start=1):
                code = try_extract_code_from_image(crop, tesseract_bin)
                if code:
                    logging.info("ocr crop success: %s crop=%d code=%s", p.name, i, code)
                    break
            if code is None:
                raise RuntimeError("no valid style code matched regex")
        except Exception as e:
            logging.warning("ocr failed: %s err=%s", p.name, e)
            skipped.append(p.name)
            continue

        prefix = code_to_filename_prefix(code)
        idx = seq[prefix]
        seq[prefix] += 1
        new_name = f"{prefix}_{idx:03d}{p.suffix.lower()}"
        plan.append((p, p.with_name(new_name), code))

    if args.dry_run:
        for old, new, code in plan:
            logging.info("DRY %s -> %s (code=%s)", old.name, new.name, code)
        if skipped:
            logging.info("DRY skipped (ocr failed): %s", ", ".join(skipped))
        logging.info("DRY summary: success=%d skipped=%d", len(plan), len(skipped))
        return

    temp_paths = []
    for i, (old, _, _) in enumerate(plan):
        tmp = old.with_name(f".__tmp_rename_{i:05d}{old.suffix.lower()}")
        old.rename(tmp)
        temp_paths.append(tmp)

    for tmp, (_, new, code) in zip(temp_paths, plan):
        tmp.rename(new)
        logging.info("%s -> %s (code=%s)", tmp.name, new.name, code)

    if skipped:
        logging.info("skipped (ocr failed): %s", ", ".join(skipped))
    logging.info("rename done. success=%d skipped=%d", len(plan), len(skipped))


if __name__ == "__main__":
    main()
