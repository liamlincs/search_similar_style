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
SCENE_TOKEN_RE = re.compile(r"[A-Z0-9]{4,}")
KF_CODE_RE = re.compile(r"\bK[FEP8][A-Z]?\d{2}[-_ ]?\d{3,4}(?:[-_ ]?\d{1,2}[A-Z]?)?\b", re.IGNORECASE)
JC_CODE_RE = re.compile(r"\bJ[C0][A-Z]?\d{2}[-_ ]?\d{3,4}(?:[-_ ]?\d{1,2}[A-Z]?)?\b", re.IGNORECASE)
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


def collect_images_recursive(base: Path, exts: list[str]) -> list[Path]:
    allow = {e.lower().lstrip(".") for e in exts}
    out = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower().lstrip(".")
        if ext in allow:
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


def _prep_for_scene_ocr(img: Image.Image) -> list[Image.Image]:
    base = img.convert("RGB")
    max_edge = max(base.size)
    if max_edge < 960:
        scale = 960.0 / max(1, max_edge)
        nw = max(1, int(round(base.width * scale)))
        nh = max(1, int(round(base.height * scale)))
        base = base.resize((nw, nh), Image.BICUBIC)

    gray = ImageOps.grayscale(base)
    gray = ImageEnhance.Contrast(gray).enhance(1.9)
    gray = ImageOps.autocontrast(gray)
    strong = ImageEnhance.Sharpness(gray).enhance(1.8)
    return [
        base,
        gray.convert("RGB"),
        strong.convert("RGB"),
    ]


def extract_scene_text(img_rgb: Image.Image, max_variants: int | None = None) -> str:
    texts: list[str] = []
    seen = set()
    variants = _prep_for_scene_ocr(img_rgb)
    if max_variants is not None and max_variants > 0:
        variants = variants[:max_variants]
    for v in variants:
        raw = _run_rapidocr(v).strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        texts.append(raw)
    return "\n".join(texts)


def extract_text_tokens(text: str, min_len: int = 4) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    seen = set()
    for raw in SCENE_TOKEN_RE.findall(text.upper()):
        token = re.sub(r"^[0-9]+|[0-9]+$", "", raw)
        if len(token) < max(2, int(min_len)):
            continue
        if not re.search(r"[A-Z]", token):
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


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


def _clean_style_code(value: str) -> str:
    code = str(value or "").strip().upper()
    code = re.sub(r"[#＃]+$", "", code)
    return SAFE_RE.sub("_", code).strip("_")


def _looks_like_alpha_style_code(value: str) -> bool:
    code = _clean_style_code(value)
    if len(code) < 3:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_-]*", code))


def _normalize_kf_candidate(value: str) -> str:
    code = str(value or "").upper()
    code = re.sub(r"[^A-Z0-9]+", "-", code).strip("-")
    code = re.sub(r"^K[EP8]", "KF", code)
    code = re.sub(r"^K-F", "KF", code)
    code = re.sub(r"-+", "-", code)
    return code


def _normalize_jc_candidate(value: str) -> str:
    code = str(value or "").upper()
    code = re.sub(r"[^A-Z0-9]+", "-", code).strip("-")
    code = re.sub(r"^J0", "JC", code)
    code = re.sub(r"^J-C", "JC", code)
    code = re.sub(r"-+", "-", code)
    return code


def _extract_prefixed_style_from_text(text: str) -> str:
    if not text:
        return ""
    expanded = str(text).upper()
    expanded = expanded.replace("Ｋ", "K").replace("Ｆ", "F").replace("Ｊ", "J").replace("Ｃ", "C")
    expanded = re.sub(r"\bK\s+F\b", "KF", expanded)
    expanded = re.sub(r"\bJ\s+C\b", "JC", expanded)
    expanded = expanded.replace("O", "0")
    expanded = expanded.replace("＃", "#")
    candidates: list[str] = []
    for raw in (expanded, re.sub(r"\s+", "", expanded), re.sub(r"[^A-Z0-9]+", "-", expanded)):
        candidates.extend(m.group(0) for m in KF_CODE_RE.finditer(raw))
        candidates.extend(m.group(0) for m in JC_CODE_RE.finditer(raw))
    for candidate in candidates:
        code = _normalize_kf_candidate(candidate)
        if re.fullmatch(r"KF[A-Z]?\d{2}-?\d{3,4}(?:-?\d{1,2}[A-Z]?)?", code):
            return f"{code}#"
        code = _normalize_jc_candidate(candidate)
        if re.fullmatch(r"JC[A-Z]?\d{2}-?\d{3,4}(?:-?\d{1,2}[A-Z]?)?", code):
            return f"{code}#"
    return ""


def try_extract_code_from_image(header_crop_rgb: Image.Image, tesseract_bin: Optional[str]) -> Optional[str]:
    raw_candidates: list[str] = []
    seen = set()
    raw_direct = _run_rapidocr(header_crop_rgb).strip()
    if raw_direct:
        seen.add(raw_direct)
        raw_candidates.append(raw_direct)
    if tesseract_bin:
        raw_direct_t = _run_tesseract(header_crop_rgb, tesseract_bin).strip()
        if raw_direct_t and raw_direct_t not in seen:
            seen.add(raw_direct_t)
            raw_candidates.append(raw_direct_t)

    for i, v in enumerate(_prep_for_ocr(header_crop_rgb), start=1):
        raw = _run_rapidocr(v)
        logging.info("ocr raw(v%d/rapidocr): %s", i, raw.replace("\n", " ")[:240])
        if raw and raw not in seen:
            seen.add(raw)
            raw_candidates.append(raw)
        code = _extract_code(raw)
        if code and _looks_like_alpha_style_code(code):
            return code

        if tesseract_bin:
            raw_t = _run_tesseract(v, tesseract_bin)
            logging.info("ocr raw(v%d/tesseract): %s", i, raw_t.replace("\n", " ")[:240])
            if raw_t and raw_t not in seen:
                seen.add(raw_t)
                raw_candidates.append(raw_t)
            code_t = _extract_code(raw_t)
            if code_t and _looks_like_alpha_style_code(code_t):
                return code_t
    for raw in raw_candidates:
        code = _extract_prefixed_style_from_text(raw)
        if code:
            return code
    return None


def code_to_filename_prefix(code: str) -> str:
    core = code[:-1] if code.endswith("#") else code
    core = SAFE_RE.sub("_", core).strip("_")
    return core if core else "UNKNOWN"


def _style_before_hash(text: str) -> str:
    if "#" not in str(text or "") and "＃" not in str(text or ""):
        return ""
    before = re.split(r"[#＃]", str(text), maxsplit=1)[0]
    lines = [ln.strip() for ln in before.splitlines() if ln.strip()]
    raw = lines[-1] if lines else before
    code = _clean_style_code(raw)
    return code if _looks_like_alpha_style_code(code) else ""


def _style_from_filename(path: Path) -> str:
    stem = path.stem.strip()
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem).strip()
    if "_" in stem:
        prefix, suffix_num = stem.rsplit("_", 1)
        if suffix_num.isdigit():
            stem = prefix.strip()
    code = _style_before_hash(stem)
    if code:
        return code
    code = _clean_style_code(stem)
    return code if _looks_like_alpha_style_code(code) else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="递归提取左上角款号并原地重命名为 款号_000.jpg（本地OCR）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--standard-dir", type=Path, default=Path("data/standard_samples"))
    parser.add_argument("--pattern", type=str, default="")
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
    if str(args.pattern or "").strip():
        files = collect_images(args.standard_dir, args.pattern, exts)
    else:
        files = collect_images_recursive(args.standard_dir, exts)
    if not files:
        raise RuntimeError(f"no standard images found in {args.standard_dir}")
    total = len(files)
    progress_every = 50
    logging.info("scan start: total=%d root=%s", total, args.standard_dir)

    plan = []
    skipped = []
    seq = defaultdict(int)
    existing_by_dir: dict[Path, set[str]] = defaultdict(set)
    for path in files:
        existing_by_dir[path.parent].add(path.name.lower())
        style_code = _style_from_filename(path)
        if style_code:
            next_idx = 1
            stem = path.stem.strip()
            if "_" in stem:
                prefix, suffix_num = stem.rsplit("_", 1)
                if suffix_num.isdigit() and prefix.strip() == style_code:
                    next_idx = int(suffix_num) + 1
            seq[(path.parent, style_code)] = max(seq[(path.parent, style_code)], next_idx)

    for index, p in enumerate(files, start=1):
        try:
            code = None
            for i, crop in enumerate(build_header_crops(p), start=1):
                code = try_extract_code_from_image(crop, tesseract_bin)
                if code:
                    logging.info("ocr crop success: %s crop=%d code=%s", p.name, i, code)
                    break
            if code is None:
                fallback = _style_from_filename(p)
                if fallback:
                    code = f"{fallback}#"
                    logging.info("filename fallback success: %s code=%s", p.name, code)
            if code is None:
                raise RuntimeError("no valid style code matched regex or filename fallback")
        except Exception as e:
            logging.warning("ocr failed: %s err=%s", p.name, e)
            skipped.append(p.name)
            continue

        prefix = code_to_filename_prefix(code)
        dir_key = p.parent
        idx = seq[(dir_key, prefix)]
        new_name = f"{prefix}_{idx:03d}{p.suffix.lower()}"
        while new_name.lower() in existing_by_dir[dir_key] and new_name.lower() != p.name.lower():
            idx += 1
            new_name = f"{prefix}_{idx:03d}{p.suffix.lower()}"
        seq[(dir_key, prefix)] = idx + 1
        plan.append((p, p.with_name(new_name), code))
        existing_by_dir[dir_key].discard(p.name.lower())
        existing_by_dir[dir_key].add(new_name.lower())
        if index == total or index == 1 or index % progress_every == 0:
            logging.info(
                "scan progress: %d/%d success=%d failed=%d current=%s",
                index,
                total,
                len(plan),
                len(skipped),
                p.name,
            )

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
