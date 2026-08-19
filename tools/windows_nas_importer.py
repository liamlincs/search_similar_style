import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from catalog_store import filename_to_style_code

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_-]+")
KF_CODE_RE = re.compile(r"\bK[FEP8][A-Z]?\d{2}[-_ ]?\d{3,4}(?:[-_ ]?\d{1,2}[A-Z]?)?\b", re.IGNORECASE)
JC_CODE_RE = re.compile(r"\bJ[C0][A-Z]?\d{2}[-_ ]?\d{3,4}(?:[-_ ]?\d{1,2}[A-Z]?)?\b", re.IGNORECASE)
GENERIC_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9]?\d{2,4}(?:[-_ ]?\d{1,4})+(?:[-_ ]?\d{1,2}[A-Z]?)?\b", re.IGNORECASE)
ALPHA_DASH_NUM_CODE_RE = re.compile(r"\b[A-Z]{1,4}(?:[-_ ]?\d{1,4}[A-Z]?)+(?:[-_ ]?\d{1,2}[A-Z]?)?\b", re.IGNORECASE)
ALNUM_PREFIX_CODE_RE = re.compile(r"\b[A-Z]{1,5}\d{1,4}(?:[-_ ]?[A-Z0-9]{1,4})*\b", re.IGNORECASE)
ALPHA_ONLY_CODE_RE = re.compile(r"\b[A-Z]{3,6}\b", re.IGNORECASE)
NUMERIC_HASH_CODE_RE = re.compile(r"^\d{3,4}(?:-\d{1,2})?$")
UNRECOGNIZED_DIR = Path(__file__).resolve().parent / "未识别"
HASH_INDEX_NAME = "_nas_image_hash_index.jsonl"


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _clean_style_code(value: str) -> str:
    code = str(value or "").strip().upper()
    code = re.split(r"[#＃]+", code, maxsplit=1)[0]
    code = re.sub(r"[*+]+", "-", code)
    return SAFE_STEM_RE.sub("_", code).strip("_")


def _is_valid_style_code(value: str) -> bool:
    code = _clean_style_code(value)
    return _looks_like_style_code(code)


def _looks_like_style_code(value: str) -> bool:
    code = _clean_style_code(value)
    if len(code) < 3:
        return False
    return bool(re.fullmatch(r"[A-Z0-9][A-Z0-9_-]*", code))


def _looks_like_alpha_style_code(value: str) -> bool:
    code = _clean_style_code(value)
    if len(code) < 3:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_-]*", code))


def _looks_like_numeric_hash_style(value: str) -> bool:
    code = _clean_style_code(value)
    return bool(NUMERIC_HASH_CODE_RE.fullmatch(code))


def _looks_like_effective_style_code(value: str) -> bool:
    return _looks_like_alpha_style_code(value) or _looks_like_numeric_hash_style(value)


def _sanitize_filename(filename: str, fallback_suffix: str = ".jpg") -> str:
    raw = Path(str(filename or "").replace("\\", "/")).name
    stem = SAFE_STEM_RE.sub("_", Path(raw).stem).strip("_")
    suffix = Path(raw).suffix.lower() or fallback_suffix.lower()
    if suffix not in IMAGE_EXTS:
        suffix = fallback_suffix.lower()
    if not stem:
        stem = "UNKNOWN"
    return f"{stem}{suffix}"


def _split_tags(value: Any) -> list[str]:
    out: list[str] = []
    seen = set()
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[、,，\s]+", str(value or ""))
    for item in raw_items:
        tag = str(item or "").strip()
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def _tags_from_source_path(source_dir: Path, image_path: Path) -> tuple[list[str], str]:
    try:
        rel_parent = image_path.parent.relative_to(source_dir)
    except Exception:
        rel_parent = Path()
    parts = [p.strip() for p in rel_parent.parts if str(p).strip()]
    source_year = source_dir.name.strip()
    if re.fullmatch(r"\d{4}", source_year):
        parts = [source_year] + parts
    tags: list[str] = []
    display: list[str] = []
    if len(parts) >= 1:
        tags.append(f"year:{parts[0]}")
        display.append(f"年份：{parts[0]}")
    if len(parts) >= 2:
        tags.append(f"category:{parts[1]}")
        display.append(f"类别：{parts[1]}")
    if len(parts) >= 3:
        subcategory_tags: list[str] = []
        for part in parts[2:]:
            subcategory_tags.extend(_split_tags(part))
        for tag in subcategory_tags:
            tags.append(f"subcategory:{tag}")
        if subcategory_tags:
            display.append(f"细类：{' / '.join(subcategory_tags)}")
    return tags, "；".join(display)


def _next_target_name(prefix: str, suffix: str, used_names: set[str], next_seq: dict[str, int]) -> str:
    clean = SAFE_STEM_RE.sub("_", str(prefix or "").strip()).strip("_") or "UNKNOWN"
    seq = int(next_seq.get(clean, 0))
    suffix = suffix.lower() if suffix.lower() in IMAGE_EXTS else ".jpg"
    while seq < 100000:
        candidate = f"{clean}_{seq:03d}{suffix}"
        key = candidate.lower()
        if key not in used_names:
            used_names.add(key)
            next_seq[clean] = seq + 1
            return candidate
        seq += 1
    raise RuntimeError(f"cannot allocate filename for {clean}")


def _code_to_filename_prefix(code: str) -> str:
    core = str(code or "")
    core = re.split(r"[#＃]+", core, maxsplit=1)[0]
    core = re.sub(r"[*+]+", "-", core)
    core = SAFE_STEM_RE.sub("_", core).strip("_")
    return core if core else "UNKNOWN"


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
    expanded = re.sub(r"[*+]+", "-", expanded)
    candidates: list[str] = []
    for raw in (expanded, re.sub(r"\s+", "", expanded), re.sub(r"[^A-Z0-9]+", "-", expanded)):
        candidates.extend(m.group(0) for m in KF_CODE_RE.finditer(raw))
        candidates.extend(m.group(0) for m in JC_CODE_RE.finditer(raw))
    for candidate in candidates:
        code = _normalize_kf_candidate(candidate)
        if re.fullmatch(r"KF[A-Z]?\d{2}-?\d{3,4}(?:-?\d{1,2}[A-Z]?)?", code):
            return code
        code = _normalize_jc_candidate(candidate)
        if re.fullmatch(r"JC[A-Z]?\d{2}-?\d{3,4}(?:-?\d{1,2}[A-Z]?)?", code):
            return code
    for raw in (expanded, re.sub(r"\s+", "", expanded), re.sub(r"[^A-Z0-9]+", "-", expanded)):
        for match in GENERIC_CODE_RE.finditer(raw):
            code = str(match.group(0) or "").upper()
            code = re.sub(r"[^A-Z0-9]+", "-", code).strip("-")
            code = re.sub(r"-+", "-", code)
            if re.fullmatch(r"[A-Z][A-Z0-9]?\d{2,4}(?:-\d{1,4})+(?:-\d{1,2}[A-Z]?)?", code):
                return code
    for raw in (expanded, re.sub(r"\s+", "", expanded), re.sub(r"[^A-Z0-9]+", "-", expanded)):
        for match in ALPHA_DASH_NUM_CODE_RE.finditer(raw):
            code = str(match.group(0) or "").upper()
            code = re.sub(r"[^A-Z0-9]+", "-", code).strip("-")
            code = re.sub(r"-+", "-", code)
            if re.fullmatch(r"[A-Z]{1,4}(?:-\d{1,4}[A-Z]?)+(?:-\d{1,2}[A-Z]?)?", code):
                return code
    for raw in (expanded, re.sub(r"\s+", "", expanded), re.sub(r"[^A-Z0-9]+", "-", expanded)):
        for match in ALNUM_PREFIX_CODE_RE.finditer(raw):
            code = str(match.group(0) or "").upper()
            code = re.sub(r"[^A-Z0-9]+", "-", code).strip("-")
            code = re.sub(r"-+", "-", code)
            if re.fullmatch(r"[A-Z]{1,5}\d{1,4}(?:-[A-Z0-9]{1,4})*", code):
                return code
    for raw in (expanded, re.sub(r"\s+", "", expanded)):
        if "#" in raw:
            before = re.split(r"[#＃]", raw, maxsplit=1)[0].strip()
            before = re.sub(r"[^A-Z0-9]+", "-", before).strip("-")
            if NUMERIC_HASH_CODE_RE.fullmatch(before):
                return before
    for raw in (expanded, re.sub(r"\s+", "", expanded)):
        for match in ALPHA_ONLY_CODE_RE.finditer(raw):
            code = str(match.group(0) or "").upper().strip()
            if re.fullmatch(r"[A-Z]{3,6}", code):
                return code
    return ""


def _style_before_hash(text: str) -> str:
    if "#" not in str(text or "") and "＃" not in str(text or ""):
        return ""
    before = re.split(r"[#＃]", str(text), maxsplit=1)[0]
    lines = [ln.strip() for ln in before.splitlines() if ln.strip()]
    raw = lines[-1] if lines else before
    code = _clean_style_code(raw)
    return code if _looks_like_effective_style_code(code) else ""


def _style_from_top_left_text(text: str) -> str:
    for line in str(text or "").splitlines():
        code = _clean_style_code(line)
        if _looks_like_style_code(code):
            return code
    code = _clean_style_code(text)
    return code if _looks_like_style_code(code) else ""


def _style_from_filename(path: Path) -> tuple[str, str]:
    stem = path.stem.strip()
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem).strip()
    code = _style_before_hash(stem)
    if code:
        return code, "filename_hash"
    stem = re.sub(r"(?:_\d{3})+$", "", stem).strip()
    code = _clean_style_code(stem)
    if _looks_like_effective_style_code(code):
        return code, "filename"
    return "", ""


def _has_corner_label(img: Image.Image) -> bool:
    arr = np.asarray(img.convert("RGB"))
    if arr.size == 0:
        return False
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    red = (r > 170) & (g < 130) & (b < 130) & ((r - g) > 45) & ((r - b) > 35)
    blue = (b > 150) & (g > 90) & (r < 130) & ((b - r) > 45)
    mask = red | blue
    return bool(mask.mean() > 0.035)


def _corner_label_crops(path: Path) -> list[Image.Image]:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    regions = [
        (0, 0, int(w * 0.42), int(h * 0.24)),
        (int(w * 0.58), 0, w, int(h * 0.24)),
        (0, 0, int(w * 0.52), int(h * 0.34)),
        (int(w * 0.48), 0, w, int(h * 0.34)),
    ]
    crops: list[Image.Image] = []
    for box in regions:
        crop = img.crop(box)
        if _has_corner_label(crop):
            crops.append(crop)
    return crops


def _copy_unrecognized(source_dir: Path, image_path: Path) -> str:
    try:
        rel = image_path.relative_to(source_dir)
    except Exception:
        rel = Path(image_path.name)
    target = UNRECOGNIZED_DIR / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, target)
    return str(target)


def _copy_to_unique_target(
    src: Path,
    target_dir: Path,
    first_filename: str,
    used_names: set[str],
    next_seq: dict[str, int],
    style_code: str,
) -> str:
    first = _sanitize_filename(first_filename, src.suffix.lower())
    suffix = Path(first).suffix
    stem = re.sub(r"_\d+$", "", Path(first).stem).strip() or style_code
    candidates = [first]
    while True:
        if len(candidates) > 1 or candidates[-1].lower() in used_names:
            candidates.append(_next_target_name(stem, suffix, used_names, next_seq))
        candidate = candidates[-1]
        target = target_dir / candidate
        try:
            with src.open("rb") as rf, target.open("xb") as wf:
                shutil.copyfileobj(rf, wf, length=1024 * 1024)
            shutil.copystat(src, target)
            used_names.add(candidate.lower())
            return candidate
        except FileExistsError:
            used_names.add(candidate.lower())
            continue
        except Exception:
            try:
                if target.exists() and target.stat().st_size == 0:
                    target.unlink()
            except Exception:
                pass
            raise


def _file_hash(path: Path) -> str:
    h = hashlib.blake2b(digest_size=32)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_hash_index(target_dir: Path) -> dict[str, str]:
    index_path = target_dir / HASH_INDEX_NAME
    out: dict[str, str] = {}
    if not index_path.exists() or not index_path.is_file():
        return out
    for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        image_hash = str(item.get("hash") or "").strip()
        image_name = str(item.get("image_name") or "").strip()
        if image_hash and image_name and image_hash not in out:
            out[image_hash] = image_name
    return out


def _append_hash_index(target_dir: Path, image_hash: str, image_name: str) -> None:
    row = {
        "hash": image_hash,
        "image_name": image_name,
        "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (target_dir / HASH_INDEX_NAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def _ensure_hash_index(target_dir: Path) -> dict[str, str]:
    index_path = target_dir / HASH_INDEX_NAME
    existing = _read_hash_index(target_dir)
    if existing:
        return existing
    if index_path.exists():
        return existing
    for path in sorted(target_dir.iterdir()) if target_dir.exists() else []:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            image_hash = _file_hash(path)
        except Exception:
            logging.warning("failed to hash existing target image: %s", path, exc_info=True)
            continue
        if image_hash in existing:
            continue
        existing[image_hash] = path.name
        _append_hash_index(target_dir, image_hash, path.name)
    logging.info("hash index ready: %d items path=%s", len(existing), index_path)
    return existing


def _build_name_allocator(target_dir: Path) -> tuple[set[str], dict[str, int]]:
    used = set()
    next_seq: dict[str, int] = {}
    if target_dir.exists():
        for path in sorted(target_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            used.add(path.name.lower())
            stem = path.stem
            if "_" not in stem:
                continue
            prefix, seq_text = stem.rsplit("_", 1)
            if seq_text.isdigit():
                next_seq[prefix] = max(next_seq.get(prefix, 0), int(seq_text) + 1)
    return used, next_seq


def _scan_images(source_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def _extract_style(path: Path, tesseract_bin: str | None, recognition_mode: str = "ocr") -> tuple[str, str]:
    from extract_style_codes import (
        _prep_for_ocr,
        _run_rapidocr,
        _run_tesseract,
        try_extract_code_from_image,
    )

    filename_code, filename_source = _style_from_filename(path)
    if recognition_mode == "filename_first" and filename_code:
        return filename_code, filename_source

    try:
        corner_crops = _corner_label_crops(path)
        for crop in corner_crops:
            code = str(try_extract_code_from_image(crop, tesseract_bin) or "").strip()
            if code and _looks_like_effective_style_code(code):
                return code, "normal"
    except Exception:
        if filename_code:
            logging.info("filename fallback after OCR read error: %s code=%s", path.name, filename_code)
            return filename_code, filename_source
        raise

    loose_texts: list[str] = []
    seen_texts = set()
    for crop in corner_crops:
        raw_crop = _run_rapidocr(crop).strip()
        if raw_crop and raw_crop not in seen_texts:
            seen_texts.add(raw_crop)
            loose_texts.append(raw_crop)
        if tesseract_bin:
            raw_crop_t = _run_tesseract(crop, tesseract_bin).strip()
            if raw_crop_t and raw_crop_t not in seen_texts:
                seen_texts.add(raw_crop_t)
                loose_texts.append(raw_crop_t)
        for variant in _prep_for_ocr(crop):
            raw = _run_rapidocr(variant).strip()
            if raw and raw not in seen_texts:
                seen_texts.add(raw)
                loose_texts.append(raw)
            if tesseract_bin:
                raw_t = _run_tesseract(variant, tesseract_bin).strip()
                if raw_t and raw_t not in seen_texts:
                    seen_texts.add(raw_t)
                    loose_texts.append(raw_t)

    for raw in loose_texts:
        code = _extract_prefixed_style_from_text(raw)
        if code:
            logging.info("prefixed fallback label success: %s code=%s", path.name, code)
            return code, "prefixed_fallback"

    for raw in loose_texts:
        code = _style_before_hash(raw)
        if code:
            logging.info("hash-prefix fallback success: %s code=%s", path.name, code)
            return code, "hash_prefix"

    if filename_code:
        return filename_code, filename_source

    return "", ""


@dataclass
class ImportJob:
    job_id: str
    source_dir: Path
    target_dir: Path
    recognition_mode: str = "ocr"
    status: str = "pending"
    message: str = "任务已创建"
    total: int = 0
    processed: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    committed: bool = False
    result: dict[str, Any] | None = None


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ImportJob] = {}

    def create(self, source_dir: Path, target_dir: Path, recognition_mode: str = "ocr") -> ImportJob:
        job = ImportJob(uuid.uuid4().hex, source_dir, target_dir, recognition_mode)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> ImportJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in changes.items():
                setattr(job, key, value)

    def snapshot(self, job: ImportJob) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": job.job_id,
                "source_dir": str(job.source_dir),
                "target_dir": str(job.target_dir),
                "recognition_mode": job.recognition_mode,
                "status": job.status,
                "message": job.message,
                "total": job.total,
                "processed": job.processed,
                "items": list(job.items),
                "committed": job.committed,
                "result": job.result or {},
            }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NAS 款图批量入库</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f6f7f9; }
    header { position: sticky; top: 0; z-index: 5; background: #fff; border-bottom: 1px solid #e5e7eb; padding: 14px 18px; }
    h1 { margin: 0; font-size: 20px; }
    .title-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
    .top-progress { display: grid; gap: 6px; min-width: min(360px, 45vw); }
    .progress-text { color: #64748b; font-size: 14px; font-weight: 900; text-align: right; }
    .progress-track { height: 8px; border-radius: 999px; background: #e5e7eb; overflow: hidden; }
    .progress-bar { width: 0%; height: 100%; border-radius: inherit; background: #2563eb; transition: width .18s ease; }
    .path-grid { display: grid; grid-template-columns: 1fr 1fr 150px 150px auto; gap: 10px; align-items: end; }
    .scan-actions { display: flex; gap: 8px; align-items: center; }
    label { display: grid; gap: 5px; font-size: 12px; font-weight: 700; color: #4b5563; }
    input[type="text"], select { width: 100%; height: 42px; border: 1px solid #cfd5df; border-radius: 6px; padding: 9px 12px; font-size: 14px; background: #fff; }
    button { border: 0; border-radius: 6px; padding: 10px 14px; background: #111827; color: #fff; font-weight: 800; cursor: pointer; min-height: 40px; }
    button.secondary { background: #e5e7eb; color: #111827; }
    button:disabled { opacity: .52; cursor: not-allowed; }
    main { padding: 16px 18px 40px; }
    .status { margin: 0 0 12px; color: #64748b; font-weight: 700; }
    .status:empty { display: none; }
    .status.err { color: #b91c1c; }
    .toolbar { display: flex; gap: 10px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
    .table-wrap { overflow: auto; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; min-width: 1080px; }
    th, td { border-bottom: 1px solid #e5e7eb; padding: 10px; text-align: left; vertical-align: middle; }
    th { position: sticky; top: 0; background: #f9fafb; z-index: 1; font-size: 13px; }
    tr.warn { background: #fff7ed; }
    tr.err { background: #fef2f2; }
    .source { max-width: 220px; word-break: break-all; color: #374151; font-weight: 700; }
    .source button { appearance: none; border: 0; background: transparent; color: #2563eb; padding: 0; min-height: 0; font: inherit; font-weight: 800; text-align: left; cursor: pointer; }
    .muted { color: #6b7280; font-size: 12px; margin-top: 4px; }
    .pill { display: inline-block; border-radius: 999px; padding: 4px 9px; font-size: 12px; font-weight: 800; background: #e5f3ff; color: #075985; }
    .pill.err { background: #fee2e2; color: #b91c1c; }
    .pill.warn { background: #ffedd5; color: #c2410c; }
    .modal { position: fixed; inset: 0; display: none; place-items: center; background: rgba(17,24,39,.48); z-index: 20; padding: 20px; }
    .modal.open { display: grid; }
    .modal-box { width: min(680px, 100%); max-height: min(82vh, 720px); display: flex; flex-direction: column; border-radius: 8px; background: #fff; box-shadow: 0 18px 50px rgba(0,0,0,.22); padding: 18px; }
    .modal-box.image-box { width: min(920px, 100%); }
    .modal-message { max-height: 60vh; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; line-height: 1.55; font-weight: 700; color: #111827; }
    .modal-image { display: none; width: 100%; max-height: 76vh; object-fit: contain; background: #f3f4f6; border-radius: 6px; margin-bottom: 12px; }
    .modal-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 16px; }
    @media (max-width: 900px) {
      .path-grid { grid-template-columns: 1fr; }
      .title-row { display: grid; }
      .top-progress { min-width: 0; width: 100%; }
      .progress-text { text-align: left; }
      header { position: static; }
    }
  </style>
</head>
<body>
  <header>
    <div class="title-row">
      <h1>NAS 款图批量入库</h1>
      <div class="top-progress" aria-label="扫描进度">
        <div id="progressText" class="progress-text">已处理 0/0</div>
        <div class="progress-track"><div id="progressBar" class="progress-bar"></div></div>
      </div>
    </div>
    <div class="path-grid">
      <label>源目录
        <input id="sourceDir" type="text" placeholder="例如 Z:\2018\成衣" />
      </label>
      <label>目标目录
        <input id="targetDir" type="text" />
      </label>
      <label>模式
        <select id="runMode">
          <option value="scan_import" selected>扫描入库</option>
          <option value="scan_only">扫描识别</option>
        </select>
      </label>
      <label>识别方式
        <select id="recognitionMode">
          <option value="ocr" selected>OCR识别</option>
          <option value="filename_first">文件名优先</option>
        </select>
      </label>
      <div class="scan-actions">
        <button id="startBtn" type="button">执行</button>
        <button id="cancelScanBtn" type="button" class="secondary" disabled>停止扫描</button>
      </div>
    </div>
  </header>
  <main>
    <div id="status" class="status"></div>
    <div class="toolbar">
      <button id="toggleBtn" type="button" class="secondary" data-manual-only>全选/反选</button>
      <button id="commitBtn" type="button" data-manual-only>确认复制入库</button>
      <span class="muted" id="countText"></span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th data-manual-only>导入</th>
            <th>源文件</th>
            <th>款号</th>
            <th>导入后文件名</th>
            <th>目录标签</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody id="rows">
          <tr><td colspan="6" class="muted">填写源目录后开始扫描。</td></tr>
        </tbody>
      </table>
    </div>
  </main>
  <div id="modal" class="modal">
    <div id="modalBox" class="modal-box">
      <img id="modalImage" class="modal-image" alt="图片预览" />
      <div id="modalMessage" class="modal-message"></div>
      <div class="modal-actions">
        <button id="modalCancel" type="button" class="secondary" style="display:none">取消</button>
        <button id="modalOk" type="button">确定</button>
      </div>
    </div>
  </div>
  <script>
    const DEFAULT_SOURCE = "__DEFAULT_SOURCE__";
    const DEFAULT_TARGET = "__DEFAULT_TARGET__";
    const $ = (id) => document.getElementById(id);
    let currentJob = null;
    let pollTimer = null;
    let autoCommitAfterScan = true;
    let autoCommitRunning = false;

    const PATH_MEMORY_KEYS = {
      source: "nas_importer_source_dir",
      target: "nas_importer_target_dir",
    };
    $("sourceDir").value = localStorage.getItem(PATH_MEMORY_KEYS.source) || DEFAULT_SOURCE;
    $("targetDir").value = localStorage.getItem(PATH_MEMORY_KEYS.target) || DEFAULT_TARGET;

    function isManualMode() {
      return $("runMode").value === "scan_only";
    }
    function applyModeUI() {
      const manual = isManualMode();
      document.querySelectorAll("[data-manual-only]").forEach((el) => {
        el.style.display = manual ? "" : "none";
      });
      $("startBtn").textContent = "执行";
      if (currentJob && currentJob.items && currentJob.items.length) {
        renderJob(currentJob);
      }
    }
    function rememberPaths() {
      const source = $("sourceDir").value.trim();
      const target = $("targetDir").value.trim();
      if (source) localStorage.setItem(PATH_MEMORY_KEYS.source, source);
      if (target) localStorage.setItem(PATH_MEMORY_KEYS.target, target);
    }
    function alertBox(message, imageUrl = "") {
      return new Promise((resolve) => {
        $("modalBox").classList.toggle("image-box", !!imageUrl);
        $("modalImage").style.display = imageUrl ? "block" : "none";
        $("modalImage").src = imageUrl || "";
        $("modalMessage").textContent = String(message || "");
        $("modalCancel").style.display = "none";
        $("modal").classList.add("open");
        $("modalOk").onclick = () => {
          $("modal").classList.remove("open");
          $("modalImage").removeAttribute("src");
          $("modalOk").onclick = null;
          $("modalCancel").onclick = null;
          resolve();
        };
      });
    }
    function confirmBox(message) {
      return new Promise((resolve) => {
        $("modalBox").classList.remove("image-box");
        $("modalImage").style.display = "none";
        $("modalImage").removeAttribute("src");
        $("modalMessage").textContent = String(message || "");
        $("modalCancel").style.display = "inline-block";
        $("modal").classList.add("open");
        $("modalOk").onclick = () => {
          $("modal").classList.remove("open");
          $("modalOk").onclick = null;
          $("modalCancel").onclick = null;
          resolve(true);
        };
        $("modalCancel").onclick = () => {
          $("modal").classList.remove("open");
          $("modalOk").onclick = null;
          $("modalCancel").onclick = null;
          resolve(false);
        };
      });
    }
    function setStatus(message, err = false) {
      $("status").textContent = message || "";
      $("status").className = err ? "status err" : "status";
    }
    function setTopProgressStatus(message = "", processed = 0, total = 0) {
      const done = Math.max(0, Number(processed || 0));
      const all = Math.max(0, Number(total || 0));
      const prefix = String(message || "已处理").trim();
      $("progressText").textContent = all > 0 ? `${prefix} ${done}/${all}` : prefix;
      const pct = all > 0 ? Math.max(0, Math.min(100, (done / all) * 100)) : 0;
      $("progressBar").style.width = pct.toFixed(1) + "%";
    }
    function setScanning(active) {
      $("startBtn").disabled = !!active;
      $("startBtn").textContent = active ? "执行中..." : "执行";
      $("cancelScanBtn").disabled = !active;
      $("sourceDir").disabled = !!active;
      $("targetDir").disabled = !!active;
      $("runMode").disabled = !!active;
      $("recognitionMode").disabled = !!active;
    }
    async function api(path, options = {}) {
      const resp = await fetch(path, options);
      if (!resp.ok) {
        let message = await resp.text();
        try { message = JSON.parse(message).detail || message; } catch (_) {}
        throw new Error(message);
      }
      return resp.json();
    }
    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }
    function cleanCode(value) {
      return String(value || "").trim().toUpperCase().replace(/#$/, "").replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "");
    }
    function updateFilename(row) {
      const codeInput = row.querySelector('[data-role="style"]');
      const filenameInput = row.querySelector('[data-role="filename"]');
      const code = cleanCode(codeInput.value);
      if (code) filenameInput.value = code + "_000" + (row.dataset.suffix || ".jpg");
    }
    function rowPayload(row) {
      return {
        selected: row.querySelector('[data-role="selected"]').checked,
        source_rel_path: row.dataset.relPath || "",
        style_code: row.querySelector('[data-role="style"]').value.trim(),
        target_filename: row.querySelector('[data-role="filename"]').value.trim(),
      };
    }
    function jobItemPayload(item) {
      return {
        selected: true,
        source_rel_path: item.source_rel_path || "",
        style_code: item.proposed_style_code || "",
        target_filename: item.proposed_filename || "",
      };
    }
    function previewSource(jobId, relPath, name) {
      const qs = new URLSearchParams({ job_id: jobId, source_rel_path: relPath || "", max_edge: "900" });
      return alertBox(name || "图片预览", "/api/source-image?" + qs.toString());
    }
    function renderJob(job) {
      currentJob = job;
      $("countText").textContent = job.total ? `${job.processed}/${job.total}` : "";
      const status = String(job.status || "");
      const label = status === "canceled" ? "已停止扫描" : (status === "failed" ? "扫描失败" : (status === "completed" ? "预处理完成" : "已处理"));
      setTopProgressStatus(label, job.processed || 0, job.total || 0);
      setStatus("");
      if (!job.items || !job.items.length) {
        $("rows").innerHTML = '<tr><td colspan="6" class="muted">暂无待确认图片。</td></tr>';
        return;
      }
      $("rows").innerHTML = job.items.map((item, index) => {
        const manual = isManualMode();
        const statusClass = item.status === "ok" ? "" : (item.status === "ocr_failed" || item.status === "invalid_style_code" ? "err" : "warn");
        const pillClass = item.status === "ok" ? "pill" : (statusClass === "err" ? "pill err" : "pill warn");
        const sourceLabels = {
          kf_fallback: "识别成功（KF增强）",
          hash_prefix: "识别成功（#前文本）",
          top_left: "识别成功（左上角）",
          filename_hash: "识别成功（文件名#前）",
          filename: "识别成功（文件名）",
        };
        const sourceText = sourceLabels[item.ocr_source] || "识别成功";
        const statusText = item.error || (item.status === "ok" ? sourceText : item.status);
        return `
          <tr class="${statusClass}" data-rel-path="${escapeHtml(item.source_rel_path || "")}" data-suffix="${escapeHtml(item.suffix || ".jpg")}">
            ${manual ? '<td data-manual-only><input data-role="selected" type="checkbox" checked></td>' : ''}
            <td><div class="source"><button type="button" data-role="previewSource">${escapeHtml(item.source_name || "")}</button></div><div class="muted">${escapeHtml(item.source_rel_path || "")}</div></td>
            <td><input data-role="style" type="text" value="${escapeHtml(item.proposed_style_code || "")}"></td>
            <td><input data-role="filename" type="text" value="${escapeHtml(item.proposed_filename || "")}"></td>
            <td><div class="muted">${escapeHtml(item.tag_display || "")}</div></td>
            <td><span class="${pillClass}">${escapeHtml(statusText)}</span></td>
          </tr>`;
      }).join("");
      $("rows").querySelectorAll('[data-role="style"]').forEach((input) => {
        input.addEventListener("input", () => updateFilename(input.closest("tr")));
      });
      $("rows").querySelectorAll('[data-role="previewSource"]').forEach((button) => {
        button.addEventListener("click", () => {
          const row = button.closest("tr");
          previewSource(job.job_id, row.dataset.relPath || "", button.textContent || "图片预览");
        });
      });
    }
    async function startScan(autoCommit = false) {
      const source_dir = $("sourceDir").value.trim();
      const target_dir = $("targetDir").value.trim();
      const recognition_mode = $("recognitionMode").value;
      if (!source_dir || !target_dir) return alertBox("请填写源目录和目标目录");
      rememberPaths();
      autoCommitAfterScan = !!autoCommit;
      autoCommitRunning = false;
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      currentJob = null;
      setTopProgressStatus("正在创建扫描任务", 0, 0);
      setScanning(true);
      setStatus("正在创建扫描任务...");
      try {
        const job = await api("/api/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source_dir, target_dir, recognition_mode }),
        });
        renderJob(job);
        pollTimer = setInterval(pollJob, 1200);
      } catch (err) {
        setStatus(err.message || "扫描失败", true);
        await alertBox("扫描失败：" + (err.message || "未知错误"));
      } finally {
        if (!currentJob || ["completed", "failed", "canceled"].includes(currentJob.status)) setScanning(false);
      }
    }
    async function startScanAndImport() {
      const source_dir = $("sourceDir").value.trim();
      const target_dir = $("targetDir").value.trim();
      if (!source_dir || !target_dir) return alertBox("请填写源目录和目标目录");
      const ok = await confirmBox(`确认扫描并直接入库？\n\n源目录：${source_dir}\n目标目录：${target_dir}\n\n扫描完成后会自动复制入库，不再停下来人工确认。`);
      if (!ok) return;
      await startScan(true);
    }
    async function startSelectedMode() {
      if (isManualMode()) {
        await startScan(false);
      } else {
        await startScanAndImport();
      }
    }
    async function pollJob() {
      if (!currentJob) return;
      try {
        const job = await api("/api/jobs/" + encodeURIComponent(currentJob.job_id));
        renderJob(job);
        if (["completed", "failed", "canceled"].includes(job.status)) {
          clearInterval(pollTimer);
          pollTimer = null;
          setScanning(false);
          if (job.status === "failed") await alertBox("扫描失败：" + (job.message || ""));
          if (job.status === "completed" && autoCommitAfterScan && !autoCommitRunning) {
            autoCommitRunning = true;
            await commitItems(job.items.map(jobItemPayload));
            autoCommitAfterScan = false;
            autoCommitRunning = false;
          }
        }
      } catch (err) {
        clearInterval(pollTimer);
        pollTimer = null;
        setScanning(false);
        setStatus(err.message || "读取任务失败", true);
      }
    }
    async function cancelScan() {
      if (!currentJob || !["pending", "running"].includes(currentJob.status)) return;
      $("cancelScanBtn").disabled = true;
      setTopProgressStatus("正在停止扫描", currentJob.processed || 0, currentJob.total || 0);
      setStatus("正在停止扫描...");
      try {
        const job = await api("/api/jobs/" + encodeURIComponent(currentJob.job_id) + "/cancel", { method: "POST" });
        renderJob(job);
        if (["completed", "failed", "canceled"].includes(job.status)) {
          if (pollTimer) clearInterval(pollTimer);
          pollTimer = null;
          setScanning(false);
          autoCommitAfterScan = false;
        }
      } catch (err) {
        setStatus(err.message || "停止扫描失败", true);
        $("cancelScanBtn").disabled = false;
      }
    }
    async function commitJob() {
      if (!currentJob || currentJob.status !== "completed") return alertBox("请先等待扫描完成");
      const items = Array.from($("rows").querySelectorAll("tr[data-rel-path]")).map(rowPayload);
      if (!items.some((item) => item.selected)) return alertBox("请至少选择一张要导入的图片");
      await commitItems(items);
    }
    async function commitItems(items) {
      if (!currentJob || currentJob.status !== "completed") return alertBox("请先等待扫描完成");
      if (!items.some((item) => item.selected)) return alertBox("请至少选择一张要导入的图片");
      $("commitBtn").disabled = true;
      $("startBtn").disabled = true;
      setStatus("正在复制到目标目录...");
      try {
        const result = await api("/api/jobs/" + encodeURIComponent(currentJob.job_id) + "/commit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ items }),
        });
        const failed = result.failed || [];
        const message = `导入完成：成功 ${result.imported || 0} 张，重复跳过 ${result.duplicates || 0} 张，失败 ${failed.length} 张` +
          (result.manifest_path ? `\n标签文件：${result.manifest_path}` : "") +
          (result.hash_index_path ? `\n去重索引：${result.hash_index_path}` : "") +
          (failed.length ? "\n\n" + failed.slice(0, 20).map(x => `${x.source_rel_path || ""}：${x.error || "失败"}`).join("\n") : "");
        setStatus(message, failed.length > 0);
        await alertBox(message);
      } catch (err) {
        setStatus(err.message || "导入失败", true);
        await alertBox("导入失败：" + (err.message || "未知错误"));
      } finally {
        $("commitBtn").disabled = false;
        $("startBtn").disabled = false;
      }
    }
    $("startBtn").addEventListener("click", startSelectedMode);
    $("cancelScanBtn").addEventListener("click", cancelScan);
    $("commitBtn").addEventListener("click", commitJob);
    $("toggleBtn").addEventListener("click", () => {
      const boxes = Array.from(document.querySelectorAll('[data-role="selected"]'));
      const should = boxes.some(box => !box.checked);
      boxes.forEach(box => { box.checked = should; });
    });
    ["sourceDir", "targetDir"].forEach((id) => {
      $(id).addEventListener("blur", rememberPaths);
    });
    $("runMode").addEventListener("change", applyModeUI);
    applyModeUI();
  </script>
</body>
</html>
"""


class ImportHandler(BaseHTTPRequestHandler):
    server: "ImportServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, content: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: int, payload: Any) -> None:
        self._send(status, _json_bytes(payload))

    def _error(self, status: int, detail: str) -> None:
        self._send_json(status, {"detail": detail})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                html = HTML.replace("__DEFAULT_SOURCE__", self.server.default_source.replace("\\", "\\\\")).replace(
                    "__DEFAULT_TARGET__", self.server.default_target.replace("\\", "\\\\")
                )
                self._send(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path.startswith("/api/jobs/"):
                job_id = unquote(parsed.path.rsplit("/", 1)[-1])
                job = self.server.jobs.get(job_id)
                if not job:
                    self._error(HTTPStatus.NOT_FOUND, "任务不存在")
                    return
                self._send_json(HTTPStatus.OK, self.server.jobs.snapshot(job))
                return
            if parsed.path == "/api/source-image":
                self._serve_source_image(parsed)
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            logging.exception("GET failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/jobs":
                payload = _read_json(self)
                source_dir = Path(str(payload.get("source_dir") or "").strip())
                target_dir = Path(str(payload.get("target_dir") or "").strip())
                recognition_mode = str(payload.get("recognition_mode") or "ocr").strip()
                if recognition_mode not in {"ocr", "filename_first"}:
                    recognition_mode = "ocr"
                if not source_dir.exists() or not source_dir.is_dir():
                    self._error(HTTPStatus.BAD_REQUEST, "源目录不存在")
                    return
                target_dir.mkdir(parents=True, exist_ok=True)
                job = self.server.jobs.create(source_dir, target_dir, recognition_mode)
                threading.Thread(target=self.server.prepare_job, args=(job.job_id,), daemon=True).start()
                self._send_json(HTTPStatus.OK, self.server.jobs.snapshot(job))
                return
            if parsed.path.endswith("/cancel") and parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.split("/")[3]
                job = self.server.jobs.get(job_id)
                if not job:
                    self._error(HTTPStatus.NOT_FOUND, "任务不存在")
                    return
                if job.status in {"completed", "failed", "canceled"}:
                    self._send_json(HTTPStatus.OK, self.server.jobs.snapshot(job))
                    return
                self.server.jobs.update(job_id, cancel_requested=True, message="正在停止扫描...")
                self._send_json(HTTPStatus.OK, self.server.jobs.snapshot(job))
                return
            if parsed.path.endswith("/commit") and parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.split("/")[3]
                payload = _read_json(self)
                result = self.server.commit_job(job_id, payload.get("items") or [])
                self._send_json(HTTPStatus.OK, result)
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            logging.exception("POST failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _serve_source_image(self, parsed: Any) -> None:
        qs = parse_qs(parsed.query)
        job_id = str((qs.get("job_id") or [""])[0])
        rel = str((qs.get("source_rel_path") or [""])[0])
        max_edge = max(128, min(2048, int((qs.get("max_edge") or ["360"])[0] or "360")))
        job = self.server.jobs.get(job_id)
        if not job:
            self._error(HTTPStatus.NOT_FOUND, "任务不存在")
            return
        fp = (job.source_dir / rel).resolve()
        try:
            fp.relative_to(job.source_dir.resolve())
        except Exception:
            self._error(HTTPStatus.BAD_REQUEST, "图片路径无效")
            return
        if not fp.exists() or not fp.is_file():
            self._error(HTTPStatus.NOT_FOUND, "图片不存在")
            return
        with Image.open(fp) as im0:
            im = im0.convert("RGB")
            im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            import io

            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=72)
        self._send(HTTPStatus.OK, buf.getvalue(), "image/jpeg")


class ImportServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        default_source: str,
        default_target: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.jobs = JobStore()
        self.default_source = default_source
        self.default_target = default_target
        self.tesseract_bin = shutil.which("tesseract")

    def prepare_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        try:
            files = _scan_images(job.source_dir)
            used_names, next_seq = _build_name_allocator(job.target_dir)
            self.jobs.update(job_id, status="running", total=len(files), message=f"发现 {len(files)} 张图片")
            items: list[dict[str, Any]] = []
            for index, path in enumerate(files, start=1):
                latest = self.jobs.get(job_id)
                if not latest or latest.cancel_requested:
                    self.jobs.update(job_id, status="canceled", message=f"已停止扫描，已处理 {index - 1}/{len(files)}")
                    return
                code = ""
                source = ""
                error = ""
                try:
                    code, source = _extract_style(path, self.tesseract_bin, job.recognition_mode)
                except Exception as exc:
                    logging.warning("OCR failed: %s", path, exc_info=True)
                    error = f"OCR 失败：{exc}"
                style_code = _clean_style_code(code)
                valid = _is_valid_style_code(style_code)
                if not code and not error:
                    error = "OCR 未识别到款号"
                elif code and not valid:
                    error = "识别款号必须以字母开头"
                prefix = _code_to_filename_prefix(code) if code else SAFE_STEM_RE.sub("_", path.stem).strip("_") or "UNKNOWN"
                filename = _next_target_name(prefix, path.suffix.lower(), used_names, next_seq)
                rel = str(path.relative_to(job.source_dir)).replace("\\", "/")
                tags, tag_display = _tags_from_source_path(job.source_dir, path)
                unrecognized_copy = ""
                if not valid:
                    try:
                        unrecognized_copy = _copy_unrecognized(job.source_dir, path)
                        if not error:
                            error = "OCR 未识别到可用款号"
                        error = f"{error}；已保存到 {unrecognized_copy}"
                    except Exception as copy_exc:
                        logging.warning("copy unrecognized failed: %s", path, exc_info=True)
                        error = f"{error or 'OCR 未识别到可用款号'}；未识别图片保存失败：{copy_exc}"
                latest = self.jobs.get(job_id)
                if not latest or latest.cancel_requested:
                    self.jobs.update(job_id, status="canceled", message=f"已停止扫描，已处理 {index - 1}/{len(files)}")
                    return
                items.append(
                    {
                        "source_rel_path": rel,
                        "source_name": path.name,
                        "suffix": path.suffix.lower() if path.suffix.lower() in IMAGE_EXTS else ".jpg",
                        "proposed_style_code": style_code,
                        "proposed_filename": filename,
                        "status": "ok" if valid else ("invalid_style_code" if code else "ocr_failed"),
                        "error": error,
                        "ocr_source": source,
                        "tags": tags,
                        "tag_display": tag_display,
                        "unrecognized_copy": unrecognized_copy,
                    }
                )
                self.jobs.update(job_id, processed=index, items=list(items), message=f"已处理 {index}/{len(files)}")
            self.jobs.update(job_id, status="completed", message=f"预处理完成，共 {len(files)} 张", items=items)
        except Exception as exc:
            logging.exception("prepare job failed")
            self.jobs.update(job_id, status="failed", message=str(exc))

    def commit_job(self, job_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            raise RuntimeError("任务不存在")
        if job.status != "completed":
            raise RuntimeError("任务尚未完成识别")
        if job.committed:
            raise RuntimeError("该批次已导入")
        prepared = {str(item.get("source_rel_path")): item for item in job.items}
        used_names, next_seq = _build_name_allocator(job.target_dir)
        hash_index = _ensure_hash_index(job.target_dir)
        imported = 0
        duplicates = 0
        failed: list[dict[str, str]] = []
        manifest_path = job.target_dir / f"_nas_import_manifest_{time.strftime('%Y%m%d_%H%M%S')}_{job.job_id[:8]}.jsonl"
        active_lock_path = job.target_dir / f"_nas_import_active_{time.strftime('%Y%m%d_%H%M%S')}_{job.job_id[:8]}.lock"
        active_lock_path.write_text(
            json.dumps(
                {
                    "job_id": job.job_id,
                    "source_dir": str(job.source_dir),
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        try:
            for item in items:
                if not item.get("selected"):
                    continue
                rel = str(item.get("source_rel_path") or "")
                source = prepared.get(rel)
                if not source:
                    failed.append({"source_rel_path": rel, "error": "源记录不存在"})
                    continue
                src = (job.source_dir / rel).resolve()
                try:
                    src.relative_to(job.source_dir.resolve())
                except Exception:
                    failed.append({"source_rel_path": rel, "error": "源路径无效"})
                    continue
                if not src.exists() or not src.is_file():
                    failed.append({"source_rel_path": rel, "error": "源文件不存在"})
                    continue
                style_code = _clean_style_code(str(item.get("style_code") or ""))
                raw_filename = str(item.get("target_filename") or "").strip()
                if not raw_filename and style_code:
                    raw_filename = f"{style_code}_000{src.suffix.lower()}"
                if not _is_valid_style_code(style_code):
                    failed.append({"source_rel_path": rel, "error": "款号必须以字母开头"})
                    continue
                try:
                    image_hash = _file_hash(src)
                    duplicate_name = hash_index.get(image_hash)
                    if duplicate_name:
                        duplicates += 1
                        logging.info("duplicate image skipped: source=%s existing=%s", rel, duplicate_name)
                        continue
                    target_name = _copy_to_unique_target(src, job.target_dir, raw_filename, used_names, next_seq, style_code)
                    hash_index[image_hash] = target_name
                    _append_hash_index(job.target_dir, image_hash, target_name)
                    imported += 1
                    final_style_code = filename_to_style_code(target_name).strip()
                    tags = list(source.get("tags") or [])
                    row = {
                        "source_rel_path": rel,
                        "image_name": target_name,
                        "style_code": final_style_code,
                        "tags": tags,
                        "imported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "manifest_job_id": job.job_id,
                    }
                    with manifest_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                except Exception as exc:
                    failed.append({"source_rel_path": rel, "error": str(exc)})
        finally:
            try:
                active_lock_path.unlink(missing_ok=True)
            except Exception:
                logging.warning("failed to remove import active lock: %s", active_lock_path, exc_info=True)
        result = {
            "imported": imported,
            "duplicates": duplicates,
            "failed": failed,
            "target_dir": str(job.target_dir),
            "manifest_path": str(manifest_path),
            "hash_index_path": str(job.target_dir / HASH_INDEX_NAME),
        }
        self.jobs.update(job_id, committed=True, result=result, message=f"导入完成：成功 {imported} 张，重复跳过 {duplicates} 张，失败 {len(failed)} 张")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows NAS product image batch importer")
    parser.add_argument("--source", default="/Users/tk/Downloads/产品库", help=r"源目录，例如 Z:\2018\成衣")
    parser.add_argument("--target", default=r"Z:\products\standard_samples", help=r"目标目录，默认 Z:\products\standard_samples")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    server = ImportServer(
        (args.host, args.port),
        ImportHandler,
        default_source=args.source,
        default_target=args.target,
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"NAS 款图批量入库工具已启动：{url}")
    print("按 Ctrl+C 退出。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
