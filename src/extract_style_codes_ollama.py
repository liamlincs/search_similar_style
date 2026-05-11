import argparse
import base64
import json
import logging
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, ImageEnhance, ImageOps

STYLE_RE = re.compile(r"([A-Za-z0-9_-]+#)")
SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")
DEFAULT_CONFIG = Path("config/search_config.json")


def crop_header_gray(img_path: Path) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    crop = img.crop((0, 0, int(w * 0.45), int(h * 0.12)))
    gray = ImageOps.grayscale(crop)
    gray = ImageEnhance.Contrast(gray).enhance(2.8)
    return gray


def to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def call_ollama_ocr_chat(img_bytes: bytes, model: str, host: str, timeout_sec: int) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = (
        "只提取图片左上角第一行款号，并且只返回一个字符串。\n"
        "返回格式必须匹配: [A-Za-z0-9_-]+#\n"
        "如果无法识别，严格返回: UNKNOWN#\n"
        "禁止输出解释、禁止换行、禁止其他字符。"
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0},
        "keep_alive": "30m",
    }
    resp = requests.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message", {}) if isinstance(data, dict) else {}
    return str(msg.get("content", "")).strip()


def normalize_code(text: str, fallback_stem: str) -> str:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for ln in lines:
        m = STYLE_RE.search(ln)
        if m:
            return m.group(1)
        if ln.endswith("#"):
            return ln
        if "#" in ln:
            token = ln[: ln.find("#") + 1].strip()
            if token:
                return token
    return f"UNKNOWN#{fallback_stem}"


def try_extract_code_from_image(gray: Image.Image, model: str, host: str, timeout_sec: int) -> Optional[str]:
    # Multi-threshold retry; accept only strict code regex.
    arr = ImageOps.autocontrast(gray)
    for th in (140, 160, 180):
        bw = arr.point(lambda p: 255 if p > th else 0, mode="1").convert("L")
        raw = call_ollama_ocr_chat(to_png_bytes(bw), model, host, timeout_sec)
        code = normalize_code(raw, "UNKNOWN")
        if re.fullmatch(r"[A-Za-z0-9_-]+#", code):
            return code
        if code == "UNKNOWN#":
            continue
    return None


def code_to_filename_prefix(code: str) -> str:
    # "AB12#" -> "AB12"; keep only [A-Za-z0-9_-]
    core = code[:-1] if code.endswith("#") else code
    core = SAFE_RE.sub("_", core).strip("_")
    return core if core else "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(description="提取左上角款号并重命名为 款号_000.png")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--standard-dir", type=Path, default=Path("data/standard_samples"))
    parser.add_argument("--pattern", type=str, default="B*.png")
    parser.add_argument("--model", type=str, default="deepseek-ocr")
    parser.add_argument("--host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.config.exists():
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        ollama_cfg = cfg.get("ollama", {})
        path_cfg = cfg.get("paths", {})
        if args.standard_dir == Path("data/standard_samples") and path_cfg.get("standard_dir"):
            args.standard_dir = Path(path_cfg["standard_dir"])
        if args.model == "deepseek-ocr" and ollama_cfg.get("model"):
            args.model = str(ollama_cfg["model"])
        if args.host == "http://127.0.0.1:11434" and ollama_cfg.get("host"):
            args.host = str(ollama_cfg["host"])
        if args.timeout == 60 and ollama_cfg.get("timeout_sec") is not None:
            args.timeout = int(ollama_cfg["timeout_sec"])

    files = sorted(args.standard_dir.glob(args.pattern))
    if not files:
        files = sorted(args.standard_dir.glob("*.png"))
    if not files:
        raise RuntimeError(f"no standard images found in {args.standard_dir}")

    # 1) OCR -> style code (failed OCR will be skipped, not renamed)
    plan = []
    skipped = []
    seq = defaultdict(int)
    for p in files:
        try:
            gray = crop_header_gray(p)
            code = try_extract_code_from_image(gray, args.model, args.host, args.timeout)
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

    # avoid collisions during rename by 2-phase temp names
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
