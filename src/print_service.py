from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

PaperSize = Literal["A4", "A5", "4R"]
TemplateId = Literal["single_full", "two_vertical", "two_horizontal", "grid_2x2", "grid_3x3"]

MM_TO_PT = 2.83464567
MAX_IMAGE_EDGE = 2400
JPEG_QUALITY = 88

BASE_DIR = Path(__file__).resolve().parent
PRINT_DIR = BASE_DIR / "print_runtime"
PRINT_STATIC_DIR = PRINT_DIR / "static"
PRINT_EXPORT_DIR = PRINT_STATIC_DIR / "exports"
PRINT_PREVIEW_DIR = PRINT_STATIC_DIR / "previews"
PRINT_STORAGE_DIR = PRINT_DIR / "storage"
PRINT_ORIGINAL_DIR = PRINT_STORAGE_DIR / "originals"
PRINT_PROCESSED_DIR = PRINT_STORAGE_DIR / "processed"

for _folder in [PRINT_EXPORT_DIR, PRINT_PREVIEW_DIR, PRINT_ORIGINAL_DIR, PRINT_PROCESSED_DIR]:
    _folder.mkdir(parents=True, exist_ok=True)


@dataclass
class ImageMeta:
    image_id: str
    original_path: str
    processed_path: str
    width: int
    height: int


IMAGES: dict[str, ImageMeta] = {}


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float


PAPER_SIZES_MM: dict[PaperSize, tuple[float, float]] = {
    "A4": (210, 297),
    "A5": (148, 210),
    "4R": (102, 152),
}


def page_size_pt(paper_size: PaperSize) -> tuple[float, float]:
    w_mm, h_mm = PAPER_SIZES_MM[paper_size]
    return w_mm * MM_TO_PT, h_mm * MM_TO_PT


def template_slots(template_id: TemplateId) -> int:
    return {
        "single_full": 1,
        "two_vertical": 2,
        "two_horizontal": 2,
        "grid_2x2": 4,
        "grid_3x3": 9,
    }[template_id]


def build_slots(page_w: float, page_h: float, template_id: TemplateId) -> list[Rect]:
    margin = 0
    inner_w = page_w - margin * 2
    inner_h = page_h - margin * 2
    gap = 0

    if template_id == "single_full":
        return [Rect(margin, margin, inner_w, inner_h)]

    if template_id == "two_vertical":
        cols, rows = 1, 2
    elif template_id == "two_horizontal":
        cols, rows = 2, 1
    elif template_id == "grid_2x2":
        cols, rows = 2, 2
    else:
        cols, rows = 3, 3

    cell_w = (inner_w - gap * (cols - 1)) / cols
    cell_h = (inner_h - gap * (rows - 1)) / rows

    slots: list[Rect] = []
    for r in range(rows):
        for c in range(cols):
            x = margin + c * (cell_w + gap)
            y = margin + (rows - 1 - r) * (cell_h + gap)
            slots.append(Rect(x, y, cell_w, cell_h))
    return slots


def process_upload(file_bytes: bytes, suffix: str = ".jpg") -> dict:
    image_id = uuid.uuid4().hex
    original_path = PRINT_ORIGINAL_DIR / f"{image_id}{suffix}"
    processed_path = PRINT_PROCESSED_DIR / f"{image_id}.jpg"

    original_path.write_bytes(file_bytes)

    with Image.open(original_path) as img:
        normalized = ImageOps.exif_transpose(img)
        normalized.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        rgb = normalized.convert("RGB")
        rgb.save(processed_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        width, height = rgb.size

    IMAGES[image_id] = ImageMeta(
        image_id=image_id,
        original_path=str(original_path),
        processed_path=str(processed_path),
        width=width,
        height=height,
    )

    return {
        "image_id": image_id,
        "original_url": f"/print-storage/originals/{original_path.name}",
        "processed_url": f"/print-storage/processed/{processed_path.name}",
        "width": width,
        "height": height,
    }


def _contain_resize(source: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = source.size
    scale = min(target_w / src_w, target_h / src_h)
    draw_w = max(1, int(src_w * scale))
    draw_h = max(1, int(src_h * scale))
    resized = source.resize((draw_w, draw_h), Image.Resampling.LANCZOS)
    offset_x = (target_w - draw_w) // 2
    offset_y = (target_h - draw_h) // 2
    canvas_img = Image.new("RGB", (target_w, target_h), color=(255, 255, 255))
    canvas_img.paste(resized, (offset_x, offset_y))
    return canvas_img


def render_layout(payload: dict) -> dict:
    paper_size = str(payload.get("paper_size", "A4"))
    template_id = str(payload.get("template_id", "grid_2x2"))
    placements = payload.get("placements", []) or []
    auto_fill = bool(payload.get("auto_fill", True))

    if paper_size not in {"A4", "A5", "4R"}:
        raise ValueError("paper_size 仅支持 A4/A5/4R")
    if template_id not in {"single_full", "two_vertical", "two_horizontal", "grid_2x2", "grid_3x3"}:
        raise ValueError("template_id 不合法")
    if not isinstance(placements, list) or not placements:
        raise ValueError("placements 不能为空")

    page_w, page_h = page_size_pt(paper_size)
    slots = build_slots(page_w, page_h, template_id)

    assigned: list[str | None] = [None for _ in slots]
    for p in placements:
        if not isinstance(p, dict):
            continue
        image_id = str(p.get("image_id", "")).strip()
        try:
            slot_index = int(p.get("slot_index", -1))
        except Exception:
            slot_index = -1
        if image_id and 0 <= slot_index < len(slots):
            assigned[slot_index] = image_id

    used = [image_id for image_id in assigned if image_id]
    if not used:
        for p in placements:
            image_id = str((p or {}).get("image_id", "")).strip() if isinstance(p, dict) else ""
            if image_id:
                used.append(image_id)

    if auto_fill and used:
        fill_idx = 0
        for idx, image_id in enumerate(assigned):
            if image_id is None:
                assigned[idx] = used[fill_idx % len(used)]
                fill_idx += 1

    job_id = uuid.uuid4().hex
    pdf_path = PRINT_EXPORT_DIR / f"{job_id}.pdf"
    preview_path = PRINT_PREVIEW_DIR / f"{job_id}.jpg"

    c = canvas.Canvas(str(pdf_path), pagesize=(page_w, page_h))
    c.setTitle(f"print-layout-{job_id}")

    bg = Image.new("RGB", (int(page_w), int(page_h)), color=(250, 250, 250))

    for idx, image_id in enumerate(assigned):
        if not image_id:
            continue
        image_meta = IMAGES.get(image_id)
        if not image_meta:
            continue

        slot = slots[idx]
        with Image.open(image_meta.processed_path) as img:
            slot_w = max(1, int(slot.width))
            slot_h = max(1, int(slot.height))
            slot_canvas = _contain_resize(img.convert("RGB"), slot_w, slot_h)
            c.drawImage(ImageReader(slot_canvas), slot.x, slot.y, slot.width, slot.height, mask="auto")

            px = int(slot.x)
            py = int(page_h - slot.y - slot.height)
            bg.paste(slot_canvas, (px, py))

    c.showPage()
    c.save()

    bg.save(preview_path, quality=90)

    return {
        "job_id": job_id,
        "paper_size": paper_size,
        "template_id": template_id,
        "pdf_url": f"/print-static/exports/{pdf_path.name}",
        "preview_url": f"/print-static/previews/{preview_path.name}",
        "page_count": 1,
    }


def list_templates() -> list[dict]:
    return [
        {"template_id": "single_full", "name": "单图铺满", "description": "适合证件照/海报风", "slots": template_slots("single_full")},
        {"template_id": "two_vertical", "name": "两张竖排", "description": "一页上下两张，适合对比图或双联排版", "slots": template_slots("two_vertical")},
        {"template_id": "two_horizontal", "name": "两张横排", "description": "一页左右两张，适合并排展示", "slots": template_slots("two_horizontal")},
        {"template_id": "grid_2x2", "name": "2x2 网格", "description": "一页四张，适合拼版", "slots": template_slots("grid_2x2")},
        {"template_id": "grid_3x3", "name": "3x3 网格", "description": "一页九张，适合小尺寸批量", "slots": template_slots("grid_3x3")},
    ]
