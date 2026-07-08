from __future__ import annotations

import base64
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import uuid
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = os.getenv("LOG_FILE", str(BASE_DIR / "logs" / "garment_ai_server.log")).strip()
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(20 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
Path(LOG_FILE).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        ),
    ],
    force=True,
)

ARK_IMAGE_GENERATION_URL = os.getenv(
    "ARK_IMAGE_GENERATION_URL",
    "https://ark.cn-beijing.volces.com/api/v3/images/generations",
)
ARK_CONTENT_TASKS_URL = os.getenv(
    "ARK_CONTENT_TASKS_URL",
    "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks",
)
ARK_API_KEY = os.getenv("ARK_API_KEY", "").strip()
ARK_MODEL = os.getenv("ARK_IMAGE_MODEL", "doubao-seedream-5-0-260128").strip()
ARK_3D_MODEL = os.getenv("ARK_3D_MODEL", "doubao-seed3d-2-0-260328").strip()
ARK_SIZE = os.getenv("ARK_IMAGE_SIZE", "2K").strip() or "2K"
ARK_OUTPUT_FORMAT = os.getenv("ARK_OUTPUT_FORMAT", "png").strip() or "png"
ARK_WATERMARK = os.getenv("ARK_WATERMARK", "0").strip().lower() in {"1", "true", "yes"}
ARK_3D_FILE_FORMAT = os.getenv("ARK_3D_FILE_FORMAT", "usdz").strip().lower() or "usdz"
ARK_3D_SUBDIVISION = os.getenv("ARK_3D_SUBDIVISION", "low").strip().lower() or "low"
ARK_3D_POLL_INTERVAL = float(os.getenv("ARK_3D_POLL_INTERVAL", "60"))
ARK_3D_TIMEOUT_SEC = float(os.getenv("ARK_3D_TIMEOUT_SEC", "1200"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR = BASE_DIR / "inputs"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GarmentMeasure AI Preview")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static-inputs", StaticFiles(directory=str(INPUT_DIR)), name="static-inputs")


def _truncate(value: str, limit: int = 1800) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _downscale_image(raw: bytes, max_side: int = 1280) -> bytes:
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m > max_side:
        scale = max_side / float(m)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=88, optimize=True)
    return out.getvalue()


def _format_measurements(measurements: dict[str, Any]) -> str:
    labels = {
        "bodyLength": "衣长",
        "shoulderWidth": "肩宽",
        "chestWidth": "胸宽",
        "hemWidth": "下摆宽",
        "leftSleeveLength": "左袖长",
        "rightSleeveLength": "右袖长",
        "neckWidth": "领宽",
    }
    parts: list[str] = []
    for key, label in labels.items():
        value = measurements.get(key)
        if value is None:
            continue
        try:
            parts.append(f"{label} {float(value):.1f} cm")
        except Exception:
            continue
    return "，".join(parts) if parts else "使用标准 T 恤比例"


def _measurement_m(measurements: dict[str, Any], key: str, fallback_cm: float) -> float:
    try:
        return max(1.0, float(measurements.get(key, fallback_cm))) / 100.0
    except Exception:
        return fallback_cm / 100.0


def _dominant_color(raw: bytes) -> list[float]:
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((96, 96))
        pixels = list(img.getdata())
        if not pixels:
            return [0.12, 0.38, 0.28]
        # Avoid white/gray floor dominating simple flat-lay photos.
        colored = []
        for r, g, b in pixels:
            mx, mn = max(r, g, b), min(r, g, b)
            if mx > 236 and mn > 215:
                continue
            if mx - mn < 10 and mx > 170:
                continue
            colored.append((r, g, b))
        sample = colored or pixels
        r = sum(px[0] for px in sample) / len(sample) / 255.0
        g = sum(px[1] for px in sample) / len(sample) / 255.0
        b = sum(px[2] for px in sample) / len(sample) / 255.0
        return [round(r, 4), round(g, 4), round(b, 4)]
    except Exception:
        return [0.12, 0.38, 0.28]


def _shirt_params(measurements: dict[str, Any]) -> dict[str, float]:
    body = _measurement_m(measurements, "bodyLength", 68)
    shoulder = _measurement_m(measurements, "shoulderWidth", 45)
    chest = _measurement_m(measurements, "chestWidth", 52)
    hem = _measurement_m(measurements, "hemWidth", 50)
    sleeve = (_measurement_m(measurements, "leftSleeveLength", 22) + _measurement_m(measurements, "rightSleeveLength", 22)) / 2.0
    neck = _measurement_m(measurements, "neckWidth", 18)

    top_y = body / 2.0
    bottom_y = -body / 2.0
    shoulder_y = top_y - max(0.045, body * 0.08)
    underarm_y = top_y - max(0.16, body * 0.29)
    shoulder_half = shoulder / 2.0
    chest_half = chest / 2.0
    hem_half = hem / 2.0
    sleeve_reach = max(0.12, sleeve * 1.06)
    sleeve_drop = max(0.09, sleeve * 0.68)
    cuff_h = max(0.055, sleeve * 0.36)

    return {
        "body": body,
        "shoulder": shoulder,
        "chest": chest,
        "hem": hem,
        "sleeve": sleeve,
        "neck": neck,
        "top_y": top_y,
        "bottom_y": bottom_y,
        "shoulder_y": shoulder_y,
        "underarm_y": underarm_y,
        "shoulder_half": shoulder_half,
        "chest_half": chest_half,
        "hem_half": hem_half,
        "sleeve_reach": sleeve_reach,
        "sleeve_drop": sleeve_drop,
        "cuff_h": cuff_h,
    }


def _shirt_outline(measurements: dict[str, Any]) -> list[tuple[float, float]]:
    p = _shirt_params(measurements)
    body = p["body"]
    top_y = p["top_y"]
    bottom_y = p["bottom_y"]
    shoulder_y = p["shoulder_y"]
    underarm_y = p["underarm_y"]
    shoulder_half = p["shoulder_half"]
    chest_half = p["chest_half"]
    hem_half = p["hem_half"]
    sleeve_reach = p["sleeve_reach"]
    sleeve_drop = p["sleeve_drop"]
    cuff_h = p["cuff_h"]
    neck = p["neck"]

    # Clockwise front outline, enough points to look curved after triangulation.
    left = [
        (-hem_half, bottom_y),
        (-hem_half * 0.98, bottom_y + body * 0.16),
        (-chest_half * 0.98, underarm_y - body * 0.10),
        (-chest_half, underarm_y),
        (-shoulder_half - sleeve_reach * 0.68, shoulder_y - sleeve_drop - cuff_h),
        (-shoulder_half - sleeve_reach, shoulder_y - sleeve_drop),
        (-shoulder_half - sleeve_reach * 0.55, shoulder_y - sleeve_drop * 0.38),
        (-shoulder_half, shoulder_y),
        (-neck * 0.46, top_y - 0.006),
        (0.0, top_y - max(0.035, neck * 0.24)),
    ]
    right = [(-x, y) for x, y in reversed(left[:-1])]
    return left + right


def _half_width_at_y(y: float, p: dict[str, float]) -> float:
    top_y = p["top_y"]
    bottom_y = p["bottom_y"]
    shoulder_y = p["shoulder_y"]
    underarm_y = p["underarm_y"]
    body = p["body"]

    if y <= underarm_y:
        t = (y - bottom_y) / max(1e-6, underarm_y - bottom_y)
        # Slight waist curve, then hem.
        base = p["hem_half"] * (1.0 - t) + p["chest_half"] * t
        curve = 1.0 - 0.035 * math.sin(max(0.0, min(1.0, t)) * math.pi)
        return base * curve
    if y <= shoulder_y:
        t = (y - underarm_y) / max(1e-6, shoulder_y - underarm_y)
        return p["chest_half"] * (1.0 - t) + p["shoulder_half"] * t
    # Shoulder cap narrows near collar.
    t = (y - shoulder_y) / max(1e-6, top_y - shoulder_y)
    neck_half = p["neck"] * 0.45
    return p["shoulder_half"] * (1.0 - t) + neck_half * t + body * 0.015


def _make_grid_panel(measurements: dict[str, Any], front: bool, depth: float) -> tuple[list[list[float]], list[list[int]]]:
    p = _shirt_params(measurements)
    rows = 18
    cols = 14
    vertices: list[list[float]] = []
    for row in range(rows + 1):
        v = row / rows
        y = p["bottom_y"] * (1 - v) + p["top_y"] * v
        half = _half_width_at_y(y, p)
        for col in range(cols + 1):
            u = col / cols
            x = -half + 2.0 * half * u
            nx = x / max(1e-6, p["chest_half"])
            ny = (y - (p["bottom_y"] + p["body"] * 0.52)) / max(1e-6, p["body"])
            bulge = 0.026 * math.exp(-(nx * nx * 1.4 + ny * ny * 5.0))
            wrinkle = 0.004 * math.sin((u * 5.5 + v * 1.7) * math.pi) * math.sin(v * math.pi)
            z = (depth / 2.0 + bulge + wrinkle) if front else -depth / 2.0
            vertices.append([round(x, 5), round(y, 5), round(z, 5)])

    triangles: list[list[int]] = []
    width = cols + 1
    for row in range(rows):
        for col in range(cols):
            a = row * width + col
            b = a + 1
            c = a + width
            d = c + 1
            if front:
                triangles.append([a, c, b])
                triangles.append([b, c, d])
            else:
                triangles.append([a, b, c])
                triangles.append([b, d, c])
    return vertices, triangles


def _append_sleeve(
    vertices: list[list[float]],
    triangles: list[list[int]],
    p: dict[str, float],
    depth: float,
    side: int,
) -> None:
    base = len(vertices)
    shoulder_x = side * p["shoulder_half"]
    shoulder_y = p["shoulder_y"]
    underarm_x = side * p["chest_half"]
    underarm_y = p["underarm_y"]
    reach = p["sleeve_reach"]
    drop = p["sleeve_drop"]
    cuff_h = p["cuff_h"]
    outer_top = (shoulder_x + side * reach, shoulder_y - drop)
    outer_bottom = (shoulder_x + side * reach * 0.70, shoulder_y - drop - cuff_h)

    pts = [
        (shoulder_x, shoulder_y, depth / 2.0 + 0.008),
        (outer_top[0], outer_top[1], depth / 2.0 + 0.004),
        (outer_bottom[0], outer_bottom[1], depth / 2.0 + 0.002),
        (underarm_x, underarm_y, depth / 2.0 + 0.006),
        (shoulder_x * 0.98, shoulder_y - 0.012, -depth / 2.0),
        (outer_top[0], outer_top[1], -depth / 2.0),
        (outer_bottom[0], outer_bottom[1], -depth / 2.0),
        (underarm_x * 0.98, underarm_y, -depth / 2.0),
    ]
    vertices.extend([[round(x, 5), round(y, 5), round(z, 5)] for x, y, z in pts])
    triangles.extend([
        [base + 0, base + 3, base + 1],
        [base + 1, base + 3, base + 2],
        [base + 4, base + 5, base + 7],
        [base + 5, base + 6, base + 7],
        [base + 0, base + 1, base + 5],
        [base + 0, base + 5, base + 4],
        [base + 1, base + 2, base + 6],
        [base + 1, base + 6, base + 5],
        [base + 2, base + 3, base + 7],
        [base + 2, base + 7, base + 6],
    ])


def _make_parametric_mesh(image_bytes: bytes, measurements: dict[str, Any]) -> dict[str, Any]:
    p = _shirt_params(measurements)
    depth = max(0.035, _measurement_m(measurements, "chestWidth", 52) * 0.18)
    vertices, triangles = _make_grid_panel(measurements, front=True, depth=depth)
    back_vertices, back_triangles = _make_grid_panel(measurements, front=False, depth=depth)
    offset = len(vertices)
    vertices.extend(back_vertices)
    triangles.extend([[a + offset, b + offset, c + offset] for a, b, c in back_triangles])

    cols = 14
    rows = 18
    width = cols + 1
    # Side strips connect front/back edges.
    for row in range(rows):
        for col in (0, cols):
            a = row * width + col
            b = (row + 1) * width + col
            c = offset + row * width + col
            d = offset + (row + 1) * width + col
            triangles.append([a, b, c])
            triangles.append([b, d, c])
    for row in (0, rows):
        for col in range(cols):
            a = row * width + col
            b = row * width + col + 1
            c = offset + row * width + col
            d = offset + row * width + col + 1
            triangles.append([a, c, b])
            triangles.append([b, c, d])

    _append_sleeve(vertices, triangles, p, depth, side=-1)
    _append_sleeve(vertices, triangles, p, depth, side=1)

    return {
        "vertices": vertices,
        "triangles": triangles,
        "base_color": _dominant_color(image_bytes),
        "metadata": {
            "kind": "parametric_tshirt_mesh",
            "vertex_count": len(vertices),
            "triangle_count": len(triangles),
        },
    }


def _find_first_url(value: Any) -> str:
    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            return value
        return ""
    if isinstance(value, dict):
        preferred_keys = (
            "model_url",
            "asset_url",
            "mesh_url",
            "glb_url",
            "usdz_url",
            "obj_url",
            "url",
            "download_url",
            "file_url",
        )
        for key in preferred_keys:
            found = _find_first_url(value.get(key))
            if found:
                return found
        for item in value.values():
            found = _find_first_url(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_first_url(item)
            if found:
                return found
    return ""


def _extract_task_id(data: dict[str, Any]) -> str:
    for key in ("id", "task_id", "taskId"):
        value = data.get(key)
        if value:
            return str(value)
    nested = data.get("data")
    if isinstance(nested, dict):
        for key in ("id", "task_id", "taskId"):
            value = nested.get(key)
            if value:
                return str(value)
    raise HTTPException(status_code=502, detail=f"Seed3D 未返回任务 ID: {data}")


def _task_status(data: dict[str, Any]) -> str:
    candidates: list[Any] = [
        data.get("status"),
        data.get("state"),
        data.get("task_status"),
        data.get("taskStatus"),
    ]
    if isinstance(data.get("data"), dict):
        d = data["data"]
        candidates.extend([d.get("status"), d.get("state"), d.get("task_status"), d.get("taskStatus")])
    for item in candidates:
        if item:
            return str(item).lower()
    return ""


def _ark_request_json(url: str, payload: dict[str, Any] | None = None, method: str = "POST") -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={
            "Authorization": f"Bearer {ARK_API_KEY}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        logging.info("ark request start method=%s url=%s", method, url)
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            logging.info("ark request ok method=%s url=%s", method, url)
            return data
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")
        logging.error("ark request error method=%s url=%s status=%s body=%s", method, url, exc.code, _truncate(err, 4000))
        raise HTTPException(status_code=502, detail=f"火山方舟调用失败: HTTP {exc.code} {_truncate(err, 1000)}") from exc
    except Exception as exc:
        logging.exception("ark request failed method=%s url=%s", method, url)
        raise HTTPException(status_code=502, detail=f"火山方舟调用失败: {exc}") from exc


def _is_image_fetch_timeout(exc: HTTPException) -> bool:
    detail = str(exc.detail or "").lower()
    return "content[1].image_url" in detail and "timeout while fetching resource" in detail


def _create_seed3d_task(payload: dict[str, Any]) -> dict[str, Any]:
    delays = [0, 8, 16, 30]
    last_error: HTTPException | None = None
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        try:
            logging.info("seed3d create task attempt=%s", attempt)
            return _ark_request_json(ARK_CONTENT_TASKS_URL, payload, method="POST")
        except HTTPException as exc:
            last_error = exc
            if not _is_image_fetch_timeout(exc) or attempt == len(delays):
                raise
            logging.warning("seed3d image fetch timed out, retrying attempt=%s detail=%s", attempt, _truncate(exc.detail, 600))
    raise last_error or HTTPException(status_code=502, detail="Seed3D 创建任务失败")


def _prepare_model_file(result_bytes: bytes, fallback_ext: str) -> tuple[bytes, str, str]:
    ext = fallback_ext.strip(".").lower() or "bin"
    if result_bytes[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(result_bytes)) as zf:
            names = zf.namelist()
            preferred = [n for n in names if n.lower().endswith((".usdz", ".glb", ".obj"))]
            if preferred:
                requested_suffix = f".{ext}"
                name = sorted(preferred, key=lambda n: (not n.lower().endswith(requested_suffix), len(n)))[0]
                ext = Path(name).suffix.lower().lstrip(".") or ext
                return zf.read(name), ext, Path(name).name
        return result_bytes, "zip", "seed3d_result.zip"
    return result_bytes, ext, f"seed3d_result.{ext}"


def _public_image_url(image_bytes: bytes) -> str:
    if not PUBLIC_BASE_URL:
        raise HTTPException(
            status_code=503,
            detail="Seed3D 需要公网可访问图片 URL；请设置 PUBLIC_BASE_URL，例如 https://你的域名",
        )
    name = f"{uuid.uuid4().hex}.jpg"
    path = INPUT_DIR / name
    path.write_bytes(_downscale_image(image_bytes, max_side=1280))
    return f"{PUBLIC_BASE_URL}/static-inputs/{name}"


def _call_seed3d(image_bytes: bytes) -> dict[str, Any]:
    if not ARK_API_KEY:
        raise HTTPException(status_code=503, detail="缺少 ARK_API_KEY，无法调用 Seed3D")

    file_format = ARK_3D_FILE_FORMAT.strip(".").lower()
    image_url = _public_image_url(image_bytes)
    prompt = f"--subdivisionlevel {ARK_3D_SUBDIVISION} --fileformat {file_format}"
    payload = {
        "model": ARK_3D_MODEL,
        "content": [
            {"type": "text", "text": f" {prompt} "},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }
    logging.info("seed3d create task model=%s prompt=%s image_url=%s", ARK_3D_MODEL, prompt, image_url)
    created = _create_seed3d_task(payload)
    task_id = _extract_task_id(created)
    logging.info("seed3d task created id=%s raw=%s", task_id, _truncate(json.dumps(created, ensure_ascii=False), 1200))
    task_url = f"{ARK_CONTENT_TASKS_URL}/{task_id}"

    started_at = time.time()
    deadline = started_at + ARK_3D_TIMEOUT_SEC
    last_data: dict[str, Any] = created
    while time.time() < deadline:
        data = _ark_request_json(task_url, None, method="GET")
        last_data = data
        status = _task_status(data)
        elapsed = int(time.time() - started_at)
        remaining = max(0, int(deadline - time.time()))
        logging.info(
            "seed3d task id=%s status=%s elapsed=%ss remaining=%ss timeout=%ss",
            task_id,
            status or "(unknown)",
            elapsed,
            remaining,
            int(ARK_3D_TIMEOUT_SEC),
        )
        if status in {"succeeded", "success", "completed", "done"}:
            model_url = _find_first_url(data)
            if not model_url:
                raise HTTPException(status_code=502, detail=f"Seed3D 成功但未找到模型 URL: {data}")
            logging.info("seed3d downloading result id=%s url=%s", task_id, model_url)
            with urllib.request.urlopen(model_url, timeout=240) as resp:
                result_bytes = resp.read()
            model_bytes, ext, file_name = _prepare_model_file(result_bytes, file_format)
            logging.info("seed3d result downloaded id=%s file=%s bytes=%s", task_id, file_name, len(model_bytes))
            return {
                "task_id": task_id,
                "bytes": model_bytes,
                "ext": ext,
                "file_name": file_name,
                "source_url": model_url,
                "raw_task": data,
            }
        if status in {"failed", "fail", "error", "cancelled", "canceled"}:
            raise HTTPException(status_code=502, detail=f"Seed3D 任务失败: {data}")
        time.sleep(ARK_3D_POLL_INTERVAL)

    raise HTTPException(status_code=504, detail=f"Seed3D 任务超时，已等待 {int(ARK_3D_TIMEOUT_SEC)} 秒: {last_data}")


def _build_prompt(measurements: dict[str, Any]) -> str:
    dims = _format_measurements(measurements)
    return (
        "参考输入图片中的 T 恤颜色、印花、Logo、领口、袖型和面料纹理，"
        "生成一张干净高级的 3D 商品渲染图。"
        "要求：单件短袖 T 恤，正面略微 3/4 角度，悬浮在深色中性背景上，"
        "真实布料褶皱，柔和棚拍光影，边缘清晰，不要人物、不要衣架、不要文字说明、不要尺寸标注、不要多件衣服。"
        f"版型尺寸参考：{dims}。"
        "输出应像电商 3D 服装预览，不要保留照片背景。"
    )


def _call_ark(image_bytes: bytes, measurements: dict[str, Any]) -> bytes:
    if not ARK_API_KEY:
        raise HTTPException(status_code=503, detail="缺少 ARK_API_KEY，无法调用火山方舟")

    src_b64 = base64.b64encode(_downscale_image(image_bytes)).decode("ascii")
    payload = {
        "model": ARK_MODEL,
        "prompt": _build_prompt(measurements),
        "image": f"data:image/jpeg;base64,{src_b64}",
        "sequential_image_generation": "disabled",
        "size": ARK_SIZE,
        "output_format": ARK_OUTPUT_FORMAT,
        "watermark": ARK_WATERMARK,
    }

    logging.info("ark garment preview request model=%s size=%s prompt=%s", ARK_MODEL, ARK_SIZE, _truncate(payload["prompt"]))
    req = urllib.request.Request(
        ARK_IMAGE_GENERATION_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {ARK_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")
        logging.error("ark garment preview http error status=%s body=%s", exc.code, _truncate(err, 4000))
        raise HTTPException(status_code=502, detail=f"火山方舟调用失败: HTTP {exc.code} {_truncate(err, 800)}") from exc
    except Exception as exc:
        logging.exception("ark garment preview request failed")
        raise HTTPException(status_code=502, detail=f"火山方舟调用失败: {exc}") from exc

    images = data.get("data") or data.get("images") or []
    if not images or not isinstance(images[0], dict) or not images[0].get("url"):
        logging.error("ark garment preview missing url response=%s", _truncate(json.dumps(data, ensure_ascii=False), 4000))
        raise HTTPException(status_code=502, detail="火山方舟未返回图片链接")

    result_url = str(images[0]["url"])
    try:
        with urllib.request.urlopen(result_url, timeout=180) as result_resp:
            return result_resp.read()
    except Exception as exc:
        logging.exception("ark result download failed url=%s", _truncate(result_url, 1000))
        raise HTTPException(status_code=502, detail=f"下载生成结果失败: {exc}") from exc


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "provider": "volcengine_ark",
        "image_model": ARK_MODEL,
        "seed3d_model": ARK_3D_MODEL,
        "seed3d_file_format": ARK_3D_FILE_FORMAT,
        "public_base_url": bool(PUBLIC_BASE_URL),
        "has_api_key": bool(ARK_API_KEY),
    }


@app.post("/api/v1/garment/ai-preview")
async def create_garment_ai_preview(
    file: UploadFile = File(...),
    measurements: str = Form("{}"),
) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        dims = json.loads(measurements or "{}")
        if not isinstance(dims, dict):
            dims = {}
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid measurements json") from exc

    result_bytes = await run_in_threadpool(_call_ark, raw, dims)
    job_id = uuid.uuid4().hex
    ext = "png" if ARK_OUTPUT_FORMAT.lower() == "png" else "jpg"
    out_path = OUTPUT_DIR / f"{job_id}.{ext}"
    out_path.write_bytes(result_bytes)

    return {
        "job_id": job_id,
        "image_url": f"/api/v1/garment/ai-preview/{out_path.name}",
        "image_base64": base64.b64encode(result_bytes).decode("ascii"),
        "mime": "image/png" if ext == "png" else "image/jpeg",
        "used_params": {
            "provider": "volcengine_ark",
            "model": ARK_MODEL,
            "size": ARK_SIZE,
        },
    }


@app.post("/api/v1/garment/model")
async def create_garment_model(
    file: UploadFile = File(...),
    measurements: str = Form("{}"),
) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        dims = json.loads(measurements or "{}")
        if not isinstance(dims, dict):
            dims = {}
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid measurements json") from exc

    if ARK_API_KEY:
        try:
            asset = await run_in_threadpool(_call_seed3d, raw)
            job_id = uuid.uuid4().hex
            ext = str(asset["ext"])
            out_path = OUTPUT_DIR / f"{job_id}.{ext}"
            out_path.write_bytes(asset["bytes"])
            mesh = _make_parametric_mesh(raw, dims)
            logging.info("garment model response ready job=%s file=%s bytes=%s", job_id, out_path.name, len(asset["bytes"]))
            return {
                "job_id": job_id,
                "provider": "volcengine_seed3d",
                "model_url": f"/api/v1/garment/model/{out_path.name}",
                "mesh": mesh,
                "file_name": asset["file_name"],
                "file_ext": ext,
                "source_url": asset["source_url"],
                "used_params": {
                    "model": ARK_3D_MODEL,
                    "file_format": ARK_3D_FILE_FORMAT,
                    "subdivision": ARK_3D_SUBDIVISION,
                    "task_id": asset["task_id"],
                },
            }
        except HTTPException:
            raise
        except Exception as exc:
            logging.exception("seed3d failed, fallback to parametric mesh")
            if os.getenv("ARK_3D_STRICT", "0").strip().lower() in {"1", "true", "yes"}:
                raise HTTPException(status_code=502, detail=f"Seed3D 生成失败: {exc}") from exc

    mesh = _make_parametric_mesh(raw, dims)
    job_id = uuid.uuid4().hex
    mesh_path = OUTPUT_DIR / f"{job_id}.json"
    mesh_path.write_text(json.dumps(mesh, ensure_ascii=False), encoding="utf-8")
    return {
        "job_id": job_id,
        "model_url": f"/api/v1/garment/model/{mesh_path.name}",
        "mesh": mesh,
    }


@app.get("/api/v1/garment/model/{name}")
def get_model(name: str) -> FileResponse:
    safe = Path(name).name
    path = OUTPUT_DIR / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    media_types = {
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".usdz": "model/vnd.usdz+zip",
        ".obj": "text/plain",
        ".json": "application/json",
    }
    return FileResponse(path, media_type=media_types.get(path.suffix.lower(), "application/octet-stream"))


@app.get("/api/v1/garment/ai-preview/{name}")
def get_preview(name: str) -> FileResponse:
    safe = Path(name).name
    path = OUTPUT_DIR / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media_type)
