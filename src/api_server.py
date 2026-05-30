import json
import logging
import sys
import tempfile
import base64
import time
import os
import hmac
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from search_similar_return_code import (
    DEFAULT_CONFIG,
    build_feature_db_with_cache,
    build_label_memory_prior_from_refs,
    precompute_label_memory_refs,
    precompute_rerank_candidate_cache,
    rerank_candidates_with_model,
    search_topk_images,
    topk_style_codes,
    try_extract_query_style_code,
)
from print_service import PRINT_STATIC_DIR, PRINT_STORAGE_DIR, list_templates, process_upload, render_layout
from recolor_service import RECOLOR_OUTPUT_DIR, recolor_region, recolor_region_ai


class SearchResponse(BaseModel):
    query_image: str
    topk_style_codes: List[Dict[str, Any]]


class ImageUrlResponse(BaseModel):
    image_name: str
    image_url: str
    expires_at: int


def _load_cfg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _setup_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "api_server.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers)
    has_file = any(isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")).name == log_path.name for h in root.handlers)

    if not has_stream:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if not has_file:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def create_app(config_path: Path = DEFAULT_CONFIG) -> FastAPI:
    cfg = _load_cfg(config_path)
    path_cfg = cfg.get("paths", {})
    search_cfg = cfg.get("search", {})

    standard_dir = Path(path_cfg.get("standard_dir", "data/standard_samples"))
    standard_pattern = str(path_cfg.get("standard_pattern", "*"))
    image_exts = list(path_cfg.get("image_exts", ["png", "jpg", "jpeg"]))

    top_k = int(search_cfg.get("top_k", 5))
    candidate_multiplier = int(search_cfg.get("candidate_multiplier", 10))
    feature_backend = str(search_cfg.get("feature_backend", "hybrid"))
    min_score = float(search_cfg.get("min_score", 0.64))
    code_agg_top_n = int(search_cfg.get("code_agg_top_n", 3))
    code_agg_alpha = float(search_cfg.get("code_agg_alpha", 0.7))
    query_multicrop = bool(search_cfg.get("query_multicrop", True))
    query_crop_ratio = float(search_cfg.get("query_crop_ratio", 0.72))
    query_component_views = bool(search_cfg.get("query_component_views", False))
    standard_multicrop = bool(search_cfg.get("standard_multicrop", False))
    standard_crop_ratio = float(search_cfg.get("standard_crop_ratio", 0.72))
    ocr_hint_enabled = bool(search_cfg.get("ocr_hint_enabled", False))
    ocr_hint_boost = float(search_cfg.get("ocr_hint_boost", 0.08))
    rerank_enabled = bool(search_cfg.get("rerank_enabled", True))
    rerank_topn = int(search_cfg.get("rerank_topn", 12))
    rerank_weight = float(search_cfg.get("rerank_weight", 0.45))
    reranker_model = Path(search_cfg.get("reranker_model", "models/reranker_v1.npz"))
    rerank_query_views_max = int(search_cfg.get("rerank_query_views_max", 2))
    rerank_candidate_views_max = int(search_cfg.get("rerank_candidate_views_max", 1))
    label_memory_enabled = bool(search_cfg.get("label_memory_enabled", True))
    label_memory_path = Path(search_cfg.get("label_memory_path", "data/query_labels.json"))
    label_memory_sim_threshold = float(search_cfg.get("label_memory_sim_threshold", 0.9))
    label_memory_max_boost = float(search_cfg.get("label_memory_max_boost", 0.09))
    hybrid_weights = search_cfg.get("hybrid_weights", {})
    w_clip = float(hybrid_weights.get("clip", 0.55))
    w_shape = float(hybrid_weights.get("shape", 0.30))
    w_color = float(hybrid_weights.get("color", 0.15))
    w_stripe = float(hybrid_weights.get("stripe", 0.10))
    feature_cache_enabled = bool(search_cfg.get("feature_cache_enabled", True))
    db_feature_dtype = str(search_cfg.get("db_feature_dtype", "float32")).lower()
    recall_topn_cap = int(search_cfg.get("recall_topn_cap", 0))
    preload_rerank_candidate_cache = bool(search_cfg.get("preload_rerank_candidate_cache", False))
    rerank_max_unique_codes = int(search_cfg.get("rerank_max_unique_codes", 0))
    result_image_max_edge = int(search_cfg.get("result_image_max_edge", 0))
    result_image_quality = int(search_cfg.get("result_image_quality", 82))
    adaptive_second_pass_enabled = bool(search_cfg.get("adaptive_second_pass_enabled", False))
    adaptive_trigger_top1_below = float(search_cfg.get("adaptive_trigger_top1_below", 0.72))
    adaptive_trigger_margin_below = float(search_cfg.get("adaptive_trigger_margin_below", 0.02))
    adaptive_recall_topn_cap = int(search_cfg.get("adaptive_recall_topn_cap", 1024))
    adaptive_candidate_multiplier = int(search_cfg.get("adaptive_candidate_multiplier", 12))
    adaptive_query_component_views = bool(search_cfg.get("adaptive_query_component_views", True))
    adaptive_rerank_topn = int(search_cfg.get("adaptive_rerank_topn", 36))
    adaptive_rerank_query_views_max = int(search_cfg.get("adaptive_rerank_query_views_max", 2))
    adaptive_rerank_max_unique_codes = int(search_cfg.get("adaptive_rerank_max_unique_codes", 24))
    query_view_consensus_weight = float(search_cfg.get("query_view_consensus_weight", 0.0))
    adaptive_query_view_consensus_weight = float(search_cfg.get("adaptive_query_view_consensus_weight", 0.35))
    shape_consistency_enabled = bool(search_cfg.get("shape_consistency_enabled", False))
    shape_consistency_aspect_weight = float(search_cfg.get("shape_consistency_aspect_weight", 0.10))
    shape_consistency_fill_weight = float(search_cfg.get("shape_consistency_fill_weight", 0.04))
    shape_consistency_apply_topn = int(search_cfg.get("shape_consistency_apply_topn", 256))
    strip_mode_enabled = bool(search_cfg.get("strip_mode_enabled", True))
    strip_aspect_threshold = float(search_cfg.get("strip_aspect_threshold", 2.4))
    strip_fill_threshold = float(search_cfg.get("strip_fill_threshold", 0.42))
    strip_w_clip = float(search_cfg.get("strip_w_clip", 0.35))
    strip_w_shape = float(search_cfg.get("strip_w_shape", 0.30))
    strip_w_color = float(search_cfg.get("strip_w_color", 0.10))
    strip_w_stripe = float(search_cfg.get("strip_w_stripe", 0.25))
    auth_cfg = cfg.get("auth", {})
    api_key_enabled = bool(auth_cfg.get("enabled", True))
    image_url_secret = str(auth_cfg.get("image_url_secret", "")).strip()
    image_url_ttl_sec = int(auth_cfg.get("image_url_ttl_sec", 600))
    api_keys_cfg = auth_cfg.get("api_keys", [])
    api_key_map: Dict[str, str] = {}
    for item in api_keys_cfg:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        user = str(item.get("user", "")).strip() or "unknown"
        if key:
            api_key_map[key] = user

    _setup_logging()
    names, feats = build_feature_db_with_cache(
        standard_dir,
        standard_pattern,
        feature_backend,
        image_exts,
        w_clip,
        w_shape,
        w_color,
        w_stripe,
        standard_multicrop=standard_multicrop,
        standard_crop_ratio=standard_crop_ratio,
        use_cache=feature_cache_enabled,
        db_feature_dtype=db_feature_dtype,
    )
    logging.info("api preloaded db: %d items", len(names))
    rerank_candidate_cache: Dict[str, List[Dict[str, Any]]] = {}
    if rerank_enabled and preload_rerank_candidate_cache:
        t0 = time.perf_counter()
        rerank_candidate_cache = precompute_rerank_candidate_cache(
            standard_dir=standard_dir,
            names=names,
            candidate_views_max=rerank_candidate_views_max,
        )
        logging.info(
            "api preloaded rerank candidate cache: %d files in %.2fs",
            len(rerank_candidate_cache),
            time.perf_counter() - t0,
        )
    elif rerank_enabled:
        logging.info("api rerank candidate cache preload disabled; using lazy cache on requests")
    label_memory_refs = precompute_label_memory_refs(label_memory_path) if label_memory_enabled else []
    if label_memory_enabled:
        logging.info("api preloaded label memory refs: %d", len(label_memory_refs))

    app = FastAPI(title="search-similar-style-api", version="1.0.0")
    app.mount("/print-static", StaticFiles(directory=str(PRINT_STATIC_DIR)), name="print-static")
    app.mount("/print-storage", StaticFiles(directory=str(PRINT_STORAGE_DIR)), name="print-storage")
    app.mount("/recolor-static", StaticFiles(directory=str(RECOLOR_OUTPUT_DIR.parent)), name="recolor-static")

    app.state.ready = False
    app.state.ready_detail = "initializing"
    image_cache_dir = Path("outputs/image_cache")
    image_cache_dir.mkdir(parents=True, exist_ok=True)

    @app.middleware("http")
    async def check_api_key(request: Request, call_next):
        t0 = time.perf_counter()
        client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
            request.client.host if request.client else "-"
        )
        req_len = request.headers.get("content-length", "-")
        ua = request.headers.get("user-agent", "-")

        path = request.url.path
        allow_public = (
            path in {"/health", "/ready"}
            or path.startswith("/print-static/")
            or path.startswith("/print-storage/")
            or path.startswith("/recolor-static/")
        )
        allow_api = (
            path in {"/search", "/image-url", "/api/v1/templates", "/api/v1/render", "/api/v1/images/upload", "/recolor", "/recolor-ai"}
            or path.startswith("/images/")
            or path.startswith("/print-static/")
            or path.startswith("/print-storage/")
            or path.startswith("/recolor-static/")
        )
        if not (allow_public or allow_api):
            resp = JSONResponse(status_code=404, content={"detail": "not found"})
            logging.debug(
                'access ip=%s method=%s path=%s status=%s ms=%.1f len=%s ua="%s"',
                client_ip,
                request.method,
                request.url.path,
                resp.status_code,
                (time.perf_counter() - t0) * 1000.0,
                req_len,
                ua[:200],
            )
            return resp

        if allow_public:
            resp = await call_next(request)
            logging.info(
                'access ip=%s method=%s path=%s status=%s ms=%.1f len=%s ua="%s"',
                client_ip,
                request.method,
                request.url.path,
                resp.status_code,
                (time.perf_counter() - t0) * 1000.0,
                req_len,
                ua[:200],
            )
            return resp
        if api_key_enabled:
            key = request.headers.get("X-API-Key", "").strip()
            user = api_key_map.get(key, "")
            if path.startswith("/images/"):
                if user:
                    request.state.api_user = user
                    resp = await call_next(request)
                    logging.info(
                        'access ip=%s user=%s method=%s path=%s status=%s ms=%.1f len=%s ua="%s"',
                        client_ip,
                        getattr(request.state, "api_user", "unknown"),
                        request.method,
                        request.url.path,
                        resp.status_code,
                        (time.perf_counter() - t0) * 1000.0,
                        req_len,
                        ua[:200],
                    )
                    return resp

                exp = request.query_params.get("exp", "").strip()
                sig = request.query_params.get("sig", "").strip()
                image_name = Path(path.split("/images/", 1)[-1]).name
                if image_url_secret and exp.isdigit() and sig and image_name:
                    now_ts = int(time.time())
                    exp_ts = int(exp)
                    if exp_ts >= now_ts:
                        msg = f"{image_name}:{exp_ts}".encode("utf-8")
                        expected = hmac.new(
                            image_url_secret.encode("utf-8"),
                            msg,
                            hashlib.sha256,
                        ).hexdigest()
                        if hmac.compare_digest(expected, sig):
                            request.state.api_user = "signed-image-url"
                            resp = await call_next(request)
                            logging.info(
                                'access ip=%s user=%s method=%s path=%s status=%s ms=%.1f len=%s ua="%s"',
                                client_ip,
                                getattr(request.state, "api_user", "unknown"),
                                request.method,
                                request.url.path,
                                resp.status_code,
                                (time.perf_counter() - t0) * 1000.0,
                                req_len,
                                ua[:200],
                            )
                            return resp
            if not user:
                resp = JSONResponse(status_code=401, content={"detail": "invalid api key"})
                logging.info(
                    'access ip=%s method=%s path=%s status=%s ms=%.1f len=%s ua="%s"',
                    client_ip,
                    request.method,
                    request.url.path,
                    resp.status_code,
                    (time.perf_counter() - t0) * 1000.0,
                    req_len,
                    ua[:200],
                )
                return resp
            request.state.api_user = user
        else:
            request.state.api_user = "anonymous"
        resp = await call_next(request)
        logging.info(
            'access ip=%s user=%s method=%s path=%s status=%s ms=%.1f len=%s ua="%s"',
            client_ip,
            getattr(request.state, "api_user", "unknown"),
            request.method,
            request.url.path,
            resp.status_code,
            (time.perf_counter() - t0) * 1000.0,
            req_len,
            ua[:200],
        )
        return resp

    def _build_image_url(base_url: str, image_name: str) -> str:
        safe = Path(image_name).name
        if not safe:
            return f"{base_url}/images/"
        query_parts: List[str] = []
        if result_image_max_edge > 0:
            query_parts.append(f"max_edge={result_image_max_edge}")
            query_parts.append(f"q={max(40, min(95, result_image_quality))}")
        if api_key_enabled and image_url_secret:
            exp_ts = int(time.time()) + max(60, image_url_ttl_sec)
            msg = f"{safe}:{exp_ts}".encode("utf-8")
            sig = hmac.new(image_url_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
            query_parts.extend([f"exp={exp_ts}", f"sig={sig}"])
        qs = f"?{'&'.join(query_parts)}" if query_parts else ""
        return f"{base_url}/images/{safe}{qs}"

    def _build_image_url_with_exp(base_url: str, image_name: str) -> tuple[str, int]:
        safe = Path(image_name).name
        if not safe:
            return f"{base_url}/images/", 0
        query_parts: List[str] = []
        if result_image_max_edge > 0:
            query_parts.append(f"max_edge={result_image_max_edge}")
            query_parts.append(f"q={max(40, min(95, result_image_quality))}")
        if api_key_enabled and image_url_secret:
            exp_ts = int(time.time()) + max(60, image_url_ttl_sec)
            msg = f"{safe}:{exp_ts}".encode("utf-8")
            sig = hmac.new(image_url_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
            query_parts.extend([f"exp={exp_ts}", f"sig={sig}"])
            qs = f"?{'&'.join(query_parts)}" if query_parts else ""
            return f"{base_url}/images/{safe}{qs}", exp_ts
        qs = f"?{'&'.join(query_parts)}" if query_parts else ""
        return f"{base_url}/images/{safe}{qs}", 0

    def _external_base_url(request: Request) -> str:
        forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
        scheme = forwarded_proto if forwarded_proto in {"http", "https"} else request.url.scheme
        host = request.headers.get("host", "").strip() or request.url.netloc
        return f"{scheme}://{host}".rstrip("/")

    def _guess_mime(name: str) -> str:
        s = name.lower()
        if s.endswith(".png"):
            return "image/png"
        if s.endswith(".jpg") or s.endswith(".jpeg"):
            return "image/jpeg"
        return "application/octet-stream"

    def _image_b64(image_name: str) -> tuple[str, str]:
        safe = Path(image_name).name
        fp = standard_dir / safe
        if not fp.exists() or not fp.is_file():
            return "", ""
        raw = fp.read_bytes()
        return base64.b64encode(raw).decode("ascii"), _guess_mime(safe)

    def _cached_preview_path(image_name: str, max_edge: int, quality: int) -> Path:
        stem = Path(image_name).stem
        return image_cache_dir / f"{stem}__e{max_edge}_q{quality}.jpg"

    def _extract_fg_shape(path: Path) -> tuple[float, float] | None:
        try:
            with Image.open(path) as im0:
                rgb = np.asarray(im0.convert("RGB"), dtype=np.uint8)
        except Exception:
            return None
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return None
        fg = np.any(rgb < 240, axis=-1)
        cut = min(int(fg.shape[0] * 0.12), 120)
        fg[:cut, :] = False
        ys, xs = np.where(fg)
        if ys.size < 32:
            return None
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        hh = max(1, y1 - y0 + 1)
        ww = max(1, x1 - x0 + 1)
        aspect = float(ww) / float(hh)
        fill = float(fg[y0 : y1 + 1, x0 : x1 + 1].mean())
        return aspect, fill

    def _apply_shape_consistency(
        ranked: List[tuple[str, float]],
        query_shape: tuple[float, float] | None,
    ) -> List[tuple[str, float]]:
        if not ranked or query_shape is None:
            return ranked
        qa, qf = query_shape
        eps = 1e-6
        head_n = min(len(ranked), max(1, shape_consistency_apply_topn))
        head = ranked[:head_n]
        tail = ranked[head_n:]
        adjusted: List[tuple[str, float]] = []
        for name, score in head:
            file_name = name.split("@", 1)[0]
            cshape = fg_shape_cache.get(file_name)
            if cshape is None:
                adjusted.append((name, score))
                continue
            ca, cf = cshape
            d_aspect = min(1.0, abs(np.log((qa + eps) / (ca + eps))))
            d_fill = min(1.0, abs(qf - cf))
            penalty = shape_consistency_aspect_weight * d_aspect + shape_consistency_fill_weight * d_fill
            adjusted.append((name, float(score) - float(penalty)))
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted + tail

    def _ensure_preview(image_fp: Path, max_edge: int, quality: int) -> Path:
        quality = max(40, min(95, int(quality)))
        out_fp = _cached_preview_path(image_fp.name, max_edge, quality)
        if out_fp.exists() and out_fp.stat().st_mtime >= image_fp.stat().st_mtime:
            return out_fp
        with Image.open(image_fp) as im0:
            im = im0.convert("RGB")
            im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            out_fp.write_bytes(buf.getvalue())
        return out_fp

    fg_shape_cache: Dict[str, tuple[float, float]] = {}
    if shape_consistency_enabled:
        uniq = sorted({n.split("@", 1)[0] for n in names})
        for file_name in uniq:
            fp = standard_dir / file_name
            if not fp.exists() or not fp.is_file():
                continue
            shp = _extract_fg_shape(fp)
            if shp is not None:
                fg_shape_cache[file_name] = shp
        logging.info("api preloaded shape cache: %d", len(fg_shape_cache))

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        if bool(getattr(app.state, "ready", False)):
            return JSONResponse(status_code=200, content={"status": "ready"})
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": str(getattr(app.state, "ready_detail", "initializing"))},
        )

    @app.get("/images/{image_name}")
    def get_standard_image(image_name: str, max_edge: int = 0, q: int = 82) -> FileResponse:
        safe = Path(image_name).name
        fp = standard_dir / safe
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        if max_edge > 0:
            edge = max(128, min(2048, int(max_edge)))
            out_fp = _ensure_preview(fp, edge, q)
            return FileResponse(path=str(out_fp), media_type="image/jpeg")
        return FileResponse(path=str(fp))

    @app.get("/image-url", response_model=ImageUrlResponse)
    def refresh_image_url(request: Request, image_name: str) -> Dict[str, Any]:
        safe = Path(image_name).name
        fp = standard_dir / safe
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        base_url = _external_base_url(request)
        image_url, exp_ts = _build_image_url_with_exp(base_url, safe)
        return {"image_name": safe, "image_url": image_url, "expires_at": exp_ts}

    @app.post("/search", response_model=SearchResponse)
    async def search(
        request: Request,
        file: UploadFile = File(...),
        include_image_base64: bool = False,
        base64_topn: int = 0,
    ) -> Dict[str, Any]:
        t_all = time.perf_counter()
        if not file.filename:
            raise HTTPException(status_code=400, detail="missing file name")
        suffix = Path(file.filename).suffix.lower()
        if suffix.lstrip(".") not in {"png", "jpg", "jpeg"}:
            raise HTTPException(status_code=400, detail="only png/jpg/jpeg supported")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tf:
            tf.write(await file.read())
            tf.flush()
            query_path = Path(tf.name)

            query_hint_code = try_extract_query_style_code(query_path) if ocr_hint_enabled else ""
            code_prior_boost = (
                build_label_memory_prior_from_refs(
                    query_path,
                    label_memory_refs,
                    sim_threshold=label_memory_sim_threshold,
                    max_boost=label_memory_max_boost,
                )
                if label_memory_enabled
                else {}
            )

            def _run_search_pass(
                cand_multiplier: int,
                recall_cap: int,
                pass_query_component_views: bool,
                pass_rerank_topn: int,
                pass_rerank_query_views_max: int,
                pass_rerank_max_unique_codes: int,
                pass_query_view_consensus_weight: float,
                pass_w_clip: float,
                pass_w_shape: float,
                pass_w_color: float,
                pass_w_stripe: float,
            ) -> tuple[List[tuple[str, float]], List[Dict[str, Any]], float, float, float]:
                t0 = time.perf_counter()
                image_topk = min(len(names), max(top_k * max(cand_multiplier, 1), top_k))
                if recall_cap > 0:
                    image_topk = min(image_topk, recall_cap)
                ranked = search_topk_images(
                    query_path,
                    names,
                    feats,
                    image_topk,
                    feature_backend,
                    pass_w_clip,
                    pass_w_shape,
                    pass_w_color,
                    pass_w_stripe,
                    query_multicrop=query_multicrop,
                    query_crop_ratio=query_crop_ratio,
                    query_component_views=pass_query_component_views,
                    query_view_consensus_weight=pass_query_view_consensus_weight,
                )
                t_recall_local = time.perf_counter() - t0
                if rerank_enabled:
                    t1 = time.perf_counter()
                    ranked = rerank_candidates_with_model(
                        query_path,
                        ranked,
                        standard_dir=standard_dir,
                        reranker_model_path=reranker_model,
                        rerank_topn=pass_rerank_topn,
                        rerank_weight=rerank_weight,
                        query_multicrop=query_multicrop,
                        query_crop_ratio=query_crop_ratio,
                        query_component_views=pass_query_component_views,
                        rerank_query_views_max=pass_rerank_query_views_max,
                        rerank_candidate_views_max=rerank_candidate_views_max,
                        candidate_feature_cache=rerank_candidate_cache,
                        max_unique_codes=pass_rerank_max_unique_codes,
                    )
                    t_rerank_local = time.perf_counter() - t1
                else:
                    t_rerank_local = 0.0
                t2 = time.perf_counter()
                rows_local = topk_style_codes(
                    ranked,
                    top_k,
                    min_score=min_score,
                    code_agg_top_n=code_agg_top_n,
                    code_agg_alpha=code_agg_alpha,
                    query_hint_code=query_hint_code,
                    query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                    code_prior_boost=code_prior_boost,
                )
                t_post_local = time.perf_counter() - t2
                return ranked, rows_local, t_recall_local, t_rerank_local, t_post_local

            q_shape = _extract_fg_shape(query_path)
            use_strip_mode = False
            if strip_mode_enabled and q_shape is not None:
                qa, qf = q_shape
                use_strip_mode = (qa >= strip_aspect_threshold) or (qf <= strip_fill_threshold)
            if use_strip_mode:
                w_clip_pass = strip_w_clip
                w_shape_pass = strip_w_shape
                w_color_pass = strip_w_color
                w_stripe_pass = strip_w_stripe
            else:
                w_clip_pass = w_clip
                w_shape_pass = w_shape
                w_color_pass = w_color
                w_stripe_pass = w_stripe

            ranked_images, rows, t_recall, t_rerank, t_post = _run_search_pass(
                cand_multiplier=candidate_multiplier,
                recall_cap=recall_topn_cap,
                pass_query_component_views=query_component_views,
                pass_rerank_topn=rerank_topn,
                pass_rerank_query_views_max=rerank_query_views_max,
                pass_rerank_max_unique_codes=rerank_max_unique_codes,
                pass_query_view_consensus_weight=query_view_consensus_weight,
                pass_w_clip=w_clip_pass,
                pass_w_shape=w_shape_pass,
                pass_w_color=w_color_pass,
                pass_w_stripe=w_stripe_pass,
            )
            second_pass_used = False
            if adaptive_second_pass_enabled:
                top1_score = float(rows[0]["score"]) if rows else 0.0
                top2_score = float(rows[1]["score"]) if len(rows) > 1 else 0.0
                margin = top1_score - top2_score
                if (top1_score < adaptive_trigger_top1_below) or (len(rows) > 1 and margin < adaptive_trigger_margin_below):
                    ranked_images, rows, t2_recall, t2_rerank, t2_post = _run_search_pass(
                        cand_multiplier=adaptive_candidate_multiplier,
                        recall_cap=adaptive_recall_topn_cap,
                        pass_query_component_views=adaptive_query_component_views,
                        pass_rerank_topn=adaptive_rerank_topn,
                        pass_rerank_query_views_max=adaptive_rerank_query_views_max,
                        pass_rerank_max_unique_codes=adaptive_rerank_max_unique_codes,
                        pass_query_view_consensus_weight=adaptive_query_view_consensus_weight,
                        pass_w_clip=w_clip_pass,
                        pass_w_shape=w_shape_pass,
                        pass_w_color=w_color_pass,
                        pass_w_stripe=w_stripe_pass,
                    )
                    t_recall += t2_recall
                    t_rerank += t2_rerank
                    t_post += t2_post
                    second_pass_used = True
            if shape_consistency_enabled and q_shape is not None:
                    ranked_images = _apply_shape_consistency(ranked_images, q_shape)
                    rows = topk_style_codes(
                        ranked_images,
                        top_k,
                        min_score=min_score,
                        code_agg_top_n=code_agg_top_n,
                        code_agg_alpha=code_agg_alpha,
                        query_hint_code=query_hint_code,
                        query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                        code_prior_boost=code_prior_boost,
                    )

        base_url = _external_base_url(request)
        for row in rows:
            img = str(row.get("best_standard_image", "")).strip()
            row["best_standard_image_url"] = _build_image_url(base_url, img)

        if include_image_base64:
            n = len(rows) if base64_topn <= 0 else min(len(rows), base64_topn)
            for i in range(n):
                img = str(rows[i].get("best_standard_image", "")).strip()
                b64, mime = _image_b64(img)
                rows[i]["best_standard_image_base64"] = b64
                rows[i]["best_standard_image_mime"] = mime

        logging.info(
            "search timing user=%s file=%s recall=%.3fs rerank=%.3fs post=%.3fs second_pass=%s strip_mode=%s total=%.3fs",
            getattr(request.state, "api_user", "unknown"),
            file.filename,
            t_recall,
            t_rerank,
            t_post,
            second_pass_used,
            use_strip_mode,
            time.perf_counter() - t_all,
        )

        return {
            "query_image": file.filename,
            "topk_style_codes": rows,
            "api_user": getattr(request.state, "api_user", "unknown"),
        }

    @app.get("/api/v1/templates")
    def api_list_templates() -> List[Dict[str, Any]]:
        return list_templates()

    @app.post("/api/v1/images/upload")
    async def api_upload_image(file: UploadFile = File(...)) -> Dict[str, Any]:
        suffix = Path(file.filename or "").suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise HTTPException(status_code=400, detail="仅支持 jpg/jpeg/png/webp")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="空文件")

        try:
            return process_upload(content, suffix=suffix)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"处理图片失败: {exc}") from exc

    @app.post("/api/v1/render")
    def api_render_layout(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            return render_layout(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/recolor")
    async def api_recolor(
        file: UploadFile = File(...),
        target_hex: str = Form("FF5500"),
        x_ratio: float = Form(0.2),
        y_ratio: float = Form(0.2),
        w_ratio: float = Form(0.4),
        h_ratio: float = Form(0.4),
        strength: float = Form(0.8),
        feather_ratio: float = Form(0.02),
        auto_mask: int = Form(0),
    ) -> Dict[str, Any]:
        suffix = Path(file.filename or "").suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise HTTPException(status_code=400, detail="仅支持 jpg/jpeg/png/webp")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="空文件")
        try:
            return recolor_region(
                file_bytes=content,
                suffix=suffix,
                target_hex=target_hex,
                x_ratio=x_ratio,
                y_ratio=y_ratio,
                w_ratio=w_ratio,
                h_ratio=h_ratio,
                strength=strength,
                feather_ratio=feather_ratio,
                auto_mask=bool(int(auto_mask)),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/recolor-ai")
    async def api_recolor_ai(
        file: UploadFile = File(...),
        model: str = Form("Qwen/Qwen-Image-Edit-2509"),
        target_hex: str = Form("FF5500"),
        x_ratio: float = Form(0.2),
        y_ratio: float = Form(0.2),
        w_ratio: float = Form(0.4),
        h_ratio: float = Form(0.4),
        strength: float = Form(0.7),
        prompt: str = Form(""),
        negative_prompt: str = Form(""),
        seed: int | None = Form(None),
        cfg_scale: float | None = Form(None),
        num_inference_steps: int | None = Form(None),
        postprocess: int = Form(1),
        image2: str | None = Form(None),
        image3: str | None = Form(None),
    ) -> Dict[str, Any]:
        suffix = Path(file.filename or "").suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise HTTPException(status_code=400, detail="仅支持 jpg/jpeg/png/webp")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="空文件")
        try:
            return recolor_region_ai(
                file_bytes=content,
                suffix=suffix,
                api_key=os.getenv("SILICONFLOW_API_KEY", "").strip(),
                model=model,
                target_hex=target_hex,
                x_ratio=x_ratio,
                y_ratio=y_ratio,
                w_ratio=w_ratio,
                h_ratio=h_ratio,
                strength=strength,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                postprocess=bool(int(postprocess)),
                image2=image2,
                image3=image3,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.state.ready = True
    app.state.ready_detail = "ready"
    return app


app = create_app(Path(os.getenv("SEARCH_CONFIG", str(DEFAULT_CONFIG))))
