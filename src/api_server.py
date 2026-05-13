import json
import logging
import sys
import tempfile
import base64
import time
import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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


class SearchResponse(BaseModel):
    query_image: str
    topk_style_codes: List[Dict[str, Any]]


def _load_cfg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
    auth_cfg = cfg.get("auth", {})
    api_key_enabled = bool(auth_cfg.get("enabled", True))
    api_keys_cfg = auth_cfg.get("api_keys", [])
    api_key_map: Dict[str, str] = {}
    for item in api_keys_cfg:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        user = str(item.get("user", "")).strip() or "unknown"
        if key:
            api_key_map[key] = user

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
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
    )
    logging.info("api preloaded db: %d items", len(names))
    rerank_candidate_cache: Dict[str, List[Dict[str, Any]]] = {}
    if rerank_enabled:
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
    label_memory_refs = precompute_label_memory_refs(label_memory_path) if label_memory_enabled else []
    if label_memory_enabled:
        logging.info("api preloaded label memory refs: %d", len(label_memory_refs))

    app = FastAPI(title="search-similar-style-api", version="1.0.0")

    app.state.ready = False
    app.state.ready_detail = "initializing"

    @app.middleware("http")
    async def check_api_key(request: Request, call_next):
        if request.url.path in {"/health", "/ready"}:
            return await call_next(request)
        if api_key_enabled:
            key = request.headers.get("X-API-Key", "").strip()
            user = api_key_map.get(key, "")
            if not user:
                return JSONResponse(status_code=401, content={"detail": "invalid api key"})
            request.state.api_user = user
        else:
            request.state.api_user = "anonymous"
        return await call_next(request)

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
    def get_standard_image(image_name: str) -> FileResponse:
        safe = Path(image_name).name
        fp = standard_dir / safe
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(path=str(fp))

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

            t0 = time.perf_counter()
            image_topk = min(len(names), max(top_k * max(candidate_multiplier, 1), top_k))
            ranked_images = search_topk_images(
                query_path,
                names,
                feats,
                image_topk,
                feature_backend,
                w_clip,
                w_shape,
                w_color,
                w_stripe,
                query_multicrop=query_multicrop,
                query_crop_ratio=query_crop_ratio,
                query_component_views=query_component_views,
            )
            t_recall = time.perf_counter() - t0
            if rerank_enabled:
                t1 = time.perf_counter()
                ranked_images = rerank_candidates_with_model(
                    query_path,
                    ranked_images,
                    standard_dir=standard_dir,
                    reranker_model_path=reranker_model,
                    rerank_topn=rerank_topn,
                    rerank_weight=rerank_weight,
                    query_multicrop=query_multicrop,
                    query_crop_ratio=query_crop_ratio,
                    query_component_views=query_component_views,
                    rerank_query_views_max=rerank_query_views_max,
                    rerank_candidate_views_max=rerank_candidate_views_max,
                    candidate_feature_cache=rerank_candidate_cache,
                )
                t_rerank = time.perf_counter() - t1
            else:
                t_rerank = 0.0

            t2 = time.perf_counter()
            rows = topk_style_codes(
                ranked_images,
                top_k,
                min_score=min_score,
                code_agg_top_n=code_agg_top_n,
                code_agg_alpha=code_agg_alpha,
                query_hint_code=try_extract_query_style_code(query_path) if ocr_hint_enabled else "",
                query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                code_prior_boost=build_label_memory_prior_from_refs(
                    query_path,
                    label_memory_refs,
                    sim_threshold=label_memory_sim_threshold,
                    max_boost=label_memory_max_boost,
                )
                if label_memory_enabled
                else {},
            )
            t_post = time.perf_counter() - t2

        base_url = str(request.base_url).rstrip("/")
        for row in rows:
            img = str(row.get("best_standard_image", "")).strip()
            row["best_standard_image_url"] = f"{base_url}/images/{img}"

        if include_image_base64:
            n = len(rows) if base64_topn <= 0 else min(len(rows), base64_topn)
            for i in range(n):
                img = str(rows[i].get("best_standard_image", "")).strip()
                b64, mime = _image_b64(img)
                rows[i]["best_standard_image_base64"] = b64
                rows[i]["best_standard_image_mime"] = mime

        logging.info(
            "search timing user=%s file=%s recall=%.3fs rerank=%.3fs post=%.3fs total=%.3fs",
            getattr(request.state, "api_user", "unknown"),
            file.filename,
            t_recall,
            t_rerank,
            t_post,
            time.perf_counter() - t_all,
        )

        return {
            "query_image": file.filename,
            "topk_style_codes": rows,
            "api_user": getattr(request.state, "api_user", "unknown"),
        }

    app.state.ready = True
    app.state.ready_detail = "ready"
    return app


app = create_app(Path(os.getenv("SEARCH_CONFIG", str(DEFAULT_CONFIG))))
