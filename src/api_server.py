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
import math
import datetime as dt
import difflib
import re
import shutil
import threading
import uuid
import urllib.error
import urllib.parse
import urllib.request
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageOps
from zoneinfo import ZoneInfo

try:
    import cv2
except Exception:
    cv2 = None

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from search_similar_return_code import (
    DEFAULT_CONFIG,
    build_feature_db_with_cache,
    build_label_memory_prior_from_refs,
    collect_images,
    extract_embedding,
    filename_to_style_code,
    merge_scene_text_candidates,
    merge_ranked_image_lists,
    precompute_label_memory_refs,
    precompute_rerank_candidate_cache,
    precompute_scene_text_index,
    rerank_candidates_with_model,
    search_topk_images,
    topk_style_codes,
    try_extract_query_style_code,
)
from features import extract_garment_color_feature
from recolor_service import RECOLOR_OUTPUT_DIR, recolor_region, recolor_region_ai
from catalog_store import CatalogStore, derive_year_from_style_code, make_typed_tag, parse_catalog_tag
from color_card_store import ColorCardStore
from color_card_importer import read_color_rows, slugify_library_id
from extract_style_codes import build_header_crops, code_to_filename_prefix, try_extract_code_from_image


def _style_code_key(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "", str(code).strip().upper())


def _expand_region_crop(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    min_w: float,
    min_h: float,
    min_area: float,
    context_pad_ratio: float,
    context_min_area: float,
    wide_strip_aspect_threshold: float,
    wide_strip_max_h: float,
) -> tuple[float, float, float, float, bool]:
    x = max(0.0, min(0.98, float(x)))
    y = max(0.0, min(0.98, float(y)))
    w = max(0.02, min(1.0 - x, float(w)))
    h = max(0.02, min(1.0 - y, float(h)))

    cx = x + w * 0.5
    cy = y + h * 0.5
    target_w = max(w, min_w)
    target_h = max(h, min_h)
    needs_context = w < min_w or h < min_h or (w * h) < max(min_area, context_min_area)
    orig_aspect = float(w) / max(1e-6, float(h))

    pad = max(0.0, float(context_pad_ratio))
    if needs_context and pad > 0.0:
        target_w = max(target_w, w * (1.0 + pad * 2.0))
        target_h = max(target_h, h * (1.0 + pad * 2.0))

    target_area = max(min_area, context_min_area if needs_context else 0.0)
    if target_w * target_h < target_area:
        scale = math.sqrt(target_area / max(1e-6, target_w * target_h))
        target_w *= scale
        target_h *= scale

    target_w = max(0.02, min(1.0, target_w))
    target_h = max(0.02, min(1.0, target_h))
    if orig_aspect >= max(1.0, float(wide_strip_aspect_threshold)):
        target_h = min(target_h, max(h, min(1.0, float(wide_strip_max_h))))
    next_x = max(0.0, min(1.0 - target_w, cx - target_w * 0.5))
    next_y = max(0.0, min(1.0 - target_h, cy - target_h * 0.5))
    expanded = (
        abs(next_x - x) > 1e-6
        or abs(next_y - y) > 1e-6
        or abs(target_w - w) > 1e-6
        or abs(target_h - h) > 1e-6
    )
    return next_x, next_y, target_w, target_h, expanded

try:
    from print_service import PRINT_STATIC_DIR, PRINT_STORAGE_DIR, list_templates, process_upload, render_layout
    PRINT_SERVICE_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    PRINT_SERVICE_IMPORT_ERROR = exc
    PRINT_STATIC_DIR = (THIS_DIR / "print_runtime" / "static")
    PRINT_STORAGE_DIR = (THIS_DIR / "print_runtime" / "storage")
    PRINT_STATIC_DIR.mkdir(parents=True, exist_ok=True)
    PRINT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    def _raise_print_service_unavailable() -> None:
        raise RuntimeError(
            "print service unavailable; install optional dependency 'reportlab' first"
        ) from PRINT_SERVICE_IMPORT_ERROR

    def list_templates() -> List[Dict[str, Any]]:
        _raise_print_service_unavailable()

    def process_upload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _raise_print_service_unavailable()

    def render_layout(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _raise_print_service_unavailable()


class SearchResponse(BaseModel):
    query_image: str
    topk_style_codes: List[Dict[str, Any]]
    similar_images: List[Dict[str, Any]] = []
    is_ambiguous: bool = False
    confidence_band: str = "low"


class ImageUrlResponse(BaseModel):
    image_name: str
    image_url: str
    expires_at: int


class CatalogTagUpdateRequest(BaseModel):
    tags: List[str]


class CatalogTagCreateRequest(BaseModel):
    name: str
    type: str = ""


class CatalogPersonalProductRequest(BaseModel):
    source_style_code: str
    image_names: List[str]
    folder_name: str
    user_tag: str = ""


class CatalogImportPrepareRequest(BaseModel):
    source_dir: str


class CatalogImportWechatUploadFile(BaseModel):
    filename: str
    data_base64: str


class CatalogImportWechatUploadRequest(BaseModel):
    files: List[CatalogImportWechatUploadFile]


class WechatSessionRequest(BaseModel):
    code: str


class WechatContentSecurityError(RuntimeError):
    pass


class CatalogImportCommitItem(BaseModel):
    source_rel_path: str
    target_filename: str = ""
    year_tag: str = ""
    tags: List[str] = []
    selected: bool = True


class CatalogImportCommitRequest(BaseModel):
    job_id: str
    items: List[CatalogImportCommitItem]


class ColorCardMatchRequest(BaseModel):
    L: float
    a: float
    b: float
    library_id: str = ""
    limit: int = 12


class ColorCardLibraryUpsertRequest(BaseModel):
    id: str = ""
    name: str


class ColorCardUpsertRequest(BaseModel):
    library_id: str
    library_name: str = ""
    name: str
    note: str = ""
    illuminant: str = "D65"
    angle: float | None = 10
    L: float
    a: float
    b: float
    spectral: List[Any] = []


class ColorCardFavoriteRequest(BaseModel):
    user_tag: str
    card_id: int


class ColorCardPickedFavoriteRequest(BaseModel):
    user_tag: str
    name: str = ""
    hex: str = ""
    L: float
    a: float
    b: float


class ColorMeterNativeReadingRequest(BaseModel):
    user_tag: str
    request_id: str = ""
    L: float | None = None
    a: float | None = None
    b: float | None = None
    sample_count: int = 1
    progress_count: int = 0
    total_count: int = 0
    device_id: str = ""
    device_name: str = ""
    event: str = "measure"
    ts: float | None = None


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
    content_security_cfg = cfg.get("content_security", {})
    wechat_security_cfg = content_security_cfg.get("wechat", {})
    wechat_security_env_enabled = os.getenv("WECHAT_CONTENT_SECURITY_ENABLED", "").strip().lower()
    wechat_content_security_enabled = bool(wechat_security_cfg.get("enabled", False))
    if wechat_security_env_enabled in {"1", "true", "yes"}:
        wechat_content_security_enabled = True
    elif wechat_security_env_enabled in {"0", "false", "no"}:
        wechat_content_security_enabled = False
    wechat_appid = str(os.getenv("WECHAT_APPID", "") or wechat_security_cfg.get("appid", "")).strip()
    wechat_appsecret = str(os.getenv("WECHAT_APPSECRET", "") or wechat_security_cfg.get("appsecret", "")).strip()
    wechat_security_openid = str(os.getenv("WECHAT_SECURITY_OPENID", "") or wechat_security_cfg.get("openid", "")).strip()
    wechat_security_scene = int(wechat_security_cfg.get("scene", 2))
    wechat_security_fail_open = bool(wechat_security_cfg.get("fail_open", False))
    wechat_security_timeout = float(wechat_security_cfg.get("timeout_sec", 10))
    ai_generation_cfg = cfg.get("ai_generation", {})
    ai_generation_model = str(ai_generation_cfg.get("model", "doubao-seedream-5-0-260128")).strip() or "doubao-seedream-5-0-260128"
    ai_generation_size = str(ai_generation_cfg.get("size", "2K")).strip() or "2K"
    ai_generation_output_format = str(ai_generation_cfg.get("output_format", "png")).strip() or "png"
    ai_generation_sequential = str(ai_generation_cfg.get("sequential_image_generation", "disabled")).strip() or "disabled"
    ai_generation_watermark = bool(ai_generation_cfg.get("watermark", False))
    ai_generation_seed_raw = ai_generation_cfg.get("seed", None)
    ai_generation_seed: int | None = None
    if ai_generation_seed_raw is not None and str(ai_generation_seed_raw).strip() != "":
        try:
            ai_generation_seed = int(ai_generation_seed_raw)
        except (TypeError, ValueError):
            raise ValueError("ai_generation.seed must be an integer or null")

    standard_dir = Path(path_cfg.get("standard_dir", "data/standard_samples"))
    standard_pattern = str(path_cfg.get("standard_pattern", "*"))
    image_exts = list(path_cfg.get("image_exts", ["png", "jpg", "jpeg"]))
    search_feature_dir = Path("outputs/search_feature_samples")

    top_k = int(search_cfg.get("top_k", 5))
    candidate_multiplier = int(search_cfg.get("candidate_multiplier", 10))
    feature_backend = str(search_cfg.get("feature_backend", "hybrid"))
    min_score = float(search_cfg.get("min_score", 0.64))
    code_agg_top_n = int(search_cfg.get("code_agg_top_n", 3))
    code_agg_alpha = float(search_cfg.get("code_agg_alpha", 0.7))
    query_multicrop = bool(search_cfg.get("query_multicrop", True))
    query_crop_ratio = float(search_cfg.get("query_crop_ratio", 0.72))
    query_component_views = bool(search_cfg.get("query_component_views", False))
    query_max_edge = int(search_cfg.get("query_max_edge", 0))
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
    secondary_feature_backend = str(search_cfg.get("secondary_feature_backend", "")).strip().lower()
    secondary_hybrid_weights = search_cfg.get("secondary_hybrid_weights", {})
    secondary_w_clip = float(secondary_hybrid_weights.get("clip", w_clip))
    secondary_w_shape = float(secondary_hybrid_weights.get("shape", w_shape))
    secondary_w_color = float(secondary_hybrid_weights.get("color", w_color))
    secondary_w_stripe = float(secondary_hybrid_weights.get("stripe", w_stripe))
    secondary_recall_weight = float(search_cfg.get("secondary_recall_weight", 0.92))
    feature_cache_enabled = bool(search_cfg.get("feature_cache_enabled", True))
    db_feature_dtype = str(search_cfg.get("db_feature_dtype", "float32")).lower()
    recall_topn_cap = int(search_cfg.get("recall_topn_cap", 0))
    preload_rerank_candidate_cache = bool(search_cfg.get("preload_rerank_candidate_cache", False))
    rerank_max_unique_codes = int(search_cfg.get("rerank_max_unique_codes", 0))
    result_image_max_edge = int(search_cfg.get("result_image_max_edge", 0))
    result_image_quality = int(search_cfg.get("result_image_quality", 82))
    default_match_mode = str(search_cfg.get("match_mode", "similar_style")).strip().lower()
    region_primary_when_crop = bool(search_cfg.get("region_primary_when_crop", True))
    region_crop_auto_expand_enabled = bool(search_cfg.get("region_crop_auto_expand_enabled", True))
    region_crop_auto_expand_min_w = float(search_cfg.get("region_crop_auto_expand_min_w", 0.34))
    region_crop_auto_expand_min_h = float(search_cfg.get("region_crop_auto_expand_min_h", 0.34))
    region_crop_auto_expand_min_area = float(search_cfg.get("region_crop_auto_expand_min_area", 0.12))
    region_crop_context_pad_ratio = float(search_cfg.get("region_crop_context_pad_ratio", 0.35))
    region_crop_context_min_area = float(search_cfg.get("region_crop_context_min_area", 0.18))
    region_crop_strict_small_enabled = bool(search_cfg.get("region_crop_strict_small_enabled", True))
    region_crop_strict_small_max_orig_area = float(search_cfg.get("region_crop_strict_small_max_orig_area", 0.10))
    region_crop_strict_small_min_expand_ratio = float(search_cfg.get("region_crop_strict_small_min_expand_ratio", 1.6))
    region_crop_wide_strip_aspect_threshold = float(search_cfg.get("region_crop_wide_strip_aspect_threshold", 1.8))
    region_crop_wide_strip_max_h = float(search_cfg.get("region_crop_wide_strip_max_h", 0.42))
    region_crop_disable_accent_when_strip = bool(search_cfg.get("region_crop_disable_accent_when_strip", True))
    region_strip_query_multicrop_enabled = bool(search_cfg.get("region_strip_query_multicrop_enabled", True))
    region_strip_query_crop_ratio = float(search_cfg.get("region_strip_query_crop_ratio", 0.62))
    region_strip_query_component_views = bool(search_cfg.get("region_strip_query_component_views", False))
    region_strip_query_view_consensus_weight = float(search_cfg.get("region_strip_query_view_consensus_weight", 0.0))
    strict_small_query_weights = search_cfg.get("strict_small_query_weights", {})
    strict_small_w_clip = float(strict_small_query_weights.get("clip", 0.52))
    strict_small_w_shape = float(strict_small_query_weights.get("shape", 0.30))
    strict_small_w_color = float(strict_small_query_weights.get("color", 0.06))
    strict_small_w_stripe = float(strict_small_query_weights.get("stripe", 0.12))
    strict_small_region_code_prior_min_score = float(search_cfg.get("strict_small_region_code_prior_min_score", 0.62))
    strict_small_region_result_rescue_min_score = float(search_cfg.get("strict_small_region_result_rescue_min_score", 0.60))
    strict_small_region_force_top_min_score = float(search_cfg.get("strict_small_region_force_top_min_score", 0.62))
    strict_small_region_force_topn = int(search_cfg.get("strict_small_region_force_topn", 3))
    strict_small_region_recall_topn_cap = int(search_cfg.get("strict_small_region_recall_topn_cap", 240))
    strict_small_region_query_multicrop_enabled = bool(search_cfg.get("strict_small_region_query_multicrop_enabled", True))
    strict_small_region_query_crop_ratio = float(search_cfg.get("strict_small_region_query_crop_ratio", 0.72))
    strict_small_region_query_component_views = bool(search_cfg.get("strict_small_region_query_component_views", True))
    strict_small_disable_consistency = bool(search_cfg.get("strict_small_disable_consistency", True))
    strict_small_pattern_region_order_enabled = bool(search_cfg.get("strict_small_pattern_region_order_enabled", True))
    strict_small_pattern_region_order_weight = float(search_cfg.get("strict_small_pattern_region_order_weight", 0.07))
    strict_small_pattern_region_order_min_sim = float(search_cfg.get("strict_small_pattern_region_order_min_sim", 0.58))
    strict_small_pattern_region_rescue_enabled = bool(search_cfg.get("strict_small_pattern_region_rescue_enabled", True))
    strict_small_pattern_region_rescue_scan_images = int(search_cfg.get("strict_small_pattern_region_rescue_scan_images", 180))
    strict_small_pattern_region_rescue_max_rows = int(search_cfg.get("strict_small_pattern_region_rescue_max_rows", 6))
    strict_small_pattern_region_rescue_min_sim = float(search_cfg.get("strict_small_pattern_region_rescue_min_sim", 0.56))
    strict_small_pattern_region_rescue_near_delta = float(search_cfg.get("strict_small_pattern_region_rescue_near_delta", 0.05))
    strict_small_dark_motif_rescue_enabled = bool(search_cfg.get("strict_small_dark_motif_rescue_enabled", True))
    strict_small_dark_motif_rescue_min_sim = float(search_cfg.get("strict_small_dark_motif_rescue_min_sim", 0.58))
    strict_small_dark_motif_full_presence_min_sim = float(search_cfg.get("strict_small_dark_motif_full_presence_min_sim", 0.46))
    strict_small_dark_motif_full_presence_boost = float(search_cfg.get("strict_small_dark_motif_full_presence_boost", 0.10))
    strict_small_dark_motif_rescue_weight = float(search_cfg.get("strict_small_dark_motif_rescue_weight", 0.14))
    strict_small_dark_motif_rescue_max_rows = int(search_cfg.get("strict_small_dark_motif_rescue_max_rows", 9))
    exact_region_code_prior_scale = float(search_cfg.get("exact_region_code_prior_scale", 0.45))
    exact_region_rescue_enabled = bool(search_cfg.get("exact_region_rescue_enabled", False))
    region_crop_recall_enabled = bool(search_cfg.get("region_crop_recall_enabled", True))
    region_crop_recall_backend = str(search_cfg.get("region_crop_recall_backend", secondary_feature_backend or feature_backend)).strip().lower()
    region_crop_recall_weight = float(search_cfg.get("region_crop_recall_weight", 1.12))
    region_crop_recall_topn_cap = int(search_cfg.get("region_crop_recall_topn_cap", 1200))
    full_context_region_probe_enabled = bool(search_cfg.get("full_context_region_probe_enabled", True))
    full_context_region_probe_max_aspect = float(search_cfg.get("full_context_region_probe_max_aspect", 0.72))
    full_context_region_probe_min_height = int(search_cfg.get("full_context_region_probe_min_height", 640))
    region_crop_suppress_accessory_enabled = bool(search_cfg.get("region_crop_suppress_accessory_enabled", True))
    region_crop_suppress_accessory_min_score = float(search_cfg.get("region_crop_suppress_accessory_min_score", 0.68))
    region_crop_suppress_accessory_wide_min_score = float(search_cfg.get("region_crop_suppress_accessory_wide_min_score", 0.74))
    region_crop_suppress_accessory_topn = int(search_cfg.get("region_crop_suppress_accessory_topn", 10))
    region_crop_suppress_accessory_min_hits = int(search_cfg.get("region_crop_suppress_accessory_min_hits", 3))
    region_crop_code_prior_enabled = bool(search_cfg.get("region_crop_code_prior_enabled", True))
    region_crop_code_prior_min_score = float(search_cfg.get("region_crop_code_prior_min_score", 0.74))
    region_crop_code_prior_boost = float(search_cfg.get("region_crop_code_prior_boost", 0.08))
    region_crop_code_prior_topn = int(search_cfg.get("region_crop_code_prior_topn", 8))
    region_crop_result_rescue_enabled = bool(search_cfg.get("region_crop_result_rescue_enabled", True))
    region_crop_result_rescue_min_score = float(search_cfg.get("region_crop_result_rescue_min_score", 0.74))
    region_crop_result_rescue_topn = int(search_cfg.get("region_crop_result_rescue_topn", 8))
    region_crop_result_rescue_scan_codes = int(search_cfg.get("region_crop_result_rescue_scan_codes", 80))
    region_crop_large_result_rescue_min_score = float(search_cfg.get("region_crop_large_result_rescue_min_score", 0.64))
    region_crop_large_result_rescue_top_delta = float(search_cfg.get("region_crop_large_result_rescue_top_delta", 0.055))
    region_crop_large_result_rescue_topn = int(search_cfg.get("region_crop_large_result_rescue_topn", 16))
    region_crop_large_result_rescue_order_max_best = float(
        search_cfg.get("region_crop_large_result_rescue_order_max_best", 0.72)
    )
    region_crop_force_top_enabled = bool(search_cfg.get("region_crop_force_top_enabled", True))
    region_crop_force_top_min_score = float(search_cfg.get("region_crop_force_top_min_score", 0.80))
    region_crop_force_topn = int(search_cfg.get("region_crop_force_topn", 1))
    region_crop_large_force_top_area = float(search_cfg.get("region_crop_large_force_top_area", 0.30))
    region_crop_large_force_top_min_score = float(search_cfg.get("region_crop_large_force_top_min_score", 0.66))
    region_crop_large_force_topn = int(search_cfg.get("region_crop_large_force_topn", 3))
    region_crop_repeat_force_enabled = bool(search_cfg.get("region_crop_repeat_force_enabled", True))
    region_crop_repeat_force_topn = int(search_cfg.get("region_crop_repeat_force_topn", 12))
    region_crop_repeat_force_min_score = float(search_cfg.get("region_crop_repeat_force_min_score", 0.62))
    region_crop_repeat_force_min_hits = int(search_cfg.get("region_crop_repeat_force_min_hits", 4))
    region_crop_repeat_force_seed_score = float(search_cfg.get("region_crop_repeat_force_seed_score", 1.50))
    region_crop_dominant_repeat_min_hits = int(search_cfg.get("region_crop_dominant_repeat_min_hits", 8))
    region_crop_dominant_repeat_min_score = float(search_cfg.get("region_crop_dominant_repeat_min_score", 0.64))
    region_crop_sleeve_rescue_enabled = bool(search_cfg.get("region_crop_sleeve_rescue_enabled", True))
    region_crop_sleeve_rescue_min_sim = float(search_cfg.get("region_crop_sleeve_rescue_min_sim", 0.70))
    region_crop_sleeve_rescue_min_pair_prior = float(search_cfg.get("region_crop_sleeve_rescue_min_pair_prior", 0.70))
    region_crop_sleeve_rescue_strong_sim = float(search_cfg.get("region_crop_sleeve_rescue_strong_sim", 0.82))
    region_crop_sleeve_rescue_strong_min_pair_prior = float(
        search_cfg.get("region_crop_sleeve_rescue_strong_min_pair_prior", 0.50)
    )
    region_crop_sleeve_rescue_weight = float(search_cfg.get("region_crop_sleeve_rescue_weight", 0.18))
    region_crop_color_consistency_enabled = bool(search_cfg.get("region_crop_color_consistency_enabled", True))
    region_crop_color_consistency_weight = float(search_cfg.get("region_crop_color_consistency_weight", 0.14))
    region_crop_color_consistency_apply_topn = int(search_cfg.get("region_crop_color_consistency_apply_topn", 256))
    region_crop_order_by_region_enabled = bool(search_cfg.get("region_crop_order_by_region_enabled", True))
    region_standard_crop_ratio = float(search_cfg.get("region_standard_crop_ratio", 0.55))
    region_hybrid_weights = search_cfg.get("region_hybrid_weights", secondary_hybrid_weights or hybrid_weights)
    region_w_clip = float(region_hybrid_weights.get("clip", secondary_w_clip))
    region_w_shape = float(region_hybrid_weights.get("shape", secondary_w_shape))
    region_w_color = float(region_hybrid_weights.get("color", secondary_w_color))
    region_w_stripe = float(region_hybrid_weights.get("stripe", secondary_w_stripe))
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
    mask_consistency_enabled = bool(search_cfg.get("mask_consistency_enabled", True))
    mask_consistency_weight = float(search_cfg.get("mask_consistency_weight", 0.10))
    mask_consistency_apply_topn = int(search_cfg.get("mask_consistency_apply_topn", 256))
    stripe_consistency_enabled = bool(search_cfg.get("stripe_consistency_enabled", True))
    stripe_consistency_weight = float(search_cfg.get("stripe_consistency_weight", 0.12))
    stripe_consistency_apply_topn = int(search_cfg.get("stripe_consistency_apply_topn", 256))
    pattern_consistency_enabled = bool(search_cfg.get("pattern_consistency_enabled", False))
    pattern_consistency_weight = float(search_cfg.get("pattern_consistency_weight", 0.16))
    pattern_consistency_apply_topn = int(search_cfg.get("pattern_consistency_apply_topn", 256))
    pattern_code_boost_enabled = bool(search_cfg.get("pattern_code_boost_enabled", False))
    pattern_code_boost_weight = float(search_cfg.get("pattern_code_boost_weight", 0.08))
    pattern_code_boost_topn = int(search_cfg.get("pattern_code_boost_topn", 24))
    checker_consistency_enabled = bool(search_cfg.get("checker_consistency_enabled", False))
    checker_query_threshold = float(search_cfg.get("checker_query_threshold", 0.12))
    checker_boost_weight = float(search_cfg.get("checker_boost_weight", 0.28))
    checker_stripe_penalty_weight = float(search_cfg.get("checker_stripe_penalty_weight", 0.18))
    checker_apply_topn = int(search_cfg.get("checker_apply_topn", 160))
    checker_code_boost_weight = float(search_cfg.get("checker_code_boost_weight", 0.12))
    checker_code_boost_topn = int(search_cfg.get("checker_code_boost_topn", 24))
    checker_seed_enabled = bool(search_cfg.get("checker_seed_enabled", True))
    checker_seed_score_base = float(search_cfg.get("checker_seed_score_base", 1.30))
    checker_seed_boost_scale = float(search_cfg.get("checker_seed_boost_scale", 0.75))
    checker_seed_min_score = float(search_cfg.get("checker_seed_min_score", 0.08))
    checker_seed_max_injected = int(search_cfg.get("checker_seed_max_injected", 24))
    checker_crop_max_area = float(search_cfg.get("checker_crop_max_area", 0.28))
    checker_large_crop_max_area = float(search_cfg.get("checker_large_crop_max_area", 0.42))
    checker_large_crop_query_threshold = float(search_cfg.get("checker_large_crop_query_threshold", checker_query_threshold))
    checker_large_crop_bw_mix = float(search_cfg.get("checker_large_crop_bw_mix", 0.45))
    checker_region_rescue_enabled = bool(search_cfg.get("checker_region_rescue_enabled", True))
    checker_region_rescue_min_seed = float(search_cfg.get("checker_region_rescue_min_seed", 1.38))
    checker_region_rescue_max_rows = int(search_cfg.get("checker_region_rescue_max_rows", 3))
    checker_suppress_sleeve_threshold = float(search_cfg.get("checker_suppress_sleeve_threshold", 0.14))
    checker_suppress_sleeve_bw_mix = float(search_cfg.get("checker_suppress_sleeve_bw_mix", 0.55))
    accent_pattern_enabled = bool(search_cfg.get("accent_pattern_enabled", False))
    accent_pattern_seed_score_base = float(search_cfg.get("accent_pattern_seed_score_base", 0.90))
    accent_pattern_boost_scale = float(search_cfg.get("accent_pattern_boost_scale", 0.24))
    accent_pattern_min_score = float(search_cfg.get("accent_pattern_min_score", 0.42))
    accent_pattern_max_injected = int(search_cfg.get("accent_pattern_max_injected", 24))
    accent_pattern_min_pixels = int(search_cfg.get("accent_pattern_min_pixels", 80))
    accent_pattern_max_edge = int(search_cfg.get("accent_pattern_max_edge", 192))
    accent_pattern_crop_enabled = bool(search_cfg.get("accent_pattern_crop_enabled", True))
    accent_pattern_small_region_max_score = float(search_cfg.get("accent_pattern_small_region_max_score", 0.68))
    accent_region_rescue_enabled = bool(search_cfg.get("accent_region_rescue_enabled", True))
    accent_region_rescue_min_sim = float(search_cfg.get("accent_region_rescue_min_sim", 0.70))
    accent_region_rescue_max_rows = int(search_cfg.get("accent_region_rescue_max_rows", 3))
    collar_contour_enabled = bool(search_cfg.get("collar_contour_enabled", True))
    collar_contour_seed_score_base = float(search_cfg.get("collar_contour_seed_score_base", 1.12))
    collar_contour_boost_scale = float(search_cfg.get("collar_contour_boost_scale", 0.28))
    collar_contour_min_score = float(search_cfg.get("collar_contour_min_score", 0.52))
    collar_contour_max_injected = int(search_cfg.get("collar_contour_max_injected", 24))
    collar_contour_size = int(search_cfg.get("collar_contour_size", 48))
    collar_contour_query_component_views = bool(search_cfg.get("collar_contour_query_component_views", True))
    collar_contour_query_max_sigs = int(search_cfg.get("collar_contour_query_max_sigs", 32))
    collar_contour_code_prior_boost = float(search_cfg.get("collar_contour_code_prior_boost", 0.10))
    collar_contour_region_score_base = float(search_cfg.get("collar_contour_region_score_base", 0.84))
    collar_contour_region_score_scale = float(search_cfg.get("collar_contour_region_score_scale", 0.18))
    collar_contour_region_score_max = float(search_cfg.get("collar_contour_region_score_max", 1.80))
    collar_contour_repeat_min_score = float(search_cfg.get("collar_contour_repeat_min_score", 0.62))
    collar_contour_repeat_min_hits = int(search_cfg.get("collar_contour_repeat_min_hits", 2))
    collar_contour_repeat_boost = float(search_cfg.get("collar_contour_repeat_boost", 0.16))
    collar_contour_repeat_view_boost = float(search_cfg.get("collar_contour_repeat_view_boost", 0.04))
    collar_contour_multi_image_boost = float(search_cfg.get("collar_contour_multi_image_boost", 0.08))
    collar_contour_repeat_max_boost = float(search_cfg.get("collar_contour_repeat_max_boost", 0.24))
    collar_contour_near_tie_diversify_enabled = bool(
        search_cfg.get("collar_contour_near_tie_diversify_enabled", True)
    )
    collar_contour_near_tie_margin = float(search_cfg.get("collar_contour_near_tie_margin", 0.008))
    collar_contour_near_tie_min_window = int(search_cfg.get("collar_contour_near_tie_min_window", 12))
    collar_contour_near_tie_window_extra = int(search_cfg.get("collar_contour_near_tie_window_extra", 5))
    collar_chevron_enabled = bool(search_cfg.get("collar_chevron_enabled", True))
    collar_chevron_query_min_score = float(search_cfg.get("collar_chevron_query_min_score", 0.30))
    collar_chevron_standard_min_score = float(search_cfg.get("collar_chevron_standard_min_score", 0.50))
    collar_chevron_min_contour_score = float(search_cfg.get("collar_chevron_min_contour_score", 0.30))
    collar_chevron_score_boost = float(search_cfg.get("collar_chevron_score_boost", 0.22))
    collar_chevron_seed_score_base = float(search_cfg.get("collar_chevron_seed_score_base", 1.24))
    collar_chevron_max_injected = int(search_cfg.get("collar_chevron_max_injected", 48))
    collar_chevron_code_min_score = float(
        search_cfg.get("collar_chevron_code_min_score", max(0.28, collar_chevron_standard_min_score - 0.22))
    )
    collar_chevron_code_contour_min_score = float(search_cfg.get("collar_chevron_code_contour_min_score", 0.40))
    collar_chevron_code_contour_boost = float(search_cfg.get("collar_chevron_code_contour_boost", 0.32))
    collar_chevron_code_fallback_contour_min_score = float(
        search_cfg.get("collar_chevron_code_fallback_contour_min_score", max(collar_chevron_code_contour_min_score, 0.42))
    )
    collar_chevron_code_fallback_boost = float(search_cfg.get("collar_chevron_code_fallback_boost", 0.22))
    checker_suppress_when_accent = bool(search_cfg.get("checker_suppress_when_accent", True))
    checker_accent_suppress_below = float(search_cfg.get("checker_accent_suppress_below", 0.14))
    sleeve_pattern_enabled = bool(search_cfg.get("sleeve_pattern_enabled", False))
    sleeve_pattern_seed_score_base = float(search_cfg.get("sleeve_pattern_seed_score_base", 0.91))
    sleeve_pattern_boost_scale = float(search_cfg.get("sleeve_pattern_boost_scale", 0.25))
    sleeve_pattern_min_score = float(search_cfg.get("sleeve_pattern_min_score", 0.48))
    sleeve_pattern_max_injected = int(search_cfg.get("sleeve_pattern_max_injected", 16))
    sleeve_pair_prior_boost = float(search_cfg.get("sleeve_pair_prior_boost", 0.08))
    sleeve_pair_prior_candidate_boost = float(search_cfg.get("sleeve_pair_prior_candidate_boost", 0.20))
    sleeve_pattern_skip_when_full_accent = bool(search_cfg.get("sleeve_pattern_skip_when_full_accent", True))
    sleeve_pattern_crop_max_area = float(search_cfg.get("sleeve_pattern_crop_max_area", 0.28))
    sleeve_pattern_small_region_enabled = bool(search_cfg.get("sleeve_pattern_small_region_enabled", True))
    sleeve_pattern_small_region_max_score = float(search_cfg.get("sleeve_pattern_small_region_max_score", 0.72))
    sleeve_pattern_large_region_rescue_enabled = bool(search_cfg.get("sleeve_pattern_large_region_rescue_enabled", True))
    sleeve_pattern_large_region_rescue_max_area = float(search_cfg.get("sleeve_pattern_large_region_rescue_max_area", 0.40))
    sleeve_pattern_large_region_rescue_max_score = float(search_cfg.get("sleeve_pattern_large_region_rescue_max_score", 0.74))
    accessory_pattern_enabled = bool(search_cfg.get("accessory_pattern_enabled", False))
    accessory_pattern_seed_score_base = float(search_cfg.get("accessory_pattern_seed_score_base", 0.92))
    accessory_pattern_boost_scale = float(search_cfg.get("accessory_pattern_boost_scale", 0.24))
    accessory_pattern_min_score = float(search_cfg.get("accessory_pattern_min_score", 0.50))
    accessory_pattern_max_injected = int(search_cfg.get("accessory_pattern_max_injected", 16))
    accessory_hat_prior_boost = float(search_cfg.get("accessory_hat_prior_boost", 0.10))
    accessory_hat_code_prefixes = [
        str(x).strip().upper()
        for x in search_cfg.get("accessory_hat_code_prefixes", ["BM"])
        if str(x).strip()
    ]
    accessory_hat_code_boost = float(search_cfg.get("accessory_hat_code_boost", 0.16))
    accessory_near_square_crop_enabled = bool(search_cfg.get("accessory_near_square_crop_enabled", True))
    accessory_near_square_crop_min_aspect = float(search_cfg.get("accessory_near_square_crop_min_aspect", 0.65))
    accessory_near_square_crop_max_aspect = float(search_cfg.get("accessory_near_square_crop_max_aspect", 1.25))
    accessory_near_square_crop_max_area = float(search_cfg.get("accessory_near_square_crop_max_area", 0.28))
    accessory_hat_override_max_aspect = float(search_cfg.get("accessory_hat_override_max_aspect", 1.10))
    accessory_disable_wide_crop_enabled = bool(search_cfg.get("accessory_disable_wide_crop_enabled", True))
    accessory_hat_prior_seed_enabled = bool(search_cfg.get("accessory_hat_prior_seed_enabled", True))
    accessory_hat_prior_query_threshold = float(search_cfg.get("accessory_hat_prior_query_threshold", 0.42))
    accessory_hat_prior_seed_min_score = float(search_cfg.get("accessory_hat_prior_seed_min_score", 0.45))
    accessory_hat_prior_seed_score_base = float(search_cfg.get("accessory_hat_prior_seed_score_base", 1.22))
    accessory_hat_prior_seed_boost_scale = float(search_cfg.get("accessory_hat_prior_seed_boost_scale", 0.22))
    accessory_hat_prior_seed_max_injected = int(search_cfg.get("accessory_hat_prior_seed_max_injected", 16))
    accessory_hat_region_rescue_enabled = bool(search_cfg.get("accessory_hat_region_rescue_enabled", True))
    accessory_hat_region_rescue_min_seed = float(search_cfg.get("accessory_hat_region_rescue_min_seed", 1.45))
    accessory_hat_region_rescue_min_aspect = float(search_cfg.get("accessory_hat_region_rescue_min_aspect", 1.35))
    accessory_hat_region_rescue_max_aspect = float(search_cfg.get("accessory_hat_region_rescue_max_aspect", 2.40))
    accessory_hat_region_rescue_max_rows = int(search_cfg.get("accessory_hat_region_rescue_max_rows", 1))
    accessory_hat_family_region_rescue_enabled = bool(search_cfg.get("accessory_hat_family_region_rescue_enabled", True))
    accessory_hat_family_region_rescue_min_aspect = float(
        search_cfg.get("accessory_hat_family_region_rescue_min_aspect", accessory_hat_region_rescue_min_aspect)
    )
    accessory_hat_family_region_rescue_max_aspect = float(
        search_cfg.get("accessory_hat_family_region_rescue_max_aspect", accessory_hat_region_rescue_max_aspect)
    )
    accessory_hat_family_region_rescue_min_prior = float(search_cfg.get("accessory_hat_family_region_rescue_min_prior", 0.55))
    accessory_hat_family_region_rescue_score = float(search_cfg.get("accessory_hat_family_region_rescue_score", 1.36))
    accessory_hat_family_region_rescue_max_rows = int(search_cfg.get("accessory_hat_family_region_rescue_max_rows", 1))
    accessory_hat_from_sleeve_region_rescue_enabled = bool(search_cfg.get("accessory_hat_from_sleeve_region_rescue_enabled", True))
    accessory_hat_from_sleeve_region_rescue_min_seed = float(search_cfg.get("accessory_hat_from_sleeve_region_rescue_min_seed", 1.38))
    accessory_hat_from_sleeve_region_rescue_max_rows = int(search_cfg.get("accessory_hat_from_sleeve_region_rescue_max_rows", 1))
    accessory_region_requires_hat_prior = bool(search_cfg.get("accessory_region_requires_hat_prior", True))
    accessory_region_hat_prior_threshold = float(search_cfg.get("accessory_region_hat_prior_threshold", accessory_hat_prior_query_threshold))
    accessory_region_suppress_when_accent = bool(search_cfg.get("accessory_region_suppress_when_accent", True))
    low_confidence_enabled = bool(search_cfg.get("low_confidence_enabled", True))
    low_confidence_margin_threshold = float(search_cfg.get("low_confidence_margin_threshold", 0.015))
    low_confidence_top1_threshold = float(search_cfg.get("low_confidence_top1_threshold", 0.72))
    similar_images_topn = int(search_cfg.get("similar_images_topn", 8))
    region_similar_images_topn = int(search_cfg.get("region_similar_images_topn", max(8, similar_images_topn)))
    confidence_high_threshold = float(search_cfg.get("confidence_high_threshold", 0.08))
    confidence_medium_threshold = float(search_cfg.get("confidence_medium_threshold", 0.03))
    display_score_scale = float(search_cfg.get("display_score_scale", 8.0))
    display_score_bias = float(search_cfg.get("display_score_bias", 0.72))
    phash_enabled = bool(search_cfg.get("phash_enabled", True))
    phash_boost_weight = float(search_cfg.get("phash_boost_weight", 0.18))
    phash_apply_topn = int(search_cfg.get("phash_apply_topn", 256))
    scene_text_hint_enabled = bool(search_cfg.get("scene_text_hint_enabled", False))
    scene_text_min_token_len = int(search_cfg.get("scene_text_min_token_len", 4))
    scene_text_seed_score_base = float(search_cfg.get("scene_text_seed_score_base", 0.88))
    scene_text_boost_scale = float(search_cfg.get("scene_text_boost_scale", 0.18))
    scene_text_max_candidates_per_token = int(search_cfg.get("scene_text_max_candidates_per_token", 64))
    scene_text_max_injected = int(search_cfg.get("scene_text_max_injected", 24))
    scene_text_region_rescue_enabled = bool(search_cfg.get("scene_text_region_rescue_enabled", True))
    scene_text_region_rescue_min_score = float(search_cfg.get("scene_text_region_rescue_min_score", 1.25))
    scene_text_region_rescue_min_ratio = float(search_cfg.get("scene_text_region_rescue_min_ratio", 0.66))
    scene_text_region_rescue_max_rows = int(search_cfg.get("scene_text_region_rescue_max_rows", 3))
    scene_text_suppress_when_region_min_score = float(search_cfg.get("scene_text_suppress_when_region_min_score", 0.62))
    scene_text_small_region_max_score = float(search_cfg.get("scene_text_small_region_max_score", 0.68))
    strip_mode_enabled = bool(search_cfg.get("strip_mode_enabled", True))
    strip_aspect_threshold = float(search_cfg.get("strip_aspect_threshold", 2.4))
    strip_fill_threshold = float(search_cfg.get("strip_fill_threshold", 0.42))
    strip_w_clip = float(search_cfg.get("strip_w_clip", 0.35))
    strip_w_shape = float(search_cfg.get("strip_w_shape", 0.30))
    strip_w_color = float(search_cfg.get("strip_w_color", 0.10))
    strip_w_stripe = float(search_cfg.get("strip_w_stripe", 0.25))
    auth_cfg = cfg.get("auth", {})
    catalog_cfg = cfg.get("catalog", {})
    color_card_cfg = cfg.get("color_card", {})
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

    catalog_db_path = Path(catalog_cfg.get("db_path", "data/product_catalog.db"))
    color_card_db_path = Path(color_card_cfg.get("db_path", "data/color_cards.db"))
    catalog_import_source_dir = str(catalog_cfg.get("import_source_dir", "")).strip()
    catalog_public = bool(catalog_cfg.get("public_endpoints", True))
    catalog_image_max_edge = int(catalog_cfg.get("image_max_edge", 420))
    catalog_image_quality = int(catalog_cfg.get("image_quality", 68))
    catalog_browser_upload_max_files = max(1, int(catalog_cfg.get("browser_upload_max_files", 30)))
    catalog_web_auth_cfg = catalog_cfg.get("web_auth", {})
    catalog_web_auth_enabled = bool(catalog_web_auth_cfg.get("enabled", True))
    catalog_web_users_cfg = catalog_web_auth_cfg.get("users", [])
    catalog_web_users: Dict[str, str] = {}
    if isinstance(catalog_web_users_cfg, list):
        for item in catalog_web_users_cfg:
            if not isinstance(item, dict):
                continue
            username = str(item.get("username", "")).strip()
            password = str(item.get("password", ""))
            if username and password:
                catalog_web_users[username] = password
    if not catalog_web_users:
        catalog_web_username = str(catalog_web_auth_cfg.get("username", "admin")).strip()
        catalog_web_password = str(catalog_web_auth_cfg.get("password", "change-me"))
        if catalog_web_username and catalog_web_password:
            catalog_web_users[catalog_web_username] = catalog_web_password
    catalog_web_captcha_enabled = bool(catalog_web_auth_cfg.get("captcha_enabled", True))
    catalog_web_captcha_timezone = str(catalog_web_auth_cfg.get("captcha_timezone", "Asia/Shanghai")).strip() or "Asia/Shanghai"
    catalog_web_session_secret = str(catalog_web_auth_cfg.get("session_secret", "replace-with-catalog-session-secret")).strip()
    catalog_web_session_ttl_sec = int(catalog_web_auth_cfg.get("session_ttl_sec", 43200))
    catalog_web_cookie_name = str(catalog_web_auth_cfg.get("cookie_name", "catalog_session")).strip() or "catalog_session"
    catalog_external_auth_cfg = catalog_cfg.get("external_token_auth", {})
    catalog_external_token_enabled = bool(catalog_external_auth_cfg.get("enabled", True))
    catalog_external_verify_url = str(catalog_external_auth_cfg.get("verify_url", "")).strip()
    catalog_external_verify_method = str(catalog_external_auth_cfg.get("verify_method", "POST")).strip().upper() or "POST"
    catalog_external_verify_timeout_sec = float(catalog_external_auth_cfg.get("verify_timeout_sec", 3.0))
    catalog_external_cache_ttl_sec = int(catalog_external_auth_cfg.get("cache_ttl_sec", 300))
    catalog_external_fail_open = bool(catalog_external_auth_cfg.get("fail_open", False))
    catalog_external_allow_unverified_env = os.getenv("CATALOG_ALLOW_UNVERIFIED_TOKENS", "").strip().lower()
    catalog_external_allow_unverified_tokens = bool(catalog_external_auth_cfg.get("allow_unverified_tokens", False))
    if catalog_external_allow_unverified_env in {"1", "true", "yes"}:
        catalog_external_allow_unverified_tokens = True
    elif catalog_external_allow_unverified_env in {"0", "false", "no"}:
        catalog_external_allow_unverified_tokens = False
    catalog_external_allowed_tokens = {
        str(item).strip()
        for item in catalog_external_auth_cfg.get("allowed_tokens", [])
        if str(item).strip()
    }
    catalog_external_allowed_tokens.update(
        item.strip()
        for item in os.getenv("CATALOG_EXTERNAL_TOKENS", "").split(",")
        if item.strip()
    )
    catalog_external_default_permissions = [
        str(item).strip()
        for item in catalog_external_auth_cfg.get(
            "default_permissions",
            ["product:view", "product:create", "color:view", "color:create"],
        )
        if str(item).strip()
    ]
    catalog_external_user_fields = [
        str(item).strip()
        for item in catalog_external_auth_cfg.get("user_fields", ["user_id", "userId", "user", "openid", "sub"])
        if str(item).strip()
    ]
    catalog_external_permission_fields = [
        str(item).strip()
        for item in catalog_external_auth_cfg.get("permission_fields", ["permissions", "perms", "scope"])
        if str(item).strip()
    ]

    def _region_feature_cache_path(backend: str, feature_dir: Path) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(feature_dir))
        return Path("outputs") / f"region_feature_cache_{backend}_{safe}.npz"

    def _region_standard_views(img: Image.Image, max_component_views: int = 3) -> List[tuple[str, Image.Image]]:
        w, h = img.size
        if w < 40 or h < 40:
            return [("full", img)]
        boxes: List[tuple[str, tuple[int, int, int, int]]] = [
            ("full", (0, 0, w, h)),
            ("center", (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))),
            ("left", (0, 0, int(w * 0.58), h)),
            ("right", (int(w * 0.42), 0, w, h)),
            ("top", (0, 0, w, int(h * 0.58))),
            ("top_narrow", (0, 0, w, int(h * 0.42))),
            ("bottom", (0, int(h * 0.42), w, h)),
            ("mid_band", (0, int(h * 0.22), w, int(h * 0.82))),
            ("upper_band", (0, int(h * 0.08), w, int(h * 0.55))),
            ("upper_narrow_band", (0, int(h * 0.04), w, int(h * 0.42))),
            ("lower_band", (0, int(h * 0.45), w, int(h * 0.95))),
            ("tl", (0, 0, int(w * 0.62), int(h * 0.62))),
            ("tr", (int(w * 0.38), 0, w, int(h * 0.62))),
            ("bl", (0, int(h * 0.38), int(w * 0.62), h)),
            ("br", (int(w * 0.38), int(h * 0.38), w, h)),
            ("top_left_band", (0, 0, int(w * 0.58), int(h * 0.46))),
            ("top_right_band", (int(w * 0.42), 0, w, int(h * 0.46))),
            ("collar_left_focus", (0, int(h * 0.02), int(w * 0.46), int(h * 0.52))),
            ("collar_right_focus", (int(w * 0.54), int(h * 0.02), w, int(h * 0.52))),
            ("collar_center_bridge", (int(w * 0.22), 0, int(w * 0.78), int(h * 0.40))),
            ("collar_right_mid", (int(w * 0.50), int(h * 0.20), w, int(h * 0.78))),
            ("collar_right_lower", (int(w * 0.50), int(h * 0.36), w, int(h * 0.92))),
            ("collar_left_mid_strip", (0, int(h * 0.18), int(w * 0.72), int(h * 0.72))),
            ("collar_lower_strip", (0, int(h * 0.40), int(w * 0.72), int(h * 0.92))),
        ]

        views: List[tuple[str, Image.Image]] = []
        seen = set()
        for tag, (x0, y0, x1, y1) in boxes:
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 - x0 < 32 or y1 - y0 < 32:
                continue
            key = (x0, y0, x1, y1)
            if key in seen:
                continue
            seen.add(key)
            views.append((tag, img.crop(key)))

        if cv2 is not None:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
            gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
            maxc = arr.max(axis=-1).astype(np.float32)
            minc = arr.min(axis=-1).astype(np.float32)
            sat = (maxc - minc) / np.maximum(maxc, 1.0)
            border_band = max(16, int(min(w, h) * 0.10))
            border_rgb = np.concatenate(
                [
                    arr[max(0, h - border_band):, :, :].reshape(-1, 3),
                    arr[:, max(0, w - border_band):, :].reshape(-1, 3),
                ],
                axis=0,
            )
            bg_rgb = np.median(border_rgb, axis=0).astype(np.float32)
            bg_gray = float(0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2])
            color_dist = np.sqrt(((arr.astype(np.float32) - bg_rgb.reshape(1, 1, 3)) ** 2).sum(axis=-1))
            bright_fg = gray > (bg_gray + 55.0)
            dark_or_colored = (gray < (bg_gray - 18.0)) | (sat > 0.08)
            dark_or_colored = cv2.dilate(dark_or_colored.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
            fg_mask = bright_fg | (bright_fg & dark_or_colored) | (dark_or_colored & (gray > (bg_gray + 25.0))) | (color_dist > 42.0)
            if int(fg_mask.sum()) < max(64, int(w * h * 0.004)):
                fg_mask = (gray < 235.0) & ((sat > 0.08) | (gray < 170.0))
            fg = fg_mask.astype(np.uint8)
            fg[: min(int(h * 0.10), 80), :] = 0
            num, _labels, stats, _centers = cv2.connectedComponentsWithStats(fg, connectivity=8)
            comps: List[tuple[int, tuple[int, int, int, int]]] = []
            min_area = max(80, int(w * h * 0.015))
            for i in range(1, num):
                x, y, ww, hh, area = stats[i]
                if int(area) < min_area or int(ww) < 24 or int(hh) < 24:
                    continue
                comps.append((int(area), (int(x), int(y), int(ww), int(hh))))
            comps.sort(key=lambda item: item[0], reverse=True)
            top_comps = comps[: max(2, max_component_views + 1)]
            for idx, (_area, (x, y, ww, hh)) in enumerate(comps[:max_component_views]):
                pad_x = max(8, int(ww * 0.12))
                pad_y = max(8, int(hh * 0.12))
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(w, x + ww + pad_x)
                y1 = min(h, y + hh + pad_y)
                key = (x0, y0, x1, y1)
                if x1 - x0 < 32 or y1 - y0 < 32 or key in seen:
                    continue
                seen.add(key)
                views.append((f"comp{idx}", img.crop(key)))
                if hh >= 72 and ww >= 24:
                    for sub_idx, (sy0, sy1) in enumerate(((0.00, 0.44), (0.24, 0.70), (0.50, 0.92), (0.62, 1.00))):
                        sx0 = max(0, x - pad_x)
                        sx1 = min(w, x + ww + pad_x)
                        sub_y0 = max(0, y + int(hh * sy0) - max(4, int(pad_y * 0.45)))
                        sub_y1 = min(h, y + int(hh * sy1) + max(4, int(pad_y * 0.45)))
                        sub_key = (sx0, sub_y0, sx1, sub_y1)
                        if sx1 - sx0 < 24 or sub_y1 - sub_y0 < 24 or sub_key in seen:
                            continue
                        seen.add(sub_key)
                        views.append((f"comp{idx}_stripe{sub_idx}", img.crop(sub_key)))
            pair_idx = 0
            for left_idx in range(len(top_comps)):
                _la, (lx, ly, lww, lhh) = top_comps[left_idx]
                for right_idx in range(left_idx + 1, len(top_comps)):
                    _ra, (rx, ry, rww, rhh) = top_comps[right_idx]
                    if max(ly + lhh, ry + rhh) > int(h * 0.68):
                        continue
                    pad_x = max(10, int(min(lww, rww) * 0.10))
                    pad_y = max(8, int(min(lhh, rhh) * 0.12))
                    x0 = max(0, min(lx, rx) - pad_x)
                    y0 = max(0, min(ly, ry) - pad_y)
                    x1 = min(w, max(lx + lww, rx + rww) + pad_x)
                    y1 = min(h, max(ly + lhh, ry + rhh) + pad_y)
                    key = (x0, y0, x1, y1)
                    if x1 - x0 < 40 or y1 - y0 < 32 or y1 - y0 > int(h * 0.62) or key in seen:
                        continue
                    seen.add(key)
                    views.append((f"comp_pair{pair_idx}", img.crop(key)))
                    pair_idx += 1
                    if pair_idx >= 4:
                        break
                if pair_idx >= 4:
                    break
            top_cut = min(h, max(48, int(h * 0.62)))
            fg_top = fg[:top_cut, :]
            num_top, _labels_top, stats_top, _centers_top = cv2.connectedComponentsWithStats(fg_top, connectivity=8)
            fine_comps: List[tuple[int, tuple[int, int, int, int]]] = []
            fine_min_area = max(48, int(w * h * 0.004))
            for i in range(1, num_top):
                x, y, ww, hh, area = stats_top[i]
                if int(area) < fine_min_area or int(ww) < 18 or int(hh) < 18:
                    continue
                if int(y + hh) > int(h * 0.68):
                    continue
                fine_comps.append((int(area), (int(x), int(y), int(ww), int(hh))))
            fine_comps.sort(key=lambda item: item[0], reverse=True)
            for idx, (_area, (x, y, ww, hh)) in enumerate(fine_comps[:6]):
                pad_x = max(6, int(ww * 0.10))
                pad_y = max(6, int(hh * 0.10))
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(w, x + ww + pad_x)
                y1 = min(h, y + hh + pad_y)
                key = (x0, y0, x1, y1)
                if x1 - x0 < 24 or y1 - y0 < 24 or key in seen:
                    continue
                seen.add(key)
                views.append((f"top_comp{idx}", img.crop(key)))
        return views

    def _sync_search_feature_dir() -> Path:
        src_files = [
            fp
            for fp in collect_images(standard_dir, standard_pattern, image_exts)
            if not fp.name.upper().startswith("MY-")
        ]
        search_feature_dir.mkdir(parents=True, exist_ok=True)
        wanted = {fp.name for fp in src_files}
        for stale in search_feature_dir.iterdir():
            if stale.is_file() and stale.name not in wanted:
                try:
                    stale.unlink()
                except Exception:
                    logging.warning("failed to remove stale search feature file: %s", stale)
        for src in src_files:
            dst = search_feature_dir / src.name
            try:
                src_stat = src.stat()
                if dst.exists():
                    dst_stat = dst.stat()
                    if dst_stat.st_size == src_stat.st_size and int(dst_stat.st_mtime) == int(src_stat.st_mtime):
                        continue
                    dst.unlink()
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            except Exception:
                logging.warning("failed to sync search feature file: %s", src)
        return search_feature_dir

    def _build_region_feature_db_with_cache(feature_dir: Path) -> tuple[List[str], np.ndarray]:
        files = collect_images(feature_dir, standard_pattern, image_exts)
        sigs = []
        for fp in files:
            st = fp.stat()
            sigs.append(f"{fp.name}|{st.st_size}|{int(st.st_mtime)}")
        cache_key = json.dumps(
            {
                "kind": "region_crop_recall",
                "version": 7,
                "backend": region_crop_recall_backend,
                "weights": [region_w_clip, region_w_shape, region_w_color, region_w_stripe],
                "standard_views": "grid_halves_bands_components_topdetail_collar_mid_strip_comp_stripes",
                "standard_crop_ratio": float(region_standard_crop_ratio),
                "pattern": standard_pattern,
                "exts": list(image_exts),
                "db_feature_dtype": str(db_feature_dtype),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_path = _region_feature_cache_path(region_crop_recall_backend, feature_dir)
        if feature_cache_enabled and cache_path.exists():
            try:
                arr = np.load(cache_path, allow_pickle=True)
                if str(arr["cache_key"].item()) == cache_key and list(arr["file_sigs"]) == sigs:
                    cache_names = [str(x) for x in arr["names"]]
                    if str(db_feature_dtype).lower() == "float16":
                        cache_feats = arr["feats"].astype(np.float16)
                    else:
                        cache_feats = arr["feats"].astype(np.float32)
                    logging.info("region feature cache hit: %s (%d items)", cache_path, len(cache_names))
                    return cache_names, cache_feats
            except Exception:
                pass

        t0 = time.perf_counter()
        cache_names: List[str] = []
        feat_list: List[np.ndarray] = []
        for fp in files:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            for idx, (tag, view) in enumerate(_region_standard_views(img)):
                cache_names.append(f"{fp.name}@r{idx}_{tag}")
                feat_list.append(
                    extract_embedding(
                        view,
                        region_crop_recall_backend,
                        region_w_clip,
                        region_w_shape,
                        region_w_color,
                        region_w_stripe,
                    )
                )
        cache_feats = np.vstack(feat_list).astype(np.float32) if feat_list else np.zeros((0, 1), dtype=np.float32)
        if str(db_feature_dtype).lower() == "float16":
            cache_feats = cache_feats.astype(np.float16)
        else:
            cache_feats = cache_feats.astype(np.float32)
        if feature_cache_enabled:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                cache_key=np.array([cache_key], dtype=object),
                file_sigs=np.array(sigs, dtype=object),
                names=np.array(cache_names, dtype=object),
                feats=cache_feats,
            )
            logging.info("region feature cache write: %s", cache_path)
        logging.info("build region feature db done: backend=%s items=%d in %.2fs", region_crop_recall_backend, len(cache_names), time.perf_counter() - t0)
        return cache_names, cache_feats

    _setup_logging()
    search_assets_lock = threading.Lock()
    names: List[str] = []
    feats: np.ndarray | None = None
    secondary_names: List[str] = []
    secondary_feats: np.ndarray | None = None
    region_names: List[str] = []
    region_feats: np.ndarray | None = None
    rerank_candidate_cache: Dict[str, List[Dict[str, Any]]] = {}
    label_memory_refs: List[tuple[str, np.ndarray]] = []
    scene_text_index: Dict[str, Any] | None = None
    standard_image_by_code_key: Dict[str, str] = {}

    def _reload_search_assets(reason: str = "startup") -> None:
        nonlocal names, feats, secondary_names, secondary_feats, region_names, region_feats
        nonlocal rerank_candidate_cache, label_memory_refs, scene_text_index
        nonlocal standard_image_by_code_key
        t_reload = time.perf_counter()
        feature_dir = _sync_search_feature_dir()
        next_names, next_feats = build_feature_db_with_cache(
            feature_dir,
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
        logging.info("api loaded primary db: %d items", len(next_names))

        next_secondary_names: List[str] = []
        next_secondary_feats: np.ndarray | None = None
        if secondary_feature_backend and secondary_feature_backend != feature_backend:
            next_secondary_names, next_secondary_feats = build_feature_db_with_cache(
                feature_dir,
                standard_pattern,
                secondary_feature_backend,
                image_exts,
                secondary_w_clip,
                secondary_w_shape,
                secondary_w_color,
                secondary_w_stripe,
                standard_multicrop=standard_multicrop,
                standard_crop_ratio=standard_crop_ratio,
                use_cache=feature_cache_enabled,
                db_feature_dtype=db_feature_dtype,
            )
            logging.info(
                "api loaded secondary db: backend=%s items=%d",
                secondary_feature_backend,
                len(next_secondary_names),
            )

        next_region_names: List[str] = []
        next_region_feats: np.ndarray | None = None
        if region_crop_recall_enabled:
            next_region_names, next_region_feats = _build_region_feature_db_with_cache(feature_dir)
            logging.info(
                "api loaded region crop db: backend=%s items=%d",
                region_crop_recall_backend,
                len(next_region_names),
            )

        next_rerank_candidate_cache: Dict[str, List[Dict[str, Any]]] = {}
        if rerank_enabled and preload_rerank_candidate_cache:
            t0 = time.perf_counter()
            next_rerank_candidate_cache = precompute_rerank_candidate_cache(
                standard_dir=standard_dir,
                names=next_names,
                candidate_views_max=rerank_candidate_views_max,
            )
            logging.info(
                "api loaded rerank candidate cache: %d files in %.2fs",
                len(next_rerank_candidate_cache),
                time.perf_counter() - t0,
            )
        elif rerank_enabled:
            logging.info("api rerank candidate cache preload disabled; using lazy cache on requests")

        next_label_memory_refs = precompute_label_memory_refs(label_memory_path) if label_memory_enabled else []
        if label_memory_enabled:
            logging.info("api loaded label memory refs: %d", len(next_label_memory_refs))

        next_scene_text_index: Dict[str, Any] | None = None
        if scene_text_hint_enabled:
            t0 = time.perf_counter()
            next_scene_text_index = precompute_scene_text_index(
                standard_dir=standard_dir,
                pattern=standard_pattern,
                exts=image_exts,
                min_token_len=scene_text_min_token_len,
                use_cache=True,
            )
            logging.info(
                "api loaded scene text index: %d images in %.2fs",
                int(next_scene_text_index.get("total_images", 0)) if isinstance(next_scene_text_index, dict) else 0,
                time.perf_counter() - t0,
            )

        next_standard_image_by_code_key: Dict[str, str] = {}
        for image_name in list(next_names) + list(next_region_names):
            base_name = str(image_name).split("@", 1)[0]
            code_key = _style_code_key(filename_to_style_code(base_name))
            if code_key and code_key not in next_standard_image_by_code_key:
                next_standard_image_by_code_key[code_key] = base_name

        with search_assets_lock:
            names = next_names
            feats = next_feats
            secondary_names = next_secondary_names
            secondary_feats = next_secondary_feats
            region_names = next_region_names
            region_feats = next_region_feats
            rerank_candidate_cache = next_rerank_candidate_cache
            label_memory_refs = next_label_memory_refs
            scene_text_index = next_scene_text_index
            standard_image_by_code_key = next_standard_image_by_code_key
        logging.info("search assets reloaded: reason=%s in %.2fs", reason, time.perf_counter() - t_reload)

    _reload_search_assets("startup")
    catalog_store = CatalogStore(catalog_db_path)
    color_card_store = ColorCardStore(color_card_db_path)
    catalog_write_lock = threading.Lock()
    with catalog_write_lock:
        sync_stats = catalog_store.sync_from_standard_dir(standard_dir, image_exts)
        logging.info("catalog sync done: %s", sync_stats)
    catalog_import_jobs: Dict[str, Dict[str, Any]] = {}
    catalog_import_lock = threading.Lock()
    catalog_external_token_cache: Dict[str, Dict[str, Any]] = {}
    catalog_external_token_cache_lock = threading.Lock()
    allowed_image_exts = {f".{str(ext).lower().lstrip('.')}" for ext in image_exts}
    tesseract_bin = shutil.which("tesseract")
    debug_cfg = cfg.get("debug", {})
    debug_query_enabled = bool(debug_cfg.get("save_query_images", True))
    debug_query_dir = Path(debug_cfg.get("query_image_dir", "outputs/debug_queries"))
    if debug_query_enabled:
        debug_query_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="search-similar-style-api", version="1.0.0")
    app.mount("/print-static", StaticFiles(directory=str(PRINT_STATIC_DIR)), name="print-static")
    app.mount("/print-storage", StaticFiles(directory=str(PRINT_STORAGE_DIR)), name="print-storage")
    app.mount("/recolor-static", StaticFiles(directory=str(RECOLOR_OUTPUT_DIR.parent)), name="recolor-static")

    app.state.ready = False
    app.state.ready_detail = "initializing"
    image_cache_dir = Path("outputs/image_cache")
    image_cache_dir.mkdir(parents=True, exist_ok=True)
    catalog_upload_dir = Path("outputs/catalog_import_uploads")
    catalog_upload_dir.mkdir(parents=True, exist_ok=True)
    wechat_access_token_cache: Dict[str, Any] = {"token": "", "expires_at": 0.0}
    color_meter_native_readings: Dict[str, Dict[str, Any]] = {}

    def _wechat_get_access_token() -> str:
        now = time.time()
        cached_token = str(wechat_access_token_cache.get("token", ""))
        cached_expires = float(wechat_access_token_cache.get("expires_at", 0.0))
        if cached_token and cached_expires - 120 > now:
            return cached_token
        if not wechat_appid or not wechat_appsecret:
            raise WechatContentSecurityError("微信内容安全未配置 AppID/AppSecret")
        params = urllib.parse.urlencode(
            {
                "grant_type": "client_credential",
                "appid": wechat_appid,
                "secret": wechat_appsecret,
            }
        )
        url = f"https://api.weixin.qq.com/cgi-bin/token?{params}"
        try:
            with urllib.request.urlopen(url, timeout=wechat_security_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            raise WechatContentSecurityError(f"获取微信内容安全 token 失败: {exc}") from exc
        token = str(data.get("access_token", ""))
        if not token:
            raise WechatContentSecurityError(f"获取微信内容安全 token 失败: {data}")
        expires_in = int(data.get("expires_in", 7200))
        wechat_access_token_cache["token"] = token
        wechat_access_token_cache["expires_at"] = now + max(300, expires_in)
        return token

    def _wechat_jscode2session(code: str) -> Dict[str, Any]:
        raw_code = str(code or "").strip()
        if not raw_code:
            raise HTTPException(status_code=400, detail="missing code")
        if not wechat_appid or not wechat_appsecret:
            raise HTTPException(status_code=503, detail="微信登录未配置")
        params = urllib.parse.urlencode(
            {
                "appid": wechat_appid,
                "secret": wechat_appsecret,
                "js_code": raw_code,
                "grant_type": "authorization_code",
            }
        )
        url = f"https://api.weixin.qq.com/sns/jscode2session?{params}"
        try:
            with urllib.request.urlopen(url, timeout=wechat_security_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="微信登录暂不可用") from exc
        openid = str(data.get("openid", "")).strip()
        if not openid:
            raise HTTPException(status_code=400, detail="微信登录失败")
        return {"openid": openid}

    def _wechat_openid_from_request(request: Request | None) -> str:
        if request is None:
            return ""
        return str(request.headers.get("X-WeChat-Openid", "") or request.headers.get("X-WECHAT-OPENID", "")).strip()

    def _prepare_wechat_sec_image_bytes(image_bytes: bytes) -> bytes:
        try:
            with Image.open(io.BytesIO(image_bytes)) as im0:
                im = ImageOps.exif_transpose(im0).convert("RGB")
                im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=85, optimize=True)
                return buf.getvalue()
        except Exception:
            return image_bytes

    def _wechat_img_sec_check(image_bytes: bytes, filename: str) -> None:
        if not wechat_content_security_enabled:
            return
        check_bytes = _prepare_wechat_sec_image_bytes(image_bytes)
        boundary = f"----searchstyle{uuid.uuid4().hex}"
        safe_name = f"{Path(filename or 'upload.jpg').stem or 'upload'}.jpg"
        head = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="media"; filename="{safe_name}"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8")
        tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
        body = head + check_bytes + tail
        try:
            token = _wechat_get_access_token()
            url = f"https://api.weixin.qq.com/wxa/img_sec_check?access_token={urllib.parse.quote(token)}"
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Content-Length": str(len(body)),
                },
            )
            with urllib.request.urlopen(req, timeout=wechat_security_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise WechatContentSecurityError(f"微信图片内容安全接口失败: HTTP {exc.code} {raw}") from exc
        except Exception as exc:
            raise WechatContentSecurityError(f"微信图片内容安全接口失败: {exc}") from exc
        errcode = int(data.get("errcode", -1))
        if errcode == 0:
            return
        if errcode in {87014, 87015}:
            raise HTTPException(status_code=400, detail="内容含违规信息，请修改后再试")
        raise WechatContentSecurityError(f"微信图片内容安全接口返回异常: {data}")

    def _wechat_msg_sec_check(text: str, openid: str = "") -> None:
        if not wechat_content_security_enabled:
            return
        content = str(text or "").strip()
        if not content:
            return
        payload: Dict[str, Any] = {"content": content[:2500]}
        security_openid = str(openid or wechat_security_openid or "").strip()
        if security_openid:
            payload.update(
                {
                    "version": 2,
                    "scene": wechat_security_scene,
                    "openid": security_openid,
                }
            )
        try:
            token = _wechat_get_access_token()
            url = f"https://api.weixin.qq.com/wxa/msg_sec_check?access_token={urllib.parse.quote(token)}"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=wechat_security_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise WechatContentSecurityError(f"微信文本内容安全接口失败: HTTP {exc.code} {raw}") from exc
        except Exception as exc:
            raise WechatContentSecurityError(f"微信文本内容安全接口失败: {exc}") from exc
        errcode = int(data.get("errcode", -1))
        result = data.get("result") if isinstance(data.get("result"), dict) else {}
        suggest = str(result.get("suggest", "")).lower()
        if errcode == 0 and (not suggest or suggest == "pass"):
            return
        if errcode == 87014 or suggest in {"risky", "review"}:
            raise HTTPException(status_code=400, detail="内容含违规信息，请修改后再试")
        raise WechatContentSecurityError(f"微信文本内容安全接口返回异常: {data}")

    def _check_search_upload_content_security(image_bytes: bytes, filename: str) -> None:
        if not wechat_content_security_enabled:
            return
        try:
            _wechat_img_sec_check(image_bytes, filename)
        except HTTPException:
            raise
        except WechatContentSecurityError as exc:
            logging.warning("wechat content security check failed filename=%s error=%s", filename, exc)
            if not wechat_security_fail_open:
                raise HTTPException(status_code=503, detail="内容安全校验暂不可用，请稍后再试") from exc

    def _check_text_content_security(*values: Any, openid: str = "") -> None:
        if not wechat_content_security_enabled:
            return
        texts = [str(value or "").strip() for value in values if str(value or "").strip()]
        if not texts:
            return
        try:
            for text in texts:
                _wechat_msg_sec_check(text, openid=openid)
        except HTTPException:
            raise
        except WechatContentSecurityError as exc:
            logging.warning("wechat text content security check failed error=%s", exc)
            if not wechat_security_fail_open:
                raise HTTPException(status_code=503, detail="内容安全校验暂不可用，请稍后再试") from exc

    def _list_import_source_images(source_dir: Path) -> List[Path]:
        return [
            path for path in sorted(source_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in allowed_image_exts
        ]

    def _resolve_catalog_import_source_dir(raw: str) -> Path:
        value = os.path.expandvars(str(raw or "").strip())
        return Path(value).expanduser()

    def _sanitize_import_filename(filename: str, fallback_suffix: str) -> str:
        raw = Path(str(filename or "").strip()).name
        stem = Path(raw).stem.strip()
        suffix = Path(raw).suffix.lower() or fallback_suffix.lower()
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
        if not stem:
            raise ValueError("filename is empty")
        if suffix not in allowed_image_exts:
            raise ValueError(f"unsupported image suffix: {suffix}")
        return f"{stem}{suffix}"

    def _sanitize_upload_filename(filename: str, fallback_name: str) -> str:
        raw = Path(str(filename or "").replace("\\", "/").split("/")[-1].strip() or fallback_name).name
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(raw).stem).strip("_") or Path(fallback_name).stem
        suffix = Path(raw).suffix.lower()
        if suffix not in allowed_image_exts:
            raise ValueError(f"unsupported image suffix: {suffix or '(none)'}")
        return f"{stem}{suffix}"

    def _derive_year_tag_from_style_code(style_code: str) -> str:
        return derive_year_from_style_code(style_code)

    def _sanitize_year_tag(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        if re.fullmatch(r"20\d{2}", raw):
            return raw
        raise ValueError("year_tag must be YYYY, e.g. 2024")

    def _normalize_import_tags(tags: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for tag in tags or []:
            clean = str(tag or "").strip()
            if not clean:
                continue
            lower = clean.casefold()
            if lower in seen:
                continue
            seen.add(lower)
            out.append(clean)
        return out

    def _is_valid_import_style_code(style_code: str) -> bool:
        code = str(style_code or "").strip()
        return bool(code) and bool(re.match(r"^[A-Za-z]", code))

    def _next_import_filename(prefix: str, suffix: str, used_names: set[str], next_seq: Dict[str, int]) -> str:
        clean_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(prefix or "").strip()).strip("_") or "UNKNOWN"
        seq = int(next_seq.get(clean_prefix, 0))
        while True:
            candidate = f"{clean_prefix}_{seq:03d}{suffix.lower()}"
            if candidate.lower() not in used_names:
                used_names.add(candidate.lower())
                next_seq[clean_prefix] = seq + 1
                return candidate
            seq += 1

    def _build_import_name_allocator() -> tuple[set[str], Dict[str, int]]:
        used_names = {
            path.name.lower()
            for path in standard_dir.glob("*")
            if path.is_file() and path.suffix.lower() in allowed_image_exts
        }
        next_seq: Dict[str, int] = {}
        for path in standard_dir.glob("*"):
            if not path.is_file() or path.suffix.lower() not in allowed_image_exts:
                continue
            stem = path.stem
            if "_" not in stem:
                continue
            prefix, suffix_num = stem.rsplit("_", 1)
            if suffix_num.isdigit():
                next_seq[prefix] = max(int(suffix_num) + 1, int(next_seq.get(prefix, 0)))
        return used_names, next_seq

    def _run_catalog_import_prepare(job_id: str, source_dir: Path) -> None:
        try:
            files = _list_import_source_images(source_dir)
            used_names, next_seq = _build_import_name_allocator()
            total = len(files)
            with catalog_import_lock:
                job = catalog_import_jobs.get(job_id)
                if job is None:
                    return
                job["total"] = total
                job["status"] = "running"
                job["message"] = f"发现 {total} 张图片"

            results: List[Dict[str, Any]] = []
            for index, path in enumerate(files, start=1):
                rel_path = str(path.relative_to(source_dir)).replace("\\", "/")
                code = ""
                error = ""
                for crop in build_header_crops(path):
                    code = str(try_extract_code_from_image(crop, tesseract_bin) or "").strip()
                    if code:
                        break
                style_code = code[:-1] if code.endswith("#") else code
                is_valid_code = _is_valid_import_style_code(style_code)
                if not code:
                    error = "OCR 未识别到款号"
                elif not is_valid_code:
                    error = "识别款号必须以字母开头"
                prefix = code_to_filename_prefix(code) if code else re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "UNKNOWN"
                proposed_filename = _next_import_filename(prefix, path.suffix.lower(), used_names, next_seq)
                results.append(
                    {
                        "source_rel_path": rel_path,
                        "source_name": path.name,
                        "proposed_style_code": style_code,
                        "proposed_year_tag": _derive_year_tag_from_style_code(style_code),
                        "proposed_filename": proposed_filename,
                        "tags": [],
                        "status": "ok" if (code and is_valid_code) else ("invalid_style_code" if code else "ocr_failed"),
                        "error": error,
                    }
                )
                with catalog_import_lock:
                    job = catalog_import_jobs.get(job_id)
                    if job is None:
                        return
                    job["processed"] = index
                    job["items"] = list(results)
                    job["message"] = f"已处理 {index}/{total}"

            with catalog_import_lock:
                job = catalog_import_jobs.get(job_id)
                if job is None:
                    return
                job["status"] = "completed"
                job["message"] = f"预处理完成，共 {total} 张"
        except Exception as exc:
            logging.exception("catalog import prepare failed: %s", exc)
            with catalog_import_lock:
                job = catalog_import_jobs.get(job_id)
                if job is not None:
                    job["status"] = "failed"
                    job["message"] = str(exc)

    def _create_catalog_import_job(source_dir: Path, source_type: str = "server_dir") -> Dict[str, Any]:
        files = _list_import_source_images(source_dir)
        if not files:
            raise ValueError("source_dir has no supported images")
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "source_dir": str(source_dir),
            "source_type": source_type,
            "status": "pending",
            "message": "任务已创建",
            "total": 0,
            "processed": 0,
            "items": [],
            "committed": False,
        }
        with catalog_import_lock:
            catalog_import_jobs[job_id] = job
        thread = threading.Thread(target=_run_catalog_import_prepare, args=(job_id, source_dir), daemon=True)
        thread.start()
        return job

    def _serialize_catalog_import_job(job: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "job_id": str(job.get("job_id", "")),
            "source_dir": str(job.get("source_dir", "")),
            "source_type": str(job.get("source_type", "")),
            "status": str(job.get("status", "pending")),
            "message": str(job.get("message", "")),
            "total": int(job.get("total", 0)),
            "processed": int(job.get("processed", 0)),
            "items": list(job.get("items", [])),
            "committed": bool(job.get("committed", False)),
        }

    def _catalog_import_job_item(job: Dict[str, Any], source_rel_path: str) -> Dict[str, Any] | None:
        target = str(source_rel_path or "").strip()
        if not target:
            return None
        for item in job.get("items", []):
            if str(item.get("source_rel_path", "")).strip() == target:
                return item
        return None

    def _catalog_session_sign(username: str, exp_ts: int) -> str:
        payload = f"{username}:{exp_ts}".encode("utf-8")
        return hmac.new(catalog_web_session_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    def _catalog_build_session_value(username: str) -> tuple[str, int]:
        exp_ts = int(time.time()) + max(300, catalog_web_session_ttl_sec)
        sig = _catalog_session_sign(username, exp_ts)
        return f"{username}:{exp_ts}:{sig}", exp_ts

    def _catalog_token_cache_key(token: str) -> str:
        clean = str(token or "").strip()
        return hashlib.sha256(clean.encode("utf-8")).hexdigest()

    def _catalog_permissions_from_payload(payload: Dict[str, Any]) -> List[str]:
        permissions: List[str] = []
        for field in catalog_external_permission_fields:
            raw_permissions = payload.get(field)
            if raw_permissions:
                break
        else:
            raw_permissions = []
        if isinstance(raw_permissions, str):
            raw_permissions = re.split(r"[\s,]+", raw_permissions)
        if isinstance(raw_permissions, list):
            permissions = [str(item).strip() for item in raw_permissions if str(item).strip()]
        inferred: List[str] = []

        def add_perm(value: str) -> None:
            if value and value not in inferred:
                inferred.append(value)

        def walk_menus(rows: Any) -> None:
            if not isinstance(rows, list):
                return
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name", "") or (row.get("data") or {}).get("name", "")).strip()
                permission = str((row.get("data") or {}).get("permission", "") or row.get("permission", "")).strip()
                if name == "产品库" or permission in {"product:view", "catalog:product:view"}:
                    add_perm("product:view")
                if name == "色卡库" or permission in {"color:view", "color-card:view"}:
                    add_perm("color:view")
                walk_menus(row.get("children"))

        walk_menus(payload.get("menus"))
        known = {"*", "product:view", "product:create", "color:view", "color:create"}
        normalized = [item for item in permissions if item in known]
        if normalized:
            for item in inferred:
                if item not in normalized:
                    normalized.append(item)
            return normalized
        if permissions:
            return list(catalog_external_default_permissions)
        for item in inferred:
            if item not in normalized:
                normalized.append(item)
        return normalized or list(catalog_external_default_permissions)

    def _catalog_jwt_payload(token: str) -> Dict[str, Any]:
        clean = str(token or "").strip()
        parts = clean.split(".")
        if len(parts) >= 2:
            payload_raw = parts[1]
            payload_raw += "=" * (-len(payload_raw) % 4)
            try:
                payload = json.loads(base64.urlsafe_b64decode(payload_raw.encode("utf-8")).decode("utf-8"))
                if isinstance(payload, dict):
                    return payload
            except Exception:
                return {}
        return {}

    def _catalog_token_user_from_payload(payload: Dict[str, Any], token: str) -> str:
        for field in catalog_external_user_fields:
            value = str(payload.get(field, "")).strip()
            if value:
                return value
        return f"external_{_catalog_token_cache_key(token)[:16]}"

    def _catalog_normalize_verify_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        data = payload.get("data")
        if isinstance(data, dict):
            merged = dict(payload)
            merged.update(data)
            return merged
        return payload

    def _catalog_verify_token_remote(token: str) -> Dict[str, Any] | None:
        if not catalog_external_verify_url:
            return None
        method = catalog_external_verify_method if catalog_external_verify_method in {"GET", "POST"} else "POST"
        body = None if method == "GET" else json.dumps({"token": token}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            catalog_external_verify_url,
            data=body,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "X-Catalog-Token": token,
            },
        )
        with urllib.request.urlopen(req, timeout=max(0.5, catalog_external_verify_timeout_sec)) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
        if not isinstance(data, dict):
            return None
        data = _catalog_normalize_verify_payload(data)
        valid_value = data.get("valid", data.get("active", data.get("ok", data.get("success", True))))
        if "succeed" in data:
            valid_value = data.get("succeed")
        if "code" in data and str(data.get("code")) not in {"0", "200"}:
            return None
        if valid_value is False or str(valid_value).strip().lower() in {"0", "false", "no", "invalid", "expired"}:
            return None
        user = _catalog_token_user_from_payload(data, token)
        return {
            "user": user,
            "permissions": _catalog_permissions_from_payload(data),
            "source": "verify_url",
        }

    def _catalog_verify_token(token: str) -> Dict[str, Any] | None:
        clean = str(token or "").strip()
        if not catalog_external_token_enabled or len(clean) < 8:
            return None
        cache_key = _catalog_token_cache_key(clean)
        now = time.time()
        with catalog_external_token_cache_lock:
            cached = catalog_external_token_cache.get(cache_key)
            if cached and float(cached.get("expires_at", 0.0)) > now:
                return dict(cached.get("result") or {}) or None

        result: Dict[str, Any] | None = None
        if catalog_external_verify_url:
            try:
                result = _catalog_verify_token_remote(clean)
            except Exception as exc:
                logging.warning("catalog external token verify failed: %s", exc)
                if not catalog_external_fail_open:
                    result = None
                else:
                    payload = _catalog_jwt_payload(clean)
                    result = {
                        "user": _catalog_token_user_from_payload(payload, clean),
                        "permissions": _catalog_permissions_from_payload(payload),
                        "source": "fail_open",
                    }

        if result is None and (clean in catalog_external_allowed_tokens or catalog_external_allow_unverified_tokens):
            payload = _catalog_jwt_payload(clean)
            result = {
                "user": _catalog_token_user_from_payload(payload, clean),
                "permissions": _catalog_permissions_from_payload(payload),
                "source": "local",
            }

        if result is not None and catalog_external_cache_ttl_sec > 0:
            with catalog_external_token_cache_lock:
                catalog_external_token_cache[cache_key] = {
                    "expires_at": now + max(1, catalog_external_cache_ttl_sec),
                    "result": dict(result),
                }
        return result

    def _catalog_read_session_user(request: Request) -> str:
        if not catalog_web_auth_enabled:
            return ""
        raw = str(request.cookies.get(catalog_web_cookie_name, "")).strip()
        if not raw:
            return ""
        parts = raw.split(":", 2)
        if len(parts) != 3:
            return ""
        username, exp_raw, sig = parts
        if not username or not exp_raw.isdigit() or not sig:
            return ""
        exp_ts = int(exp_raw)
        if exp_ts < int(time.time()):
            return ""
        expected = _catalog_session_sign(username, exp_ts)
        if not hmac.compare_digest(expected, sig):
            return ""
        if username.startswith("external_"):
            return ""
        return username

    def _catalog_request_token(request: Request) -> str:
        token = str(request.query_params.get("token", "")).strip()
        if token:
            return token
        token = str(request.query_params.get("access_token", "")).strip()
        if token:
            return token
        token = str(request.headers.get("X-Catalog-Token", "")).strip()
        if token:
            return token
        auth = str(request.headers.get("Authorization", "")).strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return ""

    def _catalog_has_permission(request: Request, permission: str) -> bool:
        permissions = getattr(request.state, "catalog_permissions", None)
        if permissions is None:
            return True
        return "*" in permissions or permission in permissions

    def _catalog_require_permission(request: Request, permission: str) -> None:
        if not _catalog_has_permission(request, permission):
            raise HTTPException(status_code=403, detail=f"missing permission: {permission}")

    def _catalog_is_login_ok(username: str, password: str) -> bool:
        user = username.strip()
        expected = catalog_web_users.get(user)
        return bool(expected) and hmac.compare_digest(password, expected)

    def _catalog_today_captcha() -> str:
        weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
        try:
            now = dt.datetime.now(ZoneInfo(catalog_web_captcha_timezone))
        except Exception:
            now = dt.datetime.now()
        return f"{now:%Y%m%d}{weekday_map[now.weekday()]}"

    def _catalog_is_captcha_ok(captcha: str) -> bool:
        if not catalog_web_captcha_enabled:
            return True
        actual = str(captcha or "").strip()
        expected = _catalog_today_captcha()
        return bool(actual) and hmac.compare_digest(actual.encode("utf-8"), expected.encode("utf-8"))

    def _save_debug_query_image(request: Request, query_path: Path, original_name: str) -> Path | None:
        if not debug_query_enabled:
            return None
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(original_name or "query").stem).strip("._") or "query"
        user = str(getattr(request.state, "api_user", "anonymous")).replace(":", "_")
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = debug_query_dir / f"{stamp}__{user}__{safe_name}.jpg"
        try:
            with Image.open(query_path) as im0:
                im = im0.convert("RGB")
                im.save(out_path, format="JPEG", quality=92)
            return out_path
        except Exception:
            return None

    @app.middleware("http")
    async def check_api_key(request: Request, call_next):
        t0 = time.perf_counter()
        client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
            request.client.host if request.client else "-"
        )
        req_len = request.headers.get("content-length", "-")
        ua = request.headers.get("user-agent", "-")

        path = request.url.path
        is_catalog_ui = path == "/catalog"
        is_catalog_mobile_alias = path in {"/product", "/color"}
        is_catalog_login = path == "/catalog/login"
        is_catalog_logout = path == "/catalog/logout"
        is_catalog_api = path.startswith("/api/v1/catalog/")
        is_color_card_api = path.startswith("/api/v1/color-card/")
        is_catalog_route = is_catalog_ui or is_catalog_mobile_alias or is_catalog_login or is_catalog_logout or is_catalog_api or is_color_card_api
        allow_public = (
            path in {"/health", "/ready"}
            or is_catalog_login
            or is_catalog_logout
            or ((not catalog_web_auth_enabled) and is_catalog_ui)
            or ((not catalog_web_auth_enabled) and is_catalog_mobile_alias)
            or ((not catalog_web_auth_enabled) and catalog_public and (is_catalog_api or is_color_card_api))
            or path.startswith("/print-static/")
            or path.startswith("/print-storage/")
            or path.startswith("/recolor-static/")
        )
        allow_api = (
            path in {"/search", "/image-url", "/api/v1/image-url", "/api/v1/wechat/session", "/api/v1/templates", "/api/v1/render", "/api/v1/images/upload", "/recolor", "/recolor-ai"}
            or is_catalog_route
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

        if is_catalog_route and catalog_web_auth_enabled and not is_catalog_login:
            key = request.headers.get("X-API-Key", "").strip() if api_key_enabled else ""
            api_user = api_key_map.get(key, "") if key else ""
            catalog_token = _catalog_request_token(request)
            token_auth = _catalog_verify_token(catalog_token)
            if api_user:
                request.state.api_user = api_user
                request.state.catalog_permissions = None
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
            if token_auth:
                request.state.catalog_user_id = str(token_auth.get("user", "")).strip()
                request.state.api_user = f"catalog-token:{token_auth.get('user', 'unknown')}"
                request.state.catalog_permissions = set(token_auth.get("permissions") or [])
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
            if catalog_token and not token_auth:
                resp = JSONResponse(status_code=401, content={"detail": "invalid catalog token"})
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
            web_user = _catalog_read_session_user(request)
            if web_user:
                request.state.api_user = f"catalog-web:{web_user}"
                request.state.catalog_permissions = None
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
            if is_catalog_ui or is_catalog_logout:
                resp = RedirectResponse(url="/catalog/login", status_code=303)
            else:
                resp = JSONResponse(status_code=401, content={"detail": "catalog login required"})
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
            if path == "/search":
                catalog_token = _catalog_request_token(request)
                token_auth = _catalog_verify_token(catalog_token)
                if token_auth:
                    request.state.catalog_user_id = str(token_auth.get("user", "")).strip()
                    request.state.api_user = f"catalog-token:{token_auth.get('user', 'unknown')}"
                    request.state.catalog_permissions = set(token_auth.get("permissions") or [])
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
                if catalog_token and not token_auth:
                    resp = JSONResponse(status_code=401, content={"detail": "invalid catalog token"})
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

                catalog_token = _catalog_request_token(request)
                token_auth = _catalog_verify_token(catalog_token)
                if token_auth:
                    permissions = set(token_auth.get("permissions") or [])
                    if "*" in permissions or "product:view" in permissions:
                        request.state.catalog_user_id = str(token_auth.get("user", "")).strip()
                        request.state.api_user = f"catalog-token:{token_auth.get('user', 'unknown')}"
                        request.state.catalog_permissions = permissions
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
                    resp = JSONResponse(status_code=403, content={"detail": "missing permission: product:view"})
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
                if catalog_token and not token_auth:
                    resp = JSONResponse(status_code=401, content={"detail": "invalid catalog token"})
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
                if path == "/search" and (not catalog_web_auth_enabled) and catalog_public:
                    request.state.api_user = "catalog-public-search"
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

    def _build_image_url_with_preview(base_url: str, image_name: str, max_edge: int, quality: int) -> str:
        safe = Path(image_name).name
        if not safe:
            return f"{base_url}/images/"
        query_parts: List[str] = []
        if max_edge > 0:
            query_parts.append(f"max_edge={max(128, min(2048, int(max_edge)))}")
            query_parts.append(f"q={max(40, min(95, int(quality)))}")
        if api_key_enabled and image_url_secret:
            exp_ts = int(time.time()) + max(60, image_url_ttl_sec)
            msg = f"{safe}:{exp_ts}".encode("utf-8")
            sig = hmac.new(image_url_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
            query_parts.extend([f"exp={exp_ts}", f"sig={sig}"])
        qs = f"?{'&'.join(query_parts)}" if query_parts else ""
        return f"{base_url}/images/{safe}{qs}"

    def _build_image_url(base_url: str, image_name: str) -> str:
        return _build_image_url_with_preview(base_url, image_name, result_image_max_edge, result_image_quality)

    def _build_catalog_image_url(base_url: str, image_name: str) -> str:
        return _build_image_url_with_preview(base_url, image_name, catalog_image_max_edge, catalog_image_quality)

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

    def _safe_catalog_part(value: str, fallback: str = "item", max_len: int = 36) -> str:
        clean = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", str(value or "").strip()).strip("-_")
        return (clean or fallback)[:max_len]

    def _owner_tag_from_request(request: Request, user_tag: str = "") -> str:
        incoming = str(user_tag or "").strip()
        if incoming.startswith("owner:") and len(incoming) > len("owner:"):
            return incoming
        user_id_local = str(getattr(request.state, "catalog_user_id", "") or "").strip()
        if user_id_local:
            return f"owner:{user_id_local}"
        return ""

    def _personal_folder_tag(owner_tag: str, folder_name: str) -> str:
        folder = str(folder_name or "").strip()
        if not owner_tag or not folder:
            return ""
        return f"{owner_tag}:folder:{folder}"

    def _image_b64(image_name: str) -> tuple[str, str]:
        safe = Path(image_name).name
        fp = standard_dir / safe
        if not fp.exists() or not fp.is_file():
            return "", ""
        raw = fp.read_bytes()
        return base64.b64encode(raw).decode("ascii"), _guess_mime(safe)

    def _serialize_catalog_product(base_url: str, product: Dict[str, Any]) -> Dict[str, Any]:
        images = [
            {
                "image_name": str(item.get("image_name", "")),
                "sort_order": int(item.get("sort_order", 0)),
                "image_url": _build_catalog_image_url(base_url, str(item.get("image_name", ""))),
            }
            for item in list(product.get("images", []))
        ]
        cover_image = str(product.get("cover_image", "")).strip()
        return {
            "style_code": str(product.get("style_code", "")),
            "cover_image": cover_image,
            "cover_image_url": _build_catalog_image_url(base_url, cover_image) if cover_image else "",
            "note": str(product.get("note", "")),
            "tags": list(product.get("tags", [])),
            "raw_tags": list(product.get("raw_tags", [])),
            "tag_groups": dict(product.get("tag_groups", {}) or {}),
            "images": images,
            "image_count": int(product.get("image_count", len(images)) or 0),
            "created_at": str(product.get("created_at", "")),
            "updated_at": str(product.get("updated_at", "")),
        }

    def _enrich_search_rows(base_url: str, rows_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        style_codes = [str(row.get("style_code", "")).strip() for row in rows_in if str(row.get("style_code", "")).strip()]
        product_map = {
            str(item.get("style_code", "")): item
            for item in catalog_store.get_products_by_codes(style_codes)
        }
        for row in rows_in:
            style_code = str(row.get("style_code", "")).strip()
            product = product_map.get(style_code)
            if not product:
                row["tags"] = []
                row["catalog_cover_image"] = row.get("best_standard_image", "")
                row["catalog_cover_image_url"] = row.get("best_standard_image_url", "")
                continue
            row["tags"] = list(product.get("tags", []))
            row["catalog_cover_image"] = str(product.get("cover_image", ""))
            row["catalog_cover_image_url"] = _build_catalog_image_url(base_url, str(product.get("cover_image", "")))
        return rows_in

    def _enrich_similar_images(base_url: str, rows_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        style_codes = [str(row.get("style_code", "")).strip() for row in rows_in if str(row.get("style_code", "")).strip()]
        product_map = {
            str(item.get("style_code", "")): item
            for item in catalog_store.get_products_by_codes(style_codes)
        }
        for row in rows_in:
            style_code = str(row.get("style_code", "")).strip()
            product = product_map.get(style_code)
            row["tags"] = list(product.get("tags", [])) if product else []
        return rows_in

    def _dedupe_search_rows(rows_in: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen_codes: set[str] = set()
        for row in rows_in:
            code = str(row.get("style_code", "")).strip().upper()
            if code.endswith("#"):
                code = code[:-1]
            image_name = str(row.get("best_standard_image", "")).strip()
            if code.startswith("MY-") or Path(image_name).name.upper().startswith("MY-"):
                continue
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            out.append(row)
            if len(out) >= max(1, int(limit)):
                break
        return out

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

    def _extract_fg_mask_vec(path: Path, size: int = 64) -> np.ndarray | None:
        try:
            with Image.open(path) as im0:
                rgb = np.asarray(im0.convert("RGB"), dtype=np.uint8)
        except Exception:
            return None
        fg = np.any(rgb < 240, axis=-1).astype(np.float32)
        cut = min(int(fg.shape[0] * 0.12), 120)
        fg[:cut, :] = 0.0
        if float(fg.sum()) < 16:
            return None
        im = Image.fromarray((fg * 255.0).astype(np.uint8), mode="L").resize((size, size), Image.BILINEAR)
        v = (np.asarray(im, dtype=np.float32) / 255.0).reshape(-1)
        n = float(np.linalg.norm(v)) + 1e-8
        return (v / n).astype(np.float32)

    def _extract_stripe_sig(path: Path, keep: int = 24) -> np.ndarray | None:
        try:
            with Image.open(path) as im0:
                rgb = np.asarray(im0.convert("RGB").resize((192, 192), Image.Resampling.BILINEAR), dtype=np.uint8)
        except Exception:
            return None
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        fg = np.any(rgb < 240, axis=-1).astype(np.float32)
        cut = min(int(fg.shape[0] * 0.12), 24)
        fg[:cut, :] = 0.0
        if float(fg.sum()) < 64:
            fg = np.ones_like(fg, dtype=np.float32)
        proj_y = (gray * fg).sum(axis=1) / np.clip(fg.sum(axis=1), 1.0, None)
        proj_x = (gray * fg).sum(axis=0) / np.clip(fg.sum(axis=0), 1.0, None)

        def _fft_mag(sig: np.ndarray, k: int) -> np.ndarray:
            s = sig.astype(np.float32)
            s = s - s.mean()
            spec = np.abs(np.fft.rfft(s))[1 : 1 + k]
            if spec.shape[0] < k:
                spec = np.pad(spec, (0, k - spec.shape[0]))
            spec = spec.astype(np.float32)
            n = float(np.linalg.norm(spec)) + 1e-8
            return spec / n

        fy = _fft_mag(proj_y, keep)
        fx = _fft_mag(proj_x, keep)
        return np.concatenate([fy, fx]).astype(np.float32)

    def _extract_color_sig(path: Path) -> np.ndarray | None:
        try:
            with Image.open(path) as im0:
                feat = extract_garment_color_feature(im0.convert("RGB")).astype(np.float32)
        except Exception:
            return None
        norm = float(np.linalg.norm(feat)) + 1e-8
        return (feat / norm).astype(np.float32)

    def _extract_pattern_sig(path: Path, size: int = 14) -> np.ndarray | None:
        try:
            with Image.open(path) as im0:
                rgb = np.asarray(im0.convert("RGB"), dtype=np.uint8)
        except Exception:
            return None
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        h, w = gray.shape
        if h < 32 or w < 32:
            return None

        fg = np.any(rgb < 240, axis=-1)
        cut = min(int(h * 0.12), 120)
        fg[:cut, :] = False
        ys, xs = np.where(fg)
        if ys.size >= 32:
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            bbox = gray[y0 : y1 + 1, x0 : x1 + 1]
        else:
            bbox = gray

        ch = max(24, int(h * 0.72))
        cw = max(24, int(w * 0.72))
        cy0 = max(0, (h - ch) // 2)
        cx0 = max(0, (w - cw) // 2)
        center = gray[cy0 : cy0 + ch, cx0 : cx0 + cw]

        def _norm_patch(arr: np.ndarray) -> np.ndarray:
            patch = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").resize((size, size), Image.BILINEAR)
            v = np.asarray(patch, dtype=np.float32)
            v = (v - v.mean()) / (v.std() + 1e-6)
            return v.reshape(-1)

        v1 = _norm_patch(bbox)
        v2 = _norm_patch(center)
        v = np.concatenate([v1, v2]).astype(np.float32)
        n = float(np.linalg.norm(v)) + 1e-8
        return (v / n).astype(np.float32)

    def _extract_checker_profile(path: Path, grid: int = 10) -> Dict[str, float] | None:
        try:
            with Image.open(path) as im0:
                rgb = np.asarray(im0.convert("RGB"), dtype=np.uint8)
        except Exception:
            return None
        if rgb.ndim != 3 or rgb.shape[0] < 32 or rgb.shape[1] < 32:
            return None
        h, w = rgb.shape[:2]
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)

        def _component_count(mask: np.ndarray) -> int:
            seen = np.zeros(mask.shape, dtype=bool)
            count = 0
            hh, ww = mask.shape
            for yy in range(hh):
                for xx in range(ww):
                    if seen[yy, xx] or not bool(mask[yy, xx]):
                        continue
                    count += 1
                    stack = [(yy, xx)]
                    seen[yy, xx] = True
                    while stack:
                        cy, cx = stack.pop()
                        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                            if ny < 0 or ny >= hh or nx < 0 or nx >= ww:
                                continue
                            if seen[ny, nx] or not bool(mask[ny, nx]):
                                continue
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            return count

        def _score_crop(crop_rgb: np.ndarray) -> Dict[str, float] | None:
            if crop_rgb.ndim != 3 or crop_rgb.shape[0] < 24 or crop_rgb.shape[1] < 24:
                return None
            patch = Image.fromarray(np.clip(crop_rgb, 0, 255).astype(np.uint8), mode="RGB").resize((grid, grid), Image.BILINEAR)
            cell_rgb = np.asarray(patch, dtype=np.float32)
            arr = (0.299 * cell_rgb[..., 0] + 0.587 * cell_rgb[..., 1] + 0.114 * cell_rgb[..., 2]).astype(np.float32)
            chroma = (cell_rgb.max(axis=-1) - cell_rgb.min(axis=-1)).astype(np.float32)
            contrast = min(1.0, float(arr.std()) / 64.0)
            if contrast < 0.08:
                return {"checker": 0.0, "stripe": 0.0, "contrast": contrast, "bw_mix": 0.0}

            dark_cut = min(120.0, float(np.quantile(arr, 0.35)))
            light_cut = max(112.0, float(np.quantile(arr, 0.65)))
            dark_cells = arr <= dark_cut
            light_cells = (arr >= light_cut) & (chroma <= 48.0)
            dark_ratio = float(np.mean(dark_cells))
            light_ratio = float(np.mean(light_cells))
            neutral_light_ratio = float(np.mean(chroma[arr >= light_cut] <= 48.0)) if np.any(arr >= light_cut) else 0.0
            bw_mix = min(1.0, 2.0 * min(dark_ratio, light_ratio))
            if dark_ratio < 0.12 or light_ratio < 0.12 or neutral_light_ratio < 0.35:
                return {
                    "checker": 0.0,
                    "stripe": 0.0,
                    "contrast": float(contrast),
                    "bw_mix": float(bw_mix),
                    "dark_ratio": float(dark_ratio),
                    "light_ratio": float(light_ratio),
                    "neutral_light_ratio": float(neutral_light_ratio),
                }

            med = float(np.median(arr))
            bits = arr > med
            alt_x = float(np.mean(bits[:, 1:] != bits[:, :-1])) if grid > 1 else 0.0
            alt_y = float(np.mean(bits[1:, :] != bits[:-1, :])) if grid > 1 else 0.0
            dark_components = _component_count(dark_cells)
            light_components = _component_count(light_cells)
            component_factor = min(1.0, dark_components / 3.0) * min(1.0, light_components / 2.0)
            if dark_components < 2 or light_components < 2:
                component_factor = 0.0

            checker = min(alt_x, alt_y) * contrast * bw_mix * component_factor
            stripe = max(0.0, abs(alt_x - alt_y)) * contrast * bw_mix
            return {
                "checker": float(checker),
                "stripe": float(stripe),
                "contrast": float(contrast),
                "bw_mix": float(bw_mix),
                "dark_ratio": float(dark_ratio),
                "light_ratio": float(light_ratio),
                "dark_components": float(dark_components),
                "light_components": float(light_components),
                "component_factor": float(component_factor),
                "alt_x": float(alt_x),
                "alt_y": float(alt_y),
            }

        windows: List[tuple[int, int, int, int]] = []
        windows.append((int(w * 0.12), int(h * 0.06), int(w * 0.88), int(h * 0.92)))
        windows.append((int(w * 0.20), int(h * 0.12), int(w * 0.80), int(h * 0.78)))

        central = gray[int(h * 0.06) : int(h * 0.92), int(w * 0.12) : int(w * 0.88)]
        dark = central < 105
        if int(dark.sum()) >= 24:
            ys, xs = np.where(dark)
            px = max(4, int((xs.max() - xs.min() + 1) * 0.15))
            py = max(4, int((ys.max() - ys.min() + 1) * 0.15))
            cx0 = int(w * 0.12)
            cy0 = int(h * 0.06)
            windows.append((
                max(0, cx0 + int(xs.min()) - px),
                max(0, cy0 + int(ys.min()) - py),
                min(w, cx0 + int(xs.max()) + px + 1),
                min(h, cy0 + int(ys.max()) + py + 1),
            ))

        win_w = max(48, int(w * 0.42))
        win_h = max(48, int(h * 0.42))
        for cy in (0.22, 0.38, 0.54):
            for cx in (0.28, 0.50, 0.72):
                x0 = max(0, min(w - win_w, int(w * cx - win_w / 2)))
                y0 = max(0, min(h - win_h, int(h * cy - win_h / 2)))
                windows.append((x0, y0, x0 + win_w, y0 + win_h))

        best: Dict[str, float] | None = None
        seen = set()
        for x0, y0, x1, y1 in windows:
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 - x0 < 24 or y1 - y0 < 24:
                continue
            key = (x0, y0, x1, y1)
            if key in seen:
                continue
            seen.add(key)
            prof = _score_crop(rgb[y0:y1, x0:x1])
            if prof is None:
                continue
            prof["window_area"] = float((x1 - x0) * (y1 - y0)) / float(max(1, w * h))
            if best is None or (
                float(prof.get("checker", 0.0)),
                float(prof.get("bw_mix", 0.0)),
                float(prof.get("contrast", 0.0)),
            ) > (
                float(best.get("checker", 0.0)),
                float(best.get("bw_mix", 0.0)),
                float(best.get("contrast", 0.0)),
            ):
                best = prof
        return best

    def _filter_accent_motif_components(mask: np.ndarray) -> np.ndarray:
        if mask.ndim != 2 or int(mask.sum()) <= 0:
            return mask
        h, w = mask.shape
        img_area = float(max(1, h * w))
        seen = np.zeros(mask.shape, dtype=bool)
        out = np.zeros(mask.shape, dtype=bool)
        min_comp = max(3, int(img_area * 0.00004))
        max_comp = max(32, int(img_area * 0.045))
        max_bbox_area = max(64, int(img_area * 0.16))
        max_fill = 0.72

        if cv2 is not None:
            n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 4)
            for label in range(1, int(n_labels)):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < min_comp:
                    continue
                bw = int(stats[label, cv2.CC_STAT_WIDTH])
                bh = int(stats[label, cv2.CC_STAT_HEIGHT])
                bbox_area = max(1, bw * bh)
                fill = area / float(bbox_area)
                if area > max_comp and (bbox_area > max_bbox_area or fill > max_fill):
                    continue
                out[labels == label] = True
            return out

        for sy, sx in zip(*np.where(mask & ~seen)):
            stack = [(int(sy), int(sx))]
            seen[sy, sx] = True
            pts: List[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if seen[ny, nx] or not bool(mask[ny, nx]):
                        continue
                    seen[ny, nx] = True
                    stack.append((ny, nx))

            area = len(pts)
            if area < min_comp:
                continue
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            bbox_area = max(1, (y1 - y0 + 1) * (x1 - x0 + 1))
            fill = area / float(bbox_area)
            # Large solid blocks are usually garment body colors or labels, not local motifs.
            if area > max_comp and (bbox_area > max_bbox_area or fill > max_fill):
                continue
            for py, px in pts:
                out[py, px] = True
        return out

    def _extract_accent_pattern_sig(path: Path, grid: int = 12) -> np.ndarray | None:
        try:
            with Image.open(path) as im0:
                im = im0.convert("RGB")
                if accent_pattern_max_edge > 0:
                    edge = max(128, int(accent_pattern_max_edge))
                    im.thumbnail((edge, edge), Image.Resampling.BILINEAR)
                rgb = np.asarray(im, dtype=np.uint8)
        except Exception:
            return None
        if rgb.ndim != 3 or rgb.shape[0] < 32 or rgb.shape[1] < 32:
            return None

        h, w = rgb.shape[:2]
        arr = rgb.astype(np.float32)
        mx = arr.max(axis=-1)
        mn = arr.min(axis=-1)
        chroma = mx - mn
        gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
        sat = chroma / np.clip(mx, 1.0, None)

        # Local colorful embroidery/printing is usually high-chroma and small-area.
        # Drop top labels and very bright/white backgrounds so the motif drives matching.
        valid = np.ones((h, w), dtype=bool)
        valid[: min(int(h * 0.10), 100), :] = False
        colorful = valid & (chroma >= 38.0) & (sat >= 0.22) & (gray >= 35.0) & (gray <= 235.0)
        colorful = _filter_accent_motif_components(colorful)
        if int(colorful.sum()) < max(8, accent_pattern_min_pixels):
            return None

        def _hue_diversity(mask: np.ndarray) -> float:
            if int(mask.sum()) < 8:
                return 0.0
            pix = rgb[mask].astype(np.uint8)
            if cv2 is not None:
                hsv = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                hue = hsv[:, 0].astype(np.float32) / 180.0
            else:
                vals = pix.astype(np.float32) / 255.0
                mxv = vals.max(axis=1)
                mnv = vals.min(axis=1)
                delta = np.clip(mxv - mnv, 1e-6, None)
                hue = np.zeros(vals.shape[0], dtype=np.float32)
                r, g, b = vals[:, 0], vals[:, 1], vals[:, 2]
                idx = mxv == r
                hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
                idx = mxv == g
                hue[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
                idx = mxv == b
                hue[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0
                hue = (hue / 6.0) % 1.0
            hist, _ = np.histogram(hue, bins=10, range=(0.0, 1.0))
            active = int(np.count_nonzero(hist >= max(3, int(mask.sum() * 0.04))))
            return min(1.0, active / 5.0)

        windows = [
            (0.00, 0.00, 1.00, 1.00, 0.60),
            (0.12, 0.12, 0.88, 0.92, 0.90),
            (0.18, 0.25, 0.82, 0.92, 1.15),
            (0.24, 0.32, 0.76, 0.92, 1.45),
            (0.30, 0.38, 0.72, 0.90, 1.70),
        ]
        best_mask = colorful
        best_score = -1.0
        local_min = max(8, min(accent_pattern_min_pixels, int(h * w * 0.0015)))
        for x0r, y0r, x1r, y1r, weight in windows:
            wx0, wy0 = int(w * x0r), int(h * y0r)
            wx1, wy1 = max(wx0 + 1, int(w * x1r)), max(wy0 + 1, int(h * y1r))
            win_mask = np.zeros_like(colorful, dtype=bool)
            win_mask[wy0:wy1, wx0:wx1] = colorful[wy0:wy1, wx0:wx1]
            count = int(win_mask.sum())
            if count < local_min:
                continue
            win_area = float(max(1, (wx1 - wx0) * (wy1 - wy0)))
            density = min(1.0, count / max(1.0, win_area * 0.08))
            diversity = _hue_diversity(win_mask)
            score = float(weight) * (0.55 + diversity) * (0.35 + density)
            if score > best_score:
                best_score = score
                best_mask = win_mask

        colorful = best_mask
        if int(colorful.sum()) < local_min:
            return None

        ys, xs = np.where(colorful)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)
        pad_x = max(8, int(bw * 0.35))
        pad_y = max(8, int(bh * 0.35))
        x0 = max(0, x0 - pad_x)
        x1 = min(w, x1 + pad_x + 1)
        y0 = max(0, y0 - pad_y)
        y1 = min(h, y1 + pad_y + 1)

        crop_rgb = rgb[y0:y1, x0:x1]
        crop_mask = colorful[y0:y1, x0:x1]
        if crop_rgb.shape[0] < 16 or crop_rgb.shape[1] < 16 or int(crop_mask.sum()) < max(8, accent_pattern_min_pixels):
            return None

        mask_img = Image.fromarray((crop_mask.astype(np.uint8) * 255), mode="L").resize((grid, grid), Image.BILINEAR)
        rgb_img = Image.fromarray(crop_rgb, mode="RGB").resize((grid, grid), Image.BILINEAR)
        mask_grid = np.asarray(mask_img, dtype=np.float32) / 255.0
        rgb_grid = np.asarray(rgb_img, dtype=np.float32) / 255.0

        weighted_rgb = (rgb_grid * mask_grid[..., None]).reshape(-1)
        mask_vec = mask_grid.reshape(-1)

        selected = crop_rgb[crop_mask]
        color_hist_parts: List[np.ndarray] = []
        if selected.shape[0] > 0:
            sel = selected.astype(np.float32) / 255.0
            for ci in range(3):
                hist, _ = np.histogram(sel[:, ci], bins=8, range=(0.0, 1.0))
                color_hist_parts.append(hist.astype(np.float32))
            local_selected = selected.astype(np.float32)
            local_chroma = (local_selected.max(axis=-1) - local_selected.min(axis=-1)) / 255.0
            chroma_hist, _ = np.histogram(local_chroma, bins=8, range=(0.0, 1.0))
            color_hist_parts.append(chroma_hist.astype(np.float32))
        hist_vec = np.concatenate(color_hist_parts).astype(np.float32) if color_hist_parts else np.zeros(32, dtype=np.float32)
        hist_vec = hist_vec / (float(hist_vec.sum()) + 1e-6)
        diversity_scalar = np.array([_hue_diversity(colorful)], dtype=np.float32)
        motif_coverage = float(crop_mask.mean())
        coverage_centers = np.array([0.015, 0.04, 0.08, 0.16, 0.32], dtype=np.float32)
        coverage_vec = np.exp(-((motif_coverage - coverage_centers) ** 2) / (2.0 * (0.045 ** 2))).astype(np.float32)
        coverage_vec = coverage_vec / (float(np.linalg.norm(coverage_vec)) + 1e-8)
        crop_gray = (
            0.299 * crop_rgb[..., 0].astype(np.float32)
            + 0.587 * crop_rgb[..., 1].astype(np.float32)
            + 0.114 * crop_rgb[..., 2].astype(np.float32)
        )
        bg = ~crop_mask
        dark_base = float(np.mean(crop_gray[bg] < 112.0)) if np.any(bg) else 0.0
        dark_base_scalar = np.array([dark_base], dtype=np.float32)

        # Geometry keeps diamond/vertical-bar layouts separate from generic colorful blocks.
        proj_x = mask_grid.sum(axis=0)
        proj_y = mask_grid.sum(axis=1)
        proj_x = proj_x / (float(np.linalg.norm(proj_x)) + 1e-8)
        proj_y = proj_y / (float(np.linalg.norm(proj_y)) + 1e-8)

        v = np.concatenate([
            mask_vec * 0.75,
            weighted_rgb * 0.55,
            hist_vec * 1.25,
            proj_x * 0.80,
            proj_y * 0.80,
            diversity_scalar * 1.50,
            coverage_vec * 1.20,
            dark_base_scalar * 0.80,
        ]).astype(np.float32)
        n = float(np.linalg.norm(v)) + 1e-8
        return (v / n).astype(np.float32)

    def _extract_dark_motif_sig(path: Path, size: int = 18) -> np.ndarray | None:
        try:
            with Image.open(path) as im0:
                im = im0.convert("RGB")
                im.thumbnail((520, 520), Image.Resampling.BILINEAR)
                rgb = np.asarray(im, dtype=np.uint8)
        except Exception:
            return None
        if rgb.ndim != 3 or rgb.shape[0] < 32 or rgb.shape[1] < 32:
            return None
        h, w = rgb.shape[:2]
        arr = rgb.astype(np.float32)
        gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
        mx = arr.max(axis=-1)
        mn = arr.min(axis=-1)
        sat = (mx - mn) / np.clip(mx, 1.0, None)

        valid = np.ones((h, w), dtype=bool)
        valid[: int(h * 0.08), :] = False
        dark = valid & (gray < 95.0) & (sat < 0.75)
        if cv2 is not None:
            n, labels, stats, _centers = cv2.connectedComponentsWithStats(dark.astype("uint8"), 8)
            best: tuple[float, int, int, int, int] | None = None
            min_area = max(30, int(h * w * 0.002))
            for idx in range(1, n):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                x = int(stats[idx, cv2.CC_STAT_LEFT])
                y = int(stats[idx, cv2.CC_STAT_TOP])
                ww = int(stats[idx, cv2.CC_STAT_WIDTH])
                hh = int(stats[idx, cv2.CC_STAT_HEIGHT])
                cx = (x + ww / 2.0) / max(1, w)
                cy = (y + hh / 2.0) / max(1, h)
                central = 1.0 - abs(cx - 0.5) * 0.7 - abs(cy * 0.85 - 0.5) * 0.2
                score = float(area) * max(0.2, float(central))
                if best is None or score > best[0]:
                    best = (score, x, y, ww, hh)
            if best is None:
                return None
            _score, x, y, ww, hh = best
            pad = max(4, int(max(ww, hh) * 0.10))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + ww + pad)
            y1 = min(h, y + hh + pad)
        else:
            ys, xs = np.where(dark)
            if ys.size < 30:
                return None
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1

        crop = rgb[y0:y1, x0:x1]
        if crop.shape[0] < 16 or crop.shape[1] < 16:
            return None
        ca = crop.astype(np.float32)
        cgray = (0.299 * ca[..., 0] + 0.587 * ca[..., 1] + 0.114 * ca[..., 2]).astype(np.float32)
        cmx = ca.max(axis=-1)
        cmn = ca.min(axis=-1)
        csat = (cmx - cmn) / np.clip(cmx, 1.0, None)
        cr, cg, cb = ca[..., 0], ca[..., 1], ca[..., 2]
        chroma = cmx - cmn
        motif = ((csat > 0.26) & (cmx > 95.0) & (chroma > 34.0)) | (
            (csat < 0.24) & (cgray > 138.0) & (cgray < 245.0)
        )
        dark_local = cgray < 100.0
        if cv2 is not None:
            near_dark = cv2.dilate(dark_local.astype("uint8"), np.ones((5, 5), np.uint8), iterations=2).astype(bool)
            motif &= near_dark
            n, labels, stats, _centers = cv2.connectedComponentsWithStats(motif.astype("uint8"), 8)
            keep = np.zeros_like(motif, dtype=bool)
            ch, cw = motif.shape
            for idx in range(1, n):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < 4 or area > ch * cw * 0.35:
                    continue
                keep[labels == idx] = True
            motif = keep
        if int(motif.sum()) < 10:
            return None

        color_motif = motif & (csat > 0.34) & (chroma > 45.0)
        cats = [
            motif & (cgray > 138.0) & (csat < 0.24),
            color_motif & (cr > 125.0) & (cb > 95.0) & (cg < 175.0) & (np.abs(cr - cb) < 95.0),
            color_motif & (cr > 135.0) & (cg > 105.0) & (cb < 135.0) & ((cr + cg) > (cb * 2.2)),
            color_motif & (cb > 115.0) & (cb > cr + 22.0) & (cb > cg + 6.0),
            color_motif & (cg > 105.0) & (cg > cr + 15.0) & (cg > cb + 2.0),
        ]
        vecs: List[np.ndarray] = []
        for mask in cats + [motif]:
            img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize((size, size), Image.BILINEAR)
            v = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
            v = v / (float(np.linalg.norm(v)) + 1e-8)
            vecs.append(v)
        presence = np.array([float(int(mask.sum()) > 8) for mask in cats], dtype=np.float32)
        v = np.concatenate(
            [
                vecs[0] * 1.6,
                vecs[1] * 1.4,
                vecs[2] * 1.4,
                vecs[3] * 1.0,
                vecs[4] * 1.0,
                vecs[5] * 0.8,
                presence * 1.2,
            ]
        ).astype(np.float32)
        return (v / (float(np.linalg.norm(v)) + 1e-8)).astype(np.float32)

    def _merge_accent_pattern_candidates(
        ranked: List[tuple[str, float]],
        query_sig: np.ndarray | None,
    ) -> tuple[List[tuple[str, float]], str]:
        if not ranked or query_sig is None or not accent_pattern_cache:
            return ranked, ""
        scored: List[tuple[str, float]] = []
        for file_name, sig in accent_pattern_cache.items():
            sim = float(query_sig @ sig)
            if sim >= accent_pattern_min_score:
                scored.append((file_name, sim))
        if not scored:
            return ranked, ""
        scored.sort(key=lambda x: x[1], reverse=True)
        injected = scored[: max(1, accent_pattern_max_injected)]

        merged: Dict[str, float] = {}
        for name, score in ranked:
            merged[name] = max(float(score), merged.get(name, -1e9))
        for file_name, sim in injected:
            seed = accent_pattern_seed_score_base + accent_pattern_boost_scale * max(0.0, sim)
            merged[file_name] = max(merged.get(file_name, -1e9), float(seed))

        debug_items = [
            f"{filename_to_style_code(file_name)}:{sim:.3f}/{accent_pattern_seed_score_base + accent_pattern_boost_scale * max(0.0, sim):.3f}"
            for file_name, sim in injected[:40]
        ]
        out = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return out, ",".join(debug_items)

    def _normalize_collar_contour_map(sig_map: np.ndarray, size: int, min_pixels: int) -> np.ndarray | None:
        if int(sig_map.sum()) < max(1, min_pixels):
            return None
        arr = np.asarray(
            Image.fromarray((sig_map.astype(np.uint8) * 255), mode="L").resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        arr = arr - float(arr.mean())
        norm = float(np.linalg.norm(arr)) + 1e-8
        if norm <= 1e-8:
            return None
        return (arr.ravel() / norm).astype(np.float32)

    def _extract_collar_contour_sigs_from_image(image: Image.Image, size: int = 48) -> List[np.ndarray]:
        im = image.convert("RGB")
        im.thumbnail((256, 256), Image.Resampling.BILINEAR)
        rgb = np.asarray(im, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[0] < 24 or rgb.shape[1] < 24:
            return []
        h, w = rgb.shape[:2]
        top_cut = min(h, max(24, int(h * 0.72)))
        rgb = rgb[:top_cut, :, :]
        h, w = rgb.shape[:2]
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        maxc = rgb.max(axis=-1).astype(np.float32)
        minc = rgb.min(axis=-1).astype(np.float32)
        sat = (maxc - minc) / np.maximum(maxc, 1.0)
        fg = ((gray < 242.0) & np.any(rgb < 246, axis=-1))
        fg[: min(int(h * 0.10), 20), :] = False
        if int(fg.sum()) < max(32, int(h * w * 0.015)):
            return []
        line_core = (((sat > 0.10) & (gray < 245.0)) | (gray < 105.0)) & fg
        dark = ((gray < 228.0) & fg).astype(np.uint8)
        if cv2 is not None:
            edge = (cv2.Canny(gray.astype(np.uint8), 60, 140) > 0).astype(np.uint8)
            line_near = cv2.dilate(line_core.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
            gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        else:
            gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
            gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
            edge = ((gx + gy) > 28.0).astype(np.uint8)
            line_near = line_core
        line_edge = (edge > 0) & line_near
        grad_mag = np.sqrt(gx.astype(np.float32) * gx.astype(np.float32) + gy.astype(np.float32) * gy.astype(np.float32))
        tangent = (np.degrees(np.arctan2(gy.astype(np.float32), gx.astype(np.float32))) + 90.0) % 180.0
        diagonal = ((tangent >= 24.0) & (tangent <= 76.0)) | ((tangent >= 104.0) & (tangent <= 156.0))
        vline_edge = line_edge & diagonal & (grad_mag > 18.0)
        sigs: List[np.ndarray] = []
        vline_sig = _normalize_collar_contour_map(
            vline_edge.astype(np.uint8),
            size,
            max(6, int(h * w * 0.0010)),
        )
        if vline_sig is not None:
            sigs.append(vline_sig)
        line_sig = _normalize_collar_contour_map(
            line_edge.astype(np.uint8),
            size,
            max(8, int(h * w * 0.0015)),
        )
        if line_sig is not None:
            sigs.append(line_sig)
        contour_sig = _normalize_collar_contour_map(
            np.maximum(edge * fg.astype(np.uint8), dark),
            size,
            max(24, int(h * w * 0.008)),
        )
        if contour_sig is not None and all(float(contour_sig @ sig) < 0.985 for sig in sigs):
            sigs.append(contour_sig)
        return sigs

    def _extract_collar_contour_sig_from_image(image: Image.Image, size: int = 48) -> np.ndarray | None:
        sigs = _extract_collar_contour_sigs_from_image(image, size=size)
        return sigs[0] if sigs else None

    def _append_unique_collar_sigs(dst: List[np.ndarray], sigs: List[np.ndarray], max_sigs: int) -> None:
        for sig in sigs:
            if len(dst) >= max(1, int(max_sigs)):
                break
            if all(float(sig @ existing) < 0.992 for existing in dst):
                dst.append(sig)

    def _extract_collar_contour_sig(path: Path, size: int = 48) -> np.ndarray | None:
        try:
            with Image.open(path) as im0:
                return _extract_collar_contour_sig_from_image(im0.convert("RGB"), size=size)
        except Exception:
            return None

    def _extract_collar_contour_sigs(path: Path, size: int = 48) -> List[np.ndarray]:
        try:
            with Image.open(path) as im0:
                return _extract_collar_contour_sigs_from_image(im0.convert("RGB"), size=size)
        except Exception:
            return []

    def _extract_collar_chevron_score_from_image(image: Image.Image) -> float:
        if cv2 is None:
            return 0.0
        im = image.convert("RGB")
        im.thumbnail((320, 320), Image.Resampling.BILINEAR)
        rgb = np.asarray(im, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[0] < 28 or rgb.shape[1] < 28:
            return 0.0
        h, w = rgb.shape[:2]
        top_cut = min(h, max(24, int(h * 0.75)))
        rgb = rgb[:top_cut, :, :]
        h, w = rgb.shape[:2]
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        maxc = rgb.max(axis=-1).astype(np.float32)
        minc = rgb.min(axis=-1).astype(np.float32)
        sat = (maxc - minc) / np.maximum(maxc, 1.0)
        bright_support = cv2.dilate(((gray > 165.0) & (sat < 0.45)).astype(np.uint8), np.ones((9, 9), np.uint8), iterations=2) > 0
        line_core = (((sat > 0.10) & (gray < 245.0)) | ((gray < 120.0) & bright_support))
        line_core[: min(int(h * 0.08), 20), :] = False
        edges = (cv2.Canny(gray.astype(np.uint8), 50, 130) > 0).astype(np.uint8) * 255
        line_mask = cv2.dilate(line_core.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, edges, mask=line_mask)
        min_line = max(14, int(min(h, w) * 0.10))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=16,
            minLineLength=min_line,
            maxLineGap=7,
        )
        if lines is None:
            return 0.0
        pos_len = 0.0
        neg_len = 0.0
        hor_len = 0.0
        pos_segments: List[tuple[float, float, float, float, float]] = []
        neg_segments: List[tuple[float, float, float, float, float]] = []
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = math.hypot(dx, dy)
            if length < min_line:
                continue
            angle = (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0
            if 20.0 <= angle <= 75.0:
                pos_len += length
                pos_segments.append((float(x1), float(y1), float(x2), float(y2), length))
            elif 105.0 <= angle <= 160.0:
                neg_len += length
                neg_segments.append((float(x1), float(y1), float(x2), float(y2), length))
            elif angle < 12.0 or angle > 168.0:
                hor_len += length
        diag_len = pos_len + neg_len
        both_len = min(pos_len, neg_len)
        if diag_len < 20.0 or both_len < 10.0:
            return 0.0
        balance = 2.0 * both_len / max(1e-6, diag_len)
        density = min(1.0, diag_len / max(1.0, float(min(h, w)) * 1.6))
        horizontal_penalty = 1.0 - min(0.5, hor_len / max(1e-6, hor_len + diag_len))
        dark_fill = float(np.mean((gray < 120.0) & (sat < 0.35) & bright_support))
        dark_fill_penalty = 1.0 - min(0.55, max(0.0, dark_fill - 0.06) * 2.8)
        solid_mask = ((gray < 125.0) & (sat < 0.45) & bright_support).astype(np.uint8)
        solid_mask[: min(int(h * 0.08), 20), :] = 0
        solid_count, _, solid_stats, _ = cv2.connectedComponentsWithStats(solid_mask, 8)
        largest_solid = 0.0
        if solid_count > 1:
            largest_solid = float(np.max(solid_stats[1:, cv2.CC_STAT_AREA])) / max(1.0, float(h * w))
        solid_penalty = 1.0 - min(0.50, max(0.0, largest_solid - 0.025) * 5.0)
        base_score = balance * density * horizontal_penalty * dark_fill_penalty * solid_penalty

        corner_points: List[tuple[float, float, float]] = []
        corner_limit = max(10.0, float(min(h, w)) * 0.16)
        for px1, py1, px2, py2, plen in pos_segments[:24]:
            for nx1, ny1, nx2, ny2, nlen in neg_segments[:24]:
                best_dist = 1e9
                best_point = (0.0, 0.0)
                for pa in ((px1, py1), (px2, py2)):
                    for na in ((nx1, ny1), (nx2, ny2)):
                        dist = math.hypot(float(pa[0] - na[0]), float(pa[1] - na[1]))
                        if dist < best_dist:
                            best_dist = dist
                            best_point = ((float(pa[0]) + float(na[0])) * 0.5, (float(pa[1]) + float(na[1])) * 0.5)
                if best_dist > corner_limit:
                    continue
                weight = min(plen, nlen) / max(1.0, float(min(h, w)) * 0.18)
                corner_points.append((best_point[0], best_point[1], min(1.0, weight)))
        corner_points.sort(key=lambda item: item[2], reverse=True)
        distinct_corners: List[tuple[float, float, float]] = []
        for cx, cy, weight in corner_points:
            if any(math.hypot(cx - ox, cy - oy) < corner_limit for ox, oy, _ in distinct_corners):
                continue
            distinct_corners.append((cx, cy, weight))
            if len(distinct_corners) >= 4:
                break
        corner_strength = sum(weight for _cx, _cy, weight in distinct_corners)
        corner_gate = min(1.0, corner_strength / 1.8)
        if len(distinct_corners) <= 1:
            corner_gate *= 0.65
        if distinct_corners:
            xs = [cx for cx, _cy, _weight in distinct_corners]
            spread = (max(xs) - min(xs)) / max(1.0, float(w))
            if spread < 0.18 and len(distinct_corners) >= 2:
                corner_gate *= 0.82
        gated_score = base_score * (0.45 + 0.55 * corner_gate)
        return float(max(0.0, min(1.0, gated_score)))

    def _extract_collar_chevron_score(path: Path) -> float:
        try:
            with Image.open(path) as im0:
                return _extract_collar_chevron_score_from_image(im0.convert("RGB"))
        except Exception:
            return 0.0

    def _merge_collar_contour_candidates(
        ranked: List[tuple[str, float]],
        query_sigs: List[np.ndarray] | np.ndarray | None,
        query_sig_mirrors: List[np.ndarray] | np.ndarray | None,
        query_chevron_score: float = 0.0,
    ) -> tuple[List[tuple[str, float]], str, Dict[str, tuple[float, str]]]:
        if isinstance(query_sigs, np.ndarray):
            q_sigs = [query_sigs]
        else:
            q_sigs = list(query_sigs or [])
        if isinstance(query_sig_mirrors, np.ndarray):
            q_mirror_sigs = [query_sig_mirrors]
        else:
            q_mirror_sigs = list(query_sig_mirrors or [])
        if not ranked or not q_sigs or not collar_contour_cache:
            return ranked, "", {}
        scored: List[tuple[str, float]] = []
        base_best_contour: Dict[str, tuple[float, str]] = {}
        code_repeat_view_hits: Dict[str, set[str]] = {}
        for file_name, sig in collar_contour_cache.items():
            sim = max(float(query_sig @ sig) for query_sig in q_sigs)
            if q_mirror_sigs:
                sim = max(sim, max(float(query_sig_mirror @ sig) for query_sig_mirror in q_mirror_sigs))
            base_file_name = file_name.split("@", 1)[0]
            if sim >= float(collar_contour_repeat_min_score):
                code = filename_to_style_code(base_file_name)
                if code:
                    code_repeat_view_hits.setdefault(code, set()).add(file_name)
            current_base = base_best_contour.get(base_file_name)
            if current_base is None or sim > current_base[0]:
                base_best_contour[base_file_name] = (float(sim), file_name)
            chevron_score = float(collar_chevron_cache.get(base_file_name, 0.0)) if collar_chevron_enabled else 0.0
            chevron_match = (
                collar_chevron_enabled
                and float(query_chevron_score) >= float(collar_chevron_query_min_score)
                and chevron_score >= float(collar_chevron_standard_min_score)
                and sim >= float(collar_chevron_min_contour_score)
            )
            if sim >= collar_contour_min_score or chevron_match:
                effective_sim = max(sim, sim + float(collar_chevron_score_boost) * chevron_score) if chevron_match else sim
                scored.append((file_name, effective_sim))
        chevron_debug: List[str] = []
        if collar_chevron_enabled and float(query_chevron_score) >= float(collar_chevron_query_min_score):
            code_best_contour: Dict[str, tuple[float, str]] = {}
            for base_file_name, (contour_sim, contour_file_name) in base_best_contour.items():
                code = filename_to_style_code(base_file_name)
                current = code_best_contour.get(code)
                if current is None or float(contour_sim) > float(current[0]):
                    code_best_contour[code] = (float(contour_sim), contour_file_name)
            code_best_chevron: Dict[str, tuple[float, str]] = {}
            for base_file_name, chevron_score in collar_chevron_cache.items():
                code = filename_to_style_code(base_file_name)
                current = code_best_chevron.get(code)
                if current is None or float(chevron_score) > float(current[0]):
                    code_best_chevron[code] = (float(chevron_score), base_file_name)
            chevron_ranked = sorted(
                (
                    (file_name, float(score), base_best_contour.get(file_name, (0.0, file_name))[0])
                    for file_name, score in collar_chevron_cache.items()
                    if float(score) >= float(collar_chevron_standard_min_score)
                ),
                key=lambda item: (item[1], item[2]),
                reverse=True,
            )[: max(1, collar_chevron_max_injected)]
            for base_file_name, chevron_score, contour_sim in chevron_ranked:
                if contour_sim < float(collar_chevron_min_contour_score):
                    continue
                score = max(
                    contour_sim,
                    float(collar_chevron_seed_score_base) + float(collar_chevron_score_boost) * chevron_score,
                )
                scored.append((base_best_contour.get(base_file_name, (contour_sim, base_file_name))[1], float(score)))
                chevron_debug.append(f"{filename_to_style_code(base_file_name)}:{chevron_score:.3f}/{contour_sim:.3f}/{score:.3f}")
            for code, (chevron_score, base_file_name) in code_best_chevron.items():
                if chevron_score < float(collar_chevron_code_min_score):
                    continue
                contour_sim, contour_file_name = code_best_contour.get(code, (0.0, base_file_name))
                if contour_sim < float(collar_chevron_code_contour_min_score):
                    continue
                score = max(
                    contour_sim,
                    float(collar_chevron_seed_score_base)
                    + float(collar_chevron_score_boost) * chevron_score
                    + float(collar_chevron_code_contour_boost) * min(1.0, max(0.0, contour_sim)),
                )
                if contour_sim >= float(collar_chevron_code_fallback_contour_min_score):
                    score = max(
                        score,
                        float(collar_chevron_seed_score_base)
                        + float(collar_chevron_code_fallback_boost)
                        + float(collar_chevron_code_contour_boost) * min(1.0, max(0.0, contour_sim)),
                    )
                scored.append((contour_file_name, float(score)))
                chevron_debug.append(f"code:{code}:{chevron_score:.3f}/{contour_sim:.3f}/{score:.3f}")
        if not scored:
            return ranked, "", {}
        repeat_boost_by_code: Dict[str, float] = {}
        repeat_min_hits = max(2, int(collar_contour_repeat_min_hits))
        if float(collar_contour_repeat_boost) > 0.0:
            code_repeat_hits: Dict[str, set[str]] = {}
            for base_file_name, (contour_sim, _contour_file_name) in base_best_contour.items():
                if float(contour_sim) < float(collar_contour_repeat_min_score):
                    continue
                code = filename_to_style_code(base_file_name)
                if not code:
                    continue
                code_repeat_hits.setdefault(code, set()).add(base_file_name)
            for code, base_names in code_repeat_hits.items():
                base_hit_count = len(base_names)
                view_hit_count = len(code_repeat_view_hits.get(code, set()))
                if base_hit_count >= repeat_min_hits:
                    repeat_steps = max(1, base_hit_count - repeat_min_hits + 1)
                    repeat_boost = float(collar_contour_repeat_boost) * float(repeat_steps)
                    repeat_boost += float(collar_contour_multi_image_boost)
                elif view_hit_count >= repeat_min_hits and float(collar_contour_repeat_view_boost) > 0.0:
                    repeat_steps = max(1, view_hit_count - repeat_min_hits + 1)
                    repeat_boost = float(collar_contour_repeat_view_boost) * float(repeat_steps)
                else:
                    continue
                repeat_boost_by_code[code] = min(
                    float(collar_contour_repeat_max_boost),
                    float(repeat_boost),
                )
        if repeat_boost_by_code:
            scored = [
                (file_name, float(sim) + float(repeat_boost_by_code.get(filename_to_style_code(file_name), 0.0)))
                for file_name, sim in scored
            ]
        scored.sort(key=lambda x: x[1], reverse=True)
        injected: List[tuple[str, float]] = []
        seen_injected_codes: set[str] = set()
        for file_name, sim in scored:
            code = filename_to_style_code(file_name)
            if code in seen_injected_codes:
                continue
            seen_injected_codes.add(code)
            injected.append((file_name, sim))
            if len(injected) >= max(1, collar_contour_max_injected):
                break
        merged: Dict[str, float] = {}
        for name, score in ranked:
            merged[name] = max(float(score), merged.get(name, -1e9))
        code_matches: Dict[str, tuple[float, str]] = {}
        for file_name, sim in injected:
            seed = collar_contour_seed_score_base + collar_contour_boost_scale * max(0.0, sim)
            merged[file_name] = max(merged.get(file_name, -1e9), float(seed))
            code = filename_to_style_code(file_name)
            current = code_matches.get(code)
            if current is None or float(sim) > float(current[0]):
                code_matches[code] = (float(sim), file_name.split("@", 1)[0])
        debug_items = [
            f"{filename_to_style_code(file_name)}:{sim:.3f}/{collar_contour_seed_score_base + collar_contour_boost_scale * max(0.0, sim):.3f}"
            for file_name, sim in injected[:24]
        ]
        if chevron_debug:
            debug_items.append("chev=" + ",".join(chevron_debug[:64]))
        return sorted(merged.items(), key=lambda x: x[1], reverse=True), ",".join(debug_items), code_matches

    def _extract_sleeve_pattern_sig_from_image(image: Image.Image, size: int = 32) -> np.ndarray | None:
        im = image.convert("RGB")
        im.thumbnail((320, 320), Image.Resampling.BILINEAR)
        rgb = np.asarray(im, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[0] < 40 or rgb.shape[1] < 40:
            return None

        h, w = rgb.shape[:2]
        arr = rgb.astype(np.float32)
        maxc = arr.max(axis=-1)
        minc = arr.min(axis=-1)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        sat = (maxc - minc) / np.maximum(maxc, 1.0)
        # Sleeve details in model photos are usually colored blocks plus white/dark stripe bands.
        colorful = (sat > 0.20) & (maxc > 70.0) & (minc < 245.0)
        colorful[: min(int(h * 0.08), 32), :] = False
        if int(colorful.sum()) < 80:
            return None
        stripe_detail = (
            colorful
            | ((gray < 112.0) & (maxc > 18.0))
            | ((gray > 168.0) & (gray < 246.0) & (sat < 0.22))
        )
        stripe_detail[: min(int(h * 0.08), 32), :] = False

        ys, xs = np.where(colorful)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        pad_x = max(8, int((x1 - x0 + 1) * 0.45))
        pad_y = max(8, int((y1 - y0 + 1) * 0.55))
        x0 = max(0, x0 - pad_x)
        x1 = min(w, x1 + pad_x + 1)
        y0 = max(0, y0 - pad_y)
        y1 = min(h, y1 + pad_y + 1)
        crop_rgb = rgb[y0:y1, x0:x1]
        crop_mask = stripe_detail[y0:y1, x0:x1]
        if crop_rgb.shape[0] < 20 or crop_rgb.shape[1] < 20 or int(crop_mask.sum()) < 60:
            return None

        rgb_img = Image.fromarray(crop_rgb, mode="RGB").resize((size, size), Image.BILINEAR)
        mask_img = Image.fromarray((crop_mask.astype(np.uint8) * 255), mode="L").resize((size, size), Image.BILINEAR)
        grid_rgb = np.asarray(rgb_img, dtype=np.float32) / 255.0
        grid_mask = np.asarray(mask_img, dtype=np.float32) / 255.0
        grid_gray = 0.299 * grid_rgb[..., 0] + 0.587 * grid_rgb[..., 1] + 0.114 * grid_rgb[..., 2]
        grid_gray_raw = grid_gray.copy()
        grid_chroma = grid_rgb.max(axis=-1) - grid_rgb.min(axis=-1)
        dark_band = (grid_gray < 0.26).astype(np.float32).mean(axis=1)
        light_band = (grid_gray > 0.72).astype(np.float32).mean(axis=1)
        chroma_band = grid_chroma.mean(axis=1).astype(np.float32)
        band_profile = np.concatenate([dark_band, light_band, chroma_band]).astype(np.float32)
        band_profile = band_profile / (float(np.linalg.norm(band_profile)) + 1e-8)
        grid_gray = (grid_gray - float(grid_gray.mean())) / (float(grid_gray.std()) + 1e-6)
        grid_gray = np.clip(grid_gray / 3.0, -1.0, 1.0)

        edge_y = np.abs(np.diff(grid_gray, axis=0)).mean(axis=1).astype(np.float32)
        edge_x = np.abs(np.diff(grid_gray, axis=1)).mean(axis=0).astype(np.float32)
        edge_y = edge_y / (float(np.linalg.norm(edge_y)) + 1e-8)
        edge_x = edge_x / (float(np.linalg.norm(edge_x)) + 1e-8)
        proj_y = grid_mask.mean(axis=1).astype(np.float32)
        proj_x = grid_mask.mean(axis=0).astype(np.float32)
        proj_y = proj_y / (float(np.linalg.norm(proj_y)) + 1e-8)
        proj_x = proj_x / (float(np.linalg.norm(proj_x)) + 1e-8)

        selected = crop_rgb[crop_mask]
        hist_parts: List[np.ndarray] = []
        if selected.shape[0] > 0:
            sel = selected.astype(np.float32) / 255.0
            for ci in range(3):
                hist, _ = np.histogram(sel[:, ci], bins=6, range=(0.0, 1.0))
                hist_parts.append(hist.astype(np.float32))
            chroma = sel.max(axis=-1) - sel.min(axis=-1)
            hist, _ = np.histogram(chroma, bins=6, range=(0.0, 1.0))
            hist_parts.append(hist.astype(np.float32))
        hist_vec = np.concatenate(hist_parts).astype(np.float32) if hist_parts else np.zeros(24, dtype=np.float32)
        hist_vec = hist_vec / (float(hist_vec.sum()) + 1e-6)

        def _run_count(flags: np.ndarray) -> float:
            if flags.size == 0:
                return 0.0
            vals = flags.astype(np.uint8)
            starts = vals.copy()
            starts[1:] = np.maximum(0, vals[1:] - vals[:-1])
            return float(starts.sum())

        neutral_light = ((grid_gray_raw > 0.70) & (grid_chroma < 0.26)).astype(np.float32)
        row_light = neutral_light.mean(axis=1).astype(np.float32)
        row_dark = (grid_gray_raw < 0.30).astype(np.float32).mean(axis=1)
        row_color = (grid_chroma > 0.14).astype(np.float32).mean(axis=1)
        light_runs = _run_count(row_light > 0.16)
        dark_runs = _run_count(row_dark > 0.22)
        color_runs = _run_count(row_color > 0.18)
        row_band_delta = np.diff(row_dark - row_light).astype(np.float32)
        stripe_strength = (
            min(1.0, light_runs / 3.0)
            * min(1.0, float(row_light.max(initial=0.0)) * 3.0)
            * min(1.0, float(row_dark.max(initial=0.0)) * 2.5)
        )
        colored_panel = min(1.0, color_runs / 2.0) * min(1.0, float(row_color.max(initial=0.0)) * 2.2)
        band_alternation = min(1.0, float(np.mean(np.abs(row_band_delta))) * 4.0) if row_band_delta.size else 0.0
        sleeve_structure = np.array(
            [
                stripe_strength,
                min(1.0, light_runs / 5.0),
                min(1.0, dark_runs / 4.0),
                colored_panel,
                band_alternation,
                float(row_light.max(initial=0.0)),
                float(row_dark.max(initial=0.0)),
                float(row_color.max(initial=0.0)),
            ],
            dtype=np.float32,
        )
        sleeve_row_profile = np.concatenate([row_light, row_dark, row_color]).astype(np.float32)
        sleeve_row_profile = sleeve_row_profile / (float(np.linalg.norm(sleeve_row_profile)) + 1e-8)

        aspect = float((x1 - x0) / max(1, (y1 - y0)))
        coverage = float(grid_mask.mean())
        horizontal_strength = float(np.mean(edge_y > (edge_y.mean() + edge_y.std()))) if edge_y.size else 0.0
        shape_vec = np.array([
            min(1.0, aspect / 2.4),
            min(1.0, 2.4 / max(0.1, aspect)),
            coverage,
            horizontal_strength,
        ], dtype=np.float32)

        v = np.concatenate([
            grid_gray.reshape(-1) * 0.06,
            grid_mask.reshape(-1) * 0.28,
            edge_y * 3.20,
            edge_x * 0.28,
            band_profile * 4.60,
            sleeve_row_profile * 8.20,
            proj_y * 1.80,
            proj_x * 0.35,
            hist_vec * 0.08,
            sleeve_structure * 10.00,
            shape_vec * 1.35,
        ]).astype(np.float32)
        n = float(np.linalg.norm(v)) + 1e-8
        return (v / n).astype(np.float32)

    def _extract_sleeve_pattern_sig(path: Path, size: int = 32) -> np.ndarray | None:
        try:
            with Image.open(path) as im0:
                return _extract_sleeve_pattern_sig_from_image(im0, size=size)
        except Exception:
            return None

    def _merge_sleeve_pattern_candidates(
        ranked: List[tuple[str, float]],
        query_sig: np.ndarray | None,
    ) -> tuple[List[tuple[str, float]], str]:
        if not ranked or query_sig is None or not sleeve_pattern_cache:
            return ranked, ""
        scored: List[tuple[str, float, float]] = []
        for file_name, sig in sleeve_pattern_cache.items():
            sim = float(query_sig @ sig)
            if sim >= sleeve_pattern_min_score:
                base_name = file_name.split("@", 1)[0]
                pair_prior = float(sleeve_pair_prior_cache.get(base_name, 0.0))
                candidate_score = sim + max(0.0, sleeve_pair_prior_candidate_boost) * pair_prior
                scored.append((file_name, sim, candidate_score))
        if not scored:
            return ranked, ""
        scored.sort(key=lambda x: x[2], reverse=True)
        injected: List[tuple[str, float]] = []
        injected_keys = set()
        for file_name, sim, _candidate_score in scored:
            key = re.sub(r"[^A-Za-z0-9_-]+", "", filename_to_style_code(file_name).strip().upper())
            if not key or key in injected_keys:
                continue
            injected_keys.add(key)
            injected.append((file_name, sim))
            if len(injected) >= max(1, sleeve_pattern_max_injected):
                break
        merged: Dict[str, float] = {}
        for name, score in ranked:
            merged[name] = max(float(score), merged.get(name, -1e9))
        for file_name, sim in injected:
            base_name = file_name.split("@", 1)[0]
            pair_prior = float(sleeve_pair_prior_cache.get(base_name, 0.0))
            seed = (
                sleeve_pattern_seed_score_base
                + sleeve_pattern_boost_scale * max(0.0, sim)
                + max(0.0, sleeve_pair_prior_boost) * pair_prior
            )
            merged[file_name] = max(merged.get(file_name, -1e9), float(seed))
        debug_items = [
            (
                f"{filename_to_style_code(file_name)}:{sim:.3f}/"
                f"{sleeve_pattern_seed_score_base + sleeve_pattern_boost_scale * max(0.0, sim) + max(0.0, sleeve_pair_prior_boost) * float(sleeve_pair_prior_cache.get(file_name.split('@', 1)[0], 0.0)):.3f}/"
                f"{float(sleeve_pair_prior_cache.get(file_name.split('@', 1)[0], 0.0)):.2f}"
            )
            for file_name, sim in injected[:40]
        ]
        out = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return out, ",".join(debug_items)

    def _extract_accessory_pattern_sig(path: Path, size: int = 48) -> np.ndarray | None:
        try:
            with Image.open(path) as im0:
                im = im0.convert("RGB")
                im.thumbnail((320, 320), Image.Resampling.BILINEAR)
                rgb = np.asarray(im, dtype=np.uint8)
        except Exception:
            return None
        if rgb.ndim != 3 or rgb.shape[0] < 40 or rgb.shape[1] < 40:
            return None
        h, w = rgb.shape[:2]
        arr = rgb.astype(np.float32)
        gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
        # Prefer non-background foreground; model photos still keep hat/cord high contrast.
        fg = (gray < 235.0) & (np.any(rgb < 245, axis=-1))
        fg[: min(int(h * 0.08), 32), :] = False
        if int(fg.sum()) < 64:
            return None

        ys, xs = np.where(fg)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        crop = fg[y0 : y1 + 1, x0 : x1 + 1]
        if crop.shape[0] < 24 or crop.shape[1] < 24:
            return None

        mask_img = Image.fromarray((crop.astype(np.uint8) * 255), mode="L").resize((size, size), Image.BILINEAR)
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0
        bits = mask > 0.35
        if int(bits.sum()) < 24:
            return None

        proj_x = bits.mean(axis=0).astype(np.float32)
        proj_y = bits.mean(axis=1).astype(np.float32)
        lower = bits[int(size * 0.38) :, :]
        col_strength = lower.mean(axis=0)
        cord_cols = int(np.count_nonzero(col_strength > 0.10))
        # Hats with hanging cords have sparse, separated vertical foreground in lower half.
        cord_score = min(1.0, cord_cols / 10.0)
        top_mass = float(bits[: int(size * 0.45), :].mean())
        lower_mass = float(lower.mean())
        bottom_mass = float(bits[int(size * 0.70) :, :].mean())
        lower_col = col_strength.astype(np.float32)
        lower_col = lower_col / (float(np.linalg.norm(lower_col)) + 1e-8)
        lower_row = lower.mean(axis=1).astype(np.float32)
        lower_row = lower_row / (float(np.linalg.norm(lower_row)) + 1e-8)
        gray_crop = gray[y0 : y1 + 1, x0 : x1 + 1]
        rgb_crop = rgb[y0 : y1 + 1, x0 : x1 + 1]
        crop_mask = crop.astype(bool)
        color_hist_parts: List[np.ndarray] = []
        if np.any(crop_mask):
            selected_rgb = rgb_crop[crop_mask].astype(np.float32) / 255.0
            for ci in range(3):
                hist, _ = np.histogram(selected_rgb[:, ci], bins=8, range=(0.0, 1.0))
                color_hist_parts.append(hist.astype(np.float32))
            selected_gray = gray_crop[crop_mask].astype(np.float32) / 255.0
            hist, _ = np.histogram(selected_gray, bins=8, range=(0.0, 1.0))
            color_hist_parts.append(hist.astype(np.float32))
        color_hist = np.concatenate(color_hist_parts).astype(np.float32) if color_hist_parts else np.zeros(32, dtype=np.float32)
        color_hist = color_hist / (float(color_hist.sum()) + 1e-6)
        cord_gap = 0.0
        active_cols = np.where(col_strength > 0.10)[0]
        if active_cols.size >= 2:
            cord_gap = min(1.0, float(active_cols.max() - active_cols.min()) / float(max(1, size)))
        aspect = float((x1 - x0 + 1) / max(1, (y1 - y0 + 1)))
        aspect_vec = np.array([
            min(1.0, aspect / 1.8),
            min(1.0, 1.8 / max(0.1, aspect)),
            top_mass,
            lower_mass,
            bottom_mass,
            cord_score,
            cord_gap,
        ], dtype=np.float32)

        v = np.concatenate([
            mask.reshape(-1) * 0.45,
            proj_x * 1.00,
            proj_y * 1.00,
            lower_col * 3.00,
            lower_row * 2.40,
            color_hist * 1.80,
            aspect_vec * 3.20,
        ]).astype(np.float32)
        n = float(np.linalg.norm(v)) + 1e-8
        return (v / n).astype(np.float32)

    def _merge_accessory_pattern_candidates(
        ranked: List[tuple[str, float]],
        query_sig: np.ndarray | None,
        query_hat_prior: float = 0.0,
    ) -> tuple[List[tuple[str, float]], str]:
        if not ranked or query_sig is None or not accessory_pattern_cache:
            return ranked, ""
        scored: List[tuple[str, float]] = []
        for file_name, sig in accessory_pattern_cache.items():
            sim = float(query_sig @ sig)
            if sim >= accessory_pattern_min_score:
                scored.append((file_name, sim))
        if not scored:
            return ranked, ""
        scored.sort(key=lambda x: x[1], reverse=True)
        injected = scored[: max(1, accessory_pattern_max_injected)]
        merged: Dict[str, float] = {}

        def _hat_code_boost(name: str) -> float:
            if accessory_hat_code_boost <= 0.0 or not accessory_hat_code_prefixes:
                return 0.0
            code = filename_to_style_code(name.split("@", 1)[0]).strip().upper()
            if any(code.startswith(prefix) for prefix in accessory_hat_code_prefixes):
                return float(accessory_hat_code_boost)
            return 0.0

        for name, score in ranked:
            merged[name] = max(float(score) + _hat_code_boost(name), merged.get(name, -1e9))
        for file_name, sim in injected:
            base_name = file_name.split("@", 1)[0]
            hat_prior = float(accessory_hat_prior_cache.get(base_name, 0.0))
            seed = (
                accessory_pattern_seed_score_base
                + accessory_pattern_boost_scale * max(0.0, sim)
                + max(0.0, accessory_hat_prior_boost) * hat_prior
                + _hat_code_boost(file_name)
            )
            merged[file_name] = max(merged.get(file_name, -1e9), float(seed))
        hat_injected: List[tuple[str, float]] = []
        if (
            accessory_hat_prior_seed_enabled
            and query_hat_prior >= accessory_hat_prior_query_threshold
            and accessory_hat_prior_cache
        ):
            for file_name, hat_prior in accessory_hat_prior_cache.items():
                prior = float(hat_prior)
                if prior < accessory_hat_prior_seed_min_score:
                    continue
                hat_injected.append((file_name, prior + _hat_code_boost(file_name)))
            hat_injected.sort(key=lambda x: x[1], reverse=True)
            for file_name, prior_score in hat_injected[: max(1, accessory_hat_prior_seed_max_injected)]:
                base_name = file_name.split("@", 1)[0]
                prior = float(accessory_hat_prior_cache.get(base_name, 0.0))
                seed = (
                    accessory_hat_prior_seed_score_base
                    + accessory_hat_prior_seed_boost_scale * max(0.0, prior)
                    + _hat_code_boost(file_name)
                )
                merged[file_name] = max(merged.get(file_name, -1e9), float(seed))
        debug_items = [
            (
                f"{filename_to_style_code(file_name)}:{sim:.3f}/"
                f"{accessory_pattern_seed_score_base + accessory_pattern_boost_scale * max(0.0, sim) + max(0.0, accessory_hat_prior_boost) * float(accessory_hat_prior_cache.get(file_name.split('@', 1)[0], 0.0)) + _hat_code_boost(file_name):.3f}/"
                f"{float(accessory_hat_prior_cache.get(file_name.split('@', 1)[0], 0.0)):.2f}/"
                f"{_hat_code_boost(file_name):.2f}"
            )
            for file_name, sim in injected[:12]
        ]
        if hat_injected:
            debug_items.extend(
                f"hat:{filename_to_style_code(file_name)}:{float(accessory_hat_prior_cache.get(file_name.split('@', 1)[0], 0.0)):.3f}/"
                f"{accessory_hat_prior_seed_score_base + accessory_hat_prior_seed_boost_scale * max(0.0, float(accessory_hat_prior_cache.get(file_name.split('@', 1)[0], 0.0))) + _hat_code_boost(file_name):.3f}"
                for file_name, _prior_score in hat_injected[:6]
            )
        out = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return out, ",".join(debug_items)

    def _extract_phash_bits(path: Path, size: int = 32, keep: int = 8) -> np.ndarray | None:
        try:
            with Image.open(path) as im0:
                arr = np.asarray(im0.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
        except Exception:
            return None
        dct = np.fft.fft2(arr).real
        low = dct[:keep, :keep].copy()
        med = float(np.median(low[1:, 1:])) if keep > 1 else float(np.median(low))
        bits = (low > med).astype(np.uint8).reshape(-1)
        return bits

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

    def _apply_mask_consistency(
        ranked: List[tuple[str, float]],
        query_mask_vec: np.ndarray | None,
    ) -> List[tuple[str, float]]:
        if not ranked or query_mask_vec is None:
            return ranked
        head_n = min(len(ranked), max(1, mask_consistency_apply_topn))
        head = ranked[:head_n]
        tail = ranked[head_n:]
        adjusted: List[tuple[str, float]] = []
        w = max(0.0, float(mask_consistency_weight))
        for name, score in head:
            file_name = name.split("@", 1)[0]
            cvec = fg_mask_cache.get(file_name)
            if cvec is None:
                adjusted.append((name, score))
                continue
            m = float(query_mask_vec @ cvec)  # cosine similarity in [0,1]
            adjusted.append((name, float(score) + w * m))
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted + tail

    def _apply_stripe_consistency(
        ranked: List[tuple[str, float]],
        query_sig: np.ndarray | None,
    ) -> List[tuple[str, float]]:
        if not ranked or query_sig is None:
            return ranked
        head_n = min(len(ranked), max(1, stripe_consistency_apply_topn))
        head = ranked[:head_n]
        tail = ranked[head_n:]
        w = max(0.0, float(stripe_consistency_weight))
        adjusted: List[tuple[str, float]] = []
        for name, score in head:
            file_name = name.split("@", 1)[0]
            cs = stripe_sig_cache.get(file_name)
            if cs is None:
                adjusted.append((name, score))
                continue
            sim = float(query_sig @ cs)
            adjusted.append((name, float(score) + w * sim))
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted + tail

    def _apply_region_color_consistency(
        ranked: List[tuple[str, float]],
        query_sig: np.ndarray | None,
    ) -> List[tuple[str, float]]:
        if not ranked or query_sig is None or not color_sig_cache:
            return ranked
        head_n = min(len(ranked), max(1, region_crop_color_consistency_apply_topn))
        head = ranked[:head_n]
        tail = ranked[head_n:]
        w = max(0.0, float(region_crop_color_consistency_weight))
        adjusted: List[tuple[str, float]] = []
        for name, score in head:
            file_name = name.split("@", 1)[0]
            cs = color_sig_cache.get(file_name)
            if cs is None:
                adjusted.append((name, score))
                continue
            sim = float(query_sig @ cs)
            adjusted.append((name, float(score) + w * max(0.0, sim)))
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted + tail

    def _apply_pattern_consistency(
        ranked: List[tuple[str, float]],
        query_sig: np.ndarray | None,
    ) -> List[tuple[str, float]]:
        if not ranked or query_sig is None:
            return ranked
        head_n = min(len(ranked), max(1, pattern_consistency_apply_topn))
        head = ranked[:head_n]
        tail = ranked[head_n:]
        w = max(0.0, float(pattern_consistency_weight))
        adjusted: List[tuple[str, float]] = []
        for name, score in head:
            file_name = name.split("@", 1)[0]
            cs = pattern_sig_cache.get(file_name)
            if cs is None:
                adjusted.append((name, score))
                continue
            sim = float(query_sig @ cs)
            adjusted.append((name, float(score) + w * sim))
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted + tail

    def _build_pattern_code_boost(
        ranked: List[tuple[str, float]],
        query_sig: np.ndarray | None,
    ) -> Dict[str, float]:
        if (not pattern_code_boost_enabled) or (not ranked) or query_sig is None:
            return {}
        head_n = min(len(ranked), max(1, pattern_code_boost_topn))
        code_best: Dict[str, float] = {}
        for name, _score in ranked[:head_n]:
            file_name = name.split("@", 1)[0]
            cs = pattern_sig_cache.get(file_name)
            if cs is None:
                continue
            sim = float(query_sig @ cs)
            code = filename_to_style_code(file_name)
            prev = code_best.get(code)
            if prev is None or sim > prev:
                code_best[code] = sim
        if not code_best:
            return {}
        max_sim = max(code_best.values()) + 1e-6
        boosts: Dict[str, float] = {}
        for code, sim in code_best.items():
            boosts[str(code).strip().upper()] = float(pattern_code_boost_weight) * max(0.0, sim / max_sim)
        return boosts

    def _apply_checker_consistency(
        ranked: List[tuple[str, float]],
        query_profile: Dict[str, float] | None,
    ) -> tuple[List[tuple[str, float]], Dict[str, float], str]:
        if not ranked or not query_profile:
            return ranked, {}, ""
        q_checker = float(query_profile.get("checker", 0.0))
        if q_checker < checker_query_threshold:
            return ranked, {}, ""
        head_n = min(len(ranked), max(1, checker_apply_topn))
        head = ranked[:head_n]
        tail = ranked[head_n:]
        adjusted: List[tuple[str, float]] = []
        code_best: Dict[str, float] = {}
        debug_items: List[str] = []
        for name, score in head:
            file_name = name.split("@", 1)[0]
            prof = checker_profile_cache.get(file_name)
            if not prof:
                adjusted.append((name, score))
                continue
            c_checker = float(prof.get("checker", 0.0))
            c_stripe = float(prof.get("stripe", 0.0))
            c_bw_mix = float(prof.get("bw_mix", 0.0))
            c_dark_components = int(float(prof.get("dark_components", 0.0)))
            c_light_components = int(float(prof.get("light_components", 0.0)))
            delta = checker_boost_weight * c_checker - checker_stripe_penalty_weight * max(0.0, c_stripe - c_checker)
            adjusted.append((name, float(score) + float(delta)))
            code = filename_to_style_code(file_name)
            prev = code_best.get(code)
            if prev is None or c_checker > prev:
                code_best[code] = c_checker
            if len(debug_items) < 6:
                debug_items.append(
                    f"{code}:{c_checker:.3f}/{c_stripe:.3f}/{c_bw_mix:.3f}/"
                    f"{c_dark_components}x{c_light_components}/{delta:.3f}"
                )
        adjusted.sort(key=lambda x: x[1], reverse=True)
        debug_text = ",".join(debug_items)

        if not code_best:
            return adjusted + tail, {}, debug_text
        max_checker = max(code_best.values()) + 1e-6
        boosts: Dict[str, float] = {}
        for code, strength in code_best.items():
            if strength <= 0.0:
                continue
            boosts[str(code).strip().upper()] = checker_code_boost_weight * (strength / max_checker)
        return adjusted + tail, boosts, debug_text

    def _merge_checker_seed_candidates(
        ranked: List[tuple[str, float]],
        query_profile: Dict[str, float] | None,
    ) -> tuple[List[tuple[str, float]], str]:
        if not ranked or not query_profile or not checker_seed_enabled or not checker_profile_cache:
            return ranked, ""
        q_checker = float(query_profile.get("checker", 0.0))
        q_bw_mix = float(query_profile.get("bw_mix", 0.0))
        if q_checker < checker_query_threshold:
            return ranked, ""
        scored: List[tuple[str, float]] = []
        for file_name, prof in checker_profile_cache.items():
            c_checker = float(prof.get("checker", 0.0))
            if c_checker <= 0.0:
                continue
            c_stripe = float(prof.get("stripe", 0.0))
            c_bw_mix = float(prof.get("bw_mix", 0.0))
            bw_close = 1.0 - min(1.0, abs(q_bw_mix - c_bw_mix))
            stripe_penalty = 1.0 - min(0.45, max(0.0, c_stripe - c_checker) * 1.8)
            score = c_checker * (0.55 + 0.45 * bw_close) * stripe_penalty
            if score >= checker_seed_min_score:
                scored.append((file_name, float(score)))
        if not scored:
            return ranked, ""
        scored.sort(key=lambda x: x[1], reverse=True)
        injected = scored[: max(1, checker_seed_max_injected)]

        merged: Dict[str, float] = {}
        for name, score in ranked:
            merged[name] = max(float(score), merged.get(name, -1e9))
        for file_name, score in injected:
            seed = checker_seed_score_base + checker_seed_boost_scale * max(0.0, score)
            merged[file_name] = max(merged.get(file_name, -1e9), float(seed))

        debug_items = [
            f"{filename_to_style_code(file_name)}:{score:.3f}/"
            f"{checker_seed_score_base + checker_seed_boost_scale * max(0.0, score):.3f}"
            for file_name, score in injected[:12]
        ]
        return sorted(merged.items(), key=lambda x: x[1], reverse=True), ",".join(debug_items)

    def _apply_phash_consistency(
        ranked: List[tuple[str, float]],
        query_bits: np.ndarray | None,
    ) -> List[tuple[str, float]]:
        if not ranked or query_bits is None:
            return ranked
        head_n = min(len(ranked), max(1, phash_apply_topn))
        head = ranked[:head_n]
        tail = ranked[head_n:]
        w = max(0.0, float(phash_boost_weight))
        nbits = max(1, int(query_bits.shape[0]))
        adjusted: List[tuple[str, float]] = []
        for name, score in head:
            file_name = name.split("@", 1)[0]
            cbits = phash_cache.get(file_name)
            if cbits is None or cbits.shape[0] != query_bits.shape[0]:
                adjusted.append((name, score))
                continue
            hdist = int(np.count_nonzero(query_bits != cbits))
            sim = 1.0 - (hdist / float(nbits))
            # Emphasize very near duplicates
            boost = w * (sim ** 2)
            adjusted.append((name, float(score) + float(boost)))
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

    def _local_file_sig(p: Path) -> str:
        st = p.stat()
        return f"{p.name}|{st.st_size}|{int(st.st_mtime)}"

    def _accent_pattern_cache_path() -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(standard_dir))
        return Path("outputs") / f"accent_pattern_cache_{safe}.npz"

    def _dark_motif_cache_path() -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(standard_dir))
        return Path("outputs") / f"dark_motif_cache_{safe}.npz"

    def _sleeve_pattern_cache_path() -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(standard_dir))
        return Path("outputs") / f"sleeve_pattern_cache_{safe}.npz"

    def _sleeve_pair_prior_cache_path() -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(standard_dir))
        return Path("outputs") / f"sleeve_pair_prior_cache_{safe}.npz"

    def _accessory_pattern_cache_path() -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(standard_dir))
        return Path("outputs") / f"accessory_pattern_cache_{safe}.npz"

    def _accessory_hat_prior_cache_path() -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(standard_dir))
        return Path("outputs") / f"accessory_hat_prior_cache_{safe}.npz"

    def _collar_contour_cache_path() -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(standard_dir))
        return Path("outputs") / f"collar_contour_cache_{safe}.npz"

    def _collar_chevron_cache_path() -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(standard_dir))
        return Path("outputs") / f"collar_chevron_cache_{safe}.npz"

    def _load_or_build_accent_pattern_cache(file_names: List[str]) -> Dict[str, np.ndarray]:
        uniq = sorted({n.split("@", 1)[0] for n in file_names})
        files = [standard_dir / n for n in uniq if (standard_dir / n).exists() and (standard_dir / n).is_file()]
        sigs = [_local_file_sig(p) for p in files]
        cache_key = json.dumps(
            {
                "kind": "accent_pattern",
                "version": 8,
                "grid": 12,
                "min_pixels": int(accent_pattern_min_pixels),
                "max_edge": int(accent_pattern_max_edge),
                "pattern": standard_pattern,
                "exts": list(image_exts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_path = _accent_pattern_cache_path()
        if feature_cache_enabled and cache_path.exists():
            try:
                arr = np.load(cache_path, allow_pickle=True)
                if str(arr["cache_key"].item()) == cache_key and list(arr["file_sigs"]) == sigs:
                    cached_names = [str(x) for x in arr["names"]]
                    feats = arr["feats"].astype(np.float32)
                    out = {name: feats[i] for i, name in enumerate(cached_names)}
                    logging.info("accent pattern cache hit: %s (%d items)", cache_path, len(out))
                    return out
            except Exception:
                pass

        t0 = time.perf_counter()
        out: Dict[str, np.ndarray] = {}
        for fp in files:
            sig = _extract_accent_pattern_sig(fp, grid=12)
            if sig is not None:
                out[fp.name] = sig.astype(np.float32)

        if feature_cache_enabled:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            names_arr = np.array(list(out.keys()), dtype=object)
            feats_arr = np.vstack([out[n] for n in out.keys()]).astype(np.float32) if out else np.zeros((0, 1), dtype=np.float32)
            np.savez_compressed(
                cache_path,
                cache_key=np.array([cache_key], dtype=object),
                file_sigs=np.array(sigs, dtype=object),
                names=names_arr,
                feats=feats_arr,
            )
            logging.info("accent pattern cache write: %s", cache_path)
        logging.info("accent pattern cache built: %d items in %.2fs", len(out), time.perf_counter() - t0)
        return out

    def _load_or_build_dark_motif_cache(file_names: List[str]) -> Dict[str, np.ndarray]:
        uniq = sorted({n.split("@", 1)[0] for n in file_names})
        files = [standard_dir / n for n in uniq if (standard_dir / n).exists() and (standard_dir / n).is_file()]
        sigs = [_local_file_sig(p) for p in files]
        cache_key = json.dumps(
            {
                "kind": "dark_motif",
                "version": 3,
                "size": 18,
                "pattern": standard_pattern,
                "exts": list(image_exts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_path = _dark_motif_cache_path()
        if feature_cache_enabled and cache_path.exists():
            try:
                arr = np.load(cache_path, allow_pickle=True)
                if str(arr["cache_key"].item()) == cache_key and list(arr["file_sigs"]) == sigs:
                    cached_names = [str(x) for x in arr["names"]]
                    feats = arr["feats"].astype(np.float32)
                    out = {name: feats[i] for i, name in enumerate(cached_names)}
                    logging.info("dark motif cache hit: %s (%d items)", cache_path, len(out))
                    return out
            except Exception:
                pass

        t0 = time.perf_counter()
        out: Dict[str, np.ndarray] = {}
        for fp in files:
            sig = _extract_dark_motif_sig(fp, size=18)
            if sig is not None:
                out[fp.name] = sig.astype(np.float32)
        if feature_cache_enabled:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            names_arr = np.array(list(out.keys()), dtype=object)
            feats_arr = np.vstack([out[n] for n in out.keys()]).astype(np.float32) if out else np.zeros((0, 1), dtype=np.float32)
            np.savez_compressed(
                cache_path,
                cache_key=np.array([cache_key], dtype=object),
                file_sigs=np.array(sigs, dtype=object),
                names=names_arr,
                feats=feats_arr,
            )
            logging.info("dark motif cache write: %s", cache_path)
        logging.info("dark motif cache built: %d items in %.2fs", len(out), time.perf_counter() - t0)
        return out

    def _load_or_build_sleeve_pattern_cache(file_names: List[str]) -> Dict[str, np.ndarray]:
        uniq = sorted({n.split("@", 1)[0] for n in file_names})
        files = [standard_dir / n for n in uniq if (standard_dir / n).exists() and (standard_dir / n).is_file()]
        sigs = [_local_file_sig(p) for p in files]
        cache_key = json.dumps(
            {
                "kind": "sleeve_pattern",
                "version": 7,
                "size": 32,
                "standard_views": "grid_halves_bands_components_comp_stripes_band_weighted",
                "pattern": standard_pattern,
                "exts": list(image_exts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_path = _sleeve_pattern_cache_path()
        if feature_cache_enabled and cache_path.exists():
            try:
                arr = np.load(cache_path, allow_pickle=True)
                if str(arr["cache_key"].item()) == cache_key and list(arr["file_sigs"]) == sigs:
                    cached_names = [str(x) for x in arr["names"]]
                    feats = arr["feats"].astype(np.float32)
                    out = {name: feats[i] for i, name in enumerate(cached_names)}
                    logging.info("sleeve pattern cache hit: %s (%d items)", cache_path, len(out))
                    return out
            except Exception:
                pass

        t0 = time.perf_counter()
        out: Dict[str, np.ndarray] = {}
        for fp in files:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            for idx, (tag, view) in enumerate(_region_standard_views(img)):
                sig = _extract_sleeve_pattern_sig_from_image(view, size=32)
                if sig is not None:
                    out[f"{fp.name}@s{idx}_{tag}"] = sig.astype(np.float32)
        if feature_cache_enabled:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            names_arr = np.array(list(out.keys()), dtype=object)
            feats_arr = np.vstack([out[n] for n in out.keys()]).astype(np.float32) if out else np.zeros((0, 1), dtype=np.float32)
            np.savez_compressed(
                cache_path,
                cache_key=np.array([cache_key], dtype=object),
                file_sigs=np.array(sigs, dtype=object),
                names=names_arr,
                feats=feats_arr,
            )
            logging.info("sleeve pattern cache write: %s", cache_path)
        logging.info("sleeve pattern cache built: %d items in %.2fs", len(out), time.perf_counter() - t0)
        return out

    def _extract_sleeve_pair_prior(path: Path) -> float:
        try:
            with Image.open(path) as im0:
                im = im0.convert("RGB")
                im.thumbnail((256, 256), Image.Resampling.BILINEAR)
                rgb = np.asarray(im, dtype=np.uint8)
        except Exception:
            return 0.0
        if rgb.ndim != 3 or rgb.shape[0] < 40 or rgb.shape[1] < 40:
            return 0.0
        h, w = rgb.shape[:2]
        arr = rgb.astype(np.float32)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        maxc = arr.max(axis=-1)
        minc = arr.min(axis=-1)
        sat = (maxc - minc) / np.maximum(maxc, 1.0)

        # Ignore top labels; sleeve pieces are usually dark/colored regions on a gray floor.
        fg = ((sat > 0.10) | (gray < 82.0)) & (gray < 245.0)
        fg[: min(int(h * 0.12), 36), :] = False
        min_area = max(40, int(h * w * 0.006))
        comps: List[tuple[int, int, int, int, int]] = []
        if cv2 is not None:
            n_labels, labels, stats, _centers = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
            for label in range(1, int(n_labels)):
                x = int(stats[label, cv2.CC_STAT_LEFT])
                y = int(stats[label, cv2.CC_STAT_TOP])
                ww = int(stats[label, cv2.CC_STAT_WIDTH])
                hh = int(stats[label, cv2.CC_STAT_HEIGHT])
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < min_area or ww < 12 or hh < 18:
                    continue
                comps.append((x, y, ww, hh, area))
        else:
            seen = np.zeros(fg.shape, dtype=bool)
            for sy, sx in zip(*np.where(fg & ~seen)):
                stack = [(int(sy), int(sx))]
                seen[sy, sx] = True
                pts: List[tuple[int, int]] = []
                while stack:
                    cy, cx = stack.pop()
                    pts.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        if seen[ny, nx] or not bool(fg[ny, nx]):
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
                if len(pts) < min_area:
                    continue
                ys = [p[0] for p in pts]
                xs = [p[1] for p in pts]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                ww, hh = x1 - x0 + 1, y1 - y0 + 1
                if ww < 12 or hh < 18:
                    continue
                comps.append((x0, y0, ww, hh, len(pts)))

        comps.sort(key=lambda item: item[4], reverse=True)
        comps = comps[:6]
        pair_score = 0.0
        for i in range(len(comps)):
            x1, y1, w1, h1, a1 = comps[i]
            c1 = x1 + w1 / 2.0
            for j in range(i + 1, len(comps)):
                x2, y2, w2, h2, a2 = comps[j]
                c2 = x2 + w2 / 2.0
                if abs(c1 - c2) < max(16.0, 0.16 * w):
                    continue
                height_sim = min(h1, h2) / max(h1, h2, 1)
                width_sim = min(w1, w2) / max(w1, w2, 1)
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2)) / max(1, min(h1, h2))
                vertical = min(1.0, ((h1 / max(1, w1)) + (h2 / max(1, w2))) / 3.2)
                score = height_sim * width_sim * y_overlap * vertical
                pair_score = max(pair_score, float(score))

        neutral_light = ((gray > 178.0) & (sat < 0.16)).astype(np.float32)
        dark = (gray < 82.0).astype(np.float32)
        row_light = neutral_light.mean(axis=1)
        row_dark = dark.mean(axis=1)

        def _run_count(flags: np.ndarray) -> float:
            vals = flags.astype(np.uint8)
            if vals.size == 0:
                return 0.0
            starts = vals.copy()
            starts[1:] = np.maximum(0, vals[1:] - vals[:-1])
            return float(starts.sum())

        light_runs = _run_count(row_light > 0.10)
        dark_runs = _run_count(row_dark > 0.12)
        stripe_score = min(1.0, light_runs / 4.0) * min(1.0, dark_runs / 4.0)
        return float(max(0.0, min(1.0, 0.65 * pair_score + 0.35 * stripe_score)))

    def _load_or_build_sleeve_pair_prior_cache(file_names: List[str]) -> Dict[str, float]:
        uniq = sorted({n.split("@", 1)[0] for n in file_names})
        files = [standard_dir / n for n in uniq if (standard_dir / n).exists() and (standard_dir / n).is_file()]
        sigs = [_local_file_sig(p) for p in files]
        cache_key = json.dumps(
            {
                "kind": "sleeve_pair_prior",
                "version": 1,
                "pattern": standard_pattern,
                "exts": list(image_exts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_path = _sleeve_pair_prior_cache_path()
        if feature_cache_enabled and cache_path.exists():
            try:
                arr = np.load(cache_path, allow_pickle=True)
                if str(arr["cache_key"].item()) == cache_key and list(arr["file_sigs"]) == sigs:
                    cached_names = [str(x) for x in arr["names"]]
                    scores = arr["scores"].astype(np.float32)
                    out = {name: float(scores[i]) for i, name in enumerate(cached_names)}
                    logging.info("sleeve pair prior cache hit: %s (%d items)", cache_path, len(out))
                    return out
            except Exception:
                pass
        t0 = time.perf_counter()
        out: Dict[str, float] = {}
        for fp in files:
            score = _extract_sleeve_pair_prior(fp)
            if score > 0.0:
                out[fp.name] = float(score)
        if feature_cache_enabled:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            names_arr = np.array(list(out.keys()), dtype=object)
            scores_arr = np.array([out[n] for n in out.keys()], dtype=np.float32)
            np.savez_compressed(
                cache_path,
                cache_key=np.array([cache_key], dtype=object),
                file_sigs=np.array(sigs, dtype=object),
                names=names_arr,
                scores=scores_arr,
            )
            logging.info("sleeve pair prior cache write: %s", cache_path)
        logging.info("sleeve pair prior cache built: %d items in %.2fs", len(out), time.perf_counter() - t0)
        return out

    def _load_or_build_accessory_pattern_cache(file_names: List[str]) -> Dict[str, np.ndarray]:
        uniq = sorted({n.split("@", 1)[0] for n in file_names})
        files = [standard_dir / n for n in uniq if (standard_dir / n).exists() and (standard_dir / n).is_file()]
        sigs = [_local_file_sig(p) for p in files]
        cache_key = json.dumps(
            {
                "kind": "accessory_pattern",
                "version": 4,
                "size": 48,
                "pattern": standard_pattern,
                "exts": list(image_exts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_path = _accessory_pattern_cache_path()
        if feature_cache_enabled and cache_path.exists():
            try:
                arr = np.load(cache_path, allow_pickle=True)
                if str(arr["cache_key"].item()) == cache_key and list(arr["file_sigs"]) == sigs:
                    cached_names = [str(x) for x in arr["names"]]
                    feats = arr["feats"].astype(np.float32)
                    out = {name: feats[i] for i, name in enumerate(cached_names)}
                    logging.info("accessory pattern cache hit: %s (%d items)", cache_path, len(out))
                    return out
            except Exception:
                pass

        t0 = time.perf_counter()
        out: Dict[str, np.ndarray] = {}
        for fp in files:
            sig = _extract_accessory_pattern_sig(fp, size=48)
            if sig is not None:
                out[fp.name] = sig.astype(np.float32)
        if feature_cache_enabled:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            names_arr = np.array(list(out.keys()), dtype=object)
            feats_arr = np.vstack([out[n] for n in out.keys()]).astype(np.float32) if out else np.zeros((0, 1), dtype=np.float32)
            np.savez_compressed(
                cache_path,
                cache_key=np.array([cache_key], dtype=object),
                file_sigs=np.array(sigs, dtype=object),
                names=names_arr,
                feats=feats_arr,
            )
            logging.info("accessory pattern cache write: %s", cache_path)
        logging.info("accessory pattern cache built: %d items in %.2fs", len(out), time.perf_counter() - t0)
        return out

    def _extract_accessory_hat_prior(path: Path) -> float:
        try:
            with Image.open(path) as im0:
                im = im0.convert("RGB")
                im.thumbnail((256, 256), Image.Resampling.BILINEAR)
                rgb = np.asarray(im, dtype=np.uint8)
        except Exception:
            return 0.0
        if rgb.ndim != 3 or rgb.shape[0] < 40 or rgb.shape[1] < 40:
            return 0.0
        h, w = rgb.shape[:2]
        arr = rgb.astype(np.float32)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        fg = (gray < 235.0) & (np.any(rgb < 245, axis=-1))
        fg[: min(int(h * 0.08), 32), :] = False
        if int(fg.sum()) < max(48, int(h * w * 0.004)):
            return 0.0
        ys, xs = np.where(fg)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        crop = fg[y0 : y1 + 1, x0 : x1 + 1]
        if crop.shape[0] < 24 or crop.shape[1] < 24:
            return 0.0
        mask = np.asarray(
            Image.fromarray((crop.astype(np.uint8) * 255), mode="L").resize((48, 48), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        bits = mask > 0.35
        top = bits[:20, :]
        lower = bits[18:, :]
        top_mass = float(top.mean())
        lower_mass = float(lower.mean())
        col_strength = lower.mean(axis=0)
        active_cols = np.where(col_strength > 0.10)[0]
        cord_cols = int(active_cols.size)
        cord_gap = 0.0
        if active_cols.size >= 2:
            cord_gap = float(active_cols.max() - active_cols.min()) / 48.0
        sparse_cords = min(1.0, cord_cols / 10.0) * max(0.0, 1.0 - max(0.0, lower_mass - 0.35) * 2.0)
        body_score = min(1.0, top_mass * 4.0)
        aspect = float((x1 - x0 + 1) / max(1, (y1 - y0 + 1)))
        aspect_score = 1.0 - min(1.0, abs(aspect - 1.0) / 1.4)
        return float(max(0.0, min(1.0, 0.40 * body_score + 0.35 * sparse_cords + 0.15 * cord_gap + 0.10 * aspect_score)))

    def _load_or_build_accessory_hat_prior_cache(file_names: List[str]) -> Dict[str, float]:
        uniq = sorted({n.split("@", 1)[0] for n in file_names})
        files = [standard_dir / n for n in uniq if (standard_dir / n).exists() and (standard_dir / n).is_file()]
        sigs = [_local_file_sig(p) for p in files]
        cache_key = json.dumps(
            {
                "kind": "accessory_hat_prior",
                "version": 1,
                "pattern": standard_pattern,
                "exts": list(image_exts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_path = _accessory_hat_prior_cache_path()
        if feature_cache_enabled and cache_path.exists():
            try:
                arr = np.load(cache_path, allow_pickle=True)
                if str(arr["cache_key"].item()) == cache_key and list(arr["file_sigs"]) == sigs:
                    cached_names = [str(x) for x in arr["names"]]
                    scores = arr["scores"].astype(np.float32)
                    out = {name: float(scores[i]) for i, name in enumerate(cached_names)}
                    logging.info("accessory hat prior cache hit: %s (%d items)", cache_path, len(out))
                    return out
            except Exception:
                pass
        t0 = time.perf_counter()
        out: Dict[str, float] = {}
        for fp in files:
            score = _extract_accessory_hat_prior(fp)
            if score > 0.0:
                out[fp.name] = score
        if feature_cache_enabled:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            names_arr = np.array(list(out.keys()), dtype=object)
            scores_arr = np.array([out[n] for n in out.keys()], dtype=np.float32)
            np.savez_compressed(
                cache_path,
                cache_key=np.array([cache_key], dtype=object),
                file_sigs=np.array(sigs, dtype=object),
                names=names_arr,
                scores=scores_arr,
            )
            logging.info("accessory hat prior cache write: %s", cache_path)
        logging.info("accessory hat prior cache built: %d items in %.2fs", len(out), time.perf_counter() - t0)
        return out

    def _load_or_build_collar_contour_cache(file_names: List[str]) -> Dict[str, np.ndarray]:
        uniq = sorted({n.split("@", 1)[0] for n in file_names})
        files = [standard_dir / n for n in uniq if (standard_dir / n).exists() and (standard_dir / n).is_file()]
        sigs = [_local_file_sig(p) for p in files]
        cache_key = json.dumps(
            {
                "kind": "collar_contour",
                "version": 7,
                "size": int(collar_contour_size),
                "standard_views": "collar_focus_components_topcomp_vline_lineedge_contourfallback_mid_strip",
                "pattern": standard_pattern,
                "exts": list(image_exts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_path = _collar_contour_cache_path()
        if feature_cache_enabled and cache_path.exists():
            try:
                arr = np.load(cache_path, allow_pickle=True)
                if str(arr["cache_key"].item()) == cache_key and list(arr["file_sigs"]) == sigs:
                    cached_names = [str(x) for x in arr["names"]]
                    feats = arr["feats"].astype(np.float32)
                    out = {name: feats[i] for i, name in enumerate(cached_names)}
                    logging.info("collar contour cache hit: %s (%d items)", cache_path, len(out))
                    return out
            except Exception:
                pass
        t0 = time.perf_counter()
        out: Dict[str, np.ndarray] = {}
        collar_tags = {
            "top",
            "top_narrow",
            "upper_band",
            "upper_narrow_band",
            "top_left_band",
            "top_right_band",
            "collar_left_focus",
            "collar_right_focus",
            "collar_center_bridge",
            "collar_right_mid",
            "collar_right_lower",
            "collar_left_mid_strip",
            "collar_lower_strip",
        }
        for fp in files:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            for idx, (tag, view) in enumerate(_region_standard_views(img, max_component_views=4)):
                if not ((tag in collar_tags) or tag.startswith("comp") or tag.startswith("top_comp")):
                    continue
                for sig_idx, sig in enumerate(_extract_collar_contour_sigs_from_image(view, size=collar_contour_size)):
                    out[f"{fp.name}@c{idx}_{tag}_s{sig_idx}"] = sig.astype(np.float32)
        if feature_cache_enabled:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            names_arr = np.array(list(out.keys()), dtype=object)
            feats_arr = np.vstack([out[n] for n in out.keys()]).astype(np.float32) if out else np.zeros((0, 1), dtype=np.float32)
            np.savez_compressed(
                cache_path,
                cache_key=np.array([cache_key], dtype=object),
                file_sigs=np.array(sigs, dtype=object),
                names=names_arr,
                feats=feats_arr,
            )
            logging.info("collar contour cache write: %s", cache_path)
        logging.info("collar contour cache built: %d items in %.2fs", len(out), time.perf_counter() - t0)
        return out

    def _load_or_build_collar_chevron_cache(file_names: List[str]) -> Dict[str, float]:
        if not collar_chevron_enabled or cv2 is None:
            return {}
        uniq = sorted({n.split("@", 1)[0] for n in file_names})
        files = [standard_dir / n for n in uniq if (standard_dir / n).exists() and (standard_dir / n).is_file()]
        sigs = [_local_file_sig(p) for p in files]
        cache_key = json.dumps(
            {
                "kind": "collar_chevron",
                "version": 6,
                "standard_views": "collar_focus_components_topcomp_hough_mid_strip",
                "pattern": standard_pattern,
                "exts": list(image_exts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_path = _collar_chevron_cache_path()
        if feature_cache_enabled and cache_path.exists():
            try:
                arr = np.load(cache_path, allow_pickle=True)
                if str(arr["cache_key"].item()) == cache_key and list(arr["file_sigs"]) == sigs:
                    cached_names = [str(x) for x in arr["names"]]
                    scores = arr["scores"].astype(np.float32)
                    out = {name: float(scores[i]) for i, name in enumerate(cached_names)}
                    logging.info("collar chevron cache hit: %s (%d items)", cache_path, len(out))
                    return out
            except Exception:
                pass
        t0 = time.perf_counter()
        out: Dict[str, float] = {}
        collar_tags = {
            "top",
            "top_narrow",
            "upper_band",
            "upper_narrow_band",
            "top_left_band",
            "top_right_band",
            "collar_left_focus",
            "collar_right_focus",
            "collar_center_bridge",
            "collar_right_mid",
            "collar_right_lower",
            "collar_left_mid_strip",
            "collar_lower_strip",
        }
        for fp in files:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            for idx, (tag, view) in enumerate(_region_standard_views(img, max_component_views=4)):
                if not ((tag in collar_tags) or tag.startswith("comp") or tag.startswith("top_comp")):
                    continue
                score = _extract_collar_chevron_score_from_image(view)
                if score > 0.0:
                    out[fp.name] = max(float(out.get(fp.name, 0.0)), float(score))
        if feature_cache_enabled:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            names_arr = np.array(list(out.keys()), dtype=object)
            scores_arr = np.array([out[n] for n in out.keys()], dtype=np.float32)
            np.savez_compressed(
                cache_path,
                cache_key=np.array([cache_key], dtype=object),
                file_sigs=np.array(sigs, dtype=object),
                names=names_arr,
                scores=scores_arr,
            )
            logging.info("collar chevron cache write: %s", cache_path)
        logging.info("collar chevron cache built: %d items in %.2fs", len(out), time.perf_counter() - t0)
        return out

    fg_shape_cache: Dict[str, tuple[float, float]] = {}
    fg_mask_cache: Dict[str, np.ndarray] = {}
    color_sig_cache: Dict[str, np.ndarray] = {}
    stripe_sig_cache: Dict[str, np.ndarray] = {}
    pattern_sig_cache: Dict[str, np.ndarray] = {}
    checker_profile_cache: Dict[str, Dict[str, float]] = {}
    accent_pattern_cache: Dict[str, np.ndarray] = {}
    dark_motif_cache: Dict[str, np.ndarray] = {}
    collar_contour_cache: Dict[str, np.ndarray] = {}
    collar_chevron_cache: Dict[str, float] = {}
    sleeve_pattern_cache: Dict[str, np.ndarray] = {}
    sleeve_pair_prior_cache: Dict[str, float] = {}
    accessory_pattern_cache: Dict[str, np.ndarray] = {}
    accessory_hat_prior_cache: Dict[str, float] = {}
    phash_cache: Dict[str, np.ndarray] = {}
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
    if mask_consistency_enabled:
        uniq = sorted({n.split("@", 1)[0] for n in names})
        for file_name in uniq:
            fp = standard_dir / file_name
            if not fp.exists() or not fp.is_file():
                continue
            mv = _extract_fg_mask_vec(fp, size=64)
            if mv is not None:
                fg_mask_cache[file_name] = mv
        logging.info("api preloaded mask cache: %d", len(fg_mask_cache))
    if region_crop_color_consistency_enabled:
        uniq = sorted({n.split("@", 1)[0] for n in names})
        for file_name in uniq:
            fp = standard_dir / file_name
            if not fp.exists() or not fp.is_file():
                continue
            cv = _extract_color_sig(fp)
            if cv is not None:
                color_sig_cache[file_name] = cv
        logging.info("api preloaded color cache: %d", len(color_sig_cache))
    if stripe_consistency_enabled:
        uniq = sorted({n.split("@", 1)[0] for n in names})
        for file_name in uniq:
            fp = standard_dir / file_name
            if not fp.exists() or not fp.is_file():
                continue
            sv = _extract_stripe_sig(fp, keep=24)
            if sv is not None:
                stripe_sig_cache[file_name] = sv
        logging.info("api preloaded stripe cache: %d", len(stripe_sig_cache))
    if pattern_consistency_enabled or strict_small_pattern_region_order_enabled or strict_small_pattern_region_rescue_enabled:
        uniq = sorted({n.split("@", 1)[0] for n in names})
        for file_name in uniq:
            fp = standard_dir / file_name
            if not fp.exists() or not fp.is_file():
                continue
            pv = _extract_pattern_sig(fp, size=14)
            if pv is not None:
                pattern_sig_cache[file_name] = pv
        logging.info("api preloaded pattern cache: %d", len(pattern_sig_cache))
    if checker_consistency_enabled:
        uniq = sorted({n.split("@", 1)[0] for n in names})
        for file_name in uniq:
            fp = standard_dir / file_name
            if not fp.exists() or not fp.is_file():
                continue
            cp = _extract_checker_profile(fp, grid=10)
            if cp is not None:
                checker_profile_cache[file_name] = cp
        logging.info("api preloaded checker cache: %d", len(checker_profile_cache))
    if accent_pattern_enabled:
        accent_pattern_cache = _load_or_build_accent_pattern_cache(names)
        logging.info("api preloaded accent pattern cache: %d", len(accent_pattern_cache))
    if strict_small_dark_motif_rescue_enabled:
        dark_motif_cache = _load_or_build_dark_motif_cache(names)
        logging.info("api preloaded dark motif cache: %d", len(dark_motif_cache))
    if collar_contour_enabled:
        collar_contour_cache = _load_or_build_collar_contour_cache(names)
        logging.info("api preloaded collar contour cache: %d", len(collar_contour_cache))
        if collar_chevron_enabled:
            collar_chevron_cache = _load_or_build_collar_chevron_cache(names)
            logging.info("api preloaded collar chevron cache: %d", len(collar_chevron_cache))
    if sleeve_pattern_enabled:
        sleeve_pattern_cache = _load_or_build_sleeve_pattern_cache(names)
        logging.info("api preloaded sleeve pattern cache: %d", len(sleeve_pattern_cache))
        if sleeve_pair_prior_boost > 0.0:
            sleeve_pair_prior_cache = _load_or_build_sleeve_pair_prior_cache(names)
            logging.info("api preloaded sleeve pair prior cache: %d", len(sleeve_pair_prior_cache))
    if accessory_pattern_enabled:
        accessory_pattern_cache = _load_or_build_accessory_pattern_cache(names)
        logging.info("api preloaded accessory pattern cache: %d", len(accessory_pattern_cache))
        if accessory_hat_prior_boost > 0.0:
            accessory_hat_prior_cache = _load_or_build_accessory_hat_prior_cache(names)
            logging.info("api preloaded accessory hat prior cache: %d", len(accessory_hat_prior_cache))
    if phash_enabled:
        uniq = sorted({n.split("@", 1)[0] for n in names})
        for file_name in uniq:
            fp = standard_dir / file_name
            if not fp.exists() or not fp.is_file():
                continue
            bits = _extract_phash_bits(fp, size=32, keep=8)
            if bits is not None:
                phash_cache[file_name] = bits
        logging.info("api preloaded phash cache: %d", len(phash_cache))

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

    def _catalog_mobile_page(
        initial_type: str,
        server_permissions: List[str] | None = None,
        server_user_id: str = "",
    ) -> str:
        safe_type = "color" if str(initial_type or "").strip().lower() == "color" else "product"
        permissions_json = json.dumps(server_permissions or [], ensure_ascii=False)
        user_id_json = json.dumps(str(server_user_id or "").strip(), ensure_ascii=False)
        return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>产品库</title>
  <script src="https://res.wx.qq.com/open/js/jweixin-1.6.0.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #f5f6f8; color: #111827; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    .app { width: 100%; min-height: 100vh; background: #f5f6f8; }
    .top { position: sticky; top: 0; z-index: 20; background: rgba(255,255,255,.98); backdrop-filter: blur(12px); padding: max(12px, env(safe-area-inset-top)) 14px 10px; border-bottom: 1px solid #edf0f3; }
    .head { display: none; }
    h1 { margin: 0; font-size: 20px; line-height: 1.2; }
    .status { color: #64748b; font-size: 12px; min-height: 18px; text-align: right; }
    .tabs { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }
    .library-tabs { display: none; }
    .app-tabs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0; width: min(540px, 100%); margin: 0 auto 14px; padding: 3px; border-radius: 999px; background: #eef0f4; box-shadow: inset 0 1px 2px rgba(15,23,42,.08); }
    .app-tab { min-height: 44px; border-radius: 999px; background: transparent; color: #7b8798; border: 0; font-size: 15px; font-weight: 800; }
    .app-tab.active { background: #fff; color: #1f2937; box-shadow: 0 2px 8px rgba(15,23,42,.14); }
    body.product-image-mode .top { padding-bottom: 0; }
    body.product-image-mode .app-tabs { margin-bottom: 0; }
    body.product-image-mode .body { padding-top: 0; }
    body.product-image-mode .image-search-card { margin-top: 0; }
    .tab { min-height: 40px; border: 1px solid #d1d5db; border-radius: 8px; background: #fff; color: #334155; font-size: 15px; font-weight: 700; }
    .tab.active { background: #0f172a; color: #fff; border-color: #0f172a; }
    .search { display: grid; grid-template-columns: minmax(0, 1fr) 76px 70px; gap: 8px; align-items: center; }
    input, select, textarea, button { font: inherit; }
    input, select, textarea { width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; background: #f5f7fa; color: #111827; padding: 10px 11px; min-height: 42px; }
    textarea { resize: vertical; min-height: 72px; }
    button { border: 0; border-radius: 6px; min-height: 42px; padding: 0 12px; background: #1677d2; color: #fff; font-weight: 700; }
    button.secondary { background: #fff; color: #111827; border: 1px solid #e5e7eb; }
    button.danger { background: #fff1f2; color: #be123c; border: 1px solid #fecdd3; }
    button:disabled { opacity: .45; }
    button.loading { position: relative; color: transparent !important; pointer-events: none; }
    button.loading::after { content: ""; position: absolute; left: 50%; top: 50%; width: 18px; height: 18px; margin: -9px 0 0 -9px; border-radius: 50%; border: 2px solid rgba(255,255,255,.55); border-top-color: #fff; animation: spin .8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .body { padding: 12px 14px calc(28px + env(safe-area-inset-bottom)); }
    .panel { display: none; }
    .panel.active { display: block; }
    .list { display: grid; gap: 10px; }
    .product-mode-tabs { display: none; }
    .filter-tags { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 0; }
    .filter-summary { color: #64748b; font-size: 12px; min-height: 18px; padding: 0 2px; }
    .filter-section { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; width: 100%; }
    .filter-label { min-width: 38px; color: #64748b; font-size: 12px; font-weight: 700; }
    .filter-chip { min-height: 28px; padding: 0 9px; border-radius: 999px; border: 1px solid #c7d2fe; background: #eef2ff; color: #3730a3; font-size: 12px; }
    .filter-chip.active { background: #3730a3; color: #fff; border-color: #3730a3; }
    .personal-folder-bar { display: flex; gap: 6px; overflow-x: auto; padding: 8px 0 0; }
    .personal-folder-chip { flex: none; min-height: 30px; border-radius: 999px; padding: 0 11px; background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe; font-size: 12px; }
    .personal-folder-chip.active { background: #0f172a; color: #fff; border-color: #0f172a; }
    .filter-modal { position: fixed; inset: 0; z-index: 120; display: none; align-items: flex-end; background: rgba(15,23,42,.48); }
    .filter-modal.open { display: flex; }
    .filter-sheet { width: 100%; max-height: 78vh; overflow: auto; background: #fff; border-radius: 24px 24px 0 0; padding: 20px 18px calc(18px + env(safe-area-inset-bottom)); box-shadow: 0 -18px 48px rgba(15,23,42,.18); }
    .filter-sheet-head { position: relative; display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 18px; }
    .filter-sheet-title { font-size: 22px; font-weight: 900; }
    .icon-btn { position: absolute; right: 0; width: 40px; min-width: 40px; min-height: 40px; padding: 0; border-radius: 999px; background: transparent; color: #555; font-size: 30px; line-height: 1; }
    .filter-group { margin-bottom: 18px; }
    .filter-group-title { font-size: 16px; font-weight: 900; margin-bottom: 10px; }
    .filter-options { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .filter-option { min-height: 44px; border-radius: 6px; background: #f4f4f5; color: #666; font-weight: 800; border: 1px solid transparent; }
    .filter-option.active { background: #e8f3ff; color: #0b77d8; border-color: #bfdbfe; }
    .filter-actions-bar { position: sticky; bottom: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding-top: 14px; background: linear-gradient(180deg, rgba(255,255,255,.72), #fff 30%); }
    .filter-actions-bar button { min-height: 52px; border-radius: 999px; font-size: 17px; }
    .filter-actions-bar .reset { background: #f4f4f5; color: #0b77d8; }
    .filter-actions-bar .apply { background: #1683df; color: #fff; }
    .card { background: #fff; border: 1px solid #edf0f3; border-radius: 8px; padding: 10px; box-shadow: 0 2px 8px rgba(15,23,42,.03); }
    .product-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
    .product-tile { min-width: 0; padding: 0; overflow: hidden; }
    .product-tile .thumb { width: 100%; height: auto; aspect-ratio: 1 / 1; border-radius: 0; }
    .product-tile-body { padding: 5px 6px 6px; }
    .product-tile .title { font-size: 12px; line-height: 1.2; margin-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .product-tile .muted { font-size: 11px; }
    .product-tile .tags { display: none; }
    .load-more { padding: 12px 0; color: #64748b; font-size: 12px; text-align: center; }
    .product { display: grid; grid-template-columns: 92px minmax(0,1fr); gap: 10px; }
    .thumb { width: 92px; height: 92px; border-radius: 8px; object-fit: cover; background: #e5e7eb; }
    .title { font-weight: 800; font-size: 16px; margin-bottom: 4px; word-break: break-all; }
    .muted { color: #64748b; font-size: 12px; }
    .tags { display: flex; flex-wrap: wrap; gap: 4px; margin: 7px 0; }
    .tag { border-radius: 4px; background: #eef2ff; color: #3730a3; padding: 2px 6px; font-size: 11px; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    .actions button { min-height: 34px; font-size: 13px; padding: 0 10px; }
    .form { display: grid; gap: 10px; margin-bottom: 12px; background: #fff; border-radius: 8px; padding: 12px; border: 1px solid #edf0f3; }
    .form-title { position: relative; font-weight: 800; margin: 0 0 2px; padding-left: 10px; }
    .form-title::before { content: ""; position: absolute; left: 0; top: 3px; width: 4px; height: 16px; border-radius: 3px; background: #1677d2; }
    .review { display: grid; gap: 10px; margin: 12px 0; }
    .review-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .review-title { font-weight: 800; font-size: 15px; }
    .review-list { display: grid; gap: 10px; }
    .review-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 10px; }
    .review-img { width: 92px; height: 92px; border-radius: 8px; object-fit: cover; background: #e5e7eb; }
    .review-img-btn { border: 0; background: transparent; padding: 0; min-height: auto; border-radius: 8px; }
    .review-fields { display: grid; gap: 7px; }
    .review-fields input { min-height: 36px; padding: 7px 9px; font-size: 14px; }
    .review-check { display: flex; align-items: center; gap: 6px; font-size: 13px; color: #334155; }
    .review-check input { width: 18px; min-height: 18px; }
    .review-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .bulk-fields { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .bulk-note { color: #64748b; font-size: 12px; line-height: 1.4; }
    .edit-select-row { display: grid; grid-template-columns: minmax(0, 1fr); gap: 6px; }
    .tag-edit { display: grid; gap: 7px; margin-top: 8px; }
    .tag-edit-row { display: grid; grid-template-columns: 44px minmax(0,1fr); align-items: center; gap: 7px; }
    .tag-edit-row label { color: #64748b; font-size: 12px; font-weight: 700; }
    .tag-edit-row input { min-height: 34px; padding: 6px 9px; font-size: 13px; }
    .modal { position: fixed; inset: 0; background: rgba(0,0,0,.42); display: none; align-items: center; justify-content: center; padding: 14px; z-index: 99; }
    .modal.open { display: flex; }
    .modal-panel { width: min(680px, 100%); max-height: 90vh; overflow: auto; background: #fff; border-radius: 8px; padding: 12px; }
    .modal-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 10px; }
    .modal-head > div:first-child { min-width: 0; flex: 1; }
    .modal-close-btn { flex: none; width: 42px; min-width: 42px; min-height: 42px; padding: 0; border-radius: 10px; border: 1px solid #e5e7eb; background: #fff; color: #111827; font-size: 30px; line-height: 1; font-weight: 700; display: inline-grid; place-items: center; }
    .gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(128px, 1fr)); gap: 10px; }
    .gallery-item img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 8px; background: #e5e7eb; }
    .gallery-item.preview-full { grid-column: 1 / -1; }
    .gallery-item.preview-full img { aspect-ratio: auto; max-height: 72vh; object-fit: contain; }
    .gallery-caption { margin-top: 4px; color: #64748b; font-size: 11px; word-break: break-all; }
    .gallery-item.selectable { position: relative; border: 2px solid transparent; border-radius: 10px; padding: 2px; }
    .gallery-item.selectable.selected { border-color: #1677d2; background: #e8f3ff; }
    .gallery-check { position: absolute; right: 8px; top: 8px; width: 26px; height: 26px; border-radius: 999px; display: grid; place-items: center; background: rgba(15,23,42,.72); color: #fff; font-size: 16px; font-weight: 900; }
    .gallery-item.selected .gallery-check { background: #1677d2; }
    .gallery-delete { position: absolute; right: 8px; top: 8px; width: 28px; height: 28px; min-height: 28px; padding: 0; border-radius: 999px; background: rgba(190,18,60,.92); color: #fff; font-size: 20px; line-height: 1; display: grid; place-items: center; }
    .zoom-modal { position: fixed; inset: 0; z-index: 180; display: none; align-items: center; justify-content: center; background: rgba(0,0,0,.88); padding: 16px; }
    .zoom-modal.open { display: flex; }
    .zoom-img { max-width: 100%; max-height: 88vh; object-fit: contain; border-radius: 8px; background: #111827; }
    .zoom-original { position: fixed; right: 14px; bottom: max(14px, env(safe-area-inset-bottom)); min-height: 38px; padding: 0 14px; border-radius: 999px; display: inline-flex; align-items: center; justify-content: center; background: rgba(255,255,255,.92); color: #0f172a; font-size: 14px; font-weight: 800; text-decoration: none; }
    .zoom-close { position: fixed; right: 14px; top: max(14px, env(safe-area-inset-top)); width: 42px; min-width: 42px; min-height: 42px; padding: 0; border-radius: 999px; background: rgba(255,255,255,.92); color: #111827; font-size: 30px; line-height: 1; }
    .personal-btn { min-height: 40px; padding: 0 14px; border: 0; border-radius: 8px; background: #e8f3ff; color: #0b77d8; font-weight: 800; white-space: nowrap; }
    .personal-btn.added { background: #eef6ff; color: #0f5fa8; }
    .personal-btn.remove { background: #fff1f2; color: #be123c; }
    .personal-sheet-row { display: grid; gap: 8px; margin-bottom: 10px; }
    .personal-sheet-label { color: #64748b; font-size: 12px; font-weight: 800; }
    .float-actions { position: fixed; right: 14px; bottom: calc(24px + env(safe-area-inset-bottom)); z-index: 70; display: grid; gap: 10px; }
    .float-action { width: 54px; min-height: 54px; padding: 6px 4px; border-radius: 14px; border: 1px solid #dbeafe; background: #fff; color: #0b77d8; box-shadow: 0 8px 22px rgba(15,23,42,.16); font-size: 22px; line-height: 1; display: grid; place-items: center; }
    .float-action span { display: block; margin-top: 2px; font-size: 11px; font-weight: 800; color: #334155; }
    .float-action.active { background: #e8f3ff; color: #0b77d8; border-color: #bfdbfe; }
    .float-action.return { background: #1677d2; color: #fff; border-color: #1677d2; }
    .float-action.return span { color: #fff; }
    .grid3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; }
    .swatch { width: 58px; min-width: 58px; height: 58px; border-radius: 8px; border: 1px solid rgba(15,23,42,.14); }
    .color-row { display: flex; gap: 10px; align-items: center; }
    .color-mode-tabs { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }
    .color-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .color-name-builder { display: grid; grid-template-columns: 1fr 86px 1fr; gap: 8px; }
    .color-status { padding: 6px 9px; border-radius: 8px; background: #eef6ff; color: #1e3a8a; font-size: 11px; line-height: 1.35; }
    .color-status.hidden { display: none; }
    .color-status.err { background: #fee2e2; color: #b91c1c; }
    .color-match-list { display: grid; gap: 6px; user-select: none; -webkit-user-select: none; -webkit-touch-callout: none; }
    .color-match-list * { user-select: none; -webkit-user-select: none; -webkit-touch-callout: none; }
    .color-match-item { border-radius: 7px; padding: 7px 10px; border: 1px solid rgba(15,23,42,.12); font-size: 14px; line-height: 1.35; }
    .color-match-item .title { font-size: 15px; line-height: 1.25; font-weight: 700; margin-bottom: 2px; }
    .color-match-item.compare-target { outline: 3px solid rgba(95,216,233,.95); outline-offset: 2px; }
    body.color-h5 { background: #080b16; color: #fff; overscroll-behavior: none; }
    body.color-h5 .app { background: #080b16; color: #fff; min-height: 100vh; }
    body.color-h5 .top { display: none; }
    body.color-h5 .body { padding: 0 0 calc(118px + env(safe-area-inset-bottom)); min-height: 100vh; background: #080b16; }
    .color-home { display: none; min-height: 100vh; padding: 22px 12px 0; color: #f8fafc; font-family: -apple-system,BlinkMacSystemFont,"PingFang SC","Helvetica Neue",Arial,sans-serif; font-weight: 400; }
    body.color-h5 .color-home { display: block; }
    body.color-h5 #colorPanel { display: block; }
    body.color-h5 #colorPanel > .form,
    body.color-h5 #colorPanel > .list { display: none; }
    .color-screen { display: none; min-height: calc(100vh - 136px); }
    .color-screen.active { display: block; }
    .color-screen-title { display: none; }
    .meter-preview-card { border-radius: 6px; background: #202435; padding: 10px; margin: 0 0 10px; touch-action: none; overscroll-behavior: contain; }
    .meter-preview-inner { height: 118px; background: #9d9d9b; display: grid; place-items: center; color: #080b16; font-size: 17px; position: relative; touch-action: none; user-select: none; -webkit-user-select: none; -webkit-touch-callout: none; }
    .meter-preview-inner * { user-select: none; -webkit-user-select: none; -webkit-touch-callout: none; }
    .meter-preview-inner.params { background: #202435; color: #f8fafc; }
    .meter-param-grid { width: 100%; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); text-align: center; gap: 8px; font-size: 18px; line-height: 1.75; }
    .meter-param-title { font-size: 18px; margin-bottom: 12px; }
    .color-compare-float { position: fixed; z-index: 130; width: 126px; height: 74px; border-radius: 8px; display: grid; place-items: center; color: #050505; font-size: 16px; font-weight: 700; box-shadow: 0 10px 28px rgba(0,0,0,.34); border: 2px solid rgba(255,255,255,.86); pointer-events: none; transform: translate(-50%, -50%); }
    .meter-dots { display: flex; justify-content: center; gap: 10px; padding: 8px 0 0; }
    .meter-dot { width: 10px; min-width: 10px; height: 10px; min-height: 10px; padding: 0; border-radius: 999px; background: #8b8f98; cursor: pointer; }
    .meter-dot.active { background: #59d1e9; }
    .meter-row-actions { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; align-items: center; margin: 10px 12px; }
    .meter-pill { min-height: 32px; border-radius: 999px; background: #3a3f58; color: #fff; display: inline-flex; align-items: center; justify-content: center; padding: 0 12px; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .meter-cyan-btn { min-height: 34px; border-radius: 999px; background: linear-gradient(90deg, #5fc7ea, #65ded7); color: #fff; font-size: 15px; font-weight: 500; }
    .meter-connect-btn { position: fixed; right: 18px; bottom: calc(86px + env(safe-area-inset-bottom)); z-index: 80; width: 56px; height: 56px; min-height: 56px; padding: 0; border-radius: 999px; background: linear-gradient(135deg, #5fc7ea, #65ded7); color: #fff; font-size: 12px; line-height: 1.1; display: grid; place-items: center; box-shadow: 0 8px 20px rgba(0,0,0,.22); }
    .color-bottom-nav { position: fixed; left: 0; right: 0; bottom: 0; z-index: 75; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); padding: 8px 8px calc(10px + env(safe-area-inset-bottom)); background: #080b16; }
    .color-nav-btn { min-height: 58px; padding: 4px 0; border-radius: 0; background: transparent; color: #fff; display: grid; place-items: center; gap: 2px; font-size: 14px; font-weight: 400; }
    .color-nav-btn.active { color: #35d7e8; }
    .color-nav-icon { font-size: 25px; line-height: 1; }
    .color-nav-icon.svg-icon { width: 28px; height: 28px; display: block; }
    .color-nav-icon.svg-icon svg { width: 100%; height: 100%; display: block; stroke: currentColor; }
    .library-hero { display: grid; grid-template-columns: minmax(0, 1fr) 120px; gap: 10px; align-items: center; margin: -18px -12px 16px; padding: 24px 14px 18px; background: #202435; }
    .library-hero-text { font-size: 16px; line-height: 1.5; color: #f8fafc; }
    .library-device { display: flex; align-items: center; justify-content: flex-end; gap: 10px; font-size: 17px; line-height: 1.25; }
    .library-device-img { width: 62px; height: 62px; border-radius: 999px; background: #fff; display: grid; place-items: center; font-size: 28px; }
    .color-screen-back { position: fixed; right: 14px; bottom: calc(84px + env(safe-area-inset-bottom)); z-index: 82; width: 52px; height: 52px; min-height: 52px; padding: 0; border-radius: 14px; background: linear-gradient(135deg, #5fc7ea, #65ded7); color: #fff; font-size: 12px; line-height: 1.05; display: grid; place-items: center; box-shadow: 0 8px 20px rgba(0,0,0,.22); }
    .color-screen-back::before { content: "↩"; display: block; font-size: 22px; line-height: 1; }
    .color-screen-back span { display: none; }
    .color-screen-add { position: fixed; right: 14px; bottom: calc(144px + env(safe-area-inset-bottom)); z-index: 82; width: 52px; height: 52px; min-height: 52px; padding: 0; border-radius: 14px; background: linear-gradient(135deg, #5fc7ea, #65ded7); color: #fff; font-size: 28px; line-height: 1; display: grid; place-items: center; box-shadow: 0 8px 20px rgba(0,0,0,.22); }
    .library-search { display: grid; grid-template-columns: 42px minmax(0, 1fr) 54px; align-items: center; min-height: 42px; border: 1px solid #8c91a1; border-radius: 999px; margin-bottom: 14px; color: #fff; }
    .library-search input { min-height: 38px; border: 0; background: transparent; color: #fff; padding: 0; }
    .library-search button { min-height: 38px; padding: 0; border-radius: 999px; background: transparent; color: #fff; }
    .library-section-bar { display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 54px; padding: 0 12px; border-radius: 5px; background: #202435; color: #5fd9e9; }
    .library-section-title { color: #fff; font-size: 18px; }
    .library-list { display: grid; gap: 8px; max-height: calc(100vh - 250px); overflow: auto; padding: 0 12px 168px; }
    .library-swipe { position: relative; overflow: hidden; border-radius: 5px; touch-action: pan-y; }
    .library-row-actions { position: absolute; right: 0; top: 0; bottom: 0; width: 152px; display: grid; grid-template-columns: 1fr 1fr; z-index: 0; }
    .library-add-card, .library-delete-card { border-radius: 0; color: #fff; font-size: 16px; }
    .library-add-card { background: #1987d7; }
    .library-delete-card { border-radius: 0 5px 5px 0; background: #dc2626; }
    .library-item { min-height: 62px; border-radius: 5px; background: #373b52; color: #fff; display: grid; grid-template-columns: minmax(0, 1fr) 34px; align-items: center; gap: 8px; padding: 8px 12px; }
    .library-item.active { background: linear-gradient(90deg, #5dc8e8, #64ded7); }
    .library-swipe .library-item { position: relative; z-index: 1; width: 100%; transition: transform .18s ease; }
    .library-swipe.open .library-item { transform: translateX(-152px); }
    .library-item-title { font-size: 17px; line-height: 1.25; word-break: break-word; font-weight: 600; }
    .library-check { width: 24px; height: 24px; border-radius: 999px; border: 2px solid #fff; display: grid; place-items: center; font-weight: 900; }
    .library-confirm { position: fixed; left: 12px; right: 12px; bottom: calc(18px + env(safe-area-inset-bottom)); min-height: 48px; border-radius: 999px; background: linear-gradient(90deg, #5fc7ea, #65ded7); color: #fff; font-size: 17px; }
    .device-steps { display: grid; grid-template-columns: 96px minmax(0,1fr); gap: 10px; align-items: center; margin: -18px -12px 0; padding: 22px 18px; background: #202435; font-size: 17px; line-height: 1.6; }
    .device-mode-box { margin: 14px 0 14px; display: grid; gap: 8px; }
    .device-mode-title { color: #fff; font-size: 16px; }
    .device-mode-tabs { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .device-mode-btn { min-height: 42px; border-radius: 999px; border: 1px solid #5fd9e9; background: transparent; color: #5fd9e9; font-size: 15px; }
    .device-mode-btn.active { border-color: transparent; background: linear-gradient(90deg, #5fc7ea, #65ded7); color: #fff; }
    .device-mode-note { color: #8b91a3; font-size: 13px; line-height: 1.35; }
    .device-list-title { margin: 18px 0 12px; font-size: 19px; color: #fff; }
    #colorScreenDevice { padding-bottom: calc(88px + env(safe-area-inset-bottom)); }
    .device-list { display: grid; gap: 12px; }
    .device-row { min-height: 72px; border-radius: 6px; background: #202435; color: #fff; display: grid; grid-template-columns: 28px minmax(0, 1fr) 44px 44px; align-items: center; gap: 10px; padding: 10px 12px; font-size: 16px; }
    .device-row.connected { box-shadow: inset 0 0 0 1px rgba(95,217,233,.75); }
    .device-row-main { min-width: 0; display: grid; gap: 4px; }
    .device-row-name { display: block; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-size: 18px; line-height: 1.15; }
    .device-row-status { margin-top: 3px; color: #8b91a3; font-size: 12px; line-height: 1.2; }
    .device-row.connected .device-row-status { color: #5fd9e9; }
    .device-icon-btn { width: 42px; height: 42px; min-height: 42px; padding: 0; border-radius: 999px; display: grid; place-items: center; line-height: 1; }
    .device-icon-btn svg { width: 22px; height: 22px; display: block; stroke: currentColor; }
    .device-connect-row-btn { border: 1px solid #5fd9e9; background: transparent; color: #5fd9e9; }
    .device-connect-row-btn:disabled { opacity: .38; border-color: #6b7280; color: #8b91a3; background: transparent; }
    .device-disconnect-row-btn { border: 1px solid #ff8fa3; background: transparent; color: #ff8fa3; }
    .device-disconnect-row-btn:disabled { opacity: .38; border-color: #6b7280; color: #8b91a3; }
    .device-empty { color: #8b91a3; font-size: 15px; padding: 16px 2px; }
    .radar-wrap { display: none; }
    .radar { position: relative; width: 118px; height: 118px; border-radius: 999px; background: #37d3df; display: grid; place-items: center; color: #fff; font-size: 42px; box-shadow: 0 0 0 12px rgba(55,211,223,.12), 0 0 0 24px rgba(55,211,223,.07); }
    .radar::before { content: ""; position: absolute; inset: -16px; border-radius: 999px; border: 2px dotted rgba(95,217,233,.45); animation: radarSpin 3.2s linear infinite; }
    .radar::after { content: ""; position: absolute; inset: -27px; border-radius: 999px; border-top: 5px solid rgba(95,217,233,.65); border-right: 5px solid transparent; border-bottom: 5px solid transparent; border-left: 5px solid transparent; animation: radarSpin 1.8s linear infinite; }
    @keyframes radarSpin { to { transform: rotate(360deg); } }
    .image-pick-tip { margin: 0 -12px; padding: 0 14px 18px; color: #f8fafc; font-size: 18px; display: flex; justify-content: space-between; align-items: center; }
    .image-menu-btn { min-width: 42px; min-height: 34px; padding: 0; border-radius: 8px; background: transparent; color: #fff; font-size: 26px; }
    .image-pick-card { margin: 0 -12px; min-height: 360px; display: grid; place-items: center; color: #b8bfce; background: #0f1422; overflow: hidden; }
    .image-pick-card.empty { border: 1px dashed #30364b; border-radius: 8px; margin: 20px 0; }
    .image-pick-stage { position: relative; width: 100%; min-height: 360px; display: grid; place-items: center; background: #111827; touch-action: manipulation; }
    .image-pick-stage img { width: 100%; max-height: 58vh; object-fit: cover; display: block; }
    .color-pick-cursor { position: absolute; width: 34px; height: 34px; border: 2px solid #fff; border-radius: 999px; transform: translate(-50%, -50%); box-shadow: 0 0 0 1px #0f172a; pointer-events: none; }
    .color-pick-cursor::before, .color-pick-cursor::after { content: ""; position: absolute; background: #fff; box-shadow: 0 0 0 1px #0f172a; }
    .color-pick-cursor::before { left: 50%; top: -7px; width: 2px; height: 46px; transform: translateX(-50%); }
    .color-pick-cursor::after { top: 50%; left: -7px; width: 46px; height: 2px; transform: translateY(-50%); }
    .color-pick-result { position: relative; margin: 0 -12px; min-height: 176px; padding: 34px 18px 30px; text-align: center; color: #090b12; background: #202435; display: grid; place-items: center; font-size: 20px; line-height: 1.45; }
    .color-pick-result.hidden { display: none; }
    .color-pick-result.has-color { color: #050505; }
    .color-pick-actions { position: absolute; right: 18px; top: 18px; display: flex; gap: 18px; font-size: 28px; color: currentColor; }
    .color-pick-action-btn { min-width: 34px; min-height: 34px; padding: 0; border-radius: 8px; background: transparent; color: currentColor; font-size: 28px; }
    .my-color-list { display: grid; gap: 10px; padding-bottom: 80px; }
    .my-color-filter { width: 100%; min-height: 54px; border-radius: 5px; background: #202435; display: grid; grid-template-columns: minmax(0,1fr) 48px; align-items: center; gap: 10px; padding: 0 14px; margin: 0 0 12px; color: #fff; font-size: 17px; text-align: left; }
    .my-color-filter-icon { text-align: right; font-size: 24px; line-height: 1; }
    .my-color-swipe { position: relative; overflow: hidden; border-radius: 5px; touch-action: pan-y; }
    .my-color-delete { position: absolute; right: 0; top: 0; bottom: 0; width: 72px; border-radius: 0 5px 5px 0; background: #ef4444; color: #fff; font-size: 16px; z-index: 0; }
    .my-color-card { position: relative; z-index: 1; min-height: 98px; border-radius: 5px; padding: 14px 12px 14px 18px; display: grid; grid-template-columns: minmax(0,1fr) 40px; align-items: start; gap: 8px; color: #fff; transition: transform .18s ease; }
    .my-color-swipe.open .my-color-card { transform: translateX(-72px); }
    .my-color-swatch { display: none; }
    .my-color-info { min-width: 0; }
    .my-color-name { font-size: 18px; line-height: 1.25; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 700; }
    .my-color-meta { font-size: 14px; line-height: 1.45; color: rgba(255,255,255,.92); font-weight: 500; }
    .my-color-menu-btn { width: 38px; height: 34px; min-height: 34px; padding: 0; border-radius: 8px; background: rgba(0,0,0,.12); color: currentColor; font-size: 24px; line-height: 1; }
    .my-color-list .color-match-item, .my-color-list .card { background: #202435; color: #fff; border-color: #30364b; }
    .selected-library-card { min-height: 62px; border-radius: 5px; background: #202435; color: #fff; display: grid; grid-template-columns: minmax(0,1fr) 32px; align-items: center; gap: 10px; padding: 10px 14px; font-size: 17px; }
    .selected-library-meta { color: #aab3c5; font-size: 13px; margin-top: 3px; }
    .color-action-sheet { position: fixed; inset: 0; z-index: 120; background: rgba(0,0,0,.45); display: grid; align-items: end; }
    .color-action-sheet.hidden { display: none; }
    .color-action-panel { background: #fff; color: #111827; border-radius: 16px 16px 0 0; overflow: hidden; text-align: center; font-size: 18px; }
    .color-action-panel button { min-height: 58px; width: 100%; border-radius: 0; background: #fff; color: #111827; border-bottom: 1px solid #edf0f4; font-size: 18px; font-weight: 500; }
    .color-action-panel button:last-child { border-bottom: 0; margin-top: 8px; }
    .color-form-panel { padding: 18px 16px calc(18px + env(safe-area-inset-bottom)); text-align: left; display: grid; gap: 10px; }
    .color-form-panel .panel-title { text-align: center; font-size: 20px; font-weight: 700; color: #111827; }
    .color-form-panel input { width: 100%; min-height: 44px; border: 1px solid #d8dde6; border-radius: 8px; padding: 0 12px; font-size: 16px; box-sizing: border-box; }
    .color-form-panel .row3 { display: grid; grid-template-columns: 1fr 92px 1fr; gap: 8px; }
    .color-form-panel .lab-status { min-height: 24px; color: #64748b; font-size: 14px; line-height: 1.4; }
    .color-form-panel button { border-radius: 8px; min-height: 46px; font-size: 16px; border: 0; }
    .color-form-panel .primary { background: #1987d7; color: #fff; }
    .color-form-panel .secondary { background: #eef2f7; color: #0f172a; }
    .empty { padding: 28px 0; color: #64748b; text-align: center; }
    .image-search-card { position: sticky; top: 0; z-index: 15; display: grid; gap: 10px; margin: -12px -14px 8px; padding: 0 14px 9px; color: #64748b; background: rgba(245,246,248,.98); border-bottom: 1px solid #e5e7eb; backdrop-filter: blur(10px); }
    .image-search-hint { display: none; }
    .image-upload-box { min-height: 260px; position: relative; display: grid; place-items: center; text-align: center; border: 1px dashed #cfd5df; border-radius: 6px; background: #fafafa; color: #777; padding: 18px; overflow: hidden; }
    .image-upload-box.has-image { min-height: 260px; border-style: solid; background: #f5f7fa; padding: 8px; }
    .image-upload-icon { width: 52px; height: 52px; margin: 0 auto 12px; border: 2px solid #777; border-radius: 12px; display: grid; place-items: center; font-size: 26px; color: #777; }
    .image-preview-stage { position: relative; width: 100%; touch-action: none; }
    .image-preview { display: block; width: 100%; max-height: 320px; object-fit: contain; border-radius: 6px; background: #eef2f7; user-select: none; -webkit-user-drag: none; }
    .crop-rect { position: absolute; border: 2px solid #1677d2; background: rgba(22,119,210,.16); box-shadow: 0 0 0 999px rgba(0,0,0,.18); pointer-events: none; }
    .image-search-run { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .image-search-run button { min-height: 38px; border-radius: 6px; font-size: 14px; }
    .choice-modal { position: fixed; inset: 0; z-index: 130; display: none; align-items: flex-end; background: rgba(0,0,0,.42); }
    .choice-modal.open { display: flex; }
    .choice-sheet { width: 100%; background: #fff; border-radius: 18px 18px 0 0; padding: 12px 16px calc(16px + env(safe-area-inset-bottom)); display: grid; gap: 10px; }
    .choice-sheet button { min-height: 52px; border-radius: 6px; font-size: 17px; }
    .app-confirm-modal { position: fixed; inset: 0; z-index: 180; display: none; align-items: center; justify-content: center; padding: 24px; background: rgba(0,0,0,.52); }
    .app-confirm-modal.open { display: flex; }
    .app-confirm-box { width: min(320px, 92vw); border-radius: 14px; background: #fff; color: #111827; overflow: hidden; box-shadow: 0 18px 40px rgba(0,0,0,.28); }
    .app-confirm-message { padding: 28px 22px; text-align: center; font-size: 17px; line-height: 1.5; color: #475569; white-space: pre-line; }
    .app-confirm-actions { display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid #e5e7eb; }
    .app-confirm-actions button { min-height: 52px; border-radius: 0; background: #fff; color: #0f172a; font-size: 17px; border-right: 1px solid #e5e7eb; }
    .app-confirm-actions button:last-child { border-right: 0; color: #64748b; }
    .image-result-list { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; margin-top: 8px; }
    .image-result-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }
    .image-result-card img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; background: #e5e7eb; }
    .image-result-body { padding: 5px 6px 6px; }
    .image-result-body .title { font-size: 12px; line-height: 1.2; margin-bottom: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .image-result-body .muted { font-size: 11px; }
    .hidden { display: none !important; }
  </style>
</head>
<body>
  <div class="app">
    <div class="top">
      <div class="head">
        <h1 id="libraryTitle">产品库</h1>
        <div class="status" id="status"></div>
      </div>
      <div class="app-tabs" id="appTabs">
        <button class="app-tab active" id="categoryTab" type="button">产品分类</button>
        <button class="app-tab" id="imageSearchTab" type="button">图搜</button>
        <button class="app-tab" id="mineTab" type="button">个人产品</button>
      </div>
      <div class="tabs library-tabs">
        <button class="tab" id="productTab" type="button">产品库</button>
        <button class="tab" id="colorTab" type="button">色卡库</button>
      </div>
      <div class="product-mode-tabs" id="productModeTabs">
        <button class="tab active" id="productQueryTab" type="button">查询</button>
        <button class="tab" id="productManageTab" type="button">管理</button>
      </div>
      <div class="color-mode-tabs hidden" id="colorModeTabs">
        <button class="tab active" id="colorQueryTab" type="button">查询</button>
        <button class="tab" id="colorManageTab" type="button">管理</button>
      </div>
      <div class="search">
        <input id="keyword" placeholder="输入款号、色号或名称" />
        <button id="filterBtn" class="secondary" type="button">筛选</button>
        <button id="searchBtn" type="button">搜索</button>
      </div>
      <div class="filter-summary" id="filterSummary"></div>
      <div class="personal-folder-bar hidden" id="personalFolderBar"></div>
    </div>
    <div class="body">
      <section class="panel" id="productPanel">
        <datalist id="yearOptions"></datalist>
        <datalist id="categoryOptions"></datalist>
        <datalist id="subcategoryOptions"></datalist>
        <div class="form hidden" id="productCreateBox">
          <div class="form-title">产品图片录入</div>
          <input id="productFiles" type="file" accept="image/*" multiple />
          <div class="muted">上传后先识别预览，确认年份、类别、细类后再入库；预览里修改款号时，导入文件名会自动跟着改。</div>
          <button id="uploadProductsBtn" type="button">上传识别</button>
        </div>
        <div class="review hidden" id="importReviewBox">
          <div class="review-head">
            <div>
              <div class="review-title">确认导入信息</div>
              <div class="muted" id="importReviewMeta"></div>
            </div>
            <button class="secondary" id="cancelImportBtn" type="button">取消</button>
          </div>
          <div class="bulk-note">批量标签：类别和细类会统一加到本次勾选导入的图片所属款号，可选择已有标签，也可直接输入新增标签。修改单张图片的款号时，导入文件名会自动同步修改。</div>
          <div class="bulk-fields">
            <select id="bulkImportCategorySelect"></select>
            <select id="bulkImportSubcategorySelect"></select>
            <input id="bulkImportCategory" placeholder="新增类别" />
            <input id="bulkImportSubcategory" placeholder="新增细类" />
          </div>
          <div class="review-list" id="importReviewList"></div>
          <div class="review-actions">
            <button class="secondary" id="selectAllImportBtn" type="button">全选/全不选</button>
            <button id="commitImportBtn" type="button">确认入库</button>
          </div>
        </div>
        <div class="list" id="productList"></div>
        <div class="load-more" id="productLoadMore"></div>
      </section>
      <section class="panel" id="imageSearchPanel">
        <div class="image-search-card">
          <div class="image-upload-box" id="imageUploadBox">
            <div id="imageUploadEmpty">
              <div class="image-upload-icon">□</div>
              <div>点击上传图片</div>
            </div>
            <div class="image-preview-stage hidden" id="imagePreviewStage">
              <img class="image-preview" id="imageSearchPreview" alt="查询图片" />
              <div class="crop-rect hidden" id="imageCropRect"></div>
            </div>
          </div>
          <input class="hidden" id="imageSearchFile" type="file" accept="image/*" />
          <input class="hidden" id="imageSearchCamera" type="file" accept="image/*" capture="environment" />
          <div class="image-search-run">
            <button class="secondary" id="clearImageSearchBtn" type="button">取消</button>
            <button id="runImageSearchBtn" type="button">确定</button>
          </div>
        </div>
        <div class="image-result-list" id="imageSearchResults"></div>
      </section>
      <section class="panel" id="colorPanel">
        <div class="color-home">
          <section class="color-screen active" id="colorScreenInstrument">
            <div class="color-screen-title">仪器对色</div>
            <div class="meter-preview-card">
              <div class="meter-preview-inner" id="meterPreviewInner"></div>
              <div class="meter-dots">
                <button class="meter-dot active" id="meterSwatchDot" type="button" aria-label="颜色"></button>
                <button class="meter-dot" id="meterParamsDot" type="button" aria-label="参数"></button>
              </div>
            </div>
            <div class="meter-row-actions">
              <div class="meter-pill" id="selectedColorLibraryName">请选择色彩库</div>
              <button class="meter-cyan-btn" id="chooseColorLibraryBtn" type="button">选择色彩库</button>
            </div>
            <div class="color-status hidden" id="colorMeterStatus"></div>
            <div class="grid3 hidden">
              <input id="colorL" type="number" step="0.01" placeholder="L" />
              <input id="colorA" type="number" step="0.01" placeholder="a" />
              <input id="colorB" type="number" step="0.01" placeholder="b" />
            </div>
            <div class="swatch hidden" id="colorSwatch" style="width:100%;height:64px;background:#f1f5f9;"></div>
            <div id="colorMatchBox" class="hidden">
              <button id="matchColorBtn" class="secondary hidden" type="button">匹配近似色号</button>
              <div class="muted" id="colorMatchStatus">测量/拖动比对或图片取色后可匹配近似色。</div>
              <div class="color-match-list" id="colorMatchList"></div>
            </div>
            <button id="colorMeterConnectBtn" class="meter-connect-btn" type="button">◎<br>测量</button>
            <button id="colorMeterMeasureBtn" class="hidden" type="button" disabled>测量</button>
          </section>
          <section class="color-screen" id="colorScreenLibrary">
            <div class="color-screen-title">选择色彩库</div>
            <div class="library-search"><span style="text-align:center;font-size:26px;">⌕</span><input id="colorLibrarySearch" placeholder="" /><button id="colorLibrarySearchBtn" type="button">搜索</button></div>
            <select id="colorLibrarySelect" class="hidden"></select>
            <div class="library-list" id="colorLibraryList"></div>
            <button class="library-confirm" id="confirmColorLibraryBtn" type="button">确认选择</button>
            <button class="color-screen-add" id="colorLibraryAddBtn" type="button" aria-label="新增色彩库">+</button>
            <button class="color-screen-back" id="colorLibraryBackBtn" type="button" aria-label="返回"><span>返回</span></button>
          </section>
          <section class="color-screen" id="colorScreenImage">
            <div class="color-screen-title">图片取色</div>
            <div class="image-pick-tip"><span>请点击需要取色的位置</span><button class="image-menu-btn" id="colorImageMenuBtn" type="button">•••</button></div>
            <input id="colorImageFile" class="hidden" type="file" accept="image/*" />
            <div class="image-pick-card" id="colorImageCard">
              <div class="image-pick-stage" id="colorImageStage">
                <img id="colorImagePreview" alt="" />
                <span class="color-pick-cursor hidden" id="colorPickCursor"></span>
              </div>
            </div>
            <div class="color-pick-result" id="colorPickResult">
              <div>上传图片后点击图片取色</div>
            </div>
          </section>
          <section class="color-screen" id="colorScreenMine">
            <div class="color-screen-title">我的色彩</div>
            <button class="my-color-filter" id="colorMineLibraryBtn" type="button"><span>当前色卡库：<span id="colorMineLibraryName">请选择色彩库</span></span><span class="my-color-filter-icon">⌕</span></button>
            <div class="my-color-list" id="colorList"></div>
          </section>
          <div class="color-action-sheet hidden" id="colorImageMenuSheet">
            <div class="color-action-panel">
              <button id="chooseColorImageBtn" type="button">选择图片</button>
              <button id="cancelColorImageMenuBtn" type="button">取消</button>
            </div>
          </div>
          <div class="color-action-sheet hidden" id="colorPickActionSheet">
            <div class="color-action-panel">
              <button id="savePickedColorBtn" type="button">保存到我的色彩</button>
              <button id="findPickedColorBtn" type="button">找接近色</button>
              <button id="cancelPickedColorActionBtn" type="button">取消</button>
            </div>
          </div>
          <div class="color-action-sheet hidden" id="colorCompareConfirmSheet">
            <div class="color-action-panel">
              <button id="confirmCompareFavoriteBtn" type="button">保存到我的色彩</button>
              <button id="cancelCompareFavoriteBtn" type="button">取消</button>
            </div>
          </div>
          <div class="color-action-sheet hidden" id="colorLibraryAddSheet">
            <div class="color-action-panel color-form-panel">
              <div class="panel-title">新增色彩库</div>
              <input id="newColorLibraryName" placeholder="输入色彩库名称" />
              <button class="primary" id="confirmAddColorLibraryBtn" type="button">新增</button>
              <button class="secondary" id="cancelAddColorLibraryBtn" type="button">取消</button>
            </div>
          </div>
          <div class="color-action-sheet hidden" id="colorCardAddSheet">
            <div class="color-action-panel color-form-panel">
              <div class="panel-title" id="colorCardAddTitle">新增色卡</div>
              <div class="row3">
                <input id="newColorCardPrefix" placeholder="固定前缀" />
                <input id="newColorCardNumber" inputmode="numeric" placeholder="编号" />
                <input id="newColorCardSuffix" placeholder="输入色名" />
              </div>
              <input id="newColorCardName" placeholder="色号名称" />
              <button class="secondary" id="measureNewColorCardBtn" type="button">测量 Lab</button>
              <div class="lab-status" id="newColorCardLabStatus">请先测量 Lab</div>
              <button class="primary" id="confirmAddColorCardBtn" type="button">保存色卡，继续下一个</button>
              <button class="secondary" id="cancelAddColorCardBtn" type="button">取消</button>
            </div>
          </div>
          <section class="color-screen" id="colorScreenDevice">
            <div class="color-screen-title">连接设备</div>
            <div class="device-steps"><div>连接步骤:</div><div>1.打开手机蓝牙，确保微信有蓝牙和定位权限<br>2.按一下测色仪顶部按钮唤醒设备<br>3.在搜索到的仪器列表中点击连接按钮</div></div>
            <div class="device-mode-box">
              <div class="device-mode-title">测量方式</div>
              <div class="device-mode-tabs">
                <button class="device-mode-btn" id="colorMeterSingleModeBtn" type="button" data-meter-mode="single">单次测量</button>
                <button class="device-mode-btn" id="colorMeterAverageModeBtn" type="button" data-meter-mode="average">平均测量</button>
              </div>
              <div class="device-mode-note" id="colorMeterModeNote"></div>
            </div>
            <div class="device-list-title">连接过的设备</div>
            <div class="device-list" id="colorDeviceList"></div>
            <div class="radar-wrap"><div class="radar">◎</div></div>
          </section>
          <div class="form hidden" id="colorCreateBox">
            <input id="colorLibrary" placeholder="新色卡库名称，可选" />
            <div class="color-name-builder">
              <input id="colorNamePrefix" placeholder="前缀，如彩龙" />
              <input id="colorNameNumber" inputmode="numeric" placeholder="编号" />
              <input id="colorNameSuffix" placeholder="色名，如浅灰" />
            </div>
            <input id="colorName" placeholder="色号名称，如 彩龙3351浅灰" />
            <button id="saveColorBtn" type="button">保存色卡</button>
          </div>
          <div class="color-compare-float hidden" id="colorCompareFloat"></div>
          <nav class="color-bottom-nav">
            <button class="color-nav-btn active" id="colorNavInstrument" type="button" data-color-view="instrument"><span class="color-nav-icon svg-icon"><svg viewBox="0 0 24 24" fill="none" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="8"></circle><path d="M12 12l3.2-4.2"></path><path d="M8 18.2h8"></path><path d="M7.2 12h2"></path><path d="M14.8 12h2"></path><path d="M12 5.2v2"></path></svg></span><span>仪器对色</span></button>
            <button class="color-nav-btn" id="colorNavImage" type="button" data-color-view="image"><span class="color-nav-icon svg-icon"><svg viewBox="0 0 24 24" fill="none" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3.5" y="5" width="17" height="14" rx="2.2"></rect><circle cx="8.2" cy="9.2" r="1.7"></circle><path d="M5.5 17l4.7-4.8 3.1 3.1 2.1-2.2 3.1 3.9"></path></svg></span><span>图片取色</span></button>
            <button class="color-nav-btn" id="colorNavMine" type="button" data-color-view="mine"><span class="color-nav-icon svg-icon"><svg viewBox="0 0 24 24" fill="none" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 4a8 8 0 1 0 8 8h-8z"></path><path d="M12 4v8h8"></path><path d="M16.7 6.1A8 8 0 0 1 20 12"></path></svg></span><span>我的色彩</span></button>
            <button class="color-nav-btn" id="colorNavDevice" type="button" data-color-view="device"><span class="color-nav-icon svg-icon"><svg viewBox="0 0 24 24" fill="none" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="8" y="3.5" width="8" height="17" rx="3"></rect><path d="M10 7h4"></path><path d="M10 16h4"></path><circle cx="12" cy="11.5" r="1.7"></circle></svg></span><span>管理仪器</span></button>
          </nav>
        </div>
      </section>
    </div>
    <div class="filter-modal" id="filterModal">
      <div class="filter-sheet">
        <div class="filter-sheet-head">
          <div class="filter-sheet-title">产品筛选</div>
          <button class="icon-btn" id="closeFilterBtn" type="button">×</button>
        </div>
        <div id="filterSheetBody"></div>
        <div class="filter-actions-bar">
          <button class="reset" id="resetFilterBtn" type="button">重置</button>
          <button class="apply" id="applyFilterBtn" type="button">确认</button>
        </div>
      </div>
    </div>
    <div class="modal" id="galleryModal">
      <div class="modal-panel">
        <div class="modal-head">
          <div>
            <div class="title" id="galleryTitle"></div>
            <div class="muted" id="gallerySubTitle"></div>
          </div>
          <button class="personal-btn" id="addPersonalBtn" type="button">加入个人产品</button>
          <button class="modal-close-btn" id="closeGalleryBtn" type="button" aria-label="关闭" title="关闭">×</button>
        </div>
        <div class="gallery-grid" id="galleryGrid"></div>
      </div>
    </div>
    <div class="zoom-modal" id="zoomModal">
      <button class="zoom-close" id="zoomCloseBtn" type="button">×</button>
      <img class="zoom-img" id="zoomImage" alt="图片预览" />
      <a class="zoom-original" id="zoomOriginalLink" href="#" target="_blank" rel="noopener">查看原图</a>
    </div>
    <div class="filter-modal" id="personalProductModal">
      <div class="filter-sheet">
        <div class="filter-sheet-head">
          <div class="filter-sheet-title">加入个人产品</div>
          <button class="icon-btn" id="closePersonalProductBtn" type="button">×</button>
        </div>
        <div class="personal-sheet-row">
          <div class="personal-sheet-label">已选图片</div>
          <div class="muted" id="personalProductSelectedText">0 张</div>
        </div>
        <div class="personal-sheet-row">
          <div class="personal-sheet-label">选择已有子目录</div>
          <select id="personalFolderSelect"></select>
        </div>
        <div class="personal-sheet-row">
          <div class="personal-sheet-label">或新建子目录</div>
          <input id="personalFolderInput" placeholder="例如：客户A、客户B" />
        </div>
        <div class="filter-actions-bar">
          <button class="reset" id="cancelPersonalProductBtn" type="button">取消</button>
          <button class="apply" id="confirmPersonalProductBtn" type="button">确认加入</button>
        </div>
      </div>
    </div>
    <div class="float-actions hidden" id="productFloatActions">
      <button class="float-action" id="editProductsBtn" type="button" aria-label="修改标签">✎<span>修改</span></button>
      <button class="float-action" id="addProductsBtn" type="button" aria-label="录入产品">＋<span>录入</span></button>
    </div>
    <div class="choice-modal" id="imageChoiceModal">
      <div class="choice-sheet">
        <button id="choiceAlbumBtn" type="button">从相册上传</button>
        <button class="secondary" id="choiceCameraBtn" type="button">拍照上传</button>
        <button class="secondary" id="choiceCancelBtn" type="button">取消</button>
      </div>
    </div>
  </div>
  <script>
    const INITIAL_TYPE = "__INITIAL_TYPE__";
    const SERVER_PERMISSIONS = __SERVER_PERMISSIONS__;
    const SERVER_USER_ID = __SERVER_USER_ID__;
    const tokenKey = "openfire_catalog_token";
    const params = new URLSearchParams(location.search);
    const urlToken = params.get("token") || params.get("access_token") || "";
    if (urlToken) {
      localStorage.setItem(tokenKey, urlToken);
      params.delete("token");
      params.delete("access_token");
    }
    if (SERVER_USER_ID) {
      localStorage.setItem("openfire_catalog_user_id", SERVER_USER_ID);
    }
    if (urlToken) {
      const next = location.pathname + (params.toString() ? "?" + params.toString() : "");
      history.replaceState(null, "", next);
    }
    const token = localStorage.getItem(tokenKey) || "";
    let userId = localStorage.getItem("openfire_catalog_user_id") || "";
    if (!userId) {
      userId = "web_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
      localStorage.setItem("openfire_catalog_user_id", userId);
    }
    const state = {
      type: params.get("type") || INITIAL_TYPE,
      products: [],
      colors: [],
      importJob: null,
      tagGroups: { year: [], category: [], subcategory: [] },
      selectedTags: [],
      filterDraftTags: [],
      appMode: params.get("mode") === "mine" ? "mine" : (params.get("mode") === "image" ? "image" : "category"),
      productMode: "query",
      productTool: "view",
      productLimit: 9,
      productOffset: 0,
      productHasMore: true,
      productLoading: false,
      productLoadSeq: 0,
      colorMode: "query",
      colorView: "instrument",
      previousColorView: "instrument",
      colorLibraries: [],
      colorLibraryIds: [],
      colorPickPreviewUrl: "",
      pickedColorLab: null,
      pickedColorHex: "",
      pickedColorRgb: null,
      colorActionTarget: null,
      meterPreviewMode: "swatch",
      meterSuppressClick: false,
      colorCompareDragging: false,
      colorCompareStartX: null,
      colorCompareStartY: null,
      colorCompareTarget: null,
      colorComparePendingTarget: null,
      colorFavoriteSwipe: null,
      colorLibrarySwipe: null,
      colorCardAddTargetLibraryId: "",
      colorCardAddMeasuredLab: null,
      nativeMeterPollTimer: null,
      nativeMeterPollSeq: 0,
      currentGalleryProduct: null,
      selectedGalleryImages: [],
      personalFolders: [],
      selectedPersonalFolder: "",
      personalDeletePending: false,
      imageSearchFile: null,
      imageSearchPreviewUrl: "",
      imageSearchCrop: null,
      imageSearchDrag: null,
    };
    const permissions = SERVER_PERMISSIONS.length ? SERVER_PERMISSIONS : readPermissions(token);
    const canProductView = hasPerm("product:view");
    const canProductCreate = hasPerm("product:create");
    const canColorView = hasPerm("color:view");
    const canColorCreate = hasPerm("color:create");
    const COLOR_METER_DEVICES_KEY = "openfire_color_meter_devices";
    const COLOR_METER_MODE_KEY = "openfire_color_meter_measure_mode";
    const COLOR_METER_CONNECTED_KEY = "openfire_color_meter_connected_device";
    const $ = (id) => document.getElementById(id);
    const COLOR_SERVICE_UUID = 0xFFE0;
    const COLOR_CHARACTERISTIC_UUID = 0xFFE1;
    let colorDevice = null;
    let colorCharacteristic = null;
    let colorPending = null;
    let colorResponseBytes = [];
    let colorMeasureId = 1;

    function appConfirm(message) {
      return new Promise((resolve) => {
        let modal = $("appConfirmModal");
        if (!modal) {
          modal = document.createElement("div");
          modal.id = "appConfirmModal";
          modal.className = "app-confirm-modal";
          modal.innerHTML = `
            <div class="app-confirm-box">
              <div class="app-confirm-message" id="appConfirmMessage"></div>
              <div class="app-confirm-actions">
                <button id="appConfirmOk" type="button">确定</button>
                <button id="appConfirmCancel" type="button">取消</button>
              </div>
            </div>`;
          document.body.appendChild(modal);
        }
        $("appConfirmMessage").textContent = String(message || "");
        const cleanup = (value) => {
          modal.classList.remove("open");
          $("appConfirmOk").onclick = null;
          $("appConfirmCancel").onclick = null;
          resolve(value);
        };
        $("appConfirmOk").onclick = () => cleanup(true);
        $("appConfirmCancel").onclick = () => cleanup(false);
        modal.classList.add("open");
      });
    }

    function readPermissions(raw) {
      const defaults = ["product:view", "product:create", "color:view", "color:create"];
      const parts = String(raw || "").split(".");
      if (parts.length < 2) return defaults;
      try {
        const payload = JSON.parse(atob(parts[1].replace(/-/g, "+").replace(/_/g, "/")));
        const value = payload.permissions || payload.perms || payload.scope || defaults;
        return Array.isArray(value) ? value : String(value).split(/[\\s,]+/);
      } catch (_) {
        return defaults;
      }
    }
    function hasPerm(name) {
      return permissions.includes("*") || permissions.includes(name);
    }
    function setStatus(text, isError) {
      $("status").textContent = text || "";
      $("status").style.color = isError ? "#b91c1c" : "#64748b";
    }
    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }
    async function api(path, options = {}) {
      const headers = Object.assign({}, options.headers || {});
      if (token) headers["X-Catalog-Token"] = token;
      const resp = await fetch(path, Object.assign({}, options, { headers }));
      if (resp.status === 401) throw new Error("登录已失效，请从小程序入口重新打开");
      if (resp.status === 403) throw new Error("当前用户没有此操作权限");
      if (!resp.ok) {
        const text = await resp.text();
        let message = text || "请求失败";
        try {
          const data = JSON.parse(text);
          message = data.detail || data.msg || data.message || message;
        } catch (_) {}
        throw new Error(message);
      }
      return resp.json();
    }
    function splitTag(raw) {
      const text = String(raw || "").trim();
      const index = text.indexOf(":");
      return index > 0 ? { type: text.slice(0, index), name: text.slice(index + 1) } : { type: "", name: text };
    }
    function tagLabel(type) {
      return { year: "年份", category: "类别", subcategory: "细类" }[type] || "标签";
    }
    async function loadTags() {
      if (!canProductView) return;
      const data = await api("/api/v1/catalog/tags");
      const groups = data.tag_groups || {};
      state.tagGroups = {
        year: groups.year || [],
        category: groups.category || ["单品", "罗纹", "毛织配件", "布匹"],
        subcategory: (groups.subcategory || []).filter((name) => String(name || "").trim() !== "暂无"),
      };
      $("yearOptions").innerHTML = state.tagGroups.year.map((x) => `<option value="${escapeHtml(x)}"></option>`).join("");
      $("categoryOptions").innerHTML = state.tagGroups.category.map((x) => `<option value="${escapeHtml(x)}"></option>`).join("");
      $("subcategoryOptions").innerHTML = state.tagGroups.subcategory.map((x) => `<option value="${escapeHtml(x)}"></option>`).join("");
      renderBulkImportSelects();
      renderProductFilters();
    }
    function renderBulkImportSelects() {
      if ($("bulkImportCategorySelect")) $("bulkImportCategorySelect").innerHTML = optionList("category", []);
      if ($("bulkImportSubcategorySelect")) $("bulkImportSubcategorySelect").innerHTML = optionList("subcategory", []);
    }
    function renderProductFilters() {
      renderProductFilterSummary();
      renderProductFilterSheet();
      renderPersonalFolders();
    }
    function selectedFilterNames() {
      return state.selectedTags.map((tag) => splitTag(tag).name).filter(Boolean);
    }
    function renderProductFilterSummary() {
      const box = $("filterSummary");
      if (!box) return;
      box.textContent = "";
    }
    function renderProductFilterSheet() {
      const box = $("filterSheetBody");
      if (!box) return;
      const draftTags = state.filterDraftTags || [];
      const rows = [
        ["年份", "year", state.tagGroups.year],
        ["类别", "category", state.tagGroups.category],
        ["细类", "subcategory", state.tagGroups.subcategory],
      ];
      box.innerHTML = rows.map(([label, type, list]) => `
        <div class="filter-group">
          <div class="filter-group-title">${label}</div>
          <div class="filter-options">
            <button class="filter-option ${!(list || []).some((name) => draftTags.includes(typedTag(type, name))) ? "active" : ""}" type="button" data-clear-type="${type}">全部</button>
          ${(list || []).map((name) => {
            const tag = typedTag(type, name);
            return `<button class="filter-option ${draftTags.includes(tag) ? "active" : ""}" type="button" data-filter-type="${type}" data-tag="${escapeHtml(tag)}">${escapeHtml(name)}</button>`;
          }).join("")}
          </div>
        </div>
      `).join("");
      box.querySelectorAll("[data-tag]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const tag = btn.dataset.tag || "";
          const type = btn.dataset.filterType || "";
          state.filterDraftTags = (state.filterDraftTags || []).filter((x) => splitTag(x).type !== type);
          if (!btn.classList.contains("active")) state.filterDraftTags = state.filterDraftTags.concat([tag]);
          renderProductFilterSheet();
        });
      });
      box.querySelectorAll("[data-clear-type]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const type = btn.dataset.clearType || "";
          state.filterDraftTags = (state.filterDraftTags || []).filter((x) => splitTag(x).type !== type);
          renderProductFilterSheet();
        });
      });
    }
    function openFilterModal() {
      state.filterDraftTags = state.selectedTags.slice();
      renderProductFilters();
      $("filterModal").classList.add("open");
    }
    function closeFilterModal() {
      $("filterModal").classList.remove("open");
    }
    function applyFilterModal() {
      state.selectedTags = (state.filterDraftTags || []).slice();
      closeFilterModal();
      renderProductFilters();
      loadProducts(true).catch((err) => setStatus(err.message || "加载失败", true));
    }
    function resetFilterModal() {
      state.filterDraftTags = [];
      renderProductFilterSheet();
    }
    function renderPersonalFolders() {
      const box = $("personalFolderBar");
      if (!box) return;
      if (state.appMode !== "mine") {
        box.classList.add("hidden");
        box.innerHTML = "";
        return;
      }
      const folders = state.personalFolders || [];
      box.classList.remove("hidden");
      box.innerHTML = [
        `<button class="personal-folder-chip ${!state.selectedPersonalFolder ? "active" : ""}" type="button" data-folder="">全部</button>`,
        ...folders.map((name) => `<button class="personal-folder-chip ${state.selectedPersonalFolder === name ? "active" : ""}" type="button" data-folder="${escapeHtml(name)}">${escapeHtml(name)}</button>`),
      ].join("");
      box.querySelectorAll("[data-folder]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.selectedPersonalFolder = btn.dataset.folder || "";
          renderPersonalFolders();
          loadProducts(true).catch((err) => setStatus(err.message || "加载失败", true));
        });
      });
    }
    function switchAppMode(mode) {
      state.appMode = mode === "mine" ? "mine" : (mode === "image" ? "image" : "category");
      state.type = "product";
      state.productMode = "query";
      state.productTool = "view";
      if (state.appMode !== "mine") state.selectedPersonalFolder = "";
      if (state.appMode !== "category") resetImportState(true);
      state.productOffset = 0;
      state.productHasMore = true;
      switchType("product");
    }
    function setProductTool(tool) {
      state.appMode = "category";
      state.type = "product";
      state.productTool = tool === "import" || tool === "edit" ? tool : "view";
      state.productMode = state.productTool === "view" ? "query" : "manage";
      state.productOffset = 0;
      state.productHasMore = true;
      if (state.productTool !== "import") resetImportState(true);
      switchType("product");
      if (state.productTool === "import") window.scrollTo({ top: 0, behavior: "smooth" });
    }
    function switchType(type) {
      state.type = type === "color" ? "color" : "product";
      if (state.type !== "product") {
        state.productTool = "view";
        state.productMode = "query";
      }
      $("productTab").classList.toggle("active", state.type === "product");
      $("colorTab").classList.toggle("active", state.type === "color");
      $("categoryTab").classList.toggle("active", state.appMode === "category");
      $("imageSearchTab").classList.toggle("active", state.appMode === "image");
      $("mineTab").classList.toggle("active", state.appMode === "mine");
      $("productPanel").classList.toggle("active", state.type === "product" && state.appMode !== "image");
      $("imageSearchPanel").classList.toggle("active", state.type === "product" && state.appMode === "image");
      $("colorPanel").classList.toggle("active", state.type === "color");
      $("libraryTitle").textContent = state.type === "color" ? "色卡库" : "产品库";
      $("productModeTabs").classList.toggle("hidden", state.type !== "product");
      $("colorModeTabs").classList.toggle("hidden", state.type !== "color");
      $("productQueryTab").classList.toggle("active", state.productMode === "query");
      $("productManageTab").classList.toggle("active", state.productMode === "manage");
      $("colorQueryTab").classList.toggle("active", state.colorMode === "query");
      $("colorManageTab").classList.toggle("active", state.colorMode === "manage");
      const isProductImport = canProductCreate && state.type === "product" && state.appMode === "category" && state.productTool === "import";
      const isProductEdit = state.type === "product" && state.appMode === "category" && state.productTool === "edit";
      $("productCreateBox").classList.toggle("hidden", !isProductImport);
      $("productList").classList.toggle("hidden", isProductImport);
      $("productLoadMore").classList.toggle("hidden", isProductImport);
      $("productFloatActions").classList.toggle("hidden", !(canProductCreate && state.type === "product" && state.appMode === "category"));
      $("addProductsBtn").classList.toggle("active", state.productTool === "import");
      $("addProductsBtn").classList.toggle("return", state.productTool === "import");
      $("addProductsBtn").innerHTML = state.productTool === "import" ? "↩<span>返回</span>" : "＋<span>录入</span>";
      $("editProductsBtn").classList.toggle("active", state.productTool === "edit");
      $("editProductsBtn").classList.toggle("return", state.productTool === "edit");
      $("editProductsBtn").innerHTML = state.productTool === "edit" ? "↩<span>返回</span>" : "✎<span>修改</span>";
      $("colorCreateBox").classList.toggle("hidden", !(canColorCreate && state.type === "color" && state.colorMode === "manage"));
      if ($("colorList")) $("colorList").classList.toggle("hidden", !(state.type === "color" && state.colorView === "mine"));
      if ($("colorMatchBox")) $("colorMatchBox").classList.toggle("hidden", !(state.type === "color" && state.colorView === "instrument"));
      document.body.classList.toggle("color-h5", state.type === "color");
      document.body.classList.toggle("product-image-mode", state.type === "product" && state.appMode === "image");
      if (state.type === "color") syncColorPageTitle();
      else if ($("libraryTitle")) $("libraryTitle").textContent = "产品库";
      const isManage = isProductImport || isProductEdit || (state.type === "color" && state.colorMode === "manage");
      document.querySelector(".search").classList.toggle("hidden", isManage || state.appMode === "image" || state.appMode === "mine");
      $("filterBtn").classList.toggle("hidden", !(state.type === "product" && state.appMode === "category"));
      renderProductFilters();
      $("filterSummary").classList.toggle("hidden", state.appMode === "image");
      $("keyword").placeholder = state.type === "product" ? "输入款号" : "输入色号或名称";
      const nextParams = new URLSearchParams(location.search);
      nextParams.set("type", state.type);
      nextParams.set("mode", state.appMode);
      history.replaceState(null, "", location.pathname + "?" + nextParams.toString());
      if (state.appMode !== "image") loadCurrent();
    }
    function switchProductMode(mode) {
      setProductTool(mode === "manage" ? "edit" : "view");
    }
    function switchColorMode(mode) {
      state.colorMode = mode === "manage" ? "manage" : "query";
      switchType("color");
    }
    function colorViewTitle(view) {
      return {
        instrument: "仪器对色",
        image: "图片取色",
        mine: "我的色彩",
        device: "管理仪器",
        library: "选择色彩库",
      }[view] || "色卡库";
    }
    function syncColorPageTitle() {
      const title = "色卡库";
      if ($("libraryTitle")) $("libraryTitle").textContent = title;
      document.title = title;
      try {
        window.parent?.postMessage?.({ title, pageTitle: title }, "*");
        if (window.wx?.miniProgram?.postMessage) {
          window.wx.miniProgram.postMessage({ data: { title, pageTitle: title } });
        }
      } catch (_) {}
    }
    function requestNativeColorMeter(action, deviceId = "") {
      try {
        if (!window.wx?.miniProgram?.navigateTo) return false;
        const nativeAction = action === "colorMeterConnect" ? "connect" : (action === "colorMeterDisconnect" ? "disconnect" : "measure");
        const actionText = nativeAction === "measure" ? "请点击测量按钮并按设备按钮测量" : (nativeAction === "disconnect" ? "已请求小程序原生断开..." : "已请求小程序原生连接...");
        setColorStatus(actionText, false);
        const requestId = `${Date.now()}_${Math.random().toString(16).slice(2)}`;
        startNativeMeterPolling(requestId, action);
        const query = new URLSearchParams({
          action: nativeAction,
          api_base: location.origin,
          user_tag: ownerTag(),
          token,
          request_id: requestId,
          measure_mode: readColorMeterMode(),
        });
        if (deviceId) query.set("device_id", deviceId);
        window.wx.miniProgram.navigateTo({
          url: `/pages/catalog_meter_bridge/index?${query.toString()}`,
        });
        return true;
      } catch (_) {
        return false;
      }
    }
    function startNativeMeterPolling(requestId, action) {
      const tag = ownerTag();
      if (!tag) return;
      if (state.nativeMeterPollTimer) clearTimeout(state.nativeMeterPollTimer);
      const seq = ++state.nativeMeterPollSeq;
      let attempts = 0;
      const poll = async () => {
        if (seq !== state.nativeMeterPollSeq) return;
        attempts += 1;
        try {
          const query = new URLSearchParams({ user_tag: tag, request_id: requestId || "", consume: "true" });
          const data = await api("/api/v1/color-card/native-meter/reading?" + query.toString());
          const reading = data.reading;
            if (reading) {
              if (reading.device_id || reading.device_name) saveColorDevice({ deviceId: reading.device_id || "", name: reading.device_name || "" });
              if (reading.event === "connect") {
                setConnectedColorDevice({ deviceId: reading.device_id || "", name: reading.device_name || "" });
                setColorStatus("色差仪已连接", false);
                return;
              }
              if (reading.event === "disconnect") {
                clearConnectedColorDevice({ deviceId: reading.device_id || "", name: reading.device_name || "" });
                setColorStatus("色差仪已断开", false);
                setColorCardAddLabStatus("色差仪已断开");
                return;
            }
            if (reading.event === "measure_progress") {
              const got = Number(reading.progress_count || reading.progressCount || 0);
              const total = Number(reading.total_count || reading.totalCount || 0);
              const progressText = got && total ? `已获取 ${got}/${total} 次，请继续按设备按钮测量` : "已获取一次测量，请继续";
              setColorStatus(progressText, false);
              setColorCardAddLabStatus(progressText);
            } else {
              const sampleText = Number(reading.sample_count || reading.sampleCount) > 1 ? `已获取 ${Number(reading.sample_count || reading.sampleCount)} 次平均 Lab` : "已获取小程序原生测色 Lab";
              if (applyNativeMeterLab(reading, sampleText)) {
                if (state.colorCardAddTargetLibraryId) {
                  const lab = currentLab();
                  state.colorCardAddMeasuredLab = lab;
                  if (lab) {
                    setColorCardAddLabStatus(`已获取 Lab：L ${lab.L.toFixed(2)} / a ${lab.a.toFixed(2)} / b ${lab.b.toFixed(2)}`);
                    setColorCardMeasureButtonColor(lab);
                  }
                  return;
                }
                switchColorView("instrument");
                matchColorCards().catch((err) => setStatus(err.message || "匹配失败", true));
              }
              return;
            }
          }
        } catch (err) {
          if (attempts > 3) setColorStatus(err.message || "读取小程序测色结果失败", true);
        }
        if (attempts < 50 && seq === state.nativeMeterPollSeq) {
          state.nativeMeterPollTimer = setTimeout(poll, 500);
        } else if (action === "colorMeterMeasure") {
          setColorStatus("未收到小程序原生测色结果，请重试", true);
        }
      };
      state.nativeMeterPollTimer = setTimeout(poll, 500);
    }
    function switchColorView(view) {
      state.colorView = ["instrument", "image", "mine", "device", "library"].includes(view) ? view : "instrument";
      const screens = {
        instrument: "colorScreenInstrument",
        image: "colorScreenImage",
        mine: "colorScreenMine",
        device: "colorScreenDevice",
        library: "colorScreenLibrary",
      };
      Object.values(screens).forEach((id) => $(id)?.classList.toggle("active", id === screens[state.colorView]));
      syncColorPageTitle();
      document.querySelectorAll("[data-color-view]").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.colorView === state.colorView);
      });
      if ($("colorMatchBox")) $("colorMatchBox").classList.toggle("hidden", !(state.type === "color" && state.colorView === "instrument"));
      if ($("colorList")) $("colorList").classList.toggle("hidden", !(state.type === "color" && state.colorView === "mine"));
      if (state.colorView === "mine") {
        loadColors().catch((err) => setStatus(err.message || "加载失败", true));
      }
      if (state.colorView === "device") renderColorDevices();
    }
    function productTags(item) {
      const groups = item.tag_groups || {};
      return [].concat(groups.year || [], groups.category || [], (groups.subcategory || []).filter((name) => String(name || "").trim() !== "暂无"));
    }
    function isPersonalProduct(item) {
      return (item && item.style_code && String(item.style_code).startsWith("MY-")) || (item && (item.raw_tags || []).some((tag) => String(tag).startsWith("owner:")));
    }
    function personalFolderName(item) {
      const owner = ownerTag();
      const prefix = owner ? `${owner}:folder:` : ":folder:";
      const tag = (item && item.raw_tags || []).find((x) => String(x).startsWith(prefix));
      return tag ? String(tag).slice(prefix.length) : "";
    }
    function productDisplayTitle(item) {
      if (!isPersonalProduct(item)) return item && item.style_code || "";
      const folder = personalFolderName(item);
      return folder || "个人产品";
    }
    function imageDisplayName(name) {
      const text = String(name || "");
      if (!text.startsWith("MY-")) return text;
      const match = text.match(/_([0-9a-f]{10})_(\d{3})(\.[^.]+)$/i);
      return match ? `图片 ${Number(match[2]) + 1}${match[3]}` : text;
    }
    function productImageCount(item) {
      return Number(item && item.image_count) || (item && item.images || []).length || 0;
    }
    async function loadProductDetail(product) {
      if (!product || !product.style_code) return product;
      if ((product.images || []).length >= productImageCount(product) && product.images) return product;
      const detail = await api("/api/v1/catalog/products/" + encodeURIComponent(product.style_code));
      const next = detail && detail.style_code ? detail : product;
      state.products = state.products.map((item) => item.style_code === next.style_code ? next : item);
      return next;
    }
    function optionList(type, selectedValues) {
      const selected = new Set(selectedValues || []);
      const list = (state.tagGroups[type] || []).filter((name) => type !== "subcategory" || String(name || "").trim() !== "暂无");
      selected.forEach((name) => {
        if (name && (type !== "subcategory" || String(name).trim() !== "暂无") && !list.includes(name)) list.unshift(name);
      });
      return `<option value="">${type === "subcategory" ? "可留空（暂无）" : "请选择"}</option>` + list.map((name) => `<option value="${escapeHtml(name)}" ${selected.has(name) ? "selected" : ""}>${escapeHtml(name)}</option>`).join("");
    }
    function productOriginalImageUrl(imageName, fallbackUrl = "") {
      const name = String(imageName || "").trim();
      if (!name) return fallbackUrl || "";
      const query = new URLSearchParams();
      if (token) query.set("token", token);
      if (!token && fallbackUrl) {
        try {
          const sourceUrl = new URL(fallbackUrl, window.location.origin);
          const exp = sourceUrl.searchParams.get("exp") || "";
          const sig = sourceUrl.searchParams.get("sig") || "";
          if (exp && sig) {
            query.set("exp", exp);
            query.set("sig", sig);
          }
        } catch (_) {}
      }
      return "/images/" + encodeURIComponent(name) + (query.toString() ? "?" + query.toString() : "");
    }
    function renderProducts() {
      const box = $("productList");
      if (!canProductView) {
        box.innerHTML = '<div class="empty">当前用户没有产品库查询权限</div>';
        return;
      }
      if (state.appMode === "category" && state.productTool === "import") {
        box.className = "list hidden";
        $("productLoadMore").textContent = "";
        return;
      }
      if (state.productLoading && !state.products.length) {
        box.className = "product-grid";
        box.innerHTML = '<div class="empty">加载中...</div>';
        $("productLoadMore").textContent = "";
        return;
      }
      if (!state.products.length) {
        box.innerHTML = '<div class="empty">暂无产品数据</div>';
        $("productLoadMore").textContent = "";
        return;
      }
      const isEditingTags = state.appMode === "category" && state.productTool === "edit";
      if (!isEditingTags) {
        box.className = "product-grid";
        box.innerHTML = state.products.map((item) => `
          <div class="card product-tile" data-role="viewProductTile" data-code="${item.style_code || ""}">
            <img class="thumb" src="${thumbnailUrl(item.cover_image_url || "", 220)}" alt="${escapeHtml(productDisplayTitle(item))}" loading="lazy" />
            <div class="product-tile-body">
              <div class="title">${escapeHtml(productDisplayTitle(item))}</div>
              <div class="muted">${productImageCount(item)} 张</div>
              <div class="tags">${productTags(item).map((tag) => `<span class="tag">${tag}</span>`).join("")}</div>
            </div>
          </div>
        `).join("");
        box.querySelectorAll("[data-role=viewProductTile]").forEach((tile) => {
          tile.addEventListener("click", () => openGallery(state.products.find((row) => row.style_code === tile.dataset.code)).catch((err) => setStatus(err.message || "加载图片失败", true)));
        });
        $("productLoadMore").textContent = state.productLoading ? "加载中..." : (state.productHasMore ? "向下滑动加载更多" : "已加载全部");
        return;
      }
      box.className = "list";
      $("productLoadMore").textContent = "";
      box.innerHTML = state.products.map((item) => `
        <div class="card product">
          <img class="thumb" data-role="viewProduct" data-code="${item.style_code || ""}" src="${thumbnailUrl(item.cover_image_url || "", 220)}" alt="${escapeHtml(productDisplayTitle(item))}" loading="lazy" />
          <div>
            <div class="title">${escapeHtml(productDisplayTitle(item))}</div>
            <div class="muted">图片数：${productImageCount(item)}</div>
            <div class="tags">${productTags(item).map((tag) => `<span class="tag">${tag}</span>`).join("")}</div>
            <div class="tag-edit">
              <div class="tag-edit-row"><label>年份</label><input data-role="yearInput" value="${((item.tag_groups || {}).year || []).join("、")}" inputmode="numeric" placeholder="年份" /></div>
              <div class="tag-edit-row"><label>类别</label><select data-role="categorySelect">${optionList("category", ((item.tag_groups || {}).category || []))}</select></div>
              <div class="tag-edit-row"><label>细类</label><select data-role="subcategorySelect">${optionList("subcategory", ((item.tag_groups || {}).subcategory || []))}</select></div>
              <button type="button" data-role="saveProductTags" data-code="${item.style_code || ""}">保存标签</button>
            </div>
          </div>
        </div>
      `).join("");
      box.querySelectorAll("[data-role=saveProductTags]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const card = btn.closest(".card");
          const code = btn.dataset.code || "";
          const split = (role) => String((card.querySelector(`[data-role="${role}"]`) || {}).value || "")
            .split(/[、,，\\s]+/).map((x) => x.trim()).filter(Boolean);
          const category = String((card.querySelector('[data-role="categorySelect"]') || {}).value || "").trim();
          const subcategory = String((card.querySelector('[data-role="subcategorySelect"]') || {}).value || "").trim();
          const preservedTags = (state.products.find((row) => row.style_code === code)?.raw_tags || state.products.find((row) => row.style_code === code)?.tags || [])
            .filter((tag) => {
              const parsed = splitTag(tag);
              return !["year", "category", "subcategory"].includes(parsed.type);
            });
          const tags = normalizeTags([
            ...preservedTags,
            ...split("yearInput").map((x) => typedTag("year", x)),
            typedTag("category", category),
            typedTag("subcategory", subcategory),
          ]);
          try {
            await api("/api/v1/catalog/products/" + encodeURIComponent(code) + "/tags", {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ tags }),
            });
            await loadTags();
            await loadProducts(true);
            setStatus("标签已保存", false);
          } catch (err) {
            setStatus(err.message || "保存标签失败", true);
          }
        });
      });
    }
    async function openGallery(product) {
      if (!product) return;
      state.currentGalleryProduct = product;
      $("galleryTitle").textContent = productDisplayTitle(product);
      $("gallerySubTitle").textContent = "图片加载中...";
      $("galleryGrid").innerHTML = '<div class="empty">图片加载中...</div>';
      $("galleryModal").classList.add("open");
      product = await loadProductDetail(product);
      state.currentGalleryProduct = product;
      state.selectedGalleryImages = [];
      const isPersonal = isPersonalProduct(product);
      $("galleryTitle").textContent = productDisplayTitle(product);
      $("gallerySubTitle").textContent = isPersonal ? `共 ${productImageCount(product)} 张图片` : `共 ${productImageCount(product)} 张图片，点击图片多选`;
      $("addPersonalBtn").textContent = isPersonal ? "取消个人产品" : "加入个人产品";
      $("addPersonalBtn").classList.toggle("remove", isPersonal);
      $("addPersonalBtn").classList.toggle("added", false);
      $("addPersonalBtn").classList.remove("hidden");
      $("galleryGrid").innerHTML = (product.images || []).map((img) => isPersonal ? `
        <div class="gallery-item selectable" data-image-name="${escapeHtml(img.image_name || "")}">
          <img data-role="zoomGalleryImage" src="${img.image_url || ""}" alt="${escapeHtml(imageDisplayName(img.image_name || ""))}" />
          <button class="gallery-delete" data-role="deletePersonalImage" data-image-name="${escapeHtml(img.image_name || "")}" type="button">×</button>
          <div class="gallery-caption">${escapeHtml(imageDisplayName(img.image_name || ""))}</div>
        </div>
      ` : `
        <div class="gallery-item selectable" data-image-name="${escapeHtml(img.image_name || "")}">
          <img data-role="zoomGalleryImage" src="${img.image_url || ""}" alt="${escapeHtml(img.image_name || "")}" />
          <span class="gallery-check" data-role="toggleGalleryImage">＋</span>
          <div class="gallery-caption">${escapeHtml(imageDisplayName(img.image_name || ""))}</div>
        </div>
      `).join("");
      $("galleryGrid").querySelectorAll('[data-role="zoomGalleryImage"]').forEach((img) => {
        img.addEventListener("click", (event) => {
          event.stopPropagation();
          const item = img.closest(".gallery-item");
          const imageName = (item && item.dataset.imageName) || "";
          openZoomImage(img.getAttribute("src") || "", img.getAttribute("alt") || "", productOriginalImageUrl(imageName, img.getAttribute("src") || ""));
        });
      });
      if (!isPersonal) {
        $("galleryGrid").querySelectorAll('[data-role="toggleGalleryImage"]').forEach((btn) => {
          btn.addEventListener("click", (event) => {
            event.stopPropagation();
            const item = btn.closest(".gallery-item");
            toggleGalleryImageSelection((item && item.dataset.imageName) || "");
          });
        });
        updateGallerySelectionUi();
      } else {
        $("galleryGrid").querySelectorAll('[data-role="deletePersonalImage"]').forEach((btn) => {
          btn.addEventListener("click", (event) => {
            event.stopPropagation();
            deleteCurrentPersonalProductImage(btn.dataset.imageName || "").catch((err) => setStatus(err.message || "删除失败", true));
          });
        });
      }
    }
    function openZoomImage(imageUrl, title, originalUrl = "") {
      if (!imageUrl) return;
      $("zoomImage").src = imageUrl;
      $("zoomImage").alt = title || "图片预览";
      const link = $("zoomOriginalLink");
      if (link) link.href = originalUrl || imageUrl;
      $("zoomModal").classList.add("open");
    }
    function closeZoomImage() {
      $("zoomModal").classList.remove("open");
      $("zoomImage").removeAttribute("src");
      const link = $("zoomOriginalLink");
      if (link) link.href = "#";
    }
    function openImagePreview(title, imageUrl) {
      state.currentGalleryProduct = null;
      $("galleryTitle").textContent = title || "图片预览";
      $("gallerySubTitle").textContent = "";
      $("addPersonalBtn").classList.add("hidden");
      $("galleryGrid").innerHTML = `
        <div class="gallery-item preview-full">
          <img src="${imageUrl || ""}" alt="${escapeHtml(title || "")}" />
          <div class="gallery-caption">${escapeHtml(title || "")}</div>
        </div>
      `;
      $("galleryModal").classList.add("open");
    }
    function closeGallery() {
      $("galleryModal").classList.remove("open");
    }
    function updateGallerySelectionUi() {
      const product = state.currentGalleryProduct;
      if (isPersonalProduct(product)) {
        $("gallerySubTitle").textContent = `共 ${productImageCount(product)} 张图片`;
        $("addPersonalBtn").textContent = "取消个人产品";
        $("addPersonalBtn").classList.add("remove");
        return;
      }
      const selected = new Set(state.selectedGalleryImages || []);
      $("galleryGrid").querySelectorAll(".gallery-item.selectable").forEach((item) => {
        const active = selected.has(item.dataset.imageName || "");
        item.classList.toggle("selected", active);
        const mark = item.querySelector(".gallery-check");
        if (mark) mark.textContent = active ? "✓" : "＋";
      });
      const total = productImageCount(product);
      const count = selected.size;
      $("gallerySubTitle").textContent = `共 ${total} 张图片，已选 ${count} 张`;
      $("addPersonalBtn").textContent = count ? `加入个人产品 (${count})` : "加入个人产品";
    }
    function toggleGalleryImageSelection(imageName) {
      const name = String(imageName || "").trim();
      if (!name) return;
      const selected = new Set(state.selectedGalleryImages || []);
      if (selected.has(name)) selected.delete(name);
      else selected.add(name);
      state.selectedGalleryImages = Array.from(selected);
      updateGallerySelectionUi();
    }
    async function loadPersonalFolders() {
      const tag = ownerTag();
      if (!tag) return [];
      const data = await api("/api/v1/catalog/personal-folders?" + new URLSearchParams({ user_tag: tag }).toString());
      state.personalFolders = data.folders || [];
      if (state.selectedPersonalFolder && !state.personalFolders.includes(state.selectedPersonalFolder)) {
        state.selectedPersonalFolder = "";
      }
      return state.personalFolders;
    }
    function renderPersonalFolderOptions() {
      const select = $("personalFolderSelect");
      const folders = state.personalFolders || [];
      select.innerHTML = '<option value="">请选择已有子目录</option>' + folders.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
    }
    async function openPersonalProductModal() {
      const count = (state.selectedGalleryImages || []).length;
      if (!count) {
        setStatus("请先点击选择要加入的图片", true);
        return;
      }
      $("personalProductSelectedText").textContent = `${count} 张`;
      $("personalFolderInput").value = "";
      await loadPersonalFolders().catch(() => []);
      renderPersonalFolderOptions();
      $("personalProductModal").classList.add("open");
    }
    function closePersonalProductModal() {
      $("personalProductModal").classList.remove("open");
    }
    async function confirmAddPersonalProduct() {
      const product = state.currentGalleryProduct;
      if (!product || !product.style_code) return;
      if (isPersonalProduct(product)) return;
      const selected = state.selectedGalleryImages || [];
      if (!selected.length) return setStatus("请先选择图片", true);
      const folder = ($("personalFolderInput").value.trim() || $("personalFolderSelect").value.trim());
      if (!folder) return setStatus("请选择或新建子目录", true);
      const button = $("confirmPersonalProductBtn");
      setButtonLoading(button, true);
      try {
        await api("/api/v1/catalog/personal-products", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            source_style_code: product.style_code,
            image_names: selected,
            folder_name: folder,
            user_tag: ownerTag(),
          }),
        });
        closePersonalProductModal();
        closeGallery();
        state.selectedPersonalFolder = folder;
        await loadPersonalFolders().catch(() => []);
        renderPersonalFolders();
        setStatus(`已加入个人产品：${folder}`, false);
        if (state.appMode === "mine") await loadProducts(true);
      } finally {
        setButtonLoading(button, false);
      }
    }
    async function deleteCurrentPersonalProduct() {
      const product = state.currentGalleryProduct;
      if (!product || !product.style_code || !isPersonalProduct(product) || state.personalDeletePending) return;
      const ok = await appConfirm("确定取消这个个人产品？");
      if (!ok) return;
      state.personalDeletePending = true;
      const button = $("addPersonalBtn");
      setButtonLoading(button, true);
      try {
        const query = new URLSearchParams({ user_tag: ownerTag() });
        await api("/api/v1/catalog/personal-products/" + encodeURIComponent(product.style_code) + "?" + query.toString(), { method: "DELETE" });
        closeGallery();
        await loadPersonalFolders().catch(() => []);
        renderPersonalFolders();
        if (state.appMode === "mine") await loadProducts(true);
        setStatus("已取消个人产品", false);
      } finally {
        state.personalDeletePending = false;
        setButtonLoading(button, false);
      }
    }
    async function deleteCurrentPersonalProductImage(imageName) {
      const product = state.currentGalleryProduct;
      const name = String(imageName || "").trim();
      if (!product || !product.style_code || !isPersonalProduct(product) || !name || state.personalDeletePending) return;
      const ok = await appConfirm("确定删除这张图片？");
      if (!ok) return;
      state.personalDeletePending = true;
      try {
        const query = new URLSearchParams({ user_tag: ownerTag() });
        const data = await api(
          "/api/v1/catalog/personal-products/" + encodeURIComponent(product.style_code) +
          "/images/" + encodeURIComponent(name) + "?" + query.toString(),
          { method: "DELETE" },
        );
        if (data.product) {
          state.currentGalleryProduct = data.product;
          state.products = state.products.map((item) => item.style_code === product.style_code ? data.product : item);
          await openGallery(data.product);
          setStatus("已删除图片", false);
        } else {
          closeGallery();
          await loadPersonalFolders().catch(() => []);
          renderPersonalFolders();
          if (state.appMode === "mine") await loadProducts(true);
          setStatus("已删除个人产品", false);
        }
      } finally {
        state.personalDeletePending = false;
      }
    }
    function renderColors() {
      const box = $("colorList");
      if (!canColorView) {
        box.innerHTML = '<div class="empty">当前用户没有色卡库查询权限</div>';
        return;
      }
      if (!state.colors.length) {
        box.innerHTML = '<div class="empty">暂无色卡数据</div>';
        return;
      }
      box.innerHTML = state.colors.map((item) => {
        const canDelete = item.is_favorite || isEditableColorLibraryId(item.library_id);
        const hex = normalizeHex(item.hex || labToHex({ L: Number(item.l) || 0, a: Number(item.a) || 0, b: Number(item.b) || 0 }));
        const l = Number(item.l) || 0;
        const a = Number(item.a) || 0;
        const b = Number(item.b) || 0;
        return `
        <div class="my-color-swipe" data-card-id="${escapeHtml(String(item.id || ""))}" data-card-name="${escapeHtml(item.name || "")}" data-card-hex="${hex}" data-card-l="${l}" data-card-a="${a}" data-card-b="${b}" data-favorite="${item.is_favorite ? "1" : "0"}" data-user-card="${(!item.is_favorite && isEditableColorLibraryId(item.library_id)) ? "1" : "0"}">
          ${canDelete ? '<button class="my-color-delete" type="button">删除</button>' : ""}
          <div class="my-color-card" style="background:#${hex};">
            <div class="my-color-swatch" style="background:#${hex};"></div>
            <div class="my-color-info">
              <div class="my-color-name">名称: ${escapeHtml(item.name || "")}</div>
              <div class="my-color-meta">色彩库: ${escapeHtml(item.library_name || item.library_id || "")}</div>
              <div class="my-color-meta">色值: L*: ${l.toFixed(2)} a*: ${a.toFixed(2)} b*: ${b.toFixed(2)}</div>
              ${item.created_at ? `<div class="my-color-meta">创建时间: ${escapeHtml(item.created_at)}</div>` : ""}
            </div>
            <button class="my-color-menu-btn" type="button" aria-label="更多操作">•••</button>
          </div>
        </div>
      `}).join("");
    }
    function renderSelectedColorLibraries() {
      const box = $("colorList");
      if (!box) return;
      const ids = activeColorLibraryIds();
      const selected = (state.colorLibraries || []).filter((lib) => ids.includes(lib.id));
      updateColorLibraryLabels();
      if (!selected.length) {
        box.innerHTML = '<div class="empty" style="color:#8b91a3;">暂无已选色彩库</div>';
        return;
      }
      box.innerHTML = selected.map((lib) => `
        <button class="selected-library-card" type="button" data-library-id="${escapeHtml(lib.id || "")}">
          <span>
            <strong>${escapeHtml(lib.name || lib.id || "")}</strong>
            <div class="selected-library-meta">${Number(lib.color_count || 0)} 个色号</div>
          </span>
          <span>›</span>
        </button>
      `).join("");
      box.querySelectorAll("[data-library-id]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.colorLibraryIds = [btn.dataset.libraryId].filter(Boolean);
          updateColorLibraryLabels();
          switchColorView("instrument");
        });
      });
    }
    function normalizeHex(raw) {
      const text = String(raw || "").replace("#", "").trim();
      return /^[0-9a-fA-F]{6}$/.test(text) ? text.toUpperCase() : "CCCCCC";
    }
    function activeColorLibraryIds() {
      if (state.colorLibraryIds.length) return state.colorLibraryIds.slice();
      return (state.colorLibraries || []).map((lib) => lib.id).filter(Boolean);
    }
    function isVirtualFavoriteLibraryId(id) {
      return String(id || "") === "__favorites__";
    }
    function userLibraryPrefix() {
      return "custom_";
    }
    function isEditableColorLibraryId(id) {
      return String(id || "").startsWith(userLibraryPrefix());
    }
    function normalizeColorLibraries(rawLibraries) {
      const seen = new Set();
      const publicLibraries = (rawLibraries || [])
        .filter((lib) => {
          const id = String(lib.id || "");
          const name = String(lib.name || "");
          if (!id) return false;
          if (seen.has(id)) return false;
          seen.add(id);
          if (/pantone|color-pt|潘通/i.test(`${id} ${name}`)) return false;
          if (id === "__favorites__" || /^my_color_/i.test(id) || /我的收藏|收藏夹/.test(name)) return false;
          if (/^user_/i.test(id)) return false;
          return true;
        });
      return [{ id: "__favorites__", name: "我的收藏", color_count: 0 }].concat(publicLibraries);
    }
    function colorLibraryLabel() {
      const ids = activeColorLibraryIds();
      const libraries = state.colorLibraries || [];
      if (!ids.length) return "请选择色彩库";
      if (ids.length === 1) return libraries.find((lib) => lib.id === ids[0])?.name || ids[0];
      return `已选 ${ids.length} 个色彩库`;
    }
    function updateColorLibraryLabels() {
      const label = colorLibraryLabel();
      if ($("selectedColorLibraryName")) $("selectedColorLibraryName").textContent = label;
      if ($("colorMineLibraryName")) $("colorMineLibraryName").textContent = label;
      if ($("colorLibrarySelect")) {
        const realId = activeColorLibraryIds().find((id) => !isVirtualFavoriteLibraryId(id)) || "";
        $("colorLibrarySelect").value = realId;
      }
    }
    function labToHex(lab) {
      let y = (lab.L + 16) / 116;
      let x = lab.a / 500 + y;
      let z = y - lab.b / 200;
      const pivot = (v) => v > 6 / 29 ? Math.pow(v, 3) : (v - 16 / 116) / 7.787;
      x = pivot(x) * 0.95047;
      y = pivot(y);
      z = pivot(z) * 1.08883;
      const gamma = (v) => Math.max(0, Math.min(255, Math.round((v > 0.0031308 ? 1.055 * Math.pow(v, 1 / 2.4) - 0.055 : 12.92 * v) * 255)));
      return [
        gamma(3.2406 * x - 1.5372 * y - 0.4986 * z),
        gamma(-0.9689 * x + 1.8758 * y + 0.0415 * z),
        gamma(0.0557 * x - 0.2040 * y + 1.0570 * z),
      ].map((n) => n.toString(16).padStart(2, "0")).join("").toUpperCase();
    }
    function rgbToLab(r, g, b) {
      const pivotRgb = (v) => {
        v = v / 255;
        return v > 0.04045 ? Math.pow((v + 0.055) / 1.055, 2.4) : v / 12.92;
      };
      let rr = pivotRgb(r);
      let gg = pivotRgb(g);
      let bb = pivotRgb(b);
      let x = (rr * 0.4124 + gg * 0.3576 + bb * 0.1805) / 0.95047;
      let y = (rr * 0.2126 + gg * 0.7152 + bb * 0.0722) / 1.00000;
      let z = (rr * 0.0193 + gg * 0.1192 + bb * 0.9505) / 1.08883;
      const pivotXyz = (v) => v > 0.008856 ? Math.pow(v, 1 / 3) : (7.787 * v) + (16 / 116);
      x = pivotXyz(x);
      y = pivotXyz(y);
      z = pivotXyz(z);
      return { L: (116 * y) - 16, a: 500 * (x - y), b: 200 * (y - z) };
    }
    function currentLab() {
      const L = Number($("colorL").value);
      const a = Number($("colorA").value);
      const b = Number($("colorB").value);
      if (!Number.isFinite(L) || !Number.isFinite(a) || !Number.isFinite(b)) return null;
      return { L, a, b };
    }
    function readColorDevices() {
      try {
        const rows = JSON.parse(localStorage.getItem(COLOR_METER_DEVICES_KEY) || "[]");
        return Array.isArray(rows) ? rows.filter((item) => item && (item.deviceId || item.name)) : [];
      } catch (_) {
        return [];
      }
    }
    function saveColorDevice(device) {
      const clean = {
        deviceId: String(device?.deviceId || "").trim(),
        name: String(device?.name || "").trim(),
        ts: Date.now(),
      };
      if (!clean.deviceId && !clean.name) return;
      const rows = readColorDevices().filter((item) => item.deviceId !== clean.deviceId && item.name !== clean.name);
      rows.unshift(clean);
      localStorage.setItem(COLOR_METER_DEVICES_KEY, JSON.stringify(rows.slice(0, 8)));
      renderColorDevices();
    }
    function readConnectedColorDevice() {
      try {
        const data = JSON.parse(localStorage.getItem(COLOR_METER_CONNECTED_KEY) || "{}");
        return data && typeof data === "object" ? data : {};
      } catch (_) {
        return {};
      }
    }
    function setConnectedColorDevice(device) {
      const clean = {
        deviceId: String(device?.deviceId || device?.device_id || "").trim(),
        name: String(device?.name || device?.deviceName || device?.device_name || "").trim(),
        ts: Date.now(),
      };
      if (!clean.deviceId && !clean.name) return;
      localStorage.setItem(COLOR_METER_CONNECTED_KEY, JSON.stringify(clean));
      saveColorDevice(clean);
      renderColorDevices();
    }
    function clearConnectedColorDevice(device) {
      const current = readConnectedColorDevice();
      const targetId = String(device?.deviceId || device?.device_id || "").trim();
      const targetName = String(device?.name || device?.deviceName || device?.device_name || "").trim();
      if (!targetId && !targetName) {
        localStorage.removeItem(COLOR_METER_CONNECTED_KEY);
      } else if (!current.deviceId || current.deviceId === targetId || current.name === targetName) {
        localStorage.removeItem(COLOR_METER_CONNECTED_KEY);
      }
      renderColorDevices();
    }
    function readColorMeterMode() {
      return localStorage.getItem(COLOR_METER_MODE_KEY) === "average" ? "average" : "single";
    }
    function setColorMeterMode(mode) {
      const next = mode === "average" ? "average" : "single";
      localStorage.setItem(COLOR_METER_MODE_KEY, next);
      renderColorMeterMode();
    }
    function renderColorMeterMode() {
      const mode = readColorMeterMode();
      document.querySelectorAll("[data-meter-mode]").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.meterMode === mode);
      });
      const note = $("colorMeterModeNote");
      if (note) {
        note.textContent = mode === "average"
          ? "平均测量：连续按 3 次设备顶部按钮，取平均 Lab 后匹配近似色。"
          : "单次测量：按 1 次设备顶部按钮后读取 Lab。";
      }
    }
    function renderColorDevices() {
      const box = $("colorDeviceList");
      if (!box) return;
      renderColorMeterMode();
      const rows = readColorDevices();
      const connected = readConnectedColorDevice();
      if (!rows.length) {
        box.innerHTML = '<div class="device-empty">暂无连接记录</div><button class="device-connect-row-btn" type="button" data-device-id="">搜索连接</button>';
      } else {
        box.innerHTML = rows.map((item, index) => {
          const isConnected = (connected.deviceId && item.deviceId && connected.deviceId === item.deviceId) || (!connected.deviceId && connected.name && connected.name === item.name);
          return `
          <div class="device-row ${isConnected ? "connected" : ""}">
            <span>${index + 1}</span>
            <span class="device-row-main">
              <span class="device-row-name">${escapeHtml(item.name || item.deviceId || "色差仪")}</span>
              <span class="device-row-status">${isConnected ? "● 已连接" : "○ 未连接"}</span>
            </span>
            <button class="device-icon-btn device-connect-row-btn" type="button" data-meter-action="connect" data-device-id="${escapeHtml(item.deviceId || "")}" aria-label="连接" title="连接" ${isConnected ? "disabled" : ""}>
              <svg viewBox="0 0 24 24" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 7l10 10-5 5V2l5 5L7 17"></path><path d="M5 12h14"></path></svg>
            </button>
            <button class="device-icon-btn device-disconnect-row-btn" type="button" data-meter-action="disconnect" data-device-id="${escapeHtml(item.deviceId || "")}" aria-label="断开" title="断开" ${isConnected ? "" : "disabled"}>
              <svg viewBox="0 0 24 24" fill="none" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v9"></path><path d="M6.4 6.8a8 8 0 1 0 11.2 0"></path></svg>
            </button>
          </div>
        `}).join("");
      }
      box.querySelectorAll("[data-meter-action]").forEach((btn) => {
        btn.addEventListener("click", () => {
          if (btn.disabled) return;
          const action = btn.dataset.meterAction === "disconnect" ? "colorMeterDisconnect" : "colorMeterConnect";
          requestNativeColorMeter(action, btn.dataset.deviceId || "");
        });
      });
      box.querySelectorAll(".device-connect-row-btn:not([data-meter-action])").forEach((btn) => {
        btn.addEventListener("click", () => requestNativeColorMeter("colorMeterConnect", btn.dataset.deviceId || ""));
      });
    }
    function applyNativeMeterDeviceFromUrl() {
      const params = new URLSearchParams(window.location.search || "");
      const deviceId = params.get("meter_device_id") || "";
      const name = params.get("meter_device_name") || "";
      if (!deviceId && !name) return false;
      saveColorDevice({ deviceId, name });
      return true;
    }
    function applyNativeMeterLab(raw, statusText = "已获取小程序原生测色 Lab") {
      const lab = {
        L: Number(raw?.L),
        a: Number(raw?.a),
        b: Number(raw?.b),
      };
      if (!Number.isFinite(lab.L) || !Number.isFinite(lab.a) || !Number.isFinite(lab.b)) return false;
      saveColorDevice({ deviceId: raw?.device_id || raw?.deviceId || "", name: raw?.device_name || raw?.deviceName || "" });
      setConnectedColorDevice({ deviceId: raw?.device_id || raw?.deviceId || "", name: raw?.device_name || raw?.deviceName || "" });
      state.pickedColorLab = null;
      state.pickedColorHex = "";
      state.pickedColorRgb = null;
      $("colorL").value = lab.L.toFixed(2);
      $("colorA").value = lab.a.toFixed(2);
      $("colorB").value = lab.b.toFixed(2);
      refreshColorSwatch();
      setMeterPreviewMode("swatch");
      setColorStatus(statusText, false);
      return true;
    }
    function applyNativeMeterLabFromUrl() {
      const params = new URLSearchParams(window.location.search || "");
      if (!params.has("meter_l") || !params.has("meter_a") || !params.has("meter_b")) return false;
      return applyNativeMeterLab({
        L: Number(params.get("meter_l")),
        a: Number(params.get("meter_a")),
        b: Number(params.get("meter_b")),
        device_id: params.get("meter_device_id") || "",
        device_name: params.get("meter_device_name") || "",
      });
    }
    function refreshColorSwatch() {
      const lab = currentLab();
      if (!lab) return;
      const hex = labToHex(lab);
      $("colorSwatch").style.background = "#" + hex;
      $("colorSwatch").textContent = "#" + hex;
      $("colorSwatch").style.color = lab.L < 55 ? "#fff" : "#0f172a";
    }
    const DEFAULT_COLOR_IMAGE = "/recolor-static/static/low_poly.png";
    function setColorImageFile(file) {
      if (state.colorPickPreviewUrl) URL.revokeObjectURL(state.colorPickPreviewUrl);
      state.colorPickPreviewUrl = file ? URL.createObjectURL(file) : "";
      $("colorPickCursor").classList.add("hidden");
      if (file) {
        $("colorImagePreview").src = state.colorPickPreviewUrl;
        $("colorPickResult").classList.remove("has-color");
        $("colorPickResult").classList.remove("hidden");
        $("colorPickResult").style.background = "#202435";
        $("colorPickResult").style.color = "#f8fafc";
        $("colorPickResult").innerHTML = "<div>点击图片中的位置取色</div>";
      } else {
        $("colorImagePreview").src = DEFAULT_COLOR_IMAGE;
        state.pickedColorLab = null;
        state.pickedColorHex = "";
        state.pickedColorRgb = null;
        $("colorL").value = "";
        $("colorA").value = "";
        $("colorB").value = "";
        $("colorPickResult").classList.remove("has-color");
        $("colorPickResult").classList.add("hidden");
        $("colorPickResult").style.background = "#202435";
        $("colorPickResult").style.color = "#f8fafc";
        $("colorPickResult").innerHTML = "";
        updateMeterPreview();
      }
    }
    function setPickedColor(r, g, b, point) {
      const hex = [r, g, b].map((n) => n.toString(16).padStart(2, "0")).join("").toUpperCase();
      const lab = rgbToLab(r, g, b);
      state.pickedColorLab = lab;
      state.pickedColorHex = hex;
      state.pickedColorRgb = { r, g, b };
      if (point) {
        const cursor = $("colorPickCursor");
        cursor.style.left = `${point.x}px`;
        cursor.style.top = `${point.y}px`;
        cursor.classList.remove("hidden");
      }
      const result = $("colorPickResult");
      result.classList.remove("hidden");
      result.classList.add("has-color");
      result.style.background = "#" + hex;
      result.style.color = lab.L < 55 ? "#fff" : "#050505";
      result.innerHTML = `<div>RGB [${r},${g},${b}]<br>Hex #${hex}<br>Lab [${lab.L.toFixed(2)},${lab.a.toFixed(2)},${lab.b.toFixed(2)}]</div><div class="color-pick-actions"><button class="color-pick-action-btn" id="colorPickMoreBtn" type="button">•••</button></div>`;
      $("colorPickMoreBtn")?.addEventListener("click", (event) => {
        event.stopPropagation();
        openColorPickActions();
      });
      $("colorL").value = lab.L.toFixed(2);
      $("colorA").value = lab.a.toFixed(2);
      $("colorB").value = lab.b.toFixed(2);
      refreshColorSwatch();
      updateMeterPreview();
    }
    function openColorImageMenu() {
      $("colorImageMenuSheet")?.classList.remove("hidden");
    }
    function closeColorImageMenu() {
      $("colorImageMenuSheet")?.classList.add("hidden");
    }
    function openColorPickActions(target = null) {
      state.colorActionTarget = target;
      $("colorPickActionSheet")?.classList.remove("hidden");
    }
    function closeColorPickActions() {
      state.colorActionTarget = null;
      $("colorPickActionSheet")?.classList.add("hidden");
    }
    async function saveLabColorToFavorites(name, lab, hex) {
      const tag = ownerTag();
      if (!tag) return setStatus("缺少用户 token，无法保存到我的色彩", true);
      await api("/api/v1/color-card/favorites/picked", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_tag: tag, name, hex, L: lab.L, a: lab.a, b: lab.b }),
      });
      if (state.colorView === "mine") await loadColors();
    }
    async function savePickedColorToFavorites() {
      const lab = state.pickedColorLab;
      const hex = state.pickedColorHex;
      if (!lab || !hex) return setStatus("请先取色", true);
      const name = `取色 #${hex}`;
      await saveLabColorToFavorites(name, lab, hex);
      setStatus("已保存到我的色彩", false);
    }
    async function addColorCardToFavorites(cardId) {
      const tag = ownerTag();
      if (!tag) throw new Error("缺少用户 token，无法保存到我的色彩");
      if (!cardId) throw new Error("色号数据不完整，无法保存");
      await api("/api/v1/color-card/favorites", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_tag: tag, card_id: Number(cardId) }),
      });
      if (state.colorView === "mine") await loadColors();
    }
    async function saveCurrentActionColorToFavorites() {
      const target = state.colorActionTarget;
      if (!target) return savePickedColorToFavorites();
      if (target.cardId && target.isExistingCard) {
        await addColorCardToFavorites(target.cardId);
      } else {
        await saveLabColorToFavorites(target.name || `色号 #${target.hex}`, target.lab, target.hex);
      }
      setStatus("已保存到我的色彩", false);
    }
    function applyActionTargetToMeter(target) {
      if (!target || !target.lab || !target.hex) return false;
      state.pickedColorLab = target.lab;
      state.pickedColorHex = target.hex;
      state.pickedColorRgb = null;
      $("colorL").value = target.lab.L.toFixed(2);
      $("colorA").value = target.lab.a.toFixed(2);
      $("colorB").value = target.lab.b.toFixed(2);
      refreshColorSwatch();
      updateMeterPreview();
      return true;
    }
    function currentColorActionTarget() {
      if (state.colorActionTarget) return state.colorActionTarget;
      const lab = state.pickedColorLab;
      const hex = state.pickedColorHex;
      return lab && hex ? { lab, hex, name: `取色 #${hex}`, isExistingCard: false } : null;
    }
    async function deleteColorFavorite(cardId) {
      const tag = ownerTag();
      if (!tag) return setStatus("缺少用户 token，无法删除收藏", true);
      if (!cardId) return;
      await api("/api/v1/color-card/favorites/" + encodeURIComponent(cardId) + "?" + new URLSearchParams({ user_tag: tag }).toString(), {
        method: "DELETE",
      });
      state.colors = state.colors.filter((item) => String(item.id || "") !== String(cardId));
      renderColors();
      setStatus("已删除收藏", false);
    }
    async function deleteUserColorCard(cardId) {
      const tag = ownerTag();
      if (!tag) return setStatus("缺少用户 token，无法删除色卡", true);
      if (!cardId) return;
      await api("/api/v1/color-card/cards/" + encodeURIComponent(cardId) + "?" + new URLSearchParams({ user_tag: tag }).toString(), {
        method: "DELETE",
      });
      state.colors = state.colors.filter((item) => String(item.id || "") !== String(cardId));
      renderColors();
      setStatus("已删除色卡", false);
    }
    function setMeterPreviewMode(mode) {
      state.meterPreviewMode = mode === "params" ? "params" : "swatch";
      updateMeterPreview();
    }
    function updateMeterPreview() {
      const lab = state.pickedColorLab || currentLab();
      const hex = state.pickedColorHex || (lab ? labToHex(lab) : "");
      const rgb = state.pickedColorRgb;
      const box = $("meterPreviewInner");
      if (!box) return;
      $("meterSwatchDot")?.classList.toggle("active", state.meterPreviewMode === "swatch");
      $("meterParamsDot")?.classList.toggle("active", state.meterPreviewMode === "params");
      if (!hex) {
        box.classList.toggle("params", state.meterPreviewMode === "params");
        box.style.background = state.meterPreviewMode === "params" ? "" : "#9d9d9b";
        box.style.color = "#f8fafc";
        box.innerHTML = state.meterPreviewMode === "params" ? `<div class="meter-param-grid">
          <div><div class="meter-param-title">Lab</div><div>L*: --</div><div>a*: --</div><div>b*: --</div></div>
          <div><div class="meter-param-title">RGB</div><div>R: --</div><div>G: --</div><div>B: --</div></div>
          <div><div class="meter-param-title">Hex</div><div>--</div></div>
        </div>` : "";
        return;
      }
      if (state.meterPreviewMode === "params") {
        box.classList.add("params");
        box.style.background = "";
        box.style.color = "";
        box.innerHTML = `<div class="meter-param-grid">
          <div><div class="meter-param-title">Lab</div><div>L*: ${lab ? lab.L.toFixed(2) : "--"}</div><div>a*: ${lab ? lab.a.toFixed(2) : "--"}</div><div>b*: ${lab ? lab.b.toFixed(2) : "--"}</div></div>
          <div><div class="meter-param-title">RGB</div><div>R: ${rgb ? rgb.r : "--"}</div><div>G: ${rgb ? rgb.g : "--"}</div><div>B: ${rgb ? rgb.b : "--"}</div></div>
          <div><div class="meter-param-title">Hex</div><div>#${escapeHtml(hex)}</div></div>
        </div>`;
        return;
      }
      box.classList.remove("params");
      box.style.background = "#" + hex;
      box.style.color = (lab && lab.L < 55) ? "#fff" : "#050505";
      box.innerHTML = `<span>#${escapeHtml(hex)}</span>`;
    }
    function currentMeterColor() {
      const lab = state.pickedColorLab || currentLab();
      const hex = state.pickedColorHex || (lab ? labToHex(lab) : "");
      if (!lab || !hex) return null;
      return { lab, hex: normalizeHex(hex) };
    }
    function clearCompareTarget() {
      if (state.colorCompareTarget) state.colorCompareTarget.classList.remove("compare-target");
      state.colorCompareTarget = null;
    }
    function setCompareFloatPosition(event) {
      const color = currentMeterColor();
      const floater = $("colorCompareFloat");
      if (!color || !floater) return;
      floater.style.left = `${event.clientX}px`;
      floater.style.top = `${event.clientY}px`;
      floater.style.background = "#" + color.hex;
      floater.style.color = color.lab.L < 55 ? "#fff" : "#050505";
      floater.textContent = "#" + color.hex;
      floater.classList.remove("hidden");
    }
    function updateCompareTarget(event) {
      clearCompareTarget();
      const stack = document.elementsFromPoint(event.clientX, event.clientY);
      const target = stack.find((node) => node?.classList?.contains("color-match-item"));
      if (!target) return null;
      target.classList.add("compare-target");
      state.colorCompareTarget = target;
      return target;
    }
    function startColorCompareDrag(event) {
      if (state.meterPreviewMode !== "swatch" || !currentMeterColor()) return;
      event.preventDefault();
      state.meterSuppressClick = false;
      state.colorCompareStartX = event.clientX;
      state.colorCompareStartY = event.clientY;
      state.colorCompareDragging = false;
    }
    function moveColorCompareDrag(event) {
      if (!Number.isFinite(state.colorCompareStartX) || !Number.isFinite(state.colorCompareStartY)) return;
      const dx = event.clientX - state.colorCompareStartX;
      const dy = event.clientY - state.colorCompareStartY;
      if (!state.colorCompareDragging && Math.hypot(dx, dy) < 18) return;
      event.preventDefault();
      if (!state.colorCompareDragging) {
        state.colorCompareDragging = true;
        state.meterSuppressClick = true;
      }
      setCompareFloatPosition(event);
      updateCompareTarget(event);
    }
    async function finishColorCompareDrag(event) {
      if (!state.colorCompareDragging) {
        state.colorCompareStartX = null;
        state.colorCompareStartY = null;
        return false;
      }
      event.preventDefault();
      setCompareFloatPosition(event);
      const target = updateCompareTarget(event);
      const color = currentMeterColor();
      $("colorCompareFloat")?.classList.add("hidden");
      state.colorCompareDragging = false;
      state.colorCompareStartX = null;
      state.colorCompareStartY = null;
      if (!target || !color) {
        clearCompareTarget();
        return true;
      }
      const targetColor = {
        name: target.dataset.matchName || target.querySelector(".title")?.textContent || "当前色号",
        hex: normalizeHex(target.dataset.matchHex || ""),
        lab: {
          L: Number(target.dataset.matchL),
          a: Number(target.dataset.matchA),
          b: Number(target.dataset.matchB),
        },
      };
      if (!targetColor.hex || !Number.isFinite(targetColor.lab.L) || !Number.isFinite(targetColor.lab.a) || !Number.isFinite(targetColor.lab.b)) {
        clearCompareTarget();
        setStatus("目标色号数据不完整，无法收藏", true);
        return true;
      }
      clearCompareTarget();
      state.colorComparePendingTarget = targetColor;
      openCompareConfirm();
      return true;
    }
    function cancelColorCompareDrag() {
      state.colorCompareDragging = false;
      state.colorCompareStartX = null;
      state.colorCompareStartY = null;
      $("colorCompareFloat")?.classList.add("hidden");
      clearCompareTarget();
    }
    function openCompareConfirm() {
      $("colorCompareConfirmSheet")?.classList.remove("hidden");
    }
    function closeCompareConfirm() {
      $("colorCompareConfirmSheet")?.classList.add("hidden");
    }
    async function confirmCompareFavorite() {
      const target = state.colorComparePendingTarget;
      closeCompareConfirm();
      state.colorComparePendingTarget = null;
      if (!target) return;
      await saveLabColorToFavorites(`对色 #${target.hex}`, target.lab, target.hex);
      setStatus("已加入我的色彩", false);
    }
    function showInstrumentPickedColor() {
      setMeterPreviewMode("swatch");
    }
    function pickColorFromImage(event) {
      const img = $("colorImagePreview");
      if (!img.src || !img.naturalWidth || !img.naturalHeight) return;
      const rect = img.getBoundingClientRect();
      const clientX = event.clientX ?? event.touches?.[0]?.clientX;
      const clientY = event.clientY ?? event.touches?.[0]?.clientY;
      if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return;
      if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) return;
      const x = Math.max(0, Math.min(img.naturalWidth - 1, Math.round((clientX - rect.left) / rect.width * img.naturalWidth)));
      const y = Math.max(0, Math.min(img.naturalHeight - 1, Math.round((clientY - rect.top) / rect.height * img.naturalHeight)));
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(img, 0, 0);
      const pixel = ctx.getImageData(x, y, 1, 1).data;
      const r = pixel[0], g = pixel[1], b = pixel[2];
      setPickedColor(r, g, b, {
        x: clientX - $("colorImageStage").getBoundingClientRect().left,
        y: clientY - $("colorImageStage").getBoundingClientRect().top,
      });
    }
    async function loadColorLibraries(selectedId = "") {
      const data = await api("/api/v1/color-card/libraries");
      const select = $("colorLibrarySelect");
      const libraries = normalizeColorLibraries(data.libraries || []);
      state.colorLibraries = libraries;
      if (!state.colorLibraryIds.length) {
        state.colorLibraryIds = selectedId ? [selectedId] : libraries.map((lib) => lib.id).filter(Boolean);
      }
      state.colorLibraryIds = state.colorLibraryIds.filter((id) => libraries.some((lib) => lib.id === id));
      select.innerHTML = libraries
        .filter((lib) => !isVirtualFavoriteLibraryId(lib.id))
        .map((lib) => `<option value="${escapeHtml(lib.id)}">${escapeHtml(lib.name)} (${lib.color_count || 0})</option>`)
        .join("");
      updateColorLibraryLabels();
      renderColorLibraryList(libraries, state.colorLibraryIds);
      maybeFillColorNamePrefix(select.options[select.selectedIndex]?.textContent || "");
    }
    function renderColorLibraryList(libraries, selectedIds = []) {
      const box = $("colorLibraryList");
      if (!box) return;
      const list = (libraries || []).slice().sort((left, right) => {
        const leftMine = /我的收藏/.test(`${left.id || ""} ${left.name || ""}`) ? 0 : 1;
        const rightMine = /我的收藏/.test(`${right.id || ""} ${right.name || ""}`) ? 0 : 1;
        return leftMine - rightMine;
      });
      const activeIds = new Set((selectedIds || []).filter(Boolean));
      if (!list.length) {
        box.innerHTML = '<div class="empty" style="color:#8b91a3;">暂无色彩库</div>';
        return;
      }
      box.innerHTML = list.map((lib) => {
        const itemHtml = `
        <button class="library-item ${activeIds.has(lib.id) ? "active" : ""}" type="button" data-library-id="${escapeHtml(lib.id || "")}">
          <span class="library-item-title">${escapeHtml(lib.name || lib.id || "")}${lib.year ? `<br>${escapeHtml(lib.year)}` : ""}</span>
          <span class="library-check">${activeIds.has(lib.id) ? "✓" : "+"}</span>
        </button>`;
        return isEditableColorLibraryId(lib.id)
          ? `<div class="library-swipe" data-library-id="${escapeHtml(lib.id || "")}"><div class="library-row-actions"><button class="library-add-card" type="button">录入</button><button class="library-delete-card" type="button">删除</button></div>${itemHtml}</div>`
          : itemHtml;
      }).join("");
      box.querySelectorAll("[data-library-id]").forEach((btn) => {
        btn.addEventListener("click", () => {
          if (btn.classList.contains("library-swipe")) return;
          const id = btn.dataset.libraryId || "";
          if (!id) return;
          if (state.colorLibraryIds.includes(id)) {
            state.colorLibraryIds = state.colorLibraryIds.filter((item) => item !== id);
          } else {
            state.colorLibraryIds = state.colorLibraryIds.concat(id);
          }
          updateColorLibraryLabels();
          renderColorLibraryList(list, state.colorLibraryIds);
        });
      });
      box.querySelectorAll(".library-add-card").forEach((btn) => {
        btn.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          const row = btn.closest(".library-swipe");
          row?.classList.remove("open");
          openColorCardAddSheet(row?.dataset.libraryId || "");
        });
      });
      box.querySelectorAll(".library-delete-card").forEach((btn) => {
        btn.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          const row = btn.closest(".library-swipe");
          row?.classList.remove("open");
          deleteColorLibrary(row?.dataset.libraryId || "").catch((err) => setStatus(err.message || "删除色彩库失败", true));
        });
      });
    }
    function setColorCardAddLabStatus(message) {
      const box = $("newColorCardLabStatus");
      if (!box || !state.colorCardAddTargetLibraryId) return;
      box.textContent = String(message || "");
    }
    function resetColorCardMeasureButton() {
      const btn = $("measureNewColorCardBtn");
      if (!btn) return;
      btn.style.background = "";
      btn.style.color = "";
    }
    function setColorCardMeasureButtonColor(lab) {
      const btn = $("measureNewColorCardBtn");
      if (!btn || !lab) return;
      const hex = labToHex(lab);
      btn.style.background = "#" + hex;
      btn.style.color = lab.L < 58 ? "#fff" : "#0f172a";
    }
    function colorLibraryById(id) {
      return (state.colorLibraries || []).find((lib) => String(lib.id || "") === String(id || "")) || null;
    }
    function colorCardFormKey(libraryId) {
      return "openfire_color_card_form_" + String(libraryId || "");
    }
    function readColorCardFormDefaults(libraryId) {
      try {
        const raw = JSON.parse(localStorage.getItem(colorCardFormKey(libraryId)) || "{}");
        return raw && typeof raw === "object" ? raw : {};
      } catch (_) {
        return {};
      }
    }
    function writeColorCardFormDefaults(libraryId, prefix, number) {
      localStorage.setItem(colorCardFormKey(libraryId), JSON.stringify({ prefix: String(prefix || ""), number: String(number || "") }));
    }
    function nextColorCardNumber(raw) {
      const text = String(raw || "").trim();
      if (!/^\\d+$/.test(text)) return "001";
      return String(Number(text) + 1).padStart(text.length, "0");
    }
    function updateNewColorCardName() {
      const name = `${$("newColorCardPrefix").value.trim()}${$("newColorCardNumber").value.trim()}${$("newColorCardSuffix").value.trim()}`.trim();
      $("newColorCardName").value = name;
      return name;
    }
    function openColorLibraryAddSheet() {
      $("newColorLibraryName").value = "";
      $("colorLibraryAddSheet")?.classList.remove("hidden");
      setTimeout(() => $("newColorLibraryName")?.focus(), 40);
    }
    function closeColorLibraryAddSheet() {
      $("colorLibraryAddSheet")?.classList.add("hidden");
    }
    async function addColorLibrary() {
      const name = $("newColorLibraryName").value.trim();
      if (!name) return setStatus("请输入色彩库名称", true);
      const id = userLibraryPrefix() + Date.now().toString(36);
      const data = await api("/api/v1/color-card/libraries", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, name }),
      });
      closeColorLibraryAddSheet();
      state.colorLibraryIds = Array.from(new Set(state.colorLibraryIds.concat([data.library?.id || id])));
      await loadColorLibraries(data.library?.id || id);
      setStatus("色彩库已新增", false);
    }
    async function deleteColorLibrary(libraryId) {
      const lib = colorLibraryById(libraryId);
      if (!lib || !isEditableColorLibraryId(libraryId)) return setStatus("只能删除自建色彩库", true);
      if (!(await appConfirm(`删除色彩库“${lib.name || libraryId}”及其中色卡？`))) return;
      await api(`/api/v1/color-card/libraries/${encodeURIComponent(libraryId)}`, { method: "DELETE" });
      state.colorLibraryIds = state.colorLibraryIds.filter((id) => id !== libraryId);
      await loadColorLibraries();
      if (state.colorView === "mine") await loadColors();
      setStatus("色彩库已删除", false);
    }
    function openColorCardAddSheet(libraryId) {
      const lib = colorLibraryById(libraryId);
      if (!lib || !isEditableColorLibraryId(lib.id)) {
        return setStatus("只能在自建色彩库中录入色卡", true);
      }
      state.colorCardAddTargetLibraryId = lib.id;
      state.colorCardAddMeasuredLab = null;
      $("colorCardAddTitle").textContent = `新增色卡：${lib.name || lib.id}`;
      const defaults = readColorCardFormDefaults(lib.id);
      $("newColorCardPrefix").value = String(defaults.prefix || inferColorNamePrefix(lib.name || "") || "");
      $("newColorCardNumber").value = nextColorCardNumber(defaults.number || "");
      $("newColorCardSuffix").value = "";
      updateNewColorCardName();
      resetColorCardMeasureButton();
      setColorCardAddLabStatus("请点击测量按钮并按设备按钮测量");
      $("colorCardAddSheet")?.classList.remove("hidden");
    }
    function closeColorCardAddSheet() {
      $("colorCardAddSheet")?.classList.add("hidden");
      state.colorCardAddTargetLibraryId = "";
      state.colorCardAddMeasuredLab = null;
    }
    async function saveNewColorCard() {
      const libraryId = state.colorCardAddTargetLibraryId;
      const lib = colorLibraryById(libraryId);
      const lab = state.colorCardAddMeasuredLab || currentLab();
      const name = $("newColorCardName").value.trim() || updateNewColorCardName();
      if (!lib || !isEditableColorLibraryId(libraryId)) return setStatus("请选择自建色彩库", true);
      if (!name) return setStatus("请填写色号名称", true);
      if (!lab) return setStatus("请先测量 Lab", true);
      await api("/api/v1/color-card/cards", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          library_id: libraryId,
          library_name: lib.name || libraryId,
          name,
          note: "",
          L: lab.L,
          a: lab.a,
          b: lab.b,
        }),
      });
      const savedNumber = $("newColorCardNumber").value.trim();
      writeColorCardFormDefaults(libraryId, $("newColorCardPrefix").value.trim(), savedNumber);
      $("newColorCardNumber").value = nextColorCardNumber(savedNumber);
      $("newColorCardSuffix").value = "";
      state.colorCardAddMeasuredLab = null;
      updateNewColorCardName();
      resetColorCardMeasureButton();
      setColorCardAddLabStatus("请点击测量按钮并按设备按钮测量");
      await loadColorLibraries(libraryId);
      if (state.colorView === "mine") await loadColors();
      setStatus("色卡已保存，可继续录入下一张", false);
    }
    function inferColorNamePrefix(raw) {
      const text = String(raw || "");
      if (text.includes("彩龙")) return "彩龙";
      if (text.includes("国彩")) return "国彩";
      if (text.includes("恩盛")) return "恩盛";
      return "";
    }
    function maybeFillColorNamePrefix(raw) {
      const input = $("colorNamePrefix");
      if (!input || input.value.trim()) return;
      const prefix = inferColorNamePrefix(raw);
      if (!prefix) return;
      input.value = prefix;
      refreshColorName();
    }
    function buildColorName() {
      return `${$("colorNamePrefix").value.trim()}${$("colorNameNumber").value.trim()}${$("colorNameSuffix").value.trim()}`.trim();
    }
    function refreshColorName() {
      const built = buildColorName();
      if (built) $("colorName").value = built;
    }
    function incrementColorNameNumber() {
      const raw = $("colorNameNumber").value.trim();
      if (!/^\\d+$/.test(raw)) return;
      $("colorNameNumber").value = String(Number(raw) + 1).padStart(raw.length, "0");
      $("colorNameSuffix").value = "";
      refreshColorName();
    }
    async function matchColorCards() {
      const lab = currentLab();
      if (!lab) return setStatus("请先测量或输入 Lab 数值", true);
      refreshColorSwatch();
      $("colorMatchStatus").textContent = "正在匹配近似色号...";
      const ids = activeColorLibraryIds().filter((id) => !isVirtualFavoriteLibraryId(id));
      const batches = await Promise.all((ids.length ? ids : [""]).map((libraryId) => api("/api/v1/color-card/match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ L: lab.L, a: lab.a, b: lab.b, library_id: libraryId, limit: 12 }),
      })));
      const matches = batches.flatMap((data) => data.matches || [])
        .sort((left, right) => Number(left.delta_e_00 || 999) - Number(right.delta_e_00 || 999))
        .slice(0, 12);
      $("colorMatchStatus").textContent = `找到 ${matches.length} 条近似色号`;
      $("colorMatchList").innerHTML = matches.map((item) => {
        const hex = normalizeHex(item.hex || "CCCCCC");
        return `<div class="color-match-item" data-match-name="${escapeHtml(item.name || "")}" data-match-hex="${hex}" data-match-l="${Number(item.l) || 0}" data-match-a="${Number(item.a) || 0}" data-match-b="${Number(item.b) || 0}" style="background:#${hex};color:#fff;">
          <div class="title">${escapeHtml(item.name || "")}</div>
          <div>色卡库：${escapeHtml(item.library_name || "")}</div>
          <div>dE*00：${Number(item.delta_e_00 || 0).toFixed(2)} · L ${Number(item.l).toFixed(1)} / a ${Number(item.a).toFixed(1)} / b ${Number(item.b).toFixed(1)}</div>
        </div>`;
      }).join("");
    }
    async function loadProducts(reset = true) {
      if (!canProductView) return renderProducts();
      if (state.productLoading && !reset) return;
      if (state.productMode === "query" && !reset && !state.productHasMore) return;
      const loadSeq = ++state.productLoadSeq;
      const loadAppMode = state.appMode;
      const loadProductMode = state.productMode;
      if (reset) {
        state.productOffset = 0;
        state.productHasMore = true;
        if (state.productMode === "query") state.products = [];
      }
      state.productLoading = true;
      $("productLoadMore").textContent = state.productMode === "query" ? "加载中..." : "";
      if (reset) renderProducts();
      setStatus("加载中...", false);
      const limit = state.productMode === "query" ? state.productLimit : 80;
      if (state.appMode === "mine") {
        const tag = ownerTag();
        if (!tag) {
          state.products = [];
          state.productHasMore = false;
          if (loadSeq === state.productLoadSeq) state.productLoading = false;
          renderProducts();
          setStatus("", false);
          return;
        }
        try {
          await loadPersonalFolders().catch(() => []);
          const activeTag = state.selectedPersonalFolder ? `${tag}:folder:${state.selectedPersonalFolder}` : tag;
          const mineQuery = new URLSearchParams({ limit: String(limit), offset: "0", tags: activeTag, exclude_personal: "0", include_images: "0" });
          const data = await api("/api/v1/catalog/products?" + mineQuery.toString());
          if (loadSeq !== state.productLoadSeq || state.appMode !== loadAppMode || state.productMode !== loadProductMode) return;
          state.products = data.products || [];
          state.productHasMore = false;
          state.productLoading = false;
          renderProducts();
          renderProductFilters();
          setStatus(`已加载 ${state.products.length} 条`, false);
        } finally {
          if (loadSeq === state.productLoadSeq) {
            state.productLoading = false;
            $("productLoadMore").textContent = "";
          }
        }
        return;
      }
      const query = new URLSearchParams({ limit: String(limit), offset: String(state.productMode === "query" ? state.productOffset : 0), include_images: "0" });
      query.set("exclude_personal", "1");
      if (state.productMode === "query") query.set("style_code", $("keyword").value.trim());
      const groups = { year: [], category: [], subcategory: [] };
      if (state.productMode === "query") {
        state.selectedTags.forEach((tag) => {
          const parsed = splitTag(tag);
          if (groups[parsed.type]) groups[parsed.type].push(parsed.name);
        });
        if (groups.year.length) query.set("year_tags", groups.year.join(","));
        if (groups.category.length) query.set("category_tags", groups.category.join(","));
        if (groups.subcategory.length) query.set("subcategory_tags", groups.subcategory.join(","));
      }
      try {
        const data = await api("/api/v1/catalog/products?" + query.toString());
        if (loadSeq !== state.productLoadSeq || state.appMode !== loadAppMode || state.productMode !== loadProductMode) return;
        const rows = data.products || [];
        if (state.productMode === "query") {
          state.products = reset ? rows : state.products.concat(rows);
          state.productOffset += rows.length;
          state.productHasMore = rows.length >= limit;
        } else {
          state.products = rows;
          state.productHasMore = false;
        }
        state.productLoading = false;
        renderProducts();
        renderProductFilters();
        setStatus(`已加载 ${state.products.length} 条`, false);
      } finally {
        if (loadSeq === state.productLoadSeq) {
          state.productLoading = false;
        }
        if (loadSeq === state.productLoadSeq && state.productMode === "query") {
          $("productLoadMore").textContent = state.productHasMore ? "向下滑动加载更多" : (state.products.length ? "已加载全部" : "");
        }
      }
    }
    async function loadColors() {
      if (!canColorView) return renderColors();
      if (state.colorMode !== "query") {
        state.colors = [];
        renderColors();
        setStatus("", false);
        return;
      }
      if (state.colorView === "mine" && !state.colorLibraries.length) {
        await loadColorLibraries();
      }
      setStatus("加载中...", false);
      if (state.colorView === "mine") {
        const tag = ownerTag();
        const ids = activeColorLibraryIds();
        const idsWithoutFavorites = ids.filter((id) => !isVirtualFavoriteLibraryId(id));
        const favoritePromise = tag && (!ids.length || ids.some((id) => isVirtualFavoriteLibraryId(id)))
          ? api("/api/v1/color-card/favorites?" + new URLSearchParams({ limit: "1000", user_tag: tag }).toString())
          : Promise.resolve({ cards: [] });
        const libraryPromises = (idsWithoutFavorites.length ? idsWithoutFavorites : []).map((libraryId) => {
          const query = new URLSearchParams({ limit: "1000", keyword: "" });
          if (libraryId) query.set("library_id", libraryId);
          return api("/api/v1/color-card/cards?" + query.toString());
        });
        const [favoriteData, ...libraryBatches] = await Promise.all([favoritePromise, ...libraryPromises]);
        const seen = new Set();
        state.colors = []
          .concat((favoriteData.cards || []).map((item) => ({ ...item, is_favorite: true })))
          .concat(libraryBatches.flatMap((data) => data.cards || []))
          .filter((item) => {
            const key = String(item.id || `${item.library_id}:${item.name}`);
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
          });
        renderColors();
        setStatus(`已加载 ${state.colors.length} 条`, false);
        return;
      }
      const ids = activeColorLibraryIds();
      const idsWithoutFavorites = ids.filter((id) => !isVirtualFavoriteLibraryId(id));
      const keyword = $("keyword").value.trim();
      const batches = await Promise.all((idsWithoutFavorites.length ? idsWithoutFavorites : [""]).map((libraryId) => {
        const query = new URLSearchParams({ limit: "100", keyword });
        if (libraryId) query.set("library_id", libraryId);
        return api("/api/v1/color-card/cards?" + query.toString());
      }));
      state.colors = batches.flatMap((data) => data.cards || []);
      renderColors();
      setStatus(`已加载 ${state.colors.length} 条`, false);
    }
    function loadCurrent() {
      (state.type === "color" ? loadColors() : loadProducts(true)).catch((err) => setStatus(err.message || "加载失败", true));
    }
    function thumbnailUrl(raw, edge = 260) {
      const text = String(raw || "").trim();
      if (!text) return "";
      try {
        const url = new URL(text, location.origin);
        if (url.pathname.startsWith("/images/")) {
          url.searchParams.set("max_edge", String(edge));
          url.searchParams.set("q", "68");
        }
        return url.pathname + url.search + url.hash;
      } catch (_) {
        return text;
      }
    }
    function renderImageSearchResults(rows) {
      const box = $("imageSearchResults");
      const list = rows || [];
      if (!list.length) {
        box.innerHTML = '<div class="empty">暂无匹配结果</div>';
        return;
      }
      box.innerHTML = list.map((item, index) => `
        <div class="image-result-card" data-index="${index}">
          <img src="${thumbnailUrl(item.catalog_cover_image_url || item.best_standard_image_url || "", 210)}" alt="${escapeHtml(item.style_code || "")}" loading="lazy" />
          <div class="image-result-body">
            <div class="title">${escapeHtml(item.style_code || "")}</div>
            <div class="muted">相似度 ${Math.round(Number(item.score || 0) * 100)}%</div>
          </div>
        </div>
      `).join("");
      box.querySelectorAll("[data-index]").forEach((card) => {
        card.addEventListener("click", async () => {
          const item = list[Number(card.dataset.index || 0)] || {};
          const code = String(item.style_code || "").trim();
          if (!code) return;
          try {
            const detail = await api("/api/v1/catalog/products/" + encodeURIComponent(code));
            await openGallery(detail);
          } catch (err) {
            setStatus(err.message || "打开产品失败", true);
          }
        });
      });
    }
    function setImageSearchFile(file) {
      state.imageSearchFile = file || null;
      if (state.imageSearchPreviewUrl) URL.revokeObjectURL(state.imageSearchPreviewUrl);
      state.imageSearchCrop = null;
      state.imageSearchDrag = null;
      renderImageCropRect();
      if (file) {
        state.imageSearchPreviewUrl = URL.createObjectURL(file);
        $("imageSearchPreview").src = state.imageSearchPreviewUrl;
        $("imagePreviewStage").classList.remove("hidden");
        $("imageUploadEmpty").classList.add("hidden");
        $("imageUploadBox").classList.add("has-image");
      } else {
        state.imageSearchPreviewUrl = "";
        $("imageSearchPreview").removeAttribute("src");
        $("imagePreviewStage").classList.add("hidden");
        $("imageUploadEmpty").classList.remove("hidden");
        $("imageUploadBox").classList.remove("has-image");
      }
    }
    function openImageChoice() {
      $("imageChoiceModal").classList.add("open");
    }
    function closeImageChoice() {
      $("imageChoiceModal").classList.remove("open");
    }
    function chooseImageSource(kind) {
      closeImageChoice();
      if (kind === "camera") $("imageSearchCamera").click();
      else $("imageSearchFile").click();
    }
    function imagePoint(event) {
      const rect = $("imageSearchPreview").getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(rect.width, Number(event.clientX || 0) - rect.left)),
        y: Math.max(0, Math.min(rect.height, Number(event.clientY || 0) - rect.top)),
        w: rect.width,
        h: rect.height,
      };
    }
    function renderImageCropRect() {
      const rect = $("imageCropRect");
      if (!rect) return;
      const crop = state.imageSearchCrop;
      if (!crop || crop.w <= 0 || crop.h <= 0) {
        rect.classList.add("hidden");
        return;
      }
      rect.classList.remove("hidden");
      rect.style.left = `${crop.x * 100}%`;
      rect.style.top = `${crop.y * 100}%`;
      rect.style.width = `${crop.w * 100}%`;
      rect.style.height = `${crop.h * 100}%`;
    }
    function startImageCrop(event) {
      if (!state.imageSearchFile) {
        openImageChoice();
        return;
      }
      const p = imagePoint(event);
      state.imageSearchDrag = { x: p.x, y: p.y, w: p.w, h: p.h };
      state.imageSearchCrop = { x: p.x / p.w, y: p.y / p.h, w: 0.001, h: 0.001 };
      $("imagePreviewStage").setPointerCapture?.(event.pointerId);
      renderImageCropRect();
    }
    function moveImageCrop(event) {
      const start = state.imageSearchDrag;
      if (!start) return;
      const p = imagePoint(event);
      const left = Math.min(start.x, p.x);
      const top = Math.min(start.y, p.y);
      const width = Math.abs(p.x - start.x);
      const height = Math.abs(p.y - start.y);
      state.imageSearchCrop = {
        x: left / start.w,
        y: top / start.h,
        w: width / start.w,
        h: height / start.h,
      };
      renderImageCropRect();
    }
    function endImageCrop(event) {
      if (!state.imageSearchDrag) return;
      $("imagePreviewStage").releasePointerCapture?.(event.pointerId);
      state.imageSearchDrag = null;
      const crop = state.imageSearchCrop;
      if (!crop || crop.w < 0.04 || crop.h < 0.04) {
        state.imageSearchCrop = null;
        renderImageCropRect();
      }
    }
    async function runImageSearch() {
      const file = state.imageSearchFile || ($("imageSearchFile").files || [])[0];
      if (!file) {
        setStatus("请先选择图片", true);
        return;
      }
      const button = $("runImageSearchBtn");
      setButtonLoading(button, true);
      try {
        setStatus("图搜中...", false);
        const form = new FormData();
        form.append("file", file);
        form.append("result_top_k", "30");
        const crop = state.imageSearchCrop;
        if (crop && crop.w > 0.04 && crop.h > 0.04) {
          form.append("crop_x", String(crop.x));
          form.append("crop_y", String(crop.y));
          form.append("crop_w", String(crop.w));
          form.append("crop_h", String(crop.h));
        }
        const headers = {};
        if (token) headers["X-Catalog-Token"] = token;
        const resp = await fetch("/search", { method: "POST", body: form, headers });
        if (resp.status === 401) throw new Error("图搜未授权，请检查内网调测配置");
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        renderImageSearchResults(data.topk_style_codes || []);
        setStatus(`找到 ${(data.topk_style_codes || []).length} 个相似款`, false);
      } finally {
        setButtonLoading(button, false);
      }
    }
    function clearImageSearch() {
      setImageSearchFile(null);
      $("imageSearchFile").value = "";
      $("imageSearchCamera").value = "";
      $("imageSearchResults").innerHTML = "";
      state.imageSearchCrop = null;
      state.imageSearchDrag = null;
      renderImageCropRect();
      setStatus("", false);
    }
    function typedTag(kind, value) {
      const clean = String(value || "").trim();
      if (!clean) return "";
      return `${kind}:${clean}`;
    }
    function ownerTag() {
      return typedTag("owner", userId);
    }
    function normalizeTags(tags) {
      const seen = new Set();
      return (tags || []).map((tag) => String(tag || "").trim()).filter((tag) => {
        if (!tag || seen.has(tag)) return false;
        seen.add(tag);
        return true;
      });
    }
    function resetImportState(clearFiles = true) {
      state.importJob = null;
      const review = $("importReviewBox");
      const list = $("importReviewList");
      if (review) review.classList.add("hidden");
      if (list) list.innerHTML = "";
      ["bulkImportCategory", "bulkImportSubcategory", "bulkImportCategorySelect", "bulkImportSubcategorySelect"].forEach((id) => {
        if ($(id)) $(id).value = "";
      });
      if (clearFiles && $("productFiles")) $("productFiles").value = "";
    }
    function setButtonLoading(button, loading) {
      if (!button) return;
      button.classList.toggle("loading", !!loading);
      button.disabled = !!loading;
    }
    function sourceImageUrl(jobId, sourceRelPath, edge = 360) {
      const query = new URLSearchParams({ source_rel_path: sourceRelPath || "", max_edge: String(edge) });
      if (token) query.set("token", token);
      return "/api/v1/catalog/imports/" + encodeURIComponent(jobId) + "/source-image?" + query.toString();
    }
    function importFilenameFromStyle(styleCode, fallbackName) {
      const clean = String(styleCode || "").trim();
      const fallback = String(fallbackName || "").trim();
      if (!clean) return fallback;
      const match = fallback.match(/(_\\d+)?(\\.[^.]+)$/);
      return clean + (match ? match[0] : ".jpg");
    }
    function renderImportReview(job) {
      state.importJob = job;
      const items = job && job.items ? job.items : [];
      $("importReviewBox").classList.toggle("hidden", !items.length);
      $("importReviewMeta").textContent = items.length ? `已识别 ${items.length} 张，请确认后入库` : "";
      $("importReviewList").innerHTML = items.map((item, index) => `
        <div class="review-card" data-index="${index}">
          <button class="review-img-btn" type="button" data-role="previewImportImage" data-index="${index}">
            <img class="review-img" src="${sourceImageUrl(job.job_id, item.source_rel_path)}" alt="${item.source_name || ""}" />
          </button>
          <div class="review-fields">
            <label class="review-check">
              <input type="checkbox" data-role="importSelected" ${item.status === "ok" ? "checked" : ""} />
              <span>${item.status === "ok" ? "导入此图" : "需人工确认后导入"}</span>
            </label>
            <input data-role="importStyleCode" value="${item.proposed_style_code || ""}" placeholder="识别款号" />
            <input class="hidden" data-role="importFilename" value="${item.target_filename || item.proposed_filename || ""}" />
            <input data-role="importYear" value="${item.year_tag || item.proposed_year_tag || ""}" placeholder="年份，如 2026" list="yearOptions" />
            <div class="muted">${item.source_name || item.source_rel_path || ""}${item.error ? " · " + item.error : ""}</div>
          </div>
        </div>
      `).join("");
      $("importReviewList").querySelectorAll('[data-role="previewImportImage"]').forEach((button) => {
        button.addEventListener("click", () => {
          const item = items[Number(button.dataset.index || "-1")];
          if (!item) return;
          openImagePreview(item.source_name || item.source_rel_path || "导入图片", sourceImageUrl(job.job_id, item.source_rel_path, 1200));
        });
      });
    }
    function sleep(ms) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    }
    function setColorStatus(message, isError) {
      const box = $("colorMeterStatus");
      const text = String(message || "").trim();
      box.textContent = text;
      box.className = text ? (isError ? "color-status err" : "color-status") : "color-status hidden";
    }
    function colorChecksum(bytes) {
      let sum = 0;
      for (let i = 0; i < bytes.length - 1; i += 1) sum += bytes[i];
      return sum & 255;
    }
    function colorU32le(n) {
      const bytes = new Uint8Array(4);
      new DataView(bytes.buffer).setUint32(0, n, true);
      return Array.from(bytes);
    }
    function colorCommand(content, responseSize, timeout, needSign = true) {
      const data = Uint8Array.from(content);
      if (needSign) data[data.length - 1] = colorChecksum(data);
      return { data, responseSize, timeout: timeout || 3000 };
    }
    function onColorNotify(event) {
      if (!colorPending) return;
      colorResponseBytes.push(...new Uint8Array(event.target.value.buffer));
      if (colorResponseBytes.length < colorPending.responseSize) return;
      const response = Uint8Array.from(colorResponseBytes);
      const pending = colorPending;
      colorPending = null;
      colorResponseBytes = [];
      clearTimeout(pending.timer);
      colorChecksum(response) === response[response.length - 1] ? pending.resolve(response) : pending.reject(new Error("色差仪返回校验失败"));
    }
    async function colorWrite(buffer) {
      if (colorCharacteristic.writeValueWithResponse) return colorCharacteristic.writeValueWithResponse(buffer);
      return colorCharacteristic.writeValue(buffer);
    }
    async function colorExec(command) {
      if (!colorCharacteristic) throw new Error("未连接色差仪");
      if (colorPending) throw new Error("已有蓝牙命令执行中");
      for (let i = 0; i < command.data.length; i += 20) await colorWrite(command.data.slice(i, i + 20));
      if (!command.responseSize) return null;
      return new Promise((resolve, reject) => {
        colorPending = {
          responseSize: command.responseSize,
          resolve,
          reject,
          timer: setTimeout(() => {
            colorPending = null;
            colorResponseBytes = [];
            reject(new Error("色差仪响应超时"));
          }, command.timeout),
        };
      });
    }
    async function measureColorLab() {
      await colorExec(colorCommand([0xf0], 0, 0, false));
      await sleep(50);
      colorMeasureId += 1;
      await colorExec(colorCommand([0xbb, 1, 0, ...colorU32le(colorMeasureId), 0, 0xff, 0], 10, 5000));
      await sleep(50);
      await colorExec(colorCommand([0xf0], 0, 0, false));
      await sleep(50);
      const data = await colorExec(colorCommand([0xbb, 3, 0, 0, 0, 0, 0, 0, 0xff, 0], 20, 3000));
      const view = new DataView(data.buffer);
      return { L: view.getFloat32(5, true), a: view.getFloat32(9, true), b: view.getFloat32(13, true) };
    }
    async function connectColorMeter() {
      if (!navigator.bluetooth) throw new Error("当前浏览器不支持 Web Bluetooth，请使用 Android Chrome 或电脑 Chrome/Edge");
      if (!window.isSecureContext) throw new Error("Web Bluetooth 需要 HTTPS 或 localhost");
      colorDevice = await navigator.bluetooth.requestDevice({ acceptAllDevices: true, optionalServices: [COLOR_SERVICE_UUID] });
      colorDevice.addEventListener("gattserverdisconnected", () => {
        colorCharacteristic = null;
        $("colorMeterMeasureBtn").disabled = true;
        setColorStatus("色差仪已断开", false);
      });
      const server = await colorDevice.gatt.connect();
      const service = await server.getPrimaryService(COLOR_SERVICE_UUID);
      colorCharacteristic = await service.getCharacteristic(COLOR_CHARACTERISTIC_UUID);
      await colorCharacteristic.startNotifications();
      colorCharacteristic.addEventListener("characteristicvaluechanged", onColorNotify);
      $("colorMeterMeasureBtn").disabled = false;
      setColorStatus("已连接：" + (colorDevice.name || "BLE 色差仪"), false);
    }
    async function waitImportJob(jobId) {
      for (let i = 0; i < 90; i += 1) {
        const job = await api("/api/v1/catalog/imports/" + encodeURIComponent(jobId));
        const total = Number(job.total || 0);
        const processed = Number(job.processed || 0);
        if (job.status === "completed") return job;
        if (job.status === "failed") throw new Error(job.message || "图片识别失败");
        setStatus(total > 0 ? `识别中 ${processed}/${total}` : "识别中...", false);
        await sleep(800);
      }
      throw new Error("图片识别超时，请稍后在后台查看导入任务");
    }
    async function uploadProducts() {
      const files = Array.from($("productFiles").files || []);
      if (!files.length) return setStatus("请选择产品图片", true);
      const form = new FormData();
      files.forEach((file) => form.append("files", file, file.name));
      const button = $("uploadProductsBtn");
      setButtonLoading(button, true);
      try {
        setStatus("上传中...", false);
        const createdJob = await api("/api/v1/catalog/imports/upload", { method: "POST", body: form });
        const job = await waitImportJob(createdJob.job_id);
        renderImportReview(job);
        setStatus("识别完成，请确认导入信息", false);
      } finally {
        setButtonLoading(button, false);
      }
    }
    function collectImportReviewItems() {
      const job = state.importJob;
      if (!job) return [];
      const bulkCategory = $("bulkImportCategory").value.trim() || $("bulkImportCategorySelect").value.trim();
      const bulkSubcategory = $("bulkImportSubcategory").value.trim() || $("bulkImportSubcategorySelect").value.trim();
      return Array.from($("importReviewList").querySelectorAll(".review-card")).map((card) => {
        const index = Number(card.dataset.index || "-1");
        const item = (job.items || [])[index] || {};
        const fallbackName = card.querySelector('[data-role="importFilename"]').value.trim() || item.proposed_filename || "";
        const styleCode = card.querySelector('[data-role="importStyleCode"]').value.trim() || item.proposed_style_code || "";
        return {
          source_rel_path: item.source_rel_path,
          selected: !!card.querySelector('[data-role="importSelected"]').checked,
          target_filename: importFilenameFromStyle(styleCode, fallbackName),
          year_tag: card.querySelector('[data-role="importYear"]').value.trim(),
          tags: normalizeTags([
            typedTag("category", bulkCategory),
            typedTag("subcategory", bulkSubcategory),
          ]),
        };
      });
    }
    async function commitImportReview() {
      const job = state.importJob;
      if (!job) return setStatus("请先上传识别图片", true);
      const rows = collectImportReviewItems();
      if (!rows.some((item) => item.selected)) {
        return setStatus("请至少选择一张要导入的图片", true);
      }
      const button = $("commitImportBtn");
      setButtonLoading(button, true);
      setStatus("正在写入产品库...", false);
      await sleep(50);
      try {
        const result = await api("/api/v1/catalog/imports/commit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ job_id: job.job_id, items: rows }),
        });
        state.importJob = null;
        $("importReviewBox").classList.add("hidden");
        $("importReviewList").innerHTML = "";
        $("productFiles").value = "";
        $("bulkImportCategory").value = "";
        $("bulkImportSubcategory").value = "";
        $("bulkImportCategorySelect").value = "";
        $("bulkImportSubcategorySelect").value = "";
        setProductTool("view");
        await loadProducts(true);
        setStatus(`产品图片已导入 ${result.imported || 0} 张`, false);
      } finally {
        setButtonLoading(button, false);
      }
    }
    function toggleImportSelection() {
      const boxes = Array.from($("importReviewList").querySelectorAll('[data-role="importSelected"]'));
      const shouldCheck = boxes.some((box) => !box.checked);
      boxes.forEach((box) => { box.checked = shouldCheck; });
    }
    function cancelImportReview() {
      resetImportState(false);
      setStatus("已取消本次导入", false);
    }
    async function saveColor() {
      const newLibrary = $("colorLibrary").value.trim();
      const selected = $("colorLibrarySelect");
      const library = newLibrary || selected.value;
      const libraryName = newLibrary || (selected.options[selected.selectedIndex]?.textContent || library).replace(/\\s*\\(\\d+\\)\\s*$/, "");
      const name = $("colorName").value.trim();
      const L = Number($("colorL").value);
      const a = Number($("colorA").value);
      const b = Number($("colorB").value);
      if (!library || !name || !Number.isFinite(L) || !Number.isFinite(a) || !Number.isFinite(b)) {
        return setStatus("请填写色卡库、色号和 Lab 数值", true);
      }
      await api("/api/v1/color-card/cards", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ library_id: library, library_name: libraryName, name, note: "", L, a, b }),
      });
      await loadColorLibraries(library);
      incrementColorNameNumber();
      await loadColors();
      setStatus("色卡已保存", false);
    }
    $("productTab").addEventListener("click", () => switchType("product"));
    $("colorTab").addEventListener("click", () => switchType("color"));
    $("categoryTab").addEventListener("click", () => switchAppMode("category"));
    $("imageSearchTab").addEventListener("click", () => switchAppMode("image"));
    $("mineTab").addEventListener("click", () => switchAppMode("mine"));
    $("productQueryTab").addEventListener("click", () => switchProductMode("query"));
    $("productManageTab").addEventListener("click", () => switchProductMode("manage"));
    $("addProductsBtn").addEventListener("click", () => setProductTool(state.productTool === "import" ? "view" : "import"));
    $("editProductsBtn").addEventListener("click", () => setProductTool(state.productTool === "edit" ? "view" : "edit"));
    $("colorQueryTab").addEventListener("click", () => switchColorMode("query"));
    $("colorManageTab").addEventListener("click", () => switchColorMode("manage"));
    document.querySelectorAll("[data-color-view]").forEach((btn) => {
      btn.addEventListener("click", () => switchColorView(btn.dataset.colorView || "instrument"));
    });
    function openColorLibraryView() {
      state.previousColorView = state.colorView === "library" ? (state.previousColorView || "instrument") : state.colorView;
      switchColorView("library");
      loadColorLibraries().catch((err) => setStatus(err.message || "加载色彩库失败", true));
    }
    $("chooseColorLibraryBtn")?.addEventListener("click", () => {
      openColorLibraryView();
    });
    $("colorMineLibraryBtn")?.addEventListener("click", () => {
      openColorLibraryView();
    });
    $("colorLibraryBackBtn")?.addEventListener("click", () => {
      switchColorView(state.previousColorView || "instrument");
    });
    $("colorLibraryAddBtn")?.addEventListener("click", openColorLibraryAddSheet);
    $("confirmColorLibraryBtn")?.addEventListener("click", () => {
      updateColorLibraryLabels();
      switchColorView(state.previousColorView || "instrument");
    });
    $("confirmAddColorLibraryBtn")?.addEventListener("click", () => addColorLibrary().catch((err) => setStatus(err.message || "新增色彩库失败", true)));
    $("cancelAddColorLibraryBtn")?.addEventListener("click", closeColorLibraryAddSheet);
    $("colorLibraryAddSheet")?.addEventListener("click", (event) => {
      if (event.target === $("colorLibraryAddSheet")) closeColorLibraryAddSheet();
    });
    ["newColorCardPrefix", "newColorCardNumber", "newColorCardSuffix"].forEach((id) => {
      $(id)?.addEventListener("input", updateNewColorCardName);
    });
    $("measureNewColorCardBtn")?.addEventListener("click", () => {
      resetColorCardMeasureButton();
      setColorCardAddLabStatus("请点击测量按钮并按设备按钮测量");
      if (!requestNativeColorMeter("colorMeterMeasure")) setColorCardAddLabStatus("请在小程序内打开后测量");
    });
    $("confirmAddColorCardBtn")?.addEventListener("click", () => saveNewColorCard().catch((err) => setStatus(err.message || "保存色卡失败", true)));
    $("cancelAddColorCardBtn")?.addEventListener("click", closeColorCardAddSheet);
    $("colorCardAddSheet")?.addEventListener("click", (event) => {
      if (event.target === $("colorCardAddSheet")) closeColorCardAddSheet();
    });
    function filterColorLibraryList() {
      const keyword = String($("colorLibrarySearch")?.value || "").trim().toLowerCase();
      const items = Array.from(document.querySelectorAll(".library-item"));
      items.forEach((item) => item.classList.toggle("hidden", keyword && !item.textContent.toLowerCase().includes(keyword)));
    }
    $("colorLibrarySearchBtn")?.addEventListener("click", filterColorLibraryList);
    $("colorLibrarySearch")?.addEventListener("input", filterColorLibraryList);
    $("colorLibraryList")?.addEventListener("pointerdown", (event) => {
      const row = event.target.closest(".library-swipe");
      if (!row || event.target.closest(".library-row-actions")) return;
      state.colorLibrarySwipe = { row, startX: event.clientX, startY: event.clientY };
    });
    $("colorLibraryList")?.addEventListener("pointermove", (event) => {
      const swipe = state.colorLibrarySwipe;
      if (!swipe || !swipe.row) return;
      const dx = event.clientX - swipe.startX;
      const dy = event.clientY - swipe.startY;
      if (Math.abs(dx) < 12 || Math.abs(dx) < Math.abs(dy) * 1.2) return;
      event.preventDefault();
      document.querySelectorAll(".library-swipe.open").forEach((item) => {
        if (item !== swipe.row) item.classList.remove("open");
      });
      swipe.row.classList.toggle("open", dx < -24);
    });
    ["pointerup", "pointercancel"].forEach((name) => {
      $("colorLibraryList")?.addEventListener(name, () => {
        state.colorLibrarySwipe = null;
      });
    });
    $("searchBtn").addEventListener("click", loadCurrent);
    $("filterBtn").addEventListener("click", openFilterModal);
    $("closeFilterBtn").addEventListener("click", closeFilterModal);
    $("resetFilterBtn").addEventListener("click", resetFilterModal);
    $("applyFilterBtn").addEventListener("click", applyFilterModal);
    $("filterModal").addEventListener("click", (event) => {
      if (event.target === $("filterModal")) closeFilterModal();
    });
    $("zoomCloseBtn").addEventListener("click", closeZoomImage);
    $("zoomModal").addEventListener("click", (event) => {
      if (event.target === $("zoomModal")) closeZoomImage();
    });
    $("addPersonalBtn").addEventListener("click", () => {
      const product = state.currentGalleryProduct;
      const action = isPersonalProduct(product) ? deleteCurrentPersonalProduct() : openPersonalProductModal();
      action.catch((err) => setStatus(err.message || "操作失败", true));
    });
    $("closePersonalProductBtn").addEventListener("click", closePersonalProductModal);
    $("cancelPersonalProductBtn").addEventListener("click", closePersonalProductModal);
    $("confirmPersonalProductBtn").addEventListener("click", () => confirmAddPersonalProduct().catch((err) => setStatus(err.message || "加入失败", true)));
    $("personalProductModal").addEventListener("click", (event) => {
      if (event.target === $("personalProductModal")) closePersonalProductModal();
    });
    $("choiceAlbumBtn").addEventListener("click", () => chooseImageSource("album"));
    $("choiceCameraBtn").addEventListener("click", () => chooseImageSource("camera"));
    $("choiceCancelBtn").addEventListener("click", closeImageChoice);
    $("imageChoiceModal").addEventListener("click", (event) => {
      if (event.target === $("imageChoiceModal")) closeImageChoice();
    });
    $("imageUploadBox").addEventListener("click", () => {
      if (!state.imageSearchFile) openImageChoice();
    });
    $("imagePreviewStage").addEventListener("click", (event) => event.stopPropagation());
    $("imagePreviewStage").addEventListener("pointerdown", startImageCrop);
    $("imagePreviewStage").addEventListener("pointermove", moveImageCrop);
    $("imagePreviewStage").addEventListener("pointerup", endImageCrop);
    $("imagePreviewStage").addEventListener("pointercancel", endImageCrop);
    $("imageSearchFile").addEventListener("change", () => setImageSearchFile(($("imageSearchFile").files || [])[0] || null));
    $("imageSearchCamera").addEventListener("change", () => setImageSearchFile(($("imageSearchCamera").files || [])[0] || null));
    $("colorImageMenuBtn")?.addEventListener("click", (event) => {
      event.stopPropagation();
      openColorImageMenu();
    });
    $("chooseColorImageBtn")?.addEventListener("click", () => {
      closeColorImageMenu();
      $("colorImageFile")?.click();
    });
    $("cancelColorImageMenuBtn")?.addEventListener("click", closeColorImageMenu);
    $("colorImageMenuSheet")?.addEventListener("click", (event) => {
      if (event.target === $("colorImageMenuSheet")) closeColorImageMenu();
    });
    $("savePickedColorBtn")?.addEventListener("click", () => {
      const target = currentColorActionTarget();
      closeColorPickActions();
      state.colorActionTarget = target;
      saveCurrentActionColorToFavorites()
        .catch((err) => setStatus(err.message || "保存失败", true))
        .finally(() => { state.colorActionTarget = null; });
    });
    $("findPickedColorBtn")?.addEventListener("click", () => {
      const target = currentColorActionTarget();
      closeColorPickActions();
      if (target) applyActionTargetToMeter(target);
      showInstrumentPickedColor();
      switchColorView("instrument");
      matchColorCards().catch((err) => setStatus(err.message || "匹配失败", true));
    });
    $("cancelPickedColorActionBtn")?.addEventListener("click", closeColorPickActions);
    $("colorPickActionSheet")?.addEventListener("click", (event) => {
      if (event.target === $("colorPickActionSheet")) closeColorPickActions();
    });
    $("colorList")?.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".my-color-menu-btn")) return;
      const row = event.target.closest(".my-color-swipe[data-favorite='1'], .my-color-swipe[data-user-card='1']");
      if (!row) return;
      state.colorFavoriteSwipe = { row, startX: event.clientX, startY: event.clientY };
    });
    $("colorList")?.addEventListener("pointermove", (event) => {
      const swipe = state.colorFavoriteSwipe;
      if (!swipe || !swipe.row) return;
      const dx = event.clientX - swipe.startX;
      const dy = event.clientY - swipe.startY;
      if (Math.abs(dx) < 12 || Math.abs(dx) < Math.abs(dy) * 1.2) return;
      event.preventDefault();
      document.querySelectorAll(".my-color-swipe.open").forEach((item) => {
        if (item !== swipe.row) item.classList.remove("open");
      });
      swipe.row.classList.toggle("open", dx < -24);
    });
    ["pointerup", "pointercancel"].forEach((name) => {
      $("colorList")?.addEventListener(name, () => {
        state.colorFavoriteSwipe = null;
      });
    });
    $("colorList")?.addEventListener("click", (event) => {
      const menuBtn = event.target.closest(".my-color-menu-btn");
      if (menuBtn) {
        event.preventDefault();
        event.stopPropagation();
        const row = menuBtn.closest(".my-color-swipe");
        const lab = {
          L: Number(row?.dataset.cardL),
          a: Number(row?.dataset.cardA),
          b: Number(row?.dataset.cardB),
        };
        const hex = normalizeHex(row?.dataset.cardHex || "");
        if (!row || !hex || !Number.isFinite(lab.L) || !Number.isFinite(lab.a) || !Number.isFinite(lab.b)) {
          return setStatus("色号数据不完整，无法操作", true);
        }
        document.querySelectorAll(".my-color-swipe.open").forEach((item) => item.classList.remove("open"));
        openColorPickActions({
          cardId: row.dataset.cardId || "",
          name: row.dataset.cardName || `色号 #${hex}`,
          hex,
          lab,
          isExistingCard: true,
          isFavorite: row.dataset.favorite === "1",
          isUserCard: row.dataset.userCard === "1",
        });
        return;
      }
      const deleteBtn = event.target.closest(".my-color-delete");
      if (!deleteBtn) return;
      event.preventDefault();
      event.stopPropagation();
      const row = deleteBtn.closest(".my-color-swipe");
      const cardId = row?.dataset.cardId || "";
      const task = row?.dataset.userCard === "1" ? deleteUserColorCard(cardId) : deleteColorFavorite(cardId);
      task.catch((err) => setStatus(err.message || "删除失败", true));
    });
    $("confirmCompareFavoriteBtn")?.addEventListener("click", () => {
      confirmCompareFavorite().catch((err) => setStatus(err.message || "加入收藏失败", true));
    });
    document.querySelectorAll(".color-action-panel").forEach((panel) => {
      panel.addEventListener("click", (event) => event.stopPropagation());
    });
    $("cancelCompareFavoriteBtn")?.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      state.colorComparePendingTarget = null;
      closeCompareConfirm();
    });
    $("colorCompareConfirmSheet")?.addEventListener("click", (event) => {
      if (event.target === $("colorCompareConfirmSheet")) {
        state.colorComparePendingTarget = null;
        closeCompareConfirm();
      }
    });
    $("colorImageStage")?.addEventListener("click", (event) => {
      event.stopPropagation();
      pickColorFromImage(event);
    });
    $("colorImageFile")?.addEventListener("change", () => setColorImageFile(($("colorImageFile").files || [])[0] || null));
    $("runImageSearchBtn").addEventListener("click", () => runImageSearch().catch((err) => setStatus(err.message || "图搜失败", true)));
    $("clearImageSearchBtn").addEventListener("click", clearImageSearch);
    $("keyword").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        loadCurrent();
      }
    });
    $("uploadProductsBtn").addEventListener("click", () => uploadProducts().catch((err) => setStatus(err.message || "上传失败", true)));
    $("commitImportBtn").addEventListener("click", () => commitImportReview().catch((err) => setStatus(err.message || "入库失败", true)));
    $("selectAllImportBtn").addEventListener("click", toggleImportSelection);
    $("cancelImportBtn").addEventListener("click", cancelImportReview);
    $("saveColorBtn").addEventListener("click", () => saveColor().catch((err) => setStatus(err.message || "保存失败", true)));
    $("matchColorBtn").addEventListener("click", () => matchColorCards().catch((err) => setStatus(err.message || "匹配失败", true)));
    $("colorMeterConnectBtn").addEventListener("click", () => {
      if (requestNativeColorMeter("colorMeterMeasure")) return;
      switchColorView("device");
    });
    document.querySelectorAll("[data-meter-mode]").forEach((btn) => {
      btn.addEventListener("click", () => setColorMeterMode(btn.dataset.meterMode || "single"));
    });
    $("meterSwatchDot")?.addEventListener("click", () => {
      setMeterPreviewMode("swatch");
    });
    $("meterParamsDot")?.addEventListener("click", () => {
      setMeterPreviewMode("params");
    });
    $("meterPreviewInner")?.addEventListener("click", (event) => {
      if (state.meterSuppressClick) {
        state.meterSuppressClick = false;
        return;
      }
      const rect = $("meterPreviewInner").getBoundingClientRect();
      setMeterPreviewMode(event.clientX - rect.left > rect.width / 2 ? "params" : "swatch");
    });
    $("meterPreviewInner")?.addEventListener("pointerdown", (event) => {
      try { $("meterPreviewInner").setPointerCapture(event.pointerId); } catch (_) {}
      startColorCompareDrag(event);
    });
    $("meterPreviewInner")?.addEventListener("pointermove", (event) => {
      moveColorCompareDrag(event);
    });
    $("meterPreviewInner")?.addEventListener("pointerup", async (event) => {
      try {
        if (await finishColorCompareDrag(event)) return;
      } catch (err) {
        cancelColorCompareDrag();
        setStatus(err.message || "加入收藏失败", true);
        return;
      }
    });
    $("meterPreviewInner")?.addEventListener("pointercancel", () => {
      cancelColorCompareDrag();
    });
    document.addEventListener("touchmove", (event) => {
      if (!state.colorCompareDragging) return;
      event.preventDefault();
    }, { passive: false });
    $("colorMeterMeasureBtn").addEventListener("click", async () => {
      try {
        setColorStatus("正在测量...", false);
        const lab = await measureColorLab();
        $("colorL").value = lab.L.toFixed(2);
        $("colorA").value = lab.a.toFixed(2);
        $("colorB").value = lab.b.toFixed(2);
        refreshColorSwatch();
        setColorStatus("测量完成", false);
        if (state.colorMode === "query") await matchColorCards();
      } catch (err) {
        setColorStatus(err.message || "测量失败", true);
      }
    });
    ["colorL", "colorA", "colorB"].forEach((id) => $(id).addEventListener("input", refreshColorSwatch));
    ["colorNamePrefix", "colorNameNumber", "colorNameSuffix"].forEach((id) => $(id).addEventListener("input", refreshColorName));
    $("colorLibrary").addEventListener("input", () => maybeFillColorNamePrefix($("colorLibrary").value));
    $("colorLibrarySelect").addEventListener("change", () => {
      const id = $("colorLibrarySelect").value;
      state.colorLibraryIds = id ? [id] : [];
      updateColorLibraryLabels();
      maybeFillColorNamePrefix($("colorLibrarySelect").options[$("colorLibrarySelect").selectedIndex]?.textContent || "");
    });
    $("closeGalleryBtn").addEventListener("click", closeGallery);
    $("galleryModal").addEventListener("click", (event) => {
      if (event.target === $("galleryModal")) closeGallery();
    });
    setColorStatus("", false);
    window.addEventListener("scroll", () => {
      if (state.type !== "product" || state.productMode !== "query" || state.productLoading || !state.productHasMore) return;
      const nearBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
      if (nearBottom) loadProducts(false).catch((err) => setStatus(err.message || "加载失败", true));
    }, { passive: true });
    setColorImageFile(null);
    $("productCreateBox").classList.toggle("hidden", !(canProductCreate && state.type === "product" && state.appMode === "category" && state.productTool === "import"));
    $("colorCreateBox").classList.toggle("hidden", !(canColorCreate && state.type === "color" && state.colorMode === "manage"));
    Promise.all([loadTags(), loadColorLibraries()]).finally(() => {
      switchType(state.type);
      if (applyNativeMeterLabFromUrl()) {
        switchColorView("instrument");
        matchColorCards().catch((err) => setStatus(err.message || "匹配失败", true));
      } else {
        applyNativeMeterDeviceFromUrl();
      }
    });
  </script>
</body>
</html>""".replace("__INITIAL_TYPE__", safe_type).replace("__SERVER_PERMISSIONS__", permissions_json).replace("__SERVER_USER_ID__", user_id_json)

    @app.get("/catalog/login", response_class=HTMLResponse)
    def catalog_login_page(request: Request, error: int = 0) -> HTMLResponse:
        if catalog_web_auth_enabled and _catalog_read_session_user(request):
            return HTMLResponse("", status_code=303, headers={"Location": "/catalog"})
        err_html = '<div class="err">用户名、密码或验证码错误</div>' if int(error or 0) else ""
        html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>产品库登录</title>
  <style>
    body {{ margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center; background: linear-gradient(180deg,#f8fafc,#eef2f7); font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    .card {{ width: min(420px, calc(100vw - 32px)); background: #fff; border-radius: 18px; box-shadow: 0 16px 36px rgba(15,23,42,.12); padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; color: #0f172a; }}
    .sub {{ color: #64748b; font-size: 14px; margin-bottom: 18px; }}
    .err {{ margin-bottom: 14px; padding: 10px 12px; border-radius: 10px; background: #fef2f2; color: #b91c1c; font-size: 14px; }}
    label {{ display: block; font-size: 13px; color: #475569; margin: 14px 0 6px; }}
    input {{ width: 100%; box-sizing: border-box; border: 1px solid #d1d5db; border-radius: 12px; padding: 12px 14px; font-size: 15px; }}
    button {{ width: 100%; margin-top: 18px; border: none; border-radius: 12px; padding: 12px 14px; font-size: 15px; background: #0f172a; color: #fff; cursor: pointer; }}
  </style>
</head>
<body>
  <form class="card" method="post" action="/catalog/login">
    <h1>产品库登录</h1>
    <div class="sub">登录后可访问 Web 产品库管理页。</div>
    {err_html}
    <label for="username">用户名</label>
    <input id="username" name="username" autocomplete="username" />
    <label for="password">密码</label>
    <input id="password" name="password" type="password" autocomplete="current-password" />
    <label for="captcha">验证码</label>
    <input id="captcha" name="captcha" autocomplete="off" />
    <button type="submit">登录</button>
  </form>
</body>
</html>"""
        return HTMLResponse(html)

    @app.post("/catalog/login")
    async def catalog_login_submit(
        username: str = Form(""),
        password: str = Form(""),
        captcha: str = Form(""),
    ) -> RedirectResponse:
        if not catalog_web_auth_enabled:
            return RedirectResponse(url="/catalog", status_code=303)
        if (not _catalog_is_login_ok(username, password)) or (not _catalog_is_captcha_ok(captcha)):
            return RedirectResponse(url="/catalog/login?error=1", status_code=303)
        session_value, exp_ts = _catalog_build_session_value(username.strip())
        resp = RedirectResponse(url="/catalog", status_code=303)
        resp.set_cookie(
            key=catalog_web_cookie_name,
            value=session_value,
            max_age=max(300, catalog_web_session_ttl_sec),
            expires=exp_ts,
            httponly=True,
            samesite="lax",
        )
        return resp

    @app.get("/catalog/logout")
    def catalog_logout() -> RedirectResponse:
        resp = RedirectResponse(url="/catalog/login", status_code=303)
        resp.delete_cookie(catalog_web_cookie_name)
        return resp

    def _catalog_mobile_response(request: Request, catalog_type: str, token: str = "") -> HTMLResponse:
        catalog_type = str(catalog_type or "").strip().lower()
        permissions = getattr(request.state, "catalog_permissions", None)
        user_id = str(getattr(request.state, "catalog_user_id", "") or "").strip()
        if str(token or "").strip() and not user_id:
            token_auth = _catalog_verify_token(str(token or "").strip())
            if token_auth:
                user_id = str(token_auth.get("user", "")).strip()
                if permissions is None:
                    permissions = set(token_auth.get("permissions") or [])
        return HTMLResponse(
            _catalog_mobile_page(
                catalog_type,
                sorted(permissions) if permissions is not None else None,
                user_id,
            )
        )

    @app.get("/catalog", response_class=HTMLResponse)
    def catalog_page(request: Request, type: str = "", token: str = "", access_token: str = ""):
        catalog_type = str(type or "").strip().lower()
        external_token = str(token or access_token or "").strip()
        if external_token or catalog_type in {"product", "color"}:
            return _catalog_mobile_response(request, catalog_type, external_token)
        return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>产品库</title>
  <style>
    body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin: 0; background: #f5f7fa; color: #111827; }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 20px; }
    .toolbar { display: grid; grid-template-columns: minmax(260px, 1.6fr) repeat(4, 124px); gap: 10px; margin-bottom: 14px; }
    input, button { font-size: 14px; padding: 8px 12px; border-radius: 10px; border: 1px solid #d1d5db; min-height: 42px; box-sizing: border-box; }
    button { cursor: pointer; background: #111827; color: #fff; border: none; }
    button.secondary { background: #fff; color: #111827; border: 1px solid #d1d5db; }
    button.weak { background: #f8fafc; color: #64748b; border: 1px solid #dbe2ea; }
    .muted { color: #6b7280; font-size: 13px; }
    .filter-tags { display: flex; flex-wrap: wrap; gap: 6px 10px; margin: 0 0 14px; min-height: 24px; }
    .filter-tag-section { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }
    .filter-tag-section > .muted { font-size: 12px; margin-right: 2px !important; }
    .filter-tag { border: 1px solid #c7d2fe; background: #eef2ff; color: #3730a3; border-radius: 999px; padding: 4px 8px; font-size: 12px; cursor: pointer; min-height: 26px; }
    .filter-tag.active { background: #3730a3; color: #fff; border-color: #3730a3; }
    .filter-tag-wrap { display: inline-flex; align-items: center; border: 1px solid #c7d2fe; background: #eef2ff; border-radius: 999px; overflow: hidden; }
    .filter-tag-wrap.active { background: #3730a3; border-color: #3730a3; }
    .filter-tag-wrap .filter-tag { border: none; background: transparent; border-radius: 0; }
    .filter-tag-wrap.active .filter-tag { color: #fff; }
    .filter-tag-delete { min-height: 24px; height: 100%; padding: 3px 7px 3px 1px; border: none; border-radius: 0; background: transparent; color: #64748b; font-size: 12px; line-height: 1; }
    .filter-tag-wrap.active .filter-tag-delete { color: #e0e7ff; }
    .filter-tag-delete:hover { color: #b91c1c; background: rgba(255,255,255,0.5); }
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
    .card { background: #fff; border-radius: 14px; padding: 14px; box-shadow: 0 4px 18px rgba(0,0,0,0.06); }
    .thumb { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; background: #e5e7eb; border-radius: 10px; cursor: pointer; }
    .code { font-weight: 700; margin: 10px 0 8px; }
    .card-meta { margin-bottom: 8px; }
    .card-section-title { font-size: 12px; color: #64748b; font-weight: 700; margin: 10px 0 6px; }
    .tags { display: flex; flex-wrap: wrap; gap: 5px; min-height: 24px; margin-bottom: 10px; }
    .tag { background: #f8fafc; color: #334155; border: 1px solid #dbe2ea; border-radius: 4px; padding: 2px 6px; font-size: 11px; display: inline-flex; align-items: center; gap: 4px; line-height: 1.1; }
    .tag-remove { min-height: auto; height: auto; padding: 0; margin: 0; border: none; background: transparent; color: #475569; cursor: pointer; font-size: 12px; line-height: 1; border-radius: 0; box-shadow: none; }
    .row { display: flex; gap: 8px; position: relative; align-items: center; }
    .tag-edit { display: grid; gap: 8px; margin-top: 10px; }
    .tag-edit-grid { display: grid; grid-template-columns: 1fr; gap: 7px; }
    .tag-edit-field { display: grid; grid-template-columns: 42px minmax(0, 1fr); align-items: center; gap: 8px; }
    .tag-edit-field label { font-size: 12px; color: #64748b; font-weight: 700; white-space: nowrap; }
    .tag-edit-field input { width: 100%; min-width: 0; min-height: 34px; padding: 6px 9px; font-size: 13px; border-radius: 9px; }
    .quick-picks { grid-column: 2 / -1; display: flex; flex-wrap: wrap; gap: 5px; margin-top: -2px; }
    .quick-pick { min-height: 24px; padding: 3px 8px; border-radius: 999px; border: 1px solid #dbe2ea; background: #f8fafc; color: #475569; font-size: 12px; }
    .quick-pick:hover { background: #eef2ff; color: #3730a3; border-color: #c7d2fe; }
    .tag-edit .picker-add-btn { width: 100%; min-height: 34px; font-weight: 700; }
    .picker-trigger { flex: 1; font-size: 12px; padding: 6px 10px; border-radius: 9px; border: 1px dashed #c7d2fe; background: #f8faff; min-height: 30px; display: flex; align-items: center; color: #6366f1; cursor: pointer; }
    .picker-trigger.active { border-color: #818cf8; background: #eef2ff; box-shadow: none; }
    .picker-pop { position: absolute; left: 0; right: 64px; top: calc(100% + 8px); background: #fff; border: 1px solid #d1d5db; border-radius: 12px; box-shadow: 0 10px 28px rgba(0,0,0,0.12); padding: 10px; display: none; z-index: 10; }
    .picker-pop.open { display: block; }
    .picker-options { display: flex; flex-wrap: wrap; gap: 8px; max-height: 180px; overflow: auto; }
    .picker-option-item { display: inline-flex; align-items: center; gap: 4px; border: 1px solid #c7d2fe; background: #eef2ff; color: #3730a3; border-radius: 999px; padding: 4px 8px; font-size: 12px; }
    .picker-option-item.active { background: #3730a3; color: #fff; border-color: #3730a3; }
    .picker-option { border: none; background: transparent; color: inherit; padding: 0; min-height: auto; height: auto; font-size: 12px; cursor: pointer; }
    .picker-option-delete { border: none; background: transparent; color: inherit; padding: 0; min-height: auto; height: auto; font-size: 12px; cursor: pointer; opacity: 0.8; }
    .picker-add-btn { min-width: 52px; padding: 6px 9px; font-size: 12px; border-radius: 9px; }
    .status { margin: 8px 0 16px; min-height: 20px; }
    .logout-btn { display:inline-flex; align-items:center; justify-content:center; height:38px; padding:0 14px; border-radius:10px; background:#fff1f2; color:#be123c; border:1px solid #fecdd3; text-decoration:none; font-size:14px; font-weight:600; }
    .logout-btn:hover { background:#ffe4e6; }
    .modal { position: fixed; inset: 0; background: rgba(17,24,39,0.68); display: none; align-items: center; justify-content: center; padding: 24px; z-index: 999; }
    .modal.open { display: flex; }
    .modal-panel { width: min(1080px, 100%); max-height: 90vh; overflow: auto; background: #fff; border-radius: 16px; padding: 18px; }
    .modal-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; }
    .modal-head > div:first-child { min-width: 0; flex: 1; }
    .modal-close-btn { flex: none; width: 42px; min-width: 42px; min-height: 42px; padding: 0; border-radius: 10px; border: 1px solid #e5e7eb; background: #fff; color: #111827; font-size: 30px; line-height: 1; font-weight: 700; display: inline-grid; place-items: center; }
    .app-confirm-modal { position: fixed; inset: 0; z-index: 1200; display: none; align-items: center; justify-content: center; padding: 24px; background: rgba(17,24,39,.58); }
    .app-confirm-modal.open { display: flex; }
    .app-confirm-box { width: min(380px, 100%); background: #fff; border-radius: 14px; padding: 18px; box-shadow: 0 24px 70px rgba(15,23,42,.28); }
    .app-confirm-message { color: #111827; font-size: 15px; line-height: 1.55; white-space: pre-line; }
    .app-confirm-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 18px; }
    .app-confirm-actions button { min-height: 38px; border-radius: 10px; padding: 0 16px; }
    .modal-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }
    .gallery-item { position: relative; background: #f9fafb; border-radius: 12px; padding: 10px; }
    .gallery-item img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 10px; background: #e5e7eb; }
    .gallery-caption { margin-top: 8px; font-size: 12px; color: #4b5563; word-break: break-all; }
    .gallery-delete-btn { position: absolute; right: 18px; top: 18px; width: 32px; min-width: 32px; min-height: 32px; padding: 0; border-radius: 999px; background: rgba(190,18,60,.92); color: #fff; font-size: 22px; line-height: 1; display: grid; place-items: center; }
    .import-panel { width: min(1120px, 100%); }
    .import-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; margin-bottom: 12px; }
    .import-source-block { border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px; margin-bottom: 12px; background: #fafbfc; }
    .import-source-title { font-size: 12px; font-weight: 700; color: #334155; margin-bottom: 8px; }
    .import-upload-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; }
    .import-upload-row input[type="file"] { width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #d1d5db; border-radius: 10px; background: #fff; }
    .import-progress { height: 10px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin-bottom: 10px; }
    .import-progress-bar { height: 100%; width: 0%; background: linear-gradient(90deg, #4f46e5, #6366f1); transition: width 0.2s ease; }
    .import-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .import-table th, .import-table td { padding: 9px 8px; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top; }
    .import-table input[type="text"] { width: 100%; box-sizing: border-box; padding: 8px 10px; font-size: 13px; }
    .import-table tr.row-error td { background: #fff1f2; }
    .import-select-cell { white-space: nowrap; vertical-align: middle !important; }
    .import-select-label { display: inline-flex; align-items: center; gap: 7px; white-space: nowrap; line-height: 1; font-weight: 600; color: #111827; }
    .import-select-label input[type="checkbox"] { flex: none; margin: 0; }
    .import-batch-tag-box { border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px; margin: 0 0 12px; background: #fafbfc; }
    .import-batch-tag-title { font-size: 12px; color: #475569; margin-bottom: 8px; }
    .tag-admin-box { margin-top: 8px; }
    .import-tag-list { display: flex; flex-wrap: wrap; gap: 4px; min-height: 22px; margin-bottom: 6px; }
    .import-tag-chip { display: inline-flex; align-items: center; gap: 4px; background: #f8fafc; color: #334155; border: 1px solid #dbe2ea; border-radius: 4px; padding: 2px 6px; font-size: 11px; line-height: 1.1; }
    .import-tag-remove { min-height: auto; height: auto; padding: 0; margin: 0; border: none; background: transparent; color: #64748b; cursor: pointer; font-size: 12px; line-height: 1; }
    .import-batch-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .import-batch-field label { display: block; font-size: 12px; color: #475569; font-weight: 600; margin-bottom: 6px; }
    .import-batch-field input { width: 100%; box-sizing: border-box; min-height: 34px; padding: 6px 8px; font-size: 12px; }
    .import-batch-field .quick-picks { margin-top: 6px; }
    .import-tag-row { display: grid; grid-template-columns: 1fr 72px; gap: 6px; }
    .import-tag-row input[type="text"] { min-height: 34px; padding: 6px 8px; font-size: 12px; }
    .import-tag-add-btn { min-height: 34px; padding: 6px 8px; font-size: 12px; border-radius: 8px; }
    .import-badge { display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 8px; font-size: 12px; }
    .import-badge.ok { background: #ecfdf5; color: #047857; }
    .import-badge.warn { background: #fff7ed; color: #c2410c; }
    .import-source-link { border: none; background: transparent; padding: 0; margin: 0; color: #2563eb; cursor: pointer; font-size: 13px; font-weight: 600; text-align: left; }
    .import-source-link:hover { text-decoration: underline; }
    .import-actions { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-top: 14px; }
    .import-table-wrap { max-height: 52vh; overflow: auto; border: 1px solid #e5e7eb; border-radius: 12px; }
    .import-preview-img { width: 100%; max-height: 72vh; object-fit: contain; background: #f9fafb; border-radius: 12px; }
    .color-meter-panel { width: min(1040px, 100%); }
    .color-meter-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .color-meter-card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #fafbfc; }
    .color-meter-row { display: grid; grid-template-columns: 96px minmax(0, 1fr); gap: 10px; align-items: center; margin: 10px 0; }
    .color-meter-row label { color: #475569; font-size: 13px; font-weight: 700; }
    .color-meter-row input, .color-meter-row select, .color-meter-row textarea { width: 100%; min-height: 38px; box-sizing: border-box; }
    .color-meter-row textarea { min-height: 72px; resize: vertical; }
    .color-library-manage { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: center; }
    .color-library-delete-btn { min-height: 38px; padding: 0 12px; border-radius: 10px; border: 1px solid #fecdd3; background: #fff1f2; color: #be123c; font-weight: 700; }
    .color-library-delete-btn:disabled { background: #f8fafc; border-color: #d1d5db; color: #94a3b8; }
    .color-xlsx-upload-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; align-items: center; }
    .color-xlsx-upload-row input[type="file"] { width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #d1d5db; border-radius: 10px; background: #fff; }
    .color-xlsx-upload-status { min-height: 18px; margin-top: 8px; font-size: 12px; color: #64748b; }
    .color-xlsx-upload-status.err { color: #b91c1c; }
    .color-name-builder { display: grid; grid-template-columns: 1fr 96px 1fr; gap: 8px; }
    .color-meter-mode { margin-top: 12px; display: grid; gap: 8px; }
    .color-meter-mode-title { color: #475569; font-size: 13px; font-weight: 700; }
    .color-meter-mode-tabs { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .color-meter-mode-btn { min-height: 42px; border-radius: 10px; border: 1px solid #111827; background: #fff; color: #111827; font-weight: 700; }
    .color-meter-mode-btn.active { background: #111827; color: #fff; }
    .color-meter-mode-note { color: #64748b; font-size: 12px; line-height: 1.45; }
    .color-meter-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
    .color-meter-actions.end { justify-content: flex-end; }
    .color-meter-connect-actions { gap: 10px; }
    .color-meter-connect-actions button { min-height: 42px; border-radius: 10px; padding: 0 20px; font-weight: 700; }
    .color-meter-connect-primary { background: #111827; border: 1px solid #111827; color: #fff; }
    .color-meter-connect-secondary { background: #fff; border: 1px solid #111827; color: #111827; }
    .color-meter-connect-secondary:disabled { background: #f8fafc; border-color: #cbd5e1; color: #94a3b8; opacity: 1; }
    .color-meter-lab { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px; }
    .color-meter-metric { border: 1px solid #dbe2ea; border-radius: 10px; padding: 10px; background: #fff; }
    .color-meter-metric .k { color: #64748b; font-size: 12px; }
    .color-meter-metric .v { margin-top: 4px; font-size: 20px; font-weight: 700; }
    .color-meter-swatch { height: 88px; border-radius: 12px; border: 1px solid rgba(15,23,42,.16); display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: 700; margin-top: 12px; }
    .color-meter-status { padding: 9px 10px; border-radius: 10px; background: #eef6ff; color: #1e3a8a; font-size: 13px; line-height: 1.45; }
    .color-meter-status.err { background: #fee2e2; color: #b91c1c; }
    .color-match-list { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
    .color-match-item { min-height: 78px; border-radius: 12px; padding: 12px; display: flex; justify-content: space-between; gap: 12px; box-shadow: inset 0 0 0 1px rgba(15,23,42,.14); }
    .color-match-name { font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 620px; }
    .color-match-meta { margin-top: 4px; font-size: 12px; opacity: .9; }
    .input-pop-wrap { position: relative; width: 100%; }
    .input-pop-wrap > input { width: 100%; }
    .tag-suggest-pop { position: absolute; left: 0; top: calc(100% + 6px); width: max(100%, 420px); max-width: min(560px, calc(100vw - 32px)); background: #fff; border: 1px solid #d1d5db; border-radius: 12px; box-shadow: 0 10px 28px rgba(0,0,0,0.12); padding: 10px; display: none; z-index: 20; max-height: 240px; overflow: auto; scrollbar-width: thin; scrollbar-color: #cbd5e1 transparent; }
    .tag-suggest-pop.open { display: block; }
    .tag-suggest-list { display: flex; flex-wrap: wrap; gap: 6px; }
    .tag-suggest-pop .import-tag-chip.active,
    .tag-suggest-pop .import-tag-chip:hover { background: #eef2ff; border-color: #c7d2fe; }
    .tag-suggest-pop::-webkit-scrollbar { width: 8px; }
    .tag-suggest-pop::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 999px; }
    .tag-suggest-pop::-webkit-scrollbar-track { background: transparent; }
    .load-more { padding: 18px 0 8px; text-align: center; color: #6b7280; font-size: 13px; }
    @media (max-width: 720px) {
      .wrap { padding: 12px; }
      .toolbar { grid-template-columns: 1fr 1fr; }
      .toolbar input { grid-column: 1 / -1; }
      .toolbar button { min-height: 40px; }
      .filter-tags { gap: 5px 8px; margin-bottom: 12px; }
      .filter-tag-section { gap: 5px; width: 100%; }
      .filter-tag-section > .muted { width: 40px; font-size: 12px; }
      .filter-tag-section[data-type="year"] { flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; padding-bottom: 2px; }
      .filter-tag-section[data-type="year"]::-webkit-scrollbar { display: none; }
      .filter-tag-section[data-type="year"] .filter-tag-wrap { flex: 0 0 auto; }
      .filter-tag { padding: 3px 7px; font-size: 12px; min-height: 24px; }
      .filter-tag-delete { min-height: 22px; padding: 2px 6px 2px 0; font-size: 12px; }
      .cards { grid-template-columns: 1fr; gap: 12px; }
      .card { padding: 12px; }
      .modal { padding: 12px; }
      .modal-panel { padding: 14px; }
      .picker-pop { right: 0; top: calc(100% + 6px); }
      .row { flex-wrap: wrap; }
      .picker-trigger { min-width: 0; }
      .picker-add-btn { width: 100%; min-width: 0; }
      .logout-btn { height: 34px; padding: 0 10px; font-size: 13px; }
      .tag-suggest-list { flex-direction: column; gap: 8px; }
      .tag-suggest-list .import-tag-chip { width: 100%; justify-content: space-between; box-sizing: border-box; }
      .tag-suggest-pop { left: 0; right: 0; width: auto; max-width: none; max-height: 320px; }
      .import-panel { width: 100%; max-height: 94vh; border-radius: 14px; }
      .import-row,
      .import-upload-row,
      .import-batch-grid { grid-template-columns: 1fr; }
      .import-table-wrap { border: none; max-height: none; overflow: visible; }
      .import-table,
      .import-table thead,
      .import-table tbody,
      .import-table tr,
      .import-table td { display: block; width: 100%; box-sizing: border-box; }
      .import-table thead { display: none; }
      .import-table tr { border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px; margin-bottom: 10px; background: #fff; }
      .import-table th,
      .import-table td { border-bottom: none; padding: 7px 0; }
      .import-table td::before { content: attr(data-label); display: block; margin-bottom: 5px; font-size: 12px; color: #64748b; font-weight: 700; }
      .import-table td[data-label="导入"] { display: flex; align-items: center; gap: 8px; }
      .import-table td[data-label="导入"]::before { margin: 0; }
      .import-table input[type="text"] { min-height: 40px; font-size: 14px; }
      .import-source-link { font-size: 14px; line-height: 1.35; word-break: break-all; }
      .import-actions { position: sticky; bottom: -14px; background: #fff; padding-top: 10px; }
      .color-meter-grid { grid-template-columns: 1fr; }
      .color-meter-row { grid-template-columns: 1fr; gap: 6px; }
      .color-library-manage { grid-template-columns: 1fr; }
      .color-xlsx-upload-row { grid-template-columns: 1fr; }
      .color-name-builder { grid-template-columns: 1fr; }
      .color-meter-lab { grid-template-columns: 1fr; }
      .color-match-item { flex-direction: column; }
      .color-match-name { max-width: 100%; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
      <h1>产品库</h1>
      <a href="/catalog/logout" class="logout-btn">退出登录</a>
    </div>
    <div class="muted">按款号管理图片与材料标签。标签支持新增，搜索接口后续可直接按款号和标签筛选。</div>
    <div class="toolbar">
      <input id="styleCodeQuery" placeholder="按款号搜索，如 GZ25-1177 或 J0831" />
      <button id="searchBtn">查询</button>
      <button id="importBtn" class="secondary">款图录入</button>
      <button id="syncBtn" class="weak" title="手动扫描标准图片目录，款图录入完成后通常不需要点击">同步款图</button>
      <button id="colorCardBtn" class="secondary">色卡录入</button>
    </div>
    <div id="activeFilterTags" class="filter-tags"></div>
    <div id="status" class="status muted"></div>
    <div id="cards" class="cards"></div>
    <div id="loadMore" class="load-more"></div>
  </div>
  <datalist id="allTagsList"></datalist>
  <datalist id="yearTagsList"></datalist>
  <datalist id="categoryTagsList"></datalist>
  <datalist id="subcategoryTagsList"></datalist>
  <div id="galleryModal" class="modal">
    <div class="modal-panel">
      <div class="modal-head">
        <div>
          <div id="galleryTitle" class="code" style="margin:0;"></div>
          <div id="gallerySubTitle" class="muted"></div>
        </div>
        <button id="closeGalleryBtn" class="modal-close-btn" aria-label="关闭" title="关闭">×</button>
      </div>
      <div id="galleryGrid" class="modal-grid"></div>
    </div>
  </div>
  <div id="importModal" class="modal">
    <div class="modal-panel import-panel">
      <div class="modal-head">
        <div>
          <div class="code" style="margin:0;">款图录入</div>
          <div class="muted">支持服务器目录或浏览器上传图片，浏览器上传最多一次 __CATALOG_BROWSER_UPLOAD_MAX_FILES__ 张。识别后可手工修改款号、年份和最终文件名；修改款号时，导入后文件名会自动跟着改，再导入到产品库图片目录。</div>
        </div>
        <button id="closeImportBtn" class="modal-close-btn" aria-label="关闭" title="关闭">×</button>
      </div>
      <div class="import-source-block">
        <div class="import-source-title">服务器目录</div>
        <div class="import-row">
          <input id="importSourceDir" value="__CATALOG_IMPORT_SOURCE_DIR__" placeholder="例如 /data/new_samples 或 D:\\samples\\new" />
          <button id="startImportBtn">目录识别</button>
        </div>
      </div>
      <div class="import-source-block">
        <div class="import-source-title">浏览器上传</div>
        <div class="import-upload-row">
          <input id="importUploadFiles" type="file" accept="image/*" multiple />
          <button id="startUploadImportBtn" type="button">上传识别</button>
        </div>
        <div class="muted" style="margin-top:8px;">支持 jpg、jpeg、png、webp、bmp；浏览器上传最多一次 __CATALOG_BROWSER_UPLOAD_MAX_FILES__ 张图片。</div>
      </div>
      <div class="import-batch-tag-box">
        <div class="import-batch-tag-title">批量标签：年份在下方每行修改或填写，类别和细类统一加到本次勾选导入的图片所属款号</div>
        <div class="import-batch-grid">
          <div class="import-batch-field">
            <label for="importBatchCategoryInput">类别</label>
            <input id="importBatchCategoryInput" type="text" placeholder="如 单品、罗纹、毛织配件、布匹" />
            <div id="importBatchCategoryPicks" class="quick-picks"></div>
          </div>
          <div class="import-batch-field">
            <label for="importBatchSubcategoryInput">细类</label>
            <input id="importBatchSubcategoryInput" type="text" placeholder="如 暂无，或输入新增细类" />
            <div id="importBatchSubcategoryPicks" class="quick-picks"></div>
          </div>
        </div>
      </div>
      <div class="import-progress"><div id="importProgressBar" class="import-progress-bar"></div></div>
      <div id="importMeta" class="muted" style="margin-bottom:10px;"></div>
      <div class="import-table-wrap">
        <table class="import-table">
          <thead>
            <tr>
              <th style="width:92px;">导入</th>
              <th>源文件</th>
              <th style="width:170px;">款号</th>
              <th style="width:110px;">年份标签</th>
              <th>导入后文件名</th>
              <th style="width:130px;">状态</th>
            </tr>
          </thead>
          <tbody id="importTableBody">
            <tr><td colspan="6" class="muted">尚未开始导入预处理。</td></tr>
          </tbody>
        </table>
      </div>
      <div class="import-actions">
        <div id="importCommitStatus" class="muted"></div>
        <button id="commitImportBtn">确认导入</button>
      </div>
    </div>
  </div>
  <div id="importPreviewModal" class="modal">
    <div class="modal-panel">
      <div class="modal-head">
        <div>
          <div id="importPreviewTitle" class="code" style="margin:0;"></div>
          <div id="importPreviewSubTitle" class="muted"></div>
        </div>
        <button id="closeImportPreviewBtn" class="modal-close-btn" aria-label="关闭" title="关闭">×</button>
      </div>
      <img id="importPreviewImg" class="import-preview-img" alt="source preview" />
    </div>
  </div>
  <div id="colorCardModal" class="modal">
    <div class="modal-panel color-meter-panel">
      <div class="modal-head">
        <div>
          <div class="code" style="margin:0;">色卡蓝牙/文件录入</div>
        </div>
        <button id="closeColorCardBtn" class="modal-close-btn" aria-label="关闭" title="关闭">×</button>
      </div>
      <div class="color-meter-grid">
        <section class="color-meter-card">
          <div class="code" style="margin:0 0 10px;">连接色差仪</div>
          <div id="colorMeterStatus" class="color-meter-status">正在检查浏览器蓝牙能力...</div>
          <div class="color-meter-actions color-meter-connect-actions">
            <button id="colorMeterConnectBtn" class="color-meter-connect-primary">连接色差仪</button>
            <button id="colorMeterDisconnectBtn" class="color-meter-connect-secondary" disabled>断开</button>
          </div>
          <div class="color-meter-mode">
            <div class="color-meter-mode-title">测量方式</div>
            <div class="color-meter-mode-tabs">
              <button class="color-meter-mode-btn" id="colorMeterSingleModeBtn" type="button" data-web-meter-mode="single">单次测量</button>
              <button class="color-meter-mode-btn" id="colorMeterAverageModeBtn" type="button" data-web-meter-mode="average">平均测量</button>
            </div>
            <div class="color-meter-mode-note" id="colorMeterModeNote"></div>
          </div>
          <div class="color-meter-lab">
            <div class="color-meter-metric"><div class="k">L</div><div id="colorMeterL" class="v">--</div></div>
            <div class="color-meter-metric"><div class="k">a</div><div id="colorMeterA" class="v">--</div></div>
            <div class="color-meter-metric"><div class="k">b</div><div id="colorMeterB" class="v">--</div></div>
          </div>
          <div id="colorMeterSwatch" class="color-meter-swatch" style="background:#f1f5f9;color:#334155;">未测量</div>
        </section>
        <section class="color-meter-card">
          <div class="code" style="margin:0 0 10px;">录入色号</div>
          <div class="color-meter-row">
            <label>色卡库</label>
            <div class="color-library-manage">
              <select id="colorLibrarySelect"></select>
              <button id="colorDeleteLibraryBtn" class="color-library-delete-btn" type="button" disabled>删除库</button>
            </div>
          </div>
          <div class="color-meter-row"><label>新色卡库</label><input id="colorNewLibrary" placeholder="可选：输入后新建/切换到该库" /></div>
          <div class="color-meter-row">
            <label>名称模板</label>
            <div class="color-name-builder">
              <input id="colorNamePrefix" placeholder="前缀，如彩龙" />
              <input id="colorNameNumber" inputmode="numeric" placeholder="编号" />
              <input id="colorNameSuffix" placeholder="色名，如浅灰" />
            </div>
          </div>
          <div class="color-meter-row"><label>色号名称</label><input id="colorNameInput" placeholder="例如 彩龙3351浅灰" /></div>
          <div class="color-meter-actions end">
            <button id="colorMeterMeasureBtn" class="secondary" disabled>点击测量</button>
            <button id="colorSaveBtn" disabled>保存到色卡库</button>
          </div>
          <div class="muted" style="margin-top:10px;">同一色卡库内色号名称重复时，会更新原记录。</div>
        </section>
      </div>
      <div class="color-meter-card" style="margin-top:14px;">
        <div class="code" style="margin:0 0 8px;">xlsx 批量导入</div>
        <div class="color-xlsx-upload-row">
          <input id="colorXlsxFile" type="file" accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" multiple />
          <button id="colorXlsxUploadBtn" type="button">上传导入</button>
        </div>
        <div id="colorXlsxUploadStatus" class="color-xlsx-upload-status">支持一次选择多个 .xlsx；格式需与“东莞国彩丝光棉.xlsx”一致，上传后会按每个文件名创建或替换同名色卡库。</div>
      </div>
      <div class="color-meter-card" style="margin-top:14px;">
        <div class="code" style="margin:0 0 8px;">相似色号列表</div>
        <div id="colorMatchStatus" class="muted">测量后会按 dE*00 从小到大返回相似色号。</div>
        <div id="colorMatchList" class="color-match-list"></div>
      </div>
    </div>
  </div>
  <script>
    let globalTags = [];
    let globalTagGroups = { year: [], category: [], subcategory: [] };
    let currentProducts = [];
    let selectedFilterTags = [];
    let currentOffset = 0;
    let pageSize = 24;
    let hasMore = true;
    let isLoadingMore = false;
    let observer = null;
    let importJobId = '';
    let importPollTimer = null;
    let importJobData = null;
    const catalogBrowserUploadMaxFiles = __CATALOG_BROWSER_UPLOAD_MAX_FILES__;
    const els = {
      styleCodeQuery: document.getElementById('styleCodeQuery'),
      searchBtn: document.getElementById('searchBtn'),
      syncBtn: document.getElementById('syncBtn'),
      colorCardBtn: document.getElementById('colorCardBtn'),
      importBtn: document.getElementById('importBtn'),
      status: document.getElementById('status'),
      cards: document.getElementById('cards'),
      loadMore: document.getElementById('loadMore'),
      activeFilterTags: document.getElementById('activeFilterTags'),
      allTagsList: document.getElementById('allTagsList'),
      yearTagsList: document.getElementById('yearTagsList'),
      categoryTagsList: document.getElementById('categoryTagsList'),
      subcategoryTagsList: document.getElementById('subcategoryTagsList'),
      galleryModal: document.getElementById('galleryModal'),
      galleryTitle: document.getElementById('galleryTitle'),
      gallerySubTitle: document.getElementById('gallerySubTitle'),
      galleryGrid: document.getElementById('galleryGrid'),
      closeGalleryBtn: document.getElementById('closeGalleryBtn'),
      importModal: document.getElementById('importModal'),
      closeImportBtn: document.getElementById('closeImportBtn'),
      importSourceDir: document.getElementById('importSourceDir'),
      startImportBtn: document.getElementById('startImportBtn'),
      importUploadFiles: document.getElementById('importUploadFiles'),
      startUploadImportBtn: document.getElementById('startUploadImportBtn'),
      importProgressBar: document.getElementById('importProgressBar'),
      importMeta: document.getElementById('importMeta'),
      importBatchCategoryInput: document.getElementById('importBatchCategoryInput'),
      importBatchSubcategoryInput: document.getElementById('importBatchSubcategoryInput'),
      importTableBody: document.getElementById('importTableBody'),
      commitImportBtn: document.getElementById('commitImportBtn'),
      importCommitStatus: document.getElementById('importCommitStatus'),
      importPreviewModal: document.getElementById('importPreviewModal'),
      importPreviewTitle: document.getElementById('importPreviewTitle'),
      importPreviewSubTitle: document.getElementById('importPreviewSubTitle'),
      importPreviewImg: document.getElementById('importPreviewImg'),
      closeImportPreviewBtn: document.getElementById('closeImportPreviewBtn'),
      importBatchCategoryPicks: document.getElementById('importBatchCategoryPicks'),
      importBatchSubcategoryPicks: document.getElementById('importBatchSubcategoryPicks'),
      colorCardModal: document.getElementById('colorCardModal'),
      closeColorCardBtn: document.getElementById('closeColorCardBtn'),
      colorMeterStatus: document.getElementById('colorMeterStatus'),
      colorMeterConnectBtn: document.getElementById('colorMeterConnectBtn'),
      colorMeterMeasureBtn: document.getElementById('colorMeterMeasureBtn'),
      colorMeterDisconnectBtn: document.getElementById('colorMeterDisconnectBtn'),
      colorMeterModeNote: document.getElementById('colorMeterModeNote'),
      colorMeterL: document.getElementById('colorMeterL'),
      colorMeterA: document.getElementById('colorMeterA'),
      colorMeterB: document.getElementById('colorMeterB'),
      colorMeterSwatch: document.getElementById('colorMeterSwatch'),
      colorLibrarySelect: document.getElementById('colorLibrarySelect'),
      colorDeleteLibraryBtn: document.getElementById('colorDeleteLibraryBtn'),
      colorNewLibrary: document.getElementById('colorNewLibrary'),
      colorNamePrefix: document.getElementById('colorNamePrefix'),
      colorNameNumber: document.getElementById('colorNameNumber'),
      colorNameSuffix: document.getElementById('colorNameSuffix'),
      colorNameInput: document.getElementById('colorNameInput'),
      colorSaveBtn: document.getElementById('colorSaveBtn'),
      colorXlsxFile: document.getElementById('colorXlsxFile'),
      colorXlsxUploadBtn: document.getElementById('colorXlsxUploadBtn'),
      colorXlsxUploadStatus: document.getElementById('colorXlsxUploadStatus'),
      colorMatchStatus: document.getElementById('colorMatchStatus'),
      colorMatchList: document.getElementById('colorMatchList'),
    };

    function setStatus(msg, isError) {
      if (!els.status) return;
      els.status.textContent = msg || '';
      els.status.style.color = isError ? '#b91c1c' : '#6b7280';
    }

    function setNodeText(node, value) {
      if (!node) return;
      node.textContent = value || '';
    }

    function appAlert(message) {
      return new Promise((resolve) => {
        let modal = document.getElementById('appAlertModal');
        if (!modal) {
          modal = document.createElement('div');
          modal.id = 'appAlertModal';
          modal.className = 'app-confirm-modal';
          modal.innerHTML = `
            <div class="app-confirm-box">
              <div class="app-confirm-message" id="appAlertMessage"></div>
              <div class="app-confirm-actions">
                <button id="appAlertOk" type="button">确定</button>
              </div>
            </div>`;
          document.body.appendChild(modal);
        }
        document.getElementById('appAlertMessage').textContent = String(message || '');
        const ok = document.getElementById('appAlertOk');
        const cleanup = () => {
          modal.classList.remove('open');
          ok.onclick = null;
          resolve();
        };
        ok.onclick = cleanup;
        modal.classList.add('open');
        ok.focus();
      });
    }

    function appConfirm(message) {
      return new Promise((resolve) => {
        let modal = document.getElementById('appConfirmModal');
        if (!modal) {
          modal = document.createElement('div');
          modal.id = 'appConfirmModal';
          modal.className = 'app-confirm-modal';
          modal.innerHTML = `
            <div class="app-confirm-box">
              <div class="app-confirm-message" id="appConfirmMessage"></div>
              <div class="app-confirm-actions">
                <button id="appConfirmCancel" type="button" class="secondary">取消</button>
                <button id="appConfirmOk" type="button">确定</button>
              </div>
            </div>`;
          document.body.appendChild(modal);
        }
        document.getElementById('appConfirmMessage').textContent = String(message || '');
        const ok = document.getElementById('appConfirmOk');
        const cancel = document.getElementById('appConfirmCancel');
        const cleanup = (value) => {
          modal.classList.remove('open');
          ok.onclick = null;
          cancel.onclick = null;
          resolve(value);
        };
        ok.onclick = () => cleanup(true);
        cancel.onclick = () => cleanup(false);
        modal.classList.add('open');
        ok.focus();
      });
    }

    function escapeHtml(value) {
      return String(value || '').replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
    }

    const nativeFetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
      const resp = await nativeFetch(...args);
      if (resp.status === 401) {
        window.location.href = '/catalog/login';
      }
      return resp;
    };

    function uniqTags(tags) {
      return Array.from(new Set((tags || []).filter(Boolean)));
    }

    function displayTag(tag) {
      const raw = String(tag || '').trim();
      const index = raw.indexOf(':');
      return index > 0 ? raw.slice(index + 1) : raw;
    }

    function displayTagType(tag) {
      const raw = String(tag || '').trim();
      const index = raw.indexOf(':');
      const kind = index > 0 ? raw.slice(0, index) : '';
      return { year: '年份', category: '类别', subcategory: '细类' }[kind] || '标签';
    }

    function typedTag(type, value) {
      const kind = String(type || '').trim();
      const name = String(value || '').trim();
      return kind && name ? `${kind}:${name}` : '';
    }

    function splitTagsByType(tags) {
      const groups = { year: [], category: [], subcategory: [] };
      (tags || []).forEach((tag) => {
        const raw = String(tag || '').trim();
        const index = raw.indexOf(':');
        if (index > 0) {
          const kind = raw.slice(0, index);
          const name = raw.slice(index + 1).trim();
          if (groups[kind] && name) groups[kind].push(name);
          return;
        }
        if (/^20\\d{2}$/.test(raw)) groups.year.push(raw);
      });
      return {
        year: uniqTags(groups.year),
        category: uniqTags(groups.category),
        subcategory: uniqTags(groups.subcategory),
      };
    }

    function normalizeTagGroups(data) {
      const groups = (data && data.tag_groups) || {};
      return {
        year: uniqTags(groups.year || []),
        category: uniqTags(groups.category || ['单品', '罗纹', '毛织配件', '布匹']),
        subcategory: uniqTags(groups.subcategory || []).filter(name => String(name || '').trim() !== '暂无'),
      };
    }

    function quickPickHtml(type) {
      const list = ((globalTagGroups && globalTagGroups[type]) || []).filter(name => type !== 'subcategory' || String(name || '').trim() !== '暂无');
      if (!list.length) return '';
      return `<div class="quick-picks">${list.map(name => `
        <button type="button" class="quick-pick" data-role="quickPick" data-value="${name}">${name}</button>
      `).join('')}</div>`;
    }

    function bindQuickPicks(root) {
      if (!root) return;
      root.querySelectorAll('[data-role="quickPick"]').forEach((button) => {
        button.addEventListener('click', () => {
          const field = button.closest('.tag-edit-field, .import-batch-field');
          const input = field ? field.querySelector('input') : null;
          if (!input) return;
          input.value = button.dataset.value || '';
          input.dispatchEvent(new Event('input', { bubbles: true }));
        });
      });
    }

    async function saveTags(styleCode, tags) {
      const resp = await fetch('/api/v1/catalog/products/' + encodeURIComponent(styleCode) + '/tags', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tags: uniqTags(tags) })
      });
      if (!resp.ok) throw new Error(await resp.text());
    }

    async function loadGlobalTags() {
      const resp = await fetch('/api/v1/catalog/tags');
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      globalTags = data.tags || [];
      globalTagGroups = normalizeTagGroups(data);
      els.allTagsList.innerHTML = globalTags.map(tag => `<option value="${displayTag(tag)}"></option>`).join('');
      if (els.yearTagsList) {
        els.yearTagsList.innerHTML = (globalTagGroups.year || []).map(tag => `<option value="${tag}"></option>`).join('');
      }
      if (els.categoryTagsList) {
        els.categoryTagsList.innerHTML = (globalTagGroups.category || []).map(tag => `<option value="${tag}"></option>`).join('');
      }
      if (els.subcategoryTagsList) {
        els.subcategoryTagsList.innerHTML = (globalTagGroups.subcategory || []).filter(tag => String(tag || '').trim() !== '暂无').map(tag => `<option value="${tag}"></option>`).join('');
      }
      if (els.importBatchCategoryPicks) {
        els.importBatchCategoryPicks.innerHTML = (globalTagGroups.category || []).map(name => `<button type="button" class="quick-pick" data-role="quickPick" data-value="${name}">${name}</button>`).join('');
      }
      if (els.importBatchSubcategoryPicks) {
        els.importBatchSubcategoryPicks.innerHTML = (globalTagGroups.subcategory || []).filter(name => String(name || '').trim() !== '暂无').map(name => `<button type="button" class="quick-pick" data-role="quickPick" data-value="${name}">${name}</button>`).join('');
      }
      bindQuickPicks(els.importModal);
      renderActiveFilterTags();
    }

    async function deleteGlobalTag(tag) {
      const resp = await fetch('/api/v1/catalog/tags/' + encodeURIComponent(tag), { method: 'DELETE' });
      if (!resp.ok) throw new Error(await resp.text());
    }

    function closeTagSuggestPops() {
    }

    function renderTagSuggestPopover(pop, input, allowDelete) {
      if (!pop) return;
      const keyword = String((input && input.value) || '').trim().toLowerCase();
      const tags = globalTags.filter((tag) => !keyword || String(tag).toLowerCase().includes(keyword));
      let activeIndex = Number(pop.dataset.activeIndex || '0');
      if (!Number.isFinite(activeIndex) || activeIndex < 0) activeIndex = 0;
      if (tags.length && activeIndex >= tags.length) activeIndex = tags.length - 1;
      pop.dataset.activeIndex = String(activeIndex);
      if (!tags.length) {
        pop.innerHTML = '<div class="muted">暂无标签</div>';
      } else {
        pop.innerHTML = `<div class="tag-suggest-list">${tags.map((tag, index) => `
          <span class="import-tag-chip ${index === activeIndex ? 'active' : ''}">
            <button type="button" class="import-tag-remove" data-role="suggestPickTagBtn" data-tag="${tag}" title="选择标签">${tag}</button>
            ${allowDelete ? `<button type="button" class="import-tag-remove" data-role="suggestDeleteTagBtn" data-tag="${tag}" title="删除标签">×</button>` : ''}
          </span>
        `).join('')}</div>`;
      }
      pop.querySelectorAll('[data-role="suggestPickTagBtn"]').forEach((button) => {
        button.addEventListener('click', (event) => {
          event.stopPropagation();
          const tag = String(button.dataset.tag || '').trim();
          if (input) input.value = tag;
          pop.classList.remove('open');
        });
      });
      pop.querySelectorAll('[data-role="suggestDeleteTagBtn"]').forEach((button) => {
        button.addEventListener('click', async (event) => {
          event.stopPropagation();
          const tag = String(button.dataset.tag || '').trim();
          if (!tag) return;
          const ok = await appConfirm(`确认删除标签“${tag}”吗？\n\n这会同步删除所有产品与该标签的关联，且不可撤销。`);
          if (!ok) return;
          try {
            await deleteGlobalTag(tag);
            await loadGlobalTags();
            await loadProducts(true);
            setStatus('标签已删除', false);
          } catch (err) {
            setStatus(err.message || '删除标签失败', true);
          }
        });
      });
    }

    function moveTagSuggestActive(pop, input, step) {
      if (!pop) return;
      const keyword = String((input && input.value) || '').trim().toLowerCase();
      const tags = globalTags.filter((tag) => !keyword || String(tag).toLowerCase().includes(keyword));
      if (!tags.length) return;
      let activeIndex = Number(pop.dataset.activeIndex || '0');
      if (!Number.isFinite(activeIndex)) activeIndex = 0;
      activeIndex = (activeIndex + step + tags.length) % tags.length;
      pop.dataset.activeIndex = String(activeIndex);
      renderTagSuggestPopover(pop, input, true);
      pop.classList.add('open');
    }

    function pickActiveTagSuggest(pop, input) {
      if (!pop || !input) return false;
      const keyword = String(input.value || '').trim().toLowerCase();
      const tags = globalTags.filter((tag) => !keyword || String(tag).toLowerCase().includes(keyword));
      if (!tags.length) return false;
      let activeIndex = Number(pop.dataset.activeIndex || '0');
      if (!Number.isFinite(activeIndex) || activeIndex < 0 || activeIndex >= tags.length) activeIndex = 0;
      input.value = tags[activeIndex];
      pop.classList.remove('open');
      return true;
    }

    function setLoadMoreText() {
      if (!currentProducts.length && !isLoadingMore) {
        els.loadMore.textContent = '';
        return;
      }
      if (isLoadingMore) {
        els.loadMore.textContent = '加载中...';
        return;
      }
      els.loadMore.textContent = hasMore ? '继续下滑加载更多' : '已加载全部';
    }

    function buildProductQueryParams() {
      const params = new URLSearchParams();
      params.set('exclude_personal', '1');
      if (els.styleCodeQuery.value.trim()) params.set('style_code', els.styleCodeQuery.value.trim());
      const groups = splitTagsByType(selectedFilterTags);
      if (groups.year.length) params.set('year_tags', groups.year.join(','));
      if (groups.category.length) params.set('category_tags', groups.category.join(','));
      if (groups.subcategory.length) params.set('subcategory_tags', groups.subcategory.join(','));
      return params;
    }

    async function loadProducts(reset = true) {
      if (isLoadingMore) return;
      if (reset) {
        currentOffset = 0;
        hasMore = true;
        currentProducts = [];
        els.cards.innerHTML = '';
        setStatus('加载中...', false);
      } else if (!hasMore) {
        return;
      }
      isLoadingMore = true;
      setLoadMoreText();
      try {
        const params = buildProductQueryParams();
        params.set('limit', String(pageSize));
        params.set('offset', String(currentOffset));
        params.set('include_images', '0');
        const resp = await fetch('/api/v1/catalog/products?' + params.toString());
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        const rows = data.products || [];
        currentProducts = reset ? rows : currentProducts.concat(rows);
        currentOffset = currentProducts.length;
        hasMore = rows.length >= pageSize;
        renderCards(reset ? currentProducts : rows, reset);
        renderActiveFilterTags();
        setStatus(`已加载 ${currentProducts.length} 个款`, false);
      } finally {
        isLoadingMore = false;
        setLoadMoreText();
      }
    }

    function renderActiveFilterTags() {
      const sections = [
        ['年份', 'year', globalTagGroups.year || []],
        ['类别', 'category', globalTagGroups.category || []],
        ['细类', 'subcategory', globalTagGroups.subcategory || []],
      ];
      els.activeFilterTags.innerHTML = sections.map(([title, type, list]) => `
        <div class="filter-tag-section" data-type="${type}">
          <span class="muted">${title}</span>
          ${list.map(name => {
            const tag = typedTag(type, name);
            const active = selectedFilterTags.includes(tag);
            return `<span class="filter-tag-wrap ${active ? 'active' : ''}">
              <button type="button" class="filter-tag" data-role="filterTagBtn" data-tag="${tag}">${name}</button>
              <button type="button" class="filter-tag-delete" data-role="deleteFilterTagBtn" data-tag="${tag}" title="删除该标签">×</button>
            </span>`;
          }).join('')}
        </div>
      `).join('');
      els.activeFilterTags.querySelectorAll('[data-role="filterTagBtn"]').forEach((button) => {
        button.addEventListener('click', async () => {
          const tag = button.dataset.tag || '';
          if (!tag) return;
          if (selectedFilterTags.includes(tag)) {
            selectedFilterTags = selectedFilterTags.filter(x => x !== tag);
          } else {
            selectedFilterTags = uniqTags([...selectedFilterTags, tag]);
          }
          await loadProducts(true);
        });
      });
      els.activeFilterTags.querySelectorAll('[data-role="deleteFilterTagBtn"]').forEach((button) => {
        button.addEventListener('click', async (event) => {
          event.stopPropagation();
          const tag = button.dataset.tag || '';
          if (!tag) return;
          const ok = await appConfirm(`确认删除${displayTagType(tag)}标签“${displayTag(tag)}”吗？\n\n这会同步删除所有产品与该标签的关联，且不可撤销。`);
          if (!ok) return;
          try {
            await deleteGlobalTag(tag);
            selectedFilterTags = selectedFilterTags.filter(x => x !== tag);
            await loadGlobalTags();
            await loadProducts(true);
            setStatus('标签已删除', false);
          } catch (err) {
            setStatus(err.message || '删除标签失败', true);
          }
        });
      });
    }

    function toggleFilterTag(tag) {
      if (!tag) return;
      if (selectedFilterTags.includes(tag)) {
        selectedFilterTags = selectedFilterTags.filter(x => x !== tag);
      } else {
        selectedFilterTags = uniqTags([...selectedFilterTags, tag]);
      }
      loadProducts(true).catch(err => setStatus(err.message || '加载失败', true));
    }

    function buildCardTagsHtml(tags) {
      return (tags || []).map(tag => `
          <span class="tag">
            <button type="button" class="tag-remove" data-role="filterFromCardBtn" data-tag="${tag}" title="按该标签筛选">${tag}</button>
            <button type="button" class="tag-remove" data-role="removeTagBtn" data-tag="${tag}" title="删除标签">×</button>
          </span>
        `).join('');
    }

    function buildPickerOptions(selectedTags) {
      const selected = new Set(selectedTags || []);
      const options = globalTags.map(tag => `
            <span class="picker-option-item ${selected.has(tag) ? 'active' : ''}">
              <button class="picker-option" type="button" data-role="pickerOption" data-tag="${tag}">
                ${tag}
              </button>
              <button class="picker-option-delete" type="button" data-role="pickerDeleteOption" data-tag="${tag}" title="删除标签">×</button>
            </span>
      `).join('');
      return options || '<div class="muted">暂无可选标签，请先在顶部新增。</div>';
    }

    function updatePickerTrigger(trigger, pendingTags) {
      const list = uniqTags(pendingTags);
      trigger.textContent = list.length ? `+ ${list.join('、')}` : '+ 标签';
      trigger.classList.toggle('active', list.length > 0);
    }

    function closeAllPickers() {
      document.querySelectorAll('[data-role="pickerPop"]').forEach((el) => el.classList.remove('open'));
    }

    async function deleteProductImage(styleCode, imageName) {
      const resp = await fetch('/api/v1/catalog/products/' + encodeURIComponent(styleCode) + '/images/' + encodeURIComponent(imageName), { method: 'DELETE' });
      if (!resp.ok) throw new Error(await resp.text());
      return resp.json();
    }

    async function openGallery(product) {
      if (!product) return;
      els.galleryTitle.textContent = product.style_code || '';
      els.gallerySubTitle.textContent = '图片加载中...';
      els.galleryGrid.innerHTML = '<div class="muted">图片加载中...</div>';
      els.galleryModal.classList.add('open');
      product = await loadProductDetail(product);
      els.galleryTitle.textContent = product.style_code || '';
      els.gallerySubTitle.textContent = `共 ${productImageCount(product)} 张图片`;
      els.galleryGrid.innerHTML = (product.images || []).map(item => `
        <div class="gallery-item" data-image-name="${escapeHtml(item.image_name || '')}">
          <button type="button" class="gallery-delete-btn" data-role="deleteGalleryImage" data-image-name="${escapeHtml(item.image_name || '')}" title="删除图片">×</button>
          <img src="${item.image_url || ''}" loading="lazy" alt="${escapeHtml(item.image_name || '')}" />
          <div class="gallery-caption">${escapeHtml(item.image_name || '')}</div>
        </div>
      `).join('');
      els.galleryGrid.querySelectorAll('[data-role="deleteGalleryImage"]').forEach((button) => {
        button.addEventListener('click', async (event) => {
          event.stopPropagation();
          const imageName = button.dataset.imageName || '';
          if (!imageName || !product.style_code) return;
          const ok = window.confirm(`确认删除图片“${imageName}”吗？\n\n会同时删除本地图片文件，且不可撤销。`);
          if (!ok) return;
          try {
            const data = await deleteProductImage(product.style_code, imageName);
            if (data.product) {
              currentProducts = currentProducts.map(item => item.style_code === product.style_code ? data.product : item);
              await openGallery(data.product);
            } else {
              currentProducts = currentProducts.filter(item => item.style_code !== product.style_code);
              closeGallery();
            }
            renderCards(currentProducts, true);
            setStatus('图片已删除', false);
          } catch (err) {
            setStatus(err.message || '删除图片失败', true);
          }
        });
      });
    }

    function closeGallery() {
      els.galleryModal.classList.remove('open');
    }

    function openImportModal() {
      if (els.importBatchCategoryInput) els.importBatchCategoryInput.value = '';
      if (els.importBatchSubcategoryInput) els.importBatchSubcategoryInput.value = '';
      els.importModal.classList.add('open');
    }

    function closeImportModal() {
      els.importModal.classList.remove('open');
    }

    const COLOR_SERVICE_UUID = 0xFFE0;
    const COLOR_CHARACTERISTIC_UUID = 0xFFE1;
    const COLOR_METER_MODE_KEY = 'openfire_web_color_meter_measure_mode';
    const COLOR_METER_AVERAGE_SAMPLE_COUNT = 3;
    let colorDevice = null;
    let colorCharacteristic = null;
    let colorPending = null;
    let colorButtonMeasurePending = null;
    let colorPassiveMeasureBusy = false;
    let colorPassiveMeasureSamples = [];
    let colorResponseBytes = [];
    let colorNotifyBytes = [];
    let colorMeasureId = 1;
    let colorLastLab = null;

    function setColorMeterStatus(message, isError) {
      if (!els.colorMeterStatus) return;
      els.colorMeterStatus.textContent = message || '';
      els.colorMeterStatus.className = isError ? 'color-meter-status err' : 'color-meter-status';
    }

    function colorChecksum(bytes) {
      let sum = 0;
      for (let i = 0; i < bytes.length - 1; i += 1) sum += bytes[i];
      return sum & 255;
    }

    function colorU32le(n) {
      const bytes = new Uint8Array(4);
      new DataView(bytes.buffer).setUint32(0, n, true);
      return Array.from(bytes);
    }

    function colorCommand(content, responseSize, timeout, needSign) {
      const data = Uint8Array.from(content);
      if (needSign !== false) data[data.length - 1] = colorChecksum(data);
      return { data, responseSize, timeout: timeout || 3000 };
    }

    function colorWakeCommand() {
      return colorCommand([0xf0], 0, 0, false);
    }

    function colorMeasureCommand() {
      colorMeasureId += 1;
      return colorCommand([0xbb, 1, 0, ...colorU32le(colorMeasureId), 0, 0xff, 0], 10, 5000, true);
    }

    function colorGetLabCommand(modeValue) {
      const mode = Number(modeValue || 0) || 0;
      return colorCommand([0xbb, 3, mode, 0, 0, 0, 0, 0, 0xff, 0], 20, 3000, true);
    }

    function onColorNotify(event) {
      const value = event.target.value;
      const incoming = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
      if (!colorPending) {
        colorNotifyBytes.push(...incoming);
        while (colorNotifyBytes.length >= 10) {
          const start = colorNotifyBytes.indexOf(0xbb);
          if (start < 0) {
            colorNotifyBytes = [];
            return;
          }
          if (start > 0) colorNotifyBytes.splice(0, start);
          if (colorNotifyBytes.length < 10) return;
          const frame = Uint8Array.from(colorNotifyBytes.slice(0, 10));
          colorNotifyBytes.splice(0, 10);
          if (frame[0] === 0xbb && frame[1] === 1 && colorButtonMeasurePending) {
            const pending = colorButtonMeasurePending;
            colorButtonMeasurePending = null;
            clearTimeout(pending.timer);
            readColorLabAfterButton(frame[2] || 0, pending.index, pending.total).then(pending.resolve, pending.reject);
            return;
          }
          if (frame[0] === 0xbb && frame[1] === 1 && colorCharacteristic) {
            handlePassiveButtonColorMeasure(frame[2] || 0).catch((err) => setColorMeterStatus(err.message || '测量失败', true));
            return;
          }
        }
        return;
      }
      colorResponseBytes.push(...incoming);
      if (colorResponseBytes.length < colorPending.responseSize) return;
      const response = Uint8Array.from(colorResponseBytes);
      const ok = colorChecksum(response) === response[response.length - 1];
      const pending = colorPending;
      colorPending = null;
      colorResponseBytes = [];
      clearTimeout(pending.timer);
      ok ? pending.resolve(response) : pending.reject(new Error('色差仪返回校验失败'));
    }

    async function colorWrite(buffer) {
      if (colorCharacteristic.writeValueWithResponse) return colorCharacteristic.writeValueWithResponse(buffer);
      return colorCharacteristic.writeValue(buffer);
    }

    async function colorExec(command) {
      if (!colorCharacteristic) throw new Error('未连接色差仪');
      if (colorPending) throw new Error('已有蓝牙命令执行中');
      for (let i = 0; i < command.data.length; i += 20) {
        await colorWrite(command.data.slice(i, i + 20));
      }
      if (!command.responseSize) return null;
      return new Promise((resolve, reject) => {
        colorPending = {
          responseSize: command.responseSize,
          resolve,
          reject,
          timer: setTimeout(() => {
            colorPending = null;
            colorResponseBytes = [];
            reject(new Error('色差仪响应超时'));
          }, command.timeout),
        };
      });
    }

    function waitColor(ms) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    }

    async function measureColorLab() {
      await colorExec(colorWakeCommand());
      await waitColor(50);
      await colorExec(colorMeasureCommand());
      await waitColor(50);
      await colorExec(colorWakeCommand());
      await waitColor(50);
      const data = await colorExec(colorGetLabCommand());
      const view = new DataView(data.buffer);
      return {
        L: view.getFloat32(5, true),
        a: view.getFloat32(9, true),
        b: view.getFloat32(13, true),
      };
    }

    async function readColorLabAfterButton(mode, index, total) {
      setColorMeterStatus(total > 1 ? `正在读取第 ${index}/${total} 次 Lab...` : '正在读取 Lab...', false);
      await waitColor(120);
      const data = await colorExec(colorGetLabCommand(mode || 0));
      const view = new DataView(data.buffer);
      const lab = {
        L: view.getFloat32(5, true),
        a: view.getFloat32(9, true),
        b: view.getFloat32(13, true),
      };
      if (!Number.isFinite(lab.L) || !Number.isFinite(lab.a) || !Number.isFinite(lab.b)) throw new Error('色差仪返回 Lab 数据异常');
      return lab;
    }

    async function handlePassiveButtonColorMeasure(mode) {
      if (colorPassiveMeasureBusy || colorButtonMeasurePending || colorPending) return;
      colorPassiveMeasureBusy = true;
      try {
        const measureMode = readWebColorMeterMode();
        const targetCount = measureMode === 'average' ? COLOR_METER_AVERAGE_SAMPLE_COUNT : 1;
        const nextIndex = measureMode === 'average' ? colorPassiveMeasureSamples.length + 1 : 1;
        const lab = await readColorLabAfterButton(mode || 0, nextIndex, targetCount);
        if (measureMode !== 'average') {
          colorPassiveMeasureSamples = [];
          setColorLab(lab);
          setColorMeterStatus('测量完成', false);
          await matchColorCards();
          return;
        }
        colorPassiveMeasureSamples.push(lab);
        if (colorPassiveMeasureSamples.length < targetCount) {
          setColorMeterStatus(`已获取 ${colorPassiveMeasureSamples.length}/${targetCount} 次，请继续按设备顶部按钮`, false);
          return;
        }
        const sum = colorPassiveMeasureSamples.reduce((acc, item) => ({
          L: acc.L + item.L,
          a: acc.a + item.a,
          b: acc.b + item.b,
        }), { L: 0, a: 0, b: 0 });
        const avg = {
          L: sum.L / colorPassiveMeasureSamples.length,
          a: sum.a / colorPassiveMeasureSamples.length,
          b: sum.b / colorPassiveMeasureSamples.length,
        };
        colorPassiveMeasureSamples = [];
        setColorLab(avg);
        setColorMeterStatus(`平均测量完成，已获取 ${targetCount} 次`, false);
        await matchColorCards();
      } finally {
        colorPassiveMeasureBusy = false;
      }
    }

    async function clickMeasureColorLab() {
      const mode = readWebColorMeterMode();
      const targetCount = mode === 'average' ? COLOR_METER_AVERAGE_SAMPLE_COUNT : 1;
      const samples = [];
      for (let i = 0; i < targetCount; i += 1) {
        setColorMeterStatus(targetCount > 1 ? `正在点击测量 ${i + 1}/${targetCount}...` : '正在点击测量...', false);
        samples.push(await measureColorLab());
        if (i + 1 < targetCount) await waitColor(250);
      }
      if (samples.length === 1) return samples[0];
      const sum = samples.reduce((acc, item) => ({
        L: acc.L + item.L,
        a: acc.a + item.a,
        b: acc.b + item.b,
      }), { L: 0, a: 0, b: 0 });
      return {
        L: sum.L / samples.length,
        a: sum.a / samples.length,
        b: sum.b / samples.length,
      };
    }

    function cancelButtonColorMeasure(message) {
      if (!colorButtonMeasurePending) return;
      const pending = colorButtonMeasurePending;
      colorButtonMeasurePending = null;
      clearTimeout(pending.timer);
      pending.reject(new Error(message || '测量已取消'));
    }

    function waitForSingleButtonColorLab(index, total) {
      cancelButtonColorMeasure('已开始新的测量');
      colorNotifyBytes = [];
      return new Promise((resolve, reject) => {
        colorButtonMeasurePending = {
          index,
          total,
          resolve,
          reject,
          timer: setTimeout(() => {
            colorButtonMeasurePending = null;
            reject(new Error('未收到设备按键测量结果，请确认色差仪已唤醒后重试'));
          }, 60000),
        };
      });
    }

    async function waitForButtonColorLab() {
      const mode = readWebColorMeterMode();
      const targetCount = mode === 'average' ? COLOR_METER_AVERAGE_SAMPLE_COUNT : 1;
      const samples = [];
      setColorMeterStatus(targetCount > 1 ? `等待设备顶部按钮，需获取 ${targetCount} 次` : '等待设备顶部按钮...', false);
      while (samples.length < targetCount) {
        const lab = await waitForSingleButtonColorLab(samples.length + 1, targetCount);
        samples.push(lab);
        if (samples.length < targetCount) {
          setColorMeterStatus(`已获取 ${samples.length}/${targetCount} 次，请继续按设备顶部按钮测量`, false);
        }
      }
      if (samples.length === 1) return samples[0];
      const sum = samples.reduce((acc, lab) => ({
        L: acc.L + lab.L,
        a: acc.a + lab.a,
        b: acc.b + lab.b,
      }), { L: 0, a: 0, b: 0 });
      return {
        L: sum.L / samples.length,
        a: sum.a / samples.length,
        b: sum.b / samples.length,
      };
    }

    function readWebColorMeterMode() {
      return localStorage.getItem(COLOR_METER_MODE_KEY) === 'average' ? 'average' : 'single';
    }

    function setWebColorMeterMode(mode) {
      localStorage.setItem(COLOR_METER_MODE_KEY, mode === 'average' ? 'average' : 'single');
      colorPassiveMeasureSamples = [];
      renderWebColorMeterMode();
    }

    function renderWebColorMeterMode() {
      const mode = readWebColorMeterMode();
      document.querySelectorAll('[data-web-meter-mode]').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.webMeterMode === mode);
      });
      if (els.colorMeterModeNote) {
        els.colorMeterModeNote.textContent = mode === 'average'
          ? '平均测量：直接连续按 3 次设备顶部按钮，也可点击按钮测量。'
          : '单次测量：直接按 1 次设备顶部按钮，也可点击按钮测量。';
      }
    }

    function colorLabToHex(lab) {
      let y = (lab.L + 16) / 116;
      let x = lab.a / 500 + y;
      let z = y - lab.b / 200;
      const pivot = (v) => v > 6 / 29 ? Math.pow(v, 3) : (v - 16 / 116) / 7.787;
      x = pivot(x) * 0.95047;
      y = pivot(y);
      z = pivot(z) * 1.08883;
      let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
      let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
      let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;
      const gamma = (v) => Math.max(0, Math.min(255, Math.round((v > 0.0031308 ? 1.055 * Math.pow(v, 1 / 2.4) - 0.055 : 12.92 * v) * 255)));
      return [gamma(r), gamma(g), gamma(b)].map((n) => n.toString(16).padStart(2, '0')).join('').toUpperCase();
    }

    function setColorLab(lab) {
      colorLastLab = lab;
      els.colorMeterL.textContent = lab.L.toFixed(2);
      els.colorMeterA.textContent = lab.a.toFixed(2);
      els.colorMeterB.textContent = lab.b.toFixed(2);
      const hex = colorLabToHex(lab);
      els.colorMeterSwatch.textContent = '#' + hex;
      els.colorMeterSwatch.style.background = '#' + hex;
      els.colorMeterSwatch.style.color = lab.L < 55 ? '#fff' : '#0f172a';
      els.colorSaveBtn.disabled = false;
    }

    function buildColorNameFromParts() {
      const prefix = (els.colorNamePrefix && els.colorNamePrefix.value || '').trim();
      const number = (els.colorNameNumber && els.colorNameNumber.value || '').trim();
      const suffix = (els.colorNameSuffix && els.colorNameSuffix.value || '').trim();
      return `${prefix}${number}${suffix}`.trim();
    }

    function refreshColorNameFromParts() {
      const built = buildColorNameFromParts();
      if (built && els.colorNameInput) els.colorNameInput.value = built;
    }

    function inferColorNamePrefixFromText(raw) {
      const text = String(raw || '');
      if (text.includes('彩龙')) return '彩龙';
      if (text.includes('国彩')) return '国彩';
      if (text.includes('恩盛')) return '恩盛';
      return '';
    }

    function maybeFillColorNamePrefix(raw) {
      if (!els.colorNamePrefix || els.colorNamePrefix.value.trim()) return;
      const prefix = inferColorNamePrefixFromText(raw);
      if (!prefix) return;
      els.colorNamePrefix.value = prefix;
      refreshColorNameFromParts();
    }

    function incrementColorNameNumber() {
      const raw = (els.colorNameNumber && els.colorNameNumber.value || '').trim();
      if (!raw || !/^\\d+$/.test(raw)) return;
      const width = raw.length;
      const next = String(Number(raw) + 1).padStart(width, '0');
      els.colorNameNumber.value = next;
      if (els.colorNameSuffix) els.colorNameSuffix.value = '';
      refreshColorNameFromParts();
    }

    function selectedColorLibraryOption() {
      if (!els.colorLibrarySelect || els.colorLibrarySelect.selectedIndex < 0) return null;
      return els.colorLibrarySelect.options[els.colorLibrarySelect.selectedIndex] || null;
    }

    function selectedColorLibraryName() {
      const option = selectedColorLibraryOption();
      return (option ? option.textContent : '').replace(/\\s*\\(\\d+\\)\\s*$/, '').trim();
    }

    function updateColorLibraryDeleteButton() {
      if (!els.colorDeleteLibraryBtn) return;
      els.colorDeleteLibraryBtn.disabled = !els.colorLibrarySelect || !els.colorLibrarySelect.value;
    }

    async function loadColorLibraries(selectedId) {
      const resp = await fetch('/api/v1/color-card/libraries');
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      els.colorLibrarySelect.innerHTML = '';
      (data.libraries || [])
        .filter((library) => {
          const id = String(library.id || '');
          const name = String(library.name || '');
          return !(/^my_color_/i.test(id) || /我的收藏|收藏夹/.test(name));
        })
        .forEach((library) => {
        const option = document.createElement('option');
        option.value = library.id;
        option.textContent = `${library.name} (${library.color_count || 0})`;
        if (selectedId && selectedId === library.id) option.selected = true;
        els.colorLibrarySelect.appendChild(option);
      });
      updateColorLibraryDeleteButton();
    }

    async function matchColorCards() {
      if (!colorLastLab) return;
      setNodeText(els.colorMatchStatus, '正在匹配...');
      els.colorMatchList.innerHTML = '';
      const resp = await fetch('/api/v1/color-card/match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          L: colorLastLab.L,
          a: colorLastLab.a,
          b: colorLastLab.b,
          library_id: els.colorLibrarySelect.value,
          limit: 12,
        }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      setNodeText(els.colorMatchStatus, `找到 ${(data.matches || []).length} 条相似色号`);
      els.colorMatchList.innerHTML = (data.matches || []).map((item) => {
        const textColor = Number(item.l) < 55 ? '#fff' : '#0f172a';
        return `<div class="color-match-item" style="background:#${item.hex || 'CCCCCC'};color:${textColor};">
          <div><div class="color-match-name">名称：${escapeHtml(item.name || '')}</div>
          <div class="color-match-meta">色彩库：${escapeHtml(item.library_name || '')}</div>
          <div class="color-match-meta">dE*00：${Number(item.delta_e_00 || 0).toFixed(2)} · L ${Number(item.l).toFixed(1)} / a ${Number(item.a).toFixed(1)} / b ${Number(item.b).toFixed(1)}</div></div>
          <div class="tag">#${escapeHtml(item.hex || '')}</div>
        </div>`;
      }).join('');
    }

    async function saveColorCard() {
      if (!colorLastLab) throw new Error('请先测量');
      const colorName = els.colorNameInput.value.trim();
      if (!colorName) throw new Error('请填写色号名称');
      let libraryId = els.colorLibrarySelect.value;
      let libraryName = selectedColorLibraryName() || libraryId;
      if (els.colorNewLibrary.value.trim()) {
        libraryId = els.colorNewLibrary.value.trim();
        libraryName = libraryId;
      }
      const resp = await fetch('/api/v1/color-card/cards', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          library_id: libraryId,
          library_name: libraryName,
          name: colorName,
          note: '',
          L: colorLastLab.L,
          a: colorLastLab.a,
          b: colorLastLab.b,
        }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      els.colorNewLibrary.value = '';
      await loadColorLibraries(data.card.library_id);
      await matchColorCards();
      incrementColorNameNumber();
      setColorMeterStatus('已保存：' + data.card.name, false);
    }

    async function deleteSelectedColorLibrary() {
      const libraryId = els.colorLibrarySelect.value;
      const libraryName = selectedColorLibraryName() || libraryId;
      if (!libraryId) throw new Error('请选择要删除的色卡库');
      const ok = await appConfirm(`确认删除色卡库“${libraryName}”及其中所有色号？\\n\\n此操作不可撤销。`);
      if (!ok) return;
      els.colorDeleteLibraryBtn.disabled = true;
      const resp = await fetch('/api/v1/color-card/libraries/' + encodeURIComponent(libraryId) + '?allow_builtin=1', {
        method: 'DELETE',
      });
      if (!resp.ok) throw new Error(await resp.text());
      await loadColorLibraries();
      if (els.colorMatchList) els.colorMatchList.innerHTML = '';
      setNodeText(els.colorMatchStatus, '测量后会按 dE*00 从小到大返回相似色号。');
      setColorMeterStatus('已删除色卡库：' + libraryName, false);
    }

    function setColorXlsxUploadStatus(message, isError) {
      if (!els.colorXlsxUploadStatus) return;
      els.colorXlsxUploadStatus.textContent = message || '';
      els.colorXlsxUploadStatus.classList.toggle('err', !!isError);
    }

    async function uploadColorCardXlsx() {
      const files = Array.from((els.colorXlsxFile && els.colorXlsxFile.files) || []);
      if (!files.length) throw new Error('请选择 xlsx 文件');
      const invalid = files.find((file) => !/\\.xlsx$/i.test(file.name || ''));
      if (invalid) throw new Error('只支持上传 .xlsx 文件：' + (invalid.name || '未命名文件'));
      const form = new FormData();
      files.forEach((file) => {
        form.append('files', file, file.name);
      });
      els.colorXlsxUploadBtn.disabled = true;
      setColorXlsxUploadStatus(`正在上传并导入 ${files.length} 个文件...`, false);
      try {
        const resp = await fetch('/api/v1/color-card/libraries/upload-xlsx', {
          method: 'POST',
          body: form,
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        const results = data.results || [];
        const failed = results.filter((item) => !item.ok);
        const success = results.filter((item) => item.ok);
        const lastLibrary = (success[success.length - 1] || {}).library || data.library || {};
        await loadColorLibraries(lastLibrary.id || '');
        if (els.colorXlsxFile) els.colorXlsxFile.value = '';
        const totalCount = Number(data.total_count || data.count || 0);
        const message = failed.length
          ? `部分导入完成：成功 ${success.length} 个文件、${totalCount} 个色号；失败 ${failed.length} 个文件`
          : `导入成功：${success.length || 1} 个文件，共 ${totalCount} 个色号`;
        setColorXlsxUploadStatus(message, !!failed.length);
        await appAlert(failed.length ? `${message}\\n\\n${failed.map((item) => `${item.filename || '文件'}：${item.error || '导入失败'}`).join('\\n')}` : message);
      } catch (err) {
        const message = err.message || 'xlsx 导入失败';
        setColorXlsxUploadStatus(message, true);
        await appAlert('导入失败：' + message);
      } finally {
        els.colorXlsxUploadBtn.disabled = false;
      }
    }

    async function connectColorMeter() {
      if (!navigator.bluetooth) throw new Error('当前浏览器不支持 Web Bluetooth，请使用 Android Chrome 或电脑 Chrome/Edge');
      colorDevice = await navigator.bluetooth.requestDevice({ acceptAllDevices: true, optionalServices: [COLOR_SERVICE_UUID] });
      colorDevice.addEventListener('gattserverdisconnected', () => {
        colorCharacteristic = null;
        cancelButtonColorMeasure('色差仪已断开');
        els.colorMeterMeasureBtn.disabled = true;
        els.colorMeterDisconnectBtn.disabled = true;
        setColorMeterStatus('色差仪已断开', false);
      });
      const server = await colorDevice.gatt.connect();
      const service = await server.getPrimaryService(COLOR_SERVICE_UUID);
      colorCharacteristic = await service.getCharacteristic(COLOR_CHARACTERISTIC_UUID);
      await colorCharacteristic.startNotifications();
      colorCharacteristic.addEventListener('characteristicvaluechanged', onColorNotify);
      els.colorMeterMeasureBtn.disabled = false;
      els.colorMeterDisconnectBtn.disabled = false;
      setColorMeterStatus('已连接：' + (colorDevice.name || 'BLE 色差仪') + '，可直接按设备顶部按钮测量', false);
    }

    async function openColorCardModal() {
      els.colorCardModal.classList.add('open');
      renderWebColorMeterMode();
      if (!navigator.bluetooth) setColorMeterStatus('当前浏览器不支持 Web Bluetooth，请使用 Android Chrome 或电脑 Chrome/Edge', true);
      else if (!window.isSecureContext) setColorMeterStatus('Web Bluetooth 需要 HTTPS 或 localhost', true);
      else setColorMeterStatus('浏览器支持 Web Bluetooth，可以连接色差仪', false);
      await loadColorLibraries().catch((err) => setColorMeterStatus(err.message || '加载色卡库失败', true));
    }

    function closeColorCardModal() {
      els.colorCardModal.classList.remove('open');
    }

    function openImportPreview(item) {
      if (!item || !importJobId) return;
      setNodeText(els.importPreviewTitle, item.source_name || '');
      setNodeText(els.importPreviewSubTitle, item.source_rel_path || '');
      if (els.importPreviewImg) {
        const params = new URLSearchParams();
        params.set('source_rel_path', item.source_rel_path || '');
        params.set('max_edge', '1600');
        els.importPreviewImg.src = '/api/v1/catalog/imports/' + encodeURIComponent(importJobId) + '/source-image?' + params.toString();
      }
      if (els.importPreviewModal) els.importPreviewModal.classList.add('open');
    }

    function closeImportPreview() {
      if (els.importPreviewModal) els.importPreviewModal.classList.remove('open');
      if (els.importPreviewImg) els.importPreviewImg.src = '';
    }

    function stopImportPolling() {
      if (importPollTimer) {
        clearTimeout(importPollTimer);
        importPollTimer = null;
      }
    }

    function deriveYearTagFromFilename(filename) {
      const raw = String(filename || '').trim();
      if (!raw) return '';
      const stem = raw.replace(/\\.[^.]+$/, '');
      const styleCode = stem.includes('_') ? stem.slice(0, stem.lastIndexOf('_')) : stem;
      const prefix = styleCode.split('-', 1)[0] || '';
      const match = prefix.match(/(\\d{2})$/);
      return match ? `20${match[1]}` : '';
    }

    function styleCodeFromImportFilename(filename) {
      const raw = String(filename || '').trim();
      if (!raw) return '';
      const stem = raw.replace(/\\.[^.]+$/, '');
      return stem.includes('_') ? stem.slice(0, stem.lastIndexOf('_')) : stem;
    }

    function buildImportFilenameFromStyleCode(styleCode, currentFilename, fallbackExt) {
      const cleanCode = String(styleCode || '').trim().replace(/[^A-Za-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '');
      if (!cleanCode) return String(currentFilename || '').trim();
      const raw = String(currentFilename || '').trim();
      const extMatch = raw.match(/(\\.[^.]+)$/);
      const ext = extMatch ? extMatch[1] : (fallbackExt || '.jpg');
      const stem = raw.replace(/\\.[^.]+$/, '');
      const seqMatch = stem.match(/(_\\d+)$/);
      return `${cleanCode}${seqMatch ? seqMatch[1] : '_000'}${ext}`;
    }

    function normalizeImportTags(tags) {
      const seen = new Set();
      return (tags || []).map(tag => String(tag || '').trim()).filter((tag) => {
        if (!tag) return false;
        const key = tag.toLowerCase();
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    }

    function updateGlobalTagOptions(tag) {
      const value = String(tag || '').trim();
      if (!value) return;
      if (!globalTags.some(item => String(item).toLowerCase() === value.toLowerCase())) {
        globalTags = globalTags.concat([value]).sort((a, b) => String(a).localeCompare(String(b), 'zh-CN'));
        els.allTagsList.innerHTML = globalTags.map(item => `<option value="${item}"></option>`).join('');
      }
    }

    function renderImportJob(job) {
      importJobData = job;
      const total = Number(job.total || 0);
      const processed = Number(job.processed || 0);
      const pct = total > 0 ? Math.min(100, Math.round(processed * 100 / total)) : 0;
      if (els.importProgressBar) els.importProgressBar.style.width = pct + '%';
      setNodeText(els.importMeta, `${job.message || ''}${total ? ` · ${processed}/${total}` : ''}`);
      const items = job.items || [];
      if (!items.length) {
        if (els.importTableBody) els.importTableBody.innerHTML = '<tr><td colspan="6" class="muted">处理中...</td></tr>';
      } else {
        if (els.importTableBody) els.importTableBody.innerHTML = items.map((item, index) => `
          <tr class="${item.status === 'ok' ? '' : 'row-error'}">
            <td data-label="导入" class="import-select-cell"><label class="import-select-label"><input type="checkbox" data-role="importSelect" data-index="${index}" ${item.selected === false ? '' : 'checked'} /><span>导入此图</span></label></td>
            <td data-label="源文件">
              <button type="button" class="import-source-link" data-role="importPreviewBtn" data-index="${index}">${item.source_name || ''}</button>
              <div class="muted">${item.source_rel_path || ''}</div>
            </td>
            <td data-label="款号"><input type="text" data-role="importStyleCode" data-index="${index}" value="${item.style_code || item.proposed_style_code || styleCodeFromImportFilename(item.target_filename || item.proposed_filename || '')}" placeholder="如 GZ25-1177" /></td>
            <td data-label="年份标签"><input type="text" data-role="importYearTag" data-index="${index}" value="${item.year_tag || item.proposed_year_tag || ''}" list="yearTagsList" placeholder="如 2024" /></td>
            <td data-label="导入后文件名"><input type="text" data-role="importFilename" data-index="${index}" value="${item.target_filename || item.proposed_filename || ''}" /></td>
            <td data-label="状态"><span class="import-badge ${item.status === 'ok' ? 'ok' : 'warn'}">${item.status === 'ok' ? '已识别' : '需人工确认'}</span>${item.error ? `<div class="muted" style="margin-top:4px;color:#b91c1c;">${item.error}</div>` : ''}</td>
          </tr>
        `).join('');
      }
      if (els.importTableBody) {
        els.importTableBody.querySelectorAll('[data-role="importPreviewBtn"]').forEach((button) => {
          button.addEventListener('click', () => {
            const index = Number(button.dataset.index || '-1');
            const rows = importJobData && importJobData.items ? importJobData.items : [];
            if (index < 0 || index >= rows.length) return;
            openImportPreview(rows[index]);
          });
        });
        els.importTableBody.querySelectorAll('[data-role="importFilename"]').forEach((input) => {
          input.addEventListener('input', () => {
            const index = input.dataset.index || '';
            const yearInput = els.importTableBody.querySelector(`[data-role="importYearTag"][data-index="${index}"]`);
            if (!yearInput) return;
            const nextYearTag = deriveYearTagFromFilename(input.value);
            if (nextYearTag) yearInput.value = nextYearTag;
            const styleInput = els.importTableBody.querySelector(`[data-role="importStyleCode"][data-index="${index}"]`);
            if (styleInput) styleInput.value = styleCodeFromImportFilename(input.value);
          });
        });
        els.importTableBody.querySelectorAll('[data-role="importStyleCode"]').forEach((input) => {
          input.addEventListener('input', () => {
            const index = input.dataset.index || '';
            const fileInput = els.importTableBody.querySelector(`[data-role="importFilename"][data-index="${index}"]`);
            const yearInput = els.importTableBody.querySelector(`[data-role="importYearTag"][data-index="${index}"]`);
            const rows = importJobData && importJobData.items ? importJobData.items : [];
            const row = rows[Number(index)] || {};
            if (fileInput) {
              fileInput.value = buildImportFilenameFromStyleCode(input.value, fileInput.value || row.proposed_filename || row.source_name || '', '.jpg');
            }
            if (yearInput) {
              const nextYearTag = deriveYearTagFromFilename(fileInput ? fileInput.value : input.value);
              if (nextYearTag) yearInput.value = nextYearTag;
            }
          });
        });
      }
      setNodeText(els.importCommitStatus, job.committed ? '该批次已导入' : '');
      if (els.commitImportBtn) els.commitImportBtn.disabled = job.status !== 'completed' || !!job.committed;
    }

    async function pollImportJob() {
      if (!importJobId) return;
      const resp = await fetch('/api/v1/catalog/imports/' + encodeURIComponent(importJobId));
      if (!resp.ok) throw new Error(await resp.text());
      const job = await resp.json();
      renderImportJob(job);
      if (job.status === 'pending' || job.status === 'running') {
        importPollTimer = setTimeout(() => {
          pollImportJob().catch(err => {
            setNodeText(els.importCommitStatus, err.message || '导入进度查询失败');
          });
        }, 800);
      } else {
        stopImportPolling();
      }
    }

    function collectImportItemsFromTable() {
      if (!importJobData) return [];
      const rows = importJobData.items || [];
      return rows.map((item, index) => {
        const checkbox = els.importTableBody.querySelector(`[data-role="importSelect"][data-index="${index}"]`);
        const input = els.importTableBody.querySelector(`[data-role="importFilename"][data-index="${index}"]`);
        const styleInput = els.importTableBody.querySelector(`[data-role="importStyleCode"][data-index="${index}"]`);
        const yearInput = els.importTableBody.querySelector(`[data-role="importYearTag"][data-index="${index}"]`);
        const targetFilename = input ? input.value.trim() : (item.target_filename || item.proposed_filename || '');
        const editedStyleCode = styleInput ? styleInput.value.trim() : '';
        return {
          source_rel_path: item.source_rel_path,
          selected: !!(checkbox && checkbox.checked),
          tags: normalizeImportTags([
            typedTag('category', els.importBatchCategoryInput ? els.importBatchCategoryInput.value.trim() : ''),
            typedTag('subcategory', els.importBatchSubcategoryInput ? els.importBatchSubcategoryInput.value.trim() : ''),
          ]),
          year_tag: yearInput ? yearInput.value.trim() : (item.year_tag || item.proposed_year_tag || ''),
          target_filename: editedStyleCode
            ? buildImportFilenameFromStyleCode(editedStyleCode, targetFilename, '.jpg')
            : targetFilename,
        };
      });
    }

    function productImageCount(item) {
      return Number(item && item.image_count) || (item && item.images || []).length || 0;
    }

    async function loadProductDetail(product) {
      if (!product || !product.style_code) return product;
      if ((product.images || []).length >= productImageCount(product) && product.images) return product;
      const resp = await fetch('/api/v1/catalog/products/' + encodeURIComponent(product.style_code));
      if (!resp.ok) throw new Error(await resp.text());
      const detail = await resp.json();
      currentProducts = currentProducts.map(item => item.style_code === detail.style_code ? detail : item);
      return detail;
    }

    function renderCards(products, reset = true) {
      if (reset) els.cards.innerHTML = '';
      if (!products.length && reset) {
        els.cards.innerHTML = '<div class="muted">没有符合条件的产品。</div>';
        return;
      }
      products.forEach((item) => {
        const card = document.createElement('div');
        card.className = 'card';
        const groups = item.tag_groups || splitTagsByType(item.raw_tags || item.tags || []);
        const yearValue = (groups.year || []).join('、');
        const categoryValue = (groups.category || []).join('、');
        const subcategoryList = (groups.subcategory || []).filter(tag => String(tag || '').trim() !== '暂无');
        const subcategoryValue = subcategoryList.join('、');
        card.innerHTML = `
          <img class="thumb" src="${item.cover_image_url || ''}" loading="lazy" alt="${item.style_code}" title="点击查看该款全部图片" />
          <div class="code">${item.style_code}</div>
          <div class="muted card-meta">图片数：${productImageCount(item)}</div>
          <div class="card-section-title">当前标签（点击可过滤）</div>
          <div class="tags">
            ${(groups.year || []).map(tag => `<span class="tag"><button type="button" class="tag-remove" data-role="filterFromCardBtn" data-tag="${typedTag('year', tag)}">年份：${tag}</button></span>`).join('')}
            ${(groups.category || []).map(tag => `<span class="tag"><button type="button" class="tag-remove" data-role="filterFromCardBtn" data-tag="${typedTag('category', tag)}">类别：${tag}</button></span>`).join('')}
            ${subcategoryList.map(tag => `<span class="tag"><button type="button" class="tag-remove" data-role="filterFromCardBtn" data-tag="${typedTag('subcategory', tag)}">细类：${tag}</button></span>`).join('')}
          </div>
          <div class="tag-edit">
            <div class="card-section-title" style="margin:0;">标签修改</div>
            <div class="tag-edit-grid">
              <div class="tag-edit-field">
                <label>年份</label>
                <input type="text" data-role="yearInput" value="${yearValue}" placeholder="如 2026" list="yearTagsList" />
              </div>
              <div class="tag-edit-field">
                <label>类别</label>
                <input type="text" data-role="categoryInput" value="${categoryValue}" placeholder="如 单品" list="categoryTagsList" />
              </div>
              <div class="tag-edit-field">
                <label>细类</label>
                <input type="text" data-role="subcategoryInput" value="${subcategoryValue}" placeholder="如 暂无" list="subcategoryTagsList" />
              </div>
            </div>
            <button type="button" class="picker-add-btn" data-role="saveBtn">保存标签修改</button>
          </div>
        `;
        card.querySelector('.thumb').addEventListener('click', () => openGallery(item).catch(err => setStatus(err.message || '加载图片失败', true)));
        bindQuickPicks(card);
        card.querySelectorAll('[data-role="filterFromCardBtn"]').forEach((button) => {
          button.addEventListener('click', () => toggleFilterTag(button.dataset.tag || ''));
        });
        card.querySelector('[data-role="saveBtn"]').addEventListener('click', async () => {
          const splitInput = (role) => String((card.querySelector(`[data-role="${role}"]`) || {}).value || '')
            .split(/[、,，\\s]+/)
            .map(value => value.trim())
            .filter(Boolean);
          const nextTags = uniqTags([
            ...splitInput('yearInput').map(value => typedTag('year', value)),
            ...splitInput('categoryInput').map(value => typedTag('category', value)),
            ...splitInput('subcategoryInput').map(value => typedTag('subcategory', value)),
          ].filter(Boolean));
          try {
            await saveTags(item.style_code, nextTags);
            await loadGlobalTags();
            await loadProducts(true);
            setStatus('标签已保存', false);
          } catch (err) {
            setStatus(err.message || '保存失败', true);
          }
        });
        els.cards.appendChild(card);
      });
    }

    els.searchBtn.addEventListener('click', () => {
      loadProducts(true).catch(err => setStatus(err.message || '加载失败', true));
    });
    els.styleCodeQuery.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter') return;
      event.preventDefault();
      loadProducts(true).catch(err => setStatus(err.message || '加载失败', true));
    });
    els.syncBtn.addEventListener('click', async () => {
      try {
        const resp = await fetch('/api/v1/catalog/sync', { method: 'POST' });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        await loadGlobalTags();
        await loadProducts(true);
        setStatus(`同步完成：新增款 ${data.products_added}，新增/更新图 ${data.images_added_or_updated}，补年份标签 ${data.year_tags_added || 0}`, false);
      } catch (err) {
        setStatus(err.message || '同步失败', true);
      }
    });
    els.colorCardBtn.addEventListener('click', () => {
      openColorCardModal().catch(err => setColorMeterStatus(err.message || '打开色卡录入失败', true));
    });
    els.closeColorCardBtn.addEventListener('click', closeColorCardModal);
    els.colorCardModal.addEventListener('click', (event) => {
      if (event.target === els.colorCardModal) closeColorCardModal();
    });
    els.colorMeterConnectBtn.addEventListener('click', () => {
      connectColorMeter().catch(err => setColorMeterStatus(err.message || '连接失败', true));
    });
    els.colorMeterMeasureBtn.addEventListener('click', async () => {
      try {
        els.colorMeterMeasureBtn.disabled = true;
        cancelButtonColorMeasure('已切换为点击测量');
        const lab = await clickMeasureColorLab();
        setColorLab(lab);
        setColorMeterStatus(readWebColorMeterMode() === 'average' ? '平均测量完成' : '测量完成', false);
        await matchColorCards();
      } catch (err) {
        setColorMeterStatus(err.message || '测量失败', true);
      } finally {
        els.colorMeterMeasureBtn.disabled = !colorCharacteristic;
      }
    });
    els.colorMeterDisconnectBtn.addEventListener('click', () => {
      if (colorDevice && colorDevice.gatt && colorDevice.gatt.connected) colorDevice.gatt.disconnect();
      colorCharacteristic = null;
      cancelButtonColorMeasure('色差仪已断开');
      els.colorMeterMeasureBtn.disabled = true;
      els.colorMeterDisconnectBtn.disabled = true;
    });
    document.querySelectorAll('[data-web-meter-mode]').forEach((button) => {
      button.addEventListener('click', () => setWebColorMeterMode(button.dataset.webMeterMode || 'single'));
    });
    [els.colorNamePrefix, els.colorNameNumber, els.colorNameSuffix].forEach((node) => {
      if (node) node.addEventListener('input', refreshColorNameFromParts);
    });
    els.colorNewLibrary.addEventListener('input', () => {
      maybeFillColorNamePrefix(els.colorNewLibrary.value);
    });
    els.colorLibrarySelect.addEventListener('change', () => {
      const label = els.colorLibrarySelect.options[els.colorLibrarySelect.selectedIndex]?.textContent || '';
      maybeFillColorNamePrefix(label);
      updateColorLibraryDeleteButton();
    });
    els.colorSaveBtn.addEventListener('click', () => {
      saveColorCard().catch(err => setColorMeterStatus(err.message || '保存失败', true));
    });
    els.colorDeleteLibraryBtn.addEventListener('click', () => {
      deleteSelectedColorLibrary().catch(err => {
        updateColorLibraryDeleteButton();
        setColorMeterStatus(err.message || '删除色卡库失败', true);
      });
    });
    els.colorXlsxUploadBtn.addEventListener('click', () => {
      uploadColorCardXlsx().catch(err => {
        const message = err.message || 'xlsx 导入失败';
        setColorXlsxUploadStatus(message, true);
        appAlert('导入失败：' + message);
      });
    });
    els.importBtn.addEventListener('click', openImportModal);
    els.closeImportBtn.addEventListener('click', closeImportModal);
    els.importModal.addEventListener('click', (event) => {
      if (event.target === els.importModal) closeImportModal();
    });
    els.closeImportPreviewBtn.addEventListener('click', closeImportPreview);
    els.importPreviewModal.addEventListener('click', (event) => {
      if (event.target === els.importPreviewModal) closeImportPreview();
    });
    els.startImportBtn.addEventListener('click', async () => {
      try {
        const sourceDir = els.importSourceDir.value.trim();
        if (!sourceDir) {
          setNodeText(els.importCommitStatus, '请输入服务器目录');
          return;
        }
        stopImportPolling();
        setNodeText(els.importCommitStatus, '');
        if (els.importTableBody) els.importTableBody.innerHTML = '<tr><td colspan="6" class="muted">任务创建中...</td></tr>';
        const resp = await fetch('/api/v1/catalog/imports/prepare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source_dir: sourceDir })
        });
        if (!resp.ok) throw new Error(await resp.text());
        const job = await resp.json();
        importJobId = job.job_id || '';
        renderImportJob(job);
        await pollImportJob();
      } catch (err) {
        setNodeText(els.importCommitStatus, err.message || '导入预处理失败');
      }
    });
    els.startUploadImportBtn.addEventListener('click', async () => {
      try {
        const files = Array.from((els.importUploadFiles && els.importUploadFiles.files) || []);
        if (!files.length) {
          setNodeText(els.importCommitStatus, '请选择要上传的图片');
          return;
        }
        if (files.length > catalogBrowserUploadMaxFiles) {
          setNodeText(els.importCommitStatus, `浏览器上传最多一次 ${catalogBrowserUploadMaxFiles} 张图片，请分批上传`);
          return;
        }
        stopImportPolling();
        setNodeText(els.importCommitStatus, '');
        if (els.importTableBody) els.importTableBody.innerHTML = '<tr><td colspan="6" class="muted">图片上传中...</td></tr>';
        const form = new FormData();
        files.forEach((file) => form.append('files', file, file.name));
        const resp = await fetch('/api/v1/catalog/imports/upload', {
          method: 'POST',
          body: form
        });
        if (!resp.ok) throw new Error(await resp.text());
        const job = await resp.json();
        importJobId = job.job_id || '';
        renderImportJob(job);
        await pollImportJob();
      } catch (err) {
        setNodeText(els.importCommitStatus, err.message || '上传导入预处理失败');
      }
    });
    els.commitImportBtn.addEventListener('click', async () => {
      try {
        if (!importJobId) {
          setNodeText(els.importCommitStatus, '请先完成识别');
          return;
        }
        const items = collectImportItemsFromTable();
        const resp = await fetch('/api/v1/catalog/imports/commit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ job_id: importJobId, items })
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        setNodeText(els.importCommitStatus, `已导入 ${data.imported} 张；新增款 ${data.sync.products_added}，新增/更新图 ${data.sync.images_added_or_updated}`);
        await loadProducts(true);
      } catch (err) {
        setNodeText(els.importCommitStatus, err.message || '导入失败');
      }
    });
    els.closeGalleryBtn.addEventListener('click', closeGallery);
    els.galleryModal.addEventListener('click', (event) => {
      if (event.target === els.galleryModal) closeGallery();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      if (els.importPreviewModal && els.importPreviewModal.classList.contains('open')) {
        event.preventDefault();
        closeImportPreview();
        return;
      }
      if (els.galleryModal && els.galleryModal.classList.contains('open')) {
        event.preventDefault();
        closeGallery();
        return;
      }
      if (els.importModal && els.importModal.classList.contains('open')) {
        event.preventDefault();
        closeImportModal();
        return;
      }
      if (els.colorCardModal && els.colorCardModal.classList.contains('open')) {
        event.preventDefault();
        closeColorCardModal();
      }
    });
    document.addEventListener('click', () => {
      closeAllPickers();
      closeTagSuggestPops();
    });
    if ('IntersectionObserver' in window) {
      observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && hasMore && !isLoadingMore) {
            loadProducts(false).catch(err => setStatus(err.message || '加载失败', true));
          }
        });
      }, { rootMargin: '300px 0px' });
      observer.observe(els.loadMore);
    }

    Promise.all([loadGlobalTags(), loadProducts(true)]).catch(err => setStatus(err.message || '加载失败', true));
  </script>
</body>
</html>""".replace("__CATALOG_IMPORT_SOURCE_DIR__", html_escape(catalog_import_source_dir, quote=True)).replace("__CATALOG_BROWSER_UPLOAD_MAX_FILES__", str(catalog_browser_upload_max_files))

    @app.get("/product", response_class=HTMLResponse)
    def catalog_product_page(request: Request, token: str = "", access_token: str = ""):
        return _catalog_mobile_response(request, "product", str(token or access_token or "").strip())

    @app.get("/color", response_class=HTMLResponse)
    def catalog_color_page(request: Request, token: str = "", access_token: str = ""):
        return _catalog_mobile_response(request, "color", str(token or access_token or "").strip())

    @app.get("/api/v1/catalog/products")
    def api_list_catalog_products(
        request: Request,
        style_code: str = "",
        tags: str = "",
        year_tags: str = "",
        category_tags: str = "",
        subcategory_tags: str = "",
        exclude_personal: int = 1,
        include_images: int = 1,
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:view")
        _check_text_content_security(style_code, tags, year_tags, category_tags, subcategory_tags, openid=_wechat_openid_from_request(request))
        base_url = _external_base_url(request)
        tag_list = [item.strip() for item in tags.split(",") if item.strip()]
        for kind, raw in (
            ("year", year_tags),
            ("category", category_tags),
            ("subcategory", subcategory_tags),
        ):
            for item in [value.strip() for value in str(raw or "").split(",") if value.strip()]:
                typed = make_typed_tag(kind, item)
                if typed:
                    tag_list.append(typed)
        products = catalog_store.list_products(
            style_code=style_code,
            tags=tag_list,
            limit=limit,
            offset=offset,
            exclude_owner=bool(exclude_personal),
            include_images=bool(include_images),
        )
        return {"products": [_serialize_catalog_product(base_url, item) for item in products]}

    @app.get("/api/v1/catalog/products/{style_code}")
    def api_get_catalog_product(request: Request, style_code: str) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:view")
        product = catalog_store.get_product(style_code)
        if not product:
            raise HTTPException(status_code=404, detail="product not found")
        return _serialize_catalog_product(_external_base_url(request), product)

    @app.get("/api/v1/catalog/personal-folders")
    def api_list_catalog_personal_folders(request: Request, user_tag: str = "") -> Dict[str, Any]:
        _catalog_require_permission(request, "product:view")
        _check_text_content_security(user_tag, openid=_wechat_openid_from_request(request))
        owner_tag = _owner_tag_from_request(request, user_tag)
        if not owner_tag:
            return {"folders": []}
        prefix = f"{owner_tag}:folder:"
        folders = sorted(
            {
                tag[len(prefix):].strip()
                for tag in catalog_store.list_used_tags()
                if str(tag).startswith(prefix) and tag[len(prefix):].strip()
            },
            key=lambda item: item.lower(),
        )
        return {"folders": folders}

    @app.post("/api/v1/catalog/personal-products")
    def api_create_catalog_personal_product(request: Request, payload: CatalogPersonalProductRequest) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        _check_text_content_security(
            payload.source_style_code,
            payload.folder_name,
            payload.user_tag,
            *payload.image_names,
            openid=_wechat_openid_from_request(request),
        )
        owner_tag = _owner_tag_from_request(request, payload.user_tag)
        folder_name = str(payload.folder_name or "").strip()
        if not owner_tag:
            raise HTTPException(status_code=400, detail="missing user tag")
        if not folder_name:
            raise HTTPException(status_code=400, detail="missing folder name")
        source = catalog_store.get_product(str(payload.source_style_code or "").strip())
        if not source:
            raise HTTPException(status_code=404, detail="source product not found")
        allowed = {str(item.get("image_name", "")) for item in source.get("images", [])}
        selected = []
        seen = set()
        for raw in payload.image_names or []:
            name = Path(str(raw or "")).name
            if name and name in allowed and name not in seen:
                selected.append(name)
                seen.add(name)
        if not selected:
            raise HTTPException(status_code=400, detail="no valid image selected")
        owner_hash = hashlib.sha1(owner_tag.encode("utf-8")).hexdigest()[:8]
        folder_part = _safe_catalog_part(folder_name, "folder", 24)
        source_part = _safe_catalog_part(source.get("style_code", ""), "source", 32)
        unique_part = dt.datetime.now().strftime("%m%d%H%M%S") + "-" + uuid.uuid4().hex[:6]
        personal_code = f"MY-{owner_hash}-{folder_part}-{source_part}-{unique_part}"
        copied: List[str] = []
        for index, image_name in enumerate(selected):
            src_path = standard_dir / Path(image_name).name
            if not src_path.exists() or not src_path.is_file():
                raise HTTPException(status_code=404, detail=f"image not found: {image_name}")
            suffix = src_path.suffix.lower() or ".jpg"
            digest = hashlib.sha1(f"{personal_code}:{image_name}".encode("utf-8")).hexdigest()[:10]
            target_name = f"{personal_code}_{digest}_{index:03d}{suffix}"
            target_path = standard_dir / target_name
            if not target_path.exists():
                shutil.copyfile(src_path, target_path)
            copied.append(target_name)
        tags = [owner_tag, _personal_folder_tag(owner_tag, folder_name)]
        product = catalog_store.upsert_product(
            personal_code,
            copied,
            tags=tags,
            note=f"个人产品/{folder_name}，来源：{source.get('style_code', '')}",
        )
        return {
            "product": _serialize_catalog_product(_external_base_url(request), product),
            "folder": folder_name,
            "images_added": len(copied),
        }

    @app.delete("/api/v1/catalog/personal-products/{style_code}")
    def api_delete_catalog_personal_product(request: Request, style_code: str, user_tag: str = "") -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        _check_text_content_security(style_code, user_tag, openid=_wechat_openid_from_request(request))
        owner_tag = _owner_tag_from_request(request, user_tag)
        product = catalog_store.get_product(style_code)
        if not product:
            raise HTTPException(status_code=404, detail="personal product not found")
        raw_tags = set(str(tag) for tag in product.get("raw_tags", []))
        if not owner_tag or owner_tag not in raw_tags:
            raise HTTPException(status_code=403, detail="not your personal product")
        try:
            image_names = catalog_store.delete_product(style_code)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        for image_name in image_names:
            safe = Path(image_name).name
            if safe.startswith("MY-"):
                try:
                    (standard_dir / safe).unlink(missing_ok=True)
                except Exception:
                    logging.warning("failed to remove personal product image: %s", safe)
        return {"ok": True, "style_code": style_code}

    @app.delete("/api/v1/catalog/personal-products/{style_code}/images/{image_name}")
    def api_delete_catalog_personal_product_image(
        request: Request,
        style_code: str,
        image_name: str,
        user_tag: str = "",
    ) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        _check_text_content_security(style_code, image_name, user_tag, openid=_wechat_openid_from_request(request))
        owner_tag = _owner_tag_from_request(request, user_tag)
        product = catalog_store.get_product(style_code)
        if not product:
            raise HTTPException(status_code=404, detail="personal product not found")
        raw_tags = set(str(tag) for tag in product.get("raw_tags", []))
        if not owner_tag or owner_tag not in raw_tags:
            raise HTTPException(status_code=403, detail="not your personal product")
        safe_image_name = Path(image_name).name
        image_names = {str(item.get("image_name", "")) for item in product.get("images", [])}
        if safe_image_name not in image_names:
            raise HTTPException(status_code=404, detail="image not found")
        try:
            result = catalog_store.delete_product_image(style_code, safe_image_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if safe_image_name.startswith("MY-"):
            try:
                (standard_dir / safe_image_name).unlink(missing_ok=True)
            except Exception:
                logging.warning("failed to remove personal product image: %s", safe_image_name)
        remaining_product = catalog_store.get_product(style_code)
        return {
            "ok": True,
            "style_code": style_code,
            "deleted": result.get("deleted", safe_image_name),
            "remaining": result.get("remaining", []),
            "product": _serialize_catalog_product(_external_base_url(request), remaining_product) if remaining_product else None,
        }

    @app.delete("/api/v1/catalog/products/{style_code}/images/{image_name}")
    def api_delete_catalog_product_image(request: Request, style_code: str, image_name: str) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        _check_text_content_security(style_code, image_name, openid=_wechat_openid_from_request(request))
        product = catalog_store.get_product(style_code)
        if not product:
            raise HTTPException(status_code=404, detail="product not found")
        safe_image_name = Path(image_name).name
        image_names = {str(item.get("image_name", "")) for item in product.get("images", [])}
        if safe_image_name not in image_names:
            raise HTTPException(status_code=404, detail="image not found")
        try:
            with catalog_write_lock:
                result = catalog_store.delete_product_image(style_code, safe_image_name)
                try:
                    (standard_dir / safe_image_name).unlink(missing_ok=True)
                except Exception:
                    logging.warning("failed to remove product image: %s", safe_image_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        remaining_product = catalog_store.get_product(style_code)
        threading.Thread(
            target=_reload_search_assets,
            args=("catalog_product_image_delete",),
            daemon=True,
        ).start()
        return {
            "ok": True,
            "style_code": style_code,
            "deleted": result.get("deleted", safe_image_name),
            "remaining": result.get("remaining", []),
            "product": _serialize_catalog_product(_external_base_url(request), remaining_product) if remaining_product else None,
        }

    @app.put("/api/v1/catalog/products/{style_code}/tags")
    def api_replace_catalog_product_tags(request: Request, style_code: str, payload: CatalogTagUpdateRequest) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        _check_text_content_security(style_code, *payload.tags, openid=_wechat_openid_from_request(request))
        try:
            tags_local = catalog_store.replace_product_tags(style_code, payload.tags)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        parsed_groups = {"year": [], "category": [], "subcategory": []}
        for tag in tags_local:
            parsed = parse_catalog_tag(tag)
            tag_type = str(parsed.get("type", ""))
            if tag_type in parsed_groups:
                parsed_groups[tag_type].append(str(parsed.get("name", "")).strip())
        return {"style_code": style_code, "tags": tags_local, "tag_groups": parsed_groups}

    @app.get("/api/v1/catalog/tags")
    def api_list_catalog_tags(request: Request) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:view")
        return {
            "tags": catalog_store.list_tags(),
            "tag_groups": catalog_store.list_tag_groups(),
        }

    @app.get("/api/v1/color-card/libraries")
    def api_list_color_card_libraries(request: Request) -> Dict[str, Any]:
        _catalog_require_permission(request, "color:view")
        return {"libraries": color_card_store.list_libraries()}

    @app.post("/api/v1/color-card/libraries")
    def api_upsert_color_card_library(request: Request, payload: ColorCardLibraryUpsertRequest) -> Dict[str, Any]:
        _catalog_require_permission(request, "color:create")
        _check_text_content_security(payload.id, payload.name, openid=_wechat_openid_from_request(request))
        try:
            library = color_card_store.upsert_library(payload.id, payload.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"library": library, "libraries": color_card_store.list_libraries()}

    async def _import_color_card_xlsx_upload(request: Request, file: UploadFile) -> Dict[str, Any]:
        filename = Path(str(file.filename or "").replace("\\", "/").split("/")[-1]).name
        suffix = Path(filename).suffix.lower()
        if suffix != ".xlsx":
            raise ValueError("只支持上传 .xlsx 文件")
        library_name = Path(filename).stem.strip()
        if not library_name:
            raise ValueError("文件名不能为空")
        _check_text_content_security(library_name, filename, openid=_wechat_openid_from_request(request))
        content = await file.read()
        if not content:
            raise ValueError("上传文件为空")
        if len(content) > 20 * 1024 * 1024:
            raise ValueError("xlsx 文件不能超过 20MB")
        rows = read_color_rows(content)
        if not rows:
            raise ValueError("未读取到有效色卡行，请确认表头包含 名称、L、a、b")
        library_id = slugify_library_id(library_name)
        count = color_card_store.replace_library(library_id, library_name, filename, rows)
        library = {"id": library_id, "name": library_name, "source_file": filename, "color_count": count}
        return {"ok": True, "filename": filename, "library": library, "count": count}

    @app.post("/api/v1/color-card/libraries/upload-xlsx")
    async def api_upload_color_card_library_xlsx(
        request: Request,
        files: List[UploadFile] | None = File(default=None),
        file: UploadFile | None = File(default=None),
    ) -> Dict[str, Any]:
        _catalog_require_permission(request, "color:create")
        upload_files = [item for item in (files or []) if item and str(item.filename or "").strip()]
        if not upload_files and file is not None and str(file.filename or "").strip():
            upload_files = [file]
        if not upload_files:
            raise HTTPException(status_code=400, detail="请选择 xlsx 文件")

        results: List[Dict[str, Any]] = []
        for upload in upload_files:
            filename = Path(str(upload.filename or "").replace("\\", "/").split("/")[-1]).name
            try:
                results.append(await _import_color_card_xlsx_upload(request, upload))
            except ValueError as exc:
                results.append({"ok": False, "filename": filename, "error": str(exc)})
            except Exception as exc:
                logging.exception("color card xlsx import failed: %s", filename)
                results.append({"ok": False, "filename": filename, "error": f"xlsx 解析失败：{exc}"})

        success = [item for item in results if item.get("ok")]
        if not success:
            first_error = str((results[0] if results else {}).get("error") or "xlsx 导入失败")
            raise HTTPException(status_code=400, detail=first_error)
        total_count = sum(int(item.get("count") or 0) for item in success)
        last = success[-1]
        return {
            "library": last.get("library") or {},
            "count": int(last.get("count") or 0),
            "total_count": total_count,
            "results": results,
            "libraries": color_card_store.list_libraries(),
        }

    @app.delete("/api/v1/color-card/libraries/{library_id}")
    def api_delete_color_card_library(request: Request, library_id: str, allow_builtin: bool = False) -> Dict[str, Any]:
        _catalog_require_permission(request, "color:create")
        _check_text_content_security(library_id, openid=_wechat_openid_from_request(request))
        try:
            removed = color_card_store.remove_library(library_id, allow_builtin=allow_builtin)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": bool(removed), "libraries": color_card_store.list_libraries()}

    @app.get("/api/v1/color-card/cards")
    def api_list_color_cards(
        request: Request,
        library_id: str = "",
        keyword: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        _catalog_require_permission(request, "color:view")
        _check_text_content_security(library_id, keyword, openid=_wechat_openid_from_request(request))
        return {"cards": color_card_store.list_cards(library_id=library_id, keyword=keyword, limit=limit, offset=offset)}

    @app.post("/api/v1/color-card/cards")
    def api_upsert_color_card(request: Request, payload: ColorCardUpsertRequest) -> Dict[str, Any]:
        _catalog_require_permission(request, "color:create")
        _check_text_content_security(payload.library_id, payload.library_name, payload.name, payload.note, openid=_wechat_openid_from_request(request))
        try:
            card = color_card_store.upsert_card(
                library_id=payload.library_id,
                library_name=payload.library_name,
                name=payload.name,
                note=payload.note,
                illuminant=payload.illuminant,
                angle=payload.angle,
                l=payload.L,
                a=payload.a,
                b=payload.b,
                spectral=payload.spectral,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"card": card}

    @app.delete("/api/v1/color-card/cards/{card_id}")
    def api_delete_user_color_card(request: Request, card_id: int, user_tag: str = "") -> Dict[str, Any]:
        _catalog_require_permission(request, "color:create")
        _check_text_content_security(user_tag, openid=_wechat_openid_from_request(request))
        try:
            removed = color_card_store.remove_user_card(user_tag=user_tag, card_id=card_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"removed": removed}

    @app.get("/api/v1/color-card/favorites")
    def api_list_color_card_favorites(
        request: Request,
        user_tag: str = "",
        keyword: str = "",
        limit: int = 300,
        offset: int = 0,
    ) -> Dict[str, Any]:
        _catalog_require_permission(request, "color:view")
        _check_text_content_security(user_tag, keyword, openid=_wechat_openid_from_request(request))
        return {"cards": color_card_store.list_favorites(user_tag=user_tag, keyword=keyword, limit=limit, offset=offset)}

    @app.post("/api/v1/color-card/favorites")
    def api_add_color_card_favorite(request: Request, payload: ColorCardFavoriteRequest) -> Dict[str, Any]:
        _catalog_require_permission(request, "color:view")
        _check_text_content_security(payload.user_tag, openid=_wechat_openid_from_request(request))
        try:
            card = color_card_store.add_favorite(payload.user_tag, payload.card_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"card": card}

    @app.delete("/api/v1/color-card/favorites/{card_id}")
    def api_delete_color_card_favorite(request: Request, card_id: int, user_tag: str = "") -> Dict[str, Any]:
        _catalog_require_permission(request, "color:view")
        _check_text_content_security(user_tag, openid=_wechat_openid_from_request(request))
        try:
            removed = color_card_store.remove_favorite(user_tag=user_tag, card_id=card_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"removed": removed}

    @app.post("/api/v1/color-card/favorites/picked")
    def api_add_picked_color_card_favorite(request: Request, payload: ColorCardPickedFavoriteRequest) -> Dict[str, Any]:
        _catalog_require_permission(request, "color:view")
        _check_text_content_security(payload.user_tag, payload.name, payload.hex, openid=_wechat_openid_from_request(request))
        try:
            card = color_card_store.add_picked_favorite(
                user_tag=payload.user_tag,
                name=payload.name,
                hex_value=payload.hex,
                l=payload.L,
                a=payload.a,
                b=payload.b,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"card": card}

    @app.post("/api/v1/color-card/native-meter/reading")
    def api_set_color_meter_native_reading(request: Request, payload: ColorMeterNativeReadingRequest) -> Dict[str, Any]:
        _catalog_require_permission(request, "color:view")
        user_tag = str(payload.user_tag or "").strip()
        if not user_tag:
            raise HTTPException(status_code=400, detail="missing user_tag")
        _check_text_content_security(user_tag, payload.device_id, payload.device_name, payload.event, openid=_wechat_openid_from_request(request))
        now_ms = int(time.time() * 1000)
        reading = {
            "user_tag": user_tag,
            "request_id": str(payload.request_id or ""),
            "event": str(payload.event or "measure"),
            "L": payload.L,
            "a": payload.a,
            "b": payload.b,
            "sample_count": max(1, int(payload.sample_count or 1)),
            "progress_count": max(0, int(payload.progress_count or 0)),
            "total_count": max(0, int(payload.total_count or 0)),
            "device_id": str(payload.device_id or ""),
            "device_name": str(payload.device_name or ""),
            "ts": float(payload.ts or now_ms),
            "server_ts": now_ms,
        }
        color_meter_native_readings[user_tag] = reading
        return {"ok": True, "reading": reading}

    @app.get("/api/v1/color-card/native-meter/reading")
    def api_get_color_meter_native_reading(
        request: Request,
        user_tag: str = "",
        request_id: str = "",
        since: float = 0,
        consume: bool = True,
    ) -> Dict[str, Any]:
        _catalog_require_permission(request, "color:view")
        clean_user_tag = str(user_tag or "").strip()
        if not clean_user_tag:
            raise HTTPException(status_code=400, detail="missing user_tag")
        _check_text_content_security(clean_user_tag, openid=_wechat_openid_from_request(request))
        reading = color_meter_native_readings.get(clean_user_tag)
        if reading and request_id and str(reading.get("request_id") or "") != str(request_id):
            return {"reading": None}
        if not reading or float(reading.get("server_ts") or 0) <= float(since or 0):
            return {"reading": None}
        if consume:
            color_meter_native_readings.pop(clean_user_tag, None)
        return {"reading": reading}

    @app.post("/api/v1/color-card/match")
    def api_match_color_cards(request: Request, payload: ColorCardMatchRequest) -> Dict[str, Any]:
        _catalog_require_permission(request, "color:view")
        limit = max(1, min(int(payload.limit or 12), 100))
        library_id = str(payload.library_id or "").strip()
        matches = color_card_store.match((float(payload.L), float(payload.a), float(payload.b)), library_id=library_id, limit=limit)
        return {
            "query_lab": {"L": payload.L, "a": payload.a, "b": payload.b},
            "library_id": library_id,
            "matches": matches,
        }

    @app.post("/api/v1/catalog/tags")
    def api_create_catalog_tag(request: Request, payload: CatalogTagCreateRequest) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        _check_text_content_security(payload.name, openid=_wechat_openid_from_request(request))
        name = payload.name
        if str(payload.type or "").strip():
            name = make_typed_tag(str(payload.type).strip(), payload.name) or payload.name
        try:
            tag = catalog_store.create_tag(name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"tag": tag}

    @app.delete("/api/v1/catalog/tags/{tag_name}")
    def api_delete_catalog_tag(request: Request, tag_name: str) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        try:
            tag = catalog_store.delete_tag(tag_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"tag": tag, "deleted": True}

    @app.post("/api/v1/catalog/sync")
    def api_sync_catalog(request: Request) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        with catalog_write_lock:
            return catalog_store.sync_from_standard_dir(standard_dir, image_exts)

    @app.post("/api/v1/catalog/imports/prepare")
    def api_prepare_catalog_import(request: Request, payload: CatalogImportPrepareRequest) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        source_dir_raw = str(payload.source_dir or "").strip() or catalog_import_source_dir
        if not source_dir_raw:
            raise HTTPException(status_code=400, detail="source_dir is empty; set catalog.import_source_dir in config or input it manually")
        source_dir = _resolve_catalog_import_source_dir(source_dir_raw)
        if not source_dir.exists() or not source_dir.is_dir():
            raise HTTPException(status_code=400, detail="source_dir not found")
        try:
            job = _create_catalog_import_job(source_dir, "server_dir")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _serialize_catalog_import_job(job)

    @app.post("/api/v1/catalog/imports/upload")
    async def api_upload_catalog_import(request: Request, files: List[UploadFile] = File(...)) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        upload_files = [item for item in files if item and str(item.filename or "").strip()]
        if not upload_files:
            raise HTTPException(status_code=400, detail="no files uploaded")
        if len(upload_files) > catalog_browser_upload_max_files:
            raise HTTPException(status_code=400, detail=f"浏览器上传最多一次 {catalog_browser_upload_max_files} 张图片")
        job_id = uuid.uuid4().hex
        source_dir = (catalog_upload_dir / job_id).resolve()
        source_dir.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        saved = 0
        skipped: List[str] = []
        for index, item in enumerate(upload_files, start=1):
            try:
                safe = _sanitize_upload_filename(item.filename or "", f"upload_{index}.jpg")
            except ValueError:
                skipped.append(str(item.filename or f"file-{index}"))
                continue
            stem = Path(safe).stem
            suffix = Path(safe).suffix.lower()
            candidate = safe
            seq = 1
            while candidate.lower() in used_names or (source_dir / candidate).exists():
                candidate = f"{stem}_{seq}{suffix}"
                seq += 1
            raw = await item.read()
            if not raw:
                skipped.append(str(item.filename or f"file-{index}"))
                continue
            try:
                _check_search_upload_content_security(raw, item.filename or candidate)
            except HTTPException:
                shutil.rmtree(source_dir, ignore_errors=True)
                raise
            (source_dir / candidate).write_bytes(raw)
            used_names.add(candidate.lower())
            saved += 1
        if saved <= 0:
            shutil.rmtree(source_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="uploaded files have no supported images")
        try:
            job = _create_catalog_import_job(source_dir, "browser_upload")
        except ValueError as exc:
            shutil.rmtree(source_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if skipped:
            job["message"] = f"任务已创建，已跳过 {len(skipped)} 个非图片文件"
        logging.info("catalog import upload created job=%s saved=%d skipped=%d", job.get("job_id"), saved, len(skipped))
        return _serialize_catalog_import_job(job)

    @app.post("/api/v1/catalog/imports/wechat-upload")
    def api_wechat_upload_catalog_import(request: Request, payload: CatalogImportWechatUploadRequest) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        upload_files = [item for item in payload.files if item and str(item.filename or "").strip() and str(item.data_base64 or "").strip()]
        if not upload_files:
            raise HTTPException(status_code=400, detail="no files uploaded")
        if len(upload_files) > 9:
            raise HTTPException(status_code=400, detail="最多一次上传 9 张图片")
        job_id = uuid.uuid4().hex
        source_dir = (catalog_upload_dir / job_id).resolve()
        source_dir.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        saved = 0
        skipped: List[str] = []
        for index, item in enumerate(upload_files, start=1):
            try:
                safe = _sanitize_upload_filename(item.filename or "", f"upload_{index}.jpg")
            except ValueError:
                skipped.append(str(item.filename or f"file-{index}"))
                continue
            stem = Path(safe).stem
            suffix = Path(safe).suffix.lower()
            candidate = safe
            seq = 1
            while candidate.lower() in used_names or (source_dir / candidate).exists():
                candidate = f"{stem}_{seq}{suffix}"
                seq += 1
            raw_text = str(item.data_base64 or "").strip()
            if "," in raw_text and raw_text.lower().startswith("data:"):
                raw_text = raw_text.split(",", 1)[1]
            try:
                raw = base64.b64decode(raw_text, validate=True)
            except Exception:
                skipped.append(str(item.filename or f"file-{index}"))
                continue
            if not raw:
                skipped.append(str(item.filename or f"file-{index}"))
                continue
            try:
                _check_search_upload_content_security(raw, item.filename or candidate)
            except HTTPException:
                shutil.rmtree(source_dir, ignore_errors=True)
                raise
            (source_dir / candidate).write_bytes(raw)
            used_names.add(candidate.lower())
            saved += 1
        if saved <= 0:
            shutil.rmtree(source_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="uploaded files have no supported images")
        try:
            job = _create_catalog_import_job(source_dir, "wechat_upload")
        except ValueError as exc:
            shutil.rmtree(source_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if skipped:
            job["message"] = f"任务已创建，已跳过 {len(skipped)} 个非图片文件"
        logging.info("catalog wechat upload created job=%s saved=%d skipped=%d", job.get("job_id"), saved, len(skipped))
        return _serialize_catalog_import_job(job)

    @app.post("/api/v1/wechat/session")
    def api_wechat_session(payload: WechatSessionRequest) -> Dict[str, Any]:
        return _wechat_jscode2session(payload.code)

    @app.get("/api/v1/catalog/imports/{job_id}")
    def api_get_catalog_import_job(request: Request, job_id: str) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        with catalog_import_lock:
            job = catalog_import_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="import job not found")
            return _serialize_catalog_import_job(job)

    @app.get("/api/v1/catalog/imports/{job_id}/source-image")
    def api_get_catalog_import_source_image(request: Request, job_id: str, source_rel_path: str, max_edge: int = 0, q: int = 82) -> FileResponse:
        _catalog_require_permission(request, "product:create")
        with catalog_import_lock:
            job = catalog_import_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="import job not found")
            source_dir = Path(str(job.get("source_dir", "")))
            item = _catalog_import_job_item(job, source_rel_path)
        if item is None:
            raise HTTPException(status_code=404, detail="source image not found")
        fp = (source_dir / source_rel_path).resolve()
        try:
            fp.relative_to(source_dir.resolve())
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid source_rel_path") from exc
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="source image not found")
        if max_edge > 0:
            edge = max(128, min(2048, int(max_edge)))
            out_fp = _ensure_preview(fp, edge, q)
            return FileResponse(
                path=str(out_fp),
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=604800, immutable"},
            )
        return FileResponse(path=str(fp), headers={"Cache-Control": "public, max-age=86400"})

    @app.post("/api/v1/catalog/imports/commit")
    def api_commit_catalog_import(request: Request, payload: CatalogImportCommitRequest) -> Dict[str, Any]:
        _catalog_require_permission(request, "product:create")
        with catalog_import_lock:
            job = catalog_import_jobs.get(payload.job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="import job not found")
            snapshot = _serialize_catalog_import_job(job)
        if snapshot["status"] != "completed":
            raise HTTPException(status_code=400, detail="import job is not completed")
        if snapshot["committed"]:
            raise HTTPException(status_code=400, detail="import job already committed")

        source_dir = Path(snapshot["source_dir"])
        prepared_items = {
            str(item.get("source_rel_path", "")): item
            for item in snapshot["items"]
            if str(item.get("source_rel_path", "")).strip()
        }
        selected_items = [item for item in payload.items if item.selected]
        if not selected_items:
            raise HTTPException(status_code=400, detail="no selected items")

        with catalog_write_lock:
            with catalog_import_lock:
                latest_job = catalog_import_jobs.get(payload.job_id)
                if latest_job is not None and bool(latest_job.get("committed")):
                    raise HTTPException(status_code=400, detail="import job already committed")

            planned: List[tuple[Path, str]] = []
            seen_targets: set[str] = set()
            style_year_tags: Dict[str, set[str]] = {}
            style_extra_tags: Dict[str, set[str]] = {}
            def _next_available_import_target(raw_name: str, suffix: str) -> str:
                try:
                    first = _sanitize_import_filename(raw_name, suffix)
                except ValueError:
                    raise
                first_key = first.lower()
                if first_key not in seen_targets and not (standard_dir / first).exists():
                    return first
                first_path = Path(first)
                stem = re.sub(r"_\d+$", "", first_path.stem).strip() or first_path.stem
                ext = first_path.suffix or suffix
                seq = 0
                while seq < 10000:
                    candidate = f"{stem}_{seq:03d}{ext}"
                    key = candidate.lower()
                    if key not in seen_targets and not (standard_dir / candidate).exists():
                        return candidate
                    seq += 1
                raise ValueError(f"cannot allocate target filename for {raw_name}")

            for item in selected_items:
                prepared = prepared_items.get(item.source_rel_path)
                if prepared is None:
                    raise HTTPException(status_code=400, detail=f"unknown source_rel_path: {item.source_rel_path}")
                src = source_dir / item.source_rel_path
                if not src.exists() or not src.is_file():
                    raise HTTPException(status_code=400, detail=f"source file not found: {item.source_rel_path}")
                fallback_name = str(prepared.get("proposed_filename", "")).strip() or src.name
                raw_target = item.target_filename.strip() or fallback_name
                try:
                    target_name = _next_available_import_target(raw_target, src.suffix.lower())
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=f"{item.source_rel_path}: {exc}") from exc
                seen_targets.add(target_name.lower())
                try:
                    year_tag = _sanitize_year_tag(item.year_tag.strip() or str(prepared.get("proposed_year_tag", "")).strip())
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=f"{item.source_rel_path}: {exc}") from exc
                style_code = filename_to_style_code(target_name).strip()
                if year_tag and style_code:
                    typed_year_tag = make_typed_tag("year", year_tag)
                    if typed_year_tag:
                        style_year_tags.setdefault(style_code, set()).add(typed_year_tag)
                import_tags = _normalize_import_tags(item.tags)
                if import_tags and style_code:
                    style_extra_tags.setdefault(style_code, set()).update(import_tags)
                planned.append((src, target_name))

            imported = 0
            for src, target_name in planned:
                shutil.copy2(src, standard_dir / target_name)
                imported += 1

            sync_result = catalog_store.sync_from_standard_dir(standard_dir, image_exts)
            for style_code, year_tags in style_year_tags.items():
                if year_tags:
                    catalog_store.add_product_tags(style_code, sorted(year_tags))
            for style_code, import_tags in style_extra_tags.items():
                if import_tags:
                    catalog_store.add_product_tags(style_code, sorted(import_tags))
            with catalog_import_lock:
                job = catalog_import_jobs.get(payload.job_id)
                if job is not None:
                    job["committed"] = True
                    job["message"] = f"已导入 {imported} 张图片"
            threading.Thread(
                target=_reload_search_assets,
                args=("catalog_import_commit",),
                daemon=True,
            ).start()
        return {
            "job_id": payload.job_id,
            "imported": imported,
            "sync": sync_result,
        }

    @app.get("/images/{image_name}")
    def get_standard_image(image_name: str, max_edge: int = 0, q: int = 82) -> FileResponse:
        safe = Path(image_name).name
        fp = standard_dir / safe
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        if max_edge > 0:
            edge = max(128, min(2048, int(max_edge)))
            out_fp = _ensure_preview(fp, edge, q)
            return FileResponse(
                path=str(out_fp),
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=604800, immutable"},
            )
        return FileResponse(path=str(fp), headers={"Cache-Control": "public, max-age=86400"})

    def _refresh_image_url_response(request: Request, image_name: str, kind: str = "") -> Dict[str, Any]:
        safe = Path(image_name).name
        fp = standard_dir / safe
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        base_url = _external_base_url(request)
        if str(kind).strip().lower() == "catalog":
            image_url = _build_catalog_image_url(base_url, safe)
            exp_ts = 0
        else:
            image_url, exp_ts = _build_image_url_with_exp(base_url, safe)
        return {"image_name": safe, "image_url": image_url, "expires_at": exp_ts}

    @app.get("/image-url", response_model=ImageUrlResponse)
    def refresh_image_url(request: Request, image_name: str, kind: str = "") -> Dict[str, Any]:
        return _refresh_image_url_response(request, image_name, kind)

    @app.get("/api/v1/image-url", response_model=ImageUrlResponse)
    def api_refresh_image_url(request: Request, image_name: str, kind: str = "") -> Dict[str, Any]:
        return _refresh_image_url_response(request, image_name, kind)

    @app.post("/search", response_model=SearchResponse)
    async def search(
        request: Request,
        file: UploadFile = File(...),
        include_image_base64: bool = False,
        base64_topn: int = 0,
        crop_x: float = Form(0.0),
        crop_y: float = Form(0.0),
        crop_w: float = Form(0.0),
        crop_h: float = Form(0.0),
        match_mode: str = Form(""),
        result_top_k: int = Form(0),
    ) -> Dict[str, Any]:
        t_all = time.perf_counter()
        top_k = max(1, min(int(result_top_k or int(search_cfg.get("top_k", 5))), 60))
        if not file.filename:
            raise HTTPException(status_code=400, detail="missing file name")
        suffix = Path(file.filename).suffix.lower()
        if suffix.lstrip(".") not in {"png", "jpg", "jpeg"}:
            raise HTTPException(status_code=400, detail="only png/jpg/jpeg supported")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tf, tempfile.NamedTemporaryFile(suffix=".tight.jpg", delete=True) as tight_tf:
            upload_bytes = await file.read()
            if not upload_bytes:
                raise HTTPException(status_code=400, detail="empty file")
            _check_search_upload_content_security(upload_bytes, file.filename)
            tf.write(upload_bytes)
            tf.flush()
            query_path = Path(tf.name)
            if query_max_edge > 0:
                try:
                    with Image.open(query_path) as qim0:
                        qim = qim0.convert("RGB")
                        w, h = qim.size
                        mx = max(w, h)
                        limit = max(320, int(query_max_edge))
                        if mx > limit:
                            scale = float(limit) / float(mx)
                            nw = max(1, int(round(w * scale)))
                            nh = max(1, int(round(h * scale)))
                            q2 = qim.resize((nw, nh), Image.Resampling.BILINEAR)
                            q2.save(query_path, format="JPEG", quality=90)
                except Exception:
                    pass
            crop_debug = ""
            crop_active = False
            crop_orig_area = 0.0
            crop_final_area = 0.0
            crop_expand_ratio = 1.0
            strict_small_region_crop = False
            crop_norm_x = 0.0
            crop_norm_y = 0.0
            crop_norm_w = 0.0
            crop_norm_h = 0.0
            tight_query_path = Path(tight_tf.name)
            tight_query_available = False
            if crop_w > 0.02 and crop_h > 0.02:
                try:
                    with Image.open(query_path) as crop_im0:
                        crop_im = crop_im0.convert("RGB")
                        iw, ih = crop_im.size
                        x = max(0.0, min(0.98, float(crop_x)))
                        y = max(0.0, min(0.98, float(crop_y)))
                        cw = max(0.02, min(1.0 - x, float(crop_w)))
                        ch = max(0.02, min(1.0 - y, float(crop_h)))
                        crop_orig_area = max(0.0, float(cw) * float(ch))
                        orig_left = int(round(x * iw))
                        orig_top = int(round(y * ih))
                        orig_right = int(round((x + cw) * iw))
                        orig_bottom = int(round((y + ch) * ih))
                        if orig_right - orig_left >= 24 and orig_bottom - orig_top >= 24:
                            crop_im.crop((orig_left, orig_top, orig_right, orig_bottom)).save(
                                tight_query_path,
                                format="JPEG",
                                quality=94,
                            )
                            tight_query_available = True
                        expanded = False
                        if region_crop_auto_expand_enabled:
                            min_w = max(0.02, min(1.0, region_crop_auto_expand_min_w))
                            min_h = max(0.02, min(1.0, region_crop_auto_expand_min_h))
                            min_area = max(0.0, min(1.0, region_crop_auto_expand_min_area))
                            x, y, cw, ch, expanded = _expand_region_crop(
                                x,
                                y,
                                cw,
                                ch,
                                min_w=min_w,
                                min_h=min_h,
                                min_area=min_area,
                                context_pad_ratio=max(0.0, min(1.0, region_crop_context_pad_ratio)),
                                context_min_area=max(0.0, min(1.0, region_crop_context_min_area)),
                                wide_strip_aspect_threshold=max(1.0, float(region_crop_wide_strip_aspect_threshold)),
                                wide_strip_max_h=max(0.02, min(1.0, float(region_crop_wide_strip_max_h))),
                            )
                        crop_final_area = max(0.0, float(cw) * float(ch))
                        crop_expand_ratio = crop_final_area / max(1e-6, crop_orig_area)
                        strict_small_region_crop = (
                            region_crop_strict_small_enabled
                            and expanded
                            and crop_orig_area > 0.0
                            and crop_orig_area <= max(0.0, min(1.0, region_crop_strict_small_max_orig_area))
                            and crop_expand_ratio >= max(1.0, float(region_crop_strict_small_min_expand_ratio))
                        )
                        left = int(round(x * iw))
                        top = int(round(y * ih))
                        right = int(round((x + cw) * iw))
                        bottom = int(round((y + ch) * ih))
                        if right - left >= 32 and bottom - top >= 32:
                            crop_im.crop((left, top, right, bottom)).save(query_path, format="JPEG", quality=92)
                            crop_debug = f"{x:.3f},{y:.3f},{cw:.3f},{ch:.3f}"
                            if expanded:
                                crop_debug += ":expanded"
                            if strict_small_region_crop:
                                crop_debug += ":strict-small"
                            crop_active = True
                            crop_norm_x = float(x)
                            crop_norm_y = float(y)
                            crop_norm_w = float(cw)
                            crop_norm_h = float(ch)
                except Exception:
                    crop_debug = ""
            query_width = 0
            query_height = 0
            try:
                with Image.open(query_path) as qim1:
                    query_width, query_height = qim1.size
            except Exception:
                pass
            active_match_mode = (str(match_mode).strip().lower() or default_match_mode or "similar_style")
            if active_match_mode not in {"similar_style", "exact"}:
                active_match_mode = "similar_style"
            search_scope = "region_primary" if crop_active and region_primary_when_crop else "full_context"
            search_strategy = f"{active_match_mode}:{search_scope}"
            auto_region_probe_views: List[Image.Image] = []
            auto_region_probe_active = False
            if (
                full_context_region_probe_enabled
                and not crop_active
                and active_match_mode == "similar_style"
                and query_width > 0
                and query_height >= max(1, int(full_context_region_probe_min_height))
                and (float(query_width) / float(query_height)) <= max(0.1, float(full_context_region_probe_max_aspect))
            ):
                try:
                    with Image.open(query_path) as probe_im0:
                        probe_im = probe_im0.convert("RGB")
                        pw, ph = probe_im.size
                        probe_boxes = [
                            (0.04, 0.16, 0.78, 0.70),
                            (0.00, 0.22, 0.78, 0.82),
                            (0.18, 0.12, 0.76, 0.68),
                        ]
                        for bx, by, bw, bh in probe_boxes:
                            left = int(round(max(0.0, min(0.98, bx)) * pw))
                            top = int(round(max(0.0, min(0.98, by)) * ph))
                            right = int(round(min(1.0, bx + bw) * pw))
                            bottom = int(round(min(1.0, by + bh) * ph))
                            if right - left >= 48 and bottom - top >= 48:
                                auto_region_probe_views.append(probe_im.crop((left, top, right, bottom)))
                        auto_region_probe_active = bool(auto_region_probe_views)
                except Exception:
                    auto_region_probe_views = []
                    auto_region_probe_active = False
            region_probe_active = bool(crop_active or auto_region_probe_active)
            debug_saved = _save_debug_query_image(request, query_path, file.filename or "query")
            logging.info(
                "search upload user=%s file=%s bytes=%d final_size=%sx%s crop=%s strategy=%s saved=%s",
                getattr(request.state, "api_user", "unknown"),
                file.filename,
                len(upload_bytes),
                query_width,
                query_height,
                crop_debug,
                search_strategy,
                str(debug_saved or ""),
            )

            with search_assets_lock:
                req_names = names
                req_feats = feats
                req_secondary_names = secondary_names
                req_secondary_feats = secondary_feats
                req_region_names = region_names
                req_region_feats = region_feats
                req_rerank_candidate_cache = rerank_candidate_cache
                req_label_memory_refs = label_memory_refs
                req_scene_text_index = scene_text_index
                req_standard_image_by_code_key = standard_image_by_code_key

            query_hint_code = try_extract_query_style_code(query_path) if ocr_hint_enabled else ""
            scene_text_tokens: List[str] = []
            scene_text_small_region_allowed = False
            checker_debug = ""
            checker_candidates_debug = ""
            accent_debug = ""
            accent_candidates_debug = ""
            accent_small_region_allowed = False
            sleeve_debug = ""
            sleeve_candidates_debug = ""
            accessory_debug = ""
            accessory_candidates_debug = ""
            region_debug = ""
            region_strong_code = ""
            region_best_score = 0.0
            region_has_confident_match = False
            region_code_scores: Dict[str, float] = {}
            region_code_best_images: Dict[str, str] = {}
            collar_candidate_scores: Dict[str, float] = {}
            region_repeat_force_scores: Dict[str, tuple[float, int]] = {}
            region_boost_debug = ""
            region_rescue_debug = ""
            region_order_debug = ""
            q_pattern_sig: np.ndarray | None = None
            q_dark_motif_sig: np.ndarray | None = None
            base_code_prior_boost = (
                build_label_memory_prior_from_refs(
                    query_path,
                    req_label_memory_refs,
                    sim_threshold=label_memory_sim_threshold,
                    max_boost=label_memory_max_boost,
                )
                if label_memory_enabled
                else {}
            )
            code_prior_boost = dict(base_code_prior_boost)

            def _code_prior_key(code: str) -> str:
                return _style_code_key(code)

            def _rows_from_ranked(
                ranked_in: List[tuple[str, float]],
                topn: int = top_k,
                score_floor: float = min_score,
            ) -> List[Dict[str, Any]]:
                return topk_style_codes(
                    ranked_in,
                    topn,
                    min_score=score_floor,
                    code_agg_top_n=code_agg_top_n,
                    code_agg_alpha=code_agg_alpha,
                    query_hint_code=query_hint_code,
                    query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                    code_prior_boost=code_prior_boost,
                    display_score_scale=display_score_scale,
                    display_score_bias=display_score_bias,
                )

            def _apply_region_code_prior() -> None:
                nonlocal region_boost_debug
                if not (region_probe_active and region_crop_code_prior_enabled and region_code_scores):
                    return
                boosted_codes = []
                min_region_code_prior_score = (
                    strict_small_region_code_prior_min_score if strict_small_region_crop else region_crop_code_prior_min_score
                )
                for code, score in sorted(region_code_scores.items(), key=lambda item: item[1], reverse=True)[: max(1, region_crop_code_prior_topn)]:
                    if float(score) < min_region_code_prior_score:
                        continue
                    code_key = _code_prior_key(code)
                    boost = float(region_crop_code_prior_boost)
                    if active_match_mode == "exact":
                        boost *= max(0.0, float(exact_region_code_prior_scale))
                    code_prior_boost[code_key] = max(
                        float(code_prior_boost.get(code_key, 0.0)),
                        boost,
                    )
                    boosted_codes.append(f"{code}:{float(score):.3f}/{boost:.3f}")
                if boosted_codes:
                    region_boost_debug = ",".join(boosted_codes)

            def _large_weak_region_rescue_mode() -> bool:
                return bool(
                    crop_active
                    and not strict_small_region_crop
                    and crop_final_area >= max(0.0, min(1.0, float(region_crop_large_force_top_area)))
                    and region_best_score < float(region_crop_force_top_min_score)
                )

            def _large_region_rescue_order_mode() -> bool:
                return bool(
                    crop_active
                    and not strict_small_region_crop
                    and crop_final_area >= max(0.0, min(1.0, float(region_crop_large_force_top_area)))
                    and region_best_score < float(region_crop_large_result_rescue_order_max_best)
                )

            def _dominant_region_repeat_key() -> str:
                if not region_repeat_force_scores:
                    return ""
                key, (score, hits) = max(
                    region_repeat_force_scores.items(),
                    key=lambda item: (int(item[1][1]), float(item[1][0])),
                )
                if int(hits) < max(1, int(region_crop_dominant_repeat_min_hits)):
                    return ""
                if float(score) < float(region_crop_dominant_repeat_min_score):
                    return ""
                return _code_prior_key(key)

            def _sleeve_rescue_candidate_allowed(sim: float, pair_prior: float) -> bool:
                if float(sim) < float(region_crop_sleeve_rescue_min_sim):
                    return False
                if float(pair_prior) >= float(region_crop_sleeve_rescue_min_pair_prior):
                    return True
                return bool(
                    float(sim) >= float(region_crop_sleeve_rescue_strong_sim)
                    and float(pair_prior) >= float(region_crop_sleeve_rescue_strong_min_pair_prior)
                )

            def _strong_sleeve_rescue_candidate(sim: float, pair_prior: float) -> bool:
                return bool(
                    float(sim) >= float(region_crop_sleeve_rescue_strong_sim)
                    and float(pair_prior) >= float(region_crop_sleeve_rescue_strong_min_pair_prior)
                )

            def _rescue_region_rows(rows_in: List[Dict[str, Any]], ranked_in: List[tuple[str, float]]) -> List[Dict[str, Any]]:
                nonlocal region_rescue_debug
                if not (
                    region_probe_active
                    and region_crop_result_rescue_enabled
                    and (active_match_mode != "exact" or exact_region_rescue_enabled)
                    and region_code_scores
                    and rows_in
                    and ranked_in
                ):
                    return rows_in
                min_region_rescue_score = (
                    strict_small_region_result_rescue_min_score if strict_small_region_crop else region_crop_result_rescue_min_score
                )
                rescue_topn_local = max(1, region_crop_result_rescue_topn)
                large_weak_region_rescue = _large_weak_region_rescue_mode()
                large_region_rescue_order = _large_region_rescue_order_mode()
                if large_weak_region_rescue:
                    min_region_rescue_score = min(
                        float(min_region_rescue_score),
                        max(
                            float(region_crop_large_result_rescue_min_score),
                            float(region_best_score) - max(0.0, float(region_crop_large_result_rescue_top_delta)),
                        ),
                    )
                    rescue_topn_local = max(rescue_topn_local, int(region_crop_large_result_rescue_topn))
                rescue_codes = [
                    code
                    for code, score in sorted(region_code_scores.items(), key=lambda item: item[1], reverse=True)[:rescue_topn_local]
                    if float(score) >= min_region_rescue_score
                ]
                if not rescue_codes:
                    return rows_in
                existing_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in rows_in}
                missing_keys = [_code_prior_key(code) for code in rescue_codes if _code_prior_key(code) not in existing_keys]
                if not missing_keys:
                    return rows_in
                scan_n = max(top_k, min(max(1, region_crop_result_rescue_scan_codes), max(len(ranked_in), top_k)))
                broad_rows = _rows_from_ranked(ranked_in, topn=scan_n, score_floor=0.0)
                broad_by_key = {
                    _code_prior_key(str(row.get("style_code", ""))): row
                    for row in broad_rows
                }
                best_ranked_by_key: Dict[str, tuple[str, float]] = {}
                for img_name, score in ranked_in:
                    code = filename_to_style_code(img_name)
                    key = _code_prior_key(code)
                    current = best_ranked_by_key.get(key)
                    if current is None or float(score) > float(current[1]):
                        best_ranked_by_key[key] = (img_name.split("@", 1)[0], float(score))

                def _fallback_rescue_row(key: str) -> Dict[str, Any] | None:
                    ranked_item = best_ranked_by_key.get(key)
                    if ranked_item is None and key in region_code_best_images:
                        ranked_item = (
                            region_code_best_images[key],
                            float(region_code_scores.get(key, 0.0)),
                        )
                    if ranked_item is None:
                        return None
                    image_name, ranked_score = ranked_item
                    style_code = filename_to_style_code(image_name)
                    region_score = float(region_code_scores.get(style_code, ranked_score))
                    raw_score = max(
                        ranked_score + float(code_prior_boost.get(key, 0.0)),
                        region_score + float(region_crop_code_prior_boost),
                    )
                    z = float(display_score_scale) * (float(raw_score) - float(display_score_bias))
                    disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                    disp = min(0.9999, max(0.0, float(disp)))
                    return {
                        "style_code": style_code,
                        "best_standard_image": image_name,
                        "score": round(disp, 4),
                        "rank_score": round(float(raw_score), 6),
                    }

                rescue_rows: List[Dict[str, Any]] = []
                for key in missing_keys:
                    row = broad_by_key.get(key)
                    if row is not None:
                        rescue_row = dict(row)
                        if large_region_rescue_order:
                            rescue_row["_region_rescue_keep"] = True
                        rescue_rows.append(rescue_row)
                        continue
                    fallback_row = _fallback_rescue_row(key)
                    if fallback_row is not None:
                        if large_region_rescue_order:
                            fallback_row["_region_rescue_keep"] = True
                        rescue_rows.append(fallback_row)
                if not rescue_rows:
                    return rows_in
                region_rescue_debug = ",".join(
                    f"{row.get('style_code', '')}:{float(row.get('rank_score', 0.0)):.3f}"
                    for row in rescue_rows
                )
                rescue_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in rescue_rows}
                kept_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in rescue_keys
                ]
                target_n = max(top_k, len(rows_in))
                keep_n = max(0, target_n - len(rescue_rows))
                return (kept_rows[:keep_n] + rescue_rows)[:target_n]

            def _force_top_region_rows(rows_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                nonlocal region_rescue_debug
                if not (
                    region_probe_active
                    and region_crop_force_top_enabled
                    and active_match_mode == "similar_style"
                    and (search_scope == "region_primary" or auto_region_probe_active)
                    and region_code_scores
                    and region_code_best_images
                    and rows_in
                ):
                    return rows_in
                min_region_force_top_score = (
                    strict_small_region_force_top_min_score if strict_small_region_crop else region_crop_force_top_min_score
                )
                force_topn_local = (
                    strict_small_region_force_topn if strict_small_region_crop else region_crop_force_topn
                )
                if (
                    auto_region_probe_active
                    or (
                        not strict_small_region_crop
                        and crop_final_area >= max(0.0, min(1.0, float(region_crop_large_force_top_area)))
                    )
                ):
                    min_region_force_top_score = min(
                        float(min_region_force_top_score),
                        float(region_crop_large_force_top_min_score),
                    )
                    force_topn_local = max(int(force_topn_local), int(region_crop_large_force_topn))
                forced_rows: List[Dict[str, Any]] = []
                dominant_repeat_key = _dominant_region_repeat_key()
                repeat_force_allowed = bool(dominant_repeat_key) or not _large_weak_region_rescue_mode()
                if region_crop_repeat_force_enabled and repeat_force_allowed and region_repeat_force_scores:
                    for code, (repeat_score, hit_count) in sorted(
                        region_repeat_force_scores.items(),
                        key=lambda item: (item[1][1], item[1][0]),
                        reverse=True,
                    ):
                        if int(hit_count) < max(1, int(region_crop_repeat_force_min_hits)):
                            continue
                        if float(repeat_score) < float(region_crop_repeat_force_min_score):
                            continue
                        key = _code_prior_key(code)
                        image_name = region_code_best_images.get(key)
                        if not image_name:
                            continue
                        raw_score = max(
                            float(region_crop_repeat_force_seed_score),
                            float(repeat_score) + float(region_crop_code_prior_boost),
                        )
                        z = float(display_score_scale) * (float(raw_score) - float(display_score_bias))
                        disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                        disp = min(0.9999, max(0.0, float(disp)))
                        forced_rows.append(
                            {
                                "style_code": filename_to_style_code(image_name),
                                "best_standard_image": image_name,
                                "score": round(disp, 4),
                                "rank_score": round(float(raw_score), 6),
                                "_force_keep": True,
                            }
                        )
                for code, score in sorted(region_code_scores.items(), key=lambda item: item[1], reverse=True):
                    if len(forced_rows) >= max(1, force_topn_local):
                        break
                    if float(score) < min_region_force_top_score:
                        break
                    key = _code_prior_key(code)
                    image_name = region_code_best_images.get(key)
                    if not image_name:
                        continue
                    raw_score = max(
                        float(score) + float(region_crop_code_prior_boost),
                        float(score),
                    )
                    z = float(display_score_scale) * (float(raw_score) - float(display_score_bias))
                    disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                    disp = min(0.9999, max(0.0, float(disp)))
                    forced_rows.append(
                        {
                            "style_code": filename_to_style_code(image_name),
                            "best_standard_image": image_name,
                            "score": round(disp, 4),
                            "rank_score": round(float(raw_score), 6),
                        }
                    )
                if not forced_rows:
                    return rows_in
                force_debug = ",".join(
                    f"{row.get('style_code', '')}:{float(row.get('rank_score', 0.0)):.3f}"
                    for row in forced_rows
                )
                region_rescue_debug = (
                    f"{region_rescue_debug}|force_region={force_debug}"
                    if region_rescue_debug
                    else f"force_region={force_debug}"
                )
                forced_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in forced_rows}
                kept_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in forced_keys
                ]
                target_n = max(top_k, len(rows_in))
                diversity_rows = [row for row in kept_rows if row.get("_sleeve_diversity_keep")]
                if diversity_rows:
                    diversity_keys = {
                        _code_prior_key(str(row.get("style_code", "")))
                        for row in diversity_rows
                    }
                    kept_rows = [
                        row
                        for row in kept_rows
                        if _code_prior_key(str(row.get("style_code", ""))) not in diversity_keys
                    ]
                    return (forced_rows + diversity_rows + kept_rows)[:target_n]
                return (forced_rows + kept_rows)[:target_n]

            def _order_region_primary_rows(rows_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                nonlocal region_order_debug, region_rescue_debug
                if not (
                    region_probe_active
                    and region_crop_order_by_region_enabled
                    and active_match_mode == "similar_style"
                    and (search_scope == "region_primary" or auto_region_probe_active)
                    and region_code_scores
                    and rows_in
                ):
                    return rows_in
                accessory_hat_query = (
                    accessory_near_square_region
                    and crop_aspect <= accessory_hat_override_max_aspect
                    and q_accessory_hat_prior >= accessory_region_hat_prior_threshold
                    and not region_has_confident_match
                    and bool(accessory_candidates_debug)
                )
                if accessory_hat_query:
                    region_order_debug = "skip-accessory-hat"
                    return rows_in

                def _row_pattern_region_boost(row: Dict[str, Any]) -> tuple[float, float]:
                    if not (
                        strict_small_region_crop
                        and strict_small_pattern_region_order_enabled
                        and q_pattern_sig is not None
                    ):
                        return 0.0, -1.0
                    code_key = _code_prior_key(str(row.get("style_code", "")))
                    candidates = [
                        str(row.get("best_standard_image", "")).split("@", 1)[0],
                        str(region_code_best_images.get(code_key, "")).split("@", 1)[0],
                    ]
                    best_sim = -1.0
                    for image_name in candidates:
                        if not image_name:
                            continue
                        cs = pattern_sig_cache.get(image_name)
                        if cs is None:
                            continue
                        best_sim = max(best_sim, float(q_pattern_sig @ cs))
                    min_sim = float(strict_small_pattern_region_order_min_sim)
                    if best_sim < min_sim:
                        return 0.0, best_sim
                    boost = max(0.0, float(strict_small_pattern_region_order_weight)) * max(0.0, best_sim)
                    return float(boost), best_sim

                def _row_region_score(row: Dict[str, Any]) -> float:
                    code_key = _code_prior_key(str(row.get("style_code", "")))
                    region_score = max(
                        float(region_code_scores.get(str(row.get("style_code", "")), -1.0)),
                        float(region_code_scores.get(code_key, -1.0)),
                    )
                    if row.get("_region_rescue_keep"):
                        region_score = max(
                            region_score,
                            float(row.get("rank_score", -1.0)) - max(0.0, float(region_crop_code_prior_boost)),
                        )
                    if row.get("_dark_motif_sim") is not None:
                        region_score = max(region_score, float(row.get("rank_score", -1.0)))
                    pattern_boost, _pattern_sim = _row_pattern_region_boost(row)
                    return float(region_score) + float(pattern_boost)

                rows_for_order = list(rows_in)
                if collar_candidate_scores:
                    existing_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in rows_for_order}
                    sorted_existing = sorted(rows_for_order, key=_row_region_score, reverse=True)
                    cutoff_score = (
                        _row_region_score(sorted_existing[min(len(sorted_existing), top_k) - 1])
                        if sorted_existing
                        else -1.0
                    )
                    for code_key, sim in sorted(collar_candidate_scores.items(), key=lambda item: item[1], reverse=True):
                        if code_key in existing_keys:
                            continue
                        region_score = max(
                            float(region_code_scores.get(code_key, -1.0)),
                            float(collar_contour_region_score_base)
                            + float(collar_contour_region_score_scale)
                            * max(0.0, min(float(collar_contour_region_score_max), float(sim))),
                        )
                        if region_score < cutoff_score - float(collar_contour_near_tie_margin):
                            continue
                        image_name = region_code_best_images.get(code_key)
                        if not image_name:
                            continue
                        raw_score = (
                            float(collar_contour_seed_score_base)
                            + float(collar_contour_boost_scale) * max(0.0, float(sim))
                            + float(code_prior_boost.get(code_key, 0.0))
                        )
                        z = float(display_score_scale) * (float(raw_score) - float(display_score_bias))
                        disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                        disp = min(0.9999, max(0.0, float(disp)))
                        rows_for_order.append(
                            {
                                "style_code": code_key,
                                "best_standard_image": image_name,
                                "score": round(disp, 4),
                                "rank_score": round(float(raw_score), 6),
                            }
                        )
                        existing_keys.add(code_key)
                protected_source_rows = [row for row in rows_for_order if row.get("_force_keep")]
                diversity_rows = [row for row in protected_source_rows if row.get("_sleeve_diversity_keep")]
                if diversity_rows:
                    diversity_keys = {
                        _code_prior_key(str(row.get("style_code", "")))
                        for row in diversity_rows
                    }
                    regular_protected_rows = [
                        row
                        for row in protected_source_rows
                        if _code_prior_key(str(row.get("style_code", ""))) not in diversity_keys
                    ]
                    protected_rows = regular_protected_rows[: max(0, top_k - len(diversity_rows))] + diversity_rows
                else:
                    protected_rows = protected_source_rows
                protected_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in protected_rows}
                sortable_rows = [
                    row
                    for row in rows_for_order
                    if _code_prior_key(str(row.get("style_code", ""))) not in protected_keys
                ]
                ordered = sorted(
                    sortable_rows,
                    key=lambda row: (_row_region_score(row), float(row.get("rank_score", 0.0))),
                    reverse=True,
                )
                ordered_all = protected_rows + ordered
                if (
                    collar_contour_near_tie_diversify_enabled
                    and collar_candidate_scores
                    and len(ordered_all) > top_k
                ):
                    base_top = ordered_all[:top_k]
                    cutoff_score = min(_row_region_score(row) for row in base_top) if base_top else -1.0
                    near_tie_window = [
                        row
                        for row in ordered_all
                        if _code_prior_key(str(row.get("style_code", ""))) in collar_candidate_scores
                        and _row_region_score(row) >= cutoff_score - float(collar_contour_near_tie_margin)
                    ]
                    near_tie_window = near_tie_window[: max(top_k, top_k + int(collar_contour_near_tie_window_extra))]
                    if len(near_tie_window) >= max(top_k + 1, int(collar_contour_near_tie_min_window)):
                        selected_rows: List[Dict[str, Any]] = []
                        selected_keys: set[str] = set()
                        slots = max(1, top_k)
                        for pos in range(slots):
                            idx = 0 if slots == 1 else round(pos * (len(near_tie_window) - 1) / (slots - 1))
                            row = near_tie_window[int(idx)]
                            key = _code_prior_key(str(row.get("style_code", "")))
                            if key in selected_keys:
                                continue
                            selected_keys.add(key)
                            selected_rows.append(row)
                        for row in ordered_all:
                            if len(selected_rows) >= top_k:
                                break
                            key = _code_prior_key(str(row.get("style_code", "")))
                            if key in selected_keys:
                                continue
                            selected_keys.add(key)
                            selected_rows.append(row)
                        ordered_all = selected_rows
                ordered = ordered_all[:top_k]
                pattern_order_debug = []
                if strict_small_region_crop and q_pattern_sig is not None:
                    for row in ordered[: min(top_k, 12)]:
                        boost, sim = _row_pattern_region_boost(row)
                        if boost > 0:
                            pattern_order_debug.append(
                                f"{row.get('style_code', '')}:{sim:.3f}/{boost:.3f}"
                            )
                if pattern_order_debug:
                    region_rescue_debug = (
                        f"{region_rescue_debug}|pattern_order={','.join(pattern_order_debug)}"
                        if region_rescue_debug
                        else f"pattern_order={','.join(pattern_order_debug)}"
                    )
                region_order_debug = ",".join(
                    f"{row.get('style_code', '')}:{_row_region_score(row):.3f}"
                    for row in ordered[:top_k]
                )
                return ordered[:top_k]

            def _rescue_pattern_region_rows(rows_in: List[Dict[str, Any]], ranked_in: List[tuple[str, float]]) -> List[Dict[str, Any]]:
                nonlocal region_rescue_debug
                if not (
                    strict_small_region_crop
                    and strict_small_pattern_region_rescue_enabled
                    and active_match_mode == "similar_style"
                    and search_scope == "region_primary"
                    and q_pattern_sig is not None
                    and ranked_in
                    and rows_in
                ):
                    return rows_in
                existing_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in rows_in}
                ordered_region_scores = sorted(
                    [float(v) for v in region_code_scores.values() if float(v) >= 0.0],
                    reverse=True,
                )
                cutoff_idx = min(len(ordered_region_scores), max(1, top_k)) - 1
                cutoff_score = ordered_region_scores[cutoff_idx] if ordered_region_scores else 0.0
                min_region_score = max(0.0, cutoff_score - max(0.0, float(strict_small_pattern_region_rescue_near_delta)))
                min_sim = float(strict_small_pattern_region_rescue_min_sim)
                scan_n = max(top_k, min(len(ranked_in), max(1, int(strict_small_pattern_region_rescue_scan_images))))
                best_by_key: Dict[str, tuple[float, float, float, str]] = {}
                for image_name, rank_score in ranked_in[:scan_n]:
                    file_name = image_name.split("@", 1)[0]
                    cs = pattern_sig_cache.get(file_name)
                    if cs is None:
                        continue
                    sim = float(q_pattern_sig @ cs)
                    if sim < min_sim:
                        continue
                    code = filename_to_style_code(file_name)
                    key = _code_prior_key(code)
                    if key in existing_keys:
                        continue
                    region_score = max(
                        float(region_code_scores.get(code, -1.0)),
                        float(region_code_scores.get(key, -1.0)),
                        float(rank_score) - max(0.0, float(region_crop_code_prior_boost)),
                    )
                    if region_score < min_region_score:
                        continue
                    combined = float(region_score) + max(0.0, float(strict_small_pattern_region_order_weight)) * max(0.0, sim)
                    current = best_by_key.get(key)
                    if current is None or combined > current[0]:
                        best_by_key[key] = (combined, sim, region_score, file_name)
                if not best_by_key:
                    if pattern_sig_cache:
                        region_rescue_debug = (
                            f"{region_rescue_debug}|pattern_rescue=none:{min_region_score:.3f}/{min_sim:.3f}"
                            if region_rescue_debug
                            else f"pattern_rescue=none:{min_region_score:.3f}/{min_sim:.3f}"
                        )
                    return rows_in
                rescue_rows: List[Dict[str, Any]] = []
                for key, (combined, sim, region_score, file_name) in sorted(best_by_key.items(), key=lambda item: item[1][0], reverse=True)[
                    : max(1, int(strict_small_pattern_region_rescue_max_rows))
                ]:
                    z = float(display_score_scale) * (float(combined) - float(display_score_bias))
                    disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                    disp = min(0.9999, max(0.0, float(disp)))
                    rescue_rows.append(
                        {
                            "style_code": filename_to_style_code(file_name),
                            "best_standard_image": file_name,
                            "score": round(disp, 4),
                            "rank_score": round(float(combined), 6),
                            "_region_rescue_keep": True,
                            "_pattern_rescue_sim": round(float(sim), 6),
                            "_pattern_rescue_region_score": round(float(region_score), 6),
                        }
                    )
                if not rescue_rows:
                    return rows_in
                rescue_debug = ",".join(
                    f"{row.get('style_code', '')}:{float(row.get('_pattern_rescue_region_score', 0.0)):.3f}/{float(row.get('_pattern_rescue_sim', 0.0)):.3f}"
                    for row in rescue_rows
                )
                region_rescue_debug = (
                    f"{region_rescue_debug}|pattern_rescue={rescue_debug}"
                    if region_rescue_debug
                    else f"pattern_rescue={rescue_debug}"
                )
                rescue_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in rescue_rows}
                kept_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in rescue_keys
                ]
                target_n = max(top_k, len(rows_in))
                return (rescue_rows + kept_rows)[:target_n]

            def _rescue_dark_motif_region_rows(rows_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                nonlocal region_rescue_debug
                if not (
                    strict_small_region_crop
                    and strict_small_dark_motif_rescue_enabled
                    and active_match_mode == "similar_style"
                    and search_scope == "region_primary"
                    and q_dark_motif_sig is not None
                    and dark_motif_cache
                    and rows_in
                ):
                    return rows_in
                existing_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in rows_in}
                ordered_region_scores = sorted(
                    [float(v) for v in region_code_scores.values() if float(v) >= 0.0],
                    reverse=True,
                )
                cutoff_idx = min(len(ordered_region_scores), max(1, top_k)) - 1
                cutoff_score = ordered_region_scores[cutoff_idx] if ordered_region_scores else 0.0
                min_region_score = max(0.0, cutoff_score - max(0.0, float(strict_small_pattern_region_rescue_near_delta)))
                min_sim = float(strict_small_dark_motif_rescue_min_sim)
                full_presence_min_sim = min(float(min_sim), float(strict_small_dark_motif_full_presence_min_sim))
                full_presence_boost = max(0.0, float(strict_small_dark_motif_full_presence_boost))
                def _dark_motif_presence_count(sig: np.ndarray | None) -> int:
                    if sig is None or sig.size < 5:
                        return 0
                    return int(np.count_nonzero(np.asarray(sig[-5:], dtype=np.float32) > 1e-4))

                query_presence_count = _dark_motif_presence_count(q_dark_motif_sig)
                min_presence_count = max(0, min(5, query_presence_count - 1)) if query_presence_count >= 4 else 0
                best_by_key: Dict[str, tuple[float, float, float, float, str]] = {}
                debug_by_key: Dict[str, tuple[float, float, int, float]] = {}
                for file_name, sig in dark_motif_cache.items():
                    presence_count = _dark_motif_presence_count(sig)
                    sim = float(q_dark_motif_sig @ sig)
                    full_presence_match = query_presence_count >= 4 and presence_count >= 4
                    effective_sim = sim + full_presence_boost if full_presence_match else sim
                    code = filename_to_style_code(file_name)
                    key = _code_prior_key(code)
                    region_score_raw = max(
                        float(region_code_scores.get(code, -1.0)),
                        float(region_code_scores.get(key, -1.0)),
                    )
                    prev_debug = debug_by_key.get(key)
                    if prev_debug is None or effective_sim > prev_debug[1]:
                        debug_by_key[key] = (sim, effective_sim, presence_count, region_score_raw)
                    if min_presence_count and presence_count < min_presence_count:
                        continue
                    candidate_min_sim = full_presence_min_sim if full_presence_match else min_sim
                    if sim < candidate_min_sim:
                        continue
                    if key in existing_keys:
                        continue
                    region_score = region_score_raw
                    if region_score < 0.0 and sim >= candidate_min_sim:
                        region_score = min_region_score
                    if region_score < min_region_score:
                        continue
                    combined = float(region_score) + max(0.0, float(strict_small_dark_motif_rescue_weight)) * max(0.0, effective_sim)
                    current = best_by_key.get(key)
                    if current is None or combined > current[0]:
                        best_by_key[key] = (combined, sim, effective_sim, region_score, file_name)
                if not best_by_key:
                    pool_debug = ",".join(
                        f"{key}:{sim:.3f}>{eff:.3f}/p{presence}/{region_score:.3f}"
                        for key, (sim, eff, presence, region_score) in sorted(debug_by_key.items(), key=lambda item: item[1][1], reverse=True)[:12]
                    )
                    region_rescue_debug = (
                        f"{region_rescue_debug}|dark_motif=none:{min_region_score:.3f}/{min_sim:.3f}/{full_presence_min_sim:.3f}/p{min_presence_count}|dark_pool={pool_debug}"
                        if region_rescue_debug
                        else f"dark_motif=none:{min_region_score:.3f}/{min_sim:.3f}/{full_presence_min_sim:.3f}/p{min_presence_count}|dark_pool={pool_debug}"
                    )
                    return rows_in
                rescue_rows: List[Dict[str, Any]] = []
                for _key, (combined, sim, effective_sim, region_score, file_name) in sorted(best_by_key.items(), key=lambda item: item[1][2], reverse=True)[
                    : max(1, int(strict_small_dark_motif_rescue_max_rows))
                ]:
                    z = float(display_score_scale) * (float(combined) - float(display_score_bias))
                    disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                    disp = min(0.9999, max(0.0, float(disp)))
                    rescue_rows.append(
                        {
                            "style_code": filename_to_style_code(file_name),
                            "best_standard_image": file_name,
                            "score": round(disp, 4),
                            "rank_score": round(float(combined), 6),
                            "_force_keep": True,
                            "_region_rescue_keep": True,
                            "_dark_motif_sim": round(float(sim), 6),
                            "_dark_motif_effective_sim": round(float(effective_sim), 6),
                            "_dark_motif_region_score": round(float(region_score), 6),
                        }
                    )
                rescue_debug = ",".join(
                    f"{row.get('style_code', '')}:{float(row.get('_dark_motif_region_score', 0.0)):.3f}/{float(row.get('_dark_motif_sim', 0.0)):.3f}>{float(row.get('_dark_motif_effective_sim', 0.0)):.3f}"
                    for row in rescue_rows
                )
                pool_debug = ",".join(
                    f"{key}:{sim:.3f}>{eff:.3f}/p{presence}/{region_score:.3f}"
                    for key, (sim, eff, presence, region_score) in sorted(debug_by_key.items(), key=lambda item: item[1][1], reverse=True)[:16]
                )
                region_rescue_debug = (
                    f"{region_rescue_debug}|dark_motif={rescue_debug}|dark_pool={pool_debug}"
                    if region_rescue_debug
                    else f"dark_motif={rescue_debug}|dark_pool={pool_debug}"
                )
                rescue_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in rescue_rows}
                kept_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in rescue_keys
                ]
                target_n = max(top_k, len(rows_in))
                return (rescue_rows + kept_rows)[:target_n]

            def _merge_rescue_rows_preserving_forced(
                rescue_rows: List[Dict[str, Any]], rows_in: List[Dict[str, Any]]
            ) -> List[Dict[str, Any]]:
                """Keep deliberate rescue rows from being truncated by later generic rescues."""
                rescue_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in rescue_rows}
                forced_rows = [
                    row
                    for row in rows_in
                    if row.get("_force_keep")
                    and _code_prior_key(str(row.get("style_code", ""))) not in rescue_keys
                ]
                forced_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in forced_rows}
                kept_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in rescue_keys
                    and _code_prior_key(str(row.get("style_code", ""))) not in forced_keys
                ]
                return (forced_rows + rescue_rows + kept_rows)[:top_k]

            def _apply_sleeve_region_rescue() -> None:
                if not (
                    crop_active
                    and region_crop_sleeve_rescue_enabled
                    and (not strict_small_region_crop or sleeve_small_region_rescue_allowed)
                    and active_match_mode == "similar_style"
                    and search_scope == "region_primary"
                    and sleeve_candidates_debug
                ):
                    return
                for part in sleeve_candidates_debug.split(","):
                    fields = part.split(":")
                    if len(fields) < 2:
                        continue
                    code = fields[0].strip()
                    nums = fields[1].split("/")
                    if len(nums) < 3 or not code:
                        continue
                    try:
                        sim = float(nums[0])
                        pair_prior = float(nums[2])
                    except ValueError:
                        continue
                    if not _sleeve_rescue_candidate_allowed(sim, pair_prior):
                        continue
                    dominant_repeat_key = _dominant_region_repeat_key()
                    if (
                        dominant_repeat_key
                        and _code_prior_key(code) != dominant_repeat_key
                        and not _strong_sleeve_rescue_candidate(sim, pair_prior)
                    ):
                        continue
                    current = float(region_code_scores.get(code, -1.0))
                    sleeve_score = sim + max(0.0, region_crop_sleeve_rescue_weight) * pair_prior
                    if sleeve_score > current:
                        region_code_scores[code] = sleeve_score

            def _rescue_sleeve_region_rows(rows_in: List[Dict[str, Any]], ranked_in: List[tuple[str, float]]) -> List[Dict[str, Any]]:
                nonlocal region_rescue_debug
                if not (
                    crop_active
                    and region_crop_sleeve_rescue_enabled
                    and (not strict_small_region_crop or sleeve_small_region_rescue_allowed)
                    and active_match_mode == "similar_style"
                    and search_scope == "region_primary"
                    and sleeve_candidates_debug
                    and rows_in
                    and ranked_in
                ):
                    return rows_in
                best_ranked_by_key: Dict[str, tuple[str, float]] = {}
                for img_name, score in ranked_in:
                    code = filename_to_style_code(img_name)
                    key = _code_prior_key(code)
                    current = best_ranked_by_key.get(key)
                    if current is None or float(score) > float(current[1]):
                        best_ranked_by_key[key] = (img_name.split("@", 1)[0], float(score))
                sleeve_rows: List[Dict[str, Any]] = []
                seen_keys = set()
                for part in sleeve_candidates_debug.split(","):
                    fields = part.split(":")
                    if len(fields) < 2:
                        continue
                    code = fields[0].strip()
                    key = _code_prior_key(code)
                    if not key or key in seen_keys:
                        continue
                    nums = fields[1].split("/")
                    if len(nums) < 3:
                        continue
                    try:
                        sim = float(nums[0])
                        seed_score = float(nums[1])
                        pair_prior = float(nums[2])
                    except ValueError:
                        continue
                    if not _sleeve_rescue_candidate_allowed(sim, pair_prior):
                        continue
                    dominant_repeat_key = _dominant_region_repeat_key()
                    if (
                        dominant_repeat_key
                        and key != dominant_repeat_key
                        and not _strong_sleeve_rescue_candidate(sim, pair_prior)
                    ):
                        continue
                    ranked_item = best_ranked_by_key.get(key)
                    if ranked_item is None:
                        fallback_image = region_code_best_images.get(key) or req_standard_image_by_code_key.get(key, "")
                        if not fallback_image:
                            continue
                        ranked_item = (fallback_image, 0.0)
                    image_name, ranked_score = ranked_item
                    raw_score = max(float(seed_score), float(ranked_score))
                    z = float(display_score_scale) * (raw_score - float(display_score_bias))
                    disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                    disp = min(0.9999, max(0.0, float(disp)))
                    sleeve_score = sim + max(0.0, region_crop_sleeve_rescue_weight) * pair_prior
                    region_code_scores[code] = max(float(region_code_scores.get(code, -1.0)), sleeve_score)
                    sleeve_rows.append(
                        {
                            "style_code": filename_to_style_code(image_name),
                            "best_standard_image": image_name,
                            "score": round(disp, 4),
                            "rank_score": round(raw_score, 6),
                            "_force_keep": True,
                            "_sleeve_sim": round(float(sim), 6),
                            "_sleeve_pair_prior": round(float(pair_prior), 6),
                        }
                    )
                    seen_keys.add(key)
                if not sleeve_rows:
                    return rows_in
                sleeve_rows.sort(key=lambda row: float(row.get("rank_score", 0.0)), reverse=True)
                if strict_small_region_crop and len(sleeve_rows) > top_k:
                    sleeve_keys_all = {
                        _code_prior_key(str(row.get("style_code", "")))
                        for row in sleeve_rows
                    }
                    forced_visible_count = sum(
                        1
                        for row in rows_in
                        if row.get("_force_keep")
                        and _code_prior_key(str(row.get("style_code", ""))) not in sleeve_keys_all
                    )
                    visible_sleeve_slots = max(1, top_k - forced_visible_count)
                    low_pair_limit = float(region_crop_sleeve_rescue_strong_min_pair_prior) + 0.05
                    min_diverse_rank = max(
                        float(display_score_bias),
                        float(region_crop_sleeve_rescue_strong_sim)
                        + max(0.0, float(region_crop_sleeve_rescue_weight))
                        * float(region_crop_sleeve_rescue_strong_min_pair_prior),
                    )
                    diverse_index = next(
                        (
                            idx
                            for idx, row in enumerate(sleeve_rows[visible_sleeve_slots:], start=visible_sleeve_slots)
                            if float(row.get("_sleeve_sim", 0.0)) >= float(region_crop_sleeve_rescue_strong_sim)
                            and float(row.get("_sleeve_pair_prior", 0.0)) <= low_pair_limit
                            and float(row.get("rank_score", 0.0)) >= min_diverse_rank
                        ),
                        None,
                    )
                    if diverse_index is not None:
                        diverse_row = sleeve_rows.pop(diverse_index)
                        diverse_row["_sleeve_diversity_keep"] = True
                        diverse_row["_region_rescue_keep"] = True
                        insert_at = max(0, visible_sleeve_slots - 1)
                        sleeve_rows.insert(insert_at, diverse_row)
                sleeve_debug_rescue = ",".join(
                    f"{row.get('style_code', '')}:{float(row.get('rank_score', 0.0)):.3f}"
                    for row in sleeve_rows[:top_k]
                )
                region_rescue_debug = (
                    f"{region_rescue_debug}|sleeve={sleeve_debug_rescue}"
                    if region_rescue_debug
                    else f"sleeve={sleeve_debug_rescue}"
                )
                sleeve_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in sleeve_rows}
                kept_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in sleeve_keys
                ]
                return _merge_rescue_rows_preserving_forced(sleeve_rows, kept_rows)

            def _rescue_hat_from_sleeve_region_rows(rows_in: List[Dict[str, Any]], ranked_in: List[tuple[str, float]]) -> List[Dict[str, Any]]:
                nonlocal region_rescue_debug
                if not (
                    crop_active
                    and accessory_hat_from_sleeve_region_rescue_enabled
                    and not strict_small_region_crop
                    and active_match_mode == "similar_style"
                    and search_scope == "region_primary"
                    and sleeve_candidates_debug
                    and rows_in
                    and accessory_hat_code_prefixes
                    and q_accessory_hat_prior >= accessory_region_hat_prior_threshold
                ):
                    return rows_in
                existing_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in rows_in}
                best_ranked_by_key: Dict[str, tuple[str, float]] = {}
                for img_name, score in ranked_in:
                    code = filename_to_style_code(img_name)
                    key = _code_prior_key(code)
                    current = best_ranked_by_key.get(key)
                    if current is None or float(score) > float(current[1]):
                        best_ranked_by_key[key] = (img_name.split("@", 1)[0], float(score))
                best_hat_by_key: Dict[str, tuple[str, float]] = {}
                for file_name, prior in accessory_hat_prior_cache.items():
                    base_name = file_name.split("@", 1)[0]
                    code = filename_to_style_code(base_name)
                    key = _code_prior_key(code)
                    if not key:
                        continue
                    current = best_hat_by_key.get(key)
                    if current is None or float(prior) > float(current[1]):
                        best_hat_by_key[key] = (base_name, float(prior))
                rescue_rows: List[Dict[str, Any]] = []
                seen_keys = set()
                for part in sleeve_candidates_debug.split(","):
                    fields = part.split(":")
                    if len(fields) < 2:
                        continue
                    code = fields[0].strip()
                    if not any(code.upper().startswith(prefix) for prefix in accessory_hat_code_prefixes):
                        continue
                    key = _code_prior_key(code)
                    if not key or key in seen_keys or key in existing_keys:
                        continue
                    nums = fields[1].split("/")
                    if len(nums) < 2:
                        continue
                    try:
                        seed_score = float(nums[1])
                    except ValueError:
                        continue
                    if seed_score < accessory_hat_from_sleeve_region_rescue_min_seed:
                        continue
                    ranked_item = best_ranked_by_key.get(key)
                    image_name = ranked_item[0] if ranked_item is not None else ""
                    ranked_score = float(ranked_item[1]) if ranked_item is not None else 0.0
                    if not image_name:
                        hat_item = best_hat_by_key.get(key)
                        if hat_item is None:
                            continue
                        image_name = hat_item[0]
                    raw_score = max(float(seed_score), float(ranked_score))
                    z = float(display_score_scale) * (raw_score - float(display_score_bias))
                    disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                    disp = min(0.9999, max(0.0, float(disp)))
                    rescue_rows.append(
                        {
                            "style_code": filename_to_style_code(image_name),
                            "best_standard_image": image_name,
                            "score": round(disp, 4),
                            "rank_score": round(raw_score, 6),
                            "_force_keep": True,
                        }
                    )
                    seen_keys.add(key)
                    if len(rescue_rows) >= max(1, accessory_hat_from_sleeve_region_rescue_max_rows):
                        break
                if not rescue_rows:
                    return rows_in
                hat_sleeve_debug = ",".join(
                    f"{row.get('style_code', '')}:{float(row.get('rank_score', 0.0)):.3f}"
                    for row in rescue_rows
                )
                region_rescue_debug = (
                    f"{region_rescue_debug}|hat_sleeve={hat_sleeve_debug}"
                    if region_rescue_debug
                    else f"hat_sleeve={hat_sleeve_debug}"
                )
                rescue_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in rescue_rows}
                kept_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in rescue_keys
                ]
                return _merge_rescue_rows_preserving_forced(rescue_rows, kept_rows)

            def _rescue_checker_region_rows(rows_in: List[Dict[str, Any]], ranked_in: List[tuple[str, float]]) -> List[Dict[str, Any]]:
                nonlocal region_rescue_debug
                if not (
                    crop_active
                    and checker_region_rescue_enabled
                    and active_match_mode == "similar_style"
                    and search_scope == "region_primary"
                    and checker_candidates_debug
                    and "seed=" in checker_candidates_debug
                    and rows_in
                    and ranked_in
                ):
                    return rows_in
                best_ranked_by_key: Dict[str, tuple[str, float]] = {}
                for img_name, score in ranked_in:
                    code = filename_to_style_code(img_name)
                    key = _code_prior_key(code)
                    current = best_ranked_by_key.get(key)
                    if current is None or float(score) > float(current[1]):
                        best_ranked_by_key[key] = (img_name.split("@", 1)[0], float(score))
                seed_part = checker_candidates_debug.split("seed=", 1)[1].split("|", 1)[0]
                checker_rows: List[Dict[str, Any]] = []
                seen_keys = set()
                for part in seed_part.split(","):
                    fields = part.split(":")
                    if len(fields) < 2:
                        continue
                    code = fields[0].strip()
                    key = _code_prior_key(code)
                    if not key or key in seen_keys:
                        continue
                    nums = fields[1].split("/")
                    if len(nums) < 2:
                        continue
                    try:
                        seed_score = float(nums[1])
                    except ValueError:
                        continue
                    if seed_score < checker_region_rescue_min_seed:
                        continue
                    ranked_item = best_ranked_by_key.get(key)
                    if ranked_item is None:
                        continue
                    image_name, ranked_score = ranked_item
                    raw_score = max(float(seed_score), float(ranked_score))
                    z = float(display_score_scale) * (raw_score - float(display_score_bias))
                    disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                    disp = min(0.9999, max(0.0, float(disp)))
                    checker_rows.append(
                        {
                            "style_code": filename_to_style_code(image_name),
                            "best_standard_image": image_name,
                            "score": round(disp, 4),
                            "rank_score": round(raw_score, 6),
                        }
                    )
                    seen_keys.add(key)
                    if len(checker_rows) >= max(1, checker_region_rescue_max_rows):
                        break
                if not checker_rows:
                    return rows_in
                checker_debug_rescue = ",".join(
                    f"{row.get('style_code', '')}:{float(row.get('rank_score', 0.0)):.3f}"
                    for row in checker_rows
                )
                region_rescue_debug = (
                    f"{region_rescue_debug}|checker={checker_debug_rescue}"
                    if region_rescue_debug
                    else f"checker={checker_debug_rescue}"
                )
                checker_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in checker_rows}
                kept_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in checker_keys
                ]
                return _merge_rescue_rows_preserving_forced(checker_rows, kept_rows)

            def _rescue_hat_region_rows(rows_in: List[Dict[str, Any]], ranked_in: List[tuple[str, float]]) -> List[Dict[str, Any]]:
                nonlocal region_rescue_debug
                if not (
                    crop_active
                    and accessory_hat_region_rescue_enabled
                    and active_match_mode == "similar_style"
                    and search_scope == "region_primary"
                    and accessory_candidates_debug
                    and "hat:" in accessory_candidates_debug
                    and accessory_hat_region_rescue_min_aspect <= float(crop_aspect or 0.0) <= accessory_hat_region_rescue_max_aspect
                    and rows_in
                    and ranked_in
                ):
                    return rows_in
                best_ranked_by_key: Dict[str, tuple[str, float]] = {}
                for img_name, score in ranked_in:
                    code = filename_to_style_code(img_name)
                    key = _code_prior_key(code)
                    current = best_ranked_by_key.get(key)
                    if current is None or float(score) > float(current[1]):
                        best_ranked_by_key[key] = (img_name.split("@", 1)[0], float(score))
                hat_rows: List[Dict[str, Any]] = []
                seen_keys = set()
                for part in accessory_candidates_debug.split(","):
                    fields = part.split(":")
                    if len(fields) < 3 or fields[0].strip() != "hat":
                        continue
                    code = fields[1].strip()
                    if accessory_hat_code_prefixes and not any(code.upper().startswith(prefix) for prefix in accessory_hat_code_prefixes):
                        continue
                    key = _code_prior_key(code)
                    if not key or key in seen_keys:
                        continue
                    nums = fields[2].split("/")
                    if len(nums) < 2:
                        continue
                    try:
                        seed_score = float(nums[1])
                    except ValueError:
                        continue
                    if seed_score < accessory_hat_region_rescue_min_seed:
                        continue
                    ranked_item = best_ranked_by_key.get(key)
                    if ranked_item is None:
                        continue
                    image_name, ranked_score = ranked_item
                    raw_score = max(float(seed_score), float(ranked_score))
                    z = float(display_score_scale) * (raw_score - float(display_score_bias))
                    disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                    disp = min(0.9999, max(0.0, float(disp)))
                    hat_rows.append(
                        {
                            "style_code": filename_to_style_code(image_name),
                            "best_standard_image": image_name,
                            "score": round(disp, 4),
                            "rank_score": round(raw_score, 6),
                        }
                    )
                    seen_keys.add(key)
                    if len(hat_rows) >= max(1, accessory_hat_region_rescue_max_rows):
                        break
                if not hat_rows:
                    return rows_in
                hat_debug_rescue = ",".join(
                    f"{row.get('style_code', '')}:{float(row.get('rank_score', 0.0)):.3f}"
                    for row in hat_rows
                )
                region_rescue_debug = (
                    f"{region_rescue_debug}|hat={hat_debug_rescue}"
                    if region_rescue_debug
                    else f"hat={hat_debug_rescue}"
                )
                hat_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in hat_rows}
                kept_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in hat_keys
                ]
                return _merge_rescue_rows_preserving_forced(hat_rows, kept_rows)

            def _rescue_hat_family_region_rows(rows_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                nonlocal region_rescue_debug
                if not (
                    crop_active
                    and accessory_hat_family_region_rescue_enabled
                    and not use_strip_mode
                    and not strict_small_region_crop
                    and active_match_mode == "similar_style"
                    and search_scope == "region_primary"
                    and rows_in
                    and accessory_hat_prior_cache
                    and q_accessory_hat_prior >= accessory_region_hat_prior_threshold
                    and accessory_hat_family_region_rescue_min_aspect
                    <= float(crop_aspect or 0.0)
                    <= accessory_hat_family_region_rescue_max_aspect
                    and not accessory_candidates_debug
                    and not accent_candidates_debug
                    and not sleeve_candidates_debug
                    and not checker_candidates_debug
                    and str(accessory_debug).startswith("skip-region:")
                ):
                    return rows_in
                existing_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in rows_in}
                family_candidates: List[tuple[str, str, float]] = []
                for file_name, prior in accessory_hat_prior_cache.items():
                    base_name = file_name.split("@", 1)[0]
                    code = filename_to_style_code(base_name)
                    key = _code_prior_key(code)
                    if not key or key in existing_keys:
                        continue
                    if accessory_hat_code_prefixes and not any(code.upper().startswith(prefix) for prefix in accessory_hat_code_prefixes):
                        continue
                    prior_score = float(prior)
                    if prior_score < accessory_hat_family_region_rescue_min_prior:
                        continue
                    code_boost = float(accessory_hat_code_boost) if (
                        accessory_hat_code_boost > 0.0
                        and accessory_hat_code_prefixes
                        and any(code.upper().startswith(prefix) for prefix in accessory_hat_code_prefixes)
                    ) else 0.0
                    family_candidates.append((key, base_name, prior_score + code_boost))
                if not family_candidates:
                    return rows_in
                best_by_key: Dict[str, tuple[str, float]] = {}
                for key, image_name, score in family_candidates:
                    current = best_by_key.get(key)
                    if current is None or float(score) > float(current[1]):
                        best_by_key[key] = (image_name, float(score))
                selected = sorted(best_by_key.items(), key=lambda item: item[1][1], reverse=True)[
                    : max(1, accessory_hat_family_region_rescue_max_rows)
                ]
                rescue_rows: List[Dict[str, Any]] = []
                for _key, (image_name, prior_score) in selected:
                    raw_score = max(
                        float(accessory_hat_family_region_rescue_score),
                        float(accessory_hat_prior_seed_score_base)
                        + float(accessory_hat_prior_seed_boost_scale) * max(0.0, float(prior_score)),
                    )
                    z = float(display_score_scale) * (float(raw_score) - float(display_score_bias))
                    disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                    disp = min(0.9999, max(0.0, float(disp)))
                    rescue_rows.append(
                        {
                            "style_code": filename_to_style_code(image_name),
                            "best_standard_image": image_name,
                            "score": round(disp, 4),
                            "rank_score": round(float(raw_score), 6),
                        }
                    )
                if not rescue_rows:
                    return rows_in
                family_debug = ",".join(
                    f"{row.get('style_code', '')}:{float(row.get('rank_score', 0.0)):.3f}"
                    for row in rescue_rows
                )
                region_rescue_debug = (
                    f"{region_rescue_debug}|hat_family={family_debug}"
                    if region_rescue_debug
                    else f"hat_family={family_debug}"
                )
                rescue_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in rescue_rows}
                kept_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in rescue_keys
                ]
                return _merge_rescue_rows_preserving_forced(rescue_rows, kept_rows)

            def _rescue_accent_region_rows(rows_in: List[Dict[str, Any]], ranked_in: List[tuple[str, float]]) -> List[Dict[str, Any]]:
                nonlocal region_rescue_debug
                if not (
                    crop_active
                    and accent_region_rescue_enabled
                    and ((not strict_small_region_crop and not partial_region_crop) or accent_small_region_allowed)
                    and active_match_mode == "similar_style"
                    and search_scope == "region_primary"
                    and accent_candidates_debug
                    and rows_in
                    and ranked_in
                ):
                    return rows_in
                best_ranked_by_key: Dict[str, tuple[str, float]] = {}
                for img_name, score in ranked_in:
                    code = filename_to_style_code(img_name)
                    key = _code_prior_key(code)
                    current = best_ranked_by_key.get(key)
                    if current is None or float(score) > float(current[1]):
                        best_ranked_by_key[key] = (img_name.split("@", 1)[0], float(score))
                accent_rows: List[Dict[str, Any]] = []
                seen_keys = set()
                for part in accent_candidates_debug.split(","):
                    fields = part.split(":")
                    if len(fields) < 2:
                        continue
                    code = fields[0].strip()
                    key = _code_prior_key(code)
                    if not key or key in seen_keys:
                        continue
                    nums = fields[1].split("/")
                    if len(nums) < 2:
                        continue
                    try:
                        sim = float(nums[0])
                        seed_score = float(nums[1])
                    except ValueError:
                        continue
                    if sim < accent_region_rescue_min_sim:
                        continue
                    ranked_item = best_ranked_by_key.get(key)
                    if ranked_item is None:
                        continue
                    image_name, ranked_score = ranked_item
                    raw_score = max(float(seed_score), float(ranked_score))
                    z = float(display_score_scale) * (raw_score - float(display_score_bias))
                    disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                    disp = min(0.9999, max(0.0, float(disp)))
                    accent_rows.append(
                        {
                            "style_code": filename_to_style_code(image_name),
                            "best_standard_image": image_name,
                            "score": round(disp, 4),
                            "rank_score": round(raw_score, 6),
                        }
                    )
                    seen_keys.add(key)
                    if len(accent_rows) >= max(1, accent_region_rescue_max_rows):
                        break
                if not accent_rows:
                    return rows_in
                accent_debug_rescue = ",".join(
                    f"{row.get('style_code', '')}:{float(row.get('rank_score', 0.0)):.3f}"
                    for row in accent_rows
                )
                region_rescue_debug = (
                    f"{region_rescue_debug}|accent={accent_debug_rescue}"
                    if region_rescue_debug
                    else f"accent={accent_debug_rescue}"
                )
                accent_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in accent_rows}
                kept_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in accent_keys
                ]
                return _merge_rescue_rows_preserving_forced(accent_rows, kept_rows)

            def _rescue_scene_text_region_rows(rows_in: List[Dict[str, Any]], ranked_in: List[tuple[str, float]]) -> List[Dict[str, Any]]:
                nonlocal region_rescue_debug
                if not (
                    crop_active
                    and scene_text_region_rescue_enabled
                    and (not strict_small_region_crop or scene_text_small_region_allowed)
                    and active_match_mode == "similar_style"
                    and search_scope == "region_primary"
                    and scene_text_tokens
                    and isinstance(scene_text_index, dict)
                    and rows_in
                ):
                    return rows_in
                if region_best_score >= float(scene_text_suppress_when_region_min_score) and not scene_text_small_region_allowed:
                    return rows_in
                image_tokens = dict(scene_text_index.get("image_tokens", {}))
                token_idf = dict(scene_text_index.get("token_idf", {}))
                if not image_tokens:
                    return rows_in
                ranked_by_image = {name.split("@", 1)[0]: float(score) for name, score in ranked_in}
                best_by_code: Dict[str, Dict[str, Any]] = {}
                min_ratio = max(0.0, min(1.0, scene_text_region_rescue_min_ratio))
                for image_name, toks_raw in image_tokens.items():
                    toks = [str(tok).upper() for tok in toks_raw if str(tok).strip()]
                    if not toks:
                        continue
                    code = filename_to_style_code(image_name)
                    if not code:
                        continue
                    text_score = 0.0
                    hit_count = 0
                    for query_tok_raw in scene_text_tokens:
                        query_tok = str(query_tok_raw).upper().strip()
                        if len(query_tok) < scene_text_min_token_len:
                            continue
                        best_ratio = 0.0
                        best_tok = ""
                        for tok in toks:
                            if len(tok) < scene_text_min_token_len:
                                continue
                            if query_tok == tok:
                                ratio = 1.0
                            elif len(query_tok) >= 6 and len(tok) >= 6 and (query_tok in tok or tok in query_tok):
                                ratio = min(1.0, min(len(query_tok), len(tok)) / max(len(query_tok), len(tok)))
                            else:
                                ratio = difflib.SequenceMatcher(None, query_tok, tok).ratio()
                            if ratio > best_ratio:
                                best_ratio = float(ratio)
                                best_tok = tok
                        if best_ratio < min_ratio:
                            continue
                        idf = float(token_idf.get(best_tok, 1.0))
                        text_score += best_ratio * max(1.0, idf)
                        hit_count += 1
                    if hit_count <= 0 or text_score < scene_text_region_rescue_min_score:
                        continue
                    key = _code_prior_key(code)
                    current = best_by_code.get(key)
                    if current is None or text_score > float(current.get("text_score", 0.0)):
                        best_by_code[key] = {
                            "code": code,
                            "image_name": image_name,
                            "text_score": text_score,
                            "hit_count": hit_count,
                        }
                if not best_by_code:
                    return rows_in
                candidates = sorted(
                    best_by_code.values(),
                    key=lambda item: (float(item.get("text_score", 0.0)), int(item.get("hit_count", 0))),
                    reverse=True,
                )[: max(1, scene_text_region_rescue_max_rows)]
                if not candidates:
                    return rows_in
                max_text_score = max(float(item.get("text_score", 0.0)) for item in candidates) + 1e-6
                text_rows: List[Dict[str, Any]] = []
                for item in candidates:
                    image_name = str(item["image_name"])
                    raw_score = max(
                        float(ranked_by_image.get(image_name, -1e9)),
                        scene_text_seed_score_base
                        + scene_text_boost_scale * (float(item.get("text_score", 0.0)) / max_text_score)
                        + 0.03 * min(3, max(0, int(item.get("hit_count", 1)) - 1)),
                    )
                    z = float(display_score_scale) * (raw_score - float(display_score_bias))
                    disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                    disp = min(0.9999, max(0.0, float(disp)))
                    text_rows.append(
                        {
                            "style_code": str(item["code"]),
                            "best_standard_image": image_name,
                            "score": round(disp, 4),
                            "rank_score": round(raw_score, 6),
                        }
                    )
                if not text_rows:
                    return rows_in
                text_debug = ",".join(
                    f"{row.get('style_code', '')}:{float(row.get('rank_score', 0.0)):.3f}"
                    for row in text_rows
                )
                region_rescue_debug = (
                    f"{region_rescue_debug}|text={text_debug}"
                    if region_rescue_debug
                    else f"text={text_debug}"
                )
                text_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in text_rows}
                kept_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in text_keys
                ]
                return _merge_rescue_rows_preserving_forced(text_rows, kept_rows)

            def _make_display_scores_follow_order(rows_in: List[Dict[str, Any]]) -> None:
                """Keep UI percentages consistent with the final ranked order."""
                prev_score: float | None = None
                for row in rows_in:
                    raw_score = float(row.get("score", 0.0))
                    row.setdefault("score_raw", round(raw_score, 4))
                    display_score = min(0.9999, max(0.0, raw_score))
                    if prev_score is not None and display_score >= prev_score:
                        display_score = max(0.0, prev_score - 0.0001)
                        row["score_adjusted"] = True
                    row["score"] = round(display_score, 4)
                    prev_score = display_score

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
                pass_region_query_multicrop: bool,
                pass_region_query_crop_ratio: float,
                pass_region_query_component_views: bool,
                pass_region_query_view_consensus_weight: float,
            ) -> tuple[List[tuple[str, float]], List[Dict[str, Any]], float, float, float]:
                nonlocal region_debug, region_strong_code, region_best_score, region_has_confident_match

                def _search_topk_images_from_views(
                    query_views: List[Image.Image],
                    names_local: List[str],
                    feats_local: np.ndarray,
                    top_k_local: int,
                    backend_local: str,
                    w_clip_local: float,
                    w_shape_local: float,
                    w_color_local: float,
                    w_stripe_local: float,
                    query_view_consensus_weight_local: float = 0.0,
                ) -> List[tuple[str, float]]:
                    if not query_views or len(names_local) != len(feats_local):
                        return []
                    q_feats = [
                        extract_embedding(
                            view,
                            backend_local,
                            w_clip_local,
                            w_shape_local,
                            w_color_local,
                            w_stripe_local,
                        )
                        for view in query_views
                    ]
                    if not q_feats:
                        return []
                    sims_stack = np.vstack([(feats_local @ q).astype(np.float32) for q in q_feats])
                    consensus = max(0.0, min(1.0, float(query_view_consensus_weight_local)))
                    if consensus <= 1e-6:
                        sims = sims_stack.max(axis=0)
                    else:
                        sims_max = sims_stack.max(axis=0)
                        sims_mean = sims_stack.mean(axis=0)
                        sims = (1.0 - consensus) * sims_max + consensus * sims_mean
                    order = np.argsort(-sims)[: max(1, int(top_k_local))]
                    return [(names_local[int(i)], float(sims[int(i)])) for i in order]

                t0 = time.perf_counter()
                eff_w_clip = pass_w_clip
                eff_w_shape = pass_w_shape
                eff_w_color = pass_w_color
                eff_w_stripe = pass_w_stripe
                if strict_small_region_crop:
                    eff_w_clip = strict_small_w_clip
                    eff_w_shape = strict_small_w_shape
                    eff_w_color = strict_small_w_color
                    eff_w_stripe = strict_small_w_stripe
                image_topk = min(len(req_names), max(top_k * max(cand_multiplier, 1), top_k))
                if recall_cap > 0:
                    image_topk = min(image_topk, recall_cap)
                ranked = search_topk_images(
                    query_path,
                    req_names,
                    req_feats,
                    image_topk,
                    feature_backend,
                    eff_w_clip,
                    eff_w_shape,
                    eff_w_color,
                    eff_w_stripe,
                    query_multicrop=query_multicrop,
                    query_crop_ratio=query_crop_ratio,
                    query_component_views=pass_query_component_views,
                    query_view_consensus_weight=pass_query_view_consensus_weight,
                )
                if secondary_feature_backend and req_secondary_feats is not None and len(req_secondary_names) == len(req_secondary_feats):
                    ranked_secondary = search_topk_images(
                        query_path,
                        req_secondary_names,
                        req_secondary_feats,
                        image_topk,
                        secondary_feature_backend,
                        eff_w_clip,
                        eff_w_shape,
                        eff_w_color,
                        eff_w_stripe,
                        query_multicrop=query_multicrop,
                        query_crop_ratio=query_crop_ratio,
                        query_component_views=pass_query_component_views,
                        query_view_consensus_weight=pass_query_view_consensus_weight,
                    )
                    ranked = merge_ranked_image_lists(
                        ranked,
                        ranked_secondary,
                        secondary_weight=secondary_recall_weight,
                    )
                if region_probe_active and region_crop_recall_enabled and req_region_feats is not None and len(req_region_names) == len(req_region_feats):
                    effective_region_recall_topn_cap = (
                        strict_small_region_recall_topn_cap if strict_small_region_crop else region_crop_recall_topn_cap
                    )
                    if effective_region_recall_topn_cap > 0:
                        region_topk = min(len(req_region_names), effective_region_recall_topn_cap)
                    else:
                        region_topk = min(
                            len(req_region_names),
                            max(top_k * max(cand_multiplier, 1), top_k),
                        )
                    if auto_region_probe_active and auto_region_probe_views:
                        ranked_region = _search_topk_images_from_views(
                            auto_region_probe_views,
                            req_region_names,
                            req_region_feats,
                            region_topk,
                            region_crop_recall_backend,
                            eff_w_clip,
                            eff_w_shape,
                            eff_w_color,
                            eff_w_stripe,
                            query_view_consensus_weight_local=max(
                                float(pass_region_query_view_consensus_weight),
                                0.12,
                            ),
                        )
                    else:
                        ranked_region = search_topk_images(
                            query_path,
                            req_region_names,
                            req_region_feats,
                            region_topk,
                            region_crop_recall_backend,
                            eff_w_clip,
                            eff_w_shape,
                            eff_w_color,
                            eff_w_stripe,
                            query_multicrop=pass_region_query_multicrop,
                            query_crop_ratio=pass_region_query_crop_ratio,
                            query_component_views=pass_region_query_component_views,
                            query_view_consensus_weight=pass_region_query_view_consensus_weight,
                        )
                    if ranked_region:
                        region_focus_debug = ""
                        if crop_active and (use_strip_mode or partial_region_crop or vertical_stripe_region_crop) and not strict_small_region_crop:
                            focus_query_img = Image.open(query_path).convert("RGB")
                            focus_tags = {
                                "top",
                                "top_narrow",
                                "upper_band",
                                "upper_narrow_band",
                                "top_left_band",
                                "top_right_band",
                                "collar_left_focus",
                                "collar_right_focus",
                                "collar_center_bridge",
                            }
                            focus_query_views = [
                                view
                                for tag, view in _region_standard_views(focus_query_img, max_component_views=4)
                                if (tag in focus_tags) or tag.startswith("comp")
                            ]
                            fw, fh = focus_query_img.size
                            grid_boxes = [
                                (0.00, 0.00, 0.55, 0.52),
                                (0.45, 0.00, 1.00, 0.52),
                                (0.00, 0.32, 0.55, 0.84),
                                (0.45, 0.32, 1.00, 0.84),
                                (0.00, 0.00, 0.42, 0.46),
                                (0.29, 0.00, 0.71, 0.46),
                                (0.58, 0.00, 1.00, 0.46),
                                (0.00, 0.38, 0.42, 0.88),
                                (0.29, 0.38, 0.71, 0.88),
                                (0.58, 0.38, 1.00, 0.88),
                            ]
                            seen_focus_keys = {
                                (view.size[0], view.size[1], int(np.asarray(view).mean()))
                                for view in focus_query_views
                            }
                            for x0f, y0f, x1f, y1f in grid_boxes:
                                left = int(round(x0f * fw))
                                top = int(round(y0f * fh))
                                right = int(round(x1f * fw))
                                bottom = int(round(y1f * fh))
                                if right - left < 32 or bottom - top < 32:
                                    continue
                                grid_view = focus_query_img.crop((left, top, right, bottom))
                                key = (grid_view.size[0], grid_view.size[1], int(np.asarray(grid_view).mean()))
                                if key in seen_focus_keys:
                                    continue
                                seen_focus_keys.add(key)
                                focus_query_views.append(grid_view)
                            if partial_region_crop and not use_strip_mode:
                                mirrored_focus_views: List[Image.Image] = []
                                seen_mirror_keys = {
                                    (view.size[0], view.size[1], int(np.asarray(view).mean()))
                                    for view in focus_query_views
                                }
                                for view in list(focus_query_views):
                                    flipped = ImageOps.mirror(view)
                                    key = (flipped.size[0], flipped.size[1], int(np.asarray(flipped).mean()))
                                    if key in seen_mirror_keys:
                                        continue
                                    seen_mirror_keys.add(key)
                                    mirrored_focus_views.append(flipped)
                                focus_query_views.extend(mirrored_focus_views)
                            if vertical_stripe_region_crop:
                                rotated_focus_views: List[Image.Image] = []
                                seen_rotate_keys = {
                                    (view.size[0], view.size[1], int(np.asarray(view).mean()))
                                    for view in focus_query_views
                                }
                                for view in list(focus_query_views):
                                    if view.height <= view.width * 1.15:
                                        continue
                                    for method in (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270):
                                        rotated = view.transpose(method)
                                        key = (rotated.size[0], rotated.size[1], int(np.asarray(rotated).mean()))
                                        if key in seen_rotate_keys:
                                            continue
                                        seen_rotate_keys.add(key)
                                        rotated_focus_views.append(rotated)
                                focus_query_views.extend(rotated_focus_views)
                            focus_region_view_consensus = max(
                                float(pass_region_query_view_consensus_weight),
                                0.18 if (partial_region_crop or vertical_stripe_region_crop) and not use_strip_mode else 0.12,
                            )
                            ranked_region_focus = _search_topk_images_from_views(
                                focus_query_views,
                                req_region_names,
                                req_region_feats,
                                region_topk,
                                region_crop_recall_backend,
                                max(float(pass_w_clip), float(strict_small_w_clip if partial_region_crop else pass_w_clip)),
                                max(float(pass_w_shape), float(strict_small_w_shape if partial_region_crop else pass_w_shape)),
                                min(float(pass_w_color), float(strict_small_w_color if partial_region_crop else pass_w_color)),
                                max(float(pass_w_stripe), float(strict_small_w_stripe if partial_region_crop else pass_w_stripe)),
                                query_view_consensus_weight_local=focus_region_view_consensus,
                            )
                            if ranked_region_focus:
                                ranked_region = merge_ranked_image_lists(
                                    ranked_region,
                                    ranked_region_focus,
                                    secondary_weight=1.0,
                                )
                                region_focus_debug = ",".join(
                                    f"{filename_to_style_code(n)}:{float(s):.3f}"
                                    for n, s in ranked_region_focus[:12]
                                )
                        if region_crop_color_consistency_enabled:
                            q_region_color_sig = _extract_color_sig(query_path)
                            ranked_region = _apply_region_color_consistency(ranked_region, q_region_color_sig)
                        region_best_score = max(float(s) for _n, s in ranked_region[: max(1, min(len(ranked_region), 10))])
                        if region_best_score >= region_crop_suppress_accessory_wide_min_score:
                            region_has_confident_match = True
                        region_code_scores.clear()
                        region_code_best_images.clear()
                        region_code_scan_n = min(
                            len(ranked_region),
                            max(
                                1,
                                region_crop_result_rescue_topn,
                                region_crop_result_rescue_scan_codes,
                                region_crop_code_prior_topn,
                                region_crop_suppress_accessory_topn,
                            ),
                        )
                        for n, s in ranked_region[:region_code_scan_n]:
                            code = filename_to_style_code(n)
                            score = float(s)
                            if score > region_code_scores.get(code, -1e9):
                                region_code_scores[code] = score
                                region_code_best_images[_code_prior_key(code)] = n.split("@", 1)[0]
                        region_repeat_force_scores.clear()
                        if region_crop_repeat_force_enabled:
                            repeat_hits: Dict[str, tuple[float, int]] = {}
                            for n, s in ranked_region[: max(1, int(region_crop_repeat_force_topn))]:
                                code = filename_to_style_code(n)
                                score = float(s)
                                if score < float(region_crop_repeat_force_min_score):
                                    continue
                                best_score, hit_count = repeat_hits.get(code, (-1.0, 0))
                                repeat_hits[code] = (max(float(best_score), score), int(hit_count) + 1)
                            region_repeat_force_scores.update(repeat_hits)
                        region_debug = ",".join(
                            f"{filename_to_style_code(n)}:{float(s):.3f}"
                            for n, s in ranked_region[:40]
                        )
                        if region_focus_debug:
                            region_debug = f"{region_debug}|focus={region_focus_debug}"
                        region_strong_code = ""
                        if region_crop_suppress_accessory_enabled:
                            code_hits: Dict[str, int] = {}
                            for n, s in ranked_region[: max(1, region_crop_suppress_accessory_topn)]:
                                if float(s) < region_crop_suppress_accessory_min_score:
                                    continue
                                code = filename_to_style_code(n)
                                code_hits[code] = code_hits.get(code, 0) + 1
                            if code_hits:
                                best_code, best_hits = max(code_hits.items(), key=lambda item: item[1])
                                if best_hits >= max(1, region_crop_suppress_accessory_min_hits):
                                    region_strong_code = best_code
                        ranked = merge_ranked_image_lists(
                            ranked,
                            ranked_region,
                            secondary_weight=region_crop_recall_weight,
                        )
                        _apply_region_code_prior()
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
                        candidate_feature_cache=req_rerank_candidate_cache,
                        max_unique_codes=pass_rerank_max_unique_codes,
                    )
                    t_rerank_local = time.perf_counter() - t1
                else:
                    t_rerank_local = 0.0
                t2 = time.perf_counter()
                rows_local = _rows_from_ranked(ranked)
                t_post_local = time.perf_counter() - t2
                return ranked, rows_local, t_recall_local, t_rerank_local, t_post_local

            q_shape = _extract_fg_shape(query_path)
            use_strip_mode = False
            if strip_mode_enabled and q_shape is not None:
                qa, qf = q_shape
                use_strip_mode = (qa >= strip_aspect_threshold) or (qf <= strip_fill_threshold)
            q_pre_checker_profile = None
            vertical_stripe_region_crop = False
            if crop_active and not strict_small_region_crop and not use_strip_mode:
                q_pre_checker_profile = _extract_checker_profile(query_path, grid=10)
                if q_pre_checker_profile:
                    vertical_stripe_region_crop = bool(
                        crop_norm_h >= 0.45
                        and float(query_height or 0) > float(query_width or 0) * 1.25
                        and float(q_pre_checker_profile.get("stripe", 0.0)) > float(q_pre_checker_profile.get("checker", 0.0))
                        and float(q_pre_checker_profile.get("bw_mix", 0.0)) >= 0.45
                    )
            partial_region_crop = bool(
                crop_active
                and not use_strip_mode
                and not strict_small_region_crop
                and crop_norm_y <= 0.12
                and crop_orig_area <= 0.36
                and (
                    crop_norm_w <= 0.58
                    or crop_norm_h <= 0.68
                    or crop_norm_x <= 0.18
                    or (crop_norm_x + crop_norm_w) >= 0.82
                )
            )
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
                pass_region_query_multicrop=bool(
                    (crop_active and use_strip_mode and region_strip_query_multicrop_enabled)
                    or (strict_small_region_crop and strict_small_region_query_multicrop_enabled)
                ),
                pass_region_query_crop_ratio=(
                    float(region_strip_query_crop_ratio)
                    if (crop_active and use_strip_mode and region_strip_query_multicrop_enabled)
                    else (
                        float(strict_small_region_query_crop_ratio)
                        if (strict_small_region_crop and strict_small_region_query_multicrop_enabled)
                        else float(query_crop_ratio)
                    )
                ),
                pass_region_query_component_views=bool(
                    (crop_active and use_strip_mode and region_strip_query_component_views)
                    or (strict_small_region_crop and strict_small_region_query_component_views)
                ),
                pass_region_query_view_consensus_weight=(
                    float(region_strip_query_view_consensus_weight)
                    if (crop_active and use_strip_mode)
                    else 0.0
                ),
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
                        pass_region_query_multicrop=bool(crop_active and use_strip_mode and region_strip_query_multicrop_enabled),
                        pass_region_query_crop_ratio=(
                            float(region_strip_query_crop_ratio)
                            if (crop_active and use_strip_mode and region_strip_query_multicrop_enabled)
                            else float(query_crop_ratio)
                        ),
                        pass_region_query_component_views=bool(crop_active and use_strip_mode and region_strip_query_component_views),
                        pass_region_query_view_consensus_weight=(
                            float(region_strip_query_view_consensus_weight)
                            if (crop_active and use_strip_mode)
                            else 0.0
                        ),
                    )
                    t_recall += t2_recall
                    t_rerank += t2_rerank
                    t_post += t2_post
                    second_pass_used = True
            if shape_consistency_enabled and q_shape is not None and not (strict_small_region_crop and strict_small_disable_consistency):
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
                    display_score_scale=display_score_scale,
                    display_score_bias=display_score_bias,
                )
            if mask_consistency_enabled and not (strict_small_region_crop and strict_small_disable_consistency):
                q_mask_vec = _extract_fg_mask_vec(query_path, size=64)
                if q_mask_vec is not None:
                    ranked_images = _apply_mask_consistency(ranked_images, q_mask_vec)
                    rows = topk_style_codes(
                        ranked_images,
                        top_k,
                        min_score=min_score,
                        code_agg_top_n=code_agg_top_n,
                        code_agg_alpha=code_agg_alpha,
                        query_hint_code=query_hint_code,
                        query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                        code_prior_boost=code_prior_boost,
                        display_score_scale=display_score_scale,
                        display_score_bias=display_score_bias,
                    )
            if stripe_consistency_enabled and not (strict_small_region_crop and strict_small_disable_consistency):
                q_stripe_sig = _extract_stripe_sig(query_path, keep=24)
                if q_stripe_sig is not None:
                    ranked_images = _apply_stripe_consistency(ranked_images, q_stripe_sig)
                    rows = topk_style_codes(
                        ranked_images,
                        top_k,
                        min_score=min_score,
                        code_agg_top_n=code_agg_top_n,
                        code_agg_alpha=code_agg_alpha,
                        query_hint_code=query_hint_code,
                        query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                        code_prior_boost=code_prior_boost,
                        display_score_scale=display_score_scale,
                        display_score_bias=display_score_bias,
                    )
            if (pattern_consistency_enabled or strict_small_pattern_region_order_enabled or strict_small_pattern_region_rescue_enabled) and q_pattern_sig is None:
                pattern_query_path = tight_query_path if (strict_small_region_crop and tight_query_available) else query_path
                q_pattern_sig = _extract_pattern_sig(pattern_query_path, size=14)
            if strict_small_dark_motif_rescue_enabled and q_dark_motif_sig is None:
                dark_motif_query_path = tight_query_path if (strict_small_region_crop and tight_query_available) else query_path
                q_dark_motif_sig = _extract_dark_motif_sig(dark_motif_query_path, size=18)
            if pattern_consistency_enabled:
                if q_pattern_sig is not None:
                    ranked_images = _apply_pattern_consistency(ranked_images, q_pattern_sig)
                    pattern_code_boost = _build_pattern_code_boost(ranked_images, q_pattern_sig)
                    if pattern_code_boost:
                        code_prior_boost = dict(base_code_prior_boost)
                        for code_key, boost in pattern_code_boost.items():
                            code_prior_boost[code_key] = code_prior_boost.get(code_key, 0.0) + float(boost)
                        _apply_region_code_prior()
                    rows = topk_style_codes(
                        ranked_images,
                        top_k,
                        min_score=min_score,
                        code_agg_top_n=code_agg_top_n,
                        code_agg_alpha=code_agg_alpha,
                        query_hint_code=query_hint_code,
                        query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                        code_prior_boost=code_prior_boost,
                        display_score_scale=display_score_scale,
                        display_score_bias=display_score_bias,
                    )
            q_accent_sig = None
            q_checker_profile = None
            strip_crop_accent_disabled = bool(
                crop_active and use_strip_mode and region_crop_disable_accent_when_strip
            )
            accent_small_region_allowed = bool(
                crop_active
                and (strict_small_region_crop or partial_region_crop)
                and region_best_score <= max(0.0, float(accent_pattern_small_region_max_score))
            )
            accent_pattern_allowed = (
                accent_pattern_enabled
                and ((not strict_small_region_crop and not partial_region_crop) or accent_small_region_allowed)
                and not strip_crop_accent_disabled
                and (not crop_active or accent_pattern_crop_enabled)
            )
            if accent_pattern_allowed:
                q_accent_sig = _extract_accent_pattern_sig(query_path, grid=12)
                if q_accent_sig is not None:
                    accent_debug = "1"
            checker_large_crop_blocked = bool(
                crop_active
                and crop_final_area > max(0.0, min(1.0, float(checker_crop_max_area)))
            )
            checker_large_crop_allowed = bool(
                checker_large_crop_blocked
                and q_pre_checker_profile is not None
                and crop_final_area <= max(float(checker_crop_max_area), min(1.0, float(checker_large_crop_max_area)))
                and float(q_pre_checker_profile.get("checker", 0.0)) >= float(checker_large_crop_query_threshold)
                and float(q_pre_checker_profile.get("bw_mix", 0.0)) >= float(checker_large_crop_bw_mix)
            )
            checker_blocked_by_region_probe = bool(
                auto_region_probe_active
                and region_best_score >= float(scene_text_suppress_when_region_min_score)
            )
            if (
                checker_consistency_enabled
                and not strict_small_region_crop
                and (not checker_large_crop_blocked or checker_large_crop_allowed)
                and not checker_blocked_by_region_probe
            ):
                q_checker_profile = q_pre_checker_profile or _extract_checker_profile(query_path, grid=10)
                if q_checker_profile:
                    checker_debug = (
                        f"{float(q_checker_profile.get('checker', 0.0)):.3f}/"
                        f"{float(q_checker_profile.get('stripe', 0.0)):.3f}/"
                        f"{float(q_checker_profile.get('bw_mix', 0.0)):.3f}"
                    )
                    if (
                        checker_suppress_when_accent
                        and q_accent_sig is not None
                        and float(q_checker_profile.get("checker", 0.0)) < checker_accent_suppress_below
                    ):
                        q_checker_profile = None
                ranked_images, checker_code_boost, checker_candidates_debug = _apply_checker_consistency(
                    ranked_images,
                    q_checker_profile,
                )
                ranked_images, checker_seed_debug = _merge_checker_seed_candidates(
                    ranked_images,
                    q_checker_profile,
                )
                if checker_seed_debug:
                    checker_candidates_debug = (
                        f"{checker_candidates_debug}|seed={checker_seed_debug}"
                        if checker_candidates_debug
                        else f"seed={checker_seed_debug}"
                    )
                if checker_code_boost:
                    code_prior_boost = dict(code_prior_boost)
                    for code_key, boost in checker_code_boost.items():
                        code_prior_boost[code_key] = code_prior_boost.get(code_key, 0.0) + float(boost)
                if checker_code_boost or checker_seed_debug:
                    rows = topk_style_codes(
                        ranked_images,
                        top_k,
                        min_score=min_score,
                        code_agg_top_n=code_agg_top_n,
                        code_agg_alpha=code_agg_alpha,
                        query_hint_code=query_hint_code,
                        query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                        code_prior_boost=code_prior_boost,
                        display_score_scale=display_score_scale,
                        display_score_bias=display_score_bias,
                    )
            elif checker_consistency_enabled and checker_large_crop_blocked:
                checker_debug = f"skip-large-crop:{crop_final_area:.3f}"
            elif checker_consistency_enabled and checker_blocked_by_region_probe:
                checker_debug = f"skip-region-probe:{region_best_score:.3f}"
            if accent_pattern_allowed:
                if q_accent_sig is not None:
                    ranked_images, accent_candidates_debug = _merge_accent_pattern_candidates(
                        ranked_images,
                        q_accent_sig,
                    )
                    if accent_candidates_debug:
                        rows = topk_style_codes(
                            ranked_images,
                            top_k,
                            min_score=min_score,
                            code_agg_top_n=code_agg_top_n,
                            code_agg_alpha=code_agg_alpha,
                            query_hint_code=query_hint_code,
                            query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                            code_prior_boost=code_prior_boost,
                            display_score_scale=display_score_scale,
                            display_score_bias=display_score_bias,
                        )
            collar_contour_query_allowed = bool(
                collar_contour_enabled
                and crop_active
                and not strict_small_region_crop
                and (partial_region_crop or use_strip_mode)
            )
            if collar_contour_query_allowed:
                q_collar_sigs = _extract_collar_contour_sigs(query_path, size=collar_contour_size)
                q_collar_sig_mirrors: List[np.ndarray] = []
                q_collar_chevron_score = 0.0
                if q_collar_sigs:
                    try:
                        with Image.open(query_path) as q_im0:
                            q_img = q_im0.convert("RGB")
                            if collar_contour_query_component_views:
                                query_collar_tags = {
                                    "top",
                                    "top_narrow",
                                    "upper_band",
                                    "upper_narrow_band",
                                    "top_left_band",
                                    "top_right_band",
                                    "collar_left_focus",
                                    "collar_right_focus",
                                    "collar_center_bridge",
                                }
                                for tag, view in _region_standard_views(q_img, max_component_views=3):
                                    if not (
                                        tag in query_collar_tags
                                        or tag.startswith("comp")
                                        or tag.startswith("top_comp")
                                    ):
                                        continue
                                    _append_unique_collar_sigs(
                                        q_collar_sigs,
                                        _extract_collar_contour_sigs_from_image(view, size=collar_contour_size),
                                        collar_contour_query_max_sigs,
                                    )
                            q_collar_sig_mirrors = _extract_collar_contour_sigs_from_image(
                                ImageOps.mirror(q_img),
                                size=collar_contour_size,
                            )
                            if collar_contour_query_component_views:
                                mirrored_img = ImageOps.mirror(q_img)
                                for tag, view in _region_standard_views(mirrored_img, max_component_views=3):
                                    if not (
                                        tag in query_collar_tags
                                        or tag.startswith("comp")
                                        or tag.startswith("top_comp")
                                    ):
                                        continue
                                    _append_unique_collar_sigs(
                                        q_collar_sig_mirrors,
                                        _extract_collar_contour_sigs_from_image(view, size=collar_contour_size),
                                        collar_contour_query_max_sigs,
                                    )
                            if collar_chevron_enabled:
                                q_collar_chevron_score = max(
                                    _extract_collar_chevron_score_from_image(q_img),
                                    _extract_collar_chevron_score_from_image(ImageOps.mirror(q_img)),
                                )
                    except Exception:
                        q_collar_sig_mirrors = []
                        q_collar_chevron_score = 0.0
                    ranked_images, collar_candidates_debug, collar_code_matches = _merge_collar_contour_candidates(
                        ranked_images,
                        q_collar_sigs,
                        q_collar_sig_mirrors,
                        query_chevron_score=q_collar_chevron_score,
                    )
                    if collar_candidates_debug:
                        for code, (sim, image_name) in collar_code_matches.items():
                            code_key = _code_prior_key(code)
                            collar_candidate_scores[code_key] = max(
                                float(collar_candidate_scores.get(code_key, -1e9)),
                                float(sim),
                            )
                            region_score = float(collar_contour_region_score_base) + float(collar_contour_region_score_scale) * max(
                                0.0,
                                min(float(collar_contour_region_score_max), float(sim)),
                            )
                            region_code_scores[code] = max(float(region_code_scores.get(code, -1e9)), region_score)
                            region_code_best_images[code_key] = image_name
                            if collar_contour_code_prior_boost > 0.0:
                                code_prior_boost[code_key] = max(
                                    float(code_prior_boost.get(code_key, 0.0)),
                                    float(collar_contour_code_prior_boost),
                                )
                        region_debug = (
                            f"{region_debug}|collar={collar_candidates_debug}"
                            if region_debug
                            else f"collar={collar_candidates_debug}"
                        )
                        rows = topk_style_codes(
                            ranked_images,
                            max(top_k, collar_contour_max_injected),
                            min_score=min_score,
                            code_agg_top_n=code_agg_top_n,
                            code_agg_alpha=code_agg_alpha,
                            query_hint_code=query_hint_code,
                            query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                            code_prior_boost=code_prior_boost,
                            display_score_scale=display_score_scale,
                            display_score_bias=display_score_bias,
                        )
                        existing_rows_by_key = {
                            _code_prior_key(str(row.get("style_code", ""))): row
                            for row in rows
                        }
                        for code, (sim, image_name) in collar_code_matches.items():
                            code_key = _code_prior_key(code)
                            if not code_key:
                                continue
                            raw_score = (
                                float(collar_contour_seed_score_base)
                                + float(collar_contour_boost_scale) * max(0.0, float(sim))
                                + float(code_prior_boost.get(code_key, 0.0))
                            )
                            z = float(display_score_scale) * (float(raw_score) - float(display_score_bias))
                            disp = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                            disp = min(0.9999, max(0.0, float(disp)))
                            existing_row = existing_rows_by_key.get(code_key)
                            if existing_row is not None:
                                if float(raw_score) > float(existing_row.get("rank_score", -1e9)):
                                    existing_row["best_standard_image"] = image_name
                                    existing_row["score"] = round(disp, 4)
                                    existing_row["rank_score"] = round(float(raw_score), 6)
                                continue
                            rows.append(
                                {
                                    "style_code": code,
                                    "best_standard_image": image_name,
                                    "score": round(disp, 4),
                                    "rank_score": round(float(raw_score), 6),
                                }
                            )
                            existing_rows_by_key[code_key] = rows[-1]
            accessory_like_region = False
            accessory_near_square_region = False
            crop_aspect = 0.0
            q_accessory_hat_prior = 0.0
            if crop_active:
                try:
                    with Image.open(query_path) as q_im:
                        qw, qh = q_im.size
                    crop_aspect = float(qw) / float(qh) if qh > 0 else 0.0
                    accessory_like_region = crop_aspect >= 1.10
                    checker_is_strong = (
                        q_checker_profile is not None
                        and float(q_checker_profile.get("checker", 0.0)) >= checker_suppress_sleeve_threshold
                        and float(q_checker_profile.get("bw_mix", 0.0)) >= checker_suppress_sleeve_bw_mix
                    )
                    accessory_near_square_region = (
                        accessory_near_square_crop_enabled
                        and not checker_is_strong
                        and crop_final_area <= max(0.0, min(1.0, float(accessory_near_square_crop_max_area)))
                        and accessory_near_square_crop_min_aspect <= crop_aspect <= accessory_near_square_crop_max_aspect
                    )
                    if accessory_near_square_region:
                        q_accessory_hat_prior = _extract_accessory_hat_prior(query_path)
                except Exception:
                    accessory_like_region = False
                    accessory_near_square_region = False
                    q_accessory_hat_prior = 0.0
            if crop_active and region_debug and not region_has_confident_match:
                try:
                    parsed_region_scores = [
                        float(part.rsplit(":", 1)[1])
                        for part in region_debug.split(",")[: max(1, region_crop_suppress_accessory_topn)]
                        if ":" in part
                    ]
                    if parsed_region_scores:
                        region_best_score = max(region_best_score, max(parsed_region_scores))
                        if region_best_score >= region_crop_suppress_accessory_wide_min_score:
                            region_has_confident_match = True
                except Exception:
                    pass
            suppress_accessory_for_region_hit = crop_active and (
                bool(region_strong_code)
                or (
                    bool(accent_candidates_debug)
                    and (
                        (active_match_mode == "similar_style" and accessory_region_suppress_when_accent)
                        or
                        not accessory_near_square_region
                        or (
                            accessory_region_requires_hat_prior
                            and q_accessory_hat_prior < accessory_region_hat_prior_threshold
                        )
                    )
                )
                or region_has_confident_match
                or (accessory_disable_wide_crop_enabled and accessory_like_region and not accessory_near_square_region)
            )
            if (
                suppress_accessory_for_region_hit
                and not region_has_confident_match
                and accessory_near_square_region
                and crop_aspect <= accessory_hat_override_max_aspect
                and q_accessory_hat_prior >= accessory_region_hat_prior_threshold
            ):
                suppress_accessory_for_region_hit = False
            if suppress_accessory_for_region_hit:
                accessory_debug = f"skip-region:{region_best_score:.3f}/{q_accessory_hat_prior:.3f}"

            if (
                accessory_pattern_enabled
                and not strict_small_region_crop
                and (accessory_like_region or accessory_near_square_region)
                and not suppress_accessory_for_region_hit
            ):
                q_accessory_sig = _extract_accessory_pattern_sig(query_path, size=48)
                if q_accessory_sig is not None:
                    accessory_debug = "1"
                    ranked_images, accessory_candidates_debug = _merge_accessory_pattern_candidates(
                        ranked_images,
                        q_accessory_sig,
                        query_hat_prior=q_accessory_hat_prior,
                    )
                    if accessory_candidates_debug:
                        rows = topk_style_codes(
                            ranked_images,
                            top_k,
                            min_score=min_score,
                            code_agg_top_n=code_agg_top_n,
                            code_agg_alpha=code_agg_alpha,
                            query_hint_code=query_hint_code,
                            query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                            code_prior_boost=code_prior_boost,
                            display_score_scale=display_score_scale,
                            display_score_bias=display_score_bias,
                        )

            # Full model/catalog photos with logo/letter/color-block accents often trigger a
            # false sleeve match from small local regions. Keep sleeve matching for explicit
            # region crops, but do not let it override strong full-image accent candidates.
            suppress_sleeve_for_accent_query = (
                sleeve_pattern_skip_when_full_accent
                and not crop_active
                and bool(accent_candidates_debug)
            )
            suppress_sleeve_for_checker_query = (
                crop_active
                and q_checker_profile is not None
                and float(q_checker_profile.get("checker", 0.0)) >= checker_suppress_sleeve_threshold
                and float(q_checker_profile.get("bw_mix", 0.0)) >= checker_suppress_sleeve_bw_mix
            )
            suppress_sleeve_for_accessory_query = (
                crop_active
                and accessory_near_square_region
                and bool(accessory_candidates_debug)
            )
            sleeve_large_region_query = bool(
                crop_active
                and crop_final_area > max(0.0, min(1.0, float(sleeve_pattern_crop_max_area)))
            )
            sleeve_large_region_rescue_allowed = bool(
                sleeve_large_region_query
                and sleeve_pattern_large_region_rescue_enabled
                and crop_final_area <= max(0.0, min(1.0, float(sleeve_pattern_large_region_rescue_max_area)))
                and region_best_score <= max(0.0, float(sleeve_pattern_large_region_rescue_max_score))
            )
            suppress_sleeve_for_large_region_query = bool(
                sleeve_large_region_query and not sleeve_large_region_rescue_allowed
            )
            sleeve_small_region_rescue_allowed = bool(
                crop_active
                and strict_small_region_crop
                and sleeve_pattern_small_region_enabled
                and not suppress_sleeve_for_large_region_query
                and region_best_score <= max(0.0, float(sleeve_pattern_small_region_max_score))
            )
            suppress_sleeve_for_small_region_query = bool(
                strict_small_region_crop and not sleeve_small_region_rescue_allowed
            )
            if (
                sleeve_pattern_enabled
                and not accessory_like_region
                and not suppress_sleeve_for_accent_query
                and not suppress_sleeve_for_checker_query
                and not suppress_sleeve_for_accessory_query
                and not suppress_sleeve_for_small_region_query
                and not suppress_sleeve_for_large_region_query
            ):
                q_sleeve_sig = _extract_sleeve_pattern_sig(query_path, size=32)
                if q_sleeve_sig is not None:
                    sleeve_debug = "1"
                    ranked_images, sleeve_candidates_debug = _merge_sleeve_pattern_candidates(
                        ranked_images,
                        q_sleeve_sig,
                    )
                    if sleeve_candidates_debug:
                        rows = topk_style_codes(
                            ranked_images,
                            top_k,
                            min_score=min_score,
                            code_agg_top_n=code_agg_top_n,
                            code_agg_alpha=code_agg_alpha,
                            query_hint_code=query_hint_code,
                            query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                            code_prior_boost=code_prior_boost,
                            display_score_scale=display_score_scale,
                            display_score_bias=display_score_bias,
                        )
            elif sleeve_pattern_enabled and suppress_sleeve_for_large_region_query:
                sleeve_debug = f"skip-large-crop:{crop_final_area:.3f}"
            if (
                accessory_pattern_enabled
                and not strict_small_region_crop
                and not accessory_like_region
                and not suppress_accessory_for_region_hit
                and not sleeve_candidates_debug
                and not accessory_candidates_debug
                and not (not crop_active and bool(accent_candidates_debug))
            ):
                q_accessory_sig = _extract_accessory_pattern_sig(query_path, size=48)
                if q_accessory_sig is not None:
                    accessory_debug = "1"
                    q_accessory_hat_prior = 0.0
                    ranked_images, accessory_candidates_debug = _merge_accessory_pattern_candidates(
                        ranked_images,
                        q_accessory_sig,
                        query_hat_prior=q_accessory_hat_prior,
                    )
                    if accessory_candidates_debug:
                        rows = topk_style_codes(
                            ranked_images,
                            top_k,
                            min_score=min_score,
                            code_agg_top_n=code_agg_top_n,
                            code_agg_alpha=code_agg_alpha,
                            query_hint_code=query_hint_code,
                            query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                            code_prior_boost=code_prior_boost,
                            display_score_scale=display_score_scale,
                            display_score_bias=display_score_bias,
                        )
            if phash_enabled and not (strict_small_region_crop and strict_small_disable_consistency):
                q_bits = _extract_phash_bits(query_path, size=32, keep=8)
                if q_bits is not None:
                    ranked_images = _apply_phash_consistency(ranked_images, q_bits)
                    rows = topk_style_codes(
                        ranked_images,
                        top_k,
                        min_score=min_score,
                        code_agg_top_n=code_agg_top_n,
                        code_agg_alpha=code_agg_alpha,
                        query_hint_code=query_hint_code,
                        query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                        code_prior_boost=code_prior_boost,
                        display_score_scale=display_score_scale,
                        display_score_bias=display_score_bias,
                    )
            scene_text_small_region_allowed = bool(
                crop_active
                and (strict_small_region_crop or partial_region_crop)
                and region_best_score <= max(0.0, float(scene_text_small_region_max_score))
            )
            scene_text_blocked_by_region = bool(
                region_probe_active
                and (search_scope == "region_primary" or auto_region_probe_active)
                and region_best_score >= float(scene_text_suppress_when_region_min_score)
                and not scene_text_small_region_allowed
            )
            if (
                scene_text_hint_enabled
                and (not strict_small_region_crop or scene_text_small_region_allowed)
                and not scene_text_blocked_by_region
            ):
                ranked_images, scene_text_tokens = merge_scene_text_candidates(
                    ranked_images,
                    query_path,
                    req_scene_text_index,
                    seed_score_base=scene_text_seed_score_base,
                    boost_scale=scene_text_boost_scale,
                    min_token_len=scene_text_min_token_len,
                    include_components=True,
                    max_candidates_per_token=scene_text_max_candidates_per_token,
                    max_injected=scene_text_max_injected,
                )
                if scene_text_tokens:
                    rows = topk_style_codes(
                        ranked_images,
                        top_k,
                        min_score=min_score,
                        code_agg_top_n=code_agg_top_n,
                        code_agg_alpha=code_agg_alpha,
                        query_hint_code=query_hint_code,
                        query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
                        code_prior_boost=code_prior_boost,
                        display_score_scale=display_score_scale,
                        display_score_bias=display_score_bias,
                    )
            elif scene_text_hint_enabled and scene_text_blocked_by_region:
                scene_text_tokens = [f"skip-region:{region_best_score:.3f}"]

            _apply_sleeve_region_rescue()
            rows = _rescue_region_rows(rows, ranked_images)
            rows = _rescue_sleeve_region_rows(rows, ranked_images)
            rows = _rescue_hat_from_sleeve_region_rows(rows, ranked_images)
            rows = _force_top_region_rows(rows)
            rows = _rescue_pattern_region_rows(rows, ranked_images)
            rows = _rescue_dark_motif_region_rows(rows)
            rows = _order_region_primary_rows(rows)
            rows = _rescue_hat_region_rows(rows, ranked_images)
            rows = _rescue_hat_family_region_rows(rows)
            rows = _rescue_checker_region_rows(rows, ranked_images)
            rows = _rescue_accent_region_rows(rows, ranked_images)
            rows = _rescue_scene_text_region_rows(rows, ranked_images)
            _make_display_scores_follow_order(rows)

            rows = _dedupe_search_rows(rows, top_k)

            if low_confidence_enabled and rows:
                top1 = float(rows[0].get("score", 0.0))
                top2 = float(rows[1].get("score", 0.0)) if len(rows) > 1 else 0.0
                gap = top1 - top2
                low_conf = len(rows) > 1 and gap < low_confidence_margin_threshold
                rows[0]["low_confidence"] = bool(low_conf)
                rows[0]["confidence_gap"] = round(gap, 4)
                if low_conf and len(rows) > 1:
                    rows[1]["low_confidence"] = True
                    rows[1]["confidence_gap"] = round(gap, 4)

        base_url = _external_base_url(request)
        for row in rows:
            row.pop("_force_keep", None)
            row.pop("_region_rescue_keep", None)
            row.pop("_sleeve_diversity_keep", None)
            row.pop("_sleeve_sim", None)
            row.pop("_sleeve_pair_prior", None)
            img = str(row.get("best_standard_image", "")).strip()
            row["best_standard_image_url"] = _build_image_url(base_url, img)
        rows = _enrich_search_rows(base_url, rows)

        similar_images: List[Dict[str, Any]] = []
        seen = set()
        max_n = max(1, region_similar_images_topn if crop_active else similar_images_topn)

        def _append_similar_image(
            image_name: str,
            style_code: str,
            rank_score: float,
            display_score: float | None = None,
        ) -> None:
            if len(similar_images) >= max_n:
                return
            file_name = image_name.split("@", 1)[0]
            code = style_code or filename_to_style_code(file_name)
            seen_key = str(code).strip().upper()
            if seen_key.startswith("MY-") or Path(file_name).name.upper().startswith("MY-"):
                return
            if not file_name or seen_key in seen:
                return
            seen.add(seen_key)
            if display_score is None:
                z = float(display_score_scale) * (float(rank_score) - float(display_score_bias))
                display_score = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                display_score = min(0.9999, max(0.0, float(display_score)))
            similar_images.append(
                {
                    "image_name": file_name,
                    "style_code": code,
                    "image_url": _build_image_url(base_url, file_name),
                    "rank_score": round(float(rank_score), 6),
                    "score": round(float(display_score), 4),
                }
            )

        if crop_active:
            for row in rows:
                img = str(row.get("best_standard_image", "")).strip()
                if not img:
                    continue
                _append_similar_image(
                    img,
                    str(row.get("style_code", "")).strip(),
                    float(row.get("rank_score", 0.0)),
                    float(row.get("score", 0.0)),
                )
                if len(similar_images) >= max_n:
                    break

        for name, score in ranked_images:
            file_name = name.split("@", 1)[0]
            style_code = filename_to_style_code(file_name)
            _append_similar_image(file_name, style_code, float(score))
            if len(similar_images) >= max_n:
                break
        similar_images = _enrich_similar_images(base_url, similar_images[:max_n])

        if include_image_base64:
            n = len(rows) if base64_topn <= 0 else min(len(rows), base64_topn)
            for i in range(n):
                img = str(rows[i].get("best_standard_image", "")).strip()
                b64, mime = _image_b64(img)
                rows[i]["best_standard_image_base64"] = b64
                rows[i]["best_standard_image_mime"] = mime

        logging.info(
            "search timing user=%s file=%s recall=%.3fs rerank=%.3fs post=%.3fs second_pass=%s strip_mode=%s strategy=%s result_codes=%s similar_codes=%s region=%s region_boost=%s region_rescue=%s region_order=%s checker=%s checker_candidates=%s accent=%s accent_candidates=%s sleeve=%s sleeve_candidates=%s accessory=%s accessory_candidates=%s scene_tokens=%s total=%.3fs",
            getattr(request.state, "api_user", "unknown"),
            file.filename,
            t_recall,
            t_rerank,
            t_post,
            second_pass_used,
            use_strip_mode,
            search_strategy,
            ",".join(str(row.get("style_code", "")) for row in rows[:top_k]),
            ",".join(str(item.get("style_code", "")) for item in similar_images[:max_n]),
            region_debug,
            region_boost_debug,
            region_rescue_debug,
            region_order_debug,
            checker_debug,
            checker_candidates_debug,
            accent_debug,
            accent_candidates_debug,
            sleeve_debug,
            sleeve_candidates_debug,
            accessory_debug,
            accessory_candidates_debug,
            ",".join(scene_text_tokens[:6]),
            time.perf_counter() - t_all,
        )

        top1_rank = float(rows[0].get("rank_score", 0.0)) if rows else 0.0
        top2_rank = float(rows[1].get("rank_score", 0.0)) if len(rows) > 1 else 0.0
        top1_disp = float(rows[0].get("score", 0.0)) if rows else 0.0
        top2_disp = float(rows[1].get("score", 0.0)) if len(rows) > 1 else 0.0
        rank_gap = top1_rank - top2_rank
        disp_gap = top1_disp - top2_disp
        is_ambiguous = (len(rows) > 1 and rank_gap < low_confidence_margin_threshold) or (top1_disp < low_confidence_top1_threshold)
        if rank_gap >= confidence_high_threshold:
            confidence_band = "high"
        elif rank_gap >= confidence_medium_threshold:
            confidence_band = "medium"
        else:
            confidence_band = "low"
        if top1_disp < low_confidence_top1_threshold:
            confidence_band = "low"
        logging.info(
            "search confidence user=%s top1_rank=%.4f top2_rank=%.4f rank_gap=%.4f top1_score=%.4f top2_score=%.4f score_gap=%.4f ambiguous=%s band=%s",
            getattr(request.state, "api_user", "unknown"),
            top1_rank,
            top2_rank,
            rank_gap,
            top1_disp,
            top2_disp,
            disp_gap,
            is_ambiguous,
            confidence_band,
        )

        return {
            "query_image": file.filename,
            "topk_style_codes": rows,
            "similar_images": similar_images,
            "is_ambiguous": is_ambiguous,
            "confidence_band": confidence_band,
            "api_user": getattr(request.state, "api_user", "unknown"),
        }

    @app.get("/api/v1/templates")
    def api_list_templates() -> List[Dict[str, Any]]:
        try:
            return list_templates()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/v1/images/upload")
    async def api_upload_image(file: UploadFile = File(...)) -> Dict[str, Any]:
        suffix = Path(file.filename or "").suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise HTTPException(status_code=400, detail="仅支持 jpg/jpeg/png/webp")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="空文件")
        _check_search_upload_content_security(content, file.filename)

        try:
            return process_upload(content, suffix=suffix)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"处理图片失败: {exc}") from exc

    @app.post("/api/v1/render")
    def api_render_layout(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            return render_layout(payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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
        _check_search_upload_content_security(content, file.filename)
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
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(""),
        target_hex: str = Form("FF5500"),
        x_ratio: float = Form(0.2),
        y_ratio: float = Form(0.2),
        w_ratio: float = Form(0.4),
        h_ratio: float = Form(0.4),
        strength: float = Form(0.7),
        prompt: str = Form(""),
        negative_prompt: str = Form(""),
        seed: int | None = Form(None),
        cfg: float | None = Form(None),
        cfg_scale: float | None = Form(None),
        num_inference_steps: int | None = Form(None),
        postprocess: int = Form(1),
        image2: str | None = Form(None),
        image3: str | None = Form(None),
        image2_crop_x: float | None = Form(None),
        image2_crop_y: float | None = Form(None),
        image2_crop_w: float | None = Form(None),
        image2_crop_h: float | None = Form(None),
    ) -> Dict[str, Any]:
        suffix = Path(file.filename or "").suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise HTTPException(status_code=400, detail="仅支持 jpg/jpeg/png/webp")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="空文件")
        _check_text_content_security(prompt, negative_prompt, openid=_wechat_openid_from_request(request))
        _check_search_upload_content_security(content, file.filename)
        try:
            ark_public_base_url = _external_base_url(request)
            if "127.0.0.1" in ark_public_base_url or "localhost" in ark_public_base_url:
                ark_public_base_url = ""
            return recolor_region_ai(
                file_bytes=content,
                suffix=suffix,
                api_key=os.getenv("ARK_API_KEY", "").strip(),
                model=ai_generation_model or model,
                target_hex=target_hex,
                x_ratio=x_ratio,
                y_ratio=y_ratio,
                w_ratio=w_ratio,
                h_ratio=h_ratio,
                strength=strength,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=ai_generation_seed if ai_generation_seed is not None else seed,
                cfg=cfg if cfg is not None else cfg_scale,
                num_inference_steps=num_inference_steps,
                postprocess=bool(int(postprocess)),
                image2=image2,
                image3=image3,
                image2_crop_x=image2_crop_x,
                image2_crop_y=image2_crop_y,
                image2_crop_w=image2_crop_w,
                image2_crop_h=image2_crop_h,
                size=ai_generation_size,
                watermark=ai_generation_watermark,
                output_format=ai_generation_output_format,
                sequential_image_generation=ai_generation_sequential,
                public_base_url=ark_public_base_url,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logging.exception("recolor-ai failed unexpectedly")
            raise HTTPException(status_code=500, detail="融合预览服务暂不可用，请稍后再试") from exc

    app.state.ready = True
    app.state.ready_detail = "ready"
    return app


app = create_app(Path(os.getenv("SEARCH_CONFIG", str(DEFAULT_CONFIG))))
