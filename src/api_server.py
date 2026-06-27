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
from catalog_store import CatalogStore
from extract_style_codes import build_header_crops, code_to_filename_prefix, try_extract_code_from_image


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


class CatalogImportPrepareRequest(BaseModel):
    source_dir: str


class CatalogImportCommitItem(BaseModel):
    source_rel_path: str
    target_filename: str = ""
    year_tag: str = ""
    tags: List[str] = []
    selected: bool = True


class CatalogImportCommitRequest(BaseModel):
    job_id: str
    items: List[CatalogImportCommitItem]


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
    region_crop_sleeve_rescue_enabled = bool(search_cfg.get("region_crop_sleeve_rescue_enabled", True))
    region_crop_sleeve_rescue_min_sim = float(search_cfg.get("region_crop_sleeve_rescue_min_sim", 0.70))
    region_crop_sleeve_rescue_min_pair_prior = float(search_cfg.get("region_crop_sleeve_rescue_min_pair_prior", 0.70))
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
    accent_region_rescue_enabled = bool(search_cfg.get("accent_region_rescue_enabled", True))
    accent_region_rescue_min_sim = float(search_cfg.get("accent_region_rescue_min_sim", 0.70))
    accent_region_rescue_max_rows = int(search_cfg.get("accent_region_rescue_max_rows", 3))
    collar_contour_enabled = bool(search_cfg.get("collar_contour_enabled", True))
    collar_contour_seed_score_base = float(search_cfg.get("collar_contour_seed_score_base", 1.12))
    collar_contour_boost_scale = float(search_cfg.get("collar_contour_boost_scale", 0.28))
    collar_contour_min_score = float(search_cfg.get("collar_contour_min_score", 0.52))
    collar_contour_max_injected = int(search_cfg.get("collar_contour_max_injected", 24))
    collar_contour_size = int(search_cfg.get("collar_contour_size", 48))
    collar_contour_code_prior_boost = float(search_cfg.get("collar_contour_code_prior_boost", 0.10))
    collar_contour_region_score_base = float(search_cfg.get("collar_contour_region_score_base", 0.84))
    collar_contour_region_score_scale = float(search_cfg.get("collar_contour_region_score_scale", 0.18))
    collar_contour_force_keep_min_rank = float(search_cfg.get("collar_contour_force_keep_min_rank", 1.60))
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
    sleeve_pattern_skip_when_full_accent = bool(search_cfg.get("sleeve_pattern_skip_when_full_accent", True))
    sleeve_pattern_crop_max_area = float(search_cfg.get("sleeve_pattern_crop_max_area", 0.28))
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
    strip_mode_enabled = bool(search_cfg.get("strip_mode_enabled", True))
    strip_aspect_threshold = float(search_cfg.get("strip_aspect_threshold", 2.4))
    strip_fill_threshold = float(search_cfg.get("strip_fill_threshold", 0.42))
    strip_w_clip = float(search_cfg.get("strip_w_clip", 0.35))
    strip_w_shape = float(search_cfg.get("strip_w_shape", 0.30))
    strip_w_color = float(search_cfg.get("strip_w_color", 0.10))
    strip_w_stripe = float(search_cfg.get("strip_w_stripe", 0.25))
    auth_cfg = cfg.get("auth", {})
    catalog_cfg = cfg.get("catalog", {})
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
    catalog_import_source_dir = str(catalog_cfg.get("import_source_dir", "")).strip()
    catalog_public = bool(catalog_cfg.get("public_endpoints", True))
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

    def _region_feature_cache_path(backend: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(standard_dir))
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

    def _build_region_feature_db_with_cache() -> tuple[List[str], np.ndarray]:
        files = collect_images(standard_dir, standard_pattern, image_exts)
        sigs = []
        for fp in files:
            st = fp.stat()
            sigs.append(f"{fp.name}|{st.st_size}|{int(st.st_mtime)}")
        cache_key = json.dumps(
            {
                "kind": "region_crop_recall",
                "version": 6,
                "backend": region_crop_recall_backend,
                "weights": [region_w_clip, region_w_shape, region_w_color, region_w_stripe],
                "standard_views": "grid_halves_bands_components_topdetail_collar_mid_strip",
                "standard_crop_ratio": float(region_standard_crop_ratio),
                "pattern": standard_pattern,
                "exts": list(image_exts),
                "db_feature_dtype": str(db_feature_dtype),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_path = _region_feature_cache_path(region_crop_recall_backend)
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

    def _reload_search_assets(reason: str = "startup") -> None:
        nonlocal names, feats, secondary_names, secondary_feats, region_names, region_feats
        nonlocal rerank_candidate_cache, label_memory_refs, scene_text_index
        t_reload = time.perf_counter()
        next_names, next_feats = build_feature_db_with_cache(
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
        logging.info("api loaded primary db: %d items", len(next_names))

        next_secondary_names: List[str] = []
        next_secondary_feats: np.ndarray | None = None
        if secondary_feature_backend and secondary_feature_backend != feature_backend:
            next_secondary_names, next_secondary_feats = build_feature_db_with_cache(
                standard_dir,
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
            next_region_names, next_region_feats = _build_region_feature_db_with_cache()
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
        logging.info("search assets reloaded: reason=%s in %.2fs", reason, time.perf_counter() - t_reload)

    _reload_search_assets("startup")
    catalog_store = CatalogStore(catalog_db_path)
    sync_stats = catalog_store.sync_from_standard_dir(standard_dir, image_exts)
    logging.info("catalog sync done: %s", sync_stats)
    catalog_import_jobs: Dict[str, Dict[str, Any]] = {}
    catalog_import_lock = threading.Lock()
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

    def _derive_year_tag_from_style_code(style_code: str) -> str:
        code = str(style_code or "").strip()
        if not code:
            return ""
        prefix = code.split("-", 1)[0].strip()
        match = re.search(r"(\d{2})$", prefix)
        if not match:
            return ""
        return f"20{match.group(1)}"

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

    def _serialize_catalog_import_job(job: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "job_id": str(job.get("job_id", "")),
            "source_dir": str(job.get("source_dir", "")),
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
        return username

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
        is_catalog_login = path == "/catalog/login"
        is_catalog_logout = path == "/catalog/logout"
        is_catalog_api = path.startswith("/api/v1/catalog/")
        is_catalog_route = is_catalog_ui or is_catalog_login or is_catalog_logout or is_catalog_api
        allow_public = (
            path in {"/health", "/ready"}
            or is_catalog_login
            or is_catalog_logout
            or ((not catalog_web_auth_enabled) and is_catalog_ui)
            or ((not catalog_web_auth_enabled) and catalog_public and is_catalog_api)
            or path.startswith("/print-static/")
            or path.startswith("/print-storage/")
            or path.startswith("/recolor-static/")
        )
        allow_api = (
            path in {"/search", "/image-url", "/api/v1/templates", "/api/v1/render", "/api/v1/images/upload", "/recolor", "/recolor-ai"}
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
            web_user = _catalog_read_session_user(request)
            if api_user:
                request.state.api_user = api_user
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
            if web_user:
                request.state.api_user = f"catalog-web:{web_user}"
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

    def _serialize_catalog_product(base_url: str, product: Dict[str, Any]) -> Dict[str, Any]:
        images = [
            {
                "image_name": str(item.get("image_name", "")),
                "sort_order": int(item.get("sort_order", 0)),
                "image_url": _build_image_url(base_url, str(item.get("image_name", ""))),
            }
            for item in list(product.get("images", []))
        ]
        cover_image = str(product.get("cover_image", "")).strip()
        return {
            "style_code": str(product.get("style_code", "")),
            "cover_image": cover_image,
            "cover_image_url": _build_image_url(base_url, cover_image) if cover_image else "",
            "note": str(product.get("note", "")),
            "tags": list(product.get("tags", [])),
            "images": images,
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
            row["catalog_cover_image_url"] = _build_image_url(base_url, str(product.get("cover_image", "")))
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
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = math.hypot(dx, dy)
            if length < min_line:
                continue
            angle = (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0
            if 20.0 <= angle <= 75.0:
                pos_len += length
            elif 105.0 <= angle <= 160.0:
                neg_len += length
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
        return float(max(0.0, min(1.0, balance * density * horizontal_penalty * dark_fill_penalty * solid_penalty)))

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
        for file_name, sig in collar_contour_cache.items():
            sim = max(float(query_sig @ sig) for query_sig in q_sigs)
            if q_mirror_sigs:
                sim = max(sim, max(float(query_sig_mirror @ sig) for query_sig_mirror in q_mirror_sigs))
            base_file_name = file_name.split("@", 1)[0]
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
        crop_mask = colorful[y0:y1, x0:x1]
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
            grid_gray.reshape(-1) * 0.20,
            grid_mask.reshape(-1) * 0.45,
            edge_y * 2.30,
            edge_x * 0.70,
            band_profile * 2.80,
            sleeve_row_profile * 4.80,
            proj_y * 1.20,
            proj_x * 0.80,
            hist_vec * 0.45,
            sleeve_structure * 7.00,
            shape_vec * 0.90,
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
        scored: List[tuple[str, float]] = []
        for file_name, sig in sleeve_pattern_cache.items():
            sim = float(query_sig @ sig)
            if sim >= sleeve_pattern_min_score:
                scored.append((file_name, sim))
        if not scored:
            return ranked, ""
        scored.sort(key=lambda x: x[1], reverse=True)
        injected = scored[: max(1, sleeve_pattern_max_injected)]
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

    def _load_or_build_sleeve_pattern_cache(file_names: List[str]) -> Dict[str, np.ndarray]:
        uniq = sorted({n.split("@", 1)[0] for n in file_names})
        files = [standard_dir / n for n in uniq if (standard_dir / n).exists() and (standard_dir / n).is_file()]
        sigs = [_local_file_sig(p) for p in files]
        cache_key = json.dumps(
            {
                "kind": "sleeve_pattern",
                "version": 5,
                "size": 32,
                "standard_views": "grid_halves_bands_components",
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
                "version": 5,
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
    if pattern_consistency_enabled:
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

    @app.get("/catalog", response_class=HTMLResponse)
    def catalog_page() -> str:
        return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>产品库</title>
  <style>
    body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin: 0; background: #f5f7fa; color: #111827; }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 20px; }
    .toolbar { display: grid; grid-template-columns: minmax(260px, 1.6fr) repeat(3, 124px); gap: 10px; margin-bottom: 14px; }
    .toolbar-secondary { display: grid; grid-template-columns: minmax(260px, 1fr) repeat(2, 124px); gap: 10px; margin-bottom: 14px; }
    input, button { font-size: 14px; padding: 8px 12px; border-radius: 10px; border: 1px solid #d1d5db; min-height: 42px; box-sizing: border-box; }
    button { cursor: pointer; background: #111827; color: #fff; border: none; }
    button.secondary { background: #fff; color: #111827; border: 1px solid #d1d5db; }
    .muted { color: #6b7280; font-size: 13px; }
    .filter-tags { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 16px; min-height: 28px; }
    .filter-tag { border: 1px solid #c7d2fe; background: #eef2ff; color: #3730a3; border-radius: 999px; padding: 6px 10px; font-size: 12px; cursor: pointer; }
    .filter-tag.active { background: #3730a3; color: #fff; border-color: #3730a3; }
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
    .card { background: #fff; border-radius: 14px; padding: 14px; box-shadow: 0 4px 18px rgba(0,0,0,0.06); }
    .thumb { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; background: #e5e7eb; border-radius: 10px; cursor: pointer; }
    .code { font-weight: 700; margin: 10px 0 8px; }
    .tags { display: flex; flex-wrap: wrap; gap: 5px; min-height: 24px; margin-bottom: 10px; }
    .tag { background: #f8fafc; color: #334155; border: 1px solid #dbe2ea; border-radius: 4px; padding: 2px 6px; font-size: 11px; display: inline-flex; align-items: center; gap: 4px; line-height: 1.1; }
    .tag-remove { min-height: auto; height: auto; padding: 0; margin: 0; border: none; background: transparent; color: #475569; cursor: pointer; font-size: 12px; line-height: 1; border-radius: 0; box-shadow: none; }
    .row { display: flex; gap: 8px; position: relative; align-items: center; }
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
    .modal-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
    .modal-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }
    .gallery-item { background: #f9fafb; border-radius: 12px; padding: 10px; }
    .gallery-item img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 10px; background: #e5e7eb; }
    .gallery-caption { margin-top: 8px; font-size: 12px; color: #4b5563; word-break: break-all; }
    .import-panel { width: min(1120px, 100%); }
    .import-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; margin-bottom: 12px; }
    .import-progress { height: 10px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin-bottom: 10px; }
    .import-progress-bar { height: 100%; width: 0%; background: linear-gradient(90deg, #4f46e5, #6366f1); transition: width 0.2s ease; }
    .import-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .import-table th, .import-table td { padding: 9px 8px; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top; }
    .import-table input[type="text"] { width: 100%; box-sizing: border-box; padding: 8px 10px; font-size: 13px; }
    .import-table tr.row-error td { background: #fff1f2; }
    .import-batch-tag-box { border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px; margin: 0 0 12px; background: #fafbfc; }
    .import-batch-tag-title { font-size: 12px; color: #475569; margin-bottom: 8px; }
    .tag-admin-box { margin-top: 8px; }
    .import-tag-list { display: flex; flex-wrap: wrap; gap: 4px; min-height: 22px; margin-bottom: 6px; }
    .import-tag-chip { display: inline-flex; align-items: center; gap: 4px; background: #f8fafc; color: #334155; border: 1px solid #dbe2ea; border-radius: 4px; padding: 2px 6px; font-size: 11px; line-height: 1.1; }
    .import-tag-remove { min-height: auto; height: auto; padding: 0; margin: 0; border: none; background: transparent; color: #64748b; cursor: pointer; font-size: 12px; line-height: 1; }
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
      .toolbar, .toolbar-secondary { grid-template-columns: 1fr 1fr; }
      .toolbar input, .toolbar-secondary input { grid-column: 1 / -1; }
      .toolbar button, .toolbar-secondary button { min-height: 40px; }
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
      <button id="syncBtn" class="secondary">同步图片</button>
      <button id="importBtn" class="secondary">目录批量导入</button>
    </div>
    <div class="toolbar-secondary">
      <div class="input-pop-wrap">
        <input id="newTagName" placeholder="新增标签名称；输入时会提示已有标签" />
        <div id="newTagSuggestPop" class="tag-suggest-pop"></div>
      </div>
      <button id="addTagBtn" class="secondary">新增标签</button>
      <button id="reloadBtn" class="secondary">刷新</button>
    </div>
    <div id="activeFilterTags" class="filter-tags"></div>
    <div id="status" class="status muted"></div>
    <div id="cards" class="cards"></div>
    <div id="loadMore" class="load-more"></div>
  </div>
  <datalist id="allTagsList"></datalist>
  <div id="galleryModal" class="modal">
    <div class="modal-panel">
      <div class="modal-head">
        <div>
          <div id="galleryTitle" class="code" style="margin:0;"></div>
          <div id="gallerySubTitle" class="muted"></div>
        </div>
        <button id="closeGalleryBtn" class="secondary">关闭</button>
      </div>
      <div id="galleryGrid" class="modal-grid"></div>
    </div>
  </div>
  <div id="importModal" class="modal">
    <div class="modal-panel import-panel">
      <div class="modal-head">
        <div>
          <div class="code" style="margin:0;">服务器目录批量导入</div>
          <div class="muted">输入服务器本地目录，先 OCR 生成候选文件名，再手工修改后导入到产品库图片目录。</div>
        </div>
        <button id="closeImportBtn" class="secondary">关闭</button>
      </div>
      <div class="import-row">
        <input id="importSourceDir" value="__CATALOG_IMPORT_SOURCE_DIR__" placeholder="例如 /data/new_samples 或 D:\\samples\\new" />
        <button id="startImportBtn">开始识别</button>
      </div>
      <div class="import-batch-tag-box">
        <div class="import-batch-tag-title">批量标签：统一加到本次勾选导入的图片所属款号</div>
        <div id="importBatchTags" class="import-tag-list"><div class="muted">未添加标签</div></div>
        <div class="import-tag-row">
          <div class="input-pop-wrap">
            <input id="importBatchTagInput" type="text" placeholder="添加标签，可选已有，也可直接新增" />
            <div id="importBatchTagSuggestPop" class="tag-suggest-pop"></div>
          </div>
          <button id="importBatchTagAddBtn" type="button" class="secondary import-tag-add-btn">添加</button>
        </div>
      </div>
      <div class="import-progress"><div id="importProgressBar" class="import-progress-bar"></div></div>
      <div id="importMeta" class="muted" style="margin-bottom:10px;"></div>
      <div class="import-table-wrap">
        <table class="import-table">
          <thead>
            <tr>
              <th style="width:52px;">导入</th>
              <th>源文件</th>
              <th style="width:150px;">识别款号</th>
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
        <button id="closeImportPreviewBtn" class="secondary">关闭</button>
      </div>
      <img id="importPreviewImg" class="import-preview-img" alt="source preview" />
    </div>
  </div>
  <script>
    let globalTags = [];
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
    let importBatchTags = [];
    const els = {
      styleCodeQuery: document.getElementById('styleCodeQuery'),
      searchBtn: document.getElementById('searchBtn'),
      syncBtn: document.getElementById('syncBtn'),
      importBtn: document.getElementById('importBtn'),
      newTagName: document.getElementById('newTagName'),
      newTagSuggestPop: document.getElementById('newTagSuggestPop'),
      addTagBtn: document.getElementById('addTagBtn'),
      reloadBtn: document.getElementById('reloadBtn'),
      status: document.getElementById('status'),
      cards: document.getElementById('cards'),
      loadMore: document.getElementById('loadMore'),
      activeFilterTags: document.getElementById('activeFilterTags'),
      allTagsList: document.getElementById('allTagsList'),
      galleryModal: document.getElementById('galleryModal'),
      galleryTitle: document.getElementById('galleryTitle'),
      gallerySubTitle: document.getElementById('gallerySubTitle'),
      galleryGrid: document.getElementById('galleryGrid'),
      closeGalleryBtn: document.getElementById('closeGalleryBtn'),
      importModal: document.getElementById('importModal'),
      closeImportBtn: document.getElementById('closeImportBtn'),
      importSourceDir: document.getElementById('importSourceDir'),
      startImportBtn: document.getElementById('startImportBtn'),
      importProgressBar: document.getElementById('importProgressBar'),
      importMeta: document.getElementById('importMeta'),
      importBatchTags: document.getElementById('importBatchTags'),
      importBatchTagInput: document.getElementById('importBatchTagInput'),
      importBatchTagSuggestPop: document.getElementById('importBatchTagSuggestPop'),
      importBatchTagAddBtn: document.getElementById('importBatchTagAddBtn'),
      importTableBody: document.getElementById('importTableBody'),
      commitImportBtn: document.getElementById('commitImportBtn'),
      importCommitStatus: document.getElementById('importCommitStatus'),
      importPreviewModal: document.getElementById('importPreviewModal'),
      importPreviewTitle: document.getElementById('importPreviewTitle'),
      importPreviewSubTitle: document.getElementById('importPreviewSubTitle'),
      importPreviewImg: document.getElementById('importPreviewImg'),
      closeImportPreviewBtn: document.getElementById('closeImportPreviewBtn'),
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

    function uniqTags(tags) {
      return Array.from(new Set((tags || []).filter(Boolean)));
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
      els.allTagsList.innerHTML = globalTags.map(tag => `<option value="${tag}"></option>`).join('');
      renderGlobalTagAdminLists();
    }

    async function deleteGlobalTag(tag) {
      const resp = await fetch('/api/v1/catalog/tags/' + encodeURIComponent(tag), { method: 'DELETE' });
      if (!resp.ok) throw new Error(await resp.text());
    }

    function renderGlobalTagAdminLists() {
      renderTagSuggestPopover(els.newTagSuggestPop, els.newTagName, true);
      renderTagSuggestPopover(els.importBatchTagSuggestPop, els.importBatchTagInput, true);
    }

    function closeTagSuggestPops() {
      [els.newTagSuggestPop, els.importBatchTagSuggestPop].forEach((pop) => {
        if (pop) pop.classList.remove('open');
      });
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
          const ok = window.confirm(`确认删除标签“${tag}”吗？\n\n这会同步删除所有产品与该标签的关联，且不可撤销。`);
          if (!ok) return;
          try {
            await deleteGlobalTag(tag);
            importBatchTags = normalizeImportTags(importBatchTags.filter(x => String(x) !== tag));
            renderImportBatchTags();
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
      if (els.styleCodeQuery.value.trim()) params.set('style_code', els.styleCodeQuery.value.trim());
      const tags = selectedFilterTags;
      if (tags.length) params.set('tags', tags.join(','));
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
      const all = uniqTags(globalTags);
      els.activeFilterTags.innerHTML = all.map(tag => `
        <button type="button" class="filter-tag ${selectedFilterTags.includes(tag) ? 'active' : ''}" data-role="filterTagBtn" data-tag="${tag}">
          ${tag}
        </button>
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

    function openGallery(product) {
      els.galleryTitle.textContent = product.style_code || '';
      els.gallerySubTitle.textContent = `共 ${(product.images || []).length} 张图片`;
      els.galleryGrid.innerHTML = (product.images || []).map(item => `
        <div class="gallery-item">
          <img src="${item.image_url || ''}" loading="lazy" alt="${item.image_name || ''}" />
          <div class="gallery-caption">${item.image_name || ''}</div>
        </div>
      `).join('');
      els.galleryModal.classList.add('open');
    }

    function closeGallery() {
      els.galleryModal.classList.remove('open');
    }

    function openImportModal() {
      importBatchTags = [];
      renderImportBatchTags();
      if (els.importBatchTagInput) els.importBatchTagInput.value = '';
      els.importModal.classList.add('open');
    }

    function closeImportModal() {
      els.importModal.classList.remove('open');
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
      const stem = raw.replace(/\.[^.]+$/, '');
      const styleCode = stem.includes('_') ? stem.slice(0, stem.lastIndexOf('_')) : stem;
      const prefix = styleCode.split('-', 1)[0] || '';
      const match = prefix.match(/(\d{2})$/);
      return match ? `20${match[1]}` : '';
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

    function buildImportTagListHtml(tags) {
      const list = normalizeImportTags(tags);
      if (!list.length) return '<div class="muted">未添加标签</div>';
      return list.map(tag => `
        <span class="import-tag-chip">
          <span>${tag}</span>
          <button type="button" class="import-tag-remove" data-role="importRemoveTagBtn" data-tag="${tag}" title="删除标签">×</button>
        </span>
      `).join('');
    }

    function renderImportBatchTags() {
      if (!els.importBatchTags) return;
      els.importBatchTags.innerHTML = buildImportTagListHtml(importBatchTags);
      els.importBatchTags.querySelectorAll('[data-role="importRemoveTagBtn"]').forEach((button) => {
        button.addEventListener('click', () => {
          const tag = String(button.dataset.tag || '').trim();
          importBatchTags = normalizeImportTags(importBatchTags.filter(x => String(x) !== tag));
          renderImportBatchTags();
        });
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
            <td><input type="checkbox" data-role="importSelect" data-index="${index}" ${item.selected === false ? '' : 'checked'} /></td>
            <td>
              <button type="button" class="import-source-link" data-role="importPreviewBtn" data-index="${index}">${item.source_name || ''}</button>
              <div class="muted">${item.source_rel_path || ''}</div>
            </td>
            <td>${item.proposed_style_code || '-'}</td>
            <td><input type="text" data-role="importYearTag" data-index="${index}" value="${item.year_tag || item.proposed_year_tag || ''}" placeholder="如 2024" /></td>
            <td><input type="text" data-role="importFilename" data-index="${index}" value="${item.target_filename || item.proposed_filename || ''}" /></td>
            <td><span class="import-badge ${item.status === 'ok' ? 'ok' : 'warn'}">${item.status === 'ok' ? '已识别' : '需人工确认'}</span>${item.error ? `<div class="muted" style="margin-top:4px;color:#b91c1c;">${item.error}</div>` : ''}</td>
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
        const yearInput = els.importTableBody.querySelector(`[data-role="importYearTag"][data-index="${index}"]`);
        return {
          source_rel_path: item.source_rel_path,
          selected: !!(checkbox && checkbox.checked),
          tags: normalizeImportTags(importBatchTags),
          year_tag: yearInput ? yearInput.value.trim() : (item.year_tag || item.proposed_year_tag || ''),
          target_filename: input ? input.value.trim() : (item.target_filename || item.proposed_filename || ''),
        };
      });
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
        const tags = item.tags || [];
        const tagsHtml = buildCardTagsHtml(tags);
        card.innerHTML = `
          <img class="thumb" src="${item.cover_image_url || ''}" loading="lazy" alt="${item.style_code}" title="点击查看该款全部图片" />
          <div class="code">${item.style_code}</div>
          <div class="tags">${tagsHtml}</div>
          <div class="muted" style="margin-bottom:10px;">图片数：${(item.images || []).length}</div>
          <div class="row">
            <div class="picker-trigger" data-role="pickerTrigger">点击选择已有标签</div>
            <div class="picker-pop" data-role="pickerPop">
              <div class="picker-options" data-role="pickerOptions">${buildPickerOptions([])}</div>
            </div>
            <button type="button" class="picker-add-btn" data-role="saveBtn">添加</button>
          </div>
        `;
        card.querySelector('.thumb').addEventListener('click', () => openGallery(item));
        let pendingTags = [];
        const pickerTrigger = card.querySelector('[data-role="pickerTrigger"]');
        const pickerPop = card.querySelector('[data-role="pickerPop"]');
        const pickerOptions = card.querySelector('[data-role="pickerOptions"]');
        updatePickerTrigger(pickerTrigger, pendingTags);
        pickerTrigger.addEventListener('click', (event) => {
          event.stopPropagation();
          const isOpen = pickerPop.classList.contains('open');
          closeAllPickers();
          if (!isOpen) pickerPop.classList.add('open');
        });
        pickerOptions.querySelectorAll('[data-role="pickerOption"]').forEach((button) => {
          button.addEventListener('click', (event) => {
            event.stopPropagation();
            const tag = button.dataset.tag || '';
            if (!tag) return;
            if (pendingTags.includes(tag)) {
              pendingTags = pendingTags.filter(x => x !== tag);
            } else {
              pendingTags = uniqTags([...pendingTags, tag]);
            }
            const item = button.closest('.picker-option-item');
            if (item) item.classList.toggle('active', pendingTags.includes(tag));
            updatePickerTrigger(pickerTrigger, pendingTags);
          });
        });
        pickerOptions.querySelectorAll('[data-role="pickerDeleteOption"]').forEach((button) => {
          button.addEventListener('click', async (event) => {
            event.stopPropagation();
            const tag = String(button.dataset.tag || '').trim();
            if (!tag) return;
            const ok = window.confirm(`确认删除标签“${tag}”吗？\n\n这会同步删除所有产品与该标签的关联，且不可撤销。`);
            if (!ok) return;
            try {
              await deleteGlobalTag(tag);
              pendingTags = pendingTags.filter(x => x !== tag);
              importBatchTags = normalizeImportTags(importBatchTags.filter(x => String(x) !== tag));
              renderImportBatchTags();
              await loadGlobalTags();
              await loadProducts(true);
              setStatus('标签已删除', false);
            } catch (err) {
              setStatus(err.message || '删除标签失败', true);
            }
          });
        });
        card.querySelectorAll('[data-role="filterFromCardBtn"]').forEach((button) => {
          button.addEventListener('click', () => toggleFilterTag(button.dataset.tag || ''));
        });
        card.querySelectorAll('[data-role="removeTagBtn"]').forEach((button) => {
          button.addEventListener('click', async () => {
            const tag = button.dataset.tag || '';
            const nextTags = tags.filter(x => x !== tag);
            try {
              await saveTags(item.style_code, nextTags);
              await loadProducts(true);
              setStatus('标签已删除', false);
            } catch (err) {
              setStatus(err.message || '删除失败', true);
            }
          });
        });
        card.querySelector('[data-role="saveBtn"]').addEventListener('click', async () => {
          if (!pendingTags.length) {
            setStatus('请先选择至少一个已有标签', true);
            return;
          }
          const nextTags = uniqTags([...(item.tags || []), ...pendingTags]);
          try {
            await saveTags(item.style_code, nextTags);
            pendingTags = [];
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
    [els.newTagName, els.importBatchTagInput].forEach((input) => {
      if (!input) return;
      input.addEventListener('focus', () => {
        const pop = input === els.newTagName ? els.newTagSuggestPop : els.importBatchTagSuggestPop;
        closeTagSuggestPops();
        pop.dataset.activeIndex = '0';
        renderTagSuggestPopover(pop, input, true);
        if (pop) pop.classList.add('open');
      });
      input.addEventListener('input', () => {
        const pop = input === els.newTagName ? els.newTagSuggestPop : els.importBatchTagSuggestPop;
        pop.dataset.activeIndex = '0';
        renderTagSuggestPopover(pop, input, true);
        if (pop) pop.classList.add('open');
      });
      input.addEventListener('keydown', (event) => {
        const pop = input === els.newTagName ? els.newTagSuggestPop : els.importBatchTagSuggestPop;
        if (event.key === 'ArrowDown') {
          event.preventDefault();
          moveTagSuggestActive(pop, input, 1);
          return;
        }
        if (event.key === 'ArrowUp') {
          event.preventDefault();
          moveTagSuggestActive(pop, input, -1);
          return;
        }
        if (event.key === 'Escape') {
          if (pop) pop.classList.remove('open');
          return;
        }
        if (event.key === 'Enter' && pop && pop.classList.contains('open')) {
          const picked = pickActiveTagSuggest(pop, input);
          if (picked) {
            event.preventDefault();
            return;
          }
        }
      });
      input.addEventListener('click', (event) => {
        event.stopPropagation();
      });
    });
    els.reloadBtn.addEventListener('click', () => Promise.all([loadGlobalTags(), loadProducts(true)]).catch(err => setStatus(err.message || '加载失败', true)));
    els.addTagBtn.addEventListener('click', async () => {
      try {
        const value = els.newTagName.value.trim();
        if (!value) {
          setStatus('请输入标签名称', true);
          return;
        }
        const resp = await fetch('/api/v1/catalog/tags', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: value })
        });
        if (!resp.ok) throw new Error(await resp.text());
        els.newTagName.value = '';
        await loadGlobalTags();
        setStatus('标签已新增', false);
      } catch (err) {
        setStatus(err.message || '新增标签失败', true);
      }
    });
    els.syncBtn.addEventListener('click', async () => {
      try {
        const resp = await fetch('/api/v1/catalog/sync', { method: 'POST' });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        await loadGlobalTags();
        await loadProducts(true);
        setStatus(`同步完成：新增款 ${data.products_added}，新增/更新图 ${data.images_added_or_updated}`, false);
      } catch (err) {
        setStatus(err.message || '同步失败', true);
      }
    });
    els.importBtn.addEventListener('click', openImportModal);
    els.importBatchTagAddBtn.addEventListener('click', () => {
      const value = els.importBatchTagInput ? String(els.importBatchTagInput.value || '').trim() : '';
      if (!value) return;
      importBatchTags = normalizeImportTags([...(importBatchTags || []), value]);
      updateGlobalTagOptions(value);
      if (els.importBatchTagInput) els.importBatchTagInput.value = '';
      renderImportBatchTags();
    });
    els.importBatchTagInput.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter') return;
      event.preventDefault();
      els.importBatchTagAddBtn.click();
    });
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
        if (els.importTableBody) els.importTableBody.innerHTML = '<tr><td colspan="5" class="muted">任务创建中...</td></tr>';
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
</html>""".replace("__CATALOG_IMPORT_SOURCE_DIR__", html_escape(catalog_import_source_dir, quote=True))

    @app.get("/api/v1/catalog/products")
    def api_list_catalog_products(
        request: Request,
        style_code: str = "",
        tags: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        base_url = _external_base_url(request)
        tag_list = [item.strip() for item in tags.split(",") if item.strip()]
        products = catalog_store.list_products(style_code=style_code, tags=tag_list, limit=limit, offset=offset)
        return {"products": [_serialize_catalog_product(base_url, item) for item in products]}

    @app.get("/api/v1/catalog/products/{style_code}")
    def api_get_catalog_product(request: Request, style_code: str) -> Dict[str, Any]:
        product = catalog_store.get_product(style_code)
        if not product:
            raise HTTPException(status_code=404, detail="product not found")
        return _serialize_catalog_product(_external_base_url(request), product)

    @app.put("/api/v1/catalog/products/{style_code}/tags")
    def api_replace_catalog_product_tags(style_code: str, payload: CatalogTagUpdateRequest) -> Dict[str, Any]:
        try:
            tags_local = catalog_store.replace_product_tags(style_code, payload.tags)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"style_code": style_code, "tags": tags_local}

    @app.get("/api/v1/catalog/tags")
    def api_list_catalog_tags() -> Dict[str, Any]:
        return {"tags": catalog_store.list_tags()}

    @app.post("/api/v1/catalog/tags")
    def api_create_catalog_tag(payload: CatalogTagCreateRequest) -> Dict[str, Any]:
        try:
            tag = catalog_store.create_tag(payload.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"tag": tag}

    @app.delete("/api/v1/catalog/tags/{tag_name}")
    def api_delete_catalog_tag(tag_name: str) -> Dict[str, Any]:
        try:
            tag = catalog_store.delete_tag(tag_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"tag": tag, "deleted": True}

    @app.post("/api/v1/catalog/sync")
    def api_sync_catalog() -> Dict[str, Any]:
        return catalog_store.sync_from_standard_dir(standard_dir, image_exts)

    @app.post("/api/v1/catalog/imports/prepare")
    def api_prepare_catalog_import(payload: CatalogImportPrepareRequest) -> Dict[str, Any]:
        source_dir_raw = str(payload.source_dir or "").strip() or catalog_import_source_dir
        if not source_dir_raw:
            raise HTTPException(status_code=400, detail="source_dir is empty; set catalog.import_source_dir in config or input it manually")
        source_dir = _resolve_catalog_import_source_dir(source_dir_raw)
        if not source_dir.exists() or not source_dir.is_dir():
            raise HTTPException(status_code=400, detail="source_dir not found")
        files = _list_import_source_images(source_dir)
        if not files:
            raise HTTPException(status_code=400, detail="source_dir has no supported images")
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "source_dir": str(source_dir),
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
        return _serialize_catalog_import_job(job)

    @app.get("/api/v1/catalog/imports/{job_id}")
    def api_get_catalog_import_job(job_id: str) -> Dict[str, Any]:
        with catalog_import_lock:
            job = catalog_import_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="import job not found")
            return _serialize_catalog_import_job(job)

    @app.get("/api/v1/catalog/imports/{job_id}/source-image")
    def api_get_catalog_import_source_image(job_id: str, source_rel_path: str, max_edge: int = 0, q: int = 82) -> FileResponse:
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
            return FileResponse(path=str(out_fp), media_type="image/jpeg")
        return FileResponse(path=str(fp))

    @app.post("/api/v1/catalog/imports/commit")
    def api_commit_catalog_import(payload: CatalogImportCommitRequest) -> Dict[str, Any]:
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

        planned: List[tuple[Path, str]] = []
        seen_targets: set[str] = set()
        style_year_tags: Dict[str, set[str]] = {}
        style_extra_tags: Dict[str, set[str]] = {}
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
                target_name = _sanitize_import_filename(raw_target, src.suffix.lower())
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{item.source_rel_path}: {exc}") from exc
            if target_name.lower() in seen_targets:
                raise HTTPException(status_code=400, detail=f"duplicate target filename: {target_name}")
            if (standard_dir / target_name).exists():
                raise HTTPException(status_code=400, detail=f"target filename already exists: {target_name}")
            seen_targets.add(target_name.lower())
            try:
                year_tag = _sanitize_year_tag(item.year_tag.strip() or str(prepared.get("proposed_year_tag", "")).strip())
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{item.source_rel_path}: {exc}") from exc
            style_code = filename_to_style_code(target_name).strip()
            if year_tag and style_code:
                style_year_tags.setdefault(style_code, set()).add(year_tag)
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
        _reload_search_assets("catalog_import_commit")
        with catalog_import_lock:
            job = catalog_import_jobs.get(payload.job_id)
            if job is not None:
                job["committed"] = True
                job["message"] = f"已导入 {imported} 张图片"
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
        crop_x: float = Form(0.0),
        crop_y: float = Form(0.0),
        crop_w: float = Form(0.0),
        crop_h: float = Form(0.0),
        match_mode: str = Form(""),
    ) -> Dict[str, Any]:
        t_all = time.perf_counter()
        if not file.filename:
            raise HTTPException(status_code=400, detail="missing file name")
        suffix = Path(file.filename).suffix.lower()
        if suffix.lstrip(".") not in {"png", "jpg", "jpeg"}:
            raise HTTPException(status_code=400, detail="only png/jpg/jpeg supported")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tf:
            upload_bytes = await file.read()
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

            query_hint_code = try_extract_query_style_code(query_path) if ocr_hint_enabled else ""
            scene_text_tokens: List[str] = []
            checker_debug = ""
            checker_candidates_debug = ""
            accent_debug = ""
            accent_candidates_debug = ""
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
            region_repeat_force_scores: Dict[str, tuple[float, int]] = {}
            region_boost_debug = ""
            region_rescue_debug = ""
            region_order_debug = ""
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
                return re.sub(r"[^A-Za-z0-9_-]+", "", str(code).strip().upper())

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
                rescue_codes = [
                    code
                    for code, score in sorted(region_code_scores.items(), key=lambda item: item[1], reverse=True)[: max(1, region_crop_result_rescue_topn)]
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
                        rescue_rows.append(dict(row))
                        continue
                    fallback_row = _fallback_rescue_row(key)
                    if fallback_row is not None:
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
                keep_n = max(0, top_k - len(rescue_rows))
                return (kept_rows[:keep_n] + rescue_rows)[:top_k]

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
                if region_crop_repeat_force_enabled and region_repeat_force_scores:
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
                return (forced_rows + kept_rows)[:top_k]

            def _order_region_primary_rows(rows_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                nonlocal region_order_debug
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
                protected_rows = [row for row in rows_in if row.get("_force_keep")]
                protected_keys = {_code_prior_key(str(row.get("style_code", ""))) for row in protected_rows}
                sortable_rows = [
                    row
                    for row in rows_in
                    if _code_prior_key(str(row.get("style_code", ""))) not in protected_keys
                ]
                ordered = sorted(
                    sortable_rows,
                    key=lambda row: (
                        float(region_code_scores.get(str(row.get("style_code", "")), -1.0)),
                        float(row.get("rank_score", 0.0)),
                    ),
                    reverse=True,
                )
                ordered = (protected_rows + ordered)[:top_k]
                region_order_debug = ",".join(
                    f"{row.get('style_code', '')}:{float(region_code_scores.get(str(row.get('style_code', '')), -1.0)):.3f}"
                    for row in ordered[:top_k]
                )
                return ordered[:top_k]

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
                    and not strict_small_region_crop
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
                    if sim < region_crop_sleeve_rescue_min_sim or pair_prior < region_crop_sleeve_rescue_min_pair_prior:
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
                    and not strict_small_region_crop
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
                    if sim < region_crop_sleeve_rescue_min_sim or pair_prior < region_crop_sleeve_rescue_min_pair_prior:
                        continue
                    ranked_item = best_ranked_by_key.get(key)
                    if ranked_item is None:
                        continue
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
                        }
                    )
                    seen_keys.add(key)
                if not sleeve_rows:
                    return rows_in
                sleeve_rows.sort(key=lambda row: float(row.get("rank_score", 0.0)), reverse=True)
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
                    and not strict_small_region_crop
                    and not partial_region_crop
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
                    and not strict_small_region_crop
                    and active_match_mode == "similar_style"
                    and search_scope == "region_primary"
                    and scene_text_tokens
                    and isinstance(scene_text_index, dict)
                    and rows_in
                ):
                    return rows_in
                if region_best_score >= float(scene_text_suppress_when_region_min_score):
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
            if pattern_consistency_enabled:
                q_pattern_sig = _extract_pattern_sig(query_path, size=14)
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
            accent_pattern_allowed = (
                accent_pattern_enabled
                and not strict_small_region_crop
                and not partial_region_crop
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
            checker_blocked_by_region_probe = bool(
                auto_region_probe_active
                and region_best_score >= float(scene_text_suppress_when_region_min_score)
            )
            if (
                checker_consistency_enabled
                and not strict_small_region_crop
                and not checker_large_crop_blocked
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
                            q_collar_sig_mirrors = _extract_collar_contour_sigs_from_image(
                                ImageOps.mirror(q_img),
                                size=collar_contour_size,
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
                            region_score = float(collar_contour_region_score_base) + float(collar_contour_region_score_scale) * max(
                                0.0,
                                min(1.0, float(sim)),
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
                        collar_match_keys = {_code_prior_key(code) for code in collar_code_matches}
                        for row in rows:
                            if (
                                _code_prior_key(str(row.get("style_code", ""))) in collar_match_keys
                                and float(row.get("rank_score", 0.0)) >= float(collar_contour_force_keep_min_rank)
                            ):
                                row["_force_keep"] = True
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
            suppress_sleeve_for_small_region_query = bool(strict_small_region_crop)
            suppress_sleeve_for_large_region_query = bool(
                crop_active
                and crop_final_area > max(0.0, min(1.0, float(sleeve_pattern_crop_max_area)))
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
            scene_text_blocked_by_region = bool(
                region_probe_active
                and (search_scope == "region_primary" or auto_region_probe_active)
                and region_best_score >= float(scene_text_suppress_when_region_min_score)
            )
            if scene_text_hint_enabled and not strict_small_region_crop and not scene_text_blocked_by_region:
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
