import argparse
import json
import logging
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

# Silence known transformers/torch deprecation warnings that do not affect runtime results.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.utils\._pytree\._register_pytree_node is deprecated.*",
    category=UserWarning,
)

from clip_features import extract_feature_clip
from features import extract_feature, extract_garment_color_feature, extract_stripe_feature
from reranker import LinearReranker, extract_modal_features, pair_features
DEFAULT_CONFIG = Path("config/search_config.json")
CODE_NORM_RE = re.compile(r"[^A-Za-z0-9_-]+")


def collect_images(base: Path, pattern: str, exts: List[str]) -> List[Path]:
    allow = {e.lower().lstrip(".") for e in exts}
    out: List[Path] = []
    for p in sorted(base.glob(pattern)):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") in allow:
            out.append(p)
    if out:
        return out
    # fallback: if pattern misses files, scan directory and keep only allowed image extensions
    for p in sorted(base.glob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") in allow and p not in out:
            out.append(p)
    return out


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-8
    return (x / n).astype(np.float32)


def extract_embedding(
    image: Image.Image,
    backend: str,
    w_clip: float,
    w_shape: float,
    w_color: float,
    w_stripe: float,
) -> np.ndarray:
    if backend == "clip":
        return extract_feature_clip(image)
    if backend in ("classic", "handcrafted"):
        return extract_feature(image)
    if backend == "hybrid":
        f_clip = _l2norm(extract_feature_clip(image))
        f_shape = _l2norm(extract_feature(image))
        f_color = _l2norm(extract_garment_color_feature(image))
        f_stripe = _l2norm(extract_stripe_feature(image))
        feat = np.concatenate(
            [w_clip * f_clip, w_shape * f_shape, w_color * f_color, w_stripe * f_stripe]
        ).astype(np.float32)
        return _l2norm(feat)
    raise ValueError(f"unsupported feature backend: {backend}")


def build_feature_db(
    standard_dir: Path,
    pattern: str,
    backend: str,
    exts: List[str],
    w_clip: float,
    w_shape: float,
    w_color: float,
    w_stripe: float,
    standard_multicrop: bool = True,
    standard_crop_ratio: float = 0.72,
) -> Tuple[List[str], np.ndarray]:
    files = collect_images(standard_dir, pattern, exts)
    if not files:
        raise RuntimeError(f"no standard images found in {standard_dir}")

    names = []
    feats = []
    for p in files:
        img = Image.open(p).convert("RGB")
        views = _multi_crop_views(img, crop_ratio=standard_crop_ratio) if standard_multicrop else [img]
        for i, v in enumerate(views):
            names.append(f"{p.name}@c{i}")
            feats.append(extract_embedding(v, backend, w_clip, w_shape, w_color, w_stripe))
    return names, np.vstack(feats).astype(np.float32)


def _feature_cache_path(standard_dir: Path, backend: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(standard_dir))
    return Path("outputs") / f"feature_cache_{backend}_{safe}.npz"


def _file_sig(p: Path) -> str:
    st = p.stat()
    return f"{p.name}|{st.st_size}|{int(st.st_mtime)}"


def build_feature_db_with_cache(
    standard_dir: Path,
    pattern: str,
    backend: str,
    exts: List[str],
    w_clip: float,
    w_shape: float,
    w_color: float,
    w_stripe: float,
    standard_multicrop: bool = True,
    standard_crop_ratio: float = 0.72,
    use_cache: bool = True,
    db_feature_dtype: str = "float32",
) -> Tuple[List[str], np.ndarray]:
    files = collect_images(standard_dir, pattern, exts)
    if not files:
        raise RuntimeError(f"no standard images found in {standard_dir}")
    sigs = [_file_sig(p) for p in files]
    cache_path = _feature_cache_path(standard_dir, backend)
    cache_key = json.dumps(
        {
            "backend": backend,
            "weights": [w_clip, w_shape, w_color, w_stripe],
            "standard_multicrop": bool(standard_multicrop),
            "standard_crop_ratio": float(standard_crop_ratio),
            "pattern": pattern,
            "exts": list(exts),
            "db_feature_dtype": str(db_feature_dtype),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if use_cache and cache_path.exists():
        try:
            arr = np.load(cache_path, allow_pickle=True)
            if (
                str(arr["cache_key"].item()) == cache_key
                and list(arr["file_sigs"]) == sigs
            ):
                names = [str(x) for x in arr["names"]]
                if str(db_feature_dtype).lower() == "float16":
                    feats = arr["feats"].astype(np.float16)
                else:
                    feats = arr["feats"].astype(np.float32)
                logging.info("feature cache hit: %s (%d items)", cache_path, len(names))
                return names, feats
        except Exception:
            pass

    t0 = time.perf_counter()
    names, feats = build_feature_db(
        standard_dir,
        pattern,
        backend,
        exts,
        w_clip,
        w_shape,
        w_color,
        w_stripe,
        standard_multicrop=standard_multicrop,
        standard_crop_ratio=standard_crop_ratio,
    )
    if str(db_feature_dtype).lower() == "float16":
        feats = feats.astype(np.float16)
    else:
        feats = feats.astype(np.float32)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            cache_key=np.array([cache_key], dtype=object),
            file_sigs=np.array(sigs, dtype=object),
            names=np.array(names, dtype=object),
            feats=feats,
        )
        logging.info("feature cache write: %s", cache_path)
    logging.info("build feature db done: %d items in %.2fs", len(names), time.perf_counter() - t0)
    return names, feats


def _multi_crop_views(img: Image.Image, crop_ratio: float = 0.75) -> List[Image.Image]:
    w, h = img.size
    cw = max(32, int(w * crop_ratio))
    ch = max(32, int(h * crop_ratio))

    x1 = max(0, w - cw)
    y1 = max(0, h - ch)
    xc = max(0, (w - cw) // 2)
    yc = max(0, (h - ch) // 2)
    boxes = [
        (0, 0, cw, ch),
        (x1, 0, x1 + cw, ch),
        (0, y1, cw, y1 + ch),
        (x1, y1, x1 + cw, y1 + ch),
        (xc, yc, xc + cw, yc + ch),
    ]

    views = [img]
    seen = {(0, 0, w, h)}
    for b in boxes:
        if b in seen:
            continue
        seen.add(b)
        views.append(img.crop(b))
    return views


def _foreground_component_views(
    img: Image.Image,
    max_components: int = 3,
    min_area_ratio: float = 0.02,
) -> List[Image.Image]:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    # gray floor background suppression
    bg = (arr[..., 0] > 225) & (arr[..., 1] > 225) & (arr[..., 2] > 225)
    # keep colorful / dark foreground
    fg = (~bg).astype(np.uint8)
    # suppress the top red label area a bit
    cut = min(int(h * 0.12), 120)
    fg[:cut, :] = 0
    if fg.sum() < 32:
        return []

    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    views: List[Image.Image] = []
    min_area = int(h * w * min_area_ratio)
    comps: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < min_area:
            continue
        comps.append((int(area), (int(x), int(y), int(ww), int(hh))))
    comps.sort(key=lambda t: t[0], reverse=True)
    for _, (x, y, ww, hh) in comps[:max_components]:
        pad_x = max(8, int(ww * 0.08))
        pad_y = max(8, int(hh * 0.08))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + ww + pad_x)
        y1 = min(h, y + hh + pad_y)
        if x1 - x0 >= 24 and y1 - y0 >= 24:
            views.append(img.crop((x0, y0, x1, y1)))
    return views


def _build_query_views(
    q_img: Image.Image,
    query_multicrop: bool,
    query_crop_ratio: float,
    component_views: bool,
) -> List[Image.Image]:
    views = [q_img]
    if query_multicrop:
        views.extend(_multi_crop_views(q_img, crop_ratio=query_crop_ratio))
    if component_views:
        views.extend(_foreground_component_views(q_img))
    uniq: List[Image.Image] = []
    seen = set()
    for v in views:
        key = (v.size[0], v.size[1], int(np.asarray(v).mean()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(v)
    return uniq


def search_topk_images(
    query_img: Path,
    names: List[str],
    feats: np.ndarray,
    top_k: int,
    backend: str,
    w_clip: float,
    w_shape: float,
    w_color: float,
    w_stripe: float,
    query_multicrop: bool = True,
    query_crop_ratio: float = 0.75,
    query_component_views: bool = True,
) -> List[Tuple[str, float]]:
    q_img = Image.open(query_img).convert("RGB")
    q_views = _build_query_views(
        q_img,
        query_multicrop=query_multicrop,
        query_crop_ratio=query_crop_ratio,
        component_views=query_component_views,
    )
    q_feats = [
        extract_embedding(v, backend, w_clip, w_shape, w_color, w_stripe)
        for v in q_views
    ]
    sims_stack = np.vstack([(feats @ q).astype(np.float32) for q in q_feats])
    sims = sims_stack.max(axis=0)
    k = min(top_k, len(names))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(names[i], float(sims[i])) for i in idx]


def filename_to_style_code(img_name: str) -> str:
    base = img_name.split("@", 1)[0]
    stem = Path(base).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def display_image_name(img_name: str) -> str:
    return img_name.split("@", 1)[0]


def _normalize_style_code(code: str) -> str:
    c = code.strip().upper()
    if c.endswith("#"):
        c = c[:-1]
    c = CODE_NORM_RE.sub("", c)
    return c


def try_extract_query_style_code(query_img: Path) -> str:
    try:
        from extract_style_codes import build_header_crops, try_extract_code_from_image
    except Exception:
        return ""
    try:
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        try:
            # Suppress verbose OCR debug/info logs from helper module.
            root_logger.setLevel(max(prev_level, logging.WARNING))
            for crop in build_header_crops(query_img):
                code = try_extract_code_from_image(crop, tesseract_bin=None)
                if code:
                    return _normalize_style_code(code)
        finally:
            root_logger.setLevel(prev_level)
    except Exception:
        return ""
    return ""


def topk_style_codes(
    ranked_images: List[Tuple[str, float]],
    top_k_codes: int,
    min_score: float = 0.0,
    code_agg_top_n: int = 3,
    code_agg_alpha: float = 0.7,
    query_hint_code: str = "",
    query_hint_boost: float = 0.0,
    code_prior_boost: Dict[str, float] | None = None,
) -> List[Dict[str, object]]:
    by_code: Dict[str, List[Tuple[str, float]]] = {}
    for img_name, score in ranked_images:
        code = filename_to_style_code(img_name)
        by_code.setdefault(code, []).append((img_name, score))

    rows: List[Dict[str, object]] = []
    for code, items in by_code.items():
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        best_img, best_score = items_sorted[0]
        topn = items_sorted[: max(1, code_agg_top_n)]
        mean_topn = float(np.mean([s for _, s in topn]))
        agg_score = code_agg_alpha * float(best_score) + (1.0 - code_agg_alpha) * mean_topn
        code_norm = _normalize_style_code(code)
        if query_hint_code and code_norm == query_hint_code:
            agg_score += query_hint_boost
        if code_prior_boost:
            agg_score += float(code_prior_boost.get(code_norm, 0.0))
        rows.append(
            {
                "style_code": code,
                "best_standard_image": display_image_name(best_img),
                "score": round(agg_score, 4),
            }
        )

    rows.sort(key=lambda x: float(x["score"]), reverse=True)
    filtered = [x for x in rows if float(x["score"]) >= min_score]
    return filtered[:top_k_codes]


def build_label_memory_prior(
    query_img: Path,
    labels_path: Path,
    sim_threshold: float = 0.90,
    max_boost: float = 0.08,
) -> Dict[str, float]:
    if not labels_path.exists():
        return {}
    try:
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        labels = data.get("labels", [])
    except Exception:
        return {}
    if not labels:
        return {}

    try:
        q = extract_feature_clip(Image.open(query_img).convert("RGB"))
    except Exception:
        return {}

    prior: Dict[str, float] = {}
    for row in labels:
        p = Path(str(row.get("query_image", "")))
        code = _normalize_style_code(str(row.get("style_code", "")))
        if not p.exists() or not code:
            continue
        try:
            r = extract_feature_clip(Image.open(p).convert("RGB"))
        except Exception:
            continue
        sim = float(q @ r)
        if sim >= sim_threshold:
            # linear ramp above threshold
            t = min(1.0, (sim - sim_threshold) / max(1e-6, 1.0 - sim_threshold))
            boost = max_boost * t
            prior[code] = max(prior.get(code, 0.0), boost)
    return prior


def precompute_label_memory_refs(labels_path: Path) -> List[Tuple[str, np.ndarray]]:
    if not labels_path.exists():
        return []
    try:
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        labels = data.get("labels", [])
    except Exception:
        return []
    refs: List[Tuple[str, np.ndarray]] = []
    for row in labels:
        p = Path(str(row.get("query_image", "")))
        code = _normalize_style_code(str(row.get("style_code", "")))
        if not p.exists() or not code:
            continue
        try:
            r = extract_feature_clip(Image.open(p).convert("RGB"))
            refs.append((code, r))
        except Exception:
            continue
    return refs


def build_label_memory_prior_from_refs(
    query_img: Path,
    refs: List[Tuple[str, np.ndarray]],
    sim_threshold: float = 0.90,
    max_boost: float = 0.08,
) -> Dict[str, float]:
    if not refs:
        return {}
    try:
        q = extract_feature_clip(Image.open(query_img).convert("RGB"))
    except Exception:
        return {}

    prior: Dict[str, float] = {}
    for code, r in refs:
        sim = float(q @ r)
        if sim >= sim_threshold:
            t = min(1.0, (sim - sim_threshold) / max(1e-6, 1.0 - sim_threshold))
            boost = max_boost * t
            prior[code] = max(prior.get(code, 0.0), boost)
    return prior


def precompute_rerank_candidate_cache(
    standard_dir: Path,
    names: List[str],
    candidate_views_max: int = 1,
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    cache: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for name in names:
        file_name = name.split("@", 1)[0]
        if file_name in cache:
            continue
        fp = standard_dir / file_name
        if not fp.exists() or not fp.is_file():
            continue
        try:
            c_img = Image.open(fp).convert("RGB")
            c_views = [c_img] + _multi_crop_views(c_img, crop_ratio=0.82)[: max(0, candidate_views_max - 1)]
            cache[file_name] = [extract_modal_features(cv) for cv in c_views]
        except Exception:
            continue
    return cache


def rerank_candidates_with_model(
    query_img: Path,
    ranked_images: List[Tuple[str, float]],
    standard_dir: Path,
    reranker_model_path: Path,
    rerank_topn: int = 60,
    rerank_weight: float = 0.4,
    query_multicrop: bool = True,
    query_crop_ratio: float = 0.75,
    query_component_views: bool = True,
    rerank_query_views_max: int = 4,
    rerank_candidate_views_max: int = 2,
    candidate_feature_cache: Dict[str, List[Dict[str, np.ndarray]]] | None = None,
    max_unique_codes: int = 0,
) -> List[Tuple[str, float]]:
    if not reranker_model_path.exists():
        return ranked_images
    try:
        model = LinearReranker.load(reranker_model_path)
    except Exception:
        return ranked_images

    q_img = Image.open(query_img).convert("RGB")
    q_views = _build_query_views(
        q_img,
        query_multicrop=query_multicrop,
        query_crop_ratio=query_crop_ratio,
        component_views=query_component_views,
    )
    q_views = q_views[: max(1, rerank_query_views_max)]
    q_feats = [extract_modal_features(v) for v in q_views]
    topn = min(len(ranked_images), max(1, rerank_topn))
    head = ranked_images[:topn]
    if max_unique_codes > 0:
        filtered: List[Tuple[str, float]] = []
        code_seen = set()
        for item in head:
            code = filename_to_style_code(item[0])
            if code in code_seen:
                continue
            code_seen.add(code)
            filtered.append(item)
            if len(code_seen) >= max_unique_codes:
                break
        if filtered:
            head = filtered
    tail = ranked_images[topn:]

    rescored: List[Tuple[str, float, float]] = []
    cand_cache: Dict[str, List[Dict[str, np.ndarray]]] = candidate_feature_cache if candidate_feature_cache is not None else {}
    for name, base_score in head:
        file_name = name.split("@", 1)[0]
        c_path = standard_dir / file_name
        if not c_path.exists():
            rescored.append((name, base_score, base_score))
            continue
        try:
            cached = cand_cache.get(file_name)
            if cached is None:
                c_img = Image.open(c_path).convert("RGB")
                c_views = [c_img] + _multi_crop_views(c_img, crop_ratio=0.82)[: max(0, rerank_candidate_views_max - 1)]
                cached = [extract_modal_features(cv) for cv in c_views]
                cand_cache[file_name] = cached
            p_same = 0.0
            for qf in q_feats:
                for cf in cached:
                    feats = pair_features(qf, cf, base_score)[None, :]
                    p = float(model.prob(feats)[0])
                    if p > p_same:
                        p_same = p
            merged = (1.0 - rerank_weight) * base_score + rerank_weight * p_same
            rescored.append((name, base_score, merged))
        except Exception:
            rescored.append((name, base_score, base_score))

    rescored.sort(key=lambda x: x[2], reverse=True)
    reranked_head = [(name, base_score) for name, base_score, _ in rescored]
    return reranked_head + tail


def main() -> None:
    parser = argparse.ArgumentParser(description="输入1张测试图，返回近似款号(JSON)")
    parser.add_argument("query_image", type=Path, help="需要检索的图片路径（支持 png/jpg/jpeg）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not args.config.exists():
        raise FileNotFoundError(f"config not found: {args.config}")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    path_cfg = cfg.get("paths", {})
    search_cfg = cfg.get("search", {})

    standard_dir = Path(path_cfg.get("standard_dir", "data/standard_samples"))
    standard_pattern = str(path_cfg.get("standard_pattern", "B*.png"))
    image_exts = list(path_cfg.get("image_exts", ["png", "jpg", "jpeg"]))
    top_k = int(search_cfg.get("top_k", 5))
    candidate_multiplier = int(search_cfg.get("candidate_multiplier", 20))
    feature_backend = str(search_cfg.get("feature_backend", "clip"))
    min_score = float(search_cfg.get("min_score", 0.6))
    code_agg_top_n = int(search_cfg.get("code_agg_top_n", 3))
    code_agg_alpha = float(search_cfg.get("code_agg_alpha", 0.7))
    hybrid_weights = search_cfg.get("hybrid_weights", {})
    w_clip = float(hybrid_weights.get("clip", 0.55))
    w_shape = float(hybrid_weights.get("shape", 0.30))
    w_color = float(hybrid_weights.get("color", 0.15))
    w_stripe = float(hybrid_weights.get("stripe", 0.20))
    query_multicrop = bool(search_cfg.get("query_multicrop", True))
    query_crop_ratio = float(search_cfg.get("query_crop_ratio", 0.75))
    query_component_views = bool(search_cfg.get("query_component_views", True))
    standard_multicrop = bool(search_cfg.get("standard_multicrop", True))
    standard_crop_ratio = float(search_cfg.get("standard_crop_ratio", 0.72))
    ocr_hint_enabled = bool(search_cfg.get("ocr_hint_enabled", True))
    ocr_hint_boost = float(search_cfg.get("ocr_hint_boost", 0.08))
    rerank_enabled = bool(search_cfg.get("rerank_enabled", False))
    rerank_topn = int(search_cfg.get("rerank_topn", 60))
    rerank_weight = float(search_cfg.get("rerank_weight", 0.4))
    reranker_model = Path(search_cfg.get("reranker_model", "models/reranker_v1.npz"))
    rerank_query_views_max = int(search_cfg.get("rerank_query_views_max", 4))
    rerank_candidate_views_max = int(search_cfg.get("rerank_candidate_views_max", 2))
    label_memory_enabled = bool(search_cfg.get("label_memory_enabled", True))
    label_memory_path = Path(search_cfg.get("label_memory_path", "data/query_labels.json"))
    label_memory_sim_threshold = float(search_cfg.get("label_memory_sim_threshold", 0.90))
    label_memory_max_boost = float(search_cfg.get("label_memory_max_boost", 0.08))
    feature_cache_enabled = bool(search_cfg.get("feature_cache_enabled", True))
    db_feature_dtype = str(search_cfg.get("db_feature_dtype", "float32")).lower()
    recall_topn_cap = int(search_cfg.get("recall_topn_cap", 0))
    rerank_max_unique_codes = int(search_cfg.get("rerank_max_unique_codes", 0))

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

    query = args.query_image
    if not query.exists():
        raise FileNotFoundError(f"query image not found: {query}")

    image_topk = min(len(names), max(top_k * max(candidate_multiplier, 1), top_k))
    if recall_topn_cap > 0:
        image_topk = min(image_topk, recall_topn_cap)
    ranked_images = search_topk_images(
        query,
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
    if rerank_enabled:
        ranked_images = rerank_candidates_with_model(
            query,
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
            max_unique_codes=rerank_max_unique_codes,
        )
    rows = topk_style_codes(
        ranked_images,
        top_k,
        min_score=min_score,
        code_agg_top_n=code_agg_top_n,
        code_agg_alpha=code_agg_alpha,
        query_hint_code=try_extract_query_style_code(query) if ocr_hint_enabled else "",
        query_hint_boost=ocr_hint_boost if ocr_hint_enabled else 0.0,
        code_prior_boost=build_label_memory_prior(
            query,
            label_memory_path,
            sim_threshold=label_memory_sim_threshold,
            max_boost=label_memory_max_boost,
        )
        if label_memory_enabled
        else {},
    )
    result = {
        "query_image": str(query),
        "topk_style_codes": rows,
    }
    if rows:
        logging.info(
            "%s -> top1_code %s | best_image %s | %.4f",
            query.name,
            rows[0]["style_code"],
            rows[0]["best_standard_image"],
            rows[0]["score"],
        )
    else:
        logging.info("%s -> no result", query.name)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
