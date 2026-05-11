import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Silence known transformers/torch deprecation warnings that do not affect runtime results.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.utils\._pytree\._register_pytree_node is deprecated.*",
    category=UserWarning,
)

from clip_features import extract_feature_clip
from features import extract_feature
DEFAULT_CONFIG = Path("config/search_config.json")


def build_feature_db(standard_dir: Path, pattern: str, backend: str) -> Tuple[List[str], np.ndarray]:
    files = sorted(standard_dir.glob(pattern))
    if not files:
        files = sorted(standard_dir.glob("*.png"))
    if not files:
        raise RuntimeError(f"no standard images found in {standard_dir}")

    names = []
    feats = []
    for p in files:
        names.append(p.name)
        img = Image.open(p).convert("RGB")
        if backend == "clip":
            feats.append(extract_feature_clip(img))
        else:
            feats.append(extract_feature(img))
    return names, np.vstack(feats).astype(np.float32)


def search_topk_images(query_img: Path, names: List[str], feats: np.ndarray, top_k: int, backend: str) -> List[Tuple[str, float]]:
    q_img = Image.open(query_img).convert("RGB")
    if backend == "clip":
        q = extract_feature_clip(q_img)
    else:
        q = extract_feature(q_img)
    sims = feats @ q
    k = min(top_k, len(names))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(names[i], float(sims[i])) for i in idx]


def filename_to_style_code(img_name: str) -> str:
    stem = Path(img_name).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def topk_style_codes(ranked_images: List[Tuple[str, float]], top_k_codes: int) -> List[Dict[str, object]]:
    best_by_code: Dict[str, Tuple[str, float]] = {}
    for img_name, score in ranked_images:
        code = filename_to_style_code(img_name)
        prev = best_by_code.get(code)
        if prev is None or score > prev[1]:
            best_by_code[code] = (img_name, score)

    rows = [
        {
            "style_code": code,
            "best_standard_image": img_name,
            "score": round(score, 4),
        }
        for code, (img_name, score) in best_by_code.items()
    ]
    rows.sort(key=lambda x: float(x["score"]), reverse=True)
    return rows[:top_k_codes]


def main() -> None:
    parser = argparse.ArgumentParser(description="输入1张测试图，返回近似款号(JSON)")
    parser.add_argument("query_png", type=Path, help="需要检索的png图片路径")
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
    top_k = int(search_cfg.get("top_k", 5))
    candidate_multiplier = int(search_cfg.get("candidate_multiplier", 20))
    feature_backend = str(search_cfg.get("feature_backend", "clip"))

    names, feats = build_feature_db(standard_dir, standard_pattern, feature_backend)

    query = args.query_png
    if not query.exists():
        raise FileNotFoundError(f"query image not found: {query}")

    image_topk = min(len(names), max(top_k * max(candidate_multiplier, 1), top_k))
    ranked_images = search_topk_images(query, names, feats, image_topk, feature_backend)
    rows = topk_style_codes(ranked_images, top_k)
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
