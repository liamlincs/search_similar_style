import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from reranker import (
    extract_modal_features,
    pair_features,
    read_style_code_from_name,
    save_training_report,
    train_logreg,
)
from search_similar_return_code import collect_images


DEFAULT_CONFIG = Path("config/search_config.json")


def _multi_crop_views(img: Image.Image, crop_ratio: float = 0.72) -> List[Image.Image]:
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
    out = [img]
    for b in boxes:
        out.append(img.crop(b))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight same-style reranker.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("models/reranker_v1.npz"))
    parser.add_argument("--report", type=Path, default=Path("outputs/reranker_train_report.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-labels", type=Path, default=Path("data/query_labels.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    p = cfg["paths"]
    files = collect_images(Path(p["standard_dir"]), str(p.get("standard_pattern", "*")), p.get("image_exts", ["png", "jpg", "jpeg"]))
    if len(files) < 4:
        raise RuntimeError("not enough standard images to train reranker")

    names: List[str] = []
    feats: List[Dict[str, np.ndarray]] = []
    codes: List[str] = []
    by_code: Dict[str, List[int]] = {}

    for i, fp in enumerate(files):
        img = Image.open(fp).convert("RGB")
        f = extract_modal_features(img)
        names.append(fp.name)
        feats.append(f)
        c = read_style_code_from_name(fp.name)
        codes.append(c)
        by_code.setdefault(c, []).append(i)

    x_rows: List[np.ndarray] = []
    y_rows: List[float] = []

    # synthetic query views from each standard image
    for qi, fp in enumerate(files):
        q_code = codes[qi]
        img = Image.open(fp).convert("RGB")
        q_views = _multi_crop_views(img, crop_ratio=0.72)
        q_mods = [extract_modal_features(v) for v in q_views]

        # positives: same code
        pos_ids = [idx for idx in by_code[q_code] if idx != qi]
        if not pos_ids:
            pos_ids = [qi]

        # hard negatives by clip similarity against full query
        q_clip = q_mods[0]["clip"]
        clip_scores = []
        for ci, cfeat in enumerate(feats):
            if codes[ci] == q_code:
                continue
            clip_scores.append((ci, float(q_clip @ cfeat["clip"])))
        clip_scores.sort(key=lambda t: t[1], reverse=True)
        neg_ids = [ci for ci, _ in clip_scores[:16]]
        if len(neg_ids) < 8:
            all_neg = [i for i in range(len(files)) if codes[i] != q_code]
            random.shuffle(all_neg)
            neg_ids.extend(all_neg[: 8 - len(neg_ids)])

        for ci in pos_ids:
            best = -1.0
            for qmod in q_mods:
                base = float(qmod["clip"] @ feats[ci]["clip"])
                v = pair_features(qmod, feats[ci], base)
                x_rows.append(v)
                y_rows.append(1.0)
                best = max(best, base)

        for ci in neg_ids:
            for qmod in q_mods[:3]:
                base = float(qmod["clip"] @ feats[ci]["clip"])
                v = pair_features(qmod, feats[ci], base)
                x_rows.append(v)
                y_rows.append(0.0)

    x = np.vstack(x_rows).astype(np.float32)
    y = np.array(y_rows, dtype=np.float32)

    # Optional supervised query labels for hard-case correction.
    if args.query_labels.exists():
        data = json.loads(args.query_labels.read_text(encoding="utf-8"))
        labeled = data.get("labels", [])
        for row in labeled:
            q_path = Path(str(row.get("query_image", "")))
            gt_code = str(row.get("style_code", "")).strip()
            if not q_path.exists() or not gt_code:
                continue
            q_img = Image.open(q_path).convert("RGB")
            q_views = _multi_crop_views(q_img, crop_ratio=0.72)
            q_mods = [extract_modal_features(v) for v in q_views]

            pos_ids = [i for i, c in enumerate(codes) if c == gt_code]
            neg_ids = [i for i, c in enumerate(codes) if c != gt_code]
            if not pos_ids or not neg_ids:
                continue

            for pi in pos_ids:
                for qmod in q_mods:
                    base = float(qmod["clip"] @ feats[pi]["clip"])
                    x_rows.append(pair_features(qmod, feats[pi], base))
                    y_rows.append(1.0)

            # hard negatives from clip
            q_clip = q_mods[0]["clip"]
            hard = sorted(
                [(ni, float(q_clip @ feats[ni]["clip"])) for ni in neg_ids],
                key=lambda t: t[1],
                reverse=True,
            )[:24]
            for ni, _ in hard:
                for qmod in q_mods[:3]:
                    base = float(qmod["clip"] @ feats[ni]["clip"])
                    x_rows.append(pair_features(qmod, feats[ni], base))
                    y_rows.append(0.0)

    x = np.vstack(x_rows).astype(np.float32)
    y = np.array(y_rows, dtype=np.float32)
    model = train_logreg(x, y, lr=0.05, epochs=1200, l2=1e-4)
    model.save(args.output)

    p_hat = model.prob(x)
    pred = (p_hat >= 0.5).astype(np.float32)
    acc = float((pred == y).mean())
    pos_rate = float(y.mean())
    report = {
        "samples": int(len(y)),
        "pos_rate": round(pos_rate, 4),
        "train_acc": round(acc, 4),
    }
    save_training_report(args.report, report, {"seed": args.seed, "output": str(args.output)})
    print(json.dumps({"model": str(args.output), "report": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
