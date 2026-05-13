import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from features import extract_feature, extract_garment_color_feature, extract_stripe_feature
from clip_features import extract_feature_clip


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x) + 1e-8)


def extract_modal_features(image: Image.Image) -> Dict[str, np.ndarray]:
    return {
        "clip": _l2norm(extract_feature_clip(image)),
        "shape": _l2norm(extract_feature(image)),
        "color": _l2norm(extract_garment_color_feature(image)),
        "stripe": _l2norm(extract_stripe_feature(image)),
    }


def pair_features(q: Dict[str, np.ndarray], c: Dict[str, np.ndarray], base_score: float) -> np.ndarray:
    s_clip = float(q["clip"] @ c["clip"])
    s_shape = float(q["shape"] @ c["shape"])
    s_color = float(q["color"] @ c["color"])
    s_stripe = float(q["stripe"] @ c["stripe"])
    feats = np.array(
        [
            base_score,
            s_clip,
            s_shape,
            s_color,
            s_stripe,
            s_clip - s_shape,
            s_shape - s_color,
            s_stripe - s_color,
            s_clip * s_stripe,
            s_shape * s_stripe,
        ],
        dtype=np.float32,
    )
    return feats


@dataclass
class LinearReranker:
    w: np.ndarray
    b: float
    mean: np.ndarray
    std: np.ndarray

    def score(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        return z @ self.w + self.b

    def prob(self, x: np.ndarray) -> np.ndarray:
        s = self.score(x)
        return 1.0 / (1.0 + np.exp(-np.clip(s, -20.0, 20.0)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w=self.w, b=np.array([self.b], dtype=np.float32), mean=self.mean, std=self.std)

    @staticmethod
    def load(path: Path) -> "LinearReranker":
        arr = np.load(path)
        return LinearReranker(
            w=arr["w"].astype(np.float32),
            b=float(arr["b"][0]),
            mean=arr["mean"].astype(np.float32),
            std=arr["std"].astype(np.float32),
        )


def train_logreg(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 1200,
    l2: float = 1e-4,
) -> LinearReranker:
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    z = (x - mean) / std

    w = np.zeros(z.shape[1], dtype=np.float32)
    b = 0.0
    n = float(z.shape[0])

    for _ in range(epochs):
        s = z @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -20.0, 20.0)))
        err = p - y
        gw = (z.T @ err) / n + l2 * w
        gb = float(err.mean())
        w -= lr * gw
        b -= lr * gb

    return LinearReranker(w=w, b=b, mean=mean, std=std)


def read_style_code_from_name(name: str) -> str:
    stem = Path(name.split("@", 1)[0]).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def save_training_report(path: Path, metrics: Dict[str, float], config: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"metrics": metrics, "config": config}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
