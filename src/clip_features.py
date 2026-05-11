from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LOCAL_CLIP_DIR = Path("models/clip-vit-base-patch32")


@lru_cache(maxsize=1)
def _clip_bundle():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not LOCAL_CLIP_DIR.exists():
        raise FileNotFoundError(
            f"Local CLIP model not found: {LOCAL_CLIP_DIR}. "
            "Run: python scripts/download_clip_model.py"
        )
    model = CLIPModel.from_pretrained(str(LOCAL_CLIP_DIR), local_files_only=True)
    processor = CLIPProcessor.from_pretrained(str(LOCAL_CLIP_DIR), local_files_only=True)
    model.eval()
    model.to(device)
    return model, processor, device


def _to_embedding_tensor(out: object) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"Unsupported CLIP output type: {type(out)}")


def extract_feature_clip(image: Image.Image) -> np.ndarray:
    model, processor, device = _clip_bundle()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        feat = _to_embedding_tensor(model.get_image_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat[0].detach().cpu().numpy().astype(np.float32)


def extract_feature_clip_batch(images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
    model, processor, device = _clip_bundle()
    feats = []

    for i in range(0, len(images), batch_size):
        chunk = [x.convert("RGB") for x in images[i : i + batch_size]]
        inputs = processor(images=chunk, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            f = _to_embedding_tensor(model.get_image_features(**inputs))
            f = f / f.norm(dim=-1, keepdim=True)

        feats.append(f.detach().cpu().numpy().astype(np.float32))

    return np.vstack(feats)
