from pathlib import Path

from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
OUT = Path("models/clip-vit-base-patch32")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.save_pretrained(OUT)
    processor.save_pretrained(OUT)
    print(f"Saved CLIP model to: {OUT}")


if __name__ == "__main__":
    main()
