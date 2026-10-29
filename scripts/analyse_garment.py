"""Run a reproducible zero-shot prediction on one garment image."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from attributes import AttributeClassifier
from PIL import Image, ImageOps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    print(json.dumps(AttributeClassifier(args.device).predict(image), indent=2))
