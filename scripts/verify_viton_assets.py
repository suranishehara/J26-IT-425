"""Check prerequisites before starting a VITON-HD inference run."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "config" / "viton_hd.json").read_text(encoding="utf-8"))
missing = []
for relative in config["checkpoint_files"]:
    target = ROOT / config["checkpoint_directory"] / relative
    if not target.is_file():
        missing.append(f"checkpoint: {target.relative_to(ROOT)}")
dataset = ROOT / config["dataset_directory"] / config["dataset_mode"]
for folder in ("image", "cloth", "cloth-mask", "image-parse", "openpose-img", "openpose-json"):
    if not (dataset / folder).is_dir():
        missing.append(f"dataset folder: {(dataset / folder).relative_to(ROOT)}")
pairs = ROOT / config["dataset_directory"] / config["dataset_list"]
if not pairs.is_file():
    missing.append(f"pair list: {pairs.relative_to(ROOT)}")
try:
    import torch
    if not torch.cuda.is_available():
        missing.append("CUDA: PyTorch cannot see an NVIDIA CUDA device")
    else:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime: {torch.version.cuda}")
except ImportError:
    missing.append("Python dependency: torch is not installed")
if missing:
    print("VITON-HD is not ready:")
    print("\n".join(f"- {item}" for item in missing))
    sys.exit(1)
print("VITON-HD assets and CUDA prerequisites are ready.")
