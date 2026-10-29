"""Run official VITON-HD inference using the local reproducible configuration."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "config" / "viton_hd.json").read_text(encoding="utf-8"))
subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_viton_assets.py")], check=True)
repo = ROOT / "external" / "VITON-HD"
command = [sys.executable, "test.py", "--name", "baseline_native_1024x768", "--batch_size", str(config["batch_size"]), "--load_width", str(config["input_width"]), "--load_height", str(config["input_height"]), "--dataset_dir", str((ROOT / config["dataset_directory"]).resolve()), "--dataset_mode", config["dataset_mode"], "--dataset_list", config["dataset_list"], "--checkpoint_dir", str((ROOT / config["checkpoint_directory"]).resolve()), "--save_dir", str((ROOT / config["result_directory"]).resolve())]
print("Running:", " ".join(command))
subprocess.run(command, cwd=repo, check=True)
