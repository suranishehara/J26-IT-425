"""Create a small reproducible subset before launching the full VITON-HD test run."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "external" / "VITON-HD" / "datasets" / "test_pairs.txt"
target = ROOT / "external" / "VITON-HD" / "datasets" / "test_pairs_pilot.txt"
lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines:
    raise SystemExit("The VITON-HD test-pair list is empty.")
count = min(10, len(lines))
target.write_text("\n".join(lines[:count]) + "\n", encoding="utf-8")
print(f"Created {target.name} with {count} pair(s). Use test_pairs.txt for the full run.")
