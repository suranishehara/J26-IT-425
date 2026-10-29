import json
from pathlib import Path


def load_catalogue(path: Path) -> list[dict]:
    """Load a small, auditable garment catalogue used for routing."""
    return json.loads(path.read_text(encoding="utf-8"))["garments"]
