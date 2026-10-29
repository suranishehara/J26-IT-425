"""Export completed CatVTON jobs to a pre-filled human-evaluation CSV.

The application saves each completed request under runtime/outputs/jobs/<job-id>.
This script turns those technical records into review rows so the reviewer only
has to enter the subjective 1--5 scores and a short note.
"""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "runtime" / "outputs" / "jobs"
DEFAULT_OUTPUT = ROOT / "data" / "evaluation_results_prefilled.csv"

HEADERS = [
    "sample_id",
    "split",
    "garment_family",
    "garment_type",
    "model_version",
    "visual_quality_1_to_5",
    "identity_retention_1_to_5",
    "colour_print_preservation_1_to_5",
    "drape_fidelity_1_to_5",
    "cultural_appropriateness_1_to_5",
    "reviewer_id",
    "notes",
    "job_id",
    "result_image",
    "person_input",
    "garment_input",
    "generation_resolution",
    "generation_steps",
    "seed",
]


def category_details(category: str) -> tuple[str, str]:
    """Map the UI category to transparent report labels."""
    mapping = {
        "upper-body": ("western", "shirt_blouse_kurta"),
        "dress": ("western", "dress"),
        "saree": ("traditional", "saree"),
        "kandyan-osariya": ("traditional", "kandyan_osariya"),
    }
    return mapping.get(category, ("unknown", category or "unknown"))


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def completed_jobs() -> list[tuple[Path, dict, dict]]:
    records = []
    if not JOBS_DIR.exists():
        return records
    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir() or not (job_dir / "result.png").is_file():
            continue
        request = load_json(job_dir / "request.json")
        metrics = load_json(job_dir / "metrics.json")
        if request and metrics:
            records.append((job_dir, request, metrics))
    return sorted(records, key=lambda item: item[0].stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export completed try-on jobs for review.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviewer", default="Ravindu")
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    records = completed_jobs()

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        writer.writeheader()
        for index, (job_dir, request, metrics) in enumerate(records, start=1):
            category = request.get("category", metrics.get("category", "unknown"))
            family, garment_type = category_details(category)
            resolution = metrics.get("resolution", [])
            steps = metrics.get("steps", "")
            quality = request.get("quality") or ("high" if steps == 50 else "balanced")
            writer.writerow(
                {
                    "sample_id": f"T{index:03d}",
                    "split": "pilot",
                    "garment_family": family,
                    "garment_type": garment_type,
                    "model_version": f"CatVTON-{quality}",
                    # Human judgement must remain blank; it cannot be inferred truthfully.
                    "visual_quality_1_to_5": "",
                    "identity_retention_1_to_5": "",
                    "colour_print_preservation_1_to_5": "",
                    "drape_fidelity_1_to_5": "",
                    "cultural_appropriateness_1_to_5": "",
                    "reviewer_id": args.reviewer,
                    "notes": "",
                    "job_id": job_dir.name,
                    "result_image": relative(job_dir / "result.png"),
                    "person_input": relative(job_dir / "person.png"),
                    "garment_input": relative(job_dir / "garment.png"),
                    "generation_resolution": "x".join(map(str, resolution)),
                    "generation_steps": steps,
                    "seed": request.get("seed", metrics.get("seed", "")),
                }
            )
    print(f"Exported {len(records)} completed jobs to {output}")


if __name__ == "__main__":
    main()
