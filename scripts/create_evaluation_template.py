"""Create an empty CSV for transparent baseline/adaptation human evaluation."""
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data" / "evaluation_template.csv"
headers = ["sample_id", "split", "garment_family", "garment_type", "model_version", "visual_quality_1_to_5", "identity_retention_1_to_5", "colour_print_preservation_1_to_5", "drape_fidelity_1_to_5", "cultural_appropriateness_1_to_5", "reviewer_id", "notes"]
with path.open("w", newline="", encoding="utf-8") as file:
    csv.writer(file).writerow(headers)
print(path)
