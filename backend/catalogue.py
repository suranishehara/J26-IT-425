"""Validate catalogue records without inferring missing facts."""
import math
from taxonomy import ATTRIBUTES
from sizing import validate_sizes

def validate_catalogue(records):
    if not isinstance(records, list):
        raise ValueError("Catalogue must be a list.")
    ids = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Each catalogue entry must be an object.")
        validate_sizes(row.get('size_chart'))
        for field in ("id", "name", "family", "data_status"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("Missing text field: " + field)
        if row["id"] in ids:
            raise ValueError("Duplicate garment ID: " + row["id"])
        ids.add(row["id"])
        if row["family"] not in ("global", "traditional"):
            raise ValueError("Unknown garment family.")
        for field, labels in ATTRIBUTES.items():
            if row.get(field) not in labels and row.get(field) != "unknown":
                raise ValueError("Invalid " + field + " for " + row["id"])
        price = row.get("price_lkr")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price < 0:
            raise ValueError("Price must be a finite non-negative number.")
        for field in ("occasions", "weather", "colours", "styles", "textile_techniques"):
            if not isinstance(row.get(field), list) or any(not isinstance(x, str) for x in row[field]):
                raise ValueError("Expected a text list: " + field)
        if row.get("price_source") not in ("illustrative", "verified"):
            raise ValueError("Price source must be illustrative or verified.")
        if not isinstance(row.get("source"), dict) or not row["source"].get("kind"):
            raise ValueError("Document the source kind.")
    return records
