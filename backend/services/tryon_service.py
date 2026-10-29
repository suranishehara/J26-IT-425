from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


class ValidationError(ValueError):
    pass


class TryOnService:
    """Routes a garment and returns an auditable try-on job contract.

    A production implementation replaces the demo renderer at this boundary,
    keeping client, catalogue and evaluation contracts unchanged.
    """

    def __init__(self, catalogue: list[dict]):
        self.catalogue = catalogue

    def create_job(self, request: dict) -> dict:
        garment_id = request.get("garmentId")
        photo_name = str(request.get("photoName", "")).strip()
        if not photo_name:
            raise ValidationError("A person photo is required.")
        garment = next((g for g in self.catalogue if g["id"] == garment_id), None)
        if garment is None:
            raise ValidationError("Select a supported garment.")
        route = "traditional_draping" if garment["family"] == "traditional" else "global_tryon"
        checks = ["person identity retained", "garment colour and print retained"]
        if route == "traditional_draping":
            checks += ["traditional pleat/front-fall representation", "fabric fold and texture representation"]
        return {
            "jobId": str(uuid4()),
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "status": "preview_ready",
            "renderer": "demo_overlay",
            "route": route,
            "garment": garment,
            "checks": checks,
            "notice": "Pilot preview only. It is not an AI-generated or physically accurate drape.",
        }
