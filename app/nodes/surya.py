from typing import Any

from app.graph.state import ExtractionState
from app.services.surya_service import SuryaOCRService


async def surya_ocr_node(state: ExtractionState) -> ExtractionState:
    """Run Surya OCR when the Paddle OCR confidence is insufficient."""
    service = SuryaOCRService(threshold=state.get("confidence_threshold", 0.85))
    results = []
    for image in state.get("processed_images", []):
        results.extend(await service.run(image))
    state["surya_results"] = results
    return state
