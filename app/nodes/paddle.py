from typing import Any

from app.graph.state import ExtractionState
from app.services.paddle_service import PaddleOCRService


async def paddle_ocr_node(state: ExtractionState) -> ExtractionState:
    """Run Paddle OCR on the processed images."""
    service = PaddleOCRService(threshold=state.get("confidence_threshold", 0.85))
    results = []
    for image in state.get("processed_images", []):
        results.extend(await service.run(image))
    state["paddle_results"] = results
    return state
