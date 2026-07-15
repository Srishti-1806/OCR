from typing import Any

from app.graph.state import ExtractionState
from app.services.vision_service import QwenVisionService


async def vision_llm_node(state: ExtractionState) -> ExtractionState:
    """Use the Qwen vision model only on cropped low-confidence regions."""
    service = QwenVisionService()
    results = []
    for image in state.get("processed_images", []):
        text = await service.analyze(image)
        if text:
            results.append({"text": text, "confidence": 0.6, "source": "vision"})
    state["vision_results"] = results
    return state
