from typing import Any

from app.graph.state import ExtractionState


async def confidence_node(state: ExtractionState) -> ExtractionState:
    """Evaluate OCR confidence and decide whether to fall back to Surya."""
    results = state.get("paddle_results", [])
    threshold = state.get("confidence_threshold", 0.85)
    available = [item for item in results if item.get("confidence", 0.0) >= threshold]
    state["used_source"] = "paddle" if available else "surya"
    return state
