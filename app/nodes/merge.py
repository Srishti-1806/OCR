from typing import Any

from app.graph.state import ExtractionState


async def merge_node(state: ExtractionState) -> ExtractionState:
    """Merge OCR sources into a single candidate list."""
    merged = []
    merged.extend(state.get("paddle_results", []))
    merged.extend(state.get("surya_results", []))
    merged.extend(state.get("vision_results", []))
    state["merged_results"] = merged
    return state
