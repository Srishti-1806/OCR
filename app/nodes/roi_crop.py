from typing import Any

from app.graph.state import ExtractionState


async def roi_crop_node(state: ExtractionState) -> ExtractionState:
    """Crop low-confidence regions for the vision fallback."""
    state["vision_results"] = []
    return state
