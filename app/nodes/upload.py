from typing import Any

from app.graph.state import ExtractionState


async def upload_node(state: ExtractionState) -> ExtractionState:
    """Persist uploaded content and initialize the work state."""
    file_path = state.get("file_path")
    if not file_path:
        raise ValueError("file_path is required")
    state["file_name"] = state.get("file_name") or file_path.split("/")[-1]
    state["confidence_threshold"] = state.get("confidence_threshold", 0.85)
    state["validation_errors"] = []
    return state
