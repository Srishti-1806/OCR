from typing import Any

from app.graph.state import ExtractionState


async def consistency_validation_node(state: ExtractionState) -> ExtractionState:
    """Perform a simple consistency guard before mapping fields."""
    state["detected_fields"] = []
    state["validation_errors"] = []
    return state
