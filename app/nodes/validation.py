from typing import Any

from app.graph.state import ExtractionState


async def schema_validation_node(state: ExtractionState) -> ExtractionState:
    """Validate the generated JSON structure and store validation errors if any."""
    payload = state.get("final_json") or {}
    errors = []
    if not isinstance(payload.get("fields"), list):
        errors.append("fields must be a list")
    if not isinstance(payload.get("document_type"), str):
        errors.append("document_type must be a string")
    state["validation_errors"] = errors
    return state
