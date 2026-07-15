from typing import Any

from app.graph.state import ExtractionState


async def json_repair_node(state: ExtractionState) -> ExtractionState:
    """Repair malformed JSON output into the expected schema."""
    payload = state.get("final_json") or {}
    repaired = {
        "document_type": payload.get("document_type", "unknown"),
        "fields": payload.get("fields", []),
    }
    state["final_json"] = repaired
    state["validation_errors"] = []
    return state
