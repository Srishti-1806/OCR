from typing import Any

from app.graph.state import ExtractionState


async def document_understanding_node(state: ExtractionState) -> ExtractionState:
    """Add document-level metadata to the extraction state."""
    state["final_json"] = {"document_type": "form", "fields": state.get("detected_fields", [])}
    return state
