from typing import Any

from app.graph.state import ExtractionState


async def field_mapping_node(state: ExtractionState) -> ExtractionState:
    """Map merged OCR candidates into form fields."""
    fields = []
    for item in state.get("merged_results", []):
        fields.append(
            {
                "label": item.get("text", "field"),
                "value": item.get("text", ""),
                "confidence": item.get("confidence", 0.0),
                "source": item.get("source", "paddle"),
            }
        )
    state["detected_fields"] = fields
    return state
