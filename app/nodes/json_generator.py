from typing import Any

from app.graph.state import ExtractionState
from app.services.llm_service import LLMService


async def json_generation_node(state: ExtractionState) -> ExtractionState:
    """Generate a structured JSON payload from the merged extraction state."""
    service = LLMService()
    payload = await service.generate({"merged_results": state.get("merged_results", [])})
    state["final_json"] = payload
    return state
