from typing import Any

from app.graph.state import ExtractionState
from app.services.ocr_service import preprocess


async def preprocessing_node(state: ExtractionState) -> ExtractionState:
    """Preprocess each loaded page image."""
    pages = state.get("pages", [])
    processed_images = [preprocess(page) for page in pages]
    state["processed_images"] = processed_images
    return state
