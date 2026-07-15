from typing import Any

from app.graph.state import ExtractionState
from app.services.ocr_service import load_image_any


async def loader_node(state: ExtractionState) -> ExtractionState:
    """Load the uploaded document into page images."""
    file_path = state.get("file_path")
    if not file_path:
        raise ValueError("file_path is required")
    pages = load_image_any(__import__("pathlib").Path(file_path))
    state["pages"] = pages
    return state
