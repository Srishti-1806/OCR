from typing import Any, List, Optional, TypedDict


class ExtractionState(TypedDict, total=False):
    """State carried through the LangGraph form extraction workflow."""

    file_path: Optional[str]
    file_name: Optional[str]
    pages: List[Any]
    processed_images: List[Any]
    paddle_results: List[dict[str, Any]]
    surya_results: List[dict[str, Any]]
    vision_results: List[dict[str, Any]]
    merged_results: List[dict[str, Any]]
    detected_fields: List[dict[str, Any]]
    final_json: dict[str, Any]
    validation_errors: List[str]
    confidence_threshold: float
    used_source: str
