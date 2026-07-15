from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.graph.state import ExtractionState
from app.nodes.upload import upload_node
from app.nodes.loader import loader_node
from app.nodes.preprocess import preprocessing_node
from app.nodes.paddle import paddle_ocr_node
from app.nodes.confidence import confidence_node
from app.nodes.surya import surya_ocr_node
from app.nodes.roi_crop import roi_crop_node
from app.nodes.vision import vision_llm_node
from app.nodes.merge import merge_node
from app.nodes.consistency import consistency_validation_node
from app.nodes.field_mapping import field_mapping_node
from app.nodes.document_understanding import document_understanding_node
from app.nodes.json_generator import json_generation_node
from app.nodes.validation import schema_validation_node
from app.nodes.repair import json_repair_node


def build_extraction_graph() -> Any:
    """Build and return the LangGraph workflow used for form extraction."""
    workflow = StateGraph(ExtractionState)

    workflow.add_node("upload_node", upload_node)
    workflow.add_node("loader_node", loader_node)
    workflow.add_node("preprocessing_node", preprocessing_node)
    workflow.add_node("paddle_ocr_node", paddle_ocr_node)
    workflow.add_node("confidence_node", confidence_node)
    workflow.add_node("surya_ocr_node", surya_ocr_node)
    workflow.add_node("roi_crop_node", roi_crop_node)
    workflow.add_node("vision_llm_node", vision_llm_node)
    workflow.add_node("merge_node", merge_node)
    workflow.add_node("consistency_validation_node", consistency_validation_node)
    workflow.add_node("field_mapping_node", field_mapping_node)
    workflow.add_node("document_understanding_node", document_understanding_node)
    workflow.add_node("json_generation_node", json_generation_node)
    workflow.add_node("schema_validation_node", schema_validation_node)
    workflow.add_node("json_repair_node", json_repair_node)

    workflow.set_entry_point("upload_node")
    workflow.add_edge("upload_node", "loader_node")
    workflow.add_edge("loader_node", "preprocessing_node")
    workflow.add_edge("preprocessing_node", "paddle_ocr_node")
    workflow.add_edge("paddle_ocr_node", "confidence_node")

    workflow.add_conditional_edges(
        "confidence_node",
        lambda state: "surya_ocr_node" if state.get("used_source") != "paddle" else "merge_node",
        {
            "merge_node": "merge_node",
            "surya_ocr_node": "surya_ocr_node",
        },
    )

    workflow.add_edge("surya_ocr_node", "confidence_node")
    workflow.add_edge("confidence_node", "roi_crop_node")
    workflow.add_edge("roi_crop_node", "vision_llm_node")
    workflow.add_edge("vision_llm_node", "merge_node")
    workflow.add_edge("merge_node", "consistency_validation_node")
    workflow.add_edge("consistency_validation_node", "field_mapping_node")
    workflow.add_edge("field_mapping_node", "document_understanding_node")
    workflow.add_edge("document_understanding_node", "json_generation_node")
    workflow.add_edge("json_generation_node", "schema_validation_node")

    workflow.add_conditional_edges(
        "schema_validation_node",
        lambda state: "json_repair_node" if state.get("validation_errors") else END,
        {
            "json_repair_node": "json_repair_node",
            END: END,
        },
    )
    workflow.add_edge("json_repair_node", "schema_validation_node")

    return workflow.compile()
