"""
API SERVICE (router layer)
---------------------------
Pure orchestration: receives the upload, calls OCR Service, calls Extraction
Service, wraps the result, handles errors/logging. No OCR or LLM logic lives
here — that's the whole point of the split.
"""
import json
import uuid
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.schemas import ExtractionResponse, ExtractionResult, OCRResult
from app.services import ocr_service, extraction_service

logger = get_logger(__name__)
router = APIRouter(prefix="/extract", tags=["extraction"])


def _save_extraction_output(extraction_response: ExtractionResponse) -> Path:
    safe_name = f"{Path(extraction_response.data.filename if extraction_response.data else 'output').stem}_{uuid.uuid4().hex}.json"
    dest = settings.OUTPUT_DIR / safe_name
    dest.write_text(json.dumps(extraction_response.dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


@router.post("", response_model=ExtractionResponse)
async def extract_form(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        logger.info(f"Received file: {file.filename} ({len(file_bytes)} bytes)")

        # 1. OCR Service: raw bytes -> normalized tokens
        ocr_result = ocr_service.run_full_ocr_pipeline(file_bytes, file.filename)
        logger.info(
            f"OCR complete: {len(ocr_result.tokens)} tokens, "
            f"avg_confidence={ocr_result.avg_confidence}"
        )

        if not ocr_result.tokens:
            return ExtractionResponse(
                success=False, error="No text could be detected in the uploaded document."
            )

        # 2. Extraction Service: tokens -> validated JSON fields (via LLM, with retries)
        extraction_result = extraction_service.extract_fields(ocr_result)
        logger.info(
            f"Extraction complete: {len(extraction_result.fields)} fields "
            f"(attempts={extraction_result.llm_attempts})"
        )

        response = ExtractionResponse(success=True, data=extraction_result)
        try:
            saved_path = _save_extraction_output(response)
            logger.info(f"Saved extraction JSON output: {saved_path}")
        except Exception as save_exc:
            logger.warning(f"Failed to save extraction JSON output: {save_exc}")

        return response

    except ValueError as e:
        # bad file type, too large, unreadable image, etc.
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("Unexpected error during extraction")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.get("/outputs")
def list_outputs():
    files = []
    for path in sorted(settings.OUTPUT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        files.append({
            "filename": path.name,
            "path": str(path),
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "size": path.stat().st_size,
        })
    return {"outputs": files}


@router.get("/outputs/{output_name}")
def download_output(output_name: str):
    output_path = settings.OUTPUT_DIR / output_name
    if not output_path.exists() or not output_path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(output_path, filename=output_name, media_type="application/json")
