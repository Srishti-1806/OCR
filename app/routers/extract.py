"""
API SERVICE (router layer)
---------------------------
Pure orchestration: receives the upload, calls OCR Service, calls Extraction
Service, wraps the result, handles errors/logging. No OCR or LLM logic lives
here — that's the whole point of the split.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.logging_config import get_logger
from app.models.schemas import ExtractionResponse
from app.services import ocr_service, extraction_service

logger = get_logger(__name__)
router = APIRouter(prefix="/extract", tags=["extraction"])


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

        return ExtractionResponse(success=True, data=extraction_result)

    except ValueError as e:
        # bad file type, too large, unreadable image, etc.
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("Unexpected error during extraction")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
