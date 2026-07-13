"""
OCR SERVICE
-----------
Owns everything from "raw uploaded bytes" to "normalized OCR tokens".
If PaddleOCR is swapped for another engine tomorrow (Surya, Tesseract, etc.),
only `run_ocr_on_image` and `_load_engine` need to change — nothing else in
the pipeline depends on which engine produced the tokens.

Public API used by the router:
    save_upload(file_bytes, filename) -> Path
    pdf_to_images(pdf_path) -> List[np.ndarray]
    load_image_any(path) -> List[np.ndarray]     # handles pdf OR image input
    preprocess(image) -> np.ndarray
    run_ocr_on_image(image, page_num) -> List[OCRToken]
    build_ocr_result(filename, tokens, page_count) -> OCRResult
"""
import os
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
import fitz

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.schemas import OCRToken, OCRResult, BoundingBox
from app.utils.geometry import group_into_lines

logger = get_logger(__name__)

_engine = None  # lazy-loaded singleton


# --------------------------------------------------------------------------
# File handling
# --------------------------------------------------------------------------
def save_upload(file_bytes: bytes, filename: str) -> Path:
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {size_mb:.1f}MB (max {settings.MAX_FILE_SIZE_MB}MB)")

    safe_name = f"{uuid.uuid4().hex}{ext}"
    dest = settings.UPLOAD_DIR / safe_name
    dest.write_bytes(file_bytes)
    logger.info(f"Saved upload '{filename}' -> {dest}")
    return dest


# --------------------------------------------------------------------------
# PDF -> images
# --------------------------------------------------------------------------
def pdf_to_images(pdf_path: Path) -> List[np.ndarray]:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError

    try:
        kwargs = {}
        if settings.POPPLER_PATH:
            kwargs["poppler_path"] = settings.POPPLER_PATH
        pil_pages = convert_from_path(str(pdf_path), dpi=settings.PDF_DPI, **kwargs)
    except PDFInfoNotInstalledError as e:
        raise ValueError(
            "Unable to get page count. Is poppler installed and in PATH? "
            "On Windows, install Poppler and add its Library\\bin folder to PATH. "
            "Then restart your terminal and retry."
        ) from e
    except OSError as e:
        if "Unable to get page count" in str(e) or "pdfinfo" in str(e).lower():
            raise ValueError(
                "Unable to get page count. Is poppler installed and in PATH? "
                "On Windows, install Poppler and add its Library\\bin folder to PATH. "
                "Then restart your terminal and retry."
            ) from e
        raise

    images = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pil_pages]
    logger.info(f"Converted PDF -> {len(images)} page image(s)")
    return images


def load_image_any(path: Path) -> List[np.ndarray]:
    """Returns a list of page images regardless of whether input is PDF or a plain image."""
    if path.suffix.lower() == ".pdf":
        return pdf_to_images(path)
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image file: {path}")
    return [img]


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def _deskew(gray: np.ndarray) -> np.ndarray:
    binarized = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(binarized > 0))
    if coords.shape[0] < 20:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.1:
        return gray
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess(image: np.ndarray) -> np.ndarray:
    """Gray -> denoise -> adaptive threshold -> deskew. Returns a 3-channel
    BGR image because PaddleOCR expects color input, even though the content
    is effectively binarized."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    deskewed = _deskew(thresh)
    return cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------
# OCR engine (PaddleOCR by default; swap here if engine changes)
# --------------------------------------------------------------------------
def _load_engine():
    global _engine
    if _engine is None:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("FLAGS_use_ngraph", "0")
        os.environ.setdefault("FLAGS_use_gpu", "0")
        os.environ.setdefault("PADDLE_WITH_CUDA", "0")
        from paddleocr import PaddleOCR
        logger.info(f"Loading PaddleOCR engine (lang={settings.OCR_LANG}) in CPU-only mode...")
        _engine = PaddleOCR(use_angle_cls=True, lang=settings.OCR_LANG, show_log=False, use_gpu=False)
    return _engine


def run_ocr_on_image(image: np.ndarray, page_num: int = 0) -> List[OCRToken]:
    engine = _load_engine()
    result = engine.ocr(image, cls=True)

    tokens: List[OCRToken] = []
    if not result or result[0] is None:
        return tokens

    for line in result[0]:
        box_points, (text, confidence) = line
        xs = [p[0] for p in box_points]
        ys = [p[1] for p in box_points]
        bbox = BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))
        tokens.append(
            OCRToken(text=text.strip(), bbox=bbox, confidence=float(confidence), page=page_num)
        )

    # assign line_id using geometry-based line grouping so downstream
    # consumers (LLM prompt builder) can reconstruct reading order cheaply
    lines = group_into_lines(tokens)
    for line_idx, idx_list in enumerate(lines):
        for tok_idx in idx_list:
            tokens[tok_idx].line_id = line_idx

    return tokens


def build_ocr_result(filename: str, all_tokens: List[OCRToken], page_count: int) -> OCRResult:
    avg_conf = (
        sum(t.confidence for t in all_tokens) / len(all_tokens) if all_tokens else 0.0
    )
    return OCRResult(
        filename=filename,
        page_count=page_count,
        tokens=all_tokens,
        avg_confidence=round(avg_conf, 4),
    )


# --------------------------------------------------------------------------
# Orchestration helper (used by router)
# --------------------------------------------------------------------------
def run_full_ocr_pipeline(file_bytes: bytes, filename: str) -> OCRResult:
    path = save_upload(file_bytes, filename)
    pages = load_image_any(path)

    all_tokens: List[OCRToken] = []
    running_line_offset = 0
    for page_num, page_img in enumerate(pages):
        try:
            pre = preprocess(page_img)
            page_tokens = run_ocr_on_image(pre, page_num=page_num)
        except Exception as exc:
            logger.warning(f"PaddleOCR failed for {filename} page {page_num}: {exc}")
            page_tokens = []

        if not page_tokens and path.suffix.lower() == ".pdf":
            try:
                doc = fitz.open(str(path))
                text = doc[page_num].get_text("text") if page_num < doc.page_count else ""
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                for line_idx, line in enumerate(lines):
                    bbox = BoundingBox(x1=0, y1=0, x2=1, y2=1)
                    all_tokens.append(
                        OCRToken(text=line, bbox=bbox, confidence=0.85, page=page_num, line_id=running_line_offset + line_idx)
                    )
                if lines:
                    running_line_offset += len(lines)
                continue
            except Exception as fallback_exc:
                logger.warning(f"PDF text fallback failed for {filename}: {fallback_exc}")

        # keep line_id unique across pages
        for t in page_tokens:
            if t.line_id is not None:
                t.line_id += running_line_offset
        if page_tokens:
            running_line_offset += max(t.line_id for t in page_tokens if t.line_id is not None) + 1
        all_tokens.extend(page_tokens)

    return build_ocr_result(filename=filename, all_tokens=all_tokens, page_count=len(pages))
