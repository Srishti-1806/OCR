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

NOTE on Surya OCR:
    Requires: pip install "surya-ocr==0.17.1" --break-system-packages
    (0.20+ switched to a vllm/llama.cpp server-based API with different
    imports — pin 0.17.1 to keep the simple DetectionPredictor /
    RecognitionPredictor API used below)
"""
import os
import uuid
from pathlib import Path
from typing import List, Optional

try:
    import cv2
except Exception:  # pragma: no cover - headless/container fallback
    cv2 = None

import numpy as np
from PIL import Image
import fitz

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.schemas import OCRToken, OCRResult, BoundingBox
from app.utils.geometry import group_into_lines

logger = get_logger(__name__)

_engine = None  # lazy-loaded singleton
_surya_engine = None  # lazy-loaded (det_predictor, rec_predictor) tuple


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
def _pil_to_numpy(image: Image.Image) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return np.array(image)


def _fitz_to_numpy(pdf_path: Path) -> List[np.ndarray]:
    doc = fitz.open(str(pdf_path))
    images: List[np.ndarray] = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(_pil_to_numpy(image))
    doc.close()
    return images


def pdf_to_images(pdf_path: Path) -> List[np.ndarray]:
    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError

        kwargs = {}
        if settings.POPPLER_PATH:
            kwargs["poppler_path"] = settings.POPPLER_PATH
        pil_pages = convert_from_path(str(pdf_path), dpi=settings.PDF_DPI, **kwargs)
    except (PDFInfoNotInstalledError, OSError, Exception) as exc:
        logger.info("Falling back to PyMuPDF rendering for %s: %s", pdf_path, exc)
        return _fitz_to_numpy(pdf_path)

    images = [_pil_to_numpy(page) for page in pil_pages]
    logger.info(f"Converted PDF -> {len(images)} page image(s)")
    return images


def load_image_any(path: Path) -> List[np.ndarray]:
    """Returns a list of page images regardless of whether input is PDF or a plain image."""
    if path.suffix.lower() == ".pdf":
        return pdf_to_images(path)

    if cv2 is not None:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image file: {path}")
        return [img]

    with Image.open(path) as img:
        image = img.convert("RGB")
        return [_pil_to_numpy(image)]


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
    if cv2 is None:
        pil_image = Image.fromarray(image).convert("L")
        return np.array(pil_image.convert("RGB"))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    deskewed = _deskew(thresh)
    return cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------
# OCR engine (RapidOCR by default; optional Surya fallback for low-confidence regions)
# --------------------------------------------------------------------------
def _load_engine():
    global _engine
    if _engine is None:
        backend = (settings.OCR_ENGINE or "rapidocr").lower()
        if backend == "surya":
            try:
                from surya.foundation import FoundationPredictor
                from surya.detection import DetectionPredictor
                from surya.recognition import RecognitionPredictor
                logger.info("Loading Surya OCR engine...")
                foundation_predictor = FoundationPredictor()
                _engine = {
                    "det": DetectionPredictor(),
                    "rec": RecognitionPredictor(foundation_predictor),
                }
                return _engine
            except Exception as exc:
                logger.warning("Surya OCR requested but unavailable: %s; falling back to RapidOCR", exc)

        from rapidocr_onnxruntime import RapidOCR
        logger.info(f"Loading RapidOCR engine (ONNX Runtime, lang={settings.OCR_LANG})...")
        _engine = RapidOCR()
    return _engine


def _load_surya_engine():
    """Lazily builds and caches the (det_predictor, rec_predictor) pair.
    Returns None if surya-ocr isn't installed / fails to import."""
    global _surya_engine
    if _surya_engine is None:
        try:
            from surya.foundation import FoundationPredictor
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            foundation_predictor = FoundationPredictor()
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor(foundation_predictor)
            _surya_engine = (det_predictor, rec_predictor)
        except Exception as exc:
            logger.info("Surya OCR module unavailable: %s", exc)
            _surya_engine = False
            return None
    return None if _surya_engine is False else _surya_engine


def _run_surya_ocr_on_region(image: np.ndarray) -> List[tuple]:
    """Runs Surya OCR on a cropped BGR numpy region and returns results in
    the same (box_points, text, confidence) tuple shape RapidOCR produces,
    so nothing downstream needs to know which engine ran."""
    engines = _load_surya_engine()
    if engines is None or image.size == 0:
        return []
    det_predictor, rec_predictor = engines

    # Surya expects RGB PIL images, not BGR numpy arrays
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    try:
        predictions = rec_predictor([pil_img], det_predictor=det_predictor)
    except Exception as exc:  # pragma: no cover - runtime dependency path
        logger.info("Surya OCR fallback failed: %s", exc)
        return []

    if not predictions or not predictions[0].text_lines:
        return []

    results = []
    for line in predictions[0].text_lines:
        x1, y1, x2, y2 = line.bbox
        box_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        confidence = float(line.confidence) if line.confidence is not None else 0.0
        results.append((box_points, line.text.strip(), confidence))

    return results


def _crop_region(image: np.ndarray, bbox: BoundingBox, padding: float = 0.15) -> np.ndarray:
    h, w = image.shape[:2]
    pad_x = int(max(1, (bbox.x2 - bbox.x1) * padding))
    pad_y = int(max(1, (bbox.y2 - bbox.y1) * padding))
    x1 = max(0, int(bbox.x1) - pad_x)
    y1 = max(0, int(bbox.y1) - pad_y)
    x2 = min(w, int(bbox.x2) + pad_x)
    y2 = min(h, int(bbox.y2) + pad_y)
    return image[y1:y2, x1:x2]


def _detect_checkbox_regions(image: np.ndarray) -> List[BoundingBox]:
    if cv2 is None:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[BoundingBox] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12:
            continue
        aspect = w / float(h)
        if aspect < 0.7 or aspect > 1.4:
            continue
        boxes.append(BoundingBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h)))
    return boxes


def run_ocr_on_image(image: np.ndarray, page_num: int = 0) -> List[OCRToken]:
    if cv2 is None:
        return []
    engine = _load_engine()
    result, _elapse = engine(image)

    tokens: List[OCRToken] = []
    if not result:
        return tokens

    for box_points, text, confidence in result:
        xs = [p[0] for p in box_points]
        ys = [p[1] for p in box_points]
        bbox = BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))
        tokens.append(
            OCRToken(
                text=text.strip(),
                bbox=bbox,
                confidence=float(confidence),
                page=page_num,
                ocr_source="rapidocr",
            )
        )

    threshold = settings.OCR_CONFIDENCE_THRESHOLD
    refined_tokens: List[OCRToken] = []
    for token in tokens:
        if token.confidence >= threshold:
            refined_tokens.append(token)
            continue

        region = _crop_region(image, token.bbox, padding=0.2)
        if region.size == 0:
            refined_tokens.append(token)
            continue

        surya_result = _run_surya_ocr_on_region(region)
        if surya_result:
            first = surya_result[0]
            if isinstance(first, tuple) and len(first) >= 2:
                box_points, text, confidence = first
                xs = [p[0] for p in box_points]
                ys = [p[1] for p in box_points]
                bbox = BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))
                refined_tokens.append(
                    OCRToken(
                        text=str(text).strip(),
                        bbox=bbox,
                        confidence=float(confidence),
                        page=page_num,
                        ocr_source="surya",
                    )
                )
                continue

        refined_tokens.append(token)

    for checkbox in _detect_checkbox_regions(image):
        refined_tokens.append(
            OCRToken(
                text="[checkbox]",
                bbox=checkbox,
                confidence=0.95,
                page=page_num,
                is_checkbox=True,
                ocr_source="opencv",
            )
        )

    # assign line_id using geometry-based line grouping so downstream
    # consumers (LLM prompt builder) can reconstruct reading order cheaply
    lines = group_into_lines(refined_tokens)
    for line_idx, idx_list in enumerate(lines):
        for tok_idx in idx_list:
            refined_tokens[tok_idx].line_id = line_idx

    return refined_tokens


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
            logger.warning(f"OCR engine failed for {filename} page {page_num}: {exc}")
            page_tokens = []

        if not page_tokens and path.suffix.lower() == ".pdf":
            # Fallback only helps born-digital PDFs that already have a text
            # layer (e.g. exported from Word). Scanned/handwritten PDFs have
            # no embedded text, so this will still yield 0 tokens for those —
            # PaddleOCR actually working is the only real fix in that case.
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