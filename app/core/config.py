"""
App-wide configuration. All values are overridable via environment variables
so the OCR engine, LLM model, and directories can be swapped without touching code.
"""
import os
from pathlib import Path


class Settings:
    # --- Directories ---
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "/tmp/form_extractor/uploads"))
    IMAGE_DIR: Path = Path(os.getenv("IMAGE_DIR", "/tmp/form_extractor/images"))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "/tmp/form_extractor/output"))

    # --- OCR ---
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "rapidocr")
    OCR_LANG: str = os.getenv("OCR_LANG", "en")
    PDF_DPI: int = int(os.getenv("PDF_DPI", "300"))
    OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.85"))
    # Optional: explicit path to poppler's /bin folder (Windows). If unset,
    # pdf2image falls back to whatever's on the system PATH.
    POPPLER_PATH: str = os.getenv("POPPLER_PATH", "")

    # --- LLM (Extraction Service) — internal MCP LLM client (OpenAI/vLLM-compatible) ---
    LLM_API_BASE_URL: str = os.getenv("LLM_API_BASE_URL", "https://mcp-llm-client.qryde.net/v1/")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")  # only sent if non-empty
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    # --- Misc ---
    ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)