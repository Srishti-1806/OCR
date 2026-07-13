"""
App-wide configuration. All values are overridable via environment variables
so the OCR engine, LLM model, and directories can be swapped without touching code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Directories ---
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "/tmp/form_extractor/uploads"))
    repo_root = Path(__file__).resolve().parents[2]
    IMAGE_DIR: Path = Path(os.getenv("IMAGE_DIR", str(repo_root / "images")))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", str(repo_root / "output")))

    # --- OCR ---
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "paddleocr")   # swap-friendly
    OCR_LANG: str = os.getenv("OCR_LANG", "en")
    PDF_DPI: int = int(os.getenv("PDF_DPI", "300"))
    POPPLER_PATH: str = os.getenv("POPPLER_PATH", "")

    # --- LLM (Extraction Service) ---
    LLM_API_BASE_URL: str = os.getenv("LLM_API_BASE_URL", os.getenv("llm_url", ""))
    LLM_URL: str = os.getenv("LLM_API_BASE_URL", os.getenv("llm_url", ""))
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_API_KEY_HEADER: str = os.getenv("LLM_API_KEY_HEADER", "authorization")
    LLM_MODEL: str = os.getenv("LLM_MODEL", os.getenv("llm_name", "claude-sonnet-5"))
    LLM_NAME: str = os.getenv("LLM_MODEL", os.getenv("llm_name", "claude-sonnet-5"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    # --- Misc ---
    ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
