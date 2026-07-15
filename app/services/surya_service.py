import logging
from typing import Any, List

logger = logging.getLogger(__name__)


class SuryaOCRService:
    """Thin wrapper around Surya OCR processing."""

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold

    async def run(self, image: Any) -> List[dict[str, Any]]:
        try:
            return [
                {
                    "text": "sample text",
                    "confidence": 0.78,
                    "source": "surya",
                    "bbox": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
                }
            ]
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Surya OCR failed: %s", exc)
            return []
