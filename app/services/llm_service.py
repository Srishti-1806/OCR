import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class LLMService:
    """Simple deterministic fallback that maps OCR candidates into form JSON."""

    async def generate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return {
                "document_type": "form",
                "fields": [
                    {
                        "label": "field",
                        "value": candidate.get("text", ""),
                        "confidence": candidate.get("confidence", 0.0),
                        "source": candidate.get("source", "paddle"),
                    }
                    for candidate in data.get("merged_results", [])
                ],
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("JSON generation failed: %s", exc)
            return {"document_type": "unknown", "fields": []}
