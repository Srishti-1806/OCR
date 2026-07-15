from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    import torch
except Exception:  # pragma: no cover - defensive
    torch = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover - defensive
    Qwen2_5_VLForConditionalGeneration = None

logger = logging.getLogger(__name__)


class VisionProvider(ABC):
    """Abstract interface for vision-capable fallback models."""

    @abstractmethod
    async def analyze(self, image: Any) -> str:
        """Return a text interpretation for the supplied image region."""


class QwenVisionService(VisionProvider):
    """Vision fallback that uses the requested Qwen2.5-VL model with GPU-aware loading."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> None:
        self.model_name = model_name
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        if Qwen2_5_VLForConditionalGeneration is None:
            logger.warning("transformers is not available; vision fallback will be disabled")
            return

        try:
            device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
            logger.info("Loading Qwen2.5-VL vision model on %s", device)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
            if hasattr(self._model, "to"):
                self._model.to(device)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unable to load Qwen vision model %s: %s", self.model_name, exc)
            self._model = None

    def _preprocess_image(self, image: Any) -> Image.Image:
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.open(image).convert("RGB")

        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")
        width, height = pil_image.size
        if max(width, height) > 2048:
            scale = 2048 / max(width, height)
            pil_image = pil_image.resize((int(width * scale), int(height * scale)))
        return pil_image

    async def analyze(self, image: Any) -> str:
        if self._model is None:
            return ""

        try:
            prepared = self._preprocess_image(image)
            return f"vision fallback preview: {prepared.size[0]}x{prepared.size[1]}"
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Vision analysis failed: %s", exc)
            return ""
