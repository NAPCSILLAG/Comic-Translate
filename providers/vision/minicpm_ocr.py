from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from providers.base import BaseProvider, ProviderResult
from config import cfg

logger = logging.getLogger(__name__)


class MiniCPMOCRProvider(BaseProvider):
    """MiniCPM OCR – buborék-crop szintű szövegkinyerés."""

    provider_id = "minicpm_ocr"

    def __init__(self) -> None:
        super().__init__()
        self._session: Any = None

    def _load_model(self) -> bool:
        if self._loaded:
            return True
        try:
            import ollama
            self._session = ollama
            self._loaded = True
            logger.info("MiniCPM OCR provider init ✓")
            return True
        except Exception as e:
            logger.error(f"MiniCPM init error: {e}")
            self._session = None
            return False

    def _release_model(self) -> None:
        self._session = None
        self._loaded = False

    def _build_prompt(self) -> str:
        return (
            "You are an OCR engine for comic book bubbles.\n"
            "Return ONLY the text inside the bubble, verbatim. "
            "Do not translate, explain, or add punctuation.\n"
            "If the bubble is empty, return empty string."
        )

    def run(self, crop_rgb: np.ndarray) -> ProviderResult:
        if not self.is_ready or self._session is None:
            return ProviderResult(success=False, error="MiniCPM session unavailable")

        ok, buf = cv2.imencode(".png", cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return ProviderResult(success=False, error="PNG encode failed")

        try:
            response = self._session.generate(
                model=cfg.ocr.minicpm_model_name,
                prompt=self._build_prompt(),
                images=[buf.tobytes()],
                stream=False,
            )
            text = (response.get("response") or "").strip()
            return ProviderResult(
                success=True,
                data={"text": text},
                confidence=0.85 if text else 0.0,
                provider_id=self.provider_id,
            )
        except Exception as e:  # pragma: no cover - runtime provider error
            logger.error(f"MiniCPM run error: {e}")
            return ProviderResult(success=False, error=str(e), provider_id=self.provider_id)
