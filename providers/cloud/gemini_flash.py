from __future__ import annotations

import json
import logging
from typing import Any, Optional

import cv2
import numpy as np

from providers.base import BaseProvider, ProviderResult
from config import cfg

logger = logging.getLogger(__name__)


def _rgb_b64_uri(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return ""
    import base64
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


class GeminiFlashCloudProvider(BaseProvider):
    """Gemini Flash: egy API hívásban OCR + karakter + fordítás."""

    provider_id = "gemini_flash"

    def __init__(self) -> None:
        super().__init__()
        self._session: Any = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=cfg.ocr.gemini_api_key)
            self._session = genai
            self._loaded = True
            logger.info("Gemini Flash provider init ✓")
        except Exception as e:
            logger.error(f"Gemini init error: {e}")
            self._session = None

    def release(self) -> None:
        self._session = None
        self._loaded = False

    def run(
        self,
        full_page_rgb: np.ndarray,
        bubble_list: list[dict],
    ) -> ProviderResult:
        self._assert_loaded()
        if self._session is None:
            return ProviderResult(success=False, error="Gemini session unavailable")

        try:
            model = self._session.GenerativeModel(cfg.ocr.gemini_model)
            prompt = self._build_prompt(bubble_list)
            parts = [prompt, _rgb_b64_uri(full_page_rgb)]
            response = model.generate_content(parts)
            text = (getattr(response, "text", "") or "").strip()
            data = self._parse_response(text, bubble_list)
            return ProviderResult(
                success=bool(data),
                data=data,
                confidence=0.92,
                provider_id=self.provider_id,
            )
        except Exception as e:  # pragma: no cover - runtime provider error
            logger.error(f"Gemini run error: {e}")
            return ProviderResult(success=False, error=str(e), provider_id=self.provider_id)

    @staticmethod
    def _build_prompt(bubble_list: list[dict]) -> str:
        count = len(bubble_list)
        return (
            "You are a comic localization engine for a single page.\n"
            "Bubbles: [x1, y1, x2, y2]. Same index in OUTPUT.\n"
            "Return ONLY JSON object with key 'bubbles', list of objects with keys: "
            "bubble_id (int), source_text (str), translated_text (str), speaker (str), confidence (float).\n"
            "Detect bubble text accurately, keep speaker if visible, translate to Hungarian when possible.\n"
            f"Bubble count: {count}\n"
        )

    @staticmethod
    def _parse_response(text: str, bubble_list: list[dict]) -> list[dict]:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                return []
            payload = json.loads(text[start:end + 1])
            bubbles = payload.get("bubbles", [])
        except Exception:
            return []

        out: list[dict] = []
        for i, item in enumerate(bubbles):
            if not isinstance(item, dict):
                continue
            try:
                out.append(
                    {
                        "bubble_id": int(item.get("bubble_id", i)),
                        "source_text": str(item.get("source_text", "")),
                        "translated_text": str(item.get("translated_text", "")),
                        "speaker": str(item.get("speaker", "")),
                        "confidence": float(item.get("confidence", 0.0)),
                    }
                )
            except Exception:
                continue
        return out
