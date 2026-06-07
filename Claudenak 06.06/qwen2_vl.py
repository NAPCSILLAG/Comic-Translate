"""
providers/vision/qwen2_vl.py - Ollama API alapú vizualis OCR validator + context analyzer.

Szerepe a pipeline-ban:
  - Confidence-gated: csak ha OCR confidence < threshold fut
  - Vizualis OCR validalas: "l am lnevitabIe" -> "I am inevitable"
  - Jelenet kontextus elemzes (tone, scene)
  - Nehez szoveg recovery (ferdo, kismeretu, zajos)

VRAM: Ollama szerver kezeli a modellet, a provider kliensként működik.
"""
from __future__ import annotations
import json
import logging
import re
import base64
import io
from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
from PIL import Image

from providers.base import BaseProvider, ProviderResult, ConfidenceGate
from config import cfg

logger = logging.getLogger(__name__)

VISION_PROMPT = """Analyze this comic panel image carefully.
Respond ONLY with valid JSON (no markdown, no explanation):
{
  "tone": "<angry|sad|happy|neutral|tense|comedic|dramatic|whispering>",
  "scene": "<one sentence description>",
  "ocr_correction": "<corrected text if OCR errors detected, else null>",
  "emphasis_words": ["<word1>", "<word2>"]
}"""

EXTRACTION_PROMPT = """You are a precise comic book OCR and layout analysis tool. Your task is to detect EVERY single speech bubble on the page. Do not skip any.
For each bubble, you must return the EXACT coordinates on a 0-1000 normalized scale using the format [ymin, xmin, ymax, xmax] and the text inside it.

CRITICAL INSTRUCTIONS:
- Scan the page systematically from top-left to bottom-right.
- Count the bubbles. Ensure you don't stop generating until ALL bubbles are extracted.
- Do NOT round the coordinates to the nearest hundred (e.g., do NOT output [100, 50, 300, 400] if the precise edge is at [123, 55, 294, 412]). Look closely at the exact pixel boundaries of the speech bubbles and convert them accurately to the 0-1000 scale.
- Return the data in the exact JSON format required by the schema: {"bubbles": [{"box_2d": [ymin, xmin, ymax, xmax], "text": "text"}]}

JSON ONLY. NO THINKING. NO EXPLANATION. NO IMAGE DESCRIPTION. A válaszod azonnal a { karakterrel kezdődjön! Pre-fill JSON format."""

FALLBACK_CONTEXT = {
    "tone": "neutral",
    "scene": "Comic panel scene.",
    "ocr_correction": None,
    "emphasis_words": [],
}


@dataclass
class VisionResult:
    tone:            str
    scene:           str
    ocr_correction:  Optional[str]
    emphasis_words:  list[str]
    confidence:      float = 0.85

    @classmethod
    def fallback(cls) -> "VisionResult":
        return cls(**FALLBACK_CONTEXT, confidence=0.0)


class Qwen2VLProvider(BaseProvider):
    """
    Qwen2-VL (Ollama API) multimodalis vizualis elemzo provider.
    A modell futtatása az Ollama szerveren történik.
    """
    OCR_GATE = ConfidenceGate(threshold=0.70, mode="below")

    def __init__(
        self,
        model_id: Optional[str] = None,
        enabled:  Optional[bool] = None,
    ) -> None:
        super().__init__(
            provider_id="qwen2_vl",
            min_input_confidence=0.0,
            enabled=enabled if enabled is not None else cfg.vision.enabled,
        )
        # Ollama modell név (pl. qwen3-vl:8b)
        self.model_name = cfg.vision.vlm_model_name
        self._client = None

    def _load_model(self) -> bool:
        """
        Ollama kliens inicializalása és modell létezésének ellenőrzése.
        Fix-first: itt most a próba lekérés a joint/chat endpointot használja,
        mert az Ollama /api/tags választhat formátumú (dict is lehet, de nem csak).
        A /api/chat kérés: ha 200-as a válasz, akkor biztosan elérhető a modell.
        """
        try:
            import ollama
            logger.info(f"[qwen2_vl] Betoltes: {self.model_name}...")

            # Fallback check: próbáljuk meg meghívni a modellt egy egyszerű lekérés
            self._client = ollama.Client(host=cfg.translation.base_url)

            # Egyszerű próba lekérés a modellel
            try:
                response = self._client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "OK"}],
                    stream=False,
                    options={"num_predict": 1, "temperature": 0.0},
                )
                # Ha eljutunk idáig, a modell elérhető
                logger.info(f"[qwen2_vl] Ollama kapcsolat kiépítve: {self.model_name} ✓")
                return True
            except Exception as e:
                # Ha a /api/chat hiba, akkor a modell nem elérhető
                logger.warning(
                    f"\n\n[OLLAMA WARNING] A kért modell '{self.model_name}' nem elérhető!\n"
                    f"Kérlek, töltsd le a következő paranccsal:\n"
                    f"    ollama pull {self.model_name}\n\n"
                    f"Részletek: {e}"
                )
                if os.environ.get("COMIC_VLM_SOFT_FAIL") == "1":
                    logger.info("COMIC_VLM_SOFT_FAIL=1 enabled -> qwen2_vl soft unavailable, continuing without raising")
                return False
        except ImportError:
            logger.error("A 'ollama' python csomag nem található. Kérlek futtasd: pip install ollama")
            return False
        except Exception as e:
            logger.warning(f"[qwen2_vl] Ollama kapcsolati hiba: {e}")
            if os.environ.get("COMIC_VLM_SOFT_FAIL") == "1":
                logger.info("COMIC_VLM_SOFT_FAIL=1 enabled -> qwen2_vl soft unavailable, continuing without raising")
                return True
            return False

    def _release_model(self) -> None:
        """Ollama API esetén nincs helyi VRAM kezelés, így csak pass."""
        self._client = None

    def _encode_image(self, image_rgb: np.ndarray) -> str:
        """RGB numpy array -> Base64 string."""
        img = Image.fromarray(image_rgb)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def run(
        self,
        image_rgb: np.ndarray,
        ocr_text:  str = "",
        ocr_confidence: float = 1.0,
    ) -> ProviderResult:
        """
        Vizualis elemzes + opcionales OCR korrekció Ollama API-n keresztül.
        """
        if not self.OCR_GATE.should_run(ocr_confidence) and not ocr_text:
            return ProviderResult.skip(
                f"OCR confidence OK ({ocr_confidence:.2f}) - vizualis korrekció kihagyva",
                provider_id=self.provider_id,
            )

        try:
            if self._client is None:
                return ProviderResult.fail("Ollama client nem inicializálva", provider_id=self.provider_id)

            img_b64 = self._encode_image(image_rgb)
            prompt = VISION_PROMPT
            if ocr_text:
                prompt += f"\nDetected OCR text: \"{ocr_text}\""

            try:
                response = self._client.chat(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": f"{prompt}\n<image>",
                        "images": [img_b64]
                    }],
                    format="json",
                    stream=False,
                    options={"temperature": 0.1, "num_predict": 256},
                )
            except TypeError:
                response = self._client.chat(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": f"{prompt}\n<image>",
                        "images": [img_b64]
                    }],
                    format="json",
                    stream=False,
                    options={"temperature": 0.1, "num_predict": 256},
                )

            output = response.get("message", {}).get("content", "")
            parsed = self._parse_output(output)

            return ProviderResult.ok(
                parsed,
                confidence=parsed.confidence,
                provider_id=self.provider_id,
            )
        except Exception as e:
            logger.warning(f"[qwen2_vl] run hiba: {e}")
            return ProviderResult.ok(
                VisionResult.fallback(),
                confidence=0.0,
                provider_id=self.provider_id,
            )

    def extract_text(
        self,
        image_rgb: np.ndarray,
    ) -> ProviderResult:
        """
        Extracts all text and bounding boxes from the image using visual grounding via Ollama.
        """
        try:
            if self._client is None:
                return ProviderResult.fail("Ollama client nem inicializálva", provider_id=self.provider_id)

            img_b64 = self._encode_image(image_rgb)

            try:
                response = self._client.chat(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": f"{EXTRACTION_PROMPT}\n<image>",
                        "images": [img_b64]
                    }],
                    format="json",
                    stream=False,
                    options={"temperature": 0.1, "repeat_penalty": 1.2, "num_predict": 4096},
                )
            except TypeError:
                response = self._client.chat(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": f"{EXTRACTION_PROMPT}\n<image>",
                        "images": [img_b64]
                    }],
                    format="json",
                    stream=False,
                    options={"temperature": 0.1, "repeat_penalty": 1.2, "num_predict": 4096},
                )
            logger.error(f"[Qwen2VL][DEBUG] Full Ollama response: {response}")

            content = response.get("message", {}).get("content", "") or ""
            thinking = response.get("message", {}).get("thinking", "") or ""
            done_reason = response.get("done_reason", "unknown")
            logger.info(
                f"[Qwen2VL][TEST] done_reason={done_reason} | "
                f"content_len={len(content)} | "
                f"thinking_len={len(thinking)}"
            )
            logger.info(f"[Qwen2VL][TEST] Output first 500 chars: {content[:500]}")
            parsed = self._parse_extraction_output(content)

            if parsed:
                return ProviderResult.ok(
                    parsed,
                    confidence=0.9,
                    provider_id=self.provider_id,
                )

            return ProviderResult.fail(
                f"Failed to parse VLM output as JSON: {content}",
                provider_id=self.provider_id,
            )
        except Exception as e:
            logger.warning(f"[qwen2_vl] extract_text hiba: {e}")
            return ProviderResult.fail(
                f"Unexpected error during extraction: {e}",
                provider_id=self.provider_id,
            )

    @staticmethod
    def _parse_output(text: str) -> VisionResult:
        # Strip markdown code fences if the VLM wraps JSON in ```json ... ```
        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx != -1 and end_idx != -1:
            try:
                d = json.loads(text[start_idx:end_idx + 1])
                valid_tones = {
                    "angry","sad","happy","neutral",
                    "tense","comedic","dramatic","whispering"}
                tone = str(d.get("tone","neutral")).lower()
                if tone not in valid_tones:
                    tone = "neutral"
                return VisionResult(
                    tone=tone,
                    scene=str(d.get("scene","Comic panel.")),
                    ocr_correction=d.get("ocr_correction"),
                    emphasis_words=list(d.get("emphasis_words") or []),
                    confidence=0.85,
                )
            except Exception:
                pass
        return VisionResult.fallback()

    @staticmethod
    def _parse_extraction_output(text: str) -> Optional[dict]:
        """Cleans and parses the JSON output from the VLM."""
        raw_output = text
        print(f"[Qwen2VL][DEBUG] Raw VLM output (first 500 chars): {raw_output[:500]}")
        logger.error(f"[Qwen2VL][DEBUG] Raw VLM output (first 1000 chars): {raw_output[:1000]}")

        # Strip markdown fences (```json ... ``` or ``` ... ```)
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw_output).strip()
        print(f"[Qwen2VL][DEBUG] Cleaned JSON input before parsing (first 200 chars): {cleaned[:200]}")
        logger.error(f"[Qwen2VL][DEBUG] Cleaned JSON input before parsing (first 100 chars): {cleaned[:100]}")

        start_idx = cleaned.find('{')
        if start_idx == -1:
            print("[Qwen2VL][DEBUG] No '{' found in cleaned text")
            logger.error("[Qwen2VL][DEBUG] No '{' found in cleaned text")
            return None

        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned, start_idx)
            print(f"[Qwen2VL][DEBUG] Parsed JSON object keys: {list(obj.keys())}")
            logger.error(f"[Qwen2VL][DEBUG] Parsed JSON object keys: {list(obj.keys())}")
            return obj
        except json.JSONDecodeError as e:
            err_msg = f"[qwen2_vl] JSON parse error: {e}. Text: {cleaned[:1000]}"
            print(err_msg)
            logger.error(err_msg)
        return None
