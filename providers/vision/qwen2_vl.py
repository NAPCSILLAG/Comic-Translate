"""
providers/vision/qwen2_vl.py - Qwen2-VL vizualis OCR validator + context analyzer.

Szerepe a pipeline-ban:
  - Confidence-gated: csak ha OCR confidence < threshold fut
  - Vizualis OCR validalas: "l am lnevitabIe" -> "I am inevitable"
  - Jelenet kontextus elemzes (tone, scene)
  - Nehez szoveg recovery (ferdo, kismeretu, zajos)

Szekvencialis VRAM: load -> run -> release (orchestrator kezeli)
"""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional
import numpy as np
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
    Qwen2-VL multimodalis vizualis elemzo provider.

    VRAM: ~4.5GB float16-ban (RTX 4070 kompatibilis)
    Szekvencialis: az orchestrator load/release-eli.
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
            enabled=enabled if enabled is not None else cfg.vlm.enabled,
        )
        self.model_id = model_id or cfg.vlm.vlm_model_id
        self._model     = None
        self._processor = None
        self._proc_vi   = None

    def _load_model(self) -> bool:
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            import torch
            logger.info(f"[qwen2_vl] Betoltes: {self.model_id}")
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=cfg.device.torch_dtype,
                device_map=cfg.device.device,
            )
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True)
            self._proc_vi   = process_vision_info
            logger.info("[qwen2_vl] Betoltve ok")
            return True
        except ImportError as e:
            logger.warning(f"[qwen2_vl] Hianyzo csomag: {e}")
            return False
        except Exception as e:
            logger.warning(f"[qwen2_vl] Betoltesi hiba: {e}")
            return False

    def _release_model(self) -> None:
        self._model     = None
        self._processor = None
        self._proc_vi   = None

    def run(
        self,
        image_rgb: np.ndarray,
        ocr_text:  str = "",
        ocr_confidence: float = 1.0,
    ) -> ProviderResult:
        """
        Vizualis elemzes + opcionales OCR korrekció.

        Args:
            image_rgb:       [H,W,3] uint8 RGB
            ocr_text:        az OCR altal felismert szoveg
            ocr_confidence:  OCR confidence (0..1)

        Returns:
            ProviderResult[VisionResult]
        """
        if not self.OCR_GATE.should_run(ocr_confidence) and not ocr_text:
            return ProviderResult.skip(
                f"OCR confidence OK ({ocr_confidence:.2f}) - vizualis korrekció kihagyva",
                provider_id=self.provider_id,
            )

        try:
            from PIL import Image
            pil_img = Image.fromarray(image_rgb)
            prompt  = VISION_PROMPT
            if ocr_text:
                prompt += f"\nDetected OCR text: \"{ocr_text}\""

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text",  "text": prompt},
                ],
            }]
            import torch
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self._proc_vi(messages)
            inputs = self._processor(
                text=[text], images=image_inputs,
                videos=video_inputs, padding=True,
                return_tensors="pt",
            ).to(cfg.device.device)

            with torch.no_grad():
                gen_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=cfg.vlm.max_new_tokens,
                    do_sample=False,
                )
            output = self._processor.batch_decode(
                gen_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0]

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

    @staticmethod
    def _parse_output(text: str) -> VisionResult:
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
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
