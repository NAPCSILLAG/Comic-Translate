"""
providers/vision/gemma4.py - Gemma 4 szemantikai intelligencia provider.

Szerepe a pipeline-ban (NEM determinisztikus geometria!):
  - OCR szoveg korrekció es validacio
  - Forditas finomitas (translation refinement)
  - Tone/emotion elemzes (typography hint-ekhez)
  - Szemantikai QA ellenorzes (dialógus konzisztencia)
  - Hangsúlyos szavak azonositasa

Architektúra:
  - Confidence-gated: csak alacsony OCR / forditas score eseten fut
  - Strukturalt JSON output -> renderer es orchestrator hasznalja
  - Szekvencialis VRAM: load -> run -> release
  - Ollama-n keresztul fut (gemma4 modell)

Fontos: a determinisztikus pipeline (YOLO/PPOCR/LaMa) marad a source of truth.
Gemma csak "tanacsado" reteg - nem irja felul a geometriat.
"""
from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import requests
from providers.base import LightweightProvider, ProviderResult, ConfidenceGate
from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class GemmaOCRResult:
    """OCR korrekció eredmenye."""
    corrected_text:   str
    corrections_made: list[str]    = field(default_factory=list)
    confidence:       float        = 0.85


@dataclass
class GemmaToneResult:
    """Hangulat + tipografiai hint."""
    tone:            str           = "neutral"
    emphasis_words:  list[str]     = field(default_factory=list)
    font_weight:     str           = "bold"      # "bold" | "regular" | "light"
    text_size_hint:  str           = "normal"    # "normal" | "large" | "small"
    confidence:      float         = 0.85


@dataclass
class GemmaRefinementResult:
    """Forditas finomitas eredmenye."""
    refined_text:     str
    was_changed:      bool         = False
    change_reason:    str          = ""
    confidence:       float        = 0.85


@dataclass
class GemmaQAResult:
    """Szemantikai QA ellenorzes eredmenye."""
    passed:           bool         = True
    issues:           list[str]    = field(default_factory=list)
    suggestions:      list[str]    = field(default_factory=list)
    confidence:       float        = 0.85


# Prompt templates
_OCR_CORRECTION_PROMPT = """You are an expert OCR error corrector for comic book text.
Fix any OCR errors in the following text. Common errors: l/I confusion, 0/O confusion, rn/m confusion.
Respond ONLY with JSON: {{"corrected": "<fixed text>", "corrections": ["<what was fixed>"]}}
Text to fix: "{text}"
"""

_TONE_PROMPT = """Analyze the emotion and typography hints for this comic dialogue:
Text: "{text}"
Context: {context}
Respond ONLY with JSON:
{{"tone": "<angry|sad|happy|neutral|tense|comedic|dramatic|whispering>",
  "emphasis_words": ["<word>"],
  "font_weight": "<bold|regular|light>",
  "text_size_hint": "<normal|large|small>"}}
"""

_REFINEMENT_PROMPT = """You are a Hungarian comic book translator QA editor.
Review this translation and improve if needed. Keep it natural and concise.
Original (English): "{source}"
Current translation (Hungarian): "{translation}"
Context/tone: {tone}
Respond ONLY with JSON:
{{"refined": "<improved Hungarian text or same if OK>",
  "changed": <true|false>,
  "reason": "<why changed or empty string>"}}
"""

_QA_PROMPT = """Check this comic dialogue translation for quality issues.
English: "{source}"
Hungarian: "{translation}"
Respond ONLY with JSON:
{{"passed": <true|false>,
  "issues": ["<issue>"],
  "suggestions": ["<suggestion>"]}}
"""


class Gemma4Provider(LightweightProvider):
    """
    Gemma 4 szemantikai intelligencia provider Ollama-n keresztul.

    Lightweight: API hivas, nincs direkt VRAM foglalás.
    Az Ollama kezeli a Gemma 4 modell lifecycle-jat.

    Konfiguralható:
      - OCR korrekció mód
      - Forditas finomitas mód
      - Tone elemzes mód
      - QA mód
    """

    # Confidence gate-ek
    OCR_GATE         = ConfidenceGate(threshold=0.70, mode="below")
    TRANSLATION_GATE = ConfidenceGate(threshold=0.75, mode="below")

    def __init__(
        self,
        model:      str  = "gemma3:12b",
        base_url:   str  = "",
        enabled:    bool = True,
        ocr_correction_enabled:    bool = True,
        refinement_enabled:        bool = True,
        tone_analysis_enabled:     bool = True,
        qa_enabled:                bool = False,  # lassabb, opcionalis
        timeout:    int  = 60,
    ) -> None:
        super().__init__(provider_id="gemma4", enabled=enabled)
        self.model    = model
        self.base_url = base_url or cfg.translation.base_url
        self.timeout  = timeout
        self.ocr_correction_enabled = ocr_correction_enabled
        self.refinement_enabled     = refinement_enabled
        self.tone_analysis_enabled  = tone_analysis_enabled
        self.qa_enabled             = qa_enabled

        if enabled:
            self._check_model()

    def _check_model(self) -> None:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                mbase  = self.model.split(":")[0]
                if any(mbase in m for m in models):
                    logger.info(f"[gemma4] Modell elerheto: {self.model} ok")
                else:
                    logger.warning(
                        f"[gemma4] Modell nem talalhato: {self.model}\n"
                        f"  Letoltes: ollama pull {self.model}")
        except Exception:
            logger.warning(f"[gemma4] Ollama nem elheto: {self.base_url}")

    def run(self, *args, **kwargs) -> ProviderResult:
        """Altalanos run - hasznald a specifikus metodusokat."""
        return ProviderResult.fail(
            "Hasznald: correct_ocr(), analyze_tone(), refine_translation(), qa_check()",
            provider_id=self.provider_id,
        )

    def correct_ocr(
        self,
        text: str,
        confidence: float = 1.0,
    ) -> ProviderResult:
        """
        OCR hibak korrekcioja.
        Csak alacsony confidence eseten fut (OCR_GATE).
        """
        if not self.ocr_correction_enabled:
            return ProviderResult.skip("OCR correction disabled", self.provider_id)
        if not self.OCR_GATE.should_run(confidence):
            return ProviderResult.skip(
                f"OCR confidence OK ({confidence:.2f})", self.provider_id)

        prompt = _OCR_CORRECTION_PROMPT.format(text=text)
        raw    = self._ollama_call(prompt)
        if not raw:
            return ProviderResult.fail("Ollama hivas sikertelen", self.provider_id)

        try:
            d = self._parse_json(raw)
            result = GemmaOCRResult(
                corrected_text=d.get("corrected", text),
                corrections_made=d.get("corrections", []),
            )
            changed = result.corrected_text.strip() != text.strip()
            if changed:
                logger.debug(
                    f"[gemma4] OCR korrekció: '{text[:30]}' -> '{result.corrected_text[:30]}'")
            return ProviderResult.ok(result, confidence=0.85, provider_id=self.provider_id)
        except Exception as e:
            logger.debug(f"[gemma4] OCR parse hiba: {e}")
            return ProviderResult.ok(
                GemmaOCRResult(corrected_text=text),
                confidence=0.5, provider_id=self.provider_id)

    def analyze_tone(
        self,
        text: str,
        context: str = "",
    ) -> ProviderResult:
        """Hangulat es tipografiai hint elemzes."""
        if not self.tone_analysis_enabled:
            return ProviderResult.skip("Tone analysis disabled", self.provider_id)

        prompt = _TONE_PROMPT.format(text=text, context=context or "comic panel")
        raw    = self._ollama_call(prompt)
        if not raw:
            return ProviderResult.ok(
                GemmaToneResult(), confidence=0.0, provider_id=self.provider_id)

        try:
            d = self._parse_json(raw)
            valid = {"angry","sad","happy","neutral","tense","comedic","dramatic","whispering"}
            tone  = d.get("tone","neutral").lower()
            if tone not in valid:
                tone = "neutral"
            result = GemmaToneResult(
                tone=tone,
                emphasis_words=list(d.get("emphasis_words") or []),
                font_weight=d.get("font_weight","bold"),
                text_size_hint=d.get("text_size_hint","normal"),
            )
            return ProviderResult.ok(result, confidence=0.85, provider_id=self.provider_id)
        except Exception as e:
            logger.debug(f"[gemma4] tone parse hiba: {e}")
            return ProviderResult.ok(
                GemmaToneResult(), confidence=0.0, provider_id=self.provider_id)

    def refine_translation(
        self,
        source: str,
        translation: str,
        tone: str = "neutral",
        translation_confidence: float = 1.0,
    ) -> ProviderResult:
        """
        Forditas finomitas.
        Csak alacsony forditas confidence eseten fut.
        """
        if not self.refinement_enabled:
            return ProviderResult.skip("Refinement disabled", self.provider_id)
        if not self.TRANSLATION_GATE.should_run(translation_confidence):
            return ProviderResult.skip(
                f"Translation confidence OK ({translation_confidence:.2f})",
                self.provider_id)

        prompt = _REFINEMENT_PROMPT.format(
            source=source, translation=translation, tone=tone)
        raw = self._ollama_call(prompt)
        if not raw:
            return ProviderResult.ok(
                GemmaRefinementResult(refined_text=translation),
                confidence=0.0, provider_id=self.provider_id)

        try:
            d = self._parse_json(raw)
            refined  = d.get("refined", translation)
            changed  = bool(d.get("changed", False))
            reason   = str(d.get("reason",""))
            if changed:
                logger.debug(
                    f"[gemma4] Forditas finomitas: '{translation[:30]}' -> '{refined[:30]}'")
            return ProviderResult.ok(
                GemmaRefinementResult(
                    refined_text=refined,
                    was_changed=changed,
                    change_reason=reason,
                ),
                confidence=0.85, provider_id=self.provider_id)
        except Exception as e:
            logger.debug(f"[gemma4] refinement parse hiba: {e}")
            return ProviderResult.ok(
                GemmaRefinementResult(refined_text=translation),
                confidence=0.0, provider_id=self.provider_id)

    def qa_check(
        self,
        source: str,
        translation: str,
    ) -> ProviderResult:
        """Szemantikai QA ellenorzes (opcionalis, lassabb)."""
        if not self.qa_enabled:
            return ProviderResult.skip("QA disabled", self.provider_id)

        prompt = _QA_PROMPT.format(source=source, translation=translation)
        raw    = self._ollama_call(prompt)
        if not raw:
            return ProviderResult.ok(
                GemmaQAResult(), confidence=0.0, provider_id=self.provider_id)

        try:
            d = self._parse_json(raw)
            result = GemmaQAResult(
                passed=bool(d.get("passed", True)),
                issues=list(d.get("issues") or []),
                suggestions=list(d.get("suggestions") or []),
            )
            if not result.passed:
                logger.debug(f"[gemma4] QA failed: {result.issues}")
            return ProviderResult.ok(result, confidence=0.85, provider_id=self.provider_id)
        except Exception as e:
            logger.debug(f"[gemma4] QA parse hiba: {e}")
            return ProviderResult.ok(GemmaQAResult(), confidence=0.0, provider_id=self.provider_id)

    def _ollama_call(self, prompt: str, retries: int = 1) -> Optional[str]:
        """Egyszerü Ollama API hivas."""
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model":  self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 256},
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json().get("response","").strip()
            except Exception as e:
                if attempt < retries:
                    time.sleep(0.5)
                else:
                    logger.debug(f"[gemma4] Ollama hivas hiba: {e}")
        return None

    @staticmethod
    def _parse_json(text: str) -> dict:
        """JSON kinyerese a modell outputjabol."""
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {}
