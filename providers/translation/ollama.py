"""
providers/translation/ollama.py - Ollama forditas provider.

Teljes implementacio: qwen2.5:14b (vagy barmely Ollama modell).
A meglevo translator_engine.py prompt template-jeit hasznalja.
Lightweight provider: nincs VRAM foglalás (API call).
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

import requests

from config import cfg
from providers.base import LightweightProvider, ProviderResult
from .base_translation import TranslationRequest, TranslationResult

logger = logging.getLogger(__name__)

# Prompt templates

SYSTEM_PROMPT = """Te egy tapasztalt képregény-fordító vagy. Angolból magyarra fordítász speech bubble szövegeket.

SZIGORÚ SZABÁLYOK:
1. CSAK a lefordított szöveget add vissza. Semmi más.
2. Tegező forma KÖTELEZŐ (te, neked, veled).
3. Természetes, folyékony magyar - nem szó szerinti fordítás.
4. Felkiáltójelek, kérdőjelek, ... megmaradnak pontosan.
5. Ha az eredeti ALL CAPS, a fordítás is hangsúlyos legyen.
6. OCR hibás bemenelnél javítsd ki fordítás közben.
7. SOHA ne kezdd: "Fordítás:", "Magyar:", "Íme", stb.

PÉLDÁK:
  "I TAKE THIS TO MEAN MY OFFER HAS BEEN REJECTED?"
  -> "EZT ÚGY ÉRTEM, HOGY AZ AJÁNLATOMAT VISSZAUTASÍTOTTÁD?"

  "As ever, thanks to your keen intellect..."
  -> "Mint mindig... az éles eszed most is lenyügöz."

  "Right the first time."
  -> "Első nekifutásra."
"""

USER_TEMPLATE = """Jelénet: {scene}
Hangulat: {tone}
Buborék típus: {bubble_type}
{context_block}
Fordítsd angolról magyarra:
"{source_text}"

Csak a fordítást írd:"""

TONE_MAP = {
    "angry":      "dühös, indulatos",
    "sad":        "szomorú, szívszoritó",
    "happy":      "vidám, lelkes",
    "neutral":    "semleges",
    "tense":      "feszült, drámai",
    "comedic":    "humoros, könnyű",
    "dramatic":   "drámai, melodramatikus",
    "whispering": "halk, suttogó",
}


class OllamaProvider(LightweightProvider):
    """
    Ollama API alapú fordítás provider.

    Nem foglal VRAM-ot - az Ollama külső folyamatként fut.
    Az Ollama saját maga kezeli a modell lifecycle-ját.
    """

    def __init__(
        self,
        model:       Optional[str]   = None,
        base_url:    Optional[str]   = None,
        temperature: Optional[float] = None,
        max_tokens:  Optional[int]   = None,
        timeout:     Optional[int]   = None,
        retries:     int             = 2,
    ) -> None:
        super().__init__(provider_id="ollama", enabled=True)
        self.model       = model       or cfg.translation.model
        self.base_url    = base_url    or cfg.translation.base_url
        self.temperature = temperature or cfg.translation.temperature
        self.max_tokens  = max_tokens  or cfg.translation.max_tokens
        self.timeout     = timeout     or cfg.translation.timeout_sec
        self.retries     = retries
        self._check_connection()

    def _check_connection(self) -> None:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=4)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                model_base = self.model.split(":")[0]
                if any(model_base in m for m in models):
                    logger.info(f"[ollama] OK - modell: {self.model} ok")
                else:
                    logger.warning(
                        f"[ollama] Modell nem talalhato: {self.model}\n"
                        f"  Futtasd: ollama pull {self.model}")
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"[ollama] Nem elheto: {self.base_url}\n"
                "  Inditas: ollama serve")

    def run(self, request: TranslationRequest) -> ProviderResult:
        if not request.source_text.strip():
            return ProviderResult.ok(
                TranslationResult.empty(self.provider_id),
                provider_id=self.provider_id)

        messages  = self._build_messages(request)
        last_err  = ""

        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model":    self.model,
                        "messages": messages,
                        "stream":   False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                            "stop": ["\n\n", "###"],
                        },
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                raw     = resp.json().get("message", {}).get("content", "").strip()
                cleaned = self._post_process(raw, request.source_text)
                return ProviderResult.ok(
                    TranslationResult(
                        translated_text=cleaned,
                        confidence=0.85,
                        provider_id=self.provider_id,
                        original_output=raw,
                    ),
                    confidence=0.85,
                    provider_id=self.provider_id,
                )
            except requests.exceptions.Timeout:
                last_err = f"timeout (attempt {attempt+1})"
                logger.warning(f"[ollama] {last_err}")
                if attempt < self.retries:
                    time.sleep(1.0)
            except requests.exceptions.ConnectionError as e:
                last_err = f"connection error: {e}"
                logger.error(f"[ollama] {last_err}")
                break
            except Exception as e:
                last_err = str(e)
                if attempt < self.retries:
                    time.sleep(0.5)

        return ProviderResult.fail(last_err, provider_id=self.provider_id)

    def _build_messages(self, req: TranslationRequest) -> list[dict]:
        tone_hint = TONE_MAP.get(req.tone, "semleges")
        scene     = req.scene or "Kepregeny panel"
        ctx_block = ""
        if req.page_context:
            ctx_block = "Elozo szovegek:\n" + "\n".join(
                f"  - {t}" for t in req.page_context[-3:]) + "\n"
        user_content = USER_TEMPLATE.format(
            scene=f"{scene} ({tone_hint})",
            tone=tone_hint,
            bubble_type=req.bubble_type,
            context_block=ctx_block,
            source_text=req.source_text,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

    @staticmethod
    def _post_process(translated: str, original: str) -> str:
        if not translated:
            return ""
        translated = translated.strip()
        for q in ['"', "'", "\u201e", "\u201d"]:
            if len(translated) > 2 and translated[0] == q and translated[-1] == q:
                translated = translated[1:-1].strip()
                break
        prefixes = ["Magyar forditas:", "Forditas:", "Hungarian:","A forditas:"]
        for prefix in prefixes:
            if translated.lower().startswith(prefix.lower()):
                translated = translated[len(prefix):].strip()
                break
        lines = [ln.strip() for ln in translated.split("\n") if ln.strip()]
        clean = []
        for line in lines:
            if any(m in line.lower() for m in ["megjegyzes:", "note:"]):
                break
            clean.append(line)
        translated = " ".join(clean) if clean else (lines[0] if lines else "")
        if translated.strip().lower() == original.strip().lower():
            return ""
        return translated.strip()
