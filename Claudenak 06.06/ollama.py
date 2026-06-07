"""
providers/translation/ollama.py - Ollama translation provider.

Full implementation compatible with any Ollama model.
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


SYSTEM_PROMPT = """Te egy tapasztalt képregény-fordító vagy. Angolból magyarra fordítasz.

SZIGORÚ SZABÁLYOK:
1. A válaszod KIZÁRÓLAG egy érvényes JSON objektum lehet, semmi más!
2. A JSON struktúrája pontosan ez legyen: {"translation": "itt a lefordított magyar szöveg"}
3. SOHA ne írj magyarázatot, markdown kódblokkot vagy egyéb szöveget a JSON köré.
4. Ha nem tudod lefordítani, add vissza az eredeti angol szöveget a JSON-ben.
5. Tegező forma KÖTELEZŐ (te, neked, veled). Természetes, folyékony magyar.
"""

USER_TEMPLATE = """Jelenet: {scene}
Hangulat: {tone}
Buborék típus: {bubble_type}
{context_block}
Fordítsd angolról magyarra a következő szöveget, és add vissza a kért JSON formátumban:

"{source_text}"
"""

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
                if self.model in models:
                    logger.info(f"[ollama] Modell elérhető: {self.model}")
                else:
                    logger.warning(
                        f"[ollama] Modell nem található: {self.model}"
                    )
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"[ollama] Nem elérhető: {self.base_url}"
            )

    @staticmethod
    def _build_messages(req: TranslationRequest) -> list[dict]:
        tone_hint = TONE_MAP.get(req.tone, "semleges")
        scene     = req.scene or "Képregény panel"
        ctx_block = ""
        if req.page_context:
            ctx_block = "Előző szövegek:\n" + "\n".join(
                f"  - {t}" for t in req.page_context[-3:]
            ) + "\n"

        words = req.source_text.strip().split()
        if len(words) == 1:
            user_content = (
                f"{ctx_block}\n"
                "Kérlek fordítsd le magyarra ezt az EGYETLEN SZÓT, "
                f"ami egy képregény szövegbuborékból származik ({tone_hint} hangulat).\n"
                "A választ KIZÁRÓLAG JSON formátumban add meg: "
                '{"translation": "lefordított szó"}\n\n'
                f"Szó: {req.source_text}"
            )
        else:
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

    def run(self, request: TranslationRequest) -> ProviderResult:
        if not request.source_text.strip():
            return ProviderResult.ok(
                TranslationResult.empty(self.provider_id),
                provider_id=self.provider_id,
            )

        messages = self._build_messages(request)
        last_err = ""

        for attempt in range(self.retries + 1):
            try:
                return self._call_chat(messages, request.source_text)
            except RuntimeError as exc:
                last_err = str(exc)
                logger.warning(f"[ollama] {last_err}")
                if attempt < self.retries:
                    time.sleep(0.5)
            except Exception as exc:
                last_err = str(exc)
                if attempt < self.retries:
                    time.sleep(0.5)

        return ProviderResult.fail(last_err, provider_id=self.provider_id)

    def _call_chat(self, messages: list[dict], source_text: str) -> ProviderResult:
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "format":   "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Ollama connection error: {exc}") from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(f"Ollama request timeout") from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama request error: {exc}") from exc

        if resp.status_code == 404:
            logger.warning("[ollama] /api/chat 404 -> /api/generate")
            return self._call_generate_fallback(messages, source_text)

        resp.raise_for_status()
        raw = (resp.json().get("message") or {}).get("content", "").strip()
        return self._create_result(raw, source_text)

    def _call_generate_fallback(self, messages: list[dict], source_text: str) -> ProviderResult:
        prompt = "\n\n".join(
            f"[{message.get('role', 'user').upper()}]\n{message.get('content', '')}"
            for message in messages
        )
        payload = {
            "model":    self.model,
            "prompt":   prompt,
            "stream":   False,
            "format":   "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        raw = (resp.json().get("response") or "").strip()
        return self._create_result(raw, source_text)

    def _create_result(self, raw: str, source_text: str) -> ProviderResult:
        cleaned = self._post_process(raw, source_text)

        if not cleaned:
            logger.warning(f"[ollama] Failed, fallback original: {source_text}")
            cleaned = source_text

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

    @staticmethod
    def _post_process(translated: str, original: str) -> str:
        """JSON response parser with aggressive markdown cleanup."""
        if not translated:
            return ""

        text = translated.strip()

        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    text = text[start:end + 1]

        try:
            data = __import__("json").loads(text)
            parsed = data.get("translation", "").strip()

            if parsed and parsed.lower() != original.strip().lower():
                return parsed
            return ""
        except Exception as exc:
            logger.warning(f"[ollama] decode error: {exc}")
            return ""
