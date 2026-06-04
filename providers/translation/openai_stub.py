"""
providers/translation/openai_stub.py - OpenAI forditas provider stub.

Jelenleg NOT IMPLEMENTED - clean error uzenettel.
Interface stabil: amikor implementalod, csak a run() metodust kell kitolteni.
"""
from __future__ import annotations
from providers.base import LightweightProvider, ProviderResult
from .base_translation import TranslationRequest, TranslationResult
import logging
logger = logging.getLogger(__name__)

class OpenAIProvider(LightweightProvider):
    """
    OpenAI API fordítás provider (stub).

    Aktivalás:
      1. pip install openai
      2. OPENAI_API_KEY kornyezeti valtozo beallitasa
      3. A run() metodus implementalasa az openai SDK-val
    """
    def __init__(self, model: str = "gpt-4o", api_key: str = "", **kwargs):
        super().__init__(provider_id="openai", enabled=False)
        self.model   = model
        self.api_key = api_key
        logger.info("[openai] Provider stub - NOT IMPLEMENTED")

    def run(self, request: TranslationRequest) -> ProviderResult:
        return ProviderResult.fail(
            "OpenAI provider nincs implementalva.\n"
            "Implementalashoz: providers/translation/openai_stub.py run() metodus.\n"
            "Szukseges: pip install openai && OPENAI_API_KEY env var.",
            provider_id=self.provider_id,
        )
