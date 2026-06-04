"""
providers/translation/deepl_stub.py - DeepL forditas provider stub.

Jelenleg NOT IMPLEMENTED - clean error uzenettel.
"""
from __future__ import annotations
from providers.base import LightweightProvider, ProviderResult
from .base_translation import TranslationRequest, TranslationResult
import logging
logger = logging.getLogger(__name__)

class DeepLProvider(LightweightProvider):
    """
    DeepL API fordítás provider (stub).

    Aktivalás:
      1. pip install deepl
      2. DEEPL_API_KEY kornyezeti valtozo beallitasa
      3. A run() metodus implementalasa a deepl SDK-val
    """
    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(provider_id="deepl", enabled=False)
        self.api_key = api_key
        logger.info("[deepl] Provider stub - NOT IMPLEMENTED")

    def run(self, request: TranslationRequest) -> ProviderResult:
        return ProviderResult.fail(
            "DeepL provider nincs implementalva.\n"
            "Implementalashoz: providers/translation/deepl_stub.py run() metodus.\n"
            "Szukseges: pip install deepl && DEEPL_API_KEY env var.",
            provider_id=self.provider_id,
        )
