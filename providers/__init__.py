"""
providers/ - Comic Translator provider ecosystem.

Hasznalat:
  from providers.translation import OllamaProvider, get_provider
  from providers.vision import Qwen2VLProvider, Gemma4Provider
  from providers.base import ProviderRegistry, ConfidenceGate, ProviderContext
"""
from .base import (
    BaseProvider, LightweightProvider,
    ProviderResult, ProviderStatus,
    ProviderRegistry, ConfidenceGate, ProviderContext,
)
__all__ = [
    "BaseProvider", "LightweightProvider",
    "ProviderResult", "ProviderStatus",
    "ProviderRegistry", "ConfidenceGate", "ProviderContext",
]
