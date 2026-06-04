"""
providers/translation/ - Fordítás providerek.

Elérhető:
  OllamaProvider    → teljes implementáció (qwen2.5:14b)
  OpenAIProvider    → stub + clean error
  DeepLProvider     → stub + clean error

Publikus API:
  TranslationRequest  – bemenet
  TranslationResult   – kimenet
  get_provider(name)  – factory
"""

from .base_translation import TranslationRequest, TranslationResult
from .ollama import OllamaProvider
from .openai_stub import OpenAIProvider
from .deepl_stub import DeepLProvider

__all__ = [
    "TranslationRequest",
    "TranslationResult",
    "OllamaProvider",
    "OpenAIProvider",
    "DeepLProvider",
    "get_provider",
]


def get_provider(name: str, **kwargs):
    """
    Translation provider factory.

    Args:
        name: "ollama" | "openai" | "deepl"
        **kwargs: provider-specifikus konfiguráció

    Returns:
        BaseProvider implementáció.
    """
    providers = {
        "ollama":  OllamaProvider,
        "openai":  OpenAIProvider,
        "deepl":   DeepLProvider,
    }
    cls = providers.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Ismeretlen translation provider: '{name}'\n"
            f"Elérhető: {list(providers.keys())}"
        )
    return cls(**kwargs)
