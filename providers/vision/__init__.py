"""
providers/vision/ - Vizualis AI providerek.

  Qwen2VLProvider  -> OCR validator + visual context
  Gemma4Provider   -> OCR correction + tone + translation refinement + QA
"""
from .qwen2_vl import Qwen2VLProvider, VisionResult
from .gemma4 import (
    Gemma4Provider,
    GemmaOCRResult,
    GemmaToneResult,
    GemmaRefinementResult,
    GemmaQAResult,
)
__all__ = [
    "Qwen2VLProvider", "VisionResult",
    "Gemma4Provider", "GemmaOCRResult",
    "GemmaToneResult", "GemmaRefinementResult", "GemmaQAResult",
]
