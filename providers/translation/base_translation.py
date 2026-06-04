"""
providers/translation/base_translation.py - Közös adatstruktúrák.

TranslationRequest és TranslationResult minden provider közös kontraktusa.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranslationRequest:
    """
    Fordítási kérés – provider-agnosztikus.

    Fields:
        source_text:    fordítandó szöveg
        source_lang:    forrás nyelv kód ("en", "ja", ...)
        target_lang:    cél nyelv kód ("hu", ...)
        tone:           hangulat kontextus ("angry", "neutral", ...)
        scene:          jelenet leírás (VLM output)
        bubble_type:    "bubble" | "narration" | "sfx"
        ocr_confidence: OCR bizonyossága (0..1)
        page_context:   előző buborékok szövege (dialógus folytonosság)
        metadata:       extra mezők (provider-specifikus hint-ek)
    """
    source_text:    str
    source_lang:    str              = "en"
    target_lang:    str              = "hu"
    tone:           str              = "neutral"
    scene:          str              = ""
    bubble_type:    str              = "bubble"
    ocr_confidence: float            = 1.0
    page_context:   list[str]        = field(default_factory=list)
    metadata:       dict             = field(default_factory=dict)


@dataclass
class TranslationResult:
    """
    Fordítási eredmény – provider-agnosztikus.

    Fields:
        translated_text:  a lefordított szöveg
        confidence:       fordítás megbízhatósága (0..1)
        provider_id:      melyik provider fordította
        was_refined:      Gemma/más által finomítva?
        original_output:  nyers provider output (debug)
        elapsed_ms:       fordítási idő
    """
    translated_text: str
    confidence:      float  = 1.0
    provider_id:     str    = "unknown"
    was_refined:     bool   = False
    original_output: str    = ""
    elapsed_ms:      float  = 0.0

    @classmethod
    def empty(cls, provider_id: str = "unknown") -> "TranslationResult":
        return cls(translated_text="", confidence=0.0,
                   provider_id=provider_id)
