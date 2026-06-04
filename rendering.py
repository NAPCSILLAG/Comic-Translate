"""
rendering.py - Professzionális tipográfiai rendering engine.

Architektúra (szétválasztott rétegek):
  FontRegistry:       dinamikus font scanner + registry
  TypographyPreset:   tipográfiai stílus leírók (bubble/narbox/sfx)
  OCRStyleHint:       per-word stílus metaadat (jövőbeli emphasis)
  TextLayoutEngine:   layout számítás – NINCS rajzolás itt
  LayoutPlan:         egyetlen layout terv (sorok, pozíciók, score)
  Rasterizer:         RGBA réteg renderelés supersampling-gal
  ComicRenderer:      publikus API orchestrátor

Tervezési elvek:
  - Layout és raszterizáció TELJESEN szét van választva
  - Szöveg CSAK RGBA transparent layer-re kerül
  - Supersampling: 2x default, 4x config-ból
  - Score-driven, determinisztikus elhelyezés
  - Valódi glyph metrikák (font.getbbox, kerning-aware)
  - Adaptive typography: kis font = vastagabb outline + tágabb spacing
  - SFX: detektálás + skip
  - Pipeline failure resilience: buborék hiba nem törli az oldalt
  - GUI-ready: runtime font switching, preset rendszer
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import cfg
from utils import (
    BubbleGeometry, BubbleShape, BubbleAnalyzer,
    GlyphMetrics, LayoutCandidate, LayoutScorer,
    SuperSampler, ImageUtils,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SFX DETEKTOR
# ══════════════════════════════════════════════════════════════════════════════

# Ismert SFX szavak – ezeket NEM fordítjuk, NEM rendereljük
_SFX_WORDS: frozenset[str] = frozenset([
    "BOOM", "POW", "BAM", "CRASH", "BANG", "WHAM", "ZAP", "KAPOW",
    "WHOOSH", "THUD", "CRACK", "SPLASH", "RATTLE", "GRUNT", "GROAN",
    "GASP", "ROAR", "HISS", "BUZZ", "CLICK", "SNAP", "SLAM", "THUMP",
    "CREAK", "SQUEAK", "WHOMP", "KRAK", "KRASH", "SPLOOSH", "BLAM",
    "ZAPP", "KRAKOOM", "SKREEEE", "FWOOSH", "WOOOSH", "THWIP",
])

_SFX_PATTERN = re.compile(
    r"^[A-Z]{2,}[!?]+$|"              # BOOM! CRASH!! – felkiáltójel kötelező
    r"^[*~][A-Z\s]+[*~]$",            # *BOOM* ~whoosh~
    re.IGNORECASE,
)

# Hétköznapi angol szavak amik nagybetűsen is előfordulnak speech bubble-ben
# és SOHA nem SFX-ek – a pattern-szűrőből kizárjuk őket
_COMMON_WORDS: frozenset[str] = frozenset([
    "THE", "AND", "BUT", "FOR", "NOT", "YOU", "ARE", "WAS", "HAS", "HAD",
    "HIS", "HER", "ITS", "OUR", "OUT", "WHO", "ALL", "ONE", "CAN", "MAY",
    "YES", "NO", "OH", "AH", "WELL", "NOW", "JUST", "EVEN", "ALSO", "ONLY",
    "VERY", "MUCH", "MORE", "SOME", "ANY", "HOW", "WHY", "WHAT", "WHEN",
    "THEN", "THAN", "THAT", "THIS", "WITH", "FROM", "INTO", "UPON",
    "HAVE", "BEEN", "WILL", "WOULD", "COULD", "SHOULD", "MIGHT", "MUST",
    # Rövid elöljárók és névelők – SOHA nem SFX (hiányoztak!)
    "OF", "IN", "ON", "TO", "AT", "BY", "UP", "IF", "AS", "AN",
    "OR", "SO", "DO", "GO", "HE", "IT", "ME", "MY", "US", "WE",
    "AM", "IS", "BE", "HI", "OK",
    # Hétköznapi melléknevek és névmások – SOHA nem SFX (hiányoztak!)
    "SUCH", "EACH", "BOTH", "MANY", "MOST", "LESS", "ELSE",
    "OWN", "NEW", "OLD", "BIG", "BAD", "FEW", "FAR", "OFF",
    "GET", "GOT", "LET", "SAY", "SEE", "SET", "PUT", "RUN",
    "USE", "TRY", "ASK", "BUY", "CUT", "EAT", "HIT", "WIN",
    # Képregényes egyedi szavak amik nem SFX-ek
    "INDEED", "OUTSIDE", "INSIDE", "ABOVE", "BELOW", "NEVER", "EVER",
    "REALLY", "TRULY", "CLEARLY", "SIMPLY", "MERELY", "QUITE", "RATHER",
    "PERHAPS", "ALREADY", "ALWAYS", "AGAIN", "AWAY", "BACK", "DOWN",
    "GOOD", "GREAT", "SMALL", "LARGE", "LONG", "LAST", "NEXT", "SAME",
    "RIGHT", "LEFT", "STILL", "OVER", "DONE", "GONE", "COME", "KNOW",
    "LOOK", "TAKE", "MAKE", "SEEM", "NEED", "WANT", "KEEP", "HOLD",
    "WELL", "FINE", "SURE", "TRUE", "FREE", "FULL", "HIGH", "HALF",
    "FOOL", "WISE", "DEAD", "LIVE", "MIND", "SOUL", "HAND", "EYES",
    "FACE", "TIME", "LIFE", "WORD", "PLACE", "WORLD", "POWER", "ORDER",
    # Egyéb képregény-specifikus szavak
    "WAIT", "STOP", "HELP", "PLEASE", "THANK", "SORRY", "EXCUSE",
    "LISTEN", "WATCH", "HURRY", "CAREFUL", "QUICKLY", "SLOWLY",
    "AMAZING", "TERRIBLE", "WONDERFUL", "HORRIBLE", "IMPOSSIBLE",
    "WHAT", "WHERE", "WHICH", "WHILE", "WHOSE", "WHOM",
])


def is_sfx(text: str) -> bool:
    """
    Meghatározza hogy a szöveg SFX-e (hangutánzó szó).

    SFX-ek NEM kerülnek fordításra és renderelésre.

    Csak akkor SFX ha:
      1. Szerepel az ismert SFX szólistában (BOOM, POW stb.), VAGY
      2. *csillag* vagy ~hullám~ köré van zárva, VAGY
      3. Egyszavas, felkiáltójellel végződő nagybetűs szó ÉS nem
         szerepel a hétköznapi szavak listájában

    Fontos: csupa nagybetűs szöveg NEM elég az SFX detektáláshoz –
    a képregények általában csupa nagybetűvel írnak, az OCR is így adja vissza.
    """
    if not text:
        return False
    stripped = text.strip().upper()
    clean = stripped.rstrip("!?").strip()

    # 1. Ismert SFX szólista
    if clean in _SFX_WORDS:
        return True

    # 2. *csillag* / ~hullám~ wrapper
    if re.match(r"^[*~][A-Z\s]+[*~]$", stripped, re.IGNORECASE):
        return True

    # 3. Egyszavas, felkiáltójellel végződő – de CSAK ha nem hétköznapi szó
    words = stripped.split()
    if len(words) == 1 and re.match(r"^[A-Z]{2,}!+[?]?$", stripped):
        if clean not in _COMMON_WORDS:
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# OCR STÍLUS HINT – per-word emphasis metaadat
# ══════════════════════════════════════════════════════════════════════════════

class EmphasisType(Enum):
    NONE    = auto()
    BOLD    = auto()    # vastagított szó
    LARGE   = auto()    # nagyobb szó
    ITALIC  = auto()    # dőlt (jövőbeli)
    STRETCH = auto()    # nyújtott szó (jövőbeli)


@dataclass
class WordStyleHint:
    """
    Egyetlen szó stílus metaadatai az OCR-ből.

    Megőrzött a jövőbeli emphasis reconstruction-höz:
      - per-word scaling
      - mixed font weights
      - inline emphasis
      - variable line emphasis

    Jelenleg detektálás szintjén van – renderelés later.
    """
    word:           str
    emphasis:       EmphasisType = EmphasisType.NONE
    size_ratio:     float        = 1.0    # eredeti méret / átlag méret
    is_caps:        bool         = False  # ALL CAPS volt az eredetiben
    bbox_orig:      Optional[tuple[int,int,int,int]] = None  # OCR bbox


@dataclass
class OCRStyleHint:
    """
    Teljes szövegpéldány stílus metaadatai.

    A TextLayoutEngine felhasználja a layout optimalizáláshoz,
    a Rasterizer a rendereléshez (emphasis – jövőbeli).
    """
    raw_text:       str
    words:          list[WordStyleHint] = field(default_factory=list)
    dominant_caps:  bool = False    # szöveg többsége ALL CAPS?
    line_count_est: int  = 1        # OCR-ben látott sorok száma

    @classmethod
    def from_text(cls, text: str) -> "OCRStyleHint":
        """Alap stílus hint szövegből – geometry nélkül."""
        words_raw = text.split()
        words = []
        caps_count = 0
        for w in words_raw:
            is_caps = w.isupper() and len(w) > 1
            if is_caps:
                caps_count += 1
            words.append(WordStyleHint(
                word=w,
                emphasis=EmphasisType.BOLD if is_caps else EmphasisType.NONE,
                is_caps=is_caps,
            ))
        dominant_caps = caps_count > len(words_raw) / 2 if words_raw else False
        return cls(
            raw_text=text,
            words=words,
            dominant_caps=dominant_caps,
            line_count_est=max(1, text.count("\n") + 1),
        )


# ══════════════════════════════════════════════════════════════════════════════
# FONT REGISTRY – dinamikus font scanner
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FontEntry:
    """Egyetlen font leírója a registry-ben."""
    family:    str           # pl. "NotoSans"
    style:     str           # "Regular", "Bold", "Italic"
    path:      Path
    preset:    str = "bubble"   # "bubble" | "narbox" | "sfx"


class FontRegistry:
    """
    Dinamikus font scanner és registry.

    Automatikusan beolvassa:
      fonts/          → bubble preset
      fonts/narbox/   → narráció box preset

    GUI-ready: runtime font switching, preset alapú lekérés.
    Minden font cache-elve van méret szerint.
    """

    def __init__(self) -> None:
        self._entries: list[FontEntry]                          = []
        self._cache:   dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
        self._scan()

    def _scan(self) -> None:
        """Font mappák beolvasása."""
        font_dir = cfg.paths.font_dir

        # Fő font mappa → bubble preset
        self._scan_dir(font_dir, preset="bubble")

        # Narráció box almappa
        narbox_dir = font_dir / "narbox"
        if narbox_dir.exists():
            self._scan_dir(narbox_dir, preset="narbox")

        if not self._entries:
            logger.warning(
                f"Nincs font a {font_dir} mappában!\n"
                "Töltsd le: fonts/NotoSans-Bold.ttf és fonts/NotoSans-Regular.ttf\n"
                "https://fonts.google.com/noto/specimen/Noto+Sans"
            )
        else:
            names = [e.path.name for e in self._entries]
            logger.info(f"FontRegistry: {len(self._entries)} font | {names}")

    def _scan_dir(self, directory: Path, preset: str) -> None:
        """Egy mappa TTF/OTF fájljainak beolvasása."""
        if not directory.exists():
            return
        for ext in ("*.ttf", "*.TTF", "*.otf", "*.OTF"):
            for font_path in sorted(directory.glob(ext)):
                family, style = self._parse_name(font_path.stem)
                self._entries.append(FontEntry(
                    family=family,
                    style=style,
                    path=font_path,
                    preset=preset,
                ))

    @staticmethod
    def _parse_name(stem: str) -> tuple[str, str]:
        """
        Font fájlnévből family + style kinyerése.

        Példák:
          NotoSans-Bold     → ("NotoSans", "Bold")
          Arial             → ("Arial",    "Regular")
          OpenDyslexic-Bold → ("OpenDyslexic", "Bold")
        """
        style_keywords = ["Bold", "Italic", "Light", "Medium",
                          "Regular", "Black", "Thin", "ExtraBold"]
        for kw in style_keywords:
            if stem.endswith(f"-{kw}") or stem.endswith(kw):
                family = stem.replace(f"-{kw}", "").replace(kw, "").strip("-")
                return family or stem, kw
        return stem, "Regular"

    def get(
        self,
        size: int,
        bold: bool = True,
        preset: str = "bubble",
        family: Optional[str] = None,
    ) -> ImageFont.FreeTypeFont:
        """
        Font lekérés cache-sel.

        Args:
            size:   font méret pontban
            bold:   Bold stílus preferált
            preset: "bubble" | "narbox"
            family: opcionális family override (GUI-ból)

        Returns:
            PIL ImageFont.FreeTypeFont
        """
        # Font keresés
        path = self._find_font_path(bold, preset, family)
        cache_key = (str(path), size)

        if cache_key not in self._cache:
            try:
                self._cache[cache_key] = ImageFont.truetype(str(path), size)
            except (IOError, OSError):
                logger.warning(f"Font betöltési hiba: {path} – default")
                self._cache[cache_key] = ImageFont.load_default()

        return self._cache[cache_key]

    def _find_font_path(
        self,
        bold: bool,
        preset: str,
        family: Optional[str],
    ) -> Path:
        """
        Legjobb illeszkedő font keresése a registry-ben.

        Prioritás: preset → family → style → fallback
        """
        preferred_style = "Bold" if bold else "Regular"
        candidates = [e for e in self._entries if e.preset == preset]

        if not candidates:
            candidates = self._entries   # fallback: bármely font

        if family:
            fam_candidates = [e for e in candidates
                              if e.family.lower() == family.lower()]
            if fam_candidates:
                candidates = fam_candidates

        # Stílus szerint rendezés
        style_match = [e for e in candidates if e.style == preferred_style]
        if style_match:
            return style_match[0].path

        if candidates:
            return candidates[0].path

        # Abszolút fallback: config-ban lévő út
        fallback = cfg.font_path(cfg.rendering.font_bold)
        return fallback if fallback.exists() else Path(cfg.rendering.font_fallback)

    def list_families(self, preset: str = "bubble") -> list[str]:
        """Elérhető font family-k listája (GUI-hoz)."""
        return sorted({e.family for e in self._entries if e.preset == preset})

    def clear_cache(self) -> None:
        self._cache.clear()


# ══════════════════════════════════════════════════════════════════════════════
# TYPOGRAPHY PRESET
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TypographyPreset:
    """
    Tipográfiai stílus leíró egy buborék típushoz.

    Különböző preset-ek:
      bubble:   normál speech bubble
      narbox:   narráció doboz
      thought:  gondolatbuborék (jövőbeli)
    """
    name:              str
    bold:              bool  = True
    font_family:       Optional[str] = None     # None = auto
    text_color:        tuple = (15, 15, 15)
    outline_color:     tuple = (255, 255, 255)
    # Adaptive outline: kis fontnál vastagabb, nagyobb fontnál vékonyabb
    outline_width_base: int   = 2
    outline_width_min:  int   = 1
    outline_width_max:  int   = 3
    shadow_enabled:    bool  = False
    shadow_offset:     tuple = (1, 1)
    shadow_color:      tuple = (0, 0, 0)
    shadow_alpha:      float = 0.35
    line_spacing:      float = 1.22
    # Igazítás
    align:             str   = "center"   # "center" | "left" | "right"
    # Padding a bubble méretének arányában
    padding_ratio:     float = 0.12
    padding_min_px:    int   = 6

    def adaptive_outline_width(self, font_size: int) -> int:
        """
        Adaptive outline: kis font → vastagabb, nagy font → vékonyabb.

        Ez kritikus az olvashatósághoz kis méretben.
        """
        if font_size <= 12:
            return self.outline_width_max
        elif font_size >= 28:
            return self.outline_width_min
        else:
            # Lineáris interpoláció
            t = (font_size - 12) / (28 - 12)
            w = self.outline_width_max - t * (
                self.outline_width_max - self.outline_width_min)
            return max(self.outline_width_min, round(w))

    def adaptive_line_spacing(self, font_size: int) -> float:
        """
        Adaptive line spacing: kis font → tágabb, nagy → szűkebb.
        """
        if font_size <= 12:
            return self.line_spacing * 1.15
        elif font_size >= 28:
            return self.line_spacing * 0.95
        return self.line_spacing


# Beépített presetek
PRESET_BUBBLE = TypographyPreset(
    name="bubble",
    bold=True,
    text_color=(15, 15, 15),
    outline_color=(255, 255, 255),
    outline_width_base=2,
    shadow_enabled=False,
    align="center",
    padding_ratio=0.12,
)

PRESET_NARBOX = TypographyPreset(
    name="narbox",
    bold=False,
    text_color=(10, 10, 10),
    outline_color=(255, 255, 255),
    outline_width_base=1,
    outline_width_max=2,
    shadow_enabled=False,
    align="left",
    padding_ratio=0.08,
    line_spacing=1.18,
)

PRESET_THOUGHT = TypographyPreset(
    name="thought",
    bold=False,
    text_color=(30, 30, 30),
    outline_color=(255, 255, 255),
    outline_width_base=2,
    align="center",
    padding_ratio=0.14,
)

_PRESETS: dict[str, TypographyPreset] = {
    "bubble":  PRESET_BUBBLE,
    "narbox":  PRESET_NARBOX,
    "thought": PRESET_THOUGHT,
}


def get_preset(bubble_type: str) -> TypographyPreset:
    """Preset lekérés bubble típus alapján."""
    return _PRESETS.get(bubble_type, PRESET_BUBBLE)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT PLAN – egyetlen layout terv
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LayoutPlan:
    """
    Egyetlen szöveg elhelyezési terv teljes geometriával.

    Ez az objektum a TextLayoutEngine outputja.
    A Rasterizer ezt használja – nincs geometria számítás a raszterizálásban.

    Fields:
        lines:       tördelés utáni szöveg sorok
        font_size:   optimális font méret
        line_height: sor magasság px (supersampling NÉLKÜL)
        total_w:     szövegblokk teljes szélessége
        total_h:     szövegblokk teljes magassága
        start_x:     szöveg bal széle az oldalon
        start_y:     szöveg teteje az oldalon
        score:       layout minőség score (0..1)
        preset:      alkalmazott tipográfiai preset
        geom:        buborék geometria referencia
    """
    lines:       list[str]
    font_size:   int
    line_height: int
    total_w:     int
    total_h:     int
    start_x:     int
    start_y:     int
    score:       float
    preset:      TypographyPreset
    geom:        BubbleGeometry
    style_hint:  Optional[OCRStyleHint] = None


# ══════════════════════════════════════════════════════════════════════════════
# WORD WRAPPER – intelligens tördelés
# ══════════════════════════════════════════════════════════════════════════════

class WordWrapper:
    """
    Intelligens szótördelő – NEM karakter-szám alapú.

    Stratégiák:
      - Valódi pixel-szélességű sorhosszak
      - "Gyémánt" elrendezés: középső sorok leghosszabbak
      - Widow/orphan elkerülés
      - Magyar szóhatár megőrzés

    Teljesen stateless – csak tiszta függvények.
    """

    @staticmethod
    def wrap(
        text: str,
        max_width: int,
        font: ImageFont.FreeTypeFont,
        metrics: GlyphMetrics,
        shape: BubbleShape = BubbleShape.ELLIPSE,
    ) -> list[str]:
        """
        Szöveg tördelés bubble shape-aware módon.

        Args:
            text:      fordított szöveg
            max_width: maximum sor szélesség pixelben
            font:      PIL font
            metrics:   GlyphMetrics (cache-elt mérés)
            shape:     buborék alakzat (befolyásolja a tördelési stratégiát)

        Returns:
            Szöveg sorok listája.
        """
        if not text.strip():
            return []

        words = text.split()
        if not words:
            return []

        # Alap tördelés
        raw_lines = WordWrapper._greedy_wrap(words, max_width, font, metrics)

        if not raw_lines:
            return [text]

        # Shape-specifikus optimalizálás
        if shape in (BubbleShape.ELLIPSE, BubbleShape.IRREGULAR):
            raw_lines = WordWrapper._balance_diamond(
                raw_lines, max_width, font, metrics)
        elif shape == BubbleShape.TALL:
            raw_lines = WordWrapper._balance_narrow(
                raw_lines, max_width, font, metrics)

        # Widow elkerülés
        raw_lines = WordWrapper._fix_widow(raw_lines)

        return raw_lines

    @staticmethod
    def _greedy_wrap(
        words: list[str],
        max_width: int,
        font: ImageFont.FreeTypeFont,
        metrics: GlyphMetrics,
    ) -> list[str]:
        """Greedy szótördelés pixel-pontos szélességgel."""
        lines: list[str] = []
        current = ""

        for word in words:
            test = f"{current} {word}".strip() if current else word
            m    = metrics.measure_line(test, font)
            if m.width_px <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                # Egyetlen szó túl hosszú → megtartjuk így is
                current = word
        if current:
            lines.append(current)
        return lines

    @staticmethod
    def _balance_diamond(
        lines: list[str],
        max_width: int,
        font: ImageFont.FreeTypeFont,
        metrics: GlyphMetrics,
    ) -> list[str]:
        """
        'Gyémánt' elrendezés: középső sorok leghosszabbak.

        Elliptikus buborékoknál a természetes szöveg-elrendezés.
        Heurisztika: ha 3+ sor van, próbáljuk a középső sort kibővíteni.
        """
        if len(lines) < 3:
            return lines

        # Csak akkor módosítjuk ha az utolsó sor nagyon rövid
        last_m = metrics.measure_line(lines[-1], font)
        if len(lines) >= 2:
            prev_m = metrics.measure_line(lines[-2], font)
            if last_m.width_px < prev_m.width_px * 0.35:
                # Widow: utolsó szót a megelőző sorba vonjuk
                all_words = " ".join(lines).split()
                if len(all_words) > len(lines):
                    return WordWrapper._greedy_wrap(
                        all_words, max_width, font, metrics)
        return lines

    @staticmethod
    def _balance_narrow(
        lines: list[str],
        max_width: int,
        font: ImageFont.FreeTypeFont,
        metrics: GlyphMetrics,
    ) -> list[str]:
        """
        Narrow/tall buboréknál szűkebb tördelés.
        Kisebb max_width-del újratördelünk.
        """
        narrow_width = int(max_width * 0.82)
        all_words = " ".join(lines).split()
        return WordWrapper._greedy_wrap(
            all_words, narrow_width, font, metrics)

    @staticmethod
    def _fix_widow(lines: list[str]) -> list[str]:
        """
        Widow/orphan javítás: egyszavas utolsó sor elkerülése.

        Ha az utolsó sor egyetlen szó és van megelőző sor,
        átvesszük az utolsó szót a megelőző sorból.
        """
        if len(lines) < 2:
            return lines
        last = lines[-1].strip()
        if len(last.split()) == 1 and len(lines[-2].split()) > 1:
            prev_words = lines[-2].split()
            moved      = prev_words[-1]
            new_prev   = " ".join(prev_words[:-1])
            new_last   = f"{moved} {last}"
            result     = lines[:-2] + [new_prev, new_last]
            return result
        return lines


# ══════════════════════════════════════════════════════════════════════════════
# TEXT LAYOUT ENGINE – csak geometria, nulla rajzolás
# ══════════════════════════════════════════════════════════════════════════════

class TextLayoutEngine:
    """
    Szöveg elhelyezési motor – NINCS benne semmilyen rajzolás.

    Input:  szöveg + BubbleGeometry + TypographyPreset
    Output: LayoutPlan (sorok, pozíciók, score)

    A Rasterizer ezt kapja meg és csak rajzol.

    Determinisztikus: ugyanaz a bemenet → ugyanaz az output.
    Score-driven: több kandidánst generál, a legjobbat adja vissza.
    """

    def __init__(
        self,
        font_registry: FontRegistry,
        metrics: GlyphMetrics,
        ss: SuperSampler,
    ) -> None:
        self._fonts   = font_registry
        self._metrics = metrics
        self._ss      = ss
        self._wrapper = WordWrapper()

    def compute(
        self,
        text: str,
        geom: BubbleGeometry,
        preset: TypographyPreset,
        style_hint: Optional[OCRStyleHint] = None,
    ) -> Optional[LayoutPlan]:
        """
        Optimális LayoutPlan számítása.

        Algoritmus:
          1. Safe bbox meghatározás (bubble shape alapján)
          2. Font méret tartomány iterálás (binary search)
          3. Minden méretnél tördelés + LayoutCandidate
          4. Score-based kiválasztás
          5. LayoutPlan összeállítás pozíciókkal

        Args:
            text:       fordított szöveg
            geom:       BubbleGeometry
            preset:     tipográfiai preset
            style_hint: OCR stílus metaadat (opcionális)

        Returns:
            LayoutPlan vagy None ha nem fér el semmi.
        """
        if not text.strip():
            return None

        safe = geom.safe_bbox or geom.text_safe_bbox(
            preset.padding_ratio, preset.padding_min_px)
        sw = safe[2] - safe[0]
        sh = safe[3] - safe[1]

        if sw < 10 or sh < 8:
            return None

        # Dummy draw objektum a méréshez (nincs tényleges rajzolás)
        dummy_img  = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)

        candidates: list[LayoutCandidate] = []
        font_min = cfg.rendering.font_size_min
        font_max = cfg.rendering.font_size_max

        # Binary search: legnagyobb beleferő font
        lo, hi  = font_min, font_max
        best_ok: Optional[tuple[int, list[str], int, int]] = None

        while lo <= hi:
            mid  = (lo + hi) // 2
            font = self._fonts.get(mid, bold=preset.bold,
                                   preset=preset.name,
                                   family=preset.font_family)
            spacing = preset.adaptive_line_spacing(mid)
            lines   = WordWrapper.wrap(text, sw, font, self._metrics, geom.shape)
            if not lines:
                hi = mid - 1
                continue
            tw, th = self._metrics.measure_block(lines, font, spacing)
            if tw <= sw and th <= sh:
                best_ok = (mid, lines, tw, th)
                lo = mid + 1   # próbáljunk nagyobbat
            else:
                hi = mid - 1

        if best_ok is None:
            # Minimum fonttal sem fér el → kényszer wrap legkisebb mérettel
            font = self._fonts.get(font_min, bold=preset.bold,
                                   preset=preset.name)
            spacing = preset.adaptive_line_spacing(font_min)
            lines   = WordWrapper.wrap(text, sw, font, self._metrics, geom.shape)
            if not lines:
                return None
            tw, th  = self._metrics.measure_block(lines, font, spacing)
            best_ok = (font_min, lines, tw, th)

        font_size, lines, tw, th = best_ok

        # LayoutCandidate összeállítás
        cand = LayoutCandidate(
            lines=lines, font_size=font_size,
            total_width=tw, total_height=th)
        LayoutScorer.score(cand, geom)

        # Pozíció számítás (eredeti kép koordináta-terében)
        sx, sy, sx2, sy2 = safe
        center_x = (sx + sx2) // 2
        center_y = (sy + sy2) // 2
        start_x  = center_x - tw // 2
        start_y  = center_y - th // 2
        # Klip a safe területre
        start_x  = max(sx, min(start_x, sx2 - tw))
        start_y  = max(sy, min(start_y, sy2 - th))

        # line_height (supersampling NÉLKÜLI)
        font_obj  = self._fonts.get(font_size, bold=preset.bold,
                                    preset=preset.name)
        ascent, descent = font_obj.getmetrics()
        spacing_f = preset.adaptive_line_spacing(font_size)
        line_h    = int((ascent + descent) * spacing_f)

        return LayoutPlan(
            lines=lines,
            font_size=font_size,
            line_height=line_h,
            total_w=tw,
            total_h=th,
            start_x=start_x,
            start_y=start_y,
            score=cand.score,
            preset=preset,
            geom=geom,
            style_hint=style_hint,
        )


# ══════════════════════════════════════════════════════════════════════════════
# RASTERIZER – RGBA réteg renderelés
# ══════════════════════════════════════════════════════════════════════════════

class Rasterizer:
    """
    Szöveg raszterizáló – KIZÁRÓLAG RGBA transparent layer-re rajzol.

    A végleges kompozitálás a Compositor dolga.
    Supersampling: 2x (config: 4x is lehetséges).

    Fontos:
      - Nincs BGR/RGB konverzió itt
      - Nincs float↔uint8 ciklus
      - Szöveg CSAK RGBA-ra kerül, soha nem közvetlenül a forrásképre
    """

    def __init__(
        self,
        font_registry: FontRegistry,
        ss: SuperSampler,
        metrics: GlyphMetrics,
    ) -> None:
        self._fonts   = font_registry
        self._ss      = ss
        self._metrics = metrics

    def render_plan(
        self,
        plan: LayoutPlan,
        canvas_size: tuple[int, int],
    ) -> Image.Image:
        """
        LayoutPlan raszterizálása RGBA transparent layer-re.

        Args:
            plan:        TextLayoutEngine output
            canvas_size: (width, height) az oldalon

        Returns:
            RGBA PIL Image – teljes oldal méretű, átlátszó háttérrel.
            A szöveg csak a bubble területén van.
        """
        cw, ch = canvas_size

        # Supersampled canvas
        ss_w, ss_h = self._ss.upscale_canvas(cw, ch)
        ss_canvas  = Image.new("RGBA", (ss_w, ss_h), (0, 0, 0, 0))
        draw       = ImageDraw.Draw(ss_canvas)

        preset    = plan.preset
        font_size = self._ss.upscale_font_size(plan.font_size)
        font      = self._fonts.get(
            font_size, bold=preset.bold,
            preset=preset.name, family=preset.font_family)

        spacing    = preset.adaptive_line_spacing(plan.font_size)
        ascent, d  = font.getmetrics()
        line_h_ss  = int((ascent + d) * spacing)

        outline_w  = preset.adaptive_outline_width(plan.font_size)
        outline_ss = outline_w * self._ss.factor

        total_h_ss = line_h_ss * len(plan.lines)
        # Pozíció a supersampled canvas-on
        sx_ss = self._ss.upscale_coord(plan.start_x)
        sy_ss = self._ss.upscale_coord(plan.start_y)
        sw_ss = self._ss.upscale_coord(plan.total_w)

        # Árnyék (ha engedélyezett)
        if preset.shadow_enabled:
            self._draw_shadow(
                draw, plan.lines, font, sx_ss, sy_ss,
                sw_ss, line_h_ss, preset, outline_ss)

        # Sorok kirajzolása
        for i, line in enumerate(plan.lines):
            if not line.strip():
                continue
            m      = self._metrics.measure_line(
                line, self._fonts.get(plan.font_size, bold=preset.bold,
                                      preset=preset.name))
            line_w_ss = m.width_px * self._ss.factor

            # Vízszintes igazítás
            if preset.align == "center":
                text_x = sx_ss + (sw_ss - line_w_ss) // 2
            elif preset.align == "right":
                text_x = sx_ss + sw_ss - line_w_ss
            else:
                text_x = sx_ss

            text_y = sy_ss + i * line_h_ss

            # Outline (körvonal) – 8 irányban eltolva
            self._draw_outline(
                draw, line, font, text_x, text_y,
                preset.outline_color, outline_ss)

            # Fő szöveg
            draw.text(
                (text_x, text_y), line,
                font=font,
                fill=(*preset.text_color, 255),
            )

        # Lanczos downsample az eredeti méretre
        final = self._ss.downscale(ss_canvas, (cw, ch))
        return final

    @staticmethod
    def _draw_outline(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        x: int,
        y: int,
        color: tuple,
        width: int,
    ) -> None:
        """Outline rajzolás 8 irányban – gamma helyes, nincs dark halo."""
        rgba = (*color, 255)
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                if dx == 0 and dy == 0:
                    continue
                dist = abs(dx) + abs(dy)
                if dist <= width + 1:
                    draw.text((x + dx, y + dy), text, font=font, fill=rgba)

    @staticmethod
    def _draw_shadow(
        draw: ImageDraw.ImageDraw,
        lines: list[str],
        font: ImageFont.FreeTypeFont,
        sx: int, sy: int,
        sw: int, line_h: int,
        preset: TypographyPreset,
        outline_w: int,
    ) -> None:
        """Opcionális drop shadow renderelés."""
        ox = preset.shadow_offset[0] * 2
        oy = preset.shadow_offset[1] * 2
        alpha = int(preset.shadow_alpha * 255)
        shadow_color = (*preset.shadow_color, alpha)

        for i, line in enumerate(lines):
            if not line.strip():
                continue
            draw.text(
                (sx + ox, sy + i * line_h + oy),
                line, font=font, fill=shadow_color)

    def render_debug_overlay(
        self,
        plan: LayoutPlan,
        canvas_size: tuple[int, int],
    ) -> Image.Image:
        """
        Debug overlay: safe zone, text bbox, score vizualizáció.

        Csak debug módban hívódik – nem érinti a végleges kimenetet.
        """
        cw, ch = canvas_size
        overlay = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)

        # Safe bbox – zöld keret
        if plan.geom.safe_bbox:
            sx1, sy1, sx2, sy2 = plan.geom.safe_bbox
            draw.rectangle(
                [sx1, sy1, sx2, sy2],
                outline=(0, 200, 0, 180), width=1)

        # Szövegblokk – sárga keret
        tx1 = plan.start_x
        ty1 = plan.start_y
        tx2 = plan.start_x + plan.total_w
        ty2 = plan.start_y + plan.total_h
        draw.rectangle(
            [tx1, ty1, tx2, ty2],
            outline=(255, 220, 0, 200), width=1)

        # Score felirat
        try:
            debug_font = ImageFont.load_default()
            label = f"s={plan.score:.2f} f={plan.font_size}"
            draw.text(
                (tx1, ty1 - 14), label,
                font=debug_font, fill=(255, 220, 0, 220))
        except Exception:
            pass

        return overlay


# ══════════════════════════════════════════════════════════════════════════════
# COMIC RENDERER – publikus API
# ══════════════════════════════════════════════════════════════════════════════

class ComicRenderer:
    """
    Publikus rendering API – orchestrátor.

    Workflow egy buborékra:
      1. SFX detektálás → skip ha SFX
      2. BubbleGeometry elemzés
      3. Preset kiválasztás (bubble type alapján)
      4. OCRStyleHint összeállítás
      5. TextLayoutEngine.compute() → LayoutPlan
      6. Rasterizer.render_plan() → RGBA layer
      7. Debug overlay (ha debug mód)

    Hibatűrő: egy buborék hibája nem törli le az oldalt.
    """

    def __init__(self) -> None:
        self._fonts   = FontRegistry()
        self._metrics = GlyphMetrics()
        self._ss      = SuperSampler(cfg.rendering.supersample_factor)
        self._layout  = TextLayoutEngine(self._fonts, self._metrics, self._ss)
        self._raster  = Rasterizer(self._fonts, self._ss, self._metrics)
        logger.info(
            f"ComicRenderer inicializálva ✓ | "
            f"SS={cfg.rendering.supersample_factor}x | "
            f"fonts={self._fonts.list_families()}"
        )

    def render_bubble(
        self,
        image_bgr: np.ndarray,
        bubble: dict,
    ) -> Optional[tuple[Image.Image, Optional[Image.Image]]]:
        """
        Egyetlen buborék renderelése.

        Args:
            image_bgr: [H, W, 3] uint8 BGR – csak a méretért kell
            bubble:    bubble dict (translated_text, bbox, type kötelező)

        Returns:
            (text_layer_rgba, debug_layer_rgba) tuple
            text_layer_rgba: RGBA PIL Image teljes oldal méretben
            debug_layer_rgba: debug overlay (None ha nem debug mód)
            vagy None ha skip (SFX / üres szöveg / hiba)
        """
        text = bubble.get("translated_text", "").strip()
        if not text:
            return None

        # SFX detektálás – kihagyjuk
        if is_sfx(text):
            logger.debug(f"SFX kihagyva: '{text}'")
            return None

        h, w = image_bgr.shape[:2]
        bbox = bubble.get("bbox", [0, 0, w, h])
        btype = bubble.get("type", "bubble")
        preset = get_preset(btype)

        try:
            # BubbleGeometry elemzés
            geom = BubbleAnalyzer.analyze(image_bgr, bbox)

            # OCRStyleHint
            style_hint = OCRStyleHint.from_text(
                bubble.get("raw_text", text))

            # Layout számítás
            plan = self._layout.compute(text, geom, preset, style_hint)
            if plan is None:
                logger.warning(
                    f"Layout nem számítható: '{text[:30]}' [{bbox}]")
                return None

            # Raszterizálás RGBA layer-re
            text_layer = self._raster.render_plan(plan, (w, h))

            # Debug overlay
            debug_layer = None
            if cfg.pipeline.debug or cfg.rendering.debug_layout:
                debug_layer = self._raster.render_debug_overlay(
                    plan, (w, h))

            logger.debug(
                f"Render OK [{bbox}]: '{text[:25]}' "
                f"f={plan.font_size} s={plan.score:.2f} "
                f"lines={len(plan.lines)}"
            )
            return text_layer, debug_layer

        except Exception as e:
            logger.warning(
                f"Render hiba [{bbox}] '{text[:25]}': {e}")
            return None

    def render_page(
        self,
        image_bgr: np.ndarray,
        bubbles: list[dict],
    ) -> tuple[list[Image.Image], list[Optional[Image.Image]]]:
        """
        Egy oldal összes buborékjának renderelése.

        Returns:
            (text_layers, debug_layers):
              text_layers:  RGBA layer lista (bubble sorrendben)
              debug_layers: debug overlay lista (None ha nem debug)
        """
        h, w = image_bgr.shape[:2]
        text_layers:  list[Image.Image]          = []
        debug_layers: list[Optional[Image.Image]] = []

        to_render = sorted(
            [b for b in bubbles if b.get("translated_text", "").strip()],
            key=lambda b: b.get("order", 0),
        )

        rendered = 0
        skipped  = 0

        for bubble in to_render:
            result = self.render_bubble(image_bgr, bubble)
            if result is None:
                skipped += 1
                continue
            text_layer, debug_layer = result
            text_layers.append(text_layer)
            debug_layers.append(debug_layer)
            rendered += 1

        logger.info(
            f"Render kész: {rendered} buborék | "
            f"{skipped} kihagyva (SFX/üres/hiba)"
        )
        return text_layers, debug_layers

    # GUI / runtime API
    def list_fonts(self, preset: str = "bubble") -> list[str]:
        """Elérhető font family-k (GUI-hoz)."""
        return self._fonts.list_families(preset)

    def set_supersample(self, factor: int) -> None:
        """Runtime supersampling váltás (GUI-ból)."""
        self._ss = SuperSampler(factor)
        self._layout._ss = self._ss
        self._raster._ss = self._ss
        logger.info(f"Supersampling: {factor}x")

    def reload_fonts(self) -> None:
        """Font registry újratöltés (GUI live reload)."""
        self._fonts.clear_cache()
        self._fonts._entries.clear()
        self._fonts._scan()
        logger.info("FontRegistry újratöltve")


# ══════════════════════════════════════════════════════════════════════════════
# Singleton + moduláris API
# ══════════════════════════════════════════════════════════════════════════════

_renderer: Optional[ComicRenderer] = None


def get_renderer() -> ComicRenderer:
    global _renderer
    if _renderer is None:
        _renderer = ComicRenderer()
    return _renderer


def render_page(
    image_bgr: np.ndarray,
    bubbles: list[dict],
) -> tuple[list[Image.Image], list[Optional[Image.Image]]]:
    """Moduláris API: RGBA text layer lista generálása."""
    return get_renderer().render_page(image_bgr, bubbles)


def render_debug(
    image_bgr: np.ndarray,
    bubbles: list[dict],
) -> np.ndarray:
    """
    Debug helper: text layer-ek közvetlenül az image-re kompozitálva.
    Csak teszteléshez – a végleges pipeline a Compositor-t használja.
    """
    from PIL import Image as PILImage
    h, w = image_bgr.shape[:2]
    base = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    text_layers, debug_layers = render_page(image_bgr, bubbles)
    for layer in text_layers:
        base = PILImage.alpha_composite(base.convert("RGBA"), layer)
    for dl in debug_layers:
        if dl is not None:
            base = PILImage.alpha_composite(base.convert("RGBA"), dl)
    return cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)
