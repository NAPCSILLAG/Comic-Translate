"""
utils.py - Közös segédeszközök a Comic Translator pipeline-hoz.

Tartalom:
  1. MaskOps          – maszk műveletek (dilation, feather, distance field)
  2. BubbleAnalyzer   – bubble shape classification + geometry
  3. LayoutScorer     – score-driven layout kiértékelés
  4. SuperSampler     – 2x/4x supersampling + Lanczos downsample
  5. GlyphMetrics     – valódi glyph méretek (kerning-aware)
  6. ImageUtils       – általános képfeldolgozó segédfüggvények

Tervezési elvek:
  - Minden függvény pure (nincs side effect) ahol lehetséges
  - Resolution independent: relatív mértékek, nem abszolút pixelek
  - Extensible: új feature = új osztály, nem módosítás
  - Determinisztikus: nincs random viselkedés
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. MASZK MŰVELETEK
# ══════════════════════════════════════════════════════════════════════════════

class MaskOps:
    """
    Maszk generálás, kiterjesztés és simítás.

    Minden metódus statikus – nincs állapot, könnyen tesztelhető.
    A maszk mindig uint8 numpy tömb: 0 = megtartás, 255 = inpaint.
    """

    @staticmethod
    def from_bbox(
        image_shape: tuple[int, int],
        bbox: list[int],
        padding: int = 0,
    ) -> np.ndarray:
        """Téglalap maszk bounding box-ból.

        Clamp + validáció: érvénytelen bbox → üres (all-zero) maszk,
        nem crash és nem hibás területfestés.
        """
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        mask = np.zeros((h, w), dtype=np.uint8)
        # Érvénytelen vagy üres bbox → üres maszk (nincs crash)
        if x2 <= x1 or y2 <= y1:
            return mask
        mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def from_text_detection(
        image: np.ndarray,
        bbox: list[int],
        dilation_px: int = 12,
        extra_dilation: int = 6,
    ) -> np.ndarray:
        """
        Szöveg maszk generálás adaptív threshold + morfológia alapján.

        Két lépéses dilation:
          1. Alap dilation: összekötjük a szöveg pixeleket
          2. Extra dilation: fedik az anti-aliased széleket
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros((h, w), dtype=np.uint8)

        region = image[y1:y2, x1:x2]
        gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Adaptív threshold – sötét szöveg fehér háttéren
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=13, C=7,
        )

        # 1. lépés: szöveg pixelek összekötése
        k1 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k1, iterations=2)
        dilated1 = cv2.dilate(closed, k1, iterations=1)

        # 2. lépés: anti-aliased szélek lefedése
        if extra_dilation > 0:
            k2 = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (extra_dilation, extra_dilation))
            dilated2 = cv2.dilate(dilated1, k2, iterations=1)
        else:
            dilated2 = dilated1

        full = np.zeros((h, w), dtype=np.uint8)
        full[y1:y2, x1:x2] = dilated2
        return full

    @staticmethod
    def feather(mask: np.ndarray, radius: int = 8) -> np.ndarray:
        """
        Maszk széleinek elmosása (feathering) – varrat elkerüléséhez.

        Gaussian blur alapú alpha átmenet a maszk határán.
        """
        if radius <= 0:
            return mask
        blurred = cv2.GaussianBlur(
            mask.astype(np.float32),
            (radius * 2 + 1, radius * 2 + 1),
            radius / 2,
        )
        return np.clip(blurred, 0, 255).astype(np.uint8)

    @staticmethod
    def distance_field(mask: np.ndarray) -> np.ndarray:
        """
        Euklideszi distance field a maszk belsejéből a szélekig.

        Returns:
            float32 tömb: minden pixel távolsága a legközelebbi
            maszk szélétől (pixelben). 0.0 = szél, max = belső közép.
        """
        binary = (mask > 127).astype(np.uint8)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        return dist.astype(np.float32)

    @staticmethod
    def combine(masks: list[np.ndarray], mode: str = "union") -> np.ndarray:
        """Több maszk kombinálása (union / intersection)."""
        if not masks:
            raise ValueError("Legalább egy maszk szükséges")
        result = masks[0].copy()
        for m in masks[1:]:
            if mode == "union":
                result = np.maximum(result, m)
            else:
                result = np.minimum(result, m)
        return result

    @staticmethod
    def alpha_blend(
        base: np.ndarray,
        overlay: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        High-quality alpha compositing.

        Args:
            base:    háttér BGR kép
            overlay: előtér BGR kép
            mask:    uint8 alpha maszk (0=base, 255=overlay)

        Returns:
            Kompozit BGR kép.
        """
        alpha = mask.astype(np.float32) / 255.0
        if alpha.ndim == 2:
            alpha = alpha[:, :, np.newaxis]
        result = base.astype(np.float32) * (1 - alpha) + \
                 overlay.astype(np.float32) * alpha
        return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUBBLE SHAPE ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class BubbleShape(Enum):
    ELLIPSE     = auto()   # kerek/ovális buborék
    RECTANGLE   = auto()   # szögletes narráció doboz
    TALL        = auto()   # magas, keskeny buborék
    WIDE        = auto()   # széles, alacsony buborék
    IRREGULAR   = auto()   # szabálytalan (tüskés, kézi rajz)
    UNKNOWN     = auto()


@dataclass
class BubbleGeometry:
    """Egy speech bubble teljes geometriai leírása."""
    bbox:          list[int]          # [x1, y1, x2, y2]
    shape:         BubbleShape        # alakzat típus
    aspect_ratio:  float              # width / height
    fill_ratio:    float              # polygon terület / bbox terület
    centroid:      tuple[float, float]  # (cx, cy) relatív a bbox-hoz (0..1)
    polygon:       Optional[np.ndarray] = None   # kontúr pontok ha elérhető
    dist_field:    Optional[np.ndarray] = None   # distance field a maszkon
    safe_bbox:     Optional[list[int]]  = None   # szöveg-biztonságos belső terület

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    def text_safe_bbox(self, padding_ratio: float = 0.12,
                       min_px: int = 6) -> list[int]:
        """
        Szöveg elhelyezéséhez biztonságos belső terület.

        Ellipszis esetén erősebb margó a sarkok miatt.
        """
        x1, y1, x2, y2 = self.bbox
        w, h = self.width, self.height

        if self.shape == BubbleShape.ELLIPSE:
            # Ellipszis: a sarkok 'elvesznek' → ~18% margó
            px = max(min_px, int(w * max(padding_ratio, 0.18)))
            py = max(min_px, int(h * max(padding_ratio, 0.15)))
        elif self.shape == BubbleShape.RECTANGLE:
            px = max(min_px, int(w * padding_ratio))
            py = max(min_px, int(h * padding_ratio))
        elif self.shape == BubbleShape.TALL:
            px = max(min_px, int(w * (padding_ratio + 0.05)))
            py = max(min_px, int(h * padding_ratio))
        else:
            px = max(min_px, int(w * padding_ratio))
            py = max(min_px, int(h * padding_ratio))

        return [x1 + px, y1 + py, x2 - px, y2 - py]


class BubbleAnalyzer:
    """
    Speech bubble alakzat elemzés és geometria meghatározás.

    Statikus metódusok – nincs állapot.
    """

    @staticmethod
    def analyze(
        image: np.ndarray,
        bbox: list[int],
        min_contour_area: int = 100,
    ) -> BubbleGeometry:
        """
        Teljes geometriai elemzés egy buborékra.

        Args:
            image: teljes oldal BGR kép
            bbox:  [x1, y1, x2, y2]

        Returns:
            BubbleGeometry a bubble teljes geometriájával.
        """
        x1, y1, x2, y2 = bbox
        h_img, w_img = image.shape[:2]
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w_img, int(x2)); y2 = min(h_img, int(y2))

        # Enforce minimum box size of 2x2px
        min_size = 2
        if x2 - x1 < min_size:
            if x1 + min_size <= w_img:
                x2 = x1 + min_size
            elif x2 - min_size >= 0:
                x1 = x2 - min_size
            else:
                x1 = 0
                x2 = min(w_img, min_size)

        if y2 - y1 < min_size:
            if y1 + min_size <= h_img:
                y2 = y1 + min_size
            elif y2 - min_size >= 0:
                y1 = y2 - min_size
            else:
                y1 = 0
                y2 = min(h_img, min_size)

        bw = x2 - x1
        bh = y2 - y1

        if bw < 5 or bh < 5:
            return BubbleGeometry(
                bbox=[x1, y1, x2, y2], shape=BubbleShape.UNKNOWN,
                aspect_ratio=1.0, fill_ratio=1.0,
                centroid=(0.5, 0.5),
            )

        region = image[y1:y2, x1:x2]
        aspect = bw / max(bh, 1)

        # Kontúr keresés a shape meghatározáshoz
        gray  = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygon      = None
        fill_ratio   = 0.85
        centroid     = (0.5, 0.5)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area    = cv2.contourArea(largest)

            if area > min_contour_area:
                fill_ratio = min(area / (bw * bh), 1.0)

                # Centroid
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"] / bw
                    cy = M["m01"] / M["m00"] / bh
                    centroid = (float(cx), float(cy))

                # Polygon közelítés
                epsilon   = 0.02 * cv2.arcLength(largest, True)
                approx    = cv2.approxPolyDP(largest, epsilon, True)
                polygon   = approx.reshape(-1, 2).astype(np.float32)

        # Shape classification
        shape = BubbleAnalyzer._classify_shape(
            aspect, fill_ratio, polygon, bw, bh)

        # Distance field a maszkon
        mask = np.zeros((bh, bw), dtype=np.uint8)
        if contours:
            cv2.drawContours(mask, contours, -1, 255, -1)
        else:
            mask[:] = 255
        dist = MaskOps.distance_field(mask)

        geom = BubbleGeometry(
            bbox=bbox,
            shape=shape,
            aspect_ratio=aspect,
            fill_ratio=fill_ratio,
            centroid=centroid,
            polygon=polygon,
            dist_field=dist,
        )
        geom.safe_bbox = geom.text_safe_bbox()
        return geom

    @staticmethod
    def _classify_shape(
        aspect: float,
        fill_ratio: float,
        polygon: Optional[np.ndarray],
        bw: int,
        bh: int,
    ) -> BubbleShape:
        """Alakzat osztályozás heurisztikák alapján."""

        # Magas/széles arány
        if aspect < 0.55:
            return BubbleShape.TALL
        if aspect > 2.2:
            return BubbleShape.WIDE

        # Narráció doboz: szögletes, magas fill ratio
        if fill_ratio > 0.90 and aspect > 0.8:
            if polygon is not None and len(polygon) <= 6:
                return BubbleShape.RECTANGLE

        # Irreguláris: kevés fill vagy sok sarok
        if fill_ratio < 0.65:
            return BubbleShape.IRREGULAR
        if polygon is not None and len(polygon) > 12:
            return BubbleShape.IRREGULAR

        # Ellipszis: kerek fill, kevés sarok
        if fill_ratio > 0.72 and (polygon is None or len(polygon) <= 12):
            return BubbleShape.ELLIPSE

        return BubbleShape.ELLIPSE  # default


# ══════════════════════════════════════════════════════════════════════════════
# 3. GLYPH METRIKÁK
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LineMetrics:
    """Egy renderelt sor pontos mérési eredménye."""
    text:        str
    width_px:    int
    height_px:   int
    ascent_px:   int
    descent_px:  int
    bbox:        tuple[int, int, int, int]   # (left, top, right, bottom)


class GlyphMetrics:
    """
    Valódi glyph-szintű szövegmérés Pillow-val.

    Fontánként cache-eli a mérési eredményeket a teljesítményért.
    Kerning-aware: teljes szöveg méret, nem karakter-összeg.
    """

    def __init__(self) -> None:
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
        self._measure_cache: dict[tuple[str, str, int], LineMetrics] = {}

    def load_font(
        self, path: str, size: int
    ) -> ImageFont.FreeTypeFont:
        """Font betöltés cache-sel."""
        key = (path, size)
        if key not in self._font_cache:
            try:
                self._font_cache[key] = ImageFont.truetype(path, size)
            except (IOError, OSError):
                logger.warning(f"Font nem tölthető: {path} – default használata")
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def measure_line(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
    ) -> LineMetrics:
        """
        Egy szövegsor pontos mérése glyph metrikákkal.

        Használ: font.getbbox() – kerning-aware, ascent/descent pontos.
        """
        font_key = id(font)
        cache_key = (text, str(font_key), getattr(font, "size", 0))

        if cache_key in self._measure_cache:
            return self._measure_cache[cache_key]

        if not text:
            ascent, descent = font.getmetrics()
            m = LineMetrics(
                text="", width_px=0,
                height_px=ascent + descent,
                ascent_px=ascent, descent_px=descent,
                bbox=(0, 0, 0, ascent + descent),
            )
            self._measure_cache[cache_key] = m
            return m

        # getbbox: (left, top, right, bottom) – kerning-aware
        bbox = font.getbbox(text)
        left, top, right, bottom = bbox
        ascent, descent = font.getmetrics()

        m = LineMetrics(
            text=text,
            width_px=right - left,
            height_px=bottom - top,
            ascent_px=ascent,
            descent_px=descent,
            bbox=bbox,
        )
        self._measure_cache[cache_key] = m
        return m

    def measure_block(
        self,
        lines: list[str],
        font: ImageFont.FreeTypeFont,
        line_spacing: float = 1.22,
    ) -> tuple[int, int]:
        """
        Több sor szövegblokk teljes mérete.

        Returns:
            (max_width, total_height) pixelben.
        """
        if not lines:
            return 0, 0
        metrics  = [self.measure_line(ln, font) for ln in lines]
        max_w    = max(m.width_px for m in metrics)
        ascent, descent = font.getmetrics()
        line_h   = int((ascent + descent) * line_spacing)
        total_h  = line_h * len(lines)
        return max_w, total_h

    def clear_measure_cache(self) -> None:
        self._measure_cache.clear()


# Singleton – importálható
_glyph_metrics = GlyphMetrics()


# ══════════════════════════════════════════════════════════════════════════════
# 4. LAYOUT SCORER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LayoutCandidate:
    """Egyetlen layout variáns leírása és score-ja."""
    lines:          list[str]
    font_size:      int
    total_width:    int
    total_height:   int
    score:          float = 0.0
    penalties:      dict  = field(default_factory=dict)

    # Elhelyezési koordináták (kitöltés után)
    x: int = 0
    y: int = 0


class LayoutScorer:
    """
    Score-driven layout kiértékelő.

    Több layout kandidátot értékel és a legjobbat választja.
    Determinisztikus: ugyanaz a bemenet → ugyanaz az eredmény.
    """

    # Score súlyok (0..1, összegük ~1.0)
    W_LINE_BALANCE    = 0.25   # sorhosszak egyensúlya
    W_COVERAGE        = 0.20   # terület kihasználtság
    W_FONT_SIZE       = 0.20   # nagyobb font = jobb (olvashatóság)
    W_EDGE_SAFETY     = 0.20   # szöveg távolsága a szélektől
    W_ORPHAN_PENALTY  = 0.15   # widow/orphan büntetés

    @classmethod
    def score(
        cls,
        candidate: LayoutCandidate,
        geom: BubbleGeometry,
        min_edge_dist: int = 4,
    ) -> float:
        """
        Egyetlen layout kandidáns teljes score-ja.

        Returns:
            0.0 (legrosszabb) .. 1.0 (legjobb)
        """
        scores: dict[str, float] = {}

        # 1. Sorhosszak egyensúlya – "gyémánt" elrendezés ideális
        scores["balance"] = cls._line_balance_score(candidate.lines)

        # 2. Terület kihasználtság
        safe = geom.safe_bbox or geom.bbox
        sw = safe[2] - safe[0]
        sh = safe[3] - safe[1]
        safe_area = max(sw * sh, 1)
        text_area = candidate.total_width * candidate.total_height
        coverage  = text_area / safe_area
        # 0.4–0.75 az ideális tartomány
        if coverage < 0.25:
            scores["coverage"] = coverage / 0.25 * 0.6
        elif coverage <= 0.78:
            scores["coverage"] = 0.8 + (coverage - 0.25) / 0.53 * 0.2
        else:
            scores["coverage"] = max(0.0, 1.0 - (coverage - 0.78) * 3)

        # 3. Font méret score (normalizált a max-hoz)
        from config import cfg
        fmax = cfg.rendering.font_size_max
        fmin = cfg.rendering.font_size_min
        scores["font_size"] = (candidate.font_size - fmin) / max(fmax - fmin, 1)

        # 4. Edge safety (distance field alapján ha elérhető)
        scores["edge_safety"] = cls._edge_safety_score(
            candidate, geom, min_edge_dist)

        # 5. Orphan/widow büntetés
        scores["orphan"] = cls._orphan_score(candidate.lines)

        # Súlyozott összeg
        total = (
            cls.W_LINE_BALANCE   * scores["balance"]  +
            cls.W_COVERAGE       * scores["coverage"] +
            cls.W_FONT_SIZE      * scores["font_size"] +
            cls.W_EDGE_SAFETY    * scores["edge_safety"] +
            cls.W_ORPHAN_PENALTY * scores["orphan"]
        )

        candidate.score    = float(np.clip(total, 0.0, 1.0))
        candidate.penalties = scores
        return candidate.score

    @staticmethod
    def _line_balance_score(lines: list[str]) -> float:
        """
        Sorhosszak egyensúlya – ideális a 'gyémánt' elrendezés.

        A középső sorok legyenek a leghosszabbak,
        az első és az utolsó sor rövidebbek.
        """
        if len(lines) <= 1:
            return 0.85
        lengths = [len(ln) for ln in lines]
        max_l   = max(lengths)
        if max_l == 0:
            return 0.5

        # Variancia büntetés (egyforma hossz is rossz)
        norm    = [l / max_l for l in lengths]
        n       = len(norm)

        if n >= 3:
            mid   = n // 2
            # Ideális: középső a leghosszabb
            mid_bonus = norm[mid] if norm[mid] == max(norm) else norm[mid] * 0.8
            edge_ok   = (norm[0] < norm[mid]) and (norm[-1] < norm[mid])
            balance   = mid_bonus * (1.1 if edge_ok else 0.85)
        else:
            balance = sum(norm) / n

        return float(np.clip(balance, 0.0, 1.0))

    @staticmethod
    def _edge_safety_score(
        candidate: LayoutCandidate,
        geom: BubbleGeometry,
        min_dist: int,
    ) -> float:
        """Szöveg biztonsági távolsága a buborék szélétől."""
        safe = geom.safe_bbox or geom.bbox
        sw = safe[2] - safe[0]
        sh = safe[3] - safe[1]
        if sw <= 0 or sh <= 0:
            return 0.5

        # Hány % a szöveg vs. a biztonságos terület
        w_ratio = candidate.total_width  / max(sw, 1)
        h_ratio = candidate.total_height / max(sh, 1)
        worst   = max(w_ratio, h_ratio)

        if worst <= 0.85:
            return 1.0
        elif worst <= 1.0:
            return 1.0 - (worst - 0.85) * 6.0
        else:
            return 0.0

    @staticmethod
    def _orphan_score(lines: list[str]) -> float:
        """Widow/orphan büntetés."""
        if not lines:
            return 0.5
        if len(lines) == 1:
            return 0.9
        last = lines[-1].strip()
        if not last:
            return 0.5
        # Egyszavas utolsó sor büntetés
        if len(last.split()) == 1 and len(lines) > 2:
            return 0.3
        # Nagyon rövid utolsó sor
        avg_len = sum(len(ln) for ln in lines[:-1]) / max(len(lines) - 1, 1)
        if avg_len > 0 and len(last) / avg_len < 0.25:
            return 0.5
        return 1.0

    @classmethod
    def best(
        cls,
        candidates: list[LayoutCandidate],
        geom: BubbleGeometry,
    ) -> Optional[LayoutCandidate]:
        """Legmagasabb score-ú kandidáns kiválasztása."""
        if not candidates:
            return None
        for c in candidates:
            cls.score(c, geom)
        return max(candidates, key=lambda c: c.score)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SUPERSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

class SuperSampler:
    """
    Supersampling alapú anti-aliasing szöveg rendereléshez.

    Workflow:
      1. Szöveg renderelése factor*méretű canvas-ra
      2. Lanczos downsample az eredeti méretre
      3. Alpha-compositing az alap képre

    Ez adja a professzionális, éles de sima szövegéleket.
    """

    def __init__(self, factor: int = 2) -> None:
        self.factor = max(1, factor)

    def upscale_canvas(
        self,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """Supersampled canvas méretek."""
        return width * self.factor, height * self.factor

    def upscale_font_size(self, size: int) -> int:
        """Font méret a supersampled canvas-ra."""
        return size * self.factor

    def upscale_coord(self, coord: int) -> int:
        """Koordináta a supersampled canvas-ra."""
        return coord * self.factor

    def upscale_bbox(self, bbox: list[int]) -> list[int]:
        """Bounding box a supersampled canvas-ra."""
        return [v * self.factor for v in bbox]

    def downscale(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
    ) -> Image.Image:
        """
        Lanczos downsample – legjobb minőségű kicsinyítés.

        Args:
            image:       supersampled PIL Image (RGBA vagy RGB)
            target_size: (width, height) a célméreten

        Returns:
            Downsamplelt PIL Image.
        """
        return image.resize(target_size, Image.LANCZOS)

    def downscale_numpy(
        self,
        arr: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """Numpy tömb Lanczos downscale."""
        if arr.shape[2] == 4:
            pil = Image.fromarray(arr, mode="RGBA")
        else:
            pil = Image.fromarray(arr, mode="RGB")
        resized = pil.resize((target_w, target_h), Image.LANCZOS)
        return np.array(resized)


# ══════════════════════════════════════════════════════════════════════════════
# 6. IMAGE UTILS
# ══════════════════════════════════════════════════════════════════════════════

class ImageUtils:
    """Általános képfeldolgozó segédfüggvények."""

    @staticmethod
    def bgr_to_pil(image: np.ndarray) -> Image.Image:
        """OpenCV BGR → PIL RGB."""
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def pil_to_bgr(image: Image.Image) -> np.ndarray:
        """PIL RGB → OpenCV BGR."""
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def pil_to_rgba(image: Image.Image) -> Image.Image:
        """PIL kép → RGBA (ha még nem az)."""
        return image.convert("RGBA") if image.mode != "RGBA" else image

    @staticmethod
    def sample_border_color(
        image: np.ndarray,
        bbox: list[int],
        border_px: int = 12,
    ) -> tuple[int, int, int]:
        """
        Buborék szélének domináns háttérszíne mintavételezéssel.

        Négy sávból (top/bottom/left/right) vett medián szín.
        Inpainting előfeldolgozáshoz hasznos.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        b = border_px

        strips = [
            image[y1:y1+b,    x1:x2],   # top
            image[y2-b:y2,    x1:x2],   # bottom
            image[y1:y2,      x1:x1+b], # left
            image[y1:y2,      x2-b:x2], # right
        ]
        valid = [s.reshape(-1, 3) for s in strips if s.size > 0]
        if not valid:
            return (255, 255, 255)
        pixels = np.vstack(valid)
        median = np.median(pixels, axis=0).astype(np.uint8)
        return (int(median[2]), int(median[1]), int(median[0]))  # BGR→RGB

    @staticmethod
    def is_simple_background(
        image: np.ndarray,
        bbox: list[int],
        threshold: float = 0.92,
    ) -> bool:
        """
        Meghatározza hogy a buborék háttere egyszerű-e
        (fehér/egyszínű) – LaMa kihagyható-e.

        Args:
            threshold: 0.0..1.0 – magasabb = szigorúbb

        Returns:
            True ha egyszerű háttér (nincs szükség LaMa-ra)
        """
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_img, x2); y2 = min(h_img, y2)
        if x2 <= x1 or y2 <= y1:
            return True

        region = image[y1:y2, x1:x2]
        gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Fehér arány
        white_ratio = np.sum(gray > 230) / gray.size
        if white_ratio >= threshold:
            return True

        # Alacsony szórás = egyszínű háttér
        std = float(np.std(gray.astype(np.float32)))
        if std < 18.0:
            return True

        return False

    @staticmethod
    def resize_if_needed(
        image: np.ndarray,
        max_side: int = 2560,
    ) -> tuple[np.ndarray, float]:
        """
        Nagy képek átméretezése ha szükséges.

        Returns:
            (átméretezett_kép, scale_factor)
            scale_factor=1.0 ha nem volt szükség átméretezésre.
        """
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return image, 1.0
        scale  = max_side / longest
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"Átméretezve: {w}x{h} → {new_w}x{new_h} (scale={scale:.3f})")
        return resized, scale

    @staticmethod
    def scale_bbox(bbox: list[int], scale: float) -> list[int]:
        """Bounding box visszaskálázása az eredeti méretre."""
        return [int(v / scale) for v in bbox]

    @staticmethod
    def draw_debug_geometry(
        image: np.ndarray,
        geom: BubbleGeometry,
        layout: Optional[LayoutCandidate] = None,
        color_safe: tuple = (0, 200, 0),
        color_bbox: tuple = (0, 100, 255),
    ) -> np.ndarray:
        """
        Debug vizualizáció: buborék geometria + layout terv.

        Args:
            image:  alap BGR kép
            geom:   BubbleGeometry
            layout: opcionális LayoutCandidate a score-ral

        Returns:
            Annotált BGR kép.
        """
        vis = image.copy()
        x1, y1, x2, y2 = geom.bbox

        # Buborék bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_bbox, 2)

        # Biztonságos szövegterület
        if geom.safe_bbox:
            sx1, sy1, sx2, sy2 = geom.safe_bbox
            cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), color_safe, 1)

        # Centroid
        cx = int(x1 + geom.centroid[0] * geom.width)
        cy = int(y1 + geom.centroid[1] * geom.height)
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

        # Shape + score felirat
        label = f"{geom.shape.name}"
        if layout:
            label += f" s={layout.score:.2f} f={layout.font_size}"
        cv2.rectangle(vis, (x1, y1-22), (x1+len(label)*7+4, y1), color_bbox, -1)
        cv2.putText(vis, label, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        return vis
