"""
layout.py - Speech bubble detektálás és panel-aware olvasási sorrend.

Architektúra:
  CoordClipper:       koordináta clip + int (nincs double scaling)
  BubbleDetector:     YOLO inference + NMS + típus osztályozás
  PanelGrouper:       heurisztikus panel szegmentálás whitespace alapján
  ReadingOrderSolver: panel-aware olvasási sorrend, overlap logika
  LayoutDetector:     publikus API orchestrátor

Tervezési elvek:
  - Koordináta integritás: float precision végig, egyetlen visszaskálázás
  - Panel-aware reading order: helyi csoporton belül, nem globálisan
  - Overlap-aware prioritás: metsző buborékok lokális rendezése
  - Graceful fallback: bizonytalanság esetén stabil, determinisztikus sorrend
  - Minden buborék hiba lokálisan kezelt – oldal feldolgozás folytatódik
  - Visszafelé kompatibilis API: detect_bubbles() dict-listát ad vissza
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from config import cfg
from utils import BubbleAnalyzer, BubbleShape

logger = logging.getLogger(__name__)

# Model keresési útvonalak (prioritás sorrendben)
_MODEL_CANDIDATES = [
    Path("models/detection/comic-speech-bubble-detector.pt"),
    Path("models/comic_bubble_detector.pt"),
    Path("models/detection/best.pt"),
]


# ══════════════════════════════════════════════════════════════════════════════
# ADATSTRUKTÚRÁK
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BubbleRegion:
    """
    Egyetlen speech bubble detektálás teljes leírója.

    Koordináták MINDIG az eredeti kép felbontásában vannak.
    Float precision: nincs korai int-re kerekítés.
    """
    id:           int
    bbox:         list[int]       # [x1, y1, x2, y2] – eredeti kép koordinátái
    order:        int             # olvasási sorrend index
    confidence:   float
    type:         str = "bubble"  # "bubble" | "narration" | "sfx"
    panel_id:     int = -1        # melyik panelhez tartozik (-1 = ismeretlen)
    overlap_ids:  list[int] = field(default_factory=list)  # átfedő buborékok

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def width(self)    -> int: return self.bbox[2] - self.bbox[0]
    @property
    def height(self)   -> int: return self.bbox[3] - self.bbox[1]
    @property
    def center_x(self) -> float: return (self.bbox[0] + self.bbox[2]) / 2.0
    @property
    def center_y(self) -> float: return (self.bbox[1] + self.bbox[3]) / 2.0
    @property
    def area(self)     -> int: return self.width * self.height


@dataclass
class PanelGroup:
    """
    Heurisztikusan azonosított panel / buborék-csoport.

    Nem teljes AI szegmentálás – whitespace + proximity alapú közelítés.
    """
    id:         int
    bbox:       list[int]         # a csoport befoglaló téglalapja
    bubble_ids: list[int]         # a csoportba tartozó buborék indexek


# ══════════════════════════════════════════════════════════════════════════════
# KOORDINÁTA MAPPER – float precision
# ══════════════════════════════════════════════════════════════════════════════

class CoordClipper:
    """
    Koordináta clip + int konverzió – NEM skálázás.

    Az Ultralytics result.boxes.xyxy már az EREDETI kép koordinátaterében
    adja vissza az értékeket → második skálázás NEM kell, csak:
      - float → int (floor/ceil)
      - kép határaira klip

    Ha nem-Ultralytics modellt (pl. ONNX RT-DETR) használunk ahol
    valódi visszaskálázás kell, a scale_x/scale_y paramétereket
    1.0-tól eltérő értékre kell állítani.

    Egyetlen int konverzió pontja – nincs kumulatív kerekítési drift.
    """

    def __init__(
        self,
        orig_w:  int,
        orig_h:  int,
        scale_x: float = 1.0,   # 1.0 = Ultralytics (már eredeti tér)
        scale_y: float = 1.0,   # != 1.0 = ONNX / custom model
    ) -> None:
        self.orig_w  = orig_w
        self.orig_h  = orig_h
        self.scale_x = scale_x
        self.scale_y = scale_y

    # OCR-safe minimális dimenzió: PPOCRv5 DB detektor legalább 64px-t
    # igényel hogy a Conv.33 ne crasheljen (feat_map >= 2×2 kell).
    # A clip_box min_size-t 8px-re állítjuk – ez az optikai minimum,
    # a tényleges OCR-upscale az ocr.py _preprocess() kezeli (64px).
    OCR_MIN_SIZE: int = 8

    def clip_box(
        self,
        x1f: float, y1f: float,
        x2f: float, y2f: float,
        min_size: int = 8,
    ) -> list[int]:
        """
        Float koordináta → int bbox, kép határaira klipselve.

        Biztosítja a minimum méretet (default: 8px) és azt, hogy a koordináták
        mindig a képen belül maradjanak és ne képezzenek üres területet.

        A 8px minimum az optikai minimum – az OCR pipeline (ocr.py _preprocess)
        a tényleges OCR-safe méretre (64px) skálázza fel a kis buborékokat.
        """
        x1 = max(0,           int(np.floor(x1f * self.scale_x)))
        y1 = max(0,           int(np.floor(y1f * self.scale_y)))
        x2 = min(self.orig_w, int(np.ceil( x2f * self.scale_x)))
        y2 = min(self.orig_h, int(np.ceil( y2f * self.scale_y)))

        # Enforce minimum width
        if x2 - x1 < min_size:
            if x1 + min_size <= self.orig_w:
                x2 = x1 + min_size
            elif x2 - min_size >= 0:
                x1 = x2 - min_size
            else:
                x1 = 0
                x2 = min(self.orig_w, min_size)

        # Enforce minimum height
        if y2 - y1 < min_size:
            if y1 + min_size <= self.orig_h:
                y2 = y1 + min_size
            elif y2 - min_size >= 0:
                y1 = y2 - min_size
            else:
                y1 = 0
                y2 = min(self.orig_h, min_size)

        return [x1, y1, x2, y2]


# ══════════════════════════════════════════════════════════════════════════════
# NMS – Non-Maximum Suppression
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a: list[int], b: list[int]) -> float:
    """Intersection over Union két bbox között."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1)


def _overlap_ratio(a: list[int], b: list[int]) -> float:
    """Kisebb bbox területének hány %-a fedi a másikat."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    smaller = min(area_a, area_b)
    return inter / max(smaller, 1)


def _nms(
    boxes: list[list[int]],
    confs: list[float],
    iou_thresh: float,
) -> list[int]:
    """Standard NMS – confidence szerint rendezett."""
    order = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
    keep        = []
    suppressed  = [False] * len(boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j <= i or suppressed[j]:
                continue
            if _iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed[j] = True
    return keep


# ══════════════════════════════════════════════════════════════════════════════
# PANEL GROUPER – heurisztikus panel szegmentálás
# ══════════════════════════════════════════════════════════════════════════════

class PanelGrouper:
    """
    Heurisztikus panel / buborék-csoport azonosítás.

    NEM teljes AI panel szegmentálás.
    Stratégia:
      1. Whitespace gap detektálás (vízszintes és függőleges)
      2. Proximity clustering (közeli buborékok egy panelbe kerülnek)
      3. Átfedő buborékok egy csoportba sorolva

    Bizonytalan esetben: graceful fallback → minden buborék egy csoportban.
    Determinisztikus: azonos bemenet → azonos csoportok.
    """

    def __init__(
        self,
        gap_threshold_ratio: float = 0.04,   # oldal méret %-ában
        proximity_ratio:     float = 0.15,   # buborék méret %-ában
    ) -> None:
        self.gap_threshold_ratio = gap_threshold_ratio
        self.proximity_ratio     = proximity_ratio

    def group(
        self,
        bubbles: list[BubbleRegion],
        page_w:  int,
        page_h:  int,
    ) -> list[PanelGroup]:
        """
        Buborékok csoportosítása panel heurisztikával.

        Args:
            bubbles: detektált buborékok (koordináták eredeti térben)
            page_w:  oldal szélessége
            page_h:  oldal magassága

        Returns:
            PanelGroup lista. Ha bizonytalan → egyetlen csoport.
        """
        if not bubbles:
            return []

        if len(bubbles) == 1:
            return [PanelGroup(
                id=0,
                bbox=bubbles[0].bbox,
                bubble_ids=[0],
            )]

        try:
            return self._cluster(bubbles, page_w, page_h)
        except Exception as e:
            logger.debug(f"Panel grouper hiba, fallback: {e}")
            return self._fallback_single_group(bubbles)

    def _cluster(
        self,
        bubbles: list[BubbleRegion],
        page_w:  int,
        page_h:  int,
    ) -> list[PanelGroup]:
        """
        Union-Find alapú proximity clustering.

        Két buborék azonos csoportba kerül ha:
          - Átfednek (overlap_ratio > 0.1)
          - Közel vannak egymáshoz (gap < threshold)
        """
        n       = len(bubbles)
        parent  = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        gap_px = max(
            int(min(page_w, page_h) * self.gap_threshold_ratio),
            8,
        )

        for i in range(n):
            for j in range(i + 1, n):
                bi = bubbles[i].bbox
                bj = bubbles[j].bbox

                # Átfedés → egy csoport
                if _overlap_ratio(bi, bj) > 0.05:
                    union(i, j)
                    continue

                # Proximity: vízszintes és függőleges gap
                h_gap = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
                v_gap = max(0, max(bi[1], bj[1]) - min(bi[3], bj[3]))

                # X-átfedés esetén csak v_gap számít (ugyanaz a oszlop)
                x_overlap = min(bi[2], bj[2]) - max(bi[0], bj[0]) > 0
                y_overlap = min(bi[3], bj[3]) - max(bi[1], bj[1]) > 0

                if x_overlap and v_gap <= gap_px:
                    union(i, j)
                elif y_overlap and h_gap <= gap_px:
                    union(i, j)
                elif h_gap <= gap_px and v_gap <= gap_px:
                    # Sarokközelség
                    union(i, j)

        # Csoportok összegyűjtése
        groups_map: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups_map.setdefault(root, []).append(i)

        # PanelGroup-ok összeállítása
        panels: list[PanelGroup] = []
        for gid, (root, indices) in enumerate(
                sorted(groups_map.items(), key=lambda kv: kv[0])):
            # Befoglaló bbox
            all_bboxes = [bubbles[i].bbox for i in indices]
            gx1 = min(b[0] for b in all_bboxes)
            gy1 = min(b[1] for b in all_bboxes)
            gx2 = max(b[2] for b in all_bboxes)
            gy2 = max(b[3] for b in all_bboxes)
            panels.append(PanelGroup(
                id=gid,
                bbox=[gx1, gy1, gx2, gy2],
                bubble_ids=sorted(indices),
            ))

        logger.debug(f"Panel grouper: {len(panels)} csoport "
                     f"({n} buborékból)")
        return panels

    @staticmethod
    def _fallback_single_group(
        bubbles: list[BubbleRegion],
    ) -> list[PanelGroup]:
        """Fallback: minden buborék egyetlen csoportban."""
        all_bboxes = [b.bbox for b in bubbles]
        return [PanelGroup(
            id=0,
            bbox=[
                min(b[0] for b in all_bboxes),
                min(b[1] for b in all_bboxes),
                max(b[2] for b in all_bboxes),
                max(b[3] for b in all_bboxes),
            ],
            bubble_ids=list(range(len(bubbles))),
        )]


# ══════════════════════════════════════════════════════════════════════════════
# READING ORDER SOLVER – panel-aware, overlap-aware
# ══════════════════════════════════════════════════════════════════════════════

class ReadingOrderSolver:
    """
    Panel-aware olvasási sorrend meghatározás.

    Stratégia:
      1. Panel-on belüli lokális rendezés (nem globális oldal-rendezés)
      2. Panel-ok globális rendezése (bal→jobb, fel→le)
      3. Átfedő buborékok overlap-aware prioritása
      4. Determinisztikus: tie-breaking mindig koordináta alapú

    Western comics konvenció: bal→jobb, fel→le soronként.
    """

    def __init__(
        self,
        y_tolerance: int = None,
    ) -> None:
        self.y_tolerance = y_tolerance or cfg.layout.reading_order_y_tolerance

    def solve(
        self,
        bubbles: list[BubbleRegion],
        panels:  list[PanelGroup],
    ) -> list[BubbleRegion]:
        """
        Teljes olvasási sorrend meghatározás.

        Args:
            bubbles: detektált buborékok
            panels:  panel csoportok

        Returns:
            BubbleRegion lista olvasási sorrendben (order mező kitöltve).
        """
        if not bubbles:
            return []

        # Panel-on belüli rendezés
        panel_order = self._order_panels(panels, bubbles)

        ordered: list[BubbleRegion] = []
        reading_idx = 0

        for panel in panel_order:
            panel_bubbles = [bubbles[i] for i in panel.bubble_ids]

            # Átfedés detektálás a panelen belül
            overlap_groups = self._find_overlap_groups(panel_bubbles)

            # Panel-on belüli rendezés
            local_ordered = self._order_within_panel(
                panel_bubbles, overlap_groups)

            for b in local_ordered:
                # Új BubbleRegion a frissített order-rel és panel_id-vel
                ordered.append(BubbleRegion(
                    id=b.id,
                    bbox=b.bbox,
                    order=reading_idx,
                    confidence=b.confidence,
                    type=b.type,
                    panel_id=panel.id,
                    overlap_ids=b.overlap_ids,
                ))
                reading_idx += 1

        return ordered

    def _order_panels(
        self,
        panels: list[PanelGroup],
        bubbles: list[BubbleRegion],
    ) -> list[PanelGroup]:
        """
        Panel-ok globális rendezése: soronként bal→jobb, sorok fel→le.

        Western comics olvasási konvenció.
        Tie-breaking: koordináta alapú, determinisztikus.
        """
        if not panels:
            return []

        # Panel centroidok
        def panel_sort_key(p: PanelGroup) -> tuple:
            cx = (p.bbox[0] + p.bbox[2]) / 2.0
            cy = (p.bbox[1] + p.bbox[3]) / 2.0
            return (cy, cx)  # elsősorban Y, másodsorban X

        return sorted(panels, key=panel_sort_key)

    def _order_within_panel(
        self,
        bubbles: list[BubbleRegion],
        overlap_groups: list[list[int]],
    ) -> list[BubbleRegion]:
        """
        Panel-on belüli olvasási sorrend – overlap-aware.

        Algoritmus:
          1. Átfedő buborékok lokális rendezése (Fix #3)
          2. Nem-átfedő buborékok sorokba rendezése Y-centroid alapján
          3. Sorokon belül X szerint rendezve
          4. Sorok Y szerint rendezve
          5. Overlap csoportok beillesztése a sorrendbe

        Overlap prioritás szabályok (Fix #3):
          - Kisebb buborék általában előbb (belső balloon)
          - Hasonló méret esetén top-left prioritás (kisebb Y, majd kisebb X)
          - Determinisztikus: azonos méret → ID alapú tie-break
        """
        if not bubbles:
            return []
        if len(bubbles) == 1:
            return bubbles

        # Átfedő indexek halmaza
        overlapping_idx: set[int] = set()
        for grp in overlap_groups:
            for i in grp:
                overlapping_idx.add(i)

        # Átfedő buborékok rendezése lokálisan
        overlap_ordered: list[int] = []
        processed_grps: set[int]   = set()
        for grp in overlap_groups:
            grp_key = tuple(sorted(grp))
            if grp_key in processed_grps:
                continue
            processed_grps.add(grp_key)

            def overlap_sort_key(idx: int) -> tuple:
                b = bubbles[idx]
                area = b.area
                # Kisebb buborék előbb, tie-break: top-left (Y, X, ID)
                return (area, b.center_y, b.center_x, b.id)

            sorted_grp = sorted(grp, key=overlap_sort_key)
            overlap_ordered.extend(sorted_grp)

        # Nem-átfedő buborékok sorokba rendezése
        non_overlap = [i for i in range(len(bubbles))
                       if i not in overlapping_idx]

        rows: list[list[int]] = []
        for i in non_overlap:
            cy     = bubbles[i].center_y
            placed = False
            for row in rows:
                row_cy = np.mean([bubbles[j].center_y for j in row])
                if abs(cy - row_cy) <= self.y_tolerance:
                    row.append(i)
                    placed = True
                    break
            if not placed:
                rows.append([i])

        rows.sort(key=lambda row: np.mean(
            [bubbles[j].center_y for j in row]))

        non_overlap_ordered: list[int] = []
        for row in rows:
            row_sorted = sorted(row, key=lambda j: bubbles[j].center_x)
            non_overlap_ordered.extend(row_sorted)

        # Overlap csoportok beillesztése a sorrendbe:
        # Az overlap csoport legfelső buborékának Y-pozíciója
        # alapján illesztjük be a nem-átfedő sorrendbe
        if not overlap_ordered:
            final_order = non_overlap_ordered
        else:
            # Overlap csoport első elemének Y pozíciója
            ovl_y = bubbles[overlap_ordered[0]].center_y
            insert_pos = 0
            for pos, idx in enumerate(non_overlap_ordered):
                if bubbles[idx].center_y <= ovl_y:
                    insert_pos = pos + 1
            final_order = (non_overlap_ordered[:insert_pos] +
                           overlap_ordered +
                           non_overlap_ordered[insert_pos:])

        return [bubbles[i] for i in final_order]

    def _find_overlap_groups(
        self,
        bubbles: list[BubbleRegion],
    ) -> list[list[int]]:
        """
        Átfedő buborékok csoportjainak meghatározása.

        Returns:
            Lista of listák: minden inner lista átfedő buborék indexeket tartalmaz.
        """
        n      = len(bubbles)
        groups = []
        used   = [False] * n

        for i in range(n):
            if used[i]:
                continue
            group = [i]
            for j in range(i + 1, n):
                if not used[j]:
                    if _overlap_ratio(bubbles[i].bbox, bubbles[j].bbox) > 0.1:
                        group.append(j)
                        used[j] = True
            if len(group) > 1:
                groups.append(group)
                used[i] = True

        return groups


# ══════════════════════════════════════════════════════════════════════════════
# BUBBLE DETECTOR – YOLO inference wrapper
# ══════════════════════════════════════════════════════════════════════════════

class BubbleDetector:
    """
    YOLO alapú speech bubble detektor.

    Koordináta integritás:
      - YOLO átméretezett bemeneten fut (imgsz=1280)
      - CoordClipper: Ultralytics xyxy már eredeti térben van
      - Egyetlen float→int konverzió, clip only

    Típus osztályozás:
      - class neve alapján: "narration", "sfx", "bubble" (default)
    """

    def __init__(self) -> None:
        self._model: Any  = None
        self._names: dict = {}
        self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("pip install ultralytics")
            return

        chosen = None
        for c in _MODEL_CANDIDATES:
            if c.exists():
                chosen = str(c)
                logger.info(f"Comics modell: {chosen}")
                break

        if chosen is None:
            fallback = f"{cfg.layout.fallback_model}.pt"
            logger.warning(
                f"Comics modell nem található – fallback: {fallback}\n"
                f"Javasolt helyek: {[str(c) for c in _MODEL_CANDIDATES]}"
            )
            chosen = fallback

        try:
            self._model = YOLO(chosen)
            self._model.to(cfg.device.device)
            self._names = self._model.names if hasattr(self._model, "names") else {}
            logger.info(f"Detektor betöltve ({cfg.device.device}) ✓ "
                        f"| classes={list(self._names.values())[:8]}")
        except Exception as e:
            logger.error(f"YOLO betöltési hiba: {e}")

    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[list[list[int]], list[float], list[str]]:
        """
        Speech bubble-ok detektálása.

        Returns:
            (boxes, confidences, types) – koordináták eredeti kép terében.
        """
        if self._model is None:
            return [], [], []

        h_orig, w_orig = image.shape[:2]

        try:
            results = self._model.predict(
                source=image,
                conf=cfg.layout.conf_threshold,
                iou=cfg.layout.iou_threshold,
                device=cfg.device.device,
                verbose=False,
                imgsz=cfg.layout.inference_imgsz,
            )
        except Exception as e:
            logger.error(f"YOLO predict hiba: {e}")
            return [], [], []

        # Ultralytics: xyxy ALREADY in original image coordinates
        # CoordClipper scale=1.0 → clip + int only, NO rescaling
        clipper = CoordClipper(w_orig, h_orig, scale_x=1.0, scale_y=1.0)

        raw_boxes: list[list[int]] = []
        raw_confs: list[float]     = []
        raw_types: list[str]       = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id   = int(box.cls[0].item())
                cls_name = self._names.get(cls_id, "bubble").lower()

                # Típus meghatározás
                # text_free = képregény panel szöveg nélküli területe – kihagyjuk
                # Ezek nem speech bubble-ök, nincs bennük fordítható szöveg
                if any(k in cls_name for k in ("text_free", "free", "panel", "background")):
                    continue

                if any(k in cls_name for k in ("narr", "box", "caption")):
                    btype = "narration"
                elif any(k in cls_name for k in ("sfx", "sound", "effect")):
                    btype = "sfx"
                else:
                    btype = "bubble"

                # Osztály szűrés (üres lista = mindent elfogadunk)
                if cfg.layout.bubble_class_ids and                         cls_id not in cfg.layout.bubble_class_ids:
                    continue

                # Class-specific confidence threshold (Fix #4)
                conf_val = float(box.conf[0].item())
                thresh   = {
                    "narration": cfg.layout.conf_narration,
                    "sfx":       cfg.layout.conf_sfx,
                }.get(btype, cfg.layout.conf_bubble)
                if conf_val < thresh:
                    continue

                # Float koordináták – Ultralytics már eredeti térben adja
                x1f, y1f, x2f, y2f = box.xyxy[0].tolist()

                # Egyetlen float→int konverzió – clip only, nincs skálázás
                bbox = clipper.clip_box(x1f, y1f, x2f, y2f)

                # Méret szűrők
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if bw < cfg.layout.min_bubble_width:
                    continue
                if bh < cfg.layout.min_bubble_height:
                    continue
                # Teljes oldalt lefedő téves detektálás szűrő
                if bw * bh > cfg.layout.max_page_coverage * w_orig * h_orig:
                    continue

                # OCR minőség figyelmeztetés – nem szűrjük ki, az ocr.py kezeli
                # PPOCRv5 conv.33-hoz legalább 64px kell; alatta upscale történik
                _OCR_SAFE = 64
                if bw < _OCR_SAFE or bh < _OCR_SAFE:
                    logger.debug(
                        f"Kis bubble #{len(raw_boxes)}: {bw}×{bh}px "
                        f"(OCR-safe minimum: {_OCR_SAFE}px) – "
                        f"ocr.py upscale-eli")

                raw_boxes.append(bbox)
                raw_confs.append(conf_val)
                raw_types.append(btype)

        return raw_boxes, raw_confs, raw_types


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT DETECTOR – publikus orchestrátor
# ══════════════════════════════════════════════════════════════════════════════

class LayoutDetector:
    """
    Fő layout detektálás orchestrátor.

    Pipeline:
      1. BubbleDetector → raw boxes + confs + types
      2. NMS duplikáció szűrés
      3. PanelGrouper → panel csoportok
      4. ReadingOrderSolver → olvasási sorrend
      5. Átfedés annotálás

    Minden lépés hibatűrő – részleges eredmény is visszaadható.
    """

    def __init__(self) -> None:
        self._detector = BubbleDetector()
        self._grouper  = PanelGrouper(
            gap_threshold_ratio=0.04,
            proximity_ratio=0.15,
        )
        self._solver   = ReadingOrderSolver()

    def detect(
        self,
        image: np.ndarray,
        image_path: Any = None,
    ) -> list[BubbleRegion]:
        """
        Teljes detektálás és rendezés egy képen.

        Args:
            image:      [H, W, 3] uint8 BGR
            image_path: opcionális (logoláshoz)

        Returns:
            BubbleRegion lista olvasási sorrendben.
        """
        src  = str(image_path) if image_path else "<mem>"
        h, w = image.shape[:2]

        # 1. Detektálás
        raw_boxes, raw_confs, raw_types = self._detector.detect(image)

        if not raw_boxes:
            logger.warning(f"Nincs detektált buborék: {src}")
            return self._fallback(w, h)

        # 2. NMS
        keep      = _nms(raw_boxes, raw_confs, cfg.layout.iou_threshold)
        raw_boxes = [raw_boxes[i] for i in keep]
        raw_confs = [raw_confs[i] for i in keep]
        raw_types = [raw_types[i] for i in keep]

        # 3. BubbleRegion lista (ideiglenes order=-1)
        bubbles: list[BubbleRegion] = []
        for idx, (bbox, conf, btype) in enumerate(
                zip(raw_boxes, raw_confs, raw_types)):
            bubbles.append(BubbleRegion(
                id=idx, bbox=bbox, order=-1,
                confidence=conf, type=btype,
            ))

        # 4. Átfedés annotálás
        bubbles = self._annotate_overlaps(bubbles)

        # 5. Panel csoportosítás
        try:
            panels = self._grouper.group(bubbles, w, h)
        except Exception as e:
            logger.debug(f"Panel grouper hiba, fallback: {e}")
            panels = PanelGrouper._fallback_single_group(bubbles)

        # 6. Olvasási sorrend
        try:
            ordered = self._solver.solve(bubbles, panels)
        except Exception as e:
            logger.warning(f"Reading order hiba, fallback: {e}")
            ordered = self._fallback_order(bubbles)

        logger.info(
            f"Detektált: {len(ordered)} buborék | "
            f"{len(panels)} panel | {src}"
        )
        return ordered

    @staticmethod
    def _annotate_overlaps(
        bubbles: list[BubbleRegion],
    ) -> list[BubbleRegion]:
        """Átfedő buborékok ID listájának kitöltése."""
        n = len(bubbles)
        overlap_map: dict[int, list[int]] = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                if _overlap_ratio(bubbles[i].bbox, bubbles[j].bbox) > 0.08:
                    overlap_map[i].append(j)
                    overlap_map[j].append(i)

        result = []
        for i, b in enumerate(bubbles):
            result.append(BubbleRegion(
                id=b.id, bbox=b.bbox, order=b.order,
                confidence=b.confidence, type=b.type,
                panel_id=b.panel_id,
                overlap_ids=overlap_map[i],
            ))
        return result

    @staticmethod
    def _fallback(w: int, h: int) -> list[BubbleRegion]:
        """
        Nincs detektálás: üres lista visszaadása.

        Fix #2: A teljes oldalt lefedő fallback buborék ELTÁVOLÍTVA.
        Okozhatott volna:
          - teljes oldal inpainting (artwork törlés)
          - érvénytelen óriás szöveg régiók
          - LaMa / compositor katasztrofális mellékhatások

        A pipeline kezelje le a "nincs buborék" esetet (oldal kihagyás).
        """
        logger.info(f"Nincs detektált buborék ({w}x{h}) – oldal kihagyva")
        return []

    @staticmethod
    def _fallback_order(
        bubbles: list[BubbleRegion],
    ) -> list[BubbleRegion]:
        """
        Stabil fallback rendezés: Y majd X koordináta.

        Determinisztikus – azonos bemenet → azonos sorrend.
        """
        sorted_b = sorted(bubbles, key=lambda b: (b.center_y, b.center_x))
        result = []
        for i, b in enumerate(sorted_b):
            result.append(BubbleRegion(
                id=b.id, bbox=b.bbox, order=i,
                confidence=b.confidence, type=b.type,
                panel_id=b.panel_id, overlap_ids=b.overlap_ids,
            ))
        return result

    def draw_debug(
        self,
        image: np.ndarray,
        bubbles: list[BubbleRegion],
    ) -> np.ndarray:
        """Debug vizualizáció: bbox, sorrend, panel ID, overlap."""
        vis    = image.copy()
        colors = [
            (0,220,0),(255,140,0),(0,0,255),
            (220,0,220),(0,220,220),(220,220,0),
        ]
        for b in bubbles:
            x1,y1,x2,y2 = b.bbox
            c = colors[b.panel_id % len(colors)] if b.panel_id >= 0 \
                else (128,128,128)
            thickness = 3 if b.overlap_ids else 2
            cv2.rectangle(vis, (x1,y1), (x2,y2), c, thickness)

            # Sorrend + típus + panel
            label = f"#{b.order+1} {b.type[:3]} p{b.panel_id}"
            if b.overlap_ids:
                label += " OVL"
            lw = len(label) * 7 + 4
            cv2.rectangle(vis,(x1,y1-20),(x1+lw,y1),c,-1)
            cv2.putText(vis, label, (x1+2,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0,0,0), 1, cv2.LINE_AA)

            # Centroid
            cx, cy = int(b.center_x), int(b.center_y)
            cv2.circle(vis, (cx,cy), 3, (0,0,255), -1)

        return vis


# ══════════════════════════════════════════════════════════════════════════════
# Singleton + moduláris API
# ══════════════════════════════════════════════════════════════════════════════

_detector_singleton: Optional[LayoutDetector] = None


def get_detector() -> LayoutDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = LayoutDetector()
    return _detector_singleton


def detect_bubbles(
    image: np.ndarray,
    image_path: Any = None,
) -> list[dict[str, Any]]:
    """
    Moduláris API: detektált buborékok dict-listája.

    Visszafelé kompatibilis – a meglévő pipeline kód nem változik.
    """
    return [b.to_dict() for b in get_detector().detect(image, image_path)]
