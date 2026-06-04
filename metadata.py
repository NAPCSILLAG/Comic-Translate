"""
metadata.py - PageData, BubbleData, Manifest, DirtyTracker, atomic save.

Ez a projekt autoritativ allapota - nem tranziense adatszerkezet.
A JSON emberileg olvasható/szerkeszthető - GUI/QA workflow-hoz tervezve.

Tervezesi elvek:
  - Atomic save: temp file + rename (crash-safe)
  - Human-in-the-loop: kezzel szerkesztett JSON prioritast elvez
  - Dirty tracking: dependency-aware, stage-szintu ES bubble-szintu
  - Metadata versioning: backward compatibility migration-nel
  - Bubble-level rerender: minden bubble onallo dirty flaggel
  - Resume: reszleges feldolgozas folytathato crash utan
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

METADATA_VERSION = "1.0"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ENUM + DEPENDENCY GRAPH
# ══════════════════════════════════════════════════════════════════════════════

class Stage(str, Enum):
    LAYOUT      = "layout"
    OCR         = "ocr"
    OCR_CORRECT = "ocr_correction"   # Gemma/Qwen OCR javitas
    TRANSLATION = "translation"
    REFINEMENT  = "refinement"       # Gemma forditas finomitas
    INPAINT     = "inpaint"
    RENDER      = "render"
    COMPOSITE   = "composite"
    EXPORT      = "export"


# Dependency graph: ha egy stage dirty, ezek is dirty lesznek
STAGE_DOWNSTREAM: dict[Stage, list[Stage]] = {
    Stage.LAYOUT:      [Stage.OCR, Stage.INPAINT, Stage.RENDER, Stage.COMPOSITE],
    Stage.OCR:         [Stage.OCR_CORRECT, Stage.TRANSLATION, Stage.INPAINT],
    Stage.OCR_CORRECT: [Stage.TRANSLATION],
    Stage.TRANSLATION: [Stage.REFINEMENT, Stage.RENDER],
    Stage.REFINEMENT:  [Stage.RENDER],
    Stage.INPAINT:     [Stage.COMPOSITE],
    Stage.RENDER:      [Stage.COMPOSITE],
    Stage.COMPOSITE:   [Stage.EXPORT],
    Stage.EXPORT:      [],
}


def downstream_stages(dirty_stage: Stage) -> set[Stage]:
    """
    Topologiai rendezesben: ha dirty_stage megvaltozott,
    mely stage-ek valnak dirty-ve automatikusan.

    Pelda:
      font_changed -> render dirty
      -> composite dirty (render downstream-je)
    """
    visited: set[Stage] = set()
    queue = [dirty_stage]
    while queue:
        current = queue.pop()
        for dep in STAGE_DOWNSTREAM.get(current, []):
            if dep not in visited:
                visited.add(dep)
                queue.append(dep)
    return visited


# ══════════════════════════════════════════════════════════════════════════════
# DIRTY FLAGS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DirtyFlags:
    """
    Stage-szintu dirty tracking egy oldalhoz.

    Minden flag True = ujra kell futtatni.
    False = cachelt eredmeny hasznalhato.
    """
    layout:      bool = True
    ocr:         bool = True
    ocr_correction: bool = True
    translation: bool = True
    refinement:  bool = True
    inpaint:     bool = True
    render:      bool = True
    composite:   bool = True
    export:      bool = True

    def mark_dirty(self, stage: Stage) -> None:
        """Stage + osszes downstream stage dirty jelolese."""
        setattr(self, stage.value, True)
        for ds in downstream_stages(stage):
            setattr(self, ds.value, True)

    def is_dirty(self, stage: Stage) -> bool:
        return getattr(self, stage.value, True)

    def mark_done(self, stage: Stage) -> None:
        setattr(self, stage.value, False)

    def all_clean(self) -> bool:
        return not any([
            self.layout, self.ocr, self.ocr_correction,
            self.translation, self.refinement,
            self.inpaint, self.render, self.composite, self.export,
        ])

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DirtyFlags":
        valid = {f.value for f in Stage}
        kwargs = {k: bool(v) for k, v in d.items() if k in valid}
        return cls(**kwargs)

    @classmethod
    def all_dirty(cls) -> "DirtyFlags":
        return cls()   # minden True alapertelmezetten


# ══════════════════════════════════════════════════════════════════════════════
# STAGE HASH – input fingerprint cache invalidalashoz
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StageHashes:
    """
    Stage-szintu input hash-ek.

    Ha az input hash nem valtozott -> stage kihagyhato.
    Ha valtozott -> dirty flag + downstream dirty.
    """
    layout_hash:      str = ""
    ocr_hash:         str = ""
    translation_hash: str = ""
    inpaint_hash:     str = ""
    render_hash:      str = ""
    composite_hash:   str = ""
    font_hash:        str = ""   # font valtozas -> render dirty
    provider_hash:    str = ""   # provider valtozas -> adott stage dirty

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StageHashes":
        return cls(**{k: str(v) for k, v in d.items() if hasattr(cls, k)})


# ══════════════════════════════════════════════════════════════════════════════
# BUBBLE DATA – egyetlen buborek teljes allapota
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BubbleData:
    """
    Egyetlen speech bubble autoritativ allapota.

    Ez az objektum a pipeline-on at aramlik es a JSON-ba kerul.
    Kezzel szerkesztheto - a pipeline prioritaskent kezeli.

    Fields:
        bubble_id:        egyedi azonosito az oldalon belul
        bbox:             [x1, y1, x2, y2] eredeti kep koordinatakban
        polygon:          OCR polygon pontok [[x,y],...]
        reading_order:    olvasasi sorrend index
        panel_id:         panel csoport ID
        bubble_type:      "bubble" | "narration" | "sfx"
        detection_conf:   YOLO detection confidence
        raw_text:         PPOCR nyers output (szerkesztheto)
        corrected_text:   Gemma/Qwen altal javitott OCR (szerkesztheto)
        ocr_confidence:   OCR confidence score
        translated_text:  forditas (szerkesztheto - human-in-the-loop)
        refined_text:     Gemma altal finomitott forditas
        tone:             hangulat (angry/neutral/...)
        emphasis_words:   hangsúlyos szavak listaja
        typography_preset: "bubble" | "narbox" | "thought"
        font_size_used:   vegul hasznalt font meret
        render_score:     layout score (0..1)
        provider_used:    melyik provider forditotta
        inpaint_provider: melyik inpaint provider
        render_hash:      render bemenet hash (bubble-szintu dirty)
        manually_edited:  True ha human szerkesztette - NE regenerald!
        skip_reason:      miert lett kihagyva (SFX, ures, stb.)
        timings:          stage-szintu futasi idok (ms)
    """
    bubble_id:         int
    bbox:              list[int]
    polygon:           list[list[float]]   = field(default_factory=list)
    reading_order:     int                 = -1
    panel_id:          int                 = -1
    bubble_type:       str                 = "bubble"
    detection_conf:    float               = 0.0

    # OCR
    raw_text:          str                 = ""
    corrected_text:    str                 = ""
    ocr_confidence:    float               = 0.0

    # Forditas
    translated_text:   str                 = ""
    refined_text:      str                 = ""
    tone:              str                 = "neutral"
    emphasis_words:    list[str]           = field(default_factory=list)

    # Rendering
    typography_preset: str                 = "bubble"
    font_size_used:    int                 = 0
    render_score:      float               = 0.0

    # Provider info
    provider_used:     str                 = ""
    inpaint_provider:  str                 = ""

    # Dirty tracking
    render_hash:       str                 = ""
    manually_edited:   bool               = False

    # Pipeline info
    skip_reason:       str                 = ""
    timings:           dict[str, float]    = field(default_factory=dict)

    @property
    def effective_text(self) -> str:
        """
        A legmagasabb minosegu elerheto szoveg.
        Human szerkesztes > finomitas > forditas > javitott OCR > nyers OCR.
        """
        if self.manually_edited and self.translated_text:
            return self.translated_text
        return (self.refined_text or
                self.translated_text or
                self.corrected_text or
                self.raw_text or "")

    @property
    def effective_ocr_text(self) -> str:
        """Legjobb OCR szoveg: javitott ha van, kulonben nyers."""
        return self.corrected_text or self.raw_text or ""

    def compute_render_hash(self) -> str:
        """
        Bubble-szintu render hash.

        Ha ez valtozik -> csak ezt a buborekot kell ujra renderelni.
        Tartalmazza: szoveg + tipografia + font + bbox.
        """
        key = (
            f"{self.effective_text}|"
            f"{self.typography_preset}|"
            f"{self.tone}|"
            f"{self.bbox}|"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def is_render_dirty(self) -> bool:
        """Kell-e ezt a buborekot ujra renderelni?"""
        if self.manually_edited:
            return True
        return self.render_hash != self.compute_render_hash()

    def mark_render_clean(self) -> None:
        self.render_hash = self.compute_render_hash()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Kerekites az olvashatosagert
        d["ocr_confidence"]  = round(d["ocr_confidence"],  4)
        d["detection_conf"]  = round(d["detection_conf"],  4)
        d["render_score"]    = round(d["render_score"],     4)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BubbleData":
        """
        Dict -> BubbleData.
        Ismeretlen mezok figyelmen kivul hagyva (schema migration).
        """
        valid = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in valid}
        return cls(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE DATA – egyetlen oldal teljes allapota
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PageData:
    """
    Egyetlen kepregeny oldal teljes feldolgozasi allapota.

    Ez a JSON-ba iro autoritativ allapot.
    A pipeline-nak mindig ezt kell prioritaskent kezelni.

    Ha a JSON mar tartalmaz forditas adatot es manually_edited=True,
    a pipeline NE generalja ujra - hasznald a meglevo adatot.
    """
    metadata_version: str                  = METADATA_VERSION
    page_id:          str                  = ""
    source_path:      str                  = ""
    output_path:      str                  = ""

    # Oldal metaadat
    image_width:      int                  = 0
    image_height:     int                  = 0
    processed_at:     str                  = ""

    # Buborek adatok
    bubbles:          list[BubbleData]     = field(default_factory=list)

    # Stage allapot
    dirty_flags:      DirtyFlags           = field(default_factory=DirtyFlags.all_dirty)
    stage_hashes:     StageHashes          = field(default_factory=StageHashes)

    # Statisztikak
    total_bubbles:    int                  = 0
    translated_count: int                  = 0
    skipped_count:    int                  = 0
    avg_ocr_conf:     float                = 0.0
    avg_render_score: float                = 0.0

    # Provider info
    providers_used:   dict[str, str]       = field(default_factory=dict)

    # Stage timingok (ms)
    timings:          dict[str, float]     = field(default_factory=dict)

    # Hiba info
    error:            Optional[str]        = None
    failed_stages:    list[str]            = field(default_factory=list)

    def get_bubble(self, bubble_id: int) -> Optional[BubbleData]:
        for b in self.bubbles:
            if b.bubble_id == bubble_id:
                return b
        return None

    def dirty_bubbles(self) -> list[BubbleData]:
        """Csak azok a buborekek, amelyek render-dirty-k."""
        return [b for b in self.bubbles if b.is_render_dirty()]

    def update_stats(self) -> None:
        """Statisztikak frissitese a bubbles lista alapjan."""
        self.total_bubbles    = len(self.bubbles)
        self.translated_count = sum(
            1 for b in self.bubbles if b.translated_text.strip())
        self.skipped_count    = sum(
            1 for b in self.bubbles if b.skip_reason)
        confs = [b.ocr_confidence for b in self.bubbles if b.ocr_confidence > 0]
        self.avg_ocr_conf     = round(sum(confs)/len(confs), 4) if confs else 0.0
        scores = [b.render_score for b in self.bubbles if b.render_score > 0]
        self.avg_render_score = round(sum(scores)/len(scores), 4) if scores else 0.0

    def add_timing(self, stage: str, elapsed_ms: float) -> None:
        self.timings[stage] = round(elapsed_ms, 1)

    def mark_stage_done(self, stage: Stage) -> None:
        self.dirty_flags.mark_done(stage)

    def mark_stage_dirty(self, stage: Stage) -> None:
        self.dirty_flags.mark_dirty(stage)

    def is_stage_needed(
        self,
        stage: Stage,
        force: bool = False,
    ) -> bool:
        """
        Kell-e ezt a stage-et futtatni?

        Human-in-the-loop: ha a bubble manually_edited,
        a render stage szukseges akkor is.
        """
        if force:
            return True
        return self.dirty_flags.is_dirty(stage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_version": self.metadata_version,
            "page_id":          self.page_id,
            "source_path":      self.source_path,
            "output_path":      self.output_path,
            "image_width":      self.image_width,
            "image_height":     self.image_height,
            "processed_at":     self.processed_at,
            "bubbles":          [b.to_dict() for b in self.bubbles],
            "dirty_flags":      self.dirty_flags.to_dict(),
            "stage_hashes":     self.stage_hashes.to_dict(),
            "total_bubbles":    self.total_bubbles,
            "translated_count": self.translated_count,
            "skipped_count":    self.skipped_count,
            "avg_ocr_conf":     self.avg_ocr_conf,
            "avg_render_score": self.avg_render_score,
            "providers_used":   self.providers_used,
            "timings":          self.timings,
            "error":            self.error,
            "failed_stages":    self.failed_stages,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PageData":
        """
        Dict -> PageData.
        Schema migration: ismeretlen mezok figyelmen kivul hagyva.
        Verzio ellenorzes: regi verziok betolthetok.
        """
        ver = d.get("metadata_version", "1.0")
        if ver != METADATA_VERSION:
            logger.warning(
                f"Metadata verzio: {ver} (aktualis: {METADATA_VERSION}) "
                "- backward compatibility mod")

        bubbles = [BubbleData.from_dict(b) for b in d.get("bubbles", [])]
        df_raw  = d.get("dirty_flags", {})
        sh_raw  = d.get("stage_hashes", {})

        return cls(
            metadata_version = d.get("metadata_version", METADATA_VERSION),
            page_id          = d.get("page_id", ""),
            source_path      = d.get("source_path", ""),
            output_path      = d.get("output_path", ""),
            image_width      = d.get("image_width", 0),
            image_height     = d.get("image_height", 0),
            processed_at     = d.get("processed_at", ""),
            bubbles          = bubbles,
            dirty_flags      = DirtyFlags.from_dict(df_raw) if df_raw
                               else DirtyFlags.all_dirty(),
            stage_hashes     = StageHashes.from_dict(sh_raw) if sh_raw
                               else StageHashes(),
            total_bubbles    = d.get("total_bubbles", len(bubbles)),
            translated_count = d.get("translated_count", 0),
            skipped_count    = d.get("skipped_count", 0),
            avg_ocr_conf     = d.get("avg_ocr_conf", 0.0),
            avg_render_score = d.get("avg_render_score", 0.0),
            providers_used   = d.get("providers_used", {}),
            timings          = d.get("timings", {}),
            error            = d.get("error"),
            failed_stages    = d.get("failed_stages", []),
        )


# ══════════════════════════════════════════════════════════════════════════════
# ATOMIC FILE IO
# ══════════════════════════════════════════════════════════════════════════════

def _atomic_write_json(path: Path, data: dict) -> None:
    """
    Atomic JSON mentes: temp file -> rename.

    Crash-safe: ha a mentes kozben megszakad,
    a regi JSON marad sertetlen.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)   # atomikus rename
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Atomic JSON mentes sikertelen [{path}]: {e}")


def _load_json(path: Path) -> Optional[dict]:
    """JSON betoltes hibatürő módon."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse hiba [{path}]: {e}")
        return None
    except Exception as e:
        logger.error(f"JSON betoltes hiba [{path}]: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE METADATA MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class PageMetadataManager:
    """
    Egy oldal JSON metadata kezelo.

    Feladatai:
      - PageData betoltese / mentese
      - Human-in-the-loop prioritas: szerkesztett JSON > generalt adat
      - Resume: meglevo JSON-bol folytatja a pipeline-t
      - Bubble-szintu dirty tracking

    Hasznalat:
        mgr  = PageMetadataManager(output_dir / "page_001")
        page = mgr.load_or_create(source_path)
        # ... pipeline fut ...
        mgr.save(page)
    """

    JSON_FILENAME = "page_data.json"

    def __init__(self, page_dir: Path) -> None:
        self.page_dir  = page_dir
        self.json_path = page_dir / self.JSON_FILENAME

    def load_or_create(
        self,
        source_path: Path,
        force_reset: bool = False,
    ) -> PageData:
        """
        PageData betoltese ha letezik, kulonben uj letrehozasa.

        Human-in-the-loop:
          Ha letezik JSON es force_reset=False:
            -> betoltjuk es megorizuk a kezzel szerkesztett adatokat
          Ha force_reset=True:
            -> minden dirty, ujraindul

        Args:
            source_path: eredeti kep utvonala
            force_reset: True = mindent ujra generalunk

        Returns:
            PageData - feltoltott allapottal vagy ures.
        """
        page_id = source_path.stem
        existing = _load_json(self.json_path)

        if existing and not force_reset:
            page = PageData.from_dict(existing)
            # Ellenorzés: a forras fajl valtozot-e?
            src_hash = self._hash_file(source_path)
            if src_hash != page.stage_hashes.layout_hash:
                logger.info(
                    f"[{page_id}] Forras kep megvaltozott – "
                    "layout + ocr dirty")
                page.mark_stage_dirty(Stage.LAYOUT)
                page.stage_hashes.layout_hash = src_hash
            else:
                logger.info(
                    f"[{page_id}] Meglevo metadata betoltve – "
                    f"dirty stages: {self._dirty_stages(page.dirty_flags)}")
            return page

        # Uj PageData
        page = PageData(
            page_id      = page_id,
            source_path  = str(source_path),
            output_path  = str(self.page_dir),
            processed_at = self._now(),
        )
        page.stage_hashes.layout_hash = self._hash_file(source_path)
        logger.info(f"[{page_id}] Uj PageData letrehozva")
        return page

    def save(self, page: PageData) -> None:
        """
        PageData mentese atomikusan.

        Statisztikak frissitese mentes elott.
        """
        page.update_stats()
        page.processed_at = self._now()
        self.page_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.json_path, page.to_dict())
        logger.debug(f"PageData mentve: {self.json_path}")

    def load(self) -> Optional[PageData]:
        """PageData betoltese ha letezik."""
        d = _load_json(self.json_path)
        return PageData.from_dict(d) if d else None

    def exists(self) -> bool:
        return self.json_path.exists()

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Fajl tartalom hash (MD5 elso 64KB alapjan - gyors)."""
        if not path.exists():
            return ""
        try:
            h = hashlib.md5()
            with open(path, "rb") as f:
                h.update(f.read(65536))
            return h.hexdigest()[:16]
        except Exception:
            return ""

    @staticmethod
    def _dirty_stages(flags: DirtyFlags) -> list[str]:
        return [
            s.value for s in Stage
            if getattr(flags, s.value, True)
        ]

    @staticmethod
    def _now() -> str:
        from datetime import datetime
        return datetime.now().isoformat(timespec="seconds")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE HASH – stage cache invalidalas
# ══════════════════════════════════════════════════════════════════════════════

def compute_image_hash(image_path: Path) -> str:
    """Kep fajl hash (MD5, elso 64KB)."""
    return PageMetadataManager._hash_file(image_path)


def compute_config_hash(**params) -> str:
    """
    Konfiguracioós hash adott stage parameterekbol.

    Pelda:
      font_hash = compute_config_hash(
          font_path=cfg.rendering.font_bold,
          font_size_max=cfg.rendering.font_size_max,
      )
    """
    key = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# MANIFEST
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ManifestEntry:
    """Egyetlen oldal bejegyze a manifestben."""
    page_id:          str
    source_path:      str
    status:           str          # "success" | "failed" | "skipped" | "partial"
    total_bubbles:    int          = 0
    translated_count: int          = 0
    avg_ocr_conf:     float        = 0.0
    avg_render_score: float        = 0.0
    providers_used:   dict         = field(default_factory=dict)
    elapsed_sec:      float        = 0.0
    error:            Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Manifest:
    """
    Teljes batch futtas összefoglalo.

    Generalt a batch vegén, tartalmaz minden oldal statuszat
    es aggregalt statisztikakat.
    """
    metadata_version:  str                = METADATA_VERSION
    created_at:        str                = ""
    total_pages:       int                = 0
    success_count:     int                = 0
    failed_count:      int                = 0
    skipped_count:     int                = 0
    partial_count:     int                = 0
    total_elapsed_sec: float              = 0.0
    avg_ocr_conf:      float              = 0.0
    avg_render_score:  float              = 0.0
    providers_summary: dict[str, int]     = field(default_factory=dict)
    pages:             list[ManifestEntry] = field(default_factory=list)

    def add_page(self, entry: ManifestEntry) -> None:
        self.pages.append(entry)
        self.total_pages = len(self.pages)
        counts = {"success": 0, "failed": 0, "skipped": 0, "partial": 0}
        for p in self.pages:
            counts[p.status] = counts.get(p.status, 0) + 1
        self.success_count = counts["success"]
        self.failed_count  = counts["failed"]
        self.skipped_count = counts["skipped"]
        self.partial_count = counts["partial"]
        # Aggregat statisztikak
        confs   = [p.avg_ocr_conf    for p in self.pages if p.avg_ocr_conf > 0]
        scores  = [p.avg_render_score for p in self.pages if p.avg_render_score > 0]
        self.avg_ocr_conf    = round(sum(confs)/len(confs), 4) if confs else 0.0
        self.avg_render_score= round(sum(scores)/len(scores),4) if scores else 0.0
        # Provider összesito
        for p in self.pages:
            for k, v in p.providers_used.items():
                key = f"{k}:{v}"
                self.providers_summary[key] =                     self.providers_summary.get(key, 0) + 1

    def to_dict(self) -> dict:
        return {
            "metadata_version":  self.metadata_version,
            "created_at":        self.created_at,
            "total_pages":       self.total_pages,
            "success_count":     self.success_count,
            "failed_count":      self.failed_count,
            "skipped_count":     self.skipped_count,
            "partial_count":     self.partial_count,
            "total_elapsed_sec": round(self.total_elapsed_sec, 1),
            "avg_ocr_conf":      self.avg_ocr_conf,
            "avg_render_score":  self.avg_render_score,
            "providers_summary": self.providers_summary,
            "pages":             [p.to_dict() for p in self.pages],
        }

    def save(self, output_dir: Path) -> None:
        """Manifest mentese atomikusan."""
        from datetime import datetime
        self.created_at = datetime.now().isoformat(timespec="seconds")
        _atomic_write_json(output_dir / "manifest.json", self.to_dict())
        logger.info(
            f"Manifest mentve: {output_dir / 'manifest.json'} "
            f"({self.total_pages} oldal, "
            f"{self.success_count} ok, {self.failed_count} hiba)")
