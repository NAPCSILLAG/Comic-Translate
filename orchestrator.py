"""
orchestrator.py - PipelineOrchestrator: a kepregeny-fordito engine szive.

Architektura:
  StageExecutor:       centralizalt stage vegrehajtás (timing, log, exception, metadata)
  ProviderPool:        VRAM-tudatos provider lifecycle (lazy load, aggressive release)
  BubbleLayerCache:    bubble-szintu RGBA layer cache (B opció: csak dirty buborék renderel)
  PipelineOrchestrator: fő orchestrator, stage koordinalas, batch, resume

Pipeline sorrend (helyes):
  layout → ocr → ocr_correction → translation → inpaint → render → composite → export

Tervezesi elvek:
  - Stage isolation: egy hiba nem torol le mas stage-et / oldalt
  - Resume: dirty flag + hash alapu stage skip
  - Bubble-level rerender: cache-elt clean RGBA layerek + dirty regeneralas
  - VRAM: szekvencialis load/release, ProviderPool aggressziv cleanup
  - Determinizmus: hash-alapu cache reuse (nem temperature=0)
  - Atomic metadata save minden stage utan
  - Serializable state: nincs hidden in-memory-only state
  - GUI-ready: minden allapot metadata-bol visszatoltheto
"""
from __future__ import annotations

import gc
import hashlib
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import cv2
import numpy as np
from PIL import Image

from config import cfg, Config
from metadata import (
    BubbleData, DirtyFlags, Manifest, ManifestEntry,
    PageData, PageMetadataManager, Stage,
    compute_config_hash, METADATA_VERSION,
)
from providers.base import (
    BaseProvider, ConfidenceGate, ProviderContext,
    ProviderRegistry, ProviderResult, ProviderStatus,
)
from providers.translation import (
    OllamaProvider, TranslationRequest,
    get_provider as get_translation_provider,
)
from providers.vision.gemma4 import Gemma4Provider
from providers.vision.qwen2_vl import Qwen2VLProvider, VisionResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """
    Runtime pipeline konfiguracio - CLI-bol teljes egeszeben felulirható.
    Nincs hardcoded ertek: minden a Config-ból szarmazik alapertelmezetten.
    """
    # I/O
    input_dir:  Path = field(default_factory=lambda: cfg.paths.input_dir)
    output_dir: Path = field(default_factory=lambda: cfg.paths.output_dir)

    # Stage kapcsolók
    skip_vlm:      bool = False
    skip_inpaint:  bool = False
    skip_gemma:    bool = False
    dry_run:       bool = False
    force:         bool = False

    # Resume
    resume:        bool = False   # meglevo metadata prioritas
    no_overwrite:  bool = False

    # Debug
    debug:         bool = False
    save_stages:   bool = False

    # Batch
    pause_after:   int  = 0

    # Provider
    translation_provider: str = "ollama"
    translation_model:    str = ""

    # Thresholds
    ocr_correction_threshold:     float = 0.70
    translation_refine_threshold: float = 0.75
    ocr_backend:                  str   = ""   # "" = config default

    # Device / rendering
    device:      str = ""
    supersample: int = 0

    # Warmup
    warmup_lightweight: bool = True   # Ollama elore ellenorzes

    def effective_model(self) -> str:
        return self.translation_model or cfg.translation.model

    def effective_device(self) -> str:
        return self.device or cfg.device.device

    def effective_supersample(self) -> int:
        return self.supersample or cfg.rendering.supersample_factor


# ══════════════════════════════════════════════════════════════════════════════
# STAGE EXECUTOR – centralizalt stage logika
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StageOutcome:
    """Egyetlen stage vegrehajtasanak teljes eredmenye."""
    stage:       Stage
    success:     bool
    skipped:     bool      = False
    elapsed_ms:  float     = 0.0
    error:       str       = ""
    warnings:    list[str] = field(default_factory=list)
    fallback_used: bool    = False
    provider_id: str       = ""

    def __str__(self) -> str:
        if self.skipped:
            return f"{self.stage.value}:SKIP"
        status = "OK" if self.success else "ERR"
        fb = "+fallback" if self.fallback_used else ""
        return f"{self.stage.value}:{status}{fb} {self.elapsed_ms:.0f}ms"


class StageExecutor:
    """
    Centralizalt stage vegrehajtasi wrapper.

    Automatikusan kezeli:
      - Timing meres
      - Strukturalt logging
      - Exception isolation (egy stage hiba nem torol masik stage-et)
      - Metadata save hiba eseten is
      - Warning aggregalas
      - Stage status tranzicio (dirty -> done / failed)
      - Skip logika (dirty flag + force alapjan)

    Hasznalat:
        with StageExecutor(page, Stage.OCR, mgr) as exec:
            result = do_ocr(...)
            exec.set_provider("ppocr_v5")
        # automatikus: timing + log + exception + metadata save
    """

    def __init__(
        self,
        page:  PageData,
        stage: Stage,
        mgr:   PageMetadataManager,
        force: bool = False,
    ) -> None:
        self.page    = page
        self.stage   = stage
        self.mgr     = mgr
        self.force   = force
        self._outcome = StageOutcome(stage=stage, success=False)
        self._t0:    float = 0.0

    def __enter__(self) -> "StageExecutor":
        self._t0 = time.perf_counter()
        logger.info(f"  [STAGE] {self.stage.value.upper()}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = (time.perf_counter() - self._t0) * 1000
        self._outcome.elapsed_ms = elapsed

        if exc_type is not None and exc_type is not KeyboardInterrupt:
            self._outcome.success = False
            self._outcome.error   = str(exc_val)
            self.page.failed_stages.append(self.stage.value)
            logger.error(
                f"  [{self.stage.value}] HIBA: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb),
            )
            # Metadata mentese hiba eseten is
            try:
                self.page.add_timing(self.stage.value, elapsed)
                self.mgr.save(self.page)
            except Exception:
                pass
            return True   # elnyeljük a kivételt (stage isolation)

        if not self._outcome.skipped:
            self._outcome.success = True
            self.page.mark_stage_done(self.stage)
            self.page.add_timing(self.stage.value, elapsed)
            self._log_outcome()
            try:
                self.mgr.save(self.page)
            except Exception as e:
                logger.warning(f"  Metadata save hiba: {e}")

        return False

    def skip(self, reason: str = "") -> None:
        """Stage kihagyasa - nem fut, nem dirty."""
        self._outcome.skipped = True
        self._outcome.success = True
        elapsed = (time.perf_counter() - self._t0) * 1000
        self._outcome.elapsed_ms = elapsed
        logger.debug(
            f"  [{self.stage.value}] SKIP"
            f"{': ' + reason if reason else ''}")

    def warn(self, msg: str) -> None:
        self._outcome.warnings.append(msg)
        logger.warning(f"  [{self.stage.value}] WARN: {msg}")

    def set_provider(self, provider_id: str) -> None:
        self._outcome.provider_id = provider_id
        self.page.providers_used[self.stage.value] = provider_id

    def set_fallback(self) -> None:
        self._outcome.fallback_used = True

    def should_run(self) -> bool:
        """Kell-e ezt a stage-et futtatni?"""
        return self.page.is_stage_needed(self.stage, self.force)

    @property
    def outcome(self) -> StageOutcome:
        return self._outcome

    def _log_outcome(self) -> None:
        fb  = " [fallback]" if self._outcome.fallback_used else ""
        wrn = f" | {len(self._outcome.warnings)} warn"               if self._outcome.warnings else ""
        pid = f" | {self._outcome.provider_id}"               if self._outcome.provider_id else ""
        logger.info(
            f"  [{self.stage.value}] OK "
            f"{self._outcome.elapsed_ms:.0f}ms{fb}{wrn}{pid}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER POOL – VRAM-tudatos provider lifecycle
# ══════════════════════════════════════════════════════════════════════════════

# Provider suly: mennyi VRAM-ot foglal (GB)
_PROVIDER_VRAM: dict[str, float] = {
    "qwen2_vl":   4.5,
    "gemma4":     0.0,   # Ollama API, nem direkt VRAM
    "ollama":     0.0,   # API
    "lama_onnx":  2.0,
    "ppocr_v5":   0.5,
    "easyocr":    1.0,
}

_LIGHTWEIGHT_THRESHOLD = 0.1   # ennyi VRAM alatt "lightweight" (nem unloadoljuk)


class ProviderPool:
    """
    VRAM-tudatos provider lifecycle manager.

    Feladatai:
      - Lazy loading: csak akkor tolt be, amikor kell
      - Aggressive release: nagy modellek azonnal unloadolnak hasznalat utan
      - Lightweight provider-ek warm maradhatnak (Ollama, rule-based)
      - Capability tracking: mikor elerheto melyik provider
      - VRAM budget tracking (kozelito)

    Hasznalat:
        with pool.acquire("qwen2_vl") as provider:
            result = provider.timed_run(...)
        # automatikus release + VRAM cleanup
    """

    def __init__(
        self,
        registry:     ProviderRegistry,
        max_vram_gb:  float = 8.0,
        keep_warm:    list[str] = None,
    ) -> None:
        self._registry    = registry
        self._max_vram_gb = max_vram_gb
        self._keep_warm   = set(keep_warm or ["ollama"])
        self._loaded:     set[str] = set()

    @contextmanager
    def acquire(
        self,
        provider_id: str,
        fallback_id: Optional[str] = None,
    ) -> Generator[Optional[BaseProvider], None, None]:
        """
        Provider acquire context manager.

        Automatikus load elott + release utan (heavy providereknel).
        Ha a provider nem elerheto es van fallback: azt adja vissza.
        Ha egyik sem elerheto: None-t ad vissza (pipeline folytatodik).

        Hasznalat:
            with pool.acquire("qwen2_vl", fallback_id="easyocr") as p:
                if p:
                    result = p.timed_run(...)
        """
        provider = self._registry.get(provider_id)

        if provider is None or not provider.enabled:
            if fallback_id:
                provider = self._registry.get(fallback_id)
            if provider is None:
                yield None
                return

        # Load ha szukseges
        loaded_now = False
        if not provider.is_ready:
            success = provider.load()
            if not success:
                logger.warning(
                    f"[ProviderPool] Betoltes sikertelen: {provider_id}")
                if fallback_id:
                    fb = self._registry.get(fallback_id)
                    if fb and fb.load():
                        yield fb
                        self._maybe_release(fb)
                        return
                yield None
                return
            loaded_now = True
            self._loaded.add(provider_id)

        try:
            yield provider
        finally:
            # Heavy provider: mindig release
            # Lightweight provider: keep warm ha konfiguralt
            self._maybe_release(provider, force=loaded_now)

    def _maybe_release(
        self,
        provider: BaseProvider,
        force: bool = False,
    ) -> None:
        """Felszabaditja a providert ha heavy vagy force=True."""
        pid   = provider.provider_id
        vram  = _PROVIDER_VRAM.get(pid, 0.0)
        heavy = vram > _LIGHTWEIGHT_THRESHOLD
        warm  = pid in self._keep_warm

        if heavy and not warm:
            provider.release()
            self._loaded.discard(pid)
            _vram_cleanup()
            logger.debug(
                f"[ProviderPool] Released: {pid} "
                f"(freed ~{vram:.1f}GB)")
        elif force and not warm:
            provider.release()
            self._loaded.discard(pid)

    def release_all(self) -> None:
        """Minden provider felszabaditasa (batch vegen)."""
        self._registry.release_all()
        self._loaded.clear()
        _vram_cleanup()
        logger.info("[ProviderPool] Minden provider felszabaditva")

    def loaded_providers(self) -> list[str]:
        return list(self._loaded)


# ══════════════════════════════════════════════════════════════════════════════
# BUBBLE LAYER CACHE – bubble-szintu RGBA PNG cache
# ══════════════════════════════════════════════════════════════════════════════

class BubbleLayerCache:
    """
    Bubble-szintu RGBA text layer cache (B opcio).

    Minden buborékhoz kuelonallo RGBA PNG fa cache-ben.
    Csak dirty buborekot generaljuk ujra - a tobb marad a cache-bol.

    File struktura:
        output/page_001/.cache/layers/bubble_0042_<hash>.png

    Hash tartalma: szoveg + tipografia + font + bbox
    Ha a hash egyezik: cache hit, nem renderelünk ujra.
    """

    def __init__(self, page_dir: Path) -> None:
        self.cache_dir = page_dir / ".cache" / "layers"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, bubble: BubbleData) -> Optional[Image.Image]:
        """
        Cache hit ha a bubble render hash nem valtozott.

        Returns:
            PIL RGBA Image ha cache hit, None ha miss/dirty.
        """
        if not bubble.render_hash:
            return None
        path = self._path(bubble.bubble_id, bubble.render_hash)
        if path.exists():
            try:
                img = Image.open(path).convert("RGBA")
                logger.debug(
                    f"  Layer cache HIT: bubble #{bubble.bubble_id}")
                return img
            except Exception:
                return None
        return None

    def save(self, bubble: BubbleData, layer: Image.Image) -> None:
        """RGBA layer mentese a cache-be."""
        h    = bubble.compute_render_hash()
        path = self._path(bubble.bubble_id, h)
        try:
            layer.save(str(path), format="PNG")
            bubble.render_hash = h
            logger.debug(
                f"  Layer cache SAVE: bubble #{bubble.bubble_id}")
        except Exception as e:
            logger.warning(f"  Layer cache save hiba: {e}")

    def invalidate(self, bubble_id: int) -> None:
        """Egy buborek cache torles."""
        for f in self.cache_dir.glob(f"bubble_{bubble_id:04d}_*.png"):
            f.unlink(missing_ok=True)

    def invalidate_all(self) -> None:
        """Teljes cache torles."""
        for f in self.cache_dir.glob("bubble_*.png"):
            f.unlink(missing_ok=True)

    def _path(self, bubble_id: int, render_hash: str) -> Path:
        return self.cache_dir / f"bubble_{bubble_id:04d}_{render_hash}.png"


# ══════════════════════════════════════════════════════════════════════════════
# SEGÉDFÜGGVÉNYEK
# ══════════════════════════════════════════════════════════════════════════════

def _vram_cleanup() -> None:
    """GPU + CPU memory cleanup."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def _load_image(path: Path) -> Optional[np.ndarray]:
    """Kep betoltese hibaturo modon."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Kep nem toltheto be: {path}")
        return img
    except Exception as e:
        logger.error(f"Kep betoltes hiba [{path}]: {e}")
        return None


def _collect_images(input_dir: Path, formats: tuple) -> list[Path]:
    """Tamogatott kepfajlok osszegyujtese az input mappaban."""
    images: list[Path] = []
    for ext in formats:
        images.extend(sorted(input_dir.glob(f"*{ext}")))
        images.extend(sorted(input_dir.glob(f"*{ext.upper()}")))
    seen: set[str] = set()
    unique: list[Path] = []
    for p in images:
        if p.name.lower() not in seen:
            seen.add(p.name.lower())
            unique.append(p)
    return sorted(unique, key=lambda p: p.name)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class PipelineOrchestrator:
    """
    A kepregeny-fordito engine fo orchestratora.

    Egyetlen oldal feldolgozasi ciklusa:
      1. PageData load or create (resume support)
      2. StageExecutor wrapper minden stage-hez
      3. ProviderPool acquire/release per stage
      4. BubbleLayerCache: csak dirty buborekot renderel
      5. Atomic metadata save minden stage utan
      6. Manifest entry generalas
    """

    def __init__(self, pcfg: PipelineConfig) -> None:
        self.pcfg      = pcfg
        self.registry  = ProviderRegistry()
        self.pool:     Optional[ProviderPool] = None
        self._manifest = Manifest()
        self._t_batch  = time.perf_counter()

        # Lazy rendering modulok
        self._renderer_inst   = None
        self._compositor_inst = None
        self._inpainter_inst  = None

        self._init_providers()
        logger.info(
            f"PipelineOrchestrator init | "
            f"device={self.pcfg.effective_device()} | "
            f"dry_run={self.pcfg.dry_run} | "
            f"force={self.pcfg.force}"
        )

    # ── Provider init ─────────────────────────────────────────────────────────

    def _init_providers(self) -> None:
        """Provider-ek regisztralasa (nem tolti be a modellt meg)."""

        # Forditas provider
        trans = get_translation_provider(
            self.pcfg.translation_provider,
            model=self.pcfg.effective_model(),
        )
        self.registry.register(trans)

        # Qwen2-VL
        if not self.pcfg.skip_vlm:
            self.registry.register(
                Qwen2VLProvider(enabled=cfg.vision.enabled))

        # Gemma4
        if not self.pcfg.skip_gemma:
            self.registry.register(
                Gemma4Provider(enabled=True))

        # ProviderPool
        self.pool = ProviderPool(
            registry=self.registry,
            max_vram_gb=cfg.device.vram_free_gb(),
            keep_warm=["ollama"] if self.pcfg.warmup_lightweight else [],
        )

        # Lightweight warmup: Ollama elerheto-e?
        if self.pcfg.warmup_lightweight:
            trans_p = self.registry.get(self.pcfg.translation_provider)
            if trans_p:
                trans_p.load()

        logger.info(f"Providerek: {self.registry.list_ids()}")

    # ── Lazy renderer getters ─────────────────────────────────────────────────

    def _renderer(self):
        if self._renderer_inst is None:
            from rendering import get_renderer
            self._renderer_inst = get_renderer()
            ss = self.pcfg.effective_supersample()
            if ss != cfg.rendering.supersample_factor:
                self._renderer_inst.set_supersample(ss)
        return self._renderer_inst

    def _compositor(self):
        if self._compositor_inst is None:
            from compositor import get_compositor
            self._compositor_inst = get_compositor()
        return self._compositor_inst

    def _inpainter(self):
        if self._inpainter_inst is None:
            from inpainting import get_inpainter
            self._inpainter_inst = get_inpainter()
        return self._inpainter_inst

    # ── Batch ─────────────────────────────────────────────────────────────────

    def run_batch(self, image_paths: list[Path]) -> Manifest:
        """
        Batch feldolgozas tqdm progress bar-ral.

        Failure isolation: egy kep hibaja nem torolja a batcht.
        Pause support: --pause-after-pages.
        """
        from tqdm import tqdm
        total = len(image_paths)
        logger.info(
            f"Batch: {total} kep | "
            f"{self.pcfg.input_dir} -> {self.pcfg.output_dir}"
        )
        self.pcfg.output_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(
            total=total, desc="Forditas", unit="kep",
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}]",
        ) as pbar:
            for i, img_path in enumerate(image_paths):
                t0    = time.perf_counter()
                logger.info(
                    f"[PAGE {i+1}/{total}] {img_path.name}")
                entry = self._process_page(img_path)
                entry.elapsed_sec = round(time.perf_counter() - t0, 2)
                self._manifest.add_page(entry)

                pbar.update(1)
                pbar.set_postfix({
                    "ok":  self._manifest.success_count,
                    "err": self._manifest.failed_count,
                    "utolso": img_path.name[:14],
                })

                if (self.pcfg.pause_after > 0 and
                        (i + 1) % self.pcfg.pause_after == 0 and
                        i + 1 < total):
                    self._pause(i + 1, total)

        self.pool.release_all()
        self._manifest.total_elapsed_sec = (
            time.perf_counter() - self._t_batch)
        self._manifest.save(self.pcfg.output_dir)
        self._log_batch_summary()
        return self._manifest

    def run_single(self, img_path: Path) -> ManifestEntry:
        """Egyetlen kep feldolgozasa (CLI single-image mod)."""
        self.pcfg.output_dir.mkdir(parents=True, exist_ok=True)
        t0    = time.perf_counter()
        entry = self._process_page(img_path)
        entry.elapsed_sec = round(time.perf_counter() - t0, 2)
        self.pool.release_all()
        return entry

    # ── Egyetlen oldal ────────────────────────────────────────────────────────

    def _process_page(self, img_path: Path) -> ManifestEntry:
        """
        Egyetlen kep teljes pipeline-on at.
        Failure isolation: barmely unhandled exception -> failed entry.
        """
        page_id  = img_path.stem
        page_dir = self.pcfg.output_dir / page_id
        mgr      = PageMetadataManager(page_dir)

        try:
            image = _load_image(img_path)
            if image is None:
                return _failed_entry(page_id, img_path,
                                     "Kep betoltes sikertelen")

            page = mgr.load_or_create(
                img_path, force_reset=self.pcfg.force)
            page.image_width  = image.shape[1]
            page.image_height = image.shape[0]
            mgr.save(page)

            # Korai skip
            if (self.pcfg.no_overwrite and
                    page.dirty_flags.all_clean() and mgr.exists()):
                logger.info(f"[{page_id}] Kihagyva (mar feldolgozva)")
                return _page_to_entry(page, img_path, "skipped")

            layer_cache = BubbleLayerCache(page_dir)

            # ── STAGE 1: Layout ───────────────────────────────────────────
            with StageExecutor(page, Stage.LAYOUT, mgr,
                               self.pcfg.force) as ex:
                if not ex.should_run():
                    ex.skip("clean")
                else:
                    self._run_layout(page, image, img_path, ex)

            if not page.bubbles:
                logger.warning(f"[{page_id}] Nincs buborek - skip")
                return _page_to_entry(page, img_path, "skipped")

            # ── STAGE 2: OCR ──────────────────────────────────────────────
            with StageExecutor(page, Stage.OCR, mgr,
                               self.pcfg.force) as ex:
                if not ex.should_run():
                    ex.skip("clean")
                else:
                    self._run_ocr(page, image, ex)

            # ── STAGE 3: OCR Correction ───────────────────────────────────
            with StageExecutor(page, Stage.OCR_CORRECT, mgr,
                               self.pcfg.force) as ex:
                if not ex.should_run() or (
                        self.pcfg.skip_vlm and self.pcfg.skip_gemma):
                    ex.skip()
                else:
                    self._run_ocr_correction(page, image, ex)

            # ── STAGE 4: Translation ──────────────────────────────────────
            with StageExecutor(page, Stage.TRANSLATION, mgr,
                               self.pcfg.force) as ex:
                if not ex.should_run():
                    ex.skip("clean")
                else:
                    self._run_translation(page, ex)

            # ── STAGE 5: Refinement ───────────────────────────────────────
            with StageExecutor(page, Stage.REFINEMENT, mgr,
                               self.pcfg.force) as ex:
                if not ex.should_run() or self.pcfg.skip_gemma:
                    ex.skip()
                else:
                    self._run_refinement(page, ex)

            # Dry-run: itt megallunk
            if self.pcfg.dry_run:
                logger.info(f"[{page_id}] Dry-run: render/inpaint skip")
                return _page_to_entry(page, img_path, "success")

            # ── STAGE 6: Inpaint ──────────────────────────────────────────
            inpainted: np.ndarray
            with StageExecutor(page, Stage.INPAINT, mgr,
                               self.pcfg.force) as ex:
                if self.pcfg.skip_inpaint:
                    ex.skip("--skip-inpaint")
                    inpainted = image.copy()
                elif not ex.should_run():
                    cached = _load_stage_img(page_dir, "inpainted")
                    inpainted = cached if cached is not None else image.copy()
                    ex.skip("cached")
                else:
                    inpainted = self._run_inpaint(page, image, ex)
                    if self.pcfg.save_stages:
                        _save_stage_img(inpainted, page_dir, "inpainted")

            # ── STAGE 7: Render ───────────────────────────────────────────
            text_layers:  list[Image.Image]
            debug_layers: list[Optional[Image.Image]]
            with StageExecutor(page, Stage.RENDER, mgr,
                               self.pcfg.force) as ex:
                if not ex.should_run():
                    ex.skip("clean")
                    text_layers, debug_layers = [], []
                else:
                    text_layers, debug_layers = self._run_render(
                        page, inpainted, layer_cache, ex)

            # ── STAGE 8: Composite ────────────────────────────────────────
            final: np.ndarray
            with StageExecutor(page, Stage.COMPOSITE, mgr,
                               self.pcfg.force) as ex:
                if not ex.should_run():
                    cached = _load_stage_img(page_dir, "composite")
                    final  = cached if cached is not None else inpainted
                    ex.skip("cached")
                else:
                    final = self._run_composite(
                        page, image, inpainted,
                        text_layers, debug_layers, ex)
                    if self.pcfg.save_stages:
                        _save_stage_img(final, page_dir, "composite")

            # ── STAGE 9: Export ───────────────────────────────────────────
            with StageExecutor(page, Stage.EXPORT, mgr,
                               self.pcfg.force) as ex:
                self._run_export(page, final, page_dir, img_path, ex)

            # Debug artifacts
            if self.pcfg.debug:
                self._save_debug(
                    page, image, inpainted, final, page_dir)

            # Oldal összefoglalo
            page.update_stats()
            self._log_page_summary(page)
            return _page_to_entry(page, img_path, "success")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(
                f"[{page_id}] Kritikus hiba: {e}", exc_info=True)
            try:
                page.error = str(e)
                mgr.save(page)
            except Exception:
                pass
            return _failed_entry(page_id, img_path, str(e))
        finally:
            _vram_cleanup()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE IMPLEMENTACIÓK
    # ══════════════════════════════════════════════════════════════════════════

    def _run_layout(
        self,
        page:     PageData,
        image:    np.ndarray,
        img_path: Path,
        ex:       StageExecutor,
    ) -> None:
        from layout import detect_bubbles
        bubble_dicts = detect_bubbles(image, img_path)
        ex.set_provider("yolo")

        new_bubbles: list[BubbleData] = []
        for d in bubble_dicts:
            existing = page.get_bubble(d["id"])
            if existing and existing.manually_edited:
                new_bubbles.append(existing)
                continue
            new_bubbles.append(BubbleData(
                bubble_id      = d["id"],
                bbox           = d["bbox"],
                reading_order  = d["order"],
                panel_id       = d.get("panel_id", -1),
                bubble_type    = d.get("type", "bubble"),
                detection_conf = d.get("confidence", 0.0),
            ))
        page.bubbles = new_bubbles
        logger.info(f"  Layout: {len(new_bubbles)} buborek")

    def _run_ocr(
        self,
        page:  PageData,
        image: np.ndarray,
        ex:    StageExecutor,
    ) -> None:
        from ocr import get_ocr
        ocr = get_ocr()

        # Oldal-kezdeti reset: PPOCRv5 session-szamlalo nullazasa
        ocr.reset_session()

        # CLI / config backend override alkalmazasa
        if self.pcfg.ocr_backend:
            ocr.set_backend(self.pcfg.ocr_backend)

        bubble_dicts = [b.to_dict() for b in page.bubbles]
        
        enriched = []
        try:
            enriched = ocr.process_page(image, bubble_dicts)
        except Exception as e:
            ex.warn(f"Fatal error during OCR process_page execution: {e}")
            logger.error(f"Fatal OCR failure, executing fallback for all page bubbles: {e}", exc_info=True)
            # Safe fallback: create empty enrich data for all bubbles to prevent pipeline termination
            for b in bubble_dicts:
                b_fallback = dict(b)
                b_fallback["raw_text"] = ""
                b_fallback["ocr_confidence"] = 0.0
                b_fallback["ocr_results"] = []
                enriched.append(b_fallback)

        ex.set_provider(f"ppocr_v5/{ocr._backend}")

        for bd, ed in zip(page.bubbles, enriched):
            if bd.manually_edited:
                continue
            bd.raw_text       = ed.get("raw_text", "")
            bd.ocr_confidence = ed.get("ocr_confidence", 0.0)
            # Megjegyzes: az OCR polygon NEM irja felul a YOLO bbox-ot!
            # Az OCR csak az elso szovegsor polygonját adja vissza (kisebb terulet),
            # a buborek hatarait a layout.py YOLO detektora tartalmazza helyesen.

        found = sum(1 for b in page.bubbles if b.raw_text)
        if found == 0:
            ex.warn("Egy buborékban sem talalt szöveget az OCR")
        logger.info(
            f"  OCR: {found}/{len(page.bubbles)} buborekban szöveg")

    def _run_ocr_correction(
        self,
        page:  PageData,
        image: np.ndarray,
        ex:    StageExecutor,
    ) -> None:
        ocr_gate = ConfidenceGate(
            self.pcfg.ocr_correction_threshold, mode="below")

        # Qwen2-VL: oldal-szintu vizualis elemzes
        if not self.pcfg.skip_vlm:
            with self.pool.acquire("qwen2_vl") as qwen:
                if qwen:
                    avg_conf  = (
                        sum(b.ocr_confidence for b in page.bubbles)
                        / max(len(page.bubbles), 1)
                    )
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result    = qwen.timed_run(
                        image_rgb=image_rgb,
                        ocr_text="",
                        ocr_confidence=avg_conf,
                    )
                    if result.success and result.data:
                        vr: VisionResult = result.data
                        for bd in page.bubbles:
                            if not bd.manually_edited:
                                bd.tone = vr.tone
                        ex.set_provider("qwen2_vl")

        # Gemma4: buborek-szintu OCR korrekció
        corrected = 0
        if not self.pcfg.skip_gemma:
            with self.pool.acquire("gemma4") as gemma:
                if gemma:
                    for bd in page.bubbles:
                        if bd.manually_edited or not bd.raw_text:
                            continue
                        if not ocr_gate.should_run(bd.ocr_confidence):
                            continue
                        result = gemma.correct_ocr(
                            bd.raw_text, bd.ocr_confidence)
                        if result.success and result.data:
                            bd.corrected_text = result.data.corrected_text
                            corrected += 1

        logger.info(f"  OCR korrekció: {corrected} javitva")

    def _run_translation(
        self,
        page: PageData,
        ex:   StageExecutor,
    ) -> None:
        from rendering import is_sfx
        provider_id = self.pcfg.translation_provider
        ex.set_provider(provider_id)
        page_context: list[str] = []
        translated = 0

        with self.pool.acquire(provider_id) as trans:
            if trans is None:
                ex.warn(f"Forditas provider nem elerheto: {provider_id}")
                return

            for bd in sorted(page.bubbles, key=lambda b: b.reading_order):
                if bd.manually_edited and bd.translated_text:
                    page_context.append(bd.translated_text)
                    continue
                source = bd.effective_ocr_text
                if not source.strip():
                    continue
                if is_sfx(source):
                    bd.skip_reason = "sfx"
                    continue
                req    = TranslationRequest(
                    source_text    = source,
                    tone           = bd.tone,
                    bubble_type    = bd.bubble_type,
                    page_context   = page_context[-3:],
                    ocr_confidence = bd.ocr_confidence,
                )
                result = trans.timed_run(req)
                if result.success and result.data:
                    bd.translated_text      = result.data.translated_text
                    bd.provider_used        = result.provider_id
                    bd.timings["translation"] = result.elapsed_ms
                    if bd.translated_text:
                        page_context.append(bd.translated_text)
                        translated += 1
                else:
                    ex.warn(
                        f"Forditas hiba buborek #{bd.bubble_id}: "
                        f"{result.error}")

        logger.info(
            f"  Forditas: {translated}/{len(page.bubbles)}")

    def _run_refinement(
        self,
        page: PageData,
        ex:   StageExecutor,
    ) -> None:
        refined = 0
        with self.pool.acquire("gemma4") as gemma:
            if gemma is None:
                ex.skip("gemma4 nem elerheto")
                return
            for bd in page.bubbles:
                if bd.manually_edited or not bd.translated_text:
                    continue
                result = gemma.refine_translation(
                    source=bd.effective_ocr_text,
                    translation=bd.translated_text,
                    tone=bd.tone,
                    translation_confidence=0.85,
                )
                if result.success and result.data and result.data.was_changed:
                    bd.refined_text = result.data.refined_text
                    refined += 1
        logger.info(f"  Refinement: {refined} finomitva")

    def _run_inpaint(
        self,
        page:  PageData,
        image: np.ndarray,
        ex:    StageExecutor,
    ) -> np.ndarray:
        inpainter    = self._inpainter()
        bubble_dicts = []
        for bd in page.bubbles:
            d = bd.to_dict()
            d["translated_text"] = bd.effective_text
            bubble_dicts.append(d)

        result = inpainter.inpaint_page(image, bubble_dicts)
        ex.set_provider(
            page.providers_used.get("inpaint", "lama_onnx"))
        _vram_cleanup()
        return result

    def _run_render(
        self,
        page:        PageData,
        image:       np.ndarray,
        layer_cache: BubbleLayerCache,
        ex:          StageExecutor,
    ) -> tuple[list[Image.Image], list[Optional[Image.Image]]]:
        """
        Bubble-szintu render – csak dirty buborékot renderel ujra.

        1. Minden buborekra: cache HIT -> reuse
        2. Cache MISS (dirty) -> renderer.render_bubble()
        3. Uj layer mentese cache-be
        """
        renderer     = self._renderer()
        h, w         = image.shape[:2]
        text_layers:  list[Image.Image]          = []
        debug_layers: list[Optional[Image.Image]] = []
        rendered = 0
        cached_n = 0

        for bd in sorted(page.bubbles, key=lambda b: b.reading_order):
            effective = bd.effective_text
            if not effective.strip():
                continue

            # Cache ellenorzes
            cached_layer = layer_cache.get(bd)
            if cached_layer is not None and not self.pcfg.force:
                text_layers.append(cached_layer)
                debug_layers.append(None)
                cached_n += 1
                continue

            # Dirty buborek: ujra renderelés
            bubble_dict = bd.to_dict()
            bubble_dict["translated_text"] = effective
            bubble_dict["raw_text"]        = bd.effective_ocr_text

            result = renderer.render_bubble(image, bubble_dict)
            if result is not None:
                text_layer, debug_layer = result
                text_layers.append(text_layer)
                debug_layers.append(debug_layer)
                layer_cache.save(bd, text_layer)
                rendered += 1
            else:
                logger.debug(
                    f"Render skip: buborek #{bd.bubble_id} "
                    f"(SFX / túl kicsi buborék / üres szöveg)")

        ex.set_provider("pillow")
        logger.info(
            f"  Render: {rendered} uj | {cached_n} cache | "
            f"{len(text_layers)} layer ossz")
        return text_layers, debug_layers

    def _run_composite(
        self,
        page:         PageData,
        original:     np.ndarray,
        inpainted:    np.ndarray,
        text_layers:  list[Image.Image],
        debug_layers: list[Optional[Image.Image]],
        ex:           StageExecutor,
    ) -> np.ndarray:
        if not text_layers:
            ex.warn("Nincsenek text layerek – inpainted visszaadva")
            return inpainted

        compositor   = self._compositor()
        bubble_dicts = [b.to_dict() for b in page.bubbles]
        result_obj   = compositor.compose_page(
            original_bgr=original,
            inpainted_bgr=inpainted,
            bubbles=bubble_dicts,
            text_layers=text_layers,
            debug_layers=debug_layers,
        )
        ex.set_provider("compositor")
        # CompositorResult.final_image_bgr kicsomagolása – numpy BGR array
        if hasattr(result_obj, "final_image_bgr"):
            return result_obj.final_image_bgr
        return result_obj  # fallback: már numpy

    def _run_export(
        self,
        page:      PageData,
        final:     np.ndarray,
        page_dir:  Path,
        img_path:  Path,
        ex:        StageExecutor,
    ) -> None:
        page_dir.mkdir(parents=True, exist_ok=True)
        out_path = page_dir / f"{img_path.stem}_hu.png"

        # Biztonsági konverzió ha valahonnan PIL Image érkezne
        from PIL import Image as _PILImage
        if isinstance(final, _PILImage.Image):
            import cv2 as _cv2
            final_np = _cv2.cvtColor(np.array(final.convert("RGB")), _cv2.COLOR_RGB2BGR)
        else:
            final_np = final  # numpy BGR

        cv2.imwrite(str(out_path), final_np,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        page.output_path = str(out_path)
        logger.info(f"  Export: {out_path.name}")

    # ── Debug artifacts ───────────────────────────────────────────────────────

    def _save_debug(
        self,
        page:      PageData,
        original:  np.ndarray,
        inpainted: np.ndarray,
        final:     np.ndarray,
        page_dir:  Path,
    ) -> None:
        debug_dir = page_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Layout debug
            from layout import get_detector
            from layout import BubbleRegion
            detector = get_detector()
            regions  = [
                BubbleRegion(
                    id=b.bubble_id, bbox=b.bbox,
                    order=b.reading_order,
                    confidence=b.detection_conf,
                    panel_id=b.panel_id,
                    type=b.bubble_type,
                )
                for b in page.bubbles
            ]
            cv2.imwrite(
                str(debug_dir / "01_layout.png"),
                detector.draw_debug(original, regions))

            cv2.imwrite(str(debug_dir / "02_inpainted.png"), inpainted)

            from rendering import render_debug
            bubble_dicts = [b.to_dict() for b in page.bubbles]
            cv2.imwrite(
                str(debug_dir / "03_render_overlay.png"),
                render_debug(inpainted, bubble_dicts))

            # Biztonsági konverzió ha valahonnan PIL Image érkezne
            from PIL import Image as _PILImage
            if isinstance(final, _PILImage.Image):
                final_dbg = cv2.cvtColor(
                    np.array(final.convert("RGB")), cv2.COLOR_RGB2BGR)
            else:
                final_dbg = final
            cv2.imwrite(str(debug_dir / "04_final.png"), final_dbg)
            logger.debug(f"  Debug artifacts: {debug_dir}")

        except Exception as e:
            logger.warning(f"  Debug artifact hiba: {e}")

    # ── Logging ───────────────────────────────────────────────────────────────

    @staticmethod
    def _log_page_summary(page: PageData) -> None:
        timing_str = " | ".join(
            f"{k}: {v:.0f}ms"
            for k, v in page.timings.items()
        )
        logger.info(
            f"[{page.page_id}] KESZ "
            f"{page.translated_count}/{page.total_bubbles} buborek | "
            f"{timing_str}"
        )

    def _log_batch_summary(self) -> None:
        m = self._manifest
        elapsed = m.total_elapsed_sec
        avg = elapsed / max(1, m.total_pages)
        logger.info("=" * 54)
        logger.info("  BATCH KESZ")
        logger.info(f"  Osszes:   {m.total_pages}")
        logger.info(f"  Sikeres:  {m.success_count}")
        logger.info(f"  Hiba:     {m.failed_count}")
        logger.info(f"  Kihagyva: {m.skipped_count + m.partial_count}")
        logger.info(f"  Ido:      {elapsed:.1f}s ({avg:.1f}s/kep)")
        logger.info(f"  Output:   {self.pcfg.output_dir}")
        logger.info("=" * 54)

    # ── Pause ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _pause(done: int, total: int) -> None:
        print(f"--- Pause: {done}/{total} oldal kesz ---")
        print("Ellenorizd az outputot, nyomj ENTER-t a folytatashoz...")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt

    # ── Publikus API GUI-hoz ──────────────────────────────────────────────────

    def rerender_bubble(
        self,
        page_dir:  Path,
        bubble_id: int,
    ) -> bool:
        """
        Egyetlen buborek ujrarenderelese GUI-bol.

        Betolti a meglevo PageData-t, megjeloli a buborekat dirty-kent,
        lefuttatja csak a render + composite stage-eket.

        Returns:
            True ha sikeres.
        """
        mgr  = PageMetadataManager(page_dir)
        page = mgr.load()
        if page is None:
            logger.error(f"PageData nem talalhato: {page_dir}")
            return False

        bd = page.get_bubble(bubble_id)
        if bd is None:
            logger.error(f"Buborek nem talalhato: #{bubble_id}")
            return False

        # Bubble dirty jelolese
        bd.render_hash = ""   # invalidalja a cache-t
        page.mark_stage_dirty(Stage.RENDER)

        # Eredeti kep betoltese
        image = _load_image(Path(page.source_path))
        if image is None:
            return False

        inpainted = _load_stage_img(page_dir, "inpainted") or image
        layer_cache = BubbleLayerCache(page_dir)

        with StageExecutor(page, Stage.RENDER, mgr, True) as ex:
            text_layers, debug_layers = self._run_render(
                page, inpainted, layer_cache, ex)

        with StageExecutor(page, Stage.COMPOSITE, mgr, True) as ex:
            final = self._run_composite(
                page, image, inpainted,
                text_layers, debug_layers, ex)

        out_path = Path(page.output_path) if page.output_path else \
            page_dir / f"{Path(page.source_path).stem}_hu.png"
        # Biztonsági konverzió ha valahonnan PIL Image érkezne
        from PIL import Image as _PILImage
        if isinstance(final, _PILImage.Image):
            final_np = cv2.cvtColor(
                np.array(final.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            final_np = final
        cv2.imwrite(str(out_path), final_np,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        mgr.save(page)
        logger.info(
            f"Buborek #{bubble_id} ujrarenderelve: {out_path.name}")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# SEGÉDFÜGGVÉNYEK (modul-szintű)
# ══════════════════════════════════════════════════════════════════════════════

def _page_to_entry(
    page:     PageData,
    img_path: Path,
    status:   str,
) -> ManifestEntry:
    page.update_stats()
    return ManifestEntry(
        page_id          = page.page_id,
        source_path      = str(img_path),
        status           = status,
        total_bubbles    = page.total_bubbles,
        translated_count = page.translated_count,
        avg_ocr_conf     = page.avg_ocr_conf,
        avg_render_score = page.avg_render_score,
        providers_used   = dict(page.providers_used),
        error            = page.error,
    )


def _failed_entry(
    page_id:  str,
    img_path: Path,
    error:    str,
) -> ManifestEntry:
    return ManifestEntry(
        page_id=page_id,
        source_path=str(img_path),
        status="failed",
        error=error,
    )


def _save_stage_img(
    image:    np.ndarray,
    page_dir: Path,
    name:     str,
) -> None:
    cache_dir = page_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache_dir / f"{name}.png"), image)


def _load_stage_img(
    page_dir: Path,
    name:     str,
) -> Optional[np.ndarray]:
    path = page_dir / ".cache" / f"{name}.png"
    if path.exists():
        img = cv2.imread(str(path))
        if img is not None:
            logger.debug(f"Stage cache hit: {name}")
            return img
    return None
