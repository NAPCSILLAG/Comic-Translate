"""
compositor.py - Professzionális réteg compositor.

Architektúra (réteg sorrend):
  Layer 0: Eredeti kép (source)
  Layer 1: Inpaintált patch-ek (LaMa / OpenCV)
  Layer 2: RGBA text layer-ek (Rasterizer output)
  Layer 3: Debug overlay (opcionális)

Tervezési elvek:
  - Gamma-helyes compositing: sRGB → lineáris → sRGB
  - Seamless feathering: 3-5px soft alpha átmenet
  - Smart patch preservation: változatlan patch → eredeti megtartva
  - Provider-based inpainting: LaMa / direct_fill / telea / future SD
  - Texture matching hook: future screentone / grain preservation
  - Pipeline failure resilience: rész-hiba nem töri le az oldalt
  - Nincs BGR↔RGB felesleges konverzió – float32 lineáris térben dolgozik
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Any, Protocol

import cv2
import numpy as np
from PIL import Image

from config import cfg
from utils import MaskOps, ImageUtils

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# GAMMA KONVERZIÓ – sRGB ↔ lineáris
# ══════════════════════════════════════════════════════════════════════════════

class GammaConv:
    """
    sRGB ↔ lineáris szín-tér konverzió.

    KRITIKUS az anti-aliased szöveg élekhez és outline blending-hez.
    Kerüli a dark halo és premultiplied alpha hibákat.

    Minden compositing lineáris térben zajlik,
    csak a végső output konvertálódik vissza sRGB-be.
    """

    # Lookup table a gyors sRGB → lineáris konverzióhoz (uint8 → float32)
    _TO_LINEAR: Optional[np.ndarray] = None
    # Lookup table lineáris → sRGB (float 0..1 → uint8)
    _TO_SRGB:   Optional[np.ndarray] = None

    @classmethod
    def _build_luts(cls) -> None:
        if cls._TO_LINEAR is not None:
            return
        # sRGB → lineáris: IEC 61966-2-1 standard
        lut = np.arange(256, dtype=np.float32) / 255.0
        linear = np.where(
            lut <= 0.04045,
            lut / 12.92,
            ((lut + 0.055) / 1.055) ** 2.4,
        )
        cls._TO_LINEAR = linear.astype(np.float32)

        # Lineáris → sRGB (float 0..1 → float 0..1)
        # Nem LUT hanem formula (float input esetén)

    @classmethod
    def to_linear(cls, image_f32: np.ndarray) -> np.ndarray:
        """
        sRGB float32 [0..1] → lineáris float32 [0..1].

        Args:
            image_f32: [..., 3] float32, sRGB

        Returns:
            [..., 3] float32, lineáris
        """
        return np.where(
            image_f32 <= 0.04045,
            image_f32 / 12.92,
            ((image_f32 + 0.055) / 1.055) ** 2.4,
        ).astype(np.float32)

    @classmethod
    def to_srgb(cls, image_linear: np.ndarray) -> np.ndarray:
        """
        Lineáris float32 [0..1] → sRGB float32 [0..1].

        Args:
            image_linear: [..., 3] float32, lineáris

        Returns:
            [..., 3] float32, sRGB
        """
        img = np.clip(image_linear, 0.0, 1.0)
        return np.where(
            img <= 0.0031308,
            img * 12.92,
            1.055 * (img ** (1.0 / 2.4)) - 0.055,
        ).astype(np.float32)

    @classmethod
    def composite_linear(
        cls,
        base_srgb: np.ndarray,
        overlay_srgb: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Gamma-helyes alpha compositing.

        Workflow:
          1. sRGB → lineáris
          2. Alpha blend lineáris térben
          3. Lineáris → sRGB

        Args:
            base_srgb:    [H, W, 3] float32, sRGB 0..1
            overlay_srgb: [H, W, 3] float32, sRGB 0..1
            alpha:        [H, W] float32, 0..1

        Returns:
            [H, W, 3] float32, sRGB 0..1
        """
        base_lin    = cls.to_linear(base_srgb)
        overlay_lin = cls.to_linear(overlay_srgb)

        a = alpha[:, :, np.newaxis]
        blended_lin = base_lin * (1.0 - a) + overlay_lin * a

        return cls.to_srgb(blended_lin)


# ══════════════════════════════════════════════════════════════════════════════
# TEXTURE MATCHING HOOK – jövőbeli screentone / grain
# ══════════════════════════════════════════════════════════════════════════════

class TextureMode(Enum):
    NONE       = auto()   # nincs textúra (default)
    GRAIN      = auto()   # film grain injektálás
    SCREENTONE = auto()   # manga screentone megőrzés (future)
    PAPER      = auto()   # papír textúra (future)
    SCAN_NOISE = auto()   # scan zaj (future)


@dataclass
class TextureConfig:
    """
    Textúra matching konfiguráció.

    Jelenleg csak grain injection van implementálva.
    A többi mód architektúrálisan előkészített (future).
    """
    mode:         TextureMode = TextureMode.NONE
    grain_amount: float       = 0.02   # 0..1, grain intenzitás
    grain_seed:   int         = 42     # reprodukálható eredményhez


class TextureMatcher:
    """
    Opcionális textúra matching / grain injection.

    Provider-based: új textúra mód = új metódus, nincs pipeline változás.
    Future: SD-alapú screentone reconstruction pluggolható ide.
    """

    def __init__(self, config: TextureConfig) -> None:
        self._cfg = config

    def apply(
        self,
        image_f32: np.ndarray,
        mask_f32: np.ndarray,
    ) -> np.ndarray:
        """
        Textúra alkalmazása az inpaintált területre.

        Args:
            image_f32: [H, W, 3] float32 lineáris RGB
            mask_f32:  [H, W] float32, 0..1 – érintett terület

        Returns:
            [H, W, 3] float32 lineáris RGB
        """
        if self._cfg.mode == TextureMode.NONE:
            return image_f32

        if self._cfg.mode == TextureMode.GRAIN:
            return self._apply_grain(image_f32, mask_f32)

        # Future módok: screentone, paper, scan_noise
        logger.debug(f"Textúra mód {self._cfg.mode} még nem implementált")
        return image_f32

    def _apply_grain(
        self,
        image_f32: np.ndarray,
        mask_f32: np.ndarray,
    ) -> np.ndarray:
        """Enyhe film grain az inpaintált területen."""
        rng   = np.random.default_rng(self._cfg.grain_seed)
        grain = rng.normal(
            0, self._cfg.grain_amount, image_f32.shape
        ).astype(np.float32)

        alpha = mask_f32[:, :, np.newaxis]
        result = image_f32 + grain * alpha
        return np.clip(result, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# INPAINTING PROVIDER PROTOCOL – provider-based architektúra
# ══════════════════════════════════════════════════════════════════════════════

class InpaintingProvider(Protocol):
    """
    Inpainting provider interface.

    Minden provider implementálja ezt:
      - LaMaProvider (current)
      - DirectFillProvider (current)
      - TeleaProvider (current fallback)
      - StableDiffusionProvider (future)
      - SDXLProvider (future)
      - FluxFillProvider (future)

    A Compositor csak ezt a protokollt látja – a konkrét
    implementáció cserélhető pipeline változtatás nélkül.
    """

    def inpaint(
        self,
        image_f32: np.ndarray,
        mask_f32: np.ndarray,
        bbox: list[int],
    ) -> np.ndarray:
        """
        Args:
            image_f32: [H, W, 3] float32 RGB, 0..1
            mask_f32:  [H, W] float32, 0..1
            bbox:      [x1, y1, x2, y2]

        Returns:
            [H, W, 3] float32 RGB, 0..1
        """
        ...

    @property
    def name(self) -> str:
        """Provider neve (logoláshoz)."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# PATCH RESULT – egyetlen inpaintált patch leírója
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PatchResult:
    """
    Egyetlen inpaintált patch eredménye és metaadatai.

    A smart preservation döntés itt dokumentálódik.
    """
    bbox:           list[int]
    was_modified:   bool         # False = eredeti megőrizve
    provider_used:  str          # "lama" | "direct_fill" | "telea" | "preserved"
    patch_f32:      np.ndarray   # [ph, pw, 3] float32 RGB – a javított patch
    original_f32:   np.ndarray   # [ph, pw, 3] float32 RGB – eredeti patch
    similarity:     float        # 0..1, 1.0 = teljesen azonos


# ══════════════════════════════════════════════════════════════════════════════
# FEATHERING HELPER
# ══════════════════════════════════════════════════════════════════════════════

class FeatherBlender:
    """
    Seamless feathered blending patch határokhoz.

    Soft alpha átmenet a patch széleinél – nincs látható él.
    Konfiguálható radius (default: 4px).
    """

    def __init__(self, radius: int = 4) -> None:
        self.radius = max(1, radius)

    def blend_patch(
        self,
        base_f32: np.ndarray,
        patch_f32: np.ndarray,
        mask_hard: np.ndarray,
        bbox: list[int],
    ) -> np.ndarray:
        """
        Feathered alpha blend – patch a base-be.

        Args:
            base_f32:   [H, W, 3] float32 RGB – teljes kép
            patch_f32:  [H, W, 3] float32 RGB – inpaintált kép
            mask_hard:  [H, W] uint8 0/255 – kemény maszk
            bbox:       [x1, y1, x2, y2]

        Returns:
            [H, W, 3] float32 RGB – blendelt kép
        """
        # Feathered (puha) maszk
        feathered = MaskOps.feather(mask_hard, self.radius)
        alpha_f32 = feathered.astype(np.float32) / 255.0

        # Gamma-helyes compositing
        result = GammaConv.composite_linear(
            base_srgb=base_f32,
            overlay_srgb=patch_f32,
            alpha=alpha_f32,
        )
        return result

    def blend_text_layer(
        self,
        base_f32: np.ndarray,
        text_rgba_pil: Image.Image,
    ) -> np.ndarray:
        """
        RGBA text layer gamma-helyes compositing a base képre.

        A Rasterizer RGBA outputját kompozitálja.
        Nincs dark halo, nincs premultiplied alpha hiba.

        Args:
            base_f32:      [H, W, 3] float32 RGB sRGB 0..1
            text_rgba_pil: PIL RGBA Image – teljes oldal méretű

        Returns:
            [H, W, 3] float32 RGB sRGB 0..1
        """
        rgba_arr = np.array(text_rgba_pil).astype(np.float32) / 255.0
        # [H, W, 4] → [H, W, 3] RGB + [H, W] alpha
        text_rgb = rgba_arr[:, :, :3]
        alpha    = rgba_arr[:, :, 3]

        # Szöveg alpha feathering (anti-aliased él megőrzés)
        # A PIL RGBA már tartalmaz soft edge-t a supersampling miatt,
        # de enyhe extra feathering a compositing határán segíthet
        if self.radius > 1:
            alpha_smooth = cv2.GaussianBlur(
                alpha,
                (self.radius * 2 - 1, self.radius * 2 - 1),
                self.radius * 0.3,
            )
        else:
            alpha_smooth = alpha

        return GammaConv.composite_linear(
            base_srgb=base_f32,
            overlay_srgb=text_rgb,
            alpha=alpha_smooth,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SMART PATCH ANALYZER – változatlan patch detektálás
# ══════════════════════════════════════════════════════════════════════════════

class SmartPatchAnalyzer:
    """
    Meghatározza hogy egy inpaintált patch ténylegesen változott-e.

    Ha nem változott érdemben → eredeti pixel megőrzése.
    Kerüli a szükségtelen minőségromlást.
    """

    # Hasonlóság küszöb: e felett az eredeti megőrzendő
    SIMILARITY_THRESHOLD = 0.98

    @classmethod
    def similarity(
        cls,
        original: np.ndarray,
        inpainted: np.ndarray,
    ) -> float:
        """
        Struktúrális hasonlóság becslése (SSIM közelítés).

        Gyors: MSE alapú, nem teljes SSIM számítás.

        Returns:
            0.0..1.0, ahol 1.0 = teljesen azonos
        """
        if original.shape != inpainted.shape:
            return 0.0
        diff    = (original.astype(np.float32) - inpainted.astype(np.float32))
        mse     = float(np.mean(diff ** 2))
        # MSE → similarity: 0 MSE = 1.0, nagy MSE → 0.0
        sim     = 1.0 / (1.0 + mse * 100.0)
        return float(np.clip(sim, 0.0, 1.0))

    @classmethod
    def should_preserve(cls, similarity: float) -> bool:
        return similarity >= cls.SIMILARITY_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITOR – fő orchestrátor
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompositorResult:
    """A compositor teljes futás eredménye egy oldalra."""
    final_image_bgr:   np.ndarray         # [H, W, 3] uint8 BGR – végleges
    patches_applied:   int
    patches_preserved: int
    text_layers_comp:  int
    has_debug_overlay: bool


class Compositor:
    """
    Fő compositor orchestrátor – minden réteg összerakása.

    Réteg sorrend:
      0. Eredeti kép (float32 sRGB)
      1. Inpaintált patch-ek (feathered blend)
      2. Opcionális textúra matching
      3. RGBA text layer-ek (gamma-helyes composite)
      4. Debug overlay (ha debug mód)

    Minden lépés float32 lineáris/sRGB terében dolgozik.
    Egyetlen uint8 konverzió csak a kimenetkor.
    """

    def __init__(
        self,
        feather_radius: int = 4,
        texture_cfg:    Optional[TextureConfig] = None,
    ) -> None:
        self._feather  = FeatherBlender(feather_radius)
        self._texture  = TextureMatcher(
            texture_cfg or TextureConfig(mode=TextureMode.NONE))
        self._analyzer = SmartPatchAnalyzer()
        logger.info(
            f"Compositor inicializálva | "
            f"feather={feather_radius}px | "
            f"texture={texture_cfg.mode if texture_cfg else 'NONE'}"
        )

    def compose_page(
        self,
        original_bgr:   np.ndarray,
        inpainted_bgr:  np.ndarray,
        bubbles:        list[dict],
        text_layers:    list[Image.Image],
        debug_layers:   list[Optional[Image.Image]],
    ) -> CompositorResult:
        """
        Teljes oldal compositor pipeline.

        Args:
            original_bgr:  [H, W, 3] uint8 BGR – eredeti kép
            inpainted_bgr: [H, W, 3] uint8 BGR – inpaintált kép
            bubbles:       bubble dict lista
            text_layers:   RGBA PIL Image lista (Rasterizer output)
            debug_layers:  debug RGBA lista (None elemekkel)

        Returns:
            CompositorResult a végleges képpel és statisztikákkal.
        """
        h, w = original_bgr.shape[:2]

        # ── 0. Konverzió float32 sRGB-be (egyetlen BGR→RGB konverzió) ────────
        orig_f32 = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)\
                     .astype(np.float32) / 255.0
        inp_f32  = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)\
                     .astype(np.float32) / 255.0

        result_f32 = orig_f32.copy()
        patches_applied   = 0
        patches_preserved = 0

        # ── 1. Inpaintált patch-ek blending ───────────────────────────────────
        translated_bubbles = [
            b for b in bubbles
            if b.get("translated_text", "").strip()
        ]

        for bubble in translated_bubbles:
            bbox = bubble.get("bbox")
            if bbox is None:
                continue
            try:
                patch_result = self._compose_patch(
                    orig_f32, inp_f32, bbox)

                if patch_result.was_modified:
                    result_f32 = patch_result.patch_f32
                    patches_applied += 1
                else:
                    patches_preserved += 1

            except Exception as e:
                logger.warning(
                    f"Patch compose hiba [{bbox}]: {e} – eredeti megtartva")
                patches_preserved += 1

        logger.debug(
            f"Patch compose: {patches_applied} alkalmazva, "
            f"{patches_preserved} megőrizve"
        )

        # ── 2. Opcionális textúra matching ────────────────────────────────────
        # Ha van textúra konfig, az érintett területeken alkalmazzuk
        if self._texture._cfg.mode != TextureMode.NONE:
            combined_mask = self._build_combined_mask(
                (h, w), translated_bubbles)
            mask_f32 = combined_mask.astype(np.float32) / 255.0
            # Lineáris térbe, textúra, vissza sRGB
            lin      = GammaConv.to_linear(result_f32)
            lin      = self._texture.apply(lin, mask_f32)
            result_f32 = GammaConv.to_srgb(lin)

        # ── 3. RGBA text layer-ek compositing ────────────────────────────────
        text_layers_comp = 0
        for text_layer in text_layers:
            try:
                result_f32 = self._feather.blend_text_layer(
                    result_f32, text_layer)
                text_layers_comp += 1
            except Exception as e:
                logger.warning(f"Text layer composite hiba: {e}")

        logger.info(f"Text layers composited: {text_layers_comp}")

        # ── 4. Debug overlay ──────────────────────────────────────────────────
        has_debug = False
        if cfg.pipeline.debug:
            for debug_layer in debug_layers:
                if debug_layer is not None:
                    try:
                        result_f32 = self._feather.blend_text_layer(
                            result_f32, debug_layer)
                        has_debug = True
                    except Exception as e:
                        logger.debug(f"Debug layer hiba: {e}")

        # ── 5. Float32 sRGB → uint8 BGR (egyetlen visszakonverzió) ───────────
        result_u8 = (np.clip(result_f32, 0.0, 1.0) * 255)\
                      .astype(np.uint8)
        result_bgr = cv2.cvtColor(result_u8, cv2.COLOR_RGB2BGR)

        return CompositorResult(
            final_image_bgr=result_bgr,
            patches_applied=patches_applied,
            patches_preserved=patches_preserved,
            text_layers_comp=text_layers_comp,
            has_debug_overlay=has_debug,
        )

    def _compose_patch(
        self,
        orig_f32: np.ndarray,
        inp_f32:  np.ndarray,
        bbox:     list[int],
    ) -> PatchResult:
        """
        Egyetlen patch feathered blending + smart preservation.

        Workflow:
          1. Patch kivágás
          2. Hasonlóság mérés
          3. Ha hasonló → preserved (nincs változás)
          4. Ha különböző → maszk generálás + feathered blend
        """
        h, w = orig_f32.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0,x1); y1 = max(0,y1)
        x2 = min(w,x2); y2 = min(h,y2)
        if x2 <= x1 or y2 <= y1:
            return PatchResult(
                bbox=bbox, was_modified=False,
                provider_used="skip", patch_f32=orig_f32,
                original_f32=orig_f32[y1:y2,x1:x2],
                similarity=1.0,
            )

        orig_patch = orig_f32[y1:y2, x1:x2]
        inp_patch  = inp_f32[y1:y2,  x1:x2]

        # Smart preservation: hasonlóság ellenőrzés
        sim = self._analyzer.similarity(orig_patch, inp_patch)
        if self._analyzer.should_preserve(sim):
            logger.debug(
                f"Patch preserved (sim={sim:.3f}) [{bbox}]")
            return PatchResult(
                bbox=bbox, was_modified=False,
                provider_used="preserved",
                patch_f32=orig_f32,
                original_f32=orig_patch,
                similarity=sim,
            )

        # Különbség maszk: ahol az inpainting változtatott
        diff       = np.abs(inp_f32 - orig_f32).mean(axis=2)
        diff_mask  = (diff > 0.015).astype(np.uint8) * 255

        # Feathered blend
        result_f32 = self._feather.blend_patch(
            orig_f32, inp_f32, diff_mask, bbox)

        return PatchResult(
            bbox=bbox, was_modified=True,
            provider_used="lama_or_opencv",
            patch_f32=result_f32,
            original_f32=orig_patch,
            similarity=sim,
        )

    @staticmethod
    def _build_combined_mask(
        image_shape: tuple[int, int],
        bubbles: list[dict],
    ) -> np.ndarray:
        """Összes bubble bbox uniója egyetlen maszkban."""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for b in bubbles:
            bbox = b.get("bbox")
            if bbox:
                x1,y1,x2,y2 = bbox
                x1=max(0,x1); y1=max(0,y1)
                x2=min(w,x2); y2=min(h,y2)
                mask[y1:y2, x1:x2] = 255
        return mask

    def save_debug(
        self,
        result: CompositorResult,
        output_path: Path,
        bubbles: list[dict],
    ) -> None:
        """Debug kép mentése részletes annotációval."""
        try:
            vis = result.final_image_bgr.copy()
            for b in bubbles:
                bbox = b.get("bbox")
                if not bbox:
                    continue
                x1,y1,x2,y2 = bbox
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,200,0), 1)
                trans = b.get("translated_text","")[:20]
                if trans:
                    cv2.putText(
                        vis, trans, (x1+2, y1+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0,100,255), 1, cv2.LINE_AA)
            cv2.imwrite(str(output_path), vis)
            logger.debug(f"Debug kép mentve: {output_path}")
        except Exception as e:
            logger.debug(f"Debug mentési hiba: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Singleton + moduláris API
# ══════════════════════════════════════════════════════════════════════════════

_compositor: Optional[Compositor] = None


def get_compositor() -> Compositor:
    global _compositor
    if _compositor is None:
        _compositor = Compositor(feather_radius=4)
    return _compositor


def compose_page(
    original_bgr:  np.ndarray,
    inpainted_bgr: np.ndarray,
    bubbles:       list[dict],
    text_layers:   list[Image.Image],
    debug_layers:  list[Optional[Image.Image]],
) -> np.ndarray:
    """
    Moduláris API: teljes oldal kompozitálás.

    Returns:
        [H, W, 3] uint8 BGR – végleges kép.
    """
    result = get_compositor().compose_page(
        original_bgr, inpainted_bgr,
        bubbles, text_layers, debug_layers,
    )
    return result.final_image_bgr
