"""
inpainting.py - LaMa ONNX GPU inpainting pipeline.

Architektúra:
  - LaMaInpainter:        ONNX Runtime GPU inference, dinamikus felbontás
  - MaskPipeline:         szöveg maszk generálás + preprocessing
  - BackgroundEstimator:  context-aware háttérszín becslés
  - InpaintingManager:    orchestrátor, fallback logika

Tervezési elvek:
  - Nincs aggressive downscaling – ghosting quality > speed
  - Float32 precision végig, uint8 csak a végső outputnál
  - Tiled inference support nagy képekhez (future-ready)
  - Pipeline failure resilience: egy buborék hibája nem töri le az oldalt
  - Feathered mask blending – nincs látható varrat
  - BGR↔RGB konverzió minimalizálva (csak ONNX határon)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np

from config import cfg
from utils import MaskOps, BubbleAnalyzer, ImageUtils

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ONNX Runtime provider helper
# ══════════════════════════════════════════════════════════════════════════════

def _get_ort_session(model_path: str) -> Any:
    """
    ONNX Runtime session létrehozása GPU prioritással.

    Provider sorrend:
      1. CUDAExecutionProvider  – GPU (ha elérhető)
      2. CPUExecutionProvider   – fallback

    Returns:
        ort.InferenceSession
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime-gpu nincs telepítve.\n"
            "Telepítés: pip install onnxruntime-gpu"
        )

    providers: list[Any] = []

    if cfg.device.device == "cuda" and cfg.inpainting.use_gpu:
        cuda_opts = {
            "device_id": cfg.device.cuda_device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit":         8 * 1024 ** 3,   # 8 GB limit
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }
        providers.append(("CUDAExecutionProvider", cuda_opts))

    providers.append("CPUExecutionProvider")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = \
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers)
        active = session.get_providers()
        logger.info(f"ONNX session: {Path(model_path).name} | providers={active}")
        return session
    except Exception as e:
        raise RuntimeError(f"ONNX session létrehozása sikertelen: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# HÁTTÉR BECSLÉS
# ══════════════════════════════════════════════════════════════════════════════

class BackgroundEstimator:
    """
    Context-aware háttérszín becslés speech bubble régiókhoz.

    Stratégia:
      1. A buborék belső szélének mintavételezése
      2. Domináns szín meghatározása (median, nem mean)
      3. Egyszerű háttér detektálás (LaMa kihagyható-e)
    """

    @staticmethod
    def estimate(
        image_f32: np.ndarray,
        bbox: list[int],
        mask_soft: np.ndarray = None,
        border_px: int = None,
    ) -> np.ndarray:
        """
        Domináns háttérszín becslése float32 képből.
        Ha elérhető a maszk, a buborék belső, de szövegmentes területéről
        vesz mintát (elkerülve a fekete szegélyeket).
        """
        if border_px is None:
            border_px = cfg.inpainting.bg_sample_border_px

        h, w = image_f32.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)

        region = image_f32[y1:y2, x1:x2]

        # 1. Precíz mintavétel a maszk alapján (belső terület)
        if mask_soft is not None:
            mask_region = mask_soft[y1:y2, x1:x2]
            # Csak azokat a pixeleket nézzük, amik nincsenek a maszkon (háttér)
            bg_pixels = region[mask_region < 0.1]
            if len(bg_pixels) > 20:
                return np.median(bg_pixels, axis=0).astype(np.float32)

        # 2. Fallback: szél mintavétel (ha nincs maszk)
        if x2 - x1 < 2 * border_px or y2 - y1 < 2 * border_px:
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)

        b = border_px
        strips = [
            image_f32[y1:y1+b,    x1:x2],
            image_f32[y2-b:y2,    x1:x2],
            image_f32[y1:y2,      x1:x1+b],
            image_f32[y1:y2,      x2-b:x2],
        ]
        valid = [s.reshape(-1, 3) for s in strips if s.size > 0]
        if not valid:
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)

        pixels = np.vstack(valid)
        # Median robusztusabb az outlier-ek ellen mint a mean
        return np.median(pixels, axis=0).astype(np.float32)

    @staticmethod
    def is_simple(
        image_f32: np.ndarray,
        bbox: list[int],
        threshold: float = None,
    ) -> bool:
        """
        Egyszerű háttér detektálás – LaMa kihagyható-e.

        Float32 precision – nincs uint8 clipping.
        """
        if threshold is None:
            threshold = cfg.inpainting.simple_fill_threshold

        h, w = image_f32.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return True

        region = image_f32[y1:y2, x1:x2]

        # Fehér / világos arány (float32: >0.90)
        white_ratio = float(np.mean(region > 0.90))
        if white_ratio >= threshold:
            return True

        # Alacsony szórás = egyszínű
        std = float(np.std(region))
        if std < 0.07:
            return True

        return False

    @staticmethod
    def prefill(
        image_f32: np.ndarray,
        mask_f32: np.ndarray,
        bbox: list[int],
        bg_color: np.ndarray,
    ) -> np.ndarray:
        """
        Maszk terület előzetes kitöltése a becsült háttérszínnel.

        Ez segít a LaMa-nak: kevesebb 'ghost' marad,
        mert a modell már homogén hátteret lát.

        Args:
            image_f32: [H, W, 3] float32 RGB, in-place NEM módosítva
            mask_f32:  [H, W] float32, 0..1 (1 = kitöltendő)
            bbox:      [x1, y1, x2, y2]
            bg_color:  [3] float32 RGB

        Returns:
            Új float32 kép a kitöltött területtel.
        """
        result = image_f32.copy()
        x1, y1, x2, y2 = bbox
        h, w = image_f32.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        alpha = mask_f32[y1:y2, x1:x2, np.newaxis]
        fill  = np.full(
            (y2-y1, x2-x1, 3), bg_color, dtype=np.float32)
        region = result[y1:y2, x1:x2]
        result[y1:y2, x1:x2] = region * (1.0 - alpha) + fill * alpha
        return result


# ══════════════════════════════════════════════════════════════════════════════
# MASZK PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class MaskPipeline:
    """
    Robusztus szövegmaszk generálás és preprocessing.

    Lépések:
      1. Szöveg detektálás adaptív threshold-dal
      2. Kétlépéses dilation (szöveg + anti-aliased szélek)
      3. Feathering (varrat elleni simítás)
      4. Float32 normalizálás

    Minden eredmény float32, 0..1 – nincs uint8 clipping ciklus.
    """

    def __init__(self) -> None:
        self.dilation_px  = cfg.inpainting.mask_dilation_px
        self.extra_dil    = cfg.inpainting.mask_dilation_extra
        self.feather_px   = cfg.inpainting.mask_feather_px

    def generate(
        self,
        image_bgr: np.ndarray,
        bbox: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Szövegmaszk generálás egy buborék régiójához.

        Args:
            image_bgr: [H, W, 3] uint8 BGR
            bbox:      [x1, y1, x2, y2]

        Returns:
            (mask_hard, mask_soft):
              mask_hard: [H, W] uint8, 0/255 – inpainting bemenet
              mask_soft: [H, W] float32, 0..1 – feathered blending
        """
        h, w = image_bgr.shape[:2]

        # Szöveg detektálás a régión
        mask_hard = MaskOps.from_text_detection(
            image_bgr, bbox,
            dilation_px=self.dilation_px,
            extra_dilation=self.extra_dil,
        )

        # Ha a maszk üres → teljes bbox maszkot generálunk fallback-ként
        if mask_hard.sum() == 0:
            logger.debug(f"Üres szövegmaszk [{bbox}] – bbox fallback")
            mask_hard = MaskOps.from_bbox(
                (h, w), bbox, padding=2)

        # Feathered változat a seamless blendinghez
        mask_feathered = MaskOps.feather(mask_hard, self.feather_px)
        mask_soft      = mask_feathered.astype(np.float32) / 255.0

        if cfg.inpainting.debug_save_masks:
            self._save_debug(mask_hard, mask_soft, bbox)

        return mask_hard, mask_soft

    @staticmethod
    def _save_debug(
        mask_hard: np.ndarray,
        mask_soft: np.ndarray,
        bbox: list[int],
    ) -> None:
        try:
            debug_dir = cfg.paths.debug_dir
            tag = f"{bbox[0]}_{bbox[1]}"
            cv2.imwrite(
                str(debug_dir / f"mask_hard_{tag}.png"), mask_hard)
            cv2.imwrite(
                str(debug_dir / f"mask_soft_{tag}.png"),
                (mask_soft * 255).astype(np.uint8))
        except Exception as e:
            logger.debug(f"Debug maszk mentési hiba: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# LAMA ONNX INPAINTER
# ══════════════════════════════════════════════════════════════════════════════

class LamaInpainter:
    """
    LaMa ONNX Runtime GPU inpainter.

    Bemeneti formátum (lama-manga-dynamic.onnx):
      image: [1, 3, H, W] float32, 0..1, RGB
      mask:  [1, 1, H, W] float32, 0..1

    Dinamikus input méret – nincs kényszer resize.
    Teljes felbontáson fut (ghosting quality > speed).
    """

    # LaMa előszereti ha a méretek oszthatók ezzel
    _PAD_MULTIPLE = 8

    def __init__(self) -> None:
        self._session: Any   = None
        self._available: bool = False
        self._input_names: list[str] = []
        self._load()

    def _load(self) -> None:
        model_path = cfg.model_path(cfg.inpainting.lama_model_path)
        if not model_path.exists():
            logger.warning(
                f"LaMa modell nem található: {model_path}\n"
                "Elvárt elérési út: models/inpainting/lama-manga-dynamic.onnx\n"
                "OpenCV fallback aktív."
            )
            return
        try:
            self._session = _get_ort_session(str(model_path))
            self._input_names = [i.name for i in self._session.get_inputs()]
            self._available = True
            logger.info(f"LaMa ONNX betöltve ✓ | inputs={self._input_names}")
        except Exception as e:
            logger.warning(f"LaMa betöltési hiba: {e} – OpenCV fallback aktív")

    @property
    def available(self) -> bool:
        return self._available

    def inpaint(
        self,
        image_f32: np.ndarray,
        mask_f32: np.ndarray,
    ) -> np.ndarray:
        """
        LaMa inpainting futtatása.

        Args:
            image_f32: [H, W, 3] float32 RGB, 0..1
            mask_f32:  [H, W] float32, 0..1

        Returns:
            [H, W, 3] float32 RGB, 0..1
        """
        h, w = image_f32.shape[:2]

        # Padding hogy osztható legyen _PAD_MULTIPLE-lel
        ph = self._pad_size(h)
        pw = self._pad_size(w)

        img_padded  = self._pad(image_f32, ph, pw, mode="reflect")
        mask_padded = self._pad(
            mask_f32[:, :, np.newaxis], ph, pw, mode="constant")[:, :, 0]

        # [H, W, 3] → [1, 3, H, W]
        img_tensor  = img_padded.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        mask_tensor = mask_padded[np.newaxis, np.newaxis].astype(np.float32)

        # ONNX input dict – model-specifikus nevek
        feed: dict[str, np.ndarray] = {}
        for name in self._input_names:
            nl = name.lower()
            if "mask" in nl:
                feed[name] = mask_tensor
            else:
                feed[name] = img_tensor

        # Inference
        outputs = self._session.run(None, feed)
        result  = outputs[0]  # [1, 3, H, W]

        # [1, 3, H, W] → [H, W, 3]
        result = result[0].transpose(1, 2, 0)

        # Padding eltávolítása
        result = result[:h, :w]
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _pad_size(n: int, multiple: int = 8) -> int:
        return n if n % multiple == 0 else n + (multiple - n % multiple)

    @staticmethod
    def _pad(
        arr: np.ndarray,
        target_h: int,
        target_w: int,
        mode: str = "reflect",
    ) -> np.ndarray:
        h, w = arr.shape[:2]
        ph   = target_h - h
        pw   = target_w - w
        if arr.ndim == 3:
            pad_width = ((0, ph), (0, pw), (0, 0))
        else:
            pad_width = ((0, ph), (0, pw))
        return np.pad(arr, pad_width, mode=mode)


# ══════════════════════════════════════════════════════════════════════════════
# OPENCV FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

class OpenCVInpainter:
    """
    OpenCV TELEA alapú inpainting fallback.

    Minőségben elmarad a LaMa-tól, de mindig elérhető.
    Float32 precision-t megőrzi a kimenetben.
    """

    @staticmethod
    def inpaint(
        image_f32: np.ndarray,
        mask_hard: np.ndarray,
        bbox: list[int],
    ) -> np.ndarray:
        """
        Args:
            image_f32: [H, W, 3] float32 RGB, 0..1
            mask_hard: [H, W] uint8, 0/255
            bbox:      buborék bbox a háttérszín mintavételhez

        Returns:
            [H, W, 3] float32 RGB, 0..1
        """
        # uint8 szükséges az OpenCV inpaint-hoz
        img_u8     = (image_f32 * 255).clip(0, 255).astype(np.uint8)
        # RGB → BGR (OpenCV)
        img_bgr    = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

        result_bgr = cv2.inpaint(
            img_bgr, mask_hard,
            inpaintRadius=6,
            flags=cv2.INPAINT_TELEA,
        )

        # Háttérszín alapú simítás a buborék területén
        bg_rgb = BackgroundEstimator.estimate(image_f32, bbox)
        h, w   = image_f32.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0,x1); y1 = max(0,y1)
        x2 = min(w,x2); y2 = min(h,y2)

        local_mask = mask_hard[y1:y2, x1:x2].astype(np.float32) / 255.0
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result_f32 = result_rgb.astype(np.float32) / 255.0

        if local_mask.sum() > 0:
            alpha = local_mask[:, :, np.newaxis]
            fill  = np.full(
                (y2-y1, x2-x1, 3), bg_rgb, dtype=np.float32)
            blend = result_f32[y1:y2, x1:x2] * (1-alpha) + fill * alpha
            # Enyhe Gaussian a varrat elkerüléséhez
            smooth = cv2.GaussianBlur(blend, (5,5), 1.2)
            result_f32[y1:y2, x1:x2] = smooth

        return np.clip(result_f32, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# INPAINTING MANAGER – orchestrátor
# ══════════════════════════════════════════════════════════════════════════════

class InpaintingManager:
    """
    Fő orchestrátor: LaMa ONNX → OpenCV fallback.

    Workflow egy buborékra:
      1. Maszk generálás + dilation + feathering
      2. Háttér becslés
      3. Ha egyszerű háttér → direct fill (LaMa kihagyva)
      4. Ha LaMa elérhető → prefill + LaMa inference
      5. Fallback → OpenCV TELEA
      6. Feathered alpha blend a végső képbe

    Minden lépés float32 precision-ban fut.
    Egy buborék hibája NEM törli le az egész oldalt.
    """

    def __init__(self) -> None:
        self._lama    = LamaInpainter()
        self._opencv  = OpenCVInpainter()
        self._masks   = MaskPipeline()
        self._bg      = BackgroundEstimator()

        if self._lama.available:
            logger.info("Inpainting: LaMa ONNX GPU ✓")
        else:
            logger.info("Inpainting: OpenCV fallback (LaMa nem elérhető)")

    def inpaint_bubble(
        self,
        image_bgr: np.ndarray,
        bbox: list[int],
    ) -> np.ndarray:
        """
        Egyetlen buborék inpaintálása.

        Args:
            image_bgr: [H, W, 3] uint8 BGR – az EREDETI oldal
            bbox:      [x1, y1, x2, y2]

        Returns:
            [H, W, 3] uint8 BGR – módosított kép (másolat)
        """
        # 1. BGR → float32 RGB (egyetlen konverzió az egész pipeline-ra)
        image_rgb_f32 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)\
                          .astype(np.float32) / 255.0

        result_f32 = self._process_bubble(image_rgb_f32, bbox)

        # Float32 RGB → uint8 BGR (egyetlen visszakonverzió)
        result_u8  = (result_f32 * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(result_u8, cv2.COLOR_RGB2BGR)

    def _process_bubble(
        self,
        image_f32: np.ndarray,
        bbox: list[int],
    ) -> np.ndarray:
        """
        Belső feldolgozás – végig float32 RGB.

        Returns:
            [H, W, 3] float32 RGB, 0..1
        """
        # 2. Maszk generálás
        # A mask generátor BGR-t vár → temp konverzió csak erre
        image_bgr_u8 = (image_f32 * 255).clip(0,255).astype(np.uint8)
        image_bgr_u8 = cv2.cvtColor(image_bgr_u8, cv2.COLOR_RGB2BGR)
        mask_hard, mask_soft = self._masks.generate(image_bgr_u8, bbox)

        if mask_hard.sum() == 0:
            logger.debug(f"Üres maszk [{bbox}] – kihagyva")
            return image_f32

        # 3. Háttér becslés
        bg_color  = self._bg.estimate(image_f32, bbox, mask_soft)
        is_simple = self._bg.is_simple(image_f32, bbox)

        # 4. Egyszerű háttér → direct fill, LaMa kihagyva
        if is_simple:
            logger.debug(f"Egyszerű háttér [{bbox}] – direct fill")
            result_f32 = self._direct_fill(
                image_f32, mask_soft, bbox, bg_color)
            return result_f32

        # 5. LaMa inpainting
        if self._lama.available:
            try:
                # Prefill: LaMa-nak homogén hátteret adunk
                prefilled = self._bg.prefill(
                    image_f32, mask_soft, bbox, bg_color)
                lama_result = self._lama.inpaint(prefilled, mask_soft)
                # Seamless blend: csak a maszk területén alkalmazzuk
                result_f32 = MaskOps.alpha_blend_f32(
                    image_f32, lama_result, mask_soft)
                logger.debug(f"LaMa inpaint kész [{bbox}]")
                return result_f32
            except Exception as e:
                logger.warning(
                    f"LaMa hiba [{bbox}]: {e} – OpenCV fallback")

        # 6. OpenCV fallback
        result_f32 = self._opencv.inpaint(image_f32, mask_hard, bbox)
        # Feathered blend
        result_f32 = MaskOps.alpha_blend_f32(
            image_f32, result_f32, mask_soft)
        return result_f32

    @staticmethod
    def _direct_fill(
        image_f32: np.ndarray,
        mask_soft: np.ndarray,
        bbox: list[int],
        bg_color: np.ndarray,
    ) -> np.ndarray:
        """
        Egyszerű háttér esetén: közvetlen szín kitöltés + Gaussian simítás.

        Gyors és tiszta, LaMa overhead nélkül.
        """
        h, w = image_f32.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0,x1); y1 = max(0,y1)
        x2 = min(w,x2); y2 = min(h,y2)

        result = image_f32.copy()
        alpha  = mask_soft[y1:y2, x1:x2, np.newaxis]
        fill   = np.full((y2-y1, x2-x1, 3), bg_color, dtype=np.float32)
        region = result[y1:y2, x1:x2]
        blended = region * (1.0 - alpha) + fill * alpha

        # Határon Gaussian simítás a varrat ellen
        border = max(2, cfg.inpainting.mask_feather_px // 2)
        kernel = border * 2 + 1
        smooth = cv2.GaussianBlur(blended, (kernel, kernel), border / 2.0)

        # Csak a magas alpha területen alkalmazzuk a simítást
        high_alpha = np.clip(alpha * 2.0 - 0.5, 0.0, 1.0)
        result[y1:y2, x1:x2] = blended * (1.0-high_alpha) + smooth * high_alpha
        return result

    def inpaint_page(
        self,
        image_bgr: np.ndarray,
        bubbles: list[dict],
    ) -> np.ndarray:
        """
        Egy teljes oldal inpaintálása.

        Minden buborék hibája lokálisan kezelve – az oldal feldolgozás
        folytatódik hiba esetén is.

        Args:
            image_bgr: [H, W, 3] uint8 BGR
            bubbles:   bubble dict lista (translated_text szükséges)

        Returns:
            [H, W, 3] uint8 BGR inpaintált kép.
        """
        # Egyetlen BGR→RGB konverzió az egész oldalra
        image_f32 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)\
                      .astype(np.float32) / 255.0

        to_process = [
            b for b in bubbles
            if b.get("translated_text", "").strip()
        ]
        logger.info(f"Inpainting: {len(to_process)} buborék")

        success = 0
        for bubble in to_process:
            bbox = bubble["bbox"]
            try:
                image_f32 = self._process_bubble(image_f32, bbox)
                success += 1
            except Exception as e:
                # Buborék hiba nem töri le az oldalt
                logger.warning(
                    f"Inpainting hiba, buborék kihagyva "
                    f"[{bbox}]: {e}"
                )

        logger.info(f"Inpainting kész: {success}/{len(to_process)}")

        # Egyetlen RGB→BGR visszakonverzió
        result_u8 = (image_f32 * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(result_u8, cv2.COLOR_RGB2BGR)


# ── MaskOps kiegészítés: float32 alpha blend ──────────────────────────────────
# (utils.py-ban lévő MaskOps.alpha_blend uint8-ra van, itt float32 változat)

def _alpha_blend_f32(
    base: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Float32 alpha blend – gamma artifact és premultiplied alpha hiba nélkül.

    Args:
        base:    [H, W, 3] float32, 0..1
        overlay: [H, W, 3] float32, 0..1
        mask:    [H, W] float32, 0..1

    Returns:
        [H, W, 3] float32, 0..1
    """
    alpha = mask[:, :, np.newaxis]
    # Lineáris kompoziting (nem gamma-compressed tér!)
    result = base * (1.0 - alpha) + overlay * alpha
    return np.clip(result, 0.0, 1.0, out=result)


# Monkey-patch a MaskOps-ra (import nélkül elérhető az egész fájlban)
MaskOps.alpha_blend_f32 = staticmethod(_alpha_blend_f32)


# ── Singleton ──────────────────────────────────────────────────────────────────
_manager: Optional[InpaintingManager] = None


def get_inpainter() -> InpaintingManager:
    global _manager
    if _manager is None:
        _manager = InpaintingManager()
    return _manager


def inpaint_page(
    image_bgr: np.ndarray,
    bubbles: list[dict],
) -> np.ndarray:
    """Moduláris API: teljes oldal inpainting."""
    return get_inpainter().inpaint_page(image_bgr, bubbles)
