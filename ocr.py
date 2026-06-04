"""
ocr.py - Language-agnostic OCR pipeline, ppocr-v5-onnx alapon.

Architektúra:
  - OCRResult:         immutable text instance (polygon, bbox, text, conf, ...)
  - PPOCRv5Detector:   ONNX detektor – szöveg régió polygon-ok
  - PPOCRv5Recognizer: ONNX felismerő – szöveg + confidence per régió
  - PPOCRv5Pipeline:   detektor + felismerő orchestráció
  - EasyOCRFallback:   teljes fallback ha ppocr-v5 nem elérhető
  - ComicOCR:          publikus API, model-agnostic

Tervezési elvek:
  - Teljesen állapotmentes: nincs globális kép feltételezés
  - Layout döntés NINCS itt – csak strukturált adat kinyerés
  - Polygon koordináták az eredeti kép terében maradnak
  - Raw model output megőrzése, korai normalizálás nélkül
  - Teljes csere lehetséges: csak ComicOCR publikus API kell
  - Pipeline failure resilience: buborék hiba nem töri le az oldalt
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np

from config import cfg

logger = logging.getLogger(__name__)

# ── Debug mód ────────────────────────────────────────────────────────────────
# Bekapcsolás: DEBUG_COMIC=1 env var VAGY cfg.debug
_OCR_DEBUG: bool = os.environ.get("DEBUG_COMIC", "0") == "1"

def _ocr_debug_active() -> bool:
    """Futási időben is lekérdezhető debug állapot."""
    try:
        return _OCR_DEBUG or bool(getattr(cfg, "debug", False))
    except Exception:
        return _OCR_DEBUG

def _ocr_debug_dir(page_tag: str = "") -> Path:
    """Debug könyvtár – automatikusan létrehozza ha nem létezik."""
    tag = f"_{page_tag}" if page_tag else ""
    d = Path("output") / "debug" / f"ocr_crops{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _save_debug_crop(
    crop:    np.ndarray,
    label:   str,
    status:  str,           # "OK" | "INVALID" | "EMPTY" | "TINY" | "CRASH"
    debug_dir: Path,
    idx:     int = 0,
) -> None:
    """Crop mentése debug könyvtárba annotációval.

    Fájlnév: crop_{idx:02d}_{status}_{label}.png
    A képre rárajzolja a státuszt és a crop méretét.
    """
    try:
        if crop is None or crop.size == 0:
            # Üres placeholder 64×64-es vörös kép
            vis = np.zeros((64, 200, 3), dtype=np.uint8)
            vis[:, :] = (0, 0, 180)
        else:
            vis = crop.copy() if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            # RGB → BGR ha kell
            if vis.dtype != np.uint8:
                vis = np.clip(vis, 0, 255).astype(np.uint8)

        h, w = vis.shape[:2]
        color_map = {
            "OK":      (0, 200, 0),
            "INVALID": (0, 0, 255),
            "EMPTY":   (0, 165, 255),
            "TINY":    (0, 255, 255),
            "CRASH":   (128, 0, 128),
        }
        color = color_map.get(status, (200, 200, 200))

        # Keret rajzolása a státusz szerint
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), color, 3)

        # Szöveg: státusz + méret
        txt = f"{status} {w}x{h}"
        cv2.putText(vis, txt, (4, min(h - 4, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        safe_label = label.replace("/", "_").replace(" ", "_")[:30]
        fname = f"crop_{idx:02d}_{status}_{safe_label}.png"
        cv2.imwrite(str(debug_dir / fname), vis)
    except Exception as e:
        logger.debug(f"Debug crop mentési hiba: {e}")

def _save_layout_debug(
    image_bgr: np.ndarray,
    bubbles:   list[dict],
    debug_dir: Path,
) -> None:
    """Eredeti kép + YOLO bbox-ok overlay-je mentve layout_boxes.png-ként."""
    try:
        vis = image_bgr.copy()
        ih, iw = vis.shape[:2]
        for i, b in enumerate(bubbles):
            bbox = b.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox
            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(iw, x2); y2c = min(ih, y2)
            valid = x2c > x1c and y2c > y1c
            color = (0, 200, 0) if valid else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"#{i}", (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.imwrite(str(debug_dir / "layout_boxes.png"), vis)
    except Exception as e:
        logger.debug(f"Layout debug mentési hiba: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# OCRResult – immutable text instance
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OCRResult:
    """
    Egyetlen felismert szöveg példány teljes geometriával.

    Frozen dataclass: immutable, hashable, thread-safe.
    Koordináták MINDIG az eredeti kép terében vannak.

    Fields:
        text:            felismert szöveg (Unicode, raw model output)
        confidence:      per-instance score, 0.0..1.0
        polygon:         eredeti polygon koordináták [[x,y], ...] (N×2)
        bbox:            axis-aligned [x1, y1, x2, y2] (polygon-ból számított)
        rotation_angle:  becsült szöveg forgásszög fokokban (-180..180)
        language:        forrás nyelv kód ('en', 'ja', 'unknown', ...)
        line_group_id:   azonos sorhoz tartozó példányok azonos ID-t kapnak
        reading_order:   olvasási sorrend index (0 = első)
        line_height_est: becsült betűmagasság pixelben
        raw_det_score:   nyers detektor confidence (debug/fallback)
    """
    text:             str
    confidence:       float
    polygon:          tuple            # tuple of (x, y) pairs – hashable
    bbox:             tuple[int,int,int,int]
    rotation_angle:   float     = 0.0
    language:         str       = "unknown"
    line_group_id:    int       = -1
    reading_order:    int       = -1
    line_height_est:  float     = 0.0
    raw_det_score:    float     = 0.0

    @classmethod
    def from_polygon_and_text(
        cls,
        polygon_pts: np.ndarray,
        text: str,
        confidence: float,
        language: str = "unknown",
        reading_order: int = -1,
        raw_det_score: float = 0.0,
    ) -> "OCRResult":
        """
        Factory: numpy polygon + szöveg → OCRResult.

        Koordináták az EREDETI kép terében, nincs normalizálás.
        """
        pts = np.array(polygon_pts, dtype=np.float32)

        # Axis-aligned bbox a polygonból
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        x1, y1 = int(np.floor(x_coords.min())), int(np.floor(y_coords.min()))
        x2, y2 = int(np.ceil(x_coords.max())),  int(np.ceil(y_coords.max()))

        # Forgásszög becslés: az alsó él iránya (pt[1]→pt[0] vektor)
        rotation = 0.0
        if len(pts) >= 2:
            dx = float(pts[1, 0] - pts[0, 0])
            dy = float(pts[1, 1] - pts[0, 1])
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                rotation = math.degrees(math.atan2(dy, dx))

        # Sor magasság becslés
        line_h = float(y2 - y1)
        if len(pts) >= 4:
            left_h  = float(np.linalg.norm(pts[3] - pts[0]))
            right_h = float(np.linalg.norm(pts[2] - pts[1]))
            line_h  = (left_h + right_h) / 2.0

        # Polygon tuple-ként (hashable)
        poly_tuple = tuple((float(p[0]), float(p[1])) for p in pts)

        return cls(
            text=text,
            confidence=float(confidence),
            polygon=poly_tuple,
            bbox=(x1, y1, x2, y2),
            rotation_angle=float(rotation),
            language=language,
            line_group_id=-1,
            reading_order=reading_order,
            line_height_est=line_h,
            raw_det_score=float(raw_det_score),
        )

    @property
    def polygon_np(self) -> np.ndarray:
        """Polygon koordináták numpy tömbként [[x, y], ...]."""
        return np.array(self.polygon, dtype=np.float32)

    @property
    def is_valid(self) -> bool:
        """Elfogadható minőségű-e az eredmény."""
        return (
            bool(self.text.strip()) and
            self.confidence >= cfg.ocr.rec_thresh and
            len(self.polygon) >= 3
        )

    def to_dict(self) -> dict:
        """Szerializálható dict – JSON exporthoz."""
        return {
            "text":            self.text,
            "confidence":      round(self.confidence, 4),
            "polygon":         list(self.polygon),
            "bbox":            list(self.bbox),
            "rotation_angle":  round(self.rotation_angle, 2),
            "language":        self.language,
            "line_group_id":   self.line_group_id,
            "reading_order":   self.reading_order,
            "line_height_est": round(self.line_height_est, 1),
            "raw_det_score":   round(self.raw_det_score, 4),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ONNX session helper (ugyanaz mint inpainting.py-ban, de lokális)
# ══════════════════════════════════════════════════════════════════════════════

def _get_ort_session(model_path: str) -> Any:
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("pip install onnxruntime-gpu")

    providers: list[Any] = []
    if cfg.device.device == "cuda" and cfg.ocr.use_gpu:
        providers.append((
            "CUDAExecutionProvider",
            {"device_id": cfg.device.cuda_device_id},
        ))
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    active  = session.get_providers()
    logger.info(f"ONNX: {Path(model_path).name} | {active}")
    return session


# ══════════════════════════════════════════════════════════════════════════════
# PPOCR-v5 DETEKTOR
# ══════════════════════════════════════════════════════════════════════════════

class PPOCRv5Detector:
    """
    PP-OCRv5 szövégterület detektor.

    Input:  [1, 3, H, W] float32, ImageNet normalizált
    Output: probability map → polygon-ok (DB algorithm)

    Koordináták az EREDETI kép terében adódnak vissza.
    Nincs belső méretezés a kimeneti geometrián.
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _LIMIT_SIDE = 960   # max oldalhossz az inference-hez

    def __init__(self, model_path: str) -> None:
        self._session = _get_ort_session(model_path)
        self._input_name = self._session.get_inputs()[0].name

    def detect(
        self,
        image_rgb: np.ndarray,
        det_thresh: float = None,
    ) -> list[np.ndarray]:
        """
        Szöveg régiók detektálása.

        Args:
            image_rgb: [H, W, 3] uint8 RGB – EREDETI felbontás
            det_thresh: confidence küszöb

        Returns:
            list of [N, 2] float32 polygon numpy tömbök,
            koordináták az EREDETI kép terében.
        """
        if det_thresh is None:
            det_thresh = cfg.ocr.det_thresh

        orig_h, orig_w = image_rgb.shape[:2]

        # Input validáció ONNX előtt
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            logger.warning(f"Detektor: érvénytelen input shape {image_rgb.shape}")
            return []
        if orig_h < 4 or orig_w < 4:
            logger.warning(f"Detektor: túl kis kép {orig_w}x{orig_h}px – skip")
            return []

        # Átméretezés inference-hez (arány megőrzve)
        inp, scale_x, scale_y = self._preprocess(image_rgb)

        # Post-preprocess shape ellenőrzés
        if inp.shape[2] < 2 or inp.shape[3] < 2:
            logger.error(
                f"Detektor: _preprocess után érvénytelen tensor shape {inp.shape} "
                f"(input: {orig_w}x{orig_h}) – ONNX hívás megtagadva")
            return []

        # ONNX inference
        try:
            outputs = self._session.run(None, {self._input_name: inp})
        except Exception as e:
            logger.warning(f"Detektor inference hiba [{orig_w}x{orig_h}px]: {e}")
            return []

        prob_map = outputs[0][0, 0]   # [H_inf, W_inf]

        # Polygon-ok kinyerése a probability map-ből
        polygons = self._decode_polygons(
            prob_map, det_thresh, orig_w, orig_h, scale_x, scale_y)
        return polygons

    def _preprocess(
        self,
        image_rgb: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """
        Kép előkészítése: resize (arány megőrzve) + ImageNet normalizálás.

        Returns:
            (tensor [1,3,H,W], scale_x, scale_y)
            scale_x/y: eredeti → inference méret aránya (visszaskálázáshoz)
        """
        h, w = image_rgb.shape[:2]
        limit = self._LIMIT_SIDE

        # Arányos átméretezés
        _MIN_SAFE = 64  # 2 × stride(32) – Conv.33 minimum
        if min(h, w) < _MIN_SAFE:
            scale = _MIN_SAFE / min(h, w)   # upscale kis cropok
        else:
            scale = min(limit / max(h, w), 1.0)  # downscale nagy képek
        new_h = max(32, int(round((h * scale) / 32) * 32))
        new_w = max(32, int(round((w * scale) / 32) * 32))

        resized = cv2.resize(
            image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Post-resize dimenzió ellenőrzés – biztonsági háló
        rh, rw = resized.shape[:2]
        if rh < 32 or rw < 32:
            logger.error(
                f"_preprocess: resize eredmény túl kicsi {rw}x{rh}px "
                f"(orig: {w}x{h}, scale: {scale:.3f}) – Conv.33 crash várható!")
        if rh // 32 < 2 or rw // 32 < 2:
            logger.warning(
                f"_preprocess: feature map < 2×2 ({rw//32}×{rh//32}) "
                f"– Conv.33 instabilitás lehetséges")

        img_f32 = resized.astype(np.float32) / 255.0
        img_norm = (img_f32 - self._MEAN) / self._STD

        # [H, W, 3] → [1, 3, H, W]
        tensor = img_norm.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        # Skálatényező az eredeti koordináta-térbe visszaskálázáshoz
        scale_x = w / new_w
        scale_y = h / new_h
        return tensor, scale_x, scale_y

    @staticmethod
    def _decode_polygons(
        prob_map: np.ndarray,
        thresh: float,
        orig_w: int,
        orig_h: int,
        scale_x: float,
        scale_y: float,
        min_area: float = 10.0,
        unclip_ratio: float = 1.6,
    ) -> list[np.ndarray]:
        """
        DB (Differentiable Binarization) polygon dekódolás.

        Fontos: a visszaskálázás itt történik, így a visszaadott
        polygon-ok MINDIG az eredeti kép koordináta-terében vannak.
        """
        binary = (prob_map > thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        polygons: list[np.ndarray] = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            # Unclip: kissé kiterjesztjük a detektált területet
            poly = PPOCRv5Detector._unclip_polygon(contour, unclip_ratio)
            if poly is None or len(poly) < 4:
                continue

            # Visszaskálázás az eredeti koordináta-térbe
            poly_scaled = poly.astype(np.float32)
            poly_scaled[:, 0] *= scale_x
            poly_scaled[:, 1] *= scale_y

            # Klip az eredeti kép határaira
            poly_scaled[:, 0] = np.clip(poly_scaled[:, 0], 0, orig_w - 1)
            poly_scaled[:, 1] = np.clip(poly_scaled[:, 1], 0, orig_h - 1)

            polygons.append(poly_scaled)

        return polygons

    @staticmethod
    def _unclip_polygon(
        contour: np.ndarray,
        ratio: float,
    ) -> Optional[np.ndarray]:
        """Pyclipper-szerű polygon kiterjesztés OpenCV-vel."""
        try:
            rect = cv2.minAreaRect(contour)
            box  = cv2.boxPoints(rect)
            return box.astype(np.float32)
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# PPOCR-v5 FELISMERŐ
# ══════════════════════════════════════════════════════════════════════════════

class PPOCRv5Recognizer:
    """
    PP-OCRv5 szöveg felismerő (CRNN alapú).

    Input:  [1, 3, 48, W] float32 – perspective-corrected crop
    Output: CTC decoded text + per-char confidence

    A perspektíva korrekció itt történik, de az EREDETI koordináták
    megőrződnek a visszaadott OCRResult-ban.
    """

    _IMG_HEIGHT = 48
    _MEAN = 0.5
    _STD  = 0.5

    def __init__(self, model_path: str, dict_path: str) -> None:
        self._session    = _get_ort_session(model_path)
        self._input_name = self._session.get_inputs()[0].name
        self._charset    = self._load_charset(dict_path)
        logger.info(f"Recognizer charset: {len(self._charset)} karakter")

    @staticmethod
    def _load_charset(dict_path: str) -> list[str]:
        """Karakter szótár betöltése."""
        try:
            chars = Path(dict_path).read_text(encoding="utf-8").splitlines()
            # Blank token a CTC-hez
            return ["blank"] + [c for c in chars if c]
        except Exception as e:
            logger.error(f"Charset betöltési hiba: {e}")
            return ["blank"]

    def recognize(
        self,
        image_rgb: np.ndarray,
        polygon: np.ndarray,
        bubble_id: int = -1,
    ) -> tuple[str, float]:
        """
        Szöveg felismerés egy polygon régióból.

        Args:
            image_rgb: [H, W, 3] uint8 RGB – EREDETI kép
            polygon:   [N, 2] float32 – EREDETI koordinátákban

        Returns:
            (szöveg, confidence)
            Pontosan a modell raw outputja, minimális post-process.
        """
        # Perspective crop az eredeti képből
        crop = self._perspective_crop(image_rgb, polygon, bubble_id)
        if crop is None or crop.size == 0:
            logger.warning(f"[Recognizer #{bubble_id}] _perspective_crop returned empty/None.")
            return "", 0.0

        # Preprocessing
        def _preprocess_crop(self, crop):
   	 h, w = crop.shape[:2]
    	# ← IDE:
    	import cv2 as _cv2
    	_cv2.imwrite(f"output/rec_input_{w}x{h}.png", 
                 _cv2.cvtColor(crop, _cv2.COLOR_RGB2BGR))
    	import numpy as _np
    	print(f"REC INPUT: shape={crop.shape} mean={crop.mean():.1f} "
          f"min={crop.min()} max={crop.max()}")
    	# →
    	target_h = self._IMG_HEIGHT
    	...
        tensor = self._preprocess_crop(crop, bubble_id)

        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as e:
            logger.debug(f"Recognizer inference hiba: {e}")
            return "", 0.0

        # CTC decode
        text, conf = self._ctc_decode(outputs[0][0])
        return text, conf

    def _perspective_crop(
        self,
        image_rgb: np.ndarray,
        polygon: np.ndarray,
        bubble_id: int = -1,
    ) -> Optional[np.ndarray]:
        """
        Perspective transform: polygon régió → egyenes téglalap.

        Ez a koordináta-transzformáció BELSŐ lépés –
        a visszaadott OCRResult-ba NEM kerül bele.
        """
        logger.debug(f"[_perspective_crop #{bubble_id}] Input Polygon: {polygon.tolist()}")
        logger.debug(f"[_perspective_crop #{bubble_id}] Input Image Shape (bubble crop): {image_rgb.shape[:2]}")
        pts = polygon.astype(np.float32)

        if len(pts) == 4:
            # 4 sarokpont: direkt perspektíva korrekció
            width  = max(
                int(np.linalg.norm(pts[1] - pts[0])),
                int(np.linalg.norm(pts[2] - pts[3])),
                1,
            )
            height = max(
                int(np.linalg.norm(pts[3] - pts[0])),
                int(np.linalg.norm(pts[2] - pts[1])),
                1,
            )
            logger.debug(f"[_perspective_crop #{bubble_id}] Calculated Perspective Target Size: {width}x{height}")

            # Degenerate polygon guard: ha a szövegsor 4px-nél kisebb,
            # a recognizer értelmetlen outputot ad vissza
            if width < 4 or height < 4:
                logger.debug(
                    f"[_perspective_crop #{bubble_id}] Degenerate polygon guard triggered: width={width}, height={height} - Returning None."
                )
                return None
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ], dtype=np.float32)
            logger.debug(f"[_perspective_crop #{bubble_id}] Source pts: {pts.tolist()}, Dest dst: {dst.tolist()}")

            M     = cv2.getPerspectiveTransform(pts, dst)
            crop  = cv2.warpPerspective(image_rgb, M, (width, height))
            logger.debug(f"[_perspective_crop #{bubble_id}] cv2.warpPerspective() output shape: {crop.shape[:2]} (size: {crop.size})")
        else:
            # Több pont: bounding rect crop
            x1 = int(np.floor(pts[:, 0].min()))
            y1 = int(np.floor(pts[:, 1].min()))
            x2 = int(np.ceil(pts[:, 0].max()))
            y2 = int(np.ceil(pts[:, 1].max()))
            h_img, w_img = image_rgb.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_img, x2); y2 = min(h_img, y2)
            logger.debug(f"[_perspective_crop #{bubble_id}] Bounding rect crop coords: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            if x2 <= x1 or y2 <= y1:
                logger.debug(f"[_perspective_crop #{bubble_id}] Bounding rect degeneracy triggered: x1={x1}, y1={y1}, x2={x2}, y2={y2} - Returning None.")
                return None
            crop = image_rgb[y1:y2, x1:x2]
            logger.debug(f"[_perspective_crop #{bubble_id}] Bounding rect crop shape: {crop.shape[:2]} (size: {crop.size})")

        logger.debug(f"[_perspective_crop #{bubble_id}] Final crop shape before return: {crop.shape if crop is not None else 'None'}")
        return crop if crop.size > 0 else None

    def _preprocess_crop(self, crop: np.ndarray, bubble_id: int = -1) -> np.ndarray:
        """Crop előkészítése: resize → normalize → tensor."""
        logger.debug(f"[_preprocess_crop #{bubble_id}] Input crop shape: {crop.shape if crop is not None else 'None'}")
        h, w = crop.shape[:2]
        logger.debug(f"[_preprocess_crop #{bubble_id}] Crop dimensions: h={h}, w={w}")
        # Arányos átméretezés a rögzített magasságra
        target_h = self._IMG_HEIGHT
        target_w = max(int(w * target_h / max(h, 1)), 1)
        target_w = min(target_w, 1200)  # max szélesség

        resized = cv2.resize(
            crop, (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )

        img_f32  = resized.astype(np.float32) / 255.0
        img_norm = (img_f32 - self._MEAN) / self._STD

        # [H, W, 3] → [1, 3, H, W]
        tensor = img_norm.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        logger.debug(f"[_preprocess_crop #{bubble_id}] Output tensor shape: {tensor.shape}")
        return tensor

    def _ctc_decode(
        self,
        logits: np.ndarray,
    ) -> tuple[str, float]:
        """
        CTC greedy decode.

        Args:
            logits: [T, vocab_size] float32 – raw model output

        Returns:
            (szöveg, avg_confidence)
            Raw output – nincs language-specific normalizálás.
        """
        if logits.ndim != 2 or logits.shape[1] == 0:
            return "", 0.0

        # Softmax
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits_shifted)
        probs = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-9)

        # Greedy: argmax per timestep
        indices = np.argmax(probs, axis=1)
        confs   = probs[np.arange(len(indices)), indices]

        # CTC collapse: blank (0) és ismétlések eltávolítása
        chars: list[str]   = []
        scores: list[float] = []
        prev_idx = -1

        for idx, conf in zip(indices, confs):
            if idx == 0:  # blank
                prev_idx = idx
                continue
            if idx == prev_idx:
                continue
            if 0 < idx < len(self._charset):
                chars.append(self._charset[idx])
                scores.append(float(conf))
            prev_idx = idx

        text     = "".join(chars)
        avg_conf = float(np.mean(scores)) if scores else 0.0
        return text, avg_conf


# ══════════════════════════════════════════════════════════════════════════════
# READING ORDER + LINE GROUPING
# ══════════════════════════════════════════════════════════════════════════════

def _assign_reading_order_and_groups(
    results: list[OCRResult],
    y_tolerance_ratio: float = 0.6,
) -> list[OCRResult]:
    """
    Olvasási sorrend és sor-csoport hozzárendelés.

    Stratégia:
      - Y-koordináta alapú sor-csoportosítás
      - Sorokon belül X szerint rendezés
      - Sorok Y szerint rendezve

    A koordináták az EREDETI kép terében vannak – nincs normalizálás.

    Args:
        y_tolerance_ratio: sor-magasság arányos Y tolerancia

    Returns:
        Új OCRResult lista olvasási sorrenddel és group ID-val.
    """
    if not results:
        return []

    # Sorok képzése Y-centroid alapján
    rows: list[list[int]] = []
    for i, r in enumerate(results):
        y_center  = (r.bbox[1] + r.bbox[3]) / 2.0
        tol = max(r.line_height_est * y_tolerance_ratio, 8.0)
        placed = False
        for row in rows:
            row_y = np.mean([
                (results[j].bbox[1] + results[j].bbox[3]) / 2.0
                for j in row
            ])
            if abs(y_center - row_y) <= tol:
                row.append(i); placed = True; break
        if not placed:
            rows.append([i])

    # Sorok Y szerint rendezve
    rows.sort(key=lambda row: np.mean([
        (results[j].bbox[1] + results[j].bbox[3]) / 2.0
        for j in row
    ]))

    # Reading order + group ID hozzárendelés
    updated: list[OCRResult] = []
    reading_idx = 0

    for group_id, row in enumerate(rows):
        # Soron belül X szerint
        row.sort(key=lambda j: results[j].bbox[0])
        for j in row:
            r = results[j]
            # Frozen dataclass → új példány kell
            updated.append(OCRResult(
                text=r.text,
                confidence=r.confidence,
                polygon=r.polygon,
                bbox=r.bbox,
                rotation_angle=r.rotation_angle,
                language=r.language,
                line_group_id=group_id,
                reading_order=reading_idx,
                line_height_est=r.line_height_est,
                raw_det_score=r.raw_det_score,
            ))
            reading_idx += 1

    return updated


# ══════════════════════════════════════════════════════════════════════════════
# PPOCR-v5 PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class PPOCRv5Pipeline:
    """
    Detektor + Felismerő orchestráció.

    Stateless: nincs globális állapot, nincs kép feltételezés.
    Minden hívás önálló – párhuzamosítható (future).
    """

    def __init__(self) -> None:
        models = cfg.paths.models_dir
        det_path  = str(models / cfg.ocr.det_model_path)
        rec_path  = str(models / cfg.ocr.rec_model_path)
        dict_path = str(models / cfg.ocr.dict_path)

        self._check_files(det_path, rec_path, dict_path)

        self._detector   = PPOCRv5Detector(det_path)
        self._recognizer = PPOCRv5Recognizer(rec_path, dict_path)
        logger.info("PPOCRv5Pipeline inicializálva ✓")

    @staticmethod
    def _check_files(*paths: str) -> None:
        """Modellfájlok ellenőrzése – érthetű hibaüzenettel."""
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"OCR modellfájl nem található: {p}\n"
                    f"Elvárt hely: models/ocr/ppocr-v5-onnx/\n"
                    f"A fájlokat másold a models/ocr/ppocr-v5-onnx/ mappába."
                )

    def run(
        self,
        image_rgb: np.ndarray,
        bbox: Optional[list[int]] = None,
        language: str = None,
        bubble_idx: int = 0,  # <--- JAVÍTVA: Megkapja a buborék indexét
    ) -> list[OCRResult]:
        """
        Teljes OCR futtatása egy képen vagy régión.

        Args:
            image_rgb: [H, W, 3] uint8 RGB – EREDETI felbontás
            bbox:      opcionális [x1, y1, x2, y2] – csak erre a régióra fut
            language:  metaadat (nem befolyásolja a modell működését)
            bubble_idx:aktuális buborék azonosító indexe debug célra

        Returns:
            OCRResult lista, olvasási sorrendben.
            Koordináták az EREDETI kép terében.
        """
        if language is None:
            language = cfg.ocr.source_language

        h_orig, w_orig = image_rgb.shape[:2]
        debug = _ocr_debug_active()
        debug_dir = _ocr_debug_dir() if debug else None

        # ── Bbox validáció és crop készítés ──────────────────────────────────
        if bbox is not None:
            x1_raw, y1_raw, x2_raw, y2_raw = bbox

            # 1. Clamp a képhatárra
            x1 = max(0, int(x1_raw)); y1 = max(0, int(y1_raw))
            x2 = min(w_orig, int(x2_raw)); y2 = min(h_orig, int(y2_raw))

            # 2. Érvénytelen bbox
            if x2 <= x1 or y2 <= y1:
                logger.warning(
                    f"OCR: érvénytelen bbox skip "
                    f"[{x1_raw},{y1_raw},{x2_raw},{y2_raw}] → "
                    f"clamp után [{x1},{y1},{x2},{y2}] – skip")
                if debug and debug_dir:
                    _save_debug_crop(None, f"bbox_{x1}_{y1}", "INVALID",
                                     debug_dir, bubble_idx)  # <--- JAVÍTVA: bubble_idx használata
                return []

            crop_w = x2 - x1
            crop_h = y2 - y1

            # 3. Túl kis crop
            if crop_w < 4 or crop_h < 4:
                logger.warning(
                    f"OCR: degenerate crop {crop_w}x{crop_h}px [{bbox}] – skip")
                if debug and debug_dir:
                    region_tiny = image_rgb[y1:y2, x1:x2] if crop_w > 0 and crop_h > 0 else None
                    _save_debug_crop(region_tiny, f"b{bubble_idx}", "TINY",
                                     debug_dir, bubble_idx)  # <--- JAVÍTVA: bubble_idx használata
                return []

            region   = image_rgb[y1:y2, x1:x2]
            offset_x = float(x1)
            offset_y = float(y1)

            # 4. Crop shape sanity
            rh, rw = region.shape[:2]
            if rh != crop_h or rw != crop_w or region.size == 0:
                logger.error(
                    f"OCR: crop shape mismatch expected {crop_w}x{crop_h} "
                    f"got {rw}x{rh} – skip")
                return []

            if debug and debug_dir:
                _save_debug_crop(
                    cv2.cvtColor(region, cv2.COLOR_RGB2BGR),
                    f"b{bubble_idx}", "OK", debug_dir, bubble_idx)  # <--- JAVÍTVA: bubble_idx használata

            logger.debug(
                f"OCR crop #{bubble_idx}: [{x1},{y1},{x2},{y2}] "
                f"→ {crop_w}x{crop_h}px "
                f"(raw bbox: [{x1_raw},{y1_raw},{x2_raw},{y2_raw}])")
        else:
            region   = image_rgb
            offset_x = 0.0
            offset_y = 0.0
            if debug and debug_dir:
                _save_debug_crop(
                    cv2.cvtColor(region, cv2.COLOR_RGB2BGR),
                    "full_image", "OK", debug_dir, bubble_idx)  # <--- JAVÍTVA: bubble_idx használata

        # 1. Detektálás – polygon-ok a RÉGIÓ koordinátaterében
        try:
            polygons = self._detector.detect(region)
        except Exception as e:
            logger.error(f"OCR: detektor crash [{bbox}] {type(e).__name__}: {e}")
            if debug and debug_dir:
                _save_debug_crop(
                    cv2.cvtColor(region, cv2.COLOR_RGB2BGR) if bbox else None,
                    f"b{bubble_idx}", "CRASH", debug_dir, bubble_idx)  # <--- JAVÍTVA: bubble_idx használata
            return []

        if not polygons:
            return []

        # 2. Felismerés + OCRResult összeállítás
        results: list[OCRResult] = []
        for i, poly in enumerate(polygons):
            try:
                # <--- JAVÍTVA: A hibás _bubble_idx helyett a lokális bubble_idx-et adjuk át
                text, conf = self._recognizer.recognize(region, poly, bubble_id=bubble_idx)
            except Exception as e:
                logger.debug(f"Recognizer hiba polygon {i}: {e}")
                continue

            if not text.strip():
                continue

            # Offset visszaadás az EREDETI kép koordinátaterébe
            poly_orig = poly.copy()
            poly_orig[:, 0] += offset_x
            poly_orig[:, 1] += offset_y

            r = OCRResult.from_polygon_and_text(
                polygon_pts=poly_orig,
                text=text,
                confidence=conf,
                language=language,
                reading_order=i,
            )
            results.append(r)

        # 3. Reading order + line grouping
        results = _assign_reading_order_and_groups(results)

        return results


# ══════════════════════════════════════════════════════════════════════════════
# EASYOCR FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

class EasyOCRFallback:
    """
    EasyOCR alapú teljes fallback ha ppocr-v5 nem elérhető.

    Ugyanazt az OCRResult struktúrát adja vissza –
    a hívó kód nem tudja megkülönböztetni melyik backend fut.
    """

    def __init__(self) -> None:
        self._reader: Any = None
        self._load()

    def _load(self) -> None:
        if not cfg.ocr.fallback_to_easyocr:
            return
        try:
            import easyocr
            use_gpu = (cfg.device.device == "cuda")
            self._reader = easyocr.Reader(
                cfg.ocr.easyocr_lang,
                gpu=use_gpu,
                verbose=False,
            )
            logger.info("EasyOCR fallback betöltve ✓")
        except ImportError:
            logger.warning("EasyOCR sincs telepítve – OCR nem elérhető")
        except Exception as e:
            logger.warning(f"EasyOCR betöltési hiba: {e}")

    @property
    def available(self) -> bool:
        return self._reader is not None

    def run(
        self,
        image_rgb: np.ndarray,
        bbox: Optional[list[int]] = None,
        language: str = None,
    ) -> list[OCRResult]:
        if not self.available:
            return []
        if language is None:
            language = cfg.ocr.source_language

        h_orig, w_orig = image_rgb.shape[:2]
        if bbox:
            x1,y1,x2,y2 = bbox
            x1=max(0,x1); y1=max(0,y1)
            x2=min(w_orig,x2); y2=min(h_orig,y2)
            region   = image_rgb[y1:y2, x1:x2]
            offset_x = float(x1)
            offset_y = float(y1)
        else:
            region   = image_rgb
            offset_x = 0.0
            offset_y = 0.0

        try:
            raw = self._reader.readtext(
                region, detail=1, paragraph=False,
                min_size=5, text_threshold=0.6,
                low_text=0.3, link_threshold=0.4,
            )
        except Exception as e:
            logger.warning(f"EasyOCR hiba: {e}")
            return []

        results: list[OCRResult] = []
        for i, item in enumerate(raw):
            if len(item) < 3:
                continue
            easy_bbox, text, conf = item[0], item[1], item[2]
            if not text.strip():
                continue
            try:
                poly = np.array(easy_bbox, dtype=np.float32)
                poly[:, 0] += offset_x
                poly[:, 1] += offset_y
            except Exception:
                continue

            r = OCRResult.from_polygon_and_text(
                polygon_pts=poly, text=text,
                confidence=float(conf), language=language,
                reading_order=i,
            )
            results.append(r)

        return _assign_reading_order_and_groups(results)


# ══════════════════════════════════════════════════════════════════════════════
# COMIC OCR – publikus API
# ══════════════════════════════════════════════════════════════════════════════

_AUTO_MIN_VALID      = 1       # legalább ennyi valid találat kell
_AUTO_MIN_AVG_CONF   = 0.40    # átlag confidence alatt fallback
_AUTO_MIN_TEXT_LEN   = 2       # karakterek száma alatt garbage


def _score_results(results: list["OCRResult"]) -> tuple[int, float]:
    """
    OCR eredmény minőség scoring.

    Returns:
        (valid_count, avg_confidence)
    """
    valid = [r for r in results
             if r.text.strip() and len(r.text.strip()) >= _AUTO_MIN_TEXT_LEN
             and r.confidence >= 0.30]
    if not valid:
        return 0, 0.0
    avg_conf = sum(r.confidence for r in valid) / len(valid)
    return len(valid), round(avg_conf, 3)


def _is_usable(results: list["OCRResult"]) -> tuple[bool, str]:
    """
    Eldönti hogy az OCR eredmény használható-e.

    Returns:
        (usable, reason)
    """
    if not results:
        return False, "üres eredmény"
    valid_count, avg_conf = _score_results(results)
    if valid_count < _AUTO_MIN_VALID:
        return False, f"0 valid találat (összes: {len(results)})"
    if avg_conf < _AUTO_MIN_AVG_CONF:
        return False, f"alacsony avg_conf={avg_conf:.2f} < {_AUTO_MIN_AVG_CONF}"
    return True, "ok"


class ComicOCR:
    """
    Publikus OCR API – model-agnostic, backend-választással.
    """

    def __init__(self) -> None:
        self._primary:  Optional[PPOCRv5Pipeline] = None
        self._fallback: EasyOCRFallback           = EasyOCRFallback()
        self._backend:  str                       = cfg.ocr.backend
        self._init_primary()

    def _init_primary(self) -> None:
        """PPOCRv5 init – csak ha backend nem easyocr."""
        if self._backend == "easyocr":
            logger.info("OCR backend: EasyOCR (konfigurálva)")
            return
        try:
            self._primary = PPOCRv5Pipeline()
            logger.info("OCR backend: PPOCRv5 ONNX ✓")
        except FileNotFoundError as e:
            logger.warning(f"PPOCRv5 modellfájl hiányzik: {e}")
            logger.info("OCR backend: EasyOCR fallback (PPOCRv5 unavailable)")
        except Exception as e:
            logger.warning(f"PPOCRv5 init hiba: {e} – EasyOCR fallback")

    def reset_session(self) -> None:
        pass  # stateless – nincs resetelendő állapot

    def set_backend(self, backend: str) -> None:
        """Runtime backend váltás (CLI / orchestrator hívja)."""
        valid = ("auto", "ppocr", "easyocr")
        if backend not in valid:
            logger.warning(f"Ismeretlen OCR backend: {backend!r} – marad: {self._backend}")
            return
        if backend != self._backend:
            self._backend = backend
            logger.info(f"OCR backend váltva: {backend}")

    def extract(
        self,
        image_bgr: np.ndarray,
        bbox: Optional[list[int]] = None,
        language: str = None,
        bubble_idx: int = 0,  # <--- JAVÍTVA: Képes fogadni a bubble indexet
    ) -> list[OCRResult]:
        """
        Szöveg kinyerése – backend-aware, runtime fallback logikával.
        """
        if language is None:
            language = cfg.ocr.source_language

        # BGR → RGB (egyetlen konverzió)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        backend_mode = self._backend

        # ── Csak EasyOCR ──────────────────────────────────────────────────
        if backend_mode == "easyocr":
            if not self._fallback.available:
                logger.warning("EasyOCR backend kért, de nem elérhető")
                return []
            return self._run_backend(
                self._fallback, image_rgb, bbox, language, "easyocr")

        # ── Csak PPOCRv5 ──────────────────────────────────────────────────
        if backend_mode == "ppocr":
            if self._primary is None:
                logger.warning("PPOCRv5 backend kért, de nem elérhető")
                return []
            # <--- JAVÍTVA: bubble_idx továbbítása a pipeline felé
            return self._primary.run(image_rgb, bbox=bbox, language=language, bubble_idx=bubble_idx)

        # ── AUTO mód: PPOCRv5 → EasyOCR fallback ─────────────────────────
        results: list[OCRResult] = []
        if self._primary is not None:
            try:
                # <--- JAVÍTVA: bubble_idx továbbítása a pipeline felé
                results = self._primary.run(image_rgb, bbox=bbox, language=language, bubble_idx=bubble_idx)
            except Exception as e:
                logger.warning(f"PPOCRv5 kivétel: {e}")
                results = []

        # Runtime usability check
        usable, reason = _is_usable(results)
        valid_count, avg_conf = _score_results(results)

        logger.debug(
            f"OCR AUTO | backend=ppocr | valid={valid_count} | avg_conf={avg_conf:.2f} | usable={usable} | reason={reason}"
        )

        if not usable:
            if self._fallback.available:
                logger.info(f"Primary OCR unusable ({reason}) -> EasyOCR fallback")
                try:
                    fb_results = self._run_backend(
                        self._fallback, image_rgb, bbox, language, "easyocr")
                    fb_valid, fb_conf = _score_results(fb_results)
                    logger.debug(
                        f"OCR fallback | backend=easyocr | valid={fb_valid} | avg_conf={fb_conf:.2f}"
                    )
                    if fb_valid > valid_count or (
                            fb_valid == valid_count and fb_conf > avg_conf):
                        return fb_results
                except Exception as e:
                    logger.warning(f"EasyOCR fallback hiba: {e}")
            else:
                logger.debug("EasyOCR fallback nem elérhető – PPOCRv5 eredmény megtartva")

        return results

    @staticmethod
    def _run_backend(backend, image_rgb, bbox, language, name: str) -> list[OCRResult]:
        """Backend futtatás egységes exception handling-gel."""
        try:
            return backend.run(image_rgb, bbox=bbox, language=language)
        except Exception as e:
            logger.warning(f"OCR backend [{name}] hiba: {e}")
            return []

    def process_page(
        self,
        image_bgr: np.ndarray,
        bubbles: list[dict],
    ) -> list[dict]:
        """
        Egy oldal összes buborékjának OCR feldolgozása.
        """
        enriched: list[dict] = []
        debug = _ocr_debug_active()

        # Debug: layout boxes overlay az eredeti képen
        if debug:
            _ddir = _ocr_debug_dir()
            _save_layout_debug(image_bgr, bubbles, _ddir)
            logger.info(f"OCR debug: crop-ok mentve → {_ddir}")

        for _bubble_idx, bubble in enumerate(bubbles):
            bbox = bubble.get("bbox")
            try:
                # <--- JAVÍTVA: Átadjuk az aktuális _bubble_idx-et az extract hívásnak
                results = self.extract(
                    image_bgr,
                    bbox=bbox,
                    language=cfg.ocr.source_language,
                    bubble_idx=_bubble_idx
                )
                results_sorted = sorted(
                    results, key=lambda r: r.reading_order)
                texts = [r.text for r in results_sorted if r.is_valid]
                confs = [r.confidence for r in results_sorted if r.is_valid]

                raw_text = " ".join(texts).strip()
                avg_conf = float(np.mean(confs)) if confs else 0.0

                b = dict(bubble)
                b["raw_text"]       = raw_text
                b["ocr_confidence"] = round(avg_conf, 4)
                b["ocr_results"]    = [r.to_dict() for r in results]
                enriched.append(b)

                if raw_text:
                    logger.debug(
                        f"OCR #{bubble.get('order', '?')}: "
                        f"'{raw_text[:40]}' (conf={avg_conf:.2f})"
                    )

            except Exception as e:
                logger.warning(
                    f"OCR hiba buborék #{bubble.get('order', '?')} "
                    f"[{bbox}]: {e} – kihagyva"
                )
                b = dict(bubble)
                b["raw_text"]       = ""
                b["ocr_confidence"] = 0.0
                b["ocr_results"]    = []
                enriched.append(b)

        found = sum(1 for b in enriched if b["raw_text"])
        logger.info(f"OCR kész: {found}/{len(enriched)} buborékban szöveg")
        return enriched


# ══════════════════════════════════════════════════════════════════════════════
# Singleton + moduláris API
# ══════════════════════════════════════════════════════════════════════════════

_ocr_singleton: Optional[ComicOCR] = None


def get_ocr() -> ComicOCR:
    global _ocr_singleton
    if _ocr_singleton is None:
        _ocr_singleton = ComicOCR()
    return _ocr_singleton


def run_ocr(
    image_bgr: np.ndarray,
    bubbles: list[dict],
) -> list[dict]:
    """Moduláris API: OCR futtatása az összes buborékon."""
    return get_ocr().process_page(image_bgr, bubbles)
