#!/usr/bin/env python3
"""
OCR Pipeline – Bizonyíték alapú audit script
============================================
Futtatás: python audit_ocr.py input/teszt.jpg

Kimenet: output/audit/
  A1_normalization_check.txt       – ONNX model metaadatok + norm összehasonlítás
  A2_det_norm_diff.png             – Prob map: ImageNet vs 0.5/0.5 különbség
  B_bubble_trace.csv               – 13 bubble teljes lifecycle tábla
  C_bubble_XX_unclip.png           – Unclip összehasonlítás (saját vs okaglu2)
  D_bubble_XX_poly_order.png       – Polygon pont rendezés + crop before/after
  E_bubble_XX_logit_audit.txt      – Logit shape, top5, transpose teszt

Semmi nem változik az ocr.py-ban – csak monkey-patch + vizualizáció.
"""

import sys
import os
import csv
import logging
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AUDIT] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audit_ocr")

AUDIT_DIR = Path("output/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# AUDIT DATA COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BubbleRecord:
    """Egy buborék teljes lifecycle rekordja."""
    layout_id:              int   = 0
    bbox:                   list  = field(default_factory=list)
    crop_shape:             tuple = (0, 0)
    # Detection
    det_input_shape:        tuple = ()
    det_boxes_count:        int   = 0
    det_failed:             bool  = False
    det_fallback:           bool  = False
    # per-polygon detection data (első polygon)
    raw_contour:            Any   = None   # numpy contour
    pre_unclip_pts:         Any   = None   # raw boxPoints
    own_unclip_pts:         Any   = None   # what _unclip_polygon returns
    okaglu2_unclip_pts:     Any   = None   # proper pyclipper result
    poly_pts_raw:           Any   = None   # cv2.boxPoints order
    poly_pts_reordered:     Any   = None   # TL-TR-BR-BL order
    # Recognition
    rec_input_shape:        tuple = ()
    logits_shape_raw:       tuple = ()
    logits_needs_transpose: bool  = False
    decoded_own:            str   = ""    # without transpose
    decoded_transposed:     str   = ""    # with transpose
    top5_own:               list  = field(default_factory=list)
    top5_transposed:        list  = field(default_factory=list)
    # Pipeline results
    final_text:             str   = ""
    usable:                 bool  = False
    fallback_used:          bool  = False
    translated:             bool  = False
    translated_text:        str   = ""
    inpainted:              bool  = False


class AuditCollector:
    _instance = None

    def __init__(self):
        self.bubbles: Dict[int, BubbleRecord] = {}
        self.current_id: int = 0
        self.full_image_bgr: Optional[np.ndarray] = None
        self.charset: List[str] = []

    @classmethod
    def get(cls) -> "AuditCollector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rec(self, bubble_id: int) -> BubbleRecord:
        if bubble_id not in self.bubbles:
            self.bubbles[bubble_id] = BubbleRecord(layout_id=bubble_id)
        return self.bubbles[bubble_id]


collector = AuditCollector.get()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A: NORMALIZATION AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def audit_normalization(det_model_path: str, rec_model_path: str,
                        test_image_rgb: np.ndarray) -> None:
    """
    A1: ONNX model metaadatok + normalizáció összehasonlítás.
    Futtatja a det modelt kétszer (ImageNet vs 0.5/0.5) és menti a diff-et.
    """
    import onnxruntime as ort
    log.info("=== A: Normalizáció audit ===")
    out_lines = []

    for name, path in [("DET", det_model_path), ("REC", rec_model_path)]:
        try:
            sess = ort.InferenceSession(
                path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            meta  = sess.get_modelmeta()
            inp   = sess.get_inputs()[0]
            outp  = sess.get_outputs()[0]
            out_lines.append(f"\n=== {name} MODEL: {Path(path).name} ===")
            out_lines.append(f"  Input  name={inp.name}  shape={inp.shape}  dtype={inp.type}")
            out_lines.append(f"  Output name={outp.name}  shape={outp.shape}  dtype={outp.type}")
            out_lines.append(f"  Model version: {meta.version}")
            out_lines.append(f"  Custom metadata: {dict(meta.custom_metadata_map)}")
        except Exception as e:
            out_lines.append(f"\n=== {name} MODEL: HIBA: {e} ===")

    # Detektor futtatása kétféle normalizációval
    try:
        det_sess = ort.InferenceSession(
            det_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        inp_name = det_sess.get_inputs()[0].name

        h, w = test_image_rgb.shape[:2]
        limit = 960
        scale = min(limit / max(h, w), 1.0)
        new_h = max(32, (int(h * scale) // 32) * 32)
        new_w = max(32, (int(w * scale) // 32) * 32)
        resized = cv2.resize(test_image_rgb, (new_w, new_h)).astype(np.float32) / 255.0

        # ImageNet normalization (saját kód)
        mean_inet = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std_inet  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_inet = ((resized - mean_inet) / std_inet).transpose(2, 0, 1)[np.newaxis]

        # 0.5/0.5 normalization (okaglu2)
        norm_half = ((resized - 0.5) / 0.5).transpose(2, 0, 1)[np.newaxis]

        out_inet = det_sess.run(None, {inp_name: norm_inet.astype(np.float32)})[0][0, 0]
        out_half = det_sess.run(None, {inp_name: norm_half.astype(np.float32)})[0][0, 0]

        out_lines.append("\n=== NORMALIZÁCIÓ ÖSSZEHASONLÍTÁS (DET) ===")
        out_lines.append(f"  prob_map shape: {out_inet.shape}")
        out_lines.append(f"  ImageNet  – mean={out_inet.mean():.4f}  max={out_inet.max():.4f}  "
                         f"pixels>0.3: {(out_inet > 0.3).sum()}")
        out_lines.append(f"  0.5/0.5   – mean={out_half.mean():.4f}  max={out_half.max():.4f}  "
                         f"pixels>0.3: {(out_half > 0.3).sum()}")

        diff = np.abs(out_inet.astype(np.float32) - out_half.astype(np.float32))
        out_lines.append(f"  Különbség – mean={diff.mean():.4f}  max={diff.max():.4f}")

        # Melyiket érdemes használni?
        # Több pixel > 0.3 = több szöveg detektálva = valószínűleg helyes norm
        inet_score = (out_inet > 0.3).sum()
        half_score = (out_half > 0.3).sum()
        if inet_score > half_score * 1.5:
            out_lines.append("  KÖVETKEZTETÉS: ImageNet normalizáció több szöveget detektál → valószínűleg HELYES")
        elif half_score > inet_score * 1.5:
            out_lines.append("  KÖVETKEZTETÉS: 0.5/0.5 normalizáció több szöveget detektál → saját kód HIBÁS")
        else:
            out_lines.append("  KÖVETKEZTETÉS: Hasonló detektálás – mindkét normalizáció elfogadható ennél a modellnél")

        # Vizualizáció: diff kép
        vis_h, vis_w = out_inet.shape
        vis = np.zeros((vis_h, vis_w * 3 + 20, 3), dtype=np.uint8)
        def to_gray(pm):
            return np.clip(pm * 255, 0, 255).astype(np.uint8)
        vis[:, :vis_w]              = cv2.cvtColor(to_gray(out_inet), cv2.COLOR_GRAY2BGR)
        vis[:, vis_w+10:vis_w*2+10] = cv2.cvtColor(to_gray(out_half), cv2.COLOR_GRAY2BGR)
        diff_vis = np.clip(diff * 5 * 255, 0, 255).astype(np.uint8)
        vis[:, vis_w*2+20:]         = cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2BGR)
        for x, lbl in [(5, "ImageNet"), (vis_w + 15, "0.5/0.5"), (vis_w*2 + 25, "Diff x5")]:
            cv2.putText(vis, lbl, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imwrite(str(AUDIT_DIR / "A2_det_norm_diff.png"), vis)
        log.info(f"  Mentve: A2_det_norm_diff.png")

    except Exception as e:
        out_lines.append(f"\nNormalizáció összehasonlítás HIBA: {e}")
        import traceback
        out_lines.append(traceback.format_exc())

    report = "\n".join(out_lines)
    (AUDIT_DIR / "A1_normalization_check.txt").write_text(report, encoding="utf-8")
    log.info("  Mentve: A1_normalization_check.txt")
    print(report)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B: UNCLIP COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def _okaglu2_unclip(contour: np.ndarray, ratio: float = 2.0) -> Optional[np.ndarray]:
    """
    Valódi pyclipper-alapú polygon kiterjesztés – okaglu2 logika portja.
    Visszaad: [N, 2] float32 vagy None.
    """
    try:
        import pyclipper
        from shapely.geometry import Polygon as SPolygon

        pts = contour.reshape(-1, 2).tolist()
        if len(pts) < 3:
            return None
        poly    = SPolygon(pts)
        area    = poly.area
        length  = poly.length
        if length < 1e-6:
            return None
        distance = area * ratio / (length + 1e-6)

        pc = pyclipper.PyclipperOffset()
        pc.AddPath(
            [[int(p[0]), int(p[1])] for p in pts],
            pyclipper.JT_ROUND,
            pyclipper.ET_CLOSEDPOLYGON,
        )
        result = pc.Execute(distance)
        if not result:
            return None
        expanded = np.array(result[0], dtype=np.float32)

        # Visszaalakítás 4 pontos quadhoz (minAreaRect)
        rect = cv2.minAreaRect(expanded.astype(np.int32))
        box  = cv2.boxPoints(rect)
        return box.astype(np.float32)

    except ImportError:
        return None   # pyclipper nem elérhető
    except Exception:
        return None


def _reorder_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """TL → TR → BR → BL sorrend garantálása (okaglu2 _min_box logika)."""
    pts = pts.reshape(4, 2)
    sorted_x = pts[np.argsort(pts[:, 0])]   # X szerint rendez
    left  = sorted_x[:2][np.argsort(sorted_x[:2, 1])]   # bal 2: kisebb Y = TL
    right = sorted_x[2:][np.argsort(sorted_x[2:, 1])]   # jobb 2: kisebb Y = TR
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def _draw_quad(img, pts, color, label="", thickness=2):
    """4 sarokpont rajzolása képre számozott csúcsokkal."""
    pts_i = pts.reshape(4, 2).astype(np.int32)
    cv2.polylines(img, [pts_i], True, color, thickness)
    for i, (x, y) in enumerate(pts_i):
        cv2.circle(img, (int(x), int(y)), 4, color, -1)
        cv2.putText(img, f"{i}:{label}", (int(x)+4, int(y)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def visualize_unclip(bubble_id: int, region_rgb: np.ndarray,
                     contour: np.ndarray, ratio: float = 1.6) -> None:
    """C: Unclip összehasonlítás – raw contour, saját, okaglu2."""
    canvas_h = region_rgb.shape[0]
    canvas_w = region_rgb.shape[1]
    vis = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2BGR).copy()

    # Raw contour – kék
    cv2.drawContours(vis, [contour], -1, (255, 100, 0), 1)

    # Saját unclip (cv2.minAreaRect, ratio IGNORED)
    rect_own = cv2.minAreaRect(contour)
    box_own  = cv2.boxPoints(rect_own).astype(np.int32)
    cv2.polylines(vis, [box_own], True, (0, 0, 255), 2)
    _label_center(vis, box_own, "OWN(no ratio)", (0, 0, 255))

    # okaglu2 unclip (pyclipper)
    box_ok = _okaglu2_unclip(contour, ratio=2.0)
    if box_ok is not None:
        box_ok_i = box_ok.astype(np.int32)
        cv2.polylines(vis, [box_ok_i], True, (0, 220, 0), 2)
        _label_center(vis, box_ok_i, "OK2(ratio=2.0)", (0, 220, 0))
    else:
        cv2.putText(vis, "pyclipper N/A", (5, canvas_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)

    # Legend
    for i, (lbl, col) in enumerate([
        ("Blue=raw contour", (255, 100, 0)),
        ("Red=own(NO ratio)", (0, 0, 255)),
        ("Green=okaglu2(pyclipper)", (0, 220, 0)),
    ]):
        cv2.putText(vis, lbl, (4, 15 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    fname = AUDIT_DIR / f"C_bubble_{bubble_id:02d}_unclip.png"
    cv2.imwrite(str(fname), vis)
    log.info(f"  Mentve: {fname.name}")


def _label_center(img, pts, label, color):
    cx = int(pts[:, 0].mean())
    cy = int(pts[:, 1].mean())
    cv2.putText(img, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)


def visualize_polygon_ordering(bubble_id: int, region_rgb: np.ndarray,
                                pts_raw: np.ndarray, pts_reordered: np.ndarray) -> None:
    """D: Polygon pont-sorrend + perspective crop before/after."""
    h, w = region_rgb.shape[:2]
    pad   = 10
    crop_w = 200

    # Bal: raw ordering, Jobb: reordered
    vis = np.zeros((h + pad * 2, w * 2 + pad * 3, 3), dtype=np.uint8)
    bgr = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2BGR)
    vis[pad:pad+h, pad:pad+w]               = bgr
    vis[pad:pad+h, w+pad*2:w*2+pad*2]      = bgr

    _draw_quad(vis[pad:pad+h, pad:pad+w],
               pts_raw - pts_raw.min(axis=0),
               (0, 80, 255), "raw")
    _draw_quad(vis[pad:pad+h, w+pad*2:w*2+pad*2],
               pts_reordered - pts_reordered.min(axis=0),
               (0, 200, 0), "TL-TR-BR-BL")

    cv2.putText(vis, f"B#{bubble_id} RAW order (cv2.boxPoints)", (pad, pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
    cv2.putText(vis, f"B#{bubble_id} REORDERED (TL-TR-BR-BL)", (w + pad*2, pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

    # Perspective crop – raw ordering
    crop_raw  = _do_perspective_crop(region_rgb, pts_raw)
    crop_reor = _do_perspective_crop(region_rgb, pts_reordered)

    if crop_raw is not None and crop_reor is not None:
        th = 48
        def make_strip(crop):
            ch, cw = crop.shape[:2]
            tw = max(1, int(cw * th / max(ch, 1)))
            tw = min(tw, 400)
            resized = cv2.resize(
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), (tw, th))
            # pad to 400px wide
            strip = np.zeros((th, 400, 3), dtype=np.uint8)
            strip[:, :tw] = resized
            return strip

        strip_raw  = make_strip(crop_raw)
        strip_reor = make_strip(crop_reor)

        combined = np.zeros((th * 2 + 30, 400, 3), dtype=np.uint8)
        combined[:th] = strip_raw
        combined[th+30:] = strip_reor
        cv2.putText(combined, "RAW crop:",   (2, th - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 80, 255), 1)
        cv2.putText(combined, "REORD crop:", (2, th * 2 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 0), 1)

        # Paste crop strips next to quad comparison
        if combined.shape[0] <= vis.shape[0]:
            new_vis = np.zeros((vis.shape[0], vis.shape[1] + 410, 3), dtype=np.uint8)
            new_vis[:, :vis.shape[1]] = vis
            new_vis[:combined.shape[0], vis.shape[1] + 5:vis.shape[1] + 405] = combined
            vis = new_vis

    fname = AUDIT_DIR / f"D_bubble_{bubble_id:02d}_poly_order.png"
    cv2.imwrite(str(fname), vis)
    log.info(f"  Mentve: {fname.name}")


def _do_perspective_crop(image_rgb: np.ndarray,
                          pts: np.ndarray) -> Optional[np.ndarray]:
    pts = pts.reshape(4, 2).astype(np.float32)
    w = max(int(np.linalg.norm(pts[1]-pts[0])),
            int(np.linalg.norm(pts[2]-pts[3])), 1)
    h = max(int(np.linalg.norm(pts[3]-pts[0])),
            int(np.linalg.norm(pts[2]-pts[1])), 1)
    if w < 2 or h < 2:
        return None
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    try:
        M    = cv2.getPerspectiveTransform(pts, dst)
        crop = cv2.warpPerspective(image_rgb, M, (w, h))
        return crop if crop.size > 0 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E: LOGIT SHAPE + TRANSPOSE AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def audit_logit_shape(bubble_id: int, rec_session, input_name: str,
                      tensor: np.ndarray, charset: List[str]) -> Dict:
    """
    E: Logit shape ellenőrzés + decode mindkét orientációban.
    Visszaad dict-et az auditált adatokkal.
    """
    result = {
        "bubble_id": bubble_id,
        "tensor_input_shape": str(tensor.shape),
        "raw_output_shape": "",
        "needs_transpose": False,
        "decoded_own": "",
        "decoded_transposed": "",
        "top5_own": [],
        "top5_transposed": [],
        "conf_own": 0.0,
        "conf_transposed": 0.0,
    }

    try:
        outputs = rec_session.run(None, {input_name: tensor})
        raw_out = outputs[0]   # (N, T, C) or (N, C, T)
        result["raw_output_shape"] = str(raw_out.shape)

        logits_raw = raw_out[0]   # first sample: (T, C) or (C, T)

        # Determine if transpose needed
        # Convention: CRNN outputs (T, C) where T=timesteps, C=vocab_size
        # If shape is (C, T) then C >> T typically (vocab_size > time_steps)
        needs_T = (logits_raw.ndim == 2 and
                   logits_raw.shape[0] > logits_raw.shape[1])
        result["needs_transpose"] = bool(needs_T)

        # Decode WITHOUT transpose (own code behavior)
        text_own, conf_own, top5_own = _ctc_decode_audit(logits_raw, charset)
        result["decoded_own"]    = text_own
        result["conf_own"]       = conf_own
        result["top5_own"]       = top5_own

        # Decode WITH transpose
        logits_T = logits_raw.T if needs_T else logits_raw
        text_tr, conf_tr, top5_tr = _ctc_decode_audit(logits_T, charset)
        result["decoded_transposed"] = text_tr
        result["conf_transposed"]    = conf_tr
        result["top5_transposed"]    = top5_tr

    except Exception as e:
        result["error"] = str(e)

    return result


def _ctc_decode_audit(logits: np.ndarray,
                      charset: List[str]) -> Tuple[str, float, List]:
    """CTC decode + top5 per első timestep."""
    if logits.ndim != 2 or logits.shape[1] == 0:
        return "", 0.0, []

    logits_s = logits - logits.max(axis=1, keepdims=True)
    exp_l    = np.exp(logits_s)
    probs    = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-9)

    # Top5 az első timestepből
    t0_probs  = probs[0]
    top5_idx  = np.argsort(t0_probs)[::-1][:5]
    top5      = [(int(i),
                  charset[i] if i < len(charset) else f"<{i}>",
                  float(t0_probs[i]))
                 for i in top5_idx]

    # Greedy CTC
    indices  = np.argmax(probs, axis=1)
    confs    = probs[np.arange(len(indices)), indices]
    chars    = []
    scores   = []
    prev     = -1
    for idx, conf in zip(indices, confs):
        if idx == 0:
            prev = idx; continue
        if idx == prev:
            continue
        if 0 < idx < len(charset):
            chars.append(charset[idx])
            scores.append(float(conf))
        prev = idx

    text = "".join(chars)
    avg  = float(np.mean(scores)) if scores else 0.0
    return text, avg, top5


def save_logit_audit(bubble_id: int, result: Dict) -> None:
    lines = [
        f"=== BUBBLE #{bubble_id:02d} – LOGIT AUDIT ===",
        f"  Tensor input shape  : {result.get('tensor_input_shape')}",
        f"  ONNX output shape   : {result.get('raw_output_shape')}",
        f"  Needs transpose?    : {result.get('needs_transpose')}",
        "",
        f"  --- WITHOUT transpose (saját kód viselkedés) ---",
        f"  Decoded text  : {result.get('decoded_own')!r}",
        f"  Confidence    : {result.get('conf_own', 0):.3f}",
        f"  Top5 t=0      : {result.get('top5_own')}",
        "",
        f"  --- WITH transpose ---",
        f"  Decoded text  : {result.get('decoded_transposed')!r}",
        f"  Confidence    : {result.get('conf_transposed', 0):.3f}",
        f"  Top5 t=0      : {result.get('top5_transposed')}",
    ]
    if "error" in result:
        lines.append(f"\n  ERROR: {result['error']}")

    fname = AUDIT_DIR / f"E_bubble_{bubble_id:02d}_logit_audit.txt"
    fname.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  Mentve: {fname.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MONKEY-PATCHING – ocr.py hívások instrumentálása
# ══════════════════════════════════════════════════════════════════════════════

def install_patches(ocr_module) -> None:
    """
    Monkey-patch-eli az ocr.py kulcsfüggvényeit audit adatgyűjtéshez.
    Az eredeti viselkedés VÁLTOZATLAN marad – csak adatot gyűjtünk.
    """
    PPOCRv5Detector   = ocr_module.PPOCRv5Detector
    PPOCRv5Recognizer = ocr_module.PPOCRv5Recognizer
    PPOCRv5Pipeline   = ocr_module.PPOCRv5Pipeline

    # ── Patch 1: PPOCRv5Detector._decode_polygons ────────────────────────────
    orig_decode = PPOCRv5Detector._decode_polygons.__func__ \
        if hasattr(PPOCRv5Detector._decode_polygons, '__func__') \
        else staticmethod(PPOCRv5Detector._decode_polygons)

    @staticmethod
    def _decode_polygons_patched(prob_map, thresh, orig_w, orig_h,
                                  scale_x, scale_y,
                                  min_area=10.0, unclip_ratio=1.6):
        # Binárisan binarizáljuk – contourokat elkapjuk
        binary   = (prob_map > thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bid = collector.current_id
        rec = collector.rec(bid)
        rec.det_boxes_count = len([c for c in contours
                                   if cv2.contourArea(c) >= min_area])

        # Első valid contour audit mentés
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            rec.raw_contour = contour.copy()

            # Saját unclip – cv2.minAreaRect (ratio IGNORED)
            own_box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
            rec.own_unclip_pts = own_box.copy()

            # okaglu2 unclip – pyclipper
            rec.okaglu2_unclip_pts = _okaglu2_unclip(contour, ratio=2.0)

            # Polygon ordering audit
            rec.poly_pts_raw       = own_box.copy()
            rec.poly_pts_reordered = _reorder_tl_tr_br_bl(own_box).copy()
            break

        # Eredeti implementáció futtatása változatlanul
        return PPOCRv5Detector.__dict__["_decode_polygons"].__func__(
            prob_map, thresh, orig_w, orig_h,
            scale_x, scale_y, min_area, unclip_ratio
        ) if "_decode_polygons" in PPOCRv5Detector.__dict__ else []

    # Fallback: ha az orig_decode nem hívható így, wrap differently
    _orig_static = staticmethod(ocr_module.PPOCRv5Detector._decode_polygons)

    @staticmethod
    def _decode_polygons_patched_v2(prob_map, thresh, orig_w, orig_h,
                                     scale_x, scale_y,
                                     min_area=10.0, unclip_ratio=1.6):
        # Audit collect
        binary   = (prob_map > thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bid = collector.current_id
        rec = collector.rec(bid)
        rec.det_boxes_count = len([c for c in contours
                                   if cv2.contourArea(c) >= min_area])
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                rec.raw_contour    = contour.copy()
                own_box            = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
                rec.own_unclip_pts = own_box.copy()
                rec.okaglu2_unclip_pts = _okaglu2_unclip(contour, ratio=2.0)
                rec.poly_pts_raw       = own_box.copy()
                rec.poly_pts_reordered = _reorder_tl_tr_br_bl(own_box).copy()
                break
        # Call original
        return _orig_static.__func__(prob_map, thresh, orig_w, orig_h,
                                      scale_x, scale_y, min_area, unclip_ratio)

    PPOCRv5Detector._decode_polygons = _decode_polygons_patched_v2
    log.info("  Patched: PPOCRv5Detector._decode_polygons")

    # ── Patch 2: PPOCRv5Recognizer.recognize ─────────────────────────────────
    _orig_recognize = PPOCRv5Recognizer.recognize

    def recognize_patched(self, image_rgb, polygon):
        bid = collector.current_id
        rec = collector.rec(bid)

        crop   = self._perspective_crop(image_rgb, polygon)
        if crop is None or crop.size == 0:
            return "", 0.0

        tensor = self._preprocess_crop(crop)
        rec.rec_input_shape = tuple(tensor.shape)

        # Logit audit
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
            logits_raw = outputs[0][0]
            rec.logits_shape_raw = tuple(logits_raw.shape)

            needs_T = (logits_raw.ndim == 2 and
                       logits_raw.shape[0] > logits_raw.shape[1])
            rec.logits_needs_transpose = needs_T

            charset = self._charset
            t_own, c_own, top5_own = _ctc_decode_audit(logits_raw, charset)
            rec.decoded_own = t_own

            logits_T = logits_raw.T if needs_T else logits_raw
            t_tr, c_tr, top5_tr = _ctc_decode_audit(logits_T, charset)
            rec.decoded_transposed = t_tr
            rec.top5_own        = top5_own
            rec.top5_transposed = top5_tr

            # Logit audit mentése
            save_logit_audit(bid, {
                "tensor_input_shape":  str(tensor.shape),
                "raw_output_shape":    str(outputs[0].shape),
                "needs_transpose":     needs_T,
                "decoded_own":         t_own,
                "conf_own":            c_own,
                "top5_own":            top5_own,
                "decoded_transposed":  t_tr,
                "conf_transposed":     c_tr,
                "top5_transposed":     top5_tr,
            })
        except Exception as e:
            log.warning(f"  Logit audit hiba bubble #{bid}: {e}")

        # Eredeti hívás
        return _orig_recognize(self, image_rgb, polygon)

    PPOCRv5Recognizer.recognize = recognize_patched
    log.info("  Patched: PPOCRv5Recognizer.recognize")

    # ── Patch 3: PPOCRv5Pipeline.run ─────────────────────────────────────────
    _orig_run = PPOCRv5Pipeline.run

    def run_patched(self, image_rgb, bbox=None, language=None):
        bid = collector.current_id
        rec = collector.rec(bid)
        rec.bbox = list(bbox) if bbox else []

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            h, w = image_rgb.shape[:2]
            x1c = max(0, int(x1)); y1c = max(0, int(y1))
            x2c = min(w, int(x2)); y2c = min(h, int(y2))
            if x2c > x1c and y2c > y1c:
                rec.crop_shape = (y2c - y1c, x2c - x1c)
            else:
                rec.crop_shape = (0, 0)
        else:
            rec.crop_shape = image_rgb.shape[:2]

        result = _orig_run(self, image_rgb, bbox, language)
        rec.final_text = " ".join(r.text for r in result if r.text.strip())
        rec.usable     = bool(rec.final_text.strip())
        return result

    PPOCRv5Pipeline.run = run_patched
    log.info("  Patched: PPOCRv5Pipeline.run")

    # ── Patch 4: ComicOCR.extract – fallback tracking ─────────────────────────
    ComicOCR = ocr_module.ComicOCR
    _orig_extract = ComicOCR.extract

    def extract_patched(self, image_bgr, bbox=None, language=None):
        bid = collector.current_id
        result = _orig_extract(self, image_bgr, bbox, language)
        rec = collector.rec(bid)
        # Check if primary was unusable (fallback triggered)
        # We infer this from the "unusable" log – primary returns [] → fallback
        # A clean way: check if result came from primary or fallback
        # by seeing if final_text was set by run_patched (primary) or not
        if not rec.usable and result:
            rec.fallback_used = True
            rec.final_text    = " ".join(r.text for r in result if r.text.strip())
            rec.usable        = bool(rec.final_text.strip())
        return result

    ComicOCR.extract = extract_patched
    log.info("  Patched: ComicOCR.extract")

    log.info("Összes patch telepítve.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B: BUBBLE TRACE TABLE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_trace_table() -> str:
    """B: CSV trace tábla mind a 13 buborékra."""
    fname = AUDIT_DIR / "B_bubble_trace.csv"
    fields = [
        "layout_id", "bbox", "crop_shape",
        "det_boxes_count", "det_failed",
        "rec_input_shape", "logits_shape", "needs_transpose",
        "decoded_own", "decoded_transposed",
        "final_text", "usable", "fallback_used",
        "translated", "translated_text", "inpainted",
    ]
    rows = []
    for bid in sorted(collector.bubbles):
        r = collector.bubbles[bid]
        rows.append({
            "layout_id":         r.layout_id,
            "bbox":              str(r.bbox),
            "crop_shape":        str(r.crop_shape),
            "det_boxes_count":   r.det_boxes_count,
            "det_failed":        r.det_failed,
            "rec_input_shape":   str(r.rec_input_shape),
            "logits_shape":      str(r.logits_shape_raw),
            "needs_transpose":   r.logits_needs_transpose,
            "decoded_own":       r.decoded_own,
            "decoded_transposed":r.decoded_transposed,
            "final_text":        r.final_text,
            "usable":            r.usable,
            "fallback_used":     r.fallback_used,
            "translated":        r.translated,
            "translated_text":   r.translated_text,
            "inpainted":         r.inpainted,
        })

    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Ascii összefoglalás
    total   = len(rows)
    usable  = sum(1 for r in rows if r["usable"])
    fallback = sum(1 for r in rows if r["fallback_used"])
    transl  = sum(1 for r in rows if r["translated"])
    inpaint = sum(1 for r in rows if r["inpainted"])
    needs_T = sum(1 for r in rows if r["needs_transpose"])

    summary = textwrap.dedent(f"""
    ╔══════════════════════════════════════════╗
    ║         BUBBLE LIFECYCLE SUMMARY         ║
    ╠══════════════════════════════════════════╣
    ║  Detektált buborékok:  {total:>3}                ║
    ║  OCR szöveg van:       {usable:>3}  ({usable/max(total,1)*100:.0f}%)           ║
    ║  EasyOCR fallback:     {fallback:>3}                ║
    ║  Logit transpose kell: {needs_T:>3}                ║
    ║  Lefordítva:           {transl:>3}                ║
    ║  Inpainting kész:      {inpaint:>3}                ║
    ╚══════════════════════════════════════════╝
    """)

    log.info(summary)
    log.info(f"  Mentve: {fname.name}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AUDIT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_audit(image_path: str) -> None:
    """Teljes audit futtatása egy képen."""
    import importlib

    log.info(f"======================================")
    log.info(f"OCR Pipeline Audit: {image_path}")
    log.info(f"Output: {AUDIT_DIR}/")
    log.info(f"======================================")

    # Kép betöltése
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        log.error(f"Kép nem betölthető: {image_path}")
        sys.exit(1)
    collector.full_image_bgr = img_bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    log.info(f"Kép: {img_bgr.shape[1]}×{img_bgr.shape[0]}px")

    # OCR modul importálása
    log.info("ocr.py importálása és monkey-patch telepítése...")
    try:
        import ocr as ocr_module
        install_patches(ocr_module)
        collector.charset = ocr_module.PPOCRv5Recognizer._load_charset(
            "models/ocr/ppocr-v5-onnx/ppocrv5_en_dict.txt")
    except Exception as e:
        log.error(f"Import hiba: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # Section A: Normalizáció audit
    log.info("\n=== A: Normalizáció audit ===")
    audit_normalization(
        det_model_path="models/ocr/ppocr-v5-onnx/ml_PP-OCRv5_mobile_det.onnx",
        rec_model_path="models/ocr/ppocr-v5-onnx/en_PP-OCRv5_rec_mobile_infer.onnx",
        test_image_rgb=img_rgb,
    )

    # OCR + Layout futtatása – patch-elt módban
    log.info("\n=== Pipeline futtatás (audit módban) ===")
    try:
        # ── Layout detection ─────────────────────────────────────────────────
        # A layout.py publikus API-ja detect_bubbles() -> list[dict]
        # BubbleDetector.detect() -> tuple (raw), NINCS .bbox!
        # LayoutDetector.detect() -> list[BubbleRegion]
        # detect_bubbles() -> [b.to_dict() for b in LayoutDetector.detect()]
        from layout import detect_bubbles, BubbleRegion, LayoutDetector

        raw_result = detect_bubbles(img_bgr)

        # ── DIAGNÓZIS PRINT ─────────────────────────────────────────────────
        log.info("--- Layout output diagnózis ---")
        log.info(f"  type(raw_result)    = {type(raw_result)}")
        log.info(f"  len(raw_result)     = {len(raw_result)}")
        if raw_result:
            b0 = raw_result[0]
            log.info(f"  type(raw_result[0]) = {type(b0)}")
            log.info(f"  repr(raw_result[0]) = {repr(b0)}")

        # ── Robusztus bbox kinyerés ──────────────────────────────────────────
        def get_bbox(item) -> list:
            """dict, BubbleRegion, vagy list[int] kezelése."""
            if isinstance(item, dict):
                return item.get("bbox", [])
            if hasattr(item, "bbox"):          # BubbleRegion dataclass
                return item.bbox
            if isinstance(item, (list, tuple)) and len(item) == 4:
                return list(item)              # raw [x1,y1,x2,y2]
            return []

        bubble_dicts = [
            {
                "bbox":      get_bbox(b),
                "bubble_id": i,
                "id":        b.get("id", i) if isinstance(b, dict) else getattr(b, "id", i),
                "type":      b.get("type", "bubble") if isinstance(b, dict) else getattr(b, "type", "bubble"),
                "confidence":b.get("confidence", 0.0) if isinstance(b, dict) else getattr(b, "confidence", 0.0),
            }
            for i, b in enumerate(raw_result)
        ]
        log.info(f"  {len(bubble_dicts)} bubble dict előkészítve")

        # Collector inicializálás
        for i, b_dict in enumerate(bubble_dicts):
            collector.current_id = i
            rec = collector.rec(i)
            rec.layout_id = i
            rec.bbox      = b_dict["bbox"]

        # process_page fut – patch-elt recognizer és detector hívódik
        comic_ocr = ocr_module.ComicOCR()
        enriched  = comic_ocr.process_page(img_bgr, bubble_dicts)

    except Exception as e:
        log.error(f"Pipeline futtatás hiba: {e}")
        import traceback; traceback.print_exc()

    # Vizualizációk generálása az összegyűjtött adatokból
    log.info("\n=== C+D: Vizualizációk generálása ===")
    for bid, rec in collector.bubbles.items():
        bbox = rec.bbox if isinstance(rec.bbox, (list, tuple)) and len(rec.bbox) == 4 else []
        if not bbox:
            continue
        x1,y1,x2,y2 = bbox
        x1c=max(0,int(x1)); y1c=max(0,int(y1))
        x2c=min(img_rgb.shape[1],int(x2)); y2c=min(img_rgb.shape[0],int(y2))
        if x2c <= x1c or y2c <= y1c:
            continue
        region = img_rgb[y1c:y2c, x1c:x2c]

        # C: Unclip összehasonlítás – csak ha volt detektált kontúr
        if rec.raw_contour is not None:
            visualize_unclip(bid, region, rec.raw_contour)

        # D: Polygon ordering – csak ha volt polygon adat
        if rec.poly_pts_raw is not None and rec.poly_pts_reordered is not None:
            visualize_polygon_ordering(
                bid, region,
                rec.poly_pts_raw,
                rec.poly_pts_reordered,
            )

    # B: Bubble trace CSV
    log.info("\n=== B: Bubble trace tábla ===")
    build_trace_table()

    log.info(f"\n======================================")
    log.info(f"Audit kész. Output: {AUDIT_DIR}/")
    log.info(f"======================================")

    # Összefoglaló a terminálra
    print("\nFontosabb megállapítások:")
    for bid in sorted(collector.bubbles):
        r = collector.bubbles[bid]
        transpose_flag = "⚠️ TRANSPOSE KELL" if r.logits_needs_transpose else "ok"
        usable_flag    = "✓" if r.usable else "✗ ELVESZETT"
        fallback_flag  = " [EasyOCR]" if r.fallback_used else ""
        print(f"  Bubble #{bid:02d}: crop={r.crop_shape}  "
              f"det_boxes={r.det_boxes_count}  "
              f"text={r.final_text[:30]!r}  "
              f"{usable_flag}{fallback_flag}  {transpose_flag}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Használat: python audit_ocr.py input/teszt.jpg")
        sys.exit(1)
    run_audit(sys.argv[1])
