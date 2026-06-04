#!/usr/bin/env python3
"""
OCR Pipeline – Gyökérok diagnózis
===================================
Futtatás: python diagnose_ocr.py input/teszt.jpg

Bizonyítja:
  1. Az audit script crop_shape=(0,0) tracking bug volt (nem pipeline hiba)
  2. Az összes buborék VALÓDI crop_shape-jét
  3. Egyetlen buborék teljes útja: bbox → crop → det → polygon → persp.crop → rec → logit → text
  4. Pontosan hol keletkezik hiba

Output: output/diagnose/
"""

import sys, os, csv, logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

logging.basicConfig(level=logging.DEBUG,
    format="%(asctime)s [DIAG] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("diagnose")

OUT = Path("output/diagnose")
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PER-BUBBLE RECORD
# ══════════════════════════════════════════════════════════════════════════════

class BubbleTrace:
    def __init__(self, idx):
        self.idx = idx
        # Layout
        self.bbox           = []          # [x1,y1,x2,y2] from layout
        # run()
        self.bbox_clamped   = []          # after max(0,x1) etc.
        self.crop_shape     = (0, 0)      # (h, w) of region
        self.skip_reason    = ""          # miért lett kihagyva ha (0,0)
        # Detector
        self.det_input_hw   = (0, 0)      # _preprocess kimenet H×W
        self.det_output_shape = ()
        self.polygon_count  = 0
        self.polygons       = []          # list of np arrays
        # Per-polygon (első polygon)
        self.poly0_raw      = None        # cv2.boxPoints sorrend
        self.poly0_reordered= None        # TL-TR-BR-BL
        self.persp_size     = (0, 0)      # perspective transform target
        self.persp_crop_hw  = (0, 0)      # warpPerspective kimenet
        # Recognizer
        self.rec_input_shape = ()
        self.logits_shape    = ()
        self.needs_transpose = False
        self.decoded_text    = ""
        self.conf            = 0.0
        self.top5            = []
        # Final
        self.final_text      = ""
        self.fallback_used   = False

# Globális trace tábla
_traces: dict[int, BubbleTrace] = {}
_current_idx = [0]   # mutable int – process_page patch frissíti

def get_trace(idx: int) -> BubbleTrace:
    if idx not in _traces:
        _traces[idx] = BubbleTrace(idx)
    return _traces[idx]

# ══════════════════════════════════════════════════════════════════════════════
# PATCH FÜGGVÉNYEK
# ══════════════════════════════════════════════════════════════════════════════

def patch_process_page(ComicOCR):
    """process_page patch: current_idx frissítése MINDEN buboréknál."""
    _orig = ComicOCR.process_page

    def patched(self, image_bgr, bubbles):
        for i, bubble in enumerate(bubbles):
            _current_idx[0] = i
            t = get_trace(i)
            t.bbox = list(bubble.get("bbox") or [])

        # Eredeti hívás – de most current_idx frissül futás közben
        return _orig(self, image_bgr, bubbles)

    ComicOCR.process_page = patched
    log.info("  Patched: ComicOCR.process_page")


def patch_extract(ComicOCR):
    """extract patch: current_idx folyamatos frissítése."""
    _orig = ComicOCR.extract

    def patched(self, image_bgr, bbox=None, language=None):
        # Az extract() hívása sorrendben → index a sorrend alapján
        # Keressük meg melyik bubble-hoz tartozik ez a bbox
        for i, t in _traces.items():
            if t.bbox == list(bbox or []):
                _current_idx[0] = i
                break
        return _orig(self, image_bgr, bbox, language)

    ComicOCR.extract = patched
    log.info("  Patched: ComicOCR.extract")


def patch_run(PPOCRv5Pipeline):
    """run() patch: teljes bbox és crop trace."""
    _orig = PPOCRv5Pipeline.run

    def patched(self, image_rgb, bbox=None, language=None):
        idx = _current_idx[0]
        t   = get_trace(idx)

        h_orig, w_orig = image_rgb.shape[:2]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cx1 = max(0, int(x1)); cy1 = max(0, int(y1))
            cx2 = min(w_orig, int(x2)); cy2 = min(h_orig, int(y2))
            t.bbox_clamped = [cx1, cy1, cx2, cy2]

            if cx2 <= cx1:
                t.crop_shape  = (0, 0)
                t.skip_reason = f"x2({cx2}) <= x1({cx1}) after clamp – image_w={w_orig}"
                log.warning(f"  B#{idx:02d}: SKIP – {t.skip_reason}")
            elif cy2 <= cy1:
                t.crop_shape  = (0, 0)
                t.skip_reason = f"y2({cy2}) <= y1({cy1}) after clamp – image_h={h_orig}"
                log.warning(f"  B#{idx:02d}: SKIP – {t.skip_reason}")
            elif (cx2-cx1) < 8 or (cy2-cy1) < 8:
                t.crop_shape  = (cy2-cy1, cx2-cx1)
                t.skip_reason = f"< 8px guard: {cx2-cx1}x{cy2-cy1}"
                log.warning(f"  B#{idx:02d}: SKIP – {t.skip_reason}")
            else:
                t.crop_shape = (cy2-cy1, cx2-cx1)
                log.debug(f"  B#{idx:02d}: bbox={bbox} → clamped=[{cx1},{cy1},{cx2},{cy2}] "
                           f"crop={t.crop_shape}")

                # Crop mentése debug képként (első 3 bubble)
                if idx < 3:
                    crop_bgr = cv2.cvtColor(
                        image_rgb[cy1:cy2, cx1:cx2], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(OUT / f"crop_{idx:02d}.png"), crop_bgr)
        else:
            t.crop_shape = image_rgb.shape[:2]

        return _orig(self, image_rgb, bbox, language)

    PPOCRv5Pipeline.run = patched
    log.info("  Patched: PPOCRv5Pipeline.run")


def patch_preprocess(PPOCRv5Detector):
    """_preprocess patch: det input shape trace."""
    _orig_static = PPOCRv5Detector._preprocess

    @staticmethod
    def patched(image_rgb):
        idx = _current_idx[0]
        t   = get_trace(idx)
        result = _orig_static(image_rgb)   # tensor, scale_x, scale_y
        tensor = result[0]
        t.det_input_hw = (tensor.shape[2], tensor.shape[3])  # (H, W)
        log.debug(f"  B#{idx:02d}: det_preprocess input={image_rgb.shape[:2]} "
                   f"→ tensor={tensor.shape}")
        return result

    PPOCRv5Detector._preprocess = patched
    log.info("  Patched: PPOCRv5Detector._preprocess")


def patch_detect(PPOCRv5Detector):
    """detect() patch: ONNX output shape + polygon trace."""
    _orig = PPOCRv5Detector.detect

    def patched(self, image_rgb):
        idx = _current_idx[0]
        t   = get_trace(idx)
        polygons = _orig(self, image_rgb)
        t.polygon_count = len(polygons)
        t.polygons      = [p.copy() for p in polygons]

        if polygons:
            t.poly0_raw = polygons[0].copy()
            # TL-TR-BR-BL reorder
            p4 = polygons[0].reshape(4, 2)
            sx = p4[np.argsort(p4[:, 0])]
            left  = sx[:2][np.argsort(sx[:2,  1])]
            right = sx[2:][np.argsort(sx[2:,  1])]
            t.poly0_reordered = np.array(
                [left[0], right[0], right[1], left[1]], dtype=np.float32)

        log.debug(f"  B#{idx:02d}: detect polygons={len(polygons)}")
        return polygons

    PPOCRv5Detector.detect = patched
    log.info("  Patched: PPOCRv5Detector.detect")


def patch_perspective_crop(PPOCRv5Recognizer):
    """_perspective_crop patch: target size és kimenet shape."""
    _orig = PPOCRv5Recognizer._perspective_crop

    def patched(self, image_rgb, polygon):
        idx = _current_idx[0]
        t   = get_trace(idx)

        pts = polygon.reshape(-1, 2).astype(np.float32)

        if len(pts) == 4:
            w = max(int(np.linalg.norm(pts[1]-pts[0])),
                    int(np.linalg.norm(pts[2]-pts[3])), 1)
            h = max(int(np.linalg.norm(pts[3]-pts[0])),
                    int(np.linalg.norm(pts[2]-pts[1])), 1)
            t.persp_size = (h, w)
            log.debug(f"  B#{idx:02d}: perspective target h={h} w={w} "
                       f"pts={pts.tolist()}")
        else:
            x1c = max(0, int(pts[:,0].min()))
            y1c = max(0, int(pts[:,1].min()))
            x2c = int(pts[:,0].max())
            y2c = int(pts[:,1].max())
            t.persp_size = (y2c-y1c, x2c-x1c)

        crop = _orig(self, image_rgb, polygon)

        if crop is not None:
            t.persp_crop_hw = crop.shape[:2]
            # Első buborék perspektíva cropjának mentése
            if idx < 3:
                cv2.imwrite(str(OUT / f"persp_crop_{idx:02d}.png"),
                            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        else:
            t.persp_crop_hw = (0, 0)
            log.warning(f"  B#{idx:02d}: _perspective_crop → None!")

        return crop

    PPOCRv5Recognizer._perspective_crop = patched
    log.info("  Patched: PPOCRv5Recognizer._perspective_crop")


def patch_recognize(PPOCRv5Recognizer, charset: list):
    """recognize() patch: logit shape + transpose teszt + top5."""
    _orig = PPOCRv5Recognizer.recognize

    def patched(self, image_rgb, polygon):
        idx = _current_idx[0]
        t   = get_trace(idx)

        # Pre-call: perspective crop mentve fent
        result = _orig(self, image_rgb, polygon)
        t.decoded_text = result[0] if result else ""
        t.conf         = result[1] if result else 0.0

        # Logit audit – direkt ONNX hívás ugyanarra a cropra
        try:
            crop   = self._perspective_crop(image_rgb, polygon)
            if crop is not None and crop.size > 0:
                tensor = self._preprocess_crop(crop)
                t.rec_input_shape = tuple(tensor.shape)
                outputs = self._session.run(None, {self._input_name: tensor})
                raw     = outputs[0]          # (N, T, C) or (N, C, T)
                t.logits_shape = tuple(raw.shape)
                logits0 = raw[0]              # (T, C) or (C, T)

                needs_T = (logits0.ndim == 2 and
                           logits0.shape[0] > logits0.shape[1])
                t.needs_transpose = needs_T

                # Top5 az első timestepből
                lref = logits0.T if needs_T else logits0
                probs0 = np.exp(lref[0]) / (np.exp(lref[0]).sum() + 1e-9)
                top5idx = np.argsort(probs0)[::-1][:5]
                t.top5  = [(int(i),
                             charset[i] if i < len(charset) else f"<{i}>",
                             float(probs0[i]))
                           for i in top5idx]

                log.debug(f"  B#{idx:02d}: logits={t.logits_shape} "
                           f"needs_T={needs_T} text={t.decoded_text!r} "
                           f"top5={t.top5}")
        except Exception as e:
            log.warning(f"  B#{idx:02d}: logit audit hiba: {e}")

        return result

    PPOCRv5Recognizer.recognize = patched
    log.info("  Patched: PPOCRv5Recognizer.recognize")


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERÁLÁS
# ══════════════════════════════════════════════════════════════════════════════

def generate_report():
    lines = []
    sep   = "═" * 72

    lines.append(sep)
    lines.append("  OCR PIPELINE GYÖKÉROK DIAGNÓZIS")
    lines.append(sep)

    # ── Per-bubble táblázat ────────────────────────────────────────────────
    lines.append("\n── BUBBLE TRACE TÁBLA ──────────────────────────────────────────")
    header = (f"{'#':>3} │ {'bbox':>22} │ {'crop':>11} │ "
              f"{'det':>5} │ {'poly':>4} │ {'persp':>11} │ "
              f"{'logit':>12} │ {'text':>20}")
    lines.append(header)
    lines.append("─" * len(header))

    crop_zero_count  = 0
    crop_valid_count = 0
    poly_zero_count  = 0

    for idx in sorted(_traces):
        t  = _traces[idx]
        bb = str(t.bbox) if t.bbox else "?"
        crop_s = f"{t.crop_shape[0]}×{t.crop_shape[1]}"
        det_hw = f"{t.det_input_hw[0]}×{t.det_input_hw[1]}" \
                 if t.det_input_hw != (0,0) else "—"
        persp  = f"{t.persp_crop_hw[0]}×{t.persp_crop_hw[1]}" \
                 if t.persp_crop_hw != (0,0) else "—"
        logit  = str(t.logits_shape) if t.logits_shape else "—"
        text   = repr(t.decoded_text[:18]) if t.decoded_text else "—"

        if t.crop_shape == (0, 0):
            crop_zero_count += 1
            reason = f"  ← {t.skip_reason}" if t.skip_reason else "  ← DEFAULT (tracking bug?)"
        else:
            crop_valid_count += 1
            reason = ""

        if t.polygon_count == 0 and t.crop_shape != (0, 0):
            poly_zero_count += 1

        row = (f"{idx:>3} │ {bb:>22} │ {crop_s:>11} │ "
               f"{det_hw:>5} │ {t.polygon_count:>4} │ {persp:>11} │ "
               f"{logit:>12} │ {text:>20}{reason}")
        lines.append(row)

    lines.append(f"\n  Összesítés: crop_valid={crop_valid_count}  "
                 f"crop_zero={crop_zero_count}  poly_zero={poly_zero_count}")

    # ── ROOT CAUSE analízis ────────────────────────────────────────────────
    lines.append("\n\n" + sep)
    lines.append("  ROOT CAUSE ANALÍZIS")
    lines.append(sep)

    # 1. Tracking bug bizonyítás
    default_zeros = sum(1 for t in _traces.values()
                        if t.crop_shape == (0,0) and not t.skip_reason)
    skip_zeros    = sum(1 for t in _traces.values()
                        if t.crop_shape == (0,0) and t.skip_reason)

    lines.append("\n1. crop_shape=(0,0) EREDETE:")
    if default_zeros > 0:
        lines.append(f"   TRACKING BUG: {default_zeros} buborék soha nem kapott crop_shape frissítést")
        lines.append("   (current_id nem frissült process_page() futása közben)")
        lines.append("   EVIDENCE: default BubbleTrace.crop_shape = (0,0)")
        lines.append("   AFFECTED: AuditCollector.current_id beállítás logikája az előző audit scriptben")
    if skip_zeros > 0:
        lines.append(f"   PIPELINE BUG: {skip_zeros} buborék valóban üres/invalid cropot kapott")
        for t in _traces.values():
            if t.crop_shape == (0,0) and t.skip_reason:
                lines.append(f"     B#{t.idx}: {t.skip_reason}")

    # 2. Polygon count
    if poly_zero_count > 0:
        lines.append(f"\n2. POLYGON HIBA: {poly_zero_count} valid crop → detektor 0 polygont adott")
        lines.append("   LEHETSÉGES OK: Conv.33 crash (crop < 64px) VAGY detektor nem talál szöveget")
        lines.append("   AFFECTED FUNCTION: PPOCRv5Detector.detect() → _preprocess() scale logika")

    # 3. Garbage text
    garbage = [(t.idx, t.decoded_text) for t in _traces.values()
               if t.decoded_text and len(t.decoded_text) > 1
               and not any(c.isalpha() for c in t.decoded_text.replace(' ', ''))]
    if garbage:
        lines.append(f"\n3. GARBAGE TEXT: {len(garbage)} buboréknál értelmetlen szöveg")
        for idx, txt in garbage:
            t = _traces[idx]
            lines.append(f"   B#{idx}: {txt!r}  logits={t.logits_shape}  "
                          f"needs_T={t.needs_transpose}  top5={t.top5}")
        lines.append("   GYANÚ: polygon pont-sorrend hiba (cv2.boxPoints ≠ TL-TR-BR-BL)")
        lines.append("   AFFECTED: PPOCRv5Detector._unclip_polygon + _perspective_crop")

    # 4. Polygon ordering evidence
    lines.append("\n4. POLYGON ORDERING AUDIT:")
    for idx in sorted(_traces)[:5]:  # első 5
        t = _traces[idx]
        if t.poly0_raw is not None:
            raw_s  = str(t.poly0_raw.tolist())
            reor_s = str(t.poly0_reordered.tolist()) if t.poly0_reordered is not None else "—"
            equal  = np.allclose(t.poly0_raw, t.poly0_reordered) \
                     if t.poly0_reordered is not None else False
            lines.append(f"   B#{idx}: raw={raw_s}")
            lines.append(f"        reord={reor_s}")
            lines.append(f"        {'SAME (rendezés nem szükséges)' if equal else 'KÜLÖNBÖZIK (reorder befolyásolja cropot)'}")

    # ── Mentés ─────────────────────────────────────────────────────────────
    report = "\n".join(lines)
    (OUT / "ROOT_CAUSE_REPORT.txt").write_text(report, encoding="utf-8")

    # CSV
    with open(OUT / "bubble_trace.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx","bbox","bbox_clamped","crop_shape","skip_reason",
                    "det_input_hw","polygon_count","persp_size","persp_crop_hw",
                    "rec_input_shape","logits_shape","needs_transpose",
                    "decoded_text","conf","fallback_used"])
        for idx in sorted(_traces):
            t = _traces[idx]
            w.writerow([idx, t.bbox, t.bbox_clamped, t.crop_shape,
                        t.skip_reason, t.det_input_hw, t.polygon_count,
                        t.persp_size, t.persp_crop_hw, t.rec_input_shape,
                        t.logits_shape, t.needs_transpose, t.decoded_text,
                        round(t.conf,4), t.fallback_used])

    print(report)
    print(f"\nMentve: {OUT}/ROOT_CAUSE_REPORT.txt")
    print(f"Mentve: {OUT}/bubble_trace.csv")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(image_path: str):
    log.info(f"Diagnózis: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        log.error(f"Kép nem olvasható: {image_path}"); sys.exit(1)
    log.info(f"Kép: {img_bgr.shape[1]}×{img_bgr.shape[0]}px")

    # Import
    try:
        import ocr as ocr_mod
    except Exception as e:
        log.error(f"ocr.py import hiba: {e}"); sys.exit(1)

    # Charset betöltés
    charset = ["blank"]
    dict_path = "models/ocr/ppocr-v5-onnx/ppocrv5_en_dict.txt"
    if Path(dict_path).exists():
        chars = Path(dict_path).read_text(encoding="utf-8").splitlines()
        charset = ["blank"] + [c for c in chars if c]
        log.info(f"Charset: {len(charset)} elem (blank + {len(charset)-1} char)")
        # Ellenőrzés: van-e space token?
        has_space = any(c == " " for c in charset)
        log.info(f"Space token a charsetben: {has_space}")
        if not has_space:
            log.warning("HIÁNYZÓ SPACE TOKEN – szóközök elvesznek a dekódolásból!")

    # Patches telepítése
    log.info("Patches telepítése...")
    patch_process_page(ocr_mod.ComicOCR)
    patch_extract(ocr_mod.ComicOCR)
    patch_run(ocr_mod.PPOCRv5Pipeline)
    patch_preprocess(ocr_mod.PPOCRv5Detector)
    patch_detect(ocr_mod.PPOCRv5Detector)
    patch_perspective_crop(ocr_mod.PPOCRv5Recognizer)
    patch_recognize(ocr_mod.PPOCRv5Recognizer, charset)

    # Layout
    log.info("Layout detection...")
    try:
        from layout import detect_bubbles
        bubbles = detect_bubbles(img_bgr)
        log.info(f"Layout: {len(bubbles)} buborék")
        log.info(f"  type(bubbles[0]) = {type(bubbles[0])}")
        log.info(f"  repr(bubbles[0]) = {repr(bubbles[0])[:120]}")

        # Trace inicializálás bbox-szal
        for i, b in enumerate(bubbles):
            bbox = b.get("bbox") if isinstance(b, dict) else \
                   getattr(b, "bbox", [])
            _traces[i] = BubbleTrace(i)
            _traces[i].bbox = list(bbox)
    except Exception as e:
        log.error(f"Layout hiba: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # OCR futtatás patch-elt módban
    log.info("OCR futtatás (patch-elt)...")
    try:
        comic_ocr = ocr_mod.ComicOCR()
        enriched  = comic_ocr.process_page(img_bgr, bubbles)
        log.info(f"process_page kész: {len(enriched)} eredmény")
        # Final text feltöltése
        for i, b in enumerate(enriched):
            if i in _traces:
                _traces[i].final_text  = b.get("raw_text", "")
                _traces[i].fallback_used = False  # heuristic
    except Exception as e:
        log.error(f"OCR hiba: {e}")
        import traceback; traceback.print_exc()

    # Report
    log.info("Riport generálása...")
    generate_report()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Használat: python diagnose_ocr.py input/teszt.jpg")
        sys.exit(1)
    main(sys.argv[1])
