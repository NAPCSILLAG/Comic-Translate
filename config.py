"""
config.py - Typed, dataclass-alapú konfiguráció.

Architektúra:
  - Minden alrendszernek saját config dataclass-a van
  - A globális Config objektum ezeket aggregálja
  - YAML override támogatás (opcionális)
  - Környezeti változók override-olják a default értékeket

Bővítés:
  - Új alrendszer = új dataclass + beillesztés a Config-ba
  - UI / külső AI könnyen override-olhatja az értékeket
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch

logger = logging.getLogger(__name__)

# ── Projekt gyökér ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

# ── CUDA detektálás ────────────────────────────────────────────────────────────
_cuda_available = torch.cuda.is_available()
_device_default  = "cuda" if _cuda_available else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Alrendszer konfigurációk
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PathConfig:
    """I/O útvonalak."""
    base_dir:   Path = BASE_DIR
    input_dir:  Path = BASE_DIR / "input"
    output_dir: Path = BASE_DIR / "output"
    debug_dir:  Path = BASE_DIR / "debug"
    font_dir:   Path = BASE_DIR / "fonts"
    log_dir:    Path = BASE_DIR / "logs"
    models_dir: Path = BASE_DIR / "models"

    def ensure_dirs(self) -> None:
        for d in [self.input_dir, self.output_dir,
                  self.debug_dir, self.font_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class DeviceConfig:
    """CUDA / CPU eszköz beállítások."""
    device:            str   = _device_default
    cuda_device_id:    int   = 0
    # float16 GPU-n, float32 CPU-n
    use_fp16:          bool  = True
    # VRAM felszabadítás pipeline lépések között
    sequential_mode:   bool  = True
    vram_safety_margin_gb: float = 1.5

    @property
    def torch_dtype(self):
        import torch
        return torch.float16 if (self.device == "cuda" and self.use_fp16) else torch.float32

    def vram_free_gb(self) -> float:
        if not torch.cuda.is_available(): return 0.0
        free, _ = torch.cuda.mem_get_info(self.cuda_device_id)
        return free / 1e9

    def vram_used_gb(self) -> float:
        if not torch.cuda.is_available(): return 0.0
        _, total = torch.cuda.mem_get_info(self.cuda_device_id)
        free, _  = torch.cuda.mem_get_info(self.cuda_device_id)
        return (total - free) / 1e9

    def cleanup(self) -> None:
        """VRAM cache ürítés pipeline lépések között."""
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


@dataclass
class LayoutConfig:
    """Speech bubble detektálás beállításai."""
    # Model útvonal (relatív a models_dir-hez)
    model_path:       str   = "detection/comic-speech-bubble-detector.pt"
    fallback_model:   str   = "yolov8m"
    conf_threshold:   float = 0.25
    iou_threshold:    float = 0.40
    # Üres lista = minden osztály elfogadva (comics modellnél helyes)
    bubble_class_ids: list  = field(default_factory=list)
    # Class-specific confidence thresholds
    # Narráció: szigorúbb (nagy, könnyen detektálható)
    # Bubble:   közepes
    # SFX:      permisszívebb (gyenge detection score-ok)
    conf_bubble:      float = 0.25
    conf_narration:   float = 0.40
    conf_sfx:         float = 0.18
    # Olvasási sorrend Y-tolerancia (px) – nagyobb = széles panelek jobb kezelése
    reading_order_y_tolerance: int = 50
    # Inference képméret – nagyobb = jobb kis buborék detektálás
    inference_imgsz:  int   = 1280
    # Minimális buborék méret (px)
    min_bubble_width:  int  = 20
    min_bubble_height: int  = 15
    # Teljes oldalt lefedő téves detektálás szűrő (arány)
    max_page_coverage: float = 0.70


@dataclass
class OCRConfig:
    """OCR beállítások – language-agnostic design."""
    # Model útvonalak (relatív a models_dir-hez)
    det_model_path: str = "ocr/ppocr-v5-onnx/ml_PP-OCRv5_mobile_det.onnx"
    rec_model_path: str = "ocr/ppocr-v5-onnx/en_PP-OCRv5_rec_mobile_infer.onnx"
    dict_path:      str = "ocr/ppocr-v5-onnx/ppocrv5_en_dict.txt"
    # Forrás nyelv (csak metaadathoz – az OCR engine maga nem feltételez nyelvet)
    source_language: str = "en"
    # ONNX Runtime provider prioritás
    use_gpu:        bool  = True
    # OCR backend valasztas
    # auto    = PPOCRv5 elsonek, EasyOCR fallback ha 0 valid result
    # ppocr   = csak PPOCRv5
    # easyocr = csak EasyOCR
    # paddleocr = lokális PaddleOCR pipeline
    # minicpm_ocr = helyi MiniCPM OCR provider (buborék szinten)
    # gemini_flash = Gemini Flash cloud AI (OCR + felismerés + fordítás)
    # qwen2_vl = Ollama-alapú VLM grounding OCR
    backend: str = "qwen2_vl"
    # Confidence threshold – alacsonyabb = több szöveg detektálva
    det_thresh: float = 0.30
    rec_thresh: float = 0.50
    # Preprocessing
    use_angle_cls: bool = True
    max_image_side: int = 2560   # felskálázás limit
    # Fallback: EasyOCR ha ppocr-v5 nem elérhető
    fallback_to_easyocr: bool = True
    easyocr_lang: list = field(default_factory=lambda: ["en"])
    # Extras
    minicpm_model_name: str = "minicpm-v:latest"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-exp"


@dataclass
class InpaintingConfig:
    """LaMa ONNX inpainting beállítások."""
    # Model útvonal
    lama_model_path: str  = "inpainting/lama-manga-dynamic.onnx"
    # Maszk kiterjesztés – KRITIKUS: fednie kell az anti-aliased széleket
    mask_dilation_px:   int   = 12    # alap dilation
    mask_dilation_extra: int  = 6     # extra a szöveg szélein
    # Feathering a varrat elkerüléséhez
    mask_feather_px:    int   = 8
    # Ha a terület egyszerű (alacsony részletesség) → LaMa kihagyása
    simple_fill_threshold: float = 0.92   # háttér homogenitás küszöb
    # Context-aware fill: hány px-es szegélyt mintavételezünk
    bg_sample_border_px: int  = 12
    # ONNX futtatás
    use_gpu:            bool  = True
    # OpenCV fallback ha LaMa nem elérhető
    fallback_to_opencv: bool  = True
    # Debug: maszk vizualizáció mentése
    debug_save_masks:   bool  = False


@dataclass
class RenderingConfig:
    """Tipográfiai rendering beállítások."""
    # Font útvonalak (relatív BASE_DIR-hez)
    font_bold:       str  = "fonts/NotoSans-Bold.ttf"
    font_regular:    str  = "fonts/NotoSans-Regular.ttf"
    font_fallback:   str  = "fonts/NotoSans-Regular.ttf"  # Unicode fallback
    # Font méret határok
    font_size_min:   int  = 9
    font_size_max:   int  = 42
    font_size_step:  int  = 1    # bináris keresés lépésköze
    # Szöveg megjelenés
    text_color:      tuple = (15, 15, 15)         # majdnem fekete
    outline_color:   tuple = (255, 255, 255)       # fehér körvonal
    outline_width:   int   = 2
    shadow_enabled:  bool  = False
    shadow_offset:   tuple = (1, 1)
    shadow_color:    tuple = (0, 0, 0)
    shadow_alpha:    float = 0.35
    # Sortörés és margók
    line_spacing:    float = 1.22   # sor magasság szorzó
    # Buborék belsejének relatív margója (buborék méret arányában)
    padding_ratio:   float = 0.12   # 12% minden oldaltól
    padding_min_px:  int   = 6      # minimális abszolút margó
    # Supersampling – KRITIKUS a professzionális minőséghez
    supersample_factor: int = 2     # 2x = jó minőség, 4x = maximális
    # Lanczos downsample
    use_lanczos:     bool = True
    # Score-driven layout
    layout_candidates: int = 8      # hány layout variánst próbáljon
    min_coverage_score: float = 0.3 # minimum kihasználtság score
    # Bubble shape adaptation
    shape_adaptation: bool = True   # ellipszis vs. téglalap vs. irreguláris
    # Distance field safety
    min_edge_distance_px: int = 4   # glyph minimum távolsága a buborék szélétől
    # Debug mód
    debug_layout:    bool = False    # layout terv vizualizáció mentése


@dataclass
class TranslationConfig:
    """Ollama / LLM fordítás beállítások."""
    base_url:       str   = "http://localhost:11434"
    model:          str   = "translategemma:latest"
    max_tokens:     int   = 256
    temperature:    float = 0.2
    timeout_sec:    int   = 90
    retries:        int   = 2
    # Ha az OCR confidence alacsony → fordítás kihagyása
    min_ocr_confidence: float = 0.45


@dataclass
class VisionConfig:
    """
    Unified vision konfiguráció – configurable VLM.

    Minden vision provider ebből olvas.
    Modell nevek CSAK itt vannak definiálva – a providerekben nincs hardcode.
    """
    enabled:        bool  = True
    vlm_model_id:   str   = "Qwen/Qwen2-VL-2B-Instruct"
    vlm_model_name: str   = "blaifa/InternVL3_5:4B"
    max_new_tokens: int   = 128
    min_vram_gb:    float = 4.0


# Backwards compatibility alias – cfg.vlm továbbra is működik
VLMConfig = VisionConfig


@dataclass
class PipelineConfig:
    """Pipeline szintű beállítások."""
    supported_formats: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    debug:           bool  = False
    debug_save_stages: bool = False
    log_level:       str   = "INFO"
    log_file:        str   = "logs/translator.log"


# ══════════════════════════════════════════════════════════════════════════════
# Aggregált Config objektum
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """
    Fő konfiguráció – minden alrendszer beállítása egy helyen.

    Használat:
        from config import cfg
        cfg.device.device       # "cuda"
        cfg.rendering.font_size_max  # 42
        cfg.layout.conf_threshold    # 0.25

    Override CLI-ből:
        cfg.pipeline.debug = True
        cfg.rendering.supersample_factor = 4

    Override YAML-ból (opcionális):
        cfg = Config.from_yaml("myconfig.yaml")
    """
    paths:       PathConfig       = field(default_factory=PathConfig)
    device:      DeviceConfig     = field(default_factory=DeviceConfig)
    layout:      LayoutConfig     = field(default_factory=LayoutConfig)
    ocr:         OCRConfig        = field(default_factory=OCRConfig)
    inpainting:  InpaintingConfig = field(default_factory=InpaintingConfig)
    rendering:   RenderingConfig  = field(default_factory=RenderingConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    vision:      VisionConfig     = field(default_factory=VisionConfig)

    @property
    def vlm(self) -> "VisionConfig":
        """Backwards compatibility: cfg.vlm -> cfg.vision"""
        return self.vision
    pipeline:    PipelineConfig   = field(default_factory=PipelineConfig)

    def model_path(self, relative: str) -> Path:
        """Modell útvonal feloldása (relatív a models_dir-hez)."""
        return self.paths.models_dir / relative

    def font_path(self, relative: str) -> Path:
        """Font útvonal feloldása (relatív a BASE_DIR-hez)."""
        return self.paths.base_dir / relative

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """
        YAML fájlból tölt be override értékeket.
        Csak a megadott kulcsokat írja felül – a többi default marad.
        """
        try:
            import yaml
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            cfg = cls()
            for section, values in data.items():
                if hasattr(cfg, section) and isinstance(values, dict):
                    sub = getattr(cfg, section)
                    for k, v in values.items():
                        if hasattr(sub, k):
                            setattr(sub, k, v)
            logger.info(f"Config betöltve YAML-ból: {yaml_path}")
            return cfg
        except ImportError:
            logger.warning("PyYAML nincs telepítve – YAML config nem elérhető")
            return cls()
        except Exception as e:
            logger.warning(f"YAML config hiba ({e}) – default értékek használata")
            return cls()

    def apply_env_overrides(self) -> None:
        """
        Környezeti változók override-ja.

        Type-safe parsing: bool/int/float/path mindegyik helyesen konvertálva.
        Üres string biztonságosan kezelt (nem írja felül a defaultot).

        Támogatott változók:
          COMIC_TRANSLATION_MODEL  – fordítás LLM modell neve
          COMIC_VLM_MODEL          – Qwen2-VL / HF VLM modell ID
          COMIC_LAYOUT_MODEL       – YOLO layout modell neve (fallback)
          COMIC_OCR_REC_MODEL      – OCR felismerő ONNX útvonal
          COMIC_OCR_DET_MODEL      – OCR detektor ONNX útvonal
          OLLAMA_MODEL             – alias: COMIC_TRANSLATION_MODEL
          OLLAMA_BASE_URL          – Ollama API URL
          CUDA_VISIBLE_DEVICES     – GPU device ID
          DEBUG_COMIC              – debug mód bekapcsolása
        """

        _log = logging.getLogger(__name__)

        def _str(key: str) -> str | None:
            """Env string – üres string nem számít."""
            v = os.environ.get(key, "").strip()
            return v if v else None

        def _bool(key: str) -> bool | None:
            """Strict bool: 1/true/yes/on -> True, 0/false/no/off -> False."""
            v = os.environ.get(key, "").strip().lower()
            if v in ("1", "true", "yes", "on"):
                return True
            if v in ("0", "false", "no", "off"):
                return False
            return None

        def _int(key: str) -> int | None:
            """Strict int konverzió – nem int értéket figyelmeztetéssel kihagyja."""
            v = os.environ.get(key, "").strip()
            if not v:
                return None
            try:
                return int(v)
            except ValueError:
                _log.warning(f"Env {key} nem int: '{v}' – kihagyva")
                return None

        def _float(key: str) -> float | None:
            """Strict float konverzió – nem float értéket figyelmeztetéssel kihagyja."""
            v = os.environ.get(key, "").strip()
            if not v:
                return None
            try:
                return float(v)
            except ValueError:
                _log.warning(f"Env {key} nem float: '{v}' – kihagyva")
                return None

        def _path(key: str, must_exist: bool = False) -> Path | None:
            """
            Path konverzió env változóból.

            Args:
                key:        env változó neve
                must_exist: True -> csak létező path fogadható el

            Returns:
                Path objektum vagy None ha üres / nem létező (must_exist=True).
            """
            v = os.environ.get(key, "").strip()
            if not v:
                return None
            p = Path(v)
            if must_exist and not p.exists():
                _log.warning(
                    f"Env {key} path nem létezik: '{v}' – kihagyva")
                return None
            return p

        # ── Fordítás ───────────────────────────────────────────────────────
        if v := _str("COMIC_TRANSLATION_MODEL"):
            self.translation.model = v
        if v := _str("OLLAMA_MODEL"):              # legacy alias
            self.translation.model = v
        if v := _str("OLLAMA_BASE_URL"):
            self.translation.base_url = v
        if v := _float("COMIC_TRANSLATION_TEMPERATURE"):
            self.translation.temperature = v
        if v := _int("COMIC_TRANSLATION_TIMEOUT"):
            self.translation.timeout_sec = v

        # ── Vision / VLM ───────────────────────────────────────────────────
        if v := _str("COMIC_VLM_MODEL"):
            self.vision.vlm_model_id = v
        if v := _bool("COMIC_VLM_ENABLED"):
            self.vision.enabled = v
        if v := _float("COMIC_VLM_MIN_VRAM_GB"):
            self.vision.min_vram_gb = v

        # ── Layout ─────────────────────────────────────────────────────────
        if v := _str("COMIC_LAYOUT_MODEL"):
            self.layout.fallback_model = v
        if v := _path("COMIC_LAYOUT_MODEL_PATH", must_exist=True):
            self.layout.model_path = str(v)
        if v := _float("COMIC_LAYOUT_CONF_THRESHOLD"):
            self.layout.conf_threshold = v

        # ── OCR modellek (string path vagy abszolút path) ──────────────────
        if v := _str("COMIC_OCR_REC_MODEL"):
            self.ocr.rec_model_path = v
        if v := _str("COMIC_OCR_DET_MODEL"):
            self.ocr.det_model_path = v
        if v := _str("COMIC_OCR_DICT"):
            self.ocr.dict_path = v
        if v := _bool("COMIC_OCR_GPU"):
            self.ocr.use_gpu = v
        if v := _float("COMIC_OCR_DET_THRESH"):
            self.ocr.det_thresh = v
        if v := _str("COMIC_OCR_BACKEND"):
            if v in ("auto", "ppocr", "easyocr", "paddleocr", "minicpm_ocr", "gemini_flash", "qwen2_vl"):
                self.ocr.backend = v
            else:
                _log.warning(
                    f"COMIC_OCR_BACKEND ervenytelen: '{v}' "
                    "– kihagyva")

        # Extras
        if v := _str("COMIC_MINICPM_MODEL"):
            self.ocr.minicpm_model_name = v
        if v := _str("COMIC_GEMINI_API_KEY"):
            self.ocr.gemini_api_key = v
        if v := _str("COMIC_GEMINI_MODEL"):
            self.ocr.gemini_model = v

        # ── Inpainting ─────────────────────────────────────────────────────
        if v := _path("COMIC_LAMA_MODEL_PATH", must_exist=True):
            self.inpainting.lama_model_path = str(v)
        if v := _int("COMIC_INPAINT_DILATION"):
            self.inpainting.mask_dilation_px = v

        # ── Rendering ──────────────────────────────────────────────────────
        if v := _int("COMIC_SUPERSAMPLE"):
            self.rendering.supersample_factor = v
        if v := _int("COMIC_FONT_SIZE_MAX"):
            self.rendering.font_size_max = v
        if v := _int("COMIC_FONT_SIZE_MIN"):
            self.rendering.font_size_min = v
        if v := _path("COMIC_FONT_BOLD", must_exist=True):
            self.rendering.font_bold = str(v)
        if v := _path("COMIC_FONT_REGULAR", must_exist=True):
            self.rendering.font_regular = str(v)

        # ── Device ─────────────────────────────────────────────────────────
        if v := _int("CUDA_VISIBLE_DEVICES"):
            self.device.cuda_device_id = v
        if v := _str("COMIC_DEVICE"):
            if v in ("cuda", "cpu"):
                self.device.device = v
            else:
                _log.warning(f"COMIC_DEVICE érvénytelen érték: '{v}' – kihagyva")
        if v := _bool("COMIC_USE_FP16"):
            self.device.use_fp16 = v
        if v := _float("COMIC_VRAM_SAFETY_MARGIN"):
            self.device.vram_safety_margin_gb = v

        # ── VRAM szekvenciális mód ─────────────────────────────────────────
        if v := _bool("COMIC_SEQUENTIAL_MODE"):
            self.device.sequential_mode = v

        # ── Debug ──────────────────────────────────────────────────────────
        if _bool("DEBUG_COMIC") is True:
            self.pipeline.debug             = True
            self.rendering.debug_layout     = True
            self.inpainting.debug_save_masks = True
        if v := _bool("COMIC_SAVE_STAGES"):
            self.pipeline.debug_save_stages = v

    def print_summary(self) -> None:
        """Konfiguráció összefoglaló – induláskor logolva."""
        lines = [
            "=" * 58,
            "  Comic Translator – Konfiguráció",
            "=" * 58,
        ]
        if _cuda_available:
            gpu = torch.cuda.get_device_name(self.device.cuda_device_id)
            vram = torch.cuda.get_device_properties(
                self.device.cuda_device_id).total_memory / 1e9
            free = self.device.vram_free_gb()
            lines += [
                f"  GPU:    {gpu}",
                f"  VRAM:   {vram:.1f} GB total | {free:.1f} GB szabad",
                f"  Mód:    {'Szekvenciális' if self.device.sequential_mode else 'Párhuzamos'}",
            ]
        else:
            lines.append("  ⚠️  CPU mód – CUDA nem elérhető")

        lines += [
            f"  LLM:    {self.translation.model} @ {self.translation.base_url}",
            f"  VLM:    {self.vision.vlm_model_id} ({self.vision.vlm_model_name})",
            f"  OCR:    {self.ocr.backend} (GPU={self.ocr.use_gpu})",
            f"  Inpaint:{self.inpainting.lama_model_path}",
            f"  SS:     {self.rendering.supersample_factor}x supersampling",
            f"  Debug:  {self.pipeline.debug}",
            "=" * 58,
        ]
        for line in lines:
            logger.info(line)


# ══════════════════════════════════════════════════════════════════════════════
# Globális singleton – importálható bárhonnan
# ══════════════════════════════════════════════════════════════════════════════

cfg = Config()
cfg.paths.ensure_dirs()
cfg.apply_env_overrides()

# ── Visszafelé kompatibilitás (régi kód ami config.DEVICE-t használ) ──────────
DEVICE              = cfg.device.device
LAYOUT_MODEL        = cfg.layout.fallback_model
LAYOUT_CONF_THRESHOLD = cfg.layout.conf_threshold
LAYOUT_IOU_THRESHOLD  = cfg.layout.iou_threshold
BUBBLE_CLASS_IDS      = cfg.layout.bubble_class_ids
READING_ORDER_Y_TOLERANCE = cfg.layout.reading_order_y_tolerance
OCR_LANG            = cfg.ocr.source_language
OCR_USE_ANGLE_CLS   = cfg.ocr.use_angle_cls
OCR_USE_GPU         = cfg.ocr.use_gpu
OCR_BATCH_SIZE      = 4
OCR_DET_DB_THRESH   = cfg.ocr.det_thresh
OCR_REC_ALGORITHM   = "SVTR_LCNet"
OLLAMA_BASE_URL     = cfg.translation.base_url
OLLAMA_MODEL        = cfg.translation.model
TRANSLATION_MAX_TOKENS  = cfg.translation.max_tokens
TRANSLATION_TEMPERATURE = cfg.translation.temperature
TRANSLATION_TIMEOUT     = cfg.translation.timeout_sec
VLM_MODEL_ID        = cfg.vision.vlm_model_id
VLM_MAX_NEW_TOKENS  = cfg.vision.max_new_tokens
VLM_ENABLED         = cfg.vision.enabled
INPAINT_MASK_DILATION = cfg.inpainting.mask_dilation_px
INPAINT_FALLBACK    = cfg.inpainting.fallback_to_opencv
FONT_PATH           = str(cfg.font_path(cfg.rendering.font_bold))
FONT_PATH_REGULAR   = str(cfg.font_path(cfg.rendering.font_regular))
FONT_SIZE_DEFAULT   = 20
FONT_SIZE_MIN       = cfg.rendering.font_size_min
FONT_SIZE_MAX       = cfg.rendering.font_size_max
FONT_COLOR          = cfg.rendering.text_color
FONT_OUTLINE_COLOR  = cfg.rendering.outline_color
FONT_OUTLINE_WIDTH  = cfg.rendering.outline_width
TEXT_PADDING        = cfg.rendering.padding_min_px
LINE_SPACING_FACTOR = cfg.rendering.line_spacing
DEBUG               = cfg.pipeline.debug
DEBUG_SAVE_STAGES   = cfg.pipeline.debug_save_stages
LOG_LEVEL           = cfg.pipeline.log_level
LOG_FILE            = cfg.pipeline.log_file
SUPPORTED_FORMATS   = cfg.pipeline.supported_formats
INPUT_DIR           = cfg.paths.input_dir
OUTPUT_DIR          = cfg.paths.output_dir
DEBUG_DIR           = cfg.paths.debug_dir
FONT_DIR            = cfg.paths.font_dir
LOG_DIR             = cfg.paths.log_dir
VRAM_SEQUENTIAL_MODE = cfg.device.sequential_mode
VRAM_SAFETY_MARGIN_GB = cfg.device.vram_safety_margin_gb


def vram_free_gb() -> float:
    return cfg.device.vram_free_gb()

def vram_used_gb() -> float:
    return cfg.device.vram_used_gb()

def cuda_cleanup() -> None:
    cfg.device.cleanup()
