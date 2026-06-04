# -*- coding: utf-8 -*-
"""
doctor.py - Standalone preflight és diagnosztika modul.

Feladatai:
  - GPU / CUDA állapot
  - Model fájlok ellenőrzése
  - Font registry scan
  - I/O útvonalak
  - Ollama API + modell detektálás (valódi API adatok)
  - Betöltött modellek (ollama ps)
  - VRAM becslő
  - AI stack összefoglaló
  - Konfigurált vs telepített modell összehasonlítás

Tervezési elvek:
  - Mindig graceful degrade: nincs crash ha ollama offline, nvidia-smi hiányzik stb.
  - GUI-ból és CLI-ból is hívható (nincs argparse dependencia)
  - Minden ellenőrzés önálló metódus: könnyen bővíthető
  - Rich UI ha elérhető, plain fallback ha nem
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Rich / plain fallback
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    HAS_RICH = True
    _console = Console()
except ImportError:
    HAS_RICH = False
    _console = None


# ══════════════════════════════════════════════════════════════════════════════
# PREFLIGHT ITEM / REPORT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PreflightItem:
    label:  str
    status: str   # "ok" | "warn" | "error" | "info"
    detail: str = ""

    @property
    def icon(self) -> str:
        return {
            "ok":    "[OK]",
            "warn":  "[WARN]",
            "error": "[ERROR]",
            "info":  "[INFO]",
        }.get(self.status, "[?]")


@dataclass
class PreflightReport:
    items:        list = field(default_factory=list)
    fatal:        bool = False
    has_warnings: bool = False

    def add(self, label: str, status: str, detail: str = "") -> None:
        self.items.append(PreflightItem(label, status, detail))
        if status == "error":
            self.fatal = True
        elif status == "warn":
            self.has_warnings = True

    def ok(self,    label: str, detail: str = "") -> None: self.add(label, "ok",    detail)
    def warn(self,  label: str, detail: str = "") -> None: self.add(label, "warn",  detail)
    def error(self, label: str, detail: str = "") -> None: self.add(label, "error", detail)
    def info(self,  label: str, detail: str = "") -> None: self.add(label, "info",  detail)

    def print_report(self) -> None:
        if HAS_RICH:
            self._print_rich()
        else:
            self._print_plain()

    def _print_rich(self) -> None:
        table = Table(
            title="Preflight Report",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Státusz",    width=10)
        table.add_column("Ellenőrzés", width=38)
        table.add_column("Részlet",    width=52)

        color_map = {
            "ok":    "green",
            "warn":  "yellow",
            "error": "red",
            "info":  "cyan",
        }
        for item in self.items:
            c = color_map.get(item.status, "white")
            table.add_row(
                f"[{c}]{item.icon}[/{c}]",
                item.label,
                f"[dim]{item.detail}[/dim]" if item.detail else "",
            )
        _console.print(table)

        if self.fatal:
            _console.print(Panel(
                "[bold red]FATÁLIS HIBA – pipeline nem indítható[/bold red]",
                border_style="red"))
        elif self.has_warnings:
            _console.print(Panel(
                "[yellow]Figyelmeztetések – ellenőrizd![/yellow]",
                border_style="yellow"))
        else:
            _console.print(Panel(
                "[bold green]Minden OK – pipeline készen áll[/bold green]",
                border_style="green"))

    def _print_plain(self) -> None:
        print("\n=== Preflight Report ===")
        for item in self.items:
            d = f" – {item.detail}" if item.detail else ""
            print(f"  {item.icon} {item.label}{d}")
        print()
        if self.fatal:
            print("  [FATAL] Pipeline nem indítható!")
        elif self.has_warnings:
            print("  [WARN] Figyelmeztetések vannak.")
        else:
            print("  [OK] Pipeline készen áll.")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OllamaModel:
    name:    str
    size_gb: float = 0.0
    loaded:  bool  = False
    processor: str = ""


def _ollama_api_tags(base_url: str) -> list[OllamaModel]:
    """
    GET /api/tags – telepített modellek listája.
    Graceful: üres lista ha offline vagy hiba.
    """
    try:
        import requests
        resp = requests.get(f"{base_url}/api/tags", timeout=4)
        resp.raise_for_status()
        data   = resp.json()
        models = []
        for m in data.get("models", []):
            name    = m.get("name", "")
            size_b  = m.get("size", 0)
            size_gb = round(size_b / 1e9, 1) if size_b else 0.0
            if name:
                models.append(OllamaModel(name=name, size_gb=size_gb))
        return models
    except ImportError:
        logger.debug("requests nincs telepítve – ollama API skip")
        return []
    except Exception as e:
        logger.debug(f"ollama /api/tags hiba: {e}")
        return []


def _ollama_api_ps(base_url: str) -> list[OllamaModel]:
    """
    GET /api/ps – futó (GPU-ban betöltött) modellek.
    Graceful: üres lista ha nem támogatott vagy hiba.
    """
    try:
        import requests
        resp = requests.get(f"{base_url}/api/ps", timeout=4)
        resp.raise_for_status()
        data   = resp.json()
        models = []
        for m in data.get("models", []):
            name      = m.get("name", "")
            processor = m.get("details", {}).get("processor", "")
            if not processor:
                # Fallback: size_vram > 0 → GPU
                vram = m.get("size_vram", 0)
                processor = "GPU" if vram > 0 else "CPU"
            if name:
                models.append(OllamaModel(
                    name=name, loaded=True, processor=processor))
        return models
    except ImportError:
        return []
    except Exception as e:
        logger.debug(f"ollama /api/ps hiba: {e}")
        return []


def _ollama_cli_list() -> list[OllamaModel]:
    """
    `ollama list` CLI fallback ha az API nem elérhető.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=6,
        )
        models = []
        for line in result.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                models.append(OllamaModel(name=parts[0]))
        return models
    except Exception as e:
        logger.debug(f"ollama list CLI hiba: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# VRAM BECSLŐ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VRAMEstimate:
    components:   dict[str, float] = field(default_factory=dict)
    total_gb:     float = 0.0
    gpu_total_gb: float = 0.0
    gpu_free_gb:  float = 0.0
    safe:         bool  = True
    suggestion:   str   = ""


def estimate_vram(pcfg=None) -> VRAMEstimate:
    """
    Lightweight VRAM becslő a jelenlegi konfiguráció alapján.

    Közelítő értékek – nem mérnöki pontosság, csak tájékoztató.
    Ollama-n futó modellek NEM számítanak bele (külső process).
    """
    from config import cfg

    skip_vlm    = getattr(pcfg, "skip_vlm",    False) if pcfg else False
    skip_gemma  = getattr(pcfg, "skip_gemma",  False) if pcfg else False
    skip_inpaint= getattr(pcfg, "skip_inpaint",False) if pcfg else False

    components: dict[str, float] = {}

    if not skip_vlm and cfg.vision.enabled:
        # Qwen2-VL-2B float16 ≈ 4.5GB
        components["Qwen2-VL"] = 4.5

    # OCR: ppocr-v5 ONNX GPU
    if cfg.ocr.use_gpu:
        components["OCR (ppocr-v5)"] = 0.8

    # LaMa ONNX
    if not skip_inpaint:
        components["LaMa inpainting"] = 1.2

    # YOLO layout
    components["YOLO layout"] = 0.4

    # Gemma és fordítás: Ollama kezeli, nem direkt VRAM
    if not skip_gemma:
        components["Gemma4 (Ollama – külső)"] = 0.0
    components["Translation (Ollama – külső)"] = 0.0

    total = sum(v for v in components.values() if v > 0)

    gpu_total = 0.0
    gpu_free  = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_free  = torch.cuda.mem_get_info(0)[0] / 1e9
    except Exception:
        pass

    safe       = True
    suggestion = ""
    if gpu_free > 0:
        margin = gpu_free - total
        if margin < 1.5:
            safe = True if margin >= 0 else False
            suggestion = "--skip-vlm kapcsoló vagy QUICK profil javasolt"

    return VRAMEstimate(
        components   = components,
        total_gb     = round(total, 1),
        gpu_total_gb = round(gpu_total, 1),
        gpu_free_gb  = round(gpu_free, 1),
        safe         = safe,
        suggestion   = suggestion,
    )


# ══════════════════════════════════════════════════════════════════════════════
# AI STACK LEÍRÓ
# ══════════════════════════════════════════════════════════════════════════════

def _ai_stack_summary() -> dict[str, str]:
    """Aktív AI komponensek összefoglalója."""
    from config import cfg

    layout_model = cfg.model_path(cfg.layout.model_path)
    layout_name  = (layout_model.name
                    if layout_model.exists()
                    else f"{cfg.layout.fallback_model} (fallback)")

    lama_model  = cfg.model_path(cfg.inpainting.lama_model_path)
    lama_name   = lama_model.name if lama_model.exists() else "HIÁNYZIK"

    return {
        "Translation": cfg.translation.model,
        "Gemma":       cfg.vision.gemma_model,
        "Vision (VLM)": cfg.vision.vlm_model_id,
        "OCR":         "PP-OCRv5 ONNX",
        "Layout":      layout_name,
        "Inpainting":  lama_name,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PREFLIGHT CHECKER
# ══════════════════════════════════════════════════════════════════════════════

class PreflightChecker:
    """
    Standalone preflight ellenőrző – GUI-ból és CLI-ból is hívható.

    Nincs argparse vagy CLI dependencia.
    Minden ellenőrzés önálló metódus: könnyen bővíthető / cserélhető.
    """

    def __init__(self, pcfg=None) -> None:
        self.pcfg = pcfg

    def run(self) -> PreflightReport:
        """Teljes preflight futtatása."""
        report = PreflightReport()

        if HAS_RICH:
            _console.rule("[bold cyan]Comic Translator – Preflight[/bold cyan]")
        else:
            print("\n" + "=" * 54)
            print("  Preflight ellenőrzés")
            print("=" * 54)

        self._check_gpu(report)
        self._check_models(report)
        self._check_fonts(report)
        self._check_paths(report)
        self._check_ollama_full(report)
        self._check_vram(report)
        self._check_ai_stack(report)
        self._check_config(report)
        return report

    # ── GPU ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_gpu(report: PreflightReport) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                name  = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                free  = torch.cuda.mem_get_info(0)[0] / 1e9
                report.ok(
                    "CUDA elérhető",
                    f"{name} | {total:.1f}GB total | {free:.1f}GB szabad")
                if free < 4.0:
                    report.warn(
                        "Kevés szabad VRAM",
                        f"{free:.1f}GB < 4GB – használj --skip-vlm")
            else:
                report.warn("CUDA nem elérhető", "CPU módban fog futni (lassú!)")
        except ImportError:
            report.error("PyTorch nincs telepítve", "pip install torch")

    # ── Model fájlok ──────────────────────────────────────────────────────────

    @staticmethod
    def _check_models(report: PreflightReport) -> None:
        from config import cfg
        checks = [
            ("Layout (YOLO)",     cfg.model_path(cfg.layout.model_path)),
            ("Inpainting (LaMa)", cfg.model_path(cfg.inpainting.lama_model_path)),
            ("OCR Detektor",      cfg.model_path(cfg.ocr.det_model_path)),
            ("OCR Felismerő",     cfg.model_path(cfg.ocr.rec_model_path)),
            ("OCR Szótár",        cfg.model_path(cfg.ocr.dict_path)),
        ]
        for label, path in checks:
            if path.exists():
                sz = path.stat().st_size / 1e6
                report.ok(label, f"{path.name} ({sz:.0f}MB)")
            else:
                report.warn(label, f"Nem található: {path}")

    # ── Fontok ────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_fonts(report: PreflightReport) -> None:
        from config import cfg
        fonts = (list(cfg.paths.font_dir.rglob("*.ttf")) +
                 list(cfg.paths.font_dir.rglob("*.otf")))
        if fonts:
            sample = ", ".join(f.name for f in fonts[:3])
            report.ok("Fontok", f"{len(fonts)} db: {sample}")
        else:
            report.warn("Nincs font",
                        f"Másold TTF fájlokat: {cfg.paths.font_dir}")

    # ── I/O útvonalak ─────────────────────────────────────────────────────────

    def _check_paths(self, report: PreflightReport) -> None:
        from config import cfg
        input_dir  = (self.pcfg.input_dir  if self.pcfg
                      else cfg.paths.input_dir)
        output_dir = (self.pcfg.output_dir if self.pcfg
                      else cfg.paths.output_dir)

        if input_dir.exists():
            imgs = []
            for ext in cfg.pipeline.supported_formats:
                imgs += list(input_dir.glob(f"*{ext}"))
                imgs += list(input_dir.glob(f"*{ext.upper()}"))
            if imgs:
                report.ok("Input mappa", f"{input_dir} ({len(imgs)} kép)")
            else:
                report.warn("Input mappa üres", str(input_dir))
        else:
            report.error("Input mappa nem létezik", str(input_dir))

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            test = output_dir / ".write_test"
            test.touch()
            test.unlink()
            report.ok("Output mappa", str(output_dir))
        except OSError as e:
            report.error("Output mappa nem írható", str(e))

    # ── Ollama – teljes ellenőrzés ────────────────────────────────────────────

    def _check_ollama_full(self, report: PreflightReport) -> None:
        """
        Valódi Ollama API adatok:
          1. API elérhetőség
          2. Telepített modellek listája + méret
          3. Konfigurált modell vs telepített összehasonlítás
          4. Futó (GPU-ban betöltött) modellek (ollama ps)
        """
        from config import cfg
        base_url     = cfg.translation.base_url
        cfg_model    = (self.pcfg.effective_model()
                        if self.pcfg else cfg.translation.model)

        # API elérhetőség
        api_online = False
        try:
            import requests
            resp = requests.get(f"{base_url}/api/tags", timeout=4)
            if resp.status_code == 200:
                api_online = True
                report.ok("Ollama API", base_url)
            else:
                report.warn("Ollama API", f"HTTP {resp.status_code}")
        except ImportError:
            report.warn("Ollama API", "requests nincs telepítve")
        except Exception:
            report.warn("Ollama offline",
                        f"{base_url} – indítsd: ollama serve")

        # Telepített modellek
        installed: list[OllamaModel] = []
        if api_online:
            installed = _ollama_api_tags(base_url)
        if not installed:
            installed = _ollama_cli_list()

        if installed:
            for m in installed[:8]:   # max 8 db a report-ban
                sz = f" ({m.size_gb}GB)" if m.size_gb > 0 else ""
                report.ok(f"  Telepítve: {m.name}", f"{sz}")
        else:
            report.warn("Telepített Ollama modellek", "Nem találhatók")

        # Konfigurált modell vs telepített
        if installed:
            installed_names = [m.name for m in installed]
            cfg_base        = cfg_model.split(":")[0]
            found = any(
                cfg_base in name or name == cfg_model
                for name in installed_names
            )
            if found:
                report.ok(
                    "Konfigurált modell telepítve",
                    cfg_model)
            else:
                avail = ", ".join(m.name for m in installed[:4])
                report.warn(
                    "Konfigurált modell hiányzik",
                    f"Konfigurált: {cfg_model} | Telepítve: {avail}")

        # Futó modellek (ollama ps)
        if api_online:
            loaded = _ollama_api_ps(base_url)
            if loaded:
                for m in loaded:
                    report.info(
                        f"  Betöltve (GPU): {m.name}",
                        f"processor={m.processor}")
            else:
                report.info("Betöltött modellek", "Nincs aktívan betöltött modell")

    # ── VRAM becslő ───────────────────────────────────────────────────────────

    def _check_vram(self, report: PreflightReport) -> None:
        est = estimate_vram(self.pcfg)

        report.info(
            "VRAM becslő",
            f"GPU: {est.gpu_total_gb}GB total | {est.gpu_free_gb}GB szabad")

        for component, gb in est.components.items():
            if gb > 0:
                report.info(f"  {component}", f"~{gb}GB")

        if est.total_gb > 0:
            if est.safe:
                report.ok(
                    "Becsült VRAM csúcs",
                    f"~{est.total_gb}GB – BIZTONSÁGOS")
            else:
                report.warn(
                    "Becsült VRAM csúcs",
                    f"~{est.total_gb}GB – {est.suggestion}")

    # ── AI Stack összefoglaló ─────────────────────────────────────────────────

    @staticmethod
    def _check_ai_stack(report: PreflightReport) -> None:
        stack = _ai_stack_summary()
        report.info("── AI Stack ──", "")
        for component, model in stack.items():
            report.info(f"  {component}", model)

    # ── Config sanity ─────────────────────────────────────────────────────────

    @staticmethod
    def _check_config(report: PreflightReport) -> None:
        from config import cfg
        report.info(
            "Konfig",
            f"device={cfg.device.device} | "
            f"SS={cfg.rendering.supersample_factor}x | "
            f"LLM={cfg.translation.model} | "
            f"Gemma={cfg.vision.gemma_model}")
