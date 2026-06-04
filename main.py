"""
main.py - Comic Translator CLI belepes pont.

Szandekosan vekony: csak argparse + config override + orchestrator inditas.
Nincs pipeline logika itt.

Exit kodok:
  0 = siker
  1 = batch warning (nehany oldal hibas, de folytatta)
  2 = fatalis konfig/provider hiba (nem indult el)
  3 = megszakitva (Ctrl+C)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.table import Table
    HAS_RICH = True
    _console = Console()
except ImportError:
    HAS_RICH = False
    _console = None

# PreflightChecker és PreflightReport a doctor.py-ból jön
from doctor import PreflightChecker, PreflightReport

logger = logging.getLogger(__name__)


# ── Logging bootstrap ─────────────────────────────────────────────────────────

def setup_logging(
    verbose:  bool = False,
    quiet:    bool = False,
    log_file: Optional[str] = None,
) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if HAS_RICH and not quiet:
        ch = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
            level=level,
        )
        root.addHandler(ch)
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError as e:
            logger.warning(f"Log file nem irható: {e}")

    for noisy in ("urllib3", "httpx", "PIL", "transformers",
                  "ultralytics", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Emit helper ───────────────────────────────────────────────────────────────

def _emit(msg: str) -> None:
    if HAS_RICH:
        _console.log(msg)
    else:
        print(msg)


def print_banner() -> None:
    if HAS_RICH:
        _console.rule()
        _console.print(Panel(
            "[bold cyan]Comic Translator[/bold cyan]  "
            "[dim]angol -> magyar | AI pipeline[/dim]",
            border_style="cyan", expand=False))
        _console.rule()
    else:
        print()
        print("=" * 54)
        print("  Comic Translator  |  angol -> magyar  |  AI pipeline")
        print("=" * 54)
        print()


# ── Preflight ─────────────────────────────────────────────────────────────────

# ── Argparse ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comic-translator",
        description="Kepregeny-fordito: angol -> magyar (AI pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Peldak:
  python main.py --input-dir input/
  python main.py --input-page input/page01.jpg --debug
  python main.py --dry-run --skip-vlm
  python main.py --doctor
  python main.py --resume --input-dir input/
  python main.py --rerender-bubble page_001 3
        """,
    )

    io = parser.add_argument_group("I/O")
    io.add_argument("--input-dir",  type=Path, default=None, metavar="DIR")
    io.add_argument("--input-page", type=Path, default=None, metavar="FILE")
    io.add_argument("--output-dir", type=Path, default=None, metavar="DIR")

    pipe = parser.add_argument_group("Pipeline")
    pipe.add_argument("--skip-vlm",     action="store_true")
    pipe.add_argument("--skip-inpaint", action="store_true")
    pipe.add_argument("--skip-gemma",   action="store_true")
    pipe.add_argument("--dry-run",      action="store_true")
    pipe.add_argument("--force",        action="store_true",
                      help="Minden stage ujrafut")

    resume = parser.add_argument_group("Resume")
    resume.add_argument("--resume",       action="store_true")
    resume.add_argument("--no-overwrite", action="store_true")

    prov = parser.add_argument_group("Provider")
    prov.add_argument("--translation-provider", default="ollama",
                      choices=["ollama", "openai", "deepl"])
    prov.add_argument("--translation-model",    default="", metavar="MODEL")
    prov.add_argument("--no-warmup", action="store_true")

    thresh = parser.add_argument_group("Thresholds")
    thresh.add_argument("--ocr-threshold",    type=float, default=0.70)
    thresh.add_argument("--refine-threshold", type=float, default=0.75)
    thresh.add_argument("--ocr-backend",
                        default="",
                        choices=["auto", "ppocr", "easyocr", ""],
                        metavar="BACKEND",
                        help="OCR backend: auto|ppocr|easyocr (default: config)")

    dev = parser.add_argument_group("Device")
    dev.add_argument("--device", default="",
                     choices=["cuda", "cpu", ""])
    dev.add_argument("--supersample", type=int, default=0,
                     choices=[0, 1, 2, 4], metavar="N")

    batch = parser.add_argument_group("Batch")
    batch.add_argument("--pause-after", type=int, default=0, metavar="N")

    dbg = parser.add_argument_group("Debug")
    dbg.add_argument("--debug",       action="store_true")
    dbg.add_argument("--save-stages", action="store_true")
    dbg.add_argument("--verbose", "-v", action="store_true")
    dbg.add_argument("--quiet",   "-q", action="store_true")
    dbg.add_argument("--log-file", default="", metavar="FILE")

    spec = parser.add_argument_group("Specialis modok")
    spec.add_argument("--doctor", action="store_true",
                      help="Diagnosztika (GPU, modellek, providerek)")
    spec.add_argument("--rerender-bubble", nargs=2,
                      metavar=("PAGE_ID", "BUBBLE_ID"),
                      help="Egyetlen buborek ujrarenderelése")

    return parser


# ── Config override ───────────────────────────────────────────────────────────

def build_pipeline_config(args: argparse.Namespace):
    from config import cfg
    from orchestrator import PipelineConfig
    return PipelineConfig(
        input_dir  = args.input_dir  or cfg.paths.input_dir,
        output_dir = args.output_dir or cfg.paths.output_dir,
        skip_vlm      = args.skip_vlm,
        skip_inpaint  = args.skip_inpaint,
        skip_gemma    = args.skip_gemma,
        dry_run       = args.dry_run,
        force         = args.force,
        resume        = args.resume,
        no_overwrite  = args.no_overwrite,
        debug         = args.debug or args.verbose,
        save_stages   = args.save_stages,
        pause_after   = args.pause_after,
        translation_provider = args.translation_provider,
        translation_model    = args.translation_model,
        ocr_correction_threshold     = args.ocr_threshold,
        translation_refine_threshold = args.refine_threshold,
        ocr_backend                  = args.ocr_backend or "",
        device      = args.device,
        supersample = args.supersample,
        warmup_lightweight = not args.no_warmup,
    )


# ── Execution modes ───────────────────────────────────────────────────────────

def run_doctor(args: argparse.Namespace) -> int:
    checker = PreflightChecker(pcfg=None)
    report  = checker.run()
    report.print_report()
    return 2 if report.fatal else (1 if report.has_warnings else 0)


def run_rerender(args: argparse.Namespace) -> int:
    page_id, bubble_id_str = args.rerender_bubble
    try:
        bubble_id = int(bubble_id_str)
    except ValueError:
        logger.error(f"BUBBLE_ID egesz szam kell: {bubble_id_str}")
        return 2
    pcfg     = build_pipeline_config(args)
    page_dir = pcfg.output_dir / page_id
    if not page_dir.exists():
        logger.error(f"Page mappa nem letezik: {page_dir}")
        return 2
    from orchestrator import PipelineOrchestrator
    orch = PipelineOrchestrator(pcfg)
    ok   = orch.rerender_bubble(page_dir, bubble_id)
    return 0 if ok else 1


def run_single(args: argparse.Namespace) -> int:
    img_path = args.input_page.resolve()
    if not img_path.exists():
        logger.error(f"Kep nem letezik: {img_path}")
        return 2
    pcfg    = build_pipeline_config(args)
    checker = PreflightChecker(pcfg)
    report  = checker.run()
    report.print_report()
    if report.fatal:
        return 2
    _emit(f"[READY] Egyetlen kep: {img_path.name}")
    from orchestrator import PipelineOrchestrator
    orch  = PipelineOrchestrator(pcfg)
    entry = orch.run_single(img_path)
    if entry.status == "failed":
        logger.error(f"Feldolgozas sikertelen: {entry.error}")
        return 1
    return 0


def run_batch(args: argparse.Namespace) -> int:
    from config import cfg
    pcfg      = build_pipeline_config(args)
    input_dir = pcfg.input_dir
    if not input_dir.exists():
        logger.error(f"Input mappa nem letezik: {input_dir}")
        return 2
    checker = PreflightChecker(pcfg)
    report  = checker.run()
    report.print_report()
    if report.fatal:
        return 2
    images: list[Path] = []
    for ext in cfg.pipeline.supported_formats:
        images += sorted(input_dir.glob(f"*{ext}"))
        images += sorted(input_dir.glob(f"*{ext.upper()}"))
    seen: set = set()
    unique: list[Path] = []
    for p in images:
        if p.name.lower() not in seen:
            seen.add(p.name.lower())
            unique.append(p)
    unique.sort(key=lambda p: p.name)
    if not unique:
        logger.warning(f"Nincs feldolgozhato kep: {input_dir}")
        return 1
    _emit(f"[READY] Batch: {len(unique)} kep | {input_dir}")
    from orchestrator import PipelineOrchestrator
    orch     = PipelineOrchestrator(pcfg)
    manifest = orch.run_batch(unique)
    if manifest.failed_count == 0:
        return 0
    elif manifest.success_count > 0:
        return 1
    else:
        return 2


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    setup_logging(
        verbose  = args.verbose,
        quiet    = args.quiet,
        log_file = args.log_file or "logs/translator.log",
    )
    print_banner()

    if args.doctor:
        return run_doctor(args)
    if args.rerender_bubble:
        return run_rerender(args)

    if not args.input_page and not args.input_dir:
        from config import cfg
        args.input_dir = cfg.paths.input_dir
        logger.info(f"--input-dir default: {args.input_dir}")

    if args.input_page and args.input_dir:
        logger.error("--input-page es --input-dir egyidejuleg nem adhato meg")
        return 2

    t_start = time.perf_counter()
    try:
        if args.input_page:
            exit_code = run_single(args)
        else:
            exit_code = run_batch(args)
    except KeyboardInterrupt:
        print()
        logger.warning("Megszakitva (Ctrl+C)")
        exit_code = 3
    except Exception as e:
        logger.critical(f"Varatlan hiba: {e}", exc_info=True)
        exit_code = 2
    finally:
        elapsed = time.perf_counter() - t_start
        logger.info(f"Teljes futasi ido: {elapsed:.1f}s")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
