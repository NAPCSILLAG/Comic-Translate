#!/usr/bin/env bash
# =============================================================================
# start.sh — Comic Translator launcher
# Linux / CachyOS optimized
# =============================================================================

set -euo pipefail

# =============================================================================
# UTF-8 / locale
# =============================================================================

export PYTHONIOENCODING=utf-8
export LANG="${LANG:-en_US.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

# =============================================================================
# Ctrl+C handling
# =============================================================================

trap 'echo -e "\n${YELLOW}[STOP] Megszakítva (Ctrl+C).${NC}"; exit 3' INT

# =============================================================================
# Alap konfiguráció
# =============================================================================

INPUT_DIR="./input"
OUTPUT_DIR="./output"

PROFILE="quality"

# Model override – environment felülírja, autodetect tölti ki ha üres
# Példa: MODEL=qwen2.5:7b ./start.sh
MODEL="${MODEL:-}"

MODEL_SOURCE="autodetect"

# Extra CLI argumentumok (biztonságos array)
EXTRA_ARGS=()

# =============================================================================
# Ollama env
# =============================================================================

export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}"

# =============================================================================
# CUDA / GPU env
# =============================================================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512,garbage_collection_threshold:0.8}"

# =============================================================================
# Színek
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# =============================================================================
# Runtime globals
# =============================================================================

PYTHON=""
VENV_ACTIVE=0

# =============================================================================
# Banner
# =============================================================================

print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "  +----------------------------------------------------------+"
    echo "  |  Comic Translator                                        |"
    echo "  |  AI localization pipeline                                |"
    echo "  |  Linux / CachyOS optimized                               |"
    echo "  +----------------------------------------------------------+"
    echo -e "${NC}"
}

# =============================================================================
# Virtual environment
# =============================================================================

detect_active_venv() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        VENV_ACTIVE=1
        PYTHON="python"
        echo -e "${GREEN}[OK]${NC} Aktív virtual environment:"
        echo -e "     ${DIM}${VIRTUAL_ENV}${NC}"
        return 0
    fi
    return 1
}

activate_project_venv() {
    local candidates=(
        ".venv/bin/activate"
        "venv/bin/activate"
    )

    local found=0

    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            echo -e "${CYAN}[INFO]${NC} Projekt virtual environment aktiválása..."
            echo -e "       ${DIM}${candidate}${NC}"

            # shellcheck disable=SC1090
            source "${candidate}"

            PYTHON="python"
            VENV_ACTIVE=1
            found=1
            break
        fi
    done

    if [[ "${found}" -eq 0 ]]; then
        echo -e "${RED}[ERROR]${NC} Projekt virtual environment nem található."
        echo ""
        echo "Hozz létre egyet:"
        echo ""
        echo "  python3 -m venv .venv"
        echo "  vagy"
        echo "  uv venv"
        echo ""
        echo "Aktiválás Fish shellben:"
        echo ""
        echo "  source .venv/bin/activate.fish"
        echo ""
        exit 2
    fi
}

ensure_virtual_environment() {
    if detect_active_venv; then
        return
    fi
    activate_project_venv
}

# =============================================================================
# Python ellenőrzés
# =============================================================================

detect_python() {
    if command -v python >/dev/null 2>&1; then
        PYTHON="python"
        return 0
    fi
    echo -e "${RED}[ERROR]${NC} Python nem található a virtual environmentben."
    exit 2
}

check_python_version() {
    echo -e "${CYAN}[INFO]${NC} Python verzió ellenőrzése..."

    if ! "${PYTHON}" -c 'import sys; exit(0 if sys.version_info >= (3,10) else 1)'; then
        echo -e "${RED}[ERROR]${NC} Python 3.10+ szükséges."
        exit 2
    fi

    local pyver
    pyver=$("${PYTHON}" -c 'import sys; print(sys.version.split()[0])')
    echo -e "${GREEN}[OK]${NC} Python ${pyver}"
}

# =============================================================================
# Ollama model autodetection
# =============================================================================

# Preferált fordítási modellek prioritás szerint
PREFERRED_MODELS=(
    "qwen2.5:7b"
    "qwen2.5:14b"
    "qwen2.5:32b"
    "gemma4:e4b-it-q8_0"
)

detect_ollama_model() {
    # Próbáljuk az API-t először
    local api_output=""
    if command -v curl >/dev/null 2>&1; then
        api_output=$(curl -s --max-time 3 \
            "http://localhost:11434/api/tags" 2>/dev/null || true)
    fi

    local installed_models=()

    if [[ -n "${api_output}" ]]; then
        # JSON-ból kinyerjük a modell neveket (python segítségével ha van)
        if command -v python >/dev/null 2>&1; then
            mapfile -t installed_models < <(
                echo "${api_output}" | \
                python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for m in data.get('models', []):
        print(m.get('name', ''))
except Exception:
    pass
" 2>/dev/null || true)
        fi
    fi

    # Fallback: ollama list CLI
    if [[ ${#installed_models[@]} -eq 0 ]] && command -v ollama >/dev/null 2>&1; then
        mapfile -t installed_models < <(
            ollama list 2>/dev/null | \
            awk 'NR>1 {print $1}' | \
            grep -v '^$' || true)
    fi

    if [[ ${#installed_models[@]} -eq 0 ]]; then
        echo ""
        return
    fi

    # Preferált modell keresése prioritás szerint
    for preferred in "${PREFERRED_MODELS[@]}"; do
        for installed in "${installed_models[@]}"; do
            if [[ "${installed}" == "${preferred}" ]]; then
                echo "${preferred}"
                return
            fi
        done
    done

    # Ha egyik preferred sem volt meg: első telepített modell
    echo "${installed_models[0]}"
}

resolve_model() {
    # Environment override esetén ne írjuk felül
    if [[ -n "${MODEL}" ]]; then
        MODEL_SOURCE="environment override"
        return
    fi

    echo -e "${CYAN}[INFO]${NC} Fordítási modell automatikus keresése..."

    local detected
    detected=$(detect_ollama_model)

    if [[ -n "${detected}" ]]; then
        MODEL="${detected}"
        MODEL_SOURCE="autodetect"
        echo -e "${GREEN}[OK]${NC} Fordítási modell:"
        echo -e "     ${CYAN}${MODEL}${NC}"
        echo -e "     ${DIM}forrás: ${MODEL_SOURCE}${NC}"
    else
        MODEL=""
        MODEL_SOURCE="none"
        echo -e "${YELLOW}[WARN]${NC} Ollama modell nem található."
        echo -e "       ${DIM}Fordítás a config.py alapértelmezettjét használja.${NC}"
    fi
}

# =============================================================================
# Előfeltételek
# =============================================================================

check_prerequisites() {
    echo -e "${CYAN}[INFO]${NC} Környezet ellenőrzése..."

    if [[ ! -f "main.py" ]]; then
        echo -e "${RED}[ERROR]${NC} main.py nem található."
        echo "Futtasd a projekt gyökérkönyvtárából."
        exit 2
    fi

    if [[ ! -d "${INPUT_DIR}" ]]; then
        echo -e "${YELLOW}[WARN]${NC} Input mappa nem létezik: ${INPUT_DIR}"
        echo -e "${CYAN}[INFO]${NC} Létrehozás..."
        mkdir -p "${INPUT_DIR}"
    fi

    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "logs"

    touch "logs/.write_test"
    rm -f "logs/.write_test"

    echo -e "${GREEN}[OK]${NC} Könyvtárak rendben."
}

# =============================================================================
# GPU info
# =============================================================================

print_gpu_info() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${CYAN}[INFO]${NC} GPU:"
        nvidia-smi --query-gpu=name,memory.total,driver_version \
            --format=csv,noheader \
            | sed 's/^/       /' || true
    fi
}

# =============================================================================
# Futási konfiguráció
# =============================================================================

print_run_info() {
    local profile="$1"
    local extra="$2"

    echo ""
    echo -e "${BOLD}Indítási konfiguráció${NC}"
    echo "------------------------------------------------------------"
    echo -e "${DIM}Profil:${NC}        ${CYAN}${profile}${NC}"
    echo -e "${DIM}Input:${NC}         ${INPUT_DIR}"
    echo -e "${DIM}Output:${NC}        ${OUTPUT_DIR}"

    if [[ -n "${MODEL}" ]]; then
        echo -e "${DIM}Modell:${NC}        ${MODEL}"
        echo -e "${DIM}Forrás:${NC}        ${MODEL_SOURCE}"
    else
        echo -e "${DIM}Modell:${NC}        ${DIM}(config/provider auto)${NC}"
    fi

    if [[ -n "${extra}" ]]; then
        echo -e "${DIM}Extra:${NC}         ${extra}"
    fi

    echo -e "${DIM}OLLAMA_MAX_LOADED_MODELS:${NC} ${OLLAMA_MAX_LOADED_MODELS}"
    echo -e "${DIM}OLLAMA_NUM_PARALLEL:${NC}      ${OLLAMA_NUM_PARALLEL}"
    echo ""
}

# =============================================================================
# Main.py launcher
# =============================================================================

run_main() {
    local -a args=("$@")

    if [[ -n "${MODEL}" ]]; then
        args+=("--translation-model" "${MODEL}")
    fi

    set +e
    "${PYTHON}" main.py "${args[@]}"
    local code=$?
    set -e

    return "${code}"
}

# =============================================================================
# Exit code kezelés
# =============================================================================

# Returns:
#   0 = folytatás / menu reload
#   3 = teljes leállás
handle_exit() {
    local code="$1"
    local interactive="${2:-0}"

    case "${code}" in
        0)
            echo ""
            echo -e "${GREEN}${BOLD}[OK] Befejezve.${NC}"
            ;;
        1)
            echo ""
            echo -e "${YELLOW}${BOLD}[WARN] Befejezve figyelmeztetésekkel.${NC}"
            echo -e "${YELLOW}Néhány oldal hibás lehet – ellenőrizd az outputot.${NC}"
            ;;
        2)
            echo ""
            echo -e "${RED}${BOLD}[ERROR] Fatális pipeline hiba.${NC}"
            echo ""
            echo "Futtasd:"
            echo "  ./start.sh doctor"
            echo ""
            # Interaktív módban NEM állunk le – visszamegyünk a menübe
            if [[ "${interactive}" -eq 0 ]]; then
                exit 2
            fi
            ;;
        3)
            # Ctrl+C / megszakítás: MINDIG teljes leállás
            echo ""
            echo -e "${YELLOW}[STOP] Megszakítva.${NC}"
            exit 3
            ;;
        *)
            echo ""
            echo -e "${RED}[ERROR] Ismeretlen kilépési kód: ${code}${NC}"
            if [[ "${interactive}" -eq 0 ]]; then
                exit "${code}"
            fi
            ;;
    esac
}

# =============================================================================
# OCR Backend választás
# =============================================================================
select_ocr_backend() {
    local backend=""
    while true; do
        echo ""
        echo -e "${BOLD}OCR Backend:${NC}"
        echo ""
        echo -e "  ${CYAN}[1]${NC} PPOCRv5"
        echo -e "  ${CYAN}[2]${NC} EasyOCR"
        echo -e "  ${CYAN}[3]${NC} PaddleOCR"
        echo -e "  ${CYAN}[4]${NC} MiniCPM OCR (Ollama)"
        echo -e "  ${CYAN}[5]${NC} Gemini Flash Cloud"
        echo -e "  ${CYAN}[6]${NC} Qwen2-VL Legacy"
        echo ""
        echo -ne "${BOLD}Választás [1-6]: ${NC}"
        local choice
        read -r choice
        case "${choice}" in
            1) backend="ppocr" ;;
            2) backend="easyocr" ;;
            3) backend="paddleocr" ;;
            4) backend="minicpm_ocr" ;;
            5) backend="gemini_flash" ;;
            6) backend="qwen2_vl" ;;
            *) echo -e "${RED}[ERROR] Érvénytelen választás: ${choice}${NC}"; continue ;;
        esac
        export COMIC_OCR_BACKEND="${backend}"
        break
    done
    echo -e "${GREEN}[OK]${NC} OCR backend: ${CYAN}${backend}${NC}"
}

# =============================================================================
# Profilok és futtatás
# =============================================================================

run_quality() {
    local interactive="${1:-0}"
    print_run_info "QUALITY" "teljes pipeline | max minőség"
    echo -e "${DIM}OCR Backend:${NC} ${CYAN}${COMIC_OCR_BACKEND:-ppocr}${NC}"
    local code
    run_main \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        "${EXTRA_ARGS[@]}"
    code=$?

    handle_exit "${code}" "${interactive}"
}

run_quick() {
    local interactive="${1:-0}"
    print_run_info "QUICK" "--skip-vlm --skip-gemma"
    echo -e "${DIM}OCR Backend:${NC} ${CYAN}${COMIC_OCR_BACKEND:-ppocr}${NC}"
    local code
    run_main \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --skip-vlm \
        --skip-gemma \
        "${EXTRA_ARGS[@]}"
    code=$?

    handle_exit "${code}" "${interactive}"
}

run_debug() {
    local interactive="${1:-0}"
    print_run_info "DEBUG" "--debug --save-stages --verbose"
    echo -e "${DIM}OCR Backend:${NC} ${CYAN}${COMIC_OCR_BACKEND:-ppocr}${NC}"
    local code
    run_main \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --debug \
        --save-stages \
        --verbose \
        "${EXTRA_ARGS[@]}"
    code=$?

    handle_exit "${code}" "${interactive}"
}

run_dry() {
    local interactive="${1:-0}"
    print_run_info "DRY-RUN" "--dry-run"
    echo -e "${DIM}OCR Backend:${NC} ${CYAN}${COMIC_OCR_BACKEND:-ppocr}${NC}"
    local code
    run_main \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --dry-run \
        "${EXTRA_ARGS[@]}"
    code=$?

    handle_exit "${code}" "${interactive}"
}

run_doctor() {
    local interactive="${1:-0}"

    echo ""
    echo -e "${CYAN}${BOLD}[DOCTOR] Diagnosztika indítása...${NC}"
    echo ""

    local code
    run_main --doctor
    code=$?

    case "${code}" in
        0) echo -e "\n${GREEN}[OK] Minden rendben.${NC}" ;;
        1) echo -e "\n${YELLOW}[WARN] Figyelmeztetések találhatók.${NC}" ;;
        2) echo -e "\n${RED}[ERROR] Kritikus problémák találhatók.${NC}" ;;
        3)
            echo -e "\n${YELLOW}[STOP] Megszakítva.${NC}"
            exit 3
            ;;
    esac
    # Doctor után mindig visszamegyünk a menübe ha interaktív
}

# =============================================================================
# OCR Backend választás
# =============================================================================
select_ocr_backend() {
    local backend=""
    while true; do
        echo ""
        echo -e "${BOLD}OCR Backend:${NC}"
        echo ""
        echo -e "  ${CYAN}[1]${NC} PPOCRv5"
        echo -e "  ${CYAN}[2]${NC} EasyOCR"
        echo -e "  ${CYAN}[3]${NC} PaddleOCR"
        echo -e "  ${CYAN}[4]${NC} MiniCPM OCR (Ollama)"
        echo -e "  ${CYAN}[5]${NC} Gemini Flash Cloud"
        echo -e "  ${CYAN}[6]${NC} Qwen2-VL Legacy"
        echo ""
        echo -ne "${BOLD}Választás [1-6]: ${NC}"
        local choice
        read -r choice
        case "${choice}" in
            1) backend="ppocr" ;;
            2) backend="easyocr" ;;
            3) backend="paddleocr" ;;
            4) backend="minicpm_ocr" ;;
            5) backend="gemini_flash" ;;
            6) backend="qwen2_vl" ;;
            *) echo -e "${RED}[ERROR] Érvénytelen választás: ${choice}${NC}"; continue ;;
        esac
        export COMIC_OCR_BACKEND="${backend}"
        break
    done
    echo -e "${GREEN}[OK]${NC} OCR backend: ${CYAN}${backend}${NC}"
}

# =============================================================================
# Interaktív menü – persistent loop
# =============================================================================

show_menu() {
    # OCR labellek összeállítása
    OCR_LABEL_PPOCR="PPOCRv5"
    OCR_LABEL_MINICPM=$("${PYTHON}" -c "
import sys
sys.path.insert(0, '.')
from config import cfg
print(cfg.ocr.minicpm_model_name or 'nem konfigurált')
" 2>/dev/null || echo "nem konfigurált")
    OCR_LABEL_GEMINI=$("${PYTHON}" -c "
import sys
sys.path.insert(0, '.')
from config import cfg
print(cfg.ocr.gemini_model or 'nem konfigurált')
" 2>/dev/null || echo "nem konfigurált")
    OCR_LABEL_VLM=$("${PYTHON}" -c "
import sys
sys.path.insert(0, '.')
from config import cfg
print(cfg.vision.vlm_model_name or 'nem konfigurált')
" 2>/dev/null || echo "nem konfigurált")

    # Persistent menu loop – csak EXIT vagy Ctrl+C löki ki
    while true; do
        echo ""
        echo -e "${BOLD}OCR mód:${NC}"
        echo ""
        echo -e "  ${CYAN}[1]${NC} Hagyományos OCR      (${OCR_LABEL_PPOCR})"
        echo -e "  ${CYAN}[2]${NC} Lokális AI OCR       (${OCR_LABEL_MINICPM})"
        echo -e "  ${CYAN}[3]${NC} Felhős AI            (${OCR_LABEL_GEMINI})"
        echo -e "  ${CYAN}[4]${NC} Lokális VLM (Legacy) (${OCR_LABEL_VLM})"
        echo ""
        echo -e "${BOLD}Egyéb:${NC}"
        echo ""
        echo -e "  ${CYAN}[5]${NC} DEBUG      debug + stage mentés + verbose"
        echo -e "  ${CYAN}[6]${NC} DRY-RUN    OCR + fordítás | render nélkül"
        echo -e "  ${CYAN}[7]${NC} DOCTOR     diagnosztika"
        echo -e "  ${CYAN}[8]${NC} EXIT"
        echo ""
        echo -ne "${BOLD}Választás [1-8]: ${NC}"

        read -r choice

        case "${choice}" in
            1) COMIC_OCR_BACKEND="ppocr";         run_quality 1  ;;
            2) COMIC_OCR_BACKEND="minicpm_ocr";   run_quality 1  ;;
            3) COMIC_OCR_BACKEND="gemini_flash";  run_quality 1  ;;
            4) COMIC_OCR_BACKEND="qwen2_vl";      run_quality 1  ;;
            5) run_debug  1 ;;
            6) run_dry    1 ;;
            7) run_doctor 1 ;;
            8)
                echo ""
                echo -e "${DIM}Kilépés.${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}[ERROR] Érvénytelen választás: ${choice}${NC}"
                ;;
        esac

        echo ""
        echo -e "${DIM}──────────────────────────────────────${NC}"
    done
}

# =============================================================================
# CLI argument parsing
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "${1,,}" in
            quality)
                PROFILE="quality"
                shift
                ;;
            quick)
                PROFILE="quick"
                shift
                ;;
            debug)
                PROFILE="debug"
                shift
                ;;
            dry|dry-run)
                PROFILE="dry"
                shift
                ;;
            doctor)
                PROFILE="doctor"
                shift
                ;;
            --model)
                if [[ -z "${2:-}" ]]; then
                    echo -e "${RED}[ERROR]${NC} --model után modellnév szükséges."
                    exit 2
                fi
                MODEL="${2}"
                MODEL_SOURCE="CLI argument"
                shift 2
                ;;
            --input-dir)
                INPUT_DIR="${2:-}"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="${2:-}"
                shift 2
                ;;
            *)
                EXTRA_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Argumentumszám mentése MIELŐTT parse_args módosítaná a $@-t
    local original_argc=$#

    print_banner
    ensure_virtual_environment
    detect_python
    check_python_version
    check_prerequisites
    print_gpu_info
    resolve_model

    parse_args "$@"

    # Argumentum nélküli indítás → persistent interaktív menü
    if [[ ${original_argc} -eq 0 ]]; then
        show_menu
        return
    fi

    # Argumentummal indítva → egyszeri futtatás, nem menu loop
    case "${PROFILE}" in
        quality) run_quality 0 ;;
        quick)   run_quick   0 ;;
        debug)   run_debug   0 ;;
        dry)     run_dry     0 ;;
        doctor)  run_doctor  0 ;;
        *)
            echo -e "${RED}[ERROR] Ismeretlen profil: ${PROFILE}${NC}"
            exit 2
            ;;
    esac
}

main "$@"
