#!/bin/bash
set -e

# Create a persistent venv for SQL-R1-MORL that survives pod restarts.
# Usage:
#   bash sh/setup_venv.sh           # create venv (skip if exists)
#   bash sh/setup_venv.sh --force   # recreate from scratch
#   VENV_PATH=/my/path bash sh/setup_venv.sh  # custom location

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="${VENV_PATH:-$PROJ_DIR/.venv}"
FORCE=0
if [ "$1" = "--force" ]; then FORCE=1; fi

# ── Logging helpers ──
BOLD='\033[1m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
RESET='\033[0m'

banner() { echo ""; echo "${CYAN}╔══════════════════════════════════════════════════════╗${RESET}"; echo "${CYAN}║${RESET}  ${BOLD}$1${RESET}"; echo "${CYAN}╚══════════════════════════════════════════════════════╝${RESET}"; }
step()   { echo "${GREEN}>>>${RESET} ${BOLD}$1${RESET}"; }
info()   { echo "    ${YELLOW}→${RESET} $1"; }
ok()     { echo "    ${GREEN}✓${RESET} $1"; }
fail()   { echo "    ${RED}✗${RESET} $1"; exit 1; }

banner "SQL-R1-MORL Persistent Venv Setup"
info "Project dir: $PROJ_DIR"
info "Venv path:   $VENV_PATH"

# ── Skip if venv already exists ──
if [ -f "$VENV_PATH/bin/activate" ] && [ "$FORCE" -eq 0 ]; then
    ok "Venv already exists at $VENV_PATH — skipping (use --force to recreate)"
    info "Activate with: source $VENV_PATH/bin/activate"
    exit 0
fi

# ── Validate Python version ──
banner "Step 1/6: Validate Python"
PYTHON_VER=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
step "Detected Python $PYTHON_VER"
if [ "$PYTHON_VER" != "3.11" ]; then
    fail "Python 3.11 required (flash-attn wheel is cp311). Got Python $PYTHON_VER"
fi
ok "Python 3.11 confirmed"

# ── Create venv ──
banner "Step 2/6: Create Venv"
if [ "$FORCE" -eq 1 ] && [ -d "$VENV_PATH" ]; then
    step "Removing existing venv (--force)..."
    rm -rf "$VENV_PATH"
fi
step "Creating venv at $VENV_PATH..."
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
ok "Venv created and activated"

# ── Pip bootstrap ──
banner "Step 3/6: Pip Setup"
step "Upgrading pip, wheel, setuptools..."
pip install --upgrade pip wheel setuptools
ok "Pip ready"

# ── PyTorch + CUDA ──
banner "Step 4/6: PyTorch 2.4.0 + CUDA 12.1"
step "Installing Torch 2.4.0 (CUDA 12.1 wheels)..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
ok "PyTorch installed"

# ── Flash Attention ──
step "Installing Flash Attention 2.6.3 (pre-compiled wheel)..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-build-isolation
ok "Flash Attention installed"

# ── Python dependencies ──
banner "Step 5/6: Python Dependencies"
step "Installing requirements.txt..."
pip install -r "$PROJ_DIR/requirements.txt"

step "Installing verl package in editable mode..."
pip install -e "$PROJ_DIR"
ok "All Python dependencies installed"

# ── Wandb login ──
if [ -f "$PROJ_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJ_DIR/.env" | xargs)
fi
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY" --relogin
    ok "Logged in to wandb"
else
    info "WANDB_API_KEY not set — skipping login (set it in .env)"
fi

# ── Sanity check ──
banner "Step 6/6: Sanity Check"
python -c "
import torch
print(f'  Torch:    {torch.__version__}')
print(f'  CUDA:     {torch.version.cuda}')
print(f'  cuDNN:    {torch.backends.cudnn.version()}')
print(f'  Devices:  {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}:    {torch.cuda.get_device_name(i)}')
import flash_attn
print(f'  FlashAttn: OK')
import vllm
print(f'  vLLM:     OK')
"
ok "All checks passed"

banner "Venv Setup Complete"
echo ""
info "Venv:     $VENV_PATH"
info "Activate: source $VENV_PATH/bin/activate"
info "This venv persists across pod restarts — no need to re-run install.sh"
echo ""
step "Next: sh sh/setup_data.sh"
echo ""
