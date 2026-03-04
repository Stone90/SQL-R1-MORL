#!/bin/bash
set -e

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

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
fail()   { echo "    ${RED}✗${RESET} $1"; }

banner "SQL-R1-MORL Full Install"

# ── 1. System essentials ──
banner "Step 1/7: System Packages"
step "Installing system dependencies..."
apt update && apt install -y git git-lfs python3-pip wget unzip pv build-essential ninja-build tmux
git lfs install
ok "System packages installed"

# ── 2. Pip bootstrap ──
banner "Step 2/7: Pip Setup"
step "Upgrading pip, wheel, setuptools..."
pip install --upgrade pip wheel setuptools
ok "Pip ready"

# ── 3. PyTorch + CUDA ──
banner "Step 3/7: PyTorch 2.4.0 + CUDA 12.1"
step "Removing incompatible versions..."
pip uninstall -y torch torchvision torchaudio flash-attn vllm transformers tensordict 2>/dev/null || true

step "Installing Torch 2.4.0 (CUDA 12.1 wheels)..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
ok "PyTorch installed"

# ── 4. Flash Attention ──
banner "Step 4/7: Flash Attention 2.6.3"
step "Installing pre-compiled wheel (Torch 2.4 + Python 3.11)..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-build-isolation
ok "Flash Attention installed"

# ── 5. Python dependencies ──
banner "Step 5/7: Python Dependencies"
step "Installing requirements.txt (vLLM, Ray, transformers, wandb, etc.)..."
pip install -r "$PROJ_DIR/requirements.txt"

step "Installing verl package in editable mode..."
pip install -e "$PROJ_DIR"
ok "All Python dependencies installed"

# ── 6. Wandb login ──
banner "Step 6/7: Wandb Login"
if [ -f "$PROJ_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJ_DIR/.env" | xargs)
fi
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY" --relogin
    ok "Logged in to wandb"
else
    info "WANDB_API_KEY not set — skipping login (set it in .env)"
fi

# ── 7. Sanity check ──
banner "Step 7/7: Sanity Check"
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

banner "Install Complete"
echo ""
step "Next: download model, data, and databases"
info "sh sh/setup_data.sh"
echo ""
info "Tip: use sh/setup_venv.sh for a persistent venv that survives pod restarts"
echo ""
