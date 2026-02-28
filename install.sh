#!/bin/bash
set -e  # Exit immediately if a command fails

echo ">>> 1. Nuke existing incompatible versions..."
pip uninstall -y torch torchvision torchaudio flash-attn vllm transformers tensordict

echo ">>> 2. Installing Torch 2.4.0 (The Foundation)..."
# We force the CUDA 12.1 index so we get pre-compiled binaries
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo ">>> 3. Installing Flash Attention (Pre-compiled Wheel)..."
# We download the specific wheel for Torch 2.4 + Python 3.11 to avoid compilation
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-build-isolation

echo ">>> 4. Installing the rest of the stack..."
pip install -r requirements.txt

echo ">>> 5. Sanity Check..."
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}, CUDNN: {torch.backends.cudnn.version()}'); import flash_attn; print('Flash Attn: Loaded')"
