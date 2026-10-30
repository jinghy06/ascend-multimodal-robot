#!/usr/bin/env bash
set -eu
BASE=/root/smolvla-test-20261020
export HTTPS_PROXY=http://127.0.0.1:18089
export HTTP_PROXY=http://127.0.0.1:18089
export UV_CACHE_DIR="$BASE/cache/uv"
export UV_PYTHON_INSTALL_DIR="$BASE/python"
export HF_HOME="$BASE/cache/huggingface"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=180
UV="$BASE/tools/bin/uv"
"$UV" python install 3.10 --install-dir "$BASE/python" --no-bin
PYTHON=$(find "$BASE/python" -path '*/bin/python3.10' -type f | head -1)
"$UV" venv --python "$PYTHON" "$BASE/venv"
"$UV" pip install --index-url https://pypi.org/simple --python "$BASE/venv/bin/python" 'torch==2.3.1' 'torchvision==0.18.1' 'torch-npu==2.3.1' 'transformers==4.51.3' 'numpy==1.26.4' 'accelerate==1.7.0' 'huggingface-hub==0.34.4' 'safetensors==0.5.3' 'draccus==0.10.0' einops num2words 'gymnasium==0.29.1' termcolor psutil 'opencv-python-headless<4.12'
"$UV" pip install --index-url https://pypi.org/simple --python "$BASE/venv/bin/python" --no-deps 'lerobot==0.3.3'
"$BASE/venv/bin/python" -u "$BASE/download_model.py"
echo BOOTSTRAP_DONE
