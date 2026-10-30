#!/usr/bin/env bash
set -eu
BASE=/root/smolvla-test-20261020
export HTTPS_PROXY=http://127.0.0.1:18089
export HTTP_PROXY=http://127.0.0.1:18089
export UV_CACHE_DIR="$BASE/cache/uv"
"$BASE/tools/bin/uv" pip install --index-url https://pypi.org/simple --python "$BASE/venv/bin/python" "$BASE/wheels/torch-2.3.1-cp310-cp310-manylinux2014_aarch64.whl" 'torchvision==0.18.1' 'torch-npu==2.3.1' 'transformers==4.51.3' 'numpy==1.26.4' 'accelerate==1.7.0' 'huggingface-hub==0.34.4' 'safetensors==0.5.3' 'draccus==0.10.0' einops num2words 'gymnasium==0.29.1' termcolor psutil 'opencv-python-headless<4.12'
"$BASE/tools/bin/uv" pip install --index-url https://pypi.org/simple --python "$BASE/venv/bin/python" --no-deps 'lerobot==0.3.3'
echo ENV_READY
