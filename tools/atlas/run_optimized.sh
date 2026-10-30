#!/usr/bin/env bash
set -eo pipefail
BASE=/root/smolvla-test-20261020
source /opt/Ascend/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export TE_PARALLEL_COMPILER=1
export MAX_COMPILE_CORE_NUMBER=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec timeout 600 "$BASE/venv/bin/python" -u "$BASE/optimization-v1/run_optimized.py" "$@"
