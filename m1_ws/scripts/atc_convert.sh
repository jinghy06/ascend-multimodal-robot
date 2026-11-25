#!/usr/bin/env bash
# Atlas 200I DK A2 板上执行：ONNX → OM（昇腾 310B4）
# 前置：source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -euo pipefail

ONNX=${1:-yolov8n-pose.onnx}
OUT=${2:-yolov8n-pose_640}

atc --model="${ONNX}" \
    --framework=5 \
    --soc_version=Ascend310B4 \
    --output="${OUT}" \
    --input_shape="images:1,3,640,640"

echo "生成 ${OUT}.om —— 默认 FP16；帧率不达标再评估 INT8（需 100-200 张校准图，见设计文档 6.4）"
