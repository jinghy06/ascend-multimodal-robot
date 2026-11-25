#!/usr/bin/env bash
# 开发机（有 GPU/无 GPU 均可）执行：YOLOv8n-pose → ONNX
# 依赖：pip install ultralytics
# 说明：opset=11、固定输入 640x640（ATC 上 INT8 与动态 shape 不可兼得，输入固定档位）
set -euo pipefail

yolo export model=yolov8n-pose.pt format=onnx opset=11 imgsz=640

echo "生成 yolov8n-pose.onnx，拷贝到 Atlas 200I DK A2 后执行 atc_convert.sh"
