#!/usr/bin/env python3
"""OM 模型 NPU 推理基准与链路验证（板上直接运行，无需 ROS）。

用法：
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 bench_om.py /path/to/yolov8n-pose_640.om [测试图片.jpg]

不带图片：随机输入跑 100 帧，报告纯推理 FPS（验证 OmModel 封装与 NPU 链路）。
带图片  ：letterbox → NPU 推理 → 解码 + NMS → 打印人体检测框与关键点数
          （验证 yolo_postprocess 与真实数据全链路）。
"""

import sys
import time

import numpy as np

sys.path.insert(0, "/root/ascend-multimodal-robot/m1_ws/src/m1_vision")
from m1_vision.acl_inference import OmModel  # noqa: E402
from m1_vision import yolo_postprocess as yp  # noqa: E402


def bench_fps(model, iters=100):
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    for _ in range(10):  # 预热
        out = model.infer(x)[0]
    print(f"output shape: {out.shape}（期望 (1, 56, 8400)）")
    t0 = time.perf_counter()
    for _ in range(iters):
        model.infer(x)
    dt = time.perf_counter() - t0
    print(f"NPU 纯推理：{iters / dt:.1f} FPS（{1000 * dt / iters:.2f} ms/帧，{iters} 帧）")


def test_image(model, image_path):
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        print(f"读图失败：{image_path}")
        return
    img_lb, ratio, dwdh = yp.letterbox(img, 640)
    t0 = time.perf_counter()
    out = model.infer(yp.to_input_tensor(img_lb))[0]
    infer_ms = (time.perf_counter() - t0) * 1000
    boxes, scores, kpts = yp.decode_pose_output(out, conf_th=0.5)
    keep = yp.nms(yp.cxcywh_to_xyxy(boxes), scores, iou_th=0.45)
    boxes_xyxy, kpts = yp.scale_back(yp.cxcywh_to_xyxy(boxes)[keep], kpts[keep],
                                     ratio, dwdh, img.shape)
    print(f"单帧推理 {infer_ms:.1f} ms，检出 {len(keep)} 个人体：")
    for i in range(len(keep)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        n_kpt = int((kpts[i][:, 2] > 0.5).sum())
        print(f"  [{i}] 框=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) "
              f"置信度={scores[keep][i]:.2f} 有效关键点={n_kpt}/17")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    model = OmModel(sys.argv[1])
    try:
        bench_fps(model)
        if len(sys.argv) >= 3:
            test_image(model, sys.argv[2])
    finally:
        model.close()


if __name__ == "__main__":
    main()
