#!/usr/bin/env python3
"""生成 M1 测试视频（板上运行）：
  /root/test_bus.mp4       bus.jpg 静止循环（人体检测端到端测试）
  /root/test_spot.mp4      bus.jpg + 右上角红色圆点（光斑检测测试）
"""
import cv2

ASSET = ("/root/resident-integration-20261021/ros/lib/python3.14/"
         "site-packages/ultralytics/assets/bus.jpg")

img = cv2.imread(ASSET)
h, w = img.shape[:2]

wr = cv2.VideoWriter("/root/test_bus.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
for _ in range(300):
    wr.write(img)
wr.release()

spot = img.copy()
cv2.circle(spot, (600, 300), 6, (0, 0, 255), -1)  # 纯红亮点模拟激光光斑
wr = cv2.VideoWriter("/root/test_spot.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
for _ in range(300):
    wr.write(spot)
wr.release()
print("videos ok", w, h)
