---
name: m1-vision
description: M1 视觉感知模块（YOLO NPU 推理 + 激光笔光斑检测 + 雷达融合测距 + 目标跟踪）在 Atlas 200I DK A2 上的构建、运行与验收测试流程
type: prompt
whenToUse: 当用户要求构建/运行/调试/测试 M1 视觉感知模块，或涉及 /m1/ 话题、YOLO NPU 推理、光斑检测、融合测距、人体跟随视觉部分时
---

# M1 视觉感知模块操作指南（Atlas 200I DK A2）

## 环境事实（已实测，勿重复探测）

- 板卡：`davinci-mini`，openEuler 22.03，kernel 5.10 aarch64，昇腾 310B4（npu-smi 23.0.rc3，健康）
- CANN：7.0.RC1，位于 `/usr/local/Ascend/ascend-toolkit`（`source .../set_env.sh`）
- ROS 2：Humble，**RoboStack conda 环境**，前缀 `/root/resident-integration-20261021/ros`
  - 激活：`source /root/resident-integration-20261021/ros/setup.bash`
  - Python 3.14（conda 环境内），colcon 已 pip 装入该环境（`ros/bin/colcon`）
  - cv2 = opencv-python-headless 5.0.0（pip 装入该环境）
  - **无 cv_bridge**：M1 用 `m1_vision/ros_image.py`（纯 numpy）替代，不要 import cv_bridge
  - **无 vision_msgs**：`/m1/grasp_target`（M3 接口，W9 任务）实现时再装（mamba 或源码构建）
- 系统 Python 3.9 的 pip 是 21.3.1，**永远不要用它往 conda 环境装包**；一律 `python3 -m pip`（激活环境后）
- 构建工具链坑（已修复，重建系统时注意）：需 `dnf install -y libatomic` 且
  `ln -sf /usr/lib64/libatomic.so.1 /usr/lib64/libatomic.so`
- mamba：通道 `robostack-humble` 在 conda.anaconda.org 直连常超时；
  USTC 镜像 `https://mirrors.ustc.edu.cn/anaconda/cloud/robostack-humble` 的 noarch 会 404 导致事务中止，
  需要装包时优先重试官方通道（断点续传），或改用 pip / 源码构建
- PyPI 走清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

## 代码位置

- 板上：`/root/ascend-multimodal-robot/m1_ws`（本 git 仓库内，src/m1_msgs + src/m1_vision）
- 开发机侧源副本在模块负责人的本地工作目录（改动后重新上传部署，连接方法见开发机侧 skill `m1-vision-atlas`，不含在公开仓库中）
- OM 模型已就绪：`/root/models/yolov8n-pose_640.om`（2026-11-08 ATC 转换，Ascend310B4 FP16）
- **NPU 实测（2026-11-08）：纯推理 49.7 FPS（20.1 ms/帧），真实图片全链路（letterbox→推理→NMS）
  检出正常；ROS 端到端（camera_driver 视频文件 → yolo_npu_node NPU → /m1/person_detections
  带框+17 关键点）已验证**；复现 `python3 m1_ws/scripts/bench_om.py /root/models/yolov8n-pose_640.om [图片]`
- 长时间跑节点看 FPS 统计日志时加 `PYTHONUNBUFFERED=1`（输出重定向到文件时默认块缓冲，
  定期报告可能延迟落盘）
- pyACL 7.0.RC1 已验证的 API 形式：`acl.mdl.get_desc(desc, model_id)` 两参数；
  `get_output_data_type` 返回裸 int；`acl.util.numpy_to_ptr` 在 numpy 2.x 下报 `_ARRAY_API`，
  必须用 `acl.util.bytes_to_ptr`（`m1_vision/acl_inference.py` 已按此实现，勿回退）

## 构建

```bash
source /root/resident-integration-20261021/ros/setup.bash
cd /root/ascend-multimodal-robot/m1_ws
colcon build --symlink-install
```

## 运行（实机）

```bash
source /root/resident-integration-20261021/ros/setup.bash
source /root/ascend-multimodal-robot/m1_ws/install/setup.bash
ros2 launch m1_vision m1_bringup.launch.py     # 5 节点：camera/yolo/laser_spot/fusion/tracking
```

参数文件：`m1_ws/src/m1_vision/config/m1_vision.yaml`（标注了各周需实测整定的项）。
NPU 推理需要先把 OM 模型放到 `om_model_path`（转换流程：`scripts/export_onnx.sh` → `scripts/atc_convert.sh`）；
无 OM 时节点自动降级（onnx 兜底 → 空检测保活），不会崩溃。

## 无硬件链路测试（已通过 2026-11-07）

```bash
ros2 run m1_vision mock_perception_node &   # 仿真 scan+检测+光斑
ros2 run m1_vision fusion_ranging_node &
ros2 run m1_vision target_tracking_node &
```

预期：日志出现"目标锁定 → TRACKING"，`ros2 topic echo /m1/follow_target` 有 20 Hz 输出。
mock 无 map TF，laser_goal 路径报"map TF 不可用"属预期（整机 SLAM 起来后自然解决）。
注意：`pkill -f m1_vision` 会误杀包含该串的包装 shell，结束进程用保存的 PID。

## 验收指标自测点

- NPU ≥20 FPS：yolo_npu_node 每 100 帧打端到端 FPS 日志
- 丢失恢复 ≤3 s：target_tracking_node 恢复时打印实际耗时
- 距离/定位精度：/m1/follow_target、/m1/laser_goal 与卷尺真值对比（见设计文档第 9 章）

## 对外接口

- M2（voice_control 包）订阅 `/m1/target_status`（std_msgs/String：TRACKING/LOST/RECOVERING）
- M3 需要 `/m1/grasp_target`（vision_msgs/Detection2DArray）——**尚未实现**，W9 在
  fusion_ranging_node 增加转换发布，届时先装 vision_msgs
- M4 消费 `/m1/follow_target`（20 Hz, base_link 系）、`/m1/laser_goal`（map 系，0.3 m 安全内缩已做）
