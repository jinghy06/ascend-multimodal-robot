---
name: m1-vision
description: M1 视觉感知模块（人体/物体 YOLO、激光笔、雷达融合、跟随及抓取接口）在 Atlas 200I DK A2 上的构建、运行与验收测试流程
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
  - `vision_msgs 4.1.1` 由仓库子模块提供；不构建其依赖 Qt 的 RViz 插件
- 系统 Python 3.9 的 pip 是 21.3.1，**永远不要用它往 conda 环境装包**；一律 `python3 -m pip`（激活环境后）
- 构建工具链坑（已修复，重建系统时注意）：需 `dnf install -y libatomic` 且
  `ln -sf /usr/lib64/libatomic.so.1 /usr/lib64/libatomic.so`
- mamba：通道 `robostack-humble` 在 conda.anaconda.org 直连常超时；
  USTC 镜像 `https://mirrors.ustc.edu.cn/anaconda/cloud/robostack-humble` 的 noarch 会 404 导致事务中止，
  需要装包时优先重试官方通道（断点续传），或改用 pip / 源码构建
- PyPI 走清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

## 代码位置与模型

- 板上：`/root/ascend-multimodal-robot/m1_ws`（本 git 仓库内）
- 开发机侧源副本在模块负责人的本地工作目录；连接方法见开发机侧 skill `m1-vision-atlas`，不含在公开仓库中
- 人体模型：`/root/models/yolov8n-pose_640.om`
- 通用物体模型：`/root/models/yolov8n_640.om`
- **NPU 实测（2026-11-08）：姿态模型纯推理 49.7 FPS（20.1 ms/帧），真实图片全链路
  （letterbox→推理→NMS）检出正常；ROS 端到端输出人体框和 17 关键点**
- 通用物体、光斑、融合测距、跟随目标和抓取接口已分别用真实 OM 视频链路或仿真验证；
  真实相机、雷达和机械臂精度仍需实机验收
- `ros_image.py` 用 `array.array('B', img.tobytes())` 给 Image.data 赋值；直接给 bytes 会在
  当前 ROS 绑定逐元素校验，实测单帧约 830 ms，而 array 赋值约 0.02 ms，勿回退
- 长时间跑节点看 FPS 统计日志时加 `PYTHONUNBUFFERED=1`
- pyACL 7.0.RC1 已验证：`acl.mdl.get_desc(desc, model_id)` 两参数；
  `get_output_data_type` 返回裸 int；`acl.util.numpy_to_ptr` 在 numpy 2.x 下报 `_ARRAY_API`，
  必须用 `acl.util.bytes_to_ptr`

## 构建

```bash
cd /root/ascend-multimodal-robot
git submodule update --init --recursive
source /root/resident-integration-20261021/ros/setup.bash
cd m1_ws
if test ! -e src/voice_control; then ln -s ../../voice_control src/voice_control; fi
colcon build --symlink-install \
  --packages-select m1_msgs vision_msgs m1_vision voice_control \
  --cmake-args -DBUILD_TESTING=OFF
source install/setup.bash
```

`vision_msgs` 自带 C++ 测试在当前 GCC/RoboStack 组合下有 GLIBCXX 链接问题，因此只构建运行包，
不为该第三方依赖启用测试。项目语音测试与 M1 功能检查应单独执行。

## 运行

```bash
ros2 launch m1_vision m1_bringup.launch.py     # 6 节点实机视觉
ros2 launch m1_vision m1_mock.launch.py        # 无传感器仿真，含 map/base/laser TF
scripts/run_processor.sh hardware              # M1+M2 统一入口
```

参数文件：`m1_ws/src/m1_vision/config/m1_vision.yaml`。
物体检测默认 `target_fps=5.0`，为人体姿态跟随预留 NPU 算力；限流后的组合帧率仍需实机复测。
统一入口默认启用 ASR、关闭开放语义和 TTS；队长语音包提供 `tts_node` 后才可开启 TTS。
无 OM 时推理节点按 ONNX 兜底或空检测保活，但不能据此认定视觉功能通过。

## 无硬件链路测试

```bash
ros2 launch m1_vision m1_mock.launch.py
ros2 topic echo --once /m1/follow_target
ros2 topic echo --once /m1/grasp_target
ros2 topic echo --once /m1/laser_goal
```

预期：日志出现“目标锁定 → TRACKING”，跟随话题约 20 Hz，抓取目标带类别/像素框/测距位置，
激光目标使用 `map` 坐标系。仿真通过不代表真实导航或抓取通过。
结束测试时用进程组发送 SIGINT，确认没有遗留孤儿节点；不要只 kill `ros2 run` 包装进程。

## 验收指标自测点

- NPU ≥20 FPS：yolo_npu_node 每 100 帧打端到端 FPS 日志，不能用纯推理 FPS 代替
- 丢失恢复 ≤3 s：target_tracking_node 恢复时打印实际耗时
- 距离/定位精度：/m1/follow_target、/m1/laser_goal 与卷尺真值对比

## 对外接口

- M2 可消费 `/m1/target_status`（std_msgs/String：TRACKING/LOST/RECOVERING）；
  语音结构化命令通过 `/voice/command` 交给整机任务管理器
- M3 消费 `/m1/grasp_target`（vision_msgs/Detection2DArray，类别、像素框及 laser_frame 测距位置）；
  二维框中心测距不等于完整六自由度抓取位姿
- M4 消费 `/m1/follow_target`（20 Hz, base_link 系）、`/m1/laser_goal`（map 系，0.3 m 安全内缩）
- M3/M4 执行器及整机任务管理器不在本模块内，需队友代码与真实硬件联调
