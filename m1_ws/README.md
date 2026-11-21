# M1 视觉感知模块 — ROS 2 工作区（Atlas 200I DK A2）

《M1 视觉感知模块设计与实施文档（Atlas 200I DK A2）》第 4–6 章的代码实现。
平台：HUAWEI Atlas 200I DK A2（昇腾 310B4）+ Ubuntu 22.04 + ROS 2 Humble + CANN。

## 目录结构

```
m1_ws/
├── src/
│   ├── m1_msgs/                  # 自定义消息（人体检测+关键点、融合测距目标）
│   └── m1_vision/                # Python 功能包（ament_python）
│       ├── m1_vision/
│       │   ├── camera_driver_node.py    # USB 相机采集 /camera/image_raw（支持视频文件调试）
│       │   ├── yolo_npu_node.py         # YOLOv8n-pose 推理（pyACL NPU → ONNX CPU 兜底）
│       │   ├── object_detection_node.py # YOLOv8n 通用物体检测（M3 抓取感知）
│       │   ├── laser_spot_node.py       # 光斑检测（HSV+形态学+连通域，亚像素质心）
│       │   ├── fusion_ranging_node.py   # 相机-雷达融合测距（方位角窗口聚类）
│       │   ├── target_tracking_node.py  # 卡尔曼跟踪 + 状态机 + 激光笔目标解算
│       │   ├── mock_perception_node.py  # 无硬件仿真数据源（联调工具）
│       │   ├── acl_inference.py         # pyACL OM 模型封装
│       │   ├── yolo_postprocess.py      # letterbox / 解码 / NMS（纯 numpy）
│       │   ├── ros_image.py             # Image 消息与 cv2 互转（替代 cv_bridge）
│       │   └── tf_utils.py              # 四元数旋转、角度归一化
│       ├── launch/m1_bringup.launch.py  # 一键启动 6 节点
│       ├── config/m1_vision.yaml        # 全部参数（标注了各周需实测整定的项）
│       └── urdf/m1_sensors.urdf.xacro   # 传感器 TF 片段（并入整机 URDF）
└── scripts/
    ├── export_onnx.sh      # 开发机：pt → ONNX（opset=11, 640 固定档）
    ├── atc_convert.sh      # 板上：ONNX → OM（Ascend310B4）
    └── setup_atlas_env.sh  # W1 环境搭建步骤（驱动/固件/CANN/ROS 2）
```

## 数据流

```
相机 ─→ /camera/image_raw ─┬─→ yolo_npu_node ─→ /m1/person_detections ─┐
                           ├─→ object_detection_node ─→ /m1/object_detections ─┤
                           │                           └→ /detected_object    │
                           └─→ laser_spot_node ─→ /m1/laser_spot ────────────┤
雷达 ─→ /scan ────────────────────────────────────────────────────────────┤
                                                                            ▼
                                                     fusion_ranging_node ─→ /m1/ranged_targets
                                                                         ─→ /m1/laser_spot_ranged
                                                                         ─→ /m1/grasp_target
                                                                            ▼
                                                     target_tracking_node ─→ /m1/follow_target (20 Hz)
                                                                         ─→ /m1/laser_goal (map)
                                                                         ─→ /m1/target_status
```

`/m1/person_detections` 用自定义 `m1_msgs/PersonDetectionArray` 携带 17 个关键点。
`/m1/grasp_target` 使用 `vision_msgs/Detection2DArray`，保留类别和像素框，并在
`results[].pose.pose.position` 中写入 `laser_frame` 下的雷达测距位置；无有效雷达回波时不输出该目标。
二维框中心测距只是抓取位姿的近似，不能代替机械臂所需的完整六自由度位姿与标定。

## 构建（板上）

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

`vision_msgs 4.1.1` 只需消息包；不构建其依赖 Qt 的 RViz 插件。`voice_control` 通过
`src/voice_control` 符号链接接入同一工作区。

## 运行

```bash
# 模型位于 /root/models/yolov8n-pose_640.om 和 /root/models/yolov8n_640.om
ros2 launch m1_vision m1_bringup.launch.py                    # M1 实机
ros2 launch m1_vision m1_mock.launch.py                       # M1 无传感器仿真
ros2 launch m1_vision processor_bringup.launch.py use_mock:=true enable_asr:=false
scripts/run_processor.sh hardware                             # M1+M2 实机统一入口
```

`processor_bringup.launch.py` 默认启用麦克风 ASR，开放语义服务默认关闭。无音频设备时设
`enable_asr:=false`。队长当前语音包没有 `tts_node`，因此 TTS 默认关闭；只有语音包提供
`tts_node` 与 `configs/tts.yaml` 时才设置 `enable_tts:=true`，不能把 WAV 生成当作声卡播放。
物体检测默认 `target_fps:=5.0`，为人体姿态跟随预留 NPU 算力；该调度值仍需在实机复测。

无硬件仿真会补充 `map → base_link → laser_frame` 静态 TF，可检查 `/m1/follow_target`、
`/m1/grasp_target` 和 `/m1/laser_goal`。仿真通过不代表真实相机、雷达、导航或机械臂验收通过。

## 验收指标对应

| 指标（文档第 3 章） | 验证方式 |
|---|---|
| NPU ≥20 FPS（1080p） | yolo_npu_node 每 100 帧打一次端到端 FPS 日志，60 s 取均值（文档 9.1） |
| 距离误差 ≤±0.3 m | fusion 窗口聚类中位距离，卷尺抽样实测（文档 9.2） |
| 丢失恢复 ≤3 s | target_tracking_node 恢复时日志打印实际耗时（文档 9.2） |
| 光斑定位 ≤0.2 m | /m1/laser_goal 与卷尺真值对比（文档 9.3） |

## 版本锁定记录（W1 结束时填写）

- CANN toolkit：7.0.RC1（已验证 ATC + pyACL 推理全链路，2026-11-08）
- NPU 驱动 / 固件：npu-smi 23.0.rc3
- ROS 2：Humble（RoboStack conda 环境，Python 3.14）
- 模型：YOLOv8n-pose 与 YOLOv8n detect，ONNX opset=11，输入 1×3×640×640，FP16
- **NPU 实测：姿态模型纯推理 49.7 FPS（20.1 ms/帧），bus.jpg 全链路检出 3 人 + 17 关键点正常**
  （复现：`python3 m1_ws/scripts/bench_om.py /root/models/yolov8n-pose_640.om [图片]`）
- 通用检测 OM 已在 ROS 图像链路中检出 4 个人体候选和 bus；红色合成光斑实测质心
  `(599.92, 299.88)`，合成真值为 `(600, 300)`。这些结果不代替真实物体与强光环境验收。
- pyACL 7.0.RC1 注意：`acl.mdl.get_desc(desc, model_id)` 两参数形式；
  `get_output_data_type` 返回裸 int；`numpy_to_ptr` 不支持 numpy 2.x，须用 `bytes_to_ptr`

## 已知待办 / 板上需验证项

1. 内参标定前 fusion 用 `fallback_hfov_deg` 线性近似，实测前务必配置 `camera_info_yaml`。
2. 相机曝光锁定（强光下光斑检测）需按具体 USB 相机 v4l2 能力继续整定。
3. 前处理目前在 CPU（letterbox），真实相机端到端帧率需在不残留旧节点的情况下验收。
4. 板上 ROS 为 RoboStack conda 环境（Python 3.14），未装 cv_bridge；
   图像转换已由 `ros_image.py`（纯 numpy）替代，无该依赖。
5. M3/M4 执行器、整机任务管理器、真实声卡及导航/抓取精度仍需对应硬件和队友模块联调；
   本工作区的仿真输出不能替代这些验收。
