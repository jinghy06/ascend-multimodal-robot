# SmolVLA + LeKiwi 数据采集与后训练指南

> 更新时间：2026-11-24  
> 适用对象：项目成员 & 后续接手的 AI Agent  
> 适用平台：Huawei Atlas 200I DK A2（Ascend 310B4）+ SO-101 机械臂 + LeKiwi 轮式底盘

本文档整理了本项目中**示教数据采集、USB 带宽配置、SmolVLA 后训练（微调）以及板端部署对齐**的已验证事实与操作规范。所有硬件参数、路径与实测结论均基于板端实际验证。

---

## 1. 硬件与环境事实（Agent 快速同步）

| 项目 | 实测值 / 路径 | 说明 |
|---|---|---|
| 主控板 | Atlas 200I DK A2（Ascend 310B4），`root@<ATLAS_IP>` | 可见内存约 3513 MiB + 8 GB swap；SSH 方法见 `.agents/skills/ssh-atlas/SKILL.md` |
| 板端 Python 环境 | `/root/smolvla-test-20261020/venv/bin/python` | Python 3.10 + `torch_npu 2.3.1` + `lerobot 0.3.3` |
| 板端遥控与录制目录 | `/root/so101-lekiwi/` | 对应本地根目录的 `atlas_demo_recorder.py`、`so101_lekiwi_control.py`、`so101_lekiwi_gamepad.py` |
| 机械臂/底盘串口 | CH343 (`1a86:55d3`)，自动识别 `/dev/serial/by-id/...` | 机器人 ID 默认 `my_awesome_kiwi` |
| 手柄 | Xbox 360 USB (`045e:028e`) | 默认 `--input-backend usb` |
| 前视相机 (`front`) | `/dev/video0`，USB ID `4c4a:4a55`（Jieli / IVOOMI Ne） | 仅 1 个等时带宽档位（固定 3069 B/µframe）；使用 `MJPG 640×480@10fps` |
| 腕部相机 (`wrist`) | `/dev/video2`，USB ID `1e45:0209`（USB Camera） | 7 个等时档位；配合 `uvcvideo quirks=0x80` 使用 `YUYV 640×480@10fps`（altsetting 3 = 1024 B/µframe） |

---

## 2. 双摄 USB 带宽修复（已验证，切勿误删）

### 根因
两路摄像头均为 USB 2.0 高速设备，且挂载在同一条 USB 2.0 480M 总线（`Bus 03` 外接 USB 2.0 Hub `214b:7260`）下：
- `4c4a:4a55` 一开流即固定占用 **3069 B/µframe**；
- `1e45:0209` 固件会虚报带宽（默认请求 `altsetting 5` 2048 B 或 `altsetting 7` 3072 B），叠加超限导致内核报 `Not enough bandwidth for altsetting 5`，录制时 `Camera wrist returned no frame after retries`。

### 已落地修复
1. **板端内核模块持久化配置**：`/etc/modprobe.d/uvcvideo.conf` 已写入：
   ```text
   options uvcvideo quirks=0x80
   ```
   `0x80` 即 `UVC_QUIRK_FIX_BANDWIDTH`，使内核按实际像素吞吐计算未压缩流带宽，将 `1e45:0209` 的 `YUYV 640×480@10fps` 降到 `altsetting 3`（1024 B/µframe），与前摄稳定共存。
2. **代码自动匹配格式**：`atlas_demo_recorder.py` 中的 `open_v4l2_camera()` 会按 sysfs 中的 USB `idVendor:idProduct` 自动分配格式：
   - `1e45:0209` → `YUYV 640×480@fps`
   - 其余相机（含 `4c4a:4a55`）→ `MJPG 640×480@fps`
   - 统一设置 `CAP_PROP_BUFFERSIZE = 1`。
   - 即使重启或插拔导致 `/dev/video0` 与 `/dev/video2` 互换，格式仍会自动匹配。
3. **备用硬件方案**：板上 `Bus 01 / Bus 02` 是一套独立的空闲 xHCI 控制器，后续若增加更多 USB 2.0 摄像头，可物理分流到另一组 USB 接口。

> **排障提示**：若某次重插摄像头后再次出现无帧，先检查 `cat /sys/module/uvcvideo/parameters/quirks` 是否为 `128`（即 `0x80`），以及 `dmesg | grep -iE 'bandwidth|altsetting'`。

---

## 3. 数据集格式与采集规范

### 3.1 当前录制输出 Schema（标准 LeRobot 格式）
- `fps`: `10`
- `robot_type`: `"lekiwi"`
- `observation.images.front`: `(480, 640, 3)` RGB，AV1 MP4 视频
- `observation.images.wrist`: `(480, 640, 3)` RGB，AV1 MP4 视频
- `observation.state` / `action`: 9 维 `float32`，顺序为：
  ```text
  arm_shoulder_pan.pos, arm_shoulder_lift.pos, arm_elbow_flex.pos,
  arm_wrist_flex.pos, arm_wrist_roll.pos, arm_gripper.pos,
  x.vel, y.vel, theta.vel
  ```
- `task`: 语言指令字符串（如 `"pick up the orange"`）
- `timestamp`: 由 `self.frames / self.fps` 生成，与 AV1 视频时间轴严格对齐（修复了此前 `save_episode()` 的 `check_timestamps_sync` 报错）。

### 3.2 机位与构图要求（对齐 LeKiwi 官方）
- **`front`（底盘前视）**：提供全局视野。**抓取地面物体**时当前低位合适；若改为**抓取桌面物体**，需适当调高/微仰镜头，确保画面中同时看清目标物体与机械臂工作区。
- **`wrist`（腕部夹爪）**：保持画面底部可见夹爪指尖开合，用于近距离对准与闭合判断。

### 3.3 采集操作建议
1. **控制频率对齐**：`so101_lekiwi_gamepad.py` 默认 `--rate 20.0`，而录制器默认 `fps=10`。正式采集时建议显式指定 **`--rate 10`**，避免控制节拍与数据集 `fps` 漂移：
   ```bash
   /root/smolvla-test-20261020/venv/bin/python /root/so101-lekiwi/so101_lekiwi_gamepad.py --rate 10
   ```
2. **数据量与轨迹一致性**：
   - 单任务抓取建议至少采集 **50～100 条**完整成功示范（快速验证可先用 30 条跑通闭环）。
   - 底盘移动靠近目标的路线尽量平滑、可复现，避免原地反复微调或无效停顿。
   - 覆盖物体在工作区内的不同位置、轻微光照/背景变化；同一任务的 `task` 文本保持完全一致。
3. **单集目录合并（当前工程待办）**：
   - 目前 `DemoCommandManager.start()` 每次录制会新建独立目录 `demonstrations/<YYYYMMDD_HHMMSS>/`（每目录仅 1 条 episode）。
   - **训练前必须把所有合格 episode 合并为同一个 LeRobot 数据集**（或后续将 `EpisodeRecorder` 改造为支持向同一数据集目录连续追加 episode）。

---

## 4. SmolVLA 后训练（微调）规范

训练在 GPU 主机（建议 ≥ 24 GB 显存，如 RTX 4090）上执行，不要在 Atlas 310B 上训练。

### 4.1 标准微调命令模板（双摄适配 `lerobot/smolvla_base`）
`lerobot/smolvla_base` 预训练默认有 3 个相机输入槽位（`camera1` / `camera2` / `camera3`）。用 LeKiwi 双摄微调时，通过 `--rename_map` 映射前两路，并用 `--policy.empty_cameras=1` 补空位：

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<YOUR_DATASET_ID_OR_LOCAL_REPO> \
  --output_dir=outputs/train/smolvla_lekiwi \
  --policy.device=cuda \
  --batch_size=16 \
  --steps=40000 \
  --policy.empty_cameras=1 \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --rename_map='{
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2"
  }'
```

### 4.2 关键超参说明
| 参数 | 推荐值 | 原因 |
|---|---|---|
| `--policy.empty_cameras` | `1` | 补齐 `smolvla_base` 第 3 路相机槽位，避免特征校验报错 |
| `--rename_map` | `front -> camera1`, `wrist -> camera2` | 与预训练视觉槽位对齐；**顺序不可颠倒** |
| `--policy.chunk_size` / `n_action_steps` | `15～25`（推荐 `20`） | 预训练默认 `50` 是按 ~30 fps 设的；在 `10 fps` 下 50 步 = 5 秒开环，底盘容易跑偏，20 步 = 2 秒更稳 |
| `--policy.resize_imgs_with_padding` | `(512, 512)` 或 `(256, 256)` | **必须与板端推理配置一致**；若板端为低延迟采用 256 分辨率，微调时应同步设为 `[256, 256]` |
| `--policy.num_steps`（流匹配步数） | `10`（或与板端一致的 `3～5`） | 板端实测 10 步偏慢、3 步快但需匹配训练/验证后再定 |
| 冻结策略 | 默认 `freeze_vision_encoder=True`, `train_expert_only=True`, `train_state_proj=True` | 显存友好、小样本不易过拟合；样本充足（≥100 条）时可再评估放开 |

### 4.3 备选回退策略（ACT）
若 SmolVLA 在特定单任务上收敛不佳或板端延迟超预算，同一份 LeRobot 数据集可直接无缝训练 ACT（`--policy.type=act`），无需重新采数据。

---

## 5. 板端部署对齐清单（Atlas 310B 必查）

把微调好的 checkpoint 拷回 Atlas 200I DK A2 推理前，必须逐项核对以下一致性：

1. **相机键名与顺序严格一致**：SmolVLA 按字典顺序拼接视觉 token，训练时是 `front` 在前、`wrist` 在后，推理输入字典顺序和键名必须完全一致，否则策略直接失效。
2. **分辨率与去噪步数一致**：推理时的 `resize_imgs_with_padding` 和 `num_steps` 应与微调配置匹配，不要拿 512/10 步训出的权重直接硬切 256/3 步而不验证动作偏差。
3. **归一化统计量随权重同步**：确保微调产出的 `dataset_stats` / preprocessor 配置随 checkpoint 一并带到板上（状态/动作使用 `MEAN_STD` 归一化）。
4. **310B NPU 已知限制**：
   - 运行前需 `source /opt/Ascend/ascend-toolkit/set_env.sh`；
   - 不要升级板端驱动（当前 `23.0.rc3`，与 CANN `8.0.RC2` 配套）；
   - 内存总预算按 3.5 GiB 计算，VLA 需常驻暖机（冷启动约 72～75 s），详见 `VLA常驻与多功能低延迟部署计划.md`。

---

## 6. 给后续 Agent 的交接约定

- **修改录制脚本时**：保持本地根目录 `atlas_demo_recorder.py` 与板端 `/root/so101-lekiwi/atlas_demo_recorder.py` 同步，修改后先在本地跑 `python -m pytest tests/test_atlas_demo_recorder.py -v`，再在板上做双摄真机短录验证。
- **不要改动 `/etc/modprobe.d/uvcvideo.conf`**：该文件是双 USB 2.0 摄像头共存的底层前提。
- **当前已确认的下一步高价值工作**：
  1. 将 `atlas_demo_recorder.py` 改造为支持**同一数据集连续追加多条 episode**（或编写单集 LeRobot 数据集合并脚本）；
  2. 正式采集首批 30～50 条目标任务示范并验证数据质量（播放视频抽检 + 检查 `action`/`state` 曲线无跳变）；
  3. 在 GPU 主机完成首次 SmolVLA 微调，并将 checkpoint 回传 Atlas 310B 做真机闭环抓取验收。

