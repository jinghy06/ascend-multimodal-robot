#!/usr/bin/env bash
# Atlas 200I DK A2 环境搭建（W1 任务，对应设计文档 6.1 节）
# 用法：逐段人工执行并核对输出，不要用 set -e 一把梭；W1 结束把实际版本号记入 README。
set -uo pipefail

echo "== 1) 驱动与固件（昇腾社区下载 Atlas 200I DK A2 配套 run 包，版本号按实际替换）=="
echo "  sudo ./Ascend-hdk-310b-npu-driver_<ver>_linux-aarch64.run --full"
echo "  sudo ./Ascend-hdk-310b-npu-firmware_<ver>.run --full"
echo "  npu-smi info   # 验证：应看到 310B4 与温度/功耗"

echo "== 2) CANN toolkit（版本与驱动严格配套，装完锁定）=="
echo "  ./Ascend-cann-toolkit_<ver>_linux-aarch64.run --install"
echo "  echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc"

echo "== 3) ROS 2 Humble（Ubuntu 22.04）=="
echo "  sudo apt install ros-humble-desktop ros-humble-cv-bridge ros-humble-vision-msgs \\"
echo "    ros-humble-tf2-geometry-msgs python3-colcon-common-extensions python3-opencv"

echo "== 4) 构建本工作区 =="
echo "  cd ~/m1_ws && colcon build --symlink-install && source install/setup.bash"

echo "== 5) 验证官方 YOLO 样例（昇腾 samples），锁定 CANN 版本后写入 README =="
