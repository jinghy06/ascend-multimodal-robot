"""Atlas 端无激光雷达纯视觉自主跟随模块（人体跟随 + 红点激光指引）。

核心特性：
- 直接在 Atlas 200I DK A2 昇腾 310B4 NPU 上运行 YOLOv8n-pose OM 模型（~49 FPS），
  与常驻语音唤醒/ASR + Qwen3.5 CPU 共存零 OOM。
- 复用 m1_vision 的 pyACL 封装（OmModel）、后处理（yolo_postprocess）及激光光斑 HSV 检测算法。
- 直接监听 /root/so101-lekiwi/last_voice_command.json 语音指令：
    * follow_start ("跟着我/跟我走/开始跟随") -> 开启纯视觉人体跟随
    * follow_stop  ("停止跟随/别跟了")       -> 停止并释放摄像头与舵机串口
    * follow_adjust("跟近一点/跟远一点")      -> 动态调整目标跟随距离
    * laser_start  ("激光笔模式/指引模式")    -> 开启红点激光指引跟随
    * laser_stop   ("关闭指引模式")           -> 停止并释放硬件
    * stop_all / pause / resume              -> 紧急停止 / 暂停 / 恢复
- 空闲（idle）状态下完全释放 /dev/video0 和舵机串口，与手柄遥控、SmolVLA 数据采集零冲突。
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def open_front_camera(index: int, width: int = 640, height: int = 480, fps: int = 15):
    try:
        from atlas_demo_recorder import open_v4l2_camera

        return open_v4l2_camera(index, width=width, height=height, fps=fps)
    except Exception:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

DEFAULT_M1_VISION_ROOT = Path("/root/ascend-multimodal-robot/m1_ws/src/m1_vision")
DEFAULT_MODEL_PATH = Path("/root/models/yolov8n-pose_640.om")
DEFAULT_VOICE_CMD_PATH = Path("/root/so101-lekiwi/last_voice_command.json")
DEFAULT_STATUS_PATH = Path("/root/so101-lekiwi/visual_follow_status.json")

ARM_JOINTS = (
    "arm_shoulder_pan.pos",
    "arm_shoulder_lift.pos",
    "arm_elbow_flex.pos",
    "arm_wrist_flex.pos",
    "arm_wrist_roll.pos",
    "arm_gripper.pos",
)


@dataclass
class PersonBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class FollowConfig:
    frame_width: int = 640
    frame_height: int = 480
    # 人体跟随参数
    target_h_ratio: float = 0.58
    min_h_ratio: float = 0.35
    max_h_ratio: float = 0.78
    h_ratio_step: float = 0.08
    person_x_deadband: float = 0.08
    person_z_deadband: float = 0.06
    kp_linear: float = 0.45
    kp_angular: float = 55.0
    max_linear_fwd: float = 0.16
    max_linear_rev: float = 0.08
    max_angular: float = 45.0
    # 激光红点跟随参数
    laser_target_v_ratio: float = 0.78
    laser_x_deadband: float = 0.07
    laser_v_deadband: float = 0.05
    laser_kp_linear: float = 0.35
    laser_kp_angular: float = 50.0
    # 平滑与丢失处理
    ema_alpha: float = 0.45
    lost_coast_sec: float = 0.6
    lost_search_sec: float = 3.5
    search_angular: float = 18.0
    enable_search: bool = True


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def box_iou(a: PersonBox, b: PersonBox) -> float:
    xx1 = max(a.x1, b.x1)
    yy1 = max(a.y1, b.y1)
    xx2 = min(a.x2, b.x2)
    yy2 = min(a.y2, b.y2)
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    denom = a.area + b.area - inter
    return inter / denom if denom > 1e-6 else 0.0


def select_person_target(
    detections: Sequence[PersonBox],
    prev_target: Optional[PersonBox],
    frame_width: int,
    frame_height: int,
) -> Optional[PersonBox]:
    """选择要跟随的人体目标：无历史目标时选画面中面积最大者；有历史目标时优先保持跟踪锁定。"""
    if not detections:
        return None
    if prev_target is None:
        return max(detections, key=lambda d: d.area)

    diag = math.hypot(frame_width, frame_height)
    best = None
    best_score = -1.0
    for det in detections:
        iou = box_iou(det, prev_target)
        dist = math.hypot(det.cx - prev_target.cx, det.cy - prev_target.cy) / max(diag, 1.0)
        area_norm = det.area / max(frame_width * frame_height, 1.0)
        # IoU 与中心接近度占主导，面积作为平局与显著度加权
        score = 2.0 * iou + 1.2 * max(0.0, 1.0 - 2.5 * dist) + 0.4 * area_norm
        if score > best_score:
            best_score = score
            best = det
    return best


def compute_person_velocity(
    target: PersonBox,
    cfg: FollowConfig,
) -> tuple[float, float]:
    """根据人体框位置与高度占比计算 (vx m/s, omega deg/s)。

    坐标约定与 so101_lekiwi_control.py 一致：
    - vx > 0 前进, vx < 0 后退
    - omega > 0 左转 (CCW), omega < 0 右转 (CW)
    """
    half_w = 0.5 * cfg.frame_width
    err_x = (half_w - target.cx) / max(half_w, 1.0)  # 目标偏左 -> err_x > 0 -> 左转
    h_ratio = target.height / max(cfg.frame_height, 1.0)
    err_z = cfg.target_h_ratio - h_ratio  # 目标偏远(框小) -> err_z > 0 -> 前进

    if abs(err_x) < cfg.person_x_deadband:
        omega = 0.0
    else:
        eff_x = math.copysign(abs(err_x) - cfg.person_x_deadband, err_x)
        omega = _clamp(cfg.kp_angular * eff_x, -cfg.max_angular, cfg.max_angular)

    if abs(err_z) < cfg.person_z_deadband:
        vx = 0.0
    else:
        eff_z = math.copysign(abs(err_z) - cfg.person_z_deadband, err_z)
        vx = _clamp(cfg.kp_linear * eff_z, -cfg.max_linear_rev, cfg.max_linear_fwd)

    # 目标偏离画面中心过大时先减速转向，避免画大弧线撞障碍
    if vx > 0.0:
        turn_scale = max(0.0, 1.0 - max(0.0, abs(err_x) - 0.15) / 0.55)
        vx *= turn_scale

    return vx, omega


def detect_laser_spot(
    frame_bgr: np.ndarray,
    sat_min: int = 120,
    val_min: int = 150,
    area_min: float = 5.0,
    area_max: float = 2000.0,
    circularity_min: float = 0.55,
) -> Optional[tuple[float, float, float]]:
    """红点激光光斑检测（算法与 m1_vision/laser_spot_node.py 对齐）。

    返回 (u, v, strength) 或 None。
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, sat_min, val_min]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, sat_min, val_min]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    val_channel = hsv[:, :, 2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_peak = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (area_min <= area <= area_max):
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < circularity_min:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi_mask = mask[y : y + h, x : x + w]
        roi_val = val_channel[y : y + h, x : x + w].astype(np.float32)
        roi_val[roi_mask == 0] = 0.0
        total = float(roi_val.sum())
        if total <= 0.0:
            continue
        ys, xs = np.mgrid[y : y + h, x : x + w]
        u = float((roi_val * xs).sum() / total)
        v = float((roi_val * ys).sum() / total)
        peak = float(roi_val.max())
        if peak > best_peak:
            best_peak = peak
            best = (u, v, peak / 255.0)
    return best


def compute_laser_velocity(
    spot: tuple[float, float, float],
    cfg: FollowConfig,
) -> tuple[float, float]:
    """根据红点像素坐标 (u, v, strength) 计算 (vx, omega)。"""
    u, v, _ = spot
    half_w = 0.5 * cfg.frame_width
    err_x = (half_w - u) / max(half_w, 1.0)
    v_ratio = v / max(cfg.frame_height, 1.0)
    err_v = cfg.laser_target_v_ratio - v_ratio  # 光斑偏上(较远) -> err_v > 0 -> 前进

    if abs(err_x) < cfg.laser_x_deadband:
        omega = 0.0
    else:
        eff_x = math.copysign(abs(err_x) - cfg.laser_x_deadband, err_x)
        omega = _clamp(cfg.laser_kp_angular * eff_x, -cfg.max_angular, cfg.max_angular)

    if err_v <= cfg.laser_v_deadband:
        vx = 0.0
    else:
        eff_v = err_v - cfg.laser_v_deadband
        vx = _clamp(cfg.laser_kp_linear * eff_v, 0.0, cfg.max_linear_fwd)
        turn_scale = max(0.0, 1.0 - max(0.0, abs(err_x) - 0.15) / 0.55)
        vx *= turn_scale

    return vx, omega


class FollowController:
    """带语音状态机、速度平滑与目标丢失恢复的纯视觉跟随控制器（纯逻辑，可独立单测）。"""

    def __init__(self, cfg: Optional[FollowConfig] = None, initial_mode: str = "idle"):
        self.cfg = cfg or FollowConfig()
        self.mode = initial_mode  # "idle" | "person" | "laser"
        self.last_active_mode: Optional[str] = initial_mode if initial_mode != "idle" else None
        self.prev_target: Optional[PersonBox] = None
        self.last_seen_time: float = 0.0
        self.last_err_x: float = 0.0
        self.lost_notified: bool = False
        self.vx: float = 0.0
        self.omega: float = 0.0

    def reset_motion(self) -> None:
        self.prev_target = None
        self.vx = 0.0
        self.omega = 0.0
        self.lost_notified = False

    def apply_voice_command(self, record: dict) -> Optional[str]:
        """解析 last_voice_command.json，返回可选语音播报提示。"""
        task_id = record.get("task_id") or ""
        params = record.get("params") or {}
        if task_id == "follow_start":
            self.mode = "person"
            self.last_active_mode = "person"
            self.reset_motion()
            return None
        if task_id == "follow_stop":
            self.mode = "idle"
            self.reset_motion()
            return None
        if task_id == "follow_adjust":
            dist = params.get("distance")
            if dist == "near":
                self.cfg.target_h_ratio = _clamp(
                    self.cfg.target_h_ratio + self.cfg.h_ratio_step,
                    self.cfg.min_h_ratio,
                    self.cfg.max_h_ratio,
                )
            elif dist == "far":
                self.cfg.target_h_ratio = _clamp(
                    self.cfg.target_h_ratio - self.cfg.h_ratio_step,
                    self.cfg.min_h_ratio,
                    self.cfg.max_h_ratio,
                )
            return None
        if task_id == "laser_start":
            self.mode = "laser"
            self.last_active_mode = "laser"
            self.reset_motion()
            return None
        if task_id == "laser_stop":
            self.mode = "idle"
            self.reset_motion()
            return None
        if task_id in ("stop_all", "pause"):
            if task_id == "stop_all":
                self.last_active_mode = None
            self.mode = "idle"
            self.reset_motion()
            return None
        if task_id == "resume" and self.last_active_mode:
            self.mode = self.last_active_mode
            self.reset_motion()
            return None
        return None

    def step_person(
        self,
        detections: Sequence[PersonBox],
        now: float,
    ) -> tuple[float, float, Optional[PersonBox], Optional[str]]:
        """返回 (vx, omega, locked_target, tts_alert)。"""
        target = select_person_target(
            detections, self.prev_target, self.cfg.frame_width, self.cfg.frame_height
        )
        if target is not None:
            self.prev_target = target
            self.last_seen_time = now
            self.lost_notified = False
            half_w = 0.5 * self.cfg.frame_width
            self.last_err_x = (half_w - target.cx) / max(half_w, 1.0)
            raw_vx, raw_omega = compute_person_velocity(target, self.cfg)
            self.vx = self.cfg.ema_alpha * raw_vx + (1.0 - self.cfg.ema_alpha) * self.vx
            self.omega = self.cfg.ema_alpha * raw_omega + (1.0 - self.cfg.ema_alpha) * self.omega
            return self.vx, self.omega, target, None

        return self._handle_lost(now, "没看到人，先停一下")

    def step_laser(
        self,
        spot: Optional[tuple[float, float, float]],
        now: float,
    ) -> tuple[float, float, Optional[tuple[float, float, float]], Optional[str]]:
        if spot is not None:
            self.last_seen_time = now
            self.lost_notified = False
            half_w = 0.5 * self.cfg.frame_width
            self.last_err_x = (half_w - spot[0]) / max(half_w, 1.0)
            raw_vx, raw_omega = compute_laser_velocity(spot, self.cfg)
            self.vx = self.cfg.ema_alpha * raw_vx + (1.0 - self.cfg.ema_alpha) * self.vx
            self.omega = self.cfg.ema_alpha * raw_omega + (1.0 - self.cfg.ema_alpha) * self.omega
            return self.vx, self.omega, spot, None

        return self._handle_lost(now, None)

    def _handle_lost(
        self, now: float, lost_msg: Optional[str]
    ) -> tuple[float, float, None, Optional[str]]:
        if self.last_seen_time <= 0.0:
            self.vx = 0.0
            self.omega = 0.0
            return 0.0, 0.0, None, None
        lost_dt = now - self.last_seen_time
        if lost_dt < self.cfg.lost_coast_sec:
            self.vx *= 0.5
            self.omega *= 0.5
            return self.vx, self.omega, None, None
        if self.cfg.enable_search and lost_dt < self.cfg.lost_search_sec and abs(self.last_err_x) > 0.12:
            self.vx = 0.0
            self.omega = math.copysign(self.cfg.search_angular, self.last_err_x)
            return self.vx, self.omega, None, None
        self.vx = 0.0
        self.omega = 0.0
        self.prev_target = None
        alert = None
        if lost_msg and not self.lost_notified:
            self.lost_notified = True
            alert = lost_msg
        return 0.0, 0.0, None, alert


def _ensure_aicpu_cgroup() -> None:
    """Atlas 200I DK A2 上 systemctl daemon-reload 可能清理空的 AICPU cpuset cgroup，
    导致 acl.rt.set_device 返回 507033；启动 NPU 前自动确保该目录存在。"""
    aicpu = Path("/sys/fs/cgroup/cpuset/AICPU")
    ctrl = Path("/sys/fs/cgroup/cpuset/CtrlCPU")
    if not ctrl.is_dir() or aicpu.is_dir():
        return
    try:
        aicpu.mkdir(parents=True, exist_ok=True)
        (aicpu / "cpuset.cpus").write_text("3\n", encoding="utf-8")
        (aicpu / "cpuset.mems").write_text("0\n", encoding="utf-8")
    except OSError:
        pass


class NpuPoseDetector:
    """复用 m1_vision 的 pyACL OmModel + yolo_postprocess 进行人体框检测。"""

    def __init__(
        self,
        model_path: Path,
        m1_vision_root: Path = DEFAULT_M1_VISION_ROOT,
        input_size: int = 640,
        conf_threshold: float = 0.45,
        nms_threshold: float = 0.45,
        device_id: int = 0,
    ):
        _ensure_aicpu_cgroup()
        root_str = str(m1_vision_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        from m1_vision import yolo_postprocess as yp
        from m1_vision.acl_inference import OmModel

        self._yp = yp
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.model = OmModel(str(model_path), device_id=device_id)

    def detect(self, frame_bgr: np.ndarray) -> list[PersonBox]:
        img_lb, ratio, dwdh = self._yp.letterbox(frame_bgr, self.input_size)
        output = self.model.infer(self._yp.to_input_tensor(img_lb))[0]
        boxes, scores, kpts = self._yp.decode_pose_output(output, self.conf_threshold)
        if len(boxes) == 0:
            return []
        xyxy = self._yp.cxcywh_to_xyxy(boxes)
        keep = self._yp.nms(xyxy, scores, self.nms_threshold, max_det=10)
        if len(keep) == 0:
            return []
        scaled_boxes, _ = self._yp.scale_back(xyxy[keep], kpts[keep], ratio, dwdh, frame_bgr.shape)
        kept_scores = scores[keep]
        return [
            PersonBox(
                x1=float(b[0]),
                y1=float(b[1]),
                x2=float(b[2]),
                y2=float(b[3]),
                score=float(s),
            )
            for b, s in zip(scaled_boxes, kept_scores)
        ]

    def close(self) -> None:
        self.model.close()


class LeKiwiBaseDriver:
    """仅驱动 LeKiwi 底盘速度（保持机械臂在初始位置不动），释放时自动停轮断连。"""

    def __init__(self, port: Optional[str] = None, robot_id: str = "my_awesome_kiwi", dry_run: bool = False):
        self.port = port
        self.robot_id = robot_id
        self.dry_run = dry_run
        self.robot = None
        self.arm_hold: dict[str, float] = {}

    @property
    def connected(self) -> bool:
        return self.dry_run or (self.robot is not None and self.robot.is_connected)

    def connect(self) -> None:
        if self.dry_run or self.connected:
            return
        from lerobot.robots.lekiwi import LeKiwi
        from so101_lekiwi_control import configure_bus_retries, ensure_calibration, find_ch343_port, make_robot_config

        port = self.port or find_ch343_port()
        if not port:
            raise RuntimeError("未找到 LeKiwi 舵机串口 (CH343)")
        robot = LeKiwi(make_robot_config(port, self.robot_id))
        configure_bus_retries(robot)
        ensure_calibration(robot, calibrate=False)
        robot.connect(calibrate=False)
        obs = robot.get_observation()
        self.arm_hold = {joint: float(obs[joint]) for joint in ARM_JOINTS if joint in obs}
        self.robot = robot

    def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        if self.dry_run or not self.connected:
            return
        action = {
            **self.arm_hold,
            "x.vel": float(vx),
            "y.vel": float(vy),
            "theta.vel": float(omega),
        }
        self.robot.send_action(action)

    def stop_and_disconnect(self) -> None:
        if self.robot is None:
            return
        try:
            if self.robot.is_connected:
                self.robot.stop_base()
        except Exception as exc:
            print(f"[FOLLOW] stop_base error: {exc}", flush=True)
        try:
            if self.robot.is_connected:
                self.robot.disconnect()
        except Exception as exc:
            print(f"[FOLLOW] disconnect error: {exc}", flush=True)
        self.robot = None
        self.arm_hold = {}


def send_esp_tts(text: str, host: str = "127.0.0.1", port: int = 5007) -> None:
    text = (text or "").strip()
    if not text:
        return
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(text.encode("utf-8"), (host, port))
    except OSError:
        pass
    finally:
        sock.close()


def _is_gamepad_running() -> bool:
    """检查 so101_lekiwi_gamepad.py 是否正在占用底盘串口。"""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return False
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_text(errors="ignore")
        except OSError:
            continue
        if "so101_lekiwi_gamepad.py" in cmdline:
            return True
    return False


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Atlas 无激光雷达纯视觉跟随（NPU 人体跟随 + 红点指引）")
    parser.add_argument("--mode", choices=("voice", "person", "laser"), default="voice", help="启动模式：voice=常驻等待语音指令, person=立即人体跟随, laser=立即红点跟随")
    parser.add_argument("--camera", type=int, default=0, help="前视摄像头 V4L2 索引（默认 0）")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="YOLOv8n-pose OM 模型路径")
    parser.add_argument("--m1-vision-root", default=str(DEFAULT_M1_VISION_ROOT), help="m1_vision 源码目录")
    parser.add_argument("--port", default=None, help="LeKiwi 舵机串口（留空自动查找 CH343）")
    parser.add_argument("--id", default="my_awesome_kiwi", help="LeRobot 校准 ID")
    parser.add_argument("--voice-cmd", default=str(DEFAULT_VOICE_CMD_PATH), help="语音指令 JSON 路径")
    parser.add_argument("--status-out", default=str(DEFAULT_STATUS_PATH), help="状态输出 JSON 路径")
    parser.add_argument("--rate", type=float, default=15.0, help="视觉跟随控制频率 Hz")
    parser.add_argument("--target-h-ratio", type=float, default=0.58, help="目标人体框占画面高度比")
    parser.add_argument("--max-linear", type=float, default=0.16, help="最大前进线速度 m/s")
    parser.add_argument("--max-angular", type=float, default=45.0, help="最大转向角速度 deg/s")
    parser.add_argument("--no-search", action="store_true", help="目标丢失时不原地慢转寻找")
    parser.add_argument("--dry-run", action="store_true", help="仅运行相机与 NPU 检测并打印速度，不连接底盘电机")
    parser.add_argument("--max-frames", type=int, default=0, help=">0 时运行指定帧数后退出（便于测试）")
    args = parser.parse_args(argv)

    cfg = FollowConfig(
        target_h_ratio=args.target_h_ratio,
        max_linear_fwd=args.max_linear,
        max_angular=args.max_angular,
        enable_search=not args.no_search,
    )
    initial_mode = "idle" if args.mode == "voice" else args.mode
    controller = FollowController(cfg, initial_mode=initial_mode)
    driver = LeKiwiBaseDriver(port=args.port, robot_id=args.id, dry_run=args.dry_run)

    voice_cmd_path = Path(args.voice_cmd)
    status_path = Path(args.status_out)
    last_cmd_ts = float(read_json(voice_cmd_path, {}).get("timestamp", 0.0))

    detector: Optional[NpuPoseDetector] = None
    capture = None
    running = True

    def _handle_stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    def _release_hw():
        nonlocal capture
        driver.stop_and_disconnect()
        if capture is not None:
            capture.release()
            capture = None

    def _ensure_hw(need_npu: bool) -> bool:
        nonlocal capture, detector
        if not args.dry_run and not driver.connected:
            if _is_gamepad_running():
                print("[FOLLOW] Gamepad process is active; cannot claim servo bus", flush=True)
                send_esp_tts("请先关闭手柄遥控")
                return False
            driver.connect()
            print(f"[FOLLOW] LeKiwi connected (id={args.id})", flush=True)
        if capture is None:
            cap = open_front_camera(args.camera, width=cfg.frame_width, height=cfg.frame_height, fps=15)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"无法打开摄像头 /dev/video{args.camera}")
            capture = cap
            print(f"[FOLLOW] Camera /dev/video{args.camera} opened", flush=True)
        if need_npu and detector is None:
            detector = NpuPoseDetector(Path(args.model), Path(args.m1_vision_root))
            print(f"[FOLLOW] NPU YOLOv8n-pose ready: {args.model}", flush=True)
        return True

    period = 1.0 / max(args.rate, 1.0)
    frames_done = 0
    last_log = 0.0
    print(f"[FOLLOW] Service ready (mode={args.mode}, dry_run={args.dry_run})", flush=True)
    write_json(status_path, {"mode": controller.mode, "active": False, "vx": 0.0, "omega": 0.0, "error": None})

    try:
        while running:
            t0 = time.monotonic()
            # 1. 轮询语音指令
            if args.mode == "voice":
                rec = read_json(voice_cmd_path, {})
                ts = float(rec.get("timestamp", 0.0))
                if ts > last_cmd_ts:
                    last_cmd_ts = ts
                    prev_mode = controller.mode
                    controller.apply_voice_command(rec)
                    if controller.mode != prev_mode:
                        print(f"[FOLLOW] Voice switch: {prev_mode} -> {controller.mode} (task={rec.get('task_id')})", flush=True)
                        if controller.mode == "idle":
                            _release_hw()
                            write_json(status_path, {"mode": "idle", "active": False, "vx": 0.0, "omega": 0.0, "error": None})

            if controller.mode == "idle":
                time.sleep(0.15)
                continue

            # 2. 按需打开硬件与 NPU
            try:
                if not _ensure_hw(need_npu=(controller.mode == "person")):
                    controller.mode = "idle"
                    _release_hw()
                    continue
            except Exception as exc:
                print(f"[FOLLOW] Hardware init error: {exc}", flush=True)
                send_esp_tts("视觉跟随启动失败")
                write_json(status_path, {"mode": "idle", "active": False, "vx": 0.0, "omega": 0.0, "error": str(exc)})
                controller.mode = "idle"
                _release_hw()
                if args.mode != "voice":
                    break
                continue

            # 3. 读帧 + 推理 + 控制
            ok, frame = capture.read()
            if not ok or frame is None:
                driver.send_velocity(0.0, 0.0, 0.0)
                time.sleep(0.05)
                continue

            now = time.monotonic()
            target_found = False
            alert = None
            if controller.mode == "person":
                dets = detector.detect(frame)
                vx, omega, locked, alert = controller.step_person(dets, now)
                target_found = locked is not None
            else:
                spot = detect_laser_spot(frame)
                vx, omega, locked_spot, alert = controller.step_laser(spot, now)
                target_found = locked_spot is not None

            driver.send_velocity(vx, 0.0, omega)
            if alert:
                send_esp_tts(alert)

            frames_done += 1
            if now - last_log >= 0.5:
                dt_ms = (time.monotonic() - t0) * 1000.0
                print(
                    f"[FOLLOW] mode={controller.mode} found={target_found} "
                    f"vx={vx:+.3f} m/s omega={omega:+.1f} deg/s ({dt_ms:.1f} ms)",
                    flush=True,
                )
                write_json(
                    status_path,
                    {
                        "mode": controller.mode,
                        "active": True,
                        "target_found": target_found,
                        "vx": round(vx, 4),
                        "omega": round(omega, 2),
                        "target_h_ratio": round(cfg.target_h_ratio, 3),
                        "frame_ms": round(dt_ms, 1),
                        "error": None,
                    },
                )
                last_log = now

            if args.max_frames > 0 and frames_done >= args.max_frames:
                break

            sleep_for = period - (time.monotonic() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        _release_hw()
        if detector is not None:
            detector.close()
        write_json(status_path, {"mode": "idle", "active": False, "vx": 0.0, "omega": 0.0, "error": None})
        print("[FOLLOW] Stopped cleanly", flush=True)


if __name__ == "__main__":
    main()
