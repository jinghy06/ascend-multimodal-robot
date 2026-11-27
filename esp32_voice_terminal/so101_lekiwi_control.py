import argparse
import glob
import os
import select
import struct
import sys
import time
from threading import Lock, Thread


EVDEV_EVENT = struct.Struct("@llHHI")
EVDEV_KEY_MAP = {
    1: "esc",
    16: "q",
    17: "w",
    18: "e",
    22: "u",
    23: "i",
    24: "o",
    30: "a",
    31: "s",
    32: "d",
    36: "j",
    37: "k",
    38: "l",
    44: "z",
    45: "x",
    57: "space",
    103: "up",
    105: "left",
    106: "right",
    108: "down",
}
BASE_KEYS = frozenset({"w", "a", "s", "d", "z", "x"})
ARM_BINDINGS = {
    "arm_shoulder_pan.pos": ("j", "l"),
    "arm_shoulder_lift.pos": ("i", "k"),
    "arm_elbow_flex.pos": ("u", "o"),
    "arm_wrist_flex.pos": ("up", "down"),
    "arm_wrist_roll.pos": ("left", "right"),
    "arm_gripper.pos": ("e", "q"),
}

HELP_TEXT = """
底盘: W/S 前后, A/D 左右平移, Z/X 左右旋转
机械臂: J/L 肩旋转, I/K 肩抬升, U/O 肘弯曲
       方向键上下 腕弯曲, 方向键左右 腕旋转, E/Q 打开/闭合夹爪
停止: Space 锁定底盘停止; 松开后按新的底盘移动键可重新启动
退出: Esc
"""


def decode_evdev_events(data, on_press, on_release):
    for offset in range(0, len(data) - EVDEV_EVENT.size + 1, EVDEV_EVENT.size):
        _, _, event_type, code, value = EVDEV_EVENT.unpack_from(data, offset)
        key = EVDEV_KEY_MAP.get(code) if event_type == 1 else None
        if key is None:
            continue
        if value in (1, 2):
            on_press(key)
        elif value == 0:
            on_release(key)


class EvdevKeyboard:
    def __init__(self, device, on_press, on_release):
        self.device = device
        self.on_press = on_press
        self.on_release = on_release
        self._fd = None
        self._thread = None
        self._running = False

    def start(self):
        self._fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        self._running = True
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while self._running:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    continue
                data = os.read(self._fd, EVDEV_EVENT.size * 64)
                if not data:
                    raise OSError("keyboard disconnected")
                decode_evdev_events(data, self.on_press, self.on_release)
        except OSError:
            if self._running:
                self.on_press("esc")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class PynputKeyboard:
    def __init__(self, on_press, on_release):
        from pynput import keyboard

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()


def make_keyboard_listener(device, on_press, on_release):
    if sys.platform.startswith("linux"):
        return EvdevKeyboard(device, on_press, on_release)
    return PynputKeyboard(on_press, on_release)


def _normalize_key(key):
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    return getattr(key, "name", key)


class KeyboardState:
    def __init__(self):
        self._pressed = set()
        self._base_stopped = False
        self._blocked_base_keys = set()
        self._running = True
        self._lock = Lock()

    @property
    def running(self):
        with self._lock:
            return self._running

    def on_press(self, key):
        key = _normalize_key(key)
        with self._lock:
            if key == "esc":
                self._running = False
                return
            if key == "space":
                self._base_stopped = True
                self._blocked_base_keys = self._pressed & BASE_KEYS
                self._pressed.difference_update(BASE_KEYS)
                return
            if key in BASE_KEYS:
                if key in self._blocked_base_keys:
                    return
                self._base_stopped = False
            self._pressed.add(key)

    def on_release(self, key):
        key = _normalize_key(key)
        with self._lock:
            self._pressed.discard(key)
            self._blocked_base_keys.discard(key)

    def snapshot(self):
        with self._lock:
            return set(self._pressed), self._base_stopped


def _axis(pressed, positive, negative, scales=None):
    if scales is None:
        return float(positive in pressed) - float(negative in pressed)
    pos = float(scales.get(positive, 1.0)) if positive in pressed else 0.0
    neg = float(scales.get(negative, 1.0)) if negative in pressed else 0.0
    return pos - neg


def base_velocity(pressed, stopped, linear_speed, angular_speed, scales=None):
    if stopped:
        return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
    multiplier = 2.0 if "boost" in pressed else 1.0
    return {
        "x.vel": _axis(pressed, "w", "s", scales) * linear_speed * multiplier,
        "y.vel": _axis(pressed, "a", "d", scales) * linear_speed * multiplier,
        "theta.vel": _axis(pressed, "z", "x", scales) * angular_speed * multiplier,
    }


def arm_targets(observation, pressed, step, scales=None):
    pressed = {_normalize_key(key) for key in pressed}
    targets = {}
    for joint, (positive, negative) in ARM_BINDINGS.items():
        minimum, maximum = (0.0, 100.0) if joint == "arm_gripper.pos" else (-100.0, 100.0)
        value = float(observation[joint]) + _axis(pressed, positive, negative, scales) * step
        targets[joint] = min(maximum, max(minimum, value))
    return targets


def ensure_calibration(robot, calibrate):
    if not calibrate and not robot.calibration_fpath.is_file():
        raise RuntimeError(
            f"缺少 calibration 文件: {robot.calibration_fpath}\n"
            "请使用已有校准文件的 --id，或先检查设备后显式使用 --calibrate。"
        )


def run_control_loop(robot, controls, linear_speed, angular_speed, arm_step, rate, demo=None):
    period = 1.0 / rate
    try:
        targets = robot.get_observation()
        while controls.running:
            started = time.monotonic()
            if demo is None:
                need_obs = False
            elif hasattr(demo, "wants_observation"):
                need_obs = bool(demo.wants_observation())
            else:
                need_obs = True
            observation = robot.get_observation() if need_obs else targets
            pressed, stopped = controls.snapshot()
            scales = controls.analog_scales() if hasattr(controls, "analog_scales") else None
            targets = arm_targets(targets, pressed, arm_step, scales=scales)
            action = {
                **targets,
                **base_velocity(pressed, stopped, linear_speed, angular_speed, scales=scales),
            }
            actual_action = robot.send_action(action)
            if demo is not None:
                demo.record(observation, actual_action)
            remaining = period - (time.monotonic() - started)
            if remaining > 0 and controls.running:
                time.sleep(remaining)
    finally:
        try:
            robot.stop_base()
        finally:
            robot.disconnect()


def find_ch343_port():
    matches = sorted(glob.glob("/dev/serial/by-id/usb-1a86_USB_Single_Serial_*-if00"))
    return matches[0] if matches else None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="通过 LeRobot 键盘控制 SO101 + LeKiwi（默认不启用摄像头）。"
    )
    default_port = (find_ch343_port() or "/dev/ttyACM0") if sys.platform.startswith("linux") else "COM9"
    parser.add_argument("--port", default=default_port, help=f"串口，默认 {default_port}")
    parser.add_argument("--keyboard-device", default="/dev/input/event0", help="Linux 键盘 event 节点")
    parser.add_argument("--id", default="my_awesome_kiwi", help="机器人 ID / 校准文件名")
    parser.add_argument("--linear-speed", type=float, default=0.10, help="底盘线速度 m/s")
    parser.add_argument("--angular-speed", type=float, default=30.0, help="底盘角速度 deg/s")
    parser.add_argument("--arm-step", type=float, default=0.8, help="每周期关节位置增量")
    parser.add_argument("--rate", type=float, default=20.0, help="控制频率 Hz")
    parser.add_argument("--calibrate", action="store_true", help="显式进入 LeRobot 校准流程")
    args = parser.parse_args(argv)
    for name in ("linear_speed", "angular_speed", "arm_step", "rate"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    return args


def make_robot_config(port, robot_id):
    from lerobot.robots.lekiwi import LeKiwiConfig

    return LeKiwiConfig(
        port=port,
        id=robot_id,
        cameras={},
        max_relative_target=None,
    )


def configure_bus_retries(robot, retries=3):
    original_write = robot.bus.write

    def write_with_retry(*args, **kwargs):
        kwargs["num_retry"] = max(retries, kwargs.get("num_retry", 0))
        return original_write(*args, **kwargs)

    robot.bus.write = write_with_retry


def main(argv=None):
    args = parse_args(argv)
    from lerobot.robots.lekiwi import LeKiwi

    config = make_robot_config(args.port, args.id)
    robot = LeKiwi(config)
    configure_bus_retries(robot)
    try:
        ensure_calibration(robot, args.calibrate)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    controls = KeyboardState()
    listener = make_keyboard_listener(args.keyboard_device, controls.on_press, controls.on_release)
    print(HELP_TEXT)
    print(f"连接 {args.port}，机器人 ID: {args.id}，控制频率: {args.rate:g} Hz")
    if sys.platform.startswith("linux"):
        print(f"键盘设备: {args.keyboard_device}")

    try:
        robot.connect(calibrate=args.calibrate)
        listener.start()
        run_control_loop(
            robot,
            controls,
            args.linear_speed,
            args.angular_speed,
            args.arm_step,
            args.rate,
        )
    finally:
        listener.stop()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
