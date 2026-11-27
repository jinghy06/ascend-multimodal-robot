import argparse
import os
import select
import socket
import struct
import sys
import time
from pathlib import Path
from threading import Lock, Thread

from atlas_demo_recorder import EpisodeRecorder, read_json, write_json
from so101_lekiwi_control import ensure_calibration, find_ch343_port, make_robot_config, run_control_loop


JOYSTICK_EVENT = struct.Struct("IhBB")
BUTTON = 1
AXIS = 2
BASE_KEYS = frozenset({"w", "a", "s", "d", "z", "x"})
BASE_AXES = frozenset({0, 1, 3})
AXIS_BINDINGS = {
    0: ("a", "d"),
    1: ("w", "s"),
    3: ("z", "x"),
    4: ("i", "k"),
    6: ("j", "l"),
    7: ("u", "o"),
}
TRIGGER_BINDINGS = {2: "left", 5: "right"}
BUTTON_BINDINGS = {0: "e", 1: "q", 2: "boost", 4: "up", 5: "down"}

HELP_TEXT = """
左摇杆: 底盘前后/平移; 右摇杆左右: 底盘旋转; 右摇杆上下: 肩部抬升
十字键左右: 肩部旋转; 十字键上下: 肘部弯曲
LB/RB: 腕部弯曲; LT/RT: 腕部旋转; A/B: 打开/闭合夹爪
按住 X: 底盘 2 倍加速; Y: 开始/结束示教录制; Start+Y: 放弃当前示教
Start: 锁定底盘停止, 摇杆回中后可重新移动; Back: 退出
"""


class GamepadState:
    def __init__(self, deadzone=12000):
        self.deadzone = deadzone
        self._axis_keys = {}
        self._axis_scales = {}
        self._button_keys = set()
        self._base_stopped = False
        self._running = True
        self._start_held = False
        self._y_held = False
        self._demo_actions = []
        self._lock = Lock()

    @property
    def running(self):
        with self._lock:
            return self._running

    def handle_event(self, event_type, number, value):
        with self._lock:
            if event_type == AXIS:
                self._handle_axis(number, value)
            elif event_type == BUTTON:
                self._handle_button(number, value)

    def _axis_response(self, value):
        magnitude = abs(value)
        if magnitude <= self.deadzone:
            return 0.0
        span = max(1, 32767 - self.deadzone)
        norm = min(1.0, (magnitude - self.deadzone) / span)
        return 0.35 * norm + 0.65 * (norm * norm)

    def _handle_axis(self, number, value):
        keys = set()
        scales = {}
        if number in AXIS_BINDINGS:
            negative, positive = AXIS_BINDINGS[number]
            if value < -self.deadzone:
                keys.add(negative)
                scales[negative] = self._axis_response(value)
            elif value > self.deadzone:
                keys.add(positive)
                scales[positive] = self._axis_response(value)
        elif number in TRIGGER_BINDINGS and value > self.deadzone:
            key = TRIGGER_BINDINGS[number]
            keys.add(key)
            scales[key] = self._axis_response(value)
        self._axis_keys[number] = keys
        self._axis_scales[number] = scales
        if self._base_stopped and not any(self._axis_keys.get(axis) for axis in BASE_AXES):
            self._base_stopped = False

    def _handle_button(self, number, value):
        if number == 6 and value == 1:
            self._running = False
            return
        if number == 7:
            self._start_held = value == 1
            if value == 1:
                self._base_stopped = True
            return
        if number == 3:
            pressed = value == 1
            if pressed and not self._y_held:
                self._demo_actions.append("discard" if self._start_held else "toggle")
            self._y_held = pressed
            return
        key = BUTTON_BINDINGS.get(number)
        if key is not None:
            if value == 1:
                self._button_keys.add(key)
            elif value == 0:
                self._button_keys.discard(key)

    def pop_demo_actions(self):
        with self._lock:
            if not self._demo_actions:
                return []
            actions = list(self._demo_actions)
            self._demo_actions.clear()
            return actions

    def analog_scales(self):
        with self._lock:
            merged = {}
            for axis_map in self._axis_scales.values():
                merged.update(axis_map)
            return merged

    def snapshot(self):
        with self._lock:
            pressed = set(self._button_keys).union(*self._axis_keys.values())
            if self._base_stopped:
                pressed.difference_update(BASE_KEYS)
            return pressed, self._base_stopped


def _send_udp_tts(text, port, host="127.0.0.1"):
    if not port or port <= 0 or not text:
        return
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(text.encode("utf-8"), (host, port))
    except OSError:
        pass
    finally:
        sock.close()


def handle_gamepad_demo_action(action, directory, task="pick up the orange", tts_port=5007):
    directory = Path(directory)
    command_path = directory / "demo_command.json"
    status_path = directory / "demo_status.json"
    current_status = read_json(status_path, {"recording": False})
    current_cmd = read_json(command_path, {"active": False})
    is_recording = bool(current_status.get("recording") or current_cmd.get("active"))

    if action == "toggle":
        if not is_recording:
            task = (task or "").strip() or "pick up the orange"
            base_id = time.strftime("%Y%m%d_%H%M%S")
            episode_id = base_id
            root = directory / "demonstrations" / episode_id
            suffix = 1
            while root.exists():
                episode_id = f"{base_id}_{suffix:02d}"
                root = directory / "demonstrations" / episode_id
                suffix += 1
            write_json(
                status_path,
                {
                    "recording": False,
                    "status": "starting",
                    "episode_id": episode_id,
                    "root": str(root),
                    "frames": 0,
                    "task": task,
                    "error": None,
                },
            )
            write_json(
                command_path,
                {"active": True, "episode_id": episode_id, "root": str(root), "task": task},
            )
            _send_udp_tts("开始示教", tts_port)
            return "started"
        write_json(command_path, {"active": False})
        _send_udp_tts("正在保存示教", tts_port)
        return "stopping"

    if action == "discard":
        if is_recording:
            write_json(command_path, {"active": False, "discard": True})
            _send_udp_tts("已取消本次示教", tts_port)
            return "discarding"
        return "ignored"

    return "ignored"


class GamepadDemoWrapper:
    def __init__(self, recorder, state, directory, task="pick up the orange", tts_port=5007):
        self.recorder = recorder
        self.state = state
        self.directory = Path(directory)
        self.task = task
        self.tts_port = tts_port

    def _drain_demo_actions(self):
        actions = self.state.pop_demo_actions()
        for demo_action in actions:
            handle_gamepad_demo_action(demo_action, self.directory, self.task, self.tts_port)
        if actions and hasattr(self.recorder, "last_command_check"):
            self.recorder.last_command_check = 0.0

    def wants_observation(self):
        self._drain_demo_actions()
        if hasattr(self.recorder, "wants_observation"):
            return self.recorder.wants_observation()
        return True

    def record(self, observation, action):
        self._drain_demo_actions()
        self.recorder.record(observation, action)

    def close(self):
        self.recorder.close()


def decode_joystick_events(data, state):
    for offset in range(0, len(data) - JOYSTICK_EVENT.size + 1, JOYSTICK_EVENT.size):
        _, value, event_type, number = JOYSTICK_EVENT.unpack_from(data, offset)
        event_type &= 0x7F
        if event_type in (BUTTON, AXIS):
            state.handle_event(event_type, number, value)


class JoystickReader:
    def __init__(self, device, state):
        self.device = device
        self.state = state
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
                data = os.read(self._fd, JOYSTICK_EVENT.size * 64)
                if not data:
                    raise OSError("joystick disconnected")
                decode_joystick_events(data, self.state)
        except OSError:
            if self._running:
                self.state.handle_event(BUTTON, 6, 1)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def decode_xbox360_report(data, state):
    if len(data) < 14:
        return
    buttons = int.from_bytes(data[2:4], "little")
    left_x, left_y, right_x, right_y = struct.unpack_from("<hhhh", data, 6)
    for mask, number in (
        (0x1000, 0),
        (0x2000, 1),
        (0x4000, 2),
        (0x8000, 3),
        (0x0100, 4),
        (0x0200, 5),
    ):
        state.handle_event(BUTTON, number, int(bool(buttons & mask)))
    state.handle_event(AXIS, 0, left_x)
    state.handle_event(AXIS, 1, -left_y)
    state.handle_event(AXIS, 2, data[4] * 257)
    state.handle_event(AXIS, 3, right_x)
    state.handle_event(AXIS, 4, -right_y)
    state.handle_event(AXIS, 5, data[5] * 257)
    state.handle_event(AXIS, 6, (int(bool(buttons & 0x0008)) - int(bool(buttons & 0x0004))) * 32767)
    state.handle_event(AXIS, 7, (int(bool(buttons & 0x0002)) - int(bool(buttons & 0x0001))) * 32767)
    for mask, number in ((0x0020, 6), (0x0010, 7)):
        state.handle_event(BUTTON, number, int(bool(buttons & mask)))


class Xbox360USBReader:
    def __init__(self, state):
        self.state = state
        self.device = None
        self.usb = None
        self._thread = None
        self._running = False

    def start(self):
        import libusb_package
        import usb.core
        import usb.util

        self.usb = usb
        self.device = libusb_package.find(idVendor=0x045E, idProduct=0x028E)
        if self.device is None:
            raise RuntimeError("未找到 Xbox 360 USB 手柄 (045e:028e)。")
        try:
            if self.device.is_kernel_driver_active(0):
                self.device.detach_kernel_driver(0)
        except (NotImplementedError, usb.core.USBError):
            pass
        usb.util.claim_interface(self.device, 0)
        self._running = True
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while self._running:
                try:
                    data = self.device.read(0x81, 32, timeout=100)
                except self.usb.core.USBError as exc:
                    if exc.errno == 19:
                        raise
                    continue
                decode_xbox360_report(data, self.state)
        except self.usb.core.USBError:
            if self._running:
                self.state.handle_event(BUTTON, 6, 1)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.device is not None:
            self.usb.util.dispose_resources(self.device)
            self.device = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="使用 Xbox 布局手柄控制 SO101 + LeKiwi。")
    parser.add_argument("--port", default=find_ch343_port() or "/dev/ttyACM0", help="舵机串口")
    parser.add_argument("--input-backend", choices=("usb", "js"), default="usb", help="手柄读取后端")
    parser.add_argument("--joystick-device", default="/dev/input/js0", help="js 后端的 Linux 手柄设备")
    parser.add_argument("--id", default="my_awesome_kiwi", help="机器人 ID / 校准文件名")
    parser.add_argument("--linear-speed", type=float, default=0.10, help="底盘线速度 m/s")
    parser.add_argument("--angular-speed", type=float, default=30.0, help="底盘角速度 deg/s")
    parser.add_argument("--arm-step", type=float, default=0.7, help="每周期关节位置增量")
    parser.add_argument("--rate", type=float, default=30.0, help="遥控控制频率 Hz")
    parser.add_argument("--record-fps", type=int, default=10, help="示教录制帧率 fps")
    parser.add_argument("--deadzone", type=int, default=12000, help="摇杆死区，范围 0–32767")
    parser.add_argument("--demo-task", default="pick up the orange", help="手柄 Y 键录制示教时的默认任务文本")
    parser.add_argument("--tts-udp-port", type=int, default=5007, help="示教提示语音 UDP 端口（0 表示关闭）")
    parser.add_argument("--calibrate", action="store_true", help="显式进入 LeRobot 校准流程")
    args = parser.parse_args(argv)
    for name in ("linear_speed", "angular_speed", "arm_step", "rate", "record_fps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    if not 0 <= args.deadzone < 32767:
        parser.error("--deadzone 必须在 0–32766 之间")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not sys.platform.startswith("linux"):
        raise SystemExit("该原生手柄版本需要在 Linux 上运行。")
    from lerobot.robots.lekiwi import LeKiwi
    from so101_lekiwi_control import configure_bus_retries

    robot = LeKiwi(make_robot_config(args.port, args.id))
    configure_bus_retries(robot)
    try:
        ensure_calibration(robot, args.calibrate)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    state = GamepadState(args.deadzone)
    if args.input_backend == "usb":
        reader = Xbox360USBReader(state)
        input_description = "Xbox 360 USB (045e:028e)"
    else:
        reader = JoystickReader(args.joystick_device, state)
        input_description = args.joystick_device
    print(HELP_TEXT)
    print(f"连接 {args.port}，手柄 {input_description}，机器人 ID: {args.id}")
    base_dir = Path(__file__).resolve().parent
    recorder = EpisodeRecorder(
        robot,
        base_dir / "demo_command.json",
        base_dir / "demo_status.json",
        fps=args.record_fps,
    )
    demo = GamepadDemoWrapper(
        recorder,
        state,
        base_dir,
        task=args.demo_task,
        tts_port=args.tts_udp_port,
    )

    try:
        robot.connect(calibrate=args.calibrate)
        reader.start()
        run_control_loop(
            robot,
            state,
            args.linear_speed,
            args.angular_speed,
            args.arm_step,
            args.rate,
            demo=demo,
        )
    finally:
        demo.close()
        reader.stop()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
