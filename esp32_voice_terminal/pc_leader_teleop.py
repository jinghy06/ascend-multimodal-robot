"""PC-side wireless SO-101 Leader Arm + left-hand Keyboard sender for LeKiwi."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from typing import Any, Sequence

from atlas_leader_host import ARM_JOINTS, DEFAULT_UDP_PORT, map_leader_to_lekiwi_arm
from so101_lekiwi_control import BASE_KEYS, _normalize_key, base_velocity

HELP_TEXT = """
右手: SO-101 主动臂直接带动从臂与夹爪
左手底盘: W/S 前后, A/D 左右平移, Z/X 左右旋转, 按住 Shift 2 倍加速
示教录制: R 开始/结束保存示教, Backspace 放弃当前录制(不编码)
安全控制: Space 急停锁定底盘(松开后按新方向解锁), Esc 退出
"""


def load_so101_leader_api():
    """Support both lerobot 0.4.x (so_leader) and 0.3.x (so101_leader) import paths."""
    try:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        return SO101Leader, SO101LeaderConfig
    except ImportError:
        from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

        return SO101Leader, SO101LeaderConfig


def list_serial_ports() -> list[tuple[str, str]]:
    from serial.tools import list_ports

    return [(p.device, p.description) for p in sorted(list_ports.comports(), key=lambda x: x.device)]


class LeaderBaseKeyboardState:
    """Tracks WASDZX base movement, Shift boost, Space stop-lock, and R/Backspace demo hotkeys."""

    def __init__(self):
        self._pressed: set[str] = set()
        self._held_hotkeys: set[str] = set()
        self._blocked_base_keys: set[str] = set()
        self._base_stopped = False
        self._running = True
        self._demo_actions: list[str] = []
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def on_press(self, key: Any) -> None:
        k = _normalize_key(key)
        with self._lock:
            if k == "esc":
                self._running = False
                return
            if k in {"shift", "shift_l", "shift_r"}:
                self._pressed.add("boost")
                return
            if k == "space":
                self._base_stopped = True
                self._blocked_base_keys = self._pressed & BASE_KEYS
                self._pressed.difference_update(BASE_KEYS)
                return
            if k == "r":
                if "r" not in self._held_hotkeys:
                    self._demo_actions.append("toggle")
                self._held_hotkeys.add("r")
                return
            if k == "backspace":
                if "backspace" not in self._held_hotkeys:
                    self._demo_actions.append("discard")
                self._held_hotkeys.add("backspace")
                return
            if k in BASE_KEYS:
                if k in self._blocked_base_keys:
                    return
                self._base_stopped = False
            self._pressed.add(k)

    def on_release(self, key: Any) -> None:
        k = _normalize_key(key)
        with self._lock:
            if k in {"shift", "shift_l", "shift_r"}:
                self._pressed.discard("boost")
                return
            self._held_hotkeys.discard(k)
            self._pressed.discard(k)
            self._blocked_base_keys.discard(k)

    def pop_demo_actions(self) -> list[str]:
        with self._lock:
            if not self._demo_actions:
                return []
            actions = list(self._demo_actions)
            self._demo_actions.clear()
            return actions

    def snapshot(self) -> tuple[set[str], bool]:
        with self._lock:
            return set(self._pressed), self._base_stopped


class _MockLeader:
    def get_action(self) -> dict[str, float]:
        t = time.monotonic()
        return {
            "shoulder_pan.pos": 10.0 * float(np.sin(t)) if "np" in globals() else 0.0,
            "shoulder_lift.pos": -15.0,
            "elbow_flex.pos": 25.0,
            "wrist_flex.pos": -10.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 30.0,
        }

    def disconnect(self) -> None:
        pass


def run_sender_loop(
    leader,
    controls: LeaderBaseKeyboardState,
    atlas_ip: str,
    udp_port: int = DEFAULT_UDP_PORT,
    linear_speed: float = 0.10,
    angular_speed: float = 30.0,
    rate: float = 30.0,
    task: str = "pick up the orange",
    max_seconds: float = 0.0,
    quiet: bool = False,
) -> dict[str, Any]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    target = (atlas_ip, int(udp_port))
    period = 1.0 / max(1.0, float(rate))
    started = time.monotonic()
    seq = 0
    acks = 0
    last_ack_payload: dict[str, Any] | None = None
    last_print = 0.0
    last_rec_state: Any = None

    try:
        while controls.running:
            t0 = time.monotonic()
            if max_seconds > 0.0 and (t0 - started) >= max_seconds:
                break

            raw_arm = leader.get_action()
            arm = map_leader_to_lekiwi_arm(raw_arm)
            pressed, stopped = controls.snapshot()
            base = base_velocity(pressed, stopped, linear_speed, angular_speed)
            demo_actions = controls.pop_demo_actions()

            if demo_actions:
                for da in demo_actions:
                    seq += 1
                    pkt = {"seq": seq, "ts": time.time(), "arm": arm, "base": base, "demo_action": da, "task": task}
                    sock.sendto(json.dumps(pkt, ensure_ascii=False).encode("utf-8"), target)
            else:
                seq += 1
                pkt = {"seq": seq, "ts": time.time(), "arm": arm, "base": base, "task": task}
                sock.sendto(json.dumps(pkt, ensure_ascii=False).encode("utf-8"), target)

            while True:
                try:
                    data, _ = sock.recvfrom(4096)
                except BlockingIOError:
                    break
                except OSError:
                    break
                try:
                    ack = json.loads(data.decode("utf-8", "replace"))
                    acks += 1
                    last_ack_payload = ack
                except json.JSONDecodeError:
                    continue

            now = time.monotonic()
            rec_state = (last_ack_payload or {}).get("status")
            if not quiet and last_ack_payload is not None and (rec_state != last_rec_state or now - last_print >= 1.0):
                last_rec_state = rec_state
                last_print = now
                print(
                    f"[TELEOP] seq={seq} acks={acks} status={last_ack_payload.get('status')} "
                    f"frames={last_ack_payload.get('frames')} stale={last_ack_payload.get('stale')}",
                    flush=True,
                )

            sleep_for = period - (time.monotonic() - t0)
            if sleep_for > 0.0 and controls.running:
                time.sleep(sleep_for)
    finally:
        seq += 1
        zero_pkt = {
            "seq": seq,
            "ts": time.time(),
            "base": {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0},
            "stop": True,
        }
        try:
            sock.sendto(json.dumps(zero_pkt).encode("utf-8"), target)
        except OSError:
            pass
        sock.close()

    return {"sent": seq, "acks": acks, "last_ack": last_ack_payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PC-side wireless SO-101 Leader + Keyboard sender")
    parser.add_argument("--list-ports", action="store_true", help="列出可用串口并退出")
    parser.add_argument("--atlas-ip", default=None, help="Atlas 主控板 IP 地址")
    parser.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    parser.add_argument("--leader-port", default=None, help="SO-101 主动臂串口（如 COM7 或 /dev/ttyACM0）")
    parser.add_argument("--leader-id", default="my_so101_leader")
    parser.add_argument("--calibrate", action="store_true", help="首次使用时校准主动臂")
    parser.add_argument("--linear-speed", type=float, default=0.10)
    parser.add_argument("--angular-speed", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--task", default="pick up the orange")
    parser.add_argument("--no-keyboard", action="store_true", help="不开启本地键盘监听")
    parser.add_argument("--mock-leader", action="store_true", help="无实体主动臂时发送测试姿态（用于联调）")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)

    if args.list_ports:
        ports = list_serial_ports()
        if not ports:
            print("未检测到可用串口。")
        for dev, desc in ports:
            print(f"{dev}: {desc}")
        return 0

    if not args.atlas_ip:
        parser.error("请指定 --atlas-ip <Atlas板IP>")
    if not args.mock_leader and not args.leader_port:
        parser.error("请指定 --leader-port <主动臂串口>（可先用 --list-ports 查看）")

    if args.mock_leader:
        import math as _m

        class _WaveMockLeader:
            def get_action(self):
                t = time.monotonic()
                return {
                    "shoulder_pan.pos": 10.0 * _m.sin(t),
                    "shoulder_lift.pos": -15.0,
                    "elbow_flex.pos": 25.0,
                    "wrist_flex.pos": -10.0,
                    "wrist_roll.pos": 0.0,
                    "gripper.pos": 30.0,
                }

            def disconnect(self):
                pass

        leader = _WaveMockLeader()
    else:
        SO101Leader, SO101LeaderConfig = load_so101_leader_api()
        cfg = SO101LeaderConfig(port=args.leader_port, id=args.leader_id)
        leader = SO101Leader(cfg)
        leader.connect(calibrate=args.calibrate)

    controls = LeaderBaseKeyboardState()
    listener = None
    if not args.no_keyboard:
        from pynput import keyboard

        listener = keyboard.Listener(on_press=controls.on_press, on_release=controls.on_release)
        listener.start()
        print(HELP_TEXT, flush=True)

    print(
        f"[TELEOP] Sending to {args.atlas_ip}:{args.udp_port} at {args.rate} Hz "
        f"(mock_leader={args.mock_leader}, task={args.task!r})",
        flush=True,
    )
    try:
        summary = run_sender_loop(
            leader,
            controls,
            atlas_ip=args.atlas_ip,
            udp_port=args.udp_port,
            linear_speed=args.linear_speed,
            angular_speed=args.angular_speed,
            rate=args.rate,
            task=args.task,
            max_seconds=args.max_seconds,
        )
        print(f"[TELEOP] Finished: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    finally:
        if listener is not None:
            listener.stop()
        leader.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
