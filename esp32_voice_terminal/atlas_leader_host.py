"""Atlas-side wireless SO-101 Leader + keyboard base receiver and 10 fps recorder."""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from atlas_demo_recorder import EpisodeRecorder, read_json
from so101_lekiwi_control import configure_bus_retries, ensure_calibration, find_ch343_port, make_robot_config
from so101_lekiwi_gamepad import handle_gamepad_demo_action

ARM_JOINTS = (
    "arm_shoulder_pan.pos",
    "arm_shoulder_lift.pos",
    "arm_elbow_flex.pos",
    "arm_wrist_flex.pos",
    "arm_wrist_roll.pos",
    "arm_gripper.pos",
)
BASE_KEYS = ("x.vel", "y.vel", "theta.vel")
ALL_FEATURES = ARM_JOINTS + BASE_KEYS
DEFAULT_UDP_PORT = 5055

_LEADER_TO_LEKIWI = {
    "shoulder_pan.pos": "arm_shoulder_pan.pos",
    "shoulder_lift.pos": "arm_shoulder_lift.pos",
    "elbow_flex.pos": "arm_elbow_flex.pos",
    "wrist_flex.pos": "arm_wrist_flex.pos",
    "wrist_roll.pos": "arm_wrist_roll.pos",
    "gripper.pos": "arm_gripper.pos",
    "arm_shoulder_pan.pos": "arm_shoulder_pan.pos",
    "arm_shoulder_lift.pos": "arm_shoulder_lift.pos",
    "arm_elbow_flex.pos": "arm_elbow_flex.pos",
    "arm_wrist_flex.pos": "arm_wrist_flex.pos",
    "arm_wrist_roll.pos": "arm_wrist_roll.pos",
    "arm_gripper.pos": "arm_gripper.pos",
}


def map_leader_to_lekiwi_arm(raw_arm: dict[str, Any]) -> dict[str, float]:
    """Map SO-101 Leader joint dict to clamped LeKiwi 6-DoF arm targets."""
    if not isinstance(raw_arm, dict):
        raise ValueError("raw_arm must be a dict")
    mapped: dict[str, float] = {}
    for raw_key, raw_val in raw_arm.items():
        target_key = _LEADER_TO_LEKIWI.get(str(raw_key))
        if target_key is None:
            continue
        val = float(raw_val)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite joint value for {target_key}: {raw_val}")
        low, high = (0.0, 100.0) if target_key == "arm_gripper.pos" else (-100.0, 100.0)
        mapped[target_key] = float(min(high, max(low, val)))
    missing = [j for j in ARM_JOINTS if j not in mapped]
    if missing:
        raise ValueError(f"Missing leader joints: {missing}")
    return mapped


class SoftStartArmFollower:
    """Blends follower arm smoothly from initial pose to leader pose and caps step delta."""

    def __init__(self, initial_obs: dict[str, float], soft_start_sec: float = 1.2, max_arm_delta: float = 6.0):
        self.soft_start_sec = max(0.0, float(soft_start_sec))
        self.max_arm_delta = max(0.1, float(max_arm_delta))
        self._start_pose: dict[str, float] = {}
        self._start_time: float = 0.0
        self.reset(initial_obs)

    def reset(self, current_obs: dict[str, float], now: float | None = None) -> None:
        self._start_pose = {j: float(current_obs.get(j, 0.0)) for j in ARM_JOINTS}
        self._start_time = time.monotonic() if now is None else float(now)

    def compute_targets(
        self,
        desired_arm: dict[str, float] | None,
        current_targets: dict[str, float],
        now: float | None = None,
    ) -> dict[str, float]:
        if desired_arm is None:
            return {j: float(current_targets.get(j, 0.0)) for j in ARM_JOINTS}
        cur_t = time.monotonic() if now is None else float(now)
        if self.soft_start_sec <= 0.0:
            alpha = 1.0
        else:
            alpha = min(1.0, max(0.0, (cur_t - self._start_time) / self.soft_start_sec))

        out: dict[str, float] = {}
        for j in ARM_JOINTS:
            blended = self._start_pose[j] + alpha * (float(desired_arm[j]) - self._start_pose[j])
            prev = float(current_targets.get(j, self._start_pose[j]))
            delta = max(-self.max_arm_delta, min(self.max_arm_delta, blended - prev))
            low, high = (0.0, 100.0) if j == "arm_gripper.pos" else (-100.0, 100.0)
            out[j] = float(min(high, max(low, prev + delta)))
        return out


class WirelessTeleopState:
    """Thread-safe packet state with watchdog safety stop and seq deduplication."""

    def __init__(self, watchdog_sec: float = 0.5, max_lin_vel: float = 0.22, max_ang_vel: float = 65.0):
        self.watchdog_sec = max(0.05, float(watchdog_sec))
        self.max_lin_vel = float(max_lin_vel)
        self.max_ang_vel = float(max_ang_vel)
        self._lock = threading.Lock()
        self._last_seq = -1
        self._last_rx_time = 0.0
        self._desired_arm: dict[str, float] | None = None
        self._base = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        self._demo_actions: list[tuple[str, str]] = []
        self._stop_requested = False
        self.last_sender: tuple[str, int] | None = None
        self.packets_received = 0

    def _clamp_base(self, raw_base: Any) -> dict[str, float]:
        if not isinstance(raw_base, dict):
            return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        vx = float(raw_base.get("x.vel", 0.0))
        vy = float(raw_base.get("y.vel", 0.0))
        om = float(raw_base.get("theta.vel", 0.0))
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(om)):
            return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        return {
            "x.vel": max(-self.max_lin_vel, min(self.max_lin_vel, vx)),
            "y.vel": max(-self.max_lin_vel, min(self.max_lin_vel, vy)),
            "theta.vel": max(-self.max_ang_vel, min(self.max_ang_vel, om)),
        }

    def apply_packet(
        self,
        packet: dict[str, Any],
        sender: tuple[str, int] | None = None,
        now: float | None = None,
    ) -> bool:
        if not isinstance(packet, dict):
            return False
        seq = int(packet.get("seq", -1))
        arm = None
        if packet.get("arm") is not None:
            try:
                arm = map_leader_to_lekiwi_arm(packet["arm"])
            except (ValueError, TypeError):
                return False

        cur_t = time.monotonic() if now is None else float(now)
        with self._lock:
            if seq >= 0 and seq <= self._last_seq:
                return False
            if seq >= 0:
                self._last_seq = seq
            self._last_rx_time = cur_t
            self.packets_received += 1
            if sender is not None:
                self.last_sender = sender
            if arm is not None:
                self._desired_arm = arm
            self._base = self._clamp_base(packet.get("base"))
            if packet.get("stop"):
                self._stop_requested = True
            demo_action = packet.get("demo_action")
            if demo_action in ("toggle", "discard"):
                task = str(packet.get("task") or "pick up the orange").strip() or "pick up the orange"
                self._demo_actions.append((demo_action, task))
        return True

    def pop_demo_actions(self) -> list[tuple[str, str]]:
        with self._lock:
            if not self._demo_actions:
                return []
            items = list(self._demo_actions)
            self._demo_actions.clear()
            return items

    def snapshot(self, now: float | None = None) -> tuple[dict[str, float] | None, dict[str, float], bool, bool]:
        cur_t = time.monotonic() if now is None else float(now)
        with self._lock:
            stale = self._last_rx_time <= 0.0 or (cur_t - self._last_rx_time) > self.watchdog_sec
            base = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0} if stale else dict(self._base)
            arm = dict(self._desired_arm) if self._desired_arm is not None else None
            return arm, base, stale, self._stop_requested


class _MockLeKiwiRobot:
    action_features = {k: float for k in ALL_FEATURES}
    observation_features = {k: float for k in ALL_FEATURES}

    def __init__(self):
        self.is_connected = False
        self.state = {k: 0.0 for k in ALL_FEATURES}
        self.actions: list[dict[str, float]] = []

    def connect(self, calibrate: bool = False):
        self.is_connected = True

    def get_observation(self) -> dict[str, float]:
        return dict(self.state)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.state.update({k: float(v) for k, v in action.items() if k in self.state})
        self.actions.append(dict(self.state))
        return dict(self.state)

    def stop_base(self) -> None:
        for k in BASE_KEYS:
            self.state[k] = 0.0

    def disconnect(self) -> None:
        self.is_connected = False


class _MockCapture:
    def read(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        pass


class AtlasLeaderHost:
    """Runs UDP server, soft-start arm follower, watchdog chassis safety, and 10 fps recording."""

    def __init__(
        self,
        *,
        bind_host: str = "0.0.0.0",
        udp_port: int = DEFAULT_UDP_PORT,
        robot=None,
        work_dir: str | Path | None = None,
        rate: float = 30.0,
        record_fps: int = 10,
        soft_start_sec: float = 1.2,
        max_arm_delta: float = 6.0,
        watchdog_sec: float = 0.5,
        tts_udp_port: int = 5007,
        dry_run: bool = False,
        dataset_factory: Callable[..., Any] | None = None,
        capture_factory: Callable[[int], Any] | None = None,
    ):
        self.bind_host = bind_host
        self.udp_port = udp_port
        self.work_dir = Path(work_dir) if work_dir is not None else Path(__file__).resolve().parent
        self.rate = rate
        self.record_fps = record_fps
        self.soft_start_sec = soft_start_sec
        self.max_arm_delta = max_arm_delta
        self.tts_udp_port = tts_udp_port
        self.dry_run = dry_run
        self.robot = robot if robot is not None else _MockLeKiwiRobot()
        self.state = WirelessTeleopState(watchdog_sec=watchdog_sec)
        self.recorder = EpisodeRecorder(
            self.robot,
            self.work_dir / "demo_command.json",
            self.work_dir / "demo_status.json",
            fps=self.record_fps,
            dataset_factory=dataset_factory,
            capture_factory=capture_factory or ((lambda _idx: _MockCapture()) if dry_run else None),
        )
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.bind_host, self.udp_port))
        self._sock.settimeout(0.05)
        self.bound_port = int(self._sock.getsockname()[1])
        self._running = False
        self._rx_thread: threading.Thread | None = None

    def _build_ack(self, stale: bool) -> bytes:
        st = read_json(self.work_dir / "demo_status.json", {"recording": False, "status": "idle", "frames": 0})
        payload = {
            "ok": True,
            "recording": bool(st.get("recording", False)),
            "status": st.get("status", "idle"),
            "frames": int(st.get("frames", 0)),
            "episode_id": st.get("episode_id"),
            "stale": stale,
            "packets": self.state.packets_received,
            "error": st.get("error"),
        }
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _rx_loop(self) -> None:
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(data.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            self.state.apply_packet(packet, sender=addr)

    def run(self, max_seconds: float = 0.0, exit_on_stop: bool = False) -> None:
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

        started_at = time.monotonic()
        period = 1.0 / max(1.0, self.rate)
        last_ack = 0.0
        was_stale = True

        try:
            obs = self.robot.get_observation()
            follower = SoftStartArmFollower(obs, soft_start_sec=self.soft_start_sec, max_arm_delta=self.max_arm_delta)
            targets = {j: float(obs.get(j, 0.0)) for j in ARM_JOINTS}

            while self._running:
                tick_start = time.monotonic()
                if max_seconds > 0.0 and (tick_start - started_at) >= max_seconds:
                    break

                demo_actions = self.state.pop_demo_actions()
                for action_name, task in demo_actions:
                    handle_gamepad_demo_action(action_name, self.work_dir, task=task, tts_port=self.tts_udp_port)
                    self.recorder.last_command_check = 0.0

                desired_arm, base_vel, stale, stop_req = self.state.snapshot(tick_start)
                if exit_on_stop and stop_req:
                    break

                need_obs = self.recorder.wants_observation()
                current_obs = self.robot.get_observation() if need_obs else targets

                if was_stale and not stale:
                    follower.reset(current_obs, now=tick_start)
                was_stale = stale

                targets = follower.compute_targets(desired_arm if not stale else None, targets, now=tick_start)
                action = {**targets, **base_vel}
                actual_action = self.robot.send_action(action)
                self.recorder.record(current_obs, actual_action)

                if self.state.last_sender is not None and (demo_actions or (tick_start - last_ack) >= 0.2):
                    last_ack = tick_start
                    try:
                        self._sock.sendto(self._build_ack(stale), self.state.last_sender)
                    except OSError:
                        pass

                remaining = period - (time.monotonic() - tick_start)
                if remaining > 0.0 and self._running:
                    time.sleep(remaining)
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        self.recorder.close()
        try:
            if self.robot.is_connected:
                self.robot.stop_base()
        finally:
            if self.robot.is_connected:
                self.robot.disconnect()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Atlas wireless SO-101 Leader host + 10 fps recorder")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    parser.add_argument("--port", default=None, help="LeKiwi CH343 serial port")
    parser.add_argument("--id", default="my_awesome_kiwi", help="LeKiwi calibration ID")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--record-fps", type=int, default=10)
    parser.add_argument("--soft-start-sec", type=float, default=1.2)
    parser.add_argument("--max-arm-delta", type=float, default=6.0)
    parser.add_argument("--watchdog-sec", type=float, default=0.5)
    parser.add_argument("--tts-udp-port", type=int, default=5007)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)

    if args.dry_run:
        robot = _MockLeKiwiRobot()
        robot.connect()
    else:
        from lerobot.robots.lekiwi import LeKiwi

        serial_port = args.port or find_ch343_port() or "/dev/ttyACM0"
        robot = LeKiwi(make_robot_config(serial_port, args.id))
        configure_bus_retries(robot)
        ensure_calibration(robot, args.calibrate)
        robot.connect(calibrate=args.calibrate)

    host = AtlasLeaderHost(
        bind_host=args.bind,
        udp_port=args.udp_port,
        robot=robot,
        work_dir=args.work_dir,
        rate=args.rate,
        record_fps=args.record_fps,
        soft_start_sec=args.soft_start_sec,
        max_arm_delta=args.max_arm_delta,
        watchdog_sec=args.watchdog_sec,
        tts_udp_port=args.tts_udp_port,
        dry_run=args.dry_run,
    )

    def _sig_handler(*_):
        host._running = False

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    print(
        f"[LEADER-HOST] Listening UDP {args.bind}:{host.bound_port} "
        f"(rate={args.rate}Hz, record_fps={args.record_fps}, dry_run={args.dry_run})",
        flush=True,
    )
    host.run(max_seconds=args.max_seconds)
    print("[LEADER-HOST] Stopped cleanly", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
