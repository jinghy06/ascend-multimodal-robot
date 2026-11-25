import inspect
import json
import time
import traceback
from pathlib import Path

import cv2

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False))
    temporary.replace(path)


def add_dataset_frame(dataset, frame, task, timestamp):
    if "task" in inspect.signature(dataset.add_frame).parameters:
        dataset.add_frame(frame, task=task, timestamp=timestamp)
    else:
        frame["task"] = task
        frame["timestamp"] = timestamp
        dataset.add_frame(frame)


# 1e45:0209 exposes 7 isochronous altsettings; with uvcvideo quirks=0x80
# (UVC_QUIRK_FIX_BANDWIDTH), YUYV 640x480@10fps uses altsetting 3 (1024 B/uframe)
# so it coexists on the same USB 2.0 hub with 4c4a:4a55 (fixed 3069 B/uframe).
YUYV_BANDWIDTH_FIX_USB_IDS = frozenset({("1e45", "0209")})


def detect_usb_vid_pid(index, sysfs_root="/sys/class/video4linux"):
    device_path = Path(sysfs_root) / f"video{index}" / "device"
    if not device_path.exists():
        return None
    current = device_path.resolve()
    for _ in range(4):
        vid_path = current / "idVendor"
        pid_path = current / "idProduct"
        if vid_path.is_file() and pid_path.is_file():
            return (
                vid_path.read_text().strip().lower(),
                pid_path.read_text().strip().lower(),
            )
        if current.parent == current:
            break
        current = current.parent
    return None


def preferred_camera_fourcc(index, usb_id=None):
    if usb_id is None:
        usb_id = detect_usb_vid_pid(index)
    if usb_id in YUYV_BANDWIDTH_FIX_USB_IDS:
        return "YUYV"
    return "MJPG"


def open_v4l2_camera(index, width=640, height=480, fps=10, fourcc=None):
    capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    codec = fourcc or preferred_camera_fourcc(index)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class EpisodeRecorder:
    def __init__(
        self,
        robot,
        command_path,
        status_path,
        fps=10,
        dataset_factory=None,
        capture_factory=None,
        capture_names=("front", "wrist"),
    ):
        self.robot = robot
        self.command_path = Path(command_path)
        self.status_path = Path(status_path)
        self.fps = fps
        self.capture_names = capture_names
        self.dataset_factory = dataset_factory or (lambda **kwargs: LeRobotDataset.create(**kwargs))
        self.capture_factory = capture_factory or (lambda index: open_v4l2_camera(index, fps=self.fps))
        self.dataset = None
        self.captures = {}
        self.episode_id = None
        self.root = None
        self.task = None
        self.episode_started = 0.0
        self.frames = 0
        self.last_command_check = 0.0
        self.command = {"active": False}
        self.failed_episode_id = None

    @property
    def recording(self):
        return self.dataset is not None

    def _write_status(self, status, error=None):
        write_json(
            self.status_path,
            {
                "recording": self.recording,
                "status": status,
                "episode_id": self.episode_id,
                "root": str(self.root) if self.root else None,
                "frames": self.frames,
                "task": self.task,
                "error": error,
            },
        )

    def _ensure_captures(self):
        camera_indices = {"front": 0, "wrist": 2}
        for name in self.capture_names:
            if name not in self.captures:
                self.captures[name] = self.capture_factory(camera_indices[name])

    def _release_captures(self):
        for capture in self.captures.values():
            capture.release()
        self.captures = {}

    def _start(self):
        self.episode_id = self.command["episode_id"]
        self.root = Path(self.command["root"])
        self.task = self.command["task"]
        image_features = hw_to_dataset_features(
            {name: (480, 640, 3) for name in self.capture_names}, "observation"
        )
        self.features = {
            **hw_to_dataset_features(self.robot.action_features, "action"),
            **hw_to_dataset_features(self.robot.observation_features, "observation"),
            **image_features,
        }
        self.dataset = self.dataset_factory(
            repo_id=f"local/{self.episode_id}",
            root=self.root,
            fps=self.fps,
            features=self.features,
            robot_type="lekiwi",
            use_videos=True,
            image_writer_threads=2,
        )
        self._ensure_captures()
        self.frames = 0
        self.episode_started = time.monotonic()
        self._write_status("recording")

    def _read_images(self):
        images = {}
        for name, capture in self.captures.items():
            for _ in range(10):
                ok, frame = capture.read()
                if ok and frame is not None:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError(f"Camera {name} returned no frame after retries")
            images[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return images

    def _save(self):
        self.dataset.save_episode()
        self.dataset = None
        self._write_status("saved")

    def record(self, observation, action):
        now = time.monotonic()
        if now - self.last_command_check >= 0.1:
            self.command = read_json(self.command_path, {"active": False})
            self.last_command_check = now
        try:
            if (
                self.command.get("active")
                and not self.recording
                and self.failed_episode_id != self.command.get("episode_id")
            ):
                self._start()
            if self.recording and not self.command.get("active"):
                self._save()
                return
            if not self.recording:
                return
            values = {
                **observation,
                **action,
                **self._read_images(),
            }
            frame = build_dataset_frame(self.features, values, "observation")
            frame.update(build_dataset_frame(self.features, values, "action"))
            add_dataset_frame(self.dataset, frame, self.task, self.frames / self.fps)
            self.frames += 1
            self._write_status("recording")
        except Exception as exc:
            traceback.print_exc()
            self.failed_episode_id = self.episode_id
            self._release_captures()
            self.dataset = None
            self._write_status("error", str(exc))

    def close(self):
        if self.recording:
            self._save()
        self._release_captures()


class DemoCommandManager:
    def __init__(self, gamepad_manager, directory):
        self.gamepad_manager = gamepad_manager
        self.directory = Path(directory)
        self.command_path = self.directory / "demo_command.json"
        self.status_path = self.directory / "demo_status.json"

    def _saved_episode_count(self):
        demos_dir = self.directory / "demonstrations"
        if not demos_dir.is_dir():
            return 0
        return sum(1 for item in demos_dir.iterdir() if item.is_dir() and (item / "meta" / "info.json").is_file())

    def status(self):
        current = read_json(
            self.status_path,
            {"recording": False, "status": "idle", "episode_id": None, "root": None, "frames": 0},
        )
        current["saved_episodes"] = self._saved_episode_count()
        return current

    def _wait(self, predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status()
            if predicate(status):
                return status
            if status.get("status") == "error":
                raise RuntimeError(status.get("error") or "Demo recorder error")
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for demo recorder status")

    def start(self, task="pick up the orange"):
        task = (task or "").strip() or "pick up the orange"
        if self.status().get("recording"):
            return {"ok": True, "status": "already_recording", **self.status()}
        self.gamepad_manager.start()
        base_id = time.strftime("%Y%m%d_%H%M%S")
        episode_id = base_id
        root = self.directory / "demonstrations" / episode_id
        suffix = 1
        while root.exists():
            episode_id = f"{base_id}_{suffix:02d}"
            root = self.directory / "demonstrations" / episode_id
            suffix += 1
        write_json(
            self.status_path,
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
            self.command_path,
            {"active": True, "episode_id": episode_id, "root": str(root), "task": task},
        )
        status = self._wait(lambda current: current.get("recording", False), timeout=20)
        return {"ok": True, "status": "recording", **status}

    def stop(self):
        if self.status().get("status") == "error":
            return {"ok": False, **self.status()}
        if not self.status().get("recording"):
            return {"ok": True, "status": "already_stopped", **self.status()}
        write_json(self.command_path, {"active": False})
        status = self._wait(lambda current: not current.get("recording", False), timeout=120)
        return {"ok": True, **status}
