import inspect
import json
import shutil
import threading
import time
import traceback
from pathlib import Path

import cv2


def _load_lerobot_dataset_api():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features

    return LeRobotDataset, build_dataset_frame, hw_to_dataset_features


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def add_dataset_frame(dataset, frame, task, timestamp):
    if "task" in inspect.signature(dataset.add_frame).parameters:
        dataset.add_frame(frame, task=task, timestamp=timestamp)
    else:
        frame["task"] = task
        frame["timestamp"] = timestamp
        dataset.add_frame(frame)


def is_valid_episode_dir(path):
    """Return True only when an episode directory has saved frames in meta/info.json."""
    info_path = Path(path) / "meta" / "info.json"
    if not info_path.is_file():
        return False
    info = read_json(info_path, None)
    if not isinstance(info, dict):
        return False
    return int(info.get("total_episodes") or 0) > 0 and int(info.get("total_frames") or 0) > 0


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


class AsyncCameraGrabber:
    """Background reader thread keeping the latest RGB frame without blocking teleop."""

    def __init__(self, name, capture, max_consecutive_failures=10):
        self.name = name
        self.capture = capture
        self.max_consecutive_failures = max_consecutive_failures
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame = None
        self._error = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        failures = 0
        while self._running:
            t0 = time.monotonic()
            try:
                ok, bgr = self.capture.read()
            except Exception as exc:
                with self._cond:
                    self._error = exc
                    self._cond.notify_all()
                break
            if ok and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                failures = 0
                with self._cond:
                    self._frame = rgb
                    self._error = None
                    self._cond.notify_all()
            else:
                failures += 1
                if failures >= self.max_consecutive_failures:
                    with self._cond:
                        self._error = RuntimeError(f"Camera {self.name} returned no frame after retries")
                        self._cond.notify_all()
                    break
            if time.monotonic() - t0 < 0.002:
                time.sleep(0.005)

    def latest(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._frame is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"Camera {self.name} timed out waiting for frame")
                self._cond.wait(timeout=remaining)
            if self._error is not None:
                raise self._error
            return self._frame

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None


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
        self.dataset_factory = dataset_factory or (
            lambda **kwargs: _load_lerobot_dataset_api()[0].create(**kwargs)
        )
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
        self._last_record_time = 0.0
        self._last_status_write = 0.0

    @property
    def recording(self):
        return self.dataset is not None

    def _refresh_command(self):
        now = time.monotonic()
        if now - self.last_command_check >= 0.1:
            self.command = read_json(self.command_path, {"active": False})
            self.last_command_check = now
        return now

    def wants_observation(self):
        now = self._refresh_command()
        if not self.recording:
            return bool(self.command.get("active") and self.failed_episode_id != self.command.get("episode_id"))
        if not self.command.get("active"):
            return False
        if self.frames == 0:
            return True
        return (now - self._last_record_time) >= (1.0 / self.fps) - 0.005

    def _write_status(self, status, error=None):
        self._last_status_write = time.monotonic()
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
                self.captures[name] = AsyncCameraGrabber(name, self.capture_factory(camera_indices[name]))

    def _release_captures(self):
        for capture in self.captures.values():
            capture.release()
        self.captures = {}

    def _start(self):
        _, _, hw_to_dataset_features = _load_lerobot_dataset_api()
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
        self._last_record_time = 0.0
        self._write_status("recording")

    def _read_images(self):
        return {name: grabber.latest() for name, grabber in self.captures.items()}

    def _save(self):
        self.dataset.save_episode()
        self.dataset = None
        self._write_status("saved")

    def _discard(self):
        if self.dataset is not None:
            if hasattr(self.dataset, "stop_image_writer"):
                try:
                    self.dataset.stop_image_writer()
                except Exception:
                    pass
            if hasattr(self.dataset, "clear_episode_buffer"):
                try:
                    self.dataset.clear_episode_buffer()
                except Exception:
                    pass
        self.dataset = None
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
        self.frames = 0
        self._write_status("discarded")

    def record(self, observation, action):
        now = self._refresh_command()
        try:
            if (
                self.command.get("active")
                and not self.recording
                and self.failed_episode_id != self.command.get("episode_id")
            ):
                self._start()
            if self.recording and not self.command.get("active"):
                if self.command.get("discard"):
                    self._discard()
                else:
                    self._save()
                return
            if not self.recording:
                return
            if self.frames > 0 and (now - self._last_record_time) < (1.0 / self.fps) - 0.005:
                return
            values = {
                **observation,
                **action,
                **self._read_images(),
            }
            _, build_dataset_frame, _ = _load_lerobot_dataset_api()
            frame = build_dataset_frame(self.features, values, "observation")
            frame.update(build_dataset_frame(self.features, values, "action"))
            add_dataset_frame(self.dataset, frame, self.task, self.frames / self.fps)
            self.frames += 1
            self._last_record_time = now
            if self.frames == 1 or (now - self._last_status_write) >= 0.5:
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

    def _valid_episode_dirs(self):
        demos_dir = self.directory / "demonstrations"
        if not demos_dir.is_dir():
            return []
        return sorted(
            (item for item in demos_dir.iterdir() if item.is_dir() and is_valid_episode_dir(item)),
            key=lambda p: p.name,
        )

    def _saved_episode_count(self):
        return len(self._valid_episode_dirs())

    def _reconcile_stale_recording(self, current):
        if not current.get("recording"):
            return current
        status_fn = getattr(self.gamepad_manager, "status", None)
        if not callable(status_fn):
            return current
        try:
            gp_status = status_fn()
        except Exception:
            return current
        if isinstance(gp_status, dict) and gp_status.get("running") is False:
            write_json(self.command_path, {"active": False})
            reconciled = {
                **current,
                "recording": False,
                "status": "idle",
            }
            write_json(self.status_path, reconciled)
            return reconciled
        return current

    def status(self):
        current = read_json(
            self.status_path,
            {"recording": False, "status": "idle", "episode_id": None, "root": None, "frames": 0},
        )
        current = self._reconcile_stale_recording(current)
        valid_dirs = self._valid_episode_dirs()
        current["saved_episodes"] = len(valid_dirs)
        current["last_episode_id"] = valid_dirs[-1].name if valid_dirs else None
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

    def cancel(self):
        if not self.status().get("recording"):
            return {"ok": True, "status": "already_stopped", **self.status()}
        write_json(self.command_path, {"active": False, "discard": True})
        status = self._wait(lambda current: not current.get("recording", False), timeout=20)
        return {"ok": True, **status}

    def discard_last(self):
        current = self.status()
        if current.get("recording"):
            return self.cancel()
        valid_dirs = self._valid_episode_dirs()
        if not valid_dirs:
            return {
                "ok": False,
                "status": "empty",
                "error": "没有可丢弃的示教记录",
                "saved_episodes": 0,
                "last_episode_id": None,
            }
        target = valid_dirs[-1]
        shutil.rmtree(target, ignore_errors=True)
        remaining = self._valid_episode_dirs()
        payload = {
            "recording": False,
            "status": "discarded",
            "episode_id": target.name,
            "discarded_episode_id": target.name,
            "root": None,
            "frames": 0,
            "task": current.get("task"),
            "error": None,
        }
        write_json(self.status_path, payload)
        return {
            "ok": True,
            **payload,
            "saved_episodes": len(remaining),
            "last_episode_id": remaining[-1].name if remaining else None,
        }

    def prune_empty_dirs(self):
        demos_dir = self.directory / "demonstrations"
        if not demos_dir.is_dir():
            return []
        current = self.status()
        active_root = Path(current["root"]).resolve() if current.get("recording") and current.get("root") else None
        removed = []
        for item in sorted(demos_dir.iterdir()):
            if not item.is_dir():
                continue
            if active_root is not None and item.resolve() == active_root:
                continue
            if not is_valid_episode_dir(item):
                shutil.rmtree(item, ignore_errors=True)
                removed.append(item.name)
        return removed
