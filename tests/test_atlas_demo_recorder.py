import json
import time

import numpy as np
import pytest

from atlas_demo_recorder import (
    DemoCommandManager,
    EpisodeRecorder,
    add_dataset_frame,
    detect_usb_vid_pid,
    open_v4l2_camera,
    preferred_camera_fourcc,
)


ARM_OBSERVATION = {
    "arm_shoulder_pan.pos": 1.0,
    "arm_shoulder_lift.pos": 2.0,
    "arm_elbow_flex.pos": 3.0,
    "arm_wrist_flex.pos": 4.0,
    "arm_wrist_roll.pos": 5.0,
    "arm_gripper.pos": 6.0,
    "x.vel": 0.0,
    "y.vel": 0.1,
    "theta.vel": 0.0,
}


class FakeDataset:
    def __init__(self):
        self.frames = []
        self.saved = False
        self.cleared = False

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved = True

    def clear_episode_buffer(self):
        self.cleared = True


class FakeCapture:
    def __init__(self, index, failures=0):
        self.index = index
        self.failures = failures
        self.released = False

    def read(self):
        if self.failures > 0:
            self.failures -= 1
            return False, None
        return True, np.full((480, 640, 3), self.index, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeRobot:
    action_features = {
        "arm_shoulder_pan.pos": float,
        "arm_shoulder_lift.pos": float,
        "arm_elbow_flex.pos": float,
        "arm_wrist_flex.pos": float,
        "arm_wrist_roll.pos": float,
        "arm_gripper.pos": float,
        "x.vel": float,
        "y.vel": float,
        "theta.vel": float,
    }
    observation_features = dict(action_features)


def write_command(path, active, root, episode_id="episode_000", task="pick up the orange"):
    path.write_text(json.dumps({"active": active, "root": str(root), "episode_id": episode_id, "task": task}))


def test_episode_recorder_saves_synchronized_lerobot_frames(tmp_path):
    command_path = tmp_path / "command.json"
    status_path = tmp_path / "status.json"
    episode_root = tmp_path / "episode_000"
    dataset = FakeDataset()
    write_command(command_path, True, episode_root)
    captures = {}

    def capture_factory(index):
        captures[index] = FakeCapture(index)
        return captures[index]

    recorder = EpisodeRecorder(
        FakeRobot(),
        command_path,
        status_path,
        fps=10,
        dataset_factory=lambda **kwargs: dataset,
        capture_factory=capture_factory,
        capture_names=("front", "wrist"),
    )

    action = dict(ARM_OBSERVATION)
    recorder.record(ARM_OBSERVATION, action)

    status = json.loads(status_path.read_text())
    assert status["recording"] is True
    assert status["frames"] == 1
    assert len(dataset.frames) == 1
    frame = dataset.frames[0]
    assert frame["task"] == "pick up the orange"
    assert frame["action"].shape == (9,)
    assert frame["observation.state"].shape == (9,)
    assert frame["observation.images.front"].shape == (480, 640, 3)
    assert frame["observation.images.wrist"].shape == (480, 640, 3)
    assert float(frame["action"][0]) == 1.0

    write_command(command_path, False, episode_root)
    time.sleep(0.11)
    recorder.record(ARM_OBSERVATION, action)

    status = json.loads(status_path.read_text())
    assert status["recording"] is False
    assert status["status"] == "saved"
    assert status["frames"] == 1
    assert dataset.saved is True
    # Captures stay open across episodes to avoid USB UVC re-open timeouts
    assert captures[0].released is False
    assert captures[2].released is False

    recorder.close()
    assert captures[0].released is True
    assert captures[2].released is True


def test_episode_recorder_does_nothing_when_demo_is_inactive(tmp_path):
    command_path = tmp_path / "command.json"
    status_path = tmp_path / "status.json"
    write_command(command_path, False, tmp_path / "unused")
    dataset = FakeDataset()
    recorder = EpisodeRecorder(
        FakeRobot(),
        command_path,
        status_path,
        dataset_factory=lambda **kwargs: dataset,
        capture_factory=lambda index: FakeCapture(index),
    )

    recorder.record(ARM_OBSERVATION, dict(ARM_OBSERVATION))

    assert dataset.frames == []
    assert dataset.saved is False
    assert status_path.exists() is False


class PositionalTaskDataset:
    def __init__(self):
        self.calls = []

    def add_frame(self, frame, task, timestamp):
        self.calls.append((frame, task, timestamp))


def test_add_dataset_frame_supports_positional_task_api():
    dataset = PositionalTaskDataset()

    add_dataset_frame(dataset, {"action": np.array([1.0])}, "pick up the orange", 0.25)

    assert dataset.calls[0][1] == "pick up the orange"
    assert dataset.calls[0][2] == 0.25
    assert dataset.calls[0][0]["action"][0] == 1.0


def test_demo_start_resets_stale_error_and_avoids_existing_episode(tmp_path, monkeypatch):
    monkeypatch.setattr("atlas_demo_recorder.time.strftime", lambda fmt: "20240101_000000")
    (tmp_path / "demo_status.json").write_text(json.dumps({"recording": False, "status": "error", "error": "old"}))
    (tmp_path / "demonstrations" / "20240101_000000").mkdir(parents=True)

    class Gamepad:
        def start(self):
            return {"ok": True}

    manager = DemoCommandManager(Gamepad(), tmp_path)
    manager._wait = lambda predicate, timeout=20: {
        "recording": True,
        "status": "recording",
        "episode_id": "20240101_000000_01",
        "root": str(tmp_path / "demonstrations" / "20240101_000000_01"),
        "frames": 0,
    }

    result = manager.start()

    assert result["episode_id"] == "20240101_000000_01"
    status = json.loads((tmp_path / "demo_status.json").read_text())
    assert status["status"] == "starting"
    assert status["error"] is None
    assert json.loads((tmp_path / "demo_command.json").read_text())["active"] is True


def test_camera_warmup_retries_before_first_frame(tmp_path):
    command_path = tmp_path / "command.json"
    status_path = tmp_path / "status.json"
    write_command(command_path, True, tmp_path / "episode_000")
    dataset = FakeDataset()
    recorder = EpisodeRecorder(
        FakeRobot(),
        command_path,
        status_path,
        dataset_factory=lambda **kwargs: dataset,
        capture_factory=lambda index: FakeCapture(index, failures=2),
    )

    recorder.record(ARM_OBSERVATION, dict(ARM_OBSERVATION))

    status = json.loads(status_path.read_text())
    assert status["recording"] is True
    assert status["frames"] == 1


def test_camera_failure_does_not_recreate_episode(tmp_path):
    command_path = tmp_path / "command.json"
    status_path = tmp_path / "status.json"
    write_command(command_path, True, tmp_path / "episode_000")
    dataset = FakeDataset()
    creations = []

    def factory(**kwargs):
        creations.append(kwargs)
        return dataset

    recorder = EpisodeRecorder(
        FakeRobot(),
        command_path,
        status_path,
        dataset_factory=factory,
        capture_factory=lambda index: FakeCapture(index, failures=99),
    )

    recorder.record(ARM_OBSERVATION, dict(ARM_OBSERVATION))
    recorder.record(ARM_OBSERVATION, dict(ARM_OBSERVATION))

    status = json.loads(status_path.read_text())
    assert status["status"] == "error"
    assert len(creations) == 1


def test_preferred_camera_fourcc_selects_yuyv_for_1e45_0209_and_mjpg_otherwise(tmp_path):
    sysfs_video = tmp_path / "sys_v4l" / "video2" / "device"
    sysfs_video.mkdir(parents=True)
    (tmp_path / "sys_v4l" / "video2" / "idVendor").write_text("1e45\n")
    (tmp_path / "sys_v4l" / "video2" / "idProduct").write_text("0209\n")

    assert detect_usb_vid_pid(2, sysfs_root=tmp_path / "sys_v4l") == ("1e45", "0209")
    assert preferred_camera_fourcc(2, usb_id=("1e45", "0209")) == "YUYV"
    assert preferred_camera_fourcc(0, usb_id=("4c4a", "4a55")) == "MJPG"
    assert preferred_camera_fourcc(0, usb_id=None) == "MJPG"


def test_open_v4l2_camera_applies_format_resolution_fps_and_buffer(monkeypatch):
    import cv2

    calls = []

    class RecordingCapture:
        def __init__(self, index, api):
            calls.append(("open", index, api))

        def set(self, prop, value):
            calls.append(("set", prop, value))
            return True

    monkeypatch.setattr("atlas_demo_recorder.cv2.VideoCapture", RecordingCapture)
    monkeypatch.setattr("atlas_demo_recorder.detect_usb_vid_pid", lambda idx: ("1e45", "0209"))

    open_v4l2_camera(2, width=640, height=480, fps=10)

    assert calls == [
        ("open", 2, cv2.CAP_V4L2),
        ("set", cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV")),
        ("set", cv2.CAP_PROP_FRAME_WIDTH, 640),
        ("set", cv2.CAP_PROP_FRAME_HEIGHT, 480),
        ("set", cv2.CAP_PROP_FPS, 10),
        ("set", cv2.CAP_PROP_BUFFERSIZE, 1),
    ]


def _make_ep_dir(root, ep_id, total_episodes=1, total_frames=5):
    meta = root / "demonstrations" / ep_id / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(
        json.dumps({"total_episodes": total_episodes, "total_frames": total_frames}),
        encoding="utf-8",
    )


def test_demo_status_reconciles_stale_recording_and_counts_only_valid_episodes(tmp_path):
    _make_ep_dir(tmp_path, "20261124_000001", total_episodes=0, total_frames=0)
    _make_ep_dir(tmp_path, "20261124_233607", total_episodes=1, total_frames=7)
    (tmp_path / "demo_status.json").write_text(
        json.dumps({"recording": True, "status": "recording", "episode_id": "20261124_233622", "frames": 0}),
        encoding="utf-8",
    )

    class StoppedGamepad:
        def status(self):
            return {"running": False, "pid": None}

    manager = DemoCommandManager(StoppedGamepad(), tmp_path)
    st = manager.status()

    assert st["recording"] is False
    assert st["status"] == "idle"
    assert st["saved_episodes"] == 1
    assert st["last_episode_id"] == "20261124_233607"
    assert json.loads((tmp_path / "demo_command.json").read_text())["active"] is False


def test_episode_recorder_cancel_discards_without_saving(tmp_path):
    command_path = tmp_path / "command.json"
    status_path = tmp_path / "status.json"
    episode_root = tmp_path / "demonstrations" / "ep_cancel"
    episode_root.mkdir(parents=True)
    dataset = FakeDataset()
    write_command(command_path, True, episode_root, episode_id="ep_cancel")

    recorder = EpisodeRecorder(
        FakeRobot(),
        command_path,
        status_path,
        dataset_factory=lambda **kwargs: dataset,
        capture_factory=lambda index: FakeCapture(index),
    )
    recorder.record(ARM_OBSERVATION, dict(ARM_OBSERVATION))
    assert recorder.recording is True

    command_path.write_text(json.dumps({"active": False, "discard": True}), encoding="utf-8")
    time.sleep(0.11)
    recorder.record(ARM_OBSERVATION, dict(ARM_OBSERVATION))

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert recorder.recording is False
    assert dataset.saved is False
    assert dataset.cleared is True
    assert status["status"] == "discarded"
    assert episode_root.exists() is False


def test_demo_manager_discard_last_deletes_newest_valid_episode(tmp_path):
    _make_ep_dir(tmp_path, "20261124_100000", total_episodes=1, total_frames=10)
    _make_ep_dir(tmp_path, "20261124_100500", total_episodes=1, total_frames=12)

    class IdleGamepad:
        def status(self):
            return {"running": False}

    manager = DemoCommandManager(IdleGamepad(), tmp_path)
    res = manager.discard_last()

    assert res["ok"] is True
    assert res["discarded_episode_id"] == "20261124_100500"
    assert res["saved_episodes"] == 1
    assert (tmp_path / "demonstrations" / "20261124_100500").exists() is False
    assert (tmp_path / "demonstrations" / "20261124_100000").exists() is True

    manager.discard_last()
    empty_res = manager.discard_last()
    assert empty_res["ok"] is False
    assert empty_res["status"] == "empty"


def test_episode_recorder_decouples_high_rate_control_to_target_fps(tmp_path):
    command_path = tmp_path / "command.json"
    status_path = tmp_path / "status.json"
    episode_root = tmp_path / "ep_decoupled"
    dataset = FakeDataset()
    write_command(command_path, False, episode_root, episode_id="ep_decoupled")

    recorder = EpisodeRecorder(
        FakeRobot(),
        command_path,
        status_path,
        fps=10,
        dataset_factory=lambda **kwargs: dataset,
        capture_factory=lambda index: FakeCapture(index),
    )

    # Idle: should not request observation
    assert recorder.wants_observation() is False

    write_command(command_path, True, episode_root, episode_id="ep_decoupled")
    recorder.last_command_check = 0.0
    assert recorder.wants_observation() is True

    # Simulate ~30 Hz control ticks over 0.23 s -> at 10 fps should record 3 frames (t=0.0, 0.1, 0.2)
    for _ in range(8):
        recorder.record(ARM_OBSERVATION, dict(ARM_OBSERVATION))
        time.sleep(0.033)

    assert 2 <= len(dataset.frames) <= 3
    assert dataset.frames[0]["timestamp"] == pytest.approx(0.0)
    assert dataset.frames[1]["timestamp"] == pytest.approx(0.1)
    recorder.close()
