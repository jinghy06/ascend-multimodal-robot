import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_dataset_tools import inspect_demos, merge_v21_datasets
from scripts.train_lekiwi import build_train_command


def _write_synthetic_v21_episode(root: Path, task: str, frames: int = 12, arm_jump_at: int | None = None):
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for vkey in ("observation.images.front", "observation.images.wrist"):
        vdir = root / "videos" / "chunk-000" / vkey
        vdir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(vdir / "episode_000000.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        for i in range(frames):
            img = np.full((48, 64, 3), (i * 15) % 255, dtype=np.uint8)
            writer.write(img)
        writer.release()

    states = []
    actions = []
    for i in range(frames):
        vec = [float(i)] * 6 + [0.05, 0.0, 0.0]
        if arm_jump_at is not None and i >= arm_jump_at:
            vec[0] += 60.0
        states.append(vec)
        actions.append(vec)

    table = pa.table(
        {
            "observation.state": pa.array(states, type=pa.list_(pa.float32(), 9)),
            "action": pa.array(actions, type=pa.list_(pa.float32(), 9)),
            "episode_index": pa.array([0] * frames, type=pa.int64()),
            "frame_index": pa.array(list(range(frames)), type=pa.int64()),
            "timestamp": pa.array([i / 10.0 for i in range(frames)], type=pa.float32()),
            "index": pa.array(list(range(frames)), type=pa.int64()),
            "task_index": pa.array([0] * frames, type=pa.int64()),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "episode_000000.parquet")

    info = {
        "codebase_version": "v2.1",
        "robot_type": "lekiwi",
        "total_episodes": 1,
        "total_frames": frames,
        "total_tasks": 1,
        "total_videos": 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.front": {"dtype": "video", "shape": [48, 64, 3]},
            "observation.images.wrist": {"dtype": "video", "shape": [48, 64, 3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n", encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": [task], "length": frames}) + "\n", encoding="utf-8"
    )
    (root / "meta" / "episodes_stats.jsonl").write_text(
        json.dumps({"episode_index": 0, "stats": {"episode_index": {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [frames]}}}) + "\n",
        encoding="utf-8",
    )


def test_inspect_demos_classifies_valid_empty_and_corrupt(tmp_path):
    demos = tmp_path / "demonstrations"
    _write_synthetic_v21_episode(demos / "ep_ok", "pick up the orange", frames=12)
    _write_synthetic_v21_episode(demos / "ep_jump", "pick up the orange", frames=12, arm_jump_at=6)

    empty_dir = demos / "ep_empty" / "meta"
    empty_dir.mkdir(parents=True)
    (empty_dir / "info.json").write_text(json.dumps({"total_episodes": 0, "total_frames": 0}), encoding="utf-8")

    active_empty = demos / "ep_active" / "meta"
    active_empty.mkdir(parents=True)
    (active_empty / "info.json").write_text(json.dumps({"total_episodes": 0, "total_frames": 0}), encoding="utf-8")
    (tmp_path / "demo_status.json").write_text(
        json.dumps({"recording": True, "root": str(demos / "ep_active")}), encoding="utf-8"
    )

    preview_dir = tmp_path / "previews"
    report = inspect_demos(demos, export_preview_dir=preview_dir, prune_empty=True)

    assert report["summary"]["valid"] == 1
    assert report["summary"]["corrupt"] == 1
    assert report["summary"]["empty"] == 2
    assert report["pruned"] == ["ep_empty"]
    assert (demos / "ep_empty").exists() is False
    assert (demos / "ep_active").exists() is True
    assert (preview_dir / "ep_ok.jpg").is_file()


def test_merge_v21_datasets_combines_episodes_and_reindexes(tmp_path):
    demos = tmp_path / "demonstrations"
    _write_synthetic_v21_episode(demos / "20261127_01", "pick up the orange", frames=10)
    _write_synthetic_v21_episode(demos / "20261127_02", "pick up the red cup", frames=15)

    merged = tmp_path / "merged"
    summary = merge_v21_datasets(demos, merged, min_frames=10)

    assert summary["total_episodes"] == 2
    assert summary["total_frames"] == 25
    assert summary["total_tasks"] == 2

    info = json.loads((merged / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 25
    assert info["total_videos"] == 4
    assert info["splits"] == {"train": "0:2"}

    pq1 = pq.read_table(merged / "data" / "chunk-000" / "episode_000001.parquet")
    assert pq1["episode_index"].to_pylist() == [1] * 15
    assert pq1["index"].to_pylist() == list(range(10, 25))
    assert pq1["task_index"].to_pylist() == [1] * 15
    assert (merged / "videos" / "chunk-000" / "observation.images.front" / "episode_000001.mp4").is_file()


def test_train_lekiwi_command_presets():
    fast_cmd = build_train_command("smolvla_fast256", "/data/merged", "outputs/fast")
    assert "--policy.empty_cameras=1" in fast_cmd
    assert "--policy.num_steps=3" in fast_cmd
    assert "--policy.resize_imgs_with_padding=[256,256]" in fast_cmd

    act_cmd = build_train_command("act", "/data/merged", "outputs/act")
    assert "--policy.type=act" in act_cmd
