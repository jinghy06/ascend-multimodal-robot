"""LeRobot v2.1 demonstration dataset inspection and merging tools for Atlas + LeKiwi."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def _video_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features") or {}
    return [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]


def _probe_video_pyav(path: Path) -> tuple[bool, int, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    try:
        import av
    except ImportError:
        return False, 0, None, None, None
    try:
        with av.open(str(path)) as container:
            frames = [frame.to_ndarray(format="bgr24") for frame in container.decode(video=0)]
    except Exception:
        return False, 0, None, None, None
    if not frames:
        return False, 0, None, None, None
    n = len(frames)
    return True, n, frames[0], frames[n // 2], frames[-1]


def _probe_video(path: Path) -> tuple[bool, int, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, 0, None, None, None
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            ok_first, first = cap.read()
            if ok_first and first is not None:
                mid = None
                last = None
                if frame_count > 2:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                    ok_mid, mid_frame = cap.read()
                    if ok_mid and mid_frame is not None:
                        mid = mid_frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
                    ok_last, last_frame = cap.read()
                    if ok_last and last_frame is not None:
                        last = last_frame
                return True, frame_count, first, mid if mid is not None else first, last if last is not None else first
        finally:
            cap.release()
    return _probe_video_pyav(path)


def _export_preview(out_path: Path, episode_name: str, video_samples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    rows = []
    for key in sorted(video_samples):
        first, mid, last = video_samples[key]
        thumbs = [cv2.resize(img, (213, 160)) for img in (first, mid, last)]
        strip = np.hstack(thumbs)
        cv2.putText(strip, key.replace("observation.images.", ""), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)
        rows.append(strip)
    if not rows:
        return
    canvas = np.vstack(rows)
    out_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path / f"{episode_name}.jpg"), canvas)


def inspect_demos(
    demos_dir: str | Path,
    max_arm_jump: float = 25.0,
    max_vel_jump: float = 0.5,
    export_preview_dir: str | Path | None = None,
    prune_empty: bool = False,
) -> dict[str, Any]:
    """Inspect all episode folders under demos_dir and report valid, empty, and corrupt episodes."""
    demos_dir = Path(demos_dir)
    preview_dir = Path(export_preview_dir) if export_preview_dir else None
    active_root: Path | None = None
    parent_status = _read_json(demos_dir.parent / "demo_status.json", {})
    if isinstance(parent_status, dict) and parent_status.get("recording") and parent_status.get("root"):
        active_root = Path(parent_status["root"]).resolve()

    episodes: list[dict[str, Any]] = []
    pruned: list[str] = []

    if not demos_dir.is_dir():
        return {"demos_dir": str(demos_dir), "episodes": [], "summary": {"valid": 0, "empty": 0, "corrupt": 0}, "pruned": []}

    for item in sorted(demos_dir.iterdir(), key=lambda p: p.name):
        if not item.is_dir():
            continue
        info = _read_json(item / "meta" / "info.json", None)
        if not isinstance(info, dict):
            entry = {"name": item.name, "status": "corrupt", "frames": 0, "tasks": [], "issues": ["missing meta/info.json"]}
            episodes.append(entry)
            continue

        total_eps = int(info.get("total_episodes") or 0)
        total_frames = int(info.get("total_frames") or 0)
        if total_eps == 0 or total_frames == 0:
            if prune_empty and (active_root is None or item.resolve() != active_root):
                shutil.rmtree(item, ignore_errors=True)
                pruned.append(item.name)
            episodes.append({"name": item.name, "status": "empty", "frames": total_frames, "tasks": [], "issues": ["zero episodes/frames"]})
            continue

        issues: list[str] = []
        fps = float(info.get("fps") or 10.0)
        tasks_rows = _read_jsonl(item / "meta" / "tasks.jsonl")
        tasks = [str(row.get("task", "")) for row in tasks_rows if "task" in row]

        parquet_path = item / "data" / "chunk-000" / "episode_000000.parquet"
        max_arm_delta = 0.0
        max_vel_delta = 0.0
        row_count = 0
        if not parquet_path.is_file():
            issues.append("missing episode_000000.parquet")
        else:
            table = pq.read_table(parquet_path)
            row_count = table.num_rows
            if row_count != total_frames:
                issues.append(f"parquet rows ({row_count}) != info.total_frames ({total_frames})")
            cols = set(table.column_names)
            for col in ("observation.state", "action"):
                if col in cols:
                    arr = np.asarray(table[col].to_pylist(), dtype=np.float32)
                    if not np.isfinite(arr).all():
                        issues.append(f"{col} contains NaN/Inf")
                    elif len(arr) >= 2:
                        diffs = np.abs(np.diff(arr, axis=0))
                        if arr.shape[1] >= 6:
                            max_arm_delta = max(max_arm_delta, float(diffs[:, :6].max()))
                        if arr.shape[1] >= 9:
                            max_vel_delta = max(max_vel_delta, float(diffs[:, 6:9].max()))
            if "timestamp" in cols and row_count >= 2:
                ts = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64).reshape(-1)
                dt = np.diff(ts)
                if np.any(dt <= 0):
                    issues.append("non-monotonic timestamp")
                else:
                    expected_dt = 1.0 / max(fps, 1e-6)
                    if float(np.max(np.abs(dt - expected_dt))) > expected_dt * 0.5:
                        issues.append(f"timestamp jitter exceeds 50% of {expected_dt:.3f}s")

        if max_arm_delta > max_arm_jump:
            issues.append(f"arm jump {max_arm_delta:.2f} > {max_arm_jump}")
        if max_vel_delta > max_vel_jump:
            issues.append(f"vel jump {max_vel_delta:.3f} > {max_vel_jump}")

        video_samples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for vkey in _video_keys(info):
            vpath = item / "videos" / "chunk-000" / vkey / "episode_000000.mp4"
            ok, vframes, first, mid, last = _probe_video(vpath)
            if not ok:
                issues.append(f"unreadable video {vkey}")
            else:
                if vframes > 0 and row_count > 0 and abs(vframes - row_count) > 2:
                    issues.append(f"{vkey} frames ({vframes}) != parquet ({row_count})")
                if first is not None and mid is not None and last is not None:
                    video_samples[vkey] = (first, mid, last)

        if preview_dir is not None and video_samples:
            _export_preview(preview_dir, item.name, video_samples)

        status = "valid" if not issues else "corrupt"
        episodes.append(
            {
                "name": item.name,
                "status": status,
                "frames": row_count or total_frames,
                "tasks": tasks,
                "max_arm_delta": round(max_arm_delta, 3),
                "max_vel_delta": round(max_vel_delta, 3),
                "issues": issues,
            }
        )

    summary = {
        "valid": sum(1 for e in episodes if e["status"] == "valid"),
        "empty": sum(1 for e in episodes if e["status"] == "empty"),
        "corrupt": sum(1 for e in episodes if e["status"] == "corrupt"),
        "valid_frames": sum(e["frames"] for e in episodes if e["status"] == "valid"),
    }
    return {"demos_dir": str(demos_dir), "episodes": episodes, "summary": summary, "pruned": pruned}


def _rewrite_scalar_stat(stats: dict[str, Any], name: str, values: list[int]) -> None:
    if name not in stats or not values:
        return
    arr = np.asarray(values, dtype=np.float64)
    stats[name] = {
        "min": [int(arr.min())],
        "max": [int(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [len(values)],
    }


def merge_v21_datasets(
    source_dirs: str | Path | Iterable[str | Path],
    output_dir: str | Path,
    task_filter: str | None = None,
    min_frames: int = 10,
    copy_mode: str = "copy",
) -> dict[str, Any]:
    """Merge single-episode or multi-episode LeRobot v2.1 directories into one unified v2.1 dataset."""
    if isinstance(source_dirs, (str, Path)):
        src_root = Path(source_dirs)
        candidates = sorted((p for p in src_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    else:
        candidates = [Path(p) for p in source_dirs]

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    task_to_index: dict[str, int] = {}
    episodes_meta: list[dict[str, Any]] = []
    episodes_stats_meta: list[dict[str, Any]] = []
    base_info: dict[str, Any] | None = None
    global_frame_index = 0
    out_ep_idx = 0

    for src in candidates:
        info = _read_json(src / "meta" / "info.json", None)
        if not isinstance(info, dict):
            continue
        if int(info.get("total_episodes") or 0) <= 0 or int(info.get("total_frames") or 0) <= 0:
            continue

        local_tasks = {int(r["task_index"]): str(r["task"]) for r in _read_jsonl(src / "meta" / "tasks.jsonl")}
        local_episodes = _read_jsonl(src / "meta" / "episodes.jsonl")
        if not local_episodes:
            local_episodes = [{"episode_index": 0, "tasks": list(local_tasks.values()) or ["default"], "length": int(info["total_frames"])}]
        local_stats = {int(r["episode_index"]): r.get("stats", {}) for r in _read_jsonl(src / "meta" / "episodes_stats.jsonl")}
        chunks_size = int(info.get("chunks_size") or 1000)
        video_keys = _video_keys(info)

        for ep_row in local_episodes:
            src_ep_idx = int(ep_row["episode_index"])
            ep_tasks = [str(t) for t in (ep_row.get("tasks") or [])] or [local_tasks.get(0, "default")]
            primary_task = ep_tasks[0]
            if task_filter and primary_task != task_filter:
                continue

            src_chunk = src_ep_idx // chunks_size
            src_pq = src / "data" / f"chunk-{src_chunk:03d}" / f"episode_{src_ep_idx:06d}.parquet"
            if not src_pq.is_file():
                continue
            table = pq.read_table(src_pq)
            n_rows = table.num_rows
            if n_rows < min_frames:
                continue

            if base_info is None:
                base_info = json.loads(json.dumps(info))

            for t in ep_tasks:
                if t not in task_to_index:
                    task_to_index[t] = len(task_to_index)

            if "task_index" in table.column_names:
                old_t_idx = table["task_index"].to_pylist()
                new_t_idx = [task_to_index[local_tasks.get(int(ti), primary_task)] for ti in old_t_idx]
            else:
                new_t_idx = [task_to_index[primary_task]] * n_rows

            new_ep_col = [out_ep_idx] * n_rows
            new_idx_col = list(range(global_frame_index, global_frame_index + n_rows))

            for col_name, values in (("episode_index", new_ep_col), ("index", new_idx_col), ("task_index", new_t_idx)):
                if col_name in table.column_names:
                    col_idx = table.schema.get_field_index(col_name)
                    field_type = table.schema.field(col_name).type
                    table = table.set_column(col_idx, col_name, pa.array(values, type=field_type))

            out_chunk = out_ep_idx // chunks_size
            out_pq = output_dir / "data" / f"chunk-{out_chunk:03d}" / f"episode_{out_ep_idx:06d}.parquet"
            out_pq.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out_pq)

            for vkey in video_keys:
                src_vid = src / "videos" / f"chunk-{src_chunk:03d}" / vkey / f"episode_{src_ep_idx:06d}.mp4"
                out_vid = output_dir / "videos" / f"chunk-{out_chunk:03d}" / vkey / f"episode_{out_ep_idx:06d}.mp4"
                if not src_vid.is_file():
                    raise FileNotFoundError(f"Missing video file: {src_vid}")
                out_vid.parent.mkdir(parents=True, exist_ok=True)
                if copy_mode == "hardlink":
                    os.link(src_vid, out_vid)
                else:
                    shutil.copy2(src_vid, out_vid)

            episodes_meta.append({"episode_index": out_ep_idx, "tasks": ep_tasks, "length": n_rows})
            ep_stats = json.loads(json.dumps(local_stats.get(src_ep_idx, {})))
            if ep_stats:
                _rewrite_scalar_stat(ep_stats, "episode_index", new_ep_col)
                _rewrite_scalar_stat(ep_stats, "index", new_idx_col)
                _rewrite_scalar_stat(ep_stats, "task_index", new_t_idx)
                episodes_stats_meta.append({"episode_index": out_ep_idx, "stats": ep_stats})

            global_frame_index += n_rows
            out_ep_idx += 1

    if out_ep_idx == 0 or base_info is None:
        raise ValueError("No valid episodes matched merge criteria")

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    chunks_size = int(base_info.get("chunks_size") or 1000)
    video_keys = _video_keys(base_info)

    base_info["codebase_version"] = "v2.1"
    base_info["total_episodes"] = out_ep_idx
    base_info["total_frames"] = global_frame_index
    base_info["total_tasks"] = len(task_to_index)
    base_info["total_videos"] = out_ep_idx * len(video_keys)
    base_info["total_chunks"] = (out_ep_idx + chunks_size - 1) // chunks_size
    base_info["splits"] = {"train": f"0:{out_ep_idx}"}
    (meta_dir / "info.json").write_text(json.dumps(base_info, indent=4, ensure_ascii=False), encoding="utf-8")

    _write_jsonl(meta_dir / "tasks.jsonl", ({"task_index": idx, "task": t} for t, idx in task_to_index.items()))
    _write_jsonl(meta_dir / "episodes.jsonl", episodes_meta)
    if episodes_stats_meta:
        _write_jsonl(meta_dir / "episodes_stats.jsonl", episodes_stats_meta)

    return {
        "output_dir": str(output_dir),
        "total_episodes": out_ep_idx,
        "total_frames": global_frame_index,
        "total_tasks": len(task_to_index),
        "tasks": list(task_to_index.keys()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LeRobot v2.1 demonstration dataset inspect & merge tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect recorded episodes and check data integrity")
    p_inspect.add_argument("demos_dir", help="Directory containing single-episode demo folders")
    p_inspect.add_argument("--max-arm-jump", type=float, default=25.0)
    p_inspect.add_argument("--max-vel-jump", type=float, default=0.5)
    p_inspect.add_argument("--preview-dir", default=None, help="Optional directory to export JPEG contact sheets")
    p_inspect.add_argument("--prune-empty", action="store_true", help="Remove empty 0-frame directories")
    p_inspect.add_argument("--json", action="store_true", help="Output JSON report")

    p_merge = sub.add_parser("merge", help="Merge valid v2.1 demo folders into one unified dataset")
    p_merge.add_argument("demos_dir", help="Source directory containing episode folders")
    p_merge.add_argument("output_dir", help="Destination merged dataset directory")
    p_merge.add_argument("--task", default=None, help="Optional exact task string filter")
    p_merge.add_argument("--min-frames", type=int, default=10, help="Minimum frames per episode")
    p_merge.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")

    args = parser.parse_args(argv)
    if args.cmd == "inspect":
        report = inspect_demos(
            args.demos_dir,
            max_arm_jump=args.max_arm_jump,
            max_vel_jump=args.max_vel_jump,
            export_preview_dir=args.preview_dir,
            prune_empty=args.prune_empty,
        )
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            s = report["summary"]
            print(
                f"Demos in {report['demos_dir']}: valid={s['valid']} ({s['valid_frames']} frames), "
                f"empty={s['empty']}, corrupt={s['corrupt']}, pruned={len(report['pruned'])}"
            )
            for ep in report["episodes"]:
                issues = "; ".join(ep["issues"]) if ep["issues"] else "-"
                print(f"  [{ep['status']:7s}] {ep['name']} | frames={ep['frames']} | tasks={ep['tasks']} | issues={issues}")
        return 0

    if args.cmd == "merge":
        res = merge_v21_datasets(
            args.demos_dir,
            args.output_dir,
            task_filter=args.task,
            min_frames=args.min_frames,
            copy_mode=args.copy_mode,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
