"""SmolVLA & ACT post-training command builder/runner for merged LeKiwi datasets."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def build_train_command(
    preset: str,
    dataset_root: str | Path,
    output_dir: str | Path,
    repo_id: str = "local/lekiwi_merged",
    batch_size: int | None = None,
    steps: int | None = None,
    device: str = "cuda",
) -> list[str]:
    dataset_root = str(dataset_root)
    output_dir = str(output_dir)

    if preset == "act":
        return [
            "lerobot-train",
            "--policy.type=act",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={dataset_root}",
            f"--output_dir={output_dir}",
            f"--policy.device={device}",
            f"--batch_size={batch_size or 8}",
            f"--steps={steps or 40000}",
            "--policy.chunk_size=20",
            "--policy.n_action_steps=20",
        ]

    rename_map = json.dumps(
        {
            "observation.images.front": "observation.images.camera1",
            "observation.images.wrist": "observation.images.camera2",
        },
        separators=(",", ":"),
    )
    cmd = [
        "lerobot-train",
        "--policy.path=lerobot/smolvla_base",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        f"--output_dir={output_dir}",
        f"--policy.device={device}",
        f"--batch_size={batch_size or 16}",
        f"--steps={steps or 40000}",
        "--policy.empty_cameras=1",
        "--policy.chunk_size=20",
        "--policy.n_action_steps=20",
        f"--rename_map={rename_map}",
    ]
    if preset == "smolvla_fast256":
        cmd.extend(
            [
                "--policy.resize_imgs_with_padding=[256,256]",
                "--policy.num_steps=3",
            ]
        )
    elif preset != "smolvla":
        raise ValueError(f"Unsupported preset: {preset}")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or run LeRobot post-training for LeKiwi")
    parser.add_argument("--preset", choices=("smolvla", "smolvla_fast256", "act"), default="smolvla")
    parser.add_argument("--dataset-root", required=True, help="Local path to merged LeRobot dataset")
    parser.add_argument("--output-dir", default="outputs/train/lekiwi_policy")
    parser.add_argument("--repo-id", default="local/lekiwi_merged")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true", help="Execute instead of dry-run printing")
    args = parser.parse_args(argv)

    cmd = build_train_command(
        preset=args.preset,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        batch_size=args.batch_size,
        steps=args.steps,
        device=args.device,
    )
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.execute:
        return subprocess.call(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
