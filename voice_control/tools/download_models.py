#!/usr/bin/env python3
"""跨平台模型下载脚本（Windows / Linux / 香橙派均可运行）。

用法：
  python tools/download_models.py                  # 下载语音链路必需模型
  python tools/download_models.py --with-sensevoice  # 追加非流式高精度 ASR（需 huggingface_hub）
  python tools/download_models.py --qwen qwen2.5   # LLM（需 huggingface_hub）
  python tools/download_models.py --qwen both

模型来源（国内网络可直接访问）：
  KWS 唤醒   -> ModelScope (pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01)
  流式 ASR   -> hf-mirror（csukuangfj，int8 逐文件下载）
  VAD        -> hf-mirror（deepghs/silero-vad-onnx）
  SenseVoice -> hf-mirror（可选，回退 ASR）
  Qwen       -> hf-mirror（HF_ENDPOINT 可覆盖）
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

HF_MIRROR = "https://hf-mirror.com"
MODELSCOPE = "https://modelscope.cn/models"

# 流式 ASR 从 hf-mirror 逐文件下载（int8 版，GitHub tar 在国内过慢）
STREAMING_ASR_REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
STREAMING_ASR_FILES = [
    "tokens.txt",
    "bpe.model",
    "encoder-epoch-99-avg-1.int8.onnx",
    "decoder-epoch-99-avg-1.int8.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
]
# VAD 从 hf-mirror 下载（GitHub release 在国内不稳定）
VAD_URL = "https://hf-mirror.com/deepghs/silero-vad-onnx/resolve/main/silero_vad.onnx"
VAD_FILE = "silero_vad_v4.onnx"

KWS_REPO = "pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
KWS_FILES = [
    "tokens.txt",
    "keywords.txt",
    "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "test_wavs/0.wav",           # 官方唤醒测试音频，冒烟测试用
    "test_wavs/test_keywords.txt",
]

SENSEVOICE_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

QWEN_REPOS = {
    "qwen2.5": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3": "Qwen/Qwen3-0.6B",
}


def log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            f.write(block)
    tmp.replace(dest)


def mark_done(target: Path, files: list[str]) -> bool:
    return all((target / f).exists() for f in files)


def download_kws(models_dir: Path) -> Path:
    target = models_dir / KWS_REPO.split("/")[-1]
    if mark_done(target, KWS_FILES):
        log(f"跳过（已完成）: {target.name}")
        return target
    for name in KWS_FILES:
        dest = target / name
        if dest.exists():
            continue
        url = f"{MODELSCOPE}/{KWS_REPO}/resolve/master/{name}"
        log(f"下载 {target.name}/{name} ...")
        fetch(url, dest)
    return target


def download_streaming_asr(models_dir: Path) -> Path:
    target = models_dir / STREAMING_ASR_REPO.split("/")[-1]
    if mark_done(target, STREAMING_ASR_FILES):
        log(f"跳过（已完成）: {target.name}")
        return target
    base = os.environ.get("HF_ENDPOINT", HF_MIRROR)
    for name in STREAMING_ASR_FILES:
        dest = target / name
        if dest.exists():
            continue
        url = f"{base}/{STREAMING_ASR_REPO}/resolve/main/{name}"
        log(f"下载 {target.name}/{name} ...")
        fetch(url, dest)
    return target


def download_vad(models_dir: Path) -> Path:
    dest = models_dir / VAD_FILE
    if not dest.exists():
        log(f"下载 {VAD_FILE} ...")
        fetch(VAD_URL, dest)
    return dest


def hf_download(repo: str, local: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("请先 pip install huggingface_hub，再下载该模型")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    log(f"下载 {repo}（镜像: {os.environ['HF_ENDPOINT']}）...")
    snapshot_download(
        repo_id=repo,
        local_dir=str(local),
        local_dir_use_symlinks=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--with-sensevoice", action="store_true",
                    help="额外下载非流式高精度 ASR（SenseVoice-small，需 huggingface_hub）")
    ap.add_argument("--qwen", choices=["qwen2.5", "qwen3", "both"],
                    help="下载 LLM 评测模型（需 huggingface_hub）")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    download_kws(models_dir)
    download_streaming_asr(models_dir)
    download_vad(models_dir)
    if args.with_sensevoice:
        hf_download(SENSEVOICE_REPO, models_dir / SENSEVOICE_REPO.split("/")[-1])
    if args.qwen:
        repos = QWEN_REPOS.values() if args.qwen == "both" else [QWEN_REPOS[args.qwen]]
        for repo in repos:
            hf_download(repo, models_dir / repo.split("/")[-1])
    log("完成。")


if __name__ == "__main__":
    main()
