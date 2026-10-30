#!/usr/bin/env python3
"""批量录制测试音频（16kHz / 16bit / 单声道 wav，跨平台）。

用法：
  python tools/record_wavs.py --out dataset/wake --duration 3
  python tools/record_wavs.py --out dataset/commands --duration 4 --label 抓起红色杯子

流程：按回车开始录一条 -> 录满 --duration 秒自动停止 -> 自动编号保存；
      输入 q 回车结束。每条之间不用赶时间。

建议：
  - 唤醒词集：正常/偏轻/偏响各几条，距离 0.5m 和 1m 各几条，
    外加开着电视/音乐的 3-5 条（兼做负样本）；
  - 指令集：照 commands.yaml 的 20 条指令逐条念，
    文件名用 --label 自动带前缀，如 grasp_红杯子_001.wav。
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--duration", type=float, default=3.0, help="每条秒数")
    ap.add_argument("--label", default="take", help="文件名前缀（建议用指令内容）")
    ap.add_argument("--sample-rate", type=int, default=16000)
    args = ap.parse_args()

    import numpy as np
    import sounddevice as sd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob(f"{args.label}_*.wav"))
    idx = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 1
    n_samples = int(args.duration * args.sample_rate)

    print(f"输出目录: {out}（已存在 {len(existing)} 条，从 {idx} 开始编号）")
    print(f"每条录 {args.duration}s；按回车开始，输入 q 回车结束。")
    while True:
        try:
            cue = input(f"[第 {idx} 条] 按回车开始 / q 退出 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if cue.lower() == "q":
            break
        audio = sd.rec(n_samples, samplerate=args.sample_rate,
                       channels=1, dtype="int16")
        sd.wait()
        path = out / f"{args.label}_{idx:03d}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(args.sample_rate)
            w.writeframes(audio.tobytes())
        peak = int(np.abs(audio).max())
        print(f"  已保存 {path.name}（峰值 {peak}，{'OK' if peak > 500 else '声音偏小!'}）")
        idx += 1
    print(f"共录制 {idx - 1 - len(existing)} 条，累计 {len(sorted(out.glob('*.wav')))} 条。")


if __name__ == "__main__":
    main()
