"""demo：离线 wav 跑 唤醒→VAD→流式ASR 全链路（无 ROS）。

用法：
  python -m voice_control.demos.demo_asr_file --wav test.wav \
      --kws-dir models/<kws模型目录> --asr-dir models/<流式asr模型目录> \
      [--vad models/sherpa-onnx-silero-vad-4m/silero_vad.onnx] \
      [--keywords configs/keywords.txt]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..asr_engine import AsrPipeline, StreamingAsrEngine, VadSegmenter
from ..audio_source import WavSource
from ..kws_engine import KwsEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="16bit PCM wav（任意采样率，内部重采样到 16k）")
    ap.add_argument("--kws-dir", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--vad", default="models/silero_vad_v4.onnx")
    ap.add_argument("--keywords", default=str(
        Path(__file__).resolve().parents[2] / "configs" / "keywords.txt"))
    ap.add_argument("--provider", default="cpu")
    args = ap.parse_args()

    kws = KwsEngine(args.kws_dir, keywords_file=args.keywords, provider=args.provider)
    asr = StreamingAsrEngine(args.asr_dir, provider=args.provider)
    vad = VadSegmenter(args.vad) if args.vad else None
    pipeline = AsrPipeline(kws, asr, vad=vad)

    for chunk in WavSource(args.wav):
        for ev in pipeline.feed(chunk):
            print(json.dumps(ev, ensure_ascii=False))
    for ev in pipeline.flush():
        print(json.dumps(ev, ensure_ascii=False))


if __name__ == "__main__":
    main()
