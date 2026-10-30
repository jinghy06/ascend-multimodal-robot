#!/usr/bin/env python3
"""唤醒评测：批量跑录制集，输出唤醒率 / 误唤醒数。

用法：
  python tools/eval_wake.py --pos dataset/wake
  python tools/eval_wake.py --pos dataset/wake --neg dataset/neg

  --pos 目录里的 wav 都应包含唤醒词（正样本），--neg 里的都不应触发（负样本）。
  每个文件单独加载引擎（3.3M 小模型，加载很快），互不串状态。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_control.kws_engine import KwsEngine, load_kws_from_wav  # noqa: E402


def iter_wavs(path: Path):
    if path.is_file():
        yield path
    else:
        yield from sorted(path.glob("*.wav"))


def run(kws_dir: Path, keywords: Path, wavs: list[Path], provider: str) -> list[tuple[str, str | None]]:
    results = []
    for wav in wavs:
        engine = KwsEngine(kws_dir, keywords_file=keywords, provider=provider)
        results.append((wav.name, load_kws_from_wav(wav, engine)))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kws-dir",
                    default="models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01")
    ap.add_argument("--keywords", default="configs/keywords.txt")
    ap.add_argument("--pos", required=True, help="正样本 wav 目录/文件（应唤醒）")
    ap.add_argument("--neg", default=None, help="负样本 wav 目录/文件（不应唤醒）")
    ap.add_argument("--provider", default="cpu")
    args = ap.parse_args()

    kws_dir, keywords = Path(args.kws_dir), Path(args.keywords)
    report: dict = {"keywords": str(keywords)}

    pos_wavs = list(iter_wavs(Path(args.pos)))
    pos = run(kws_dir, keywords, pos_wavs, args.provider)
    hits = sum(1 for _, h in pos if h)
    report["positive"] = {
        "total": len(pos), "hit": hits,
        "wake_rate": round(hits / len(pos), 4) if pos else None,
        "missed": [name for name, h in pos if not h],
    }
    print(f"[正样本] {hits}/{len(pos)} 命中，唤醒率 "
          f"{report['positive']['wake_rate']}")
    for name, h in pos:
        print(f"    {'HIT ' if h else 'miss'} {name} {h or ''}")

    if args.neg:
        neg_wavs = list(iter_wavs(Path(args.neg)))
        neg = run(kws_dir, keywords, neg_wavs, args.provider)
        false_hits = [(n, h) for n, h in neg if h]
        report["negative"] = {
            "total": len(neg), "false_wake": len(false_hits),
            "files": [n for n, _ in false_hits],
        }
        print(f"[负样本] {len(false_hits)}/{len(neg)} 误唤醒")
        for n, h in false_hits:
            print(f"    FALSE-WAKE {n} -> {h}")

    out = Path(args.pos) / "_wake_report.json" if Path(args.pos).is_dir() \
        else Path(args.pos).with_suffix(".wake_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {out}")


if __name__ == "__main__":
    main()
