#!/usr/bin/env python3
"""生成 sherpa-onnx KWS 唤醒词文件（拼音 token 格式）。

sherpa-onnx KWS 的 keywords.txt 要求每行：
    <声母> <韵母> <声母> <韵母> ... @<中文唤醒词>
例如 "你好军哥" -> "n ǐ h ǎo j ūn g ē @你好军哥"

用法：
  python tools/make_keywords.py --model-dir models/<kws模型目录> \
      --phrases 你好小车 小车小车 --out configs/keywords.txt

依赖 pypinyin（仅生成时需要）：pip install pypinyin
生成的每个符号都会对照模型 tokens.txt 校验，非法会直接报错。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

INITIALS = [
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x",
    "r", "z", "c", "s", "y", "w",
]

TONE_MARKS = {
    "a": "āáǎà", "o": "ōóǒò", "e": "ēéěè",
    "i": "īíǐì", "u": "ūúǔù", "v": "ǖǘǚǜ",
}


def split_syllable(syl: str) -> tuple[str, str]:
    """'ha3o' / 'ni3' / 'jun1' -> ('n', 'i', 3) 形式；无声母如 'ai4' -> ('', 'ai', 4)。"""
    m = re.fullmatch(r"([a-zv]*?)(\d)([a-zv]*)", syl)
    if not m:
        raise ValueError(f"无法解析拼音音节: {syl!r}")
    letters, tone = m.group(1) + m.group(3), int(m.group(2))
    for initial in INITIALS:
        if letters.startswith(initial):
            return initial, letters[len(initial):], tone
    return "", letters, tone


def add_tone_mark(final: str, tone: int) -> str:
    if tone in (0, 5) or not final:
        return final
    for v in "aoe":
        if v in final:
            return final.replace(v, TONE_MARKS[v][tone - 1], 1)
    if "iu" in final:
        idx = final.index("u")
    elif "ui" in final:
        idx = final.index("i")
    else:
        for v in "iuv":
            if v in final:
                idx = final.index(v)
                break
        else:
            return final
    return final[:idx] + TONE_MARKS[final[idx]][tone - 1] + final[idx + 1:]


def phrase_to_symbols(phrase: str) -> list[str]:
    from pypinyin import Style, lazy_pinyin

    symbols: list[str] = []
    for syl in lazy_pinyin(phrase, style=Style.TONE2):
        initial, final, tone = split_syllable(syl)
        if initial:
            symbols.append(initial)
        symbols.append(add_tone_mark(final, tone))
    return symbols


def load_token_symbols(tokens_path: Path) -> set[str]:
    symbols = set()
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            symbols.add(parts[0])
    return symbols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="KWS 模型目录（含 tokens.txt）")
    ap.add_argument("--phrases", nargs="+", required=True, help="中文唤醒词列表")
    ap.add_argument("--out", required=True, help="输出 keywords.txt 路径")
    args = ap.parse_args()

    tokens_path = Path(args.model_dir) / "tokens.txt"
    if not tokens_path.exists():
        sys.exit(f"找不到 {tokens_path}")
    valid = load_token_symbols(tokens_path)

    lines = []
    for phrase in args.phrases:
        symbols = phrase_to_symbols(phrase)
        missing = [s for s in symbols if s not in valid]
        if missing:
            sys.exit(f"唤醒词 {phrase!r} 含模型不支持的符号 {missing}，请换词")
        lines.append(" ".join(symbols) + " @" + phrase)

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {out}（{len(lines)} 个唤醒词）:")
    for line in lines:
        print("  " + line)


if __name__ == "__main__":
    main()
