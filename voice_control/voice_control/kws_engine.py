"""唤醒引擎：sherpa-onnx KeywordSpotter 封装。

模型目录（如 sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01）应包含：
tokens.txt / encoder*.onnx / decoder*.onnx / joiner*.onnx。

keywords.txt 格式（注意：不是整行中文！该 KWS 模型 tokens 为拼音声韵母）：
    n ǐ h ǎo x iǎo ch ē @你好小车
用 tools/make_keywords.py 从中文短语生成并自动校验：
    python tools/make_keywords.py --model-dir <kws目录> --phrases 你好小车 --out configs/keywords.txt

构造前会做符号级预校验——sherpa-onnx 对非法关键词会直接 abort 整个进程。
API 已对 sherpa-onnx 1.13.x 实测。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .audio_source import resample_linear


def _find_one(model_dir: Path, stem: str) -> str:
    hits = sorted(model_dir.glob(f"{stem}*.onnx"))
    if not hits:
        raise FileNotFoundError(f"{model_dir} 中找不到 {stem}*.onnx")
    return str(hits[0])


def _load_token_symbols(model_dir: Path) -> set[str]:
    tokens_path = model_dir / "tokens.txt"
    if not tokens_path.exists():
        raise FileNotFoundError(f"KWS 模型目录缺少 tokens.txt: {model_dir}")
    symbols = set()
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            symbols.add(parts[0])
    return symbols


def validate_keywords_file(keywords_file: Path, token_symbols: set[str]) -> None:
    """符号级校验，避免原生库 abort。每行格式: 符号... @显示词。"""
    bad_lines = []
    for lineno, raw in enumerate(
        keywords_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for symbol in line.split():
            if symbol.startswith("@"):
                continue
            if symbol not in token_symbols:
                bad_lines.append((lineno, symbol))
                break
    if bad_lines:
        details = ", ".join(f"第{ln}行: {sym!r}" for ln, sym in bad_lines)
        raise ValueError(
            f"唤醒词文件 {keywords_file} 含模型不支持的符号（{details}）。"
            "该模型 tokens 为拼音声韵母，请用 tools/make_keywords.py 生成。"
        )


class KwsEngine:
    def __init__(
        self,
        model_dir: str | Path,
        keywords_file: Optional[str | Path] = None,
        sample_rate: int = 16000,
        num_threads: int = 2,
        provider: str = "cpu",
        max_window_sec: float = 8.0,
    ):
        import sherpa_onnx  # 惰性导入：本模块可导入可单测，无需装 sherpa-onnx

        self.sample_rate = sample_rate
        self.max_window_samples = int(max_window_sec * sample_rate)
        model_dir = Path(model_dir)
        if keywords_file is None:
            keywords_file = model_dir / "keywords.txt"
        keywords_file = Path(keywords_file)
        if not keywords_file.exists():
            raise FileNotFoundError(
                f"唤醒词文件不存在: {keywords_file}"
                f"（每行一个中文唤醒词；参考 configs/keywords.txt）"
            )
        for required in ("tokens.txt",):
            if not (model_dir / required).exists():
                raise FileNotFoundError(f"KWS 模型目录缺少 {required}: {model_dir}")
        validate_keywords_file(keywords_file, _load_token_symbols(model_dir))
        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(model_dir / "tokens.txt"),
            encoder=_find_one(model_dir, "encoder"),
            decoder=_find_one(model_dir, "decoder"),
            joiner=_find_one(model_dir, "joiner"),
            keywords_file=str(keywords_file),
            num_threads=num_threads,
            provider=provider,
        )
        self._stream = self._new_stream()
        self._window = 0

    def _new_stream(self):
        return self._spotter.create_stream()

    def feed(self, samples: np.ndarray) -> Optional[str]:
        """送入一块音频；命中唤醒词时返回该词，否则返回 None。"""
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if self._window > self.max_window_samples:
            # 长时间未命中，重建流防止内部状态漂移
            self._stream = self._new_stream()
            self._window = 0
        self._window += len(samples)
        self._stream.accept_waveform(self.sample_rate, samples)
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
        keyword = self._spotter.get_result(self._stream)
        if keyword:
            self._spotter.reset_stream(self._stream)
            self._window = 0
            return keyword
        return None


def load_kws_from_wav(wav_path: str | Path, kws: KwsEngine,
                      src_rate: int = 16000) -> Optional[str]:
    """离线测试辅助：从 wav 文件扫唤醒词。"""
    import wave

    with wave.open(str(wav_path), "rb") as w:
        raw = w.readframes(w.getnframes())
        rate = w.getframerate()
        ch = w.getnchannels()
    data = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).mean(axis=1)
    samples = resample_linear(data / 32768.0, rate, kws.sample_rate)
    # 结尾补 0.66s 静音，让尾部帧完成解码（与 sherpa-onnx 官方示例一致）
    samples = np.concatenate(
        [samples, np.zeros(int(0.66 * kws.sample_rate), dtype=np.float32)]
    )
    chunk = int(0.06 * kws.sample_rate)
    for i in range(0, len(samples), chunk):
        hit = kws.feed(samples[i : i + chunk])
        if hit:
            return hit
    return None
