"""跨平台音频源：麦克风采集与离线 wav 文件。

引擎层只消费 float32 numpy 块（16kHz 单声道），不关心数据来自哪里，
因此 Windows 开发机与香橙派/Linux 可共用同一套引擎代码与测试。
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """朴素线性插值重采样（仅用于提示/测试音频，麦克风采集请直接用 16kHz）。"""
    if src_rate == dst_rate:
        return samples
    duration = len(samples) / src_rate
    dst_len = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


class WavSource:
    """按固定块长从 wav 文件产出 16kHz float32 块（文件末尾补 0.66s 静音）。"""

    def __init__(self, path: str | Path, sample_rate: int = 16000, chunk_ms: int = 60):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_ms / 1000)

    def __iter__(self) -> Iterator[np.ndarray]:
        with wave.open(str(self.path), "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            src_rate = w.getframerate()
            raw = w.readframes(w.getnframes())
        if width != 2:
            raise ValueError(f"{self.path}: 仅支持 16bit PCM wav")
        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels).mean(axis=1)
        samples = resample_linear(
            (data / 32768.0).astype(np.float32), src_rate, self.sample_rate
        )
        for i in range(0, len(samples), self.chunk_size):
            yield samples[i : i + self.chunk_size]
        # 结尾静音让 VAD/端点检测收尾（与 sherpa-onnx 官方示例一致）
        tail = np.zeros(int(0.66 * self.sample_rate), dtype=np.float32)
        for i in range(0, len(tail), self.chunk_size):
            yield tail[i : i + self.chunk_size]


class MicSource:
    """麦克风采集（sounddevice 惰性导入，未安装时给出清晰报错）。"""

    def __init__(self, sample_rate: int = 16000, block_ms: int = 60,
                 device: Optional[int] = None):
        try:
            import sounddevice as sd  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "缺少 sounddevice：pip install sounddevice；"
                "Linux 若报 libportaudio 错误请安装 libportaudio2"
            ) from exc
        self._sd = sd
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_ms / 1000)
        self.device = device
        self._stream = None

    def __enter__(self) -> "MicSource":
        import queue

        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()

        def callback(indata, frames, time_info, status):
            self._queue.put(indata[:, 0].copy())

        self._stream = self._sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            channels=1,
            device=self.device,
            callback=callback,
        )
        self._stream.start()
        return self

    def read(self) -> np.ndarray:
        return self._queue.get()

    def __exit__(self, *exc):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
