"""流式 ASR + VAD 引擎与「唤醒→录音→finalize」状态机。

- StreamingAsrEngine: sherpa-onnx OnlineRecognizer（流式 transducer）封装；
- VadSegmenter: sherpa-onnx VoiceActivityDetector（silero-vad）封装，可选
  （无 VAD 模型时退化为固定时长窗口）；
- AsrPipeline: KwsEngine + StreamingAsrEngine + VadSegmenter 的状态机，
  feed(160Hz float32 块) → 事件列表（wake / partial / final）。

事件 dict 约定：
  {"type": "wake", "keyword": str}
  {"type": "partial", "text": str}
  {"type": "final", "text": str}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .kws_engine import KwsEngine, _find_one


def _find_tokens(model_dir: Path) -> str:
    path = model_dir / "tokens.txt"
    if not path.exists():
        raise FileNotFoundError(f"{model_dir} 中缺少 tokens.txt")
    return str(path)


class StreamingAsrEngine:
    """流式识别器。模型目录需含 tokens.txt / encoder / decoder / joiner 的 onnx。"""

    def __init__(
        self,
        model_dir: str | Path,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        num_threads: int = 2,
        provider: str = "cpu",
    ):
        import sherpa_onnx

        self.sample_rate = sample_rate
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=_find_tokens(Path(model_dir)),
            encoder=_find_one(Path(model_dir), "encoder"),
            decoder=_find_one(Path(model_dir), "decoder"),
            joiner=_find_one(Path(model_dir), "joiner"),
            num_threads=num_threads,
            sample_rate=sample_rate,
            feature_dim=feature_dim,
            provider=provider,
        )

    def new_stream(self):
        return self._recognizer.create_stream()

    def accept(self, stream, samples: np.ndarray) -> None:
        stream.accept_waveform(self.sample_rate, samples.astype(np.float32))

    def decode(self, stream) -> None:
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

    def partial_text(self, stream) -> str:
        return self._recognizer.get_result(stream).strip()

    def finalize(self, stream) -> str:
        """输入结束后取最终结果并重置流。"""
        self.decode(stream)
        text = self.partial_text(stream)
        self._recognizer.reset(stream)
        return text

    def is_endpoint(self, stream) -> bool:
        return self._recognizer.is_endpoint(stream)


class VadSegmenter:
    """silero-vad 语音段切分。model_path 为 None 或文件缺失时不可用时返回 segments= []。"""

    def __init__(
        self,
        model_path: Optional[str | Path],
        sample_rate: int = 16000,
        min_silence_duration: float = 0.6,
        min_speech_duration: float = 0.25,
        max_speech_duration: float = 15.0,
        num_threads: int = 1,
        provider: str = "cpu",
    ):
        self.sample_rate = sample_rate
        self._vad = None
        if model_path is None:
            return
        model_path = Path(model_path)
        if not model_path.exists():
            return
        import sherpa_onnx

        # 预检通过后再调原生库（sherpa-onnx 对缺失文件会直接 abort 整个进程）
        if model_path.stat().st_size < 100_000:
            raise ValueError(f"VAD 模型文件异常（过小）: {model_path}")

        silero = sherpa_onnx.SileroVadModelConfig(
            model=str(model_path),
            min_silence_duration=min_silence_duration,
            min_speech_duration=min_speech_duration,
            max_speech_duration=max_speech_duration,
        )
        config = sherpa_onnx.VadModelConfig(
            silero_vad=silero,
            sample_rate=sample_rate,
            num_threads=num_threads,
            provider=provider,
        )
        self._vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=max_speech_duration + 5.0
        )

    @property
    def available(self) -> bool:
        return self._vad is not None

    def is_speech_detected(self) -> bool:
        return self._vad is not None and self._vad.is_speech_detected()

    def feed(self, samples: np.ndarray) -> List[np.ndarray]:
        """送入音频，返回本块内完成的语音段列表。"""
        if self._vad is None:
            return []
        self._vad.accept_waveform(samples.astype(np.float32))
        segments: List[np.ndarray] = []
        while not self._vad.empty():
            segments.append(np.asarray(self._vad.front, dtype=np.float32))
            self._vad.pop()
        return segments

    def flush(self) -> List[np.ndarray]:
        if self._vad is None:
            return []
        self._vad.flush()
        segments: List[np.ndarray] = []
        while not self._vad.empty():
            segments.append(np.asarray(self._vad.front, dtype=np.float32))
            self._vad.pop()
        return segments


class AsrPipeline:
    """KWS 唤醒 → VAD + 流式 ASR → final 的状态机。"""

    LISTENING = "listening"
    RECORDING = "recording"

    def __init__(
        self,
        kws: KwsEngine,
        asr: StreamingAsrEngine,
        vad: Optional[VadSegmenter] = None,
        max_record_sec: float = 6.0,
    ):
        self.kws = kws
        self.asr = asr
        self.vad = vad
        self.max_record_samples = int(max_record_sec * asr.sample_rate)
        self.state = self.LISTENING
        self._stream = None
        self._recorded = 0
        self._speech_seen = False
        self._silence_samples = 0

    # ------------------------------------------------------------------ #
    def feed(self, samples: np.ndarray) -> List[Dict]:
        events: List[Dict] = []
        if self.state == self.LISTENING:
            keyword = self.kws.feed(samples)
            if keyword:
                events.append({"type": "wake", "keyword": keyword})
                self._start_recording()
            return events

        # RECORDING
        self._recorded += len(samples)
        if self.vad is not None and self.vad.available:
            for segment in self.vad.feed(samples):
                self._speech_seen = True
                self._silence_samples = 0
                self.asr.accept(self._stream, segment)
                self.asr.decode(self._stream)
            if self._stream is not None:
                text = self.asr.partial_text(self._stream)
                if text:
                    events.append({"type": "partial", "text": text})
            if self._speech_seen and not self.vad.is_speech_detected():
                self._silence_samples += len(samples)
                if self._silence_samples > 0.6 * self.asr.sample_rate:
                    events.extend(self._finalize())
            else:
                self._silence_samples = 0
        else:
            # 无 VAD：固定窗口，达到 max_record_sec 直接 finalize
            self.asr.accept(self._stream, samples)
            self.asr.decode(self._stream)
            text = self.asr.partial_text(self._stream)
            if text:
                events.append({"type": "partial", "text": text})

        if self._recorded >= self.max_record_samples:
            events.extend(self._finalize())
        return events

    # ------------------------------------------------------------------ #
    def _start_recording(self) -> None:
        self.state = self.RECORDING
        self._stream = self.asr.new_stream()
        self._recorded = 0
        self._speech_seen = False
        self._silence_samples = 0

    def _finalize(self) -> List[Dict]:
        events: List[Dict] = []
        if self.vad is not None and self.vad.available:
            for segment in self.vad.flush():
                self.asr.accept(self._stream, segment)
        text = self.asr.finalize(self._stream) if self._stream is not None else ""
        self._stream = None
        self.state = self.LISTENING
        if text:
            events.append({"type": "final", "text": text})
        return events

    def flush(self) -> List[Dict]:
        """输入流结束时调用，强制 finalize。"""
        if self.state == self.RECORDING:
            return self._finalize()
        return []
