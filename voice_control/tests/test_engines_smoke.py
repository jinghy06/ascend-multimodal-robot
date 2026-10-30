"""引擎冒烟测试：需要真实模型（tools/download_models.py 下载后运行）。

覆盖：KwsEngine 构建与唤醒（官方测试音频 0.wav）、StreamingAsrEngine 构建、
VadSegmenter 可用性、AsrPipeline 状态机（静音监听不崩溃、强制录音 finalize）。
模型缺失时整文件 skip，不影响 CI/其他平台。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_control.asr_engine import AsrPipeline, StreamingAsrEngine, VadSegmenter
from voice_control.kws_engine import KwsEngine, load_kws_from_wav

MODELS = Path(__file__).resolve().parents[1] / "models"
KWS_DIR = MODELS / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
ASR_DIR = MODELS / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
VAD_MODEL = MODELS / "silero_vad_v4.onnx"
KWS_TEST_WAV = KWS_DIR / "test_wavs" / "0.wav"

pytestmark = pytest.mark.skipif(
    not (KWS_DIR / "tokens.txt").exists() or not (ASR_DIR / "tokens.txt").exists(),
    reason="模型未下载（tools/download_models.py）",
)

SR = 16000


@pytest.fixture(scope="module")
def kws():
    keywords = Path(__file__).resolve().parents[1] / "configs" / "keywords.txt"
    return KwsEngine(KWS_DIR, keywords_file=keywords, sample_rate=SR)


@pytest.fixture(scope="module")
def asr():
    return StreamingAsrEngine(ASR_DIR, sample_rate=SR)


def _tone(freq=440.0, seconds=0.5):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds=0.5):
    return np.zeros(int(SR * seconds), dtype=np.float32)


# ---------------------------------------------------------------------- #
def test_kws_no_false_wake_on_speech(kws):
    """官方 0.wav 是讲座语音（不含任何关键词）：连续语音不得误唤醒。"""
    if not KWS_TEST_WAV.exists():
        pytest.skip("无官方测试音频")
    hit = load_kws_from_wav(KWS_TEST_WAV, kws)
    assert hit is None, f"连续讲话误触发唤醒: {hit!r}"


def test_kws_no_wake_on_silence(kws):
    for _ in range(10):
        assert kws.feed(_silence()) is None


def test_asr_engine_constructed(asr):
    stream = asr.new_stream()
    asr.accept(stream, _tone())
    asr.decode(stream)
    asr.finalize(stream)  # 纯音无文字结果，但流程不得异常


def test_vad_available_when_model_exists():
    vad = VadSegmenter(VAD_MODEL, sample_rate=SR)
    if VAD_MODEL.exists():
        assert vad.available
    segments = vad.feed(_silence(0.5))
    assert isinstance(segments, list)


def test_pipeline_state_machine(kws, asr):
    vad = VadSegmenter(VAD_MODEL, sample_rate=SR) if VAD_MODEL.exists() else None
    pipeline = AsrPipeline(kws, asr, vad=vad, max_record_sec=2.0)

    # 监听态：静音不产出事件、状态不变
    for _ in range(10):
        assert pipeline.feed(_silence()) == []
    assert pipeline.state == AsrPipeline.LISTENING

    # 强制进入录音态，喂 1s 音 + 静音，finalize 后回到监听态
    pipeline._start_recording()
    assert pipeline.state == AsrPipeline.RECORDING
    events = pipeline.feed(_tone(seconds=1.0))
    for ev in events:
        assert ev["type"] in ("partial", "final")
    remaining = pipeline.flush()
    assert pipeline.state == AsrPipeline.LISTENING
    for ev in remaining:
        assert ev["type"] == "final"
