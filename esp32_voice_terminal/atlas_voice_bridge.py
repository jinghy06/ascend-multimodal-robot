"""Atlas 端语音读取与语音回复桥接服务（阶段 1）。

数据流：
  UDP 127.0.0.1:5004 (16kHz int16 PCM, 20ms/640B, 来自 ESP32 USB 串口复用器)
    -> sherpa-onnx KWS ("你好小车"/"小车小车") + silero-VAD + streaming-zipformer 中文 ASR
    -> voice_control.router.Router 规则优先解析（未命中时可选调用本机 llama-server 8080 自由回复）
    -> voice_control.tts_engine.TTSEngine (VITS 中文离线合成 + 高频短句缓存)
    -> 16kHz int16 PCM 按 20ms 节拍发往 UDP 127.0.0.1:5006 -> ESP32 MAX98357A 播报
"""

from __future__ import annotations

import argparse
import json
import queue
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
PCM_FRAME_BYTES = FRAME_SAMPLES * 2  # 640 bytes = 20 ms
CHUNK_FRAMES = 3  # 60 ms = 960 samples fed to AsrPipeline


def add_voice_control_to_path(voice_root: Path) -> None:
    root_str = str(voice_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


SYSTEM_PROMPT = "你是机器人小车助手，用简短自然的中文口语回答，不超过25字。"
SPLIT_PUNCT = set("，。！？；,!?;：:\n")


def stream_local_llama(text: str, url: str, on_sentence, timeout: float = 15.0) -> str | None:
    """流式调用本机 llama-server，每凑满一小句立刻回调 on_sentence 实现边生成边播报。"""
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.4,
        "max_tokens": 42,
        "cache_prompt": True,
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    full_chunks: list[str] = []
    pending = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                data = json.loads(body)
                delta = data["choices"][0].get("delta", {}).get("content", "")
                if not delta:
                    continue
                full_chunks.append(delta)
                pending += delta
                if (len(pending) >= 4 and pending[-1] in SPLIT_PUNCT) or len(pending) >= 12:
                    seg = pending.strip()
                    if seg and on_sentence is not None:
                        on_sentence(seg)
                    pending = ""
        rem = pending.strip()
        if rem and on_sentence is not None:
            on_sentence(rem)
        result = "".join(full_chunks).strip()
        return result or None
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError, OSError):
        rem = pending.strip()
        if rem and on_sentence is not None:
            on_sentence(rem)
        result = "".join(full_chunks).strip()
        return result or None


def query_local_llama(text: str, url: str, timeout: float = 15.0) -> str | None:
    return stream_local_llama(text, url, on_sentence=None, timeout=timeout)


class SpeakerStream:
    """将文本经 VITS 合成为 16kHz int16 PCM，并按 20ms 节拍发送到 UDP 5006。"""

    def __init__(self, tts_engine, target_host: str, target_port: int, resample_fn):
        self.tts = tts_engine
        self.target = (target_host, target_port)
        self.resample_fn = resample_fn
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cache: dict[str, bytes] = {}
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=8)
        self._speaking_until = 0.0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @property
    def is_speaking(self) -> bool:
        return time.monotonic() < self._speaking_until

    def say(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        self.sock.close()

    def _synthesize_pcm16k(self, text: str) -> bytes:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        t0 = time.monotonic()
        audio = self.tts.synthesize(text)
        samples = np.asarray(audio.samples, dtype=np.float32)
        if audio.sample_rate != SAMPLE_RATE:
            samples = self.resample_fn(samples, audio.sample_rate, SAMPLE_RATE)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        rem = len(pcm) % PCM_FRAME_BYTES
        if rem:
            pcm += b"\x00" * (PCM_FRAME_BYTES - rem)
        dt = time.monotonic() - t0
        print(f"[TTS] '{text}' -> {len(pcm)}B ({len(pcm)/32000:.2f}s audio) in {dt:.2f}s", flush=True)
        if len(text) <= 24 and len(self._cache) < 64:
            self._cache[text] = pcm
        return pcm

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                pcm = self._synthesize_pcm16k(text)
                total_frames = len(pcm) // PCM_FRAME_BYTES
                self._speaking_until = time.monotonic() + total_frames * 0.02 + 0.35
                next_tick = time.monotonic()
                for offset in range(0, len(pcm), PCM_FRAME_BYTES):
                    self.sock.sendto(pcm[offset : offset + PCM_FRAME_BYTES], self.target)
                    next_tick += 0.02
                    sleep_for = next_tick - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
            except Exception as exc:
                print(f"[TTS] error: {exc}", flush=True)


class EspTextSpeaker:
    """将中文回复文本发往 UDP 5007 -> atlas_serial_control -> ESP32-S3 本地 ESP-TTS 流式播报。"""

    def __init__(self, target_host: str, target_port: int):
        self.target = (target_host, target_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._speaking_until = 0.0

    @property
    def is_speaking(self) -> bool:
        return time.monotonic() < self._speaking_until

    def say(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        # 累加估算播报时长做半双工保护（支持流式分段连续追加）
        now = time.monotonic()
        base = max(now, self._speaking_until)
        self._speaking_until = base + max(0.7, len(text) * 0.20 + 0.30)
        try:
            self.sock.sendto(text.encode("utf-8"), self.target)
            print(f"[ESP-TTS] -> '{text}'", flush=True)
        except OSError as exc:
            print(f"[ESP-TTS] send error: {exc}", flush=True)

    def close(self) -> None:
        self.sock.close()


class SubtitleRenderer:
    """将顶部(听到的声音)和底部(对话回复)渲染为 160x28 1-bit 位图(560B)发往 ESP32 叠加显示。"""

    FONT_CANDIDATES = (
        "/usr/share/fonts/cjkuni-ukai/ukai.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    )

    def __init__(self, target_host: str, target_port: int):
        self.target = (target_host, target_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._font = None
        self._Image = None
        self._ImageDraw = None
        self._last_pair = ("", "")
        try:
            from PIL import Image, ImageDraw, ImageFont

            self._Image = Image
            self._ImageDraw = ImageDraw
            for fp in self.FONT_CANDIDATES:
                if Path(fp).is_file():
                    self._font = ImageFont.truetype(fp, 12)
                    break
            if self._font is None:
                self._font = ImageFont.load_default()
        except Exception as exc:
            print(f"[SUBTITLE] Pillow/font unavailable: {exc}", flush=True)

    def _fit_tail(self, text: str, max_chars: int = 12) -> str:
        text = (text or "").replace("\n", " ").strip()
        if len(text) <= max_chars:
            return text
        return "…" + text[-(max_chars - 1) :]

    def update(self, heard: str, reply: str) -> None:
        if self._Image is None or self._font is None:
            return
        pair = (heard or "", reply or "")
        if pair == self._last_pair:
            return
        self._last_pair = pair
        img = self._Image.new("1", (160, 28), 0)
        draw = self._ImageDraw.Draw(img)
        if pair[0]:
            draw.text((2, 1), self._fit_tail(pair[0]), fill=1, font=self._font)
        if pair[1]:
            draw.text((2, 15), self._fit_tail(pair[1]), fill=1, font=self._font)
        raw = img.tobytes()
        if len(raw) == 560:
            try:
                self.sock.sendto(raw, self.target)
            except OSError:
                pass

    def close(self) -> None:
        self.sock.close()


def write_command_record(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas ESP32 UDP voice bridge (ASR + reply + VITS TTS)")
    parser.add_argument("--voice-root", default="/root/ascend-multimodal-robot/voice_control")
    parser.add_argument("--mic-port", type=int, default=5004)
    parser.add_argument("--speaker-host", default="127.0.0.1")
    parser.add_argument("--speaker-port", type=int, default=5006)
    parser.add_argument("--tts-port", type=int, default=5007)
    parser.add_argument("--subtitle-port", type=int, default=5008)
    parser.add_argument("--tts-mode", choices=("esp", "vits", "none"), default="esp", help="esp=发文本给ESP32本地ESP-TTS, vits=Atlas本地VITS, none=静默")
    parser.add_argument("--llama-url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--command-out", default="/root/so101-lekiwi/last_voice_command.json")
    parser.add_argument("--no-kws", action="store_true", help="跳过唤醒词，直接进行流式语音识别")
    parser.add_argument("--wake-reply", default="我在")
    parser.add_argument("--mic-gain", type=float, default=1.0, help="软件麦克风增益（默认 1.0 保持原波形）")
    parser.add_argument("--max-record-sec", type=float, default=5.0, help="唤醒后单次指令最长录音秒数")
    args = parser.parse_args()

    voice_root = Path(args.voice_root)
    add_voice_control_to_path(voice_root)

    from voice_control.asr_engine import AsrPipeline, StreamingAsrEngine, VadSegmenter
    from voice_control.audio_source import resample_linear
    from voice_control.command_rules import CommandCatalog
    from voice_control.kws_engine import KwsEngine
    from voice_control.router import Router
    from voice_control.tts_engine import TTSConfig, TTSEngine

    models_dir = voice_root / "models"
    catalog = CommandCatalog(voice_root / "configs" / "commands.yaml")
    router = Router(catalog, llm=None)

    class _AlwaysWake:
        def feed(self, _samples):
            return "always"

        def reset(self):
            pass

    if args.no_kws:
        kws = _AlwaysWake()
    else:
        kws = KwsEngine(
            models_dir / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
            keywords_file=voice_root / "configs" / "keywords.txt",
        )
    asr = StreamingAsrEngine(models_dir / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20")
    vad = VadSegmenter(models_dir / "silero_vad_v4.onnx")
    pipeline = AsrPipeline(kws, asr, vad=vad, max_record_sec=args.max_record_sec)

    speaker = None
    if args.tts_mode == "esp":
        speaker = EspTextSpeaker(args.speaker_host, args.tts_port)
    elif args.tts_mode == "vits":
        tts_cfg = TTSConfig.from_yaml(voice_root / "configs" / "tts.yaml")
        tts_cfg.play = False
        tts_cfg.num_threads = 4
        tts = TTSEngine(tts_cfg)
        speaker = SpeakerStream(tts, args.speaker_host, args.speaker_port, resample_linear)
        if args.wake_reply:
            try:
                speaker._synthesize_pcm16k(args.wake_reply)
            except Exception as exc:
                print(f"[TTS] warmup failed: {exc}", flush=True)
    subtitle = SubtitleRenderer(args.speaker_host, args.subtitle_port)

    mic_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mic_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    mic_sock.bind(("127.0.0.1", args.mic_port))
    mic_sock.settimeout(0.2)

    running = True

    def handle_stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    print(
        f"[VOICE] Bridge ready: mic=127.0.0.1:{args.mic_port} -> spk={args.speaker_host}:{args.speaker_port} "
        f"(require_wake={not args.no_kws})",
        flush=True,
    )

    pcm_acc = bytearray()
    command_out = Path(args.command_out)
    try:
        while running:
            try:
                data, _ = mic_sock.recvfrom(2048)
            except socket.timeout:
                continue
            if len(data) != PCM_FRAME_BYTES:
                continue
            if speaker is not None and speaker.is_speaking:
                pcm_acc.clear()
                continue
            pcm_acc.extend(data)
            if len(pcm_acc) < PCM_FRAME_BYTES * CHUNK_FRAMES:
                continue
            samples = np.frombuffer(pcm_acc, dtype=np.int16).astype(np.float32) / 32768.0
            if args.mic_gain != 1.0:
                samples = np.clip(samples * args.mic_gain, -1.0, 1.0)
            pcm_acc.clear()
            events = pipeline.feed(samples)
            for event in events:
                etype = event.get("type")
                if etype == "wake":
                    print(f"[WAKE] {event}", flush=True)
                    subtitle.update(f"听:{event.get('keyword', '你好小车')}", f"答:{args.wake_reply}" if args.wake_reply else "")
                    if speaker is not None and args.wake_reply:
                        speaker.say(args.wake_reply)
                elif etype == "partial":
                    ptext = (event.get("text") or "").strip()
                    print(f"[PARTIAL] {ptext}", flush=True)
                    if ptext:
                        subtitle.update(f"听:{ptext}", "")
                elif etype == "final":
                    text = (event.get("text") or "").strip()
                    if not text:
                        continue
                    subtitle.update(f"听:{text}", "思考中…")
                    outcome = router.route(text)
                    reply = outcome.reply
                    source = outcome.source
                    streamed_already = False
                    if not outcome.ok and args.llama_url:
                        shown_reply = []

                        def _on_seg(seg):
                            shown_reply.append(seg)
                            subtitle.update(f"听:{text}", f"答:{''.join(shown_reply)}")
                            if speaker is not None:
                                speaker.say(seg)

                        llm_reply = stream_local_llama(
                            text,
                            args.llama_url,
                            on_sentence=_on_seg,
                        )
                        if llm_reply:
                            reply = llm_reply
                            source = "llama_chat"
                            streamed_already = True
                    if not reply:
                        reply = catalog.fallback_reply
                    subtitle.update(f"听:{text}", f"答:{reply}")
                    record = {
                        "timestamp": time.time(),
                        "text": text,
                        "ok": outcome.ok,
                        "source": source,
                        "task_id": outcome.command_id,
                        "params": outcome.params,
                        "reply": reply,
                    }
                    print(f"[FINAL] {json.dumps(record, ensure_ascii=False)}", flush=True)
                    try:
                        write_command_record(command_out, record)
                    except OSError:
                        pass
                    if speaker is not None and not streamed_already:
                        speaker.say(reply)
    finally:
        mic_sock.close()
        subtitle.close()
        if speaker is not None:
            speaker.close()


if __name__ == "__main__":
    main()
