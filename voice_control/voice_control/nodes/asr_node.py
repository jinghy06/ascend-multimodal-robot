"""asr_node：麦克风 → KWS 唤醒 → VAD + 流式 ASR → 发布文本。

话题：
  /voice/utterance (std_msgs/String, JSON {"text": ...})    finalized 指令文本
  /voice/partial   (std_msgs/String, JSON {"text": ...})    流式中间结果（可选消费）

参数（均可在启动时覆盖）：
  kws_dir / asr_dir / vad_model / keywords_file / sample_rate / provider
"""

from __future__ import annotations

import json
from pathlib import Path


def make_pipeline_from_params(node):
    from ..asr_engine import AsrPipeline, StreamingAsrEngine, VadSegmenter
    from ..kws_engine import KwsEngine

    def p(name, default):
        return node.declare_parameter(name, default).value

    kws_dir = p("kws_dir", "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile")
    asr_dir = p("asr_dir", "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20")
    vad_model = p("vad_model", "models/silero_vad_v4.onnx")
    keywords_file = p("keywords_file", str(Path(__file__).resolve().parents[2] / "configs" / "keywords.txt"))
    sample_rate = int(p("sample_rate", 16000))
    provider = p("provider", "cpu")

    kws = KwsEngine(kws_dir, keywords_file=keywords_file, sample_rate=sample_rate, provider=provider)
    asr = StreamingAsrEngine(asr_dir, sample_rate=sample_rate, provider=provider)
    vad = VadSegmenter(vad_model, sample_rate=sample_rate) if vad_model else None
    return AsrPipeline(kws, asr, vad=vad)


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init(args=args)

    class AsrNode(Node):
        def __init__(self):
            super().__init__("asr_node")
            self.pub_utter = self.create_publisher(String, "/voice/utterance", 10)
            self.pub_partial = self.create_publisher(String, "/voice/partial", 10)
            self.pipeline = make_pipeline_from_params(self)
            self.get_logger().info("asr_node 就绪，等待唤醒词……")

        def handle_events(self, events):
            for ev in events:
                if ev["type"] == "wake":
                    self.get_logger().info(f"唤醒: {ev['keyword']}")
                elif ev["type"] == "partial":
                    self.pub_partial.publish(String(data=json.dumps(
                        {"text": ev["text"]}, ensure_ascii=False)))
                elif ev["type"] == "final":
                    self.get_logger().info(f"识别结果: {ev['text']}")
                    self.pub_utter.publish(String(data=json.dumps(
                        {"text": ev["text"]}, ensure_ascii=False)))

    node = AsrNode()

    from ..audio_source import MicSource
    with MicSource(sample_rate=node.pipeline.asr.sample_rate) as mic:
        while rclpy.ok():
            chunk = mic.read()
            node.handle_events(node.pipeline.feed(chunk))
    node.handle_events(node.pipeline.flush())
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
