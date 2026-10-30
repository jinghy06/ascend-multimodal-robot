"""llm_service：独立进程运行 LLM（可用与 ROS 主环境不同的 venv）。

话题：
  订阅 /voice/llm_request  (std_msgs/String, JSON {"request_id":..., "text":...})
  发布 /voice/llm_result    (std_msgs/String, JSON {"request_id":..., "ok":...,
                                                "task_id":..., "params":..., "error":...})
  发布 /voice/reply         (std_msgs/String, JSON {"request_id":..., "chunk":...,
                                                   "done": bool})   —— 逐 chunk 流式回复

职责：解析调用（校验+重试，见 LLMService.parse）→ 路由结果；随后流式生成回复。
解析失败时由本节点发布兜底话术，parser_node 只处理超时/无服务的情况。
"""

from __future__ import annotations

import json
import uuid


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    from ..command_rules import CommandCatalog
    from ..llm_parser import LLMConfig, LLMService

    rclpy.init(args=args)

    class LlmServiceNode(Node):
        def __init__(self):
            super().__init__("llm_service")
            self.catalog = CommandCatalog()
            llm_yaml = self.declare_parameter(
                "llm_yaml", ""
            ).value
            cfg = LLMConfig.from_yaml(llm_yaml) if llm_yaml else LLMConfig()
            try:
                self.llm = LLMService(cfg, self.catalog)
                self.llm._load()
                self.get_logger().info(f"LLM 就绪: {cfg.model_path} @ {self.llm._device}")
            except Exception as exc:
                self.llm = None
                self.get_logger().error(f"LLM 加载失败（进入不可用状态）: {exc}")

            self.pub_result = self.create_publisher(String, "/voice/llm_result", 10)
            self.pub_reply = self.create_publisher(String, "/voice/reply", 10)
            self.create_subscription(String, "/voice/llm_request",
                                     self.on_request, 10)

        # -------------------------------------------------------------- #
        def on_request(self, msg):
            try:
                req = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            request_id = str(req.get("request_id") or uuid.uuid4())
            text = str(req.get("text") or "")
            result = {"request_id": request_id, "ok": False, "error": "llm unavailable"}
            if self.llm is not None:
                try:
                    result = self.llm.parse(text)
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
            result["request_id"] = request_id
            self.pub_result.publish(String(data=json.dumps(result, ensure_ascii=False)))
            self._stream_reply(request_id, text, bool(result.get("ok")))

        def _stream_reply(self, request_id, text, parsed_ok):
            if self.llm is None or not parsed_ok:
                self.pub_reply.publish(String(data=json.dumps(
                    {"request_id": request_id, "chunk": self.catalog.fallback_reply,
                     "done": True}, ensure_ascii=False)))
                return
            try:
                for chunk in self.llm.stream_reply(text):
                    self.pub_reply.publish(String(data=json.dumps(
                        {"request_id": request_id, "chunk": chunk,
                         "done": False}, ensure_ascii=False)))
            except Exception as exc:
                self.get_logger().error(f"回复生成失败: {exc}")
            finally:
                self.pub_reply.publish(String(data=json.dumps(
                    {"request_id": request_id, "chunk": "",
                     "done": True}, ensure_ascii=False)))

    node = LlmServiceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
