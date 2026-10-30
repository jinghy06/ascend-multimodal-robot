"""parser_node：文本 → 规则/LLM 路由 → 发布结构化指令与回复。

话题：
  订阅 /voice/utterance   (asr_node 产出的最终文本)
  发布 /voice/command     (JSON {"task_id":..., "params":..., "source":..., "ok":...})
                          → task_manager（接口表约定 topic）
  发布 /voice/reply       (规则路径的模板回复；LLM 路径的回复由 llm_service 流式发布)
  订阅 /voice/llm_result  (llm_service 的解析结果，按 request_id 关联)

设计：规则路径零 LLM 依赖；LLM 路径经 /voice/llm_request → /voice/llm_result，
解析失败或超时只回兜底话术，永不把未知指令路由给 task_manager。
"""

from __future__ import annotations

import json
import threading
import uuid


class RosLlmClient:
    """实现 Router 需要的 parse() 协议：发请求并等待关联结果。"""

    def __init__(self, node, timeout_sec: float):
        self._node = node
        self._timeout = timeout_sec
        self._pending: dict[str, threading.Event] = {}
        self._results: dict[str, dict] = {}

    def on_result(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        request_id = data.get("request_id")
        if request_id in self._pending:
            self._results[request_id] = data
            self._pending[request_id].set()

    def parse(self, text: str) -> dict:
        request_id = uuid.uuid4().hex
        event = threading.Event()
        self._pending[request_id] = event
        self._node.pub_llm_request.publish(json.dumps(
            {"request_id": request_id, "text": text}, ensure_ascii=False))
        if not event.wait(self._timeout):
            return {"ok": False, "error": f"llm timeout {self._timeout}s"}
        result = self._results.pop(request_id, {"ok": False, "error": "no result"})
        self._pending.pop(request_id, None)
        return result


def main(args=None):
    import rclpy
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import String

    from ..command_rules import CommandCatalog
    from ..router import Router

    rclpy.init(args=args)

    class ParserNode(Node):
        def __init__(self):
            super().__init__("parser_node")
            self.catalog = CommandCatalog()
            timeout = float(self.declare_parameter(
                "llm_timeout_sec", 5.0).value)
            self.group = ReentrantCallbackGroup()
            self.client = RosLlmClient(self, timeout)
            self.router = Router(self.catalog, llm=self.client)

            self.pub_command = self.create_publisher(String, "/voice/command", 10)
            self.pub_reply = self.create_publisher(String, "/voice/reply", 10)
            self.pub_llm_request = self.create_publisher(String, "/voice/llm_request", 10)
            self.create_subscription(String, "/voice/utterance", self.on_utterance,
                                     10, callback_group=self.group)
            self.create_subscription(String, "/voice/llm_result", self.client.on_result,
                                     10, callback_group=self.group)

        # -------------------------------------------------------------- #
        def on_utterance(self, msg):
            try:
                text = json.loads(msg.data).get("text", "")
            except json.JSONDecodeError:
                text = msg.data
            if not text.strip():
                return
            outcome = self.router.route(text)
            self.get_logger().info(
                f"路由: source={outcome.source} task={outcome.command_id} ok={outcome.ok}")
            self.pub_command.publish(String(data=outcome.command_json()))
            # LLM 路径的回复由 llm_service 流式发布；规则/兜底路径在这里发
            if outcome.source != "llm":
                self.pub_reply.publish(String(data=json.dumps(
                    {"request_id": "", "chunk": outcome.reply, "done": True},
                    ensure_ascii=False)))

    node = ParserNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
