"""路由层：规则优先，LLM 兜底。

LLM 客户端通过协议方法注入（parse），便于测试与 ROS 解耦：
- parse(text) -> {"ok": bool, "task_id": str, "params": dict, "error": str}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol

from .command_rules import CommandCatalog


class LlmParseClient(Protocol):
    def parse(self, text: str) -> Dict: ...


@dataclass
class RouteOutcome:
    source: str                       # "rule" | "llm" | "fallback"
    command_id: str
    params: Dict[str, str] = field(default_factory=dict)
    reply: str = ""
    ok: bool = True                   # False 时不应路由到 task_manager
    error: str = ""

    def command_json(self) -> str:
        """发布到 /voice/command 的载荷。"""
        return json.dumps(
            {
                "task_id": self.command_id,
                "params": self.params,
                "source": self.source,
                "ok": self.ok,
            },
            ensure_ascii=False,
        )


class Router:
    def __init__(self, catalog: CommandCatalog, llm: Optional[LlmParseClient] = None):
        self.catalog = catalog
        self.llm = llm

    def route(self, text: str) -> RouteOutcome:
        # 1) 规则层（预定义指令，确定性命中）
        m = self.catalog.match(text)
        if m is not None:
            return RouteOutcome(
                source="rule",
                command_id=m.command_id,
                params=m.params,
                reply=m.reply,
            )

        # 2) LLM 兜底（开放词汇）
        err = ""
        if self.llm is not None:
            try:
                result = self.llm.parse(text)
            except Exception as exc:  # LLM 进程异常不得影响整机
                result = {"ok": False, "error": f"llm exception: {exc}"}
            if result.get("ok"):
                task_id = str(result.get("task_id") or "unknown")
                params = dict(result.get("params") or {})
                ok, err = self.catalog.validate_parse(task_id, params)
                if ok:
                    return RouteOutcome(
                        source="llm", command_id=task_id, params=params,
                        reply=str(result.get("reply") or ""),
                    )
                result = {"ok": False, "error": err or "whitelist check failed"}
            err = str(result.get("error") or "")

        # 3) 兜底话术（永不路由）
        return RouteOutcome(
            source="fallback",
            command_id="unknown",
            ok=False,
            reply=self.catalog.fallback_reply,
            error=err,
        )
