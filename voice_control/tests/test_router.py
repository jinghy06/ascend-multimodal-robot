"""路由层测试：FakeLLM 注入，验证 规则优先/LLM兜底/三级防护。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_control.command_rules import CommandCatalog
from voice_control.router import Router


class FakeLlm:
    def __init__(self, result=None, exc=None):
        self.result = result or {"ok": False, "error": "fake"}
        self.exc = exc
        self.calls = 0

    def parse(self, text):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.result


@pytest.fixture(scope="module")
def catalog():
    return CommandCatalog()


# ---------------------------------------------------------------------- #
def test_rule_hit_skips_llm(catalog):
    llm = FakeLlm({"ok": True, "task_id": "grasp", "params": {"color": "red", "object": "cup"}})
    outcome = Router(catalog, llm).route("跟着我")
    assert outcome.source == "rule"
    assert outcome.command_id == "follow_start"
    assert llm.calls == 0


def test_llm_path_valid(catalog):
    llm = FakeLlm({"ok": True, "task_id": "grasp", "params": {"color": "red", "object": "cup"}})
    outcome = Router(catalog, llm).route("帮我把红色杯子拿起来")
    assert outcome.source == "llm"
    assert outcome.ok
    assert outcome.command_id == "grasp"
    assert llm.calls == 1


def test_llm_whitelist_violation_falls_back(catalog):
    llm = FakeLlm({"ok": True, "task_id": "dance", "params": {}})
    outcome = Router(catalog, llm).route("跳个舞")
    assert outcome.source == "fallback"
    assert not outcome.ok
    assert outcome.command_id == "unknown"


def test_llm_exception_falls_back(catalog):
    llm = FakeLlm(exc=RuntimeError("npu oom"))
    outcome = Router(catalog, llm).route("随便说句什么")
    assert outcome.source == "fallback"
    assert "npu oom" in outcome.error


def test_no_llm_falls_back(catalog):
    outcome = Router(catalog, None).route("随便说句什么")
    assert outcome.source == "fallback"
    assert outcome.reply == catalog.fallback_reply


def test_fallback_never_routed(catalog):
    for llm in (None, FakeLlm({"ok": False, "error": "x"})):
        outcome = Router(catalog, llm).route("听不懂的话xyz")
        payload = __import__("json").loads(outcome.command_json())
        assert payload["ok"] is False
        assert payload["task_id"] == "unknown"
