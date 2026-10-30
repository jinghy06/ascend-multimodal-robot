"""LLM 辅助函数测试：JSON 提取与系统提示词（不需要模型权重）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_control.command_rules import CommandCatalog
from voice_control.llm_parser import build_system_prompt, extract_json


@pytest.fixture(scope="module")
def catalog():
    return CommandCatalog()


# ---------------------------------------------------------------------- #
def test_extract_json_pure():
    obj, err = extract_json('{"task_id": "grasp", "params": {"color": "red"}}')
    assert err == ""
    assert obj["task_id"] == "grasp"


def test_extract_json_with_surrounding_junk():
    raw = '好的！{"task_id": "nav_to", "params": {"location": "kitchen"}} 完成。'
    obj, err = extract_json(raw)
    assert err == ""
    assert obj["task_id"] == "nav_to"


def test_extract_json_with_code_fence():
    raw = '```json\n{"task_id": "pause", "params": {}}\n```'
    obj, err = extract_json(raw)
    assert err == ""
    assert obj["task_id"] == "pause"


def test_extract_json_invalid():
    obj, err = extract_json("没有 JSON")
    assert obj is None
    assert err


def test_extract_json_broken():
    obj, err = extract_json('{"task_id": "pause"')
    assert obj is None


def test_system_prompt_contains_whitelist(catalog):
    prompt = build_system_prompt(catalog)
    for cid in catalog.command_ids:
        assert cid in prompt
    assert "unknown" in prompt
    assert "只输出" in prompt  # 强约束 JSON 输出
