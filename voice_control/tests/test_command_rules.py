"""规则解析器测试：全部不依赖模型/ROS/torch，任何平台可跑。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_control.command_rules import CommandCatalog, normalize


@pytest.fixture(scope="module")
def catalog():
    return CommandCatalog()


# ---------------------------------------------------------------------- #
def test_normalize(catalog):
    assert normalize("去 桌子！") == "去桌子"
    assert normalize("抓起红色杯子。") == "抓起红色杯子"


def test_all_templates_match_themselves(catalog):
    """每条模板填槽后必须命中它自己声明的任务（目录自洽性）。"""
    import re

    token_re = re.compile(r"\{(\w+)\}")
    data = catalog.config_path.read_text(encoding="utf-8")
    specs = __import__("yaml").safe_load(data)
    n = 0
    for spec in specs["commands"]:
        for tpl in spec["templates"]:
            # 用每个槽位最短的词填充，如 "跟{distance}一点" -> "跟近一点"
            utterance = token_re.sub(
                lambda m: catalog._slot_words[m.group(1)][-1], tpl
            )
            m = catalog.match(utterance)
            assert m is not None, f"模板填槽后未命中: {tpl} -> {utterance}"
            assert m.command_id == spec["id"]
            n += 1
    assert n >= 20


def test_grasp_slots(catalog):
    m = catalog.match("抓起红色杯子")
    assert m.command_id == "grasp"
    assert m.params == {"color": "red", "object": "cup"}
    assert m.surface == {"color": "红色", "object": "杯子"}
    assert "红色杯子" in m.reply


def test_nav_to_slot(catalog):
    m = catalog.match("开到厨房去")
    assert m.command_id == "nav_to"
    assert m.params == {"location": "kitchen"}


def test_follow_adjust(catalog):
    m = catalog.match("跟远一点")
    assert m.command_id == "follow_adjust"
    assert m.params == {"distance": "far"}


def test_no_punctuation_variants(catalog):
    assert catalog.match("  停止跟随！!").command_id == "follow_stop"
    assert catalog.match("，跟我走。").command_id == "follow_start"


def test_unknown_returns_none(catalog):
    assert catalog.match("今天天气怎么样") is None
    assert catalog.match("") is None


def test_exactly_20_commands(catalog):
    assert len(catalog.command_ids) == 20


# ---------------------------------------------------------------------- #
def test_validate_parse_ok(catalog):
    ok, err = catalog.validate_parse("grasp", {"color": "red", "object": "cup"})
    assert ok, err


def test_validate_parse_rejects_unknown_task(catalog):
    ok, err = catalog.validate_parse("dance", {})
    assert not ok


def test_validate_parse_rejects_bad_param_value(catalog):
    ok, err = catalog.validate_parse("grasp", {"color": "crimson", "object": "cup"})
    assert not ok


def test_validate_parse_rejects_missing_param(catalog):
    ok, err = catalog.validate_parse("grasp", {"color": "red"})
    assert not ok


def test_llm_catalog_text(catalog):
    text = catalog.llm_catalog_text()
    for cid in catalog.command_ids:
        assert cid in text
    assert "red" in text  # 槽位编码出现在白名单中
