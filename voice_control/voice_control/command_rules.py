"""规则指令解析器：模板 + 槽位同义词填充。

纯 Python 实现（仅依赖 PyYAML），Linux/Windows/香橙派均可运行。
预定义指令必须 100% 路由正确，因此规则层优先于 LLM，且本模块不含任何随机性。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "commands.yaml"

_PUNCT_RE = re.compile(r"[\s，。！？；：、,.!?;:\~～·…\"'“”‘’（）()\[\]【】]+")
_SLOT_TOKEN_RE = re.compile(r"\{(\w+)\}")


def normalize(text: str) -> str:
    """归一化：NFKC、去标点空白、小写。中英文混合指令足够用。"""
    text = unicodedata.normalize("NFKC", text)
    return _PUNCT_RE.sub("", text).lower()


@dataclass
class Match:
    command_id: str
    params: Dict[str, str] = field(default_factory=dict)      # 编码值
    surface: Dict[str, str] = field(default_factory=dict)     # 用户原词（用于回复渲染）
    reply: str = ""


@dataclass
class _CompiledCommand:
    command_id: str
    params_decl: List[str]
    reply_tpl: str
    patterns: List[Tuple[re.Pattern, List[str]]] = field(default_factory=list)


class CommandCatalog:
    """从 commands.yaml 加载指令目录，编译为带槽位的正则模板。

    同一指令多条模板按序匹配，先命中先得；目录顺序即优先级。
    """

    def __init__(self, config_path: Path | str = CONFIG_PATH):
        self.config_path = Path(config_path)
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.fallback_reply: str = data.get("fallback_reply", "没听懂，请再说一次。")
        self._slots: Dict[str, Dict[str, str]] = {
            k: {normalize(w): code for w, code in v.items()}
            for k, v in (data.get("slots") or {}).items()
        }
        self._slot_words: Dict[str, List[str]] = {
            k: sorted(v, key=len, reverse=True) for k, v in self._slots.items()
        }
        self._commands: Dict[str, _CompiledCommand] = {}
        self._order: List[str] = []
        for spec in data.get("commands") or []:
            cmd = _CompiledCommand(
                command_id=spec["id"],
                params_decl=list(spec.get("params") or []),
                reply_tpl=spec.get("reply", ""),
            )
            for tpl in spec.get("templates") or []:
                cmd.patterns.append(self._compile_template(tpl))
            self._commands[cmd.command_id] = cmd
            self._order.append(cmd.command_id)

    # ------------------------------------------------------------------ #
    def _compile_template(self, template: str) -> Tuple[re.Pattern, List[str]]:
        pattern_parts: List[str] = []
        slot_seq: List[str] = []
        pos = 0
        for m in _SLOT_TOKEN_RE.finditer(template):
            pattern_parts.append(re.escape(normalize(template[pos:m.start()])))
            slot = m.group(1)
            if slot not in self._slot_words:
                raise ValueError(f"模板 {template!r} 引用了未定义的槽位 {slot!r}")
            alternation = "|".join(re.escape(w) for w in self._slot_words[slot])
            pattern_parts.append(f"({alternation})")
            slot_seq.append(slot)
            pos = m.end()
        pattern_parts.append(re.escape(normalize(template[pos:])))
        return re.compile("^" + "".join(pattern_parts) + "$"), slot_seq

    # ------------------------------------------------------------------ #
    def match(self, text: str) -> Optional[Match]:
        norm = normalize(text)
        for cid in self._order:
            cmd = self._commands[cid]
            for pattern, slot_seq in cmd.patterns:
                m = pattern.match(norm)
                if not m:
                    continue
                surface = {slot: m.group(i + 1) for i, slot in enumerate(slot_seq)}
                params = {
                    slot: self._slots[slot][word] for slot, word in surface.items()
                }
                return Match(
                    command_id=cid,
                    params=params,
                    surface=surface,
                    reply=self.render_reply(cmd.reply_tpl, surface),
                )
        return None

    # ------------------------------------------------------------------ #
    def render_reply(self, template: str, surface: Dict[str, str]) -> str:
        reply = template
        for slot, word in surface.items():
            reply = reply.replace("{" + slot + "}", word)
        return reply

    # ------------------------------------------------------------------ #
    def validate_parse(self, task_id: str, params: Dict[str, str]) -> Tuple[bool, str]:
        """校验 LLM 解析结果是否在指令目录白名单内。返回 (ok, error)。"""
        if task_id == "unknown":
            return False, "task_id 为 unknown"
        cmd = self._commands.get(task_id)
        if cmd is None:
            return False, f"task_id {task_id!r} 不在白名单"
        for key, value in (params or {}).items():
            base_key = key.removesuffix("_guess")
            if base_key not in cmd.params_decl:
                return False, f"参数 {key!r} 不是任务 {task_id} 的声明参数"
            if base_key in self._slots and value not in set(self._slots[base_key].values()):
                return False, f"参数 {key}={value!r} 不在槽位取值集合内"
        missing = [p for p in cmd.params_decl if p not in params]
        if missing:
            return False, f"缺少必要参数 {missing}"
        return True, ""

    # ------------------------------------------------------------------ #
    def llm_catalog_text(self) -> str:
        """给 LLM 系统提示用的目录文本（白名单 + 参数取值编码）。"""
        lines = []
        for cid in self._order:
            cmd = self._commands[cid]
            if cmd.params_decl:
                parts = []
                for p in cmd.params_decl:
                    codes = sorted(set(self._slots[p].values()))
                    parts.append(f"{p} ∈ {codes}")
                lines.append(f"- {cid} | {'; '.join(parts)}")
            else:
                lines.append(f"- {cid} | 无参数")
        return "\n".join(lines)

    @property
    def command_ids(self) -> List[str]:
        return list(self._order)
