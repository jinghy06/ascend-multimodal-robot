"""LLM 解析与回复生成。

设计要点：
- torch / transformers 惰性导入：本模块可在无 GPU/无 torch 的环境导入并跑单测；
- 两次调用：parse() 只输出短 JSON（先路由先动作），stream_reply() 流式生成中文回复；
- Qwen3 混合思考模型必须显式关闭思考模式（enable_thinking=False），否则延迟爆炸
  且 <think> 标签会污染 JSON；
- 所有输出经 CommandCatalog 白名单校验后才允许路由（见 Router）。

设备优先级：NPU（torch_npu）→ CUDA → CPU，符合香橙派 310B / PC 训练机 / 普通电脑三种环境。
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import yaml

from .command_rules import CommandCatalog

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_zh.txt"
LLM_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "llm.yaml"


@dataclass
class LLMConfig:
    model_path: str = "models/Qwen2.5-0.5B-Instruct"
    device: str = "auto"
    dtype: str = "auto"
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_parse_tokens: int = 64
    max_reply_tokens: int = 64
    enable_thinking: bool = False
    request_timeout_sec: float = 5.0

    @classmethod
    def from_yaml(cls, path: Path | str = LLM_CONFIG_PATH) -> "LLMConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


def extract_json(text: str) -> Tuple[Optional[dict], str]:
    """从模型输出中提取第一个 JSON 对象。返回 (obj, error)。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, "输出中没有 JSON 对象"
    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"JSON 解析失败: {exc}"
    if not isinstance(obj, dict):
        return None, "JSON 不是对象"
    return obj, ""


def build_system_prompt(catalog: CommandCatalog) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{catalog}", catalog.llm_catalog_text())


class LLMService:
    """加载本地 Qwen，提供 parse / stream_reply 两次调用接口。"""

    def __init__(self, config: LLMConfig, catalog: CommandCatalog):
        self.config = config
        self.catalog = catalog
        self.system_prompt = build_system_prompt(catalog)
        self._tokenizer = None
        self._model = None
        self._device = None

    # ------------------------------------------------------------------ #
    def _load(self):
        """惰性加载模型（首次调用时才 import torch / transformers）。"""
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.config.device
        if device == "auto":
            if hasattr(torch, "npu") and torch.npu.is_available():
                device = "npu"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        dtype = self.config.dtype
        if dtype == "auto":
            dtype = "bfloat16" if device == "npu" else (
                "float16" if device == "cuda" else "float32"
            )

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"LLM 模型目录不存在: {model_path}（用 tools/download_models.py 下载）"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=getattr(torch, dtype),
        ).to(device)
        self._model.eval()
        self._device = device

    def _apply_chat(self, user_text: str, extra_system: str = "") -> str:
        system = self.system_prompt + extra_system
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]
        kwargs = dict(messages, add_generation_prompt=True, tokenize=False)
        try:
            return self._tokenizer.apply_chat_template(
                **kwargs, enable_thinking=self.config.enable_thinking
            )
        except TypeError:
            # Qwen2.5 等不支持 enable_thinking 参数的模板
            return self._tokenizer.apply_chat_template(**kwargs)

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ #
    def parse(self, text: str, retries: int = 1) -> Dict:
        """解析调用：输出短 JSON，校验 + 重试，返回结构化结果。"""
        self._load()
        user = f"用户指令：{text}"
        last_error = ""
        for attempt in range(retries + 1):
            extra = ""
            if last_error:
                extra = (
                    "\n上一次你的输出存在问题，请修正后重新输出，"
                    f"只输出 JSON：{last_error}"
                )
            raw = self._generate(
                self._apply_chat(user, extra), self.config.max_parse_tokens
            )
            obj, err = extract_json(raw)
            if obj is None:
                last_error = err
                continue
            task_id = str(obj.get("task_id") or "unknown")
            params = obj.get("params")
            if not isinstance(params, dict):
                params = {}
            params = {str(k): str(v) for k, v in params.items()}
            ok, err = self.catalog.validate_parse(task_id, params)
            if ok:
                return {"ok": True, "task_id": task_id, "params": params,
                        "raw": raw, "reply": obj.get("reply", "")}
            last_error = err
        return {"ok": False, "error": last_error, "raw": raw}

    # ------------------------------------------------------------------ #
    def stream_reply(self, text: str) -> Iterator[str]:
        """回复调用：流式生成中文回复文案（供 TTS 边生成边播报）。"""
        self._load()
        from transformers import TextIteratorStreamer

        system_extra = (
            "\n现在请用一句简短的中文（不超过20个字）回应用户的指令，"
            "表明你已经理解并准备执行。只输出这句话本身。"
        )
        prompt = self._apply_chat(f"用户指令：{text}", system_extra)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        thread = threading.Thread(
            target=self._model.generate,
            kwargs=dict(
                **inputs,
                max_new_tokens=self.config.max_reply_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                streamer=streamer,
            ),
        )
        thread.start()
        for chunk in streamer:
            chunk = chunk.strip()
            if chunk:
                yield chunk
        thread.join()
