"""Turn a trusted task event into speech. Qwen chooses phrasing, never the event."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable
from urllib.request import ProxyHandler, Request, build_opener


# These keys are emitted by task/perception code after checking state. An ASR
# transcript, a classifier prediction or an accepted command is not an event.
PHRASES = {
    "follow_started": ("开始跟随", (
        "好，我跟上了。你往前走就行。", "跟上你了，我在后面。", "好，你走吧，我跟着。")),
    "follow_target_lost_stopped": ("跟随目标丢失，底盘已停", (
        "我刚才没看见你，先停在这儿了。", "我跟丢了，已经停下。", "没跟上你，我先停在这儿了。")),
    "grasp_failed_not_held_stopped": ("夹爪没有夹住，抓取已结束", (
        "还没拿到，刚才没夹稳。", "这次没夹住，没拿到。", "没拿到，刚才夹爪没夹住。")),
    "grasp_confirmed_held": ("已抬起目标并确认夹稳", (
        "拿到了，夹稳了。", "好，这次夹住了。", "拿起来了，东西在夹爪里。")),
    "two_red_blocks_left_right": ("确认左右各一块红方块，尚未抓取", (
        "有两个红方块。你要左边那个，还是右边那个？",
        "我看到两块红的。左边的还是右边的？",
        "左边和右边各有一块红方块，你说的是哪个？")),
    "stopped_confirmed": ("控制层确认已停止", (
        "停好了。", "好，已经停下了。", "我停住了。")),
    "speech_unclear": ("没有听清，未发出任务", (
        "刚才没听清，你再说一遍？", "我没听清那句，能再说一次吗？", "刚那句没听清，再说一遍就好。")),
}
TONES = {"natural": "平常说话", "short": "尽量短", "gentle": "稍微柔和"}


@dataclass(frozen=True)
class Feedback:
    text: str
    variant: str
    source: str


def make_style_request(event: str, tone: str = "natural", previous: str | None = None):
    """Build a one-token grammar and server payload from closed vocabulary only."""
    facts, choices = PHRASES[event]
    style = TONES[tone]
    allowed = {key: phrase for key, phrase in zip("ABC", choices) if key != previous}
    menu = "\n".join(f"{key} {phrase}" for key, phrase in allowed.items())
    grammar = 'root ::= ' + ' | '.join(json.dumps(key) for key in allowed)
    return allowed, {
        "messages": [{"role": "user", "content": f"小车状态：{facts}。语气：{style}。选一句最自然的中文语音回复，只给字母。\n{menu}"}],
        "grammar": grammar,
        "max_tokens": 1,
        "temperature": 0.35,
        "top_p": 1.0,
        "top_k": 0,
        "stream": False,
        "cache_prompt": False,
    }


def render_feedback(event: str, select: Callable[[dict], str] | None = None, *,
                    tone: str = "natural", previous: str | None = None) -> Feedback:
    """Unknown state fails closed; invalid model text never reaches TTS."""
    if event not in PHRASES:
        return Feedback("我暂时确认不了现在的状态。", "", "unknown")
    allowed, request = make_style_request(event, tone if tone in TONES else "natural", previous)
    first = next(iter(allowed))
    # Confirming a stop should be immediate and independent of generation.
    if select is not None and event != "stopped_confirmed":
        try:
            choice = select(request)
            if isinstance(choice, str) and choice in allowed:
                return Feedback(allowed[choice], choice, "qwen")
        except (OSError, ValueError, LookupError, TypeError, AttributeError):
            pass
    return Feedback(allowed[first], first, "local")


class LocalLlamaStyleClient:
    """Optional loopback-only, no-proxy llama.cpp style selection."""

    def __init__(self, port: int = 8080, timeout: float = 0.8):
        if not isinstance(port, int) or not 1 <= port <= 65535 or timeout <= 0:
            raise ValueError("invalid local feedback endpoint")
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))

    def __call__(self, payload: dict) -> str:
        req = Request(self.url, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      {"Content-Type": "application/json"})
        with self.opener.open(req, timeout=self.timeout) as response:
            result = json.load(response)
        return result["choices"][0]["message"]["content"]
