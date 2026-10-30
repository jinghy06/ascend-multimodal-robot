"""demo：文本 → 规则/LLM 路由（无 ROS，无麦克风）。

用法：
  规则路径：  python -m voice_control.demos.demo_parse_text
  LLM 路径：  python -m voice_control.demos.demo_parse_text --llm models/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse

from ..command_rules import CommandCatalog
from ..router import Router


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=None, help="本地模型目录（缺省=纯规则模式）")
    ap.add_argument("--llm-yaml", default=None)
    args = ap.parse_args()

    catalog = CommandCatalog()
    llm = None
    if args.llm:
        from ..llm_parser import LLMConfig, LLMService

        cfg = LLMConfig.from_yaml(args.llm_yaml) if args.llm_yaml else LLMConfig()
        cfg.model_path = args.llm
        llm = LLMService(cfg, catalog)
        print(f"[llm] 模型加载中: {cfg.model_path} ...")
        llm._load()
        print(f"[llm] 设备: {llm._device}")

    router = Router(catalog, llm=llm)
    print("输入中文指令（回车解析，q 退出）。纯规则模式只认 20 条预定义指令。")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text in ("q", "quit"):
            break
        outcome = router.route(text)
        print(f"  source={outcome.source}  task={outcome.command_id}  "
              f"params={outcome.params}  ok={outcome.ok}")
        print(f"  reply: {outcome.reply or '（由 LLM 流式生成）'}")
        print(f"  /voice/command: {outcome.command_json()}")


if __name__ == "__main__":
    main()
