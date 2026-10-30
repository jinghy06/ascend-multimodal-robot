# voice_control —— M2 语音控制包

唤醒（KWS）→ 流式 ASR → 规则/LLM 混合指令解析 → 中文回复，全部离线运行。
目标平台为香橙派 AIpro 8T（昇腾 310B），开发与单测可在 Windows/Linux PC 进行。

## 架构

```
麦克风 16k
  → KWS 常驻唤醒（sherpa-onnx KeywordSpotter + keywords.txt）
  → VAD（silero-vad）+ 流式 ASR（streaming-zipformer-zh-en，partial 实时上屏）
  → final 文本 → parser_node
        ├─ 规则层命中（20 条预定义，<1ms，零 LLM 依赖）
        └─ 未命中 → llm_service（独立进程/独立 venv）
              ① parse：短 JSON → 白名单校验 → /voice/command（先路由先动作）
              ② stream_reply：流式中文回复 → /voice/reply（供 TTS 边生成边播报）
```

ROS 2 topic（均为 `std_msgs/String` + JSON 载荷）：

| topic | 方向 | 载荷 |
|---|---|---|
| `/voice/utterance` | asr_node → parser_node | `{"text": "抓起红色杯子"}` |
| `/voice/partial` | asr_node → 任意 | `{"text": "..."}` 流式中间结果 |
| `/voice/command` | parser_node → task_manager | `{"task_id","params","source","ok"}` |
| `/voice/llm_request` / `/voice/llm_result` | parser_node ↔ llm_service | 见 nodes/llm_service.py |
| `/voice/reply` | llm_service / parser_node → TTS | `{"request_id","chunk","done"}` |

安全约定：`ok=false`（unknown）的指令**永不路由**到 task_manager，只回兜底话术。

## 快速开始（PC，无 ROS）

```bash
cd voice_control
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/download_models.py                      # 语音模型（KWS/流式ASR/VAD）
python tools/download_models.py --qwen qwen2.5       # LLM（可选，需 huggingface_hub）

# 生成唤醒词文件（该 KWS 模型 tokens 为拼音声韵母，不能直接写整句中文；
# 换唤醒词时用本工具重新生成，符号会对照模型自动校验）
pip install pypinyin
python tools/make_keywords.py --model-dir \
    models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 \
    --phrases 你好小车 小车小车 --out configs/keywords.txt

# 1) 规则解析联调（无需任何模型）
python -m voice_control.demos.demo_parse_text

# 2) 语音全链路（离线 wav）
python -m voice_control.demos.demo_asr_file --wav your.wav \
    --kws-dir models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile \
    --asr-dir models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

# 3) 带 LLM 的解析（PC 显卡即可跑）
python -m voice_control.demos.demo_parse_text --llm models/Qwen2.5-0.5B-Instruct

# 单测（无模型/无 ROS 依赖）
pytest tests/ -v
```

> Windows 控制台跑 demo 若中文变乱码：`set PYTHONUTF8=1`（或 `chcp 65001`）后再运行，
> Linux/香橙派无此问题。

## 录制测试集与评测（验收数据）

```bash
# 1) 录唤醒词（每条 3s，按回车逐条录，自动编号）
python tools/record_wavs.py --out dataset/wake --duration 3
#    建议：正常/偏轻/偏响各几条，0.5m 与 1m 各几条；
#    再开着电视录 3-5 条放 dataset/neg 当负样本。

# 2) 录 20 条指令（对照 configs/commands.yaml 逐条念）
python tools/record_wavs.py --out dataset/commands --duration 4 --label grasp_红杯子

# 3) 唤醒评测：输出唤醒率 / 误唤醒数，报告写 _wake_report.json
python tools/eval_wake.py --pos dataset/wake --neg dataset/neg
```

录制小提示：录音峰值应在几千以上（工具会提示"声音偏小"）；验收指标参考
README 上文——预定义指令识别率 ≥90%、路由正确率 100%（规则层保证）、
唤醒率 ≥90%、误唤醒 ≤2 次/小时（负样本折算）。

## LLM 环境注意

- **Qwen2.5-0.5B-Instruct**：transformers ≥4.37（默认保底方案）；
- **Qwen3-0.6B**：transformers ≥4.51，且代码已显式 `enable_thinking=False`
  （思考模式会显著增加延迟并污染 JSON；若用其他框架加载务必关闭）；
- 模型路径在 `configs/llm.yaml` 改，切换模型零代码；
- 香橙派 310B：transformers + torch_npu（版本与 CANN 对齐，见项目 W1 环境记录），
  bf16 ≈1.2GB 显存，eager 推理即可。

## 香橙派部署（ROS 2 Humble）

```bash
sudo apt install libportaudio2 portaudio19-dev    # sounddevice 依赖
pip install -r requirements.txt
python tools/download_models.py --qwen qwen2.5

# 把本包放入工作空间 src/ 后：
colcon build --packages-select voice_control
ros2 run voice_control asr_node
ros2 run voice_control llm_service        # 需要 transformers/torch_npu 的环境
ros2 run voice_control parser_node
```

`llm_service` 与另外两个节点可用**不同 Python 环境**运行（ROS topic 跨环境通信），
用于隔离 Qwen3 所需的新版 transformers 与 lerobot 钉死版本。

## 目录

```
voice_control/
├── configs/                 # commands.yaml(指令目录) keywords.txt(唤醒词) llm.yaml
├── voice_control/           # 核心包（可无 ROS 导入）
│   ├── command_rules.py     #   规则解析（模板+槽位，100% 确定性）
│   ├── router.py            #   规则优先 / LLM 兜底 / 兜底话术
│   ├── llm_parser.py        #   两次调用（parse + stream_reply），白名单校验
│   ├── kws_engine.py        #   唤醒引擎
│   ├── asr_engine.py        #   VAD + 流式 ASR + 状态机
│   ├── audio_source.py      #   麦克风 / 离线 wav（跨平台）
│   ├── prompts/system_zh.txt
│   ├── demos/               # 无 ROS demo
│   └── nodes/               # ROS 2 节点（rclpy 惰性导入）
├── tests/                   # pytest（全平台可跑）
├── tools/download_models.py # 跨平台模型下载（hf-mirror 默认可用）
└── requirements.txt
```

## 跨平台说明

- 引擎层只消费 16kHz float32 numpy 块，麦克风/wav 源可互换；
- 所有路径经 `pathlib`，无 Windows 专用逻辑；Python ≥3.9（用到 `removesuffix`）；
- Linux 麦克风需 `libportaudio2`；香橙派为 aarch64，sherpa-onnx 官方提供 linux aarch64 wheel。
