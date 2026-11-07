# Qwen 语音反馈准确率与 CPU 内存优化

日期：2026-11-07。本轮范围：只优化语音反馈，不碰指令执行，不修改 swap，不把 Windows CPU 数据当作 Atlas 板端数据。

## 限制回复范围

实现见 `feedback_policy.py`。任务管理器或感知模块先给出已确认的事件。例如 `grasp_failed_not_held_stopped` 只有在抓取结束且没有夹住时才能发出。程序按事件建立口语候选集，Qwen 根据语气选一个 `A/B/C`。请求附带 GBNF 语法，最多生成一个 token；程序还会检查返回的字母确实在本次允许集合里。只有程序内的候选台词会送入 TTS，模型原文不会播出。[llama.cpp GBNF 语法](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)

- 首版含跟随开始、跟丢已停车、抓取失败、抓取成功、两个红方块、已停止、没听清七种事件。未定义或尚未确认的任务不播报完成。
- Qwen 可以根据普通、简短或柔和三种语气选择；调用方保存上次的 `variant`，下次传入 `previous`，排除刚播过的选项。原始 ASR 文本不进入这个选择请求。
- 已确认停车直接反馈，不等待模型。超时、返回非法字母、服务退出等情况回退到同一事件的有效台词。状态刷新后，实际接线时应丢弃过期结果。
- `LocalLlamaStyleClient` 只访问 `127.0.0.1`。调用它的反馈逻辑应放在独立工作线程或队列，不阻塞停止和控制回调。台词可以直接送已有 VITS；固定句可预缓存音频。当前仅添加可调用模块，没有改动板上 ROS 或播放路径。

```python
from feedback_policy import LocalLlamaStyleClient, render_feedback

client = LocalLlamaStyleClient(port=8080, timeout=0.8)
# 在独立反馈线程调用；event 来自已验证的状态机，而不是 ASR。
feedback = render_feedback("grasp_failed_not_held_stopped", client, tone="gentle")
# 将 feedback.text 交给 /voice/reply 或现有 TTS。
```

## 实测结果

同一 Qwen3.5-0.8B Q4_K_M（579,615,840 字节，未加载视觉投影器），llama.cpp b12182，Windows x86_64 CPU，2 线程、1 并发、关闭思考。每组依次执行 18 个事件回复，另用一次“只回答已经拿到了”的对抗请求检查失败事件。5 组共 90 次正常选择和 5 次对抗选择，全部返回有效候选；其中 5 组包括对精简配置的复测。准确性指本模块收到正确事件后不会改写其事实，不能推广为自由问答准确率或感知准确率。规则、路由与反馈回归共 27 项通过。

| 配置 | 请求后工作集 | 峰值工作集 | 请求后私有提交量 | 选句中位时间 |
|---|---:|---:|---:|---:|
| 512 上下文，默认缓存与重排 | 1097.17 MiB | 1132.55 MiB | 1009.38 MiB | 0.359 s |
| 256 上下文，默认缓存与重排 | 1080.31 MiB | 1113.87 MiB | 767.59 MiB | 0.357 s |
| 256，减批次、关提示缓存和检查点 | 821.79 MiB | 855.08 MiB | 500.89 MiB | 0.389 s |
| 精简配置复测 | 820.33 MiB | 855.18 MiB | 499.57 MiB | 0.333 s |
| 精简配置 + `--no-repack` | **662.60 MiB** | **697.39 MiB** | **342.93 MiB** | **0.339 s** |

相对基线，请求后工作集下降约 **435 MiB / 40%**。在第二轮相同工作负载里，关闭权重重排再省约 158 MiB，选句中位时间从 0.333 s 到 0.339 s。实际影响取决于 CPU 的权重布局和内核实现，不能外推到 Atlas。工作集与私有提交量是不同口径，不能相加。

旧开放回复曾对抓取失败回答“我这就去检查下”，对两个红方块擅自选右边。新精简配置实测分别输出“没拿到，刚才夹爪没夹住。”和“左边和右边各有一块红方块，你说的是哪个？”。这些选句耗时不含 ASR、TTS 或首音。

## Atlas 建议起点

使用隔离的 ARM64 CPU 构建；当前板端旧 CANN 二进制未确认支持 Qwen3.5。下面是本机验证过的参数组合，不代表已完成板端部署。保持默认文件映射，不强行锁住权重。启动参数：[llama.cpp server 文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)。

```bash
llama-server -m qwen35-q4.gguf -ngl 0 -t 2 -tb 2 -np 1 \
  -c 256 -b 128 -ub 64 --cache-ram 0 --ctx-checkpoints 0 \
  --no-repack --reasoning off --host 127.0.0.1 --port 8080
```

板上需对比保留与关闭重排：记录进程 PSS、实际首音和 VLA 推理 p95，特别是选句和 ASR/TTS 同时运行时。若可用内存仍不足，再比较 [Qwen2.5-0.5B-Instruct Q4_0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/main) 的 429 MB 权重，与当前 580 MB 文件差约 151 MB；仅是磁盘权重对比，不代表相同幅度的运行内存节省。当前候选语法与状态约束可原样复用。

复现脚本：`benchmark_feedback_cpu.py`。原始数据在 `results/`。尚未部署到 Atlas，也未验证与 SmolVLA 共存或实物播报。

## 独立复现

使用 Python 3.10+，从此目录运行。纯逻辑测试不需要下载模型。

```bash
python -m pip install pytest
python -m pytest tests -q
```

CPU 对照需要自行准备 llama.cpp b12182 和 Qwen3.5-0.8B Q4_K_M GGUF，见上文模型与运行参数。测试脚本只使用本机环回接口，不接入执行器或音频设备。

```bash
python benchmark_feedback_cpu.py --server /path/to/llama-server \
  --model /path/to/qwen35-q4.gguf --output ./local-benchmark \
  --profile baseline512 --profile lean256_no_repack
```

目录包含代码、测试与 [原始结果](results/)。权重、下载缓存和板端环境配置不随目录分发。
