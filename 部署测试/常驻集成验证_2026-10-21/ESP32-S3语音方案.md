# ESP32-S3 语音分工核查

日期：2026-10-21。用户回忆容量为“8、16”，暂按 N16R8（16 MB Flash、8 MB PSRAM）规划，尚未核对模组标识；板未连接电脑，未烧录、未做实物性能测试。

## 结论

**动态中文 TTS 可以下移到 S3；唤醒、音频前端、固定控制口令也适合下移。当前不能把低延迟、开放式中文 ASR 当成 S3 已解决的功能。**

| 功能 | 支持依据 | 本项目处理 |
|---|---|---|
| 回声处理、降噪、VAD | ESP-SR AFE | S3 常驻，接入实际播放参考信号 |
| 唤醒词 | WakeNet | S3 常驻 |
| “停止、跟随、抓取”等固定口令 | MultiNet 中文命令识别 | S3 常驻，可直接发送结构化命令 |
| 任意中文问题转文字 | 官方 MultiNet 不是开放式中文听写 | 传音频给 Atlas 或外置 ASR |
| 任意动态中文文本播报 | ESP-TTS 支持 UTF-8 中文文本、流式输出 | S3 接收文本后合成，需试听音质 |

来源：[ESP-SR 总览](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/getting_started/readme.html)、[命令识别](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/speech_command_recognition/README.html)、[中文 TTS](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/speech_synthesis/readme.html)。TTS 是轻量拼接合成，支持动态文本，并非只能播放预录提示；中文以外的混合文本需额外验证。

## 容量依据与限制

官方组件基准（不是本项目整机实测）：

| 组件 | 内部 RAM | PSRAM / Flash |
|---|---:|---:|
| 单麦 + 播放参考、SR 低成本 AFE | 60.1 KB | PSRAM 739.7 KB |
| MultiNet 5 Q8 | 16 KB | PSRAM 2310 KB |
| MultiNet 6 | 32 KB | PSRAM 4100 KB |
| 中文 TTS | 运行 RAM 20 KB | Flash 2.2 MB |

来源：[ESP-SR 资源基准](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/benchmark/README.html)。上述 AFE 已包含指定 WakeNet，不重复加一次唤醒模型。Wi-Fi/USB、FreeRTOS 栈、DMA、音频缓冲、固件与模型分区均另计；8 MB PSRAM 只是有希望容纳这些组件，不能直接据此判定堆与实时性通过。Flash 还要安排命令模型、应用与 OTA，16 MB 也应检查实际分区表。

有开放词汇 ASR 的新案例：[Oído 论文](https://arxiv.org/abs/2610.13447)，在 ESP32-S3 N16R8 上运行约 13M 参数模型。但论文的英语实板处理时间约为音频时长的 1.97 倍，4 秒语音约在结束后 8.7 秒给出转写；流式版仍有数秒尾延迟。其英语/西班牙语结果不能证明中文可用，也不满足本项目的低延迟目标。可作后续研究，暂不放入主路径。

## 与 Atlas 的接口

- 上行音频：16 kHz、16-bit、单声道 PCM，连续数据约 32,000 B/s；20 ms 一帧为 640 B。2 秒预录环形缓冲约 64,000 B。
- 普通 115200 baud UART 在 8N1 下理论有效载荷约 11,520 B/s，不能传上述未压缩连续音频。优先测试 USB 或 Wi-Fi；如用 921600 baud UART，仍需测有效吞吐、校验、丢帧和背压。
- 消息：`wake`、`vad_start`、`audio_chunk(seq,timestamp,pcm)`、`vad_end`、`command(id)`、`tts_text(id,text)`、`tts_cancel(id)`、`tts_started/finished`、`health(free_heap,largest_block,free_psram)`。
- 控制与音频使用独立有界队列；“停止”不能排在音频/长文本后。断线时下位机自主停止，不能等待语言模型。
- TTS 按句分段，设置上限并支持打断。对话模型生成中文文本，S3 合成音频；S3 不负责大模型问答。

## 实物到位后验收

1. 核对型号与 PSRAM/Flash，再备份原固件、确认 I2S 麦克风/功放引脚。
2. 先运行官方独立 TTS、AFE、MultiNet 示例，再组合；记录固件与模型版本。
3. 同时采集音频、识别口令、联网、播报，持续 30 分钟，记录最小空闲堆、最大连续块、PSRAM 与丢帧。
4. 测录音/播放同时工作的回声、抢话停止、首段音频延迟及长文本分段。
5. 只有开放式中文 ASR 也有实际后端后，才算“自由语音对话”整条链路完成。
