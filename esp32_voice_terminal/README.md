# ESP32-S3 无线语音终端（第一版）

基于 ESP32-S3 N8R16、INMP441、MAX98357A 和 ST7735S 的开发版本。当前固件提供 16 kHz、单声道、16-bit PCM 的双向 UDP 音频和屏幕状态显示，供 Wi-Fi 语音链路逐项验证。

## 当前边界

已实现源代码：Wi-Fi SoftAP、INMP441 采集、MAX98357A 播放队列、ST7735S 状态页、串口目标地址设置，以及电脑端 WAV/UDP 工具。

验证状态：`esp32:esp32 4.0.0` 下编译通过；电脑端 UDP/WAV 录音与播放回环测试通过。实际外设、烧录和整机语音链路尚待硬件测试。

尚未完成：Opus、回声消除、Atlas ASR/TTS 对接、USB ECM/RNDIS 网桥、电机状态或视频转发。不要把当前版本描述为已完成整机语音或无线网卡。默认热点密码仅用于实验室，正式部署应修改且不要开放到互联网。

## 接线

断电接线，所有模块共地。引脚可按实际板卡修改固件顶部常量。

### INMP441

| 模块 | ESP32-S3 |
|---|---|
| VDD | 3V3（不要接 5V） |
| GND | GND |
| SCK | GPIO4 |
| WS | GPIO5 |
| SD | GPIO6 |
| L/R | GND（左声道） |

### MAX98357A

| 模块 | ESP32-S3 / 扬声器 |
|---|---|
| VIN | 5V |
| GND | GND |
| BCLK | GPIO4 |
| LRC | GPIO5 |
| DIN | GPIO7 |
| SD、GAIN | 常见模块可暂不连接 |
| OUT+、OUT- | 4–8Ω 扬声器；OUT- 不接 GND |

INMP441 与 MAX98357A 共用 BCLK/WS，输入与输出数据线分开。

### ST7735S（四线 SPI）

| 模块丝印 | ESP32-S3 |
|---|---|
| VCC | 3V3 |
| GND | GND |
| SCL/SCK | GPIO12 |
| SDA/MOSI | GPIO11 |
| CS | GPIO10 |
| DC/A0 | GPIO13 |
| RES/RST | GPIO14 |
| BLK/LED | GPIO15 |

GPIO19/20 保留给原生 USB。不同屏幕模块的电压和色板配置可能不同，若花屏或偏移，应按模块资料调整供电、初始化和旋转参数。

## 构建与烧录

安装 [Arduino CLI](https://arduino.github.io/arduino-cli/) 后：

```bash
arduino-cli core install esp32:esp32@4.0.0 \
  --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli lib install "Adafruit GFX Library" "Adafruit ST7735 and ST7789 Library"
```

使用 ESP32-S3 N8R16 的 8MB Flash 与 OPI PSRAM 配置：

```bash
arduino-cli compile --clean \
  --fqbn esp32:esp32:esp32s3:FlashSize=8M,PSRAM=opi,USBMode=hwcdc,CDCOnBoot=cdc \
  esp32_voice_terminal
arduino-cli upload -p COM12 \
  --fqbn esp32:esp32:esp32s3:FlashSize=8M,PSRAM=opi,USBMode=hwcdc,CDCOnBoot=cdc \
  esp32_voice_terminal
```

Linux 下将 `COM12` 换成实际串口。上传期间不要同时打开串口监视器。

## Wi-Fi 音频测试

1. 电脑连接热点 `ATLAS_VOICE`，开发密码 `atlas123`。
2. ESP32 地址为 `192.168.4.1`，默认音频目标为 `192.168.4.2`。
3. 若电脑获得其他地址，在串口执行 `target <电脑IPv4>`；`status` 查看计数。
4. 电脑录音：

```bash
python udp_audio_tool.py record --seconds 5 --output mic_capture.wav
```

5. 向 ESP32 播放 WAV：

```bash
python udp_audio_tool.py play mic_capture.wav --target 192.168.4.1
```

工具只接受 16 kHz、单声道、16-bit PCM WAV。麦克风 UDP 端口为 5004，扬声器为 5006，每包 320 个采样（20 ms）。防火墙需要允许相应 UDP 流量。

## 后续阶段

USB ECM/RNDIS 将基于乐鑫 USB Device/ESP-IoT-Bridge 单独集成，再与音频任务联调。视频流不应通过音频队列处理；其带宽与延迟需要单独验收。

## 器件与接口资料

- [INMP441 数据手册](https://www.mouser.com/datasheet/2/400/INMP441-1112508.pdf)
- [MAX98357A 数据手册](https://www.analog.com/media/en/technical-documentation/data-sheets/max98357a-max98357b.pdf)
- [Arduino ESP32 I2S API](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/i2s.html)
- [ESP32-S3 USB Device](https://docs.espressif.com/projects/esp-usb/en/latest/esp32s3/usb_device.html)
- [ESP-IoT-Bridge](https://documentation.espressif.com/esp-iot-bridge/master/components/iot_bridge/User_Guide.md)
