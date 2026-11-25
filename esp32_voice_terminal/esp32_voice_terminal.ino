#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebServer.h>
#include <ESP_I2S.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <math.h>

namespace {
constexpr uint32_t SAMPLE_RATE = 16000;
constexpr size_t FRAME_SAMPLES = 320;
constexpr size_t PCM_BYTES = FRAME_SAMPLES * sizeof(int16_t);
constexpr size_t RAW_WORDS = FRAME_SAMPLES * 2;
constexpr uint16_t MIC_PORT = 5004;
constexpr uint16_t SPEAKER_PORT = 5006;
constexpr size_t PLAY_QUEUE_DEPTH = 12;
constexpr uint8_t USB_AUDIO_MAGIC0 = 0xAA;
constexpr uint8_t USB_AUDIO_MAGIC1 = 0x55;
constexpr uint8_t USB_FRAME_MIC = 0x01;
constexpr uint8_t USB_FRAME_SPK = 0x02;
constexpr uint32_t ECHO_MUTE_HOLD_MS = 320;
constexpr size_t RECORD_SAMPLES = SAMPLE_RATE * 3;
constexpr size_t RECORD_BYTES = RECORD_SAMPLES * sizeof(int16_t);

constexpr int I2S_BCLK = 4;
constexpr int I2S_WS = 5;
constexpr int MIC_SD = 6;
constexpr int AMP_DIN = 7;
constexpr int TFT_MOSI = 11;
constexpr int TFT_SCK = 12;
constexpr int TFT_CS = 10;
constexpr int TFT_DC = 13;
constexpr int TFT_RST = 14;
constexpr int TFT_BL = 15;

const char* AP_SSID = "ATLAS_VOICE";
const char* AP_PASSWORD = "atlas123";
const char INDEX_HTML[] PROGMEM = R"HTML(
<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESP32 语音终端测试</title><style>body{font:16px sans-serif;max-width:520px;margin:32px auto;padding:0 16px;background:#111827;color:#f9fafb}section{background:#1f2937;padding:20px;border-radius:12px;margin:16px 0}button{font-size:18px;padding:12px 20px;border:0;border-radius:8px;background:#22c55e}button:disabled{background:#4b5563}#bar{height:24px;background:#374151;margin-top:12px}#level{height:100%;width:0;background:#38bdf8;transition:width .15s}audio{width:100%;margin-top:12px}</style>
<h1>语音终端离线测试</h1><section><h2>INMP441 麦克风</h2><p>对麦克风说话，观察能量变化。</p><div id="bar"><div id="level"></div></div><p>峰值：<b id="peak">0</b>　RMS：<b id="rms">0</b></p><button id="recordBtn" onclick="startRecording()">录制 3 秒并回放</button> <b id="recmsg"></b><audio id="playback" controls></audio><br><br><button onclick="playOnSpeaker()">让扬声器播放最近录音</button></section>
<section><h2>MAX98357A 功放</h2><p>点击后应播放约 1 秒、440 Hz 测试音。</p><button onclick="fetch('/tone').then(()=>document.getElementById('msg').textContent='正在播放')">播放测试音</button><p id="msg"></p></section>
<section><h2>Atlas 手柄遥控</h2><p>ESP32 需通过 USB 直接连接 Atlas，网页可用来手机或电脑访问。成功后显示 Atlas 返回的状态与 PID。</p><button onclick="atlasControl('start')">启动手柄遥控</button> <button style="background:#ef4444" onclick="atlasControl('stop')">停止遥控</button><p id="atlasmsg"></p></section>
<section><h2>SmolVLA 示教录制</h2><p>Atlas 已开机自启录制守护服务，插上 ESP32 后可直接点“示教开始”（会自动拉起手柄控制）。双路 640×480@10fps 保存为 LeRobot episode。</p><p>任务指令：<input id="taskInput" value="pick up the orange" style="width:100%;box-sizing:border-box;font-size:16px;padding:8px;margin-top:6px;border-radius:6px;border:1px solid #4b5563;background:#111827;color:#f9fafb"></p><button id="demoStartBtn" onclick="demoControl('start')">示教开始</button> <button id="demoStopBtn" style="background:#ef4444" onclick="demoControl('stop')">示教结束</button> <button style="background:#3b82f6" onclick="refreshDemoStatus()">刷新状态</button><p id="demomsg">点击“刷新状态”查看 Atlas 录制状态</p></section>
<p>状态：麦克风 <span id="m">0</span> 包；扬声器 <span id="s">0</span> 包；欠载 <span id="u">0</span></p>
<script>
function refresh(){fetch('/status').then(r=>r.json()).then(s=>{
  document.getElementById('peak').textContent=s.peak;
  document.getElementById('rms').textContent=s.rms;
  document.getElementById('level').style.width=Math.min(100,s.rms/30)+'%';
  document.getElementById('m').textContent=s.mic;
  document.getElementById('s').textContent=s.speaker;
  document.getElementById('u').textContent=s.underrun;
}).catch(()=>{});}
async function startRecording(){
  const button=document.getElementById('recordBtn');
  const message=document.getElementById('recmsg');
  button.disabled=true; message.textContent='正在录音';
  await fetch('/record/start',{method:'POST'});
  const timer=setInterval(async()=>{
    const response=await fetch('/record/status');
    const state=await response.json();
    if(state.done){
      clearInterval(timer); message.textContent='完成，正在回放';
      const player=document.getElementById('playback');
      player.src='/record.wav?'+Date.now(); player.load();
      player.play().catch(()=>{}); button.disabled=false;
    }
  },500);
}
async function playOnSpeaker(){
  const response=await fetch('/recording/play',{method:'POST'});
  document.getElementById('recmsg').textContent=response.ok?'扬声器正在播放最近录音':'请先录制音频';
}
async function atlasControl(action){
  const message=document.getElementById('atlasmsg');
  message.textContent='正在请求 Atlas...';
  try{
    const response=await fetch('/atlas/'+action,{method:'POST'});
    const result=await response.json();
    message.textContent=response.ok&&result.ok
      ?'成功：'+result.status+'，PID：'+result.pid
      :'失败：'+(result.error||'HTTP '+response.status);
  }catch(error){message.textContent='连接 Atlas 失败：'+error.message;}
}
function formatDemoStatus(result){
  const saved=(result.saved_episodes!==undefined)?'，已存条数：'+result.saved_episodes:'';
  const frames=(result.frames!==undefined)?'，帧数：'+result.frames:'';
  const root=result.root?'，目录：'+result.root:'';
  return '状态：'+result.status+frames+saved+root;
}
async function refreshDemoStatus(){
  const message=document.getElementById('demomsg');
  try{
    const response=await fetch('/demo/status');
    const result=await response.json();
    message.textContent=response.ok&&result.ok
      ?formatDemoStatus(result)
      :'查询失败：'+(result.error||'HTTP '+response.status);
  }catch(error){message.textContent='查询示教状态失败：'+error.message;}
}
async function demoControl(action){
  const message=document.getElementById('demomsg');
  const startBtn=document.getElementById('demoStartBtn');
  const stopBtn=document.getElementById('demoStopBtn');
  startBtn.disabled=true; stopBtn.disabled=true;
  message.textContent=action==='start'?'正在启动示教（约 2~5 秒）...':'正在编码保存 AV1 视频，请勿断电（长视频需十几秒）...';
  try{
    const task=encodeURIComponent((document.getElementById('taskInput').value||'').trim());
    const url=action==='start'&&task?'/demo/start?task='+task:'/demo/'+action;
    const response=await fetch(url,{method:'POST'});
    const result=await response.json();
    message.textContent=response.ok&&result.ok
      ?formatDemoStatus(result)
      :'失败：'+(result.error||'HTTP '+response.status);
  }catch(error){message.textContent='示教服务连接失败：'+error.message;}
  finally{startBtn.disabled=false; stopBtn.disabled=false;}
}
setInterval(refresh,500);refresh();
</script>
</html>)HTML";

I2SClass I2S;
Adafruit_ST7735 tft(&SPI, TFT_CS, TFT_DC, TFT_RST);
GFXcanvas16 canvas(160, 128);
WiFiUDP micUdp;
WiFiUDP speakerUdp;
WebServer server(80);
QueueHandle_t playQueue = nullptr;
SemaphoreHandle_t serialWriteMutex = nullptr;
volatile bool usbAudioEnabled = false;
volatile bool atlasCmdPending = false;
String atlasCmdResponse = "";
volatile bool atlasCmdGotResponse = false;
IPAddress targetIp(192, 168, 4, 2);
volatile uint32_t micPackets = 0;
volatile uint32_t speakerPackets = 0;
volatile uint32_t speakerUnderruns = 0;
volatile int micPeak = 0;
volatile int micRms = 0;
volatile uint32_t toneFrames = 0;
uint32_t tonePhase = 0;
uint32_t lastMicActivity = 0;
uint32_t lastSpeakerActivity = 0;
int16_t recording[RECORD_SAMPLES];
volatile size_t recordIndex = 0;
volatile bool recordingActive = false;
volatile bool recordingDone = false;
volatile bool replayActive = false;
size_t replayIndex = 0;

String sanitizeJsonText(const String& raw) {
  String out;
  out.reserve(raw.length());
  for (size_t i = 0; i < raw.length() && out.length() < 120; ++i) {
    char c = raw[i];
    if (c == '"' || c == '\\' || c == '\r' || c == '\n') continue;
    out += c;
  }
  out.trim();
  return out;
}

void sendUsbMicFrame(const int16_t* pcm, size_t bytes) {
  if (!usbAudioEnabled || serialWriteMutex == nullptr) return;
  if (xSemaphoreTake(serialWriteMutex, 0) != pdTRUE) return;
  if (Serial.availableForWrite() > 0) {
    uint8_t header[6] = {
      USB_AUDIO_MAGIC0,
      USB_AUDIO_MAGIC1,
      USB_FRAME_MIC,
      0x00,
      static_cast<uint8_t>(bytes & 0xFF),
      static_cast<uint8_t>((bytes >> 8) & 0xFF),
    };
    Serial.write(header, sizeof(header));
    Serial.write(reinterpret_cast<const uint8_t*>(pcm), bytes);
  }
  xSemaphoreGive(serialWriteMutex);
}

void pushSpeakerPcm(const int16_t* pcm) {
  if (playQueue == nullptr) return;
  if (xQueueSend(playQueue, pcm, 0) != pdTRUE) {
    int16_t discarded[FRAME_SAMPLES];
    xQueueReceive(playQueue, discarded, 0);
    xQueueSend(playQueue, pcm, 0);
  }
  ++speakerPackets;
  lastSpeakerActivity = millis();
}

void pollUsbSerial() {
  static enum { S_IDLE, S_MAGIC1, S_TYPE, S_RES, S_LEN0, S_LEN1, S_PAYLOAD, S_JSON } state = S_IDLE;
  static uint8_t frameType = 0;
  static uint16_t payloadLen = 0;
  static uint16_t payloadPos = 0;
  static uint8_t payloadBuf[PCM_BYTES];
  static String lineBuf;

  while (Serial.available()) {
    int raw = Serial.read();
    if (raw < 0) break;
    uint8_t b = static_cast<uint8_t>(raw);

    if (state == S_IDLE) {
      if (b == USB_AUDIO_MAGIC0) {
        state = S_MAGIC1;
      } else if (b == '{') {
        lineBuf = "{";
        state = S_JSON;
      }
      continue;
    }

    if (state == S_JSON) {
      if (b == '\n') {
        lineBuf.trim();
        if (lineBuf.indexOf("\"cmd\":\"audio_on\"") >= 0) {
          usbAudioEnabled = true;
        } else if (lineBuf.indexOf("\"cmd\":\"audio_off\"") >= 0) {
          usbAudioEnabled = false;
        } else if (atlasCmdPending) {
          atlasCmdResponse = lineBuf;
          atlasCmdGotResponse = true;
        }
        lineBuf = "";
        state = S_IDLE;
      } else if (b != '\r' && lineBuf.length() < 512) {
        lineBuf += static_cast<char>(b);
      }
      continue;
    }

    if (state == S_MAGIC1) {
      state = (b == USB_AUDIO_MAGIC1) ? S_TYPE : S_IDLE;
    } else if (state == S_TYPE) {
      frameType = b;
      state = S_RES;
    } else if (state == S_RES) {
      state = S_LEN0;
    } else if (state == S_LEN0) {
      payloadLen = b;
      state = S_LEN1;
    } else if (state == S_LEN1) {
      payloadLen |= static_cast<uint16_t>(b) << 8;
      payloadPos = 0;
      if (frameType == USB_FRAME_SPK && payloadLen == PCM_BYTES) {
        state = S_PAYLOAD;
      } else {
        state = S_IDLE;
      }
    } else if (state == S_PAYLOAD) {
      payloadBuf[payloadPos++] = b;
      if (payloadPos >= payloadLen) {
        pushSpeakerPcm(reinterpret_cast<const int16_t*>(payloadBuf));
        state = S_IDLE;
      }
    }
  }
}

String callAtlas(const char* command, uint32_t timeoutMs = 20000, const String& task = "") {
  String cleanTask = sanitizeJsonText(task);
  atlasCmdGotResponse = false;
  atlasCmdResponse = "";
  atlasCmdPending = true;
  if (serialWriteMutex != nullptr) xSemaphoreTake(serialWriteMutex, portMAX_DELAY);
  if (cleanTask.length() > 0) {
    Serial.printf("{\"cmd\":\"%s\",\"task\":\"%s\"}\n", command, cleanTask.c_str());
  } else {
    Serial.printf("{\"cmd\":\"%s\"}\n", command);
  }
  Serial.flush();
  if (serialWriteMutex != nullptr) xSemaphoreGive(serialWriteMutex);

  uint32_t started = millis();
  while (millis() - started < timeoutMs) {
    pollUsbSerial();
    if (atlasCmdGotResponse) {
      atlasCmdPending = false;
      return atlasCmdResponse;
    }
    delay(1);
  }
  atlasCmdPending = false;
  return "{\"ok\":false,\"error\":\"Atlas response timeout\"}";
}

void handleAtlasStart() {
  server.send(200, "application/json", callAtlas("start", 20000));
}

void handleAtlasStop() {
  server.send(200, "application/json", callAtlas("stop", 20000));
}

void handleDemoStart() {
  String task = server.hasArg("task") ? server.arg("task") : "";
  server.send(200, "application/json", callAtlas("demo_start", 25000, task));
}

void handleDemoStop() {
  server.send(200, "application/json", callAtlas("demo_stop", 120000));
}

void handleDemoStatus() {
  server.send(200, "application/json", callAtlas("demo_status", 5000));
}

void handleWebStatus() {
  char json[224];
  snprintf(json, sizeof(json),
           "{\"mic\":%lu,\"speaker\":%lu,\"underrun\":%lu,\"peak\":%d,\"rms\":%d}",
           static_cast<unsigned long>(micPackets), static_cast<unsigned long>(speakerPackets),
           static_cast<unsigned long>(speakerUnderruns), micPeak, micRms);
  server.send(200, "application/json", json);
}

void handleTone() {
  tonePhase = 0;
  toneFrames = 50;
  server.send(200, "text/plain", "OK");
}

void handleRecordStart() {
  recordingActive = false;
  recordIndex = 0;
  recordingDone = false;
  recordingActive = true;
  server.send(200, "text/plain", "OK");
}

void handleRecordStatus() {
  char json[96];
  snprintf(json, sizeof(json), "{\"active\":%s,\"done\":%s}",
           recordingActive ? "true" : "false", recordingDone ? "true" : "false");
  server.send(200, "application/json", json);
}

void handleReplayStart() {
  if (!recordingDone || recordingActive) {
    server.send(409, "text/plain", "No completed recording");
    return;
  }
  replayIndex = 0;
  replayActive = true;
  lastSpeakerActivity = millis();
  server.send(200, "text/plain", "OK");
}

void put16(uint8_t* data, size_t offset, uint16_t value) {
  data[offset] = value & 0xFF;
  data[offset + 1] = value >> 8;
}

void put32(uint8_t* data, size_t offset, uint32_t value) {
  put16(data, offset, value & 0xFFFF);
  put16(data, offset + 2, value >> 16);
}

void handleRecordingWav() {
  if (!recordingDone) {
    server.send(409, "text/plain", "Recording not ready");
    return;
  }
  uint8_t header[44] = {};
  memcpy(header, "RIFF", 4);
  put32(header, 4, 36 + RECORD_BYTES);
  memcpy(header + 8, "WAVEfmt ", 8);
  put32(header, 16, 16);
  put16(header, 20, 1);
  put16(header, 22, 1);
  put32(header, 24, SAMPLE_RATE);
  put32(header, 28, SAMPLE_RATE * sizeof(int16_t));
  put16(header, 32, sizeof(int16_t));
  put16(header, 34, 16);
  memcpy(header + 36, "data", 4);
  put32(header, 40, RECORD_BYTES);
  server.setContentLength(sizeof(header) + RECORD_BYTES);
  server.send(200, "audio/wav", "");
  server.sendContent(reinterpret_cast<const char*>(header), sizeof(header));
  for (size_t offset = 0; offset < RECORD_BYTES; offset += 2048) {
    size_t chunk = min(static_cast<size_t>(2048), RECORD_BYTES - offset);
    server.sendContent(reinterpret_cast<const char*>(recording) + offset, chunk);
  }
}

enum class FaceState { Idle, Listening, Speaking };

void drawFace() {
  uint32_t now = millis();
  if (micRms > 250) lastMicActivity = now;
  bool speaking = toneFrames > 0 || (lastSpeakerActivity != 0 && now - lastSpeakerActivity < 300);
  FaceState state = speaking ? FaceState::Speaking
                    : (lastMicActivity != 0 && now - lastMicActivity < 600) ? FaceState::Listening
                                                                           : FaceState::Idle;
  uint16_t color = state == FaceState::Speaking ? ST77XX_YELLOW
                   : state == FaceState::Listening ? ST77XX_GREEN
                                                   : ST77XX_CYAN;
  canvas.fillScreen(ST77XX_BLACK);
  canvas.setTextSize(1);
  canvas.setTextColor(ST77XX_WHITE);
  canvas.setCursor(42, 5);
  canvas.print("ATLAS VOICE");

  bool blink = state == FaceState::Idle && now % 4000 < 160;
  if (blink) {
    canvas.fillRoundRect(28, 42, 30, 4, 2, color);
    canvas.fillRoundRect(102, 42, 30, 4, 2, color);
  } else {
    canvas.fillCircle(45, 43, state == FaceState::Listening ? 15 : 12, color);
    canvas.fillCircle(115, 43, state == FaceState::Listening ? 15 : 12, color);
    canvas.fillCircle(49, 39, 3, ST77XX_WHITE);
    canvas.fillCircle(119, 39, 3, ST77XX_WHITE);
  }
  if (state == FaceState::Listening) {
    canvas.drawCircle(45, 43, 20, color);
    canvas.drawCircle(115, 43, 20, color);
  }

  if (state == FaceState::Speaking) {
    int mouthHeight = (now / 140) % 2 ? 24 : 12;
    canvas.fillRoundRect(60, 82 - mouthHeight / 2, 40, mouthHeight, 6, color);
    canvas.fillRoundRect(65, 82 - mouthHeight / 2 + 4, 30, mouthHeight - 8, 4, ST77XX_BLACK);
  } else if (state == FaceState::Listening) {
    canvas.drawCircle(80, 80, 7, color);
  } else {
    canvas.drawLine(60, 76, 70, 84, color);
    canvas.drawLine(70, 84, 90, 84, color);
    canvas.drawLine(90, 84, 100, 76, color);
  }

  const char* label = state == FaceState::Speaking ? "SPEAKING"
                      : state == FaceState::Listening ? "LISTENING" : "IDLE";
  canvas.setTextSize(2);
  canvas.setTextColor(color);
  canvas.setCursor(state == FaceState::Idle ? 50 : state == FaceState::Listening ? 26 : 32, 106);
  canvas.print(label);
  tft.drawRGBBitmap(0, 0, canvas.getBuffer(), 160, 128);
}

void audioTask(void*) {
  int32_t rawRx[RAW_WORDS];
  int32_t rawTx[RAW_WORDS];
  int16_t micPcm[FRAME_SAMPLES];
  int16_t playPcm[FRAME_SAMPLES];
  float x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  while (true) {
    size_t count = I2S.readBytes(reinterpret_cast<char*>(rawRx), sizeof(rawRx));
    size_t frames = count / (2 * sizeof(int32_t));
    int peak = 0;
    float sumSquares = 0;
    for (size_t i = 0; i < frames; ++i) {
      float input = static_cast<float>(rawRx[2 * i] >> 16);
      float output = 0.97803048f * input - 1.95606096f * x1 + 0.97803048f * x2
                     + 1.95557824f * y1 - 0.95654368f * y2;
      x2 = x1;
      x1 = input;
      y2 = y1;
      y1 = output;
      sumSquares += output * output;
      int32_t gained = static_cast<int32_t>(lrintf(output * 2.0f));
      gained = min(32767L, max(-32768L, gained));
      int16_t sample = static_cast<int16_t>(gained);
      micPcm[i] = sample;
      int level = abs(static_cast<int>(sample));
      if (level > peak) peak = level;
    }
    micPeak = peak;
    micRms = frames > 0 ? static_cast<int>(sqrtf(sumSquares / frames)) : 0;
    if (frames == FRAME_SAMPLES) {
      uint32_t nowMs = millis();
      bool speakerBusy = toneFrames > 0 || replayActive ||
                         (lastSpeakerActivity != 0 && nowMs - lastSpeakerActivity < ECHO_MUTE_HOLD_MS);
      if (usbAudioEnabled) {
        if (speakerBusy) {
          static const int16_t silence[FRAME_SAMPLES] = {};
          sendUsbMicFrame(silence, PCM_BYTES);
        } else {
          sendUsbMicFrame(micPcm, PCM_BYTES);
        }
      }
      micUdp.beginPacket(targetIp, MIC_PORT);
      micUdp.write(reinterpret_cast<const uint8_t*>(micPcm), PCM_BYTES);
      if (micUdp.endPacket() == 1) ++micPackets;
      if (recordingActive) {
        size_t remaining = RECORD_SAMPLES - recordIndex;
        size_t samplesToCopy = min(FRAME_SAMPLES, remaining);
        memcpy(recording + recordIndex, micPcm, samplesToCopy * sizeof(int16_t));
        recordIndex += samplesToCopy;
        if (recordIndex >= RECORD_SAMPLES) {
          recordingActive = false;
          recordingDone = true;
        }
      }
    }

    if (toneFrames > 0) {
      for (size_t i = 0; i < FRAME_SAMPLES; ++i) {
        playPcm[i] = static_cast<int16_t>(6000.0 * sin(2.0 * PI * 440.0 * (tonePhase + i) / SAMPLE_RATE));
      }
      tonePhase += FRAME_SAMPLES;
      --toneFrames;
    } else if (replayActive) {
      memcpy(playPcm, recording + replayIndex, PCM_BYTES);
      replayIndex += FRAME_SAMPLES;
      lastSpeakerActivity = millis();
      if (replayIndex >= RECORD_SAMPLES) replayActive = false;
    } else if (xQueueReceive(playQueue, playPcm, 0) != pdTRUE) {
      memset(playPcm, 0, PCM_BYTES);
      ++speakerUnderruns;
    }
    for (size_t i = 0; i < FRAME_SAMPLES; ++i) {
      int32_t sample = static_cast<int32_t>(playPcm[i]) << 16;
      rawTx[2 * i] = sample;
      rawTx[2 * i + 1] = sample;
    }
    I2S.write(rawTx, sizeof(rawTx));
  }
}

void receiveSpeakerAudio() {
  int packetSize = speakerUdp.parsePacket();
  if (packetSize != static_cast<int>(PCM_BYTES)) {
    if (packetSize > 0) {
      while (speakerUdp.available()) speakerUdp.read();
    }
    return;
  }
  int16_t pcm[FRAME_SAMPLES];
  if (speakerUdp.read(reinterpret_cast<uint8_t*>(pcm), PCM_BYTES) != PCM_BYTES) return;
  if (xQueueSend(playQueue, pcm, 0) != pdTRUE) {
    int16_t discarded[FRAME_SAMPLES];
    xQueueReceive(playQueue, discarded, 0);
    xQueueSend(playQueue, pcm, 0);
  }
  ++speakerPackets;
  lastSpeakerActivity = millis();
}

void handleSerial() {
  if (!Serial.available()) return;
  String command = Serial.readStringUntil('\n');
  command.trim();
  if (command.startsWith("target ")) {
    IPAddress parsed;
    if (parsed.fromString(command.substring(7))) {
      targetIp = parsed;
      Serial.printf("Mic target: %s\n", targetIp.toString().c_str());
    }
  } else if (command == "status") {
    Serial.printf("AP %s, IP %s, target %s, mic %lu, speaker %lu, underrun %lu\n",
                  AP_SSID, WiFi.softAPIP().toString().c_str(), targetIp.toString().c_str(),
                  static_cast<unsigned long>(micPackets), static_cast<unsigned long>(speakerPackets),
                  static_cast<unsigned long>(speakerUnderruns));
  } else {
    Serial.println("Commands: target <IPv4>, status");
  }
}
}  // namespace

void setup() {
  Serial.setTxBufferSize(4096);
  Serial.setRxBufferSize(4096);
  Serial.begin(115200);
  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, HIGH);
  SPI.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(1);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  micUdp.begin(MIC_PORT);
  speakerUdp.begin(SPEAKER_PORT);
  server.on("/", []() { server.send(200, "text/html", INDEX_HTML); });
  server.on("/status", handleWebStatus);
  server.on("/tone", handleTone);
  server.on("/record/start", HTTP_POST, handleRecordStart);
  server.on("/record/status", handleRecordStatus);
  server.on("/record.wav", handleRecordingWav);
  server.on("/recording/play", HTTP_POST, handleReplayStart);
  server.on("/atlas/start", HTTP_POST, handleAtlasStart);
  server.on("/atlas/stop", HTTP_POST, handleAtlasStop);
  server.on("/demo/start", HTTP_POST, handleDemoStart);
  server.on("/demo/stop", HTTP_POST, handleDemoStop);
  server.on("/demo/status", HTTP_GET, handleDemoStatus);
  server.begin();
  playQueue = xQueueCreate(PLAY_QUEUE_DEPTH, PCM_BYTES);
  serialWriteMutex = xSemaphoreCreateMutex();

  I2S.setPins(I2S_BCLK, I2S_WS, AMP_DIN, MIC_SD);
  bool ok = I2S.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO);
  if (!ok || playQueue == nullptr) {
    Serial.println("Audio initialization failed");
    while (true) delay(1000);
  }
  xTaskCreatePinnedToCore(audioTask, "audio", 16384, nullptr, 3, nullptr, 1);
  drawFace();
  Serial.printf("AP %s, IP %s, mic UDP %d, speaker UDP %d\n",
                AP_SSID, WiFi.softAPIP().toString().c_str(), MIC_PORT, SPEAKER_PORT);
}

void loop() {
  pollUsbSerial();
  server.handleClient();
  receiveSpeakerAudio();
  static uint32_t lastDisplay = 0;
  if (millis() - lastDisplay >= 100) {
    lastDisplay = millis();
    drawFace();
  }
  delay(1);
}
