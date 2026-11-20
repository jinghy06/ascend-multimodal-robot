#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESP_I2S.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>

namespace {
constexpr uint32_t SAMPLE_RATE = 16000;
constexpr size_t FRAME_SAMPLES = 320;
constexpr size_t PCM_BYTES = FRAME_SAMPLES * sizeof(int16_t);
constexpr size_t RAW_WORDS = FRAME_SAMPLES * 2;
constexpr uint16_t MIC_PORT = 5004;
constexpr uint16_t SPEAKER_PORT = 5006;
constexpr size_t PLAY_QUEUE_DEPTH = 6;

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

I2SClass I2S;
Adafruit_ST7735 tft(&SPI, TFT_CS, TFT_DC, TFT_RST);
WiFiUDP micUdp;
WiFiUDP speakerUdp;
QueueHandle_t playQueue = nullptr;
IPAddress targetIp(192, 168, 4, 2);
volatile uint32_t micPackets = 0;
volatile uint32_t speakerPackets = 0;
volatile uint32_t speakerUnderruns = 0;
volatile int micPeak = 0;

void showStatus() {
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_CYAN);
  tft.setTextSize(2);
  tft.setCursor(4, 4);
  tft.print("ATLAS VOICE");
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(4, 28);
  tft.print("AP: ");
  tft.print(AP_SSID);
  tft.setCursor(4, 40);
  tft.print("IP: ");
  tft.print(WiFi.softAPIP());
  tft.setCursor(4, 52);
  tft.print("To: ");
  tft.print(targetIp);
  tft.setCursor(4, 70);
  tft.printf("Mic pkts: %lu", static_cast<unsigned long>(micPackets));
  tft.setCursor(4, 82);
  tft.printf("Spk pkts: %lu", static_cast<unsigned long>(speakerPackets));
  tft.setCursor(4, 94);
  tft.printf("Underrun: %lu", static_cast<unsigned long>(speakerUnderruns));
  tft.setCursor(4, 106);
  tft.printf("Mic peak: %d", micPeak);
}

void audioTask(void*) {
  int32_t rawRx[RAW_WORDS];
  int32_t rawTx[RAW_WORDS];
  int16_t micPcm[FRAME_SAMPLES];
  int16_t playPcm[FRAME_SAMPLES];

  while (true) {
    size_t count = I2S.readBytes(reinterpret_cast<char*>(rawRx), sizeof(rawRx));
    size_t frames = count / (2 * sizeof(int32_t));
    int peak = 0;
    for (size_t i = 0; i < frames; ++i) {
      int16_t sample = static_cast<int16_t>(rawRx[2 * i] >> 16);
      micPcm[i] = sample;
      int level = abs(static_cast<int>(sample));
      if (level > peak) peak = level;
    }
    micPeak = peak;
    if (frames == FRAME_SAMPLES) {
      micUdp.beginPacket(targetIp, MIC_PORT);
      micUdp.write(reinterpret_cast<const uint8_t*>(micPcm), PCM_BYTES);
      if (micUdp.endPacket() == 1) ++micPackets;
    }

    if (xQueueReceive(playQueue, playPcm, 0) != pdTRUE) {
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
  playQueue = xQueueCreate(PLAY_QUEUE_DEPTH, PCM_BYTES);

  I2S.setPins(I2S_BCLK, I2S_WS, AMP_DIN, MIC_SD);
  bool ok = I2S.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO);
  if (!ok || playQueue == nullptr) {
    Serial.println("Audio initialization failed");
    while (true) delay(1000);
  }
  xTaskCreatePinnedToCore(audioTask, "audio", 8192, nullptr, 3, nullptr, 1);
  showStatus();
  Serial.printf("AP %s, IP %s, mic UDP %d, speaker UDP %d\n",
                AP_SSID, WiFi.softAPIP().toString().c_str(), MIC_PORT, SPEAKER_PORT);
}

void loop() {
  receiveSpeakerAudio();
  handleSerial();
  static uint32_t lastDisplay = 0;
  if (millis() - lastDisplay >= 500) {
    lastDisplay = millis();
    showStatus();
  }
  delay(1);
}
