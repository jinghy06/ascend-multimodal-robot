#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "driver/i2s_std.h"
#include "driver/usb_serial_jtag.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "esp_tts.h"
#include "esp_tts_voice_template.h"
#include "esp_tts_voice_xiaole.h"
#include "spi_flash_mmap.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "st7735_face.h"
#include "web_ui.h"

#define SAMPLE_RATE      16000
#define FRAME_SAMPLES    320
#define PCM_BYTES        (FRAME_SAMPLES * sizeof(int16_t))
#define PLAY_QUEUE_DEPTH 24

#define I2S_BCLK_IO      4
#define I2S_WS_IO        5
#define I2S_MIC_SD_IO    6
#define I2S_AMP_DIN_IO   7

#define USB_MAGIC0          0xAA
#define USB_MAGIC1          0x55
#define USB_FRAME_MIC       0x01
#define USB_FRAME_SPK       0x02
#define USB_FRAME_SUBTITLE  0x03
#define ECHO_MUTE_MS        320

#define AP_SSID          "ATLAS_VOICE"
#define AP_PASS          "atlas123"

static i2s_chan_handle_t s_tx_chan = NULL;
static i2s_chan_handle_t s_rx_chan = NULL;
static QueueHandle_t s_play_queue = NULL;
static QueueHandle_t s_tts_queue = NULL;
static SemaphoreHandle_t s_usb_tx_mutex = NULL;
static SemaphoreHandle_t s_cmd_sem = NULL;

static volatile bool s_usb_audio_enabled = false;
static volatile bool s_tts_cancel = false;
static volatile bool s_tts_busy = false;
static volatile uint32_t s_tone_frames = 0;
static volatile int s_mic_peak = 0;
static volatile int s_mic_rms = 0;
static volatile uint32_t s_last_mic_ms = 0;
static volatile uint32_t s_last_spk_ms = 0;

static volatile bool s_atlas_cmd_pending = false;
static char s_atlas_cmd_resp[512];

static inline uint32_t now_ms(void) {
    return (uint32_t)(esp_timer_get_time() / 1000ULL);
}

static void push_speaker_frame(const int16_t *pcm) {
    if (!s_play_queue) return;
    if (xQueueSend(s_play_queue, pcm, pdMS_TO_TICKS(50)) != pdTRUE) {
        int16_t dummy[FRAME_SAMPLES];
        xQueueReceive(s_play_queue, dummy, 0);
        xQueueSend(s_play_queue, pcm, 0);
    }
    s_last_spk_ms = now_ms();
}

static void enqueue_tts_text(const char *text) {
    if (!s_tts_queue || !text || !text[0]) return;
    char buf[256];
    strncpy(buf, text, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    xQueueSend(s_tts_queue, buf, 0);
}

static void cancel_tts(void) {
    s_tts_cancel = true;
    if (s_tts_queue) xQueueReset(s_tts_queue);
    if (s_play_queue) xQueueReset(s_play_queue);
}

static void usb_write_locked(const void *data, size_t len, TickType_t wait_ticks) {
    if (!s_usb_tx_mutex) return;
    if (xSemaphoreTake(s_usb_tx_mutex, wait_ticks) != pdTRUE) return;
    usb_serial_jtag_write_bytes((const char *)data, len, wait_ticks);
    xSemaphoreGive(s_usb_tx_mutex);
}

static void send_usb_mic_frame(const int16_t *pcm) {
    if (!s_usb_audio_enabled || !s_usb_tx_mutex) return;
    if (xSemaphoreTake(s_usb_tx_mutex, 0) != pdTRUE) return;
    uint8_t packet[6 + PCM_BYTES];
    packet[0] = USB_MAGIC0;
    packet[1] = USB_MAGIC1;
    packet[2] = USB_FRAME_MIC;
    packet[3] = 0x00;
    packet[4] = (uint8_t)(PCM_BYTES & 0xFF);
    packet[5] = (uint8_t)((PCM_BYTES >> 8) & 0xFF);
    memcpy(packet + 6, pcm, PCM_BYTES);
    usb_serial_jtag_write_bytes((const char *)packet, sizeof(packet), 0);
    xSemaphoreGive(s_usb_tx_mutex);
}

static void tts_task(void *arg) {
    (void)arg;
    esp_tts_voice_t *voice = (esp_tts_voice_t *)&esp_tts_voice_xiaole;
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "voice_data");
    if (part) {
        const void *voicedata = NULL;
        esp_partition_mmap_handle_t mmap_handle = 0;
        if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &voicedata, &mmap_handle) == ESP_OK && voicedata) {
            esp_tts_voice_t *mapped = esp_tts_voice_set_init(&esp_tts_voice_template, (void *)voicedata);
            if (mapped) voice = mapped;
        }
    }
    esp_tts_handle_t tts = esp_tts_create(voice);
    if (!tts) {
        vTaskDelete(NULL);
        return;
    }

    char text[256];
    int16_t out_frame[FRAME_SAMPLES];
    size_t out_pos = 0;

    while (true) {
        if (xQueueReceive(s_tts_queue, text, portMAX_DELAY) != pdTRUE) continue;
        s_tts_cancel = false;
        s_tts_busy = true;
        out_pos = 0;

        if (esp_tts_parse_chinese(tts, text)) {
            int len = 0;
            while (!s_tts_cancel) {
                short *pcm = esp_tts_stream_play(tts, &len, 3);
                if (!pcm || len <= 0) break;
                for (int i = 0; i < len && !s_tts_cancel; ++i) {
                    out_frame[out_pos++] = pcm[i];
                    if (out_pos == FRAME_SAMPLES) {
                        push_speaker_frame(out_frame);
                        out_pos = 0;
                    }
                }
            }
            if (!s_tts_cancel && out_pos > 0) {
                memset(out_frame + out_pos, 0, (FRAME_SAMPLES - out_pos) * sizeof(int16_t));
                push_speaker_frame(out_frame);
            }
        }
        esp_tts_stream_reset(tts);
        s_tts_busy = false;
    }
}

static void audio_task(void *arg) {
    (void)arg;
    static int32_t raw_rx[FRAME_SAMPLES * 2];
    static int32_t raw_tx[FRAME_SAMPLES * 2];
    static int16_t mic_pcm[FRAME_SAMPLES];
    static int16_t play_pcm[FRAME_SAMPLES];
    static const int16_t silence[FRAME_SAMPLES] = {0};

    float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    uint32_t tone_phase = 0;

    while (true) {
        size_t bytes_read = 0;
        if (i2s_channel_read(s_rx_chan, raw_rx, sizeof(raw_rx), &bytes_read, pdMS_TO_TICKS(100)) == ESP_OK &&
            bytes_read == sizeof(raw_rx)) {
            int peak = 0;
            float sum_sq = 0.0f;
            for (size_t i = 0; i < FRAME_SAMPLES; ++i) {
                float input = (float)(raw_rx[2 * i] >> 16);
                float output = 0.97803048f * input - 1.95606096f * x1 + 0.97803048f * x2
                               + 1.95557824f * y1 - 0.95654368f * y2;
                x2 = x1; x1 = input;
                y2 = y1; y1 = output;
                sum_sq += output * output;
                int32_t gained = (int32_t)lrintf(output * 2.0f);
                if (gained > 32767) gained = 32767;
                if (gained < -32768) gained = -32768;
                mic_pcm[i] = (int16_t)gained;
                int lvl = abs((int)mic_pcm[i]);
                if (lvl > peak) peak = lvl;
            }
            s_mic_peak = peak;
            s_mic_rms = (int)sqrtf(sum_sq / FRAME_SAMPLES);
            if (s_mic_rms > 250) s_last_mic_ms = now_ms();

            uint32_t t_now = now_ms();
            bool spk_busy = s_tts_busy || (s_tone_frames > 0) ||
                            (s_last_spk_ms != 0 && (t_now - s_last_spk_ms) < ECHO_MUTE_MS);
            send_usb_mic_frame(spk_busy ? silence : mic_pcm);
        }

        if (s_tone_frames > 0) {
            for (size_t i = 0; i < FRAME_SAMPLES; ++i) {
                play_pcm[i] = (int16_t)(6000.0 * sin(2.0 * M_PI * 440.0 * (tone_phase + i) / SAMPLE_RATE));
            }
            tone_phase += FRAME_SAMPLES;
            s_tone_frames--;
            s_last_spk_ms = now_ms();
        } else if (xQueueReceive(s_play_queue, play_pcm, 0) == pdTRUE) {
            s_last_spk_ms = now_ms();
        } else {
            memset(play_pcm, 0, sizeof(play_pcm));
        }

        for (size_t i = 0; i < FRAME_SAMPLES; ++i) {
            int32_t s = ((int32_t)play_pcm[i]) << 16;
            raw_tx[2 * i] = s;
            raw_tx[2 * i + 1] = s;
        }
        size_t written = 0;
        i2s_channel_write(s_tx_chan, raw_tx, sizeof(raw_tx), &written, pdMS_TO_TICKS(100));
    }
}

static void handle_usb_json_line(const char *line) {
    cJSON *root = cJSON_Parse(line);
    if (!root) return;
    cJSON *cmd = cJSON_GetObjectItem(root, "cmd");
    if (cJSON_IsString(cmd) && cmd->valuestring) {
        if (strcmp(cmd->valuestring, "audio_on") == 0) {
            s_usb_audio_enabled = true;
        } else if (strcmp(cmd->valuestring, "audio_off") == 0) {
            s_usb_audio_enabled = false;
        } else if (strcmp(cmd->valuestring, "tts") == 0) {
            cJSON *txt = cJSON_GetObjectItem(root, "text");
            if (cJSON_IsString(txt) && txt->valuestring) {
                enqueue_tts_text(txt->valuestring);
            }
        } else if (strcmp(cmd->valuestring, "tts_cancel") == 0) {
            cancel_tts();
        }
        cJSON_Delete(root);
        return;
    }
    cJSON_Delete(root);

    if (s_atlas_cmd_pending) {
        strncpy(s_atlas_cmd_resp, line, sizeof(s_atlas_cmd_resp) - 1);
        s_atlas_cmd_resp[sizeof(s_atlas_cmd_resp) - 1] = '\0';
        s_atlas_cmd_pending = false;
        xSemaphoreGive(s_cmd_sem);
    }
}

static void usb_rx_task(void *arg) {
    (void)arg;
    enum { S_IDLE, S_M1, S_TYPE, S_RES, S_L0, S_L1, S_PAY, S_JSON } st = S_IDLE;
    uint8_t ftype = 0;
    uint16_t plen = 0, ppos = 0;
    static uint8_t pbuf[PCM_BYTES];
    static char jbuf[512];
    size_t jpos = 0;
    uint8_t rx[256];

    while (true) {
        int n = usb_serial_jtag_read_bytes(rx, sizeof(rx), pdMS_TO_TICKS(20));
        if (n <= 0) continue;
        for (int i = 0; i < n; ++i) {
            uint8_t b = rx[i];
            if (st == S_IDLE) {
                if (b == USB_MAGIC0) st = S_M1;
                else if (b == '{') { jpos = 0; jbuf[jpos++] = '{'; st = S_JSON; }
            } else if (st == S_JSON) {
                if (b == '\n') {
                    jbuf[jpos] = '\0';
                    handle_usb_json_line(jbuf);
                    st = S_IDLE;
                } else if (b != '\r' && jpos + 1 < sizeof(jbuf)) {
                    jbuf[jpos++] = (char)b;
                }
            } else if (st == S_M1) {
                st = (b == USB_MAGIC1) ? S_TYPE : S_IDLE;
            } else if (st == S_TYPE) {
                ftype = b; st = S_RES;
            } else if (st == S_RES) {
                st = S_L0;
            } else if (st == S_L0) {
                plen = b; st = S_L1;
            } else if (st == S_L1) {
                plen |= ((uint16_t)b << 8);
                ppos = 0;
                bool valid_spk = (ftype == USB_FRAME_SPK && plen == PCM_BYTES);
                bool valid_sub = (ftype == USB_FRAME_SUBTITLE && plen == SUBTITLE_BYTES);
                st = (valid_spk || valid_sub) ? S_PAY : S_IDLE;
            } else if (st == S_PAY) {
                pbuf[ppos++] = b;
                if (ppos >= plen) {
                    if (ftype == USB_FRAME_SPK) {
                        push_speaker_frame((const int16_t *)pbuf);
                    } else if (ftype == USB_FRAME_SUBTITLE) {
                        st7735_face_set_subtitle(pbuf, now_ms());
                    }
                    st = S_IDLE;
                }
            }
        }
    }
}

static void display_task(void *arg) {
    (void)arg;
    while (true) {
        uint32_t t = now_ms();
        bool speaking = s_tts_busy || (s_tone_frames > 0) || (s_last_spk_ms != 0 && (t - s_last_spk_ms) < 300);
        bool listening = (s_last_mic_ms != 0 && (t - s_last_mic_ms) < 600);
        uint32_t last_act = (s_last_mic_ms > s_last_spk_ms) ? s_last_mic_ms : s_last_spk_ms;
        bool sleeping = !speaking && !listening && (t > 12000) && (last_act == 0 || (t - last_act) > 12000);
        face_state_t fs = speaking ? FACE_SPEAKING
                        : listening ? FACE_LISTENING
                        : sleeping ? FACE_SLEEP
                                   : FACE_IDLE;
        st7735_face_draw(fs, t);
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

static void url_decode(char *dst, const char *src, size_t max_len) {
    size_t i = 0;
    while (*src && i + 1 < max_len) {
        if (*src == '%' && src[1] && src[2]) {
            char hex[3] = { src[1], src[2], '\0' };
            dst[i++] = (char)strtol(hex, NULL, 16);
            src += 3;
        } else if (*src == '+') {
            dst[i++] = ' ';
            src++;
        } else {
            dst[i++] = *src++;
        }
    }
    dst[i] = '\0';
}

static void call_atlas(const char *cmd, const char *task, uint32_t timeout_ms, char *out, size_t out_len) {
    xSemaphoreTake(s_cmd_sem, 0);
    s_atlas_cmd_pending = true;

    cJSON *req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "cmd", cmd);
    if (task && task[0]) cJSON_AddStringToObject(req, "task", task);
    char *printed = cJSON_PrintUnformatted(req);
    cJSON_Delete(req);
    if (printed) {
        size_t len = strlen(printed);
        char *line = malloc(len + 2);
        if (line) {
            memcpy(line, printed, len);
            line[len] = '\n';
            line[len + 1] = '\0';
            usb_write_locked(line, len + 1, pdMS_TO_TICKS(500));
            free(line);
        }
        cJSON_free(printed);
    }

    if (xSemaphoreTake(s_cmd_sem, pdMS_TO_TICKS(timeout_ms)) == pdTRUE) {
        strncpy(out, s_atlas_cmd_resp, out_len - 1);
        out[out_len - 1] = '\0';
    } else {
        s_atlas_cmd_pending = false;
        snprintf(out, out_len, "{\"ok\":false,\"error\":\"Atlas response timeout\"}");
    }
}

static esp_err_t send_json(httpd_req_t *req, const char *json) {
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, json);
}

static esp_err_t h_index(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    return httpd_resp_sendstr(req, INDEX_HTML);
}

static esp_err_t h_status(httpd_req_t *req) {
    char buf[128];
    snprintf(buf, sizeof(buf), "{\"peak\":%d,\"rms\":%d,\"usb_audio\":%s}",
             s_mic_peak, s_mic_rms, s_usb_audio_enabled ? "true" : "false");
    return send_json(req, buf);
}

static esp_err_t h_tone(httpd_req_t *req) {
    s_tone_frames = 50;
    return send_json(req, "{\"ok\":true}");
}

static esp_err_t h_tts(httpd_req_t *req) {
    char query[384] = {0}, encoded[320] = {0}, decoded[256] = {0};
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK &&
        httpd_query_key_value(query, "text", encoded, sizeof(encoded)) == ESP_OK) {
        url_decode(decoded, encoded, sizeof(decoded));
        enqueue_tts_text(decoded);
        return send_json(req, "{\"ok\":true}");
    }
    return send_json(req, "{\"ok\":false,\"error\":\"missing text\"}");
}

static esp_err_t h_tts_cancel(httpd_req_t *req) {
    cancel_tts();
    return send_json(req, "{\"ok\":true}");
}

static esp_err_t h_atlas_start(httpd_req_t *req) {
    char resp[512];
    call_atlas("start", NULL, 20000, resp, sizeof(resp));
    return send_json(req, resp);
}

static esp_err_t h_atlas_stop(httpd_req_t *req) {
    char resp[512];
    call_atlas("stop", NULL, 20000, resp, sizeof(resp));
    return send_json(req, resp);
}

static esp_err_t h_demo_start(httpd_req_t *req) {
    char query[256] = {0}, encoded[160] = {0}, task[160] = {0}, resp[512];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK &&
        httpd_query_key_value(query, "task", encoded, sizeof(encoded)) == ESP_OK) {
        url_decode(task, encoded, sizeof(task));
    }
    call_atlas("demo_start", task, 25000, resp, sizeof(resp));
    return send_json(req, resp);
}

static esp_err_t h_demo_stop(httpd_req_t *req) {
    char resp[512];
    call_atlas("demo_stop", NULL, 120000, resp, sizeof(resp));
    return send_json(req, resp);
}

static esp_err_t h_demo_status(httpd_req_t *req) {
    char resp[512];
    call_atlas("demo_status", NULL, 5000, resp, sizeof(resp));
    return send_json(req, resp);
}

static void start_wifi_and_http(void) {
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    wifi_config_t ap = {
        .ap = {
            .ssid = AP_SSID,
            .ssid_len = strlen(AP_SSID),
            .password = AP_PASS,
            .max_connection = 4,
            .authmode = WIFI_AUTH_WPA2_PSK,
            .channel = 6,
        },
    };
    esp_wifi_set_mode(WIFI_MODE_AP);
    esp_wifi_set_config(WIFI_IF_AP, &ap);
    esp_wifi_start();

    httpd_config_t hc = HTTPD_DEFAULT_CONFIG();
    hc.max_uri_handlers = 12;
    hc.stack_size = 6144;
    httpd_handle_t srv = NULL;
    if (httpd_start(&srv, &hc) == ESP_OK) {
        const httpd_uri_t uris[] = {
            { .uri = "/",            .method = HTTP_GET,  .handler = h_index },
            { .uri = "/status",      .method = HTTP_GET,  .handler = h_status },
            { .uri = "/tone",        .method = HTTP_POST, .handler = h_tone },
            { .uri = "/tts",         .method = HTTP_POST, .handler = h_tts },
            { .uri = "/tts/cancel",  .method = HTTP_POST, .handler = h_tts_cancel },
            { .uri = "/atlas/start", .method = HTTP_POST, .handler = h_atlas_start },
            { .uri = "/atlas/stop",  .method = HTTP_POST, .handler = h_atlas_stop },
            { .uri = "/demo/start",  .method = HTTP_POST, .handler = h_demo_start },
            { .uri = "/demo/stop",   .method = HTTP_POST, .handler = h_demo_stop },
            { .uri = "/demo/status", .method = HTTP_GET,  .handler = h_demo_status },
        };
        for (size_t i = 0; i < sizeof(uris) / sizeof(uris[0]); ++i) {
            httpd_register_uri_handler(srv, &uris[i]);
        }
    }
}

static void init_i2s(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 6;
    chan_cfg.dma_frame_num = FRAME_SAMPLES;
    i2s_new_channel(&chan_cfg, &s_tx_chan, &s_rx_chan);

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK_IO,
            .ws = I2S_WS_IO,
            .dout = I2S_AMP_DIN_IO,
            .din = I2S_MIC_SD_IO,
        },
    };
    i2s_channel_init_std_mode(s_tx_chan, &std_cfg);
    i2s_channel_init_std_mode(s_rx_chan, &std_cfg);
    i2s_channel_enable(s_tx_chan);
    i2s_channel_enable(s_rx_chan);
}

void app_main(void) {
    usb_serial_jtag_driver_config_t usb_cfg = {
        .rx_buffer_size = 4096,
        .tx_buffer_size = 4096,
    };
    usb_serial_jtag_driver_install(&usb_cfg);

    s_play_queue = xQueueCreate(PLAY_QUEUE_DEPTH, PCM_BYTES);
    s_tts_queue = xQueueCreate(8, 256);
    s_usb_tx_mutex = xSemaphoreCreateMutex();
    s_cmd_sem = xSemaphoreCreateBinary();

    st7735_face_init();
    init_i2s();
    start_wifi_and_http();

    xTaskCreatePinnedToCore(audio_task, "audio", 8192, NULL, 6, NULL, 1);
    xTaskCreatePinnedToCore(tts_task, "tts", 8192, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(usb_rx_task, "usb_rx", 4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(display_task, "display", 4096, NULL, 3, NULL, 0);
}
