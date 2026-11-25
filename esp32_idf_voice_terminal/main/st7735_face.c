#include "st7735_face.h"
#include "whale_sprites.h"

#include <math.h>
#include <string.h>
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define TFT_CS   10
#define TFT_MOSI 11
#define TFT_SCK  12
#define TFT_DC   13
#define TFT_RST  14
#define TFT_BL   15

#define LCD_W 160
#define LCD_H 128
#define STRIP_ROWS 16

static spi_device_handle_t s_spi = NULL;
static uint16_t *s_fb = NULL;
static uint16_t *s_dma_strip = NULL;
static uint8_t s_sub_buf[SUBTITLE_BYTES];
static volatile bool s_sub_has_top = false;
static volatile bool s_sub_has_bot = false;
static volatile uint32_t s_sub_updated_ms = 0;

void st7735_face_set_subtitle(const uint8_t *bitmap_560, uint32_t now_ms) {
    if (!bitmap_560) return;
    memcpy(s_sub_buf, bitmap_560, SUBTITLE_BYTES);
    bool top_any = false, bot_any = false;
    const size_t half = SUBTITLE_BYTES / 2;
    for (size_t i = 0; i < half; ++i) {
        if (s_sub_buf[i]) { top_any = true; break; }
    }
    for (size_t i = half; i < SUBTITLE_BYTES; ++i) {
        if (s_sub_buf[i]) { bot_any = true; break; }
    }
    s_sub_has_top = top_any;
    s_sub_has_bot = bot_any;
    s_sub_updated_ms = now_ms;
}

static void draw_subtitle_band(const uint8_t *band, int y0, uint16_t fg_col, uint16_t border_col) {
    // Semi-transparent dark banner (darken existing pixels) + crisp 1-bit text
    for (int row = 0; row < SUBTITLE_BAND_H; ++row) {
        int dy = y0 + row;
        if ((unsigned)dy >= LCD_H) continue;
        uint16_t *dst = s_fb + dy * LCD_W;
        const uint8_t *src = band + row * (SUBTITLE_W / 8);
        for (int x = 0; x < LCD_W; ++x) {
            bool on = (src[x >> 3] & (0x80 >> (x & 7))) != 0;
            if (on) {
                dst[x] = fg_col;
            } else {
                // Dim underlying pixel by ~75% for clean readable glass banner
                uint16_t raw = dst[x];
                uint16_t c = (uint16_t)((raw >> 8) | (raw << 8));
                uint16_t dimmed = (uint16_t)(((c >> 2) & 0x39E7));
                dst[x] = (uint16_t)((dimmed >> 8) | (dimmed << 8));
            }
        }
    }
    int line_y = (y0 == 0) ? (SUBTITLE_BAND_H - 1) : y0;
    if ((unsigned)line_y < LCD_H) {
        for (int x = 0; x < LCD_W; ++x) s_fb[line_y * LCD_W + x] = border_col;
    }
}

static inline uint16_t rgb565(uint8_t r, uint8_t g, uint8_t b) {
    uint16_t c = (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
    return (uint16_t)((c >> 8) | (c << 8));
}

static void lcd_cmd(uint8_t cmd) {
    gpio_set_level(TFT_DC, 0);
    spi_transaction_t t = { .length = 8, .tx_buffer = &cmd };
    spi_device_polling_transmit(s_spi, &t);
}

static void lcd_data(const void *data, size_t len) {
    if (len == 0) return;
    gpio_set_level(TFT_DC, 1);
    spi_transaction_t t = { .length = len * 8, .tx_buffer = data };
    spi_device_polling_transmit(s_spi, &t);
}

static void lcd_data_byte(uint8_t b) {
    lcd_data(&b, 1);
}

static void lcd_set_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1) {
    uint8_t col[4] = { (uint8_t)(x0 >> 8), (uint8_t)(x0 & 0xFF), (uint8_t)(x1 >> 8), (uint8_t)(x1 & 0xFF) };
    uint8_t row[4] = { (uint8_t)(y0 >> 8), (uint8_t)(y0 & 0xFF), (uint8_t)(y1 >> 8), (uint8_t)(y1 & 0xFF) };
    lcd_cmd(0x2A);
    lcd_data(col, 4);
    lcd_cmd(0x2B);
    lcd_data(row, 4);
    lcd_cmd(0x2C);
}

static inline void put_px(int x, int y, uint16_t c) {
    if ((unsigned)x < LCD_W && (unsigned)y < LCD_H) {
        s_fb[y * LCD_W + x] = c;
    }
}

static void fill_rect(int x, int y, int w, int h, uint16_t c) {
    for (int yy = y; yy < y + h; ++yy) {
        if ((unsigned)yy >= LCD_H) continue;
        for (int xx = x; xx < x + w; ++xx) {
            if ((unsigned)xx < LCD_W) s_fb[yy * LCD_W + xx] = c;
        }
    }
}

static void draw_circle(int cx, int cy, int r, uint16_t c) {
    int x = r, y = 0, err = 1 - r;
    while (x >= y) {
        put_px(cx + x, cy + y, c); put_px(cx + y, cy + x, c);
        put_px(cx - y, cy + x, c); put_px(cx - x, cy + y, c);
        put_px(cx - x, cy - y, c); put_px(cx - y, cy - x, c);
        put_px(cx + y, cy - x, c); put_px(cx + x, cy - y, c);
        ++y;
        if (err < 0) err += 2 * y + 1;
        else { --x; err += 2 * (y - x + 1); }
    }
}

static void draw_z_char(int x, int y, int s, uint16_t c) {
    // Minimal crisp pixel "Z" for sleeping animation
    fill_rect(x, y, 4 * s, s, c);
    fill_rect(x + 2 * s, y + s, s, s, c);
    fill_rect(x + s, y + 2 * s, s, s, c);
    fill_rect(x, y + 3 * s, 4 * s, s, c);
}

static void blit_sprite(const uint16_t *sprite, int dst_x, int dst_y) {
    for (int sy = 0; sy < WHALE_SPRITE_H; ++sy) {
        int dy = dst_y + sy;
        if ((unsigned)dy >= LCD_H) continue;
        const uint16_t *src_row = sprite + sy * WHALE_SPRITE_W;
        uint16_t *dst_row = s_fb + dy * LCD_W;
        for (int sx = 0; sx < WHALE_SPRITE_W; ++sx) {
            uint16_t px = src_row[sx];
            if (px != WHALE_COLOR_KEY) {
                int dx = dst_x + sx;
                if ((unsigned)dx < LCD_W) dst_row[dx] = px;
            }
        }
    }
}

void st7735_face_init(void) {
    gpio_config_t io = {
        .pin_bit_mask = (1ULL << TFT_DC) | (1ULL << TFT_RST) | (1ULL << TFT_BL),
        .mode = GPIO_MODE_OUTPUT,
    };
    gpio_config(&io);
    gpio_set_level(TFT_BL, 1);

    spi_bus_config_t bus = {
        .mosi_io_num = TFT_MOSI,
        .miso_io_num = -1,
        .sclk_io_num = TFT_SCK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = LCD_W * STRIP_ROWS * 2 + 16,
    };
    esp_err_t err_bus = spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO);

    spi_device_interface_config_t dev = {
        .clock_speed_hz = 20 * 1000 * 1000,
        .mode = 0,
        .spics_io_num = TFT_CS,
        .queue_size = 2,
    };
    esp_err_t err_dev = spi_bus_add_device(SPI2_HOST, &dev, &s_spi);

    s_fb = heap_caps_malloc(LCD_W * LCD_H * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_fb) s_fb = malloc(LCD_W * LCD_H * sizeof(uint16_t));
    s_dma_strip = heap_caps_malloc(LCD_W * STRIP_ROWS * sizeof(uint16_t), MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    ESP_LOGW("ST7735", "init bus=%d dev=%d fb=%p strip=%p", err_bus, err_dev, s_fb, s_dma_strip);

    gpio_set_level(TFT_RST, 0);
    vTaskDelay(pdMS_TO_TICKS(50));
    gpio_set_level(TFT_RST, 1);
    vTaskDelay(pdMS_TO_TICKS(120));

    lcd_cmd(0x01); vTaskDelay(pdMS_TO_TICKS(120));
    lcd_cmd(0x11); vTaskDelay(pdMS_TO_TICKS(120));
    lcd_cmd(0x3A); lcd_data_byte(0x05);
    lcd_cmd(0x36); lcd_data_byte(0x60);  // Landscape 160x128
    lcd_cmd(0x13);
    lcd_cmd(0x29); vTaskDelay(pdMS_TO_TICKS(50));
}

void st7735_face_draw(face_state_t state, uint32_t now_ms) {
    if (!s_spi || !s_fb) return;

    float t = (float)now_ms * 0.001f;

    // 1. Deep-ocean ambient gradient background
    for (int y = 0; y < LCD_H; ++y) {
        uint8_t r = (uint8_t)(8 + (y * 6) / LCD_H);
        uint8_t g = (uint8_t)(18 + (y * 18) / LCD_H);
        uint8_t b = (uint8_t)(40 + (y * 28) / LCD_H);
        if (state == FACE_LISTENING) g = (uint8_t)(g + 10);
        else if (state == FACE_SPEAKING) { r = (uint8_t)(r + 10); b = (uint8_t)(b + 10); }
        uint16_t row_c = rgb565(r, g, b);
        for (int x = 0; x < LCD_W; ++x) s_fb[y * LCD_W + x] = row_c;
    }

    // 2. Ambient rising bubbles on left/right edges
    uint16_t bub_c = rgb565(96, 178, 255);
    uint16_t bub_hi = rgb565(215, 240, 255);
    const int bx_arr[4] = { 12, 22, 144, 150 };
    const int br_arr[4] = { 3, 2, 4, 2 };
    const uint32_t bs_arr[4] = { 28, 22, 34, 25 };
    for (int i = 0; i < 4; ++i) {
        int by = LCD_H - (int)(((now_ms / bs_arr[i]) + i * 33) % (LCD_H + 14));
        int bx = bx_arr[i] + (int)(2.0f * sinf(t * 3.0f + i * 1.7f));
        draw_circle(bx, by, br_arr[i], bub_c);
        put_px(bx - 1, by - 1, bub_hi);
    }

    // 3. Select DeepSeek Whale Girl sprite + lively bounce/float motion
    const uint16_t *sprite = whale_idle;
    int offset_x = 0;
    int offset_y = 0;
    if (state == FACE_SPEAKING) {
        // Energetic cheerful bounce while speaking!
        sprite = (((now_ms / 240) % 4) == 3) ? whale_idle : whale_speaking;
        offset_y = -(int)lrintf(4.0f * fabsf(sinf(t * 9.0f)));
    } else if (state == FACE_LISTENING) {
        // Leaning/walking eagerly toward the speaker
        sprite = whale_listening;
        offset_x = (int)lrintf(3.0f * sinf(t * 5.0f));
        offset_y = -(int)lrintf(2.5f * fabsf(sinf(t * 6.0f)));
    } else if (state == FACE_SLEEP) {
        // Gentle slow breathing while hugging whale tail
        sprite = whale_sleep;
        offset_y = (int)lrintf(1.5f * sinf(t * 1.8f));
    } else {
        // Soft breathing float in IDLE
        sprite = whale_idle;
        offset_y = (int)lrintf(2.0f * sinf(t * 2.4f));
    }

    int sx = (LCD_W - WHALE_SPRITE_W) / 2 + offset_x;
    int sy = (LCD_H - WHALE_SPRITE_H) - 4 + offset_y;
    blit_sprite(sprite, sx, sy);

    // 4. Foreground state indicators (non-intrusive, frames the character)
    if (state == FACE_SLEEP) {
        // Floating "Z z" sleep bubbles on upper right
        uint16_t z_col = rgb565(150, 210, 255);
        int p1 = (int)((now_ms / 55) % 34);
        int p2 = (int)(((now_ms / 55) + 17) % 34);
        draw_z_char(124 + p1 / 5, 42 - p1, 2, z_col);
        draw_z_char(136 + p2 / 6, 52 - p2, 1, bub_hi);
    } else if (state == FACE_LISTENING) {
        // Side emerald sound wave bars
        uint16_t eq_c = rgb565(72, 240, 180);
        for (int i = 0; i < 4; ++i) {
            int h = 4 + (int)lrintf(14.0f * fabsf(sinf(t * 7.5f + i * 1.1f)));
            fill_rect(6 + i * 4, 72 - h / 2, 2, h, eq_c);
            fill_rect(142 + i * 4, 72 - h / 2, 2, h, eq_c);
        }
    } else if (state == FACE_SPEAKING) {
        // Warm golden equalizer bars on both sides + top sparkle
        uint16_t eq_c = rgb565(255, 214, 82);
        for (int i = 0; i < 4; ++i) {
            int h = 6 + (int)lrintf(18.0f * fabsf(sinf(t * 12.5f + i * 0.95f)));
            fill_rect(6 + i * 4, 72 - h / 2, 2, h, eq_c);
            fill_rect(142 + i * 4, 72 - h / 2, 2, h, eq_c);
        }
    }

    // 5. Subtitle overlay (top = heard speech, bottom = assistant reply, auto-hide after 10s)
    if (s_sub_updated_ms != 0 && (now_ms - s_sub_updated_ms) < 10000) {
        if (s_sub_has_top) {
            draw_subtitle_band(s_sub_buf, 0, rgb565(130, 255, 210), rgb565(48, 175, 135));
        }
        if (s_sub_has_bot) {
            draw_subtitle_band(s_sub_buf + SUBTITLE_BYTES / 2, LCD_H - SUBTITLE_BAND_H,
                               rgb565(255, 240, 160), rgb565(215, 175, 65));
        }
    }

    lcd_set_window(0, 0, LCD_W - 1, LCD_H - 1);
    if (s_dma_strip) {
        for (int y = 0; y < LCD_H; y += STRIP_ROWS) {
            int rows = (y + STRIP_ROWS <= LCD_H) ? STRIP_ROWS : (LCD_H - y);
            size_t bytes = (size_t)rows * LCD_W * sizeof(uint16_t);
            memcpy(s_dma_strip, s_fb + y * LCD_W, bytes);
            lcd_data(s_dma_strip, bytes);
        }
    } else {
        lcd_data(s_fb, LCD_W * LCD_H * sizeof(uint16_t));
    }
}
