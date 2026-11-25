#pragma once
#include <stdbool.h>
#include <stdint.h>

#define SUBTITLE_W     160
#define SUBTITLE_BAND_H 14
#define SUBTITLE_BYTES ((SUBTITLE_W / 8) * SUBTITLE_BAND_H * 2)  // 560 bytes (top 14px + bottom 14px)

typedef enum {
    FACE_IDLE = 0,
    FACE_LISTENING = 1,
    FACE_SPEAKING = 2,
    FACE_SLEEP = 3,
} face_state_t;

void st7735_face_init(void);
void st7735_face_set_subtitle(const uint8_t *bitmap_560, uint32_t now_ms);
void st7735_face_draw(face_state_t state, uint32_t now_ms);
