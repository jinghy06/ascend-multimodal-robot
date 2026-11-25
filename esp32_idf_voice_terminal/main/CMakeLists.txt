idf_component_register(
    SRCS "main.c" "st7735_face.c"
    INCLUDE_DIRS "."
    REQUIRES driver esp_wifi esp_http_server nvs_flash esp_netif esp_event esp_timer esp_partition spi_flash lwip json esp_tts_local
)
