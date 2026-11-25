idf_component_register(
    INCLUDE_DIRS "esp_tts_chinese/include"
)

# Copy prebuilt archives into the binary dir so GNU ld never sees non-ASCII source paths on Windows
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/esp_tts_chinese/esp32s3/libesp_tts_chinese.a"
    "${CMAKE_CURRENT_BINARY_DIR}/libesp_tts_chinese.a"
    COPYONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/esp_tts_chinese/esp32s3/libvoice_set_xiaole.a"
    "${CMAKE_CURRENT_BINARY_DIR}/libvoice_set_xiaole.a"
    COPYONLY
)

add_prebuilt_library(esp_tts_chinese
    "${CMAKE_CURRENT_BINARY_DIR}/libesp_tts_chinese.a"
)
add_prebuilt_library(voice_set_xiaole
    "${CMAKE_CURRENT_BINARY_DIR}/libvoice_set_xiaole.a"
)

target_link_libraries(${COMPONENT_TARGET} INTERFACE
    "-Wl,--start-group"
    esp_tts_chinese
    voice_set_xiaole
    "-Wl,--end-group"
)
