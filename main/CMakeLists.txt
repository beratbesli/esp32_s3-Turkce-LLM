idf_component_register(
    SRCS "app_main.cpp"
    INCLUDE_DIRS "."
    REQUIRES sabir_runtime esp_partition esp_timer esp_psram
)
