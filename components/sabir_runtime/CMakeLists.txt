idf_component_register(
    SRCS "sabir_model.cpp" "sabir_tokenizer.cpp"
    INCLUDE_DIRS "include"
)
target_compile_options(${COMPONENT_LIB} PRIVATE -O3)

