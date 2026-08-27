cmake_minimum_required(VERSION 3.16)
project(sabir_host_verify LANGUAGES CXX)

add_executable(sabir_host_verify
    ../tests/host_runtime_test.cpp
    ../components/sabir_runtime/sabir_model.cpp
    ../components/sabir_runtime/sabir_tokenizer.cpp
)
target_include_directories(sabir_host_verify PRIVATE ../components/sabir_runtime/include)
target_compile_features(sabir_host_verify PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(sabir_host_verify PRIVATE /O2 /utf-8)
else()
  target_compile_options(sabir_host_verify PRIVATE -O3 -Wall -Wextra)
endif()

