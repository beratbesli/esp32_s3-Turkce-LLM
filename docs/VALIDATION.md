# Validation record

Date: 2026-08-28
Source revision: `jetbabareal/Sabir-20M@6932f64c62ba27b75cd9a806962ce103a9af481d`  
Project commit: `f9584d116e2e31654191178206dcff122ff6571b`

## Passed

- Upstream inventory: 278 stored tensors/elements groups, 23,583,296 stored
  elements, 3,145,728 causal-mask buffer elements and 20,437,568 learned
  parameters.
- Group-64 deployment image: 11,161,472 bytes, SHA-256
  `6a5923fc46994e3c408872903803486a987262c492a216ee8afac6d64a894629`,
  payload CRC32 `ea98f70f`.
- Python tests: 10 passed, including six representative upstream
  SentencePiece/token-ID parity cases.
- FP32 vs exported Q4 last-position logits for
  `Kullanıcı: Nasılsın?\nModel: `: cosine 0.9895328, RMSE 0.56047,
  top-10 overlap 7/10.
- Portable C++ runtime vs Python Q4 for the same prompt: tokenizer IDs match,
  top-1 774/774, logit RMSE 0.00000208, maximum absolute error 0.00000858.
- [GitHub Actions run 33209428978](https://github.com/beratbesli/esp32_s3-Turkce-LLM/actions/runs/33209428978):
  host tests/build and official ESP-IDF 5.5.5 ESP32-S3 cross-build passed.
  Firmware binary was 216,768 bytes, leaving 2,928,960 bytes (`0x2cb140`,
  93%) of the 3 MiB app partition free. The flashable firmware artifact was
  published by the workflow.
- Full bootloader, partition table, application and 11,161,472-byte Q4 model
  image were flashed through COM12 using esptool 5.3.1; every write passed
  post-write hash verification. Later application-only updates preserved the
  model partition and also passed hash verification.
- The tested ESP32-S3 revision v0.2 reported 16 MiB MXIC flash at 80 MHz QIO and
  8 MiB AP octal PSRAM at 80 MHz. The ESP-IDF PSRAM memory test passed.
- The model partition CRC validated at boot. At runtime context 64, the KV cache
  used 1.50 MiB; 6.44 MiB PSRAM and 347.3 KiB internal memory remained free
  after initialization.
- The 115200-baud serial demo accepted `Merhaba`, generated 48 tokens without a
  task-watchdog warning, and returned to the prompt. The measured run took
  66.14 seconds including prefill, or 0.726 token/s. Serial CR/LF coalescing was
  then separately verified on the final commit.

## Not performed / scope limits

- Context 128/256 has not been validated on hardware.
- No long-duration generation soak, repeated reset/brownout test, current-draw
  measurement or thermal characterization has been performed.
- Only one board and one CH343 serial connection were tested.

Accordingly, a context-64 on-device functional run is supported. Claims of
context-128/256 stability, multi-board compatibility or production readiness
are not supported yet.
