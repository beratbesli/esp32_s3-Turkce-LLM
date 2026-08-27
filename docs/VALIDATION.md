# Validation record

Date: 2026-08-27  
Source revision: `jetbabareal/Sabir-20M@6932f64c62ba27b75cd9a806962ce103a9af481d`  
Project commit: `0a5c82374e616dd475c97fa94e9a32fe77d4d86a`

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
- [GitHub Actions run 33077356311](https://github.com/beratbesli/esp32_s3-Turkce-LLM/actions/runs/33077356311):
  host tests/build and official ESP-IDF 5.5.5 ESP32-S3 cross-build passed.
  Firmware binary was 216,224 bytes (`0x34ca0`), leaving 2,929,504 bytes
  (`0x2cb360`, 93%) of the 3 MiB app partition free.

## Not performed

- No image was flashed to a physical ESP32-S3 N16R8.
- No serial transcript, reset/brownout stability run, PSRAM allocation log,
  current draw, temperature or on-device token/s measurement exists yet.
- Context 128/256 has not been validated on hardware.

Accordingly, “CI-builds and host reference agrees” is supported; “runs on the
physical board” is not yet supported.
