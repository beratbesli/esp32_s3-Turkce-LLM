# SABIR4 model-image format

The production image is a little-endian, CRC-protected binary designed to be
memory-mapped from the ESP-IDF `model` data partition. Version 1 contains:

1. an 80-byte header (`SABIR4\0\0`, architecture, directory layout, CRC);
2. fixed 80-byte tensor directory entries;
3. 64-byte-aligned tensor payloads;
4. the compact tokenizer asset as a `tokenizer` blob.

Matrices use row-aligned group-wise symmetric int4. Each row stores two signed
codes per byte as `(q + 8)`, followed by one IEEE-FP16 scale per 64-column group.
Codes are `round(weight / fp16(scale))`, clamped to `[-7, 7]`. LayerNorm values
and biases stay FP32. Causal `tril` buffers are not stored.

`tools/quantize_model.py` is the canonical writer and
`components/sabir_runtime/sabir_model.cpp` is the canonical reader. The payload
CRC covers all bytes from the aligned payload start through the image end.

The optional `tools/emit_c_array.py` converts the exact image to an aligned
`const uint8_t[]`. It is useful for host fixtures or alternate packaging. The
default ESP32 build does not link that 10+ MiB array into the application image:
doing so would duplicate the weights and overflow the 3 MiB app partition. The
same bytes are instead written once at `0x310000` and memory-mapped in place.

