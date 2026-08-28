# Sabır-20M on ESP32-S3 N16R8

**Status: alpha — host-validated, ESP-IDF 5.5.5 CI build passing, and tested on
one ESP32-S3 N16R8 at context 64.**

This project ports the Turkish
[jetbabareal/Sabir-20M](https://huggingface.co/jetbabareal/Sabir-20M)
decoder-only Transformer to an ESP32-S3 with 16 MiB flash and 8 MiB octal
PSRAM. Inference, tokenization and sampling run entirely on the device after
deployment; no network service is used.

The repository contains source and reproducible conversion tools, not the
upstream safetensors or generated 10+ MiB deployment image. The current GitHub
repository is intended to remain private until its owner deliberately changes
that setting.

> **Hardware validation disclosure:** commit `f9584d1` was flashed to one
> ESP32-S3 N16R8 (ESP32-S3 revision v0.2, 16 MiB flash, 8 MiB AP octal PSRAM)
> through a CH343 USB/UART adapter. Boot, PSRAM self-test, model CRC validation,
> serial input and context-64 generation passed. A 48-token `Merhaba` run on the
> immediately preceding inference-identical build took 66.14 seconds including
> prefill (0.726 token/s). This is a single-board functional test, not a
> stability, thermal or production qualification.

## What is implemented

- Exact 8-layer, 384-wide, 6-head, 8,000-vocabulary architecture, including
  learned positional embeddings, ReLU FFN, biases, pre-LayerNorm and the
  upstream model's unusual `1/sqrt(384)` attention scale.
- Reproducible safetensors inspection and group-wise int4 export with FP16
  scales, payload CRC, manifest hashes and embedded compact tokenizer asset.
- Minimal C++17 runtime for flash-mapped Q4 weights, FP32 activations, KV cache,
  causal attention and top-k/temperature sampling.
- ESP-IDF serial prompt demo with configurable context (default 64).
- Host tests for quantization, binary integrity, SentencePiece token-ID parity,
  FP32-to-Q4 logit similarity and Python-Q4-to-C++ runtime parity.
- Optional C-array emitter. Production deployment keeps the identical bytes in
  the separate model partition to avoid duplicating them in the app image.

The architecture investigation and exact parameter accounting are in
[`docs/MODEL_CARD_FINDINGS.md`](docs/MODEL_CARD_FINDINGS.md). The format is in
[`docs/MODEL_FORMAT.md`](docs/MODEL_FORMAT.md).

## Hardware and software

- ESP32-S3 N16R8: 16 MiB flash, 8 MiB OPI PSRAM
- USB/UART serial connection at 115200 baud
- ESP-IDF 5.5.5 (the CI target; newer compatible releases may also work)
- Python 3.10+ for conversion and tests
- Approximately 200 MiB free host disk for the source and generated model files

Arduino is not used. ESP-IDF provides direct partition mmap, capability-aware
SRAM/PSRAM allocation, explicit build configuration and a smaller, auditable
runtime surface.

## Flash plan

`partitions.csv` fills exactly 16 MiB without OTA slots:

| Region | Offset | Size | Purpose |
|---|---:|---:|---|
| Bootloader | `0x000000` | up to 32 KiB | ESP-IDF second-stage bootloader |
| Partition table | `0x008000` | 4 KiB | flash layout |
| NVS | `0x009000` | 24 KiB | system data |
| PHY init | `0x00F000` | 4 KiB | radio calibration area |
| Factory app | `0x010000` | 3 MiB | firmware/runtime |
| Model data | `0x310000` | 12.9375 MiB | mmap'd SABIR4 image |

The verified group-64 image is 11,161,472 bytes (10.644 MiB), leaving 2,404,480
bytes in the model partition. A regular firmware reflash does not need to write
the model again; an erase-flash operation does.

## RAM plan

| Memory | Contents |
|---|---|
| Internal SRAM | current activation, LayerNorm buffer, QKV, attention output, 1,536-value FFN workspace, attention scores |
| 8 MiB PSRAM | K/V cache, 8,000-float logits, tokenizer hash table and large persistent buffers |
| Flash mmap | Q4 matrices, FP16 group scales, FP32 biases/norms, tokenizer strings |

KV cache cost is `2 × layers × context × embedding × 4`: 1.5 MiB at context 64,
3 MiB at 128 and 6 MiB at 256. The conservative default is 64. Context 128 may
be selected after measuring free PSRAM on the target. Context 256 is compiled
as an option but must not be treated as supported until allocation, stability
and long-prompt tests pass on the physical board.

Dense Q4 weights are read directly from mapped flash. Unlike `esp32-ai`, this
port cannot stage the entire Sabir dense core as int8 in 8 MiB PSRAM, and it does
not add Per-Layer Embeddings (PLE). PLE would require a changed architecture and
retraining, not a converter-only change.

## 1. Install host tools

From this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## 2. Download and inspect the upstream model

For a frozen build, use the commit recorded by a previous manifest rather than
`main`. The revision inspected for the current validation was
`6932f64c62ba27b75cd9a806962ce103a9af481d`.

```powershell
python tools/fetch_model.py `
  --revision 6932f64c62ba27b75cd9a806962ce103a9af481d `
  --output artifacts/source

python tools/inspect_model.py `
  --weights artifacts/source/model.safetensors `
  --config artifacts/source/config.json
```

The download tool records byte sizes, SHA-256 values and the resolved Hugging
Face revision in `artifacts/source/source-manifest.json`. Everything under
`artifacts/` is ignored by Git.

## 3. Quantize

```powershell
python tools/quantize_model.py `
  --weights artifacts/source/model.safetensors `
  --config artifacts/source/config.json `
  --tokenizer artifacts/source/tokenizer.model `
  --group-size 64 `
  --output artifacts/sabir-20m-q4.sabir
```

The sidecar `.json` reports every tensor's RMSE/max error, input hashes, output
hash, image size and CRC. C-array form is optional:

```powershell
python tools/emit_c_array.py `
  --input artifacts/sabir-20m-q4.sabir `
  --output artifacts/generated/sabir_model_image.c
```

Do not add that generated C file or the binary to Git. Linking the array into
the 3 MiB app is not the production path; flash the same image to the model
partition as shown below.

## 4. Validate on the host

Fast unit/tokenizer tests:

```powershell
$env:SABIR_TOKENIZER = (Resolve-Path artifacts/source/tokenizer.model)
python -m pytest -q
```

Full FP32/Q4 reference and golden fixture:

```powershell
python tools/verify_reference.py `
  --weights artifacts/source/model.safetensors `
  --tokenizer artifacts/source/tokenizer.model `
  --image artifacts/sabir-20m-q4.sabir `
  --golden-output artifacts/golden-q4.bin

cmake -S host -B build-host -DCMAKE_BUILD_TYPE=Release
cmake --build build-host --config Release
.\build-host\Release\sabir_host_verify.exe `
  artifacts/sabir-20m-q4.sabir artifacts/golden-q4.bin
```

On single-config generators the executable is usually at
`build-host/sabir_host_verify` instead.

Current host results:

- 10 Python tests passed.
- FP32 vs Q4: cosine 0.9895328, logit RMSE 0.56047, top-10 overlap 7/10.
- Python Q4 vs C++ Q4: same top-1 (`774`), logit RMSE 0.00000208,
  maximum absolute difference 0.00000858.
- ESP-IDF 5.5.5 CI: successful ESP32-S3 cross-build; app binary 216,224 bytes
  (`0x34ca0`), leaving 93% of the 3 MiB app partition free.

The recorded evidence and scope are in [`docs/VALIDATION.md`](docs/VALIDATION.md).

## 5. Build and flash firmware

Open an ESP-IDF 5.5.5 environment, then:

```powershell
idf.py set-target esp32s3
idf.py build
idf.py -p COM7 flash
python tools/flash_model.py --port COM7 --image artifacts/sabir-20m-q4.sabir
idf.py -p COM7 monitor
```

Replace `COM7`. `flash_model.py` validates image CRC and the 12.9375 MiB size
limit before calling esptool at offset `0x310000`. The firmware validates the
payload CRC again before allocation/inference.

To change context or generation defaults:

```powershell
idf.py menuconfig
```

Use **Sabir-20M runtime** to select context length, max new tokens, temperature
and top-k. Rebuild after changing them.

## Serial demo

After startup and CRC validation, the monitor prints `Siz>`. Enter a UTF-8
Turkish prompt and press Enter. Firmware wraps it exactly as the model card does:

```text
Kullanıcı: <prompt>
Model: 
```

Generated SentencePiece pieces stream to the same serial terminal. Generation
stops at EOS, the model's user-defined speaker tokens, the configured token
limit or the runtime context boundary.

## Tokenizer compatibility

The compact device BPE reproduces SentencePiece IDs for normal Turkish UTF-8,
space collapsing and the model's `Kullanıcı:`/`Model:` user-defined symbols.
Tests cover representative Turkish strings. The upstream tokenizer uses a
240 KiB `nmt_nfkc` normalization charsmap; the device intentionally omits the
full Unicode compatibility table to save flash/runtime complexity. Exotic
compatibility characters may therefore tokenize differently. Normalize prompts
to ordinary NFKC Unicode on the sending side when exact parity matters.

## Known limitations and expected performance

- One ESP32-S3 N16R8 passed boot and a context-64 serial generation test. The
  observed 48-token run was 0.726 token/s including prefill. Context 128/256,
  long-duration stability, reset/brownout behavior, current draw and thermal
  behavior remain untested.
- Q4 changes logits and can change sampled output; the measured host similarity
  is reported above rather than described as bit-identical to FP32.
- The 8,000-row untied output head is scanned for every generated token and the
  dense core remains Q4 in flash. Performance is expected to be flash-bandwidth
  and scalar-matvec limited until device profiling guides optimization.
- The roughly 9.5–9.88 token/s reported by
  [`slvDev/esp32-ai`](https://github.com/slvDev/esp32-ai) for its 28.9M PLE model
  is **not a Sabir-20M estimate or guarantee**. The architectures, head cost,
  memory staging and kernels differ.
- There is no OTA slot in the 16 MiB partition plan. Firmware updates are wired
  updates unless the partition layout is redesigned.
- Sabir-20M itself is a small, narrow-domain model trained on a constrained
  corpus; it should not be expected to provide robust factual or reasoning
  answers.

The next concrete engineering step is to profile the output head and dense
matvec on the tested board, then run context-128 allocation and long-prompt
stability tests before changing kernels.

## License, attribution and data warning

Original code in this repository is MIT licensed; see [`LICENSE`](LICENSE).
Attributions and third-party notices are in [`NOTICE`](NOTICE).

Sabir-20M is credited to `jetbabareal` and its Hugging Face model card labels it
MIT licensed:

```bibtex
@misc{sabir20m,
  title  = {Sabir-20M: A Turkish Micro Language Model},
  author = {jetbabareal},
  year   = {2025},
  url    = {https://huggingface.co/jetbabareal/Sabir-20M}
}
```

This port produces an ESP32-S3 int4 derivative of Sabir-20M weights when the
user runs the converter. No weights are committed here. At the time of review,
the model repository exposed the MIT label/model-card statement but no separate
LICENSE file. Its training-data description (approximately 1.5M tokens of
Turkish dialogues) does not provide adequate source/license detail for this
project to assert complete legal compliance. Review upstream terms and data
provenance before redistributing a generated model image.
