#!/usr/bin/env python3
"""Export upstream Sabir-20M safetensors to the device's group-wise int4 image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from sabir_tools.model_format import (
    DTYPE_BLOB,
    DTYPE_F32,
    DTYPE_Q4,
    FLAG_HAS_TOKENIZER,
    TensorRecord,
    quantize_q4,
    write_image,
)
from sabir_tools.tokenizer import TokenizerAsset


EXPECTED = {"n_layer": 8, "n_embd": 384, "n_head": 6, "block_size": 256, "vocab_size": 8000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def f32_record(name: str, value: np.ndarray) -> TensorRecord:
    value = np.asarray(value, dtype="<f4")
    return TensorRecord(name, DTYPE_F32, tuple(value.shape), value.tobytes())


def q4_record(name: str, value: np.ndarray, group: int, statistics: list[dict]) -> TensorRecord:
    codes, scales, reconstructed = quantize_q4(value, group)
    value = np.asarray(value, dtype=np.float32)
    error = reconstructed - value
    statistics.append({
        "name": name,
        "shape": list(value.shape),
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error, dtype=np.float64))),
        "source_abs_max": float(np.max(np.abs(value))),
    })
    return TensorRecord(name, DTYPE_Q4, tuple(value.shape), codes, scales)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sabir-20m-q4.sabir"))
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    mismatch = {key: (value, config.get(key)) for key, value in EXPECTED.items() if config.get(key) != value}
    if mismatch:
        raise SystemExit(f"refusing unexpected architecture: {mismatch}")
    if args.group_size < 8 or args.group_size > 256 or args.group_size & (args.group_size - 1):
        raise SystemExit("group size must be a power of two between 8 and 256")

    records: list[TensorRecord] = []
    stats: list[dict] = []
    with safe_open(args.weights, framework="np", device="cpu") as source:
        keys = set(source.keys())
        expected_masks = config["n_layer"] * config["n_head"]
        masks = [key for key in keys if key.endswith(".tril")]
        if len(masks) != expected_masks:
            raise SystemExit(f"expected {expected_masks} causal masks, found {len(masks)}")

        def get(name: str) -> np.ndarray:
            if name not in keys:
                raise KeyError(f"missing upstream tensor: {name}")
            return source.get_tensor(name)

        records.append(q4_record("tok_emb.weight", get("token_embedding_table.weight"), args.group_size, stats))
        records.append(q4_record("pos_emb.weight", get("position_embedding_table.weight"), args.group_size, stats))
        for layer in range(config["n_layer"]):
            prefix = f"blocks.{layer}"
            records.extend([
                f32_record(f"b{layer}.ln1.weight", get(f"{prefix}.ln1.weight")),
                f32_record(f"b{layer}.ln1.bias", get(f"{prefix}.ln1.bias")),
                f32_record(f"b{layer}.ln2.weight", get(f"{prefix}.ln2.weight")),
                f32_record(f"b{layer}.ln2.bias", get(f"{prefix}.ln2.bias")),
            ])
            q = np.concatenate([get(f"{prefix}.sa.heads.{head}.query.weight") for head in range(config["n_head"])])
            k = np.concatenate([get(f"{prefix}.sa.heads.{head}.key.weight") for head in range(config["n_head"])])
            v = np.concatenate([get(f"{prefix}.sa.heads.{head}.value.weight") for head in range(config["n_head"])])
            records.extend([
                q4_record(f"b{layer}.qkv.weight", np.concatenate((q, k, v)), args.group_size, stats),
                q4_record(f"b{layer}.attn.weight", get(f"{prefix}.sa.proj.weight"), args.group_size, stats),
                f32_record(f"b{layer}.attn.bias", get(f"{prefix}.sa.proj.bias")),
                q4_record(f"b{layer}.ff1.weight", get(f"{prefix}.ffwd.net.0.weight"), args.group_size, stats),
                f32_record(f"b{layer}.ff1.bias", get(f"{prefix}.ffwd.net.0.bias")),
                q4_record(f"b{layer}.ff2.weight", get(f"{prefix}.ffwd.net.3.weight"), args.group_size, stats),
                f32_record(f"b{layer}.ff2.bias", get(f"{prefix}.ffwd.net.3.bias")),
            ])
        records.extend([
            f32_record("ln_f.weight", get("ln_f.weight")),
            f32_record("ln_f.bias", get("ln_f.bias")),
            q4_record("lm_head.weight", get("lm_head.weight"), args.group_size, stats),
            f32_record("lm_head.bias", get("lm_head.bias")),
        ])

    tokenizer = TokenizerAsset.from_sentencepiece(args.tokenizer)
    if len(tokenizer.pieces) != config["vocab_size"]:
        raise SystemExit("tokenizer vocabulary does not match model")
    records.append(TensorRecord("tokenizer", DTYPE_BLOB, (len(tokenizer.to_bytes()),), tokenizer.to_bytes()))
    output_config = {**config, "group_size": args.group_size}
    header = write_image(args.output, output_config, records, FLAG_HAS_TOKENIZER)
    manifest = {
        "format": "SABIR4",
        "format_version": 1,
        "source_weights_sha256": sha256(args.weights),
        "source_tokenizer_sha256": sha256(args.tokenizer),
        "output_sha256": sha256(args.output),
        "output_bytes": header.image_size,
        "payload_crc32": f"{header.payload_crc32:08x}",
        "config": output_config,
        "stored_tensor_count": header.tensor_count,
        "quantized_tensors": stats,
        "omitted_upstream_buffers": config["n_layer"] * config["n_head"],
        "note": "Causal tril buffers are regenerated algorithmically by the runtime.",
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "output_sha256", "output_bytes", "payload_crc32", "stored_tensor_count",
        "omitted_upstream_buffers")}, indent=2))


if __name__ == "__main__":
    main()

