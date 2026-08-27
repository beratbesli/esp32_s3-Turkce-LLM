#!/usr/bin/env python3
"""Inspect Sabir safetensors without constructing the PyTorch model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from safetensors import safe_open


EXPECTED = {"n_layer": 8, "n_embd": 384, "n_head": 6, "block_size": 256, "vocab_size": 8000}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    mismatch = {key: (EXPECTED[key], config.get(key)) for key in EXPECTED if config.get(key) != EXPECTED[key]}
    if mismatch:
        raise SystemExit(f"unexpected architecture: {mismatch}")
    total = trainable = masks = 0
    rows = []
    with safe_open(args.weights, framework="np", device="cpu") as source:
        for name in source.keys():
            tensor = source.get_tensor(name)
            count = math.prod(tensor.shape)
            total += count
            if name.endswith(".tril"):
                masks += count
            else:
                trainable += count
            rows.append((name, str(tensor.dtype), list(tensor.shape), count))
    report = {"config": config, "tensor_count": len(rows), "stored_elements": total,
              "causal_mask_buffer_elements": masks, "learned_parameters": trainable,
              "tensors": rows}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

