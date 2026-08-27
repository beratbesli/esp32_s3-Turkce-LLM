"""NumPy reference forwards for upstream FP32 and exported Q4 Sabir weights."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
from safetensors import safe_open

from .model_format import DTYPE_F32, DTYPE_Q4, dequantize_q4, read_image


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return ((x - x.mean(axis=-1, keepdims=True)) /
            np.sqrt(x.var(axis=-1, keepdims=True) + 1.0e-5)) * weight + bias


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def transformer_forward(ids: list[int], get: Callable[[str], np.ndarray], layers: int = 8,
                        heads: int = 6, embedding: int = 384) -> np.ndarray:
    tokens = np.asarray(ids, dtype=np.int64)
    x = get("tok_emb.weight")[tokens] + get("pos_emb.weight")[np.arange(len(tokens))]
    head_dim = embedding // heads
    causal = np.triu(np.full((len(tokens), len(tokens)), -np.inf, dtype=np.float32), 1)
    for layer in range(layers):
        norm = layer_norm(x, get(f"b{layer}.ln1.weight"), get(f"b{layer}.ln1.bias"))
        qkv = norm @ get(f"b{layer}.qkv.weight").T
        q, k, v = np.split(qkv, 3, axis=-1)
        outputs = []
        for head in range(heads):
            begin = head * head_dim
            end = begin + head_dim
            # Upstream Head.forward scales by C=n_embd, not by head_size.
            scores = q[:, begin:end] @ k[:, begin:end].T / math.sqrt(embedding)
            outputs.append(softmax(scores + causal) @ v[:, begin:end])
        attention = np.concatenate(outputs, axis=-1)
        x = x + attention @ get(f"b{layer}.attn.weight").T + get(f"b{layer}.attn.bias")
        norm = layer_norm(x, get(f"b{layer}.ln2.weight"), get(f"b{layer}.ln2.bias"))
        hidden = np.maximum(0.0, norm @ get(f"b{layer}.ff1.weight").T + get(f"b{layer}.ff1.bias"))
        x = x + hidden @ get(f"b{layer}.ff2.weight").T + get(f"b{layer}.ff2.bias")
    x = layer_norm(x, get("ln_f.weight"), get("ln_f.bias"))
    return x @ get("lm_head.weight").T + get("lm_head.bias")


class Q4Image:
    def __init__(self, path: Path):
        self.path = path
        self.header, entries = read_image(path)
        self.entries = {entry.name: entry for entry in entries}
        self.data = path.read_bytes()
        self.cache: dict[str, np.ndarray] = {}

    def get(self, name: str) -> np.ndarray:
        if name in self.cache:
            return self.cache[name]
        entry = self.entries[name]
        raw = self.data[entry.data_offset:entry.data_offset + entry.data_bytes]
        if entry.dtype == DTYPE_F32:
            value = np.frombuffer(raw, dtype="<f4").reshape(entry.shape)
        elif entry.dtype == DTYPE_Q4:
            aux = self.data[entry.aux_offset:entry.aux_offset + entry.aux_bytes]
            value = dequantize_q4(raw, aux, entry.shape, self.header.group_size)
        else:
            raise TypeError(f"{name} is not numeric")
        self.cache[name] = value
        return value


class UpstreamImage:
    def __init__(self, path: Path):
        self.source = safe_open(path, framework="np", device="cpu")
        self.cache: dict[str, np.ndarray] = {}

    def __enter__(self) -> "UpstreamImage":
        self.source.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self.source.__exit__(*args)

    def get(self, name: str) -> np.ndarray:
        if name in self.cache:
            return self.cache[name]
        if name == "tok_emb.weight":
            source_name = "token_embedding_table.weight"
        elif name == "pos_emb.weight":
            source_name = "position_embedding_table.weight"
        elif name.startswith("b"):
            layer, suffix = name.split(".", 1)
            number = int(layer[1:])
            prefix = f"blocks.{number}"
            mapping = {
                "ln1.weight": f"{prefix}.ln1.weight", "ln1.bias": f"{prefix}.ln1.bias",
                "ln2.weight": f"{prefix}.ln2.weight", "ln2.bias": f"{prefix}.ln2.bias",
                "attn.weight": f"{prefix}.sa.proj.weight", "attn.bias": f"{prefix}.sa.proj.bias",
                "ff1.weight": f"{prefix}.ffwd.net.0.weight", "ff1.bias": f"{prefix}.ffwd.net.0.bias",
                "ff2.weight": f"{prefix}.ffwd.net.3.weight", "ff2.bias": f"{prefix}.ffwd.net.3.bias",
            }
            if suffix == "qkv.weight":
                parts = []
                for projection in ("query", "key", "value"):
                    parts.append(np.concatenate([
                        self.source.get_tensor(f"{prefix}.sa.heads.{head}.{projection}.weight")
                        for head in range(6)
                    ]))
                value = np.concatenate(parts).astype(np.float32, copy=False)
                self.cache[name] = value
                return value
            source_name = mapping[suffix]
        else:
            source_name = name
        value = self.source.get_tensor(source_name).astype(np.float32, copy=False)
        self.cache[name] = value
        return value

