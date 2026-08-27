"""Sabir ESP32 model-image format and group-wise symmetric int4 primitives."""

from __future__ import annotations

import binascii
import dataclasses
import math
import struct
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np

MAGIC = b"SABIR4\0\0"
VERSION = 1
HEADER = struct.Struct("<8s12I3Q")
ENTRY = struct.Struct("<28sBBH4IQQQII")
ALIGNMENT = 64

DTYPE_Q4 = 1
DTYPE_F32 = 2
DTYPE_BLOB = 3
FLAG_HAS_TOKENIZER = 1


@dataclasses.dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: int
    shape: tuple[int, ...]
    data: bytes
    aux: bytes = b""


@dataclasses.dataclass(frozen=True)
class ImageHeader:
    flags: int
    n_layer: int
    n_embd: int
    n_head: int
    block_size: int
    vocab_size: int
    group_size: int
    tensor_count: int
    payload_crc32: int
    directory_offset: int
    data_offset: int
    image_size: int


@dataclasses.dataclass(frozen=True)
class ImageTensor:
    name: str
    dtype: int
    shape: tuple[int, ...]
    data_offset: int
    data_bytes: int
    aux_offset: int
    aux_bytes: int


def align(value: int, boundary: int = ALIGNMENT) -> int:
    return (value + boundary - 1) & ~(boundary - 1)


def quantize_q4(array: np.ndarray, group_size: int = 64) -> tuple[bytes, bytes, np.ndarray]:
    """Quantize a 2-D matrix row-wise; return packed nibbles, fp16 scales and dequantized values."""
    w = np.asarray(array, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"q4 tensors must be 2-D, got {w.shape}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rows, cols = w.shape
    groups = math.ceil(cols / group_size)
    row_bytes = math.ceil(cols / 2)
    packed = np.zeros((rows, row_bytes), dtype=np.uint8)
    scales = np.empty((rows, groups), dtype=np.float16)
    dequant = np.empty_like(w)
    for row in range(rows):
        for group in range(groups):
            begin = group * group_size
            end = min(begin + group_size, cols)
            values = w[row, begin:end]
            peak = float(np.max(np.abs(values), initial=0.0))
            # Round the scale to the representation consumed on device before deriving codes.
            scale16 = np.float16(peak / 7.0 if peak else 1.0)
            scale = float(scale16)
            codes = np.clip(np.rint(values / scale), -7, 7).astype(np.int8)
            scales[row, group] = scale16
            dequant[row, begin:end] = codes.astype(np.float32) * scale
            unsigned = (codes.astype(np.int16) + 8).astype(np.uint8)
            for index, code in enumerate(unsigned, start=begin):
                shift = 4 if index & 1 else 0
                packed[row, index >> 1] |= int(code) << shift
    return packed.tobytes(), scales.astype("<f2", copy=False).tobytes(), dequant


def dequantize_q4(codes: bytes, scales: bytes, shape: tuple[int, int], group_size: int) -> np.ndarray:
    rows, cols = shape
    groups = math.ceil(cols / group_size)
    row_bytes = math.ceil(cols / 2)
    packed = np.frombuffer(codes, dtype=np.uint8).reshape(rows, row_bytes)
    scale_array = np.frombuffer(scales, dtype="<f2").astype(np.float32).reshape(rows, groups)
    out = np.empty((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            byte = packed[row, col >> 1]
            code = int(byte >> 4) if col & 1 else int(byte & 0x0F)
            out[row, col] = (code - 8) * scale_array[row, col // group_size]
    return out


def _write_padding(stream: BinaryIO, target: int) -> None:
    current = stream.tell()
    if current > target:
        raise ValueError("model-image layout overlap")
    stream.write(b"\0" * (target - current))


def write_image(path: Path, config: dict[str, int], records: Iterable[TensorRecord], flags: int = 0) -> ImageHeader:
    records = list(records)
    directory_offset = HEADER.size
    data_offset = align(directory_offset + len(records) * ENTRY.size)
    cursor = data_offset
    layouts: list[tuple[TensorRecord, int, int]] = []
    for record in records:
        if not record.name or len(record.name.encode("ascii")) >= 28:
            raise ValueError(f"tensor name must be non-empty ASCII shorter than 28 bytes: {record.name!r}")
        if not 1 <= len(record.shape) <= 4:
            raise ValueError(f"invalid shape for {record.name}: {record.shape}")
        data_at = cursor
        cursor = align(cursor + len(record.data))
        aux_at = cursor if record.aux else 0
        if record.aux:
            cursor = align(cursor + len(record.aux))
        layouts.append((record, data_at, aux_at))
    image_size = cursor
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb+") as stream:
        stream.write(b"\0" * HEADER.size)
        for record, data_at, aux_at in layouts:
            dims = (*record.shape, 0, 0, 0, 0)[:4]
            stream.write(ENTRY.pack(
                record.name.encode("ascii"), record.dtype, len(record.shape), 0, *dims,
                data_at, len(record.data), aux_at, len(record.aux), 0,
            ))
        _write_padding(stream, data_offset)
        for record, data_at, aux_at in layouts:
            _write_padding(stream, data_at)
            stream.write(record.data)
            if record.aux:
                _write_padding(stream, aux_at)
                stream.write(record.aux)
        _write_padding(stream, image_size)
        stream.seek(data_offset)
        crc = 0
        while chunk := stream.read(1024 * 1024):
            crc = binascii.crc32(chunk, crc)
        crc &= 0xFFFFFFFF
        stream.seek(0)
        stream.write(HEADER.pack(
            MAGIC, VERSION, HEADER.size, flags,
            config["n_layer"], config["n_embd"], config["n_head"],
            config["block_size"], config["vocab_size"], config["group_size"],
            len(records), ENTRY.size, crc,
            directory_offset, data_offset, image_size,
        ))
    return ImageHeader(flags, config["n_layer"], config["n_embd"], config["n_head"],
                       config["block_size"], config["vocab_size"], config["group_size"],
                       len(records), crc, directory_offset, data_offset, image_size)


def read_image(path: Path) -> tuple[ImageHeader, list[ImageTensor]]:
    with path.open("rb") as stream:
        values = HEADER.unpack(stream.read(HEADER.size))
        (magic, version, header_size, flags, n_layer, n_embd, n_head, block_size,
         vocab_size, group_size, count, entry_size, crc, directory_offset,
         data_offset, image_size) = values
        if magic != MAGIC or version != VERSION or header_size != HEADER.size or entry_size != ENTRY.size:
            raise ValueError("unsupported or corrupt Sabir model image")
        actual_size = path.stat().st_size
        if image_size > actual_size or directory_offset + count * ENTRY.size > data_offset:
            raise ValueError("model-image offsets exceed file")
        stream.seek(data_offset)
        computed = 0
        remaining = image_size - data_offset
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("truncated model image")
            computed = binascii.crc32(chunk, computed)
            remaining -= len(chunk)
        if computed & 0xFFFFFFFF != crc:
            raise ValueError("model-image payload CRC mismatch")
        tensors: list[ImageTensor] = []
        stream.seek(directory_offset)
        for _ in range(count):
            raw = ENTRY.unpack(stream.read(ENTRY.size))
            name = raw[0].split(b"\0", 1)[0].decode("ascii")
            dtype, ndim = raw[1], raw[2]
            shape = tuple(int(x) for x in raw[4:8][:ndim])
            data_at, data_bytes, aux_at, aux_bytes = raw[8:12]
            if data_at + data_bytes > image_size or (aux_bytes and aux_at + aux_bytes > image_size):
                raise ValueError(f"tensor {name} exceeds model image")
            tensors.append(ImageTensor(name, dtype, shape, data_at, data_bytes, aux_at, aux_bytes))
    header = ImageHeader(flags, n_layer, n_embd, n_head, block_size, vocab_size,
                         group_size, count, crc, directory_offset, data_offset, image_size)
    return header, tensors

