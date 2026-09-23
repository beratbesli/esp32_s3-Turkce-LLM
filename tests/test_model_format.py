import tempfile
import struct
from pathlib import Path

import numpy as np
import pytest

from sabir_tools.model_format import (
    DTYPE_BLOB,
    DTYPE_F32,
    DTYPE_Q4,
    ENTRY,
    HEADER,
    TensorRecord,
    quantize_q4,
    read_image,
    write_image,
)


def test_image_layout_crc_and_directory():
    source = np.arange(32, dtype=np.float32).reshape(4, 8) / 10
    codes, scales, _ = quantize_q4(source, 8)
    records = [
        TensorRecord("matrix", DTYPE_Q4, source.shape, codes, scales),
        TensorRecord("bias", DTYPE_F32, (4,), np.zeros(4, dtype="<f4").tobytes()),
        TensorRecord("tokenizer", DTYPE_BLOB, (3,), b"tok"),
    ]
    Path(".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=".cache") as temporary:
        path = Path(temporary) / "tiny.sabir"
        config = dict(n_layer=8, n_embd=384, n_head=6, block_size=256,
                      vocab_size=8000, group_size=8)
        written = write_image(path, config, records)
        header, entries = read_image(path)
        assert header.image_size == path.stat().st_size == written.image_size
        assert [entry.name for entry in entries] == ["matrix", "bias", "tokenizer"]

        damaged = bytearray(path.read_bytes())
        damaged[-1] ^= 1
        path.write_bytes(damaged)
        with pytest.raises(ValueError, match="CRC"):
            read_image(path)


def test_reader_rejects_truncated_and_corrupt_offsets(tmp_path):
    path = tmp_path / "tiny.sabir"
    config = dict(n_layer=8, n_embd=384, n_head=6, block_size=256,
                  vocab_size=8000, group_size=64)
    write_image(path, config, [TensorRecord("unknown", DTYPE_BLOB, (1,), b"x")])
    original = path.read_bytes()

    for damaged in (original[:HEADER.size - 1], original[:-1]):
        path.write_bytes(damaged)
        with pytest.raises(ValueError, match="truncated|offsets"):
            read_image(path)

    damaged = bytearray(original)
    struct.pack_into("<Q", damaged, 64, len(original) + 64)  # data_offset
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="offsets"):
        read_image(path)

    damaged = bytearray(original)
    struct.pack_into("<Q", damaged, 72, HEADER.size)  # image_size
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="offsets"):
        read_image(path)

    damaged = bytearray(original)
    data_field = HEADER.size + struct.calcsize("<28sBBH4I")
    struct.pack_into("<Q", damaged, data_field, len(original) + 64)
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="tensor"):
        read_image(path)

    damaged = bytearray(original)
    aux_field = HEADER.size + struct.calcsize("<28sBBH4IQQ")
    struct.pack_into("<Q", damaged, aux_field, len(original) + 64)
    struct.pack_into("<I", damaged, HEADER.size + ENTRY.size - 8, 1)
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="tensor"):
        read_image(path)
