"""Convert and exercise the small tokenizer asset used by the device runtime."""

from __future__ import annotations

import dataclasses
import re
import struct
import unicodedata
from pathlib import Path

from sentencepiece import sentencepiece_model_pb2

TOKENIZER_MAGIC = b"SBTOK1\0\0"
TOKENIZER_HEADER = struct.Struct("<8s10I")
TOKENIZER_ENTRY = struct.Struct("<IHBBf")
TOKENIZER_VERSION = 1
FLAG_ADD_DUMMY_PREFIX = 1
FLAG_REMOVE_EXTRA_WHITESPACE = 2
FLAG_ESCAPE_WHITESPACE = 4


@dataclasses.dataclass(frozen=True)
class Piece:
    text: str
    score: float
    kind: int


@dataclasses.dataclass
class TokenizerAsset:
    pieces: list[Piece]
    unk_id: int
    bos_id: int
    eos_id: int
    pad_id: int
    flags: int
    normalizer_name: str

    @classmethod
    def from_sentencepiece(cls, path: Path) -> "TokenizerAsset":
        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(path.read_bytes())
        spec = proto.normalizer_spec
        flags = 0
        if spec.add_dummy_prefix:
            flags |= FLAG_ADD_DUMMY_PREFIX
        if spec.remove_extra_whitespaces:
            flags |= FLAG_REMOVE_EXTRA_WHITESPACE
        if spec.escape_whitespaces:
            flags |= FLAG_ESCAPE_WHITESPACE
        return cls(
            [Piece(item.piece, item.score, item.type) for item in proto.pieces],
            proto.trainer_spec.unk_id, proto.trainer_spec.bos_id,
            proto.trainer_spec.eos_id, proto.trainer_spec.pad_id,
            flags, spec.name,
        )

    def encode_portable(self, text: str) -> list[int]:
        """Python mirror of the firmware BPE path for ordinary normalized Turkish text."""
        # nmt_nfkc is approximated with host NFKC here. Firmware documents its smaller normalizer.
        text = unicodedata.normalize("NFKC", text)
        if self.flags & FLAG_REMOVE_EXTRA_WHITESPACE:
            text = re.sub(r"\s+", " ", text).strip()
        if text and self.flags & FLAG_ADD_DUMMY_PREFIX:
            text = " " + text
        if self.flags & FLAG_ESCAPE_WHITESPACE:
            text = text.replace(" ", "▁")
        user_defined = sorted(
            ((piece.text, index) for index, piece in enumerate(self.pieces) if piece.kind == 4),
            key=lambda item: len(item[0]), reverse=True,
        )
        symbols: list[tuple[str, int | None]] = []
        offset = 0
        while offset < len(text):
            match = next(((value, idx) for value, idx in user_defined if text.startswith(value, offset)), None)
            if match:
                symbols.append(match)
                offset += len(match[0])
            else:
                symbols.append((text[offset], None))
                offset += 1
        index_by_piece = {piece.text: index for index, piece in enumerate(self.pieces)}
        while len(symbols) > 1:
            best_at = -1
            best_id = -1
            best_score = float("-inf")
            for index in range(len(symbols) - 1):
                candidate = symbols[index][0] + symbols[index + 1][0]
                piece_id = index_by_piece.get(candidate)
                if piece_id is None:
                    continue
                score = self.pieces[piece_id].score
                if score > best_score:
                    best_at, best_id, best_score = index, piece_id, score
            if best_at < 0:
                break
            merged = symbols[best_at][0] + symbols[best_at + 1][0]
            symbols[best_at:best_at + 2] = [(merged, best_id)]
        result: list[int] = []
        for text_piece, known_id in symbols:
            result.append(known_id if known_id is not None else index_by_piece.get(text_piece, self.unk_id))
        return result

    def to_bytes(self) -> bytes:
        strings = bytearray()
        entries = bytearray()
        for piece in self.pieces:
            encoded = piece.text.encode("utf-8")
            if len(encoded) > 0xFFFF:
                raise ValueError("tokenizer piece exceeds uint16 length")
            offset = len(strings)
            strings.extend(encoded)
            entries.extend(TOKENIZER_ENTRY.pack(offset, len(encoded), piece.kind, 0, piece.score))
        entries_offset = TOKENIZER_HEADER.size
        strings_offset = entries_offset + len(entries)
        header = TOKENIZER_HEADER.pack(
            TOKENIZER_MAGIC, TOKENIZER_VERSION, len(self.pieces), self.unk_id,
            self.bos_id, self.eos_id, self.pad_id, self.flags,
            entries_offset, strings_offset, len(strings),
        )
        return header + entries + strings


def convert_tokenizer(source: Path, destination: Path) -> TokenizerAsset:
    asset = TokenizerAsset.from_sentencepiece(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(asset.to_bytes())
    return asset

