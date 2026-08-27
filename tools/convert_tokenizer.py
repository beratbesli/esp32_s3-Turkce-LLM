#!/usr/bin/env python3
"""Convert SentencePiece protobuf to the compact tokenizer asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from sabir_tools.tokenizer import convert_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asset = convert_tokenizer(args.input, args.output)
    print(f"pieces={len(asset.pieces)} normalizer={asset.normalizer_name} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()

