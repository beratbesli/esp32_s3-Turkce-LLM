#!/usr/bin/env python3
"""Emit any deployment image as a const C byte array for integration or host fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", default="sabir_model_image")
    parser.add_argument("--section", default=".rodata.sabir_model")
    args = parser.parse_args()
    if not args.symbol.replace("_", "a").isalnum():
        raise SystemExit("symbol must be a C identifier")
    data = args.input.read_bytes()
    lines = [
        "/* Generated file; do not commit large model arrays. */",
        "#include <stddef.h>",
        "#include <stdint.h>",
        f'const uint8_t {args.symbol}[] __attribute__((aligned(64), section("{args.section}"))) = {{',
    ]
    for offset in range(0, len(data), 16):
        lines.append("  " + ", ".join(f"0x{byte:02x}" for byte in data[offset:offset + 16]) + ",")
    lines.extend(["};", f"const size_t {args.symbol}_size = sizeof({args.symbol});", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="ascii")
    print(f"wrote {len(data)} bytes as {args.symbol} to {args.output}")


if __name__ == "__main__":
    main()

