#!/usr/bin/env python3
"""Validate and flash only the model partition via esptool."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sabir_tools.model_format import read_image

MODEL_OFFSET = 0x310000
MODEL_PARTITION_BYTES = 0xCF0000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    header, _ = read_image(args.image)
    if header.image_size > MODEL_PARTITION_BYTES:
        raise SystemExit(f"image is {header.image_size} bytes; partition allows {MODEL_PARTITION_BYTES}")
    command = [sys.executable, "-m", "esptool", "--chip", "esp32s3", "--port", args.port,
               "--baud", str(args.baud), "write_flash", "--flash_size", "16MB",
               hex(MODEL_OFFSET), str(args.image)]
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

