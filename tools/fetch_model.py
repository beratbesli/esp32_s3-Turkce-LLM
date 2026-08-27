#!/usr/bin/env python3
"""Download only the two upstream files required to reproduce the deployment image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="jetbabareal/Sabir-20M")
    parser.add_argument("--revision", default="main", help="Use a commit SHA for release reproducibility")
    parser.add_argument("--output", type=Path, default=Path("artifacts/source"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    resolved_revision = HfApi().model_info(args.repo, revision=args.revision).sha
    files = {}
    for name in ("config.json", "model.safetensors", "tokenizer.model"):
        downloaded = Path(hf_hub_download(args.repo, name, revision=resolved_revision, local_dir=args.output))
        files[name] = {"bytes": downloaded.stat().st_size, "sha256": sha256(downloaded)}
    manifest = {"repo": args.repo, "requested_revision": args.revision,
                "resolved_revision": resolved_revision, "files": files}
    (args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
