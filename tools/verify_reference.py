#!/usr/bin/env python3
"""Compare exported Q4 logits with the exact upstream FP32 NumPy reference."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import sentencepiece as spm

from sabir_tools.reference import Q4Image, UpstreamImage, transformer_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="Kullanıcı: Nasılsın?\nModel: ")
    parser.add_argument("--golden-output", type=Path,
                        help="Optional binary Q4 logits fixture for the C++ host verifier")
    args = parser.parse_args()
    tokenizer = spm.SentencePieceProcessor(model_proto=args.tokenizer.read_bytes())
    ids = tokenizer.encode(args.prompt)
    if len(ids) > 32:
        raise SystemExit("verification prompt should stay short")
    quantized = Q4Image(args.image)
    q_logits = transformer_forward(ids, quantized.get)[-1]
    with UpstreamImage(args.weights) as upstream:
        fp_logits = transformer_forward(ids, upstream.get)[-1]
    difference = q_logits - fp_logits
    fp_top = np.argsort(fp_logits)[-10:][::-1]
    q_top = np.argsort(q_logits)[-10:][::-1]
    cosine = float(np.dot(q_logits, fp_logits) /
                   (np.linalg.norm(q_logits) * np.linalg.norm(fp_logits)))
    report = {
        "prompt": args.prompt,
        "token_ids": ids,
        "fp32_top10_ids": fp_top.tolist(),
        "q4_top10_ids": q_top.tolist(),
        "top10_overlap": len(set(fp_top.tolist()) & set(q_top.tolist())),
        "logit_rmse": float(np.sqrt(np.mean(difference * difference))),
        "logit_max_abs_error": float(np.max(np.abs(difference))),
        "logit_cosine_similarity": cosine,
    }
    if args.golden_output:
        args.golden_output.parent.mkdir(parents=True, exist_ok=True)
        with args.golden_output.open("wb") as stream:
            stream.write(struct.pack("<II", len(ids), len(q_logits)))
            stream.write(np.asarray(ids, dtype="<u2").tobytes())
            stream.write(np.asarray(q_logits, dtype="<f4").tobytes())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not np.isfinite(q_logits).all() or cosine < 0.80:
        raise SystemExit("Q4 reference check failed")


if __name__ == "__main__":
    main()
