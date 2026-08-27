# Sabir-20M architecture findings

Inspected upstream: `jetbabareal/Sabir-20M` on Hugging Face. Use
`python tools/inspect_model.py` to reproduce the tensor inventory after download.

## Confirmed configuration

| Property | Value |
|---|---:|
| Architecture | custom NanoGPT, decoder-only Transformer |
| Layers | 8 |
| Embedding width | 384 |
| Attention heads | 6 |
| Head width | 64 |
| FFN width | 1,536 |
| Trained context | 256 |
| Vocabulary | 8,000 |
| Activation | ReLU |
| Normalization | pre-residual LayerNorm, epsilon `1e-5` |
| Tokenizer | SentencePiece BPE, `nmt_nfkc`, custom symbols |
| Input/output embeddings | separate, not tied |

The source implementation computes attention scale from `C` after unpacking
`B, T, C = x.shape`; therefore it uses `1/sqrt(384)`, not the usual
`1/sqrt(64)`. The port intentionally preserves this behavior.

## Parameter accounting

The safetensors file contains 278 entries and 23,583,296 stored floating-point
elements. Forty-eight entries are identical 256x256 lower-triangular causal-mask
buffers, one per layer/head. Those buffers account for 3,145,728 elements and
are not learned parameters. The exact learned count is therefore 20,437,568,
which agrees with the model card's approximately 20.44M claim. The device image
omits the masks and enforces causality in the attention loop.

The model card currently marks the model MIT and describes roughly 1.5M tokens
of Turkish dialogues. It does not provide sufficient corpus provenance or
license detail to support a broader legal-compliance claim.

