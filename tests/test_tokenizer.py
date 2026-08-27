import os
from pathlib import Path

import pytest
import sentencepiece as spm

from sabir_tools.tokenizer import TokenizerAsset


MODEL = Path(os.environ.get("SABIR_TOKENIZER", "artifacts/source/tokenizer.model"))


@pytest.mark.skipif(not MODEL.exists(), reason="download tokenizer with tools/fetch_model.py")
@pytest.mark.parametrize("text", [
    "Merhaba",
    "Nasılsın?",
    "Kullanıcı: Merhaba\nModel: ",
    "İstanbul güzel bir şehir.",
    "  çok   sabırlı  ",
    "En sevdiğin renk ne?",
])
def test_portable_tokenizer_ids_match_sentencepiece(text):
    # sentencepiece's Windows file API cannot open every non-ASCII checkout path.
    upstream = spm.SentencePieceProcessor(model_proto=MODEL.read_bytes())
    portable = TokenizerAsset.from_sentencepiece(MODEL)
    assert portable.encode_portable(text) == upstream.encode(text)


@pytest.mark.skipif(not MODEL.exists(), reason="download tokenizer with tools/fetch_model.py")
def test_tokenizer_asset_is_materially_smaller_than_protobuf():
    asset = TokenizerAsset.from_sentencepiece(MODEL).to_bytes()
    assert asset.startswith(b"SBTOK1\0\0")
    assert len(asset) < MODEL.stat().st_size
