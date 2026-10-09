"""Token counting with a real tokenizer.

Widget 9 / section 11: V4 estimated tokens as words x 1.3, which was wrong
for Indic by 2-10x. Here every count comes from actually tokenizing the text
with the Qwen3 byte-level BPE tokenizer (151,669 vocab), pinned to a fixed
Hugging Face revision so counts are reproducible.
"""
from __future__ import annotations

from functools import lru_cache

TOKENIZER_REPO = "Qwen/Qwen3-0.6B"
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@lru_cache(maxsize=1)
def get_tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION)
    return Tokenizer.from_file(path)


def count_tokens(texts: list[str], batch: int = 512) -> list[int]:
    tok = get_tokenizer()
    out: list[int] = []
    for i in range(0, len(texts), batch):
        enc = tok.encode_batch(texts[i : i + batch], add_special_tokens=False)
        out.extend(len(e.ids) for e in enc)
    return out
