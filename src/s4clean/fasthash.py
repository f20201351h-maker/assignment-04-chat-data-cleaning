"""Fast word n-gram hashing shared by dedup and decontamination.

Each word is hashed once (xxh64), then every n-gram hash is built with a
polynomial rolling combination in numpy (uint64 arithmetic wraps mod 2^64):

    H(w_i..w_{i+n-1}) = sum_j h(w_{i+j}) * P^(n-1-j)

Same words in the same order -> same hash, so it is a drop-in replacement for
hashing the joined n-gram string, just without building millions of strings.
"""
from __future__ import annotations

import re

import numpy as np
import xxhash

_WORD_RE = re.compile(r"\w+", re.UNICODE)
P = np.uint64(1099511628211)  # FNV prime, odd


def words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def word_hashes(text: str) -> np.ndarray:
    w = words(text)
    return np.fromiter((xxhash.xxh64_intdigest(x.encode("utf-8")) for x in w), dtype=np.uint64, count=len(w))


def ngram_hashes(wh: np.ndarray, n: int) -> np.ndarray:
    """All n-gram hashes of a word-hash array (empty if fewer than n words)."""
    L = wh.size - n + 1
    if L <= 0:
        return np.empty(0, dtype=np.uint64)
    acc = np.zeros(L, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            acc = acc * P + wh[j : j + L]
    return acc


def shingle_set_u32(text: str, k: int) -> np.ndarray:
    """Sorted unique 32-bit shingle hashes for MinHash (short docs -> one shingle)."""
    wh = word_hashes(text)
    if wh.size == 0:
        return np.empty(0, dtype=np.uint32)
    g = ngram_hashes(wh, k) if wh.size >= k else ngram_hashes(wh, wh.size)
    return np.unique((g >> np.uint64(32)).astype(np.uint32))
