"""Exact and near-duplicate detection (Session 4, sections 6-7).

Exact duplicates: sha256 of the cleaned text.

Near duplicates follow the three moves from the lesson:
  1. shingles   - break each document into overlapping word n-grams
  2. MinHash    - replace the shingle set with a fixed-length signature of
                  minimum hash values; P(slot match) == Jaccard(A, B)
  3. LSH        - split signatures into b bands of r rows and only compare
                  documents that collide in at least one band

LSH only proposes candidate pairs. Every candidate pair is then checked
against the true Jaccard of the two shingle sets, so the final threshold is
exact and the LSH settings only affect recall (and run time).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np
import xxhash

MERSENNE_P = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint64(0xFFFFFFFF)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def exact_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def shingles(text: str, k: int) -> set[str]:
    """Word k-grams. Documents shorter than k words become one shingle."""
    w = words(text)
    if not w:
        return set()
    if len(w) < k:
        return {" ".join(w)}
    return {" ".join(w[i : i + k]) for i in range(len(w) - k + 1)}


def shingle_hashes(sh: set[str]) -> np.ndarray:
    """Sorted unique 32-bit hashes of the shingles. Stored as uint32 so the
    whole corpus fits in RAM (a Python string per shingle would not)."""
    return np.unique(
        np.fromiter((xxhash.xxh32_intdigest(s.encode("utf-8")) for s in sh), dtype=np.uint32, count=len(sh))
    )


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def jaccard_arrays(a: np.ndarray, b: np.ndarray) -> float:
    """True Jaccard on sorted unique shingle-hash arrays."""
    if a.size == 0 and b.size == 0:
        return 1.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


def lsh_threshold(bands: int, rows: int) -> float:
    """Approximate similarity where the S-curve crosses 0.5: (1/b)^(1/r)."""
    return (1.0 / bands) ** (1.0 / rows)


def lsh_catch_probability(s: float, bands: int, rows: int) -> float:
    """Probability that a pair with Jaccard s collides in at least one band."""
    return 1.0 - (1.0 - s**rows) ** bands


class MinHasher:
    """Universal hashing h(x) = (a*x + b) mod p, p = 2^61-1, truncated to 32 bits.

    Fixed seed, so signatures are reproducible run to run.
    """

    def __init__(self, num_perm: int, seed: int = 1):
        rng = np.random.RandomState(seed)
        # a, b < 2^31 and x < 2^32 keeps a*x + b inside uint64
        self.a = rng.randint(1, 1 << 31, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.b = rng.randint(0, 1 << 31, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.num_perm = num_perm

    def signature(self, hashes: np.ndarray) -> np.ndarray:
        if hashes.size == 0:
            return np.full(self.num_perm, MAX_HASH, dtype=np.uint64)
        hashes = hashes.astype(np.uint64, copy=False)
        sig = np.full(self.num_perm, MAX_HASH, dtype=np.uint64)
        # chunk long documents to bound memory (num_perm x chunk matrix)
        for start in range(0, hashes.size, 4096):
            h = hashes[start : start + 4096]
            phv = ((self.a[:, None] * h[None, :] + self.b[:, None]) % MERSENNE_P) & MAX_HASH
            np.minimum(sig, phv.min(axis=1), out=sig)
        return sig


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


@dataclass
class NearDupResult:
    candidate_pairs: int = 0
    verified_pairs: list = field(default_factory=list)  # (i, j, true_jaccard, est)
    rejected_pairs: list = field(default_factory=list)  # candidates below threshold
    clusters: list = field(default_factory=list)  # list of sorted index lists, size >= 2


def lsh_candidates(signatures: np.ndarray, bands: int, rows: int) -> set[tuple[int, int]]:
    n, perm = signatures.shape
    assert bands * rows <= perm, "bands * rows must not exceed num_perm"
    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        block = signatures[:, band * rows : (band + 1) * rows]
        for i in range(n):
            buckets.setdefault(block[i].tobytes(), []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > 2000:
                # a giant bucket means one template copied thousands of times;
                # link each member to the first instead of all O(n^2) pairs.
                # Each link is still verified with true Jaccard below.
                head = members[0]
                pairs.update((head, m) for m in members[1:])
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return pairs


def greedy_keep(n: int, verified_pairs) -> dict[int, tuple[int, float]]:
    """Walk documents in order. A document is removed only if it has a verified
    near-duplicate (true Jaccard >= threshold) among the documents already KEPT.
    Returns {removed_index: (kept_index, jaccard)}.

    Unlike union-find this never removes a document just because it is similar
    to another removed document (no chaining A~B~C where A and C differ)."""
    nbrs: dict[int, list[tuple[int, float]]] = {}
    for i, j, tj, _ in verified_pairs:
        a, b = (i, j) if i < j else (j, i)
        nbrs.setdefault(b, []).append((a, tj))
    kept: set[int] = set()
    removed: dict[int, tuple[int, float]] = {}
    for d in range(n):
        best = None
        for a, tj in nbrs.get(d, ()):
            if a in kept and (best is None or tj > best[1] or (tj == best[1] and a < best[0])):
                best = (a, tj)
        if best is None:
            kept.add(d)
        else:
            removed[d] = best
    return removed


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # smaller index becomes the root, so the kept doc is deterministic
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def near_duplicates(
    shingle_sets: list[np.ndarray],
    signatures: np.ndarray,
    bands: int,
    rows: int,
    threshold: float,
) -> NearDupResult:
    res = NearDupResult()
    cands = lsh_candidates(signatures, bands, rows)
    res.candidate_pairs = len(cands)
    uf = UnionFind(len(shingle_sets))
    for i, j in sorted(cands):
        tj = jaccard_arrays(shingle_sets[i], shingle_sets[j])
        est = estimate_jaccard(signatures[i], signatures[j])
        if tj >= threshold:
            res.verified_pairs.append((i, j, tj, est))
            uf.union(i, j)
        else:
            res.rejected_pairs.append((i, j, tj, est))
    groups: dict[int, list[int]] = {}
    for i, j, _, _ in res.verified_pairs:
        for x in (i, j):
            groups.setdefault(uf.find(x), [])
    for x in range(len(shingle_sets)):
        r = uf.find(x)
        if r in groups:
            groups[r].append(x)
    res.clusters = [sorted(set(v)) for v in groups.values() if len(set(v)) >= 2]
    res.clusters.sort(key=lambda c: c[0])
    return res
