"""Quality layer 2: a cheap classifier that imitates LLM labels (Session 4, section 5).

Recipe from the lesson: a large model labels a sample, a light model learns
those labels and scores everything. Here the light model is logistic
regression on hashed character 2-4-grams (works for any script, no
tokenizer needed), trained with plain numpy gradient descent.
"""
from __future__ import annotations

import numpy as np

DIM = 1 << 18
_P = np.uint64(1000003)
_MAX_CHARS = 6000


def features(text: str, dim: int = DIM) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalised counts of hashed character 2/3/4-grams, as (indices, values)."""
    t = " ".join((text or "").lower().split())[:_MAX_CHARS]
    cp = np.frombuffer(t.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    parts = []
    with np.errstate(over="ignore"):
        for n in (2, 3, 4):
            L = cp.size - n + 1
            if L <= 0:
                continue
            h = np.zeros(L, dtype=np.uint64)
            for j in range(n):
                h = h * _P + cp[j : j + L]
            h = h + np.uint64(n)  # keep n-gram sizes apart
            parts.append((h % np.uint64(dim)).astype(np.int64))
    if not parts:
        return np.empty(0, dtype=np.int64), np.empty(0)
    idx, cnt = np.unique(np.concatenate(parts), return_counts=True)
    val = cnt.astype(np.float64)
    return idx, val / np.sqrt((val * val).sum())


def to_matrix(feats):
    rows = np.concatenate([np.full(f[0].size, i) for i, f in enumerate(feats)]) if feats else np.empty(0, dtype=np.int64)
    cols = np.concatenate([f[0] for f in feats]) if feats else np.empty(0, dtype=np.int64)
    vals = np.concatenate([f[1] for f in feats]) if feats else np.empty(0)
    return rows, cols, vals


class LogReg:
    def __init__(self, dim: int = DIM, l2: float = 1e-4, lr: float = 0.5, epochs: int = 300):
        self.dim, self.l2, self.lr, self.epochs = dim, l2, lr, epochs
        self.w = np.zeros(dim)
        self.b = 0.0

    def _dot(self, R, C, V, n):
        z = np.zeros(n)
        np.add.at(z, R, V * self.w[C])
        return z + self.b

    def fit(self, feats, y):
        y = np.asarray(y, dtype=np.float64)
        R, C, V = to_matrix(feats)
        n = len(feats)
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-self._dot(R, C, V, n)))
            g = p - y
            gw = np.zeros(self.dim)
            np.add.at(gw, C, V * g[R])
            self.w -= self.lr * (gw / n + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def predict_proba(self, feats):
        R, C, V = to_matrix(feats)
        return 1 / (1 + np.exp(-self._dot(R, C, V, len(feats))))


def auc(y, p) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))
