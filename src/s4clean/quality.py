"""Quality filtering, heuristic layer (Session 4, section 5 + widget 4).

The nine rules and thresholds are copied from widget 4's analyze():

  meanlen    mean word length (code points)            3 <= x <= 10
  symratio   symbols #@^*~<>{}[]|\\_=+`$%& per word      < 0.10
  termpunc   lines ending in . ! ? or danda             >= 0.30
  duplines   repeated (lower-cased) lines / lines       < 0.30
  repeat     top character-bigram share                 < 0.20
  stopwords  distinct Gopher stop words present         >= 2
  bullets    lines starting with a bullet               < 0.90
  ellipsis   lines ending in ... or the ellipsis char   < 0.30
  wordcount  words                                      50 .. 100000

Widget 4 marks `stopwords` and `wordcount` as English-biased: they fail good
Telugu text. The same thing happens to code and maths, which is measured
separately in the pipeline rather than assumed.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

STOPWORDS = ("the", "be", "to", "of", "and", "that", "have", "with")  # Gopher set, widget 4
SYM_RE = re.compile(r"[#@^*~<>{}\[\]|\\_=+`$%&]")
TERM_RE = re.compile(r"[.!?।]$")
BULLET_RE = re.compile(r"^[-*•▪‣·–—]")
ELLIPSIS_RE = re.compile(r"(\.\.\.|…)\s*$")
_HAS_ALNUM = re.compile(r"[^\W_]", re.UNICODE)

RULES = ("meanlen", "symratio", "termpunc", "duplines", "repeat", "stopwords", "bullets", "ellipsis", "wordcount")
EN_BIASED = frozenset({"stopwords", "wordcount"})


def analyze(text: str) -> dict:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    words = [w for w in text.split() if _HAS_ALNUM.search(w)]
    nw = len(words)
    m: dict = {"words": nw, "lines": len(lines)}
    m["meanlen"] = sum(len(w) for w in words) / nw if nw else 0.0
    m["symratio"] = len(SYM_RE.findall(text)) / nw if nw else 1.0
    m["termpunc"] = sum(bool(TERM_RE.search(ln)) for ln in lines) / len(lines) if lines else 0.0
    if lines:
        cnt = Counter(ln.lower() for ln in lines)
        m["duplines"] = sum(c - 1 for c in cnt.values() if c > 1) / len(lines)
    else:
        m["duplines"] = 0.0
    compact = re.sub(r"\s+", "", text)
    if len(compact) > 1:
        # most frequent character bigram, counted in numpy over code points
        cp = np.frombuffer(compact.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
        bg = cp[:-1] * np.uint64(0x110000) + cp[1:]
        _, counts = np.unique(bg, return_counts=True)
        m["repeat"] = float(counts.max()) / (len(compact) - 1)
    else:
        m["repeat"] = 0.0
    lw = {re.sub(r"[^\w]", "", w.lower()) for w in words}
    m["stopwords"] = sum(s in lw for s in STOPWORDS)
    m["bullets"] = sum(bool(BULLET_RE.match(ln)) for ln in lines) / len(lines) if lines else 0.0
    m["ellipsis"] = sum(bool(ELLIPSIS_RE.search(ln)) for ln in lines) / len(lines) if lines else 0.0
    return m


# ---- applied rule for chat replies: runaway / looping output -----------------
# Measured on the first runs: widget 4's character-bigram and duplicate-line
# rules flagged markdown tables, guitar tabs, binary strings and meal plans far
# more often than real loops. This rule looks for the loop itself.
_CODE_RE = re.compile(r"```.*?(?:```|\Z)|`[^`\n]+`", re.S)
_RULE_OR_TABLE = re.compile(r"^\s*(?:\|.*|[\s:\-=_*~+.#|]*)$")
_LETTER_LOOP_RE = re.compile(r"([^\W\d_])\1{29,}")
_HAS_LETTER = re.compile(r"[^\W\d_]")


def _loop_re(min_repeats: int) -> re.Pattern:
    return re.compile(r"([^\n]{2,10}?)\1{%d,}" % (min_repeats - 1))


def looping_unit(reply: str, min_repeats: int = 15) -> str | None:
    """Return the repeated unit if the reply (code blocks, table rows and rule
    lines removed) contains a short unit repeated min_repeats+ times in a row.
    The unit must contain a letter, or digits mixed with punctuation (",000");
    pure digit runs like 0.000000000001 are not loops."""
    body = "\n".join(ln for ln in _CODE_RE.sub(" ", reply or "").split("\n") if not _RULE_OR_TABLE.match(ln))
    for m in _loop_re(min_repeats).finditer(body):
        u = m.group(1)
        if _HAS_LETTER.search(u):
            return u
        if any(ch.isdigit() for ch in u) and any(not ch.isdigit() and not ch.isspace() for ch in u):
            return u
    m = _LETTER_LOOP_RE.search(body)
    return m.group(1) if m else None


def verdicts(m: dict) -> dict[str, bool]:
    """True = pass. Same thresholds as widget 4."""
    return {
        "meanlen": 3 <= m["meanlen"] <= 10,
        "symratio": m["symratio"] < 0.10,
        "termpunc": m["termpunc"] >= 0.30,
        "duplines": m["duplines"] < 0.30,
        "repeat": m["repeat"] < 0.20,
        "stopwords": m["stopwords"] >= 2,
        "bullets": m["bullets"] < 0.90,
        "ellipsis": m["ellipsis"] < 0.30,
        "wordcount": 50 <= m["words"] <= 100000,
    }
