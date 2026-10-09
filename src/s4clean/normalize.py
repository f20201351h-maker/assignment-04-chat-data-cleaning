"""Text normalisation (Session 4, section 3 + widget 2).

One deterministic function, run on every text field before anything else:
  1. Unicode NFC
  2. HTML entities unescaped (outside fenced code blocks)
  3. invisible noise removed: zero-width space, BOM, word joiner, soft
     hyphen, bidi overrides/isolates and LRM/RLM, C0/C1 control characters,
     private-use characters, U+FFFD replacement characters (same set as the
     widget 2 CTRL_RE / BIDI_RE / PUA_RE / FFFD_RE)
  4. ZWJ / ZWNJ kept when they are doing real work: next to a Brahmic or
     Arabic-script letter, or inside an emoji ZWJ sequence. Only stray ones
     (e.g. inside a Latin word) are removed.
  5. whitespace: odd space characters -> ' ', runs of spaces collapsed in
     prose, trailing spaces dropped, 3+ blank lines -> 1 blank line.
     Inside ``` code fences and `inline code` only line endings and trailing
     spaces change, so indentation survives. (Widget 2's demo turns every
     whitespace run into one space; Rohan says at 62m-64m that code
     whitespace must not be collapsed, so this version is code-aware.)

Returns the cleaned text and a Counter of what was changed, so every stage
can report what it did.
"""
from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter

ZWSP = "\u200b"
ZWNJ = "\u200c"
ZWJ = "\u200d"
WORD_JOINER = "\u2060"
BOM = "\ufeff"
SOFT_HYPHEN = "\u00ad"
REPLACEMENT = "\ufffd"

# bidi embeddings / overrides / isolates. LRM/RLM (200E/200F) are included:
# this corpus has no RTL prose where they would be load-bearing.
BIDI = set("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200e\u200f\u061c")

ODD_SPACES = set("\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f")

# scripts where ZWJ/ZWNJ are part of correct spelling
_JOINER_SCRIPTS = (
    (0x0600, 0x06FF),  # Arabic (ZWNJ used in Persian/Urdu)
    (0x0900, 0x0DFF),  # Devanagari .. Sinhala (all Brahmic Indic blocks)
    (0x0E00, 0x0EFF),  # Thai, Lao
    (0x0F00, 0x0FFF),  # Tibetan
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0xA8E0, 0xA8FF),  # Devanagari extended
)

ENTITY_RE = re.compile(r"&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
FENCE_RE = re.compile(r"(```.*?(?:```|\Z)|`[^`\n]+`)", re.S)
MULTISPACE_RE = re.compile(r"(?<=\S) {2,}(?=\S)")
TRAILING_WS_RE = re.compile(r"[ \t]+$", re.M)
MANY_BLANKS_RE = re.compile(r"\n{3,}")


def _in_joiner_script(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _JOINER_SCRIPTS)


def _is_emoji(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    return (
        0x1F000 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
        or cp in (0x2640, 0x2642, 0x2695, 0x2696, 0x2708, 0x2764, 0xFE0F, 0x200D)
        or 0x1F3FB <= cp <= 0x1F3FF
    )


def _is_noise_char(ch: str) -> str | None:
    """Return a counter key if ch is always-noise, else None."""
    if ch == ZWSP:
        return "zero_width_space"
    if ch == BOM:
        return "bom"
    if ch == WORD_JOINER:
        return "word_joiner"
    if ch == SOFT_HYPHEN:
        return "soft_hyphen"
    if ch in BIDI:
        return "bidi_control"
    if ch == REPLACEMENT:
        return "replacement_char"
    cp = ord(ch)
    if (cp < 0x20 and ch not in "\t\n") or cp == 0x7F or 0x80 <= cp <= 0x9F:
        return "control_char"
    if 0xE000 <= cp <= 0xF8FF:
        return "private_use"
    return None


def _special_class() -> re.Pattern:
    """Every character _strip_invisible may act on, as one character class."""
    ranges = [(0x00, 0x08), (0x0B, 0x1F), (0x7F, 0x9F), (0xA0, 0xA0), (0xAD, 0xAD), (0x061C, 0x061C),
              (0x2000, 0x200F), (0x202A, 0x202F), (0x205F, 0x2060), (0x2066, 0x2069), (0xE000, 0xF8FF),
              (0xFEFF, 0xFEFF), (0xFFFD, 0xFFFD)]
    return re.compile("[" + "".join(re.escape(chr(a)) + "-" + re.escape(chr(b)) for a, b in ranges) + "]")


_SPECIAL_RE = _special_class()


def _strip_invisible(text: str, c: Counter) -> str:
    if not _SPECIAL_RE.search(text):
        return text  # fast path: nothing invisible or odd in this text
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        key = _is_noise_char(ch)
        if key:
            c[key] += 1
            continue
        if ch in (ZWJ, ZWNJ):
            prev = out[-1] if out else ""
            nxt = text[i + 1] if i + 1 < n else ""
            name = "zwj" if ch == ZWJ else "zwnj"
            if _in_joiner_script(prev) or _in_joiner_script(nxt):
                c[f"{name}_kept_indic"] += 1
                out.append(ch)
            elif ch == ZWJ and (_is_emoji(prev) and _is_emoji(nxt)):
                c["zwj_kept_emoji"] += 1
                out.append(ch)
            else:
                c[f"{name}_removed_stray"] += 1
            continue
        if ch in ODD_SPACES:
            c["odd_space"] += 1
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def _clean_prose(seg: str, c: Counter, unescape_html: bool) -> str:
    if unescape_html and "&" in seg:
        def _one(m: re.Match) -> str:
            rep = html.unescape(m.group(0))
            if rep != m.group(0):
                c["html_entity"] += 1
            return rep

        # only real "&...;" entities; html.unescape on the whole string would
        # also rewrite "?a=1&copy=2" inside URLs
        seg = ENTITY_RE.sub(_one, seg)
    before = len(seg)
    seg = MULTISPACE_RE.sub(" ", seg)
    if len(seg) != before:
        c["space_runs_collapsed"] += 1
    return seg


def normalize_text(text: str, unescape_html: bool = True) -> tuple[str, Counter]:
    c: Counter = Counter()
    if not text:
        return text, c
    if "\r" in text:
        c["crlf"] += text.count("\r")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        c["nfc_changed"] += 1
        text = nfc
    text = _strip_invisible(text, c)

    # code: leave fenced blocks and inline spans alone (indentation, entities)
    parts = FENCE_RE.split(text)
    for k, part in enumerate(parts):
        if k % 2 == 1:  # a fenced block
            continue
        parts[k] = _clean_prose(part, c, unescape_html)
    text = "".join(parts)

    new = TRAILING_WS_RE.sub("", text)
    if new != text:
        c["trailing_ws"] += 1
        text = new
    new = MANY_BLANKS_RE.sub("\n\n", text)
    if new != text:
        c["blank_lines_collapsed"] += 1
        text = new
    stripped = text.strip()
    if stripped != text:
        c["edge_ws"] += 1
    return stripped, c
