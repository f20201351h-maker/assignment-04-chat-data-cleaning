"""Language identification and validation (Session 4, section 8 + widget 7).

Detect the language of each text at runtime instead of trusting a label.

* Script histogram over letters (widget 7 uses the same idea for Latin /
  Devanagari / Bengali / Telugu). Widget 7's code-switch rule is kept:
  top script share < 0.80 and second >= 0.20 -> "mixed".
* For Latin-script text, py3langid (langid.py's model) picks the language.
* Codes are ISO 639-1 and are validated against an explicit table that
  raises on unknown codes - widget 7's te/tel bug "passed by luck" through
  a silent .get(key, default) fallback.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

SCRIPT_RANGES = [
    ("Latin", [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)]),
    ("Devanagari", [(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("Bengali", [(0x0980, 0x09FF)]),
    ("Gurmukhi", [(0x0A00, 0x0A7F)]),
    ("Gujarati", [(0x0A80, 0x0AFF)]),
    ("Oriya", [(0x0B00, 0x0B7F)]),
    ("Tamil", [(0x0B80, 0x0BFF)]),
    ("Telugu", [(0x0C00, 0x0C7F)]),
    ("Kannada", [(0x0C80, 0x0CFF)]),
    ("Malayalam", [(0x0D00, 0x0D7F)]),
    ("Arabic", [(0x0600, 0x06FF), (0x0750, 0x077F)]),
    ("Cyrillic", [(0x0400, 0x04FF)]),
    ("Greek", [(0x0370, 0x03FF)]),
    ("Hebrew", [(0x0590, 0x05FF)]),
    ("Thai", [(0x0E00, 0x0E7F)]),
    ("Hangul", [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)]),
    ("Kana", [(0x3040, 0x30FF)]),
    ("Han", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)]),
]

# script -> default language code when the script identifies the language
SCRIPT_LANG = {
    "Bengali": "bn", "Gurmukhi": "pa", "Gujarati": "gu", "Oriya": "or", "Tamil": "ta",
    "Telugu": "te", "Kannada": "kn", "Malayalam": "ml", "Greek": "el", "Hebrew": "he",
    "Thai": "th", "Hangul": "ko", "Kana": "ja",
}

# explicit, closed table; unknown codes raise instead of falling back
ISO639_1 = {
    "en", "hi", "bn", "as", "mr", "ne", "sa", "pa", "gu", "or", "ta", "te", "kn", "ml", "ur",
    "ar", "fa", "ru", "uk", "bg", "el", "he", "th", "ko", "ja", "zh", "es", "fr", "de", "it",
    "pt", "nl", "pl", "tr", "vi", "id", "sv", "da", "no", "fi", "cs", "ro", "hu", "und", "mixed", "hi-latn",
}
ISO639_3_TO_1 = {"eng": "en", "hin": "hi", "ben": "bn", "asm": "as", "tel": "te", "tam": "ta",
                 "kan": "kn", "mal": "ml", "mar": "mr", "guj": "gu", "pan": "pa", "ori": "or",
                 "urd": "ur", "zho": "zh", "jpn": "ja", "kor": "ko", "rus": "ru", "spa": "es",
                 "fra": "fr", "deu": "de", "por": "pt", "ara": "ar"}


class LanguageCodeError(ValueError):
    pass


def validate_code(code: str) -> str:
    """Fail loudly on anything that is not a known code (widget 7 te/tel bug)."""
    c = code.strip().lower()
    if len(c) == 3 and c in ISO639_3_TO_1:
        return ISO639_3_TO_1[c]
    if c in ISO639_1:
        return c
    raise LanguageCodeError(f"unknown language code {code!r}")


_RANGE_TABLE = sorted((lo, hi, name) for name, rs in SCRIPT_RANGES for lo, hi in rs)
_CODE_RE = re.compile(r"```.*?(?:```|\Z)|`[^`\n]+`", re.S)
_URL_RE = re.compile(r"https?://\S+")


def _script_of(cp: int) -> str | None:
    for lo, hi, name in _RANGE_TABLE:
        if cp < lo:
            return None
        if cp <= hi:
            return name
    return None


def _script_regexes():
    out = []
    for name, rs in SCRIPT_RANGES:
        cls = "".join(re.escape(chr(a)) + "-" + re.escape(chr(b)) for a, b in rs)
        out.append((name, re.compile("[" + cls + "]")))
    return out


_SCRIPT_RES = _script_regexes()
_ASCII_LETTER = re.compile(r"[A-Za-z]")


def script_histogram(text: str, max_chars: int = 20000) -> dict[str, int]:
    """Letters per script in the first max_chars characters (code and URLs removed).
    Counting is done with one compiled regex per script so it runs in C."""
    text = _URL_RE.sub(" ", _CODE_RE.sub(" ", (text or "")[:max_chars]))
    if text.isascii():
        n = len(_ASCII_LETTER.findall(text))
        return {"Latin": n} if n else {}
    counts: dict[str, int] = {}
    for name, rx in _SCRIPT_RES:
        n = len(rx.findall(text))
        if n:
            counts[name] = n
    return counts


# very common Hindi/Urdu function words as typed in Latin script. Words that
# are also English or Spanish ("to", "me", "the", "hum", "par", "ho", "se")
# are deliberately left out.
_HINGLISH = frozenset(
    "hai hain nahi nahin kya kyun kyon mujhe mera meri mere tum tumhe tumhara aap aapka hamara "
    "yeh ye woh wo kaise kaisa kaun kahan kab aur lekin magar bahut thoda accha acha theek thik "
    "karo karna karta karti karte kar raha rahi rahe hoga hogi tha thi gaya gayi diya liya "
    "matlab samajh bata batao chahiye sakta sakti sakte kuch sab koi abhi phir jaldi yaar bhai "
    "ki ka ke ko mein".split()
)
_TOK_RE = re.compile(r"[a-z]+")


def _romanized_hindi_share(text: str) -> float:
    toks = _TOK_RE.findall(text.lower())
    if len(toks) < 5:
        return 0.0
    hits = [t for t in toks if t in _HINGLISH]
    if len(set(hits)) < 3:
        return 0.0
    return len(hits) / len(toks)


@lru_cache(maxsize=1)
def _langid():
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    return LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)


def detect(text: str, min_letters: int = 20) -> dict:
    """Return {lang, conf, scripts, status} where status is one of
    ok / mixed / too_short. Code blocks and URLs are ignored."""
    text = text or ""
    hist = script_histogram(text)
    total = sum(hist.values())
    if total < min_letters:
        return {"lang": "und", "conf": 0.0, "scripts": hist, "status": "too_short"}
    ranked = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))
    top, top_n = ranked[0]
    top_share = top_n / total
    second_share = ranked[1][1] / total if len(ranked) > 1 else 0.0
    if top_share < 0.80 and second_share >= 0.20:
        return {"lang": "mixed", "conf": round(top_share, 3), "scripts": hist, "status": "mixed",
                "parts": [s for s, n in ranked if n / total >= 0.15]}
    if top in SCRIPT_LANG:
        return {"lang": SCRIPT_LANG[top], "conf": round(top_share, 3), "scripts": hist, "status": "ok"}
    # scripts shared by several languages: let the n-gram model decide
    plain = _URL_RE.sub(" ", _CODE_RE.sub(" ", text))
    if top == "Latin" and _romanized_hindi_share(plain) >= 0.15:
        # langid has no class for Hindi written in Latin script (36m-38m:
        # "classify this as a Latin way of representing Telugu")
        return {"lang": "hi-Latn", "conf": round(_romanized_hindi_share(plain), 3), "scripts": hist, "status": "ok"}
    lang, prob = _langid().classify(plain[:4000])
    if lang not in ISO639_1:
        return {"lang": "und", "conf": round(float(prob), 3), "scripts": hist, "status": "low_conf", "guess": lang}
    if top == "Latin" and prob < 0.5:
        return {"lang": "und", "conf": round(float(prob), 3), "scripts": hist, "status": "low_conf",
                "guess": lang}
    if top == "Devanagari" and lang not in ("hi", "mr", "ne", "sa"):
        lang = "hi"
    if top == "Arabic" and lang not in ("ar", "fa", "ur"):
        lang = "ar"
    if top == "Han" and lang != "zh":
        lang = "zh"
    if top == "Cyrillic" and lang not in ("ru", "uk", "bg"):
        lang = "ru"
    return {"lang": validate_code(lang), "conf": round(float(prob), 3),
            "scripts": hist, "status": "ok"}
