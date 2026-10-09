"""Rewrite invisible / odd characters in this repo's Python files as escapes.

A cleaner that removes zero-width and control characters should not have
them hiding in its own source. Run after editing; tests/test_normalize.py
fails if any are left.
"""
import pathlib
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[1]
BS = chr(92)


def needs_escape(ch: str) -> bool:
    cat = unicodedata.category(ch)
    if cat in ("Cf", "Co", "Cc") and ch not in "\t\n":
        return True
    if cat == "Zs" and ch != " ":
        return True
    return ord(ch) == 0xFFFD


targets = [f for d in ("src", "scripts", "tests") for f in (ROOT / d).rglob("*.py")]
targets += [f for f in (ROOT / "site").glob("*.js")] + [f for f in (ROOT / "site").glob("*.mjs")]
for f in targets:
    s = f.read_text(encoding="utf-8")
    new = "".join(BS + "u" + format(ord(ch), "04x") if needs_escape(ch) else ch for ch in s)
    if new != s:
        f.write_text(new, encoding="utf-8", newline="\n")
        print("rewrote", f.relative_to(ROOT))
