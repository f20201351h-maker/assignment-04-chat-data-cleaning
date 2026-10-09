import pathlib
import unicodedata

from s4clean.normalize import normalize_text

ZWSP, ZWNJ, ZWJ, BOM = chr(0x200B), chr(0x200C), chr(0x200D), chr(0xFEFF)
RLO, PDF = chr(0x202E), chr(0x202C)


def test_deterministic():
    s = "caf" + "e" + chr(0x301) + " &amp; " + ZWSP + "x  y"
    assert normalize_text(s) == normalize_text(s)


def test_nfc_composes():
    out, c = normalize_text("cafe" + chr(0x301))
    assert out == "caf" + chr(0xE9)
    assert c["nfc_changed"] == 1


def test_noise_removed():
    s = BOM + "a" + ZWSP + "b " + RLO + "c" + PDF + chr(0xFFFD) + chr(0x07) + chr(0xE000) + chr(0xAD) + "d"
    out, c = normalize_text(s)
    assert out == "ab cd"
    for k in ("bom", "zero_width_space", "bidi_control", "replacement_char", "control_char", "private_use", "soft_hyphen"):
        assert c[k] == 1 or (k == "bidi_control" and c[k] == 2), k


def test_indic_joiners_kept():
    # Devanagari half-form with ZWJ and the ZWNJ variant: both are real spelling
    for joiner in (ZWJ, ZWNJ):
        word = "क्" + joiner + "ष"
        out, c = normalize_text(word)
        assert out == word
    # Malayalam chillu-style and Bengali ZWNJ
    for word in ("ന്" + ZWJ, "র্" + ZWNJ + "য"):
        assert normalize_text(word)[0] == word


def test_stray_latin_joiner_removed():
    out, c = normalize_text("hel" + ZWJ + "lo")
    assert out == "hello"
    assert c["zwj_removed_stray"] == 1


def test_emoji_zwj_sequence_kept():
    fam = chr(0x1F468) + ZWJ + chr(0x1F469) + ZWJ + chr(0x1F467)
    assert normalize_text(fam)[0] == fam


def test_html_entities_only_real_ones():
    out, _ = normalize_text("Tom &amp; Jerry&#39;s ?a=1&copy=2")
    assert out == "Tom & Jerry's ?a=1&copy=2"


def test_code_whitespace_preserved():
    src = "```python\ndef f():\n\tif x:\n        return  1\n```\nsome   prose"
    out, _ = normalize_text(src)
    assert "\tif x:\n        return  1" in out
    assert out.endswith("some prose")


def test_inline_code_left_alone():
    out, _ = normalize_text("use `a  &lt; b` here")
    assert "`a  &lt; b`" in out


def test_blank_lines_and_crlf():
    out, c = normalize_text("a\r\n\r\n\r\n\r\nb   \n")
    assert out == "a\n\nb"


def test_source_files_have_no_invisible_characters():
    """The cleaner's own code must not hide the characters it removes."""
    root = pathlib.Path(__file__).resolve().parents[1]
    for f in list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            bad = [hex(ord(ch)) for ch in line if unicodedata.category(ch) in ("Cf", "Cc", "Co") and ch != "\t"]
            assert not bad, f"{f}:{i} contains {bad}"
