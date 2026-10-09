"""PII removal, regex layer (Session 4, section 9 + widget 8).

Widget 8 masks emails, phone numbers and IP addresses with typed placeholders
[EMAIL] [PHONE] [IP]. Its phone pattern is any 9-13 digit run, which on a
maths / code corpus would mask answers, timestamps and ids. So:

  * email  - widget 8 pattern; addresses on reserved example domains
             (example.com, *.test, ...) are placeholders, counted but kept
  * phone  - needs phone-like shape: a +country code, or (area) code, or
             digit groups split by space / dash / dot. Bare digit runs are
             not treated as phones.
  * ip     - IPv4 / IPv6 parsed with the ipaddress module; only globally
             routable addresses are masked. 127.0.0.1, 192.168.x.x, 10.x,
             documentation ranges (203.0.113.x ...) are not personal data.
  * secret - API-key-like tokens (Rohan at 23m: leaked passwords / keys).

Every decision is counted so false positives can be audited.
"""
from __future__ import annotations

import ipaddress
import re
from collections import Counter

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# only the names reserved for examples (RFC 2606 / 6761). Domains like
# mail.com or company.com are real, so addresses on them are masked.
RESERVED_EMAIL_DOMAINS = re.compile(
    r"@(?:[\w.-]+\.)?(?:example\.(?:com|net|org)|[\w-]+\.(?:test|example|invalid|localhost))$",
    re.I,
)

# +country code with grouped digits, or (area) code, or 3-3-4 / 5-5 groupings
PHONE_RE = re.compile(
    r"(?<![\w.+-])(?:"
    r"\+\d{1,3}[\s.-]?\(?\d{1,5}\)?(?:[\s.-]?\d{2,5}){2,4}"  # +91 98450 12345, +1 (415) 555-0100
    r"|\(\d{3}\)\s?\d{3}[\s.-]\d{4}"  # (415) 555-0100
    r"|\d{3}[.-]\d{3}[.-]\d{4}"  # 415-555-0100 / 415.555.0100
    r"|\d{5}\s\d{5}"  # 98450 12345 (Indian mobile, spaced)
    r")(?![\w-])"
)
# fictional 555-01xx numbers used in examples
FICTIONAL_PHONE_RE = re.compile(r"555[\s.-]?01\d\d")

# (?<!\w/) skips version strings like Chrome/91.0.4.12 in user-agent text
IPV4_RE = re.compile(r"(?<![\w.])(?<!\w/)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\.?\d)(?!\w)")
IPV6_RE = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{1,4}(?![\w:])")

SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})\b"
)


def _ip_is_personal(s: str) -> bool:
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return False
    return ip.is_global


def scrub(text: str) -> tuple[str, Counter, list[tuple[str, str]]]:
    """Return (masked_text, counts, findings). findings = [(kind, matched_text)]
    for audit only; it is never written to the public site."""
    c: Counter = Counter()
    found: list[tuple[str, str]] = []

    def _email(m: re.Match) -> str:
        s = m.group(0)
        if RESERVED_EMAIL_DOMAINS.search(s):
            c["email_placeholder_kept"] += 1
            return s
        c["email"] += 1
        found.append(("email", s))
        return "[EMAIL]"

    def _ip(m: re.Match) -> str:
        s = m.group(0)
        if _ip_is_personal(s):
            c["ip"] += 1
            found.append(("ip", s))
            return "[IP]"
        c["ip_nonpublic_kept"] += 1
        return s

    def _phone(m: re.Match) -> str:
        s = m.group(0)
        if FICTIONAL_PHONE_RE.search(s):
            c["phone_fictional_kept"] += 1
            return s
        digits = sum(ch.isdigit() for ch in s)
        if not 9 <= digits <= 15:
            c["phone_shape_rejected"] += 1
            return s
        c["phone"] += 1
        found.append(("phone", s))
        return "[PHONE]"

    def _secret(m: re.Match) -> str:
        c["secret"] += 1
        found.append(("secret", m.group(0)[:6] + "..."))
        return "[SECRET]"

    text = EMAIL_RE.sub(_email, text)
    text = SECRET_RE.sub(_secret, text)
    text = IPV4_RE.sub(_ip, text)
    if ":" in text:
        text = IPV6_RE.sub(lambda m: _ip(m) if _looks_ipv6(m.group(0)) else m.group(0), text)
    text = PHONE_RE.sub(_phone, text)
    return text, c, found


def _looks_ipv6(s: str) -> bool:
    # Python slices (a[0::2]) and Anki cloze markers ({{c1::...}}) parse as IPv6
    # and even count as "global" (::2). Real addresses in text have 3+ groups.
    if sum(1 for g in s.split(":") if g) < 3:
        return False
    try:
        return isinstance(ipaddress.ip_address(s), ipaddress.IPv6Address)
    except ValueError:
        return False
