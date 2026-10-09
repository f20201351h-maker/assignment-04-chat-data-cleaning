"""Provenance manifest helpers (Session 4, section 11).

Rules taken from the lesson:
  * identifiers come from content, never from a running counter or a clock
  * the content hash is computed AFTER cleaning
  * every cleaning script is recorded by human-readable name + hash of its code
  * a shard with unknown license or a missing required field is BLOCKED
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REQUIRED_FIELDS = (
    "source",
    "license",
    "contributor",
    "cleaning_scripts",
    "content_sha256",
    "token_count",
    "languages",
)
UNSAFE_LICENSES = {"", "unknown", "other", "proprietary", "none"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash(path: str | Path) -> str:
    """Hash of a source file with line endings normalised, so a git checkout
    on Windows (CRLF) and Linux (LF) give the same script hash."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return sha256_bytes(data)


def doc_id(text: str) -> str:
    """Content-derived document id: same cleaned text -> same id, every run."""
    return "d_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def manifest_status(m: dict) -> tuple[str, list[str]]:
    problems = []
    for f in REQUIRED_FIELDS:
        v = m.get(f)
        if v is None or v == "" or v == [] or v == {}:
            problems.append(f"missing:{f}")
    lic = str(m.get("license", "")).strip().lower()
    if lic in UNSAFE_LICENSES:
        problems.append(f"unsafe_license:{lic or 'empty'}")
    return ("blocked" if problems else "clean"), problems
