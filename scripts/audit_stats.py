"""Independent statistics audit.

Recomputes the headline numbers from the files on disk instead of trusting
the pipeline's counters, and checks invariants of the cleaned shards.

python scripts/audit_stats.py   ->  artifacts/audits/stats_audit.json (exit 1 on any failure)
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s4clean.formats import GHOST_RE  # noqa: E402
from s4clean.manifest import sha256_file  # noqa: E402
from s4clean.pii import EMAIL_RE, RESERVED_EMAIL_DOMAINS  # noqa: E402
from s4clean.tokens import count_tokens  # noqa: E402

checks = []


def check(name, ok, detail=None):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail is not None else ""))


metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
manifest = json.loads((ROOT / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
stages = {s["stage"]: s for s in metrics["stages"]}

# 1. raw input counts straight from the source files
raw_glaive = len(json.loads((ROOT / "data/raw/glaive-function-calling-v2.json").read_text(encoding="utf-8")))
import pyarrow.parquet as pq  # noqa: E402

raw_anudesh = pq.ParquetFile(ROOT / "data/raw/anudesh1.parquet").metadata.num_rows
check("raw glaive rows match extract docs_in", raw_glaive == stages["extract"]["docs_in"]["glaive"], raw_glaive)
check("raw anudesh rows match extract docs_in", raw_anudesh == stages["extract"]["docs_in"]["anudesh"], raw_anudesh)

# 2. conservation: every input row is either in a clean shard or in exactly one quarantine file
clean = {}
for sh in ("glaive", "anudesh"):
    clean[sh] = [json.loads(l) for l in open(ROOT / f"data/clean/{sh}.jsonl", encoding="utf-8")]
quar = Counter()
quar_keys = Counter()
qreason = Counter()
for f in sorted((ROOT / "data/quarantine").glob("*.jsonl")):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        quar[r["source"]] += 1
        quar_keys[(r["source"], r["src_index"])] += 1
        qreason[(f.stem, r["reason"])] += 1
for sh, raw_n in (("glaive", raw_glaive), ("anudesh", raw_anudesh)):
    kept = len(clean[sh])
    check(f"{sh}: clean + quarantined == raw", kept + quar[sh] == raw_n, f"{kept} + {quar[sh]} vs {raw_n}")
dup_q = [k for k, v in quar_keys.items() if v > 1]
check("no row quarantined twice", not dup_q, len(dup_q))
kept_keys = {(sh, r["src_index"]) for sh in clean for r in clean[sh]}
check("no row both kept and quarantined", not (kept_keys & set(quar_keys)), len(kept_keys & set(quar_keys)))

# 3. pipeline removal counters agree with quarantine files
for st in metrics["stages"]:
    for src, reasons in st["removed"].items():
        for reason, n in reasons.items():
            got = qreason[(st["stage"], reason)]
            # quarantine is per stage+reason across sources; compare totals
    totals = Counter()
    for src, reasons in st["removed"].items():
        for reason, n in reasons.items():
            totals[reason] += n
    for reason, n in totals.items():
        check(f"stage {st['stage']}: removed[{reason}] == quarantine lines", qreason[(st["stage"], reason)] == n,
              f"{n} vs {qreason[(st['stage'], reason)]}")

# 4. manifest hashes match files, ids unique, token counts recomputed
for sh in manifest["shards"]:
    name = "glaive" if "glaive" in sh["source"] else "anudesh"
    path = ROOT / f"data/clean/{name}.jsonl"
    check(f"{name}: manifest content_sha256 matches file", sh["content_sha256"] == sha256_file(path))
    check(f"{name}: manifest docs == lines", sh["docs"] == len(clean[name]), sh["docs"])
    check(f"{name}: manifest status clean", sh["status"] == "clean", sh["problems"])
    ids = [r["id"] for r in clean[name]]
    check(f"{name}: doc ids unique", len(ids) == len(set(ids)))
    # recount tokens independently: content tokens + one per message + <|end|>
    texts, owners = [], []
    for i, r in enumerate(clean[name]):
        for m in r["messages"]:
            body = m.get("content", "")
            if m.get("tool_call"):
                body = (body + "\n" if body else "") + json.dumps(m["tool_call"], ensure_ascii=False, sort_keys=True)
            texts.append(body)
            owners.append(i)
    c = count_tokens(texts)
    total = sum(c) + sum(len(r["messages"]) + 1 for r in clean[name])
    check(f"{name}: recomputed tokens == manifest token_count", total == sh["token_count"], f"{total} vs {sh['token_count']}")

# 5. invariants of cleaned content
ghost = Counter()
emails = 0
zw_bad = Counter()
for name in clean:
    for r in clean[name]:
        for m in r["messages"]:
            t = m.get("content", "") + (json.dumps(m["tool_call"], ensure_ascii=False) if m.get("tool_call") else "")
            for g in GHOST_RE.findall(t):
                ghost[g] += 1
            for e in EMAIL_RE.findall(t):
                if not RESERVED_EMAIL_DOMAINS.search(e):
                    emails += 1
            for ch, nm in (("\u200b", "ZWSP"), ("\ufeff", "BOM"), ("\ufffd", "U+FFFD"), ("\u202e", "RLO")):
                zw_bad[nm] += t.count(ch)
check("no ghost markers left in clean content", not ghost, dict(ghost))
check("no unmasked real-looking emails in clean content", emails == 0, emails)
check("no ZWSP/BOM/U+FFFD/RLO in clean content", not +zw_bad, dict(zw_bad))
joiners = sum(m.get("content", "").count("\u200c") + m.get("content", "").count("\u200d")
              for r in clean["anudesh"] for m in r["messages"])
check("Indic joiners survive in clean anudesh shard", joiners > 0, joiners)

out = {"checks": checks, "all_ok": all(c["ok"] for c in checks),
       "recomputed": {"raw_rows": {"glaive": raw_glaive, "anudesh": raw_anudesh},
                      "clean_docs": {k: len(v) for k, v in clean.items()},
                      "quarantined": dict(quar), "joiners_in_clean_anudesh": joiners}}
(ROOT / "artifacts" / "audits").mkdir(parents=True, exist_ok=True)
(ROOT / "artifacts" / "audits" / "stats_audit.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
sys.exit(0 if out["all_ok"] else 1)
