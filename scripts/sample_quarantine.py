"""Draw a reproducible review sample from every (stage, reason) in data/quarantine.

python scripts/sample_quarantine.py [per_reason]

Writes data/audit/review_pack/<stage>__<reason>.jsonl (local only, may contain PII).
Each line is shortened for reading: messages are cut to 600 characters each.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
per = int(sys.argv[1]) if len(sys.argv) > 1 else 25
out = ROOT / "data" / "audit" / "review_pack"
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("*.jsonl"):
    old.unlink()


def shorten(rec):
    r = {k: v for k, v in rec.items() if k not in ("messages", "raw")}
    if rec.get("messages"):
        r["messages"] = [{"role": m["role"], "content": (m.get("content") or "")[:600],
                          **({"tool_call": m["tool_call"]} if m.get("tool_call") else {})} for m in rec["messages"]]
    elif rec.get("raw"):
        r["raw"] = json.dumps(rec["raw"], ensure_ascii=False)[:1500]
    return r


clean_index = {}
for shard in ("glaive", "anudesh"):
    p = ROOT / "data" / "clean" / f"{shard}.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            clean_index[(shard, r["src_index"])] = r


def kept_partner(rec):
    d = rec.get("detail") or {}
    if isinstance(d, dict) and "kept" in d:
        k = clean_index.get(tuple(d["kept"]))
        if k:
            return shorten({"source": k["source"], "src_index": k["src_index"], "messages": k["messages"]})
    return None


summary = {}
for f in sorted((ROOT / "data" / "quarantine").glob("*.jsonl")):
    groups = defaultdict(list)
    for line in open(f, encoding="utf-8"):
        rec = json.loads(line)
        groups[rec["reason"]].append(rec)
    for reason, recs in sorted(groups.items()):
        rng = random.Random(f"{f.stem}:{reason}")
        pick = recs if len(recs) <= per else rng.sample(recs, per)
        pick.sort(key=lambda r: (r["source"], r["src_index"]))
        with open(out / f"{f.stem}__{reason}.jsonl", "w", encoding="utf-8", newline="\n") as w:
            for rec in pick:
                row = shorten(rec)
                partner = kept_partner(rec)
                if partner:
                    row["kept_partner"] = partner
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[f"{f.stem}/{reason}"] = {"total": len(recs), "sampled": len(pick)}

# PII findings: a sample of masked values per kind, to judge precision
pf = ROOT / "data" / "audit" / "pii_findings.jsonl"
if pf.exists():
    by_kind = defaultdict(list)
    for line in open(pf, encoding="utf-8"):
        r = json.loads(line)
        by_kind[r["kind"]].append(r)
    with open(out / "pii__findings_sample.jsonl", "w", encoding="utf-8", newline="\n") as w:
        for kind, recs in sorted(by_kind.items()):
            rng = random.Random(f"pii:{kind}")
            pick = recs if len(recs) <= 40 else rng.sample(recs, 40)
            for rec in sorted(pick, key=lambda r: (r["source"], r["src_index"])):
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            summary[f"pii/{kind}"] = {"total": len(recs), "sampled": len(pick)}
print(json.dumps(summary, indent=1))
