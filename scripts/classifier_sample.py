"""Pick the conversations an LLM will label for the quality classifier (layer 2).

python scripts/classifier_sample.py [n_per_source]

Takes a seeded random sample from the cleaned shards (after the heuristic
layer, dedup, PII masking), and writes numbered batches to
data/audit/label_batches/batch_<k>.jsonl. PII is already masked in the
cleaned shards, so the labeller never sees raw identifiers.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
batch_size = 100
out = ROOT / "data" / "audit" / "label_batches"
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("*.jsonl"):
    old.unlink()

items = []
for shard in ("glaive", "anudesh"):
    rows = [json.loads(l) for l in open(ROOT / f"data/clean/{shard}.jsonl", encoding="utf-8")]
    rng = random.Random(f"classifier-sample:{shard}")
    for r in rng.sample(rows, min(n, len(rows))):
        conv = []
        for m in r["messages"]:
            if m["role"] == "system":
                continue  # function schemas are long and not what we're judging
            body = m.get("content") or ""
            if m.get("tool_call"):
                body = (body + "\n" if body else "") + "CALL " + json.dumps(m["tool_call"], ensure_ascii=False)
            conv.append({"role": m["role"], "text": body[:1500]})
        items.append({"id": r["id"], "source": shard, "lang": r["lang"]["doc"], "conversation": conv})
random.Random("classifier-sample:order").shuffle(items)
for k in range(0, len(items), batch_size):
    with open(out / f"batch_{k // batch_size:02d}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in items[k : k + batch_size]:
            f.write(json.dumps({"id": it["id"], "conversation": it["conversation"]}, ensure_ascii=False) + "\n")
(out / "index.json").write_text(json.dumps([{k: it[k] for k in ("id", "source", "lang")} for it in items], indent=0),
                                encoding="utf-8")
print(len(items), "items in", (len(items) + batch_size - 1) // batch_size, "batches")
