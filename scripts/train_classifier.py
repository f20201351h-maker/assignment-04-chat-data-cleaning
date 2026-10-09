"""Quality layer 2: train the cheap classifier on LLM labels and score every conversation.

python scripts/train_classifier.py

Inputs : data/audit/label_batches/index.json            (which ids were sampled)
         data/audit/labels/batch_XX.labels.jsonl        ({"id", "score" 0-5, "why"} from the LLM labeller)
         data/clean/*.jsonl                             (the conversations)
Outputs: artifacts/quality_classifier.json              (agreement, bias check, score distribution)
         artifacts/quality_classifier_weights.npz       (weights, so the gate can be re-applied)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from s4clean.classifier import LogReg, auc, features  # noqa: E402
from s4clean.manifest import sha256_bytes  # noqa: E402

INDIC = {"hi", "hi-Latn", "mr", "bn", "te", "ta", "kn", "ml", "gu", "pa", "or", "as", "ne", "ur", "sa"}


def conv_text(rec):
    parts = []
    for m in rec["messages"]:
        if m["role"] == "system":
            continue
        body = m.get("content") or ""
        if m.get("tool_call"):
            body += " " + json.dumps(m["tool_call"], ensure_ascii=False)
        parts.append(f"{m['role']}: {body}")
    return "\n".join(parts)


clean = {}
for shard in ("glaive", "anudesh"):
    for line in open(ROOT / f"data/clean/{shard}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        clean[r["id"]] = r

labels = {}
for f in sorted((ROOT / "data/audit/labels").glob("batch_*.labels.jsonl")):
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if isinstance(r.get("score"), (int, float)) and 0 <= r["score"] <= 5:
            labels[r["id"]] = r
index = json.loads((ROOT / "data/audit/label_batches/index.json").read_text(encoding="utf-8"))
items = [it for it in index if it["id"] in labels and it["id"] in clean]
missing = len(index) - len(items)

ids = [it["id"] for it in items]
y = np.array([1 if labels[i]["score"] >= 3 else 0 for i in ids])
raw_scores = np.array([labels[i]["score"] for i in ids])
X = [features(conv_text(clean[i])) for i in ids]
# deterministic split by id hash: ~25% held out
test = np.array([int(sha256_bytes(i.encode())[:8], 16) % 4 == 0 for i in ids])

model = LogReg().fit([x for x, t in zip(X, test) if not t], y[~test])
p_test = model.predict_proba([x for x, t in zip(X, test) if t])
acc = float(((p_test >= 0.5).astype(int) == y[test]).mean())
base_rate = float(y[~test].mean())


def group_of(it):
    return "indic" if it["lang"] in INDIC else "latin_or_other"


by_group = defaultdict(lambda: {"n": 0, "label_keep": 0, "pred_keep": 0})
test_items = [it for it, t in zip(items, test) if t]
for it, p, yy in zip(test_items, p_test, y[test]):
    g = by_group[f"{it['source']}/{group_of(it)}"]
    g["n"] += 1
    g["label_keep"] += int(yy)
    g["pred_keep"] += int(p >= 0.5)

# score everything
all_ids = sorted(clean)
probs = {}
for k in range(0, len(all_ids), 2000):
    chunk = all_ids[k : k + 2000]
    for i, p in zip(chunk, model.predict_proba([features(conv_text(clean[i])) for i in chunk])):
        probs[i] = float(p)
dist = defaultdict(Counter)
for i, p in probs.items():
    r = clean[i]
    g = f"{r['source']}/{'indic' if r['lang']['doc'] in INDIC else 'latin_or_other'}"
    dist[g]["n"] += 1
    for th in (0.1, 0.2, 0.3, 0.5):
        if p < th:
            dist[g][f"below_{th}"] += 1

# the script-aware view (widget 4's suggested fix): how well does it rank inside each group,
# and does a model trained only on Indic labels do better?
grp = np.array(["indic" if it["lang"] in INDIC else "latin_or_other" for it in items])
within = {}
for g in ("indic", "latin_or_other"):
    sel = grp[test] == g
    within[g] = {"test_n": int(sel.sum()), "label_keep_rate": round(float(y[test][sel].mean()), 3),
                 "pred_keep_rate": round(float((p_test[sel] >= 0.5).mean()), 3),
                 "within_group_auc": round(auc(y[test][sel], p_test[sel]), 3)}
label_rate_all = {g: {"n": int((grp == g).sum()), "keep_rate": round(float(y[grp == g].mean()), 3)} for g in ("indic", "latin_or_other")}
tr = (~test) & (grp == "indic")
te = test & (grp == "indic")
indic_model = LogReg().fit([X[i] for i in np.nonzero(tr)[0]], y[tr])
indic_auc = auc(y[te], indic_model.predict_proba([X[i] for i in np.nonzero(te)[0]]))

np.savez_compressed(ROOT / "artifacts/quality_classifier_weights.npz", w=model.w, b=np.array([model.b]))
label_hist = Counter(int(s) for s in raw_scores)
out = {
    "labeller": "Claude Sonnet subagents, rubric in docs/notes/labelling_rubric.md",
    "labelled": len(items), "labelled_missing_or_unparsed": missing,
    "label_histogram_0to5": {str(k): label_hist.get(k, 0) for k in range(6)},
    "keep_if_score_at_least": 3,
    "train_n": int((~test).sum()), "test_n": int(test.sum()), "train_keep_rate": round(base_rate, 3),
    "test_auc": round(auc(y[test], p_test), 3), "test_accuracy": round(acc, 3),
    "majority_baseline_accuracy": round(max(float(y[test].mean()), 1 - float(y[test].mean())), 3),
    "test_by_group": {k: dict(v) for k, v in sorted(by_group.items())},
    "score_distribution_all_docs": {k: dict(v) for k, v in sorted(dist.items())},
    "model": "logistic regression on hashed char 2-4-grams (2^18 dims), numpy, 300 epochs, l2=1e-4",
    "label_keep_rate_by_script_all_labels": label_rate_all,
    "test_within_group": within,
    "indic_only_model": {"train_n": int(tr.sum()), "test_n": int(te.sum()), "test_auc": round(indic_auc, 3)},
    "used_to_remove_documents": False,
    "decision": "Not used as a filter. The one model for both scripts predicts almost every Indic conversation as "
                "low quality, far below the LLM labels' own keep rate for Indic. A per-script model ranks better but "
                "is trained on too few Indic labels to delete data with. Scores are kept as an audit.",
}
(ROOT / "artifacts/quality_classifier.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: out[k] for k in ("labelled", "test_auc", "test_accuracy", "majority_baseline_accuracy", "test_by_group")}, indent=1))
