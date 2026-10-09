"""Summarise the false-positive review into a committed artifact.

Input : data/audit/fp_audit_raw.json   (hand review of random removals from run 2; no PII in it)
Output: artifacts/audits/false_positive_audit.json

The review was done on the second full run. Each reviewed rule gets the action
taken afterwards, so the site can show what the review changed.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
raw = json.loads((ROOT / "data/audit/fp_audit_raw.json").read_text(encoding="utf-8"))

ACTIONS = {
    "quality__repeated_lines.jsonl": "rule dropped: mostly hit tables, meal plans and seating charts",
    "quality__degenerate_repetition.jsonl": "replaced by a looping-output detector that ignores tables, rule lines and pure digits",
    "format__unrecoverable_function_call.jsonl": "repair added for a stray extra closing brace",
    "decontam__eval_overlap.jsonl": "still removed (conservative), reported as phrase overlap; item coverage now recorded",
    "dedup__near_duplicate.jsonl": "kept: refusal templates that differ only in which tool they mention",
    "langid__user_language_outside_source_claim.jsonl": "kept: one English chat quoting a French poem",
    "langid__assistant_language_outside_source_claim.jsonl": "kept: unclear cases are user-requested French/Spanish turns",
}
files = []
for name, v in raw["files"].items():
    files.append({"file": name, "stage": name.split("__")[0], "reason": name.split("__")[1].removesuffix(".jsonl"),
                  "sampled": v["n"], "correct": v["tp"], "false_positive": v["fp"], "unclear": v["unclear"],
                  "action": ACTIONS.get(name, "no change: no false positives in the sample")})
pii = raw["pii"]
out = {
    "reviewed_run": "run 2 (before the rule changes)",
    "method": "random sample of up to 25 removals per (stage, reason), read by a Claude Sonnet reviewer; "
              "consequential cases re-checked by the lead agent",
    "records_reviewed": sum(f["sampled"] for f in files),
    "files": files,
    "pii_sample": {"masked_values_reviewed": pii["n"], "personal": pii["personal"],
                   "synthetic_realistic": pii["synthetic_realistic"], "not_pii": pii["not_pii"],
                   "by_kind": pii.get("by_kind", {}),
                   "action": "IPv6 now needs 3+ groups (Python slices and cloze markers were matching); "
                             "version strings after 'Name/' are skipped"},
}
(ROOT / "artifacts/audits").mkdir(parents=True, exist_ok=True)
(ROOT / "artifacts/audits/false_positive_audit.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({f["reason"]: [f["sampled"], f["correct"], f["false_positive"]] for f in files}))
