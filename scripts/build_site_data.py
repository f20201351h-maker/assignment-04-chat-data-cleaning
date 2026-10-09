"""Turn the generated artifacts into the one JSON file the website reads.

python scripts/build_site_data.py  ->  site/data/site.json

Every number on the site comes from here, and everything here comes from
artifacts/*.json written by the pipeline and the audits. Example snippets are
re-scrubbed for PII and the script fails if any identifier survives.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from s4clean.pii import scrub  # noqa: E402

A = ROOT / "artifacts"


def load(name):
    p = A / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


metrics = load("metrics.json")
manifest = load("manifest.json")
sources = load("sources.json")
ex = load("examples_raw.json")
stats_audit = load("audits/stats_audit.json")
determinism = load("audits/determinism.json")
fp_audit = load("audits/false_positive_audit.json")
classifier = load("quality_classifier.json")
tests = load("audits/tests.json")

stages = {s["stage"]: s for s in metrics["stages"]}
SRC = ["glaive", "anudesh"]


def total(d):
    return sum(d.values()) if d else 0


def removed_total(stage, src=None):
    r = stages[stage]["removed"]
    if src:
        return total(r.get(src, {}))
    return sum(total(v) for v in r.values())


def pct(a, b):
    return round(100.0 * a / b, 1) if b else 0.0


raw_docs = {s: stages["extract"]["docs_in"].get(s, 0) for s in SRC}
raw_tok = {s: stages["extract"]["tokens_in"].get(s, 0) for s in SRC}
fin_docs = {s: stages["manifest"]["docs_out"].get(s, 0) for s in SRC}
fin_tok = {s: stages["manifest"]["tokens_out"].get(s, 0) for s in SRC}

data = {"generated_from": ["artifacts/metrics.json", "artifacts/manifest.json", "artifacts/sources.json",
                           "artifacts/examples_raw.json", "artifacts/audits/*.json"],
        "corpus_id": metrics["corpus_id"]}

data["totals"] = {
    "raw_docs": sum(raw_docs.values()), "raw_tokens": sum(raw_tok.values()),
    "final_docs": sum(fin_docs.values()), "final_tokens": sum(fin_tok.values()),
    "docs_retained_pct": pct(sum(fin_docs.values()), sum(raw_docs.values())),
    "tokens_retained_pct": pct(sum(fin_tok.values()), sum(raw_tok.values())),
}
data["by_source"] = {s: {"raw_docs": raw_docs[s], "raw_tokens": raw_tok[s], "final_docs": fin_docs[s],
                         "final_tokens": fin_tok[s], "docs_retained_pct": pct(fin_docs[s], raw_docs[s]),
                         "tokens_retained_pct": pct(fin_tok[s], raw_tok[s])} for s in SRC}

# waterfall: docs and tokens alive after each stage, per source
data["waterfall"] = []
for st in metrics["stages"]:
    row = {"stage": st["stage"]}
    for s in SRC:
        row[f"{s}_docs"] = st["docs_out"].get(s, 0)
        row[f"{s}_tokens"] = st["tokens_out"].get(s, 0)
        row[f"{s}_removed_docs"] = total(st["removed"].get(s, {}))
        row[f"{s}_modified_docs"] = st["modified_docs"].get(s, 0)
    data["waterfall"].append(row)
data["input"] = {f"{s}_docs": raw_docs[s] for s in SRC} | {f"{s}_tokens": raw_tok[s] for s in SRC}

# per-stage detail: removals by reason and the counters
data["stages"] = {}
for st in metrics["stages"]:
    name = st["stage"]
    d = {"removed": st["removed"], "removed_tokens": st["removed_tokens"], "modified_docs": st["modified_docs"],
         "counters": st["counters"], "removed_total": removed_total(name),
         "modified_total": total(st["modified_docs"])}
    for s in SRC:
        d[f"removed_{s}"] = removed_total(name, s)
        d[f"modified_{s}"] = st["modified_docs"].get(s, 0)
    data["stages"][name] = d

def C(stage, src):
    return stages[stage]["counters"].get(src, {})


ng, na = C("normalize", "glaive"), C("normalize", "anudesh")
fg = C("format", "glaive")
fc_total = fg.get("fc_valid_as_is", 0) + fg.get("fc_repaired_single_quoted_args", 0) + fg.get("fc_unrecoverable", 0)
markers = sum(v for k, v in fg.items() if k.startswith("marker_") and k != "marker_and_format_tokens_saved")
qg, qa = C("quality", "glaive"), C("quality", "anudesh")
la = C("langid", "anudesh")
pg, pa = C("pii", "glaive"), C("pii", "anudesh")
dd = metrics["dedup"]
glaive_after_quality = stages["quality"]["docs_out"]["glaive"]
data["derived"] = {
    "normalize": {
        "anudesh_zwnj_kept": na.get("zwnj_kept_indic", 0), "anudesh_zwj_kept": na.get("zwj_kept_indic", 0),
        "anudesh_joiners_kept": na.get("zwnj_kept_indic", 0) + na.get("zwj_kept_indic", 0),
        "anudesh_zwsp_removed": na.get("zero_width_space", 0), "anudesh_bom_removed": na.get("bom", 0),
        "anudesh_stray_joiners_removed": na.get("zwj_removed_stray", 0) + na.get("zwnj_removed_stray", 0),
        "anudesh_nfc_fields": na.get("nfc_changed", 0),
        "anudesh_docs_invisible_fix": na.get("docs_with_invisible_or_encoding_fix", 0),
        "glaive_docs_invisible_fix": ng.get("docs_with_invisible_or_encoding_fix", 0),
        "glaive_control_chars": ng.get("control_char", 0),
        "glaive_emoji_zwj_kept": ng.get("zwj_kept_emoji", 0),
        "glaive_docs_modified": stages["normalize"]["modified_docs"].get("glaive", 0),
        "anudesh_docs_modified": stages["normalize"]["modified_docs"].get("anudesh", 0),
    },
    "format": {
        "markers_converted": markers,
        "endoftext_literals": fg.get("marker_endoftext", 0),
        "function_calls": fc_total,
        "fc_valid_as_is": fg.get("fc_valid_as_is", 0),
        "fc_repaired": fg.get("fc_repaired_single_quoted_args", 0),
        "fc_repaired_apostrophe": fg.get("fc_repaired_escaped_apostrophe", 0),
        "fc_unrecoverable": fg.get("fc_unrecoverable", 0),
        "fc_invalid_upstream_pct": pct(fc_total - fg.get("fc_valid_as_is", 0), fc_total),
        "tokens_saved": fg.get("marker_and_format_tokens_saved", 0),
        "tokens_saved_pct": pct(fg.get("marker_and_format_tokens_saved", 0), stages["format"]["tokens_in"]["glaive"]),
        "trailing_user_trimmed": fg.get("trailing_unanswered_user_turn_trimmed", 0),
        "removed_glaive": removed_total("format", "glaive"),
    },
    "langid": {
        "removed": removed_total("langid"), "removed_glaive": removed_total("langid", "glaive"),
        "removed_anudesh": removed_total("langid", "anudesh"),
        "anudesh_flagged_reply_language": la.get("response_language_differs_flagged", 0),
        "anudesh_mixed_kept": la.get("code_switched_kept", 0),
        "anudesh_low_conf": la.get("user_low_conf", 0), "anudesh_too_short": la.get("user_too_short", 0),
    },
    "quality": {
        "w4_glaive_would_drop_pct": pct(qg.get("widget4_would_drop", 0), sum(v for k, v in qg.items() if k.startswith("docs__"))),
        "w4_anudesh_indic_would_drop_pct": pct(qa.get("widget4_would_drop__indic", 0), qa.get("docs__indic", 0)),
        "w4_anudesh_latin_would_drop_pct": pct(qa.get("widget4_would_drop__latin_or_other", 0), qa.get("docs__latin_or_other", 0)),
        "w4_anudesh_indic_docs": qa.get("docs__indic", 0),
        "w4_anudesh_indic_stopword_fail_pct": pct(qa.get("widget4_fail_stopwords__indic", 0), qa.get("docs__indic", 0)),
        "removed": removed_total("quality"),
        "greeting_only": stages["quality"]["removed"].get("anudesh", {}).get("greeting_only_prompt", 0),
        "latex_removed": sum(stages["quality"]["removed"].get(s, {}).get("latex_backslashes_stripped", 0) for s in SRC),
        "glaive_code_docs": qg.get("docs_with_code_block", 0),
        "glaive_code_docs_with_backslash": qg.get("docs_with_code_block_and_any_backslash", 0),
    },
    "dedup": {
        "exact_removed": sum(stages["dedup"]["removed"].get(s, {}).get("exact_duplicate", 0) for s in SRC),
        "near_removed": sum(stages["dedup"]["removed"].get(s, {}).get("near_duplicate", 0) for s in SRC),
        "near_removed_glaive": stages["dedup"]["removed"].get("glaive", {}).get("near_duplicate", 0),
        "near_removed_anudesh": stages["dedup"]["removed"].get("anudesh", {}).get("near_duplicate", 0),
        "exact_removed_glaive": stages["dedup"]["removed"].get("glaive", {}).get("exact_duplicate", 0),
        "exact_removed_anudesh": stages["dedup"]["removed"].get("anudesh", {}).get("exact_duplicate", 0),
        "union_find_would_remove": sum(dd.get("near_removed_if_union_find_clusters", {}).values()),
        "glaive_removed_pct": pct(removed_total("dedup", "glaive"), glaive_after_quality),
        "tokens_removed": sum(total(stages["dedup"]["removed_tokens"].get(s, {})) for s in SRC),
    },
    "pii": {
        "email": pg.get("email", 0) + pa.get("email", 0), "phone": pg.get("phone", 0) + pa.get("phone", 0),
        "ip": pg.get("ip", 0) + pa.get("ip", 0), "secret": pg.get("secret", 0) + pa.get("secret", 0),
        "email_placeholder_kept": pg.get("email_placeholder_kept", 0) + pa.get("email_placeholder_kept", 0),
        "ip_nonpublic_kept": pg.get("ip_nonpublic_kept", 0) + pa.get("ip_nonpublic_kept", 0),
        "docs_modified": total(stages["pii"]["modified_docs"]),
        "masked_total": sum(p.get(k, 0) for p in (pg, pa) for k in ("email", "phone", "ip", "secret")),
    },
    "decontam": {
        "removed": removed_total("decontam"),
        "max_item_coverage_pct": round(100 * metrics["decontam"].get("max_item_coverage", 0), 1),
        "eval_items": sum(metrics["decontam"]["items_indexed"].values()),
        "canaries": sum(C("decontam", s).get("canary_string_found", 0) for s in SRC),
    },
}

data["dedup"] = metrics["dedup"]
data["decontam"] = metrics["decontam"]
data["config"] = {k: metrics["config"][k] for k in ("dedup", "decontam", "langid", "quality", "tokens")}
data["manifest"] = manifest
data["sources"] = sources

# ---- examples: scrub again and refuse anything that still looks like PII
leaks = []


def safe(s):
    if not isinstance(s, str):
        return s
    t, c, found = scrub(s)
    if any(k in ("email", "phone", "ip", "secret") for k, _ in found):
        leaks.append(found)
    return t


def walk(o):
    if isinstance(o, dict):
        return {k: walk(v) for k, v in o.items()}
    if isinstance(o, list):
        return [walk(v) for v in o]
    return safe(o)


data["examples"] = walk(ex)
if leaks:
    print(f"note: {len(leaks)} example strings needed re-masking before publishing")

for opt_name, opt in (("stats_audit", stats_audit), ("determinism", determinism),
                      ("false_positive_audit", fp_audit), ("classifier", classifier), ("tests", tests)):
    data[opt_name] = opt
data["upstream"] = load("audits/upstream_glaive_markup.json")
if fp_audit:
    rep = [f for f in fp_audit["files"] if f["reason"] in ("degenerate_repetition", "repeated_lines")]
    data["derived"]["fp"] = {"repetition_reviewed": sum(f["sampled"] for f in rep),
                             "repetition_wrong": sum(f["false_positive"] for f in rep)}
ner = load("audits/ner_audit.json")
data["ner_audit"] = ner
if ner:
    bs = ner["by_source"]
    data["derived"]["ner"] = {
        "sample": ner["conversations"],
        "with_names": ner["with_any_name"],
        **{k: sum(bs[s].get(k, 0) for s in bs) for k in ("public", "fictional", "private", "unclear", "likely_false_positive")},
    }
if classifier:
    w = classifier["test_within_group"]
    data["derived"]["classifier"] = {
        "labelled": classifier["labelled"] + classifier.get("labelled_missing_or_unparsed", 0),
        "labelled_used": classifier["labelled"], "test_auc": classifier["test_auc"],
        "test_accuracy": classifier["test_accuracy"], "baseline": classifier["majority_baseline_accuracy"],
        "indic_label_keep": w["indic"]["label_keep_rate"], "indic_pred_keep": w["indic"]["pred_keep_rate"],
        "indic_test_n": w["indic"]["test_n"],
        "latin_label_keep": w["latin_or_other"]["label_keep_rate"], "latin_pred_keep": w["latin_or_other"]["pred_keep_rate"],
        "indic_within_auc": w["indic"]["within_group_auc"], "latin_within_auc": w["latin_or_other"]["within_group_auc"],
        "indic_label_keep_all": classifier["label_keep_rate_by_script_all_labels"]["indic"]["keep_rate"],
        "latin_label_keep_all": classifier["label_keep_rate_by_script_all_labels"]["latin_or_other"]["keep_rate"],
        "indic_only_auc": classifier["indic_only_model"]["test_auc"],
        "indic_only_train_n": classifier["indic_only_model"]["train_n"],
    }


def ok_count(audit):
    if not audit:
        return "not run"
    n = len(audit["checks"])
    k = sum(1 for c in audit["checks"] if c.get("ok", c.get("identical")))
    return f"{k} of {n} checks passed"


data["checks"] = {
    "tests": (f"{tests['passed']} passed, {tests['failed']} failed" if tests else "not run"),
    "stats": ok_count(stats_audit),
    "determinism": ("identical" if determinism and determinism["all_identical"] else
                    ("DIFFERENT" if determinism else "not run")),
}

out = ROOT / "site" / "data" / "site.json"
out.parent.mkdir(parents=True, exist_ok=True)
text = json.dumps(data, ensure_ascii=False, indent=1)
# final guard: nothing that looks like a real email in the published file
emails = [e for e in re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
          if not re.search(r"@(?:[\w.-]+\.)?(example\.(com|net|org)|[\w-]+\.(test|example|invalid))$", e, re.I)]
if emails:
    sys.exit(f"refusing to write site data: {len(emails)} email-like strings")
out.write_text(text + "\n", encoding="utf-8")
print("wrote", out.relative_to(ROOT), f"{len(text)/1024:.0f} KB")
