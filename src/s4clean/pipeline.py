"""The Session 4 pipeline, run end to end on the two source shards.

Stage order follows widget 1, with format discipline (lesson section 4) as
its own stage right after normalisation:

  1 extract   2 normalize   3 format   4 langid   5 quality
  6 dedup     7 pii         8 decontam 9 manifest

Every stage records what came in, what was removed (with a reason), what was
modified, and how many tokens survived. Removed records are written to
data/quarantine/<stage>.jsonl so they can be audited by hand.
"""
from __future__ import annotations

import json
import re
import time
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import dedup as D
from . import formats as F
from . import langid as L
from . import manifest as M
from . import normalize as N
from . import pii as P
from . import quality as Q
from .decontam import EvalIndex, has_canary
from .fasthash import shingle_set_u32
from .tokens import count_tokens

ROOT = Path(__file__).resolve().parents[2]
STAGES = ["extract", "normalize", "format", "langid", "quality", "dedup", "pii", "decontam", "manifest"]
INVISIBLE_KEYS = {"zero_width_space", "bom", "word_joiner", "soft_hyphen", "bidi_control", "replacement_char",
                  "control_char", "private_use", "zwj_removed_stray", "zwnj_removed_stray", "odd_space",
                  "nfc_changed", "html_entity"}
KEPT_KEYS = {"zwj_kept_indic", "zwnj_kept_indic", "zwj_kept_emoji"}
INDIC_SCRIPTS = {"Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu", "Kannada", "Malayalam"}

GREETINGS = {"hi", "hii", "hiii", "hello", "hey", "helo", "hlo", "hy", "namaste", "namaskar", "namaskaram",
             "vanakkam", "hai", "good", "morning", "evening", "afternoon", "night", "there", "sir", "bro",
             "नमस्ते", "नमस्कार", "हाय", "हेलो", "हैलो", "নমস্কার", "হ্যালো", "నమస్కారం", "హాయ్", "హలో",
             "வணக்கம்", "ஹாய்", "ನಮಸ್ಕಾರ", "ಹಾಯ್", "നമസ്കാരം", "ഹായ്", "નમસ્તે", "ਸਤ", "ਸ੍ਰੀ", "ਅਕਾਲ"}
_WORD = re.compile(r"\w+", re.UNICODE)
# LaTeX commands whose backslash is missing (Glaive's export stripped backslashes)
LATEX_STRIPPED_RE = re.compile(
    r"(?<![\\\w])(?:begin|end|section\*?|subsection\*?|textbf|textit|emph|usepackage|documentclass|frac|sqrt|"
    r"hline|caption|includegraphics|maketitle|toprule|midrule|bottomrule|multicolumn)\{")
_CODE = re.compile(r"```.*?(?:```|\Z)|`[^`\n]+`", re.S)


# --------------------------------------------------------------------- helpers
class StageLog:
    def __init__(self, name: str):
        self.name = name
        self.docs_in: Counter = Counter()
        self.docs_out: Counter = Counter()
        self.removed: dict = defaultdict(Counter)  # source -> reason -> docs
        self.removed_tokens: dict = defaultdict(Counter)  # source -> reason -> tokens
        self.modified: Counter = Counter()  # source -> docs changed in place
        self.counters: dict = defaultdict(Counter)  # source -> free-form counters
        self.tokens_in: Counter = Counter()
        self.tokens_out: Counter = Counter()
        self.seconds = 0.0

    def to_dict(self) -> dict:
        def plain(d):
            return {k: dict(v) if isinstance(v, Counter) else v for k, v in d.items()}

        return {
            "stage": self.name,
            "docs_in": dict(self.docs_in),
            "docs_out": dict(self.docs_out),
            "removed": plain(self.removed),
            "removed_tokens": plain(self.removed_tokens),
            "modified_docs": dict(self.modified),
            "counters": plain(self.counters),
            "tokens_in": dict(self.tokens_in),
            "tokens_out": dict(self.tokens_out),
        }


class Quarantine:
    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder
        self.files: dict = {}

    def write(self, stage: str, rec: dict, reason: str, detail=None) -> None:
        if stage not in self.files:
            self.files[stage] = open(self.folder / f"{stage}.jsonl", "w", encoding="utf-8", newline="\n")
        out = {"source": rec["source"], "src_index": rec["idx"], "reason": reason, "detail": detail}
        if "messages" in rec and rec["messages"] is not None:
            out["messages"] = rec["messages"]
        else:
            out["raw"] = rec.get("raw")
        self.files[stage].write(json.dumps(out, ensure_ascii=False) + "\n")

    def close(self) -> None:
        for f in self.files.values():
            f.close()


def raw_text(rec: dict) -> str:
    r = rec["raw"]
    if rec["kind"] == "glaive":
        return (r.get("system") or "") + "\n\n" + (r.get("chat") or "")
    return "\n\n".join((q or "") + "\n\n" + (a or "") for q, a in (r.get("pairs") or []))


def canonical_tokens(records: list[dict]) -> list[int]:
    """Content tokens + one special token per role boundary + one <|end|>."""
    texts, owners = [], []
    for i, rec in enumerate(records):
        for m in rec["messages"]:
            body = m.get("content", "")
            if m.get("tool_call"):
                body = (body + "\n" if body else "") + json.dumps(m["tool_call"], ensure_ascii=False, sort_keys=True)
            texts.append(body)
            owners.append(i)
    counts = count_tokens(texts)
    out = [len(rec["messages"]) + 1 for rec in records]
    for i, c in zip(owners, counts):
        out[i] += c
    return out


def role_text(messages, roles) -> str:
    return "\n\n".join(m.get("content", "") for m in messages if m["role"] in roles and m.get("content"))


def strip_code(text: str) -> str:
    return _CODE.sub(" ", text)


def is_greeting(text: str, max_words: int) -> bool:
    w = _WORD.findall(text.lower())
    return 0 < len(w) <= max_words and all(x in GREETINGS for x in w)


def snippet_around(text: str, pos: int, width: int = 60) -> str:
    a = max(0, pos - width)
    return text[a : pos + width]


# ---------------------------------------------------------------------- stages
def stage_extract(cfg, log: StageLog, q: Quarantine) -> list[dict]:
    recs = []
    for src in cfg["sources"]:
        shard, kind, path = src["shard"], src["kind"], ROOT / src["path"]
        if kind == "glaive":
            rows = json.loads(path.read_text(encoding="utf-8"))
            for i, r in enumerate(rows):
                recs.append({"source": shard, "kind": kind, "idx": i,
                             "raw": {"system": r.get("system"), "chat": r.get("chat")}})
        else:
            for i, r in enumerate(pq.read_table(path).to_pylist()):
                recs.append({"source": shard, "kind": kind, "idx": i,
                             "raw": {"pairs": r.get("interactions"), "id": r.get("id")}})
    toks = count_tokens([raw_text(r) for r in recs])
    keep = []
    for rec, t in zip(recs, toks):
        rec["tok"] = t
        log.docs_in[rec["source"]] += 1
        log.tokens_in[rec["source"]] += t
        reason = None
        if rec["kind"] == "glaive":
            if not isinstance(rec["raw"]["chat"], str) or not rec["raw"]["chat"].strip():
                reason = "empty_chat"
        else:
            pairs = rec["raw"]["pairs"]
            if not pairs:
                reason = "no_interactions"
            elif any(p is None or len(p) != 2 for p in pairs):
                reason = "malformed_pair"
            elif any(x is None or not str(x).strip() for p in pairs for x in p):
                reason = "null_or_empty_turn"
        if reason:
            log.removed[rec["source"]][reason] += 1
            log.removed_tokens[rec["source"]][reason] += t
            q.write("extract", rec, reason)
            continue
        keep.append(rec)
        log.docs_out[rec["source"]] += 1
        log.tokens_out[rec["source"]] += t
    return keep


def stage_normalize(cfg, recs, log: StageLog, examples: dict) -> list[dict]:
    want = {"zwj_kept_indic": 3, "zwnj_kept_indic": 3, "zero_width_space": 3, "bom": 2, "html_entity": 3,
            "nfc_changed": 2, "control_char": 2, "replacement_char": 2, "zwj_removed_stray": 2, "bidi_control": 2}
    got: dict = defaultdict(list)
    for rec in recs:
        s = rec["source"]
        log.docs_in[s] += 1
        log.tokens_in[s] += rec["tok"]
        changed_any = False
        invisible_changed = False
        r = rec["raw"]
        fields = [("system", r["system"]), ("chat", r["chat"])] if rec["kind"] == "glaive" else \
            [(f"p{i}{j}", x) for i, p in enumerate(r["pairs"]) for j, x in enumerate(p)]
        new_vals = {}
        for name, val in fields:
            if not val:
                new_vals[name] = val
                continue
            out, c = N.normalize_text(val, unescape_html=cfg["normalize"]["unescape_html"])
            log.counters[s].update(c)
            if out != val:
                changed_any = True
                if any(k in INVISIBLE_KEYS for k in c):
                    invisible_changed = True
            for k in c:
                if k in want and len(got[k]) < want[k] and (k in KEPT_KEYS or out != val):
                    pos = _first_pos(val, k)
                    got[k].append({"source": s, "src_index": rec["idx"], "kind": k,
                                   "before": snippet_around(val, pos), "after_doc_excerpt": _after_excerpt(out, val, pos)})
            new_vals[name] = out
        if rec["kind"] == "glaive":
            rec["raw"] = {"system": new_vals["system"], "chat": new_vals["chat"]}
        else:
            pairs = []
            for i, p in enumerate(r["pairs"]):
                pairs.append([new_vals[f"p{i}0"], new_vals[f"p{i}1"]])
            rec["raw"] = {"pairs": pairs, "id": r.get("id")}
        if changed_any:
            log.modified[s] += 1
        if invisible_changed:
            log.counters[s]["docs_with_invisible_or_encoding_fix"] += 1
        log.docs_out[s] += 1
    toks = count_tokens([raw_text(r) for r in recs])
    for rec, t in zip(recs, toks):
        rec["tok"] = t
        log.tokens_out[rec["source"]] += t
    examples["normalize"] = got
    return recs


def _first_pos(text: str, key: str) -> int:
    probes = {"zwj_kept_indic": "\u200d", "zwnj_kept_indic": "\u200c", "zwj_removed_stray": "\u200d",
              "zero_width_space": "\u200b", "bom": "\ufeff", "replacement_char": "\ufffd"}
    ch = probes.get(key)
    if ch and ch in text:
        return text.index(ch)
    if key == "html_entity":
        m = N.ENTITY_RE.search(text)
        return m.start() if m else 0
    if key == "bidi_control":
        for i, x in enumerate(text):
            if x in N.BIDI:
                return i
    if key == "control_char":
        for i, x in enumerate(text):
            if N._is_noise_char(x) == "control_char":
                return i
    if key == "nfc_changed":
        import unicodedata

        for i, x in enumerate(text):
            if unicodedata.combining(x):
                return i
    return 0


def _after_excerpt(out: str, before: str, pos: int) -> str:
    # same neighbourhood in the cleaned text (positions shift slightly; good enough to show)
    a = max(0, pos - 60)
    return out[a : pos + 60]


def stage_format(cfg, recs, log: StageLog, q: Quarantine, examples: dict) -> list[dict]:
    keep = []
    fmt_examples = []
    for rec in recs:
        s = rec["source"]
        log.docs_in[s] += 1
        log.tokens_in[s] += rec["tok"]
        if rec["kind"] == "glaive":
            msgs, c = F.parse_glaive(rec["raw"]["system"], rec["raw"]["chat"])
        else:
            msgs, c = F.parse_pairs(rec["raw"]["pairs"])
        log.counters[s].update(c)
        reason, detail = None, None
        if msgs is None:
            reason = "unrecoverable_function_call" if rec["kind"] == "glaive" else "malformed_pairs"
        else:
            resid = []
            for m in msgs:
                resid += F.ghost_markers(m.get("content", ""))
            if resid:
                reason, detail = "residual_ghost_marker_in_content", sorted(set(resid))
            elif any(m["role"] == "tool" and not _json_ok(m["content"]) for m in msgs):
                bad = next(m["content"] for m in msgs if m["role"] == "tool" and not _json_ok(m["content"]))
                reason = "tool_response_truncated" if not bad.rstrip().endswith(("}", "]")) else "tool_response_invalid_json"
            else:
                trimmed = 0
                while msgs and msgs[-1]["role"] == "user":
                    msgs.pop()
                    trimmed += 1
                if trimmed:
                    log.counters[s]["trailing_unanswered_user_turn_trimmed"] += 1
                    log.modified[s] += 1
                has_user = any(m["role"] == "user" and m["content"] for m in msgs)
                has_asst = any(m["role"] == "assistant" and (m["content"] or m.get("tool_call")) for m in msgs)
                if not (has_user and has_asst):
                    reason = "no_complete_exchange"
        if reason:
            rec["messages"] = msgs
            log.removed[s][reason] += 1
            log.removed_tokens[s][reason] += rec["tok"]
            q.write("format", rec, reason, detail)
            continue
        if rec["kind"] == "glaive" and len(fmt_examples) < 2 and any(m.get("tool_call") for m in msgs) \
                and c.get("fc_repaired_single_quoted_args"):
            fmt_examples.append({"source": s, "src_index": rec["idx"], "raw_chat": rec["raw"]["chat"][:900],
                                 "canonical": F.render(msgs)[-900:], "messages": msgs})
        rec["messages"] = msgs
        rec["raw_tok"] = rec["tok"]
        del rec["raw"]
        keep.append(rec)
    toks = canonical_tokens(keep)
    for rec, t in zip(keep, toks):
        log.counters[rec["source"]]["marker_and_format_tokens_saved"] += rec["raw_tok"] - t
        rec["tok"] = t
        del rec["raw_tok"]
        log.docs_out[rec["source"]] += 1
        log.tokens_out[rec["source"]] += t
    examples["format"] = fmt_examples
    return keep


def _json_ok(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def _base(lang: str) -> str:
    return "hi" if lang == "hi-Latn" else lang


def stage_langid(cfg, recs, log: StageLog, q: Quarantine, examples: dict) -> list[dict]:
    lc = cfg["langid"]
    claims = {s["shard"]: set(s["claimed_languages"]) for s in cfg["sources"]}
    tr = re.compile(lc["translation_request_pattern"])
    keep, ex = [], defaultdict(list)
    for rec in recs:
        s = rec["source"]
        log.docs_in[s] += 1
        log.tokens_in[s] += rec["tok"]
        u = role_text(rec["messages"], ("user",))
        a = role_text(rec["messages"], ("assistant",))
        du, da = L.detect(u, lc["min_letters"]), L.detect(a, lc["min_letters"])
        conf_u = du["status"] == "ok" and du["conf"] >= lc["min_conf"]
        conf_a = da["status"] == "ok" and da["conf"] >= lc["min_conf"]
        doc_lang = du["lang"] if conf_u else (da["lang"] if conf_a else ("mixed" if "mixed" in (du["status"], da["status"]) else "und"))
        rec["lang"] = {"user": du["lang"], "assistant": da["lang"], "doc": doc_lang,
                       "user_status": du["status"], "assistant_status": da["status"]}
        log.counters[s][f"user_{du['status']}"] += 1
        reason, detail = None, None
        if lc["quarantine_outside_claim"]:
            if conf_u and _base(du["lang"]) not in claims[s]:
                reason, detail = "user_language_outside_source_claim", du["lang"]
            elif conf_a and _base(da["lang"]) not in claims[s]:
                reason, detail = "assistant_language_outside_source_claim", da["lang"]
        if not reason and conf_u and conf_a and _base(du["lang"]) != _base(da["lang"]):
            if tr.search(u):
                log.counters[s]["lang_switch_translation_request_kept"] += 1
            elif lc["quarantine_prompt_response_mismatch"]:
                reason, detail = "response_in_different_language", f"{du['lang']}->{da['lang']}"
            else:
                log.counters[s]["response_language_differs_flagged"] += 1
                rec.setdefault("flags", []).append(f"lang:{du['lang']}->{da['lang']}")
                if len(ex["response_language_differs_flagged"]) < 6:
                    ex["response_language_differs_flagged"].append(
                        {"source": s, "src_index": rec["idx"], "detail": f"{du['lang']}->{da['lang']}",
                         "user": u[:200], "assistant": a[:200]})
        if du["status"] == "mixed" or da["status"] == "mixed":
            log.counters[s]["code_switched_kept"] += 1
            if len(ex["mixed"]) < 3:
                ex["mixed"].append({"source": s, "src_index": rec["idx"], "user": u[:200], "scripts": du["scripts"]})
        if reason:
            log.removed[s][reason] += 1
            log.removed_tokens[s][reason] += rec["tok"]
            q.write("langid", rec, reason, detail)
            if len(ex[reason]) < 4:
                ex[reason].append({"source": s, "src_index": rec["idx"], "detail": detail, "user": u[:240], "assistant": a[:240]})
            continue
        keep.append(rec)
        log.docs_out[s] += 1
        log.tokens_out[s] += rec["tok"]
    examples["langid"] = ex
    return keep


def stage_quality(cfg, recs, log: StageLog, q: Quarantine, examples: dict) -> list[dict]:
    qc = cfg["quality"]
    keep, ex = [], defaultdict(list)
    for rec in recs:
        s = rec["source"]
        log.docs_in[s] += 1
        log.tokens_in[s] += rec["tok"]
        msgs = rec["messages"]
        users = [m["content"] for m in msgs if m["role"] == "user"]
        asst = [m for m in msgs if m["role"] == "assistant"]
        prose = strip_code("\n\n".join(m["content"] for m in asst if m["content"]))
        # --- audit: the nine widget-4 rules exactly as taught, on the assistant prose
        audit = Q.verdicts(Q.analyze(prose))
        fails = [k for k, ok in audit.items() if not ok]
        hist = L.script_histogram(prose)
        indic_share = sum(v for k, v in hist.items() if k in INDIC_SCRIPTS) / max(1, sum(hist.values()))
        script = "indic" if indic_share > 0.3 else "latin_or_other"
        for k in fails:
            log.counters[s][f"widget4_fail_{k}"] += 1
            log.counters[s][f"widget4_fail_{k}__{script}"] += 1
        if fails:
            log.counters[s]["widget4_would_drop"] += 1
            log.counters[s][f"widget4_would_drop__{script}"] += 1
        log.counters[s][f"docs__{script}"] += 1
        asst_all = "\n".join(m["content"] for m in asst)
        if "```" in asst_all:
            log.counters[s]["docs_with_code_block"] += 1
            if chr(92) in asst_all:
                log.counters[s]["docs_with_code_block_and_any_backslash"] += 1
        # --- applied rules
        reason, detail = None, None
        if len(users) == 1 and is_greeting(users[0], qc["greeting_max_words"]):
            reason, detail = "greeting_only_prompt", users[0][:40]
        elif any(not m["content"] and not m.get("tool_call") for m in asst):
            reason = "empty_assistant_turn"
        elif len(LATEX_STRIPPED_RE.findall("\n".join(m["content"] for m in asst))) >= 2:
            reason, detail = "latex_backslashes_stripped", LATEX_STRIPPED_RE.findall("\n".join(m["content"] for m in asst))[:4]
        else:
            for m in asst:
                unit = Q.looping_unit(m["content"], qc["loop_min_repeats"])
                if unit:
                    reason, detail = "looping_output", unit[:12]
                    break
        rec["quality"] = {"widget4_fails": fails}
        if reason:
            log.removed[s][reason] += 1
            log.removed_tokens[s][reason] += rec["tok"]
            q.write("quality", rec, reason, detail)
            if len(ex[reason]) < 4:
                ex[reason].append({"source": s, "src_index": rec["idx"], "detail": detail,
                                   "user": users[0][:200] if users else "", "assistant": (asst[0]["content"] if asst else "")[:400]})
            continue
        keep.append(rec)
        log.docs_out[s] += 1
        log.tokens_out[s] += rec["tok"]
    examples["quality"] = ex
    return keep


def stage_dedup(cfg, recs, log: StageLog, q: Quarantine, examples: dict) -> tuple[list[dict], dict]:
    dc = cfg["dedup"]
    for rec in recs:
        log.docs_in[rec["source"]] += 1
        log.tokens_in[rec["source"]] += rec["tok"]
    # exact: whole canonical conversation, system prompt included
    first: dict = {}
    survivors, exact_groups = [], Counter()
    cross_exact = 0
    for rec in recs:
        key = D.exact_key(F.render(rec["messages"]))
        if key in first:
            kept = first[key]
            exact_groups[key] += 1
            if kept["source"] != rec["source"]:
                cross_exact += 1
            log.removed[rec["source"]]["exact_duplicate"] += 1
            log.removed_tokens[rec["source"]]["exact_duplicate"] += rec["tok"]
            q.write("dedup", rec, "exact_duplicate", {"kept": [kept["source"], kept["idx"]]})
            continue
        first[key] = rec
        survivors.append(rec)
    # near: shingles of the conversation without the system prompt
    mh = D.MinHasher(dc["num_perm"], seed=dc["minhash_seed"])
    sh = [shingle_set_u32(F.plain_text(r["messages"], ("user", "assistant", "tool")), dc["shingle_words"]) for r in survivors]
    sigs = np.stack([mh.signature(x) for x in sh]) if sh else np.empty((0, dc["num_perm"]), dtype=np.uint64)
    res = D.near_duplicates(sh, sigs, dc["bands"], dc["rows"], dc["jaccard_threshold"])
    # applied rule: remove a doc only if it is a verified near copy of a doc we KEEP
    drop = D.greedy_keep(len(survivors), res.verified_pairs)
    cross_clusters = 0
    sizes = Counter()
    for cl in res.clusters:
        srcs = {survivors[i]["source"] for i in cl}
        if len(srcs) > 1:
            cross_clusters += 1
        sizes[min(len(cl), 50)] += 1
    keep = []
    for k, rec in enumerate(survivors):
        if k in drop:
            kept_i, tj = drop[k]
            kept = survivors[kept_i]
            log.removed[rec["source"]]["near_duplicate"] += 1
            log.removed_tokens[rec["source"]]["near_duplicate"] += rec["tok"]
            q.write("dedup", rec, "near_duplicate", {"kept": [kept["source"], kept["idx"]], "jaccard": round(tj, 3)})
            continue
        keep.append(rec)
        log.docs_out[rec["source"]] += 1
        log.tokens_out[rec["source"]] += rec["tok"]
    # local (per-source) view for the global-vs-local comparison: same rule,
    # but only pairs whose two documents come from the same source
    same_src = [p for p in res.verified_pairs if survivors[p[0]]["source"] == survivors[p[1]]["source"]]
    within = Counter(survivors[i]["source"] for i in D.greedy_keep(len(survivors), same_src))
    union_find_removed = Counter()
    for cl in res.clusters:
        for i in cl[1:]:
            union_find_removed[survivors[i]["source"]] += 1
    info = {
        "near_removed_applied_rule": dict(Counter(survivors[i]["source"] for i in drop)),
        "near_removed_if_union_find_clusters": dict(union_find_removed),
        "exact_duplicate_groups": len(exact_groups),
        "exact_cross_source": cross_exact,
        "near_candidate_pairs": res.candidate_pairs,
        "near_verified_pairs": len(res.verified_pairs),
        "near_rejected_pairs": len(res.rejected_pairs),
        "near_clusters": len(res.clusters),
        "near_cluster_size_hist": {str(k): v for k, v in sorted(sizes.items())},
        "near_clusters_spanning_sources": cross_clusters,
        "near_removed_if_each_source_deduped_alone": dict(within),
        "lsh_threshold_formula": round(D.lsh_threshold(dc["bands"], dc["rows"]), 4),
        "largest_cluster": max((len(c) for c in res.clusters), default=0),
    }
    # pair examples for the site: some just above threshold, some high, some rejected
    def pair_ex(i, j, tj, est):
        a, b = survivors[i], survivors[j]
        return {"a": [a["source"], a["idx"]], "b": [b["source"], b["idx"]], "jaccard": round(tj, 3),
                "minhash_estimate": round(est, 3),
                "a_text": F.plain_text(a["messages"], ("user", "assistant", "tool"))[:500],
                "b_text": F.plain_text(b["messages"], ("user", "assistant", "tool"))[:500]}

    vp = sorted(res.verified_pairs, key=lambda x: x[2])
    rp = sorted(res.rejected_pairs, key=lambda x: -x[2])
    examples["dedup"] = {
        "near_threshold": [pair_ex(*p) for p in vp[:6]],
        "high": [pair_ex(*p) for p in vp[-3:]],
        "rejected_close": [pair_ex(*p) for p in rp[:4]],
        "largest_clusters": [
            {"size": len(c), "head": [survivors[c[0]]["source"], survivors[c[0]]["idx"]],
             "head_text": F.plain_text(survivors[c[0]]["messages"], ("user", "assistant"))[:300]}
            for c in sorted(res.clusters, key=len, reverse=True)[:5]],
    }
    # all verified Jaccards (for the histogram) and rejected ones
    info["verified_jaccard_hist"] = _hist([p[2] for p in res.verified_pairs])
    info["rejected_jaccard_hist"] = _hist([p[2] for p in res.rejected_pairs])
    return keep, info


def _hist(vals, edges=(0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0000001)):
    h = np.histogram(np.array(vals, dtype=float), bins=np.array(edges))[0] if vals else [0] * (len(edges) - 1)
    return {f"{edges[i]:.2f}-{min(edges[i + 1], 1.0):.2f}": int(h[i]) for i in range(len(edges) - 1)}


def _scrub_obj(obj, c: Counter, found: list):
    if isinstance(obj, str):
        t, cc, f = P.scrub(obj)
        c.update(cc)
        found.extend(f)
        return t
    if isinstance(obj, dict):
        return {k: _scrub_obj(v, c, found) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_obj(v, c, found) for v in obj]
    return obj


def stage_pii(cfg, recs, log: StageLog, q: Quarantine, audit_path: Path, examples: dict) -> list[dict]:
    audit = open(audit_path, "w", encoding="utf-8", newline="\n")
    ex = []
    for rec in recs:
        s = rec["source"]
        log.docs_in[s] += 1
        log.tokens_in[s] += rec["tok"]
        c: Counter = Counter()
        found: list = []
        new = []
        for m in rec["messages"]:
            m2 = dict(m)
            m2["content"] = _scrub_obj(m.get("content", ""), c, found)
            if m.get("tool_call"):
                m2["tool_call"] = _scrub_obj(m["tool_call"], c, found)
            new.append(m2)
        log.counters[s].update(c)
        if found:
            log.modified[s] += 1
            for kind, val in found:
                audit.write(json.dumps({"source": s, "src_index": rec["idx"], "kind": kind, "value": val},
                                       ensure_ascii=False) + "\n")
            if len(ex) < 6 and any(k in ("email", "phone", "ip") for k, _ in found):
                masked = F.plain_text(new, ("user", "assistant", "tool"))
                pos = min((masked.find(t) for t in ("[EMAIL]", "[PHONE]", "[IP]") if t in masked), default=0)
                ex.append({"source": s, "src_index": rec["idx"], "kinds": sorted({k for k, _ in found}),
                           "masked_excerpt": masked[max(0, pos - 160): pos + 160]})
        rec["messages"] = new
    audit.close()
    # Two conversations that differed only in an email or phone number become
    # identical once both are masked. Exact dedup ran before masking (course
    # order), so catch those here; the doc id must stay unique.
    seen: dict = {}
    keep = []
    for rec in recs:
        key = D.exact_key(F.render(rec["messages"]))
        if key in seen:
            kept = seen[key]
            log.removed[rec["source"]]["exact_duplicate_after_masking"] += 1
            log.removed_tokens[rec["source"]]["exact_duplicate_after_masking"] += rec["tok"]
            q.write("pii", rec, "exact_duplicate_after_masking", {"kept": [kept["source"], kept["idx"]]})
            continue
        seen[key] = rec
        keep.append(rec)
    toks = canonical_tokens(keep)
    for rec, t in zip(keep, toks):
        rec["tok"] = t
        log.docs_out[rec["source"]] += 1
        log.tokens_out[rec["source"]] += t
    examples["pii"] = ex
    return keep


def load_eval_index(cfg) -> tuple[EvalIndex, dict]:
    idx = EvalIndex(n=cfg["decontam"]["ngram"])
    ev = ROOT / "data" / "eval"
    for i, r in enumerate(pq.read_table(ev / "gsm8k_test.parquet").to_pylist()):
        idx.add("gsm8k", i, r["question"])
    with open(ev / "math500_test.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            idx.add("math500", i, json.loads(line)["problem"])
    for i, r in enumerate(pq.read_table(ev / "humaneval_test.parquet").to_pylist()):
        idx.add("humaneval", i, r["prompt"])
    for i, r in enumerate(pq.read_table(ev / "mmlu_test.parquet").to_pylist()):
        idx.add("mmlu", i, r["question"] + " " + " ".join(r["choices"]))
    info = {"items_indexed": idx.items_indexed, "items_skipped_too_short": idx.items_skipped_short,
            "fingerprints": idx.fingerprints, "ngram": idx.n}
    return idx, info


def stage_decontam(cfg, recs, log: StageLog, q: Quarantine, examples: dict) -> tuple[list[dict], dict]:
    idx, info = load_eval_index(cfg)
    keep, ex, all_hits = [], [], []
    by_bench = Counter()
    for rec in recs:
        s = rec["source"]
        log.docs_in[s] += 1
        log.tokens_in[s] += rec["tok"]
        text = F.plain_text(rec["messages"], ("user", "assistant", "tool"))
        if has_canary(text):
            log.counters[s]["canary_string_found"] += 1
        hits = idx.scan(text)
        if hits:
            for b, *_ in hits:
                by_bench[b] += 1
            best = max(hits, key=lambda h: h[3])
            all_hits.append({"source": s, "src_index": rec["idx"], "benchmark": best[0], "item": best[1],
                             "shared_13grams": best[2], "item_coverage": best[3]})
            log.removed[s]["eval_overlap"] += 1
            log.removed_tokens[s]["eval_overlap"] += rec["tok"]
            q.write("decontam", rec, "eval_overlap", hits[:5])
            if len(ex) < 8:
                ex.append({"source": s, "src_index": rec["idx"], "hits": hits[:3],
                           "user": role_text(rec["messages"], ("user",))[:300]})
            continue
        keep.append(rec)
        log.docs_out[s] += 1
        log.tokens_out[s] += rec["tok"]
    info["docs_with_overlap_by_benchmark"] = dict(by_bench)
    info["hits"] = all_hits
    info["max_item_coverage"] = max((h["item_coverage"] for h in all_hits), default=0.0)
    examples["decontam"] = ex
    return keep, info


# ---------------------------------------------------------------- the manifest
def code_hashes() -> list[dict]:
    files = sorted((ROOT / "src" / "s4clean").glob("*.py")) + [ROOT / "scripts" / "run_pipeline.py"]
    return [{"name": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": M.code_hash(p)} for p in files]


def stage_manifest(cfg, cfg_bytes: bytes, recs, log: StageLog, sources_meta: dict, out_dir: Path) -> dict:
    clean = ROOT / "data" / "clean"
    clean.mkdir(parents=True, exist_ok=True)
    scripts = code_hashes()
    shards = []
    by_src = defaultdict(list)
    for rec in recs:
        by_src[rec["source"]].append(rec)
    for src in cfg["sources"]:
        s = src["shard"]
        rows = sorted(by_src[s], key=lambda r: r["idx"])
        log.docs_in[s] += len(rows)
        log.tokens_in[s] += sum(r["tok"] for r in rows)
        path = clean / f"{s}.jsonl"
        lang_tok: Counter = Counter()
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                out = {"id": M.doc_id(F.render(r["messages"])), "source": s, "src_index": r["idx"],
                       "messages": r["messages"], "lang": r["lang"], "quality": r["quality"], "tokens": r["tok"],
                       "flags": r.get("flags", [])}
                lang_tok[r["lang"]["doc"]] += r["tok"]
                f.write(M.canonical_json(out) + "\n")
        total = sum(lang_tok.values())
        meta = next(x for x in sources_meta["sources"] if x["shard"] == s)
        man = {
            "shard_id": None,
            "source": f"https://huggingface.co/datasets/{meta['repo']}",
            "source_revision": meta["revision"],
            "source_file": meta["file"],
            "source_file_sha256": meta["sha256"],
            "license": meta["license"],
            "terms_note": meta["terms_note"],
            "contributor": cfg["run"]["contributor"],
            "upstream_contributor": meta["contributor"],
            "cleaning_scripts": scripts,
            "config_sha256": M.sha256_bytes(cfg_bytes.replace(b"\r\n", b"\n")),
            "content_sha256": M.sha256_file(path),
            "docs": len(rows),
            "token_count": total,
            "tokenizer": f"{cfg['tokens']['repo']}@{cfg['tokens']['revision']}",
            "languages": {k: round(v / total, 4) for k, v in sorted(lang_tok.items(), key=lambda kv: -kv[1])} if total else {},
        }
        status, problems = M.manifest_status(man)
        man["status"] = status
        man["problems"] = problems
        man["shard_id"] = "shard_" + man["content_sha256"][:12]
        shards.append(man)
        log.docs_out[s] += len(rows) if status == "clean" else 0
        log.tokens_out[s] += total if status == "clean" else 0
    corpus_id = M.sha256_bytes("".join(sh["content_sha256"] for sh in shards).encode())[:16]
    return {"corpus_id": "corpus_" + corpus_id, "shards": shards}


# --------------------------------------------------------------------- driver
def run(config_path: Path, out_dir: Path) -> dict:
    cfg_bytes = config_path.read_bytes()
    cfg = tomllib.loads(cfg_bytes.decode("utf-8"))
    sources_meta = json.loads((ROOT / "artifacts" / "sources.json").read_text(encoding="utf-8"))
    q = Quarantine(ROOT / "data" / "quarantine")
    (ROOT / "data" / "audit").mkdir(parents=True, exist_ok=True)
    logs = {s: StageLog(s) for s in STAGES}
    examples: dict = {}
    timings = {}

    def timed(name, fn, *a):
        t = time.time()
        r = fn(*a)
        timings[name] = round(time.time() - t, 1)
        print(f"[{name}] done in {timings[name]}s", flush=True)
        return r

    recs = timed("extract", stage_extract, cfg, logs["extract"], q)
    recs = timed("normalize", stage_normalize, cfg, recs, logs["normalize"], examples)
    recs = timed("format", stage_format, cfg, recs, logs["format"], q, examples)
    recs = timed("langid", stage_langid, cfg, recs, logs["langid"], q, examples)
    recs = timed("quality", stage_quality, cfg, recs, logs["quality"], q, examples)
    recs, dedup_info = timed("dedup", stage_dedup, cfg, recs, logs["dedup"], q, examples)
    recs = timed("pii", stage_pii, cfg, recs, logs["pii"], q, ROOT / "data" / "audit" / "pii_findings.jsonl", examples)
    recs, decon_info = timed("decontam", stage_decontam, cfg, recs, logs["decontam"], q, examples)
    manifest = timed("manifest", stage_manifest, cfg, cfg_bytes, recs, logs["manifest"], sources_meta, out_dir)
    q.close()

    metrics = {
        "corpus_id": manifest["corpus_id"],
        "stages": [logs[s].to_dict() for s in STAGES],
        "dedup": dedup_info,
        "decontam": decon_info,
        "config": cfg,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # examples are captured before the PII stage, so scrub them before they touch disk
    examples = _scrub_obj(examples, Counter(), [])
    (out_dir / "examples_raw.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False, default=str) + "\n",
                                                encoding="utf-8")
    (out_dir / "run_info.json").write_text(json.dumps({"stage_seconds": timings,
                                                       "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2) + "\n",
                                           encoding="utf-8")
    return metrics
