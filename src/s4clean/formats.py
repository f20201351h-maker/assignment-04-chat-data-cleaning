"""Format discipline / the ghost-tag trap (Session 4, section 4 + widget 3).

Every source is rewritten at ingestion into ONE canonical conversation:

    {"messages": [{"role": "system"|"user"|"assistant"|"tool",
                   "content": str,
                   "tool_call": {"name": str, "arguments": {...}}   # assistant only, optional
                  }, ...]}

Role boundaries live in the structure, never as literal text inside content.
When the data is turned into training text, render() emits the tokenizer's
real special tokens, one per role boundary, and a single <|end|> (widget 3).

Literal markers that a source used as structure (Glaive's "USER:",
"ASSISTANT:", "FUNCTION RESPONSE:", "<functioncall>", "<|endoftext|>") are
converted into roles. Any marker still found INSIDE content afterwards is
counted as a residual ghost tag (widget 2: flag, never silently delete).
"""
from __future__ import annotations

import json
import re
from collections import Counter

CANON_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>", "tool": "<|tool|>"}
END = "<|end|>"

# widget 2 GHOST_RE plus the markers seen in the sources used here
GHOST_RE = re.compile(
    r"\[USER\]|\[ASSISTANT\]|\[SYSTEM\]|<\|endoftext\|>|<\|im_start\|>|<\|im_end\|>|</?(?:USER|ASSISTANT|SYSTEM)>"
    r"|<functioncall>|^(?:USER|ASSISTANT|SYSTEM|FUNCTION RESPONSE):|\[/?INST\]|<<SYS>>|### (?:Instruction|Response):",
    re.M,
)

# Glaive separates turns with 3 newlines in function-calling rows but 2 in the
# plain-chat rows, so accept 2 or more. A marker after a single newline is
# left in the content and shows up as a residual ghost tag for inspection.
_GLAIVE_TURN_RE = re.compile(r"(?:^|\n{2,})(USER|ASSISTANT|FUNCTION RESPONSE): ?")
_EOT = "<|endoftext|>"
# a few calls end with a stray extra "}" - accept one or more closing braces
_ARGS_STR_RE = re.compile(r'^\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*\'(.*)\'\s*\}+\s*$', re.S)


def ghost_markers(text: str) -> list[str]:
    return GHOST_RE.findall(text or "")


def parse_function_call(raw: str, c: Counter) -> dict | None:
    """Glaive writes {"name": "f", "arguments": '{"a": 1}'} - arguments is a
    single-quoted string, so the whole thing is not JSON. Recover the object."""
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        args = obj.get("arguments")
        if isinstance(args, str):
            obj["arguments"] = json.loads(args)
        c["fc_valid_as_is"] += 1
        return {"name": obj["name"], "arguments": obj.get("arguments") or {}}
    except Exception:
        pass
    m = _ARGS_STR_RE.match(raw)
    if m:
        inner = m.group(2)
        try:
            args = json.loads(inner)
            c["fc_repaired_single_quoted_args"] += 1
            return {"name": m.group(1), "arguments": args}
        except Exception:
            pass
        # inside the single-quoted string, apostrophes were escaped as \' which
        # is not a JSON escape ("It\'s"); undo that one layer and retry
        try:
            args = json.loads(inner.replace(chr(92) + "'", "'"))
            c["fc_repaired_single_quoted_args"] += 1
            c["fc_repaired_escaped_apostrophe"] += 1
            return {"name": m.group(1), "arguments": args}
        except Exception:
            c["fc_unrecoverable"] += 1
            return None
    c["fc_unrecoverable"] += 1
    return None


def parse_glaive(system: str, chat: str) -> tuple[list[dict] | None, Counter]:
    """Return (messages, counters). messages is None if the row is corrupt."""
    c: Counter = Counter()
    msgs: list[dict] = []
    sys_text = (system or "").strip()
    if sys_text.startswith("SYSTEM:"):
        sys_text = sys_text[len("SYSTEM:"):].strip()
        c["marker_SYSTEM"] += 1
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})

    parts = _GLAIVE_TURN_RE.split(chat or "")
    # parts = [prefix, ROLE, body, ROLE, body, ...]
    if parts[0].strip():
        c["text_before_first_marker"] += 1
    for k in range(1, len(parts), 2):
        role_raw, body = parts[k], parts[k + 1]
        c[f"marker_{role_raw.replace(' ', '_')}"] += 1
        body = body.strip()
        if body.endswith(_EOT):
            body = body[: -len(_EOT)].rstrip()
            c["marker_endoftext"] += 1
        if role_raw == "USER":
            msgs.append({"role": "user", "content": body})
        elif role_raw == "FUNCTION RESPONSE":
            try:
                json.loads(body)
            except Exception:
                c["tool_response_invalid_json"] += 1
            msgs.append({"role": "tool", "content": body})
        else:
            if "<functioncall>" in body:
                # usually the whole turn; a few turns have a short preamble first
                pre, _, call_raw = body.partition("<functioncall>")
                c["marker_functioncall"] += 1
                if pre.strip():
                    c["functioncall_after_preamble"] += 1
                call = parse_function_call(call_raw, c)
                if call is None:
                    return None, c
                msgs.append({"role": "assistant", "content": pre.strip(), "tool_call": call})
            else:
                msgs.append({"role": "assistant", "content": body})
    return msgs, c


def parse_pairs(interactions) -> tuple[list[dict] | None, Counter]:
    """IndicAlign style: [[prompt, response], ...]. None strings are malformed."""
    c: Counter = Counter()
    msgs: list[dict] = []
    for pair in interactions or []:
        if pair is None or len(pair) != 2:
            c["malformed_pair"] += 1
            return None, c
        q, a = pair
        if q is None or a is None:
            c["null_turn"] += 1
            return None, c
        msgs.append({"role": "user", "content": q.strip()})
        msgs.append({"role": "assistant", "content": a.strip()})
    return msgs, c


def render(messages: list[dict]) -> str:
    """Training-text view with the canonical special tokens (widget 3)."""
    out = []
    for m in messages:
        body = m.get("content", "")
        if m.get("tool_call"):
            body = (body + "\n" if body else "") + json.dumps(m["tool_call"], ensure_ascii=False, sort_keys=True)
        out.append(CANON_TOKENS[m["role"]] + body)
    return "".join(out) + END


def plain_text(messages: list[dict], roles=("system", "user", "assistant", "tool")) -> str:
    """Content only, no role tokens - used by language ID, quality, dedup, PII, decontam."""
    parts = []
    for m in messages:
        if m["role"] not in roles:
            continue
        if m.get("content"):
            parts.append(m["content"])
        if m.get("tool_call"):
            parts.append(json.dumps(m["tool_call"], ensure_ascii=False, sort_keys=True))
    return "\n\n".join(parts)
