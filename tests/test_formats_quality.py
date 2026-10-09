import json

from s4clean.formats import ghost_markers, parse_glaive, parse_pairs, plain_text, render
from s4clean.quality import analyze, verdicts

SYS = "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n{}"


def test_glaive_markers_become_roles_and_args_are_repaired():
    chat = ("USER: What's the news in France?\n\n\n"
            "ASSISTANT: <functioncall> {\"name\": \"get_news\", \"arguments\": '{\"country\": \"France\"}'} <|endoftext|>\n\n\n"
            "FUNCTION RESPONSE: {\"headlines\": [\"a\", \"b\"]}\n\n\n"
            "ASSISTANT: Here are the headlines. <|endoftext|>\n\n\n")
    msgs, c = parse_glaive(SYS, chat)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant"]
    assert msgs[2]["tool_call"] == {"name": "get_news", "arguments": {"country": "France"}}
    assert c["fc_repaired_single_quoted_args"] == 1
    for m in msgs:
        assert ghost_markers(m.get("content", "")) == []


def test_glaive_two_newline_separator_rows_also_split():
    chat = "USER: Hi there, what is 2+2?\n\nASSISTANT: It is 4. <|endoftext|>\n\n"
    msgs, _ = parse_glaive("SYSTEM: You are a helpful assistant, with no access to external functions.", chat)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[-1]["content"] == "It is 4."


def test_glaive_preamble_before_function_call():
    chat = ("USER: find laptops\n\n\nASSISTANT: Sure, let me search.\n<functioncall> "
            "{\"name\": \"search\", \"arguments\": '{\"q\": \"laptop\"}'} <|endoftext|>\n\n\n")
    msgs, c = parse_glaive(SYS, chat)
    assert msgs[-1]["content"] == "Sure, let me search."
    assert msgs[-1]["tool_call"]["arguments"] == {"q": "laptop"}
    assert c["functioncall_after_preamble"] == 1


def test_unrecoverable_call_rejects_row():
    chat = "USER: x\n\n\nASSISTANT: <functioncall> {\"name\": \"f\", \"arguments\": '{broken'} <|endoftext|>"
    msgs, c = parse_glaive(SYS, chat)
    assert msgs is None and c["fc_unrecoverable"] == 1


def test_truncated_tool_response_is_counted():
    chat = "USER: lyrics?\n\n\nASSISTANT: <functioncall> {\"name\": \"l\", \"arguments\": '{}'}\n\n\nFUNCTION RESPONSE: {\"lyrics\": \"The"
    msgs, c = parse_glaive(SYS, chat)
    assert c["tool_response_invalid_json"] == 1


def test_pairs_null_turn_is_malformed():
    assert parse_pairs([["hi", None]])[0] is None
    msgs, _ = parse_pairs([["नमस्ते", "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?"]])
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_render_uses_one_token_per_boundary_and_single_end():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "", "tool_call": {"name": "f", "arguments": {}}}]
    out = render(msgs)
    assert out.startswith("<|user|>hi<|assistant|>") and out.endswith("<|end|>") and out.count("<|end|>") == 1
    assert json.loads(out[len("<|user|>hi<|assistant|>"):-len("<|end|>")]) == {"arguments": {}, "name": "f"}
    assert "hi" in plain_text(msgs)


def test_ghost_markers_detected_including_unescaped_variants():
    assert ghost_markers("text [USER] more <|im_start|> x") == ["[USER]", "<|im_start|>"]
    assert ghost_markers("line one\nASSISTANT: leaked") == ["ASSISTANT:"]
    assert ghost_markers("an ASSISTANT: in the middle of a line is not a marker") == []


def test_widget4_rules_pass_prose_and_fail_code():
    prose = ("Photosynthesis is the process by which a green plant converts light into chemical energy. "
             "In the leaves, the chlorophyll absorbs sunlight and drives a reaction that turns carbon dioxide "
             "and water into glucose and oxygen. The glucose stores energy for growth, and the oxygen is released. "
             "This is the base of nearly every food chain on Earth, and it keeps the air topped up with oxygen too.")
    assert all(verdicts(analyze(prose)).values())
    code = "def f(x):\n    return {k: v for k, v in x.items() if v[0] == '$'}\n" * 3
    fails = [k for k, ok in verdicts(analyze(code)).items() if not ok]
    assert "symratio" in fails and "stopwords" in fails


def test_escaped_apostrophe_inside_single_quoted_args_is_repaired():
    bs = chr(92)
    raw = ' {"name": "wc", "arguments": ' + "'" + '{"text": "It' + bs + "'" + 's amazing"}' + "'" + '} '
    from collections import Counter
    from s4clean.formats import parse_function_call
    c = Counter()
    assert parse_function_call(raw, c) == {"name": "wc", "arguments": {"text": "It's amazing"}}
    assert c["fc_repaired_escaped_apostrophe"] == 1


def test_looping_unit_catches_loops_not_tables_or_numbers():
    from s4clean.quality import looping_unit
    assert looping_unit("The total is 1" + ",000" * 30) == ",000"
    assert looping_unit("quad " * 40) == "quad "
    table = "| a | b |\n|---|---|\n" + "| x | y |\n" * 30
    assert looping_unit(table) is None
    assert looping_unit("The answer is 0.00000000000000000000000000000001.") is None
    assert looping_unit("Gift bag 1:\n- duck\nGift bag 2:\n- duck\nGift bag 3:\n- duck") is None
    assert looping_unit("```python\n" + "x = 1\n" * 50 + "```") is None
    assert looping_unit("नहीं " * 20) == "नहीं "


def test_pii_stage_drops_docs_that_become_identical_after_masking(tmp_path):
    import tomllib
    from pathlib import Path
    from s4clean import pipeline as P
    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "config" / "pipeline.toml").read_text(encoding="utf-8"))

    def mk(i, e):
        return {"source": "glaive", "kind": "glaive", "idx": i, "tok": 20, "lang": {"doc": "en"}, "quality": {},
                "messages": [{"role": "user", "content": f"Send the report to {e} please"},
                             {"role": "assistant", "content": "Done, the report was sent."}]}
    q = P.Quarantine(tmp_path)
    log = P.StageLog("pii")
    out = P.stage_pii(cfg, [mk(0, "a.kumar@gmail.com"), mk(1, "r.iyer@yahoo.com"), mk(2, "x@example.com")],
                      log, q, tmp_path / "f.jsonl", {})
    q.close()
    assert [r["idx"] for r in out] == [0, 2]
    assert log.removed["glaive"]["exact_duplicate_after_masking"] == 1
