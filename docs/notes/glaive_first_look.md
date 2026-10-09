# glaiveai/glaive-function-calling-v2 - first look (measured 2026-10-01)

Revision e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac, Apache-2.0, ungated, one 271.2 MB JSON array.

- 112,960 rows, fields `system` + `chat` (both plain strings with literal role markers)
- ~55.1M Qwen3 tokens (5,000-row random sample, seed 0, avg 487.8 tokens/row, extrapolated)
- literal markers: ASSISTANT: 356,885 / <|endoftext|> 356,885 / USER: 266,599 / SYSTEM: 112,960 /
  FUNCTION RESPONSE: 83,992 / <functioncall> 83,834
- 8,805 exact duplicate rows (system+chat md5)
- 36,536 distinct system prompts; top one ("no access to external functions") on 34,598 rows
- function calls: 2,922 valid JSON, 80,906 invalid - all because "arguments" is a single-quoted string
- 12,610 boilerplate refusal phrases
- same-role consecutive turns: ASSISTANT->ASSISTANT 8,293 (preamble + tool call, legitimate), USER->USER 2
- 1,997 chats not ending in <|endoftext|>: 1,599 end on an unanswered USER turn, 398 end inside a
  truncated FUNCTION RESPONSE
