# Labelling rubric for the quality classifier (layer 2)

Two-layer recipe: a large model labels a sample, a light model imitates the labels at scale.
The labeller sees one conversation at a time (system prompt removed, PII already masked) and gives
one integer score for **how useful this conversation is as supervised fine-tuning data for an
assistant**, plus a reason of at most 12 words.

| Score | Meaning |
|---|---|
| 5 | Correct, helpful, well-formed; a good example to teach from |
| 4 | Good, minor issues (slightly generic, small formatting slip) |
| 3 | Acceptable; nothing wrong enough to drop it |
| 2 | Weak: partly wrong, off-topic, padded, or the answer is in the wrong language for the question |
| 1 | Bad: mostly wrong, broken or garbled text, fabricated nonsense, does not answer |
| 0 | Unusable: empty, gibberish, harmful, or only boilerplate |

Rules for the labeller:

* Judge the content, not the language. An answer in Telugu, Hindi or any Indic language is judged on
  the same scale as English. Romanised or code-mixed text is fine if it matches how the user wrote.
* Tool-calling conversations: a correct function call with sensible arguments counts as a good answer.
  Function responses are simulated; judge whether the assistant used them correctly.
* A polite refusal is fine (3+) when the request is genuinely outside the assistant's tools; it is weak
  (2) when the assistant could have answered.
* Do not reward length. A short correct answer can be a 5.
* Placeholders like [EMAIL] or [PHONE] are masking, not a defect.

Output format, one JSON object per line: `{"id": "...", "score": 0-5, "why": "..."}`
