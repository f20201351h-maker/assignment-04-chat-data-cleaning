# ai4bharat/indic-align, subset indicalign-instruct/anudesh (measured 2026-10-01)

Revision 032b6a9070 (short sha at inspection time), CC-BY-4.0, ungated. File anudesh1.parquet, 42.2 MB.

- 36,820 rows: id, interactions (list of [prompt, response]), num_turns. No per-row language field.
- ~38.0M Qwen3 tokens (3,000-row sample, JSON incl. keys, so a slight overestimate)
- 337 null strings inside interactions (missing prompt or response)
- detected prompt language on a 3,000 sample: en 758, und 592 (too short), hi 464, mr 406, bn 210,
  te 201, kn 174, ta 147, mixed 14, hi-Latn 12, ml 7, gu 6
- invisibles in whole file: ZWNJ 2,141, ZWJ 487 (legit joiners), ZWSP 58, BOM 16 (noise)
- 465 exact duplicate conversations; 1,640 repeated first prompts (top: "hi" 282, "Hi" 271, "Hello" 76)
- no literal chat markers in the text (structured format)
- ids unique
