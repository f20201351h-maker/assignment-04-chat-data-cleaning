# The nine cleaning strategies

The course material I worked from (ERA V5, Session 4) lists the cleaning pipeline twice, and the two lists differ by one item each way:

* An eight-stage pipeline map: Extract, Normalize, Language ID, Quality filter, Deduplicate, PII scrub, Decontaminate, Manifest. Ghost markers sit inside Normalize.
* An eight-item closing list: normalization, format discipline, quality filtering, deduplication, language validation, PII removal, decontamination, manifest. Extraction is left out because it was covered earlier.

Format discipline is treated as its own topic in the material (its own section and example), and extraction is the first stage of the pipeline map, so I counted both. The union is **9 strategies**. I did not count sub-mechanisms separately: the nine heuristic rules, exact vs near dedup, local vs global dedup, regex vs NER, and canary strings each belong to one stage. Counting them would push the number past 20 without adding a new cleaning decision.

My pipeline follows the eight-stage order, with format discipline right after normalization. Cleaning happens before any content hash is computed.

| # | Strategy | Problem it solves | How it works | Without it |
|---|---|---|---|---|
| 1 | Extraction | Raw HTML carries nav bars, cookie banners, footers | Pull the main text out of the page (e.g. trafilatura rather than naive tag stripping) | Menus and legal text become "content" |
| 2 | Text normalization | Same character encoded several ways; invisible junk; HTML entities; stray whitespace | NFC; strip ZWSP, BOM, bidi controls, C0/C1 controls, soft hyphen, private-use, U+FFFD; unescape entities; collapse whitespace; **keep ZWJ/ZWNJ** for Brahmic scripts | A byte-level tokenizer learns garbage tokens; stripping all invisibles breaks Indic words |
| 3 | Format discipline (ghost-tag trap) | Every source marks speakers its own way (`[USER]`, `<USER>`, `### Instruction:`, `<\|im_start\|>`) | Pick one canonical format with real special tokens and rewrite every source into it at ingestion | The model learns fake markers as text, which then fight the real special tokens at SFT; tokens wasted on markers |
| 4 | Language ID and validation | Folder or metadata labels are wrong; code-switched text | Detect language per document at runtime, compare with the claimed label, quarantine mismatches, flag mixed documents; validate language codes strictly (`te` vs `tel`) | Per-language pools and fertility numbers are wrong |
| 5 | Quality filtering | Spam, list walls, SEO pages, broken documents | Layer 1: heuristic rules (mean word length 3-10, symbol ratio < 0.10, lines ending in punctuation >= 0.30, duplicate lines < 0.30, repetition, >= 2 stop words, bullet lines < 0.90, ellipsis lines < 0.30, 50-100k words). Layer 2: a cheap classifier trained on LLM labels | Junk eats compute; but English-tuned rules wrongly drop Indic text |
| 6 | Deduplication | Exact copies and near copies | Exact hash, then near-dup: word shingles -> MinHash signatures -> LSH bands, threshold (1/b)^(1/r). Global, not per shard | Wasted compute and memorisation; per-shard dedup misses cross-shard copies |
| 7 | PII removal | Emails, phones, IPs, usernames, names | Regex layer for structured identifiers; NER layer for names, with a precision/recall trade-off (Indic names and places get wrongly masked) | The model memorises personal data; legal risk |
| 8 | Decontamination | Benchmark test items leaking into training | Fingerprint eval sets, scan shards for n-gram overlap, remove hits; canary strings for after-the-fact detection | Benchmarks measure memorisation, not ability |
| 9 | Reproducibility and provenance manifest | Copy-pasted sizes, non-deterministic IDs, unknown licence, wrong token estimates | Deterministic pipeline; content-derived IDs; per-shard manifest with source, licence, contributor, script names + code hashes, sha256 of cleaned content, token count, languages; no manifest, no ingest | The corpus cannot be reproduced or audited; `words x 1.3` token estimates are off by 2-10x for Indic text |

What each strategy actually did on the two datasets is on the site (sections 1 and 4), with numbers generated from `artifacts/metrics.json`.
