# Cleaning two chat datasets for SFT

A CPU-only data-cleaning pipeline run over two public chat datasets, plus a static report site that shows what every stage removed or changed. Built as ERA V5 Session 4 assignment.

The two sources are cleaned as two shards of one SFT corpus:

| Shard | Source | Licence | Conversations in → out | Tokens in → out |
|---|---|---|---:|---:|
| Glaive | `glaiveai/glaive-function-calling-v2` (synthetic English function-calling chats) | Apache-2.0 | 112,960 → 73,539 | 55.9M → 40.6M |
| Anudesh | `ai4bharat/indic-align`, Anudesh config (crowd-written Indic + English prompts, Llama-2-70B-Chat answers) | CC-BY-4.0 | 36,820 → 35,819 | 36.0M → 35.9M |
| Total | | | 149,780 → 109,358 (73.0%) | 91.9M → 76.5M (83.3%) |

Token counts use the Qwen3 tokenizer at a pinned revision. All numbers on the site and in this table come from `artifacts/metrics.json`.

## What the pipeline does

Nine strategies, one module each in `src/s4clean/`: extraction, text normalization, format discipline (ghost tags), language ID and validation, quality filtering, deduplication, PII removal, decontamination, and a reproducibility manifest. How I arrived at nine, and what each one is for, is in [`docs/cleaning_strategies.md`](docs/cleaning_strategies.md).

Findings that shaped the pipeline:

* **Deduplication did most of the removing.** Glaive's generator repeats the same scenarios: 9,366 exact and 29,194 near duplicates were removed (MinHash + LSH candidates, confirmed with true Jaccard). A conversation is removed only if it is a near copy of a conversation that is kept. Plain union-find clustering would have removed 33,716.
* **Glaive's tool calls are broken upstream.** 96.5% of its function calls are not valid JSON because `arguments` is a single-quoted string. I decode them back into JSON objects. Glaive also uses two different turn separators in one file and stores literal `USER:` / `ASSISTANT:` / `<|endoftext|>` markers, which are parsed into roles.
* **Standard heuristic quality rules are biased against Indic text.** Run as written (Gopher/C4 style), they would drop 98.6% of Anudesh's Indic-script answers but 21.1% of its Latin-script answers. The stop-word rule alone fails 98.3% of Indic answers. I kept those rules as an audit only and used chat-specific rules instead (greeting-only prompts, empty replies, looping output, LaTeX broken upstream; 1,008 removals).
* **A small quality classifier learned "Indic script = low quality".** An LLM (Claude Sonnet, following [`docs/notes/labelling_rubric.md`](docs/notes/labelling_rubric.md)) scored 600 conversations, and a logistic-regression model on hashed character n-grams imitated the scores with test AUC 0.798. On held-out Indic conversations the labels keep 58.5% and the classifier keeps 4.9%. It removes nothing; its scores are kept as an audit.
* **Decontamination hits were phrases, not leaks.** 13 conversations shared a 13-gram with the test splits of GSM8K, MATH-500, HumanEval or MMLU (16,008 test items). I read all of them: none is a copied test question (highest single-item coverage 25.7%). They were removed anyway.

## Layout

```
config/pipeline.toml        every threshold in one place (its hash goes into the manifest)
src/s4clean/                cleaning code, one module per strategy
  normalize.py  formats.py  langid.py  quality.py  classifier.py
  dedup.py  fasthash.py  pii.py  decontam.py  manifest.py  tokens.py  pipeline.py
scripts/                    download, run, audits, site data
tests/                      unit tests
artifacts/                  generated metrics, manifest, sources, examples and audits (committed)
docs/                       strategy inventory, dataset selection, first-look notes, labelling rubric, reproduction
site/                       static report (index.html, style.css, app.js, data/site.json)
data/                       raw, cleaned and quarantined conversations (not committed)
```

## Running it

Full instructions are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md). In short:

```bash
python scripts/download.py
python scripts/run_pipeline.py
python scripts/audit_stats.py
python scripts/check_determinism.py
python scripts/run_tests.py
python scripts/build_site_data.py
node site/build.mjs          # or: npm run build; npm run serve to preview
```

The pipeline is CPU-only. On an 8-core laptop with 8 GB RAM a full run takes about 20 minutes. To view the report without rerunning anything, `node site/build.mjs` builds `site/dist` from the committed `site/data/site.json`.

## Verification

* Unit tests: 45 passed (`artifacts/audits/tests.json`).
* Statistics recomputed from the output files: 35/35 checks (`artifacts/audits/stats_audit.json`). Every input row is in a clean shard or exactly one quarantine file, manifest hashes match the shard files, and token counts recomputed from scratch match the manifest.
* Determinism: a second full run produced an identical corpus id, shard hashes and stage counters (`artifacts/audits/determinism.json`).
* False positives: an LLM reviewer read up to 25 removals per (stage, reason), 275 conversations and 110 masked PII values in all, on an earlier run. Several rules were changed because of it (`artifacts/audits/false_positive_audit.json`). The two rules added afterwards were hand-checked on the next run.

## Limitations

* "Deduplicated" means within these two shards only.
* "Decontaminated" means no 13-gram overlap with the GSM8K, MATH-500, HumanEval and MMLU test splits. MILU was not checked (gated).
* The classifier labels and the false-positive and name reviews are LLM judgements (Claude Sonnet), spot-checked by me. They are recorded with their method in `artifacts/`.
* Person names were audited on a 150-conversation sample, not masked.
* Glaive's upstream export stripped almost every backslash. LaTeX answers that obviously broke were removed. Code answers that silently lost an escape sequence (`\n`, `\d`) cannot be found reliably and are still in the corpus.
* The cleaned conversations are not published. Anudesh's answers were generated by Llama-2-70B-Chat, and the Llama 2 licence restricts using its outputs to improve other models.
* The cleaning code (`src/s4clean/`, `scripts/run_pipeline.py`) and `config/pipeline.toml` are byte-identical to the versions that produced the artifacts, because the manifest records their sha256. That is why their docstrings and the manifest's contributor field still reference the course sections they were written against.
