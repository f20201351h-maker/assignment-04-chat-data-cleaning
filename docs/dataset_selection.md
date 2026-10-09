# Dataset selection

## What "like this" pointed at

The task's reference link is a model page (`lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled`).
Its card names its training data: `lordx64/reasoning-distill-opus-4-7-max-sft` (7,823 rows, ChatML text
with `<think>`) built from `lordx64/reasoning-distill-claude-opus-4-7-max` (8,124 rows), with prompts
taken from Crownelius / TeichAI / Delta-Vector Opus and Sonnet distill sets. That is the
example pool; I did not use any of it. By a rough bytes/4 estimate the example itself is
only about 8-11M tokens.

## How the search was done

* An LLM search pass (Claude Sonnet with Exa web search) read Hugging Face cards / viewer APIs for ~25 candidates in
  four groups (reasoning distills, real chat logs, agentic / function-calling, Indic). Its numbers were
  estimates (bytes / 4) from metadata only.
* I then downloaded and measured the two strongest options myself with the Qwen3 tokenizer, and re-checked
  licences and claims on the primary cards. Measured notes: `docs/notes/glaive_first_look.md`,
  `docs/notes/anudesh_first_look.md`.

## Candidates (short version)

Token figures marked "est." are bytes/4 estimates from that search pass; "measured" ones are mine.

| Dataset | Group | Size | Licence / access | Why not (or why) |
|---|---|---|---|---|
| lordx64 Opus 4.7 distill sets | reasoning | ~8-11M est. | - | The task's reference example. Excluded. |
| Gryphe/Opus-4.6-Reasoning-24k | reasoning | ~36M est. | apache-2.0 | Already MinHash-deduped and zero-width-cleaned by its author; overlaps the example's prompt pool |
| angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k | reasoning | ~17M (card) | apache-2.0 | Viewer rows (38,504) are overlapping variant files of 8,706 unique; overlaps Gryphe |
| Roman1111111/claude-sonnet-4.6-100000X-filtered | reasoning | ~160-180M est. | MIT on Claude output | Over range, already filtered, licence on model output unresolved |
| Roman1111111/claude-opus-4.6-10000x | reasoning | ~3-5M est. | MIT | Below 10M |
| bespokelabs/Bespoke-Stratos-17k | reasoning | ~67M est. | apache-2.0 | Good runner-up: literal `<|begin_of_thought|>` tags, MATH/AIME-derived problems. English only, one system prompt, one format |
| allenai/WildChat-1M (one shard) | real chat | ~100M est. per shard | odc-by | Real users, already Presidio-redacted; very little Indic text; at the top of the range |
| OpenAssistant/oasst2 | real chat | ~18M est. | apache-2.0 | Message trees with review labels; little Indic |
| Anthropic/hh-rlhf | real chat | ~37-74M est. | MIT | Card says not for SFT of dialogue agents |
| lmsys/lmsys-chat-1m | real chat | ~568M | gated agreement | Gated, no redistribution, over range |
| ShareGPT_Vicuna_unfiltered | real chat | ~168M est. | apache-2.0 label | Already heavily filtered despite the name; over range |
| **glaiveai/glaive-function-calling-v2** | agentic | **55.1M measured** | apache-2.0, open | **Picked**  |
| NousResearch/hermes-function-calling-v1 | agentic | ~15M est. | apache-2.0 | Partly a cleaned Glaive subset |
| Team-ACE/ToolACE | agentic | ~8-10M est. | apache-2.0 | Borderline size; already verified by its authors |
| interstellarninja/hermes_reasoning_tool_use | agentic | ~94M est. | apache-2.0 | Seeded from Glaive/ToolACE; schema already validated |
| Salesforce/xlam-function-calling-60k | agentic | ~24M est. | gated | Gated (accepting terms) |
| CohereLabs/aya_dataset | Indic/multi | ~38-47M est. | apache-2.0 | Strong option (human-written, labels per row); only partly Indic, labels already very clean |
| sarvamai/samvaad-hi-v1 | Indic | ~80-106M est. | apache-2.0 | Already used as a worked example in the course material |
| ai4bharat/indic-instruct-data-v0.1 | Indic | per config | no licence field | Licence unclear |
| maya-research/IndicVault | Indic | each config >100M est. | MIT | Synthetic, generator not stated |
| SandLogicTechnologies/Indic_Chat_Dataset | Indic | far over range | "Open Source" | Already deduped / filtered; vague licence |
| **ai4bharat/indic-align, Anudesh config** | Indic | **~38M measured (upper bound)** | cc-by-4.0, open | **Picked** (IndicAlign comes from IndicLLMSuite, AI4Bharat) |

## What I picked and why

**Two sources, cleaned as two shards of one SFT corpus:**

1. `glaiveai/glaive-function-calling-v2` at revision `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac`
   (Apache-2.0). 112,960 conversations, ~55M tokens. Synthetic function-calling chats; generator not
   stated on the card.
2. `ai4bharat/indic-align`, file `indicalign-instruct/anudesh/anudesh1.parquet` at revision
   `032b6a9070…` (CC-BY-4.0). 36,820 conversations. Crowd-sourced prompts with responses from
   Llama-2-70B-Chat (per the card).

No sampling was needed: each source is inside 10-100M tokens on its own, and together they are still
under 100M. The "subset" is simply: from IndicAlign (28 GB, 97M rows) take only the Anudesh config.

Reasons:

* **Planned SFT mix.** An earlier dataset plan of mine called for SFT data with agentic conversations
  and a 30% Indic share. Glaive covers tool calling and IndicAlign (from IndicLLMSuite) covers Indic
  instructions, so this is a small pilot of exactly that mix.
* **It is the ghost-tag situation.** The sources arrive in incompatible formats that must be unified
  at ingestion. Glaive writes speakers as literal text (`USER:`, `ASSISTANT:`,
  `FUNCTION RESPONSE:`, `<functioncall>`, `<|endoftext|>`); Anudesh stores nested `[prompt, response]`
  lists. Deduplicating across the two shards tests global rather than per-shard dedup.
* **The defects are real, not planted.** Measured before any cleaning: 96.5% of Glaive's function calls
  are invalid JSON, 8,805 exact duplicate rows, two different turn separators in one file, 356,885 literal
  `<|endoftext|>` strings; Anudesh has 2,628 legitimate ZWJ/ZWNJ that a careless cleaner would delete,
  337 null turns, greeting-only prompts.
* **Not too clean, not chosen for drama.** Neither source has been deduplicated or PII-scrubbed by its
  authors (cards say nothing about it). I did not pick the source with the biggest removal rate.

Things I checked and did *not* find, so I won't claim them:

* Anudesh's English prompts (about a quarter in a sample) are not a mislabel. The card does not say the
  prompts are Indic-only and lists `en` among the languages.
* Glaive's back-to-back assistant turns (8,293) are legitimate: a short preamble followed by the tool call.

## Licence and provenance notes that go into the manifest

* Glaive: Apache-2.0. The card is empty, so the generating model is unknown - recorded as a gap.
* Anudesh: CC-BY-4.0 requires attribution. The responses were generated by Llama-2-70B-Chat; the Llama 2
  licence restricts using its outputs to improve other LLMs, which matters for anyone training on this.
  Recorded as a terms note. I am not redistributing the cleaned files publicly either way.
