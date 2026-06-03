---
name: SentenceSplitter project context
description: Core facts about the Russian sentence-splitting NLP project — goals, key artifacts, dataset, and open issues
type: project
---

Goal: build a sampling notebook and annotation pipeline for splitting complex Russian sentences into clauses.
Domain: social-media mental-health text (Russian).

Key artifacts:
- `notebooks/Split_llm.ipynb` — LLM-based splitter using Qwen2.5-7B-Instruct via HuggingFace `transformers.pipeline`. Input: single Russian sentence. Output: `{"simple_parts": [...]}`. Runs in Colab; requires GPU and HF_TOKEN.
- `notebooks/sampling/heuristics/conjunctions.json` — extracted conjunction/marker table (SSP/SPP/BSP) from annotation guide.
- `notebooks/sampling/notes/edge_cases.md` — include/exclude rules for sampling, derived from GitHub issue #5.
- `notebooks/sampling/notes/llm_splitter_notes.md` — audit of Split_llm.ipynb for reuse in simple-sentence validation.

Annotation instruction Google Doc: https://docs.google.com/document/d/1Qs6AVKL_Fi3dovBxS_Ad_mP3VMZK71cq_4MguGIY2hY

GitHub repo (upstream/reference): psytechlab/SentenceSplitter

Issue #5 (opened 2026-04-19, @Astromis, assigned @sh-alanova): data exploration task — find overlooked patterns, inspect target and related datasets. Key edge case: simple sentences with multi-polarity enumerations need splitting despite lacking a subordinating conjunction. No comments on issue at time of retrieval.

**Why:** Understand project scope before recommending datasets or heuristics.
**How to apply:** Tailor dataset and heuristic recommendations to social-media Russian; flag enumeration-only splits as a known edge case.
