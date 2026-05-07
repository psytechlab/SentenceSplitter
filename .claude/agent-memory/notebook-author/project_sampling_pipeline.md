---
name: SentenceSplitter sampling pipeline context
description: Key facts about the annotation sampling pipeline — datasets, heuristic module layout, V1/V2 decision, and execution environment
type: project
---

This project builds a Label Studio annotation batch from two Russian corpora using heuristic-based stratified sampling.

**Heuristic module:** `notebooks/sampling/heuristics/filters.py`
- Exports: `has_conjunction`, `has_asyndetic_markers`, `is_multilabel`, `token_length`, `count_finite_verbs`, `is_complex`, `classify_stratum`
- `filters.is_complex` is **V2** (requires >=2 finite verbs). V2 was rejected — always use V1 inline: `has_conjunction(text) OR has_asyndetic_markers(text)`

**V1 vs V2 decision:** Documented in `notebooks/sampling/heuristics/validation.md`. V1 has perfect recall on Shirin-120; V2 had lower recall (0.745) and its precision advantage was an artefact of NaN ground-truth rows being treated as negatives.

**Datasets present in `./data/`:**
- `presuicidal_train.parquet` (43,584 rows): columns `data_id, text, label`; multilabel sep = `;`
- `solyanka_train.parquet` (9,790 rows): columns `text, source, label, src_dataset`; multilabel sep = `,`; sources: lj, lenta, twitter, social_media
- `data.csv` (120 rows, Shirin annotation): columns include `text`, used for leakage reference
- No presuicidal test parquet in `./data/` — test-split leakage step must be skipped or `DATA_PATHS['test_leakage_ref']` filled in manually

**EDA thresholds (from `eda/*/summary.md`):**
- presuicidal long_tokens: 20 (p75)
- solyanka per source: lj=26, lenta=20, twitter=14, social_media=14

**Notebook execution:** `nbclient` must be used directly instead of `jupyter nbconvert` — the user's `~/.jupyter/jupyter_nbconvert_config.json` references `jupyter_contrib_nbextensions` (not installed), which crashes nbconvert. Use:
```python
from nbclient import NotebookClient
client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': '.'}})
client.execute()
```

**Output format:** Parquet + CSV to `notebooks/sampling/output/sample_for_annotation.{parquet,csv}`; schema: `text, source_dataset, source_subdomain, original_label, stratum, data_id`

**Why:** sampling pipeline for sentence splitting annotation; TARGET_N=1000 is a placeholder pending case-holder confirmation.
**How to apply:** When updating or re-running the sampling notebook, keep V1 in place, do not use `filters.is_complex`, and use nbclient to execute.
