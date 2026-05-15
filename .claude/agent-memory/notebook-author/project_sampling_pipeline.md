---
name: SentenceSplitter sampling pipeline context
description: Key facts about the annotation sampling pipeline — datasets, heuristic module layout, V1/V2 decision, execution environment, and current config
type: project
---

This project builds a Label Studio annotation batch from two Russian corpora using heuristic-based stratified sampling.

**Heuristic module:** `notebooks/sampling/heuristics/filters.py`
- Exports: `has_conjunction`, `has_asyndetic_markers`, `is_multilabel`, `token_length`, `count_finite_verbs`, `is_complex`
- `is_complex` in the module is **V1**: `has_conjunction OR has_asyndetic_markers`. V2 (finite-verb gating) was rejected — see `notebooks/sampling/heuristics/validation.md`.
- V1 has perfect recall (R=1.000) on Shirin-120; V2 sacrificed ~25% recall for a precision gain that was an artefact of NaN ground-truth rows.

**Datasets in `./data/`:**
- `presuicidal_train.parquet` (43,584 rows): columns `data_id, text, label`; multilabel sep = `;`
- `solyanka_train.parquet` (9,790 rows): columns `text, source, label, src_dataset`; multilabel sep = `,`; sources: lj, lenta, twitter, social_media
- `data.csv` (120 rows, Shirin annotation): columns include `text`, used for leakage reference
- No presuicidal test parquet locally — test split loaded from HuggingFace at runtime (`psytechlab/presuisidal_antisuisidal_dataset-master`, split="test")

**EDA thresholds (from `eda/*/summary.md`):**
- presuicidal long_tokens: 20 (p75)
- solyanka per source: lj=26, lenta=20, twitter=14, social_media=14

**Current notebook config (as of 2026-05-15 Igor Buyanov feedback):**
- `TARGET_N = 200` (was 1000)
- `OUTPUT_DIR = "./data"` (was `./notebooks/sampling/output`)
- 9 strata: original 8 rescaled by 0.95 + `"other": 0.05`
- STRATA_PROPORTIONS: conj=0.2375, bsp=0.1425, monolabel_long=0.190, short_simple=0.095, lenta_long=0.114, lj_long=0.095, social_media_multi=0.0475, twitter_short_simple=0.0285, other=0.050
- "other" pool: combined presuicidal + solyanka rows classified as `stratum=="other"`

**Output schema:** `text, source_dataset, source_subdomain, original_label, stratum, data_id`
- Output files: `./data/sample_for_annotation.{parquet,csv}`
- Old `notebooks/sampling/output/` directory deleted after 2026-05-15 run

**Notebook execution:** `nbclient` must be used directly — `~/.jupyter/jupyter_nbconvert_config.json` references `jupyter_contrib_nbextensions` (not in venv), crashing nbconvert. Use:
```python
import nbformat, nbclient
with open(nb_path) as f: nb = nbformat.read(f, as_version=4)
nbclient.NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
with open(nb_path, "w") as f: nbformat.write(nb, f)
```

**gitignore convention:** `data/*` with explicit un-ignores for `data.csv`, `sample_for_annotation.parquet`, `sample_for_annotation.csv`, `data/.gitkeep`. Train parquets stay gitignored.

**Why:** Label Studio annotation batch; case holder Igor Buyanov reduced sample to 200 and requested "other" safety-net stratum.
**How to apply:** When re-running or updating, use nbclient (not nbconvert), keep V1 is_complex, write outputs to `./data/`.
