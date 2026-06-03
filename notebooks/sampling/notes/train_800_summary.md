# Train-800 freelancer batch — summary

**Generated:** 2026-05-26
**Seed:** `42` (everywhere — split, sampling, all `df.sample` calls)
**Heuristics:** V1 — `is_complex = has_conjunction OR has_asyndetic_markers`
(from `notebooks/sampling/heuristics/filters.py`; V2 was rejected, see
`heuristics/validation.md`).

## Purpose

Prepare three artifacts for the next freelance-annotation cycle:

1. **`screening_set_50.csv`** — 50 rows, stratified subset of the already-annotated
   200; used as a trial task to screen freelancer candidates (we hold the ground-truth
   annotation already, so we can score their submissions).
2. **`test_set_150.csv`** — the remaining 150 of the original 200; held-out model-eval
   test set, never given to freelancers.
3. **`sample_for_annotation_train_800.csv`** (+ `.parquet`) — 800 freshly-sampled rows
   with the same stratification, given to freelancers for annotation. Zero overlap with
   any of the held-out sets.

## Source data

| File | Rows | Role |
|---|---:|---|
| `data/sample_for_annotation.csv` | 200 | already-annotated; split into screening + test |
| `data/data.csv` (Shirin-120) | 120 | leakage reference (excluded from train) |
| HF `psytechlab/presuisidal_antisuisidal_dataset-master` (test split) | 9,777 | leakage reference (excluded from train) |
| `data/presuicidal_train.parquet` | 43,584 | candidate pool A |
| `data/solyanka_train.parquet` | 9,790 | candidate pool B |

## Artifacts produced

| File | Rows | Size |
|---|---:|---:|
| `data/screening_set_50.csv` | 50 | 20 KB |
| `data/test_set_150.csv` | 150 | 61 KB |
| `data/sample_for_annotation_train_800.csv` | 800 | 305 KB |
| `data/sample_for_annotation_train_800.parquet` | 800 | 121 KB |

The earlier `data/sample_for_annotation_train_2000.{csv,parquet}` is a separate
artifact (Shirin batch) and was **not touched** by this run.

## Stratum distribution (rows per stratum)

| Stratum | screening (50) | test (150) | train (800) | source 200 total |
|---|---:|---:|---:|---:|
| presuicidal_multilabel_conj   | 12 | 36 | 190 | 48 |
| presuicidal_monolabel_long    |  9 | 29 | 152 | 38 |
| presuicidal_multilabel_bsp    |  7 | 21 | 114 | 28 |
| solyanka_lenta_long           |  6 | 17 |  91 | 23 |
| presuicidal_short_simple      |  5 | 14 |  76 | 19 |
| solyanka_lj_long              |  5 | 14 |  76 | 19 |
| other                         |  3 |  7 |  40 | 10 |
| solyanka_social_media_multi   |  2 |  7 |  38 |  9 |
| solyanka_twitter_short_simple |  1 |  5 |  23 |  6 |
| **TOTAL**                     | **50** | **150** | **800** | **200** |

Screening proportions are within ±1 row of an exact 25% slice of every stratum
(largest-remainders allocation). Train proportions match `STRATA_PROPORTIONS`
(the same ones from `sampling.ipynb`) exactly — every quota equals
`round(TARGET_N × proportion)` with the largest-remainder tie-break, with no
shortfall in any pool.

## Leakage guarantees (all verified)

| Check | Result |
|---|---|
| `screening ∪ test == original 200` | ✅ |
| `screening ∩ test == ∅` | ✅ (0) |
| `train_800 ∩ screening_set == ∅` | ✅ (0) |
| `train_800 ∩ test_set == ∅` | ✅ (0) |
| `train_800 ∩ sample_for_annotation == ∅` | ✅ (0; supersedes the previous two) |
| `train_800 ∩ data.csv (Shirin-120) == ∅` | ✅ (0) |
| `train_800 ∩ HF presuicidal test split == ∅` | ✅ (0) |
| `train_800` has 800 unique texts (no internal dupes) | ✅ |

In the train pipeline, exclusion is enforced **twice**:
1. `pre_df` and `sol_df` are filtered by `~text.isin(already_annotated_texts)`
   *before* stratum classification → all candidate pools are clean. Per-stratum
   exclusion counts confirm 141 presuicidal + 59 solyanka = 200 rows removed
   from the corpora (every annotated row matched exactly one corpus row).
2. After sampling, **Pass 4** runs a hard assertion that
   `set(combined_df.text) ∩ already_annotated_texts == ∅`. If the upstream
   filter ever fails, this catches it.

## Token-length stats (train 800)

| stat | value |
|---|---:|
| mean | 19.6 |
| median | 18 |
| p25 / p75 | 10 / 26 |
| min / max | 2 / 141 |

## Source-dataset mix (train 800)

| source_dataset | rows |
|---|---:|
| presuicidal | 560 |
| solyanka | 240 |

## Reproduction

From the repo root, with the project's `.venv` active:

```bash
# Step 1 — split 200 into 50 screening + 150 test
python scripts/prepare_screening_and_test.py

# Step 2 — sample 800 train rows (excludes all 200 annotated rows)
python - <<'PY'
import nbformat, os
from nbclient import NotebookClient
nb_path = "notebooks/sampling/sampling_train_800.ipynb"
nb = nbformat.read(nb_path, as_version=4)
NotebookClient(nb, timeout=900, kernel_name="python3",
               resources={"metadata": {"path": os.getcwd()}}).execute()
nbformat.write(nb, nb_path)
PY
```

Both steps are deterministic at `SEED=42`. The notebook needs network access
on first run to fetch the HF presuicidal test split (used for the leakage
check); subsequent runs reuse the local HF cache.

## Files

- Sampling notebook: `notebooks/sampling/sampling_train_800.ipynb`
- Split script: `scripts/prepare_screening_and_test.py`
- Heuristics module: `notebooks/sampling/heuristics/filters.py`
- Conjunction table: `notebooks/sampling/heuristics/conjunctions.json`
- Strata logic & thresholds: `notebooks/sampling/sampling.ipynb` (unchanged)
