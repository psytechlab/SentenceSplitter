---
name: V1/V2 heuristic baseline metrics on 120-row data.csv
description: Reference precision/recall for V1 (conj OR bsp) vs V2 (is_complex with finite-verb gate), under the three GT-handling views
type: project
---

`./data/data.csv` rows by status: 94 well-formed JSON, 24 NaN, 2 plain-string. Three GT-handling views:
- **All** (legacy): NaN + plain-string treated as 1-clause/simple → 94 pos / 26 neg.
- **Scenario A**: drop NaN rows → 96 rows, 94 pos / 2 neg (the 2 plain-string rows kept as simple).
- **Scenario B**: keep only well-formed JSON → 94 rows, 94 pos / 0 neg.

Metrics (May 2026):

| Variant | View | Precision | Recall | F1 |
|---|---|---|---|---|
| V1 (conj OR bsp) | All | 0.825 | 1.000 | 0.904 |
| V1 (conj OR bsp) | A | 1.000 | 1.000 | 1.000 |
| V1 (conj OR bsp) | B | 1.000 | 1.000 | 1.000 |
| V2 (is_complex)  | All | 0.959 | 0.745 | 0.838 |
| V2 (is_complex)  | A | 1.000 | 0.745 | 0.854 |
| V2 (is_complex)  | B | 1.000 | 0.745 | 0.854 |

**Key correction:** the V2-precision-win over V1 (0.959 vs 0.825 under "All") was an artefact of treating the 24 NaN rows as confirmed-simple. Under Scenario A/B both variants reach precision 1.000 because the cleaner GT contains essentially no negatives — V2's gate cannot be evaluated on this dataset for precision. Need annotated negatives to re-evaluate.

V2 FN breakdown under Scenario B (24 FNs total):
- **Zero-copula nominal** (no finite verb, no impersonal lemma): 4 (16.7%)
- **Impersonal / predicative-adverb** (no finite verb, predicative lemma present): 1 (4.2%)
- **Other** (≥1 finite verb, fails for other reasons — infinitive in 2nd clause, comparative "чем" not listed, single finite verb with multiple nominal clauses): 19 (79.2%)

**Why:** Future iterations should beat F1 0.854 (Scenario A/B) on recall, since precision is currently un-measurable on this GT. The "other" bucket dominates FNs and is the highest-ROI target for refinement, not zero-copula/impersonal.
**How to apply:** Before claiming a V2 precision improvement, ask the user to label the 24 NaN rows (or supply a fresh negatives set). For recall improvements, target the "other" cluster: comparative "чем", infinitive-as-second-clause-predicate, single-finite-verb sentences with comma-separated nominal clauses.
