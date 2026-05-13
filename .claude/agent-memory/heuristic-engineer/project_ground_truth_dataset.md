---
name: ground-truth dataset shape and quirks
description: Schema and parsing quirks of ./data/data.csv — the 120-row human-annotated splitting dataset
type: project
---

`./data/data.csv` (120 rows, utf-8) is the ground-truth file for sentence-splitting heuristic validation.

Key columns: `text` (original sentence), `simple_sentences` (annotator's split), `label`.

`simple_sentences` parsing quirks — must be handled defensively:
- ~80% of rows: JSON object `{"text": ["clause1", "clause2", ...]}`. `len(parsed["text"]) > 1` is the positive class (complex).
- ~20% of rows (24/120 in May 2026 snapshot): NaN — annotator did not split → treat as 1-clause / simple / negative.
- A small number of rows: cell is a plain string (not JSON), e.g. just the original sentence verbatim → treat as 1-clause / simple.

Wrap `json.loads` in try/except `JSONDecodeError`, and check `pd.isna(cell)` before parsing. Pandas reads the CSV with Arrow-backed strings, so `isinstance(cell, float)` does NOT catch NaN — use `pd.isna`.

Class balance with this parsing rule: 94 positives / 26 negatives.

**Why:** Without the NaN/non-JSON fallback, validation crashes; misclassifying NaN as positive would invert the class balance.
**How to apply:** Reuse the parsing logic at `notebooks/sampling/heuristics/` validation drivers; never assume `simple_sentences` is well-formed JSON.
