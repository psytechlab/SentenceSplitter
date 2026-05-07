# Side-by-side comparison

| Metric | presuicidal | solyanka (overall) |
|---|---|---|
| N train rows | 43584 | 9790 |
| N test rows | 9777 | 2874 |
| Multilabel separator | ; | , |
| Multilabel ratio | 27.01% | 4.28% |
| Char length p50 / p75 / p95 | 75 / 117 / 236 | 77 / 115 / 178 |
| Token length p50 / p75 / p95 | 13 / 20 / 41 | 12 / 18 / 28 |
| Recommended long-sentence threshold (tokens, p75) | 20 | 18 |
| Test-train leakage (rows / % of test) | 0 / 0.00% | 0 / 0.00% |
| # unique labels | 46 | 7 |

## Recommendation

The presuicidal corpus uses a `;` multilabel separator and the solyanka corpus uses `,`; their multilabel densities and label cardinalities differ, so they sit in different sampling strata. Whichever dataset has the heavier multilabel ratio is better suited to the multi-clause sampling stratum (where coordinated/embedded structure is expected), while the lower-multilabel one is better used for the single-clause / short-sentence stratum. Per-source breakdown of solyanka should drive intra-source stratification — sources with longer p75 (e.g. lj/lenta if present) feed the long-sentence stratum, shorter ones (e.g. twitter) the short stratum.

The recommended p75 token thresholds above are the cut-points to feed into the sampling notebook's long-sentence filter — anything ≥ threshold goes into the "long" stratum, below into "short".
