# solyanka — EDA summary

Source: `psytechlab/solyanka_emotion_dataset`

## Row counts
- train: 9790
- test: 2874
- sources observed: ['lj', 'lenta', 'twitter', 'social_media']

## Schema
- columns: ['text', 'source', 'label', 'src_dataset']
- text column: `text`
- label column: `label`
- source column: `source`

## Char length (overall)
- count: 9790
- min: 3
- max: 609
- mean: 84.45
- median: 77.00
- p25: 47.00
- p50: 77.00
- p75: 115.00
- p90: 152.00
- p95: 178.00
- p99: 209.00


## Token length (overall, whitespace split)
- count: 9790
- min: 1
- max: 94
- mean: 13.45
- median: 12.00
- p25: 8.00
- p50: 12.00
- p75: 18.00
- p90: 24.00
- p95: 28.00
- p99: 36.00


## Multilabel (overall)
- separator: `,`
- multilabel rows: 419 / 9790
- multilabel ratio: 4.28%
- unique labels (after split): 7

## Top 30 labels
| label | count |
|---|---|
| no_emotion | 3391 |
| joy | 2043 |
| sadness | 1760 |
| surprise | 930 |
| anger | 929 |
| fear | 884 |
| disgust | 273 |

## Per-source breakdown

| source | rows | char p50 | char p75 | char p95 | tok p50 | tok p75 | tok p95 | multilabel % | rec p75 tokens |
|---|---|---|---|---|---|---|---|---|---|
| lj | 2324 | 100 | 144 | 197 | 18 | 26 | 35 | 1.29% | 26 |
| lenta | 2166 | 112 | 147 | 193 | 15 | 20 | 27 | 0.18% | 20 |
| twitter | 2624 | 60 | 84 | 122 | 10 | 14 | 20 | 2.63% | 14 |
| social_media | 2676 | 56 | 87 | 128 | 9 | 14 | 20 | 11.81% | 14 |


## Leakage (overall, train ∩ test, exact text match)
- overlapping rows: 0
- % of test that leaks: 0.00%

## Leakage per source (train ∩ test)

| source | test rows | overlap | % of test |
|---|---|---|---|
| lj | 623 | 0 | 0.00% |
| lenta | 550 | 0 | 0.00% |
| twitter | 702 | 0 | 0.00% |
| social_media | 999 | 0 | 0.00% |


## Recommended long-sentence thresholds
- overall p75 tokens (rounded): **18**
- per source: {'lj': 26, 'lenta': 20, 'twitter': 14, 'social_media': 14}
