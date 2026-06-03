# notebooks/sampling/ — отбор выборки для разметки

Отбор сложных русских предложений из исходных корпусов (presuicidal, solyanka)
в выборку для ручной разметки «разделение сложного предложения на клаузы».

## Структура

```
notebooks/sampling/
├── sampling.ipynb              # основной нотбук отбора (стратификация, фильтры)
├── sampling_train_800.ipynb    # отбор train-выборки 800
├── sampling_train_2000.ipynb   # отбор train-выборки 2000
├── eda/                        # разведочный анализ исходных корпусов
│   ├── presuicidal/            #   гистограммы длин, топ-лейблы, summary.md
│   ├── solyanka/               #   распределения по источникам, summary.md
│   └── comparison.md           # сопоставление корпусов
├── heuristics/                 # эвристики отбора «сложное предложение»
│   ├── filters.py              #   функции-фильтры (союзы, БСП-маркеры, длина)
│   ├── conjunctions.json       #   таблица союзов
│   └── validation.md           #   валидация эвристик
└── notes/                      # сопроводительные заметки
    ├── edge_cases.md           #   include/exclude правила (из annotation doc)
    ├── llm_splitter_notes.md   #   аудит Split_llm.ipynb
    └── train_800_summary.md    #   итоги отбора train-800
```

## Порядок работы

1. **EDA** (`eda/`) — анализ длин, лейблов, распределений по источникам → пороги стратификации.
2. **Эвристики** (`heuristics/filters.py`) — разметка «сложное/простое» по союзам, БСП-маркерам, числу финитных глаголов; правила include/exclude из `notes/edge_cases.md`.
3. **Отбор** (`sampling*.ipynb`) — стратифицированная выборка; результаты пишутся в `data/sample_for_annotation*.csv`.

## Связь с остальным проектом

- Эвристики `heuristics/filters.py` валидируются против majority GT в `validation/scripts/heuristics_eval.py`.
- Итоговая выборка `data/sample_for_annotation.csv` (200 текстов) → размечена тремя аннотаторами → `annotations/` → meta-валидация в `validation/` → эксперимент `experiments/agent_annotator/`.
