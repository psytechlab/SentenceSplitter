# validation/ — мета-валидация IAA

Независимая перепроверка отчёта `analysis/iaa_report/` по согласию трёх аннотаторов:
метрики переписаны с нуля (`lib.py`), расхождения перекатегоризированы, эвристики
отбора выборки сверены с majority-voting ground truth.

## Структура

```
validation/
├── lib.py            # общая библиотека: загрузка аннотаций, нормализация,
│                     #   метрики (Krippendorff α, boundary F1, span Jaccard).
│                     #   Импортируется скриптами и экспериментом agent_annotator.
├── scripts/          # генераторы отчётов и данных (запускать из этой папки)
├── reports/          # markdown-отчёты (нумерация = порядок задач)
├── data/             # выходные данные (CSV) + промежуточные дампы
└── plots/            # графики
```

## Пайплайн (порядок запуска)

Все скрипты запускаются из `validation/scripts/` и сами кладут результаты в
`reports/`, `data/`, `plots/` (пути прописаны абсолютно от корня репозитория).

| Скрипт | Что делает | Выход |
|---|---|---|
| `scripts/categorize_indep.py` | модуль-хелпер: независимая категоризация спорной области (импортируется другими) | — |
| `scripts/generate_task1.py` | построчный список всех расхождений | `reports/01_disagreement_full_list.md` |
| `scripts/generate_task2.py` | независимая валидация метрик из отчёта | `reports/02_metrics_validation.md` |
| `scripts/generate_task3.py` | эвристики категоризации vs поведение метрик | `reports/03_heuristics_vs_metrics.md`, `plots/35_stratum_category.png` |
| `scripts/heuristics_eval.py` | эвристики отбора выборки vs majority GT (Шаг 5) | `data/ground_truth.csv`, `data/heuristic_predictions.csv`, `data/_heur_results.json` |
| `scripts/generate_final_data.py` | экспорт категоризации + числа для финального отчёта | `data/categorization_by_agent.csv` |

Итоговый синтез — `reports/FINAL_REPORT.md` (writeup Шага 5 — `reports/04_heuristics_vs_ground_truth.md`).

## Входные данные (внешние, не меняются)

- `annotations/{daniil,shirin,igor}.json` — разметки трёх аннотаторов (читает `lib.py`).
- `analysis/iaa_report/{disagreements,metrics}.csv` — исходный отчёт для сверки.
- `data/sample_for_annotation.csv` — метаданные 200 задач.

## Ключевые артефакты

- `data/ground_truth.csv` — majority-voting GT (сложное/простое) по 200 задачам.
- `data/categorization_by_agent.csv` — независимая категоризация всех расхождений.
- `data/heuristic_predictions.csv` — предсказания эвристик отбора по 200 задачам.
- `lib.py` — переиспользуется экспериментом `experiments/agent_annotator/`.

## Запуск

```bash
cd validation/scripts
python3 generate_task1.py        # и т.д. — каждый скрипт самодостаточен
```
