"""Шаг 5: валидация эвристик ОТБОРА выборки против majority-voting ground truth."""
import csv
import sys
from pathlib import Path
from collections import Counter

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import lib

# импорт эвристик отбора
HEUR_DIR = lib.ROOT / "notebooks/sampling/heuristics"
sys.path.insert(0, str(HEUR_DIR))
import filters  # noqa: E402

COMPLEX_TYPES = {"ССП", "СПП", "БСП", "Смешанного типа"}
SIMPLE_TYPE = "Не сложное"

# ---------------------------------------------------------------------------
# Шаг 1. Ground truth через majority voting (только по sentence_type)
# ---------------------------------------------------------------------------
tasks = lib.load_tasks()

# метаданные из sample_for_annotation.csv (для is_multilabel)
sample = {}
for r in csv.DictReader(open(lib.ROOT / "data/sample_for_annotation.csv")):
    sample[r["text"]] = (r["source_dataset"], r["original_label"], r["stratum"])

gt_rows = []
for tk in tasks:
    types = {a: tk["anns"][a]["stype"] for a in lib.ANNOTATORS}
    votes = [types[a] for a in lib.ANNOTATORS]
    n_complex = sum(1 for v in votes if v in COMPLEX_TYPES)
    n_simple = sum(1 for v in votes if v == SIMPLE_TYPE)
    if n_complex >= 2:
        is_complex, ambiguous, maj = True, False, "complex"
    elif n_simple >= 2:
        is_complex, ambiguous, maj = False, False, "simple"
    else:
        is_complex, ambiguous, maj = None, True, "ambiguous"
    gt_rows.append({
        "task_id": tk["task_idx"], "text": tk["text"], "stratum": tk["stratum"],
        "daniil_type": types["daniil"] or "", "shirin_type": types["shirin"] or "",
        "igor_type": types["igor"] or "", "majority_type": maj,
        "is_complex_gt": "" if is_complex is None else is_complex,
        "ambiguous": ambiguous,
    })

with open(lib.ROOT / "validation/data/ground_truth.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(gt_rows[0].keys()))
    w.writeheader()
    w.writerows(gt_rows)

n_amb = sum(1 for r in gt_rows if r["ambiguous"])
n_cx = sum(1 for r in gt_rows if r["is_complex_gt"] is True)
n_sx = sum(1 for r in gt_rows if r["is_complex_gt"] is False)
print(f"GT: complex={n_cx}  simple={n_sx}  ambiguous={n_amb}  total={len(gt_rows)}")

# ---------------------------------------------------------------------------
# Шаг 2. Применить эвристики
# ---------------------------------------------------------------------------
pred_rows = []
for tk in tasks:
    text = tk["text"]
    ds, label, _ = sample.get(text, ("", "", ""))
    hc = filters.has_conjunction(text)
    ha = filters.has_asyndetic_markers(text, min_tokens=8)
    nfv = filters.count_finite_verbs(text)
    fv2 = nfv >= 2
    v1 = filters.is_complex(text)                       # has_conj OR has_asyndetic
    v2 = filters._is_complex_v2_rejected(text)          # (>=2 finite) AND (conj OR bsp)
    ml = filters.is_multilabel(label, ds) if ds in {"presuicidal", "solyanka"} else False
    pred_rows.append({
        "task_id": tk["task_idx"], "has_conjunction": hc, "has_asyndetic": ha,
        "finite_verbs_ge_2": fv2, "finite_verbs": nfv,
        "is_complex_v1": v1, "is_complex_v2": v2, "is_multilabel": ml,
    })

with open(lib.ROOT / "validation/data/heuristic_predictions.csv", "w",
          encoding="utf-8", newline="") as f:
    cols = ["task_id", "has_conjunction", "has_asyndetic", "finite_verbs_ge_2",
            "is_complex_v1", "is_complex_v2", "is_multilabel"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(pred_rows)

# ---------------------------------------------------------------------------
# Шаг 3. Метрики
# ---------------------------------------------------------------------------
gt = {r["task_id"]: r["is_complex_gt"] for r in gt_rows}
pred = {r["task_id"]: r for r in pred_rows}
eval_ids = [tid for tid in gt if gt[tid] in (True, False)]


def metrics(pred_key):
    tp = fp = tn = fn = 0
    for tid in eval_ids:
        g = gt[tid]
        p = pred[tid][pred_key]
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif not p and not g:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) else float("nan"))
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, tn=tn, fn=fn)


HEURS = [("has_conjunction", "has_conjunction"),
         ("has_asyndetic", "has_asyndetic_markers"),
         ("finite_verbs_ge_2", "finite_verbs ≥ 2"),
         ("is_complex_v1", "is_complex_v1 (used)"),
         ("is_complex_v2", "is_complex_v2 (rejected)")]

results = {key: metrics(key) for key, _ in HEURS}

print(f"\nГлавная таблица (n_eval={len(eval_ids)}, исключено ambiguous={n_amb}):\n")
print("| Эвристика | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |")
print("|---|---|---|---|---|---|---|---|---|")
for key, name in HEURS:
    m = results[key]
    print(f"| {name} | {m['acc']:.3f} | {m['prec']:.3f} | {m['rec']:.3f} | "
          f"{m['f1']:.3f} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")

if __name__ == "__main__":
    import json
    Path(lib.ROOT / "validation/data/_heur_results.json").write_text(
        json.dumps({"results": results, "n_eval": len(eval_ids),
                    "n_amb": n_amb, "n_cx": n_cx, "n_sx": n_sx},
                   ensure_ascii=False))
