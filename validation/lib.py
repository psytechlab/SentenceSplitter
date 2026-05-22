"""Независимая реимплементация загрузки/нормализации/метрик IAA.

Написано с нуля для мета-валидации отчёта analysis/iaa_report/report.md.
НЕ импортирует analysis/* — полностью самостоятельный код.
"""
from __future__ import annotations
import json
import re
import math
from pathlib import Path
from collections import Counter

import numpy as np
import razdel

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = ROOT / "annotations"
ANNOTATORS = ["daniil", "shirin", "igor"]
PAIRS = [("daniil", "shirin"), ("daniil", "igor"), ("shirin", "igor")]
SENTENCE_TYPES = ["Не сложное", "СПП", "ССП", "БСП", "Смешанного типа"]

# пунктуация, снимаемая с краёв span'а при нормализации
EDGE_PUNCT = set(" \t\n\r.,;:!?…\"'«»()[]{}—–-")


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------
def _norm_offsets(start, end, text):
    """Обрезка пробелов + краевой пунктуации; проекция на содержимое."""
    s = max(0, min(start, len(text)))
    e = max(0, min(end, len(text)))
    while s < e and text[s] in EDGE_PUNCT:
        s += 1
    while e > s and text[e - 1] in EDGE_PUNCT:
        e -= 1
    return s, e


def load_annotator(name):
    """text -> dict(spans=[(start,end,norm_start,norm_end,txt)], stype=str|None)."""
    raw = json.load(open(ANN_DIR / f"{name}.json", encoding="utf-8"))
    out = {}
    for t in raw:
        text = t["data"]["text"]
        spans, stype = [], None
        anns = t.get("annotations", [])
        if anns:
            for r in anns[0].get("result", []):
                if r["type"] == "labels":
                    v = r["value"]
                    ns, ne = _norm_offsets(v["start"], v["end"], text)
                    spans.append({"start": v["start"], "end": v["end"],
                                  "norm_start": ns, "norm_end": ne,
                                  "text": text[v["start"]:v["end"]]})
                elif r["type"] == "choices":
                    ch = r["value"].get("choices", [])
                    if ch:
                        stype = ch[0]
        spans.sort(key=lambda s: (s["start"], s["end"]))
        out[text] = {"spans": spans, "stype": stype,
                      "stratum": t["data"].get("stratum", "other"),
                      "data_id": t["data"].get("data_id")}
    return out


def load_tasks():
    """Список задач, выровненных по точному совпадению data.text."""
    per = {n: load_annotator(n) for n in ANNOTATORS}
    common = set(per["daniil"]) & set(per["shirin"]) & set(per["igor"])
    tasks = []
    for idx, text in enumerate(sorted(common)):
        toks = list(razdel.tokenize(text))
        tk = {"task_idx": idx, "text": text,
              "stratum": per["daniil"][text]["stratum"] or "other",
              "data_id": per["daniil"][text]["data_id"],
              "tokens": toks, "token_starts": [t.start for t in toks],
              "anns": {n: per[n][text] for n in ANNOTATORS}}
        tasks.append(tk)
    return tasks


# ---------------------------------------------------------------------------
# Krippendorff alpha — ДВЕ независимые реализации
# ---------------------------------------------------------------------------
def alpha_lib(rows):
    """Через библиотеку krippendorff. rows = список векторов по аннотаторам."""
    import krippendorff
    arr = np.array(rows, dtype=float)
    if arr.shape[1] == 0:
        return float("nan")
    try:
        return float(krippendorff.alpha(reliability_data=arr,
                                        level_of_measurement="nominal"))
    except Exception:
        return float("nan")


def alpha_manual(rows):
    """Ручная реализация nominal Krippendorff alpha через матрицу совпадений.

    rows: список (по аннотаторам) последовательностей значений; np.nan = пропуск.
    Формула: alpha = 1 - Do/De, считается по coincidence matrix.
    """
    rows = [list(r) for r in rows]
    n_units = len(rows[0])
    # значения по каждой единице (units), отбрасывая nan
    units = []
    for u in range(n_units):
        vals = [r[u] for r in rows if not (isinstance(r[u], float) and math.isnan(r[u]))]
        if len(vals) >= 2:  # для alpha нужна единица минимум с 2 оценками
            units.append(vals)
    # множество категорий
    cats = sorted({v for u in units for v in u})
    cidx = {c: i for i, c in enumerate(cats)}
    K = len(cats)
    if K < 2:
        return float("nan")  # нет дисперсии -> alpha неопределена
    # coincidence matrix
    coinc = np.zeros((K, K))
    for u in units:
        m = len(u)
        cnt = Counter(cidx[v] for v in u)
        for c in range(K):
            for k in range(K):
                if c == k:
                    pairs = cnt[c] * (cnt[c] - 1)
                else:
                    pairs = cnt[c] * cnt[k]
                coinc[c, k] += pairs / (m - 1)
    n_c = coinc.sum(axis=1)
    n_total = n_c.sum()
    if n_total == 0:
        return float("nan")
    # nominal: метрика расстояния = 1 если c!=k, иначе 0
    Do = sum(coinc[c, k] for c in range(K) for k in range(K) if c != k)
    De = sum(n_c[c] * n_c[k] for c in range(K) for k in range(K) if c != k) / (n_total - 1)
    if De == 0:
        return float("nan")
    return 1.0 - Do / De


def alpha_simpledorff(rows):
    """Через simpledorff (3-я перекрёстная проверка)."""
    import pandas as pd
    import simpledorff
    recs = []
    for ai, r in enumerate(rows):
        for u, v in enumerate(r):
            if isinstance(v, float) and math.isnan(v):
                continue
            recs.append({"doc": u, "annotator": ai, "label": v})
    df = pd.DataFrame(recs)
    if df.empty or df["label"].nunique() < 2:
        return float("nan")
    try:
        return float(simpledorff.calculate_krippendorffs_alpha_for_df(
            df, experiment_col="doc", annotator_col="annotator",
            class_col="label", metric_fn=simpledorff.metrics.nominal_metric))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Токенные бинарные векторы границ
# ---------------------------------------------------------------------------
def snap_to_tokens(offsets, token_starts):
    """Каждое char-смещение -> индекс ближайшего начала токена."""
    out = set()
    for s in offsets:
        if not token_starts:
            continue
        idx = min(range(len(token_starts)),
                  key=lambda i: abs(token_starts[i] - s))
        out.add(idx)
    return out


def task_starts(tk, ann, use_norm=True):
    key = "norm_start" if use_norm else "start"
    return [sp[key] for sp in tk["anns"][ann]["spans"]]


def token_vectors(tasks, anns, use_norm=True):
    """Плоский бинарный вектор по всем токенам всех задач (token 0 исключён)."""
    vecs = {a: [] for a in anns}
    for tk in tasks:
        ts = tk["token_starts"]
        n = len(ts)
        if n <= 1:
            continue
        for a in anns:
            idx = snap_to_tokens(task_starts(tk, a, use_norm), ts)
            for i in range(1, n):
                vecs[a].append(1 if i in idx else 0)
    return vecs


def task_token_vectors(tk, anns, use_norm=True):
    """Бинарные векторы для ОДНОЙ задачи (token 0 исключён)."""
    ts = tk["token_starts"]
    n = len(ts)
    if n <= 1:
        return {a: [] for a in anns}
    vecs = {}
    for a in anns:
        idx = snap_to_tokens(task_starts(tk, a, use_norm), ts)
        vecs[a] = [1 if i in idx else 0 for i in range(1, n)]
    return vecs


# ---------------------------------------------------------------------------
# Boundary F1
# ---------------------------------------------------------------------------
def match_offsets(a, b, tol):
    """Жадное паросочетание смещений в пределах tol. -> (tp, fp, fn)."""
    a, b = sorted(a), sorted(b)
    used = set()
    tp = 0
    for x in a:
        best, best_d = None, tol + 1
        for j, y in enumerate(b):
            if j in used:
                continue
            d = abs(x - y)
            if d <= tol and d < best_d:
                best_d, best = d, j
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(a) - tp, len(b) - tp


def boundary_f1_task(tk, a, b, tol, use_norm=True):
    """F1 границ для одной задачи и одной пары."""
    sa = [s for s in task_starts(tk, a, use_norm) if s > 0]
    sb = [s for s in task_starts(tk, b, use_norm) if s > 0]
    tp, fp, fn = match_offsets(sa, sb, tol)
    if tp + fp + fn == 0:
        return 1.0, (tp, fp, fn)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return f1, (tp, fp, fn)


def boundary_f1(tasks, tol, use_norm=True):
    """macro/micro F1 границ по парам."""
    out = {}
    for a, b in PAIRS:
        macro, mtp, mfp, mfn = [], 0, 0, 0
        for tk in tasks:
            f1, (tp, fp, fn) = boundary_f1_task(tk, a, b, tol, use_norm)
            macro.append(f1)
            mtp += tp; mfp += fp; mfn += fn
        mp = mtp / (mtp + mfp) if (mtp + mfp) else 0.0
        mr = mtp / (mtp + mfn) if (mtp + mfn) else 0.0
        mf = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
        out[f"{a}-{b}"] = {"macro_f1": float(np.mean(macro)),
                           "micro_f1": float(mf), "micro_p": float(mp),
                           "micro_r": float(mr)}
    return out


# ---------------------------------------------------------------------------
# Span Jaccard
# ---------------------------------------------------------------------------
def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def span_jaccard_task(tk, a, b, relaxed=False, thr=0.5, use_norm=True):
    key = ("norm_start", "norm_end") if use_norm else ("start", "end")
    A = [(s[key[0]], s[key[1]]) for s in tk["anns"][a]["spans"]]
    B = [(s[key[0]], s[key[1]]) for s in tk["anns"][b]["spans"]]
    if not A and not B:
        return 1.0
    if not relaxed:
        sa, sb = set(A), set(B)
        union = len(sa | sb)
        return len(sa & sb) / union if union else 1.0
    used, match = set(), 0
    for x in A:
        best, best_iou = None, thr
        for j, y in enumerate(B):
            if j in used:
                continue
            v = iou(x, y)
            if v > best_iou:
                best_iou, best = v, j
        if best is not None:
            used.add(best)
            match += 1
    union = len(A) + len(B) - match
    return match / union if union else 1.0


def span_jaccard(tasks, relaxed=False, thr=0.5, use_norm=True):
    out = {}
    for a, b in PAIRS:
        out[f"{a}-{b}"] = float(np.mean(
            [span_jaccard_task(tk, a, b, relaxed, thr, use_norm) for tk in tasks]))
    return out


# ---------------------------------------------------------------------------
# Per-task aggregate metrics
# ---------------------------------------------------------------------------
def task_token_alpha_triple(tk, use_norm=True):
    vecs = task_token_vectors(tk, ANNOTATORS, use_norm)
    if not vecs["daniil"]:
        return float("nan")
    return alpha_manual([vecs[a] for a in ANNOTATORS])


def task_boundary_f1_relaxed_avg(tk, tol=3, use_norm=True):
    return float(np.mean([boundary_f1_task(tk, a, b, tol, use_norm)[0]
                          for a, b in PAIRS]))


def task_span_jaccard_exact_avg(tk, use_norm=True):
    return float(np.mean([span_jaccard_task(tk, a, b, False, 0.5, use_norm)
                          for a, b in PAIRS]))
