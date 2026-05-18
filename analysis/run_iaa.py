"""IAA full diagnostic — orchestrator.

Run:  .venv/bin/python analysis/run_iaa.py
Produces everything under analysis/iaa_report/.
"""
from __future__ import annotations
import csv
import json
import os
import random
import statistics as st
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import iaa_common as C
from iaa_common import ANNOTATORS, PAIRS, SENTENCE_TYPES, STRATA
import iaa_metrics as M
import iaa_categorize as CAT

random.seed(C.SEED)
np.random.seed(C.SEED)
sns.set_theme(style="whitegrid")

C.OUT_DIR.mkdir(parents=True, exist_ok=True)
C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
C.PER_ROW_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.7
LEN_BINS = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 10**6)]
LEN_LABELS = ["0-10", "10-20", "20-40", "40-80", "80+"]

print("Loading tasks ...")
tasks = C.load_tasks()
print(f"  {len(tasks)} aligned tasks")

REPORT = []  # list of markdown lines
METRICS_ROWS = []  # for metrics.csv: (section, metric, scope, value)


def add_metric(section, metric, scope, value):
    METRICS_ROWS.append({"section": section, "metric": metric,
                         "scope": scope, "value": value})


def fnum(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}"


def flag(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "⚪"
    return "🟢" if v >= THRESHOLD else "🔴"


# ===========================================================================
# 1. DESCRIPTIVE STATISTICS
# ===========================================================================
print("Section 1: descriptive statistics ...")
desc = {}
for a in ANNOTATORS:
    spans_per_task = []
    span_len_chars = []
    span_len_tokens = []
    st_dist = Counter()
    n_missing_type = 0
    strat_spans = defaultdict(list)
    for tk in tasks:
        ta = tk.anns[a]
        spans_per_task.append(len(ta.spans))
        strat_spans[tk.stratum].append(len(ta.spans))
        for sp in ta.spans:
            span_len_chars.append(sp.norm_end - sp.norm_start)
            ntok = len(list(__import__("razdel").tokenize(sp.norm_text)))
            span_len_tokens.append(ntok)
        if ta.sentence_type is None:
            n_missing_type += 1
        else:
            st_dist[ta.sentence_type] += 1
    desc[a] = {
        "n_tasks": len(tasks),
        "n_missing_type": n_missing_type,
        "spans_mean": st.mean(spans_per_task),
        "spans_median": st.median(spans_per_task),
        "spans_std": st.pstdev(spans_per_task),
        "span_chars_mean": st.mean(span_len_chars) if span_len_chars else 0,
        "span_chars_median": st.median(span_len_chars) if span_len_chars else 0,
        "span_chars_std": st.pstdev(span_len_chars) if span_len_chars else 0,
        "span_tok_mean": st.mean(span_len_tokens) if span_len_tokens else 0,
        "span_tok_median": st.median(span_len_tokens) if span_len_tokens else 0,
        "span_tok_std": st.pstdev(span_len_tokens) if span_len_tokens else 0,
        "type_dist": dict(st_dist),
        "spans_per_task_list": spans_per_task,
        "strat_spans": {k: st.mean(v) for k, v in strat_spans.items()},
    }
    add_metric("descriptive", "spans_per_task_mean", a, desc[a]["spans_mean"])
    add_metric("descriptive", "span_len_tokens_mean", a, desc[a]["span_tok_mean"])

# Plot: histogram spans-per-task (overlapping)
plt.figure(figsize=(8, 5))
maxs = max(max(desc[a]["spans_per_task_list"]) for a in ANNOTATORS)
bins = np.arange(0, maxs + 2) - 0.5
for a in ANNOTATORS:
    plt.hist(desc[a]["spans_per_task_list"], bins=bins, alpha=0.5, label=a)
plt.xlabel("Число span'ов (простых клауз) на задачу")
plt.ylabel("Количество задач")
plt.title("Распределение числа span'ов на задачу по аннотаторам")
plt.legend()
plt.tight_layout()
plt.savefig(C.PLOTS_DIR / "01_spans_per_task_hist.png", dpi=130)
plt.close()

# Plot: bar chart sentence_type distribution
plt.figure(figsize=(9, 5))
x = np.arange(len(SENTENCE_TYPES))
w = 0.25
for i, a in enumerate(ANNOTATORS):
    vals = [desc[a]["type_dist"].get(t, 0) for t in SENTENCE_TYPES]
    plt.bar(x + (i - 1) * w, vals, w, label=a)
plt.xticks(x, SENTENCE_TYPES, rotation=15)
plt.ylabel("Количество задач")
plt.title("Распределение типов предложения (sentence_type) по аннотаторам")
plt.legend()
plt.tight_layout()
plt.savefig(C.PLOTS_DIR / "02_sentence_type_dist.png", dpi=130)
plt.close()


# ===========================================================================
# 2. NORMALIZATION EFFECT (punctuation noise)
# ===========================================================================
print("Section 2: normalization effect ...")
raw_disagree_tasks = 0
norm_disagree_tasks = 0
for tk in tasks:
    raw = {a: frozenset((s.start, s.end) for s in tk.anns[a].spans) for a in ANNOTATORS}
    nrm = {a: frozenset((s.norm_start, s.norm_end) for s in tk.anns[a].spans) for a in ANNOTATORS}
    if len(set(raw.values())) > 1:
        raw_disagree_tasks += 1
    if len(set(nrm.values())) > 1:
        norm_disagree_tasks += 1
punct_noise_tasks = raw_disagree_tasks - norm_disagree_tasks
add_metric("normalization", "raw_disagree_tasks", "all", raw_disagree_tasks)
add_metric("normalization", "norm_disagree_tasks", "all", norm_disagree_tasks)
add_metric("normalization", "punctuation_noise_tasks", "all", punct_noise_tasks)


# ===========================================================================
# 3. AGREEMENT METRICS
# ===========================================================================
print("Section 3: agreement metrics ...")
tok_alpha_norm = M.token_alpha(tasks, use_norm=True)
tok_alpha_raw = M.token_alpha(tasks, use_norm=False)
bf1_exact = M.boundary_f1(tasks, tol=0, use_norm=True)
bf1_relax = M.boundary_f1(tasks, tol=3, use_norm=True)
jac_exact = M.span_jaccard(tasks, relaxed=False, use_norm=True)
jac_relax = M.span_jaccard(tasks, relaxed=True, iou_thr=0.5, use_norm=True)
type_alpha = M.sentence_type_alpha(tasks)

for k, v in tok_alpha_norm.items():
    add_metric("agreement", "token_krippendorff_alpha", k, v)
for k, v in bf1_exact.items():
    add_metric("agreement", "boundary_f1_exact_macro", k, v["macro_f1"])
    add_metric("agreement", "boundary_f1_exact_micro", k, v["micro_f1"])
for k, v in bf1_relax.items():
    add_metric("agreement", "boundary_f1_relaxed_macro", k, v["macro_f1"])
    add_metric("agreement", "boundary_f1_relaxed_micro", k, v["micro_f1"])
for k, v in jac_exact.items():
    add_metric("agreement", "span_jaccard_exact", k, v)
for k, v in jac_relax.items():
    add_metric("agreement", "span_jaccard_relaxed", k, v)
for k, v in type_alpha.items():
    add_metric("agreement", "sentence_type_alpha", k, v)

# Heatmap of pairwise token alpha
plt.figure(figsize=(5.5, 4.5))
mat = np.ones((3, 3))
for i, a in enumerate(ANNOTATORS):
    for j, b in enumerate(ANNOTATORS):
        if i == j:
            mat[i, j] = 1.0
        else:
            key = f"{a}-{b}" if f"{a}-{b}" in tok_alpha_norm else f"{b}-{a}"
            mat[i, j] = tok_alpha_norm[key]
sns.heatmap(mat, annot=True, fmt=".3f", xticklabels=ANNOTATORS,
            yticklabels=ANNOTATORS, cmap="RdYlGn", vmin=0, vmax=1,
            cbar_kws={"label": "Krippendorff α"})
plt.title("Попарная token-level Krippendorff α (границы клауз)")
plt.tight_layout()
plt.savefig(C.PLOTS_DIR / "03_token_alpha_heatmap.png", dpi=130)
plt.close()


# ===========================================================================
# 4. DISAGREEMENT CATEGORIZATION
# ===========================================================================
print("Section 4: disagreement categorization ...")
has_llm = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
disagreement_rows = []
cat_counter = Counter()
for ti, tk in enumerate(tasks):
    raw_sets = {a: [(s.start, s.end) for s in tk.anns[a].spans] for a in ANNOTATORS}
    nrm_sets = {a: [(s.norm_start, s.norm_end) for s in tk.anns[a].spans] for a in ANNOTATORS}
    nrm_frozen = {a: frozenset(nrm_sets[a]) for a in ANNOTATORS}
    raw_frozen = {a: frozenset(raw_sets[a]) for a in ANNOTATORS}
    is_disagree = len(set(nrm_frozen.values())) > 1 or len(set(raw_frozen.values())) > 1
    if not is_disagree:
        continue
    cat, descr = CAT.categorize(
        tk.text, raw_sets, nrm_sets,
        {a: tk.anns[a].sentence_type for a in ANNOTATORS})
    cat_counter[cat] += 1
    row = {
        "task_idx": ti,
        "text": tk.text,
        "stratum": tk.stratum,
        "category": cat,
        "description": descr,
    }
    for a in ANNOTATORS:
        row[f"{a}_spans"] = "; ".join(
            f"[{s.norm_start},{s.norm_end}]{s.norm_text}" for s in tk.anns[a].spans)
        row[f"{a}_n_spans"] = len(tk.anns[a].spans)
        row[f"{a}_type"] = tk.anns[a].sentence_type or "MISSING"
    disagreement_rows.append(row)

# Plot: bar chart of category counts
plt.figure(figsize=(10, 5))
cats_sorted = cat_counter.most_common()
plt.barh([c for c, _ in cats_sorted][::-1], [n for _, n in cats_sorted][::-1],
         color=sns.color_palette("viridis", len(cats_sorted)))
plt.xlabel("Число задач с расхождением")
plt.title("Категории несогласий аннотаторов")
plt.tight_layout()
plt.savefig(C.PLOTS_DIR / "04_disagreement_categories.png", dpi=130)
plt.close()


# ===========================================================================
# 5. PER-STRATUM METRICS
# ===========================================================================
print("Section 5: per-stratum metrics ...")
strat_alpha = {}
for s in STRATA:
    sub = [tk for tk in tasks if tk.stratum == s]
    if len(sub) < 2:
        strat_alpha[s] = (float("nan"), len(sub))
        continue
    a = M.token_alpha(sub, use_norm=True)["triple"]
    strat_alpha[s] = (a, len(sub))
    add_metric("stratum", "token_alpha_triple", s, a)

plt.figure(figsize=(11, 5))
order = sorted(STRATA, key=lambda s: (strat_alpha[s][0]
                                      if not np.isnan(strat_alpha[s][0]) else 1))
vals = [strat_alpha[s][0] for s in order]
colors = ["#d62728" if (not np.isnan(v) and v < THRESHOLD) else "#2ca02c" for v in vals]
plt.bar(range(len(order)), [0 if np.isnan(v) else v for v in vals], color=colors)
plt.axhline(THRESHOLD, ls="--", c="black", label=f"порог {THRESHOLD}")
plt.xticks(range(len(order)),
           [f"{s}\n(n={strat_alpha[s][1]})" for s in order], rotation=35, ha="right")
plt.ylabel("Token Krippendorff α (triple)")
plt.title("IAA по стратам исходного датасета")
plt.legend()
plt.tight_layout()
plt.savefig(C.PLOTS_DIR / "05_alpha_by_stratum.png", dpi=130)
plt.close()


# ===========================================================================
# 6. PER-LENGTH-BIN METRICS
# ===========================================================================
print("Section 6: per-length metrics ...")
len_alpha = {}
for (lo, hi), lab in zip(LEN_BINS, LEN_LABELS):
    sub = [tk for tk in tasks if lo <= len(tk.tokens) < hi]
    if len(sub) < 2:
        len_alpha[lab] = (float("nan"), len(sub))
        continue
    a = M.token_alpha(sub, use_norm=True)["triple"]
    len_alpha[lab] = (a, len(sub))
    add_metric("length", "token_alpha_triple", lab, a)

plt.figure(figsize=(8, 5))
xs = [l for l in LEN_LABELS if not np.isnan(len_alpha[l][0])]
ys = [len_alpha[l][0] for l in xs]
plt.plot(xs, ys, "o-", lw=2)
for l, y in zip(xs, ys):
    plt.annotate(f"n={len_alpha[l][1]}", (l, y), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8)
plt.axhline(THRESHOLD, ls="--", c="red", label=f"порог {THRESHOLD}")
plt.xlabel("Длина предложения (токены)")
plt.ylabel("Token Krippendorff α (triple)")
plt.title("Зависимость IAA от длины предложения")
plt.legend()
plt.tight_layout()
plt.savefig(C.PLOTS_DIR / "06_alpha_by_length.png", dpi=130)
plt.close()


# ===========================================================================
# 7. PER-SENTENCE-TYPE METRICS
# ===========================================================================
print("Section 7: per-sentence-type metrics ...")
def mode_type(tk):
    votes = [tk.anns[a].sentence_type for a in ANNOTATORS
             if tk.anns[a].sentence_type]
    if not votes:
        return None
    c = Counter(votes)
    top = c.most_common()
    # require at least 2 of 3 agree, else 'нет консенсуса'
    if top[0][1] >= 2:
        return top[0][0]
    return "нет консенсуса"

type_alpha_split = {}
for t in SENTENCE_TYPES + ["нет консенсуса"]:
    sub = [tk for tk in tasks if mode_type(tk) == t]
    if len(sub) < 2:
        type_alpha_split[t] = (float("nan"), len(sub))
        continue
    a = M.token_alpha(sub, use_norm=True)["triple"]
    type_alpha_split[t] = (a, len(sub))
    add_metric("sentence_type", "token_alpha_triple", t, a)

plt.figure(figsize=(9, 5))
order2 = SENTENCE_TYPES + ["нет консенсуса"]
vals2 = [type_alpha_split[t][0] for t in order2]
colors2 = ["#d62728" if (not np.isnan(v) and v < THRESHOLD) else "#2ca02c"
           for v in vals2]
plt.bar(range(len(order2)), [0 if np.isnan(v) else v for v in vals2], color=colors2)
plt.axhline(THRESHOLD, ls="--", c="black")
plt.xticks(range(len(order2)),
           [f"{t}\n(n={type_alpha_split[t][1]})" for t in order2],
           rotation=20, ha="right")
plt.ylabel("Token Krippendorff α (triple)")
plt.title("IAA по типу предложения (mode-голосование)")
plt.tight_layout()
plt.savefig(C.PLOTS_DIR / "07_alpha_by_sentence_type.png", dpi=130)
plt.close()


# ===========================================================================
# 8. TOP-DISAGREEMENT DEEP DIVE
# ===========================================================================
print("Section 8: top-disagreement deep dive ...")
# rank tasks by mean pairwise boundary F1 relaxed (lower = worse)
def task_pair_bf1(tk, a, b, tol=3):
    sa = [s for s in (sp.norm_start for sp in tk.anns[a].spans) if s > 0]
    sb = [s for s in (sp.norm_start for sp in tk.anns[b].spans) if s > 0]
    tp, fp, fn = M._match_sets(sa, sb, tol)
    if tp + fp + fn == 0:
        return 1.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0

task_scores = []
for ti, tk in enumerate(tasks):
    f1s = [task_pair_bf1(tk, a, b) for a, b in PAIRS]
    task_scores.append((np.mean(f1s), ti))
task_scores.sort()
top20 = task_scores[:20]


def render_spans(tk, a):
    """Render text with [ ] bracketed clause spans for annotator a."""
    text = tk.text
    spans = sorted(tk.anns[a].spans, key=lambda s: s.norm_start)
    out = []
    cursor = 0
    for s in spans:
        if s.norm_start < cursor:  # overlap — show inline marker
            out.append(f" «OVERLAP[{s.norm_start},{s.norm_end}]» ")
            continue
        out.append(text[cursor:s.norm_start])
        out.append("⟦" + text[s.norm_start:s.norm_end] + "⟧")
        cursor = s.norm_end
    out.append(text[cursor:])
    return "".join(out)


# ===========================================================================
# 9. SENTENCE-TYPE CONFUSION MATRICES
# ===========================================================================
print("Section 9: sentence-type confusion matrices ...")
conf = {}
for a, b in PAIRS:
    idx = {t: i for i, t in enumerate(SENTENCE_TYPES)}
    m = np.zeros((len(SENTENCE_TYPES), len(SENTENCE_TYPES)), dtype=int)
    for tk in tasks:
        ta, tb = tk.anns[a].sentence_type, tk.anns[b].sentence_type
        if ta in idx and tb in idx:
            m[idx[ta], idx[tb]] += 1
    conf[f"{a}-{b}"] = m
    plt.figure(figsize=(6, 5))
    sns.heatmap(m, annot=True, fmt="d", cmap="Blues",
                xticklabels=SENTENCE_TYPES, yticklabels=SENTENCE_TYPES)
    plt.xlabel(b)
    plt.ylabel(a)
    plt.title(f"Confusion matrix sentence_type: {a} × {b}")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / f"09_confusion_{a}_{b}.png", dpi=130)
    plt.close()


# ===========================================================================
# WRITE metrics.csv
# ===========================================================================
print("Writing metrics.csv ...")
with open(C.OUT_DIR / "metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["section", "metric", "scope", "value"])
    w.writeheader()
    for r in METRICS_ROWS:
        v = r["value"]
        if isinstance(v, float):
            v = "" if np.isnan(v) else f"{v:.6f}"
        w.writerow({**r, "value": v})

# ===========================================================================
# WRITE disagreements.csv
# ===========================================================================
print("Writing disagreements.csv ...")
dcols = ["task_idx", "text", "stratum", "category", "description"]
for a in ANNOTATORS:
    dcols += [f"{a}_n_spans", f"{a}_spans", f"{a}_type"]
with open(C.OUT_DIR / "disagreements.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=dcols)
    w.writeheader()
    for r in disagreement_rows:
        w.writerow(r)

# ===========================================================================
# WRITE per_row JSON
# ===========================================================================
print("Writing per_row JSON ...")
for ti, tk in enumerate(tasks):
    obj = {
        "task_idx": ti,
        "text": tk.text,
        "stratum": tk.stratum,
        "data_id": tk.data_id,
        "n_tokens": len(tk.tokens),
        "annotations": {},
    }
    for a in ANNOTATORS:
        obj["annotations"][a] = {
            "sentence_type": tk.anns[a].sentence_type,
            "spans": [{"start": s.start, "end": s.end,
                       "norm_start": s.norm_start, "norm_end": s.norm_end,
                       "norm_text": s.norm_text} for s in tk.anns[a].spans],
        }
    with open(C.PER_ROW_DIR / f"task_{ti:03d}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)

# ===========================================================================
# BUILD report.md
# ===========================================================================
print("Building report.md ...")
R = REPORT.append

R("# Анализ IAA: разбиение сложных предложений\n")
R(f"_Сгенерировано автоматически (`analysis/run_iaa.py`, SEED={C.SEED}). "
  f"Дата: 2026-05-19._\n")

# ---- summary table data ----
def pair_keys():
    return ["daniil-shirin", "daniil-igor", "shirin-igor", "triple"]

summary = {
    "Token Krippendorff α": {k: tok_alpha_norm.get(k) for k in pair_keys()},
    "Boundary F1 exact": {**{k: bf1_exact[k]["macro_f1"] for k in bf1_exact},
                          "triple": None},
    "Boundary F1 relaxed (±3)": {**{k: bf1_relax[k]["macro_f1"] for k in bf1_relax},
                                 "triple": None},
    "Span Jaccard exact": {**jac_exact, "triple": None},
    "Span Jaccard relaxed (IoU>0.5)": {**jac_relax, "triple": None},
    "Sentence type α": {k: type_alpha.get(k) for k in pair_keys()},
}

# ---- Executive summary ----
R("## Executive summary\n")
R("Три аннотатора (Daniil, Shirin, Igor) независимо разметили **одни и те же "
  f"{len(tasks)} предложений** в Label Studio: выделяли простые клаузы (span'ы) "
  "и указывали тип сложного предложения.\n")
R("### Главные числа\n")
R("| Метрика | Daniil↔Shirin | Daniil↔Igor | Shirin↔Igor | Triple |")
R("|---|---|---|---|---|")
for mname, vals in summary.items():
    cells = []
    for k in pair_keys():
        v = vals.get(k)
        if v is None:
            cells.append("—")
        else:
            cells.append(f"{flag(v)} {fnum(v)}")
    R(f"| {mname} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
R("")
R(f"_Порог приёмки: **{THRESHOLD}**. 🟢 ≥ порога, 🔴 < порога, ⚪ нет данных, "
  "— метрика только попарная._\n")

# verdict
tok_triple = tok_alpha_norm["triple"]
typ_triple = type_alpha["triple"]
R("### Вердикт по порогу 0.7\n")
verdict_lines = []
verdict_lines.append(
    f"- **Token Krippendorff α (границы клауз), triple = {fnum(tok_triple)}** — "
    + ("🟢 порог достигнут." if tok_triple >= THRESHOLD
       else "🔴 порог НЕ достигнут."))
verdict_lines.append(
    f"- **Sentence type α, triple = {fnum(typ_triple)}** — "
    + ("🟢 порог достигнут." if (not np.isnan(typ_triple) and typ_triple >= THRESHOLD)
       else "🔴 порог НЕ достигнут."))
br = bf1_relax
mean_brelax = np.mean([br[k]["macro_f1"] for k in br])
verdict_lines.append(
    f"- **Boundary F1 relaxed (±3), среднее по парам = {fnum(mean_brelax)}** — "
    + ("🟢" if mean_brelax >= THRESHOLD else "🔴")
    + " мягкая метрика границ.")
for l in verdict_lines:
    R(l)
R("")

# normalization headline
R(f"**Пунктуационный шум:** до нормализации задач с расхождением границ — "
  f"{raw_disagree_tasks}, после нормализации — {norm_disagree_tasks}. "
  f"Нормализация (обрезка краевой пунктуации/пробелов) сняла "
  f"**{punct_noise_tasks}** ложных расхождений "
  f"({round(100*punct_noise_tasks/max(raw_disagree_tasks,1))}% всех расхождений "
  "были чисто пунктуационными).\n")

# ---- Methodology ----
R("## Методология\n")
R("**Сопоставление задач.** LS присваивает свои `id` при каждом импорте, "
  "поэтому задачи сопоставлены по **точному совпадению `data.text`**. Во всех "
  f"трёх файлах {len(tasks)} уникальных текстов, множества совпадают полностью. "
  "`stratum` и `data_id` присутствуют в `data` каждой задачи и согласованы "
  "между аннотаторами (страты взяты прямо из экспорта, отдельное сопоставление "
  "с `sample_for_annotation.csv` не потребовалось).\n")
R("**Нормализация span'ов.** Перед сравнением каждый span: (a) обрезается по "
  "пробелам; (b) с краёв снимается пунктуация (`. , ; : ! ? … « » \" ' ( ) "
  "[ ] { } — – -`); (c) смещения проецируются на границы содержимого. "
  "Хранятся и оригинальные, и нормализованные версии — разница даёт оценку "
  "пунктуационного шума.\n")
R("**Метрики (все считаются и попарно, и тройной вариант, где применимо):**\n")
R("- *Token-level Krippendorff α (nominal, binary).* Каждый токен (razdel) "
  "получает бинарную метку «является ли он началом новой клаузы». "
  "Первый токен каждого предложения исключён (тривиально начало для всех). "
  "α считается по плоскому вектору всех токенов всех задач.")
R("- *Boundary-level F1.* Множество смещений начала клауз (кроме позиции 0). "
  "Жадное паросочетание; exact (tol=0) и relaxed (tol=±3 символа). "
  "Macro = среднее по задачам, micro = по суммарным TP/FP/FN.")
R("- *Span-level Jaccard.* Множества `(start,end)`; exact — полное совпадение "
  "пары смещений; relaxed — IoU > 0.5. Macro-среднее по задачам.")
R("- *Sentence-type α (nominal).* По одиночному выбору типа. Выбор в LS "
  "оказался single-select; пропуск типа кодируется как missing.")
R("**Категоризация несогласий** выполнена детерминированной эвристикой "
  "(regex + морфоанализ Natasha: финитные глаголы, инфинитивы, "
  "деепричастия/причастия). "
  + ("LLM-судья: ключи ANTHROPIC_API_KEY / OPENAI_API_KEY обнаружены — "
     "доступен гибридный путь."
     if has_llm else
     "LLM-судья **не использовался**: переменные ANTHROPIC_API_KEY / "
     "OPENAI_API_KEY не заданы в окружении — применён чисто эвристический путь.")
  + "\n")

# ---- Section 1 ----
R("## 1. Описательная статистика\n")
R("| Показатель | Daniil | Shirin | Igor |")
R("|---|---|---|---|")
def drow(label, key, d=2):
    return (f"| {label} | " +
            " | ".join(fnum(desc[a][key], d) for a in ANNOTATORS) + " |")
R(f"| Размечено задач | " + " | ".join(str(desc[a]['n_tasks']) for a in ANNOTATORS) + " |")
R(f"| Без типа предложения (missing) | " +
  " | ".join(str(desc[a]['n_missing_type']) for a in ANNOTATORS) + " |")
R(f"| Всего span'ов | " +
  " | ".join(str(sum(desc[a]['spans_per_task_list'])) for a in ANNOTATORS) + " |")
R(drow("Span'ов на задачу — среднее", "spans_mean"))
R(drow("Span'ов на задачу — медиана", "spans_median"))
R(drow("Span'ов на задачу — std", "spans_std"))
R(drow("Длина span'а, символы — среднее", "span_chars_mean", 1))
R(drow("Длина span'а, символы — медиана", "span_chars_median", 1))
R(drow("Длина span'а, токены — среднее", "span_tok_mean", 2))
R(drow("Длина span'а, токены — медиана", "span_tok_median", 1))
R("")
R("**Распределение типов предложения:**\n")
R("| Тип | Daniil | Shirin | Igor |")
R("|---|---|---|---|")
for t in SENTENCE_TYPES:
    R(f"| {t} | " + " | ".join(str(desc[a]['type_dist'].get(t, 0))
                               for a in ANNOTATORS) + " |")
R("")
R("Графики: `plots/01_spans_per_task_hist.png`, `plots/02_sentence_type_dist.png`.\n")
# observation
spans_tot = {a: sum(desc[a]['spans_per_task_list']) for a in ANNOTATORS}
most = max(spans_tot, key=spans_tot.get)
least = min(spans_tot, key=spans_tot.get)
R(f"_Наблюдение._ {most.capitalize()} выделяет больше всего клауз "
  f"({spans_tot[most]}), {least} — меньше всего ({spans_tot[least]}); "
  f"разница {spans_tot[most]-spans_tot[least]} span'ов "
  f"({round(100*(spans_tot[most]-spans_tot[least])/spans_tot[least])}%) "
  "указывает на систематически разный «порог дробления».\n")

# ---- Section 2 ----
R("## 2. Нормализация span'ов и пунктуационный шум\n")
R(f"- Задач с расхождением границ **до** нормализации: **{raw_disagree_tasks}**")
R(f"- Задач с расхождением границ **после** нормализации: **{norm_disagree_tasks}**")
R(f"- Снято нормализацией (пунктуационный шум): **{punct_noise_tasks}** задач")
R("")
R("Пунктуационный шум — это случаи, когда аннотаторы согласны по сути разбиения, "
  "но один включил в span конечную точку/запятую, а другой нет. Такие расхождения "
  "не отражают методологического несогласия и в категоризации помечены как "
  "`punctuation_noise`.\n")

# ---- Section 3 ----
R("## 3. Метрики согласованности\n")
R("### Сводная таблица\n")
R("| Метрика | Daniil↔Shirin | Daniil↔Igor | Shirin↔Igor | Triple |")
R("|---|---|---|---|---|")
for mname, vals in summary.items():
    cells = []
    for k in pair_keys():
        v = vals.get(k)
        cells.append("—" if v is None else f"{flag(v)} {fnum(v)}")
    R(f"| {mname} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
R("")
R("**Micro-варианты Boundary F1** (по суммарным TP/FP/FN):\n")
R("| Пара | exact micro F1 | relaxed micro F1 | relaxed micro P | relaxed micro R |")
R("|---|---|---|---|---|")
for k in bf1_exact:
    R(f"| {k} | {fnum(bf1_exact[k]['micro_f1'])} | "
      f"{fnum(bf1_relax[k]['micro_f1'])} | {fnum(bf1_relax[k]['micro_p'])} | "
      f"{fnum(bf1_relax[k]['micro_r'])} |")
R("")
R("**Token α — эффект нормализации:**\n")
R("| Scope | α (raw) | α (normalized) |")
R("|---|---|---|")
for k in pair_keys():
    R(f"| {k} | {fnum(tok_alpha_raw.get(k))} | {fnum(tok_alpha_norm.get(k))} |")
R("")
R("График: `plots/03_token_alpha_heatmap.png`.\n")

# ---- Section 4 ----
R("## 4. Категоризация несогласий\n")
R(f"Всего задач с расхождением (raw или normalized): **{len(disagreement_rows)}** "
  f"из {len(tasks)}.\n")
R("| Категория | Кол-во | Доля от расхождений |")
R("|---|---|---|")
for c, n in cat_counter.most_common():
    R(f"| `{c}` | {n} | {round(100*n/max(len(disagreement_rows),1))}% |")
R("")
CAT_RU = {
    "punctuation_noise": "Пунктуационный шум (не реальное несогласие)",
    "homogeneous_predicates": "Однородные сказуемые vs БСП",
    "double_ellipsis": "Двойной эллипсис («поступила туда, куда хотела»)",
    "direct_speech": "Прямая речь",
    "parcellation": "Парцелляция (точка/многоточие в середине)",
    "split_main_clause": "Разорванное главное со встроенным придаточным",
    "demonstrative_words": "Указательные слова (тот/туда/тогда)",
    "infinitive_clause": "Спорная клауза с инфинитивом/неполным предикатом",
    "parenthetical": "Вводные/вставные конструкции",
    "adverbial_participle": "Деепричастные/причастные обороты",
    "comparative": "Сравнительные обороты («как ветер»)",
    "text_artifact": "Артефакты текста (скобки, run-on, опечатки)",
    "other": "Прочее",
}
R("Расшифровка категорий:\n")
for c, _ in cat_counter.most_common():
    R(f"- `{c}` — {CAT_RU.get(c, c)}")
R("")
R("График: `plots/04_disagreement_categories.png`. "
  "Построчная разбивка: `disagreements.csv`.\n")

# ---- Section 5 ----
R("## 5. Срезы по стратам\n")
R("| Страта | n | Token α (triple) | Порог |")
R("|---|---|---|---|")
for s in sorted(STRATA, key=lambda s: (strat_alpha[s][0]
                                       if not np.isnan(strat_alpha[s][0]) else 1)):
    a, n = strat_alpha[s]
    R(f"| {s} | {n} | {fnum(a)} | {flag(a)} |")
R("")
worst_strata = [s for s in STRATA if not np.isnan(strat_alpha[s][0])
                and strat_alpha[s][0] < THRESHOLD]
R(f"_Страты ниже порога:_ {', '.join(worst_strata) if worst_strata else 'нет'}. "
  "График: `plots/05_alpha_by_stratum.png`.\n")

# ---- Section 6 ----
R("## 6. Анализ по длине предложения\n")
R("| Длина (токены) | n | Token α (triple) |")
R("|---|---|---|")
for l in LEN_LABELS:
    a, n = len_alpha[l]
    R(f"| {l} | {n} | {fnum(a)} |")
R("")
valid_len = [(l, len_alpha[l][0]) for l in LEN_LABELS
             if not np.isnan(len_alpha[l][0])]
if len(valid_len) >= 2:
    trend = valid_len[-1][1] - valid_len[0][1]
    R(f"_Гипотеза «чем длиннее — тем хуже IAA»:_ α меняется с "
      f"{fnum(valid_len[0][1])} (короткие) до {fnum(valid_len[-1][1])} (длинные), "
      f"изменение {fnum(trend)} — "
      + ("гипотеза подтверждается." if trend < -0.03
         else "гипотеза НЕ подтверждается / эффект слабый." if abs(trend) <= 0.03
         else "наблюдается обратный тренд.") + "\n")
R("График: `plots/06_alpha_by_length.png`.\n")

# ---- Section 7 ----
R("## 7. Анализ по типу предложения\n")
R("Тип задачи определён mode-голосованием (≥2 из 3 аннотаторов; иначе "
  "«нет консенсуса»).\n")
R("| Тип (mode) | n | Token α (triple) |")
R("|---|---|---|")
for t in SENTENCE_TYPES + ["нет консенсуса"]:
    a, n = type_alpha_split[t]
    R(f"| {t} | {n} | {fnum(a)} |")
R("")
mixed = type_alpha_split.get("Смешанного типа", (float('nan'), 0))
noncons = type_alpha_split.get("нет консенсуса", (float('nan'), 0))
R(f"_Гипотеза «Смешанного типа даёт худший IAA»:_ α(Смешанного) = "
  f"{fnum(mixed[0])}. "
  + ("Подтверждается — это один из худших типов."
     if (not np.isnan(mixed[0]) and mixed[0] ==
         min(v[0] for v in type_alpha_split.values() if not np.isnan(v[0])))
     else "Частично — не самый худший тип; см. таблицу.")
  + f" Задачи без консенсуса по типу (n={noncons[1]}) ожидаемо имеют "
  f"низкое согласие по границам (α={fnum(noncons[0])}).\n")
R("> **Внимание — артефакт малой дисперсии.** Низкая α у категории «Не "
  "сложное» НЕ означает плохого согласия. Простые предложения корректно "
  "размечаются одним span'ом без внутренних границ, поэтому бинарный "
  "token-вектор почти весь нулевой (≈4–7 единиц на ~490 токенов). При "
  "near-zero дисперсии Krippendorff α становится нестабильной: единичные "
  "расхождения резко её обваливают, хотя «сырое» согласие ~99%. Для этой "
  "категории ориентируйтесь на Boundary F1 / долю совпавших разбиений, а не "
  "на α.\n")
R("График: `plots/07_alpha_by_sentence_type.png`.\n")

# ---- Section 8 ----
R("## 8. Top-20 deep dive (худшее попарное согласие границ)\n")
R("Задачи отранжированы по среднему попарному Boundary F1 relaxed (±3). "
  "Span'ы показаны маркерами ⟦…⟧.\n")
for rank, (score, ti) in enumerate(top20, 1):
    tk = tasks[ti]
    cat, descr = CAT.categorize(
        tk.text,
        {a: [(s.start, s.end) for s in tk.anns[a].spans] for a in ANNOTATORS},
        {a: [(s.norm_start, s.norm_end) for s in tk.anns[a].spans] for a in ANNOTATORS},
        {a: tk.anns[a].sentence_type for a in ANNOTATORS})
    R(f"### {rank}. Задача #{ti} — mean Boundary F1 relaxed = {fnum(score)}\n")
    R(f"**Страта:** {tk.stratum}  \n**Исходный текст:** {tk.text}\n")
    for a in ANNOTATORS:
        st_lbl = tk.anns[a].sentence_type or "—"
        R(f"- **{a}** ({len(tk.anns[a].spans)} клауз, тип: {st_lbl}): "
          f"{render_spans(tk, a)}")
    R("")
    R(f"**Категория расхождения:** `{cat}` — {descr}\n")
    # suggested resolution heuristic
    counts = Counter(len(tk.anns[a].spans) for a in ANNOTATORS)
    maj_count = counts.most_common(1)[0]
    if maj_count[1] >= 2:
        R(f"**Предлагаемая разметка (опора на большинство + инструкцию):** "
          f"{maj_count[0]} клауз(ы) — так разметили 2 из 3 аннотаторов; "
          "рекомендуется зафиксировать это как эталон, если не противоречит "
          "правилу о финитном предикате в каждой клаузе.")
    else:
        R("**Предлагаемая разметка:** консенсуса нет (3 разных варианта) — "
          "требует ручного арбитража кейсодержателя.")
    # clarifying question
    qmap = {
        "homogeneous_predicates": "Считать ли цепочку однородных сказуемых "
            "(«ездила, помогала, готовила») одной клаузой или дробить по каждому "
            "сказуемому?",
        "double_ellipsis": "Является ли неполная придаточная часть с "
            "соотносительным словом («туда, куда хотела») отдельной клаузой?",
        "direct_speech": "Размечаются ли клаузы ВНУТРИ прямой речи, или реплика "
            "берётся целиком одним span'ом?",
        "parcellation": "Парцеллят (отделённый точкой фрагмент) — отдельная "
            "клауза или часть предыдущей?",
        "split_main_clause": "Как размечать главное со встроенным придаточным: "
            "двумя span'ами главного + придаточное, перекрытием или иначе?",
        "demonstrative_words": "К какой части относить указательное слово "
            "(тот/туда/тогда) — к главной или к придаточной?",
        "infinitive_clause": "Считать ли оборот с зависимым инфинитивом "
            "(«хочет рассказать», «не знаю как дышать») отдельной клаузой?",
        "parenthetical": "Вводные/вставные конструкции — выделять как отдельную "
            "клаузу или нет?",
        "adverbial_participle": "Подтвердить: деепричастные/причастные обороты "
            "НЕ являются отдельными клаузами.",
        "comparative": "Подтвердить: сравнительные обороты («как ветер») НЕ "
            "являются отдельными клаузами.",
        "text_artifact": "Как размечать предложения с артефактами "
            "(несбалансированные скобки, run-on без пунктуации)?",
        "punctuation_noise": "Уточнить в инструкции: включать ли конечную "
            "пунктуацию в границы span'а.",
        "other": "Требуется ручной разбор кейсодержателем.",
    }
    R(f"**Вопрос кейсодержателю:** {qmap.get(cat, qmap['other'])}\n")

# ---- Section 9 ----
R("## 9. Анализ sentence_type\n")
R("3×3 confusion-матрицы по парам (строки — первый аннотатор, столбцы — второй). "
  "Графики: `plots/09_confusion_*.png`.\n")
for k, m in conf.items():
    a, b = k.split("-")
    total = m.sum()
    agree = np.trace(m)
    R(f"**{a} × {b}:** совпало {agree}/{total} "
      f"({round(100*agree/max(total,1))}%).")
    # top confusion cell off-diagonal
    off = m.copy()
    np.fill_diagonal(off, 0)
    if off.max() > 0:
        i, j = np.unravel_index(off.argmax(), off.shape)
        R(f"  Чаще всего путают: «{SENTENCE_TYPES[i]}» ({a}) ↔ "
          f"«{SENTENCE_TYPES[j]}» ({b}) — {off[i,j]} раз.")
R("")
# aggregate most-confused pair
agg = np.zeros((len(SENTENCE_TYPES), len(SENTENCE_TYPES)))
for m in conf.values():
    sym = m + m.T
    agg += sym
np.fill_diagonal(agg, 0)
i, j = np.unravel_index(agg.argmax(), agg.shape)
R(f"_Главный источник путаницы по типу_ (сумма по всем парам): "
  f"«{SENTENCE_TYPES[i]}» ↔ «{SENTENCE_TYPES[j]}» — {int(agg[i,j])} расхождений. "
  "Это ожидаемо: «Смешанного типа» по определению пересекается с СПП/ССП/БСП.\n")

# ---- Recommendations ----
R("## Рекомендации по доработке инструкции\n")
R("Приоритет = (импакт на IAA) × (частота категории). От высокого к низкому.\n")
# build prioritized list from category counts excluding pure noise
prio = [(c, n) for c, n in cat_counter.most_common()
        if c not in ("punctuation_noise",)]
rec_text = {
    "homogeneous_predicates": "**Однородные сказуемые.** Дать явное правило: "
        "цепочка однородных сказуемых при одном подлежащем — ОДНА клауза "
        "(«я встала, умылась, пошла»), даже при запятых. Привести 3-4 примера "
        "с границей между «однородные сказуемые» и «БСП».",
    "split_main_clause": "**Разорванное главное со встроенным придаточным.** "
        "Зафиксировать единый способ: либо два span'а главной части + "
        "отдельный span придаточного, либо непрерывный span с вложением. "
        "Сейчас аннотаторы используют разные схемы — это даёт overlap'ы.",
    "double_ellipsis": "**Неполные придаточные с соотносительными словами** "
        "(«поступила туда, куда хотела»). Указать, что придаточное с эллипсисом "
        "сказуемого всё равно выделяется как отдельная клауза (или наоборот) — "
        "и придерживаться одного варианта.",
    "infinitive_clause": "**Зависимый инфинитив.** Прописать критерий: оборот с "
        "инфинitивом при модальном/фазовом глаголе («хочет рассказать», "
        "«начал писать») — НЕ отдельная клауза; инфинитив с собственной "
        "придаточной семантикой — отдельная.",
    "demonstrative_words": "**Указательные/соотносительные слова** "
        "(тот, туда, тогда, так). Правило: соотносительное слово остаётся в "
        "ГЛАВНОЙ части, придаточное начинается с союзного слова.",
    "direct_speech": "**Прямая речь.** Явно указать, размечаются ли клаузы "
        "внутри кавычек или реплика берётся одним span'ом.",
    "parcellation": "**Парцелляция.** Определить, считается ли отделённый "
        "точкой фрагмент отдельной клаузой.",
    "parenthetical": "**Вводные vs вставные.** Дать список вводных слов "
        "(не выделяются) и критерий вставной конструкции (выделяется).",
    "adverbial_participle": "**Деепричастные/причастные обороты.** Прямо "
        "написать: обособленные обороты НЕ являются отдельными клаузами "
        "(нет финитного глагола).",
    "comparative": "**Сравнительные обороты.** Прямо написать: оборот с «как», "
        "«словно» без собственного сказуемого — НЕ клауза.",
    "text_artifact": "**Артефакты текста.** Дать инструкцию по run-on "
        "предложениям и несбалансированным скобкам (например, размечать по "
        "смыслу, помечать комментарием).",
    "other": "**Прочие случаи.** Собрать в отдельный регулярный разбор "
        "сложных кейсов с кейсодержателем.",
}
for rank, (c, n) in enumerate(prio, 1):
    R(f"{rank}. {rec_text.get(c, CAT_RU.get(c, c))} _(категория `{c}`, "
      f"{n} задач)_")
R("")
R("Дополнительно: добавить правило о **краевой пунктуации** "
  f"(снимает {punct_noise_tasks} ложных расхождений) — span НЕ должен включать "
  "конечную точку/запятую, либо это нормализуется автоматически и не считается "
  "ошибкой.\n")

# ---- Artifacts ----
R("## Артефакты\n")
R("- `analysis/iaa_report/report.md` — настоящий отчёт")
R("- `analysis/iaa_report/metrics.csv` — все числовые метрики")
R("- `analysis/iaa_report/disagreements.csv` — построчные несогласия с категориями")
R(f"- `analysis/iaa_report/per_row/` — {len(tasks)} JSON-файлов (разбор по задаче)")
R("- `analysis/iaa_report/plots/01_spans_per_task_hist.png`")
R("- `analysis/iaa_report/plots/02_sentence_type_dist.png`")
R("- `analysis/iaa_report/plots/03_token_alpha_heatmap.png`")
R("- `analysis/iaa_report/plots/04_disagreement_categories.png`")
R("- `analysis/iaa_report/plots/05_alpha_by_stratum.png`")
R("- `analysis/iaa_report/plots/06_alpha_by_length.png`")
R("- `analysis/iaa_report/plots/07_alpha_by_sentence_type.png`")
R("- `analysis/iaa_report/plots/09_confusion_{daniil_shirin,daniil_igor,shirin_igor}.png`")
R("")
R("**Воспроизведение:** `.venv/bin/python analysis/run_iaa.py` "
  f"(SEED={C.SEED}; библиотеки: krippendorff, simpledorff, natasha, razdel, "
  "pandas, numpy, matplotlib, seaborn, scikit-learn).\n")

with open(C.OUT_DIR / "report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT))

# ---------------------------------------------------------------------------
# console summary
# ---------------------------------------------------------------------------
print("\n===== IAA SUMMARY =====")
print(f"tasks={len(tasks)}  LLM judge={'yes' if has_llm else 'no (heuristic only)'}")
print(f"Token alpha triple (norm) = {fnum(tok_triple)}")
print(f"Sentence-type alpha triple = {fnum(typ_triple)}")
print(f"Boundary F1 relaxed (mean pairs) = {fnum(mean_brelax)}")
print(f"punctuation noise removed = {punct_noise_tasks} tasks")
print(f"disagreement tasks = {len(disagreement_rows)}")
print("category counts:", dict(cat_counter.most_common()))
print("DONE -> analysis/iaa_report/")
