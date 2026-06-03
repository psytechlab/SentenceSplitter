"""Подготовка данных для FINAL_REPORT: экспорт categorization_by_agent.csv +
расчёт чисел для разделов про гранулярность и дивергенцию метрик."""
import csv
from collections import Counter
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import lib
import categorize_indep as ci

tasks = lib.load_tasks()
A = lib.ANNOTATORS
auto_cat = {int(r["task_idx"]): r["category"]
            for r in csv.DictReader(open(lib.ROOT / "analysis/iaa_report/disagreements.csv"))}


def spanstr(tk, a):
    return "; ".join(
        f"[{s['norm_start']},{s['norm_end']}]{tk['text'][s['norm_start']:s['norm_end']]}"
        for s in tk["anns"][a]["spans"])


# ---- экспорт categorization_by_agent.csv ----------------------------------
rows = []
for tk in tasks:
    ti = tk["task_idx"]
    if ti not in auto_cat:
        continue
    mc, mr = ci.categorize(tk)
    rows.append({
        "task_idx": ti, "text": tk["text"], "stratum": tk["stratum"],
        "auto_category": auto_cat[ti], "agent_category": mc,
        "agent_reason": mr, "agreement": int(mc == auto_cat[ti]),
        "token_alpha_triple": round(lib.task_token_alpha_triple(tk, True), 4),
        "boundary_f1_relaxed_avg": round(lib.task_boundary_f1_relaxed_avg(tk, 3, True), 4),
        "span_jaccard_exact_avg": round(lib.task_span_jaccard_exact_avg(tk, True), 4),
        "daniil_n_spans": len(tk["anns"]["daniil"]["spans"]),
        "shirin_n_spans": len(tk["anns"]["shirin"]["spans"]),
        "igor_n_spans": len(tk["anns"]["igor"]["spans"]),
        "daniil_spans": spanstr(tk, "daniil"),
        "shirin_spans": spanstr(tk, "shirin"),
        "igor_spans": spanstr(tk, "igor"),
        "daniil_type": tk["anns"]["daniil"]["stype"] or "",
        "shirin_type": tk["anns"]["shirin"]["stype"] or "",
        "igor_type": tk["anns"]["igor"]["stype"] or "",
    })
with open(lib.ROOT / "validation/data/categorization_by_agent.csv", "w",
          encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("categorization_by_agent.csv:", len(rows), "строк")

# ---- описательная статистика по аннотаторам -------------------------------
print("\n=== ГРАНУЛЯРНОСТЬ ===")
for a in A:
    tot = sum(len(tk["anns"][a]["spans"]) for tk in tasks)
    lens = [s["norm_end"] - s["norm_start"]
            for tk in tasks for s in tk["anns"][a]["spans"]]
    print(f"{a}: spans={tot}  spans/task={tot/200:.2f}  "
          f"mean_span_len={np.mean(lens):.1f}  median={np.median(lens):.0f}")

# гранулярность по моим категориям: средн. число span'ов по аннотаторам
print("\n=== spans/task по категориям (моя категоризация) ===")
mycat = {r["task_idx"]: r["agent_category"] for r in rows}
bycat = {}
for tk in tasks:
    ti = tk["task_idx"]
    if ti not in mycat:
        continue
    bycat.setdefault(mycat[ti], []).append(tk)
for c, sub in sorted(bycat.items(), key=lambda x: -len(x[1])):
    means = {a: np.mean([len(tk["anns"][a]["spans"]) for tk in sub]) for a in A}
    spread = max(means.values()) - min(means.values())
    print(f"{c:24s} n={len(sub):3d}  d={means['daniil']:.2f} "
          f"s={means['shirin']:.2f} i={means['igor']:.2f}  "
          f"spread={spread:.2f}")

# доля задач, где igor дал МЕНЬШЕ всех span'ов
igor_fewest = sum(1 for tk in tasks
                  if len(tk["anns"]["igor"]["spans"]) <
                  min(len(tk["anns"]["daniil"]["spans"]),
                      len(tk["anns"]["shirin"]["spans"])))
shirin_most = sum(1 for tk in tasks
                  if len(tk["anns"]["shirin"]["spans"]) >
                  max(len(tk["anns"]["daniil"]["spans"]),
                      len(tk["anns"]["igor"]["spans"])))
print(f"\nigor строго наименьшее число span'ов: {igor_fewest}/200")
print(f"shirin строго наибольшее: {shirin_most}/200")

# ---- примеры дивергенции метрик -------------------------------------------
print("\n=== ПРИМЕРЫ ДИВЕРГЕНЦИИ ===")
def viz(tk, a):
    text = tk["text"]
    marks = {}
    for s in tk["anns"][a]["spans"]:
        marks.setdefault(s["norm_start"], []).append("⟦")
        marks.setdefault(s["norm_end"], []).append("⟧")
    out = []
    for i, ch in enumerate(text):
        for m in marks.get(i, []):
            out.append(m)
        out.append(ch)
    for m in marks.get(len(text), []):
        out.append(m)
    return "".join(out)

tdict = {tk["task_idx"]: tk for tk in tasks}
for ti in [4, 101, 199, 21, 22, 181]:
    tk = tdict[ti]
    print(f"\n--- #{ti}  α={lib.task_token_alpha_triple(tk,True):.3f} "
          f"Jacc={lib.task_span_jaccard_exact_avg(tk,True):.3f} ---")
    for a in A:
        print(f"  {a}: {viz(tk,a)}")
