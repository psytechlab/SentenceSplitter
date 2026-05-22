"""Задача 1: полный построчный список расхождений -> 01_disagreement_full_list.md"""
import csv
import numpy as np
import lib
import categorize_indep as ci

tasks = lib.load_tasks()
auto_rows = {int(r["task_idx"]): r for r in
             csv.DictReader(open(lib.ROOT / "analysis/iaa_report/disagreements.csv"))}

CAT_RU = {
    "punctuation_noise": "Пунктуационный шум",
    "infinitive_clause": "Инфинитивная клауза",
    "text_artifact": "Артефакт текста",
    "adverbial_participle": "Деепричастный/причастный оборот",
    "other": "Прочее",
    "split_main_clause": "Разорванное главное + придаточное",
    "direct_speech": "Прямая речь",
    "homogeneous_predicates": "Однородные сказуемые",
    "parenthetical": "Вводные/вставные",
    "parcellation": "Парцелляция",
    "demonstrative_words": "Указательные слова",
    "double_ellipsis": "Двойной эллипсис",
    "comparative": "Сравнительный оборот",
}
# порядок групп = по убыванию частоты в auto
GROUP_ORDER = ["punctuation_noise", "infinitive_clause", "text_artifact",
               "adverbial_participle", "other", "split_main_clause",
               "direct_speech", "homogeneous_predicates", "parenthetical",
               "parcellation", "demonstrative_words", "double_ellipsis",
               "comparative"]


def viz(text, spans):
    """Рендер разметки маркерами ⟦…⟧ по нормализованным офсетам."""
    marks = {}
    for s in spans:
        marks.setdefault(s["norm_start"], []).append("open")
        marks.setdefault(s["norm_end"], []).append("close")
    out = []
    for i, ch in enumerate(text):
        for m in marks.get(i, []):
            out.append("⟦" if m == "open" else "⟧")
        out.append(ch)
    for m in marks.get(len(text), []):
        out.append("⟦" if m == "open" else "⟧")
    return "".join(out).replace("\n", " ")


def task_metrics(tk):
    return (lib.task_token_alpha_triple(tk, True),
            lib.task_boundary_f1_relaxed_avg(tk, 3, True),
            lib.task_span_jaccard_exact_avg(tk, True))


# собрать данные по всем 195 задачам с расхождением
rows = []
for tk in tasks:
    ti = tk["task_idx"]
    if ti not in auto_rows:
        continue
    auto_cat = auto_rows[ti]["category"]
    my_cat, my_reason = ci.categorize(tk)
    a, f1, jac = task_metrics(tk)
    rows.append({"tk": tk, "ti": ti, "auto": auto_cat, "mine": my_cat,
                 "reason": my_reason, "alpha": a, "f1": f1, "jac": jac})


def fmt(x):
    return "nan" if x != x else f"{x:.3f}"


L = []
L.append("# Задача 1. Полный построчный список расхождений\n")
L.append("_Независимая мета-валидация. Все 195 задач с расхождением границ "
         "(raw). Группировка по **автоматической** категории из "
         "`disagreements.csv`; внутри группы — сортировка по возрастанию "
         "Boundary F1 relaxed (severity ↓)._\n")
L.append("Метрики задачи: **token α (triple)** — Krippendorff α по бинарному "
         "вектору границ 3 аннотаторов; **F1** — Boundary F1 relaxed (±3), "
         "среднее по 3 парам; **Jacc** — Span Jaccard exact, среднее по парам. "
         "α=nan — нулевая дисперсия (все три согласны, что внутренних границ "
         "нет, либо данных недостаточно).\n")

dist = []
for cat in GROUP_ORDER:
    grp = [r for r in rows if r["auto"] == cat]
    if not grp:
        continue
    grp.sort(key=lambda r: r["f1"])
    L.append(f"\n## {cat} — {CAT_RU.get(cat, cat)} ({len(grp)})\n")
    for r in grp:
        tk = r["tk"]
        L.append(f"### Задача #{r['ti']} · `{tk['stratum']}`\n")
        L.append(f"> {tk['text']}\n")
        for ann in lib.ANNOTATORS:
            ta = tk["anns"][ann]
            st = ta["stype"] or "—"
            L.append(f"- **{ann}** ({len(ta['spans'])} кл., {st}): "
                     f"{viz(tk['text'], ta['spans'])}")
        L.append("")
        L.append(f"- авто-категория: `{r['auto']}` · моя категория: `{r['mine']}`")
        if r["mine"] != r["auto"]:
            L.append(f"- ⚠️ расхождение категоризации: {r['reason']}")
        L.append(f"- метрики: token α={fmt(r['alpha'])} · "
                 f"F1 relaxed={fmt(r['f1'])} · Jaccard exact={fmt(r['jac'])}")
        L.append("")
    al = [r["alpha"] for r in grp if r["alpha"] == r["alpha"]]
    f1s = [r["f1"] for r in grp]
    js = [r["jac"] for r in grp]
    agree = sum(1 for r in grp if r["mine"] == r["auto"])
    L.append(f"**Summary `{cat}`:** n={len(grp)} · "
             f"mean token α={fmt(np.mean(al)) if al else 'nan'} "
             f"(n_valid={len(al)}) · mean F1={fmt(np.mean(f1s))} · "
             f"mean Jaccard={fmt(np.mean(js))} · "
             f"согласие моей категоризации с авто: {agree}/{len(grp)} "
             f"= {100*agree/len(grp):.1f}%")
    dist.append((cat, len(grp), sum(1 for r in rows if r["mine"] == cat), agree))
    L.append("")

# итоговая таблица распределения
L.append("\n## Итоговое распределение категорий\n")
L.append("| Категория | n (auto) | n (моё) | согласие (по auto-группе) % |")
L.append("|---|---|---|---|")
for cat in GROUP_ORDER:
    d = next((x for x in dist if x[0] == cat), None)
    if d:
        L.append(f"| `{cat}` | {d[1]} | {d[2]} | {100*d[3]/d[1]:.1f} |")
# категории, появившиеся только у меня
mine_only = set(r["mine"] for r in rows) - set(GROUP_ORDER)
for cat in sorted(mine_only):
    L.append(f"| `{cat}` (только моё) | 0 | "
             f"{sum(1 for r in rows if r['mine']==cat)} | — |")
total_agree = sum(1 for r in rows if r["mine"] == r["auto"])
L.append(f"| **ИТОГО** | {len(rows)} | {len(rows)} | "
         f"**{100*total_agree/len(rows):.1f}** |")

real = [r for r in rows if r["auto"] != "punctuation_noise"]
real_agree = sum(1 for r in real if r["mine"] == r["auto"])
L.append(f"\n_Согласие на 97 `punctuation_noise`: "
         f"{sum(1 for r in rows if r['auto']=='punctuation_noise' and r['mine']==r['auto'])}/97. "
         f"Согласие на {len(real)} задачах с реальным (норм.) расхождением: "
         f"{real_agree}/{len(real)} = {100*real_agree/len(real):.1f}%._")

open(lib.ROOT / "validation/01_disagreement_full_list.md", "w",
     encoding="utf-8").write("\n".join(L))
print("written 01_disagreement_full_list.md  lines:", len(L))
print("total agreement:", total_agree, "/", len(rows),
      f"= {100*total_agree/len(rows):.1f}%")
print("real-disagreement agreement:", real_agree, "/", len(real),
      f"= {100*real_agree/len(real):.1f}%")
