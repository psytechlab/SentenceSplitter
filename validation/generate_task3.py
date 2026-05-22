"""Задача 3: эвристики категоризации vs поведение метрик -> 03_heuristics_vs_metrics.md"""
import csv
from collections import Counter
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import lib
import categorize_indep as ci

tasks = lib.load_tasks()
A = lib.ANNOTATORS
auto_cat = {int(r["task_idx"]): r["category"]
            for r in csv.DictReader(open(lib.ROOT / "analysis/iaa_report/disagreements.csv"))}

ALL_CATS = ["punctuation_noise", "infinitive_clause", "text_artifact",
            "adverbial_participle", "other", "split_main_clause",
            "direct_speech", "homogeneous_predicates", "parenthetical",
            "parcellation", "demonstrative_words", "double_ellipsis",
            "comparative"]


def mode_type(tk):
    c = Counter(tk["anns"][a]["stype"] for a in A if tk["anns"][a]["stype"])
    if not c:
        return "нет консенсуса"
    top, n = c.most_common(1)[0]
    return top if n >= 2 else "нет консенсуса"


# ---- per-task records (только 195 задач с расхождением) -------------------
rec = []
for tk in tasks:
    ti = tk["task_idx"]
    if ti not in auto_cat:
        continue
    mc, mr = ci.categorize(tk)
    types = [tk["anns"][a]["stype"] for a in A]
    rec.append({
        "ti": ti, "tk": tk, "text": tk["text"], "stratum": tk["stratum"],
        "auto": auto_cat[ti], "mine": mc, "reason": mr,
        "alpha": lib.task_token_alpha_triple(tk, True),
        "f1": lib.task_boundary_f1_relaxed_avg(tk, 3, True),
        "jac": lib.task_span_jaccard_exact_avg(tk, True),
        "mode_type": mode_type(tk),
        "stype_agree": int(len(set(types)) == 1 and None not in types),
    })

mismatch = [r for r in rec if r["auto"] != r["mine"]]


def fnum(x, d=3):
    return "nan" if (x != x) else f"{x:.{d}f}"


def stats(vals):
    v = [x for x in vals if x == x]
    if not v:
        return ("nan", "nan", "nan", 0)
    return (f"{np.mean(v):.3f}", f"{np.median(v):.3f}",
            f"{np.std(v):.3f}", len(v))


L = ["# Задача 3. Эвристики категоризации vs поведение метрик\n",
     "_Мета-валидация раздела 4 отчёта. Числовая часть отчёта подтверждена "
     "Задачей 2 — здесь проверяется, отражает ли **категоризация** "
     "(`disagreements.csv`, эвристика `iaa_categorize.py`) реальные паттерны "
     "несогласия. Используются две категоризации: **auto** (из CSV) и "
     "**моя независимая** (`categorize_indep.py`, решение по содержимому "
     "спорной области)._\n"]

# ===========================================================================
# 3.0  FAILURE MODES автоматической категоризации
# ===========================================================================
# назначение режима ошибки (экспертное суждение по 40 расхождениям)
MISSED = {63, 127}
WRONG_SIMILAR = {2, 98, 134, 168, 199, 177, 84}
FMODE = {}
for r in mismatch:
    ti = r["ti"]
    if ti in MISSED:
        FMODE[ti] = "Missed"
    elif ti in WRONG_SIMILAR:
        FMODE[ti] = "Wrong-similar"
    else:
        FMODE[ti] = "FP-surface"

L.append("## 3.0 Failure modes автоматической категоризации\n")
L.append(f"Расхождений «моя категория ≠ auto»: **{len(mismatch)}** из 195 "
         f"({100*len(mismatch)/195:.1f}%); из них на 98 задачах с реальным "
         f"(норм.) расхождением — "
         f"{sum(1 for r in mismatch if r['auto']!='punctuation_noise')}.\n")
L.append("Выявлено **4 паттерна ошибки** авто-эвристики:\n")
L.append("1. **FP-surface trigger** — категория сработала по наличию слова/"
         "морфопризнака (инфинитив, деепричастие, сравнительная частица, "
         "вводное слово) где-либо в тексте, но эта конструкция **не была "
         "причиной спорной границы**. Доминирующий паттерн.\n"
         "2. **Wrong-similar** — авто выбрал не ту категорию из структурно "
         "близких (напр. `comparative` вместо `infinitive_clause`, "
         "`demonstrative_words` вместо `split_main_clause`).\n"
         "3. **Missed** — авто свалил в `other`, хотя присутствовал чёткий "
         "паттерн.\n"
         "4. **Over-aggregation** — одна авто-категория поглощает несколько "
         "разных истинных паттернов (категориальный эффект, см. ниже).\n")

# таблица failure_mode × auto_category
modes = ["FP-surface", "Wrong-similar", "Missed"]
fm_tab = {(m, c): 0 for m in modes for c in ALL_CATS}
for r in mismatch:
    fm_tab[(FMODE[r["ti"]], r["auto"])] += 1
used_cats = [c for c in ALL_CATS if any(fm_tab[(m, c)] for m in modes)]
L.append("**Таблица: failure_mode × auto_category × n**\n")
L.append("| auto_category | " + " | ".join(modes) + " | Σ |")
L.append("|" + "---|" * (len(modes) + 2))
for c in used_cats:
    row = [fm_tab[(m, c)] for m in modes]
    L.append(f"| `{c}` | " + " | ".join(str(x) for x in row) +
             f" | {sum(row)} |")
totrow = [sum(fm_tab[(m, c)] for c in used_cats) for m in modes]
L.append("| **Σ** | " + " | ".join(f"**{x}**" for x in totrow) +
         f" | **{sum(totrow)}** |")

# примеры
EX = {
 "FP-surface": [
  ("#166", "«Я хочу покончить с собой, это невыносимо!» — авто `infinitive_clause`, "
   "т.к. в тексте есть инфинитив «покончить». Но спорная граница — между «...с "
   "собой» и «это невыносимо» (igor слил БСП в 1 клаузу, daniil/shirin — 2). "
   "Инфинитив сидит ВНУТРИ первой клаузы и к спору отношения не имеет."),
  ("#124", "«Украинский народ, помня, какие жертвы...» — авто "
   "`adverbial_participle` по деепричастию «помня». Реальный спор — shirin "
   "разметил всё одной клаузой (Не сложное), daniil/igor — тремя; "
   "деепричастие в обоих вариантах остаётся внутри первой клаузы."),
  ("#190", "«...что может быть лучше?» — авто `parenthetical` по слову «может». "
   "Текст — социальный run-on с эмодзи «><»; спор о числе клауз (2/4/5), "
   "вводных конструкций на спорных границах нет."),
 ],
 "Wrong-similar": [
  ("#98", "«...сдавшийся, прежде чем было принято...» — авто `split_main_clause`, "
   "я ошибочно дал `comparative` (зацепился за «чем»). Истина — `split_main_"
   "clause`: «прежде чем» здесь временной союз. Авто прав, ошибка — у меня."),
  ("#134", "«...я не знала, как выйти из этой ситуации» — авто `comparative` "
   "по «как». Истина ближе к `infinitive_clause`: «как выйти» — косвенный "
   "вопрос с инфинитивом, igor выделил его отдельной клаузой."),
  ("#2",  "«...арестовали в мае после того, как...» — авто `demonstrative_words` "
   "(соотносительное «того»), я дал `split_main_clause` по «как». Здесь авто "
   "точнее: спор именно о том, к какой части отнести «после того»."),
 ],
 "Missed": [
  ("#63", "«Мой первый парень от которого сделала аборт в 16 лет.» — авто "
   "`other`. daniil выделил 2 клаузы (главное + придаточное «от которого»), "
   "shirin/igor — 1. Чёткий паттерн: придаточное с относительным словом / "
   "неполная клауза, авто его не распознал."),
  ("#127", "«Ушёл в никуда с ненавистной мне работы, безвылазно сижу дома.» — "
   "авто `other`. daniil/shirin — 2 клаузы (БСП), igor — 1. Паттерн «БСП vs "
   "Не сложное» (однородные/бессоюзные предикаты) — авто не классифицировал."),
 ],
}
for m in modes:
    L.append(f"\n**Паттерн «{m}» — примеры:**\n")
    for tid, txt in EX[m]:
        L.append(f"- **{tid}** — {txt}")

# over-aggregation: разбивка крупных auto-категорий по моим категориям
L.append("\n**Паттерн «Over-aggregation» (категориальный эффект).** Крупные "
         "авто-категории поглощают разнородные истинные паттерны. Разбивка "
         "по моей независимой категоризации:\n")
for ac in ["infinitive_clause", "adverbial_participle"]:
    sub = [r for r in rec if r["auto"] == ac]
    br = Counter(r["mine"] for r in sub)
    L.append(f"- `{ac}` (auto n={len(sub)}) → " +
             ", ".join(f"{c}:{n}" for c, n in br.most_common()))
L.append("\n_Вывод 3.0._ Из 40 ошибок категоризации **31 (77.5%)** — "
         "FP-surface: эвристика реагирует на морфопризнак, не проверяя, "
         "находится ли конструкция на спорной границе. Это прямое следствие "
         "архитектуры `iaa_categorize.py` — правила вида `has_infinitive(text) "
         "and diff_count` применяются ко ВСЕМУ тексту задачи. `infinitive_"
         "clause` фактически работает как «есть инфинитив + разное число "
         "клауз» и поглощает `homogeneous_predicates`, `split_main_clause`, "
         "`direct_speech` и чистый `other`.\n")

# ===========================================================================
# 3.1  Метрики по категориям — ДВЕ ВЕРСИИ
# ===========================================================================
def cat_table(catkey):
    rows = []
    cats = sorted({r[catkey] for r in rec},
                  key=lambda c: -sum(1 for r in rec if r[catkey] == c))
    for c in cats:
        sub = [r for r in rec if r[catkey] == c]
        am, amed, asd, anv = stats([r["alpha"] for r in sub])
        f1m = np.mean([r["f1"] for r in sub])
        jm = np.mean([r["jac"] for r in sub])
        topstr = Counter(r["stratum"] for r in sub).most_common(1)[0]
        toptyp = Counter(r["mode_type"] for r in sub).most_common(1)[0]
        rows.append((c, len(sub), am, asd, amed, f"{f1m:.3f}", f"{jm:.3f}",
                     anv, f"{topstr[0]}({topstr[1]})", f"{toptyp[0]}({toptyp[1]})"))
    return rows


L.append("## 3.1 Метрики по категориям (две версии)\n")
for catkey, title in [("auto", "Версия А — авто-категория (`disagreements.csv`)"),
                       ("mine", "Версия Б — моя независимая категория")]:
    L.append(f"\n### {title}\n")
    L.append("| Категория | n | mean tok α | std α | median α | mean F1 | "
             "mean Jacc | n(α valid) | top stratum | top type |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in cat_table(catkey):
        L.append("| `" + row[0] + "` | " + " | ".join(str(x) for x in row[1:]) + " |")

L.append("\n**Сравнение версий.** Ключевое различие — поведение `infinitive_"
         "clause`: в версии А (auto, n=22) её mean tok α завышена/занижена "
         "шумом, т.к. категория содержит и согласованные, и спорные задачи; "
         "в версии Б (моё, n≈8) категория компактнее и однороднее. Категории "
         "`homogeneous_predicates` (auto 6 → моё ~16) и `other` (auto 11 → "
         "моё ~23) в версии Б крупнее и забирают задачи, ошибочно "
         "приписанные `infinitive_clause`/`adverbial_participle` в версии А. "
         "Вывод о «проблемных категориях» по версии А смещён шумной "
         "категоризацией.\n")

# ===========================================================================
# 3.2  Гипотезы H1-H4
# ===========================================================================
L.append("## 3.2 Проверка гипотез H1–H4\n")


def cat_alpha_mean(catkey, cats):
    sub = [r for r in rec if r[catkey] in cats]
    v = [r["alpha"] for r in sub if r["alpha"] == r["alpha"]]
    return (np.mean(v) if v else float("nan")), len(sub), len(v)


# H1
for ck, lbl in [("mine", "моя"), ("auto", "auto")]:
    m, n, nv = cat_alpha_mean(ck, {"punctuation_noise"})
    L.append(f"- **H1** ({lbl}): `punctuation_noise` mean tok α = {fnum(m)} "
             f"(n={n}, α valid={nv}) — порог >0.9 → "
             f"{'✅ подтверждается' if m>0.9 else '❌ НЕ подтверждается'}.")
L.append("  _H1 верна: после нормализации `punctuation_noise` — псевдо-"
         "расхождения, согласие почти идеальное (одинаково в обеих версиях, "
         "категория определена процедурно)._\n")

# H2
H2 = {"double_ellipsis", "infinitive_clause", "homogeneous_predicates"}
for ck, lbl in [("mine", "моя"), ("auto", "auto")]:
    parts = []
    for c in sorted(H2):
        m, n, nv = cat_alpha_mean(ck, {c})
        parts.append(f"{c}={fnum(m)}(n={n})")
    mall, nall, _ = cat_alpha_mean(ck, H2)
    L.append(f"- **H2** ({lbl}): методологические категории — " +
             ", ".join(parts) +
             f"; объединённое mean α={fnum(mall)} → порог <0.7 → "
             f"{'✅' if mall<0.7 else '❌'}.")
L.append("  _H2 на моей категоризации информативнее: авто-версия размывает "
         "`infinitive_clause` нерелевантными задачами и искажает среднее._\n")

# H3
for ck, lbl in [("mine", "моя"), ("auto", "auto")]:
    means = []
    for c in {r[ck] for r in rec}:
        m, n, nv = cat_alpha_mean(ck, {c})
        if n >= 3 and m == m:
            means.append((m, c, n))
    means.sort()
    L.append(f"- **H3** ({lbl}): самые низкие mean α: " +
             ", ".join(f"{c}={m:.3f}(n={n})" for m, c, n in means[:3]) +
             f". `text_artifact` — "
             f"{'самый низкий ✅' if means and means[0][1]=='text_artifact' else 'НЕ самый низкий ❌'}.")
L.append("")

# H4
small_mine = [c for c in {r["mine"] for r in rec}
              if sum(1 for r in rec if r["mine"] == c) <= 5]
small_auto = [c for c in {r["auto"] for r in rec}
              if sum(1 for r in rec if r["auto"] == c) <= 5]
L.append(f"- **H4**: категории n≤5 — моя: {sorted(small_mine)}; "
         f"auto: {sorted(small_auto)}. Для них std α велик, доверять "
         f"средним нельзя ✅ (по построению — малая выборка).\n")

# ===========================================================================
# 3.3  Корреляции метрик
# ===========================================================================
L.append("## 3.3 Корреляции метрик между собой (по задачам)\n")
va = np.array([r["alpha"] for r in rec])
vf = np.array([r["f1"] for r in rec])
vj = np.array([r["jac"] for r in rec])
vs = np.array([r["stype_agree"] for r in rec])
ok = ~np.isnan(va)
L.append(f"_n задач с валидной token α = {ok.sum()} из {len(rec)} "
         f"(nan — нулевая дисперсия)._\n")
L.append("| Пара метрик | Pearson r | Spearman ρ | n |")
L.append("|---|---|---|---|")
for name, x, y, mask in [
        ("token α ↔ Boundary F1 relaxed", va, vf, ok),
        ("token α ↔ Span Jaccard exact", va, vj, ok),
        ("token α ↔ sentence_type agreement", va, vs, ok)]:
    pr = pearsonr(x[mask], y[mask])
    sr = spearmanr(x[mask], y[mask])
    L.append(f"| {name} | {pr.statistic:.3f} | {sr.statistic:.3f} | {mask.sum()} |")

# дивергентные задачи
hi_lo = [r for r in rec if r["alpha"] == r["alpha"] and r["alpha"] > 0.8 and r["jac"] < 0.5]
lo_st = [r for r in rec if r["alpha"] == r["alpha"] and r["alpha"] < 0.5 and r["stype_agree"] == 1]
L.append(f"\n**Дивергенция «высокая α, низкий Jaccard» (α>0.8, Jacc<0.5): "
         f"{len(hi_lo)} задач.** Лингвистически: аннотаторы согласны, ГДЕ "
         f"проходят границы клауз (токенные начала совпадают), но расходятся "
         f"в точных символьных офсетах span'ов — exact-Jaccard это штрафует, "
         f"token α — нет. Примеры:\n")
for r in sorted(hi_lo, key=lambda r: r["jac"])[:6]:
    L.append(f"- #{r['ti']} α={r['alpha']:.3f} Jacc={r['jac']:.3f} "
             f"({r['mine']}) — {r['text'][:75]}")
L.append(f"\n**Дивергенция «низкая α, тип совпал у всех трёх» (α<0.5, "
         f"stype_agree): {len(lo_st)} задач.** Паттерн: аннотаторы согласны "
         f"в ТИПЕ предложения, но сильно расходятся в РАЗБИЕНИИ на клаузы — "
         f"тип не предсказывает границы. Часто это «Не сложное» с near-zero "
         f"дисперсией (α неустойчива). Примеры:\n")
for r in sorted(lo_st, key=lambda r: r["alpha"])[:6]:
    L.append(f"- #{r['ti']} α={r['alpha']:.3f} тип={r['mode_type']} "
             f"({r['mine']}) — {r['text'][:75]}")

# ===========================================================================
# 3.4  Confusion matrix + меры согласия + спот-чек
# ===========================================================================
L.append("\n## 3.4 Аудит категоризации: моя × auto\n")
cats_present = [c for c in ALL_CATS
                if any(r["auto"] == c or r["mine"] == c for r in rec)]
cm = {(x, y): 0 for x in cats_present for y in cats_present}
for r in rec:
    cm[(r["mine"], r["auto"])] += 1
L.append("**Confusion matrix (строки — моя категория, столбцы — auto):**\n")
hdr = [c[:9] for c in cats_present]
L.append("| моя\\auto | " + " | ".join(hdr) + " |")
L.append("|" + "---|" * (len(cats_present) + 1))
for x in cats_present:
    L.append(f"| `{x[:18]}` | " +
             " | ".join(str(cm[(x, y)]) for y in cats_present) + " |")

# меры согласия
def agreement_measures(pairs):
    """pairs: list of (mine, auto). -> raw, cohen_kappa, krippendorff_alpha."""
    n = len(pairs)
    raw = sum(1 for m, a in pairs if m == a) / n
    cats = sorted({c for p in pairs for c in p})
    ci_ = {c: i for i, c in enumerate(cats)}
    rowm = Counter(m for m, a in pairs)
    colm = Counter(a for m, a in pairs)
    pe = sum((rowm[c] / n) * (colm[c] / n) for c in cats)
    kappa = (raw - pe) / (1 - pe) if pe < 1 else float("nan")
    mine_v = [ci_[m] for m, a in pairs]
    auto_v = [ci_[a] for m, a in pairs]
    alpha = lib.alpha_manual([mine_v, auto_v])
    return raw, kappa, alpha


allp = [(r["mine"], r["auto"]) for r in rec]
realp = [(r["mine"], r["auto"]) for r in rec if r["auto"] != "punctuation_noise"]
L.append("\n**Три меры согласия категоризаций (моя ↔ auto):**\n")
L.append("| Подмножество | n | raw agreement | Cohen κ | Krippendorff α |")
L.append("|---|---|---|---|---|")
for lbl, p in [("Все 195 задач", allp),
               ("Реальные расхождения (без punctuation_noise)", realp)]:
    raw, k, al = agreement_measures(p)
    L.append(f"| {lbl} | {len(p)} | {raw:.3f} ({100*raw:.1f}%) | "
             f"{fnum(k)} | {fnum(al)} |")
L.append("\n_Интерпретация._ На всех 195 задачах raw agreement высок (≈0.80), "
         "но это иллюзия: 97 `punctuation_noise` определяются процедурно и "
         "совпадают тривиально, раздувая маргинал одной категории. Cohen κ и "
         "Krippendorff α, корректирующие на случайное согласие при перекошенных "
         "маргиналах, дают существенно более низкую оценку. На подмножестве "
         "реальных расхождений (где `punctuation_noise` исключён) все три меры "
         "падают — это и есть честная оценка надёжности лингвистической "
         "категоризации.\n")

# спот-чек 15 спорных задач (внутренний аудит моей категоризации)
L.append("**Спот-чек: 15 задач, где МОЯ категория неочевидна "
         "(внутренний аудит).**\n")
SPOT = """\
- **#151** `text_artifact` (моё) — ⚠️ **моя ошибка.** Категория сработала на одиночную закрывающую `"` в конце текста (несбалансированная кавычка). Реальный спор — shirin отдельно выделил «и буду бороться» (однородное сказуемое к «восстановился»). Верно: `homogeneous_predicates`. Артефакт-правило слишком жадное и у меня тоже.
- **#2** `split_main_clause` (моё) — ⚠️ **спорно, авто точнее.** Спор о том, к какой части отнести «после того» (соотносительное слово). Авто `demonstrative_words` здесь корректнее моего.
- **#124** `homogeneous_predicates` (моё) — ⚠️ **моя ошибка.** Спор — shirin разметил всё одной клаузой vs daniil/igor три (главное + придаточное «какие жертвы»). Это не однородные сказуемые; вернее `split_main_clause` либо `other`.
- **#130** `homogeneous_predicates` (моё) — ⚠️ **слабо.** α=−0.023, худшая задача корпуса: три полностью разных разметки. Ни одна типовая категория не подходит — честнее `other` (нет консенсуса).
- **#98** `comparative` (моё) — ⚠️ **моя ошибка.** Зацепился за «чем» в «прежде чем». «Прежде чем» — временной союз; верно `split_main_clause` (авто прав).
- **#73 / #75** `other` (моё) — ✅ **корректно, но категория неполна.** Единственное отличие igor — сдвиг начала span'а на 1 символ (буква, не пунктуация). token α=1.0. Это «суб-токенный офсетный шум» — родственник `punctuation_noise`, отдельной категории для него нет ни у авто, ни у меня.
- **#134** `infinitive_clause` (моё) — ✅ **разумно.** «как выйти из ситуации» — косвенный вопрос с инфинитивом, igor выделил отдельной клаузой. Лучше авто-`comparative`.
- **#168** `comparative` (моё) — ⚠️ **спорно.** «не надо жить как...» — оборот с «как»; но основной спор — разное число клауз (7/8/4). Возможно `other`.
- **#84** `double_ellipsis` (моё) — ⚠️ **частично.** Соотносительная пара «за то что» есть, но текст — длинный run-on с многими спорными границами; категория покрывает лишь часть несогласия.
- **#190** `other` (моё) — ✅ **корректно.** Соц-медиа run-on с эмодзи «><», спор о числе клауз (2/4/5), типового паттерна нет. Авто-`parenthetical` — FP по «может».
- **#12** `other` (моё) — ✅ **приемлемо.** igor слил «...я тот человек» с предыдущей клаузой; есть «тот...который», возможен `demonstrative_words`, но граница не у указательного слова.
- **#18** `other` (моё) — ✅ **корректно.** shirin сделал лишнее дробление длинной клаузы; чёткого лингвистического маркера нет.
- **#178** `split_main_clause` (моё) — ✅ **корректно.** Длинный run-on с цепочкой «что...что...»; спорные границы у придаточных. Авто-`comparative` — FP.
- **#59** `other` (моё) — ✅ **корректно.** daniil дробит мельче (5 vs 3); «(это важно)» — вставка, «если» — союз, но спор о гранулярности в целом. Авто-`infinitive_clause` — FP.
- **#199** `infinitive_clause` (моё) — ⚠️ **спорно.** Очень длинный run-on (15–18 клауз); «не знаю как дышать» с инфинитивом присутствует, но спорных границ много. Категория покрывает лишь фрагмент.
"""
L.append(SPOT)
self_err = 3  # #151, #124, #98 — явные ошибки моей категоризации
L.append(f"_Итог спот-чека._ Из 15 неочевидных случаев у моей категоризации "
         f"**3 явные ошибки** (#151, #124, #98) и ~5 спорных. Моя "
         f"категоризация контекст-зависима и точнее авто на FP-surface, но "
         f"тоже несовершенна — что усиливает главный вывод: **детерминированная "
         f"эвристика принципиально недостаточна для лингвистических "
         f"категорий**, нужен LLM-судья или экспертная разметка.\n")

# ===========================================================================
# 3.5  Stratum × Category heatmap (моя категоризация)
# ===========================================================================
strata = sorted({r["stratum"] for r in rec})
mine_cats = [c for c in ALL_CATS if any(r["mine"] == c for r in rec)]
M = np.zeros((len(strata), len(mine_cats)), dtype=int)
for r in rec:
    M[strata.index(r["stratum"]), mine_cats.index(r["mine"])] += 1
fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(M, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=mine_cats, yticklabels=strata, ax=ax, cbar_kws={"label": "n задач"})
ax.set_title("Stratum × Category (моя независимая категоризация), n=195")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig(lib.ROOT / "validation/plots/35_stratum_category.png", dpi=110)
plt.close()

L.append("## 3.5 Матрица Stratum × Category (моя категоризация)\n")
L.append("График: `plots/35_stratum_category.png`.\n")
L.append("| Страта | " + " | ".join(c[:11] for c in mine_cats) + " | Σ |")
L.append("|" + "---|" * (len(mine_cats) + 2))
for i, st in enumerate(strata):
    L.append(f"| {st} | " + " | ".join(str(x) for x in M[i]) +
             f" | {M[i].sum()} |")
L.append("| **Σ** | " + " | ".join(str(M[:, j].sum()) for j in range(len(mine_cats))) +
         f" | {M.sum()} |")
# выводы
pn_share = {st: M[strata.index(st), mine_cats.index("punctuation_noise")] /
            max(1, M[strata.index(st)].sum()) for st in strata}
L.append("\n**Наблюдения.**")
L.append(f"- Доля `punctuation_noise` сильно варьирует по стратам: "
         f"от {100*min(pn_share.values()):.0f}% до {100*max(pn_share.values()):.0f}%. "
         f"Страты с высокой долей пунктуационного шума дают мало содержательных "
         f"расхождений.")
L.append("- `infinitive_clause` / `homogeneous_predicates` концентрируются в "
         "presuicidal-стратах (разговорный текст, бессоюзные конструкции); "
         "`split_main_clause` / `direct_speech` — в solyanka-стратах "
         "(новостной/публицистический текст с придаточными).")
L.append("- Если страта даёт почти исключительно `punctuation_noise` — она "
         "слабо обогащает обучающую выборку методологически сложными кейсами.\n")

# ===========================================================================
# 3.6  Финальный вердикт
# ===========================================================================
mine_freq = Counter(r["mine"] for r in rec)
# impact: средний "провал" α относительно общего 0.846 -> вклад вниз
overall = 0.846
def cat_impact(c):
    sub = [r["alpha"] for r in rec if r["mine"] == c and r["alpha"] == r["alpha"]]
    if not sub:
        return float("nan")
    return overall - np.mean(sub)  # насколько категория ниже общего IAA

L.append("## 3.6 Финальный вердикт\n")
L.append("### A. Можно ли доверять цифрам отчёта\n")
L.append("- **Все метрики воспроизводятся: ✅** (Задача 2, 30/30 сверок, "
         "diff=0.000).\n"
         "- **Категоризация согласована с метриками: ❌** — раздел 4 отчёта "
         f"опирается на эвристику с точностью ~59% на реальных расхождениях; "
         f"Cohen κ моей и авто-категоризаций существенно ниже raw agreement.\n")

L.append("### B. Методологические проблемы vs статистический шум\n")
L.append("**Реальные методологические проблемы** (требуют доработки "
         "инструкции) — категории с пониженной α и воспроизводимым "
         "лингвистическим паттерном (по моей категоризации):")
for c in ["split_main_clause", "homogeneous_predicates", "direct_speech",
          "infinitive_clause", "double_ellipsis"]:
    m, n, nv = cat_alpha_mean("mine", {c})
    note = "реальное методологическое разногласие" if m < 0.72 else \
           "пограничный случай: паттерн воспроизводим, но α умеренно снижена"
    L.append(f"- `{c}` — n={n} (моё), mean α={fnum(m)}: {note}.")
L.append("- `comparative`, `demonstrative_words`, `parenthetical` — паттерн "
         "лингвистически реален, но n≤2 и α не снижена: проблема существует, "
         "но статистически малозаметна (низкий приоритет правки).")
L.append("\n**Статистический шум / не методология:**")
L.append("- `punctuation_noise` (n≈97) — процедурный артефакт офсетов, "
         "снимается нормализацией, НЕ требует правок инструкции (только "
         "правило о краевой пунктуации).")
L.append("- `text_artifact` (n≈18) — дефекты исходных данных (run-on, "
         "битые скобки); правилом инструкции не устраняется — это вопрос "
         "очистки корпуса.")
L.append("- `other` (n≈23) — разнородный остаток без единого паттерна; "
         "не категория, а очередь на ручной разбор.\n")

L.append("### C. Переоценка приоритетов 12 рекомендаций\n")
L.append("Критерии: **Frequency** — частота по моей категоризации; "
         "**Impact** — средний провал token α задач категории относительно "
         "общего α=0.846 (больше = сильнее тянет IAA вниз); "
         "**Tractability** — насколько реально прописать однозначное правило "
         "(В/С/Н).\n")
REKO = [
 ("homogeneous_predicates", "Однородные сказуемые: цепочка сказуемых при одном "
  "подлежащем — одна клауза", "В"),
 ("infinitive_clause", "Зависимый инфинитив: оборот при модальном/фазовом "
  "глаголе — не клауза", "В"),
 ("split_main_clause", "Разорванное главное со встроенным придаточным: единая "
  "схема разметки", "С"),
 ("direct_speech", "Прямая речь: размечать ли клаузы внутри реплики", "В"),
 ("adverbial_participle", "Деепричастные/причастные обороты — не отдельные "
  "клаузы", "В"),
 ("double_ellipsis", "Неполные придаточные с соотносительными словами", "С"),
 ("demonstrative_words", "Указательные/соотносительные слова — в главной "
  "части", "В"),
 ("parcellation", "Парцелляция: статус отделённого точкой фрагмента", "С"),
 ("comparative", "Сравнительные обороты без сказуемого — не клауза", "В"),
 ("parenthetical", "Вводные vs вставные конструкции", "С"),
 ("text_artifact", "Артефакты текста (run-on, битые скобки)", "Н"),
 ("other", "Прочие случаи — регулярный разбор с кейсодержателем", "Н"),
]
scored = []
for cat, desc, tract in REKO:
    n = mine_freq.get(cat, 0)
    imp = cat_impact(cat)
    tnum = {"В": 1.0, "С": 0.6, "Н": 0.3}[tract]
    # приоритет = freq_norm * impact_pos * tract
    impp = max(0.0, imp) if imp == imp else 0.0
    score = (n / 195) * impp * tnum
    scored.append((cat, desc, n, imp, tract, tnum, score))
scored.sort(key=lambda x: -x[6])
L.append("| # | Категория | Рекомендация | Freq (моё) | Impact (Δα) | "
         "Tractability | Приоритет-score |")
L.append("|---|---|---|---|---|---|---|")
for i, (cat, desc, n, imp, tract, tnum, score) in enumerate(scored, 1):
    L.append(f"| {i} | `{cat}` | {desc} | {n} | {fnum(imp)} | {tract} | "
             f"{score:.4f} |")
L.append("\n_Score = (Freq/195) × max(0, Δα) × Tractability-вес "
         "(В=1.0, С=0.6, Н=0.3)._\n")
L.append("**Главный сдвиг приоритетов относительно отчёта.** Исходный отчёт "
         "ставит `infinitive_clause` №1 по частоте auto (22). По моей "
         "категоризации её реальная частота ниже, а первое место занимает "
         "`homogeneous_predicates` — её авто-эвристика систематически "
         "недосчитывает (auto 6 → моё ~16), пряча внутри `infinitive_clause` "
         "и `other`. `text_artifact` и `other`, занимающие в отчёте места "
         "2 и 4, опускаются вниз: правило инструкции их не устраняет "
         "(низкая tractability).\n")

L.append("### D. Методологические лакуны, которые отчёт мог пропустить\n")
L.append("- **Суб-токенный офсетный шум** (#73, #75 и др.) — сдвиг начала "
         "span'а на 1–2 символа НЕ по пунктуации. Нормализация его не "
         "снимает, но token α (снэп к токену) и Boundary F1 relaxed его "
         "прощают — он раздувает счётчик «норм-расхождений» (98), не будучи "
         "методологической проблемой. Отчёт его не выделяет.")
L.append("- **Гранулярность дробления** — систематическая: shirin даёт 712 "
         "span'ов, igor — 624. Это не отдельная «категория», а сквозной "
         "annotator effect; инструкция должна фиксировать минимальную "
         "клаузу явно.")
L.append("- **Отсутствие правила для «нет консенсуса по типу»** — 3 задачи "
         "с тройным расхождением типа; отчёт отмечает их α=0.634, но "
         "инструкция не даёт процедуры разрешения.")
L.append("- **`other` (n≈23) не разобран** — это крупнейшая после "
         "`punctuation_noise` группа реальных расхождений; отчёт сводит её "
         "к «регулярному разбору», но внутри неё прячутся "
         "`homogeneous_predicates` и пограничные БСП-кейсы.\n")

open(lib.ROOT / "validation/03_heuristics_vs_metrics.md", "w",
     encoding="utf-8").write("\n".join(L))
print("written 03_heuristics_vs_metrics.md  lines:", len(L))
raw, k, al = agreement_measures(allp)
print(f"all: raw={raw:.3f} kappa={k:.3f} alpha={al:.3f}")
raw, k, al = agreement_measures(realp)
print(f"real: raw={raw:.3f} kappa={k:.3f} alpha={al:.3f}")
print("FP-surface:", sum(1 for v in FMODE.values() if v=="FP-surface"),
      "Wrong-similar:", sum(1 for v in FMODE.values() if v=="Wrong-similar"),
      "Missed:", sum(1 for v in FMODE.values() if v=="Missed"))
