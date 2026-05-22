"""Независимая категоризация расхождений.

Принцип, отличающий её от analysis/iaa_categorize.py: решение принимается
по СОДЕРЖИМОМУ СПОРНОЙ ОБЛАСТИ (там, где границы клауз реально расходятся),
а не по наличию признака где-либо в тексте. Это сделано намеренно — чтобы
независимо проверить, не «прилипают» ли категории к нерелевантным признакам.
"""
from __future__ import annotations
import re
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, Doc

import lib

_seg = Segmenter()
_emb = NewsEmbedding()
_morph = NewsMorphTagger(_emb)
_cache = {}

DEMONSTRATIVE = {"тот", "та", "то", "те", "того", "той", "тех", "тому", "там",
                 "туда", "тогда", "оттуда", "потому", "затем", "так", "такой",
                 "столько", "настолько"}
CORREL = {"туда", "там", "то", "тот", "так", "тогда", "оттуда", "столько"}
SUBORD = {"что", "чтобы", "чтоб", "ибо", "как", "когда", "пока", "едва", "если",
          "хотя", "будто", "словно", "точно", "который", "которая", "которое",
          "которые", "которых", "которым", "которой", "которого", "которому",
          "которыми", "чей", "где", "куда", "откуда", "зачем", "почему",
          "сколько", "кто", "чем", "чём"}
COMPARATIVE = {"как", "словно", "будто", "точно", "подобно", "нежели", "чем"}
PARENTH = {"конечно", "наверное", "возможно", "кажется", "видимо", "по-моему",
           "во-первых", "во-вторых", "наконец", "впрочем", "однако", "значит",
           "например", "итак", "следовательно", "правда", "может", "пожалуй",
           "вообще", "кстати", "короче", "помоему", "вероятно", "видать"}


def _doc(text):
    d = _cache.get(text)
    if d is None:
        d = Doc(text)
        d.segment(_seg)
        d.tag_morph(_morph)
        _cache[text] = d
    return d


def _is_finite(tok):
    if tok.pos != "VERB":
        return False
    f = tok.feats or {}
    if f.get("VerbForm") in {"Inf", "Part", "Conv"}:
        return False
    return f.get("Tense") in {"Past", "Pres", "Fut"} or f.get("Mood") == "Imp"


def _fragment_predicate(text, lo, hi):
    """Тип главного предиката фрагмента text[lo:hi]:
    'finite' / 'inf' / 'part' / 'conv' / 'none'."""
    d = _doc(text)
    has_fin = has_inf = has_part = has_conv = False
    for t in d.tokens:
        if t.start < lo or t.start >= hi:
            continue
        f = t.feats or {}
        if _is_finite(t):
            has_fin = True
        vf = f.get("VerbForm")
        if vf == "Inf":
            has_inf = True
        elif vf == "Part":
            has_part = True
        elif vf == "Conv":
            has_conv = True
    if has_fin:
        return "finite"
    if has_conv:
        return "conv"
    if has_part:
        return "part"
    if has_inf:
        return "inf"
    return "none"


def _word_at(text, off):
    """Первое слово, начинающееся в позиции >= off."""
    m = re.search(r"[А-Яа-яЁё][А-Яа-яЁё-]*", text[off:off + 40])
    return m.group(0).lower() if m else ""


def _in_quotes(text, off):
    before = text[:off]
    return (before.count("«") > before.count("»")) or (before.count('"') % 2 == 1)


def contested_offsets(tk):
    """char-смещения начал клауз, по которым аннотаторы расходятся (norm, >0).

    Группировка с допуском ±3 символа; смещение считается спорным, если
    его поддержали не все 3 аннотатора."""
    per = {a: sorted(s for s in lib.task_starts(tk, a, True) if s > 0)
           for a in lib.ANNOTATORS}
    alloff = sorted({o for a in lib.ANNOTATORS for o in per[a]})
    # кластеризация близких смещений
    clusters = []
    for o in alloff:
        if clusters and o - clusters[-1][-1] <= 3:
            clusters[-1].append(o)
        else:
            clusters.append([o])
    contested = []
    for cl in clusters:
        center = cl[len(cl) // 2]
        support = sum(1 for a in lib.ANNOTATORS
                      if any(abs(o - center) <= 3 for o in per[a]))
        if support < 3:
            contested.append(center)
    return contested


def categorize(tk):
    """-> (category, reason)."""
    text = tk["text"]
    A = lib.ANNOTATORS
    raw = {a: frozenset((s["start"], s["end"]) for s in tk["anns"][a]["spans"]) for a in A}
    nrm = {a: frozenset((s["norm_start"], s["norm_end"]) for s in tk["anns"][a]["spans"]) for a in A}
    raw_dis = len(set(raw.values())) > 1
    norm_dis = len(set(nrm.values())) > 1
    counts = {a: len(nrm[a]) for a in A}
    diff_count = len(set(counts.values())) > 1

    if raw_dis and not norm_dis:
        return ("punctuation_noise", "raw≠, norm= : чисто пунктуационный сдвиг краёв span'а")

    # текстовые артефакты
    if text.count("(") != text.count(")") or text.count("«") != text.count("»") \
            or text.count('"') % 2 == 1:
        return ("text_artifact", "несбалансированные скобки/кавычки в тексте")
    if len(re.findall(r"[,.;:]", text)) <= 1 and len(text) > 60:
        return ("text_artifact", "run-on: длинный текст почти без пунктуации")

    cont = contested_offsets(tk)
    starts_all = sorted({0} | {s for a in A for s in lib.task_starts(tk, a, True) if s > 0}
                        | {len(text)})

    # содержимое спорных фрагментов
    frag_info = []
    for o in cont:
        nxt = min((s for s in starts_all if s > o), default=len(text))
        w = _word_at(text, o)
        pred = _fragment_predicate(text, o, nxt)
        frag_info.append({"off": o, "word": w, "pred": pred,
                          "in_q": _in_quotes(text, o)})

    words_lc = [m.lower() for m in re.findall(r"[А-Яа-яЁё-]+", text)]

    # прямая речь
    if ("«" in text or '"' in text or text.count("—") >= 1):
        if any(fi["in_q"] for fi in frag_info) or \
           (text.count("—") >= 1 and re.search(r"—\s*[а-яё]", text)):
            if diff_count or any(fi["in_q"] for fi in frag_info):
                return ("direct_speech", "спорная граница внутри прямой речи / реплики")

    # парцелляция
    if re.search(r"[а-яё]\s*[.…]\s+[А-ЯЁ]", text) or "…" in text or "..." in text:
        if diff_count:
            return ("parcellation", "точка/многоточие в середине высказывания")

    # двойной эллипсис (соотносительные пары)
    if re.search(r"\b(туда|там|то|тот|так|тогда|столько)\s*,?\s*(куда|где|что|как|когда|кто|сколько)\b", text):
        for fi in frag_info:
            if fi["word"] in {"куда", "где", "что", "как", "когда", "кто", "сколько"}:
                return ("double_ellipsis", f"соотносительная пара, спорное придаточное «{fi['word']}…»")

    # сравнительный оборот
    for fi in frag_info:
        if fi["word"] in COMPARATIVE and fi["pred"] in {"none", "part"}:
            return ("comparative", f"спорный оборот «{fi['word']}…» без финитного глагола")

    # деепричастный/причастный оборот
    for fi in frag_info:
        if fi["pred"] in {"conv", "part"}:
            return ("adverbial_participle",
                    f"спорный фрагмент «{fi['word']}…» — оборот без финитного глагола ({fi['pred']})")

    # вводное/вставное
    for fi in frag_info:
        if fi["word"] in PARENTH:
            return ("parenthetical", f"спорный фрагмент начинается с вводного «{fi['word']}»")

    # инфинитивная клауза
    for fi in frag_info:
        if fi["pred"] == "inf":
            return ("infinitive_clause",
                    f"спорный фрагмент «{fi['word']}…» с инфинитивом без своего финитного глагола")

    # разорванное главное со встроенным придаточным
    for fi in frag_info:
        if fi["word"] in SUBORD:
            return ("split_main_clause",
                    f"спорная граница у придаточного «{fi['word']}…» (встроенное придаточное)")

    # указательные слова при равном числе клауз
    if not diff_count and norm_dis:
        for fi in frag_info:
            if fi["word"] in DEMONSTRATIVE:
                return ("demonstrative_words",
                        f"сдвиг границы у указательного слова «{fi['word']}» при равном числе клауз")
        return ("demonstrative_words" if any(w in DEMONSTRATIVE for w in words_lc)
                else "other",
                "равное число клауз, разные границы")

    # однородные сказуемые: все спорные фрагменты — финитные, при diff_count
    if diff_count and frag_info and all(fi["pred"] == "finite" for fi in frag_info):
        return ("homogeneous_predicates",
                "спорные границы между финитными предикатами — однородные сказуемые vs БСП")

    return ("other", f"разное число клауз {counts}, без явного паттерна"
            if diff_count else "несовпадение границ без явного паттерна")
