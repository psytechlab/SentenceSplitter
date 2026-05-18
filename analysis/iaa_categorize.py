"""Heuristic categorization of annotator disagreements.

Categories (per task spec):
  a punctuation_noise        — disagreement vanishes after normalization
  b homogeneous_predicates   — homogeneous predicates split vs kept as one clause
  c double_ellipsis          — "поступила туда, куда хотела" type
  d direct_speech            — split inside quotes
  e parcellation             — sentence-medial period / ellipsis
  f split_main_clause        — main clause with embedded subordinate
  g demonstrative_words      — тот/туда/тогда placement
  h infinitive_clause        — disputed clause with infinitive / incomplete predicate
  i parenthetical            — вводное vs вставное
  j adverbial_participle     — деепричастный/причастный оборот wrongly marked
  k comparative              — сравнительный оборот wrongly marked
  l text_artifact            — unbalanced brackets, typos, run-on w/o punctuation
  m other
"""
from __future__ import annotations
import re

from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, Doc

_segmenter = Segmenter()
_emb = NewsEmbedding()
_morph = NewsMorphTagger(_emb)

SUBORD_CONJ = {
    "что", "чтобы", "чтоб", "потому", "оттого", "ибо", "так", "как",
    "когда", "пока", "едва", "если", "хотя", "несмотря", "будто", "словно",
    "точно", "который", "которая", "которое", "которые", "которых", "которым",
    "чей", "где", "куда", "откуда", "зачем", "почему", "сколько", "кто", "чем",
}
COORD_CONJ = {"и", "а", "но", "да", "или", "либо", "однако", "зато", "тоже", "также"}
DEMONSTRATIVE = {
    "тот", "та", "то", "те", "того", "той", "тех", "тому", "там", "туда",
    "тогда", "оттуда", "потому", "затем", "так", "такой", "столько",
}
PARENTHETICAL = {
    "конечно", "наверное", "возможно", "кажется", "видимо", "по-моему",
    "во-первых", "во-вторых", "наконец", "впрочем", "однако", "значит",
    "например", "итак", "следовательно", "правда", "может", "пожалуй",
}
COMPARATIVE_MARKERS = {"как", "словно", "будто", "точно", "подобно"}


_morph_cache: dict[str, Doc] = {}


def morph_doc(text: str) -> Doc:
    cached = _morph_cache.get(text)
    if cached is not None:
        return cached
    doc = Doc(text)
    doc.segment(_segmenter)
    doc.tag_morph(_morph)
    _morph_cache[text] = doc
    return doc


def is_finite_verb(token) -> bool:
    if token.pos != "VERB":
        return False
    feats = token.feats or {}
    if feats.get("VerbForm") in {"Inf", "Part", "Conv"}:
        return False
    return (feats.get("Tense") in {"Past", "Pres", "Fut"}
            or feats.get("Mood") == "Imp")


def count_finite_verbs(text: str) -> int:
    doc = morph_doc(text)
    return sum(1 for t in doc.tokens if is_finite_verb(t))


def has_infinitive(text: str) -> bool:
    doc = morph_doc(text)
    return any((t.feats or {}).get("VerbForm") == "Inf" for t in doc.tokens)


def has_adverbial_participle(text: str) -> bool:
    doc = morph_doc(text)
    return any((t.feats or {}).get("VerbForm") in {"Conv", "Part"}
               for t in doc.tokens)


def _words(text: str) -> list[str]:
    return re.findall(r"[а-яёА-ЯЁ-]+", text.lower())


def categorize(text: str, span_sets: dict, norm_span_sets: dict,
               sentence_types: dict) -> tuple[str, str]:
    """span_sets / norm_span_sets: annotator -> list[(start,end)].
    Returns (category, human-readable description)."""
    annset = list(span_sets.keys())
    counts = {a: len(span_sets[a]) for a in annset}
    norm_counts = {a: len(norm_span_sets[a]) for a in annset}

    # --- (a) punctuation noise: raw differs, normalized does not -----------
    raw_norm_sets = {a: frozenset(span_sets[a]) for a in annset}
    nrm_norm_sets = {a: frozenset(norm_span_sets[a]) for a in annset}
    raw_disagree = len(set(raw_norm_sets.values())) > 1
    norm_disagree = len(set(nrm_norm_sets.values())) > 1
    if raw_disagree and not norm_disagree:
        return ("punctuation_noise",
                "Расхождение в границах span'ов исчезает после нормализации "
                "(обрезки пунктуации/пробелов) — пунктуационный шум, не реальное несогласие.")

    # --- text artifacts -----------------------------------------------------
    if text.count("(") != text.count(")") or text.count("«") != text.count("»"):
        return ("text_artifact",
                "Несбалансированные скобки/кавычки в исходном тексте — артефакт, "
                "затрудняющий согласованную разметку границ.")
    # run-on: long text, few delimiters but annotators split differently
    if len(re.findall(r"[,.;:]", text)) <= 1 and len(text) > 60 and norm_disagree:
        return ("text_artifact",
                "Длинное предложение почти без пунктуации (run-on) — границы клауз "
                "приходится угадывать, отсюда расхождение.")

    diff_count = len(set(norm_counts.values())) > 1
    low = _words(text)

    # --- direct speech ------------------------------------------------------
    if ('"' in text or "«" in text or "—" in text) and diff_count:
        if re.search(r"[«\"].*[,—].*[»\"]", text) or text.count("—") >= 1:
            return ("direct_speech",
                    "Внутри прямой речи / реплики один аннотатор выделил клаузы, "
                    "другой оставил реплику целиком.")

    # --- parcellation: sentence-medial period / ellipsis -------------------
    if re.search(r"[а-яё]\s*\.\s+[А-ЯЁ]", text) or "…" in text or "..." in text:
        if diff_count:
            return ("parcellation",
                    "Точка/многоточие в середине высказывания (парцелляция) — один "
                    "объединил парцеллят с основным предложением, другой разделил.")

    # --- double ellipsis: "...туда, куда...", "...то, что..." --------------
    if re.search(r"\b(туда|там|то|тот|так|тогда)\s*,\s*(куда|где|что|как|когда|кто)\b", text):
        if diff_count or norm_disagree:
            return ("double_ellipsis",
                    "Конструкция с соотносительными словами и неполной клаузой "
                    "(двойной эллипсис, тип «поступила туда, куда хотела») — спор о "
                    "том, отдельная ли это клауза.")

    # --- demonstrative words placement -------------------------------------
    if any(w in DEMONSTRATIVE for w in low) and norm_disagree and not diff_count:
        # same count, different boundaries near a demonstrative
        return ("demonstrative_words",
                "Указательное/соотносительное слово (тот/туда/тогда/так) — "
                "аннотаторы по-разному отнесли его к главной или придаточной части "
                "(несовпадение границ при равном числе клауз).")

    # --- comparative turn ---------------------------------------------------
    if re.search(r",\s*(как|словно|будто|точно|подобно)\s+[а-яё]+\b", text):
        # comparative without a finite verb after the marker -> not a clause
        if diff_count:
            return ("comparative",
                    "Сравнительный оборот («как ветер») — один аннотатор ошибочно "
                    "выделил его как отдельную клаузу.")

    # --- adverbial / verbal participle turn --------------------------------
    if has_adverbial_participle(text) and diff_count:
        return ("adverbial_participle",
                "Деепричастный/причастный оборот — один аннотатор ошибочно выделил "
                "его как самостоятельную клаузу.")

    # --- parenthetical ------------------------------------------------------
    if any(w in PARENTHETICAL for w in low) and diff_count:
        return ("parenthetical",
                "Вводное/вставное слово или конструкция — расхождение в том, считать "
                "ли её отдельной вставной клаузой.")

    # --- infinitive / incomplete predicate ---------------------------------
    if has_infinitive(text) and diff_count:
        return ("infinitive_clause",
                "Спорная клауза с инфинитивом / неполным предикатом "
                "(тип «не знаю как дышать», «хочет рассказать»).")

    # --- homogeneous predicates --------------------------------------------
    # heuristic: comma-separated finite verbs, count differs
    if diff_count and re.search(r"[а-яё]+(л|ла|ло|ли|ет|ит|ют|ут|ал|ил)\s*,\s*[а-яё]+", text):
        nfv = count_finite_verbs(text)
        if nfv >= 2:
            return ("homogeneous_predicates",
                    "Однородные сказуемые через запятую — один аннотатор разделил их "
                    "на несколько клауз, другой счёл их одной клаузой с однородными "
                    "членами.")

    # --- split main clause: embedded subordinate ---------------------------
    if any(w in SUBORD_CONJ for w in low) and norm_disagree:
        return ("split_main_clause",
                "Главное предложение со встроенным придаточным — варианты разметки: "
                "2 span'а главной части + придаточное / перекрытие / 2 span'а.")

    # --- fallback -----------------------------------------------------------
    if diff_count:
        return ("other",
                f"Разное число клауз ({norm_counts}) без явного типового паттерна.")
    return ("other",
            "Несовпадение границ span'ов без явного типового паттерна.")
