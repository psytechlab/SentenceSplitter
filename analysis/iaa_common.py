"""Shared loading / normalization / tokenization utilities for IAA analysis.

Label Studio export structure (verified 2026-05-19):
  - 3 files: annotations/{daniil,shirin,igor}.json, each a list of 200 tasks.
  - Tasks aligned across files by EXACT data.text match (LS task `id` differs per import).
  - Each task: data.text, data.stratum, data.data_id (only 141/200 have data_id; stratum is on all 200).
  - annotations[0].result holds:
      * type="labels", from_name="label": span {start,end,text,labels=["простое предложение"]}
      * type="choices", from_name="сложное предложение": {choices:[ONE of
          'Не сложное','СПП','ССП','БСП','Смешанного типа']}
  - choices is single-select. 1 task has no choice for daniil, 1 for igor -> treated as missing.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import razdel

SEED = 42
ANNOTATORS = ["daniil", "shirin", "igor"]
PAIRS = [("daniil", "shirin"), ("daniil", "igor"), ("shirin", "igor")]
ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = ROOT / "annotations"
OUT_DIR = ROOT / "analysis" / "iaa_report"
PLOTS_DIR = OUT_DIR / "plots"
PER_ROW_DIR = OUT_DIR / "per_row"

SENTENCE_TYPES = ["Не сложное", "СПП", "ССП", "БСП", "Смешанного типа"]
STRATA = [
    "presuicidal_multilabel_conj", "presuicidal_multilabel_bsp",
    "presuicidal_monolabel_long", "presuicidal_short_simple",
    "solyanka_lenta_long", "solyanka_lj_long", "solyanka_social_media_multi",
    "solyanka_twitter_short_simple", "other",
]

# Punctuation stripped at span edges during normalization.
_EDGE_PUNCT = " \t\n\r.,;:!?…\"'«»()[]{}—–-"


@dataclass
class Span:
    start: int
    end: int
    text: str
    # filled by normalize()
    norm_start: int = -1
    norm_end: int = -1
    norm_text: str = ""


@dataclass
class TaskAnn:
    """One annotator's annotation of one task."""
    text: str
    spans: list[Span] = field(default_factory=list)
    sentence_type: str | None = None  # None == missing


@dataclass
class Task:
    text: str
    stratum: str
    data_id: str | None
    anns: dict[str, TaskAnn] = field(default_factory=dict)  # annotator -> TaskAnn
    tokens: list = field(default_factory=list)  # razdel substrings (start,stop,text)


def load_annotator(name: str) -> dict[str, TaskAnn]:
    """Return text -> TaskAnn for one annotator file."""
    with open(ANN_DIR / f"{name}.json", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, TaskAnn] = {}
    for t in raw:
        text = t["data"]["text"]
        ta = TaskAnn(text=text)
        anns = t.get("annotations", [])
        if anns:
            for r in anns[0].get("result", []):
                if r["type"] == "labels":
                    v = r["value"]
                    ta.spans.append(Span(v["start"], v["end"], v.get("text", "")))
                elif r["type"] == "choices":
                    ch = r["value"].get("choices", [])
                    if ch:
                        ta.sentence_type = ch[0]
        ta.spans.sort(key=lambda s: (s.start, s.end))
        out[text] = ta
    return out


def load_tasks() -> list[Task]:
    """Load all 3 annotators, align by text, return list of Task."""
    per = {n: load_annotator(n) for n in ANNOTATORS}
    # metadata (stratum/data_id) — read from any file; verified consistent.
    with open(ANN_DIR / "daniil.json", encoding="utf-8") as f:
        raw = json.load(f)
    meta = {t["data"]["text"]: (t["data"].get("stratum", "other"),
                                t["data"].get("data_id")) for t in raw}
    common = set(per["daniil"]) & set(per["shirin"]) & set(per["igor"])
    tasks: list[Task] = []
    for text in sorted(common):
        stratum, data_id = meta.get(text, ("other", None))
        tk = Task(text=text, stratum=stratum or "other", data_id=data_id)
        tk.tokens = list(razdel.tokenize(text))
        for n in ANNOTATORS:
            ta = per[n][text]
            for sp in ta.spans:
                normalize_span(sp, text)
            tk.anns[n] = ta
        tasks.append(tk)
    return tasks


def normalize_span(sp: Span, text: str) -> None:
    """Trim whitespace + strip edge punctuation, snap offsets to span content.

    Sets sp.norm_start / norm_end / norm_text. If the span becomes empty
    after stripping it collapses to a zero-length span at original start.
    """
    s, e = sp.start, sp.end
    s = max(0, min(s, len(text)))
    e = max(0, min(e, len(text)))
    # advance start past edge chars, retreat end past edge chars
    while s < e and text[s] in _EDGE_PUNCT:
        s += 1
    while e > s and text[e - 1] in _EDGE_PUNCT:
        e -= 1
    sp.norm_start = s
    sp.norm_end = e
    sp.norm_text = text[s:e]


def token_starts(text: str, tokens) -> list[int]:
    return [t.start for t in tokens]


def text_to_token_index(text: str, tokens):
    """Map a char offset -> index of the token whose span contains it
    (or the nearest following token start)."""
    starts = [t.start for t in tokens]
    return starts
