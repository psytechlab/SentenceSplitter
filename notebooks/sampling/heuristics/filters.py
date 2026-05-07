"""Sampling heuristics for identifying complex Russian sentences.

Pure-function filters used by the sampling pipeline. Loaded once at import:
- conjunction lists from ./conjunctions.json (sibling file)
- natasha morphology pipeline (lazy fallback to pymorphy3 if unavailable)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level resources (loaded once)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_CONJ_PATH = _HERE / "conjunctions.json"

with _CONJ_PATH.open("r", encoding="utf-8") as _f:
    _CONJ_RAW = json.load(_f)

# Union of ssp.* and spp.* values. Keep ellipses (…) and split into "phrase pieces"
# for multi-word matching. We treat "то… то" as two single-token markers ("то", "то")
# joined by ellipsis — for word-boundary matching we expand on whitespace and
# strip ellipses.

def _flatten_conj(raw: dict) -> list[str]:
    out: list[str] = []
    for top in ("ssp", "spp"):
        for _, items in raw.get(top, {}).items():
            for item in items:
                # Drop ellipsis-joined alternation patterns like "то… то" — match each side.
                if "…" in item:
                    for piece in item.split("…"):
                        piece = piece.strip()
                        if piece:
                            out.append(piece)
                else:
                    out.append(item.strip())
    # Dedup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for c in out:
        cl = c.lower()
        if cl and cl not in seen:
            seen.add(cl)
            uniq.append(cl)
    return uniq


_CONJUNCTIONS: list[str] = _flatten_conj(_CONJ_RAW)

# Pre-compile regex: word-boundary, case-insensitive, multi-word phrases as
# whitespace-tolerant. Sort by length desc so multi-word phrases match first.
_CONJ_SORTED = sorted(_CONJUNCTIONS, key=lambda s: -len(s))


def _build_conj_regex(conjs: list[str]) -> re.Pattern[str]:
    parts = []
    for c in conjs:
        # Tokens separated by spaces -> tolerate any whitespace run.
        toks = c.split()
        escaped = r"\s+".join(re.escape(t) for t in toks)
        parts.append(escaped)
    # \b works on Cyrillic letters in Python's re with re.UNICODE (default in Py3).
    pattern = r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)"
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


_CONJ_RE = _build_conj_regex(_CONJ_SORTED)


# ---------------------------------------------------------------------------
# Morphology backend (natasha preferred, pymorphy3 fallback)
# ---------------------------------------------------------------------------

_NATASHA = None  # tuple (Segmenter, NewsMorphTagger, Doc)
_PYMORPHY = None  # MorphAnalyzer instance


def _init_natasha() -> Optional[tuple]:
    global _NATASHA
    if _NATASHA is not None:
        return _NATASHA
    try:
        from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, Doc  # type: ignore
        seg = Segmenter()
        emb = NewsEmbedding()
        tagger = NewsMorphTagger(emb)
        _NATASHA = (seg, tagger, Doc)
        return _NATASHA
    except Exception:
        _NATASHA = None
        return None


def _init_pymorphy() -> Optional[object]:
    global _PYMORPHY
    if _PYMORPHY is not None:
        return _PYMORPHY
    try:
        import pymorphy3  # type: ignore
        _PYMORPHY = pymorphy3.MorphAnalyzer()
        return _PYMORPHY
    except Exception:
        _PYMORPHY = None
        return None


# Eagerly try natasha at import (matches "load once" requirement),
# but tolerate failure so module still imports.
_init_natasha()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_multilabel(label: str, dataset: str) -> bool:
    """True if the row's label string encodes more than one label."""
    if label is None:
        return False
    sep = ";" if dataset == "presuicidal" else ","
    if dataset not in {"presuicidal", "solyanka"}:
        raise ValueError(f"Unknown dataset {dataset!r}")
    parts = [p.strip() for p in str(label).split(sep)]
    parts = [p for p in parts if p]
    return len(parts) > 1


_FINITE_TENSES = {"Past", "Pres", "Fut"}


def count_finite_verbs(text: str) -> int:
    """Count finite-verb tokens (Mood=Ind/Imp/Cnd with Tense, excludes Inf/Part/Conv)."""
    if not text or not text.strip():
        return 0
    nt = _init_natasha()
    if nt is not None:
        seg, tagger, Doc = nt
        doc = Doc(text)
        doc.segment(seg)
        doc.tag_morph(tagger)
        n = 0
        for t in doc.tokens:
            if t.pos != "VERB":
                continue
            feats = t.feats or {}
            vform = feats.get("VerbForm")
            if vform in {"Inf", "Part", "Conv"}:
                continue
            tense = feats.get("Tense")
            if tense in _FINITE_TENSES:
                n += 1
                continue
            # Imperative has Mood=Imp, no Tense — count as finite too.
            if feats.get("Mood") == "Imp":
                n += 1
        return n

    morph = _init_pymorphy()
    if morph is None:
        # No backend available: return 0 (caller decides what to do).
        return 0
    n = 0
    for tok in re.findall(r"\w+", text, flags=re.UNICODE):
        parses = morph.parse(tok)
        if not parses:
            continue
        p = parses[0]
        tag = p.tag
        if "VERB" in tag and "INFN" not in tag and "PRTF" not in tag and "PRTS" not in tag and "GRND" not in tag:
            n += 1
    return n


def has_conjunction(text: str) -> bool:
    """True if text contains any ССП/СПП conjunction (whole-word, case-insensitive)."""
    if not text:
        return False
    return _CONJ_RE.search(text) is not None


_BSP_PUNCT = {",", ":", "—", "–", ";"}
_DASH_CHARS = {"—", "–"}


def _split_on_bsp_marker(text: str) -> Optional[tuple[str, str]]:
    """Find a BSP-style splitter and return (left, right) on the first hit.

    Excludes hyphens between alphabetic word characters (e.g., 'из-за').
    Picks the rightmost occurrence with both sides non-trivially populated.
    """
    candidates: list[int] = []
    for i, ch in enumerate(text):
        if ch in _BSP_PUNCT:
            candidates.append(i)
        elif ch == "-":
            # Plain hyphen — skip if it is a word-internal hyphen.
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if prev.isalpha() and nxt.isalpha():
                continue
            candidates.append(i)
    # Prefer markers that produce two substantive halves; pick the one
    # closest to the middle so length thresholds on both sides are sensible.
    best: Optional[tuple[str, str]] = None
    best_balance = -1
    for idx in candidates:
        left = text[:idx].strip()
        right = text[idx + 1 :].strip()
        if not left or not right:
            continue
        l_tokens = re.findall(r"\w+", left, flags=re.UNICODE)
        r_tokens = re.findall(r"\w+", right, flags=re.UNICODE)
        l_alpha = any(re.search(r"[^\W\d_]", t, flags=re.UNICODE) for t in l_tokens)
        r_alpha = any(re.search(r"[^\W\d_]", t, flags=re.UNICODE) for t in r_tokens)
        if len(l_tokens) >= 2 and len(r_tokens) >= 2 and l_alpha and r_alpha:
            balance = min(len(l_tokens), len(r_tokens))
            if balance > best_balance:
                best_balance = balance
                best = (left, right)
    return best


def has_asyndetic_markers(text: str, min_tokens: int = 8) -> bool:
    """True if text is long enough and a BSP punctuation mark splits it into two halves."""
    if not text:
        return False
    if token_length(text) < min_tokens:
        return False
    return _split_on_bsp_marker(text) is not None


def token_length(text: str) -> int:
    """Token count via razdel.tokenize, fallback to whitespace split."""
    if not text:
        return 0
    try:
        from razdel import tokenize  # type: ignore
        return sum(1 for _ in tokenize(text))
    except Exception:
        return len(text.split())


def is_complex(text: str) -> bool:
    """V1 complexity heuristic: True iff text has a conjunction OR an asyndetic BSP marker.

    Validated in `notebooks/sampling/heuristics/validation.md` against Shirin's
    120 human-annotated sentences. V1 was chosen because it has perfect recall
    (R=1.000) on confirmed-complex sentences under cleaner ground-truth handling
    (NaN annotator-skip rows excluded). Precision could not be discriminated
    on this dataset because Scenario B contains zero ground-truth negatives.

    V2 (finite-verb gated: ≥2 finite verbs AND (conj OR bsp)) was evaluated and
    REJECTED. Its previously-observed precision advantage was an artefact of
    treating skipped rows as confirmed-simple negatives; on clean data V2 simply
    has lower recall (0.745) without measurable precision benefit. The recall
    gap was robust under cleaner GT — V2 missed 24/94 positives, ~21% of which
    are zero-copula or impersonal sentences with no finite verb at all.

    See `is_complex_v2_rejected` (kept for reference only) for the rejected
    finite-verb-gated variant.
    """
    return has_conjunction(text) or has_asyndetic_markers(text)


def _is_complex_v2_rejected(text: str) -> bool:
    """Rejected V2 variant — retained for reference only. See is_complex docstring."""
    if count_finite_verbs(text) < 2:
        return False
    return has_conjunction(text) or has_asyndetic_markers(text)


__all__ = [
    "is_multilabel",
    "count_finite_verbs",
    "has_conjunction",
    "has_asyndetic_markers",
    "token_length",
    "is_complex",
]
