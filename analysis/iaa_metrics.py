"""IAA metric computations: token-level Krippendorff alpha, boundary F1,
span Jaccard, sentence-type alpha."""
from __future__ import annotations
import numpy as np
import krippendorff

from iaa_common import ANNOTATORS, PAIRS, SENTENCE_TYPES, Task


# ---------------------------------------------------------------------------
# Token-level boundary representation
# ---------------------------------------------------------------------------
def boundary_tolerance(a: int, b: int, tol: int) -> bool:
    return abs(a - b) <= tol


def snap_starts_to_tokens(starts: list[int], token_starts: list[int]) -> set[int]:
    """Snap each char-offset span start to the nearest token start (token index)."""
    out = set()
    for s in starts:
        if not token_starts:
            continue
        idx = min(range(len(token_starts)),
                  key=lambda i: abs(token_starts[i] - s))
        out.add(idx)
    return out


def task_norm_starts(task: Task, annotator: str) -> list[int]:
    """Normalized span start offsets (chars) for an annotator on a task."""
    return [sp.norm_start for sp in task.anns[annotator].spans]


def task_raw_starts(task: Task, annotator: str) -> list[int]:
    return [sp.start for sp in task.anns[annotator].spans]


# ---------------------------------------------------------------------------
# 3.1  Token-level Krippendorff alpha (binary: is-token-a-clause-start)
# ---------------------------------------------------------------------------
def token_boundary_vectors(tasks: list[Task], annotators: list[str], use_norm=True):
    """Build, for each annotator, a flat binary vector over ALL tokens of all
    tasks: 1 if the token starts a clause (== a span start), else 0.

    The very first token of every task is excluded — it is trivially a clause
    start for everyone and would inflate agreement.
    """
    vectors = {a: [] for a in annotators}
    for tk in tasks:
        tstarts = [t.start for t in tk.tokens]
        n = len(tstarts)
        if n <= 1:
            continue
        for a in annotators:
            starts = task_norm_starts(tk, a) if use_norm else task_raw_starts(tk, a)
            tok_idx = snap_starts_to_tokens(starts, tstarts)
            # exclude token 0 (trivial start)
            for i in range(1, n):
                vectors[a].append(1 if i in tok_idx else 0)
    return vectors


def krippendorff_alpha_nominal(rows: list[list]) -> float:
    """rows = list of per-rater value sequences. Returns nominal alpha."""
    arr = np.array(rows, dtype=float)
    if arr.shape[1] == 0:
        return float("nan")
    try:
        return float(krippendorff.alpha(reliability_data=arr,
                                        level_of_measurement="nominal"))
    except Exception:
        return float("nan")


def token_alpha(tasks, use_norm=True):
    """Return dict: 'triple' + pair keys -> token-level alpha."""
    vecs = token_boundary_vectors(tasks, ANNOTATORS, use_norm=use_norm)
    res = {}
    res["triple"] = krippendorff_alpha_nominal([vecs[a] for a in ANNOTATORS])
    for a, b in PAIRS:
        res[f"{a}-{b}"] = krippendorff_alpha_nominal([vecs[a], vecs[b]])
    return res


# ---------------------------------------------------------------------------
# 3.2  Boundary-level F1 (set of clause-start offsets)
# ---------------------------------------------------------------------------
def _match_sets(set_a: list[int], set_b: list[int], tol: int):
    """Greedy bipartite matching of offsets within tolerance.
    Returns (tp, fp, fn) where fp counts unmatched in a, fn unmatched in b."""
    a = sorted(set_a)
    b = sorted(set_b)
    used_b = set()
    tp = 0
    for x in a:
        best = None
        best_d = tol + 1
        for j, y in enumerate(b):
            if j in used_b:
                continue
            d = abs(x - y)
            if d <= tol and d < best_d:
                best_d = d
                best = j
        if best is not None:
            used_b.add(best)
            tp += 1
    fp = len(a) - tp
    fn = len(b) - tp
    return tp, fp, fn


def boundary_f1(tasks, tol: int, use_norm=True):
    """Macro + micro boundary F1 per pair. Clause starts EXCLUDE offset 0
    (the trivial sentence-initial boundary)."""
    out = {}
    for a, b in PAIRS:
        macro_f1s = []
        m_tp = m_fp = m_fn = 0
        for tk in tasks:
            sa = [s for s in (task_norm_starts(tk, a) if use_norm else task_raw_starts(tk, a)) if s > 0]
            sb = [s for s in (task_norm_starts(tk, b) if use_norm else task_raw_starts(tk, b)) if s > 0]
            tp, fp, fn = _match_sets(sa, sb, tol)
            m_tp += tp; m_fp += fp; m_fn += fn
            if tp + fp + fn == 0:
                macro_f1s.append(1.0)  # both agree: no internal boundaries
            else:
                p = tp / (tp + fp) if (tp + fp) else 0.0
                r = tp / (tp + fn) if (tp + fn) else 0.0
                macro_f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
        micro_p = m_tp / (m_tp + m_fp) if (m_tp + m_fp) else 0.0
        micro_r = m_tp / (m_tp + m_fn) if (m_tp + m_fn) else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
        out[f"{a}-{b}"] = {
            "macro_f1": float(np.mean(macro_f1s)),
            "micro_f1": float(micro_f1),
            "micro_p": float(micro_p),
            "micro_r": float(micro_r),
        }
    return out


# ---------------------------------------------------------------------------
# 3.3  Span-level Jaccard
# ---------------------------------------------------------------------------
def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def span_jaccard(tasks, relaxed=False, iou_thr=0.5, use_norm=True):
    """Macro Jaccard of span sets per pair.
    exact: spans equal iff (start,end) identical.
    relaxed: spans match iff IoU > iou_thr (greedy)."""
    out = {}
    for a, b in PAIRS:
        jacs = []
        for tk in tasks:
            if use_norm:
                A = [(s.norm_start, s.norm_end) for s in tk.anns[a].spans]
                B = [(s.norm_start, s.norm_end) for s in tk.anns[b].spans]
            else:
                A = [(s.start, s.end) for s in tk.anns[a].spans]
                B = [(s.start, s.end) for s in tk.anns[b].spans]
            if not A and not B:
                jacs.append(1.0)
                continue
            if not relaxed:
                sa, sb = set(A), set(B)
                inter = len(sa & sb)
                union = len(sa | sb)
                jacs.append(inter / union if union else 1.0)
            else:
                used = set()
                match = 0
                for x in A:
                    best = None; best_iou = iou_thr
                    for j, y in enumerate(B):
                        if j in used:
                            continue
                        v = _iou(x, y)
                        if v > best_iou:
                            best_iou = v; best = j
                    if best is not None:
                        used.add(best); match += 1
                union = len(A) + len(B) - match
                jacs.append(match / union if union else 1.0)
        out[f"{a}-{b}"] = float(np.mean(jacs))
    return out


# ---------------------------------------------------------------------------
# 3.4  Sentence-type categorical alpha
# ---------------------------------------------------------------------------
def sentence_type_alpha(tasks):
    """Nominal Krippendorff alpha over the single-select sentence type.
    Missing choice -> np.nan (krippendorff treats nan as missing)."""
    type_idx = {t: i for i, t in enumerate(SENTENCE_TYPES)}

    def vec(annset):
        rows = []
        for a in annset:
            row = []
            for tk in tasks:
                st = tk.anns[a].sentence_type
                row.append(type_idx[st] if st in type_idx else np.nan)
            rows.append(row)
        return rows

    res = {}
    res["triple"] = krippendorff_alpha_nominal(vec(ANNOTATORS))
    for a, b in PAIRS:
        res[f"{a}-{b}"] = krippendorff_alpha_nominal(vec([a, b]))
    return res
