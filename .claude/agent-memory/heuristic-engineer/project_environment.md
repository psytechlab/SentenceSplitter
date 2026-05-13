---
name: python environment and morphology backend
description: Project venv quirks and how to install Russian NLP libs
type: project
---

Python venv at `.venv/` (Python 3.14). Use `.venv/bin/python` directly. Install packages with `uv pip install --python .venv/bin/python <pkgs>` — `pip3 install` is blocked by PEP 668.

Russian NLP backend choices:
- **Natasha** is preferred for morphology. It works in this venv WITHOUT `MorphVocab` — `MorphVocab` pulls in pymorphy2 which fails on Python 3.14 because `pkg_resources` is missing. Use `Segmenter + NewsEmbedding + NewsMorphTagger + Doc` only — token features (`t.pos`, `t.feats`) are sufficient for finite-verb detection.
- pymorphy3 works as a fallback (POS tag + form check for INFN/PRTF/PRTS/GRND).
- `razdel` for tokenization.

Finite verb detection in natasha: `t.pos == 'VERB'` AND `t.feats.get('VerbForm') not in {'Inf', 'Part', 'Conv'}` AND (`t.feats.get('Tense') in {'Past', 'Pres', 'Fut'}` OR `t.feats.get('Mood') == 'Imp'`).

**Why:** Saves 10+ minutes of debugging the pymorphy2 / pkg_resources / Python 3.14 incompatibility chain on every fresh task.
**How to apply:** When wiring up a new heuristic that needs morphology, instantiate Natasha without MorphVocab and trust `t.feats` directly.
