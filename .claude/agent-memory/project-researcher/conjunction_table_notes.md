---
name: Conjunction table — annotation guide extraction
description: What was extracted from the annotation guide Google Doc for the conjunctions.json heuristic file
type: reference
---

Source doc: https://docs.google.com/document/d/1Qs6AVKL_Fi3dovBxS_Ad_mP3VMZK71cq_4MguGIY2hY
Extracted to: notebooks/sampling/heuristics/conjunctions.json

SSP (coordinate):
- adversative: а, но, да, зато, однако, однако же, все же
- copulative: и, да, не только… но и, также, тоже, и… и, ни… ни, как… так и
- disjunctive: или, либо, то… то, то ли… то ли, не то… не то, или… или

SPP (subordinate) — flattened into three categories:
- explanatory (изъяснительные): что, как, будто, как бы, как бы не, чтобы, чтобы не, ли, не… ли, ли… ли, то ли
- attributive (определительные): который, какой, чей, куда, где, когда, откуда, что
- adverbial (обстоятельственные — flattened from temporal, locative, causal, consequential, conditional, purpose, concessive, comparative, degree, manner, connective subtypes): ~34 items

BSP (asyndetic):
- punctuation cues: , ; : —
- semantic patterns: enumeration, cause/explanation, opposition, condition, time

The Google Doc export redirected through googleusercontent.com CDN before serving plaintext — follow the redirect when re-fetching.

## Doc structure (verified 2026-05-07)

Section 2.3: Правила разделения по типам сложных предложений
  2.3.1 ССП — split before coordinating conjunction
  2.3.2 СПП — split before subordinating conjunction/союзное слово; 12 subclause types
  2.3.3 БСП — split on comma (enumeration), colon (explanation/cause), dash (condition/time/consequence)

Section 2.4: Особые конструкции и уточнения
  2.4.1 Multiple subordination in СПП — each придаточная split separately (homogeneous, sequential, parallel)
  2.4.2 Указательные слова — корреляты (тот, такой, там, тогда…) stay in main clause; boundary after them
  2.4.3 Эллипсис — restore omitted subject in square brackets; do NOT exclude sentence

Section 2.6: Исключения — что НЕ разделять (verified verbatim)
  1. Однородные сказуемые/подлежащие → EXCLUDE (1 basis)
  2. Причастные и деепричастные обороты → EXCLUDE (no own predicate)
  3. Сравнительные обороты без сказуемого → EXCLUDE
  4. Вводные слова и конструкции → EXCLUDE (no predicate base)
  5. Прямая речь внутри авторских слов (unless standalone utterance) → FLAG / defer to project rules

Phenomena NOT in doc: парцелляция, косвенная речь (no explicit rules given).
