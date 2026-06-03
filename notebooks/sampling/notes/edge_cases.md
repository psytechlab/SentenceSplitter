# Edge cases — sourced from annotation instruction

Source: https://docs.google.com/document/d/1Qs6AVKL_Fi3dovBxS_Ad_mP3VMZK71cq_4MguGIY2hY
Retrieved: 2026-05-07

---

## Phenomenon: Однородные сказуемые / однородные члены

- **Decision: EXCLUDE**
- **Source quote:** "Однородные сказуемые/подлежащие: Она встала и пошла к окну. (1 основа → не делим)."
- **Section / location in doc:** 2.6 — Исключения: что НЕ разделять (item 1)
- **Notes:** A sentence with coordinated predicates sharing one grammatical subject counts as a single predicate base and must NOT be split. Sampler should filter these out of the positive-example pool to avoid false-split training signal.

---

## Phenomenon: Причастные и деепричастные обороты

- **Decision: EXCLUDE**
- **Source quote:** "Причастные и деепричастные обороты: Уставший от дороги, он лег спать. (Нет собственного сказуемого → не делим)."
- **Section / location in doc:** 2.6 — Исключения: что НЕ разделять (item 2)
- **Notes:** Participial (причастный) and adverbial-participial (деепричастный) phrases lack their own predicate base and must NOT be extracted as separate clauses. Sentences where complexity is entirely due to these phrases should be excluded from the splitting corpus.

---

## Phenomenon: Парцелляция

- **Decision: OUT-OF-SCOPE (handled upstream)**
- **Source quote:** N/A
- **Section / location in doc:** Not mentioned anywhere in the document.
- **Case-holder ruling:** — parcellated fragments are treated as separate sentences upstream during the dataset's initial sentence segmentation. They won't normally appear as single dataset entries. Noted as a potential edge case for future reference.

---

## Phenomenon: Эллипсис

- **Decision: FLAG (restore in brackets)**
- **Source quote:** "Если в выделенной части отсутствует подлежащее, но оно легко восстанавливается из контекста и необходимо для грамматической завершённости, восстановите его в квадратных скобках."
- **Section / location in doc:** 2.4.3 — Восстановление опущенных элементов (эллипсис)
- **Notes:** Ellipsis is not grounds for exclusion. If a split clause is missing a subject that is recoverable from context, annotators must restore it in square brackets (e.g., "[мы] вышли на улицу"). Sampler should retain elliptic sentences but flag them so annotators know to apply bracket restoration.

---

## Phenomenon: Прямая и косвенная речь

- **Decision: FLAG (прямая речь) / INCLUDE (косвенная речь)**
- **Source quote (прямая речь):** "Прямая речь внутри авторских слов, если не образует самостоятельной реплики с интонацией конца (обрабатывается по отдельным правилам проекта)."
- **Section / location in doc:** 2.6 — Исключения: что НЕ разделять (item 5)
- **Notes:** Direct speech embedded in attribution text is excluded UNLESS it forms a standalone utterance with final intonation. The doc explicitly defers to separate project rules. Flag all direct-speech sentences; do not include them as routine splitting examples without separate project-level guidance.
- **Case-holder ruling (косвенная речь):** — INCLUDE. Indirect speech is treated as сложноподчинённое with изъяснительное придаточное (formally a complex sentence). Annotators split into clauses per the linguistic-theory approach already documented in the annotation instruction.

---

## Phenomenon: Вводные конструкции

- **Decision: EXCLUDE**
- **Source quote:** "Вводные слова и конструкции: К счастью, мы успели. (Не имеют основы → не делим)."
- **Section / location in doc:** 2.6 — Исключения: что НЕ разделять (item 4)
- **Notes:** Parenthetical / introductory words and phrases (вводные слова) have no predicate base and are explicitly excluded from splitting. Sentences whose only apparent "additional clause" is a вводная конструкция should be excluded from the splitting sample.

---

## Section 2.3.3 — Asyndetic (БСП) markers

- **Verbatim header:** "Бессоюзные сложные предложения (БСП). Части связаны только интонацией и смыслом. Союзы отсутствуют."
- **Punctuation markers and rules (verbatim from table):**

| Смысловое отношение | Маркер | Правило разметки |
|---|---|---|
| Перечисление | Запятая между независимыми основами | "Делим по запятой, если каждая часть имеет свою основу." |
| Пояснение / Причина | Двоеточие (:) | "Делим после двоеточия. Вторая часть раскрывает или объясняет первую." |
| Условие / Время / Следствие | Тире (—) | "Делим после тире. Первая часть задаёт условие/время, вторая содержит результат." |

- **Worked example (verbatim):** "Солнце село: стало прохладно. → 1. Солнце село: 2. Стало прохладно."
- **Inclusion guidance:** БСП sentences INCLUDE in the splitting corpus. The boundary is placed: after the comma (enumeration), after the colon (explanation/cause), after the dash (condition/time/consequence). The key inclusion criterion is that each part must have its own predicate base ("если каждая часть имеет свою основу").

---

## Summary table

| Phenomenon | Decision | Doc section |
|---|---|---|
| Однородные сказуемые / однородные члены | EXCLUDE | 2.6 item 1 |
| Причастные и деепричастные обороты | EXCLUDE | 2.6 item 2 |
| Парцелляция | OUT-OF-SCOPE — handled upstream (Buyanov, 2026-05-08) | Not in doc |
| Эллипсис | FLAG — restore subject in [ ] | 2.4.3 |
| Прямая речь | FLAG — defer to project rules | 2.6 item 5 |
| Косвенная речь | INCLUDE — сложноподчинённое with изъяснительное придаточное (Buyanov, 2026-05-08) | Not in doc |
| Вводные конструкции | EXCLUDE | 2.6 item 4 |
| БСП (asyndetic) | INCLUDE — split on comma/colon/dash | 2.3.3 |

---

## Open questions

- **Прямая речь — project rules referenced but not included:** Section 2.6 item 5 defers to "отдельные правила проекта" without specifying them. These rules need to be located and appended here.
- **Comparative phrases without a predicate:** Section 2.6 item 3 lists "Сравнительные обороты без сказуемого: Она бежала быстро, как ветер. (Нет основы → не делим)" — not in the original 7 phenomena but relevant for sampler filtering.
- **Прямая речь forming a standalone utterance:** The doc creates an exception ("если не образует самостоятельной реплики с интонацией конца") but gives no example or test for what counts as a standalone utterance with final intonation.
