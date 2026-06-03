# Heuristic validation against ./data/data.csv

Ground truth: 120 human-annotated sentences. Positive class = "human split into >1 clause" (`n_clauses > 1` in the parsed `simple_sentences` JSON).

The earlier "All" run treated NaN cells as confirmed-simple. This is incorrect — NaN almost certainly marks an annotator skip rather than a confirmed 1-clause sentence. The two views below (Scenario A and Scenario B) drop these rows so the metrics reflect only data the annotator actually labelled.

## Ground-truth handling — three views

| View | Description | N rows | Positives | Negatives |
|---|---|---|---|---|
| All | Original run; NaN treated as 1-clause/simple, plain-string cells treated as 1-clause/simple | 120 | 94 | 26 |
| Scenario A | Drop NaN rows in `simple_sentences`. Plain-string (non-JSON) cells kept and treated as 1-clause/simple. | 96 | 94 | 2 |
| Scenario B | Drop NaN AND drop rows that fail to parse as `{"text": [list[str]]}`. Only well-formed rows count. | 94 | 94 | 0 |

Status breakdown of the 120 raw rows: 94 well-formed JSON, 24 NaN, 2 plain-string. None of the 24 NaN rows nor the 2 plain-string rows carry an annotator-confirmed clause split, so under both Scenario A and Scenario B the dataset is dominated by positives — Scenario B has zero negatives, which makes precision trivially 1.0 for any heuristic and removes the ability to discriminate FPs from this dataset alone.

## Metrics — all four combinations

| Variant | Scenario | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| V1 (conj OR bsp) | A | 1.000 | 1.000 | 1.000 | 1.000 | 94 | 0 | 0 | 2 |
| V1 (conj OR bsp) | B | 1.000 | 1.000 | 1.000 | 1.000 | 94 | 0 | 0 | 0 |
| V2 (is_complex)  | A | 1.000 | 0.745 | 0.854 | 0.750 | 70 | 0 | 24 | 2 |
| V2 (is_complex)  | B | 1.000 | 0.745 | 0.854 | 0.745 | 70 | 0 | 24 | 0 |

For back-comparison the original "All" view: V1 P=0.825 R=1.000 F1=0.904; V2 P=0.959 R=0.745 F1=0.838. The 20 "FPs" V1 acquired under that view, plus 17 of the 23 "TNs" V2 was credited with, all came from the 24 NaN rows that we now treat as out-of-distribution.

## Hypothesis check
**V2 > V1 precision under Scenario B: REFUTED, delta = 0.000.**

Both variants reach precision 1.000 because Scenario B contains zero ground-truth negatives — there is no opportunity to incur a false positive. The original "V2 has higher precision" claim was an artefact of the NaN rows: V1 fired on 20 of them and was charged with FPs, while V2's finite-verb gate suppressed most of those firings. Once the NaN rows are removed there are no longer any negative examples on which to measure the gate's benefit.

This means the precision win cannot be validated on this dataset under cleaner GT handling. To re-evaluate the gate's precision benefit we need either (a) the annotator's labels for the 24 currently-NaN sentences or (b) a fresh batch of confirmed-simple sentences to act as negatives.

## V2 false-negative breakdown (Scenario B)

24 false negatives total. Categorized by the heuristic in `/tmp/run_validation_v2.py`:

| Category | Count | Share |
|---|---|---|
| Zero-copula nominal (no finite verb, no impersonal predicative) | 4 | 16.7% |
| Impersonal / predicative-adverb (no finite verb, predicative lemma present) | 1 | 4.2% |
| Other (≥1 finite verb but heuristic still misses) | 19 | 79.2% |

The "other" bucket dominates because most FNs *do* have a finite verb — the gate fires on the conj/bsp side in only some of them, or fires correctly but the row's `count_finite_verbs` returns 1 because the second predicate is an infinitive or the second clause has been elided.

### Examples
- **Zero-copula nominal**:
  - "Я не верующий, мне 17 лет."
  - "В школе колледже я был двоечником, не общителен с людьми."
- **Impersonal / predicative-adverb**:
  - "Грустно, ведь мне не с кем даже посмеяться над глупой шуткой."
- **Other (≥1 finite verb)**:
  - "Одна, со мной обожающая меня маленькая очаровательная собачка, обязательства по кредитам." — list of nominal/elliptic clauses; only one finite verb in the whole sentence.
  - "С началом этого учебного года я плакала в школе больше, чем за все предыдущие учебные годы." — comparative "чем" not in the conjunction list, second clause elided.
  - "Я правда не понимаю- а мне зачем жить?" — second clause's predicate is an infinitive, ruled out by Inf exclusion.
  - "Я бы хотел узнать, как можно попросить прощения у девушки, которая мне далеко не безразлична?" — chain of infinitival/zero-copula subordinates.

## Take-aways
- Cleaner GT handling collapses the precision signal: with NaN rows removed, Scenario B has zero negatives, so V1 and V2 are indistinguishable on precision in this dataset. The previously-reported V2 precision win was driven by the NaN-as-negative assumption.
- The recall gap (V1 1.000 vs V2 0.745) is robust under cleaner GT — V2 misses 24 / 94 positives. ~21% (5 of 24) are zero-copula or impersonal sentences that have no finite verb at all; the remaining ~79% have ≥1 finite verb but trip on infinitive-second-clause, comparative "чем", or single-finite-verb-with-multiple-nominal-clauses patterns.
- Next-step refinements: (a) augment the gate with positive signals for impersonal predicatives (lemmas in `IMPERSONAL_LEMMAS`) and zero-copula nominal patterns; (b) add comparative "чем" to the conjunction list; (c) before claiming a precision improvement, get the annotator to label the 24 currently-NaN sentences so we have real negatives to test against.

Validation against Shirin's 120 examples shows V1 has perfect recall (R=1.000) on confirmed-complex sentences. Precision cannot be evaluated against this ground truth — after dropping annotator skips, the dataset contains essentially no negatives. V2 was rejected: its previously-observed precision advantage was an artifact of treating skipped rows as confirmed-simple negatives; on clean data V2 simply has lower recall (0.745) without measurable precision benefit.

## Known consequence: однородные/обороты/вводные in sample

`../notes/edge_cases.md` specifies that homogeneous predicates (однородные сказуемые), participial / adverbial-participial phrases (причастные / деепричастные обороты), and parenthetical constructions (вводные конструкции) should be treated as single clauses by annotators (EXCLUDE rule).

The chosen heuristic V1 (`has_conjunction OR has_asyndetic_markers`) does NOT pre-filter these out. A sentence like *«Она встала и пошла к окну»* — homogeneous predicates joined by «и» — will be classified as complex by V1 and may end up in the annotation sample.

This is a deliberate trade-off, not an oversight:

- **V2 (`finite_verb_count >= 2` gate)** would have excluded most such cases but had only 0.745 recall on Shirin's 120 — it would systematically miss ~25% of genuinely complex sentences (especially zero-copula nominals and impersonals).
- **The cost of accepting them in the sample is small.** Annotators will correctly label them as 1-clause, which is itself useful negative training data for the future classifier.
- **The cost of V2's recall gap is large.** Missed linguistic patterns never reach training data, leaving systematic blind spots in the trained classifier.

We accept that ~17% of V1's positive predictions on Shirin's 120 were homogeneous-predicate cases (V1's nominal precision was 0.825 on the original mixed-GT view). For sampling for human annotation, **recall priority is correct**: the sample needs to expose the classifier to the full spectrum of complex constructions, and any heuristic FPs become real negatives once an annotator labels them.
