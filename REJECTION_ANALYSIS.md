# Rejection Analysis — run `allfix_test`

Read-only analysis of `runs/allfix_test/` (discrete + science + CARS). No code modified, no API calls.

Sources: `generation_metrics.json` (discrete), `science_generation_metrics.json`, `cars_generation_metrics.json`, `run.log` (664 lines), `rejected_science.jsonl` (1 rec), `rejected_cars.jsonl` (4 recs). Note: **no `rejected_discrete.jsonl` exists** — discrete rejections were recovered from the 12 `Adversarial review REJECTED` lines in `run.log`.

---

## 1. Funnel summary per pipeline

Counts are raw question counts from the metrics files. "Failed adversarial / blind" = `generation_parsed − passed_*` (the two gates run on every parsed question).

| Pipeline | attempted | parsed | passed_adversarial | passed_blind_solve | accepted | slots_failed |
|---|---|---|---|---|---|---|
| **discrete** | 101 | 97 | 84 | 91 | 80 | 0 |
| **science**  | 91  | 68 | 67 | 67 | 67 | 1 |
| **cars**     | 76  | 64 | 61 | 63 | 60 | 0 |
| **TOTAL**    | 268 | 229 | 212 | 221 | 207 | 1 |

### Implied losses (where attempts die)

| Pipeline | lost to **parse failure** (attempted−parsed) | failed **adversarial** (parsed−passed_adv) | failed **blind-solve** (parsed−passed_blind) | total lost after parse (parsed−accepted) |
|---|---|---|---|---|
| discrete | **4** | 13 | 6 \* | 17 |
| science  | **23** ⚠ | 1 | 1 | 1 |
| cars     | **12** | 3 | 1 | 4 |
| TOTAL    | **39** | 17 | 8 | 22 |

\* All 6 discrete "blind-solve failures" are **blind-solver JSON-parse failures** (the Sonnet checker emitted prose instead of JSON), *not* answer disagreements — see §4.

**Headline:** Science's dominant loss is **parse failure (23 of 91 attempts ≈ 25%)**, not review rejection — its adversarial/blind gates are nearly clean (1 each). Discrete's losses are spread across adversarial review (13). CARS loses most to parse (12) then adversarial (3).

---

## 2. Rejection reason breakdown (adversarial review)

16 adversarial-rejection events total (12 discrete + 1 science + 3 CARS). Each event lists multiple issues; the table tallies **how many rejection events cite each category at least once** (ranked, most common first). Many issues are tagged by the reviewer with an explicit category prefix (`SKILL ALIGNMENT`, `ANSWER BALANCE`, etc.).

| Rank | Category | Events citing | Notes |
|---|---|---|---|
| 1 | **SKILL ALIGNMENT** (item mislabeled — usually "labeled Skill 1, actually Skill 2") | **9** | The single most common complaint; rarely the *only* issue but pervasive |
| 2 | **AMBIGUITY** (≥2 defensible answers / distractor too close) | **7** | Often the genuinely disqualifying issue |
| 2 | **ACCURACY** (factual / answer-key / explanation errors) | **7** | But only ~3 are *disqualifying-major*; the other ~4 are flagged "minor"/"acceptable" |
| 4 | **PHANTOM OPTION E** (explanation discusses a nonexistent option E) | **5** | See §3 — distinct from the structural `E`-key stripping |
| 4 | **ANSWER BALANCE / GIVEAWAY** (correct choice noticeably longer/more detailed) | **5** | |
| 6 | **DISTRACTOR QUALITY** (distractors too obviously wrong, or duplicate of key) | **4** | |
| 7 | **STEM QUALITY / PRECISION** (vague or self-contradicting stem) | **3** | |
| 8 | **EXPLANATION INCOHERENCE** (scratch-work artifacts like "…recompute" in final text) | **2** | Overlaps with major ACCURACY events |

### Verbatim examples per category (truncated ~120 chars)

- **SKILL ALIGNMENT:**
  - `"SKILL MISALIGNMENT: The question is labeled Skill 1 ... but it requires multi-step application of constructs to a novel scenario..."`
  - `"Skill alignment is questionable. The stated skill is Skill 3 ... but the question does not require evaluating experimental design..."`
- **AMBIGUITY:**
  - `"Ambiguity / multiple defensible answers: ... This is the textbook definition of learned helplessness ... A student could strongly defend D."`
  - `"Distractor D contains a subtle internal contradiction that creates ambiguity ... making two answers defensibly pointing to the same pH."`
- **ACCURACY (major):**
  - `"ACCURACY/ANSWER ERROR: The actual specificity-constant ratio is 100 ... but the keyed answer A states 'about 50'. ... unanswerable as written"`
  - `"the keyed answer C ... contradicts the math; the smallest per-percentage-point rise is the 20%-to-40% interval ... which is Choice A."` (science)
- **PHANTOM OPTION E:**
  - `"The explanation references a choice E ('E is wrong because DNA replication uses the existing DNA strand as a template...') that does not exist"`
  - `"Explanation references a fifth option that does not exist ... written for a different (5-choice) version of the item"`
- **ANSWER BALANCE / GIVEAWAY:**
  - `"Choice A is noticeably longer and contains two clauses ... a potential test-taking giveaway that the answer is correct."`
  - `"ANSWER BALANCE: Choice A is noticeably longer and more elaborately constructed than B, C, and D..."` (CARS)
- **DISTRACTOR QUALITY:**
  - `"DISTRACTOR QUALITY: Distractors C and D contain factual misstatements that are quite easy to eliminate ... read as obviously wrong"` (CARS)
  - `"Answer A and Answer C give the same anticodon sequence (3'-CCC-5') ... a student ... would face an ambiguous choice between A and C"`
- **STEM QUALITY / PRECISION:**
  - `"STEM PRECISION: The phrase 'despite differing in another construct' is vague and awkward ... forcing the student to reverse-engineer..."`
- **EXPLANATION INCOHERENCE:**
  - `"EXPLANATION INCOHERENCE: The explanation visibly computes the factor as 100, then contradicts itself ... contains scratch-work artifacts ('...recompute')"`

---

## 3. The "Option E" / phantom-choice issue

Two **distinct** phenomena — keep them separate:

**(a) Structural junk keys auto-stripped from `choices` — 120 `Stripping unexpected choice key(s)` warnings (all from `src.schemas`).** These are silently repaired and do **not** by themselves cause rejection. Breakdown of stripped keys:

| Key | Count | | Key | Count |
|---|---|---|---|---|
| `'E'` | **91** | | `'extra'` | 3 |
| `'B2'` | 3 | | `'answer_basis'` | 2 |
| `'F'` | 2 | | `'answer_unused'` | 2 |
| `'A2'`,`'D2'`,`'B_'`,`'B_correct'` | 1 each | | misc junk (`_comment`, `selected_answer`, `correct_answer`, `difficulty`, `key`, `default`, `$\Delta$`, `minus`, `downward`, `explanation_note`, `answer_blocker`, `_unused`, `'I have read the rules and there is no option E.'`) | 1 each |

→ Phantom **option-letter** keys (`E`/`F`/`A2`/`B2`/`D2`): **~99 of 120**. The lone `'I have read the rules and there is no option E.'` key is the model arguing back inside the JSON. So the no-E instruction is *mostly* obeyed structurally (the key is dropped) but the model still emits an `E` key on **~91 occasions** before stripping.

**(b) Adversarial rejections that specifically cite a nonexistent option E in the explanation prose — 5 events** (4 discrete + 1 science). This is the part the stripper does **not** catch: the `choices` object is clean A–D, but the **explanation text** still says "E is wrong because…". These are genuine rejections.

**Relative size:** phantom-E-in-explanation = **5 of 16 rejection events (~31%)** — tied for the #4 rejection cause. It is a meaningful, recurring quality defect, though smaller than skill-misalignment (9) and ambiguity (7). The 91 structural `E` strips are "free" (auto-fixed) but signal the model is still internally generating 5-option items.

---

## 4. Blind-solve mismatches

8 total blind-solve "failures," but they split into two very different kinds:

- **Discrete — 6 failures, ALL solver-side JSON parse failures, zero answer mismatches.** Every one is a `Blind solve failed: Could not parse JSON from response: I need to find...` line — the Sonnet checker wrote prose reasoning instead of the required JSON, so the gate couldn't read its answer. These are *not* evidence of flawed questions; they are checker-output-format failures. (e.g. "I need to find the minimum height of fluid in the IV bag…", "I need to find the depth of the reflecting structure…")
- **Science — 1 genuine mismatch (`SP_CP_5D_002_01_q06`):** blind chose **A**, key said **C**, confidence **high**. The checker was **right** — the same question also failed adversarial because the explanation's own math points to A. A correctly-caught broken item.
- **CARS — 1 genuine mismatch (`CARS_P0002_q03`):** blind chose **D**, key said **A**, confidence **high**, *adversarial passed with no issues*. This is checker-vs-author disagreement on a hard "Reasoning Beyond" multi-position item — ambiguous rather than clearly broken.

**Takeaway:** genuine blind-solve answer disagreements are rare (**2 across the whole run**), and one of those correctly flagged a wrong key. The discrete blind "failures" are a **checker output-format problem**, not a question-quality problem.

---

## 5. Parse failures / truncation

- **Total parse failures lost at generation: 39** (discrete 4, **science 23**, cars 12), from `attempted − generation_parsed`.
- **No truncation detected** — 0 occurrences of "truncat" in the log. The malformed JSON is not length-driven.
- `run.log` parse-fail lines: 32 generic `src.llm_client` "Failed to parse" wrappers; pipeline-tagged lines split cars 12 / science 12 / discrete 8 (these include blind-solve/review responses, so they don't map 1:1 to the funnel's generation parse losses — the funnel counts above are authoritative).
- 25 of the parse failures begin with `{` (malformed/early-terminated JSON object); the remainder begin with prose ("I need to…", "Let me verify the calculation.") — i.e. the model started reasoning in plain text instead of emitting JSON.
- **1 `Exhausted retries`** event: `science_passage SP_BB_2C_006_01 q7` (→ the 1 science `slots_failed`), with "0 answer_basis rejection(s)" — so it exhausted on parse/quality, not answer-basis mislabeling.
- **answer_basis mislabel rejections: 1** total mention — the answer_basis-honesty check is essentially not a rejection driver this run.

---

## 6. What's worth fixing (where the leverage is — factual, not prescriptive)

Ranked by how many attempts each issue actually costs:

1. **Science generation parse failures — 23 lost attempts (~25% of science attempts).** This is the single largest leak in the entire run, and it is purely a JSON-formatting/output-contract problem (no truncation), not question quality. Science's review gates are otherwise nearly perfect (1 adversarial, 1 blind fail). CARS shares a smaller version (12).
2. **SKILL ALIGNMENT mislabeling — cited in 9 of 16 rejection events.** Overwhelmingly "labeled Skill 1, actually requires Skill 2 reasoning" (and a few Skill 3 "decorative experimental wrapper" cases). The most frequent adversarial complaint, concentrated in discrete.
3. **AMBIGUITY (7 events) and PHANTOM-E-in-explanation (5 events)** are the next tier. Ambiguity tends to be the genuinely disqualifying flaw; phantom-E-in-explanation is a self-contained, recurring editing defect (explanation written as if a 5th option exists) that the schema stripper cannot catch because it lives in prose.

Lower-leverage / not worth chasing on these numbers: the 91 auto-stripped structural `E` keys (already silently repaired), the discrete blind-solve "failures" (checker output-format, not question quality), and answer_basis mislabeling (1 mention).
