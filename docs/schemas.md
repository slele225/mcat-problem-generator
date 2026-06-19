# Item types & schemas

Source of truth: `src/schemas.py`. This file documents the **fields you'll actually
see in the output JSONL** and the non-obvious validation behavior. The pipeline
parses raw LLM output into a `Raw*` model first, then (for accepted items) emits the
stored shape below.

Two model layers per item type:
- `RawDiscreteQuestion` / `RawSciencePassage` / `RawScienceQuestion` /
  `RawCARSPassage` / `RawCARSQuestion` — lenient parse of the generation LLM's JSON.
- `DiscreteQuestion` / `SciencePassage` / `ScienceQuestion` / `CARSPassage` /
  `CARSQuestion` — the stored, fully-populated record (the pipeline adds ids,
  topic/section metadata, `validation`, etc. before writing).

The stored records are assembled by hand in each pipeline's
`generate_and_validate_*` (e.g. `src/pipelines/discrete.py:194`), so the dict keys
in the JSONL are authoritative — match them, don't guess from the Pydantic class.

## Shared rules (all item types)

- **`choices` is EXACTLY `{A, B, C, D}`** — non-empty strings. `normalize_choices`
  strips any extra key (a stray `"E"`, `"C2"`) with a warning; if all four aren't
  present after stripping it raises → regeneration. This cleans the **choices object
  only**, not explanation prose.
- **`correct_answer` ∈ {A, B, C, D}** (upper-cased on parse).
- **`difficulty` ∈ {easy, medium, hard}** (defaults to `"medium"` where optional).
- **`skill_tested` / `skill_type` is the pipeline-ASSIGNED label, not the model's
  echo.** Each pipeline persists the skill it *planned* for the slot, because the
  model's echoed `skill_tested` is often blank or mislabeled. See
  `discrete.py:209`, `science_passage.py:645`, `cars.py:336`.
- **Phantom fifth-option references are rejected**, not edited — see
  [conventions.md](conventions.md#phantom-option-e).

## Discrete question (`discrete_questions.jsonl`)

One JSON object per line.

| field | notes |
|---|---|
| `question_id` | `{topic_id}_q{NNN}` (e.g. `BB_1A_001_q003`) |
| `topic_id` | from `mcat_topics.json` |
| `section`, `content_category`, `topic_group`, `topic` | copied from the topic |
| `subtopics_tested` | `list[str]`, model-supplied |
| `stem`, `choices`, `correct_answer`, `explanation` | the question |
| `difficulty` | easy/medium/hard |
| `skill_tested` | AAMC SIRS skill (Skill 1–4), pipeline-assigned |
| `validation` | `{adversarial_pass, blind_solve_pass}` (both true for stored items) |

## Science passage (`science_passages.jsonl`)

One **passage** per line, with its linked questions nested under `questions`.

Passage fields: `passage_id`, `section`, `content_category`, `topic_ids` (the
cluster's 2–3 topics), `topic_group`, `passage_text`, `table_markdown`
(`Optional[str]`, normalized to `null` when the model emits "none"/""), `figures`
(`list[FigureSpec]`, passage-level), `word_count`, `questions`, `validation`.

Each nested **science question** (`ScienceQuestion`):

| field | notes |
|---|---|
| `question_id` | `{passage_id}_q{NN}` |
| `passage_id` | parent |
| **`topic_id`** | **per-question content topic** — one of the passage's cluster topics, steered at generation (`build_question_plan`) so science questions feed the weak-topic engine. `Optional` (older records predate it). |
| `topic` | human topic name mirror |
| `skill_tested` | SIRS skill, pipeline-assigned (`plan["skill_label"]`) |
| `answer_basis` | `from_passage` \| `apply_knowledge` \| `data_interpretation` — normalized leniently (`normalize_answer_basis`); judged by a separate lenient reviewer verdict |
| `stem`, `choices`, `correct_answer`, `explanation`, `difficulty` | the question |
| `figures` | `list[FigureSpec]`, question-level |
| `validation` | `{adversarial_pass, blind_solve_pass, answer_basis_ok}` |

`answer_basis_ok` comes from `ScienceAdversarialReview` — a **separate, lenient**
verdict that defaults `True` and is only set `False` for a *clear* mislabel, so a
reviewer who omits it never trips the check.

## CARS passage (`cars_passages.jsonl`)

One **passage** per line. Passage fields: `passage_id` (`CARS_P{NNNN}`),
`passage_text`, `word_count`, `subject` (e.g. "Philosophy"), `category`
(`humanities`/`social_science`), `structure_type` (`single_voice`/`multi_position`),
`questions`, `validation`.

Each nested **CARS question** (`CARSQuestion`):

| field | notes |
|---|---|
| `question_id` | `{passage_id}_q{NN}` |
| `skill_type` | Foundations of Comprehension / Reasoning Within the Text / Reasoning Beyond the Text — pipeline-assigned |
| `stem`, `choices`, `correct_answer`, `explanation` | the question |
| `difficulty` | per-question (Phase 2); defaults `"medium"` for older records |
| `validation` | `{adversarial_pass, blind_solve_pass}` |

**CARS deliberately has NO content `topic_id`** — it tests reading *skill*, not a
content topic, so it does not feed the weak-topic engine (the Supabase loader sets
`topic_id = None` for CARS at `scripts/load_to_supabase.py:191`).

## Figure specs

`FigureSpec` (stored) / `RawFigureSpec` (generated). `figure_type` ∈ {`smiles`,
`plot`}; exactly the matching payload must be present (`_validate_figure_payload`).
- `smiles` → `SmilesPayload{ molecules: [{smiles, label}] }` (≥1 molecule).
- `plot` → `PlotPayload{ chart_type ∈ {bar,line,scatter,histogram}, title, x_label,
  y_label, series: [{name, x, y, y_err?}] }` (x/y must be equal length).
- `caption`, `alt_text` optional; `figure_id` pipeline-assigned; `image_path` filled
  by the separate render pass (`null` until then).

See [figures.md](figures.md) for how figures flow through generation, validation,
and rendering.

## Review / blind-solve result models

- `AdversarialReview{ passed: bool, issues: list[str], reasoning }` — `issues`
  accepts strings OR objects and coerces each to a string (`_normalize_issue_list`),
  so a reviewer that returns `{"issue": "..."}` objects doesn't get recorded as a
  failed review.
- `ScienceAdversarialReview` — adds `answer_basis_ok` (default `True`) +
  `answer_basis_note`.
- `BlindSolveResult{ chosen_answer, confidence ∈ {high,medium,low}, reasoning }` —
  blind-solve **passes iff `chosen_answer == correct_answer`**.
