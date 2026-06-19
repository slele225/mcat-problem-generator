# CLAUDE.md — mcat-gen (question-generation pipeline)

> This repo is the **question-generation pipeline** (`mcat-gen`). It is NOT the
> `mcat-app` web app — that is a separate repo. This repo produces validated
> question/passage JSONL; mcat-app consumes it. `README.md` is the human-facing
> overview; this file + `/docs` are the detailed context for code work.

## What this is

`mcat-gen` generates validated MCAT practice questions across three item types via an
LLM pipeline, each item gated by adversarial review and an independent blind-solve
check before it is kept:
- **discrete** — standalone science MCQs (one per topic).
- **science-passage** — a science passage (+ optional table/figures) with 4–7 linked
  questions, generated for a cluster of 2–3 related topics.
- **CARS** — a 500–600 word humanities/social-science passage with ~5–7 questions.

There are **746 non-CARS topics** in `mcat_topics.json` (discrete + science clusters);
CARS is passage-based and draws subjects from configured discipline lists, not topics.

> **No `ANTHROPIC_API_KEY` here.** Claude Code edits code and **prints** the run
> command; **the user runs it**. `src.main` health-checks the key at startup and exits
> if missing. Do not attempt pipeline runs / API calls from a Claude Code session.

## Architecture

**Three pipelines**, one per item type: `src/pipelines/{discrete,science_passage,cars}.py`.
Each question goes through a three-stage funnel:

1. **Generation** (Opus) — produce one raw item (JSON), parse into a `Raw*` schema.
2. **Adversarial review** (Opus) — reviewer critiques it (`passed` bool + issues).
3. **Blind-solve verification** (Sonnet) — an independent checker re-solves with **no
   answer key**; passes iff its chosen letter == the keyed answer.

**Accepted iff `adversarial_pass AND blind_solve_pass`** (discrete
`src/pipelines/discrete.py:191`; science also requires `answer_basis_ok`). **No
revision** — failures are discarded and a retry regenerates from scratch
(`max_retries`, default 3). Roughly **3–4 API calls per accepted item**; acceptance
is high — **~74–79%** on the most recent all-pipeline run (see Key metrics below),
higher on some individually-tuned runs.
Accepted stems are fed back so later items in the same topic/passage cover a different
angle (within-topic/within-passage diversity). Discarded attempts are logged to
`rejected_*.jsonl` (diagnostic only — never affects accept/reject).

**Why Sonnet for blind-solve, not Opus:** independence. An independent (and
cheaper-than-Opus) checker re-solving the item is a real signal; "Opus checking Opus"
is not. Each pipeline has its own checker field — `discrete.discrete_checker_model`,
`science_passage.science_checker_model`, `cars.cars_checker_model` — all set to
`claude-sonnet-4-6` in `configs/opus.yaml`. A top-level `Config.checker_model`
(default Haiku) exists but is **unused** by any pipeline.

**Funnel + token/cost** are tracked by `MetricsTracker`/`TopicMetrics`
(`src/metrics.py`), priced per the **actual model that ran each call**, written to
`generation_metrics.json` (discrete), `science_generation_metrics.json`,
`cars_generation_metrics.json`.

**Science pipeline extra stages** (beyond the per-question funnel): passage generation
→ passage review → **dedicated FIGURE PASS** → per-question funnel. Figures come from
the figure-pass stage, NOT the prompts. See [docs/figures.md](docs/figures.md).

### Subsystem guides (`/docs`)
- **[docs/schemas.md](docs/schemas.md)** — item types & every JSONL field (discrete,
  science passage + questions, CARS passage + questions, figure specs, review models).
- **[docs/figures.md](docs/figures.md)** — figure pass, `REQUIRED_FIGURES`, monochrome
  rendering, ionic→formula-card fallback, `--render-figures`.
- **[docs/conventions.md](docs/conventions.md)** — KaTeX/LaTeX (`\textmu`),
  generation-side bolding, phantom-E, JSON repair, Opus temperature, Windows/PowerShell.
- **[docs/operations.md](docs/operations.md)** — running, all CLI flags, `--stats`,
  Supabase load + figure upload, helper scripts.

## Config (`configs/opus.yaml`)

| knob | controls | cost implication |
|---|---|---|
| `model` | generation + adversarial review (`claude-opus-4-8`) | dominant cost driver |
| `*_checker_model` (per pipeline) | blind-solve model (`claude-sonnet-4-6`) | the cheaper independent check |
| `checker_model` (top-level) | **unused** by pipelines (legacy) | — |
| `discrete.questions_per_topic` | discrete items per topic (10 in opus.yaml) | linear in bank size |
| `cars.passages_per_topic` | TOTAL CARS passages this run (not per-category; 60) | linear |
| `cars.questions_per_passage_range` | per-passage question count, drawn uniformly ([5,7]) | linear |
| `science_passage.passages_per_topic_cluster` | passages per 2–3 topic cluster (2) | linear |
| `science_passage.questions_per_passage_range` | per-passage question count ([4,7]) | linear |
| `science_passage.enable_figures` | turns on figures + `REQUIRED_FIGURES` (needs rdkit/matplotlib) | a few extra calls/passage |
| `*.temperature_generate` / `_validate` | 0.8 / 0.3 — **but Opus 4.7+ ignores `temperature`** (see conventions) | — |
| `*.batch_size`, `*.max_retries` | concurrency / retry budget | retries cost calls |

`configs/sonnet.yaml` is identical except `model`. Root `config.yaml` is a tiny smoke
config (the default when `--config` is omitted).

**Generation token budgets** (raised so LaTeX-heavy JSON doesn't truncate mid-string):
discrete question 3072, science question 3072, science passage 4096, CARS passage 3072,
CARS question 3072, figure pass 1536.

## Hard-won rules — DO NOT BREAK THESE

1. **ACCURACY-FIRST ORDERING.** Discrete and science question prompts open with a
   `#1 PRIORITY — SCIENTIFIC ACCURACY` block **before** formatting rules. Do NOT
   reorder formatting above accuracy. Overstuffing formatting once crashed discrete
   acceptance from ~73% to **21%** (`runs/pretest3_all/`: adversarial 11/52).
2. **DO NOT DUPLICATE RULES.** Each rule (notably the LaTeX rule) appears **once** per
   prompt. Check before adding (and remember CARS has its own inline copies).
3. **DO NOT OVERSTUFF PROMPTS.** Keep them lean; competing instructions must be
   **field-scoped** so they don't contradict (stem length cap applies to `stem` only;
   `explanation` is "thorough but focused").
4. **JSON-OUTPUT DISCIPLINE.** Generation + blind-solve prompts end with
   `OUTPUT FORMAT — ABSOLUTE: respond with ONLY the JSON object`. Models emit prose
   preamble that breaks parsing; the balanced-brace extraction is a net, not a
   substitute. (See [conventions.md](docs/conventions.md#json-parsing--repair).)
5. **PHANTOM OPTION E — detect-and-reject (IMPLEMENTED).** `normalize_choices` strips
   a stray `E` from the choices object; `explanation_has_phantom_option`
   (`src/schemas.py`) rejects any item whose **explanation prose** references a phantom
   fifth option, wired into all three pipelines. It **rejects and regenerates** — never
   edits the explanation. (See [conventions.md](docs/conventions.md#phantom-option-e).)
6. **ONE CHANGE AT A TIME.** Change one prompt thing, re-test small
   (`--max-topics 10 --seed 42`), verify, then stack the next. Show before/after of
   only the changed lines.
7. **CHECKPOINTING.** Runs checkpoint by `--run-name` and resume (under-quota topics
   stay open; `--recount` rebuilds the discrete checkpoint from the JSONL). Use a
   **fresh `--run-name`** for long bank runs so they don't resume stale state.
8. **SHARED vs. CARS prompts.** `src/prompts/common.py` (`NO_FIFTH_OPTION_*`,
   `LATEX_NOTATION_RULE`, `CHOICE_REFERENCE_BOLD_RULE`, `LATEX_REVIEW_NOTE`) is imported
   by **discrete and science only**. **CARS does not import `common.py`** — it keeps its
   own inline copies and has no LaTeX rule (prose, no math). A shared-fragment edit does
   **not** reach CARS; edit CARS separately.

## Key metrics / economics (from verified runs)

From `runs/allfix_test/` (most recent all-pipeline run, opus.yaml):

| Pipeline | accepted / attempted | rate | cost / accepted |
|---|---|---|---|
| discrete | 80 / 101 | **79%** | ~$0.083 |
| science  | 67 / 91  | **74%** (pre-JSON-fix) | ~$0.105 |
| CARS     | 60 / 76  | **79%** | ~$0.083 |

These are the healthy targets — a regression should be visible against them. Science's
74% was pre-JSON-fix (23/91 failed to parse because the model emitted prose instead of
bare JSON; the output-contract + balanced-brace fix targets that). The 21% discrete
crash in `runs/pretest3_all/` is the cautionary baseline for rule #1.

## Known pending work

- **Skill mislabeling on discrete** — the generator sometimes assigns a SIRS skill the
  question doesn't actually test (a top adversarial-rejection cause). Note: the
  **persisted** `skill_tested` is always the pipeline-ASSIGNED label (authoritative),
  not the model's echo — so the saved tag is consistent even when the question content
  drifts from it.
- **Figure-pass model A/B** — the stage is built model-swappable (e.g. a cheaper model
  for the figure decision alone) but currently always uses `config.model` (Opus).

## Repo map

```
src/
  main.py            CLI entry point + run modes (--stats/--recount/--render-figures/--reset)
  config.py          dataclass config + YAML loader
  llm_client.py      async Anthropic client: batching, backoff, JSON parse/repair, Opus-temp handling
  schemas.py         Pydantic models (Raw* + stored), normalize_choices, phantom-option detect
  metrics.py         MetricsTracker / TopicMetrics — funnel + per-model token/cost
  checkpoint.py      CheckpointManager (resume) + OutputWriter (jsonl append)
  figures.py         figure validation, text serialization, monochrome rendering
  prompts/           common.py (shared fragments) + discrete.py / science_passage.py / cars.py
  pipelines/         discrete.py / science_passage.py / cars.py + figure_pass.py
configs/             opus.yaml (prod) · sonnet.yaml (A/B) ; root config.yaml = smoke
scripts/             load_to_supabase.py, upload_figures.py, fix_textmu.py, inspectors (see operations.md)
runs/<name>/         per-run artifacts: *.jsonl banks, *_generation_metrics.json, checkpoints/, run.log, figures/
mcat_topics.json     the MCAT topic catalog (UTF-8 BOM — read with utf-8-sig in ad-hoc scripts)
```
