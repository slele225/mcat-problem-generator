# CLAUDE.md — mcat-gen (question-generation pipeline)

> This repo is the **question-generation pipeline** (`mcat-gen`). It is NOT the
> `mcat-app` web app — that is a separate repo. This repo produces validated
> question/passage JSONL; mcat-app consumes it.

## Project overview

`mcat-gen` generates validated MCAT practice questions across three item types —
**discrete** (standalone science), **science-passage** (passage + linked
questions), and **CARS** (humanities/social-science passage + questions) — via an
LLM pipeline, each item gated by adversarial review and an independent blind-solve
check before it is kept. Output JSONL is consumed by the separate `mcat-app` web app.

Run pattern (the user runs these — see next note):
```
uv run python -m src.main --config configs/opus.yaml --run-name X --max-topics N --seed S
```
- `--max-topics N` is **per-pipeline** (caps not-yet-completed topics for *each*
  pipeline this run; for CARS it caps passages). Applied after a seeded shuffle.
- Flags: `--discrete-only` / `--cars-only` / `--science-passage-only` restrict to one
  pipeline; with none, all three run. `--seed` makes topic selection reproducible.
  `--stats`, `--reset`, `--recount`, `--render-figures` are utility modes.
- All artifacts (questions JSONL, checkpoints, `*_generation_metrics.json`, `run.log`)
  go to `runs/<run-name>/`. Omitting `--run-name` uses the shared `output/`.
- **There is NO `ANTHROPIC_API_KEY` in the Claude Code environment.** Claude Code
  edits code and **prints** the run command; the **user runs it**. Do not attempt
  pipeline runs / API calls from here. (`src.main` health-checks the key at startup
  and exits if missing.)

## Architecture (verified against code)

**Three pipelines**, one per item type: `src/pipelines/discrete.py`,
`science_passage.py`, `cars.py`. Each follows a three-stage funnel:

1. **Generation** — produce one raw item (JSON), parse into a `Raw*` schema.
2. **Adversarial review** — a reviewer model critiques it (`passed` bool + issues).
3. **Blind-solve verification** — an independent checker re-solves with no answer key;
   pass iff its chosen letter == the keyed answer.

An item is **accepted iff `adversarial_pass AND blind_solve_pass`** (discrete:
`src/pipelines/discrete.py:171`; science/CARS analogous). Items are generated one at a
time with retries (`max_retries`, default 3); accepted stems are fed back for
within-topic/within-passage diversity. Funnel + token/cost counters are tracked by
`MetricsTracker`/`TopicMetrics` (`src/metrics.py`) and written to:
`generation_metrics.json` (discrete), `science_generation_metrics.json`,
`cars_generation_metrics.json`. Discarded attempts are logged to `rejected_*.jsonl`
for analysis (diagnostic only — never affects accept/reject).

**Model per stage** (`configs/opus.yaml`): main `model: claude-opus-4-8` runs
**generation + adversarial review**. **Blind-solve** runs on a *per-pipeline* checker
model field, each set to `claude-sonnet-4-6`:
- `discrete.discrete_checker_model`
- `science_passage.science_checker_model`
- `cars.cars_checker_model`

(Confirmed in `src/config.py` and the pipelines.) Rationale: an independent,
cheaper-than-Opus checker rather than "Opus checking Opus." NOTE: a top-level
`Config.checker_model` (default `claude-haiku-4-5-20251001`) still exists but is **not
used by any pipeline** — all three use their per-pipeline field. (This corrects the
older `cost-metrics-model-driven` memory, which says discrete blind-solve runs on
Haiku; it now runs on Sonnet via `discrete_checker_model`.)

**Science pipeline extra stages** (beyond the per-question funnel): passage
generation → passage review → **dedicated FIGURE PASS** (`src/pipelines/figure_pass.py`,
`run_figure_pass`, wired in `_generate_passage_set_once`, gated on
`enable_figures`) → per-question generation/review/blind-solve. Figures come **from
the figure-pass stage, NOT inlined in the passage/question prompts** — do not move
figure-decision logic back into a prompt. The figure pass is best-effort (never
crashes the run); invalid specs are dropped per-spec and the original table kept as
fallback. `model` is an injected param (Opus today; swappable for an A/B later). A
`REQUIRED_FIGURES` backstop in `science_passage.py` enforces a required figure type
for certain content categories. Figures are two declarative types only — `smiles`
(RDKit) and `plot` (matplotlib); validation models see a TEXT serialization
(`src/figures.py:figure_to_text`), never the image. Rendering is a separate
`--render-figures` pass. IONIC/DISCONNECTED BACKSTOP: within the `smiles` render
path, `_render_smiles` detects disconnected/ionic species (`Chem.GetMolFrags(mol) > 1`
or a metal atom — salts, hydrates, ion pairs that RDKit can only draw as
"fragment soup") and renders a clean monochrome FORMULA card instead of a 2D
structure; single connected covalent molecules (incl. zwitterions) still draw
normally. The figure-pass prompt also instructs the model not to emit structure
figures for ionic compounds/salts/hydrates in the first place. This is a
render-time fallback, NOT a new figure type.

**Shared prompt fragments** live in `src/prompts/common.py`:
`NO_FIFTH_OPTION_RULE`, `NO_FIFTH_OPTION_EXPLANATION_RULE`,
`NO_FIFTH_OPTION_CONTRACT`, `LATEX_NOTATION_RULE`, `LATEX_REVIEW_NOTE`.
- **Only `discrete.py` and `science_passage.py` import these** — editing a shared
  fragment affects *both* importers. **CARS (`cars.py`) does NOT import from
  `common.py`**: it keeps its own inline no-fifth-option wording (by design — see the
  `common.py` docstring) and does not use the LaTeX rules (CARS is prose, no math
  notation). So a fix to a shared fragment does **not** propagate to CARS; CARS must
  be edited separately.

**`parse_json_response`** (`src/llm_client.py`): strict `json.loads` first (well-formed
responses returned untouched), then repair fallbacks — markdown-fence strip,
**balanced-brace extraction** (`_extract_balanced_object`, recovers a JSON object
embedded in prose, string-literal aware so a `}` inside a value doesn't close early),
and LaTeX backslash escape-repair (`_repair_json_escapes`, doubles single backslashes
so `$\Delta$` etc. survive JSON). Genuinely truncated JSON **fails cleanly**
(`ValueError`) — no fabrication.

**`src/schemas.py`** (`normalize_choices`): enforces EXACTLY keys A/B/C/D, each a
non-empty string; **strips unexpected choice keys** (e.g. a stray `"E"`) from the
`choices` object, logging a warning. If the four required keys aren't all present after
stripping, it raises → triggers regeneration. Note: this only cleans the `choices`
object — it does NOT scrub "Option E" references from explanation **prose**.

**Generation params** (verified): `max_tokens=2048` for question generation (raised
from 1024 to stop LaTeX-heavy JSON truncating mid-string); figure pass uses 1536.
Temperatures: `temperature_generate=0.8`, `temperature_validate=0.3` (all three
pipeline configs).

## Hard-won rules — DO NOT BREAK THESE

1. **ACCURACY-FIRST ORDERING.** The discrete and science question-generation prompts
   open with a `#1 PRIORITY — SCIENTIFIC ACCURACY` block **before** the formatting
   rules. Do NOT reorder formatting above accuracy. Overstuffing these prompts with
   formatting rules previously crashed discrete acceptance from ~73% to **21%**
   (visible in `runs/pretest3_all/`: adversarial 11/52) by pulling attention off
   correctness.
2. **DO NOT DUPLICATE RULES.** Each rule (notably the LaTeX rule) must appear **once**
   per prompt — it was previously duplicated, causing bloat. Before adding any
   instruction, check it isn't already present (and remember CARS has its own inline
   copies, separate from `common.py`).
3. **DO NOT OVERSTUFF PROMPTS.** Keep prompts lean. Competing instructions must be
   **field-scoped** so they don't contradict: the stem length cap applies to the
   `stem` field ONLY; the `explanation` is "thorough but focused." Keep these scopes
   explicit.
4. **JSON-OUTPUT DISCIPLINE.** Question-gen and blind-solve prompts end with an
   `OUTPUT FORMAT — ABSOLUTE: respond with ONLY the JSON object` instruction. Models
   tend to emit prose preamble ("I need to…", "Let me…") that breaks parsing. Keep
   this instruction; the parse-side balanced-brace extraction is the safety net, not a
   substitute.
5. **PHANTOM OPTION E.** Models trained on 5-option tests persistently emit an "E"
   choice and reference "Option E" in explanations despite instructions. `schemas.py`
   strips E from the `choices` object, and prompts forbid E references — but **E
   references can still survive in explanation prose**. Intended approach is
   **DETECT-AND-REJECT** (let a retry regenerate), NOT edit-and-keep (editing risks
   incoherence). **Status: NOT yet implemented** — there is currently no code that
   scans explanation prose for "Option E" and rejects. Prevention today is
   prompt-only + choices-object stripping. (See Pending work.)
6. **ONE CHANGE AT A TIME.** When fixing prompts, change one thing, re-test small
   (`--max-topics 10 --seed 42`), verify, then stack the next change. Show before/after
   diffs of only the changed lines and confirm the rest is byte-identical.
7. **CHECKPOINTING.** Runs checkpoint by `--run-name` and resume (under-quota topics
   stay open; `--recount` rebuilds the discrete checkpoint from the JSONL). Use a fresh
   `--run-name` for long bank runs so they don't resume stale state.

## Key metrics / economics (from verified runs)

From `runs/allfix_test/` (most recent all-pipeline run, opus.yaml):

| Pipeline | accepted / attempted | rate | cost / accepted |
|---|---|---|---|
| discrete | 80 / 101 | **79%** | ~$0.083 |
| science  | 67 / 91  | **74%** (pre-JSON-fix) | ~$0.105 |
| CARS     | 60 / 76  | **79%** | ~$0.083 |

These are the healthy targets — a future regression should be visible against them.
Science's 74% was **pre-JSON-fix**: 23 of 91 attempts failed to parse (parsed 68/91)
because the model emitted prose instead of bare JSON; the output-contract + balanced-
brace-extraction fix targets that loss. The 21% discrete crash in `runs/pretest3_all/`
is the cautionary baseline for rule #1.

## Known pending work

- **Detect-and-reject for phantom-E-in-explanation** (rule #5) — not implemented;
  only prompt prevention + `choices`-object stripping exist today.
- **Skill mislabeling on discrete** — the generator sometimes assigns a SIRS skill the
  question doesn't actually test (a top adversarial-rejection cause); the
  adversarial-review prompt checks "SKILL ALIGNMENT" but mislabels still drive
  rejections.
- **Figure-pass model A/B** — the stage is built model-swappable (e.g. Fable 5 for the
  figure decision alone) but currently always uses `config.model` (Opus).

## Memory reconciliation note

The auto-memory file **`cars-pipeline-phase2` is STALE**: it describes CARS *before*
the Phase 2.1 fixes. The current `cars.py` (`cars_question_prompt`) includes all three
2.1 fixes — verified: **(a) no phantom Option E** (explicit "exactly four options A–D,
no option E" plus explanation ban on "Option E"); **(b) D-position nudge** ("writers
tend to under-use later letters such as D … resist that bias for whichever letter is
assigned"); **(c) target-letter fix** — guidance references the actual
`{target_answer}`, NOT a hardcoded "D". The `figure-pass-stage` and
`cost-metrics-model-driven` memories are otherwise accurate, except the latter's claim
that discrete blind-solve runs on Haiku (now Sonnet via `discrete_checker_model`).
