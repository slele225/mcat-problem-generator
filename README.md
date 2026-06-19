# MCAT Question Generator (`mcat-gen`)

Generates validated MCAT practice questions with the **Anthropic Claude API**. Every
item is gated by adversarial review **and** an independent blind-solve check before it
is kept. Output is JSONL, consumed by the separate `mcat-app` web app.

> This is the **generation pipeline**, not the web app. For the full architecture,
> conventions, and gotchas, read **[CLAUDE.md](CLAUDE.md)** and **[docs/](docs/)**.

## Item types

- **Discrete** — standalone science MCQs, generated per topic.
- **Science passage** — a science passage (+ optional table/figures) with 4–7 linked
  questions, for a cluster of 2–3 related topics.
- **CARS** — a 500–600 word humanities/social-science passage with ~5–7 questions.

## How it works

Each question runs through a three-stage funnel, and is **accepted only if both
validation stages pass** (failures are discarded — no revision — and a retry
regenerates):

1. **Generate** (Opus) — produce one item as JSON.
2. **Adversarial review** (Opus) — critique it for flaws.
3. **Blind solve** (Sonnet) — an *independent* model re-solves with no answer key;
   passes only if it picks the keyed answer.

The blind-solve checker is a **different, cheaper model** (Sonnet) on purpose — an
independent check is real signal, whereas "Opus checking Opus" is not.

## Quick start

You need an Anthropic API key and [uv](https://docs.astral.sh/uv/).

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
uv sync                                   # install deps (anthropic, rdkit, matplotlib, psycopg, ...)

# Production config (Opus generation/review, Sonnet checkers). --run-name keeps a run's
# artifacts together; --max-topics + --seed give a small reproducible test run.
uv run python -m src.main --config configs/opus.yaml --run-name myrun --max-topics 10 --seed 42
```

On **Windows**, use the project venv (`.venv\Scripts\python.exe -m src.main ...`) or
`uv run` — the system Python lacks the dependencies.

`./run.sh` is a Linux/macOS convenience wrapper (installs uv, `uv sync`, runs the
pipeline) — but it uses the default smoke `config.yaml`; pass `--config configs/opus.yaml`
for a real run.

## Usage

```bash
uv run python -m src.main --config configs/opus.yaml [flags]
```

| flag | effect |
|---|---|
| `--discrete-only` / `--cars-only` / `--science-passage-only` | run just one pipeline (default: all three) |
| `--run-name NAME` | write all artifacts into `runs/NAME/` (else shared `output/`) |
| `--max-topics N` | process at most N topics this run, per pipeline (N passages for CARS) |
| `--seed S` | reproducible topic/cluster selection |
| `--topic-ids ID1,ID2` | targeted test: build ONE science passage from these ids, then stop |
| `--stats` | show checkpoint/output counts and exit |
| `--render-figures` | render science figures from the JSONL (no API calls); `--force-render` to redo |
| `--recount` | rebuild the discrete checkpoint from its JSONL |
| `--reset` | clear all checkpoints |
| `-v` | verbose logging |

Runs **checkpoint and resume** — re-run the same `--run-name` to continue. Use a fresh
run name for long bank runs so they don't resume stale state.

## Output

Per-run artifacts land in `runs/<name>/` (or `output/` with no `--run-name`):

- `discrete_questions.jsonl` — one question per line.
- `science_passages.jsonl` / `cars_passages.jsonl` — one passage (with nested
  questions) per line.
- `*_generation_metrics.json` — funnel + per-model token/cost.
- `rejected_*.jsonl` — discarded attempts (diagnostic only).
- `checkpoints/`, `run.log`, and (after `--render-figures`) `figures/`.

Field-by-field schemas are in **[docs/schemas.md](docs/schemas.md)**. Example discrete
question:

```json
{
  "question_id": "BB_1A_001_q003",
  "topic_id": "BB_1A_001",
  "section": "Biological and Biochemical Foundations of Living Systems",
  "content_category": "1A: Structure and function of proteins...",
  "topic": "Amino acid structure and classification",
  "subtopics_tested": ["..."],
  "stem": "A researcher observes that glycine...",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "B",
  "explanation": "**Choice B** is correct because...",
  "difficulty": "medium",
  "skill_tested": "Skill 2",
  "validation": {"adversarial_pass": true, "blind_solve_pass": true}
}
```

## Configuration

`configs/opus.yaml` is the production config; `configs/sonnet.yaml` is identical except
the generation `model`. Root `config.yaml` is a tiny smoke config (the default when
`--config` is omitted). Key knobs:

```yaml
model: claude-opus-4-8          # generation + adversarial review

discrete:
  questions_per_topic: 10
  discrete_checker_model: claude-sonnet-4-6   # independent blind-solve checker

cars:
  passages_per_topic: 60                       # TOTAL passages this run
  questions_per_passage_range: [5, 7]
  cars_checker_model: claude-sonnet-4-6

science_passage:
  passages_per_topic_cluster: 2
  questions_per_passage_range: [4, 7]
  science_checker_model: claude-sonnet-4-6
  enable_figures: true                         # SMILES (rdkit) + plots (matplotlib)
```

See the config table in [CLAUDE.md](CLAUDE.md#config-configsopusyaml) for what each
knob controls and its cost implication.

## Figures (science passages)

When `enable_figures: true`, science passages can carry chemistry structures (SMILES,
via RDKit) and data plots (matplotlib), produced by a dedicated figure-pass stage and
rendered in austere monochrome. Rendering is a separate `--render-figures` step. See
**[docs/figures.md](docs/figures.md)**.

## Loading into Supabase

`scripts/load_to_supabase.py` upserts the JSONL banks into Postgres (re-runnable;
needs `SUPABASE_DB_URL`); `scripts/upload_figures.py` uploads rendered figure PNGs to
Storage (needs `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`). Details and the known
harmless pooler note are in **[docs/operations.md](docs/operations.md)**.

## Project structure

```
src/
  main.py            CLI entry point + run modes
  config.py          config dataclasses + YAML loader
  llm_client.py      async Anthropic client: batching, backoff, JSON parse/repair
  schemas.py         Pydantic models, choice normalization, phantom-option detection
  metrics.py         funnel + per-model token/cost tracking
  figures.py         figure validation, text serialization, monochrome rendering
  prompts/           common.py (shared fragments) + discrete/science_passage/cars
  pipelines/         discrete / science_passage / cars + figure_pass
configs/             opus.yaml (prod) · sonnet.yaml ; root config.yaml = smoke
scripts/             Supabase load/upload, fixers, inspectors
runs/<name>/         per-run output, metrics, checkpoints, logs, figures
mcat_topics.json     the MCAT topic catalog
```

## Docs

- **[CLAUDE.md](CLAUDE.md)** — architecture, hard-won rules, config, metrics.
- **[docs/schemas.md](docs/schemas.md)** — item types & every JSONL field.
- **[docs/figures.md](docs/figures.md)** — figure pipeline & rendering conventions.
- **[docs/conventions.md](docs/conventions.md)** — LaTeX/KaTeX, bolding, JSON repair, gotchas.
- **[docs/operations.md](docs/operations.md)** — running, flags, Supabase, scripts.
