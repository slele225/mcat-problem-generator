# How-to / operations

> **There is NO `ANTHROPIC_API_KEY` in the Claude Code environment.** Claude Code
> edits code and **prints** the run command; the **user runs it**. `src.main`
> health-checks the key at startup and exits if missing. Do not attempt pipeline
> runs / API calls from a Claude Code session.

On Windows, invoke via the project venv: `.venv\Scripts\python.exe -m src.main ...`
(or `uv run python -m src.main ...`). System `python` lacks the deps.

## Running a generation

```
uv run python -m src.main --config configs/opus.yaml --run-name myrun --max-topics 10 --seed 42
```

- `--config` — pipeline config. `configs/opus.yaml` is the production config (Opus
  generation/review, Sonnet checkers). `configs/sonnet.yaml` is identical except
  `model`. The default `config.yaml` is a tiny smoke config.
- `--run-name NAME` — write **all** artifacts into `runs/NAME/` (questions JSONL,
  `checkpoints/`, `*_generation_metrics.json`, `run.log`). Omit → shared `output/`.
  Use a **fresh run name for long bank runs** so they don't resume stale checkpoints.
- `--max-topics N` — process at most N not-yet-completed topics **per pipeline**
  (for CARS, N passages), applied after a seeded shuffle. Great for small test runs.
- `--seed S` — reproducible topic/cluster selection (dedicated seeded RNG).
- Pipeline selectors: `--discrete-only` / `--cars-only` / `--science-passage-only`.
  With none, all three run.
- `--topic-ids ID1,ID2` — **targeted science-passage test**: build ONE passage from
  exactly those topic_ids (in order) and stop. Bypasses clustering/shuffle/checkpoints
  (always fresh), overrides `--max-topics`. Composes with `--science-passage-only`,
  `--run-name`, `--config`. Fails fast naming any unknown id.
- `--verbose` / `-v` — debug logging.

## Utility modes (no API calls)

- `--stats` — checkpoint + output counts for all three pipelines.
- `--render-figures` — render figures from `science_passages.jsonl` into
  `runs/<name>/figures/`, stamp `image_path` back into the JSONL, write
  `figures_manifest.json`. Add `--force-render` to re-render existing images. See
  [figures.md](figures.md).
- `--recount` — rebuild the **discrete** checkpoint from its questions JSONL: reopen
  under-quota topics, keep completed ones, never modify the questions file. Use after
  manual edits or to resume a partially-counted run.
- `--reset` — clear ALL checkpoints (interactive yes/no prompt).

## Run artifacts (`runs/<name>/`)

- `discrete_questions.jsonl`, `science_passages.jsonl`, `cars_passages.jsonl` — banks.
- `generation_metrics.json` (discrete), `science_generation_metrics.json`,
  `cars_generation_metrics.json` — funnel + per-model token/cost (`MetricsTracker`).
- `rejected_*.jsonl` — discarded attempts, **diagnostic only** (never affects
  accept/reject).
- `checkpoints/{discrete,cars,science_passage}/` — resume state.
- `figures/` + `figures_manifest.json` — after `--render-figures`.

## Loading into Supabase

Two separate steps with **different credentials** — read the script docstrings.

### 1. Load the banks → Postgres tables (`scripts/load_to_supabase.py`)

```
.venv\Scripts\python.exe scripts/load_to_supabase.py --dry-run   # parse + report, no write
.venv\Scripts\python.exe scripts/load_to_supabase.py             # upsert into Postgres
```

- Reads the three JSONL banks + `figures_manifest.json` from `runs/beta_bank_v1/`
  (paths are constants near the top of the script — edit them to load a different
  run). Note it reads `science_passages.filtered.jsonl` for science.
- **Needs only `SUPABASE_DB_URL`** — the Postgres connection string (service-role /
  `postgres` user, which bypasses RLS for ETL). Requires `psycopg` (v3).
- **Additive / re-runnable**: every write is `INSERT ... ON CONFLICT (id) DO UPDATE`,
  so loading a fuller bank over a sample produces no duplicates. Single
  all-or-nothing transaction. Order: passages → questions → figures (FK-safe).
- Populates `image_path` but **does NOT upload figure PNGs** (`image_url` left NULL).

> **Known harmless note:** when connecting through the Supabase **transaction**
> pooler, psycopg's server-side prepared statements can collide and surface a
> `prepared statement "..." already exists` error. It's a pgbouncer/pooler artifact,
> not a data problem — re-run, or use the session pooler / direct connection
> (port 5432) instead of the transaction pooler.

### 2. Upload figure PNGs → Storage (`scripts/upload_figures.py`)

```
.venv\Scripts\python.exe scripts/upload_figures.py --dry-run
.venv\Scripts\python.exe scripts/upload_figures.py
.venv\Scripts\python.exe scripts/upload_figures.py --bucket-only --figures-dir <dir>
```

Uploads rendered PNGs to a public `figures` Storage bucket and sets
`figures.image_url`. **Needs `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`** (service
role, not anon) **and `SUPABASE_DB_URL`** (to know which figures were actually
loaded). Idempotent (`x-upsert`). The figures **table** is authoritative for what to
upload.

`--bucket-only` pushes **every** PNG in `--figures-dir` to the bucket with **no DB
I/O** (no rows read, no `image_url` set) — for staging figures *before* their rows /
passages are loaded, so the later load never produces broken figure refs. Needs only
`SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`. Run a normal (non-bucket-only) pass
after the load to set `image_url`.

## Helper scripts (`scripts/`)

Read-only inspectors and one-off fixers — most read from `runs/beta_bank_v1/`:
- `show_passage.py` / `show_cars.py` / `show_fig_passage.py` — pretty-print a passage.
- `check_answer_distribution.py` — A/B/C/D balance.
- `check_blind_solve_confidence.py` — blind-solve confidence breakdown.
- `check_latex_json_repair.py` — exercise the JSON/LaTeX repair path.
- `recost.py` — recompute cost from metrics.
- `fix_textmu.py` — rewrite KaTeX-incompatible `\textmu` in a bank (report /
  `--write-fixed` / `--apply`); see [conventions.md](conventions.md).
