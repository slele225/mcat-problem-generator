# Figure pipeline (science passages only)

Figures are **only** for the science-passage pipeline, and **only** when
`science_passage.enable_figures: true` (default `false`). When off, the generation
prompts don't offer figures, any stray spec is dropped, and `rdkit`/`matplotlib`
are never imported. When on, those deps are required and checked at startup
(`ensure_figure_deps`).

Two declarative figure types only — `smiles` (RDKit) and `plot` (matplotlib). There
is no free-form / model-generated rendering code. Adding a new type means editing
`src/schemas.py` (the spec) + `src/figures.py` (validate/serialize/render), not a
prompt.

## Where figures come from: the dedicated figure pass

`src/pipelines/figure_pass.py` (`run_figure_pass`). **Figure decisions are NOT made
in the passage or question prompts** — do not move that logic back into a prompt; it
was buried there before and a markdown table always won, so figures were never
produced.

Flow (in `_generate_passage_set_once`, `science_passage.py:744`):
1. Passage prose + optional `table_markdown` are generated and reviewed.
2. **Figure pass** runs on that *final* passage: one LLM call decides whether any
   result/entity should be a plot or a structure (vs. a table), emits `FigureSpec`s,
   and **reconciles the table** (rows moved into a plot are removed; the table may
   become `null`). Best-effort — any failure leaves passage+table unchanged and
   never crashes the run. Invalid specs are dropped per-spec, original table kept.
3. Questions are generated *after*, so they see the final exhibit set.

The figure-pass `model` is an **injected parameter** (currently `config.model`, i.e.
Opus). It's built swappable for a future A/B (e.g. a cheaper model for the figure
decision alone) — pass a different `model` to `run_figure_pass`, no caller refactor.
Token budget: figure pass uses `max_tokens=1536`.

## REQUIRED_FIGURES backstop

Some content categories essentially require a figure on the real MCAT, but prompting
alone reliably fails to produce one. `science_passage.py:REQUIRED_FIGURES` enforces a
type **structurally** (deterministic, content-category code based — never text
sniffing):

| category code | required type | why |
|---|---|---|
| `5B`, `5D` | `smiles` | covalent structure / stereochem (5B), biological molecules (5D) |
| `5A`, `5E` | `plot` | titration curves / solutions (5A), thermo & kinetics data (5E) |

If a required type is absent from **both** the passage and **every** question, the
whole passage set is regenerated (up to `max_retries`) with an escalating directive
injected into passage generation (`generate_science_passage_set`). After retries
still missing → log ERROR and accept anyway (so the miss rate is measurable). `1A`
is deliberately not enforced (mixed content; 5E already catches enzyme kinetics).

## Validation sees TEXT, never the image

The adversarial-review and blind-solve models **never see the rendered image**. They
see `figures.figure_to_text(spec)` — a faithful serialization derived from the *same*
spec the image renders from (so it can't drift from what the student sees):
- `smiles` → each SMILES string + RDKit molecular formula.
- `plot` → chart type, axes, and the explicit series numbers as a markdown table.

## Rendering (`--render-figures`)

Rendering is a **separate pass** (`python -m src.main --render-figures`), not part of
generation. `render_jsonl_figures` reads `science_passages.jsonl`, renders each
passage- and question-level figure to `figures/{figure_id}.{fmt}`, stamps a relative
`image_path` back onto each figure, rewrites the JSONL atomically, and writes
`figures_manifest.json`. Idempotent: existing images are skipped unless
`--force-render`.

### Rendering conventions (hard-won)

- **Monochrome by design.** Plots and structures render in austere grayscale/B&W to
  match real MCAT figures. Series are distinguished by hatch/shade (bars),
  linestyle+marker (lines), or marker shape (scatter) — never matplotlib's rainbow
  color cycle. SMILES atom labels are drawn in black (`useBWAtomPalette`), not the
  default CPK palette (red O, blue N). All per-series patterns are chosen
  deterministically by series index, so re-rendering a spec is byte-reproducible.
- **Ionic / disconnected → formula card, not a structure.** In `_render_smiles`, if
  any molecule is a salt/hydrate/ion pair (multiple fragments via
  `Chem.GetMolFrags`) or contains a metal atom, RDKit can only draw scattered
  "fragment soup" — so a clean monochrome **FORMULA card** is rendered instead (e.g.
  `CuSO₄·5H₂O`, `NaCl`). Single connected covalent molecules (incl. zwitterions)
  still draw as a normal 2D structure. This is a **render-time fallback, not a new
  figure type**. The figure-pass prompt also tells the model not to emit structure
  figures for ionic compounds/salts/hydrates in the first place.
- **Bar/histogram axis handling.** Bar charts widen the figure with the category
  count (capped at 14 in), slim the bars (fractions of the category slot, not
  matplotlib's fat 0.8 default), and wrap/rotate long x-tick labels (~30°) so they
  never collide. Single-series plots get no legend; multi-series get a frameless
  legend placed *outside* the axes (saved with `bbox_inches="tight"`).

If you change rendering style, keep determinism: same spec must render the same
image (no `Math.random`, no global matplotlib state mutation — styling is scoped via
`plt.rc_context(_plot_rc())`).
