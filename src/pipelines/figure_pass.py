"""Dedicated figure-generation stage for science passages (Phase 1.1).

WHY THIS IS ITS OWN STAGE
-------------------------
Figures were essentially never produced when the figure decision was buried in
the overloaded passage-generation prompt — a markdown table always won. This
stage isolates that decision: it runs AFTER the passage (prose + optional table)
is generated and reviewed, and BEFORE questions are generated, so questions are
written against the FINAL set of exhibits.

WHAT IT DOES
------------
Given the generated passage_text + table_markdown (and section/category/topic
context), one LLM call decides whether any result or entity would, on the REAL
MCAT, be shown as a PLOT or a chemical STRUCTURE rather than (or in addition to)
a markdown table:
  * Quantitative results across a CONTINUOUS variable (pH, concentration, time,
    temperature, …) -> PREFER a plot (line/scatter/bar).
  * A specific named molecule, functional group, reaction substrate/product, or
    structural comparison -> a SMILES structure.
  * A purely conceptual passage with no quantitative results and no specific
    molecule -> "no figure" (the passage/table are left unchanged).
It emits the figure(s) in the EXISTING FigureSpec format and a RECONCILED table
(rows/data moved into a plot are removed; the table may become null when a plot
fully replaces it). Produced specs are validated (SMILES parse / plot series
alignment); an invalid spec is dropped and the original table kept as a fallback.

SWAPPABILITY (later phases plug in here; do NOT bake this back into the passage
prompt):
  * MODEL — `model` is injected. The pipeline passes config.model (Opus) today; a
    later phase can A/B a different model (e.g. Fable 5) for THIS stage alone by
    passing a different `model`, with no caller refactor.
  * RENDERER / VALIDATOR — figure specs flow through the schema (RawFigureSpec /
    FigureSpec) and src/figures.py (validate_figure_spec, figures_to_text,
    render_*). New DECLARATIVE figure types (e.g. reaction-SMILES) are added
    there + in the schema; this stage only needs its prompt taught to emit them.

OUT OF SCOPE (designed-for, NOT implemented here): reaction-SMILES / multi-step
reaction schemes, and any model-generated rendering CODE. Declarative plot/smiles
specs only.

This stage is BEST-EFFORT: any failure (generation, parse, validation) leaves the
passage and table unchanged and never crashes the run.
"""

import logging
from typing import Optional

from ..schemas import RawFigureSpec, FigureSpec
from ..figures import validate_figure_spec, figures_to_text, passage_figure_id

logger = logging.getLogger(__name__)

# Table cell sentinels the model may emit to mean "no table" (mirrors
# RawSciencePassage._empty_table_to_none).
_EMPTY_TABLE_VALUES = {"", "none", "n/a", "null"}


# --- Prompt -----------------------------------------------------------------

# The two declarative render types and their exact JSON shapes. Kept LOCAL to
# this stage (not shared with the passage prompt) so the figure decision lives
# here. New declarative types are added here + in the schema + src/figures.py.
_FIGURE_PASS_SCHEMA = """\
- Plot: {"figure_type": "plot", "caption": "short caption", "alt_text": "short text description", \
"plot": {"chart_type": "bar"|"line"|"scatter"|"histogram", "title": "...", "x_label": "...", \
"y_label": "...", "series": [{"name": "Group A", "x": [...], "y": [...]}]}}  (x and y MUST be \
equal length; provide the ACTUAL numbers from the passage/table)
- Chemical structure: {"figure_type": "smiles", "caption": "short caption", "alt_text": "short \
text description", "smiles": {"molecules": [{"smiles": "CCO", "label": "ethanol"}]}}  (one or \
more molecules; every SMILES must be valid and parseable)"""


def figure_pass_prompt(
    section: str,
    content_category: str,
    topic_group: str,
    topics: list[dict],
    passage_text: str,
    table_markdown: Optional[str],
    existing_figures_text: str = "",
    extra_instruction: str = "",
) -> list[dict]:
    """Build the figure-pass prompt for one already-generated passage.

    Inputs are the FINAL passage prose, its optional markdown table, any figures
    the passage already carries (as text, so they are preserved/reconciled), and
    the topic context. `extra_instruction`, when set, is appended to the user
    turn (used by the structural figure-enforcement backstop to demand a missing
    required figure type).
    """
    topic_lines = []
    for t in topics:
        line = f"- {t.get('topic', '?')}"
        subs = t.get("subtopics") or []
        if subs:
            line += f" (subtopics: {', '.join(subs)})"
        topic_lines.append(line)
    topics_str = "\n".join(topic_lines)

    table_block = (
        f"\n\nCURRENT TABLE (table_markdown):\n{table_markdown}"
        if table_markdown
        else "\n\n(No table currently.)"
    )
    existing_block = (
        f"\n\nFIGURE(S) ALREADY ON THE PASSAGE (preserve these unless you replace them):\n"
        f"{existing_figures_text}"
        if existing_figures_text
        else ""
    )

    system = f"""You are an MCAT exhibit designer for the AAMC. A science passage for the \
{section} section has already been written (prose + an optional markdown table). Your ONLY job \
is to decide how its results/entities should be SHOWN: as a PLOT, as a chemical STRUCTURE, or \
left as a markdown table — exactly as the REAL MCAT would present them. You do NOT rewrite the \
passage and you do NOT write questions.

DECIDE, then output figures (if any) plus a reconciled table:
- A PLOT is the natural MCAT representation when the passage reports quantitative results \
across a CONTINUOUS variable (pH, concentration, substrate/dose, time, temperature, voltage, \
etc.) or compares a measured quantity across experimental groups. PREFER a plot over a table \
for such data — that is how the real MCAT shows it. Use chart_type line/scatter for a \
continuous x, bar for discrete groups.
- A chemical STRUCTURE (SMILES) is required when the passage centers on a SPECIFIC named \
molecule, functional group, reaction substrate/product, or a structural comparison — draw it \
rather than only naming it in prose. Only create a structure figure for a single COVALENT/ORGANIC \
molecule whose 2D structure aids the question; do NOT draw ionic compounds, simple salts, or \
hydrates (e.g. CuSO4·5H2O, NaCl) — refer to those by formula/name in the passage/question text \
instead.
- If the passage is CONCEPTUAL with no quantitative results and no specific molecule (e.g. gas \
laws, thermodynamics in the abstract, general theory), NO figure is warranted: return an empty \
"figures" array and leave the table unchanged. Do NOT force a figure onto such a passage.

You may use ONLY these two figure types — provide the COMPLETE underlying data so each can be \
rendered deterministically and validated as text:
{_FIGURE_PASS_SCHEMA}

RECONCILE THE TABLE:
- If you move data from the table INTO a plot, REMOVE those rows/columns from "table_markdown" \
(return the trimmed table). If a plot FULLY replaces the table, set "table_markdown" to null.
- If you add a figure but a residual table is still useful, return that residual table.
- If you add NO figure, return "table_markdown" unchanged (echo the current table, or null if \
there was none).
- Never reference a figure that you do not include, and never leave table data that a figure \
you added has fully absorbed.

NOTES:
- Keep figure titles, axis labels, and legend names as PLAIN readable text (e.g. "Substrate \
concentration (mM)") — do NOT use LaTeX in figure fields; the renderer typesets them directly.
- All numbers must be internally consistent with the passage/table and plausible at the \
introductory college level.

Respond with ONLY a JSON object in this exact shape, no other text:
{{
  "reasoning": "<one or two sentences: what (if anything) becomes a figure and why>",
  "figures": [ <zero or more figure specs using ONLY the two shapes above> ],
  "table_markdown": "<the reconciled markdown table, or null>"
}}"""

    user = f"""Section: {section}
Content Category: {content_category}
Topic Group: {topic_group}
Topics this passage covers:
{topics_str}

PASSAGE PROSE:
{passage_text}{table_block}{existing_block}

Decide whether any result or entity here should be a PLOT or a chemical STRUCTURE on the real \
MCAT. Output the figure spec(s) (or an empty array if none is warranted) and the reconciled \
table_markdown."""

    if extra_instruction:
        user = f"{user}\n\n{extra_instruction}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# --- Orchestration ----------------------------------------------------------

def _reconcile_table(raw: dict, original: Optional[str]) -> Optional[str]:
    """Resolve the reconciled table_markdown the figure pass returned.

    Key absent          -> keep the ORIGINAL table (model didn't touch it).
    null / empty token   -> table fully replaced by a figure (None).
    a markdown string    -> the model's reconciled (trimmed) table.
    """
    if "table_markdown" not in raw:
        return original
    v = raw.get("table_markdown")
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in _EMPTY_TABLE_VALUES:
        return None
    return v


async def run_figure_pass(
    client,
    cluster: dict,
    passage_id: str,
    passage_text: str,
    table_markdown: Optional[str],
    existing_figures: list,
    config,
    metrics,
    *,
    model: Optional[str] = None,
    extra_instruction: str = "",
) -> dict:
    """Run the dedicated figure pass for one passage. BEST-EFFORT, never raises.

    Returns a dict:
      {
        "figures":       list[FigureSpec],   # final passage-level figures
        "table_markdown": Optional[str],     # reconciled (or original on fallback)
        "figures_text":  str,                # text serialization of "figures"
        "changed":       bool,               # True iff this pass produced figures
      }

    When the model declines (no figure warranted) or every produced spec fails
    validation, the passage's existing figures and the original table are
    returned unchanged (`changed=False`). `model` is injected (defaults to
    config.model) so a later phase can A/B a different model for THIS stage only.
    """
    model = model or config.model
    existing_text = figures_to_text(existing_figures)
    unchanged = {
        "figures": existing_figures,
        "table_markdown": table_markdown,
        "figures_text": existing_text,
        "changed": False,
    }

    msgs = figure_pass_prompt(
        cluster["section"],
        cluster["content_category"],
        cluster["topic_group"],
        cluster["topics"],
        passage_text,
        table_markdown,
        existing_figures_text=existing_text,
        extra_instruction=extra_instruction,
    )

    try:
        raw = await client.generate_json(
            msgs,
            temperature=config.science_passage.temperature_generate,
            max_tokens=1536,
            model=model,
            on_usage=lambda m, i, o: metrics.record_usage("figure_pass", m, i, o),
        )
    except Exception as e:
        logger.warning(f"figure pass: generation failed for {passage_id}: {e}")
        return unchanged

    produced = raw.get("figures") or []
    if not produced:
        logger.debug(f"figure pass {passage_id}: model judged no figure warranted")
        return unchanged

    # Validate each spec individually so ONE bad spec is dropped without losing
    # the others (unlike the all-or-nothing generation-time check).
    specs: list[FigureSpec] = []
    any_dropped = False
    for fig in produced:
        try:
            rspec = RawFigureSpec(**fig)
        except Exception as e:
            any_dropped = True
            logger.info(f"figure pass DROP {passage_id}: invalid spec ({e})")
            continue
        ok, err = validate_figure_spec(rspec)
        if not ok:
            any_dropped = True
            logger.info(f"figure pass DROP {passage_id}: {err}")
            continue
        specs.append(
            FigureSpec(
                figure_id=passage_figure_id(passage_id, len(specs) + 1),
                **rspec.model_dump(),
            )
        )

    if not specs:
        # Nothing valid survived — keep the table fallback, change nothing.
        logger.info(f"figure pass {passage_id}: no valid figure produced; passage unchanged")
        return unchanged

    # Trust the model's reconciled table ONLY when every spec was valid; if any
    # spec was dropped, keep the ORIGINAL table so no data a dropped figure was
    # meant to carry is lost.
    final_table = table_markdown if any_dropped else _reconcile_table(raw, table_markdown)

    types = ", ".join(s.figure_type for s in specs)
    note = "; some specs dropped, kept original table" if any_dropped else ""
    logger.info(f"figure pass {passage_id}: produced {len(specs)} figure(s) [{types}]{note}")

    return {
        "figures": specs,
        "table_markdown": final_table,
        "figures_text": figures_to_text(specs),
        "changed": True,
    }
