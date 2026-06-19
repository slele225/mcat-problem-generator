# Conventions & gotchas

The non-obvious stuff. The code is the source of truth; this captures things you
can't infer from a single file.

## LaTeX must be KaTeX-compatible

All math/chemistry notation in any output field (passage prose, table cells, stems,
choices, explanations) is written as LaTeX delimited with `$...$` (inline) or
`$$...$$` (genuine display only). The rule lives once in
`src/prompts/common.py:LATEX_NOTATION_RULE` and is imported by **discrete and science
only** (CARS is prose — no math, no LaTeX rule).

**The renderer downstream is KaTeX.** Anything KaTeX doesn't support renders broken.
The known landmine:
- **`\textmu` / `\textmicro` are NOT supported by KaTeX.** For the micro symbol use
  `\mu` in math mode with units in `\text{}` — e.g. `$\mu\text{m/s}$`, `$\mu\text{M}$`,
  `$\mu\text{g}$`. **Never** `\textmu` / `\textmicro`. The generation prompt forbids
  them; `scripts/fix_textmu.py` retroactively rewrites any that slipped into an
  already-generated bank (`\text{\textmu m/s}` → `\mu\text{m/s}`).

JSON escaping: because output is a JSON object, every LaTeX backslash inside a string
must be a **double** backslash (`"$K_\\text{a}$"`) to stay valid JSON. The parser has
a repair net for when the model emits single backslashes anyway — see
[JSON parsing](#json-parsing--repair).

## Explanation bolding is GENERATION-side

When an explanation refers to an answer choice, the model writes the **bolded markdown
phrase directly into the explanation text** — e.g. `**Choice B** incorrectly
attributes…`. The consuming app no longer does render-side regex bolding.

Why: a render-time regex mis-bolded bare letters (the `C` in a `C–Br` bond, the `A`
in "Series A"). Doing it at generation time — where the model knows what's a choice
vs. a chemical symbol or experimental label — and bolding the **whole phrase
"Choice X"** (never a bare letter) keeps it unambiguous. Rule:
`src/prompts/common.py:CHOICE_REFERENCE_BOLD_RULE`.

## Phantom "Option E"

Models trained on 5-option tests persistently emit a 5th choice and reference
"Option E" in explanations. Two layers handle it:
1. `normalize_choices` strips a stray `E` key from the **choices object**.
2. **Detect-and-reject** (implemented — `schemas.py:explanation_has_phantom_option`,
   wired into all three pipelines): if the **explanation prose** references a phantom
   option, the whole attempt is rejected and a retry regenerates. It never edits the
   explanation (editing risks incoherence). It catches both letter-anchored
   references ("Option E", "(E)", "E is incorrect") and letter-less descriptive ones
   ("the remaining option", "the duplicate option") — conservatively scoped so real
   A–D discussion, Roman numerals, `E. coli`, and physics variables don't trip it.

Prompts forbid E references as prevention (`NO_FIFTH_OPTION_*` in `common.py` for
discrete/science; CARS keeps its own inline copy by design).

## JSON parsing / repair

`src/llm_client.py:parse_json_response` is the shared parser. Order: strict
`json.loads` first (well-formed responses untouched), then repair fallbacks —
markdown-fence strip, **balanced-brace extraction** (`_extract_balanced_object`,
string-literal aware so a `}` inside a value doesn't close early), and **LaTeX
backslash repair** (`_repair_json_escapes`, doubles stray single backslashes so
`$\Delta$` etc. survive). A one-element JSON **array** wrapping the object is
unwrapped. Genuinely truncated JSON **fails cleanly** (`ValueError`) — no fabrication.

This is why generation prompts end with `OUTPUT FORMAT — ABSOLUTE: respond with ONLY
the JSON object`: models love to emit a prose preamble ("Let me…") that breaks
parsing. The balanced-brace extraction is the safety net, not a license to drop that
instruction.

## Opus 4.7+ rejects sampling params

`temperature` (and `top_p`/`top_k`) are **deprecated on Opus 4.7+** and return a 400.
`llm_client` omits `temperature` up front for those models (`_model_rejects_sampling_params`)
and learns-and-strips at runtime as a backstop. Practical effect: with `opus.yaml`,
`temperature_generate`/`temperature_validate` only actually apply to the **Sonnet
blind-solve checker** — Opus generation/review ignore them. Don't "fix" a config by
expecting temperature to change Opus output.

## Windows / PowerShell

This repo runs on Windows.

- **Use `.venv\Scripts\python.exe`, not the system `python`.** System Python lacks the
  deps (anthropic, rdkit, matplotlib, psycopg). The system interpreter here is
  Python 3.14 with nothing installed; the project venv has everything.
- **`mcat_topics.json` has a UTF-8 BOM** — read it with `encoding="utf-8-sig"` in
  ad-hoc scripts (the pipeline already handles this).
- **PowerShell ≠ bash**: `$null` not `/dev/null`, `$env:VAR` not `$VAR`,
  `Get-Content` not `cat`, no `&&` chaining in Windows PowerShell 5.1 (use `;` +
  `if ($?)`). The `uv run` invocations work the same in both shells.

## Hard prompt rules (don't regress)

These come straight from painful regressions — see CLAUDE.md "Hard-won rules":
accuracy-first ordering (the `#1 PRIORITY — SCIENTIFIC ACCURACY` block stays above
formatting), no duplicated rules, field-scoped competing instructions (stem cap vs.
"thorough but focused" explanation), and one-change-at-a-time prompt edits.
