"""Figure validation, text serialization, and batch rendering.

Two figure types are supported (see src/schemas.py FigureSpec):
  - "smiles": one or more chemical structures given as SMILES, rendered with RDKit.
  - "plot":   a bar/line/scatter/histogram rendered with matplotlib from explicit
              series data.

Design (see the project plan):
  * VALIDATION VISIBILITY: the adversarial-review and blind-solve models NEVER see
    the rendered image. They see `figure_to_text(spec)` — the SMILES string(s)
    (+ RDKit-derived formula) for structures, and the chart type/axes/explicit
    series numbers (as a markdown table) for plots. The text is derived from the
    same spec the image is rendered from, so it cannot drift from what the
    student sees.
  * GENERATION-TIME validation (`validate_figure_spec`) only parses/those-checks
    a spec (SMILES must parse under RDKit; plot data must be consistent). Actual
    image drawing is deferred to a batch pass (`render_jsonl_figures`).

rdkit / matplotlib are imported LAZILY inside functions so that `import
figures` (and the rest of the pipeline) works even when the optional figure
dependencies are not installed. They are only required when figures are actually
validated or rendered.
"""

import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Optional, Union

from .schemas import FigureSpec, RawFigureSpec

logger = logging.getLogger(__name__)

# A duck-typed figure: either RawFigureSpec (generation) or FigureSpec (stored).
AnyFigure = Union[FigureSpec, RawFigureSpec]


def ensure_figure_deps() -> None:
    """Raise a clear error if the optional figure dependencies are missing.

    Called at startup when figures are enabled and at the top of the render pass.
    """
    missing = []
    try:
        import rdkit  # noqa: F401
    except Exception:
        missing.append("rdkit")
    try:
        import matplotlib  # noqa: F401
    except Exception:
        missing.append("matplotlib")
    if missing:
        raise RuntimeError(
            f"Figure support requires {', '.join(missing)}. "
            f"Install with: uv pip install {' '.join(missing)}"
        )


# --- Figure id helpers (assigned by the pipeline, stable before rendering) ---

def passage_figure_id(passage_id: str, n: int) -> str:
    return f"{passage_id}_pf{n:02d}"


def question_figure_id(question_id: str, n: int) -> str:
    return f"{question_id}_qf{n:02d}"


# --- Generation-time validation ---

def validate_figure_spec(spec: AnyFigure) -> tuple[bool, str]:
    """Validate a figure spec's semantics (beyond the structural Pydantic checks).

    For SMILES: every SMILES string must parse under RDKit.
    For plots:  series must be non-empty and contain finite numeric y values
                (length alignment is already enforced by the schema).

    Returns (ok, error_message). A failing figure should be treated like a
    generation parse failure upstream (triggers regeneration).
    """
    if spec.figure_type == "smiles":
        try:
            from rdkit import Chem
        except Exception as e:
            return False, f"rdkit unavailable for SMILES validation: {e}"
        if spec.smiles is None or not spec.smiles.molecules:
            return False, "smiles figure has no molecules"
        for i, mol in enumerate(spec.smiles.molecules):
            m = Chem.MolFromSmiles(mol.smiles)
            if m is None:
                return False, f"SMILES #{i+1} did not parse: {mol.smiles!r}"
        return True, ""

    if spec.figure_type == "plot":
        if spec.plot is None or not spec.plot.series:
            return False, "plot figure has no series"
        import math
        for s in spec.plot.series:
            if not s.y:
                return False, "plot series has no y values"
            for val in s.y:
                if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                    return False, f"plot series {s.name!r} has non-finite y value"
        return True, ""

    return False, f"unknown figure_type {spec.figure_type!r}"


def validate_figures(figures: list[AnyFigure]) -> tuple[bool, list[str]]:
    """Validate a list of figure specs. Returns (all_ok, [errors])."""
    errors = []
    for fig in figures:
        ok, err = validate_figure_spec(fig)
        if not ok:
            errors.append(err)
    return (not errors), errors


# --- Validation-time text serialization (what the review/solve models see) ---

def _smiles_formula(smiles: str) -> Optional[str]:
    """RDKit molecular formula for a SMILES, or None if unavailable/unparseable."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except Exception:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(m)
    except Exception:
        return None


def _markdown_table(headers: list[str], rows: list[list]) -> str:
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def _plot_to_text(spec: AnyFigure, fid: str) -> str:
    p = spec.plot
    lines = [f"[Figure {fid} — {p.chart_type} chart]"]
    if p.title:
        lines.append(f"Title: {p.title}")
    axis_bits = []
    if p.x_label:
        axis_bits.append(f"X axis: {p.x_label}")
    if p.y_label:
        axis_bits.append(f"Y axis: {p.y_label}")
    if axis_bits:
        lines.append("; ".join(axis_bits))
    if spec.caption:
        lines.append(f"Caption: {spec.caption}")

    # Combine into one table when every series shares identical x values.
    xs = [list(map(str, s.x)) for s in p.series]
    same_x = len(xs) > 0 and all(x == xs[0] for x in xs)

    if same_x and len(p.series) >= 1:
        x_header = p.x_label or "x"
        headers = [x_header] + [s.name or f"series {i+1}" for i, s in enumerate(p.series)]
        rows = []
        for r in range(len(p.series[0].x)):
            row = [p.series[0].x[r]] + [s.y[r] for s in p.series]
            rows.append(row)
        lines.append("Data:")
        lines.append(_markdown_table(headers, rows))
    else:
        for i, s in enumerate(p.series):
            name = s.name or f"series {i+1}"
            lines.append(f'Series "{name}":')
            headers = [p.x_label or "x", p.y_label or "y"]
            rows = [[s.x[r], s.y[r]] for r in range(len(s.x))]
            lines.append(_markdown_table(headers, rows))
    return "\n".join(lines)


def _smiles_to_text(spec: AnyFigure, fid: str) -> str:
    lines = [f"[Figure {fid} — chemical structure]"]
    if spec.caption:
        lines.append(f"Caption: {spec.caption}")
    for i, mol in enumerate(spec.smiles.molecules, start=1):
        formula = _smiles_formula(mol.smiles)
        label = f' (label "{mol.label}")' if mol.label else ""
        formula_str = f"  [formula {formula}]" if formula else ""
        lines.append(f"Molecule {i}{label}: SMILES {mol.smiles}{formula_str}")
    if spec.alt_text:
        lines.append(f"Description: {spec.alt_text}")
    return "\n".join(lines)


def figure_to_text(spec: AnyFigure) -> str:
    """Faithful TEXT serialization of a figure spec for the validation models.

    SMILES strings (+ RDKit formula) for structures; chart type, axes, and the
    explicit series numbers (as a markdown table) for plots. The student sees the
    rendered image; the models reason over this text.
    """
    fid = getattr(spec, "figure_id", None) or "figure"
    if spec.figure_type == "smiles" and spec.smiles is not None:
        return _smiles_to_text(spec, fid)
    if spec.figure_type == "plot" and spec.plot is not None:
        return _plot_to_text(spec, fid)
    return f"[Figure {fid} — unrenderable spec]"


def figures_to_text(figures: list[AnyFigure]) -> str:
    """Serialize a list of figures into one text block (empty string if none)."""
    if not figures:
        return ""
    return "\n\n".join(figure_to_text(f) for f in figures)


# --- Plot styling: austere grayscale, publication-clean (MCAT-like) ----------
#
# Real MCAT figures are black-and-white: data series are distinguished by
# hatch/shade (bars), linestyle + marker (lines), or marker shape (scatter) —
# never by matplotlib's default rainbow color cycle. This section centralizes
# that style so every chart type shares ONE look, and the per-series pattern is
# chosen DETERMINISTICALLY by series index so re-rendering a spec is reproducible.
#
# These are plain-data constants (no matplotlib import) so importing this module
# stays dependency-free; matplotlib is still imported lazily inside _render_plot.

# Optional single accent color (theme red). NOT applied by default — everything
# renders grayscale. It exists as one constant so a LATER change can emphasize
# exactly ONE series/element by swapping a grayscale entry for ACCENT (and later
# wire it to the data-theme system). Nothing auto-applies it today.
ACCENT = "#A23B3B"

# Core grayscale ink + backgrounds.
_INK = "#000000"          # spines, ticks, edges, text
_GRID = "#E6E6E6"         # very light y-gridline
_BG = "#FFFFFF"           # figure + axes background

# Bar/area fills cycled by series index, paired with a hatch so that 3+ series
# stay unambiguous in PURE grayscale (shade alone is not enough). The ramp favors
# LIGHT fills (white -> light gray -> mid gray) for an austere exam line-art look
# — the real MCAT uses light/outlined bars, not heavy solid black. Solid black is
# kept only as the LAST entry (never the default first series) because hatching is
# invisible on it. 6 distinct fill+hatch combos before repeating: white-plain,
# light-gray-diagonal, white-crosshatch, light-gray-dots, mid-gray-backslash,
# black-solid.
_GRAY_FILLS = ["#FFFFFF", "#D9D9D9", "#FFFFFF", "#BFBFBF", "#808080", "#000000"]
_HATCHES = [None, "///", "xxx", "...", "\\\\\\", None]

# Line series: distinguish by LINESTYLE + MARKER (not color); all near-black.
_LINESTYLES = ["-", "--", ":", "-.", "-", "--"]
_LINE_MARKERS = ["o", "s", "^", "D", "v", "x"]
_LINE_COLORS = ["#000000", "#000000", "#000000", "#000000", "#333333", "#333333"]

# Scatter series: distinguish by MARKER SHAPE, edges always black, grayscale fill.
_SCATTER_MARKERS = ["o", "s", "^", "x", "D", "v"]
_SCATTER_FILLS = ["#000000", "#FFFFFF", "#808080", "#000000", "#BFBFBF", "#FFFFFF"]


def _plot_rc() -> dict:
    """rcParams for the austere MCAT look — clean sans font, no rainbow, white bg.

    Returned as a dict so the whole render runs inside a scoped `plt.rc_context`
    and never mutates global matplotlib state. DejaVu Sans ships with matplotlib,
    so no new dependency is introduced.
    """
    return {
        "figure.facecolor": _BG,
        "axes.facecolor": _BG,
        "savefig.facecolor": _BG,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 11,
        # Modest title — real MCAT figure titles are understated, not large/bold.
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "text.color": _INK,
        "axes.labelcolor": _INK,
        "axes.edgecolor": _INK,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.color": _INK,
        "ytick.color": _INK,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "hatch.linewidth": 0.6,
    }


def _cyc(seq: list, i: int):
    """Deterministic cycle: the i-th series gets seq[i % len(seq)]."""
    return seq[i % len(seq)]


# Category-label thresholds (auto, no per-figure tuning): wrap each label to short
# lines, and rotate when labels are long or numerous so they never collide.
_LABEL_WRAP = 14          # max chars per wrapped line
_LABEL_ROTATE_LEN = 12    # any label longer than this -> rotate
_LABEL_ROTATE_COUNT = 6   # more categories than this -> rotate


def _should_rotate(labels: list) -> bool:
    """Whether x-tick labels need ~30° rotation to avoid colliding.

    True when there are many categories OR any single label is long. Centralized
    so the tick-setter and the figure-width calc agree on the same decision
    (their spacing math must match).
    """
    return (
        len(labels) > _LABEL_ROTATE_COUNT
        or any(len(str(lbl)) > _LABEL_ROTATE_LEN for lbl in labels)
    )


def _apply_category_ticks(ax, base: list, raw_labels: list) -> None:
    """Set x-tick labels that won't overlap, robust to long generated names.

    WRAPS each label at spaces to short lines (<= _LABEL_WRAP chars) and ROTATES
    ~30° (right-anchored) when any label is long or there are many categories;
    short, few labels stay horizontal for a cleaner look. Deterministic — same
    labels always produce the same ticks (reproducible renders). The caller widens
    the figure with the category count (see _plot_figsize) and tight_layout then
    expands the bottom margin so rotated labels aren't clipped.
    """
    labels = [str(x) for x in raw_labels]
    wrapped = [
        textwrap.fill(lbl, width=_LABEL_WRAP, break_long_words=False) for lbl in labels
    ]
    ax.set_xticks(base)
    if _should_rotate(labels):
        ax.set_xticklabels(
            wrapped, rotation=30, ha="right", rotation_mode="anchor"
        )
    else:
        ax.set_xticklabels(wrapped, ha="center")
    # Dense axes backstop: when there are very many categories (so the figure has
    # hit its width cap), shave the tick font a little so neighbours stay clear —
    # still >= 8pt, readable. Width scaling carries the normal case; this only
    # bites for unusually long category lists.
    n = len(labels)
    if n > 12:
        ax.tick_params(axis="x", labelsize=8)
    elif n > 8:
        ax.tick_params(axis="x", labelsize=9)


def _finalize_axes(ax, p) -> bool:
    """Apply the shared publication-clean treatment to a finished axes.

    Removes the top/right spines (the matplotlib "box" tell), keeps thin black
    left/bottom spines, sets a single very-light y-only gridline behind the data,
    and titles/labels at modest weight. Shared by every chart type so styling
    never drifts between branches.

    LEGEND: suppressed for SINGLE-series plots (redundant with the axis label and
    prone to floating over the data); for MULTI-series plots a frameless legend is
    placed OUTSIDE the axes (upper-left of a right-side margin) so it never
    overlaps the data. Returns True iff such an outside legend was drawn, so the
    caller can save with bbox_inches="tight" to avoid clipping it.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK)
        ax.spines[side].set_linewidth(0.8)

    # Minimal gridlines: very light gray, y-axis only, behind the data.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.6)
    ax.xaxis.grid(False)

    if p.title:
        ax.set_title(p.title)
    if p.x_label:
        ax.set_xlabel(p.x_label)
    if p.y_label:
        ax.set_ylabel(p.y_label)

    labels = [lbl for lbl in ax.get_legend_handles_labels()[1] if lbl]
    if len(labels) >= 2:
        # Multi-series: frameless legend just outside the upper-right of the axes.
        ax.legend(
            frameon=False, loc="upper left",
            bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
        )
        return True
    # Single-series (or unlabeled): no legend — the axis labels carry the meaning.
    return False


# --- Render-time rasterization ---

# --- Ionic / disconnected structure handling (formula-card fallback) ---------
#
# RDKit lays out only a CONNECTED COVALENT molecule cleanly. A salt, hydrate, or
# ion pair (multiple '.'-separated fragments) or any metal-containing species
# renders as scattered, overlapping "fragment soup" with no structural meaning —
# and chemically there is no single 2D structure for it (chemists just write the
# formula). For those we draw a clean monochrome FORMULA card instead of the soup.

# Atoms that count as covalent/organic for "is this a single drawable structure".
# Anything OUTSIDE this set is treated as a metal -> formula card, not a scattered
# metal-complex/salt layout. Metalloids commonly drawn covalently (B, Si, As, Te)
# are included; noble gases are included so they never trip the check.
_NONMETALS = {
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br",
    "Te", "I", "At", "He", "Ne", "Ar", "Kr", "Xe", "Rn",
}

_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

_ELEMENT_SYMBOLS: Optional[set] = None


def _element_symbols() -> set:
    """All real element symbols (memoized), for the formula-string sniffer."""
    global _ELEMENT_SYMBOLS
    if _ELEMENT_SYMBOLS is None:
        try:
            from rdkit.Chem import GetPeriodicTable
            pt = GetPeriodicTable()
            syms = set()
            for z in range(1, 119):
                try:
                    syms.add(pt.GetElementSymbol(z))
                except Exception:
                    continue
            _ELEMENT_SYMBOLS = syms
        except Exception:
            _ELEMENT_SYMBOLS = set()
    return _ELEMENT_SYMBOLS


def _mol_is_drawable_structure(mol) -> bool:
    """True iff `mol` is a SINGLE connected covalent molecule RDKit draws cleanly.

    False for disconnected/ionic species (salts, hydrates, ion pairs — more than
    one fragment via Chem.GetMolFrags) and for any molecule containing a metal
    atom; those have no meaningful single 2D structure and are shown as a formula
    instead. A zwitterion (opposite charges within ONE fragment) stays drawable.
    """
    from rdkit import Chem
    if mol is None:
        return False
    if len(Chem.GetMolFrags(mol)) > 1:
        return False
    return all(a.GetSymbol() in _NONMETALS for a in mol.GetAtoms())


def _looks_like_formula(s: str) -> bool:
    """Whether `s` already reads as a chemical formula (e.g. "CuSO4·5H2O", "NaCl").

    A formula has NO spaces and decomposes entirely into element symbols, digits,
    and formula punctuation (·, parentheses, charges). Used to prefer a clean,
    conventionally-written label over an RDKit-derived (Hill-ordered) formula.
    """
    s = (s or "").strip()
    if not s or any(ch.isspace() for ch in s):
        return False
    syms = _element_symbols()
    if not syms:
        return False
    i, n_elements, has_digit_or_dot = 0, 0, False
    while i < len(s):
        c = s[i]
        if c.isalpha():
            two = s[i:i + 2]
            if len(two) == 2 and two[0].isupper() and two[1].islower() and two in syms:
                n_elements += 1
                i += 2
                continue
            if c.isupper() and c in syms:
                n_elements += 1
                i += 1
                continue
            return False
        if c.isdigit():
            has_digit_or_dot = True
            i += 1
            continue
        if c in "·.*()[]+-":
            if c in "·.*":
                has_digit_or_dot = True
            i += 1
            continue
        return False
    # Require a real decomposition; a lone symbol with no digit (e.g. "H") is too
    # ambiguous to treat as an intentional formula.
    return n_elements >= 1 and (n_elements >= 2 or has_digit_or_dot)


def _strip_formula_charge(f: str) -> str:
    """Drop the trailing charge CalcMolFormula appends (e.g. 'Cu+2' -> 'Cu')."""
    return re.sub(r"[+-]\d*$", "", f)


def _rdkit_species_formula(mol) -> Optional[str]:
    """Conventional-ish formula for a (possibly disconnected) species via RDKit.

    Per-fragment Hill formula, charges stripped. Singleton fragments (the salt's
    own ions) are concatenated directly; fragments that REPEAT (hydrate waters,
    coordinated solvent) are grouped as "·n<formula>" — e.g. NaCl's SMILES yields
    "NaCl" and CuSO4·5H2O's yields "CuO4S·5H2O". Returns None if nothing usable.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        return None
    counts: dict[str, int] = {}
    order: list[str] = []
    for fr in frags:
        try:
            f = _strip_formula_charge(rdMolDescriptors.CalcMolFormula(fr))
        except Exception:
            return None
        if not f:
            continue
        if f not in counts:
            counts[f] = 0
            order.append(f)
        counts[f] += 1
    if not order:
        return None
    base = "".join(f for f in order if counts[f] == 1)
    parts = ([base] if base else []) + [
        f"{counts[f]}{f}" for f in order if counts[f] > 1
    ]
    return "·".join(parts)


def _format_formula(formula: str) -> str:
    """Subscript the formula digits for display (CuSO4·5H2O -> CuSO₄·5H₂O).

    A digit run is subscripted only when it follows an element/closing paren (a
    stoichiometric count); a leading coefficient (e.g. the 5 in ·5H2O) stays full
    size. Middots and other punctuation pass through unchanged.
    """
    out, i, prev_atomish = [], 0, False
    while i < len(formula):
        c = formula[i]
        if c.isdigit():
            j = i
            while j < len(formula) and formula[j].isdigit():
                j += 1
            run = formula[i:j]
            out.append(run.translate(_SUBSCRIPTS) if prev_atomish else run)
            prev_atomish = False
            i = j
        else:
            out.append(c)
            prev_atomish = c.isalpha() or c == ")"
            i += 1
    return "".join(out)


def _molecule_formula_entry(mol_spec, rdmol) -> tuple:
    """(display_formula, name) for one molecule on a formula card.

    Big-formula priority: (1) the molecule's own label when it already reads as a
    formula; (2) an RDKit-derived per-fragment formula; (3) the label text as a
    last resort. `name` is a human-readable sub-line (the label) shown under the
    formula — None when the label IS the formula (the card caption supplies the
    name instead).
    """
    label = (mol_spec.label or "").strip()
    if _looks_like_formula(label):
        return _format_formula(label), None
    derived = _rdkit_species_formula(rdmol)
    if derived:
        return _format_formula(derived), (label or None)
    return (label or "structure"), None


def _render_formula_card(out_path: Path, entries: list, caption: str = "") -> None:
    """Render a clean monochrome 'here is the compound' FORMULA label PNG.

    Used instead of an RDKit structure for disconnected/ionic species. `entries` is
    a list of (display_formula, name) tuples; `caption` is the figure caption, used
    as the descriptive name line for a single-molecule card. Same white background /
    black ink / DejaVu Sans as the plots; square (350x350 px) to match the size of a
    single rendered structure, so it reads as an intentional label, not a broken
    diagram.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(_plot_rc()):
        fig = plt.figure(figsize=(3.5, 3.5), dpi=100)  # 350x350 px
        fig.patch.set_facecolor(_BG)
        ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        n = max(1, len(entries))
        longest = max((len(f) for f, _ in entries), default=4)
        size = 30
        if longest > 10:
            size = 24
        if longest > 16:
            size = 19
        if n >= 2:
            size = min(size, 20)
        if n >= 3:
            size = min(size, 15)

        slot = 1.0 / n
        for idx, (formula, name) in enumerate(entries):
            cy = 1.0 - slot * (idx + 0.5)
            show_name = bool(name and name.strip())
            ax.text(
                0.5, cy + (0.07 if show_name else 0.0), formula,
                ha="center", va="center", fontsize=size, color=_INK,
            )
            if show_name:
                ax.text(
                    0.5, cy - 0.11, name.strip(),
                    ha="center", va="center", fontsize=10, color=_INK,
                )

        # Single-molecule card: use the figure caption as a descriptive name line
        # when it adds something beyond the formula itself.
        if n == 1 and caption and caption.strip():
            formula, name = entries[0]
            if not (name and name.strip()) and caption.strip() != formula:
                ax.text(
                    0.5, 0.18, caption.strip(),
                    ha="center", va="center", fontsize=10, color=_INK, wrap=True,
                )

        fig.savefig(str(out_path), dpi=100, facecolor=_BG)
        plt.close(fig)


def _render_smiles(spec: FigureSpec, out_path: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    parsed = []
    for mol in spec.smiles.molecules:
        m = Chem.MolFromSmiles(mol.smiles)
        if m is None:
            raise ValueError(f"SMILES did not parse at render time: {mol.smiles!r}")
        parsed.append((mol, m))

    # IONIC / DISCONNECTED BACKSTOP: if ANY molecule is a salt/hydrate/ion pair
    # (multiple fragments) or contains a metal, RDKit can only draw scattered
    # "fragment soup". Render a clean FORMULA card instead (Part A of the fix).
    if not all(_mol_is_drawable_structure(m) for _, m in parsed):
        entries = [_molecule_formula_entry(mol, m) for mol, m in parsed]
        caption = (spec.caption or "").strip()
        _render_formula_card(
            out_path, entries, caption=caption if len(parsed) == 1 else "",
        )
        return

    mols = [m for _, m in parsed]
    legends = [mol.label or "" for mol, _ in parsed]

    # Austere monochrome: every atom label drawn in BLACK (not the default CPK
    # palette — red O, blue N, etc.) to match the grayscale plots. Bonds are
    # already black by default; useBWAtomPalette only recolors atom labels. This
    # is the ONLY change vs. RDKit defaults — size/DPI/background/fonts are
    # untouched. options= for the single-mol path, drawOptions= for the grid.
    bw_opts = rdMolDraw2D.MolDrawOptions()
    bw_opts.useBWAtomPalette()

    if len(mols) == 1:
        img = Draw.MolToImage(mols[0], size=(350, 350), legend=legends[0], options=bw_opts)
    else:
        per_row = min(len(mols), 3)
        img = Draw.MolsToGridImage(
            mols, molsPerRow=per_row, subImgSize=(300, 300), legends=legends,
            drawOptions=bw_opts,
        )
    img.save(str(out_path))


# Bar widths as a FRACTION of the unit category slot. Matplotlib's default 0.8
# renders fat, heavy-looking bars; these slimmer fractions give clean,
# publication/exam-style whitespace. Both single- and multi-series widths derive
# from these two tunables, so the look is easy to adjust in one place.
_BAR_SINGLE_FRAC = 0.55   # single-series: one slim bar, this fraction of the slot
_BAR_GROUP_FRAC = 0.7     # multi-series: TOTAL group width (of the slot);
                          # per-series width = _BAR_GROUP_FRAC / n_series


def _draw_bars(ax, p) -> None:
    """Grouped bars distinguished by grayscale fill + hatch (not color).

    Bar widths are a FRACTION of the unit category slot rather than matplotlib's
    fat 0.8 default. A single series renders one slim bar (_BAR_SINGLE_FRAC of the
    slot) centered on its tick. Multiple series render a group whose TOTAL width is
    _BAR_GROUP_FRAC of the slot (each series = _BAR_GROUP_FRAC / n), centered on the
    tick: bars sit side by side within the group, and because the group spans only
    _BAR_GROUP_FRAC (< 1) of the slot there is always whitespace before the next
    group's bars. Widths are slot-relative, so this stays robust no matter how many
    categories there are (single bars never go hairline-thin; groups never overlap).
    Deterministic — same spec renders byte-identical bars.
    """
    n = len(p.series)
    n_cat = len(p.series[0].x)
    # Guard the earlier shape-mismatch crash: every series must span the same
    # number of categories as the first (each series' own x/y lengths are already
    # matched by the schema, but two series can still differ from each other). A
    # ragged series would make ax.bar raise deep inside matplotlib; fail fast with
    # a clear message so the batch renderer drops just THIS figure (keeping the
    # table fallback) instead of crashing the whole render pass.
    for s in p.series:
        if len(s.x) != n_cat or len(s.y) != n_cat:
            raise ValueError(
                f"bar/histogram series {s.name or '?'!r} has {len(s.y)} value(s) "
                f"but the chart has {n_cat} categories; skipping figure"
            )
    base = list(range(n_cat))
    width = _BAR_SINGLE_FRAC if n <= 1 else _BAR_GROUP_FRAC / n
    for i, s in enumerate(p.series):
        offsets = [b + (i - (n - 1) / 2) * width for b in base]
        yerr = s.y_err if s.y_err else None
        ax.bar(
            offsets, s.y, width=width, label=(s.name or None), yerr=yerr,
            color=_cyc(_GRAY_FILLS, i), hatch=_cyc(_HATCHES, i),
            edgecolor=_INK, linewidth=0.9,
            error_kw={"ecolor": _INK, "elinewidth": 0.8, "capthick": 0.8},
            capsize=3,
        )
    _apply_category_ticks(ax, base, p.series[0].x)


def _draw_lines(ax, p) -> None:
    """Line series distinguished by linestyle + marker shape, all near-black."""
    for i, s in enumerate(p.series):
        color = _cyc(_LINE_COLORS, i)
        common = dict(
            color=color, linestyle=_cyc(_LINESTYLES, i), linewidth=1.4,
            marker=_cyc(_LINE_MARKERS, i), markersize=5,
            markerfacecolor=_BG, markeredgecolor=color, markeredgewidth=1.0,
            label=(s.name or None),
        )
        yerr = s.y_err if s.y_err else None
        if yerr:
            ax.errorbar(
                s.x, s.y, yerr=yerr, capsize=3,
                ecolor=color, elinewidth=0.8, capthick=0.8, **common,
            )
        else:
            ax.plot(s.x, s.y, **common)


def _draw_scatter(ax, p) -> None:
    """Scatter series distinguished by marker shape, black edges, grayscale fill."""
    for i, s in enumerate(p.series):
        ax.scatter(
            s.x, s.y, label=(s.name or None),
            marker=_cyc(_SCATTER_MARKERS, i), s=40,
            facecolors=_cyc(_SCATTER_FILLS, i), edgecolors=_INK, linewidths=0.9,
        )


# Figure sizing. The base canvas is 6x4 in. Bar/histogram charts WIDEN with the
# category count (and grouped-series count) so x-axis labels get horizontal room
# and never collide; width is capped so PNGs stay reasonable. Other chart types
# (line/scatter use numeric, auto-spaced ticks) keep the base width.
_PLOT_BASE_W = 6.0
_PLOT_H = 4.0
_PLOT_MAX_W = 14.0       # cap: the app scales the PNG to the passage column anyway


def _plot_figsize(p) -> tuple:
    """(width, height) inches — bar/histogram widen with the category count.

    Each category needs a horizontal slot: a single bar wants ~0.85 in, and each
    extra grouped series widens the slot so side-by-side bars + their labels stay
    legible. When labels are kept horizontal (few + short, so no rotation), the
    slot must also fit the WIDEST label, so width grows with the longest label.
    Capped at _PLOT_MAX_W. Deterministic (purely a function of the spec), so
    re-renders are reproducible. Non-bar charts keep the base 6x4 canvas.
    """
    if p.chart_type not in ("bar", "histogram") or not p.series:
        return (_PLOT_BASE_W, _PLOT_H)
    n_cat = len(p.series[0].x)
    n_series = len(p.series)
    labels = [str(x) for x in p.series[0].x]
    per_cat = 0.85 + 0.25 * max(0, n_series - 1)
    if not _should_rotate(labels):
        # Horizontal labels need room for the widest one (~0.11 in/char @10pt).
        longest = max((len(lbl) for lbl in labels), default=0)
        per_cat = max(per_cat, 0.11 * longest)
    width = max(_PLOT_BASE_W, n_cat * per_cat)
    return (min(width, _PLOT_MAX_W), _PLOT_H)


def _render_plot(spec: FigureSpec, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive, deterministic
    import matplotlib.pyplot as plt

    p = spec.plot
    # Scope ALL styling to this render via rc_context so global matplotlib state
    # is never mutated (keeps re-rendering reproducible and side-effect free).
    with plt.rc_context(_plot_rc()):
        fig, ax = plt.subplots(figsize=_plot_figsize(p))

        # histogram shares the bar treatment (gray/white fill, black edges, hatch).
        if p.chart_type in ("bar", "histogram"):
            _draw_bars(ax, p)
        elif p.chart_type == "line":
            _draw_lines(ax, p)
        elif p.chart_type == "scatter":
            _draw_scatter(ax, p)
        else:
            plt.close(fig)
            raise ValueError(f"unsupported chart_type at render time: {p.chart_type!r}")

        outside_legend = _finalize_axes(ax, p)
        fig.tight_layout()
        # bbox_inches="tight" only when a legend sits OUTSIDE the axes, so it is
        # captured rather than clipped; single-series figures keep the plain
        # tight_layout canvas. Deterministic either way (reproducible renders).
        save_kwargs = {"dpi": 150}
        if outside_legend:
            save_kwargs["bbox_inches"] = "tight"
        fig.savefig(str(out_path), **save_kwargs)
        plt.close(fig)


def render_figure(spec: FigureSpec, figures_dir: Path, fmt: str = "png") -> str:
    """Render one figure spec to an image file. Returns the absolute path string.

    The caller is responsible for assigning spec.figure_id; the filename is
    `{figure_id}.{fmt}`.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"{spec.figure_id}.{fmt}"
    if spec.figure_type == "smiles":
        _render_smiles(spec, out_path)
    elif spec.figure_type == "plot":
        _render_plot(spec, out_path)
    else:
        raise ValueError(f"unknown figure_type {spec.figure_type!r}")
    return str(out_path)


def _iter_record_figures(record: dict):
    """Yield (parent_id, figure_dict) for every figure in a passage record."""
    for fig in record.get("figures", []) or []:
        yield record.get("passage_id", "?"), fig
    for q in record.get("questions", []) or []:
        for fig in q.get("figures", []) or []:
            yield q.get("question_id", "?"), fig


def render_jsonl_figures(
    jsonl_path: str,
    output_dir: str,
    fmt: str = "png",
    force: bool = False,
) -> dict:
    """Batch-render every figure referenced in a science_passages.jsonl file.

    Reads the JSONL, renders each passage- and question-level figure to
    `{output_dir}/figures/{figure_id}.{fmt}`, stamps a relative `image_path` back
    onto each figure object, rewrites the JSONL atomically, and writes a
    `figures_manifest.json`. Idempotent: existing images are skipped unless
    `force`. Returns a report dict.
    """
    ensure_figure_deps()
    jsonl = Path(jsonl_path)
    if not jsonl.exists():
        logger.warning(f"No JSONL at {jsonl}; nothing to render.")
        return {"total": 0, "rendered": 0, "skipped": 0, "failed": 0}

    out_dir = Path(output_dir)
    figures_dir = out_dir / "figures"

    records = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    report = {"total": 0, "rendered": 0, "skipped": 0, "failed": 0}
    manifest = []

    for record in records:
        for parent_id, fig in _iter_record_figures(record):
            report["total"] += 1
            fid = fig.get("figure_id")
            status, error, rel_path = "failed", "", None
            try:
                if not fid:
                    raise ValueError("figure missing figure_id")
                rel_path = f"figures/{fid}.{fmt}"
                out_path = figures_dir / f"{fid}.{fmt}"
                if out_path.exists() and not force:
                    status = "skipped"
                    report["skipped"] += 1
                else:
                    spec = FigureSpec(**fig)
                    render_figure(spec, figures_dir, fmt=fmt)
                    status = "rendered"
                    report["rendered"] += 1
                fig["image_path"] = rel_path  # stamp back (idempotent)
            except Exception as e:
                error = str(e)
                report["failed"] += 1
                logger.warning(f"Figure render failed for {fid} ({parent_id}): {e}")
            manifest.append({
                "figure_id": fid,
                "figure_type": fig.get("figure_type"),
                "parent_id": parent_id,
                "image_path": rel_path,
                "status": status,
                "error": error,
            })

    # Rewrite the JSONL atomically (temp then replace) with image_path filled in.
    tmp = jsonl.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    tmp.replace(jsonl)

    manifest_path = out_dir / "figures_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"figures": manifest, "report": report}, f, indent=2)

    logger.info(
        f"Figure render: {report['rendered']} rendered, {report['skipped']} skipped, "
        f"{report['failed']} failed (of {report['total']}). Manifest: {manifest_path}"
    )
    return report
