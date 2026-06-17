"""Pydantic models for question/passage data and LLM response parsing."""

import logging
import re

from pydantic import BaseModel, field_validator, model_validator
from typing import Optional

logger = logging.getLogger(__name__)

REQUIRED_CHOICE_KEYS = {"A", "B", "C", "D"}


# --- Phantom-option (fifth-choice) detection ---------------------------------
#
# Models trained on 5-option tests persistently reference a nonexistent option
# in EXPLANATION prose ("Option E is wrong because...") even though the choices
# object is always exactly A-D (normalize_choices strips a stray "E" key). This
# regex catches those references so the pipeline can DETECT-AND-REJECT the whole
# question (let a retry regenerate) rather than ship an explanation that talks
# about an option the student will never see.
#
# Conservative by design (CLAUDE.md rule #5 — err toward only catching genuine
# option references):
#   * Restricted to letters E-H (not E-Z) so Roman numerals (I, V, X) and
#     incidental capitals never trip it.
#   * Every alternative is anchored on an explicit option context — "option E",
#     "(E)", "E)", or "E is/are (in)correct/wrong/right/best" — so a bare capital
#     inside a word ("Energy"), a species name ("E. coli"), or a physics variable
#     ("the field E", "activation energy E_a") does NOT match.
_PHANTOM_OPTION_RE = re.compile(
    r"(?:"
    r"(?i:option|choice|answer|distractor)s?\s+\(?[E-H]\)?"  # "option E", "choice (E)"
    r"|\([E-H]\)"                                            # "(E)"
    r"|(?<![A-Za-z0-9])[E-H]\)"                              # "E)"
    # "E is incorrect", "E are wrong", "E was the right/best (answer)"
    r"|\b[E-H]\s+(?:is|are|was|were)\s+(?:not\s+|also\s+|the\s+|a\s+|an\s+)*"
    r"(?:in)?correct\b"
    r"|\b[E-H]\s+(?:is|are|was|were)\s+(?:not\s+|also\s+|the\s+|a\s+|an\s+)*"
    r"(?:wrong|right|best)\b"
    r")"
)

# Descriptive (letter-LESS) references to a stripped fifth option. The same model
# habit that produces "Option E" also produces references with NO letter at all —
# the model emits a 5th choice, normalize_choices strips it from the choices
# object, but the explanation still talks about it as "the remaining option", "the
# duplicate option", "C and the duplicate option", or "the claim in the final
# option is incorrect". The letter-anchored regex above cannot see these (there is
# no E-H letter to anchor on), so they ship. This pattern catches a SINGULAR
# leftover/extra/duplicate option reference and is only applied to the standard
# four-choice (A-D) case (see explanation_has_phantom_option).
#
# Tightly scoped to avoid rejecting legitimate A-D discussion (CLAUDE.md rule #5):
#   * The noun is SINGULAR ("option"/"choice"/"distractor"/"answer choice"); plural
#     distractor talk ("the other three options", "the remaining three choices",
#     "options B and C") never matches.
#   * Bare "answer" is NOT a noun, so the idiom "the final answer is D" is ignored.
#   * Ordinals that name a real choice are excluded — only "fifth"/"five" is numeric
#     ("the fourth option" = D is NOT flagged).
#   * A reference immediately disambiguated by a real letter ("the remaining option,
#     D", "the final option (D)") is skipped via the trailing case-sensitive [A-D]
#     guard — that is a genuine reference to choice D, not a phantom.
_DESCRIPTIVE_PHANTOM_RE = re.compile(
    r"(?i:"
    r"(?:the|a|an|this|that)\s+"
    r"(?:remaining|duplicate|other|extra|leftover|fifth|final|last)\s+"
    r"(?:option|choice|distractor|answer\s+choice)\b"
    r"|"
    r"(?:option|choice|distractor|answer\s+choice)\s+(?:five|5)\b"
    r")"
    r"(?![\s,;:.()\-]*[A-D]\b)"
)


def explanation_has_phantom_option(text, valid_keys=frozenset("ABCD")):
    """Return the offending substring if `text` references a nonexistent option.

    Scans explanation prose for a reference to a multiple-choice option that is
    NOT among `valid_keys` (the real choices, always A-D here), in two ways:

      1. LETTER-ANCHORED: an explicit out-of-range letter ("Option E", "(E)", "E)",
         "E is incorrect", "H)").
      2. DESCRIPTIVE (letter-less): a singular reference to a leftover/duplicate
         fifth option with no letter ("C and the duplicate option", "the remaining
         option that...", "the final option is incorrect"). Only applied to the
         standard four-choice (A-D) case, since "the remaining option" is only a
         phantom when exactly A-D exist.

    Returns the matched fragment (e.g. "Option E", "the duplicate option") on the
    first genuine hit, else None. Used by all three pipelines to deterministically
    reject-and-regenerate any question whose explanation talks about a phantom
    fifth option.
    """
    if not isinstance(text, str) or not text:
        return None
    # (1) Letter-anchored references to an out-of-range option.
    for m in _PHANTOM_OPTION_RE.finditer(text):
        frag = m.group(0)
        # The option letter is the E-H char the pattern anchored on (NOT the
        # uppercase initial of a keyword like "Choice"/"Option").
        letter = next((ch for ch in frag if ch in "EFGH"), None)
        if letter and letter not in valid_keys:
            return frag
    # (2) Descriptive references to a stripped fifth option, ONLY for the standard
    # four-option case (A-D). Outside that case "the remaining option" may legitimately
    # name a real choice, so the descriptive scan does not apply.
    if set(valid_keys) == REQUIRED_CHOICE_KEYS:
        m = _DESCRIPTIVE_PHANTOM_RE.search(text)
        if m:
            return m.group(0)
    return None


def normalize_choices(v):
    """Validate and normalize an MCAT multiple-choice ``choices`` object.

    Enforces EXACTLY the keys A, B, C, D, each a non-empty string. Extra keys
    that aren't A/B/C/D (trailing artifacts the model sometimes emits, e.g. an
    empty ``"E": ""`` or a junk ``"C2": "ignore"``) are stripped. If, after
    stripping, the four required keys aren't all present and non-empty, raises
    ValueError — which fails parsing and triggers regeneration upstream.
    """
    if not isinstance(v, dict):
        raise ValueError(f"choices must be an object, got {type(v).__name__}")
    extra = [k for k in v if k not in REQUIRED_CHOICE_KEYS]
    if extra:
        logger.warning(f"Stripping unexpected choice key(s) {extra} from choices")
    cleaned = {k: v[k] for k in REQUIRED_CHOICE_KEYS if k in v}
    missing = sorted(REQUIRED_CHOICE_KEYS - set(cleaned))
    if missing:
        raise ValueError(f"choices is missing required key(s): {missing}")
    for k in ("A", "B", "C", "D"):
        val = cleaned[k]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"choice {k!r} must be a non-empty string, got {val!r}")
    return {k: cleaned[k] for k in ("A", "B", "C", "D")}


class DiscreteQuestion(BaseModel):
    """A single discrete (non-CARS) MCAT question."""
    question_id: str
    topic_id: str
    section: str
    content_category: str
    topic_group: str
    topic: str
    subtopics_tested: list[str]
    stem: str
    choices: dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: str  # "A", "B", "C", or "D"
    explanation: str
    difficulty: str  # "easy", "medium", "hard"
    skill_tested: str  # AAMC SIRS skill (Skill 1-4)
    validation: dict  # {"adversarial_pass": bool, "blind_solve_pass": bool}

    @field_validator("correct_answer")
    @classmethod
    def validate_answer(cls, v):
        if v not in ("A", "B", "C", "D"):
            raise ValueError(f"correct_answer must be A/B/C/D, got {v}")
        return v

    @field_validator("choices", mode="before")
    @classmethod
    def validate_choices(cls, v):
        return normalize_choices(v)


class CARSQuestion(BaseModel):
    """A single CARS question tied to a passage."""
    question_id: str
    skill_type: str  # "Foundations of Comprehension", "Reasoning Within the Text", "Reasoning Beyond the Text"
    stem: str
    choices: dict[str, str]
    correct_answer: str
    explanation: str
    difficulty: str = "medium"  # "easy", "medium", "hard" (matches discrete/science)

    @field_validator("correct_answer")
    @classmethod
    def validate_answer(cls, v):
        if v not in ("A", "B", "C", "D"):
            raise ValueError(f"correct_answer must be A/B/C/D, got {v}")
        return v

    @field_validator("choices", mode="before")
    @classmethod
    def validate_choices(cls, v):
        return normalize_choices(v)


class CARSPassage(BaseModel):
    """A CARS passage with its associated questions."""
    passage_id: str
    passage_text: str
    word_count: int
    subject: str
    questions: list[CARSQuestion]
    validation: dict


# --- Models for parsing raw LLM output (before validation) ---

class RawDiscreteQuestion(BaseModel):
    """Raw question as returned by the generation LLM."""
    stem: str
    choices: dict[str, str]
    correct_answer: str
    explanation: str
    difficulty: str
    subtopics_tested: list[str] = []
    skill_tested: str = ""

    @field_validator("correct_answer")
    @classmethod
    def validate_answer(cls, v):
        v = v.strip().upper()
        if v not in ("A", "B", "C", "D"):
            raise ValueError(f"correct_answer must be A/B/C/D, got {v}")
        return v

    @field_validator("choices", mode="before")
    @classmethod
    def validate_choices(cls, v):
        return normalize_choices(v)


class RawCARSPassage(BaseModel):
    """Raw passage as returned by the generation LLM."""
    passage_text: str
    subject: str


class RawCARSQuestion(BaseModel):
    """Raw CARS question as returned by the generation LLM."""
    skill_type: str
    stem: str
    choices: dict[str, str]
    correct_answer: str
    explanation: str
    difficulty: str = "medium"  # "easy", "medium", "hard"

    @field_validator("correct_answer")
    @classmethod
    def validate_answer(cls, v):
        v = v.strip().upper()
        if v not in ("A", "B", "C", "D"):
            raise ValueError(f"correct_answer must be A/B/C/D, got {v}")
        return v

    @field_validator("choices", mode="before")
    @classmethod
    def validate_choices(cls, v):
        return normalize_choices(v)


ANSWER_BASIS_VALUES = {"from_passage", "apply_knowledge", "data_interpretation"}

# Common phrasings the model emits instead of the canonical token.
_ANSWER_BASIS_ALIASES = {
    "passage": "from_passage",
    "from the passage": "from_passage",
    "answerable from passage": "from_passage",
    "outside knowledge": "apply_knowledge",
    "apply knowledge": "apply_knowledge",
    "application": "apply_knowledge",
    "data": "data_interpretation",
    "data interpretation": "data_interpretation",
}


def normalize_answer_basis(v) -> str:
    """Coerce an ``answer_basis`` label to its canonical token (lenient).

    Lowercases, swaps spaces for underscores, and maps a few common phrasings to
    the canonical token. Unknown values are returned cleaned-but-unchanged rather
    than raising, so a sloppy label never blocks parsing — the (lenient)
    answer-basis review is what actually judges correctness of the label.
    """
    if not isinstance(v, str):
        return ""
    s = v.strip().lower()
    if s in _ANSWER_BASIS_ALIASES:
        return _ANSWER_BASIS_ALIASES[s]
    s = s.replace(" ", "_").replace("-", "_")
    return s


# --- Figure specs (chemistry structures via SMILES, plots via matplotlib) ---
#
# Figures are a structured, render-deterministic representation a passage or a
# question can carry. The student sees a rendered image; the validation models
# (adversarial review + blind solve) see a TEXT serialization of this same spec
# (see src/figures.py:figure_to_text) — never the image. Structural/consistency
# validation lives here (dependency-free); semantic checks that need RDKit
# (SMILES parseability) live in src/figures.py so importing schemas never pulls
# in rdkit/matplotlib.

ALLOWED_CHART_TYPES = {"bar", "line", "scatter", "histogram"}
ALLOWED_FIGURE_TYPES = {"smiles", "plot"}


class SmilesMolecule(BaseModel):
    """One chemical structure given as a SMILES string."""
    smiles: str
    label: str = ""  # optional caption shown under this structure

    @field_validator("smiles")
    @classmethod
    def _non_empty(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("smiles must be a non-empty string")
        return v.strip()


class SmilesPayload(BaseModel):
    """One or more SMILES structures (rendered as a grid when more than one)."""
    molecules: list[SmilesMolecule]

    @field_validator("molecules")
    @classmethod
    def _at_least_one(cls, v):
        if not v:
            raise ValueError("smiles figure requires at least one molecule")
        return v


class PlotSeries(BaseModel):
    """A single data series. x is numeric (line/scatter/histogram) or categorical
    strings (bar). y is numeric. Lengths must align."""
    name: str = ""  # legend label
    x: list
    y: list[float]
    y_err: Optional[list[float]] = None  # optional error bars (extensible)

    @model_validator(mode="after")
    def _aligned(self):
        if len(self.x) != len(self.y):
            raise ValueError(
                f"series x ({len(self.x)}) and y ({len(self.y)}) must be equal length"
            )
        if self.y_err is not None and len(self.y_err) != len(self.y):
            raise ValueError("y_err length must match y length")
        return self


class PlotPayload(BaseModel):
    """Enough to render a plot deterministically with matplotlib."""
    chart_type: str
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    series: list[PlotSeries]

    @field_validator("chart_type", mode="before")
    @classmethod
    def _known_chart(cls, v):
        s = (v or "").strip().lower()
        if s not in ALLOWED_CHART_TYPES:
            raise ValueError(
                f"chart_type must be one of {sorted(ALLOWED_CHART_TYPES)}, got {v!r}"
            )
        return s

    @field_validator("series")
    @classmethod
    def _at_least_one(cls, v):
        if not v:
            raise ValueError("plot requires at least one series")
        return v


def _validate_figure_payload(self):
    """Shared model_validator: exactly the payload matching figure_type present."""
    ft = self.figure_type
    if ft not in ALLOWED_FIGURE_TYPES:
        raise ValueError(f"unknown figure_type {ft!r}; expected one of {sorted(ALLOWED_FIGURE_TYPES)}")
    if ft == "smiles":
        if self.smiles is None:
            raise ValueError("figure_type 'smiles' requires a 'smiles' payload")
        if self.plot is not None:
            raise ValueError("figure_type 'smiles' must not carry a 'plot' payload")
    elif ft == "plot":
        if self.plot is None:
            raise ValueError("figure_type 'plot' requires a 'plot' payload")
        if self.smiles is not None:
            raise ValueError("figure_type 'plot' must not carry a 'smiles' payload")
    return self


class RawFigureSpec(BaseModel):
    """Figure spec as emitted by the generation LLM (no id / image_path yet)."""
    figure_type: str
    caption: str = ""
    alt_text: str = ""
    smiles: Optional[SmilesPayload] = None
    plot: Optional[PlotPayload] = None

    @field_validator("figure_type", mode="before")
    @classmethod
    def _norm_type(cls, v):
        return (v or "").strip().lower()

    _check_payload = model_validator(mode="after")(_validate_figure_payload)


class FigureSpec(BaseModel):
    """Stored figure spec: pipeline-assigned id, render-pass-filled image_path."""
    figure_id: str
    figure_type: str
    caption: str = ""
    alt_text: str = ""
    smiles: Optional[SmilesPayload] = None
    plot: Optional[PlotPayload] = None
    image_path: Optional[str] = None  # relative to output_dir; filled by render pass

    @field_validator("figure_type", mode="before")
    @classmethod
    def _norm_type(cls, v):
        return (v or "").strip().lower()

    _check_payload = model_validator(mode="after")(_validate_figure_payload)


class ScienceQuestion(BaseModel):
    """A single passage-linked science question."""
    question_id: str
    passage_id: str
    skill_tested: str  # AAMC SIRS skill (Skill 1-4)
    answer_basis: str  # "from_passage" | "apply_knowledge" | "data_interpretation"
    stem: str
    choices: dict[str, str]
    correct_answer: str
    explanation: str
    difficulty: str  # "easy", "medium", "hard"
    figures: list[FigureSpec] = []
    validation: dict  # {"adversarial_pass", "blind_solve_pass", "answer_basis_ok"}

    @field_validator("correct_answer")
    @classmethod
    def validate_answer(cls, v):
        if v not in ("A", "B", "C", "D"):
            raise ValueError(f"correct_answer must be A/B/C/D, got {v}")
        return v

    @field_validator("choices", mode="before")
    @classmethod
    def validate_choices(cls, v):
        return normalize_choices(v)


class SciencePassage(BaseModel):
    """A science passage with its linked questions."""
    passage_id: str
    section: str
    content_category: str
    topic_ids: list[str]
    topic_group: str
    passage_text: str
    table_markdown: Optional[str] = None
    figures: list[FigureSpec] = []
    word_count: int
    questions: list[ScienceQuestion]
    validation: dict


class RawSciencePassage(BaseModel):
    """Raw science passage as returned by the generation LLM."""
    passage_text: str
    table_markdown: Optional[str] = None
    figures: list[RawFigureSpec] = []

    @field_validator("table_markdown", mode="before")
    @classmethod
    def _empty_table_to_none(cls, v):
        # The model often emits "" or "none" when no table is appropriate.
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in ("", "none", "n/a", "null"):
            return None
        return v

    @field_validator("figures", mode="before")
    @classmethod
    def _none_figures_to_list(cls, v):
        return v or []


class RawScienceQuestion(BaseModel):
    """Raw passage-linked science question as returned by the generation LLM."""
    skill_tested: str = ""
    answer_basis: str = ""
    stem: str
    choices: dict[str, str]
    correct_answer: str
    explanation: str
    difficulty: str = "medium"
    figures: list[RawFigureSpec] = []

    @field_validator("correct_answer")
    @classmethod
    def validate_answer(cls, v):
        v = v.strip().upper()
        if v not in ("A", "B", "C", "D"):
            raise ValueError(f"correct_answer must be A/B/C/D, got {v}")
        return v

    @field_validator("answer_basis", mode="before")
    @classmethod
    def _normalize_basis(cls, v):
        return normalize_answer_basis(v)

    @field_validator("choices", mode="before")
    @classmethod
    def validate_choices(cls, v):
        return normalize_choices(v)

    @field_validator("figures", mode="before")
    @classmethod
    def _none_figures_to_list(cls, v):
        return v or []


def _normalize_issue_list(v):
    """Accept issues as strings OR objects; normalize each to a string.

    Some reviewer responses return issues as objects (e.g. {"issue": "..."}
    or {"category": "...", "issue": "..."}) instead of plain strings.
    Previously that raised a validation error and the whole review was
    recorded as a FAILED review even when the question was fine. Coercing to
    strings makes the schema tolerant of that formatting choice.
    """
    if v is None:
        return []
    if isinstance(v, (str, dict)):
        v = [v]
    if not isinstance(v, list):
        return [str(v)]
    out = []
    for item in v:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            text = None
            for key in ("issue", "description", "text", "problem", "message", "detail"):
                if item.get(key):
                    text = item[key]
                    break
            if text is None:
                text = "; ".join(f"{k}: {val}" for k, val in item.items()) or str(item)
            out.append(str(text))
        else:
            out.append(str(item))
    return out


class AdversarialReview(BaseModel):
    """Result of adversarial review."""
    passed: bool
    issues: list[str] = []
    reasoning: str = ""

    @field_validator("issues", mode="before")
    @classmethod
    def _normalize_issues(cls, v):
        return _normalize_issue_list(v)


class BlindSolveResult(BaseModel):
    """Result of blind solve attempt."""
    chosen_answer: str
    confidence: str  # "high", "medium", "low"
    reasoning: str = ""


class ScienceAdversarialReview(BaseModel):
    """Adversarial review for a passage-linked science question.

    Extends the standard review with a SEPARATE, lenient verdict on whether the
    question's ``answer_basis`` label is honest. ``answer_basis_ok`` defaults to
    True so that a reviewer who omits the field never trips the check — it must
    be explicitly set False (only for a CLEAR mislabel) to reject on that basis.
    """
    passed: bool
    issues: list[str] = []
    reasoning: str = ""
    answer_basis_ok: bool = True
    answer_basis_note: str = ""

    @field_validator("issues", mode="before")
    @classmethod
    def _normalize_issues(cls, v):
        return _normalize_issue_list(v)
