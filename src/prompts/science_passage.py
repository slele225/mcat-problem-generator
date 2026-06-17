"""Prompt templates for passage-based MCAT science question generation.

The three MCAT science sections (Bio/Biochem, Chem/Phys, Psych/Soc) are mostly
PASSAGE-based: a ~200-350 word science passage (typically describing an
experiment or study) followed by 4-7 linked questions. Unlike CARS, science
passages are NOT self-contained — questions deliberately mix:

  - from_passage          answerable directly from the passage text/table
  - apply_knowledge       require the student's own intro-college science
                          knowledge applied to the passage scenario
  - data_interpretation   require reading/interpreting data presented in the
                          passage (prose results or a markdown table)

These passages are the natural home for SIRS Skills 2 (reasoning), 3 (research
design), and 4 (data interpretation), with some Skill 1. The per-passage skill
mix is therefore weighted toward 2/3/4 (see SciencePassageConfig.skill_weights),
NOT the discrete 35/45/10/10 split.

This module reuses the discrete pipeline's prompt structure (SIRS skill
definitions, AAMC-style examples, strict question-writing rules, JSON output)
adapted for passage-based items. NO graph/figure images: passages may use prose
and markdown tables ONLY.
"""

import random

from .common import (
    NO_FIFTH_OPTION_RULE,
    NO_FIFTH_OPTION_EXPLANATION_RULE,
    NO_FIFTH_OPTION_CONTRACT,
    CHOICE_REFERENCE_BOLD_RULE,
    LATEX_NOTATION_RULE,
    LATEX_REVIEW_NOTE,
)


SKILL_LABELS = {
    "skill_1": "Skill 1: Knowledge of Scientific Concepts and Principles",
    "skill_2": "Skill 2: Scientific Reasoning and Problem-Solving",
    "skill_3": "Skill 3: Reasoning About the Design and Execution of Research",
    "skill_4": "Skill 4: Data-Based and Statistical Reasoning",
}

# Default per-passage skill mix, weighted toward 2/3/4 with some 1. The pipeline
# passes the configured weights in; this is only the fallback.
DEFAULT_SKILL_WEIGHTS = {
    "skill_1": 15,
    "skill_2": 35,
    "skill_3": 25,
    "skill_4": 25,
}

DIFFICULTY_WEIGHTS = {
    "easy": 20,
    "medium": 50,
    "hard": 30,
}

ANSWER_BASIS_LABELS = {
    "from_passage": (
        "Answerable directly from the passage. A careful reader can find or "
        "infer the answer from the passage text/table without outside knowledge "
        "beyond basic scientific literacy."
    ),
    "apply_knowledge": (
        "Requires the student to APPLY their own introductory college-level "
        "science knowledge to the passage scenario. The passage alone is not "
        "sufficient; the student must bring in a concept, principle, or formula "
        "not stated in the passage."
    ),
    "data_interpretation": (
        "Requires interpreting DATA presented in the passage (a trend in the "
        "results, a value or comparison in the table, a relationship between "
        "variables). The student must reason FROM the data shown."
    ),
}

# The CONCRETE writing requirement for each answer_basis — only the ONE assigned
# to a question is injected (prominently) into the generation prompt. This is the
# meat that used to be listed for all three at once; keying it to the assigned
# basis keeps the prompt focused (the other two are mentioned only in one short
# context line). Text is preserved verbatim from the former three-bullet block.
ANSWER_BASIS_REQUIREMENTS = {
    "from_passage": (
        "Make it answerable from the passage alone (no outside knowledge beyond basic "
        "scientific literacy), BUT require INFERENCE, not retrieval: the student must "
        "SYNTHESIZE at least two separate statements from the passage, OR draw a light "
        "inference that combines/reasons over passage content. It must NOT be answerable by "
        "locating and quoting a single sentence. BAN pure-lookup stems such as \"According to "
        "the passage, X is ___\" or \"The passage states that ___\"; instead require the "
        "student to connect or reason across passage content while still needing no outside "
        "knowledge."
    ),
    "apply_knowledge": (
        "Require the student's OWN introductory college-level science knowledge that is NOT "
        "present anywhere in the passage — it must not be answerable from the passage text or "
        "its exhibits alone."
    ),
    "data_interpretation": (
        "Require reading a specific value, comparison, or trend directly FROM the markdown "
        "table or a figure. The passage prose deliberately does NOT state or summarize any "
        "result, so this question must NOT be answerable from any sentence of the prose — the "
        "student must extract and interpret the data from the exhibit itself."
    ),
}

# One-line gloss of each basis, used ONLY to give brief context about the two
# bases NOT in force for a question (so the assigned one keeps the spotlight).
ANSWER_BASIS_BRIEF = {
    "from_passage": "answerable by reasoning/synthesizing within the passage (no outside knowledge)",
    "apply_knowledge": "needs the student's own outside intro-science knowledge",
    "data_interpretation": "hinges on reading a value/trend from the exhibit (table/figure)",
}


# LaTeX notation/review rules now live in prompts.common (LATEX_NOTATION_RULE,
# LATEX_REVIEW_NOTE) and are imported above, so the discrete and science question
# prompts share one source of truth. Text is unchanged from Phase 1.1.


# --- Figure instructions (only injected when figures are enabled) ---

# Appended to a generation system prompt to let the model OPTIONALLY emit a
# figure spec. The student sees a rendered image; the spec must carry the
# complete underlying data so it can be validated as text and rendered
# deterministically.
FIGURE_GEN_INSTRUCTIONS = """\
FIGURES (optional): If a chemical structure or a data plot would genuinely strengthen this \
item, you MAY specify it in the "figures" array. Each figure is rendered to an IMAGE the \
student sees, so you MUST provide the complete underlying data.
- Chemical structure: {"figure_type": "smiles", "caption": "...", "alt_text": "short text \
description", "smiles": {"molecules": [{"smiles": "CCO", "label": "ethanol"}]}}  (one or more \
molecules; SMILES must be valid and parseable)
- Plot: {"figure_type": "plot", "caption": "...", "alt_text": "short text description", \
"plot": {"chart_type": "bar"|"line"|"scatter"|"histogram", "title": "...", "x_label": "...", \
"y_label": "...", "series": [{"name": "Group A", "x": [...], "y": [...]}]}}  (x and y must be \
equal length; provide the actual numbers)
RULES: Do NOT reference or describe a figure you have not specified (no "as shown below", \
"the figure", "the structure", "the graph" without a matching spec in "figures"). Conversely, \
do not specify a figure the item never uses. If no figure is needed, use an empty array \
("figures": []). Prefer a markdown table over a plot when a table conveys the data just as \
well.

WHEN A SMILES FIGURE IS REQUIRED: When the passage or a question centers on a SPECIFIC \
molecule, functional group, reaction substrate/product, or structural comparison, you MUST \
include that molecule as a SMILES figure rather than only naming it in prose. Examples that \
REQUIRE a SMILES figure: a question asking about the product of a named reaction, identifying \
a functional group on a given structure, comparing two molecules' structures, or \
stereochemistry. Do NOT include a SMILES figure for purely conceptual topics (e.g. gas laws, \
thermodynamics, atomic structure) where no specific molecule is depicted.

WHEN A PLOT FIGURE IS REQUIRED: When the passage reports quantitative experimental results \
across conditions, time points, concentrations, or groups, present those results as a PLOT \
figure (bar/line/scatter) when a graph is the natural representation on the real MCAT — for \
example: reaction rate vs substrate concentration (line/scatter), a measured quantity across \
experimental groups (bar), or a time course (line). Provide the actual data values in the \
plot spec. You may ALSO keep a small data table if helpful. Do NOT invent a plot for topics \
with no quantitative results.

For topics where neither applies (e.g. conceptual topics like gas laws or atomic nucleus), a \
text-only passage with no figures is correct — do not force a figure."""

# Replaces the strict "no figures" bullet in the passage generator when enabled.
_PASSAGE_FIGURE_RULE_ENABLED = (
    "- You MAY include chemical structures (SMILES) or plots via the \"figures\" array "
    "(see FIGURES below). You may also use a markdown table. Do NOT describe a figure you "
    "have not specified."
)
_PASSAGE_FIGURE_RULE_DISABLED = (
    "- Prose and an OPTIONAL markdown TABLE only. You may NOT rely on any figure, graph, "
    "diagram, micrograph, chemical-structure image, or plot. Do NOT write a passage that "
    "would require the student to see a figure that cannot be expressed as text or a "
    "markdown table. If you would normally show results as a graph, instead either describe "
    "the trend in prose or present the numbers as a markdown table."
)


def _weighted_choice(weights: dict) -> str:
    keys = list(weights.keys())
    vals = list(weights.values())
    return random.choices(keys, weights=vals, k=1)[0]


def _pick_difficulty() -> str:
    return _weighted_choice(DIFFICULTY_WEIGHTS)


def _pick_answer() -> str:
    """Random target correct-answer position to keep the key balanced A/B/C/D."""
    return random.choice(["A", "B", "C", "D"])


def pick_num_questions(q_range: list) -> int:
    """Uniformly pick a question count within the inclusive [min, max] range."""
    lo, hi = int(q_range[0]), int(q_range[1])
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def build_question_plan(n: int, skill_weights: dict, has_exhibit: bool) -> list[dict]:
    """Build a per-question plan: skill, answer_basis, difficulty, target answer.

    Guarantees a MIX of answer bases across the passage: it seeds the first
    slots with one of each basis and fills the rest with weighted picks. Skill is
    drawn from the (config) weights but nudged to stay coherent with the assigned
    basis.

    REBALANCE (Phase 1.1): now that passages are built around rich exhibits
    (tables/figures), the fill leans toward data_interpretation so it is at least
    as common as from_passage, and at least one data_interpretation slot is
    GUARANTEED whenever the passage carries an exhibit (`has_exhibit` = table OR
    figure). This shifts the overall skill mix away from the old Skill-1-heavy
    split toward Skill 4 (data reasoning), while the Skill 3 guarantee below is
    preserved from Phase 1.
    """
    skill_weights = skill_weights or DEFAULT_SKILL_WEIGHTS

    # Seed one of each basis so every passage mixes them (data_interpretation is
    # always seeded — even without a table, prose-withheld results live in an
    # exhibit the data question reads from).
    seed_bases = ["from_passage", "apply_knowledge", "data_interpretation"]
    random.shuffle(seed_bases)

    # Fill weights lean toward data_interpretation so Skill 4 is at least as
    # common as from_passage (was 35/40/25, which kept the mix Skill-1-heavy).
    # NOTE: from_passage->Skill 1 and data_interpretation->Skill 4 are the natural
    # maps, so requiring data_interpretation >= from_passage necessarily lifts
    # Skill 4 to roughly match/exceed Skill 1 — the two cannot both rank exactly
    # as 30/.../25 with Skill 1 on top. These weights land a de-skewed mix
    # (≈ Skill1 21 / Skill2 29 / Skill3 19 / Skill4 30) — no longer Skill-1-heavy,
    # with Skill 4 well-represented — rather than the broken ≈41/.../15 split.
    basis_weights = {"from_passage": 30, "apply_knowledge": 35, "data_interpretation": 35}

    bases: list[str] = []
    for i in range(n):
        if i < len(seed_bases):
            bases.append(seed_bases[i])
        else:
            bases.append(_weighted_choice(basis_weights))

    def skill_for(basis: str) -> str:
        # Keep skill coherent with basis, but still honor the weighted mix:
        # mostly map basis->natural skill, occasionally fall back to a weighted pick.
        natural = {
            "from_passage": "skill_1",
            "apply_knowledge": "skill_2",
            "data_interpretation": "skill_4",
        }[basis]
        # ~30% of the time, draw from the configured weights instead so Skill 3
        # (research design) and the rest of the mix appear too.
        if random.random() < 0.30:
            return _weighted_choice(skill_weights)
        return natural

    plan = []
    for basis in bases:
        skill_key = skill_for(basis)
        plan.append({
            "skill_key": skill_key,
            "skill_label": SKILL_LABELS[skill_key],
            "answer_basis": basis,
            "difficulty": _pick_difficulty(),
            "target_answer": _pick_answer(),
        })

    # GUARANTEE at least one data_interpretation (Skill 4) slot when the passage
    # carries an exhibit to read — exhibit-rich passages must exercise data
    # reasoning. Seeding already provides one; this is a robust backstop (e.g. if
    # n < the number of seed bases trimmed it). Convert a from_passage slot when
    # available (closest in spirit), else the first slot; its natural skill is
    # set to Skill 4. Runs BEFORE the Skill 3 guarantee so the two don't collide.
    if has_exhibit and not any(p["answer_basis"] == "data_interpretation" for p in plan):
        idx = next(
            (i for i, p in enumerate(plan) if p["answer_basis"] == "from_passage"),
            0,
        )
        plan[idx]["answer_basis"] = "data_interpretation"
        plan[idx]["skill_key"] = "skill_4"
        plan[idx]["skill_label"] = SKILL_LABELS["skill_4"]

    # GUARANTEE at least one Skill 3 (research-design) slot for passages with >= 4
    # questions — real science passages reliably support one, and the weighted
    # draw alone leaves Skill 3 underrepresented (~6%). Analogous to how one of
    # each answer_basis is seeded above: if the draw produced no skill_3 slot,
    # convert one. Prefer a slot whose basis pairs naturally with research design
    # (from_passage / apply_knowledge over data_interpretation); fall back to the
    # first slot. Basis/difficulty/target_answer are left untouched.
    if n >= 4 and not any(p["skill_key"] == "skill_3" for p in plan):
        idx = next(
            (i for i, p in enumerate(plan) if p["answer_basis"] != "data_interpretation"),
            0,
        )
        plan[idx]["skill_key"] = "skill_3"
        plan[idx]["skill_label"] = SKILL_LABELS["skill_3"]

    return plan


# --- SIRS skill guidance, adapted for passage-linked items ---

_SKILL_GUIDANCE = {
    "skill_1": """\
This question tests Skill 1: Knowledge of Scientific Concepts and Principles.
Ask the student to recognize/identify a concept, principle, or relationship as
it appears in the passage scenario, classify something described in the passage,
or apply a given equation. Anchor it in the passage — do not write a bare
definition question divorced from the passage context.""",

    "skill_2": """\
This question tests Skill 2: Scientific Reasoning and Problem-Solving.
Ask the student to use a theory/model to explain a passage observation or
predict an outcome, evaluate a causal claim about the study, combine the passage
with their own knowledge to draw a conclusion, identify a finding that would
challenge the study's interpretation, or carry out a multi-step calculation
grounded in the passage.""",

    "skill_3": """\
This question tests Skill 3: Reasoning About the Design and Execution of Research.
Use the experiment/study DESCRIBED IN THE PASSAGE. Ask the student to identify
the independent/dependent/confounding variable, evaluate the appropriateness of
a method or control, identify a design limitation or threat to validity,
distinguish correlation from causation, or reason about what additional control
is needed. Refer to the passage's actual design — do not invent a new study.""",

    "skill_4": """\
This question tests Skill 4: Data-Based and Statistical Reasoning.
Use the DATA in the passage (the prose results or the markdown table). Ask the
student to interpret a trend, compare values across groups/conditions, use a
measure of central tendency or dispersion, reason about error/significance, or
identify which conclusion the data do (or do not) support. The student must
reason FROM the passage's data, not from memorized facts.""",
}


def passage_generation_prompt(
    section: str,
    content_category: str,
    topic_group: str,
    topics: list[dict],
    word_min: int,
    word_max: int,
    enable_figures: bool = False,
) -> list[dict]:
    """Build the prompt for generating a single science passage for a cluster."""

    topic_lines = []
    for t in topics:
        line = f"- {t.get('topic', '?')}"
        subs = t.get("subtopics") or []
        if subs:
            line += f" (subtopics: {', '.join(subs)})"
        topic_lines.append(line)
    topics_str = "\n".join(topic_lines)

    figure_rule = (
        _PASSAGE_FIGURE_RULE_ENABLED if enable_figures else _PASSAGE_FIGURE_RULE_DISABLED
    )
    figure_section = f"\n\n{FIGURE_GEN_INSTRUCTIONS}" if enable_figures else ""
    figures_json_line = ',\n  "figures": []  // optional; see FIGURES above' if enable_figures else ""
    figures_user_note = (
        " You may add a SMILES structure or a plot via the figures array if it genuinely "
        "helps; otherwise leave figures empty."
        if enable_figures
        else " Remember: prose and a markdown table ONLY — nothing that needs a figure or diagram."
    )

    # REALISM RULES: real AAMC science passages withhold their conclusions — the
    # prose carries background/methods/conditions, and ALL results live in
    # exhibits (table/figure) for the reader to interpret. These constraints push
    # the generator toward that, and toward enough design detail for a Skill 3
    # (research-design) question.
    if enable_figures:
        exhibit_rule = (
            "- PREFER EXHIBITS OVER PROSE: When results are quantitative, present them as a PLOT "
            "figure (or a markdown table); when the passage centers on a specific named molecule, "
            "functional group, or structural comparison, include it as a SMILES figure. Use ONLY "
            "the two existing render types (plot, smiles). Favor density over length — lean prose "
            "wrapped around dense exhibits, not long narration."
        )
    else:
        exhibit_rule = (
            "- PREFER EXHIBITS OVER PROSE: When results are quantitative, present them as a "
            "markdown TABLE of raw values. Favor density over length — lean prose wrapped around a "
            "dense data table, not long narration."
        )
    realism_rules = (
        "- RESULTS WITHHELD FROM PROSE: The passage prose may describe background, motivation, "
        "methods, and experimental CONDITIONS, but must NOT state, characterize, or interpret the "
        "RESULTS or any trend. Do NOT write sentences like \"X increased with Y\", \"the higher "
        "the load, the slower the lift\", or \"the treatment outperformed the control\". ALL "
        "quantitative results must appear ONLY in a markdown table or a figure, as raw "
        "values/data — never narrated, summarized, or interpreted in prose. The reader must "
        "interpret the data themselves.\n"
        "- RESEARCH-DESIGN DETAIL: Describe a real study/experiment with enough DESIGN detail to "
        "support a research-design question: name or imply a control or comparison condition, "
        "state what was held constant and what was varied, and where natural include a limitation "
        "or potential confound.\n"
        f"{exhibit_rule}"
    )

    system = f"""You are an expert MCAT passage writer for the Association of American \
Medical Colleges (AAMC). You write realistic science passages for the {section} section \
of the MCAT.

A science passage is a {word_min}-{word_max} word piece that typically describes an \
EXPERIMENT or STUDY: its background/motivation, methods, and results. It reads like a \
condensed methods-and-results section written for an educated reader, and it gives \
students concrete material to reason about.

STRICT CONSTRAINTS:
- Length: {word_min}-{word_max} words of passage prose (not counting any table).
{figure_rule}
- If (and only if) a data table genuinely helps, include ONE markdown table in the \
"table_markdown" field (use standard GitHub-flavored markdown: header row, separator row, \
data rows). Otherwise set "table_markdown" to null. Do NOT embed the table inside \
"passage_text".
- The science must be accurate and at the introductory college level (the level the MCAT \
tests). Any numbers/data must be internally consistent and plausible.
- The passage should give enough material that some questions are answerable directly \
from it, some require the reader to apply outside introductory science knowledge, and \
some require interpreting the passage's data.
{realism_rules}
- {LATEX_NOTATION_RULE}
- Do NOT include any questions in the passage. Write ONLY the passage (and optional table{', figures' if enable_figures else ''}).{figure_section}

Respond with ONLY a JSON object in this exact format, no other text:
{{
  "passage_text": "The full passage prose here.",
  "table_markdown": "| Group | Value |\\n| --- | --- |\\n| A | 1 |"{figures_json_line}
}}
(Set "table_markdown" to null if no table.)"""

    user = f"""Write one MCAT science passage for the following topic cluster.

Section: {section}
Content Category: {content_category}
Topic Group: {topic_group}
Related topics this passage should give material for:
{topics_str}

Write a {word_min}-{word_max} word passage (describing an experiment or study where \
natural) that ties these related topics together and gives a student concrete material to \
reason about. Include a markdown table only if it genuinely helps present data.{figures_user_note}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def passage_review_prompt(
    passage_text: str,
    table_markdown: str,
    topics: list[dict],
    word_min: int,
    word_max: int,
    figures_text: str = "",
    enable_figures: bool = False,
) -> list[dict]:
    """Review a science passage for accuracy and figure/no-figure constraints.

    `figures_text` is the TEXT serialization of any passage-level figure specs
    (SMILES strings / plot data) — the reviewer reasons over that, not an image.
    """

    table_block = (
        f"\n\nAccompanying table:\n{table_markdown}" if table_markdown else "\n\n(No table.)"
    )
    figure_block = f"\n\nAccompanying figure(s) (shown to the student as images):\n{figures_text}" if figures_text else ""
    topic_names = ", ".join(t.get("topic", "?") for t in topics)

    if enable_figures:
        figure_check = (
            "3. FIGURE CONSISTENCY: The passage may include chemical structures (SMILES) or "
            "plots, provided above as text. FAIL it if the passage references a figure/graph/"
            "structure that is NOT provided above (an unspecified figure), or if a provided "
            "figure's data contradicts the prose. A specified figure is fine. Tables and "
            "specified figures are the only non-prose elements allowed."
        )
    else:
        figure_check = (
            "3. NO-FIGURE CONSTRAINT: Does the passage stand on its own with ONLY prose and "
            "the optional markdown table? FAIL it if it references or requires a figure, "
            "graph, diagram, image, or structure that is not expressible as text/a table "
            "(e.g. \"as shown in Figure 1\", \"the graph below\", \"the structure depicted\")."
        )

    system = f"""You are a rigorous MCAT passage reviewer for the AAMC. You vet science \
passages before questions are written for them. Check the passage and answer with a \
JSON verdict.

{LATEX_REVIEW_NOTE}

Check for:
1. SCIENTIFIC ACCURACY: Is everything stated scientifically correct at the introductory \
college level? Flag any factual errors, impossible values, or misleading claims.
2. INTERNAL CONSISTENCY: Are any data/numbers (in prose, the table, or a figure) \
self-consistent and plausible? Does the table, if present, parse as valid markdown and \
match the prose?
{figure_check}
4. LENGTH/STYLE: Roughly {word_min}-{word_max} words, written like a real MCAT science \
passage (experiment/study style preferred).
5. USABILITY: Does it give enough concrete material (a scenario, methods, results/data) to \
support a mix of passage-based, knowledge-application, and data-interpretation questions?

Respond with ONLY a JSON object:
{{
  "passed": true,
  "issues": [],
  "reasoning": "Brief assessment"
}}

Set "passed" to false if ANY significant issue is found, and list the issues."""

    user = f"""Review this MCAT science passage (intended topics: {topic_names}).

Passage:
{passage_text}{table_block}{figure_block}

Critically evaluate it for accuracy, consistency, the figure constraint, and usability."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _passage_context_block(
    passage_text: str, table_markdown: str, figures_text: str = ""
) -> str:
    block = f"PASSAGE:\n{passage_text}"
    if table_markdown:
        block += f"\n\nTABLE:\n{table_markdown}"
    if figures_text:
        block += f"\n\nPASSAGE FIGURE(S) (the student sees these as images):\n{figures_text}"
    return block


def question_generation_prompt(
    passage_text: str,
    table_markdown: str,
    section: str,
    plan: dict,
    previous_stems: list[str] = None,
    passage_figures_text: str = "",
    enable_figures: bool = False,
) -> list[dict]:
    """Build the prompt for generating ONE passage-linked science question.

    `plan` carries the assigned skill, answer_basis, difficulty, and target
    answer position. `previous_stems` are stems already accepted FOR THIS
    PASSAGE — fed back for within-passage diversity (cover different angles).
    `passage_figures_text` is the text serialization of any passage-level figures
    (so the writer can reference them); when `enable_figures` is set the question
    may also specify its own figure.
    """
    skill_key = plan["skill_key"]
    skill_label = plan["skill_label"]
    answer_basis = plan["answer_basis"]
    difficulty = plan["difficulty"]
    target_answer = plan["target_answer"]

    difficulty_guidance = {
        "easy": (
            "EASY: answerable in well under a minute; direct application or a clear "
            "read of the passage. Distractors still plausible."
        ),
        "medium": (
            "MEDIUM: typical MCAT difficulty; careful reading plus a non-trivial step "
            "of reasoning or data interpretation."
        ),
        "hard": (
            "HARD: combine the passage with a concept or multiple steps, or require "
            "careful elimination of subtly wrong distractors. Still introductory "
            "college level, not graduate obscurity."
        ),
    }

    diversity_str = ""
    if previous_stems:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(previous_stems, 1))
        diversity_str = (
            "\n\nQuestions already written for THIS passage test the following. Do NOT "
            "duplicate them — write a question on a different aspect of the passage, a "
            "different concept, or a different part of the data:\n"
            f"{numbered}"
        )

    context = _passage_context_block(passage_text, table_markdown, passage_figures_text)

    # FIGURES: questions reference figures that ALREADY exist on the passage; the
    # dedicated figure-pass stage owns figure decisions. So the question-gen prompt
    # gets only a SHORT note (not the full FIGURE_GEN_INSTRUCTIONS block, which is
    # injected at passage generation and in the figure-pass) — this de-duplicates a
    # ~35-line block and trims input tokens without removing the ability to attach a
    # figure when a question genuinely needs one the passage lacks.
    figure_section = (
        "\n- FIGURES: This question may REFERENCE figures already shown on the passage above; "
        "it rarely needs its own. Normally set \"figures\" to []. Only add a figure if THIS "
        "question genuinely needs one the passage lacks (e.g. a specific structure to identify "
        "or a plot to read) — if so, provide its COMPLETE underlying data (valid parseable "
        "SMILES, or plot series with equal-length x/y arrays). Never reference a "
        "figure/structure/graph you have not provided in \"figures\"."
        if enable_figures else ""
    )
    figures_json_line = ',\n  "figures": <array of figure specs; use [] if none — see FIGURES note above>' if enable_figures else ""

    # Exact top-level keys allowed in the question JSON ("figures" only when enabled).
    allowed_keys = [
        '"stem"', '"choices"', '"correct_answer"', '"explanation"',
        '"difficulty"', '"answer_basis"', '"skill_tested"',
    ]
    if enable_figures:
        allowed_keys.append('"figures"')
    allowed_keys_str = ", ".join(allowed_keys)

    # Brief context line naming the two bases NOT in force, so the assigned one
    # keeps the spotlight (audit: defining all three with equal weight was bloat).
    other_bases = [b for b in ("from_passage", "apply_knowledge", "data_interpretation") if b != answer_basis]
    other_bases_str = "; ".join(f"\"{b}\" = {ANSWER_BASIS_BRIEF[b]}" for b in other_bases)

    system = f"""You are an expert MCAT question writer for the AAMC, writing a question \
tied to a science passage in the {section} section.

#1 PRIORITY — SCIENTIFIC ACCURACY (this is the hard part; get it right before worrying about \
formatting):
- Every element must be factually correct at the introductory-college level and consistent \
with the passage and its exhibits: the stem, the keyed correct answer, EACH distractor's \
wrongness, and the explanation.
- The keyed answer must be unambiguously and verifiably correct given the passage/data, and \
each distractor must be genuinely WRONG (ideally a real student misconception, partial \
reasoning, or data misreading) — never a second defensible answer.
- Before finalizing, DOUBLE-CHECK every mechanism, calculation, stereochemistry, and data \
interpretation against the passage, step by step. This is where subtle errors hide — verify \
them explicitly. If you are not fully certain a claim is correct, revise until it is.
The formatting rules further below (answer position, no fifth option, LaTeX, JSON output \
contract, the answer_basis label) are easy compliance items — do not let them pull attention \
away from getting the science right.

{_SKILL_GUIDANCE[skill_key]}

This question's ANSWER BASIS is "{answer_basis}" — make this genuinely true:
{ANSWER_BASIS_LABELS[answer_basis]}
{ANSWER_BASIS_REQUIREMENTS[answer_basis]}
(For context only, the other two bases — NOT in force here — are {other_bases_str}. THIS \
question must satisfy "{answer_basis}".)

Target difficulty: {difficulty.upper()}
{difficulty_guidance[difficulty]}

Question-writing rules:
- STEM LENGTH (applies to the "stem" field ONLY): the question must be tied to the passage \
above; the stem may quote or paraphrase the passage but keep it concise (under ~120 words). \
This cap does NOT apply to the explanation.
- For THIS question the correct answer MUST be option {target_answer}; write the other three \
as plausible distractors (common misconceptions, partial reasoning, or data misreadings). Do \
not let the correct option be guessable from length, specificity, or phrasing.
- Choices should be similar in length/specificity.
- Introductory college level; algebra/logs/basic trig/dimensional analysis only (no \
calculus); a periodic table is available.
- EXPLANATION (the "explanation" field — separate from the stem cap above): be thorough but \
focused — state why the correct answer is right and why EACH distractor is wrong, in as many \
words as it genuinely needs and no more (no padding or repetition). It is FINAL, PUBLISHED \
content shown to a student — not a draft or a note to a reviewer. {NO_FIFTH_OPTION_EXPLANATION_RULE} \
It must contain NO meta-commentary about the question itself — no hedging, disclaimers, or \
remarks about the item's format, validity, or completeness (e.g. never write "this item is \
illustrative", "placeholder", "a properly formatted question would have four options", or \
similar). {CHOICE_REFERENCE_BOLD_RULE}
- {NO_FIFTH_OPTION_RULE}{figure_section}

Respond with ONLY a JSON object that has EXACTLY these keys. Each value is described in \
angle brackets — produce a real value matching the description; do NOT output the \
angle-bracket text itself:
{{
  "stem": <string: the full question text, tied to the passage>,
  "choices": <object with EXACTLY keys "A","B","C","D", each a non-empty string>,
  "correct_answer": <one of "A","B","C","D"; for this question must be "{target_answer}">,
  "explanation": <string: final published explanation referencing only A-D, no meta-commentary>,
  "difficulty": <exactly "{difficulty}">,
  "answer_basis": <exactly "{answer_basis}">,
  "skill_tested": <exactly "{skill_label}">{figures_json_line}
}}

STRICT OUTPUT CONTRACT — follow EXACTLY or the response is rejected:
- The angle-bracket descriptions above tell you what each value must be — replace each with \
real content; never emit a literal "<...>" or the word "placeholder" as a value or key.
- {NO_FIFTH_OPTION_CONTRACT}
- {LATEX_NOTATION_RULE}
- Output EXACTLY these top-level keys and NOTHING else: {allowed_keys_str}. Do NOT add any \
other top-level key for any reason — no "comment", "note", "explanation_placeholder", \
"answer_placeholder", "choices_extra", or any other commentary, metadata, or placeholder key.
Output the raw JSON object only — no markdown fences, no preamble, no trailing notes.
OUTPUT FORMAT — ABSOLUTE: Respond with ONLY the raw JSON object and NOTHING else. Do NOT \
write any reasoning, preamble, explanation, or text before or after the JSON. Do NOT write \
phrases like 'I need to', 'Let me', 'First,' or any thinking out loud. Do NOT wrap the JSON \
in markdown fences. Your entire response must start with {{ and end with }} and be valid \
parseable JSON. Any text outside the JSON object will cause the response to be discarded."""

    user = f"""{context}

Write ONE MCAT question about this passage.
Target skill: {skill_label}
Answer basis: {answer_basis}
Target difficulty: {difficulty}
Target correct-answer position: {target_answer}{diversity_str}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _question_figures_block(question_figures_text: str) -> str:
    if not question_figures_text:
        return ""
    return (
        "\n\nFIGURE(S) FOR THIS QUESTION (the student sees these as images; reason over "
        f"the data below):\n{question_figures_text}"
    )


def adversarial_review_prompt(
    passage_text: str,
    table_markdown: str,
    question_data: dict,
    passage_figures_text: str = "",
    question_figures_text: str = "",
) -> list[dict]:
    """Adversarial review of a passage-linked question (passage as context).

    Includes a SEPARATE, LENIENT verdict on whether the question's answer_basis
    label is honest. The answer_basis check should catch CLEAR mislabels only —
    it must tolerate reasonable judgment calls to avoid excessive regeneration.

    Figures are passed as TEXT (SMILES strings / plot data), never as an image:
    `passage_figures_text` for passage-level figures and `question_figures_text`
    for figures attached to this question.
    """
    context = _passage_context_block(passage_text, table_markdown, passage_figures_text)
    q_fig_block = _question_figures_block(question_figures_text)
    basis = question_data.get("answer_basis", "")

    system = """You are a rigorous MCAT question reviewer for the AAMC. You vet \
passage-based science questions before they reach students. You are given the passage and \
one question written about it. Be critical and thorough.

Any figures (chemical structures or plots) are provided to you as TEXT (SMILES strings or \
the plot's underlying data); the student sees them rendered as images. Reason over that data.

%s

Check for:
1. ACCURACY: Is the stated correct answer actually correct given the passage, any figure \
data, and correct science? Any factual errors in stem, choices, or explanation?
2. PASSAGE GROUNDING: Is the question genuinely tied to the passage/figure? If it claims to \
be answerable from the passage, does the passage/figure actually support the keyed answer?
3. AMBIGUITY: Could more than one answer be defensibly correct? Is the stem clear?
4. DISTRACTORS: Are the wrong answers plausible (real misconceptions / common errors / \
data misreadings)?
5. SKILL ALIGNMENT: Does it test the stated SIRS skill (reasoning/research-design/data, \
not bare recall when a higher skill is claimed)?
6. ANSWER BALANCE & STRUCTURE: Choices roughly similar in length/specificity; EXACTLY four \
choices A/B/C/D, none missing/extra/empty. Fail if malformed.
7. FIGURE INTEGRITY: If the stem references a figure/structure/graph, a corresponding \
figure must be provided above. FAIL if the stem references a figure that is not provided, \
or if the keyed answer contradicts the figure's data.

SEPARATELY, judge the ANSWER BASIS label. The question is labeled answer_basis = \
"%s", which means:
  - from_passage: answerable directly from the passage (no outside knowledge beyond basic \
literacy).
  - apply_knowledge: requires the student's own intro-college science knowledge applied to \
the passage (NOT answerable from the passage text alone).
  - data_interpretation: hinges on interpreting data presented in the passage (including a \
figure's data).
Set "answer_basis_ok" to false ONLY for a CLEAR mislabel — for example, a question labeled \
"from_passage" whose answer is pure outside recall the passage never supports, or one \
labeled "apply_knowledge" that is in fact answered verbatim by a sentence in the passage. \
If the label is reasonable or it's a borderline judgment call, set "answer_basis_ok" to \
true. Do NOT nitpick. This check should be forgiving.

Respond with ONLY a JSON object:
{
  "passed": true,
  "issues": [],
  "reasoning": "Brief assessment of overall quality",
  "answer_basis_ok": true,
  "answer_basis_note": "Only if answer_basis_ok is false: briefly why it's a clear mislabel"
}

Set "passed" to false if ANY significant quality issue (items 1-7) is found. The \
answer_basis verdict is reported SEPARATELY in "answer_basis_ok" and should not, by itself, \
drive "passed".""" % (LATEX_REVIEW_NOTE, basis)

    user = f"""{context}{q_fig_block}

QUESTION TO REVIEW:
Stem: {question_data['stem']}

A) {question_data['choices']['A']}
B) {question_data['choices']['B']}
C) {question_data['choices']['C']}
D) {question_data['choices']['D']}

Stated correct answer: {question_data['correct_answer']}
Explanation: {question_data['explanation']}
Skill tested: {question_data.get('skill_tested', 'Not specified')}
Labeled answer_basis: {basis}

Critically evaluate this question against the passage and any figure data. Find any flaws, \
and give the separate (lenient) answer_basis verdict."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def blind_solve_prompt(
    passage_text: str,
    table_markdown: str,
    question_data: dict,
    passage_figures_text: str = "",
    question_figures_text: str = "",
) -> list[dict]:
    """Blind-solve a passage-linked question (passage given, no answer key).

    Figures are provided as TEXT (SMILES strings / plot data) so the solver can
    reason about them exactly as the student would from the rendered image.
    """
    context = _passage_context_block(passage_text, table_markdown, passage_figures_text)
    q_fig_block = _question_figures_block(question_figures_text)

    system = """You are an MCAT expert with introductory college-level knowledge of biology, \
biochemistry, chemistry, physics, and psychology/sociology. You are given a passage and a \
question about it. Select the SINGLE BEST answer choice.

Any figures (chemical structures or plots) are given to you as TEXT (SMILES strings or the \
plot's underlying data); the student sees them as images. Reason over that data.

""" + LATEX_REVIEW_NOTE + """

Work step by step: read the passage, figures, and stem, use the passage and your own \
knowledge as needed, eliminate wrong choices, and choose the best one.

Respond with ONLY a JSON object:
{
  "chosen_answer": "B",
  "confidence": "high",
  "reasoning": "Brief reasoning"
}

confidence should be "high", "medium", or "low".

OUTPUT FORMAT — ABSOLUTE: Respond with ONLY the raw JSON object specified above and NOTHING \
else. Do NOT show your reasoning or working in prose outside the JSON. Do NOT write 'I need \
to', 'Let me', or any preamble. Put any brief reasoning INSIDE the JSON's reasoning field if \
one exists; otherwise omit it. Your entire response must start with { and end with } and be \
valid parseable JSON."""

    user = f"""{context}{q_fig_block}

QUESTION:
{question_data['stem']}

A) {question_data['choices']['A']}
B) {question_data['choices']['B']}
C) {question_data['choices']['C']}
D) {question_data['choices']['D']}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
