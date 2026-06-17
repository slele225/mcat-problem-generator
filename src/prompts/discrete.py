"""Prompt templates for discrete (non-CARS) MCAT question generation.

Prompt design informed by the AAMC's official "What's on the MCAT Exam?" content outline,
which defines four Scientific Inquiry and Reasoning Skills (SIRS) tested across the three
science sections:

  Skill 1 (35%): Knowledge of Scientific Concepts and Principles
    - Recognize, identify, recall, or define concepts and their relationships
    - Identify examples/observations that illustrate scientific principles
    - Use given equations to solve problems

  Skill 2 (45%): Scientific Reasoning and Problem-Solving
    - Use theories to explain observations or make predictions
    - Evaluate arguments about cause and effect
    - Use models and observations to draw conclusions
    - Determine and use formulas to solve problems

  Skill 3 (10%): Reasoning About the Design and Execution of Research
    - Identify roles of variables (independent, dependent, confounding)
    - Evaluate appropriateness of research methods and tools
    - Identify limitations, ethical issues, and faulty research logic

  Skill 4 (10%): Data-Based and Statistical Reasoning
    - Interpret patterns in tables, figures, and graphs
    - Use measures of central tendency and dispersion
    - Reason about significance, uncertainty, and error
    - Draw conclusions from data and identify supported claims

Each science section has 59 questions in 95 minutes, combining passage-based and
discrete (standalone) questions. This module generates discrete-style questions.
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


# Weighted distribution matching AAMC percentages
SKILL_WEIGHTS = {
    "skill_1": 35,
    "skill_2": 45,
    "skill_3": 10,
    "skill_4": 10,
}

SKILL_LABELS = {
    "skill_1": "Skill 1: Knowledge of Scientific Concepts and Principles",
    "skill_2": "Skill 2: Scientific Reasoning and Problem-Solving",
    "skill_3": "Skill 3: Reasoning About the Design and Execution of Research",
    "skill_4": "Skill 4: Data-Based and Statistical Reasoning",
}

# Difficulty distribution: easier questions should be rarer than the meat of
# the exam, hard questions show up regularly but not as the default.
DIFFICULTY_WEIGHTS = {
    "easy": 20,
    "medium": 50,
    "hard": 30,
}


def _pick_skill() -> str:
    """Randomly select a skill category weighted by AAMC distribution."""
    keys = list(SKILL_WEIGHTS.keys())
    weights = list(SKILL_WEIGHTS.values())
    return random.choices(keys, weights=weights, k=1)[0]


def _pick_difficulty() -> str:
    """Randomly select a target difficulty weighted 20/50/30."""
    keys = list(DIFFICULTY_WEIGHTS.keys())
    weights = list(DIFFICULTY_WEIGHTS.values())
    return random.choices(keys, weights=weights, k=1)[0]


def _pick_answer() -> str:
    """Randomly select the target correct-answer position (equal probability).

    LLMs tend to favor B/C as the correct choice; fixing a uniformly random
    target letter per question keeps the answer key balanced across A/B/C/D.
    """
    return random.choice(["A", "B", "C", "D"])


def generation_prompt(
    topic_data: dict, previous_stems: list[str] = None
) -> tuple[list[dict], str]:
    """Build the prompt for generating a single discrete MCAT question.

    Returns ``(messages, skill_label)``: the chat messages and the AAMC SIRS
    skill this question was ASSIGNED to test. The skill is chosen here (the
    prompt instructs the model to echo it back), so surfacing it lets the
    pipeline persist the authoritative assigned skill rather than trusting the
    model's echo (which can be blank or mislabeled).

    Args:
        topic_data: A single topic entry from topics.json
        previous_stems: Stems of questions already accepted for this topic. When
            non-empty, a section is added to the USER message asking for a
            genuinely different angle (within-topic diversity). Empty/None leaves
            the prompt unchanged.
    """
    subtopics_str = ""
    if topic_data.get("subtopics"):
        subtopics_str = f"\nSpecific subtopics to potentially test: {', '.join(topic_data['subtopics'])}"

    focus_str = ""
    if topic_data.get("section_focus"):
        focus_str = f"\nFocus angle: {topic_data['section_focus']}"

    diversity_str = ""
    if previous_stems:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(previous_stems, 1))
        diversity_str = (
            "\n\nQuestions already accepted for this topic test the following. Do NOT "
            "duplicate them — generate a question on a genuinely different concept, "
            "application, or angle within this topic:\n"
            f"{numbered}\n"
            "If the topic is too narrow for another distinct angle, a related question "
            "is acceptable, but prioritize new ground."
        )

    skill_key = _pick_skill()
    skill_label = SKILL_LABELS[skill_key]
    difficulty = _pick_difficulty()
    target_answer = _pick_answer()

    difficulty_guidance = {
        "easy": (
            "EASY: A well-prepared student should be able to answer this in under 30 "
            "seconds. The scenario should be straightforward and the correct application "
            "of the concept should be relatively direct. Distractors should still be "
            "plausible, but the right answer should be reachable without multi-step "
            "reasoning."
        ),
        "medium": (
            "MEDIUM: Typical MCAT difficulty. The student must read carefully, apply a "
            "concept or principle to a non-trivial scenario, and reason through plausible "
            "distractors. Should take ~60-75 seconds."
        ),
        "hard": (
            "HARD: A challenging MCAT question. May require combining two concepts, "
            "multi-step calculation, careful elimination of subtly wrong distractors, or "
            "recognizing a counterintuitive application of a principle. Still solvable "
            "with introductory college-level knowledge — not graduate-level obscurity."
        ),
    }

    # Skill-specific generation guidance based on AAMC descriptions
    skill_guidance = {
        "skill_1": """\
This question should test Skill 1: Knowledge of Scientific Concepts and Principles.
Ask the student to recognize/identify a concept, principle, or relationship from a scenario, \
example, or representation (diagram, graph, structural formula), or to use a given equation. \
Do NOT write a bare recall/definition question — present something the student must interpret.
Compact example: a stem describing salivation that declines across repeated lemon-juice trials \
then recovers when switched to lime juice; correct answer "habituation and dishabituation" \
(recognized from the scenario), with distractors that are related-but-distinct learning \
processes.""",

        "skill_2": """\
This question should test Skill 2: Scientific Reasoning and Problem-Solving.
Ask the student to apply a theory/model to a novel scenario, evaluate an explanation or a \
cause-and-effect argument, draw a conclusion from evidence, or carry out a multi-step \
calculation. Require REASONING, not recall.
RECOGNITION vs. REASONING (this trips up psych/soc concept items): merely identifying which \
named concept a vignette illustrates is Skill 1 (Knowledge), NOT Skill 2 — do not write that \
here. Skill 1 example: "a traveler judges another culture by their own culture's norms" -> name \
it (ethnocentrism). Skill 2 example: given two study findings, predict how an ethnocentric \
framing would bias a researcher's interpretation. Skill 2 must demand application or analysis \
beyond labeling the concept.
Compact example: give aorta vs. capillary radii and flow velocities and ask for the approximate \
number of capillaries; the student applies continuity (A₁v₁ = N·A₂v₂) in a multi-step \
calculation, with distractors that are order-of-magnitude errors.""",

        "skill_3": """\
This question should test Skill 3: Reasoning About the Design and Execution of Research.
You MUST describe a specific experiment or study in the stem. Ask the student to identify \
variables (independent/dependent/confounding), evaluate a method or control, spot a design \
flaw, or distinguish correlation from causation.
Compact example: describe a randomized caffeine-vs-placebo word-recall study and ask for the \
dependent variable; correct answer is the measured outcome (words recalled), with distractors \
that are the other variables in the design.""",

        "skill_4": """\
This question should test Skill 4: Data-Based and Statistical Reasoning.
You MUST present specific data (numbers, values, trends) in the stem. Ask the student to \
interpret patterns, use central tendency/dispersion, reason about error or significance, or \
judge which conclusion the data support.
Compact example: give mean ± SD resting heart rate across four exercise-intensity groups and \
ask which conclusion is supported; correct answer is the descriptive trend, distinguished from \
causal overreach, an unsupported universal claim, and an SD misreading.""",
    }

    system = f"""You are an expert MCAT question writer for the Association of American Medical \
Colleges (AAMC). You create questions that match the difficulty, style, and cognitive demands of \
the actual MCAT exam.

The MCAT tests four Scientific Inquiry and Reasoning Skills across its science sections. \
You are writing one question that tests a specific skill.

#1 PRIORITY — SCIENTIFIC ACCURACY (this is the hard part; get it right before worrying about \
formatting):
- Every element must be factually correct at the introductory-college level: the stem, the \
keyed correct answer, EACH distractor, and the explanation.
- The keyed answer must be unambiguously and verifiably correct, and each distractor must be \
genuinely WRONG (ideally a real student misconception) — never a second defensible answer.
- Before finalizing, DOUBLE-CHECK every mechanism, reaction, stereochemistry, calculation, and \
factual claim, step by step. Organic-chemistry mechanisms and molecular-biology details are \
where subtle errors hide — verify them explicitly. If you are not fully certain a claim is \
correct, revise the question until it is.
The formatting rules further below (answer position, no fifth option, LaTeX, JSON contract) are \
easy compliance items — do not let them pull attention away from getting the science right.

{skill_guidance[skill_key]}

Target difficulty for this question: {difficulty.upper()}
{difficulty_guidance[difficulty]}

Question-writing rules:
- STEM LENGTH (applies to the "stem" field ONLY): keep the stem under 150 words. Discrete \
questions are standalone, not passage-based — if a scenario needs more setup than that, simplify \
it. This cap does NOT apply to the explanation.
- For THIS question, the correct answer MUST be option {target_answer}. Write the four choices so \
that option {target_answer} is the single correct answer and the other three are plausible \
distractors (common misconceptions, partially correct reasoning, or typical errors). Do not let \
the correct option's position be guessable from its length, specificity, or phrasing.
- Answer choices should be roughly similar in length and specificity (a correct answer that is \
much longer or more detailed than the others is a test-taking giveaway).
- Use introductory college-level knowledge, not graduate-level obscurity. Math uses algebra, \
logarithms, basic trig, or dimensional analysis (no calculus); a periodic table is available.
- EXPLANATION (the "explanation" field — separate from the stem cap above): be thorough but \
focused — state why the correct answer is right and why EACH distractor is wrong, in as many \
words as that genuinely needs and no more (no padding or repetition). It is FINAL, PUBLISHED \
content for a student, so write it as finished and correct: NO meta-commentary about the \
question's format, validity, or completeness (never write "this item is illustrative", \
"placeholder", "a properly formatted question would have four options", or similar), and no \
hedging or disclaimers. {NO_FIFTH_OPTION_EXPLANATION_RULE} {CHOICE_REFERENCE_BOLD_RULE}
- {NO_FIFTH_OPTION_RULE}

Respond with ONLY a JSON object that has EXACTLY these seven keys. Each value is described \
in angle brackets — produce a real value matching the description; do NOT output the \
angle-bracket text itself:
{{
  "stem": <string: the full question text>,
  "choices": <object with EXACTLY keys "A","B","C","D", each a non-empty string>,
  "correct_answer": <one of "A","B","C","D"; for this question must be "{target_answer}">,
  "explanation": <string: final published explanation referencing only A-D, no meta-commentary>,
  "difficulty": <exactly "{difficulty}">,
  "subtopics_tested": <array of 1-3 short strings>,
  "skill_tested": <exactly "{skill_label}">
}}

STRICT OUTPUT CONTRACT — follow EXACTLY or the response is rejected:
- Replace each angle-bracket description with real content; never emit a literal "<...>" or the \
word "placeholder" as a value or key.
- {NO_FIFTH_OPTION_CONTRACT}
- {LATEX_NOTATION_RULE}
- Output EXACTLY the seven keys listed above and NOTHING else. Do NOT add any other \
top-level key for any reason — no "comment", "note", "explanation_placeholder", \
"answer_placeholder", "choices_extra", or any other commentary, metadata, or placeholder key.
Output the raw JSON object only — no markdown fences, no preamble, no trailing notes."""

    user = f"""Generate an MCAT-style discrete question for:

Section: {topic_data['section']}
Content Category: {topic_data['content_category']}
Topic Group: {topic_data['topic_group']}
Topic: {topic_data['topic']}
Discipline: {topic_data.get('discipline', 'N/A')}{focus_str}{subtopics_str}

Target Skill: {skill_label}
Target Difficulty: {difficulty}
Target correct-answer position: {target_answer}{diversity_str}

Write a realistic MCAT question testing this specific skill at the target difficulty. \
Present a scenario, experiment, data, or problem that requires the student \
to think, not just remember. Above all, make sure every element — the keyed answer, each \
distractor, and the explanation — is scientifically accurate; verify any mechanism or \
calculation before you finalize."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], skill_label


def adversarial_review_prompt(question_data: dict, topic_data: dict) -> list[dict]:
    """Build the prompt for adversarial review of a generated question."""

    system = """You are a rigorous MCAT question reviewer and quality assurance expert working \
for the AAMC. Your job is to find flaws in MCAT questions before they go to students. Be \
critical and thorough.

""" + LATEX_REVIEW_NOTE + """

The MCAT tests four Scientific Inquiry and Reasoning Skills:
- Skill 1 (Knowledge): Recognize/identify concepts, use equations, interpret representations
- Skill 2 (Reasoning): Apply theories to scenarios, evaluate explanations, multi-step problems
- Skill 3 (Research Design): Identify variables, evaluate methods, spot design flaws
- Skill 4 (Data Reasoning): Interpret data/graphs/tables, use statistics, draw data-based conclusions

You must check for:
1. ACCURACY: Is the stated correct answer actually correct? Are there any factual errors in \
the stem, choices, or explanation?
2. AMBIGUITY: Could more than one answer be defensibly correct? Is the stem clear enough \
that a knowledgeable student would not be confused?
3. DISTRACTORS: Are the wrong answers plausible? Do they represent real misconceptions or \
common errors? Would a prepared student actually consider them?
4. SKILL ALIGNMENT: Does the question actually test the stated SIRS skill? A Skill 2 question \
should require reasoning/application, not just recall. A Skill 3 question must describe a study. \
A Skill 4 question must present data.
5. DIFFICULTY: Is this appropriate for the MCAT (introductory college-level, not graduate-level)?
6. ANSWER BALANCE: Are the choices roughly similar in length and specificity? A correct answer \
that is noticeably longer, more detailed, or more qualified than the distractors is a \
test-taking giveaway that must be fixed.
7. STEM QUALITY: Does the stem present a scenario or problem (not just "which of the following \
is true about X")? For Skills 3-4, does it describe a study or present data?
8. CHOICE STRUCTURE: The question must have EXACTLY four answer choices keyed A, B, C, and D — \
no missing, extra, duplicate, or empty choices. Fail the question if the choice set is malformed.

Respond with ONLY a JSON object:
{
  "passed": true,
  "issues": [],
  "reasoning": "Brief explanation of your assessment"
}

Set "passed" to false if ANY significant issue is found. List all issues found."""

    user = f"""Review this MCAT question for quality, accuracy, and AAMC alignment.

Topic context:
- Section: {topic_data['section']}
- Content Category: {topic_data['content_category']}
- Topic: {topic_data['topic']}

Question to review:
Stem: {question_data['stem']}

Choices:
A) {question_data['choices']['A']}
B) {question_data['choices']['B']}
C) {question_data['choices']['C']}
D) {question_data['choices']['D']}

Stated correct answer: {question_data['correct_answer']}
Explanation: {question_data['explanation']}
Skill tested: {question_data.get('skill_tested', 'Not specified')}

Critically evaluate this question. Find any flaws."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def blind_solve_prompt(question_data: dict) -> list[dict]:
    """Build the prompt for blind-solving a question (no answer key given)."""

    system = """You are an MCAT expert with deep knowledge of biology, biochemistry, chemistry, \
physics, and psychology/sociology at the introductory college level. Answer the following \
question by selecting the SINGLE BEST answer choice.

""" + LATEX_REVIEW_NOTE + """

Think through it step by step:
1. Read the stem carefully and identify what is being asked
2. Consider each answer choice
3. Eliminate obviously wrong answers
4. Choose the best remaining answer

Respond with ONLY a JSON object:
{
  "chosen_answer": "B",
  "confidence": "high",
  "reasoning": "Brief explanation of your reasoning"
}

confidence should be "high", "medium", or "low".

OUTPUT FORMAT — ABSOLUTE: Respond with ONLY the raw JSON object specified above and NOTHING \
else. Do NOT show your reasoning or working in prose outside the JSON. Do NOT write 'I need \
to', 'Let me', or any preamble. Put any brief reasoning INSIDE the JSON's reasoning field if \
one exists; otherwise omit it. Your entire response must start with { and end with } and be \
valid parseable JSON."""

    user = f"""Answer this MCAT question:

{question_data['stem']}

A) {question_data['choices']['A']}
B) {question_data['choices']['B']}
C) {question_data['choices']['C']}
D) {question_data['choices']['D']}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
