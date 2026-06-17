"""Prompt templates for CARS passage and question generation.

Prompt design informed by the AAMC's official "What's on the MCAT Exam?" content outline,
which describes the Critical Analysis and Reasoning Skills (CARS) section as follows:

Format:
  - 53 questions in 90 minutes, ALL passage-based (no discrete questions)
  - Passages are 500-600 words, complex, often thought-provoking with sophisticated vocabulary
  - Passages come from humanities (50%) and social sciences (50%)
  - Everything needed to answer is in the passage — no outside knowledge required

Three skills tested (AAMC distribution):
  Foundations of Comprehension (30%):
    - Understanding basic components of the text (main idea, specific details, thesis)
    - Inferring meaning or intent from immediate sentence context (word meaning,
      rhetorical devices, author's tone, text structure)

  Reasoning Within the Text (30%):
    - Integrating distant components to infer author's message, purpose, bias, assumptions
    - Recognizing and evaluating arguments and their structural elements (claims, evidence,
      support, logical relationships)
    - Detecting paradoxes, contradictions, or inconsistencies across passage sections
    - Identifying perspective: author's own vs. paraphrased/quoted others' views

  Reasoning Beyond the Text (40%):
    - Applying or extrapolating passage ideas to new contexts (analogies, hypotheticals)
    - Assessing impact of incorporating new factors, information, or conditions on passage ideas
    - "What if" questions that ask how new information would affect the argument
    - Selecting which new fact would most/least alter the passage's central thesis

Passage characteristics (per AAMC):
  - Humanities: architecture, art, dance, ethics, literature, music, philosophy,
    popular culture, religion, theater, studies of diverse cultures
  - Social Sciences: anthropology, archaeology, economics, education, geography,
    history, linguistics, political science, population health, psychology, sociology
  - Social science passages tend to be more factual/scientific in tone
  - Humanities passages focus on relationships between ideas, often conversational/opinionated
  - Passages are "multifaceted and focus on the relationships between ideas or theories"
  - Authors use "sophisticated vocabulary and, at times, intricate writing styles"
"""

import random


CARS_SKILL_TYPES = [
    "Foundations of Comprehension",
    "Reasoning Within the Text",
    "Reasoning Beyond the Text",
]

# Distribution of question types per 10-question passage
# Mirrors actual MCAT: ~30% comprehension, ~30% within, ~40% beyond
SKILL_DISTRIBUTION = {
    "Foundations of Comprehension": 3,
    "Reasoning Within the Text": 3,
    "Reasoning Beyond the Text": 4,
}

# AAMC skill proportions (~30/30/40 comprehension/within/beyond). Scaled to any
# question count by skill_breakdown_for(); skill_breakdown_for(10) reproduces
# SKILL_DISTRIBUTION (3/3/4) exactly.
CARS_SKILL_PROPORTIONS = [
    ("Foundations of Comprehension", 0.30),
    ("Reasoning Within the Text", 0.30),
    ("Reasoning Beyond the Text", 0.40),
]


def skill_breakdown_for(num_questions: int) -> dict:
    """Per-skill question counts scaled to num_questions, summing to exactly it.

    Largest-remainder apportionment of the ~30/30/40 AAMC mix (so 6 -> 2/2/2 and
    7 -> 2/2/3). skill_breakdown_for(10) == SKILL_DISTRIBUTION (3/3/4).
    """
    raw = [(name, w * num_questions) for name, w in CARS_SKILL_PROPORTIONS]
    counts = {name: int(x) for name, x in raw}
    remainder = num_questions - sum(counts.values())
    for name, _x in sorted(raw, key=lambda t: t[1] - int(t[1]), reverse=True)[:remainder]:
        counts[name] += 1
    return counts


def balanced_answer_positions(num_questions: int) -> list:
    """A near-uniform shuffle of A/B/C/D of length num_questions.

    Assigning each question a target correct-answer position keeps the answer key
    balanced across A/B/C/D (LLMs otherwise favor B/C) — the same per-question
    balancing the discrete and science-passage generators apply. With per-question
    generation, the assigned letter is enforced one question at a time, which is
    what makes the key balance actually hold (batch generation ignored it).
    """
    seq = (["A", "B", "C", "D"] * (num_questions // 4 + 1))[:num_questions]
    random.shuffle(seq)
    return seq


# --- Per-question difficulty (matches discrete/science: easy 20 / med 50 / hard 30) ---

CARS_DIFFICULTY_WEIGHTS = {"easy": 20, "medium": 50, "hard": 30}


def _pick_cars_difficulty() -> str:
    keys = list(CARS_DIFFICULTY_WEIGHTS.keys())
    return random.choices(keys, weights=list(CARS_DIFFICULTY_WEIGHTS.values()), k=1)[0]


# --- Passage STRUCTURE types ------------------------------------------------
#
# Real AAMC CARS passages are not all single-voice opinion essays: social-science
# passages in particular are often polyphonic literature-reviews surveying several
# named scholars/positions with tensions, reversals, or null results. We assign a
# structure type per passage (weighted by discipline category) and branch the
# passage prompt on it.
STRUCTURE_TYPES = ("single_voice", "multi_position")

# Weight multi_position more heavily for social science, single_voice more
# heavily for humanities — but allow BOTH for each (humanities can be polyphonic).
_STRUCTURE_WEIGHTS = {
    "humanities": {"single_voice": 0.65, "multi_position": 0.35},
    "social_science": {"single_voice": 0.35, "multi_position": 0.65},
}


def pick_structure_type(category: str) -> str:
    """Weighted choice of passage structure for a discipline category.

    `category` is "humanities" or "social_science"; anything else falls back to an
    even split. Returns one of STRUCTURE_TYPES.
    """
    weights = _STRUCTURE_WEIGHTS.get(
        category, {"single_voice": 0.5, "multi_position": 0.5}
    )
    keys = list(weights.keys())
    return random.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def build_cars_question_plan(
    num_questions: int, structure_type: str = "single_voice"
) -> list[dict]:
    """Per-question plan for one CARS passage.

    Each slot carries the assigned skill_type (apportioned ~30/30/40 via
    skill_breakdown_for), a target correct-answer letter (balanced A/B/C/D via
    balanced_answer_positions, now enforced per single-question call), and a
    difficulty (weighted 20/50/30). For multi_position passages, up to two
    Reasoning-Within/Beyond slots are flagged `track_positions` so the question
    generator tests attribution/agreement between the passage's named positions
    and the effect of new information on a specific position. Never flagged for
    single_voice passages.
    """
    breakdown = skill_breakdown_for(num_questions)
    skills: list[str] = []
    for skill, count in breakdown.items():
        skills.extend([skill] * count)
    # Defensive: keep exactly num_questions entries even if rounding drifts.
    skills = skills[:num_questions]
    while len(skills) < num_questions:
        skills.append("Reasoning Beyond the Text")
    random.shuffle(skills)

    positions = balanced_answer_positions(num_questions)

    plan = [
        {
            "skill_type": skills[i],
            "difficulty": _pick_cars_difficulty(),
            "target_answer": positions[i],
            "track_positions": False,
        }
        for i in range(num_questions)
    ]

    if structure_type == "multi_position":
        eligible = [
            i for i, p in enumerate(plan)
            if p["skill_type"] in (
                "Reasoning Within the Text", "Reasoning Beyond the Text"
            )
        ]
        for i in eligible[:2]:
            plan[i]["track_positions"] = True

    return plan


# Per-skill writing guidance for the single-question prompt (distilled from the
# AAMC skill descriptions used by the legacy batch prompt below).
_CARS_SKILL_GUIDANCE = {
    "Foundations of Comprehension": """\
FOUNDATIONS OF COMPREHENSION: Test understanding grounded in immediate sentence \
context. Ask the student to identify the thesis/main point, the purpose of a \
particular sentence or rhetorical label, the meaning of a word/expression from \
context, the text's structure, or the author's tone and its purpose. Keep the \
needed evidence local (a sentence or adjacent sentences).""",
    "Reasoning Within the Text": """\
REASONING WITHIN THE TEXT: Require integrating DISTANT passage components. Ask the \
student to infer the author's message, purpose, assumptions, or bias by combining \
information from multiple parts of the passage; detect a paradox/contradiction \
across sections; distinguish the author's own view from views they paraphrase or \
quote; or evaluate the structure of an argument. Do NOT ask for the student's \
personal opinion — the answer is grounded in the passage.""",
    "Reasoning Beyond the Text": """\
REASONING BEYOND THE TEXT: Require applying passage ideas to a NEW context, or \
assessing the impact of NEW information. Either (a) give a new situation/analogy \
and ask how the passage's ideas apply or how the author would respond, or (b) \
introduce a new fact in the stem and ask how it would strengthen, weaken, or \
otherwise affect the argument. Only one option should be defensible from the \
passage's logic.""",
}

# Extra directive injected into multi_position slots flagged for position tracking.
_POSITION_TRACKING_NOTE = """\
POSITION TRACKING (this passage presents multiple NAMED positions): Write this \
question to test the student's tracking of those positions — for example, which \
named figure/school holds a given view, whether two named figures would agree or \
disagree, or how a new finding would affect ONE specific position (not the passage \
as a whole). The correct answer must hinge on correctly attributing or relating the \
positions, and the distractors should reflect plausible misattributions."""


# Structure-specific guidance injected into the passage generator. single_voice is
# the original single-thesis behavior; multi_position is the polyphonic
# literature-review structure real AAMC CARS (esp. social science) often uses.
_PASSAGE_STRUCTURE_BLOCKS = {
    "single_voice": """\
STRUCTURE — SINGLE VOICE (one author, one thesis):
- Present a clear thesis or central argument carried by a single authorial voice.
- Develop it with nuanced qualifications, counterpoints the author raises and \
addresses, and internal tensions — but the passage advances the author's OWN position.
- Use rhetorical devices, analogies, and references to other thinkers or schools \
of thought in service of that single argument.
- Include both explicitly stated positions and implied/suggested ideas.""",
    "multi_position": """\
STRUCTURE — MULTIPLE POSITIONS (polyphonic literature-review):
- Present 2-4 NAMED scholars, schools, or positions with competing or EVOLVING views.
- Attribute each position clearly to a named figure or school (e.g. "Okonkwo argues…", \
"the structuralists hold…") so the reader can track WHO holds WHICH view.
- Include at least ONE genuine tension among them: a disagreement, a reversal (a \
scholar who revises her own earlier view), a counterexample, or a null/failed-to-replicate \
result.
- The author may adjudicate between the positions OR remain neutral and analytic — but \
the reader must be able to track who holds which position and how the positions relate.
- Do NOT collapse this into a single thesis; the point is the interplay of attributed \
positions. (This is the structure of a survey of competing hypotheses where one \
researcher revises her view and a later finding fails to replicate.)""",
}


def passage_generation_prompt(
    subject: str,
    word_min: int,
    word_max: int,
    structure_type: str = "single_voice",
) -> list[dict]:
    """Build the prompt for generating a CARS-style passage.

    `structure_type` (one of STRUCTURE_TYPES) branches the structural guidance:
    `single_voice` is the original single-thesis essay; `multi_position` produces
    the polyphonic multiple-attributed-positions structure.
    """
    structure_block = _PASSAGE_STRUCTURE_BLOCKS.get(
        structure_type, _PASSAGE_STRUCTURE_BLOCKS["single_voice"]
    )

    system = f"""You are an expert MCAT CARS passage writer working for the AAMC. You write \
passages that closely mimic the style, complexity, and structure found on the actual MCAT \
Critical Analysis and Reasoning Skills (CARS) section.

According to the AAMC content outline, CARS passages have these characteristics:
- They are "relatively short, typically between 500 and 600 words"
- They are "complex, often thought-provoking pieces of writing with sophisticated vocabulary \
and, at times, intricate writing styles"
- They are "multifaceted and focus on the relationships between ideas or theories"
- They come from "the kinds of books, journals, and magazines that college students are \
likely to read"
- No outside scientific or technical knowledge is required to understand them
- "Even those written in a conversational or opinionated style are often multifaceted"

Passage types (match the tone to the subject):
- SOCIAL SCIENCES passages "tend to be more factual and scientific in tone" — they might \
discuss how assumptions help scholars reconstruct patterns, analyze societal trends, or \
examine institutional structures
- HUMANITIES passages "often focus on the relationships between ideas and are more likely to \
be written in a conversational or opinionated style" — consider "the tone and word choice of \
the author in addition to the passage assertions themselves"

{structure_block}

General structural requirements (apply to BOTH structures):
- Have enough layers (claims, evidence, counterpoints, implications) to support several \
challenging questions across all three CARS skills.
- Vary the texture: some sections should state things directly, others should imply or hint.
- Require NO specialized scientific or technical knowledge.

ORIGINALITY & VARIETY (these passages are generated independently and tend to clone each \
other — actively resist that):
- TOPIC: pick a specific, non-obvious thesis or debate within {subject}. Do NOT fall back on \
the field's single most over-used topic (for linguistics, avoid Sapir-Whorf / linguistic \
relativity; for ethics, the trolley problem; for art, "what counts as art" — and so on).
- OPENING: do NOT open with a stock hook such as "Few questions in X have proven as durable…" \
or "For centuries, thinkers have…". Begin in a specific, varied way.
- CLOSING: do NOT end on a formulaic synthesis cliché (e.g. "what emerges is less a verdict \
than a recalibration"). Let the ending follow from this passage's particular argument.

Word count: EXACTLY {word_min}-{word_max} words. This is critical.

Respond with ONLY a JSON object:
{{
  "passage_text": "The full passage text here...",
  "subject": "{subject}"
}}"""

    user = f"""Write an MCAT CARS passage on the subject of {subject}.

The passage should read like an excerpt from an academic book, journal article, or \
sophisticated magazine piece that a college student might encounter. Follow the STRUCTURE \
guidance above for this passage. It should present enough nuance and complexity to support \
several challenging multiple-choice questions.

Remember: the passage must be between {word_min} and {word_max} words, and should NOT \
require any specialized scientific or technical knowledge to understand."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def cars_questions_prompt(passage_text: str, num_questions: int = 10) -> list[dict]:
    """Build the prompt for generating questions about a CARS passage."""

    breakdown = skill_breakdown_for(num_questions)
    skill_breakdown = "\n".join(
        f"  - {skill}: {count} question(s)"
        for skill, count in breakdown.items()
    )
    target_positions = balanced_answer_positions(num_questions)
    positions_str = ", ".join(
        f"Q{i} -> {pos}" for i, pos in enumerate(target_positions, 1)
    )

    system = f"""You are an expert MCAT CARS question writer working for the AAMC. You create \
questions that test the three Critical Analysis and Reasoning Skills defined by the AAMC.

Question type distribution (for {num_questions} questions):
{skill_breakdown}

Detailed skill descriptions (from the AAMC content outline):

FOUNDATIONS OF COMPREHENSION:
These questions focus on understanding from immediate sentence context. They should ask \
the student to:
- Identify the author's thesis, main point, or central theme
- Recognize the purpose of particular sentences or rhetorical labels ("for example," \
"therefore," "consequently")
- Interpret the meaning of words or expressions using sentence context
- Identify how the author structured the text (cause-and-effect, chronological, \
point-and-counterpoint)
- Recognize the author's tone (humorous, authoritative, satirical) and its purpose \
(persuade, instruct, inform, entertain)
Example question types: "The author's primary purpose in this passage is...", \
"As used in paragraph 2, the word X most nearly means...", \
"Which of the following best summarizes the main idea?"

REASONING WITHIN THE TEXT:
These require integrating DISTANT passage components into a complex interpretation. They \
differ from Comprehension in scope — they require synthesizing across the whole passage. \
They should ask the student to:
- Infer the author's message, purpose, position, beliefs, assumptions, or bias by \
integrating information from multiple parts of the passage
- Detect paradoxes, contradictions, or inconsistencies across different passage sections
- Identify whether the author presents their own perspective vs. others' views through \
summaries or paraphrases
- Evaluate arguments: examine evidence, relevance, faulty causality, credibility of sources
- Analyze the author's language, stance, and purpose beneath surface-level meaning
- Identify "vague or evasive terms or language that sounds self-aggrandizing, overblown, \
or otherwise suspect"
Important: These questions do NOT ask for the student's personal opinion. Even if the \
student disagrees with the author, the correct answer is based on what the passage says.
Example question types: "The author would most likely agree with which of the following?", \
"Which assumption underlies the author's argument in the third paragraph?", \
"The author's discussion of X serves primarily to..."

REASONING BEYOND THE TEXT:
These require applying passage ideas to new contexts OR assessing the impact of new \
information on the passage. Two sub-types:
1. APPLICATION/EXTRAPOLATION: The passage is the "given" and the question provides a \
new context. Ask how passage ideas apply to a new situation, what analogy fits, how the \
author would respond to a hypothetical. "Each response option yields a different result, \
but only one is defensible based on the passage."
2. INCORPORATION: Introduce new information in the question and ask how it affects the \
passage's argument. "Does the new information support or contradict the passage? Could it \
coexist, or would it negate an aspect of the argument? What modifications would be needed?"
Example question types: "If a study showed X, how would this affect the author's argument?", \
"Which situation is most analogous to the relationship described in the passage?", \
"Which new finding, if true, would most weaken the author's central claim?", \
"The author's argument could best be applied to which of the following scenarios?"

General rules:
- Each question's "choices" object must have EXACTLY four keys "A", "B", "C", and "D" and \
nothing else — no extra keys (no "E", no "C2", no duplicates), no missing/empty keys; each \
choice a non-empty string. Exactly ONE answer is unambiguously correct.
- ANSWER-KEY BALANCE: each question's correct-answer position is ASSIGNED in the task below. \
Write the four choices so the assigned letter is the single correct answer, and do not let \
the correct option be guessable from its length, specificity, or phrasing.
- Questions must be answerable SOLELY from the passage — no outside knowledge required
- Distractors should be plausible and represent common misreadings or partial understandings
- Answer choices should be roughly similar in length (no giveaway long correct answers)
- Explanations MUST reference specific parts of the passage to justify the correct answer
- Across the {num_questions} questions, cover a variety of question formats: main idea, \
detail, inference, application, tone, structure, strengthen/weaken, analogy, \
new-information-impact

Respond with ONLY a JSON array of EXACTLY {num_questions} question objects, in order:
[
  {{
    "skill_type": "Foundations of Comprehension",
    "stem": "Question text...",
    "choices": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "A",
    "explanation": "Explanation referencing specific passage content..."
  }}
]"""

    user = f"""Read this CARS passage and generate {num_questions} multiple-choice questions.

PASSAGE:
{passage_text}

Generate EXACTLY {num_questions} questions, in order, following the skill-type distribution \
above. Assign each question's correct answer to the position given here — the i-th question \
you output MUST set "correct_answer" to the i-th letter: {positions_str}

Make them challenging and realistic — they should require careful reading and analysis, \
not surface-level comprehension. For Reasoning Beyond the Text questions, introduce \
genuinely novel scenarios or information that test whether the student can extend or \
challenge the passage's ideas."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def cars_question_prompt(
    passage_text: str,
    plan: dict,
    previous_stems: list = None,
    structure_type: str = "single_voice",
) -> list[dict]:
    """Build the prompt for generating ONE CARS question about a passage.

    `plan` carries the assigned skill_type, difficulty, target_answer position,
    and a track_positions flag. `previous_stems` are stems already accepted FOR
    THIS PASSAGE — fed back for within-passage diversity (cover a different
    aspect). Mirrors the science/discrete single-question generators.
    """
    skill_type = plan["skill_type"]
    difficulty = plan["difficulty"]
    target_answer = plan["target_answer"]
    track_positions = plan.get("track_positions", False)

    skill_guidance = _CARS_SKILL_GUIDANCE.get(
        skill_type, _CARS_SKILL_GUIDANCE["Reasoning Beyond the Text"]
    )

    difficulty_guidance = {
        "easy": (
            "EASY: direct comprehension — the answer is found or lightly inferred "
            "from a specific sentence or its immediate context. Distractors still "
            "plausible, but a careful reader locates the basis quickly."
        ),
        "medium": (
            "MEDIUM: typical CARS difficulty — integrate a couple of statements, or "
            "read the author's tone/intent/structure with care."
        ),
        "hard": (
            "HARD: subtle inference — integrate DISTANT passage components, draw a "
            "fine distinction between two close answer choices, or reason about how "
            "new information bears on a specific claim. Still answerable solely from "
            "the passage; no obscurity for its own sake."
        ),
    }

    diversity_str = ""
    if previous_stems:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(previous_stems, 1))
        diversity_str = (
            "\n\nQuestions already written for THIS passage test the following. Do NOT "
            "duplicate them — write a question on a different aspect of the passage, a "
            "different part of the argument, or a different relationship between ideas:\n"
            f"{numbered}"
        )

    position_block = f"\n\n{_POSITION_TRACKING_NOTE}" if track_positions else ""

    system = f"""You are an expert MCAT CARS question writer working for the AAMC. You write \
ONE question at a time that tests a specific Critical Analysis and Reasoning Skill.

THIS QUESTION'S SKILL TYPE is "{skill_type}":
{skill_guidance}

Target difficulty: {difficulty.upper()}
{difficulty_guidance[difficulty]}{position_block}

General rules:
- The question must be answerable SOLELY from the passage — no outside knowledge required.
- The question has EXACTLY four options: A, B, C, and D. There is no option E (or F, etc.). \
The "choices" object must have EXACTLY four keys "A", "B", "C", and "D" and nothing else — \
no extra keys, no missing/empty keys; each choice a non-empty string. Exactly ONE answer is \
unambiguously correct.
- ANSWER-KEY BALANCE: the correct answer for THIS question MUST be option {target_answer}. \
This is not negotiable — make option {target_answer} the genuinely best, fully substantive \
answer, and write the other three options as plausible distractors. Do not let option \
{target_answer} be guessable from length, specificity, or phrasing, and do NOT quietly \
relocate the correct content to another letter (writers tend to under-use later letters such \
as D as the key — resist that bias for whichever letter is assigned here). Option \
{target_answer} must be a fully substantive best answer, never a weak throwaway or an \
"all/none of the above" filler.
- Distractors should be plausible — common misreadings or partial understandings.
- Answer choices should be roughly similar in length and specificity (no giveaway long \
correct answer).
- The explanation is FINAL, PUBLISHED content shown to a student. It must reference specific \
parts of the passage to justify the correct answer and briefly why the distractors are wrong. \
It must reference ONLY options A, B, C, and D (never a fifth option or one that does not \
exist). NEVER mention, discuss, or reference an option E or any option beyond D — no such \
option exists. Do NOT write phrases like "Option E", "E is...", or "E overstates...". If you \
find yourself about to reference a fifth option, stop: there are only four. The explanation \
must also contain NO meta-commentary about the question itself — no hedging, disclaimers, or \
remarks about the item's format, validity, or completeness. When the explanation refers to an \
answer choice, name it "Choice A/B/C/D" and bold that whole phrase in markdown — e.g. \
"**Choice B** incorrectly attributes the rise to the author...", "**Choice A** misreads the \
second paragraph", "**Choice D** is correct because...". Bold the entire "Choice X" phrase, \
never a bare letter, and do NOT bold letters used as experimental labels (Series A, Group B) \
or variables — only explicit "Choice X" references.

Respond with ONLY a JSON object that has EXACTLY these keys. Each value is described in \
angle brackets — produce a real value matching the description; do NOT output the \
angle-bracket text itself:
{{
  "skill_type": <exactly "{skill_type}">,
  "stem": <string: the full question text>,
  "choices": <object with EXACTLY keys "A","B","C","D" and NO others (no "E"), each a non-empty string>,
  "correct_answer": <one of "A","B","C","D"; for this question must be "{target_answer}">,
  "explanation": <string: published explanation referencing ONLY options A-D (never an option E), no meta-commentary>,
  "difficulty": <exactly "{difficulty}">
}}

STRICT OUTPUT CONTRACT — follow EXACTLY or the response is rejected:
- "choices" must contain EXACTLY the keys A, B, C, D and nothing else — no "E" or any further \
option.
- The "explanation" must not reference any option key that is not in {{A, B, C, D}}. There is \
no option E; never write "Option E" or reason about a fifth choice.
Output the raw JSON object only — no markdown fences, no preamble, no trailing notes."""

    user = f"""Read this CARS passage and write ONE multiple-choice question about it.

PASSAGE:
{passage_text}

Write ONE question.
Skill type: {skill_type}
Target difficulty: {difficulty}
Target correct-answer position: {target_answer} (MANDATORY — option {target_answer} must be \
the single correct answer; do not shift correctness to another letter). There are exactly \
four options A-D and no option E.{diversity_str}

Make it challenging and realistic — it should require careful reading and analysis, not \
surface-level comprehension. If this is a Reasoning Beyond the Text question, introduce a \
genuinely novel scenario or piece of information that tests whether the student can extend \
or challenge the passage's ideas."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def cars_adversarial_review_prompt(
    passage_text: str,
    question_data: dict,
) -> list[dict]:
    """Build the prompt for adversarial review of a CARS question."""

    system = """You are a rigorous MCAT CARS question reviewer working for the AAMC. Your job \
is to find flaws before questions reach students. Be critical and thorough.

The AAMC tests three CARS skills:
- Foundations of Comprehension (30%): basic understanding, word meaning, author's purpose
- Reasoning Within the Text (30%): integrating distant components, evaluating arguments, \
detecting bias/assumptions
- Reasoning Beyond the Text (40%): applying ideas to new contexts, assessing impact of new info

Check for:
1. ANSWERABILITY: Can this question be answered SOLELY from the passage? Does it require \
outside knowledge? (This is the #1 rule of CARS — everything must come from the passage.)
2. ACCURACY: Is the stated correct answer actually the BEST answer based on the passage? \
Could a knowledgeable reader make a strong case for a different answer?
3. AMBIGUITY: Could more than one answer be defensibly correct given the passage content? \
Are any distractors too close to the correct answer?
4. DISTRACTORS: Are wrong answers plausible misreadings or partial understandings? Or are \
they obviously wrong / absurd? (Good distractors on CARS represent things a careless reader \
might conclude.)
5. SKILL ALIGNMENT: Does the question actually test the stated skill type? \
A Comprehension question should focus on immediate sentence context. \
A Reasoning Within question should require integrating distant passage components. \
A Reasoning Beyond question should introduce a genuinely new context or information.
6. PASSAGE SUPPORT: Does the explanation correctly reference specific passage content? \
Can you trace the correct answer back to something in the passage?
7. ANSWER BALANCE: Are choices roughly similar in length and specificity?

Respond with ONLY a JSON object:
{
  "passed": true,
  "issues": [],
  "reasoning": "Brief assessment"
}

Set "passed" to false if ANY significant issue is found."""

    user = f"""Review this CARS question against the passage.

PASSAGE:
{passage_text}

QUESTION:
Skill type: {question_data['skill_type']}
Stem: {question_data['stem']}
A) {question_data['choices']['A']}
B) {question_data['choices']['B']}
C) {question_data['choices']['C']}
D) {question_data['choices']['D']}
Correct answer: {question_data['correct_answer']}
Explanation: {question_data['explanation']}

Find any flaws. Be especially strict about whether the correct answer is truly the BEST \
answer and whether the question actually tests the stated CARS skill type."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def cars_blind_solve_prompt(passage_text: str, question_data: dict) -> list[dict]:
    """Build the prompt for blind-solving a CARS question."""

    system = """You are an MCAT expert taking the CARS section. Read the passage carefully \
and answer the question based ONLY on what is in the passage.

Important CARS strategies:
- The correct answer is always supported by the passage text
- Do not bring in outside knowledge
- Pay attention to the author's tone, word choice, and rhetorical strategy
- For "Reasoning Beyond" questions, apply the passage's logic to the new scenario
- Eliminate answers that are too extreme, not supported, or contradict the passage

Think step by step, then select the SINGLE BEST answer.

Respond with ONLY a JSON object:
{
  "chosen_answer": "B",
  "confidence": "high",
  "reasoning": "Brief explanation referencing the passage"
}

confidence should be "high", "medium", or "low".

OUTPUT FORMAT — ABSOLUTE: Respond with ONLY the raw JSON object specified above and NOTHING \
else. Do NOT show your reasoning or working in prose outside the JSON. Do NOT write 'I need \
to', 'Let me', or any preamble. Put any brief reasoning INSIDE the JSON's reasoning field if \
one exists; otherwise omit it. Your entire response must start with { and end with } and be \
valid parseable JSON."""

    user = f"""Read this passage and answer the question.

PASSAGE:
{passage_text}

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


def passage_review_prompt(
    passage_text: str,
    word_min: int,
    word_max: int,
    structure_type: str = "single_voice",
) -> list[dict]:
    """Review a CARS passage for quality before generating questions.

    `structure_type` adjusts the argumentation/multifaceted checks: a
    `multi_position` passage must NOT be penalized for presenting multiple
    attributed views instead of one thesis — that is correct for that type. For
    multi_position, the review instead checks that the positions are clearly
    attributable and the relationships between them are coherent.
    """
    if structure_type == "multi_position":
        argumentation_check = (
            "2. STRUCTURE (MULTIPLE POSITIONS): This passage is INTENTIONALLY polyphonic — "
            "it presents several named scholars/schools/positions. Do NOT penalize it for "
            "presenting multiple views instead of a single thesis; that is correct for this "
            "type. Instead check: are the positions clearly ATTRIBUTABLE (the reader can tell "
            "who holds which view)? Are the RELATIONSHIPS between positions coherent (genuine "
            "agreement/disagreement/tension/reversal, not muddled)? Is there real interpretive "
            "complexity to reason about?"
        )
        multifaceted_check = (
            "6. MULTIFACETED: Does it genuinely track relationships BETWEEN the attributed "
            "positions/ideas (not just list disconnected facts)? Multiple perspectives are "
            "expected and good here."
        )
    else:
        argumentation_check = (
            "2. ARGUMENTATION: Does it present a clear thesis or argument (not just describe "
            "facts)? Does it have internal complexity — qualifications, counterpoints, tensions?"
        )
        multifaceted_check = (
            "6. MULTIFACETED: Does it focus on \"relationships between ideas or theories\" "
            "rather than just describing a single concept? Are there multiple perspectives or "
            "interpretive layers?"
        )

    system = f"""You are an MCAT CARS passage quality reviewer working for the AAMC. Evaluate \
whether this passage meets the standards for the CARS section.

According to the AAMC, CARS passages should be:
- "Complex, often thought-provoking pieces of writing with sophisticated vocabulary"
- "Multifaceted and focus on the relationships between ideas or theories"
- Similar to "the kinds of books, journals, and magazines that college students are likely to read"
- Answerable without "additional coursework or specific knowledge"
- If social sciences: "more factual and scientific in tone"
- If humanities: "focus on the relationships between ideas," "more likely to be written in a \
conversational or opinionated style"
- CARS passages take MANY shapes: some are single-author essays advancing one thesis; others \
are polyphonic surveys of several attributed positions. BOTH are valid.

Check for:
1. WORD COUNT: Is it between {word_min} and {word_max} words?
{argumentation_check}
3. QUESTION SUPPORT: Is it complex enough to support several questions across all three CARS \
skill types (comprehension, reasoning within, reasoning beyond)? Are there enough layers \
for application and incorporation questions?
4. INDEPENDENCE: Can it be understood without specialized scientific/technical knowledge? \
A reader should need NO outside information.
5. SOPHISTICATION: Is the writing at an appropriate academic level? Does it use \
sophisticated vocabulary naturally? Does it have an identifiable voice and tone?
{multifaceted_check}

Respond with ONLY a JSON object:
{{
  "passed": true,
  "word_count": 547,
  "issues": [],
  "reasoning": "Brief assessment"
}}"""

    user = f"""Evaluate this CARS passage for AAMC quality standards:

{passage_text}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
