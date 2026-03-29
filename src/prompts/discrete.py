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


def _pick_skill() -> str:
    """Randomly select a skill category weighted by AAMC distribution."""
    keys = list(SKILL_WEIGHTS.keys())
    weights = list(SKILL_WEIGHTS.values())
    return random.choices(keys, weights=weights, k=1)[0]


def generation_prompt(topic_data: dict) -> list[dict]:
    """Build the prompt for generating a single discrete MCAT question.

    Args:
        topic_data: A single topic entry from topics.json
    """
    subtopics_str = ""
    if topic_data.get("subtopics"):
        subtopics_str = f"\nSpecific subtopics to potentially test: {', '.join(topic_data['subtopics'])}"

    skill_key = _pick_skill()
    skill_label = SKILL_LABELS[skill_key]

    # Skill-specific generation guidance based on AAMC descriptions
    skill_guidance = {
        "skill_1": """\
This question should test Skill 1: Knowledge of Scientific Concepts and Principles.
The question should ask the student to:
- Recognize or identify a scientific concept, principle, or relationship from an example or scenario
- Identify relationships between closely related concepts (e.g., written vs. graphical representations)
- Identify examples or observations that illustrate a scientific principle
- Use a given mathematical equation to solve a problem
- Recognize a concept shown in a diagram, graph, or structural formula

Example formats: "What type of functional group is formed when...", \
"Which of the following best describes the relationship between...", \
"A student observes X. This is an example of..."

Do NOT write a simple recall/definition question. Even Skill 1 questions should present \
a scenario, example, or representation that the student must interpret.""",

        "skill_2": """\
This question should test Skill 2: Scientific Reasoning and Problem-Solving.
The question should ask the student to:
- Use scientific theories or models to explain observations or make predictions
- Evaluate the validity or credibility of a scientific explanation
- Evaluate arguments about cause and effect using scientific knowledge
- Bring together theory, observations, and evidence to draw conclusions
- Recognize findings that challenge or invalidate a theory or model
- Determine and use scientific formulas to solve a multi-step problem

Example formats: "A researcher observes X. Which explanation best accounts for...", \
"Based on the principle of Y, what would happen if...", \
"Which finding would most weaken the hypothesis that...", \
"Given the following data, calculate..."

Present a scenario that requires REASONING, not just recall. The student should need to \
apply a principle to a novel situation or evaluate competing explanations.""",

        "skill_3": """\
This question should test Skill 3: Reasoning About the Design and Execution of Research.
The question should ask the student to:
- Identify independent, dependent, and confounding variables in a described experiment
- Evaluate the appropriateness of a research method, tool, or measurement
- Identify limitations or flaws in a research study design
- Distinguish between correlational and causal claims
- Identify what controls are needed and why
- Reason about ethical issues in research

Example formats: "Researchers conducted a study where... What is the independent variable?", \
"Which modification to the experimental design would best control for...", \
"A study finds a correlation between X and Y. Which conclusion is most justified?", \
"Which aspect of this study design most threatens its internal validity?"

You MUST describe a specific experiment or study in the question stem. The student should \
evaluate the research design, not just recall a concept.""",

        "skill_4": """\
This question should test Skill 4: Data-Based and Statistical Reasoning.
The question should ask the student to:
- Interpret patterns in data presented in a table, graph, or figure (describe the data in text)
- Use measures of central tendency (mean, median, mode) or dispersion (range, SD)
- Reason about random vs. systematic error
- Interpret statistical significance or confidence intervals
- Use data to explain relationships between variables or draw conclusions
- Identify conclusions that are or are not supported by given results

Example formats: "A table shows the following results... What conclusion is supported?", \
"If the mean is X and the standard deviation is Y, approximately what percentage...", \
"Researchers measure Z across four groups and obtain the following values... \
Which comparison is statistically meaningful?", \
"Based on the data, which relationship between the variables is most likely?"

You MUST present specific data (numbers, values, trends) in the question stem. The student \
should reason FROM the data, not just know what a statistical concept means.""",
    }

    system = f"""You are an expert MCAT question writer for the Association of American Medical \
Colleges (AAMC). You create questions that match the difficulty, style, and cognitive demands of \
the actual MCAT exam.

The MCAT tests four Scientific Inquiry and Reasoning Skills across its science sections. \
You are writing a question that tests a specific skill.

{skill_guidance[skill_key]}

General question-writing rules:
- Each question must have exactly 4 answer choices (A, B, C, D)
- Exactly ONE answer must be unambiguously correct
- Distractors (wrong answers) must be plausible — they should represent common misconceptions, \
partially correct reasoning, or errors students commonly make
- Answer choices should be roughly similar in length and specificity (a correct answer that is \
much longer or more detailed than the others is a test-taking giveaway)
- The question should require introductory college-level knowledge, not graduate-level obscurity
- Mathematical questions should use algebra, logarithms, basic trig, or dimensional analysis \
(no calculus). A periodic table is available during the exam.
- Include a thorough explanation of why the correct answer is right AND why each distractor is wrong

Respond with ONLY a JSON object in this exact format, no other text:
{{
  "stem": "The question text here",
  "choices": {{
    "A": "First choice",
    "B": "Second choice",
    "C": "Third choice",
    "D": "Fourth choice"
  }},
  "correct_answer": "B",
  "explanation": "Explanation of correct answer and why others are wrong",
  "difficulty": "medium",
  "subtopics_tested": ["relevant subtopic 1"],
  "skill_tested": "{skill_label}"
}}"""

    user = f"""Generate an MCAT-style discrete question for:

Section: {topic_data['section']}
Content Category: {topic_data['content_category']}
Topic Group: {topic_data['topic_group']}
Topic: {topic_data['topic']}
Discipline: {topic_data.get('discipline', 'N/A')}{subtopics_str}

Target Skill: {skill_label}

Write a challenging, realistic MCAT question testing this specific skill. \
The question should be at an appropriate difficulty for the actual exam — \
not a simple definition lookup, but also not impossibly obscure. \
Present a scenario, experiment, data, or problem that requires the student \
to think, not just remember."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def adversarial_review_prompt(question_data: dict, topic_data: dict) -> list[dict]:
    """Build the prompt for adversarial review of a generated question."""

    system = """You are a rigorous MCAT question reviewer and quality assurance expert working \
for the AAMC. Your job is to find flaws in MCAT questions before they go to students. Be \
critical and thorough.

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

confidence should be "high", "medium", or "low"."""

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
