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


def passage_generation_prompt(subject: str, word_min: int, word_max: int) -> list[dict]:
    """Build the prompt for generating a CARS-style passage."""

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

Structural requirements:
- Present a clear thesis or central argument with supporting reasoning
- Include nuanced qualifications, counterpoints, or internal tensions
- Use rhetorical devices, analogies, references to other thinkers or schools of thought
- Have enough layers (claims, evidence, counterpoints, implications) to support 10 questions
- Include both explicitly stated positions and implied/suggested ideas
- Vary the structure: some sections should state things directly, others should imply or hint

Word count: EXACTLY {word_min}-{word_max} words. This is critical.

Respond with ONLY a JSON object:
{{
  "passage_text": "The full passage text here...",
  "subject": "{subject}"
}}"""

    user = f"""Write an MCAT CARS passage on the subject of {subject}.

The passage should read like an excerpt from an academic book, journal article, or \
sophisticated magazine piece that a college student might encounter. It should present an \
original argument or analysis with enough nuance and complexity to support 10 challenging \
multiple-choice questions.

Remember: the passage must be between {word_min} and {word_max} words, and should NOT \
require any specialized scientific or technical knowledge to understand."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def cars_questions_prompt(passage_text: str, num_questions: int = 10) -> list[dict]:
    """Build the prompt for generating questions about a CARS passage."""

    skill_breakdown = "\n".join(
        f"  - {skill}: {count} questions"
        for skill, count in SKILL_DISTRIBUTION.items()
    )

    system = f"""You are an expert MCAT CARS question writer working for the AAMC. You create \
questions that test the three Critical Analysis and Reasoning Skills defined by the AAMC.

Question type distribution (for {num_questions} questions):
{skill_breakdown}

Detailed skill descriptions (from the AAMC content outline):

FOUNDATIONS OF COMPREHENSION (3 questions):
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

REASONING WITHIN THE TEXT (3 questions):
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

REASONING BEYOND THE TEXT (4 questions):
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
- Each question must have exactly 4 choices (A, B, C, D), one unambiguously correct
- Questions must be answerable SOLELY from the passage — no outside knowledge required
- Distractors should be plausible and represent common misreadings or partial understandings
- Answer choices should be roughly similar in length (no giveaway long correct answers)
- Explanations MUST reference specific parts of the passage to justify the correct answer
- Across the {num_questions} questions, cover a variety of question formats: main idea, \
detail, inference, application, tone, structure, strengthen/weaken, analogy, \
new-information-impact

Respond with ONLY a JSON array of question objects:
[
  {{
    "skill_type": "Foundations of Comprehension",
    "stem": "Question text...",
    "choices": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "C",
    "explanation": "Explanation referencing specific passage content..."
  }}
]"""

    user = f"""Read this CARS passage and generate {num_questions} multiple-choice questions.

PASSAGE:
{passage_text}

Generate exactly {num_questions} questions with the specified skill type distribution. \
Make them challenging and realistic — they should require careful reading and analysis, \
not surface-level comprehension. For Reasoning Beyond the Text questions, introduce \
genuinely novel scenarios or information that test whether the student can extend or \
challenge the passage's ideas."""

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

confidence should be "high", "medium", or "low"."""

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


def passage_review_prompt(passage_text: str, word_min: int, word_max: int) -> list[dict]:
    """Review a CARS passage for quality before generating questions."""

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

Check for:
1. WORD COUNT: Is it between {word_min} and {word_max} words?
2. ARGUMENTATION: Does it present a clear thesis or argument (not just describe facts)? \
Does it have internal complexity — qualifications, counterpoints, tensions?
3. QUESTION SUPPORT: Is it complex enough to support 10 questions across all three CARS \
skill types (comprehension, reasoning within, reasoning beyond)? Are there enough layers \
for application and incorporation questions?
4. INDEPENDENCE: Can it be understood without specialized scientific/technical knowledge? \
A reader should need NO outside information.
5. SOPHISTICATION: Is the writing at an appropriate academic level? Does it use \
sophisticated vocabulary naturally? Does it have an identifiable authorial voice and tone?
6. MULTIFACETED: Does it focus on "relationships between ideas or theories" rather than \
just describing a single concept? Are there multiple perspectives or interpretive layers?

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
