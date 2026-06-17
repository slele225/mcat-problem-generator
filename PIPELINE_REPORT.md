# MCAT Question-Generation Pipeline — Review Report

*Self-contained snapshot of the `mcat-gen` repository: architecture, every prompt verbatim, validation logic, output schema, sample outputs, and distribution statistics. Generated for external review; no code was modified.*

The numbers and samples in this report come from the three **production runs** stored under `runs/`:
`runs/prod_discrete/`, `runs/prod_science/`, `runs/prod_cars/`. All three were generated with the **Opus** experiment config (`model: claude-opus-4-8`), confirmed by the metrics files. Note that the repo's default `config.yaml` at the root currently points at `claude-sonnet-4-6`; the production data was produced with `configs/opus.yaml`.

---

## 1. Pipeline Overview

### 1.1 Entry point and run configuration

Everything runs through `src/main.py` (`python -m src.main`). Key flags:

| Flag | Effect |
|---|---|
| *(none)* | Run all three pipelines: discrete, CARS, science-passage |
| `--discrete-only` / `--cars-only` / `--science-passage-only` | Restrict to one pipeline |
| `--config my.yaml` | Use an alternate config (e.g. `configs/opus.yaml`) |
| `--topic-ids ID1,ID2` | Targeted test: build ONE science passage from exactly those topic IDs, bypassing clustering/shuffle/checkpoints, then stop |
| `--render-figures` / `--force-render` | Render figures from `science_passages.jsonl` (no API calls) |
| `--max-topics N` | Cap topics/passages processed this run |
| `--run-name NAME` | Write all artifacts into `runs/<NAME>/` (questions, checkpoints, metrics, `run.log`) |
| `--seed N` | Fix RNG so topic selection is reproducible |
| `--stats` / `--reset` / `--recount` | Show checkpoint stats / clear checkpoints / rebuild discrete checkpoint from its jsonl |

On startup `async_main` loads the config, optionally redirects output into `runs/<run_name>/`, opens an `LLMClient`, runs a `health_check` against the Anthropic API, fixes the RNG seed if given, then dispatches to the selected pipelines.

The run is configured via YAML (`src/config.py` → dataclasses `Config`, `DiscreteConfig`, `CARSConfig`, `SciencePassageConfig`). Production config (`configs/opus.yaml`):

```yaml
model: claude-opus-4-8
checker_model: claude-haiku-4-5-20251001     # blind-solve checker only
discrete:
  questions_per_topic: 8
  max_retries: 3
  temperature_generate: 0.8
  temperature_validate: 0.3
  batch_size: 5
cars:
  passages_per_topic: 30          # TOTAL passages this run
  questions_per_passage_range: [5, 7]
  passage_word_range: [500, 600]
  max_retries: 3
  temperature_generate: 0.8
  temperature_validate: 0.3
  batch_size: 5
  humanities_subjects: [Architecture, Art, Dance, Ethics, Literature, Music,
    Philosophy, Popular Culture, Religion, Theater, Studies of Diverse Cultures]
  social_science_subjects: [Anthropology, Archaeology, Economics, Education,
    Geography, History, Linguistics, Political Science, Population Health,
    Psychology, Sociology, Studies of Diverse Cultures]
science_passage:
  passages_per_topic_cluster: 1
  questions_per_passage_range: [4, 7]
  enable_figures: true            # SMILES via RDKit + plots via matplotlib
```

The root `config.yaml` is identical in structure but uses `model: claude-sonnet-4-6`, `discrete.questions_per_topic: 2`, `cars.passages_per_topic: 1`, `cars.questions_per_passage: 10`, and `science_passage.enable_figures: false` with explicit `skill_weights: {skill_1:15, skill_2:35, skill_3:25, skill_4:25}`.

### 1.2 The three pipelines and the common 3-stage flow

Every question, in every pipeline, passes through the same conceptual funnel:

```
  GENERATION  →  ADVERSARIAL REVIEW  →  BLIND-SOLVE VALIDATION  →  accept / retry
   (main model)     (main model)          (checker model, cheaper)
```

The unit of work, topics file, and passage layer differ per pipeline. Topics come from `mcat_topics.json` (each entry: `topic_id`, `section`, `content_category`, `topic_group`, `topic`, `discipline`, `subtopics`, optional `section_focus`).

**A. Discrete pipeline** (`src/pipelines/discrete.py`)
- Unit of work: a **topic** (any non-CARS topic). Produces `questions_per_topic` standalone questions.
- Topics are shuffled (seeded if `--seed`), optionally capped by `--max-topics`, and processed one at a time.
- For each topic, questions are generated **sequentially** so each new question sees the stems already accepted *for this topic* (within-topic diversity feedback).
- Each question: generate → validate (adversarial review + blind solve, run concurrently) → accept if **both** pass; otherwise retry up to `max_retries`. Accepted questions are appended to `discrete_questions.jsonl`; progress is checkpointed per topic.

**B. CARS pipeline** (`src/pipelines/cars.py`)
- Unit of work: a **passage**. `passages_per_topic` total passages, each on a subject picked **50/50** from the humanities vs. social-science subject lists (`random.random() < 0.5`, then `random.choice`).
- Per passage: (1) generate passage → local word-count gate → **LLM passage quality review** (retry until `passed`); (2) draw question count from `questions_per_passage_range`; (3) generate **all questions at once**, parse each, validate each individually (adversarial review + blind solve), keep those that pass; regenerate the whole batch if too few pass.
- Passages run in concurrent batches (`batch_size`). Output: `cars_passages.jsonl`.

**C. Science-passage pipeline** (`src/pipelines/science_passage.py`)
- Unit of work: a **passage** built for a **cluster of 2–3 related topics** (grouped by `(section, content_category)`, kept topically coherent by `topic_group`, chunked into 2s/3s by `_chunk_2_3`).
- Per passage: generate passage (prose + optional markdown table + optional figures) → word-count gate → figure validation → **LLM passage review** → then draw a question count from `questions_per_passage_range` and build a **per-question plan** (skill, answer_basis, difficulty, target answer letter).
- Questions are generated **sequentially** (within-passage diversity feedback), each validated with adversarial review + blind solve **plus** a separate, lenient `answer_basis_ok` check. Accept if quality passes **and** basis is OK.
- **Figure enforcement** (only when `enable_figures: true`): certain content-category codes structurally require a figure type (`5A,5E → plot`; `5B,5D → smiles`). If a required type is absent from both passage and every question, the whole passage set is regenerated with an escalating directive, up to `max_retries`; after that it is accepted anyway (logged at ERROR) so the miss rate is measurable.
- Output: `science_passages.jsonl` (+ `figures/` and `figures_manifest.json` from the separate render pass).

### 1.3 Models and settings per stage

`MODEL_PRICING` and the per-stage `on_usage` callbacks (`src/metrics.py`) price each call by the **actual model that ran it**.

| Stage | Model (prod) | Temperature | max_tokens |
|---|---|---|---|
| Discrete generation | `claude-opus-4-8` (`config.model`) | `temperature_generate` = 0.8 | 1024 |
| CARS passage generation | main model | 0.8 | 2048 |
| CARS questions generation | main model | 0.8 | 4096 |
| Science passage generation | main model | 0.8 | 2048 |
| Science question generation | main model | 0.8 | 1024 |
| Adversarial review (all pipelines) | **main model** | `temperature_validate` = 0.3 | 2048 (default) |
| Passage quality review (CARS / science) | main model | 0.3 | 2048 |
| **Blind solve** (discrete & science) | **`checker_model` = `claude-haiku-4-5-20251001`** | 0.3 | 2048 |
| Blind solve (CARS) | main model* | 0.3 | 2048 |

*\*Note: the CARS pipeline's `validate_cars_question` does **not** pass `model=checker_model` to the blind-solve call, so CARS blind-solve runs on the main model, unlike discrete/science which route it to the cheaper checker. This is a real asymmetry in the code.*

Pricing per million tokens (USD), keyed by model: Opus 4.8 = $5 in / $25 out; Sonnet 4.6 = $3 / $15; Haiku 4.5 = $1 / $5.

`src/llm_client.py` is an async Anthropic wrapper: splits system messages out as the top-level `system` param, retries transient errors (429/5xx/timeouts) with exponential backoff + jitter, and learns at runtime that some models (e.g. Opus 4.8) reject `temperature` (drops it on a 400 and remembers). `parse_json_response` strips markdown fences and falls back to locating the first `{…}` or `[…]`.

### 1.4 How topic / section / question-type / difficulty / skill get assigned

- **Section & topic**: from the topic entry in `mcat_topics.json`. Discrete questions inherit the topic's `section`, `content_category`, `topic_group`, `topic` directly. Science passages take section/category from the first topic in the cluster.
- **Question type** (`answer_basis`, science only): assigned by `build_question_plan` — seeds the first slots with one of each of `from_passage`, `apply_knowledge`, `data_interpretation` (shuffled), then fills the rest by weighted pick (`from_passage` 35 / `apply_knowledge` 40 / `data_interpretation` 25). CARS "question type" is the **skill_type** (see below).
- **Difficulty**: drawn per question from weights **easy 20 / medium 50 / hard 30** (discrete and science).
- **Skill**:
  - *Discrete*: drawn per question from AAMC weights **Skill 1: 35 / Skill 2: 45 / Skill 3: 10 / Skill 4: 10**.
  - *Science*: `build_question_plan` maps each `answer_basis` to a "natural" skill (`from_passage→skill_1`, `apply_knowledge→skill_2`, `data_interpretation→skill_4`), but ~30% of the time draws from the configured `skill_weights` (default 15/35/25/25) so Skill 3 also appears.
  - *CARS*: each passage's question set is apportioned across the three CARS skills by `skill_breakdown_for(n)` using the AAMC ~30/30/40 split (Foundations of Comprehension / Reasoning Within / Reasoning Beyond).
- **Correct-answer position**: a **target letter** (A/B/C/D) is chosen per question (uniform for discrete/science; a near-uniform shuffle across the passage for CARS) and injected into the prompt, because LLMs otherwise over-favor B/C. The generator is told to make that letter the single correct answer.

---

## 2. All Prompts (verbatim)

Prompts are Python f-strings in `src/prompts/`. Below, each is reproduced as it appears in source; `{…}` are runtime substitutions and the multi-branch blocks (skill guidance, difficulty guidance) are all included in full. The pipelines send these as `[{role: system}, {role: user}]`, and `LLMClient` lifts the `system` entries into the Anthropic top-level `system` parameter.

### 2.1 Discrete — generation system prompt

The system prompt is assembled from a skill-guidance block + difficulty block + general rules. **Skill weights** = `{skill_1:35, skill_2:45, skill_3:10, skill_4:10}`; **difficulty weights** = `{easy:20, medium:50, hard:30}`; target answer = uniform A/B/C/D.

System prompt template:

```
You are an expert MCAT question writer for the Association of American Medical Colleges (AAMC). You create questions that match the difficulty, style, and cognitive demands of the actual MCAT exam.

The MCAT tests four Scientific Inquiry and Reasoning Skills across its science sections. You are writing a question that tests a specific skill.

{skill_guidance[skill_key]}        # one of the four blocks below

Target difficulty for this question: {difficulty.upper()}
{difficulty_guidance[difficulty]}  # one of the three blocks below

General question-writing rules:
- Keep discrete question stems under 150 words. Discrete questions are standalone, not passage-based — if a scenario needs more setup than that, simplify it.
- The "choices" object must contain EXACTLY four choices keyed "A", "B", "C", and "D" and nothing else — no extra keys (no "E", no "C2", no duplicates), no missing keys, and every choice must be a non-empty string
- Exactly ONE answer must be unambiguously correct
- For THIS question, the correct answer MUST be option {target_answer}. Write the four choices so that option {target_answer} is the single correct answer and the other three are plausible distractors. Do not let the correct option's position be guessable from its length, specificity, or phrasing.
- Distractors (wrong answers) must be plausible — they should represent common misconceptions, partially correct reasoning, or errors students commonly make
- Answer choices should be roughly similar in length and specificity (a correct answer that is much longer or more detailed than the others is a test-taking giveaway)
- The question should require introductory college-level knowledge, not graduate-level obscurity
- Mathematical questions should use algebra, logarithms, basic trig, or dimensional analysis (no calculus). A periodic table is available during the exam.
- Include a thorough explanation of why the correct answer is right AND why each distractor is wrong. The explanation is FINAL, PUBLISHED content shown to a student — not a draft or a note to a reviewer. It must reference ONLY options A, B, C, and D (never a fifth option or one that does not exist), and must contain NO meta-commentary about the question itself — no hedging, disclaimers, or remarks about the item's format, validity, or completeness (e.g. never write "this item is illustrative", "placeholder", "a properly formatted question would have four options", or similar). Write it as if the question is finished and correct.

Respond with ONLY a JSON object that has EXACTLY these seven keys. Each value is described in angle brackets — produce a real value matching the description; do NOT output the angle-bracket text itself:
{
  "stem": <string: the full question text>,
  "choices": <object with EXACTLY keys "A","B","C","D", each a non-empty string>,
  "correct_answer": <one of "A","B","C","D"; for this question must be "{target_answer}">,
  "explanation": <string: final published explanation referencing only A-D, no meta-commentary>,
  "difficulty": <exactly "{difficulty}">,
  "subtopics_tested": <array of 1-3 short strings>,
  "skill_tested": <exactly "{skill_label}">
}

STRICT OUTPUT CONTRACT — follow EXACTLY or the response is rejected:
- The angle-bracket descriptions above tell you what each value must be — replace each with real content; never emit a literal "<...>" or the word "placeholder" as a value or key.
- Output EXACTLY the seven keys listed above and NOTHING else. Do NOT add any other top-level key for any reason — no "comment", "note", "explanation_placeholder", "answer_placeholder", "choices_extra", or any other commentary, metadata, or placeholder key.
Output the raw JSON object only — no markdown fences, no preamble, no trailing notes.
```

**Skill-guidance blocks (embedded few-shot examples are part of the system prompt):**

`skill_1`:
```
This question should test Skill 1: Knowledge of Scientific Concepts and Principles.
The question should ask the student to:
- Recognize or identify a scientific concept, principle, or relationship from an example or scenario
- Identify relationships between closely related concepts (e.g., written vs. graphical representations)
- Identify examples or observations that illustrate a scientific principle
- Use a given mathematical equation to solve a problem
- Recognize a concept shown in a diagram, graph, or structural formula

Example formats: "What type of functional group is formed when...", "Which of the following best describes the relationship between...", "A student observes X. This is an example of..."

Do NOT write a simple recall/definition question. Even Skill 1 questions should present a scenario, example, or representation that the student must interpret.

Here is an example of a real MCAT question testing this skill:
Stem: "In a study, each trial involves administering a drop of lemon juice to the participant's tongue and measuring the participant's level of salivation. As more trials are conducted, the researcher finds that the magnitude of salivation declines. After a certain point, the researcher switches to administering lime juice. This researcher is most likely studying which process?"
A) Sensory perception
B) Habituation and dishabituation
C) Stimulus generalization in classical conditioning
D) Conditioned responses in classical conditioning
Correct: B
Why it tests Skill 1: The student must recognize the scientific concept (habituation/dishabituation) from a described scenario, rather than recall a definition.
```

`skill_2`:
```
This question should test Skill 2: Scientific Reasoning and Problem-Solving.
The question should ask the student to:
- Use scientific theories or models to explain observations or make predictions
- Evaluate the validity or credibility of a scientific explanation
- Evaluate arguments about cause and effect using scientific knowledge
- Bring together theory, observations, and evidence to draw conclusions
- Recognize findings that challenge or invalidate a theory or model
- Determine and use scientific formulas to solve a multi-step problem

Example formats: "A researcher observes X. Which explanation best accounts for...", "Based on the principle of Y, what would happen if...", "Which finding would most weaken the hypothesis that...", "Given the following data, calculate..."

Present a scenario that requires REASONING, not just recall. The student should need to apply a principle to a novel situation or evaluate competing explanations.

Here is an example of a real MCAT question testing this skill:
Stem: "The radius of the aorta is about 1.0 cm, and blood passes through it at a velocity of 30 cm/s. A typical capillary has a radius of about 4 × 10⁻⁴ cm, with blood passing through at a velocity of 5 × 10⁻² cm/s. Using these data, what is the approximate number of capillaries in a human body?"
A) 1 × 10⁴
B) 2 × 10⁷
C) 4 × 10⁹
D) 7 × 10¹²
Correct: C
Why it tests Skill 2: The student must apply the continuity equation (conservation of flow, A₁v₁ = N·A₂v₂) to a novel physiological scenario and carry out multi-step quantitative reasoning.
```

`skill_3`:
```
This question should test Skill 3: Reasoning About the Design and Execution of Research.
The question should ask the student to:
- Identify independent, dependent, and confounding variables in a described experiment
- Evaluate the appropriateness of a research method, tool, or measurement
- Identify limitations or flaws in a research study design
- Distinguish between correlational and causal claims
- Identify what controls are needed and why
- Reason about ethical issues in research

Example formats: "Researchers conducted a study where... What is the independent variable?", "Which modification to the experimental design would best control for...", "A study finds a correlation between X and Y. Which conclusion is most justified?", "Which aspect of this study design most threatens its internal validity?"

You MUST describe a specific experiment or study in the question stem. The student should evaluate the research design, not just recall a concept.

Here is an example of a question testing this skill:
Stem: "A researcher tests whether caffeine improves short-term memory. Forty undergraduates are randomly assigned to drink either 200 mg of caffeine or a placebo, then complete a word-recall task 30 minutes later. The researcher records the number of words each participant correctly recalls. In this study, what is the dependent variable?"
A) The dose of caffeine administered
B) Whether participants received caffeine or placebo
C) The number of words correctly recalled
D) The 30-minute delay before testing
Correct: C
Why it tests Skill 3: The student must identify the dependent variable in a described experimental design — the measured outcome, not the manipulated condition.
```

`skill_4`:
```
This question should test Skill 4: Data-Based and Statistical Reasoning.
The question should ask the student to:
- Interpret patterns in data presented in a table, graph, or figure (describe the data in text)
- Use measures of central tendency (mean, median, mode) or dispersion (range, SD)
- Reason about random vs. systematic error
- Interpret statistical significance or confidence intervals
- Use data to explain relationships between variables or draw conclusions
- Identify conclusions that are or are not supported by given results

Example formats: "A table shows the following results... What conclusion is supported?", "If the mean is X and the standard deviation is Y, approximately what percentage...", "Researchers measure Z across four groups and obtain the following values... Which comparison is statistically meaningful?", "Based on the data, which relationship between the variables is most likely?"

You MUST present specific data (numbers, values, trends) in the question stem. The student should reason FROM the data, not just know what a statistical concept means.

Here is an example of a question testing this skill:
Stem: "A researcher measures resting heart rate (in beats per minute) in four groups of adults, with the following results: Group 1 (sedentary): mean = 78, SD = 6; Group 2 (light exercise): mean = 72, SD = 5; Group 3 (moderate exercise): mean = 66, SD = 5; Group 4 (vigorous exercise): mean = 60, SD = 7. Which conclusion is best supported by the data?"
A) Vigorous exercise causes a decrease in resting heart rate.
B) Resting heart rate tends to decrease as habitual exercise intensity increases.
C) All adults who exercise vigorously have a resting heart rate below 65 bpm.
D) The variability in resting heart rate is greatest among sedentary adults.
Correct: B
Why it tests Skill 4: The student must read group means from described data, identify the trend across groups, and distinguish a supported descriptive claim from causal overreach (A), an unsupported universal claim (C), and a misreading of SDs (D).
```

**Difficulty-guidance blocks:**
```
easy:   EASY: A well-prepared student should be able to answer this in under 30 seconds. The scenario should be straightforward and the correct application of the concept should be relatively direct. Distractors should still be plausible, but the right answer should be reachable without multi-step reasoning.
medium: MEDIUM: Typical MCAT difficulty. The student must read carefully, apply a concept or principle to a non-trivial scenario, and reason through plausible distractors. Should take ~60-75 seconds.
hard:   HARD: A challenging MCAT question. May require combining two concepts, multi-step calculation, careful elimination of subtly wrong distractors, or recognizing a counterintuitive application of a principle. Still solvable with introductory college-level knowledge — not graduate-level obscurity.
```

**Discrete generation user prompt:**
```
Generate an MCAT-style discrete question for:

Section: {section}
Content Category: {content_category}
Topic Group: {topic_group}
Topic: {topic}
Discipline: {discipline}{focus_str}{subtopics_str}

Target Skill: {skill_label}
Target Difficulty: {difficulty}
Target correct-answer position: {target_answer}{diversity_str}

Write a realistic MCAT question testing this specific skill at the target difficulty. Present a scenario, experiment, data, or problem that requires the student to think, not just remember.
```
`{subtopics_str}` (if topic has subtopics) = `"\nSpecific subtopics to potentially test: <comma list>"`. `{focus_str}` (if `section_focus`) = `"\nFocus angle: <…>"`. `{diversity_str}` (when prior accepted stems exist for this topic):
```
Questions already accepted for this topic test the following. Do NOT duplicate them — generate a question on a genuinely different concept, application, or angle within this topic:
1. <stem 1>
2. <stem 2>
...
If the topic is too narrow for another distinct angle, a related question is acceptable, but prioritize new ground.
```

### 2.2 Discrete — adversarial review

System:
```
You are a rigorous MCAT question reviewer and quality assurance expert working for the AAMC. Your job is to find flaws in MCAT questions before they go to students. Be critical and thorough.

The MCAT tests four Scientific Inquiry and Reasoning Skills:
- Skill 1 (Knowledge): Recognize/identify concepts, use equations, interpret representations
- Skill 2 (Reasoning): Apply theories to scenarios, evaluate explanations, multi-step problems
- Skill 3 (Research Design): Identify variables, evaluate methods, spot design flaws
- Skill 4 (Data Reasoning): Interpret data/graphs/tables, use statistics, draw data-based conclusions

You must check for:
1. ACCURACY: Is the stated correct answer actually correct? Are there any factual errors in the stem, choices, or explanation?
2. AMBIGUITY: Could more than one answer be defensibly correct? Is the stem clear enough that a knowledgeable student would not be confused?
3. DISTRACTORS: Are the wrong answers plausible? Do they represent real misconceptions or common errors? Would a prepared student actually consider them?
4. SKILL ALIGNMENT: Does the question actually test the stated SIRS skill? A Skill 2 question should require reasoning/application, not just recall. A Skill 3 question must describe a study. A Skill 4 question must present data.
5. DIFFICULTY: Is this appropriate for the MCAT (introductory college-level, not graduate-level)?
6. ANSWER BALANCE: Are the choices roughly similar in length and specificity? A correct answer that is noticeably longer, more detailed, or more qualified than the distractors is a test-taking giveaway that must be fixed.
7. STEM QUALITY: Does the stem present a scenario or problem (not just "which of the following is true about X")? For Skills 3-4, does it describe a study or present data?
8. CHOICE STRUCTURE: The question must have EXACTLY four answer choices keyed A, B, C, and D — no missing, extra, duplicate, or empty choices. Fail the question if the choice set is malformed.

Respond with ONLY a JSON object:
{
  "passed": true,
  "issues": [],
  "reasoning": "Brief explanation of your assessment"
}

Set "passed" to false if ANY significant issue is found. List all issues found.
```
User:
```
Review this MCAT question for quality, accuracy, and AAMC alignment.

Topic context:
- Section: {section}
- Content Category: {content_category}
- Topic: {topic}

Question to review:
Stem: {stem}

Choices:
A) {A}
B) {B}
C) {C}
D) {D}

Stated correct answer: {correct_answer}
Explanation: {explanation}
Skill tested: {skill_tested}

Critically evaluate this question. Find any flaws.
```

### 2.3 Discrete — blind solve (runs on the cheaper checker model)

System:
```
You are an MCAT expert with deep knowledge of biology, biochemistry, chemistry, physics, and psychology/sociology at the introductory college level. Answer the following question by selecting the SINGLE BEST answer choice.

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
```
User:
```
Answer this MCAT question:

{stem}

A) {A}
B) {B}
C) {C}
D) {D}
```

### 2.4 CARS — passage generation

System:
```
You are an expert MCAT CARS passage writer working for the AAMC. You write passages that closely mimic the style, complexity, and structure found on the actual MCAT Critical Analysis and Reasoning Skills (CARS) section.

According to the AAMC content outline, CARS passages have these characteristics:
- They are "relatively short, typically between 500 and 600 words"
- They are "complex, often thought-provoking pieces of writing with sophisticated vocabulary and, at times, intricate writing styles"
- They are "multifaceted and focus on the relationships between ideas or theories"
- They come from "the kinds of books, journals, and magazines that college students are likely to read"
- No outside scientific or technical knowledge is required to understand them
- "Even those written in a conversational or opinionated style are often multifaceted"

Passage types (match the tone to the subject):
- SOCIAL SCIENCES passages "tend to be more factual and scientific in tone" — they might discuss how assumptions help scholars reconstruct patterns, analyze societal trends, or examine institutional structures
- HUMANITIES passages "often focus on the relationships between ideas and are more likely to be written in a conversational or opinionated style" — consider "the tone and word choice of the author in addition to the passage assertions themselves"

Structural requirements:
- Present a clear thesis or central argument with supporting reasoning
- Include nuanced qualifications, counterpoints, or internal tensions
- Use rhetorical devices, analogies, references to other thinkers or schools of thought
- Have enough layers (claims, evidence, counterpoints, implications) to support 10 questions
- Include both explicitly stated positions and implied/suggested ideas
- Vary the structure: some sections should state things directly, others should imply or hint

Word count: EXACTLY {word_min}-{word_max} words. This is critical.

Respond with ONLY a JSON object:
{
  "passage_text": "The full passage text here...",
  "subject": "{subject}"
}
```
User:
```
Write an MCAT CARS passage on the subject of {subject}.

The passage should read like an excerpt from an academic book, journal article, or sophisticated magazine piece that a college student might encounter. It should present an original argument or analysis with enough nuance and complexity to support 10 challenging multiple-choice questions.

Remember: the passage must be between {word_min} and {word_max} words, and should NOT require any specialized scientific or technical knowledge to understand.
```

### 2.5 CARS — passage quality review

System:
```
You are an MCAT CARS passage quality reviewer working for the AAMC. Evaluate whether this passage meets the standards for the CARS section.

According to the AAMC, CARS passages should be:
- "Complex, often thought-provoking pieces of writing with sophisticated vocabulary"
- "Multifaceted and focus on the relationships between ideas or theories"
- Similar to "the kinds of books, journals, and magazines that college students are likely to read"
- Answerable without "additional coursework or specific knowledge"
- If social sciences: "more factual and scientific in tone"
- If humanities: "focus on the relationships between ideas," "more likely to be written in a conversational or opinionated style"

Check for:
1. WORD COUNT: Is it between {word_min} and {word_max} words?
2. ARGUMENTATION: Does it present a clear thesis or argument (not just describe facts)? Does it have internal complexity — qualifications, counterpoints, tensions?
3. QUESTION SUPPORT: Is it complex enough to support 10 questions across all three CARS skill types (comprehension, reasoning within, reasoning beyond)? Are there enough layers for application and incorporation questions?
4. INDEPENDENCE: Can it be understood without specialized scientific/technical knowledge? A reader should need NO outside information.
5. SOPHISTICATION: Is the writing at an appropriate academic level? Does it use sophisticated vocabulary naturally? Does it have an identifiable authorial voice and tone?
6. MULTIFACETED: Does it focus on "relationships between ideas or theories" rather than just describing a single concept? Are there multiple perspectives or interpretive layers?

Respond with ONLY a JSON object:
{
  "passed": true,
  "word_count": 547,
  "issues": [],
  "reasoning": "Brief assessment"
}
```
User: `Evaluate this CARS passage for AAMC quality standards:\n\n{passage_text}`

### 2.6 CARS — question generation

Question counts per skill come from `skill_breakdown_for(n)` (~30/30/40). Target answer positions assigned via `balanced_answer_positions(n)`.

System:
```
You are an expert MCAT CARS question writer working for the AAMC. You create questions that test the three Critical Analysis and Reasoning Skills defined by the AAMC.

Question type distribution (for {num_questions} questions):
{skill_breakdown}                # e.g. "  - Foundations of Comprehension: 2 question(s)\n  - ..."

Detailed skill descriptions (from the AAMC content outline):

FOUNDATIONS OF COMPREHENSION:
These questions focus on understanding from immediate sentence context. They should ask the student to:
- Identify the author's thesis, main point, or central theme
- Recognize the purpose of particular sentences or rhetorical labels ("for example," "therefore," "consequently")
- Interpret the meaning of words or expressions using sentence context
- Identify how the author structured the text (cause-and-effect, chronological, point-and-counterpoint)
- Recognize the author's tone (humorous, authoritative, satirical) and its purpose (persuade, instruct, inform, entertain)
Example question types: "The author's primary purpose in this passage is...", "As used in paragraph 2, the word X most nearly means...", "Which of the following best summarizes the main idea?"

REASONING WITHIN THE TEXT:
These require integrating DISTANT passage components into a complex interpretation. They differ from Comprehension in scope — they require synthesizing across the whole passage. They should ask the student to:
- Infer the author's message, purpose, position, beliefs, assumptions, or bias by integrating information from multiple parts of the passage
- Detect paradoxes, contradictions, or inconsistencies across different passage sections
- Identify whether the author presents their own perspective vs. others' views through summaries or paraphrases
- Evaluate arguments: examine evidence, relevance, faulty causality, credibility of sources
- Analyze the author's language, stance, and purpose beneath surface-level meaning
- Identify "vague or evasive terms or language that sounds self-aggrandizing, overblown, or otherwise suspect"
Important: These questions do NOT ask for the student's personal opinion. Even if the student disagrees with the author, the correct answer is based on what the passage says.
Example question types: "The author would most likely agree with which of the following?", "Which assumption underlies the author's argument in the third paragraph?", "The author's discussion of X serves primarily to..."

REASONING BEYOND THE TEXT:
These require applying passage ideas to new contexts OR assessing the impact of new information on the passage. Two sub-types:
1. APPLICATION/EXTRAPOLATION: The passage is the "given" and the question provides a new context. Ask how passage ideas apply to a new situation, what analogy fits, how the author would respond to a hypothetical. "Each response option yields a different result, but only one is defensible based on the passage."
2. INCORPORATION: Introduce new information in the question and ask how it affects the passage's argument. "Does the new information support or contradict the passage? Could it coexist, or would it negate an aspect of the argument? What modifications would be needed?"
Example question types: "If a study showed X, how would this affect the author's argument?", "Which situation is most analogous to the relationship described in the passage?", "Which new finding, if true, would most weaken the author's central claim?", "The author's argument could best be applied to which of the following scenarios?"

General rules:
- Each question's "choices" object must have EXACTLY four keys "A", "B", "C", and "D" and nothing else — no extra keys (no "E", no "C2", no duplicates), no missing/empty keys; each choice a non-empty string. Exactly ONE answer is unambiguously correct.
- ANSWER-KEY BALANCE: each question's correct-answer position is ASSIGNED in the task below. Write the four choices so the assigned letter is the single correct answer, and do not let the correct option be guessable from its length, specificity, or phrasing.
- Questions must be answerable SOLELY from the passage — no outside knowledge required
- Distractors should be plausible and represent common misreadings or partial understandings
- Answer choices should be roughly similar in length (no giveaway long correct answers)
- Explanations MUST reference specific parts of the passage to justify the correct answer
- Across the {num_questions} questions, cover a variety of question formats: main idea, detail, inference, application, tone, structure, strengthen/weaken, analogy, new-information-impact

Respond with ONLY a JSON array of EXACTLY {num_questions} question objects, in order:
[
  {
    "skill_type": "Foundations of Comprehension",
    "stem": "Question text...",
    "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "correct_answer": "A",
    "explanation": "Explanation referencing specific passage content..."
  }
]
```
User:
```
Read this CARS passage and generate {num_questions} multiple-choice questions.

PASSAGE:
{passage_text}

Generate EXACTLY {num_questions} questions, in order, following the skill-type distribution above. Assign each question's correct answer to the position given here — the i-th question you output MUST set "correct_answer" to the i-th letter: {positions_str}

Make them challenging and realistic — they should require careful reading and analysis, not surface-level comprehension. For Reasoning Beyond the Text questions, introduce genuinely novel scenarios or information that test whether the student can extend or challenge the passage's ideas.
```

### 2.7 CARS — adversarial review

System:
```
You are a rigorous MCAT CARS question reviewer working for the AAMC. Your job is to find flaws before questions reach students. Be critical and thorough.

The AAMC tests three CARS skills:
- Foundations of Comprehension (30%): basic understanding, word meaning, author's purpose
- Reasoning Within the Text (30%): integrating distant components, evaluating arguments, detecting bias/assumptions
- Reasoning Beyond the Text (40%): applying ideas to new contexts, assessing impact of new info

Check for:
1. ANSWERABILITY: Can this question be answered SOLELY from the passage? Does it require outside knowledge? (This is the #1 rule of CARS — everything must come from the passage.)
2. ACCURACY: Is the stated correct answer actually the BEST answer based on the passage? Could a knowledgeable reader make a strong case for a different answer?
3. AMBIGUITY: Could more than one answer be defensibly correct given the passage content? Are any distractors too close to the correct answer?
4. DISTRACTORS: Are wrong answers plausible misreadings or partial understandings? Or are they obviously wrong / absurd? (Good distractors on CARS represent things a careless reader might conclude.)
5. SKILL ALIGNMENT: Does the question actually test the stated skill type? A Comprehension question should focus on immediate sentence context. A Reasoning Within question should require integrating distant passage components. A Reasoning Beyond question should introduce a genuinely new context or information.
6. PASSAGE SUPPORT: Does the explanation correctly reference specific passage content? Can you trace the correct answer back to something in the passage?
7. ANSWER BALANCE: Are choices roughly similar in length and specificity?

Respond with ONLY a JSON object:
{
  "passed": true,
  "issues": [],
  "reasoning": "Brief assessment"
}

Set "passed" to false if ANY significant issue is found.
```
User:
```
Review this CARS question against the passage.

PASSAGE:
{passage_text}

QUESTION:
Skill type: {skill_type}
Stem: {stem}
A) {A}
B) {B}
C) {C}
D) {D}
Correct answer: {correct_answer}
Explanation: {explanation}

Find any flaws. Be especially strict about whether the correct answer is truly the BEST answer and whether the question actually tests the stated CARS skill type.
```

### 2.8 CARS — blind solve

System:
```
You are an MCAT expert taking the CARS section. Read the passage carefully and answer the question based ONLY on what is in the passage.

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
```
User: passage + question stem + choices A–D.

### 2.9 Science passage — passage generation

Skill labels, the difficulty weights, and `answer_basis` labels live in `src/prompts/science_passage.py`. `{figure_rule}` switches on `enable_figures`.

System:
```
You are an expert MCAT passage writer for the Association of American Medical Colleges (AAMC). You write realistic science passages for the {section} section of the MCAT.

A science passage is a {word_min}-{word_max} word piece that typically describes an EXPERIMENT or STUDY: its background/motivation, methods, and results. It reads like a condensed methods-and-results section written for an educated reader, and it gives students concrete material to reason about.

STRICT CONSTRAINTS:
- Length: {word_min}-{word_max} words of passage prose (not counting any table).
{figure_rule}
- If (and only if) a data table genuinely helps, include ONE markdown table in the "table_markdown" field (use standard GitHub-flavored markdown: header row, separator row, data rows). Otherwise set "table_markdown" to null. Do NOT embed the table inside "passage_text".
- The science must be accurate and at the introductory college level (the level the MCAT tests). Any numbers/data must be internally consistent and plausible.
- The passage should give enough material that some questions are answerable directly from it, some require the reader to apply outside introductory science knowledge, and some require interpreting the passage's data.
- Do NOT include any questions in the passage. Write ONLY the passage (and optional table[, figures]).{figure_section}

Respond with ONLY a JSON object in this exact format, no other text:
{
  "passage_text": "The full passage prose here.",
  "table_markdown": "| Group | Value |\n| --- | --- |\n| A | 1 |"{figures_json_line}
}
(Set "table_markdown" to null if no table.)
```
`{figure_rule}` — **figures disabled**:
```
- Prose and an OPTIONAL markdown TABLE only. You may NOT rely on any figure, graph, diagram, micrograph, chemical-structure image, or plot. Do NOT write a passage that would require the student to see a figure that cannot be expressed as text or a markdown table. If you would normally show results as a graph, instead either describe the trend in prose or present the numbers as a markdown table.
```
`{figure_rule}` — **figures enabled**:
```
- You MAY include chemical structures (SMILES) or plots via the "figures" array (see FIGURES below). You may also use a markdown table. Do NOT describe a figure you have not specified.
```
User:
```
Write one MCAT science passage for the following topic cluster.

Section: {section}
Content Category: {content_category}
Topic Group: {topic_group}
Related topics this passage should give material for:
{topics_str}                 # "- <topic> (subtopics: ...)" per topic

Write a {word_min}-{word_max} word passage (describing an experiment or study where natural) that ties these related topics together and gives a student concrete material to reason about. Include a markdown table only if it genuinely helps present data.{figures_user_note}
```

### 2.10 Science passage — figure-generation instructions (`FIGURE_GEN_INSTRUCTIONS`)

Appended to passage and question system prompts only when `enable_figures: true`:
```
FIGURES (optional): If a chemical structure or a data plot would genuinely strengthen this item, you MAY specify it in the "figures" array. Each figure is rendered to an IMAGE the student sees, so you MUST provide the complete underlying data.
- Chemical structure: {"figure_type": "smiles", "caption": "...", "alt_text": "short text description", "smiles": {"molecules": [{"smiles": "CCO", "label": "ethanol"}]}}  (one or more molecules; SMILES must be valid and parseable)
- Plot: {"figure_type": "plot", "caption": "...", "alt_text": "short text description", "plot": {"chart_type": "bar"|"line"|"scatter"|"histogram", "title": "...", "x_label": "...", "y_label": "...", "series": [{"name": "Group A", "x": [...], "y": [...]}]}}  (x and y must be equal length; provide the actual numbers)
RULES: Do NOT reference or describe a figure you have not specified (no "as shown below", "the figure", "the structure", "the graph" without a matching spec in "figures"). Conversely, do not specify a figure the item never uses. If no figure is needed, use an empty array ("figures": []). Prefer a markdown table over a plot when a table conveys the data just as well.

WHEN A SMILES FIGURE IS REQUIRED: When the passage or a question centers on a SPECIFIC molecule, functional group, reaction substrate/product, or structural comparison, you MUST include that molecule as a SMILES figure rather than only naming it in prose. Examples that REQUIRE a SMILES figure: a question asking about the product of a named reaction, identifying a functional group on a given structure, comparing two molecules' structures, or stereochemistry. Do NOT include a SMILES figure for purely conceptual topics (e.g. gas laws, thermodynamics, atomic structure) where no specific molecule is depicted.

WHEN A PLOT FIGURE IS REQUIRED: When the passage reports quantitative experimental results across conditions, time points, concentrations, or groups, present those results as a PLOT figure (bar/line/scatter) when a graph is the natural representation on the real MCAT — for example: reaction rate vs substrate concentration (line/scatter), a measured quantity across experimental groups (bar), or a time course (line). Provide the actual data values in the plot spec. You may ALSO keep a small data table if helpful. Do NOT invent a plot for topics with no quantitative results.

For topics where neither applies (e.g. conceptual topics like gas laws or atomic nucleus), a text-only passage with no figures is correct — do not force a figure.
```

### 2.11 Science passage — passage review

`{figure_check}` is item 3 and switches on `enable_figures`.

System:
```
You are a rigorous MCAT passage reviewer for the AAMC. You vet science passages before questions are written for them. Check the passage and answer with a JSON verdict.

Check for:
1. SCIENTIFIC ACCURACY: Is everything stated scientifically correct at the introductory college level? Flag any factual errors, impossible values, or misleading claims.
2. INTERNAL CONSISTENCY: Are any data/numbers (in prose, the table, or a figure) self-consistent and plausible? Does the table, if present, parse as valid markdown and match the prose?
{figure_check}
4. LENGTH/STYLE: Roughly {word_min}-{word_max} words, written like a real MCAT science passage (experiment/study style preferred).
5. USABILITY: Does it give enough concrete material (a scenario, methods, results/data) to support a mix of passage-based, knowledge-application, and data-interpretation questions?

Respond with ONLY a JSON object:
{
  "passed": true,
  "issues": [],
  "reasoning": "Brief assessment"
}

Set "passed" to false if ANY significant issue is found, and list the issues.
```
`{figure_check}` — **figures disabled**:
```
3. NO-FIGURE CONSTRAINT: Does the passage stand on its own with ONLY prose and the optional markdown table? FAIL it if it references or requires a figure, graph, diagram, image, or structure that is not expressible as text/a table (e.g. "as shown in Figure 1", "the graph below", "the structure depicted").
```
`{figure_check}` — **figures enabled**:
```
3. FIGURE CONSISTENCY: The passage may include chemical structures (SMILES) or plots, provided above as text. FAIL it if the passage references a figure/graph/structure that is NOT provided above (an unspecified figure), or if a provided figure's data contradicts the prose. A specified figure is fine. Tables and specified figures are the only non-prose elements allowed.
```
User:
```
Review this MCAT science passage (intended topics: {topic_names}).

Passage:
{passage_text}{table_block}{figure_block}

Critically evaluate it for accuracy, consistency, the figure constraint, and usability.
```

### 2.12 Science passage — question generation

`{_SKILL_GUIDANCE[skill_key]}` is the passage-adapted SIRS block; `{ANSWER_BASIS_LABELS[answer_basis]}` defines the assigned basis.

System:
```
You are an expert MCAT question writer for the AAMC, writing a question tied to a science passage in the {section} section.

{_SKILL_GUIDANCE[skill_key]}

This question's ANSWER BASIS must be "{answer_basis}":
{ANSWER_BASIS_LABELS[answer_basis]}
Write the question so that this basis is genuinely true: a "from_passage" question must be answerable from the passage; an "apply_knowledge" question must require outside intro science knowledge (not be answerable from the passage text alone); a "data_interpretation" question must hinge on reading the passage's data.

Target difficulty: {difficulty.upper()}
{difficulty_guidance[difficulty]}

Question-writing rules:
- The question must be tied to the passage above. The stem may quote or paraphrase the passage, but keep it concise (under ~120 words).
- The "choices" object must contain EXACTLY four choices keyed "A", "B", "C", and "D" and nothing else — no extra/missing/duplicate/empty keys; each choice a non-empty string.
- Exactly ONE answer is unambiguously correct. For THIS question the correct answer MUST be option {target_answer}; write the other three as plausible distractors (common misconceptions, partial reasoning, or data misreadings). Do not let the correct option be guessable from length, specificity, or phrasing.
- Choices should be similar in length/specificity.
- Introductory college level; algebra/logs/basic trig/dimensional analysis only (no calculus); a periodic table is available.
- Include a thorough explanation: why the correct answer is right and why each distractor is wrong. The explanation is FINAL, PUBLISHED content shown to a student — not a draft or a note to a reviewer. It must reference ONLY options A, B, C, and D (never a fifth option or one that does not exist), and must contain NO meta-commentary about the question itself — no hedging, disclaimers, or remarks about the item's format, validity, or completeness (e.g. never write "this item is illustrative", "placeholder", "a properly formatted question would have four options", or similar).{figure_section}

Respond with ONLY a JSON object that has EXACTLY these keys. Each value is described in angle brackets — produce a real value matching the description; do NOT output the angle-bracket text itself:
{
  "stem": <string: the full question text, tied to the passage>,
  "choices": <object with EXACTLY keys "A","B","C","D", each a non-empty string>,
  "correct_answer": <one of "A","B","C","D"; for this question must be "{target_answer}">,
  "explanation": <string: final published explanation referencing only A-D, no meta-commentary>,
  "difficulty": <exactly "{difficulty}">,
  "answer_basis": <exactly "{answer_basis}">,
  "skill_tested": <exactly "{skill_label}">{figures_json_line}
}

STRICT OUTPUT CONTRACT — follow EXACTLY or the response is rejected:
- The angle-bracket descriptions above tell you what each value must be — replace each with real content; never emit a literal "<...>" or the word "placeholder" as a value or key.
- Output EXACTLY these top-level keys and NOTHING else: {allowed_keys_str}. Do NOT add any other top-level key for any reason — no "comment", "note", "explanation_placeholder", "answer_placeholder", "choices_extra", or any other commentary, metadata, or placeholder key.
Output the raw JSON object only — no markdown fences, no preamble, no trailing notes.
```

`{_SKILL_GUIDANCE}` blocks:
```
skill_1: This question tests Skill 1: Knowledge of Scientific Concepts and Principles. Ask the student to recognize/identify a concept, principle, or relationship as it appears in the passage scenario, classify something described in the passage, or apply a given equation. Anchor it in the passage — do not write a bare definition question divorced from the passage context.
skill_2: This question tests Skill 2: Scientific Reasoning and Problem-Solving. Ask the student to use a theory/model to explain a passage observation or predict an outcome, evaluate a causal claim about the study, combine the passage with their own knowledge to draw a conclusion, identify a finding that would challenge the study's interpretation, or carry out a multi-step calculation grounded in the passage.
skill_3: This question tests Skill 3: Reasoning About the Design and Execution of Research. Use the experiment/study DESCRIBED IN THE PASSAGE. Ask the student to identify the independent/dependent/confounding variable, evaluate the appropriateness of a method or control, identify a design limitation or threat to validity, distinguish correlation from causation, or reason about what additional control is needed. Refer to the passage's actual design — do not invent a new study.
skill_4: This question tests Skill 4: Data-Based and Statistical Reasoning. Use the DATA in the passage (the prose results or the markdown table). Ask the student to interpret a trend, compare values across groups/conditions, use a measure of central tendency or dispersion, reason about error/significance, or identify which conclusion the data do (or do not) support. The student must reason FROM the passage's data, not from memorized facts.
```
`{ANSWER_BASIS_LABELS}`:
```
from_passage:        Answerable directly from the passage. A careful reader can find or infer the answer from the passage text/table without outside knowledge beyond basic scientific literacy.
apply_knowledge:     Requires the student to APPLY their own introductory college-level science knowledge to the passage scenario. The passage alone is not sufficient; the student must bring in a concept, principle, or formula not stated in the passage.
data_interpretation: Requires interpreting DATA presented in the passage (a trend in the results, a value or comparison in the table, a relationship between variables). The student must reason FROM the data shown.
```
Difficulty blocks (science variant):
```
easy:   EASY: answerable in well under a minute; direct application or a clear read of the passage. Distractors still plausible.
medium: MEDIUM: typical MCAT difficulty; careful reading plus a non-trivial step of reasoning or data interpretation.
hard:   HARD: combine the passage with a concept or multiple steps, or require careful elimination of subtly wrong distractors. Still introductory college level, not graduate obscurity.
```
User:
```
{context}                    # PASSAGE: ...  [TABLE: ...]  [PASSAGE FIGURE(S): ...]

Write ONE MCAT question about this passage.
Target skill: {skill_label}
Answer basis: {answer_basis}
Target difficulty: {difficulty}
Target correct-answer position: {target_answer}{diversity_str}
```

### 2.13 Science passage — adversarial review (with separate lenient answer_basis verdict)

System (`%s` = the question's labeled basis):
```
You are a rigorous MCAT question reviewer for the AAMC. You vet passage-based science questions before they reach students. You are given the passage and one question written about it. Be critical and thorough.

Any figures (chemical structures or plots) are provided to you as TEXT (SMILES strings or the plot's underlying data); the student sees them rendered as images. Reason over that data.

Check for:
1. ACCURACY: Is the stated correct answer actually correct given the passage, any figure data, and correct science? Any factual errors in stem, choices, or explanation?
2. PASSAGE GROUNDING: Is the question genuinely tied to the passage/figure? If it claims to be answerable from the passage, does the passage/figure actually support the keyed answer?
3. AMBIGUITY: Could more than one answer be defensibly correct? Is the stem clear?
4. DISTRACTORS: Are the wrong answers plausible (real misconceptions / common errors / data misreadings)?
5. SKILL ALIGNMENT: Does it test the stated SIRS skill (reasoning/research-design/data, not bare recall when a higher skill is claimed)?
6. ANSWER BALANCE & STRUCTURE: Choices roughly similar in length/specificity; EXACTLY four choices A/B/C/D, none missing/extra/empty. Fail if malformed.
7. FIGURE INTEGRITY: If the stem references a figure/structure/graph, a corresponding figure must be provided above. FAIL if the stem references a figure that is not provided, or if the keyed answer contradicts the figure's data.

SEPARATELY, judge the ANSWER BASIS label. The question is labeled answer_basis = "%s", which means:
  - from_passage: answerable directly from the passage (no outside knowledge beyond basic literacy).
  - apply_knowledge: requires the student's own intro-college science knowledge applied to the passage (NOT answerable from the passage text alone).
  - data_interpretation: hinges on interpreting data presented in the passage (including a figure's data).
Set "answer_basis_ok" to false ONLY for a CLEAR mislabel — for example, a question labeled "from_passage" whose answer is pure outside recall the passage never supports, or one labeled "apply_knowledge" that is in fact answered verbatim by a sentence in the passage. If the label is reasonable or it's a borderline judgment call, set "answer_basis_ok" to true. Do NOT nitpick. This check should be forgiving.

Respond with ONLY a JSON object:
{
  "passed": true,
  "issues": [],
  "reasoning": "Brief assessment of overall quality",
  "answer_basis_ok": true,
  "answer_basis_note": "Only if answer_basis_ok is false: briefly why it's a clear mislabel"
}

Set "passed" to false if ANY significant quality issue (items 1-7) is found. The answer_basis verdict is reported SEPARATELY in "answer_basis_ok" and should not, by itself, drive "passed".
```
User: passage context + optional question-figure block + the question (stem, A–D, stated correct answer, explanation, skill, labeled answer_basis) + "Critically evaluate … give the separate (lenient) answer_basis verdict."

### 2.14 Science passage — blind solve (runs on the cheaper checker model)

System:
```
You are an MCAT expert with introductory college-level knowledge of biology, biochemistry, chemistry, physics, and psychology/sociology. You are given a passage and a question about it. Select the SINGLE BEST answer choice.

Any figures (chemical structures or plots) are given to you as TEXT (SMILES strings or the plot's underlying data); the student sees them as images. Reason over that data.

Work step by step: read the passage, figures, and stem, use the passage and your own knowledge as needed, eliminate wrong choices, and choose the best one.

Respond with ONLY a JSON object:
{
  "chosen_answer": "B",
  "confidence": "high",
  "reasoning": "Brief reasoning"
}

confidence should be "high", "medium", or "low".
```
User: passage context + optional figure block + question stem + choices A–D.

---

## 3. Validation Logic

### 3.1 The two checks

Every generated question is subjected to two independent checks, run **concurrently** (`asyncio.gather`):

1. **Adversarial review** — the main model is told to act as a rigorous AAMC reviewer and find flaws (accuracy, ambiguity, distractor plausibility, skill alignment, difficulty, answer balance, stem quality, choice structure). It returns `{passed, issues[], reasoning}`. `passed=false` if *any* significant issue is found.
2. **Blind solve** — an independent solver model is given **only the stem + choices** (no answer key) and must pick a letter. The check **passes only if the solver's `chosen_answer` matches the generator's keyed `correct_answer`.** For discrete and science this runs on the cheaper `checker_model` (Haiku); for CARS it runs on the main model.

For **science** questions there is a third, **separate and deliberately lenient** verdict, `answer_basis_ok`, embedded in the adversarial review. It defaults to `true` and is set `false` only for a *clear* mislabel of the `from_passage` / `apply_knowledge` / `data_interpretation` tag. It is reported separately and does not by itself drive `passed`, but the pipeline does require it to accept the question.

### 3.2 Pass/fail criteria and what happens to failures

- **Discrete** (`generate_and_validate_question`): accept iff `adversarial_pass AND blind_solve_pass`. On failure, the attempt is discarded and the question is **regenerated from scratch** (up to `max_retries = 3`, i.e. 4 attempts total). If all attempts fail, the slot is abandoned (`slots_failed += 1`) and the topic is left under quota (re-openable via `--recount`). Failures are **never** written to output — only accepted questions reach the jsonl.
- **Science** (`generate_and_validate_question`): accept iff `adversarial_pass AND blind_solve_pass AND answer_basis_ok`. Same retry-from-scratch logic. `answer_basis`-only rejections are logged at INFO (`answer_basis REJECT …`) so their rate is observable. Exhausted slots increment `slots_failed` and the slot is simply skipped (passage keeps the questions that did pass).
- **CARS** (`generate_questions_for_passage`): all N questions are generated at once; each is parsed and validated individually; those passing both checks are kept. If fewer than the target pass, the **entire batch is regenerated** (up to `max_retries`). After retries, whatever passed is returned even if short of target.
- **Passage-level gating** (CARS + science): before any questions are written, the passage itself must pass a **word-count gate** (CARS: within ±50 of range; science: within ±60) and an **LLM passage-quality review** (`passed=true`). If the review *call* errors, the passage is accepted as a fallback if the word count is in range. Science additionally validates figure specs (SMILES must parse via RDKit, plot series must align) and can structurally enforce a required figure type by regenerating the whole passage set.

### 3.3 Retry logic summary

| Pipeline | Retry unit | Max attempts | On exhaustion |
|---|---|---|---|
| Discrete question | single question, regenerated fresh | `max_retries`+1 = 4 | slot abandoned, topic left under quota |
| Science question | single question, regenerated fresh | 4 | slot skipped |
| Science passage set (figure enforcement) | whole passage + questions | 4 | accepted anyway, logged ERROR |
| CARS question batch | all questions for the passage | 4 | keep whatever passed |
| Passage (CARS/science) | whole passage | 4 | give up on that passage |
| API call (transient errors) | the HTTP call | 6 w/ backoff+jitter | raise |

### 3.4 Logged pass-rate statistics (from the production metrics files)

The `MetricsTracker` records the full funnel and writes `generation_metrics.json` (discrete) / `science_generation_metrics.json` (science). Rates are relative to **questions attempted (generation calls)**. CARS does **not** have a metrics tracker, so no pass-rate file exists for it.

**Discrete — `runs/prod_discrete/generation_metrics.json` (run-level summary):**
| Metric | Value |
|---|---|
| Model / checker | `claude-opus-4-8` / `claude-haiku-4-5-20251001` |
| Questions attempted (generation calls) | 1,165 |
| Generation parsed | 979 |
| Passed adversarial review | 826 (**70.9%** of attempts) |
| Passed blind solve | 921 (**79.1%**) |
| **Final accepted** | **786 (67.5%)** |
| Slots failed (exhausted retries) | 14 |
| Total API calls | 3,123 (≈3.97 per accepted) |
| Tokens in / out | 5,182,321 / 2,006,165 |
| **Cost** | **$63.27** ($0.0805 per accepted question) |
| Cost by stage | generation $40.75 · review $19.32 · blind_solve $3.20 |

**Science — `runs/prod_science/science_generation_metrics.json` (run-level summary):**
| Metric | Value |
|---|---|
| Questions attempted | 131 |
| Generation parsed | 102 |
| Passed adversarial review | 97 (**74.1%**) |
| Passed blind solve | 94 (**71.8%**) |
| **Final accepted** | **91 (69.5%)** |
| Slots failed | 3 |
| Total API calls | 371 (≈4.08 per accepted) |
| Tokens in / out | 872,244 / 209,685 |
| **Cost** | **$7.91** ($0.0870 per accepted) |
| Cost by stage | generation $4.67 · review (Opus) · passage_generation $0.54 · passage_review $0.31 · blind_solve (Haiku) |

Both metrics files note they cover only topics processed in that run (checkpoint-skipped topics make no API calls and are excluded).

---

## 4. Output Schema

Each pipeline writes newline-delimited JSON (`.jsonl`), one record per line, via `OutputWriter`. The Pydantic models live in `src/schemas.py`.

### 4.1 Discrete question (`discrete_questions.jsonl`, model `DiscreteQuestion`)

```jsonc
{
  "question_id": "CP_4B_002_q001",        // "<topic_id>_q<NNN>"
  "topic_id": "CP_4B_002",
  "section": "Chemical and Physical Foundations of Biological Systems",
  "content_category": "4B: Importance of fluids ...",
  "topic_group": "Fluids",
  "topic": "Buoyancy, Archimedes' Principle",
  "subtopics_tested": ["Archimedes' principle", "buoyancy and equilibrium", "density"],
  "stem": "…",
  "choices": {"A": "…", "B": "…", "C": "…", "D": "…"},   // exactly A/B/C/D
  "correct_answer": "D",                   // one of A/B/C/D
  "explanation": "…",
  "difficulty": "medium",                  // easy | medium | hard
  "skill_tested": "Skill 1: Knowledge of Scientific Concepts and Principles",
  "validation": {"adversarial_pass": true, "blind_solve_pass": true}
}
```

### 4.2 CARS passage (`cars_passages.jsonl`, model `CARSPassage` → `CARSQuestion[]`)

```jsonc
{
  "passage_id": "CARS_P0001",              // "CARS_P<NNNN>"
  "passage_text": "…",
  "word_count": 572,
  "subject": "Dance",
  "questions": [
    {
      "question_id": "CARS_P0001_q01",     // "<passage_id>_q<NN>"
      "skill_type": "Foundations of Comprehension",  // | Reasoning Within the Text | Reasoning Beyond the Text
      "stem": "…",
      "choices": {"A": "…", "B": "…", "C": "…", "D": "…"},
      "correct_answer": "A",
      "explanation": "…",
      "validation": {"adversarial_pass": true, "blind_solve_pass": true}
    }
  ],
  "validation": {"passage_reviewed": true, "questions_validated": 5, "target_questions": 5}
}
```

### 4.3 Science passage (`science_passages.jsonl`, model `SciencePassage` → `ScienceQuestion[]`)

```jsonc
{
  "passage_id": "SP_CP_4A_019_01",         // "SP_<cluster_key>_<NN>"
  "section": "Chemical and Physical Foundations of Biological Systems",
  "content_category": "4A: Translational motion, forces, work, energy, ...",
  "topic_ids": ["CP_4A_019", "CP_4A_010", "CP_4A_011"],   // the cluster
  "topic_group": "Energy of Point Object Systems / Equilibrium",
  "passage_text": "…",
  "table_markdown": "| Quantity | Value |\n| --- | --- |\n| … |",   // or null
  "figures": [ /* passage-level FigureSpec[] */ ],
  "word_count": 299,
  "questions": [
    {
      "question_id": "SP_CP_4A_019_01_q01",
      "passage_id": "SP_CP_4A_019_01",
      "skill_tested": "Skill 1: Knowledge of Scientific Concepts and Principles",
      "answer_basis": "from_passage",      // from_passage | apply_knowledge | data_interpretation
      "stem": "…",
      "choices": {"A": "…", "B": "…", "C": "…", "D": "…"},
      "correct_answer": "B",
      "explanation": "…",
      "difficulty": "medium",
      "figures": [ /* question-level FigureSpec[] */ ],
      "validation": {"adversarial_pass": true, "blind_solve_pass": true, "answer_basis_ok": true}
    }
  ],
  "validation": {"passage_reviewed": true, "questions_validated": 4, "target_questions": 4}
}
```

**FigureSpec** (when figures enabled): `{figure_id, figure_type ("smiles"|"plot"), caption, alt_text, smiles?: {molecules:[{smiles,label}]}, plot?: {chart_type, title, x_label, y_label, series:[{name,x[],y[],y_err?}]}, image_path?}`. `image_path` is filled by the separate `--render-figures` pass and is relative to the output dir (e.g. `figures/SP_CP_4E_016_01_pf01.png`).

### 4.4 How passages, questions, and figures are linked

- **CARS / science questions are nested inside their passage record** (the `questions` array). Each question also carries its parent `passage_id` (science) or an id derived from it (CARS `question_id` = `<passage_id>_q<NN>`), so the link is recoverable even if flattened.
- **Discrete questions** have no passage; they link upward to their source `topic_id` (and carry section/category/topic_group/topic denormalized onto each record).
- **Figures** attach at two levels: passage-level (`SciencePassage.figures`) and question-level (`ScienceQuestion.figures`). Each `FigureSpec` has a stable `figure_id` (`<passage_id>_pf<NN>` for passage figures, `<question_id>_qf<NN>` for question figures). The `--render-figures` pass walks the jsonl, renders each spec to `figures/`, and records `image_path` back onto the spec (also tracked in `figures_manifest.json`). Validators (review + blind solve) see a **text serialization** of the figure (SMILES string / plot data), never the rendered image.

---

## 5. Sample Outputs (verbatim)

Six complete examples drawn from the production runs, chosen as a quality mix (not cherry-picked best). Full metadata is shown for each.

### 5.1 Discrete example #1 — Chem/Phys, Skill 1, medium

```json
{
  "question_id": "CP_4B_002_q001",
  "topic_id": "CP_4B_002",
  "section": "Chemical and Physical Foundations of Biological Systems",
  "content_category": "4B: Importance of fluids for the circulation of blood, gas movement, and gas exchange",
  "topic_group": "Fluids",
  "topic": "Buoyancy, Archimedes’ Principle",
  "subtopics_tested": ["Archimedes' principle", "buoyancy and equilibrium", "density"],
  "stem": "A solid plastic sphere is released into a tall column of seawater (density 1025 kg/m³). The sphere descends a short distance, comes to rest, and remains suspended motionless at a fixed depth, neither rising nor sinking. Which statement best describes the sphere at this depth?",
  "choices": {
    "A": "The buoyant force on the sphere exceeds its weight, producing a net upward force.",
    "B": "The sphere's average density is greater than that of seawater, so it is slowly sinking.",
    "C": "The volume of seawater displaced is less than the volume of the sphere.",
    "D": "The sphere's average density equals that of seawater, and the buoyant force equals its weight."
  },
  "correct_answer": "D",
  "explanation": "A fully submerged object that remains motionless is in equilibrium: the upward buoyant force exactly balances the downward gravitational force (weight). By Archimedes' principle, the buoyant force equals the weight of the displaced fluid, ρ_fluid·V_submerged·g. For a fully submerged sphere, V_submerged equals the sphere's volume V. Setting buoyant force equal to weight gives ρ_fluid·V·g = ρ_sphere·V·g, so ρ_sphere = ρ_fluid. The sphere's average density must equal the seawater density for it to hover, making D correct. A is wrong because a net upward force would accelerate the sphere upward; since it is motionless, the forces are balanced, not unequal. B is wrong because if the sphere's density exceeded that of seawater, the weight would exceed the buoyant force and it would sink rather than remain at rest. C is wrong because the sphere is fully submerged, so the displaced volume exactly equals the sphere's volume, not less.",
  "difficulty": "medium",
  "skill_tested": "Skill 1: Knowledge of Scientific Concepts and Principles",
  "validation": {"adversarial_pass": true, "blind_solve_pass": true}
}
```

### 5.2 Discrete example #2 — Bio/Biochem, Skill 2, hard

```json
{
  "question_id": "BB_3B_018_q003",
  "topic_id": "BB_3B_018",
  "section": "Biological and Biochemical Foundations of Living Systems",
  "content_category": "3B: Structure and integrative functions of the main organ systems",
  "topic_group": "Circulatory System",
  "topic": "Coagulation, clotting mechanisms",
  "subtopics_tested": ["coagulation cascade", "PT and aPTT interpretation", "liver synthesis of clotting factors"],
  "stem": "A patient with severe liver disease shows prolonged bleeding after minor injuries. Laboratory testing reveals a markedly elevated prothrombin time (PT) and activated partial thromboplastin time (aPTT), but a normal platelet count and normal bleeding time. The hepatocytes synthesize most clotting factors, including the vitamin K-dependent factors II, VII, IX, and X. Based on these findings, which conclusion is best supported?",
  "choices": {
    "A": "The bleeding is caused primarily by impaired platelet adhesion to exposed collagen.",
    "B": "Both the intrinsic and extrinsic pathways are impaired because the liver supplies factors shared by or feeding into the common pathway.",
    "C": "Only the extrinsic pathway is affected, since factor VII deficiency selectively prolongs the PT.",
    "D": "von Willebrand factor deficiency accounts for the prolonged PT and aPTT."
  },
  "correct_answer": "B",
  "explanation": "PT measures the extrinsic and common pathways, while aPTT measures the intrinsic and common pathways. Because BOTH are prolonged, the defect cannot be limited to a single pathway-specific factor; it must affect factors common to both, or affect multiple factors across both pathways. Liver disease reduces synthesis of nearly all clotting factors (including II, VII, IX, X and others feeding the common pathway), so both the intrinsic and extrinsic arms are compromised. This makes B correct. A is wrong because platelet adhesion is reflected by bleeding time and platelet count, both of which are normal here, indicating primary hemostasis is intact and the problem lies in the secondary (coagulation factor) cascade. C is wrong because a selective factor VII deficiency would prolong only the PT and leave the aPTT normal; the observed prolongation of BOTH tests rules out an isolated extrinsic-pathway defect. D is wrong because von Willebrand factor mediates platelet adhesion and stabilizes factor VIII; its deficiency typically prolongs bleeding time and aPTT but does NOT prolong the PT, contradicting the data.",
  "difficulty": "hard",
  "skill_tested": "Skill 2: Scientific Reasoning and Problem-Solving",
  "validation": {"adversarial_pass": true, "blind_solve_pass": true}
}
```

### 5.3 Science passage example — `SP_CP_4A_019_01` (Chem/Phys, biomechanics of the elbow) with ALL 4 questions

Cluster: `CP_4A_019`, `CP_4A_010`, `CP_4A_011` · Topic group: *Energy of Point Object Systems / Equilibrium* · 299 words · table included · no figures.

**Passage text:**
> Biomechanists study the human elbow as a third-class lever, in which the effort (force from the biceps brachii) is applied between the fulcrum (the elbow joint) and the load (a weight held in the hand). Although this arrangement provides no mechanical advantage in force, it allows the hand to move quickly through a large arc.
>
> To characterize muscle performance, researchers asked subjects to hold the forearm horizontally while flexing the elbow against a load held in the hand. The biceps tendon inserts on the radius at a perpendicular distance of 4.0 cm from the joint axis. The center of mass of a dumbbell rests 32 cm from the joint along the horizontal forearm. The forearm itself was treated as a uniform rod whose weight acts at its midpoint.
>
> In one trial, a subject held a 50-N dumbbell statically with the forearm horizontal. The biceps tendon pulls nearly vertically upward. Because the system is in rotational equilibrium, the net torque about the elbow joint is zero, and the muscle must generate a large tension to balance the torque produced by the load and the weight of the forearm.
>
> In a second trial, the subject performed a controlled curl, raising the same dumbbell through a vertical height of 0.40 m in 0.80 s. The investigators recorded the average mechanical power delivered to the load. They also varied the load mass and measured the maximum angular speed achieved during the lift, observing that lighter loads were raised more rapidly.
>
> Forces were resolved into components to analyze joint loading. When the forearm is not horizontal, only the component of the muscle tension perpendicular to the forearm contributes to torque, while the parallel component is transmitted to the joint. The experimenters used these relationships to estimate compressive forces on the elbow during loaded flexion.

**Table:**
| Quantity | Value |
| --- | --- |
| Tendon lever arm | 4.0 cm |
| Load distance from joint | 32 cm |
| Dumbbell weight | 50 N |
| Lift height | 0.40 m |
| Lift time | 0.80 s |

**Q1** (`SP_CP_4A_019_01_q01` · Skill 1 · answer_basis `from_passage` · medium · correct **B**)
> According to the passage, the human elbow operates as a third-class lever. Which feature described in the passage is the defining characteristic of a third-class lever in this system?
> - A) The load is located between the fulcrum and the applied effort.
> - B) The effort is applied between the fulcrum and the load.
> - C) The fulcrum is located between the effort and the load.
> - D) The effort and the load act at the same point along the lever.
>
> *Explanation:* The passage explicitly states that in the elbow, "the effort (force from the biceps brachii) is applied between the fulcrum (the elbow joint) and the load (a weight held in the hand)." This is precisely the arrangement of a third-class lever, making B correct. Choice A describes a second-class lever, in which the load lies between the fulcrum and effort (as with a wheelbarrow); the passage's geometry places the effort, not the load, in the middle. Choice C describes a first-class lever, in which the fulcrum lies between the effort and the load (as with a seesaw); here the fulcrum is at one end (the elbow joint), not between the two forces. Choice D is not described and is geometrically inconsistent, since the tendon inserts 4.0 cm from the joint while the load acts 32 cm from the joint, meaning the effort and load act at different points.
> `validation: {adversarial_pass: true, blind_solve_pass: true, answer_basis_ok: true}`

**Q2** (`…_q02` · Skill 2 · answer_basis `apply_knowledge` · easy · correct **D**)
> In the second trial, the subject raised the 50-N dumbbell through a vertical height of 0.40 m in 0.80 s. What was the average mechanical power delivered to the load?
> - A) 16 W  · B) 20 W  · C) 40 W  · D) 25 W
>
> *Explanation:* Average power equals work done divided by time. The work done against gravity to raise the load is W = (force)(height) = (50 N)(0.40 m) = 20 J. Dividing by the elapsed time gives P = 20 J / 0.80 s = 25 W, which is option D. Option A (16 W) results from incorrectly multiplying 20 J by 0.80 s instead of dividing. Option B (20 W) mistakes the work value (20 J) for the power. Option C (40 W) comes from dividing the work by 0.50 s or otherwise mishandling the time. Only proper application of P = W/t, an outside formula not given in the passage, yields 25 W.
> `validation: {adversarial_pass: true, blind_solve_pass: true, answer_basis_ok: true}`

**Q3** (`…_q03` · Skill 4 · answer_basis `data_interpretation` · easy · correct **B**)
> In the experiment, the investigators varied the load mass and measured the maximum angular speed of the forearm during the lift. According to the passage, what relationship did they observe between load and angular speed?
> - A) Heavier loads were raised at higher maximum angular speeds.
> - B) Lighter loads were raised at higher maximum angular speeds.
> - C) Maximum angular speed was independent of the load mass.
> - D) Maximum angular speed peaked at an intermediate load mass.
>
> *Explanation:* The passage states directly that the investigators "varied the load mass and measured the maximum angular speed achieved during the lift, observing that lighter loads were raised more rapidly." This is exactly the relationship described in choice B: decreasing load corresponds to higher angular speed. Choice A reverses the observed trend and contradicts the passage. Choice C asserts no dependence on load, but the passage explicitly reports that the speed changed with load. Choice D describes a non-monotonic (peaked) relationship for which the passage provides no evidence; the text describes a simple inverse trend, not a maximum at an intermediate mass.
> `validation: {adversarial_pass: true, blind_solve_pass: true, answer_basis_ok: true}`

**Q4** (`…_q04` · Skill 1 · answer_basis `from_passage` · easy · correct **C**)
> According to the passage, when the forearm is NOT horizontal, why does only part of the muscle tension contribute to rotating the forearm about the elbow?
> - A) The entire tension always acts to rotate the forearm, but friction at the joint reduces its effect.
> - B) The parallel component produces torque while the perpendicular component is transmitted to the joint.
> - C) Only the component of tension perpendicular to the forearm produces torque, while the parallel component is transmitted to the joint.
> - D) The tension is reduced because the lever arm of the load increases as the forearm rotates upward.
>
> *Explanation:* The passage states explicitly that when the forearm is not horizontal, only the component of the muscle tension perpendicular to the forearm contributes to torque, while the parallel component is transmitted to the joint. This makes C correct. B reverses the roles of the two components; it is the perpendicular component, not the parallel one, that generates torque. A invokes friction, which the passage never mentions, and incorrectly claims the entire tension rotates the forearm. D conflates the geometry of the load's lever arm with the resolution of the muscle tension into components; the passage attributes the effect to component resolution, not to a change in the load's lever arm reducing tension.
> `validation: {adversarial_pass: true, blind_solve_pass: true, answer_basis_ok: true}`

> *Note on this passage's quality mix:* three of its four questions are `from_passage`/`data_interpretation` items that the explanation answers by quoting the passage almost verbatim (low independent difficulty despite passing both checks) — a realistic, not best-case, sample. Q2 is the strongest item (genuine `apply_knowledge` calculation with worked distractors).

### 5.4 CARS passage example — `CARS_P0001` (Dance) with ALL 5 questions

Subject: *Dance* (humanities) · 572 words · 5 questions.

**Passage text:**
> It is a curious feature of dance that, among the arts, it has been simultaneously the most universally practiced and the least esteemed by those who write about art. Painting leaves its canvas; music, since the advent of notation and later recording, leaves its score and its sound; literature persists in ink. Dance, by contrast, has traditionally vanished in the very moment of its making. The dancer's body is at once the instrument, the material, and the work—a fact that has tempted critics to treat dance as a lesser, because more ephemeral, form. Yet I want to argue that this supposed deficiency conceals dance's distinctive claim upon our attention.
>
> Consider the standard hierarchy of the arts inherited from the eighteenth century, in which the value of a work was thought to rise with its capacity to outlast the circumstances of its creation. By this measure dance must fail, for no notation has ever captured movement with the fidelity that a score captures pitch and rhythm. Labanotation and its rivals record positions and trajectories, but they cannot fix the weight, hesitation, and breath that distinguish a living performance from a diagram. The very project of preserving dance, one might say, betrays a misunderstanding of what dance is.
>
> Here the partisans of permanence and the partisans of presence reach an impasse. The former insist that an art which cannot be transmitted intact across generations forfeits its place in the cultural record; the latter reply that transmissibility is an alien standard, imported from arts whose materials happen to be more docile. I am inclined toward the second view, though not without qualification. To celebrate dance solely for its evanescence is to romanticize a limitation, and to ignore the elaborate techniques—apprenticeship, imitation, the patient correction of the body by the teacher's hand—through which dance traditions have in fact endured for centuries without notation. The ballet that reaches us today is not the ballet of the seventeenth-century court, but neither is it wholly severed from it. Transmission occurs; it simply occurs through bodies rather than documents.
>
> This observation unsettles a deeper assumption. We tend to imagine that memory resides in objects—books, recordings, monuments—and that whatever is not so deposited is lost. But the dancer's memory is a memory of muscle and nerve, sustained by repetition and passed from one practitioner to the next in an unbroken chain of demonstration. Such embodied knowledge is no less real for being unwritten; it is, if anything, more demanding, since it cannot be retrieved by anyone who has not first undergone the discipline of acquiring it. One cannot read a dance into one's possession.
>
> There is, admittedly, a cost to this mode of preservation. What survives is selective, shaped by the prestige of particular institutions and the survival of particular lineages. A folk dance whose last practitioners die unobserved leaves no trace, whereas a canonical work is rehearsed and revived precisely because someone judged it worth the labor. The history of dance is thus, more nakedly than the history of other arts, a history of what bodies were thought worth training. To study dance is therefore to study not merely a sequence of aesthetic objects but a continuous social negotiation over which movements deserve to be remembered—and, by implication, over whose bodies are entrusted with remembering them. The vanishing of dance, far from disqualifying it from serious thought, makes it an unusually honest witness to the conditions of cultural survival itself.

**Q1** (`CARS_P0001_q01` · *Foundations of Comprehension* · correct **A**)
> As used in the third paragraph, the phrase "arts whose materials happen to be more docile" most nearly refers to arts whose:
> - A) media can be readily fixed and preserved in durable form
> - B) practitioners are easier to discipline and train
> - C) audiences are more receptive to traditional standards
> - D) techniques require less rigorous apprenticeship to master
>
> *Explanation:* The contrast is between dance, which "vanishes," and arts like painting, music, and literature whose materials persist (canvas, score, ink). "Docile materials" therefore means media that submit easily to being captured and preserved—paragraph 1 lists exactly these durable forms. B confuses 'materials' with practitioners; C and D introduce audiences and training not implied by the phrase.
> `validation: {adversarial_pass: true, blind_solve_pass: true}`

**Q2** (`…_q02` · *Foundations of Comprehension* · correct **D**)
> Which of the following best states the central thesis of the passage?
> - A) Dance has been unfairly ranked below other arts because eighteenth-century critics misunderstood notation.
> - B) Embodied transmission is a superior method of artistic preservation that other arts should adopt.
> - C) Dance's evanescence should be celebrated as the purest expression of artistic freedom.
> - D) Dance's seeming impermanence is not a deficiency but reveals it as an especially honest record of how cultures decide what to preserve.
>
> *Explanation:* The author signals the thesis in paragraph 1 ("this supposed deficiency conceals dance's distinctive claim") and confirms it in the final paragraph: the vanishing of dance "makes it an unusually honest witness to the conditions of cultural survival." A captures only a subordinate point; B overstates ('superior'/'should adopt'); C is the romanticizing view the author explicitly rejects ('to romanticize a limitation').
> `validation: {adversarial_pass: true, blind_solve_pass: true}`

**Q3** (`…_q03` · *Reasoning Within the Text* · correct **B**)
> The author's overall position on the debate between "partisans of permanence" and "partisans of presence" is best described as:
> - A) fully aligned with the partisans of presence, since dance's value lies in its evanescence
> - B) leaning toward presence while rejecting the pure celebration of impermanence
> - C) neutral, since both positions rest on equally flawed assumptions
> - D) aligned with permanence, because dance traditions endure across centuries
>
> *Explanation:* In paragraph 3 the author writes, "I am inclined toward the second view, though not without qualification," and immediately adds that "to celebrate dance solely for its evanescence is to romanticize a limitation." This is a qualified leaning toward presence. A ignores the qualification; D mistakes the author's point that transmission occurs through bodies for an endorsement of the permanence camp; C wrongly claims neutrality.
> `validation: {adversarial_pass: true, blind_solve_pass: true}`

**Q4** (`…_q04` · *Reasoning Beyond the Text* · correct **C**)
> Suppose researchers develop a motion-capture technology so precise it records the exact weight, breath, and hesitation of every performance. How would this development most likely affect the author's argument?
> - A) It would fully refute the thesis, since dance would no longer vanish in the moment of its making.
> - B) It would confirm the partisans of permanence, proving notation can capture living performance.
> - C) It would challenge the claim that notation fails to fix a living performance, but not the argument that dance reveals social negotiations over cultural survival.
> - D) It would leave the argument entirely untouched, since the author dismisses all forms of recording as irrelevant.
>
> *Explanation:* The technology directly addresses paragraph 2's claim that notation cannot "fix the weight, hesitation, and breath" of a performance, so that specific point is challenged. However, the deeper thesis (final paragraph) concerns dance as a "continuous social negotiation over which movements deserve to be remembered"—a claim about whose work is judged worth preserving, which capture technology does not resolve. A overreaches; B mischaracterizes the broader argument; D wrongly says the author dismisses all recording.
> `validation: {adversarial_pass: true, blind_solve_pass: true}`

**Q5** (`…_q05` · *Reasoning Beyond the Text* · correct **A**)
> Which of the following situations is most analogous to the mode of preservation the author considers distinctive of dance?
> - A) A master blacksmith teaches an apprentice forging techniques solely by guided practice at the anvil, with no written manual.
> - B) A museum digitizes its entire collection of paintings so they can be viewed online worldwide.
> - C) A composer's unpublished symphony is discovered in an archive and performed for the first time.
> - D) A historian reconstructs an extinct language by studying surviving inscriptions on stone.
>
> *Explanation:* Paragraphs 3 and 4 describe dance's preservation as "embodied knowledge"—"a memory of muscle and nerve, sustained by repetition and passed from one practitioner to the next" through "apprenticeship, imitation, the patient correction of the body." The blacksmith-apprentice case matches this person-to-person, body-based, document-free transmission. B, C, and D all rely on durable external objects (digital files, scores, inscriptions), the very 'documentary' model the author contrasts with embodied knowledge.
> `validation: {adversarial_pass: true, blind_solve_pass: true}`

---

## 6. Distribution Statistics

Computed across the three production runs (`runs/prod_discrete`, `runs/prod_science`, `runs/prod_cars`).

### 6.1 Totals

| Output | Passages | Questions |
|---|---|---|
| Discrete | — | **786** |
| Science passages | 18 | **91** |
| CARS passages | 30 | **180** |
| **Total questions** | | **1,057** |

### 6.2 By section

**Discrete questions (786):**
| Section | Count |
|---|---|
| Biological and Biochemical Foundations of Living Systems | 369 |
| Chemical and Physical Foundations of Biological Systems | 265 |
| Psychological, Social, and Biological Foundations of Behavior | 152 |

**Science passages (18 passages / 91 questions):**
| Section | Passages | Questions |
|---|---|---|
| Biological and Biochemical Foundations | 9 | 51 |
| Chemical and Physical Foundations | 5 | 22 |
| Psychological, Social, and Biological Foundations of Behavior | 4 | 18 |

CARS is its own section (53-question section on the real exam); all 30 passages / 180 questions are CARS.

### 6.3 By question type (top level)

| Type | Questions |
|---|---|
| Discrete (standalone) | 786 |
| Science passage-linked | 91 |
| CARS passage-linked | 180 |

### 6.4 By difficulty

(Difficulty is tracked for discrete + science only; CARS records no per-question difficulty field.)

| Difficulty | Discrete | % | Science | % |
|---|---|---|---|---|
| easy | 172 | 21.9% | 22 | 24.2% |
| medium | 463 | 58.9% | 42 | 46.2% |
| hard | 151 | 19.2% | 27 | 29.7% |

(Target weights were 20/50/30; discrete lands close, science skews slightly harder.)

### 6.5 By skill (the Skill 1/2/3/4 breakdown)

**Discrete (786) — target 35/45/10/10:**
| Skill | Count | % |
|---|---|---|
| Skill 1 — Knowledge of Scientific Concepts | 281 | 35.8% |
| Skill 2 — Scientific Reasoning & Problem-Solving | 336 | 42.7% |
| Skill 3 — Research Design | 79 | 10.1% |
| Skill 4 — Data-Based & Statistical Reasoning | 90 | 11.5% |

The discrete distribution tracks the AAMC target almost exactly.

**Science questions (91) — target weights 15/35/25/25, but basis→skill mapping shifts it:**
| Skill | Count | % |
|---|---|---|
| Skill 1 | 31 | 34.1% |
| Skill 2 | 30 | 33.0% |
| Skill 3 | 6 | 6.6% |
| Skill 4 | 24 | 26.4% |

(Skill 1 is over-represented and Skill 3 under-represented vs. the configured weights, because `build_question_plan` maps `from_passage→skill_1` and only draws from the configured weights ~30% of the time.)

**CARS skill_type (180) — target ~30/30/40:**
| Skill type | Count | % |
|---|---|---|
| Foundations of Comprehension | 62 | 34.4% |
| Reasoning Within the Text | 52 | 28.9% |
| Reasoning Beyond the Text | 66 | 36.7% |

### 6.6 Science answer-basis distribution (91)

| answer_basis | Count |
|---|---|
| from_passage | 34 |
| apply_knowledge | 30 |
| data_interpretation | 27 |

### 6.7 Answer-key balance (correct-answer position)

| | A | B | C | D |
|---|---|---|---|---|
| Discrete (786) | 208 | 189 | 208 | 181 |
| Science (91) | 25 | 22 | 22 | 22 |
| CARS (180) | 60 | 53 | 39 | 28 |

Discrete and science are well balanced (the per-question target-letter trick works). **CARS is noticeably skewed toward A/B** (60/53 vs 39/28) despite the `balanced_answer_positions` shuffle — worth flagging, since CARS generates all questions in one call and the model appears not to honor the assigned positions as reliably as the single-question pipelines.

### 6.8 Passage word counts and questions per passage

| | Avg words | Min | Max | Avg Q/passage | Q/passage distribution |
|---|---|---|---|---|---|
| CARS (30) | **570.6** | 549 | 585 | 6.00 | 5→8, 6→14, 7→8 passages |
| Science (18) | **272.3** | 232 | 317 | 5.06 | 3→1, 4→6, 5→6, 6→1, 7→4 passages |

CARS passages sit squarely in the 500–600 target band; science passages sit in the 200–350 band. Figures: 2 of 18 science passages carry a figure spec (`SP_CP_4E_016_01`, `SP_BB_1A_006_01`, both rendered to PNG); 0 questions carry their own figure.

---

## 7. Notable observations for the reviewer

- **Acceptance funnel is healthy:** ~67% (discrete) and ~70% (science) of generation attempts survive both checks; only 14 and 3 slots respectively exhausted retries. Cost ≈ $0.08–0.09 per accepted question.
- **The blind-solve check is a self-consistency test, not an external oracle:** for discrete/science it runs on Haiku (an independent, cheaper model) and passes only when that model independently reproduces the keyed answer; for **CARS it runs on the main generation model**, so it is a weaker independent check there.
- **CARS has no metrics tracker and no per-question difficulty/skill funnel logging**, and its answer key is skewed toward A/B — the batch-generation design appears to honor assigned answer positions less faithfully than the one-question-at-a-time pipelines.
- **Science skill mix diverges from its configured weights** because answer-basis is mapped to a "natural" skill 70% of the time, inflating Skill 1 and starving Skill 3.
- **Figures were enabled in the science prod run but barely used** (2/18 passages); the structural-enforcement backstop exists for categories 5A/5B/5D/5E but the prod cluster sample contained few such passages.
```
