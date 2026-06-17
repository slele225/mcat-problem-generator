"""Pipeline for generating and validating CARS passages and questions.

A unit of work is a PASSAGE: a 500-600 word CARS passage (assigned a STRUCTURE
type — single_voice or multi_position) followed by per-question-generated,
validated questions. Mirrors the science-passage pipeline's structure —
sequential per-question generation with within-passage diversity, adversarial +
blind-solve validation, a MetricsTracker, CheckpointManager, OutputWriter — but
for self-contained CARS passages. The blind-solve check runs on an INDEPENDENT
checker model (config.cars.cars_checker_model, default Sonnet) rather than the
main generation model.

Does NOT touch the discrete or science-passage pipelines.
"""

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Optional

from ..config import Config
from ..llm_client import LLMClient
from ..checkpoint import CheckpointManager, OutputWriter
from ..metrics import MetricsTracker, TopicMetrics
from ..prompts.cars import (
    passage_generation_prompt,
    passage_review_prompt,
    cars_question_prompt,
    cars_adversarial_review_prompt,
    cars_blind_solve_prompt,
    build_cars_question_plan,
    pick_structure_type,
)
from ..schemas import (
    RawCARSPassage,
    RawCARSQuestion,
    AdversarialReview,
    BlindSolveResult,
    explanation_has_phantom_option,
)

logger = logging.getLogger(__name__)


def _append_rejected_cars(output_dir: str, record: dict) -> None:
    """Best-effort append of a DIAGNOSTIC reject record to rejected_cars.jsonl.

    Captures a discarded CARS question attempt (failed adversarial review and/or
    blind-solve mismatch) so the blind-solve filter can be analyzed later. This is
    logging ONLY — it never changes the accept/reject decision. Any failure (I/O,
    serialization) is swallowed so a logging problem can never alter or crash a run.
    """
    try:
        path = Path(output_dir) / "rejected_cars.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _blind_solve_diagnostics(raw: dict) -> tuple[str, str]:
    """Extract (chosen_answer, confidence) from a raw blind-solve response for the
    reject diagnostic log ONLY — this never influences the accept/reject decision.

    `chosen` is upper-cased, or "" when the field is absent/blank. `confidence` is
    the model's value, or "unknown" when the response omitted it or left it blank.
    That "unknown" is deliberately distinct from the "" left in place when the
    blind-solve response could not be parsed at all (a non-dict result).
    """
    chosen = raw.get("chosen_answer")
    chosen = chosen.strip().upper() if isinstance(chosen, str) and chosen.strip() else ""
    conf = raw.get("confidence")
    conf = conf.strip() if isinstance(conf, str) and conf.strip() else "unknown"
    return chosen, conf


def _pick_num_questions(config: Config) -> int:
    """Number of questions for one passage.

    Drawn uniformly from cars.questions_per_passage_range when that range is set
    (mirrors the science-passage range; real CARS carries ~5-7), otherwise the
    fixed cars.questions_per_passage.
    """
    rng = config.cars.questions_per_passage_range
    if rng:
        lo, hi = int(rng[0]), int(rng[1])
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)
    return config.cars.questions_per_passage


async def generate_passage(
    client: LLMClient,
    subject: str,
    structure_type: str,
    config: Config,
    metrics: TopicMetrics,
) -> Optional[dict]:
    """Generate and validate a single CARS passage of the given structure type.

    Returns passage dict or None if validation fails after retries.
    """
    word_min, word_max = config.cars.passage_word_range

    for attempt in range(config.cars.max_retries + 1):
        # Generate
        try:
            gen_msgs = passage_generation_prompt(
                subject, word_min, word_max, structure_type
            )
            raw = await client.generate_json(
                gen_msgs,
                temperature=config.cars.temperature_generate,
                # CARS passages run 500-600 words; raised 2048 -> 3072 for
                # headroom so long passages aren't truncated into unparseable JSON.
                max_tokens=3072,
                on_usage=lambda m, i, o: metrics.record_usage("passage_generation", m, i, o),
            )
            passage = RawCARSPassage(**raw)
        except Exception as e:
            logger.warning(f"Passage generation failed (attempt {attempt}): {e}")
            continue

        # Check word count locally (fast check)
        word_count = len(passage.passage_text.split())
        if word_count < word_min - 50 or word_count > word_max + 50:
            logger.debug(
                f"Passage word count {word_count} outside range "
                f"[{word_min}, {word_max}], retrying"
            )
            continue

        # LLM-based quality review (structure-aware: multi_position is not
        # penalized for presenting multiple attributed views).
        try:
            review_msgs = passage_review_prompt(
                passage.passage_text, word_min, word_max, structure_type
            )
            review_raw = await client.generate_json(
                review_msgs,
                temperature=config.cars.temperature_validate,
                on_usage=lambda m, i, o: metrics.record_usage("passage_review", m, i, o),
            )
            if review_raw.get("passed", False):
                return {
                    "passage_text": passage.passage_text,
                    "subject": subject,
                    "structure_type": structure_type,
                    "word_count": word_count,
                }
            else:
                issues = review_raw.get("issues", [])
                logger.debug(f"Passage review failed: {issues}")
        except Exception as e:
            logger.warning(f"Passage review failed: {e}")
            # If review itself fails, accept the passage if word count is ok
            if word_min <= word_count <= word_max:
                return {
                    "passage_text": passage.passage_text,
                    "subject": subject,
                    "structure_type": structure_type,
                    "word_count": word_count,
                }

    logger.warning(
        f"Failed to generate valid passage for subject: {subject} "
        f"(structure={structure_type})"
    )
    return None


async def validate_cars_question(
    client: LLMClient,
    passage_text: str,
    question_data: dict,
    temperature: float,
    checker_model: str,
    metrics: TopicMetrics,
) -> dict:
    """Run adversarial review + blind solve on a CARS question.

    Adversarial review runs on the main model; the blind solve runs on
    `checker_model` (independent, cheaper than Opus) so the check is not the
    generator marking its own work. Captures the blind-solver's chosen answer and
    confidence for diagnostic reject logging only (these never affect the
    accept/reject decision).
    """
    review_msgs = cars_adversarial_review_prompt(passage_text, question_data)
    solve_msgs = cars_blind_solve_prompt(passage_text, question_data)

    results = await asyncio.gather(
        client.generate_json(
            review_msgs,
            temperature=temperature,
            on_usage=lambda m, i, o: metrics.record_usage("review", m, i, o),
        ),
        client.generate_json(
            solve_msgs,
            temperature=temperature,
            model=checker_model,
            on_usage=lambda m, i, o: metrics.record_usage("blind_solve", m, i, o),
        ),
        return_exceptions=True,
    )

    validation = {
        "adversarial_pass": False,
        "blind_solve_pass": False,
        "adversarial_issues": [],
        # Threaded through for diagnostic reject logging only (does NOT affect the
        # accept/reject decision): the blind-solver's chosen answer + confidence.
        "blind_solve_chosen": "",
        "blind_solve_confidence": "",
    }

    if isinstance(results[0], dict):
        try:
            review = AdversarialReview(**results[0])
            validation["adversarial_pass"] = review.passed
            validation["adversarial_issues"] = review.issues
        except Exception as e:
            logger.warning(f"Failed to parse adversarial review: {e}. Raw: {results[0]}")
    else:
        logger.warning(f"Adversarial review failed: {results[0]}")

    if isinstance(results[1], dict):
        # Diagnostic capture for the reject log (NEVER affects the pass decision):
        # pull chosen + confidence from the raw response so BOTH are recorded even
        # when the strict parse below rejects the object.
        validation["blind_solve_chosen"], validation["blind_solve_confidence"] = (
            _blind_solve_diagnostics(results[1])
        )
        try:
            solve = BlindSolveResult(**results[1])
            correct = question_data["correct_answer"].strip().upper()
            chosen = solve.chosen_answer.strip().upper()
            validation["blind_solve_pass"] = chosen == correct
        except Exception as e:
            logger.warning(f"Failed to parse blind solve: {e}")
    else:
        logger.warning(f"Blind solve failed: {results[1]}")

    return validation


async def generate_and_validate_cars_question(
    client: LLMClient,
    passage_text: str,
    structure_type: str,
    q_plan: dict,
    config: Config,
    passage_id: str,
    question_number: int,
    metrics: TopicMetrics,
    previous_stems: list,
) -> Optional[dict]:
    """Generate ONE CARS question, retrying until it passes validation.

    Accepts iff adversarial review AND blind solve both pass. Tracks the funnel
    into `metrics`; logs each discarded attempt to rejected_cars.jsonl (best
    effort). Returns the validated question dict, or None after exhausting retries.
    """
    max_retries = config.cars.max_retries
    question_id = f"{passage_id}_q{question_number:02d}"

    for attempt in range(max_retries + 1):
        metrics.questions_attempted += 1

        try:
            gen_msgs = cars_question_prompt(
                passage_text, q_plan, previous_stems, structure_type
            )
            raw = await client.generate_json(
                gen_msgs,
                temperature=config.cars.temperature_generate,
                # Headroom so long stems aren't truncated into unparseable JSON
                # (matches the discrete/science generation cap).
                max_tokens=3072,
                on_usage=lambda m, i, o: metrics.record_usage("generation", m, i, o),
            )
            question = RawCARSQuestion(**raw)
            question_data = question.model_dump()
        except Exception as e:
            logger.warning(
                f"Generation failed for {passage_id} q{question_number} "
                f"attempt {attempt}: {e}"
            )
            continue

        metrics.generation_parsed += 1

        # DETECT-AND-REJECT a phantom fifth option referenced in the explanation
        # prose (the choices object is already A-D). Deterministic guarantee that
        # no "Option E is wrong..." explanation is ever accepted: reject the whole
        # attempt and let the retry regenerate (never edit the explanation). CARS
        # keeps its own inline no-fifth-option wording, but this check is shared.
        phantom = explanation_has_phantom_option(
            question_data.get("explanation", ""), set(question_data.get("choices", {}))
        )
        if phantom:
            logger.info(
                f"phantom-option REJECT {question_id} (attempt {attempt}): "
                f"explanation references {phantom!r}"
            )
            continue

        try:
            validation = await validate_cars_question(
                client,
                passage_text,
                question_data,
                config.cars.temperature_validate,
                config.cars.cars_checker_model,
                metrics,
            )
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            continue

        metrics.adversarial_runs += 1
        if validation["adversarial_pass"]:
            metrics.adversarial_passes += 1
        metrics.blind_solve_runs += 1
        if validation["blind_solve_pass"]:
            metrics.blind_solve_passes += 1

        if validation["adversarial_pass"] and validation["blind_solve_pass"]:
            metrics.accepted += 1
            return {
                "question_id": question_id,
                # Persist the pipeline-ASSIGNED skill type (authoritative), not the
                # model's echoed skill_type, so the saved tag is always correct.
                "skill_type": q_plan["skill_type"],
                "stem": question_data["stem"],
                "choices": question_data["choices"],
                "correct_answer": question_data["correct_answer"],
                "explanation": question_data["explanation"],
                "difficulty": question_data.get("difficulty", q_plan["difficulty"]),
                "validation": {
                    "adversarial_pass": validation["adversarial_pass"],
                    "blind_solve_pass": validation["blind_solve_pass"],
                },
            }

        # DIAGNOSTIC (does not change accept logic): the attempt is discarded
        # because at least one check failed. Record it to rejected_cars.jsonl so
        # the blind-solve filter can be analyzed later. Best-effort only.
        which_failed = []
        if not validation["adversarial_pass"]:
            which_failed.append("adversarial")
        if not validation["blind_solve_pass"]:
            which_failed.append("blind_solve")
        _append_rejected_cars(config.output_dir, {
            "passage_id": passage_id,
            "question_id": question_id,
            "attempt": attempt,
            "skill_type": q_plan.get("skill_type"),
            "difficulty": q_plan.get("difficulty"),
            "structure_type": structure_type,
            "target_answer": q_plan.get("target_answer"),
            "stem": question_data.get("stem"),
            "choices": question_data.get("choices"),
            "correct_answer": question_data.get("correct_answer"),
            "failed_checks": which_failed,
            "checks": {
                "adversarial_pass": validation["adversarial_pass"],
                "blind_solve_pass": validation["blind_solve_pass"],
            },
            "blind_solve_chosen": validation.get("blind_solve_chosen", ""),
            "blind_solve_confidence": validation.get("blind_solve_confidence", ""),
            "adversarial_issues": validation.get("adversarial_issues", []),
        })

        logger.debug(
            f"{passage_id} q{question_number} failed (attempt {attempt}): "
            f"adversarial={'PASS' if validation['adversarial_pass'] else 'FAIL'}, "
            f"blind_solve={'PASS' if validation['blind_solve_pass'] else 'FAIL'}, "
            f"issues={validation.get('adversarial_issues', [])}"
        )

    metrics.slots_failed += 1
    logger.warning(f"Exhausted retries for {passage_id} q{question_number}")
    return None


async def generate_cars_passage_set(
    client: LLMClient,
    passage_number: int,
    subject: str,
    category: str,
    config: Config,
    metrics: TopicMetrics,
) -> Optional[dict]:
    """Generate a complete CARS passage with per-question-generated questions.

    Assigns a STRUCTURE type (weighted by discipline `category`), generates and
    validates the passage, then generates each question ONE AT A TIME — each new
    question seeing the stems already accepted for THIS passage (within-passage
    diversity) and enforcing its assigned skill_type / difficulty / target answer
    letter. Accept iff adversarial review AND blind solve both pass.
    """
    passage_id = f"CARS_P{passage_number:04d}"

    structure_type = pick_structure_type(category)

    # Step 1: Generate and validate the passage.
    passage_data = await generate_passage(
        client, subject, structure_type, config, metrics
    )
    if passage_data is None:
        return None

    passage_text = passage_data["passage_text"]

    # Step 2: Plan and generate questions one at a time. The target count is drawn
    # per passage from the configured range (mirrors the science-passage pipeline).
    num_questions = _pick_num_questions(config)
    plan = build_cars_question_plan(num_questions, structure_type)

    accepted: list[dict] = []
    accepted_stems: list[str] = []
    for i, q_plan in enumerate(plan, start=1):
        result = await generate_and_validate_cars_question(
            client,
            passage_text,
            structure_type,
            q_plan,
            config,
            passage_id,
            len(accepted) + 1,
            metrics,
            accepted_stems,
        )
        if result is None:
            logger.warning(f"{passage_id}: question slot {i} failed after retries")
            continue
        accepted.append(result)
        accepted_stems.append(result["stem"])

    if not accepted:
        logger.warning(f"{passage_id}: no questions passed validation")
        return None

    return {
        "passage_id": passage_id,
        "passage_text": passage_text,
        "word_count": passage_data["word_count"],
        "subject": subject,
        "category": category,
        "structure_type": structure_type,
        "questions": accepted,
        "validation": {
            "passage_reviewed": True,
            "questions_validated": len(accepted),
            "target_questions": num_questions,
        },
    }


async def run_cars_pipeline(
    client: LLMClient,
    topics: list[dict],
    config: Config,
    max_topics: Optional[int] = None,
):
    """Run the full CARS passage generation pipeline.

    CARS is passage-based rather than topic-based, so `max_topics` caps how
    many passages are generated this run (each passage is a unit of work).
    Passages are generated concurrently in batches; each carries its OWN metrics
    bucket so the run-level funnel/cost (written to cars_generation_metrics.json)
    is accurate despite the concurrency.
    """
    checkpoint = CheckpointManager(f"{config.checkpoint_dir}/cars")
    writer = OutputWriter(f"{config.output_dir}/cars_passages.jsonl")
    tracker = MetricsTracker(
        model=config.model,
        checker_model=config.cars.cars_checker_model,
        metrics_path=f"{config.output_dir}/cars_generation_metrics.json",
    )

    # Total passages to generate
    total_passages = config.cars.passages_per_topic  # e.g., 100
    humanities = config.cars.humanities_subjects or [
        "Philosophy", "Ethics", "Literature", "Art", "Religion"
    ]
    social_sciences = config.cars.social_science_subjects or [
        "History", "Sociology", "Economics", "Psychology", "Anthropology"
    ]

    # De-duplicated subject assignment. The old `random.choice` drew WITH
    # replacement, so a run could cluster on the same subject (e.g. "Linguistics"
    # twice in three passages). Instead draw WITHOUT replacement from a shuffled
    # deck per category, reshuffling only when a deck empties — so subjects spread
    # across the full discipline list before any repeats. The 50/50
    # humanities/social-science split is preserved by choosing the category
    # probabilistically per passage. Called sequentially while building each
    # batch, so mutating the deck lists here is safe.
    _humanities_deck: list[str] = []
    _social_deck: list[str] = []

    def _draw(deck: list[str], source: list[str]) -> str:
        if not deck:
            deck.extend(source)
            random.shuffle(deck)
        return deck.pop()

    def pick_subject() -> tuple[str, str]:
        # 50/50 between humanities and social sciences; return the category too so
        # the structure-type weighting can favor multi_position for social science.
        if random.random() < 0.5:
            return _draw(_humanities_deck, humanities), "humanities"
        return _draw(_social_deck, social_sciences), "social_science"

    # Determine which passages are already done
    completed = checkpoint.get_completed_topics()
    start_from = len(completed)

    # --max-topics caps how many passages we generate this run.
    run_end = total_passages
    if max_topics is not None:
        run_end = min(total_passages, start_from + max_topics)
        logger.info(
            f"--max-topics: limiting CARS to {max(0, run_end - start_from)} "
            f"passage(s) this run"
        )

    logger.info(
        f"CARS pipeline: {total_passages} passages target, "
        f"{start_from} completed, {total_passages - start_from} remaining"
    )

    batch_size = config.cars.batch_size

    for batch_start in range(start_from, run_end, batch_size):
        batch_end = min(batch_start + batch_size, run_end)
        batch_range = range(batch_start + 1, batch_end + 1)

        tasks = []
        for pnum in batch_range:
            passage_id = f"CARS_P{pnum:04d}"
            if checkpoint.is_complete(passage_id):
                continue

            subject, category = pick_subject()
            bucket = TopicMetrics(topic_id=passage_id, topic=subject, target=0)
            tasks.append((pnum, passage_id, subject, category, bucket))

        if not tasks:
            continue

        # Run batch concurrently; each passage records into its own bucket.
        coros = [
            generate_cars_passage_set(client, pnum, subj, cat, config, bucket)
            for pnum, pid, subj, cat, bucket in tasks
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for (pnum, pid, subj, cat, bucket), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Failed {pid}: {result}")
            elif result is None:
                logger.warning(f"Failed to generate {pid}")
            else:
                bucket.target = result["validation"]["target_questions"]
                writer.append(result)
                checkpoint.mark_complete(pid)
                logger.info(
                    f"Completed {pid}: {len(result['questions'])} questions "
                    f"(structure={result['structure_type']})"
                )
            # Keep any bucket that did real work so the run metrics are complete,
            # even for passages whose question slots all failed.
            if bucket.questions_attempted > 0:
                tracker.topics.append(bucket)

        tracker.write_json()  # persist incrementally for crash-safety
        completed_now = checkpoint.get_completed_topics()
        logger.info(
            f"CARS progress: {len(completed_now)}/{total_passages} passages"
        )

    total = writer.count()
    logger.info(f"CARS pipeline complete. Total passages: {total}")

    tracker.log_summary()
    tracker.write_json()
