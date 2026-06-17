"""Pipeline for generating and validating discrete MCAT questions."""

import asyncio
import json
import logging
import random
from typing import Optional

from ..config import Config
from ..llm_client import LLMClient
from ..checkpoint import CheckpointManager, OutputWriter
from ..metrics import MetricsTracker, TopicMetrics
from ..prompts.discrete import (
    generation_prompt,
    adversarial_review_prompt,
    blind_solve_prompt,
)
from ..schemas import (
    RawDiscreteQuestion,
    AdversarialReview,
    BlindSolveResult,
    explanation_has_phantom_option,
)

logger = logging.getLogger(__name__)


async def validate_question(
    client: LLMClient,
    question_data: dict,
    topic_data: dict,
    temperature: float,
    checker_model: str,
    metrics: TopicMetrics,
) -> dict:
    """Run adversarial review + blind solve on a question.

    Adversarial review runs on the client's default model; blind solve runs on
    `checker_model` so it serves as an independent (cheaper) checker. Token
    usage for each call is recorded into `metrics`.

    Returns validation dict with pass/fail for each check.
    """
    # Run both validation steps concurrently
    review_msgs = adversarial_review_prompt(question_data, topic_data)
    solve_msgs = blind_solve_prompt(question_data)

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
        "blind_solve_match": False,
    }

    # Parse adversarial review
    if isinstance(results[0], dict):
        try:
            review = AdversarialReview(**results[0])
            validation["adversarial_pass"] = review.passed
            validation["adversarial_issues"] = review.issues
            if not review.passed:
                # Genuine rejection (parsed cleanly) — surface the reasons so we
                # can judge whether the reviewer is legitimately strict.
                logger.info(
                    f"Adversarial review REJECTED {topic_data.get('topic_id', '?')}: "
                    f"{review.issues}"
                )
        except Exception as e:
            logger.warning(f"Failed to parse adversarial review: {e}. Raw: {results[0]}")
    else:
        logger.warning(f"Adversarial review failed: {results[0]}")

    # Parse blind solve
    if isinstance(results[1], dict):
        try:
            solve = BlindSolveResult(**results[1])
            correct = question_data["correct_answer"].strip().upper()
            chosen = solve.chosen_answer.strip().upper()
            validation["blind_solve_pass"] = chosen == correct
            validation["blind_solve_match"] = chosen == correct
            if not validation["blind_solve_pass"]:
                logger.debug(
                    f"Blind solve mismatch: chose {chosen}, correct is {correct}. "
                    f"Confidence: {solve.confidence}"
                )
        except Exception as e:
            logger.warning(f"Failed to parse blind solve: {e}")
    else:
        logger.warning(f"Blind solve failed: {results[1]}")

    return validation


async def generate_and_validate_question(
    client: LLMClient,
    topic_data: dict,
    config: Config,
    question_number: int,
    metrics: TopicMetrics,
    previous_stems: list[str] = None,
) -> Optional[dict]:
    """Generate a single question with retries until it passes validation.

    Records funnel and token-usage counters into `metrics`. `previous_stems`
    (stems already accepted for this topic) is forwarded to the generation
    prompt for within-topic diversity. Returns the validated question dict, or
    None if all retries exhausted.
    """
    max_retries = config.discrete.max_retries

    for attempt in range(max_retries + 1):
        metrics.questions_attempted += 1

        # Step 1: Generate
        try:
            gen_msgs, assigned_skill = generation_prompt(topic_data, previous_stems)
            raw = await client.generate_json(
                gen_msgs,
                temperature=config.discrete.temperature_generate,
                # LaTeX-heavy questions (verbose $\text{...}$ plus JSON-escaped
                # backslashes) overran the old 1024 cap and were truncated
                # mid-string, producing unparseable JSON (~32% loss). Raised
                # 2048 -> 3072 for extra headroom after parse-fails persisted
                # (longest valid discrete question seen was ~980 tokens).
                max_tokens=3072,
                on_usage=lambda m, i, o: metrics.record_usage("generation", m, i, o),
            )
            question = RawDiscreteQuestion(**raw)
            question_data = question.model_dump()
        except Exception as e:
            logger.warning(
                f"Generation failed for {topic_data['topic_id']} "
                f"q{question_number} attempt {attempt}: {e}"
            )
            continue

        metrics.generation_parsed += 1

        # DETECT-AND-REJECT a phantom fifth option referenced in the explanation
        # prose (the choices object is already A-D). Deterministic guarantee that
        # no "Option E is wrong..." explanation is ever accepted: reject the whole
        # attempt and let the retry regenerate (never edit the explanation).
        phantom = explanation_has_phantom_option(
            question_data.get("explanation", ""), set(question_data.get("choices", {}))
        )
        if phantom:
            logger.info(
                f"phantom-option REJECT {topic_data['topic_id']} q{question_number} "
                f"(attempt {attempt}): explanation references {phantom!r}"
            )
            continue

        # Step 2: Validate
        try:
            validation = await validate_question(
                client,
                question_data,
                topic_data,
                config.discrete.temperature_validate,
                config.discrete.discrete_checker_model,
                metrics,
            )
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            continue

        # Record the validation funnel (both checks ran for this attempt)
        metrics.adversarial_runs += 1
        if validation["adversarial_pass"]:
            metrics.adversarial_passes += 1
        metrics.blind_solve_runs += 1
        if validation["blind_solve_pass"]:
            metrics.blind_solve_passes += 1

        # Step 3: Check if passed
        if validation["adversarial_pass"] and validation["blind_solve_pass"]:
            metrics.accepted += 1
            question_id = f"{topic_data['topic_id']}_q{question_number:03d}"
            return {
                "question_id": question_id,
                "topic_id": topic_data["topic_id"],
                "section": topic_data["section"],
                "content_category": topic_data["content_category"],
                "topic_group": topic_data["topic_group"],
                "topic": topic_data["topic"],
                "subtopics_tested": question_data.get("subtopics_tested", []),
                "stem": question_data["stem"],
                "choices": question_data["choices"],
                "correct_answer": question_data["correct_answer"],
                "explanation": question_data["explanation"],
                "difficulty": question_data.get("difficulty", "medium"),
                # Persist the pipeline-ASSIGNED skill (authoritative), not the
                # model's echoed skill_tested — the echo can be blank or mislabeled.
                "skill_tested": assigned_skill,
                "validation": {
                    "adversarial_pass": validation["adversarial_pass"],
                    "blind_solve_pass": validation["blind_solve_pass"],
                },
            }
        else:
            issues = validation.get("adversarial_issues", [])
            logger.debug(
                f"Question failed validation (attempt {attempt}): "
                f"adversarial={'PASS' if validation['adversarial_pass'] else 'FAIL'}, "
                f"blind_solve={'PASS' if validation['blind_solve_pass'] else 'FAIL'}, "
                f"issues={issues}"
            )

    metrics.slots_failed += 1
    logger.warning(
        f"Exhausted retries for {topic_data['topic_id']} q{question_number}"
    )
    return None


async def process_topic(
    client: LLMClient,
    topic_data: dict,
    config: Config,
    checkpoint: CheckpointManager,
    writer: OutputWriter,
    metrics: TopicMetrics,
):
    """Generate all questions for a single topic."""
    topic_id = topic_data["topic_id"]
    target = config.discrete.questions_per_topic

    # Check existing progress
    completed = checkpoint.get_progress(topic_id)
    if completed >= target:
        checkpoint.mark_complete(topic_id)
        return

    logger.info(
        f"Processing {topic_id}: {topic_data['topic']} "
        f"({completed}/{target} done)"
    )

    question_numbers = list(range(completed + 1, target + 1))

    # Generate sequentially so each new question can see the stems already
    # accepted for THIS topic and deliberately cover a different angle
    # (within-topic diversity). accepted_stems is per-topic and starts empty;
    # only accepted questions' stems are added.
    accepted_stems: list[str] = []
    for qn in question_numbers:
        result = await generate_and_validate_question(
            client, topic_data, config, qn, metrics, accepted_stems
        )
        if result is None:
            logger.warning(f"{topic_id}: q{qn} failed after retries")
            continue
        writer.append(result)
        accepted_stems.append(result["stem"])
        completed += 1
        checkpoint.update_progress(topic_id, completed)

    if completed >= target:
        checkpoint.mark_complete(topic_id)
        logger.info(f"Completed {topic_id}: {completed} questions generated")
    else:
        # Don't mark done below quota — preserve the partial count so a later
        # run (or --recount) can resume and fill the remaining slots.
        checkpoint.update_progress(topic_id, completed)
        logger.info(
            f"{topic_id}: only {completed}/{target} accepted — left open for resume"
        )


async def run_discrete_pipeline(
    client: LLMClient,
    topics: list[dict],
    config: Config,
    max_topics: Optional[int] = None,
    seed: Optional[int] = None,
):
    """Run the full discrete question generation pipeline.

    If `max_topics` is set, at most that many (not-yet-completed) topics are
    processed this run — handy for small test runs. If `seed` is set, the topic
    shuffle uses a dedicated seeded RNG so two runs starting from empty
    checkpoints select the identical set of topics (apples-to-apples).
    """
    # Filter to non-CARS topics
    discrete_topics = [
        t for t in topics
        if t["section"] != "Critical Analysis and Reasoning Skills"
    ]

    checkpoint = CheckpointManager(
        f"{config.checkpoint_dir}/discrete"
    )
    writer = OutputWriter(
        f"{config.output_dir}/discrete_questions.jsonl"
    )

    # Filter out already-completed topics
    completed = checkpoint.get_completed_topics()
    remaining = [t for t in discrete_topics if t["topic_id"] not in completed]

    logger.info(
        f"Discrete pipeline: {len(discrete_topics)} total topics, "
        f"{len(completed)} completed, {len(remaining)} remaining"
    )

    # Shuffle to distribute topics across sections (helps if interrupted).
    # With a fixed seed, use a dedicated RNG so topic selection is identical
    # across runs regardless of any other random usage.
    if seed is not None:
        random.Random(seed).shuffle(remaining)
    else:
        random.shuffle(remaining)

    # Optionally cap how many topics we process this run (after the shuffle so
    # the sample is spread across sections).
    if max_topics is not None and len(remaining) > max_topics:
        logger.info(
            f"--max-topics: limiting to {max_topics} of {len(remaining)} "
            f"remaining topics this run"
        )
        remaining = remaining[:max_topics]

    tracker = MetricsTracker(
        model=config.model,
        checker_model=config.discrete.discrete_checker_model,
        metrics_path=f"{config.output_dir}/generation_metrics.json",
    )

    for i, topic in enumerate(remaining):
        logger.info(f"[{i+1}/{len(remaining)}] {topic['topic_id']}: {topic['topic']}")
        bucket = tracker.start_topic(
            topic["topic_id"], topic["topic"], config.discrete.questions_per_topic
        )
        try:
            await process_topic(client, topic, config, checkpoint, writer, bucket)
        except Exception as e:
            logger.error(f"Fatal error processing {topic['topic_id']}: {e}")
            # Continue to next topic rather than crashing
        finally:
            # Log + persist this topic's metrics (captures partial work on error)
            tracker.finish_topic()

    stats = checkpoint.get_stats()
    total_questions = writer.count()
    logger.info(
        f"Discrete pipeline complete. "
        f"Topics: {stats['completed_topics']}/{len(discrete_topics)}, "
        f"Total questions: {total_questions}"
    )

    # Aggregate metrics across all topics processed this run.
    tracker.log_summary()
    tracker.write_json()
