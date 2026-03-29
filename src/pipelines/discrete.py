"""Pipeline for generating and validating discrete MCAT questions."""

import asyncio
import json
import logging
import random
from typing import Optional

from ..config import Config
from ..llm_client import LLMClient
from ..checkpoint import CheckpointManager, OutputWriter
from ..prompts.discrete import (
    generation_prompt,
    adversarial_review_prompt,
    blind_solve_prompt,
)
from ..schemas import RawDiscreteQuestion, AdversarialReview, BlindSolveResult

logger = logging.getLogger(__name__)


async def validate_question(
    client: LLMClient,
    question_data: dict,
    topic_data: dict,
    temperature: float,
) -> dict:
    """Run adversarial review + blind solve on a question.

    Returns validation dict with pass/fail for each check.
    """
    # Run both validation steps concurrently
    review_msgs = adversarial_review_prompt(question_data, topic_data)
    solve_msgs = blind_solve_prompt(question_data)

    results = await asyncio.gather(
        client.generate_json(review_msgs, temperature=temperature),
        client.generate_json(solve_msgs, temperature=temperature),
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
        except Exception as e:
            logger.warning(f"Failed to parse adversarial review: {e}")
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
) -> Optional[dict]:
    """Generate a single question with retries until it passes validation.

    Returns the validated question dict, or None if all retries exhausted.
    """
    max_retries = config.discrete.max_retries

    for attempt in range(max_retries + 1):
        # Step 1: Generate
        try:
            gen_msgs = generation_prompt(topic_data)
            raw = await client.generate_json(
                gen_msgs,
                temperature=config.discrete.temperature_generate,
                max_tokens=1024,
            )
            question = RawDiscreteQuestion(**raw)
            question_data = question.model_dump()
        except Exception as e:
            logger.warning(
                f"Generation failed for {topic_data['topic_id']} "
                f"q{question_number} attempt {attempt}: {e}"
            )
            continue

        # Step 2: Validate
        try:
            validation = await validate_question(
                client,
                question_data,
                topic_data,
                config.discrete.temperature_validate,
            )
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            continue

        # Step 3: Check if passed
        if validation["adversarial_pass"] and validation["blind_solve_pass"]:
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
                "skill_tested": question_data.get("skill_tested", ""),
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
):
    """Generate all questions for a single topic."""
    topic_id = topic_data["topic_id"]
    target = config.discrete.questions_per_topic
    batch_size = config.discrete.batch_size

    # Check existing progress
    completed = checkpoint.get_progress(topic_id)
    if completed >= target:
        checkpoint.mark_complete(topic_id)
        return

    logger.info(
        f"Processing {topic_id}: {topic_data['topic']} "
        f"({completed}/{target} done)"
    )

    remaining = target - completed
    question_numbers = list(range(completed + 1, target + 1))

    # Process in batches
    for batch_start in range(0, remaining, batch_size):
        batch_nums = question_numbers[batch_start : batch_start + batch_size]

        tasks = [
            generate_and_validate_question(client, topic_data, config, qn)
            for qn in batch_nums
        ]
        results = await asyncio.gather(*tasks)

        # Write successful results
        successful = [r for r in results if r is not None]
        if successful:
            writer.append_batch(successful)
            completed += len(successful)
            checkpoint.update_progress(topic_id, completed)

        failed = len(batch_nums) - len(successful)
        if failed > 0:
            logger.warning(
                f"{topic_id}: {failed}/{len(batch_nums)} questions failed in batch"
            )

    checkpoint.mark_complete(topic_id)
    logger.info(f"Completed {topic_id}: {completed} questions generated")


async def run_discrete_pipeline(
    client: LLMClient,
    topics: list[dict],
    config: Config,
):
    """Run the full discrete question generation pipeline."""
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

    # Shuffle to distribute topics across sections (helps if interrupted)
    random.shuffle(remaining)

    for i, topic in enumerate(remaining):
        logger.info(f"[{i+1}/{len(remaining)}] {topic['topic_id']}: {topic['topic']}")
        try:
            await process_topic(client, topic, config, checkpoint, writer)
        except Exception as e:
            logger.error(f"Fatal error processing {topic['topic_id']}: {e}")
            # Continue to next topic rather than crashing
            continue

    stats = checkpoint.get_stats()
    total_questions = writer.count()
    logger.info(
        f"Discrete pipeline complete. "
        f"Topics: {stats['completed_topics']}/{len(discrete_topics)}, "
        f"Total questions: {total_questions}"
    )
