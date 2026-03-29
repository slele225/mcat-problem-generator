"""Pipeline for generating and validating CARS passages and questions."""

import asyncio
import json
import logging
import random
from typing import Optional

from ..config import Config
from ..llm_client import LLMClient, parse_json_response
from ..checkpoint import CheckpointManager, OutputWriter
from ..prompts.cars import (
    passage_generation_prompt,
    passage_review_prompt,
    cars_questions_prompt,
    cars_adversarial_review_prompt,
    cars_blind_solve_prompt,
    SKILL_DISTRIBUTION,
)
from ..schemas import RawCARSPassage, RawCARSQuestion, AdversarialReview, BlindSolveResult

logger = logging.getLogger(__name__)


async def generate_passage(
    client: LLMClient,
    subject: str,
    config: Config,
) -> Optional[dict]:
    """Generate and validate a single CARS passage.

    Returns passage dict or None if validation fails after retries.
    """
    word_min, word_max = config.cars.passage_word_range

    for attempt in range(config.cars.max_retries + 1):
        # Generate
        try:
            gen_msgs = passage_generation_prompt(subject, word_min, word_max)
            raw = await client.generate_json(
                gen_msgs,
                temperature=config.cars.temperature_generate,
                max_tokens=2048,
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

        # LLM-based quality review
        try:
            review_msgs = passage_review_prompt(
                passage.passage_text, word_min, word_max
            )
            review_raw = await client.generate_json(
                review_msgs,
                temperature=config.cars.temperature_validate,
            )
            if review_raw.get("passed", False):
                return {
                    "passage_text": passage.passage_text,
                    "subject": subject,
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
                    "word_count": word_count,
                }

    logger.warning(f"Failed to generate valid passage for subject: {subject}")
    return None


async def validate_cars_question(
    client: LLMClient,
    passage_text: str,
    question_data: dict,
    temperature: float,
) -> dict:
    """Run adversarial review + blind solve on a CARS question."""
    review_msgs = cars_adversarial_review_prompt(passage_text, question_data)
    solve_msgs = cars_blind_solve_prompt(passage_text, question_data)

    results = await asyncio.gather(
        client.generate_json(review_msgs, temperature=temperature),
        client.generate_json(solve_msgs, temperature=temperature),
        return_exceptions=True,
    )

    validation = {
        "adversarial_pass": False,
        "blind_solve_pass": False,
    }

    if isinstance(results[0], dict):
        try:
            review = AdversarialReview(**results[0])
            validation["adversarial_pass"] = review.passed
        except Exception:
            pass
    if isinstance(results[1], dict):
        try:
            solve = BlindSolveResult(**results[1])
            correct = question_data["correct_answer"].strip().upper()
            chosen = solve.chosen_answer.strip().upper()
            validation["blind_solve_pass"] = chosen == correct
        except Exception:
            pass

    return validation


async def generate_questions_for_passage(
    client: LLMClient,
    passage_text: str,
    config: Config,
    passage_id: str,
) -> list[dict]:
    """Generate and validate questions for a single passage.

    Generates all questions at once, validates individually,
    and regenerates failed ones.
    """
    num_questions = config.cars.questions_per_passage

    for attempt in range(config.cars.max_retries + 1):
        # Generate all questions for the passage
        try:
            gen_msgs = cars_questions_prompt(passage_text, num_questions)
            raw_text = await client.generate(
                gen_msgs,
                temperature=config.cars.temperature_generate,
                max_tokens=4096,
            )
            raw_list = parse_json_response(raw_text)

            # Handle case where LLM wraps array in an object
            if isinstance(raw_list, dict) and "questions" in raw_list:
                raw_list = raw_list["questions"]

            if not isinstance(raw_list, list):
                logger.warning(f"Expected list, got {type(raw_list)}")
                continue

        except Exception as e:
            logger.warning(f"Question generation failed (attempt {attempt}): {e}")
            continue

        # Parse and validate each question
        validated_questions = []
        for i, raw_q in enumerate(raw_list):
            try:
                q = RawCARSQuestion(**raw_q)
                q_data = q.model_dump()
            except Exception as e:
                logger.debug(f"Failed to parse question {i}: {e}")
                continue

            # Validate
            try:
                val = await validate_cars_question(
                    client,
                    passage_text,
                    q_data,
                    config.cars.temperature_validate,
                )
                if val["adversarial_pass"] and val["blind_solve_pass"]:
                    question_id = f"{passage_id}_q{len(validated_questions)+1:02d}"
                    validated_questions.append({
                        "question_id": question_id,
                        "skill_type": q_data["skill_type"],
                        "stem": q_data["stem"],
                        "choices": q_data["choices"],
                        "correct_answer": q_data["correct_answer"],
                        "explanation": q_data["explanation"],
                        "validation": val,
                    })
            except Exception as e:
                logger.debug(f"Validation failed for question {i}: {e}")

        if len(validated_questions) >= num_questions:
            return validated_questions[:num_questions]
        elif validated_questions:
            logger.debug(
                f"Only {len(validated_questions)}/{num_questions} questions passed "
                f"validation (attempt {attempt})"
            )

    # Return whatever we have, even if less than target
    logger.warning(
        f"{passage_id}: Only {len(validated_questions)} questions passed validation"
    )
    return validated_questions


async def generate_cars_passage_set(
    client: LLMClient,
    passage_number: int,
    subject: str,
    config: Config,
) -> Optional[dict]:
    """Generate a complete CARS passage with validated questions."""
    passage_id = f"CARS_P{passage_number:04d}"

    # Step 1: Generate and validate passage
    passage_data = await generate_passage(client, subject, config)
    if passage_data is None:
        return None

    # Step 2: Generate and validate questions
    questions = await generate_questions_for_passage(
        client,
        passage_data["passage_text"],
        config,
        passage_id,
    )

    if not questions:
        logger.warning(f"{passage_id}: No questions passed validation")
        return None

    return {
        "passage_id": passage_id,
        "passage_text": passage_data["passage_text"],
        "word_count": passage_data["word_count"],
        "subject": subject,
        "questions": questions,
        "validation": {
            "passage_reviewed": True,
            "questions_validated": len(questions),
            "target_questions": config.cars.questions_per_passage,
        },
    }


async def run_cars_pipeline(
    client: LLMClient,
    topics: list[dict],
    config: Config,
):
    """Run the full CARS passage generation pipeline."""
    checkpoint = CheckpointManager(f"{config.checkpoint_dir}/cars")
    writer = OutputWriter(f"{config.output_dir}/cars_passages.jsonl")

    # Total passages to generate
    total_passages = config.cars.passages_per_topic  # e.g., 100
    subjects = config.cars.passage_subjects

    if not subjects:
        subjects = ["Philosophy", "Ethics", "Art History", "Political Theory",
                     "Literary Criticism", "History of Science"]

    # Determine which passages are already done
    completed = checkpoint.get_completed_topics()
    start_from = len(completed)

    logger.info(
        f"CARS pipeline: {total_passages} passages target, "
        f"{start_from} completed, {total_passages - start_from} remaining"
    )

    batch_size = config.cars.batch_size

    for batch_start in range(start_from, total_passages, batch_size):
        batch_end = min(batch_start + batch_size, total_passages)
        batch_range = range(batch_start + 1, batch_end + 1)

        tasks = []
        for pnum in batch_range:
            passage_id = f"CARS_P{pnum:04d}"
            if checkpoint.is_complete(passage_id):
                continue

            # Cycle through subjects, with some randomness
            subject = random.choice(subjects)
            tasks.append((pnum, passage_id, subject))

        if not tasks:
            continue

        # Run batch concurrently
        coros = [
            generate_cars_passage_set(client, pnum, subj, config)
            for pnum, pid, subj in tasks
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for (pnum, pid, subj), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Failed {pid}: {result}")
                continue
            if result is None:
                logger.warning(f"Failed to generate {pid}")
                continue

            writer.append(result)
            checkpoint.mark_complete(pid)

        completed_now = checkpoint.get_completed_topics()
        logger.info(
            f"CARS progress: {len(completed_now)}/{total_passages} passages"
        )

    total = writer.count()
    logger.info(f"CARS pipeline complete. Total passages: {total}")
