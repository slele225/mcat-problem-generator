"""Pipeline for generating passage-based MCAT science questions.

A unit of work is a PASSAGE: a ~200-350 word science passage (prose + optional
markdown table) generated for a small cluster of 2-3 related topics, followed by
4-7 validated, passage-linked questions. Mirrors the discrete pipeline's
structure — sequential question generation with within-passage diversity,
adversarial + blind-solve validation, MetricsTracker, CheckpointManager,
OutputWriter, run-folder support — but adds the passage layer.

Does NOT touch the discrete or CARS pipelines.
"""

import asyncio
import json
import logging
import random
from itertools import groupby
from pathlib import Path
from typing import Optional

from ..config import Config
from ..llm_client import LLMClient
from ..checkpoint import CheckpointManager, OutputWriter
from ..metrics import MetricsTracker, TopicMetrics
from ..prompts.science_passage import (
    passage_generation_prompt,
    passage_review_prompt,
    question_generation_prompt,
    adversarial_review_prompt,
    blind_solve_prompt,
    build_question_plan,
    pick_num_questions,
)
from ..schemas import (
    RawSciencePassage,
    RawScienceQuestion,
    ScienceAdversarialReview,
    BlindSolveResult,
    FigureSpec,
    explanation_has_phantom_option,
)
from ..figures import (
    validate_figures,
    figures_to_text,
    passage_figure_id,
    question_figure_id,
)
from .figure_pass import run_figure_pass

logger = logging.getLogger(__name__)

CARS_SECTION = "Critical Analysis and Reasoning Skills"


def _append_rejected_science(output_dir: str, record: dict) -> None:
    """Best-effort append of a DIAGNOSTIC reject record to rejected_science.jsonl.

    Captures a discarded question attempt (failed adversarial review, blind-solve
    mismatch, and/or answer_basis rejection) so the blind-solve filter can be
    analyzed later. This is logging ONLY — it never changes the accept/reject
    decision. Any failure (I/O, serialization) is swallowed so a logging problem
    can never alter or crash a run.
    """
    try:
        path = Path(output_dir) / "rejected_science.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# --- Structural figure enforcement (gated on enable_figures) -----------------
#
# Some content categories are ABOUT things that essentially require a figure on
# the real MCAT, yet prompting alone reliably fails to produce one (organic
# passages emit zero SMILES; enzyme-kinetics passages emit zero plots). For
# those we REQUIRE a figure type STRUCTURALLY: after a passage + its questions
# are generated and validated, if a required type is absent from BOTH the
# passage and every question, the whole passage set is regenerated (see
# generate_science_passage_set).
#
# The trigger is PURELY content-category based (deterministic) — never text
# sniffing. Keys are content-category CODES: the token before the colon in a
# topic's `content_category` (e.g. "5D: Structure, function, ..." -> "5D").
# Approved mapping:
#   5B, 5D -> "smiles"  covalent bonds / stereochemistry / isomers (5B) and
#                       biologically relevant molecules (5D) — passages center
#                       on specific structures, which must be drawn, not named.
#   5A, 5E -> "plot"    water & solutions incl. titration curves (5A), and
#                       thermodynamics & kinetics incl. enzyme kinetics and
#                       reaction-rate data (5E).
# 1A is deliberately NOT enforced: it mixes amino-acid structures, enzyme
# kinetics, and protein structure, so no single type fits, and 5E already
# catches enzyme-kinetics passages. Edit this dict to tune enforcement.
REQUIRED_FIGURES: dict[str, str] = {
    "5A": "plot",
    "5B": "smiles",
    "5D": "smiles",
    "5E": "plot",
}


def _category_code(content_category: str) -> str:
    """The content-category code: the token before the first colon, stripped.

    "5D: Structure, function, and reactivity ..." -> "5D". Returns "" if absent.
    """
    return (content_category or "").split(":", 1)[0].strip()


def required_figure_types(cluster: dict) -> set[str]:
    """Figure types a passage for this cluster MUST include (per REQUIRED_FIGURES).

    Purely content-category based: if ANY topic in the cluster belongs to a
    content category whose code is in REQUIRED_FIGURES, that type is required.
    Normal clusters are single-category; the --topic-ids path may span several,
    so every topic (and the cluster's own content_category) is scanned. Returns
    an empty set when nothing is required.
    """
    required: set[str] = set()
    for t in cluster.get("topics", []) or []:
        ft = REQUIRED_FIGURES.get(_category_code(t.get("content_category", "")))
        if ft:
            required.add(ft)
    ft = REQUIRED_FIGURES.get(_category_code(cluster.get("content_category", "")))
    if ft:
        required.add(ft)
    return required


def _present_figure_types(passage_set: dict) -> set[str]:
    """Figure types actually present on a generated passage set.

    A required type is satisfied by a figure on the PASSAGE or on ANY question.
    Reads the model_dump()'d figure dicts the set carries.
    """
    present: set[str] = set()
    for fig in passage_set.get("figures", []) or []:
        ft = fig.get("figure_type")
        if ft:
            present.add(ft)
    for q in passage_set.get("questions", []) or []:
        for fig in q.get("figures", []) or []:
            ft = fig.get("figure_type")
            if ft:
                present.add(ft)
    return present


_ENFORCE_SENTENCE = (
    "Your previous attempt omitted the REQUIRED {type} figure. You MUST include "
    "at least one {type} figure in the figures array. This is mandatory."
)


def _enforcement_note(missing: set[str], attempt: int) -> str:
    """Escalating directive injected into PASSAGE regeneration when a required
    figure type was missing on the prior attempt.

    `attempt` is the upcoming (0-based) attempt index; emphasis escalates on
    later attempts. Returns "" when nothing is missing (e.g. the first attempt).
    """
    if not missing:
        return ""
    note = " ".join(_ENFORCE_SENTENCE.format(type=ft) for ft in sorted(missing))
    if attempt >= 2:
        types = "/".join(sorted(missing))
        note = (
            f"CRITICAL - repeated omission (regeneration attempt {attempt}). "
            f"{note} Do NOT omit the {types} figure(s) again."
        )
    return note


def _append_user_instruction(messages: list[dict], instruction: str) -> list[dict]:
    """Return a copy of `messages` with `instruction` appended to the last user
    turn (a new user turn if there is none).

    Used to inject the figure-enforcement directive into a regeneration without
    touching the prompt builders. Returns `messages` unchanged when empty.
    """
    if not instruction:
        return messages
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            m["content"] = f"{m['content']}\n\n{instruction}"
            return out
    out.append({"role": "user", "content": instruction})
    return out


def _process_figures(raw_figures, enable_figures, id_fn):
    """Validate raw figure specs and assign stable figure ids.

    Returns (specs, ok, errors):
      * specs  — list[FigureSpec] with assigned figure_id (image_path still None)
      * ok     — False if any figure failed semantic validation (e.g. unparseable
                 SMILES); the caller should regenerate
      * errors — human-readable reasons (for logging)

    When figures are disabled or none were emitted, returns ([], True, []) — any
    stray spec the model emitted while figures are off is simply dropped.
    """
    if not enable_figures or not raw_figures:
        return [], True, []
    ok, errors = validate_figures(raw_figures)
    if not ok:
        return [], False, errors
    specs = [
        FigureSpec(figure_id=id_fn(i), **raw.model_dump())
        for i, raw in enumerate(raw_figures, start=1)
    ]
    return specs, True, []


def _chunk_2_3(items: list) -> list[list]:
    """Split a list into chunks of size 2-3, never leaving an orphan of 1.

    Used to size topic clusters: 2-3 related topics per passage. A list of
    length 1 is unavoidable as a single chunk; everything else is partitioned
    into 2s and 3s (e.g. 4 -> [2,2], 5 -> [3,2], 7 -> [3,2,2]).
    """
    n = len(items)
    if n <= 3:
        return [items]
    chunks = []
    i = 0
    while i < n:
        remaining = n - i
        if remaining == 4:
            chunks.append(items[i:i + 2])
            i += 2
        else:
            chunks.append(items[i:i + 3])
            i += 3
    return chunks


def build_clusters(topics: list[dict], config: Config) -> list[dict]:
    """Group non-CARS topics into coherent 2-3 topic clusters for passages.

    Topics are grouped by (section, content_category) and, within that, ordered
    so siblings sharing a topic_group are adjacent — so a singleton or very small
    topic_group is naturally joined with sibling topics from the same content
    category to give a passage enough material. Each group is then chunked into
    clusters of 2-3 topics (see `_chunk_2_3`).
    """
    science = [t for t in topics if t.get("section") != CARS_SECTION]

    # Stable grouping by (section, content_category); within each, keep same
    # topic_group together so chunks are topically coherent.
    def cc_key(t):
        return (t.get("section", ""), t.get("content_category", ""))

    science.sort(key=lambda t: (
        t.get("section", ""),
        t.get("content_category", ""),
        t.get("topic_group", ""),
        t.get("topic_id", ""),
    ))

    clusters: list[dict] = []
    for (section, content_category), group_iter in groupby(science, key=cc_key):
        group = list(group_iter)
        for chunk in _chunk_2_3(group):
            topic_groups = []
            for t in chunk:
                tg = t.get("topic_group", "")
                if tg and tg not in topic_groups:
                    topic_groups.append(tg)
            first_id = chunk[0].get("topic_id", "T")
            clusters.append({
                "cluster_key": first_id,
                "section": section,
                "content_category": content_category,
                "topic_group": " / ".join(topic_groups) if topic_groups else "",
                "topic_ids": [t.get("topic_id") for t in chunk],
                "topics": chunk,
            })
    return clusters


def build_single_cluster(topics: list[dict], topic_ids: list[str]) -> dict:
    """Build ONE explicit cluster from the given topic_ids, in the given order.

    Used by the --topic-ids targeted test path. Raises ValueError naming any
    topic_id not present in the topics file. Section/content_category/topic_group
    are taken from the listed topics (first topic's section/category drives the
    passage prompt), so it works even if the ids span groups.
    """
    by_id = {t.get("topic_id"): t for t in topics}
    missing = [tid for tid in topic_ids if tid not in by_id]
    if missing:
        raise ValueError(
            f"topic_id(s) not found in topics file: {', '.join(missing)}"
        )
    chosen = [by_id[tid] for tid in topic_ids]  # preserve the given order

    topic_groups = []
    for t in chosen:
        tg = t.get("topic_group", "")
        if tg and tg not in topic_groups:
            topic_groups.append(tg)

    return {
        "cluster_key": chosen[0].get("topic_id", "T"),
        "section": chosen[0].get("section", ""),
        "content_category": chosen[0].get("content_category", ""),
        "topic_group": " / ".join(topic_groups) if topic_groups else "",
        "topic_ids": [t.get("topic_id") for t in chosen],
        "topics": chosen,
    }


async def generate_passage(
    client: LLMClient,
    cluster: dict,
    config: Config,
    metrics: TopicMetrics,
    passage_id: str,
    extra_instruction: str = "",
) -> Optional[dict]:
    """Generate and validate a single science passage for a cluster.

    Returns {passage_text, table_markdown, word_count, figures, figures_text} or
    None on failure. `figures` is a list[FigureSpec] (validated, ids assigned;
    empty when figures are disabled); `figures_text` is their text serialization.

    `extra_instruction`, when set, is appended to every generation attempt's
    prompt — used by the figure-enforcement loop to demand a missing required
    figure on regeneration.
    """
    word_min, word_max = config.science_passage.passage_word_range
    enable_figures = config.science_passage.enable_figures

    def _result(passage, word_count, specs, figures_text):
        return {
            "passage_text": passage.passage_text,
            "table_markdown": passage.table_markdown,
            "word_count": word_count,
            "figures": specs,
            "figures_text": figures_text,
        }

    for attempt in range(config.science_passage.max_retries + 1):
        try:
            gen_msgs = passage_generation_prompt(
                cluster["section"],
                cluster["content_category"],
                cluster["topic_group"],
                cluster["topics"],
                word_min,
                word_max,
                enable_figures=enable_figures,
            )
            gen_msgs = _append_user_instruction(gen_msgs, extra_instruction)
            raw = await client.generate_json(
                gen_msgs,
                temperature=config.science_passage.temperature_generate,
                # Science passages are the longest single generation (passage +
                # table_markdown + LaTeX); 2048 truncated some mid-JSON (the
                # "Could not parse" fails). 4096 gives that headroom.
                max_tokens=4096,
                on_usage=lambda m, i, o: metrics.record_usage("passage_generation", m, i, o),
            )
            passage = RawSciencePassage(**raw)
        except Exception as e:
            logger.warning(f"Passage generation failed (attempt {attempt}): {e}")
            continue

        word_count = len(passage.passage_text.split())
        if word_count < word_min - 60 or word_count > word_max + 60:
            logger.debug(
                f"Passage word count {word_count} outside range "
                f"[{word_min}, {word_max}], retrying"
            )
            continue

        # Validate + id any passage-level figures (SMILES must parse, etc.).
        specs, fig_ok, fig_errs = _process_figures(
            passage.figures, enable_figures,
            lambda n: passage_figure_id(passage_id, n),
        )
        if not fig_ok:
            logger.info(f"figure REJECT {passage_id} passage: {fig_errs}")
            continue
        figures_text = figures_to_text(specs)

        try:
            review_msgs = passage_review_prompt(
                passage.passage_text,
                passage.table_markdown,
                cluster["topics"],
                word_min,
                word_max,
                figures_text=figures_text,
                enable_figures=enable_figures,
            )
            review_raw = await client.generate_json(
                review_msgs,
                temperature=config.science_passage.temperature_validate,
                on_usage=lambda m, i, o: metrics.record_usage("passage_review", m, i, o),
            )
            if review_raw.get("passed", False):
                return _result(passage, word_count, specs, figures_text)
            logger.debug(f"Passage review failed: {review_raw.get('issues', [])}")
        except Exception as e:
            logger.warning(f"Passage review failed: {e}")
            if word_min <= word_count <= word_max:
                return _result(passage, word_count, specs, figures_text)

    logger.warning(f"Failed to generate valid passage for cluster {cluster['cluster_key']}")
    return None


def _blind_solve_diagnostics(raw: dict) -> tuple[str, str]:
    """Extract (chosen_answer, confidence) from a raw blind-solve response for the
    reject diagnostic log ONLY — this never influences the accept/reject decision.

    `chosen` is upper-cased, or "" when the field is absent/blank. `confidence` is
    the model's value, or "unknown" when the response omitted it or left it blank.
    That "unknown" is deliberately distinct from the "" that validate_science_question
    leaves in place when the blind-solve response could not be parsed at all (a
    non-dict result): "model gave no confidence" vs. "we failed to capture it".
    """
    chosen = raw.get("chosen_answer")
    chosen = chosen.strip().upper() if isinstance(chosen, str) and chosen.strip() else ""
    conf = raw.get("confidence")
    conf = conf.strip() if isinstance(conf, str) and conf.strip() else "unknown"
    return chosen, conf


async def validate_science_question(
    client: LLMClient,
    passage_text: str,
    table_markdown: str,
    question_data: dict,
    temperature: float,
    checker_model: str,
    metrics: TopicMetrics,
    passage_figures_text: str = "",
    question_figures_text: str = "",
) -> dict:
    """Adversarial review + blind solve for a passage-linked question.

    Both calls receive the passage as context. Adversarial review runs on the
    main model; blind solve runs on `checker_model` (independent, cheaper). Any
    figures are passed as TEXT (`passage_figures_text` / `question_figures_text`),
    never as images. Returns pass/fail for each check plus the separate (lenient)
    answer_basis verdict.
    """
    review_msgs = adversarial_review_prompt(
        passage_text, table_markdown, question_data,
        passage_figures_text, question_figures_text,
    )
    solve_msgs = blind_solve_prompt(
        passage_text, table_markdown, question_data,
        passage_figures_text, question_figures_text,
    )

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
        "answer_basis_ok": True,
        "answer_basis_note": "",
        "adversarial_issues": [],
        # Threaded through for diagnostic reject logging only (does NOT affect the
        # accept/reject decision): the blind-solver's chosen answer + confidence.
        "blind_solve_chosen": "",
        "blind_solve_confidence": "",
    }

    if isinstance(results[0], dict):
        try:
            review = ScienceAdversarialReview(**results[0])
            validation["adversarial_pass"] = review.passed
            validation["adversarial_issues"] = review.issues
            validation["answer_basis_ok"] = review.answer_basis_ok
            validation["answer_basis_note"] = review.answer_basis_note
        except Exception as e:
            logger.warning(f"Failed to parse adversarial review: {e}. Raw: {results[0]}")
    else:
        logger.warning(f"Adversarial review failed: {results[0]}")

    if isinstance(results[1], dict):
        # Diagnostic capture for the reject log (NEVER affects the pass decision):
        # pull chosen + confidence straight from the raw response so BOTH are
        # recorded even when the strict parse below rejects the object (e.g. a
        # response missing the required "confidence" field). Missing/blank
        # confidence becomes "unknown" — distinct from the "" left in place when
        # the blind-solve response was unparseable (the non-dict branch below).
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


async def generate_and_validate_question(
    client: LLMClient,
    passage_text: str,
    table_markdown: str,
    section: str,
    plan: dict,
    config: Config,
    passage_id: str,
    question_number: int,
    metrics: TopicMetrics,
    previous_stems: list[str],
    passage_figures_text: str = "",
) -> Optional[dict]:
    """Generate one passage-linked question, retrying until it passes validation.

    Returns the validated question dict (or None). Tracks the funnel into
    `metrics`. Rejections that were specifically due to the (lenient)
    answer_basis check are logged at INFO via the `answer_basis REJECT` marker so
    that rejection rate is observable in the run log. `passage_figures_text` is
    the text serialization of passage-level figures (shown to writer + validators).
    """
    max_retries = config.science_passage.max_retries
    enable_figures = config.science_passage.enable_figures
    answer_basis_rejections = 0
    question_id = f"{passage_id}_q{question_number:02d}"

    for attempt in range(max_retries + 1):
        metrics.questions_attempted += 1

        try:
            gen_msgs = question_generation_prompt(
                passage_text, table_markdown, section, plan, previous_stems,
                passage_figures_text=passage_figures_text,
                enable_figures=enable_figures,
            )
            raw = await client.generate_json(
                gen_msgs,
                temperature=config.science_passage.temperature_generate,
                # Headroom so LaTeX-heavy questions aren't truncated into
                # unparseable JSON (matches the discrete generation cap).
                max_tokens=3072,
                on_usage=lambda m, i, o: metrics.record_usage("generation", m, i, o),
            )
            question = RawScienceQuestion(**raw)
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
        # attempt and let the retry regenerate (never edit the explanation).
        phantom = explanation_has_phantom_option(
            question_data.get("explanation", ""), set(question_data.get("choices", {}))
        )
        if phantom:
            logger.info(
                f"phantom-option REJECT {question_id} (attempt {attempt}): "
                f"explanation references {phantom!r}"
            )
            continue

        # Validate + id any question-level figures (SMILES must parse, etc.).
        q_specs, fig_ok, fig_errs = _process_figures(
            question.figures, enable_figures,
            lambda n: question_figure_id(question_id, n),
        )
        if not fig_ok:
            logger.info(f"figure REJECT {question_id}: {fig_errs}")
            continue
        question_figures_text = figures_to_text(q_specs)

        try:
            validation = await validate_science_question(
                client,
                passage_text,
                table_markdown,
                question_data,
                config.science_passage.temperature_validate,
                config.science_passage.science_checker_model,
                metrics,
                passage_figures_text=passage_figures_text,
                question_figures_text=question_figures_text,
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

        quality_pass = validation["adversarial_pass"] and validation["blind_solve_pass"]
        basis_ok = validation["answer_basis_ok"]

        if quality_pass and basis_ok:
            metrics.accepted += 1
            return {
                "question_id": question_id,
                "passage_id": passage_id,
                # Per-question content topic (one of the passage's cluster topics),
                # assigned in build_question_plan and steered in the generation
                # prompt. Carries a truthful topic_id so science questions feed the
                # weak-topic engine (discrete already had this; science did not).
                "topic_id": plan.get("topic_id"),
                "topic": plan.get("topic", ""),
                # Persist the pipeline-ASSIGNED skill (authoritative), not the
                # model's echoed skill_tested — the .get() fallback was dead code
                # (RawScienceQuestion always carries skill_tested, default ""), so
                # a blank/mislabeled echo was being saved (and surfaced downstream
                # as a null/empty `skill`).
                "skill_tested": plan["skill_label"],
                "answer_basis": question_data.get("answer_basis", plan["answer_basis"]),
                "stem": question_data["stem"],
                "choices": question_data["choices"],
                "correct_answer": question_data["correct_answer"],
                "explanation": question_data["explanation"],
                "difficulty": question_data.get("difficulty", plan["difficulty"]),
                "figures": [s.model_dump() for s in q_specs],
                "validation": {
                    "adversarial_pass": validation["adversarial_pass"],
                    "blind_solve_pass": validation["blind_solve_pass"],
                    "answer_basis_ok": validation["answer_basis_ok"],
                },
            }

        # DIAGNOSTIC (does not change accept logic): the attempt is discarded
        # because at least one check failed. Record it to rejected_science.jsonl
        # so the blind-solve filter can be analyzed later. Best-effort only.
        _append_rejected_science(config.output_dir, {
            "passage_id": passage_id,
            "question_id": question_id,
            "attempt": attempt,
            "skill": plan.get("skill_key"),
            "skill_label": plan.get("skill_label"),
            "difficulty": plan.get("difficulty"),
            "answer_basis": plan.get("answer_basis"),
            "stem": question_data.get("stem"),
            "choices": question_data.get("choices"),
            "correct_answer": question_data.get("correct_answer"),
            "checks": {
                "adversarial_pass": validation["adversarial_pass"],
                "blind_solve_pass": validation["blind_solve_pass"],
                "answer_basis_ok": validation["answer_basis_ok"],
            },
            "blind_solve_chosen": validation.get("blind_solve_chosen", ""),
            "blind_solve_confidence": validation.get("blind_solve_confidence", ""),
            "adversarial_issues": validation.get("adversarial_issues", []),
            "answer_basis_note": validation.get("answer_basis_note", ""),
        })

        # Failed. Surface WHY, and specifically flag answer_basis-only rejections
        # so the rejection rate on that (lenient) check is observable.
        if quality_pass and not basis_ok:
            answer_basis_rejections += 1
            logger.info(
                f"answer_basis REJECT {passage_id} q{question_number} "
                f"(labeled '{plan['answer_basis']}'): {validation['answer_basis_note']}"
            )
        else:
            logger.debug(
                f"{passage_id} q{question_number} failed (attempt {attempt}): "
                f"adversarial={'PASS' if validation['adversarial_pass'] else 'FAIL'}, "
                f"blind_solve={'PASS' if validation['blind_solve_pass'] else 'FAIL'}, "
                f"basis_ok={basis_ok}, issues={validation.get('adversarial_issues', [])}"
            )

    metrics.slots_failed += 1
    logger.warning(
        f"Exhausted retries for {passage_id} q{question_number} "
        f"({answer_basis_rejections} answer_basis rejection(s) among them)"
    )
    return None


async def _generate_passage_set_once(
    client: LLMClient,
    cluster: dict,
    passage_id: str,
    config: Config,
    metrics: TopicMetrics,
    passage_extra_instruction: str = "",
) -> Optional[dict]:
    """One full pass: generate a passage (carrying any enforcement directive) and
    its validated linked questions. Returns the passage-set dict or None.

    The figure-enforcement loop lives in generate_science_passage_set, which
    calls this repeatedly; `passage_extra_instruction` is the escalating demand
    for a missing required figure, applied to the passage generation prompt.
    """
    passage_data = await generate_passage(
        client, cluster, config, metrics, passage_id,
        extra_instruction=passage_extra_instruction,
    )
    if passage_data is None:
        return None

    passage_text = passage_data["passage_text"]
    table_markdown = passage_data["table_markdown"]
    passage_figures = passage_data["figures"]            # list[FigureSpec]
    passage_figures_text = passage_data["figures_text"]

    # NEW (Phase 1.1) — DEDICATED FIGURE PASS. On the FINAL passage prose + table,
    # decide whether results/entities belong in a plot or a chemical structure
    # (vs. a markdown table), emit those as FigureSpecs, and reconcile the table —
    # all BEFORE questions are written, so questions see the final exhibit set.
    # Runs only when figures are enabled; best-effort (never crashes the run). The
    # model is injected (config.model now; a later phase can A/B a different model
    # for this stage alone). Any structural enforcement directive is forwarded so
    # a required figure type is demanded here as well as in passage generation.
    if config.science_passage.enable_figures:
        fp = await run_figure_pass(
            client, cluster, passage_id, passage_text, table_markdown,
            passage_figures, config, metrics,
            model=config.model,
            extra_instruction=passage_extra_instruction,
        )
        if fp["changed"]:
            passage_figures = fp["figures"]
            table_markdown = fp["table_markdown"]
            passage_figures_text = fp["figures_text"]

    # An exhibit is a table OR a figure: guarantee a data-interpretation (Skill 4)
    # question whenever either is present.
    has_exhibit = bool(table_markdown) or bool(passage_figures)

    num_questions = pick_num_questions(
        config.science_passage.questions_per_passage_range
    )
    plan = build_question_plan(
        num_questions, config.science_passage.skill_weights, has_exhibit,
        topics=cluster["topics"],
    )

    # Generate sequentially so each new question sees the stems already accepted
    # for THIS passage and covers a different angle (within-passage diversity).
    accepted: list[dict] = []
    accepted_stems: list[str] = []
    for i, q_plan in enumerate(plan, start=1):
        result = await generate_and_validate_question(
            client,
            passage_text,
            table_markdown,
            cluster["section"],
            q_plan,
            config,
            passage_id,
            len(accepted) + 1,
            metrics,
            accepted_stems,
            passage_figures_text=passage_figures_text,
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
        "section": cluster["section"],
        "content_category": cluster["content_category"],
        "topic_ids": cluster["topic_ids"],
        "topic_group": cluster["topic_group"],
        "passage_text": passage_text,
        "table_markdown": table_markdown,
        "figures": [s.model_dump() for s in passage_figures],
        "word_count": passage_data["word_count"],
        "questions": accepted,
        "validation": {
            "passage_reviewed": True,
            "questions_validated": len(accepted),
            "target_questions": num_questions,
        },
    }


async def generate_science_passage_set(
    client: LLMClient,
    cluster: dict,
    passage_id: str,
    config: Config,
    metrics: TopicMetrics,
) -> Optional[dict]:
    """Generate a complete science passage with its validated linked questions.

    When figures are enabled AND the cluster's content category requires a figure
    type (REQUIRED_FIGURES), STRUCTURALLY ENFORCE it: if a required type is absent
    from both the passage and every question, regenerate the WHOLE passage set up
    to max_retries, injecting an escalating directive into passage generation.
    After max_retries still missing, log ERROR and accept anyway, so the miss
    rate can be measured. Every enforcement regeneration and the final outcome are
    logged at INFO. When no type is required (figures off, or a non-required
    category), this is a single pass identical to prior behavior.
    """
    enable_figures = config.science_passage.enable_figures
    required = required_figure_types(cluster) if enable_figures else set()

    # No enforcement -> single pass, behaviorally identical to before.
    if not required:
        return await _generate_passage_set_once(
            client, cluster, passage_id, config, metrics
        )

    max_retries = config.science_passage.max_retries
    best: Optional[dict] = None
    missing: set[str] = set()
    for attempt in range(max_retries + 1):
        note = _enforcement_note(missing, attempt) if (best is not None and missing) else ""
        candidate = await _generate_passage_set_once(
            client, cluster, passage_id, config, metrics,
            passage_extra_instruction=note,
        )
        if candidate is None:
            # Whole-set generation failed this attempt; treat as a failed attempt
            # and let the loop retry (keeping any earlier good-but-missing set).
            logger.warning(
                f"ENFORCEMENT: {passage_id} produced no passage set on attempt "
                f"{attempt + 1}/{max_retries + 1}"
            )
            continue
        best = candidate
        missing = required - _present_figure_types(candidate)
        if not missing:
            if attempt > 0:
                logger.info(
                    f"ENFORCEMENT: {passage_id} satisfied required figure type(s) "
                    f"{sorted(required)} after {attempt} regeneration(s)"
                )
            return candidate
        if attempt < max_retries:
            logger.info(
                f"ENFORCEMENT: {passage_id} missing required figure type(s) "
                f"{sorted(missing)} after attempt {attempt + 1}/{max_retries + 1}; "
                f"regenerating whole passage"
            )

    if best is None:
        return None
    # Exhausted retries with a usable set that still lacks the required type(s):
    # log ERROR per type and accept anyway (so failure rate is observable).
    for ft in sorted(missing):
        logger.error(
            f"ENFORCEMENT: {passage_id} still missing required {ft} after "
            f"{max_retries} retries"
        )
    logger.info(
        f"ENFORCEMENT: {passage_id} accepted WITHOUT required figure type(s) "
        f"{sorted(missing)} after {max_retries} retries (accepted for measurement)"
    )
    return best


async def _run_targeted_passage(
    client: LLMClient,
    topics: list[dict],
    config: Config,
    writer: OutputWriter,
    topic_ids: list[str],
):
    """Generate ONE passage from an explicit list of topic_ids (--topic-ids).

    Bypasses clustering/shuffle/max_topics and does not touch checkpoints (no
    completion-skip, no `.done` written). Output is still appended to the run's
    science_passages.jsonl so it can be rendered with --render-figures.
    """
    cluster = build_single_cluster(topics, topic_ids)  # raises ValueError on missing id
    passage_id = f"SP_{cluster['cluster_key']}_01"
    logger.info(
        f"--topic-ids: targeted single-passage run {passage_id} for "
        f"{cluster['topic_ids']} ({cluster['content_category']} — "
        f"{cluster['topic_group']}); clustering/shuffle/checkpoints bypassed"
    )

    tracker = MetricsTracker(
        model=config.model,
        checker_model=config.science_passage.science_checker_model,
        metrics_path=f"{config.output_dir}/science_generation_metrics.json",
    )
    bucket = tracker.start_topic(passage_id, cluster["topic_group"], 0)
    try:
        result = await generate_science_passage_set(
            client, cluster, passage_id, config, bucket
        )
        if result is not None:
            bucket.target = result["validation"]["target_questions"]
            writer.append(result)
            logger.info(f"Completed {passage_id}: {len(result['questions'])} questions")
        else:
            logger.warning(f"Failed to generate {passage_id}")
    finally:
        tracker.finish_topic()

    tracker.log_summary()
    tracker.write_json()


async def run_science_passage_pipeline(
    client: LLMClient,
    topics: list[dict],
    config: Config,
    max_topics: Optional[int] = None,
    seed: Optional[int] = None,
    topic_ids: Optional[list[str]] = None,
):
    """Run the full passage-based science question pipeline.

    The unit of work is a passage (one cluster x `passages_per_topic_cluster`).
    `max_topics`, if set, caps how many passages are generated this run (mirrors
    the CARS pipeline's use of the flag). `seed` makes cluster ordering
    reproducible.

    When `topic_ids` is given (the --topic-ids test path), normal clustering,
    shuffling, `max_topics`, and the checkpoint completion-skip are ALL bypassed:
    a single passage is built from exactly those ids (in order) and generated
    fresh — regardless of any prior `.done` markers — and the run stops. No
    `.done` marker is written, so a targeted test run never interacts with the
    checkpoints used by normal runs.
    """
    writer = OutputWriter(f"{config.output_dir}/science_passages.jsonl")

    if topic_ids:
        await _run_targeted_passage(client, topics, config, writer, topic_ids)
        return

    checkpoint = CheckpointManager(f"{config.checkpoint_dir}/science_passage")

    clusters = build_clusters(topics, config)

    # Expand each cluster into N passage work-items.
    per_cluster = config.science_passage.passages_per_topic_cluster
    work_items: list[tuple[str, dict]] = []
    for cluster in clusters:
        for p in range(1, per_cluster + 1):
            passage_id = f"SP_{cluster['cluster_key']}_{p:02d}"
            work_items.append((passage_id, cluster))

    completed = checkpoint.get_completed_topics()
    remaining = [(pid, c) for (pid, c) in work_items if pid not in completed]

    logger.info(
        f"Science passage pipeline: {len(clusters)} clusters, "
        f"{len(work_items)} passage(s) target, {len(completed)} completed, "
        f"{len(remaining)} remaining"
    )

    # Shuffle so an interrupted run still spreads across sections; seedable.
    if seed is not None:
        random.Random(seed).shuffle(remaining)
    else:
        random.shuffle(remaining)

    if max_topics is not None and len(remaining) > max_topics:
        logger.info(
            f"--max-topics: limiting to {max_topics} of {len(remaining)} "
            f"remaining passage(s) this run"
        )
        remaining = remaining[:max_topics]

    tracker = MetricsTracker(
        model=config.model,
        checker_model=config.science_passage.science_checker_model,
        metrics_path=f"{config.output_dir}/science_generation_metrics.json",
    )

    for i, (passage_id, cluster) in enumerate(remaining):
        logger.info(
            f"[{i+1}/{len(remaining)}] {passage_id}: "
            f"{cluster['content_category']} — {cluster['topic_group']}"
        )
        bucket = tracker.start_topic(passage_id, cluster["topic_group"], 0)
        try:
            result = await generate_science_passage_set(
                client, cluster, passage_id, config, bucket
            )
            if result is not None:
                bucket.target = result["validation"]["target_questions"]
                writer.append(result)
                checkpoint.mark_complete(passage_id)
                logger.info(
                    f"Completed {passage_id}: {len(result['questions'])} questions"
                )
            else:
                logger.warning(f"Failed to generate {passage_id}")
        except Exception as e:
            logger.error(f"Fatal error processing {passage_id}: {e}")
        finally:
            tracker.finish_topic()

    total = writer.count()
    logger.info(f"Science passage pipeline complete. Total passages: {total}")

    tracker.log_summary()
    tracker.write_json()
