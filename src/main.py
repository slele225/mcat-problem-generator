"""MCAT Question Generator — Main entry point.

Usage:
    python -m src.main                      # Run both pipelines
    python -m src.main --discrete-only      # Run discrete pipeline only
    python -m src.main --cars-only          # Run CARS pipeline only
    python -m src.main --science-passage-only  # Run science passage pipeline only
    python -m src.main --render-figures        # Render figures from science_passages.jsonl
    python -m src.main --config my.yaml     # Use custom config file
    python -m src.main --stats              # Show checkpoint statistics
    python -m src.main --reset              # Clear all checkpoints (careful!)
    python -m src.main --max-topics 5       # Process at most 5 topics (test run)
    python -m src.main --run-name myrun     # Write all output into runs/myrun/
    python -m src.main --seed 42            # Reproducible topic selection
    python -m src.main --run-name myrun --recount   # Rebuild a run's checkpoint
"""

import argparse
import asyncio
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

from .config import load_config
from .llm_client import LLMClient
from .pipelines.discrete import run_discrete_pipeline
from .pipelines.cars import run_cars_pipeline
from .pipelines.science_passage import run_science_passage_pipeline
from .figures import render_jsonl_figures, ensure_figure_deps
from .checkpoint import CheckpointManager


def setup_logging(verbose: bool = False, logfile=None):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    # Quiet down httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _run_dir(run_name: str) -> Path:
    """Directory holding all artifacts for a named run."""
    return Path("runs") / run_name


def _apply_run_name(config, run_name: str):
    """Point a config's output + checkpoint dirs at runs/<run_name>/ (no I/O)."""
    run_dir = _run_dir(run_name)
    config.output_dir = str(run_dir)
    config.checkpoint_dir = str(run_dir / "checkpoints")
    return config


def recount(config):
    """Rebuild the discrete checkpoint from the actual questions jsonl.

    Counts accepted questions per topic_id, marks topics at/over quota complete,
    records partial counts for under-quota topics, and clears bogus .done
    markers. Never modifies the questions file.
    """
    target = config.discrete.questions_per_topic
    output_path = Path(f"{config.output_dir}/discrete_questions.jsonl")
    if not output_path.exists():
        print(f"No questions file at {output_path}; nothing to recount.")
        return

    counts: Counter = Counter()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            topic_id = record.get("topic_id")
            if topic_id:
                counts[topic_id] += 1

    ckpt = CheckpointManager(f"{config.checkpoint_dir}/discrete")
    kept, reopened = ckpt.rebuild_from_counts(counts, target)
    print(f"Recount for {config.checkpoint_dir}/discrete (target={target}/topic):")
    print(f"  topics kept complete (>= target):      {kept}")
    print(f"  topics reopened for resume (< target):  {reopened}")
    print(f"  total accepted questions preserved:     {sum(counts.values())}")


def load_topics(path: str) -> list[dict]:
    """Load topics from JSON file."""
    topics_path = Path(path)
    if not topics_path.exists():
        raise FileNotFoundError(f"Topics file not found: {path}")
    with open(topics_path, encoding="utf-8") as f:
        topics = json.load(f)
    logging.info(f"Loaded {len(topics)} topics from {path}")
    return topics


def show_stats(config):
    """Display checkpoint and output statistics."""
    print("\n=== MCAT Question Generator Stats ===\n")

    # Discrete stats
    discrete_ckpt = CheckpointManager(f"{config.checkpoint_dir}/discrete")
    d_stats = discrete_ckpt.get_stats()
    discrete_output = Path(f"{config.output_dir}/discrete_questions.jsonl")
    d_questions = 0
    if discrete_output.exists():
        with open(discrete_output) as f:
            d_questions = sum(1 for line in f if line.strip())

    print("Discrete Questions:")
    print(f"  Completed topics: {d_stats['completed_topics']}")
    print(f"  In-progress topics: {d_stats['in_progress_topics']}")
    print(f"  Total questions generated: {d_questions}")
    if d_stats['in_progress_details']:
        print(f"  In-progress details: {d_stats['in_progress_details']}")

    print()

    # CARS stats
    cars_ckpt = CheckpointManager(f"{config.checkpoint_dir}/cars")
    c_stats = cars_ckpt.get_stats()
    cars_output = Path(f"{config.output_dir}/cars_passages.jsonl")
    c_passages = 0
    c_questions = 0
    if cars_output.exists():
        with open(cars_output) as f:
            for line in f:
                if line.strip():
                    c_passages += 1
                    data = json.loads(line)
                    c_questions += len(data.get("questions", []))

    print("CARS Passages:")
    print(f"  Completed passages: {c_stats['completed_topics']}")
    print(f"  Total passages written: {c_passages}")
    print(f"  Total CARS questions: {c_questions}")

    print()

    # Science passage stats
    sp_ckpt = CheckpointManager(f"{config.checkpoint_dir}/science_passage")
    sp_stats = sp_ckpt.get_stats()
    sp_output = Path(f"{config.output_dir}/science_passages.jsonl")
    sp_passages = 0
    sp_questions = 0
    sp_figures = 0
    sp_rendered = 0
    if sp_output.exists():
        with open(sp_output) as f:
            for line in f:
                if line.strip():
                    sp_passages += 1
                    data = json.loads(line)
                    questions = data.get("questions", [])
                    sp_questions += len(questions)
                    figs = list(data.get("figures", []) or [])
                    for q in questions:
                        figs.extend(q.get("figures", []) or [])
                    sp_figures += len(figs)
                    sp_rendered += sum(1 for fig in figs if fig.get("image_path"))

    print("Science Passages:")
    print(f"  Completed passages: {sp_stats['completed_topics']}")
    print(f"  Total passages written: {sp_passages}")
    print(f"  Total science passage questions: {sp_questions}")
    print(f"  Figures specified: {sp_figures} (rendered: {sp_rendered})")

    print()


async def async_main(args):
    config = load_config(args.config)

    # Redirect all artifacts into a per-run folder when --run-name is set.
    if args.run_name:
        _apply_run_name(config, args.run_name)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Run '{args.run_name}': artifacts under {config.output_dir}/")

    if args.recount:
        recount(config)
        return

    if args.stats:
        show_stats(config)
        return

    if args.render_figures:
        jsonl = f"{config.output_dir}/science_passages.jsonl"
        report = render_jsonl_figures(
            jsonl,
            config.output_dir,
            fmt=config.science_passage.figure_format,
            force=args.force_render,
        )
        print(
            f"Figures: {report['rendered']} rendered, {report['skipped']} skipped, "
            f"{report['failed']} failed (of {report['total']}). "
            f"Images in {config.output_dir}/figures/"
        )
        return

    if args.reset:
        confirm = input("This will clear ALL checkpoints. Are you sure? (yes/no): ")
        if confirm.lower() == "yes":
            CheckpointManager(f"{config.checkpoint_dir}/discrete").reset()
            CheckpointManager(f"{config.checkpoint_dir}/cars").reset()
            CheckpointManager(f"{config.checkpoint_dir}/science_passage").reset()
            print("All checkpoints cleared.")
        else:
            print("Aborted.")
        return

    topics = load_topics(config.topics_file)

    # --topic-ids: parse + validate early (no API) so missing ids fail fast.
    target_topic_ids = None
    if args.topic_ids:
        target_topic_ids = [s.strip() for s in args.topic_ids.split(",") if s.strip()]
        if not target_topic_ids:
            logging.error("--topic-ids was empty after parsing; provide >=1 topic_id.")
            sys.exit(1)
        known = {t.get("topic_id") for t in topics}
        missing = [tid for tid in target_topic_ids if tid not in known]
        if missing:
            logging.error(
                f"--topic-ids: topic_id(s) not found in {config.topics_file}: "
                f"{', '.join(missing)}"
            )
            sys.exit(1)
        if args.max_topics is not None:
            logging.info("--topic-ids overrides --max-topics (targeted single-passage run).")

    async with LLMClient(model=config.model) as client:
        healthy = await client.health_check()
        if not healthy:
            logging.error(
                "Anthropic API health check failed. "
                "Verify ANTHROPIC_API_KEY and that the model name is correct."
            )
            sys.exit(1)
        logging.info(f"Connected to Anthropic API (model={config.model})")

        # Fix the RNG before any pipeline work so topic selection (and the
        # per-question picks, on the happy path) are reproducible across runs.
        if args.seed is not None:
            random.seed(args.seed)
            logging.info(f"Random seed fixed at {args.seed}")

        # --topic-ids is a targeted science-passage test: run ONLY that single
        # passage and stop (do not process discrete/CARS or other topics).
        if target_topic_ids:
            if config.science_passage.enable_figures:
                ensure_figure_deps()
                logging.info("Figures ENABLED for science passages.")
            logging.info("=" * 60)
            logging.info(f"Targeted SCIENCE PASSAGE run for topic_ids: {target_topic_ids}")
            logging.info("=" * 60)
            await run_science_passage_pipeline(
                client, topics, config, topic_ids=target_topic_ids
            )
            logging.info("Targeted run complete!")
            show_stats(config)
            return

        # Run pipelines. With no *-only flag, all three run; each *-only flag
        # restricts the run to exactly that pipeline.
        run_discrete = not (args.cars_only or args.science_passage_only)
        run_cars = not (args.discrete_only or args.science_passage_only)
        run_science = not (args.discrete_only or args.cars_only)

        if run_discrete:
            logging.info("=" * 60)
            logging.info("Starting DISCRETE question pipeline")
            logging.info("=" * 60)
            await run_discrete_pipeline(
                client, topics, config, max_topics=args.max_topics, seed=args.seed
            )

        if run_cars:
            logging.info("=" * 60)
            logging.info("Starting CARS passage pipeline")
            logging.info("=" * 60)
            await run_cars_pipeline(
                client, topics, config, max_topics=args.max_topics
            )

        if run_science:
            # Fail fast if figures are enabled but the optional deps are missing.
            if config.science_passage.enable_figures:
                ensure_figure_deps()
                logging.info(
                    "Figures ENABLED for science passages (render later with "
                    "--render-figures)."
                )
            logging.info("=" * 60)
            logging.info("Starting SCIENCE PASSAGE pipeline")
            logging.info("=" * 60)
            await run_science_passage_pipeline(
                client, topics, config, max_topics=args.max_topics, seed=args.seed
            )

    logging.info("All pipelines complete!")
    show_stats(config)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MCAT practice questions using LLM"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--discrete-only", action="store_true",
        help="Run only the discrete question pipeline"
    )
    parser.add_argument(
        "--cars-only", action="store_true",
        help="Run only the CARS passage pipeline"
    )
    parser.add_argument(
        "--science-passage-only", action="store_true",
        help="Run only the passage-based science question pipeline"
    )
    parser.add_argument(
        "--topic-ids", default=None, metavar="ID1,ID2,...",
        help="Targeted test: generate ONE science passage from exactly these "
             "comma-separated topic_ids (in order), then stop. Bypasses clustering, "
             "shuffle, and checkpoints (always generates fresh); overrides "
             "--max-topics. Composes with --science-passage-only / --run-name / --config."
    )
    parser.add_argument(
        "--render-figures", action="store_true",
        help="Render figures from science_passages.jsonl into the output folder "
             "(no API calls), then exit. Honors --run-name and --force-render."
    )
    parser.add_argument(
        "--force-render", action="store_true",
        help="With --render-figures, re-render figures even if the image already exists."
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show checkpoint statistics and exit"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear all checkpoints and exit"
    )
    parser.add_argument(
        "--max-topics", type=int, default=None, metavar="N",
        help="Process at most N topics this run (caps passages for the CARS "
             "pipeline). Useful for quick test runs."
    )
    parser.add_argument(
        "--run-name", default=None, metavar="NAME",
        help="Write all artifacts for this run into runs/<NAME>/ (questions, "
             "checkpoints, metrics, run.log). Omit to use the shared output/ folder."
    )
    parser.add_argument(
        "--seed", type=int, default=None, metavar="N",
        help="Fix the RNG seed so topic selection is reproducible across runs."
    )
    parser.add_argument(
        "--recount", action="store_true",
        help="Rebuild the (run's) discrete checkpoint from its questions jsonl — "
             "reopen under-quota topics while keeping all questions — then exit."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()
    if args.max_topics is not None and args.max_topics < 1:
        parser.error("--max-topics must be a positive integer")
    logfile = _run_dir(args.run_name) / "run.log" if args.run_name else None
    setup_logging(args.verbose, logfile)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
