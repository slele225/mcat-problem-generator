"""MCAT Question Generator — Main entry point.

Usage:
    python -m src.main                      # Run both pipelines
    python -m src.main --discrete-only      # Run discrete pipeline only
    python -m src.main --cars-only          # Run CARS pipeline only
    python -m src.main --config my.yaml     # Use custom config file
    python -m src.main --stats              # Show checkpoint statistics
    python -m src.main --reset              # Clear all checkpoints (careful!)
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .llm_client import LLMClient
from .pipelines.discrete import run_discrete_pipeline
from .pipelines.cars import run_cars_pipeline
from .checkpoint import CheckpointManager


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_topics(path: str) -> list[dict]:
    """Load topics from JSON file."""
    topics_path = Path(path)
    if not topics_path.exists():
        raise FileNotFoundError(f"Topics file not found: {path}")
    with open(topics_path) as f:
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


async def async_main(args):
    config = load_config(args.config)

    if args.stats:
        show_stats(config)
        return

    if args.reset:
        confirm = input("This will clear ALL checkpoints. Are you sure? (yes/no): ")
        if confirm.lower() == "yes":
            CheckpointManager(f"{config.checkpoint_dir}/discrete").reset()
            CheckpointManager(f"{config.checkpoint_dir}/cars").reset()
            print("All checkpoints cleared.")
        else:
            print("Aborted.")
        return

    topics = load_topics(config.topics_file)

    # Health check vLLM server
    async with LLMClient(config.vllm_base_url, config.model) as client:
        healthy = await client.health_check()
        if not healthy:
            logging.error(
                f"Cannot connect to vLLM server at {config.vllm_base_url}. "
                f"Start it with: ./scripts/start_vllm.sh"
            )
            sys.exit(1)
        logging.info(f"Connected to vLLM server at {config.vllm_base_url}")

        # Run pipelines
        if not args.cars_only:
            logging.info("=" * 60)
            logging.info("Starting DISCRETE question pipeline")
            logging.info("=" * 60)
            await run_discrete_pipeline(client, topics, config)

        if not args.discrete_only:
            logging.info("=" * 60)
            logging.info("Starting CARS passage pipeline")
            logging.info("=" * 60)
            await run_cars_pipeline(client, topics, config)

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
        "--stats", action="store_true",
        help="Show checkpoint statistics and exit"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear all checkpoints and exit"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
