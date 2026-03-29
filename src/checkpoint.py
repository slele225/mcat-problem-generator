"""Checkpoint management for resumable generation runs."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Tracks completed work so runs can be resumed after interruption.

    Uses two mechanisms:
    1. Marker files: {checkpoint_dir}/{topic_id}.done — marks a topic as fully complete
    2. Progress file: {checkpoint_dir}/_progress.json — tracks partial progress within a topic
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._progress_file = self.checkpoint_dir / "_progress.json"
        self._progress = self._load_progress()

    def _load_progress(self) -> dict:
        """Load progress tracking file."""
        if self._progress_file.exists():
            try:
                with open(self._progress_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Corrupted progress file, starting fresh")
        return {}

    def _save_progress(self):
        """Save progress tracking file."""
        with open(self._progress_file, "w") as f:
            json.dump(self._progress, f, indent=2)

    def is_complete(self, topic_id: str) -> bool:
        """Check if a topic has been fully completed."""
        marker = self.checkpoint_dir / f"{topic_id}.done"
        return marker.exists()

    def mark_complete(self, topic_id: str):
        """Mark a topic as fully completed."""
        marker = self.checkpoint_dir / f"{topic_id}.done"
        marker.touch()
        # Clean up partial progress
        if topic_id in self._progress:
            del self._progress[topic_id]
            self._save_progress()
        logger.info(f"Checkpoint: {topic_id} marked complete")

    def get_progress(self, topic_id: str) -> int:
        """Get number of validated questions completed for a topic."""
        return self._progress.get(topic_id, 0)

    def update_progress(self, topic_id: str, completed_count: int):
        """Update partial progress for a topic."""
        self._progress[topic_id] = completed_count
        self._save_progress()

    def get_completed_topics(self) -> set[str]:
        """Get set of all completed topic IDs."""
        completed = set()
        for f in self.checkpoint_dir.glob("*.done"):
            completed.add(f.stem)
        return completed

    def get_stats(self) -> dict:
        """Get summary statistics."""
        completed = self.get_completed_topics()
        in_progress = {k: v for k, v in self._progress.items() if k not in completed}
        return {
            "completed_topics": len(completed),
            "in_progress_topics": len(in_progress),
            "in_progress_details": in_progress,
        }

    def reset(self):
        """Clear all checkpoints (use with caution)."""
        for f in self.checkpoint_dir.glob("*.done"):
            f.unlink()
        self._progress = {}
        self._save_progress()
        logger.info("All checkpoints cleared")


class OutputWriter:
    """Append-only JSONL writer for incremental output."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict):
        """Append a single record to the JSONL file."""
        with open(self.output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def append_batch(self, records: list[dict]):
        """Append multiple records at once."""
        with open(self.output_path, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        """Read all records from the JSONL file."""
        if not self.output_path.exists():
            return []
        records = []
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def count(self) -> int:
        """Count records without loading them all."""
        if not self.output_path.exists():
            return 0
        count = 0
        with open(self.output_path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
