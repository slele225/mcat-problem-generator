#!/usr/bin/env python3
"""Print the A/B/C/D correct-answer distribution for generated discrete questions.

Usage:
    python scripts/check_answer_distribution.py [path]

Defaults to <repo>/output/discrete_questions.jsonl. Useful for checking whether
answer-position balancing is working — expect roughly 25% for each letter.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "output" / "discrete_questions.jsonl"
LETTERS = ["A", "B", "C", "D"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PATH),
        help="Path to the discrete_questions.jsonl file",
    )
    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"No questions file found at {path}", file=sys.stderr)
        return 1

    counts: Counter = Counter()
    total = 0
    other = 0
    malformed = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            total += 1
            answer = str(record.get("correct_answer", "")).strip().upper()
            if answer in LETTERS:
                counts[answer] += 1
            else:
                other += 1

    suffix = f", {malformed} malformed line(s) skipped" if malformed else ""
    print(f"Answer distribution in {path}")
    print(f"  {total} question(s){suffix}\n")

    if total == 0:
        print("  (no questions to report)")
        return 0

    for letter in LETTERS:
        n = counts[letter]
        pct = n / total * 100
        bar = "#" * round(pct / 2)
        print(f"  {letter}: {n:6d}  {pct:5.1f}%  {bar}")

    if other:
        pct = other / total * 100
        print(f"  ?: {other:6d}  {pct:5.1f}%  (missing/invalid correct_answer)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
