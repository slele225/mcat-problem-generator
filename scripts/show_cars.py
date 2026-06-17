"""Show the first CARS passage record from a run's cars_passages.jsonl.

The CARS pipeline writes its output to `<output_dir>/cars_passages.jsonl`
(see src/pipelines/cars.py: OutputWriter(f"{config.output_dir}/cars_passages.jsonl")).
A CARS record has no separate "discipline" field — `subject` IS the discipline
(e.g. "Religion", "Philosophy", "Economics").

This prints the first record: subject/discipline, the full passage text, the word
count, then each question's stem, choices, and correct answer.

Usage (run from the repo root):
    python scripts/show_cars.py                            # runs/cars_test/cars_passages.jsonl
    python scripts/show_cars.py runs/cars_test             # a run dir (appends cars_passages.jsonl)
    python scripts/show_cars.py path/to/cars_passages.jsonl
"""

import json
import sys
from pathlib import Path

FILENAME = "cars_passages.jsonl"
DEFAULT = Path("runs/cars_test") / FILENAME

# CARS passages contain curly quotes / em dashes; force UTF-8 so printing does
# not blow up on a cp1252 (Windows) console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def resolve_path(arg):
    """A path arg may be the jsonl itself or a run dir (we append the filename)."""
    if not arg:
        return DEFAULT
    p = Path(arg)
    return p / FILENAME if p.is_dir() else p


def first_record(path):
    """Return the first non-empty JSON record from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No records found in {path}")


def main():
    path = resolve_path(sys.argv[1] if len(sys.argv) > 1 else None)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        print(
            "Run the CARS pipeline first, or pass a path to a cars_passages.jsonl.",
            file=sys.stderr,
        )
        return 1

    rec = first_record(path)

    passage_text = rec.get("passage_text", "")
    subject = rec.get("subject", "(unknown)")
    word_count = rec.get("word_count", len(passage_text.split()))
    questions = rec.get("questions", []) or []

    print(f"FILE:                {path}")
    print(f"PASSAGE:             {rec.get('passage_id', '(no id)')}")
    print(f"SUBJECT/DISCIPLINE:  {subject}")
    print(f"WORD COUNT:          {word_count}")
    print()
    print("PASSAGE TEXT")
    print("-" * 70)
    print(passage_text)
    print("-" * 70)
    print()
    print(f"=== {len(questions)} QUESTIONS ===")

    for i, q in enumerate(questions, start=1):
        stem = q.get("stem", "")
        choices = q.get("choices", {}) or {}
        correct = (q.get("correct_answer") or "").strip().upper()
        skill = q.get("skill_type", "")

        print()
        header = f"Q{i}"
        if skill:
            header += f"  [{skill}]"
        print(header)
        print(f"  {stem}")
        # Keep A/B/C/D order when present, else fall back to sorted keys.
        keys = [k for k in ("A", "B", "C", "D") if k in choices] or sorted(choices)
        for k in keys:
            marker = "   <-- correct" if k == correct else ""
            print(f"    {k}) {choices[k]}{marker}")
        print(f"  Correct answer: {correct or '(none)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
