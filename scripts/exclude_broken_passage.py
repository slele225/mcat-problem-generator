#!/usr/bin/env python3
"""Write a copy of a science_passages.jsonl with ONE passage removed.

Used to drop science passage SP_CP_5C_001_01 from the beta_bank_v1 load because
one of its two figures (SP_CP_5C_001_01_pf02) failed to render (64 of 65 PNGs
rendered). Dropping the whole passage removes the broken figure reference AND its
6 questions, so nothing ships pointing at a non-existent image.

This NEVER mutates the original file: it reads science_passages.jsonl and writes a
sibling science_passages.filtered.jsonl, copying every line byte-for-byte EXCEPT
the one whose passage_id matches. (Lines are copied verbatim, not re-serialized,
so key order / formatting of the kept passages is untouched.)

Safety: it asserts EXACTLY ONE passage was removed. If it finds zero (already
excluded / wrong id) or more than one (duplicate id) it exits non-zero and writes
nothing, so you can't silently load the wrong thing.

Usage:
    python scripts/exclude_broken_passage.py
    python scripts/exclude_broken_passage.py --run-dir runs/beta_bank_v1 --exclude SP_CP_5C_001_01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs/beta_bank_v1",
                    help="Run directory containing science_passages.jsonl "
                         "(default: runs/beta_bank_v1).")
    ap.add_argument("--exclude", default="SP_CP_5C_001_01",
                    help="passage_id to drop (default: SP_CP_5C_001_01).")
    args = ap.parse_args()

    run_dir = (REPO_ROOT / args.run_dir).resolve()
    src = run_dir / "science_passages.jsonl"
    dst = run_dir / "science_passages.filtered.jsonl"

    if not src.exists():
        print(f"ERROR: not found: {src}", file=sys.stderr)
        return 1

    kept: list[str] = []
    removed = 0
    removed_questions = 0
    total = 0

    with src.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            total += 1
            try:
                pid = json.loads(line)["passage_id"]
            except (ValueError, KeyError) as e:
                print(f"ERROR: unparseable line {total}: {e}", file=sys.stderr)
                return 1
            if pid == args.exclude:
                removed += 1
                # count the dropped questions for the summary only
                try:
                    removed_questions += len(json.loads(line).get("questions", []))
                except ValueError:
                    pass
                continue
            kept.append(line if line.endswith("\n") else line + "\n")

    if removed != 1:
        print(f"ERROR: expected to remove exactly 1 passage with id "
              f"'{args.exclude}', but removed {removed}. Nothing written.",
              file=sys.stderr)
        return 1

    dst.write_text("".join(kept), encoding="utf-8")

    print(f"Read    {total} passages from {src.name}")
    print(f"Dropped 1 passage  ({args.exclude}, {removed_questions} questions)")
    print(f"Wrote   {len(kept)} passages to {dst.name}")
    print()
    print(f"--> point the loader's SCIENCE_FILE at: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
