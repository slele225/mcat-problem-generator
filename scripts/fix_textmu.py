#!/usr/bin/env python3
r"""Scan + fix the KaTeX-incompatible ``\textmu`` / ``\textmicro`` micro-symbol
macro in the beta_bank_v1 JSONL banks.

KaTeX (the app's math renderer) does NOT support ``\textmu`` / ``\textmicro``, so
a stored value like ``$0.62\ \text{\textmu m/s}$`` renders broken. The
KaTeX-compatible form puts the micro symbol in MATH mode as ``\mu`` and the unit
letters in ``\text{}``::

    \text{\textmu m/s}   ->   \mu\text{m/s}      (wrapped, our actual bank data)
    \textmu m            ->   \mu\text{m}        (bare, micrometers)
    \textmu              ->   \mu                (alone, no trailing unit)

SCOPED to the micro-symbol issue only — it touches nothing else.

Every string field of every record is walked recursively (stem, choices,
explanation, passage_text, table_markdown, nested questions, ...), so no field is
missed regardless of schema shape.

Modes:
  (default)        report-only: count occurrences, show before/after snippets,
                   write NOTHING. Originals are never touched.
  --write-fixed    additionally write ``<name>.fixed.jsonl`` next to each original
                   (for inspection / re-load). Originals still untouched.
  --apply          replace originals IN PLACE, backing each up to
                   ``<name>.textmu.bak`` first. Use only after approving the report.

Usage:
    python scripts/fix_textmu.py                # report only
    python scripts/fix_textmu.py --write-fixed  # also emit *.fixed.jsonl
    python scripts/fix_textmu.py --apply         # overwrite originals (+ .bak)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BANK_DIR = REPO_ROOT / "runs" / "beta_bank_v1"
FILES = [
    BANK_DIR / "discrete_questions.jsonl",
    BANK_DIR / "science_passages.jsonl",
    BANK_DIR / "science_passages.filtered.jsonl",
    BANK_DIR / "cars_passages.jsonl",
]

# --- replacement -------------------------------------------------------------
#
# Patterns are applied in priority order on each DECODED string. The micro macro
# is matched as \textmu OR \textmicro. `\b` after the command name keeps it from
# matching a longer command. Units are letters and `/` (e.g. m, M, g, L, s, m/s,
# g/mL); a single optional space separates the macro from its unit in LaTeX.

_MU = r"\\text(?:mu|micro)"

# 1) Wrapped form: \text{ \textmu <units> }  ->  \mu\text{<units>}
#    Requires \textmu to be the first token inside \text{} (after optional ws);
#    pulls \mu into math mode and keeps the units upright in \text{}.
RE_WRAPPED = re.compile(r"\\text\{\s*" + _MU + r"\b\s*([^{}]*?)\s*\}")
# 2) Wrapped + empty: \text{\textmu}  ->  \mu
RE_WRAPPED_ALONE = re.compile(r"\\text\{\s*" + _MU + r"\s*\}")
# 3) Bare in math mode: \textmu <units>  ->  \mu\text{<units>}
RE_BARE = re.compile(_MU + r"\b\s*([A-Za-z][A-Za-z/]*)")
# 4) Bare + alone: \textmu  ->  \mu
RE_BARE_ALONE = re.compile(_MU + r"\b")


def fix_text(s: str) -> str:
    r"""Return s with every \textmu/\textmicro rewritten to KaTeX-valid \mu form."""
    s = RE_WRAPPED.sub(lambda m: r"\mu\text{" + m.group(1) + "}", s)
    s = RE_WRAPPED_ALONE.sub(r"\\mu", s)
    s = RE_BARE.sub(lambda m: r"\mu\text{" + m.group(1) + "}", s)
    s = RE_BARE_ALONE.sub(r"\\mu", s)
    return s


# --- recursive walk ----------------------------------------------------------

def walk(obj, path, changes):
    """Recursively transform every string in obj; append (path, before, after)
    tuples for each field that changed."""
    if isinstance(obj, str):
        if "textmu" in obj or "textmicro" in obj:
            new = fix_text(obj)
            if new != obj:
                changes.append((path, obj, new))
            return new
        return obj
    if isinstance(obj, list):
        return [walk(v, f"{path}[{i}]", changes) for i, v in enumerate(obj)]
    if isinstance(obj, dict):
        return {k: walk(v, f"{path}.{k}", changes) for k, v in obj.items()}
    return obj


def _snip(s: str, needle: str = "textmu", pad: int = 28) -> str:
    i = s.find(needle)
    if i < 0:
        return s[:80]
    lo, hi = max(0, i - pad), i + len(needle) + pad
    return ("..." if lo else "") + s[lo:hi] + ("..." if hi < len(s) else "")


def _id_for(path: str, record: dict) -> str:
    """Best-effort human id for a record's change path."""
    return (
        record.get("question_id")
        or record.get("passage_id")
        or path.split(".")[0]
        or "?"
    )


# --- per-file processing -----------------------------------------------------

def process_file(path: Path):
    """Return (records_out, occ_count, field_changes, examples, doubling_hits)."""
    occ = 0
    field_changes: list[tuple[str, str, str]] = []
    doubling: list[tuple[str, str]] = []
    records_out: list[dict] = []

    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            records_out.append(None)  # preserve blank line positions if any
            continue
        rec = json.loads(line)
        occ += sum(
            len(re.findall(_MU + r"\b", v))
            for v in _all_strings(rec)
        )
        changes: list[tuple[str, str, str]] = []
        new_rec = walk(rec, f"L{line_no}", changes)
        # doubling check: does any changed field contain an immediately repeated
        # \textmu-bearing chunk (e.g. "...m/s$ ...m/s$" back-to-back)?
        for fpath, before, _after in changes:
            if _looks_doubled(before):
                doubling.append((fpath, before))
        field_changes.extend(changes)
        records_out.append(new_rec)

    return records_out, occ, field_changes


def _all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_strings(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _all_strings(v)


def _looks_doubled(s: str) -> bool:
    """Heuristic for the corruption the operator flagged: the same numeric+\textmu
    chunk appearing twice back-to-back, e.g. '0.62 \textmum/s0.62 \textmum/s'."""
    # find chunks like  <number>\ \text{\textmu ...}$  and see if any repeats adjacently
    chunks = re.findall(r"[0-9.]+\\?\s*\\text\{" + _MU + r"[^}]*\}\$?", s)
    for i in range(len(chunks) - 1):
        if chunks[i] == chunks[i + 1]:
            return True
    # generic: an identical >=10-char substring containing \textmu repeated adjacently
    for m in re.finditer(_MU + r"[^\\$]{0,12}", s):
        frag = m.group(0)
        idx = s.find(frag + frag)
        if idx >= 0:
            return True
    return False


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            if rec is None:
                fh.write("\n")
            else:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Scan/fix \\textmu in beta_bank_v1.")
    ap.add_argument("--write-fixed", action="store_true",
                    help="also write <name>.fixed.jsonl next to each original")
    ap.add_argument("--apply", action="store_true",
                    help="overwrite originals in place (backs up to <name>.textmu.bak)")
    args = ap.parse_args()

    total_occ = 0
    total_fields = 0
    total_doubling = 0
    all_examples: list[tuple[str, str, str, str]] = []  # file, path, before, after

    for path in FILES:
        if not path.exists():
            print(f"  (skip, missing) {path.name}")
            continue
        records, occ, changes = process_file(path)
        total_occ += occ
        total_fields += len(changes)
        doubling = [(p, b) for (p, b, _a) in changes if _looks_doubled(b)]
        total_doubling += len(doubling)

        print(f"\n### {path.name}")
        print(f"    literal \\textmu/\\textmicro occurrences : {occ}")
        print(f"    fields changed                          : {len(changes)}")
        if doubling:
            print(f"    !! POSSIBLE DOUBLING in {len(doubling)} field(s) -- see below")
            for fpath, before in doubling[:3]:
                print(f"       {fpath}: {_snip(before, pad=40)!r}")

        for fpath, before, after in changes:
            all_examples.append((path.name, fpath, before, after))

        if args.write_fixed and changes:
            out = path.with_suffix(path.suffix + "")  # placeholder, replaced below
            out = path.parent / (path.name.replace(".jsonl", ".fixed.jsonl"))
            write_jsonl(out, records)
            print(f"    wrote -> {out.name}")
        if args.apply and changes:
            bak = path.parent / (path.name + ".textmu.bak")
            if not bak.exists():
                bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            write_jsonl(path, records)
            print(f"    APPLIED in place (backup: {bak.name})")

    # ---- examples ----
    print("\n" + "=" * 70)
    print(f"TOTAL: {total_occ} occurrence(s) across {total_fields} field(s); "
          f"doubling suspected in {total_doubling} field(s).")
    print("=" * 70)
    print("\nBEFORE / AFTER snippets (decoded strings; on disk each '\\' is stored")
    print("as '\\\\' by JSON escaping):\n")
    for i, (fname, fpath, before, after) in enumerate(all_examples[:6], 1):
        print(f"[{i}] {fname}  {fpath}")
        print(f"    before: {_snip(before, 'textmu')!r}")
        print(f"    after : {_snip(after, '\\mu')!r}")
        print()

    mode = ("APPLIED to originals" if args.apply else
            "wrote *.fixed.jsonl" if args.write_fixed else
            "REPORT ONLY -- no files written")
    print(f"Mode: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
