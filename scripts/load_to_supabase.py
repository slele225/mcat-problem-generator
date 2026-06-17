#!/usr/bin/env python3
"""Load the generated MCAT question bank into Supabase (Postgres).

Reads the three JSONL banks + the science figures manifest and upserts them
into the `passages`, `questions`, and `figures` tables. Re-runnable: every write
is an INSERT ... ON CONFLICT (id) DO UPDATE, so loading Friday's full bank over
this sample produces no duplicates. Does NOT upload figure PNGs (image_path is
populated; image_url is left untouched) and never touches profiles/attempts.

Connection: set SUPABASE_DB_URL to your Postgres connection string, e.g.
    postgresql://postgres:<pwd>@<host>.pooler.supabase.com:5432/postgres
(Supabase dashboard -> Project Settings -> Database -> Connection string).
The service-role DB connection bypasses RLS, which is what we want for ETL.

Usage:
    python scripts/load_to_supabase.py [--dry-run]

Requires: psycopg (v3).  pip install "psycopg[binary]"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# psycopg is only needed for the actual DB write; --dry-run works without it.
try:
    import psycopg
    from psycopg.types.json import Json
except ModuleNotFoundError:  # pragma: no cover - dry-run / missing driver path
    psycopg = None

    def Json(v):  # type: ignore[no-redef]
        return v

# --- paths -------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DISCRETE_FILE   = REPO_ROOT / "runs" / "beta_bank_v1" / "discrete_questions.jsonl"
SCIENCE_FILE    = REPO_ROOT / "runs" / "beta_bank_v1" / "science_passages.filtered.jsonl"
CARS_FILE       = REPO_ROOT / "runs" / "beta_bank_v1" / "cars_passages.jsonl"
FIGURES_MANIFEST = REPO_ROOT / "runs" / "beta_bank_v1" / "figures_manifest.json"

CARS_SECTION = "Critical Analysis and Reasoning Skills"


# --- small helpers -----------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def require_choices(choices) -> dict:
    """The banks are already validated, but guard anyway so a malformed row is
    skipped-with-a-reason rather than crashing the whole load."""
    if not isinstance(choices, dict) or set(choices) < {"A", "B", "C", "D"}:
        raise ValueError(f"choices missing A/B/C/D: {choices!r}")
    return {k: choices[k] for k in ("A", "B", "C", "D")}


# --- row builders ------------------------------------------------------------
#
# Each builder appends to the shared lists and records any skipped row (with a
# reason) in `skips`. Builders never raise on a single bad record.

def build_discrete(rows, q_rows, skips):
    for q in rows:
        try:
            qid = q["question_id"]
            q_rows.append({
                "id": qid,
                "type": "discrete",
                "stem": q["stem"],
                "choices": require_choices(q["choices"]),
                "correct_answer": q["correct_answer"],
                "explanation": q.get("explanation"),
                "difficulty": q.get("difficulty"),
                "skill": q.get("skill_tested"),
                "section": q.get("section"),
                "content_category": q.get("content_category"),
                "topic_id": q.get("topic_id"),
                "passage_id": None,
                "answer_basis": None,
                "metadata": {
                    "validation": q.get("validation"),
                    "subtopics_tested": q.get("subtopics_tested"),
                    "topic": q.get("topic"),
                    "topic_group": q.get("topic_group"),
                },
            })
        except Exception as e:  # noqa: BLE001 - skip-with-reason, keep loading
            skips.append(("discrete question", q.get("question_id", "?"), str(e)))


def build_science(rows, p_rows, q_rows, f_rows, skips):
    for p in rows:
        try:
            pid = p["passage_id"]
            section = p.get("section")
            content_category = p.get("content_category")
            p_rows.append({
                "id": pid,
                "type": "science",
                "passage_text": p["passage_text"],
                "table_markdown": p.get("table_markdown"),
                "subject": None,
                "section": section,
                "content_category": content_category,
                "topic_ids": p.get("topic_ids"),
                "word_count": p.get("word_count"),
            })
        except Exception as e:  # noqa: BLE001
            skips.append(("science passage", p.get("passage_id", "?"), str(e)))
            continue

        for fig in p.get("figures") or []:
            _add_figure(fig, "passage", pid, f_rows, skips)

        for q in p.get("questions") or []:
            try:
                qid = q["question_id"]
                q_rows.append({
                    "id": qid,
                    "type": "science",
                    "stem": q["stem"],
                    "choices": require_choices(q["choices"]),
                    "correct_answer": q["correct_answer"],
                    "explanation": q.get("explanation"),
                    "difficulty": q.get("difficulty"),
                    "skill": q.get("skill_tested"),
                    # section/content_category live on the passage; topic_id is now
                    # per-question (assigned at generation) so science questions feed
                    # the weak-topic engine — discrete already populated topic_id.
                    "section": section,
                    "content_category": content_category,
                    "topic_id": q.get("topic_id"),
                    "passage_id": pid,
                    "answer_basis": q.get("answer_basis"),
                    # metadata.topic mirrors discrete so fetchTopicNames() can resolve
                    # the human topic name for science weak-topic rows too.
                    "metadata": {"validation": q.get("validation"), "topic": q.get("topic")},
                })
            except Exception as e:  # noqa: BLE001
                skips.append(("science question", q.get("question_id", "?"), str(e)))
                continue
            for fig in q.get("figures") or []:
                _add_figure(fig, "question", qid, f_rows, skips)


def build_cars(rows, p_rows, q_rows, skips):
    for p in rows:
        try:
            pid = p["passage_id"]
            subject = p.get("subject")
            p_rows.append({
                "id": pid,
                "type": "cars",
                "passage_text": p["passage_text"],
                "table_markdown": None,
                "subject": subject,
                "section": None,
                "content_category": None,
                "topic_ids": None,
                "word_count": p.get("word_count"),
            })
        except Exception as e:  # noqa: BLE001
            skips.append(("cars passage", p.get("passage_id", "?"), str(e)))
            continue

        for q in p.get("questions") or []:
            try:
                qid = q["question_id"]
                q_rows.append({
                    "id": qid,
                    "type": "cars",
                    "stem": q["stem"],
                    "choices": require_choices(q["choices"]),
                    "correct_answer": q["correct_answer"],
                    "explanation": q.get("explanation"),
                    # CARS now carries a per-question difficulty (Phase 2); fall
                    # back to "medium" only for older records that predate it.
                    "difficulty": q.get("difficulty", "medium"),
                    "skill": q.get("skill_type"),
                    "section": CARS_SECTION,
                    "content_category": subject,
                    "topic_id": None,
                    "passage_id": pid,
                    "answer_basis": None,
                    "metadata": {"validation": q.get("validation")},
                })
            except Exception as e:  # noqa: BLE001
                skips.append(("cars question", q.get("question_id", "?"), str(e)))


def _add_figure(fig, parent_type, parent_id, f_rows, skips):
    try:
        f_rows.append({
            "figure_id": fig["figure_id"],
            "parent_type": parent_type,
            "parent_id": parent_id,
            "figure_type": fig["figure_type"],
            "spec": fig,  # full embedded spec (caption/alt_text/plot|smiles/image_path)
            "image_path": fig.get("image_path"),
            # image_url intentionally left NULL until Storage upload
        })
    except Exception as e:  # noqa: BLE001
        skips.append(("figure", fig.get("figure_id", "?"), str(e)))


# --- upsert ------------------------------------------------------------------

PASSAGE_COLS = ["id", "type", "passage_text", "table_markdown", "subject",
                "section", "content_category", "topic_ids", "word_count"]
QUESTION_COLS = ["id", "type", "stem", "choices", "correct_answer", "explanation",
                 "difficulty", "skill", "section", "content_category", "topic_id",
                 "passage_id", "answer_basis", "metadata"]
FIGURE_COLS = ["figure_id", "parent_type", "parent_id", "figure_type", "spec",
               "image_path"]
JSONB_COLS = {"choices", "metadata", "spec"}


def _value(col, row):
    v = row[col]
    return Json(v) if col in JSONB_COLS and v is not None else v


def upsert(cur, table, pk, cols, rows):
    """Bulk INSERT ... ON CONFLICT DO UPDATE. Returns (inserted, updated).

    inserted/updated are derived from the set of PKs that existed before the
    write, so the summary is exact without per-row RETURNING.
    """
    if not rows:
        return 0, 0
    cur.execute(f"select {pk} from {table}")  # noqa: S608 - table is a literal
    existing = {r[0] for r in cur.fetchall()}

    update_cols = [c for c in cols if c != pk]
    set_clause = ", ".join(f"{c} = excluded.{c}" for c in update_cols)
    placeholders = "(" + ", ".join(["%s"] * len(cols)) + ")"
    sql = (
        f"insert into {table} ({', '.join(cols)}) values {placeholders} "  # noqa: S608
        f"on conflict ({pk}) do update set {set_clause}"
    )
    params = [[_value(c, r) for c in cols] for r in rows]
    cur.executemany(sql, params)

    ids = {r[pk] for r in rows}
    updated = len(ids & existing)
    return len(ids) - updated, updated


# --- summary -----------------------------------------------------------------

def print_summary(q_rows, p_rows, f_rows, counts, skips, manifest_warnings, dry):
    def by_type(rows, key="type"):
        out = defaultdict(int)
        for r in rows:
            out[r[key]] += 1
        return out

    head = "DRY RUN — nothing written" if dry else "Load complete"
    print(f"\n=== {head} ===")

    print("\nPassages:")
    for t, n in sorted(by_type(p_rows).items()):
        ins, upd = counts["passages"].get(t, (0, 0))
        suffix = "" if dry else f"  (inserted {ins}, updated {upd})"
        print(f"  {t:9s}: {n}{suffix}")
    print(f"  {'TOTAL':9s}: {len(p_rows)}")

    print("\nQuestions:")
    for t, n in sorted(by_type(q_rows).items()):
        ins, upd = counts["questions"].get(t, (0, 0))
        suffix = "" if dry else f"  (inserted {ins}, updated {upd})"
        print(f"  {t:9s}: {n}{suffix}")
    print(f"  {'TOTAL':9s}: {len(q_rows)}")

    print("\nFigures:")
    for t, n in sorted(by_type(f_rows, "parent_type").items()):
        print(f"  {t:9s}: {n}")
    if not dry:
        ins = sum(v[0] for v in counts["figures"].values())
        upd = sum(v[1] for v in counts["figures"].values())
        print(f"  (inserted {ins}, updated {upd})")
    print(f"  {'TOTAL':9s}: {len(f_rows)}")

    if manifest_warnings:
        print(f"\nManifest warnings ({len(manifest_warnings)}):")
        for w in manifest_warnings:
            print(f"  ! {w}")

    if skips:
        print(f"\nSKIPPED / ERRORED rows ({len(skips)}):")
        for kind, ident, reason in skips:
            print(f"  ! {kind} {ident}: {reason}")
    else:
        print("\nNo rows skipped or errored.")
    print()


# --- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and report counts without writing to the DB.")
    args = ap.parse_args()

    for f in (DISCRETE_FILE, SCIENCE_FILE, CARS_FILE):
        if not f.exists():
            print(f"ERROR: missing input file: {f}", file=sys.stderr)
            return 1

    skips: list[tuple[str, str, str]] = []
    p_rows: list[dict] = []
    q_rows: list[dict] = []
    f_rows: list[dict] = []

    build_discrete(read_jsonl(DISCRETE_FILE), q_rows, skips)
    build_science(read_jsonl(SCIENCE_FILE), p_rows, q_rows, f_rows, skips)
    build_cars(read_jsonl(CARS_FILE), p_rows, q_rows, skips)

    # Cross-check embedded figures against the render manifest (advisory only).
    manifest_warnings: list[str] = []
    if FIGURES_MANIFEST.exists():
        manifest = json.loads(FIGURES_MANIFEST.read_text(encoding="utf-8"))
        status = {m["figure_id"]: m.get("status") for m in manifest.get("figures", [])}
        embedded = {f["figure_id"] for f in f_rows}
        for fid, st in status.items():
            if st != "rendered":
                manifest_warnings.append(f"{fid}: manifest status={st!r} (not 'rendered')")
            if fid not in embedded:
                manifest_warnings.append(f"{fid}: in manifest but no embedded spec found")
        for fid in embedded - set(status):
            manifest_warnings.append(f"{fid}: embedded but absent from manifest")

    counts = {"passages": {}, "questions": {}, "figures": {}}

    if args.dry_run:
        print_summary(q_rows, p_rows, f_rows, counts, skips, manifest_warnings, dry=True)
        return 0

    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        print("ERROR: set SUPABASE_DB_URL to your Postgres connection string.",
              file=sys.stderr)
        return 1
    if psycopg is None:
        print('ERROR: psycopg is not installed. Run: pip install "psycopg[binary]"',
              file=sys.stderr)
        return 1

    # Single transaction: all-or-nothing so a re-run never half-loads.
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # passages before questions (FK), questions before figures (parent_id).
            for t in ("science", "cars"):
                rows = [r for r in p_rows if r["type"] == t]
                counts["passages"][t] = upsert(cur, "passages", "id", PASSAGE_COLS, rows)
            for t in ("discrete", "science", "cars"):
                rows = [r for r in q_rows if r["type"] == t]
                counts["questions"][t] = upsert(cur, "questions", "id", QUESTION_COLS, rows)
            for t in ("passage", "question"):
                rows = [r for r in f_rows if r["parent_type"] == t]
                counts["figures"][t] = upsert(cur, "figures", "figure_id", FIGURE_COLS, rows)
        conn.commit()

    print_summary(q_rows, p_rows, f_rows, counts, skips, manifest_warnings, dry=False)
    return 1 if skips else 0


if __name__ == "__main__":
    raise SystemExit(main())
