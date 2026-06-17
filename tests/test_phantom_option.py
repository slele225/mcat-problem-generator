"""Offline unit + real-data tests for explanation_has_phantom_option().

No API calls. Run from the repo root:  python tests/test_phantom_option.py
Covers: descriptive-phantom POSITIVES (incl. real fix5_test offenders), NEGATIVES
that must NOT flag, the letter-anchored regression, the A-D gating, and a scan over
the actual fix5_test shipped output to confirm the 7 known leaks are now caught with
no large false-positive blast radius.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.schemas import explanation_has_phantom_option as phantom  # noqa: E402

ABCD = frozenset("ABCD")

# --- POSITIVE: descriptive references that MUST flag (real fix5_test strings) ---
POSITIVE_DESCRIPTIVE = [
    # CARS_P0007_q02
    "C and the duplicate option both wrongly claim her objection was purely ethical "
    "and unrelated to empirical claims; in fact her objection concerns whether the "
    "worldview can be tested, revised, and proven mistaken.",
    # CARS_P0002_q01
    "The text gives no basis for the claim in the remaining option that Sargent "
    "anticipated the post-2000 replication failures; that empirical point is raised "
    "by later 'recent work,' not attributed to him.",
    # CARS_P0001_q06
    "C and the remaining option invoke Vidal, but Vidal's pays emerges from the long "
    "accommodation between a community and its physical setting.",
    # BB_2C_001_q006
    "The remaining option overstates mitosis as the majority of the cycle, "
    "contradicting the data, since interphase clearly dominates.",
    # BB_1A_014_q007
    "The same pH (referenced in the remaining option) controls one variable but does "
    "not by itself guarantee the enzyme adopts its active conformation.",
    # CP_4D_014_q005
    "The misconception in the final option is also false: speed in vacuum is "
    "independent of wavelength.",
    # SP_BB_1A_018_01_q04 (the "cannot be determined" stripped-E leak)
    "The IC50 can be determined from the mAb-7 series alone, so the claim in the "
    "final option is incorrect.",
    # extra synthetic variants
    "Option five is a trap that does not exist here.",
    "The last option restates the stem and is therefore wrong.",
]

# --- NEGATIVE: legitimate A-D discussion that must NOT flag ---
NEGATIVE = [
    "The other three options are wrong because they misread the passage.",
    "The remaining three choices each overstate the author's claim.",
    "The correct option is D, which the passage supports directly.",
    "Options B and C both describe partial readings of the argument.",
    "The final answer is D, supported by the third paragraph.",          # idiom guard
    "The other distractors are incorrect because they reverse the cause.",  # plural
    "A is correct because the passage states X. B is wrong because Y. "
    "C misreads Z, and D overstates W.",                                  # clean A-D
    "The remaining option, D, is wrong because it contradicts the data.",  # letter-named
    "The final option (D) is the best answer here.",                      # letter-named
    "On the other hand, the author qualifies this claim later.",          # 'other' prose
]

# --- LETTER-ANCHORED regression: must still flag (and must still pass D) ---
POSITIVE_LETTER = [
    "Option E is wrong because it overstates the effect.",
    "(E) overstates the relationship described in the passage.",
    "E) is also incorrect since the passage never claims this.",
    "H) is wrong; there is no such option.",
]
NEGATIVE_LETTER = [
    "D) is correct because the passage supports it directly.",   # D is valid -> no flag
    "Option D is the best answer.",
]


def run_unit():
    fails = []
    for t in POSITIVE_DESCRIPTIVE:
        hit = phantom(t, ABCD)
        ok = hit is not None
        print(f"  [{'PASS' if ok else 'FAIL'}] descriptive+ -> {hit!r:30} | {t[:55]}...")
        if not ok:
            fails.append(("descriptive should flag", t))
    for t in NEGATIVE:
        hit = phantom(t, ABCD)
        ok = hit is None
        print(f"  [{'PASS' if ok else 'FAIL'}] negative      -> {hit!r:30} | {t[:55]}...")
        if not ok:
            fails.append(("negative wrongly flagged", t, hit))
    for t in POSITIVE_LETTER:
        hit = phantom(t, ABCD)
        ok = hit is not None
        print(f"  [{'PASS' if ok else 'FAIL'}] letter+        -> {hit!r:30} | {t[:55]}...")
        if not ok:
            fails.append(("letter should flag", t))
    for t in NEGATIVE_LETTER:
        hit = phantom(t, ABCD)
        ok = hit is None
        print(f"  [{'PASS' if ok else 'FAIL'}] letter-        -> {hit!r:30} | {t[:55]}...")
        if not ok:
            fails.append(("letter wrongly flagged", t, hit))

    # Gating: descriptive scan must NOT run when choices are not exactly A-D.
    g = phantom("the remaining option is wrong", frozenset("ABCDE"))
    print(f"  [{'PASS' if g is None else 'FAIL'}] gating (A-E)   -> {g!r} (descriptive off when not exactly A-D)")
    if g is not None:
        fails.append(("gating failed", g))
    return fails


def run_realdata():
    run = Path(__file__).resolve().parent.parent / "runs" / "fix5_test"
    files = {
        "cars_passages.jsonl": "passage",
        "discrete_questions.jsonl": "flat",
        "science_passages.jsonl": "passage",
    }
    flagged = []
    total = 0
    for fname, kind in files.items():
        p = run / fname
        if not p.exists():
            print(f"  (skip {fname}: not found)")
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        qs = (
            [q for r in rows for q in r["questions"]] if kind == "passage" else rows
        )
        for q in qs:
            total += 1
            hit = phantom(q.get("explanation", ""), set(q.get("choices", {})))
            if hit:
                flagged.append((q["question_id"], hit))
    print(f"  scanned {total} shipped questions; {len(flagged)} flagged:")
    for qid, hit in flagged:
        print(f"    {qid}: {hit!r}")
    return flagged


EXPECTED_REAL = {
    "CARS_P0007_q02", "CARS_P0002_q01", "CARS_P0001_q06",
    "BB_2C_001_q006", "BB_1A_014_q007", "CP_4D_014_q005",
    "SP_BB_1A_018_01_q04",
    # Eighth leak, found by the new guard — the manual investigation missed it
    # because it grepped "remaining option" only; this one says "The remaining
    # choice misreads Vidal..." after addressing A-D by letter. Genuine phantom.
    "CARS_P0001_q05",
}

if __name__ == "__main__":
    print("=== UNIT CASES ===")
    fails = run_unit()
    print("\n=== REAL DATA (runs/fix5_test) ===")
    flagged = run_realdata()
    got = {qid for qid, _ in flagged}

    print("\n=== SUMMARY ===")
    missing = EXPECTED_REAL - got
    extra = got - EXPECTED_REAL
    if missing:
        print(f"  REAL-DATA MISS (known leak not caught): {sorted(missing)}")
    if extra:
        print(f"  REAL-DATA EXTRA (investigate for FP): {sorted(extra)}")
    if not missing and not extra:
        print(f"  real-data: all {len(EXPECTED_REAL)} known leaks caught, no extras")

    if fails:
        print(f"\n  {len(fails)} UNIT FAILURE(S):")
        for f in fails:
            print("   ", f)
        sys.exit(1)
    if missing:
        sys.exit(1)
    print("\nALL TESTS PASSED")
