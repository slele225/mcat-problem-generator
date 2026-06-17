"""Offline unit check for blind-solve diagnostic capture (no API calls).

Verifies that `_blind_solve_diagnostics` threads the blind-solver's chosen answer
AND confidence into the reject record, and that a missing/blank confidence is
recorded as "unknown" (distinct from "" = "we failed to capture it").

Run:  python scripts/check_blind_solve_confidence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.science_passage import _blind_solve_diagnostics


def _reject_record_confidence(raw_solve, parseable: bool) -> tuple[str, str]:
    """Reproduce the reject-record fields exactly as validate_science_question
    builds them, for a given blind-solve `results[1]`.

    `parseable=True` models a dict result (JSON parsed); `parseable=False` models
    the non-dict branch (the blind-solve response could not be parsed at all),
    where the "" defaults are left untouched.
    """
    validation = {"blind_solve_chosen": "", "blind_solve_confidence": ""}
    if parseable and isinstance(raw_solve, dict):
        validation["blind_solve_chosen"], validation["blind_solve_confidence"] = (
            _blind_solve_diagnostics(raw_solve)
        )
    record = {
        "blind_solve_chosen": validation.get("blind_solve_chosen", ""),
        "blind_solve_confidence": validation.get("blind_solve_confidence", ""),
    }
    return record["blind_solve_chosen"], record["blind_solve_confidence"]


CASES = [
    # (description, raw_solve, parseable, expected_chosen, expected_confidence)
    ("full response", {"chosen_answer": "b", "confidence": "high", "reasoning": "x"}, True, "B", "high"),
    ("mismatch w/ confidence", {"chosen_answer": "C", "confidence": "medium"}, True, "C", "medium"),
    ("missing confidence -> unknown", {"chosen_answer": "A"}, True, "A", "unknown"),
    ("blank confidence -> unknown", {"chosen_answer": "D", "confidence": "  "}, True, "D", "unknown"),
    ("missing chosen", {"confidence": "low"}, True, "", "low"),
    ("unparseable response -> '' (failed to capture)", "Let me work through this...", False, "", ""),
]


def main() -> int:
    failures = 0
    for desc, raw, parseable, exp_chosen, exp_conf in CASES:
        chosen, conf = _reject_record_confidence(raw, parseable)
        ok = (chosen == exp_chosen) and (conf == exp_conf)
        status = "ok " if ok else "FAIL"
        print(f"[{status}] {desc}: chosen={chosen!r} confidence={conf!r} "
              f"(expected chosen={exp_chosen!r} confidence={exp_conf!r})")
        if not ok:
            failures += 1

    # The core regression: a blind-solve MISMATCH that returned confidence must
    # carry that confidence into the reject record (the reported bug).
    _, conf = _reject_record_confidence({"chosen_answer": "C", "confidence": "high"}, True)
    assert conf == "high", f"confidence not threaded into reject record: {conf!r}"

    print()
    if failures:
        print(f"FAILED: {failures} case(s)")
        return 1
    print("All blind-solve confidence diagnostics carried through correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
