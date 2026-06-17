r"""No-API unit check for the LaTeX-in-JSON parse fix and list-coercion logic.

Part 1 feeds raw model-style responses containing LaTeX through
parse_json_response and asserts (a) they parse and (b) the stored string keeps
correct single-backslash LaTeX so KaTeX renders it.

Part 2 checks _coerce_to_object (the list-unwrap helper) and the end-to-end
parse path: a bare dict must pass through untouched (the fix3_test regression
where valid objects raised "no object to unwrap"), a genuine [{...}] is
unwrapped, and an uncoercible array fails cleanly without that error escaping
parse_json_response. Run: python scripts/check_latex_json_repair.py
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_client import _coerce_to_object, parse_json_response  # noqa: E402

# (label, raw_response, field, expected_decoded_value)
CASES = [
    # 1. Correctly JSON-escaped LaTeX (what the model actually emits): the strict
    #    pass must return it untouched.
    (
        "escaped \\\\text + \\\\times (strict pass)",
        r'{"explanation": "Here $K_\\text{a} = 1.8 \\times 10^{-5}$ at equilibrium."}',
        "explanation",
        r"Here $K_\text{a} = 1.8 \times 10^{-5}$ at equilibrium.",
    ),
    # 2. The exact headline example, fully UNESCAPED. Both \text and \times start
    #    with "\t" (a valid JSON escape -> TAB), so strict parse silently corrupts
    #    them; the control-char detector catches it and re-parses via repair.
    (
        "unescaped \\text + \\times headline example",
        r'{"explanation": "At equilibrium $K_\text{a} = 1.8 \times 10^{-5}$."}',
        "explanation",
        r"At equilibrium $K_\text{a} = 1.8 \times 10^{-5}$.",
    ),
    # 3. UNESCAPED invalid-escape LaTeX commands (\Delta -> \D, \mu -> \m, bare
    #    "\ ", plus \text): strict parse fails, the repair fixes all of them.
    (
        "unescaped \\Delta, \\mu, bare backslash-space, \\text",
        r'{"stem": "Compute $\Delta G$ for a $\mu = 0.5$ ion at $5.0\ \text{V/cm}$."}',
        "stem",
        r"Compute $\Delta G$ for a $\mu = 0.5$ ion at $5.0\ \text{V/cm}$.",
    ),
    # 3. Mixed: some commands escaped, some not — repair must not double the
    #    already-escaped ones.
    (
        "mixed escaped + unescaped (repair, no double-escape)",
        r'{"e": "$\\rho V g$ and $\alpha$ with $[\\text{Fe}(\\text{H}_2\\text{O})_6]^{2+}$ plus $\pi$"}',
        "e",
        r"$\rho V g$ and $\alpha$ with $[\text{Fe}(\text{H}_2\text{O})_6]^{2+}$ plus $\pi$",
    ),
    # 4. Code-fenced response with unescaped LaTeX (fence strip + repair).
    (
        "code-fenced + unescaped (\\sigma, \\Delta)",
        '```json\n{"q": "Surface tension $\\sigma$ and $\\Delta T$ rise."}\n```',
        "q",
        r"Surface tension $\sigma$ and $\Delta T$ rise.",
    ),
    # 5. No LaTeX at all — must be completely unaffected.
    (
        "plain JSON, no latex",
        r'{"answer": "B", "confidence": "high"}',
        "answer",
        "B",
    ),
]


def check_coercion() -> int:
    """Exercise _coerce_to_object + the end-to-end parse path. Returns failures."""
    failures = 0

    def record(label, ok, detail):
        nonlocal failures
        status = "PASS" if ok else "FAIL"
        print("=" * 72)
        print(f"CASE: {label}")
        print(f"  {detail}")
        print(f"  -> {status}")
        if not ok:
            failures += 1

    # 1. THE REGRESSION CASE: a bare dict must pass through UNCHANGED.
    d = {"stem": "x"}
    out = _coerce_to_object(d)
    record("bare dict -> returned unchanged", out is d, f"GOT: {out!r}")

    # 2. THE 'len=3' BUG: a dict whose values include nested lists/structures
    #    must NOT be mistaken for a list. Returned unchanged.
    d2 = {"stem": "x", "choices": {"A": "a", "B": "b"}, "meta": [1, 2, 3]}
    out2 = _coerce_to_object(d2)
    record(
        "dict with nested list value -> returned unchanged",
        out2 is d2,
        f"GOT: {out2!r}",
    )

    # 3. A genuine list-wrapped object [{...}] -> unwrapped to the dict.
    out3 = _coerce_to_object([{"stem": "x"}])
    record(
        "[{...}] single-element list -> unwrapped to dict",
        out3 == {"stem": "x"},
        f"GOT: {out3!r}",
    )

    # 4. A multi-element list -> first dict (and a warning is logged).
    logs = []
    handler = logging.Handler()
    handler.emit = lambda rec: logs.append(rec.getMessage())  # type: ignore[assignment]
    lg = logging.getLogger("src.llm_client")
    lg.addHandler(handler)
    try:
        out4 = _coerce_to_object([{"a": 1}, {"b": 2}])
    finally:
        lg.removeHandler(handler)
    record(
        "[{...},{...}] multi-element list -> first dict + warning",
        out4 == {"a": 1} and any("array" in m for m in logs),
        f"GOT: {out4!r}; warned={any('array' in m for m in logs)}",
    )

    # 5. A non-coercible list (no dict element) -> _coerce_to_object raises.
    try:
        _coerce_to_object([1, 2, 3])
        record("[1,2,3] non-coercible -> _coerce_to_object raises", False, "no raise")
    except ValueError as e:
        record(
            "[1,2,3] non-coercible -> _coerce_to_object raises", True, f"raised: {e}"
        )

    # 6. The exact fix3_test failing example, fully valid -> parses to the dict.
    raw6 = '{"stem":"A glycosidase cleaves only $\\\\beta$-1,4 substrates."}'
    try:
        out6 = parse_json_response(raw6)
        record(
            "parse_json_response(valid $\\beta$ object) -> dict",
            isinstance(out6, dict) and out6.get("stem", "").startswith("A glycosidase"),
            f"GOT: {out6!r}",
        )
    except Exception as e:  # noqa: BLE001
        record(
            "parse_json_response(valid $\\beta$ object) -> dict",
            False,
            f"raised {type(e).__name__}: {e}",
        )

    # 7. End-to-end fallback guard: a response that fails ALL object candidates
    #    but whose "[...]" slice captures a stray array literal from the prose
    #    must raise the CLEAN "Could not parse" error, NOT "no object to unwrap".
    #    (Truncated object -> object candidates fail -> [1, 2, 3] slice reached.)
    raw7 = '{"stem": "consider the set [1, 2, 3] of values", "choices": {"A": "a'
    try:
        parse_json_response(raw7)
        record("truncated object w/ array in prose -> clean failure", False, "no raise")
    except ValueError as e:
        msg = str(e)
        record(
            "truncated object w/ array in prose -> clean failure",
            "Could not parse JSON" in msg and "no object to unwrap" not in msg,
            f"raised: {msg[:80]}",
        )
    return failures


def main() -> int:
    failures = 0
    for label, raw, field, expected in CASES:
        print("=" * 72)
        print(f"CASE: {label}")
        print(f"  RAW : {raw[:90]}{'...' if len(raw) > 90 else ''}")
        try:
            parsed = parse_json_response(raw)
            got = parsed[field]
            ok = got == expected
            status = "PASS" if ok else "FAIL"
            print(f"  GOT : {got!r}")
            if not ok:
                print(f"  WANT: {expected!r}")
            print(f"  -> {status}")
            if not ok:
                failures += 1
        except Exception as e:  # noqa: BLE001
            print(f"  -> FAIL (raised {type(e).__name__}: {e})")
            failures += 1
    print("=" * 72)
    print(f"LaTeX cases: {len(CASES) - failures}/{len(CASES)} passed")

    print("\n### LIST-COERCION CHECKS ###")
    coerce_failures = check_coercion()
    total = len(CASES) + 7
    total_failures = failures + coerce_failures
    print("=" * 72)
    print(f"OVERALL: {total - total_failures}/{total} cases passed")
    return 1 if total_failures else 0


if __name__ == "__main__":
    sys.exit(main())
