#!/usr/bin/env python3
"""Re-price a run's generation_metrics.json from stored token counts using the
CURRENT MODEL_PRICING table. Use after correcting or adding model prices.

Recomputes only cost fields (cost_usd, cost_per_accepted_usd, per-stage
cost_usd) and the recorded pricing table. Token counts and all funnel metrics
are left untouched, and the questions file is never read or modified.

Usage:
    python scripts/recost.py runs/<name>/generation_metrics.json          # preview
    python scripts/recost.py runs/<name>/generation_metrics.json --write  # save
"""
import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/recost.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import MODEL_PRICING, resolve_pricing  # noqa: E402


def _recompute(node) -> float:
    """Recompute a node's per-stage and total cost from its token counts."""
    total = 0.0
    for u in node.get("usage_by_stage", {}).values():
        price = resolve_pricing(u["model"])
        cost = (
            u["input_tokens"] / 1e6 * price["input"]
            + u["output_tokens"] / 1e6 * price["output"]
        )
        u["cost_usd"] = round(cost, 6)
        total += cost
    node["cost_usd"] = round(total, 6)
    accepted = node.get("accepted", {}).get("count", 0)
    node["cost_per_accepted_usd"] = round(total / accepted, 6) if accepted else None
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help="path to a run's generation_metrics.json")
    parser.add_argument(
        "--write", action="store_true",
        help="rewrite the file in place (default: preview only)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"No metrics file at {path}")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    old_total = summary.get("cost_usd")
    old_stage = {s: u.get("cost_usd") for s, u in summary.get("usage_by_stage", {}).items()}

    data["pricing_per_mtok_usd"] = MODEL_PRICING
    _recompute(summary)
    for topic in data.get("topics", []):
        _recompute(topic)

    print(f"{path}")
    print(f"  model={data.get('model')}  checker_model={data.get('checker_model')}")
    for stage, u in summary.get("usage_by_stage", {}).items():
        print(
            f"    {stage:12s} {u['model']:30s} "
            f"in={u['input_tokens']:>8} out={u['output_tokens']:>8}  "
            f"${old_stage.get(stage)} -> ${u['cost_usd']}"
        )
    print(f"  TOTAL cost_usd:   ${old_total} -> ${summary.get('cost_usd')}")
    print(f"  cost/accepted:    ${summary.get('cost_per_accepted_usd')}")

    if args.write:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("  (written in place)")
    else:
        print("  (preview only — pass --write to save)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
