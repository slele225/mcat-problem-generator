"""Metrics for the discrete generation pipeline.

Tracks the generation -> adversarial-review -> blind-solve funnel and the
token/cost accounting for each stage, logs a per-topic summary as each topic
completes and an aggregate summary at the end of the run, and writes the
numbers to output/generation_metrics.json.

Cost is computed from the *actual* model each stage runs on, so the estimate
stays accurate when the blind-solve checker model differs from the main model
(see Config.checker_model). Generation and adversarial review run on the main
model (Sonnet); blind solve runs on the checker model (Haiku).
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# USD per million tokens, keyed by the real model name. Keeping pricing keyed by
# model (rather than by stage) means the per-stage cost stays correct no matter
# which model a stage is pointed at.
#
# Verified against https://platform.claude.com/docs/en/about-claude/pricing on
# 2026-06-01. NOTE: prices vary WITHIN a model family across generations
# (e.g. Opus 4.1 and earlier were $15/$75; Haiku 3.5 was $0.80/$4), so keep an
# exact per-model entry here rather than relying on the family fallback for old
# model ids.
MODEL_PRICING = {
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
}


def resolve_pricing(model: str) -> dict:
    """Return {input, output} $/MTok for a model.

    Exact per-model entries in MODEL_PRICING are authoritative. The family
    fallback uses current-generation prices (Opus 4.5+, Sonnet 4.x, Haiku 4.5);
    older models in a family (Opus 4.1 = $15/$75, Haiku 3.5 = $0.80/$4) must be
    added to MODEL_PRICING explicitly or they will be mispriced here.
    """
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    lowered = (model or "").lower()
    if "opus" in lowered:
        return {"input": 5.0, "output": 25.0}
    if "sonnet" in lowered:
        return {"input": 3.0, "output": 15.0}
    if "haiku" in lowered:
        return {"input": 1.0, "output": 5.0}
    logger.warning(f"No pricing known for model '{model}'; counting its cost as $0.")
    return {"input": 0.0, "output": 0.0}


@dataclass
class TopicMetrics:
    """Funnel + token/cost counters for a single topic (or an aggregate)."""

    topic_id: str = ""
    topic: str = ""
    target: int = 0
    questions_attempted: int = 0  # generation calls (one per attempt)
    generation_parsed: int = 0    # attempts that produced a parseable question
    adversarial_runs: int = 0
    adversarial_passes: int = 0
    blind_solve_runs: int = 0
    blind_solve_passes: int = 0
    accepted: int = 0             # passed both checks (= questions kept)
    slots_failed: int = 0         # question slots that exhausted all retries
    # stage -> {"model", "calls", "input_tokens", "output_tokens"}
    usage: dict = field(default_factory=dict)

    def add_usage(
        self, stage: str, model: str, calls: int, input_tokens: int, output_tokens: int
    ):
        """Accumulate token usage for a stage over `calls` completed API calls."""
        u = self.usage.get(stage)
        if u is None:
            u = {"model": model, "calls": 0, "input_tokens": 0, "output_tokens": 0}
            self.usage[stage] = u
        u["model"] = model
        u["calls"] += int(calls)
        u["input_tokens"] += int(input_tokens or 0)
        u["output_tokens"] += int(output_tokens or 0)

    def record_usage(self, stage: str, model: str, input_tokens: int, output_tokens: int):
        """Accumulate one completed API call's token usage for a stage."""
        self.add_usage(stage, model, 1, input_tokens, output_tokens)

    @property
    def total_api_calls(self) -> int:
        return sum(u["calls"] for u in self.usage.values())

    @property
    def total_input_tokens(self) -> int:
        return sum(u["input_tokens"] for u in self.usage.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(u["output_tokens"] for u in self.usage.values())

    @staticmethod
    def _stage_cost(u: dict) -> float:
        p = resolve_pricing(u["model"])
        return u["input_tokens"] / 1e6 * p["input"] + u["output_tokens"] / 1e6 * p["output"]

    @property
    def cost_usd(self) -> float:
        return sum(self._stage_cost(u) for u in self.usage.values())

    @classmethod
    def aggregate(cls, buckets, topic_id: str = "__ALL__", topic: str = "All topics"):
        """Sum a list of TopicMetrics into a single bucket."""
        agg = cls(topic_id=topic_id, topic=topic)
        int_fields = (
            "target", "questions_attempted", "generation_parsed",
            "adversarial_runs", "adversarial_passes",
            "blind_solve_runs", "blind_solve_passes",
            "accepted", "slots_failed",
        )
        for b in buckets:
            for f_name in int_fields:
                setattr(agg, f_name, getattr(agg, f_name) + getattr(b, f_name))
            for stage, u in b.usage.items():
                agg.add_usage(
                    stage, u["model"], u["calls"], u["input_tokens"], u["output_tokens"]
                )
        return agg

    def to_report(self) -> dict:
        att = self.questions_attempted
        acc = self.accepted

        def rate(n):
            return round(n / att, 4) if att else None

        return {
            "topic_id": self.topic_id,
            "topic": self.topic,
            "target_per_topic": self.target,
            "questions_attempted": att,
            "generation_parsed": self.generation_parsed,
            "passed_adversarial_review": {
                "count": self.adversarial_passes,
                "rate_of_attempts": rate(self.adversarial_passes),
            },
            "passed_blind_solve": {
                "count": self.blind_solve_passes,
                "rate_of_attempts": rate(self.blind_solve_passes),
            },
            "accepted": {"count": acc, "rate_of_attempts": rate(acc)},
            "slots_failed": self.slots_failed,
            "api_calls_total": self.total_api_calls,
            "api_calls_per_accepted": round(self.total_api_calls / acc, 3) if acc else None,
            "tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
            },
            "cost_usd": round(self.cost_usd, 6),
            "cost_per_accepted_usd": round(self.cost_usd / acc, 6) if acc else None,
            "usage_by_stage": {
                stage: {
                    "model": u["model"],
                    "calls": u["calls"],
                    "input_tokens": u["input_tokens"],
                    "output_tokens": u["output_tokens"],
                    "cost_usd": round(self._stage_cost(u), 6),
                }
                for stage, u in self.usage.items()
            },
        }


def _summary_lines(m: TopicMetrics, header: str) -> list[str]:
    att = m.questions_attempted
    acc = m.accepted

    def pct(n):
        return f"{(n / att * 100):.1f}%" if att else "n/a"

    per_call = f"{m.total_api_calls / acc:.2f}" if acc else "n/a"
    per_cost = f"${m.cost_usd / acc:.4f}" if acc else "n/a"
    return [
        header,
        f"  Questions attempted (generation calls): {att}",
        f"  Passed adversarial review: {m.adversarial_passes} ({pct(m.adversarial_passes)})",
        f"  Passed blind solve:        {m.blind_solve_passes} ({pct(m.blind_solve_passes)})",
        f"  Final accepted:            {acc} ({pct(acc)})",
        f"  API calls / accepted:      {per_call}",
        f"  Est. cost / accepted:      {per_cost}",
        f"  Topic totals: ${m.cost_usd:.4f} | {m.total_api_calls} API calls | "
        f"tokens in/out {m.total_input_tokens:,}/{m.total_output_tokens:,}",
    ]


class MetricsTracker:
    """Owns per-topic metric buckets and the run-level aggregate + output file."""

    def __init__(self, model: str, checker_model: str, metrics_path: Optional[str] = None):
        self.model = model
        self.checker_model = checker_model
        self.metrics_path = metrics_path
        self.topics: list[TopicMetrics] = []
        self.current: Optional[TopicMetrics] = None

    def start_topic(self, topic_id: str, topic: str, target: int) -> TopicMetrics:
        self.current = TopicMetrics(topic_id=topic_id, topic=topic, target=target)
        return self.current

    def finish_topic(self) -> Optional[TopicMetrics]:
        """Log and persist the current topic's metrics. Skips no-op topics."""
        m = self.current
        self.current = None
        if m is None or m.questions_attempted == 0:
            return None
        self.topics.append(m)
        for line in _summary_lines(m, f"Topic metrics: {m.topic_id} — {m.topic}"):
            logger.info(line)
        self.write_json()  # persist incrementally for crash-safety
        return m

    def aggregate(self) -> TopicMetrics:
        return TopicMetrics.aggregate(self.topics)

    def log_summary(self) -> TopicMetrics:
        agg = self.aggregate()
        logger.info("=" * 60)
        header = f"RUN SUMMARY — {len(self.topics)} topic(s) processed this run"
        for line in _summary_lines(agg, header):
            logger.info(line)
        logger.info("=" * 60)
        return agg

    def write_json(self):
        if not self.metrics_path:
            return
        agg = self.aggregate()
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "checker_model": self.checker_model,
            "pricing_per_mtok_usd": MODEL_PRICING,
            "note": (
                "Covers only topics processed in THIS run. Topics already completed "
                "via checkpoints are skipped and excluded, since no API calls are "
                "made for them. Rates are relative to questions attempted "
                "(generation calls)."
            ),
            "summary": agg.to_report(),
            "topics": [m.to_report() for m in self.topics],
        }
        path = Path(self.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.debug(f"Wrote generation metrics to {path}")
