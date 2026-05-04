"""Eval metrics: tokens, latency, quality, cost.

Designed to mirror the columns of Uhura's sweep leaderboard so results are
directly comparable across the two repos.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import json
import math


# Pricing knobs. Defaults are placeholders — replace with whatever pricing
# assumption you're using in the Red Hat conference deck (Uhura uses
# frontier ~ $15/M input + $75/M output, small ~ $0.5/M input + $2/M output).
@dataclass(frozen=True)
class PricingModel:
    frontier_input_per_million: float = 15.0
    frontier_output_per_million: float = 75.0
    small_input_per_million: float = 0.5
    small_output_per_million: float = 2.0
    cpu_inference_per_hour: float = 0.40   # SPR CPU on Azure DC16es_v6 ballpark
    gpu_inference_per_hour: float = 3.50   # H100 ballpark


@dataclass
class EvalRecord:
    """One row of the eval table, per (input_stream, pipeline) pair."""

    pipeline: str                    # "A_tokenized" | "B_compressed_text" | "C_embedding_injection"
    input_token_count: int           # for A/B; soft-token count for C
    output_token_count: int
    prefill_latency_ms: float
    decode_latency_ms: float
    end_to_end_latency_ms: float
    output_text: str
    rouge_l: Optional[float] = None
    regime_correct: Optional[bool] = None
    cost_usd: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    tier: str,
    pricing: PricingModel = PricingModel(),
) -> float:
    """Estimate $/query for an LLM call given token counts and pricing tier."""
    if tier == "frontier":
        in_rate = pricing.frontier_input_per_million
        out_rate = pricing.frontier_output_per_million
    elif tier == "small":
        in_rate = pricing.small_input_per_million
        out_rate = pricing.small_output_per_million
    else:
        raise ValueError(f"unknown tier: {tier!r}")
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def rouge_l(prediction: str, reference: str) -> float:
    """Tiny ROUGE-L implementation — no external dependency.

    Computes longest-common-subsequence F1 over whitespace-tokenized text.
    For a "real" eval, swap to `rouge_score` package; this is here so the
    harness runs without pulling extra deps.
    """
    pred_toks = prediction.split()
    ref_toks = reference.split()
    if not pred_toks or not ref_toks:
        return 0.0

    # LCS via dynamic programming
    m, n = len(pred_toks), len(ref_toks)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if pred_toks[i] == ref_toks[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)


def compute_metrics(records: list[EvalRecord]) -> dict:
    """Aggregate per-pipeline summary from a list of records.

    Returns a dict like:
        {
            "A_tokenized": {"n": 50, "mean_tokens": 1024, "mean_cost": 0.21, ...},
            "B_compressed_text": {...},
            "C_embedding_injection": {...},
            "compression_ratios": {"B_vs_A": 2.8, "C_vs_A": 12.5, "C_vs_B": 4.5},
        }
    """
    by_pipe: dict[str, list[EvalRecord]] = {}
    for r in records:
        by_pipe.setdefault(r.pipeline, []).append(r)

    summary: dict = {}
    for pipe, rs in by_pipe.items():
        summary[pipe] = {
            "n": len(rs),
            "mean_input_tokens": sum(r.input_token_count for r in rs) / len(rs),
            "mean_output_tokens": sum(r.output_token_count for r in rs) / len(rs),
            "mean_prefill_ms": sum(r.prefill_latency_ms for r in rs) / len(rs),
            "mean_decode_ms": sum(r.decode_latency_ms for r in rs) / len(rs),
            "mean_e2e_ms": sum(r.end_to_end_latency_ms for r in rs) / len(rs),
            "mean_cost_usd": sum(r.cost_usd for r in rs) / len(rs),
            "mean_rouge_l": (
                sum(r.rouge_l or 0.0 for r in rs if r.rouge_l is not None)
                / max(1, sum(1 for r in rs if r.rouge_l is not None))
            ),
            "regime_accuracy": (
                sum(1 for r in rs if r.regime_correct)
                / max(1, sum(1 for r in rs if r.regime_correct is not None))
            ),
        }

    # Compression / cost ratios across pipelines
    def ratio(num_pipe: str, denom_pipe: str) -> Optional[float]:
        if num_pipe not in summary or denom_pipe not in summary:
            return None
        d = summary[denom_pipe]["mean_input_tokens"]
        if d == 0:
            return None
        return summary[num_pipe]["mean_input_tokens"] / d

    summary["ratios"] = {
        "B_input_vs_A": ratio("B_compressed_text", "A_tokenized"),
        "C_input_vs_A": ratio("C_embedding_injection", "A_tokenized"),
        "C_input_vs_B": ratio("C_embedding_injection", "B_compressed_text"),
    }
    return summary


def dump_summary(summary: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
