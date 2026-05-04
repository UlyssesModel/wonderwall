"""Tests for eval metrics — token cost, ROUGE-L, summary aggregation."""
from __future__ import annotations

from eval.metrics import (
    EvalRecord,
    PricingModel,
    compute_metrics,
    estimate_cost,
    rouge_l,
)


def test_estimate_cost_frontier_vs_small():
    pricing = PricingModel()
    frontier = estimate_cost(1000, 200, "frontier", pricing)
    small = estimate_cost(1000, 200, "small", pricing)
    assert frontier > small  # frontier costs more by construction
    # Ratio sanity check: ~10–30× per the Uhura sweep narrative
    assert frontier / small > 5


def test_rouge_l_identity():
    assert abs(rouge_l("the quick brown fox", "the quick brown fox") - 1.0) < 1e-6


def test_rouge_l_disjoint():
    assert rouge_l("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_rouge_l_partial_overlap():
    score = rouge_l("the cat sat on the mat", "the dog sat on the mat")
    assert 0.5 < score < 1.0  # significant overlap, not identical


def test_compute_metrics_shape():
    records = [
        EvalRecord(
            pipeline="A_tokenized", input_token_count=1000, output_token_count=200,
            prefill_latency_ms=100, decode_latency_ms=200, end_to_end_latency_ms=300,
            output_text="hello", rouge_l=0.5, cost_usd=0.02,
        ),
        EvalRecord(
            pipeline="B_compressed_text", input_token_count=300, output_token_count=200,
            prefill_latency_ms=50, decode_latency_ms=100, end_to_end_latency_ms=150,
            output_text="hello", rouge_l=0.5, cost_usd=0.001,
        ),
        EvalRecord(
            pipeline="C_embedding_injection", input_token_count=80, output_token_count=200,
            prefill_latency_ms=30, decode_latency_ms=100, end_to_end_latency_ms=130,
            output_text="hello", rouge_l=0.5, cost_usd=0.0005,
        ),
    ]
    summary = compute_metrics(records)
    assert "A_tokenized" in summary
    assert "B_compressed_text" in summary
    assert "C_embedding_injection" in summary
    assert summary["ratios"]["B_input_vs_A"] < 1
    assert summary["ratios"]["C_input_vs_A"] < 1
    assert summary["ratios"]["C_input_vs_B"] < 1
