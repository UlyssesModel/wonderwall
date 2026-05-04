"""Tests for compressed-text rendering and the inference pipeline shapes.

CompressedTextPipeline.run() actually calls Scotty so we can't unit-test it
without a live endpoint. We test the rendering function (deterministic, pure)
and the EmbeddingInjectionPipeline's `soft_token_count` calculation
(deterministic given the adapter config).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import torch

from wonderwall.adapter import KirkProjectionAdapter
from wonderwall.interfaces import AdapterConfig
from wonderwall.kirk_client import StubKirkClient
from wonderwall.pipeline import (
    EmbeddingInjectionPipeline,
    render_kirk_output_as_text,
)


# ---------------------------------------------------------------------------
# render_kirk_output_as_text
# ---------------------------------------------------------------------------


def test_render_includes_n_and_marginals():
    kirk = StubKirkClient(n=8)
    ko = kirk.infer(torch.randn(8, 8) * 0.003)
    text = render_kirk_output_as_text(ko)
    assert "window_size=8" in text
    assert "row_expectations=[" in text
    assert "col_expectations=[" in text
    assert "global_expectation=" in text


def test_render_real_path_produces_finite_strings():
    kirk = StubKirkClient(n=16, use_complex=False)
    ko = kirk.infer(torch.randn(16, 16) * 0.003)
    text = render_kirk_output_as_text(ko, max_decimals=4)
    # No NaN / Inf representations leaking through
    assert "nan" not in text.lower()
    assert "inf" not in text.lower()


def test_render_includes_entropy_when_present():
    kirk = StubKirkClient(n=32)
    ko = kirk.infer(torch.randn(32, 32))
    text = render_kirk_output_as_text(ko)
    assert "entropy=" in text


def test_render_text_compresses_vs_raw():
    """Compressed text per window should be shorter than raw N×N text."""
    kirk = StubKirkClient(n=32)
    ko = kirk.infer(torch.randn(32, 32))
    compressed = render_kirk_output_as_text(ko)

    # Raw rendering of the 32×32 input is 32 rows × 32 floats × ~7 chars each
    # = ~7K characters. Compressed should be much shorter.
    raw_estimate = 32 * 32 * 7
    assert len(compressed) < raw_estimate / 2


# ---------------------------------------------------------------------------
# EmbeddingInjectionPipeline — soft token math
# ---------------------------------------------------------------------------


def test_soft_token_count_matches_adapter_contract():
    """Pipeline's soft_token_count should equal len(tensors) × (N+2)."""
    cfg = AdapterConfig(n=32, llm_hidden_dim=256, hidden_dim=64)
    adapter = KirkProjectionAdapter(cfg)
    kirk = StubKirkClient(n=32)
    fake_llm = MagicMock()
    fake_llm.config.device = "cpu"

    pipe = EmbeddingInjectionPipeline(kirk=kirk, adapter=adapter, llm=fake_llm)

    tensors = [torch.randn(32, 32) for _ in range(5)]
    assert pipe.soft_token_count(tensors) == 5 * (32 + 2)


def test_soft_token_count_scales_linearly():
    cfg = AdapterConfig(n=16, llm_hidden_dim=128, hidden_dim=32)
    adapter = KirkProjectionAdapter(cfg)
    kirk = StubKirkClient(n=16)
    fake_llm = MagicMock()
    fake_llm.config.device = "cpu"
    pipe = EmbeddingInjectionPipeline(kirk=kirk, adapter=adapter, llm=fake_llm)

    for T in (1, 2, 4, 8, 16):
        tensors = [torch.randn(16, 16) for _ in range(T)]
        assert pipe.soft_token_count(tensors) == T * (16 + 2)
