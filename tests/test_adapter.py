"""Shape and gradient-flow tests for the projection adapter.

CPU only; no transformers needed.
"""
from __future__ import annotations

import torch

from wonderwall.adapter import KirkProjectionAdapter
from wonderwall.interfaces import AdapterConfig
from wonderwall.kirk_client import StubKirkClient


def test_default_config_is_layer2_shape():
    cfg = AdapterConfig()
    assert cfg.n == 32
    assert cfg.tokens_per_sample == 34


def test_real_valued_output_shape_n32():
    cfg = AdapterConfig(n=32, llm_hidden_dim=512, hidden_dim=128)
    adapter = KirkProjectionAdapter(cfg)
    array = torch.randn(2, 32, 32)
    vector = torch.randn(2, 64)
    out = adapter(array, vector)
    assert out.shape == (2, cfg.tokens_per_sample, cfg.llm_hidden_dim)


def test_complex_path_doubles_input_width():
    cfg = AdapterConfig(n=8, llm_hidden_dim=256, hidden_dim=64, use_complex=True)
    adapter = KirkProjectionAdapter(cfg)
    array = torch.complex(torch.randn(1, 8, 8), torch.randn(1, 8, 8))
    vector = torch.complex(torch.randn(1, 16), torch.randn(1, 16))
    out = adapter(array, vector)
    assert out.shape == (1, cfg.tokens_per_sample, cfg.llm_hidden_dim)


def test_stub_kirk_to_adapter_end_to_end():
    cfg = AdapterConfig(n=32, llm_hidden_dim=512, hidden_dim=128)
    adapter = KirkProjectionAdapter(cfg)
    kirk = StubKirkClient(n=32)
    ko = kirk.infer(torch.randn(32, 32) * 0.003)
    embeds = adapter.embed_kirk_output(ko)
    assert embeds.shape == (1, cfg.tokens_per_sample, cfg.llm_hidden_dim)


def test_streaming_concatenates():
    cfg = AdapterConfig(n=32, llm_hidden_dim=128, hidden_dim=64)
    adapter = KirkProjectionAdapter(cfg)
    kirk = StubKirkClient(n=32)
    tensors = [torch.randn(32, 32) * 0.003 for _ in range(5)]
    kos = kirk.infer_stream(tensors)
    embeds = adapter.embed_stream(kos)
    assert embeds.shape == (1, 5 * cfg.tokens_per_sample, cfg.llm_hidden_dim)


def test_gradient_flows_to_adapter():
    cfg = AdapterConfig(n=4, llm_hidden_dim=32, hidden_dim=16)
    adapter = KirkProjectionAdapter(cfg)
    a = torch.randn(1, 4, 4)
    v = torch.randn(1, 8)
    loss = adapter(a, v).sum()
    loss.backward()
    assert all(p.grad is not None for p in adapter.parameters())


def test_param_count_in_expected_range():
    cfg = AdapterConfig(n=32, llm_hidden_dim=5376, hidden_dim=1024)
    adapter = KirkProjectionAdapter(cfg)
    n = adapter.num_trainable_parameters
    assert 1_000_000 < n < 100_000_000


def test_token_count_per_window():
    """For n=32, each window emits 34 token-equivalents to the LLM."""
    cfg = AdapterConfig(n=32, llm_hidden_dim=128, hidden_dim=32)
    adapter = KirkProjectionAdapter(cfg)
    assert adapter.config.tokens_per_sample == 34
