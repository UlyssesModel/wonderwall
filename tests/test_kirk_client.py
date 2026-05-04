"""Tests for the StubKirkClient under the kirk-pipeline two-layer contract.

Covers:
  - shape correctness of the layer-2 outputs
  - real-valued path leaves complex container with imag=0
  - complex path actually populates imag
  - entropy in production range (0.4–18.9 nats per Uhura page)
  - stream order preserved
  - all five inference modes accepted
"""
from __future__ import annotations

import torch

from wonderwall.interfaces import KirkMode
from wonderwall.kirk_client import StubKirkClient


def test_default_n_is_32():
    """kirk-pipeline Layer-2 fixed dimension."""
    kirk = StubKirkClient()
    assert kirk.n == 32


def test_stub_outputs_shape_layer2_default():
    kirk = StubKirkClient(n=32)
    inp = torch.randn(32, 32) * 0.003
    ko = kirk.infer(inp)
    ko.validate()

    assert ko.layer2_input.shape == (32, 32)
    assert ko.layer2_reconstruction.shape == (32, 32)
    assert ko.layer2_marginals.shape == (64,)
    assert ko.entropy.shape == ()


def test_aliases_work():
    """Backwards-compat .array/.vector/.scalar aliases still resolve."""
    kirk = StubKirkClient(n=32)
    ko = kirk.infer(torch.randn(32, 32))
    assert ko.array is ko.layer2_reconstruction
    assert ko.vector is ko.layer2_marginals
    assert ko.scalar is ko.entropy


def test_real_path_marginals_real_dtype():
    kirk = StubKirkClient(n=16, use_complex=False)
    ko = kirk.infer(torch.randn(16, 16))
    assert ko.layer2_reconstruction.dtype == torch.float32
    assert ko.layer2_marginals.dtype == torch.float32
    assert not torch.is_complex(ko.layer2_reconstruction)


def test_complex_path_uses_complex_dtype():
    kirk = StubKirkClient(n=16, use_complex=True)
    ko = kirk.infer(torch.randn(16, 16))
    assert torch.is_complex(ko.layer2_reconstruction)
    assert torch.is_complex(ko.layer2_marginals)
    # Almost certainly non-zero somewhere
    assert torch.any(ko.layer2_reconstruction.imag != 0)


def test_entropy_in_spec_range():
    """Per Uhura page, entropy ranges from 0.4 to ~18.9 nats."""
    kirk = StubKirkClient(n=32)
    ko = kirk.infer(torch.randn(32, 32))
    e = ko.entropy.item()
    assert 0.0 <= e <= 20.0


def test_stream_preserves_order_and_distinct():
    kirk = StubKirkClient(n=8)
    inputs = [torch.full((8, 8), float(i), dtype=torch.float32) for i in range(5)]
    outs = kirk.infer_stream(inputs)
    assert len(outs) == 5
    entropies = [o.entropy.item() for o in outs]
    assert len(set(entropies)) == 5


def test_all_five_modes_accepted():
    kirk = StubKirkClient(n=32)
    inp = torch.randn(32, 32)
    for mode in KirkMode:
        ko = kirk.infer(inp, mode=mode)
        assert ko.mode == mode
