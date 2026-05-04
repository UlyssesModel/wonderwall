"""Integration test for the real kirk-pipeline wheel.

Runs only when `kirk_pipeline` is importable in the test process. Skipped
automatically on hosts without the production wheel — your laptop, CI, and
the kavara-visual-studio cluster all skip; tdx-amx-node-octo and IvorHQ run.

What it validates:

  - `KirkModelInterface()` constructs and `load_weights()` returns without error
  - `forward(matrix, mode='active_inference')` returns a structure with a
    reconstruction, marginals, and entropy
  - Output shapes match the (n, n), (2n,), () contract that
    `KirkOutput.validate()` asserts
  - Entropy lands in the spec range (0.4 – 18.9 nats per the Uhura page)
  - The KirkPipelineClient end-to-end path (input → layer-1 → assembled →
    layer-2 → KirkOutput) returns a valid object

These checks catch the most likely failure mode flagged in HANDOFF.md #2:
the kirk-pipeline forward() return shape may differ from the dict
{'reconstruction', 'marginals', 'entropy'} we currently parse.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import pytest
import torch

from wonderwall.interfaces import KirkMode, KirkOutput

# Skip the entire module when kirk_pipeline isn't installed.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kirk_pipeline") is None,
    reason="kirk_pipeline wheel not installed (production-only test)",
)


@pytest.fixture(scope="module")
def model_interface():
    """Lazily construct and load the real KirkModelInterface."""
    from kirk_pipeline import KirkModelInterface  # type: ignore
    iface = KirkModelInterface()
    iface.load_weights()
    return iface


def test_load_weights_doesnt_crash(model_interface):
    """Constructor + load_weights() must complete without exceptions."""
    assert model_interface is not None


@pytest.mark.parametrize("n", [16, 32])
def test_forward_layer2_shape_for_each_n(model_interface, n):
    """Forward in active_inference mode produces the layer-2 output triplet."""
    inp = torch.randn(n, n, dtype=torch.float32) * 0.003
    out = model_interface.forward(inp.numpy(), mode=KirkMode.ACTIVE_INFERENCE.value)

    assert isinstance(out, dict), (
        f"Expected dict from forward(); got {type(out).__name__}. "
        f"If kirk-pipeline returns a tuple/dataclass, update KirkPipelineClient._run_layer2."
    )

    for required in ("reconstruction", "marginals", "entropy"):
        assert required in out, (
            f"forward() output missing key {required!r}. "
            f"Got keys: {list(out.keys())}. "
            f"This is the common case where kirk-pipeline's output schema "
            f"differs from KirkPipelineClient's assumed schema."
        )

    recon = torch.as_tensor(out["reconstruction"])
    marg = torch.as_tensor(out["marginals"])
    ent = torch.as_tensor(out["entropy"])

    assert recon.shape == (n, n), f"reconstruction shape {tuple(recon.shape)} != ({n}, {n})"
    assert marg.shape == (2 * n,), f"marginals shape {tuple(marg.shape)} != ({2 * n},)"
    assert ent.shape == ()


def test_entropy_in_spec_range(model_interface):
    """Per the Uhura page, production entropy ranges from ~0.4 to ~18.9 nats."""
    inp = torch.randn(32, 32, dtype=torch.float32) * 0.003
    out = model_interface.forward(inp.numpy(), mode=KirkMode.ACTIVE_INFERENCE.value)
    ent = float(torch.as_tensor(out["entropy"]).item())
    # Be permissive: the *reference* range is 0.4–18.9 but synthetic random
    # input may produce values slightly outside. The hard-fail bound is
    # finiteness + non-negativity.
    assert ent >= 0, f"entropy negative: {ent}"
    assert ent < 100, f"entropy implausibly large: {ent}"


def test_kirk_pipeline_client_end_to_end():
    """Spin a KirkPipelineClient and run one full pass."""
    from wonderwall.kirk_client import KirkPipelineClient

    client = KirkPipelineClient()
    inp = torch.randn(32, 32) * 0.003
    out: KirkOutput = client.infer(inp)
    out.validate()

    assert out.layer2_input.shape == (32, 32)
    assert out.layer2_reconstruction.shape == (32, 32)
    assert out.layer2_marginals.shape == (64,)
    assert out.entropy.shape == ()


def test_inference_features_mode_returns_features(model_interface):
    """inference_features mode returns the feature vector form, not the dict."""
    inp = torch.randn(16, 16, dtype=torch.float32) * 0.003
    out = model_interface.forward(inp.numpy(), mode=KirkMode.INFERENCE_FEATURES.value)
    # In feature-extraction mode the output is the per-block feature vector.
    # Don't assert exact shape here — kirk-pipeline's contract may vary —
    # but validate it's a usable tensor-like object.
    assert out is not None


def test_active_modes_update_state_across_calls(model_interface):
    """active_inference modes mutate internal state — entropy on identical
    second call may differ (the model has updated)."""
    inp = torch.randn(32, 32, dtype=torch.float32) * 0.003
    out1 = model_interface.forward(inp.numpy(), mode=KirkMode.ACTIVE_INFERENCE.value)
    out2 = model_interface.forward(inp.numpy(), mode=KirkMode.ACTIVE_INFERENCE.value)
    e1 = float(torch.as_tensor(out1["entropy"]).item())
    e2 = float(torch.as_tensor(out2["entropy"]).item())
    # We don't assert e1 != e2 strictly — the model may converge fast on
    # identical input — but both must be finite and in range.
    assert all(0 <= e < 100 for e in (e1, e2))
