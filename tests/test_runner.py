"""Tests for the eval CLI runner.

Covers argument parsing and the not-implemented guard for non-stub Kirk.
End-to-end execution requires Scotty + LLM, so we only exercise the
plumbing layer here.
"""
from __future__ import annotations

import sys

import pytest


# Import the CLI's argument-parser by reaching into the module. Don't run
# main() — it kicks off real eval. We just verify the parser shape.


def test_runner_module_imports():
    """Sanity: the module imports without side effects."""
    from eval import runner
    assert runner is not None


def test_runner_main_exists():
    from eval import runner
    assert callable(runner.main)


def test_runner_requires_use_stub_kirk_when_real_unavailable(monkeypatch, tmp_path):
    """Running --no-pipeline-c without --use-stub-kirk should raise NotImplementedError."""
    from eval import runner

    # Build a minimal distilled .pt so the loader doesn't fail first
    import torch
    from wonderwall.distill import DistillationItem, save_distilled

    items = [
        DistillationItem(
            tensors=[torch.randn(32, 32) * 0.001],
            target_text="test",
        )
    ]
    distill_path = tmp_path / "items.pt"
    save_distilled(items, str(distill_path))

    monkeypatch.setattr(
        sys, "argv",
        [
            "runner",
            "--distilled", str(distill_path),
            "--out", str(tmp_path / "out.json"),
            "--no-pipeline-a",
            "--no-pipeline-b",
            "--no-pipeline-c",
            # NO --use-stub-kirk
        ],
    )
    with pytest.raises(NotImplementedError, match="Real Kirk client wiring"):
        runner.main()


def test_runner_pipeline_c_requires_adapter(monkeypatch, tmp_path):
    """Pipeline C without --adapter should exit cleanly with a clear error."""
    from eval import runner

    import torch
    from wonderwall.distill import DistillationItem, save_distilled

    items = [
        DistillationItem(
            tensors=[torch.randn(32, 32) * 0.001],
            target_text="test",
        )
    ]
    distill_path = tmp_path / "items.pt"
    save_distilled(items, str(distill_path))

    monkeypatch.setattr(
        sys, "argv",
        [
            "runner",
            "--distilled", str(distill_path),
            "--out", str(tmp_path / "out.json"),
            "--use-stub-kirk",
            # Default config runs pipeline-c, but --adapter not passed
        ],
    )
    with pytest.raises(SystemExit, match="--adapter"):
        runner.main()
