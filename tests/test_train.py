"""Tests for the Trainer.

Exercises the full LLaVA-style training path with everything mocked except
the adapter — confirms gradients flow, frozen weights stay frozen, and
checkpoint save/load round-trips. Runs on CPU; no real LLM, no real Kirk.

The mock LLM mimics the HuggingFace causal-LM interface that
EmbeddingInjectionLLM wraps:

  - `.config.device`
  - `.tokenizer(text, return_tensors=...)` returning .input_ids
  - `.model.get_input_embeddings()(ids)` returning embeddings
  - `.forward_with_embeds(inputs_embeds, labels, attention_mask)` returning
    an object with `.loss` (a real scalar with grad)
  - `.text_to_token_embeds(text)` returning a tensor

These are exactly the surfaces train.Trainer reaches for; nothing else.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from wonderwall.adapter import KirkProjectionAdapter
from wonderwall.distill import DistillationItem, save_distilled
from wonderwall.interfaces import AdapterConfig
from wonderwall.kirk_client import StubKirkClient
from wonderwall.train import TrainConfig, Trainer


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Returns a deterministic 4-token sequence regardless of input text."""

    def __init__(self, hidden: int):
        self.hidden = hidden

    def __call__(self, text: str, return_tensors=None, add_special_tokens=True):
        # Map characters to token ids deterministically.
        ids = [(ord(c) % 32) + 1 for c in text[:4]] or [1, 2, 3, 4]
        while len(ids) < 4:
            ids.append(0)
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


class _MockEmbedding(nn.Module):
    """Embedding lookup with `vocab_size=64` and known hidden size."""

    def __init__(self, hidden: int, vocab_size: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed(ids)


class _MockHFModel(nn.Module):
    """Minimal stand-in for `EmbeddingInjectionLLM.model`.

    `forward(inputs_embeds, labels, attention_mask)` produces a tiny CE loss
    over a synthetic logits tensor. Crucially, the loss carries a real grad
    that flows back into `inputs_embeds`, so the adapter actually trains.
    """

    def __init__(self, hidden: int, vocab_size: int = 64):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embed = _MockEmbedding(hidden, vocab_size)
        # A linear "language head" so logits depend on inputs_embeds. This
        # is the path through which gradients reach the adapter.
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        # Freeze all the model's own parameters — only the adapter should train.
        for p in self.parameters():
            p.requires_grad = False

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds=None, labels=None, attention_mask=None):
        logits = self.lm_head(inputs_embeds)  # (B, T, V)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            mask = shift_labels != -100
            if mask.any():
                masked_logits = shift_logits[mask]
                masked_labels = shift_labels[mask]
                loss = nn.functional.cross_entropy(masked_logits, masked_labels)
            else:
                loss = torch.zeros((), requires_grad=True)
        return SimpleNamespace(loss=loss, logits=logits)


@dataclass
class _MockLLMConfig:
    device: str = "cpu"
    hidden_dim: int = 64


class _MockLLM:
    """Drop-in replacement for EmbeddingInjectionLLM in unit tests."""

    def __init__(self, hidden: int = 64):
        self.config = _MockLLMConfig(device="cpu", hidden_dim=hidden)
        self.model = _MockHFModel(hidden=hidden)
        self.tokenizer = _MockTokenizer(hidden=hidden)

    def forward_with_embeds(self, inputs_embeds, labels=None, attention_mask=None):
        return self.model(
            inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask
        )

    def text_to_token_embeds(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        return self.model.get_input_embeddings()(ids)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter_cfg():
    return AdapterConfig(n=8, llm_hidden_dim=64, hidden_dim=32)


@pytest.fixture
def adapter(adapter_cfg):
    return KirkProjectionAdapter(adapter_cfg)


@pytest.fixture
def kirk():
    return StubKirkClient(n=8)


@pytest.fixture
def llm():
    return _MockLLM(hidden=64)


@pytest.fixture
def distill_path(tmp_path):
    """Write a tiny .pt with a few items so the trainer has something to load."""
    items = []
    for _ in range(4):
        tensors = [torch.randn(8, 8) * 0.003 for _ in range(2)]
        items.append(DistillationItem(
            tensors=tensors,
            target_text="market consolidating",
            metadata={"gold_regime": "calm"},
        ))
    p = tmp_path / "distilled.pt"
    save_distilled(items, str(p))
    return str(p)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trainer_constructs_optimizer_only_for_adapter(kirk, adapter, llm, distill_path, tmp_path):
    cfg = TrainConfig(
        distill_path=distill_path,
        save_path=str(tmp_path / "out.pt"),
        num_epochs=1,
        learning_rate=1e-3,
        log_every_steps=100,
        grad_accum_steps=1,
        use_amp=False,
    )
    trainer = Trainer(kirk=kirk, adapter=adapter, llm=llm, config=cfg)

    # Optimizer should hold only adapter params; not LLM params.
    opt_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    adapter_ids = {id(p) for p in adapter.parameters()}
    llm_ids = {id(p) for p in llm.model.parameters()}
    assert opt_param_ids == adapter_ids
    assert opt_param_ids.isdisjoint(llm_ids)


def test_build_inputs_emits_consistent_shapes(kirk, adapter, llm, distill_path, tmp_path):
    cfg = TrainConfig(
        distill_path=distill_path,
        save_path=str(tmp_path / "out.pt"),
        num_epochs=1,
        log_every_steps=100,
        use_amp=False,
    )
    trainer = Trainer(kirk=kirk, adapter=adapter, llm=llm, config=cfg)

    item = DistillationItem(
        tensors=[torch.randn(8, 8) * 0.003 for _ in range(3)],
        target_text="volatility spiking",
    )
    embeds, labels, mask = trainer._build_inputs(item)

    # Same time dim across all three
    assert embeds.shape[1] == labels.shape[1] == mask.shape[1]
    assert embeds.shape[2] == llm.config.hidden_dim


def test_fit_runs_one_epoch_without_crashing(kirk, adapter, llm, distill_path, tmp_path):
    cfg = TrainConfig(
        distill_path=distill_path,
        save_path=str(tmp_path / "out.pt"),
        num_epochs=1,
        learning_rate=1e-3,
        log_every_steps=100,
        save_every_epochs=1,
        grad_accum_steps=1,
        use_amp=False,
    )
    trainer = Trainer(kirk=kirk, adapter=adapter, llm=llm, config=cfg)
    trainer.fit()

    # Checkpoint exists
    assert os.path.exists(cfg.save_path)


def test_checkpoint_save_load_round_trip(kirk, adapter, llm, distill_path, tmp_path):
    cfg = TrainConfig(
        distill_path=distill_path,
        save_path=str(tmp_path / "ckpt.pt"),
        num_epochs=1,
        log_every_steps=100,
        save_every_epochs=1,
        grad_accum_steps=1,
        use_amp=False,
    )
    trainer = Trainer(kirk=kirk, adapter=adapter, llm=llm, config=cfg)
    trainer.fit()

    # Construct a fresh adapter, load the checkpoint, weights should match
    adapter2 = KirkProjectionAdapter(adapter.config)
    ck = torch.load(cfg.save_path, weights_only=False)
    adapter2.load_state_dict(ck["adapter_state_dict"])

    for p1, p2 in zip(adapter.parameters(), adapter2.parameters()):
        assert torch.allclose(p1, p2), "checkpoint round-trip altered weights"


def test_llm_weights_unchanged_after_training(kirk, adapter, llm, distill_path, tmp_path):
    """LLM is frozen — no parameter updates should land on it."""
    cfg = TrainConfig(
        distill_path=distill_path,
        save_path=str(tmp_path / "out.pt"),
        num_epochs=1,
        learning_rate=1e-3,
        log_every_steps=100,
        grad_accum_steps=1,
        use_amp=False,
    )
    # Snapshot LLM weights before
    pre = [p.clone() for p in llm.model.parameters()]
    trainer = Trainer(kirk=kirk, adapter=adapter, llm=llm, config=cfg)
    trainer.fit()
    post = list(llm.model.parameters())
    for a, b in zip(pre, post):
        assert torch.equal(a, b), "frozen LLM parameters changed during training"


def test_adapter_weights_changed_after_training(kirk, adapter, llm, distill_path, tmp_path):
    """Sanity: training actually updated the adapter (LR > 0, real loss)."""
    cfg = TrainConfig(
        distill_path=distill_path,
        save_path=str(tmp_path / "out.pt"),
        num_epochs=2,
        learning_rate=1e-2,
        log_every_steps=100,
        grad_accum_steps=1,
        use_amp=False,
    )
    pre = [p.clone() for p in adapter.parameters()]
    trainer = Trainer(kirk=kirk, adapter=adapter, llm=llm, config=cfg)
    trainer.fit()
    post = list(adapter.parameters())
    differs = any(not torch.equal(a, b) for a, b in zip(pre, post))
    assert differs, "adapter weights identical after training — gradient path broken"
