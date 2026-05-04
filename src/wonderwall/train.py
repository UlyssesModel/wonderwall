"""Training loop for the projection adapter.

Standard LLaVA-style distillation:
  - LLM frozen (no gradients)
  - Adapter trainable
  - Loss: standard next-token cross-entropy on the gold narration

The forward pass is:
  raw tensors → Kirk → adapter → kirk_embeds
  prefix_embeds + kirk_embeds + suffix_embeds + target_token_embeds → LLM
  loss = CE(logits[shifted], target_tokens)

Gradients flow back through the LLM (no parameter updates because frozen)
and into the adapter, which updates.

Memory note: training through a 31B-parameter LLM, even frozen, requires
a serious GPU (H100 80GB or A100 80GB minimum). Activation checkpointing
on the LLM forward is essential. Adapter itself trains in seconds.

For initial development on smaller hardware, swap the LLMConfig.model_name
to a Gemma 2B or Llama 3.2 1B. The adapter+pipeline code is model-agnostic.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .adapter import KirkProjectionAdapter
from .distill import DistillationItem, load_distilled
from .injection import EmbeddingInjectionLLM
from .interfaces import KirkClient
from .metrics_export import metrics


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DistillationDataset(Dataset):
    """Reads a saved distillation set; returns DistillationItem-shaped dicts.

    Tensors are kept on CPU; the training loop moves them to GPU as needed.
    """

    def __init__(self, path: str):
        self.items = load_distilled(path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DistillationItem:
        return self.items[idx]


def _collate_singleton(batch: list[DistillationItem]) -> DistillationItem:
    """Trivial collate for batch size 1 (the realistic case for this scale)."""
    if len(batch) != 1:
        raise NotImplementedError(
            "Batch size > 1 requires padding the variable-length kirk_embeds and "
            "target token sequences. Implement when you've decided on padding strategy."
        )
    return batch[0]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    distill_path: str
    save_path: str
    instruction_prefix: str = (
        "You are a market-state analyst. The following soft tokens encode "
        "recent price-tensor windows. Narrate what is happening and forecast "
        "the next interval.\n\nWindows: "
    )
    instruction_suffix: str = "\n\nNarration: "
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    log_every_steps: int = 10
    save_every_epochs: int = 1
    grad_accum_steps: int = 1
    use_amp: bool = True


class Trainer:
    """Trains the projection adapter against a distilled dataset.

    Usage:
        kirk = StubKirkClient(n=16) or InProcessKirkClient(...)
        adapter = KirkProjectionAdapter(adapter_config)
        llm = EmbeddingInjectionLLM(llm_config)
        trainer = Trainer(kirk, adapter, llm, train_config)
        trainer.fit()
    """

    def __init__(
        self,
        kirk: KirkClient,
        adapter: KirkProjectionAdapter,
        llm: EmbeddingInjectionLLM,
        config: TrainConfig,
    ):
        self.kirk = kirk
        self.adapter = adapter.to(llm.config.device)
        self.llm = llm
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.amp_enabled = config.use_amp and llm.config.device == "cuda"

    def _build_inputs(
        self, item: DistillationItem
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct the full embedding sequence + label tensor for one example.

        Returns:
            inputs_embeds: (1, T_total, H)
            labels:        (1, T_total)  -100 on positions that shouldn't contribute to loss
            attention_mask:(1, T_total)
        """
        device = self.llm.config.device

        # 1) Kirk forward pass on the input tensors
        kos = self.kirk.infer_stream(item.tensors)
        kirk_embeds = self.adapter.embed_stream(list(kos)).to(device)  # (1, T_kirk, H)

        # 2) Tokenize prefix, suffix, and target
        prefix = self.llm.text_to_token_embeds(self.config.instruction_prefix)
        suffix = self.llm.text_to_token_embeds(self.config.instruction_suffix)

        target_ids = self.llm.tokenizer(
            item.target_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        target_embeds = self.llm.model.get_input_embeddings()(target_ids)

        # 3) Stitch: [prefix | kirk | suffix | target]
        full = torch.cat([prefix, kirk_embeds, suffix, target_embeds], dim=1)

        # 4) Labels: -100 everywhere except on the target tokens
        T_total = full.shape[1]
        T_target = target_ids.shape[1]
        labels = torch.full((1, T_total), -100, dtype=torch.long, device=device)
        labels[0, -T_target:] = target_ids[0]

        attention_mask = torch.ones((1, T_total), dtype=torch.long, device=device)
        return full, labels, attention_mask

    def fit(self) -> None:
        ds = DistillationDataset(self.config.distill_path)
        loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_collate_singleton)

        import time as _time
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        step = 0
        for epoch in range(self.config.num_epochs):
            metrics.train_epoch.set(epoch)
            self.adapter.train()
            running_loss = 0.0
            for i, item in enumerate(loader):
                step_t0 = _time.perf_counter()
                with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=torch.bfloat16):
                    inputs_embeds, labels, attention_mask = self._build_inputs(item)
                    out = self.llm.forward_with_embeds(
                        inputs_embeds=inputs_embeds,
                        labels=labels,
                        attention_mask=attention_mask,
                    )
                    loss = out.loss / self.config.grad_accum_steps

                scaler.scale(loss).backward()

                if (i + 1) % self.config.grad_accum_steps == 0:
                    # Capture grad norm before optimizer step, after unscaling
                    scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.adapter.parameters(), max_norm=float("inf")
                    )
                    metrics.train_grad_norm.set(float(grad_norm))
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                step_loss = loss.item() * self.config.grad_accum_steps
                running_loss += step_loss
                step += 1
                metrics.train_steps_total.inc()
                metrics.train_loss.set(step_loss)
                metrics.train_loss_avg.set(running_loss / max(1, step))
                metrics.train_step_seconds.observe(_time.perf_counter() - step_t0)

                if step % self.config.log_every_steps == 0:
                    avg = running_loss / max(1, step)
                    print(
                        f"[train] epoch={epoch} step={step} "
                        f"loss={step_loss:.4f} avg={avg:.4f}"
                    )

            if (epoch + 1) % self.config.save_every_epochs == 0:
                self.save(f"{self.config.save_path}.epoch{epoch}.pt")

        self.save(self.config.save_path)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "adapter_state_dict": self.adapter.state_dict(),
                "adapter_config": self.adapter.config,
                "train_config": self.config,
            },
            path,
        )
        print(f"[train] saved adapter to {path}")
