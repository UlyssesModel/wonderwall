"""End-to-end inference pipeline: raw N×N tensor → narration.

Two pipelines, sharing everything upstream of the LLM:

  Pipeline B (production, today):
    raw tensor → Kirk → render-as-text → ScottyClient → narration

  Pipeline C (the new build):
    raw tensor → Kirk → KirkProjectionAdapter → EmbeddingInjectionLLM → narration

The eval harness instantiates both and runs them on the same input stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from .adapter import KirkProjectionAdapter
from .injection import EmbeddingInjectionLLM, ScottyClient
from .interfaces import KirkClient, KirkOutput
from .prompts import (
    PROMPT_REGIME_NARRATION_B_V1,
    PROMPT_REGIME_NARRATION_C_PREFIX_V1,
    PROMPT_REGIME_NARRATION_C_SUFFIX_V1,
)


# ---------------------------------------------------------------------------
# Compressed-text rendering for Pipeline B
# ---------------------------------------------------------------------------


def render_kirk_output_as_text(ko: KirkOutput, max_decimals: int = 4) -> str:
    """Render a KirkOutput into a compact text block for token-input LLMs.

    Mirrors the Pipeline B convention used in Uhura's `compare` / `sweep`:
    a 64-dim Kirk latent + a feature dict. Here we use Vector + Scalar +
    entropy as the compact summary; the full Array is omitted to keep the
    text representation small (the whole point of Pipeline B).
    """
    fmt = f"{{:.{max_decimals}f}}"

    # Vector: split row-marginals and col-marginals
    n = ko.n
    if torch.is_complex(ko.vector):
        v = ko.vector.real
    else:
        v = ko.vector
    row_marg = ", ".join(fmt.format(x) for x in v[:n].tolist())
    col_marg = ", ".join(fmt.format(x) for x in v[n:].tolist())

    scalar = ko.scalar.real if torch.is_complex(ko.scalar) else ko.scalar
    entropy = ko.entropy.item() if ko.entropy is not None else None

    parts = [
        f"window_size={n}",
        f"row_expectations=[{row_marg}]",
        f"col_expectations=[{col_marg}]",
        f"global_expectation={fmt.format(scalar.item())}",
    ]
    if entropy is not None:
        parts.append(f"entropy={fmt.format(entropy)}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Pipeline B (production today): compressed-text via Scotty
# ---------------------------------------------------------------------------


@dataclass
class CompressedTextPipeline:
    """Pipeline B: render Kirk output to text, send to Scotty.

    Today's production path. Used as the apples-to-apples baseline for the new
    embedding-injection path.
    """

    kirk: KirkClient
    scotty: ScottyClient
    system_prompt: str = PROMPT_REGIME_NARRATION_B_V1
    prompt_version: str = "regime_narration_B@v1"

    def run(self, tensors: Sequence[torch.Tensor], **gen_kwargs) -> str:
        """Process a stream of N×N tensors and return the LLM's narration."""
        kos = self.kirk.infer_stream(tensors)
        rendered_windows = [render_kirk_output_as_text(ko) for ko in kos]
        user_msg = "Recent windows (chronological):\n\n" + "\n\n".join(
            f"[t-{len(rendered_windows) - 1 - i}] {block}"
            for i, block in enumerate(rendered_windows)
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        return self.scotty.chat(messages, **gen_kwargs)


# ---------------------------------------------------------------------------
# Pipeline C (new): embedding-injection via HuggingFace
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingInjectionPipeline:
    """Pipeline C: project Kirk outputs into embedding space, inject into LLM.

    The proof point: input-side cost reduction beyond what tokenizing-the-text
    can achieve. Requires direct LLM access (vLLM `prompt_embeds` or HF
    Transformers); cannot run via Ollama / Scotty.
    """

    kirk: KirkClient
    adapter: KirkProjectionAdapter
    llm: EmbeddingInjectionLLM
    instruction_prefix: str = PROMPT_REGIME_NARRATION_C_PREFIX_V1
    instruction_suffix: str = PROMPT_REGIME_NARRATION_C_SUFFIX_V1
    prompt_version: str = "regime_narration_C@v1"

    def run(self, tensors: Sequence[torch.Tensor], max_new_tokens: int = 256) -> str:
        kos = self.kirk.infer_stream(tensors)
        # Project Kirk outputs to LLM embedding space
        kirk_embeds = self.adapter.embed_stream(list(kos)).to(self.llm.config.device)

        # Sandwich: [instruction prefix tokens] + [kirk soft tokens] + [suffix tokens]
        prefix = self.llm.text_to_token_embeds(self.instruction_prefix)
        suffix = self.llm.text_to_token_embeds(self.instruction_suffix)
        full_embeds = torch.cat([prefix, kirk_embeds, suffix], dim=1)

        return self.llm.generate_with_embeds(
            full_embeds, max_new_tokens=max_new_tokens, do_sample=False
        )

    @torch.no_grad()
    def soft_token_count(self, tensors: Sequence[torch.Tensor]) -> int:
        """How many embedding "tokens" the Kirk path emits for this input."""
        return len(tensors) * self.adapter.config.tokens_per_sample
