"""Projection adapter: Kirk outputs → LLM embedding space.

Architectural decision: project Kirk's Array + Vector outputs into a sequence of
(N + 2) embedding vectors at the target LLM's hidden dimension. The Scalar is
NOT injected into the LLM — it is preserved as an anomaly signal for routing.

Token order (deliberate, see DECISIONS.md):
    [array_row_0, array_row_1, ..., array_row_{N-1},
     row_marginal_summary, col_marginal_summary]

Real vs complex:
    Production Kirk per Ted's `kirk_data_description.md` is real-valued
    log-returns; complex128 is the container only. Default config is therefore
    `use_complex=False`. Set to True only when feeding the optional second-channel
    extension (research-tier).

Parameter count: ~5–50M depending on hidden dims and N. Tiny vs the LLM itself.
Trainable in hours on a single H100. Frozen LLM, train this adapter only.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .interfaces import AdapterConfig, KirkOutput


def _to_real(x: torch.Tensor, use_complex: bool) -> torch.Tensor:
    """Convert a possibly-complex tensor to real for downstream nn.Linear.

    If `use_complex=True` and tensor is complex, split (real, imag) and
    concatenate along the last dim — width doubles, info preserved.
    If `use_complex=False`, take the real part (production path: imag is
    uniformly zero anyway, so no information loss).
    """
    if not torch.is_complex(x):
        return x
    if use_complex:
        return torch.cat([x.real, x.imag], dim=-1)
    return x.real


class _MLPBlock(nn.Module):
    """Two-layer MLP used as a per-token projector."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KirkProjectionAdapter(nn.Module):
    """The trainable bridge from Kirk's latent space to the LLM's embedding space.

    Inputs (forward signature):
        array:  (B, N, N), real or complex per `config.use_complex`
        vector: (B, 2N),   ditto

    Output:
        embeds: (B, N + 2, llm_hidden_dim), real-valued, ready to pass as
                `inputs_embeds=` to a HuggingFace causal LM.

    For multi-window (streaming) inputs over T sliding windows, call repeatedly
    and concatenate along seq dim; see `embed_stream` helper below.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config

        self.row_proj = _MLPBlock(
            in_dim=config.row_input_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.llm_hidden_dim,
            dropout=config.dropout,
        )
        self.row_marginal_proj = _MLPBlock(
            in_dim=config.marginal_input_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.llm_hidden_dim,
            dropout=config.dropout,
        )
        self.col_marginal_proj = _MLPBlock(
            in_dim=config.marginal_input_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.llm_hidden_dim,
            dropout=config.dropout,
        )

        # Learned positional embedding across the (N + 2) per-window positions.
        # Shared across all windows in a stream — temporal ordering across windows
        # is left to the LLM's own RoPE/positional scheme.
        self.pos = nn.Parameter(
            torch.zeros(1, config.tokens_per_sample, config.llm_hidden_dim)
        )
        nn.init.normal_(self.pos, std=config.pos_embedding_init_std)

    def forward(
        self,
        array: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        # Shape sanity
        if array.dim() != 3 or array.shape[-2:] != (cfg.n, cfg.n):
            raise ValueError(
                f"array must be (B, N, N) with N={cfg.n}, got {tuple(array.shape)}"
            )
        if vector.dim() != 2 or vector.shape[-1] != 2 * cfg.n:
            raise ValueError(
                f"vector must be (B, 2N) with N={cfg.n}, got {tuple(vector.shape)}"
            )

        a = _to_real(array, cfg.use_complex)        # (B, N, N) or (B, N, 2N)
        v = _to_real(vector, cfg.use_complex)       # (B, 2N) or (B, 4N)

        # Per-row projection (rows are the N "patches" of the array)
        rows = self.row_proj(a)                     # (B, N, H)

        # Marginals: split the Vector into row-marginals (first half) and col-marginals
        n = cfg.n
        if cfg.use_complex:
            # v has shape (B, 4N) when complex was split. Layout: row-real, row-imag,
            # col-real, col-imag — but we just split into halves of length 2N each
            # since we already concatenated (real, imag) inside _to_real.
            half = 2 * n
            row_marg = v[:, :half]
            col_marg = v[:, half:]
        else:
            row_marg = v[:, :n]
            col_marg = v[:, n:]

        row_summary = self.row_marginal_proj(row_marg).unsqueeze(1)   # (B, 1, H)
        col_summary = self.col_marginal_proj(col_marg).unsqueeze(1)   # (B, 1, H)

        embeds = torch.cat([rows, row_summary, col_summary], dim=1)   # (B, N+2, H)
        embeds = embeds + self.pos
        return embeds

    def embed_kirk_output(self, ko: KirkOutput) -> torch.Tensor:
        """Convenience: project a single (un-batched) KirkOutput to (1, N+2, H)."""
        ko.validate()
        return self.forward(ko.array.unsqueeze(0), ko.vector.unsqueeze(0))

    def embed_stream(self, kos: list[KirkOutput]) -> torch.Tensor:
        """Project a stream of T KirkOutputs to a single (1, T * (N+2), H) sequence.

        Output is suitable to be passed as `inputs_embeds` to the LLM directly.
        Within-window order: as `forward`. Across windows: chronological.
        """
        if not kos:
            raise ValueError("empty stream")
        per_window = [self.embed_kirk_output(ko) for ko in kos]
        return torch.cat(per_window, dim=1)

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
