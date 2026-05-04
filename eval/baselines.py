"""Pipeline A — tokenized-raw-data baseline.

Renders the raw N×N tensor stream as numerical text and sends to the LLM.
This is the worst case in token count and the apples-to-apples reference for
both Pipeline B (compressed text) and Pipeline C (embedding injection).

Mirrors Uhura's `text_baseline` renderer so token counts are directly
comparable across the two repos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from wonderwall.injection import ScottyClient
from wonderwall.prompts import PROMPT_REGIME_NARRATION_A_V1


def render_raw_stream_as_text(tensors: Sequence[torch.Tensor], decimals: int = 5) -> str:
    """Render raw N×N tensors as a verbose text block — this is Pipeline A's input."""
    fmt = f"{{:.{decimals}f}}"
    rendered = []
    for i, t in enumerate(tensors):
        if torch.is_complex(t):
            t = t.real
        n = t.shape[0]
        rows = []
        for r in range(n):
            row_str = ", ".join(fmt.format(v) for v in t[r].tolist())
            rows.append(f"  row {r}: [{row_str}]")
        rendered.append(f"[t-{len(tensors) - 1 - i}] window ({n}×{n}):\n" + "\n".join(rows))
    return "\n\n".join(rendered)


@dataclass
class RawTextPipeline:
    """Pipeline A: send the raw tensor stream as text. Worst-case token count."""

    scotty: ScottyClient
    system_prompt: str = PROMPT_REGIME_NARRATION_A_V1
    prompt_version: str = "regime_narration_A@v1"

    def run(self, tensors: Sequence[torch.Tensor], **gen_kwargs) -> str:
        user_msg = (
            "Recent windows (chronological), raw log-returns:\n\n"
            + render_raw_stream_as_text(tensors)
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        return self.scotty.chat(messages, **gen_kwargs)
