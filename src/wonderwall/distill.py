"""Teacher-LLM distillation: generate gold narrations for adapter training.

The training signal for the projection adapter is supervised: for each input
window-stream, we need a target text narration. We don't have human labels at
scale, so we use a teacher LLM (Claude / GPT-4 / a strong open model) on the
*tokenized-raw-data* version of the same input to generate gold narrations.

The student (adapter + frozen LLM) then learns to reproduce those narrations
while consuming the much shorter Kirk-encoded embedding sequence instead of
the raw tokens.

This is the standard LLaVA-style training recipe applied to time-series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .injection import ScottyClient
from .interfaces import KirkClient
from .pipeline import render_kirk_output_as_text
from .prompts import PROMPT_TEACHER_NARRATION_V1


@dataclass
class TeacherConfig:
    """Configuration for the teacher (used to label training data).

    Default targets a frontier-class model via an OpenAI-compatible endpoint.
    The teacher is bigger than the student LLM by design — distillation only
    works if the teacher knows things the student needs to learn.
    """

    base_url: str = "https://api.anthropic.com/v1"
    model: str = "claude-opus-4-6"
    timeout_s: float = 180.0
    api_key_env: str = "ANTHROPIC_API_KEY"


def make_teacher_prompt(raw_tensors: Sequence[torch.Tensor]) -> list[dict]:
    """Render a stream of raw tensors as a verbose text block for the teacher.

    The teacher gets the *uncompressed* representation — every cell of every
    window — so it has full information when generating the gold narration.
    The student will then learn to produce the same narration from the
    Kirk-compressed representation.
    """
    rendered = []
    for i, t in enumerate(raw_tensors):
        if torch.is_complex(t):
            t = t.real
        n = t.shape[0]
        rows = []
        for r in range(n):
            row_str = ", ".join(f"{v:.5f}" for v in t[r].tolist())
            rows.append(f"  row {r}: [{row_str}]")
        rendered.append(f"[t-{len(raw_tensors) - 1 - i}] window ({n}×{n}):\n" + "\n".join(rows))

    user_msg = PROMPT_TEACHER_NARRATION_V1 + "\n\n" + "\n\n".join(rendered)
    return [{"role": "user", "content": user_msg}]


@dataclass
class DistillationItem:
    """One training example: stream of input tensors plus gold narration.

    `metadata` is optional and free-form. Common keys:
        gold_regime:    str — ground-truth regime label for the stream
                              (one of the HMM state_names). Used by the
                              eval harness to populate regime_correct.
        symbol:         str — primary ticker / underlying for this stream.
        window_size:    int — N for sanity checks downstream.
    """

    tensors: list[torch.Tensor]
    target_text: str
    timestamp_ns: Optional[int] = None
    metadata: Optional[dict] = None


def label_with_teacher(
    teacher: ScottyClient,
    streams: Sequence[Sequence[torch.Tensor]],
    show_progress: bool = True,
) -> list[DistillationItem]:
    """Generate gold narrations for a list of input streams.

    Args:
        teacher: an OpenAI-compatible client pointed at the teacher model.
                 ScottyClient works for any OpenAI-compatible endpoint, not
                 just Scotty itself.
        streams: list of tensor sequences. Each sequence is one training example.
        show_progress: print a progress line per stream.

    Returns: list of DistillationItem.
    """
    items: list[DistillationItem] = []
    for i, stream in enumerate(streams):
        prompt = make_teacher_prompt(stream)
        narration = teacher.chat(prompt, temperature=0.0)
        items.append(DistillationItem(tensors=list(stream), target_text=narration))
        if show_progress:
            print(f"[distill] {i + 1}/{len(streams)} done — {len(narration)} chars")
    return items


def save_distilled(items: Sequence[DistillationItem], path: str) -> None:
    """Persist distillation set as a single .pt file.

    Tensors stored as torch.save objects; small datasets only. For larger
    sets switch to one-file-per-example or webdataset.
    """
    payload = [
        {
            "tensors": [t.detach().cpu() for t in item.tensors],
            "target_text": item.target_text,
            "timestamp_ns": item.timestamp_ns,
            "metadata": item.metadata,
        }
        for item in items
    ]
    torch.save(payload, path)


def load_distilled(path: str) -> list[DistillationItem]:
    payload = torch.load(path, weights_only=False)
    return [
        DistillationItem(
            tensors=list(item["tensors"]),
            target_text=item["target_text"],
            timestamp_ns=item.get("timestamp_ns"),
            metadata=item.get("metadata"),
        )
        for item in payload
    ]
