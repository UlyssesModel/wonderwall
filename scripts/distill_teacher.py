"""Generate gold narrations from a teacher LLM for adapter training.

Reads input tensor streams (synthetic, or pulled from Uhura .npz frames),
calls a teacher LLM on the verbose text rendering, and saves the resulting
(tensors, target_text) pairs as a single .pt file that the trainer consumes.

Examples:

    # Stub run — synthetic 32×32 streams to validate plumbing:
    python scripts/distill_teacher.py \\
        --output data/distilled_dev.pt \\
        --num-streams 32 --windows-per-stream 4 --n 32 \\
        --use-stub-input \\
        --teacher-base-url http://127.0.0.1:11434 \\
        --teacher-model gemma4:31b

    # Real Uhura frames + frontier teacher:
    python scripts/distill_teacher.py \\
        --output data/distilled_train.pt \\
        --uhura-frames-glob "data/uhura/*.npz" \\
        --windows-per-stream 4 \\
        --teacher-base-url https://api.anthropic.com/v1 \\
        --teacher-model claude-opus-4-6
"""
from __future__ import annotations

import argparse
import os

import torch

from wonderwall.distill import (
    label_with_teacher,
    save_distilled,
)
from wonderwall.injection import ScottyClient, ScottyConfig
from wonderwall.uhura_io import streams_for_distillation


def synthetic_streams(num_streams: int, windows_per_stream: int, n: int) -> list[list[torch.Tensor]]:
    """Synthetic input streams of N×N tensors at log-return scale."""
    streams: list[list[torch.Tensor]] = []
    for s in range(num_streams):
        torch.manual_seed(1000 + s)
        stream = [torch.randn(n, n, dtype=torch.float32) * 0.003 for _ in range(windows_per_stream)]
        streams.append(stream)
    return streams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=32,
                   help="Tensor dimension. 32 matches kirk-pipeline Layer 2.")
    p.add_argument("--windows-per-stream", type=int, default=4)
    p.add_argument("--num-streams", type=int, default=64,
                   help="Used only with --use-stub-input")
    p.add_argument("--use-stub-input", action="store_true")
    p.add_argument("--uhura-frames-glob", default=None,
                   help="Glob pattern matching Uhura .npz frames")
    p.add_argument("--teacher-base-url", default="http://127.0.0.1:11434")
    p.add_argument("--teacher-model", default="gemma4:31b")
    args = p.parse_args()

    if args.use_stub_input:
        streams = synthetic_streams(args.num_streams, args.windows_per_stream, args.n)
    elif args.uhura_frames_glob:
        streams = streams_for_distillation(
            args.uhura_frames_glob, windows_per_stream=args.windows_per_stream
        )
    else:
        raise SystemExit("Must pass either --use-stub-input or --uhura-frames-glob")

    print(f"[distill] {len(streams)} input streams")
    teacher = ScottyClient(ScottyConfig(base_url=args.teacher_base_url, model=args.teacher_model))
    items = label_with_teacher(teacher, streams, show_progress=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_distilled(items, args.output)
    print(f"[distill] wrote {len(items)} items to {args.output}")


if __name__ == "__main__":
    main()
