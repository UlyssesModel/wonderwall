"""I/O helpers for consuming Uhura's tensor-frame outputs.

Uhura emits N×N tensors via two paths:

  1. Filesystem broadcaster — npz files + a manifest, one frame per file.
     Schema is whatever Uhura's `broadcaster.py` writes; we read that here.

  2. Kafka broadcaster — `ulysses.tensor.frames.<cadence>` topic, one message
     per frame, body is essentially the same payload as the npz file.

The Uhura Confluence page documents the canonical run output (mag7, N=7,
Ted-spec defaults):
    shape:    (7, 7)
    dtype:    complex128 (imag uniformly zero — real path)
    channels: real = log_return, imag = zero
    norm:     none

We accept that contract and load the real component for downstream use
(D-003: real-valued by default).

The Confluence page mentions the keys used by Uhura's broadcaster but doesn't
fully spell them out. This loader assumes a defensive set and falls back
gracefully — if the schema turns out to differ, the error message points at
exactly what's missing.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch


# Keys we'll attempt to read from each .npz, in priority order. The first
# matching key is used. Multiple aliases are tried because Uhura's broadcaster
# format may evolve and we want to keep this loader robust.
_TENSOR_KEYS = ("tensor", "frame", "matrix", "data")
_TIMESTAMP_KEYS = ("timestamp_ns", "ts_ns", "timestamp", "ts")
_DESIGN_KEYS = ("design", "tensor_design", "schema")
_UNIVERSE_KEYS = ("universe", "tickers", "rows")


@dataclass(frozen=True)
class UhuraFrame:
    """One tensor frame from Uhura.

    `tensor` is always returned as float32 with imag stripped — the production
    path uses imag=0 per Ted's spec, so no information is lost. If you need
    the imag channel for the second-channel research extension, set
    `keep_complex=True` in the loader.
    """

    tensor: torch.Tensor                   # (N, N) float32 (or complex64 if keep_complex)
    timestamp_ns: Optional[int] = None
    design: Optional[str] = None           # e.g. "cross_section_temporal"
    universe: Optional[list[str]] = None
    source_path: Optional[str] = None


def _first_present(npz: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> Optional[str]:
    for k in keys:
        if k in npz.files:
            return k
    return None


def load_uhura_frame(path: str, keep_complex: bool = False) -> UhuraFrame:
    """Load one Uhura .npz frame, defending against schema variation."""
    with np.load(path, allow_pickle=True) as npz:
        tensor_key = _first_present(npz, _TENSOR_KEYS)
        if tensor_key is None:
            raise KeyError(
                f"None of {_TENSOR_KEYS} found in {path}. "
                f"Available keys: {list(npz.files)}"
            )
        arr = npz[tensor_key]

        ts_key = _first_present(npz, _TIMESTAMP_KEYS)
        timestamp_ns = int(npz[ts_key].item()) if ts_key else None

        design_key = _first_present(npz, _DESIGN_KEYS)
        design = str(npz[design_key].item()) if design_key else None

        universe_key = _first_present(npz, _UNIVERSE_KEYS)
        universe = (
            [str(x) for x in npz[universe_key].tolist()] if universe_key else None
        )

    if np.iscomplexobj(arr):
        if keep_complex:
            tensor = torch.from_numpy(arr.astype(np.complex64))
        else:
            tensor = torch.from_numpy(arr.real.astype(np.float32))
    else:
        tensor = torch.from_numpy(arr.astype(np.float32))

    return UhuraFrame(
        tensor=tensor,
        timestamp_ns=timestamp_ns,
        design=design,
        universe=universe,
        source_path=path,
    )


def load_uhura_glob(
    pattern: str, keep_complex: bool = False
) -> list[UhuraFrame]:
    """Load all .npz frames matching a glob pattern, sorted by timestamp.

    Sort order: timestamp_ns ascending if present, else lexicographic on path.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no .npz frames matched {pattern!r}")
    frames = [load_uhura_frame(p, keep_complex=keep_complex) for p in paths]
    if all(f.timestamp_ns is not None for f in frames):
        frames.sort(key=lambda f: f.timestamp_ns)  # type: ignore[arg-type]
    return frames


def stream_windows(
    frames: list[UhuraFrame], windows_per_stream: int
) -> Iterator[list[UhuraFrame]]:
    """Yield non-overlapping consecutive windows of `windows_per_stream` frames.

    Trailing frames that don't make a full window are dropped.
    """
    for i in range(0, len(frames) - windows_per_stream + 1, windows_per_stream):
        yield frames[i : i + windows_per_stream]


def streams_for_distillation(
    pattern: str, windows_per_stream: int = 4, keep_complex: bool = False
) -> list[list[torch.Tensor]]:
    """One-shot helper: glob → frames → streams of tensors ready for distill.py.

    Returns a list of streams, each of which is a list of tensor windows.
    """
    frames = load_uhura_glob(pattern, keep_complex=keep_complex)
    return [
        [f.tensor for f in window]
        for window in stream_windows(frames, windows_per_stream)
    ]
