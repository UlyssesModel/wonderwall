"""Label market windows with regime ground truth using technical indicators.

Given a stream of N×N tensor windows of log-returns (Uhura's
cross_section_temporal output), compute a regime label per window using
simple, transparent rules:

  - **calm**:        realized vol below `low_vol_threshold` AND drawdown < 1%
  - **trending**:    realized vol moderate, |mean return| > 0.1% per period
  - **volatile**:    realized vol above `high_vol_threshold`
  - **crash**:       max drawdown over window > `crash_drawdown_threshold`

Crash takes precedence over volatile, which takes precedence over trending,
which takes precedence over calm.

These are deliberately simple rules — they're meant to provide *gold regime
labels* for two consumers:

  1. fit_supervised on the GaussianHMM (so the HMM has actual labels to fit
     emission means against, not just the default uniform priors)
  2. EvalRecord.regime_correct comparison for narrations and HMM predictions
     (so the conference deck can say "Pipeline C narration matches the
     technical-indicator label X% of the time")

Production-grade regime labeling would use a more nuanced scheme (Markov
switching models, breakpoint detection, etc.) — but this version is honest,
auditable, and good enough to seed evaluation.

Usage:

    # Label streams from Uhura .npz frames + write to a labels JSON
    python scripts/label_regimes.py \\
        --uhura-frames-glob "data/uhura/*.npz" \\
        --windows-per-stream 4 \\
        --out data/regime_labels.json

    # Apply labels to a distillation .pt file (sets DistillationItem.metadata)
    python scripts/label_regimes.py \\
        --distilled data/distilled_train.pt \\
        --out data/distilled_train_labeled.pt
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from wonderwall.distill import DistillationItem, load_distilled, save_distilled
from wonderwall.uhura_io import streams_for_distillation


REGIME_NAMES = ["calm", "trending", "volatile", "crash"]


@dataclass(frozen=True)
class RegimeThresholds:
    """All thresholds in *log-return units per minute* (log-return scale).

    Defaults are calibrated to roughly match equity minute-bar returns:
      - 1% in log-return ≈ 0.01
      - Realized vol of 0.5% / minute is moderate; 1% / minute is high
      - 5% drawdown over a window flags as crash
    """

    low_vol: float = 0.005       # σ below this → calm
    high_vol: float = 0.01       # σ above this → volatile
    trend_mean: float = 0.001    # |mean ret| above this → trending
    crash_drawdown: float = 0.05 # max drawdown above this → crash


def _stream_to_returns(stream: Sequence[torch.Tensor]) -> np.ndarray:
    """Concatenate a stream's windows into a single returns matrix.

    Each window is N×N (rows=tickers, cols=time). Concatenating along the
    time axis gives one N × (N×T) matrix with chronological order preserved.
    """
    arrays = [w.detach().cpu().float().numpy() for w in stream]
    return np.concatenate(arrays, axis=1)


def _portfolio_returns(returns: np.ndarray) -> np.ndarray:
    """Equal-weighted portfolio returns from a (n_tickers, n_periods) matrix.

    For regime classification we collapse the cross-section into one series.
    """
    return returns.mean(axis=0)


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown over a return series.

    Builds the cumulative log-return curve, finds the worst peak-to-trough.
    Returns absolute drawdown in log-return units (positive number).
    """
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    drawdown = peak - cum
    return float(drawdown.max() if drawdown.size else 0.0)


def label_stream(
    stream: Sequence[torch.Tensor],
    thresholds: RegimeThresholds = RegimeThresholds(),
) -> str:
    """Classify a stream of N×N windows into one of the regime labels."""
    returns = _stream_to_returns(stream)
    pf = _portfolio_returns(returns)

    realized_vol = float(pf.std())
    abs_mean = float(abs(pf.mean()))
    max_dd = _max_drawdown(pf)

    # Crash dominates; otherwise check volatile, trending, calm in order
    if max_dd > thresholds.crash_drawdown:
        return "crash"
    if realized_vol > thresholds.high_vol:
        return "volatile"
    if realized_vol > thresholds.low_vol and abs_mean > thresholds.trend_mean:
        return "trending"
    return "calm"


def label_streams(
    streams: Sequence[Sequence[torch.Tensor]],
    thresholds: RegimeThresholds = RegimeThresholds(),
) -> list[str]:
    return [label_stream(s, thresholds) for s in streams]


# ---------------------------------------------------------------------------
# CLI: two modes
# ---------------------------------------------------------------------------


def _label_uhura_glob(args, thresholds: RegimeThresholds) -> int:
    streams = streams_for_distillation(
        args.uhura_frames_glob, windows_per_stream=args.windows_per_stream
    )
    labels = label_streams(streams, thresholds)
    out = {
        "n_streams": len(streams),
        "windows_per_stream": args.windows_per_stream,
        "thresholds": {
            "low_vol": thresholds.low_vol,
            "high_vol": thresholds.high_vol,
            "trend_mean": thresholds.trend_mean,
            "crash_drawdown": thresholds.crash_drawdown,
        },
        "labels": labels,
        "label_counts": {name: labels.count(name) for name in REGIME_NAMES},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[label] wrote {len(labels)} labels → {args.out}")
    print(f"[label] distribution: {out['label_counts']}")
    return 0


def _label_distilled(args, thresholds: RegimeThresholds) -> int:
    items = load_distilled(args.distilled)
    counts = {name: 0 for name in REGIME_NAMES}
    for item in items:
        gold = label_stream(item.tensors, thresholds)
        item.metadata = {**(item.metadata or {}), "gold_regime": gold}
        counts[gold] += 1
    save_distilled(items, args.out)
    print(f"[label] labeled {len(items)} items → {args.out}")
    print(f"[label] distribution: {counts}")
    return 0


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--uhura-frames-glob",
                     help="Glob of Uhura .npz frames; emits a labels JSON")
    src.add_argument("--distilled",
                     help="Distilled .pt file; tags each item with gold_regime")
    p.add_argument("--out", required=True)
    p.add_argument("--windows-per-stream", type=int, default=4)
    p.add_argument("--low-vol", type=float, default=0.005)
    p.add_argument("--high-vol", type=float, default=0.01)
    p.add_argument("--trend-mean", type=float, default=0.001)
    p.add_argument("--crash-drawdown", type=float, default=0.05)
    args = p.parse_args()

    thresholds = RegimeThresholds(
        low_vol=args.low_vol,
        high_vol=args.high_vol,
        trend_mean=args.trend_mean,
        crash_drawdown=args.crash_drawdown,
    )

    if args.uhura_frames_glob:
        return _label_uhura_glob(args, thresholds)
    return _label_distilled(args, thresholds)


if __name__ == "__main__":
    raise SystemExit(main())
