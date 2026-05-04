"""V1 acceptance test: Uhura tensors → Kirk → entropy curve matches reference.

Per the Uhura Confluence page:

  > The reference outputs in *_entropy_price.parq (one entropy scalar per
  > minute, ~0.4–18.9 nats range) are what the existing Quantbot → Kirk
  > pipeline at NY5 produces. Uhura's renderer is correct when matrices
  > it produces, fed to Kirk via Tiberius /compute, generate entropy
  > values close to those reference curves. That's the v1 acceptance test.

This script automates that comparison for the wonderwall stack:

  1. Read a reference *_entropy_price.parq file
  2. For each timestamp in the reference, locate the Uhura tensor frame
     that produced it (via Uhura .npz files or by re-running the renderer)
  3. Run our Kirk client (real KirkPipelineClient when available, stub for
     plumbing tests) to get per-window entropy values
  4. Compute mean absolute deviation, max deviation, and per-percentile
     deviation between produced and reference entropy curves
  5. Emit a pass/fail report — pass when median deviation is < threshold

Default threshold is 0.05 nats (~3% relative error at typical entropy
magnitudes), matching what's reasonable for a v1 acceptance gate.

Usage:

    # Validation against a reference parquet, real Uhura .npz frames:
    python scripts/validate_against_entropy_price.py \\
        --reference data/quantbot/20240903_entropy_price.parq \\
        --uhura-frames-glob "data/uhura/2024-09-03/*.npz" \\
        --report reports/validate_2024-09-03.json

    # Plumbing run with stub Kirk + synthetic frames (sanity-check the harness):
    python scripts/validate_against_entropy_price.py \\
        --plumbing-only --report reports/validate_plumbing.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from wonderwall.interfaces import KirkClient, KirkMode
from wonderwall.kirk_client import (
    KirkPipelineClient,
    KirkSubprocessClient,
    StubKirkClient,
)
from wonderwall.uhura_io import load_uhura_glob


@dataclass
class ValidationResult:
    n_timestamps: int
    median_abs_dev: float
    mean_abs_dev: float
    max_abs_dev: float
    p95_abs_dev: float
    threshold: float
    passed: bool
    deviations: list[float]
    notes: str = ""


def _read_reference_parquet(path: str) -> tuple[list[int], np.ndarray]:
    """Read a *_entropy_price.parq reference. Returns (ts_ns_list, entropies)."""
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pandas is required to read parquet. `pip install pandas pyarrow`"
        ) from e

    df = pd.read_parquet(path)
    # Schema is whatever Quantbot writes — tolerate a few likely shapes
    ts_col = next(
        (c for c in df.columns if c.lower() in ("ts_ns", "timestamp_ns", "timestamp", "ts")),
        None,
    )
    ent_col = next(
        (c for c in df.columns if "entropy" in c.lower()),
        None,
    )
    if ts_col is None or ent_col is None:
        raise ValueError(
            f"reference {path}: couldn't find ts + entropy columns. "
            f"Have: {list(df.columns)}"
        )
    return df[ts_col].astype("int64").tolist(), df[ent_col].to_numpy(dtype=np.float64)


def _entropy_from_kirk(
    kirk: KirkClient, tensor: torch.Tensor, mode: KirkMode = KirkMode.ACTIVE_INFERENCE
) -> float:
    """Run one Kirk forward pass, return the entropy scalar."""
    out = kirk.infer(tensor, mode=mode)
    return float(out.entropy.item())


def validate(
    kirk: KirkClient,
    reference_ts_ns: list[int],
    reference_entropies: np.ndarray,
    tensors_by_ts: dict[int, torch.Tensor],
    threshold: float = 0.05,
    mode: KirkMode = KirkMode.ACTIVE_INFERENCE,
) -> ValidationResult:
    """Compare Kirk-produced entropies against reference at matching timestamps.

    Only timestamps present in BOTH the reference and the tensor map are
    compared. If a timestamp's tensor is missing, it's skipped (and noted).
    """
    deviations: list[float] = []
    skipped = 0
    for ts, ref in zip(reference_ts_ns, reference_entropies):
        tensor = tensors_by_ts.get(ts)
        if tensor is None:
            skipped += 1
            continue
        produced = _entropy_from_kirk(kirk, tensor, mode=mode)
        deviations.append(abs(produced - float(ref)))

    if not deviations:
        return ValidationResult(
            n_timestamps=0,
            median_abs_dev=float("inf"),
            mean_abs_dev=float("inf"),
            max_abs_dev=float("inf"),
            p95_abs_dev=float("inf"),
            threshold=threshold,
            passed=False,
            deviations=[],
            notes=f"no overlap between reference ({len(reference_ts_ns)}) "
                  f"and tensor map ({len(tensors_by_ts)}); {skipped} skipped",
        )

    arr = np.array(deviations)
    return ValidationResult(
        n_timestamps=len(deviations),
        median_abs_dev=float(np.median(arr)),
        mean_abs_dev=float(arr.mean()),
        max_abs_dev=float(arr.max()),
        p95_abs_dev=float(np.percentile(arr, 95)),
        threshold=threshold,
        passed=float(np.median(arr)) < threshold,
        deviations=deviations,
        notes=f"{skipped} timestamps skipped (no matching tensor frame)",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_kirk(backend: str, n: int) -> KirkClient:
    if backend == "pipeline":
        return KirkPipelineClient()
    elif backend == "subprocess":
        return KirkSubprocessClient(n=n)
    elif backend == "stub":
        return StubKirkClient(n=n)
    raise ValueError(f"unknown --kirk-backend: {backend!r}")


def _plumbing_run(threshold: float) -> ValidationResult:
    """Synthetic plumbing run: random reference + matching synthetic tensors.

    The Stub Kirk produces entropy in spec range; deviations from the
    "reference" (which we just generated) will be large but the harness
    code path is exercised end to end. Useful for catching breakage in
    the validation harness itself, not for any quality claim.
    """
    rng = np.random.default_rng(0)
    n_ts = 50
    ts_ns = [int(t) for t in range(0, n_ts * 60_000_000_000, 60_000_000_000)]
    ref_entropies = rng.uniform(2.0, 8.0, size=n_ts)
    tensors = {ts: torch.randn(32, 32) * 0.003 for ts in ts_ns}
    kirk = StubKirkClient(n=32)
    result = validate(kirk, ts_ns, ref_entropies, tensors, threshold=threshold)
    result.notes = "PLUMBING ONLY — synthetic reference + Stub Kirk; deviations meaningless"
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", help="Path to *_entropy_price.parq reference")
    p.add_argument("--uhura-frames-glob", help="Glob of Uhura .npz frames")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Median |dev| nats — pass below this. Default 0.05.")
    p.add_argument("--report", default="reports/validate_entropy_price.json")
    p.add_argument("--kirk-backend", default="pipeline",
                   choices=("pipeline", "subprocess", "stub"))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--plumbing-only", action="store_true",
                   help="Run with synthetic data; sanity-check the harness")
    args = p.parse_args()

    if args.plumbing_only:
        result = _plumbing_run(args.threshold)
    else:
        if not args.reference or not args.uhura_frames_glob:
            print("must pass --reference and --uhura-frames-glob (or --plumbing-only)",
                  file=sys.stderr)
            return 2
        ts_ns, ref = _read_reference_parquet(args.reference)
        frames = load_uhura_glob(args.uhura_frames_glob)
        tensors_by_ts = {f.timestamp_ns: f.tensor for f in frames if f.timestamp_ns is not None}
        kirk = _build_kirk(args.kirk_backend, args.n)
        result = validate(kirk, ts_ns, ref, tensors_by_ts, threshold=args.threshold)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(
            {
                "n_timestamps": result.n_timestamps,
                "median_abs_dev": result.median_abs_dev,
                "mean_abs_dev": result.mean_abs_dev,
                "max_abs_dev": result.max_abs_dev,
                "p95_abs_dev": result.p95_abs_dev,
                "threshold": result.threshold,
                "passed": result.passed,
                "notes": result.notes,
            },
            f,
            indent=2,
        )

    status = "PASS" if result.passed else "FAIL"
    print(f"[validate] {status}")
    print(f"  n_timestamps:  {result.n_timestamps}")
    print(f"  median |dev|:  {result.median_abs_dev:.4f} nats  (threshold {result.threshold:.4f})")
    print(f"  mean   |dev|:  {result.mean_abs_dev:.4f}")
    print(f"  max    |dev|:  {result.max_abs_dev:.4f}")
    print(f"  p95    |dev|:  {result.p95_abs_dev:.4f}")
    if result.notes:
        print(f"  notes: {result.notes}")
    print(f"  report:        {args.report}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
