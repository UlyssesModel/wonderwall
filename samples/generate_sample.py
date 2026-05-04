#!/usr/bin/env python3
"""Generate a V2 OpenInference request payload from random data.

Useful for smoke-testing the KServe endpoint when you don't have a real Uhura
frame handy. Output is a single JSON file matching the schema expected by
scripts/serve_kserve.py.

Usage:
    python samples/generate_sample.py                      # default: T=4, N=32
    python samples/generate_sample.py --T 8 --N 32 --out samples/eight_windows.json
    python samples/generate_sample.py --from-uhura data/uhura/2024-09-03_14-30.npz
"""
from __future__ import annotations

import argparse
import json
import random


def synthetic(T: int, N: int, seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    return [round(rng.gauss(0, 0.003), 6) for _ in range(T * N * N)]


def from_uhura(path: str) -> tuple[list[float], int, int]:
    """Load one or more .npz frames from an Uhura broadcaster directory.

    For a single file: returns one window. Caller can repeat by passing
    --T to slice the resulting array.
    """
    import numpy as np
    arr = np.load(path)
    # Pick the first array key that looks like the tensor
    for key in ("tensor", "frame", "matrix", "data"):
        if key in arr.files:
            tensor = arr[key]
            break
    else:
        raise ValueError(f"no tensor key in {path}; keys: {arr.files}")
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f"expected square 2D tensor, got shape {tensor.shape}")
    if hasattr(tensor, "real"):
        tensor = tensor.real
    return tensor.astype(float).flatten().tolist(), 1, tensor.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=4, help="windows per stream")
    p.add_argument("--N", type=int, default=32, help="window dimension")
    p.add_argument("--out", default="samples/single_window.json")
    p.add_argument("--from-uhura", default=None, help="path to a real Uhura .npz frame")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.from_uhura:
        data, T, N = from_uhura(args.from_uhura)
    else:
        T, N = args.T, args.N
        data = synthetic(T, N, args.seed)

    payload = {
        "id": "sample-generated",
        "inputs": [
            {
                "name": "tensor_windows",
                "shape": [T, N, N],
                "datatype": "FP32",
                "data": data,
            }
        ],
        "outputs": [{"name": "narration"}],
        "parameters": {"max_new_tokens": args.max_new_tokens},
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}: T={T} N={N} cells={T*N*N}")


if __name__ == "__main__":
    main()
