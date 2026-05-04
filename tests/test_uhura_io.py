"""Tests for the Uhura .npz loader.

Exercises the defensive schema-detection path — Uhura's broadcaster format
may use different key names across versions, so the loader tries several
aliases. These tests write small npz files into tmp_path and confirm the
loader handles each schema variant.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from wonderwall.uhura_io import (
    UhuraFrame,
    load_uhura_frame,
    load_uhura_glob,
    streams_for_distillation,
    stream_windows,
)


def _write_npz(path, **kwargs):
    np.savez(path, **kwargs)


# ---------------------------------------------------------------------------
# Single-frame loading with each schema variant
# ---------------------------------------------------------------------------


def test_loads_with_canonical_tensor_key(tmp_path):
    p = tmp_path / "frame_001.npz"
    _write_npz(p, tensor=np.eye(8, dtype=np.complex128), timestamp_ns=np.int64(123))
    frame = load_uhura_frame(str(p))
    assert frame.tensor.shape == (8, 8)
    assert frame.tensor.dtype == torch.float32  # imag stripped
    assert frame.timestamp_ns == 123


def test_loads_with_alias_keys(tmp_path):
    """Loader should accept frame/matrix/data as alternate tensor key names."""
    for tensor_key in ("frame", "matrix", "data"):
        p = tmp_path / f"frame_{tensor_key}.npz"
        kwargs = {tensor_key: np.eye(4, dtype=np.float32)}
        _write_npz(p, **kwargs)
        frame = load_uhura_frame(str(p))
        assert frame.tensor.shape == (4, 4)


def test_real_path_strips_imag(tmp_path):
    """Production path: complex container with imag=0 → real-valued output."""
    p = tmp_path / "f.npz"
    arr = np.zeros((4, 4), dtype=np.complex128)
    arr.real = np.eye(4)
    # imag stays zero per Ted's spec
    _write_npz(p, tensor=arr)
    frame = load_uhura_frame(str(p), keep_complex=False)
    assert frame.tensor.dtype == torch.float32
    assert torch.equal(frame.tensor, torch.eye(4, dtype=torch.float32))


def test_keep_complex_preserves_imag(tmp_path):
    """Research path: keep_complex=True preserves the imag channel."""
    p = tmp_path / "f.npz"
    arr = np.zeros((4, 4), dtype=np.complex128)
    arr.real = np.eye(4)
    arr.imag = np.eye(4) * 0.5
    _write_npz(p, tensor=arr)
    frame = load_uhura_frame(str(p), keep_complex=True)
    assert torch.is_complex(frame.tensor)
    assert torch.any(frame.tensor.imag != 0)


def test_universe_metadata_loaded(tmp_path):
    p = tmp_path / "f.npz"
    universe = np.array(["AAPL", "MSFT", "GOOGL"], dtype=object)
    _write_npz(p, tensor=np.eye(3, dtype=np.float32), universe=universe)
    frame = load_uhura_frame(str(p))
    assert frame.universe == ["AAPL", "MSFT", "GOOGL"]


def test_design_metadata_loaded(tmp_path):
    p = tmp_path / "f.npz"
    _write_npz(
        p,
        tensor=np.eye(4, dtype=np.float32),
        design=np.array("cross_section_temporal"),
    )
    frame = load_uhura_frame(str(p))
    assert frame.design == "cross_section_temporal"


def test_missing_tensor_key_raises(tmp_path):
    p = tmp_path / "bad.npz"
    _write_npz(p, irrelevant_key=np.array([1, 2, 3]))
    with pytest.raises(KeyError, match="not.*found"):
        load_uhura_frame(str(p))


# ---------------------------------------------------------------------------
# Glob loading + sort order
# ---------------------------------------------------------------------------


def test_glob_loads_multiple_frames_in_timestamp_order(tmp_path):
    # Write three frames out of timestamp order to ensure the sort works
    for filename, ts in [("c.npz", 300), ("a.npz", 100), ("b.npz", 200)]:
        p = tmp_path / filename
        _write_npz(p, tensor=np.eye(4, dtype=np.float32), timestamp_ns=np.int64(ts))

    frames = load_uhura_glob(str(tmp_path / "*.npz"))
    assert [f.timestamp_ns for f in frames] == [100, 200, 300]


def test_glob_falls_back_to_lex_order_when_timestamps_missing(tmp_path):
    """Without timestamps, sort should be lexicographic on path."""
    for filename in ("z.npz", "a.npz", "m.npz"):
        p = tmp_path / filename
        _write_npz(p, tensor=np.eye(2, dtype=np.float32))
    frames = load_uhura_glob(str(tmp_path / "*.npz"))
    assert [f.source_path.endswith(name) for f, name in zip(frames, ("a.npz", "m.npz", "z.npz"))]


def test_glob_with_no_matches_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .npz frames"):
        load_uhura_glob(str(tmp_path / "*.npz"))


# ---------------------------------------------------------------------------
# Stream windowing
# ---------------------------------------------------------------------------


def test_stream_windows_drops_trailing_partial(tmp_path):
    frames = [
        UhuraFrame(tensor=torch.eye(2), timestamp_ns=i)
        for i in range(7)
    ]
    windows = list(stream_windows(frames, windows_per_stream=3))
    # 7 frames @ 3 per stream = 2 complete streams (6 frames), 1 trailing dropped
    assert len(windows) == 2
    assert all(len(w) == 3 for w in windows)


def test_streams_for_distillation_end_to_end(tmp_path):
    for i in range(8):
        p = tmp_path / f"f_{i:03d}.npz"
        _write_npz(p, tensor=np.full((8, 8), float(i), dtype=np.float32),
                   timestamp_ns=np.int64(i * 1000))
    streams = streams_for_distillation(
        str(tmp_path / "*.npz"), windows_per_stream=4
    )
    assert len(streams) == 2  # 8 frames / 4-per-stream
    assert all(len(s) == 4 for s in streams)
    assert all(t.shape == (8, 8) for s in streams for t in s)
