"""Tests for the calibration sweep harness.

Covers:
  - SweepGrid iteration produces every (n × hidden_dim × use_complex × pipeline) cell
  - Grid loading from YAML
  - cost_per_year extrapolation math
  - write_leaderboard CSV shape + sort order (cheapest first)
  - print_leaderboard_top doesn't crash on empty rows
"""
from __future__ import annotations

import csv
import os

import pytest

from eval.sweep import (
    SweepGrid,
    cost_per_year,
    load_grid,
    print_leaderboard_top,
    write_leaderboard,
)


# ---------------------------------------------------------------------------
# Grid iteration
# ---------------------------------------------------------------------------


def test_grid_default_has_expected_size():
    grid = SweepGrid()
    # 2 n × 3 hidden_dim × 1 complex × 3 pipelines = 18
    assert len(grid) == 2 * 3 * 1 * 3


def test_grid_iter_produces_all_cells():
    grid = SweepGrid(
        n_values=[16, 32],
        hidden_dim_values=[512, 1024],
        use_complex_values=[False],
        pipelines=["A_tokenized", "B_compressed_text"],
    )
    cells = list(grid)
    assert len(cells) == len(grid)
    # Every cell has the four expected keys
    for cell in cells:
        assert set(cell.keys()) == {"n", "hidden_dim", "use_complex", "pipeline"}


def test_grid_iter_distinct_cells():
    grid = SweepGrid(
        n_values=[16, 32],
        hidden_dim_values=[512],
        use_complex_values=[False],
        pipelines=["A_tokenized"],
    )
    cells = list(grid)
    assert len(cells) == 2
    # The two cells must differ (in n)
    assert cells[0]["n"] != cells[1]["n"]


def test_load_grid_from_yaml(tmp_path):
    p = tmp_path / "grid.yaml"
    p.write_text(
        "n_values: [16, 32, 48]\n"
        "hidden_dim_values: [256]\n"
        "use_complex_values: [false, true]\n"
        "pipelines: [A_tokenized, C_embedding_injection]\n"
    )
    grid = load_grid(str(p))
    assert grid.n_values == [16, 32, 48]
    assert grid.hidden_dim_values == [256]
    assert grid.use_complex_values == [False, True]
    assert len(grid) == 3 * 1 * 2 * 2


def test_load_grid_returns_default_when_path_is_none():
    grid = load_grid(None)
    assert isinstance(grid, SweepGrid)
    assert grid.n_values == [16, 32]


# ---------------------------------------------------------------------------
# Cost-per-year extrapolation
# ---------------------------------------------------------------------------


def test_cost_per_year_default_trading_days():
    """Default 252 trading days; runs/day=8 → 2016 runs/year."""
    annual = cost_per_year(mean_cost_per_run=0.005, runs_per_day=8.0)
    assert annual == pytest.approx(0.005 * 8.0 * 252)


def test_cost_per_year_custom_trading_days():
    annual = cost_per_year(0.01, 4.0, trading_days=100)
    assert annual == pytest.approx(0.01 * 4.0 * 100)


def test_cost_per_year_zero():
    assert cost_per_year(0.0, 8.0) == 0.0


# ---------------------------------------------------------------------------
# Leaderboard writing
# ---------------------------------------------------------------------------


def _make_row(pipeline: str, cost_yr: float, n: int = 32) -> dict:
    return {
        "pipeline": pipeline,
        "n": n,
        "hidden_dim": 1024,
        "use_complex": False,
        "n_eval_items": 50,
        "mean_input_tokens": 200.0,
        "mean_output_tokens": 50.0,
        "mean_e2e_ms": 250.0,
        "mean_rouge_l": 0.5,
        "mean_cost_usd": cost_yr / (252 * 8),
        "cost_per_year_usd": cost_yr,
    }


def test_write_leaderboard_sorts_cheapest_first(tmp_path):
    rows = [
        _make_row("A_tokenized", cost_yr=21130.0),
        _make_row("B_compressed_text", cost_yr=529.0),
        _make_row("C_embedding_injection", cost_yr=120.0),
    ]
    out = str(tmp_path / "lb.csv")
    write_leaderboard(rows, out)

    with open(out) as f:
        reader = csv.DictReader(f)
        rows_out = list(reader)

    assert len(rows_out) == 3
    # Cheapest first
    assert rows_out[0]["pipeline"] == "C_embedding_injection"
    assert rows_out[1]["pipeline"] == "B_compressed_text"
    assert rows_out[2]["pipeline"] == "A_tokenized"
    # Rank assigned 1..N
    assert [r["rank"] for r in rows_out] == ["1", "2", "3"]


def test_write_leaderboard_creates_dir(tmp_path):
    """Writing to a path with a missing parent dir should still work."""
    out = str(tmp_path / "subdir" / "lb.csv")
    write_leaderboard([_make_row("X", 100.0)], out)
    assert os.path.exists(out)


def test_write_leaderboard_handles_missing_cost_field(tmp_path):
    """Rows missing cost_per_year_usd shouldn't crash the sort."""
    rows = [
        _make_row("good", cost_yr=10.0),
        {"pipeline": "missing_cost", "n": 32},  # no cost_per_year_usd
    ]
    out = str(tmp_path / "lb.csv")
    write_leaderboard(rows, out)
    # Just confirms it didn't raise
    assert os.path.exists(out)


def test_print_leaderboard_top_handles_empty(capsys):
    print_leaderboard_top([], top_n=10)
    out = capsys.readouterr().out
    # No exception is the success criterion; output is fine to be empty/header
    assert out is not None


def test_print_leaderboard_top_limits_to_top_n(capsys):
    rows = [_make_row(f"pipe{i}", cost_yr=float(i)) for i in range(5)]
    print_leaderboard_top(rows, top_n=2)
    out = capsys.readouterr().out
    # Only the cheapest 2 pipelines should appear in the printed leaderboard
    assert "pipe0" in out
    assert "pipe1" in out
    assert "pipe4" not in out
