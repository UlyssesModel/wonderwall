"""Calibration sweep harness — mirrors uhura's `sweep` command.

Iterates over a configuration grid (n × hidden_dim × use_complex × pipeline)
on a held-out eval set, produces a leaderboard CSV with the same columns
Uhura's sweep emits so cross-repo numbers compose directly.

Output shape (per row):
    rank, n, hidden_dim, use_complex, pipeline, n_eval_items,
    mean_input_tokens, mean_output_tokens, mean_e2e_ms,
    mean_rouge_l, mean_cost_usd, cost_per_year_usd

The cost-per-year column lets the leaderboard map directly onto Uhura's
"$/trading year" framing (50 steps/run × ~250 trading days × runs/day
assumption baked in via --runs-per-day).

Two run modes:
  --eval        run real evals and compute fresh numbers
  --dry-run     just iterate the grid and print what would run

Usage:
    python -m eval.sweep \\
        --distilled data/distilled_test.pt \\
        --adapter checkpoints/adapter.pt \\
        --grid configs/sweep_default.yaml \\
        --out reports/sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import torch
import yaml

from wonderwall.adapter import KirkProjectionAdapter
from wonderwall.distill import load_distilled
from wonderwall.injection import (
    EmbeddingInjectionLLM,
    LLMConfig,
    ScottyClient,
    ScottyConfig,
)
from wonderwall.interfaces import AdapterConfig
from wonderwall.kirk_client import StubKirkClient

from .harness import EvalHarness, HarnessConfig
from .metrics import EvalRecord, compute_metrics


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------


@dataclass
class SweepGrid:
    n_values: list[int] = field(default_factory=lambda: [16, 32])
    hidden_dim_values: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    use_complex_values: list[bool] = field(default_factory=lambda: [False])
    pipelines: list[str] = field(default_factory=lambda: ["A_tokenized", "B_compressed_text", "C_embedding_injection"])

    def __iter__(self) -> Iterable[dict[str, Any]]:
        for n, h, c, p in itertools.product(
            self.n_values, self.hidden_dim_values, self.use_complex_values, self.pipelines
        ):
            yield {"n": n, "hidden_dim": h, "use_complex": c, "pipeline": p}

    def __len__(self) -> int:
        return (
            len(self.n_values)
            * len(self.hidden_dim_values)
            * len(self.use_complex_values)
            * len(self.pipelines)
        )


def load_grid(path: Optional[str]) -> SweepGrid:
    if path is None:
        return SweepGrid()
    with open(path) as f:
        data = yaml.safe_load(f)
    return SweepGrid(
        n_values=data.get("n_values", SweepGrid().n_values),
        hidden_dim_values=data.get("hidden_dim_values", SweepGrid().hidden_dim_values),
        use_complex_values=data.get("use_complex_values", SweepGrid().use_complex_values),
        pipelines=data.get("pipelines", SweepGrid().pipelines),
    )


# ---------------------------------------------------------------------------
# Single-config eval
# ---------------------------------------------------------------------------


def _eval_config(
    cfg: dict[str, Any],
    items_path: str,
    adapter_ckpt_path: Optional[str],
    llm_cfg: Optional[LLMConfig],
    scotty: Optional[ScottyClient],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run one (n, hidden_dim, use_complex, pipeline) cell of the grid."""

    items = load_distilled(items_path)

    # Pipelines we actually have to run depend on `cfg["pipeline"]`
    run_a = cfg["pipeline"] == "A_tokenized"
    run_b = cfg["pipeline"] == "B_compressed_text"
    run_c = cfg["pipeline"] == "C_embedding_injection"

    # Stub Kirk for the sweep — once the real Kirk wheel is wired, swap to
    # KirkPipelineClient. The grid is what matters; the data source stays.
    kirk = StubKirkClient(n=cfg["n"], use_complex=cfg["use_complex"])

    adapter = None
    llm = None
    if run_c:
        if adapter_ckpt_path is None or llm_cfg is None:
            raise ValueError("Pipeline C requires --adapter and an --llm-config")
        adapter_cfg = AdapterConfig(
            n=cfg["n"],
            llm_hidden_dim=llm_cfg.hidden_dim,
            hidden_dim=cfg["hidden_dim"],
            use_complex=cfg["use_complex"],
        )
        adapter = KirkProjectionAdapter(adapter_cfg)
        ckpt = torch.load(adapter_ckpt_path, weights_only=False, map_location="cpu")
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        llm = EmbeddingInjectionLLM(llm_cfg)
        adapter = adapter.to(llm.config.device).eval()

    harness = EvalHarness(
        kirk=kirk,
        adapter=adapter,
        llm=llm,
        scotty=scotty,
        config=HarnessConfig(
            run_pipeline_a=run_a,
            run_pipeline_b=run_b,
            run_pipeline_c=run_c,
            max_new_tokens=max_new_tokens,
        ),
    )
    records = harness.evaluate_set(items)
    summary = compute_metrics(records)
    pipe_summary = summary.get(cfg["pipeline"], {})
    return {
        **cfg,
        "n_eval_items": len(items),
        "mean_input_tokens": pipe_summary.get("mean_input_tokens", 0.0),
        "mean_output_tokens": pipe_summary.get("mean_output_tokens", 0.0),
        "mean_e2e_ms": pipe_summary.get("mean_e2e_ms", 0.0),
        "mean_rouge_l": pipe_summary.get("mean_rouge_l", 0.0),
        "mean_cost_usd": pipe_summary.get("mean_cost_usd", 0.0),
    }


# ---------------------------------------------------------------------------
# Cost-per-year extrapolation (matches uhura's framing)
# ---------------------------------------------------------------------------


def cost_per_year(mean_cost_per_run: float, runs_per_day: float, trading_days: int = 252) -> float:
    """Extrapolate per-run cost into per-trading-year cost.

    Uhura's leaderboard uses 50-step agent runs × trading day × tickers/agents.
    We just keep it simple: cost_per_run × runs_per_day × 252.
    """
    return mean_cost_per_run * runs_per_day * trading_days


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_leaderboard(rows: list[dict[str, Any]], out_csv: str) -> None:
    """Write rows sorted by cost-per-year ascending (cheapest at the top)."""
    sorted_rows = sorted(
        rows, key=lambda r: r.get("cost_per_year_usd", float("inf"))
    )
    for i, r in enumerate(sorted_rows, 1):
        r["rank"] = i

    fieldnames = [
        "rank", "n", "hidden_dim", "use_complex", "pipeline", "n_eval_items",
        "mean_input_tokens", "mean_output_tokens", "mean_e2e_ms",
        "mean_rouge_l", "mean_cost_usd", "cost_per_year_usd",
    ]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def print_leaderboard_top(rows: list[dict[str, Any]], top_n: int = 10) -> None:
    sorted_rows = sorted(rows, key=lambda r: r.get("cost_per_year_usd", float("inf")))
    print()
    print(f"{'rank':>4}  {'pipeline':<24} {'n':>4} {'hidden':>6} {'cmplx':>5} "
          f"{'tokens':>7} {'ms':>6} {'rouge':>5} {'$/yr':>10}")
    for i, r in enumerate(sorted_rows[:top_n], 1):
        print(
            f"{i:>4}  {r['pipeline']:<24} {r['n']:>4} {r['hidden_dim']:>6} "
            f"{str(r['use_complex'])[:5]:>5} {r['mean_input_tokens']:>7.0f} "
            f"{r['mean_e2e_ms']:>6.0f} {r['mean_rouge_l']:>5.2f} "
            f"{r.get('cost_per_year_usd', 0):>10.0f}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distilled", required=True, help="Path to distilled .pt items")
    p.add_argument("--adapter", default=None, help="Adapter checkpoint (required for Pipeline C)")
    p.add_argument("--grid", default=None, help="Sweep grid YAML")
    p.add_argument("--out", default="reports/sweep.csv", help="Output CSV path")
    p.add_argument("--llm-config", default="configs/llm_gemma4.yaml")
    p.add_argument("--scotty-base-url", default="http://127.0.0.1:11434")
    p.add_argument("--scotty-model", default="gemma4:31b")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--runs-per-day", type=float, default=8.0,
                   help="For cost-per-year extrapolation")
    p.add_argument("--top", type=int, default=10, help="How many rows to print to stdout")
    p.add_argument("--dry-run", action="store_true",
                   help="Just enumerate the grid and exit")
    args = p.parse_args()

    grid = load_grid(args.grid)
    print(f"[sweep] grid: {len(grid)} cells")

    if args.dry_run:
        for cell in grid:
            print(f"[sweep] would run: {cell}")
        return 0

    llm_cfg = None
    with open(args.llm_config) as f:
        llm_cfg = LLMConfig(**yaml.safe_load(f))

    scotty = ScottyClient(
        ScottyConfig(base_url=args.scotty_base_url, model=args.scotty_model)
    )

    rows: list[dict[str, Any]] = []
    t_grid_start = time.perf_counter()
    for i, cell in enumerate(grid, 1):
        t0 = time.perf_counter()
        try:
            row = _eval_config(
                cfg=cell,
                items_path=args.distilled,
                adapter_ckpt_path=args.adapter,
                llm_cfg=llm_cfg,
                scotty=scotty,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"[sweep] cell {i}/{len(grid)} FAILED: {e}")
            continue
        row["cost_per_year_usd"] = cost_per_year(
            row["mean_cost_usd"], args.runs_per_day
        )
        elapsed = time.perf_counter() - t0
        print(
            f"[sweep] {i}/{len(grid)} ({elapsed:.1f}s)  "
            f"{row['pipeline']} n={row['n']} h={row['hidden_dim']} "
            f"complex={row['use_complex']}  $/yr={row['cost_per_year_usd']:.0f}"
        )
        rows.append(row)

    write_leaderboard(rows, args.out)
    print_leaderboard_top(rows, top_n=args.top)
    print(f"\n[sweep] wrote {len(rows)} rows to {args.out}")
    print(f"[sweep] total grid time: {time.perf_counter() - t_grid_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
