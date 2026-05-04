"""Render a self-contained HTML report from sweep + eval outputs.

Reads:
  - reports/dev_eval.json     summary from `python -m eval.runner`
  - reports/sweep.csv         leaderboard from `python -m eval.sweep`

Emits:
  - reports/demo.html         single-file HTML, copy-pasteable for the deck

Designed to be runnable without network — uses inline CSS, no external deps.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from datetime import datetime, timezone


_HTML_TMPL = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>wonderwall demo report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 980px; margin: 32px auto; padding: 0 16px; color: #222; }}
  h1, h2, h3 {{ color: #111; }}
  h1 {{ border-bottom: 2px solid #ddd; padding-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
  th, td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: right; }}
  th {{ background: #f5f5f5; font-weight: 600; text-align: left; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .ok {{ color: #2a7; font-weight: 600; }}
  .warn {{ color: #c70; }}
  .small {{ color: #777; font-size: 12px; }}
  pre {{ background: #f8f8f8; padding: 12px; overflow-x: auto; font-size: 12px;
         border-left: 3px solid #888; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 12px; }}
</style>
</head>
<body>

<h1>wonderwall — demo report</h1>
<p class="small">Generated {ts}. This is the artifact of <code>make demo</code>.
Numbers below are stub-data plumbing-validation; the real numbers land
once <code>KirkPipelineClient</code> is wired to the production wheel.</p>

<h2>Eval harness summary</h2>
{eval_table}

<h2>Sweep leaderboard (top {top_n})</h2>
{sweep_table}

<h3>Notes</h3>
<ul>
  <li>Stub-data run — synthetic Kirk outputs, real (small-LLM) Scotty endpoint
      for Pipelines A and B, mock LLM for Pipeline C if HF Transformers isn't
      loaded locally.</li>
  <li>Pipeline D is the HMM regime baseline. Its <code>$/yr</code> is essentially zero
      (CPU-only NumPy); use it as the regime-accuracy yardstick, not the cost row.</li>
  <li>Sources of truth for the math: <a href="https://kavara.atlassian.net/wiki/spaces/EDP/pages/93323266">Uhura page</a>,
      <a href="https://kavara.atlassian.net/wiki/spaces/EDP/pages/36044801">kirk-cli page</a>,
      <a href="https://kavara.atlassian.net/wiki/spaces/PE/pages/73531394">ts_sor_base-1 architecture</a>.</li>
</ul>

</body>
</html>
"""


def _eval_table_html(summary: dict) -> str:
    rows = []
    for pipe, d in summary.items():
        if not isinstance(d, dict) or pipe == "ratios":
            continue
        rows.append(
            f"<tr><td>{html.escape(pipe)}</td>"
            f"<td>{d.get('n', 0)}</td>"
            f"<td>{d.get('mean_input_tokens', 0):.1f}</td>"
            f"<td>{d.get('mean_output_tokens', 0):.1f}</td>"
            f"<td>{d.get('mean_e2e_ms', 0):.0f}</td>"
            f"<td>{d.get('mean_rouge_l', 0):.2f}</td>"
            f"<td>${d.get('mean_cost_usd', 0):.4f}</td>"
            f"<td>{d.get('regime_accuracy', 0):.2f}</td>"
            f"</tr>"
        )

    ratios = summary.get("ratios", {})
    ratio_rows = []
    for k, v in (ratios or {}).items():
        if v is None:
            continue
        ratio_rows.append(f"<tr><td>{k}</td><td>{v:.3f}</td></tr>")

    out = (
        "<table><thead><tr>"
        "<th>pipeline</th><th>n</th><th>tokens in</th><th>tokens out</th>"
        "<th>e2e ms</th><th>ROUGE-L</th><th>$/query</th><th>regime acc</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    if ratio_rows:
        out += (
            "<h3>Compression ratios</h3>"
            "<table><thead><tr><th>ratio</th><th>value</th></tr></thead><tbody>"
            + "".join(ratio_rows)
            + "</tbody></table>"
        )
    return out


def _sweep_table_html(rows: list[dict], top_n: int) -> str:
    if not rows:
        return "<p><em>No sweep data — run <code>make sweep-stub</code> first.</em></p>"
    rows = rows[:top_n]
    headers = [
        "rank", "pipeline", "n", "hidden_dim", "use_complex",
        "mean_input_tokens", "mean_e2e_ms", "mean_rouge_l", "cost_per_year_usd",
    ]
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for r in rows:
        body.append("<tr>" + "".join(
            f"<td>{html.escape(str(r.get(h, '')))}</td>"
            for h in headers
        ) + "</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", default="reports/dev_eval.json")
    p.add_argument("--sweep-csv", default="reports/sweep.csv")
    p.add_argument("--out", default="reports/demo.html")
    p.add_argument("--top-n", type=int, default=10)
    args = p.parse_args()

    summary = {}
    if os.path.exists(args.eval_json):
        with open(args.eval_json) as f:
            summary = json.load(f)
    eval_html = _eval_table_html(summary)

    rows: list[dict] = []
    if os.path.exists(args.sweep_csv):
        with open(args.sweep_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    sweep_html = _sweep_table_html(rows, top_n=args.top_n)

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rendered = _HTML_TMPL.format(
        ts=ts, eval_table=eval_html, sweep_table=sweep_html, top_n=args.top_n,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(rendered)
    print(f"[demo] wrote {args.out}")


if __name__ == "__main__":
    main()
