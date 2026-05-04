"""Evaluation harness for wonderwall.

Three pipelines under comparison, all consuming the same input streams:

  Pipeline A — tokenized-raw-data baseline
               Send the raw N×N tensors verbatim as numerical text to the LLM.
               Expensive in tokens. Establishes the worst case.

  Pipeline B — compressed-text via Scotty / Ollama (today's production)
               Run Kirk, render outputs as compact text, send to Scotty.
               Already in production; numbers are in the Uhura sweep leaderboard.

  Pipeline C — embedding-injection via vLLM / HF (the new build)
               Run Kirk, project to LLM embedding space, inject directly.
               The thesis: tighter compression than text-encoded Pipeline B,
               same or better narration quality.

Eval metrics are computed on the same input set across all three:
  - input_token_count (or soft-token-equivalent for C)
  - prefill_latency_ms
  - decode_latency_ms
  - rouge_l vs teacher gold
  - regime_accuracy vs HMM baseline (financial-narration task only)
  - dollar_per_query_estimate (CPU + GPU mixed pricing assumption)
"""

from .metrics import EvalRecord, compute_metrics  # noqa: F401
