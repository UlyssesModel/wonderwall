# ADR-001 — Why Pipeline C exists when Pipeline B already works

**Status:** Accepted
**Date:** 2026-05-02
**Author:** John Edge
**Supersedes:** none

## Context

Uhura's existing production path (Pipeline B in this repo's nomenclature)
already delivers measurable value: **2.8× input-token compression and 40×
end-to-end cost reduction** vs the raw-text baseline (Pipeline A) on the
sp500/N=21 sweep, using a 64-dim Kirk latent rendered as text into a small
LLM via Scotty/Ollama. Numbers are in production at top of Uhura's sweep
leaderboard.

When Pipeline B already pays the bills, why are we building Pipeline C
(embedding injection)? This ADR records the argument so future-self doesn't
re-litigate it.

## Decision

We are building Pipeline C as the **primary product**, with Pipeline B
remaining as the production fallback and the apples-to-apples cost baseline
in the eval harness.

Pipeline C's distinguishing value is **not** primarily cost reduction.
It is **IP protection through the air-gap-able form factor**.

## Why not just keep Pipeline B?

Pipeline B sends the Kirk latent's *numerical values* to whatever LLM
endpoint serves it. For air-gapped customers (financial services with TDX
Trust Domains, federal customers with disconnected bare-metal appliances),
this means:

1. **Numerical signal leakage.** The Kirk latent IS the financial signal —
   it's a learned representation of cross-asset structure that the model
   was trained to extract from raw market data. Sending those numbers to
   a hosted LLM (or any LLM running outside the Trust Domain) leaks
   proprietary alpha.
2. **Token-channel auditability gap.** Even with a local LLM, Pipeline B's
   tokens are human-readable strings of float values. They survive in logs,
   request traces, and any debugging artifact. This is the kind of trail
   that compliance and risk teams object to.
3. **The compression is bottom-bounded.** Pipeline B's tokens for the data
   channel are roughly proportional to `(n_features × n_decimal_chars)`.
   Below ~150 tokens per Kirk window, you can't compress the numerical
   text further without losing precision.

Pipeline C addresses all three:

1. **Signal is encoded in projected weights.** The adapter's MLP maps the
   Kirk latent to LLM-hidden-dim soft tokens. Without the adapter weights,
   the soft tokens are uninterpretable. Without the LLM weights, the soft
   tokens are unconsumable. The signal exists only in the conjunction.
2. **No human-readable token trail.** Soft tokens are floating-point
   activations. They don't pretty-print into log records.
3. **Compression floor drops 4–5×.** Pipeline C produces `T × (N+2)` soft
   tokens (default 4 windows × 34 = 136). Pipeline B produces ~150 tokens
   of float text plus instruction overhead. Below 150 the text version
   can't go; the soft-token version can.

Combined with Gemma 4 31B running locally inside an Intel TDX Trust Domain
(per the Kavara × Red Hat collaboration: confidential boundary at the
cluster-VM level, AMX-accelerated, attested boot), the entire pipeline is
**fully air-gappable end to end**. This is what financial-services
customers actually want; cost is the secondary win.

## Why Pipeline C is technically buildable

Three reasons it should hold:

1. **Information density argument.** Pipeline B serializes a 64-dim Kirk
   latent as ~150 tokens of float strings. Pipeline C compresses the same
   latent into ~34 soft tokens (N+2 at N=32) carrying the *same information
   optimized for what the LLM consumes*. End-to-end-trained projection.
2. **The LLaVA precedent.** LLaVA, BLIP-2, Flamingo all use this exact
   pattern — frozen LLM + small projection adapter from a separately-trained
   encoder. Recipe is well-trodden, training is stable, ~5–50M trainable
   parameters is enough.
3. **Pipeline B is the safety net.** Even if Pipeline C underperforms,
   Pipeline B continues paying its 40× cost reduction relative to the
   raw-text baseline. We're betting the marginal improvement, not the
   whole project.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Information loss in projection — Kirk's representations don't project linearly into Gemma's space | Adapter capacity tunable via `hidden_dim`; sweep harness iterates to find the floor at which it works. The MLP is small (~5–50M params); we can scale up. |
| Teacher-narration garbage — supervised signal is only as good as the teacher's narrations | `scripts/label_regimes.py` produces *factual* gold regime labels independent of teacher; `regime_correct` column scores narrations against ground truth, not just teacher ROUGE-L |
| Cost win smaller than headline — Pipeline B's 312 tokens are mostly instruction overhead, not Kirk-data tokens | Eval harness's per-pipeline token breakdown shows where the savings actually land. Don't lead the deck with extrapolated numbers; lead with measured A vs B vs C from the sweep. |

## Phase-4 outcomes and decision tree

After the first real-data eval on `tdx-amx-node-octo` (gated on items 1–5
in HANDOFF.md), three possible outcomes:

- **Pipeline C beats B by ≥3× cost at parity quality.**
  Ship Pipeline C as default. Deck leads with the headline cost number
  AND the IP-protection story. Pipeline B retained as a fallback.

- **Pipeline C beats B by 1.5–3×.**
  Ship Pipeline C for the IP-protection story; modest cost headline. The
  air-gap argument carries the deck; the cost is a bonus. Pipeline B
  retained.

- **Pipeline C ties or loses to B on cost-per-quality.**
  Pipeline C still ships **if the customer's threat model requires the
  no-tokenization channel** — air-gapped federal, financial services with
  alpha-leakage compliance concerns. Otherwise harden Pipeline B for
  production and pivot the deck to "Pipeline B beats GPU baseline by 40×
  in confidential compute." This is not a project failure — it's the
  expected case for customers who don't have IP-protection threat models.

The point of building Pipeline C is to **make the answer measurable**.
The build doesn't depend on a specific outcome to be valuable.

## Anti-decisions

What this ADR does NOT commit to:

- It does NOT commit to a specific cost-reduction number for the deck.
  Numbers come from `eval/sweep.py` against real data, not extrapolation
  from Uhura's optimistic-knob top-of-leaderboard.
- It does NOT commit to Gemma 4 31B as the only target model. The adapter
  is model-agnostic; we can retrain against Llama 3.x or Qwen if customer
  preference shifts. The `configs/llm_*.yaml` knob is the swap point.
- It does NOT commit to embedding-injection as the only future inference
  path. If a future LLM API exposes a structured input modality (something
  cleaner than `inputs_embeds`), we adopt it. Pipeline C is the current
  best-fit, not a permanent architecture.

## References

- [Kavara × WMD doc](../../docs/) — encoder/predictor architectural framing
- [Uhura sweep leaderboard](https://kavara.atlassian.net/wiki/spaces/EDP/pages/93323266) — Pipeline B production numbers
- [Kavara × Red Hat](https://kavara.atlassian.net/wiki/spaces/PE/pages/80379905) — TDX deployment context
- LLaVA / BLIP-2 / Flamingo — the established VLM precedent for "frozen LLM + small projection adapter from a separately-trained encoder"; this project is the financial-time-series equivalent
- [DECISIONS.md](../../DECISIONS.md) — finer-grained decisions D-001 through D-013 (note D-012 supersedes the earlier JEPA framing)
