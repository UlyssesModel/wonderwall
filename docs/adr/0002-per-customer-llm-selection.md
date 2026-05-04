# ADR-002 — Per-customer LLM target selection (Gemma 4 31B vs DeepSeek v4)

**Status:** Accepted
**Date:** 2026-05-04
**Author:** John Edge
**Supersedes:** none
**Refines:** ADR-001 anti-decision ("not committed to Gemma 4 31B as the only target"); D-015

## Context

D-015 made Gemma 4 31B and DeepSeek v4 first-class peer LLM targets in
v0.1. ADR-001's anti-decisions already noted that the projection adapter is
model-agnostic. What neither captured is the **commercial and operational
logic** for choosing between targets per customer / deployment. Without
that, the multi-LLM commitment risks becoming "two code paths nobody
chooses between"; the deck risks defaulting to Gemma because that's what
got built first; and customer conversations risk stalling on a question
nobody has thought through.

This ADR records the selection logic so future-self, future collaborators,
and customer-facing pitch material can converge on the right answer
quickly.

## Decision

**The LLM target is a per-customer / per-deployment knob, not a
project-wide commitment.** Every customer engagement starts with a
target-selection conversation; the answer drives which `configs/llm_*.yaml`
ships in their deployment, which adapter checkpoint trains against their
data, and which Pipeline-B base model `scotty-gpu` (or its equivalent in
their environment) is loading.

The default selections per customer profile are:

- **Red Hat / Intel pitch deck (CTO / DevRel audience):** Gemma 4 31B.
  Google brand recognition, established AMX work in the public domain,
  matches the existing Kavara × Red Hat collaboration narrative. DeepSeek
  v4 mentioned as the swap target to demonstrate model-agnostic claim.

- **Sovereignty-sensitive financial-services customers** (anyone who has
  a written stance on Google data-handling, US-China geopolitics with the
  *opposite* preference, or simply prefers non-hyperscaler-aligned open
  weights): DeepSeek v4. The TDX + DeepSeek v4 combination gives a full
  sovereignty story — open weights, customer-controlled compute, no
  Google or American-hyperscaler dependency in the inference path.

- **GPU-constrained deployments** (any environment where H100/A100
  capacity is the binding constraint — currently most environments per
  the 2026-05-03 GCP availability data): DeepSeek v4. MoE sparse
  activation lowers per-token GPU compute, so the same GPU footprint
  serves more requests per second. This is a real number, not a hand
  wave — measure in the eval harness once both targets are pinned.

- **Hybrid (Pipeline B with one LLM, Pipeline C with another):**
  permitted. The predictor server can hold two `WONDERWALL_LLM_CONFIG`s
  in environment, route Pipeline B to Scotty/Ollama-Gemma and Pipeline C
  to HF-DeepSeek-v4. Useful for evaluators who want to see the same
  Kirk window narrated by two different LLMs side by side.

## Why each target wins where

| Concern | Gemma 4 31B (dense) | DeepSeek v4 (MoE) |
| --- | --- | --- |
| Per-token GPU compute | Higher — full 31B forward per token | Lower — only active experts compute (typically a fraction of total params) |
| Peak memory | Predictable, ~31B × dtype | Higher peak (full expert pool resident) but active compute lower |
| License lineage | Google's Gemma terms | DeepSeek open weights, China-origin |
| Customer sovereignty story | "Google open model" — compatible with most US enterprise procurement | "Non-Google, non-Meta, non-OpenAI" — compatible with sovereignty-sensitive procurement |
| AMX-only / no-GPU fallback | Unproven for 31B dense; latency probably impractical | Active-N MoE plausibly tractable on AMX; worth measuring as the all-CPU production fallback |
| Maturity for first-deploy | Higher — more widely deployed at the 31B-dense scale | Newer; serving runtime stack (vLLM MoE support, Ollama MoE handling) less battle-tested |
| Pitch credibility | Strong with Red Hat / Intel teams | Strong with sovereignty-sensitive customers |

## Technical implications

1. **Adapter checkpoint is per-target.** `checkpoints/gemma4-31b/adapter.pt`
   and `checkpoints/deepseek-v4/adapter.pt` are independent training
   artifacts. The projection MLP's `output_dim` matches each LLM's
   `hidden_size`, and the soft-token distribution the adapter learns to
   produce is specific to each LLM's embedding-layer characteristics.
   `scripts/train.py --llm-config <path> --output-dir <per-target-dir>`
   keeps them isolated.

2. **Eval harness reports per-target numbers.** `eval/runner.py` and
   `eval/sweep.py` accept multiple `--llm-config` paths; the leaderboard
   table is `(pipeline, llm_target) → (rouge_l, regime_correct, cost,
   tokens)`. The deck picks the hero per customer.

3. **Per-target prompt registry.** `src/wonderwall/prompts.py` is
   versionable; v0.1 has one prompt set, but DeepSeek's instruct format
   (`<|user|>...<|assistant|>`-style) and Gemma's chat template
   (`<start_of_turn>user...<end_of_turn>`) are not interchangeable. When
   the first cross-LLM eval shows quality drift attributable to prompt
   formatting, branch the registry into per-target prompt families.
   Until then, the same prompt content with target-specific framing
   tokens is fine.

4. **Pipeline B (text via Scotty) requires a serving runtime that
   handles each target.** Gemma 4 31B works in Ollama today. DeepSeek v4
   on Ollama may need the `trust_remote_code` equivalent; if Ollama
   doesn't support DeepSeek's MoE routing layers cleanly, fall back to
   a vLLM-served DeepSeek for Pipeline B, exposed via the same
   OpenAI-compatible URL Scotty already speaks.

5. **TDX-attested deployment is target-orthogonal.** The TDX Trust Domain
   doesn't care which LLM runs inside it. Both targets support the
   air-gappable form factor as long as the weights can be staged inside
   the attested image.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Two-LLM matrix doubles training and eval cost | Stub-data plumbing validates both targets cheaply; full retrain runs only at customer-commitment milestones, not on every project change |
| DeepSeek v4 license / export-control surfaces a customer objection | Audit at deal-close; Gemma 4 is the safe-default fallback. Document customer LLM-choice in each engagement's deal record |
| Pipeline-C-with-DeepSeek-v4 cost win turns out smaller than the MoE pitch suggests | Don't lead the deck with extrapolated numbers; lead with measured Pipeline-A-vs-B-vs-C-per-target from the sweep harness |
| Per-LLM prompt drift breaks reproducibility | Versioned prompts; bump version on prompt change; eval records the prompt version used |
| MoE serving runtime gaps (Ollama / vLLM lag DeepSeek v4 specifics) | Pipeline C HF-Transformers path is the load-bearing one for DeepSeek v4 in v0.1; hardening Pipeline B for DeepSeek can wait until a customer pulls on it |

## Decision tree at customer first-conversation

```
Customer asks "what LLM?"
│
├── Do they have a stated sovereignty / non-Google preference?
│       → DeepSeek v4. (Lead with sovereignty story + GPU-efficiency hedge.)
│
├── Are they buying primarily on the Red Hat / Intel relationship?
│       → Gemma 4 31B. (Lead with Google open-model + AMX establish narrative.)
│
├── Are they bottlenecked on hosted GPU capacity in their target region?
│       → DeepSeek v4. (Lead with MoE per-token compute reduction.)
│
├── Do they want both, comparing side-by-side?
│       → Hybrid Pipeline-B-Gemma / Pipeline-C-DeepSeek-v4 deploy.
│         Per-target eval numbers carry the demo.
│
└── No strong preference?
        → Gemma 4 31B as the safe default. Position DeepSeek v4 as a
          one-config-swap upgrade path the customer can request later.
```

## Anti-decisions

- This ADR does NOT commit to a third LLM target (Llama, Qwen, Mistral,
  etc.). Adding one is "another `configs/llm_*.yaml` + a retraining run"
  per ADR-001's anti-decisions. We add when a customer pulls on it.

- This ADR does NOT commit to running both targets in every production
  deployment. Per-customer deploys default to one target unless they
  explicitly want both.

- This ADR does NOT commit to specific cost-reduction numbers for
  DeepSeek v4 vs Gemma 4 31B. The sweep harness measures them; the
  deck reports measured numbers, not extrapolated ones.

- This ADR does NOT commit to DeepSeek v4 being the *long-term* primary
  target. If a future open model (Llama 4, Qwen 4, an Anthropic open
  release that doesn't exist yet) becomes the obvious sovereignty-and-
  efficiency winner, we add it as `configs/llm_<name>.yaml` and
  reassess.

## References

- ADR-001: [`0001-pipeline-c-vs-b.md`](0001-pipeline-c-vs-b.md) — anti-decision
  noting model-agnostic adapter; this ADR refines that.
- D-015 in [`../../DECISIONS.md`](../../DECISIONS.md) — multi-LLM stance.
- [`../../configs/llm_gemma4.yaml`](../../configs/llm_gemma4.yaml),
  [`../../configs/llm_deepseek4.yaml`](../../configs/llm_deepseek4.yaml) —
  the two target configs.
- LLaVA / BLIP-2 / Flamingo precedent — frozen-LLM + trainable-adapter
  pattern is independent of which LLM is frozen.
- Kavara × AMI Labs JEPA collaboration (separate project) — *not* the
  pitch anchor for Wonderwall. See D-012.
