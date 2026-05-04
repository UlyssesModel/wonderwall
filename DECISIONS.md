# Architectural decisions log

> One entry per decision. Each entry: **Decision · Why · Source.**
> Add new entries at the bottom; never edit historical ones — append a
> superseding entry instead.

## D-001 Project Array + Vector together; drop Scalar from LLM input
**Decided:** 2026-05-02

**What:** The projection adapter consumes Kirk's Array (N×N) and Vector (2N)
outputs, projects them as a sequence of (N+2) embedding vectors, and feeds
that to the LLM. The Scalar output is preserved as an anomaly gate but not
injected into the LLM.

**Why:** Array carries spatial/temporal structure; Vector carries
row+column marginal summaries. Together they form a multi-resolution
representation that maps naturally onto an LLM token sequence. Scalar is
one number — useful for routing/gating, too lossy to inject as a "thought."

**Source:** Kavara Data Science Guide, slides 13–14 (output types).
Conversation 2026-05-02.

## D-002 Token order: array rows first, then row-marginal summary, then col-marginal summary
**Decided:** 2026-05-02

**What:** Inside one Kirk window, the (N+2) projected vectors are
concatenated in this order: `[row_0, row_1, ..., row_{N-1}, row_summary,
col_summary]`.

**Why:** Gives the LLM a coherent reading order: the state itself, then a
summary of "what each row aggregates to," then "what each column aggregates
to." Aligned with how transformer attention naturally accumulates context
left-to-right.

**Source:** Architectural sketch 2026-05-02. Subject to validation when
training data is available.

## D-003 Real-valued projection by default
**Decided:** 2026-05-02

**What:** `AdapterConfig.use_complex` defaults to `False`. Real component
of complex inputs is used; imag is dropped.

**Why:** Per Ted's `kirk_data_description.md`, production Kirk uses
real-valued log-returns with imag uniformly zero. The complex128 container
is preserved as an optional second-channel research extension. No
information loss in dropping imag when imag=0. When the second-channel
extension is enabled, set `use_complex=True` and the adapter splits real/imag
and concatenates along the input dim — width doubles, info preserved.

**Source:** Uhura Confluence page (Ted-spec alignment table). Ted's
`kirk_data_description.md`.

## D-004 Two inference paths in v0.1 — embedding-injection cannot run on Ollama
**Decided:** 2026-05-02

**What:** Pipeline B (compressed-text via ScottyClient → Ollama) and
Pipeline C (embedding-injection via HF Transformers). Pipeline B is
production-shape; Pipeline C is the new build. Both run side by side in the
eval harness.

**Why:** Ollama's OpenAI-compatible `/v1/chat/completions` endpoint takes
tokens, not embeddings. Pure `inputs_embeds` injection requires direct model
access (vLLM `prompt_embeds` or HF Transformers). The cost-saving thesis
lives in Pipeline C; the production baseline is Pipeline B. The eval
harness measures both against the worst-case Pipeline A (raw text).

**Source:** Scotty repo README; OpenAI API spec. Conversation 2026-05-02.

## D-005 Frozen LLM, trainable adapter only — LLaVA-style distillation
**Decided:** 2026-05-02

**What:** During training the LLM weights are frozen (`requires_grad=False`).
Gradients flow through the LLM (no parameter updates) and into the
projection adapter, which updates against next-token cross-entropy on
teacher-generated narrations.

**Why:** Standard recipe in vision-language models. Adapter is small
(~5–50M parameters); training in hours on a single H100 is realistic.
Bypasses the cost and instability of co-training a 31B-parameter LLM.

**Source:** LLaVA / BLIP-2 papers; conversation 2026-05-02.

## D-006 Layer-2 Kirk preferred for projection (when available)
**Decided:** 2026-05-02

**What:** When Spencer's two-layer Kirk pipeline is wired (per the
"Aggregating array time series" pattern in the Data Science Guide slides
20–23), the projection adapter taps off the layer-2 outputs, not layer-1.

**Why:** Layer-2 is already a learned hierarchical representation —
Kirk has done its own attention-equivalent work. Projecting from layer-2
gives the LLM a richer starting point and sidesteps any criticism that
the projection is "just an embedding model."

**Source:** Kavara Data Science Guide, slides 20–23. Conversation
2026-05-02.

## D-007 Repo positioning: sibling of uhura / tiberius-openshift / scotty
**Decided:** 2026-05-02

**What:** wonderwall is its own repo; consumes Uhura's tensor frames or
Kirk outputs via standard interfaces (Kafka or in-process); produces LLM
narrations or downstream completions.

**Why:** Matches the layered architecture in the Uhura Confluence page.
Each layer composes by Kafka topic / file handoff, no glue code. Allows
independent versioning, deployment, and testing.

**Source:** Uhura Confluence page (four-layer Kavara stack diagram).

## D-008 At N=32 the Kirk forward pass uses CPUBackend, not AMX
**Decided:** 2026-05-02

**What:** ts_sor_base-1's auto-routing logic puts N≤20 on FusedBackend (C/MKL,
zero-alloc, L1-resident), 21≤N≤500 on CPUBackend (NumPy/MKL AVX-512), and
N>500 on AMXBackend (PyTorch BF16 + oneDNN AMX). The kirk-pipeline Layer-2
dimension is fixed at N=32, which falls into the CPU bucket. **AMX does not
fire on the embedding-injection hot path.**

**Why this matters:** the Red Hat conference deck must be careful about the
AMX claim. Pipeline C's cost reduction comes from input-token compression
relative to Pipeline A and B — *not* from AMX-vs-GPU compute differences.
The AMX story is real but applies to Uhura's large-N sweeps (sp500/N=500+),
not to the Layer-2 embedding-injection path.

**Implication for the deck:** lead with input compression and the LLM-tier
shift as the wonderwall thesis. Reference the AMX numbers only in the
context of Uhura's existing leaderboard, where AMX is real and measured.
Don't conflate them.

**Source:** `ts_sor_base-1 — Architecture & Backend Design` (PE/73531394).
`ts_sor_base-1 Implementation Status — 2026-04-19` (PE/79921153).
Conversation 2026-05-02.

## D-009 Use the `amx-stride2-32` venue policy on GNR+TDX
**Decided:** 2026-05-02

**What:** OpenShift Deployments set `OMP_NUM_THREADS=32`, `OMP_PLACES={0}:32:2`,
`OMP_PROC_BIND=close` in the container env. Mirrors the calibrated
`VENUE_POLICIES['gnr-tdx']` for both pinned and spread buckets after the
E2 calibration sweep.

**Why:** stride-2 32 threads beat every other tested config (lower-32,
upper-32, full-64-close, full-64-spread) at all N≥500 on Jarett's image
digest `4e9bc9e63f`. Caveat from the 2026-04-20 peer review: the win is
partly thread-count + turbo, not pure placement. The shipped policy is
empirically correct; the attribution is being re-validated via E6.

**Source:** `ts_sor_base-1 Implementation Status — 2026-04-19`,
"E2 results" section. Conversation 2026-05-02.

## D-010 ~~JEPA framing as the external architectural narrative~~ — SUPERSEDED
**Decided:** 2026-05-02 · **Superseded:** 2026-05-04 by D-012

**Original "What":** External-facing description: "Ulysses is the JEPA encoder for
non-stationary, low-SNR time series; an LLM acts as the predictor consuming
the encoder's embedding."

**Original "Why":** Anchored in current literature (LeCun's JEPA work). Clean
role separation maps onto the Red Hat / Intel pitch (encoder on CPU, predictor
on GPU). Credibility with technical evaluators.

**Why superseded:** Kavara has a separate active collaboration with AMI Labs
working directly on JEPA. Using JEPA branding for this project (Wonderwall)
conflates the two efforts and risks (a) overclaiming on Wonderwall, (b)
diluting the AMI Labs collaboration's positioning, or (c) confusing
technical evaluators about which project is which. See D-012 for the
replacement framing.

**Source:** Kavara × WMD doc; conversation 2026-05-02. Reversed in
conversation 2026-05-04 once the AMI Labs scope conflict was surfaced.

## D-011 Single-message stream batching: T = windows-per-batch
**Decided:** 2026-05-02

**What:** The streamer accumulates `windows_per_batch` (default 4) frames
from the input topic before running one inference call and emitting one
narration. No within-batch parallelism; cross-batch parallelism is via
multiple Deployment replicas reading different Kafka partitions.

**Why:** Mirrors the way Uhura's broadcaster emits one frame per cadence
tick. Batching at this layer keeps end-to-end latency bounded (4 windows ×
12s cadence = 48s narration latency for the 12s deployment) and keeps
the LLM call dense (more soft tokens per call → better GPU utilization).

**Source:** uhura streamer pattern; conversation 2026-05-02.

## D-012 Drop "JEPA" from Wonderwall's branding entirely
**Decided:** 2026-05-04 · **Supersedes:** D-010

**What:** The external-facing architectural description for this project
(Wonderwall) does NOT use the term "JEPA" anywhere — not in the README, not
in pitch decks, not in the model card, not in published artifacts. The
encoder/predictor architectural pattern is described functionally:

> Wonderwall is the projection adapter that lets Kavara's Kirk tensor engine
> drive a frozen LLM. Kirk produces a 32×32 feature matrix per market window;
> a small trainable MLP projects each row plus row/column-marginal summaries
> into the LLM's embedding space; those soft tokens are injected via
> `inputs_embeds` to a frozen Gemma 4 31B running inside a TDX Trust Domain.
> The architectural pattern is LLaVA / BLIP-2 / Flamingo applied to
> non-stationary, low-SNR financial time series.

The pitch's technical credibility now rests on the LLaVA precedent (already
cited in ADR-001 as the LLaVA-style distillation justification), not on
borrowed JEPA framing.

**Why:** Kavara has a parallel collaboration with AMI Labs working *directly*
on JEPA. Using JEPA branding for Wonderwall would step on that
collaboration's positioning and create unnecessary confusion in any
external context where both projects might be discussed (Red Hat / Intel
deck, technical evaluations, conference talks, hiring conversations). The
LLaVA framing is also more honest: Wonderwall is literally a frozen-LLM +
trainable-projection-adapter pattern, which is what LLaVA / BLIP-2 / Flamingo
established. JEPA is a different lineage; we don't need to borrow it.

**Source:** Conversation 2026-05-04. Anti-conflation discipline with the
AMI Labs scope.

## D-013 Repo / package / image / service name = `wonderwall` end-to-end
**Decided:** 2026-05-04

**What:** Every code-level artifact carries the name `wonderwall`:

- GitHub repo: `UlyssesModel/wonderwall`
- Local folder: `wonderwall/` (formerly `ulysses-jepa/`)
- Python package: `wonderwall` (formerly `ulysses_jepa`)
- pyproject `name`: `wonderwall`
- Container image: `quay.io/kavara/wonderwall`
- KServe service / OpenShift InferenceService: `wonderwall`
- Env vars: `WONDERWALL_ADAPTER_CHECKPOINT`, `WONDERWALL_ADAPTER_CONFIG`,
  `WONDERWALL_LLM_CONFIG` (formerly `ULYSSES_JEPA_*`)

The workspace codename is also `Wonderwall` — same string, no
codename-vs-realname split.

**Why:** Two reasons. (1) The earlier name `ulysses-jepa` carried the JEPA
branding that D-012 retires; renaming the artifacts is the only way to
purge that branding from production names that survive in image registries,
log records, and deployment manifests for years. (2) Having the workspace
codename and the production name be the same string eliminates a class of
"what's this project called again?" friction in cross-team conversations.

**Source:** Conversation 2026-05-04. Implemented as a one-pass rename.

## D-014 Wonderwall dev plane runs on `stac-claude-dev` in the Office of CTO project
**Decided:** 2026-05-04

**What:** The local dev loop (`make install / preflight / test / demo /
compose-up`) runs on the GCP VM `stac-claude-dev` (10.128.0.24,
us-central1-a, project `office-of-cto-491318`), not on developer laptops
and not in the Cowork sandbox. Cowork is the planning/writing surface;
`stac-claude-dev` is the build/run surface; `tdx-amx-node-octo` is the
real-data eval surface; `openshift-intel-tdx` is the production deploy
surface.

**Why:** (a) The Cowork sandbox can't host a torch+transformers env (Python
3.10 only, no sudo, network allowlist blocks github fetches needed by uv).
(b) Developer-laptop dev loops drift across machines. (c) `stac-claude-dev`
already exists, has external IP for SSH, and sits inside the same VPC as
`scotty-gpu` (Gemma 4) and `gke-kirk-serving-confidential-c3-*` (Kirk
Layer-2), so VM-to-VM calls use internal IPs without round-trips through
the public internet. (d) GitHub-as-pipe (push from laptop, pull on
`stac-claude-dev`) gives a reproducible dev loop without yak-shaving each
contributor's local machine.

**Source:** Conversation 2026-05-04. Triggered by Cowork sandbox install
failure + revealed GCP topology.

## D-015 Multi-LLM target stance: Gemma 4 31B and DeepSeek v4 are first-class peers
**Decided:** 2026-05-04 · **Refines:** ADR-001 anti-decision ("not committed to Gemma 4 31B as the only target")

**What:** v0.1 ships with two `configs/llm_*.yaml` targets that are equally
supported in the eval harness, training scripts, and KServe predictor:

- `configs/llm_gemma4.yaml` — Gemma 4 31B (dense). Default. Served on
  `scotty-gpu` for both Pipeline B (Ollama) and Pipeline C (HF Transformers).
- `configs/llm_deepseek4.yaml` — DeepSeek v4 (MoE). Alternate. Same VM,
  different `WONDERWALL_LLM_CONFIG` env var.

The adapter is retrained per target — the projection MLP's `output_dim`
matches each LLM's `hidden_size`, and the soft-token semantics differ
between dense and MoE. Eval results report numbers per-target so the
deck can pick its hero.

**Why:** Three reasons.

(1) **GPU-availability hedge.** 2026-05-03 demonstrated that hyperscaler
H100/A100 capacity isn't reliably purchasable on demand. DeepSeek's MoE
shape (sparse activation — only a fraction of total parameters compute
per token) means per-inference compute can be lower than dense Gemma 4
31B at comparable quality. When GPU is the binding constraint, MoE is
disproportionately attractive.

(2) **Customer sovereignty.** Some financial-services customers prefer a
non-Google-lineage open model — for licensing, data-residency, or
geopolitical reasons. DeepSeek's open weights satisfy that without
forcing a project pivot.

(3) **Evaluator credibility.** "We swap LLMs by changing one config file"
is a stronger architectural claim than "we built against Gemma." It
demonstrates the projection-adapter pattern is the load-bearing
abstraction, not the choice of LLM. Anti-vendor-lock-in story for the
Red Hat / Intel deck.

**Anti-decisions:** This does NOT commit to running both LLMs in
production simultaneously — that's a deployment choice per customer.
v0.1 commits to BOTH being supported in code; pick one or both at
deploy time. Also does NOT commit to a third or fourth LLM target;
adding Llama / Qwen later means another `configs/llm_*.yaml` and a
re-training run, no architectural change.

**Source:** Conversation 2026-05-04. Triggered by JE's failed H100/A100
hunt + asked "can we add in DeepSeek v4."
