# Handoff — what's stubbed, what's real, what's next

> **Status:** v0.1 build complete. 63 files, ~7K lines Python, 105 test
> cases passing locally. All Phase-0 questions resolved. Solo execution.

## 0. Naming + dev plane (read this first if you've been away)

**Project rename, 2026-05-04.** Everything formerly named `ulysses-jepa` is
now `wonderwall` — repo (`UlyssesModel/wonderwall`), Python package
(`wonderwall`), container image (`quay.io/kavara/wonderwall`), KServe
service, env vars (`WONDERWALL_*`). The workspace codename is also
`Wonderwall`, no codename/realname split. See **D-012** and **D-013** in
`DECISIONS.md` — short version: Kavara has a parallel collaboration with
**AMI Labs working directly on JEPA**, so Wonderwall drops JEPA branding
to avoid stepping on that scope. Architectural narrative is now
LLaVA-pattern (frozen LLM + trainable projection adapter), not "JEPA
encoder + predictor."

**Dev plane lives on `stac-claude-dev`** in the GCP "Office of CTO"
project (`office-of-cto-491318`, us-central1-a, internal IP 10.128.0.24,
external IP for SSH access). Cowork sandbox cannot run the loop (Python
3.10 only, no sudo, network allowlist blocks uv's github fetches). Local
laptops drift. Use `stac-claude-dev` as the canonical build/run surface;
it sits inside the same VPC as `scotty-gpu` (Gemma 4, 10.128.0.16) and
`gke-kirk-serving-confidential-c3-*` (Kirk Layer-2), so VM-to-VM calls use
internal IPs. See **D-014**.

**GitHub-as-pipe.** Push from laptop or Cowork → `stac-claude-dev` pulls.
First push: from laptop, `git init && git add . && git commit && git
remote add origin git@github.com:UlyssesModel/wonderwall.git && git push
-u origin main`. The repo on GitHub is currently empty pending that push.

**Cleanup needed on laptop after first pull from Cowork edits:** the
Cowork bash sandbox can create files but can't delete them through the
mount. The repo currently has `.bak` and `.bak2` backup files next to
every file edited during the rename, plus a `.cowork-write-probe` marker
and the residual `.venv/` + `src/wonderwall.egg-info/` from a failed
install attempt. Run on laptop:

```bash
cd ~/Documents/Claude/Projects/Project\ Wonderwall/ulysses-jepa
find . -type f \( -name "*.bak" -o -name "*.bak2" \) -delete
rm -f .cowork-write-probe
rm -rf .venv src/wonderwall.egg-info src/ulysses_jepa.egg-info
# then rename the parent folder if you want filesystem layout to match the repo:
cd .. && mv ulysses-jepa wonderwall
```

## What's real and runnable today

| Path | How to run | What it proves |
| --- | --- | --- |
| Unit tests (105 cases, CPU only) | `make test` | Adapter, kirk-client, pipelines, eval, distill, train, serve, prompts, logging, sweep, HMM all sound |
| Plumbing-validation demo | `make demo` | Distill → train → eval → sweep end-to-end against Stub Kirk + tiny LLM, emits `reports/demo.html` (see GPU-sizing note below) |
| Local Docker dev stack | `make compose-up` | Ollama + Redpanda + predictor running locally; smoke-test all 3 routes |
| Production OpenShift deploy | `make openshift-apply` | Namespace + Strimzi cluster + KServe InferenceService + Streamer + Grafana dashboard, dependency-ordered |
| Post-deploy health check | `make smoke-test` | All 3 V2 routes respond 200 with non-empty narrations |

### `make demo` GPU sizing — pick the right cost knob

`make demo`'s defaults target Gemma 4 31B (`configs/llm_gemma4.yaml` +
`configs/adapter_default.yaml`). Pipeline C's train-stub loads the LLM in
bf16 on a single GPU; **Gemma 4 31B at bf16 doesn't fit on 40GB**, so on
anything smaller than an 80GB GPU the demo OOMs at the train step.

**Demo on `scotty-gpu` (A100-SXM4-40GB, ~40GB) uses Gemma 3 12B** as the
Pipeline C train-stub model (~24GB bf16, fits comfortably). All five
make-vars below are read by `preflight`, `distill-stub`, `train-stub`,
`eval-stub`, and `sweep-stub`:

```bash
WONDERWALL_LLM_CONFIG=configs/llm_gemma3_12b.yaml \
WONDERWALL_ADAPTER_CONFIG=configs/adapter_demo.yaml \
WONDERWALL_TEACHER_MODEL=gemma3:12b \
make demo
```

Production demo with Gemma 4 31B requires **≥80GB GPU** (H100 80GB,
A100-80GB SXM, or multi-GPU sharded). On those boxes, `make demo` works
out of the box — no env-var override needed.

To pin a new cost-knob LLM against a local Ollama:

```bash
make pin-llm MODEL=<short-name> OLLAMA_MODEL=<ollama-tag> HF_ID=<hf-id>
# then create configs/llm_<short-name>.yaml + configs/adapter_<short-name>.yaml
# with llm_hidden_dim matching the pinned value.
```

## What's stubbed and needs filling in

### 1. LLM `hidden_size` per target

Two `configs/llm_*.yaml` targets, both first-class per **D-015** / **ADR-002**:

- **Gemma 4 31B** (`configs/llm_gemma4.yaml`) — pinned at `5376`, confirmed
  2026-05-04 against `scotty-gpu` (10.128.0.16:11434), `gemma4.embedding_length`
  via `ollama show gemma4:31b`. Re-run `make pin-gemma` after model upgrade.
- **DeepSeek V4-Pro** (`configs/llm_deepseek4.yaml`) — pinned at `7168`,
  confirmed 2026-05-04 from `deepseek-ai/DeepSeek-V4-Pro` HF `config.json`
  (the script's `--no-update-adapter` flag keeps this from clobbering the
  Gemma-aligned `adapter_default.yaml`). Pipeline C (HF embedding injection)
  is unblocked. **Pipeline B (text via Ollama) is blocked** until one of:
  1. **Ollama publishes a non-`:cloud` `deepseek-v4-pro` tag** so
     `scotty-gpu` can pull weights locally. As of 2026-05-04, only
     `deepseek-v4-pro:cloud` and `deepseek-v4-flash:cloud` exist on
     ollama.com — both routed through Ollama Cloud, no local weights.
  2. **vLLM-served DeepSeek V4-Pro on `scotty-gpu`** as a swap-in for
     Pipeline B's Scotty backend, per ADR-002's listed mitigation. Larger
     ops change (vLLM is a different serving runtime than Ollama, with
     its own quantization story for a 384-expert MoE).

  **Ollama Cloud was considered and rejected on threat-model grounds** —
  it routes inference through a hosted Ollama-managed endpoint, which
  violates ADR-001's no-hosted-LLM-for-air-gap-customers thesis.
  Air-gap-eligible deployments must run weights locally.

- **DeepSeek V4-Flash** (`configs/llm_deepseek4_flash.yaml`) — placeholder
  scaffold, parallel to the Gemma `:26b` dev cost knob. `hidden_dim: null`;
  pin when first used (`make pin-llm MODEL=deepseek4_flash`). Smaller
  variant — `hidden_size=4096`, 256 experts — meant for cost-bounded
  iteration, not production.

MoE-specific note: for Pipeline C we want the embedding-layer dim (what
the projection adapter feeds into via `inputs_embeds`), not an expert-internal
dim. Both Ollama's `embedding_length` and HF's top-level `hidden_size` give
this; the script reads the right one for either source.

The adapter checkpoint is per-LLM-target (different `output_dim`, different
soft-token semantics). Train and store separately:

```text
checkpoints/
├── gemma4-31b/adapter.pt
└── deepseek-v4/adapter.pt
```

`scripts/train.py --llm-config configs/llm_deepseek4.yaml --output-dir
checkpoints/deepseek-v4/` keeps them isolated.

### 2. `KirkPipelineClient._run_layer2` output schema

The integration test at `tests/test_kirk_pipeline_integration.py` is the
load-bearing safety net here. It runs only when `kirk_pipeline` is importable
in the test process — skipped on dev hosts, fires automatically on
tdx-amx-node-octo and IvorHQ.

When the wheel lands and the test runs, the most likely failure is the dict
shape returned by `KirkModelInterface.forward()` in `active_inference` mode.
Current code assumes `{"reconstruction": (n,n), "marginals": (2n,), "entropy": ()}`.
If the real return is a tuple, dataclass, or different keys, the test points
at exactly which line of `KirkPipelineClient._run_layer2` to fix. ~1 hour
of work.

### 3. Real distillation dataset

The training path runs end-to-end against synthetic data and a teacher LLM
(Scotty/Ollama works as the teacher in dev). For real numbers:

```bash
# Pull a trading day from the Quantbot reference set
python scripts/distill_teacher.py \
    --output data/distilled_train.pt \
    --uhura-frames-glob "data/uhura/2024-09-03/*.npz" \
    --teacher-base-url https://api.anthropic.com/v1 \
    --teacher-model claude-opus-4-6

# Tag with regime labels (gives the eval harness regime_correct values)
python scripts/label_regimes.py \
    --distilled data/distilled_train.pt \
    --out data/distilled_train_labeled.pt
```

Cost estimate: ~1K labeled streams at Opus pricing is ~$5–20.

### 4. KServe predictor entry point — DONE

`scripts/serve_kserve.py` is a real FastAPI server with three routes
(Pipelines A/B/C) and a `/metrics` endpoint. 16 test cases covering happy
paths and error paths.

### 5. Streaming Kafka consumer — DONE

`scripts/stream_consumer.py` consumes Uhura tensor frames, runs the
pipeline, publishes narrations to a downstream topic. Strimzi mTLS,
cooperative-sticky rebalancing, idempotent producer, structured logging,
Prometheus metrics on a separate port.

### 6. Container image build — TODO

`quay.io/kavara/wonderwall:v0.1` is referenced in the OpenShift manifests
but not built yet. Build with:

```bash
make image && make push
```

Requires podman + quay.io credentials.

## Critical-path order of operations

1. **Pin Gemma 4 31B hidden dim** — `make pin-gemma`. Five-minute job.
2. **Validate the adapter trains end-to-end on stub data** —
   `make distill-stub && make train-stub && make eval-stub`. Exercises the
   gradient path through frozen LLM. Use a smaller LLM for first run if
   Gemma 4 31B isn't local yet.
3. **Wire `KirkPipelineClient` against the real wheel** on
   `tdx-amx-node-octo` or `IvorHQ`. First call surfaces any layer-2 output
   schema mismatch — fix in `_run_layer2`.
4. **Pull a real Uhura frame batch + distill against a frontier teacher** —
   1K labeled streams via `scripts/distill_teacher.py`.
5. **First real adapter training run** — `python scripts/train.py` with the
   real config. Validate on dev set via `eval/runner.py`. Target ROUGE-L
   within 0.15 of Pipeline B baseline at <50% of B's input token cost.
6. **Build + push the container image, deploy to OpenShift** —
   `make image && make push && make openshift-apply`. The Strimzi cluster
   manifest creates Kafka cluster from zero; smoke-test confirms the routes
   are live.

## The Phase-4 decision tree

After step 5 produces real numbers, three possible outcomes:

- **Pipeline C beats B by ≥3× cost at parity quality** — ship Pipeline C as
  the new default. Conference deck headline: cost win + IP-protection win.
- **Pipeline C beats B by 1.5–3×** — ship Pipeline C for the IP-protection
  story (the air-gappable form factor); modest cost headline.
- **Pipeline C ties or loses to B on cost-per-quality** — Pipeline C still
  ships if the customer's threat model demands the no-tokenization channel,
  otherwise harden Pipeline B for production. Conference deck pivots to
  "Pipeline B beats GPU baseline by 40× and runs in confidential compute."

ADR-001 has the full argument. Pipeline C is not the only reason this
project ships — Pipeline B already pays the bills.

## VM-to-stack mapping (GCP Office of CTO project)

| Wonderwall layer | VM | Internal IP | Notes |
| --- | --- | --- | --- |
| Dev driver | `stac-claude-dev` | 10.128.0.24 | Claude Code + `make install / test / demo` here |
| LLM serving (Gemma 4 31B) | `scotty-gpu` | 10.128.0.16 | Pipeline B Ollama + Pipeline C HF inference both |
| Kirk Layer-2 (confidential) | `gke-kirk-serving-confidential-c3-*` | 10.128.0.12 | Granite Rapids confidential C3 |
| Kirk Layer-2 (default) | `gke-kirk-serving-default-pool-*` | 10.128.0.11 | Non-confidential pool |
| Real-data eval | `tdx-amx-node-octo` | 10.128.0.4 | TDX + AMX, first real eval target |
| Production deploy | `openshift-intel-tdx` | 10.128.0.19 | Air-gapped pitch target, no external IP |
| Confidential benchmark | `amd-sevsnp-benchmark` | 10.128.0.15 | AMD SEV-SNP comparison plane |
| Demo cluster | `ulysses-demo-2g6tw-master-0` | — | Existing demo cluster master |
| Soliton dev | `trader-dev` | 10.128.0.23 | Downstream agentic-fund consumer of narrations |
| VSCode UI | `kavara-visual-studio-ui` | 10.128.0.5 | Browser-accessible IDE |

When configuring endpoints, use **internal IPs** for VM-to-VM and external
only for laptop-to-VM. The ulysses-demo cluster name remains "ulysses"
(it's an existing artifact and doesn't carry the JEPA branding); rename
opportunistically next time it's recreated.

## Things that should NOT change without re-deciding

Pinned by the architecture; updates need a new entry in `DECISIONS.md` and
re-training:

- `KirkOutput` contract (layer2_input, layer2_reconstruction, layer2_marginals, entropy)
- Adapter token order (rows, row-summary, col-summary)
- `n=32` Layer-2 dimension (kirk-pipeline-defined; not a free parameter)
- Real-valued default (`use_complex=False`)
- Frozen LLM + token-mask + CE loss form (LLaVA-style distillation)
- Three-pipelines-in-one-server (A/B/C all hit the same predictor)
- Versioned prompts in `prompts.py` (changing prompts requires version bump
  and re-train, not in-place edit)
