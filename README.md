# wonderwall

> **Status:** v0.1 — scaffold + observability + local dev stack · 2026-05-02
> **Owner:** John Edge · solo execution
> **Repo positioning:** sibling of `UlyssesModel/uhura`, `UlyssesModel/tiberius-openshift`, `UlyssesModel/kirk-runner`, `UlyssesModel/scotty`
> **Architectural narrative:** Kirk encoder → projection adapter → frozen Gemma 4 predictor (LLaVA-pattern)

## Purpose

Bridge between Kirk (Kavara's tensor engine, two-layer pipeline: 16×16 blocks →
32×32 feature matrix → entropy) and a frozen LLM. The LLM target is a config
swap (`configs/llm_*.yaml`); first-class peers as of v0.1 are **Gemma 4 31B**
(dense, served via Scotty/Ollama for Pipeline B and HF Transformers for
Pipeline C) and **DeepSeek v4** (MoE, sparse activation — relevant under GPU
constraints). Takes Kirk's outputs, projects them into the chosen LLM's
embedding space via a small trainable MLP, and injects them directly via
`inputs_embeds`.

Architecturally this is a vision-language-model pattern (LLaVA / BLIP-2 /
Flamingo) applied to non-stationary, low-SNR financial time series instead of
images: a trained encoder produces a compact representation, a small trainable
adapter projects it into a frozen LLM's embedding space, and the LLM is the
predictor. *Wonderwall does not claim to be a JEPA in the LeCun sense — that
work lives in the parallel Kavara × AMI Labs collaboration. See D-012.*

The thesis is that input-token compression beyond what tokenizing-the-text
can achieve, combined with running the entire pipeline inside an air-gapped
TDX Trust Domain, gives both a cost win and an IP-protection win for
financial-services customers. See [`docs/adr/0001-pipeline-c-vs-b.md`](docs/adr/0001-pipeline-c-vs-b.md)
for the full argument.

Three pipelines run side-by-side under the same V2 OpenInference predictor:

- **Pipeline A** — raw tokenized text via Scotty/Ollama. Worst-case baseline.
- **Pipeline B** — compressed text (Kirk latent rendered as a small numerical
  block) via Scotty. Already shipping in production today per Uhura's
  sweep leaderboard (2.8× token / 40× cost reduction).
- **Pipeline C** — embedding injection via HuggingFace Transformers. The
  new build. The thesis lives here.

Plus **Pipeline D**, a Gaussian-emission HMM regime classifier for
apples-to-apples regime-accuracy comparison against Pipelines A/B/C narrations.

## Where this fits in the Kavara stack

```
DATA INGEST          COMPRESSION (TGE)    COMPUTE SOR          KIRK              PROJECTION BRIDGE        LLM
────────────         ─────────────────    ───────────          ────              ─────────────────        ───
polygon-ingress  →   uhura            →   tiberius-openshift → kirk-pipeline  →  wonderwall         →   scotty
                                                              (Layer 1: 16×16    (this repo —             (Gemma 4 31B
                                                               Layer 2: 32×32    projection adapter        via Ollama,
                                                               entropy +         + injection wrapper       OAI-compatible)
                                                               Array+Vector)     + V2 KServe predictor)
```

Each layer is its own repo, composed by Kafka topic / file handoff. Per
[ts_sor_base-1 docs](https://kavara.atlassian.net/wiki/spaces/PE/pages/73531394):
**at N=32 the SOR routes Kirk to CPUBackend, not AMX.** AMX kicks in only at
N>500. Pipeline C's win at the Layer-2 tap point is from input compression,
not AMX vs GPU.

## Quick start

### Local dev — Docker Compose

The fastest end-to-end loop. No GPU, no OpenShift, no real wheel needed.

```bash
make compose-pull-model    # one-shot: pull gemma2:2b into Ollama (~2 GB)
make compose-up            # start ollama + redpanda + wonderwall-predictor
make smoke-test            # hit all three V2 routes with sample payload
make compose-down          # tear down
```

### Local dev — bare Python

```bash
make install               # venv + all extras
make preflight             # 30s sanity check on env, configs, endpoints
make test                  # 105 tests, CPU only
make demo                  # 5–30m: distill → train → eval → sweep → demo.html
```

### Production deploy on OpenShift

From cluster-zero — assumes OCP 4.21+, KServe RawDeployment, Strimzi,
Grafana Operator, NVIDIA GPU operator already installed.

```bash
make image && make push             # build/push container image
make openshift-apply                # Namespace + Strimzi cluster + workloads + monitoring
SMOKE_TEST_URL=https://... make smoke-test
```

## What's in the repo

```
wonderwall/
├── Makefile                            21 targets — install, test, demo, compose, openshift, smoke
├── compose.yaml                        local 3-service Docker stack
├── DECISIONS.md                        11 architectural decisions logged with sources
├── HANDOFF.md                          what's stubbed vs real, critical-path order
├── docs/
│   └── adr/0001-pipeline-c-vs-b.md     why Pipeline C exists vs Pipeline B
│   └── model_card.md                   template for Red Hat AI catalog publication
├── pyproject.toml                      6 install extras: llm, scotty, kafka, serve, dev, all
├── configs/                            5 YAMLs: adapter, llm_gemma4, scotty, train, sweep
├── samples/                            real V2 payload + generator
├── src/wonderwall/                   13 modules
│   ├── interfaces.py                   KirkOutput contract (Layer-2 shape), KirkClient Protocol
│   ├── adapter.py                      KirkProjectionAdapter (the trainable bridge)
│   ├── kirk_client.py                  Stub + KirkPipelineClient (real wheel) + KirkSubprocessClient
│   ├── injection.py                    EmbeddingInjectionLLM (HF) + ScottyClient (OAI-compat)
│   ├── pipeline.py                     CompressedTextPipeline + EmbeddingInjectionPipeline
│   ├── distill.py                      Teacher-LLM gold narration generation
│   ├── train.py                        LLaVA-style training loop with frozen LLM
│   ├── uhura_io.py                     defensive .npz loader with schema variant detection
│   ├── prompts.py                      versionable prompt registry
│   ├── metrics_export.py               Prometheus counters + histograms + gauges
│   └── logging_config.py               structured JSON logging
├── eval/                               6 modules
│   ├── baselines.py                    RawTextPipeline (A — worst-case baseline)
│   ├── metrics.py                      EvalRecord, ROUGE-L, cost estimates
│   ├── harness.py                      A/B/C/D side-by-side benchmark
│   ├── runner.py                       eval CLI
│   ├── sweep.py                        calibration grid, cost-per-year leaderboard
│   └── hmm_baseline.py                 Gaussian HMM with Viterbi + Baum-Welch EM
├── scripts/                            10 CLIs
│   ├── distill_teacher.py              gold narrations from teacher LLM
│   ├── train.py                        adapter training driver
│   ├── pin_gemma_hidden_dim.py         auto-pin LLM hidden dim from Ollama
│   ├── label_regimes.py                gold regime labels via technical indicators
│   ├── preflight.py                    env / config / endpoint sanity checks
│   ├── serve_kserve.py                 V2 OpenInference predictor (3 routes)
│   ├── stream_consumer.py              Kafka tensor frames → narrations
│   ├── render_demo_report.py           HTML report from sweep + eval
│   ├── smoke_test_routes.py            post-deploy V2 route health check
│   └── validate_against_entropy_price.py  v1 acceptance vs *_entropy_price.parq reference
├── deploy/
│   ├── openshift/
│   │   ├── 0-namespace.yaml            Namespace + ServiceAccount + RBAC
│   │   ├── 1-strimzi-cluster.yaml      Kafka 4.1.0 KRaft + topics
│   │   ├── wonderwall-inference.yaml KServe InferenceService (V2 OpenInference)
│   │   ├── wonderwall-streamer.yaml  per-cadence streaming Deployment
│   │   ├── wonderwall-podmonitor.yaml UWM Prometheus auto-scrape
│   │   └── Dockerfile                  UBI9 Python 3.12, restricted-SCC
│   ├── grafana/
│   │   └── wonderwall-dashboard.yaml 15-panel GrafanaDashboard CR
│   └── compose/
│       └── README.md                   docker compose docs
└── tests/                              11 files · 105 test cases
```

## Architectural decisions (one-line each — see DECISIONS.md for full context)

- **D-001** Project Array + Vector together; drop Scalar from LLM input
- **D-002** Token order: array rows first, then row-marginal summary, then col-marginal summary
- **D-003** Real-valued projection by default (production Kirk uses imag=0)
- **D-004** Two inference paths in v0.1 — embedding-injection cannot run on Ollama
- **D-005** Frozen LLM, trainable adapter only — LLaVA-style distillation
- **D-006** Layer-2 Kirk preferred for projection (when 2-layer pipeline is wired)
- **D-007** Repo positioning: sibling of uhura / tiberius-openshift / kirk-runner / scotty
- **D-008** At N=32 the Kirk forward pass uses CPUBackend, not AMX (deck framing)
- **D-009** Use the `amx-stride2-32` venue policy on GNR+TDX (E2 calibrated)
- **D-010** ~~JEPA framing as the external architectural narrative~~ — *superseded by D-012 (2026-05-04)*
- **D-011** Single-message stream batching: T = windows-per-batch
- **D-012** Drop "JEPA" from Wonderwall's branding; reserved for the parallel AMI Labs collaboration
- **D-013** Repo / package / image / service name = `wonderwall` end-to-end (formerly `ulysses-jepa`)
- **D-014** Dev plane on `stac-claude-dev` in the GCP Office of CTO project — not laptops, not Cowork
- **D-015** Multi-LLM target stance: Gemma 4 31B and DeepSeek v4 are first-class peers; adapter retrains per target

## Open questions before validation

These are the load-bearing items that block an end-to-end real-data run.
Tracked in HANDOFF.md.

1. **Gemma 4 31B's exact `hidden_size`** — `5376` placeholder; pin via
   `make pin-gemma` against the local Ollama once Gemma 4 31B is pulled.
2. **`KirkPipelineClient._run_layer2` output schema** — kirk-pipeline's
   `forward()` may return tuple/dataclass instead of dict. The integration
   test at `tests/test_kirk_pipeline_integration.py` lights up automatically
   when the wheel is installed and pinpoints the mismatch.
3. **Kafka frame schema** — `uhura_io.py` defends against four key-name
   variants (`tensor` / `frame` / `matrix` / `data`). Once the actual
   uhura broadcaster format settles, drop the unused aliases.

## Links and references

- [Uhura TGE Confluence](https://kavara.atlassian.net/wiki/spaces/EDP/pages/93323266) — input contract upstream
- [kirk-cli (formerly kirk-runner) Confluence](https://kavara.atlassian.net/wiki/spaces/EDP/pages/36044801) — Kirk control plane
- [ts_sor_base-1 Architecture](https://kavara.atlassian.net/wiki/spaces/PE/pages/73531394) — SOR backend / hardware routing
- [ts_sor_base-1 Implementation Status](https://kavara.atlassian.net/wiki/spaces/PE/pages/79921153) — venue policy, calibration
- [Kavara × Red Hat Collaboration](https://kavara.atlassian.net/wiki/spaces/PE/pages/80379905) — deployment context, KServe pattern
- [Scotty repo](https://github.com/UlyssesModel/scotty) — LLM serving layer downstream
- ADR-001: [`docs/adr/0001-pipeline-c-vs-b.md`](docs/adr/0001-pipeline-c-vs-b.md)
