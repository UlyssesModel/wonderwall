.PHONY: help install lint test test-cov pin-gemma pin-deepseek pin-llm distill-stub train-stub eval-stub sweep-stub serve clean image push openshift-apply openshift-delete

# Defaults
PYTHON ?= python3
VENV   ?= .venv
PIP    ?= $(VENV)/bin/pip
PY     ?= $(VENV)/bin/python
IMAGE  ?= quay.io/kavara/wonderwall
TAG    ?= v0.1
NS     ?= kavara-ai

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Local dev loop
# ---------------------------------------------------------------------------

$(VENV)/.installed: pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[llm,scotty,kafka,serve,dev]"
	@touch $@

install: $(VENV)/.installed  ## Create venv + install with all extras

lint: install  ## Lint with ruff
	$(PY) -m ruff check src eval scripts tests

test: install  ## Run pytest (no GPU needed)
	$(PY) -m pytest tests -v

test-cov: install  ## Run pytest with coverage
	$(PY) -m pytest tests -v --cov=wonderwall --cov-report=term-missing

# ---------------------------------------------------------------------------
# One-shot config + plumbing
# ---------------------------------------------------------------------------

SCOTTY_BASE_URL ?= http://127.0.0.1:11434

pin-gemma: install  ## Auto-pin Gemma 4 31B hidden dim from $$SCOTTY_BASE_URL (default: localhost)
	$(PY) scripts/pin_gemma_hidden_dim.py --base-url $(SCOTTY_BASE_URL)

pin-deepseek: install  ## Auto-pin DeepSeek V4-Pro hidden dim (per ADR-002 multi-LLM target selection)
	$(PY) scripts/pin_gemma_hidden_dim.py \
		--base-url $(SCOTTY_BASE_URL) \
		--llm-config configs/llm_deepseek4.yaml \
		--ollama-model deepseek-v4 \
		--hf-fallback deepseek-ai/DeepSeek-V4-Pro \
		--no-update-adapter

pin-llm: install  ## Generic pin: make pin-llm MODEL=<gemma4|deepseek4|gemma3_12b|...> [OLLAMA_MODEL=...] [HF_ID=...]
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make pin-llm MODEL=<gemma4|deepseek4|gemma3_12b|...> [OLLAMA_MODEL=<tag>] [HF_ID=<hf-id>]"; exit 1; \
	fi
	$(PY) scripts/pin_gemma_hidden_dim.py \
		--base-url $(SCOTTY_BASE_URL) \
		--llm-config configs/llm_$(MODEL).yaml \
		$(if $(OLLAMA_MODEL),--ollama-model $(OLLAMA_MODEL),) \
		$(if $(HF_ID),--hf-fallback $(HF_ID),) \
		--no-update-adapter

# Demo cost-knob: default to production target (Gemma 4 31B, ≥80GB GPU); a
# 40GB-class box (e.g. scotty-gpu A100-SXM4-40GB) overrides via env vars to
# swap in the smaller Gemma 3 12B + matching adapter so Pipeline C training
# fits in memory. Honored by preflight, distill-stub, train-stub, eval-stub,
# and sweep-stub — every step that touches the LLM target.
WONDERWALL_LLM_CONFIG     ?= configs/llm_gemma4.yaml
WONDERWALL_ADAPTER_CONFIG ?= configs/adapter_default.yaml
WONDERWALL_TEACHER_MODEL  ?= gemma4:31b

preflight: install  ## Run environment / config / endpoint sanity checks
	$(PY) scripts/preflight.py \
		--adapter-config $(WONDERWALL_ADAPTER_CONFIG) \
		--llm-config $(WONDERWALL_LLM_CONFIG) \
		--scotty-model $(WONDERWALL_TEACHER_MODEL)

preflight-strict: install  ## Same as preflight, but warnings are fatal
	$(PY) scripts/preflight.py --strict \
		--adapter-config $(WONDERWALL_ADAPTER_CONFIG) \
		--llm-config $(WONDERWALL_LLM_CONFIG) \
		--scotty-model $(WONDERWALL_TEACHER_MODEL)

label-regimes: install  ## Tag a distilled .pt with gold regime labels
	$(PY) scripts/label_regimes.py --distilled $(DEV_DATA) --out data/distilled_dev_labeled.pt

# ---------------------------------------------------------------------------
# Plumbing-validation pipeline (stub Kirk + local Scotty)
# ---------------------------------------------------------------------------

DEV_DATA   = data/distilled_dev.pt
DEV_CKPT   = checkpoints/adapter.pt
DEV_REPORT = reports/dev_eval.json
DEV_SWEEP  = reports/sweep.csv

distill-stub: install  ## Generate a small stub-input distilled set against local Scotty
	$(PY) scripts/distill_teacher.py \
		--output $(DEV_DATA) \
		--use-stub-input --num-streams 32 --windows-per-stream 4 --n 32 \
		--teacher-model $(WONDERWALL_TEACHER_MODEL)

train-stub: install distill-stub  ## Train the adapter on stub Kirk + local Scotty teacher
	$(PY) scripts/train.py \
		--adapter-config $(WONDERWALL_ADAPTER_CONFIG) \
		--llm-config $(WONDERWALL_LLM_CONFIG) \
		--train-config configs/train_default.yaml \
		--use-stub-kirk

eval-stub: install train-stub  ## Run A/B/C eval harness against stub Kirk
	$(PY) -m eval.runner \
		--distilled $(DEV_DATA) \
		--adapter $(DEV_CKPT) \
		--adapter-config $(WONDERWALL_ADAPTER_CONFIG) \
		--llm-config $(WONDERWALL_LLM_CONFIG) \
		--scotty-model $(WONDERWALL_TEACHER_MODEL) \
		--use-stub-kirk \
		--out $(DEV_REPORT)

sweep-stub: install train-stub  ## Run the calibration sweep harness end-to-end
	$(PY) -m eval.sweep \
		--distilled $(DEV_DATA) \
		--adapter $(DEV_CKPT) \
		--llm-config $(WONDERWALL_LLM_CONFIG) \
		--scotty-model $(WONDERWALL_TEACHER_MODEL) \
		--grid configs/sweep_default.yaml \
		--out $(DEV_SWEEP)

smoke-test: install  ## Hit all 3 deployed V2 routes; exit non-zero on any failure
	$(PY) scripts/smoke_test_routes.py \
		--base-url $${SMOKE_TEST_URL:-http://127.0.0.1:8080}

compose-up:  ## Start the local Docker dev stack (ollama + redpanda + predictor)
	docker compose up -d
	@echo ""
	@echo "Stack starting. Once ready:"
	@echo "  docker compose ps"
	@echo "  curl http://localhost:8080/v2/health/ready"

compose-pull-model:  ## One-shot pull of the dev-tier Gemma into the Ollama volume
	docker compose --profile setup run --rm ollama-pull

compose-down:  ## Tear down the local stack (use 'make compose-down -- -v' to wipe volumes)
	docker compose down

compose-logs:  ## Tail logs from the predictor container
	docker compose logs -f wonderwall-predictor

demo: install preflight distill-stub label-regimes train-stub eval-stub sweep-stub  ## Full plumbing demo: produces reports/demo.html
	$(PY) scripts/render_demo_report.py \
		--eval-json $(DEV_REPORT) \
		--sweep-csv $(DEV_SWEEP) \
		--out reports/demo.html
	@echo ""
	@echo "Demo complete. Open reports/demo.html in a browser."

serve: install  ## Run the KServe predictor locally on PORT (default 8080)
	WONDERWALL_ADAPTER_CHECKPOINT=$(DEV_CKPT) \
	WONDERWALL_ADAPTER_CONFIG=configs/adapter_default.yaml \
	WONDERWALL_LLM_CONFIG=configs/llm_gemma4.yaml \
	KIRK_BACKEND=stub \
	$(PY) -m scripts.serve_kserve

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

image:  ## Build the OpenShift image with podman
	podman build -t $(IMAGE):$(TAG) -f deploy/openshift/Dockerfile .

push: image  ## Push the image to quay.io
	podman push $(IMAGE):$(TAG)

# ---------------------------------------------------------------------------
# OpenShift apply / delete
# ---------------------------------------------------------------------------

openshift-prereqs:  ## Apply Namespace + ServiceAccount + Strimzi Kafka cluster
	oc apply -f deploy/openshift/0-namespace.yaml
	oc apply -f deploy/openshift/1-strimzi-cluster.yaml
	@echo ""
	@echo "Waiting for Kafka cluster Ready..."
	oc wait --for=condition=Ready -n $(NS) kafka/ulysses-kafka --timeout=10m

openshift-apply: openshift-prereqs  ## Apply wonderwall workloads + monitoring
	oc apply -f deploy/openshift/wonderwall-inference.yaml
	oc apply -f deploy/openshift/wonderwall-streamer.yaml
	oc apply -f deploy/openshift/wonderwall-podmonitor.yaml
	oc apply -f deploy/grafana/wonderwall-dashboard.yaml
	@echo ""
	@echo "Status:"
	oc get inferenceservice -n $(NS)
	oc get deployment -n $(NS) -l app=wonderwall-streamer
	oc get podmonitor -n $(NS) -l app=wonderwall
	oc get grafanadashboard -n $(NS) wonderwall

openshift-delete:  ## Delete all wonderwall resources
	oc delete -f deploy/grafana/wonderwall-dashboard.yaml --ignore-not-found
	oc delete -f deploy/openshift/wonderwall-podmonitor.yaml --ignore-not-found
	oc delete -f deploy/openshift/wonderwall-streamer.yaml --ignore-not-found
	oc delete -f deploy/openshift/wonderwall-inference.yaml --ignore-not-found

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:  ## Remove venv, build artifacts, generated data
	rm -rf $(VENV) build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf data/ reports/ checkpoints/
