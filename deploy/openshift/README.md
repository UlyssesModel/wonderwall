# OpenShift deployment

Two manifests, two postures:

- **`wonderwall-inference.yaml`** — KServe `InferenceService`, Red Hat AI
  catalog-publishable. V2 OpenInference protocol, RawDeployment mode (no
  Knative/Istio). Mirrors the pattern used by `tiberius-openshift` commit
  `f02fbc3` for the Phase 3A KServe wrapping.

- **`wonderwall-streamer.yaml`** — `Deployment` that consumes
  `ulysses.tensor.frames.<cadence>` from Strimzi Kafka, runs the full
  Kirk → adapter → LLM pipeline, and publishes narrations to
  `ulysses.narrations.<cadence>`. Mirrors `uhura/deploy/openshift/uhura-streamer.yaml`
  per-cadence pattern. Add more Deployments for additional cadences.

## Prerequisites

The same cluster posture used by `tiberius-openshift` and `uhura`:

- OCP 4.21+ (4.21.9 tested per Red Hat collab page)
- Strimzi 4.1.0 with mTLS listener
- KServe (RawDeployment mode)
- NVIDIA GPU operator (Gemma 4 31B forward pass needs at least one H100/A100)
- A `kavara-ai` namespace with the standard RBAC

## Build the image

```bash
podman build -t quay.io/kavara/wonderwall:v0.1 -f deploy/openshift/Dockerfile .
podman push quay.io/kavara/wonderwall:v0.1
```

## Deploy

```bash
oc apply -f deploy/openshift/wonderwall-inference.yaml
oc apply -f deploy/openshift/wonderwall-streamer.yaml

# Sanity
oc get inferenceservice -n kavara-ai
oc get deployment -n kavara-ai -l app=wonderwall-streamer
oc get kafkatopic -n kavara-ai | grep ulysses
oc get kafkauser -n kavara-ai | grep wonderwall
```

## Test the InferenceService

V2 OpenInference protocol, same as `tiberius-openshift`'s `ulysses-sor-inference`:

```bash
ROUTE=$(oc get route wonderwall-predictor -n kavara-ai -o jsonpath='{.spec.host}')
curl -X POST "https://$ROUTE/v2/models/wonderwall/infer" \
  -H "Content-Type: application/json" \
  -d @samples/single_window.json
```

## Open items before this is real

- **Container image not built yet.** `quay.io/kavara/wonderwall:v0.1` is a
  placeholder; build via the Dockerfile in this directory.
- **`scripts/serve_kserve.py` not implemented.** The Dockerfile `CMD` references
  it; needs to wrap `EmbeddingInjectionPipeline` in a V2 OpenInference HTTP
  server. ~50 lines, mirrors the consumer.py pattern from tiberius-openshift.
- **`scripts/stream_consumer.py` not implemented.** Mirrors uhura's
  `kafka_trades.py` adapter — confluent-kafka consumer + per-message inference.
- **GPU type pinned via NodeSelector.** Add `nodeSelector` blocks once the
  cluster's GPU node labels are known (typically
  `nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3`).
