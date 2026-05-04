# Local Docker dev stack

`compose.yaml` at the repo root defines a three-service stack you can stand
up locally with one command:

| Service | Image | Purpose |
| --- | --- | --- |
| `ollama` | `ollama/ollama:latest` | Local LLM, default `gemma2:2b` for speed |
| `redpanda` | `redpandadata/redpanda:latest` | Kafka-API-compatible single-node broker |
| `wonderwall-predictor` | built from `deploy/openshift/Dockerfile` | KServe V2 server, stub Kirk |

## First-run

```bash
# 1. Pull the small Gemma into Ollama (one-shot helper).
docker compose --profile setup run --rm ollama-pull

# 2. Start the stack.
docker compose up
```

## Hit the predictor

```bash
# Pipeline C — embedding injection (default route).
# Currently 503's because there's no checkpoint mounted; train one first.
curl -X POST http://localhost:8080/v2/models/wonderwall/infer \
     -H 'Content-Type: application/json' \
     -d @samples/single_window.json

# Pipeline B — compressed text via Ollama (works without a checkpoint).
curl -X POST http://localhost:8080/v2/models/wonderwall-b/infer \
     -H 'Content-Type: application/json' \
     -d @samples/single_window.json
```

## Train + serve a real Pipeline C

```bash
# On the host — produce a checkpoint via the make targets.
make train-stub                    # writes checkpoints/adapter.pt

# Restart the predictor — the bind mount picks it up automatically.
docker compose restart wonderwall-predictor

# Smoke test all three routes.
SMOKE_TEST_URL=http://127.0.0.1:8080 make smoke-test
```

## Tear down

```bash
docker compose down -v   # -v wipes the ollama-data volume too
```

## Notes

- The dev stack uses **stub Kirk**, not the real kirk-pipeline wheel.
  Pipeline-C inference works against the stub; numbers are meaningless
  but the plumbing is exercised end-to-end.
- The dev stack uses a **smaller Gemma** (`gemma2:2b` by default) to
  keep iteration time under a minute. Override via `SCOTTY_MODEL=...`.
- For real numbers, point `KIRK_BACKEND=pipeline` at a host with the
  production wheel and bind-mount it into the predictor container.
