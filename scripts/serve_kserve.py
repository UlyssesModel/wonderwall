"""KServe V2 OpenInference predictor for wonderwall.

Implements the V2 protocol that Mike validated bit-exact in
tiberius-openshift commit f02fbc3:

    GET  /v2/health/live    → 200 if process is alive
    GET  /v2/health/ready   → 200 once model + adapter loaded
    POST /v2/models/wonderwall/infer
        body: {
            "inputs": [
                {
                    "name": "tensor_windows",
                    "shape": [T, N, N],
                    "datatype": "FP32",
                    "data": [...flat row-major...]
                }
            ],
            "outputs": [{"name": "narration"}]
        }
        response: {
            "model_name": "wonderwall",
            "model_version": "...",
            "id": "...",
            "outputs": [
                {"name": "narration", "shape": [1], "datatype": "BYTES",
                 "data": ["The market is showing..."]}
            ]
        }

Deps: fastapi, uvicorn. Both pulled in by the [llm] extra in pyproject.toml.

Standalone runner:
    python -m scripts.serve_kserve
    SCOTTY_BASE_URL=http://localhost:11434 python -m scripts.serve_kserve

Loaded via env vars (set by the KServe manifest):
    WONDERWALL_ADAPTER_CHECKPOINT
    WONDERWALL_ADAPTER_CONFIG
    WONDERWALL_LLM_CONFIG
    KIRK_BACKEND        pipeline | subprocess | stub
    SCOTTY_BASE_URL     ignored when KIRK_BACKEND != stub
    PORT                default 8080
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
import yaml

from wonderwall.adapter import KirkProjectionAdapter
from wonderwall.injection import (
    EmbeddingInjectionLLM,
    LLMConfig,
    ScottyClient,
    ScottyConfig,
)
from wonderwall.interfaces import AdapterConfig, KirkClient, KirkMode
from wonderwall.kirk_client import (
    KirkPipelineClient,
    KirkSubprocessClient,
    StubKirkClient,
)
from wonderwall.logging_config import setup_logging
from wonderwall.metrics_export import metrics
from wonderwall.pipeline import (
    CompressedTextPipeline,
    EmbeddingInjectionPipeline,
)

# Eval baselines provide Pipeline A
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from eval.baselines import RawTextPipeline  # noqa: E402


logger = setup_logging("wonderwall.serve")


_state: dict[str, Any] = {
    "ready": False,
    "pipeline": None,           # Pipeline C — embedding injection (default)
    "pipeline_a": None,         # Pipeline A — raw text via Scotty
    "pipeline_b": None,         # Pipeline B — compressed text via Scotty
    "model_version": "v0.1",
    "started_at": time.time(),
}


def _load_pipelines() -> dict[str, Any]:
    """Construct all three pipelines from env config.

    Returns a dict with keys {'C', 'A', 'B'} mapping to the corresponding
    pipeline objects. Any pipeline whose dependencies aren't available (e.g.
    no Scotty endpoint reachable) maps to None — the route layer handles
    the not-available case with a 503.
    """
    adapter_cfg_path = os.environ["WONDERWALL_ADAPTER_CONFIG"]
    llm_cfg_path = os.environ["WONDERWALL_LLM_CONFIG"]
    ckpt_path = os.environ["WONDERWALL_ADAPTER_CHECKPOINT"]
    backend = os.environ.get("KIRK_BACKEND", "pipeline").lower()
    scotty_url = os.environ.get("SCOTTY_BASE_URL", "http://127.0.0.1:11434")
    scotty_model = os.environ.get("SCOTTY_MODEL", "gemma4:31b")

    with open(adapter_cfg_path) as f:
        adapter_cfg = AdapterConfig(**yaml.safe_load(f))
    with open(llm_cfg_path) as f:
        llm_cfg = LLMConfig(**yaml.safe_load(f))

    # Kirk client (shared across all pipelines)
    kirk: KirkClient
    if backend == "pipeline":
        kirk = KirkPipelineClient()
    elif backend == "subprocess":
        kirk = KirkSubprocessClient(n=adapter_cfg.n)
    elif backend == "stub":
        kirk = StubKirkClient(n=adapter_cfg.n, use_complex=adapter_cfg.use_complex)
    else:
        raise ValueError(f"unknown KIRK_BACKEND: {backend!r}")

    # Pipeline C — embedding injection (the new path)
    adapter = KirkProjectionAdapter(adapter_cfg)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    llm = EmbeddingInjectionLLM(llm_cfg)
    adapter = adapter.to(llm.config.device).eval()
    pipe_c = EmbeddingInjectionPipeline(kirk=kirk, adapter=adapter, llm=llm)

    # Pipelines A and B — both call Scotty. If Scotty is unreachable they
    # still construct (no-op until first request); the predictor routes
    # surface a 5xx if the chat call fails.
    scotty = ScottyClient(ScottyConfig(base_url=scotty_url, model=scotty_model))
    pipe_a = RawTextPipeline(scotty=scotty)
    pipe_b = CompressedTextPipeline(kirk=kirk, scotty=scotty)

    logger.info(
        "pipelines ready: backend=%s n=%d hidden=%d adapter_params=%d scotty=%s",
        backend, adapter_cfg.n, adapter_cfg.llm_hidden_dim,
        adapter.num_trainable_parameters, scotty_url,
    )
    return {"A": pipe_a, "B": pipe_b, "C": pipe_c}


_REQUIRED_ENV = (
    "WONDERWALL_ADAPTER_CONFIG",
    "WONDERWALL_LLM_CONFIG",
    "WONDERWALL_ADAPTER_CHECKPOINT",
)


@asynccontextmanager
async def _lifespan(app):
    if not all(k in os.environ for k in _REQUIRED_ENV):
        # No production config — caller (tests, dev shell) is responsible for
        # populating _state directly. Skip auto-load so we don't clobber it.
        yield
        return
    try:
        pipes = _load_pipelines()
        _state["pipeline"] = pipes["C"]
        _state["pipeline_a"] = pipes["A"]
        _state["pipeline_b"] = pipes["B"]
        _state["ready"] = True
        metrics.ready.set(1)
    except Exception:
        logger.exception("pipeline load failed; staying not-ready")
        _state["ready"] = False
        metrics.ready.set(0)
    yield


def _build_app():
    try:
        from fastapi import Body, FastAPI, HTTPException  # type: ignore
        from fastapi.responses import JSONResponse, Response  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "fastapi is required for serve_kserve. Install with `pip install fastapi uvicorn`."
        ) from e

    app = FastAPI(title="wonderwall", lifespan=_lifespan)

    # ---- Prometheus metrics endpoint ----
    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(
            content=metrics.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/v2/health/live")
    async def live():
        return {"live": True}

    @app.get("/v2/health/ready")
    async def ready():
        return JSONResponse(
            status_code=200 if _state["ready"] else 503,
            content={"ready": _state["ready"]},
        )

    @app.get("/v2/models/wonderwall")
    async def model_info():
        return {
            "name": "wonderwall",
            "versions": [_state["model_version"]],
            "platform": "wonderwall",
            "inputs": [
                {
                    "name": "tensor_windows",
                    "datatype": "FP32",
                    "shape": [-1, -1, -1],  # T, N, N — N pinned by adapter config
                }
            ],
            "outputs": [{"name": "narration", "datatype": "BYTES", "shape": [1]}],
        }

    def _parse_tensor_windows(body: dict) -> tuple[list[torch.Tensor], int, int, str, int]:
        """Parse a V2 request body, return (tensors, T, N, request_id, max_new_tokens)."""
        request_id = body.get("id", str(uuid.uuid4()))
        inputs = body.get("inputs", [])
        if not inputs:
            raise HTTPException(status_code=400, detail="no inputs")
        tw = next((i for i in inputs if i.get("name") == "tensor_windows"), None)
        if tw is None:
            raise HTTPException(status_code=400, detail="missing 'tensor_windows' input")
        shape = tw.get("shape")
        data = tw.get("data")
        if not (isinstance(shape, list) and len(shape) == 3):
            raise HTTPException(status_code=400, detail="shape must be [T, N, N]")
        T, N1, N2 = shape
        if N1 != N2:
            raise HTTPException(status_code=400, detail="windows must be square")
        expected = T * N1 * N2
        if len(data) != expected:
            raise HTTPException(
                status_code=400,
                detail=f"data length {len(data)} != T*N*N = {expected}",
            )
        arr = torch.tensor(data, dtype=torch.float32).reshape(T, N1, N2)
        tensors = [arr[t] for t in range(T)]
        max_new = int(body.get("parameters", {}).get("max_new_tokens", 256))
        return tensors, T, N1, request_id, max_new

    def _build_v2_response(model_name: str, request_id: str, output_text: str) -> dict:
        return {
            "model_name": model_name,
            "model_version": _state["model_version"],
            "id": request_id,
            "outputs": [
                {
                    "name": "narration",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": [output_text],
                }
            ],
        }

    async def _run_pipeline(
        body: dict,
        pipe_state_key: str,
        pipeline_label: str,
        model_name: str,
        max_new_tokens_kw: str,
        soft_token_for_C: bool = False,
    ):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail="pipelines not ready")
        pipe = _state.get(pipe_state_key)
        if pipe is None:
            raise HTTPException(
                status_code=503,
                detail=f"pipeline {pipeline_label} not loaded",
            )
        tensors, T, N, request_id, max_new = _parse_tensor_windows(body)

        metrics.inference_started(pipeline_label)
        metrics.batch_size.set(T)
        try:
            with metrics.inference_timer(pipeline_label):
                # CompressedTextPipeline / RawTextPipeline expose `max_tokens`,
                # EmbeddingInjectionPipeline exposes `max_new_tokens`. Use the
                # right kwarg per pipeline.
                output_text = pipe.run(tensors, **{max_new_tokens_kw: max_new})
        except Exception as e:
            metrics.inference_failed(pipeline_label, e)
            logger.exception("%s inference failed", pipeline_label)
            raise HTTPException(status_code=500, detail=f"inference error: {e}")

        soft_tokens = T * (N + 2) if soft_token_for_C else None
        metrics.inference_completed(pipeline_label, soft_tokens=soft_tokens)
        return _build_v2_response(model_name, request_id, output_text)

    @app.post("/v2/models/wonderwall/infer")
    async def infer_c(body: dict = Body(...)):
        """Pipeline C — embedding injection (the new build, default route)."""
        return await _run_pipeline(
            body,
            pipe_state_key="pipeline",
            pipeline_label="C_embedding_injection",
            model_name="wonderwall",
            max_new_tokens_kw="max_new_tokens",
            soft_token_for_C=True,
        )

    @app.post("/v2/models/wonderwall-a/infer")
    async def infer_a(body: dict = Body(...)):
        """Pipeline A — raw text via Scotty (worst-case baseline)."""
        return await _run_pipeline(
            body,
            pipe_state_key="pipeline_a",
            pipeline_label="A_tokenized",
            model_name="wonderwall-a",
            max_new_tokens_kw="max_tokens",
        )

    @app.post("/v2/models/wonderwall-b/infer")
    async def infer_b(body: dict = Body(...)):
        """Pipeline B — compressed text via Scotty (production today)."""
        return await _run_pipeline(
            body,
            pipe_state_key="pipeline_b",
            pipeline_label="B_compressed_text",
            model_name="wonderwall-b",
            max_new_tokens_kw="max_tokens",
        )

    return app


def main():
    try:
        import uvicorn  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "uvicorn is required to run the KServe predictor. "
            "Install with `pip install uvicorn`."
        ) from e

    app = _build_app()
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
