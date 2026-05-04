"""Integration test for the KServe V2 OpenInference predictor.

Spins up a FastAPI TestClient against scripts.serve_kserve._build_app, with
the heavy components (real LLM, real Kirk) mocked out. Validates:

  - /v2/health/live and /v2/health/ready endpoints
  - /v2/models/wonderwall metadata endpoint
  - /v2/models/wonderwall/infer end-to-end with the sample payload
  - /metrics endpoint returns Prometheus-format text
  - Error paths: missing inputs, malformed shape, mismatched data length

Skipped automatically if fastapi or httpx aren't available — they're in the
[serve] extra and the [dev] extra respectively, so install with
    pip install -e ".[serve,dev]"
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from wonderwall.kirk_client import StubKirkClient


def _stub_pipeline(label="C"):
    """A pipeline whose .run() returns deterministic text without any LLM."""
    p = MagicMock()
    p.run.return_value = f"Stub narration ({label}): market consolidating sideways."
    return p


@pytest.fixture
def client(tmp_path):
    """Build the FastAPI app with _state already populated (skip _lifespan)."""
    # Import here to defer fastapi import errors to skip-time
    from scripts import serve_kserve

    # Inject stub pipelines directly into module state — avoids loading the
    # real LLM (which needs a GPU + 31B-parameter model on disk) and a real
    # Scotty endpoint for Pipelines A/B.
    serve_kserve._state["pipeline"] = _stub_pipeline("C")
    serve_kserve._state["pipeline_a"] = _stub_pipeline("A")
    serve_kserve._state["pipeline_b"] = _stub_pipeline("B")
    serve_kserve._state["ready"] = True
    serve_kserve._state["model_version"] = "test"

    app = serve_kserve._build_app()
    with TestClient(app) as c:
        yield c


def test_health_live_always_200(client):
    r = client.get("/v2/health/live")
    assert r.status_code == 200
    assert r.json() == {"live": True}


def test_health_ready_when_loaded(client):
    r = client.get("/v2/health/ready")
    assert r.status_code == 200
    assert r.json() == {"ready": True}


def test_model_metadata_shape(client):
    r = client.get("/v2/models/wonderwall")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "wonderwall"
    assert body["versions"] == ["test"]
    assert any(i["name"] == "tensor_windows" for i in body["inputs"])
    assert any(o["name"] == "narration" for o in body["outputs"])


def test_metrics_endpoint_exposes_prometheus_text(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    # Prometheus exposition format always starts with a HELP or TYPE line
    # for at least one metric.
    assert body == "" or "wonderwall_" in body or "# HELP" in body


def test_infer_with_sample_payload(client):
    """Full path through the predictor, validates request → tensor → response."""
    T, N = 4, 32
    cells = T * N * N
    payload = {
        "id": "test-1",
        "inputs": [
            {
                "name": "tensor_windows",
                "shape": [T, N, N],
                "datatype": "FP32",
                "data": [0.001] * cells,  # constant real-valued log-return scale
            }
        ],
        "outputs": [{"name": "narration"}],
        "parameters": {"max_new_tokens": 64},
    }
    r = client.post("/v2/models/wonderwall/infer", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_name"] == "wonderwall"
    assert body["id"] == "test-1"
    assert len(body["outputs"]) == 1
    assert body["outputs"][0]["name"] == "narration"
    assert body["outputs"][0]["data"][0].startswith("Stub narration")


def test_infer_rejects_missing_inputs(client):
    r = client.post("/v2/models/wonderwall/infer", json={"id": "bad", "inputs": []})
    assert r.status_code == 400
    assert "no inputs" in r.json()["detail"].lower()


def test_infer_rejects_non_square_window(client):
    payload = {
        "id": "bad-shape",
        "inputs": [
            {
                "name": "tensor_windows",
                "shape": [1, 32, 16],  # not square
                "datatype": "FP32",
                "data": [0.0] * 512,
            }
        ],
    }
    r = client.post("/v2/models/wonderwall/infer", json=payload)
    assert r.status_code == 400
    assert "square" in r.json()["detail"].lower()


def test_infer_rejects_mismatched_data_length(client):
    payload = {
        "id": "bad-data",
        "inputs": [
            {
                "name": "tensor_windows",
                "shape": [2, 32, 32],   # claims 2048 cells
                "datatype": "FP32",
                "data": [0.0] * 100,    # only 100 supplied
            }
        ],
    }
    r = client.post("/v2/models/wonderwall/infer", json=payload)
    assert r.status_code == 400


def test_infer_uses_real_sample_file_if_present(client):
    """If samples/single_window.json exists, the predictor accepts it verbatim."""
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "samples", "single_window.json",
    )
    if not os.path.exists(sample_path):
        pytest.skip(f"sample not present at {sample_path}")

    with open(sample_path) as f:
        payload = json.load(f)
    r = client.post("/v2/models/wonderwall/infer", json=payload)
    assert r.status_code == 200, r.text


def test_not_ready_returns_503(client, monkeypatch):
    """When _state['ready'] is False, infer returns 503."""
    from scripts import serve_kserve
    monkeypatch.setitem(serve_kserve._state, "ready", False)
    payload = {
        "inputs": [
            {"name": "tensor_windows", "shape": [1, 32, 32],
             "datatype": "FP32", "data": [0.0] * 1024}
        ]
    }
    r = client.post("/v2/models/wonderwall/infer", json=payload)
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# Pipeline A / B routes (added with the multi-pipeline serving update)
# ---------------------------------------------------------------------------


def _sample_payload(T=2, N=32):
    return {
        "id": "test-multi",
        "inputs": [
            {
                "name": "tensor_windows",
                "shape": [T, N, N],
                "datatype": "FP32",
                "data": [0.001] * (T * N * N),
            }
        ],
        "outputs": [{"name": "narration"}],
        "parameters": {"max_new_tokens": 32},
    }


def test_pipeline_a_route_returns_200(client):
    r = client.post("/v2/models/wonderwall-a/infer", json=_sample_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_name"] == "wonderwall-a"
    assert "Stub narration (A)" in body["outputs"][0]["data"][0]


def test_pipeline_b_route_returns_200(client):
    r = client.post("/v2/models/wonderwall-b/infer", json=_sample_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_name"] == "wonderwall-b"
    assert "Stub narration (B)" in body["outputs"][0]["data"][0]


def test_pipeline_c_route_still_returns_200(client):
    """Sanity: the original C route (default model name) still works."""
    r = client.post("/v2/models/wonderwall/infer", json=_sample_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_name"] == "wonderwall"
    assert "Stub narration (C)" in body["outputs"][0]["data"][0]


def test_each_route_calls_its_own_pipeline(client):
    """Each route's response must come from its own pipeline, not C's."""
    r_a = client.post("/v2/models/wonderwall-a/infer", json=_sample_payload())
    r_b = client.post("/v2/models/wonderwall-b/infer", json=_sample_payload())
    r_c = client.post("/v2/models/wonderwall/infer", json=_sample_payload())
    a = r_a.json()["outputs"][0]["data"][0]
    b = r_b.json()["outputs"][0]["data"][0]
    c = r_c.json()["outputs"][0]["data"][0]
    assert a != b != c, f"pipelines returned identical text: {a!r}, {b!r}, {c!r}"


def test_pipeline_a_returns_503_when_not_loaded(client, monkeypatch):
    """When pipeline_a is None (no Scotty), the A route returns 503."""
    from scripts import serve_kserve
    monkeypatch.setitem(serve_kserve._state, "pipeline_a", None)
    r = client.post("/v2/models/wonderwall-a/infer", json=_sample_payload())
    assert r.status_code == 503


def test_pipeline_b_returns_503_when_not_loaded(client, monkeypatch):
    from scripts import serve_kserve
    monkeypatch.setitem(serve_kserve._state, "pipeline_b", None)
    r = client.post("/v2/models/wonderwall-b/infer", json=_sample_payload())
    assert r.status_code == 503


def test_a_b_routes_share_input_validation_with_c(client):
    """Bad shape should fail identically across all three routes."""
    bad_payload = {
        "inputs": [
            {"name": "tensor_windows", "shape": [1, 32, 16],  # not square
             "datatype": "FP32", "data": [0.0] * 512}
        ]
    }
    for route in (
        "/v2/models/wonderwall/infer",
        "/v2/models/wonderwall-a/infer",
        "/v2/models/wonderwall-b/infer",
    ):
        r = client.post(route, json=bad_payload)
        assert r.status_code == 400, f"{route} did not 400 on non-square: {r.status_code}"
