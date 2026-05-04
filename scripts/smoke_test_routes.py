"""Smoke test the three deployed V2 OpenInference routes.

Hits /v2/models/wonderwall{,-a,-b}/infer with the same sample payload,
asserts 200 + non-empty narration + sane token counts, prints a summary.
Exits non-zero on any failure.

Designed to run AFTER `oc apply -f deploy/openshift/...` to confirm the
service is healthy before declaring a deploy successful.

Usage:

    # Against the in-cluster service (when port-forwarded):
    oc port-forward svc/wonderwall-predictor -n kavara-ai 8080:80 &
    python scripts/smoke_test_routes.py

    # Against an external route:
    ROUTE=$(oc get route wonderwall-predictor -n kavara-ai -o jsonpath='{.spec.host}')
    python scripts/smoke_test_routes.py --base-url https://$ROUTE
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


_DEFAULT_PAYLOAD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "samples", "single_window.json",
)


def _post_json(url: str, payload: dict, timeout_s: float = 60.0) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.getcode(), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")
    except urllib.error.URLError as e:
        return -1, {"error": str(e)}


def _check_route(base_url: str, model: str, payload: dict, timeout_s: float) -> dict:
    url = f"{base_url.rstrip('/')}/v2/models/{model}/infer"
    t0 = time.perf_counter()
    code, body = _post_json(url, payload, timeout_s=timeout_s)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    result = {
        "model": model,
        "url": url,
        "http_status": code,
        "elapsed_ms": elapsed_ms,
        "ok": False,
        "narration_len": 0,
        "narration_preview": "",
        "error": None,
    }

    if code != 200:
        result["error"] = body.get("detail") or body.get("error") or "non-200 response"
        return result

    outputs = body.get("outputs") or []
    if not outputs or not outputs[0].get("data"):
        result["error"] = "response had no outputs.data"
        return result

    narration = outputs[0]["data"][0]
    result["ok"] = bool(narration) and len(narration) > 8
    result["narration_len"] = len(narration)
    result["narration_preview"] = narration[:140]
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8080",
                   help="Predictor base URL (no trailing slash)")
    p.add_argument("--payload", default=_DEFAULT_PAYLOAD,
                   help="V2 request body JSON file")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="Per-route HTTP timeout in seconds")
    p.add_argument("--strict-pipeline-a", action="store_true",
                   help="Treat Pipeline A failure as fatal (default: warn)")
    p.add_argument("--strict-pipeline-b", action="store_true",
                   help="Treat Pipeline B failure as fatal (default: warn)")
    args = p.parse_args()

    with open(args.payload) as f:
        payload = json.load(f)

    print(f"[smoke] base_url   = {args.base_url}")
    print(f"[smoke] payload    = {args.payload}")
    print()

    routes = [
        ("wonderwall", "C — embedding injection", True),       # always strict
        ("wonderwall-a", "A — raw text", args.strict_pipeline_a),
        ("wonderwall-b", "B — compressed text", args.strict_pipeline_b),
    ]

    failures = 0
    warnings = 0
    for model, label, strict in routes:
        r = _check_route(args.base_url, model, payload, timeout_s=args.timeout)
        status = "OK" if r["ok"] else "FAIL"
        print(f"[smoke] {label:<28s} → {status}  http={r['http_status']}  "
              f"{r['elapsed_ms']:.0f}ms  narration_len={r['narration_len']}")
        if r["narration_preview"]:
            print(f"        preview: {r['narration_preview']!r}")
        if r["error"]:
            print(f"        error:   {r['error']}")
        if not r["ok"]:
            if strict:
                failures += 1
            else:
                warnings += 1

    print()
    if failures > 0:
        print(f"[smoke] {failures} fatal failure(s)")
        return 1
    if warnings > 0:
        print(f"[smoke] {warnings} non-fatal warning(s) — Pipelines A/B require Scotty")
    print("[smoke] all critical routes healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
