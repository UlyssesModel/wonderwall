"""CLI for running the eval harness end-to-end.

Examples:

    # Pipeline B + C only (no frontier-LLM cost), against a small distilled set:
    python -m eval.runner \\
      --distilled data/distilled_dev.pt \\
      --adapter checkpoints/adapter.pt \\
      --no-pipeline-a \\
      --out reports/dev_eval.json

    # All three pipelines, full benchmark:
    python -m eval.runner \\
      --distilled data/distilled_test.pt \\
      --adapter checkpoints/adapter.pt \\
      --teacher-base-url https://api.anthropic.com/v1 \\
      --out reports/test_eval.json
"""
from __future__ import annotations

import argparse
import json

import torch

from wonderwall.adapter import KirkProjectionAdapter
from wonderwall.distill import load_distilled
from wonderwall.injection import (
    EmbeddingInjectionLLM,
    LLMConfig,
    ScottyClient,
    ScottyConfig,
)
from wonderwall.interfaces import AdapterConfig
from wonderwall.kirk_client import StubKirkClient

from .harness import EvalHarness, HarnessConfig
from .metrics import compute_metrics, dump_summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distilled", required=True, help="Path to distilled .pt file")
    p.add_argument("--adapter", default=None, help="Path to trained adapter .pt; required for Pipeline C")
    p.add_argument("--out", required=True, help="Output JSON for the summary")
    p.add_argument("--records-out", default=None, help="Optional JSONL for per-record output")

    p.add_argument("--no-pipeline-a", action="store_true")
    p.add_argument("--no-pipeline-b", action="store_true")
    p.add_argument("--no-pipeline-c", action="store_true")

    p.add_argument("--scotty-base-url", default="http://127.0.0.1:11434")
    p.add_argument("--scotty-model", default="gemma4:31b")

    p.add_argument("--llm-model-name", default="google/gemma-4-31b")
    p.add_argument("--llm-hidden-dim", type=int, default=5376)
    p.add_argument("--llm-device", default="cuda")

    p.add_argument("--n", type=int, default=16, help="Kirk window size (matches adapter config)")
    p.add_argument("--use-complex", action="store_true", help="Train/eval with complex Kirk outputs")

    p.add_argument("--use-stub-kirk", action="store_true",
                   help="Use the synthetic Kirk client. ON by default for plumbing eval.")

    p.add_argument("--max-new-tokens", type=int, default=256)

    args = p.parse_args()

    # --- Kirk ---
    if not args.use_stub_kirk:
        raise NotImplementedError(
            "Real Kirk client wiring is owned by Spencer's team. "
            "Pass --use-stub-kirk to run the harness against synthetic Kirk outputs."
        )
    kirk = StubKirkClient(n=args.n, use_complex=args.use_complex)

    # --- Scotty ---
    scotty = None
    if not (args.no_pipeline_a and args.no_pipeline_b):
        scotty = ScottyClient(ScottyConfig(base_url=args.scotty_base_url, model=args.scotty_model))

    # --- Adapter + LLM (Pipeline C only) ---
    adapter = None
    llm = None
    if not args.no_pipeline_c:
        if args.adapter is None:
            raise SystemExit("Pipeline C requires --adapter <path>")
        cfg = AdapterConfig(
            n=args.n,
            llm_hidden_dim=args.llm_hidden_dim,
            use_complex=args.use_complex,
        )
        adapter = KirkProjectionAdapter(cfg)
        ckpt = torch.load(args.adapter, weights_only=False)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        llm = EmbeddingInjectionLLM(
            LLMConfig(
                model_name=args.llm_model_name,
                hidden_dim=args.llm_hidden_dim,
                device=args.llm_device,
            )
        )
        adapter = adapter.to(args.llm_device)
        adapter.eval()

    items = load_distilled(args.distilled)
    print(f"[runner] loaded {len(items)} eval items from {args.distilled}")

    harness = EvalHarness(
        kirk=kirk,
        adapter=adapter,
        llm=llm,
        scotty=scotty,
        config=HarnessConfig(
            run_pipeline_a=not args.no_pipeline_a,
            run_pipeline_b=not args.no_pipeline_b,
            run_pipeline_c=not args.no_pipeline_c,
            max_new_tokens=args.max_new_tokens,
        ),
    )
    records = harness.evaluate_set(items)
    summary = compute_metrics(records)
    dump_summary(summary, args.out)
    print(f"[runner] summary written to {args.out}")
    print(json.dumps(summary, indent=2))

    if args.records_out:
        with open(args.records_out, "w") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")
        print(f"[runner] per-record output written to {args.records_out}")


if __name__ == "__main__":
    main()
