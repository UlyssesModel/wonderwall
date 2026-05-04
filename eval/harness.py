"""Eval harness: run all three pipelines on the same input streams, emit records.

Pipeline A is the tokenized-raw-data baseline (worst case in tokens).
Pipeline B is the production compressed-text path via Scotty.
Pipeline C is the new embedding-injection path via HF Transformers.

Quality is measured via ROUGE-L vs a teacher narration (the same one used to
distill the adapter). Cost is computed via the PricingModel in metrics.py.

The harness is designed to be skippable per-pipeline so you can run B alone
on a CPU-only machine, or skip A when you don't want to burn frontier-LLM
budget on the worst-case baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from wonderwall.adapter import KirkProjectionAdapter
from wonderwall.distill import DistillationItem
from wonderwall.injection import (
    EmbeddingInjectionLLM,
    LLMConfig,
    ScottyClient,
    ScottyConfig,
)
from wonderwall.interfaces import KirkClient
from wonderwall.pipeline import (
    CompressedTextPipeline,
    EmbeddingInjectionPipeline,
)
from .baselines import RawTextPipeline, render_raw_stream_as_text
from .hmm_baseline import GaussianHMM, classify_stream_with_hmm
from .metrics import EvalRecord, PricingModel, estimate_cost, rouge_l


@dataclass
class HarnessConfig:
    run_pipeline_a: bool = True
    run_pipeline_b: bool = True
    run_pipeline_c: bool = True
    run_hmm_baseline: bool = False  # Pipeline D — populated when an HMM is provided
    pricing: PricingModel = PricingModel()
    max_new_tokens: int = 256
    pipeline_a_tier: str = "frontier"
    pipeline_b_tier: str = "small"
    pipeline_c_tier: str = "small"


class EvalHarness:
    """Runs all three pipelines on the same input set."""

    def __init__(
        self,
        kirk: KirkClient,
        adapter: Optional[KirkProjectionAdapter] = None,
        llm: Optional[EmbeddingInjectionLLM] = None,
        scotty: Optional[ScottyClient] = None,
        hmm: Optional[GaussianHMM] = None,
        config: HarnessConfig = HarnessConfig(),
    ):
        self.kirk = kirk
        self.adapter = adapter
        self.llm = llm
        self.scotty = scotty
        self.hmm = hmm
        self.config = config

        # Wire up the pipeline objects on demand
        self.pipe_a = (
            RawTextPipeline(scotty=scotty)
            if config.run_pipeline_a and scotty is not None
            else None
        )
        self.pipe_b = (
            CompressedTextPipeline(kirk=kirk, scotty=scotty)
            if config.run_pipeline_b and scotty is not None
            else None
        )
        self.pipe_c = (
            EmbeddingInjectionPipeline(kirk=kirk, adapter=adapter, llm=llm)
            if config.run_pipeline_c and adapter is not None and llm is not None
            else None
        )

    def _approx_tokens(self, text: str) -> int:
        """Cheap token approximation when we don't have a tokenizer to hand.

        4 chars ≈ 1 token rule of thumb for English. For exact counts use the
        actual tokenizer in `EmbeddingInjectionLLM.tokenizer`.
        """
        return max(1, len(text) // 4)

    def _exact_tokens(self, text: str) -> int:
        """Exact token count using the LLM's tokenizer when available."""
        if self.llm is None:
            return self._approx_tokens(text)
        return len(self.llm.tokenizer(text).input_ids)

    def evaluate_one(
        self,
        item: DistillationItem,
    ) -> list[EvalRecord]:
        """Run all configured pipelines on one input stream + gold target."""
        records: list[EvalRecord] = []

        # ---- Pipeline A: raw text ----
        if self.pipe_a is not None:
            user_text = render_raw_stream_as_text(item.tensors)
            t0 = time.perf_counter()
            output = self.pipe_a.run(item.tensors, max_tokens=self.config.max_new_tokens)
            t1 = time.perf_counter()
            in_toks = self._exact_tokens(user_text)
            out_toks = self._exact_tokens(output)
            cost = estimate_cost(in_toks, out_toks, self.config.pipeline_a_tier, self.config.pricing)
            records.append(
                EvalRecord(
                    pipeline="A_tokenized",
                    input_token_count=in_toks,
                    output_token_count=out_toks,
                    prefill_latency_ms=(t1 - t0) * 1000,
                    decode_latency_ms=0.0,  # streaming not split here; conflate with prefill
                    end_to_end_latency_ms=(t1 - t0) * 1000,
                    output_text=output,
                    rouge_l=rouge_l(output, item.target_text),
                    cost_usd=cost,
                    notes="raw-text baseline",
                )
            )

        # ---- Pipeline B: compressed text ----
        if self.pipe_b is not None:
            t0 = time.perf_counter()
            output = self.pipe_b.run(item.tensors, max_tokens=self.config.max_new_tokens)
            t1 = time.perf_counter()
            kos = self.kirk.infer_stream(item.tensors)
            from wonderwall.pipeline import render_kirk_output_as_text
            rendered = "\n\n".join(render_kirk_output_as_text(ko) for ko in kos)
            in_toks = self._exact_tokens(rendered)
            out_toks = self._exact_tokens(output)
            cost = estimate_cost(in_toks, out_toks, self.config.pipeline_b_tier, self.config.pricing)
            records.append(
                EvalRecord(
                    pipeline="B_compressed_text",
                    input_token_count=in_toks,
                    output_token_count=out_toks,
                    prefill_latency_ms=(t1 - t0) * 1000,
                    decode_latency_ms=0.0,
                    end_to_end_latency_ms=(t1 - t0) * 1000,
                    output_text=output,
                    rouge_l=rouge_l(output, item.target_text),
                    cost_usd=cost,
                    notes="compressed-text via Scotty (production today)",
                )
            )

        # ---- Pipeline C: embedding injection ----
        if self.pipe_c is not None:
            t0 = time.perf_counter()
            output = self.pipe_c.run(item.tensors, max_new_tokens=self.config.max_new_tokens)
            t1 = time.perf_counter()
            soft_token_count = self.pipe_c.soft_token_count(item.tensors)
            out_toks = self._exact_tokens(output)
            # Cost calc treats soft tokens as input tokens — they consume the
            # same attention compute. The instruction prefix/suffix add a small
            # constant; ignored here for clarity.
            cost = estimate_cost(soft_token_count, out_toks, self.config.pipeline_c_tier, self.config.pricing)
            records.append(
                EvalRecord(
                    pipeline="C_embedding_injection",
                    input_token_count=soft_token_count,
                    output_token_count=out_toks,
                    prefill_latency_ms=(t1 - t0) * 1000,
                    decode_latency_ms=0.0,
                    end_to_end_latency_ms=(t1 - t0) * 1000,
                    output_text=output,
                    rouge_l=rouge_l(output, item.target_text),
                    cost_usd=cost,
                    notes="embedding-injection (new build)",
                )
            )

        # ---- Pipeline D: HMM regime baseline (for regime_correct comparison) ----
        if self.config.run_hmm_baseline and self.hmm is not None:
            t0 = time.perf_counter()
            states, names = classify_stream_with_hmm(self.kirk, self.hmm, item.tensors)
            t1 = time.perf_counter()
            # Use the *last-step* regime as the headline label for the stream
            pred_regime = names[-1] if names else "unknown"
            gold_regime = (item.metadata or {}).get("gold_regime") if hasattr(item, "metadata") else None
            regime_correct = (
                (pred_regime == gold_regime) if gold_regime is not None else None
            )
            # The HMM is essentially free in cost (CPU-only, NumPy) — use a
            # small token-equivalent (4 features × T windows) so the cost
            # column has a representative non-zero value.
            input_eq = 4 * len(item.tensors)
            records.append(
                EvalRecord(
                    pipeline="D_hmm_baseline",
                    input_token_count=input_eq,
                    output_token_count=len(pred_regime),  # ~6–10 chars
                    prefill_latency_ms=(t1 - t0) * 1000,
                    decode_latency_ms=0.0,
                    end_to_end_latency_ms=(t1 - t0) * 1000,
                    output_text=pred_regime,
                    rouge_l=None,  # not applicable — single-label classification
                    regime_correct=regime_correct,
                    cost_usd=0.0,  # HMM is local CPU; effectively free
                    notes=f"HMM regime classifier ({len(self.hmm.state_names)} states)",
                )
            )

        # If gold regime labels are present, also tag the narration pipelines'
        # regime_correct field by detecting which regime word appears in the
        # output narration. This is a lightweight string-match check; for
        # rigorous evaluation use a separate regime-extraction step.
        if self.hmm is not None and item.tensors:
            gold = (item.metadata or {}).get("gold_regime") if hasattr(item, "metadata") else None
            if gold is not None:
                for r in records:
                    if r.pipeline.startswith(("A_", "B_", "C_")) and r.output_text:
                        text_lower = r.output_text.lower()
                        # Pick the first state name that appears in the text
                        detected = next(
                            (s for s in self.hmm.state_names if s in text_lower),
                            None,
                        )
                        r.regime_correct = (detected == gold) if detected else False

        return records

    def evaluate_set(
        self, items: Sequence[DistillationItem]
    ) -> list[EvalRecord]:
        """Run all pipelines on a list of distilled items, return flat record list."""
        all_records: list[EvalRecord] = []
        for i, item in enumerate(items):
            print(f"[eval] {i + 1}/{len(items)}")
            all_records.extend(self.evaluate_one(item))
        return all_records
