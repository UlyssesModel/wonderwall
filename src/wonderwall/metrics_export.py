"""Prometheus metrics for wonderwall.

Mirrors the metric naming conventions used by uhura and tiberius-openshift's
`ulysses-sor` consumer:

  ulysses_*_total                 — Counter
  ulysses_*_seconds_bucket        — Histogram
  ulysses_*_seconds_sum           — Histogram sum
  ulysses_*_seconds_count         — Histogram count
  ulysses_*                       — Gauge

The metrics are scraped by UWM Prometheus (User Workload Monitoring) per
the Red Hat collab page's `oc apply -f` deployment pattern. No additional
scrape config needed beyond the standard PodMonitor.

Usage:
    from wonderwall.metrics_export import metrics
    metrics.inference_started("C_embedding_injection")
    with metrics.inference_timer("C_embedding_injection"):
        narration = pipeline.run(tensors)
    metrics.inference_completed("C_embedding_injection",
                                soft_tokens=128, output_tokens=240)

Falls back to no-op when prometheus_client isn't installed (so unit tests
don't need to pull it in).
"""
from __future__ import annotations

import contextlib
import time
from typing import Any, Iterator, Optional

try:
    from prometheus_client import (  # type: ignore
        Counter,
        Gauge,
        Histogram,
        REGISTRY,
        generate_latest,
    )
    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _AVAILABLE = False


class _NoopMetric:
    """Stand-in when prometheus_client isn't installed."""

    def labels(self, *args, **kwargs):  # noqa
        return self

    def inc(self, *args, **kwargs):  # noqa
        pass

    def dec(self, *args, **kwargs):  # noqa
        pass

    def set(self, *args, **kwargs):  # noqa
        pass

    def observe(self, *args, **kwargs):  # noqa
        pass


class _Metrics:
    """All wonderwall metric families in one place."""

    def __init__(self):
        if _AVAILABLE:
            # Counters
            self.inferences_started = Counter(
                "wonderwall_inferences_started_total",
                "Total inference requests started",
                ["pipeline"],
            )
            self.inferences_completed = Counter(
                "wonderwall_inferences_completed_total",
                "Total inference requests completed successfully",
                ["pipeline"],
            )
            self.inferences_failed = Counter(
                "wonderwall_inferences_failed_total",
                "Total inference requests that raised exceptions",
                ["pipeline", "error_class"],
            )
            self.input_tokens_total = Counter(
                "wonderwall_input_tokens_total",
                "Cumulative input tokens (or soft-token equivalents) consumed",
                ["pipeline"],
            )
            self.output_tokens_total = Counter(
                "wonderwall_output_tokens_total",
                "Cumulative output tokens generated",
                ["pipeline"],
            )

            # Histograms — inference latency, prefill latency, decode latency
            self.inference_seconds = Histogram(
                "wonderwall_inference_seconds",
                "End-to-end inference latency",
                ["pipeline"],
                buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            )
            self.kirk_seconds = Histogram(
                "wonderwall_kirk_seconds",
                "Kirk-only forward-pass latency (Layer-1 + Layer-2)",
                buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            )
            self.adapter_seconds = Histogram(
                "wonderwall_adapter_seconds",
                "Projection adapter forward-pass latency",
                buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1),
            )
            self.llm_seconds = Histogram(
                "wonderwall_llm_seconds",
                "LLM generation latency",
                ["pipeline"],
                buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            )

            # Gauges
            self.batch_size = Gauge(
                "wonderwall_batch_size",
                "Most recent batch size (windows per inference call)",
            )
            self.entropy_last = Gauge(
                "wonderwall_entropy_last",
                "Most recent Kirk entropy value (nats)",
            )
            self.ready = Gauge(
                "wonderwall_ready",
                "1 if the pipeline is loaded and ready for inference, else 0",
            )

            # Training-loop metrics
            self.train_loss = Gauge(
                "wonderwall_train_loss",
                "Most recent training loss",
            )
            self.train_loss_avg = Gauge(
                "wonderwall_train_loss_avg",
                "Running average training loss for the current epoch",
            )
            self.train_steps_total = Counter(
                "wonderwall_train_steps_total",
                "Total training steps completed",
            )
            self.train_grad_norm = Gauge(
                "wonderwall_train_grad_norm",
                "L2 norm of adapter gradients on the most recent step",
            )
            self.train_step_seconds = Histogram(
                "wonderwall_train_step_seconds",
                "Per-step training time (forward + backward + optimizer)",
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )
            self.train_epoch = Gauge(
                "wonderwall_train_epoch",
                "Current training epoch (0-indexed)",
            )

            # Streamer metrics
            self.frames_consumed_total = Counter(
                "wonderwall_frames_consumed_total",
                "Total tensor frames consumed from the input Kafka topic",
                ["topic"],
            )
            self.narrations_produced_total = Counter(
                "wonderwall_narrations_produced_total",
                "Total narration messages produced to the output topic",
                ["topic"],
            )
            self.frames_skipped_total = Counter(
                "wonderwall_frames_skipped_total",
                "Frames skipped due to malformed payload",
                ["topic", "reason"],
            )
            self.streamer_batch_seconds = Histogram(
                "wonderwall_streamer_batch_seconds",
                "End-to-end batch processing latency in the streamer",
                buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            )
            self.streamer_batch_size = Gauge(
                "wonderwall_streamer_batch_size",
                "Configured windows-per-batch for the running streamer",
            )
        else:  # pragma: no cover
            noop = _NoopMetric()
            for attr in (
                "inferences_started", "inferences_completed", "inferences_failed",
                "input_tokens_total", "output_tokens_total",
                "inference_seconds", "kirk_seconds", "adapter_seconds", "llm_seconds",
                "batch_size", "entropy_last", "ready",
                "train_loss", "train_loss_avg", "train_steps_total",
                "train_grad_norm", "train_step_seconds", "train_epoch",
                "frames_consumed_total", "narrations_produced_total",
                "frames_skipped_total", "streamer_batch_seconds", "streamer_batch_size",
            ):
                setattr(self, attr, noop)

    # Convenience wrappers -----------------------------------------------------

    def inference_started(self, pipeline: str) -> None:
        self.inferences_started.labels(pipeline=pipeline).inc()

    def inference_completed(
        self,
        pipeline: str,
        soft_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        self.inferences_completed.labels(pipeline=pipeline).inc()
        if soft_tokens is not None:
            self.input_tokens_total.labels(pipeline=pipeline).inc(soft_tokens)
        if output_tokens is not None:
            self.output_tokens_total.labels(pipeline=pipeline).inc(output_tokens)

    def inference_failed(self, pipeline: str, exc: BaseException) -> None:
        self.inferences_failed.labels(
            pipeline=pipeline, error_class=type(exc).__name__
        ).inc()

    @contextlib.contextmanager
    def inference_timer(self, pipeline: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.inference_seconds.labels(pipeline=pipeline).observe(dt)

    @contextlib.contextmanager
    def kirk_timer(self) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.kirk_seconds.observe(time.perf_counter() - t0)

    @contextlib.contextmanager
    def adapter_timer(self) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.adapter_seconds.observe(time.perf_counter() - t0)

    @contextlib.contextmanager
    def llm_timer(self, pipeline: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.llm_seconds.labels(pipeline=pipeline).observe(time.perf_counter() - t0)

    def render(self) -> bytes:
        """Render the registry as Prometheus text format."""
        if _AVAILABLE:
            return generate_latest(REGISTRY)
        return b""  # pragma: no cover


# Module-level singleton; import as `from wonderwall.metrics_export import metrics`.
metrics = _Metrics()
