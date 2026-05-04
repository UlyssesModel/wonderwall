"""Streaming Kafka consumer: Uhura tensor frames → wonderwall narrations.

Subscribes to `ulysses.tensor.frames.<cadence>` (Uhura's broadcaster output),
batches incoming frames into windows, runs the full Kirk → adapter → LLM
pipeline, publishes the resulting narration to `ulysses.narrations.<cadence>`.

Mirrors uhura's `kafka_trades.py` consumer pattern: confluent-kafka with
Strimzi mTLS, cooperative-sticky rebalancing, idempotent writer with
explicit ACL annotation in the corresponding KafkaUser CR.

Frame payload format (read defensively, like uhura_io.py for npz):
  - Either raw msgpack with keys {"tensor": [...], "shape": [n,n], "ts_ns": ...}
  - Or JSON with the same keys
  - Or numpy .npy bytes (recovered via numpy.lib.format.read_array)

If Uhura's broadcaster format is different, adjust _parse_message().
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import yaml

from wonderwall.adapter import KirkProjectionAdapter
from wonderwall.injection import EmbeddingInjectionLLM, LLMConfig
from wonderwall.interfaces import AdapterConfig
from wonderwall.kirk_client import (
    KirkPipelineClient,
    KirkSubprocessClient,
    StubKirkClient,
)
from wonderwall.logging_config import setup_logging
from wonderwall.metrics_export import metrics
from wonderwall.pipeline import EmbeddingInjectionPipeline


logger = setup_logging("wonderwall.stream")


@dataclass
class StreamConfig:
    kafka_bootstrap: str
    input_topic: str
    output_topic: str
    group_id: str = "wonderwall-streamer"
    windows_per_batch: int = 4
    max_new_tokens: int = 256
    consumer_security_protocol: str = "SSL"
    ssl_ca_path: str = "/opt/app-root/kafka-creds/ca.crt"
    ssl_cert_path: str = "/opt/app-root/kafka-creds/user.crt"
    ssl_key_path: str = "/opt/app-root/kafka-creds/user.key"


def _parse_message(msg_value: bytes) -> tuple[np.ndarray, Optional[int]]:
    """Best-effort decode of an incoming Kafka frame.

    Tries JSON first (most common in early uhura), falls back to raw .npy bytes.
    Returns (array, optional_ts_ns).
    """
    # Try JSON
    try:
        payload = json.loads(msg_value.decode("utf-8"))
        if isinstance(payload, dict) and "tensor" in payload:
            arr = np.asarray(payload["tensor"], dtype=np.float32)
            if "shape" in payload:
                arr = arr.reshape(payload["shape"])
            ts = payload.get("ts_ns") or payload.get("timestamp_ns")
            return arr, int(ts) if ts is not None else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    # Try raw .npy bytes
    try:
        from io import BytesIO
        buf = BytesIO(msg_value)
        arr = np.load(buf)
        if np.iscomplexobj(arr):
            arr = arr.real
        return arr.astype(np.float32), None
    except Exception:
        pass

    raise ValueError(
        "could not parse incoming Kafka message — neither JSON nor .npy bytes. "
        "Update _parse_message() to match uhura's broadcaster format."
    )


def _build_pipeline(args) -> EmbeddingInjectionPipeline:
    with open(args.adapter_config) as f:
        adapter_cfg = AdapterConfig(**yaml.safe_load(f))
    with open(args.llm_config) as f:
        llm_cfg = LLMConfig(**yaml.safe_load(f))

    backend = (args.kirk_backend or os.environ.get("KIRK_BACKEND", "pipeline")).lower()
    if backend == "pipeline":
        kirk = KirkPipelineClient()
    elif backend == "subprocess":
        kirk = KirkSubprocessClient(n=adapter_cfg.n)
    elif backend == "stub":
        kirk = StubKirkClient(n=adapter_cfg.n, use_complex=adapter_cfg.use_complex)
    else:
        raise SystemExit(f"unknown --kirk-backend: {backend!r}")

    adapter = KirkProjectionAdapter(adapter_cfg)
    ckpt = torch.load(args.adapter_checkpoint, weights_only=False, map_location="cpu")
    adapter.load_state_dict(ckpt["adapter_state_dict"])

    llm = EmbeddingInjectionLLM(llm_cfg)
    adapter = adapter.to(llm.config.device).eval()
    return EmbeddingInjectionPipeline(kirk=kirk, adapter=adapter, llm=llm)


def _build_consumer(cfg: StreamConfig):
    try:
        from confluent_kafka import Consumer  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "confluent-kafka is required. Install with `pip install wonderwall[kafka]`."
        ) from e
    return Consumer({
        "bootstrap.servers": cfg.kafka_bootstrap,
        "group.id": cfg.group_id,
        "enable.auto.commit": False,           # explicit commit after publish
        "auto.offset.reset": "latest",
        "partition.assignment.strategy": "cooperative-sticky",
        "security.protocol": cfg.consumer_security_protocol,
        "ssl.ca.location": cfg.ssl_ca_path,
        "ssl.certificate.location": cfg.ssl_cert_path,
        "ssl.key.location": cfg.ssl_key_path,
    })


def _build_producer(cfg: StreamConfig):
    from confluent_kafka import Producer  # type: ignore
    return Producer({
        "bootstrap.servers": cfg.kafka_bootstrap,
        "enable.idempotence": True,
        "acks": "all",
        "compression.type": "lz4",
        "security.protocol": cfg.consumer_security_protocol,
        "ssl.ca.location": cfg.ssl_ca_path,
        "ssl.certificate.location": cfg.ssl_cert_path,
        "ssl.key.location": cfg.ssl_key_path,
    })


_running = True


def _install_signal_handlers():
    def _stop(signum, frame):
        global _running
        logger.info("received signal %d; shutting down", signum)
        _running = False
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)


def _start_metrics_server(port: int) -> threading.Thread:
    """Spin a tiny HTTP server in a daemon thread that exposes /metrics.

    Using prometheus_client.start_http_server keeps this a single-line affair
    and avoids pulling in FastAPI for the streamer pod, which doesn't need it.
    """
    try:
        from prometheus_client import start_http_server  # type: ignore
    except ImportError:  # pragma: no cover
        logger.warning("prometheus_client not installed; metrics server disabled")
        return threading.Thread()

    def _serve():
        start_http_server(port)
        logger.info("metrics server listening on :%d/metrics", port)
        # start_http_server returns immediately; block here so the thread sticks.
        while _running:
            time.sleep(1.0)

    t = threading.Thread(target=_serve, name="metrics", daemon=True)
    t.start()
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kafka-bootstrap", required=True)
    p.add_argument("--input-topic", required=True)
    p.add_argument("--output-topic", required=True)
    p.add_argument("--group-id", default="wonderwall-streamer")
    p.add_argument("--windows-per-batch", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--adapter-config", required=True)
    p.add_argument("--llm-config", required=True)
    p.add_argument("--adapter-checkpoint", required=True)
    p.add_argument("--kirk-backend", default=None,
                   help="pipeline | subprocess | stub. Defaults to KIRK_BACKEND env.")
    p.add_argument("--metrics-port", type=int, default=9090,
                   help="Port for /metrics endpoint (Prometheus scrape target)")
    args = p.parse_args()

    cfg = StreamConfig(
        kafka_bootstrap=args.kafka_bootstrap,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        group_id=args.group_id,
        windows_per_batch=args.windows_per_batch,
        max_new_tokens=args.max_new_tokens,
    )

    pipeline = _build_pipeline(args)
    consumer = _build_consumer(cfg)
    producer = _build_producer(cfg)

    consumer.subscribe([cfg.input_topic])
    logger.info("consuming from %s, publishing to %s", cfg.input_topic, cfg.output_topic)

    _install_signal_handlers()
    _start_metrics_server(args.metrics_port)
    metrics.streamer_batch_size.set(cfg.windows_per_batch)
    metrics.ready.set(1)

    batch: list[torch.Tensor] = []
    last_ts_ns: Optional[int] = None

    try:
        while _running:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("kafka error: %s", msg.error())
                continue
            try:
                arr, ts_ns = _parse_message(msg.value())
            except Exception as e:
                logger.warning("skip malformed message: %s", e)
                metrics.frames_skipped_total.labels(
                    topic=cfg.input_topic, reason=type(e).__name__
                ).inc()
                consumer.commit(msg, asynchronous=False)
                continue

            tensor = torch.from_numpy(arr)
            batch.append(tensor)
            metrics.frames_consumed_total.labels(topic=cfg.input_topic).inc()
            if ts_ns is not None:
                last_ts_ns = ts_ns

            if len(batch) >= cfg.windows_per_batch:
                t0 = time.perf_counter()
                metrics.inference_started("C_streamer")
                try:
                    with metrics.inference_timer("C_streamer"):
                        narration = pipeline.run(
                            batch, max_new_tokens=cfg.max_new_tokens
                        )
                except Exception as e:
                    metrics.inference_failed("C_streamer", e)
                    logger.exception("pipeline run failed; dropping batch")
                    batch = []
                    continue
                elapsed = time.perf_counter() - t0
                elapsed_ms = elapsed * 1000
                metrics.streamer_batch_seconds.observe(elapsed)
                metrics.inference_completed(
                    "C_streamer",
                    soft_tokens=len(batch) * (batch[0].shape[0] + 2),
                )

                payload = json.dumps(
                    {
                        "narration": narration,
                        "windows_in_batch": len(batch),
                        "last_ts_ns": last_ts_ns,
                        "inference_ms": elapsed_ms,
                    }
                ).encode("utf-8")
                producer.produce(cfg.output_topic, value=payload)
                producer.flush(timeout=5.0)
                metrics.narrations_produced_total.labels(topic=cfg.output_topic).inc()
                consumer.commit(msg, asynchronous=False)
                logger.info("emitted narration: %d windows, %.1f ms", len(batch), elapsed_ms)
                batch = []
    finally:
        consumer.close()
        producer.flush(timeout=5.0)
        logger.info("clean shutdown")


if __name__ == "__main__":
    sys.exit(main() or 0)
