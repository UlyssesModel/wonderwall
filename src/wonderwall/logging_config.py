"""Structured JSON logging for wonderwall.

Mirrors the convention used by uhura and tiberius-openshift's `ulysses-sor`
(per the Uhura Confluence page — pyproject.toml `[kafka]` extra includes
python-json-logger). One ConfigDict definition adopted by both
`scripts/serve_kserve.py` and `scripts/stream_consumer.py`.

Design choices:
  - One JSON record per line (Loki / UWM-friendly)
  - Fields: timestamp, level, logger, message, plus any kwargs passed via
    `extra=` on a logger call (no escaping needed; gets merged in).
  - Fall back to plain text formatter if python-json-logger isn't installed,
    so the package still runs in dev environments without the kafka extra.
  - LOG_LEVEL env var overrides the default INFO.

Usage:

    from wonderwall.logging_config import setup_logging
    setup_logging("wonderwall.serve")
    logger.info("inference started", extra={"pipeline": "C", "T": 4})

The `extra=` kwargs land in the JSON record alongside the other fields,
so Grafana / Loki LogQL queries can filter on them directly.
"""
from __future__ import annotations

import logging
import logging.config
import os
import sys
from typing import Optional


_DEFAULT_FIELDS = {
    "timestamp": "asctime",
    "level": "levelname",
    "logger": "name",
    "message": "message",
}


def _build_dict_config(level: str) -> dict:
    """Return a logging.config dictConfig for JSON output.

    Falls back to a plain formatter if python-json-logger isn't installed.
    """
    try:
        import pythonjsonlogger.jsonlogger  # noqa: F401  (just to detect)
        json_available = True
    except ImportError:
        json_available = False

    if json_available:
        formatters = {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "rename_fields": _DEFAULT_FIELDS,
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            }
        }
        formatter_name = "json"
    else:
        formatters = {
            "plain": {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            }
        }
        formatter_name = "plain"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": formatter_name,
            }
        },
        "root": {
            "level": level,
            "handlers": ["stdout"],
        },
        # Silence noisy libraries by default
        "loggers": {
            "urllib3": {"level": "WARNING"},
            "httpx": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
            "kafka": {"level": "WARNING"},
            "confluent_kafka": {"level": "WARNING"},
        },
    }


_CONFIGURED = False


def setup_logging(component: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """Configure the root logger once, return a named logger for the caller.

    Subsequent calls are no-ops (so multiple modules can call this safely).

    Args:
        component: optional module-style logger name (e.g. "wonderwall.serve").
                   Returned logger uses this name; if omitted, returns the root.
        level: log level override; defaults to LOG_LEVEL env var or INFO.

    Returns the named logger.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        cfg_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
        logging.config.dictConfig(_build_dict_config(cfg_level))
        _CONFIGURED = True
    return logging.getLogger(component) if component else logging.getLogger()


__all__ = ["setup_logging"]
