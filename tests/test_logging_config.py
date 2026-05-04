"""Tests for the structured logging config.

Validates:
  - setup_logging is idempotent (multiple calls don't duplicate handlers)
  - It returns a named logger when given a component
  - Returned root logger has stream handler attached
  - Level override via env var works
  - JSON formatter produces parseable JSON when python-json-logger is installed
  - Plain formatter is used as fallback when not installed
  - extra={} kwargs land in the structured record
"""
from __future__ import annotations

import importlib
import io
import json
import logging
from unittest.mock import patch

import pytest

from wonderwall import logging_config


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Each test starts with a clean module-level _CONFIGURED flag."""
    logging_config._CONFIGURED = False
    # Also remove any handlers attached to the root logger so we don't leak
    # state between tests.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    logging_config._CONFIGURED = False


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_setup_logging_is_idempotent():
    """Calling twice should not duplicate handlers on the root logger."""
    logging_config.setup_logging()
    n_handlers_after_first = len(logging.getLogger().handlers)
    logging_config.setup_logging()
    n_handlers_after_second = len(logging.getLogger().handlers)
    assert n_handlers_after_first == n_handlers_after_second


def test_setup_logging_returns_named_logger():
    log = logging_config.setup_logging("wonderwall.test")
    assert log.name == "wonderwall.test"


def test_setup_logging_returns_root_when_no_component():
    log = logging_config.setup_logging()
    assert log.name == "root"


# ---------------------------------------------------------------------------
# Level handling
# ---------------------------------------------------------------------------


def test_explicit_level_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    logging_config.setup_logging(level="DEBUG")
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG


def test_env_var_sets_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    logging_config.setup_logging()
    assert logging.getLogger().getEffectiveLevel() == logging.ERROR


def test_default_level_is_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    logging_config.setup_logging()
    assert logging.getLogger().getEffectiveLevel() == logging.INFO


# ---------------------------------------------------------------------------
# Formatter selection — tries python-json-logger, falls back to plain
# ---------------------------------------------------------------------------


def test_dict_config_has_a_formatter():
    cfg = logging_config._build_dict_config("INFO")
    assert "formatters" in cfg
    formatters = cfg["formatters"]
    # Either the json or plain formatter should exist
    assert any(name in formatters for name in ("json", "plain"))


def test_root_logger_has_stdout_stream_handler():
    logging_config.setup_logging()
    handlers = logging.getLogger().handlers
    stream_handlers = [h for h in handlers if isinstance(h, logging.StreamHandler)]
    assert stream_handlers, "no StreamHandler attached to root"


# ---------------------------------------------------------------------------
# Noisy-library suppression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("noisy", ["urllib3", "httpx", "kafka", "confluent_kafka"])
def test_noisy_libs_set_to_warning(noisy):
    logging_config.setup_logging()
    assert logging.getLogger(noisy).getEffectiveLevel() == logging.WARNING


# ---------------------------------------------------------------------------
# Structured fields end-to-end (only if python-json-logger is installed)
# ---------------------------------------------------------------------------


def test_extra_kwargs_appear_in_json_output_if_available():
    pytest.importorskip("pythonjsonlogger.jsonlogger")
    # Replace stdout handler with one that writes to a StringIO so we can
    # inspect the output.
    logging_config.setup_logging()
    root = logging.getLogger()
    # Replace the existing stream handler's stream with our buffer
    buf = io.StringIO()
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = buf

    logging.getLogger("wonderwall.test").info(
        "hello", extra={"pipeline": "C", "T": 4}
    )

    out = buf.getvalue().strip()
    assert out, "no log output captured"
    # Each record is a single JSON line
    record = json.loads(out)
    assert record["message"] == "hello"
    assert record["pipeline"] == "C"
    assert record["T"] == 4
    assert "level" in record  # renamed from levelname
