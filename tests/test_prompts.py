"""Tests for the versionable prompts module.

Regression-protects the contract that pipelines + distillation depend on:
  - Each registered prompt has a stable id (`name@version`)
  - `get(name, version)` returns the body
  - Missing names/versions raise KeyError with a useful message
  - All Pipeline-{A,B,C} prompts and the teacher prompt are non-empty
  - The C-pipeline prefix and suffix are distinct strings
"""
from __future__ import annotations

import pytest

from wonderwall import prompts
from wonderwall.prompts import (
    PROMPT_REGIME_NARRATION_A_V1,
    PROMPT_REGIME_NARRATION_B_V1,
    PROMPT_REGIME_NARRATION_C_PREFIX_V1,
    PROMPT_REGIME_NARRATION_C_SUFFIX_V1,
    PROMPT_TEACHER_NARRATION_V1,
    REGISTRY,
    PromptEntry,
    get,
    list_versions,
)


# ---------------------------------------------------------------------------
# Constants are non-empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", [
    PROMPT_REGIME_NARRATION_A_V1,
    PROMPT_REGIME_NARRATION_B_V1,
    PROMPT_REGIME_NARRATION_C_PREFIX_V1,
    PROMPT_REGIME_NARRATION_C_SUFFIX_V1,
    PROMPT_TEACHER_NARRATION_V1,
])
def test_prompt_strings_non_empty(body):
    assert isinstance(body, str)
    assert len(body) >= 8, "prompts shouldn't be near-empty placeholders"


def test_pipeline_C_prefix_and_suffix_are_distinct():
    assert PROMPT_REGIME_NARRATION_C_PREFIX_V1 != PROMPT_REGIME_NARRATION_C_SUFFIX_V1


# ---------------------------------------------------------------------------
# Registry round-trips
# ---------------------------------------------------------------------------


def test_registry_contains_all_pipeline_prompts():
    expected_ids = {
        "regime_narration_A@v1",
        "regime_narration_B@v1",
        "regime_narration_C_prefix@v1",
        "regime_narration_C_suffix@v1",
        "teacher_narration@v1",
    }
    assert expected_ids.issubset(set(REGISTRY.keys()))


def test_get_returns_correct_body():
    assert get("regime_narration_A", "v1") == PROMPT_REGIME_NARRATION_A_V1
    assert get("regime_narration_B", "v1") == PROMPT_REGIME_NARRATION_B_V1
    assert get("teacher_narration", "v1") == PROMPT_TEACHER_NARRATION_V1


def test_get_default_version_is_v1():
    """Most callers don't pass version; v1 should be the default."""
    assert get("regime_narration_A") == PROMPT_REGIME_NARRATION_A_V1


def test_get_missing_name_raises_keyerror():
    with pytest.raises(KeyError, match="not found"):
        get("nonexistent_prompt")


def test_get_missing_version_raises_keyerror():
    with pytest.raises(KeyError):
        get("regime_narration_A", version="v999")


def test_list_versions_returns_known_versions():
    versions = list_versions("regime_narration_A")
    assert "v1" in versions
    assert sorted(versions) == versions  # always sorted


def test_list_versions_for_unknown_name_returns_empty():
    assert list_versions("totally_made_up") == []


# ---------------------------------------------------------------------------
# PromptEntry helpers
# ---------------------------------------------------------------------------


def test_prompt_entry_id_format():
    entry = PromptEntry("foo", "v3", "body")
    assert entry.id == "foo@v3"


def test_registry_keys_match_entry_ids():
    """Every key in REGISTRY must equal its value's .id — keeps lookups consistent."""
    for key, entry in REGISTRY.items():
        assert key == entry.id, f"REGISTRY[{key!r}].id == {entry.id!r}"


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


def test_all_exports_resolve():
    """Every symbol in __all__ must be importable."""
    for name in prompts.__all__:
        assert hasattr(prompts, name), f"__all__ lists {name!r} but module doesn't export it"
