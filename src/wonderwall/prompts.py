"""Versionable system prompts for wonderwall pipelines.

Why a module instead of inline strings:
  - **Versionable.** Each prompt has a stable name and version suffix
    (`_V1`, `_V2`, …). Adapter checkpoints record which prompt version
    they were trained against in their metadata; eval harness records
    which version produced each narration. Reproducibility for free.
  - **A/B-testable.** The eval harness can iterate over prompt variants
    in the sweep grid. Same input → multiple prompt versions → narration
    quality comparison.
  - **Re-usable.** Predictor, streamer, and eval harness import from one
    place. No copy-paste drift.

Naming convention:
    PROMPT_<TASK>_<PIPELINE>_<VERSION>

Versions only bump on semantic change. Whitespace / typo fixes don't bump.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Active prompts
# ---------------------------------------------------------------------------

# Pipeline A — raw tokenized data baseline. The LLM gets the verbose dump
# of every cell of every window and has to make sense of it.
PROMPT_REGIME_NARRATION_A_V1 = (
    "You are a market-state analyst. Given a stream of raw price-tensor "
    "windows, narrate what is happening and forecast the next interval. "
    "Be concrete; cite specific row/col cells where relevant. Output should "
    "be 2-4 sentences, no boilerplate, no apologies."
)

# Pipeline B — compressed-text via Scotty. The LLM gets a small numerical
# summary (Kirk's row/col expectations + entropy) and produces narration.
PROMPT_REGIME_NARRATION_B_V1 = (
    "You are a market-state analyst. Given a compressed Kirk-encoded "
    "summary of recent price-tensor windows (row/col expected values, "
    "global expectation, entropy), produce a brief narration of what is "
    "happening and a directional forecast for the next interval. Be "
    "concrete; cite specific row/col expectations that drive your call. "
    "Output should be 2-4 sentences."
)

# Pipeline C — embedding-injection. The "soft tokens" carry the Kirk
# representation; the instruction is short. The instruction sits *outside*
# the soft-token block (prefix + suffix) so the LLM gets context about what
# it's looking at.
PROMPT_REGIME_NARRATION_C_PREFIX_V1 = (
    "You are a market-state analyst. The following soft tokens encode "
    "recent price-tensor windows. Narrate what is happening and forecast "
    "the next interval.\n\nWindows: "
)
PROMPT_REGIME_NARRATION_C_SUFFIX_V1 = "\n\nNarration: "

# Teacher prompt — used during distillation to generate gold narrations.
# Deliberately verbose: the teacher gets all the data and has to produce
# the narration the student should learn to reproduce.
PROMPT_TEACHER_NARRATION_V1 = (
    "You are an expert market-state analyst. The following are raw "
    "log-return tensor windows of N tickers (rows) over N time intervals "
    "(cols), in chronological order. Produce a brief narration of what is "
    "happening and a directional forecast for the next interval. Cite "
    "specific tickers/intervals where appropriate. Be concrete and "
    "concise — 2-4 sentences max."
)


# ---------------------------------------------------------------------------
# Registry — for the eval harness to enumerate variants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptEntry:
    name: str
    version: str
    body: str

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


REGISTRY: dict[str, PromptEntry] = {
    p.id: p
    for p in [
        PromptEntry("regime_narration_A", "v1", PROMPT_REGIME_NARRATION_A_V1),
        PromptEntry("regime_narration_B", "v1", PROMPT_REGIME_NARRATION_B_V1),
        PromptEntry(
            "regime_narration_C_prefix", "v1", PROMPT_REGIME_NARRATION_C_PREFIX_V1
        ),
        PromptEntry(
            "regime_narration_C_suffix", "v1", PROMPT_REGIME_NARRATION_C_SUFFIX_V1
        ),
        PromptEntry("teacher_narration", "v1", PROMPT_TEACHER_NARRATION_V1),
    ]
}


def get(name: str, version: str = "v1") -> str:
    """Look up a prompt body by name + version. Raises if missing."""
    key = f"{name}@{version}"
    if key not in REGISTRY:
        raise KeyError(
            f"prompt {key!r} not found. Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[key].body


def list_versions(name: str) -> list[str]:
    """All registered versions for a given prompt name."""
    return sorted(
        entry.version for entry in REGISTRY.values() if entry.name == name
    )


# Convenience exports — keep imports terse at the call site.
__all__ = [
    "PROMPT_REGIME_NARRATION_A_V1",
    "PROMPT_REGIME_NARRATION_B_V1",
    "PROMPT_REGIME_NARRATION_C_PREFIX_V1",
    "PROMPT_REGIME_NARRATION_C_SUFFIX_V1",
    "PROMPT_TEACHER_NARRATION_V1",
    "PromptEntry",
    "REGISTRY",
    "get",
    "list_versions",
]
