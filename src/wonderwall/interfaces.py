"""Core type contracts for wonderwall.

Aligned to the actual kirk-cli + kirk-pipeline architecture per the
`kirk-cli (formerly kirk-runner)` Confluence page (EDP/36044801, 2026-04-13):

  Layer 1 (Feature Extraction):
      Input: N tickers × N time intervals → split into [16×16] blocks
      Per-block inference_features() runs → produces a feature vector per block
      Backed by KirkModelInterface.forward(paired float32 tensors)

  Layer 2 (Entropy Aggregation):
      Layer-1 feature vectors are assembled into a [32×32] matrix
      inference_entropy() runs once over that 32×32 matrix → scalar entropy
      In active_inference mode, also emits row/column expectations and
      reconstruction (the Array / Vector / Scalar trio from the Data Science Guide)

The wonderwall projection adapter taps off Layer 2's input matrix (the rich
representation; "Array" in the Data Science Guide nomenclature) plus its
output expectations and entropy. This is decision D-006 in DECISIONS.md
(layer-2 preferred for projection).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Sequence

import torch


# ---------------------------------------------------------------------------
# Inference modes (mirrors kirk-cli's --mode flag)
# ---------------------------------------------------------------------------


class KirkMode(str, Enum):
    """The five inference modes exposed by kirk-pipeline.

    Per kirk-cli docs, `active_*` modes update internal weight matrices on
    every call, suitable for non-stationary streaming data. Pure `inference_*`
    modes do not update.

    For wonderwall training we use ACTIVE_INFERENCE (returns all outputs in
    one call). For pure embedding-injection inference we use INFERENCE_FEATURES
    or INFERENCE_ENTROPY depending on whether we need the rich layer-2 input or
    just the entropy scalar.
    """

    INFERENCE_FEATURES = "inference_features"
    INFERENCE_ENTROPY = "inference_entropy"
    ACTIVE_INFERENCE_FEATURES = "active_inference_features"
    ACTIVE_INFERENCE_ENTROPY = "active_inference_entropy"
    ACTIVE_INFERENCE = "active_inference"


# ---------------------------------------------------------------------------
# Kirk outputs
# ---------------------------------------------------------------------------
#
# Production Kirk per Ted's `kirk_data_description.md`:
#   - Inputs are real-valued log-returns (complex128 container, imag=0)
#   - Layer 1 takes 16×16 blocks
#   - Layer 2 takes the 32×32 assembled feature matrix
#
# When the kirk-pipeline forward pass runs in active_inference mode, the
# layer-2 step exposes:
#   - layer2_input:        (32, 32) — the assembled feature matrix
#   - layer2_reconstruction:(32, 32) — Kirk's reconstruction ("Array" in DS guide)
#   - layer2_marginals:    (64,)    — row + col expected values ("Vector")
#   - entropy:             ()       — scalar entropy output ("Scalar")
#
# layer1_features is per-block, optional; useful for diagnostics, not used by
# the projection adapter today.


@dataclass(frozen=True)
class KirkOutput:
    """One forward pass through the two-layer Kirk pipeline.

    The projection adapter consumes `layer2_reconstruction` (Array) and
    `layer2_marginals` (Vector). Entropy is preserved as an anomaly gate.

    Attributes:
        layer2_input:          (32, 32) float32. Assembled feature matrix
                                fed into Layer 2. Useful for diagnostics.
        layer2_reconstruction: (32, 32) float32 or complex64. Kirk's
                                reconstruction of layer2_input weighted
                                against learned states. "Array" output.
        layer2_marginals:      (64,) float32 or complex64. Row+col expected
                                values. "Vector" output.
        entropy:               () float32. Final scalar entropy. "Scalar" output.
        layer1_features:       Optional (num_blocks, feat_dim) tensor. Diagnostic.
        n:                     int. Layer-2 dimension (always 32 in production).
        timestamp_ns:          Optional int. End-of-window timestamp.
        mode:                  KirkMode used to produce these outputs.
    """

    layer2_input: torch.Tensor
    layer2_reconstruction: torch.Tensor
    layer2_marginals: torch.Tensor
    entropy: torch.Tensor
    n: int = 32
    layer1_features: Optional[torch.Tensor] = None
    timestamp_ns: Optional[int] = None
    mode: KirkMode = KirkMode.ACTIVE_INFERENCE

    @property
    def is_complex(self) -> bool:
        return torch.is_complex(self.layer2_reconstruction)

    # Backwards-compat aliases — earlier scaffold code referenced .array/.vector/.scalar
    @property
    def array(self) -> torch.Tensor:
        return self.layer2_reconstruction

    @property
    def vector(self) -> torch.Tensor:
        return self.layer2_marginals

    @property
    def scalar(self) -> torch.Tensor:
        return self.entropy

    def validate(self) -> None:
        """Strict shape + dtype check. Raises on mismatch."""
        n = self.n
        if self.layer2_input.shape != (n, n):
            raise ValueError(
                f"layer2_input shape {tuple(self.layer2_input.shape)} != ({n}, {n})"
            )
        if self.layer2_reconstruction.shape != (n, n):
            raise ValueError(
                f"layer2_reconstruction shape {tuple(self.layer2_reconstruction.shape)} != ({n}, {n})"
            )
        if self.layer2_marginals.shape != (2 * n,):
            raise ValueError(
                f"layer2_marginals shape {tuple(self.layer2_marginals.shape)} != ({2 * n},)"
            )
        if self.entropy.shape != ():
            raise ValueError(f"entropy must be scalar, got {tuple(self.entropy.shape)}")


# ---------------------------------------------------------------------------
# Kirk client protocol
# ---------------------------------------------------------------------------


class KirkClient(Protocol):
    """Whatever produces KirkOutputs.

    Three implementations:
      - StubKirkClient        synthetic outputs for tests
      - KirkPipelineClient    direct import of `kirk_pipeline.KirkModelInterface`
                              (production path on tdx-amx-node-octo / IvorHQ)
      - KirkSubprocessClient  shells out to `kirk single --mode active_inference`
                              and parses JSON output (works without kirk-pipeline
                              installed in our process)
    """

    def infer(
        self,
        tensor: torch.Tensor,
        mode: KirkMode = KirkMode.ACTIVE_INFERENCE,
    ) -> KirkOutput:
        """Run a single forward pass over an input tensor.

        Input tensor is the raw N×N data matrix (typically log-returns from
        Uhura's `cross_section_temporal` renderer). Two-layer Kirk pipeline
        runs internally; output is the layer-2 results.
        """
        ...

    def infer_stream(
        self,
        tensors: Sequence[torch.Tensor],
        mode: KirkMode = KirkMode.ACTIVE_INFERENCE,
    ) -> Sequence[KirkOutput]:
        """Run a stream of inputs, preserving order.

        For active_inference modes the model state evolves across calls,
        so order matters. For pure inference modes the order is irrelevant
        but is preserved by convention.
        """
        ...


# ---------------------------------------------------------------------------
# Adapter configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterConfig:
    """Hyperparameters for the Kirk → LLM projection adapter.

    Default n=32 matches kirk-pipeline's Layer-2 dimension (the assembled
    feature matrix). The earlier scaffold defaulted to n=16; that was the
    Layer-1 block size, which is the wrong tap point per D-006.

    The adapter produces (n + 2) embedding vectors per Kirk window:
      - n vectors from the rows of the layer-2 reconstruction (Array)
      - 1 vector summarizing row marginals
      - 1 vector summarizing column marginals
    """

    n: int = 32  # kirk-pipeline Layer-2 dimension, fixed at 32 in production
    llm_hidden_dim: int = 5376  # Gemma 4 31B placeholder — pin via scripts/pin_gemma_hidden_dim.py
    hidden_dim: int = 1024
    use_complex: bool = False  # Production Kirk: real-valued, imag=0
    dropout: float = 0.0
    pos_embedding_init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(f"n must be >= 2, got {self.n}")
        if self.llm_hidden_dim < 64:
            raise ValueError(f"llm_hidden_dim suspiciously small: {self.llm_hidden_dim}")

    @property
    def tokens_per_sample(self) -> int:
        return self.n + 2

    @property
    def row_input_dim(self) -> int:
        return self.n * (2 if self.use_complex else 1)

    @property
    def marginal_input_dim(self) -> int:
        return self.n * (2 if self.use_complex else 1)
