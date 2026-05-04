"""wonderwall — Kirk-to-LLM projection adapter and inference pipeline.

Sits between Kirk (kirk-pipeline / kirk-cli) and Scotty/Gemma (the LLM agent),
projecting Kirk's two-layer outputs into the LLM's embedding space.

Sibling repos in the Kavara stack:
  - UlyssesModel/uhura            — TGE: raw market data → N×N tensors
  - UlyssesModel/tiberius-openshift — SmartOrderRouter for compute (AMX dispatch)
  - UlyssesModel/kirk-runner      — kirk-cli control plane + kirk-pipeline lib
  - UlyssesModel/scotty           — OpenAI-compatible LLM agent shim (Gemma 4 31B)

This package is the Kirk-encoder → projection-adapter → frozen-LLM-predictor bridge
(LLaVA-pattern: trainable adapter, frozen predictor).
"""

__version__ = "0.1.0"

from .interfaces import (
    AdapterConfig,
    KirkClient,
    KirkMode,
    KirkOutput,
)
from .adapter import KirkProjectionAdapter
from .kirk_client import (
    KirkPipelineClient,
    KirkSubprocessClient,
    StubKirkClient,
)

__all__ = [
    "AdapterConfig",
    "KirkClient",
    "KirkMode",
    "KirkOutput",
    "KirkProjectionAdapter",
    "KirkPipelineClient",
    "KirkSubprocessClient",
    "StubKirkClient",
]
