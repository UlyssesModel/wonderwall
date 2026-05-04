"""Kirk client implementations.

Three concrete clients, all conforming to the KirkClient Protocol from
interfaces.py:

  - StubKirkClient            synthetic outputs in the right shapes for tests
                              and plumbing validation. No kirk-pipeline needed.

  - KirkPipelineClient        production path on tdx-amx-node-octo / IvorHQ.
                              Direct import of kirk-pipeline's
                              `KirkModelInterface`. Calls `forward()` for both
                              layers in sequence; returns assembled KirkOutput.

  - KirkSubprocessClient      shells out to `kirk single --mode active_inference
                              --file <input> --format json` and parses the
                              result. Works on any host with kirk-cli installed
                              even if kirk-pipeline isn't importable in our
                              process. Used during development on hosts where
                              the production wheel isn't installed.

The two-layer Kirk pipeline (per kirk-cli docs):
  1. Split N×N input into [16×16] blocks
  2. Run inference_features on each block → layer-1 feature vectors
  3. Assemble feature vectors into [32×32] matrix
  4. Run inference_entropy on assembled matrix → layer-2 outputs
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from .interfaces import KirkClient, KirkMode, KirkOutput


# ---------------------------------------------------------------------------
# Stub: synthetic outputs for tests
# ---------------------------------------------------------------------------


@dataclass
class StubKirkClient:
    """Synthetic Kirk for unit tests + plumbing validation.

    Reproduces the canonical layer-2 output shapes:
      - layer2_input         (32, 32) float32
      - layer2_reconstruction(32, 32) float32 (complex64 if use_complex)
      - layer2_marginals     (64,)   float32 (complex64 if use_complex)
      - entropy              ()      float32
    """

    n: int = 32
    use_complex: bool = False
    seed: int = 0

    def _synthetic(self, tensor: torch.Tensor, mode: KirkMode) -> KirkOutput:
        n = self.n
        per_call_seed = (
            int(tensor.real.abs().sum().item() * 1e6) if torch.is_complex(tensor)
            else int(tensor.abs().sum().item() * 1e6)
        )
        per_call_seed = (per_call_seed + self.seed) % (2**31)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(per_call_seed)

        layer2_input = torch.randn(n, n, generator=gen) * 0.05

        recon_real = torch.randn(n, n, generator=gen) * 0.01
        marg_real = torch.randn(2 * n, generator=gen) * 0.01

        if self.use_complex:
            recon_imag = torch.randn(n, n, generator=gen) * 0.01
            marg_imag = torch.randn(2 * n, generator=gen) * 0.01
            layer2_reconstruction = torch.complex(recon_real, recon_imag)
            layer2_marginals = torch.complex(marg_real, marg_imag)
        else:
            layer2_reconstruction = recon_real
            layer2_marginals = marg_real

        # Synthetic entropy in production range (~0.4–18.9 nats per Uhura page)
        entropy = torch.tensor(
            float(torch.rand((), generator=gen) * 18.5 + 0.4), dtype=torch.float32
        )

        return KirkOutput(
            layer2_input=layer2_input,
            layer2_reconstruction=layer2_reconstruction,
            layer2_marginals=layer2_marginals,
            entropy=entropy,
            n=n,
            mode=mode,
        )

    def infer(
        self, tensor: torch.Tensor, mode: KirkMode = KirkMode.ACTIVE_INFERENCE
    ) -> KirkOutput:
        if tensor.dim() != 2 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError(f"Stub expects square input; got {tuple(tensor.shape)}")
        out = self._synthetic(tensor, mode)
        out.validate()
        return out

    def infer_stream(
        self,
        tensors: Sequence[torch.Tensor],
        mode: KirkMode = KirkMode.ACTIVE_INFERENCE,
    ) -> Sequence[KirkOutput]:
        return [self.infer(t, mode) for t in tensors]


# ---------------------------------------------------------------------------
# Production path: direct kirk-pipeline import
# ---------------------------------------------------------------------------


class KirkPipelineClient:
    """Direct in-process Kirk via kirk-pipeline.KirkModelInterface.

    Use on production VMs where kirk-pipeline is installed (tdx-amx-node-octo,
    IvorHQ). Avoids subprocess overhead. Requires the kirk-pipeline wheel to
    be importable.

    The two-layer pipeline is implemented here in Python rather than relying
    on kirk-cli's `run` command, so we get direct access to layer-2 outputs
    (which `kirk run --format json` may or may not surface depending on flags).
    """

    def __init__(self, weights_path: Optional[str] = None):
        try:
            from kirk_pipeline import KirkModelInterface  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "kirk_pipeline is not installed. Install the production wheel "
                "(uv add /dist/kirk_pipeline-*.whl) or use KirkSubprocessClient."
            ) from e
        self._iface_cls = KirkModelInterface
        self._iface = KirkModelInterface()
        self._iface.load_weights(weights_path)

    def _run_layer1(self, input_matrix: torch.Tensor) -> torch.Tensor:
        """Split into 16×16 blocks, run inference_features per block, return
        feature vectors stacked into a 32×32 matrix.
        """
        n = input_matrix.shape[0]
        # Per kirk-cli architecture: split N×N into 16×16 blocks. Production
        # universes are sized so this tiles cleanly (e.g. 32×32 → 4 blocks of 16×16).
        block = 16
        if n % block != 0:
            raise ValueError(f"Input size {n} not divisible by 16-block size")
        blocks = []
        for i in range(0, n, block):
            for j in range(0, n, block):
                sub = input_matrix[i : i + block, j : j + block].contiguous().float()
                feats = self._iface.forward(sub, mode="inference_features")
                blocks.append(torch.as_tensor(feats))
        # Stack into 32×32. For a 32×32 input that's 4 blocks of 16×16.
        # For now the simplest assembly: concatenate row-wise — Spencer's
        # kirk-pipeline may have a canonical assembly that we'd want to
        # match here. TODO: confirm.
        stacked = torch.stack(blocks, dim=0).reshape(n, n).contiguous()
        return stacked

    def _run_layer2(
        self, assembled: torch.Tensor, mode: KirkMode
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run layer-2 inference on the assembled 32×32 feature matrix.

        Returns (reconstruction, marginals, entropy).
        """
        out = self._iface.forward(assembled.float(), mode=mode.value)
        # kirk-pipeline's forward() output shape for active_inference:
        # expects {'reconstruction': (n,n), 'marginals': (2n,), 'entropy': ()}
        # If the actual return is positional or different, adjust here once
        # confirmed against kirk-pipeline source.
        if isinstance(out, dict):
            recon = torch.as_tensor(out["reconstruction"])
            marg = torch.as_tensor(out["marginals"])
            ent = torch.as_tensor(out["entropy"])
        else:
            # Fallback: assume the dict shape; raise for now.
            raise RuntimeError(
                f"Unexpected kirk-pipeline forward() output shape: {type(out)}. "
                "Inspect kirk_pipeline source to update KirkPipelineClient._run_layer2."
            )
        return recon, marg, ent

    def infer(
        self,
        tensor: torch.Tensor,
        mode: KirkMode = KirkMode.ACTIVE_INFERENCE,
    ) -> KirkOutput:
        layer2_input = self._run_layer1(tensor)
        recon, marg, ent = self._run_layer2(layer2_input, mode)
        out = KirkOutput(
            layer2_input=layer2_input,
            layer2_reconstruction=recon,
            layer2_marginals=marg,
            entropy=ent,
            n=layer2_input.shape[0],
            mode=mode,
        )
        out.validate()
        return out

    def infer_stream(
        self,
        tensors: Sequence[torch.Tensor],
        mode: KirkMode = KirkMode.ACTIVE_INFERENCE,
    ) -> Sequence[KirkOutput]:
        return [self.infer(t, mode) for t in tensors]


# ---------------------------------------------------------------------------
# Subprocess path: shell out to kirk-cli
# ---------------------------------------------------------------------------


@dataclass
class KirkSubprocessClient:
    """Shells out to `kirk single` and parses JSON output.

    Useful on dev hosts where the production kirk-pipeline wheel isn't
    importable in our process but the kirk-cli binary is on $PATH.

    Requires: `kirk` binary in PATH, with kirk-pipeline (real or stub)
    available to it. Verify with `kirk info`.
    """

    binary: str = "kirk"
    n: int = 32
    timeout_s: float = 60.0

    def _run_kirk(
        self, input_matrix: torch.Tensor, mode: KirkMode
    ) -> dict:
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "input.npy")
            np.save(in_path, input_matrix.detach().cpu().float().numpy())
            cmd = [
                self.binary, "single",
                "--file", in_path,
                "--mode", mode.value,
                "--format", "json",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_s
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"kirk single failed (exit {result.returncode}):\n"
                    f"stderr: {result.stderr}\nstdout: {result.stdout}"
                )
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"kirk single emitted non-JSON output:\n{result.stdout[:500]}"
                ) from e

    def infer(
        self,
        tensor: torch.Tensor,
        mode: KirkMode = KirkMode.ACTIVE_INFERENCE,
    ) -> KirkOutput:
        # Subprocess path runs the assembled-matrix layer-2 step only — we feed
        # the full N×N matrix and let kirk-cli handle the layer split internally
        # via `kirk run`. For `kirk single` we go straight to layer-2 semantics.
        # If you need the two-layer split, use KirkPipelineClient.
        result = self._run_kirk(tensor, mode)
        # Expected JSON shape (kirk-cli single command output):
        #   { "reconstruction": [[...], ...], "marginals": [...], "entropy": float }
        recon = torch.tensor(result["reconstruction"], dtype=torch.float32)
        marg = torch.tensor(result["marginals"], dtype=torch.float32)
        ent = torch.tensor(result["entropy"], dtype=torch.float32)
        out = KirkOutput(
            layer2_input=tensor.float(),
            layer2_reconstruction=recon,
            layer2_marginals=marg,
            entropy=ent,
            n=tensor.shape[0],
            mode=mode,
        )
        out.validate()
        return out

    def infer_stream(
        self,
        tensors: Sequence[torch.Tensor],
        mode: KirkMode = KirkMode.ACTIVE_INFERENCE,
    ) -> Sequence[KirkOutput]:
        return [self.infer(t, mode) for t in tensors]
