"""LLM embedding-injection wrapper.

Two paths supported, both targeting Gemma 4 31B (Scotty's default model):

  1. EmbeddingInjectionLLM     — direct HuggingFace Transformers path. Loads the
                                 model, accepts pre-computed `inputs_embeds`,
                                 generates next tokens. Used for both training
                                 (gradients flow through frozen LLM into the
                                 adapter) and high-fidelity eval. Requires GPU.

  2. ScottyClient              — OpenAI-compatible HTTP client to a running Scotty
                                 / Ollama backend. CANNOT inject embeddings — only
                                 takes tokens. Used for the *compressed-text*
                                 baseline (today's production path), where Kirk
                                 outputs are rendered to a compact text block and
                                 sent as a normal prompt.

The eval harness compares both. The cost-saving thesis lives in (1); the
production-shape inference path lives in (2). Today's Pipeline B in Uhura's
sweep leaderboard is mechanism (2). Mechanism (1) is the new build.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Path 1: HuggingFace direct, supports inputs_embeds
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """LLM serving config for the embedding-injection path.

    Defaults match Scotty's default model. Verify `hidden_dim` against the
    actual Gemma 4 31B model card before training: this is a placeholder
    pending the real model weights becoming locally available.
    """

    model_name: str = "google/gemma-4-31b"  # placeholder — verify exact HF id
    hidden_dim: int = 5376                  # placeholder — verify against config.json
    dtype: str = "bfloat16"                 # bf16 is the standard Gemma serving dtype
    device: str = "cuda"
    trust_remote_code: bool = False


class EmbeddingInjectionLLM:
    """Frozen LLM that accepts pre-computed embeddings.

    Loads via HuggingFace Transformers. The LLM weights are frozen; gradients
    only flow into whatever upstream produced the input embeddings (the
    projection adapter, in our case).

    Implementation note: HF causal LMs accept `inputs_embeds` as an alternative
    to `input_ids` in their `forward()` and `generate()` methods. The embedding
    table is bypassed entirely.

    This class is the ONLY place where transformers is imported, so the rest
    of the package can be installed without it for stub/CPU work.
    """

    def __init__(self, config: LLMConfig):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "transformers is required for EmbeddingInjectionLLM. "
                "Install with `pip install wonderwall[llm]`."
            ) from e

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code
        )
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype_map[config.dtype],
            trust_remote_code=config.trust_remote_code,
        ).to(config.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Sanity check: LLM hidden dim must match what the adapter was configured for
        actual_hidden = self.model.config.hidden_size
        if actual_hidden != config.hidden_dim:
            raise ValueError(
                f"LLM hidden_size {actual_hidden} != configured hidden_dim "
                f"{config.hidden_dim}. Update LLMConfig.hidden_dim to {actual_hidden}."
            )

    def forward_with_embeds(
        self,
        inputs_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Run a forward pass with pre-computed embeddings.

        Args:
            inputs_embeds: (B, T, H) — output of the projection adapter,
                           optionally concatenated with a downstream prompt-token
                           embedding for instruction prefix.
            labels:        (B, T) — for training; if provided, loss is computed.
            attention_mask: (B, T) — optional, defaults to all-ones.

        Returns the HF model output (with `.loss`, `.logits`, etc.).
        """
        return self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def generate_with_embeds(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 256,
        attention_mask: Optional[torch.Tensor] = None,
        **gen_kwargs,
    ) -> str:
        """Greedy / sampling generation starting from pre-computed embeddings.

        Returns decoded text (the new tokens only, prompt embeds are not decoded
        because they have no token-id representation).
        """
        out = self.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        # `out` contains only the generated token IDs when input is inputs_embeds.
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def text_to_token_embeds(self, text: str, device: Optional[str] = None) -> torch.Tensor:
        """Tokenize text and look up its embeddings — for prepending instruction
        prefixes ahead of the projected Kirk embeddings."""
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        if device is not None:
            ids = ids.to(device)
        else:
            ids = ids.to(self.config.device)
        embed_layer = self.model.get_input_embeddings()
        return embed_layer(ids)


# ---------------------------------------------------------------------------
# Path 2: Scotty / Ollama (OpenAI-compatible) — token-only baseline
# ---------------------------------------------------------------------------


@dataclass
class ScottyConfig:
    """Config for talking to a running Scotty / Ollama / vLLM endpoint.

    Defaults mirror Scotty's defaults from github.com/UlyssesModel/scotty.
    """

    base_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:31b"
    timeout_s: float = 120.0


class ScottyClient:
    """Thin OpenAI-compatible client for the production-shape baseline.

    This is the *compressed-text* path: render Kirk outputs into a small text
    block (numbers + structured features) and send as a normal chat prompt.
    Cannot inject pre-computed embeddings — Ollama's API doesn't support it.

    Used in the eval harness as the apples-to-apples comparison against the
    embedding-injection path.
    """

    def __init__(self, config: ScottyConfig):
        try:
            import httpx  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "httpx is required for ScottyClient. "
                "Install with `pip install wonderwall[scotty]`."
            ) from e
        self._httpx = httpx
        self.config = config

    def chat(self, messages: list[dict], **gen_kwargs) -> str:
        """Send a chat-completion request, return the assistant text.

        OpenAI-compatible, mirrors what Scotty itself does internally.
        """
        url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            **gen_kwargs,
        }
        with self._httpx.Client(timeout=self.config.timeout_s) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list[dict], **gen_kwargs):
        """Streaming variant — yields content deltas. Mirrors Scotty's streaming."""
        url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            **gen_kwargs,
        }
        with self._httpx.Client(timeout=self.config.timeout_s) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    chunk = line[len("data: ") :]
                    if chunk.strip() == "[DONE]":
                        return
                    obj = json.loads(chunk)
                    delta = obj["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
