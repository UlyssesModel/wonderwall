#!/usr/bin/env python3
"""Pin an LLM's hidden dim by querying whatever serving runtime hosts it.

Used for Gemma 4 31B and (per D-015) DeepSeek v4. Updates
configs/adapter_default.yaml and the chosen configs/llm_*.yaml in place.

Usage:
    # Default — Gemma 4 31B against local Ollama:
    python scripts/pin_gemma_hidden_dim.py

    # DeepSeek v4 against scotty-gpu's Ollama:
    python scripts/pin_gemma_hidden_dim.py \\
        --llm-config configs/llm_deepseek4.yaml \\
        --ollama-model deepseek-v4 \\
        --hf-fallback deepseek-ai/DeepSeek-V4

    # Override Scotty endpoint:
    python scripts/pin_gemma_hidden_dim.py --base-url http://10.128.0.16:11434

    # Just print, don't update configs:
    python scripts/pin_gemma_hidden_dim.py --no-write

Two query mechanisms attempted in order:
  1. Ollama /api/show endpoint — returns model details including
     architecture metadata. Works for any Ollama-served model.
  2. HuggingFace `transformers.AutoConfig.from_pretrained()` — works if
     the equivalent HF model id resolves and the user has HF auth set up.

The configs are updated atomically (write-then-rename) and the prior values
are kept in `.bak` files so you can revert.

Note for MoE models (DeepSeek v4): we want the *embedding-layer* hidden dim
(what `inputs_embeds` injection feeds), not an expert-internal dim. Both
Ollama's `embedding_length` and HF AutoConfig's `hidden_size` return the
embedding dim, so this script is correct for both dense and MoE models.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request

import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ADAPTER_CFG_PATH = os.path.join(REPO_ROOT, "configs", "adapter_default.yaml")
DEFAULT_LLM_CFG_PATH = os.path.join(REPO_ROOT, "configs", "llm_gemma4.yaml")


def _ollama_show(base_url: str, model: str, timeout: float = 10.0) -> dict:
    """Hit Ollama's /api/show; return the parsed JSON response."""
    url = base_url.rstrip("/") + "/api/show"
    req = urllib.request.Request(
        url,
        data=json.dumps({"name": model}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _hidden_dim_from_ollama_response(payload: dict) -> int | None:
    """Extract embedding length from various places Ollama might surface it.

    The shape of /api/show output evolves between Ollama versions; check the
    most likely keys in order.
    """
    # Newer Ollama: model_info dict carries architecture metadata
    info = payload.get("model_info") or {}
    candidates = [
        # gemma family
        "gemma.embedding_length",
        "gemma2.embedding_length",
        "gemma3.embedding_length",
        "gemma4.embedding_length",
        # generic
        "general.embedding_length",
        "embedding_length",
    ]
    for key in candidates:
        if key in info:
            return int(info[key])

    # Older Ollama: parameters as multi-line string
    params = payload.get("parameters") or ""
    for line in params.splitlines():
        line = line.strip()
        if line.startswith("embedding_length"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                pass
    return None


def _try_hf_config(model_id: str) -> int | None:
    """Resolve via HuggingFace if available."""
    try:
        from transformers import AutoConfig  # type: ignore
    except ImportError:
        return None
    try:
        cfg = AutoConfig.from_pretrained(model_id)
        return int(cfg.hidden_size)
    except Exception:
        return None


def _update_yaml(path: str, key: str, value: int) -> tuple[int, int]:
    """Update one YAML config in place. Returns (old_value, new_value)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    old = data.get(key)
    data[key] = value

    backup = path + ".bak"
    shutil.copy2(path, backup)

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    os.replace(tmp, path)
    return old, value


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    p.add_argument("--ollama-model", "--model", dest="ollama_model",
                   default="gemma4:31b", help="Ollama model tag")
    p.add_argument("--hf-fallback", "--hf-id", dest="hf_fallback",
                   default="google/gemma-4-31b",
                   help="HuggingFace model id to try as fallback")
    p.add_argument("--llm-config", default=DEFAULT_LLM_CFG_PATH,
                   help="Path to LLM config YAML to update (default: configs/llm_gemma4.yaml). "
                        "Use configs/llm_deepseek4.yaml for the DeepSeek v4 target (D-015).")
    p.add_argument("--no-write", action="store_true",
                   help="Just print the discovered value; don't update configs")
    args = p.parse_args()

    llm_cfg_path = os.path.abspath(args.llm_config)
    if not os.path.exists(llm_cfg_path):
        print(f"[pin] LLM config not found: {llm_cfg_path}", file=sys.stderr)
        return 2

    hidden = None
    source = None
    try:
        payload = _ollama_show(args.base_url, args.ollama_model)
        hidden = _hidden_dim_from_ollama_response(payload)
        if hidden is not None:
            source = f"ollama @ {args.base_url} ({args.ollama_model})"
    except urllib.error.URLError as e:
        print(f"[pin] Ollama query failed: {e}", file=sys.stderr)

    if hidden is None:
        hidden = _try_hf_config(args.hf_fallback)
        if hidden is not None:
            source = f"huggingface ({args.hf_fallback})"

    if hidden is None:
        print(
            "[pin] Could not determine hidden dim from any source.\n"
            "      Pass it manually:\n"
            f"        python {os.path.relpath(__file__)} --hf-fallback <model-id>\n"
            f"      Or edit {os.path.relpath(llm_cfg_path)} and configs/adapter_default.yaml directly.",
            file=sys.stderr,
        )
        return 1

    print(f"[pin] hidden_dim = {hidden}  (source: {source})")

    if args.no_write:
        return 0

    old_a, _ = _update_yaml(ADAPTER_CFG_PATH, "llm_hidden_dim", hidden)
    old_l, _ = _update_yaml(llm_cfg_path, "hidden_dim", hidden)
    print(
        f"[pin] adapter_default.yaml:  llm_hidden_dim {old_a} -> {hidden}\n"
        f"[pin] {os.path.basename(llm_cfg_path):26s} hidden_dim     {old_l} -> {hidden}"
    )
    print("[pin] Backups left at *.bak; revert with `mv configs/*.yaml.bak configs/*.yaml`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
