"""Preflight environment validator — run before any long training/eval job.

Goal: catch problems in the first 30 seconds, not 30 minutes into training.

Checks (in order; later checks may depend on earlier ones):

  1. Python version >= 3.11
  2. torch importable, CUDA visible (or explicit --no-gpu skip)
  3. Required modules importable per declared install extras
  4. Scotty / Ollama endpoint reachable + serves the configured model
  5. kirk-pipeline importable OR `kirk` binary on $PATH
  6. Adapter config + LLM config internally consistent
       — adapter.llm_hidden_dim == llm.hidden_dim
       — adapter.n divisible by 16 (kirk-pipeline Layer-1 block size)
  7. Adapter checkpoint loadable (if path provided)
  8. GPU has bf16 support (Gemma 4 31B needs it)

Each check has:
  - Severity: ERROR (blocks long jobs) or WARN (allows --strict to error)
  - Fix hint: concrete command/file edit to resolve

Exit code:
  0 → all checks pass (or warns only and --strict not set)
  1 → at least one ERROR or (any WARN under --strict)

Usage:
    python scripts/preflight.py
    python scripts/preflight.py --strict
    python scripts/preflight.py --adapter-config configs/adapter_default.yaml
    python scripts/preflight.py --no-gpu             # skip GPU/CUDA checks
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import yaml


# ANSI colors for terminal output. Disabled if stdout isn't a TTY.
_COLOR = sys.stdout.isatty()
G = "\033[32m" if _COLOR else ""
Y = "\033[33m" if _COLOR else ""
R = "\033[31m" if _COLOR else ""
B = "\033[1m" if _COLOR else ""
END = "\033[0m" if _COLOR else ""


@dataclass
class CheckResult:
    name: str
    ok: bool
    severity: str = "ERROR"  # ERROR | WARN
    detail: str = ""
    fix_hint: str = ""


@dataclass
class Preflight:
    results: list[CheckResult] = field(default_factory=list)

    def ok(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult(name, True, detail=detail))

    def warn(self, name: str, detail: str, fix_hint: str = "") -> None:
        self.results.append(CheckResult(name, False, severity="WARN", detail=detail, fix_hint=fix_hint))

    def fail(self, name: str, detail: str, fix_hint: str = "") -> None:
        self.results.append(CheckResult(name, False, severity="ERROR", detail=detail, fix_hint=fix_hint))

    def render(self) -> int:
        """Print results, return exit code."""
        any_error = False
        any_warn = False
        for r in self.results:
            if r.ok:
                tag = f"{G}[ OK ]{END}"
                print(f"{tag} {r.name:<48s} {r.detail}")
            elif r.severity == "WARN":
                tag = f"{Y}[WARN]{END}"
                print(f"{tag} {r.name:<48s} {r.detail}")
                if r.fix_hint:
                    print(f"       {Y}fix:{END} {r.fix_hint}")
                any_warn = True
            else:
                tag = f"{R}[FAIL]{END}"
                print(f"{tag} {r.name:<48s} {r.detail}")
                if r.fix_hint:
                    print(f"       {R}fix:{END} {r.fix_hint}")
                any_error = True
        print()
        n_ok = sum(1 for r in self.results if r.ok)
        n_warn = sum(1 for r in self.results if not r.ok and r.severity == "WARN")
        n_err = sum(1 for r in self.results if not r.ok and r.severity == "ERROR")
        print(f"{B}{n_ok} ok · {n_warn} warnings · {n_err} errors{END}")
        return n_err, n_warn


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_version(p: Preflight) -> None:
    v = sys.version_info
    if (v.major, v.minor) >= (3, 11):
        p.ok("Python version", detail=f"{v.major}.{v.minor}.{v.micro}")
    else:
        p.fail(
            "Python version",
            detail=f"have {v.major}.{v.minor}, need >= 3.11",
            fix_hint="install python 3.11+; in a venv: `python3.11 -m venv .venv`",
        )


def check_torch(p: Preflight, want_gpu: bool) -> Optional[object]:
    try:
        import torch  # type: ignore
    except ImportError:
        p.fail(
            "torch importable",
            detail="not installed",
            fix_hint="pip install -e \".[llm]\"",
        )
        return None
    p.ok("torch importable", detail=f"v{torch.__version__}")

    if want_gpu:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if n else "n/a"
            p.ok("CUDA available", detail=f"{n} device(s); first: {name}")
            # bf16 support (Gemma 4 31B serving dtype)
            if torch.cuda.is_bf16_supported():
                p.ok("bf16 GPU support")
            else:
                p.warn(
                    "bf16 GPU support",
                    detail="device doesn't report bf16; Gemma 4 31B will fall back to fp16",
                    fix_hint="OK on consumer GPUs; for production use H100/A100",
                )
        else:
            p.fail(
                "CUDA available",
                detail="no CUDA devices detected",
                fix_hint="run on a GPU host or pass --no-gpu to skip Pipeline C training",
            )
    return torch


def check_imports_for_extras(p: Preflight) -> None:
    """Each extra contributes a critical import. Confirm all are present."""
    pairs = [
        ("transformers", "[llm]"),
        ("httpx", "[scotty]"),
        ("yaml", "stdlib? (pyyaml)"),
        ("numpy", "[default]"),
    ]
    for mod_name, extra in pairs:
        try:
            importlib.import_module(mod_name)
            p.ok(f"import {mod_name}", detail=extra)
        except ImportError:
            p.warn(
                f"import {mod_name}",
                detail=f"missing — required by extra {extra}",
                fix_hint=f"pip install -e \".{extra}\"",
            )


def check_scotty(p: Preflight, base_url: str, model: str) -> None:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as e:
        p.warn(
            "Scotty/Ollama reachable",
            detail=f"{base_url} unreachable: {e}",
            fix_hint="start Ollama (`ollama serve`) or update --scotty-base-url",
        )
        return
    p.ok("Scotty/Ollama reachable", detail=base_url)

    models = [m.get("name") for m in data.get("models", [])]
    if model in models:
        p.ok("Scotty model available", detail=model)
    elif any(m and m.startswith(model.split(":")[0]) for m in models):
        p.warn(
            "Scotty model available",
            detail=f"{model!r} not loaded; similar tags present: {[m for m in models if m and m.startswith(model.split(':')[0])]}",
            fix_hint=f"`ollama pull {model}` or update --scotty-model",
        )
    else:
        p.warn(
            "Scotty model available",
            detail=f"{model!r} not in {models}",
            fix_hint=f"`ollama pull {model}` or update --scotty-model",
        )


def check_kirk(p: Preflight) -> None:
    # Two acceptable paths:
    #  1. kirk_pipeline importable in this Python
    #  2. `kirk` binary on $PATH
    importable = True
    try:
        importlib.import_module("kirk_pipeline")
    except ImportError:
        importable = False

    binary = shutil.which("kirk")

    if importable and binary:
        p.ok("Kirk available", detail="kirk_pipeline (Python) AND kirk binary on $PATH")
    elif importable:
        p.ok("Kirk available", detail="kirk_pipeline (Python import only)")
    elif binary:
        p.ok("Kirk available", detail=f"kirk binary at {binary}")
    else:
        p.warn(
            "Kirk available",
            detail="neither kirk_pipeline importable nor kirk binary on $PATH",
            fix_hint=(
                "uv add /dist/kirk_pipeline-*.whl  (production wheel)\n"
                "       OR add the kirk-runner repo to PATH\n"
                "       OR pass --kirk-backend stub to use the synthetic client"
            ),
        )


def check_configs(p: Preflight, adapter_path: str, llm_path: str) -> None:
    if not os.path.exists(adapter_path):
        p.fail(
            "adapter config exists",
            detail=adapter_path,
            fix_hint=f"create {adapter_path} or pass --adapter-config",
        )
        return
    p.ok("adapter config exists", detail=adapter_path)

    if not os.path.exists(llm_path):
        p.fail(
            "llm config exists",
            detail=llm_path,
            fix_hint=f"create {llm_path} or pass --llm-config",
        )
        return
    p.ok("llm config exists", detail=llm_path)

    with open(adapter_path) as f:
        a = yaml.safe_load(f)
    with open(llm_path) as f:
        l = yaml.safe_load(f)

    # Cross-config consistency
    if a.get("llm_hidden_dim") == l.get("hidden_dim"):
        p.ok("config: hidden dims agree", detail=f"hidden_dim={a.get('llm_hidden_dim')}")
    else:
        p.fail(
            "config: hidden dims agree",
            detail=f"adapter.llm_hidden_dim={a.get('llm_hidden_dim')} != "
                   f"llm.hidden_dim={l.get('hidden_dim')}",
            fix_hint="run `python scripts/pin_gemma_hidden_dim.py` to auto-align",
        )

    # Layer-1 block size compatibility
    n = a.get("n")
    if isinstance(n, int):
        if n % 16 == 0:
            p.ok("config: n divisible by 16", detail=f"n={n}")
        else:
            p.warn(
                "config: n divisible by 16",
                detail=f"n={n} doesn't tile evenly into kirk-pipeline's 16×16 blocks",
                fix_hint="use n=16, 32, 48, ... in adapter_default.yaml",
            )

    # Sanity bounds on hidden dim
    h = a.get("llm_hidden_dim")
    if isinstance(h, int) and (h < 256 or h > 16384):
        p.warn(
            "config: hidden_dim sane",
            detail=f"hidden_dim={h} outside typical range [256, 16384]",
            fix_hint="verify against actual model.config.hidden_size",
        )


def check_adapter_checkpoint(p: Preflight, ckpt_path: Optional[str]) -> None:
    if ckpt_path is None:
        return  # not requested
    if not os.path.exists(ckpt_path):
        p.warn(
            "adapter checkpoint exists",
            detail=ckpt_path,
            fix_hint="`make train-stub` to produce one, or pass --adapter to a real checkpoint",
        )
        return
    p.ok("adapter checkpoint exists", detail=ckpt_path)
    try:
        import torch  # type: ignore
        ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        keys = list(ck.keys()) if isinstance(ck, dict) else []
        if "adapter_state_dict" in keys:
            p.ok("checkpoint has adapter_state_dict")
        else:
            p.fail(
                "checkpoint has adapter_state_dict",
                detail=f"keys: {keys}",
                fix_hint="re-save via Trainer.save() — old format won't load",
            )
    except Exception as e:
        p.fail(
            "checkpoint loadable",
            detail=str(e),
            fix_hint="checkpoint corrupt; re-train",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-gpu", action="store_true", help="skip CUDA / bf16 checks")
    p.add_argument("--strict", action="store_true", help="exit non-zero on any warning")
    p.add_argument("--adapter-config", default="configs/adapter_default.yaml")
    p.add_argument("--llm-config", default="configs/llm_gemma4.yaml")
    p.add_argument("--adapter-checkpoint", default=None)
    p.add_argument("--scotty-base-url", default="http://127.0.0.1:11434")
    p.add_argument("--scotty-model", default="gemma4:31b")
    args = p.parse_args()

    pf = Preflight()
    print(f"{B}wonderwall preflight{END}\n")

    check_python_version(pf)
    check_torch(pf, want_gpu=not args.no_gpu)
    check_imports_for_extras(pf)
    check_scotty(pf, args.scotty_base_url, args.scotty_model)
    check_kirk(pf)
    check_configs(pf, args.adapter_config, args.llm_config)
    check_adapter_checkpoint(pf, args.adapter_checkpoint)

    n_err, n_warn = pf.render()
    if n_err > 0:
        return 1
    if args.strict and n_warn > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
