"""CLI for training the projection adapter.

Usage:

    python scripts/train.py \\
        --adapter-config configs/adapter_default.yaml \\
        --llm-config configs/llm_gemma4.yaml \\
        --train-config configs/train_default.yaml
"""
from __future__ import annotations

import argparse

import yaml

from wonderwall.adapter import KirkProjectionAdapter
from wonderwall.injection import EmbeddingInjectionLLM, LLMConfig
from wonderwall.interfaces import AdapterConfig
from wonderwall.kirk_client import StubKirkClient
from wonderwall.train import TrainConfig, Trainer


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-config", required=True)
    p.add_argument("--llm-config", required=True)
    p.add_argument("--train-config", required=True)
    p.add_argument("--use-stub-kirk", action="store_true",
                   help="Use the synthetic Kirk client for plumbing-test training runs")
    args = p.parse_args()

    adapter_cfg = AdapterConfig(**load_yaml(args.adapter_config))
    llm_cfg = LLMConfig(**load_yaml(args.llm_config))
    train_cfg = TrainConfig(**load_yaml(args.train_config))

    if args.use_stub_kirk:
        kirk = StubKirkClient(n=adapter_cfg.n, use_complex=adapter_cfg.use_complex)
    else:
        raise NotImplementedError(
            "Real Kirk wiring is owned by Spencer's team. "
            "Pass --use-stub-kirk for plumbing tests or implement InProcessKirkClient."
        )

    adapter = KirkProjectionAdapter(adapter_cfg)
    llm = EmbeddingInjectionLLM(llm_cfg)

    trainer = Trainer(kirk=kirk, adapter=adapter, llm=llm, config=train_cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
