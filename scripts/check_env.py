from __future__ import annotations

import importlib
import platform

import torch

REQUIRED_MODULES = [
    "yaml",
    "numpy",
    "tqdm",
    "transformers",
    "pycocoevalcap",
    "nltk",
    "sacrebleu",
    "rouge_score",
]


def main() -> None:
    print("Python:", platform.python_version())
    print("Torch:", torch.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())

    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
            print(f"[OK] {name}")
        except Exception as exc:
            print(f"[MISSING] {name}: {exc}")


if __name__ == "__main__":
    main()
