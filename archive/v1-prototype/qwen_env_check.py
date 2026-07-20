#!/usr/bin/env python3

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"torch import failed: {exc}")
        return 1

    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"device_{index}={torch.cuda.get_device_name(index)}")

    try:
        from torch.utils.collect_env import get_pretty_env_info
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        print(f"collect_env_unavailable={exc}")
        return 0

    print()
    print(get_pretty_env_info())
    return 0


if __name__ == "__main__":
    sys.exit(main())
