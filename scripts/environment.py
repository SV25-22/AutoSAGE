#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import subprocess
import sys

import torch


def command(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            arguments, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


payload = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "nvcc": command("nvcc", "--version"),
    "nvidia_smi": command(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ),
    "git_commit": command("git", "rev-parse", "HEAD"),
}
print(json.dumps(payload, indent=2))
