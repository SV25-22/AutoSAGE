import torch
from typing import Optional


# --- native loader ---
import os, glob, torch
USE_DEBUG = bool(int(os.environ.get("AUTOSAGE_DEBUG", "0")))

def _ensure_native_loaded():
    # 1) explicit hint via env (recommended in dev)
    hint = os.environ.get("AUTOSAGE_NATIVE_PATH", "")
    if hint:
        try:
            torch.ops.load_library(hint)
            if USE_DEBUG: print(f"[autosage] loaded native from {hint}")
            return
        except OSError as e:
            if USE_DEBUG: print(f"[autosage] failed to load {hint}: {e}")
    # 2) by name (works if it’s on ld path)
    try:
        torch.ops.load_library("autosage_cuda")
        if USE_DEBUG: print("[autosage] loaded native by name autosage_cuda")
        return
    except OSError:
        pass
    # 3) search common build folders (repo checkout)
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for so in (
        glob.glob(os.path.join(here, "cmake-build", "libautosage_cuda.*"))
        + glob.glob(os.path.join(here, "build", "**", "autosage_cuda.*"), recursive=True)
    ):
        try:
            torch.ops.load_library(so)
            if USE_DEBUG: print(f"[autosage] loaded native by search: {so}")
            return
        except OSError:
            continue
    if USE_DEBUG: print("[autosage] native library NOT loaded; CPU fallback will be used")

_ensure_native_loaded()
# --- end native loader ---


def _cpu_reference_spmm(crow, col, val, x):
    n, f = x.shape
    y = x.new_zeros((n, f))
    crow_l = crow.tolist()
    col_l = col.tolist()
    has_w = val is not None
    w = val if has_w else None
    for r in range(n):
        s, e = crow_l[r], crow_l[r+1]
        if s == e:
            continue
        cols = col_l[s:e]
        if has_w:
            ws = w[s:e].to(x.dtype).unsqueeze(1)
            y[r] = (x[cols] * ws).sum(dim=0)
        else:
            y[r] = x[cols].sum(dim=0)
    return y

def spmm_csr(crow: torch.Tensor,
             col: torch.Tensor,
             val: Optional[torch.Tensor],
             x: torch.Tensor) -> torch.Tensor:
    crow = crow.to(dtype=torch.long, non_blocking=True).contiguous()
    col  = col.to(dtype=torch.long, non_blocking=True).contiguous()
    if val is not None:
        val = val.to(dtype=torch.float32, non_blocking=True).contiguous()
    x = x.contiguous()
    if x.is_cuda and torch.cuda.is_available():
        try:
            if x.dtype != torch.float32:
                raise RuntimeError("P1 CUDA path supports float32 features only.")
            return torch.ops.autosage.spmm_csr(crow, col, val, x)
        except (RuntimeError, AttributeError) as e:
            # Fall through to CPU ref on any extension error
            pass
    return _cpu_reference_spmm(crow.cpu(), col.cpu(), None if val is None else val.cpu(), x.cpu()).to(x.device)


# P2 auto path
from ._auto import spmm_csr_auto
