from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


_LOAD_ATTEMPTED = False
_LOAD_ERROR: Optional[BaseException] = None


def _candidate_libraries() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    names = (
        "libautosage_cuda.so",
        "libautosage_cuda.dylib",
        "autosage_cuda.dll",
        "autosage_cuda.pyd",
    )
    paths: list[Path] = []
    explicit = os.getenv("AUTOSAGE_NATIVE_PATH")
    if explicit:
        paths.append(Path(explicit).expanduser())
    for directory in (root / "build", Path(__file__).parent / "lib"):
        paths.extend(directory / name for name in names)
    return paths


def load_native() -> bool:
    global _LOAD_ATTEMPTED, _LOAD_ERROR
    if native_available():
        return True
    if _LOAD_ATTEMPTED:
        return False
    _LOAD_ATTEMPTED = True

    errors: list[BaseException] = []
    for path in _candidate_libraries():
        if not path.is_file():
            continue
        try:
            torch.ops.load_library(str(path))
            return native_available()
        except (OSError, RuntimeError) as exc:
            errors.append(exc)
    try:
        torch.ops.load_library("autosage_cuda")
        return native_available()
    except (OSError, RuntimeError) as exc:
        errors.append(exc)

    _LOAD_ERROR = (
        errors[-1]
        if errors
        else FileNotFoundError(
            "AutoSAGE native library was not found. Build it with scripts/build.sh."
        )
    )
    return False


def native_op(name: str):
    namespace = getattr(torch.ops, "autosage", None)
    if namespace is None:
        return None
    try:
        return getattr(namespace, name)
    except AttributeError:
        return None


def native_available() -> bool:
    return native_op("spmm_csr") is not None


def native_load_error() -> Optional[BaseException]:
    return _LOAD_ERROR


def _validate_csr(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor],
    x: torch.Tensor,
) -> None:
    if crow.ndim != 1 or col.ndim != 1:
        raise ValueError("crow and col must be one-dimensional")
    if crow.dtype != torch.long or col.dtype != torch.long:
        raise TypeError("crow and col must use torch.long indices")
    if x.ndim != 2:
        raise ValueError("x must have shape [N, F]")
    if not x.is_floating_point():
        raise TypeError("x must use a floating-point dtype")
    if crow.numel() != x.size(0) + 1:
        raise ValueError("crow must contain N + 1 entries")
    if val is not None and (val.ndim != 1 or val.numel() != col.numel()):
        raise ValueError("val must contain one weight per nonzero")
    if crow.device != col.device or crow.device != x.device:
        raise ValueError("crow, col, and x must be on the same device")
    if val is not None and val.device != x.device:
        raise ValueError("val and x must be on the same device")


def _normalize_csr(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor],
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    integer_types = {torch.int32, torch.int64}
    if crow.dtype not in integer_types or col.dtype not in integer_types:
        raise TypeError("crow and col must use integer indices")
    crow = crow.to(dtype=torch.long).contiguous()
    col = col.to(dtype=torch.long).contiguous()
    x = x.contiguous()
    if val is not None:
        val = val.to(dtype=x.dtype).contiguous()
    _validate_csr(crow, col, val, x)
    return crow, col, val, x


def _cpu_spmm(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    values = (
        val
        if val is not None
        else torch.ones(col.numel(), dtype=x.dtype, device=x.device)
    )
    matrix = torch.sparse_csr_tensor(
        crow,
        col,
        values.to(dtype=x.dtype),
        size=(x.size(0), x.size(0)),
        dtype=x.dtype,
        device=x.device,
    )
    return torch.sparse.mm(matrix, x)


def spmm_csr(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    crow, col, val, x = _normalize_csr(crow, col, val, x)

    if not x.is_cuda:
        return _cpu_spmm(crow, col, val, x)
    if x.dtype != torch.float32:
        raise TypeError("AutoSAGE CUDA kernels currently support float32 features only")
    if not load_native():
        detail = f": {_LOAD_ERROR}" if _LOAD_ERROR else ""
        raise RuntimeError(f"AutoSAGE CUDA extension is unavailable{detail}")
    op = native_op("spmm_csr")
    if op is None:
        raise RuntimeError("native operator autosage::spmm_csr is not registered")
    return op(crow, col, val, x)


def spmm_csr_auto(*args, **kwargs):
    from ._auto import spmm_csr_auto as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "load_native",
    "native_available",
    "native_load_error",
    "native_op",
    "spmm_csr",
    "spmm_csr_auto",
]
