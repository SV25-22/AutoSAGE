from __future__ import annotations

import pytest
import torch

from autosage import load_native, native_available
from autosage._auto import sddmm_csr_auto


CUDA_READY = torch.cuda.is_available() and load_native() and native_available()
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not CUDA_READY, reason="CUDA extension unavailable"),
]


def random_csr(rows: int, degree: int, seed: int = 0):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    counts = torch.randint(
        0, 2 * degree + 1, (rows,), generator=generator, device="cuda"
    )
    crow = torch.cat(
        (torch.zeros(1, device="cuda", dtype=torch.long), counts.cumsum(0))
    )
    col = torch.randint(
        0, rows, (int(crow[-1].item()),), generator=generator, device="cuda"
    )
    return crow, col


@pytest.mark.parametrize("features", [32, 64, 113, 128, 256])
def test_spmm_native_matches_torch(features):
    crow, col = random_csr(256, 12)
    x = torch.randn(256, features, device="cuda")
    values = torch.rand(col.numel(), device="cuda")
    matrix = torch.sparse_csr_tensor(crow, col, values, size=(256, 256))
    expected = torch.sparse.mm(matrix, x)
    actual = torch.ops.autosage.spmm_csr(crow, col, values, x)
    assert torch.allclose(actual, expected, rtol=1e-4, atol=3e-3)


def test_split_matches_torch_on_hub_graph():
    rows = 512
    counts = torch.full((rows,), 8, device="cuda", dtype=torch.long)
    counts[0] = 400
    crow = torch.cat(
        (torch.zeros(1, device="cuda", dtype=torch.long), counts.cumsum(0))
    )
    col = torch.randint(0, rows, (int(crow[-1].item()),), device="cuda")
    x = torch.randn(rows, 128, device="cuda")
    matrix = torch.sparse_csr_tensor(
        crow, col, torch.ones(col.numel(), device="cuda"), size=(rows, rows)
    )
    expected = torch.sparse.mm(matrix, x)
    actual = torch.ops.autosage.spmm_csr_split(crow, col, None, x, 128, 8, 128)
    assert torch.allclose(actual, expected, rtol=1e-4, atol=3e-3)


def test_sddmm_matches_gather_dot():
    crow, col = random_csr(128, 8)
    query = torch.randn(128, 64, device="cuda")
    key = torch.randn(128, 64, device="cuda")
    actual, _ = sddmm_csr_auto(crow, col, query, key)
    rows = torch.repeat_interleave(
        torch.arange(128, device="cuda"), crow[1:] - crow[:-1]
    )
    expected = (query[rows] * key[col]).sum(dim=1)
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)
