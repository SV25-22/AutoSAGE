from __future__ import annotations

import pytest
import torch

import autosage
from autosage._auto import _csr_row_softmax, sddmm_csr_auto
from autosage.utils.csr import edge_index_to_csr


def test_package_imports_without_native_extension():
    assert autosage.__version__ == "0.1.0"
    assert callable(autosage.spmm_csr)


def test_edge_index_uses_destination_rows_by_default():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    crow, col = edge_index_to_csr(edge_index, 4)
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    output = autosage.spmm_csr(crow, col, None, x)
    assert torch.equal(output[0], x[3])
    assert torch.equal(output[1], x[0])


def test_cpu_spmm_supports_weights_and_empty_rows():
    crow = torch.tensor([0, 2, 2, 3], dtype=torch.long)
    col = torch.tensor([1, 2, 0], dtype=torch.long)
    val = torch.tensor([0.5, 2.0, -1.0])
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    output = autosage.spmm_csr(crow, col, val, x)
    expected = torch.stack((0.5 * x[1] + 2.0 * x[2], torch.zeros(2), -x[0]))
    assert torch.allclose(output, expected)


def test_cpu_auto_path_is_explicit_baseline():
    crow = torch.tensor([0, 1, 2], dtype=torch.long)
    col = torch.tensor([1, 0], dtype=torch.long)
    x = torch.randn(2, 4)
    output, info = autosage.spmm_csr_auto(crow, col, None, x)
    assert output.shape == x.shape
    assert info["choice"] == "baseline"
    assert info["reason"] == "AutoSAGE scheduling is CUDA-only"


def test_cpu_sddmm_and_row_softmax():
    crow = torch.tensor([0, 2, 3], dtype=torch.long)
    col = torch.tensor([0, 1, 0], dtype=torch.long)
    query = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    key = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    scores, info = sddmm_csr_auto(crow, col, query, key)
    expected = torch.tensor([4.0, 7.0, 10.0])
    assert torch.allclose(scores, expected)
    assert info["choice"] == "baseline"
    probabilities = _csr_row_softmax(crow, scores)
    assert torch.allclose(probabilities[:2].sum(), torch.tensor(1.0))
    assert torch.allclose(probabilities[2:], torch.tensor([1.0]))


def test_convolution_returns_tensor_or_telemetry():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.randn(2, 3)
    layer = autosage.AutoSAGEConv(3, 2)
    output = layer(x, edge_index)
    output_with_info, info = layer(x, edge_index, return_info=True)
    assert output.shape == (2, 2)
    assert output_with_info.shape == (2, 2)
    assert info["choice"] == "baseline"


def test_convolution_reorders_weights_with_edges():
    edge_index = torch.tensor([[2, 0, 1], [0, 1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([10.0, 20.0, 30.0])
    x = torch.tensor([[1.0], [2.0], [3.0]])
    layer = autosage.AutoSAGEConv(1, 1, mode="native", bias=False)
    with torch.no_grad():
        layer.linear.weight.fill_(1.0)
    output = layer(x, edge_index, edge_weight)
    assert torch.equal(output, torch.tensor([[90.0], [20.0], [0.0]]))


def test_spmm_rejects_fractional_indices():
    crow = torch.tensor([0.0, 1.0])
    col = torch.tensor([0.0])
    with pytest.raises(TypeError, match="integer indices"):
        autosage.spmm_csr(crow, col, None, torch.ones(1, 1))
