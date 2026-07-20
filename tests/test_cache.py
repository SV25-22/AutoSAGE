from __future__ import annotations

import json

import torch

from autosage._cache import ScheduleCache, cache_directory, graph_signature


def test_cache_directory_is_resolved_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOSAGE_CACHE_DIR", str(tmp_path))
    assert cache_directory() == tmp_path
    cache = ScheduleCache()
    device = {"type": "cpu", "torch": torch.__version__}
    graph = {"operation": "spmm", "rows": 2}
    result = {"choice": "baseline"}
    cache.store(device, graph, result)
    assert ScheduleCache().lookup(device, graph)["result"] == result
    payload = json.loads((tmp_path / "schedules.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1


def test_graph_signature_preserves_large_integer_counts():
    crow = torch.tensor([0, 1, 3, 3], dtype=torch.long)
    col = torch.tensor([0, 1, 2], dtype=torch.long)
    signature = graph_signature(crow, col, 64, "spmm")
    assert signature["rows"] == 3
    assert signature["nonzeros"] == 3
    assert signature["operation"] == "spmm"
    assert len(signature["structure_hash"]) == 24


def test_graph_signature_separates_weighted_and_dtype_variants():
    crow = torch.tensor([0, 1], dtype=torch.long)
    col = torch.tensor([0], dtype=torch.long)
    unweighted = graph_signature(crow, col, 64, "spmm", dtype=torch.float32)
    weighted = graph_signature(
        crow, col, 64, "spmm", weighted=True, dtype=torch.float32
    )
    double = graph_signature(crow, col, 64, "spmm", dtype=torch.float64)
    assert unweighted != weighted
    assert unweighted != double
