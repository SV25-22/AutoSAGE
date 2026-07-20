from __future__ import annotations

import torch

from autosage._auto import _spmm_candidates
from autosage.config import SchedulerConfig


def test_candidates_only_contain_effective_parameters():
    crow = torch.tensor([0, 1, 3, 3, 10], dtype=torch.long)
    candidates = _spmm_candidates(crow, 64, SchedulerConfig(candidate_count=18))
    assert candidates
    assert len(candidates) == len(set(candidates))
    assert all(tile in {64, 128} for tile, _, _ in candidates)
    assert all(warps in {2, 4, 8} for _, warps, _ in candidates)
    assert all(threshold > 0 for _, _, threshold in candidates)


def test_environment_overrides_are_read_for_each_configuration(monkeypatch):
    monkeypatch.setenv("AUTOSAGE_FTILE", "64")
    monkeypatch.setenv("AUTOSAGE_WPB", "8")
    monkeypatch.setenv("AUTOSAGE_HUB_T", "16")
    first = SchedulerConfig.from_env()
    monkeypatch.setenv("AUTOSAGE_HUB_T", "32")
    second = SchedulerConfig.from_env()
    assert first.hub_threshold == 16
    assert second.hub_threshold == 32


def test_tuning_signature_tracks_kernel_policy(monkeypatch):
    first = SchedulerConfig.from_env().tuning_signature()
    monkeypatch.setenv("AUTOSAGE_VEC4", "false")
    second = SchedulerConfig.from_env().tuning_signature()
    assert first["vec4"] is True
    assert second["vec4"] is False
    assert first != second
