from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _optional_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    return None if value in (None, "") else int(value)


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class SchedulerConfig:
    guardrail: float = 0.95
    probe_fraction: float = 0.03
    probe_min_rows: int = 512
    probe_iterations: int = 7
    probe_cap_ms: float = 1.0
    minimum_probe_rows: int = 64
    minimum_probe_nonzeros: int = 512
    candidate_count: int = 6
    hub_split: bool = True
    vec4: bool = True
    cache: bool = True
    replay_only: bool = False
    feature_tile: Optional[int] = None
    warps_per_block: Optional[int] = None
    hub_threshold: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0 < self.guardrail <= 1:
            raise ValueError("guardrail must be in (0, 1]")
        if not 0 < self.probe_fraction <= 1:
            raise ValueError("probe_fraction must be in (0, 1]")
        if (
            self.probe_min_rows < 1
            or self.probe_iterations < 1
            or self.candidate_count < 1
        ):
            raise ValueError(
                "probe row, iteration, and candidate counts must be positive"
            )
        if self.probe_cap_ms < 0:
            raise ValueError("probe_cap_ms cannot be negative")
        if self.feature_tile not in (None, 64, 128):
            raise ValueError("feature_tile must be 64 or 128")
        if self.warps_per_block not in (None, 2, 4, 8):
            raise ValueError("warps_per_block must be 2, 4, or 8")
        if self.hub_threshold is not None and self.hub_threshold < 1:
            raise ValueError("hub_threshold must be positive")

    def tuning_signature(self) -> dict[str, object]:
        return {
            "guardrail": self.guardrail,
            "probe_fraction": self.probe_fraction,
            "probe_min_rows": self.probe_min_rows,
            "probe_iterations": self.probe_iterations,
            "probe_cap_ms": self.probe_cap_ms,
            "minimum_probe_rows": self.minimum_probe_rows,
            "minimum_probe_nonzeros": self.minimum_probe_nonzeros,
            "candidate_count": self.candidate_count,
            "hub_split": self.hub_split,
            "vec4": self.vec4,
            "feature_tile": self.feature_tile,
            "warps_per_block": self.warps_per_block,
            "hub_threshold": self.hub_threshold,
        }

    @classmethod
    def from_env(
        cls, *, guardrail: float = 0.95, candidate_count: int = 6
    ) -> "SchedulerConfig":
        return cls(
            guardrail=float(guardrail),
            probe_fraction=float(os.getenv("AUTOSAGE_PROBE_FRAC", "0.03")),
            probe_min_rows=int(os.getenv("AUTOSAGE_PROBE_MIN_ROWS", "512")),
            probe_iterations=int(os.getenv("AUTOSAGE_PROBE_ITERS", "7")),
            probe_cap_ms=float(os.getenv("AUTOSAGE_PROBE_CAP_MS", "1.0")),
            minimum_probe_rows=int(os.getenv("AUTOSAGE_MIN_PROBE_ROWS", "64")),
            minimum_probe_nonzeros=int(os.getenv("AUTOSAGE_MIN_PROBE_NNZ", "512")),
            candidate_count=int(candidate_count),
            hub_split=_boolean("AUTOSAGE_HUB_CTA", True),
            vec4=_boolean("AUTOSAGE_VEC4", True),
            cache=_boolean("AUTOSAGE_CACHE", True),
            replay_only=_boolean("AUTOSAGE_REPLAY_ONLY", False),
            feature_tile=_optional_int("AUTOSAGE_FTILE"),
            warps_per_block=_optional_int("AUTOSAGE_WPB"),
            hub_threshold=_optional_int("AUTOSAGE_HUB_T"),
        )
