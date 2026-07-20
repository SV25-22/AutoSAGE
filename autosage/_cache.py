from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch


CACHE_SCHEMA_VERSION = 1


def cache_directory() -> Path:
    configured = os.getenv("AUTOSAGE_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "autosage"


def device_signature(device: Optional[int] = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"type": "cpu", "torch": torch.__version__}
    index = torch.cuda.current_device() if device is None else device
    properties = torch.cuda.get_device_properties(index)
    return {
        "type": "cuda",
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "multiprocessors": properties.multi_processor_count,
        "total_memory": int(properties.total_memory),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }


def _sample_hash(crow: torch.Tensor, col: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (crow, col):
        count = tensor.numel()
        if count == 0:
            continue
        width = min(count, 2048)
        sample = (
            tensor
            if count <= width
            else torch.cat((tensor[: width // 2], tensor[-width // 2 :]))
        )
        byte_view = sample.detach().contiguous().cpu().view(torch.uint8)
        digest.update(bytes(byte_view.tolist()))
    return digest.hexdigest()[:24]


def graph_signature(
    crow: torch.Tensor,
    col: torch.Tensor,
    feature_width: int,
    operation: str,
    *,
    weighted: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, Any]:
    degrees = crow[1:] - crow[:-1]
    if degrees.numel():
        quantiles = (
            torch.quantile(
                degrees.to(torch.float64),
                torch.tensor(
                    [0.5, 0.9, 0.99], dtype=torch.float64, device=degrees.device
                ),
            )
            .round()
            .to(torch.long)
            .cpu()
            .tolist()
        )
    else:
        quantiles = [0, 0, 0]
    return {
        "operation": operation,
        "feature_width": int(feature_width),
        "dtype": None if dtype is None else str(dtype),
        "weighted": bool(weighted),
        "rows": int(crow.numel() - 1),
        "nonzeros": int(col.numel()),
        "degree_q50": int(quantiles[0]),
        "degree_q90": int(quantiles[1]),
        "degree_q99": int(quantiles[2]),
        "structure_hash": _sample_hash(crow, col),
    }


class ScheduleCache:
    def __init__(self, path: Optional[os.PathLike[str] | str] = None):
        self.path = (
            Path(path) if path is not None else cache_directory() / "schedules.json"
        )
        self._records = self._load()

    @staticmethod
    def _key(device: dict[str, Any], graph: dict[str, Any]) -> str:
        return json.dumps(
            {"device": device, "graph": graph}, sort_keys=True, separators=(",", ":")
        )

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            return {}
        records = payload.get("records", {})
        return records if isinstance(records, dict) else {}

    def lookup(
        self, device: dict[str, Any], graph: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        return self._records.get(self._key(device, graph))

    def store(
        self, device: dict[str, Any], graph: dict[str, Any], result: dict[str, Any]
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": int(time.time()),
            "device": device,
            "graph": graph,
            "result": result,
        }
        self._records[self._key(device, graph)] = record
        payload = {"schema_version": CACHE_SCHEMA_VERSION, "records": self._records}
        handle, temporary = tempfile.mkstemp(
            prefix="schedules-", suffix=".json", dir=self.path.parent
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def log_probe(event: dict[str, Any]) -> None:
    if os.getenv("AUTOSAGE_LOG_PROBES", "1") != "1":
        return
    path = cache_directory() / "probe_logs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "ScheduleCache",
    "cache_directory",
    "device_signature",
    "graph_signature",
    "log_probe",
]
