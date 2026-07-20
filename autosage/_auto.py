from __future__ import annotations

import os
import statistics
import time
from typing import Any, Callable, Optional

import torch

from ._cache import ScheduleCache, device_signature, graph_signature, log_probe
from .config import SchedulerConfig
from .ops import _normalize_csr, load_native, native_op, spmm_csr


AUTOTUNE_SCHEMA_VERSION = 3


def _time_cuda(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    cap_ms: Optional[float] = None,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for _ in range(max(0, warmup)):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    output: Optional[torch.Tensor] = None
    total = 0.0
    for _ in range(max(1, iterations)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        samples.append(elapsed)
        total += elapsed
        if cap_ms is not None and total >= max(0.0, cap_ms):
            break
    torch.cuda.synchronize()
    return {
        "best_ms": min(samples),
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "iterations": len(samples),
        "total_ms": total,
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
        "output": output,
    }


def _prepare_spmm_baseline(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor],
    x: torch.Tensor,
) -> tuple[Callable[[], torch.Tensor], str]:
    values = (
        val
        if val is not None
        else torch.ones(col.numel(), device=x.device, dtype=x.dtype)
    )
    matrix = torch.sparse_csr_tensor(
        crow,
        col,
        values.to(dtype=x.dtype),
        size=(crow.numel() - 1, x.size(0)),
        device=x.device,
        dtype=x.dtype,
    )
    return lambda: torch.sparse.mm(matrix, x), "torch.sparse.mm(csr)"


def _prepare_sddmm_baseline(
    crow: torch.Tensor,
    col: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[Callable[[], torch.Tensor], str]:
    counts = (crow[1:] - crow[:-1]).to(torch.long)
    rows = torch.repeat_interleave(
        torch.arange(crow.numel() - 1, device=col.device, dtype=torch.long), counts
    )
    return (
        lambda: (query.index_select(0, rows) * key.index_select(0, col)).sum(dim=1),
        "gather-dot",
    )


def _induced_sample_csr(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor],
    *,
    fraction: float,
    minimum_rows: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    rows = crow.numel() - 1
    sample_size = min(max(int(rows * fraction), minimum_rows, 1), rows)
    generator = torch.Generator(device=crow.device)
    generator.manual_seed(seed)
    selected = (
        torch.randperm(rows, generator=generator, device=crow.device)[:sample_size]
        .sort()
        .values
    )

    mapping = torch.full((rows,), -1, dtype=torch.long, device=crow.device)
    mapping[selected] = torch.arange(sample_size, device=crow.device)
    starts = crow.index_select(0, selected)
    ends = crow.index_select(0, selected + 1)
    counts = ends - starts
    candidate_count = int(counts.sum().item())

    if candidate_count == 0:
        sampled_crow = torch.zeros(
            sample_size + 1, dtype=torch.long, device=crow.device
        )
        sampled_col = torch.empty(0, dtype=torch.long, device=col.device)
        sampled_val = (
            None if val is None else torch.empty(0, dtype=val.dtype, device=val.device)
        )
        return sampled_crow, sampled_col, sampled_val, selected

    sampled_rows = torch.repeat_interleave(
        torch.arange(sample_size, device=crow.device, dtype=torch.long), counts
    )
    row_offsets = torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
    positions = (
        starts.index_select(0, sampled_rows)
        + torch.arange(candidate_count, device=crow.device, dtype=torch.long)
        - row_offsets
    )
    mapped_columns = mapping.index_select(0, col.index_select(0, positions))
    keep = mapped_columns.ge(0)
    kept_rows = sampled_rows[keep]
    sampled_counts = torch.bincount(kept_rows, minlength=sample_size)
    sampled_crow = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=crow.device),
            sampled_counts.cumsum(0),
        )
    )
    sampled_col = mapped_columns[keep].contiguous()
    sampled_val = (
        None if val is None else val.index_select(0, positions)[keep].contiguous()
    )
    return sampled_crow, sampled_col, sampled_val, selected


def _degree_quantiles(crow: torch.Tensor) -> tuple[int, int, int]:
    degrees = crow[1:] - crow[:-1]
    if degrees.numel() == 0:
        return 1, 1, 1
    values = (
        torch.quantile(
            degrees.to(torch.float64),
            torch.tensor([0.9, 0.95, 0.99], dtype=torch.float64, device=degrees.device),
        )
        .round()
        .clamp_min(1)
        .to(torch.long)
        .cpu()
        .tolist()
    )
    return int(values[0]), int(values[1]), int(values[2])


def _spmm_candidates(
    crow: torch.Tensor,
    feature_width: int,
    config: SchedulerConfig,
) -> list[tuple[int, int, int]]:
    q90, q95, q99 = _degree_quantiles(crow)
    degrees = crow[1:] - crow[:-1]
    q50 = (
        int(torch.quantile(degrees.to(torch.float64), 0.5).round().item())
        if degrees.numel()
        else 1
    )
    skew = q99 / max(q50, 1)
    preferred_warps = 8 if skew >= 4 else 4 if skew >= 2 else 2
    preferred_tile = 64 if feature_width <= 64 else 128
    preferred_threshold = q95

    tiles = [config.feature_tile] if config.feature_tile is not None else [64, 128]
    warps = (
        [config.warps_per_block] if config.warps_per_block is not None else [2, 4, 8]
    )
    thresholds = (
        [config.hub_threshold]
        if config.hub_threshold is not None
        else list(dict.fromkeys((q95, q90, q99)))
    )

    def score(candidate: tuple[int, int, int]) -> tuple[float, tuple[int, int, int]]:
        tile, warp_count, threshold = candidate
        value = abs(tile - preferred_tile) / 64
        value += abs(warp_count - preferred_warps) / 6
        value += abs(threshold - preferred_threshold) / max(preferred_threshold, 1)
        return value, candidate

    candidates = {
        (int(tile), int(warp_count), max(1, int(threshold)))
        for tile in tiles
        for warp_count in warps
        for threshold in thresholds
    }
    ordered = sorted(candidates, key=score)
    return ordered[: max(1, config.candidate_count)]


def _cache_hit_allowed(result: dict[str, Any], config: SchedulerConfig) -> bool:
    if result.get("schema_version") != AUTOTUNE_SCHEMA_VERSION:
        return False
    if config.replay_only:
        return True
    return result.get("configuration") == config.tuning_signature()


def _cached_native_available(
    operation: str, choice: str, candidate: Optional[list[int]]
) -> bool:
    if choice != "autosage":
        return True
    suffix = "_split" if candidate is not None else ""
    return native_op(f"{operation}{suffix}") is not None


def _cpu_info(operation: str) -> dict[str, Any]:
    return {
        "operation": operation,
        "schema_version": AUTOTUNE_SCHEMA_VERSION,
        "choice": "baseline",
        "baseline": "torch CPU",
        "from_cache": False,
        "reason": "AutoSAGE scheduling is CUDA-only",
    }


def _run_spmm_choice(
    choice: str,
    candidate: Optional[list[int]],
    crow: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    if choice == "autosage":
        split = native_op("spmm_csr_split")
        core = native_op("spmm_csr")
        if split is not None and candidate is not None:
            return split(crow, col, val, x, *map(int, candidate))
        if core is not None:
            return core(crow, col, val, x)
    baseline, _ = _prepare_spmm_baseline(crow, col, val, x)
    return baseline()


def spmm_csr_auto(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor],
    x: torch.Tensor,
    *,
    guardrail: float = 0.95,
    k: int = 6,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    crow, col, val, x = _normalize_csr(crow, col, val, x)
    if not x.is_cuda:
        return spmm_csr(crow, col, val, x), _cpu_info("spmm")

    if x.dtype != torch.float32:
        baseline, name = _prepare_spmm_baseline(crow, col, val, x)
        return baseline(), {
            **_cpu_info("spmm"),
            "baseline": name,
            "reason": "AutoSAGE CUDA kernels require float32 features",
        }

    load_native()
    config = SchedulerConfig.from_env(guardrail=guardrail, candidate_count=k)
    device = device_signature()
    graph = graph_signature(
        crow,
        col,
        x.size(1),
        "spmm",
        weighted=val is not None,
        dtype=x.dtype,
    )
    cache = ScheduleCache()

    if config.cache:
        record = cache.lookup(device, graph)
        if record is not None and _cache_hit_allowed(record.get("result", {}), config):
            result = record["result"]
            choice = result.get("choice", "baseline")
            candidate = result.get("candidate")
            if _cached_native_available("spmm_csr", choice, candidate):
                output = _run_spmm_choice(choice, candidate, crow, col, val, x)
                return output, {**result, "from_cache": True}

    if config.replay_only:
        output = _run_spmm_choice("baseline", None, crow, col, val, x)
        return output, {
            "operation": "spmm",
            "schema_version": AUTOTUNE_SCHEMA_VERSION,
            "choice": "baseline",
            "baseline": "torch.sparse.mm(csr)",
            "from_cache": False,
            "replay_only": True,
            "reason": "cache miss",
        }

    torch.cuda.synchronize()
    sample_start = time.perf_counter()
    sampled_crow, sampled_col, sampled_val, selected = _induced_sample_csr(
        crow,
        col,
        val,
        fraction=config.probe_fraction,
        minimum_rows=config.probe_min_rows,
        seed=seed,
    )
    sampled_x = x.index_select(0, selected).contiguous()
    torch.cuda.synchronize()
    sample_ms = (time.perf_counter() - sample_start) * 1000.0

    baseline, baseline_name = _prepare_spmm_baseline(
        sampled_crow, sampled_col, sampled_val, sampled_x
    )
    baseline_timing = _time_cuda(
        baseline,
        warmup=1,
        iterations=config.probe_iterations,
    )

    candidates: list[dict[str, Any]] = []
    chosen_candidate: Optional[list[int]] = None
    auto_best = float("inf")
    split = native_op("spmm_csr_split") if config.hub_split else None
    core = native_op("spmm_csr")
    remaining = config.probe_cap_ms

    if split is not None:
        for index, candidate in enumerate(
            _spmm_candidates(sampled_crow, x.size(1), config)
        ):
            if index > 0 and remaining <= 0.0:
                break
            tile, warps, threshold = candidate
            timing = _time_cuda(
                lambda t=tile, w=warps, h=threshold: split(
                    sampled_crow, sampled_col, sampled_val, sampled_x, t, w, h
                ),
                warmup=1,
                iterations=config.probe_iterations,
                cap_ms=max(0.0, remaining),
            )
            remaining = max(0.0, remaining - timing["total_ms"])
            candidates.append(
                {
                    "candidate": [tile, warps, threshold],
                    **{key: value for key, value in timing.items() if key != "output"},
                }
            )
            if timing["best_ms"] < auto_best:
                auto_best = timing["best_ms"]
                chosen_candidate = [tile, warps, threshold]
    elif core is not None:
        timing = _time_cuda(
            lambda: core(sampled_crow, sampled_col, sampled_val, sampled_x),
            warmup=1,
            iterations=config.probe_iterations,
            cap_ms=config.probe_cap_ms,
        )
        auto_best = timing["best_ms"]
        candidates.append(
            {
                "candidate": None,
                **{key: value for key, value in timing.items() if key != "output"},
            }
        )

    enough_work = (
        sampled_crow.numel() - 1 >= config.minimum_probe_rows
        and sampled_col.numel() >= config.minimum_probe_nonzeros
    )
    use_autosage = (
        enough_work and auto_best <= config.guardrail * baseline_timing["best_ms"]
    )
    choice = "autosage" if use_autosage else "baseline"
    output = _run_spmm_choice(choice, chosen_candidate, crow, col, val, x)
    probe_ms = (
        sample_ms
        + baseline_timing["wall_ms"]
        + sum(candidate["wall_ms"] for candidate in candidates)
    )
    result = {
        "operation": "spmm",
        "schema_version": AUTOTUNE_SCHEMA_VERSION,
        "choice": choice,
        "candidate": chosen_candidate if use_autosage else None,
        "baseline": baseline_name,
        "guardrail": config.guardrail,
        "configuration": config.tuning_signature(),
        "probe_rows": int(sampled_crow.numel() - 1),
        "probe_nonzeros": int(sampled_col.numel()),
        "sampling_ms": sample_ms,
        "probe_ms": probe_ms,
        "baseline_probe_ms": baseline_timing["best_ms"],
        "autosage_probe_ms": None if auto_best == float("inf") else auto_best,
        "sampled_speedup": (
            baseline_timing["best_ms"] / auto_best
            if auto_best not in (0.0, float("inf"))
            else None
        ),
        "candidates": candidates,
        "from_cache": False,
    }
    if config.cache:
        cache.store(device, graph, result)
    log_probe(
        {"timestamp": time.time(), "device": device, "graph": graph, "result": result}
    )
    if verbose or os.getenv("AUTOSAGE_VERBOSE") == "1":
        print(
            f"[autosage:spmm] rows={result['probe_rows']} nnz={result['probe_nonzeros']} "
            f"baseline={result['baseline_probe_ms']:.4f} ms auto={result['autosage_probe_ms']} "
            f"choice={choice}"
        )
    return output, result


def calibrate_full(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor],
    x: torch.Tensor,
    guardrail: float = 0.95,
) -> tuple[str, dict[str, Any]]:
    crow, col, val, x = _normalize_csr(crow, col, val, x)
    if not x.is_cuda:
        return "baseline", _cpu_info("spmm")
    if x.dtype != torch.float32:
        return "baseline", {
            **_cpu_info("spmm"),
            "baseline": "torch.sparse.mm(csr)",
            "reason": "AutoSAGE CUDA kernels require float32 features",
        }
    load_native()
    config = SchedulerConfig.from_env(guardrail=guardrail, candidate_count=6)
    baseline, baseline_name = _prepare_spmm_baseline(crow, col, val, x)
    baseline_timing = _time_cuda(baseline, warmup=2, iterations=5)
    split = native_op("spmm_csr_split") if config.hub_split else None
    core = native_op("spmm_csr")
    best = float("inf")
    selected: Optional[list[int]] = None
    timings: list[dict[str, Any]] = []

    if split is not None:
        for tile, warps, threshold in _spmm_candidates(crow, x.size(1), config):
            timing = _time_cuda(
                lambda t=tile, w=warps, h=threshold: split(crow, col, val, x, t, w, h),
                warmup=2,
                iterations=5,
            )
            timings.append(
                {"candidate": [tile, warps, threshold], "best_ms": timing["best_ms"]}
            )
            if timing["best_ms"] < best:
                best = timing["best_ms"]
                selected = [tile, warps, threshold]
    elif core is not None:
        timing = _time_cuda(lambda: core(crow, col, val, x), warmup=2, iterations=5)
        best = timing["best_ms"]
        timings.append({"candidate": None, "best_ms": best})

    choice = (
        "autosage" if best <= guardrail * baseline_timing["best_ms"] else "baseline"
    )
    result = {
        "operation": "spmm",
        "schema_version": AUTOTUNE_SCHEMA_VERSION,
        "choice": choice,
        "candidate": selected if choice == "autosage" else None,
        "baseline": baseline_name,
        "guardrail": guardrail,
        "configuration": config.tuning_signature(),
        "calibration": "full_graph",
        "baseline_ms": baseline_timing["best_ms"],
        "autosage_ms": None if best == float("inf") else best,
        "speedup": baseline_timing["best_ms"] / best
        if best not in (0.0, float("inf"))
        else None,
        "candidates": timings,
        "from_cache": False,
    }
    device = device_signature()
    graph = graph_signature(
        crow,
        col,
        x.size(1),
        "spmm",
        weighted=val is not None,
        dtype=x.dtype,
    )
    if config.cache:
        ScheduleCache().store(device, graph, result)
    log_probe(
        {"timestamp": time.time(), "device": device, "graph": graph, "result": result}
    )
    return choice, result


def _run_sddmm_choice(
    choice: str,
    candidate: Optional[list[int]],
    crow: torch.Tensor,
    col: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    if choice == "autosage":
        split = native_op("sddmm_csr_split")
        core = native_op("sddmm_csr")
        if split is not None and candidate is not None:
            return split(crow, col, query, key, *map(int, candidate))
        if core is not None:
            return core(crow, col, query, key)
    baseline, _ = _prepare_sddmm_baseline(crow, col, query, key)
    return baseline()


def sddmm_csr_auto(
    crow: torch.Tensor,
    col: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    guardrail: float = 0.95,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    integer_types = {torch.int32, torch.int64}
    if crow.dtype not in integer_types or col.dtype not in integer_types:
        raise TypeError("crow and col must use integer indices")
    crow = crow.to(dtype=torch.long).contiguous()
    col = col.to(dtype=torch.long).contiguous()
    query = query.contiguous()
    key = key.contiguous()
    if query.shape != key.shape or query.ndim != 2:
        raise ValueError("query and key must have the same [N, F] shape")
    if not query.is_floating_point() or query.dtype != key.dtype:
        raise TypeError("query and key must use the same floating-point dtype")
    if crow.ndim != 1 or col.ndim != 1:
        raise ValueError("crow and col must be one-dimensional")
    if crow.numel() != query.size(0) + 1:
        raise ValueError("crow must contain N + 1 entries")
    if not (crow.device == col.device == query.device == key.device):
        raise ValueError("all SDDMM inputs must be on the same device")
    if not query.is_cuda:
        baseline, name = _prepare_sddmm_baseline(crow, col, query, key)
        return baseline(), {**_cpu_info("sddmm"), "baseline": name}

    if query.dtype != torch.float32:
        baseline, name = _prepare_sddmm_baseline(crow, col, query, key)
        return baseline(), {
            **_cpu_info("sddmm"),
            "baseline": name,
            "reason": "AutoSAGE CUDA kernels require float32 features",
        }

    load_native()
    config = SchedulerConfig.from_env(guardrail=guardrail, candidate_count=6)
    device = device_signature()
    graph = graph_signature(crow, col, query.size(1), "sddmm", dtype=query.dtype)
    cache = ScheduleCache()
    if config.cache:
        record = cache.lookup(device, graph)
        if record is not None and _cache_hit_allowed(record.get("result", {}), config):
            result = record["result"]
            choice = result.get("choice", "baseline")
            candidate = result.get("candidate")
            if _cached_native_available("sddmm_csr", choice, candidate):
                output = _run_sddmm_choice(
                    choice,
                    candidate,
                    crow,
                    col,
                    query,
                    key,
                )
                return output, {**result, "from_cache": True}

    if config.replay_only:
        output = _run_sddmm_choice("baseline", None, crow, col, query, key)
        return output, {
            "operation": "sddmm",
            "schema_version": AUTOTUNE_SCHEMA_VERSION,
            "choice": "baseline",
            "baseline": "gather-dot",
            "from_cache": False,
            "replay_only": True,
            "reason": "cache miss",
        }

    torch.cuda.synchronize()
    sample_start = time.perf_counter()
    sampled_crow, sampled_col, _, selected = _induced_sample_csr(
        crow,
        col,
        None,
        fraction=config.probe_fraction,
        minimum_rows=config.probe_min_rows,
        seed=seed,
    )
    sampled_query = query.index_select(0, selected).contiguous()
    sampled_key = key.index_select(0, selected).contiguous()
    torch.cuda.synchronize()
    sample_ms = (time.perf_counter() - sample_start) * 1000.0

    baseline, baseline_name = _prepare_sddmm_baseline(
        sampled_crow, sampled_col, sampled_query, sampled_key
    )
    baseline_timing = _time_cuda(baseline, warmup=1, iterations=config.probe_iterations)
    split = native_op("sddmm_csr_split") if config.hub_split else None
    core = native_op("sddmm_csr")
    parameter_pairs = list(
        dict.fromkeys(
            (warps, threshold)
            for _, warps, threshold in _spmm_candidates(
                sampled_crow, query.size(1), config
            )
        )
    )
    auto_best = float("inf")
    selected_candidate: Optional[list[int]] = None
    candidate_timings: list[dict[str, Any]] = []
    remaining = config.probe_cap_ms

    if split is not None:
        for index, (warps, threshold) in enumerate(parameter_pairs):
            if index > 0 and remaining <= 0.0:
                break
            timing = _time_cuda(
                lambda w=warps, h=threshold: split(
                    sampled_crow, sampled_col, sampled_query, sampled_key, w, h
                ),
                warmup=1,
                iterations=config.probe_iterations,
                cap_ms=max(0.0, remaining),
            )
            remaining = max(0.0, remaining - timing["total_ms"])
            candidate_timings.append(
                {
                    "candidate": [warps, threshold],
                    **{
                        key_name: value
                        for key_name, value in timing.items()
                        if key_name != "output"
                    },
                }
            )
            if timing["best_ms"] < auto_best:
                auto_best = timing["best_ms"]
                selected_candidate = [warps, threshold]
    elif core is not None:
        timing = _time_cuda(
            lambda: core(sampled_crow, sampled_col, sampled_query, sampled_key),
            warmup=1,
            iterations=config.probe_iterations,
            cap_ms=config.probe_cap_ms,
        )
        auto_best = timing["best_ms"]
        candidate_timings.append(
            {
                "candidate": None,
                **{
                    key_name: value
                    for key_name, value in timing.items()
                    if key_name != "output"
                },
            }
        )

    enough_work = (
        sampled_crow.numel() - 1 >= config.minimum_probe_rows
        and sampled_col.numel() >= config.minimum_probe_nonzeros
    )
    choice = (
        "autosage"
        if enough_work and auto_best <= config.guardrail * baseline_timing["best_ms"]
        else "baseline"
    )
    output = _run_sddmm_choice(choice, selected_candidate, crow, col, query, key)
    result = {
        "operation": "sddmm",
        "schema_version": AUTOTUNE_SCHEMA_VERSION,
        "choice": choice,
        "candidate": selected_candidate if choice == "autosage" else None,
        "baseline": baseline_name,
        "guardrail": config.guardrail,
        "configuration": config.tuning_signature(),
        "probe_rows": int(sampled_crow.numel() - 1),
        "probe_nonzeros": int(sampled_col.numel()),
        "sampling_ms": sample_ms,
        "probe_ms": sample_ms
        + baseline_timing["wall_ms"]
        + sum(candidate["wall_ms"] for candidate in candidate_timings),
        "baseline_probe_ms": baseline_timing["best_ms"],
        "autosage_probe_ms": None if auto_best == float("inf") else auto_best,
        "sampled_speedup": (
            baseline_timing["best_ms"] / auto_best
            if auto_best not in (0.0, float("inf"))
            else None
        ),
        "candidates": candidate_timings,
        "from_cache": False,
    }
    if config.cache:
        cache.store(device, graph, result)
    log_probe(
        {"timestamp": time.time(), "device": device, "graph": graph, "result": result}
    )
    if verbose or os.getenv("AUTOSAGE_VERBOSE") == "1":
        print(
            f"[autosage:sddmm] rows={result['probe_rows']} nnz={result['probe_nonzeros']} "
            f"baseline={result['baseline_probe_ms']:.4f} ms auto={result['autosage_probe_ms']} "
            f"choice={choice}"
        )
    return output, result


def _csr_row_softmax(crow: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    row_count = crow.numel() - 1
    counts = (crow[1:] - crow[:-1]).to(torch.long)
    rows = torch.repeat_interleave(
        torch.arange(row_count, device=crow.device, dtype=torch.long), counts
    )
    if scores.numel() == 0:
        return scores.clone()
    maxima = torch.full(
        (row_count,), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    maxima.scatter_reduce_(0, rows, scores, reduce="amax", include_self=True)
    exponentials = (scores - maxima.index_select(0, rows)).exp()
    denominators = torch.zeros_like(maxima)
    denominators.scatter_add_(0, rows, exponentials)
    return exponentials / denominators.index_select(0, rows)


def csr_attention_forward(
    crow: torch.Tensor,
    col: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    guardrail: float = 0.95,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    scores, sddmm_info = sddmm_csr_auto(
        crow, col, query, key, guardrail=guardrail, seed=seed, verbose=verbose
    )
    probabilities = _csr_row_softmax(crow, scores.to(query.dtype))
    output, spmm_info = spmm_csr_auto(
        crow, col, probabilities, value, guardrail=guardrail, seed=seed, verbose=verbose
    )
    return output, {"sddmm": sddmm_info, "spmm": spmm_info}


__all__ = [
    "AUTOTUNE_SCHEMA_VERSION",
    "calibrate_full",
    "csr_attention_forward",
    "sddmm_csr_auto",
    "spmm_csr_auto",
]
