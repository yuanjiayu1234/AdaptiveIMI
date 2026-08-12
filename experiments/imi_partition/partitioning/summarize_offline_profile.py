from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from experiments.imi_partition.partitioning.profiling import write_json


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * float(percentile)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize_cuda_peak(profiles: Iterable[Dict[str, object]]) -> Dict[str, int]:
    peak_allocated = 0
    peak_reserved = 0
    for profile in profiles:
        profiling = profile.get("profiling") or {}
        for device_peak in (profiling.get("cuda_peak_bytes") or {}).values():
            if not isinstance(device_peak, dict):
                continue
            peak_allocated = max(peak_allocated, int(device_peak.get("peak_allocated_bytes", 0)))
            peak_reserved = max(peak_reserved, int(device_peak.get("peak_reserved_bytes", 0)))
    return {
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def _calibration_summary(path: Path) -> Dict[str, object]:
    source_counts: Counter[str] = Counter()
    rows = 0
    input_chars = 0
    max_input_chars = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("input") or "")
            rows += 1
            input_chars += len(text)
            max_input_chars = max(max_input_chars, len(text))
            source_counts[str(row.get("source_dataset") or "unknown")] += 1
    return {
        "jsonl": str(path),
        "documents": rows,
        "input_characters": input_chars,
        "max_input_characters": max_input_chars,
        "source_document_counts": dict(sorted(source_counts.items())),
    }


def _stage_summary(profiles: Sequence[Dict[str, object]]) -> Dict[str, object]:
    elapsed_values = [float((profile.get("profiling") or {}).get("elapsed_s", 0.0)) for profile in profiles]
    cpu_values = [int((profile.get("profiling") or {}).get("peak_cpu_rss_bytes", 0)) for profile in profiles]
    return {
        "invocations": len(profiles),
        "elapsed_s_total": sum(elapsed_values),
        "elapsed_s_mean": sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0,
        "elapsed_s_p50": _percentile(elapsed_values, 0.5),
        "elapsed_s_p95": _percentile(elapsed_values, 0.95),
        "peak_cpu_rss_bytes": max(cpu_values, default=0),
        "cuda": _summarize_cuda_peak(profiles),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize AdaptiveIMI offline partition-training profiles.")
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wall-clock-elapsed-s", type=float, default=None)
    parser.add_argument("--layer-batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection_paths = sorted(args.profile_dir.glob("*_collection_profile.json"))
    search_paths = sorted(args.profile_dir.glob("*_search_profile.json"))
    if not collection_paths or not search_paths:
        raise FileNotFoundError(
            f"expected collection and search profiles under {args.profile_dir}; "
            f"found collection={len(collection_paths)} search={len(search_paths)}"
        )

    collection_profiles = [_load_json(path) for path in collection_paths]
    search_profiles = [_load_json(path) for path in search_paths]
    partition = _load_json(args.partition)
    partition_metadata = dict(partition.get("metadata") or {})
    log_rows = list(partition_metadata.get("log_rows") or [])
    teacher_times = [float(row.get("teacher_time_s", 0.0)) for row in log_rows]
    search_times = [float(row.get("search_time_s", 0.0)) for row in log_rows]
    scope_times = [float(row.get("scope_total_time_s", 0.0)) for row in log_rows]
    seen_tokens = []
    retained_vectors = []
    for profile in collection_profiles:
        metadata = profile.get("output_metadata") or {}
        seen_tokens.extend(int(value) for value in (metadata.get("seen_tokens") or {}).values())
        shape = metadata.get("shape") or []
        if len(shape) == 4:
            retained_vectors.append(int(shape[2]))

    payload = {
        "format": "adaptive_imi_offline_profile_v1",
        "partition": str(args.partition),
        "partition_learning_config": {
            key: partition_metadata.get(key)
            for key in (
                "objective_mode",
                "scope",
                "M",
                "target_cluster_size",
                "oversize_soft_cap",
                "oversize_hard_cap",
                "sample_tokens",
                "kmeans_iters",
                "cdist_chunk_size",
                "starts",
                "steps",
                "proposal_batch_size",
                "group_kmeans_batch_size",
                "random_seed",
            )
        },
        "calibration": _calibration_summary(args.calibration_jsonl),
        "collection": {
            **_stage_summary(collection_profiles),
            "seen_tokens_per_head_layer": {
                "min": min(seen_tokens, default=0),
                "max": max(seen_tokens, default=0),
            },
            "retained_k_vectors_per_head": {
                "min": min(retained_vectors, default=0),
                "max": max(retained_vectors, default=0),
            },
        },
        "teacher": {
            "scope_count": len(teacher_times),
            "elapsed_s_total": sum(teacher_times),
            "elapsed_s_mean": sum(teacher_times) / len(teacher_times) if teacher_times else 0.0,
            "elapsed_s_p50": _percentile(teacher_times, 0.5),
            "elapsed_s_p95": _percentile(teacher_times, 0.95),
        },
        "search": {
            "scope_count": len(search_times),
            "elapsed_s_total": sum(search_times),
            "elapsed_s_mean": sum(search_times) / len(search_times) if search_times else 0.0,
            "elapsed_s_p50": _percentile(search_times, 0.5),
            "elapsed_s_p95": _percentile(search_times, 0.95),
            "scope_elapsed_s_total": sum(scope_times),
        },
        "partition_search_process": _stage_summary(search_profiles),
        "offline_partition_learning_elapsed_s": (
            sum(float((profile.get("profiling") or {}).get("elapsed_s", 0.0)) for profile in collection_profiles)
            + sum(float((profile.get("profiling") or {}).get("elapsed_s", 0.0)) for profile in search_profiles)
        ),
    }
    if args.wall_clock_elapsed_s is not None:
        payload["offline_partition_learning_wall_clock_s"] = float(args.wall_clock_elapsed_s)
    if args.layer_batch_size is not None:
        payload["layer_batch_size"] = int(args.layer_batch_size)
    write_json(args.out, payload)
    print(f"[offline-profile] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
