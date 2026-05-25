from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-scope partition search output into layer-group M plus per-head partitions."
    )
    parser.add_argument("--partitions", type=Path, required=True, help="JSON output from search_partitions.py")
    parser.add_argument(
        "--min-keep-ratio",
        type=float,
        default=0.0,
        help="Global score-retention threshold: within_score / total_score. Comparable across M.",
    )
    parser.add_argument(
        "--min-top-edge-keep",
        type=float,
        default=0.0,
        help="Minimum fraction of strongest pair-pair edges that must remain within subspaces.",
    )
    parser.add_argument(
        "--top-edge-fraction",
        type=float,
        default=0.10,
        help="Top edge fraction used by --min-top-edge-keep.",
    )
    parser.add_argument(
        "--max-leak-ratio",
        type=float,
        default=None,
        help="Legacy global leakage threshold: leakage / total_dependency.",
    )
    parser.add_argument(
        "--max-normalized-leak",
        type=float,
        default=None,
        help="Legacy relative-to-random leakage threshold. Not comparable across M.",
    )
    parser.add_argument(
        "--constraint-metric",
        type=str,
        default="keep_ratio",
        choices=["keep_ratio", "leak_ratio", "normalized_leak"],
        help="Metric used for thresholding before selecting group M.",
    )
    parser.add_argument(
        "--min-head-fraction",
        type=float,
        default=0.75,
        help="Minimum fraction of heads that must satisfy the leak threshold for a group M.",
    )
    parser.add_argument(
        "--select-preference",
        type=str,
        default="largest",
        choices=["smallest", "largest", "lowest_leak"],
        help="How to choose group M when multiple M values satisfy the threshold.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=0,
        help="Number of KV heads to emit when partition scopes are layer/layer_group rather than per-head.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _candidate_by_m(scope: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for candidate in scope.get("candidates", []):
        out[int(candidate["M"])] = candidate
    return out


def _group_key(scope: Dict[str, Any]) -> Tuple[int, int, int]:
    meta = scope.get("meta", {}) or {}
    layer_group = int(meta.get("layer_group", -1))
    layer_start = int(meta.get("layer_start", 0))
    layer_end = int(meta.get("layer_end", layer_start + 1))
    return layer_group, layer_start, layer_end


def _choose_m(
    values: List[int],
    *,
    per_head: Dict[int, Dict[int, Dict[str, Any]]],
    prefer: str,
    metric_name: str = "keep_ratio",
) -> int:
    values = sorted(set(int(value) for value in values))
    prefer = prefer.lower()
    if prefer == "smallest":
        return values[0]
    if prefer == "largest":
        return values[-1]
    if prefer == "lowest_leak":
        def score(m: int) -> Tuple[float, int]:
            leaks = [
                float(candidates[m].get("metrics", {}).get(metric_name, 1e9))
                for candidates in per_head.values()
            ]
            return sum(leaks) / max(len(leaks), 1), m

        return min(values, key=score)
    raise ValueError("prefer must be one of: smallest, largest, lowest_leak")


def _select_group_m(
    per_head: Dict[int, Dict[int, Dict[str, Any]]],
    *,
    min_keep_ratio: float | None,
    min_top_edge_keep: float | None,
    top_edge_fraction: str,
    max_normalized_leak: float | None,
    max_leak_ratio: float | None,
    constraint_metric: str,
    min_head_fraction: float,
    prefer: str,
) -> int:
    common_m = None
    for candidates in per_head.values():
        m_set = set(candidates)
        common_m = m_set if common_m is None else common_m & m_set
    if not common_m:
        raise ValueError("no common M candidates across heads")

    head_count = max(len(per_head), 1)
    threshold_count = max(1, int(round(float(head_count) * float(min_head_fraction))))

    feasible_ms: List[int] = []
    all_head_feasible_ms: List[int] = []
    constraint_metric = constraint_metric.lower()
    if constraint_metric in ("keep_ratio", "within_ratio"):
        threshold = min_keep_ratio
        metric_name = "keep_ratio"
        higher_is_better = True
    elif constraint_metric == "leak_ratio":
        threshold = max_leak_ratio if max_leak_ratio is not None else max_normalized_leak
        metric_name = "leak_ratio"
        higher_is_better = False
    elif constraint_metric == "normalized_leak":
        threshold = max_normalized_leak
        metric_name = "normalized_leak"
        higher_is_better = False
    else:
        raise ValueError("constraint_metric must be one of: keep_ratio, leak_ratio, normalized_leak")

    for m in sorted(common_m):
        feasible_heads = 0
        for candidates in per_head.values():
            metrics = candidates[m].get("metrics", {})
            metric_value = float(metrics.get(metric_name, metrics.get("within_ratio", 0.0 if higher_is_better else 1e9)))
            passes_main = (
                True if threshold is None
                else metric_value >= float(threshold) if higher_is_better
                else metric_value <= float(threshold)
            )
            top_keep = metrics.get("top_edge_keep", {}) or {}
            top_key = f"{float(top_edge_fraction):.2f}"
            passes_top = (
                min_top_edge_keep is None or float(min_top_edge_keep) <= 0.0
                or float(top_keep.get(top_key, 0.0)) >= float(min_top_edge_keep)
            )
            if passes_main and passes_top:
                feasible_heads += 1
        if feasible_heads == head_count:
            all_head_feasible_ms.append(m)
        if feasible_heads >= threshold_count:
            feasible_ms.append(m)

    if feasible_ms:
        return _choose_m(feasible_ms, per_head=per_head, prefer=prefer, metric_name=metric_name)
    if all_head_feasible_ms:
        return _choose_m(all_head_feasible_ms, per_head=per_head, prefer=prefer, metric_name=metric_name)
    return min(common_m)


def _emit_head_count(group_scopes: List[Dict[str, Any]], explicit_num_heads: int) -> int:
    if explicit_num_heads > 0:
        return int(explicit_num_heads)
    for scope in group_scopes:
        meta = scope.get("meta", {}) or {}
        if "head_count" in meta:
            return int(meta["head_count"])
    heads = [
        int((scope.get("meta", {}) or {}).get("head", -1))
        for scope in group_scopes
        if "head" in (scope.get("meta", {}) or {})
    ]
    if heads:
        return max(heads) + 1
    raise ValueError("cannot infer KV head count; pass --num-heads")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.partitions.read_text(encoding="utf-8"))
    scopes = payload.get("scopes", [])
    if not scopes:
        raise ValueError("partition JSON has no scopes")

    grouped: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for scope in scopes:
        grouped[_group_key(scope)].append(scope)

    layer_groups = []
    for (layer_group, layer_start, layer_end), group_scopes in sorted(grouped.items()):
        has_per_head = any("head" in (scope.get("meta", {}) or {}) for scope in group_scopes)
        per_head: Dict[int, Dict[int, Dict[str, Any]]] = {}
        if has_per_head:
            for scope in group_scopes:
                meta = scope.get("meta", {}) or {}
                head = int(meta.get("head", len(per_head)))
                per_head[head] = _candidate_by_m(scope)
        else:
            if len(group_scopes) != 1:
                raise ValueError(
                    "non-head scopes should have one partition scope per output layer group; "
                    "use --scope layer_group for deployable metadata or analyze --scope layer separately"
                )
            head_count = _emit_head_count(group_scopes, int(args.num_heads))
            shared_candidates = _candidate_by_m(group_scopes[0])
            per_head = {head: shared_candidates for head in range(head_count)}

        selected_m = _select_group_m(
            per_head,
            min_keep_ratio=float(args.min_keep_ratio),
            min_top_edge_keep=float(args.min_top_edge_keep),
            top_edge_fraction=f"{float(args.top_edge_fraction):.2f}",
            max_normalized_leak=None if args.max_normalized_leak is None else float(args.max_normalized_leak),
            max_leak_ratio=None if args.max_leak_ratio is None else float(args.max_leak_ratio),
            constraint_metric=str(args.constraint_metric),
            min_head_fraction=float(args.min_head_fraction),
            prefer=str(args.select_preference),
        )

        heads: Dict[str, Dict[str, Any]] = {}
        for head, candidates in sorted(per_head.items()):
            candidate = candidates[selected_m]
            heads[str(head)] = {
                "groups": candidate["groups"],
                "pair_to_group": candidate["pair_to_group"],
                "metrics": candidate["metrics"],
                "seed": candidate.get("seed", -1),
            }

        layer_groups.append({
            "layer_group": layer_group,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "M": selected_m,
            "subspace_parts": selected_m,
            "heads": heads,
        })

    output = {
        "metadata": {
            "source": str(args.partitions),
            "min_keep_ratio": float(args.min_keep_ratio),
            "min_top_edge_keep": float(args.min_top_edge_keep),
            "top_edge_fraction": float(args.top_edge_fraction),
            "max_leak_ratio": None if args.max_leak_ratio is None else float(args.max_leak_ratio),
            "max_normalized_leak": None if args.max_normalized_leak is None else float(args.max_normalized_leak),
            "constraint_metric": str(args.constraint_metric),
            "min_head_fraction": float(args.min_head_fraction),
            "select_preference": str(args.select_preference),
            "format": "adaptive_imi_learned_partition_v1",
        },
        "layer_groups": layer_groups,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[aggregate] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
