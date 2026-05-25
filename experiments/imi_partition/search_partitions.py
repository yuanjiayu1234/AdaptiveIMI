from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from experiments.imi_partition.partition import choose_by_constraint, optimize_partition


def _parse_m_values(value: str) -> List[int]:
    return [int(item) for item in value.replace(",", " ").split() if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search balanced weighted-graph RoPE-pair partitions for multiple M values.")
    parser.add_argument("--dependency", type=Path, required=True, help="Output .pt from compute_dependency.py.")
    parser.add_argument("--m-values", type=str, default="4,8,16")
    parser.add_argument("--min-pairs", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--allow-unbalanced", action="store_true", help="Disable strict near-equal group-size constraints.")
    parser.add_argument(
        "--balance-slack",
        type=int,
        default=0,
        help="Allowed deviation from near-equal group sizes. Default 0 enforces strict balanced groups.",
    )
    parser.add_argument("--gamma", type=float, default=0.01, help="Frequency-locality regularization weight.")
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--swap-passes", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=0)
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
        help="Metric used for thresholding before applying --select-preference.",
    )
    parser.add_argument(
        "--select-preference",
        type=str,
        default="largest",
        choices=["smallest", "largest", "lowest_leak"],
        help="How to choose among candidates that satisfy the selected leakage threshold.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dep_file = torch.load(args.dependency, map_location="cpu")
    scopes = dep_file.get("scopes", [])
    if not scopes:
        raise ValueError("dependency file has no scopes")

    m_values = _parse_m_values(args.m_values)
    all_outputs: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for scope in scopes:
        name = str(scope["name"])
        dependency = scope["dependency"]
        results = []
        for m in m_values:
            try:
                result = optimize_partition(
                    dependency,
                    m,
                    min_size=int(args.min_pairs),
                    max_size=int(args.max_pairs),
                    balanced=not bool(args.allow_unbalanced),
                    balance_slack=int(args.balance_slack),
                    gamma=float(args.gamma),
                    seeds=int(args.seeds),
                    max_iter=int(args.max_iter),
                    swap_passes=int(args.swap_passes),
                    random_seed=int(args.random_seed),
                )
            except ValueError as exc:
                print(f"[partition] skip scope={name} M={m}: {exc}", flush=True)
                continue
            results.append(result)
            row = {
                "scope": name,
                "M": result.m,
                "seed": result.seed,
                **result.metrics.to_dict(),
            }
            csv_rows.append(row)
            print(
                f"[partition] {name} M={result.m} "
                f"keep_ratio={result.metrics.within_ratio:.4f} "
                f"cut_ratio={result.metrics.leak_ratio:.4f} "
                f"top5={result.metrics.top_edge_keep.get('0.05', 0.0):.4f} "
                f"top10={result.metrics.top_edge_keep.get('0.10', 0.0):.4f} "
                f"sizes={result.metrics.group_sizes}",
                flush=True,
            )

        selected = choose_by_constraint(
            results,
            min_keep_ratio=float(args.min_keep_ratio),
            min_top_edge_keep=float(args.min_top_edge_keep),
            top_edge_fraction=f"{float(args.top_edge_fraction):.2f}",
            max_normalized_leak=None if args.max_normalized_leak is None else float(args.max_normalized_leak),
            max_leak_ratio=None if args.max_leak_ratio is None else float(args.max_leak_ratio),
            constraint_metric=str(args.constraint_metric),
            prefer=str(args.select_preference),
        )
        all_outputs.append({
            "name": name,
            "meta": scope.get("meta", {}),
            "selected": selected.to_dict() if selected is not None else None,
            "candidates": [result.to_dict() for result in results],
        })
        if selected is None:
            if args.constraint_metric == "keep_ratio":
                print(f"[partition] {name}: no M satisfies min_keep_ratio={args.min_keep_ratio}", flush=True)
            elif args.constraint_metric == "leak_ratio":
                print(f"[partition] {name}: no M satisfies max_leak_ratio={args.max_leak_ratio}", flush=True)
            else:
                print(f"[partition] {name}: no M satisfies max_normalized_leak={args.max_normalized_leak}", flush=True)
        else:
            print(
                f"[partition] {name}: selected M={selected.m} "
                f"keep_ratio={selected.metrics.within_ratio:.4f} "
                f"top10={selected.metrics.top_edge_keep.get('0.10', 0.0):.4f}",
                flush=True,
            )

    output = {
        "metadata": {
            "dependency": str(args.dependency),
            "m_values": m_values,
            "min_pairs": int(args.min_pairs),
            "max_pairs": int(args.max_pairs),
            "balanced": not bool(args.allow_unbalanced),
            "balance_slack": int(args.balance_slack),
            "gamma": float(args.gamma),
            "min_keep_ratio": float(args.min_keep_ratio),
            "min_top_edge_keep": float(args.min_top_edge_keep),
            "top_edge_fraction": float(args.top_edge_fraction),
            "max_leak_ratio": None if args.max_leak_ratio is None else float(args.max_leak_ratio),
            "max_normalized_leak": None if args.max_normalized_leak is None else float(args.max_normalized_leak),
            "constraint_metric": str(args.constraint_metric),
            "select_preference": str(args.select_preference),
        },
        "scopes": all_outputs,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[partition] wrote {args.out_json}", flush=True)

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "scope",
            "M",
            "seed",
            "objective",
            "within",
            "within_score",
            "leakage",
            "total",
            "within_ratio",
            "keep_ratio",
            "leak_ratio",
            "cut_ratio",
            "normalized_leak",
            "relative_random_leak",
            "top_edge_keep",
            "top_edge_weight_keep",
            "locality_penalty",
            "cost_proxy",
            "group_sizes",
        ]
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                row = dict(row)
                row["group_sizes"] = json.dumps(row["group_sizes"])
                row["top_edge_keep"] = json.dumps(row["top_edge_keep"])
                row["top_edge_weight_keep"] = json.dumps(row["top_edge_weight_keep"])
                writer.writerow(row)
        print(f"[partition] wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
