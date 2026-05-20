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
    parser = argparse.ArgumentParser(description="Search non-uniform RoPE-pair partitions for multiple M values.")
    parser.add_argument("--dependency", type=Path, required=True, help="Output .pt from compute_dependency.py.")
    parser.add_argument("--m-values", type=str, default="2,3,4,5,6,8,10,12,16")
    parser.add_argument("--min-pairs", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.01, help="Frequency-locality regularization weight.")
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--swap-passes", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-normalized-leak", type=float, default=0.85)
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
                f"norm_leak={result.metrics.normalized_leak:.4f} "
                f"sizes={result.metrics.group_sizes}",
                flush=True,
            )

        selected = choose_by_constraint(results, max_normalized_leak=float(args.max_normalized_leak))
        all_outputs.append({
            "name": name,
            "meta": scope.get("meta", {}),
            "selected": selected.to_dict() if selected is not None else None,
            "candidates": [result.to_dict() for result in results],
        })
        if selected is None:
            print(f"[partition] {name}: no M satisfies max_normalized_leak={args.max_normalized_leak}", flush=True)
        else:
            print(f"[partition] {name}: selected M={selected.m}", flush=True)

    output = {
        "metadata": {
            "dependency": str(args.dependency),
            "m_values": m_values,
            "min_pairs": int(args.min_pairs),
            "max_pairs": int(args.max_pairs),
            "gamma": float(args.gamma),
            "max_normalized_leak": float(args.max_normalized_leak),
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
            "leakage",
            "total",
            "leak_ratio",
            "normalized_leak",
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
                writer.writerow(row)
        print(f"[partition] wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
