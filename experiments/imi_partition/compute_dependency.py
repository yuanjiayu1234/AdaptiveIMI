from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from experiments.imi_partition.dependency import compute_pair_dependency, iter_scope_keys, tensor_to_lhtd


def _load_keys(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("keys", "key_states", "rope_keys"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value
        shards = obj.get("shards")
        if isinstance(shards, list) and all(isinstance(item, torch.Tensor) for item in shards):
            return torch.cat(shards, dim=0)
    raise ValueError(f"Unsupported key file format: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RoPE-pair dependency matrices from saved key tensors.")
    parser.add_argument("--keys", type=Path, required=True, help="Path to a .pt key tensor or dict containing 'keys'.")
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "slhtd", "lhtd", "htd", "td"])
    parser.add_argument(
        "--scope",
        type=str,
        default="layer_group",
        choices=["global", "layer_group", "layer", "global_head", "layer_group_head", "layer_head"],
    )
    parser.add_argument("--layer-groups", type=int, default=3, help="Number of layer groups for layer_group scopes.")
    parser.add_argument("--max-tokens-per-scope", type=int, default=262144, help="Subsample tokens per scope for stable memory use.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mi-weight", type=float, default=0.25)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = _load_keys(args.keys)
    keys_lhtd = tensor_to_lhtd(keys, args.layout)
    gen = torch.Generator().manual_seed(int(args.seed))

    scopes: List[Dict[str, Any]] = []
    for scope_name, scope_keys, meta in iter_scope_keys(keys_lhtd, scope=args.scope, group_count=args.layer_groups):
        scope_keys = scope_keys.contiguous()
        if args.max_tokens_per_scope > 0 and scope_keys.size(0) > args.max_tokens_per_scope:
            perm = torch.randperm(scope_keys.size(0), generator=gen)[: args.max_tokens_per_scope]
            scope_keys = scope_keys.index_select(0, perm).contiguous()

        result = compute_pair_dependency(
            scope_keys,
            eps=float(args.eps),
            mi_weight=float(args.mi_weight),
        )
        payload = result.to_dict()
        payload.update({
            "name": scope_name,
            "meta": meta,
        })
        scopes.append(payload)
        print(
            f"[dependency] {scope_name}: tokens={result.num_tokens} "
            f"pairs={result.num_pairs} mean_dep={result.dependency[result.dependency > 0].mean().item():.4f}",
            flush=True,
        )

    output = {
        "metadata": {
            "source_keys": str(args.keys),
            "layout": args.layout,
            "scope": args.scope,
            "layer_groups": int(args.layer_groups),
            "max_tokens_per_scope": int(args.max_tokens_per_scope),
            "mi_weight": float(args.mi_weight),
            "eps": float(args.eps),
        },
        "scopes": scopes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    print(f"[dependency] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
