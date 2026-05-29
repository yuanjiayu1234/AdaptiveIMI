from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from cache_hub.cluster_plan import choose_learned_k_values


def _progress(iterable, *, total: Optional[int], desc: str, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


@dataclass
class Teacher:
    keys: torch.Tensor
    teacher_centroids_for_token: torch.Tensor
    teacher_mse_scale: float


@dataclass
class SearchResult:
    groups: List[List[int]]
    metrics: Dict[str, float]
    origin: str
    k_values: List[int]


def layer_groups(num_layers: int, group_count: int = 3) -> List[Tuple[int, int]]:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    groups: List[Tuple[int, int]] = []
    for group_idx in range(group_count):
        start = (num_layers * group_idx) // group_count
        end = (num_layers * (group_idx + 1)) // group_count
        if start < end:
            groups.append((start, end))
    return groups


def tensor_to_lhtd(keys: torch.Tensor, layout: str) -> torch.Tensor:
    layout = layout.lower()
    if layout == "lhtd":
        if keys.dim() != 4:
            raise ValueError("layout lhtd expects tensor [layers, heads, tokens, dim]")
        return keys
    if layout == "slhtd":
        if keys.dim() != 5:
            raise ValueError("layout slhtd expects tensor [samples, layers, heads, tokens, dim]")
        s, l, h, t, d = keys.shape
        return keys.permute(1, 2, 0, 3, 4).reshape(l, h, s * t, d)
    if layout == "htd":
        if keys.dim() != 3:
            raise ValueError("layout htd expects tensor [heads, tokens, dim]")
        return keys.unsqueeze(0)
    if layout == "td":
        if keys.dim() != 2:
            raise ValueError("layout td expects tensor [tokens, dim]")
        return keys.view(1, 1, keys.size(0), keys.size(1))
    if layout == "auto":
        if keys.dim() == 5:
            return tensor_to_lhtd(keys, "slhtd")
        if keys.dim() == 4:
            return tensor_to_lhtd(keys, "lhtd")
        if keys.dim() == 3:
            return tensor_to_lhtd(keys, "htd")
        if keys.dim() == 2:
            return tensor_to_lhtd(keys, "td")
    raise ValueError(f"unsupported key tensor layout: {layout}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search RoPE-pair partitions by minimizing the structural centroid "
            "error between full-dim K-means teacher clusters and IMI compact cells."
        )
    )
    parser.add_argument("--keys", type=Path, required=True, help="Torch file from collect_llama3_rope_keys.py.")
    parser.add_argument("--layout", type=str, default="auto", help="Key tensor layout if not stored in metadata.")
    parser.add_argument("--scope", choices=["layer_head", "layer", "layer_group", "global"], default="layer_head")
    parser.add_argument("--layer-groups", type=int, default=3, help="Used when --scope layer_group.")
    parser.add_argument("--layers", type=str, default="all", help="Layer ids to emit, e.g. all or 0-31 or 0,4,8.")
    parser.add_argument("--m", type=int, default=8, help="Number of learned subspaces. hybrid_8way_v2 expects 8.")
    parser.add_argument("--target-cluster-size", type=int, default=16)
    parser.add_argument("--full-clusters", type=int, default=0, help="Override full-dim teacher cluster count.")
    parser.add_argument("--max-full-clusters", type=int, default=8192)
    parser.add_argument("--sample-tokens", type=int, default=131072)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--starts", type=int, default=4, help="Random balanced starts per optimized scope.")
    parser.add_argument("--steps", type=int, default=16, help="Accepted/rejected swap proposals per start.")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-log", type=Path, default=None)
    parser.add_argument("--compact-json", action="store_true", help="Write JSON without indentation.")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable tqdm progress bars.")
    parser.set_defaults(progress=True)
    return parser.parse_args()


def _parse_layer_ids(value: str, available: Sequence[int]) -> List[int]:
    available_set = set(int(item) for item in available)
    text = str(value).strip().lower()
    if text in ("all", "*"):
        return list(map(int, available))
    out = set()
    for part in text.replace(",", " ").split():
        if "-" in part:
            left, right = part.split("-", 1)
            out.update(range(int(left), int(right) + 1))
        else:
            out.add(int(part))
    selected = [layer for layer in available if int(layer) in out and int(layer) in available_set]
    if not selected:
        raise ValueError(f"no requested layers are present in collected keys: {value!r}")
    return list(map(int, selected))


def _load_keys(path: Path, layout: str) -> Tuple[torch.Tensor, Dict[str, object], List[int]]:
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        keys = payload
        metadata: Dict[str, object] = {}
    elif isinstance(payload, dict) and torch.is_tensor(payload.get("keys")):
        keys = payload["keys"]
        metadata = dict(payload.get("metadata") or {})
    else:
        raise ValueError("--keys must be a tensor or a dict containing tensor field 'keys'")

    effective_layout = str(metadata.get("layout") or layout)
    keys_lhtd = tensor_to_lhtd(keys, effective_layout).contiguous()
    layer_ids = metadata.get("layers")
    if isinstance(layer_ids, list) and len(layer_ids) == int(keys_lhtd.size(0)):
        actual_layers = [int(item) for item in layer_ids]
    else:
        actual_layers = list(range(int(keys_lhtd.size(0))))
    return keys_lhtd, metadata, actual_layers


def _balanced_pair_groups(num_pairs: int, m: int, order: Sequence[int]) -> List[List[int]]:
    if m <= 0 or num_pairs < m:
        raise ValueError(f"cannot split {num_pairs} pairs into {m} groups")
    base = num_pairs // m
    extra = num_pairs % m
    sizes = [base + (1 if idx < extra else 0) for idx in range(m)]
    groups: List[List[int]] = []
    cursor = 0
    for size in sizes:
        groups.append(sorted(int(item) for item in order[cursor : cursor + size]))
        cursor += size
    return groups


def _sequential_groups(num_pairs: int, m: int) -> List[List[int]]:
    return _balanced_pair_groups(num_pairs, m, list(range(num_pairs)))


def _random_groups(num_pairs: int, m: int, rng: random.Random) -> List[List[int]]:
    order = list(range(num_pairs))
    rng.shuffle(order)
    return _balanced_pair_groups(num_pairs, m, order)


def _copy_groups(groups: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(map(int, group)) for group in groups]


def _pair_groups_to_dim_groups(groups: Sequence[Sequence[int]], head_dim: int) -> List[List[int]]:
    half = int(head_dim) // 2
    dim_groups: List[List[int]] = []
    for group in groups:
        dims: List[int] = []
        for pair in group:
            pair = int(pair)
            dims.append(pair)
            dims.append(pair + half)
        dim_groups.append(dims)
    return dim_groups


def _sample_rows(keys: torch.Tensor, limit: int, seed: int) -> torch.Tensor:
    keys = keys.reshape(-1, keys.size(-1))
    if int(limit) <= 0 or int(keys.size(0)) <= int(limit):
        return keys
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    idx = torch.randperm(int(keys.size(0)), generator=gen)[: int(limit)]
    return keys.index_select(0, idx)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError("kmeans input must be [tokens, dim]")
    n = int(x.size(0))
    if n <= 0:
        raise ValueError("kmeans input is empty")
    k = max(1, min(int(k), n))
    x = x.float().contiguous()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    init_idx = torch.randperm(n, generator=gen)[:k].to(device=x.device)
    centroids = x.index_select(0, init_idx).clone()
    labels = torch.zeros((n,), dtype=torch.long, device=x.device)

    for _ in range(max(int(iters), 1)):
        distances = torch.cdist(x, centroids, p=2.0)
        labels = torch.argmin(distances, dim=1)
        sums = torch.zeros_like(centroids)
        counts = torch.bincount(labels, minlength=k).to(dtype=x.dtype, device=x.device)
        sums.index_add_(0, labels, x)
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
    return labels, centroids


def _teacher_cluster_count(n_tokens: int, args: argparse.Namespace) -> int:
    if int(args.full_clusters) > 0:
        return max(1, min(int(args.full_clusters), int(n_tokens)))
    k = max(1, int(round(float(n_tokens) / float(max(int(args.target_cluster_size), 1)))))
    return max(1, min(k, int(args.max_full_clusters), int(n_tokens)))


def _planned_k_values(n_tokens: int, m: int, target_cluster_size: int) -> List[int]:
    return choose_learned_k_values(
        {
            "M": int(m),
            "subspace_parts": int(m),
            "min_subspace_k": 2,
            "overpartition_factor": 1.0,
        },
        int(n_tokens),
        int(target_cluster_size),
        default_m=int(m),
    )


def _build_teacher(keys: torch.Tensor, args: argparse.Namespace, seed: int) -> Teacher:
    k_full = _teacher_cluster_count(int(keys.size(0)), args)
    labels, centroids = _kmeans(keys, k_full, int(args.kmeans_iters), seed)
    teacher_for_token = centroids.index_select(0, labels)
    centered = keys.float() - keys.float().mean(dim=0, keepdim=True)
    scale = float(centered.pow(2).mean().item())
    return Teacher(
        keys=keys.float().contiguous(),
        teacher_centroids_for_token=teacher_for_token.contiguous(),
        teacher_mse_scale=max(scale, 1e-12),
    )


def _combined_codes(labels_by_group: Sequence[torch.Tensor], k_values: Sequence[int]) -> torch.Tensor:
    code = torch.zeros_like(labels_by_group[0], dtype=torch.long)
    stride = 1
    for labels, k_value in zip(labels_by_group, k_values):
        code += labels.to(dtype=torch.long) * int(stride)
        stride *= max(int(k_value), 1)
    return code


def evaluate_groups(
    teacher: Teacher,
    groups: Sequence[Sequence[int]],
    *,
    target_cluster_size: int,
    kmeans_iters: int,
    seed: int,
) -> Dict[str, float]:
    keys = teacher.keys
    head_dim = int(keys.size(1))
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE-pair partitions")
    dim_groups = _pair_groups_to_dim_groups(groups, head_dim)
    k_values = _planned_k_values(int(keys.size(0)), len(dim_groups), int(target_cluster_size))

    labels_by_group: List[torch.Tensor] = []
    for group_idx, dims in enumerate(dim_groups):
        dim_idx = torch.tensor(dims, dtype=torch.long, device=keys.device)
        sub_keys = keys.index_select(1, dim_idx)
        labels, _ = _kmeans(
            sub_keys,
            int(k_values[group_idx]),
            int(kmeans_iters),
            int(seed) + 1009 * (group_idx + 1),
        )
        labels_by_group.append(labels)

    codes = _combined_codes(labels_by_group, k_values)
    _, inverse = torch.unique(codes, sorted=True, return_inverse=True)
    cell_count = int(inverse.max().item()) + 1 if inverse.numel() else 0
    counts = torch.bincount(inverse, minlength=cell_count).to(dtype=keys.dtype, device=keys.device)
    sums = torch.zeros((cell_count, head_dim), dtype=keys.dtype, device=keys.device)
    sums.index_add_(0, inverse, keys)
    cell_centroids = sums / counts.clamp_min(1.0).unsqueeze(1)
    imi_for_token = cell_centroids.index_select(0, inverse)

    diff = teacher.teacher_centroids_for_token - imi_for_token
    centroid_mse = float(diff.pow(2).mean().item())
    normalized = centroid_mse / teacher.teacher_mse_scale
    avg_cell_size = float(keys.size(0)) / float(max(cell_count, 1))
    max_cell_size = int(torch.bincount(inverse).max().item()) if inverse.numel() else 0
    return {
        "centroid_mse": centroid_mse,
        "normalized_centroid_mse": float(normalized),
        "cell_count": float(cell_count),
        "avg_cell_size": float(avg_cell_size),
        "max_cell_size": float(max_cell_size),
        "k_product": float(math.prod(max(int(v), 1) for v in k_values)),
        "k_values": [int(value) for value in k_values],
    }


def _random_swap(groups: List[List[int]], rng: random.Random) -> List[List[int]]:
    proposal = _copy_groups(groups)
    nonempty = [idx for idx, group in enumerate(proposal) if group]
    a, b = rng.sample(nonempty, 2)
    ia = rng.randrange(len(proposal[a]))
    ib = rng.randrange(len(proposal[b]))
    proposal[a][ia], proposal[b][ib] = proposal[b][ib], proposal[a][ia]
    proposal[a].sort()
    proposal[b].sort()
    return proposal


def optimize_scope(
    keys: torch.Tensor,
    *,
    args: argparse.Namespace,
    scope_name: str,
    seed: int,
) -> SearchResult:
    device = _resolve_device(str(args.device))
    sampled = _sample_rows(keys, int(args.sample_tokens), seed).to(device=device, dtype=torch.float32)
    num_pairs = int(sampled.size(1)) // 2
    m = int(args.m)
    teacher = _build_teacher(sampled, args, seed)

    rng = random.Random(int(seed))
    starts: List[Tuple[str, List[List[int]]]] = [("sequential", _sequential_groups(num_pairs, m))]
    for idx in range(max(int(args.starts), 0)):
        starts.append((f"random{idx}", _random_groups(num_pairs, m, rng)))

    best_groups: Optional[List[List[int]]] = None
    best_metrics: Optional[Dict[str, object]] = None
    best_origin = "none"
    cache: Dict[Tuple[Tuple[int, ...], ...], Dict[str, float]] = {}

    def score(groups: Sequence[Sequence[int]], local_seed: int) -> Dict[str, object]:
        key = tuple(tuple(int(v) for v in group) for group in groups)
        if key not in cache:
            cache[key] = evaluate_groups(
                teacher,
                groups,
                target_cluster_size=int(args.target_cluster_size),
                kmeans_iters=int(args.kmeans_iters),
                seed=local_seed,
            )
        return cache[key]

    for start_idx, (origin, start_groups) in enumerate(starts):
        current = _copy_groups(start_groups)
        current_metrics = score(current, seed + start_idx * 7919)
        if best_metrics is None or current_metrics["normalized_centroid_mse"] < best_metrics["normalized_centroid_mse"]:
            best_groups = _copy_groups(current)
            best_metrics = dict(current_metrics)
            best_origin = origin

        for step in range(max(int(args.steps), 0)):
            proposal = _random_swap(current, rng)
            prop_metrics = score(proposal, seed + start_idx * 7919 + step + 1)
            if prop_metrics["normalized_centroid_mse"] < current_metrics["normalized_centroid_mse"]:
                current = proposal
                current_metrics = prop_metrics
                if best_metrics is None or current_metrics["normalized_centroid_mse"] < best_metrics["normalized_centroid_mse"]:
                    best_groups = _copy_groups(current)
                    best_metrics = dict(current_metrics)
                    best_origin = f"{origin}_swap"

    if best_groups is None or best_metrics is None:
        raise RuntimeError(f"partition search failed for {scope_name}")
    print(
        "[centroid-search] "
        f"{scope_name} mse={best_metrics['normalized_centroid_mse']:.6f} "
        f"cells={int(best_metrics['cell_count'])} origin={best_origin} "
        f"sizes={[len(group) for group in best_groups]}",
        flush=True,
    )
    k_values_obj = best_metrics.get("k_values", [])
    k_values = [int(value) for value in k_values_obj] if isinstance(k_values_obj, list) else []
    return SearchResult(groups=best_groups, metrics=best_metrics, origin=best_origin, k_values=k_values)


def _layer_index_map(actual_layers: Sequence[int]) -> Dict[int, int]:
    return {int(layer): idx for idx, layer in enumerate(actual_layers)}


def _head_objects_from_groups(
    groups: Sequence[Sequence[int]],
    num_heads: int,
    metrics: Dict[str, object],
    origin: str,
):
    return {
        str(head): {
            "groups": _copy_groups(groups),
            "strategy": origin,
            "metrics": {"centroid_structural_error": dict(metrics)},
        }
        for head in range(int(num_heads))
    }


def build_output(args: argparse.Namespace) -> Dict[str, object]:
    keys_lhtd, metadata, actual_layers = _load_keys(args.keys, str(args.layout))
    selected_layers = _parse_layer_ids(str(args.layers), actual_layers)
    layer_to_idx = _layer_index_map(actual_layers)
    num_heads = int(keys_lhtd.size(1))
    output_groups: List[Dict[str, object]] = []
    log_rows: List[Dict[str, object]] = []
    base_seed = int(args.random_seed)

    if str(args.scope) == "layer_head":
        for layer in selected_layers:
            layer_idx = layer_to_idx[int(layer)]
            heads: Dict[str, object] = {}
            layer_k_values: Optional[List[int]] = None
            head_iter = _progress(
                range(num_heads),
                total=num_heads,
                desc=f"search layer {layer}",
                enabled=bool(args.progress),
            )
            for head in head_iter:
                scope_name = f"layer{layer}_head{head}"
                result = optimize_scope(
                    keys_lhtd[layer_idx, head],
                    args=args,
                    scope_name=scope_name,
                    seed=base_seed + layer_idx * 1000 + head,
                )
                heads[str(head)] = {
                    "groups": result.groups,
                    "strategy": result.origin,
                    "metrics": {"centroid_structural_error": result.metrics},
                }
                if layer_k_values is None:
                    layer_k_values = [int(value) for value in result.k_values]
                elif layer_k_values != [int(value) for value in result.k_values]:
                    raise ValueError(
                        f"layer_head scope expected shared k_values within layer {layer}; "
                        f"got {layer_k_values} and {result.k_values}"
                    )
                log_rows.append({"scope": scope_name, "layer": int(layer), "head": int(head), **result.metrics})
            if layer_k_values is None:
                raise RuntimeError(f"no heads optimized for layer {layer}")
            output_groups.append({
                "layer_start": int(layer),
                "layer_end": int(layer) + 1,
                "M": int(args.m),
                "subspace_parts": int(args.m),
                "k_values": layer_k_values,
                "heads": heads,
            })
        return _make_payload(args, metadata, output_groups, log_rows)

    if str(args.scope) == "layer":
        layer_iter = _progress(
            selected_layers,
            total=len(selected_layers),
            desc="search layers",
            enabled=bool(args.progress),
        )
        for layer in layer_iter:
            layer_idx = layer_to_idx[int(layer)]
            scope_name = f"layer{layer}"
            result = optimize_scope(
                keys_lhtd[layer_idx].reshape(-1, keys_lhtd.size(-1)),
                args=args,
                scope_name=scope_name,
                seed=base_seed + layer_idx * 1000,
            )
            output_groups.append({
                "layer_start": int(layer),
                "layer_end": int(layer) + 1,
                "M": int(args.m),
                "subspace_parts": int(args.m),
                "k_values": result.k_values,
                "heads": _head_objects_from_groups(result.groups, num_heads, result.metrics, result.origin),
            })
            log_rows.append({"scope": scope_name, "layer": int(layer), "head": -1, **result.metrics})
        return _make_payload(args, metadata, output_groups, log_rows)

    if str(args.scope) == "layer_group":
        selected_set = set(int(layer) for layer in selected_layers)
        groups = layer_groups(len(actual_layers), int(args.layer_groups))
        group_iter = _progress(
            list(enumerate(groups)),
            total=len(groups),
            desc="search layer groups",
            enabled=bool(args.progress),
        )
        for group_idx, (start_idx, end_idx) in group_iter:
            group_layers = [int(actual_layers[idx]) for idx in range(start_idx, end_idx)]
            group_layers = [layer for layer in group_layers if layer in selected_set]
            if not group_layers:
                continue
            tensor_indices = [layer_to_idx[layer] for layer in group_layers]
            scope_name = f"layer_group{group_idx}_layers{group_layers[0]}-{group_layers[-1]}"
            result = optimize_scope(
                keys_lhtd[tensor_indices].reshape(-1, keys_lhtd.size(-1)),
                args=args,
                scope_name=scope_name,
                seed=base_seed + group_idx * 1000,
            )
            output_groups.append({
                "layer_start": int(group_layers[0]),
                "layer_end": int(group_layers[-1]) + 1,
                "M": int(args.m),
                "subspace_parts": int(args.m),
                "k_values": result.k_values,
                "heads": _head_objects_from_groups(result.groups, num_heads, result.metrics, result.origin),
            })
            log_rows.append({"scope": scope_name, "layer": int(group_layers[0]), "head": -1, **result.metrics})
        return _make_payload(args, metadata, output_groups, log_rows)

    result = optimize_scope(
        keys_lhtd.reshape(-1, keys_lhtd.size(-1)),
        args=args,
        scope_name="global",
        seed=base_seed,
    )
    first_layer = min(selected_layers)
    last_layer = max(selected_layers)
    output_groups.append({
        "layer_start": int(first_layer),
        "layer_end": int(last_layer) + 1,
        "M": int(args.m),
        "subspace_parts": int(args.m),
        "k_values": result.k_values,
        "heads": _head_objects_from_groups(result.groups, num_heads, result.metrics, result.origin),
    })
    log_rows.append({"scope": "global", "layer": -1, "head": -1, **result.metrics})
    return _make_payload(args, metadata, output_groups, log_rows)


def _make_payload(
    args: argparse.Namespace,
    input_metadata: Dict[str, object],
    layer_group_payloads: List[Dict[str, object]],
    log_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "metadata": {
            "format": "adaptive_imi_learned_partition_v1",
            "source_keys": str(args.keys),
            "source_metadata": input_metadata,
            "objective": "full_dim_teacher_centroid_alignment",
            "metric": "mean_token_mse(full_dim_kmeans_centroid, imi_compact_cell_true_centroid)",
            "scope": str(args.scope),
            "M": int(args.m),
            "target_cluster_size": int(args.target_cluster_size),
            "full_clusters": int(args.full_clusters),
            "max_full_clusters": int(args.max_full_clusters),
            "sample_tokens": int(args.sample_tokens),
            "kmeans_iters": int(args.kmeans_iters),
            "starts": int(args.starts),
            "steps": int(args.steps),
            "random_seed": int(args.random_seed),
            "log_rows": log_rows,
        },
        "format": "adaptive_imi_learned_partition_v1",
        "layer_groups": layer_group_payloads,
    }


def main() -> None:
    args = parse_args()
    if int(args.m) != 8:
        print("[centroid-search] warning: hybrid_8way_v2 expects --m 8", flush=True)
    payload = build_output(args)
    json_kwargs = {"ensure_ascii": False}
    if bool(args.compact_json):
        json_kwargs["separators"] = (",", ":")
    else:
        json_kwargs["indent"] = 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, **json_kwargs) + "\n", encoding="utf-8")
    print(f"[centroid-search] wrote {args.out_json}", flush=True)
    if args.out_log is not None:
        args.out_log.parent.mkdir(parents=True, exist_ok=True)
        args.out_log.write_text(
            json.dumps(payload["metadata"]["log_rows"], **json_kwargs) + "\n",
            encoding="utf-8",
        )
        print(f"[centroid-search] wrote {args.out_log}", flush=True)


if __name__ == "__main__":
    main()
