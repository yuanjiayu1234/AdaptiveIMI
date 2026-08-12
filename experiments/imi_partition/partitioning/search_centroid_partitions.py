from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cache_hub.cluster_plan import choose_learned_k_values
from experiments.imi_partition.partitioning.profiling import ProcessProfiler, write_json


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
    metrics: Dict[str, object]
    origin: str
    k_values: List[int]
    trace_rows: List[Dict[str, object]]


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
    parser.add_argument("--keys", type=Path, required=True, help="Torch file from collect_rope_keys.py.")
    parser.add_argument("--layout", type=str, default="auto", help="Key tensor layout if not stored in metadata.")
    parser.add_argument("--scope", choices=["layer_head", "layer", "layer_group", "global"], default="layer_head")
    parser.add_argument("--layer-groups", type=int, default=3, help="Used when --scope layer_group.")
    parser.add_argument("--layers", type=str, default="all", help="Layer ids to emit, e.g. all or 0-31 or 0,4,8.")
    parser.add_argument("--m", type=int, default=8, help="Number of learned subspaces. hybrid_8way_v2 expects 8.")
    parser.add_argument("--target-cluster-size", type=int, default=16)
    parser.add_argument("--full-clusters", type=int, default=0, help="Override full-dim teacher cluster count.")
    parser.add_argument("--max-full-clusters", type=int, default=0, help="Cap full-dim teacher clusters; 0 means no cap.")
    parser.add_argument("--sample-tokens", type=int, default=131072)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument(
        "--cdist-chunk-size",
        type=int,
        default=8192,
        help="Rows per chunk for K-means distance assignment; prevents huge [tokens, clusters] allocations.",
    )
    parser.add_argument(
        "--oversize-soft-cap",
        type=int,
        default=0,
        help="Cell size where oversized-cell penalty starts; <=0 uses target_cluster_size * 4.",
    )
    parser.add_argument(
        "--oversize-hard-cap",
        type=int,
        default=0,
        help="Cell size where a large penalty starts; <=0 uses target_cluster_size * 8.",
    )
    parser.add_argument("--oversize-weight", type=float, default=0.1)
    parser.add_argument("--oversize-hard-weight", type=float, default=10.0)
    parser.add_argument(
        "--objective-mode",
        choices=["penalty", "constrained_mse"],
        default="constrained_mse",
        help=(
            "penalty uses the soft oversized-cell penalty objective; "
            "constrained_mse first enforces max_cell_size <= oversize_hard_cap, "
            "then minimizes normalized centroid MSE."
        ),
    )
    parser.add_argument("--starts", type=int, default=4, help="Random balanced starts per optimized scope.")
    parser.add_argument("--steps", type=int, default=16, help="Accepted/rejected swap proposals per start.")
    parser.add_argument(
        "--proposal-batch-size",
        type=int,
        default=1,
        help="Generate and evaluate swap proposals in batches; larger values improve GPU utilization.",
    )
    parser.add_argument(
        "--group-kmeans-batch-size",
        type=int,
        default=32,
        help="Number of subspace groups to batch in GPU kmeans label precomputation.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-log", type=Path, default=None)
    parser.add_argument("--profile-out", type=Path, default=None, help="Optional JSON path for search time and peak-memory metrics.")
    parser.add_argument("--trace-out", type=Path, default=None, help="Optional JSONL path for best-so-far search traces.")
    parser.add_argument("--trace-every", type=int, default=0, help="Record a best-so-far trace point every N new candidate evaluations; <=0 disables tracing.")
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
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
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


def _assign_nearest_centroids(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    n = int(x.size(0))
    chunk_size = max(int(chunk_size), 1)
    labels = torch.empty((n,), dtype=torch.long, device=x.device)
    centroid_norms = centroids.pow(2).sum(dim=1)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        x_chunk = x[start:end]
        # L2 nearest centroid: argmin ||x-c||^2 is argmax 2*x*c - ||c||^2.
        scores = torch.mm(x_chunk, centroids.t())
        scores.mul_(2.0)
        scores.sub_(centroid_norms.view(1, -1))
        labels[start:end] = torch.argmax(scores, dim=1)
        del scores
    return labels


def _kmeans(
    x: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
    *,
    cdist_chunk_size: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        labels = _assign_nearest_centroids(x, centroids, chunk_size=int(cdist_chunk_size))
        sums = torch.zeros_like(centroids)
        counts = torch.bincount(labels, minlength=k).to(dtype=x.dtype, device=x.device)
        sums.index_add_(0, labels, x)
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
    return labels, centroids


def _batched_subspace_kmeans_labels(
    keys: torch.Tensor,
    pair_groups: Sequence[Sequence[int]],
    *,
    k: int,
    iters: int,
    seed: int,
    cdist_chunk_size: int,
) -> List[torch.Tensor]:
    if not pair_groups:
        return []
    head_dim = int(keys.size(1))
    dim_groups = _pair_groups_to_dim_groups(pair_groups, head_dim)
    sub_dims = {len(group) for group in dim_groups}
    if len(sub_dims) != 1:
        raise ValueError("batched subspace kmeans requires equal subspace dimensions")

    x = torch.stack(
        [
            keys.index_select(1, torch.tensor(dims, dtype=torch.long, device=keys.device))
            for dims in dim_groups
        ],
        dim=0,
    ).float().contiguous()
    group_count, n, sub_dim = x.shape
    k = max(1, min(int(k), int(n)))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    init_idx = torch.randperm(int(n), generator=gen)[:k].to(device=keys.device)
    centroids = x.index_select(1, init_idx).clone()
    labels = torch.empty((group_count, n), dtype=torch.long, device=keys.device)
    chunk_size = max(int(cdist_chunk_size), 1)

    for _ in range(max(int(iters), 1)):
        centroid_norms = centroids.pow(2).sum(dim=2)
        for start in range(0, int(n), chunk_size):
            end = min(int(n), start + chunk_size)
            scores = torch.bmm(x[:, start:end, :], centroids.transpose(1, 2))
            scores.mul_(2.0)
            scores.sub_(centroid_norms.view(group_count, 1, k))
            labels[:, start:end] = torch.argmax(scores, dim=2)
            del scores

        sums = torch.zeros_like(centroids)
        expanded = labels.unsqueeze(-1).expand(-1, -1, sub_dim)
        sums.scatter_add_(1, expanded, x)
        counts = torch.zeros((group_count, k), dtype=x.dtype, device=x.device)
        counts.scatter_add_(1, labels, torch.ones((group_count, n), dtype=x.dtype, device=x.device))
        updated = sums / counts.clamp_min(1.0).unsqueeze(2)
        centroids = torch.where((counts > 0).unsqueeze(2), updated, centroids)

    out = [labels[idx].to(dtype=torch.int16).contiguous() for idx in range(group_count)]
    del x, centroids, labels
    return out


class SubspaceLabelCache:
    def __init__(
        self,
        *,
        keys: torch.Tensor,
        k_values: Sequence[int],
        kmeans_iters: int,
        seed: int,
        cdist_chunk_size: int,
        batch_size: int,
    ) -> None:
        self.keys = keys
        self.k_values = [int(value) for value in k_values]
        self.kmeans_iters = int(kmeans_iters)
        self.seed = int(seed)
        self.cdist_chunk_size = int(cdist_chunk_size)
        self.batch_size = max(int(batch_size), 1)
        self.cache: Dict[Tuple[int, Tuple[int, ...]], torch.Tensor] = {}

    def key(self, group_idx: int, pairs: Sequence[int]) -> Tuple[int, Tuple[int, ...]]:
        return (int(group_idx), tuple(int(item) for item in pairs))

    def labels(self, group_idx: int, pairs: Sequence[int]) -> torch.Tensor:
        return self.cache[self.key(group_idx, pairs)]

    def ensure(self, candidates: Sequence[Sequence[Sequence[int]]]) -> None:
        missing_by_group: Dict[int, List[Tuple[int, ...]]] = {}
        seen_missing = set()
        for groups in candidates:
            for group_idx, pairs in enumerate(groups):
                cache_key = self.key(group_idx, pairs)
                if cache_key in self.cache or cache_key in seen_missing:
                    continue
                seen_missing.add(cache_key)
                missing_by_group.setdefault(int(group_idx), []).append(cache_key[1])

        for group_idx, pair_groups in missing_by_group.items():
            k_value = int(self.k_values[group_idx])
            group_seed = self.seed + 1009 * (group_idx + 1)
            for start in range(0, len(pair_groups), self.batch_size):
                batch = pair_groups[start : start + self.batch_size]
                labels = _batched_subspace_kmeans_labels(
                    self.keys,
                    batch,
                    k=k_value,
                    iters=self.kmeans_iters,
                    seed=group_seed,
                    cdist_chunk_size=self.cdist_chunk_size,
                )
                for pairs, label in zip(batch, labels):
                    self.cache[(int(group_idx), tuple(int(item) for item in pairs))] = label


def _teacher_cluster_count(n_tokens: int, args: argparse.Namespace) -> int:
    if int(args.full_clusters) > 0:
        return max(1, min(int(args.full_clusters), int(n_tokens)))
    k = max(1, int(math.ceil(float(n_tokens) / float(max(int(args.target_cluster_size), 1)))))
    if int(args.max_full_clusters) > 0:
        k = min(k, int(args.max_full_clusters))
    return max(1, min(k, int(n_tokens)))


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
    progress = os.getenv("IMI_PARTITION_SEARCH_PROGRESS", "0") == "1"
    start_ts = time.perf_counter() if progress else None
    if progress:
        print(
            f"[centroid-search] teacher_start tokens={int(keys.size(0))} "
            f"dim={int(keys.size(1))} full_clusters={int(k_full)} "
            f"iters={int(args.kmeans_iters)} chunk={int(args.cdist_chunk_size)}",
            flush=True,
        )
    labels, centroids = _kmeans(
        keys,
        k_full,
        int(args.kmeans_iters),
        seed,
        cdist_chunk_size=int(args.cdist_chunk_size),
    )
    if progress and start_ts is not None:
        print(
            f"[centroid-search] teacher_done elapsed_s={time.perf_counter() - start_ts:.2f}",
            flush=True,
        )
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
    cdist_chunk_size: int,
    oversize_soft_cap: int,
    oversize_hard_cap: int,
    oversize_weight: float,
    oversize_hard_weight: float,
    label_cache: Optional[SubspaceLabelCache] = None,
) -> Dict[str, float]:
    keys = teacher.keys
    head_dim = int(keys.size(1))
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE-pair partitions")
    dim_groups = _pair_groups_to_dim_groups(groups, head_dim)
    k_values = _planned_k_values(int(keys.size(0)), len(dim_groups), int(target_cluster_size))

    labels_by_group: List[torch.Tensor] = []
    for group_idx, dims in enumerate(dim_groups):
        if label_cache is not None:
            labels = label_cache.labels(group_idx, groups[group_idx])
        else:
            dim_idx = torch.tensor(dims, dtype=torch.long, device=keys.device)
            sub_keys = keys.index_select(1, dim_idx)
            labels, _ = _kmeans(
                sub_keys,
                int(k_values[group_idx]),
                int(kmeans_iters),
                int(seed) + 1009 * (group_idx + 1),
                cdist_chunk_size=int(cdist_chunk_size),
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
    cell_sizes = torch.bincount(inverse, minlength=cell_count) if inverse.numel() else torch.empty(0, device=keys.device)
    max_cell_size = int(cell_sizes.max().item()) if cell_sizes.numel() else 0
    soft_cap = int(oversize_soft_cap)
    if soft_cap <= 0:
        soft_cap = max(int(target_cluster_size) * 4, int(target_cluster_size) + 1)
    hard_cap = int(oversize_hard_cap)
    if hard_cap <= 0:
        hard_cap = max(int(target_cluster_size) * 8, soft_cap)
    hard_cap = max(hard_cap, soft_cap)
    if inverse.numel() and cell_sizes.numel():
        token_cell_sizes = cell_sizes.index_select(0, inverse).to(dtype=torch.float32)
        oversize = torch.relu(token_cell_sizes / float(max(soft_cap, 1)) - 1.0)
        oversize_loss = float(oversize.pow(2).mean().item())
    else:
        oversize_loss = 0.0
    hard_excess = max(float(max_cell_size - hard_cap), 0.0) / float(max(hard_cap, 1))
    hard_loss = hard_excess * hard_excess
    cell_cap_violation = max(float(max_cell_size - hard_cap), 0.0)
    objective_score = (
        float(normalized)
        + float(oversize_weight) * float(oversize_loss)
        + float(oversize_hard_weight) * float(hard_loss)
    )
    return {
        "centroid_mse": centroid_mse,
        "normalized_centroid_mse": float(normalized),
        "cell_count": float(cell_count),
        "avg_cell_size": float(avg_cell_size),
        "max_cell_size": float(max_cell_size),
        "oversize_soft_cap": float(soft_cap),
        "oversize_hard_cap": float(hard_cap),
        "oversize_loss": float(oversize_loss),
        "oversize_hard_loss": float(hard_loss),
        "cell_cap_violation": float(cell_cap_violation),
        "objective_score": float(objective_score),
        "k_product": float(math.prod(max(int(v), 1) for v in k_values)),
        "k_values": [int(value) for value in k_values],
    }


def _partition_score(metrics: Dict[str, float], objective_mode: str) -> Tuple[float, float, int]:
    if str(objective_mode) == "constrained_mse":
        return (
            float(metrics.get("cell_cap_violation", float("inf"))),
            float(metrics.get("normalized_centroid_mse", float("inf"))),
            int(metrics.get("max_cell_size", 0)),
        )
    return (
        float(metrics.get("objective_score", float("inf"))),
        float(metrics.get("normalized_centroid_mse", float("inf"))),
        int(metrics.get("max_cell_size", 0)),
    )


def _partition_better(candidate: Dict[str, float], current: Dict[str, float], objective_mode: str) -> bool:
    return _partition_score(candidate, objective_mode) < _partition_score(current, objective_mode)


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
    scope_start_ts = time.perf_counter()
    device = _resolve_device(str(args.device))
    sampled = _sample_rows(keys, int(args.sample_tokens), seed).to(device=device, dtype=torch.float32)
    num_pairs = int(sampled.size(1)) // 2
    m = int(args.m)
    teacher_cluster_count = _teacher_cluster_count(int(sampled.size(0)), args)
    teacher_start_ts = time.perf_counter()
    teacher = _build_teacher(sampled, args, seed)
    teacher_time_s = time.perf_counter() - teacher_start_ts
    search_start_ts = time.perf_counter()
    k_values = _planned_k_values(int(sampled.size(0)), m, int(args.target_cluster_size))
    label_cache = SubspaceLabelCache(
        keys=teacher.keys,
        k_values=k_values,
        kmeans_iters=int(args.kmeans_iters),
        seed=int(seed) + 17017,
        cdist_chunk_size=int(args.cdist_chunk_size),
        batch_size=int(args.group_kmeans_batch_size),
    )

    rng = random.Random(int(seed))
    starts: List[Tuple[str, List[List[int]]]] = [("sequential", _sequential_groups(num_pairs, m))]
    for idx in range(max(int(args.starts), 0)):
        starts.append((f"random{idx}", _random_groups(num_pairs, m, rng)))

    best_groups: Optional[List[List[int]]] = None
    best_metrics: Optional[Dict[str, object]] = None
    best_origin = "none"
    cache: Dict[Tuple[Tuple[int, ...], ...], Dict[str, float]] = {}

    eval_seed = int(seed) + 17017
    objective_mode = str(args.objective_mode)
    progress = os.getenv("IMI_PARTITION_SEARCH_PROGRESS", "0") == "1"
    progress_every = max(int(os.getenv("IMI_PARTITION_SEARCH_PROGRESS_EVERY", "128")), 1)
    eval_count = 0
    total_candidates = len(starts) * (max(int(args.steps), 0) + 1)
    progress_start_ts = search_start_ts if progress else None
    trace_rows: List[Dict[str, object]] = []
    trace_every = max(int(args.trace_every), 0)
    next_trace_eval = trace_every
    last_trace_eval = -1

    def record_trace(*, force: bool = False) -> None:
        nonlocal next_trace_eval, last_trace_eval
        if trace_every <= 0 or best_metrics is None:
            return
        if not force and eval_count < next_trace_eval:
            return
        if eval_count == last_trace_eval:
            return
        trace_rows.append(
            {
                "scope": scope_name,
                "candidate_evaluations": int(eval_count),
                "elapsed_s": time.perf_counter() - search_start_ts,
                "best_objective_score": float(best_metrics["objective_score"]),
                "best_normalized_centroid_mse": float(best_metrics["normalized_centroid_mse"]),
                "best_max_cell_size": int(best_metrics["max_cell_size"]),
                "best_cell_cap_violation": float(best_metrics.get("cell_cap_violation", 0.0)),
            }
        )
        last_trace_eval = eval_count
        while next_trace_eval <= eval_count:
            next_trace_eval += trace_every

    def score(groups: Sequence[Sequence[int]]) -> Dict[str, object]:
        nonlocal eval_count
        key = tuple(tuple(int(v) for v in group) for group in groups)
        if key not in cache:
            label_cache.ensure([groups])
            cache[key] = evaluate_groups(
                teacher,
                groups,
                target_cluster_size=int(args.target_cluster_size),
                kmeans_iters=int(args.kmeans_iters),
                seed=eval_seed,
                cdist_chunk_size=int(args.cdist_chunk_size),
                oversize_soft_cap=int(args.oversize_soft_cap),
                oversize_hard_cap=int(args.oversize_hard_cap),
                oversize_weight=float(args.oversize_weight),
                oversize_hard_weight=float(args.oversize_hard_weight),
                label_cache=label_cache,
            )
            eval_count += 1
            if progress and (eval_count == 1 or eval_count % progress_every == 0):
                metrics = cache[key]
                elapsed = time.perf_counter() - progress_start_ts if progress_start_ts is not None else 0.0
                print(
                    "[centroid-search] "
                    f"{scope_name} eval={eval_count}/{total_candidates} "
                    f"score={float(metrics['objective_score']):.6f} "
                    f"violation={float(metrics.get('cell_cap_violation', 0.0)):.1f} "
                    f"mse={float(metrics['normalized_centroid_mse']):.6f} "
                    f"max_cell={int(metrics['max_cell_size'])} "
                    f"cells={int(metrics['cell_count'])} elapsed_s={elapsed:.2f}",
                    flush=True,
                )
        return cache[key]

    for start_idx, (origin, start_groups) in enumerate(starts):
        current = _copy_groups(start_groups)
        current_metrics = score(current)
        if best_metrics is None or _partition_better(current_metrics, best_metrics, objective_mode):
            best_groups = _copy_groups(current)
            best_metrics = dict(current_metrics)
            best_origin = origin
        record_trace(force=True)

        step = 0
        max_steps = max(int(args.steps), 0)
        proposal_batch_size = max(int(args.proposal_batch_size), 1)
        while step < max_steps:
            batch_count = min(proposal_batch_size, max_steps - step)
            proposals = [_random_swap(current, rng) for _ in range(batch_count)]
            label_cache.ensure(proposals)
            best_prop_groups: Optional[List[List[int]]] = None
            best_prop_metrics: Optional[Dict[str, object]] = None
            for proposal in proposals:
                prop_metrics = score(proposal)
                if best_prop_metrics is None or _partition_better(prop_metrics, best_prop_metrics, objective_mode):
                    best_prop_groups = proposal
                    best_prop_metrics = prop_metrics
            if (
                best_prop_groups is not None
                and best_prop_metrics is not None
                and _partition_better(best_prop_metrics, current_metrics, objective_mode)
            ):
                current = best_prop_groups
                current_metrics = best_prop_metrics
                if best_metrics is None or _partition_better(current_metrics, best_metrics, objective_mode):
                    best_groups = _copy_groups(current)
                    best_metrics = dict(current_metrics)
                    best_origin = f"{origin}_swap"
            step += batch_count
            record_trace()

    if best_groups is None or best_metrics is None:
        raise RuntimeError(f"partition search failed for {scope_name}")
    record_trace(force=True)
    metrics = dict(best_metrics)
    metrics.update(
        {
            "teacher_cluster_count": int(teacher_cluster_count),
            "teacher_time_s": float(teacher_time_s),
            "search_time_s": float(time.perf_counter() - search_start_ts),
            "scope_total_time_s": float(time.perf_counter() - scope_start_ts),
            "candidate_evaluations": int(eval_count),
        }
    )
    print(
        "[centroid-search] "
        f"{scope_name} max_cell={int(metrics['max_cell_size'])} "
        f"mse={metrics['normalized_centroid_mse']:.6f} "
        f"oversize={metrics['oversize_loss']:.6f} "
        f"violation={metrics.get('cell_cap_violation', 0.0):.1f} "
        f"score={metrics['objective_score']:.6f} "
        f"cells={int(metrics['cell_count'])} mode={objective_mode} origin={best_origin} "
        f"sizes={[len(group) for group in best_groups]}",
        flush=True,
    )
    k_values_obj = metrics.get("k_values", [])
    k_values = [int(value) for value in k_values_obj] if isinstance(k_values_obj, list) else []
    return SearchResult(groups=best_groups, metrics=metrics, origin=best_origin, k_values=k_values, trace_rows=trace_rows)


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
    trace_rows: List[Dict[str, object]] = []
    args._trace_rows = trace_rows
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
                trace_rows.extend(result.trace_rows)
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
            trace_rows.extend(result.trace_rows)
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
            trace_rows.extend(result.trace_rows)
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
    trace_rows.extend(result.trace_rows)
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
            "objective": (
                "hard_cell_cap_then_centroid_alignment"
                if str(args.objective_mode) == "constrained_mse"
                else "centroid_alignment_with_oversized_cell_penalty"
            ),
            "metric": (
                "lexicographic(max(0, max_cell_size - oversize_hard_cap), "
                "normalized_centroid_mse, max_cell_size)"
                if str(args.objective_mode) == "constrained_mse"
                else "mean_token_mse(full_dim_kmeans_centroid, imi_compact_cell_true_centroid) "
                "with oversized-cell penalties"
            ),
            "objective_mode": str(args.objective_mode),
            "scope": str(args.scope),
            "M": int(args.m),
            "target_cluster_size": int(args.target_cluster_size),
            "oversize_soft_cap": int(args.oversize_soft_cap),
            "oversize_hard_cap": int(args.oversize_hard_cap),
            "oversize_weight": float(args.oversize_weight),
            "oversize_hard_weight": float(args.oversize_hard_weight),
            "full_clusters": int(args.full_clusters),
            "max_full_clusters": int(args.max_full_clusters),
            "sample_tokens": int(args.sample_tokens),
            "kmeans_iters": int(args.kmeans_iters),
            "cdist_chunk_size": int(args.cdist_chunk_size),
            "starts": int(args.starts),
            "steps": int(args.steps),
            "proposal_batch_size": int(args.proposal_batch_size),
            "group_kmeans_batch_size": int(args.group_kmeans_batch_size),
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
    profiler = ProcessProfiler(enabled=args.profile_out is not None)
    profiler.start()
    payload = build_output(args)
    profiling = profiler.stop()
    payload["metadata"]["profiling"] = profiling
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
    if args.trace_out is not None:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_rows = list(getattr(args, "_trace_rows", []))
        with args.trace_out.open("w", encoding="utf-8") as file:
            for row in trace_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[centroid-search] wrote trace {args.trace_out} points={len(trace_rows)}", flush=True)
    if args.profile_out is not None:
        write_json(
            args.profile_out,
            {
                "format": "adaptive_imi_search_profile_v1",
                "stage": "partition_search",
                "output": str(args.out_json),
                "profiling": profiling,
                "search_metadata": payload["metadata"],
            },
        )
        print(f"[centroid-search] wrote profile {args.profile_out}", flush=True)


if __name__ == "__main__":
    main()
