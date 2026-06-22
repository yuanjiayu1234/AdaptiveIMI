from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cache_hub.cluster_plan import choose_learned_k_values, product_k_values
from config import resolve_model_path


@dataclass
class LayerDecodeQK:
    queries: torch.Tensor  # [q_heads, head_dim], CPU
    keys: torch.Tensor  # [kv_heads, tokens, head_dim], CPU
    key_start: int
    key_end: int
    prompt_len: int


@dataclass
class IMIIndex:
    dim_groups: List[List[int]]
    k_values: List[int]
    subspace_centroids: List[torch.Tensor]
    active_codes: torch.Tensor  # [active_cells, M]
    full_centroids: torch.Tensor  # [active_cells, head_dim]
    token_ids_by_active: List[torch.Tensor]
    virtual_cells: int


@dataclass
class FullKMeansIndex:
    centroids: torch.Tensor
    token_ids_by_cluster: List[torch.Tensor]


@dataclass
class PQIndex:
    dim_groups: List[List[int]]
    centroids: List[torch.Tensor]
    codes: torch.Tensor  # [tokens, M]


class FirstDecodeQKCollector:
    """Collect prompt K from prefill and Q from the first generated token."""

    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        static_start: int,
        static_end: int,
        max_search_tokens: int,
    ) -> None:
        self.layer_ids = set(int(layer) for layer in layer_ids)
        self.static_start = max(int(static_start), 0)
        self.static_end = max(int(static_end), 0)
        self.max_search_tokens = max(int(max_search_tokens), 0)
        self.active_layer: Optional[int] = None
        self.phase = "prefill"
        self._prefill_keys: Dict[int, Tuple[torch.Tensor, int, int, int]] = {}
        self._decode_queries: Dict[int, torch.Tensor] = {}

    def clear(self) -> None:
        self._prefill_keys.clear()
        self._decode_queries.clear()

    def add(self, layer_idx: int, query_states: torch.Tensor, key_states: torch.Tensor) -> None:
        if int(layer_idx) not in self.layer_ids:
            return
        if query_states.dim() != 4 or key_states.dim() != 4:
            return
        if query_states.size(0) != 1 or key_states.size(0) != 1:
            raise ValueError("first-decode evaluator expects batch_size=1")

        layer_idx = int(layer_idx)
        if self.phase == "prefill":
            seq_len = int(key_states.size(2))
            key_start = min(self.static_start, seq_len)
            key_end = seq_len - self.static_end if self.static_end > 0 else seq_len
            key_end = min(max(key_end, key_start), seq_len)
            if self.max_search_tokens > 0 and key_end - key_start > self.max_search_tokens:
                key_start = key_end - self.max_search_tokens
            if key_end <= key_start:
                return
            keys = key_states[0, :, key_start:key_end, :].detach().to(device="cpu").contiguous()
            self._prefill_keys[layer_idx] = (keys, int(key_start), int(key_end), int(seq_len))
            return

        if self.phase == "decode":
            query = query_states[0, :, -1, :].detach().to(device="cpu").contiguous()
            self._decode_queries[layer_idx] = query

    def records(self) -> Dict[int, LayerDecodeQK]:
        out: Dict[int, LayerDecodeQK] = {}
        for layer_idx in sorted(self.layer_ids):
            if layer_idx not in self._prefill_keys or layer_idx not in self._decode_queries:
                continue
            keys, key_start, key_end, prompt_len = self._prefill_keys[layer_idx]
            out[layer_idx] = LayerDecodeQK(
                queries=self._decode_queries[layer_idx],
                keys=keys,
                key_start=key_start,
                key_end=key_end,
                prompt_len=prompt_len,
            )
        return out


def _extract_text(item, text_field: str) -> str:
    if text_field != "auto":
        value = item.get(text_field, "")
        return str(value) if value is not None else ""
    if isinstance(item.get("context"), str) and isinstance(item.get("question"), str):
        parts = [str(item["context"]).strip(), f"Question:\n{str(item['question']).strip()}"]
        choices = []
        for key in ("choice_A", "choice_B", "choice_C", "choice_D", "choice_E"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                choices.append(f"{key[-1]}. {value.strip()}")
        if choices:
            parts.append("Choices:\n" + "\n".join(choices))
        return "\n\n".join(part for part in parts if part)
    for key in ("input", "prompt", "context", "text", "question"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _read_jsonl(
    path: Path,
    text_field: str,
    limit: int,
    *,
    min_prompt_chars: int = 0,
    longest_first: bool = False,
) -> Iterator[str]:
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            if isinstance(obj.get("data"), list):
                items = obj["data"]
            else:
                items = list(obj.values())
        elif isinstance(obj, list):
            items = obj
        else:
            raise ValueError(f"unsupported JSON prompt file format: {path}")
        rendered = []
        min_chars = max(int(min_prompt_chars), 0)
        for item in items:
            if not isinstance(item, dict):
                continue
            text = _extract_text(item, text_field)
            if text and len(text) >= min_chars:
                rendered.append(text)
        if longest_first:
            rendered.sort(key=len, reverse=True)
        count = 0
        for text in rendered:
            if limit > 0 and count >= limit:
                break
            count += 1
            yield text
        return

    buffered = []
    count = 0
    min_chars = max(int(min_prompt_chars), 0)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not longest_first and limit > 0 and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = _extract_text(item, text_field)
            if text and len(text) >= min_chars:
                if longest_first:
                    buffered.append(text)
                    continue
                count += 1
                yield text
    if longest_first:
        buffered.sort(key=len, reverse=True)
        for text in buffered:
            if limit > 0 and count >= limit:
                break
            count += 1
            yield text


def _iter_prompts(args: argparse.Namespace) -> Iterable[str]:
    if args.jsonl is None:
        raise ValueError("provide --jsonl unless --records-in is used")
    yield from _read_jsonl(
        args.jsonl,
        args.text_field,
        args.limit,
        min_prompt_chars=int(args.min_prompt_chars),
        longest_first=bool(args.longest_first),
    )


def _parse_layers(value: str, num_layers: int) -> List[int]:
    text = value.strip().lower()
    if text in ("all", "*"):
        return list(range(num_layers))
    out = set()
    for part in text.replace(",", " ").split():
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            out.update(range(int(start_s), int(end_s) + 1))
        else:
            out.add(int(part))
    layers = sorted(item for item in out if 0 <= item < num_layers)
    if not layers:
        raise ValueError("no valid layers selected")
    return layers


def _parse_ints(value: str) -> List[int]:
    return [int(item) for item in value.replace(",", " ").split() if item.strip()]


def _parse_floats(value: str) -> List[float]:
    return [float(item) for item in value.replace(",", " ").split() if item.strip()]


def _parse_optional_ints(value: str) -> Optional[List[int]]:
    return _parse_ints(value) if str(value).strip() else None


def _resolve_modeling_module(model_path: str, model_name: str):
    import importlib

    probe = f"{model_name} {model_path}".lower()
    if "qwen3" in probe:
        return importlib.import_module("transformers.models.qwen3.modeling_qwen3")
    if "qwen2" in probe:
        return importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    if "mistral" in probe:
        return importlib.import_module("transformers.models.mistral.modeling_mistral")
    return importlib.import_module("transformers.models.llama.modeling_llama")


def _dtype_from_arg(value: str) -> torch.dtype:
    value = str(value).lower()
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def _prefill_last_logits_with_cache(model, *, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
    }
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError:
        pass
    try:
        return model(**kwargs, num_logits_to_keep=1)
    except TypeError:
        pass
    return model(**kwargs)


def _find_layer_group(layer_groups: Sequence[dict], layer_idx: int) -> dict:
    for group in layer_groups:
        start = int(group.get("layer_start", group.get("start", 0)))
        end = int(group.get("layer_end", group.get("end", start + 1)))
        if bool(group.get("layer_end_inclusive", False)):
            end += 1
        if start <= int(layer_idx) < end:
            return group
    raise ValueError(f"no learned partition layer_group covers layer {layer_idx}")


def _learned_dim_groups(partition: dict, layer_idx: int, kv_head: int, head_dim: int) -> List[List[int]]:
    half = int(head_dim) // 2
    if int(head_dim) % 2 != 0:
        raise ValueError(f"RoPE-pair partition requires even head_dim, got {head_dim}")
    group = _find_layer_group(partition.get("layer_groups", []), int(layer_idx))
    heads = group.get("heads")
    if not isinstance(heads, dict):
        raise ValueError("learned partition layer_group has no heads dict")
    raw_head = heads.get(str(kv_head), heads.get(kv_head))
    if raw_head is None:
        raise ValueError(f"learned partition missing kv head {kv_head}")
    if isinstance(raw_head, dict) and raw_head.get("dim_groups") is not None:
        return [list(map(int, dims)) for dims in raw_head["dim_groups"]]
    pair_groups = raw_head.get("groups") if isinstance(raw_head, dict) else raw_head
    if not isinstance(pair_groups, list) or not pair_groups:
        raise ValueError(f"learned partition head {kv_head} has no groups")
    dim_groups: List[List[int]] = []
    for pair_group in pair_groups:
        dims: List[int] = []
        for pair in pair_group:
            pair_idx = int(pair)
            if pair_idx < 0 or pair_idx >= half:
                raise ValueError(f"pair index {pair_idx} out of range for head_dim={head_dim}")
            dims.extend([pair_idx, pair_idx + half])
        dim_groups.append(dims)
    return dim_groups


def _sequential_dim_groups(head_dim: int, m: int) -> List[List[int]]:
    if int(head_dim) % int(m) != 0:
        raise ValueError(f"sequential split expects head_dim divisible by M, got {head_dim=} {m=}")
    width = int(head_dim) // int(m)
    return [list(range(start, start + width)) for start in range(0, int(head_dim), width)]


def _pair_sequential_dim_groups(head_dim: int, m: int) -> List[List[int]]:
    if int(head_dim) % 2 != 0:
        raise ValueError(f"pair sequential split requires even head_dim, got {head_dim}")
    half = int(head_dim) // 2
    if half % int(m) != 0:
        raise ValueError(f"pair count must be divisible by M, got pairs={half} {m=}")
    pairs_per_group = half // int(m)
    groups: List[List[int]] = []
    for start in range(0, half, pairs_per_group):
        dims: List[int] = []
        for pair in range(start, start + pairs_per_group):
            dims.extend([pair, pair + half])
        groups.append(dims)
    return groups


def _run_kmeans(values: torch.Tensor, k: int, *, iters: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    values = values.float().contiguous()
    n = int(values.size(0))
    if n <= 0:
        raise ValueError("cannot kmeans empty values")
    k = min(max(int(k), 1), n)
    gen = torch.Generator(device=values.device)
    gen.manual_seed(int(seed))
    init = torch.randperm(n, generator=gen, device=values.device)[:k]
    centroids = values.index_select(0, init).clone()
    labels = torch.zeros(n, dtype=torch.long, device=values.device)
    for _ in range(max(int(iters), 1)):
        distances = torch.cdist(values, centroids, p=2.0)
        new_labels = torch.argmin(distances, dim=1)
        sums = torch.zeros_like(centroids)
        counts = torch.bincount(new_labels, minlength=k).to(dtype=values.dtype, device=values.device)
        sums.index_add_(0, new_labels, values)
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
    return centroids.contiguous(), labels.contiguous()


def _run_subspace_clustering(
    values: torch.Tensor,
    k: int,
    *,
    iters: int,
    seed: int,
    metric: str = "l2",
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del weights
    if str(metric).lower() != "l2":
        raise ValueError("first-decode evaluator only supports l2 subspace clustering")
    return _run_kmeans(values, k, iters=iters, seed=seed)


def _combined_codes(labels_by_group: Sequence[torch.Tensor], k_values: Sequence[int]) -> torch.Tensor:
    code = torch.zeros_like(labels_by_group[0], dtype=torch.long)
    stride = 1
    for labels, k_value in zip(labels_by_group, k_values):
        code += labels.to(dtype=torch.long) * int(stride)
        stride *= max(int(k_value), 1)
    return code


def _unpack_code(value: int, k_values: Sequence[int]) -> List[int]:
    out: List[int] = []
    code = int(value)
    for k in k_values:
        k = max(int(k), 1)
        out.append(code % k)
        code //= k
    return out


def _true_centroids_from_token_lists(keys: torch.Tensor, token_lists: Sequence[torch.Tensor]) -> torch.Tensor:
    centroids = torch.empty((len(token_lists), int(keys.size(1))), dtype=torch.float32, device=keys.device)
    for row, token_ids in enumerate(token_lists):
        centroids[row] = keys.index_select(0, token_ids.to(device=keys.device)).float().mean(dim=0)
    return centroids.contiguous()


def _cluster_size_stats(token_lists: Sequence[torch.Tensor], token_count: int) -> Dict[str, float]:
    sizes = sorted(int(ids.numel()) for ids in token_lists if ids is not None)
    active = len(sizes)
    token_count = max(int(token_count), 1)
    if not sizes:
        return {
            "active_cells": 0,
            "cluster_size_min": 0,
            "cluster_size_max": 0,
            "cluster_size_mean": 0.0,
            "cluster_size_p50": 0.0,
            "cluster_size_p90": 0.0,
            "max_cluster_ratio": 0.0,
            "singleton_cell_ratio": 0.0,
        }

    def percentile(frac: float) -> float:
        idx = min(max(int(math.ceil(float(active) * frac)) - 1, 0), active - 1)
        return float(sizes[idx])

    return {
        "active_cells": int(active),
        "cluster_size_min": int(sizes[0]),
        "cluster_size_max": int(sizes[-1]),
        "cluster_size_mean": float(sum(sizes) / active),
        "cluster_size_p50": percentile(0.50),
        "cluster_size_p90": percentile(0.90),
        "max_cluster_ratio": float(sizes[-1] / token_count),
        "singleton_cell_ratio": float(sum(1 for size in sizes if size == 1) / active),
    }


def _subspace_k_values(
    *,
    spec: Optional[dict],
    num_tokens: int,
    m: int,
    target_cluster_size: int,
    override: Optional[Sequence[int]] = None,
) -> List[int]:
    if override is not None:
        values = [int(value) for value in override]
        if len(values) != int(m):
            raise ValueError(f"k_values override length mismatch: got {len(values)} expected {m}")
        if any(value <= 0 for value in values):
            raise ValueError("k_values override entries must be positive")
        return [max(1, min(int(value), int(num_tokens))) for value in values]

    runtime_spec = dict(spec or {})
    runtime_spec.setdefault("M", int(m))
    runtime_spec.setdefault("subspace_parts", int(m))
    values = choose_learned_k_values(
        runtime_spec,
        int(num_tokens),
        int(target_cluster_size),
        default_m=int(m),
    )
    if len(values) != int(m):
        raise ValueError(f"k_values length mismatch: got {len(values)} expected {m}")
    return [max(1, min(int(value), int(num_tokens))) for value in values]


def _full_kmeans_cluster_count(n_tokens: int, target_cluster_size: int, max_full_clusters: int) -> int:
    k = max(1, int(round(float(n_tokens) / float(max(int(target_cluster_size), 1)))))
    if int(max_full_clusters) > 0:
        k = min(k, int(max_full_clusters))
    return max(1, min(k, int(n_tokens)))


def build_full_kmeans_index(
    keys: torch.Tensor,
    *,
    target_cluster_size: int,
    max_full_clusters: int,
    kmeans_iters: int,
    seed: int,
) -> FullKMeansIndex:
    k = _full_kmeans_cluster_count(int(keys.size(0)), target_cluster_size, max_full_clusters)
    _, labels = _run_kmeans(keys, k, iters=kmeans_iters, seed=seed)
    token_lists: List[torch.Tensor] = []
    for cluster_idx in range(k):
        token_ids = torch.nonzero(labels == cluster_idx, as_tuple=False).flatten().to(dtype=torch.long)
        if token_ids.numel() > 0:
            token_lists.append(token_ids.cpu().contiguous())
    centroids = _true_centroids_from_token_lists(keys.float(), token_lists)
    return FullKMeansIndex(centroids=centroids, token_ids_by_cluster=token_lists)


def build_imi_index(
    keys: torch.Tensor,
    dim_groups: Sequence[Sequence[int]],
    *,
    k_values: Sequence[int],
    kmeans_iters: int,
    seed: int,
) -> IMIIndex:
    keys = keys.float().contiguous()
    labels_by_group: List[torch.Tensor] = []
    subspace_centroids: List[torch.Tensor] = []
    for group_idx, dims in enumerate(dim_groups):
        sub_keys = keys[:, list(map(int, dims))].contiguous()
        centroids, labels = _run_subspace_clustering(
            sub_keys,
            int(k_values[group_idx]),
            iters=int(kmeans_iters),
            seed=int(seed) + 1009 * group_idx,
            metric="l2",
        )
        subspace_centroids.append(centroids.float().contiguous())
        labels_by_group.append(labels.cpu().long().contiguous())

    codes = _combined_codes(labels_by_group, k_values)
    unique_codes, inverse = torch.unique(codes, sorted=True, return_inverse=True)
    token_lists: List[torch.Tensor] = []
    for row in range(int(unique_codes.numel())):
        token_lists.append(torch.nonzero(inverse == row, as_tuple=False).flatten().cpu().contiguous())
    active_codes = torch.tensor(
        [_unpack_code(int(code), k_values) for code in unique_codes.tolist()],
        dtype=torch.long,
        device=keys.device,
    )
    full_centroids = _true_centroids_from_token_lists(keys, token_lists)
    return IMIIndex(
        dim_groups=[list(map(int, dims)) for dims in dim_groups],
        k_values=[int(value) for value in k_values],
        subspace_centroids=subspace_centroids,
        active_codes=active_codes.contiguous(),
        full_centroids=full_centroids.contiguous(),
        token_ids_by_active=token_lists,
        virtual_cells=int(product_k_values(k_values)),
    )


def build_pq_index(
    keys: torch.Tensor,
    dim_groups: Sequence[Sequence[int]],
    *,
    k: int,
    kmeans_iters: int,
    seed: int,
) -> PQIndex:
    centroids: List[torch.Tensor] = []
    codes: List[torch.Tensor] = []
    for group_idx, dims in enumerate(dim_groups):
        sub_keys = keys[:, list(map(int, dims))].float().contiguous()
        sub_centroids, labels = _run_kmeans(
            sub_keys,
            int(k),
            iters=int(kmeans_iters),
            seed=int(seed) + 1009 * group_idx,
        )
        centroids.append(sub_centroids.float().contiguous())
        codes.append(labels.cpu().long().contiguous())
    return PQIndex(
        dim_groups=[list(map(int, dims)) for dims in dim_groups],
        centroids=centroids,
        codes=torch.stack(codes, dim=1).contiguous(),
    )


def _runtime_nprobe(virtual_cells: int, retrieval_budget: float, nprobe_override: int) -> int:
    if int(nprobe_override) > 0:
        return max(1, min(int(nprobe_override), int(virtual_cells)))
    nprobe = max(round((int(virtual_cells) - 1) * float(retrieval_budget)), 1)
    return max(1, min(int(nprobe), int(virtual_cells)))


def _max_retrieval_tokens(n_tokens: int, retrieval_budget: float, override: int) -> int:
    if int(override) > 0:
        return max(1, min(int(override), int(n_tokens)))
    return max(1, min(int(float(retrieval_budget) * int(n_tokens)), int(n_tokens)))


def _token_rerank_budget(n_tokens: int, ratio: float, override: int) -> int:
    if int(override) > 0:
        return max(1, min(int(override), int(n_tokens)))
    return max(1, min(int(float(ratio) * int(n_tokens)), int(n_tokens)))


def _concat_token_lists(token_lists: Sequence[torch.Tensor], positions: Sequence[int]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for pos in positions:
        token_ids = token_lists[int(pos)]
        if token_ids.numel() > 0:
            chunks.append(token_ids)
    if not chunks:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(chunks, dim=0).to(dtype=torch.long).contiguous()


def _select_positions_by_token_budget(
    token_lists: Sequence[torch.Tensor],
    ordered_positions: Sequence[int],
    *,
    nprobe: int,
    max_retrieval_tokens: int,
) -> List[int]:
    selected: List[int] = []
    prefix_tokens = 0
    for pos in list(ordered_positions)[: int(nprobe)]:
        size = int(token_lists[int(pos)].numel())
        if size > 0 and prefix_tokens < int(max_retrieval_tokens):
            selected.append(int(pos))
        if size > 0:
            prefix_tokens += size
    return selected


def search_flat_clusters(
    centroids: torch.Tensor,
    token_lists: Sequence[torch.Tensor],
    query: torch.Tensor,
    *,
    nprobe: int,
    max_retrieval_tokens: int,
) -> Tuple[torch.Tensor, int]:
    active = int(centroids.size(0))
    if active <= 0:
        return torch.empty(0, dtype=torch.long), 0
    scores = torch.mv(centroids.float(), query.float())
    width = min(int(nprobe), active)
    ordered = torch.topk(scores, width, largest=True, sorted=True).indices.cpu().tolist()
    selected = _select_positions_by_token_budget(
        token_lists,
        ordered,
        nprobe=width,
        max_retrieval_tokens=int(max_retrieval_tokens),
    )
    return _concat_token_lists(token_lists, selected), len(selected)


def search_subspace_clusters(
    index: IMIIndex,
    query: torch.Tensor,
    *,
    nprobe: int,
    candidate_clusters: int,
    max_retrieval_tokens: int,
) -> Tuple[torch.Tensor, int]:
    active = int(index.active_codes.size(0))
    if active <= 0:
        return torch.empty(0, dtype=torch.long), 0
    coarse = torch.zeros(active, dtype=torch.float32, device=query.device)
    active_codes = index.active_codes.to(device=query.device)
    for group_idx, (dims, centroids) in enumerate(zip(index.dim_groups, index.subspace_centroids)):
        sub_q = query[list(map(int, dims))].float()
        table = torch.mv(centroids.to(device=query.device).float(), sub_q)
        coarse += table.index_select(0, active_codes[:, group_idx])
    default_candidates = max(int(nprobe) * 4, 1024)
    candidate_limit = int(candidate_clusters) if int(candidate_clusters) > 0 else default_candidates
    candidate_limit = max(1, min(max(candidate_limit, int(nprobe)), active))
    ordered = torch.topk(coarse, candidate_limit, largest=True, sorted=True).indices.cpu().tolist()
    selected = _select_positions_by_token_budget(
        index.token_ids_by_active,
        ordered,
        nprobe=min(int(nprobe), candidate_limit),
        max_retrieval_tokens=int(max_retrieval_tokens),
    )
    return _concat_token_lists(index.token_ids_by_active, selected), len(selected)


def search_pq_adc(index: PQIndex, query: torch.Tensor, *, candidate_count: int) -> torch.Tensor:
    scores = torch.zeros(int(index.codes.size(0)), dtype=torch.float32, device=query.device)
    codes = index.codes.to(device=query.device)
    for group_idx, (dims, centroids) in enumerate(zip(index.dim_groups, index.centroids)):
        table = torch.mv(centroids.to(device=query.device).float(), query[list(map(int, dims))].float())
        scores += table.index_select(0, codes[:, group_idx])
    count = max(1, min(int(candidate_count), int(scores.numel())))
    return torch.topk(scores, count, largest=True, sorted=True).indices.cpu().long().contiguous()


def _rerank_and_recall(
    keys: torch.Tensor,
    query: torch.Tensor,
    candidates: torch.Tensor,
    *,
    recall_ks: Sequence[int],
    token_rerank_budget: int,
) -> Dict[int, Tuple[float, float, float]]:
    scores = torch.mv(keys.float(), query.float())
    max_k = min(max(int(k) for k in recall_ks), int(keys.size(0)))
    truth = torch.topk(scores, max_k, largest=True, sorted=True).indices.cpu().tolist()
    candidate_set = set(candidates.cpu().tolist())
    if candidates.numel() > 0:
        cand_scores = scores.index_select(0, candidates.to(device=scores.device, dtype=torch.long))
        rerank_count = min(int(token_rerank_budget), int(candidates.numel()))
        rerank_local = torch.topk(cand_scores, rerank_count, largest=True, sorted=True).indices
        reranked = candidates.index_select(0, rerank_local.cpu())
    else:
        reranked = torch.empty(0, dtype=torch.long)

    out: Dict[int, Tuple[float, float, float]] = {}
    for k in recall_ks:
        kk = min(int(k), int(keys.size(0)))
        truth_set = set(truth[:kk])
        coarse_set = set(candidates[:kk].cpu().tolist())
        rerank_set = set(reranked[:kk].cpu().tolist())
        coarse_recall = len(truth_set & coarse_set) / float(max(kk, 1))
        candidate_recall = len(truth_set & candidate_set) / float(max(kk, 1))
        rerank_recall = len(truth_set & rerank_set) / float(max(kk, 1))
        out[int(k)] = (coarse_recall, candidate_recall, rerank_recall)
    return out


def _query_for_kv_head(queries: torch.Tensor, kv_head: int, kv_heads: int, mode: str) -> Tuple[int, torch.Tensor, int]:
    q_heads = int(queries.size(0))
    if q_heads % int(kv_heads) != 0:
        raise ValueError(f"q_heads must be divisible by kv_heads, got {q_heads=} {kv_heads=}")
    group_size = q_heads // int(kv_heads)
    q_start = int(kv_head) * group_size
    q_end = (int(kv_head) + 1) * group_size
    selected = queries[q_start:q_end].float().contiguous()
    mode = str(mode).lower()
    if mode == "sum":
        return -1, selected.sum(dim=0).contiguous(), group_size
    if mode == "mean":
        return -1, selected.mean(dim=0).contiguous(), group_size
    raise ValueError("gqa-mode must be 'sum' or 'mean'; runtime scoring is equivalent to sum")


def _method_dim_groups(
    method: str,
    *,
    partition: Optional[dict],
    layer_idx: int,
    kv_head: int,
    head_dim: int,
    m: int,
) -> List[List[int]]:
    if method == "learned":
        if partition is None:
            raise ValueError("--partition is required for method learned")
        return _learned_dim_groups(partition, layer_idx, kv_head, head_dim)
    if method == "sequential":
        return _sequential_dim_groups(head_dim, m)
    if method == "pair_sequential":
        return _pair_sequential_dim_groups(head_dim, m)
    raise ValueError(f"unsupported IMI method: {method}")


def evaluate_layer_record(
    *,
    prompt_idx: int,
    layer_idx: int,
    record: LayerDecodeQK,
    partition: Optional[dict],
    methods: Sequence[str],
    decode_modes: Sequence[str],
    retrieval_budgets: Sequence[float],
    token_rerank_budget_ratios: Sequence[float],
    recall_ks: Sequence[int],
    target_cluster_size: int,
    max_full_clusters: int,
    kmeans_iters: int,
    seed: int,
    ann_device: str,
    gqa_mode: str,
    subspace_parts: int,
    max_retrieval_tokens_override: int,
    nprobe_override: int,
    candidate_clusters: int,
    token_rerank_budget_override: int,
    pq_subvecs: int,
    pq_subbits: int,
    k_values_overrides: Dict[str, Optional[Sequence[int]]],
) -> List[Dict[str, object]]:
    queries = record.queries.to(device=ann_device)
    keys_by_kv = record.keys.to(device=ann_device)
    kv_heads = int(keys_by_kv.size(0))
    head_dim = int(keys_by_kv.size(-1))
    rows: List[Dict[str, object]] = []

    for kv_head in range(kv_heads):
        keys = keys_by_kv[kv_head].float().contiguous()
        n_tokens = int(keys.size(0))
        q_head, query, group_size = _query_for_kv_head(queries, kv_head, kv_heads, str(gqa_mode))
        base_seed = int(seed) + 1000003 * int(prompt_idx) + 10007 * int(layer_idx) + 997 * int(kv_head)
        built_full: Optional[FullKMeansIndex] = None
        built_imi: Dict[str, IMIIndex] = {}
        built_pq: Optional[PQIndex] = None

        for method in methods:
            method = str(method)
            if method == "full_kmeans":
                built_full = build_full_kmeans_index(
                    keys,
                    target_cluster_size=int(target_cluster_size),
                    max_full_clusters=int(max_full_clusters),
                    kmeans_iters=int(kmeans_iters),
                    seed=base_seed + 17,
                )
                continue
            if method == "pq":
                if int(head_dim) % int(pq_subvecs) != 0:
                    raise ValueError(f"PQ expects head_dim divisible by pq_subvecs, got {head_dim=} {pq_subvecs=}")
                pq_dims = _sequential_dim_groups(head_dim, int(pq_subvecs))
                built_pq = build_pq_index(
                    keys,
                    pq_dims,
                    k=min(2 ** int(pq_subbits), n_tokens),
                    kmeans_iters=int(kmeans_iters),
                    seed=base_seed + 31,
                )
                continue

            dim_groups = _method_dim_groups(
                method,
                partition=partition,
                layer_idx=int(layer_idx),
                kv_head=int(kv_head),
                head_dim=int(head_dim),
                m=int(subspace_parts),
            )
            spec = _find_layer_group(partition.get("layer_groups", []), layer_idx) if method == "learned" and partition else None
            k_values = _subspace_k_values(
                spec=spec,
                num_tokens=n_tokens,
                m=len(dim_groups),
                target_cluster_size=int(target_cluster_size),
                override=k_values_overrides.get(method),
            )
            built_imi[method] = build_imi_index(
                keys,
                dim_groups,
                k_values=k_values,
                kmeans_iters=int(kmeans_iters),
                seed=base_seed + 43,
            )

        for retrieval_budget in retrieval_budgets:
            max_retrieval_tokens = _max_retrieval_tokens(n_tokens, retrieval_budget, max_retrieval_tokens_override)
            for token_ratio in token_rerank_budget_ratios:
                token_budget = _token_rerank_budget(n_tokens, token_ratio, token_rerank_budget_override)

                if built_full is not None:
                    virtual_cells = max(int(built_full.centroids.size(0)), 1)
                    nprobe = _runtime_nprobe(virtual_cells, retrieval_budget, nprobe_override)
                    candidates, selected_clusters = search_flat_clusters(
                        built_full.centroids,
                        built_full.token_ids_by_cluster,
                        query,
                        nprobe=nprobe,
                        max_retrieval_tokens=max_retrieval_tokens,
                    )
                    rows.extend(_rows_for_result(
                        prompt_idx=prompt_idx,
                        layer_idx=layer_idx,
                        method="full_kmeans",
                        decode_mode="flat",
                        kv_head=kv_head,
                        q_head=q_head,
                        group_size=group_size,
                        record=record,
                        n_tokens=n_tokens,
                        subspace_parts=1,
                        k_values=[virtual_cells],
                        virtual_cells=virtual_cells,
                        stats=_cluster_size_stats(built_full.token_ids_by_cluster, n_tokens),
                        retrieval_budget=retrieval_budget,
                        nprobe=nprobe,
                        max_retrieval_tokens=max_retrieval_tokens,
                        token_rerank_budget_ratio=token_ratio,
                        token_rerank_budget=token_budget,
                        candidate_clusters=0,
                        selected_clusters=selected_clusters,
                        candidates=candidates,
                        keys=keys,
                        query=query,
                        recall_ks=recall_ks,
                    ))

                for method, index in built_imi.items():
                    nprobe = _runtime_nprobe(index.virtual_cells, retrieval_budget, nprobe_override)
                    for decode_mode in decode_modes:
                        mode = str(decode_mode).lower()
                        if mode == "flat":
                            candidates, selected_clusters = search_flat_clusters(
                                index.full_centroids,
                                index.token_ids_by_active,
                                query,
                                nprobe=nprobe,
                                max_retrieval_tokens=max_retrieval_tokens,
                            )
                        elif mode == "subspace":
                            candidates, selected_clusters = search_subspace_clusters(
                                index,
                                query,
                                nprobe=nprobe,
                                candidate_clusters=int(candidate_clusters),
                                max_retrieval_tokens=max_retrieval_tokens,
                            )
                        else:
                            raise ValueError("decode_modes must contain only flat/subspace")
                        rows.extend(_rows_for_result(
                            prompt_idx=prompt_idx,
                            layer_idx=layer_idx,
                            method=method,
                            decode_mode=mode,
                            kv_head=kv_head,
                            q_head=q_head,
                            group_size=group_size,
                            record=record,
                            n_tokens=n_tokens,
                            subspace_parts=len(index.dim_groups),
                            k_values=index.k_values,
                            virtual_cells=index.virtual_cells,
                            stats=_cluster_size_stats(index.token_ids_by_active, n_tokens),
                            retrieval_budget=retrieval_budget,
                            nprobe=nprobe,
                            max_retrieval_tokens=max_retrieval_tokens,
                            token_rerank_budget_ratio=token_ratio,
                            token_rerank_budget=token_budget,
                            candidate_clusters=int(candidate_clusters),
                            selected_clusters=selected_clusters,
                            candidates=candidates,
                            keys=keys,
                            query=query,
                            recall_ks=recall_ks,
                        ))

                if built_pq is not None:
                    candidates = search_pq_adc(built_pq, query, candidate_count=max_retrieval_tokens)
                    rows.extend(_rows_for_result(
                        prompt_idx=prompt_idx,
                        layer_idx=layer_idx,
                        method="pq",
                        decode_mode="pq_adc",
                        kv_head=kv_head,
                        q_head=q_head,
                        group_size=group_size,
                        record=record,
                        n_tokens=n_tokens,
                        subspace_parts=int(pq_subvecs),
                        k_values=[min(2 ** int(pq_subbits), n_tokens)] * int(pq_subvecs),
                        virtual_cells=0,
                        stats=_cluster_size_stats([torch.tensor([idx]) for idx in range(n_tokens)], n_tokens),
                        retrieval_budget=retrieval_budget,
                        nprobe=0,
                        max_retrieval_tokens=max_retrieval_tokens,
                        token_rerank_budget_ratio=token_ratio,
                        token_rerank_budget=token_budget,
                        candidate_clusters=0,
                        selected_clusters=0,
                        candidates=candidates,
                        keys=keys,
                        query=query,
                        recall_ks=recall_ks,
                    ))
    return rows


def _rows_for_result(
    *,
    prompt_idx: int,
    layer_idx: int,
    method: str,
    decode_mode: str,
    kv_head: int,
    q_head: int,
    group_size: int,
    record: LayerDecodeQK,
    n_tokens: int,
    subspace_parts: int,
    k_values: Sequence[int],
    virtual_cells: int,
    stats: Dict[str, float],
    retrieval_budget: float,
    nprobe: int,
    max_retrieval_tokens: int,
    token_rerank_budget_ratio: float,
    token_rerank_budget: int,
    candidate_clusters: int,
    selected_clusters: int,
    candidates: torch.Tensor,
    keys: torch.Tensor,
    query: torch.Tensor,
    recall_ks: Sequence[int],
) -> List[Dict[str, object]]:
    recalls = _rerank_and_recall(
        keys,
        query,
        candidates,
        recall_ks=recall_ks,
        token_rerank_budget=int(token_rerank_budget),
    )
    rows: List[Dict[str, object]] = []
    for recall_k, (coarse_recall, candidate_recall, rerank_recall) in recalls.items():
        rows.append({
            "prompt_idx": int(prompt_idx),
            "layer": int(layer_idx),
            "method": method,
            "decode_mode": decode_mode,
            "query_source": "first_decode_token",
            "kv_head": int(kv_head),
            "q_head": int(q_head),
            "group_size": int(group_size),
            "prompt_len": int(record.prompt_len),
            "key_start": int(record.key_start),
            "key_end": int(record.key_end),
            "search_tokens": int(n_tokens),
            "M": int(subspace_parts),
            "k_values": " ".join(str(int(v)) for v in k_values),
            "k_product": int(product_k_values(k_values)) if k_values else 0,
            "virtual_cells": int(virtual_cells),
            **stats,
            "retrieval_budget": float(retrieval_budget),
            "nprobe": int(nprobe),
            "max_retrieval_tokens": int(max_retrieval_tokens),
            "candidate_clusters": int(candidate_clusters),
            "selected_clusters": int(selected_clusters),
            "candidate_count": int(candidates.numel()),
            "token_rerank_budget_ratio": float(token_rerank_budget_ratio),
            "token_rerank_budget": int(token_rerank_budget),
            "recall_k": int(recall_k),
            "coarse_order_recall": float(coarse_recall),
            "candidate_recall": float(candidate_recall),
            "rerank_recall": float(rerank_recall),
        })
    return rows


def _patch_rope_collector(model, model_path: str, model_name: str, collector: FirstDecodeQKCollector):
    modeling_module = _resolve_modeling_module(model_path, model_name)
    original_apply_rope = modeling_module.apply_rotary_pos_emb
    original_forwards = []

    def wrapped_apply_rotary_pos_emb(*rope_args, **rope_kwargs):
        q_embed, k_embed = original_apply_rope(*rope_args, **rope_kwargs)
        if collector.active_layer is not None:
            collector.add(collector.active_layer, q_embed, k_embed)
        return q_embed, k_embed

    modeling_module.apply_rotary_pos_emb = wrapped_apply_rotary_pos_emb
    for layer_idx, layer in enumerate(model.model.layers):
        original_forward = layer.self_attn.forward
        original_forwards.append((layer.self_attn, original_forward))

        def make_forward(fn, idx):
            def wrapped_forward(*forward_args, **forward_kwargs):
                collector.active_layer = idx
                try:
                    return fn(*forward_args, **forward_kwargs)
                finally:
                    collector.active_layer = None

            return wrapped_forward

        layer.self_attn.forward = make_forward(original_forward, layer_idx)

    def restore() -> None:
        modeling_module.apply_rotary_pos_emb = original_apply_rope
        for attention, original_forward in original_forwards:
            attention.forward = original_forward

    return restore


def collect_first_decode_records(args: argparse.Namespace) -> List[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_model_path(args.model)
    dtype = _dtype_from_arg(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_kwargs = {"torch_dtype": dtype}
    if str(args.attn_implementation):
        model_kwargs["attn_implementation"] = str(args.attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(args.device)
    model.eval()

    layer_ids = _parse_layers(args.layers, int(model.config.num_hidden_layers))
    collector = FirstDecodeQKCollector(
        layer_ids=layer_ids,
        static_start=int(args.static_start),
        static_end=int(args.static_end),
        max_search_tokens=int(args.max_search_tokens),
    )
    restore = _patch_rope_collector(model, model_path, args.model, collector)
    records: List[dict] = []
    try:
        with torch.inference_mode():
            for prompt_idx, prompt in enumerate(_iter_prompts(args)):
                collector.clear()
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(args.max_length),
                )
                input_ids = encoded["input_ids"].to(args.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(args.device)

                collector.phase = "prefill"
                prefill = _prefill_last_logits_with_cache(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                decode_attention_mask = None
                if attention_mask is not None:
                    decode_attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(next_token, device=attention_mask.device)],
                        dim=1,
                    )

                collector.phase = "decode"
                model(
                    input_ids=next_token,
                    attention_mask=decode_attention_mask,
                    past_key_values=prefill.past_key_values,
                    use_cache=True,
                )
                layer_records = collector.records()
                records.append({
                    "prompt_idx": int(prompt_idx),
                    "input_tokens": int(input_ids.size(1)),
                    "first_token_id": int(next_token.item()),
                    "layers": {
                        str(layer_idx): {
                            "queries": rec.queries.cpu(),
                            "keys": rec.keys.cpu(),
                            "key_start": int(rec.key_start),
                            "key_end": int(rec.key_end),
                            "prompt_len": int(rec.prompt_len),
                        }
                        for layer_idx, rec in layer_records.items()
                    },
                })
                print(
                    f"[collect] prompt={prompt_idx} tokens={int(input_ids.size(1))} "
                    f"first_token={int(next_token.item())} layers={len(layer_records)}",
                    flush=True,
                )
    finally:
        restore()
    return records


def _load_records(path: Path) -> List[dict]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("records"), list):
        return obj["records"]
    raise ValueError(f"unsupported records file: {path}")


def _write_records(path: Path, records: List[dict], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "model": args.model,
                "jsonl": None if args.jsonl is None else str(args.jsonl),
                "max_length": int(args.max_length),
                "static_start": int(args.static_start),
                "static_end": int(args.static_end),
                "max_search_tokens": int(args.max_search_tokens),
                "q_source": "first_decode_token",
            },
            "records": records,
        },
        path,
    )
    print(f"[collect] wrote records {path}", flush=True)


def _record_to_layer_qk(layer_obj: dict) -> LayerDecodeQK:
    return LayerDecodeQK(
        queries=layer_obj["queries"].contiguous(),
        keys=layer_obj["keys"].contiguous(),
        key_start=int(layer_obj["key_start"]),
        key_end=int(layer_obj["key_end"]),
        prompt_len=int(layer_obj.get("prompt_len", layer_obj.get("seq_len", 0))),
    )


def _csv_fieldnames() -> List[str]:
    return [
        "prompt_idx",
        "layer",
        "method",
        "decode_mode",
        "query_source",
        "kv_head",
        "q_head",
        "group_size",
        "prompt_len",
        "key_start",
        "key_end",
        "search_tokens",
        "M",
        "k_values",
        "k_product",
        "virtual_cells",
        "active_cells",
        "cluster_size_min",
        "cluster_size_max",
        "cluster_size_mean",
        "cluster_size_p50",
        "cluster_size_p90",
        "max_cluster_ratio",
        "singleton_cell_ratio",
        "retrieval_budget",
        "nprobe",
        "max_retrieval_tokens",
        "candidate_clusters",
        "selected_clusters",
        "candidate_count",
        "token_rerank_budget_ratio",
        "token_rerank_budget",
        "recall_k",
        "coarse_order_recall",
        "candidate_recall",
        "rerank_recall",
    ]


def _init_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames())
        writer.writeheader()


def _append_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames())
        for row in rows:
            writer.writerow(row)


def _summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            row["method"],
            row["decode_mode"],
            row["retrieval_budget"],
            row["token_rerank_budget_ratio"],
            row["recall_k"],
        )
        buckets.setdefault(key, []).append(row)
    summary: List[Dict[str, object]] = []
    for (method, mode, retrieval_budget, token_ratio, recall_k), items in sorted(buckets.items()):
        summary.append({
            "method": method,
            "decode_mode": mode,
            "retrieval_budget": float(retrieval_budget),
            "token_rerank_budget_ratio": float(token_ratio),
            "recall_k": int(recall_k),
            "samples": len(items),
            "coarse_order_recall_mean": sum(float(item["coarse_order_recall"]) for item in items) / len(items),
            "candidate_recall_mean": sum(float(item["candidate_recall"]) for item in items) / len(items),
            "rerank_recall_mean": sum(float(item["rerank_recall"]) for item in items) / len(items),
            "candidate_count_mean": sum(float(item["candidate_count"]) for item in items) / len(items),
            "selected_clusters_mean": sum(float(item["selected_clusters"]) for item in items) / len(items),
        })
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate IMI partitions with first-decode-token Q and runtime-like "
            "flat/subspace cluster selection plus exact token rerank."
        )
    )
    parser.add_argument("--model", type=str, default="llama-3-8b-1048k")
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--text-field", type=str, default="auto")
    parser.add_argument("--partition", type=Path, default=None)
    parser.add_argument("--records-in", type=Path, default=None)
    parser.add_argument("--records-out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--min-prompt-chars", type=int, default=0)
    parser.add_argument("--longest-first", action="store_true")
    parser.add_argument("--max-length", type=int, default=131072)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--static-start", type=int, default=4)
    parser.add_argument("--static-end", type=int, default=1024)
    parser.add_argument("--max-search-tokens", type=int, default=0)
    parser.add_argument("--methods", type=str, default="learned,full_kmeans,sequential,pair_sequential,pq")
    parser.add_argument("--decode-modes", type=str, default="flat,subspace")
    parser.add_argument("--retrieval-budgets", type=str, default="0.1")
    parser.add_argument("--max-retrieval-tokens", type=int, default=0)
    parser.add_argument("--nprobe", type=int, default=0)
    parser.add_argument("--candidate-clusters", type=int, default=0)
    parser.add_argument("--token-rerank-budget-ratios", type=str, default="0.1")
    parser.add_argument("--token-rerank-budget", type=int, default=0)
    parser.add_argument("--recall-ks", type=str, default="32,64,128")
    parser.add_argument("--target-cluster-size", type=int, default=16)
    parser.add_argument("--k-values", type=str, default="", help="Optional explicit comma/space separated IMI k values, e.g. 3,3,2,2,2,2,2,2.")
    parser.add_argument("--learned-k-values", type=str, default="", help="Optional explicit k values for learned IMI.")
    parser.add_argument("--sequential-k-values", type=str, default="", help="Optional explicit k values for sequential IMI.")
    parser.add_argument("--pair-sequential-k-values", type=str, default="", help="Optional explicit k values for pair_sequential IMI.")
    parser.add_argument("--max-full-clusters", type=int, default=8192)
    parser.add_argument("--subspace-parts", type=int, default=8)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gqa-mode", type=str, default="sum", choices=["sum", "mean"])
    parser.add_argument("--pq-subvecs", type=int, default=2)
    parser.add_argument("--pq-subbits", type=int, default=6)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ann-device", type=str, default="")
    parser.add_argument("--attn-implementation", type=str, default="", help="Optional HF attention backend, e.g. flash_attention_2 or sdpa.")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.replace(",", " ").split() if item.strip()]
    allowed_methods = {"learned", "full_kmeans", "sequential", "pair_sequential", "pq"}
    unknown_methods = sorted(set(methods) - allowed_methods)
    if unknown_methods:
        raise ValueError(f"unknown methods {unknown_methods}; supported: {sorted(allowed_methods)}")
    if "learned" in methods and args.partition is None:
        raise ValueError("--partition is required when methods include learned")
    decode_modes = [item.strip().lower() for item in args.decode_modes.replace(",", " ").split() if item.strip()]
    unknown_modes = sorted(set(decode_modes) - {"flat", "subspace"})
    if unknown_modes:
        raise ValueError(f"unknown decode modes {unknown_modes}; supported: flat/subspace")
    retrieval_budgets = _parse_floats(args.retrieval_budgets)
    token_ratios = _parse_floats(args.token_rerank_budget_ratios)
    recall_ks = _parse_ints(args.recall_ks)
    default_k_values = _parse_optional_ints(args.k_values)
    k_values_overrides: Dict[str, Optional[List[int]]] = {
        "learned": _parse_optional_ints(args.learned_k_values) or default_k_values,
        "sequential": _parse_optional_ints(args.sequential_k_values) or default_k_values,
        "pair_sequential": _parse_optional_ints(args.pair_sequential_k_values) or default_k_values,
    }
    ann_device = args.ann_device or args.device
    partition = json.loads(args.partition.read_text(encoding="utf-8")) if args.partition is not None else None

    if args.records_in is not None:
        records = _load_records(args.records_in)
    else:
        records = collect_first_decode_records(args)
        if args.records_out is not None:
            _write_records(args.records_out, records, args)

    _init_csv(args.out_csv)
    all_rows: List[Dict[str, object]] = []
    for item in records:
        prompt_idx = int(item.get("prompt_idx", len(all_rows)))
        layers = item.get("layers", {})
        for layer_key, layer_obj in sorted(layers.items(), key=lambda kv: int(kv[0])):
            layer_idx = int(layer_key)
            record = _record_to_layer_qk(layer_obj)
            rows = evaluate_layer_record(
                prompt_idx=prompt_idx,
                layer_idx=layer_idx,
                record=record,
                partition=partition,
                methods=methods,
                decode_modes=decode_modes,
                retrieval_budgets=retrieval_budgets,
                token_rerank_budget_ratios=token_ratios,
                recall_ks=recall_ks,
                target_cluster_size=int(args.target_cluster_size),
                max_full_clusters=int(args.max_full_clusters),
                kmeans_iters=int(args.kmeans_iters),
                seed=int(args.seed),
                ann_device=ann_device,
                gqa_mode=str(args.gqa_mode),
                subspace_parts=int(args.subspace_parts),
                max_retrieval_tokens_override=int(args.max_retrieval_tokens),
                nprobe_override=int(args.nprobe),
                candidate_clusters=int(args.candidate_clusters),
                token_rerank_budget_override=int(args.token_rerank_budget),
                pq_subvecs=int(args.pq_subvecs),
                pq_subbits=int(args.pq_subbits),
                k_values_overrides=k_values_overrides,
            )
            _append_csv(args.out_csv, rows)
            all_rows.extend(rows)
            print(
                f"[eval] prompt={prompt_idx} layer={layer_idx} rows={len(rows)}",
                flush=True,
            )

    summary = {
        "metadata": {
            "model": args.model,
            "partition": None if args.partition is None else str(args.partition),
            "records_in": None if args.records_in is None else str(args.records_in),
            "query_source": "first_decode_token",
            "decode_semantics": "runtime flat/subspace cluster selection followed by exact token rerank",
            "methods": methods,
            "decode_modes": decode_modes,
            "retrieval_budgets": retrieval_budgets,
            "token_rerank_budget_ratios": token_ratios,
            "recall_ks": recall_ks,
            "target_cluster_size": int(args.target_cluster_size),
            "k_values_overrides": k_values_overrides,
            "max_full_clusters": int(args.max_full_clusters),
            "static_start": int(args.static_start),
            "static_end": int(args.static_end),
            "max_search_tokens": int(args.max_search_tokens),
            "pq_note": "PQ is still an ADC token-preselect diagnostic, not a hybrid_8way_v2 runtime cluster mode.",
        },
        "summary": _summarize(all_rows),
    }
    if args.out_summary is not None:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[eval] wrote {args.out_summary}", flush=True)
    print(f"[eval] wrote {args.out_csv} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
