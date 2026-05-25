from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from config import resolve_model_path


@dataclass
class LayerQK:
    queries: torch.Tensor  # [q_heads, dim], CPU fp16/bf16
    keys: torch.Tensor  # [kv_heads, tokens, dim], CPU fp16/bf16
    key_start: int
    key_end: int
    seq_len: int


class LastTokenQKCollector:
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
        self.records: Dict[int, LayerQK] = {}

    def clear(self) -> None:
        self.records.clear()

    def add(self, layer_idx: int, query_states: torch.Tensor, key_states: torch.Tensor) -> None:
        if layer_idx not in self.layer_ids:
            return
        if query_states.dim() != 4 or key_states.dim() != 4:
            return
        if query_states.size(0) != 1 or key_states.size(0) != 1:
            raise ValueError("offline recall evaluator expects batch_size=1")

        seq_len = int(key_states.size(2))
        if seq_len < 2:
            return

        # Decode searches previous KV. With a prefill-only proxy, use the last
        # token's Q but never include that same token's K in the searchable set.
        key_start = min(self.static_start, seq_len - 1)
        key_end = seq_len - self.static_end if self.static_end > 0 else seq_len - 1
        key_end = min(max(key_end, key_start), seq_len - 1)
        if self.max_search_tokens > 0 and key_end - key_start > self.max_search_tokens:
            key_start = key_end - self.max_search_tokens
        if key_end <= key_start:
            return

        q_last = query_states[0, :, seq_len - 1, :].detach().to(device="cpu")
        k_search = key_states[0, :, key_start:key_end, :].detach().to(device="cpu")
        self.records[int(layer_idx)] = LayerQK(
            queries=q_last.contiguous(),
            keys=k_search.contiguous(),
            key_start=int(key_start),
            key_end=int(key_end),
            seq_len=int(seq_len),
        )


def _read_jsonl(path: Path, text_field: str, limit: int) -> Iterator[str]:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit > 0 and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = _extract_text(item, text_field)
            if text:
                count += 1
                yield text


def _extract_text(item, text_field: str) -> str:
    if text_field != "auto":
        value = item.get(text_field, "")
        return str(value) if value is not None else ""
    for key in ("input", "prompt", "context", "text", "question"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _parse_layers(value: str, num_layers: int) -> List[int]:
    value = value.strip().lower()
    if value in ("all", "*"):
        return list(range(num_layers))
    out = set()
    for part in value.replace(",", " ").split():
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    layers = sorted(item for item in out if 0 <= item < num_layers)
    if not layers:
        raise ValueError("no valid layers selected")
    return layers


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


def _iter_prompts(args: argparse.Namespace) -> Iterable[str]:
    if args.jsonl is None:
        raise ValueError("provide --jsonl for this evaluator")
    yield from _read_jsonl(args.jsonl, args.text_field, args.limit)


def _parse_ints(value: str) -> List[int]:
    return [int(item) for item in value.replace(",", " ").split() if item.strip()]


def _find_layer_group(layer_groups: Sequence[dict], layer_idx: int) -> dict:
    for group in layer_groups:
        start = int(group.get("layer_start", group.get("start", 0)))
        end = int(group.get("layer_end", group.get("end", start + 1)))
        if bool(group.get("layer_end_inclusive", False)):
            end += 1
        if start <= layer_idx < end:
            return group
    raise ValueError(f"no learned partition layer_group covers layer {layer_idx}")


def _learned_dim_groups(partition: dict, layer_idx: int, kv_head: int, head_dim: int) -> List[List[int]]:
    half = head_dim // 2
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE-pair partition requires even head_dim, got {head_dim}")
    group = _find_layer_group(partition.get("layer_groups", []), layer_idx)
    heads = group.get("heads")
    if not isinstance(heads, dict):
        raise ValueError("learned partition layer_group has no heads dict")
    raw_head = heads.get(str(kv_head), heads.get(kv_head))
    if raw_head is None:
        raise ValueError(f"learned partition missing kv head {kv_head}")
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
    if head_dim % m != 0:
        raise ValueError(f"sequential split expects head_dim divisible by M, got {head_dim=} {m=}")
    width = head_dim // m
    return [list(range(start, start + width)) for start in range(0, head_dim, width)]


def _pair_sequential_dim_groups(head_dim: int, m: int) -> List[List[int]]:
    if head_dim % 2 != 0:
        raise ValueError(f"pair sequential split requires even head_dim, got {head_dim}")
    half = head_dim // 2
    if half % m != 0:
        raise ValueError(f"pair count must be divisible by M, got pairs={half} {m=}")
    pairs_per_group = half // m
    groups: List[List[int]] = []
    for start in range(0, half, pairs_per_group):
        dims: List[int] = []
        for pair in range(start, start + pairs_per_group):
            dims.extend([pair, pair + half])
        groups.append(dims)
    return groups


def _subspace_k(num_tokens: int, m: int, target_cluster_size: int, rounding: str) -> int:
    target_cells = max(float(num_tokens) / float(max(target_cluster_size, 1)), 1.0)
    raw = target_cells ** (1.0 / float(max(m, 1)))
    if rounding == "floor":
        value = math.floor(raw)
    elif rounding == "ceil":
        value = math.ceil(raw)
    else:
        value = round(raw)
    return max(int(value), 1)


def _run_kmeans(
    values: torch.Tensor,
    k: int,
    *,
    iters: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = values.float().contiguous()
    n = int(values.size(0))
    if n <= 0:
        raise ValueError("cannot kmeans empty values")
    k = min(max(int(k), 1), n)

    gen = torch.Generator(device=values.device)
    gen.manual_seed(int(seed))
    init = torch.randperm(n, generator=gen, device=values.device)[:k]
    centroids = values.index_select(0, init).contiguous()
    labels = torch.zeros(n, dtype=torch.long, device=values.device)

    for _ in range(max(int(iters), 1)):
        distances = torch.cdist(values, centroids, p=2.0)
        new_labels = torch.argmin(distances, dim=1)
        new_centroids = centroids.clone()
        for cluster_idx in range(k):
            mask = new_labels == cluster_idx
            if torch.any(mask):
                new_centroids[cluster_idx] = values[mask].mean(dim=0)
        if torch.equal(new_labels, labels):
            labels = new_labels
            centroids = new_centroids
            break
        labels = new_labels
        centroids = new_centroids

    return centroids.contiguous(), labels.contiguous()


@dataclass
class IMIIndex:
    dim_groups: List[List[int]]
    k: int
    centroids: List[torch.Tensor]
    token_ids_by_cell: Dict[int, torch.Tensor]
    cell_labels: torch.Tensor
    cell_ids: torch.Tensor


_CELL_CACHE: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def _cell_grid(m: int, k: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (int(m), int(k), str(device))
    cached = _CELL_CACHE.get(key)
    if cached is not None:
        return cached
    ranges = [torch.arange(k, dtype=torch.long, device=device) for _ in range(m)]
    labels = torch.cartesian_prod(*ranges).contiguous()
    strides = (torch.tensor([k**idx for idx in range(m)], dtype=torch.long, device=device)).view(1, m)
    cell_ids = torch.sum(labels * strides, dim=1).contiguous()
    _CELL_CACHE[key] = (labels, cell_ids)
    return labels, cell_ids


def _build_token_cell_map(labels_by_subspace: Sequence[torch.Tensor], k: int) -> Dict[int, torch.Tensor]:
    labels = torch.stack([labels.cpu().long() for labels in labels_by_subspace], dim=0)
    m, n = labels.shape
    strides = torch.tensor([k**idx for idx in range(m)], dtype=torch.long).view(m, 1)
    cell_ids = torch.sum(labels * strides, dim=0)
    order = torch.argsort(cell_ids)
    sorted_cells = cell_ids.index_select(0, order)

    out: Dict[int, torch.Tensor] = {}
    if n == 0:
        return out
    unique_cells, counts = torch.unique_consecutive(sorted_cells, return_counts=True)
    offset = 0
    for cell, count in zip(unique_cells.tolist(), counts.tolist()):
        out[int(cell)] = order[offset : offset + count].contiguous()
        offset += int(count)
    return out


def build_imi_index(
    keys: torch.Tensor,
    dim_groups: Sequence[Sequence[int]],
    *,
    k: int,
    kmeans_iters: int,
    seed: int,
) -> IMIIndex:
    centroids: List[torch.Tensor] = []
    labels_by_subspace: List[torch.Tensor] = []
    for sub_idx, dims in enumerate(dim_groups):
        sub_keys = keys[:, list(dims)].contiguous()
        sub_centroids, labels = _run_kmeans(
            sub_keys,
            k,
            iters=kmeans_iters,
            seed=int(seed) + 1009 * sub_idx,
        )
        centroids.append(sub_centroids)
        labels_by_subspace.append(labels)

    token_ids_by_cell = _build_token_cell_map(labels_by_subspace, k)
    cell_labels, cell_ids = _cell_grid(len(dim_groups), k, keys.device)
    return IMIIndex(
        dim_groups=[list(map(int, dims)) for dims in dim_groups],
        k=int(k),
        centroids=centroids,
        token_ids_by_cell=token_ids_by_cell,
        cell_labels=cell_labels,
        cell_ids=cell_ids,
    )


def search_imi_candidates(
    index: IMIIndex,
    query: torch.Tensor,
    *,
    candidate_budget: int,
    max_probe_cells: int,
) -> Tuple[torch.Tensor, int]:
    sub_scores: List[torch.Tensor] = []
    for dims, centroids in zip(index.dim_groups, index.centroids):
        sub_q = query[list(dims)].float()
        sub_scores.append(torch.mv(centroids.float(), sub_q))

    cell_scores = torch.zeros(index.cell_labels.size(0), dtype=torch.float32, device=query.device)
    for sub_idx, scores in enumerate(sub_scores):
        cell_scores += scores.index_select(0, index.cell_labels[:, sub_idx])

    probe_count = int(cell_scores.numel())
    if max_probe_cells > 0:
        probe_count = min(probe_count, int(max_probe_cells))
    ordered_pos = torch.topk(cell_scores, probe_count, largest=True, sorted=True).indices
    ordered_cell_ids = index.cell_ids.index_select(0, ordered_pos).detach().cpu().tolist()

    chunks: List[torch.Tensor] = []
    total = 0
    probed = 0
    for cell_id in ordered_cell_ids:
        probed += 1
        token_ids = index.token_ids_by_cell.get(int(cell_id))
        if token_ids is None or token_ids.numel() == 0:
            continue
        chunks.append(token_ids)
        total += int(token_ids.numel())
        if total >= int(candidate_budget):
            break

    if not chunks:
        return torch.empty(0, dtype=torch.long, device=query.device), probed
    candidates = torch.cat(chunks, dim=0).to(device=query.device, dtype=torch.long)
    return candidates, probed


def _recall_from_candidates(
    keys: torch.Tensor,
    query: torch.Tensor,
    candidates: torch.Tensor,
    recall_ks: Sequence[int],
) -> Dict[int, Tuple[float, float, float]]:
    full_scores = torch.mv(keys.float(), query.float())
    max_k = min(max(recall_ks), int(keys.size(0)))
    exact_top = torch.topk(full_scores, max_k, largest=True, sorted=True).indices
    candidate_set = set(candidates.detach().cpu().tolist())

    if candidates.numel() > 0:
        candidate_scores = full_scores.index_select(0, candidates)
        rerank_count = min(max_k, int(candidates.numel()))
        rerank_local = torch.topk(candidate_scores, rerank_count, largest=True, sorted=True).indices
        reranked = candidates.index_select(0, rerank_local)
        rerank_set = set(reranked.detach().cpu().tolist())
    else:
        rerank_set = set()

    out: Dict[int, Tuple[float, float, float]] = {}
    exact_top_cpu = exact_top.detach().cpu().tolist()
    for k in recall_ks:
        kk = min(int(k), int(keys.size(0)))
        truth = set(exact_top_cpu[:kk])
        coarse_ordered = candidates[:kk] if candidates.numel() > 0 else candidates
        coarse_set = set(coarse_ordered.detach().cpu().tolist())
        rerank_top = reranked[:kk] if candidates.numel() > 0 else candidates
        rerank_set_k = set(rerank_top.detach().cpu().tolist())
        coarse_recall = len(truth & coarse_set) / float(max(kk, 1))
        candidate_recall = len(truth & candidate_set) / float(max(kk, 1))
        rerank_recall = len(truth & rerank_set_k) / float(max(kk, 1))
        out[int(k)] = (coarse_recall, candidate_recall, rerank_recall)
    return out


def score_decomposition_stats(
    keys: torch.Tensor,
    query: torch.Tensor,
    dim_groups: Sequence[Sequence[int]],
) -> Dict[str, float]:
    full_scores = torch.mv(keys.float(), query.float())
    subspace_sum = torch.zeros_like(full_scores)
    for dims in dim_groups:
        dim_list = list(dims)
        subspace_sum += torch.mv(keys[:, dim_list].float(), query[dim_list].float())
    diff = (full_scores - subspace_sum).abs()
    denom = full_scores.abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float((diff / denom).max().item()) if diff.numel() else 0.0,
        "mean_rel": float((diff / denom).mean().item()) if diff.numel() else 0.0,
    }


def evaluate_layer_record(
    *,
    prompt_idx: int,
    layer_idx: int,
    record: LayerQK,
    partition: dict,
    candidate_budgets: Sequence[int],
    recall_ks: Sequence[int],
    target_cluster_size: int,
    k_rounding: str,
    kmeans_iters: int,
    seed: int,
    ann_device: str,
    include_methods: Sequence[str],
    max_probe_cells: int,
    debug_score_decomposition: bool,
    score_decomposition_rows: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    queries = record.queries.to(device=ann_device)
    keys_by_kv = record.keys.to(device=ann_device)
    q_heads = int(queries.size(0))
    kv_heads = int(keys_by_kv.size(0))
    head_dim = int(keys_by_kv.size(-1))
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads must be divisible by kv_heads for GQA mapping, got {q_heads=} {kv_heads=}")
    group_size = q_heads // kv_heads

    rows: List[Dict[str, object]] = []
    for kv_head in range(kv_heads):
        keys = keys_by_kv[kv_head].float().contiguous()
        n_tokens = int(keys.size(0))
        method_dims: Dict[str, List[List[int]]] = {}
        if "learned" in include_methods:
            method_dims["learned"] = _learned_dim_groups(partition, layer_idx, kv_head, head_dim)
        if "sequential" in include_methods:
            m = len(next(iter(method_dims.values()))) if method_dims else int(partition.get("metadata", {}).get("M", 8) or 8)
            if "learned" not in method_dims:
                group = _find_layer_group(partition.get("layer_groups", []), layer_idx)
                m = int(group.get("M", group.get("subspace_parts", m)))
            method_dims["sequential"] = _sequential_dim_groups(head_dim, m)
        if "pair_sequential" in include_methods:
            m = len(next(iter(method_dims.values()))) if method_dims else int(partition.get("metadata", {}).get("M", 8) or 8)
            if "learned" not in method_dims:
                group = _find_layer_group(partition.get("layer_groups", []), layer_idx)
                m = int(group.get("M", group.get("subspace_parts", m)))
            method_dims["pair_sequential"] = _pair_sequential_dim_groups(head_dim, m)

        for method, dim_groups in method_dims.items():
            if debug_score_decomposition:
                query0 = queries[kv_head * group_size].float().contiguous()
                stats = score_decomposition_stats(keys, query0, dim_groups)
                if score_decomposition_rows is not None:
                    score_decomposition_rows.append({
                        "prompt_idx": int(prompt_idx),
                        "layer": int(layer_idx),
                        "kv_head": int(kv_head),
                        "q_head": int(kv_head * group_size),
                        "method": method,
                        **stats,
                    })
                print(
                    "[score-decomp] "
                    f"prompt={prompt_idx} layer={layer_idx} kv_head={kv_head} method={method} "
                    f"max_abs={stats['max_abs']:.6e} mean_abs={stats['mean_abs']:.6e} "
                    f"max_rel={stats['max_rel']:.6e} mean_rel={stats['mean_rel']:.6e}",
                    flush=True,
                )
            m = len(dim_groups)
            k_subspace = _subspace_k(
                n_tokens,
                m,
                target_cluster_size=target_cluster_size,
                rounding=k_rounding,
            )
            index = build_imi_index(
                keys,
                dim_groups,
                k=k_subspace,
                kmeans_iters=kmeans_iters,
                seed=int(seed) + 1000003 * prompt_idx + 10007 * layer_idx + 997 * kv_head,
            )
            for q_head in range(kv_head * group_size, (kv_head + 1) * group_size):
                query = queries[q_head].float().contiguous()
                for budget in candidate_budgets:
                    candidates, probed_cells = search_imi_candidates(
                        index,
                        query,
                        candidate_budget=int(budget),
                        max_probe_cells=int(max_probe_cells),
                    )
                    recalls = _recall_from_candidates(keys, query, candidates, recall_ks)
                    for recall_k, (coarse_recall, candidate_recall, rerank_recall) in recalls.items():
                        rows.append({
                            "prompt_idx": int(prompt_idx),
                            "layer": int(layer_idx),
                            "method": method,
                            "kv_head": int(kv_head),
                            "q_head": int(q_head),
                            "group_size": int(group_size),
                            "seq_len": int(record.seq_len),
                            "key_start": int(record.key_start),
                            "key_end": int(record.key_end),
                            "search_tokens": int(n_tokens),
                            "M": int(m),
                            "subspace_dim": int(len(dim_groups[0])),
                            "k_subspace": int(k_subspace),
                            "virtual_cells": int(k_subspace ** m),
                            "candidate_budget": int(budget),
                            "candidate_count": int(candidates.numel()),
                            "probed_cells": int(probed_cells),
                            "recall_k": int(recall_k),
                            "coarse_order_recall": float(coarse_recall),
                            "candidate_recall": float(candidate_recall),
                            "rerank_recall": float(rerank_recall),
                        })
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_idx",
        "layer",
        "method",
        "kv_head",
        "q_head",
        "group_size",
        "seq_len",
        "key_start",
        "key_end",
        "search_tokens",
        "M",
        "subspace_dim",
        "k_subspace",
        "virtual_cells",
        "candidate_budget",
        "candidate_count",
        "probed_cells",
        "recall_k",
        "coarse_order_recall",
        "candidate_recall",
        "rerank_recall",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            row["method"],
            row["candidate_budget"],
            row["recall_k"],
        )
        buckets.setdefault(key, []).append(row)
    summary: List[Dict[str, object]] = []
    for (method, budget, recall_k), items in sorted(buckets.items()):
        cand = [float(item["candidate_recall"]) for item in items]
        rerank = [float(item["rerank_recall"]) for item in items]
        coarse = [float(item["coarse_order_recall"]) for item in items]
        counts = [float(item["candidate_count"]) for item in items]
        summary.append({
            "method": method,
            "candidate_budget": int(budget),
            "recall_k": int(recall_k),
            "samples": len(items),
            "coarse_order_recall_mean": sum(coarse) / max(len(coarse), 1),
            "candidate_recall_mean": sum(cand) / max(len(cand), 1),
            "rerank_recall_mean": sum(rerank) / max(len(rerank), 1),
            "candidate_count_mean": sum(counts) / max(len(counts), 1),
        })
    return summary


def _summarize_score_decomposition(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault(str(row["method"]), []).append(row)
    summary: List[Dict[str, object]] = []
    for method, items in sorted(buckets.items()):
        summary.append({
            "method": method,
            "samples": len(items),
            "max_abs_max": max(float(item["max_abs"]) for item in items) if items else 0.0,
            "max_abs_mean": sum(float(item["max_abs"]) for item in items) / max(len(items), 1),
            "mean_abs_mean": sum(float(item["mean_abs"]) for item in items) / max(len(items), 1),
            "max_rel_max": max(float(item["max_rel"]) for item in items) if items else 0.0,
            "mean_rel_mean": sum(float(item["mean_rel"]) for item in items) / max(len(items), 1),
        })
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline two-stage IMI recall evaluator for learned pair partitions vs raw sequential splits."
    )
    parser.add_argument("--model", type=str, default="llama-3-8b-1048k")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--text-field", type=str, default="auto")
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--static-start", type=int, default=32)
    parser.add_argument("--static-end", type=int, default=64)
    parser.add_argument("--max-search-tokens", type=int, default=0)
    parser.add_argument("--candidate-budgets", type=str, default="256,512,1024,2048")
    parser.add_argument("--recall-ks", type=str, default="32,64,128")
    parser.add_argument("--target-cluster-size", type=int, default=16)
    parser.add_argument("--k-rounding", type=str, default="round", choices=["round", "floor", "ceil"])
    parser.add_argument("--kmeans-iters", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", type=str, default="learned,sequential")
    parser.add_argument("--max-probe-cells", type=int, default=0)
    parser.add_argument(
        "--debug-score-decomposition",
        action="store_true",
        help="Print whether full q@k equals the sum of subspace q_sub@k_sub scores for each method.",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ann-device", type=str, default="")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    ann_device = args.ann_device or args.device
    candidate_budgets = _parse_ints(args.candidate_budgets)
    recall_ks = _parse_ints(args.recall_ks)
    methods = [item.strip() for item in args.methods.replace(",", " ").split() if item.strip()]
    allowed = {"learned", "sequential", "pair_sequential"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"unknown methods {unknown}; supported: {sorted(allowed)}")

    partition = json.loads(args.partition.read_text(encoding="utf-8"))
    model_path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(args.device)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    layer_ids = _parse_layers(args.layers, num_layers)
    collector = LastTokenQKCollector(
        layer_ids=layer_ids,
        static_start=args.static_start,
        static_end=args.static_end,
        max_search_tokens=args.max_search_tokens,
    )

    modeling_module = _resolve_modeling_module(model_path, args.model)
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

    all_rows: List[Dict[str, object]] = []
    score_decomposition_rows: List[Dict[str, object]] = []
    prompt_idx = 0
    try:
        with torch.inference_mode():
            for prompt in _iter_prompts(args):
                collector.clear()
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                )
                input_ids = encoded["input_ids"].to(args.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(args.device)
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

                token_count = int(input_ids.size(1))
                print(
                    f"[recall] prompt={prompt_idx} tokens={token_count} collected_layers={len(collector.records)}",
                    flush=True,
                )
                for layer_idx in layer_ids:
                    record = collector.records.get(layer_idx)
                    if record is None:
                        continue
                    rows = evaluate_layer_record(
                        prompt_idx=prompt_idx,
                        layer_idx=layer_idx,
                        record=record,
                        partition=partition,
                        candidate_budgets=candidate_budgets,
                        recall_ks=recall_ks,
                        target_cluster_size=int(args.target_cluster_size),
                        k_rounding=str(args.k_rounding),
                        kmeans_iters=int(args.kmeans_iters),
                        seed=int(args.seed),
                        ann_device=ann_device,
                        include_methods=methods,
                        max_probe_cells=int(args.max_probe_cells),
                        debug_score_decomposition=bool(args.debug_score_decomposition),
                        score_decomposition_rows=score_decomposition_rows,
                    )
                    all_rows.extend(rows)
                    print(
                        f"[recall] prompt={prompt_idx} layer={layer_idx} rows={len(rows)}",
                        flush=True,
                    )
                prompt_idx += 1
    finally:
        modeling_module.apply_rotary_pos_emb = original_apply_rope
        for attention, original_forward in original_forwards:
            attention.forward = original_forward

    _write_csv(args.out_csv, all_rows)
    print(f"[recall] wrote {args.out_csv} rows={len(all_rows)}", flush=True)

    summary = {
        "metadata": {
            "model": args.model,
            "model_path": model_path,
            "jsonl": str(args.jsonl),
            "partition": str(args.partition),
            "layers": layer_ids,
            "limit": int(args.limit),
            "max_length": int(args.max_length),
            "static_start": int(args.static_start),
            "static_end": int(args.static_end),
            "max_search_tokens": int(args.max_search_tokens),
            "candidate_budgets": candidate_budgets,
            "recall_ks": recall_ks,
            "target_cluster_size": int(args.target_cluster_size),
            "k_rounding": str(args.k_rounding),
            "kmeans_iters": int(args.kmeans_iters),
            "methods": methods,
            "debug_score_decomposition": bool(args.debug_score_decomposition),
        },
        "summary": _summarize(all_rows),
        "score_decomposition": _summarize_score_decomposition(score_decomposition_rows),
        "score_decomposition_rows": score_decomposition_rows,
    }
    if args.out_summary is not None:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[recall] wrote {args.out_summary}", flush=True)


if __name__ == "__main__":
    main()
