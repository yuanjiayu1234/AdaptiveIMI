from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cache_hub.cluster_plan import choose_learned_k_values
from config import resolve_model_path
from experiments.imi_partition.eval.evaluate_two_stage_recall import (
    _extract_text,
    _learned_dim_groups,
    _parse_layers,
    _read_jsonl,
    _resolve_modeling_module,
    _run_subspace_clustering,
    _sequential_dim_groups,
    _pair_sequential_dim_groups,
)


@dataclass
class LayerDecodeQK:
    queries: torch.Tensor  # [q_heads, head_dim], CPU
    keys: torch.Tensor  # [kv_heads, tokens, head_dim], CPU
    key_start: int
    key_end: int
    prompt_len: int


class FirstDecodeQKCollector:
    """Collect prompt K from prefill and Q from the first decode-token forward."""

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
        self._prefill_keys: Dict[int, tuple[torch.Tensor, int, int, int]] = {}
        self._decode_queries: Dict[int, torch.Tensor] = {}

    def clear(self) -> None:
        self._prefill_keys.clear()
        self._decode_queries.clear()

    def add(self, layer_idx: int, query_states: torch.Tensor, key_states: torch.Tensor) -> None:
        if layer_idx not in self.layer_ids:
            return
        if query_states.dim() != 4 or key_states.dim() != 4:
            return
        if query_states.size(0) != 1 or key_states.size(0) != 1:
            raise ValueError("rope_pair_dot_error expects batch_size=1")

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
            keys = key_states[0, :, key_start:key_end, :].detach().to(device="cpu")
            self._prefill_keys[layer_idx] = (keys.contiguous(), int(key_start), int(key_end), int(seq_len))
            return

        if self.phase == "decode":
            # First decode-token forward has seq dimension 1 in normal cached decoding.
            query = query_states[0, :, -1, :].detach().to(device="cpu")
            self._decode_queries[layer_idx] = query.contiguous()

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


def _iter_prompts(args: argparse.Namespace) -> Iterable[str]:
    if args.jsonl is None:
        raise ValueError("provide --jsonl when --records is not used")
    yield from _read_jsonl(args.jsonl, args.text_field, args.limit)


def _dtype_from_arg(value: str) -> torch.dtype:
    value = value.lower()
    if value == "fp32":
        return torch.float32
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {value}")


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


def collect_first_decode_qk(args: argparse.Namespace) -> List[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_model_path(args.model)
    dtype = _dtype_from_arg(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(args.device)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    layer_ids = _parse_layers(args.layers, num_layers)
    collector = FirstDecodeQKCollector(
        layer_ids=layer_ids,
        static_start=args.static_start,
        static_end=args.static_end,
        max_search_tokens=args.max_search_tokens,
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
                    max_length=args.max_length,
                )
                input_ids = encoded["input_ids"].to(args.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(args.device)

                collector.phase = "prefill"
                prefill_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                next_token = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

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
                    past_key_values=prefill_outputs.past_key_values,
                    use_cache=True,
                )

                layer_records = collector.records()
                item = {
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
                }
                records.append(item)
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


def _parse_methods(value: str) -> List[str]:
    methods = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    allowed = {"sequential", "pair_sequential", "learned"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"unknown methods {unknown}; supported: {sorted(allowed)}")
    return methods


def _parse_k_values(value: str, m: int, token_count: int, target_cluster_size: int) -> List[int]:
    if value.strip():
        out = [int(item) for item in value.replace(",", " ").split() if item.strip()]
    else:
        out = choose_learned_k_values(
            {"M": int(m), "subspace_parts": int(m)},
            int(token_count),
            int(target_cluster_size),
            default_m=int(m),
        )
    if len(out) != int(m):
        raise ValueError(f"k_values length mismatch: got {len(out)} expected {m}")
    return [max(1, min(int(v), int(token_count))) for v in out]


def _assign_to_centroids(values: torch.Tensor, centroids: torch.Tensor, chunk_size: int) -> torch.Tensor:
    labels = []
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, int(values.size(0)), chunk_size):
        end = min(int(values.size(0)), start + chunk_size)
        distances = torch.cdist(values[start:end].float(), centroids.float(), p=2.0)
        labels.append(torch.argmin(distances, dim=1).cpu())
    return torch.cat(labels, dim=0).long() if labels else torch.empty(0, dtype=torch.long)


def reconstruct_by_subspace_kmeans(
    keys: torch.Tensor,
    dim_groups: Sequence[Sequence[int]],
    *,
    k_values: Sequence[int],
    kmeans_iters: int,
    seed: int,
    fit_tokens: int,
    assign_chunk_size: int,
) -> torch.Tensor:
    keys_f = keys.float().contiguous()
    n_tokens = int(keys_f.size(0))
    if n_tokens <= 0:
        raise ValueError("keys must be non-empty")
    fit_n = n_tokens if fit_tokens <= 0 else min(int(fit_tokens), n_tokens)
    gen = torch.Generator().manual_seed(int(seed))
    fit_ids = torch.randperm(n_tokens, generator=gen)[:fit_n] if fit_n < n_tokens else torch.arange(n_tokens)
    recon = torch.empty_like(keys_f)
    for sub_idx, dims in enumerate(dim_groups):
        dim_list = list(map(int, dims))
        train_values = keys_f.index_select(0, fit_ids)[:, dim_list].contiguous()
        centroids, _ = _run_subspace_clustering(
            train_values,
            int(k_values[sub_idx]),
            iters=int(kmeans_iters),
            seed=int(seed) + 1009 * sub_idx,
            metric="l2",
            weights=None,
        )
        all_values = keys_f[:, dim_list].contiguous()
        labels = _assign_to_centroids(all_values, centroids, assign_chunk_size)
        recon[:, dim_list] = centroids.index_select(0, labels.to(centroids.device)).to(recon.device)
    return recon.contiguous()


def pair_split_ratio(dim_groups: Sequence[Sequence[int]], head_dim: int) -> float:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    half = head_dim // 2
    dim_to_group = {}
    for group_idx, dims in enumerate(dim_groups):
        for dim in dims:
            dim_to_group[int(dim)] = int(group_idx)
    split = 0
    for pair in range(half):
        if dim_to_group.get(pair) != dim_to_group.get(pair + half):
            split += 1
    return float(split) / float(max(half, 1))


def _inverse_rope_half_layout(values: torch.Tensor, positions: torch.Tensor, rope_theta: float) -> torch.Tensor:
    values_f = values.float().contiguous()
    if values_f.dim() != 2:
        raise ValueError("RoPE inverse expects [tokens, head_dim]")
    head_dim = int(values_f.size(1))
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE inverse requires even head_dim, got {head_dim}")
    half = head_dim // 2
    pos_f = positions.float().view(-1, 1)
    if int(pos_f.size(0)) != int(values_f.size(0)):
        raise ValueError(f"positions length mismatch: got {pos_f.size(0)} tokens={values_f.size(0)}")
    inv_freq = 1.0 / (float(rope_theta) ** (torch.arange(half, dtype=torch.float32) / float(half)))
    angles = pos_f * inv_freq.view(1, half)
    cos = angles.cos()
    sin = angles.sin()
    first = values_f[:, :half]
    second = values_f[:, half:]
    out = torch.empty_like(values_f)
    out[:, :half] = first * cos + second * sin
    out[:, half:] = second * cos - first * sin
    return out


def rope_commutator_metrics(
    *,
    keys: torch.Tensor,
    dim_groups: Sequence[Sequence[int]],
    key_positions: torch.Tensor,
    rope_theta: float,
) -> Dict[str, float]:
    if float(rope_theta) <= 0.0:
        return {}
    keys_f = keys.float().contiguous()
    unrotated = _inverse_rope_half_layout(keys_f, key_positions, float(rope_theta))
    diff_sq = 0.0
    ref_sq = 0.0
    for dims in dim_groups:
        dim_list = list(map(int, dims))
        projected_rotated = torch.zeros_like(keys_f)
        projected_rotated[:, dim_list] = keys_f[:, dim_list]
        unrotated_after_projection = _inverse_rope_half_layout(projected_rotated, key_positions, float(rope_theta))
        projected_after_unrotate = torch.zeros_like(keys_f)
        projected_after_unrotate[:, dim_list] = unrotated[:, dim_list]
        diff = unrotated_after_projection - projected_after_unrotate
        diff_sq += float(diff.square().sum().item())
        ref_sq += float(projected_after_unrotate.square().sum().item())
    denom = max(ref_sq, 1e-12)
    return {
        "rope_commutator_mse": diff_sq / float(max(keys_f.numel(), 1)),
        "rope_commutator_rel": math.sqrt(diff_sq / denom),
    }


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = a0.norm() * b0.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float(torch.dot(a0, b0).div(denom).item())


def _attention_distortion(exact: torch.Tensor, approx: torch.Tensor, head_dim: int) -> Dict[str, float]:
    scale = math.sqrt(float(max(int(head_dim), 1)))
    logp = torch.log_softmax(exact.float() / scale, dim=0)
    logq = torch.log_softmax(approx.float() / scale, dim=0)
    p = logp.exp()
    q = logq.exp()
    tv = 0.5 * torch.sum(torch.abs(p - q))
    kl = torch.sum(p * (logp - logq))
    top_mass_001 = max(1, int(round(0.001 * int(exact.numel()))))
    top_mass_001 = min(top_mass_001, int(exact.numel()))
    exact_top = torch.topk(exact.float(), top_mass_001).indices
    approx_mass_on_exact_top = q.index_select(0, exact_top).sum()
    exact_mass_on_exact_top = p.index_select(0, exact_top).sum()
    return {
        "attn_tv": float(tv.item()),
        "attn_kl": float(kl.item()),
        "attn_exact_top0p1pct_mass": float(exact_mass_on_exact_top.item()),
        "attn_approx_on_exact_top0p1pct_mass": float(approx_mass_on_exact_top.item()),
    }


def _margin_flip_rate(exact: torch.Tensor, approx: torch.Tensor, sample_pairs: int, seed: int) -> float:
    n = int(exact.numel())
    if n < 2 or sample_pairs <= 0:
        return 0.0
    gen = torch.Generator().manual_seed(int(seed))
    left = torch.randint(0, n, (int(sample_pairs),), generator=gen)
    right = torch.randint(0, n, (int(sample_pairs),), generator=gen)
    mask = left != right
    if not bool(mask.any().item()):
        return 0.0
    left = left[mask]
    right = right[mask]
    exact_delta = exact[left] - exact[right]
    approx_delta = approx[left] - approx[right]
    valid = exact_delta.abs() > 1e-8
    if not bool(valid.any().item()):
        return 0.0
    flips = torch.sign(exact_delta[valid]) != torch.sign(approx_delta[valid])
    return float(flips.float().mean().item())


def dot_error_metrics(
    *,
    keys: torch.Tensor,
    reconstructed: torch.Tensor,
    query: torch.Tensor,
    dim_groups: Sequence[Sequence[int]],
    sample_pairs: int,
    seed: int,
) -> Dict[str, float]:
    keys_f = keys.float().contiguous()
    recon_f = reconstructed.float().contiguous()
    q_f = query.float().contiguous()
    exact = torch.mv(keys_f, q_f)
    approx = torch.mv(recon_f, q_f)
    err = approx - exact
    abs_err = err.abs()
    rmse = torch.sqrt(torch.mean(err.square()))
    exact_std = exact.std(unbiased=False).clamp_min(1e-6)
    exact_abs_mean = exact.abs().mean().clamp_min(1e-6)
    key_residual = recon_f - keys_f
    half = int(keys_f.size(1)) // 2
    pair_err = (
        key_residual[:, :half] * q_f[:half].view(1, half)
        + key_residual[:, half:] * q_f[half:].view(1, half)
    )
    pair_l1 = pair_err.abs().sum(dim=1)
    cancellation = pair_l1 / abs_err.clamp_min(1e-6)
    group_errs = []
    for dims in dim_groups:
        dim_list = list(map(int, dims))
        group_errs.append(torch.mv(key_residual[:, dim_list], q_f[dim_list]))
    group_err = torch.stack(group_errs, dim=1) if group_errs else err.view(-1, 1)
    group_l1 = group_err.abs().sum(dim=1)
    group_cancellation = group_l1 / abs_err.clamp_min(1e-6)
    metrics = {
        "dot_mae": float(abs_err.mean().item()),
        "dot_rmse": float(rmse.item()),
        "dot_nrmse_by_std": float((rmse / exact_std).item()),
        "dot_rel_mae": float((abs_err.mean() / exact_abs_mean).item()),
        "dot_bias": float(err.mean().item()),
        "score_pearson": _pearson_corr(exact, approx),
        "margin_flip_rate": _margin_flip_rate(exact, approx, sample_pairs, seed),
        "pair_projected_l1_mean": float(pair_l1.mean().item()),
        "pair_projected_l1_p90": float(torch.quantile(pair_l1, 0.90).item()),
        "pair_error_cancellation_mean": float(cancellation.mean().item()),
        "subspace_projected_l1_mean": float(group_l1.mean().item()),
        "subspace_projected_l1_p90": float(torch.quantile(group_l1, 0.90).item()),
        "subspace_error_cancellation_mean": float(group_cancellation.mean().item()),
        "key_recon_mse": float(torch.mean(key_residual.square()).item()),
        "pair_split_ratio": pair_split_ratio(dim_groups, int(keys_f.size(1))),
    }
    metrics.update(_attention_distortion(exact, approx, int(keys_f.size(1))))
    return metrics


def _query_items(queries: torch.Tensor, kv_head: int, kv_heads: int, mode: str) -> List[tuple[int, torch.Tensor, str]]:
    q_heads = int(queries.size(0))
    group_size = max(q_heads // max(int(kv_heads), 1), 1)
    q_start = int(kv_head) * group_size
    q_end = min(q_start + group_size, q_heads)
    selected = queries[q_start:q_end].float().contiguous()
    if selected.numel() == 0:
        return []
    mode = mode.lower()
    if mode == "gqa_sum":
        return [(-1, selected.sum(dim=0).contiguous(), "gqa_sum")]
    if mode == "gqa_mean":
        return [(-1, selected.mean(dim=0).contiguous(), "gqa_mean")]
    if mode != "per_q_head":
        raise ValueError("gqa_mode must be per_q_head, gqa_sum or gqa_mean")
    return [
        (q_start + local_idx, selected[local_idx].contiguous(), "per_q_head")
        for local_idx in range(int(selected.size(0)))
    ]


def _method_dim_groups(
    method: str,
    *,
    partition: Optional[dict],
    layer_idx: int,
    kv_head: int,
    head_dim: int,
    m: int,
) -> List[List[int]]:
    if method == "sequential":
        return _sequential_dim_groups(head_dim, m)
    if method == "pair_sequential":
        return _pair_sequential_dim_groups(head_dim, m)
    if method == "learned":
        if partition is None:
            raise ValueError("--partition is required for method learned")
        return _learned_dim_groups(partition, layer_idx, kv_head, head_dim)
    raise ValueError(f"unsupported method: {method}")


def analyze_records(args: argparse.Namespace, records: List[dict]) -> tuple[List[Dict[str, object]], dict]:
    methods = _parse_methods(args.methods)
    partition = json.loads(args.partition.read_text(encoding="utf-8")) if args.partition is not None else None
    rows: List[Dict[str, object]] = []
    for item in records:
        prompt_idx = int(item.get("prompt_idx", len(rows)))
        layers = item.get("layers", {})
        for layer_key, layer_obj in sorted(layers.items(), key=lambda kv: int(kv[0])):
            layer_idx = int(layer_key)
            queries = layer_obj["queries"].float()
            keys_all = layer_obj["keys"].float()
            key_start = int(layer_obj.get("key_start", 0))
            key_end = int(layer_obj.get("key_end", key_start + int(keys_all.size(1))))
            key_positions = torch.arange(key_start, key_end, dtype=torch.float32)
            kv_heads = int(keys_all.size(0))
            head_dim = int(keys_all.size(-1))
            for kv_head in range(kv_heads):
                keys = keys_all[kv_head].contiguous()
                n_tokens = int(keys.size(0))
                m = int(args.subspace_parts)
                k_values = _parse_k_values(args.k_values, m, n_tokens, int(args.target_cluster_size))
                recon_cache: Dict[str, tuple[List[List[int]], torch.Tensor, Dict[str, float]]] = {}
                for method in methods:
                    dim_groups = _method_dim_groups(
                        method,
                        partition=partition,
                        layer_idx=layer_idx,
                        kv_head=kv_head,
                        head_dim=head_dim,
                        m=m,
                    )
                    if len(dim_groups) != m:
                        raise ValueError(f"{method} produced {len(dim_groups)} groups, expected {m}")
                    reconstructed = reconstruct_by_subspace_kmeans(
                        keys,
                        dim_groups,
                        k_values=k_values,
                        kmeans_iters=int(args.kmeans_iters),
                        seed=int(args.seed) + 1000003 * prompt_idx + 10007 * layer_idx + 997 * kv_head,
                        fit_tokens=int(args.fit_tokens),
                        assign_chunk_size=int(args.assign_chunk_size),
                    )
                    structural_metrics = rope_commutator_metrics(
                        keys=keys,
                        dim_groups=dim_groups,
                        key_positions=key_positions,
                        rope_theta=float(args.rope_theta),
                    )
                    recon_cache[method] = (dim_groups, reconstructed, structural_metrics)

                for q_head, query, query_mode in _query_items(queries, kv_head, kv_heads, args.gqa_mode):
                    for method, (dim_groups, reconstructed, structural_metrics) in recon_cache.items():
                        metrics = dot_error_metrics(
                            keys=keys,
                            reconstructed=reconstructed,
                            query=query,
                            dim_groups=dim_groups,
                            sample_pairs=int(args.margin_sample_pairs),
                            seed=int(args.seed) + 17 * prompt_idx + 31 * layer_idx + 43 * kv_head + max(q_head, 0),
                        )
                        metrics.update(structural_metrics)
                        rows.append({
                            "prompt_idx": prompt_idx,
                            "input_tokens": int(item.get("input_tokens", 0)),
                            "first_token_id": int(item.get("first_token_id", -1)),
                            "layer": layer_idx,
                            "kv_head": int(kv_head),
                            "q_head": int(q_head),
                            "query_mode": query_mode,
                            "method": method,
                            "tokens": int(n_tokens),
                            "head_dim": int(head_dim),
                            "M": int(m),
                            "k_values": json.dumps(k_values),
                            **metrics,
                        })
                print(
                    f"[analyze] prompt={prompt_idx} layer={layer_idx} kv_head={kv_head} "
                    f"methods={','.join(methods)} tokens={n_tokens}",
                    flush=True,
                )

    summary = summarize_rows(rows)
    return rows, summary


def summarize_rows(rows: Sequence[Dict[str, object]]) -> dict:
    metric_names = [
        "dot_mae",
        "dot_rmse",
        "dot_nrmse_by_std",
        "dot_rel_mae",
        "score_pearson",
        "margin_flip_rate",
        "attn_tv",
        "attn_kl",
        "attn_approx_on_exact_top0p1pct_mass",
        "pair_projected_l1_mean",
        "subspace_projected_l1_mean",
        "subspace_error_cancellation_mean",
        "pair_split_ratio",
        "rope_commutator_rel",
        "rope_commutator_mse",
        "key_recon_mse",
    ]
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    method_summary = {}
    for method, method_rows in sorted(by_method.items()):
        out = {"count": len(method_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in method_rows if metric in row]
            if values:
                out[f"{metric}_mean"] = sum(values) / len(values)
        method_summary[method] = out

    pair = method_summary.get("pair_sequential")
    seq = method_summary.get("sequential")
    deltas = {}
    if pair and seq:
        for metric in metric_names:
            p_key = f"{metric}_mean"
            if p_key in pair and p_key in seq:
                deltas[f"sequential_minus_pair_{metric}"] = float(seq[p_key]) - float(pair[p_key])

    return {
        "methods": method_summary,
        "deltas": deltas,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[analyze] wrote csv {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU structural dot-product distortion test for raw sequential dims vs "
            "RoPE-pair-preserving splits using first decode-token Q."
        )
    )
    parser.add_argument("--records", type=Path, default=None, help="Load saved first-decode QK records and skip model collection.")
    parser.add_argument("--records-out", type=Path, default=None, help="Save collected first-decode QK records.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--text-field", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=131072)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--static-start", type=int, default=4)
    parser.add_argument("--static-end", type=int, default=1024)
    parser.add_argument("--max-search-tokens", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--methods", type=str, default="sequential,pair_sequential")
    parser.add_argument("--partition", type=Path, default=None, help="Optional learned partition JSON when methods includes learned.")
    parser.add_argument("--subspace-parts", type=int, default=8)
    parser.add_argument("--k-values", type=str, default="", help="Optional comma/space separated k per subspace. Default auto 2/3 plan.")
    parser.add_argument("--target-cluster-size", type=int, default=16)
    parser.add_argument("--kmeans-iters", type=int, default=15)
    parser.add_argument("--fit-tokens", type=int, default=32768, help="Subsample tokens to fit kmeans; <=0 uses all searchable tokens.")
    parser.add_argument("--assign-chunk-size", type=int, default=32768)
    parser.add_argument("--gqa-mode", type=str, default="per_q_head", choices=["per_q_head", "gqa_sum", "gqa_mean"])
    parser.add_argument("--margin-sample-pairs", type=int, default=20000)
    parser.add_argument("--rope-theta", type=float, default=1000000.0, help="RoPE base for structural commutator metrics; set <=0 to disable.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.records is not None:
        records = _load_records(args.records)
        print(f"[records] loaded {args.records} count={len(records)}", flush=True)
    else:
        records = collect_first_decode_qk(args)
        if args.records_out is not None:
            _write_records(args.records_out, records, args)

    rows, summary = analyze_records(args, records)
    write_csv(args.out_csv, rows)
    summary_payload = {
        "metadata": {
            "records": None if args.records is None else str(args.records),
            "jsonl": None if args.jsonl is None else str(args.jsonl),
            "q_source": "first_decode_token",
            "methods": _parse_methods(args.methods),
            "subspace_parts": int(args.subspace_parts),
            "target_cluster_size": int(args.target_cluster_size),
            "k_values": args.k_values,
            "fit_tokens": int(args.fit_tokens),
            "gqa_mode": str(args.gqa_mode),
            "rope_theta": float(args.rope_theta),
            "rows": len(rows),
        },
        **summary,
    }
    out_summary = args.out_summary or args.out_csv.with_suffix(".summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"[analyze] wrote summary {out_summary}", flush=True)


if __name__ == "__main__":
    main()
