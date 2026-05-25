from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from config import resolve_model_path
from experiments.imi_partition.evaluate_two_stage_recall import (
    LastTokenQKCollector,
    _extract_text,
    _find_layer_group,
    _learned_dim_groups,
    _parse_ints,
    _parse_layers,
    _pair_sequential_dim_groups,
    _resolve_modeling_module,
    _subspace_k,
    build_imi_index,
    search_imi_candidates,
)


@dataclass
class QuerySample:
    domain: str
    prompt_idx: int
    layer: int
    kv_head: int
    q_head: int
    query: torch.Tensor
    exact_top: torch.Tensor


@dataclass
class HeadBlock:
    domain: str
    prompt_idx: int
    layer: int
    kv_head: int
    keys: torch.Tensor
    queries: List[QuerySample]


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


def _jsonl_specs(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []
    for item in args.jsonl_spec or []:
        if "=" not in item:
            raise ValueError(f"--jsonl-spec must be name=path, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--jsonl-spec has empty name: {item}")
        specs.append((name, Path(path)))
    if args.jsonl is not None:
        specs.append((args.domain_name, args.jsonl))
    if not specs:
        raise ValueError("provide --jsonl or at least one --jsonl-spec name=path")
    seen = set()
    for name, _ in specs:
        if name in seen:
            raise ValueError(f"duplicate domain name: {name}")
        seen.add(name)
    return specs


def _pair_groups_to_dim_groups(pair_groups: Sequence[Sequence[int]], head_dim: int) -> List[List[int]]:
    if head_dim % 2 != 0:
        raise ValueError(f"pair partition expects even head_dim, got {head_dim}")
    half = head_dim // 2
    dim_groups: List[List[int]] = []
    for group in pair_groups:
        dims: List[int] = []
        for pair in group:
            pair_idx = int(pair)
            if pair_idx < 0 or pair_idx >= half:
                raise ValueError(f"pair index {pair_idx} out of range for head_dim={head_dim}")
            dims.extend([pair_idx, pair_idx + half])
        dim_groups.append(dims)
    return dim_groups


def _dim_groups_to_pair_groups(dim_groups: Sequence[Sequence[int]], head_dim: int) -> List[List[int]]:
    half = head_dim // 2
    out: List[List[int]] = []
    for dims in dim_groups:
        pairs = []
        for dim in dims:
            dim = int(dim)
            if dim < half:
                pairs.append(dim)
        out.append(sorted(pairs))
    return out


def _validate_pair_groups(pair_groups: Sequence[Sequence[int]], num_pairs: int, m: int) -> None:
    if len(pair_groups) != m:
        raise ValueError(f"expected {m} groups, got {len(pair_groups)}")
    flat = [int(pair) for group in pair_groups for pair in group]
    if len(flat) != num_pairs or len(set(flat)) != num_pairs:
        raise ValueError("pair groups must cover every pair exactly once")
    missing = set(range(num_pairs)) - set(flat)
    if missing:
        raise ValueError(f"pair groups missing pairs: {sorted(missing)[:8]}")


def _pair_sequential_groups(num_pairs: int, m: int) -> List[List[int]]:
    if num_pairs % m != 0:
        raise ValueError(f"num_pairs must be divisible by M, got {num_pairs=} {m=}")
    width = num_pairs // m
    return [list(range(start, start + width)) for start in range(0, num_pairs, width)]


def _random_balanced_groups(num_pairs: int, m: int, rng: random.Random) -> List[List[int]]:
    groups = _pair_sequential_groups(num_pairs, m)
    order = list(range(num_pairs))
    rng.shuffle(order)
    width = num_pairs // m
    return [sorted(order[idx * width : (idx + 1) * width]) for idx in range(m)]


def _canonical_groups(pair_groups: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(sorted(int(pair) for pair in group)) for group in pair_groups)


def _copy_groups(pair_groups: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(map(int, group)) for group in pair_groups]


def _propose_swap(pair_groups: Sequence[Sequence[int]], rng: random.Random) -> List[List[int]]:
    groups = _copy_groups(pair_groups)
    a, b = rng.sample(range(len(groups)), 2)
    ia = rng.randrange(len(groups[a]))
    ib = rng.randrange(len(groups[b]))
    groups[a][ia], groups[b][ib] = groups[b][ib], groups[a][ia]
    groups[a].sort()
    groups[b].sort()
    return groups


def _load_checkpoint(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"completed": {}, "starts": {}}
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"invalid checkpoint format: {path}")
    obj.setdefault("completed", {})
    obj.setdefault("starts", {})
    return obj


def _save_checkpoint(path: Optional[Path], state: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(state, f)
    tmp.replace(path)


def _state_key(group_key: Tuple[int, int, int]) -> str:
    return "|".join(map(str, group_key))


def _layer_group_key(group: dict) -> Tuple[int, int, int]:
    return (
        int(group.get("layer_group", -1)),
        int(group.get("layer_start", group.get("start", 0))),
        int(group.get("layer_end", group.get("end", 0))),
    )


def _template_group_for_layer(partition: dict, layer_idx: int) -> dict:
    return _find_layer_group(partition.get("layer_groups", []), layer_idx)


def _group_name(key: Tuple[int, int, int]) -> str:
    group_idx, start, end = key
    return f"layer_group{group_idx}_layers{start}-{end - 1}"


def _record_group_key(partition: dict, layer_idx: int, *, emit_per_layer: bool) -> Tuple[int, int, int]:
    if emit_per_layer:
        return int(layer_idx), int(layer_idx), int(layer_idx) + 1
    group = _template_group_for_layer(partition, layer_idx)
    return _layer_group_key(group)


def _template_for_key(partition: dict, key: Tuple[int, int, int], *, emit_per_layer: bool) -> dict:
    group_idx, start, end = key
    if emit_per_layer:
        try:
            source = _template_group_for_layer(partition, start)
            m = int(source.get("M", source.get("subspace_parts", 8)))
            heads = copy.deepcopy(source.get("heads", {}))
        except Exception:
            m = 8
            heads = {}
        return {
            "layer_group": int(group_idx),
            "layer_start": int(start),
            "layer_end": int(end),
            "M": int(m),
            "subspace_parts": int(m),
            "heads": heads,
        }
    return copy.deepcopy(_template_group_for_layer(partition, start))


def collect_records(args: argparse.Namespace, partition: dict) -> Dict[Tuple[int, int, int], List[Tuple[int, int, object]]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model_path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(args.device)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    layer_ids = _parse_layers(args.layers, num_layers)
    collector = LastTokenQKCollector(
        layer_ids=layer_ids,
        static_start=int(args.static_start),
        static_end=int(args.static_end),
        max_search_tokens=int(args.max_search_tokens),
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

    grouped_records: Dict[Tuple[int, int, int], List[Tuple[int, int, object]]] = {}
    prompt_idx = 0
    try:
        with torch.inference_mode():
            for domain_name, jsonl_path in _jsonl_specs(args):
                domain_prompt_idx = 0
                for prompt in _read_jsonl(jsonl_path, args.text_field, int(args.limit)):
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
                    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    print(
                        f"[collect] domain={domain_name} prompt={domain_prompt_idx} "
                        f"global_prompt={prompt_idx} tokens={int(input_ids.size(1))} "
                        f"layers={len(collector.records)}",
                        flush=True,
                    )
                    for layer_idx, record in sorted(collector.records.items()):
                        key = _record_group_key(partition, layer_idx, emit_per_layer=bool(args.emit_per_layer))
                        grouped_records.setdefault(key, []).append((domain_name, domain_prompt_idx, layer_idx, record))
                    prompt_idx += 1
                    domain_prompt_idx += 1
    finally:
        modeling_module.apply_rotary_pos_emb = original_apply_rope
        for attention, original_forward in original_forwards:
            attention.forward = original_forward

    return grouped_records


def prepare_blocks(
    records: Sequence[Tuple[str, int, int, object]],
    *,
    ann_device: str,
    recall_k: int,
    max_q_heads_per_kv: int,
) -> List[HeadBlock]:
    blocks: List[HeadBlock] = []
    for domain_name, prompt_idx, layer_idx, record in records:
        queries = record.queries.to(device=ann_device).float().contiguous()
        keys_by_kv = record.keys.to(device=ann_device).float().contiguous()
        q_heads = int(queries.size(0))
        kv_heads = int(keys_by_kv.size(0))
        if q_heads % kv_heads != 0:
            raise ValueError(f"q_heads must be divisible by kv_heads, got {q_heads=} {kv_heads=}")
        group_size = q_heads // kv_heads
        take_per_kv = group_size if max_q_heads_per_kv <= 0 else min(group_size, int(max_q_heads_per_kv))
        for kv_head in range(kv_heads):
            keys = keys_by_kv[kv_head].contiguous()
            samples: List[QuerySample] = []
            for offset in range(take_per_kv):
                q_head = kv_head * group_size + offset
                query = queries[q_head].contiguous()
                scores = torch.mv(keys, query)
                top = torch.topk(scores, min(int(recall_k), int(keys.size(0))), largest=True, sorted=True).indices
                samples.append(QuerySample(
                    domain=str(domain_name),
                    prompt_idx=int(prompt_idx),
                    layer=int(layer_idx),
                    kv_head=int(kv_head),
                    q_head=int(q_head),
                    query=query,
                    exact_top=top,
                ))
            blocks.append(HeadBlock(
                domain=str(domain_name),
                prompt_idx=int(prompt_idx),
                layer=int(layer_idx),
                kv_head=int(kv_head),
                keys=keys,
                queries=samples,
            ))
    return blocks


def evaluate_partition(
    pair_groups: Sequence[Sequence[int]],
    blocks: Sequence[HeadBlock],
    *,
    head_dim: int,
    candidate_budget: int,
    recall_k: int,
    target_cluster_size: int,
    k_rounding: str,
    kmeans_iters: int,
    seed: int,
    max_probe_cells: int,
) -> Tuple[float, Dict[str, float]]:
    dim_groups = _pair_groups_to_dim_groups(pair_groups, head_dim)
    recalls: List[float] = []
    recalls_by_domain: Dict[str, List[float]] = {}
    candidate_counts: List[int] = []
    probed_cells: List[int] = []
    for block in blocks:
        n_tokens = int(block.keys.size(0))
        m = len(pair_groups)
        k_subspace = _subspace_k(
            n_tokens,
            m,
            target_cluster_size=int(target_cluster_size),
            rounding=str(k_rounding),
        )
        block_seed = int(seed) + 1000003 * block.prompt_idx + 10007 * block.layer + 997 * block.kv_head
        index = build_imi_index(
            block.keys,
            dim_groups,
            k=k_subspace,
            kmeans_iters=int(kmeans_iters),
            seed=block_seed,
        )
        for sample in block.queries:
            candidates, probed = search_imi_candidates(
                index,
                sample.query,
                candidate_budget=int(candidate_budget),
                max_probe_cells=int(max_probe_cells),
            )
            truth = set(sample.exact_top[: int(recall_k)].detach().cpu().tolist())
            cand = set(candidates.detach().cpu().tolist())
            recall = len(truth & cand) / float(max(len(truth), 1))
            recalls.append(recall)
            recalls_by_domain.setdefault(sample.domain, []).append(recall)
            candidate_counts.append(int(candidates.numel()))
            probed_cells.append(int(probed))
    domain_means = {
        domain: sum(values) / max(len(values), 1)
        for domain, values in sorted(recalls_by_domain.items())
    }
    if domain_means:
        mean_recall = sum(domain_means.values()) / float(len(domain_means))
    else:
        mean_recall = 0.0
    stats = {
        "mean_recall": float(mean_recall),
        "samples": float(len(recalls)),
        "candidate_count_mean": sum(candidate_counts) / max(len(candidate_counts), 1),
        "probed_cells_mean": sum(probed_cells) / max(len(probed_cells), 1),
        "domain_means_json": json.dumps(domain_means, sort_keys=True),
    }
    return mean_recall, stats


def optimize_group(
    *,
    group_key: Tuple[int, int, int],
    template_group: dict,
    blocks: Sequence[HeadBlock],
    args: argparse.Namespace,
    rng: random.Random,
    log_rows: List[Dict[str, object]],
    checkpoint_state: Dict[str, Any],
) -> Tuple[List[List[int]], Dict[str, float]]:
    if not blocks:
        raise ValueError(f"no blocks for {_group_name(group_key)}")
    group_state_key = _state_key(group_key)
    completed = checkpoint_state.setdefault("completed", {})
    if group_state_key in completed:
        saved = completed[group_state_key]
        print(f"[resume] {_group_name(group_key)} completed best={saved.get('best_score', 0.0):.6f}", flush=True)
        return _copy_groups(saved["best_groups"]), dict(saved.get("best_stats", {}))

    head_dim = int(blocks[0].keys.size(-1))
    num_pairs = head_dim // 2
    m = int(template_group.get("M", template_group.get("subspace_parts", 8)))

    try:
        learned_dims = _learned_dim_groups({"layer_groups": [template_group]}, int(blocks[0].layer), 0, head_dim)
        learned_groups = _dim_groups_to_pair_groups(learned_dims, head_dim)
    except Exception:
        learned_groups = _pair_sequential_groups(num_pairs, m)
    pair_seq_groups = _pair_sequential_groups(num_pairs, m)

    starts: List[Tuple[str, List[List[int]]]] = [
        ("learned", learned_groups),
        ("pair_sequential", pair_seq_groups),
    ]
    for idx in range(int(args.random_starts)):
        starts.append((f"random{idx}", _random_balanced_groups(num_pairs, m, rng)))

    seen = set()
    best_groups: Optional[List[List[int]]] = None
    best_score = -1.0
    best_stats: Dict[str, float] = {}

    starts_state = checkpoint_state.setdefault("starts", {})

    for start_name, start_groups in starts:
        _validate_pair_groups(start_groups, num_pairs, m)
        start_state_key = f"{group_state_key}|{start_name}"
        saved_start = starts_state.get(start_state_key)
        if saved_start is not None:
            current = _copy_groups(saved_start["current_groups"])
            current_score = float(saved_start["current_score"])
            current_stats = dict(saved_start.get("current_stats", {}))
            start_step = int(saved_start.get("step", 0))
            if "rng_state" in saved_start:
                rng.setstate(saved_start["rng_state"])
            print(
                f"[resume] {_group_name(group_key)} start={start_name} "
                f"step={start_step} current={current_score:.6f}",
                flush=True,
            )
        else:
            key = _canonical_groups(start_groups)
            if key in seen:
                continue
            seen.add(key)
            current = _copy_groups(start_groups)
            current_score, current_stats = evaluate_partition(
                current,
                blocks,
                head_dim=head_dim,
                candidate_budget=int(args.candidate_budget),
                recall_k=int(args.recall_k),
                target_cluster_size=int(args.target_cluster_size),
                k_rounding=str(args.k_rounding),
                kmeans_iters=int(args.kmeans_iters),
                seed=int(args.seed),
                max_probe_cells=int(args.max_probe_cells),
            )
            start_step = 0
            log_rows.append({
                "layer_group": _group_name(group_key),
                "start": start_name,
                "step": 0,
                "accepted": 1,
                "score": current_score,
                **current_stats,
            })
            print(
                f"[search] {_group_name(group_key)} start={start_name} score={current_score:.6f}",
                flush=True,
            )

        if saved_start is not None and "best_score" in saved_start and float(saved_start["best_score"]) > best_score:
            best_score = float(saved_start["best_score"])
            best_groups = _copy_groups(saved_start["best_groups"])
            best_stats = dict(saved_start.get("best_stats", {}))
        if current_score > best_score:
            best_score = current_score
            best_groups = _copy_groups(current)
            best_stats = dict(current_stats)

        for step in range(start_step + 1, int(args.steps) + 1):
            proposal = _propose_swap(current, rng)
            prop_score, prop_stats = evaluate_partition(
                proposal,
                blocks,
                head_dim=head_dim,
                candidate_budget=int(args.candidate_budget),
                recall_k=int(args.recall_k),
                target_cluster_size=int(args.target_cluster_size),
                k_rounding=str(args.k_rounding),
                kmeans_iters=int(args.kmeans_iters),
                seed=int(args.seed),
                max_probe_cells=int(args.max_probe_cells),
            )
            accepted = prop_score > current_score + 1e-12
            if accepted:
                current = proposal
                current_score = prop_score
                current_stats = prop_stats
                if current_score > best_score:
                    best_score = current_score
                    best_groups = _copy_groups(current)
                    best_stats = dict(current_stats)
            starts_state[start_state_key] = {
                "group_key": group_state_key,
                "start": start_name,
                "step": int(step),
                "current_groups": _copy_groups(current),
                "current_score": float(current_score),
                "current_stats": dict(current_stats),
                "best_groups": _copy_groups(best_groups) if best_groups is not None else _copy_groups(current),
                "best_score": float(best_score),
                "best_stats": dict(best_stats),
                "rng_state": rng.getstate(),
            }
            log_rows.append({
                "layer_group": _group_name(group_key),
                "start": start_name,
                "step": step,
                "accepted": int(accepted),
                "score": prop_score,
                **prop_stats,
            })
            if step % max(int(args.log_every), 1) == 0:
                print(
                    f"[search] {_group_name(group_key)} start={start_name} "
                    f"step={step} current={current_score:.6f} best={best_score:.6f}",
                    flush=True,
                )
            if int(args.checkpoint_every) > 0 and step % int(args.checkpoint_every) == 0:
                _save_checkpoint(args.checkpoint, checkpoint_state)
                if bool(args.flush_log):
                    _append_log_csv(args.out_log, log_rows)

    if best_groups is None:
        raise RuntimeError(f"partition search failed for {_group_name(group_key)}")
    completed[group_state_key] = {
        "best_groups": _copy_groups(best_groups),
        "best_score": float(best_score),
        "best_stats": dict(best_stats),
    }
    _save_checkpoint(args.checkpoint, checkpoint_state)
    print(f"[search] {_group_name(group_key)} best={best_score:.6f}", flush=True)
    return best_groups, best_stats


def write_log_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer_group",
        "start",
        "step",
        "accepted",
        "score",
        "mean_recall",
        "samples",
        "candidate_count_mean",
        "probed_cells_mean",
        "domain_means_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_log_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer_group",
        "start",
        "step",
        "accepted",
        "score",
        "mean_recall",
        "samples",
        "candidate_count_mean",
        "probed_cells_mean",
        "domain_means_json",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames != fieldnames:
                existing_rows = list(reader)
                with path.open("w", newline="", encoding="utf-8") as out_f:
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                    for existing_row in existing_rows:
                        writer.writerow({name: existing_row.get(name, "") for name in fieldnames})
                write_header = False
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    rows.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search balanced RoPE-pair partitions using two-stage IMI recall as objective.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--domain-name", type=str, default="default")
    parser.add_argument(
        "--jsonl-spec",
        action="append",
        default=[],
        help="Repeatable multi-domain input in name=path format. Objective averages recall per domain first.",
    )
    parser.add_argument("--text-field", type=str, default="auto")
    parser.add_argument("--partition", type=Path, required=True, help="Template learned partition JSON for layer groups.")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--layers", type=str, default="12")
    parser.add_argument(
        "--emit-per-layer",
        action="store_true",
        help="Optimize and emit one partition entry per layer instead of sharing source layer groups.",
    )
    parser.add_argument("--static-start", type=int, default=32)
    parser.add_argument("--static-end", type=int, default=64)
    parser.add_argument("--max-search-tokens", type=int, default=0)
    parser.add_argument("--max-q-heads-per-kv", type=int, default=0)
    parser.add_argument("--candidate-budget", type=int, default=1024)
    parser.add_argument("--recall-k", type=int, default=64)
    parser.add_argument("--target-cluster-size", type=int, default=16)
    parser.add_argument("--k-rounding", type=str, default="round", choices=["round", "floor", "ceil"])
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--random-starts", type=int, default=2)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Pickle checkpoint for resumable partition search.")
    parser.add_argument("--checkpoint-every", type=int, default=8, help="Save checkpoint every N swap steps. 0 disables periodic saves.")
    parser.add_argument("--flush-log", action="store_true", help="Append log rows to --out-log whenever a checkpoint is saved.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-probe-cells", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ann-device", type=str, default="")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-log", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    ann_device = args.ann_device or args.device
    partition = json.loads(args.partition.read_text(encoding="utf-8"))
    checkpoint_state = _load_checkpoint(args.checkpoint)
    grouped_records = collect_records(args, partition)

    if args.emit_per_layer:
        template_by_key = {
            key: _template_for_key(partition, key, emit_per_layer=True)
            for key in grouped_records
        }
    else:
        template_by_key = {
            _layer_group_key(group): dict(group)
            for group in partition.get("layer_groups", [])
        }
    optimized_by_key: Dict[Tuple[int, int, int], Tuple[List[List[int]], Dict[str, float]]] = {}
    log_rows: List[Dict[str, object]] = []

    for key, records in sorted(grouped_records.items()):
        blocks = prepare_blocks(
            records,
            ann_device=ann_device,
            recall_k=int(args.recall_k),
            max_q_heads_per_kv=int(args.max_q_heads_per_kv),
        )
        print(
            f"[prepare] {_group_name(key)} records={len(records)} blocks={len(blocks)} "
            f"queries={sum(len(block.queries) for block in blocks)}",
            flush=True,
        )
        best_groups, stats = optimize_group(
            group_key=key,
            template_group=template_by_key[key],
            blocks=blocks,
            args=args,
            rng=rng,
            log_rows=log_rows,
            checkpoint_state=checkpoint_state,
        )
        optimized_by_key[key] = (best_groups, stats)
        if bool(args.flush_log):
            _append_log_csv(args.out_log, log_rows)

    output_groups = []
    for key, group in sorted(template_by_key.items()):
        group_out = copy.deepcopy(group)
        if key in optimized_by_key:
            best_groups, stats = optimized_by_key[key]
            group_out["M"] = len(best_groups)
            group_out["subspace_parts"] = len(best_groups)
            heads = group_out.get("heads", {})
            if not isinstance(heads, dict) or not heads:
                heads = {"0": {"groups": best_groups}}
            for head_key in list(heads.keys()):
                head_obj = heads[head_key]
                if not isinstance(head_obj, dict):
                    head_obj = {}
                head_obj["groups"] = best_groups
                head_obj["metrics"] = {"recall_search": stats}
                heads[head_key] = head_obj
            group_out["heads"] = heads
        if args.emit_per_layer:
            output_groups.append(group_out)
        else:
            output_groups.append(group_out)

    output = {
        "metadata": {
            "format": "adaptive_imi_learned_partition_v1",
            "source": str(args.partition),
            "objective": "two_stage_imi_recall",
            "candidate_budget": int(args.candidate_budget),
            "recall_k": int(args.recall_k),
            "target_cluster_size": int(args.target_cluster_size),
            "k_rounding": str(args.k_rounding),
            "kmeans_iters": int(args.kmeans_iters),
            "random_starts": int(args.random_starts),
            "steps": int(args.steps),
            "seed": int(args.seed),
            "jsonl": str(args.jsonl) if args.jsonl is not None else None,
            "jsonl_specs": [(name, str(path)) for name, path in _jsonl_specs(args)],
            "layers": str(args.layers),
            "limit": int(args.limit),
            "emit_per_layer": bool(args.emit_per_layer),
        },
        "layer_groups": output_groups,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if bool(args.flush_log):
        _append_log_csv(args.out_log, log_rows)
    else:
        write_log_csv(args.out_log, log_rows)
    _save_checkpoint(args.checkpoint, checkpoint_state)
    print(f"[search] wrote {args.out_json}", flush=True)
    print(f"[search] wrote {args.out_log}", flush=True)


if __name__ == "__main__":
    main()
