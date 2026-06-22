from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from config import resolve_model_path


def _progress(iterable, *, total: Optional[int], desc: str, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


class RopeKeyCollector:
    def __init__(
        self,
        *,
        layer_ids: List[int],
        static_start: int,
        static_end: int,
        max_tokens_per_prompt: int,
        sample_tokens: int,
        sample_seed: int,
    ) -> None:
        self.layer_ids = set(layer_ids)
        self.static_start = max(int(static_start), 0)
        self.static_end = max(int(static_end), 0)
        self.max_tokens_per_prompt = int(max_tokens_per_prompt)
        self.sample_tokens = max(int(sample_tokens), 0)
        self.active_layer: Optional[int] = None
        self.current_valid_starts: Optional[List[int]] = None
        self.storage: Dict[int, List[torch.Tensor]] = {layer_id: [] for layer_id in layer_ids}
        self.reservoirs: Dict[int, Optional[torch.Tensor]] = {layer_id: None for layer_id in layer_ids}
        self.seen_tokens: Dict[int, int] = {layer_id: 0 for layer_id in layer_ids}
        self.generators: Dict[int, torch.Generator] = {}
        for layer_id in layer_ids:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(sample_seed) + int(layer_id) * 1009)
            self.generators[layer_id] = gen

    def set_valid_starts(self, valid_starts: Optional[Sequence[int]]) -> None:
        if valid_starts is None:
            self.current_valid_starts = None
        else:
            self.current_valid_starts = [max(int(value), 0) for value in valid_starts]

    def add(self, layer_idx: int, key_states: torch.Tensor) -> None:
        if layer_idx not in self.layer_ids:
            return
        if key_states.dim() != 4:
            return
        # HF Llama uses [batch, kv_heads, seq, head_dim] at the RoPE call.
        bsz = int(key_states.size(0))
        seq_len = int(key_states.size(2))
        if self.current_valid_starts is None:
            valid_starts = [0] * bsz
        else:
            valid_starts = list(self.current_valid_starts)
            if len(valid_starts) != bsz:
                raise RuntimeError(f"collector valid_start count {len(valid_starts)} != batch size {bsz}")

        compact_chunks: List[torch.Tensor] = []
        for batch_idx, valid_start in enumerate(valid_starts):
            start = min(int(valid_start) + self.static_start, seq_len)
            end = seq_len - self.static_end if self.static_end > 0 else seq_len
            end = max(start, end)
            if self.max_tokens_per_prompt > 0:
                end = min(end, start + self.max_tokens_per_prompt)
            if end <= start:
                continue
            sliced = key_states[batch_idx : batch_idx + 1, :, start:end, :].detach()
            compact_chunks.append(
                sliced.permute(1, 0, 2, 3).reshape(
                    sliced.size(1),
                    sliced.size(2),
                    sliced.size(3),
                )
            )
        if not compact_chunks:
            return
        # Save as [kv_heads, tokens, head_dim].
        compact = torch.cat(compact_chunks, dim=1)
        compact = compact.to(device="cpu", dtype=torch.float16)
        if self.sample_tokens > 0:
            self._add_sampled(layer_idx, compact)
        else:
            self.storage[layer_idx].append(compact)

    def _dedupe_last(self, positions: torch.Tensor, source_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if int(positions.numel()) <= 1:
            return positions, source_indices
        try:
            order = torch.argsort(positions, stable=True)
        except TypeError:
            order = torch.argsort(positions)
        sorted_positions = positions.index_select(0, order)
        sorted_sources = source_indices.index_select(0, order)
        keep = torch.ones_like(sorted_positions, dtype=torch.bool)
        keep[:-1] = sorted_positions[:-1] != sorted_positions[1:]
        return sorted_positions[keep], sorted_sources[keep]

    def _add_sampled(self, layer_idx: int, compact: torch.Tensor) -> None:
        capacity = int(self.sample_tokens)
        if capacity <= 0:
            self.storage[layer_idx].append(compact)
            return

        heads, tokens, dim = compact.shape
        if tokens <= 0:
            return

        reservoir = self.reservoirs[layer_idx]
        if reservoir is None:
            reservoir = torch.empty((heads, capacity, dim), dtype=torch.float16, device="cpu")
            self.reservoirs[layer_idx] = reservoir

        seen = int(self.seen_tokens[layer_idx])
        cursor = 0
        fill = min(max(capacity - seen, 0), int(tokens))
        if fill > 0:
            reservoir[:, seen : seen + fill, :].copy_(compact[:, :fill, :])
            seen += fill
            cursor += fill

        remaining = int(tokens) - cursor
        if remaining > 0:
            gen = self.generators[layer_idx]
            global_indices = torch.arange(seen, seen + remaining, dtype=torch.float64)
            draws = torch.floor(torch.rand((remaining,), generator=gen, dtype=torch.float64) * (global_indices + 1.0)).to(torch.long)
            keep = draws < capacity
            if bool(keep.any().item()):
                source_indices = torch.nonzero(keep, as_tuple=False).flatten().to(torch.long) + int(cursor)
                positions = draws.index_select(0, torch.nonzero(keep, as_tuple=False).flatten())
                positions, source_indices = self._dedupe_last(positions, source_indices)
                reservoir[:, positions, :].copy_(compact.index_select(1, source_indices))
            seen += remaining

        self.seen_tokens[layer_idx] = seen

    def to_lhtd(self) -> torch.Tensor:
        layers: List[torch.Tensor] = []
        for layer_idx in sorted(self.storage):
            if self.sample_tokens > 0:
                reservoir = self.reservoirs[layer_idx]
                kept = min(int(self.seen_tokens[layer_idx]), int(self.sample_tokens))
                if reservoir is None or kept <= 0:
                    raise RuntimeError(f"no keys collected for layer {layer_idx}")
                layers.append(reservoir[:, :kept, :].clone())
            else:
                chunks = self.storage[layer_idx]
                if not chunks:
                    raise RuntimeError(f"no keys collected for layer {layer_idx}")
                layers.append(torch.cat(chunks, dim=1))
        return torch.stack(layers, dim=0).contiguous()


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


def _read_jsonl_records(path: Path, text_field: str, limit: int) -> Iterator[Dict[str, Any]]:
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
                copied = dict(item)
                copied["_collector_text"] = text
                count += 1
                yield copied


def _read_hf_dataset_records(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    text_field: str,
    limit: int,
) -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset

    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    count = 0
    for item in dataset:
        if limit > 0 and count >= limit:
            break
        text = _extract_text(item, text_field)
        if text:
            count += 1
            copied = dict(item)
            copied["_collector_text"] = text
            yield copied


def _extract_text(item, text_field: str) -> str:
    if text_field != "auto":
        value = item.get(text_field, "")
        return str(value) if value is not None else ""
    for key in ("input", "prompt", "context", "text", "question"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _iter_prompt_records(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.jsonl is not None:
        yield from _read_jsonl_records(args.jsonl, args.text_field, args.limit)
        return
    if args.hf_dataset:
        yield from _read_hf_dataset_records(args.hf_dataset, args.hf_subset, args.split, args.text_field, args.limit)
        return
    raise ValueError("provide either --jsonl or --hf-dataset")


def _fix_begin_of_text(prompt: str) -> str:
    if prompt.count("<|begin_of_text|>") > 1:
        parts = prompt.split("<|begin_of_text|>")
        return "<|begin_of_text|>" + parts[-1]
    return prompt


def _build_collection_prompt(item: Dict[str, Any], args: argparse.Namespace) -> str:
    prompt = str(item.get("_collector_text") or "")
    prompt = _fix_begin_of_text(prompt)
    answer_prefix = item.get(str(args.answer_prefix_field), "")
    if answer_prefix:
        prompt += str(answer_prefix)
    return prompt


def _batched(items: Iterable[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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


def _apply_runtime_rope_config(config, *, model_path: str, model_name: str, max_length: int):
    """Mirror the runtime Qwen3 YaRN extension used by model_hub/qwen.py."""
    probe = f"{model_name} {model_path} {getattr(config, 'model_type', '')}".lower()
    if "qwen3" not in probe or int(max_length) <= 32768:
        return config

    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict) or not rope_scaling:
        original_max = 32768
        factor = float(max(float(max_length) / float(original_max), 4.0))
        rope_scaling = {
            "rope_type": "yarn",
            "type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": original_max,
        }
        config.rope_scaling = rope_scaling
    elif rope_scaling.get("rope_type") is None and rope_scaling.get("type") is not None:
        rope_scaling["rope_type"] = rope_scaling["type"]

    if int(getattr(config, "max_position_embeddings", 0) or 0) < int(max_length):
        config.max_position_embeddings = int(max_length)
    print(
        "[collect] applied qwen3 runtime rope config "
        f"max_position_embeddings={getattr(config, 'max_position_embeddings', None)} "
        f"rope_scaling={getattr(config, 'rope_scaling', None)}",
        flush=True,
    )
    return config


def _patch_mlp_chunking(model, chunk_size: int):
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        return lambda: None

    original_forwards = []
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        required = ("gate_proj", "up_proj", "down_proj", "act_fn")
        if not all(hasattr(mlp, name) for name in required):
            continue
        original_forward = mlp.forward
        original_forwards.append((mlp, original_forward))

        def make_forward(module):
            def chunked_forward(x):
                if x.dim() < 3 or int(x.size(-2)) <= chunk_size:
                    return module.down_proj(module.act_fn(module.gate_proj(x)) * module.up_proj(x))
                chunks = []
                for start in range(0, int(x.size(-2)), chunk_size):
                    part = x[..., start : start + chunk_size, :]
                    chunks.append(module.down_proj(module.act_fn(module.gate_proj(part)) * module.up_proj(part)))
                return torch.cat(chunks, dim=-2)

            return chunked_forward

        mlp.forward = make_forward(mlp)

    def restore() -> None:
        for mlp, original_forward in original_forwards:
            mlp.forward = original_forward

    print(f"[collect] enabled MLP sequence chunking chunk_size={chunk_size} layers={len(original_forwards)}", flush=True)
    return restore


def _forward_last_logits(model, **kwargs):
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError:
        pass
    try:
        return model(**kwargs, num_logits_to_keep=1)
    except TypeError:
        pass
    raise RuntimeError(
        "This transformers model does not accept logits_to_keep/num_logits_to_keep; "
        "refusing to compute full-sequence vocabulary logits during key collection."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect online-prompt RoPE-applied keys for AdaptiveIMI partition training."
    )
    parser.add_argument("--model", type=str, default="llama-3-8b-1048k")
    parser.add_argument("--jsonl", type=Path, default=None, help="Local JSONL prompt source.")
    parser.add_argument("--hf-dataset", type=str, default="", help="Optional HF dataset name, e.g. a LongBenchV2 dataset id.")
    parser.add_argument("--hf-subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="auto")
    parser.add_argument("--answer-prefix-field", type=str, default="answer_prefix")
    parser.add_argument("--collect-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-tokens-per-prompt", type=int, default=0)
    parser.add_argument("--sample-tokens", type=int, default=0, help="Reservoir-sample at most this many collected tokens per layer; <=0 keeps all keys.")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--static-start", type=int, default=4)
    parser.add_argument("--static-end", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn-implementation", type=str, default="", help="Optional HF attention backend, e.g. flash_attention_2 or sdpa.")
    parser.add_argument("--mlp-chunk-size", type=int, default=0, help="Chunk MLP along sequence dimension to reduce 128k prefill peak memory.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable tqdm progress bars.")
    parser.set_defaults(progress=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model_path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "model_max_length", None) is None or int(getattr(tokenizer, "model_max_length", 0)) < int(args.max_length):
        tokenizer.model_max_length = int(args.max_length)
    hf_config = AutoConfig.from_pretrained(model_path)
    hf_config = _apply_runtime_rope_config(
        hf_config,
        model_path=model_path,
        model_name=str(args.model),
        max_length=int(args.max_length),
    )
    model_kwargs = {"torch_dtype": dtype, "config": hf_config}
    if str(args.attn_implementation):
        model_kwargs["attn_implementation"] = str(args.attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    ).to(args.device)
    model.eval()
    restore_mlp_chunking = _patch_mlp_chunking(model, int(args.mlp_chunk_size))

    num_layers = int(model.config.num_hidden_layers)
    layer_ids = _parse_layers(args.layers, num_layers)
    collector = RopeKeyCollector(
        layer_ids=layer_ids,
        static_start=args.static_start,
        static_end=args.static_end,
        max_tokens_per_prompt=args.max_tokens_per_prompt,
        sample_tokens=args.sample_tokens,
        sample_seed=args.sample_seed,
    )

    llama_modeling = _resolve_modeling_module(model_path, args.model)

    original_apply_rope = llama_modeling.apply_rotary_pos_emb
    original_forwards = []

    def wrapped_apply_rotary_pos_emb(*rope_args, **rope_kwargs):
        q_embed, k_embed = original_apply_rope(*rope_args, **rope_kwargs)
        if collector.active_layer is not None:
            collector.add(collector.active_layer, k_embed)
        return q_embed, k_embed

    llama_modeling.apply_rotary_pos_emb = wrapped_apply_rotary_pos_emb

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

    prompt_count = 0
    try:
        with torch.inference_mode():
            prompt_total = int(args.limit) if int(args.limit) > 0 else None
            records = _progress(
                _iter_prompt_records(args),
                total=prompt_total,
                desc="collect keys",
                enabled=bool(args.progress),
            )
            for batch in _batched(records, max(int(args.collect_batch_size), 1)):
                prompts = [_build_collection_prompt(item, args) for item in batch]
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                    padding=len(prompts) > 1,
                )
                input_ids = encoded["input_ids"].to(args.device)
                attention_mask = encoded.get("attention_mask")
                valid_starts = None
                if attention_mask is not None:
                    valid_starts = (
                        attention_mask.shape[1] - torch.sum(attention_mask, dim=-1).detach().cpu()
                    ).tolist()
                    attention_mask = attention_mask.to(args.device)
                collector.set_valid_starts(valid_starts)
                _forward_last_logits(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                prompt_count += len(prompts)
                print(
                    f"[collect] prompts={prompt_count - len(prompts) + 1}-{prompt_count} "
                    f"batch={len(prompts)} tokens={int(input_ids.size(1))}",
                    flush=True,
                )
    finally:
        collector.set_valid_starts(None)
        restore_mlp_chunking()
        llama_modeling.apply_rotary_pos_emb = original_apply_rope
        for attention, original_forward in original_forwards:
            attention.forward = original_forward

    keys = collector.to_lhtd()
    output = {
        "keys": keys,
        "metadata": {
            "model": args.model,
            "model_path": model_path,
            "layout": "lhtd",
            "layers": layer_ids,
            "prompt_count": prompt_count,
            "static_start": int(args.static_start),
            "static_end": int(args.static_end),
            "max_length": int(args.max_length),
            "max_tokens_per_prompt": int(args.max_tokens_per_prompt),
            "sample_tokens": int(args.sample_tokens),
            "sample_seed": int(args.sample_seed),
            "prompt_construction": "fix_begin_of_text_plus_answer_prefix",
            "answer_prefix_field": str(args.answer_prefix_field),
            "collect_batch_size": int(args.collect_batch_size),
            "tokenizer_padding_side": str(getattr(tokenizer, "padding_side", "")),
            "tokenizer_pad_token_id": None if tokenizer.pad_token_id is None else int(tokenizer.pad_token_id),
            "seen_tokens": {str(layer): int(collector.seen_tokens.get(layer, 0)) for layer in layer_ids},
            "shape": list(keys.shape),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(output, tmp_out)
    tmp_out.replace(args.out)
    print(f"[collect] wrote {args.out} shape={tuple(keys.shape)}", flush=True)


if __name__ == "__main__":
    main()
