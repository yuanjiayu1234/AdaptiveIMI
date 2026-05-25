from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import resolve_model_path


class RopeKeyCollector:
    def __init__(
        self,
        *,
        layer_ids: List[int],
        static_start: int,
        static_end: int,
        max_tokens_per_prompt: int,
    ) -> None:
        self.layer_ids = set(layer_ids)
        self.static_start = max(int(static_start), 0)
        self.static_end = max(int(static_end), 0)
        self.max_tokens_per_prompt = int(max_tokens_per_prompt)
        self.active_layer: Optional[int] = None
        self.storage: Dict[int, List[torch.Tensor]] = {layer_id: [] for layer_id in layer_ids}

    def add(self, layer_idx: int, key_states: torch.Tensor) -> None:
        if layer_idx not in self.layer_ids:
            return
        if key_states.dim() != 4:
            return
        # HF Llama uses [batch, kv_heads, seq, head_dim] at the RoPE call.
        seq_len = int(key_states.size(2))
        start = min(self.static_start, seq_len)
        end = seq_len - self.static_end if self.static_end > 0 else seq_len
        end = max(start, end)
        if self.max_tokens_per_prompt > 0:
            end = min(end, start + self.max_tokens_per_prompt)
        if end <= start:
            return
        sliced = key_states[:, :, start:end, :].detach()
        # Save as [kv_heads, tokens, head_dim].
        compact = sliced.permute(1, 0, 2, 3).reshape(
            sliced.size(1),
            sliced.size(0) * sliced.size(2),
            sliced.size(3),
        )
        self.storage[layer_idx].append(compact.to(device="cpu", dtype=torch.float16))

    def to_lhtd(self) -> torch.Tensor:
        layers: List[torch.Tensor] = []
        for layer_idx in sorted(self.storage):
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


def _read_hf_dataset(dataset_name: str, subset: Optional[str], split: str, text_field: str, limit: int) -> Iterator[str]:
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


def _iter_prompts(args: argparse.Namespace) -> Iterable[str]:
    if args.jsonl is not None:
        yield from _read_jsonl(args.jsonl, args.text_field, args.limit)
        return
    if args.hf_dataset:
        yield from _read_hf_dataset(args.hf_dataset, args.hf_subset, args.split, args.text_field, args.limit)
        return
    raise ValueError("provide either --jsonl or --hf-dataset")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RoPE-applied Llama keys for IMI partition experiments.")
    parser.add_argument("--model", type=str, default="llama-3-8b-1048k")
    parser.add_argument("--jsonl", type=Path, default=None, help="Local JSONL prompt source.")
    parser.add_argument("--hf-dataset", type=str, default="", help="Optional HF dataset name, e.g. a LongBenchV2 dataset id.")
    parser.add_argument("--hf-subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-tokens-per-prompt", type=int, default=0)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--static-start", type=int, default=4)
    parser.add_argument("--static-end", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model_path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    layer_ids = _parse_layers(args.layers, num_layers)
    collector = RopeKeyCollector(
        layer_ids=layer_ids,
        static_start=args.static_start,
        static_end=args.static_end,
        max_tokens_per_prompt=args.max_tokens_per_prompt,
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
            for prompt in _iter_prompts(args):
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
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                prompt_count += 1
                print(f"[collect] prompt={prompt_count} tokens={int(input_ids.size(1))}", flush=True)
    finally:
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
            "shape": list(keys.shape),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    print(f"[collect] wrote {args.out} shape={tuple(keys.shape)}", flush=True)


if __name__ == "__main__":
    main()
