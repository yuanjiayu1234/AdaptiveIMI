from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import resolve_model_path
from experiments.imi_partition.partitioning.profiling import ProcessProfiler, write_json


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

    def add(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        *,
        sequence_start: int = 0,
        sequence_total: Optional[int] = None,
    ) -> None:
        if layer_idx not in self.layer_ids:
            return
        if key_states.dim() != 4:
            return
        # HF Llama uses [batch, kv_heads, seq, head_dim] at the RoPE call.
        bsz = int(key_states.size(0))
        seq_len = int(key_states.size(2))
        sequence_start = max(int(sequence_start), 0)
        sequence_total = int(sequence_total) if sequence_total is not None else sequence_start + seq_len
        if sequence_total < sequence_start + seq_len:
            raise ValueError(
                f"collector sequence_total={sequence_total} is smaller than "
                f"sequence_start+chunk_len={sequence_start + seq_len}"
            )
        if self.current_valid_starts is None:
            valid_starts = [0] * bsz
        else:
            valid_starts = list(self.current_valid_starts)
            if len(valid_starts) != bsz:
                raise RuntimeError(f"collector valid_start count {len(valid_starts)} != batch size {bsz}")

        compact_chunks: List[torch.Tensor] = []
        for batch_idx, valid_start in enumerate(valid_starts):
            global_start = int(valid_start) + self.static_start
            global_end = sequence_total - self.static_end if self.static_end > 0 else sequence_total
            if self.max_tokens_per_prompt > 0:
                global_end = min(global_end, global_start + self.max_tokens_per_prompt)
            start = min(max(global_start - sequence_start, 0), seq_len)
            end = min(global_end - sequence_start, seq_len)
            end = max(start, end)
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
    try:
        model_type = str(AutoConfig.from_pretrained(resolve_model_path(model_path)).model_type).lower()
    except Exception:
        model_type = ""
    if model_type == "qwen3":
        return importlib.import_module("transformers.models.qwen3.modeling_qwen3")
    if model_type == "qwen2":
        return importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    if "qwen3" in probe:
        return importlib.import_module("transformers.models.qwen3.modeling_qwen3")
    if "qwen2" in probe or "qwen" in probe:
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


def _is_fp8_compressed_config(config: Any) -> bool:
    quant = getattr(config, "quantization_config", None) or {}
    if not isinstance(quant, dict):
        return False
    if str(quant.get("quant_method", "")).lower() != "compressed-tensors":
        return False
    if str(quant.get("format", "")).lower() != "float-quantized":
        return False
    for group in (quant.get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        if str(weights.get("type", "")).lower() == "float" and int(weights.get("num_bits", 0) or 0) == 8:
            return True
    return False


def _build_distributed_device_map(
    *,
    num_layers: int,
    stop_layer: int,
    num_visible_gpus: int,
) -> Dict[str, object]:
    stop_layer = max(0, min(int(stop_layer), int(num_layers) - 1))
    visible_gpus = max(1, int(num_visible_gpus))
    used_gpus = min(visible_gpus, max(1, stop_layer + 1))
    prefix_layers = stop_layer + 1
    layers_per_gpu = max(1, int(math.ceil(float(prefix_layers) / float(used_gpus))))

    device_map: Dict[str, object] = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": used_gpus - 1,
        "lm_head": used_gpus - 1,
    }
    for layer_idx in range(num_layers):
        if layer_idx <= stop_layer:
            device_map[f"model.layers.{layer_idx}"] = min(layer_idx // layers_per_gpu, used_gpus - 1)
        else:
            device_map[f"model.layers.{layer_idx}"] = "cpu"
    return device_map


def _load_model_for_collection(
    model_path: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
    hf_config,
    *,
    num_layers: int,
    stop_layer: int,
):
    model_kwargs = {"torch_dtype": dtype, "config": hf_config}
    if str(args.attn_implementation):
        model_kwargs["attn_implementation"] = str(args.attn_implementation)

    requested_device = str(args.device).strip().lower()
    if requested_device == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("--device auto requires CUDA for multi-GPU key collection")
        model_kwargs["device_map"] = _build_distributed_device_map(
            num_layers=int(num_layers),
            stop_layer=int(stop_layer),
            num_visible_gpus=torch.cuda.device_count(),
        )
        model_kwargs["low_cpu_mem_usage"] = True
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        device_map = getattr(model, "hf_device_map", None)
        if device_map is not None:
            layer_devices = {key: value for key, value in device_map.items() if key.startswith("model.layers.")}
            print(
                f"[collect] loaded model with explicit device_map entries={len(device_map)} "
                f"gpu_layers={sum(1 for value in layer_devices.values() if value != 'cpu')} "
                f"cpu_layers={sum(1 for value in layer_devices.values() if value == 'cpu')}",
                flush=True,
            )
            preview = {
                key: device_map[key]
                for key in ("model.embed_tokens", "model.rotary_emb", "model.layers.0", f"model.layers.{stop_layer}", "model.norm", "lm_head")
                if key in device_map
            }
            print(f"[collect] device_map preview={preview}", flush=True)
        return model

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(args.device)
    print(f"[collect] loaded model on device={args.device}", flush=True)
    return model


def _load_qwen_fp8_runtime(
    model_path: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
    *,
    stop_layer: int,
):
    """Load compressed-tensors FP8 Qwen models without the HF quantizer path."""
    workspace = os.environ.setdefault(
        "FLASHINFER_WORKSPACE_BASE",
        str(PROJECT_ROOT / "outputs" / "flashinfer"),
    )
    Path(workspace).mkdir(parents=True, exist_ok=True)

    try:
        from model_hub.qwen_fp8 import QwenFp8Model
    except Exception as exc:
        raise RuntimeError(
            "QwenFp8Model collection requires the AdaptiveIMI runtime dependencies "
            "(flashinfer and vLLM FP8 support)."
        ) from exc

    class CollectionQwenFp8Model(QwenFp8Model):
        def init_model(self):
            self.weight_loading = "compressed_tensors_fp8"
            stop = max(0, min(int(stop_layer), int(self.num_layers) - 1))
            if str(args.device).strip().lower() == "auto":
                visible_gpus = int(torch.cuda.device_count())
                if visible_gpus <= 0:
                    raise RuntimeError("--device auto requires at least one CUDA device")
                devices = [f"cuda:{idx}" for idx in range(visible_gpus)]
            elif str(args.device).strip().lower().startswith("cuda"):
                devices = [str(args.device)]
            else:
                raise ValueError(
                    "QwenFp8Model collection requires a CUDA device; "
                    f"got --device {args.device!r}"
                )

            used_gpus = min(len(devices), stop + 1)
            devices = devices[:used_gpus]
            layers_per_gpu = max(1, math.ceil((stop + 1) / used_gpus))
            self.device_list = list(devices)
            self.num_gpus = len(devices)
            self.layer_mapping = {}
            for layer_idx in range(self.num_layers):
                if layer_idx <= stop:
                    device_idx = min(layer_idx // layers_per_gpu, used_gpus - 1)
                    self.layer_mapping[str(layer_idx)] = devices[device_idx]
                else:
                    # Later layers are not executed for this layerwise
                    # collection pass and remain CPU-resident.
                    self.layer_mapping[str(layer_idx)] = "cpu"
            root_device = devices[0]
            self._init_model_from_safetensors(root_device)

    model = CollectionQwenFp8Model(
        model_name=str(model_path),
        max_length=int(args.max_length),
        dtype=dtype,
        device_map=str(args.device),
        model_path=str(model_path),
        parallel_mode="auto",
        gpu_memory_utilization=0.90,
    )
    print(
        "[collect] loaded QwenFp8Model runtime "
        f"devices={getattr(model, 'device_list', None)} "
        f"layers={getattr(model, 'layer_mapping', None)}",
        flush=True,
    )
    return model


def _model_input_device(model, fallback: str) -> torch.device:
    try:
        embedding = model.get_input_embeddings()
        weight = getattr(embedding, "weight", None)
        if weight is not None:
            return torch.device(weight.device)
    except Exception:
        pass
    return torch.device("cuda:0" if str(fallback).strip().lower() == "auto" else fallback)


def _limit_model_to_layers(model, stop_layer: int) -> Optional[torch.nn.ModuleList]:
    layers = getattr(model.model, "layers", None)
    if layers is None:
        return None
    if stop_layer >= len(layers) - 1:
        return None
    original_layers = layers
    model.model.layers = torch.nn.ModuleList(list(layers[: stop_layer + 1]))
    return original_layers


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


class _CollectionOnlyCache:
    """Minimal cache state required by the existing chunked prefill path."""

    def __init__(self, valid_starts: Sequence[int]) -> None:
        self.valid_start_list = [int(value) for value in valid_starts]
        self.context = 0
        self.prefill_kv_workspace_available = None

    def prefill_submit_kv_chunk(self, *args, **kwargs) -> None:
        return None

    def prefill_update_kv_cache(self, *args, **kwargs) -> None:
        return None

    def sync(self, *args, **kwargs) -> None:
        return None


def _collect_qwen_fp8_runtime(
    model,
    tokenizer,
    args: argparse.Namespace,
    collector: RopeKeyCollector,
) -> tuple[int, str]:
    """Collect keys through QwenFp8Model's existing chunked prefill."""
    if int(args.collect_batch_size) != 1:
        raise ValueError(
            "QwenFp8Model collection currently requires --collect-batch-size 1; "
            "use the default COLLECT_BATCH_SIZE=1."
        )
    root_device = str(model.layers[0].device)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device(root_device))
    original_prefill_chunk_size = int(model.prefill_chunk_size)
    original_prefill_attn_chunk_size = int(model.prefill_attn_chunk_size)
    had_prefill_bsz = hasattr(model, "prefill_bsz")
    original_prefill_bsz = getattr(model, "prefill_bsz", None)
    chunk_size = int(args.mlp_chunk_size)
    if chunk_size <= 0:
        chunk_size = 8192
    model.prefill_chunk_size = chunk_size
    model.prefill_attn_chunk_size = chunk_size
    model.prefill_bsz = 1
    prompt_count = 0
    stop_layer = max(collector.layer_ids)

    original_num_layers = int(model.num_layers)
    original_layers = model.layers
    original_apply_rope = model.apply_rotary_pos_emb
    original_layer_prefill_chunked = model.layer_prefill_chunked
    original_layer_mapping = model.layer_mapping
    had_attention_type = hasattr(model, "attention_type")
    original_attention_type = getattr(model, "attention_type", None)
    had_collector_active_layer = hasattr(model, "_collector_active_layer")
    original_collector_active_layer = getattr(model, "_collector_active_layer", None)
    had_collector = hasattr(model, "_collector")
    original_collector = getattr(model, "_collector", None)
    had_collector_sequence_total = hasattr(model, "_collector_sequence_total")
    original_collector_sequence_total = getattr(model, "_collector_sequence_total", None)
    had_kv_cache = hasattr(model, "kv_cache")
    original_kv_cache = getattr(model, "kv_cache", None)
    original_prefill_workspace = model._prefill_prefix_kv_workspace

    model.num_layers = stop_layer + 1
    model.layers = model.layers[: stop_layer + 1]
    model.layer_mapping = {
        str(layer_idx): original_layer_mapping[str(layer_idx)]
        for layer_idx in range(stop_layer + 1)
    }
    model._prefill_prefix_kv_workspace = {}
    # The runtime's prefill path only uses this for the optional debug branch,
    # but setting it makes the collector independent of that environment flag.
    model.attention_type = "Full_Flash_Attn"

    def wrapped_apply_rope(query_states, key_states, position_ids):
        query_embed, key_embed = original_apply_rope(
            query_states,
            key_states,
            position_ids,
        )
        active_layer = getattr(model, "_collector_active_layer", None)
        if active_layer is not None and active_layer in collector.layer_ids:
            if position_ids.dim() != 2 or int(position_ids.size(0)) != 1:
                raise RuntimeError(
                    "QwenFp8Model key collection currently requires batch size 1"
                )
            sequence_start = int(position_ids[0, 0].item()) if position_ids.numel() else 0
            bsz, seq_len, _ = key_embed.shape
            key_for_collection = key_embed.view(
                bsz,
                seq_len,
                model.num_key_value_heads,
                model.head_dim,
            ).transpose(1, 2).contiguous()
            collector.add(
                int(active_layer),
                key_for_collection,
                sequence_start=sequence_start,
                sequence_total=int(model._collector_sequence_total),
            )
        return query_embed, key_embed

    def wrapped_layer_prefill_chunked(
        layer_idx,
        start_bdx,
        hidden_cpu,
        output_cpu,
        seq_len,
        chunk_size_arg,
    ):
        model._collector_active_layer = int(layer_idx)
        try:
            return original_layer_prefill_chunked(
                layer_idx,
                start_bdx,
                hidden_cpu,
                output_cpu,
                seq_len,
                chunk_size_arg,
            )
        finally:
            model._collector_active_layer = None

    model.apply_rotary_pos_emb = wrapped_apply_rope
    model.layer_prefill_chunked = wrapped_layer_prefill_chunked

    try:
        records = _progress(
            _iter_prompt_records(args),
            total=int(args.limit) if int(args.limit) > 0 else None,
            desc="collect keys",
            enabled=bool(args.progress),
        )
        for item in records:
            prompt = _build_collection_prompt(item, args)
            encoded = tokenizer(
                [prompt],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding=False,
            )
            input_ids = encoded["input_ids"].to(root_device)
            attention_mask = encoded.get("attention_mask")
            valid_starts = None
            if attention_mask is not None:
                valid_starts = (
                    attention_mask.shape[1] - torch.sum(attention_mask, dim=-1).detach().cpu()
                ).tolist()
            collector.set_valid_starts(valid_starts)

            model._collector = collector
            model._collector_sequence_total = int(input_ids.size(1))
            model.kv_cache = _CollectionOnlyCache(valid_starts or [0])
            model._prefill_prefix_kv_workspace.clear()
            try:
                model.prefill_forward_chunked(input_ids)
            finally:
                model.kv_cache = None
                model._collector = None
                model._collector_sequence_total = 0
                model._prefill_prefix_kv_workspace.clear()

            prompt_count += 1
            print(
                f"[collect] prompts={prompt_count} batch=1 tokens={int(input_ids.size(1))}",
                flush=True,
            )
    finally:
        collector.set_valid_starts(None)
        model.apply_rotary_pos_emb = original_apply_rope
        model.layer_prefill_chunked = original_layer_prefill_chunked
        model.layers = original_layers
        model.num_layers = original_num_layers
        model.layer_mapping = original_layer_mapping
        model.prefill_chunk_size = original_prefill_chunk_size
        model.prefill_attn_chunk_size = original_prefill_attn_chunk_size
        if had_prefill_bsz:
            model.prefill_bsz = original_prefill_bsz
        else:
            try:
                delattr(model, "prefill_bsz")
            except AttributeError:
                pass
        if had_collector_active_layer:
            model._collector_active_layer = original_collector_active_layer
        else:
            try:
                delattr(model, "_collector_active_layer")
            except AttributeError:
                pass
        if had_attention_type:
            model.attention_type = original_attention_type
        else:
            try:
                delattr(model, "attention_type")
            except AttributeError:
                pass
        if had_collector:
            model._collector = original_collector
        else:
            try:
                delattr(model, "_collector")
            except AttributeError:
                pass
        if had_collector_sequence_total:
            model._collector_sequence_total = original_collector_sequence_total
        else:
            try:
                delattr(model, "_collector_sequence_total")
            except AttributeError:
                pass
        if had_kv_cache:
            model.kv_cache = original_kv_cache
        else:
            try:
                delattr(model, "kv_cache")
            except AttributeError:
                pass
        model._prefill_prefix_kv_workspace = original_prefill_workspace
    return prompt_count, root_device


def _runtime_device_metadata(model) -> Dict[str, Any]:
    layer_mapping = getattr(model, "layer_mapping", None)
    if not layer_mapping:
        return {}
    device_to_layers: Dict[str, List[int]] = {}
    for layer, device in layer_mapping.items():
        device_to_layers.setdefault(str(device), []).append(int(layer))
    return {
        "runtime_device_to_layers": {
            device: sorted(layers)
            for device, layers in device_to_layers.items()
        },
        "runtime_layer_to_device": {
            str(layer): str(device)
            for layer, device in layer_mapping.items()
        },
    }


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
    parser.add_argument("--static-start", type=int, default=0)
    parser.add_argument("--static-end", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Use cuda:x for single-GPU loading, or auto for multi-GPU placement.",
    )
    parser.add_argument("--attn-implementation", type=str, default="", help="Optional HF attention backend, e.g. flash_attention_2 or sdpa.")
    parser.add_argument("--mlp-chunk-size", type=int, default=0, help="Chunk MLP along sequence dimension to reduce 128k prefill peak memory.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile-out", type=Path, default=None, help="Optional JSON path for collection time and peak-memory metrics.")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable tqdm progress bars.")
    parser.set_defaults(progress=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiler = ProcessProfiler(enabled=args.profile_out is not None)
    profiler.start()
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
    num_layers = int(hf_config.num_hidden_layers)
    layer_ids = _parse_layers(args.layers, num_layers)
    stop_layer = max(layer_ids)
    collector = RopeKeyCollector(
        layer_ids=layer_ids,
        static_start=args.static_start,
        static_end=args.static_end,
        max_tokens_per_prompt=args.max_tokens_per_prompt,
        sample_tokens=args.sample_tokens,
        sample_seed=args.sample_seed,
    )

    native_qwen_fp8 = (
        str(getattr(hf_config, "model_type", "")).lower() == "qwen2"
        and _is_fp8_compressed_config(hf_config)
    )
    runtime_metadata: Dict[str, Any] = {}

    if native_qwen_fp8:
        model = _load_qwen_fp8_runtime(
            model_path,
            args,
            dtype,
            stop_layer=stop_layer,
        )
        collection_backend = "qwen_fp8_runtime"
        input_device = torch.device(str(model.layers[0].device))
        with torch.inference_mode():
            prompt_count, input_device_name = _collect_qwen_fp8_runtime(
                model,
                tokenizer,
                args,
                collector,
            )
        input_device = torch.device(input_device_name)
        runtime_metadata = _runtime_device_metadata(model)
    else:
        model = _load_model_for_collection(
            model_path,
            args,
            dtype,
            hf_config,
            num_layers=num_layers,
            stop_layer=stop_layer,
        )
        model.eval()
        if str(args.device).strip().lower() == "auto":
            original_layers = _limit_model_to_layers(model, stop_layer)
            if original_layers is not None:
                print(f"[collect] limiting forward to layers 0..{stop_layer}", flush=True)
        else:
            original_layers = None
        input_device = _model_input_device(model, str(args.device))
        print(f"[collect] input_device={input_device}", flush=True)
        restore_mlp_chunking = _patch_mlp_chunking(model, int(args.mlp_chunk_size))

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
                    input_ids = encoded["input_ids"].to(input_device)
                    attention_mask = encoded.get("attention_mask")
                    valid_starts = None
                    if attention_mask is not None:
                        valid_starts = (
                            attention_mask.shape[1] - torch.sum(attention_mask, dim=-1).detach().cpu()
                        ).tolist()
                        attention_mask = attention_mask.to(input_device)
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
            if original_layers is not None:
                model.model.layers = original_layers
            llama_modeling.apply_rotary_pos_emb = original_apply_rope
            for attention, original_forward in original_forwards:
                attention.forward = original_forward
        collection_backend = "transformers"

    keys = collector.to_lhtd()
    profiling = profiler.stop()
    output = {
        "keys": keys,
        "metadata": {
            "model": args.model,
            "model_path": model_path,
            "collection_backend": collection_backend,
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
            "device_arg": str(args.device),
            "input_device": str(input_device),
            "hf_device_map": getattr(model, "hf_device_map", None),
            **runtime_metadata,
            "tokenizer_padding_side": str(getattr(tokenizer, "padding_side", "")),
            "tokenizer_pad_token_id": None if tokenizer.pad_token_id is None else int(tokenizer.pad_token_id),
            "seen_tokens": {str(layer): int(collector.seen_tokens.get(layer, 0)) for layer in layer_ids},
            "shape": list(keys.shape),
            "profiling": profiling,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(output, tmp_out)
    tmp_out.replace(args.out)
    print(f"[collect] wrote {args.out} shape={tuple(keys.shape)}", flush=True)
    if args.profile_out is not None:
        write_json(
            args.profile_out,
            {
                "format": "adaptive_imi_collection_profile_v1",
                "stage": "key_collection",
                "output": str(args.out),
                "profiling": profiling,
                "output_metadata": output["metadata"],
            },
        )
        print(f"[collect] wrote profile {args.profile_out}", flush=True)


if __name__ == "__main__":
    main()
