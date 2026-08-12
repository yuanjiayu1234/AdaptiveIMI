#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

# Keep repeated 128K samples from fragmenting the CUDA caching allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import generate_config, resolve_model_path  # noqa: E402


DEFAULT_DATA_PATH = "/home/yjy/LongBench/data.json"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUT_PATH = PROJECT_ROOT / "benchmark" / "Longbenchv2" / "results" / "llama3_adaptiveimi.jsonl"
HF_CACHE_ROOT = Path(os.getenv("HF_HOME", "/root/.cache/huggingface")) / "hub"

PROMPT_TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LongBench v2 multiple-choice evaluation with AdaptiveIMI.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--config_model_name", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--max_model_len", type=int, default=131072)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--sample_indices",
        type=str,
        default=None,
        help="Comma-separated absolute dataset indices, e.g. 2,3,5,7,10.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--parallel_mode", "--parallel-mode", type=str, default="auto", choices=("auto", "single", "layer", "tp"))
    parser.add_argument("--weight_loading", "--weight-loading", type=str, default="hf", choices=("hf", "auto", "accelerate_meta", "streaming_safetensors"))
    parser.add_argument("--gpu_memory_utilization", "--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", type=str, default="bf16", choices=("fp16", "bf16"))
    parser.add_argument(
        "--attn_type",
        type=str,
        default="AdaptiveIMI",
        choices=("AdaptiveIMI", "Full_Flash_Attn", "Full_Flash_Attn_Offload"),
    )
    parser.add_argument("--retrieval_budget", type=float, default=None)
    parser.add_argument("--cache_ratio", type=float, default=None)
    parser.add_argument("--gpu_only", action="store_true", help="Use GPU-only hybrid index build policy.")
    parser.add_argument(
        "--subspace_parts",
        type=int,
        default=None,
        help="Compatibility option; current AdaptiveIMI runtime supports only 8-way learned partitions.",
    )
    parser.add_argument("--imi_engine", type=str, default=None, choices=(None, "hybrid_8way_v2"))
    parser.add_argument("--partition_path", type=str, default=os.getenv("IMI_PARTITION_PATH"))
    parser.add_argument("--hybrid_policy", type=str, default=None)
    parser.add_argument("--hybrid_start", type=str, default="cpu")
    parser.add_argument("--hybrid_gpu_every", type=int, default=0)
    parser.add_argument("--decode_search_mode", type=str, default=None, choices=("flat", "subspace_rerank"))
    parser.add_argument("--candidate_clusters", type=int, default=None)
    parser.add_argument("--token_rerank_budget_ratio", type=float, default=None)
    parser.add_argument("--prefill_bsz", type=int, default=1)
    parser.add_argument("--prefill_chunk_size", "--prefill-chunk-size", type=int, default=None)
    parser.add_argument("--chat_template", action="store_true", default=True)
    parser.add_argument("--no_chat_template", action="store_false", dest="chat_template")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_context_chars", type=int, default=1000)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise TypeError(f"Unsupported LongBench data type: {type(data).__name__}")


def render_prompt(item: dict[str, Any]) -> str:
    return (
        PROMPT_TEMPLATE.replace("$DOC$", str(item["context"]).strip())
        .replace("$Q$", str(item["question"]).strip())
        .replace("$C_A$", str(item["choice_A"]).strip())
        .replace("$C_B$", str(item["choice_B"]).strip())
        .replace("$C_C$", str(item["choice_C"]).strip())
        .replace("$C_D$", str(item["choice_D"]).strip())
    )


def apply_chat_template(tokenizer, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_with_middle_truncation(tokenizer, prompt: str, max_input_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > max_input_tokens:
        keep_head = max_input_tokens // 2
        keep_tail = max_input_tokens - keep_head
        input_ids = input_ids[:keep_head] + input_ids[-keep_tail:]
    ids = torch.tensor([input_ids], dtype=torch.long)
    return ids, torch.ones_like(ids)


def extract_answer(text: str) -> str | None:
    normalized = text.replace("*", "")
    if "</think>" in normalized:
        normalized = normalized.rsplit("</think>", 1)[-1]
    elif normalized.lstrip().startswith("<think>"):
        return None
    patterns = (
        r"The correct answer is\s*\(([A-D])\)",
        r"The correct answer is\s*([A-D])",
        r"answer is\s*\(([A-D])\)",
        r"answer is\s*([A-D])",
        r"\(([A-D])\)",
        r"\b([A-D])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def completed_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = row.get("_id")
            if item_id is not None:
                done.add(str(item_id))
    return done


def cleanup_after_sample(llm: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    kv_cache = getattr(llm, "kv_cache", None)
    if kv_cache is not None:
        try:
            cleanup = getattr(kv_cache, "cleanup", None)
            if callable(cleanup):
                cleanup()
        except Exception as exc:
            print(f"Warning: kv_cache cleanup failed: {exc}", file=sys.stderr, flush=True)
        finally:
            llm.kv_cache = None

    clear_prefill_workspace = getattr(llm, "_clear_prefill_prefix_kv_workspace", None)
    if callable(clear_prefill_workspace):
        try:
            clear_prefill_workspace()
        except Exception as exc:
            print(f"Warning: prefill workspace cleanup failed: {exc}", file=sys.stderr, flush=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
        torch.cuda.synchronize()


def resolve_local_hf_snapshot(model_name: str) -> str:
    resolved = resolve_model_path(model_name)
    if resolved and os.path.exists(resolved):
        return resolved
    if os.path.exists(model_name) or "/" not in model_name:
        return model_name

    org, repo = model_name.split("/", 1)
    cache_dir = HF_CACHE_ROOT / f"models--{org}--{repo}"
    snapshots_dir = cache_dir / "snapshots"
    try:
        snapshots_dir_exists = snapshots_dir.is_dir()
    except PermissionError:
        return model_name
    if not snapshots_dir_exists:
        return model_name

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return model_name
    snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(snapshots[0])


def build_hybrid_index_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.gpu_only:
        if args.hybrid_policy and args.hybrid_policy != "gpu_only":
            raise ValueError("--gpu_only conflicts with --hybrid_policy")
        return {"policy": "gpu_only", "start": "gpu"}
    if not args.hybrid_policy and args.hybrid_gpu_every <= 0:
        return None
    cfg: dict[str, Any] = {
        "policy": args.hybrid_policy or "gpu_every_n",
        "start": args.hybrid_start,
    }
    if args.hybrid_gpu_every > 0:
        cfg["gpu_every"] = args.hybrid_gpu_every
    return cfg


def build_decode_search_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.decode_search_mode:
        if args.candidate_clusters is not None or args.token_rerank_budget_ratio is not None:
            raise ValueError("candidate_clusters/token_rerank_budget_ratio require an implemented decode_search mode")
        return None
    if args.decode_search_mode != "flat":
        raise ValueError("Online subspace_rerank decode search is not implemented in this runtime")
    if args.candidate_clusters is not None or args.token_rerank_budget_ratio is not None:
        raise ValueError("candidate_clusters/token_rerank_budget_ratio are not used by flat decode search")
    return None


def select_data(data: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    if args.sample_indices:
        indices: list[int] = []
        for value in args.sample_indices.split(","):
            value = value.strip()
            if not value:
                continue
            index = int(value)
            if index < 0 or index >= len(data):
                raise IndexError(f"--sample_indices index {index} is outside dataset range [0, {len(data) - 1}]")
            indices.append(index)
        if not indices:
            raise ValueError("--sample_indices did not contain any valid indices")
        return [(index, data[index]) for index in indices]

    selected = data[args.start :]
    if args.max_samples > 0:
        selected = selected[: args.max_samples]
    return list(enumerate(selected, start=args.start))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    from model_hub import load_model, load_tokenizer

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    config_model_name = args.config_model_name or args.model_name
    model_name_or_path = resolve_local_hf_snapshot(args.model_name)
    tokenizer = load_tokenizer(model_name_or_path)
    llm = load_model(
        model_name_or_path,
        args.max_model_len,
        dtype,
        args.device,
        tokenizer,
        parallel_mode=args.parallel_mode,
        weight_loading=args.weight_loading,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    data = select_data(load_data(args.data_path), args)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(str(out_path))
    max_input_tokens = args.max_model_len - args.max_new_tokens

    with out_path.open("a", encoding="utf-8") as fout:
        for dataset_index, item in tqdm(data):
            item_id = str(item.get("_id"))
            if item_id in done:
                continue

            prompt = None
            input_ids = None
            attention_mask = None
            output_ids = None
            attn_config = None
            response = None
            record = None

            try:
                prompt = render_prompt(item)
                if args.chat_template:
                    prompt = apply_chat_template(tokenizer, prompt, args.enable_thinking)
                input_ids, attention_mask = tokenize_with_middle_truncation(tokenizer, prompt, max_input_tokens)
                input_len = int(input_ids.shape[1])

                attn_config = generate_config(
                    config_model_name,
                    input_len,
                    args.attn_type,
                    retrieval_budget=args.retrieval_budget,
                    cache_ratio=args.cache_ratio,
                    gpu_only=args.gpu_only,
                    subspace_parts=args.subspace_parts,
                    imi_engine=args.imi_engine,
                    partition_path=args.partition_path,
                    hybrid_index=build_hybrid_index_config(args),
                    decode_search=build_decode_search_config(args),
                    prefill_chunk_size=args.prefill_chunk_size,
                )

                output_ids = llm.generate(
                    attention_type=args.attn_type,
                    inputs_ids=input_ids.to(llm.layers[0].device),
                    attention_masks=attention_mask.to(llm.layers[0].device),
                    max_new_length=args.max_new_tokens,
                    attn_config=attn_config,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    ignore_eos=args.ignore_eos,
                    prefill_bsz=args.prefill_bsz,
                )

                response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                pred = extract_answer(response)
                answer = str(item.get("answer", "")).strip().upper()

                record = dict(item)
                if args.save_context_chars >= 0:
                    record["context"] = str(item.get("context", ""))[: args.save_context_chars]
                record.update(
                    {
                        "response": response,
                        "pred": pred,
                        "judge": pred == answer,
                        "dataset_index": dataset_index,
                        "input_tokens": input_len,
                        "model_name": args.model_name,
                        "model_path": model_name_or_path,
                        "attn_type": args.attn_type,
                        "imi_engine": args.imi_engine,
                        "timing": dict(getattr(llm, "last_timing_stats", {}) or {}),
                        "benchmark_config": {
                            "data_path": args.data_path,
                            "model_path": model_name_or_path,
                            "config_model_name": config_model_name,
                            "attn_type": args.attn_type,
                            "device": args.device,
                            "parallel_mode": args.parallel_mode,
                            "dtype": args.dtype,
                            "max_model_len": args.max_model_len,
                            "max_new_tokens": args.max_new_tokens,
                            "retrieval_budget": args.retrieval_budget,
                            "cache_ratio": args.cache_ratio,
                            "prefill_chunk_size": args.prefill_chunk_size,
                            "sample_indices": args.sample_indices,
                            "profile_decode": os.getenv("IMI_PROFILE_DECODE", "0") == "1",
                            "profile_steps": int(os.getenv("IMI_PROFILE_STEPS", "5")),
                            "breakdown_benchmark": os.getenv("IMI_BREAKDOWN_BENCHMARK", "0") == "1",
                            "profile_hit_rate": os.getenv("IMI_PROFILE_HIT_RATE", "0") == "1",
                            "profile_cache": os.getenv("IMI_PROFILE_CACHE", "0") == "1",
                            "profile_token_hit_rate": os.getenv("IMI_PROFILE_TOKEN_HIT_RATE", "0") == "1",
                            "tpot_valid_for_performance": not (
                                os.getenv("IMI_PROFILE_DECODE", "0") == "1"
                                or os.getenv("IMI_BREAKDOWN_BENCHMARK", "0") == "1"
                            ),
                        },
                    }
                )
                timing = record["timing"]
                decode_time = float(timing.get("decode_time", 0.0) or 0.0)
                decode_steps = int(timing.get("decode_steps", 0) or 0)
                timing["tpot_ms"] = decode_time * 1000.0 / decode_steps if decode_steps > 0 else 0.0
                profile_enabled = os.getenv("IMI_PROFILE_DECODE", "0") == "1"
                breakdown_enabled = os.getenv("IMI_BREAKDOWN_BENCHMARK", "0") == "1"
                timing["measurement_mode"] = (
                    "instrumented_profile"
                    if profile_enabled or breakdown_enabled
                    else "unprofiled_wall"
                )
                timing["tpot_valid_for_performance"] = not (
                    profile_enabled or breakdown_enabled
                )
                decode_breakdown = (
                    timing.get("breakdown", {}).get("decode", {})
                    if isinstance(timing.get("breakdown"), dict)
                    else {}
                )
                sparse_profile = decode_breakdown.get("sparse_avg_per_step_ms", {})
                if not isinstance(sparse_profile, dict):
                    sparse_profile = {}
                hit_blocks = float(sparse_profile.get("hit_blocks", 0.0) or 0.0)
                miss_blocks = float(sparse_profile.get("miss_blocks", 0.0) or 0.0)
                hit_tokens = float(sparse_profile.get("hit_tokens", 0.0) or 0.0)
                miss_tokens = float(sparse_profile.get("miss_tokens", 0.0) or 0.0)
                profile_rows = decode_breakdown.get("profile_step_rows", [])
                if not isinstance(profile_rows, list):
                    profile_rows = []
                profiled_steps = int(
                    decode_breakdown.get(
                        "profiled_steps",
                        getattr(getattr(llm, "kv_cache", None), "profile_decode_count", 0),
                    )
                    or 0
                )
                timing["profiled_steps"] = profiled_steps
                timing["decode_profile"] = {
                    "search_ms": float(sparse_profile.get("search_ms", 0.0) or 0.0),
                    "selection_d2h_ms": float(sparse_profile.get("selection_d2h_ms", 0.0) or 0.0),
                    "cpu_lookup_ms": 0.0,
                    "gpu_cache_plan_ms": float(
                        sparse_profile.get("gpu_cache_plan_ms", 0.0) or 0.0
                    ),
                    "gather_ms": float(sparse_profile.get("gather_ms", 0.0) or 0.0),
                    "attention_ms": float(sparse_profile.get("attn_ms", 0.0) or 0.0),
                    "cache_sync_ms": float(sparse_profile.get("cache_sync_ms", 0.0) or 0.0),
                    "scatter_ms": float(sparse_profile.get("scatter_ms", 0.0) or 0.0),
                    # This is the sparse-attention/retrieval window only. It
                    # is not the complete cross-layer decode step.
                    "sparse_attention_total_ms": float(sparse_profile.get("total_ms", 0.0) or 0.0),
                    "full_decode_step_ms": float(
                        decode_breakdown.get(
                            "profile_decode_step_ms",
                            timing.get("profile_decode_step_ms", timing["tpot_ms"]),
                        )
                        or 0.0
                    ),
                    "other_ms": float(sparse_profile.get("other_ms", 0.0) or 0.0),
                    "hit_blocks": hit_blocks,
                    "miss_blocks": miss_blocks,
                    "block_hit_rate": hit_blocks / (hit_blocks + miss_blocks)
                    if hit_blocks + miss_blocks > 0
                    else 0.0,
                    "hit_tokens": hit_tokens,
                    "miss_tokens": miss_tokens,
                    "token_hit_rate": hit_tokens / (hit_tokens + miss_tokens)
                    if hit_tokens + miss_tokens > 0
                    else 0.0,
                    "block_hit_rate_all_steps": float(
                        decode_breakdown.get("all_steps_block_hit_rate", 0.0) or 0.0
                    ),
                    "block_hit_rate_steady": float(
                        decode_breakdown.get("steady_steps_block_hit_rate", 0.0) or 0.0
                    ),
                    "token_hit_rate_all_steps": float(
                        decode_breakdown.get("all_steps_token_hit_rate", 0.0) or 0.0
                    ),
                    "token_hit_rate_steady": float(
                        decode_breakdown.get("steady_steps_token_hit_rate", 0.0) or 0.0
                    ),
                    "token_hit_rate_measured": bool(
                        decode_breakdown.get("token_hit_rate_measured", False)
                    ),
                    "sparse_timing_scope": "sparse_attention_only",
                    "unseparated_stages": [
                        "selection_d2h",
                        "cache_sync",
                        "scatter",
                    ],
                    "not_used_stages": ["cpu_lookup"],
                    "profile_step_rows": profile_rows,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                done.add(item_id)
            finally:
                del prompt, input_ids, attention_mask, output_ids, attn_config, response, record
                cleanup_after_sample(llm)


if __name__ == "__main__":
    main()
