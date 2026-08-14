#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault(
    "FLASHINFER_WORKSPACE_BASE",
    str(PROJECT_ROOT / "outputs" / "flashinfer"),
)

_original_home = os.environ.get("HOME")
_flashinfer_home = PROJECT_ROOT / "outputs" / "flashinfer_home"
_flashinfer_home.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(_flashinfer_home)
try:
    from config import generate_config, resolve_model_path
    from model_hub import load_model, load_tokenizer
finally:
    if _original_home is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _original_home

from aime26_utils import (  # noqa: E402
    answers_equal,
    extract_integer_answer,
    latest_attempts_by_problem,
    load_aime26_data,
    select_rows,
)


DEFAULT_DATA_PATH = PROJECT_ROOT / "benchmark" / "AIME26" / "data" / "aime2026.jsonl"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_CONFIG_MODEL_NAME = "Qwen3-8B"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "AIME26" / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AIME26 with full attention, full-attention offload, or AdaptiveIMI using pass@k."
    )
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--config_model_name", type=str, default=DEFAULT_CONFIG_MODEL_NAME)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--parallel_mode",
        "--parallel-mode",
        type=str,
        default="single",
        choices=("auto", "single", "layer", "tp"),
    )
    parser.add_argument(
        "--weight_loading",
        "--weight-loading",
        type=str,
        default="hf",
        choices=("hf", "auto", "accelerate_meta", "streaming_safetensors"),
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--dtype", type=str, default="bf16", choices=("fp16", "bf16"))
    parser.add_argument(
        "--attn_type",
        "--attention_type",
        type=str,
        default="Full_Flash_Attn",
        choices=("Full_Flash_Attn", "Full_Flash_Attn_Offload", "AdaptiveIMI"),
    )
    parser.add_argument("--retrieval_budget", type=float, default=None)
    parser.add_argument("--cache_ratio", type=float, default=None)
    parser.add_argument("--partition_path", type=str, default=None)
    parser.add_argument("--imi_engine", type=str, default="hybrid_8way_v2")
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument("--prefill_chunk_size", type=int, default=None)
    parser.add_argument("--n_generate", type=int, default=1)
    parser.add_argument("--pass_k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--early_stop_on_first_correct", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--disable_resume", action="store_true")
    return parser.parse_args()


def apply_chat_template(tokenizer: Any, prompt: str, enable_thinking: bool) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_prompt(problem: str) -> str:
    return f"""You are an expert mathematical problem solver. Solve the following AIME problem with complete step-by-step reasoning.

Problem:
{problem}

Instructions:
1. Work through the problem step by step.
2. Show the key reasoning and calculations.
3. The final answer must be an integer from 0 to 999.
4. When you reach the final answer, write it exactly as \\boxed{{integer}}.
5. Stop immediately after the boxed answer.

Solve the problem:"""


def build_hybrid_index_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.gpu_only:
        return {"policy": "gpu_only", "start": "gpu"}
    return None


def cleanup_model_after_problem(llm: Any) -> None:
    kv_cache = getattr(llm, "kv_cache", None)
    if kv_cache is not None:
        cleanup = getattr(kv_cache, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception as exc:
                print(f"Warning: KV cache cleanup failed: {exc}", file=sys.stderr)
        llm.kv_cache = None
    clear_workspace = getattr(llm, "_clear_prefill_prefix_kv_workspace", None)
    if callable(clear_workspace):
        clear_workspace()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def output_path_for(args: argparse.Namespace) -> Path:
    if args.out_path:
        return Path(args.out_path)
    mode = f"pass@{args.pass_k}"
    return DEFAULT_OUTPUT_DIR / f"{args.attn_type.lower()}_{mode}.jsonl"


def matches_experiment(row: dict[str, Any], args: argparse.Namespace) -> bool:
    """Return whether a stored record belongs to this exact experiment."""
    if row.get("attn_type") != args.attn_type:
        return False
    if int(row.get("n_generate", 1) or 1) != args.n_generate:
        return False
    if int(row.get("pass_k", 1) or 1) != args.pass_k:
        return False

    config = dict(row.get("benchmark_config") or {})
    expected = {
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "config_model_name": args.config_model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "enable_thinking": not args.disable_thinking,
    }
    return all(config.get(name) == value for name, value in expected.items())


def main() -> None:
    args = parse_args()
    args.pass_k = args.n_generate if args.pass_k is None else args.pass_k
    if args.n_generate <= 0 or args.pass_k <= 0 or args.pass_k > args.n_generate:
        raise ValueError("Require 0 < pass_k <= n_generate")
    if args.early_stop_on_first_correct and args.pass_k > 1:
        raise ValueError(
            "--early_stop_on_first_correct is incompatible with pass@k for k > 1; "
            "it would write incomplete attempt groups."
        )
    if args.max_model_len <= args.max_new_tokens:
        raise ValueError("--max_model_len must be larger than --max_new_tokens")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.attn_type == "AdaptiveIMI" and args.retrieval_budget is None:
        args.retrieval_budget = 0.1
    if args.attn_type == "AdaptiveIMI" and args.cache_ratio is None:
        args.cache_ratio = 0.05

    set_seed(args.seed)
    out_path = output_path_for(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.disable_resume and out_path.exists():
        out_path.unlink()
    existing_attempts = latest_attempts_by_problem(out_path)

    data = load_aime26_data(args.data_path)
    selected = select_rows(data, start_idx=args.start_idx, num_samples=args.num_samples)
    model_path = resolve_model_path(args.model_path)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"Dataset: {args.data_path} ({len(data)} problems)")
    print(f"Selected: {len(selected)} problems")
    print(f"Model: {model_path}")
    print(f"Attention: {args.attn_type}")
    print(f"pass@{args.pass_k}: {args.n_generate} attempt(s) per problem")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {args.device}")
    print(f"Output: {out_path}")

    tokenizer = load_tokenizer(model_path)
    llm = load_model(
        model_path,
        args.max_model_len,
        dtype,
        args.device,
        tokenizer=tokenizer,
        parallel_mode=args.parallel_mode,
        weight_loading=args.weight_loading,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    do_sample = args.temperature > 1e-5

    with out_path.open("a", encoding="utf-8") as fout:
        for dataset_index, item in selected:
            item_id = str(item.get("id", dataset_index))
            existing_by_index = existing_attempts.get(item_id, {})
            incompatible_indices = [
                attempt_index
                for attempt_index, row in existing_by_index.items()
                if not matches_experiment(row, args)
            ]
            if incompatible_indices:
                raise ValueError(
                    f"Output {out_path} already contains incompatible records for "
                    f"problem {item_id}, attempt(s) {sorted(incompatible_indices)}. "
                    "Use a new --out_path or pass --disable_resume."
                )

            problem = str(item["problem"]).strip()
            target = str(item["answer"]).strip()
            prompt = build_prompt(problem)
            prompt = apply_chat_template(
                tokenizer,
                prompt,
                enable_thinking=not args.disable_thinking,
            )
            encoded = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask
            input_tokens = int(input_ids.shape[1])
            if input_tokens + args.max_new_tokens > args.max_model_len:
                raise ValueError(
                    f"Problem {item_id} requires {input_tokens + args.max_new_tokens} tokens, "
                    f"exceeding --max_model_len={args.max_model_len}"
                )

            attn_config = generate_config(
                args.config_model_name,
                input_tokens,
                args.attn_type,
                retrieval_budget=args.retrieval_budget,
                cache_ratio=args.cache_ratio,
                partition_path=args.partition_path,
                hybrid_index=build_hybrid_index_config(args),
                gpu_only=args.gpu_only,
                imi_engine=args.imi_engine if args.attn_type == "AdaptiveIMI" else None,
                prefill_chunk_size=args.prefill_chunk_size,
            )

            attempt_rows: list[dict[str, Any]] = []
            for attempt_index in range(args.n_generate):
                stored_attempt = existing_by_index.get(attempt_index + 1)
                if stored_attempt is not None and not stored_attempt.get("error"):
                    attempt_rows.append(stored_attempt)
                    continue
                attempt_seed = args.seed + dataset_index * 1000 + attempt_index
                set_seed(attempt_seed)
                started = time.perf_counter()
                output_ids = None
                response = ""
                error = None
                try:
                    output_ids = llm.generate(
                        attention_type=args.attn_type,
                        inputs_ids=input_ids.to(llm.layers[0].device),
                        attention_masks=attention_mask.to(llm.layers[0].device),
                        max_new_length=args.max_new_tokens,
                        attn_config=attn_config,
                        do_sample=do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        ignore_eos=False,
                    )
                    response = tokenizer.batch_decode(
                        output_ids,
                        skip_special_tokens=True,
                    )[0].strip()
                    predicted = extract_integer_answer(response)
                except Exception as exc:
                    predicted = None
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"Problem {item_id} attempt {attempt_index + 1} failed: {error}", file=sys.stderr)

                elapsed = time.perf_counter() - started
                timing = dict(getattr(llm, "last_timing_stats", {}) or {})
                decode_steps = int(timing.get("decode_steps", 0) or 0)
                decode_time = float(timing.get("decode_time", 0.0) or 0.0)
                timing["tpot_ms"] = (
                    decode_time * 1000.0 / decode_steps if decode_steps else 0.0
                )
                timing["wall_time_s"] = elapsed
                output_tokens = 0
                if output_ids is not None and len(output_ids) > 0:
                    output_tokens = len(output_ids[0])
                is_correct = answers_equal(predicted, target)
                attempt = {
                    "record_type": "attempt",
                    "dataset_index": dataset_index,
                    "id": item_id,
                    "attempt_index": attempt_index + 1,
                    "n_generate": args.n_generate,
                    "pass_k": args.pass_k,
                    "problem": problem,
                    "target_answer": target,
                    "predicted_answer": predicted,
                    "response": response,
                    "is_correct": is_correct,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "timing": timing,
                    "error": error,
                    "model_path": model_path,
                    "attn_type": args.attn_type,
                    "seed": attempt_seed,
                    "benchmark_config": {
                        "data_path": str(args.data_path),
                        "max_model_len": args.max_model_len,
                        "max_new_tokens": args.max_new_tokens,
                        "dtype": args.dtype,
                        "device": args.device,
                        "retrieval_budget": args.retrieval_budget,
                        "cache_ratio": args.cache_ratio,
                        "partition_path": args.partition_path,
                        "config_model_name": args.config_model_name,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "enable_thinking": not args.disable_thinking,
                    },
                    "completion_status": (
                        "error"
                        if error
                        else str(timing.get("stop_reason", "length_limit"))
                    ),
                }
                fout.write(json.dumps(attempt, ensure_ascii=False) + "\n")
                fout.flush()
                attempt_rows.append(attempt)

                if args.early_stop_on_first_correct and is_correct:
                    break

            successful_attempts = [row for row in attempt_rows if not row.get("error")]
            if len(successful_attempts) != args.n_generate:
                print(
                    f"Problem {item_id} remains incomplete: "
                    f"{len(successful_attempts)}/{args.n_generate} successful attempts.",
                    file=sys.stderr,
                )
                cleanup_model_after_problem(llm)
                continue

            successful_attempts.sort(key=lambda row: int(row["attempt_index"]))
            correct_count = sum(bool(row["is_correct"]) for row in successful_attempts)
            summary = {
                "record_type": "summary",
                "dataset_index": dataset_index,
                "id": item_id,
                "n_generate": args.n_generate,
                "pass_k": args.pass_k,
                "attempted_generate": len(successful_attempts),
                "correct_count": correct_count,
                "pass_at_k": correct_count > 0,
                "problem": problem,
                "target_answer": target,
                "input_tokens": input_tokens,
                "mean_output_tokens": (
                    sum(row["output_tokens"] for row in successful_attempts) / len(successful_attempts)
                    if successful_attempts else 0.0
                ),
                "mean_wall_time_s": (
                    sum(row["timing"]["wall_time_s"] for row in successful_attempts) / len(successful_attempts)
                    if successful_attempts else 0.0
                ),
                "model_path": model_path,
                "attn_type": args.attn_type,
            }
            fout.write(json.dumps(summary, ensure_ascii=False) + "\n")
            fout.flush()
            cleanup_model_after_problem(llm)

    print(f"Finished. Results saved to: {out_path}")


if __name__ == "__main__":
    main()
