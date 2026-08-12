#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from tqdm import tqdm


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
    from config import generate_config, resolve_model_path  # noqa: E402
    from model_hub import load_model, load_tokenizer  # noqa: E402
finally:
    if _original_home is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _original_home

from math500_utils import (  # noqa: E402
    completed_ids,
    extract_final_answer,
    load_jsonl,
    math_answers_equal,
    select_rows,
    summarize_rows,
)


DEFAULT_DATA_PATH = PROJECT_ROOT / "benchmark" / "math500" / "test.jsonl"
DEFAULT_MODEL_PATH = (
    "/home/yjy/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
DEFAULT_CONFIG_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "math500" / "results"

MATH_PROMPT_TEMPLATE = r"""Solve the following math problem. Show your reasoning clearly and step by step.

Your final answer must be on the last line in exactly this format:
Therefore, the final answer is: \boxed{{ANSWER}}

Here ANSWER must be only the final number or expression that solves the problem.

Problem:
{problem}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MATH-500 generation with Full Attention or AdaptiveIMI."
    )
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--config_model_name", type=str, default=DEFAULT_CONFIG_MODEL_NAME)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--sample_indices", type=str, default=None)
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
        type=str,
        default="AdaptiveIMI",
        choices=("Full_Flash_Attn", "AdaptiveIMI"),
    )
    parser.add_argument("--retrieval_budget", type=float, default=None)
    parser.add_argument("--cache_ratio", type=float, default=None)
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument("--partition_path", type=str, default=os.getenv("IMI_PARTITION_PATH"))
    parser.add_argument("--imi_engine", type=str, default=None, choices=("hybrid_8way_v2",))
    parser.add_argument("--hybrid_policy", type=str, default=None)
    parser.add_argument("--hybrid_start", type=str, default="gpu")
    parser.add_argument("--hybrid_gpu_every", type=int, default=0)
    parser.add_argument("--prefill_chunk_size", type=int, default=None)
    parser.add_argument("--prefill_bsz", type=int, default=1)
    parser.add_argument("--chat_template", action="store_true", default=True)
    parser.add_argument("--no_chat_template", action="store_false", dest="chat_template")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--disable_resume", action="store_true")
    parser.add_argument("--allow_input_truncation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def apply_chat_template(tokenizer: Any, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
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


def tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    max_input_tokens: int,
    allow_input_truncation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) > max_input_tokens:
        if not allow_input_truncation:
            raise ValueError(
                f"Math prompt has {len(token_ids)} tokens, but only {max_input_tokens} "
                "tokens are available. Increase --max_model_len or use "
                "--allow_input_truncation."
            )
        keep_head = max_input_tokens // 2
        keep_tail = max_input_tokens - keep_head
        token_ids = token_ids[:keep_head] + token_ids[-keep_tail:]
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    return input_ids, torch.ones_like(input_ids)


def build_hybrid_index_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.gpu_only:
        if args.hybrid_policy and args.hybrid_policy != "gpu_only":
            raise ValueError("--gpu_only conflicts with --hybrid_policy")
        return {"policy": "gpu_only", "start": "gpu"}
    if not args.hybrid_policy and args.hybrid_gpu_every <= 0:
        return None
    config: dict[str, Any] = {
        "policy": args.hybrid_policy or "gpu_every_n",
        "start": args.hybrid_start,
    }
    if args.hybrid_gpu_every > 0:
        config["gpu_every"] = args.hybrid_gpu_every
    return config


def cleanup_after_sample(llm: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

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
        torch.cuda.synchronize()


def model_path_for_loading(args: argparse.Namespace) -> str:
    candidate = args.model_path or args.model_name
    if not candidate:
        raise ValueError("Set --model_path or --model_name")
    resolved = resolve_model_path(candidate)
    return resolved if resolved else candidate


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    out_path = (
        Path(args.out_path)
        if args.out_path
        else DEFAULT_OUTPUT_DIR / f"{args.attn_type.lower()}.jsonl"
    )
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else out_path.with_suffix(".summary.json")
    )
    return out_path, summary_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.max_model_len <= args.max_new_tokens:
        raise ValueError("--max_model_len must be larger than --max_new_tokens")

    out_path, summary_path = output_paths(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if args.disable_resume and out_path.exists():
        out_path.unlink()

    data = load_jsonl(args.data_path)
    selected = select_rows(
        data,
        start=args.start,
        max_samples=args.max_samples,
        sample_indices=args.sample_indices,
    )
    done = set() if args.disable_resume else completed_ids(out_path)

    model_path = model_path_for_loading(args)
    model_name = args.model_name or model_path
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    print(f"Data: {args.data_path} ({len(data)} rows)")
    print(f"Selected: {len(selected)} rows")
    print(f"Model: {model_path}")
    print(f"Attention: {args.attn_type}")
    print(f"Output: {out_path}")

    tokenizer = load_tokenizer(model_path)
    llm = load_model(
        model_path,
        args.max_model_len,
        dtype,
        args.device,
        tokenizer,
        parallel_mode=args.parallel_mode,
        weight_loading=args.weight_loading,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    max_input_tokens = args.max_model_len - args.max_new_tokens
    config_model_name = args.config_model_name or model_name

    with out_path.open("a", encoding="utf-8") as fout:
        for dataset_index, item in tqdm(selected, total=len(selected)):
            unique_id = str(item.get("unique_id", dataset_index))
            if unique_id in done:
                continue

            prompt = MATH_PROMPT_TEMPLATE.format(problem=str(item["problem"]).strip())
            if args.chat_template:
                prompt = apply_chat_template(tokenizer, prompt, args.enable_thinking)

            input_ids, attention_mask = tokenize_prompt(
                tokenizer,
                prompt,
                max_input_tokens=max_input_tokens,
                allow_input_truncation=args.allow_input_truncation,
            )
            input_tokens = int(input_ids.shape[1])

            attn_config = generate_config(
                config_model_name,
                input_tokens,
                args.attn_type,
                retrieval_budget=args.retrieval_budget,
                cache_ratio=args.cache_ratio,
                gpu_only=args.gpu_only,
                imi_engine=args.imi_engine,
                partition_path=args.partition_path,
                hybrid_index=build_hybrid_index_config(args),
                prefill_chunk_size=args.prefill_chunk_size,
            )

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
                    do_sample=False,
                    ignore_eos=False,
                    prefill_bsz=args.prefill_bsz,
                )
                response = tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                )[0].strip()
                predicted = extract_final_answer(response)
                target = str(item.get("answer", "")).strip()
                judge = math_answers_equal(predicted, target)
            except Exception as exc:
                predicted = ""
                target = str(item.get("answer", "")).strip()
                judge = False
                error = f"{type(exc).__name__}: {exc}"
                print(f"Sample {dataset_index} failed: {error}", file=sys.stderr, flush=True)

            timing = dict(getattr(llm, "last_timing_stats", {}) or {})
            decode_time = float(timing.get("decode_time", 0.0) or 0.0)
            decode_steps = int(timing.get("decode_steps", 0) or 0)
            timing["tpot_ms"] = (
                decode_time * 1000.0 / decode_steps if decode_steps > 0 else 0.0
            )
            timing["measurement_mode"] = (
                "instrumented_profile"
                if os.getenv("IMI_PROFILE_DECODE", "0") == "1"
                or os.getenv("IMI_BREAKDOWN_BENCHMARK", "0") == "1"
                else "unprofiled_wall"
            )

            output_tokens = 0
            if output_ids is not None and len(output_ids) > 0:
                output_tokens = len(output_ids[0])

            record = dict(item)
            record.update(
                {
                    "dataset_index": dataset_index,
                    "unique_id": unique_id,
                    "prompt": prompt,
                    "response": response,
                    "predicted_answer": predicted,
                    "target_answer": target,
                    "judge": bool(judge),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "model_name": model_name,
                    "model_path": model_path,
                    "config_model_name": config_model_name,
                    "attn_type": args.attn_type,
                    "timing": timing,
                    "error": error,
                    "benchmark_config": {
                        "data_path": args.data_path,
                        "max_model_len": args.max_model_len,
                        "max_new_tokens": args.max_new_tokens,
                        "dtype": args.dtype,
                        "device": args.device,
                        "retrieval_budget": args.retrieval_budget,
                        "cache_ratio": args.cache_ratio,
                        "partition_path": args.partition_path,
                        "imi_engine": args.imi_engine,
                        "chat_template": args.chat_template,
                        "enable_thinking": args.enable_thinking,
                        "profile_decode": os.getenv("IMI_PROFILE_DECODE", "0") == "1",
                        "profile_steps": int(os.getenv("IMI_PROFILE_STEPS", "5")),
                    },
                }
            )
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(unique_id)

            cleanup_after_sample(llm)

    rows = load_jsonl(out_path)
    summary = summarize_rows(rows)
    summary.update(
        {
            "data_path": str(args.data_path),
            "output_path": str(out_path),
            "summary_path": str(summary_path),
            "selected_start": args.start,
            "selected_max_samples": args.max_samples,
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
