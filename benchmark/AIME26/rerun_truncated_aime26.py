#!/usr/bin/env python3
"""Rerun only max-token-truncated AIME attempts and merge the replacements."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from aime26_inference import (
    apply_chat_template,
    build_hybrid_index_config,
    build_prompt,
    cleanup_model_after_problem,
    load_model,
    load_tokenizer,
    resolve_model_path,
    set_seed,
)
from aime26_utils import (
    answers_equal,
    extract_integer_answer,
    load_aime26_data,
    load_jsonl,
)
from config import generate_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "benchmark" / "AIME26" / "data" / "aime2026.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun only attempts that reached the source max_new_tokens limit."
    )
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--patch_path", required=True)
    parser.add_argument("--merged_path", required=True)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--config_model_name", default=None)
    parser.add_argument("--attn_type", choices=("Full_Flash_Attn", "AdaptiveIMI"), default=None)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--retrieval_budget", type=float, default=None)
    parser.add_argument("--cache_ratio", type=float, default=None)
    parser.add_argument("--partition_path", default=None)
    parser.add_argument("--imi_engine", default="hybrid_8way_v2")
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument("--prefill_chunk_size", type=int, default=None)
    parser.add_argument("--parallel_mode", default="single", choices=("auto", "single", "layer", "tp"))
    parser.add_argument(
        "--weight_loading",
        default="hf",
        choices=("hf", "auto", "accelerate_meta", "streaming_safetensors"),
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("id", row.get("dataset_index"))), int(row["attempt_index"]))


def source_max_new_tokens(attempts: list[dict[str, Any]]) -> int:
    values = [
        int(row.get("benchmark_config", {}).get("max_new_tokens", 0) or 0)
        for row in attempts
    ]
    values = [value for value in values if value > 0]
    if not values:
        raise ValueError("Source results do not contain benchmark_config.max_new_tokens")
    return max(values)


def load_attempts(path: str | Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    attempts = [row for row in rows if row.get("record_type") == "attempt"]
    if not attempts:
        raise ValueError(f"No attempt records found in {path}")
    return attempts


def infer_value(
    explicit: Any,
    source_attempt: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if explicit is not None:
        return explicit
    config = source_attempt.get("benchmark_config", {})
    if key in config and config[key] is not None:
        return config[key]
    return default


def build_attempt_record(
    *,
    source_attempt: dict[str, Any],
    response: str,
    predicted: str | None,
    timing: dict[str, Any],
    output_tokens: int,
    error: str | None,
    model_path: str,
    args: argparse.Namespace,
    input_tokens: int,
) -> dict[str, Any]:
    target = str(source_attempt.get("target_answer", source_attempt.get("answer", "")))
    return {
        "record_type": "attempt",
        "dataset_index": source_attempt["dataset_index"],
        "id": str(source_attempt["id"]),
        "attempt_index": int(source_attempt["attempt_index"]),
        "n_generate": int(source_attempt.get("n_generate", 8)),
        "pass_k": int(source_attempt.get("pass_k", 8)),
        "problem": source_attempt["problem"],
        "target_answer": target,
        "predicted_answer": predicted,
        "response": response,
        "is_correct": answers_equal(predicted, target),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "timing": timing,
        "error": error,
        "model_path": model_path,
        "attn_type": args.attn_type,
        "seed": int(source_attempt["seed"]),
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
        "rerun_reason": "source_attempt_reached_max_new_tokens",
        "source_max_new_tokens": source_max_new_tokens([source_attempt]),
    }


def rerun_targets(
    *,
    targets: list[dict[str, Any]],
    data_path: str,
    args: argparse.Namespace,
    patch_path: str | Path,
) -> int:
    data = load_aime26_data(data_path)
    data_by_index = {index: item for index, item in enumerate(data)}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in targets:
        grouped[int(row["dataset_index"])].append(row)

    model_path = resolve_model_path(args.model_path)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"Target attempts: {len(targets)}")
    print(f"Target problems: {len(grouped)}")
    print(f"Model: {model_path}")
    print(f"Attention: {args.attn_type}")
    print(f"New max_model_len: {args.max_model_len}")
    print(f"New max_new_tokens: {args.max_new_tokens}")

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
    completed = 0

    Path(patch_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(patch_path).open("a", encoding="utf-8") as patch_file:
        for dataset_index in sorted(grouped):
            item = data_by_index.get(dataset_index)
            if item is None:
                raise ValueError(f"Dataset index {dataset_index} is missing from {data_path}")
            problem = str(item["problem"]).strip()
            prompt = apply_chat_template(
                tokenizer,
                build_prompt(problem),
                enable_thinking=not args.disable_thinking,
            )
            encoded = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask
            input_tokens = int(input_ids.shape[1])
            if input_tokens + args.max_new_tokens > args.max_model_len:
                raise ValueError(
                    f"Problem {item.get('id', dataset_index)} requires "
                    f"{input_tokens + args.max_new_tokens} tokens, exceeding "
                    f"--max_model_len={args.max_model_len}"
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

            for source_attempt in sorted(
                grouped[dataset_index], key=lambda row: int(row["attempt_index"])
            ):
                attempt_seed = int(source_attempt["seed"])
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
                        do_sample=args.temperature > 1e-5,
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
                    print(
                        f"Problem {source_attempt['id']} attempt "
                        f"{source_attempt['attempt_index']} failed: {error}",
                        file=sys.stderr,
                    )

                elapsed = time.perf_counter() - started
                timing = dict(getattr(llm, "last_timing_stats", {}) or {})
                decode_steps = int(timing.get("decode_steps", 0) or 0)
                decode_time = float(timing.get("decode_time", 0.0) or 0.0)
                timing["tpot_ms"] = (
                    decode_time * 1000.0 / decode_steps if decode_steps else 0.0
                )
                timing["wall_time_s"] = elapsed
                output_tokens = (
                    len(output_ids[0])
                    if output_ids is not None and len(output_ids) > 0
                    else 0
                )
                result = build_attempt_record(
                    source_attempt=source_attempt,
                    response=response,
                    predicted=predicted,
                    timing=timing,
                    output_tokens=output_tokens,
                    error=error,
                    model_path=model_path,
                    args=args,
                    input_tokens=input_tokens,
                )
                patch_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                patch_file.flush()
                completed += 1
                print(
                    f"[patch] completed {completed}/{len(targets)}: "
                    f"id={source_attempt['id']} attempt={source_attempt['attempt_index']} "
                    f"generated={timing.get('actual_generated', 0)} "
                    f"correct={result['is_correct']}",
                    flush=True,
                )

            cleanup_model_after_problem(llm)

    return completed


def write_patch(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_results(
    source_path: str | Path,
    patch_path: str | Path,
    merged_path: str | Path,
) -> None:
    source_rows = load_jsonl(source_path)
    patch_rows = load_jsonl(patch_path) if Path(patch_path).exists() else []
    replacements = {
        row_key(row): row
        for row in patch_rows
        if row.get("record_type") == "attempt"
    }
    if len(replacements) != sum(
        row.get("record_type") == "attempt" for row in patch_rows
    ):
        raise ValueError("Patch contains duplicate attempt keys")

    attempts_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summaries: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        item_id = str(row.get("id", row.get("dataset_index")))
        if row.get("record_type") == "attempt":
            attempts_by_id[item_id].append(replacements.get(row_key(row), row))
        elif row.get("record_type") == "summary":
            summaries[item_id] = row

    order = sorted(
        attempts_by_id,
        key=lambda item_id: int(attempts_by_id[item_id][0].get("dataset_index", 0)),
    )
    output_rows: list[dict[str, Any]] = []
    for item_id in order:
        attempts = sorted(attempts_by_id[item_id], key=lambda row: int(row["attempt_index"]))
        output_rows.extend(attempts)
        old_summary = dict(summaries.get(item_id, {}))
        correct_count = sum(bool(row.get("is_correct")) for row in attempts)
        old_summary.update(
            {
                "record_type": "summary",
                "id": item_id,
                "dataset_index": attempts[0].get("dataset_index"),
                "n_generate": max(int(row.get("n_generate", 1)) for row in attempts),
                "pass_k": max(int(row.get("pass_k", 1)) for row in attempts),
                "attempted_generate": len(attempts),
                "correct_count": correct_count,
                "pass_at_k": correct_count > 0,
                "mean_output_tokens": sum(
                    int(row.get("output_tokens", 0) or 0) for row in attempts
                )
                / len(attempts),
                "mean_wall_time_s": sum(
                    float(row.get("timing", {}).get("wall_time_s", 0.0) or 0.0)
                    for row in attempts
                )
                / len(attempts),
            }
        )
        output_rows.append(old_summary)

    Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
    write_patch(merged_path, output_rows)


def main() -> None:
    args = parse_args()
    source_attempts = load_attempts(args.source_path)
    source_limit = source_max_new_tokens(source_attempts)
    targets = [
        row
        for row in source_attempts
        if int(row.get("timing", {}).get("actual_generated", 0) or 0) >= source_limit
    ]
    existing_patch_rows = (
        load_jsonl(args.patch_path) if Path(args.patch_path).exists() else []
    )
    existing_keys = {
        row_key(row)
        for row in existing_patch_rows
        if row.get("record_type") == "attempt"
    }
    pending_targets = [row for row in targets if row_key(row) not in existing_keys]

    source_first = source_attempts[0]
    args.model_path = args.model_path or source_first.get("model_path")
    if not args.model_path:
        raise ValueError("Provide --model_path or include model_path in source records")
    args.config_model_name = infer_value(
        args.config_model_name,
        source_first,
        "config_model_name",
        "Qwen3-8B",
    )
    args.attn_type = args.attn_type or source_first.get("attn_type", "Full_Flash_Attn")
    args.dtype = infer_value(args.dtype, source_first, "dtype", "bf16")
    args.temperature = infer_value(args.temperature, source_first, "temperature", 0.7)
    args.top_p = infer_value(args.top_p, source_first, "top_p", 0.95)
    args.top_k = infer_value(args.top_k, source_first, "top_k", 20)
    args.retrieval_budget = infer_value(
        args.retrieval_budget, source_first, "retrieval_budget"
    )
    args.cache_ratio = infer_value(args.cache_ratio, source_first, "cache_ratio")

    print(f"Source attempts: {len(source_attempts)}")
    print(f"Source truncation threshold: {source_limit} tokens")
    print(f"Truncated attempts: {len(targets)}")
    print(f"Already patched: {len(existing_keys)}")
    print(f"Pending reruns: {len(pending_targets)}")
    if args.dry_run:
        print(
            "Target keys:",
            ", ".join(f"{row['id']}:{row['attempt_index']}" for row in targets),
        )
    elif pending_targets:
        rerun_targets(
            targets=pending_targets,
            data_path=args.data_path,
            args=args,
            patch_path=args.patch_path,
        )

    if not args.dry_run:
        merge_results(args.source_path, args.patch_path, args.merged_path)
        print(f"Merged results saved to: {args.merged_path}")
        print(f"Replacement attempts saved to: {args.patch_path}")


if __name__ == "__main__":
    main()
