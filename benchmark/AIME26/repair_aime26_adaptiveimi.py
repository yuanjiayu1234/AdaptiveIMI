#!/usr/bin/env python3
"""Repair an interrupted AIME26 AdaptiveIMI pass@8 run.

Correct attempts from the source and old patch are retained, including correct
answers that reached a generation limit. Wrong, failed, and missing attempts
are regenerated. The repair patch is append-only so the command can be
interrupted and resumed safely.
"""

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
        description="Repair AdaptiveIMI AIME26 results while preserving valid attempts."
    )
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--old_patch_path", default=None)
    parser.add_argument("--repair_patch_path", required=True)
    parser.add_argument("--merged_path", required=True)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--config_model_name", default=None)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
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
    parser.add_argument(
        "--parallel_mode",
        default="single",
        choices=("auto", "single", "layer", "tp"),
    )
    parser.add_argument(
        "--weight_loading",
        default="hf",
        choices=("hf", "auto", "accelerate_meta", "streaming_safetensors"),
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--rerun_seed_base", type=int, default=100000)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("id", row.get("dataset_index"))), int(row["attempt_index"]))


def load_attempt_map(path: str | Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("record_type") == "attempt":
            result[row_key(row)] = row
    return result


def row_is_valid(row: dict[str, Any]) -> bool:
    if row.get("error"):
        return False
    target = row.get("target_answer", row.get("answer"))
    predicted = row.get("predicted_answer")
    if not answers_equal(predicted, target):
        return False
    timing = row.get("timing", {})
    generated = int(timing.get("actual_generated", row.get("output_tokens", 0)) or 0)
    return generated > 0


def build_expected_keys(data: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    expected: list[tuple[str, int, int]] = []
    for dataset_index, item in enumerate(data):
        item_id = str(item.get("id", dataset_index + 1))
        for attempt_index in range(1, 9):
            expected.append((item_id, attempt_index, dataset_index))
    return expected


def select_keep_and_pending(
    data: list[dict[str, Any]],
    source: dict[tuple[str, int], dict[str, Any]],
    old_patch: dict[tuple[str, int], dict[str, Any]],
    repair_patch: dict[tuple[str, int], dict[str, Any]],
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    list[tuple[str, int, int]],
]:
    kept: dict[tuple[str, int], dict[str, Any]] = {}
    pending: list[tuple[str, int, int]] = []
    for item_id, attempt_index, dataset_index in build_expected_keys(data):
        key = (item_id, attempt_index)
        candidates = [repair_patch.get(key), old_patch.get(key), source.get(key)]
        valid = next((row for row in candidates if row and row_is_valid(row)), None)
        if valid is not None:
            kept[key] = valid
        else:
            pending.append((item_id, attempt_index, dataset_index))
    return kept, pending


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
    source_attempt: dict[str, Any] | None,
    item: dict[str, Any],
    dataset_index: int,
    attempt_index: int,
    response: str,
    predicted: str | None,
    timing: dict[str, Any],
    output_tokens: int,
    error: str | None,
    model_path: str,
    args: argparse.Namespace,
    input_tokens: int,
    seed: int,
) -> dict[str, Any]:
    target = str(item["answer"])
    return {
        "record_type": "attempt",
        "dataset_index": dataset_index,
        "id": str(item.get("id", dataset_index + 1)),
        "attempt_index": attempt_index,
        "n_generate": 8,
        "pass_k": 8,
        "problem": str(item["problem"]).strip(),
        "target_answer": target,
        "predicted_answer": predicted,
        "response": response,
        "is_correct": answers_equal(predicted, target),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "timing": timing,
        "error": error,
        "model_path": model_path,
        "attn_type": "AdaptiveIMI",
        "seed": seed,
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
        "repair_reason": "missing_or_wrong_or_truncated_source_attempt",
        "source_attempt_present": source_attempt is not None,
    }


def run_pending(
    pending: list[tuple[str, int, int]],
    data: list[dict[str, Any]],
    args: argparse.Namespace,
    source: dict[tuple[str, int], dict[str, Any]],
    old_patch: dict[tuple[str, int], dict[str, Any]],
) -> None:
    data_by_index = {index: item for index, item in enumerate(data)}
    grouped: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    for target in pending:
        grouped[target[2]].append(target)

    model_path = resolve_model_path(args.model_path)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
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

    patch_path = Path(args.repair_patch_path)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    with patch_path.open("a", encoding="utf-8") as patch_file:
        completed = 0
        for dataset_index in sorted(grouped):
            item = data_by_index[dataset_index]
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
                    f"Problem {item.get('id', dataset_index + 1)} requires "
                    f"{input_tokens + args.max_new_tokens} tokens, exceeding "
                    f"--max_model_len={args.max_model_len}"
                )

            attn_config = generate_config(
                args.config_model_name,
                input_tokens,
                "AdaptiveIMI",
                retrieval_budget=args.retrieval_budget,
                cache_ratio=args.cache_ratio,
                partition_path=args.partition_path,
                hybrid_index=build_hybrid_index_config(args),
                gpu_only=args.gpu_only,
                imi_engine=args.imi_engine,
                prefill_chunk_size=args.prefill_chunk_size,
            )

            for item_id, attempt_index, _ in sorted(
                grouped[dataset_index], key=lambda value: value[1]
            ):
                key = (item_id, attempt_index)
                prior = old_patch.get(key) or source.get(key)
                # Invalid attempts are replaced by fresh stochastic samples.
                # The derived seed is stable across interrupted/resumed runs.
                seed = (
                    args.rerun_seed_base
                    + dataset_index * 1000
                    + (attempt_index - 1)
                )
                set_seed(seed)
                started = time.perf_counter()
                output_ids = None
                response = ""
                error = None
                try:
                    output_ids = llm.generate(
                        attention_type="AdaptiveIMI",
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
                        f"Problem {item_id} attempt {attempt_index} failed: {error}",
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
                    source_attempt=prior,
                    item=item,
                    dataset_index=dataset_index,
                    attempt_index=attempt_index,
                    response=response,
                    predicted=predicted,
                    timing=timing,
                    output_tokens=output_tokens,
                    error=error,
                    model_path=model_path,
                    args=args,
                    input_tokens=input_tokens,
                    seed=seed,
                )
                patch_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                patch_file.flush()
                completed += 1
                print(
                    f"[repair] {completed}/{len(pending)} "
                    f"id={item_id} attempt={attempt_index} "
                    f"generated={timing.get('actual_generated', 0)} "
                    f"correct={result['is_correct']}",
                    flush=True,
                )

            cleanup_model_after_problem(llm)


def write_merged(
    path: str | Path,
    data: list[dict[str, Any]],
    kept: dict[tuple[str, int], dict[str, Any]],
    repair_patch: dict[tuple[str, int], dict[str, Any]],
) -> int:
    final_rows: list[dict[str, Any]] = []
    complete = 0
    for dataset_index, item in enumerate(data):
        item_id = str(item.get("id", dataset_index + 1))
        attempts: list[dict[str, Any]] = []
        for attempt_index in range(1, 9):
            key = (item_id, attempt_index)
            row = repair_patch.get(key)
            if row is None or not row_is_valid(row):
                row = kept.get(key)
            if row is None or not row_is_valid(row):
                continue
            attempts.append(row)
        attempts.sort(key=lambda row: int(row["attempt_index"]))
        final_rows.extend(attempts)
        if len(attempts) == 8:
            complete += 1
            correct_count = sum(
                answers_equal(row.get("predicted_answer"), row.get("target_answer"))
                for row in attempts
            )
            final_rows.append(
                {
                    "record_type": "summary",
                    "dataset_index": dataset_index,
                    "id": item_id,
                    "n_generate": 8,
                    "pass_k": 8,
                    "attempted_generate": 8,
                    "correct_count": correct_count,
                    "pass_at_k": correct_count > 0,
                    "problem": str(item["problem"]).strip(),
                    "target_answer": str(item["answer"]),
                    "mean_output_tokens": sum(
                        int(row.get("output_tokens", 0) or 0) for row in attempts
                    )
                    / 8.0,
                    "mean_wall_time_s": sum(
                        float(row.get("timing", {}).get("wall_time_s", 0.0) or 0.0)
                        for row in attempts
                    )
                    / 8.0,
                    "attn_type": "AdaptiveIMI",
                }
            )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fout:
        for row in final_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return complete


def main() -> None:
    args = parse_args()
    data = load_aime26_data(args.data_path)
    source = load_attempt_map(args.source_path)
    old_patch = load_attempt_map(args.old_patch_path)
    repair_patch = load_attempt_map(args.repair_patch_path)

    source_first = next(iter(source.values()), {})
    args.model_path = args.model_path or source_first.get("model_path")
    if not args.model_path:
        raise ValueError("Provide --model_path or include model_path in source records")
    args.config_model_name = infer_value(
        args.config_model_name, source_first, "config_model_name", "Qwen3-8B"
    )
    args.dtype = infer_value(args.dtype, source_first, "dtype", "bf16")
    args.temperature = infer_value(args.temperature, source_first, "temperature", 0.7)
    args.top_p = infer_value(args.top_p, source_first, "top_p", 0.95)
    args.top_k = infer_value(args.top_k, source_first, "top_k", 20)
    args.retrieval_budget = infer_value(
        args.retrieval_budget, source_first, "retrieval_budget", 0.1
    )
    args.cache_ratio = infer_value(args.cache_ratio, source_first, "cache_ratio", 0.1)

    kept, pending = select_keep_and_pending(data, source, old_patch, repair_patch)
    print(f"Expected attempts: {len(data) * 8}")
    print(f"Valid attempts retained: {len(kept)}")
    print(f"Attempts to regenerate: {len(pending)}")
    print(f"Model max length: {args.max_model_len}")
    print(f"Generation limit: {args.max_new_tokens}")
    print(f"Attention: AdaptiveIMI")
    print(f"Retrieval budget: {args.retrieval_budget}")
    print(f"Cache ratio: {args.cache_ratio}")

    if args.dry_run:
        print(
            "Pending keys:",
            ", ".join(f"{item_id}:{attempt}" for item_id, attempt, _ in pending),
        )
        return

    if pending:
        run_pending(pending, data, args, source, old_patch)

    repair_patch = load_attempt_map(args.repair_patch_path)
    kept, pending = select_keep_and_pending(data, source, old_patch, repair_patch)
    complete = write_merged(args.merged_path, data, kept, repair_patch)
    print(f"Complete problems: {complete}/{len(data)}")
    print(f"Remaining attempts: {len(pending)}")
    print(f"Merged output: {args.merged_path}")
    if pending:
        print("The merged file is incomplete; rerun the same command to continue.")


if __name__ == "__main__":
    main()
