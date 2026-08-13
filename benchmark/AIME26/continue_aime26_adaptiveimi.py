#!/usr/bin/env python3
"""Replay selected AdaptiveIMI attempts against a Full Attention target file.

The target file defines the exact (problem_id, attempt_index) whitelist to
evaluate. Each replay uses the target attempt's seed, sampling settings, and
generation limit. Length-limited output remains a valid benchmark attempt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from aime26_utils import answers_equal, extract_integer_answer, load_aime26_data, load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "benchmark" / "AIME26" / "data" / "aime2026.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay comparable AdaptiveIMI AIME26 attempts from a target file."
    )
    parser.add_argument("--seed_path", required=True)
    parser.add_argument("--patch_path", required=True)
    parser.add_argument("--merged_path", required=True)
    parser.add_argument(
        "--target_path",
        required=True,
        help="JSONL whitelist, typically the cleaned Full Attention result.",
    )
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_model_name", default="Qwen3-8B")
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--max_new_tokens", type=int, default=39000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--retrieval_budget", type=float, default=0.1)
    parser.add_argument("--cache_ratio", type=float, default=0.1)
    parser.add_argument("--partition_path", default=None)
    parser.add_argument("--imi_engine", default="hybrid_8way_v2")
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument("--prefill_chunk_size", type=int, default=None)
    parser.add_argument("--parallel_mode", default="single",
                        choices=("auto", "single", "layer", "tp"))
    parser.add_argument("--weight_loading", default="hf",
                        choices=("hf", "auto", "accelerate_meta", "streaming_safetensors"))
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("id", row.get("dataset_index"))), int(row["attempt_index"]))


def load_attempt_map(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    file_path = Path(path)
    if not file_path.exists():
        return result
    for row in load_jsonl(file_path):
        if row.get("record_type") == "attempt":
            result[key(row)] = row
    return result


def is_finished(row: dict[str, Any], max_new_tokens: int) -> bool:
    if row.get("error"):
        return False
    timing = row.get("timing", {})
    generated = int(timing.get("actual_generated", 0) or 0)
    return 0 < generated < max_new_tokens


def is_reusable_seed(row: dict[str, Any], seed_max_new_tokens: int) -> bool:
    """Reuse any valid natural-ended attempt, including a wrong answer."""
    return not row.get("error") and is_finished(row, seed_max_new_tokens)


def is_correct(row: dict[str, Any]) -> bool:
    return answers_equal(
        row.get("predicted_answer"),
        row.get("target_answer", row.get("answer")),
    )


def reference_config(row: dict[str, Any]) -> dict[str, Any]:
    """Return the per-attempt generation settings recorded by Full Attention."""
    config = dict(row.get("benchmark_config") or {})
    required = ("max_model_len", "max_new_tokens", "temperature", "top_p", "top_k")
    missing = [name for name in required if config.get(name) is None]
    if missing:
        raise ValueError(
            f"Reference attempt id={row.get('id')} attempt={row.get('attempt_index')} "
            f"is missing benchmark_config fields: {', '.join(missing)}"
        )
    return config


def matches_reference(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> bool:
    """Only reuse an AdaptiveIMI attempt that is comparable to the reference."""
    if candidate.get("attn_type") != "AdaptiveIMI" or candidate.get("error"):
        return False
    if int(candidate.get("seed", -1)) != int(reference.get("seed", -2)):
        return False

    candidate_config = dict(candidate.get("benchmark_config") or {})
    reference_settings = reference_config(reference)
    for name in (
        "max_model_len",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "enable_thinking",
    ):
        if name in reference_settings and candidate_config.get(name) != reference_settings.get(name):
            return False
    generated = int((candidate.get("timing") or {}).get("actual_generated", 0) or 0)
    return 0 < generated <= int(reference_settings["max_new_tokens"])


def expected_keys(data: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    result = []
    for dataset_index, item in enumerate(data):
        item_id = str(item.get("id", dataset_index + 1))
        for attempt_index in range(1, 9):
            result.append((item_id, attempt_index, dataset_index))
    return result


def build_attempt(
    *,
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
    retry_index: int,
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
        "retry_index": retry_index,
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
            "enable_thinking": True,
        },
        "continuation_policy": "reject_actual_generated_at_max_new_tokens",
    }


def write_merged(
    path: str | Path,
    data: list[dict[str, Any]],
    target_keys: set[tuple[str, int]],
    seed_map: dict[tuple[str, int], dict[str, Any]],
    patch_map: dict[tuple[str, int], dict[str, Any]],
    target_map: dict[tuple[str, int], dict[str, Any]],
) -> tuple[int, int]:
    rows: list[dict[str, Any]] = []
    complete = 0
    accepted_attempts = 0
    for dataset_index, item in enumerate(data):
        item_id = str(item.get("id", dataset_index + 1))
        wanted_attempts = {
            attempt_index
            for target_id, attempt_index in target_keys
            if target_id == item_id
        }
        if not wanted_attempts:
            continue
        attempts = []
        for attempt_index in sorted(wanted_attempts):
            item_key = (item_id, attempt_index)
            patch_candidate = patch_map.get(item_key)
            seed_candidate = seed_map.get(item_key)
            reference = target_map[item_key]
            if patch_candidate is not None and matches_reference(patch_candidate, reference):
                candidate = patch_candidate
            elif seed_candidate is not None and matches_reference(seed_candidate, reference):
                candidate = seed_candidate
            else:
                candidate = None
            if candidate is not None:
                attempts.append(candidate)
        attempts.sort(key=lambda row: int(row["attempt_index"]))
        rows.extend(attempts)
        accepted_attempts += len(attempts)
        if wanted_attempts == set(range(1, 9)) and len(attempts) == 8:
            complete += 1
            correct_count = sum(is_correct(row) for row in attempts)
            rows.append(
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
                    ) / 8.0,
                    "mean_wall_time_s": sum(
                        float(row.get("timing", {}).get("wall_time_s", 0.0) or 0.0)
                        for row in attempts
                    ) / 8.0,
                    "attn_type": "AdaptiveIMI",
                }
            )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return complete, accepted_attempts


def main() -> None:
    args = parse_args()
    data = load_aime26_data(args.data_path)
    seed_map = load_attempt_map(args.seed_path)
    patch_map = load_attempt_map(args.patch_path)
    target_map = load_attempt_map(args.target_path)
    if not target_map:
        raise ValueError(f"No target attempt records found in {args.target_path}")

    data_by_id = {
        str(item.get("id", dataset_index + 1)): item
        for dataset_index, item in enumerate(data)
    }
    missing_ids = sorted(
        {item_id for item_id, _ in target_map if item_id not in data_by_id},
        key=lambda value: int(value),
    )
    if missing_ids:
        raise ValueError(
            f"Target file contains IDs absent from data: {', '.join(missing_ids)}"
        )
    target_keys = set(target_map)
    target_problem_ids = {item_id for item_id, _ in target_keys}

    pending = []
    for item_id, attempt_index in sorted(
        target_keys, key=lambda value: (int(value[0]), value[1])
    ):
        dataset_index = next(
            index
            for index, item in enumerate(data)
            if str(item.get("id", index + 1)) == item_id
        )
        item_key = (item_id, attempt_index)
        patch_candidate = patch_map.get(item_key)
        seed_candidate = seed_map.get(item_key)
        reference = target_map[item_key]
        patch_ok = patch_candidate is not None and matches_reference(patch_candidate, reference)
        seed_ok = seed_candidate is not None and matches_reference(seed_candidate, reference)
        if not patch_ok and not seed_ok:
            pending.append((item_id, attempt_index, dataset_index))

    print(f"Target attempts: {len(target_keys)}")
    print(f"Target problems: {len(target_problem_ids)}")
    print(
        "Target IDs: "
        + ",".join(sorted(target_problem_ids, key=lambda value: int(value)))
    )
    print(f"Seed attempts: {len(seed_map)}")
    print(
        "Reference-compatible seed attempts: "
        f"{sum(matches_reference(seed_map[item_key], target_map[item_key]) for item_key in target_keys if item_key in seed_map)}"
    )
    print(
        "Reference-compatible continuation attempts: "
        f"{sum(matches_reference(patch_map[item_key], target_map[item_key]) for item_key in target_keys if item_key in patch_map)}"
    )
    print(f"Pending attempts: {len(pending)}")
    print(
        "Reference max_model_len values: "
        + ", ".join(str(value) for value in sorted({int(reference_config(row)["max_model_len"]) for row in target_map.values()}))
    )
    print(
        "Reference max_new_tokens values: "
        + ", ".join(str(value) for value in sorted({int(reference_config(row)["max_new_tokens"]) for row in target_map.values()}))
    )

    if args.dry_run:
        return

    import torch
    from aime26_inference import (
        apply_chat_template,
        build_hybrid_index_config,
        build_prompt,
        cleanup_model_after_problem,
        load_model,
        load_tokenizer,
        set_seed,
    )
    from config import generate_config

    data_by_index = {index: item for index, item in enumerate(data)}
    grouped: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    for target in pending:
        grouped[target[2]].append(target)

    if pending:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        model_max_length = max(
            int(reference_config(target_map[(item_id, attempt_index)])["max_model_len"])
            for item_id, attempt_index, _ in pending
        )
        tokenizer = load_tokenizer(args.model_path)
        llm = load_model(
            args.model_path,
            model_max_length,
            dtype,
            args.device,
            tokenizer=tokenizer,
            parallel_mode=args.parallel_mode,
            weight_loading=args.weight_loading,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        patch_path = Path(args.patch_path)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        with patch_path.open("a", encoding="utf-8") as patch_file:
            for dataset_index in sorted(grouped):
                item = data_by_index[dataset_index]
                thinking_settings = {
                    bool(reference_config(target_map[(item_id, attempt_index)]).get("enable_thinking", True))
                    for item_id, attempt_index, _ in grouped[dataset_index]
                }
                if len(thinking_settings) != 1:
                    raise ValueError(
                        f"id={item.get('id', dataset_index + 1)} has mixed "
                        "enable_thinking settings in the target attempts"
                    )
                prompt = apply_chat_template(
                    tokenizer,
                    build_prompt(str(item["problem"]).strip()),
                    enable_thinking=thinking_settings.pop(),
                )
                encoded = tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = encoded.input_ids
                attention_mask = encoded.attention_mask
                input_tokens = int(input_ids.shape[1])
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
                    reference = target_map[(item_id, attempt_index)]
                    replay_config = reference_config(reference)
                    max_model_len = int(replay_config["max_model_len"])
                    max_new_tokens = int(replay_config["max_new_tokens"])
                    if input_tokens + max_new_tokens > max_model_len:
                        raise ValueError(
                            f"id={item_id} attempt={attempt_index} needs "
                            f"{input_tokens + max_new_tokens} tokens, exceeding "
                            f"reference max_model_len={max_model_len}"
                        )
                    # Cache allocation reads this field inside generate(). Keep
                    # the actual AdaptiveIMI cache capacity aligned with the
                    # per-attempt Full Attention reference, not just the JSONL
                    # metadata written after generation.
                    llm.max_length = max_model_len
                    accepted = False
                    # A retry with a different seed is not comparable to the Full
                    # Attention attempt. Run the reference seed exactly once.
                    for retry_index in range(1):
                        seed = int(reference["seed"])
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
                                max_new_length=max_new_tokens,
                                attn_config=attn_config,
                                do_sample=float(replay_config["temperature"]) > 1e-5,
                                temperature=float(replay_config["temperature"]),
                                top_p=float(replay_config["top_p"]),
                                top_k=int(replay_config["top_k"]),
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

                        elapsed = time.perf_counter() - started
                        timing = dict(getattr(llm, "last_timing_stats", {}) or {})
                        generated = int(timing.get("actual_generated", 0) or 0)
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
                        row = build_attempt(
                            item=item,
                            dataset_index=dataset_index,
                            attempt_index=attempt_index,
                            response=response,
                            predicted=predicted,
                            timing=timing,
                            output_tokens=output_tokens,
                            error=error,
                            model_path=args.model_path,
                            args=args,
                            input_tokens=input_tokens,
                            seed=seed,
                            retry_index=retry_index,
                        )
                        row["benchmark_config"].update(
                            {
                                "max_model_len": max_model_len,
                                "max_new_tokens": max_new_tokens,
                                "temperature": float(replay_config["temperature"]),
                                "top_p": float(replay_config["top_p"]),
                                "top_k": int(replay_config["top_k"]),
                                "enable_thinking": bool(replay_config.get("enable_thinking", True)),
                            }
                        )
                        row["reference_attempt"] = {
                            "id": str(reference.get("id")),
                            "attempt_index": int(reference["attempt_index"]),
                            "attn_type": reference.get("attn_type"),
                        }
                        if error or not (0 < generated <= max_new_tokens):
                            print(
                                f"[discard] id={item_id} attempt={attempt_index} "
                                f"retry={retry_index + 1} generated={generated} "
                                f"reason={'error' if error else 'empty_output'}",
                                flush=True,
                            )
                            continue
                        row["completion_status"] = (
                            "length_limit"
                            if generated == max_new_tokens
                            else "natural_stop"
                        )
                        patch_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        patch_file.flush()
                        accepted = True
                        print(
                            f"[accepted] id={item_id} attempt={attempt_index} "
                            f"generated={generated} status={row['completion_status']} "
                            f"correct={row['is_correct']}",
                            flush=True,
                        )
                        break
                    if not accepted:
                        print(
                            f"[pending] id={item_id} attempt={attempt_index}: "
                            f"all retries were rejected",
                            file=sys.stderr,
                        )
                cleanup_model_after_problem(llm)

    patch_map = load_attempt_map(args.patch_path)
    complete, accepted = write_merged(
        args.merged_path,
        data,
        target_keys,
        seed_map,
        patch_map,
        target_map,
    )
    print(f"Complete target problems: {complete}/{len(target_problem_ids)}")
    print(f"Accepted attempts in merged file: {accepted}/{len(target_keys)}")
    print(f"Merged output: {args.merged_path}")


if __name__ == "__main__":
    main()
