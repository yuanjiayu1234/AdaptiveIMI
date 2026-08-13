#!/usr/bin/env python3
"""Clean and merge AIME attempt results without retaining truncated outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from aime26_utils import answers_equal, load_aime26_data, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge AIME source and rerun results, filtering truncation."
    )
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--patch_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--source_max_new_tokens", type=int, default=8192)
    parser.add_argument("--patch_max_new_tokens", type=int, default=16384)
    return parser.parse_args()


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return (
        str(row.get("id", row.get("dataset_index"))),
        int(row["attempt_index"]),
    )


def generated_tokens(row: dict[str, Any]) -> int:
    timing = row.get("timing") or {}
    return int(timing.get("actual_generated", row.get("output_tokens", 0)) or 0)


def valid(row: dict[str, Any], max_new_tokens: int) -> bool:
    return not row.get("error") and 0 < generated_tokens(row) < max_new_tokens


def attempt_map(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("record_type") != "attempt":
            continue
        key = row_key(row)
        if key in result:
            raise ValueError(f"Duplicate attempt key in {path}: {key}")
        result[key] = row
    return result


def summary_row(
    item: dict[str, Any],
    dataset_index: int,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    correct_count = sum(
        answers_equal(
            row.get("predicted_answer"),
            row.get("target_answer", row.get("answer")),
        )
        for row in attempts
    )
    first = attempts[0]
    return {
        "record_type": "summary",
        "dataset_index": dataset_index,
        "id": str(item.get("id", dataset_index + 1)),
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
            float((row.get("timing") or {}).get("wall_time_s", 0.0) or 0.0)
            for row in attempts
        )
        / 8.0,
        "attn_type": first.get("attn_type", "Full_Flash_Attn"),
        "cleaning_policy": (
            "prefer_natural_patch_then_natural_source; "
            "exclude_error_and_limit_reached"
        ),
    }


def main() -> None:
    args = parse_args()
    source = attempt_map(args.source_path)
    patch = attempt_map(args.patch_path)

    if args.data_path:
        data = load_aime26_data(args.data_path)
    else:
        data = []
        for key in sorted(source, key=lambda value: (int(value[0]), value[1])):
            row = source[key]
            data.append(
                {
                    "id": key[0],
                    "problem": row.get("problem", ""),
                    "answer": row.get("target_answer", row.get("answer", "")),
                }
            )
        data_by_id = {str(item["id"]): item for item in data}
        data = [
            data_by_id[item_id]
            for item_id in sorted(data_by_id, key=lambda value: int(value))
        ]

    selected: dict[tuple[str, int], dict[str, Any]] = {}
    patch_selected = 0
    source_selected = 0
    excluded = 0
    for item_id, attempt_index in sorted(
        set(source) | set(patch), key=lambda value: (int(value[0]), value[1])
    ):
        key = (item_id, attempt_index)
        patch_row = patch.get(key)
        source_row = source.get(key)
        if patch_row is not None and valid(patch_row, args.patch_max_new_tokens):
            selected[key] = patch_row
            patch_selected += 1
        elif source_row is not None and valid(
            source_row, args.source_max_new_tokens
        ):
            selected[key] = source_row
            source_selected += 1
        else:
            excluded += 1

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, row in selected.items():
        by_id[key[0]].append(row)

    output_rows: list[dict[str, Any]] = []
    complete_problems = 0
    for dataset_index, item in enumerate(data):
        item_id = str(item.get("id", dataset_index + 1))
        attempts = sorted(
            by_id.get(item_id, []),
            key=lambda row: int(row["attempt_index"]),
        )
        output_rows.extend(attempts)
        if len(attempts) == 8 and {
            int(row["attempt_index"]) for row in attempts
        } == set(range(1, 9)):
            output_rows.append(summary_row(item, dataset_index, attempts))
            complete_problems += 1

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for row in output_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {
        item_id: len(rows)
        for item_id, rows in sorted(
            by_id.items(), key=lambda value: int(value[0])
        )
    }
    print(f"Source attempts: {len(source)}")
    print(f"Patch attempts: {len(patch)}")
    print(f"Selected from patch: {patch_selected}")
    print(f"Selected from source: {source_selected}")
    print(f"Excluded truncated/error attempts: {excluded}")
    print(f"Clean attempts: {len(selected)}")
    print(f"Complete problems: {complete_problems}/{len(data)}")
    print(f"Per-problem valid counts: {counts}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
