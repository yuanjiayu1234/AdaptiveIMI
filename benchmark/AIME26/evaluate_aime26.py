#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aime26_utils import (
    answers_equal,
    extract_integer_answer,
    load_jsonl,
    summarize_pass_at_k,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AIME26 pass@k JSONL results.")
    parser.add_argument("prediction_path", type=str)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--pass_k", type=int, default=None)
    parser.add_argument("--reextract", action="store_true")
    parser.add_argument(
        "--complete_only",
        action="store_true",
        help="Only evaluate problems with at least pass_k attempt records.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.prediction_path)
    pass_k = args.pass_k
    if pass_k is None:
        pass_k = max(
            [int(row.get("pass_k", row.get("n_generate", 1)) or 1) for row in rows]
            or [1]
        )
    attempts: dict[str, dict[int, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("id", row.get("dataset_index")))
        if row.get("record_type") == "attempt":
            if row.get("error"):
                continue
            if args.reextract and row.get("response"):
                row["predicted_answer"] = extract_integer_answer(str(row["response"]))
            row["is_correct"] = answers_equal(
                row.get("predicted_answer"),
                row.get("target_answer", row.get("answer")),
            )
            attempt_index = int(row.get("attempt_index", 1) or 1)
            # JSONL files are append-only during resume. Keep the final record
            # for a repeated (problem, attempt) key rather than counting it twice.
            attempts.setdefault(key, {})[attempt_index] = row
        elif row.get("record_type") == "summary":
            summaries[key] = row

    if not attempts and summaries:
        summary_rows = [
            row
            for row in summaries.values()
            if not args.complete_only
            or int(row.get("attempted_generate", 0) or 0) >= pass_k
        ]
    else:
        summary_rows = []
        for key, attempt_map in attempts.items():
            attempt_rows = [attempt_map[index] for index in sorted(attempt_map)]
            if args.complete_only and len(attempt_rows) < pass_k:
                continue
            first = attempt_rows[0]
            correct_count = sum(bool(row.get("is_correct")) for row in attempt_rows)
            summary_rows.append(
                {
                    "record_type": "summary",
                    "id": key,
                    "dataset_index": first.get("dataset_index"),
                    "attempted_generate": len(attempt_rows),
                    "correct_count": correct_count,
                    "pass_at_k": correct_count > 0,
                }
            )

    result = summarize_pass_at_k(summary_rows, pass_k=pass_k)
    result["prediction_path"] = str(args.prediction_path)
    result["num_summary_rows"] = len(summary_rows)

    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else Path(args.prediction_path).with_suffix(".summary.json")
    )
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
