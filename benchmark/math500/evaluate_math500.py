#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from math500_utils import extract_final_answer, load_jsonl, math_answers_equal, summarize_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Math-500 prediction JSONL.")
    parser.add_argument("prediction_path", type=str)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument(
        "--reextract",
        action="store_true",
        help="Re-extract predicted_answer from each stored response before judging.",
    )
    args = parser.parse_args()

    prediction_path = Path(args.prediction_path)
    rows = load_jsonl(prediction_path)
    for row in rows:
        if args.reextract and row.get("response"):
            predicted = extract_final_answer(str(row["response"]))
            row["predicted_answer"] = predicted
        else:
            predicted = str(row.get("predicted_answer", row.get("pred", "")))
        target = str(row.get("target_answer", row.get("answer", "")))
        row["judge"] = math_answers_equal(predicted, target)

    summary = summarize_rows(rows)
    summary["prediction_path"] = str(prediction_path)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else prediction_path.with_suffix(".summary.json")
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
