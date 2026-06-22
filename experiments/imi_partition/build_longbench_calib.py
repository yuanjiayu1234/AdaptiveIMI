from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _render_longbench(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    context = item.get("context")
    if isinstance(context, str) and context.strip():
        parts.append(context.strip())
    question = item.get("question")
    if isinstance(question, str) and question.strip():
        parts.append("Question:\n" + question.strip())
    choices = []
    for key in ("choice_A", "choice_B", "choice_C", "choice_D", "choice_E"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            choices.append(f"{key[-1]}. {value.strip()}")
    if choices:
        parts.append("Choices:\n" + "\n".join(choices))
    return "\n\n".join(parts).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a small long-context LongBench calibration JSONL.")
    parser.add_argument("--input", type=Path, default=Path("/root/LongBench/data.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/imi_partition/calib_longbench_long.jsonl"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--min-chars", type=int, default=200000)
    parser.add_argument("--max-chars", type=int, default=800000)
    parser.add_argument("--exclude-ids", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected LongBench JSON list: {args.input}")
    exclude_ids = {item.strip() for item in str(args.exclude_ids).replace(",", " ").split() if item.strip()}
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("_id") or item.get("id") or "")
        if source_id in exclude_ids:
            continue
        text = _render_longbench(item)
        if len(text) < int(args.min_chars):
            continue
        if int(args.max_chars) > 0 and len(text) > int(args.max_chars):
            text = text[: int(args.max_chars)].rstrip()
        rows.append(
            {
                "input": text,
                "source_dataset": "longbench_long_calib",
                "source_hf_dataset": "LongBench/data.json",
                "source_subset": item.get("domain"),
                "source_split": "calibration_subset",
                "source_id": source_id,
                "chars": len(text),
                "longbench_domain": item.get("domain"),
                "longbench_sub_domain": item.get("sub_domain"),
                "longbench_length": item.get("length"),
            }
        )
    rows.sort(key=lambda row: int(row["chars"]), reverse=True)
    rows = rows[: max(int(args.limit), 0)]
    if not rows:
        raise SystemExit(
            f"[longbench-calib] no rows found with min_chars={args.min_chars}; "
            "lower --min-chars or check input"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[longbench-calib] wrote {args.out} rows={len(rows)} "
        f"chars_min={min(row['chars'] for row in rows)} chars_max={max(row['chars'] for row in rows)}"
    )
    for row in rows:
        print(
            f"[longbench-calib] row id={row['source_id']} chars={row['chars']} "
            f"domain={row.get('longbench_domain')} sub_domain={row.get('longbench_sub_domain')}"
        )


if __name__ == "__main__":
    main()
