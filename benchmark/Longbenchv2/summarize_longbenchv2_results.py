#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable


Group = tuple[str, Callable[[dict], bool]]

GROUPS: list[Group] = [
    ("Overall", lambda row: True),
    ("Easy", lambda row: row.get("difficulty") == "easy"),
    ("Hard", lambda row: row.get("difficulty") == "hard"),
    ("Short", lambda row: row.get("length") == "short"),
    ("Medium", lambda row: row.get("length") == "medium"),
    ("Long", lambda row: row.get("length") == "long"),
    ("short-easy", lambda row: row.get("length") == "short" and row.get("difficulty") == "easy"),
    ("short-hard", lambda row: row.get("length") == "short" and row.get("difficulty") == "hard"),
    ("medium-easy", lambda row: row.get("length") == "medium" and row.get("difficulty") == "easy"),
    ("medium-hard", lambda row: row.get("length") == "medium" and row.get("difficulty") == "hard"),
    ("long-easy", lambda row: row.get("length") == "long" and row.get("difficulty") == "easy"),
    ("long-hard", lambda row: row.get("length") == "long" and row.get("difficulty") == "hard"),
]


def load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return data if isinstance(data, list) else [data]


def iter_files(paths: list[str]):
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(child for child in path.iterdir() if child.suffix in {".json", ".jsonl"})
        else:
            yield path


def score(rows: list[dict], predicate: Callable[[dict], bool]) -> tuple[int, int]:
    selected = [row for row in rows if predicate(row)]
    return sum(1 for row in selected if bool(row.get("judge"))), len(selected)


def format_score(correct: int, total: int, decimals: int, with_counts: bool) -> str:
    if total == 0:
        return "NA"
    value = f"{100.0 * correct / total:.{decimals}f}"
    return f"{value}({correct}/{total})" if with_counts else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LongBench v2 jsonl predictions.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--with-counts", action="store_true")
    args = parser.parse_args()

    print("\t".join(["source", "file", "n"] + [name for name, _ in GROUPS]))
    for path in iter_files(args.paths):
        rows = load_rows(path)
        values = [str(path.parent), path.name, str(len(rows))]
        values.extend(
            format_score(*score(rows, predicate), args.decimals, args.with_counts)
            for _, predicate in GROUPS
        )
        print("\t".join(values))


if __name__ == "__main__":
    main()
