#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object on line {line_number} of {path}")
            rows.append(row)
    return rows


def load_aime26_data(path: str | Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    for index, row in enumerate(rows, start=1):
        if "problem" not in row:
            raise ValueError(f"Missing problem field at dataset row {index}")
        if "answer" not in row:
            raise ValueError(f"Missing answer field at dataset row {index}")
        row.setdefault("id", index)
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    *,
    start_idx: int = 0,
    num_samples: int = -1,
) -> list[tuple[int, dict[str, Any]]]:
    start = max(int(start_idx), 0)
    selected = rows[start:]
    if num_samples > 0:
        selected = selected[: int(num_samples)]
    return list(enumerate(selected, start=start))


def _boxed_contents(text: str) -> list[str]:
    contents: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            content = text[start : index - 1].strip()
            if content:
                contents.append(content)
    return contents


def _strict_integer(candidate: Any) -> str | None:
    if candidate is None:
        return None
    text = str(candidate).strip()
    text = text.replace("$", "").replace(",", "").replace(" ", "")
    text = text.strip("`*_.$:;")
    if re.fullmatch(r"-?\d+", text):
        return str(int(text))
    if re.fullmatch(r"-?\d+\.0+", text):
        return str(int(float(text)))
    return None


def extract_integer_answer(response: str) -> str | None:
    """Extract the last valid integer answer using the ParisKV AIME policy."""
    text = str(response or "").strip()
    if not text:
        return None

    for candidate in reversed(_boxed_contents(text)):
        answer = _strict_integer(candidate)
        if answer is not None:
            return answer

    patterns = (
        r"(?:S|sum|total|answer|result)\s*=\s*(-?\d+)(?!\d|\.\d)",
        r"(?:the\s+)?(?:final\s+)?(?:answer|solution)\s*(?::|is)\s*(-?\d+)(?!\d|\.\d)",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            answer = _strict_integer(matches[-1].group(1))
            if answer is not None:
                return answer

    equals_matches = list(re.finditer(r"=\s*(-?\d+)(?!\d|\.\d)\s*(?:\.|$)", text))
    if equals_matches:
        answer = _strict_integer(equals_matches[-1].group(1))
        if answer is not None:
            return answer

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return _strict_integer(lines[-1])
    return None


def extract_boxed_aime_answer(response: str) -> str | None:
    """Return the first valid AIME answer explicitly written in ``\\boxed{}``."""
    for candidate in _boxed_contents(str(response or "")):
        answer = _strict_integer(candidate)
        if answer is not None and 0 <= int(answer) <= 999:
            return answer
    return None


def normalize_integer_answer(answer: Any) -> str:
    value = _strict_integer(answer)
    return value or ""


def answers_equal(predicted: Any, target: Any) -> bool:
    predicted_norm = normalize_integer_answer(predicted)
    target_norm = normalize_integer_answer(target)
    return bool(predicted_norm and target_norm and predicted_norm == target_norm)


def completed_problem_ids(path: str | Path) -> set[str]:
    completed: set[str] = set()
    file_path = Path(path)
    if not file_path.exists():
        return completed
    for row in load_jsonl(file_path):
        expected_attempts = int(row.get("n_generate", row.get("pass_k", 1)) or 1)
        attempted = int(row.get("attempted_generate", 0) or 0)
        if row.get("record_type") == "summary" and attempted >= expected_attempts:
            completed.add(str(row.get("id", row.get("dataset_index"))))
    return completed


def latest_attempts_by_problem(
    path: str | Path,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Load the last JSONL record for every ``(problem_id, attempt_index)``.

    AIME26 result files are append-only so an interrupted run may contain an
    earlier failed record followed by a successful retry. The final record is
    the one that represents the attempt during resume and evaluation.
    """
    file_path = Path(path)
    attempts: dict[str, dict[int, dict[str, Any]]] = {}
    if not file_path.exists():
        return attempts
    for row in load_jsonl(file_path):
        if row.get("record_type") != "attempt":
            continue
        item_id = str(row.get("id", row.get("dataset_index")))
        attempt_index = int(row.get("attempt_index", 1) or 1)
        attempts.setdefault(item_id, {})[attempt_index] = row
    return attempts


def summarize_pass_at_k(
    summary_rows: list[dict[str, Any]],
    *,
    pass_k: int,
) -> dict[str, Any]:
    total = len(summary_rows)
    passed = sum(bool(row.get("pass_at_k")) for row in summary_rows)
    attempted = sum(int(row.get("attempted_generate", 0) or 0) for row in summary_rows)
    correct_attempts = sum(int(row.get("correct_count", 0) or 0) for row in summary_rows)
    return {
        "total_problems": total,
        "passed_problems": passed,
        "pass_k": int(pass_k),
        "pass_at_k": passed / total if total else 0.0,
        "pass_at_k_percent": 100.0 * passed / total if total else 0.0,
        "total_attempts": attempted,
        "correct_attempts": correct_attempts,
        "mean_correct_attempts_per_problem": correct_attempts / total if total else 0.0,
    }
