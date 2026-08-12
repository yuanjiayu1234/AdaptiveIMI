#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


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


def select_rows(
    rows: list[dict[str, Any]],
    *,
    start: int = 0,
    max_samples: int = -1,
    sample_indices: str | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    if sample_indices:
        selected: list[tuple[int, dict[str, Any]]] = []
        for raw_index in sample_indices.split(","):
            raw_index = raw_index.strip()
            if not raw_index:
                continue
            index = int(raw_index)
            if index < 0 or index >= len(rows):
                raise IndexError(f"Sample index {index} is outside [0, {len(rows) - 1}]")
            selected.append((index, rows[index]))
        if not selected:
            raise ValueError("--sample_indices did not contain a valid index")
        return selected

    start = max(int(start), 0)
    selected_rows = rows[start:]
    if max_samples > 0:
        selected_rows = selected_rows[: int(max_samples)]
    return list(enumerate(selected_rows, start=start))


def boxed_contents(text: str) -> list[str]:
    """Extract balanced contents from the last-level ``\boxed{...}`` commands."""
    contents: list[str] = []
    marker = r"\boxed"
    search_start = 0
    while True:
        marker_index = text.find(marker, search_start)
        if marker_index < 0:
            break
        brace_index = marker_index + len(marker)
        while brace_index < len(text) and text[brace_index].isspace():
            brace_index += 1
        if brace_index >= len(text) or text[brace_index] != "{":
            search_start = marker_index + len(marker)
            continue

        depth = 0
        for index in range(brace_index, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[brace_index + 1 : index])
                    search_start = index + 1
                    break
        else:
            break
    return contents


def _last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def extract_final_answer(response: str) -> str:
    """Extract the final answer from a MATH-style chain-of-thought response."""
    text = str(response or "").replace("\r\n", "\n").strip()
    if not text:
        return ""

    # Qwen-style reasoning models may place the answer after </think>.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()

    boxes = boxed_contents(text)
    if boxes:
        return boxes[-1].strip()

    patterns = (
        r"therefore\s*,?\s*the\s+final\s+answer\s*(?::|is)\s*(.+)$",
        r"final\s+answer\s*(?::|is)\s*(.+)$",
        r"answer\s*(?::|is)\s*(.+)$",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            answer = matches[-1].group(1).strip()
            answer = re.split(r"\bI\s+hope\s+it\s+is\s+correct\b", answer, flags=re.IGNORECASE)[0]
            return answer.strip(" `*_.$")

    last_line = _last_nonempty_line(text).strip(" `*_.$")
    # Some responses give a conclusion such as ``the value is $14/3$``
    # without using the requested final-answer marker.
    conclusion_matches = list(
        re.finditer(r"(?:\bis\b|\bare\b|\bwas\b|\bwere\b|=)\s*(.+)$", last_line, flags=re.IGNORECASE)
    )
    if conclusion_matches:
        return conclusion_matches[-1].group(1).strip(" `*_.$")
    return last_line


def _replace_balanced_command(text: str, command_names: Iterable[str], wrapper: str) -> str:
    names = tuple(command_names)
    output: list[str] = []
    index = 0
    while index < len(text):
        matched_name = next(
            (name for name in names if text.startswith(name, index)),
            None,
        )
        if matched_name is None:
            output.append(text[index])
            index += 1
            continue

        cursor = index + len(matched_name)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            output.append(text[index])
            index += 1
            continue

        depth = 0
        end = None
        for close_index in range(cursor, len(text)):
            if text[close_index] == "{":
                depth += 1
            elif text[close_index] == "}":
                depth -= 1
                if depth == 0:
                    end = close_index
                    break
        if end is None:
            output.append(text[index])
            index += 1
            continue

        inner = text[cursor + 1 : end]
        output.append(wrapper.format(inner))
        index = end + 1
    return "".join(output)


def _rewrite_fractions(text: str) -> str:
    """Rewrite common nested ``\frac{num}{den}`` forms without SymPy."""
    for _ in range(8):
        changed = False
        output: list[str] = []
        index = 0
        while index < len(text):
            command = None
            for candidate in (r"\frac", r"\dfrac", r"\tfrac"):
                if text.startswith(candidate, index):
                    command = candidate
                    break
            if command is None:
                output.append(text[index])
                index += 1
                continue

            cursor = index + len(command)
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != "{":
                output.append(text[index])
                index += 1
                continue

            def read_group(start: int) -> tuple[str, int] | None:
                if start >= len(text) or text[start] != "{":
                    return None
                depth = 0
                for close_index in range(start, len(text)):
                    if text[close_index] == "{":
                        depth += 1
                    elif text[close_index] == "}":
                        depth -= 1
                        if depth == 0:
                            return text[start + 1 : close_index], close_index + 1
                return None

            numerator_result = read_group(cursor)
            if numerator_result is None:
                output.append(text[index])
                index += 1
                continue
            numerator, cursor = numerator_result
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            denominator_result = read_group(cursor)
            if denominator_result is None:
                output.append(text[index])
                index += 1
                continue
            denominator, cursor = denominator_result
            output.append(f"({numerator})/({denominator})")
            index = cursor
            changed = True
        text = "".join(output)
        if not changed:
            break
    return text


def normalize_math_answer(answer: str) -> str:
    """Normalize common MATH LaTeX variants for transparent exact matching."""
    text = str(answer or "").strip()
    if not text:
        return ""

    text = text.replace("\u2212", "-").replace("\u00a0", " ")
    text = text.replace("```latex", "").replace("```", "")
    text = text.replace(r"\$", "").replace("$", "")
    text = text.replace("√", r"\sqrt").replace("π", r"\pi")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace(r"\cup", "union").replace(r"\cap", "intersection")
    text = text.replace(r"\infty", "infinity").replace(r"\in", "in")
    text = text.replace(r"\,", "").replace(r"\!", "").replace(r"\;", "")
    text = text.replace(r"\:", "").replace(r"\quad", "")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\pm", "+/-")
    text = re.sub(r"\\begin\{(?:p|b)?matrix\}", "", text)
    text = re.sub(r"\\end\{(?:p|b)?matrix\}", "", text)
    text = re.sub(r"\\(?:text|mbox|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\text\s+([A-Za-z]+)", r"\1", text)
    text = _rewrite_fractions(text)
    # MATH answers also contain unbraced forms such as ``\frac 59`` and
    # ``\frac43``.
    text = re.sub(r"\\frac\s*([+-]?\d+)\s*([+-]?\d+)", r"(\1)/(\2)", text)
    text = _replace_balanced_command(text, (r"\sqrt",), "sqrt({})")
    text = re.sub(r"\\sqrt\s*([+-]?\d+)", r"sqrt(\1)", text)
    text = text.replace(r"^\circ", "degrees").replace(r"\circ", "degrees")
    text = re.sub(r"(?<=\d)\s*degrees\b", "degrees", text, flags=re.IGNORECASE)
    # Remove thousands separators but preserve commas in tuples, intervals,
    # matrices, and coordinate lists such as ``(2,12)``.
    text = re.sub(r"(?<![\d,(])\d{1,3}(?:,\d{3})+(?!\d)", lambda m: m.group(0).replace(",", ""), text)
    text = text.replace(r"\\", ";")
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", "", text).lower()
    text = text.strip(".,;:")
    return text


def _simplify_integer_radicals(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        coefficient = int(match.group("coefficient") or "1")
        radicand = int(match.group("radicand"))
        root = 1
        square_factor = int(math.isqrt(radicand))
        for factor in range(square_factor, 1, -1):
            if radicand % (factor * factor) == 0:
                root = factor
                break
        remainder = radicand // (root * root)
        coefficient *= root
        if remainder == 1:
            return str(coefficient)
        prefix = "" if coefficient == 1 else str(coefficient)
        return f"{prefix}sqrt({remainder})"

    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?P<coefficient>\d*)sqrt\((?P<radicand>\d+)\)"
    )
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(replace, text)
    return text


def _strip_outer_parentheses(text: str) -> str:
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        closes_at_end = True
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    closes_at_end = False
                    break
        if closes_at_end and depth == 0:
            text = text[1:-1]
        else:
            break
    return text


def _simple_rational(text: str) -> Fraction | None:
    text = _strip_outer_parentheses(text)
    text = re.sub(r"^\(([+-]?\d+(?:\.\d+)?)\)/\(([+-]?\d+(?:\.\d+)?)\)$", r"\1/\2", text)
    fraction_match = re.fullmatch(
        r"([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)",
        text,
    )
    if fraction_match:
        return Fraction(fraction_match.group(1)) / Fraction(fraction_match.group(2))
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return Fraction(text)
    return None


def _answer_variants(answer: str) -> set[str]:
    normalized = _simplify_integer_radicals(normalize_math_answer(answer))
    variants = {normalized}

    # A final answer such as ``x=5`` or ``x in [-2,7]`` is equivalent to the
    # value-only form commonly emitted by models.
    value_only = re.sub(r"^[a-z](?:=|in)", "", normalized)
    if value_only != normalized:
        variants.add(value_only)

    # Units and currency symbols are formatting, not the mathematical value.
    unitless = re.sub(
        r"(?:degrees|cents|dollars|inches(?:\^?2)?|cm(?:\^?2)?)$",
        "",
        normalized,
    )
    if unitless != normalized:
        variants.add(unitless)

    rational = _simple_rational(normalized)
    if rational is not None:
        variants.add(str(rational))
    for variant in list(variants):
        variant_rational = _simple_rational(variant)
        if variant_rational is not None:
            variants.add(str(variant_rational))
    return {variant for variant in variants if variant}


def math_answers_equal(predicted: str, target: str) -> bool:
    predicted_variants = _answer_variants(predicted)
    target_variants = _answer_variants(target)
    if not predicted_variants or not target_variants:
        return False
    if predicted_variants & target_variants:
        return True

    # MATH contains answer choices represented as either ``C`` or ``(C)``.
    for pred_norm in predicted_variants:
        pred_choice = re.fullmatch(r"\(?([a-e])\)?", pred_norm)
        if not pred_choice:
            continue
        for target_norm in target_variants:
            target_choice = re.fullmatch(r"\(?([a-e])\)?", target_norm)
            if target_choice and pred_choice.group(1) == target_choice.group(1):
                return True
    return False


def completed_ids(path: str | Path) -> set[str]:
    completed: set[str] = set()
    file_path = Path(path)
    if not file_path.exists():
        return completed
    with file_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "unique_id" in row:
                completed.add(str(row["unique_id"]))
            elif "dataset_index" in row:
                completed.add(f"index:{row['dataset_index']}")
    return completed


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row.get("judge")) for row in rows)

    def group_summary(key: str) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(key, "unknown"))].append(row)
        result: dict[str, dict[str, float | int]] = {}
        for group, group_rows in sorted(grouped.items()):
            group_correct = sum(bool(row.get("judge")) for row in group_rows)
            result[group] = {
                "correct": group_correct,
                "total": len(group_rows),
                "accuracy": group_correct / len(group_rows) if group_rows else 0.0,
            }
        return result

    timing_rows = [row.get("timing", {}) for row in rows if isinstance(row.get("timing"), dict)]

    def mean_timing(key: str) -> float:
        values = [float(timing.get(key, 0.0) or 0.0) for timing in timing_rows]
        return sum(values) / len(values) if values else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_subject": group_summary("subject"),
        "by_level": group_summary("level"),
        "mean_input_tokens": (
            sum(int(row.get("input_tokens", 0) or 0) for row in rows) / total if total else 0.0
        ),
        "mean_output_tokens": (
            sum(int(row.get("output_tokens", 0) or 0) for row in rows) / total if total else 0.0
        ),
        "mean_prefill_time_s": mean_timing("prefill_time"),
        "mean_decode_time_s": mean_timing("decode_time"),
        "mean_tpot_ms": mean_timing("tpot_ms"),
        "attn_type": rows[0].get("attn_type") if rows else None,
        "model_name": rows[0].get("model_name") if rows else None,
    }
