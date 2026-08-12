from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

BOOKSUM_TEMPLATE = (
    "You will answer a question based on the following book chapter.\n\n"
    "<context>\n{context}\n</context>\n\n"
    "Question: Summarize the chapter section above in a concise way."
)

NOLIMA_DEFAULT_TEMPLATE = (
    "You will answer a question based on the following book snippet:\n\n"
    "{haystack}\n\n"
    "Use the information provided in the book snippet to answer the question. "
    "Your answer should be short and based on either explicitly stated facts or strong, logical inferences.\n\n"
    "Question: {question}\n\n"
    "Return only the final answer with no additional explanation or reasoning."
)

BABILONG_TEMPLATE = (
    "You will answer questions based only on the following BABILong context.\n\n"
    "<context>\n{context}\n</context>\n\n"
    "Answer the final question in the context using only the information above."
)

DEFAULT_ANSWER_PREFIX = " Answer:"


def _compact_text(value: Any) -> str:
    text = str(value) if value is not None else ""
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()


def _truncate_middle(text: str, max_chars: int) -> str:
    text = _compact_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n[... middle truncated for calibration ...]\n\n"
    if max_chars <= len(marker):
        return text[:max_chars]
    keep_total = max_chars - len(marker)
    keep_head = keep_total // 2
    keep_tail = keep_total - keep_head
    return text[:keep_head].rstrip() + marker + text[-keep_tail:].lstrip()


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _with_answer_prefix(row: Dict[str, Any], prefix: str = DEFAULT_ANSWER_PREFIX) -> Dict[str, Any]:
    row["answer_prefix"] = prefix
    return row


def _row(
    *,
    text: str,
    source_dataset: str,
    source_id: str,
    source_path: str,
    source_split: str,
    max_chars: int,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    text = _truncate_middle(text, max_chars)
    payload: Dict[str, Any] = {
        "input": text,
        "source_dataset": source_dataset,
        "source_hf_dataset": source_path,
        "source_subset": None,
        "source_split": source_split,
        "source_id": source_id,
        "chars": len(text),
    }
    if extra:
        payload.update(extra)
    return payload


def _insert_text_at_depth(text: str, insert: str, *, depth: float = 0.5) -> str:
    text = _compact_text(text)
    insert = _compact_text(insert)
    if not text:
        return insert
    depth = min(max(float(depth), 0.0), 1.0)
    target = int(len(text) * depth)
    left_boundary = text.rfind("\n", 0, target)
    right_boundary = text.find("\n", target)
    if left_boundary >= 0 and target - left_boundary < 512:
        pos = left_boundary + 1
    elif right_boundary >= 0 and right_boundary - target < 512:
        pos = right_boundary + 1
    else:
        pos = target
    return (text[:pos].rstrip() + "\n" + insert + "\n" + text[pos:].lstrip()).strip()


def _format_template_value(template: str, args: Sequence[str], *, character: str | None = None) -> str:
    out = str(template)
    for idx, arg in enumerate(args, start=1):
        out = out.replace("{" + str(idx) + "}", str(arg))
    if character is not None:
        out = out.replace("{CHAR}", str(character))
    return out


def _load_nolima_tests(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    configs = json.loads(path.read_text(encoding="utf-8"))
    tests: List[Dict[str, Any]] = []
    for config in configs:
        exp_id = str(config.get("id", "0000"))
        task_template = str(config.get("task_template") or NOLIMA_DEFAULT_TEMPLATE)
        questions = config.get("questions") or {}
        raw_tests = config.get("tests") or {}
        character_set = list(config.get("character_set") or [])
        for question_type, question_template in sorted(questions.items()):
            for test_id, test in sorted(raw_tests.items()):
                input_args = [str(value) for value in (test.get("input_args") or [])]
                character = None
                if character_set:
                    char_index = int(hashlib.sha1(f"{exp_id}:{test_id}:{question_type}".encode("utf-8")).hexdigest(), 16)
                    character = str(character_set[char_index % len(character_set)])
                tests.append(
                    {
                        "test_name": f"{exp_id}_{test_id}_{question_type}",
                        "task_template": task_template,
                        "needle": _format_template_value(str(config.get("needle") or ""), input_args, character=character),
                        "question": _format_template_value(str(question_template), input_args, character=character),
                        "selected_character": character,
                    }
                )
    return tests


def _chunk_text(text: str, *, target_chars: int, min_chars: int) -> List[str]:
    text = _compact_text(text)
    if not text:
        return []
    target_chars = max(int(target_chars), 1)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + max(target_chars // 2, 1), end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        start = max(end, start + 1)
    return chunks


def collect_booksum(
    path: Path,
    *,
    limit: int,
    min_chars: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    usecols = [
        "bid",
        "source",
        "book_id",
        "summary_id",
        "chapter",
        "chapter_length",
        "summary_name",
    ]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=2000):
        for _, item in chunk.iterrows():
            chapter = _compact_text(item.get("chapter"))
            if len(chapter) < min_chars:
                continue
            title = _compact_text(item.get("book_id"))
            section = _compact_text(item.get("summary_id") or item.get("summary_name"))
            text = f"Book: {title}\nSection: {section}\n\n{chapter}"
            candidates.append(
                _with_answer_prefix(
                    _row(
                        text=BOOKSUM_TEMPLATE.format(context=text),
                        source_dataset="booksum_train",
                        source_id=f"booksum:{item.get('bid')}:{_digest(title + section + chapter)}",
                        source_path=str(path),
                        source_split="train",
                        max_chars=max_chars,
                        extra={
                            "booksum_source": item.get("source"),
                            "book_id": title,
                            "summary_id": section,
                            "original_chars": len(text),
                            "prompt_style": "qa_summary",
                        },
                    ),
                    " Summary:",
                )
            )
    candidates.sort(key=lambda row: int(row["chars"]), reverse=True)
    return candidates[: max(int(limit), 0)] if int(limit) > 0 else candidates


def collect_nolima(
    haystack_dir: Path,
    *,
    needle_set_path: Path,
    limit: int,
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    nolima_tests = _load_nolima_tests(needle_set_path)
    for path in sorted(haystack_dir.glob("rand_book_*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk_text(text, target_chars=target_chars, min_chars=min_chars)
        for idx, chunk in enumerate(chunks):
            extra: Dict[str, Any] = {"chunk_index": idx, "original_chars": len(chunk)}
            prompt_text = chunk
            if nolima_tests:
                test = nolima_tests[(len(rows) + idx) % len(nolima_tests)]
                haystack = _insert_text_at_depth(chunk, str(test["needle"]), depth=0.5)
                prompt_text = str(test["task_template"]).format(haystack=haystack, question=str(test["question"]))
                extra.update(
                    {
                        "prompt_style": "nolima_qa",
                        "nolima_test_name": test["test_name"],
                        "nolima_question": test["question"],
                        "nolima_needle_inserted": True,
                        "nolima_selected_character": test.get("selected_character"),
                    }
                )
            else:
                prompt_text = (
                    "You will answer a question based on the following book snippet.\n\n"
                    f"{chunk}\n\n"
                    "Question: What is the key information in the snippet?"
                )
                extra.update({"prompt_style": "generic_haystack_qa", "nolima_needle_inserted": False})
            rows.append(
                _with_answer_prefix(
                    _row(
                        text=prompt_text,
                        source_dataset="nolima_haystack_rand_shuffle",
                        source_id=f"{path.stem}:chunk{idx:04d}",
                        source_path=str(path),
                        source_split="haystack_rand_shuffle",
                        max_chars=max_chars,
                        extra=extra,
                    )
                )
            )
    rows.sort(key=lambda row: (str(row["source_id"])))
    return rows[: max(int(limit), 0)] if int(limit) > 0 else rows


def _babilong_task_name(path: str) -> str:
    base = Path(path).name
    return re.sub(r"_train\.txt$", "", base)


def _strip_babilong_answers(text: str) -> str:
    lines: List[str] = []
    for line in _compact_text(text).splitlines():
        if "\t" in line:
            line = line.split("\t", 1)[0].rstrip()
        lines.append(line)
    return "\n".join(lines).strip()


def collect_babilong(
    zip_path: Path,
    *,
    limit: int,
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            name
            for name in zf.namelist()
            if "/en-10k/" in name and name.endswith("_train.txt")
        )
        per_task_limit = max(1, math.ceil(int(limit) / max(len(names), 1))) if int(limit) > 0 else 0
        for name in names:
            data = _strip_babilong_answers(zf.read(name).decode("utf-8", errors="ignore"))
            chunks = _chunk_text(data, target_chars=target_chars, min_chars=min_chars)
            if per_task_limit > 0:
                chunks = chunks[:per_task_limit]
            task = _babilong_task_name(name)
            for idx, chunk in enumerate(chunks):
                rows.append(
                    _with_answer_prefix(
                        _row(
                            text=BABILONG_TEMPLATE.format(context=f"BABILong task: {task}\n\n{chunk}"),
                            source_dataset="babilong_en10k_train",
                            source_id=f"{task}:chunk{idx:04d}",
                            source_path=f"{zip_path}:{name}",
                            source_split="train",
                            max_chars=max_chars,
                            extra={
                                "babilong_task": task,
                                "chunk_index": idx,
                                "original_chars": len(chunk),
                                "answers_stripped": True,
                                "prompt_style": "babilong_context_final_question",
                            },
                        )
                    )
                )
    rows.sort(key=lambda row: (str(row.get("babilong_task")), int(row.get("chunk_index", 0))))
    return rows[: max(int(limit), 0)] if int(limit) > 0 else rows


def _summarize(rows: Sequence[Dict[str, Any]]) -> None:
    by_source: Dict[str, List[int]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_dataset"]), []).append(int(row["chars"]))
    print(f"[long-calib] total rows={len(rows)}")
    for source, lengths in sorted(by_source.items()):
        lengths = sorted(lengths)
        p50 = lengths[len(lengths) // 2]
        p90 = lengths[int((len(lengths) - 1) * 0.90)]
        print(
            f"[long-calib] {source}: rows={len(lengths)} "
            f"chars_min={lengths[0]} chars_p50={p50} chars_p90={p90} chars_max={lengths[-1]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build long-context calibration JSONL from BookSum, NoLiMa, and BABILong.")
    parser.add_argument("--out", type=Path, default=Path("outputs/imi_partition/calib_booksum_nolima_babilong.jsonl"))
    parser.add_argument("--booksum-train", type=Path, default=Path("/root/booksum/train.csv"))
    parser.add_argument("--nolima-haystack-dir", type=Path, default=Path("/root/NoLiMa/haystack/rand_shuffle"))
    parser.add_argument("--nolima-needle-set", type=Path, default=Path("/root/NoLiMa/needlesets/needle_set.json"))
    parser.add_argument("--babilong-zip", type=Path, default=Path("/root/babilong/data/tasks_1-20_v1-2.zip"))
    parser.add_argument("--booksum-limit", type=int, default=96)
    parser.add_argument("--nolima-limit", type=int, default=50)
    parser.add_argument("--babilong-limit", type=int, default=80)
    parser.add_argument("--min-chars", type=int, default=20000)
    parser.add_argument("--chunk-chars", type=int, default=120000)
    parser.add_argument("--max-chars", type=int, default=0, help="Maximum chars per row; <=0 disables middle truncation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    rows.extend(
        collect_booksum(
            args.booksum_train,
            limit=int(args.booksum_limit),
            min_chars=int(args.min_chars),
            max_chars=int(args.max_chars),
        )
    )
    rows.extend(
        collect_nolima(
            args.nolima_haystack_dir,
            needle_set_path=args.nolima_needle_set,
            limit=int(args.nolima_limit),
            min_chars=int(args.min_chars),
            target_chars=int(args.chunk_chars),
            max_chars=int(args.max_chars),
        )
    )
    rows.extend(
        collect_babilong(
            args.babilong_zip,
            limit=int(args.babilong_limit),
            min_chars=int(args.min_chars),
            target_chars=int(args.chunk_chars),
            max_chars=int(args.max_chars),
        )
    )
    if not rows:
        raise SystemExit("[long-calib] no rows collected")
    rows.sort(key=lambda row: (str(row["source_dataset"]), str(row["source_id"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[long-calib] wrote {args.out}")
    _summarize(rows)


if __name__ == "__main__":
    main()
