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
                _row(
                    text=text,
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
                    },
                )
            )
    candidates.sort(key=lambda row: int(row["chars"]), reverse=True)
    return candidates[: max(int(limit), 0)] if int(limit) > 0 else candidates


def collect_nolima(
    haystack_dir: Path,
    *,
    limit: int,
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(haystack_dir.glob("rand_book_*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk_text(text, target_chars=target_chars, min_chars=min_chars)
        for idx, chunk in enumerate(chunks):
            rows.append(
                _row(
                    text=chunk,
                    source_dataset="nolima_haystack_rand_shuffle",
                    source_id=f"{path.stem}:chunk{idx:04d}",
                    source_path=str(path),
                    source_split="haystack_rand_shuffle",
                    max_chars=max_chars,
                    extra={"chunk_index": idx, "original_chars": len(chunk)},
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
                    _row(
                        text=f"BABILong task: {task}\n\n{chunk}",
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
                        },
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
