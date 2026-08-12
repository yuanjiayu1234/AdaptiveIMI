from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


def _compact_text(value: Any) -> str:
    text = str(value) if value is not None else ""
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    return text[:keep_head].rstrip() + "\n\n[... middle truncated for calibration ...]\n\n" + text[-keep_tail:].lstrip()


def _stable_digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _row(
    *,
    text: str,
    source_dataset: str,
    source_id: str,
    source_path: Path,
    source_split: str = "train",
    max_chars: int,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    text = _truncate_text(_compact_text(text), max_chars)
    payload: Dict[str, Any] = {
        "input": text,
        "source_dataset": source_dataset,
        "source_hf_dataset": str(source_path),
        "source_subset": None,
        "source_split": source_split,
        "source_id": source_id,
        "chars": len(text),
    }
    if extra:
        payload.update(extra)
    return payload


def _pack_texts(
    items: Sequence[tuple[str, str]],
    *,
    source_dataset: str,
    source_path: Path,
    target_chars: int,
    min_chars: int,
    max_chars: int,
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pool = list(items)
    rng.shuffle(pool)
    rows: List[Dict[str, Any]] = []
    chunk: List[str] = []
    ids: List[str] = []
    current = 0

    def flush() -> None:
        nonlocal chunk, ids, current
        if not chunk:
            return
        text = "\n\n---\n\n".join(chunk)
        if len(text) >= min_chars:
            digest = _stable_digest("|".join(ids) + text)
            rows.append(
                _row(
                    text=text,
                    source_dataset=source_dataset,
                    source_id=f"{source_dataset}_pack:{digest}",
                    source_path=source_path,
                    max_chars=max_chars,
                    extra={"packed_source_count": len(ids), "packed_source_ids": ids[:64]},
                )
            )
        chunk = []
        ids = []
        current = 0

    for source_id, text in pool:
        text = _compact_text(text)
        if not text:
            continue
        if current > 0 and current + len(text) > target_chars:
            flush()
            if len(rows) >= limit:
                break
        chunk.append(text)
        ids.append(str(source_id))
        current += len(text) + 6
    if len(rows) < limit:
        flush()
    return rows[:limit]


def collect_wikihop(path: Path, *, min_chars: int, max_chars: int, limit: int) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    rows: List[Dict[str, Any]] = []
    for _, item in df.iterrows():
        supports = "\n\n".join(_compact_text(value) for value in item["supports"] if _compact_text(value))
        candidates = ", ".join(str(value) for value in item["candidates"])
        text = (
            f"Query: {_compact_text(item['query'])}\n"
            f"Candidates: {candidates}\n"
            f"Answer: {_compact_text(item['answer'])}\n\n"
            f"Supports:\n{supports}"
        )
        if len(text) < min_chars:
            continue
        rows.append(
            _row(
                text=text,
                source_dataset="wikihop",
                source_id=str(item["id"]),
                source_path=path,
                max_chars=max_chars,
                extra={"original_chars": len(text)},
            )
        )
    rows.sort(key=lambda row: int(row["chars"]), reverse=True)
    return rows[:limit]


def collect_squad_packed(
    path: Path,
    *,
    target_chars: int,
    min_chars: int,
    max_chars: int,
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    seen = set()
    items: List[tuple[str, str]] = []
    for _, item in df.iterrows():
        title = _compact_text(item["title"])
        context = _compact_text(item["context"])
        key = (title, context)
        if not context or key in seen:
            continue
        seen.add(key)
        source_id = str(item["id"])
        text = f"Title: {title}\n\nContext:\n{context}"
        items.append((source_id, text))
    return _pack_texts(
        items,
        source_dataset="squad_v2_packed",
        source_path=path,
        target_chars=target_chars,
        min_chars=min_chars,
        max_chars=max_chars,
        limit=limit,
        seed=seed,
    )


def collect_wikiqa_packed(
    path: Path,
    *,
    target_chars: int,
    min_chars: int,
    max_chars: int,
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    items: List[tuple[str, str]] = []
    for question_id, group in df.groupby("question_id", sort=False):
        first = group.iloc[0]
        answers = "\n".join(_compact_text(value) for value in group["answer"].tolist() if _compact_text(value))
        positives = "\n".join(
            _compact_text(value)
            for value, label in zip(group["answer"].tolist(), group["label"].tolist())
            if int(label) == 1 and _compact_text(value)
        )
        parts = [
            f"Question: {_compact_text(first['question'])}",
            f"Document title: {_compact_text(first['document_title'])}",
        ]
        if positives:
            parts.append(f"Positive answers:\n{positives}")
        if answers:
            parts.append(f"Candidate passages:\n{answers}")
        items.append((str(question_id), "\n\n".join(parts)))
    return _pack_texts(
        items,
        source_dataset="wikiqa_packed",
        source_path=path,
        target_chars=target_chars,
        min_chars=min_chars,
        max_chars=max_chars,
        limit=limit,
        seed=seed,
    )


def _summarize(rows: Sequence[Dict[str, Any]]) -> None:
    by_source: Dict[str, List[int]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_dataset"]), []).append(int(row["chars"]))
    print(f"[qa-calib] total rows={len(rows)}")
    for source, lengths in sorted(by_source.items()):
        lengths = sorted(lengths)
        p50 = lengths[len(lengths) // 2]
        print(
            f"[qa-calib] {source}: rows={len(lengths)} "
            f"chars_min={lengths[0]} chars_p50={p50} chars_max={lengths[-1]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build QA/multihop calibration JSONL from local parquet datasets.")
    parser.add_argument("--out", type=Path, default=Path("outputs/imi_partition/calib_qa_common_mix.jsonl"))
    parser.add_argument("--squad-path", type=Path, default=Path("/root/squad_v2/squad_v2/train-00000-of-00001.parquet"))
    parser.add_argument("--wikiqa-path", type=Path, default=Path("/root/wiki_qa/data/train-00000-of-00001.parquet"))
    parser.add_argument("--wikihop-path", type=Path, default=Path("/root/wikihop/data/train-00000-of-00001.parquet"))
    parser.add_argument("--wikihop-limit", type=int, default=64)
    parser.add_argument("--squad-limit", type=int, default=64)
    parser.add_argument("--wikiqa-limit", type=int, default=32)
    parser.add_argument("--min-chars", type=int, default=20000)
    parser.add_argument("--target-pack-chars", type=int, default=60000)
    parser.add_argument("--max-chars", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-output", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    rows.extend(
        collect_wikihop(
            args.wikihop_path,
            min_chars=int(args.min_chars),
            max_chars=int(args.max_chars),
            limit=int(args.wikihop_limit),
        )
    )
    rows.extend(
        collect_squad_packed(
            args.squad_path,
            target_chars=int(args.target_pack_chars),
            min_chars=int(args.min_chars),
            max_chars=int(args.max_chars),
            limit=int(args.squad_limit),
            seed=int(args.seed) + 17,
        )
    )
    rows.extend(
        collect_wikiqa_packed(
            args.wikiqa_path,
            target_chars=int(args.target_pack_chars),
            min_chars=int(args.min_chars),
            max_chars=int(args.max_chars),
            limit=int(args.wikiqa_limit),
            seed=int(args.seed) + 31,
        )
    )
    if not rows:
        raise SystemExit("[qa-calib] no rows collected")
    if bool(args.shuffle_output):
        random.Random(int(args.seed)).shuffle(rows)
    else:
        rows.sort(key=lambda row: int(row["chars"]), reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[qa-calib] wrote {args.out}")
    _summarize(rows)


if __name__ == "__main__":
    main()
