from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


class TokenBudgeter:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0) or 0), 10**12)
        self.special_tokens = int(len(tokenizer.encode("", add_special_tokens=True)))

    def _encode_text(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def count_tokens(self, text: str) -> int:
        return int(len(self._encode_text(text)) + self.special_tokens)

    def trim_to_token_budget(self, text: str, max_tokens: int) -> Tuple[str, int, bool]:
        text = _compact_text(text)
        token_ids = self._encode_text(text)
        available = int(max_tokens) - self.special_tokens
        if int(max_tokens) <= 0 or len(token_ids) <= available:
            return text, int(len(token_ids) + self.special_tokens), False
        if available <= 0:
            return "", self.special_tokens, True

        marker = "\n\n[... middle truncated for calibration ...]\n\n"
        marker_ids = self._encode_text(marker)
        if len(marker_ids) < available:
            keep_total = available - len(marker_ids)
            keep_head = keep_total // 2
            keep_tail = keep_total - keep_head
            trimmed_ids = token_ids[:keep_head] + marker_ids + token_ids[-keep_tail:]
        else:
            keep_head = available // 2
            keep_tail = available - keep_head
            trimmed_ids = token_ids[:keep_head] + token_ids[-keep_tail:]

        trimmed = self.tokenizer.decode(
            trimmed_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
        return trimmed, int(len(trimmed_ids) + self.special_tokens), True


def _load_token_budgeter(tokenizer_path: str) -> Optional[TokenBudgeter]:
    if not str(tokenizer_path).strip():
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    return TokenBudgeter(tokenizer)


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
    chunks_payload: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            name
            for name in zf.namelist()
            if "/en-10k/" in name and name.endswith("_train.txt")
        )
        for name in names:
            data = _strip_babilong_answers(zf.read(name).decode("utf-8", errors="ignore"))
            chunks = _chunk_text(data, target_chars=target_chars, min_chars=min_chars)
            task = _babilong_task_name(name)
            for idx, chunk in enumerate(chunks):
                chunks_payload.append({
                    "task": task,
                    "chunk_index": idx,
                    "text": f"BABILong task: {task}\n\n{chunk}",
                    "source_path": f"{zip_path}:{name}",
                    "chars": len(chunk),
                })

    rows: List[Dict[str, Any]] = []
    pack_target = max(int(target_chars), 1)
    current: List[Dict[str, Any]] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        text = "\n\n---\n\n".join(str(item["text"]) for item in current)
        if len(text) >= int(min_chars):
            first = current[0]
            last = current[-1]
            tasks = sorted({str(item["task"]) for item in current})
            source_ids = [f"{item['task']}:chunk{int(item['chunk_index']):04d}" for item in current]
            digest = _digest("|".join(source_ids) + text)
            rows.append(
                _row(
                    text=text,
                    source_dataset="babilong_en10k_train",
                    source_id=f"babilong_pack:{digest}",
                    source_path=f"{first['source_path']}..{last['source_path']}",
                    source_split="train",
                    max_chars=max_chars,
                    extra={
                        "babilong_tasks": tasks,
                        "packed_chunk_count": len(current),
                        "packed_source_ids": source_ids[:128],
                        "original_chars": current_chars,
                        "answers_stripped": True,
                    },
                )
            )
        current = []
        current_chars = 0

    for chunk in chunks_payload:
        chunk_chars = int(chunk["chars"])
        if current and current_chars + chunk_chars > pack_target:
            flush()
            if int(limit) > 0 and len(rows) >= int(limit):
                break
        current.append(chunk)
        current_chars += chunk_chars
    if int(limit) <= 0 or len(rows) < int(limit):
        flush()

    rows.sort(key=lambda row: str(row.get("source_id")))
    return rows[: max(int(limit), 0)] if int(limit) > 0 else rows


def _summarize(rows: Sequence[Dict[str, Any]]) -> None:
    by_source: Dict[str, List[int]] = {}
    by_source_tokens: Dict[str, List[int]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_dataset"]), []).append(int(row["chars"]))
        if row.get("calib_effective_tokens") is not None:
            by_source_tokens.setdefault(str(row["source_dataset"]), []).append(int(row["calib_effective_tokens"]))
    print(f"[long-calib] total rows={len(rows)}")
    for source, lengths in sorted(by_source.items()):
        lengths = sorted(lengths)
        p50 = lengths[len(lengths) // 2]
        p90 = lengths[int((len(lengths) - 1) * 0.90)]
        message = (
            f"[long-calib] {source}: rows={len(lengths)} "
            f"chars_min={lengths[0]} chars_p50={p50} chars_p90={p90} chars_max={lengths[-1]}"
        )
        token_lengths = sorted(by_source_tokens.get(source, []))
        if token_lengths:
            message += (
                f" tokens_sum={sum(token_lengths)} "
                f"tokens_min={token_lengths[0]} "
                f"tokens_p50={token_lengths[len(token_lengths)//2]} "
                f"tokens_max={token_lengths[-1]}"
            )
        print(message)


def _annotate_and_filter_lengths(
    rows: Sequence[Dict[str, Any]],
    *,
    token_budgeter: Optional[TokenBudgeter],
    min_tokens: int,
    max_tokens: int,
    min_chars: int,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        text = str(row.get("input") or "")
        copied = dict(row)
        if token_budgeter is not None:
            original_token_count = int(token_budgeter.count_tokens(text))
            token_count = original_token_count
            if int(max_tokens) > 0:
                text, token_count, truncated = token_budgeter.trim_to_token_budget(text, int(max_tokens))
                if truncated:
                    copied["input"] = text
                    copied["chars"] = len(text)
                    copied["calib_tokens_before_truncation"] = original_token_count
                    copied["calib_token_truncation"] = "middle_head_tail"
            effective_tokens = min(token_count, int(max_tokens)) if int(max_tokens) > 0 else token_count
            copied["calib_tokens"] = token_count
            copied["calib_effective_tokens"] = effective_tokens
            if int(min_tokens) > 0 and effective_tokens < int(min_tokens):
                continue
        elif int(min_chars) > 0 and int(copied.get("chars") or len(text)) < int(min_chars):
            continue
        filtered.append(copied)
    return filtered


def _select_by_source_token_budget(
    rows: Sequence[Dict[str, Any]],
    *,
    source_token_budget: int,
    seed: int,
    order: str,
) -> List[Dict[str, Any]]:
    if int(source_token_budget) <= 0:
        return list(rows)

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row.get("source_dataset") or "unknown"), []).append(dict(row))

    selected: List[Dict[str, Any]] = []
    rng = random.Random(int(seed))
    normalized_order = str(order).strip().lower()
    for source, bucket in sorted(by_source.items()):
        if normalized_order in ("shuffle", "random"):
            rng.shuffle(bucket)
        elif normalized_order in ("longest", "length_desc"):
            bucket.sort(
                key=lambda row: (
                    int(row.get("calib_effective_tokens") or 0),
                    int(row.get("chars") or 0),
                    str(row.get("source_id") or ""),
                ),
                reverse=True,
            )
        elif normalized_order in ("source", "stable"):
            bucket.sort(key=lambda row: str(row.get("source_id") or ""))
        else:
            raise ValueError(f"unsupported --selection-order={order!r}")

        running = 0
        for row in bucket:
            effective_tokens = int(row.get("calib_effective_tokens") or max(int(row.get("chars") or 0) // 4, 1))
            if running > 0 and running >= int(source_token_budget):
                break
            copied = dict(row)
            copied["source_token_budget"] = int(source_token_budget)
            copied["source_budget_component"] = source
            selected.append(copied)
            running += effective_tokens
        print(f"[long-calib] selected source={source} rows={sum(1 for row in selected if row.get('source_budget_component') == source)} effective_tokens={running}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build long-context calibration JSONL from BookSum, NoLiMa, and BABILong.")
    parser.add_argument("--out", type=Path, default=Path("outputs/imi_partition/calib_booksum_nolima_babilong.jsonl"))
    parser.add_argument("--booksum-train", type=Path, default=Path("experiments/data/booksum/train.csv"))
    parser.add_argument("--nolima-haystack-dir", type=Path, default=Path("experiments/data/NoLiMa/haystack/rand_shuffle"))
    parser.add_argument("--babilong-zip", type=Path, default=Path("experiments/data/babilong/data/tasks_1-20_v1-2.zip"))
    parser.add_argument("--booksum-limit", type=int, default=96)
    parser.add_argument("--nolima-limit", type=int, default=50)
    parser.add_argument("--babilong-limit", type=int, default=80)
    parser.add_argument("--min-chars", type=int, default=20000)
    parser.add_argument("--chunk-chars", type=int, default=120000)
    parser.add_argument("--max-chars", type=int, default=0, help="Maximum chars per row; <=0 disables middle truncation.")
    parser.add_argument("--tokenizer-path", type=str, default="", help="Optional tokenizer path for token-balanced source selection.")
    parser.add_argument("--min-tokens", type=int, default=0, help="Drop rows below this effective token count when tokenizer is provided.")
    parser.add_argument("--max-tokens", type=int, default=0, help="Cap effective token count per row for source balancing; 0 disables the cap.")
    parser.add_argument("--source-token-budget", type=int, default=0, help="Effective token budget per source; 0 keeps all filtered rows.")
    parser.add_argument("--selection-order", choices=["longest", "shuffle", "source"], default="longest")
    parser.add_argument("--seed", type=int, default=0)
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
    token_budgeter = _load_token_budgeter(str(args.tokenizer_path))
    rows = _annotate_and_filter_lengths(
        rows,
        token_budgeter=token_budgeter,
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_chars=int(args.min_chars),
    )
    rows = _select_by_source_token_budget(
        rows,
        source_token_budget=int(args.source_token_budget),
        seed=int(args.seed),
        order=str(args.selection_order),
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
