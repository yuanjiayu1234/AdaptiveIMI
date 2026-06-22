from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_tasks(value: str) -> List[str]:
    text = str(value).strip()
    if not text or text.lower() in ("all", "*"):
        return list(DEFAULT_RULER_TASKS)
    return [part for part in text.replace(",", " ").split() if part]


def _parse_length_weights(value: str) -> List[Tuple[int, float]]:
    pairs: List[Tuple[int, float]] = []
    for part in str(value).replace(",", " ").split():
        if not part:
            continue
        if ":" in part:
            length_s, weight_s = part.split(":", 1)
        else:
            length_s, weight_s = part, "1"
        pairs.append((int(length_s), float(weight_s)))
    if not pairs:
        raise ValueError("at least one length weight is required")
    return pairs


def _allocate_counts(total: int, weights: Sequence[Tuple[int, float]]) -> Dict[int, int]:
    if total <= 0:
        return {}
    positive = [(length, max(float(weight), 0.0)) for length, weight in weights]
    denom = sum(weight for _, weight in positive)
    if denom <= 0:
        raise ValueError("length weights must contain a positive weight")
    raw = [(length, total * weight / denom) for length, weight in positive]
    counts = {length: int(value) for length, value in raw}
    remainder = total - sum(counts.values())
    fractions = sorted(
        ((value - int(value), idx, length) for idx, (length, value) in enumerate(raw)),
        reverse=True,
    )
    for _, _, length in fractions[:remainder]:
        counts[length] += 1
    return counts


def _sample_without_replacement(
    rows: Sequence[Dict[str, Any]],
    count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[: min(int(count), len(rows))]


def _balanced_real_rows(
    rows: Sequence[Dict[str, Any]],
    count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("input") or "").strip():
            by_source[str(row.get("source_dataset") or "unknown")].append(row)
    if not by_source:
        return []

    sources = sorted(by_source)
    base = int(count) // len(sources)
    extra = int(count) % len(sources)
    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []
    for idx, source in enumerate(sources):
        want = base + (1 if idx < extra else 0)
        bucket = list(by_source[source])
        rng.shuffle(bucket)
        selected.extend(bucket[:want])
        leftovers.extend(bucket[want:])

    if len(selected) < count:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: count - len(selected)])
    rng.shuffle(selected)
    return selected[:count]


def collect_real_rows(path: Path, *, count: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    source_rows = _read_jsonl(path)
    selected = _balanced_real_rows(source_rows, count, rng)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(selected):
        text = str(row.get("input") or "")
        copied = dict(row)
        copied["input"] = text
        copied["chars"] = int(copied.get("chars") or len(text))
        copied["mix_component"] = "real"
        copied["mix_index"] = idx
        copied["mix_source_jsonl"] = str(path)
        out.append(copied)
    return out


def _iter_ruler_rows(
    ruler_root: Path,
    lengths: Iterable[int],
    tasks: Sequence[str],
    split: str,
) -> Iterable[Dict[str, Any]]:
    for length in lengths:
        for task in tasks:
            path = ruler_root / str(length) / task / f"{split}.jsonl"
            if not path.exists():
                continue
            for row_idx, row in enumerate(_read_jsonl(path)):
                text = str(row.get("input") or "")
                if not text.strip():
                    continue
                yield {
                    "input": text,
                    "answer_prefix": row.get("answer_prefix", ""),
                    "source_dataset": "ruler_synthetic",
                    "source_hf_dataset": str(ruler_root),
                    "source_subset": task,
                    "source_split": split,
                    "source_id": f"ruler:{length}:{task}:{row.get('index', row_idx)}",
                    "chars": len(text),
                    "mix_component": "synthetic",
                    "ruler_length": int(length),
                    "ruler_task": task,
                    "ruler_index": row.get("index", row_idx),
                    "ruler_outputs": row.get("outputs"),
                    "ruler_token_length": row.get("length"),
                }


def collect_ruler_rows(
    ruler_root: Path,
    *,
    count: int,
    length_weights: Sequence[Tuple[int, float]],
    tasks: Sequence[str],
    split: str,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    lengths = [length for length, _ in length_weights]
    available_by_length: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in _iter_ruler_rows(ruler_root, lengths, tasks, split):
        available_by_length[int(row["ruler_length"])].append(row)

    allocated = _allocate_counts(count, length_weights)
    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []
    for length, want in allocated.items():
        bucket = available_by_length.get(int(length), [])
        picked = _sample_without_replacement(bucket, want, rng)
        selected.extend(picked)
        picked_ids = {id(row) for row in picked}
        leftovers.extend(row for row in bucket if id(row) not in picked_ids)

    if len(selected) < count:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: count - len(selected)])

    rng.shuffle(selected)
    for idx, row in enumerate(selected[:count]):
        row["mix_index"] = idx
    return selected[:count]


def _summarize(label: str, rows: Sequence[Dict[str, Any]]) -> None:
    lengths = sorted(int(row.get("chars") or len(str(row.get("input") or ""))) for row in rows)
    print(f"[mixed-calib] {label}: rows={len(rows)}")
    if lengths:
        print(
            f"[mixed-calib] {label}: chars_min={lengths[0]} "
            f"chars_p50={lengths[len(lengths)//2]} chars_max={lengths[-1]}"
        )
    source_counts = Counter(str(row.get("source_dataset")) for row in rows)
    print(f"[mixed-calib] {label}: sources={dict(sorted(source_counts.items()))}")
    ruler_counts = Counter(str(row.get("ruler_length")) for row in rows if row.get("ruler_length") is not None)
    if ruler_counts:
        print(f"[mixed-calib] {label}: ruler_lengths={dict(sorted(ruler_counts.items(), key=lambda kv: int(kv[0])))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 50/50 real + RULER synthetic calibration JSONL.")
    parser.add_argument("--out", type=Path, default=Path("/root/AdaptiveIMI/outputs/train_data/calib_real100_ruler100_mixed.jsonl"))
    parser.add_argument("--real-jsonl", type=Path, default=Path("/root/AdaptiveIMI/outputs/imi_partition/calib_booksum_nolima_babilong.jsonl"))
    parser.add_argument("--ruler-root", type=Path, default=Path("/root/RULER"))
    parser.add_argument("--real-count", type=int, default=100)
    parser.add_argument("--synthetic-count", type=int, default=100)
    parser.add_argument(
        "--synthetic-length-weights",
        type=str,
        default="131072:40,98304:40,65536:10,32768:5,16384:3,8192:2",
        help="Comma/space separated length:weight list. Defaults emphasize 128k and 96k.",
    )
    parser.add_argument("--ruler-tasks", type=str, default="all")
    parser.add_argument("--ruler-split", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-components", action="store_true", help="Also write *_real.jsonl and *_ruler.jsonl.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_rows = collect_real_rows(args.real_jsonl, count=int(args.real_count), seed=int(args.seed))
    ruler_rows = collect_ruler_rows(
        args.ruler_root,
        count=int(args.synthetic_count),
        length_weights=_parse_length_weights(args.synthetic_length_weights),
        tasks=_parse_tasks(args.ruler_tasks),
        split=str(args.ruler_split),
        seed=int(args.seed) + 1,
    )
    if len(real_rows) != int(args.real_count):
        raise SystemExit(f"[mixed-calib] requested {args.real_count} real rows, got {len(real_rows)}")
    if len(ruler_rows) != int(args.synthetic_count):
        raise SystemExit(f"[mixed-calib] requested {args.synthetic_count} RULER rows, got {len(ruler_rows)}")

    rows = list(real_rows) + list(ruler_rows)
    random.Random(int(args.seed) + 2).shuffle(rows)
    for idx, row in enumerate(rows):
        row["calibration_mix"] = "real_ruler_50_50"
        row["calibration_mix_index"] = idx

    _write_jsonl(args.out, rows)
    print(f"[mixed-calib] wrote {args.out}")
    if args.write_components:
        stem = args.out.with_suffix("")
        _write_jsonl(stem.with_name(stem.name + "_real.jsonl"), real_rows)
        _write_jsonl(stem.with_name(stem.name + "_ruler.jsonl"), ruler_rows)
        print(f"[mixed-calib] wrote component files next to {args.out}")
    _summarize("real", real_rows)
    _summarize("ruler", ruler_rows)
    _summarize("mixed", rows)


if __name__ == "__main__":
    main()
