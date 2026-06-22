from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    subset: Optional[str]
    split: str
    renderer: str
    streaming: bool = False
    default_limit: int = 64


SOURCE_SPECS: Dict[str, SourceSpec] = {
    "longalign": SourceSpec("longalign", "zai-org/LongAlign-10k", None, "train", "generic", False, 64),
    "longalpaca": SourceSpec("longalpaca", "Yukang/LongAlpaca-12k", None, "train", "instruction", False, 64),
    "scrolls_gov_report": SourceSpec("scrolls_gov_report", "tau/scrolls", "gov_report", "train", "input", False, 64),
    "scrolls_contract_nli": SourceSpec("scrolls_contract_nli", "tau/scrolls", "contract_nli", "train", "input", False, 64),
    "scrolls_summ_screen": SourceSpec("scrolls_summ_screen", "tau/scrolls", "summ_screen_fd", "train", "input", False, 64),
    "multi_news": SourceSpec("multi_news", "alexfabbri/multi_news", None, "train", "multi_news", False, 64),
    "codesearchnet": SourceSpec("codesearchnet", "irds/codesearchnet", "docs", "docs", "code", False, 64),
    "the_stack_python": SourceSpec(
        "the_stack_python",
        "bigcode/the-stack-dedup",
        None,
        "train",
        "code",
        True,
        64,
    ),
    "lmsys_chat": SourceSpec("lmsys_chat", "lmsys/lmsys-chat-1m", None, "train", "conversation", True, 64),
    "wikitablequestions": SourceSpec(
        "wikitablequestions",
        "stanfordnlp/wikitablequestions",
        "random-split-1",
        "train",
        "table",
        False,
        64,
    ),
    "tab_fact": SourceSpec("tab_fact", "wenhu/tab_fact", "tab_fact", "train", "table", False, 64),
    "wikisql": SourceSpec("wikisql", "Salesforce/wikisql", None, "train", "table", False, 64),
}


DEFAULT_SOURCES = [
    "longalign",
    "longalpaca",
    "scrolls_gov_report",
    "scrolls_contract_nli",
    "scrolls_summ_screen",
    "multi_news",
    "codesearchnet",
    "lmsys_chat",
    "wikitablequestions",
    "tab_fact",
    "wikisql",
]


def _compact_text(value: Any) -> str:
    text = str(value) if value is not None else ""
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()


def _first_string(item: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return _compact_text(value)
    return ""


def _stable_id(source: str, item: Dict[str, Any], text: str) -> str:
    for key in ("id", "_id", "pid", "doc_id", "sample_id", "question_id"):
        value = item.get(key)
        if value is not None:
            return str(value)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


def _render_messages(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts: List[str] = []
    for turn in value:
        if isinstance(turn, dict):
            role = _compact_text(turn.get("role") or turn.get("from") or turn.get("speaker") or "")
            content = _compact_text(turn.get("content") or turn.get("value") or turn.get("text") or "")
            if content:
                parts.append(f"{role}: {content}" if role else content)
        elif isinstance(turn, str) and turn.strip():
            parts.append(_compact_text(turn))
    return "\n\n".join(parts)


def render_generic(item: Dict[str, Any]) -> str:
    for key in ("messages", "conversations", "conversation"):
        rendered = _render_messages(item.get(key))
        if rendered:
            return rendered
    text = _first_string(
        item,
        (
            "input",
            "prompt",
            "context",
            "text",
            "document",
            "article",
            "content",
            "question",
        ),
    )
    if text:
        return text
    strings = [_compact_text(value) for value in item.values() if isinstance(value, str) and value.strip()]
    return "\n\n".join(strings[:4])


def render_input(item: Dict[str, Any]) -> str:
    return _first_string(item, ("input", "context", "document", "article", "text")) or render_generic(item)


def render_instruction(item: Dict[str, Any]) -> str:
    parts = []
    for key in ("instruction", "input", "prompt", "context", "question"):
        value = _first_string(item, (key,))
        if value:
            parts.append(value)
    return "\n\n".join(parts) or render_generic(item)


def render_multi_news(item: Dict[str, Any]) -> str:
    return _first_string(item, ("document", "input", "context", "text")) or render_generic(item)


def render_code(item: Dict[str, Any]) -> str:
    text = _first_string(
        item,
        (
            "content",
            "text",
            "code",
            "func_code",
            "whole_func_string",
            "document",
        ),
    )
    if text:
        return text
    return render_generic(item)


def render_conversation(item: Dict[str, Any]) -> str:
    for key in ("conversation", "conversations", "messages"):
        rendered = _render_messages(item.get(key))
        if rendered:
            return rendered
    return render_generic(item)


def _render_table_value(value: Any) -> str:
    if isinstance(value, dict):
        header = value.get("header") or value.get("headers") or value.get("columns")
        rows = value.get("rows") or value.get("data") or value.get("table")
        if isinstance(header, list) and isinstance(rows, list):
            lines = [" | ".join(str(cell) for cell in header)]
            for row in rows[:256]:
                if isinstance(row, list):
                    lines.append(" | ".join(str(cell) for cell in row))
                else:
                    lines.append(str(row))
            return "\n".join(lines)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        lines = []
        for row in value[:256]:
            if isinstance(row, list):
                lines.append(" | ".join(str(cell) for cell in row))
            else:
                lines.append(str(row))
        return "\n".join(lines)
    return _compact_text(value)


def render_table(item: Dict[str, Any]) -> str:
    parts = []
    for key in ("question", "statement", "utterance"):
        value = _first_string(item, (key,))
        if value:
            parts.append(f"Question/statement:\n{value}")
            break
    for key in ("table", "table_text", "table_content", "table_page_title", "context"):
        value = item.get(key)
        if value is not None:
            rendered = _render_table_value(value)
            if rendered:
                parts.append(f"Table/context:\n{rendered}")
                break
    return "\n\n".join(parts) or render_generic(item)


RENDERERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "generic": render_generic,
    "input": render_input,
    "instruction": render_instruction,
    "multi_news": render_multi_news,
    "code": render_code,
    "conversation": render_conversation,
    "table": render_table,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an external long-context calibration JSONL for IMI partition training."
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/imi_partition/calib_external_mix.jsonl"))
    parser.add_argument(
        "--sources",
        type=str,
        default=",".join(DEFAULT_SOURCES),
        help=f"Comma/space separated source names. Available: {','.join(sorted(SOURCE_SPECS))}",
    )
    parser.add_argument("--limit-per-source", type=int, default=64)
    parser.add_argument("--min-chars", type=int, default=2000)
    parser.add_argument("--max-chars", type=int, default=800000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-gated", action="store_true", help="Include gated/very large sources in defaults.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HuggingFace datasets that require custom loading code.",
    )
    parser.add_argument("--the-stack-data-dir", type=str, default="data/python")
    parser.add_argument("--stream-buffer-size", type=int, default=10000)
    return parser.parse_args()


def _parse_sources(value: str, include_gated: bool) -> List[str]:
    names = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if not include_gated:
        names = [name for name in names if name not in {"lmsys_chat", "the_stack_python"}]
    unknown = [name for name in names if name not in SOURCE_SPECS]
    if unknown:
        raise ValueError(f"unknown sources: {unknown}; available={sorted(SOURCE_SPECS)}")
    return names


def _load_dataset(spec: SourceSpec, args: argparse.Namespace):
    from datasets import load_dataset

    kwargs: Dict[str, Any] = {
        "split": spec.split,
        "streaming": bool(spec.streaming),
    }
    if args.hf_cache_dir is not None and not spec.streaming:
        kwargs["cache_dir"] = str(args.hf_cache_dir)
    if bool(args.trust_remote_code):
        kwargs["trust_remote_code"] = True
    if spec.name == "the_stack_python":
        kwargs["data_dir"] = str(args.the_stack_data_dir)
    if spec.subset:
        return load_dataset(spec.dataset, spec.subset, **kwargs)
    return load_dataset(spec.dataset, **kwargs)


def _iter_dataset_rows(dataset: Any, *, streaming: bool, seed: int, buffer_size: int) -> Iterable[Dict[str, Any]]:
    if streaming:
        try:
            dataset = dataset.shuffle(seed=int(seed), buffer_size=max(int(buffer_size), 1))
        except Exception:
            pass
        return iter(dataset)

    try:
        dataset = dataset.shuffle(seed=int(seed))
    except Exception:
        pass
    return iter(dataset)


def _truncate_text(text: str, max_chars: int) -> str:
    if int(max_chars) <= 0 or len(text) <= int(max_chars):
        return text
    keep_head = int(max_chars) // 2
    keep_tail = int(max_chars) - keep_head
    return text[:keep_head].rstrip() + "\n\n[... middle truncated for calibration ...]\n\n" + text[-keep_tail:].lstrip()


def collect_source(spec: SourceSpec, args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset = _load_dataset(spec, args)
    renderer = RENDERERS[spec.renderer]
    limit = int(args.limit_per_source) if int(args.limit_per_source) > 0 else int(spec.default_limit)
    rows: List[Dict[str, Any]] = []
    seen_hashes = set()
    scanned = 0
    max_scan = max(limit * 200, 1000)

    for item in _iter_dataset_rows(
        dataset,
        streaming=bool(spec.streaming),
        seed=int(args.seed),
        buffer_size=int(args.stream_buffer_size),
    ):
        scanned += 1
        if scanned > max_scan and len(rows) >= limit:
            break
        if len(rows) >= limit:
            break
        if not isinstance(item, dict):
            continue
        text = _compact_text(renderer(item))
        if len(text) < int(args.min_chars):
            continue
        text = _truncate_text(text, int(args.max_chars))
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        rows.append(
            {
                "input": text,
                "source_dataset": spec.name,
                "source_hf_dataset": spec.dataset,
                "source_subset": spec.subset,
                "source_split": spec.split,
                "source_id": _stable_id(spec.name, item, text),
                "chars": len(text),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    sources = _parse_sources(str(args.sources), bool(args.include_gated))
    all_rows: List[Dict[str, Any]] = []

    for source_name in sources:
        spec = SOURCE_SPECS[source_name]
        try:
            rows = collect_source(spec, args)
        except Exception as exc:
            message = f"[calib-build] failed source={source_name}: {type(exc).__name__}: {exc}"
            if not bool(args.continue_on_error):
                raise RuntimeError(message) from exc
            print(message, flush=True)
            continue
        all_rows.extend(rows)
        print(f"[calib-build] source={source_name} rows={len(rows)}", flush=True)

    if not all_rows:
        raise SystemExit(
            "[calib-build] no rows collected; check network access, dataset names, "
            "--min-chars, and gated dataset permissions"
        )

    rng = random.Random(int(args.seed))
    rng.shuffle(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[calib-build] wrote {args.out} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
