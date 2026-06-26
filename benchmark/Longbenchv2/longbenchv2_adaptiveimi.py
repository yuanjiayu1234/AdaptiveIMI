#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import generate_config  # noqa: E402


DEFAULT_DATA_PATH = "/root/LongBench/data.json"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUT_PATH = PROJECT_ROOT / "benchmark" / "Longbenchv2" / "results" / "llama3_adaptiveimi.jsonl"
HF_CACHE_ROOT = Path(os.getenv("HF_HOME", "/root/.cache/huggingface")) / "hub"

PROMPT_TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LongBench v2 multiple-choice evaluation with AdaptiveIMI.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--config_model_name", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--max_model_len", type=int, default=131072)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bf16", choices=("fp16", "bf16"))
    parser.add_argument(
        "--attn_type",
        type=str,
        default="AdaptiveIMI",
        choices=("AdaptiveIMI", "Full_Flash_Attn", "Full_Flash_Attn_Offload"),
    )
    parser.add_argument("--retrieval_budget", type=float, default=0.018)
    parser.add_argument("--cache_ratio", type=float, default=0.0)
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument("--subspace_parts", type=int, default=2)
    parser.add_argument("--imi_engine", type=str, default=None)
    parser.add_argument("--partition_path", type=str, default=os.getenv("IMI_PARTITION_PATH"))
    parser.add_argument("--hybrid_policy", type=str, default=None)
    parser.add_argument("--hybrid_start", type=str, default="cpu")
    parser.add_argument("--hybrid_gpu_every", type=int, default=0)
    parser.add_argument("--decode_search_mode", type=str, default=None, choices=("flat", "subspace_rerank"))
    parser.add_argument("--candidate_clusters", type=int, default=None)
    parser.add_argument("--token_rerank_budget_ratio", type=float, default=None)
    parser.add_argument("--prefill_bsz", type=int, default=1)
    parser.add_argument("--chat_template", action="store_true", default=True)
    parser.add_argument("--no_chat_template", action="store_false", dest="chat_template")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_context_chars", type=int, default=1000)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise TypeError(f"Unsupported LongBench data type: {type(data).__name__}")


def render_prompt(item: dict[str, Any]) -> str:
    return (
        PROMPT_TEMPLATE.replace("$DOC$", str(item["context"]).strip())
        .replace("$Q$", str(item["question"]).strip())
        .replace("$C_A$", str(item["choice_A"]).strip())
        .replace("$C_B$", str(item["choice_B"]).strip())
        .replace("$C_C$", str(item["choice_C"]).strip())
        .replace("$C_D$", str(item["choice_D"]).strip())
    )


def apply_chat_template(tokenizer, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_with_middle_truncation(tokenizer, prompt: str, max_input_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > max_input_tokens:
        keep_head = max_input_tokens // 2
        keep_tail = max_input_tokens - keep_head
        input_ids = input_ids[:keep_head] + input_ids[-keep_tail:]
    ids = torch.tensor([input_ids], dtype=torch.long)
    return ids, torch.ones_like(ids)


def extract_answer(text: str) -> str | None:
    normalized = text.replace("*", "")
    if "</think>" in normalized:
        normalized = normalized.rsplit("</think>", 1)[-1]
    elif normalized.lstrip().startswith("<think>"):
        return None
    patterns = (
        r"The correct answer is\s*\(([A-D])\)",
        r"The correct answer is\s*([A-D])",
        r"answer is\s*\(([A-D])\)",
        r"answer is\s*([A-D])",
        r"\(([A-D])\)",
        r"\b([A-D])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def completed_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = row.get("_id")
            if item_id is not None:
                done.add(str(item_id))
    return done


def cleanup_after_sample(llm: Any) -> None:
    kv_cache = getattr(llm, "kv_cache", None)
    if kv_cache is not None:
        try:
            cleanup = getattr(kv_cache, "cleanup", None)
            if callable(cleanup):
                cleanup()
        except Exception as exc:
            print(f"Warning: kv_cache cleanup failed: {exc}", file=sys.stderr, flush=True)
        finally:
            llm.kv_cache = None

    clear_prefill_workspace = getattr(llm, "_clear_prefill_prefix_kv_workspace", None)
    if callable(clear_prefill_workspace):
        try:
            clear_prefill_workspace()
        except Exception as exc:
            print(f"Warning: prefill workspace cleanup failed: {exc}", file=sys.stderr, flush=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def resolve_local_hf_snapshot(model_name: str) -> str:
    if os.path.exists(model_name) or "/" not in model_name:
        return model_name

    org, repo = model_name.split("/", 1)
    cache_dir = HF_CACHE_ROOT / f"models--{org}--{repo}"
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return model_name

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return model_name
    snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(snapshots[0])


def build_hybrid_index_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.hybrid_policy and args.hybrid_gpu_every <= 0:
        return None
    cfg: dict[str, Any] = {
        "policy": args.hybrid_policy or "gpu_every_n",
        "start": args.hybrid_start,
    }
    if args.hybrid_gpu_every > 0:
        cfg["gpu_every"] = args.hybrid_gpu_every
    return cfg


def build_decode_search_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.decode_search_mode:
        return None
    cfg: dict[str, Any] = {"mode": args.decode_search_mode}
    if args.candidate_clusters is not None:
        cfg["candidate_clusters"] = int(args.candidate_clusters)
    if args.token_rerank_budget_ratio is not None:
        cfg["token_rerank_budget_ratio"] = float(args.token_rerank_budget_ratio)
    return cfg


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    from model_hub import load_model, load_tokenizer

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    config_model_name = args.config_model_name or args.model_name
    model_name_or_path = resolve_local_hf_snapshot(args.model_name)
    tokenizer = load_tokenizer(model_name_or_path)
    llm = load_model(model_name_or_path, args.max_model_len, dtype, args.device, tokenizer)

    data = load_data(args.data_path)[args.start :]
    if args.max_samples > 0:
        data = data[: args.max_samples]

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(str(out_path))
    max_input_tokens = args.max_model_len - args.max_new_tokens

    with out_path.open("a", encoding="utf-8") as fout:
        for item in tqdm(data):
            item_id = str(item.get("_id"))
            if item_id in done:
                continue

            prompt = None
            input_ids = None
            attention_mask = None
            output_ids = None
            attn_config = None
            response = None
            record = None

            try:
                prompt = render_prompt(item)
                if args.chat_template:
                    prompt = apply_chat_template(tokenizer, prompt, args.enable_thinking)
                input_ids, attention_mask = tokenize_with_middle_truncation(tokenizer, prompt, max_input_tokens)
                input_len = int(input_ids.shape[1])

                attn_config = generate_config(
                    config_model_name,
                    input_len,
                    args.attn_type,
                    retrieval_budget=args.retrieval_budget,
                    cache_ratio=args.cache_ratio,
                    gpu_only=args.gpu_only,
                    subspace_parts=args.subspace_parts,
                    imi_engine=args.imi_engine,
                    partition_path=args.partition_path,
                    hybrid_index=build_hybrid_index_config(args),
                    decode_search=build_decode_search_config(args),
                )

                output_ids = llm.generate(
                    attention_type=args.attn_type,
                    inputs_ids=input_ids.to(llm.layers[0].device),
                    attention_masks=attention_mask.to(llm.layers[0].device),
                    max_new_length=args.max_new_tokens,
                    attn_config=attn_config,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    ignore_eos=args.ignore_eos,
                    prefill_bsz=args.prefill_bsz,
                )

                response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                pred = extract_answer(response)
                answer = str(item.get("answer", "")).strip().upper()

                record = dict(item)
                if args.save_context_chars >= 0:
                    record["context"] = str(item.get("context", ""))[: args.save_context_chars]
                record.update(
                    {
                        "response": response,
                        "pred": pred,
                        "judge": pred == answer,
                        "input_tokens": input_len,
                        "model_name": args.model_name,
                        "model_path": model_name_or_path,
                        "attn_type": args.attn_type,
                        "imi_engine": args.imi_engine,
                    }
                )
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                done.add(item_id)
            finally:
                del prompt, input_ids, attention_mask, output_ids, attn_config, response, record
                cleanup_after_sample(llm)


if __name__ == "__main__":
    main()
