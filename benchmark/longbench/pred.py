import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import generate_config, resolve_model_path  # noqa: E402


def load_json(name):
    with (SCRIPT_DIR / "config" / name).open("r", encoding="utf-8") as fin:
        return json.load(fin)


model2path = load_json("model2path.json")
model2maxlen = load_json("model2maxlen.json")
dataset2prompt = load_json("dataset2prompt.json")
dataset2maxlen = load_json("dataset2maxlen.json")

LONG_BENCH_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    # "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    # "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    # "lsht",
    "lcc",
    "repobench-p",
    "passage_retrieval_en",
    "passage_count",
    "passage_retrieval_zh",
   
]


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b",
        choices=[
            "llama-3-8b-1048k",
            "qwen2.5-7b",
            "llama-3.1-8b",
            "qwen2.5-72b",
            "deepseek32b-fp8",
            "deepseek-r1-distill-qwen-32b-fp8-dynamic",
        ],
    )
    parser.add_argument("--model_path", type=str, default=None, help="HF model path. Overrides config/model2path.json.")
    parser.add_argument("--config_model_name", type=str, default=None, help="Model key used to find AdaptiveIMI config.")
    parser.add_argument("--data_dir", type=str, default="/root/LongBench/data")
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, required=True, help="Task name, category, or all.")
    parser.add_argument("--num_examples", type=int, default=-1, help="Number of examples per task. -1 for all.")
    parser.add_argument("--attn_type", type=str, default="AdaptiveIMI", choices=["Full_Flash_Attn", "Full_Flash_Attn_Offload", "AdaptiveIMI"])
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--parallel_mode", "--parallel-mode", type=str, default="auto", choices=["auto", "single", "layer", "tp"])
    parser.add_argument("--weight_loading", "--weight-loading", type=str, default="hf", choices=["hf", "auto", "accelerate_meta", "streaming_safetensors"])
    parser.add_argument("--gpu_memory_utilization", "--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--retrieval_budget", type=float, default=None)
    parser.add_argument("--cache_ratio", type=float, default=None)
    parser.add_argument("--gpu_only", action="store_true", help="Use GPU-only hybrid index build policy.")
    parser.add_argument(
        "--subspace_parts",
        type=int,
        default=None,
        help="Compatibility option; current AdaptiveIMI runtime supports only 8-way learned partitions.",
    )
    parser.add_argument("--imi_engine", type=str, default=None, choices=[None, "hybrid_8way_v2"])
    parser.add_argument("--partition_path", type=str, default=None)
    parser.add_argument("--decode_search_mode", type=str, default=None, choices=["flat", "subspace_rerank"])
    parser.add_argument("--candidate_clusters", type=int, default=None)
    parser.add_argument("--token_rerank_budget_ratio", type=float, default=None)
    parser.add_argument("--prefill_bsz", type=int, default=1)
    parser.add_argument(
        "--chat_template_mode",
        type=str,
        default="longbench",
        choices=["longbench", "always", "never"],
        help="LongBench applies chat formatting only for chat models and skips few-shot/code tasks.",
    )
    parser.add_argument("--disable_resume", action="store_true", help="Regenerate predictions even if output jsonl exists.")
    return parser.parse_args(args)


def resolve_tasks(task, e=False):
    groups = {
        "SQA": ["qasper", "multifieldqa_en", "narrativeqa"],
        "MQA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
        "SUM": ["gov_report", "qmsum", "multi_news", "vcsum"],
        "FSL": ["trec", "lsht", "samsum", "triviaqa"],
        "ST": ["passage_retrieval_en", "passage_count", "passage_retrieval_zh"],
        "CC": ["repobench-p", "lcc"],
    }
    if task == "all":
        tasks = list(LONG_BENCH_TASKS)
    else:
        tasks = groups.get(task.upper(), [task])
    if e:
        unsupported = {"multifieldqa_zh", "musique", "dureader", "qmsum", "vcsum", "lsht", "passage_retrieval_zh"}
        tasks = [name for name in tasks if name not in unsupported]
    return tasks


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def dtype_from_arg(dtype):
    return torch.float16 if dtype == "fp16" else torch.bfloat16


def load_model(model_name, model_path, max_len, dtype, device, parallel_mode="auto", weight_loading="hf", gpu_memory_utilization=0.90):
    from model_hub import LlamaModel, QwenModel, QwenFp8Model, is_fp8_compressed_model

    model_path_lower = model_path.lower()
    model_name_lower = model_name.lower()
    if "llama" in model_path_lower or "llama" in model_name_lower:
        llm = LlamaModel(
            model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device,
            model_path=model_path,
            parallel_mode=parallel_mode,
            weight_loading=weight_loading,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    elif "qwen" in model_path_lower or "qwen" in model_name_lower:
        model_cls = QwenFp8Model if is_fp8_compressed_model(model_path) else QwenModel
        llm = model_cls(
            model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device,
            model_path=model_path,
            parallel_mode=parallel_mode,
            weight_loading=weight_loading,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.tokenizer.padding_side = "left"
    return llm


def load_local_dataset(data_dir, dataset, e=False):
    suffix = "_e" if e else ""
    path = Path(data_dir) / f"{dataset}{suffix}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"LongBench dataset file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def completed_count(path):
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                count += 1
    return count


LONG_BENCH_NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def is_chat_model(model_name, model_path):
    text = f"{model_name} {model_path}".lower()
    return any(key in text for key in ("chat", "instruct", "longchat", "vicuna"))


def should_apply_chat_template(args, dataset):
    if args.chat_template_mode == "always":
        return True
    if args.chat_template_mode == "never":
        return False
    if dataset in LONG_BENCH_NO_CHAT_TASKS:
        return False
    return is_chat_model(args.config_model_name or args.model, args.model_path or "")


def apply_chat_template(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def build_decode_search_config(args):
    if not args.decode_search_mode:
        if args.candidate_clusters is not None or args.token_rerank_budget_ratio is not None:
            raise ValueError("candidate_clusters/token_rerank_budget_ratio require an implemented decode_search mode")
        return None
    if args.decode_search_mode != "flat":
        raise ValueError("Online subspace_rerank decode search is not implemented in this runtime")
    if args.candidate_clusters is not None or args.token_rerank_budget_ratio is not None:
        raise ValueError("candidate_clusters/token_rerank_budget_ratio are not used by flat decode search")
    return None


def middle_truncate(input_ids, attention_mask, max_input_tokens):
    if input_ids.shape[1] <= max_input_tokens:
        return input_ids, attention_mask
    keep_head = max_input_tokens // 2
    keep_tail = max_input_tokens - keep_head
    input_ids = torch.cat([input_ids[:, :keep_head], input_ids[:, -keep_tail:]], dim=1)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def get_pred(llm, data, max_new_tokens, prompt_format, dataset, config_model_name, out_path, args):
    start_idx = 0 if args.disable_resume else completed_count(out_path)
    if start_idx:
        print(f"Resuming {out_path.name}: skip {start_idx} completed predictions")

    with out_path.open("a", encoding="utf-8") as fout:
        for json_obj in tqdm(data[start_idx:]):
            prompt = prompt_format.format(**json_obj)
            if should_apply_chat_template(args, dataset):
                prompt = apply_chat_template(llm.tokenizer, prompt)

            inputs = llm.tokenizer([prompt], return_tensors="pt", padding=True)
            input_ids = inputs.input_ids
            attention_masks = inputs.attention_mask
            max_input_tokens = args.max_len - max_new_tokens
            if max_input_tokens <= 0:
                raise ValueError(f"max_len ({args.max_len}) must be larger than max_new_tokens ({max_new_tokens})")
            input_ids, attention_masks = middle_truncate(input_ids, attention_masks, max_input_tokens)

            attn_config = generate_config(
                config_model_name,
                input_ids.shape[1],
                args.attn_type,
                retrieval_budget=args.retrieval_budget,
                cache_ratio=args.cache_ratio,
                gpu_only=args.gpu_only,
                subspace_parts=args.subspace_parts,
                imi_engine=args.imi_engine,
                partition_path=args.partition_path,
                decode_search=build_decode_search_config(args),
            )

            out = llm.generate(
                attention_type=args.attn_type,
                inputs_ids=input_ids.to(llm.layers[0].device),
                attention_masks=attention_masks.to(llm.layers[0].device),
                max_new_length=max_new_tokens,
                attn_config=attn_config,
                do_sample=False,
                ignore_eos=False,
                prefill_bsz=args.prefill_bsz,
            )

            pred = llm.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            torch.cuda.empty_cache()

            json.dump(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                },
                fout,
                ensure_ascii=False,
            )
            fout.write("\n")
            fout.flush()


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    model_name = args.model
    config_model_name = args.config_model_name or model_name
    model_path = args.model_path or os.getenv("MODEL_PATH_OVERRIDE") or resolve_model_path(config_model_name) or model2path[model_name]
    max_length = args.max_len or model2maxlen[model_name]
    args.model_path = model_path
    args.max_len = max_length

    llm = load_model(
        config_model_name,
        model_path,
        max_length,
        dtype_from_arg(args.dtype),
        args.device,
        parallel_mode=args.parallel_mode,
        weight_loading=args.weight_loading,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    datasets = resolve_tasks(args.task, args.e)

    pred_root = Path(args.output_dir) / ("pred_e" if args.e else "pred") / model_name / args.attn_type
    pred_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        print(f"Predict {dataset}")
        data = load_local_dataset(args.data_dir, dataset, args.e)
        if args.num_examples > 0:
            data = data[: args.num_examples]

        out_path = pred_root / f"{dataset}.jsonl"
        if args.disable_resume and out_path.exists():
            out_path.unlink()

        get_pred(
            llm,
            data,
            dataset2maxlen[dataset],
            dataset2prompt[dataset],
            dataset,
            config_model_name,
            out_path,
            args,
        )
