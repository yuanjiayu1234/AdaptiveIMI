import os
import sys
import json
import time
import torch
import argparse
import random
import numpy as np
import csv
try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(text, *_args, **_kwargs):
        return text

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from model_hub import load_model, load_tokenizer, add_model_args
from config import generate_config, add_config_args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test example")
    parser.add_argument("--batch_size", type=int, default=1, help="Total Batch size")
    parser.add_argument("--prefill_bsz", type=int, default=1, help="Prefilling batch size")
    parser.add_argument("--prefill_method", type=str, default="full", choices=["full", "xattn", "minfer"],
                        help="Prefilling method")
    parser.add_argument("--context_len", type=int, default=120000, help="Input context length")
    parser.add_argument("--gen_len", type=int, default=100, help="Generation length")
    parser.add_argument("--ignore_eos", action="store_true",
                        help="Ignore EOS token and always generate full length")
    parser.add_argument("--truncate_input", action="store_true",
                        help="Truncate input to context_len tokens before generation")
    parser.add_argument("--result_out", type=str, default="",
                        help="Append CSV results to file")
    parser.add_argument(
        "--task_name",
        type=str,
        default="NIAH",
        choices=[
            "NIAH",
            "fwe",
            "vt",
            "qa1",
            "AIME",
            "longbook_sum_eng_short",
            "longbook_sum_eng_long",
            "gov_report_short",
            "gov_report_long",
        ],
        help="Test task name")
    parser.add_argument("--task_file", type=str, default="",
                        help="Absolute path to a custom task JSON file. If set, this overrides --task_name.")
    parser.add_argument("--measure", action="store_true", help="Report CPU/GPU peak memory and total time")
    parser.add_argument("--measure_out", type=str, default="", help="Append measurement JSON to file")
    parser.add_argument(
        "--breakdown_out",
        type=str,
        default="",
        help="Append decode breakdown CSV row (AdaptiveIMI only)",
    )
    parser.add_argument(
        "--prefill_breakdown_out",
        type=str,
        default="",
        help="Append per-layer prefill breakdown CSV rows (AdaptiveIMI only)",
    )
    parser.add_argument(
        "--index_stats_out",
        type=str,
        default="",
        help="Append index/centroid stats CSV row (AdaptiveIMI only)",
    )
    parser = add_model_args(parser)
    parser = add_config_args(parser)
    args = parser.parse_args()
    return args


def _read_status_value(key):
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(key):
                    value = line.split(":", 1)[1].strip().split()[0]
                    return int(value)
    except FileNotFoundError:
        return 0
    return 0


def _format_gb(value_bytes):
    return round(value_bytes / (1024 ** 3), 4)


def _format_model_name(model_name):
    base_name = os.path.basename(model_name.rstrip("/"))
    if base_name.startswith("models--"):
        parts = base_name.split("--")
        if parts:
            return parts[-1]
    return base_name


def _build_chat_prompt(tokenizer, prompt, model_name):
    model_name_lower = model_name.lower()
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    if "instruct" not in model_name_lower and "llama-3" not in model_name_lower:
        return prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


if __name__ == "__main__":
    args = parse_args()
    set_seed(2025)
    print(args)

    model_name = args.model_name
    model_label = _format_model_name(model_name)
    batch_size = args.batch_size
    attn_type = args.attn_type
    dtype = torch.bfloat16
    device = args.device
    task_name = args.task_name

    TEST_DIR = os.path.join(PROJECT_ROOT, "throughput_eval/test_data")
    if args.task_file:
        if not os.path.isabs(args.task_file):
            raise ValueError(f"--task_file must be an absolute path, got: {args.task_file}")
        TEST_FILE = args.task_file
        task_name = os.path.splitext(os.path.basename(TEST_FILE))[0]
        data = json.load(open(TEST_FILE))
        if isinstance(data, list):
            data = data[0]
        prompt = data.get('input', '')
        if 'outputs' in data:
            groundtruth = data['outputs']
        elif 'answers' in data:
            groundtruth = data['answers']
        else:
            groundtruth = data.get('answer', '')
    elif task_name == "NIAH":
        TEST_FILE = os.path.join(TEST_DIR, f"NIAH_{args.context_len}.json")
        data = json.load(open(TEST_FILE))[0]
        prompt = data['input']
        groundtruth = data['answer']
    else:
        TEST_FILE = os.path.join(TEST_DIR, f"{task_name}.json")
        data = json.load(open(TEST_FILE))
        if isinstance(data, list):
            data = data[0]
        prompt = data.get('input', '')
        if 'outputs' in data:
            groundtruth = data['outputs']
        elif 'answers' in data:
            groundtruth = data['answers']
        else:
            groundtruth = data.get('answer', '')
    prompts = [prompt for _ in range(batch_size)]

    tokenizer = load_tokenizer(model_name)
    prompts = [_build_chat_prompt(tokenizer, p, model_name) for p in prompts]
    if args.truncate_input:
        tokenizer.truncation_side = "right"
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.context_len
        )
    else:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask
    input_len = input_ids.shape[1]
    gen_len = args.gen_len
    max_len = input_len + gen_len
    print(colored(f"Input length: {input_len}, Gen length: {gen_len}", 'yellow'))

    attn_config = generate_config(model_name, input_len, attn_type, 
                                  float(args.retrieval_budget), float(args.estimation_budget), float(args.cache_ratio),
                                  args.use_cuda_graph, args.gpu_only, subspace_parts=args.subspace_parts)
    llm = load_model(model_name, max_len, dtype, device)

    if args.measure and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.measure and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    out = llm.generate(
        attention_type=attn_type,
        inputs_ids=input_ids.to(llm.layers[0].device),
        attention_masks=attention_masks,
        max_new_length=gen_len, 
        attn_config=attn_config,
        do_sample=False, 
        ignore_eos=args.ignore_eos,
        prefill_bsz=args.prefill_bsz,
        prefill_method=args.prefill_method
    )
    if args.measure and torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    
    result = tokenizer.batch_decode(out, skip_special_tokens=True)
    print(result)

    if args.measure:
        cpu_rss_kb = _read_status_value("VmRSS")
        cpu_hwm_kb = _read_status_value("VmHWM")
        gpu_peak_alloc = 0.0
        gpu_peak_reserved = 0.0
        if torch.cuda.is_available():
            gpu_peak_alloc = _format_gb(torch.cuda.max_memory_allocated())
            gpu_peak_reserved = _format_gb(torch.cuda.max_memory_reserved())

        metrics = {
            "model_name": model_name,
            "attn_type": attn_type,
            "batch_size": batch_size,
            "input_len": input_len,
            "gen_len": gen_len,
            "total_time_s": round(total_time, 4),
            "cpu_rss_gb": round(cpu_rss_kb / (1024 ** 2), 4),
            "cpu_hwm_gb": round(cpu_hwm_kb / (1024 ** 2), 4),
            "gpu_peak_alloc_gb": gpu_peak_alloc,
            "gpu_peak_reserved_gb": gpu_peak_reserved,
        }
        print(colored(f"[MEASURE] {json.dumps(metrics, ensure_ascii=False)}", "cyan"))
        if args.measure_out:
            with open(args.measure_out, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    if args.breakdown_out:
        decode_breakdown = None
        kv_cache = getattr(llm, "kv_cache", None)
        if kv_cache is not None:
            decode_breakdown = getattr(kv_cache, "decode_breakdown", None)

        if decode_breakdown is None:
            print(colored("[breakdown] kv_cache.decode_breakdown not available", "yellow"))
        else:
            breakdown_row = {
                "model_name": model_label,
                "attn_type": attn_type,
                "context_len": args.context_len,
                "input_len": input_len,
                "search_ms": round(float(decode_breakdown.get("search_ms", 0.0)), 4),
                "gather_ms": round(float(decode_breakdown.get("gather_ms", 0.0)), 4),
                "attn_ms": round(float(decode_breakdown.get("attn_ms", 0.0)), 4),
                "other_ms": round(float(decode_breakdown.get("other_ms", 0.0)), 4),
                "total_ms": round(float(decode_breakdown.get("total_ms", 0.0)), 4),
            }
            file_exists = os.path.exists(args.breakdown_out)
            with open(args.breakdown_out, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=breakdown_row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(breakdown_row)

    if args.prefill_breakdown_out:
        prefill_breakdown_rows = []
        kv_cache = getattr(llm, "kv_cache", None)
        prefill_gpu_rows_by_layer = {}
        if kv_cache is not None:
            prefill_breakdown_rows = list(getattr(kv_cache, "prefill_breakdown_rows", []) or [])
        for gpu_row in list(getattr(llm, "prefill_gpu_layer_rows", []) or []):
            layer_idx = int(gpu_row.get("layer_idx", -1))
            if layer_idx >= 0:
                prefill_gpu_rows_by_layer[layer_idx] = gpu_row

        if not prefill_breakdown_rows:
            print(colored("[prefill breakdown] kv_cache.prefill_breakdown_rows not available", "yellow"))
        else:
            fieldnames = [
                "model_name",
                "attn_type",
                "context_len",
                "input_len",
                "layer_idx",
                "d2h_ms",
                "cpu_copy_ms",
                "kmeans_ms",
                "kmeans_cpu_time_ms",
                "kmeans_cpu_util_cores",
                "prefill_gpu_attn_ms",
                "prefill_gpu_layer_total_ms",
                "reorganize_ms",
                "write_ms",
                "kmeans_gate_wait_ms",
                "total_chunks",
                "total_tokens",
            ]
            file_exists = os.path.exists(args.prefill_breakdown_out)
            with open(args.prefill_breakdown_out, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for row in prefill_breakdown_rows:
                    gpu_row = prefill_gpu_rows_by_layer.get(int(row.get("layer_idx", -1)), {})
                    writer.writerow({
                        "model_name": model_label,
                        "attn_type": attn_type,
                        "context_len": args.context_len,
                        "input_len": input_len,
                        "layer_idx": int(row.get("layer_idx", -1)),
                        "d2h_ms": round(float(row.get("d2h_ms", 0.0)), 4),
                        "cpu_copy_ms": round(float(row.get("cpu_copy_ms", 0.0)), 4),
                        "kmeans_ms": round(float(row.get("kmeans_ms", 0.0)), 4),
                        "kmeans_cpu_time_ms": round(float(row.get("kmeans_cpu_time_ms", 0.0)), 4),
                        "kmeans_cpu_util_cores": round(float(row.get("kmeans_cpu_util_cores", 0.0)), 4),
                        "prefill_gpu_attn_ms": round(float(gpu_row.get("prefill_gpu_attn_ms", 0.0)), 4),
                        "prefill_gpu_layer_total_ms": round(float(gpu_row.get("prefill_gpu_layer_total_ms", 0.0)), 4),
                        "reorganize_ms": round(float(row.get("reorganize_ms", 0.0)), 4),
                        "write_ms": round(float(row.get("write_ms", 0.0)), 4),
                        "kmeans_gate_wait_ms": round(float(row.get("kmeans_gate_wait_ms", 0.0)), 4),
                        "total_chunks": int(row.get("total_chunks", 0)),
                        "total_tokens": int(row.get("total_tokens", 0)),
                    })

    if args.index_stats_out:
        kv_cache = getattr(llm, "kv_cache", None)
        index_stats = None
        if kv_cache is not None and hasattr(kv_cache, "get_index_stats"):
            index_stats = kv_cache.get_index_stats()

        if index_stats is None:
            print(colored("[index stats] kv_cache.get_index_stats not available", "yellow"))
        else:
            cpu_rss_kb = _read_status_value("VmRSS")
            cpu_hwm_kb = _read_status_value("VmHWM")
            stats_row = {
                "model_name": model_label,
                "attn_type": attn_type,
                "context_len": args.context_len,
                "input_len": input_len,
                "n_centroids": int(index_stats.get("n_centroids", 0)),
                "ready_layers": int(index_stats.get("ready_layers", 0)),
                "total_heads": int(index_stats.get("total_heads", 0)),
                "ideal_clusters_per_head": round(float(index_stats.get("ideal_clusters_per_head", 0.0)), 4),
                "ideal_clusters_total": round(float(index_stats.get("ideal_clusters_total", 0.0)), 4),
                "non_empty_clusters_total": int(index_stats.get("non_empty_clusters_total", 0)),
                "non_empty_ratio": round(float(index_stats.get("non_empty_ratio", 0.0)), 6),
                "total_centroid_bytes_gpu": int(index_stats.get("total_centroid_bytes_gpu", 0)),
                "total_centroid_gib_gpu": round(_format_gb(float(index_stats.get("total_centroid_bytes_gpu", 0))), 4),
                "cpu_rss_gb": round(cpu_rss_kb / (1024 ** 2), 4),
                "cpu_hwm_gb": round(cpu_hwm_kb / (1024 ** 2), 4),
            }
            file_exists = os.path.exists(args.index_stats_out)
            with open(args.index_stats_out, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=stats_row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(stats_row)
