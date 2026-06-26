# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import gc
import json
import yaml
import os
import sys
import threading
import importlib
import time
import torch
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path

from config import resolve_model_path
from typing import Dict, List
import traceback
from utils import load_data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT)
from model_hub import LlamaModel
try:
    from model_hub import QwenModel
except ImportError:
    QwenModel = None

from config import generate_config



SERVER_TYPES = (
    'trtllm',
    'vllm',
    'openai',
    'gemini',
    'hf',
    'mamba',
)


def load_existing_pred_indices(pred_file: Path) -> set:
    indices = set()
    if not pred_file.exists():
        return indices

    valid_lines = []
    saw_invalid = False
    with pred_file.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                saw_invalid = True
                continue
            valid_lines.append(line)
            if "index" in sample:
                indices.add(sample["index"])

    if saw_invalid:
        with pred_file.open("w", encoding="utf-8") as fout:
            fout.writelines(valid_lines)
        print(f"Removed malformed partial lines from {pred_file}")

    return indices

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def cleanup_kv_cache(llm, force: bool = False):
    if os.getenv("IMI_DEBUG_SKIP_KV_CLEANUP", "0") == "1":
        return
    kv_cache = getattr(llm, "kv_cache", None)
    if kv_cache is not None:
        reuse_ok = hasattr(kv_cache, "reset_for_next_sequence") and hasattr(kv_cache, "can_reuse_for_next_sequence")
        if force or not reuse_ok:
            cleanup = getattr(kv_cache, "cleanup", None)
            if callable(cleanup):
                cleanup()
            llm.kv_cache = None
    clear_prefill_workspace = getattr(llm, "_clear_prefill_prefix_kv_workspace", None)
    if callable(clear_prefill_workspace):
        clear_prefill_workspace()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class HuggingFaceModel:
    def __init__(
        self,
        model_name,
        model_path,
        max_len,
        max_new_len,
        attn_type,
        dtype,
        device,
        synthetic_len,
        budget_ratio=None,
        subspace_parts=2,
        imi_engine=None,
        partition_path=None,
        decode_search_mode=None,
        candidate_clusters=None,
        token_rerank_budget_ratio=None,
        stop_at_period=False,
    ) -> None:
        load_path = model_path or model_name

        if 'llama' in model_name.lower():
            llm = LlamaModel(model_name,
                max_length=max_len+max_new_len,
                dtype=dtype,
                device_map=device,
                model_path=load_path)
        elif 'qwen' in model_name.lower():
            llm = QwenModel(model_name,
                max_length=max_len+max_new_len,
                dtype=dtype,
                device_map=device,
                model_path=load_path)
        elif 'mistral' in model_name.lower():
            from model_hub import MistralModel
            llm = MistralModel(model_name,
                max_length=max_len+max_new_len,
                dtype=dtype,
                device_map=device,
                model_path=load_path)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.llm = llm
        self.max_new_len = max_new_len
        self.attn_type = attn_type

        self.model_name = model_name
        self.budget_ratio = budget_ratio
        self.synthetic_len = synthetic_len
        self.subspace_parts = subspace_parts
        self.imi_engine = imi_engine
        self.partition_path = partition_path
        self.decode_search_mode = decode_search_mode
        self.candidate_clusters = candidate_clusters
        self.token_rerank_budget_ratio = token_rerank_budget_ratio
        self.stop_at_period = stop_at_period
        self._generate_lock = threading.Lock()

    def __call__(self, prompt: str, answer_prefix: str = "", **kwargs) -> Dict[str, List[str]]:
        with self._generate_lock:
            generated_text = get_pred(
                self.llm,
                input_text=prompt,
                max_new_tokens=self.max_new_len,
                attn_type=self.attn_type,
                model_name=self.model_name,
                budget_ratio=self.budget_ratio,
                synthetic_len=self.synthetic_len,
                answer_prefix=answer_prefix,
                subspace_parts=self.subspace_parts,
                imi_engine=self.imi_engine,
                partition_path=self.partition_path,
                decode_search_mode=self.decode_search_mode,
                candidate_clusters=self.candidate_clusters,
                token_rerank_budget_ratio=self.token_rerank_budget_ratio,
                stop_at_period=self.stop_at_period,
            )

        return {'text': [generated_text]}


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


def get_llm(
    model_name,
    max_len,
    max_new_len,
    attn_type,
    dtype,
    device,
    synthetic_len,
    budget_ratio=None,
    model_path=None,
    subspace_parts=2,
    imi_engine=None,
    partition_path=None,
    decode_search_mode=None,
    candidate_clusters=None,
    token_rerank_budget_ratio=None,
    stop_at_period=False,
):
    if args.server_type == 'hf':
        llm = HuggingFaceModel(
            model_name=model_name,
            model_path=model_path,
            max_len=max_len,
            max_new_len=max_new_len,
            attn_type=attn_type,
            dtype=dtype,
            device=device,
            budget_ratio=budget_ratio,
            synthetic_len=synthetic_len,
            subspace_parts=subspace_parts,
            imi_engine=imi_engine,
            partition_path=partition_path,
            decode_search_mode=decode_search_mode,
            candidate_clusters=candidate_clusters,
            token_rerank_budget_ratio=token_rerank_budget_ratio,
            stop_at_period=stop_at_period,
        )
    else:
        raise RuntimeError(f'Unsupported server type {args.server_type}')

    return llm


def env_flag(value: str | None) -> bool | None:
    if value is None:
        return None
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on", "always"):
        return True
    if value in ("0", "false", "no", "off", "never"):
        return False
    return None


def should_stop_at_period(task_name: str, task_config: dict) -> bool:
    """Return whether period should terminate generation for this task."""
    override = env_flag(os.getenv("RULER_STOP_AT_PERIOD"))
    if override is not None:
        return override

    # Auto mode: only use '.' as a stop token for NIAH tasks with one requested
    # value. Multi-value/multi-query and extraction tasks often answer as
    # numbered lists, so stopping at '.' truncates them to "1".
    if task_config.get("task") != "niah":
        return False

    args_cfg = task_config.get("args", {})
    try:
        num_needle_v = int(args_cfg.get("num_needle_v", 1))
        num_needle_q = int(args_cfg.get("num_needle_q", 1))
    except (TypeError, ValueError):
        return False

    return num_needle_v == 1 and num_needle_q == 1


def postprocess_pred(pred: str, stop_at_period: bool = False) -> str:
    """Post-process prediction to improve string matching accuracy."""
    # Some chat models continue by emitting prompt/control text again after the
    # short RULER answer. Drop that suffix before task-level scoring/logging.
    pred = pred.split('[/INST]', 1)[0].strip()
    pred = pred.split('<|eot_id|>', 1)[0].strip()
    pred = pred.split('<|end_of_turn|>', 1)[0].strip()

    if stop_at_period:
        period_idx = pred.find('.')
        if period_idx >= 0:
            pred = pred[: period_idx + 1].strip()

    if pred.startswith(': '):
        pred = pred[2:].strip()
    elif pred.startswith(':'):
        pred = pred[1:].strip()
    if pred.endswith('.'):
        pred = pred[:-1].rstrip()
    return pred


def fix_begin_of_text(prompt: str) -> str:
    """Fix data generation bug: some samples have multiple <|begin_of_text|> tags."""
    if prompt.count('<|begin_of_text|>') > 1:
        parts = prompt.split('<|begin_of_text|>')
        prompt = '<|begin_of_text|>' + parts[-1]
    return prompt


def get_pred(
    llm,
    input_text: str,
    max_new_tokens: int,
    attn_type: str,
    model_name: str,
    budget_ratio: float | None,
    synthetic_len: int,
    answer_prefix: str = "",
    subspace_parts: int = 2,
    imi_engine: str | None = None,
    partition_path: str | None = None,
    decode_search_mode: str | None = None,
    candidate_clusters: int | None = None,
    token_rerank_budget_ratio: float | None = None,
    stop_at_period: bool = False,
) -> str:

    # Fix multiple <|begin_of_text|> tags
    input_text = fix_begin_of_text(input_text)

    # Append answer_prefix to guide model generation format
    if answer_prefix:
        input_text = input_text + answer_prefix

    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.tokenizer.padding_side = "left"
    inputs = llm.tokenizer([input_text], return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask

    decode_search = None
    if decode_search_mode:
        decode_search = {"mode": decode_search_mode}
        if candidate_clusters is not None:
            decode_search["candidate_clusters"] = int(candidate_clusters)
        if token_rerank_budget_ratio is not None:
            decode_search["token_rerank_budget_ratio"] = float(token_rerank_budget_ratio)

    attn_config = generate_config(
        model_name,
        input_ids.shape[1],  # 使用实际输入长度，而非预设的 synthetic_len
        attn_type,
        retrieval_budget=budget_ratio,
        subspace_parts=subspace_parts,
        imi_engine=imi_engine,
        partition_path=partition_path,
        decode_search=decode_search,
    )
    stop_token_ids = None
    if stop_at_period:
        stop_token_ids = llm.tokenizer.encode(".", add_special_tokens=False)
    success = False
    try:
        out = llm.generate(
            attention_type=attn_type,
            inputs_ids=input_ids.to(llm.layers[0].device),
            attention_masks=attention_masks.to(llm.layers[0].device),
            max_new_length=max_new_tokens,
            attn_config=attn_config,
            ignore_eos=False,
            stop_token_ids=stop_token_ids,
        )

        output = llm.tokenizer.batch_decode(out, skip_special_tokens=True)
        success = True
    finally:
        cleanup_kv_cache(llm, force=not success)

    processed_output = postprocess_pred(output[0], stop_at_period=stop_at_period)
    print("Chunked generation:", processed_output)
    return processed_output


def get_output(llm, outputs_parallel, idx, index, input, outputs, others, truncation, length, answer_prefix="", stop_at_period=False):
    while True:
        try:
            pred = llm(prompt=input, answer_prefix=answer_prefix)
            break
        except Exception as e:
            traceback.print_exc()
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise

    if len(pred['text']) > 0:
        processed_pred = postprocess_pred(pred['text'][0], stop_at_period=stop_at_period)
        outputs_parallel[idx] = {
            'index': index,
            'pred': processed_pred,
            'input': input,
            'outputs': outputs,
            'others': others,
            'truncation': truncation,
            'length': length,
        }


def main(args):
    start_time = time.time()
    
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])
    stop_at_period = should_stop_at_period(args.task, config)

    task_file = args.data_dir / args.task / f'{args.subset}.jsonl'
    
    if args.chunk_amount > 1:
        pred_file = args.save_dir / f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir / f'{args.task}.jsonl'
        
    args.model_path = resolve_model_path(args.model_path or args.model_name)
    print(args)
    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    print(f"RULER period stop for {args.task}: {stop_at_period}")
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = load_existing_pred_indices(pred_file)
        data = [sample for sample in load_data(task_file) if sample['index'] not in pred_index]
    else:
        data = load_data(task_file)

    if len(data) == 0:
        print(f"Prediction for {args.task} is already complete; skip model loading.")
        return

    # Load api
    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    llm = get_llm(
        args.model_name,
        args.max_len,
        config['tokens_to_generate'],
        args.attn_type,
        dtype,
        args.device,
        budget_ratio=args.budget_ratio,
        synthetic_len=args.synthetic_len,
        model_path=args.model_path,
        subspace_parts=args.subspace_parts,
        imi_engine=args.imi_engine,
        partition_path=args.partition_path,
        decode_search_mode=args.decode_search_mode,
        candidate_clusters=args.candidate_clusters,
        token_rerank_budget_ratio=args.token_rerank_budget_ratio,
        stop_at_period=stop_at_period,
    )

    threads = []
    outputs_parallel = [{} for _ in range(len(data))]
    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        for idx, data_point in tqdm(enumerate(data), total=len(data)):
            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    llm=llm,
                    outputs_parallel=outputs_parallel,
                    idx=idx,
                    index=data_point['index'],
                    input=data_point['input'],
                    outputs=data_point['outputs'],
                    others=data_point.get('others', {}),
                    truncation=data_point.get('truncation', -1),
                    length=data_point.get('length', -1),
                    answer_prefix=data_point.get('answer_prefix', ''),
                    stop_at_period=stop_at_period,
                ),
            )
            thread.start()
            threads.append(thread)
            if len(threads) == args.threads:
                for thread in threads:
                    thread.join()
                threads = []
                for computed_idx in range(idx - args.threads + 1, idx + 1):
                    if len(outputs_parallel[computed_idx]) > 0:
                        fout.write(json.dumps(outputs_parallel[computed_idx]) + '\n')

        # collecting the final batch
        if len(data) > 0:
            for thread in threads:
                thread.join()
            for computed_idx in range(idx - len(threads) + 1, idx + 1):
                if len(outputs_parallel[computed_idx]) > 0:
                    fout.write(json.dumps(outputs_parallel[computed_idx]) + '\n')

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
    parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
    parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
    parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
    parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
    parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
    parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

    # Server
    parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
    parser.add_argument("--server_host", type=str, default='127.0.0.1')
    parser.add_argument("--server_port", type=str, default='5000')
    parser.add_argument("--ssh_server", type=str)
    parser.add_argument("--ssh_key_path", type=str)

    # Inference
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b",                                            \
            choices=[
                "qwen2.5-7b",
                "qwen3-8b",
                "Qwen/Qwen3-8B",
                "llama-3-8b-1048k",
                "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
                "Llama-3-8B-Instruct-Gradient-1048k",
                "llama-3.1-8b",
                "meta-llama/Llama-3.1-8B-Instruct",
                "Llama-3.1-8B-Instruct",
                "llama3-8b",
                "qwen2.5-72b",
                "mistral-7b-Instruct-32k",
            ], \
            help="simplified model name (matches config file names)")
    parser.add_argument("--model_path", type=str, default=None, help="Local path override for HF weights/tokenizer")
    parser.add_argument("--attn_type", type=str, default="Full_Flash_Attn",                                                      \
            choices=["Full_Flash_Attn", "Full_Flash_Attn_Offload", "AdaptiveIMI"],                                  \
            help="Attention method")
    parser.add_argument("--max_len", type=int, default=128000)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp16",choices=["fp16", "bf16"])

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--sliding_window_size", type=int)
    parser.add_argument("--threads", type=int, default=4)

    parser.add_argument("--synthetic_len", type=int, required=True)
    parser.add_argument("--budget_ratio", type=float, default=None, help="Retrieval budget override for AdaptiveIMI")
    parser.add_argument("--subspace_parts", type=int, default=2, help="IMI subspace parts (0, 2 or 4)")
    parser.add_argument("--imi_engine", type=str, default=None, choices=[None, "hybrid_8way_v2"])
    parser.add_argument("--partition_path", type=str, default=None, help="Learned partition JSON for AdaptiveIMI hybrid v2")
    parser.add_argument(
        "--decode_search_mode",
        type=str,
        default=None,
        choices=[None, "flat", "subspace_rerank"],
    )
    parser.add_argument("--candidate_clusters", type=int, default=None)
    parser.add_argument("--token_rerank_budget_ratio", type=float, default=None, help="Final token rerank budget ratio; <=0 disables token rerank")

    args = parser.parse_args()
    print(args)

    if args.server_type == 'hf' or args.server_type == 'gemini':
        args.threads = 1
    
    seed_everything(2025)
    main(args)
