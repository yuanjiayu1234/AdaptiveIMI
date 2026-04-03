
from .llama import LlamaModel
from .qwen import QwenModel
from .mistral import MistralModel
from transformers import AutoTokenizer
from config import resolve_model_path


def add_model_args(parser):
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, set to `auto` to split model across all available GPUs")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Data type")
    parser.add_argument("--model_name", type=str, default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
                        choices=["Llama-3-8B-Instruct-Gradient-1048k", "Qwen2.5-7B-Instruct",
                                 "Qwen2.5-72B-Instruct", "Llama-3.1-8B-Instruct",
                                 "gradientai/Llama-3-8B-Instruct-Gradient-1048k", "Qwen/Qwen2.5-7B-Instruct",
                                 "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
                                 "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                                 "/data/yjy/.cache/huggingface/hub/Llama-3-8B-Instruct-Gradient-1048k",
                                 "/data/yjy/.cache/huggingface/hub/qwen2.5-7b",
                                 "/data/yjy/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct"], 
                        help="Huggingface model name or local path")
    return parser


def load_tokenizer(model_name):
    model_path = resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name, max_len, dtype, device, tokenizer=None):
    model_path = resolve_model_path(model_name)
    model_name_lower = model_name.lower()
    if 'llama' in model_name_lower:
        llm = LlamaModel(model_name,
                         max_length=max_len,
                         dtype=dtype,
                         device_map=device,
                         tokenizer=tokenizer,
                         model_path=model_path)
    elif 'qwen' in model_name_lower:
        llm = QwenModel(model_name,
                        max_length=max_len,
                        dtype=dtype,
                        device_map=device,
                        tokenizer=tokenizer,
                        model_path=model_path)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return llm
