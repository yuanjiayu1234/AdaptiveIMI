
from .llama import LlamaModel
from .qwen import QwenModel
from .qwen_fp8 import QwenFp8Model
from .mistral import MistralModel
from transformers import AutoConfig, AutoTokenizer
from config import get_default_model_name, get_model_choices, resolve_model_path


def is_fp8_compressed_model(model_path: str) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(model_path)
    except Exception:
        return False
    quant = getattr(cfg, "quantization_config", None) or {}
    if not isinstance(quant, dict):
        return False
    if str(quant.get("quant_method", "")).lower() != "compressed-tensors":
        return False
    if str(quant.get("format", "")).lower() != "float-quantized":
        return False
    for group in (quant.get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        if str(weights.get("type", "")).lower() == "float" and int(weights.get("num_bits", 0) or 0) == 8:
            return True
    return False


def add_model_args(parser):
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, set to `auto` to split model across all available GPUs")
    parser.add_argument("--parallel_mode", "--parallel-mode", type=str, default="auto", choices=["auto", "single", "layer", "tp"], help="Model parallel placement policy")
    parser.add_argument("--weight_loading", "--weight-loading", type=str, default="hf", choices=["hf", "auto", "accelerate_meta", "streaming_safetensors"], help="Weight loading backend")
    parser.add_argument("--gpu_memory_utilization", "--gpu-memory-utilization", type=float, default=0.90, help="Per-GPU memory budget used by the device planner")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Data type")
    parser.add_argument("--model_name", type=str, default=get_default_model_name(),
                        choices=get_model_choices(),
                        help="Local model name or local path")
    return parser


def load_tokenizer(model_name):
    model_path = resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(
    model_name,
    max_len,
    dtype,
    device,
    tokenizer=None,
    parallel_mode="auto",
    weight_loading="hf",
    gpu_memory_utilization=0.90,
):
    model_path = resolve_model_path(model_name)
    model_probe = f"{model_name} {model_path}".lower()
    fp8_compressed = is_fp8_compressed_model(model_path)
    if 'llama' in model_probe:
        llm = LlamaModel(model_name,
                         max_length=max_len,
                         dtype=dtype,
                         device_map=device,
                         tokenizer=tokenizer,
                         model_path=model_path,
                         parallel_mode=parallel_mode,
                         weight_loading=weight_loading,
                         gpu_memory_utilization=gpu_memory_utilization)
    elif 'qwen' in model_probe:
        model_cls = QwenFp8Model if fp8_compressed else QwenModel
        llm = model_cls(model_name,
                        max_length=max_len,
                        dtype=dtype,
                        device_map=device,
                        tokenizer=tokenizer,
                        model_path=model_path,
                        parallel_mode=parallel_mode,
                        weight_loading=weight_loading,
                        gpu_memory_utilization=gpu_memory_utilization)
    elif 'mistral' in model_probe:
        llm = MistralModel(model_name,
                           max_length=max_len,
                           dtype=dtype,
                           device_map=device,
                           tokenizer=tokenizer,
                           model_path=model_path,
                           parallel_mode=parallel_mode,
                           weight_loading=weight_loading,
                           gpu_memory_utilization=gpu_memory_utilization)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return llm
