
from .llama import LlamaModel
from .qwen import QwenModel
from .mistral import MistralModel
from transformers import AutoTokenizer
from config import get_default_model_name, get_model_choices, resolve_model_path


def add_model_args(parser):
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, set to `auto` to split model across all available GPUs")
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
    elif 'mistral' in model_name_lower:
        llm = MistralModel(model_name,
                           max_length=max_len,
                           dtype=dtype,
                           device_map=device,
                           model_path=model_path)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return llm
