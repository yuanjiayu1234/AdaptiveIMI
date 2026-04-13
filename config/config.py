import copy
import os, json
from pathlib import Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_REGISTRY = {
    "Llama-3-8B-Instruct-Gradient-1048k": {
        "path": os.path.join("models", "llama3-8b-1048k"),
        "config_name": "Llama-3-8B-Instruct-Gradient-1048k.json",
    },
    "Qwen2.5-7B-Instruct": {
        "path": os.path.join("models", "qwen2.5-7b"),
        "config_name": "Qwen2.5-7B-Instruct.json",
    },
    "Llama-3.1-8B-Instruct": {
        "path": os.path.join("models", "Llama-3.1-8B-Instruct"),
        "config_name": "Llama-3.1-8B-Instruct.json",
    },
    "Mistral-7B-Instruct-v0.2": {
        "path": os.path.join("models", "models--mistralai--Mistral-7B-Instruct-v0.2"),
        "config_name": "mistral-7b-Instruct-32k.json",
    },
}

MODEL_PATH_ALIASES = {
    "qwen2.5-7b": {
        "local": os.path.join("models", "qwen2.5-7b"),
        "remote": "Qwen/Qwen2.5-7B-Instruct",
        "config": "Qwen2.5-7B-Instruct.json",
    },
    "qwen/qwen2.5-7b-instruct": {
        "local": os.path.join("models", "qwen2.5-7b"),
        "remote": "Qwen/Qwen2.5-7B-Instruct",
        "config": "Qwen2.5-7B-Instruct.json",
    },
    "qwen2.5-72b": {
        "local": None,
        "remote": "Qwen/Qwen2.5-72B-Instruct",
        "config": "Qwen2.5-72B-Instruct.json",
    },
    "qwen/qwen2.5-72b-instruct": {
        "local": None,
        "remote": "Qwen/Qwen2.5-72B-Instruct",
        "config": "Qwen2.5-72B-Instruct.json",
    },
    "llama-3.1-8b": {
        "local": os.path.join("models", "Llama-3.1-8B-Instruct"),
        "remote": "meta-llama/Llama-3.1-8B-Instruct",
        "config": "Llama-3.1-8B-Instruct.json",
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "local": os.path.join("models", "Llama-3.1-8B-Instruct"),
        "remote": "meta-llama/Llama-3.1-8B-Instruct",
        "config": "Llama-3.1-8B-Instruct.json",
    },
    "llama-3-8b-1048k": {
        "local": os.path.join("models", "llama3-8b-1048k"),
        "remote": "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        "config": "Llama-3-8B-Instruct-Gradient-1048k.json",
    },
    "gradientai/llama-3-8b-instruct-gradient-1048k": {
        "local": os.path.join("models", "llama3-8b-1048k"),
        "remote": "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        "config": "Llama-3-8B-Instruct-Gradient-1048k.json",
    },
    "mistral-7b-instruct-v0.2": {
        "local": os.path.join("models", "models--mistralai--Mistral-7B-Instruct-v0.2"),
        "remote": "mistralai/Mistral-7B-Instruct-v0.2",
        "config": "mistral-7b-Instruct-32k.json",
    },
    "mistralai/mistral-7b-instruct-v0.2": {
        "local": os.path.join("models", "models--mistralai--Mistral-7B-Instruct-v0.2"),
        "remote": "mistralai/Mistral-7B-Instruct-v0.2",
        "config": "mistral-7b-Instruct-32k.json",
    },
    "models--mistralai--mistral-7b-instruct-v0.2": {
        "local": os.path.join("models", "models--mistralai--Mistral-7B-Instruct-v0.2"),
        "remote": "mistralai/Mistral-7B-Instruct-v0.2",
        "config": "mistral-7b-Instruct-32k.json",
    },
}

def get_model_choices() -> list[str]:
    choices = []
    for model_name, model_info in MODEL_REGISTRY.items():
        choices.append(model_name)
        model_path = model_info.get("path")
        if model_path:
            choices.append(model_path)
    return list(dict.fromkeys(choices))


def get_default_model_name() -> str:
    return next(iter(MODEL_REGISTRY))


def _resolve_project_relative_model_path(model_path: str) -> str:
    if not model_path or os.path.isabs(model_path):
        return model_path
    return os.path.join(PROJECT_ROOT, model_path)


def add_config_args(parser):
    parser.add_argument(
        "--attn_type",
        type=str,
        default="AdaptiveIMI",
        choices=["Full_Flash_Attn", "Full_Flash_Attn_Offload", "AdaptiveIMI"],
        help="Attention method",
    )
    parser.add_argument("--retrieval_budget", type=float, default=0.018, help="Retrieval budget")
    parser.add_argument("--cache_ratio", type=float, default=0.0, help="Cache ratio for AdaptiveIMI")
    parser.add_argument("--gpu_only", action='store_true', help="Whether to use GPU-only mode for AdaptiveIMI")
    parser.add_argument("--subspace_parts", type=int, default=2, help="IMI subspace parts (0, 2 or 4)")
    return parser


def get_numa_node_core_count(node_id=0):
    path = Path(f"/sys/devices/system/node/node{node_id}/cpulist")
    if not path.exists():
        count = os.cpu_count()
        print(f"NUMA node{node_id} not found, set core to #total_cpu_core: {count}")
        return max(count - 2, 1)    # reserve 2 cores for system
    # get NUMA node core count
    cpulist = path.read_text().strip()
    count = 0
    for part in cpulist.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            count += end - start + 1
        else:
            count += 1
    return max(count - 2, 1)  # reserve 2 cores for system


def _resolve_snapshot_path(model_path: str) -> str:
    if not model_path or not os.path.isdir(model_path):
        return model_path
    snapshots_dir = os.path.join(model_path, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return model_path
    snapshot_dirs = [
        os.path.join(snapshots_dir, name)
        for name in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, name))
    ]
    if not snapshot_dirs:
        return model_path
    snapshot_dirs.sort(key=os.path.getmtime, reverse=True)
    return snapshot_dirs[0]


def _iter_model_lookup_keys(model_name: str):
    normalized = (model_name or "").rstrip("/")
    if not normalized:
        return []
    base_name = os.path.basename(normalized)
    keys = [normalized.lower()]
    if base_name:
        keys.append(base_name.lower())
    return list(dict.fromkeys(keys))



def _existing_local_model_path(candidate: str) -> str | None:
    if not candidate:
        return None
    resolved = _resolve_snapshot_path(_resolve_project_relative_model_path(candidate))
    return resolved if resolved and os.path.exists(resolved) else None



def resolve_model_path(model_name: str) -> str:
    direct_path = _existing_local_model_path(model_name)
    if direct_path is not None:
        return direct_path

    model_info = MODEL_REGISTRY.get(model_name)
    if model_info and model_info.get("path"):
        registry_path = _existing_local_model_path(model_info["path"])
        if registry_path is not None:
            return registry_path

    for lookup_key in _iter_model_lookup_keys(model_name):
        alias_info = MODEL_PATH_ALIASES.get(lookup_key)
        if alias_info is None:
            continue
        local_path = _existing_local_model_path(alias_info.get("local"))
        if local_path is not None:
            return local_path

    normalized = (model_name or "").rstrip("/")
    base_name = os.path.basename(normalized)
    local_candidates = []
    if base_name:
        local_candidates.append(os.path.join("models", base_name))
    if "/" in normalized:
        org, repo = normalized.split("/", 1)
        local_candidates.append(os.path.join("models", repo))
        local_candidates.append(os.path.join("models", f"models--{org}--{repo}"))

    for candidate in local_candidates:
        local_path = _existing_local_model_path(candidate)
        if local_path is not None:
            return local_path

    if model_info and model_info.get("path"):
        return _resolve_snapshot_path(_resolve_project_relative_model_path(model_info["path"]))

    for lookup_key in _iter_model_lookup_keys(model_name):
        alias_info = MODEL_PATH_ALIASES.get(lookup_key)
        if alias_info is not None:
            remote_path = alias_info.get("remote")
            if remote_path:
                return remote_path

    return _resolve_snapshot_path(_resolve_project_relative_model_path(model_name))


def resolve_config_name(model_name: str) -> str:
    model_info = MODEL_REGISTRY.get(model_name)
    if model_info and model_info.get("config_name"):
        return model_info["config_name"]

    for lookup_key in _iter_model_lookup_keys(model_name):
        alias_info = MODEL_PATH_ALIASES.get(lookup_key)
        if alias_info is not None and alias_info.get("config"):
            return alias_info["config"]

    base_name = os.path.basename(model_name.rstrip("/"))
    if base_name.endswith(".json"):
        return base_name
    return f"{base_name}.json"


def _clone_attn_config_section(section):
    return copy.deepcopy(section) if section is not None else {}


def generate_config(
    model_name, context_len, attn_type, 
    retrieval_budget=0.018, cache_ratio=0.0,
    gpu_only=False, subspace_parts=2
):
    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    model_config_name = resolve_config_name(model_name)
    CONFIG_FILE = os.path.join(CONFIG_DIR, model_config_name)
    with open(CONFIG_FILE, "r") as f:
        _config = json.load(f)
    
    if attn_type == "AdaptiveIMI":
        if subspace_parts not in (0, 2, 4):
            raise ValueError("subspace_parts must be 0, 2 or 4")

        base_config = _clone_attn_config_section(_config.get("AdaptiveIMI"))
        if retrieval_budget is None:
            retrieval_budget = base_config.get("retrieval_budget", 0.018)
        if cache_ratio is None:
            cache_ratio = base_config.get("cache_ratio", 0.0)

        base_config["core"] = get_numa_node_core_count(0)
        base_config["pages_per_cluster"] = round(16 / 8)
        base_config["retrieval_budget"] = retrieval_budget
        base_config["cache_ratio"] = cache_ratio
        base_config["subspace_parts"] = subspace_parts
        base_config["gpu_only"] = gpu_only
        if context_len <= 4096:
            base_config["buffer_cluster_num"] = 150
        _config["AdaptiveIMI"] = base_config

    if attn_type not in ("Full_Flash_Attn", "Full_Flash_Attn_Offload"):
        print(_config.get(attn_type, {}))
    
    return _config
