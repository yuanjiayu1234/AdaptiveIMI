from importlib import import_module

__all__ = [
    "flash_attn_cache",
    "flash_attn_cache_offload",
    "adpimi_cache",
]


def __getattr__(name):
    mapping = {
        "flash_attn_cache": ("full_cache", "flash_attn_cache"),
        "flash_attn_cache_offload": ("offload_cache", "flash_attn_cache_offload"),
        "adpimi_cache": ("adaptive_imi", "adpimi_cache"),
    }
    if name in mapping:
        module_name, attr_name = mapping[name]
        module = import_module(f"{__name__}.{module_name}")
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
