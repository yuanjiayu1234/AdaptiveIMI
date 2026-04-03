from importlib import import_module


__all__ = [
    "flash_attn_cache",
    "retroinfer_cache",
    "retroinfer_cache_imi",
    "retroinfer_cache_gpu",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
