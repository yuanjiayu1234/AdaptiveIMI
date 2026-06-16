"""AdaptiveIMI C++/CUDA extension exports."""

from importlib import import_module


def _import_required(module_name: str, symbol_names: list[str]):
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        raise ImportError(
            f"Failed to import AdaptiveIMI extension module '{module_name}'. "
            "Please build the C++/CUDA extensions under "
            "library/AdaptiveIMI/cpp_extensions first."
        ) from exc
    return [getattr(module, symbol_name) for symbol_name in symbol_names]


[
    gather_copy_and_concat_chunk_parallel,
    build_cache_aware_retrieval_plan,
    promote_cache_pages_from_list,
] = _import_required(
    "Copy",
    [
        "gather_copy_and_concat_chunk_parallel",
        "build_cache_aware_retrieval_plan",
        "promote_cache_pages_from_list",
    ],
)

__all__ = [
    "gather_copy_and_concat_chunk_parallel",
    "build_cache_aware_retrieval_plan",
    "promote_cache_pages_from_list",
]
