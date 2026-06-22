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
    gather_copy_static_chunks_profile,
    gather_copy_miss_chunks_profile,
    gather_copy_selected_clusters_profile,
    gather_copy_hit_chunks_profile,
    build_cache_aware_retrieval_plan,
    promote_hot_cold_cache_pages_from_execution,
] = _import_required(
    "Copy",
    [
        "gather_copy_static_chunks_profile",
        "gather_copy_miss_chunks_profile",
        "gather_copy_selected_clusters_profile",
        "gather_copy_hit_chunks_profile",
        "build_cache_aware_retrieval_plan",
        "promote_hot_cold_cache_pages_from_execution",
    ],
)

__all__ = [
    "gather_copy_static_chunks_profile",
    "gather_copy_miss_chunks_profile",
    "gather_copy_selected_clusters_profile",
    "gather_copy_hit_chunks_profile",
    "build_cache_aware_retrieval_plan",
    "promote_hot_cold_cache_pages_from_execution",
]
