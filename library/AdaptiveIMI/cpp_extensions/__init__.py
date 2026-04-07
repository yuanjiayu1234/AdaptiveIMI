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
    AdpIMI_Index,
    AdpIMI_ThreadPool,
] = _import_required("AdpIMI_Index", ["AdpIMI_Index", "AdpIMI_ThreadPool"])

[
    gather_copy_vectors,
    gather_copy_and_concat,
    gather_copy_and_concat_retrieval,
    gather_copy_and_scatter,
    reorganize_vectors,
    gather_copy_cluster_and_concat_fuse,
] = _import_required(
    "Copy",
    [
        "gather_copy_vectors",
        "gather_copy_and_concat",
        "gather_copy_and_concat_retrieval",
        "gather_copy_and_scatter",
        "reorganize_vectors",
        "gather_copy_cluster_and_concat_fuse",
    ],
)

[batch_gemm_softmax] = _import_required("gemm_softmax", ["batch_gemm_softmax"])


__all__ = [
    "AdpIMI_Index",
    "AdpIMI_ThreadPool",
    "gather_copy_vectors",
    "gather_copy_and_concat",
    "gather_copy_and_concat_retrieval",
    "gather_copy_and_scatter",
    "reorganize_vectors",
    "gather_copy_cluster_and_concat_fuse",
    "batch_gemm_softmax",
]
