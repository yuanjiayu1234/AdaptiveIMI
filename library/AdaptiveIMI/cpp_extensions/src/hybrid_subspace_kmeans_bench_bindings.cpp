#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

namespace imi_hybrid_bench {

std::vector<torch::Tensor> run_subspace_kmeans_only_cuda(
    const torch::Tensor& keys,
    const torch::Tensor& row_token_counts,
    const torch::Tensor& subspace_dim_ids,
    const torch::Tensor& subspace_dim_masks,
    const torch::Tensor& k_values,
    int64_t max_iters,
    int64_t max_k,
    bool use_kmeanspp
);

} // namespace imi_hybrid_bench

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Benchmark-only CUDA subspace k-means kernels";

    m.def(
        "run_subspace_kmeans_only",
        &imi_hybrid_bench::run_subspace_kmeans_only_cuda,
        py::arg("keys"),
        py::arg("row_token_counts"),
        py::arg("subspace_dim_ids"),
        py::arg("subspace_dim_masks"),
        py::arg("k_values"),
        py::arg("max_iters"),
        py::arg("max_k"),
        py::arg("use_kmeanspp") = true,
        R"pbdoc(
            Run only learned-subspace k-means.

            Args:
                keys: [batch, kv_heads, max_tokens, head_dim] CUDA tensor.
                row_token_counts: [batch * kv_heads] int32 CUDA tensor.
                subspace_dim_ids: [batch * kv_heads, M, max_sub_dim] int64 CUDA tensor.
                subspace_dim_masks: [batch * kv_heads, M, max_sub_dim] bool CUDA tensor.
                k_values: [M] int32 CUDA tensor.
                max_iters: Lloyd iterations.
                max_k: max(k_values), supplied by Python to avoid GPU scalar sync.
                use_kmeanspp: use kmeans++ init when true; deterministic sampled init when false.

            Returns:
                centroids: float32[rows, M, max_k, max_sub_dim]
                centroid_masks: bool[rows, M, max_k]
                labels: int32[rows, M, max_tokens]
                counts: float32[rows, M, max_k]
        )pbdoc"
    );
}
