#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <vector>

namespace py = pybind11;

namespace imi_hybrid {

std::vector<torch::Tensor> run_hybrid_8way_kmeans_cuda(
    const torch::Tensor& keys,
    const torch::Tensor& row_token_counts,
    const torch::Tensor& subspace_dim_ids,
    const torch::Tensor& subspace_dim_masks,
    const torch::Tensor& k_values,
    int64_t max_iters,
    int64_t max_k,
    int64_t dense_code_space
);

void reorder_raw_kv_by_token_ids_cpu(
    const torch::Tensor& keys_cpu,
    const torch::Tensor& values_cpu,
    const torch::Tensor& token_ids_by_cluster,
    const torch::Tensor& row_token_counts,
    const torch::Tensor& out_keys_cpu,
    const torch::Tensor& out_values_cpu
) {
    TORCH_CHECK(!keys_cpu.is_cuda(), "keys_cpu must be a CPU tensor");
    TORCH_CHECK(!values_cpu.is_cuda(), "values_cpu must be a CPU tensor");
    TORCH_CHECK(!token_ids_by_cluster.is_cuda(), "token_ids_by_cluster must be a CPU tensor");
    TORCH_CHECK(!row_token_counts.is_cuda(), "row_token_counts must be a CPU tensor");
    TORCH_CHECK(!out_keys_cpu.is_cuda(), "out_keys_cpu must be a CPU tensor");
    TORCH_CHECK(!out_values_cpu.is_cuda(), "out_values_cpu must be a CPU tensor");
    TORCH_CHECK(keys_cpu.is_contiguous(), "keys_cpu must be contiguous");
    TORCH_CHECK(values_cpu.is_contiguous(), "values_cpu must be contiguous");
    TORCH_CHECK(token_ids_by_cluster.is_contiguous(), "token_ids_by_cluster must be contiguous");
    TORCH_CHECK(row_token_counts.is_contiguous(), "row_token_counts must be contiguous");
    TORCH_CHECK(out_keys_cpu.is_contiguous(), "out_keys_cpu must be contiguous");
    TORCH_CHECK(out_values_cpu.is_contiguous(), "out_values_cpu must be contiguous");
    TORCH_CHECK(keys_cpu.dim() == 4, "keys_cpu must be [batch, kv_heads, max_tokens, head_dim]");
    TORCH_CHECK(values_cpu.dim() == 4, "values_cpu must be [batch, kv_heads, max_tokens, head_dim]");
    TORCH_CHECK(out_keys_cpu.dim() == 4, "out_keys_cpu must be [batch, kv_heads, capacity, head_dim]");
    TORCH_CHECK(out_values_cpu.dim() == 4, "out_values_cpu must be [batch, kv_heads, capacity, head_dim]");
    TORCH_CHECK(keys_cpu.size(0) == values_cpu.size(0) &&
                keys_cpu.size(1) == values_cpu.size(1) &&
                keys_cpu.size(2) == values_cpu.size(2) &&
                keys_cpu.size(3) == values_cpu.size(3),
                "keys_cpu and values_cpu shape mismatch");
    TORCH_CHECK(out_keys_cpu.size(0) == keys_cpu.size(0) &&
                out_keys_cpu.size(1) == keys_cpu.size(1) &&
                out_keys_cpu.size(3) == keys_cpu.size(3),
                "out_keys_cpu batch/head/dim shape mismatch");
    TORCH_CHECK(out_values_cpu.size(0) == keys_cpu.size(0) &&
                out_values_cpu.size(1) == keys_cpu.size(1) &&
                out_values_cpu.size(3) == keys_cpu.size(3),
                "out_values_cpu batch/head/dim shape mismatch");
    TORCH_CHECK(out_keys_cpu.size(2) >= keys_cpu.size(2), "out_keys_cpu capacity is smaller than source tokens");
    TORCH_CHECK(out_values_cpu.size(2) >= keys_cpu.size(2), "out_values_cpu capacity is smaller than source tokens");
    TORCH_CHECK(values_cpu.scalar_type() == keys_cpu.scalar_type(), "keys_cpu and values_cpu dtype mismatch");
    TORCH_CHECK(out_keys_cpu.scalar_type() == keys_cpu.scalar_type(), "out_keys_cpu dtype mismatch");
    TORCH_CHECK(out_values_cpu.scalar_type() == values_cpu.scalar_type(), "out_values_cpu dtype mismatch");
    TORCH_CHECK(token_ids_by_cluster.scalar_type() == torch::kInt32, "token_ids_by_cluster must be int32");
    TORCH_CHECK(row_token_counts.scalar_type() == torch::kInt32, "row_token_counts must be int32");

    const int64_t batch = keys_cpu.size(0);
    const int64_t kv_heads = keys_cpu.size(1);
    const int64_t max_tokens = keys_cpu.size(2);
    const int64_t out_capacity = out_keys_cpu.size(2);
    const int64_t head_dim = keys_cpu.size(3);
    const int64_t rows = batch * kv_heads;
    TORCH_CHECK(token_ids_by_cluster.size(0) == rows, "token_ids_by_cluster row mismatch");
    TORCH_CHECK(token_ids_by_cluster.size(1) >= max_tokens, "token_ids_by_cluster token dimension is too small");
    TORCH_CHECK(row_token_counts.numel() == rows, "row_token_counts row mismatch");

    const auto element_size = keys_cpu.element_size();
    const int64_t src_row_bytes = max_tokens * head_dim * static_cast<int64_t>(element_size);
    const int64_t out_row_bytes = out_capacity * head_dim * static_cast<int64_t>(element_size);
    const int64_t token_bytes = head_dim * static_cast<int64_t>(element_size);
    const char* keys_base = static_cast<const char*>(keys_cpu.data_ptr());
    const char* values_base = static_cast<const char*>(values_cpu.data_ptr());
    char* out_keys_base = static_cast<char*>(out_keys_cpu.data_ptr());
    char* out_values_base = static_cast<char*>(out_values_cpu.data_ptr());
    const int32_t* ids_base = token_ids_by_cluster.data_ptr<int32_t>();
    const int32_t* counts = row_token_counts.data_ptr<int32_t>();
    for (int64_t row = 0; row < rows; ++row) {
        TORCH_CHECK(counts[row] >= 0 && counts[row] <= max_tokens, "row_token_counts entry out of range");
        TORCH_CHECK(
            counts[row] <= token_ids_by_cluster.size(1),
            "token_ids_by_cluster token dimension is smaller than row token count"
        );
    }

    at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int32_t token_count = counts[row];
            const char* row_keys = keys_base + row * src_row_bytes;
            const char* row_values = values_base + row * src_row_bytes;
            char* row_out_keys = out_keys_base + row * out_row_bytes;
            char* row_out_values = out_values_base + row * out_row_bytes;
            const int32_t* row_ids = ids_base + row * token_ids_by_cluster.size(1);
            for (int32_t dst = 0; dst < token_count; ++dst) {
                const int32_t src = row_ids[dst];
                std::memcpy(row_out_keys + static_cast<int64_t>(dst) * token_bytes,
                            row_keys + static_cast<int64_t>(src) * token_bytes,
                            static_cast<size_t>(token_bytes));
                std::memcpy(row_out_values + static_cast<int64_t>(dst) * token_bytes,
                            row_values + static_cast<int64_t>(src) * token_bytes,
                            static_cast<size_t>(token_bytes));
            }
        }
    });
}

} // namespace imi_hybrid

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "AdaptiveIMI hybrid 8-way CUDA index builder kernels";

    m.def(
        "run_hybrid_8way_kmeans",
        &imi_hybrid::run_hybrid_8way_kmeans_cuda,
        py::arg("keys"),
        py::arg("row_token_counts"),
        py::arg("subspace_dim_ids"),
        py::arg("subspace_dim_masks"),
        py::arg("k_values"),
        py::arg("max_iters"),
        py::arg("max_k"),
        py::arg("dense_code_space"),
        R"pbdoc(
            Learned 8-way batched k-means for hybrid AdaptiveIMI.

            Args:
                keys: [batch, kv_heads, max_tokens, head_dim] CUDA tensor.
                row_token_counts: [batch * kv_heads] int32 CUDA tensor.
                subspace_dim_ids: [batch * kv_heads, 8, max_sub_dim] int64 CUDA tensor.
                subspace_dim_masks: [batch * kv_heads, 8, max_sub_dim] bool CUDA tensor.
                k_values: [8] int32 CUDA tensor.
                max_iters: Lloyd iterations.
                max_k: max(k_values), supplied by Python to avoid GPU scalar sync.
                dense_code_space: product(k_values), supplied by Python to avoid GPU scalar sync.

            Returns:
                labels_m: int32[batch_groups, 8, max_tokens]
                subspace_centroids: float32[batch_groups, 8, max_k, max_sub_dim]
                subspace_centroid_masks: bool[batch_groups, 8, max_k]
                token_labels: int32[batch_groups, max_tokens]
                token_ids_by_cluster: int32[batch_groups, max_tokens]
                compact_codes: int32[batch_groups, max_compact_clusters, 8]
                cluster_sizes: int32[batch_groups, max_compact_clusters]
                cluster_offsets: int32[batch_groups, max_compact_clusters + 1]
                cluster_counts: int32[batch_groups]
                full_centroids: float32[batch_groups, max_compact_clusters, head_dim]
        )pbdoc"
    );

    m.def(
        "reorder_raw_kv_by_token_ids",
        &imi_hybrid::reorder_raw_kv_by_token_ids_cpu,
        py::arg("keys_cpu"),
        py::arg("values_cpu"),
        py::arg("token_ids_by_cluster"),
        py::arg("row_token_counts"),
        py::arg("out_keys_cpu"),
        py::arg("out_values_cpu"),
        R"pbdoc(
            Reorder raw CPU KV by CUDA-produced token ids into compact cluster order.
        )pbdoc"
    );
}
