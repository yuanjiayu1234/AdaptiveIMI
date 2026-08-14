#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Forward declarations for compute kernels
namespace imi_decode {
    torch::Tensor fused_query_group_similarities(const torch::Tensor& query_grouped,
                                                 const torch::Tensor& centroids,
                                                 c10::optional<torch::Tensor> out);
    torch::Tensor apply_budget_after_topk(const torch::Tensor& topk_indices,
                                          const torch::Tensor& cluster_sizes,
                                          int64_t max_retrieval_tokens,
                                          int64_t padding_cluster_id,
                                          c10::optional<torch::Tensor> out);
    void assign_generated_to_centroids(const torch::Tensor& keys, const torch::Tensor& centroids,
                                       const torch::Tensor& centroid_mask, const torch::Tensor& assignments);
    void apply_generated_centroid_updates(const torch::Tensor& cluster_ids, const torch::Tensor& cluster_sums,
                                          const torch::Tensor& cluster_counts, const torch::Tensor& centroids,
                                          const torch::Tensor& base_cluster_counts, const torch::Tensor& generated_cluster_counts);
    void publish_generated_page_metadata(const torch::Tensor& cluster_ids, const torch::Tensor& cluster_tails_in,
                                         const torch::Tensor& page_ids, const torch::Tensor& page_prev_in,
                                         const torch::Tensor& page_sizes_in, const torch::Tensor& cluster_tails_out,
                                         const torch::Tensor& page_prev_out, const torch::Tensor& page_sizes_out);
    void pack_generated_pages(const torch::Tensor& assignments, const torch::Tensor& staging_keys,
                              const torch::Tensor& staging_values, const torch::Tensor& page_keys,
                              const torch::Tensor& page_values, const torch::Tensor& cluster_heads,
                              const torch::Tensor& cluster_tails, const torch::Tensor& page_prev,
                              const torch::Tensor& page_sizes, const torch::Tensor& next_pages,
                              const torch::Tensor& out_cluster_ids, const torch::Tensor& out_cluster_heads,
                              const torch::Tensor& out_cluster_tails, const torch::Tensor& out_page_ids,
                              const torch::Tensor& out_page_prev, const torch::Tensor& out_page_sizes, const torch::Tensor& out_cluster_sums,
                              const torch::Tensor& out_cluster_counts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GPU Cluster Manager: IMI decode kernels";

    // ================================================================
    // 绑定 fused_query_group_similarities
    // ================================================================
    m.def("fused_query_group_similarities",
        [](const torch::Tensor& query_grouped,
           const torch::Tensor& centroids,
           c10::optional<torch::Tensor> out) {
            return imi_decode::fused_query_group_similarities(
                query_grouped,
                centroids,
                std::move(out)
            );
        },
        py::arg("query_grouped"),
        py::arg("centroids"),
        py::arg("out") = py::none(),
        R"pbdoc(
            单步 kernel：在 shared memory 内先聚合 query groups，再与 centroids 做点积。

            Args:
                query_grouped: [kv_heads, num_query_groups, dim] BF16/FP16
                centroids: [kv_heads, n_clusters, dim] FP32

            Returns:
                [kv_heads, n_clusters] FP32 相似度矩阵
        )pbdoc");

    m.def("apply_budget_after_topk",
        [](const torch::Tensor& topk_indices,
           const torch::Tensor& cluster_sizes,
           int64_t max_retrieval_tokens,
           int64_t padding_cluster_id,
           c10::optional<torch::Tensor> out) {
            return imi_decode::apply_budget_after_topk(
                topk_indices,
                cluster_sizes,
                max_retrieval_tokens,
                padding_cluster_id,
                std::move(out)
            );
        },
        py::arg("topk_indices"),
        py::arg("cluster_sizes"),
        py::arg("max_retrieval_tokens"),
        py::arg("padding_cluster_id"),
        py::arg("out") = py::none(),
        R"pbdoc(
            Keep top-k clusters under the AdaptiveIMI retrieval token budget.

            This preserves the existing Python semantics:
            keep clusters in sorted top-k order while the token count before
            the current cluster is below max_retrieval_tokens; replace dropped
            or empty clusters with padding_cluster_id.
        )pbdoc");

    m.def("assign_generated_to_centroids", &imi_decode::assign_generated_to_centroids,
        py::arg("keys"), py::arg("centroids"), py::arg("centroid_mask"), py::arg("assignments"),
        "GPU top-1 assignment of generated keys to the active full centroids");
    m.def("apply_generated_centroid_updates", &imi_decode::apply_generated_centroid_updates,
        py::arg("cluster_ids"), py::arg("cluster_sums"), py::arg("cluster_counts"),
        py::arg("centroids"), py::arg("base_cluster_counts"), py::arg("generated_cluster_counts"),
        "Commit deduplicated generated-key sums/counts on the decode stream");
    m.def("pack_generated_pages", &imi_decode::pack_generated_pages,
        py::arg("assignments"), py::arg("staging_keys"), py::arg("staging_values"),
        py::arg("page_keys"), py::arg("page_values"), py::arg("cluster_heads"),
        py::arg("cluster_tails"), py::arg("page_prev"), py::arg("page_sizes"), py::arg("next_pages"),
        py::arg("out_cluster_ids"), py::arg("out_cluster_heads"), py::arg("out_cluster_tails"),
        py::arg("out_page_ids"), py::arg("out_page_prev"), py::arg("out_page_sizes"), py::arg("out_cluster_sums"), py::arg("out_cluster_counts"),
        "Pack a completed D2H generated-KV chunk into immutable pinned CPU delta pages");
    m.def("publish_generated_page_metadata", &imi_decode::publish_generated_page_metadata,
        py::arg("cluster_ids"), py::arg("cluster_tails_in"), py::arg("page_ids"),
        py::arg("page_prev_in"), py::arg("page_sizes_in"), py::arg("cluster_tails_out"),
        py::arg("page_prev_out"), py::arg("page_sizes_out"),
        "Publish complete generated delta-page metadata on the decode stream");
}
