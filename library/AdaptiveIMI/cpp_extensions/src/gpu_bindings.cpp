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

    torch::Tensor fused_subspace_rerank_select(const torch::Tensor& query_grouped,
                                               const torch::Tensor& subspace_centroids,
                                               const torch::Tensor& subspace_centroid_masks,
                                               const torch::Tensor& subspace_dim_ids,
                                               const torch::Tensor& subspace_dim_masks,
                                               const torch::Tensor& compact_codes,
                                               const torch::Tensor& centroids,
                                               const torch::Tensor& cluster_sizes,
                                               int64_t nprobe,
                                               int64_t candidate_limit,
                                               int64_t rerank_limit,
                                               int64_t max_retrieval_tokens,
                                               int64_t padding_cluster_id);
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

    m.def("fused_subspace_rerank_select",
        [](const torch::Tensor& query_grouped,
           const torch::Tensor& subspace_centroids,
           const torch::Tensor& subspace_centroid_masks,
           const torch::Tensor& subspace_dim_ids,
           const torch::Tensor& subspace_dim_masks,
           const torch::Tensor& compact_codes,
           const torch::Tensor& centroids,
           const torch::Tensor& cluster_sizes,
           int64_t nprobe,
           int64_t candidate_limit,
           int64_t rerank_limit,
           int64_t max_retrieval_tokens,
           int64_t padding_cluster_id) {
            return imi_decode::fused_subspace_rerank_select(
                query_grouped,
                subspace_centroids,
                subspace_centroid_masks,
                subspace_dim_ids,
                subspace_dim_masks,
                compact_codes,
                centroids,
                cluster_sizes,
                nprobe,
                candidate_limit,
                rerank_limit,
                max_retrieval_tokens,
                padding_cluster_id
            );
        },
        py::arg("query_grouped"),
        py::arg("subspace_centroids"),
        py::arg("subspace_centroid_masks"),
        py::arg("subspace_dim_ids"),
        py::arg("subspace_dim_masks"),
        py::arg("compact_codes"),
        py::arg("centroids"),
        py::arg("cluster_sizes"),
        py::arg("nprobe"),
        py::arg("candidate_limit"),
        py::arg("rerank_limit"),
        py::arg("max_retrieval_tokens"),
        py::arg("padding_cluster_id"),
        R"pbdoc(
            Hybrid 8-way decode fast path:
            subspace coarse scoring -> candidate topk -> true full-centroid rerank -> token-budget cluster selection.

            Returns:
                [rows, nprobe] int64 selected cluster ids
        )pbdoc");
}
