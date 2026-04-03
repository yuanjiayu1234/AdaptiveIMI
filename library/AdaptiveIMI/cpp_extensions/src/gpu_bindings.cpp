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
}
