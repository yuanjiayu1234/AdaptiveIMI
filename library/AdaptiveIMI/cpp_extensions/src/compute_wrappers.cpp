/**
 * compute_impl.cpp - Python Wrappers for IMI Compute Kernels
 *
 * 提供Python可调用的wrapper函数，封装CUDA kernel调用
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>

namespace imi_decode {

void fused_query_group_similarities_bf16(
    const void* query_grouped,
    const float* centroids,
    float* similarities,
    int num_query_groups,
    int n_clusters,
    int kv_heads,
    int dim,
    cudaStream_t stream = 0
);

void fused_query_group_similarities_fp16(
    const void* query_grouped,
    const float* centroids,
    float* similarities,
    int num_query_groups,
    int n_clusters,
    int kv_heads,
    int dim,
    cudaStream_t stream = 0
);

// ============================================================================
// Python Wrapper: fused_query_group_similarities
// ============================================================================
torch::Tensor fused_query_group_similarities(
    const torch::Tensor& query_grouped,  // [kv_heads, num_query_groups, dim] BF16/FP16
    const torch::Tensor& centroids,      // [kv_heads, n_clusters, dim] FP32
    c10::optional<torch::Tensor> out_tensor
) {
    TORCH_CHECK(query_grouped.is_cuda(), "query_grouped must be on GPU");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be on GPU");
    TORCH_CHECK(query_grouped.dim() == 3, "query_grouped must be 3D");
    TORCH_CHECK(centroids.dim() == 3, "centroids must be 3D");
    TORCH_CHECK(centroids.scalar_type() == torch::kFloat32, "centroids must be FP32");

    int kv_heads = query_grouped.size(0);
    int groups = query_grouped.size(1);
    int dim = query_grouped.size(2);
    int n_clusters = centroids.size(1);

    TORCH_CHECK(centroids.size(0) == kv_heads, "centroids kv_heads mismatch");
    TORCH_CHECK(centroids.size(2) == dim, "centroids dim mismatch");

    torch::Tensor similarities;
    if (out_tensor.has_value() && out_tensor->defined()) {
        auto out = out_tensor.value();
        TORCH_CHECK(out.is_cuda(), "fused_query_group_similarities: out tensor must be on GPU");
        TORCH_CHECK(out.scalar_type() == torch::kFloat32, "fused_query_group_similarities: out tensor must be FP32");
        TORCH_CHECK(out.dim() == 2, "fused_query_group_similarities: out tensor must be 2D");
        TORCH_CHECK(out.size(0) >= kv_heads && out.size(1) >= n_clusters,
                    "fused_query_group_similarities: out tensor shape insufficient");
        similarities = out.narrow(0, 0, kv_heads).narrow(1, 0, n_clusters);
    } else {
        similarities = torch::empty(
            {kv_heads, n_clusters},
            torch::TensorOptions().dtype(torch::kFloat32).device(query_grouped.device())
        );
    }

    c10::cuda::CUDAGuard device_guard(query_grouped.device());
    auto stream = c10::cuda::getCurrentCUDAStream(query_grouped.get_device());

    if (query_grouped.scalar_type() == torch::kBFloat16) {
        fused_query_group_similarities_bf16(
            query_grouped.data_ptr(),
            centroids.data_ptr<float>(),
            similarities.data_ptr<float>(),
            groups,
            n_clusters,
            kv_heads,
            dim,
            stream.stream()
        );
    } else if (query_grouped.scalar_type() == torch::kFloat16) {
        fused_query_group_similarities_fp16(
            query_grouped.data_ptr(),
            centroids.data_ptr<float>(),
            similarities.data_ptr<float>(),
            groups,
            n_clusters,
            kv_heads,
            dim,
            stream.stream()
        );
    } else {
        TORCH_CHECK(false, "fused_query_group_similarities: query dtype must be BF16 or FP16");
    }

    return similarities;
}

} // namespace imi_decode
