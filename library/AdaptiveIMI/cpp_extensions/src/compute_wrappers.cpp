/**
 * Python wrappers for AdaptiveIMI decode compute kernels.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <limits>

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

void apply_budget_after_topk(
    const int64_t* topk_indices,
    const int32_t* cluster_sizes,
    int64_t* selected,
    int64_t selected_stride0,
    int rows,
    int nprobe,
    int n_centroids,
    int max_retrieval_tokens,
    int64_t padding_cluster_id,
    cudaStream_t stream = 0
);

torch::Tensor fused_query_group_similarities(
    const torch::Tensor& query_grouped,
    const torch::Tensor& centroids,
    c10::optional<torch::Tensor> out_tensor
) {
    TORCH_CHECK(query_grouped.is_cuda(), "query_grouped must be on GPU");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be on GPU");
    TORCH_CHECK(query_grouped.dim() == 3, "query_grouped must be 3D");
    TORCH_CHECK(centroids.dim() == 3, "centroids must be 3D");
    TORCH_CHECK(centroids.scalar_type() == torch::kFloat32, "centroids must be FP32");
    TORCH_CHECK(query_grouped.is_contiguous(), "query_grouped must be contiguous");
    TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");

    constexpr int64_t int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
    TORCH_CHECK(query_grouped.size(0) <= int_max, "query_grouped kv_heads exceeds int range");
    TORCH_CHECK(query_grouped.size(1) <= int_max, "query_grouped groups exceeds int range");
    TORCH_CHECK(query_grouped.size(2) <= int_max, "query_grouped dim exceeds int range");
    TORCH_CHECK(centroids.size(1) <= int_max, "centroids n_clusters exceeds int range");

    const int kv_heads = static_cast<int>(query_grouped.size(0));
    const int groups = static_cast<int>(query_grouped.size(1));
    const int dim = static_cast<int>(query_grouped.size(2));
    const int n_clusters = static_cast<int>(centroids.size(1));

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

torch::Tensor apply_budget_after_topk(
    const torch::Tensor& topk_indices,
    const torch::Tensor& cluster_sizes,
    int64_t max_retrieval_tokens,
    int64_t padding_cluster_id,
    c10::optional<torch::Tensor> out_tensor
) {
    TORCH_CHECK(topk_indices.is_cuda(), "topk_indices must be on GPU");
    TORCH_CHECK(cluster_sizes.is_cuda(), "cluster_sizes must be on GPU");
    TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be 2D");
    TORCH_CHECK(cluster_sizes.dim() == 2, "cluster_sizes must be 2D");
    TORCH_CHECK(topk_indices.scalar_type() == torch::kInt64, "topk_indices must be int64");
    TORCH_CHECK(cluster_sizes.scalar_type() == torch::kInt32, "cluster_sizes must be int32");
    TORCH_CHECK(topk_indices.is_contiguous(), "topk_indices must be contiguous");
    TORCH_CHECK(cluster_sizes.is_contiguous(), "cluster_sizes must be contiguous");
    TORCH_CHECK(cluster_sizes.size(0) == topk_indices.size(0), "row count mismatch");
    TORCH_CHECK(max_retrieval_tokens >= 0, "max_retrieval_tokens must be non-negative");

    const int rows = static_cast<int>(topk_indices.size(0));
    const int nprobe = static_cast<int>(topk_indices.size(1));
    const int n_centroids = static_cast<int>(cluster_sizes.size(1));

    torch::Tensor selected;
    if (out_tensor.has_value() && out_tensor->defined()) {
        auto out = out_tensor.value();
        TORCH_CHECK(out.is_cuda(), "apply_budget_after_topk: out tensor must be on GPU");
        TORCH_CHECK(out.scalar_type() == torch::kInt64, "apply_budget_after_topk: out tensor must be int64");
        TORCH_CHECK(out.dim() == 2, "apply_budget_after_topk: out tensor must be 2D");
        TORCH_CHECK(out.stride(1) == 1, "apply_budget_after_topk: out tensor must have contiguous columns");
        TORCH_CHECK(out.size(0) >= rows && out.size(1) >= nprobe,
                    "apply_budget_after_topk: out tensor shape insufficient");
        selected = out.narrow(0, 0, rows).narrow(1, 0, nprobe);
    } else {
        selected = torch::empty_like(topk_indices);
    }

    c10::cuda::CUDAGuard device_guard(topk_indices.device());
    auto stream = c10::cuda::getCurrentCUDAStream(topk_indices.get_device());
    apply_budget_after_topk(
        topk_indices.data_ptr<int64_t>(),
        cluster_sizes.data_ptr<int32_t>(),
        selected.data_ptr<int64_t>(),
        selected.stride(0),
        rows,
        nprobe,
        n_centroids,
        static_cast<int>(max_retrieval_tokens),
        padding_cluster_id,
        stream.stream()
    );

    return selected;
}

} // namespace imi_decode
