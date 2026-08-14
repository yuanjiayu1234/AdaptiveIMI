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

void assign_generated_to_centroids_bf16(const void*, const float*, const bool*, int32_t*, int, int, int, int, cudaStream_t);
void assign_generated_to_centroids_fp16(const void*, const float*, const bool*, int32_t*, int, int, int, int, cudaStream_t);
void apply_generated_centroid_updates(const int32_t*, const float*, const int32_t*, float*, const int32_t*, int32_t*, int, int, int, int, cudaStream_t);
void publish_generated_page_metadata(const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int32_t*, int32_t*, int32_t*, int, int, int, int, cudaStream_t);

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

void assign_generated_to_centroids(
    const torch::Tensor& keys,
    const torch::Tensor& centroids,
    const torch::Tensor& centroid_mask,
    const torch::Tensor& assignments
) {
    TORCH_CHECK(keys.is_cuda() && centroids.is_cuda() && centroid_mask.is_cuda(), "assignment inputs must be CUDA");
    TORCH_CHECK(assignments.is_cuda(), "assignment output must be CUDA");
    TORCH_CHECK(keys.dim() == 3 && centroids.dim() == 3 && centroid_mask.dim() == 2, "invalid assignment tensor rank");
    TORCH_CHECK(keys.is_contiguous() && centroids.is_contiguous() && centroid_mask.is_contiguous() && assignments.is_contiguous(), "assignment tensors must be contiguous");
    TORCH_CHECK(centroids.scalar_type() == torch::kFloat32, "centroids must be float32");
    TORCH_CHECK(centroid_mask.scalar_type() == torch::kBool && assignments.scalar_type() == torch::kInt32, "invalid assignment metadata dtype");
    const int rows = static_cast<int>(keys.size(0));
    const int tokens = static_cast<int>(keys.size(1));
    const int dim = static_cast<int>(keys.size(2));
    const int clusters = static_cast<int>(centroids.size(1));
    TORCH_CHECK(centroids.size(0) == rows && centroids.size(2) == dim, "centroid shape mismatch");
    TORCH_CHECK(centroid_mask.size(0) == rows && centroid_mask.size(1) == clusters, "centroid mask shape mismatch");
    TORCH_CHECK(assignments.size(0) == rows && assignments.size(1) == tokens, "assignment output shape mismatch");
    c10::cuda::CUDAGuard device_guard(keys.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(keys.get_device());
    if (keys.scalar_type() == torch::kBFloat16) {
        assign_generated_to_centroids_bf16(keys.data_ptr(), centroids.data_ptr<float>(), centroid_mask.data_ptr<bool>(), assignments.data_ptr<int32_t>(), rows, tokens, clusters, dim, stream.stream());
    } else if (keys.scalar_type() == torch::kFloat16) {
        assign_generated_to_centroids_fp16(keys.data_ptr(), centroids.data_ptr<float>(), centroid_mask.data_ptr<bool>(), assignments.data_ptr<int32_t>(), rows, tokens, clusters, dim, stream.stream());
    } else {
        TORCH_CHECK(false, "generated keys must be BF16 or FP16");
    }
}

void apply_generated_centroid_updates(
    const torch::Tensor& cluster_ids,
    const torch::Tensor& cluster_sums,
    const torch::Tensor& cluster_counts,
    const torch::Tensor& centroids,
    const torch::Tensor& base_cluster_counts,
    const torch::Tensor& generated_cluster_counts
) {
    TORCH_CHECK(cluster_ids.is_cuda() && cluster_sums.is_cuda() && cluster_counts.is_cuda(), "generated centroid updates must be CUDA");
    TORCH_CHECK(centroids.is_cuda() && base_cluster_counts.is_cuda() && generated_cluster_counts.is_cuda(), "centroid state must be CUDA");
    TORCH_CHECK(cluster_ids.dim() == 2 && cluster_sums.dim() == 3 && cluster_counts.dim() == 2, "invalid generated update rank");
    TORCH_CHECK(centroids.dim() == 3 && base_cluster_counts.dim() == 2 && generated_cluster_counts.dim() == 2, "invalid centroid state rank");
    TORCH_CHECK(cluster_ids.is_contiguous() && cluster_sums.is_contiguous() && cluster_counts.is_contiguous() && centroids.is_contiguous() && base_cluster_counts.is_contiguous() && generated_cluster_counts.is_contiguous(), "generated centroid tensors must be contiguous");
    TORCH_CHECK(cluster_ids.scalar_type() == torch::kInt32 && cluster_counts.scalar_type() == torch::kInt32 && base_cluster_counts.scalar_type() == torch::kInt32 && generated_cluster_counts.scalar_type() == torch::kInt32, "generated centroid count tensors must be int32");
    TORCH_CHECK(cluster_sums.scalar_type() == torch::kFloat32 && centroids.scalar_type() == torch::kFloat32, "generated centroid sums must be float32");
    const int rows = static_cast<int>(cluster_ids.size(0));
    const int updates = static_cast<int>(cluster_ids.size(1));
    const int clusters = static_cast<int>(centroids.size(1));
    const int dim = static_cast<int>(centroids.size(2));
    TORCH_CHECK(cluster_sums.size(0) == rows && cluster_sums.size(1) == updates && cluster_sums.size(2) == dim && cluster_counts.sizes() == cluster_ids.sizes(), "generated update shape mismatch");
    TORCH_CHECK(centroids.size(0) == rows && base_cluster_counts.size(0) == rows && base_cluster_counts.size(1) == clusters && generated_cluster_counts.sizes() == base_cluster_counts.sizes(), "generated centroid state shape mismatch");
    c10::cuda::CUDAGuard device_guard(centroids.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(centroids.get_device());
    apply_generated_centroid_updates(cluster_ids.data_ptr<int32_t>(), cluster_sums.data_ptr<float>(), cluster_counts.data_ptr<int32_t>(), centroids.data_ptr<float>(), base_cluster_counts.data_ptr<int32_t>(), generated_cluster_counts.data_ptr<int32_t>(), rows, updates, clusters, dim, stream.stream());
}

void publish_generated_page_metadata(
    const torch::Tensor& cluster_ids,
    const torch::Tensor& cluster_tails_in,
    const torch::Tensor& page_ids,
    const torch::Tensor& page_prev_in,
    const torch::Tensor& page_sizes_in,
    const torch::Tensor& cluster_tails_out,
    const torch::Tensor& page_prev_out,
    const torch::Tensor& page_sizes_out
) {
    TORCH_CHECK(cluster_ids.is_cuda() && cluster_tails_in.is_cuda() && page_ids.is_cuda() && page_prev_in.is_cuda() && page_sizes_in.is_cuda(), "generated page commit inputs must be CUDA");
    TORCH_CHECK(cluster_tails_out.is_cuda() && page_prev_out.is_cuda() && page_sizes_out.is_cuda(), "generated page commit state must be CUDA");
    TORCH_CHECK(cluster_ids.dim() == 2 && cluster_tails_in.sizes() == cluster_ids.sizes() && page_ids.sizes() == cluster_ids.sizes() && page_prev_in.sizes() == cluster_ids.sizes() && page_sizes_in.sizes() == cluster_ids.sizes(), "generated page commit input shape mismatch");
    TORCH_CHECK(cluster_tails_out.dim() == 2 && page_prev_out.dim() == 2 && page_sizes_out.sizes() == page_prev_out.sizes(), "generated page commit state shape mismatch");
    TORCH_CHECK(cluster_ids.is_contiguous() && cluster_tails_in.is_contiguous() && page_ids.is_contiguous() && page_prev_in.is_contiguous() && page_sizes_in.is_contiguous() && cluster_tails_out.is_contiguous() && page_prev_out.is_contiguous() && page_sizes_out.is_contiguous(), "generated page commit tensors must be contiguous");
    TORCH_CHECK(cluster_ids.scalar_type() == torch::kInt32 && cluster_tails_in.scalar_type() == torch::kInt32 && page_ids.scalar_type() == torch::kInt32 && page_prev_in.scalar_type() == torch::kInt32 && page_sizes_in.scalar_type() == torch::kInt32 && cluster_tails_out.scalar_type() == torch::kInt32 && page_prev_out.scalar_type() == torch::kInt32 && page_sizes_out.scalar_type() == torch::kInt32, "generated page commit tensors must be int32");
    const int rows = static_cast<int>(cluster_ids.size(0));
    const int updates = static_cast<int>(cluster_ids.size(1));
    const int clusters = static_cast<int>(cluster_tails_out.size(1));
    const int total_pages = static_cast<int>(page_prev_out.size(1));
    TORCH_CHECK(cluster_tails_out.size(0) == rows && page_prev_out.size(0) == rows, "generated page commit row mismatch");
    c10::cuda::CUDAGuard device_guard(cluster_tails_out.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(cluster_tails_out.get_device());
    publish_generated_page_metadata(cluster_ids.data_ptr<int32_t>(), cluster_tails_in.data_ptr<int32_t>(), page_ids.data_ptr<int32_t>(), page_prev_in.data_ptr<int32_t>(), page_sizes_in.data_ptr<int32_t>(), cluster_tails_out.data_ptr<int32_t>(), page_prev_out.data_ptr<int32_t>(), page_sizes_out.data_ptr<int32_t>(), rows, updates, clusters, total_pages, stream.stream());
}

} // namespace imi_decode
