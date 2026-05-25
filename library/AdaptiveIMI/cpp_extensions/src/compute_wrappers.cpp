/**
 * compute_impl.cpp - Python Wrappers for IMI Compute Kernels
 *
 * 提供Python可调用的wrapper函数，封装CUDA kernel调用
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <algorithm>
#include <tuple>

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

void subspace_rerank_coarse_scores_bf16(
    const void* query_grouped,
    const float* subspace_centroids,
    const bool* subspace_centroid_masks,
    const int64_t* subspace_dim_ids,
    const bool* subspace_dim_masks,
    const int32_t* compact_codes,
    float* coarse_scores,
    int rows,
    int groups,
    int dim,
    int subspace_parts,
    int max_k,
    int max_sub_dim,
    int active_clusters,
    cudaStream_t stream = 0
);

void subspace_rerank_coarse_scores_fp16(
    const void* query_grouped,
    const float* subspace_centroids,
    const bool* subspace_centroid_masks,
    const int64_t* subspace_dim_ids,
    const bool* subspace_dim_masks,
    const int32_t* compact_codes,
    float* coarse_scores,
    int rows,
    int groups,
    int dim,
    int subspace_parts,
    int max_k,
    int max_sub_dim,
    int active_clusters,
    cudaStream_t stream = 0
);

void subspace_rerank_full_scores_bf16(
    const void* query_grouped,
    const float* centroids,
    const int64_t* candidate_ids,
    const int32_t* cluster_sizes,
    float* rerank_scores,
    int rows,
    int groups,
    int dim,
    int n_clusters,
    int candidate_limit,
    cudaStream_t stream = 0
);

void subspace_rerank_full_scores_fp16(
    const void* query_grouped,
    const float* centroids,
    const int64_t* candidate_ids,
    const int32_t* cluster_sizes,
    float* rerank_scores,
    int rows,
    int groups,
    int dim,
    int n_clusters,
    int candidate_limit,
    cudaStream_t stream = 0
);

void subspace_rerank_select_final(
    const int64_t* candidate_ids,
    const int64_t* rerank_indices,
    const int32_t* cluster_sizes,
    int64_t* selected,
    int rows,
    int n_clusters,
    int nprobe,
    int candidate_limit,
    int rerank_limit,
    int max_retrieval_tokens,
    int64_t padding_cluster_id,
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

torch::Tensor fused_subspace_rerank_select(
    const torch::Tensor& query_grouped,              // [rows, groups, dim] BF16/FP16
    const torch::Tensor& subspace_centroids,         // [rows, M, max_k, max_sub_dim] FP32
    const torch::Tensor& subspace_centroid_masks,    // [rows, M, max_k] bool
    const torch::Tensor& subspace_dim_ids,           // [rows, M, max_sub_dim] int64
    const torch::Tensor& subspace_dim_masks,         // [rows, M, max_sub_dim] bool
    const torch::Tensor& compact_codes,              // [rows, active_clusters, M] int32
    const torch::Tensor& centroids,                  // [rows, n_clusters, dim] FP32
    const torch::Tensor& cluster_sizes,              // [rows, n_clusters] int32
    int64_t nprobe,
    int64_t candidate_limit,
    int64_t rerank_limit,
    int64_t max_retrieval_tokens,
    int64_t padding_cluster_id
) {
    TORCH_CHECK(query_grouped.is_cuda(), "query_grouped must be on GPU");
    TORCH_CHECK(subspace_centroids.is_cuda(), "subspace_centroids must be on GPU");
    TORCH_CHECK(subspace_centroid_masks.is_cuda(), "subspace_centroid_masks must be on GPU");
    TORCH_CHECK(subspace_dim_ids.is_cuda(), "subspace_dim_ids must be on GPU");
    TORCH_CHECK(subspace_dim_masks.is_cuda(), "subspace_dim_masks must be on GPU");
    TORCH_CHECK(compact_codes.is_cuda(), "compact_codes must be on GPU");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be on GPU");
    TORCH_CHECK(cluster_sizes.is_cuda(), "cluster_sizes must be on GPU");

    TORCH_CHECK(query_grouped.dim() == 3, "query_grouped must be [rows, groups, dim]");
    TORCH_CHECK(subspace_centroids.dim() == 4, "subspace_centroids must be [rows, M, max_k, max_sub_dim]");
    TORCH_CHECK(subspace_centroid_masks.dim() == 3, "subspace_centroid_masks must be [rows, M, max_k]");
    TORCH_CHECK(subspace_dim_ids.dim() == 3, "subspace_dim_ids must be [rows, M, max_sub_dim]");
    TORCH_CHECK(subspace_dim_masks.dim() == 3, "subspace_dim_masks must be [rows, M, max_sub_dim]");
    TORCH_CHECK(compact_codes.dim() == 3, "compact_codes must be [rows, active_clusters, M]");
    TORCH_CHECK(centroids.dim() == 3, "centroids must be [rows, n_clusters, dim]");
    TORCH_CHECK(cluster_sizes.dim() == 2, "cluster_sizes must be [rows, n_clusters]");

    TORCH_CHECK(subspace_centroids.scalar_type() == torch::kFloat32, "subspace_centroids must be FP32");
    TORCH_CHECK(subspace_centroid_masks.scalar_type() == torch::kBool, "subspace_centroid_masks must be bool");
    TORCH_CHECK(subspace_dim_ids.scalar_type() == torch::kInt64, "subspace_dim_ids must be int64");
    TORCH_CHECK(subspace_dim_masks.scalar_type() == torch::kBool, "subspace_dim_masks must be bool");
    TORCH_CHECK(compact_codes.scalar_type() == torch::kInt32, "compact_codes must be int32");
    TORCH_CHECK(centroids.scalar_type() == torch::kFloat32, "centroids must be FP32");
    TORCH_CHECK(cluster_sizes.scalar_type() == torch::kInt32, "cluster_sizes must be int32");

    const int rows = static_cast<int>(query_grouped.size(0));
    const int groups = static_cast<int>(query_grouped.size(1));
    const int dim = static_cast<int>(query_grouped.size(2));
    const int subspace_parts = static_cast<int>(subspace_centroids.size(1));
    const int max_k = static_cast<int>(subspace_centroids.size(2));
    const int max_sub_dim = static_cast<int>(subspace_centroids.size(3));
    const int active_clusters = static_cast<int>(compact_codes.size(1));
    const int n_clusters = static_cast<int>(centroids.size(1));

    TORCH_CHECK(rows > 0, "rows must be positive");
    TORCH_CHECK(active_clusters > 0, "active_clusters must be positive");
    TORCH_CHECK(nprobe > 0, "nprobe must be positive");
    TORCH_CHECK(subspace_centroids.size(0) == rows, "subspace_centroids rows mismatch");
    TORCH_CHECK(subspace_centroid_masks.size(0) == rows, "subspace_centroid_masks rows mismatch");
    TORCH_CHECK(subspace_centroid_masks.size(1) == subspace_parts, "subspace_centroid_masks M mismatch");
    TORCH_CHECK(subspace_centroid_masks.size(2) == max_k, "subspace_centroid_masks max_k mismatch");
    TORCH_CHECK(subspace_dim_ids.size(0) == rows, "subspace_dim_ids rows mismatch");
    TORCH_CHECK(subspace_dim_ids.size(1) == subspace_parts, "subspace_dim_ids M mismatch");
    TORCH_CHECK(subspace_dim_ids.size(2) == max_sub_dim, "subspace_dim_ids max_sub_dim mismatch");
    TORCH_CHECK(subspace_dim_masks.size(0) == subspace_dim_ids.size(0) &&
                subspace_dim_masks.size(1) == subspace_dim_ids.size(1) &&
                subspace_dim_masks.size(2) == subspace_dim_ids.size(2),
                "subspace_dim_masks shape mismatch");
    TORCH_CHECK(compact_codes.size(0) == rows, "compact_codes rows mismatch");
    TORCH_CHECK(compact_codes.size(2) == subspace_parts, "compact_codes M mismatch");
    TORCH_CHECK(centroids.size(0) == rows, "centroids rows mismatch");
    TORCH_CHECK(centroids.size(2) == dim, "centroids dim mismatch");
    TORCH_CHECK(cluster_sizes.size(0) == rows, "cluster_sizes rows mismatch");
    TORCH_CHECK(cluster_sizes.size(1) == n_clusters, "cluster_sizes n_clusters mismatch");

    const int candidate_k = std::max<int>(1, std::min<int>(
        std::max<int>(static_cast<int>(candidate_limit), static_cast<int>(nprobe)),
        active_clusters
    ));
    const int rerank_k = std::max<int>(1, std::min<int>(
        std::max<int>(static_cast<int>(rerank_limit), static_cast<int>(nprobe)),
        candidate_k
    ));
    const int output_k = static_cast<int>(nprobe);

    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(query_grouped.device());
    auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(query_grouped.device());

    auto coarse_scores = torch::empty({rows, active_clusters}, float_options);
    auto rerank_scores = torch::empty({rows, candidate_k}, float_options);
    auto selected = torch::empty({rows, output_k}, long_options);

    c10::cuda::CUDAGuard device_guard(query_grouped.device());
    auto stream = c10::cuda::getCurrentCUDAStream(query_grouped.get_device());

    if (query_grouped.scalar_type() == torch::kBFloat16) {
        subspace_rerank_coarse_scores_bf16(
            query_grouped.data_ptr(),
            subspace_centroids.data_ptr<float>(),
            subspace_centroid_masks.data_ptr<bool>(),
            subspace_dim_ids.data_ptr<int64_t>(),
            subspace_dim_masks.data_ptr<bool>(),
            compact_codes.data_ptr<int32_t>(),
            coarse_scores.data_ptr<float>(),
            rows,
            groups,
            dim,
            subspace_parts,
            max_k,
            max_sub_dim,
            active_clusters,
            stream.stream()
        );
    } else if (query_grouped.scalar_type() == torch::kFloat16) {
        subspace_rerank_coarse_scores_fp16(
            query_grouped.data_ptr(),
            subspace_centroids.data_ptr<float>(),
            subspace_centroid_masks.data_ptr<bool>(),
            subspace_dim_ids.data_ptr<int64_t>(),
            subspace_dim_masks.data_ptr<bool>(),
            compact_codes.data_ptr<int32_t>(),
            coarse_scores.data_ptr<float>(),
            rows,
            groups,
            dim,
            subspace_parts,
            max_k,
            max_sub_dim,
            active_clusters,
            stream.stream()
        );
    } else {
        TORCH_CHECK(false, "fused_subspace_rerank_select: query dtype must be BF16 or FP16");
    }

    auto coarse_topk = at::topk(coarse_scores, candidate_k, 1, true, false);
    auto candidate_ids = std::get<1>(coarse_topk).contiguous();

    if (query_grouped.scalar_type() == torch::kBFloat16) {
        subspace_rerank_full_scores_bf16(
            query_grouped.data_ptr(),
            centroids.data_ptr<float>(),
            candidate_ids.data_ptr<int64_t>(),
            cluster_sizes.data_ptr<int32_t>(),
            rerank_scores.data_ptr<float>(),
            rows,
            groups,
            dim,
            n_clusters,
            candidate_k,
            stream.stream()
        );
    } else {
        subspace_rerank_full_scores_fp16(
            query_grouped.data_ptr(),
            centroids.data_ptr<float>(),
            candidate_ids.data_ptr<int64_t>(),
            cluster_sizes.data_ptr<int32_t>(),
            rerank_scores.data_ptr<float>(),
            rows,
            groups,
            dim,
            n_clusters,
            candidate_k,
            stream.stream()
        );
    }

    auto rerank_topk = at::topk(rerank_scores, rerank_k, 1, true, true);
    auto rerank_indices = std::get<1>(rerank_topk).contiguous();
    subspace_rerank_select_final(
        candidate_ids.data_ptr<int64_t>(),
        rerank_indices.data_ptr<int64_t>(),
        cluster_sizes.data_ptr<int32_t>(),
        selected.data_ptr<int64_t>(),
        rows,
        n_clusters,
        output_k,
        candidate_k,
        rerank_k,
        static_cast<int>(max_retrieval_tokens),
        padding_cluster_id,
        stream.stream()
    );

    return selected;
}

} // namespace imi_decode
