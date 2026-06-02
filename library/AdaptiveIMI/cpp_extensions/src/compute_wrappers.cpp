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

void fused_query_group_softmax_sum_similarities_bf16(
    const void* query_grouped,
    const float* centroids,
    float* logits,
    float* similarities,
    int num_query_groups,
    int n_clusters,
    int rows,
    int dim,
    cudaStream_t stream = 0
);

void fused_query_group_softmax_sum_similarities_fp16(
    const void* query_grouped,
    const float* centroids,
    float* logits,
    float* similarities,
    int num_query_groups,
    int n_clusters,
    int rows,
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

void subspace_rerank_select_final(
    const int64_t* candidate_ids,
    const int32_t* cluster_sizes,
    int64_t* selected,
    int rows,
    int n_clusters,
    int nprobe,
    int candidate_limit,
    int max_retrieval_tokens,
    int64_t padding_cluster_id,
    cudaStream_t stream = 0
);

void token_rerank_scores_bf16(
    const void* query_grouped,
    const void* execution_keys,
    const int32_t* valid_lengths,
    const int32_t* static_lengths,
    float* scores,
    int rows,
    int groups,
    int dim,
    int stride,
    int max_candidates,
    cudaStream_t stream = 0
);

void token_rerank_scores_fp16(
    const void* query_grouped,
    const void* execution_keys,
    const int32_t* valid_lengths,
    const int32_t* static_lengths,
    float* scores,
    int rows,
    int groups,
    int dim,
    int stride,
    int max_candidates,
    cudaStream_t stream = 0
);

void token_rerank_gather_bf16(
    const void* execution_keys,
    const void* execution_values,
    const int64_t* token_indices,
    const int32_t* valid_lengths,
    const int32_t* static_lengths,
    void* selected_keys,
    void* selected_values,
    int rows,
    int dim,
    int stride,
    int topk,
    cudaStream_t stream = 0
);

void token_rerank_gather_fp16(
    const void* execution_keys,
    const void* execution_values,
    const int64_t* token_indices,
    const int32_t* valid_lengths,
    const int32_t* static_lengths,
    void* selected_keys,
    void* selected_values,
    int rows,
    int dim,
    int stride,
    int topk,
    cudaStream_t stream = 0
);

void token_rerank_scatter_bf16(
    void* execution_keys,
    void* execution_values,
    const void* selected_keys,
    const void* selected_values,
    int32_t* valid_lengths,
    const int32_t* static_lengths,
    int rows,
    int dim,
    int stride,
    int topk,
    cudaStream_t stream = 0
);

void token_rerank_scatter_fp16(
    void* execution_keys,
    void* execution_values,
    const void* selected_keys,
    const void* selected_values,
    int32_t* valid_lengths,
    const int32_t* static_lengths,
    int rows,
    int dim,
    int stride,
    int topk,
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

torch::Tensor fused_query_group_softmax_sum_similarities(
    const torch::Tensor& query_grouped,  // [rows, num_query_groups, dim] BF16/FP16
    const torch::Tensor& centroids,      // [rows, n_clusters, dim] FP32
    const torch::Tensor& logits,         // [rows, num_query_groups, n_clusters] FP32 scratch
    c10::optional<torch::Tensor> out_tensor
) {
    TORCH_CHECK(query_grouped.is_cuda(), "query_grouped must be on GPU");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be on GPU");
    TORCH_CHECK(logits.is_cuda(), "logits must be on GPU");
    TORCH_CHECK(query_grouped.dim() == 3, "query_grouped must be 3D");
    TORCH_CHECK(centroids.dim() == 3, "centroids must be 3D");
    TORCH_CHECK(logits.dim() == 3, "logits must be 3D");
    TORCH_CHECK(centroids.scalar_type() == torch::kFloat32, "centroids must be FP32");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be FP32");

    int rows = query_grouped.size(0);
    int groups = query_grouped.size(1);
    int dim = query_grouped.size(2);
    int n_clusters = centroids.size(1);

    TORCH_CHECK(centroids.size(0) == rows, "centroids rows mismatch");
    TORCH_CHECK(centroids.size(2) == dim, "centroids dim mismatch");
    TORCH_CHECK(logits.size(0) >= rows && logits.size(1) >= groups && logits.size(2) >= n_clusters,
                "logits tensor shape insufficient");

    torch::Tensor similarities;
    if (out_tensor.has_value() && out_tensor->defined()) {
        auto out = out_tensor.value();
        TORCH_CHECK(out.is_cuda(), "fused_query_group_softmax_sum_similarities: out tensor must be on GPU");
        TORCH_CHECK(out.scalar_type() == torch::kFloat32, "fused_query_group_softmax_sum_similarities: out tensor must be FP32");
        TORCH_CHECK(out.dim() == 2, "fused_query_group_softmax_sum_similarities: out tensor must be 2D");
        TORCH_CHECK(out.size(0) >= rows && out.size(1) >= n_clusters,
                    "fused_query_group_softmax_sum_similarities: out tensor shape insufficient");
        similarities = out.narrow(0, 0, rows).narrow(1, 0, n_clusters);
    } else {
        similarities = torch::empty(
            {rows, n_clusters},
            torch::TensorOptions().dtype(torch::kFloat32).device(query_grouped.device())
        );
    }

    auto logits_view = logits.narrow(0, 0, rows).narrow(1, 0, groups).narrow(2, 0, n_clusters);
    TORCH_CHECK(logits_view.is_contiguous(), "logits scratch must be contiguous for the active shape");

    c10::cuda::CUDAGuard device_guard(query_grouped.device());
    auto stream = c10::cuda::getCurrentCUDAStream(query_grouped.get_device());

    if (query_grouped.scalar_type() == torch::kBFloat16) {
        fused_query_group_softmax_sum_similarities_bf16(
            query_grouped.data_ptr(),
            centroids.data_ptr<float>(),
            logits_view.data_ptr<float>(),
            similarities.data_ptr<float>(),
            groups,
            n_clusters,
            rows,
            dim,
            stream.stream()
        );
    } else if (query_grouped.scalar_type() == torch::kFloat16) {
        fused_query_group_softmax_sum_similarities_fp16(
            query_grouped.data_ptr(),
            centroids.data_ptr<float>(),
            logits_view.data_ptr<float>(),
            similarities.data_ptr<float>(),
            groups,
            n_clusters,
            rows,
            dim,
            stream.stream()
        );
    } else {
        TORCH_CHECK(false, "fused_query_group_softmax_sum_similarities: query dtype must be BF16 or FP16");
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
    const torch::Tensor& cluster_sizes,              // [rows, n_clusters] int32
    int64_t nprobe,
    int64_t candidate_limit,
    int64_t max_retrieval_tokens,
    int64_t padding_cluster_id
) {
    TORCH_CHECK(query_grouped.is_cuda(), "query_grouped must be on GPU");
    TORCH_CHECK(subspace_centroids.is_cuda(), "subspace_centroids must be on GPU");
    TORCH_CHECK(subspace_centroid_masks.is_cuda(), "subspace_centroid_masks must be on GPU");
    TORCH_CHECK(subspace_dim_ids.is_cuda(), "subspace_dim_ids must be on GPU");
    TORCH_CHECK(subspace_dim_masks.is_cuda(), "subspace_dim_masks must be on GPU");
    TORCH_CHECK(compact_codes.is_cuda(), "compact_codes must be on GPU");
    TORCH_CHECK(cluster_sizes.is_cuda(), "cluster_sizes must be on GPU");

    TORCH_CHECK(query_grouped.dim() == 3, "query_grouped must be [rows, groups, dim]");
    TORCH_CHECK(subspace_centroids.dim() == 4, "subspace_centroids must be [rows, M, max_k, max_sub_dim]");
    TORCH_CHECK(subspace_centroid_masks.dim() == 3, "subspace_centroid_masks must be [rows, M, max_k]");
    TORCH_CHECK(subspace_dim_ids.dim() == 3, "subspace_dim_ids must be [rows, M, max_sub_dim]");
    TORCH_CHECK(subspace_dim_masks.dim() == 3, "subspace_dim_masks must be [rows, M, max_sub_dim]");
    TORCH_CHECK(compact_codes.dim() == 3, "compact_codes must be [rows, active_clusters, M]");
    TORCH_CHECK(cluster_sizes.dim() == 2, "cluster_sizes must be [rows, n_clusters]");

    TORCH_CHECK(subspace_centroids.scalar_type() == torch::kFloat32, "subspace_centroids must be FP32");
    TORCH_CHECK(subspace_centroid_masks.scalar_type() == torch::kBool, "subspace_centroid_masks must be bool");
    TORCH_CHECK(subspace_dim_ids.scalar_type() == torch::kInt64, "subspace_dim_ids must be int64");
    TORCH_CHECK(subspace_dim_masks.scalar_type() == torch::kBool, "subspace_dim_masks must be bool");
    TORCH_CHECK(compact_codes.scalar_type() == torch::kInt32, "compact_codes must be int32");
    TORCH_CHECK(cluster_sizes.scalar_type() == torch::kInt32, "cluster_sizes must be int32");

    const int rows = static_cast<int>(query_grouped.size(0));
    const int groups = static_cast<int>(query_grouped.size(1));
    const int dim = static_cast<int>(query_grouped.size(2));
    const int subspace_parts = static_cast<int>(subspace_centroids.size(1));
    const int max_k = static_cast<int>(subspace_centroids.size(2));
    const int max_sub_dim = static_cast<int>(subspace_centroids.size(3));
    const int active_clusters = static_cast<int>(compact_codes.size(1));
    const int n_clusters = static_cast<int>(cluster_sizes.size(1));

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
    TORCH_CHECK(cluster_sizes.size(0) == rows, "cluster_sizes rows mismatch");
    TORCH_CHECK(cluster_sizes.size(1) == n_clusters, "cluster_sizes n_clusters mismatch");

    const int candidate_k = std::max<int>(1, std::min<int>(
        std::max<int>(static_cast<int>(candidate_limit), static_cast<int>(nprobe)),
        active_clusters
    ));
    const int output_k = static_cast<int>(nprobe);

    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(query_grouped.device());
    auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(query_grouped.device());

    auto coarse_scores = torch::empty({rows, active_clusters}, float_options);
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

    auto coarse_topk = at::topk(coarse_scores, candidate_k, 1, true, true);
    auto candidate_ids = std::get<1>(coarse_topk).contiguous();

    subspace_rerank_select_final(
        candidate_ids.data_ptr<int64_t>(),
        cluster_sizes.data_ptr<int32_t>(),
        selected.data_ptr<int64_t>(),
        rows,
        n_clusters,
        output_k,
        candidate_k,
        static_cast<int>(max_retrieval_tokens),
        padding_cluster_id,
        stream.stream()
    );

    return selected;
}

void token_rerank_compact(
    const torch::Tensor& query_grouped,       // [rows, groups, dim] BF16/FP16
    torch::Tensor& execution_keys,            // [rows, stride, 1, dim] BF16/FP16
    torch::Tensor& execution_values,          // [rows, stride, 1, dim] BF16/FP16
    torch::Tensor& valid_lengths,             // [rows] int32, updated in place
    const torch::Tensor& static_lengths,      // [rows] int32
    int64_t token_budget
) {
    TORCH_CHECK(query_grouped.is_cuda(), "query_grouped must be on GPU");
    TORCH_CHECK(execution_keys.is_cuda(), "execution_keys must be on GPU");
    TORCH_CHECK(execution_values.is_cuda(), "execution_values must be on GPU");
    TORCH_CHECK(valid_lengths.is_cuda(), "valid_lengths must be on GPU");
    TORCH_CHECK(static_lengths.is_cuda(), "static_lengths must be on GPU");
    TORCH_CHECK(query_grouped.dim() == 3, "query_grouped must be [rows, groups, dim]");
    TORCH_CHECK(execution_keys.dim() == 4, "execution_keys must be [rows, stride, 1, dim]");
    TORCH_CHECK(execution_values.dim() == 4, "execution_values must be [rows, stride, 1, dim]");
    TORCH_CHECK(valid_lengths.dim() == 1, "valid_lengths must be [rows]");
    TORCH_CHECK(static_lengths.dim() == 1, "static_lengths must be [rows]");
    TORCH_CHECK(execution_keys.size(2) == 1, "execution_keys singleton axis must be 1");
    TORCH_CHECK(execution_values.size(2) == 1, "execution_values singleton axis must be 1");
    TORCH_CHECK(execution_keys.scalar_type() == query_grouped.scalar_type(), "execution_keys dtype mismatch");
    TORCH_CHECK(execution_values.scalar_type() == query_grouped.scalar_type(), "execution_values dtype mismatch");
    TORCH_CHECK(valid_lengths.scalar_type() == torch::kInt32, "valid_lengths must be int32");
    TORCH_CHECK(static_lengths.scalar_type() == torch::kInt32, "static_lengths must be int32");
    TORCH_CHECK(query_grouped.is_contiguous(), "query_grouped must be contiguous");
    TORCH_CHECK(execution_keys.is_contiguous(), "execution_keys must be contiguous");
    TORCH_CHECK(execution_values.is_contiguous(), "execution_values must be contiguous");

    const int rows = static_cast<int>(query_grouped.size(0));
    const int groups = static_cast<int>(query_grouped.size(1));
    const int dim = static_cast<int>(query_grouped.size(2));
    const int stride = static_cast<int>(execution_keys.size(1));
    TORCH_CHECK(rows > 0, "rows must be positive");
    TORCH_CHECK(dim > 0 && dim <= 1024, "dim must be in (0, 1024]");
    TORCH_CHECK(execution_keys.size(0) == rows, "execution_keys rows mismatch");
    TORCH_CHECK(execution_values.size(0) == rows, "execution_values rows mismatch");
    TORCH_CHECK(execution_values.size(1) == stride, "execution_values stride mismatch");
    TORCH_CHECK(execution_keys.size(3) == dim, "execution_keys dim mismatch");
    TORCH_CHECK(execution_values.size(3) == dim, "execution_values dim mismatch");
    TORCH_CHECK(valid_lengths.size(0) == rows, "valid_lengths rows mismatch");
    TORCH_CHECK(static_lengths.size(0) == rows, "static_lengths rows mismatch");
    if (token_budget <= 0 || stride <= 0) {
        return;
    }

    auto valid_cpu = valid_lengths.detach().to(torch::kCPU);
    auto static_cpu = static_lengths.detach().to(torch::kCPU);
    int max_candidates = 0;
    int max_selected = 0;
    const int32_t* valid_ptr = valid_cpu.data_ptr<int32_t>();
    const int32_t* static_ptr = static_cpu.data_ptr<int32_t>();
    for (int row = 0; row < rows; ++row) {
        const int valid_len = std::max(0, std::min(static_cast<int>(valid_ptr[row]), stride));
        const int static_len = std::max(0, std::min(static_cast<int>(static_ptr[row]), valid_len));
        const int candidates = std::max(valid_len - static_len, 0);
        max_candidates = std::max(max_candidates, candidates);
        max_selected = std::max(max_selected, std::min<int>(candidates, static_cast<int>(token_budget)));
    }
    if (max_candidates <= 0 || max_selected <= 0) {
        return;
    }

    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(query_grouped.device());
    auto scores = torch::empty({rows, max_candidates}, float_options);

    c10::cuda::CUDAGuard device_guard(query_grouped.device());
    auto stream = c10::cuda::getCurrentCUDAStream(query_grouped.get_device());

    if (query_grouped.scalar_type() == torch::kBFloat16) {
        token_rerank_scores_bf16(
            query_grouped.data_ptr(),
            execution_keys.data_ptr(),
            valid_lengths.data_ptr<int32_t>(),
            static_lengths.data_ptr<int32_t>(),
            scores.data_ptr<float>(),
            rows,
            groups,
            dim,
            stride,
            max_candidates,
            stream.stream()
        );
    } else if (query_grouped.scalar_type() == torch::kFloat16) {
        token_rerank_scores_fp16(
            query_grouped.data_ptr(),
            execution_keys.data_ptr(),
            valid_lengths.data_ptr<int32_t>(),
            static_lengths.data_ptr<int32_t>(),
            scores.data_ptr<float>(),
            rows,
            groups,
            dim,
            stride,
            max_candidates,
            stream.stream()
        );
    } else {
        TORCH_CHECK(false, "token_rerank_compact: query dtype must be BF16 or FP16");
    }

    auto topk = at::topk(scores, max_selected, 1, true, true);
    auto token_indices = std::get<1>(topk).contiguous();
    auto selected_keys = torch::empty({rows, max_selected, 1, dim}, execution_keys.options());
    auto selected_values = torch::empty({rows, max_selected, 1, dim}, execution_values.options());

    if (query_grouped.scalar_type() == torch::kBFloat16) {
        token_rerank_gather_bf16(
            execution_keys.data_ptr(),
            execution_values.data_ptr(),
            token_indices.data_ptr<int64_t>(),
            valid_lengths.data_ptr<int32_t>(),
            static_lengths.data_ptr<int32_t>(),
            selected_keys.data_ptr(),
            selected_values.data_ptr(),
            rows,
            dim,
            stride,
            max_selected,
            stream.stream()
        );
        token_rerank_scatter_bf16(
            execution_keys.data_ptr(),
            execution_values.data_ptr(),
            selected_keys.data_ptr(),
            selected_values.data_ptr(),
            valid_lengths.data_ptr<int32_t>(),
            static_lengths.data_ptr<int32_t>(),
            rows,
            dim,
            stride,
            max_selected,
            stream.stream()
        );
    } else {
        token_rerank_gather_fp16(
            execution_keys.data_ptr(),
            execution_values.data_ptr(),
            token_indices.data_ptr<int64_t>(),
            valid_lengths.data_ptr<int32_t>(),
            static_lengths.data_ptr<int32_t>(),
            selected_keys.data_ptr(),
            selected_values.data_ptr(),
            rows,
            dim,
            stride,
            max_selected,
            stream.stream()
        );
        token_rerank_scatter_fp16(
            execution_keys.data_ptr(),
            execution_values.data_ptr(),
            selected_keys.data_ptr(),
            selected_values.data_ptr(),
            valid_lengths.data_ptr<int32_t>(),
            static_lengths.data_ptr<int32_t>(),
            rows,
            dim,
            stride,
            max_selected,
            stream.stream()
        );
    }
}

} // namespace imi_decode
