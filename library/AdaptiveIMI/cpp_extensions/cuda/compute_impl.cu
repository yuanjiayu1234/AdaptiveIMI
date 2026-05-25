// ============================================================================
// Compute Kernels for GPU Cluster Manager
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <string>
#include <stdexcept>

namespace cg = cooperative_groups;

namespace imi_decode {

// ============================================================================
// Helper: 类型转换
// ============================================================================
template<typename T>
__device__ __forceinline__ float to_float(T val);

template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template<>
__device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}

template<>
__device__ __forceinline__ float to_float<float>(float val) {
    return val;
}

template<typename T>
__device__ __forceinline__ T from_float(float val);

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}

template<>
__device__ __forceinline__ __half from_float<__half>(float val) {
    return __float2half(val);
}

template<>
__device__ __forceinline__ float from_float<float>(float val) {
    return val;
}

template<typename InputT>
__global__ void fused_query_group_similarity_kernel(
    const InputT* __restrict__ query_grouped,   // [kv_heads, num_query_groups, dim]
    const float* __restrict__ centroids,        // [kv_heads, n_clusters, dim]
    float* __restrict__ similarities,           // [kv_heads, n_clusters]
    int kv_heads,
    int num_query_groups,
    int n_clusters,
    int dim
) {
    extern __shared__ float shared_query[];

    int head_idx = blockIdx.x;
    if (head_idx >= kv_heads) {
        return;
    }

    // Phase 1: 聚合 query，直接写入 shared memory
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        const InputT* head_ptr = query_grouped + head_idx * (num_query_groups * dim) + d;
        for (int g = 0; g < num_query_groups; ++g) {
            acc += to_float(head_ptr[g * dim]);
        }
        shared_query[d] = acc;
    }
    __syncthreads();

    // Phase 2: 使用聚合后的 query 计算与所有簇的相似度
    const float* centroid_base = centroids + head_idx * n_clusters * dim;
    float* output_base = similarities + head_idx * n_clusters;

    for (int cluster = threadIdx.x; cluster < n_clusters; cluster += blockDim.x) {
        const float* centroid_ptr = centroid_base + cluster * dim;
        float sim = 0.0f;
#pragma unroll
        for (int d = 0; d < dim; ++d) {
            sim += shared_query[d] * centroid_ptr[d];
        }
        output_base[cluster] = sim;
    }
}

template<typename InputT>
__global__ void subspace_rerank_coarse_kernel(
    const InputT* __restrict__ query_grouped,             // [rows, groups, dim]
    const float* __restrict__ subspace_centroids,         // [rows, M, max_k, max_sub_dim]
    const bool* __restrict__ subspace_centroid_masks,     // [rows, M, max_k]
    const int64_t* __restrict__ subspace_dim_ids,         // [rows, M, max_sub_dim]
    const bool* __restrict__ subspace_dim_masks,          // [rows, M, max_sub_dim]
    const int32_t* __restrict__ compact_codes,            // [rows, active_clusters, M]
    float* __restrict__ coarse_scores,                    // [rows, active_clusters]
    int rows,
    int groups,
    int dim,
    int subspace_parts,
    int max_k,
    int max_sub_dim,
    int active_clusters
) {
    extern __shared__ float shared_query[];

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const InputT* row_query = query_grouped + row * groups * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int g = 0; g < groups; ++g) {
            acc += to_float(row_query[g * dim + d]);
        }
        shared_query[d] = acc;
    }
    __syncthreads();

    const float* row_sub_centroids =
        subspace_centroids + row * subspace_parts * max_k * max_sub_dim;
    const bool* row_sub_masks =
        subspace_centroid_masks + row * subspace_parts * max_k;
    const int64_t* row_dim_ids =
        subspace_dim_ids + row * subspace_parts * max_sub_dim;
    const bool* row_dim_masks =
        subspace_dim_masks + row * subspace_parts * max_sub_dim;
    const int32_t* row_codes = compact_codes + row * active_clusters * subspace_parts;
    float* row_scores = coarse_scores + row * active_clusters;

    const int cluster = blockIdx.y * blockDim.x + threadIdx.x;
    if (cluster >= active_clusters) {
        return;
    }

    float score = 0.0f;
    bool valid = true;
    const int32_t* code = row_codes + cluster * subspace_parts;

    for (int s = 0; s < subspace_parts; ++s) {
        const int code_digit = static_cast<int>(code[s]);
        if (code_digit < 0 || code_digit >= max_k ||
            row_sub_masks[s * max_k + code_digit]) {
            valid = false;
            break;
        }

        const float* centroid =
            row_sub_centroids + (s * max_k + code_digit) * max_sub_dim;
        const int64_t* dims = row_dim_ids + s * max_sub_dim;
        const bool* dim_mask = row_dim_masks + s * max_sub_dim;

        for (int j = 0; j < max_sub_dim; ++j) {
            if (dim_mask[j]) {
                continue;
            }
            const int d = static_cast<int>(dims[j]);
            if (d < 0 || d >= dim) {
                continue;
            }
            score += shared_query[d] * centroid[j];
        }
    }

    row_scores[cluster] = valid ? score : -FLT_MAX;
}

template<typename InputT>
__global__ void subspace_rerank_full_kernel(
    const InputT* __restrict__ query_grouped,     // [rows, groups, dim]
    const float* __restrict__ centroids,          // [rows, n_clusters, dim]
    const int64_t* __restrict__ candidate_ids,    // [rows, candidate_limit]
    const int32_t* __restrict__ cluster_sizes,    // [rows, n_clusters]
    float* __restrict__ rerank_scores,            // [rows, candidate_limit]
    int rows,
    int groups,
    int dim,
    int n_clusters,
    int candidate_limit
) {
    extern __shared__ float shared_query[];

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const InputT* row_query = query_grouped + row * groups * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int g = 0; g < groups; ++g) {
            acc += to_float(row_query[g * dim + d]);
        }
        shared_query[d] = acc;
    }
    __syncthreads();

    const float* row_centroids = centroids + row * n_clusters * dim;
    const int64_t* row_candidates = candidate_ids + row * candidate_limit;
    const int32_t* row_sizes = cluster_sizes + row * n_clusters;
    float* row_scores = rerank_scores + row * candidate_limit;

    const int idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (idx >= candidate_limit) {
        return;
    }

    const int cluster = static_cast<int>(row_candidates[idx]);
    if (cluster < 0 || cluster >= n_clusters || row_sizes[cluster] <= 0) {
        row_scores[idx] = -FLT_MAX;
        return;
    }

    const float* centroid = row_centroids + cluster * dim;
    float score = 0.0f;
    for (int d = 0; d < dim; ++d) {
        score += shared_query[d] * centroid[d];
    }
    row_scores[idx] = score;
}

__global__ void subspace_rerank_select_final_kernel(
    const int64_t* __restrict__ candidate_ids,     // [rows, candidate_limit]
    const int64_t* __restrict__ rerank_indices,    // [rows, rerank_limit]
    const int32_t* __restrict__ cluster_sizes,     // [rows, n_clusters]
    int64_t* __restrict__ selected,                // [rows, nprobe]
    int rows,
    int n_clusters,
    int nprobe,
    int candidate_limit,
    int rerank_limit,
    int max_retrieval_tokens,
    int64_t padding_cluster_id
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int64_t* row_selected = selected + row * nprobe;
    for (int idx = threadIdx.x; idx < nprobe; idx += blockDim.x) {
        row_selected[idx] = padding_cluster_id;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    const int64_t* row_candidates = candidate_ids + row * candidate_limit;
    const int64_t* row_rerank = rerank_indices + row * rerank_limit;
    const int32_t* row_sizes = cluster_sizes + row * n_clusters;

    int prefix_tokens = 0;
    const int selected_width = nprobe < rerank_limit ? nprobe : rerank_limit;
    for (int idx = 0; idx < selected_width; ++idx) {
        const int candidate_pos = static_cast<int>(row_rerank[idx]);
        if (candidate_pos < 0 || candidate_pos >= candidate_limit) {
            row_selected[idx] = padding_cluster_id;
            continue;
        }
        const int cluster = static_cast<int>(row_candidates[candidate_pos]);
        if (cluster < 0 || cluster >= n_clusters) {
            row_selected[idx] = padding_cluster_id;
            continue;
        }
        const int size = static_cast<int>(row_sizes[cluster]);
        if (size > 0 && prefix_tokens < max_retrieval_tokens) {
            row_selected[idx] = static_cast<int64_t>(cluster);
        } else {
            row_selected[idx] = padding_cluster_id;
        }
        if (size > 0) {
            prefix_tokens += size;
        }
    }
}




// ============================================================================//
// Section 1.5: Fused Query Aggregation + Similarity
// ============================================================================//

namespace {
__forceinline__ int determine_thread_block(int dim) {
    int threads = 256;
    if (dim < threads) {
        int warp_aligned = ((dim + 31) / 32) * 32;
        threads = max(32, warp_aligned);
    }
    return threads;
}
} // anonymous namespace

void fused_query_group_similarities_bf16(
    const void* query_grouped,
    const float* centroids,
    float* similarities,
    int num_query_groups,
    int n_clusters,
    int kv_heads,
    int dim,
    cudaStream_t stream
) {
    int threads = determine_thread_block(dim);
    size_t shared_mem = dim * sizeof(float);
    fused_query_group_similarity_kernel<__nv_bfloat16><<<kv_heads, threads, shared_mem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query_grouped),
        centroids,
        similarities,
        kv_heads,
        num_query_groups,
        n_clusters,
        dim
    );
}

void fused_query_group_similarities_fp16(
    const void* query_grouped,
    const float* centroids,
    float* similarities,
    int num_query_groups,
    int n_clusters,
    int kv_heads,
    int dim,
    cudaStream_t stream
) {
    int threads = determine_thread_block(dim);
    size_t shared_mem = dim * sizeof(float);
    fused_query_group_similarity_kernel<__half><<<kv_heads, threads, shared_mem, stream>>>(
        reinterpret_cast<const __half*>(query_grouped),
        centroids,
        similarities,
        kv_heads,
        num_query_groups,
        n_clusters,
        dim
    );
}

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
    cudaStream_t stream
) {
    constexpr int threads = 256;
    const size_t shared_mem = static_cast<size_t>(dim) * sizeof(float);
    dim3 grid(rows, (active_clusters + threads - 1) / threads);
    subspace_rerank_coarse_kernel<__nv_bfloat16><<<grid, threads, shared_mem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query_grouped),
        subspace_centroids,
        subspace_centroid_masks,
        subspace_dim_ids,
        subspace_dim_masks,
        compact_codes,
        coarse_scores,
        rows,
        groups,
        dim,
        subspace_parts,
        max_k,
        max_sub_dim,
        active_clusters
    );
}

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
    cudaStream_t stream
) {
    constexpr int threads = 256;
    const size_t shared_mem = static_cast<size_t>(dim) * sizeof(float);
    dim3 grid(rows, (active_clusters + threads - 1) / threads);
    subspace_rerank_coarse_kernel<__half><<<grid, threads, shared_mem, stream>>>(
        reinterpret_cast<const __half*>(query_grouped),
        subspace_centroids,
        subspace_centroid_masks,
        subspace_dim_ids,
        subspace_dim_masks,
        compact_codes,
        coarse_scores,
        rows,
        groups,
        dim,
        subspace_parts,
        max_k,
        max_sub_dim,
        active_clusters
    );
}

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
    cudaStream_t stream
) {
    constexpr int threads = 256;
    const size_t shared_mem = static_cast<size_t>(dim) * sizeof(float);
    dim3 grid(rows, (candidate_limit + threads - 1) / threads);
    subspace_rerank_full_kernel<__nv_bfloat16><<<grid, threads, shared_mem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query_grouped),
        centroids,
        candidate_ids,
        cluster_sizes,
        rerank_scores,
        rows,
        groups,
        dim,
        n_clusters,
        candidate_limit
    );
}

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
    cudaStream_t stream
) {
    constexpr int threads = 256;
    const size_t shared_mem = static_cast<size_t>(dim) * sizeof(float);
    dim3 grid(rows, (candidate_limit + threads - 1) / threads);
    subspace_rerank_full_kernel<__half><<<grid, threads, shared_mem, stream>>>(
        reinterpret_cast<const __half*>(query_grouped),
        centroids,
        candidate_ids,
        cluster_sizes,
        rerank_scores,
        rows,
        groups,
        dim,
        n_clusters,
        candidate_limit
    );
}

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
    cudaStream_t stream
) {
    constexpr int threads = 256;
    subspace_rerank_select_final_kernel<<<rows, threads, 0, stream>>>(
        candidate_ids,
        rerank_indices,
        cluster_sizes,
        selected,
        rows,
        n_clusters,
        nprobe,
        candidate_limit,
        rerank_limit,
        max_retrieval_tokens,
        padding_cluster_id
    );
}

} // namespace imi_decode
