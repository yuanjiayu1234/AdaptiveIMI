// Decode compute kernels for AdaptiveIMI.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imi_decode {

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

template<typename InputT, int WARPS_PER_BLOCK>
__global__ void fused_query_group_similarity_cluster_kernel(
    const InputT* __restrict__ query_grouped,   // [kv_heads, num_query_groups, dim]
    const float* __restrict__ centroids,        // [kv_heads, n_clusters, dim]
    float* __restrict__ similarities,           // [kv_heads, n_clusters]
    int kv_heads,
    int num_query_groups,
    int n_clusters,
    int dim
) {
    extern __shared__ float shared_query[];

    const int head_idx = blockIdx.x;
    const int warp_idx = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int cluster = blockIdx.y * WARPS_PER_BLOCK + warp_idx;
    if (head_idx >= kv_heads) {
        return;
    }

    // Fixed flat decode path: sum GQA query groups, then score every cluster centroid.
    const int64_t query_head_stride = static_cast<int64_t>(num_query_groups) * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        const InputT* head_ptr = query_grouped + static_cast<int64_t>(head_idx) * query_head_stride + d;
        for (int group = 0; group < num_query_groups; ++group) {
            acc += to_float(head_ptr[static_cast<int64_t>(group) * dim]);
        }
        shared_query[d] = acc;
    }
    __syncthreads();

    if (cluster >= n_clusters) {
        return;
    }

    const float* centroid_ptr = centroids + (static_cast<int64_t>(head_idx) * n_clusters + cluster) * dim;
    float sim = 0.0f;
    for (int d = lane; d < dim; d += 32) {
        sim += shared_query[d] * centroid_ptr[d];
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sim += __shfl_down_sync(0xffffffff, sim, offset);
    }

    if (lane == 0) {
        similarities[static_cast<int64_t>(head_idx) * n_clusters + cluster] = sim;
    }
}

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
    constexpr int warps_per_block = 4;
    constexpr int threads = warps_per_block * 32;
    const size_t shared_mem = static_cast<size_t>(dim) * sizeof(float);
    const dim3 grid(kv_heads, (n_clusters + warps_per_block - 1) / warps_per_block);
    fused_query_group_similarity_cluster_kernel<__nv_bfloat16, warps_per_block><<<grid, threads, shared_mem, stream>>>(
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
    constexpr int warps_per_block = 4;
    constexpr int threads = warps_per_block * 32;
    const size_t shared_mem = static_cast<size_t>(dim) * sizeof(float);
    const dim3 grid(kv_heads, (n_clusters + warps_per_block - 1) / warps_per_block);
    fused_query_group_similarity_cluster_kernel<__half, warps_per_block><<<grid, threads, shared_mem, stream>>>(
        reinterpret_cast<const __half*>(query_grouped),
        centroids,
        similarities,
        kv_heads,
        num_query_groups,
        n_clusters,
        dim
    );
}

__device__ __forceinline__ int warp_inclusive_scan_int(int value) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffff, value, offset);
        if ((threadIdx.x & 31) >= offset) {
            value += other;
        }
    }
    return value;
}

__global__ void apply_budget_after_topk_kernel(
    const int64_t* __restrict__ topk_indices,      // [rows, nprobe]
    const int32_t* __restrict__ cluster_sizes,     // [rows, n_centroids]
    int64_t* __restrict__ selected,                // [rows, nprobe]
    int64_t selected_stride0,
    int rows,
    int nprobe,
    int n_centroids,
    int max_retrieval_tokens,
    int64_t padding_cluster_id
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ int warp_prefix_totals[32];
    __shared__ int tile_base_tokens;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = (blockDim.x + 31) >> 5;
    const int64_t row_topk_offset = static_cast<int64_t>(row) * nprobe;
    const int64_t row_cluster_offset = static_cast<int64_t>(row) * n_centroids;

    if (tid == 0) {
        tile_base_tokens = 0;
    }
    __syncthreads();

    for (int base = 0; base < nprobe; base += blockDim.x) {
        const int col = base + tid;
        int64_t cluster_id = padding_cluster_id;
        int size = 0;
        if (col < nprobe) {
            cluster_id = topk_indices[row_topk_offset + col];
            if (cluster_id >= 0 && cluster_id < n_centroids) {
                size = cluster_sizes[row_cluster_offset + cluster_id];
            }
        }

        int prefix = warp_inclusive_scan_int(size);
        if (lane == 31) {
            warp_prefix_totals[warp] = prefix;
        }
        __syncthreads();

        if (warp == 0) {
            int warp_total = (lane < warps_per_block) ? warp_prefix_totals[lane] : 0;
            warp_total = warp_inclusive_scan_int(warp_total);
            if (lane < warps_per_block) {
                warp_prefix_totals[lane] = warp_total;
            }
        }
        __syncthreads();

        if (warp > 0) {
            prefix += warp_prefix_totals[warp - 1];
        }

        if (col < nprobe) {
            const int tokens_before_cluster = tile_base_tokens + prefix - size;
            const bool keep = (size > 0) && (tokens_before_cluster < max_retrieval_tokens);
            selected[static_cast<int64_t>(row) * selected_stride0 + col] = keep ? cluster_id : padding_cluster_id;
        }
        __syncthreads();

        if (tid == 0) {
            tile_base_tokens += warp_prefix_totals[warps_per_block - 1];
        }
        __syncthreads();
    }
}

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
    cudaStream_t stream
) {
    if (rows <= 0 || nprobe <= 0) {
        return;
    }
    constexpr int threads = 256;
    apply_budget_after_topk_kernel<<<rows, threads, 0, stream>>>(
        topk_indices,
        cluster_sizes,
        selected,
        selected_stride0,
        rows,
        nprobe,
        n_centroids,
        max_retrieval_tokens,
        padding_cluster_id
    );
}

template <typename InputT>
__global__ void assign_generated_to_centroids_kernel(
    const InputT* __restrict__ keys,              // [rows, tokens, dim]
    const float* __restrict__ centroids,           // [rows, clusters, dim]
    const bool* __restrict__ centroid_mask,        // [rows, clusters]
    int32_t* __restrict__ assignments,             // [rows, tokens]
    int rows,
    int tokens,
    int clusters,
    int dim
) {
    const int token = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows || token >= tokens) return;

    const InputT* key = keys + (static_cast<int64_t>(row) * tokens + token) * dim;
    float best_score = -INFINITY;
    int best_cluster = -1;
    for (int cluster = threadIdx.x; cluster < clusters; cluster += blockDim.x) {
        if (centroid_mask[static_cast<int64_t>(row) * clusters + cluster]) continue;
        const float* center = centroids + (static_cast<int64_t>(row) * clusters + cluster) * dim;
        float score = 0.0f;
        for (int d = 0; d < dim; ++d) score += to_float(key[d]) * center[d];
        if (score > best_score || (score == best_score && cluster < best_cluster)) {
            best_score = score;
            best_cluster = cluster;
        }
    }

    __shared__ float scores[256];
    __shared__ int ids[256];
    scores[threadIdx.x] = best_score;
    ids[threadIdx.x] = best_cluster;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_score = scores[threadIdx.x + stride];
            const int other_id = ids[threadIdx.x + stride];
            if (other_score > scores[threadIdx.x] ||
                (other_score == scores[threadIdx.x] && other_id >= 0 &&
                 (ids[threadIdx.x] < 0 || other_id < ids[threadIdx.x]))) {
                scores[threadIdx.x] = other_score;
                ids[threadIdx.x] = other_id;
            }
        }
        __syncthreads();
    }

    const int cluster = ids[0];
    if (threadIdx.x == 0) assignments[static_cast<int64_t>(row) * tokens + token] = cluster;
}

void assign_generated_to_centroids_bf16(
    const void* keys, const float* centroids, const bool* centroid_mask,
    int32_t* assignments,
    int rows, int tokens, int clusters, int dim, cudaStream_t stream
) {
    constexpr int threads = 256;
    assign_generated_to_centroids_kernel<__nv_bfloat16><<<dim3(tokens, rows), threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(keys), centroids, centroid_mask,
        assignments, rows, tokens, clusters, dim);
}

void assign_generated_to_centroids_fp16(
    const void* keys, const float* centroids, const bool* centroid_mask,
    int32_t* assignments,
    int rows, int tokens, int clusters, int dim, cudaStream_t stream
) {
    constexpr int threads = 256;
    assign_generated_to_centroids_kernel<__half><<<dim3(tokens, rows), threads, 0, stream>>>(
        reinterpret_cast<const __half*>(keys), centroids, centroid_mask,
        assignments, rows, tokens, clusters, dim);
}

__global__ void apply_generated_centroid_updates_kernel(
    const int32_t* __restrict__ cluster_ids,       // [rows, updates], -1 padding
    const float* __restrict__ cluster_sums,         // [rows, updates, dim]
    const int32_t* __restrict__ cluster_counts,     // [rows, updates]
    float* __restrict__ centroids,                  // [rows, clusters, dim]
    const int32_t* __restrict__ base_cluster_counts,// [rows, clusters]
    int32_t* __restrict__ generated_cluster_counts, // [rows, clusters]
    int rows,
    int updates,
    int clusters,
    int dim
) {
    const int row = blockIdx.x;
    const int update = blockIdx.y;
    if (row >= rows || update >= updates) return;
    const int64_t update_base = static_cast<int64_t>(row) * updates + update;
    const int cluster = cluster_ids[update_base];
    const int added = cluster_counts[update_base];
    if (cluster < 0 || cluster >= clusters || added <= 0) return;

    const int64_t cluster_base = static_cast<int64_t>(row) * clusters + cluster;
    const int old_generated = generated_cluster_counts[cluster_base];
    const int base_count = base_cluster_counts[cluster_base];
    const int old_total = base_count + old_generated;
    const int new_total = old_total + added;
    const int64_t sum_base = update_base * dim;
    const int64_t centroid_base = cluster_base * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float old_sum = centroids[centroid_base + d] * static_cast<float>(old_total);
        centroids[centroid_base + d] = (old_sum + cluster_sums[sum_base + d]) / static_cast<float>(new_total);
    }
    if (threadIdx.x == 0) generated_cluster_counts[cluster_base] = old_generated + added;
}

void apply_generated_centroid_updates(
    const int32_t* cluster_ids, const float* cluster_sums, const int32_t* cluster_counts,
    float* centroids, const int32_t* base_cluster_counts, int32_t* generated_cluster_counts,
    int rows, int updates, int clusters, int dim, cudaStream_t stream
) {
    if (rows <= 0 || updates <= 0) return;
    apply_generated_centroid_updates_kernel<<<dim3(rows, updates), 256, 0, stream>>>(
        cluster_ids, cluster_sums, cluster_counts, centroids, base_cluster_counts,
        generated_cluster_counts, rows, updates, clusters, dim);
}

__global__ void publish_generated_page_metadata_kernel(
    const int32_t* __restrict__ cluster_ids,
    const int32_t* __restrict__ cluster_tails_in,
    const int32_t* __restrict__ page_ids,
    const int32_t* __restrict__ page_prev_in,
    const int32_t* __restrict__ page_sizes_in,
    int32_t* __restrict__ cluster_tails_out,
    int32_t* __restrict__ page_prev_out,
    int32_t* __restrict__ page_sizes_out,
    int rows,
    int updates,
    int clusters,
    int total_pages
) {
    const int update = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= rows || update >= updates) return;
    const int64_t source = static_cast<int64_t>(row) * updates + update;
    const int cluster = cluster_ids[source];
    if (cluster >= 0 && cluster < clusters) {
        cluster_tails_out[static_cast<int64_t>(row) * clusters + cluster] = cluster_tails_in[source];
    }
    const int page = page_ids[source];
    if (page >= 0 && page < total_pages) {
        const int64_t destination = static_cast<int64_t>(row) * total_pages + page;
        page_prev_out[destination] = page_prev_in[source];
        page_sizes_out[destination] = page_sizes_in[source];
    }
}

void publish_generated_page_metadata(
    const int32_t* cluster_ids, const int32_t* cluster_tails_in,
    const int32_t* page_ids, const int32_t* page_prev_in, const int32_t* page_sizes_in,
    int32_t* cluster_tails_out, int32_t* page_prev_out, int32_t* page_sizes_out,
    int rows, int updates, int clusters, int total_pages, cudaStream_t stream
) {
    if (rows <= 0 || updates <= 0) return;
    constexpr int threads = 128;
    publish_generated_page_metadata_kernel<<<dim3((updates + threads - 1) / threads, rows), threads, 0, stream>>>(
        cluster_ids, cluster_tails_in, page_ids, page_prev_in, page_sizes_in,
        cluster_tails_out, page_prev_out, page_sizes_out,
        rows, updates, clusters, total_pages);
}

} // namespace imi_decode
