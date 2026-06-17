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
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        const InputT* head_ptr = query_grouped + head_idx * (num_query_groups * dim) + d;
        for (int group = 0; group < num_query_groups; ++group) {
            acc += to_float(head_ptr[group * dim]);
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

} // namespace imi_decode
