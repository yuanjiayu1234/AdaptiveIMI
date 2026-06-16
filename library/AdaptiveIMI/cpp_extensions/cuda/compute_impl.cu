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

} // namespace imi_decode
