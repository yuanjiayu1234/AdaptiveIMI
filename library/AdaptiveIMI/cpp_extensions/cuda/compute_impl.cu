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

} // namespace imi_decode
