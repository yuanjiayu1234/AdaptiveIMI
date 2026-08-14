#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cfloat>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace imi_hybrid {

namespace {

constexpr int kSubspaces = 8;

template <typename scalar_t>
__device__ __forceinline__ float load_key_value(
    const scalar_t* __restrict__ keys,
    int64_t b,
    int64_t h,
    int64_t t,
    int64_t d,
    int64_t stride_b,
    int64_t stride_h,
    int64_t stride_t,
    int64_t stride_d
) {
    const int64_t offset = b * stride_b + h * stride_h + t * stride_t + d * stride_d;
    return static_cast<float>(keys[offset]);
}

__device__ __forceinline__ int subspace_dim_count(
    const bool* __restrict__ dim_masks,
    int64_t base,
    int max_sub_dim
) {
    int count = 0;
    for (int i = 0; i < max_sub_dim; ++i) {
        if (dim_masks[base + i]) {
            break;
        }
        ++count;
    }
    return count;
}

inline int capped_blocks(int64_t total, int threads, int sm_count, int sm_multiplier = 2) {
    if (total <= 0) {
        return 1;
    }
    const int64_t needed = (total + threads - 1) / threads;
    const int64_t cap = std::max<int64_t>(1, static_cast<int64_t>(std::max(1, sm_count)) * sm_multiplier);
    return static_cast<int>(std::min<int64_t>(needed, cap));
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ uint64_t make_seed(int row, int subspace, int k, int iter, uint64_t salt) {
    uint64_t seed = salt;
    seed ^= static_cast<uint64_t>(row + 1) * 0x9E3779B185EBCA87ULL;
    seed ^= static_cast<uint64_t>(subspace + 42) * 0xC2B2AE3D27D4EB4FULL;
    seed ^= static_cast<uint64_t>(k + 1) * 0x165667B19E3779F9ULL;
    seed ^= static_cast<uint64_t>(iter + 12345) * 0x85EBCA77C2B2AE63ULL;
    return splitmix64(seed);
}

__device__ __forceinline__ int sample_token_index(int token_count, int row, int subspace, int k, int iter, uint64_t salt) {
    if (token_count <= 1) {
        return 0;
    }
    return static_cast<int>(make_seed(row, subspace, k, iter, salt) % static_cast<uint64_t>(token_count));
}

__global__ void materialize_hybrid_decode_metadata_kernel(
    const float* __restrict__ centroids_compact,
    const int32_t* __restrict__ cluster_sizes_compact,
    const int32_t* __restrict__ cluster_counts,
    float* __restrict__ centroids_full,
    int32_t* __restrict__ cluster_sizes_full,
    bool* __restrict__ centroids_mask_full,
    int64_t rows,
    int64_t compact_clusters,
    int64_t n_centroids,
    int64_t head_dim
) {
    const int64_t total_centroids = rows * compact_clusters * head_dim;
    const int64_t total_metadata = rows * n_centroids;
    const int64_t total = total_centroids > total_metadata ? total_centroids : total_metadata;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        if (idx < total_centroids) {
            const int64_t row_stride = compact_clusters * head_dim;
            const int64_t row = idx / row_stride;
            const int64_t row_rem = idx - row * row_stride;
            const int64_t cluster = row_rem / head_dim;
            const int64_t dim = row_rem - cluster * head_dim;
            centroids_full[(row * n_centroids + cluster) * head_dim + dim] =
                centroids_compact[idx];
        }
        if (idx < total_metadata) {
            const int64_t row = idx / n_centroids;
            const int64_t cluster = idx - row * n_centroids;
            const int32_t count = cluster_counts[row];
            const bool valid = cluster < compact_clusters && cluster < static_cast<int64_t>(count);
            const int32_t size = valid ? cluster_sizes_compact[row * compact_clusters + cluster] : 0;
            cluster_sizes_full[idx] = size;
            centroids_mask_full[idx] = (!valid) || (size == 0);
        }
    }
}

template <typename scalar_t>
__global__ void init_centroids_sampled_kernel(
    const scalar_t* __restrict__ keys,
    const int32_t* __restrict__ row_token_counts,
    const int64_t* __restrict__ dim_ids,
    const bool* __restrict__ dim_masks,
    const int32_t* __restrict__ k_values,
    float* __restrict__ centroids,
    bool* __restrict__ centroid_masks,
    int rows,
    int kv_heads,
    int max_tokens,
    int head_dim,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_t,
    int64_t key_stride_d,
    int max_k,
    int max_sub_dim
) {
    const int task = blockIdx.x;
    const int row = task / kSubspaces;
    const int subspace = task - row * kSubspaces;
    if (row >= rows) {
        return;
    }

    const int dim_base = (row * kSubspaces + subspace) * max_sub_dim;
    const int sub_dim = subspace_dim_count(dim_masks, dim_base, max_sub_dim);
    const int kval = k_values[subspace];
    const int token_count = row_token_counts[row];
    const int b = row / kv_heads;
    const int h = row - b * kv_heads;

    for (int linear = threadIdx.x; linear < max_k * max_sub_dim; linear += blockDim.x) {
        const int dim = linear % max_sub_dim;
        const int k = linear / max_sub_dim;
        const int centroid_base = ((row * kSubspaces + subspace) * max_k + k) * max_sub_dim;
        centroids[centroid_base + dim] = 0.0f;
        if (dim == 0) {
            centroid_masks[(row * kSubspaces + subspace) * max_k + k] =
                !(token_count > 0 && k < kval);
        }
    }
    __syncthreads();

    if (token_count <= 0 || kval <= 0 || sub_dim <= 0) {
        return;
    }

    for (int centroid_id = 0; centroid_id < kval; ++centroid_id) {
        const int selected = sample_token_index(token_count, row, subspace, centroid_id, 0, 42424242ULL);
        const int centroid_base = ((row * kSubspaces + subspace) * max_k + centroid_id) * max_sub_dim;
        for (int d = threadIdx.x; d < sub_dim; d += blockDim.x) {
            const int64_t source_dim = dim_ids[dim_base + d];
            centroids[centroid_base + d] = load_key_value(
                keys, b, h, selected, source_dim, key_stride_b, key_stride_h, key_stride_t, key_stride_d
            );
        }
        if (threadIdx.x == 0) {
            centroid_masks[(row * kSubspaces + subspace) * max_k + centroid_id] = false;
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void assign_reduce_kernel(
    const scalar_t* __restrict__ keys,
    const int32_t* __restrict__ row_token_counts,
    const int64_t* __restrict__ dim_ids,
    const bool* __restrict__ dim_masks,
    const int32_t* __restrict__ k_values,
    const float* __restrict__ centroids,
    int32_t* __restrict__ labels,
    float* __restrict__ sums,
    float* __restrict__ counts,
    int rows,
    int kv_heads,
    int max_tokens,
    int head_dim,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_t,
    int64_t key_stride_d,
    int max_k,
    int max_sub_dim
) {
    const int64_t total = static_cast<int64_t>(rows) * kSubspaces * max_tokens;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        int tmp = static_cast<int>(linear);
        const int token = tmp % max_tokens;
        tmp /= max_tokens;
        const int subspace = tmp % kSubspaces;
        const int row = tmp / kSubspaces;

        const int token_count = row_token_counts[row];
        const int label_offset = (row * kSubspaces + subspace) * max_tokens + token;
        if (token >= token_count || token_count <= 0) {
            labels[label_offset] = 0;
            continue;
        }

        const int kval = k_values[subspace];
        const int dim_base = (row * kSubspaces + subspace) * max_sub_dim;
        const int sub_dim = subspace_dim_count(dim_masks, dim_base, max_sub_dim);
        const int b = row / kv_heads;
        const int h = row - b * kv_heads;

        float best_dist = FLT_MAX;
        int best_k = 0;
        for (int k = 0; k < kval; ++k) {
            float dist = 0.0f;
            const int centroid_base = ((row * kSubspaces + subspace) * max_k + k) * max_sub_dim;
            for (int d = 0; d < sub_dim; ++d) {
                const int64_t source_dim = dim_ids[dim_base + d];
                const float x = load_key_value(keys, b, h, token, source_dim, key_stride_b, key_stride_h, key_stride_t, key_stride_d);
                const float c = centroids[centroid_base + d];
                const float diff = x - c;
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }

        labels[label_offset] = best_k;
        atomicAdd(&counts[(row * kSubspaces + subspace) * max_k + best_k], 1.0f);
        const int sum_base = ((row * kSubspaces + subspace) * max_k + best_k) * max_sub_dim;
        for (int d = 0; d < sub_dim; ++d) {
            const int64_t source_dim = dim_ids[dim_base + d];
            const float x = load_key_value(keys, b, h, token, source_dim, key_stride_b, key_stride_h, key_stride_t, key_stride_d);
            atomicAdd(&sums[sum_base + d], x);
        }
    }
}

template <typename scalar_t>
__global__ void assign_reduce_tiled_kernel(
    const scalar_t* __restrict__ keys,
    const int32_t* __restrict__ row_token_counts,
    const int64_t* __restrict__ dim_ids,
    const bool* __restrict__ dim_masks,
    const int32_t* __restrict__ k_values,
    const float* __restrict__ centroids,
    int32_t* __restrict__ labels,
    float* __restrict__ partial_sums,
    float* __restrict__ partial_counts,
    int rows,
    int kv_heads,
    int max_tokens,
    int head_dim,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_t,
    int64_t key_stride_d,
    int max_k,
    int max_sub_dim,
    int tile_tokens,
    int tiles_per_task
) {
    extern __shared__ float shared[];
    float* shared_sums = shared;
    float* shared_counts = shared + max_k * max_sub_dim;
    float* shared_centroids = shared_counts + max_k;
    int* shared_dim_ids = reinterpret_cast<int*>(shared_centroids + max_k * max_sub_dim);

    const int tile = blockIdx.x % tiles_per_task;
    const int task = blockIdx.x / tiles_per_task;
    const int row = task / kSubspaces;
    const int subspace = task - row * kSubspaces;
    if (row >= rows) {
        return;
    }

    const int dim_base = (row * kSubspaces + subspace) * max_sub_dim;
    const int sub_dim = subspace_dim_count(dim_masks, dim_base, max_sub_dim);
    const int kval = k_values[subspace];
    const int token_count = row_token_counts[row];
    const int b = row / kv_heads;
    const int h = row - b * kv_heads;

    for (int idx = threadIdx.x; idx < max_k * max_sub_dim; idx += blockDim.x) {
        shared_sums[idx] = 0.0f;
        shared_centroids[idx] = centroids[((row * kSubspaces + subspace) * max_k * max_sub_dim) + idx];
    }
    for (int idx = threadIdx.x; idx < max_k; idx += blockDim.x) {
        shared_counts[idx] = 0.0f;
    }
    for (int idx = threadIdx.x; idx < max_sub_dim; idx += blockDim.x) {
        shared_dim_ids[idx] = (idx < sub_dim) ? static_cast<int>(dim_ids[dim_base + idx]) : 0;
    }
    __syncthreads();

    const int tile_start = tile * tile_tokens;
    const int tile_end = min(tile_start + tile_tokens, token_count);
    for (int token = tile_start + threadIdx.x; token < tile_end; token += blockDim.x) {
        float best_dist = FLT_MAX;
        int best_k = 0;
        for (int k = 0; k < kval; ++k) {
            float dist = 0.0f;
            for (int d = 0; d < sub_dim; ++d) {
                const int source_dim = shared_dim_ids[d];
                const float x = load_key_value(keys, b, h, token, source_dim, key_stride_b, key_stride_h, key_stride_t, key_stride_d);
                const float c = shared_centroids[k * max_sub_dim + d];
                const float diff = x - c;
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }

        labels[(row * kSubspaces + subspace) * max_tokens + token] = best_k;
        atomicAdd(&shared_counts[best_k], 1.0f);
        const int sum_base = best_k * max_sub_dim;
        for (int d = 0; d < sub_dim; ++d) {
            const int source_dim = shared_dim_ids[d];
            const float x = load_key_value(keys, b, h, token, source_dim, key_stride_b, key_stride_h, key_stride_t, key_stride_d);
            atomicAdd(&shared_sums[sum_base + d], x);
        }
    }
    __syncthreads();

    const int64_t partial_base = static_cast<int64_t>(blockIdx.x) * max_k;
    for (int k = threadIdx.x; k < max_k; k += blockDim.x) {
        partial_counts[partial_base + k] = shared_counts[k];
    }
    const int64_t sum_base = static_cast<int64_t>(blockIdx.x) * max_k * max_sub_dim;
    for (int idx = threadIdx.x; idx < max_k * max_sub_dim; idx += blockDim.x) {
        partial_sums[sum_base + idx] = shared_sums[idx];
    }
}

template <typename scalar_t>
__global__ void assign_reduce_tiled_warp_private_kernel(
    const scalar_t* __restrict__ keys,
    const int32_t* __restrict__ row_token_counts,
    const int64_t* __restrict__ dim_ids,
    const bool* __restrict__ dim_masks,
    const int32_t* __restrict__ k_values,
    const float* __restrict__ centroids,
    int32_t* __restrict__ labels,
    float* __restrict__ partial_sums,
    float* __restrict__ partial_counts,
    int rows,
    int kv_heads,
    int max_tokens,
    int head_dim,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_t,
    int64_t key_stride_d,
    int max_k,
    int max_sub_dim,
    int tile_tokens,
    int tiles_per_task
) {
    extern __shared__ float shared[];
    const int warp_count = (blockDim.x + 31) / 32;
    const int warp_id = threadIdx.x >> 5;

    float* warp_sums = shared;
    float* warp_counts = warp_sums + warp_count * max_k * max_sub_dim;
    float* shared_centroids = warp_counts + warp_count * max_k;
    int* shared_dim_ids = reinterpret_cast<int*>(shared_centroids + max_k * max_sub_dim);

    const int tile = blockIdx.x % tiles_per_task;
    const int task = blockIdx.x / tiles_per_task;
    const int row = task / kSubspaces;
    const int subspace = task - row * kSubspaces;
    if (row >= rows) {
        return;
    }

    const int dim_base = (row * kSubspaces + subspace) * max_sub_dim;
    const int sub_dim = subspace_dim_count(dim_masks, dim_base, max_sub_dim);
    const int kval = k_values[subspace];
    const int token_count = row_token_counts[row];
    const int b = row / kv_heads;
    const int h = row - b * kv_heads;

    const int warp_sum_total = warp_count * max_k * max_sub_dim;
    const int warp_count_total = warp_count * max_k;
    for (int idx = threadIdx.x; idx < warp_sum_total; idx += blockDim.x) {
        warp_sums[idx] = 0.0f;
    }
    for (int idx = threadIdx.x; idx < warp_count_total; idx += blockDim.x) {
        warp_counts[idx] = 0.0f;
    }
    for (int idx = threadIdx.x; idx < max_k * max_sub_dim; idx += blockDim.x) {
        shared_centroids[idx] = centroids[((row * kSubspaces + subspace) * max_k * max_sub_dim) + idx];
    }
    for (int idx = threadIdx.x; idx < max_sub_dim; idx += blockDim.x) {
        shared_dim_ids[idx] = (idx < sub_dim) ? static_cast<int>(dim_ids[dim_base + idx]) : 0;
    }
    __syncthreads();

    const int tile_start = tile * tile_tokens;
    const int tile_end = min(tile_start + tile_tokens, token_count);
    for (int token = tile_start + threadIdx.x; token < tile_end; token += blockDim.x) {
        float best_dist = FLT_MAX;
        int best_k = 0;
        for (int k = 0; k < kval; ++k) {
            float dist = 0.0f;
            for (int d = 0; d < sub_dim; ++d) {
                const int source_dim = shared_dim_ids[d];
                const float x = load_key_value(keys, b, h, token, source_dim, key_stride_b, key_stride_h, key_stride_t, key_stride_d);
                const float c = shared_centroids[k * max_sub_dim + d];
                const float diff = x - c;
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }

        labels[(row * kSubspaces + subspace) * max_tokens + token] = best_k;
        atomicAdd(&warp_counts[warp_id * max_k + best_k], 1.0f);
        const int sum_base = (warp_id * max_k + best_k) * max_sub_dim;
        for (int d = 0; d < sub_dim; ++d) {
            const int source_dim = shared_dim_ids[d];
            const float x = load_key_value(keys, b, h, token, source_dim, key_stride_b, key_stride_h, key_stride_t, key_stride_d);
            atomicAdd(&warp_sums[sum_base + d], x);
        }
    }
    __syncthreads();

    const int64_t partial_count_base = static_cast<int64_t>(blockIdx.x) * max_k;
    for (int k = threadIdx.x; k < max_k; k += blockDim.x) {
        float total = 0.0f;
        for (int warp = 0; warp < warp_count; ++warp) {
            total += warp_counts[warp * max_k + k];
        }
        partial_counts[partial_count_base + k] = total;
    }

    const int64_t partial_sum_base = static_cast<int64_t>(blockIdx.x) * max_k * max_sub_dim;
    for (int idx = threadIdx.x; idx < max_k * max_sub_dim; idx += blockDim.x) {
        const int d = idx % max_sub_dim;
        const int k = idx / max_sub_dim;
        float total = 0.0f;
        for (int warp = 0; warp < warp_count; ++warp) {
            total += warp_sums[(warp * max_k + k) * max_sub_dim + d];
        }
        partial_sums[partial_sum_base + idx] = total;
    }
}

template <typename scalar_t>
__global__ void reduce_update_tiled_kernel(
    const scalar_t* __restrict__ keys,
    const int32_t* __restrict__ row_token_counts,
    const int64_t* __restrict__ dim_ids,
    const bool* __restrict__ dim_masks,
    const int32_t* __restrict__ k_values,
    const float* __restrict__ partial_sums,
    const float* __restrict__ partial_counts,
    float* __restrict__ centroids,
    bool* __restrict__ centroid_masks,
    float* __restrict__ counts_out,
    int rows,
    int kv_heads,
    int max_tokens,
    int head_dim,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_t,
    int64_t key_stride_d,
    int max_k,
    int max_sub_dim,
    int tiles_per_task,
    int iter
) {
    const int64_t total = static_cast<int64_t>(rows) * kSubspaces * max_k * max_sub_dim;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        int tmp = static_cast<int>(linear);
        const int dim = tmp % max_sub_dim;
        tmp /= max_sub_dim;
        const int k = tmp % max_k;
        tmp /= max_k;
        const int subspace = tmp % kSubspaces;
        const int row = tmp / kSubspaces;

        const int kval = k_values[subspace];
        const int dim_base = (row * kSubspaces + subspace) * max_sub_dim;
        const bool valid_dim = !dim_masks[dim_base + dim];
        const int token_count = row_token_counts[row];
        const bool valid_centroid = token_count > 0 && k < kval;
        const int mask_offset = (row * kSubspaces + subspace) * max_k + k;
        const int value_offset = ((row * kSubspaces + subspace) * max_k + k) * max_sub_dim + dim;

        if (dim == 0) {
            centroid_masks[mask_offset] = !valid_centroid;
        }
        if (!valid_centroid || !valid_dim) {
            if (!valid_dim) {
                centroids[value_offset] = 0.0f;
            }
            continue;
        }

        float count = 0.0f;
        float sum = 0.0f;
        const int task = row * kSubspaces + subspace;
        for (int tile = 0; tile < tiles_per_task; ++tile) {
            const int64_t partial_task = static_cast<int64_t>(task) * tiles_per_task + tile;
            count += partial_counts[partial_task * max_k + k];
            sum += partial_sums[(partial_task * max_k + k) * max_sub_dim + dim];
        }
        if (dim == 0) {
            counts_out[mask_offset] = count;
        }

        if (count > 0.0f) {
            centroids[value_offset] = sum / count;
        } else {
            const int b = row / kv_heads;
            const int h = row - b * kv_heads;
            const int token = sample_token_index(token_count, row, subspace, k, iter, 12345ULL);
            const int64_t source_dim = dim_ids[dim_base + dim];
            centroids[value_offset] = load_key_value(
                keys, b, h, token, source_dim, key_stride_b, key_stride_h, key_stride_t, key_stride_d
            );
        }
    }
}

template <typename scalar_t>
__global__ void update_centroids_kernel(
    const scalar_t* __restrict__ keys,
    const int32_t* __restrict__ row_token_counts,
    const int64_t* __restrict__ dim_ids,
    const bool* __restrict__ dim_masks,
    const int32_t* __restrict__ k_values,
    const float* __restrict__ sums,
    const float* __restrict__ counts,
    float* __restrict__ centroids,
    bool* __restrict__ centroid_masks,
    int rows,
    int kv_heads,
    int max_tokens,
    int head_dim,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_t,
    int64_t key_stride_d,
    int max_k,
    int max_sub_dim,
    int iter
) {
    const int64_t total = static_cast<int64_t>(rows) * kSubspaces * max_k * max_sub_dim;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        int tmp = static_cast<int>(linear);
        const int dim = tmp % max_sub_dim;
        tmp /= max_sub_dim;
        const int k = tmp % max_k;
        tmp /= max_k;
        const int subspace = tmp % kSubspaces;
        const int row = tmp / kSubspaces;

        const int kval = k_values[subspace];
        const int dim_base = (row * kSubspaces + subspace) * max_sub_dim;
        const bool valid_dim = !dim_masks[dim_base + dim];
        const int token_count = row_token_counts[row];
        const bool valid_centroid = token_count > 0 && k < kval;
        const int mask_offset = (row * kSubspaces + subspace) * max_k + k;
        const int value_offset = ((row * kSubspaces + subspace) * max_k + k) * max_sub_dim + dim;

        if (dim == 0) {
            centroid_masks[mask_offset] = !valid_centroid;
        }
        if (!valid_centroid || !valid_dim) {
            if (!valid_dim) {
                centroids[value_offset] = 0.0f;
            }
            continue;
        }

        const float count = counts[mask_offset];
        if (count > 0.0f) {
            centroids[value_offset] = sums[value_offset] / count;
        } else {
            const int b = row / kv_heads;
            const int h = row - b * kv_heads;
            const int token = sample_token_index(token_count, row, subspace, k, iter, 12345ULL);
            const int64_t source_dim = dim_ids[dim_base + dim];
            centroids[value_offset] = load_key_value(
                keys, b, h, token, source_dim, key_stride_b, key_stride_h, key_stride_t, key_stride_d
            );
        }
    }
}

__device__ __forceinline__ uint64_t pack_code8_uint64(
    const int32_t* __restrict__ labels,
    const int32_t* __restrict__ k_values,
    int row,
    int token,
    int max_tokens
) {
    uint64_t code = 0;
    uint64_t stride = 1;
    for (int s = 0; s < kSubspaces; ++s) {
        const int kval = k_values[s] > 1 ? k_values[s] : 1;
        const int label = labels[(row * kSubspaces + s) * max_tokens + token];
        if (label < 0 || label >= kval) {
            return UINT64_MAX;
        }
        if (stride > UINT64_MAX / static_cast<uint64_t>(kval)) {
            return UINT64_MAX;
        }
        code += static_cast<uint64_t>(label) * stride;
        stride *= static_cast<uint64_t>(kval);
    }
    return code;
}

__global__ void init_segment_offsets_kernel(
    int64_t* __restrict__ offsets,
    int rows,
    int max_tokens
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx <= rows; idx += gridDim.x * blockDim.x) {
        offsets[idx] = static_cast<int64_t>(idx) * max_tokens;
    }
}

__global__ void pack_sorted_code_inputs_kernel(
    const int32_t* __restrict__ labels,
    const int32_t* __restrict__ row_token_counts,
    const int32_t* __restrict__ k_values,
    uint64_t* __restrict__ codes,
    int32_t* __restrict__ token_ids,
    int rows,
    int max_tokens
) {
    const int64_t total = static_cast<int64_t>(rows) * max_tokens;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        const int token = static_cast<int>(linear % max_tokens);
        const int row = static_cast<int>(linear / max_tokens);
        if (token >= row_token_counts[row]) {
            codes[linear] = UINT64_MAX;
            token_ids[linear] = -1;
            continue;
        }
        codes[linear] = pack_code8_uint64(labels, k_values, row, token, max_tokens);
        token_ids[linear] = token;
    }
}

__global__ void mark_cluster_starts_kernel(
    const uint64_t* __restrict__ sorted_codes,
    const int32_t* __restrict__ row_token_counts,
    int32_t* __restrict__ start_flags,
    int rows,
    int max_tokens
) {
    const int64_t total = static_cast<int64_t>(rows) * max_tokens;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        const int token = static_cast<int>(linear % max_tokens);
        const int row = static_cast<int>(linear / max_tokens);
        if (token >= row_token_counts[row]) {
            start_flags[linear] = 0;
            continue;
        }
        const uint64_t code = sorted_codes[linear];
        if (code == UINT64_MAX) {
            start_flags[linear] = 0;
            continue;
        }
        start_flags[linear] = (token == 0 || code != sorted_codes[linear - 1]) ? 1 : 0;
    }
}

__global__ void fill_sorted_compact_metadata_kernel(
    const uint64_t* __restrict__ sorted_codes,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ start_flags,
    const int32_t* __restrict__ cluster_scan,
    const int32_t* __restrict__ row_token_counts,
    int32_t* __restrict__ token_ids_by_cluster,
    int32_t* __restrict__ cluster_offsets,
    int32_t* __restrict__ cluster_counts,
    int rows,
    int max_tokens,
    int max_clusters
) {
    const int64_t total = static_cast<int64_t>(rows) * max_tokens;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        const int pos = static_cast<int>(linear % max_tokens);
        const int row = static_cast<int>(linear / max_tokens);
        const int token_count = row_token_counts[row];
        if (pos >= token_count) {
            continue;
        }
        const uint64_t code = sorted_codes[linear];
        if (code == UINT64_MAX) {
            continue;
        }
        token_ids_by_cluster[linear] = sorted_token_ids[linear];
        const int is_start = start_flags[linear];
        const int compact = cluster_scan[linear];
        if (is_start != 0) {
            if (compact < max_clusters) {
                cluster_offsets[row * (max_clusters + 1) + compact] = pos;
            }
        }
        if (pos == token_count - 1) {
            const int active_count = compact + is_start;
            cluster_counts[row] = active_count;
            const int tail = active_count < max_clusters ? active_count : max_clusters;
            cluster_offsets[row * (max_clusters + 1) + tail] = token_count;
        }
    }
}

__global__ void initialize_compact_outputs_kernel(
    int32_t* __restrict__ cluster_sizes,
    int32_t* __restrict__ cluster_offsets,
    int32_t* __restrict__ cluster_counts,
    float* __restrict__ full_centroids,
    int rows,
    int max_clusters,
    int head_dim
) {
    const int64_t cluster_total = static_cast<int64_t>(rows) * max_clusters;
    const int64_t offset_total = static_cast<int64_t>(rows) * (max_clusters + 1);
    const int64_t centroid_total = cluster_total * head_dim;
    int64_t total = cluster_total > centroid_total ? cluster_total : centroid_total;
    total = total > offset_total ? total : offset_total;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        if (i < cluster_total) {
            cluster_sizes[i] = 0;
        }
        if (i < offset_total) {
            cluster_offsets[i] = 0;
        }
        if (i < centroid_total) {
            full_centroids[i] = 0.0f;
        }
        if (i < rows) {
            cluster_counts[i] = 0;
        }
    }
}

__global__ void finalize_sorted_clusters_kernel(
    const int32_t* __restrict__ cluster_counts,
    const int32_t* __restrict__ row_token_counts,
    int32_t* __restrict__ cluster_offsets,
    int32_t* __restrict__ cluster_sizes,
    int rows,
    int max_clusters
) {
    const int64_t total = static_cast<int64_t>(rows) * (max_clusters + 1);
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        const int pos = static_cast<int>(linear % (max_clusters + 1));
        const int row = static_cast<int>(linear / (max_clusters + 1));
        const int count = cluster_counts[row];
        if (pos > count) {
            cluster_offsets[linear] = row_token_counts[row];
        }
        if (pos < max_clusters) {
            if (pos < count) {
                const int32_t start = cluster_offsets[row * (max_clusters + 1) + pos];
                const int32_t end = cluster_offsets[row * (max_clusters + 1) + pos + 1];
                cluster_sizes[row * max_clusters + pos] = end - start;
            } else {
                cluster_sizes[row * max_clusters + pos] = 0;
            }
        }
    }
}

template <typename scalar_t>
__global__ void compute_true_full_centroids_kernel(
    const scalar_t* __restrict__ keys,
    const int32_t* __restrict__ token_ids,
    const int32_t* __restrict__ cluster_offsets,
    const int32_t* __restrict__ cluster_counts,
    const int32_t* __restrict__ cluster_sizes,
    float* __restrict__ full_centroids,
    int rows,
    int kv_heads,
    int max_tokens,
    int max_clusters,
    int head_dim,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_t,
    int64_t key_stride_d
) {
    const int64_t total = static_cast<int64_t>(rows) * max_clusters * head_dim;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        int tmp = static_cast<int>(linear);
        const int dim = tmp % head_dim;
        tmp /= head_dim;
        const int cluster = tmp % max_clusters;
        const int row = tmp / max_clusters;
        if (cluster >= cluster_counts[row]) {
            continue;
        }

        const int offset_base = row * (max_clusters + 1) + cluster;
        const int start = cluster_offsets[offset_base];
        const int size = cluster_sizes[row * max_clusters + cluster];
        if (size <= 0) {
            continue;
        }

        const int b = row / kv_heads;
        const int h = row - b * kv_heads;
        float sum = 0.0f;
        const int token_base = row * max_tokens;
        for (int i = 0; i < size; ++i) {
            const int token = token_ids[token_base + start + i];
            if (token >= 0 && token < max_tokens) {
                sum += load_key_value(keys, b, h, token, dim, key_stride_b, key_stride_h, key_stride_t, key_stride_d);
            }
        }
        full_centroids[(row * max_clusters + cluster) * head_dim + dim] = sum / static_cast<float>(size);
    }
}

inline void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

inline void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    check_cuda_tensor(tensor, name);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

} // namespace

std::vector<torch::Tensor> run_hybrid_8way_kmeans_cuda(
    const torch::Tensor& keys,
    const torch::Tensor& row_token_counts,
    const torch::Tensor& subspace_dim_ids,
    const torch::Tensor& subspace_dim_masks,
    const torch::Tensor& k_values,
    int64_t max_iters,
    int64_t max_k_arg,
    int64_t max_compact_clusters_arg
) {
    check_cuda_tensor(keys, "keys");
    check_cuda_contiguous(row_token_counts, "row_token_counts");
    check_cuda_contiguous(subspace_dim_ids, "subspace_dim_ids");
    check_cuda_contiguous(subspace_dim_masks, "subspace_dim_masks");
    check_cuda_contiguous(k_values, "k_values");

    TORCH_CHECK(keys.dim() == 4, "keys must be [batch, kv_heads, max_tokens, head_dim]");
    TORCH_CHECK(row_token_counts.scalar_type() == torch::kInt32, "row_token_counts must be int32");
    TORCH_CHECK(subspace_dim_ids.scalar_type() == torch::kInt64, "subspace_dim_ids must be int64");
    TORCH_CHECK(subspace_dim_masks.scalar_type() == torch::kBool, "subspace_dim_masks must be bool");
    TORCH_CHECK(k_values.scalar_type() == torch::kInt32, "k_values must be int32");
    TORCH_CHECK(k_values.numel() == kSubspaces, "k_values must contain 8 values");

    const auto batch_size = keys.size(0);
    const auto kv_heads = keys.size(1);
    const auto max_tokens = keys.size(2);
    const auto head_dim = keys.size(3);
    const int64_t key_stride_b = keys.stride(0);
    const int64_t key_stride_h = keys.stride(1);
    const int64_t key_stride_t = keys.stride(2);
    const int64_t key_stride_d = keys.stride(3);
    const int rows = static_cast<int>(batch_size * kv_heads);
    TORCH_CHECK(row_token_counts.numel() == rows, "row_token_counts rows mismatch");
    TORCH_CHECK(subspace_dim_ids.size(0) == rows, "subspace_dim_ids rows mismatch");
    TORCH_CHECK(subspace_dim_ids.size(1) == kSubspaces, "subspace_dim_ids must have 8 subspaces");
    TORCH_CHECK(subspace_dim_masks.sizes() == subspace_dim_ids.sizes(), "subspace_dim_masks shape mismatch");

    const int max_sub_dim = static_cast<int>(subspace_dim_ids.size(2));
    TORCH_CHECK(max_k_arg > 0 && max_k_arg <= std::numeric_limits<int>::max(), "max_k out of range");
    const int max_k = static_cast<int>(max_k_arg);
    const int iters = static_cast<int>(std::max<int64_t>(1, max_iters));
    TORCH_CHECK(
        max_compact_clusters_arg > 0 && max_compact_clusters_arg <= max_tokens,
        "max_compact_clusters must be in [1, max_tokens]"
    );
    const int64_t max_compact_clusters = max_compact_clusters_arg;

    const c10::cuda::CUDAGuard device_guard(keys.device());
    int current_device = -1;
    int sm_count = 1;
    cudaGetDevice(&current_device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, current_device);
    const int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t assign_total = static_cast<int64_t>(rows) * kSubspaces * max_tokens;
    const int assign_blocks = capped_blocks(assign_total, threads, sm_count);
    const int tile_tokens = 512;
    const int tiles_per_task = static_cast<int>((max_tokens + tile_tokens - 1) / tile_tokens);
    const bool use_tiled_reduce = max_k <= 32 && max_sub_dim <= 64;
    const bool use_warp_private_reduce = use_tiled_reduce && max_k <= 16 && max_sub_dim <= 32;
    const int64_t partial_tasks = static_cast<int64_t>(rows) * kSubspaces * tiles_per_task;

    auto labels = torch::zeros({rows, kSubspaces, max_tokens}, keys.options().dtype(torch::kInt32));
    auto centroids = torch::zeros({rows, kSubspaces, max_k, max_sub_dim}, keys.options().dtype(torch::kFloat32));
    auto centroid_masks = torch::ones({rows, kSubspaces, max_k}, keys.options().dtype(torch::kBool));
    auto sums = torch::zeros_like(centroids);
    auto counts = torch::zeros({rows, kSubspaces, max_k}, keys.options().dtype(torch::kFloat32));
    auto token_ids = torch::empty({rows, max_tokens}, keys.options().dtype(torch::kInt32));
    auto cluster_sizes = torch::empty({rows, max_compact_clusters}, keys.options().dtype(torch::kInt32));
    auto cluster_offsets = torch::empty({rows, max_compact_clusters + 1}, keys.options().dtype(torch::kInt32));
    auto cluster_counts = torch::empty({rows}, keys.options().dtype(torch::kInt32));
    auto full_centroids = torch::empty({rows, max_compact_clusters, head_dim}, keys.options().dtype(torch::kFloat32));
    auto code_keys_in = torch::empty({rows, max_tokens}, keys.options().dtype(torch::kInt64));
    auto code_keys_sorted = torch::empty({rows, max_tokens}, keys.options().dtype(torch::kInt64));
    auto token_ids_in = torch::empty({rows, max_tokens}, keys.options().dtype(torch::kInt32));
    auto start_flags = torch::empty({rows, max_tokens}, keys.options().dtype(torch::kInt32));
    auto cluster_scan = torch::empty({rows, max_tokens}, keys.options().dtype(torch::kInt32));
    auto segment_offsets = torch::empty({rows + 1}, keys.options().dtype(torch::kInt64));

    const int64_t init_tasks = static_cast<int64_t>(rows) * kSubspaces;
    const int init_blocks = static_cast<int>(init_tasks);
    const int64_t update_total = static_cast<int64_t>(rows) * kSubspaces * max_k * max_sub_dim;
    const int update_blocks = capped_blocks(update_total, threads, sm_count);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        keys.scalar_type(),
        "hybrid_8way_init_centroids",
        [&] {
            init_centroids_sampled_kernel<scalar_t><<<init_blocks, threads, 0, stream>>>(
                keys.data_ptr<scalar_t>(),
                row_token_counts.data_ptr<int32_t>(),
                subspace_dim_ids.data_ptr<int64_t>(),
                subspace_dim_masks.data_ptr<bool>(),
                k_values.data_ptr<int32_t>(),
                centroids.data_ptr<float>(),
                centroid_masks.data_ptr<bool>(),
                rows,
                static_cast<int>(kv_heads),
                static_cast<int>(max_tokens),
                static_cast<int>(head_dim),
                key_stride_b,
                key_stride_h,
                key_stride_t,
                key_stride_d,
                max_k,
                max_sub_dim
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    torch::Tensor partial_sums;
    torch::Tensor partial_counts;
    if (use_tiled_reduce) {
        TORCH_CHECK(
            partial_tasks <= std::numeric_limits<unsigned int>::max(),
            "hybrid 8-way tiled reduce grid is too large"
        );
        partial_sums = torch::empty(
            {partial_tasks, max_k, max_sub_dim},
            keys.options().dtype(torch::kFloat32)
        );
        partial_counts = torch::empty(
            {partial_tasks, max_k},
            keys.options().dtype(torch::kFloat32)
        );
    }

    for (int iter = 0; iter < iters; ++iter) {
        if (use_tiled_reduce) {
            cudaMemsetAsync(counts.data_ptr<float>(), 0, counts.numel() * sizeof(float), stream);

            const int warp_count = (threads + 31) / 32;
            const size_t tiled_shared_bytes = use_warp_private_reduce
                ? ((static_cast<size_t>(warp_count) * static_cast<size_t>(max_k) * static_cast<size_t>(max_sub_dim) +
                    static_cast<size_t>(warp_count) * static_cast<size_t>(max_k) +
                    static_cast<size_t>(max_k) * static_cast<size_t>(max_sub_dim)) * sizeof(float) +
                   static_cast<size_t>(max_sub_dim) * sizeof(int))
                : ((static_cast<size_t>(2) * static_cast<size_t>(max_k) * static_cast<size_t>(max_sub_dim) +
                    static_cast<size_t>(max_k)) * sizeof(float) +
                   static_cast<size_t>(max_sub_dim) * sizeof(int));
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                keys.scalar_type(),
                "hybrid_8way_assign_reduce_tiled",
                [&] {
                    if (use_warp_private_reduce) {
                        assign_reduce_tiled_warp_private_kernel<scalar_t><<<static_cast<unsigned int>(partial_tasks), threads, tiled_shared_bytes, stream>>>(
                            keys.data_ptr<scalar_t>(),
                            row_token_counts.data_ptr<int32_t>(),
                            subspace_dim_ids.data_ptr<int64_t>(),
                            subspace_dim_masks.data_ptr<bool>(),
                            k_values.data_ptr<int32_t>(),
                            centroids.data_ptr<float>(),
                            labels.data_ptr<int32_t>(),
                            partial_sums.data_ptr<float>(),
                            partial_counts.data_ptr<float>(),
                            rows,
                            static_cast<int>(kv_heads),
                            static_cast<int>(max_tokens),
                            static_cast<int>(head_dim),
                            key_stride_b,
                            key_stride_h,
                            key_stride_t,
                            key_stride_d,
                            max_k,
                            max_sub_dim,
                            tile_tokens,
                            tiles_per_task
                        );
                    } else {
                        assign_reduce_tiled_kernel<scalar_t><<<static_cast<unsigned int>(partial_tasks), threads, tiled_shared_bytes, stream>>>(
                            keys.data_ptr<scalar_t>(),
                            row_token_counts.data_ptr<int32_t>(),
                            subspace_dim_ids.data_ptr<int64_t>(),
                            subspace_dim_masks.data_ptr<bool>(),
                            k_values.data_ptr<int32_t>(),
                            centroids.data_ptr<float>(),
                            labels.data_ptr<int32_t>(),
                            partial_sums.data_ptr<float>(),
                            partial_counts.data_ptr<float>(),
                            rows,
                            static_cast<int>(kv_heads),
                            static_cast<int>(max_tokens),
                            static_cast<int>(head_dim),
                            key_stride_b,
                            key_stride_h,
                            key_stride_t,
                            key_stride_d,
                            max_k,
                            max_sub_dim,
                            tile_tokens,
                            tiles_per_task
                        );
                    }
                }
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                keys.scalar_type(),
                "hybrid_8way_reduce_update_tiled",
                [&] {
                    reduce_update_tiled_kernel<scalar_t><<<update_blocks, threads, 0, stream>>>(
                        keys.data_ptr<scalar_t>(),
                        row_token_counts.data_ptr<int32_t>(),
                        subspace_dim_ids.data_ptr<int64_t>(),
                        subspace_dim_masks.data_ptr<bool>(),
                        k_values.data_ptr<int32_t>(),
                        partial_sums.data_ptr<float>(),
                        partial_counts.data_ptr<float>(),
                        centroids.data_ptr<float>(),
                        centroid_masks.data_ptr<bool>(),
                        counts.data_ptr<float>(),
                        rows,
                        static_cast<int>(kv_heads),
                        static_cast<int>(max_tokens),
                        static_cast<int>(head_dim),
                        key_stride_b,
                        key_stride_h,
                        key_stride_t,
                        key_stride_d,
                        max_k,
                        max_sub_dim,
                        tiles_per_task,
                        iter
                    );
                }
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            cudaMemsetAsync(sums.data_ptr<float>(), 0, sums.numel() * sizeof(float), stream);
            cudaMemsetAsync(counts.data_ptr<float>(), 0, counts.numel() * sizeof(float), stream);

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                keys.scalar_type(),
                "hybrid_8way_assign_reduce",
                [&] {
                    assign_reduce_kernel<scalar_t><<<assign_blocks, threads, 0, stream>>>(
                        keys.data_ptr<scalar_t>(),
                        row_token_counts.data_ptr<int32_t>(),
                        subspace_dim_ids.data_ptr<int64_t>(),
                        subspace_dim_masks.data_ptr<bool>(),
                        k_values.data_ptr<int32_t>(),
                        centroids.data_ptr<float>(),
                        labels.data_ptr<int32_t>(),
                        sums.data_ptr<float>(),
                        counts.data_ptr<float>(),
                        rows,
                        static_cast<int>(kv_heads),
                        static_cast<int>(max_tokens),
                        static_cast<int>(head_dim),
                        key_stride_b,
                        key_stride_h,
                        key_stride_t,
                        key_stride_d,
                        max_k,
                        max_sub_dim
                    );
                }
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                keys.scalar_type(),
                "hybrid_8way_update_centroids",
                [&] {
                    update_centroids_kernel<scalar_t><<<update_blocks, threads, 0, stream>>>(
                        keys.data_ptr<scalar_t>(),
                        row_token_counts.data_ptr<int32_t>(),
                        subspace_dim_ids.data_ptr<int64_t>(),
                        subspace_dim_masks.data_ptr<bool>(),
                        k_values.data_ptr<int32_t>(),
                        sums.data_ptr<float>(),
                        counts.data_ptr<float>(),
                        centroids.data_ptr<float>(),
                        centroid_masks.data_ptr<bool>(),
                        rows,
                        static_cast<int>(kv_heads),
                        static_cast<int>(max_tokens),
                        static_cast<int>(head_dim),
                        key_stride_b,
                        key_stride_h,
                        key_stride_t,
                        key_stride_d,
                        max_k,
                        max_sub_dim,
                        iter
                    );
                }
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    const int64_t compact_total = static_cast<int64_t>(rows) * max_tokens;
    const int compact_blocks = capped_blocks(compact_total, threads, sm_count);
    const int64_t init_compact_total = std::max<int64_t>(
        static_cast<int64_t>(rows) * max_compact_clusters * head_dim,
        std::max<int64_t>(compact_total, static_cast<int64_t>(rows) * (max_compact_clusters + 1))
    );
    const int init_compact_blocks = capped_blocks(init_compact_total, threads, sm_count);
    initialize_compact_outputs_kernel<<<init_compact_blocks, threads, 0, stream>>>(
        cluster_sizes.data_ptr<int32_t>(),
        cluster_offsets.data_ptr<int32_t>(),
        cluster_counts.data_ptr<int32_t>(),
        full_centroids.data_ptr<float>(),
        rows,
        static_cast<int>(max_compact_clusters),
        static_cast<int>(head_dim)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    init_segment_offsets_kernel<<<capped_blocks(rows + 1, threads, sm_count), threads, 0, stream>>>(
        segment_offsets.data_ptr<int64_t>(),
        rows,
        static_cast<int>(max_tokens)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    pack_sorted_code_inputs_kernel<<<compact_blocks, threads, 0, stream>>>(
        labels.data_ptr<int32_t>(),
        row_token_counts.data_ptr<int32_t>(),
        k_values.data_ptr<int32_t>(),
        reinterpret_cast<uint64_t*>(code_keys_in.data_ptr<int64_t>()),
        token_ids_in.data_ptr<int32_t>(),
        rows,
        static_cast<int>(max_tokens)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t sort_items = compact_total;
    TORCH_CHECK(sort_items <= std::numeric_limits<int>::max(), "hybrid 8-way sort item count exceeds CUB int limit");
    void* sort_temp_ptr = nullptr;
    size_t sort_temp_bytes = 0;
    cudaError_t cub_status = cub::DeviceSegmentedRadixSort::SortPairs(
        sort_temp_ptr,
        sort_temp_bytes,
        reinterpret_cast<uint64_t*>(code_keys_in.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(code_keys_sorted.data_ptr<int64_t>()),
        token_ids_in.data_ptr<int32_t>(),
        token_ids.data_ptr<int32_t>(),
        static_cast<int>(sort_items),
        rows,
        segment_offsets.data_ptr<int64_t>(),
        segment_offsets.data_ptr<int64_t>() + 1,
        0,
        64,
        stream
    );
    TORCH_CHECK(cub_status == cudaSuccess, "CUB segmented radix sort size query failed: ", cudaGetErrorString(cub_status));
    auto sort_temp = torch::empty({static_cast<int64_t>(sort_temp_bytes)}, keys.options().dtype(torch::kUInt8));
    sort_temp_ptr = sort_temp.data_ptr();
    cub_status = cub::DeviceSegmentedRadixSort::SortPairs(
        sort_temp_ptr,
        sort_temp_bytes,
        reinterpret_cast<uint64_t*>(code_keys_in.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(code_keys_sorted.data_ptr<int64_t>()),
        token_ids_in.data_ptr<int32_t>(),
        token_ids.data_ptr<int32_t>(),
        static_cast<int>(sort_items),
        rows,
        segment_offsets.data_ptr<int64_t>(),
        segment_offsets.data_ptr<int64_t>() + 1,
        0,
        64,
        stream
    );
    TORCH_CHECK(cub_status == cudaSuccess, "CUB segmented radix sort failed: ", cudaGetErrorString(cub_status));

    mark_cluster_starts_kernel<<<compact_blocks, threads, 0, stream>>>(
        reinterpret_cast<uint64_t*>(code_keys_sorted.data_ptr<int64_t>()),
        row_token_counts.data_ptr<int32_t>(),
        start_flags.data_ptr<int32_t>(),
        rows,
        static_cast<int>(max_tokens)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto scan_policy = thrust::cuda::par.on(stream);
    for (int row = 0; row < rows; ++row) {
        auto flags_begin = thrust::device_pointer_cast(
            start_flags.data_ptr<int32_t>() + static_cast<int64_t>(row) * max_tokens
        );
        auto scan_begin = thrust::device_pointer_cast(
            cluster_scan.data_ptr<int32_t>() + static_cast<int64_t>(row) * max_tokens
        );
        thrust::exclusive_scan(scan_policy, flags_begin, flags_begin + max_tokens, scan_begin);
    }

    fill_sorted_compact_metadata_kernel<<<compact_blocks, threads, 0, stream>>>(
        reinterpret_cast<uint64_t*>(code_keys_sorted.data_ptr<int64_t>()),
        token_ids.data_ptr<int32_t>(),
        start_flags.data_ptr<int32_t>(),
        cluster_scan.data_ptr<int32_t>(),
        row_token_counts.data_ptr<int32_t>(),
        token_ids.data_ptr<int32_t>(),
        cluster_offsets.data_ptr<int32_t>(),
        cluster_counts.data_ptr<int32_t>(),
        rows,
        static_cast<int>(max_tokens),
        static_cast<int>(max_compact_clusters)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    finalize_sorted_clusters_kernel<<<capped_blocks(static_cast<int64_t>(rows) * (max_compact_clusters + 1), threads, sm_count), threads, 0, stream>>>(
        cluster_counts.data_ptr<int32_t>(),
        row_token_counts.data_ptr<int32_t>(),
        cluster_offsets.data_ptr<int32_t>(),
        cluster_sizes.data_ptr<int32_t>(),
        rows,
        static_cast<int>(max_compact_clusters)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t true_mean_total = static_cast<int64_t>(rows) * max_compact_clusters * head_dim;
    const int true_mean_blocks = capped_blocks(true_mean_total, threads, sm_count, 4);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        keys.scalar_type(),
        "hybrid_8way_true_full_centroids",
        [&] {
            compute_true_full_centroids_kernel<scalar_t><<<true_mean_blocks, threads, 0, stream>>>(
                keys.data_ptr<scalar_t>(),
                token_ids.data_ptr<int32_t>(),
                cluster_offsets.data_ptr<int32_t>(),
                cluster_counts.data_ptr<int32_t>(),
                cluster_sizes.data_ptr<int32_t>(),
                full_centroids.data_ptr<float>(),
                rows,
                static_cast<int>(kv_heads),
                static_cast<int>(max_tokens),
                static_cast<int>(max_compact_clusters),
                static_cast<int>(head_dim),
                key_stride_b,
                key_stride_h,
                key_stride_t,
                key_stride_d
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        token_ids,
        cluster_sizes,
        cluster_offsets,
        cluster_counts,
        full_centroids,
        centroids,
    };
}

void materialize_hybrid_decode_metadata_cuda(
    const torch::Tensor& centroids_compact,
    const torch::Tensor& cluster_sizes_compact,
    const torch::Tensor& cluster_counts,
    const torch::Tensor& centroids_full,
    const torch::Tensor& cluster_sizes_full,
    const torch::Tensor& centroids_mask_full
) {
    TORCH_CHECK(centroids_compact.is_cuda(), "centroids_compact must be a CUDA tensor");
    TORCH_CHECK(cluster_sizes_compact.is_cuda(), "cluster_sizes_compact must be a CUDA tensor");
    TORCH_CHECK(cluster_counts.is_cuda(), "cluster_counts must be a CUDA tensor");
    TORCH_CHECK(centroids_full.is_cuda(), "centroids_full must be a CUDA tensor");
    TORCH_CHECK(cluster_sizes_full.is_cuda(), "cluster_sizes_full must be a CUDA tensor");
    TORCH_CHECK(centroids_mask_full.is_cuda(), "centroids_mask_full must be a CUDA tensor");
    TORCH_CHECK(centroids_compact.is_contiguous(), "centroids_compact must be contiguous");
    TORCH_CHECK(cluster_sizes_compact.is_contiguous(), "cluster_sizes_compact must be contiguous");
    TORCH_CHECK(cluster_counts.is_contiguous(), "cluster_counts must be contiguous");
    TORCH_CHECK(centroids_full.is_contiguous(), "centroids_full must be contiguous");
    TORCH_CHECK(cluster_sizes_full.is_contiguous(), "cluster_sizes_full must be contiguous");
    TORCH_CHECK(centroids_mask_full.is_contiguous(), "centroids_mask_full must be contiguous");
    TORCH_CHECK(centroids_compact.dim() == 3, "centroids_compact must be [rows, compact_clusters, head_dim]");
    TORCH_CHECK(cluster_sizes_compact.dim() == 2, "cluster_sizes_compact must be [rows, compact_clusters]");
    TORCH_CHECK(cluster_counts.dim() == 1, "cluster_counts must be [rows]");
    TORCH_CHECK(centroids_full.dim() == 3, "centroids_full must be [rows, n_centroids, head_dim]");
    TORCH_CHECK(cluster_sizes_full.dim() == 2, "cluster_sizes_full must be [rows, n_centroids]");
    TORCH_CHECK(centroids_mask_full.dim() == 2, "centroids_mask_full must be [rows, n_centroids]");
    TORCH_CHECK(centroids_compact.scalar_type() == torch::kFloat32, "centroids_compact must be float32");
    TORCH_CHECK(cluster_sizes_compact.scalar_type() == torch::kInt32, "cluster_sizes_compact must be int32");
    TORCH_CHECK(cluster_counts.scalar_type() == torch::kInt32, "cluster_counts must be int32");
    TORCH_CHECK(centroids_full.scalar_type() == torch::kFloat32, "centroids_full must be float32");
    TORCH_CHECK(cluster_sizes_full.scalar_type() == torch::kInt32, "cluster_sizes_full must be int32");
    TORCH_CHECK(centroids_mask_full.scalar_type() == torch::kBool, "centroids_mask_full must be bool");

    const int64_t rows = centroids_compact.size(0);
    const int64_t compact_clusters = centroids_compact.size(1);
    const int64_t head_dim = centroids_compact.size(2);
    const int64_t n_centroids = centroids_full.size(1);
    TORCH_CHECK(cluster_sizes_compact.size(0) == rows, "cluster_sizes_compact row mismatch");
    TORCH_CHECK(cluster_sizes_compact.size(1) == compact_clusters, "cluster_sizes_compact cluster mismatch");
    TORCH_CHECK(cluster_counts.size(0) == rows, "cluster_counts row mismatch");
    TORCH_CHECK(centroids_full.size(0) == rows, "centroids_full row mismatch");
    TORCH_CHECK(centroids_full.size(2) == head_dim, "centroids_full head_dim mismatch");
    TORCH_CHECK(cluster_sizes_full.size(0) == rows && cluster_sizes_full.size(1) == n_centroids, "cluster_sizes_full shape mismatch");
    TORCH_CHECK(centroids_mask_full.size(0) == rows && centroids_mask_full.size(1) == n_centroids, "centroids_mask_full shape mismatch");
    TORCH_CHECK(cluster_counts.numel() == rows, "cluster_counts numel mismatch");
    TORCH_CHECK(n_centroids >= compact_clusters, "n_centroids must be >= compact_clusters");

    const auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const int64_t total_centroids = rows * compact_clusters * head_dim;
    const int64_t total_metadata = rows * n_centroids;
    const int64_t total = total_centroids > total_metadata ? total_centroids : total_metadata;
    const int blocks = capped_blocks(total, threads, at::cuda::getCurrentDeviceProperties()->multiProcessorCount, 4);

    materialize_hybrid_decode_metadata_kernel<<<blocks, threads, 0, stream>>>(
        centroids_compact.data_ptr<float>(),
        cluster_sizes_compact.data_ptr<int32_t>(),
        cluster_counts.data_ptr<int32_t>(),
        centroids_full.data_ptr<float>(),
        cluster_sizes_full.data_ptr<int32_t>(),
        centroids_mask_full.data_ptr<bool>(),
        rows,
        compact_clusters,
        n_centroids,
        head_dim
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace imi_hybrid
