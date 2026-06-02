#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#define CHUNK_SIZE 8        // number of vectors for one chunk
#define DATA_BYTES 2        // input data type, 2 use bf16/fp16, 4 use fp32
#define PTYPE int2          // int4 for 16 Bytes, int2 for 8 Bytes

#if DATA_BYTES == 4
#define MYTPE uint32_t
#else
#define MYTPE uint16_t
#endif

#include "copy_kernel.cuh"

constexpr int IMI_HASH_EMPTY = -1;
constexpr int IMI_HASH_TOMBSTONE = -2;
constexpr int IMI_HASH_PENDING = -3;

__device__ __forceinline__ uint32_t imi_hash_i64(int64_t key) {
    uint64_t x = static_cast<uint64_t>(key);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<uint32_t>(x);
}

__device__ __forceinline__ int imi_hash_lookup(
    const int64_t* __restrict__ keys,
    const int32_t* __restrict__ values,
    int hash_size,
    int64_t key
) {
    int pos = static_cast<int>(imi_hash_i64(key) & static_cast<uint32_t>(hash_size - 1));
    for (int i = 0; i < hash_size; ++i) {
        int64_t cur = keys[pos];
        if (cur == key) {
            int value = atomicAdd(const_cast<int32_t*>(values) + pos, 0);
            return value >= 0 ? value : -1;
        }
        if (cur == IMI_HASH_EMPTY) {
            return -1;
        }
        pos = (pos + 1) & (hash_size - 1);
    }
    return -1;
}

__device__ __forceinline__ void imi_hash_erase(
    int64_t* __restrict__ keys,
    int32_t* __restrict__ values,
    int hash_size,
    int64_t key
) {
    if (key < 0) {
        return;
    }
    int pos = static_cast<int>(imi_hash_i64(key) & static_cast<uint32_t>(hash_size - 1));
    for (int i = 0; i < hash_size; ++i) {
        int64_t cur = keys[pos];
        if (cur == key) {
            keys[pos] = IMI_HASH_TOMBSTONE;
            values[pos] = IMI_HASH_EMPTY;
            return;
        }
        if (cur == IMI_HASH_EMPTY) {
            return;
        }
        pos = (pos + 1) & (hash_size - 1);
    }
}

__device__ __forceinline__ void imi_hash_insert(
    int64_t* __restrict__ keys,
    int32_t* __restrict__ values,
    int hash_size,
    int64_t key,
    int32_t value
) {
    int pos = static_cast<int>(imi_hash_i64(key) & static_cast<uint32_t>(hash_size - 1));
    int first_tombstone = -1;
    for (int i = 0; i < hash_size; ++i) {
        int64_t cur = keys[pos];
        if (cur == key) {
            values[pos] = value;
            return;
        }
        if (cur == IMI_HASH_TOMBSTONE && first_tombstone < 0) {
            first_tombstone = pos;
        } else if (cur == IMI_HASH_EMPTY) {
            int dst = first_tombstone >= 0 ? first_tombstone : pos;
            keys[dst] = key;
            values[dst] = value;
            return;
        }
        pos = (pos + 1) & (hash_size - 1);
    }
    if (first_tombstone >= 0) {
        keys[first_tombstone] = key;
        values[first_tombstone] = value;
    }
}

__device__ __forceinline__ int imi_hash_claim_pending(
    int64_t* __restrict__ keys,
    int32_t* __restrict__ values,
    int hash_size,
    int64_t key
) {
    int pos = static_cast<int>(imi_hash_i64(key) & static_cast<uint32_t>(hash_size - 1));
    int first_tombstone = -1;
    for (int i = 0; i < hash_size; ++i) {
        int64_t cur = keys[pos];
        if (cur == key) {
            return -2;
        }
        if (cur == IMI_HASH_TOMBSTONE && first_tombstone < 0) {
            first_tombstone = pos;
        } else if (cur == IMI_HASH_EMPTY) {
            int dst = first_tombstone >= 0 ? first_tombstone : pos;
            int64_t expected = first_tombstone >= 0 ? static_cast<int64_t>(IMI_HASH_TOMBSTONE)
                                                   : static_cast<int64_t>(IMI_HASH_EMPTY);
            unsigned long long old = atomicCAS(
                reinterpret_cast<unsigned long long*>(keys + dst),
                static_cast<unsigned long long>(expected),
                static_cast<unsigned long long>(key)
            );
            if (static_cast<int64_t>(old) == expected) {
                atomicExch(values + dst, IMI_HASH_PENDING);
                return dst;
            }
            pos = static_cast<int>(imi_hash_i64(key) & static_cast<uint32_t>(hash_size - 1));
            first_tombstone = -1;
            continue;
        }
        pos = (pos + 1) & (hash_size - 1);
    }
    if (first_tombstone >= 0) {
        unsigned long long old = atomicCAS(
            reinterpret_cast<unsigned long long*>(keys + first_tombstone),
            static_cast<unsigned long long>(static_cast<int64_t>(IMI_HASH_TOMBSTONE)),
            static_cast<unsigned long long>(key)
        );
        if (static_cast<int64_t>(old) == static_cast<int64_t>(IMI_HASH_TOMBSTONE)) {
            atomicExch(values + first_tombstone, IMI_HASH_PENDING);
            return first_tombstone;
        }
    }
    return -1;
}

__global__ void build_cache_aware_retrieval_plan_kernel(
    const int64_t* __restrict__ selected_clusters,
    const int32_t* __restrict__ cluster_offsets,
    const int32_t* __restrict__ cluster_sizes,
    const int64_t* __restrict__ hash_keys,
    const int32_t* __restrict__ hash_values,
    int32_t* __restrict__ miss_offsets,
    int32_t* __restrict__ miss_sizes,
    int32_t* __restrict__ miss_cumsum,
    int32_t* __restrict__ miss_counts,
    int64_t* __restrict__ miss_page_keys,
    int32_t* __restrict__ hit_offsets,
    int32_t* __restrict__ hit_sizes,
    int32_t* __restrict__ hit_cumsum,
    int32_t* __restrict__ hit_counts,
    int32_t* __restrict__ valid_lengths,
    int groups,
    int nprobe,
    int n_clusters,
    int buffer_blocks,
    int block_size,
    int hash_size,
    int64_t padding_cluster_id
) {
    int row = blockIdx.x;
    if (row >= groups) {
        return;
    }

    const int64_t* row_selected = selected_clusters + static_cast<int64_t>(row) * nprobe;
    const int32_t* row_offsets = cluster_offsets + static_cast<int64_t>(row) * (n_clusters + 1);
    const int32_t* row_sizes = cluster_sizes + static_cast<int64_t>(row) * n_clusters;
    const int64_t* row_hash_keys = hash_keys + static_cast<int64_t>(row) * hash_size;
    const int32_t* row_hash_values = hash_values + static_cast<int64_t>(row) * hash_size;

    int32_t* row_miss_offsets = miss_offsets + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_miss_sizes = miss_sizes + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_miss_cumsum = miss_cumsum + static_cast<int64_t>(row) * buffer_blocks;
    int64_t* row_miss_page_keys = miss_page_keys + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_hit_offsets = hit_offsets + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_hit_sizes = hit_sizes + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_hit_cumsum = hit_cumsum + static_cast<int64_t>(row) * buffer_blocks;

    __shared__ unsigned long long s_miss_state;
    __shared__ unsigned long long s_hit_state;
    __shared__ int s_considered_blocks;
    if (threadIdx.x == 0) {
        s_miss_state = 0;
        s_hit_state = 0;
        s_considered_blocks = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < nprobe; i += blockDim.x) {
        int cluster = static_cast<int>(row_selected[i]);
        if (cluster < 0 || cluster >= n_clusters || row_selected[i] == padding_cluster_id) {
            continue;
        }
        int cluster_size = row_sizes[cluster];
        if (cluster_size <= 0) {
            continue;
        }
        int cluster_blocks = (cluster_size + block_size - 1) / block_size;
        if (cluster_blocks <= 0) {
            continue;
        }
        int prior_blocks = atomicAdd(&s_considered_blocks, cluster_blocks);
        if (prior_blocks + cluster_blocks > buffer_blocks) {
            atomicSub(&s_considered_blocks, cluster_blocks);
            continue;
        }

        int list_start = row_offsets[cluster];
        int local_miss_blocks = 0;
        int local_hit_blocks = 0;
        int local_miss_tokens = 0;
        int local_hit_tokens = 0;
        for (int page = 0; page < cluster_blocks; ++page) {
            int size = (page + 1 == cluster_blocks) ? (cluster_size - page * block_size) : block_size;
            int64_t key = (static_cast<int64_t>(cluster) << 32) | static_cast<uint32_t>(page);
            int slot = imi_hash_lookup(row_hash_keys, row_hash_values, hash_size, key);
            if (slot >= 0) {
                local_hit_blocks++;
                local_hit_tokens += size;
            } else {
                local_miss_blocks++;
                local_miss_tokens += size;
            }
        }

        unsigned long long miss_old = 0;
        unsigned long long hit_old = 0;
        if (local_miss_blocks > 0) {
            unsigned long long inc = (static_cast<unsigned long long>(local_miss_tokens) << 32) |
                                     static_cast<unsigned int>(local_miss_blocks);
            miss_old = atomicAdd(&s_miss_state, inc);
        }
        if (local_hit_blocks > 0) {
            unsigned long long inc = (static_cast<unsigned long long>(local_hit_tokens) << 32) |
                                     static_cast<unsigned int>(local_hit_blocks);
            hit_old = atomicAdd(&s_hit_state, inc);
        }

        int miss_idx = static_cast<int>(miss_old & 0xffffffffULL);
        int miss_dst = static_cast<int>(miss_old >> 32);
        int hit_idx = static_cast<int>(hit_old & 0xffffffffULL);
        int hit_dst = static_cast<int>(hit_old >> 32);
        for (int page = 0; page < cluster_blocks; ++page) {
            int size = (page + 1 == cluster_blocks) ? (cluster_size - page * block_size) : block_size;
            int64_t key = (static_cast<int64_t>(cluster) << 32) | static_cast<uint32_t>(page);
            int slot = imi_hash_lookup(row_hash_keys, row_hash_values, hash_size, key);
            if (slot >= 0) {
                row_hit_offsets[hit_idx] = slot;
                row_hit_sizes[hit_idx] = size;
                row_hit_cumsum[hit_idx] = hit_dst;
                hit_idx++;
                hit_dst += size;
            } else {
                row_miss_offsets[miss_idx] = list_start + page * block_size;
                row_miss_sizes[miss_idx] = size;
                row_miss_cumsum[miss_idx] = miss_dst;
                row_miss_page_keys[miss_idx] = key;
                miss_idx++;
                miss_dst += size;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int miss_blocks = min(static_cast<int>(s_miss_state & 0xffffffffULL), buffer_blocks);
        int hit_blocks = min(static_cast<int>(s_hit_state & 0xffffffffULL), buffer_blocks);
        miss_counts[row] = miss_blocks;
        hit_counts[row] = hit_blocks;
        valid_lengths[row] = static_cast<int>(s_miss_state >> 32) + static_cast<int>(s_hit_state >> 32);
    }
}

template <typename T>
__global__ void promote_cache_pages_from_list_kernel(
    T* __restrict__ list_keys,
    T* __restrict__ cache_keys,
    T* __restrict__ list_values,
    T* __restrict__ cache_values,
    const int32_t* __restrict__ miss_offsets,
    const int32_t* __restrict__ miss_sizes,
    const int32_t* __restrict__ miss_counts,
    const int64_t* __restrict__ miss_page_keys,
    int64_t* __restrict__ hash_keys,
    int32_t* __restrict__ hash_values,
    int64_t* __restrict__ slot_keys,
    int32_t* __restrict__ next_slots,
    int groups,
    int list_capacity,
    int cache_size,
    int hash_size,
    int buffer_blocks,
    int max_promote_pages_per_group
) {
    int row = blockIdx.x;
    if (row >= groups) {
        return;
    }

    const int available_misses = miss_counts[row];
    const int miss_count = max_promote_pages_per_group < 0
        ? available_misses
        : min(available_misses, max_promote_pages_per_group);
    if (miss_count <= 0) {
        return;
    }

    const int32_t* row_miss_offsets = miss_offsets + static_cast<int64_t>(row) * buffer_blocks;
    const int32_t* row_miss_sizes = miss_sizes + static_cast<int64_t>(row) * buffer_blocks;
    const int64_t* row_miss_page_keys = miss_page_keys + static_cast<int64_t>(row) * buffer_blocks;
    int64_t* row_hash_keys = hash_keys + static_cast<int64_t>(row) * hash_size;
    int32_t* row_hash_values = hash_values + static_cast<int64_t>(row) * hash_size;
    int64_t* row_slot_keys = slot_keys + static_cast<int64_t>(row) * cache_size;

    __shared__ int slot_shared;
    __shared__ int hash_pos_shared;
    for (int miss_idx = 0; miss_idx < miss_count; ++miss_idx) {
        if (threadIdx.x == 0) {
            int64_t key = row_miss_page_keys[miss_idx];
            int hash_pos = next_slots[row] < cache_size
                ? imi_hash_claim_pending(row_hash_keys, row_hash_values, hash_size, key)
                : -1;
            int slot = -1;
            if (hash_pos >= 0) {
                slot = atomicAdd(next_slots + row, 1);
                if (slot < cache_size) {
                    row_slot_keys[slot] = key;
                } else {
                    atomicExch(row_hash_values + hash_pos, IMI_HASH_EMPTY);
                    atomicExch(
                        reinterpret_cast<unsigned long long*>(row_hash_keys + hash_pos),
                        static_cast<unsigned long long>(static_cast<int64_t>(IMI_HASH_TOMBSTONE))
                    );
                }
            }
            hash_pos_shared = hash_pos;
            slot_shared = slot;
        }
        __syncthreads();

        int slot = slot_shared;
        int hash_pos = hash_pos_shared;
        if (slot < 0 || slot >= cache_size || hash_pos < 0) {
            __syncthreads();
            continue;
        }
        int size = row_miss_sizes[miss_idx];
        int src_start = row_miss_offsets[miss_idx];
        int copy_elems = size * VECTOR_SIZE_CP;
        int64_t list_base = (static_cast<int64_t>(row) * list_capacity + src_start) * VECTOR_SIZE_CP;
        int64_t cache_base = (static_cast<int64_t>(row) * cache_size * CHUNK_SIZE + slot * CHUNK_SIZE) * VECTOR_SIZE_CP;

        for (int idx = threadIdx.x; idx < copy_elems; idx += blockDim.x) {
            cache_keys[cache_base + idx] = list_keys[list_base + idx];
            cache_values[cache_base + idx] = list_values[list_base + idx];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            atomicExch(row_hash_values + hash_pos, slot);
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void promote_cache_pages_kernel(
    T* __restrict__ execution_keys,
    T* __restrict__ cache_keys,
    T* __restrict__ execution_values,
    T* __restrict__ cache_values,
    const int32_t* __restrict__ miss_sizes,
    const int32_t* __restrict__ miss_cumsum,
    const int32_t* __restrict__ miss_counts,
    const int64_t* __restrict__ miss_page_keys,
    int64_t* __restrict__ hash_keys,
    int32_t* __restrict__ hash_values,
    int64_t* __restrict__ slot_keys,
    int32_t* __restrict__ next_slots,
    int groups,
    int execution_stride,
    int cache_size,
    int hash_size,
    int buffer_blocks,
    int static_len
) {
    int row = blockIdx.x;
    if (row >= groups) {
        return;
    }

    const int miss_count = miss_counts[row];
    if (miss_count <= 0) {
        return;
    }

    int64_t* row_hash_keys = hash_keys + static_cast<int64_t>(row) * hash_size;
    int32_t* row_hash_values = hash_values + static_cast<int64_t>(row) * hash_size;
    int64_t* row_slot_keys = slot_keys + static_cast<int64_t>(row) * cache_size;
    const int32_t* row_miss_sizes = miss_sizes + static_cast<int64_t>(row) * buffer_blocks;
    const int32_t* row_miss_cumsum = miss_cumsum + static_cast<int64_t>(row) * buffer_blocks;
    const int64_t* row_miss_page_keys = miss_page_keys + static_cast<int64_t>(row) * buffer_blocks;

    __shared__ int slot_shared;
    for (int miss_idx = 0; miss_idx < miss_count; ++miss_idx) {
        if (threadIdx.x == 0) {
            int slot = next_slots[row];
            next_slots[row] = (slot + 1) % cache_size;
            int64_t old_key = row_slot_keys[slot];
            imi_hash_erase(row_hash_keys, row_hash_values, hash_size, old_key);
            int64_t key = row_miss_page_keys[miss_idx];
            row_slot_keys[slot] = key;
            imi_hash_insert(row_hash_keys, row_hash_values, hash_size, key, slot);
            slot_shared = slot;
        }
        __syncthreads();

        int slot = slot_shared;
        int size = row_miss_sizes[miss_idx];
        int src_start = static_len + row_miss_cumsum[miss_idx];
        int copy_elems = size * VECTOR_SIZE_CP;
        int64_t exec_base = (static_cast<int64_t>(row) * execution_stride + src_start) * VECTOR_SIZE_CP;
        int64_t cache_base = (static_cast<int64_t>(row) * cache_size * CHUNK_SIZE + slot * CHUNK_SIZE) * VECTOR_SIZE_CP;

        for (int idx = threadIdx.x; idx < copy_elems; idx += blockDim.x) {
            cache_keys[cache_base + idx] = execution_keys[exec_base + idx];
            cache_values[cache_base + idx] = execution_values[exec_base + idx];
        }
        __syncthreads();
    }
}

__global__ void build_retrieval_descriptors_kernel(
    const int64_t* __restrict__ selected_clusters,  // [groups, nprobe]
    const int32_t* __restrict__ cluster_offsets,    // [groups, n_clusters + 1]
    const int32_t* __restrict__ cluster_sizes,      // [groups, n_clusters]
    const int8_t* __restrict__ cache_flags,         // [groups, n_clusters]
    const int32_t* __restrict__ cache_block_ids,    // [groups, n_clusters, max_blocks_per_cluster]
    int32_t* __restrict__ miss_offsets,
    int32_t* __restrict__ miss_sizes,
    int32_t* __restrict__ miss_cumsum,
    int32_t* __restrict__ miss_counts,
    int32_t* __restrict__ hit_offsets,
    int32_t* __restrict__ hit_sizes,
    int32_t* __restrict__ hit_cumsum,
    int32_t* __restrict__ hit_counts,
    int32_t* __restrict__ valid_lengths,
    int groups,
    int nprobe,
    int n_clusters,
    int buffer_blocks,
    int block_size,
    int max_blocks_per_cluster,
    int64_t padding_cluster_id
) {
    const int row = blockIdx.x;
    if (row >= groups || threadIdx.x != 0) {
        return;
    }

    const int64_t* row_selected = selected_clusters + static_cast<int64_t>(row) * nprobe;
    const int32_t* row_offsets = cluster_offsets + static_cast<int64_t>(row) * (n_clusters + 1);
    const int32_t* row_sizes = cluster_sizes + static_cast<int64_t>(row) * n_clusters;
    const int8_t* row_flags = cache_flags + static_cast<int64_t>(row) * n_clusters;
    const int32_t* row_cache_blocks =
        cache_block_ids + static_cast<int64_t>(row) * n_clusters * max_blocks_per_cluster;

    int32_t* row_miss_offsets = miss_offsets + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_miss_sizes = miss_sizes + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_miss_cumsum = miss_cumsum + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_hit_offsets = hit_offsets + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_hit_sizes = hit_sizes + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_hit_cumsum = hit_cumsum + static_cast<int64_t>(row) * buffer_blocks;

    int miss_blocks = 0;
    int hit_blocks = 0;
    int miss_tokens = 0;
    int hit_tokens = 0;
    int considered_blocks = 0;

    for (int i = 0; i < nprobe; ++i) {
        const int cluster = static_cast<int>(row_selected[i]);
        if (cluster < 0 || cluster >= n_clusters || row_selected[i] == padding_cluster_id) {
            continue;
        }
        const int cluster_size = row_sizes[cluster];
        if (cluster_size <= 0) {
            continue;
        }
        const int cluster_blocks = (cluster_size + block_size - 1) / block_size;
        if (cluster_blocks <= 0 || cluster_blocks > max_blocks_per_cluster) {
            continue;
        }
        if (considered_blocks + cluster_blocks > buffer_blocks) {
            break;
        }
        considered_blocks += cluster_blocks;

        if (row_flags[cluster] != 0) {
            const int32_t* blocks = row_cache_blocks + static_cast<int64_t>(cluster) * max_blocks_per_cluster;
            for (int b = 0; b < cluster_blocks; ++b) {
                const int size = (b + 1 == cluster_blocks) ? (cluster_size - b * block_size) : block_size;
                row_hit_offsets[hit_blocks] = blocks[b];
                row_hit_sizes[hit_blocks] = size;
                row_hit_cumsum[hit_blocks] = hit_tokens;
                hit_tokens += size;
                hit_blocks++;
            }
        } else {
            int start = row_offsets[cluster];
            for (int b = 0; b < cluster_blocks; ++b) {
                const int size = (b + 1 == cluster_blocks) ? (cluster_size - b * block_size) : block_size;
                row_miss_offsets[miss_blocks] = start + b * block_size;
                row_miss_sizes[miss_blocks] = size;
                row_miss_cumsum[miss_blocks] = miss_tokens;
                miss_tokens += size;
                miss_blocks++;
            }
        }
    }

    miss_counts[row] = miss_blocks;
    hit_counts[row] = hit_blocks;
    valid_lengths[row] = miss_tokens + hit_tokens;
}

void build_retrieval_descriptors(
    torch::Tensor& selected_clusters,
    torch::Tensor& cluster_offsets,
    torch::Tensor& cluster_sizes,
    torch::Tensor& cache_flags,
    torch::Tensor& cache_block_ids,
    torch::Tensor& miss_offsets,
    torch::Tensor& miss_sizes,
    torch::Tensor& miss_cumsum,
    torch::Tensor& miss_counts,
    torch::Tensor& hit_offsets,
    torch::Tensor& hit_sizes,
    torch::Tensor& hit_cumsum,
    torch::Tensor& hit_counts,
    torch::Tensor& valid_lengths,
    int groups,
    int nprobe,
    int n_clusters,
    int buffer_blocks,
    int block_size,
    int max_blocks_per_cluster,
    int64_t padding_cluster_id
) {
    TORCH_CHECK(selected_clusters.is_cuda(), "selected_clusters must be CUDA");
    TORCH_CHECK(cluster_offsets.is_cuda(), "cluster_offsets must be CUDA");
    TORCH_CHECK(cluster_sizes.is_cuda(), "cluster_sizes must be CUDA");
    TORCH_CHECK(cache_flags.is_cuda() || cache_flags.is_pinned(), "cache_flags must be CUDA or pinned CPU");
    TORCH_CHECK(cache_block_ids.is_cuda() || cache_block_ids.is_pinned(), "cache_block_ids must be CUDA or pinned CPU");
    TORCH_CHECK(miss_offsets.is_cuda(), "miss_offsets must be CUDA");
    TORCH_CHECK(miss_sizes.is_cuda(), "miss_sizes must be CUDA");
    TORCH_CHECK(miss_cumsum.is_cuda(), "miss_cumsum must be CUDA");
    TORCH_CHECK(miss_counts.is_cuda(), "miss_counts must be CUDA");
    TORCH_CHECK(hit_offsets.is_cuda(), "hit_offsets must be CUDA");
    TORCH_CHECK(hit_sizes.is_cuda(), "hit_sizes must be CUDA");
    TORCH_CHECK(hit_cumsum.is_cuda(), "hit_cumsum must be CUDA");
    TORCH_CHECK(hit_counts.is_cuda(), "hit_counts must be CUDA");
    TORCH_CHECK(valid_lengths.is_cuda(), "valid_lengths must be CUDA");

    const auto stream = at::cuda::getCurrentCUDAStream();
    build_retrieval_descriptors_kernel<<<groups, 1, 0, stream>>>(
        reinterpret_cast<int64_t*>(selected_clusters.data_ptr<int64_t>()),
        reinterpret_cast<int32_t*>(cluster_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cluster_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int8_t*>(cache_flags.data_ptr<int8_t>()),
        reinterpret_cast<int32_t*>(cache_block_ids.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_cumsum.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_counts.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(hit_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(hit_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(hit_cumsum.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(hit_counts.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(valid_lengths.data_ptr<int32_t>()),
        groups,
        nprobe,
        n_clusters,
        buffer_blocks,
        block_size,
        max_blocks_per_cluster,
        padding_cluster_id
    );
}

void build_cache_aware_retrieval_plan(
    torch::Tensor& selected_clusters,
    torch::Tensor& cluster_offsets,
    torch::Tensor& cluster_sizes,
    torch::Tensor& cache_hash_keys,
    torch::Tensor& cache_hash_values,
    torch::Tensor& miss_offsets,
    torch::Tensor& miss_sizes,
    torch::Tensor& miss_cumsum,
    torch::Tensor& miss_counts,
    torch::Tensor& miss_page_keys,
    torch::Tensor& hit_offsets,
    torch::Tensor& hit_sizes,
    torch::Tensor& hit_cumsum,
    torch::Tensor& hit_counts,
    torch::Tensor& valid_lengths,
    int groups,
    int nprobe,
    int n_clusters,
    int buffer_blocks,
    int block_size,
    int hash_size,
    int64_t padding_cluster_id
) {
    TORCH_CHECK(selected_clusters.is_cuda(), "selected_clusters must be CUDA");
    TORCH_CHECK(cluster_offsets.is_cuda(), "cluster_offsets must be CUDA");
    TORCH_CHECK(cluster_sizes.is_cuda(), "cluster_sizes must be CUDA");
    TORCH_CHECK(cache_hash_keys.is_cuda(), "cache_hash_keys must be CUDA");
    TORCH_CHECK(cache_hash_values.is_cuda(), "cache_hash_values must be CUDA");
    TORCH_CHECK(miss_offsets.is_cuda(), "miss_offsets must be CUDA");
    TORCH_CHECK(miss_sizes.is_cuda(), "miss_sizes must be CUDA");
    TORCH_CHECK(miss_cumsum.is_cuda(), "miss_cumsum must be CUDA");
    TORCH_CHECK(miss_counts.is_cuda(), "miss_counts must be CUDA");
    TORCH_CHECK(miss_page_keys.is_cuda(), "miss_page_keys must be CUDA");
    TORCH_CHECK(hit_offsets.is_cuda(), "hit_offsets must be CUDA");
    TORCH_CHECK(hit_sizes.is_cuda(), "hit_sizes must be CUDA");
    TORCH_CHECK(hit_cumsum.is_cuda(), "hit_cumsum must be CUDA");
    TORCH_CHECK(hit_counts.is_cuda(), "hit_counts must be CUDA");
    TORCH_CHECK(valid_lengths.is_cuda(), "valid_lengths must be CUDA");
    TORCH_CHECK((hash_size & (hash_size - 1)) == 0, "hash_size must be a power of two");

    const auto stream = at::cuda::getCurrentCUDAStream();
    build_cache_aware_retrieval_plan_kernel<<<groups, 256, 0, stream>>>(
        reinterpret_cast<int64_t*>(selected_clusters.data_ptr<int64_t>()),
        reinterpret_cast<int32_t*>(cluster_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cluster_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int64_t*>(cache_hash_keys.data_ptr<int64_t>()),
        reinterpret_cast<int32_t*>(cache_hash_values.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_cumsum.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_counts.data_ptr<int32_t>()),
        reinterpret_cast<int64_t*>(miss_page_keys.data_ptr<int64_t>()),
        reinterpret_cast<int32_t*>(hit_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(hit_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(hit_cumsum.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(hit_counts.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(valid_lengths.data_ptr<int32_t>()),
        groups,
        nprobe,
        n_clusters,
        buffer_blocks,
        block_size,
        hash_size,
        padding_cluster_id
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void promote_cache_pages(
    torch::Tensor& execution_keys,
    torch::Tensor& cache_keys,
    torch::Tensor& execution_values,
    torch::Tensor& cache_values,
    torch::Tensor& miss_sizes,
    torch::Tensor& miss_cumsum,
    torch::Tensor& miss_counts,
    torch::Tensor& miss_page_keys,
    torch::Tensor& cache_hash_keys,
    torch::Tensor& cache_hash_values,
    torch::Tensor& cache_slot_keys,
    torch::Tensor& cache_next_slots,
    int groups,
    int execution_stride,
    int cache_size,
    int hash_size,
    int buffer_blocks,
    int static_len
) {
    TORCH_CHECK(execution_keys.is_cuda(), "execution_keys must be CUDA");
    TORCH_CHECK(cache_keys.is_cuda(), "cache_keys must be CUDA");
    TORCH_CHECK(execution_values.is_cuda(), "execution_values must be CUDA");
    TORCH_CHECK(cache_values.is_cuda(), "cache_values must be CUDA");
    TORCH_CHECK(miss_sizes.is_cuda(), "miss_sizes must be CUDA");
    TORCH_CHECK(miss_cumsum.is_cuda(), "miss_cumsum must be CUDA");
    TORCH_CHECK(miss_counts.is_cuda(), "miss_counts must be CUDA");
    TORCH_CHECK(miss_page_keys.is_cuda(), "miss_page_keys must be CUDA");
    TORCH_CHECK(cache_hash_keys.is_cuda(), "cache_hash_keys must be CUDA");
    TORCH_CHECK(cache_hash_values.is_cuda(), "cache_hash_values must be CUDA");
    TORCH_CHECK(cache_slot_keys.is_cuda(), "cache_slot_keys must be CUDA");
    TORCH_CHECK(cache_next_slots.is_cuda(), "cache_next_slots must be CUDA");
    TORCH_CHECK((hash_size & (hash_size - 1)) == 0, "hash_size must be a power of two");

    const int blockSize = BLOCK_SIZE_CP;
    const auto stream = at::cuda::getCurrentCUDAStream();
#if DATA_BYTES == 4
    promote_cache_pages_kernel<float><<<groups, blockSize, 0, stream>>>(
        reinterpret_cast<float*>(execution_keys.data_ptr<float>()),
        reinterpret_cast<float*>(cache_keys.data_ptr<float>()),
        reinterpret_cast<float*>(execution_values.data_ptr<float>()),
        reinterpret_cast<float*>(cache_values.data_ptr<float>()),
#else
    if (execution_keys.dtype() == torch::kFloat16) {
        promote_cache_pages_kernel<PTYPE><<<groups, blockSize, 0, stream>>>(
            reinterpret_cast<PTYPE*>(execution_keys.data_ptr<at::Half>()),
            reinterpret_cast<PTYPE*>(cache_keys.data_ptr<at::Half>()),
            reinterpret_cast<PTYPE*>(execution_values.data_ptr<at::Half>()),
            reinterpret_cast<PTYPE*>(cache_values.data_ptr<at::Half>()),
            reinterpret_cast<int32_t*>(miss_sizes.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(miss_cumsum.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(miss_counts.data_ptr<int32_t>()),
            reinterpret_cast<int64_t*>(miss_page_keys.data_ptr<int64_t>()),
            reinterpret_cast<int64_t*>(cache_hash_keys.data_ptr<int64_t>()),
            reinterpret_cast<int32_t*>(cache_hash_values.data_ptr<int32_t>()),
            reinterpret_cast<int64_t*>(cache_slot_keys.data_ptr<int64_t>()),
            reinterpret_cast<int32_t*>(cache_next_slots.data_ptr<int32_t>()),
            groups,
            execution_stride,
            cache_size,
            hash_size,
            buffer_blocks,
            static_len
        );
    } else {
        promote_cache_pages_kernel<PTYPE><<<groups, blockSize, 0, stream>>>(
            reinterpret_cast<PTYPE*>(execution_keys.data_ptr<at::BFloat16>()),
            reinterpret_cast<PTYPE*>(cache_keys.data_ptr<at::BFloat16>()),
            reinterpret_cast<PTYPE*>(execution_values.data_ptr<at::BFloat16>()),
            reinterpret_cast<PTYPE*>(cache_values.data_ptr<at::BFloat16>()),
#endif
            reinterpret_cast<int32_t*>(miss_sizes.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(miss_cumsum.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(miss_counts.data_ptr<int32_t>()),
            reinterpret_cast<int64_t*>(miss_page_keys.data_ptr<int64_t>()),
            reinterpret_cast<int64_t*>(cache_hash_keys.data_ptr<int64_t>()),
            reinterpret_cast<int32_t*>(cache_hash_values.data_ptr<int32_t>()),
            reinterpret_cast<int64_t*>(cache_slot_keys.data_ptr<int64_t>()),
            reinterpret_cast<int32_t*>(cache_next_slots.data_ptr<int32_t>()),
            groups,
            execution_stride,
            cache_size,
            hash_size,
            buffer_blocks,
            static_len
        );
#if DATA_BYTES != 4
    }
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void promote_cache_pages_from_list(
    torch::Tensor& list_keys,
    torch::Tensor& cache_keys,
    torch::Tensor& list_values,
    torch::Tensor& cache_values,
    torch::Tensor& miss_offsets,
    torch::Tensor& miss_sizes,
    torch::Tensor& miss_counts,
    torch::Tensor& miss_page_keys,
    torch::Tensor& cache_hash_keys,
    torch::Tensor& cache_hash_values,
    torch::Tensor& cache_slot_keys,
    torch::Tensor& cache_next_slots,
    int groups,
    int list_capacity,
    int cache_size,
    int hash_size,
    int buffer_blocks,
    int max_promote_pages_per_group
) {
    TORCH_CHECK(list_keys.is_pinned() || list_keys.is_cuda(), "list_keys must be pinned CPU or CUDA");
    TORCH_CHECK(list_values.is_pinned() || list_values.is_cuda(), "list_values must be pinned CPU or CUDA");
    TORCH_CHECK(cache_keys.is_cuda(), "cache_keys must be CUDA");
    TORCH_CHECK(cache_values.is_cuda(), "cache_values must be CUDA");
    TORCH_CHECK(miss_offsets.is_cuda(), "miss_offsets must be CUDA");
    TORCH_CHECK(miss_sizes.is_cuda(), "miss_sizes must be CUDA");
    TORCH_CHECK(miss_counts.is_cuda(), "miss_counts must be CUDA");
    TORCH_CHECK(miss_page_keys.is_cuda(), "miss_page_keys must be CUDA");
    TORCH_CHECK(cache_hash_keys.is_cuda(), "cache_hash_keys must be CUDA");
    TORCH_CHECK(cache_hash_values.is_cuda(), "cache_hash_values must be CUDA");
    TORCH_CHECK(cache_slot_keys.is_cuda(), "cache_slot_keys must be CUDA");
    TORCH_CHECK(cache_next_slots.is_cuda(), "cache_next_slots must be CUDA");
    TORCH_CHECK((hash_size & (hash_size - 1)) == 0, "hash_size must be a power of two");

    const int blockSize = BLOCK_SIZE_CP;
    const auto stream = at::cuda::getCurrentCUDAStream();
#if DATA_BYTES == 4
    promote_cache_pages_from_list_kernel<float><<<groups, blockSize, 0, stream>>>(
        reinterpret_cast<float*>(list_keys.data_ptr<float>()),
        reinterpret_cast<float*>(cache_keys.data_ptr<float>()),
        reinterpret_cast<float*>(list_values.data_ptr<float>()),
        reinterpret_cast<float*>(cache_values.data_ptr<float>()),
#else
    if (list_keys.dtype() == torch::kFloat16) {
        promote_cache_pages_from_list_kernel<PTYPE><<<groups, blockSize, 0, stream>>>(
            reinterpret_cast<PTYPE*>(list_keys.data_ptr<at::Half>()),
            reinterpret_cast<PTYPE*>(cache_keys.data_ptr<at::Half>()),
            reinterpret_cast<PTYPE*>(list_values.data_ptr<at::Half>()),
            reinterpret_cast<PTYPE*>(cache_values.data_ptr<at::Half>()),
            reinterpret_cast<int32_t*>(miss_offsets.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(miss_sizes.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(miss_counts.data_ptr<int32_t>()),
            reinterpret_cast<int64_t*>(miss_page_keys.data_ptr<int64_t>()),
            reinterpret_cast<int64_t*>(cache_hash_keys.data_ptr<int64_t>()),
            reinterpret_cast<int32_t*>(cache_hash_values.data_ptr<int32_t>()),
            reinterpret_cast<int64_t*>(cache_slot_keys.data_ptr<int64_t>()),
            reinterpret_cast<int32_t*>(cache_next_slots.data_ptr<int32_t>()),
            groups,
            list_capacity,
            cache_size,
            hash_size,
            buffer_blocks,
            max_promote_pages_per_group
        );
    } else {
        promote_cache_pages_from_list_kernel<PTYPE><<<groups, blockSize, 0, stream>>>(
            reinterpret_cast<PTYPE*>(list_keys.data_ptr<at::BFloat16>()),
            reinterpret_cast<PTYPE*>(cache_keys.data_ptr<at::BFloat16>()),
            reinterpret_cast<PTYPE*>(list_values.data_ptr<at::BFloat16>()),
            reinterpret_cast<PTYPE*>(cache_values.data_ptr<at::BFloat16>()),
#endif
            reinterpret_cast<int32_t*>(miss_offsets.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(miss_sizes.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(miss_counts.data_ptr<int32_t>()),
            reinterpret_cast<int64_t*>(miss_page_keys.data_ptr<int64_t>()),
            reinterpret_cast<int64_t*>(cache_hash_keys.data_ptr<int64_t>()),
            reinterpret_cast<int32_t*>(cache_hash_values.data_ptr<int32_t>()),
            reinterpret_cast<int64_t*>(cache_slot_keys.data_ptr<int64_t>()),
            reinterpret_cast<int32_t*>(cache_next_slots.data_ptr<int32_t>()),
            groups,
            list_capacity,
            cache_size,
            hash_size,
            buffer_blocks,
            max_promote_pages_per_group
        );
#if DATA_BYTES != 4
    }
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// gather copy keys, values and metadata in vector-level
void gather_copy_vectors(
    torch::Tensor& key_data,        // input, shape (groups, num_vectors, dim)
    torch::Tensor& key_buffer,      // output, shape (groups, buffer_size, dim)
    torch::Tensor& value_data,      // input, shape (groups, num_vectors, dim)
    torch::Tensor& value_buffer,    // output, shape (groups, buffer_size, dim)
    torch::Tensor& src_metadata,    // input, shape (groups, num_vectors)
    torch::Tensor& dst_metadata,    // output, shape (groups, buffer_size)
    
    torch::Tensor& data_offsets,    // input, shape (groups, index_size)

    int groups, 
    int data_length,                // num_vectors
    int buffer_length,              // buffer_size
    int index_size,                 // index_size
    int copy_start,                 // start vector index for copy
    int copy_num                    // number of vectors need to copy from data
) {
    const int blockSize = BLOCK_SIZE_CP;
    const int numBlocks = groups * SPLIT_FACTOR;
    const int maxSMBytes = copy_num * sizeof(int);

    PTYPE* key_data_ptr;
    PTYPE* key_buffer_ptr;
    PTYPE* value_data_ptr;
    PTYPE* value_buffer_ptr;
    MYTPE* src_metadata_ptr;
    MYTPE* dst_metadata_ptr;

    // Cast fp32/fp16 data pointers to int4(16 Bytes)
#if DATA_BYTES == 4
    key_data_ptr = reinterpret_cast<PTYPE*>(key_data.data_ptr<float>());
    key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<float>());
    value_data_ptr = reinterpret_cast<PTYPE*>(value_data.data_ptr<float>());
    value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<float>());
    src_metadata_ptr = reinterpret_cast<MYTPE*>(src_metadata.data_ptr<float>());
    dst_metadata_ptr = reinterpret_cast<MYTPE*>(dst_metadata.data_ptr<float>());
#else
    if (key_data.dtype() == torch::kFloat16) {     // fp16
        key_data_ptr = reinterpret_cast<PTYPE*>(key_data.data_ptr<at::Half>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::Half>());
        value_data_ptr = reinterpret_cast<PTYPE*>(value_data.data_ptr<at::Half>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::Half>());
        src_metadata_ptr = reinterpret_cast<MYTPE*>(src_metadata.data_ptr<at::Half>());
        dst_metadata_ptr = reinterpret_cast<MYTPE*>(dst_metadata.data_ptr<at::Half>());
    } else {    // bf16
        key_data_ptr = reinterpret_cast<PTYPE*>(key_data.data_ptr<at::BFloat16>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::BFloat16>());
        value_data_ptr = reinterpret_cast<PTYPE*>(value_data.data_ptr<at::BFloat16>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::BFloat16>());
        src_metadata_ptr = reinterpret_cast<MYTPE*>(src_metadata.data_ptr<at::BFloat16>());
        dst_metadata_ptr = reinterpret_cast<MYTPE*>(dst_metadata.data_ptr<at::BFloat16>());
    }
#endif
    
    int64_t* data_offsets_ptr = reinterpret_cast<int64_t*>(data_offsets.data_ptr<int64_t>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(random_gather_copy_vector<PTYPE, MYTPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);
    random_gather_copy_vector<PTYPE, MYTPE><<<numBlocks, blockSize, maxSMBytes, stream>>>(
        key_data_ptr, 
        key_buffer_ptr, 
        value_data_ptr,
        value_buffer_ptr,
        src_metadata_ptr,
        dst_metadata_ptr,
        data_length,
        buffer_length,
        data_offsets_ptr,
        index_size,
        copy_start,
        copy_num);
}


// gather copy keys, values and from three src and concat into one buffer
void gather_copy_and_concat(
    torch::Tensor& key_data1,    // input, shape (groups, num_vectors1, dim)
    torch::Tensor& key_data2,    // input, shape (groups, num_vectors2, dim)
    torch::Tensor& key_data3,    // input, shape (groups, num_units3, unit_size, dim)
    torch::Tensor& key_buffer,   // output, shape (groups, buffer_num_vectors, dim)
    torch::Tensor& value_data1,  // input, shape (groups, num_vectors1, dim)
    torch::Tensor& value_data2,  // input, shape (groups, num_vectors2, dim)
    torch::Tensor& value_data3,  // input, shape (groups, num_units3, unit_size, dim)
    torch::Tensor& value_buffer, // output, shape (groups, buffer_num_vectors, dim)
    
    torch::Tensor& data_offsets2,       // input, shape (groups, offset_length)
    torch::Tensor& data_copy_sizes2,    // input, shape (groups, offset_length)
    torch::Tensor& buffer_offsets2,     // input, shape (groups, offset_length)
    torch::Tensor& copy_chunks2,        // input, shape (groups,)
    torch::Tensor& data_offsets3,       // input, shape (groups, offset_length)
    torch::Tensor& data_copy_sizes3,    // input, shape (groups, offset_length)
    torch::Tensor& buffer_offsets3,     // input, shape (groups, offset_length)
    torch::Tensor& copy_chunks3,        // input, shape (groups,)

    torch::Tensor& valid_lengths,       // output, shape (groups,)

    int groups, 
    int data_length1,           // num_vectors1
    int data_length2,           // num_vectors2
    int data_length3,           // num_units3
    int buffer_length,          // buffer_num_vectors
    int offset_length,
    torch::Tensor& copy_vectors1 // number of vectors need to copy from data1
) {
    const int blockSize = BLOCK_SIZE_CP;
    const int numBlocks = groups * SPLIT_FACTOR;
    const int maxSMBytes = (CHUNK_SIZE + 1 + offset_length * 3) * sizeof(int);

    PTYPE* key_data_ptr1;
    PTYPE* key_data_ptr2;
    PTYPE* key_data_ptr3;
    PTYPE* key_buffer_ptr;
    PTYPE* value_data_ptr1;
    PTYPE* value_data_ptr2;
    PTYPE* value_data_ptr3;
    PTYPE* value_buffer_ptr;

    // Cast fp32/fp16 data pointers to int4(16 Bytes)
#if DATA_BYTES == 4
    key_data_ptr1 = reinterpret_cast<PTYPE*>(key_data1.data_ptr<float>());
    key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<float>());
    key_data_ptr3 = reinterpret_cast<PTYPE*>(key_data3.data_ptr<float>());
    key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<float>());
    value_data_ptr1 = reinterpret_cast<PTYPE*>(value_data1.data_ptr<float>());
    value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<float>());
    value_data_ptr3 = reinterpret_cast<PTYPE*>(value_data3.data_ptr<float>());
    value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<float>());
#else
    if (key_data1.dtype() == torch::kFloat16) {     // fp16
        key_data_ptr1 = reinterpret_cast<PTYPE*>(key_data1.data_ptr<at::Half>());
        key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<at::Half>());
        key_data_ptr3 = reinterpret_cast<PTYPE*>(key_data3.data_ptr<at::Half>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::Half>());
        value_data_ptr1 = reinterpret_cast<PTYPE*>(value_data1.data_ptr<at::Half>());
        value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<at::Half>());
        value_data_ptr3 = reinterpret_cast<PTYPE*>(value_data3.data_ptr<at::Half>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::Half>());
    } else {    // bf16
        key_data_ptr1 = reinterpret_cast<PTYPE*>(key_data1.data_ptr<at::BFloat16>());
        key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<at::BFloat16>());
        key_data_ptr3 = reinterpret_cast<PTYPE*>(key_data3.data_ptr<at::BFloat16>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::BFloat16>());
        value_data_ptr1 = reinterpret_cast<PTYPE*>(value_data1.data_ptr<at::BFloat16>());
        value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<at::BFloat16>());
        value_data_ptr3 = reinterpret_cast<PTYPE*>(value_data3.data_ptr<at::BFloat16>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::BFloat16>());
    }
#endif
    
    int* data_offsets_ptr2 = reinterpret_cast<int*>(data_offsets2.data_ptr<int32_t>());
    int* data_copy_sizes_ptr2 = reinterpret_cast<int*>(data_copy_sizes2.data_ptr<int32_t>());
    int* buffer_offsets_ptr2 = reinterpret_cast<int*>(buffer_offsets2.data_ptr<int32_t>());
    int* copy_chunks_ptr2 = reinterpret_cast<int*>(copy_chunks2.data_ptr<int32_t>());
    int* data_offsets_ptr3 = reinterpret_cast<int*>(data_offsets3.data_ptr<int32_t>());
    int* data_copy_sizes_ptr3 = reinterpret_cast<int*>(data_copy_sizes3.data_ptr<int32_t>());
    int* buffer_offsets_ptr3 = reinterpret_cast<int*>(buffer_offsets3.data_ptr<int32_t>());
    int* copy_chunks_ptr3 = reinterpret_cast<int*>(copy_chunks3.data_ptr<int32_t>());

    int* valid_lengths_ptr = reinterpret_cast<int*>(valid_lengths.data_ptr<int32_t>());
    int* copy_vectors1_ptr = reinterpret_cast<int*>(copy_vectors1.data_ptr<int32_t>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(concat_gather_copy<PTYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);
    concat_gather_copy<PTYPE><<<numBlocks, blockSize, maxSMBytes, stream>>>(
        key_data_ptr1, 
        key_data_ptr2,
        key_data_ptr3,
        key_buffer_ptr, 
        value_data_ptr1,
        value_data_ptr2,
        value_data_ptr3,
        value_buffer_ptr,
        data_length1*128,
        data_length2*128,
        data_length3*CHUNK_SIZE*128,
        buffer_length*128,
        copy_vectors1_ptr,
        offset_length,
        data_offsets_ptr2,
        data_copy_sizes_ptr2,
        buffer_offsets_ptr2,
        copy_chunks_ptr2,
        data_offsets_ptr3,
        data_copy_sizes_ptr3,
        buffer_offsets_ptr3,
        copy_chunks_ptr3,
        valid_lengths_ptr
    );
}


void gather_copy_and_concat_retrieval(
    torch::Tensor& key_data2,    // input, shape (groups, num_vectors2, dim)
    torch::Tensor& key_data3,    // input, shape (groups, num_units3, unit_size, dim)
    torch::Tensor& key_buffer,   // output, shape (groups, buffer_num_vectors, dim)
    torch::Tensor& value_data2,  // input, shape (groups, num_vectors2, dim)
    torch::Tensor& value_data3,  // input, shape (groups, num_units3, unit_size, dim)
    torch::Tensor& value_buffer, // output, shape (groups, buffer_num_vectors, dim)

    torch::Tensor& data_offsets2,       // input, shape (groups, offset_length)
    torch::Tensor& data_copy_sizes2,    // input, shape (groups, offset_length)
    torch::Tensor& buffer_offsets2,     // input, shape (groups, offset_length)
    torch::Tensor& copy_chunks2,        // input, shape (groups,)
    torch::Tensor& data_offsets3,       // input, shape (groups, offset_length)
    torch::Tensor& data_copy_sizes3,    // input, shape (groups, offset_length)
    torch::Tensor& buffer_offsets3,     // input, shape (groups, offset_length)
    torch::Tensor& copy_chunks3,        // input, shape (groups,)

    torch::Tensor& valid_lengths,       // output, shape (groups,)

    int groups,
    int data_length2,
    int data_length3,
    int buffer_length,
    int offset_length
) {
    const int blockSize = BLOCK_SIZE_CP;
    const int numBlocks = groups * SPLIT_FACTOR;
    const int maxSMBytes = (CHUNK_SIZE + 1 + offset_length * 3) * sizeof(int);

    PTYPE* key_data_ptr2;
    PTYPE* key_data_ptr3;
    PTYPE* key_buffer_ptr;
    PTYPE* value_data_ptr2;
    PTYPE* value_data_ptr3;
    PTYPE* value_buffer_ptr;

#if DATA_BYTES == 4
    key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<float>());
    key_data_ptr3 = reinterpret_cast<PTYPE*>(key_data3.data_ptr<float>());
    key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<float>());
    value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<float>());
    value_data_ptr3 = reinterpret_cast<PTYPE*>(value_data3.data_ptr<float>());
    value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<float>());
#else
    if (key_data2.dtype() == torch::kFloat16) {
        key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<at::Half>());
        key_data_ptr3 = reinterpret_cast<PTYPE*>(key_data3.data_ptr<at::Half>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::Half>());
        value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<at::Half>());
        value_data_ptr3 = reinterpret_cast<PTYPE*>(value_data3.data_ptr<at::Half>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::Half>());
    } else {
        key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<at::BFloat16>());
        key_data_ptr3 = reinterpret_cast<PTYPE*>(key_data3.data_ptr<at::BFloat16>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::BFloat16>());
        value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<at::BFloat16>());
        value_data_ptr3 = reinterpret_cast<PTYPE*>(value_data3.data_ptr<at::BFloat16>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::BFloat16>());
    }
#endif

    int* data_offsets_ptr2 = reinterpret_cast<int*>(data_offsets2.data_ptr<int32_t>());
    int* data_copy_sizes_ptr2 = reinterpret_cast<int*>(data_copy_sizes2.data_ptr<int32_t>());
    int* buffer_offsets_ptr2 = reinterpret_cast<int*>(buffer_offsets2.data_ptr<int32_t>());
    int* copy_chunks_ptr2 = reinterpret_cast<int*>(copy_chunks2.data_ptr<int32_t>());
    int* data_offsets_ptr3 = reinterpret_cast<int*>(data_offsets3.data_ptr<int32_t>());
    int* data_copy_sizes_ptr3 = reinterpret_cast<int*>(data_copy_sizes3.data_ptr<int32_t>());
    int* buffer_offsets_ptr3 = reinterpret_cast<int*>(buffer_offsets3.data_ptr<int32_t>());
    int* copy_chunks_ptr3 = reinterpret_cast<int*>(copy_chunks3.data_ptr<int32_t>());
    int* valid_lengths_ptr = reinterpret_cast<int*>(valid_lengths.data_ptr<int32_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(concat_gather_copy_retrieval<PTYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);
    concat_gather_copy_retrieval<PTYPE><<<numBlocks, blockSize, maxSMBytes, stream>>>(
        key_data_ptr2,
        key_data_ptr3,
        key_buffer_ptr,
        value_data_ptr2,
        value_data_ptr3,
        value_buffer_ptr,
        data_length2 * 128,
        data_length3 * CHUNK_SIZE * 128,
        buffer_length * 128,
        offset_length,
        data_offsets_ptr2,
        data_copy_sizes_ptr2,
        buffer_offsets_ptr2,
        copy_chunks_ptr2,
        data_offsets_ptr3,
        data_copy_sizes_ptr3,
        buffer_offsets_ptr3,
        copy_chunks_ptr3,
        valid_lengths_ptr
    );
}


// gather copy from src and scatter to dst
void gather_copy_and_scatter(
    torch::Tensor& key_data,    // input, src values, shape (groups, num_vectors, dim)
    torch::Tensor& key_buffer,  // inout, dst values, shape (groups, buffer_num_units, unit_size, dim)
    torch::Tensor& value_data,  // input, src values, shape (groups, num_vectors, dim)
    torch::Tensor& value_buffer,// inout, dst values, shape (groups, buffer_num_units, unit_size, dim)

    torch::Tensor& s_offsets,   // input, src offsets (groups, offset_length)
    torch::Tensor& copy_sizes,  // input, copy vector number for each chunk (groups, offset_length)
    torch::Tensor& d_offsets,   // input, dst offsets (groups, offset_length)
    torch::Tensor& copy_chunks, // input, number of chunks need to copy (groups,) 

    int groups, 
    int data_length,            // num_vectors
    int buffer_length,          // buffer_num_units
    int offset_length,
    torch::Tensor& s_copy_start // start vector index for copy src
) {
    const int blockSize = BLOCK_SIZE_CP;
    const int numBlocks = groups;
    const int maxSMBytes = (CHUNK_SIZE + 1 + offset_length * 3) * sizeof(int);

    PTYPE* key_data_ptr;
    PTYPE* key_buffer_ptr;
    PTYPE* value_data_ptr;
    PTYPE* value_buffer_ptr;

    // Cast fp32/fp16 data pointers to int4(16 Bytes)
#if DATA_BYTES == 4
    key_data_ptr = reinterpret_cast<PTYPE*>(key_data.data_ptr<float>());
    key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<float>());
    value_data_ptr = reinterpret_cast<PTYPE*>(value_data.data_ptr<float>());
    value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<float>());
#else
    if (key_data.dtype() == torch::kFloat16) {      // fp16
        key_data_ptr = reinterpret_cast<PTYPE*>(key_data.data_ptr<at::Half>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::Half>());
        value_data_ptr = reinterpret_cast<PTYPE*>(value_data.data_ptr<at::Half>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::Half>());
    } else {    // bf16
        key_data_ptr = reinterpret_cast<PTYPE*>(key_data.data_ptr<at::BFloat16>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::BFloat16>());
        value_data_ptr = reinterpret_cast<PTYPE*>(value_data.data_ptr<at::BFloat16>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::BFloat16>());
    }
#endif

    int* src_offsets_ptr = reinterpret_cast<int*>(s_offsets.data_ptr<int32_t>());
    int* copy_sizes_ptr = reinterpret_cast<int*>(copy_sizes.data_ptr<int32_t>());
    int* dst_offsets_ptr = reinterpret_cast<int*>(d_offsets.data_ptr<int32_t>());
    int* copy_chunks_ptr = reinterpret_cast<int*>(copy_chunks.data_ptr<int32_t>());

    int* s_copy_start_ptr = reinterpret_cast<int*>(s_copy_start.data_ptr<int32_t>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(gather_copy_scatter<PTYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);
    gather_copy_scatter<PTYPE><<<numBlocks, blockSize, maxSMBytes, stream>>>(
        key_data_ptr, 
        key_buffer_ptr, 
        value_data_ptr,
        value_buffer_ptr,
        data_length*128, 
        buffer_length*CHUNK_SIZE*128,
        offset_length,
        s_copy_start_ptr,
        src_offsets_ptr,
        copy_sizes_ptr,
        dst_offsets_ptr,
        copy_chunks_ptr
    );
}


// reorganize keys, values into clusters
void reorganize_vectors(
    torch::Tensor& key_src,         // input, shape (groups, src_num_vectors, dim)
    torch::Tensor& key_dst,         // output, shape (groups, dst_num_vectors, dim)
    torch::Tensor& value_src,       // input, shape (groups, src_num_vectors, dim)
    torch::Tensor& value_dst,       // output, shape (groups, dst_num_vectors, dim)
    
    torch::Tensor& src_offsets,     // input, shape (groups, copy_cluster_num, max_cluster_size)
    torch::Tensor& dst_cumsum,      // input, shape (groups, n_centroids)

    int groups, 
    int copy_start                  // start cluster id to copy
) {
    const int blockSize = BLOCK_SIZE_CP;
    const int numBlocks = groups * SPLIT_FACTOR;

    int copy_cluster_num = src_offsets.size(-2);
    int max_cluster_size = src_offsets.size(-1);
    int n_centroids = dst_cumsum.size(-1);
    int src_length = key_src.size(-2);
    int dst_length = key_dst.size(-2);

    int split_size = (copy_cluster_num + SPLIT_FACTOR - 1) / SPLIT_FACTOR;
    const int maxSMBytes = (split_size*max_cluster_size + split_size + split_size) * sizeof(int);

    PTYPE* key_src_ptr;
    PTYPE* key_dst_ptr;
    PTYPE* value_src_ptr;
    PTYPE* value_dst_ptr;

    // Cast fp32/fp16 data pointers to int4(16 Bytes)
#if DATA_BYTES == 4
    key_src_ptr = reinterpret_cast<PTYPE*>(key_src.data_ptr<float>());
    key_dst_ptr = reinterpret_cast<PTYPE*>(key_dst.data_ptr<float>());
    value_src_ptr = reinterpret_cast<PTYPE*>(value_src.data_ptr<float>());
    value_dst_ptr = reinterpret_cast<PTYPE*>(value_dst.data_ptr<float>());
#else
    if (key_src.dtype() == torch::kFloat16) {     // fp16
        key_src_ptr = reinterpret_cast<PTYPE*>(key_src.data_ptr<at::Half>());
        key_dst_ptr = reinterpret_cast<PTYPE*>(key_dst.data_ptr<at::Half>());
        value_src_ptr = reinterpret_cast<PTYPE*>(value_src.data_ptr<at::Half>());
        value_dst_ptr = reinterpret_cast<PTYPE*>(value_dst.data_ptr<at::Half>());
    } else {    // bf16
        key_src_ptr = reinterpret_cast<PTYPE*>(key_src.data_ptr<at::BFloat16>());
        key_dst_ptr = reinterpret_cast<PTYPE*>(key_dst.data_ptr<at::BFloat16>());
        value_src_ptr = reinterpret_cast<PTYPE*>(value_src.data_ptr<at::BFloat16>());
        value_dst_ptr = reinterpret_cast<PTYPE*>(value_dst.data_ptr<at::BFloat16>());
    }
#endif
    
    int* src_offsets_ptr = reinterpret_cast<int*>(src_offsets.data_ptr<int>());
    int* dst_cumsum_ptr = reinterpret_cast<int*>(dst_cumsum.data_ptr<int>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(gather_copy_append<PTYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);
    gather_copy_append<PTYPE><<<numBlocks, blockSize, maxSMBytes, stream>>>(
        key_src_ptr, 
        key_dst_ptr, 
        value_src_ptr,
        value_dst_ptr,
        src_length*128,
        dst_length*128,
        src_offsets_ptr,
        dst_cumsum_ptr,
        copy_cluster_num,
        max_cluster_size,
        n_centroids,
        copy_start);
}


// gather copy keys, values and from two src (streaming copy + cluster copy) and concat into one buffer
void gather_copy_cluster_and_concat_fuse(
    torch::Tensor& key_data1,    // input, shape (groups, num_vectors1, dim)
    torch::Tensor& key_data2,    // input, shape (groups, num_vectors2, dim)
    torch::Tensor& key_buffer,   // output, shape (groups, buffer_num_vectors, dim)
    torch::Tensor& value_data1,  // input, shape (groups, num_vectors1, dim)
    torch::Tensor& value_data2,  // input, shape (groups, num_vectors2, dim)
    torch::Tensor& value_buffer, // output, shape (groups, buffer_num_vectors, dim)
    
    torch::Tensor& cluster_cumsum,      // input, shape (groups, n_centroids)
    torch::Tensor& select_cluster_ids,  // input, shape (groups, index_size)
    torch::Tensor& valid_lengths,       // output, shape (groups,)

    int groups, 
    int data_length1,               // num_vectors1
    int data_length2,               // num_vectors2
    int buffer_length,              // buffer_num_vectors

    int nprobe_for_mem_alloc,       // used to compute shared memory size
    torch::Tensor& nprobe_tensor,   // input, zero-dim tensor, select cluster number
    torch::Tensor& copy_vectors1    // input, zero-dim tensor, number of vectors need to copy from data1
) {
    const int blockSize = BLOCK_SIZE_CP;
    const int numBlocks = groups * SPLIT_FACTOR;
    const int maxSMBytes = nprobe_for_mem_alloc * 3 * sizeof(int);

    int n_centroids = cluster_cumsum.size(1);
    int index_size = select_cluster_ids.size(1);

    PTYPE* key_data_ptr1;
    PTYPE* key_data_ptr2;
    PTYPE* key_buffer_ptr;
    PTYPE* value_data_ptr1;
    PTYPE* value_data_ptr2;
    PTYPE* value_buffer_ptr;

    // Cast fp32/fp16 data pointers to int4(16 Bytes)
#if DATA_BYTES == 4
    key_data_ptr1 = reinterpret_cast<PTYPE*>(key_data1.data_ptr<float>());
    key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<float>());
    key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<float>());
    value_data_ptr1 = reinterpret_cast<PTYPE*>(value_data1.data_ptr<float>());
    value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<float>());
    value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<float>());
#else
    if (key_data1.dtype() == torch::kFloat16) {     // fp16
        key_data_ptr1 = reinterpret_cast<PTYPE*>(key_data1.data_ptr<at::Half>());
        key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<at::Half>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::Half>());
        value_data_ptr1 = reinterpret_cast<PTYPE*>(value_data1.data_ptr<at::Half>());
        value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<at::Half>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::Half>());
    } else {    // bf16
        key_data_ptr1 = reinterpret_cast<PTYPE*>(key_data1.data_ptr<at::BFloat16>());
        key_data_ptr2 = reinterpret_cast<PTYPE*>(key_data2.data_ptr<at::BFloat16>());
        key_buffer_ptr = reinterpret_cast<PTYPE*>(key_buffer.data_ptr<at::BFloat16>());
        value_data_ptr1 = reinterpret_cast<PTYPE*>(value_data1.data_ptr<at::BFloat16>());
        value_data_ptr2 = reinterpret_cast<PTYPE*>(value_data2.data_ptr<at::BFloat16>());
        value_buffer_ptr = reinterpret_cast<PTYPE*>(value_buffer.data_ptr<at::BFloat16>());
    }
#endif
    
    int* cluster_cumsum_ptr = reinterpret_cast<int*>(cluster_cumsum.data_ptr<int32_t>());
    int64_t* select_cluster_ids_ptr = reinterpret_cast<int64_t*>(select_cluster_ids.data_ptr<int64_t>());
    int* valid_lengths_ptr = reinterpret_cast<int*>(valid_lengths.data_ptr<int32_t>());
    int* copy_vectors1_ptr = reinterpret_cast<int*>(copy_vectors1.data_ptr<int32_t>());
    int* nprobe_tensor_ptr = reinterpret_cast<int*>(nprobe_tensor.data_ptr<int32_t>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(concat_gather_copy_clusters_fuse<PTYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);
    concat_gather_copy_clusters_fuse<PTYPE><<<numBlocks, blockSize, maxSMBytes, stream>>>(
        key_data_ptr1, 
        key_data_ptr2,
        key_buffer_ptr, 
        value_data_ptr1,
        value_data_ptr2,
        value_buffer_ptr,
        data_length1*128,
        data_length2*128,
        buffer_length*128,
        cluster_cumsum_ptr,
        select_cluster_ids_ptr,
        n_centroids,
        index_size,
        buffer_length,
        nprobe_tensor_ptr,
        copy_vectors1_ptr,
        valid_lengths_ptr);
}




namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_retrieval_descriptors", &build_retrieval_descriptors, "Build hit/miss retrieval descriptors on GPU",
    py::arg("selected_clusters"), py::arg("cluster_offsets"), py::arg("cluster_sizes"),
    py::arg("cache_flags"), py::arg("cache_block_ids"),
    py::arg("miss_offsets"), py::arg("miss_sizes"), py::arg("miss_cumsum"), py::arg("miss_counts"),
    py::arg("hit_offsets"), py::arg("hit_sizes"), py::arg("hit_cumsum"), py::arg("hit_counts"),
    py::arg("valid_lengths"), py::arg("groups"), py::arg("nprobe"), py::arg("n_clusters"),
    py::arg("buffer_blocks"), py::arg("block_size"), py::arg("max_blocks_per_cluster"),
    py::arg("padding_cluster_id")),

    m.def("gather_copy_vectors", &gather_copy_vectors, "Gather copy keys, values and metadata in vector-level (CUDA)",
    py::arg("key_data"), py::arg("key_buffer"), py::arg("value_data"), py::arg("value_buffer"),
    py::arg("src_metadata"), py::arg("dst_metadata"), py::arg("data_offsets"),
    py::arg("groups"), py::arg("data_length"), py::arg("buffer_length"), py::arg("index_size"), py::arg("copy_start"), py::arg("copy_num")),

    m.def("gather_copy_and_concat", &gather_copy_and_concat, "Gather copy and concat from different sources (CUDA)",
    py::arg("key_data1"), py::arg("key_data2"), py::arg("key_data3"), py::arg("key_buffer"),
    py::arg("value_data1"), py::arg("value_data2"), py::arg("value_data3"), py::arg("value_buffer"),
    py::arg("data_offsets2"), py::arg("data_copy_sizes2"), py::arg("buffer_offsets2"), py::arg("copy_chunks2"),
    py::arg("data_offsets3"), py::arg("data_copy_sizes3"), py::arg("buffer_offsets3"), py::arg("copy_chunks3"),
    py::arg("valid_lengths"), py::arg("groups"), py::arg("data_length1"), py::arg("data_length2"), py::arg("data_length3"),
    py::arg("buffer_length"), py::arg("offset_length"), py::arg("copy_vectors1")),

    m.def("gather_copy_and_concat_retrieval", &gather_copy_and_concat_retrieval, "Gather copy retrieval hit/miss data and concat into one buffer (CUDA)",
    py::arg("key_data2"), py::arg("key_data3"), py::arg("key_buffer"),
    py::arg("value_data2"), py::arg("value_data3"), py::arg("value_buffer"),
    py::arg("data_offsets2"), py::arg("data_copy_sizes2"), py::arg("buffer_offsets2"), py::arg("copy_chunks2"),
    py::arg("data_offsets3"), py::arg("data_copy_sizes3"), py::arg("buffer_offsets3"), py::arg("copy_chunks3"),
    py::arg("valid_lengths"), py::arg("groups"), py::arg("data_length2"), py::arg("data_length3"),
    py::arg("buffer_length"), py::arg("offset_length")),

    m.def("gather_copy_and_scatter", &gather_copy_and_scatter, "Gather copy from src and scatter to dst (CUDA)",
    py::arg("key_data"), py::arg("key_buffer"), py::arg("value_data"), py::arg("value_buffer"),
    py::arg("s_offsets"), py::arg("copy_sizes"), py::arg("d_offsets"), py::arg("copy_chunks"),
    py::arg("groups"), py::arg("data_length"), py::arg("buffer_length"), py::arg("offset_length"), py::arg("s_copy_start")),

    m.def("reorganize_vectors", &reorganize_vectors, "Reorganize keys, values into clusters (CUDA)",
    py::arg("key_src"), py::arg("key_dst"), py::arg("value_src"), py::arg("value_dst"),
    py::arg("src_offsets"), py::arg("dst_cumsum"),
    py::arg("groups"), py::arg("copy_start")),

    m.def("gather_copy_cluster_and_concat_fuse", &gather_copy_cluster_and_concat_fuse, "Gather copy clusters and concat with fuse (CUDA)",
    py::arg("key_data1"), py::arg("key_data2"), py::arg("key_buffer"),
    py::arg("value_data1"), py::arg("value_data2"), py::arg("value_buffer"),
    py::arg("cluster_cumsum"), py::arg("select_cluster_ids"), py::arg("valid_lengths"),
    py::arg("groups"), py::arg("data_length1"), py::arg("data_length2"), py::arg("buffer_length"),
    py::arg("nprobe_for_mem_alloc"), py::arg("nprobe_tensor"), py::arg("copy_vectors1")),

    m.def("build_cache_aware_retrieval_plan", &build_cache_aware_retrieval_plan, "Build GPU cache-aware retrieval descriptors (CUDA)",
    py::arg("selected_clusters"), py::arg("cluster_offsets"), py::arg("cluster_sizes"),
    py::arg("cache_hash_keys"), py::arg("cache_hash_values"),
    py::arg("miss_offsets"), py::arg("miss_sizes"), py::arg("miss_cumsum"), py::arg("miss_counts"),
    py::arg("miss_page_keys"), py::arg("hit_offsets"), py::arg("hit_sizes"), py::arg("hit_cumsum"), py::arg("hit_counts"),
    py::arg("valid_lengths"), py::arg("groups"), py::arg("nprobe"), py::arg("n_clusters"),
    py::arg("buffer_blocks"), py::arg("block_size"), py::arg("hash_size"), py::arg("padding_cluster_id")),

    m.def("promote_cache_pages", &promote_cache_pages, "Promote missed retrieval pages into GPU cache (CUDA)",
    py::arg("execution_keys"), py::arg("cache_keys"), py::arg("execution_values"), py::arg("cache_values"),
    py::arg("miss_sizes"), py::arg("miss_cumsum"), py::arg("miss_counts"), py::arg("miss_page_keys"),
    py::arg("cache_hash_keys"), py::arg("cache_hash_values"), py::arg("cache_slot_keys"), py::arg("cache_next_slots"),
    py::arg("groups"), py::arg("execution_stride"), py::arg("cache_size"), py::arg("hash_size"),
    py::arg("buffer_blocks"), py::arg("static_len")),

    m.def("promote_cache_pages_from_list", &promote_cache_pages_from_list, "Promote missed retrieval pages from list KV into GPU cache (CUDA)",
    py::arg("list_keys"), py::arg("cache_keys"), py::arg("list_values"), py::arg("cache_values"),
    py::arg("miss_offsets"), py::arg("miss_sizes"), py::arg("miss_counts"), py::arg("miss_page_keys"),
    py::arg("cache_hash_keys"), py::arg("cache_hash_values"), py::arg("cache_slot_keys"), py::arg("cache_next_slots"),
    py::arg("groups"), py::arg("list_capacity"), py::arg("cache_size"), py::arg("hash_size"),
    py::arg("buffer_blocks"), py::arg("max_promote_pages_per_group"));

}
