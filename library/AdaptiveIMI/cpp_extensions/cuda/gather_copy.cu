#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>
#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#define CHUNK_SIZE 8        // number of vectors for one chunk
#define DATA_BYTES 2        // input data type, 2 use bf16/fp16, 4 use fp32
#define PTYPE int2          // int4 for 16 Bytes, int2 for 8 Bytes

#include "copy_kernel.cuh"

constexpr int IMI_HASH_EMPTY = -1;
constexpr int IMI_HASH_TOMBSTONE = -2;
constexpr int IMI_CLUSTER_NONE = 0;
constexpr int IMI_CLUSTER_COLD = 1;
constexpr int IMI_CLUSTER_HOT = 2;
constexpr int IMI_CLUSTER_PENDING = 3;

__device__ __forceinline__ int imi_pending_slot_code(int slot) {
    return -4 - slot;
}

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

__device__ __forceinline__ int imi_hash_find_pos(
    const int64_t* __restrict__ keys,
    int hash_size,
    int64_t key
) {
    int pos = static_cast<int>(imi_hash_i64(key) & static_cast<uint32_t>(hash_size - 1));
    for (int i = 0; i < hash_size; ++i) {
        int64_t cur = keys[pos];
        if (cur == key) {
            return pos;
        }
        if (cur == IMI_HASH_EMPTY) {
            return -1;
        }
        pos = (pos + 1) & (hash_size - 1);
    }
    return -1;
}

__device__ __forceinline__ int imi_hash_claim_pending_slot(
    int64_t* __restrict__ keys,
    int32_t* __restrict__ values,
    int hash_size,
    int64_t key,
    int slot
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
                atomicExch(values + dst, imi_pending_slot_code(slot));
                return dst;
            }
            pos = static_cast<int>(imi_hash_i64(key) & static_cast<uint32_t>(hash_size - 1));
            first_tombstone = -1;
            continue;
        }
        pos = (pos + 1) & (hash_size - 1);
    }
    return -1;
}

__device__ __forceinline__ void hc_remove_node(
    int cluster,
    int32_t* __restrict__ prev,
    int32_t* __restrict__ next,
    int32_t* __restrict__ head,
    int32_t* __restrict__ tail
) {
    int p = prev[cluster];
    int n = next[cluster];
    if (p >= 0) {
        next[p] = n;
    } else {
        *head = n;
    }
    if (n >= 0) {
        prev[n] = p;
    } else {
        *tail = p;
    }
    prev[cluster] = -1;
    next[cluster] = -1;
}

__device__ __forceinline__ void hc_push_front(
    int cluster,
    int32_t* __restrict__ prev,
    int32_t* __restrict__ next,
    int32_t* __restrict__ head,
    int32_t* __restrict__ tail
) {
    int old_head = *head;
    prev[cluster] = -1;
    next[cluster] = old_head;
    if (old_head >= 0) {
        prev[old_head] = cluster;
    } else {
        *tail = cluster;
    }
    *head = cluster;
}

__device__ __forceinline__ int hc_cluster_pages(
    const int32_t* __restrict__ cluster_sizes,
    int cluster,
    int n_clusters,
    int page_size
) {
    if (cluster < 0 || cluster >= n_clusters) {
        return 0;
    }
    int size = cluster_sizes[cluster];
    return size > 0 ? (size + page_size - 1) / page_size : 0;
}

__device__ __forceinline__ void hc_release_cluster_pages(
    int cluster,
    int pages,
    int64_t* __restrict__ hash_keys,
    int32_t* __restrict__ hash_values,
    int hash_size,
    int64_t* __restrict__ slot_keys,
    int32_t* __restrict__ free_slots,
    int32_t* __restrict__ free_count,
    int cache_size
) {
    for (int page = 0; page < pages; ++page) {
        int64_t key = (static_cast<int64_t>(cluster) << 32) | static_cast<uint32_t>(page);
        int pos = imi_hash_find_pos(hash_keys, hash_size, key);
        if (pos < 0) {
            continue;
        }
        int slot = hash_values[pos];
        hash_values[pos] = IMI_HASH_EMPTY;
        atomicExch(
            reinterpret_cast<unsigned long long*>(hash_keys + pos),
            static_cast<unsigned long long>(static_cast<int64_t>(IMI_HASH_TOMBSTONE))
        );
        if (slot >= 0 && slot < cache_size) {
            slot_keys[slot] = IMI_HASH_EMPTY;
            int free_idx = min(*free_count, cache_size - 1);
            free_slots[free_idx] = slot;
            *free_count = min(*free_count + 1, cache_size);
        }
    }
}

__device__ __forceinline__ bool hc_evict_cold_tail(
    int32_t* __restrict__ cold_head,
    int32_t* __restrict__ cold_tail,
    int32_t* __restrict__ prev,
    int32_t* __restrict__ next,
    int8_t* __restrict__ state,
    int32_t* __restrict__ cold_pages,
    const int32_t* __restrict__ cluster_sizes,
    int n_clusters,
    int page_size,
    int64_t* __restrict__ hash_keys,
    int32_t* __restrict__ hash_values,
    int hash_size,
    int64_t* __restrict__ slot_keys,
    int32_t* __restrict__ free_slots,
    int32_t* __restrict__ free_count,
    int cache_size
) {
    int victim = *cold_tail;
    if (victim < 0) {
        return false;
    }
    int pages = hc_cluster_pages(cluster_sizes, victim, n_clusters, page_size);
    hc_remove_node(victim, prev, next, cold_head, cold_tail);
    state[victim] = IMI_CLUSTER_NONE;
    *cold_pages = max(0, *cold_pages - pages);
    hc_release_cluster_pages(
        victim, pages, hash_keys, hash_values, hash_size, slot_keys,
        free_slots, free_count, cache_size
    );
    return true;
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
    int32_t* __restrict__ miss_cluster_cumsum,
    int32_t* __restrict__ miss_counts,
    int32_t* __restrict__ miss_copy_offsets,
    int32_t* __restrict__ miss_copy_sizes,
    int32_t* __restrict__ miss_copy_cumsum,
    int32_t* __restrict__ miss_copy_counts,
    int32_t* __restrict__ cluster_copy_offsets,
    int32_t* __restrict__ cluster_copy_sizes,
    int32_t* __restrict__ cluster_copy_cumsum,
    int32_t* __restrict__ cluster_copy_counts,
    int64_t* __restrict__ miss_page_keys,
    int32_t* __restrict__ admit_starts,
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
    int32_t* row_miss_cluster_cumsum = miss_cluster_cumsum + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_miss_copy_offsets = miss_copy_offsets + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_miss_copy_sizes = miss_copy_sizes + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_miss_copy_cumsum = miss_copy_cumsum + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_cluster_copy_offsets = cluster_copy_offsets + static_cast<int64_t>(row) * nprobe;
    int32_t* row_cluster_copy_sizes = cluster_copy_sizes + static_cast<int64_t>(row) * nprobe;
    int32_t* row_cluster_copy_cumsum = cluster_copy_cumsum + static_cast<int64_t>(row) * nprobe;
    int64_t* row_miss_page_keys = miss_page_keys + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_admit_starts = admit_starts + static_cast<int64_t>(row) * nprobe;
    int32_t* row_hit_offsets = hit_offsets + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_hit_sizes = hit_sizes + static_cast<int64_t>(row) * buffer_blocks;
    int32_t* row_hit_cumsum = hit_cumsum + static_cast<int64_t>(row) * buffer_blocks;

    __shared__ unsigned long long s_miss_state;
    __shared__ unsigned long long s_miss_copy_state;
    __shared__ unsigned long long s_cluster_copy_state;
    __shared__ unsigned long long s_hit_state;
    __shared__ int s_considered_blocks;
    if (threadIdx.x == 0) {
        s_miss_state = 0;
        s_miss_copy_state = 0;
        s_cluster_copy_state = 0;
        s_hit_state = 0;
        s_considered_blocks = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < nprobe; i += blockDim.x) {
        row_admit_starts[i] = -1;
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
        unsigned long long cluster_copy_inc = (static_cast<unsigned long long>(cluster_size) << 32) | 1U;
        unsigned long long cluster_copy_old = atomicAdd(&s_cluster_copy_state, cluster_copy_inc);
        int cluster_copy_idx = static_cast<int>(cluster_copy_old & 0xffffffffULL);
        int cluster_copy_dst = static_cast<int>(cluster_copy_old >> 32);
        if (cluster_copy_idx < nprobe) {
            row_cluster_copy_offsets[cluster_copy_idx] = list_start;
            row_cluster_copy_sizes[cluster_copy_idx] = cluster_size;
            row_cluster_copy_cumsum[cluster_copy_idx] = cluster_copy_dst;
        }
        int local_miss_blocks = 0;
        int local_hit_blocks = 0;
        int local_miss_tokens = 0;
        int local_hit_tokens = 0;
        int local_miss_runs = 0;
        bool prev_miss = false;
        unsigned long long hit_mask = 0;
        int cached_slots[16];
        const bool use_slot_cache = cluster_blocks <= 16;
        const bool use_page_mask = cluster_blocks <= 64;
        for (int page = 0; page < cluster_blocks; ++page) {
            int size = (page + 1 == cluster_blocks) ? (cluster_size - page * block_size) : block_size;
            int64_t key = (static_cast<int64_t>(cluster) << 32) | static_cast<uint32_t>(page);
            int slot = imi_hash_lookup(row_hash_keys, row_hash_values, hash_size, key);
            if (use_slot_cache) {
                cached_slots[page] = slot;
            }
            if (slot >= 0) {
                if (use_page_mask) {
                    hit_mask |= (1ULL << page);
                }
                local_hit_blocks++;
                local_hit_tokens += size;
                prev_miss = false;
            } else {
                if (!prev_miss) {
                    local_miss_runs++;
                }
                prev_miss = true;
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
            row_admit_starts[i] = static_cast<int>(miss_old & 0xffffffffULL);
        }
        unsigned long long miss_copy_old = 0;
        if (local_miss_runs > 0) {
            unsigned long long inc = (static_cast<unsigned long long>(local_miss_tokens) << 32) |
                                     static_cast<unsigned int>(local_miss_runs);
            miss_copy_old = atomicAdd(&s_miss_copy_state, inc);
        }
        if (local_hit_blocks > 0) {
            unsigned long long inc = (static_cast<unsigned long long>(local_hit_tokens) << 32) |
                                     static_cast<unsigned int>(local_hit_blocks);
            hit_old = atomicAdd(&s_hit_state, inc);
        }

        int miss_idx = static_cast<int>(miss_old & 0xffffffffULL);
        int miss_dst = static_cast<int>(miss_old >> 32);
        int miss_copy_idx = static_cast<int>(miss_copy_old & 0xffffffffULL);
        int active_copy_idx = -1;
        int hit_idx = static_cast<int>(hit_old & 0xffffffffULL);
        int hit_dst = static_cast<int>(hit_old >> 32);
        for (int page = 0; page < cluster_blocks; ++page) {
            int size = (page + 1 == cluster_blocks) ? (cluster_size - page * block_size) : block_size;
            int64_t key = (static_cast<int64_t>(cluster) << 32) | static_cast<uint32_t>(page);
            int slot = -1;
            if (use_slot_cache) {
                slot = cached_slots[page];
            } else if (!use_page_mask || ((hit_mask >> page) & 1ULL)) {
                slot = imi_hash_lookup(row_hash_keys, row_hash_values, hash_size, key);
            }
            if (slot >= 0) {
                row_hit_offsets[hit_idx] = slot;
                row_hit_sizes[hit_idx] = size;
                row_hit_cumsum[hit_idx] = hit_dst;
                hit_idx++;
                hit_dst += size;
                active_copy_idx = -1;
            } else {
                if (active_copy_idx < 0) {
                    active_copy_idx = miss_copy_idx++;
                    row_miss_copy_offsets[active_copy_idx] = list_start + page * block_size;
                    row_miss_copy_sizes[active_copy_idx] = 0;
                    // The foreground run descriptor must write to the same
                    // execution-buffer layout described by the page-level miss
                    // descriptor. Cache admission later reads page-level
                    // offsets from this buffer.
                    row_miss_copy_cumsum[active_copy_idx] = miss_dst;
                }
                row_miss_copy_sizes[active_copy_idx] += size;
                row_miss_offsets[miss_idx] = list_start + page * block_size;
                row_miss_sizes[miss_idx] = size;
                row_miss_cumsum[miss_idx] = miss_dst;
                row_miss_cluster_cumsum[miss_idx] = cluster_copy_dst + page * block_size;
                row_miss_page_keys[miss_idx] = key;
                miss_idx++;
                miss_dst += size;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int miss_blocks = min(static_cast<int>(s_miss_state & 0xffffffffULL), buffer_blocks);
        int miss_copy_blocks = min(static_cast<int>(s_miss_copy_state & 0xffffffffULL), buffer_blocks);
        int hit_blocks = min(static_cast<int>(s_hit_state & 0xffffffffULL), buffer_blocks);
        miss_counts[row] = miss_blocks;
        miss_copy_counts[row] = miss_copy_blocks;
        cluster_copy_counts[row] = min(static_cast<int>(s_cluster_copy_state & 0xffffffffULL), nprobe);
        hit_counts[row] = hit_blocks;
        valid_lengths[row] = static_cast<int>(s_miss_state >> 32) + static_cast<int>(s_hit_state >> 32);
    }
}

__global__ void plan_hot_cold_cache_promotion_kernel(
    const int32_t* __restrict__ miss_dst_offsets,
    const int32_t* __restrict__ miss_sizes,
    const int32_t* __restrict__ miss_counts,
    const int64_t* __restrict__ miss_page_keys,
    const int32_t* __restrict__ admit_starts,
    int32_t* __restrict__ promote_src_offsets,
    int32_t* __restrict__ promote_dst_slots,
    int32_t* __restrict__ promote_sizes,
    int32_t* __restrict__ promote_hash_positions,
    int64_t* __restrict__ promote_page_keys,
    int32_t* __restrict__ promote_counts,
    const int32_t* __restrict__ hit_offsets,
    const int32_t* __restrict__ hit_counts,
    const int32_t* __restrict__ cluster_sizes,
    int64_t* __restrict__ hash_keys,
    int32_t* __restrict__ hash_values,
    int64_t* __restrict__ slot_keys,
    int8_t* __restrict__ cluster_state,
    int32_t* __restrict__ cluster_prev,
    int32_t* __restrict__ cluster_next,
    int32_t* __restrict__ hot_head,
    int32_t* __restrict__ hot_tail,
    int32_t* __restrict__ cold_head,
    int32_t* __restrict__ cold_tail,
    int32_t* __restrict__ hot_pages,
    int32_t* __restrict__ cold_pages,
    int32_t* __restrict__ free_slots,
    int32_t* __restrict__ free_counts,
    int static_len,
    int groups,
    int execution_stride,
    int cache_size,
    int hash_size,
    int buffer_blocks,
    int nprobe,
    int n_clusters,
    int page_size,
    int hot_capacity_pages,
    int cold_capacity_pages,
    int max_promote_pages_per_group
) {
    const int row = blockIdx.x;
    if (row >= groups || threadIdx.x != 0) {
        return;
    }

    const int row_buffer_base = row * buffer_blocks;
    const int row_probe_base = row * nprobe;
    const int64_t row_hash_base = static_cast<int64_t>(row) * hash_size;
    const int64_t row_cache_base = static_cast<int64_t>(row) * cache_size;
    const int64_t row_cluster_base = static_cast<int64_t>(row) * n_clusters;

    int64_t* row_hash_keys = hash_keys + row_hash_base;
    int32_t* row_hash_values = hash_values + row_hash_base;
    int64_t* row_slot_keys = slot_keys + row_cache_base;
    int8_t* row_state = cluster_state + row_cluster_base;
    int32_t* row_prev = cluster_prev + row_cluster_base;
    int32_t* row_next = cluster_next + row_cluster_base;
    const int32_t* row_cluster_sizes = cluster_sizes + row_cluster_base;
    int32_t* row_free_slots = free_slots + row_cache_base;

    int32_t* row_hot_head = hot_head + row;
    int32_t* row_hot_tail = hot_tail + row;
    int32_t* row_cold_head = cold_head + row;
    int32_t* row_cold_tail = cold_tail + row;
    int32_t* row_hot_pages = hot_pages + row;
    int32_t* row_cold_pages = cold_pages + row;
    int32_t* row_free_count = free_counts + row;

    promote_counts[row] = 0;

    // Hits only mutate cluster-level LRU metadata. They stay in this metadata
    // kernel so the page-copy kernel can be purely parallel D2D work.
    int last_hit_cluster = -1;
    const int hit_count = hit_counts[row];
    for (int idx = 0; idx < hit_count; ++idx) {
        int slot = hit_offsets[row_buffer_base + idx];
        if (slot < 0 || slot >= cache_size) {
            continue;
        }
        int64_t key = row_slot_keys[slot];
        int cluster = static_cast<int>(key >> 32);
        if (cluster == last_hit_cluster || cluster < 0 || cluster >= n_clusters) {
            continue;
        }
        last_hit_cluster = cluster;
        int pages = hc_cluster_pages(row_cluster_sizes, cluster, n_clusters, page_size);
        if (pages <= 0) {
            continue;
        }
        if (row_state[cluster] == IMI_CLUSTER_HOT) {
            hc_remove_node(cluster, row_prev, row_next, row_hot_head, row_hot_tail);
            hc_push_front(cluster, row_prev, row_next, row_hot_head, row_hot_tail);
        } else if (row_state[cluster] == IMI_CLUSTER_COLD) {
            hc_remove_node(cluster, row_prev, row_next, row_cold_head, row_cold_tail);
            *row_cold_pages = max(0, *row_cold_pages - pages);
            row_state[cluster] = IMI_CLUSTER_HOT;
            hc_push_front(cluster, row_prev, row_next, row_hot_head, row_hot_tail);
            *row_hot_pages += pages;
        }

        while (*row_hot_pages > hot_capacity_pages && *row_hot_tail >= 0) {
            int demote = *row_hot_tail;
            int demote_pages = hc_cluster_pages(row_cluster_sizes, demote, n_clusters, page_size);
            hc_remove_node(demote, row_prev, row_next, row_hot_head, row_hot_tail);
            *row_hot_pages = max(0, *row_hot_pages - demote_pages);
            row_state[demote] = IMI_CLUSTER_COLD;
            hc_push_front(demote, row_prev, row_next, row_cold_head, row_cold_tail);
            *row_cold_pages += demote_pages;
        }
        while (*row_cold_pages > cold_capacity_pages) {
            if (!hc_evict_cold_tail(
                row_cold_head, row_cold_tail, row_prev, row_next, row_state, row_cold_pages,
                row_cluster_sizes, n_clusters, page_size, row_hash_keys, row_hash_values,
                hash_size, row_slot_keys, row_free_slots, row_free_count, cache_size
            )) {
                break;
            }
        }
    }

    const int available_misses = miss_counts[row];
    if (available_misses <= 0 || max_promote_pages_per_group == 0) {
        return;
    }

    int promoted_pages = 0;
    int probe_idx = 0;
    int plan_count = 0;
    while (probe_idx < nprobe) {
        int selected_miss_idx = -1;
        int selected_run_end = -1;
        int selected_cluster = -1;
        int selected_run_pages = 0;
        while (probe_idx < nprobe) {
            int miss_idx = admit_starts[row_probe_base + probe_idx];
            ++probe_idx;
            if (miss_idx < 0 || miss_idx >= available_misses) {
                continue;
            }
            int64_t first_key = miss_page_keys[row_buffer_base + miss_idx];
            int cluster = static_cast<int>(first_key >> 32);
            int run_end = miss_idx + 1;
            while (run_end < available_misses) {
                int64_t key = miss_page_keys[row_buffer_base + run_end];
                if (static_cast<int>(key >> 32) != cluster) {
                    break;
                }
                ++run_end;
            }
            int run_pages = run_end - miss_idx;
            if (cluster < 0 || cluster >= n_clusters || row_state[cluster] != IMI_CLUSTER_NONE) {
                continue;
            }
            int full_pages = hc_cluster_pages(row_cluster_sizes, cluster, n_clusters, page_size);
            if (full_pages != run_pages || run_pages <= 0 || run_pages > cache_size) {
                continue;
            }
            if (max_promote_pages_per_group >= 0 && promoted_pages + run_pages > max_promote_pages_per_group) {
                probe_idx = nprobe;
                break;
            }
            while (*row_free_count < run_pages) {
                if (!hc_evict_cold_tail(
                    row_cold_head, row_cold_tail, row_prev, row_next, row_state, row_cold_pages,
                    row_cluster_sizes, n_clusters, page_size, row_hash_keys, row_hash_values,
                    hash_size, row_slot_keys, row_free_slots, row_free_count, cache_size
                )) {
                    break;
                }
            }
            if (*row_free_count >= run_pages && plan_count + run_pages <= buffer_blocks) {
                selected_miss_idx = miss_idx;
                selected_run_end = run_end;
                selected_cluster = cluster;
                selected_run_pages = run_pages;
                break;
            }
        }

        int miss_idx = selected_miss_idx;
        int run_end = selected_run_end;
        int cluster = selected_cluster;
        int run_pages = selected_run_pages;
        if (miss_idx < 0) {
            break;
        }

        bool cluster_ok = true;
        int claimed_pages = 0;
        for (int idx = miss_idx; idx < run_end; ++idx) {
            int64_t key = miss_page_keys[row_buffer_base + idx];
            int src_start = static_len + miss_dst_offsets[row_buffer_base + idx];
            int size = miss_sizes[row_buffer_base + idx];
            int free_idx = *row_free_count - 1;
            int slot = free_idx >= 0 ? row_free_slots[free_idx] : -1;
            int hash_pos = -1;
            if (slot >= 0) {
                *row_free_count = free_idx;
                hash_pos = imi_hash_claim_pending_slot(
                    row_hash_keys,
                    row_hash_values,
                    hash_size,
                    key,
                    slot
                );
                if (hash_pos < 0) {
                    row_free_slots[*row_free_count] = slot;
                    *row_free_count = min(*row_free_count + 1, cache_size);
                } else {
                    row_slot_keys[slot] = key;
                }
            }
            if (slot < 0 || hash_pos < 0 || size <= 0) {
                cluster_ok = false;
                break;
            }
            int out_idx = row_buffer_base + plan_count;
            promote_src_offsets[out_idx] = src_start;
            promote_dst_slots[out_idx] = slot;
            promote_sizes[out_idx] = size;
            promote_hash_positions[out_idx] = hash_pos;
            promote_page_keys[out_idx] = key;
            ++plan_count;
            ++claimed_pages;
        }

        if (cluster_ok) {
            row_state[cluster] = IMI_CLUSTER_PENDING;
            promoted_pages += run_pages;
            promote_counts[row] = plan_count;
        } else {
            plan_count -= claimed_pages;
            promote_counts[row] = plan_count;
            row_state[cluster] = IMI_CLUSTER_NONE;
            hc_release_cluster_pages(
                cluster, claimed_pages, row_hash_keys, row_hash_values, hash_size,
                row_slot_keys, row_free_slots, row_free_count, cache_size
            );
        }
    }
}

template <typename T>
__global__ void copy_promoted_pages_from_execution_kernel(
    const T* __restrict__ execution_keys,
    T* __restrict__ cache_keys,
    const T* __restrict__ execution_values,
    T* __restrict__ cache_values,
    const int32_t* __restrict__ promote_src_offsets,
    const int32_t* __restrict__ promote_dst_slots,
    const int32_t* __restrict__ promote_sizes,
    const int32_t* __restrict__ promote_hash_positions,
    const int32_t* __restrict__ promote_counts,
    int32_t* __restrict__ hash_values,
    int groups,
    int execution_stride,
    int cache_size,
    int hash_size,
    int buffer_blocks,
    int page_size
) {
    const int page_idx = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= groups || page_idx >= promote_counts[row]) {
        return;
    }

    const int row_buffer_base = row * buffer_blocks;
    const int idx = row_buffer_base + page_idx;
    const int src_start = promote_src_offsets[idx];
    const int slot = promote_dst_slots[idx];
    const int size = promote_sizes[idx];
    const int hash_pos = promote_hash_positions[idx];
    if (slot < 0 || slot >= cache_size || size <= 0 || hash_pos < 0 || hash_pos >= hash_size) {
        return;
    }

    const int copy_elems = size * VECTOR_SIZE_CP;
    const int64_t execution_base = (static_cast<int64_t>(row) * execution_stride + src_start) * VECTOR_SIZE_CP;
    const int64_t cache_base = (static_cast<int64_t>(row) * cache_size * page_size + slot * page_size) * VECTOR_SIZE_CP;
    for (int elem = threadIdx.x; elem < copy_elems; elem += blockDim.x) {
        cache_keys[cache_base + elem] = execution_keys[execution_base + elem];
        cache_values[cache_base + elem] = execution_values[execution_base + elem];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicExch(hash_values + static_cast<int64_t>(row) * hash_size + hash_pos, slot);
    }
}

__global__ void finalize_hot_cold_cache_promotion_kernel(
    const int64_t* __restrict__ promote_page_keys,
    const int32_t* __restrict__ promote_counts,
    const int32_t* __restrict__ cluster_sizes,
    int64_t* __restrict__ hash_keys,
    int32_t* __restrict__ hash_values,
    int64_t* __restrict__ slot_keys,
    int8_t* __restrict__ cluster_state,
    int32_t* __restrict__ cluster_prev,
    int32_t* __restrict__ cluster_next,
    int32_t* __restrict__ cold_head,
    int32_t* __restrict__ cold_tail,
    int32_t* __restrict__ cold_pages,
    int32_t* __restrict__ free_slots,
    int32_t* __restrict__ free_counts,
    int groups,
    int cache_size,
    int hash_size,
    int buffer_blocks,
    int n_clusters,
    int page_size,
    int cold_capacity_pages
) {
    const int row = blockIdx.x;
    if (row >= groups || threadIdx.x != 0) {
        return;
    }

    const int row_buffer_base = row * buffer_blocks;
    const int64_t row_hash_base = static_cast<int64_t>(row) * hash_size;
    const int64_t row_cache_base = static_cast<int64_t>(row) * cache_size;
    const int64_t row_cluster_base = static_cast<int64_t>(row) * n_clusters;

    int64_t* row_hash_keys = hash_keys + row_hash_base;
    int32_t* row_hash_values = hash_values + row_hash_base;
    int64_t* row_slot_keys = slot_keys + row_cache_base;
    int8_t* row_state = cluster_state + row_cluster_base;
    int32_t* row_prev = cluster_prev + row_cluster_base;
    int32_t* row_next = cluster_next + row_cluster_base;
    const int32_t* row_cluster_sizes = cluster_sizes + row_cluster_base;
    int32_t* row_free_slots = free_slots + row_cache_base;

    int32_t* row_cold_head = cold_head + row;
    int32_t* row_cold_tail = cold_tail + row;
    int32_t* row_cold_pages = cold_pages + row;
    int32_t* row_free_count = free_counts + row;

    const int count = promote_counts[row];
    int idx = 0;
    while (idx < count) {
        int64_t key = promote_page_keys[row_buffer_base + idx];
        int cluster = static_cast<int>(key >> 32);
        int run_end = idx + 1;
        while (run_end < count) {
            int64_t next_key = promote_page_keys[row_buffer_base + run_end];
            if (static_cast<int>(next_key >> 32) != cluster) {
                break;
            }
            ++run_end;
        }
        int run_pages = run_end - idx;
        if (cluster >= 0 && cluster < n_clusters && row_state[cluster] == IMI_CLUSTER_PENDING) {
            row_state[cluster] = IMI_CLUSTER_COLD;
            hc_push_front(cluster, row_prev, row_next, row_cold_head, row_cold_tail);
            *row_cold_pages += run_pages;
            while (*row_cold_pages > cold_capacity_pages) {
                if (!hc_evict_cold_tail(
                    row_cold_head, row_cold_tail, row_prev, row_next, row_state, row_cold_pages,
                    row_cluster_sizes, n_clusters, page_size, row_hash_keys, row_hash_values,
                    hash_size, row_slot_keys, row_free_slots, row_free_count, cache_size
                )) {
                    break;
                }
            }
        }
        idx = run_end;
    }
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
    torch::Tensor& miss_cluster_cumsum,
    torch::Tensor& miss_counts,
    torch::Tensor& miss_copy_offsets,
    torch::Tensor& miss_copy_sizes,
    torch::Tensor& miss_copy_cumsum,
    torch::Tensor& miss_copy_counts,
    torch::Tensor& cluster_copy_offsets,
    torch::Tensor& cluster_copy_sizes,
    torch::Tensor& cluster_copy_cumsum,
    torch::Tensor& cluster_copy_counts,
    torch::Tensor& miss_page_keys,
    torch::Tensor& admit_starts,
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
    TORCH_CHECK(miss_cluster_cumsum.is_cuda(), "miss_cluster_cumsum must be CUDA");
    TORCH_CHECK(miss_counts.is_cuda(), "miss_counts must be CUDA");
    TORCH_CHECK(miss_copy_offsets.is_cuda(), "miss_copy_offsets must be CUDA");
    TORCH_CHECK(miss_copy_sizes.is_cuda(), "miss_copy_sizes must be CUDA");
    TORCH_CHECK(miss_copy_cumsum.is_cuda(), "miss_copy_cumsum must be CUDA");
    TORCH_CHECK(miss_copy_counts.is_cuda(), "miss_copy_counts must be CUDA");
    TORCH_CHECK(cluster_copy_offsets.is_cuda(), "cluster_copy_offsets must be CUDA");
    TORCH_CHECK(cluster_copy_sizes.is_cuda(), "cluster_copy_sizes must be CUDA");
    TORCH_CHECK(cluster_copy_cumsum.is_cuda(), "cluster_copy_cumsum must be CUDA");
    TORCH_CHECK(cluster_copy_counts.is_cuda(), "cluster_copy_counts must be CUDA");
    TORCH_CHECK(miss_page_keys.is_cuda(), "miss_page_keys must be CUDA");
    TORCH_CHECK(admit_starts.is_cuda(), "admit_starts must be CUDA");
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
        reinterpret_cast<int32_t*>(miss_cluster_cumsum.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_counts.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_copy_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_copy_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_copy_cumsum.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_copy_counts.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cluster_copy_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cluster_copy_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cluster_copy_cumsum.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cluster_copy_counts.data_ptr<int32_t>()),
        reinterpret_cast<int64_t*>(miss_page_keys.data_ptr<int64_t>()),
        reinterpret_cast<int32_t*>(admit_starts.data_ptr<int32_t>()),
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

void promote_hot_cold_cache_pages_from_execution(
    torch::Tensor& execution_keys,
    torch::Tensor& cache_keys,
    torch::Tensor& execution_values,
    torch::Tensor& cache_values,
    torch::Tensor& miss_dst_offsets,
    torch::Tensor& miss_sizes,
    torch::Tensor& miss_counts,
    torch::Tensor& miss_page_keys,
    torch::Tensor& admit_starts,
    torch::Tensor& promote_src_offsets,
    torch::Tensor& promote_dst_slots,
    torch::Tensor& promote_sizes,
    torch::Tensor& promote_hash_positions,
    torch::Tensor& promote_page_keys,
    torch::Tensor& promote_counts,
    torch::Tensor& hit_offsets,
    torch::Tensor& hit_counts,
    torch::Tensor& cluster_sizes,
    torch::Tensor& cache_hash_keys,
    torch::Tensor& cache_hash_values,
    torch::Tensor& cache_slot_keys,
    torch::Tensor& cluster_state,
    torch::Tensor& cluster_prev,
    torch::Tensor& cluster_next,
    torch::Tensor& hot_head,
    torch::Tensor& hot_tail,
    torch::Tensor& cold_head,
    torch::Tensor& cold_tail,
    torch::Tensor& hot_pages,
    torch::Tensor& cold_pages,
    torch::Tensor& free_slots,
    torch::Tensor& free_counts,
    int static_len,
    int groups,
    int execution_stride,
    int cache_size,
    int hash_size,
    int buffer_blocks,
    int nprobe,
    int n_clusters,
    int page_size,
    int hot_capacity_pages,
    int cold_capacity_pages,
    int max_promote_pages_per_group
) {
    TORCH_CHECK(execution_keys.is_cuda(), "execution_keys must be CUDA");
    TORCH_CHECK(execution_values.is_cuda(), "execution_values must be CUDA");
    TORCH_CHECK(cache_keys.is_cuda(), "cache_keys must be CUDA");
    TORCH_CHECK(cache_values.is_cuda(), "cache_values must be CUDA");
    TORCH_CHECK(miss_dst_offsets.is_cuda(), "miss_dst_offsets must be CUDA");
    TORCH_CHECK(miss_sizes.is_cuda(), "miss_sizes must be CUDA");
    TORCH_CHECK(miss_counts.is_cuda(), "miss_counts must be CUDA");
    TORCH_CHECK(miss_page_keys.is_cuda(), "miss_page_keys must be CUDA");
    TORCH_CHECK(admit_starts.is_cuda(), "admit_starts must be CUDA");
    TORCH_CHECK(promote_src_offsets.is_cuda(), "promote_src_offsets must be CUDA");
    TORCH_CHECK(promote_dst_slots.is_cuda(), "promote_dst_slots must be CUDA");
    TORCH_CHECK(promote_sizes.is_cuda(), "promote_sizes must be CUDA");
    TORCH_CHECK(promote_hash_positions.is_cuda(), "promote_hash_positions must be CUDA");
    TORCH_CHECK(promote_page_keys.is_cuda(), "promote_page_keys must be CUDA");
    TORCH_CHECK(promote_counts.is_cuda(), "promote_counts must be CUDA");
    TORCH_CHECK(hit_offsets.is_cuda(), "hit_offsets must be CUDA");
    TORCH_CHECK(hit_counts.is_cuda(), "hit_counts must be CUDA");
    TORCH_CHECK(cluster_sizes.is_cuda(), "cluster_sizes must be CUDA");
    TORCH_CHECK(cache_hash_keys.is_cuda(), "cache_hash_keys must be CUDA");
    TORCH_CHECK(cache_hash_values.is_cuda(), "cache_hash_values must be CUDA");
    TORCH_CHECK(cache_slot_keys.is_cuda(), "cache_slot_keys must be CUDA");
    TORCH_CHECK(cluster_state.is_cuda(), "cluster_state must be CUDA");
    TORCH_CHECK(cluster_prev.is_cuda(), "cluster_prev must be CUDA");
    TORCH_CHECK(cluster_next.is_cuda(), "cluster_next must be CUDA");
    TORCH_CHECK(hot_head.is_cuda(), "hot_head must be CUDA");
    TORCH_CHECK(hot_tail.is_cuda(), "hot_tail must be CUDA");
    TORCH_CHECK(cold_head.is_cuda(), "cold_head must be CUDA");
    TORCH_CHECK(cold_tail.is_cuda(), "cold_tail must be CUDA");
    TORCH_CHECK(hot_pages.is_cuda(), "hot_pages must be CUDA");
    TORCH_CHECK(cold_pages.is_cuda(), "cold_pages must be CUDA");
    TORCH_CHECK(free_slots.is_cuda(), "free_slots must be CUDA");
    TORCH_CHECK(free_counts.is_cuda(), "free_counts must be CUDA");
    TORCH_CHECK((hash_size & (hash_size - 1)) == 0, "hash_size must be a power of two");

    const int blockSize = BLOCK_SIZE_CP;
    const auto stream = at::cuda::getCurrentCUDAStream();
    plan_hot_cold_cache_promotion_kernel<<<groups, 1, 0, stream>>>(
        reinterpret_cast<int32_t*>(miss_dst_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(miss_counts.data_ptr<int32_t>()),
        reinterpret_cast<int64_t*>(miss_page_keys.data_ptr<int64_t>()),
        reinterpret_cast<int32_t*>(admit_starts.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(promote_src_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(promote_dst_slots.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(promote_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(promote_hash_positions.data_ptr<int32_t>()),
        reinterpret_cast<int64_t*>(promote_page_keys.data_ptr<int64_t>()),
        reinterpret_cast<int32_t*>(promote_counts.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(hit_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(hit_counts.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cluster_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int64_t*>(cache_hash_keys.data_ptr<int64_t>()),
        reinterpret_cast<int32_t*>(cache_hash_values.data_ptr<int32_t>()),
        reinterpret_cast<int64_t*>(cache_slot_keys.data_ptr<int64_t>()),
        reinterpret_cast<int8_t*>(cluster_state.data_ptr<int8_t>()),
        reinterpret_cast<int32_t*>(cluster_prev.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cluster_next.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(hot_head.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(hot_tail.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cold_head.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cold_tail.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(hot_pages.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cold_pages.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(free_slots.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(free_counts.data_ptr<int32_t>()),
        static_len,
        groups,
        execution_stride,
        cache_size,
        hash_size,
        buffer_blocks,
        nprobe,
        n_clusters,
        page_size,
        hot_capacity_pages,
        cold_capacity_pages,
        max_promote_pages_per_group
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int copy_blocks = buffer_blocks;
    if (max_promote_pages_per_group > 0) {
        copy_blocks = std::min(buffer_blocks, max_promote_pages_per_group);
    }
    copy_blocks = std::max(copy_blocks, 1);
    const dim3 copyGrid(copy_blocks, groups);
#if DATA_BYTES == 4
    copy_promoted_pages_from_execution_kernel<float><<<copyGrid, blockSize, 0, stream>>>(
        reinterpret_cast<float*>(execution_keys.data_ptr<float>()),
        reinterpret_cast<float*>(cache_keys.data_ptr<float>()),
        reinterpret_cast<float*>(execution_values.data_ptr<float>()),
        reinterpret_cast<float*>(cache_values.data_ptr<float>()),
#else
    if (execution_keys.dtype() == torch::kFloat16) {
        copy_promoted_pages_from_execution_kernel<PTYPE><<<copyGrid, blockSize, 0, stream>>>(
            reinterpret_cast<PTYPE*>(execution_keys.data_ptr<at::Half>()),
            reinterpret_cast<PTYPE*>(cache_keys.data_ptr<at::Half>()),
            reinterpret_cast<PTYPE*>(execution_values.data_ptr<at::Half>()),
            reinterpret_cast<PTYPE*>(cache_values.data_ptr<at::Half>()),
            reinterpret_cast<int32_t*>(promote_src_offsets.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(promote_dst_slots.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(promote_sizes.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(promote_hash_positions.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(promote_counts.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(cache_hash_values.data_ptr<int32_t>()),
            groups,
            execution_stride,
            cache_size,
            hash_size,
            buffer_blocks,
            page_size
        );
    } else {
        copy_promoted_pages_from_execution_kernel<PTYPE><<<copyGrid, blockSize, 0, stream>>>(
            reinterpret_cast<PTYPE*>(execution_keys.data_ptr<at::BFloat16>()),
            reinterpret_cast<PTYPE*>(cache_keys.data_ptr<at::BFloat16>()),
            reinterpret_cast<PTYPE*>(execution_values.data_ptr<at::BFloat16>()),
            reinterpret_cast<PTYPE*>(cache_values.data_ptr<at::BFloat16>()),
#endif
            reinterpret_cast<int32_t*>(promote_src_offsets.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(promote_dst_slots.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(promote_sizes.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(promote_hash_positions.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(promote_counts.data_ptr<int32_t>()),
            reinterpret_cast<int32_t*>(cache_hash_values.data_ptr<int32_t>()),
            groups,
            execution_stride,
            cache_size,
            hash_size,
            buffer_blocks,
            page_size
        );
#if DATA_BYTES != 4
    }
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    finalize_hot_cold_cache_promotion_kernel<<<groups, 1, 0, stream>>>(
        reinterpret_cast<int64_t*>(promote_page_keys.data_ptr<int64_t>()),
        reinterpret_cast<int32_t*>(promote_counts.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cluster_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int64_t*>(cache_hash_keys.data_ptr<int64_t>()),
        reinterpret_cast<int32_t*>(cache_hash_values.data_ptr<int32_t>()),
        reinterpret_cast<int64_t*>(cache_slot_keys.data_ptr<int64_t>()),
        reinterpret_cast<int8_t*>(cluster_state.data_ptr<int8_t>()),
        reinterpret_cast<int32_t*>(cluster_prev.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cluster_next.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cold_head.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cold_tail.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(cold_pages.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(free_slots.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(free_counts.data_ptr<int32_t>()),
        groups,
        cache_size,
        hash_size,
        buffer_blocks,
        n_clusters,
        page_size,
        cold_capacity_pages
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <typename T>
__global__ void copy_static_chunks_kernel(
    const T* __restrict__ src_keys,
    const T* __restrict__ src_values,
    T* __restrict__ dst_keys,
    T* __restrict__ dst_values,
    int* __restrict__ valid_lengths,
    const int* __restrict__ static_len_ptr,
    int groups,
    int src_vectors_per_group,
    int dst_vectors_per_group,
    int chunks_per_group
) {
    int64_t work = static_cast<int64_t>(blockIdx.x);
    int row = static_cast<int>(work / chunks_per_group);
    int chunk = static_cast<int>(work - static_cast<int64_t>(row) * chunks_per_group);
    if (row >= groups) {
        return;
    }

    int static_len = *static_len_ptr;
    if (chunk == 0 && threadIdx.x == 0) {
        valid_lengths[row] += static_len;
    }
    int vector_start = chunk * CHUNK_SIZE;
    if (vector_start >= static_len) {
        return;
    }
    int remaining = static_len - vector_start;
    int copy_vectors = remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE;
    int copy_elems = copy_vectors * VECTOR_SIZE_CP;

    int64_t src_base = (static_cast<int64_t>(row) * src_vectors_per_group + vector_start) * VECTOR_SIZE_CP;
    int64_t dst_base = (static_cast<int64_t>(row) * dst_vectors_per_group + vector_start) * VECTOR_SIZE_CP;
    for (int idx = threadIdx.x; idx < copy_elems; idx += blockDim.x) {
        dst_keys[dst_base + idx] = src_keys[src_base + idx];
        dst_values[dst_base + idx] = src_values[src_base + idx];
    }
}

template <typename T>
__global__ void copy_miss_chunks_kernel(
    const T* __restrict__ src_keys,
    const T* __restrict__ src_values,
    T* __restrict__ dst_keys,
    T* __restrict__ dst_values,
    const int* __restrict__ src_offsets,
    const int* __restrict__ copy_sizes,
    const int* __restrict__ dst_offsets,
    const int* __restrict__ copy_counts,
    const int* __restrict__ static_len_ptr,
    int groups,
    int src_vectors_per_group,
    int dst_vectors_per_group,
    int buffer_blocks
) {
    int64_t work = static_cast<int64_t>(blockIdx.x);
    int row = static_cast<int>(work / buffer_blocks);
    int chunk = static_cast<int>(work - static_cast<int64_t>(row) * buffer_blocks);
    if (row >= groups) {
        return;
    }

    const int count = copy_counts[row];
    if (chunk >= count) {
        return;
    }

    const int row_base = row * buffer_blocks;
    int descriptor_offset = row_base + chunk;
    int copy_vectors = copy_sizes[descriptor_offset];
    if (copy_vectors <= 0) {
        return;
    }

    int src_vector_start = src_offsets[descriptor_offset];
    int dst_offset = dst_offsets[descriptor_offset];
    int copy_elems = copy_vectors * VECTOR_SIZE_CP;
    int dst_vector_start = *static_len_ptr + dst_offset;

    int64_t src_base = (static_cast<int64_t>(row) * src_vectors_per_group + src_vector_start) * VECTOR_SIZE_CP;
    int64_t dst_base = (static_cast<int64_t>(row) * dst_vectors_per_group + dst_vector_start) * VECTOR_SIZE_CP;
    for (int idx = threadIdx.x; idx < copy_elems; idx += blockDim.x) {
        dst_keys[dst_base + idx] = src_keys[src_base + idx];
        dst_values[dst_base + idx] = src_values[src_base + idx];
    }
}

// Select generated overflow leaves on GPU, then gather their pinned-CPU KV
// pages through UVA.  The worker only publishes immutable leaf metadata; no
// host-side top-k, D2H scalar readback, or decode/worker mutex is required.
template <typename T>
__global__ void gather_selected_overflow_leaves_kernel(
    const int64_t* __restrict__ selected_ids,
    const int32_t* __restrict__ leaf_indices,
    const int32_t* __restrict__ leaf_sizes,
    const T* __restrict__ src_keys,
    const T* __restrict__ src_values,
    T* __restrict__ dst_keys,
    T* __restrict__ dst_values,
    int32_t* __restrict__ valid_lengths,
    int rows,
    int leaves,
    int leaf_size,
    int history_tokens,
    int dim,
    int nprobe,
    int budget,
    int dst_stride
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ int shared[];
    int* selected = shared;
    int* counts = selected + nprobe;
    int* offsets = counts + nprobe;
    int* old_length_ptr = offsets + nprobe;

    if (threadIdx.x == 0) {
        for (int i = 0; i < nprobe; ++i) {
            selected[i] = static_cast<int>(selected_ids[static_cast<int64_t>(row) * nprobe + i]);
            counts[i] = 0;
            offsets[i] = 0;
        }
        int cursor = 0;
        const int available = max(0, min(budget, dst_stride - valid_lengths[row]));
        for (int pos = 0; pos < nprobe; ++pos) {
            const int leaf = selected[pos];
            const int count = leaf >= 0 ? min(leaf_sizes[row * leaves + leaf], available - cursor) : 0;
            offsets[pos] = cursor;
            counts[pos] = count;
            cursor += count;
        }
        *old_length_ptr = valid_lengths[row];
        valid_lengths[row] += cursor;
    }
    __syncthreads();

    for (int pos = 0; pos < nprobe; ++pos) {
        const int leaf = selected[pos];
        const int count = counts[pos];
        const int offset = offsets[pos];
        if (leaf >= 0 && count > 0) {
            for (int linear = threadIdx.x; linear < count * dim; linear += blockDim.x) {
                const int token_pos = linear / dim;
                const int d = linear - token_pos * dim;
                const int source_token = leaf_indices[(row * leaves + leaf) * leaf_size + token_pos];
                if (source_token >= 0 && source_token < history_tokens) {
                    const int64_t src = (static_cast<int64_t>(row) * history_tokens + source_token) * dim + d;
                    const int64_t dst = (static_cast<int64_t>(row) * dst_stride + *old_length_ptr + offset + token_pos) * dim + d;
                    dst_keys[dst] = src_keys[src];
                    dst_values[dst] = src_values[src];
                }
            }
        }
        __syncthreads();
    }
}

void gather_selected_overflow_leaves(
    torch::Tensor& selected,
    torch::Tensor& leaf_indices,
    torch::Tensor& leaf_sizes,
    torch::Tensor& src_keys,
    torch::Tensor& src_values,
    torch::Tensor& dst_keys,
    torch::Tensor& dst_values,
    torch::Tensor& valid_lengths,
    int nprobe,
    int budget
) {
    TORCH_CHECK(selected.is_cuda() && selected.scalar_type() == torch::kInt64, "selected overflow leaves must be CUDA int64");
    TORCH_CHECK(leaf_indices.is_cuda() && leaf_sizes.is_cuda(), "overflow leaf tables must be CUDA");
    TORCH_CHECK(!src_keys.is_cuda() && !src_values.is_cuda() && src_keys.is_pinned() && src_values.is_pinned(), "overflow KV must be pinned CPU");
    TORCH_CHECK(dst_keys.is_cuda() && dst_values.is_cuda() && valid_lengths.is_cuda(), "overflow outputs must be CUDA");
    TORCH_CHECK(src_keys.scalar_type() == src_values.scalar_type() && src_keys.scalar_type() == dst_keys.scalar_type(), "overflow KV dtype mismatch");
    TORCH_CHECK(dst_values.scalar_type() == dst_keys.scalar_type(), "overflow output dtype mismatch");
    TORCH_CHECK(selected.dim() == 2 && leaf_indices.dim() == 3 && leaf_sizes.dim() == 2, "invalid overflow tensor rank");
    TORCH_CHECK(src_keys.dim() == 3 && src_values.sizes() == src_keys.sizes(), "invalid overflow source KV shape");
    TORCH_CHECK(dst_keys.dim() == 3 && dst_values.sizes() == dst_keys.sizes(), "invalid overflow destination KV shape");
    TORCH_CHECK(leaf_indices.size(0) == selected.size(0) && leaf_sizes.size(0) == selected.size(0) &&
                leaf_indices.size(1) == leaf_sizes.size(1), "overflow leaf row/leaf dimensions mismatch");
    TORCH_CHECK(src_keys.size(0) == selected.size(0) && dst_keys.size(0) == selected.size(0) &&
                src_keys.size(2) == dst_keys.size(2), "overflow KV rows or head dimension mismatch");
    TORCH_CHECK(valid_lengths.numel() == selected.size(0), "overflow valid_lengths row mismatch");
    TORCH_CHECK(selected.is_contiguous() && leaf_indices.is_contiguous() && leaf_sizes.is_contiguous() &&
                src_keys.is_contiguous() && src_values.is_contiguous() && dst_keys.is_contiguous() &&
                dst_values.is_contiguous() && valid_lengths.is_contiguous(), "overflow tensors must be contiguous");
    const int rows = static_cast<int>(selected.size(0));
    const int leaf_size = static_cast<int>(leaf_indices.size(2));
    TORCH_CHECK(nprobe > 0 && nprobe <= 64 && selected.size(1) == nprobe, "invalid selected overflow leaves");
    TORCH_CHECK(budget >= 0 && dst_keys.size(1) >= budget, "invalid overflow budget");
    const size_t shared_bytes = static_cast<size_t>(3 * nprobe + 1) * sizeof(int);
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, src_keys.scalar_type(), "gather_selected_overflow_leaves", [&] {
        gather_selected_overflow_leaves_kernel<scalar_t><<<rows, 256, shared_bytes, stream>>>(
            selected.data_ptr<int64_t>(),
            leaf_indices.data_ptr<int32_t>(), leaf_sizes.data_ptr<int32_t>(),
            src_keys.data_ptr<scalar_t>(), src_values.data_ptr<scalar_t>(),
            dst_keys.data_ptr<scalar_t>(), dst_values.data_ptr<scalar_t>(), valid_lengths.data_ptr<int32_t>(),
            rows, static_cast<int>(leaf_sizes.size(1)), leaf_size, static_cast<int>(src_keys.size(1)),
            static_cast<int>(dst_keys.size(2)), nprobe, budget, static_cast<int>(dst_keys.size(1)));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Gather immutable, published delta pages belonging to the same compact
// clusters selected by the base IMI path.  Page metadata is GPU-resident;
// page KV lives in pinned host memory and is read through UVA.  The kernel
// never consults an in-flight page: publication writes page_counts only after
// the producer's D2H event has completed.
template <typename T>
__global__ void gather_selected_cluster_delta_pages_kernel(
    const int64_t* __restrict__ selected_clusters,
    const int32_t* __restrict__ cluster_tails,
    const int32_t* __restrict__ page_prev,
    const int32_t* __restrict__ page_sizes,
    const T* __restrict__ page_keys,
    const T* __restrict__ page_values,
    T* __restrict__ dst_keys,
    T* __restrict__ dst_values,
    int32_t* __restrict__ valid_lengths,
    int rows,
    int nprobe,
    int clusters,
    int pages_per_cluster,
    int total_pages,
    int page_tokens,
    int dim,
    int budget,
    int dst_stride
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ int shared[];
    int* page_offsets = shared;
    int* page_token_counts = page_offsets + nprobe * pages_per_cluster;
    int* old_length = page_token_counts + nprobe * pages_per_cluster;
    const int descriptors = nprobe * pages_per_cluster;

    if (threadIdx.x == 0) {
        const int available = max(0, min(budget, dst_stride - valid_lengths[row]));
        int cursor = 0;
        for (int descriptor = 0; descriptor < descriptors; ++descriptor) {
            const int probe = descriptor / pages_per_cluster;
            const int page_slot = descriptor - probe * pages_per_cluster;
            const int cluster = static_cast<int>(selected_clusters[static_cast<int64_t>(row) * nprobe + probe]);
            int page = -1;
            if (cluster >= 0 && cluster < clusters) {
                page = cluster_tails[static_cast<int64_t>(row) * clusters + cluster];
                for (int hop = 0; hop < page_slot && page >= 0 && page < total_pages; ++hop) {
                    page = page_prev[static_cast<int64_t>(row) * total_pages + page];
                }
            }
            int count = 0;
            if (page >= 0 && page < total_pages) count = min(page_sizes[static_cast<int64_t>(row) * total_pages + page], available - cursor);
            page_offsets[descriptor] = cursor;
            page_token_counts[descriptor] = max(count, 0);
            cursor += max(count, 0);
        }
        *old_length = valid_lengths[row];
        valid_lengths[row] += cursor;
    }
    __syncthreads();

    for (int descriptor = 0; descriptor < descriptors; ++descriptor) {
        const int count = page_token_counts[descriptor];
        if (count <= 0) continue;
        const int probe = descriptor / pages_per_cluster;
        const int page_slot = descriptor - probe * pages_per_cluster;
        const int cluster = static_cast<int>(selected_clusters[static_cast<int64_t>(row) * nprobe + probe]);
        int page = cluster_tails[static_cast<int64_t>(row) * clusters + cluster];
        for (int hop = 0; hop < page_slot; ++hop) page = page_prev[static_cast<int64_t>(row) * total_pages + page];
        const int dst_offset = *old_length + page_offsets[descriptor];
        for (int linear = threadIdx.x; linear < count * dim; linear += blockDim.x) {
            const int token = linear / dim;
            const int d = linear - token * dim;
            const int64_t src = ((static_cast<int64_t>(row) * total_pages + page) * page_tokens + token) * dim + d;
            const int64_t dst = (static_cast<int64_t>(row) * dst_stride + dst_offset + token) * dim + d;
            dst_keys[dst] = page_keys[src];
            dst_values[dst] = page_values[src];
        }
        __syncthreads();
    }
}

void gather_selected_cluster_delta_pages(
    torch::Tensor& selected_clusters, torch::Tensor& cluster_tails, torch::Tensor& page_prev,
    torch::Tensor& page_sizes, torch::Tensor& page_keys, torch::Tensor& page_values,
    torch::Tensor& dst_keys, torch::Tensor& dst_values, torch::Tensor& valid_lengths,
    int pages_per_cluster, int budget
) {
    TORCH_CHECK(selected_clusters.is_cuda() && selected_clusters.scalar_type() == torch::kInt64, "selected clusters must be CUDA int64");
    TORCH_CHECK(cluster_tails.is_cuda() && page_prev.is_cuda() && page_sizes.is_cuda(), "delta page metadata must be CUDA");
    TORCH_CHECK(!page_keys.is_cuda() && !page_values.is_cuda() && page_keys.is_pinned() && page_values.is_pinned(), "delta page KV must be pinned CPU");
    TORCH_CHECK(dst_keys.is_cuda() && dst_values.is_cuda() && valid_lengths.is_cuda(), "delta page outputs must be CUDA");
    TORCH_CHECK(selected_clusters.dim() == 2 && cluster_tails.dim() == 2 && page_prev.dim() == 2 && page_sizes.dim() == 2, "invalid delta page metadata rank");
    TORCH_CHECK(page_keys.dim() == 4 && page_values.sizes() == page_keys.sizes(), "invalid delta page KV shape");
    TORCH_CHECK(dst_keys.dim() == 3 && dst_values.sizes() == dst_keys.sizes(), "invalid delta page output shape");
    TORCH_CHECK(selected_clusters.is_contiguous() && cluster_tails.is_contiguous() && page_prev.is_contiguous() && page_sizes.is_contiguous() && page_keys.is_contiguous() && page_values.is_contiguous() && dst_keys.is_contiguous() && dst_values.is_contiguous() && valid_lengths.is_contiguous(), "delta page tensors must be contiguous");
    const int rows = static_cast<int>(selected_clusters.size(0));
    const int nprobe = static_cast<int>(selected_clusters.size(1));
    const int clusters = static_cast<int>(cluster_tails.size(1));
    const int total_pages = static_cast<int>(page_keys.size(1));
    const int page_tokens = static_cast<int>(page_keys.size(2));
    const int dim = static_cast<int>(page_keys.size(3));
    TORCH_CHECK(pages_per_cluster > 0 && pages_per_cluster <= 128, "invalid pages_per_cluster");
    TORCH_CHECK(nprobe > 0 && nprobe <= 64 && rows == cluster_tails.size(0) && rows == page_prev.size(0) && rows == page_sizes.size(0) && rows == page_keys.size(0), "delta page row mismatch");
    TORCH_CHECK(page_prev.size(1) == total_pages && page_sizes.size(1) == total_pages && page_values.size(1) == total_pages && dst_keys.size(0) == rows && dst_keys.size(2) == dim && valid_lengths.numel() == rows, "delta page shape mismatch");
    TORCH_CHECK(budget >= 0 && dst_keys.size(1) >= budget, "invalid delta page budget");
    TORCH_CHECK(nprobe * pages_per_cluster <= 4096, "delta page descriptor count exceeds shared-memory limit");
    const size_t shared_bytes = static_cast<size_t>(2 * nprobe * pages_per_cluster + 1) * sizeof(int);
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, page_keys.scalar_type(), "gather_selected_cluster_delta_pages", [&] {
        gather_selected_cluster_delta_pages_kernel<scalar_t><<<rows, 256, shared_bytes, stream>>>(
            selected_clusters.data_ptr<int64_t>(), cluster_tails.data_ptr<int32_t>(), page_prev.data_ptr<int32_t>(), page_sizes.data_ptr<int32_t>(),
            page_keys.data_ptr<scalar_t>(), page_values.data_ptr<scalar_t>(), dst_keys.data_ptr<scalar_t>(), dst_values.data_ptr<scalar_t>(), valid_lengths.data_ptr<int32_t>(),
            rows, nprobe, clusters, pages_per_cluster, total_pages, page_tokens, dim, budget, static_cast<int>(dst_keys.size(1)));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
__global__ void copy_hit_chunks_kernel(
    const T* __restrict__ src_keys,
    const T* __restrict__ src_values,
    T* __restrict__ dst_keys,
    T* __restrict__ dst_values,
    const int* __restrict__ miss_sizes,
    const int* __restrict__ miss_offsets,
    const int* __restrict__ miss_counts,
    const int* __restrict__ hit_offsets,
    const int* __restrict__ hit_sizes,
    const int* __restrict__ hit_dst_offsets,
    const int* __restrict__ hit_counts,
    const int* __restrict__ static_len_ptr,
    int groups,
    int cache_pages_per_group,
    int dst_vectors_per_group,
    int buffer_blocks
) {
    int64_t work = static_cast<int64_t>(blockIdx.x);
    int row = static_cast<int>(work / buffer_blocks);
    int chunk = static_cast<int>(work - static_cast<int64_t>(row) * buffer_blocks);
    if (row >= groups || chunk >= hit_counts[row]) {
        return;
    }

    int row_base = row * buffer_blocks;
    int miss_count = miss_counts[row];
    int miss_vectors = 0;
    if (miss_count > 0) {
        int last_miss = row_base + miss_count - 1;
        miss_vectors = miss_offsets[last_miss] + miss_sizes[last_miss];
    }

    int descriptor_offset = row_base + chunk;
    int copy_vectors = hit_sizes[descriptor_offset];
    if (copy_vectors <= 0) {
        return;
    }
    int copy_elems = copy_vectors * VECTOR_SIZE_CP;
    int src_page = hit_offsets[descriptor_offset];
    int dst_vector_start = *static_len_ptr + miss_vectors + hit_dst_offsets[descriptor_offset];

    int64_t src_base = (static_cast<int64_t>(row) * cache_pages_per_group + src_page) * CHUNK_SIZE * VECTOR_SIZE_CP;
    int64_t dst_base = (static_cast<int64_t>(row) * dst_vectors_per_group + dst_vector_start) * VECTOR_SIZE_CP;
    for (int idx = threadIdx.x; idx < copy_elems; idx += blockDim.x) {
        dst_keys[dst_base + idx] = src_keys[src_base + idx];
        dst_values[dst_base + idx] = src_values[src_base + idx];
    }
}

void gather_copy_static_chunks_profile(
    torch::Tensor& key_data,
    torch::Tensor& key_buffer,
    torch::Tensor& value_data,
    torch::Tensor& value_buffer,
    torch::Tensor& valid_lengths,
    int groups,
    int data_length,
    int buffer_length,
    torch::Tensor& copy_vectors
) {
    const int blockSize = BLOCK_SIZE_CP;
    TORCH_CHECK(groups >= 0, "groups must be non-negative");
    TORCH_CHECK(data_length >= 0, "data_length must be non-negative");
    TORCH_CHECK(buffer_length >= 0, "buffer_length must be non-negative");
    if (groups <= 0) {
        return;
    }

    int static_chunks = (data_length + CHUNK_SIZE - 1) / CHUNK_SIZE;
    if (static_chunks <= 0) {
        return;
    }
    int64_t static_work = static_cast<int64_t>(groups) * static_chunks;
    TORCH_CHECK(static_work <= 2147483647LL, "static chunk-parallel gather grid is too large");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    copy_static_chunks_kernel<PTYPE><<<static_cast<int>(static_work), blockSize, 0, stream>>>(
        reinterpret_cast<PTYPE*>(key_data.data_ptr()),
        reinterpret_cast<PTYPE*>(value_data.data_ptr()),
        reinterpret_cast<PTYPE*>(key_buffer.data_ptr()),
        reinterpret_cast<PTYPE*>(value_buffer.data_ptr()),
        reinterpret_cast<int*>(valid_lengths.data_ptr<int32_t>()),
        reinterpret_cast<int*>(copy_vectors.data_ptr<int32_t>()),
        groups,
        data_length,
        buffer_length,
        static_chunks
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_copy_miss_chunks_profile(
    torch::Tensor& key_data,
    torch::Tensor& key_buffer,
    torch::Tensor& value_data,
    torch::Tensor& value_buffer,
    torch::Tensor& data_offsets,
    torch::Tensor& data_copy_sizes,
    torch::Tensor& buffer_offsets,
    torch::Tensor& copy_chunks,
    int groups,
    int data_length,
    int buffer_length,
    int offset_length,
    torch::Tensor& copy_vectors1
) {
    const int blockSize = BLOCK_SIZE_CP;
    TORCH_CHECK(groups >= 0, "groups must be non-negative");
    TORCH_CHECK(data_length >= 0, "data_length must be non-negative");
    TORCH_CHECK(buffer_length >= 0, "buffer_length must be non-negative");
    TORCH_CHECK(offset_length >= 0, "offset_length must be non-negative");
    if (groups <= 0 || offset_length <= 0) {
        return;
    }

    int64_t retrieval_work = static_cast<int64_t>(groups) * offset_length;
    TORCH_CHECK(retrieval_work <= 2147483647LL, "retrieval chunk-parallel gather grid is too large");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    copy_miss_chunks_kernel<PTYPE><<<static_cast<int>(retrieval_work), blockSize, 0, stream>>>(
        reinterpret_cast<PTYPE*>(key_data.data_ptr()),
        reinterpret_cast<PTYPE*>(value_data.data_ptr()),
        reinterpret_cast<PTYPE*>(key_buffer.data_ptr()),
        reinterpret_cast<PTYPE*>(value_buffer.data_ptr()),
        reinterpret_cast<int*>(data_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int*>(data_copy_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int*>(buffer_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int*>(copy_chunks.data_ptr<int32_t>()),
        reinterpret_cast<int*>(copy_vectors1.data_ptr<int32_t>()),
        groups,
        data_length,
        buffer_length,
        offset_length
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_copy_selected_clusters_profile(
    torch::Tensor& key_data,
    torch::Tensor& key_buffer,
    torch::Tensor& value_data,
    torch::Tensor& value_buffer,
    torch::Tensor& data_offsets,
    torch::Tensor& data_copy_sizes,
    torch::Tensor& buffer_offsets,
    torch::Tensor& copy_chunks,
    int groups,
    int data_length,
    int buffer_length,
    int offset_length,
    torch::Tensor& copy_vectors1
) {
    const int blockSize = BLOCK_SIZE_CP;
    TORCH_CHECK(groups >= 0, "groups must be non-negative");
    TORCH_CHECK(data_length >= 0, "data_length must be non-negative");
    TORCH_CHECK(buffer_length >= 0, "buffer_length must be non-negative");
    TORCH_CHECK(offset_length >= 0, "offset_length must be non-negative");
    if (groups <= 0 || offset_length <= 0) {
        return;
    }

    int64_t retrieval_work = static_cast<int64_t>(groups) * offset_length;
    TORCH_CHECK(retrieval_work <= 2147483647LL, "selected-cluster copy grid is too large");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    copy_miss_chunks_kernel<PTYPE><<<static_cast<int>(retrieval_work), blockSize, 0, stream>>>(
        reinterpret_cast<PTYPE*>(key_data.data_ptr()),
        reinterpret_cast<PTYPE*>(value_data.data_ptr()),
        reinterpret_cast<PTYPE*>(key_buffer.data_ptr()),
        reinterpret_cast<PTYPE*>(value_buffer.data_ptr()),
        reinterpret_cast<int*>(data_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int*>(data_copy_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int*>(buffer_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int*>(copy_chunks.data_ptr<int32_t>()),
        reinterpret_cast<int*>(copy_vectors1.data_ptr<int32_t>()),
        groups,
        data_length,
        buffer_length,
        offset_length
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_copy_hit_chunks_profile(
    torch::Tensor& key_data,
    torch::Tensor& key_buffer,
    torch::Tensor& value_data,
    torch::Tensor& value_buffer,
    torch::Tensor& miss_sizes,
    torch::Tensor& miss_offsets,
    torch::Tensor& miss_counts,
    torch::Tensor& hit_offsets,
    torch::Tensor& hit_sizes,
    torch::Tensor& hit_dst_offsets,
    torch::Tensor& hit_counts,
    int groups,
    int cache_pages_per_group,
    int buffer_length,
    int offset_length,
    torch::Tensor& copy_vectors1
) {
    const int blockSize = BLOCK_SIZE_CP;
    TORCH_CHECK(groups >= 0, "groups must be non-negative");
    TORCH_CHECK(cache_pages_per_group >= 0, "cache_pages_per_group must be non-negative");
    TORCH_CHECK(buffer_length >= 0, "buffer_length must be non-negative");
    TORCH_CHECK(offset_length >= 0, "offset_length must be non-negative");
    if (groups <= 0 || offset_length <= 0) {
        return;
    }

    int64_t retrieval_work = static_cast<int64_t>(groups) * offset_length;
    TORCH_CHECK(retrieval_work <= 2147483647LL, "retrieval chunk-parallel gather grid is too large");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    copy_hit_chunks_kernel<PTYPE><<<static_cast<int>(retrieval_work), blockSize, 0, stream>>>(
        reinterpret_cast<PTYPE*>(key_data.data_ptr()),
        reinterpret_cast<PTYPE*>(value_data.data_ptr()),
        reinterpret_cast<PTYPE*>(key_buffer.data_ptr()),
        reinterpret_cast<PTYPE*>(value_buffer.data_ptr()),
        reinterpret_cast<int*>(miss_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int*>(miss_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int*>(miss_counts.data_ptr<int32_t>()),
        reinterpret_cast<int*>(hit_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int*>(hit_sizes.data_ptr<int32_t>()),
        reinterpret_cast<int*>(hit_dst_offsets.data_ptr<int32_t>()),
        reinterpret_cast<int*>(hit_counts.data_ptr<int32_t>()),
        reinterpret_cast<int*>(copy_vectors1.data_ptr<int32_t>()),
        groups,
        cache_pages_per_group,
        buffer_length,
        offset_length
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_selected_overflow_leaves", &gather_selected_overflow_leaves, "GPU select and UVA gather generated overflow leaves",
          py::arg("selected"), py::arg("leaf_indices"), py::arg("leaf_sizes"),
          py::arg("src_keys"), py::arg("src_values"), py::arg("dst_keys"), py::arg("dst_values"), py::arg("valid_lengths"),
          py::arg("nprobe"), py::arg("budget"));
    m.def("gather_selected_cluster_delta_pages", &gather_selected_cluster_delta_pages, "GPU gather published IMI delta pages",
        py::arg("selected_clusters"), py::arg("cluster_tails"), py::arg("page_prev"), py::arg("page_sizes"),
          py::arg("page_keys"), py::arg("page_values"), py::arg("dst_keys"), py::arg("dst_values"), py::arg("valid_lengths"),
          py::arg("pages_per_cluster"), py::arg("budget"));
    m.def("gather_copy_static_chunks_profile", &gather_copy_static_chunks_profile, "Profile-only static chunk copy (CUDA)",
    py::arg("key_data"), py::arg("key_buffer"), py::arg("value_data"), py::arg("value_buffer"),
    py::arg("valid_lengths"), py::arg("groups"), py::arg("data_length"), py::arg("buffer_length"),
    py::arg("copy_vectors")),

    m.def("gather_copy_miss_chunks_profile", &gather_copy_miss_chunks_profile, "Profile-only miss chunk copy (CUDA)",
    py::arg("key_data"), py::arg("key_buffer"), py::arg("value_data"), py::arg("value_buffer"),
    py::arg("data_offsets"), py::arg("data_copy_sizes"), py::arg("buffer_offsets"), py::arg("copy_chunks"),
    py::arg("groups"), py::arg("data_length"), py::arg("buffer_length"), py::arg("offset_length"),
    py::arg("copy_vectors1")),

    m.def("gather_copy_selected_clusters_profile", &gather_copy_selected_clusters_profile, "Selected-cluster contiguous UVA copy (CUDA)",
    py::arg("key_data"), py::arg("key_buffer"), py::arg("value_data"), py::arg("value_buffer"),
    py::arg("data_offsets"), py::arg("data_copy_sizes"), py::arg("buffer_offsets"), py::arg("copy_chunks"),
    py::arg("groups"), py::arg("data_length"), py::arg("buffer_length"), py::arg("offset_length"),
    py::arg("copy_vectors1")),

    m.def("gather_copy_hit_chunks_profile", &gather_copy_hit_chunks_profile, "Profile-only hit chunk copy (CUDA)",
    py::arg("key_data"), py::arg("key_buffer"), py::arg("value_data"), py::arg("value_buffer"),
    py::arg("miss_sizes"), py::arg("miss_offsets"), py::arg("miss_counts"),
    py::arg("hit_offsets"), py::arg("hit_sizes"), py::arg("hit_dst_offsets"), py::arg("hit_counts"),
    py::arg("groups"), py::arg("cache_pages_per_group"), py::arg("buffer_length"), py::arg("offset_length"),
    py::arg("copy_vectors1")),

    m.def("build_cache_aware_retrieval_plan", &build_cache_aware_retrieval_plan, "Build GPU cache-aware retrieval descriptors (CUDA)",
    py::arg("selected_clusters"), py::arg("cluster_offsets"), py::arg("cluster_sizes"),
    py::arg("cache_hash_keys"), py::arg("cache_hash_values"),
    py::arg("miss_offsets"), py::arg("miss_sizes"), py::arg("miss_cumsum"), py::arg("miss_cluster_cumsum"), py::arg("miss_counts"),
    py::arg("miss_copy_offsets"), py::arg("miss_copy_sizes"), py::arg("miss_copy_cumsum"), py::arg("miss_copy_counts"),
    py::arg("cluster_copy_offsets"), py::arg("cluster_copy_sizes"), py::arg("cluster_copy_cumsum"), py::arg("cluster_copy_counts"),
    py::arg("miss_page_keys"), py::arg("admit_starts"), py::arg("hit_offsets"), py::arg("hit_sizes"), py::arg("hit_cumsum"), py::arg("hit_counts"),
    py::arg("valid_lengths"), py::arg("groups"), py::arg("nprobe"), py::arg("n_clusters"),
    py::arg("buffer_blocks"), py::arg("block_size"), py::arg("hash_size"), py::arg("padding_cluster_id")),

    m.def("promote_hot_cold_cache_pages_from_execution", &promote_hot_cold_cache_pages_from_execution,
    "Admit missed execution-buffer pages using the AdaptiveIMI Hot-Cold GPU cache policy (CUDA)",
    py::arg("execution_keys"), py::arg("cache_keys"), py::arg("execution_values"), py::arg("cache_values"),
    py::arg("miss_dst_offsets"), py::arg("miss_sizes"), py::arg("miss_counts"), py::arg("miss_page_keys"),
    py::arg("admit_starts"),
    py::arg("promote_src_offsets"), py::arg("promote_dst_slots"), py::arg("promote_sizes"),
    py::arg("promote_hash_positions"), py::arg("promote_page_keys"), py::arg("promote_counts"),
    py::arg("hit_offsets"), py::arg("hit_counts"), py::arg("cluster_sizes"),
    py::arg("cache_hash_keys"), py::arg("cache_hash_values"), py::arg("cache_slot_keys"),
    py::arg("cluster_state"), py::arg("cluster_prev"), py::arg("cluster_next"),
    py::arg("hot_head"), py::arg("hot_tail"), py::arg("cold_head"), py::arg("cold_tail"),
    py::arg("hot_pages"), py::arg("cold_pages"), py::arg("free_slots"), py::arg("free_counts"),
    py::arg("static_len"), py::arg("groups"), py::arg("execution_stride"), py::arg("cache_size"), py::arg("hash_size"),
    py::arg("buffer_blocks"), py::arg("nprobe"), py::arg("n_clusters"), py::arg("page_size"),
    py::arg("hot_capacity_pages"), py::arg("cold_capacity_pages"), py::arg("max_promote_pages_per_group"));

}
