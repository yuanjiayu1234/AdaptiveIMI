#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cfloat>
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
    int64_t kv_heads,
    int64_t max_tokens,
    int64_t head_dim
) {
    const int64_t offset = (((b * kv_heads + h) * max_tokens + t) * head_dim + d);
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

__device__ __forceinline__ double uniform_unit(int row, int subspace, int k, int iter, uint64_t salt) {
    const uint64_t bits = make_seed(row, subspace, k, iter, salt) >> 11;
    return static_cast<double>(bits) * (1.0 / 9007199254740992.0);
}

template <typename scalar_t>
__global__ void init_centroids_kmeanspp_kernel(
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
    int max_k,
    int max_sub_dim
) {
    extern __shared__ double shared_sums[];
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
        int selected = 0;
        if (centroid_id == 0) {
            selected = sample_token_index(token_count, row, subspace, centroid_id, 0, 42ULL);
        } else {
            double local_weight = 0.0;
            for (int token = threadIdx.x; token < token_count; token += blockDim.x) {
                double best_dist = DBL_MAX;
                for (int prev = 0; prev < centroid_id; ++prev) {
                    double dist = 0.0;
                    const int centroid_base = ((row * kSubspaces + subspace) * max_k + prev) * max_sub_dim;
                    for (int d = 0; d < sub_dim; ++d) {
                        const int64_t source_dim = dim_ids[dim_base + d];
                        const double x = static_cast<double>(
                            load_key_value(keys, b, h, token, source_dim, kv_heads, max_tokens, head_dim)
                        );
                        const double c = static_cast<double>(centroids[centroid_base + d]);
                        const double diff = x - c;
                        dist += diff * diff;
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                    }
                }
                local_weight += best_dist;
            }

            shared_sums[threadIdx.x] = local_weight;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                const double total_weight = shared_sums[0];
                if (total_weight <= 0.0 || !isfinite(total_weight)) {
                    selected = sample_token_index(token_count, row, subspace, centroid_id, 0, 4242ULL);
                } else {
                    const double target = uniform_unit(row, subspace, centroid_id, 0, 424242ULL) * total_weight;
                    double cumsum = 0.0;
                    selected = token_count - 1;
                    for (int token = 0; token < token_count; ++token) {
                        double best_dist = DBL_MAX;
                        for (int prev = 0; prev < centroid_id; ++prev) {
                            double dist = 0.0;
                            const int centroid_base = ((row * kSubspaces + subspace) * max_k + prev) * max_sub_dim;
                            for (int d = 0; d < sub_dim; ++d) {
                                const int64_t source_dim = dim_ids[dim_base + d];
                                const double x = static_cast<double>(
                                    load_key_value(keys, b, h, token, source_dim, kv_heads, max_tokens, head_dim)
                                );
                                const double c = static_cast<double>(centroids[centroid_base + d]);
                                const double diff = x - c;
                                dist += diff * diff;
                            }
                            if (dist < best_dist) {
                                best_dist = dist;
                            }
                        }
                        cumsum += best_dist;
                        if (cumsum >= target) {
                            selected = token;
                            break;
                        }
                    }
                }
                shared_sums[0] = static_cast<double>(selected);
            }
            __syncthreads();
            selected = static_cast<int>(shared_sums[0]);
        }

        const int centroid_base = ((row * kSubspaces + subspace) * max_k + centroid_id) * max_sub_dim;
        for (int d = threadIdx.x; d < sub_dim; d += blockDim.x) {
            const int64_t source_dim = dim_ids[dim_base + d];
            centroids[centroid_base + d] = load_key_value(
                keys, b, h, selected, source_dim, kv_heads, max_tokens, head_dim
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
                const float x = load_key_value(keys, b, h, token, source_dim, kv_heads, max_tokens, head_dim);
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
            const float x = load_key_value(keys, b, h, token, source_dim, kv_heads, max_tokens, head_dim);
            atomicAdd(&sums[sum_base + d], x);
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
                keys, b, h, token, source_dim, kv_heads, max_tokens, head_dim
            );
        }
    }
}

__device__ __forceinline__ int pack_dense_code8(
    const int32_t* __restrict__ labels,
    const int32_t* __restrict__ k_values,
    int row,
    int token,
    int max_tokens,
    int code_space
) {
    int code = 0;
    int stride = 1;
    for (int s = 0; s < kSubspaces; ++s) {
        const int kval = k_values[s] > 1 ? k_values[s] : 1;
        const int label = labels[(row * kSubspaces + s) * max_tokens + token];
        if (label < 0 || label >= kval) {
            return -1;
        }
        code += label * stride;
        stride *= kval;
    }
    return (code >= 0 && code < code_space) ? code : -1;
}

__device__ __forceinline__ int decode_dense_code_digit(
    int code,
    const int32_t* __restrict__ k_values,
    int subspace
) {
    for (int s = 0; s < subspace; ++s) {
        const int kval = k_values[s] > 1 ? k_values[s] : 1;
        code /= kval;
    }
    const int kval = k_values[subspace] > 1 ? k_values[subspace] : 1;
    return code % kval;
}

__global__ void initialize_dense_workspaces_kernel(
    int32_t* __restrict__ dense_counts,
    int32_t* __restrict__ dense_cursors,
    int64_t total
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        dense_counts[i] = 0;
        dense_cursors[i] = 0;
    }
}

__global__ void count_dense_codes_kernel(
    const int32_t* __restrict__ labels,
    const int32_t* __restrict__ row_token_counts,
    const int32_t* __restrict__ k_values,
    int32_t* __restrict__ dense_counts,
    int rows,
    int max_tokens,
    int code_space
) {
    const int64_t total = static_cast<int64_t>(rows) * max_tokens;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        const int token = static_cast<int>(linear % max_tokens);
        const int row = static_cast<int>(linear / max_tokens);
        if (token >= row_token_counts[row]) {
            continue;
        }
        const int code = pack_dense_code8(labels, k_values, row, token, max_tokens, code_space);
        if (code >= 0) {
            atomicAdd(&dense_counts[row * code_space + code], 1);
        }
    }
}

__global__ void dense_active_flags_kernel(
    const int32_t* __restrict__ dense_counts,
    int32_t* __restrict__ active_flags,
    int rows,
    int code_space
) {
    const int64_t total = static_cast<int64_t>(rows) * code_space;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        active_flags[linear] = dense_counts[linear] > 0 ? 1 : 0;
    }
}

__global__ void finalize_dense_compact_kernel(
    const int32_t* __restrict__ dense_counts,
    const int32_t* __restrict__ dense_offsets,
    const int32_t* __restrict__ active_scan,
    const int32_t* __restrict__ row_token_counts,
    const int32_t* __restrict__ k_values,
    int32_t* __restrict__ code_to_compact,
    int32_t* __restrict__ compact_codes,
    int32_t* __restrict__ cluster_sizes,
    int32_t* __restrict__ cluster_offsets,
    int32_t* __restrict__ cluster_counts,
    int rows,
    int code_space,
    int max_clusters
) {
    const int64_t total = static_cast<int64_t>(rows) * code_space;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        const int code = static_cast<int>(linear % code_space);
        const int row = static_cast<int>(linear / code_space);
        const int count = dense_counts[linear];
        const int active = count > 0 ? 1 : 0;
        const int compact = active_scan[linear];
        code_to_compact[linear] = -1;
        if (active != 0) {
            if (compact < max_clusters) {
                code_to_compact[linear] = compact;
                cluster_sizes[row * max_clusters + compact] = count;
                cluster_offsets[row * (max_clusters + 1) + compact] = dense_offsets[linear];
                const int code_base = (row * max_clusters + compact) * kSubspaces;
                #pragma unroll
                for (int s = 0; s < kSubspaces; ++s) {
                    compact_codes[code_base + s] = decode_dense_code_digit(code, k_values, s);
                }
            }
        }
        if (code == code_space - 1) {
            const int active_count = compact + active;
            cluster_counts[row] = active_count;
            const int tail = active_count < max_clusters ? active_count : max_clusters;
            cluster_offsets[row * (max_clusters + 1) + tail] = row_token_counts[row];
        }
    }
}

__global__ void scatter_dense_reverse_index_kernel(
    const int32_t* __restrict__ labels,
    const int32_t* __restrict__ row_token_counts,
    const int32_t* __restrict__ k_values,
    const int32_t* __restrict__ dense_offsets,
    int32_t* __restrict__ dense_cursors,
    const int32_t* __restrict__ code_to_compact,
    int32_t* __restrict__ token_labels,
    int32_t* __restrict__ token_ids,
    int rows,
    int max_tokens,
    int code_space
) {
    const int64_t total = static_cast<int64_t>(rows) * max_tokens;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        const int token = static_cast<int>(linear % max_tokens);
        const int row = static_cast<int>(linear / max_tokens);
        if (token >= row_token_counts[row]) {
            continue;
        }
        const int code = pack_dense_code8(labels, k_values, row, token, max_tokens, code_space);
        if (code < 0) {
            continue;
        }
        const int code_offset = row * code_space + code;
        const int compact = code_to_compact[code_offset];
        if (compact < 0) {
            continue;
        }
        const int dst = dense_offsets[code_offset] + atomicAdd(&dense_cursors[code_offset], 1);
        token_labels[row * max_tokens + token] = compact;
        token_ids[row * max_tokens + dst] = token;
    }
}

__global__ void initialize_compact_outputs_kernel(
    int32_t* __restrict__ token_labels,
    int32_t* __restrict__ compact_codes,
    int32_t* __restrict__ cluster_sizes,
    int32_t* __restrict__ cluster_offsets,
    int32_t* __restrict__ cluster_counts,
    float* __restrict__ full_centroids,
    int rows,
    int max_tokens,
    int max_clusters,
    int head_dim
) {
    const int64_t token_total = static_cast<int64_t>(rows) * max_tokens;
    const int64_t cluster_total = static_cast<int64_t>(rows) * max_clusters;
    const int64_t code_total = cluster_total * kSubspaces;
    const int64_t offset_total = static_cast<int64_t>(rows) * (max_clusters + 1);
    const int64_t centroid_total = cluster_total * head_dim;
    int64_t total = code_total > centroid_total ? code_total : centroid_total;
    const int64_t token_or_offset = token_total > offset_total ? token_total : offset_total;
    total = total > token_or_offset ? total : token_or_offset;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        if (i < token_total) {
            token_labels[i] = -1;
        }
        if (i < cluster_total) {
            cluster_sizes[i] = 0;
        }
        if (i < code_total) {
            compact_codes[i] = -1;
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

__global__ void finalize_offsets_kernel(
    const int32_t* __restrict__ cluster_counts,
    const int32_t* __restrict__ row_token_counts,
    int32_t* __restrict__ cluster_offsets,
    int rows,
    int max_tokens,
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
    int head_dim
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
                sum += load_key_value(keys, b, h, token, dim, kv_heads, max_tokens, head_dim);
            }
        }
        full_centroids[(row * max_clusters + cluster) * head_dim + dim] = sum / static_cast<float>(size);
    }
}

inline void check_cuda(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
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
    int64_t dense_code_space_arg
) {
    check_cuda(keys, "keys");
    check_cuda(row_token_counts, "row_token_counts");
    check_cuda(subspace_dim_ids, "subspace_dim_ids");
    check_cuda(subspace_dim_masks, "subspace_dim_masks");
    check_cuda(k_values, "k_values");

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
    const int rows = static_cast<int>(batch_size * kv_heads);
    TORCH_CHECK(row_token_counts.numel() == rows, "row_token_counts rows mismatch");
    TORCH_CHECK(subspace_dim_ids.size(0) == rows, "subspace_dim_ids rows mismatch");
    TORCH_CHECK(subspace_dim_ids.size(1) == kSubspaces, "subspace_dim_ids must have 8 subspaces");
    TORCH_CHECK(subspace_dim_masks.sizes() == subspace_dim_ids.sizes(), "subspace_dim_masks shape mismatch");

    const int max_sub_dim = static_cast<int>(subspace_dim_ids.size(2));
    TORCH_CHECK(max_k_arg > 0 && max_k_arg <= std::numeric_limits<int>::max(), "max_k out of range");
    TORCH_CHECK(dense_code_space_arg > 0 && dense_code_space_arg <= std::numeric_limits<int>::max(), "dense code space out of range");
    const int max_k = static_cast<int>(max_k_arg);
    const int code_space = static_cast<int>(dense_code_space_arg);
    const int iters = static_cast<int>(std::max<int64_t>(1, max_iters));
    const int64_t max_compact_clusters = std::max<int64_t>(1, std::min<int64_t>(code_space, max_tokens));

    const c10::cuda::CUDAGuard device_guard(keys.device());
    int current_device = -1;
    int sm_count = 1;
    cudaGetDevice(&current_device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, current_device);
    auto labels = torch::zeros({rows, kSubspaces, max_tokens}, keys.options().dtype(torch::kInt32));
    auto centroids = torch::zeros({rows, kSubspaces, max_k, max_sub_dim}, keys.options().dtype(torch::kFloat32));
    auto centroid_masks = torch::ones({rows, kSubspaces, max_k}, keys.options().dtype(torch::kBool));
    auto sums = torch::zeros_like(centroids);
    auto counts = torch::zeros({rows, kSubspaces, max_k}, keys.options().dtype(torch::kFloat32));
    auto token_ids = torch::empty({rows, max_tokens}, keys.options().dtype(torch::kInt32));
    auto token_labels = torch::empty({rows, max_tokens}, keys.options().dtype(torch::kInt32));
    auto compact_codes = torch::empty({rows, max_compact_clusters, kSubspaces}, keys.options().dtype(torch::kInt32));
    auto cluster_sizes = torch::empty({rows, max_compact_clusters}, keys.options().dtype(torch::kInt32));
    auto cluster_offsets = torch::empty({rows, max_compact_clusters + 1}, keys.options().dtype(torch::kInt32));
    auto cluster_counts = torch::empty({rows}, keys.options().dtype(torch::kInt32));
    auto full_centroids = torch::empty({rows, max_compact_clusters, head_dim}, keys.options().dtype(torch::kFloat32));
    auto dense_counts = torch::empty({rows, code_space}, keys.options().dtype(torch::kInt32));
    auto dense_offsets = torch::empty({rows, code_space}, keys.options().dtype(torch::kInt32));
    auto dense_cursors = torch::empty({rows, code_space}, keys.options().dtype(torch::kInt32));
    auto active_scan = torch::empty({rows, code_space}, keys.options().dtype(torch::kInt32));
    auto code_to_compact = torch::empty({rows, code_space}, keys.options().dtype(torch::kInt32));

    const int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t init_tasks = static_cast<int64_t>(rows) * kSubspaces;
    const int init_blocks = static_cast<int>(init_tasks);
    const int64_t update_total = static_cast<int64_t>(rows) * kSubspaces * max_k * max_sub_dim;
    const int update_blocks = capped_blocks(update_total, threads, sm_count);
    const size_t init_shared_bytes = static_cast<size_t>(threads) * sizeof(double);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        keys.scalar_type(),
        "hybrid_8way_init_centroids",
        [&] {
            init_centroids_kmeanspp_kernel<scalar_t><<<init_blocks, threads, init_shared_bytes, stream>>>(
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
                max_k,
                max_sub_dim
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t assign_total = static_cast<int64_t>(rows) * kSubspaces * max_tokens;
    const int assign_blocks = capped_blocks(assign_total, threads, sm_count);

    for (int iter = 0; iter < iters; ++iter) {
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
                    max_k,
                    max_sub_dim,
                    iter
                );
            }
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    TORCH_CHECK(max_k <= 65535, "hybrid 8-way compact CUDA path currently requires every k_value <= 65535");

    const int64_t compact_total = static_cast<int64_t>(rows) * max_tokens;
    const int compact_blocks = capped_blocks(compact_total, threads, sm_count);
    const int64_t dense_total = static_cast<int64_t>(rows) * code_space;
    const int dense_blocks = capped_blocks(dense_total, threads, sm_count);
    const int64_t init_compact_total = std::max<int64_t>(
        static_cast<int64_t>(rows) * max_compact_clusters * std::max<int64_t>(kSubspaces, head_dim),
        std::max<int64_t>(compact_total, static_cast<int64_t>(rows) * (max_compact_clusters + 1))
    );
    const int init_compact_blocks = capped_blocks(init_compact_total, threads, sm_count);
    initialize_compact_outputs_kernel<<<init_compact_blocks, threads, 0, stream>>>(
        token_labels.data_ptr<int32_t>(),
        compact_codes.data_ptr<int32_t>(),
        cluster_sizes.data_ptr<int32_t>(),
        cluster_offsets.data_ptr<int32_t>(),
        cluster_counts.data_ptr<int32_t>(),
        full_centroids.data_ptr<float>(),
        rows,
        static_cast<int>(max_tokens),
        static_cast<int>(max_compact_clusters),
        static_cast<int>(head_dim)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    initialize_dense_workspaces_kernel<<<dense_blocks, threads, 0, stream>>>(
        dense_counts.data_ptr<int32_t>(),
        dense_cursors.data_ptr<int32_t>(),
        dense_total
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    count_dense_codes_kernel<<<compact_blocks, threads, 0, stream>>>(
        labels.data_ptr<int32_t>(),
        row_token_counts.data_ptr<int32_t>(),
        k_values.data_ptr<int32_t>(),
        dense_counts.data_ptr<int32_t>(),
        rows,
        static_cast<int>(max_tokens),
        code_space
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto policy = thrust::cuda::par.on(stream);
    for (int row = 0; row < rows; ++row) {
        auto counts_begin = thrust::device_pointer_cast(dense_counts.data_ptr<int32_t>() + static_cast<int64_t>(row) * code_space);
        auto offsets_begin = thrust::device_pointer_cast(dense_offsets.data_ptr<int32_t>() + static_cast<int64_t>(row) * code_space);
        thrust::exclusive_scan(policy, counts_begin, counts_begin + code_space, offsets_begin);
    }

    dense_active_flags_kernel<<<dense_blocks, threads, 0, stream>>>(
        dense_counts.data_ptr<int32_t>(),
        code_to_compact.data_ptr<int32_t>(),
        rows,
        code_space
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int row = 0; row < rows; ++row) {
        auto flags_begin = thrust::device_pointer_cast(code_to_compact.data_ptr<int32_t>() + static_cast<int64_t>(row) * code_space);
        auto scan_begin = thrust::device_pointer_cast(active_scan.data_ptr<int32_t>() + static_cast<int64_t>(row) * code_space);
        thrust::exclusive_scan(policy, flags_begin, flags_begin + code_space, scan_begin);
    }

    finalize_dense_compact_kernel<<<dense_blocks, threads, 0, stream>>>(
        dense_counts.data_ptr<int32_t>(),
        dense_offsets.data_ptr<int32_t>(),
        active_scan.data_ptr<int32_t>(),
        row_token_counts.data_ptr<int32_t>(),
        k_values.data_ptr<int32_t>(),
        code_to_compact.data_ptr<int32_t>(),
        compact_codes.data_ptr<int32_t>(),
        cluster_sizes.data_ptr<int32_t>(),
        cluster_offsets.data_ptr<int32_t>(),
        cluster_counts.data_ptr<int32_t>(),
        rows,
        code_space,
        static_cast<int>(max_compact_clusters)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    scatter_dense_reverse_index_kernel<<<compact_blocks, threads, 0, stream>>>(
        labels.data_ptr<int32_t>(),
        row_token_counts.data_ptr<int32_t>(),
        k_values.data_ptr<int32_t>(),
        dense_offsets.data_ptr<int32_t>(),
        dense_cursors.data_ptr<int32_t>(),
        code_to_compact.data_ptr<int32_t>(),
        token_labels.data_ptr<int32_t>(),
        token_ids.data_ptr<int32_t>(),
        rows,
        static_cast<int>(max_tokens),
        code_space
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    finalize_offsets_kernel<<<capped_blocks(static_cast<int64_t>(rows) * (max_compact_clusters + 1), threads, sm_count), threads, 0, stream>>>(
        cluster_counts.data_ptr<int32_t>(),
        row_token_counts.data_ptr<int32_t>(),
        cluster_offsets.data_ptr<int32_t>(),
        rows,
        static_cast<int>(max_tokens),
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
                static_cast<int>(head_dim)
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        labels,
        centroids,
        centroid_masks,
        token_labels,
        token_ids,
        compact_codes,
        cluster_sizes,
        cluster_offsets,
        cluster_counts,
        full_centroids,
    };
}

} // namespace imi_hybrid
