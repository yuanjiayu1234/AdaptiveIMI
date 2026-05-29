#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace imi_hybrid_bench {

namespace {

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
    int subspaces,
    int kv_heads,
    int max_tokens,
    int head_dim,
    int max_k,
    int max_sub_dim,
    bool use_kmeanspp
) {
    extern __shared__ double shared_sums[];
    const int task = blockIdx.x;
    const int row = task / subspaces;
    const int subspace = task - row * subspaces;
    if (row >= rows) {
        return;
    }

    const int dim_base = (row * subspaces + subspace) * max_sub_dim;
    const int sub_dim = subspace_dim_count(dim_masks, dim_base, max_sub_dim);
    const int kval = k_values[subspace];
    const int token_count = row_token_counts[row];
    const int b = row / kv_heads;
    const int h = row - b * kv_heads;

    for (int linear = threadIdx.x; linear < max_k * max_sub_dim; linear += blockDim.x) {
        const int dim = linear % max_sub_dim;
        const int k = linear / max_sub_dim;
        const int centroid_base = ((row * subspaces + subspace) * max_k + k) * max_sub_dim;
        centroids[centroid_base + dim] = 0.0f;
        if (dim == 0) {
            centroid_masks[(row * subspaces + subspace) * max_k + k] =
                !(token_count > 0 && k < kval);
        }
    }
    __syncthreads();

    if (token_count <= 0 || kval <= 0 || sub_dim <= 0) {
        return;
    }

    for (int centroid_id = 0; centroid_id < kval; ++centroid_id) {
        int selected = 0;
        if (!use_kmeanspp) {
            selected = sample_token_index(token_count, row, subspace, centroid_id, 0, 42424242ULL);
        } else if (centroid_id == 0) {
            selected = sample_token_index(token_count, row, subspace, centroid_id, 0, 42ULL);
        } else {
            double local_weight = 0.0;
            for (int token = threadIdx.x; token < token_count; token += blockDim.x) {
                double best_dist = DBL_MAX;
                for (int prev = 0; prev < centroid_id; ++prev) {
                    double dist = 0.0;
                    const int centroid_base = ((row * subspaces + subspace) * max_k + prev) * max_sub_dim;
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
                            const int centroid_base = ((row * subspaces + subspace) * max_k + prev) * max_sub_dim;
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

        const int centroid_base = ((row * subspaces + subspace) * max_k + centroid_id) * max_sub_dim;
        for (int d = threadIdx.x; d < sub_dim; d += blockDim.x) {
            const int64_t source_dim = dim_ids[dim_base + d];
            centroids[centroid_base + d] = load_key_value(
                keys, b, h, selected, source_dim, kv_heads, max_tokens, head_dim
            );
        }
        if (threadIdx.x == 0) {
            centroid_masks[(row * subspaces + subspace) * max_k + centroid_id] = false;
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
    int subspaces,
    int kv_heads,
    int max_tokens,
    int head_dim,
    int max_k,
    int max_sub_dim
) {
    const int64_t total = static_cast<int64_t>(rows) * subspaces * max_tokens;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        int tmp = static_cast<int>(linear);
        const int token = tmp % max_tokens;
        tmp /= max_tokens;
        const int subspace = tmp % subspaces;
        const int row = tmp / subspaces;

        const int token_count = row_token_counts[row];
        const int label_offset = (row * subspaces + subspace) * max_tokens + token;
        if (token >= token_count || token_count <= 0) {
            labels[label_offset] = 0;
            continue;
        }

        const int kval = k_values[subspace];
        const int dim_base = (row * subspaces + subspace) * max_sub_dim;
        const int sub_dim = subspace_dim_count(dim_masks, dim_base, max_sub_dim);
        const int b = row / kv_heads;
        const int h = row - b * kv_heads;

        float best_dist = FLT_MAX;
        int best_k = 0;
        for (int k = 0; k < kval; ++k) {
            float dist = 0.0f;
            const int centroid_base = ((row * subspaces + subspace) * max_k + k) * max_sub_dim;
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
        atomicAdd(&counts[(row * subspaces + subspace) * max_k + best_k], 1.0f);
        const int sum_base = ((row * subspaces + subspace) * max_k + best_k) * max_sub_dim;
        for (int d = 0; d < sub_dim; ++d) {
            const int64_t source_dim = dim_ids[dim_base + d];
            const float x = load_key_value(keys, b, h, token, source_dim, kv_heads, max_tokens, head_dim);
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
    int subspaces,
    int kv_heads,
    int max_tokens,
    int head_dim,
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
    const int row = task / subspaces;
    const int subspace = task - row * subspaces;
    if (row >= rows) {
        return;
    }

    const int dim_base = (row * subspaces + subspace) * max_sub_dim;
    const int sub_dim = subspace_dim_count(dim_masks, dim_base, max_sub_dim);
    const int kval = k_values[subspace];
    const int token_count = row_token_counts[row];
    const int b = row / kv_heads;
    const int h = row - b * kv_heads;

    for (int idx = threadIdx.x; idx < max_k * max_sub_dim; idx += blockDim.x) {
        shared_sums[idx] = 0.0f;
        shared_centroids[idx] = centroids[((row * subspaces + subspace) * max_k * max_sub_dim) + idx];
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
                const float x = load_key_value(keys, b, h, token, source_dim, kv_heads, max_tokens, head_dim);
                const float c = shared_centroids[k * max_sub_dim + d];
                const float diff = x - c;
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }

        labels[(row * subspaces + subspace) * max_tokens + token] = best_k;
        atomicAdd(&shared_counts[best_k], 1.0f);
        const int sum_base = best_k * max_sub_dim;
        for (int d = 0; d < sub_dim; ++d) {
            const int source_dim = shared_dim_ids[d];
            const float x = load_key_value(keys, b, h, token, source_dim, kv_heads, max_tokens, head_dim);
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
    int subspaces,
    int kv_heads,
    int max_tokens,
    int head_dim,
    int max_k,
    int max_sub_dim,
    int tiles_per_task,
    int iter
) {
    const int64_t total = static_cast<int64_t>(rows) * subspaces * max_k * max_sub_dim;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        int tmp = static_cast<int>(linear);
        const int dim = tmp % max_sub_dim;
        tmp /= max_sub_dim;
        const int k = tmp % max_k;
        tmp /= max_k;
        const int subspace = tmp % subspaces;
        const int row = tmp / subspaces;

        const int kval = k_values[subspace];
        const int dim_base = (row * subspaces + subspace) * max_sub_dim;
        const bool valid_dim = !dim_masks[dim_base + dim];
        const int token_count = row_token_counts[row];
        const bool valid_centroid = token_count > 0 && k < kval;
        const int mask_offset = (row * subspaces + subspace) * max_k + k;
        const int value_offset = ((row * subspaces + subspace) * max_k + k) * max_sub_dim + dim;

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
        const int task = row * subspaces + subspace;
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
                keys, b, h, token, source_dim, kv_heads, max_tokens, head_dim
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
    int subspaces,
    int kv_heads,
    int max_tokens,
    int head_dim,
    int max_k,
    int max_sub_dim,
    int iter
) {
    const int64_t total = static_cast<int64_t>(rows) * subspaces * max_k * max_sub_dim;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        int tmp = static_cast<int>(linear);
        const int dim = tmp % max_sub_dim;
        tmp /= max_sub_dim;
        const int k = tmp % max_k;
        tmp /= max_k;
        const int subspace = tmp % subspaces;
        const int row = tmp / subspaces;

        const int kval = k_values[subspace];
        const int dim_base = (row * subspaces + subspace) * max_sub_dim;
        const bool valid_dim = !dim_masks[dim_base + dim];
        const int token_count = row_token_counts[row];
        const bool valid_centroid = token_count > 0 && k < kval;
        const int mask_offset = (row * subspaces + subspace) * max_k + k;
        const int value_offset = ((row * subspaces + subspace) * max_k + k) * max_sub_dim + dim;

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

inline void check_cuda(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

} // namespace

std::vector<torch::Tensor> run_subspace_kmeans_only_cuda(
    const torch::Tensor& keys,
    const torch::Tensor& row_token_counts,
    const torch::Tensor& subspace_dim_ids,
    const torch::Tensor& subspace_dim_masks,
    const torch::Tensor& k_values,
    int64_t max_iters,
    int64_t max_k_arg,
    bool use_kmeanspp
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

    const auto batch_size = keys.size(0);
    const auto kv_heads = keys.size(1);
    const auto max_tokens = keys.size(2);
    const auto head_dim = keys.size(3);
    const int rows = static_cast<int>(batch_size * kv_heads);
    const int subspaces = static_cast<int>(k_values.numel());
    TORCH_CHECK(subspaces > 0, "k_values must be non-empty");
    TORCH_CHECK(row_token_counts.numel() == rows, "row_token_counts rows mismatch");
    TORCH_CHECK(subspace_dim_ids.dim() == 3, "subspace_dim_ids must be [rows, M, max_sub_dim]");
    TORCH_CHECK(subspace_dim_ids.size(0) == rows, "subspace_dim_ids rows mismatch");
    TORCH_CHECK(subspace_dim_ids.size(1) == subspaces, "subspace_dim_ids M mismatch");
    TORCH_CHECK(subspace_dim_masks.sizes() == subspace_dim_ids.sizes(), "subspace_dim_masks shape mismatch");
    TORCH_CHECK(max_k_arg > 0 && max_k_arg <= std::numeric_limits<int>::max(), "max_k out of range");

    const int max_k = static_cast<int>(max_k_arg);
    const int max_sub_dim = static_cast<int>(subspace_dim_ids.size(2));
    const int iters = static_cast<int>(std::max<int64_t>(1, max_iters));

    const c10::cuda::CUDAGuard device_guard(keys.device());
    int current_device = -1;
    int sm_count = 1;
    cudaGetDevice(&current_device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, current_device);

    auto labels = torch::zeros({rows, subspaces, max_tokens}, keys.options().dtype(torch::kInt32));
    auto centroids = torch::zeros({rows, subspaces, max_k, max_sub_dim}, keys.options().dtype(torch::kFloat32));
    auto centroid_masks = torch::ones({rows, subspaces, max_k}, keys.options().dtype(torch::kBool));
    auto sums = torch::zeros_like(centroids);
    auto counts = torch::zeros({rows, subspaces, max_k}, keys.options().dtype(torch::kFloat32));

    const int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t init_tasks = static_cast<int64_t>(rows) * subspaces;
    const int init_blocks = static_cast<int>(init_tasks);
    const int64_t update_total = static_cast<int64_t>(rows) * subspaces * max_k * max_sub_dim;
    const int update_blocks = capped_blocks(update_total, threads, sm_count);
    const size_t init_shared_bytes = static_cast<size_t>(threads) * sizeof(double);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        keys.scalar_type(),
        "subspace_bench_init_centroids",
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
                subspaces,
                static_cast<int>(kv_heads),
                static_cast<int>(max_tokens),
                static_cast<int>(head_dim),
                max_k,
                max_sub_dim,
                use_kmeanspp
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t assign_total = static_cast<int64_t>(rows) * subspaces * max_tokens;
    const int assign_blocks = capped_blocks(assign_total, threads, sm_count);
    const int tile_tokens = 512;
    const int tiles_per_task = static_cast<int>((max_tokens + tile_tokens - 1) / tile_tokens);
    const bool use_tiled_reduce = max_k <= 32 && max_sub_dim <= 64;
    const int64_t partial_tasks = static_cast<int64_t>(rows) * subspaces * tiles_per_task;
    torch::Tensor partial_sums;
    torch::Tensor partial_counts;
    if (use_tiled_reduce) {
        TORCH_CHECK(
            partial_tasks <= std::numeric_limits<unsigned int>::max(),
            "tiled reduce grid is too large"
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

            const size_t tiled_shared_bytes =
                (static_cast<size_t>(2) * static_cast<size_t>(max_k) * static_cast<size_t>(max_sub_dim) +
                 static_cast<size_t>(max_k)) * sizeof(float) +
                static_cast<size_t>(max_sub_dim) * sizeof(int);
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                keys.scalar_type(),
                "subspace_bench_assign_reduce_tiled",
                [&] {
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
                        subspaces,
                        static_cast<int>(kv_heads),
                        static_cast<int>(max_tokens),
                        static_cast<int>(head_dim),
                        max_k,
                        max_sub_dim,
                        tile_tokens,
                        tiles_per_task
                    );
                }
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                keys.scalar_type(),
                "subspace_bench_reduce_update_tiled",
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
                        subspaces,
                        static_cast<int>(kv_heads),
                        static_cast<int>(max_tokens),
                        static_cast<int>(head_dim),
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
                "subspace_bench_assign_reduce",
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
                        subspaces,
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
                "subspace_bench_update_centroids",
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
                        subspaces,
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
    }

    return {
        centroids,
        centroid_masks,
        labels,
        counts,
    };
}

} // namespace imi_hybrid_bench
