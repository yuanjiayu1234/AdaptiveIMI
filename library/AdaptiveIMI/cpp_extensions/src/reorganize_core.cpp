// ============================================================================
// reorganize_core.cpp - 剥离的Reorganize核心算法（纯功能函数，无封装）
// ============================================================================
#include "layer_pipeline.hpp"
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <memory>
#include <cstdlib>
#include <vector>
#include <limits>
#include <cstdint>
#include <ATen/Parallel.h>
#include <c10/core/ScalarType.h>

namespace imi {

std::pair<HeadReorgInput, std::vector<float>> build_compact_reorg_input(
    const std::vector<int32_t>& labels1,
    const std::vector<int32_t>& labels2,
    const std::vector<float>& centroids1,
    const std::vector<float>& centroids2,
    int32_t T,
    int32_t dim_half
) {
    HeadReorgInput out;

    const int32_t k1 = static_cast<int32_t>(centroids1.size()) / dim_half;
    const int32_t k2 = static_cast<int32_t>(centroids2.size()) / dim_half;
    const int32_t n_clusters = k1 * k2;

    std::vector<int32_t> combined(T);
    for (int32_t i = 0; i < T; ++i) {
        combined[i] = labels1[i] * k2 + labels2[i];
    }

    std::vector<uint8_t> used(n_clusters, 0);
    for (int32_t id : combined) {
        used[id] = 1;
    }

    std::vector<int32_t> global_to_compact(n_clusters, -1);
    out.n_active = 0;
    for (int32_t i = 0; i < n_clusters; ++i) {
        if (used[i]) {
            global_to_compact[i] = out.n_active++;
        }
    }

    out.final_labels.resize(T);
    for (int32_t i = 0; i < T; ++i) {
        out.final_labels[i] = global_to_compact[combined[i]];
    }

    std::vector<int32_t> counts(out.n_active, 0);
    for (int32_t label : out.final_labels) {
        counts[label]++;
    }

    out.cluster_offsets.resize(out.n_active + 1);
    out.cluster_offsets[0] = 0;
    for (int32_t c = 0; c < out.n_active; ++c) {
        out.cluster_offsets[c + 1] = out.cluster_offsets[c] + counts[c];
    }

    const int dim = dim_half * 2;
    std::vector<float> merged_centroids(static_cast<size_t>(out.n_active) * dim);
    float* dst = merged_centroids.data();

    for (int32_t global = 0; global < n_clusters; ++global) {
        int32_t compact = global_to_compact[global];
        if (compact < 0) {
            continue;
        }

        int32_t i = global / k2;
        int32_t j = global % k2;

        std::memcpy(dst, &centroids1[i * dim_half], static_cast<size_t>(dim_half) * sizeof(float));
        std::memcpy(dst + dim_half, &centroids2[j * dim_half], static_cast<size_t>(dim_half) * sizeof(float));
        dst += dim;
    }

    return {std::move(out), std::move(merged_centroids)};
}

std::pair<HeadReorgInput, std::vector<float>> build_compact_reorg_input_4(
    const std::vector<int32_t>& labels1,
    const std::vector<int32_t>& labels2,
    const std::vector<int32_t>& labels3,
    const std::vector<int32_t>& labels4,
    const std::vector<float>& centroids1,
    const std::vector<float>& centroids2,
    const std::vector<float>& centroids3,
    const std::vector<float>& centroids4,
    int32_t T,
    int32_t dim_quarter
) {
    HeadReorgInput out;

    const int32_t k1 = static_cast<int32_t>(centroids1.size()) / dim_quarter;
    const int32_t k2 = static_cast<int32_t>(centroids2.size()) / dim_quarter;
    const int32_t k3 = static_cast<int32_t>(centroids3.size()) / dim_quarter;
    const int32_t k4 = static_cast<int32_t>(centroids4.size()) / dim_quarter;
    const int64_t total_clusters =
        static_cast<int64_t>(k1) * k2 * k3 * k4;
    if (total_clusters > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("IMI 4-way clusters exceed int32 limit");
    }
    const int32_t n_clusters = static_cast<int32_t>(total_clusters);

    std::vector<int32_t> combined(T);
    for (int32_t i = 0; i < T; ++i) {
        combined[i] = static_cast<int32_t>(
            (((static_cast<int64_t>(labels1[i]) * k2 + labels2[i]) * k3 + labels3[i]) * k4
             + labels4[i])
        );
    }

    std::vector<uint8_t> used(n_clusters, 0);
    for (int32_t id : combined) {
        used[id] = 1;
    }

    std::vector<int32_t> global_to_compact(n_clusters, -1);
    out.n_active = 0;
    for (int32_t i = 0; i < n_clusters; ++i) {
        if (used[i]) {
            global_to_compact[i] = out.n_active++;
        }
    }

    out.final_labels.resize(T);
    for (int32_t i = 0; i < T; ++i) {
        out.final_labels[i] = global_to_compact[combined[i]];
    }

    std::vector<int32_t> counts(out.n_active, 0);
    for (int32_t label : out.final_labels) {
        counts[label]++;
    }

    out.cluster_offsets.resize(out.n_active + 1);
    out.cluster_offsets[0] = 0;
    for (int32_t c = 0; c < out.n_active; ++c) {
        out.cluster_offsets[c + 1] = out.cluster_offsets[c] + counts[c];
    }

    const int dim = dim_quarter * 4;
    std::vector<float> merged_centroids(static_cast<size_t>(out.n_active) * dim);
    float* dst = merged_centroids.data();

    const int32_t stride_234 = k2 * k3 * k4;
    const int32_t stride_34 = k3 * k4;
    for (int32_t global = 0; global < n_clusters; ++global) {
        int32_t compact = global_to_compact[global];
        if (compact < 0) {
            continue;
        }

        int32_t i1 = global / stride_234;
        int32_t rem1 = global % stride_234;
        int32_t i2 = rem1 / stride_34;
        int32_t rem2 = rem1 % stride_34;
        int32_t i3 = rem2 / k4;
        int32_t i4 = rem2 % k4;

        std::memcpy(dst, &centroids1[i1 * dim_quarter], static_cast<size_t>(dim_quarter) * sizeof(float));
        std::memcpy(dst + dim_quarter, &centroids2[i2 * dim_quarter], static_cast<size_t>(dim_quarter) * sizeof(float));
        std::memcpy(dst + 2 * dim_quarter, &centroids3[i3 * dim_quarter], static_cast<size_t>(dim_quarter) * sizeof(float));
        std::memcpy(dst + 3 * dim_quarter, &centroids4[i4 * dim_quarter], static_cast<size_t>(dim_quarter) * sizeof(float));
        dst += dim;
    }

    return {std::move(out), std::move(merged_centroids)};
}

std::pair<HeadReorgInput, std::vector<float>> build_compact_reorg_input_ivf(
    const std::vector<int32_t>& labels,
    const std::vector<float>& centroids,
    int32_t T,
    int32_t dim
) {
    HeadReorgInput out;

    const int32_t n_clusters = static_cast<int32_t>(centroids.size()) / dim;

    std::vector<uint8_t> used(n_clusters, 0);
    for (int32_t id : labels) {
        if (id >= 0 && id < n_clusters) {
            used[id] = 1;
        }
    }

    std::vector<int32_t> global_to_compact(n_clusters, -1);
    out.n_active = 0;
    for (int32_t i = 0; i < n_clusters; ++i) {
        if (used[i]) {
            global_to_compact[i] = out.n_active++;
        }
    }

    out.final_labels.resize(T);
    for (int32_t i = 0; i < T; ++i) {
        out.final_labels[i] = global_to_compact[labels[i]];
    }

    std::vector<int32_t> counts(out.n_active, 0);
    for (int32_t label : out.final_labels) {
        counts[label]++;
    }

    out.cluster_offsets.resize(out.n_active + 1);
    out.cluster_offsets[0] = 0;
    for (int32_t c = 0; c < out.n_active; ++c) {
        out.cluster_offsets[c + 1] = out.cluster_offsets[c] + counts[c];
    }

    std::vector<float> compact_centroids(static_cast<size_t>(out.n_active) * dim);
    float* dst = compact_centroids.data();
    for (int32_t global = 0; global < n_clusters; ++global) {
        int32_t compact = global_to_compact[global];
        if (compact < 0) {
            continue;
        }
        std::memcpy(dst, &centroids[global * dim], static_cast<size_t>(dim) * sizeof(float));
        dst += dim;
    }

    return {std::move(out), std::move(compact_centroids)};
}

std::pair<HeadReorgInput, std::vector<float>> build_compact_reorg_input_m(
    const std::vector<std::vector<int32_t>>& labels_m,
    const std::vector<std::vector<float>>& centroids_m,
    const std::vector<int32_t>& subspace_dims,
    const std::vector<std::vector<int32_t>>& pair_groups,
    int32_t T,
    int32_t dim
) {
    HeadReorgInput out;

    const int32_t M = static_cast<int32_t>(labels_m.size());
    if (M <= 0) {
        throw std::runtime_error("learned M-way reorg requires at least one subspace");
    }
    if (
        static_cast<int32_t>(centroids_m.size()) != M ||
        static_cast<int32_t>(subspace_dims.size()) != M ||
        static_cast<int32_t>(pair_groups.size()) != M
    ) {
        throw std::runtime_error("learned M-way reorg metadata size mismatch");
    }
    if (dim % 2 != 0) {
        throw std::runtime_error("learned M-way reorg requires even head dim");
    }

    const int32_t half = dim / 2;
    std::vector<int32_t> k_values(M, 0);
    for (int32_t m = 0; m < M; ++m) {
        if (static_cast<int32_t>(labels_m[m].size()) != T) {
            throw std::runtime_error("learned M-way labels length mismatch");
        }
        const int32_t expected_dim = static_cast<int32_t>(pair_groups[m].size()) * 2;
        if (expected_dim <= 0 || subspace_dims[m] != expected_dim) {
            throw std::runtime_error("learned M-way subspace dim mismatch");
        }
        if (static_cast<int32_t>(centroids_m[m].size()) % subspace_dims[m] != 0) {
            throw std::runtime_error("learned M-way centroid buffer has invalid shape");
        }
        k_values[m] = static_cast<int32_t>(centroids_m[m].size()) / subspace_dims[m];
        if (k_values[m] <= 0) {
            throw std::runtime_error("learned M-way subspace has no centroids");
        }
        for (int32_t pair : pair_groups[m]) {
            if (pair < 0 || pair >= half) {
                throw std::runtime_error("learned M-way pair index out of range during reorg");
            }
        }
    }

    int64_t code_space64 = 1;
    std::vector<int32_t> strides(M, 1);
    for (int32_t m = 0; m < M; ++m) {
        strides[m] = static_cast<int32_t>(code_space64);
        if (code_space64 > std::numeric_limits<int32_t>::max() / std::max(1, k_values[m])) {
            throw std::runtime_error("learned M-way dense code space exceeds int32 limit");
        }
        code_space64 *= std::max(1, k_values[m]);
    }
    const int32_t code_space = static_cast<int32_t>(code_space64);
    std::vector<int32_t> dense_counts(code_space, 0);
    std::vector<int32_t> code_to_compact(code_space, -1);
    out.final_labels.resize(T);

    for (int32_t token = 0; token < T; ++token) {
        int32_t dense_code = 0;
        for (int32_t m = 0; m < M; ++m) {
            int32_t label = labels_m[m][token];
            if (label < 0 || label >= k_values[m]) {
                throw std::runtime_error("learned M-way label out of range");
            }
            dense_code += label * strides[m];
        }
        out.final_labels[token] = dense_code;
        dense_counts[dense_code]++;
    }

    out.n_active = 0;
    for (int32_t code = 0; code < code_space; ++code) {
        if (dense_counts[code] > 0) {
            code_to_compact[code] = out.n_active++;
        }
    }
    for (int32_t token = 0; token < T; ++token) {
        out.final_labels[token] = code_to_compact[out.final_labels[token]];
    }

    out.compact_codes.resize(static_cast<size_t>(out.n_active) * M);
    std::vector<int32_t> counts(out.n_active, 0);
    for (int32_t code = 0; code < code_space; ++code) {
        const int32_t compact = code_to_compact[code];
        if (compact < 0) {
            continue;
        }
        counts[compact] = dense_counts[code];
        int32_t tmp = code;
        for (int32_t m = 0; m < M; ++m) {
            out.compact_codes[static_cast<size_t>(compact) * M + m] = tmp % k_values[m];
            tmp /= k_values[m];
        }
    }

    out.cluster_offsets.resize(out.n_active + 1);
    out.cluster_offsets[0] = 0;
    for (int32_t c = 0; c < out.n_active; ++c) {
        out.cluster_offsets[c + 1] = out.cluster_offsets[c] + counts[c];
    }

    std::vector<float> merged_centroids(static_cast<size_t>(out.n_active) * dim, 0.0f);

    return {std::move(out), std::move(merged_centroids)};
}

static inline float fp16_to_fp32_local(uint16_t h) {
    union { uint32_t i; float f; } u;

    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mantissa = (h & 0x03FF) << 13;

    if (exp == 0x1F) {
        u.i = sign | 0x7F800000 | mantissa;
    } else if (exp == 0) {
        if (mantissa == 0) {
            u.i = sign;
        } else {
            exp = 0x71;
            while ((mantissa & 0x00400000) == 0) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x007FFFFF;
            u.i = sign | (exp << 23) | mantissa;
        }
    } else {
        u.i = sign | ((exp + 112) << 23) | mantissa;
    }

    return u.f;
}

static inline float bf16_to_fp32_local(uint16_t h) {
    union { uint32_t i; float f; } u;
    u.i = static_cast<uint32_t>(h) << 16;
    return u.f;
}

static inline float load_scalar_as_float(const char* ptr, c10::ScalarType dtype) {
    if (dtype == c10::ScalarType::Float) {
        return *reinterpret_cast<const float*>(ptr);
    }
    if (dtype == c10::ScalarType::Half) {
        return fp16_to_fp32_local(*reinterpret_cast<const uint16_t*>(ptr));
    }
    if (dtype == c10::ScalarType::BFloat16) {
        return bf16_to_fp32_local(*reinterpret_cast<const uint16_t*>(ptr));
    }
    throw std::runtime_error("unsupported dtype for centroid mean computation");
}


static ReorganizeResult reorganize_single_head(
    const void* keys_data,
    const void* values_data,
    const std::vector<int32_t>& labels,
    const std::vector<int32_t>& cluster_offsets,
    int32_t n_tokens,
    int32_t dim,
    int32_t n_clusters,
    size_t element_size,
    c10::ScalarType dtype,
    void* output_keys,
    void* output_values,
    size_t output_capacity,
    size_t source_token_stride_bytes,
    float* output_centroids
) {
    if (static_cast<int>(cluster_offsets.size()) != n_clusters + 1) {
        throw std::runtime_error("cluster_offsets has incorrect length");
    }

    ReorganizeResult result;

    size_t token_bytes = dim * element_size;
    if (source_token_stride_bytes == 0) {
        source_token_stride_bytes = token_bytes;
    }

    char* dst_keys = nullptr;
    char* dst_values = nullptr;
    if (output_keys != nullptr || output_values != nullptr) {
        if (output_keys == nullptr || output_values == nullptr) {
            throw std::runtime_error("output_keys and output_values must both be provided");
        }
        if (output_capacity < static_cast<size_t>(n_tokens)) {
            throw std::runtime_error("output buffer capacity is smaller than token count");
        }
        dst_keys = static_cast<char*>(output_keys);
        dst_values = static_cast<char*>(output_values);
    } else {
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(torch::kCPU);

        result.reorganized_keys = torch::empty({n_tokens, dim}, options);
        result.reorganized_values = torch::empty({n_tokens, dim}, options);
        dst_keys = static_cast<char*>(result.reorganized_keys.data_ptr());
        dst_values = static_cast<char*>(result.reorganized_values.data_ptr());
    }

    result.cluster_offsets = cluster_offsets;
    result.cluster_sizes.resize(n_clusters);
    for (int32_t c = 0; c < n_clusters; ++c) {
        result.cluster_sizes[c] = cluster_offsets[c + 1] - cluster_offsets[c];
    }

    const char* src_keys = static_cast<const char*>(keys_data);
    const char* src_values = static_cast<const char*>(values_data);

    std::vector<int32_t> write_positions(cluster_offsets.begin(), cluster_offsets.end() - 1);
    std::vector<int32_t> reorder_indices(n_tokens, 0);

    // 第一遍：根据 labels 计算每个 token 在新缓冲区的位置
    for (int32_t token_idx = 0; token_idx < n_tokens; ++token_idx) {
        int32_t cluster_id = labels[token_idx];
        int32_t write_pos = write_positions[cluster_id]++;
        reorder_indices[write_pos] = token_idx;
    }

    // 第二遍：按新顺序拷贝，每个位置独立，可并行
    at::parallel_for(0, static_cast<int64_t>(n_tokens), 1024, [&](int64_t begin, int64_t end) {
        for (int64_t pos = begin; pos < end; ++pos) {
            int32_t src_idx = reorder_indices[pos];
            const char* src_key = src_keys + static_cast<size_t>(src_idx) * source_token_stride_bytes;
            const char* src_value = src_values + static_cast<size_t>(src_idx) * source_token_stride_bytes;
            char* dst_key = dst_keys + static_cast<size_t>(pos) * token_bytes;
            char* dst_value = dst_values + static_cast<size_t>(pos) * token_bytes;
            std::memcpy(dst_key, src_key, token_bytes);
            std::memcpy(dst_value, src_value, token_bytes);
        }
    });

    if (output_centroids != nullptr) {
        at::parallel_for(0, static_cast<int64_t>(n_clusters), 1, [&](int64_t begin, int64_t end) {
            for (int64_t cluster = begin; cluster < end; ++cluster) {
                const int32_t start = cluster_offsets[cluster];
                const int32_t end_pos = cluster_offsets[cluster + 1];
                const int32_t size = end_pos - start;
                float* centroid = output_centroids + static_cast<size_t>(cluster) * dim;
                std::fill(centroid, centroid + dim, 0.0f);
                if (size <= 0) {
                    continue;
                }
                for (int32_t pos = start; pos < end_pos; ++pos) {
                    const char* key = dst_keys + static_cast<size_t>(pos) * token_bytes;
                    for (int32_t d = 0; d < dim; ++d) {
                        centroid[d] += load_scalar_as_float(key + static_cast<size_t>(d) * element_size, dtype);
                    }
                }
                const float inv_size = 1.0f / static_cast<float>(size);
                for (int32_t d = 0; d < dim; ++d) {
                    centroid[d] *= inv_size;
                }
            }
        });
    }

    result.token_ids_by_cluster = std::move(reorder_indices);
    return result;
}

void reorganize_from_kmeans(
    const torch::Tensor& keys_cpu,
    const torch::Tensor& values_cpu,
    const KmeansResult& kmeans_result,
    int32_t total_tokens,
    int32_t dim_half,
    std::vector<ReorganizeResult>& out_results,
    std::vector<torch::Tensor>& merged_centroids,
    const std::vector<CPUBufferDescriptor>* output_buffers
) {
    int32_t kv_heads = static_cast<int32_t>(kmeans_result.kv_heads);
    if (kv_heads == 0) {
        out_results.clear();
        merged_centroids.clear();
        return;
    }

    TORCH_CHECK(keys_cpu.is_contiguous(), "keys_cpu must be contiguous");
    TORCH_CHECK(values_cpu.is_contiguous(), "values_cpu must be contiguous");

    int32_t dim = keys_cpu.size(2);
    size_t element_size = keys_cpu.element_size();
    c10::ScalarType dtype = keys_cpu.scalar_type();

    out_results.resize(kv_heads);
    merged_centroids.resize(kv_heads);

    const char* keys_base = static_cast<const char*>(keys_cpu.data_ptr());
    const char* values_base = static_cast<const char*>(values_cpu.data_ptr());

    at::parallel_for(0, kv_heads, 1, [&](int64_t begin, int64_t end) {
        for (int64_t head = begin; head < end; ++head) {
            const auto& labels1 = kmeans_result.per_head_labels1[head];
            const auto& labels2 = kmeans_result.per_head_labels2[head];
            const auto& centroids1 = kmeans_result.per_head_centroids1[head];
            const auto& centroids2 = kmeans_result.per_head_centroids2[head];

            auto compact_pair = build_compact_reorg_input(
                labels1, labels2,
                centroids1, centroids2,
                total_tokens, dim_half
            );

            auto& reorg_input = compact_pair.first;

            auto centroids_storage = std::make_shared<std::vector<float>>(std::move(compact_pair.second));
            float* storage_ptr = centroids_storage->data();
            merged_centroids[head] = torch::from_blob(
                storage_ptr,
                {reorg_input.n_active, dim},
                [centroids_storage](void*) mutable { centroids_storage.reset(); },
                torch::dtype(torch::kFloat32)
            );

            const void* keys_head_ptr = keys_base + static_cast<size_t>(head) * total_tokens * dim * element_size;
            const void* values_head_ptr = values_base + static_cast<size_t>(head) * total_tokens * dim * element_size;

            const CPUBufferDescriptor* output_desc = nullptr;
            if (output_buffers != nullptr) {
                if (head >= static_cast<int64_t>(output_buffers->size())) {
                    throw std::runtime_error("output buffers size mismatch with kv_heads");
                }
                output_desc = &(*output_buffers)[head];
            }

            auto head_result = reorganize_single_head(
                keys_head_ptr,
                values_head_ptr,
                reorg_input.final_labels,
                reorg_input.cluster_offsets,
                total_tokens,
                dim,
                reorg_input.n_active,
                element_size,
                dtype,
                output_desc ? output_desc->key_buffer_ptr : nullptr,
                output_desc ? output_desc->value_buffer_ptr : nullptr,
                output_desc ? output_desc->buffer_capacity : 0,
                0,
                nullptr
            );
            head_result.compact_codes = std::move(reorg_input.compact_codes);
            out_results[head] = std::move(head_result);
        }
    });
}

void reorganize_from_kmeans_ivf(
    const torch::Tensor& keys_cpu,
    const torch::Tensor& values_cpu,
    const KmeansResult& kmeans_result,
    int32_t total_tokens,
    std::vector<ReorganizeResult>& out_results,
    std::vector<torch::Tensor>& merged_centroids,
    const std::vector<CPUBufferDescriptor>* output_buffers
) {
    int32_t kv_heads = static_cast<int32_t>(kmeans_result.kv_heads);
    if (kv_heads == 0) {
        out_results.clear();
        merged_centroids.clear();
        return;
    }

    TORCH_CHECK(keys_cpu.is_contiguous(), "keys_cpu must be contiguous");
    TORCH_CHECK(values_cpu.is_contiguous(), "values_cpu must be contiguous");

    int32_t dim = keys_cpu.size(2);
    size_t element_size = keys_cpu.element_size();
    c10::ScalarType dtype = keys_cpu.scalar_type();

    out_results.resize(kv_heads);
    merged_centroids.resize(kv_heads);

    const char* keys_base = static_cast<const char*>(keys_cpu.data_ptr());
    const char* values_base = static_cast<const char*>(values_cpu.data_ptr());

    at::parallel_for(0, kv_heads, 1, [&](int64_t begin, int64_t end) {
        for (int64_t head = begin; head < end; ++head) {
            const auto& labels = kmeans_result.per_head_labels1[head];
            const auto& centroids = kmeans_result.per_head_centroids1[head];

            auto compact_pair = build_compact_reorg_input_ivf(
                labels,
                centroids,
                total_tokens,
                dim
            );

            auto& reorg_input = compact_pair.first;

            auto centroids_storage = std::make_shared<std::vector<float>>(std::move(compact_pair.second));
            float* storage_ptr = centroids_storage->data();
            merged_centroids[head] = torch::from_blob(
                storage_ptr,
                {reorg_input.n_active, dim},
                [centroids_storage](void*) mutable { centroids_storage.reset(); },
                torch::dtype(torch::kFloat32)
            );

            const void* keys_head_ptr = keys_base + static_cast<size_t>(head) * total_tokens * dim * element_size;
            const void* values_head_ptr = values_base + static_cast<size_t>(head) * total_tokens * dim * element_size;

            const CPUBufferDescriptor* output_desc = nullptr;
            if (output_buffers != nullptr) {
                if (head >= static_cast<int64_t>(output_buffers->size())) {
                    throw std::runtime_error("output buffers size mismatch with kv_heads");
                }
                output_desc = &(*output_buffers)[head];
            }

            auto head_result = reorganize_single_head(
                keys_head_ptr,
                values_head_ptr,
                reorg_input.final_labels,
                reorg_input.cluster_offsets,
                total_tokens,
                dim,
                reorg_input.n_active,
                element_size,
                dtype,
                output_desc ? output_desc->key_buffer_ptr : nullptr,
                output_desc ? output_desc->value_buffer_ptr : nullptr,
                output_desc ? output_desc->buffer_capacity : 0,
                0,
                nullptr
            );
            head_result.compact_codes = std::move(reorg_input.compact_codes);
            out_results[head] = std::move(head_result);
        }
    });
}

void reorganize_from_kmeans_4(
    const torch::Tensor& keys_cpu,
    const torch::Tensor& values_cpu,
    const KmeansResult& kmeans_result,
    int32_t total_tokens,
    int32_t dim_quarter,
    std::vector<ReorganizeResult>& out_results,
    std::vector<torch::Tensor>& merged_centroids,
    const std::vector<CPUBufferDescriptor>* output_buffers
) {
    int32_t kv_heads = static_cast<int32_t>(kmeans_result.kv_heads);
    if (kv_heads == 0) {
        out_results.clear();
        merged_centroids.clear();
        return;
    }

    TORCH_CHECK(keys_cpu.is_contiguous(), "keys_cpu must be contiguous");
    TORCH_CHECK(values_cpu.is_contiguous(), "values_cpu must be contiguous");

    int32_t dim = keys_cpu.size(2);
    size_t element_size = keys_cpu.element_size();
    c10::ScalarType dtype = keys_cpu.scalar_type();

    out_results.resize(kv_heads);
    merged_centroids.resize(kv_heads);

    const char* keys_base = static_cast<const char*>(keys_cpu.data_ptr());
    const char* values_base = static_cast<const char*>(values_cpu.data_ptr());

    at::parallel_for(0, kv_heads, 1, [&](int64_t begin, int64_t end) {
        for (int64_t head = begin; head < end; ++head) {
            const auto& labels1 = kmeans_result.per_head_labels1[head];
            const auto& labels2 = kmeans_result.per_head_labels2[head];
            const auto& labels3 = kmeans_result.per_head_labels3[head];
            const auto& labels4 = kmeans_result.per_head_labels4[head];
            const auto& centroids1 = kmeans_result.per_head_centroids1[head];
            const auto& centroids2 = kmeans_result.per_head_centroids2[head];
            const auto& centroids3 = kmeans_result.per_head_centroids3[head];
            const auto& centroids4 = kmeans_result.per_head_centroids4[head];

            auto compact_pair = build_compact_reorg_input_4(
                labels1,
                labels2,
                labels3,
                labels4,
                centroids1,
                centroids2,
                centroids3,
                centroids4,
                total_tokens,
                dim_quarter
            );

            auto& reorg_input = compact_pair.first;

            auto centroids_storage = std::make_shared<std::vector<float>>(std::move(compact_pair.second));
            float* storage_ptr = centroids_storage->data();
            merged_centroids[head] = torch::from_blob(
                storage_ptr,
                {reorg_input.n_active, dim},
                [centroids_storage](void*) mutable { centroids_storage.reset(); },
                torch::dtype(torch::kFloat32)
            );

            const void* keys_head_ptr = keys_base + static_cast<size_t>(head) * total_tokens * dim * element_size;
            const void* values_head_ptr = values_base + static_cast<size_t>(head) * total_tokens * dim * element_size;

            const CPUBufferDescriptor* output_desc = nullptr;
            if (output_buffers != nullptr) {
                if (head >= static_cast<int64_t>(output_buffers->size())) {
                    throw std::runtime_error("output buffers size mismatch with kv_heads");
                }
                output_desc = &(*output_buffers)[head];
            }

            auto head_result = reorganize_single_head(
                keys_head_ptr,
                values_head_ptr,
                reorg_input.final_labels,
                reorg_input.cluster_offsets,
                total_tokens,
                dim,
                reorg_input.n_active,
                element_size,
                dtype,
                output_desc ? output_desc->key_buffer_ptr : nullptr,
                output_desc ? output_desc->value_buffer_ptr : nullptr,
                output_desc ? output_desc->buffer_capacity : 0,
                0,
                nullptr
            );
        }
    });
}

void reorganize_from_kmeans_m(
    const torch::Tensor& keys_cpu,
    const torch::Tensor& values_cpu,
    const KmeansResult& kmeans_result,
    const std::vector<std::vector<std::vector<int32_t>>>& head_pair_groups,
    int32_t total_tokens,
    std::vector<ReorganizeResult>& out_results,
    std::vector<torch::Tensor>& merged_centroids,
    const std::vector<CPUBufferDescriptor>* output_buffers,
    bool sequence_major
) {
    int32_t kv_heads = static_cast<int32_t>(kmeans_result.kv_heads);
    if (kv_heads == 0) {
        out_results.clear();
        merged_centroids.clear();
        return;
    }

    TORCH_CHECK(keys_cpu.is_contiguous(), "keys_cpu must be contiguous");
    TORCH_CHECK(values_cpu.is_contiguous(), "values_cpu must be contiguous");

    if (static_cast<int32_t>(head_pair_groups.size()) != kv_heads) {
        throw std::runtime_error("learned M-way reorg head partition count mismatch");
    }

    int32_t dim = keys_cpu.size(2);
    size_t element_size = keys_cpu.element_size();
    c10::ScalarType dtype = keys_cpu.scalar_type();

    out_results.resize(kv_heads);
    merged_centroids.resize(kv_heads);

    const char* keys_base = static_cast<const char*>(keys_cpu.data_ptr());
    const char* values_base = static_cast<const char*>(values_cpu.data_ptr());
    const size_t token_bytes = static_cast<size_t>(dim) * element_size;
    const size_t head_stride_bytes = static_cast<size_t>(total_tokens) * token_bytes;
    const size_t seq_token_stride_bytes = static_cast<size_t>(kv_heads) * token_bytes;

    at::parallel_for(0, kv_heads, 1, [&](int64_t begin, int64_t end) {
        for (int64_t head = begin; head < end; ++head) {
            auto compact_pair = build_compact_reorg_input_m(
                kmeans_result.per_head_labels_m[head],
                kmeans_result.per_head_centroids_m[head],
                kmeans_result.per_head_subspace_dims[head],
                head_pair_groups[head],
                total_tokens,
                dim
            );

            auto& reorg_input = compact_pair.first;

            auto centroids_storage = std::make_shared<std::vector<float>>(std::move(compact_pair.second));
            float* storage_ptr = centroids_storage->data();
            merged_centroids[head] = torch::from_blob(
                storage_ptr,
                {reorg_input.n_active, dim},
                [centroids_storage](void*) mutable { centroids_storage.reset(); },
                torch::dtype(torch::kFloat32)
            );

            const void* keys_head_ptr = sequence_major
                ? static_cast<const void*>(keys_base + static_cast<size_t>(head) * token_bytes)
                : static_cast<const void*>(keys_base + static_cast<size_t>(head) * head_stride_bytes);
            const void* values_head_ptr = sequence_major
                ? static_cast<const void*>(values_base + static_cast<size_t>(head) * token_bytes)
                : static_cast<const void*>(values_base + static_cast<size_t>(head) * head_stride_bytes);

            const CPUBufferDescriptor* output_desc = nullptr;
            if (output_buffers != nullptr) {
                if (head >= static_cast<int64_t>(output_buffers->size())) {
                    throw std::runtime_error("output buffers size mismatch with kv_heads");
                }
                output_desc = &(*output_buffers)[head];
            }

            auto head_result = reorganize_single_head(
                keys_head_ptr,
                values_head_ptr,
                reorg_input.final_labels,
                reorg_input.cluster_offsets,
                total_tokens,
                dim,
                reorg_input.n_active,
                element_size,
                dtype,
                output_desc ? output_desc->key_buffer_ptr : nullptr,
                output_desc ? output_desc->value_buffer_ptr : nullptr,
                output_desc ? output_desc->buffer_capacity : 0,
                sequence_major ? seq_token_stride_bytes : token_bytes,
                storage_ptr
            );
            head_result.compact_codes = std::move(reorg_input.compact_codes);
            out_results[head] = std::move(head_result);
        }
    });
}

} // namespace imi
