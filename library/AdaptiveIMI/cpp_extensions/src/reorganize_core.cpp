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
    size_t output_capacity
) {
    if (static_cast<int>(cluster_offsets.size()) != n_clusters + 1) {
        throw std::runtime_error("cluster_offsets has incorrect length");
    }

    ReorganizeResult result;

    size_t token_bytes = dim * element_size;

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
            const char* src_key = src_keys + static_cast<size_t>(src_idx) * token_bytes;
            const char* src_value = src_values + static_cast<size_t>(src_idx) * token_bytes;
            char* dst_key = dst_keys + static_cast<size_t>(pos) * token_bytes;
            char* dst_value = dst_values + static_cast<size_t>(pos) * token_bytes;
            std::memcpy(dst_key, src_key, token_bytes);
            std::memcpy(dst_value, src_value, token_bytes);
        }
    });

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

            out_results[head] = reorganize_single_head(
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
                output_desc ? output_desc->buffer_capacity : 0
            );
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

            out_results[head] = reorganize_single_head(
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
                output_desc ? output_desc->buffer_capacity : 0
            );
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

            out_results[head] = reorganize_single_head(
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
                output_desc ? output_desc->buffer_capacity : 0
            );
        }
    });
}

} // namespace imi
