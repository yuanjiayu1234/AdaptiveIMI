// ============================================================================
// reorganize_core.cpp - 剥离的Reorganize核心算法（纯功能函数，无封装）
// ============================================================================
#include "layer_pipeline.hpp"
#include "reorder_core.hpp"
#include <cstring>
#include <algorithm>
#include <memory>
#include <vector>
#include <limits>
#include <cstdint>
#include <ATen/Parallel.h>
#include <c10/core/ScalarType.h>

namespace imi {

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

    uint64_t code_space = 1;
    std::vector<uint64_t> strides(M, 1);
    for (int32_t m = 0; m < M; ++m) {
        const uint64_t k = static_cast<uint64_t>(std::max(1, k_values[m]));
        strides[m] = code_space;
        if (code_space > std::numeric_limits<uint64_t>::max() / k) {
            throw std::runtime_error("learned M-way packed code space exceeds uint64 limit");
        }
        code_space *= k;
    }

    std::vector<uint64_t> token_codes(T);
    for (int32_t token = 0; token < T; ++token) {
        uint64_t code = 0;
        for (int32_t m = 0; m < M; ++m) {
            int32_t label = labels_m[m][token];
            if (label < 0 || label >= k_values[m]) {
                throw std::runtime_error("learned M-way label out of range");
            }
            code += static_cast<uint64_t>(label) * strides[m];
        }
        token_codes[token] = code;
    }

    std::vector<uint64_t> active_codes = token_codes;
    std::sort(active_codes.begin(), active_codes.end());
    active_codes.erase(std::unique(active_codes.begin(), active_codes.end()), active_codes.end());
    if (active_codes.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("learned M-way active cluster count exceeds int32 limit");
    }
    out.n_active = static_cast<int32_t>(active_codes.size());

    out.compact_codes.resize(static_cast<size_t>(out.n_active) * M);
    std::vector<int32_t> counts(out.n_active, 0);
    for (int32_t compact = 0; compact < out.n_active; ++compact) {
        uint64_t tmp = active_codes[compact];
        for (int32_t m = 0; m < M; ++m) {
            const uint64_t k = static_cast<uint64_t>(k_values[m]);
            out.compact_codes[static_cast<size_t>(compact) * M + m] =
                static_cast<int32_t>(tmp % k);
            tmp /= k;
        }
    }

    out.final_labels.resize(T);
    for (int32_t token = 0; token < T; ++token) {
        auto it = std::lower_bound(active_codes.begin(), active_codes.end(), token_codes[token]);
        const int32_t compact = static_cast<int32_t>(it - active_codes.begin());
        out.final_labels[token] = compact;
        counts[compact]++;
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

    const int32_t row_token_count = n_tokens;
    if (source_token_stride_bytes == static_cast<size_t>(token_bytes)) {
        copy_kv_by_token_order_rows_cpu(
            src_keys,
            src_values,
            reorder_indices.data(),
            &row_token_count,
            1,
            n_tokens,
            n_tokens,
            dim,
            element_size,
            static_cast<size_t>(n_tokens) * token_bytes,
            static_cast<size_t>(n_tokens) * token_bytes,
            dst_keys,
            dst_values,
            n_tokens
        );
    } else {
        at::parallel_for(0, static_cast<int64_t>(n_tokens), 32, [&](int64_t begin, int64_t end) {
            for (int64_t dst = begin; dst < end; ++dst) {
                const int32_t src = reorder_indices[dst];
                if (src < 0 || src >= n_tokens) {
                    throw std::runtime_error("token id is out of source range");
                }
                std::memcpy(
                    dst_keys + dst * token_bytes,
                    src_keys + static_cast<size_t>(src) * source_token_stride_bytes,
                    static_cast<size_t>(token_bytes)
                );
                std::memcpy(
                    dst_values + dst * token_bytes,
                    src_values + static_cast<size_t>(src) * source_token_stride_bytes,
                    static_cast<size_t>(token_bytes)
                );
            }
        });
    }

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
