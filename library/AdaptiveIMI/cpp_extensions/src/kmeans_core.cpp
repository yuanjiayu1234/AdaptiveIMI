#include "layer_pipeline.hpp"
#include <omp.h>
#include <immintrin.h>  // AVX2
#include <algorithm>
#include <limits>
#include <cstring>
#include <atomic>  
#include <c10/core/ScalarType.h>

namespace imi {

struct LloydRuntimeConfig {
    int32_t max_iters = 25;
};

static LloydRuntimeConfig load_lloyd_runtime_config(const LayerPipelineRuntimeConfig& runtime_config) {
    LloydRuntimeConfig cfg;
    cfg.max_iters = std::max<int32_t>(1, runtime_config.kmeans_max_iters);
    return cfg;
}

namespace {

std::atomic<int> g_active_kmeans{0};

class KmeansConcurrencyGuard {
public:
    explicit KmeansConcurrencyGuard(std::atomic<int>& counter) : counter_(counter) {
        active_ = counter_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    ~KmeansConcurrencyGuard() {
        counter_.fetch_sub(1, std::memory_order_relaxed);
    }

    int active() const {
        return active_;
    }

private:
    std::atomic<int>& counter_;
    int active_ = 1;
};

void compute_kmeans_thread_budget(
    int worker_threads,
    int active_kmeans,
    int min_threads_per_phase,
    int& effective_threads,
    int& effective_min_threads
) {
    int available = std::max(1, worker_threads);
    int active = std::max(1, active_kmeans);
    effective_threads = std::max(1, available / active);
    effective_threads = std::min(effective_threads, available);

    int min_threads = std::max(1, std::min(min_threads_per_phase, available));
    effective_min_threads = std::min(min_threads, effective_threads);
}

uint64_t splitmix64_cpu(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

uint64_t make_seed_cpu(int row, int subspace, int k, int iter, uint64_t salt) {
    uint64_t seed = salt;
    seed ^= static_cast<uint64_t>(row + 1) * 0x9E3779B185EBCA87ULL;
    seed ^= static_cast<uint64_t>(subspace + 42) * 0xC2B2AE3D27D4EB4FULL;
    seed ^= static_cast<uint64_t>(k + 1) * 0x165667B19E3779F9ULL;
    seed ^= static_cast<uint64_t>(iter + 12345) * 0x85EBCA77C2B2AE63ULL;
    return splitmix64_cpu(seed);
}

int sample_token_index_cpu(int token_count, int row, int subspace, int k, int iter, uint64_t salt) {
    if (token_count <= 1) {
        return 0;
    }
    return static_cast<int>(make_seed_cpu(row, subspace, k, iter, salt) % static_cast<uint64_t>(token_count));
}

} // namespace

static inline float fp16_to_fp32_kmeans_local(uint16_t h) {
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
            exp = 1;
            while ((mantissa & 0x800000) == 0) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x7FFFFF;
            u.i = sign | ((exp + 127 - 15) << 23) | mantissa;
        }
    } else {
        u.i = sign | ((exp + 127 - 15) << 23) | mantissa;
    }
    return u.f;
}

static inline float bf16_to_fp32_kmeans_local(uint16_t h) {
    union { uint32_t i; float f; } u;
    u.i = static_cast<uint32_t>(h) << 16;
    return u.f;
}

static inline float load_kv16_as_float(
    const uint16_t* keys_u16,
    c10::ScalarType dtype,
    int32_t kv_heads,
    int32_t n_tokens,
    int32_t dim,
    int32_t head,
    int32_t token,
    int32_t d,
    bool sequence_major
) {
    const size_t index = sequence_major
        ? (static_cast<size_t>(token) * kv_heads + head) * dim + d
        : (static_cast<size_t>(head) * n_tokens + token) * dim + d;
    const uint16_t raw = keys_u16[index];
    return dtype == c10::ScalarType::Half
        ? fp16_to_fp32_kmeans_local(raw)
        : bf16_to_fp32_kmeans_local(raw);
}

static inline void convert_kv16_row_to_fp32(
    const uint16_t* src,
    c10::ScalarType dtype,
    int32_t dim,
    float* dst
) {
    int32_t d = 0;
    if (dtype == c10::ScalarType::Half) {
#if defined(__F16C__)
        for (; d + 7 < dim; d += 8) {
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + d));
            __m256 fp32 = _mm256_cvtph_ps(raw);
            _mm256_storeu_ps(dst + d, fp32);
        }
#endif
        for (; d < dim; ++d) {
            dst[d] = fp16_to_fp32_kmeans_local(src[d]);
        }
    } else {
#if defined(__AVX2__)
        const __m128i zero = _mm_setzero_si128();
        for (; d + 7 < dim; d += 8) {
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + d));
            __m128i lo = _mm_unpacklo_epi16(raw, zero);
            __m128i hi = _mm_unpackhi_epi16(raw, zero);
            lo = _mm_slli_epi32(lo, 16);
            hi = _mm_slli_epi32(hi, 16);
            __m256 fp32 = _mm256_castps128_ps256(_mm_castsi128_ps(lo));
            fp32 = _mm256_insertf128_ps(fp32, _mm_castsi128_ps(hi), 1);
            _mm256_storeu_ps(dst + d, fp32);
        }
#endif
        for (; d < dim; ++d) {
            dst[d] = bf16_to_fp32_kmeans_local(src[d]);
        }
    }
}

static inline void load_head_token_tile_fp32(
    const uint16_t* keys_u16,
    c10::ScalarType dtype,
    int32_t kv_heads,
    int32_t n_tokens,
    int32_t dim,
    int32_t head,
    int32_t token_begin,
    int32_t token_end,
    bool sequence_major,
    std::vector<float>& tile
) {
    const int32_t tile_tokens = token_end - token_begin;
    tile.resize(static_cast<size_t>(tile_tokens) * dim);
    for (int32_t local = 0; local < tile_tokens; ++local) {
        const int32_t token = token_begin + local;
        float* dst = tile.data() + static_cast<size_t>(local) * dim;
        const size_t src_base = sequence_major
            ? (static_cast<size_t>(token) * kv_heads + head) * dim
            : (static_cast<size_t>(head) * n_tokens + token) * dim;
        convert_kv16_row_to_fp32(keys_u16 + src_base, dtype, dim, dst);
    }
}

static inline void gather_dim_ids_sample(
    const uint16_t* keys_u16,
    c10::ScalarType dtype,
    int32_t kv_heads,
    int32_t n_tokens,
    int32_t dim,
    int32_t head,
    int32_t token,
    const int32_t* dim_ids,
    int32_t dim_count,
    bool sequence_major,
    float* dst
) {
    for (int32_t d = 0; d < dim_count; ++d) {
        dst[d] = load_kv16_as_float(
            keys_u16, dtype, kv_heads, n_tokens, dim,
            head, token, dim_ids[d], sequence_major
        );
    }
}

static inline float distance_dim_ids(
    const float* token_fp32,
    const int32_t* dim_ids,
    int32_t dim_count,
    const float* centroid
) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int32_t d = 0;
    for (; d + 7 < dim_count; d += 8) {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dim_ids + d));
        __m256 x = _mm256_i32gather_ps(token_fp32, idx, 4);
        __m256 c = _mm256_loadu_ps(centroid + d);
        __m256 diff = _mm256_sub_ps(x, c);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float dist = _mm_cvtss_f32(sum4);
    for (; d < dim_count; ++d) {
        float diff = token_fp32[dim_ids[d]] - centroid[d];
        dist += diff * diff;
    }
    return dist;
#else
    float dist = 0.0f;
    for (int32_t d = 0; d < dim_count; ++d) {
        float diff = token_fp32[dim_ids[d]] - centroid[d];
        dist += diff * diff;
    }
    return dist;
#endif
}

static inline void add_dim_ids_to_sum(
    const float* token_fp32,
    const int32_t* dim_ids,
    int32_t dim_count,
    float* sum
) {
#if defined(__AVX2__)
    int32_t d = 0;
    for (; d + 7 < dim_count; d += 8) {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dim_ids + d));
        __m256 x = _mm256_i32gather_ps(token_fp32, idx, 4);
        __m256 old = _mm256_loadu_ps(sum + d);
        _mm256_storeu_ps(sum + d, _mm256_add_ps(old, x));
    }
    for (; d < dim_count; ++d) {
        sum[d] += token_fp32[dim_ids[d]];
    }
#else
    for (int32_t d = 0; d < dim_count; ++d) {
        sum[d] += token_fp32[dim_ids[d]];
    }
#endif
}

class KmeansExecutor8WayFused {
public:
    static std::unique_ptr<KmeansResult> execute(
        const uint16_t* keys_u16,
        c10::ScalarType dtype,
        int32_t kv_heads,
        int32_t n_tokens,
        int32_t dim,
        const std::vector<int32_t>& k_values,
        const std::vector<std::vector<std::vector<int32_t>>>& head_pair_groups,
        int32_t worker_threads,
        const LloydRuntimeConfig& runtime_cfg,
        bool sequence_major
    ) {
        constexpr int32_t kStages = 8;
        if (dim % 2 != 0) {
            throw std::invalid_argument("8-way fused K-means requires even head dim");
        }
        if (n_tokens <= 0) {
            throw std::invalid_argument("8-way fused K-means requires positive token count");
        }
        if (static_cast<int32_t>(k_values.size()) != kStages) {
            throw std::invalid_argument("8-way fused K-means requires exactly 8 k_values");
        }
        if (static_cast<int32_t>(head_pair_groups.size()) != kv_heads) {
            throw std::invalid_argument("8-way fused K-means partition head count mismatch");
        }

        const int32_t half = dim / 2;
        int32_t max_k = 1;
        int32_t max_sub_dim = 1;
        uint64_t packed_code_space = 1;
        std::vector<int32_t> sub_dims(static_cast<size_t>(kv_heads) * kStages, 0);
        for (int32_t stage = 0; stage < kStages; ++stage) {
            const uint64_t k = static_cast<uint64_t>(std::max(1, k_values[stage]));
            if (packed_code_space > std::numeric_limits<uint64_t>::max() / k) {
                throw std::invalid_argument("8-way fused K-means k_values exceed uint64 packed code range");
            }
            packed_code_space *= k;
        }
        for (int32_t head = 0; head < kv_heads; ++head) {
            if (static_cast<int32_t>(head_pair_groups[head].size()) != kStages) {
                throw std::invalid_argument("8-way fused K-means requires 8 subspaces for every head");
            }
            std::vector<uint8_t> seen(half, 0);
            int32_t covered = 0;
            for (int32_t stage = 0; stage < kStages; ++stage) {
                const auto& pairs = head_pair_groups[head][stage];
                if (pairs.empty()) {
                    throw std::invalid_argument("8-way fused K-means contains empty subspace");
                }
                for (int32_t pair : pairs) {
                    if (pair < 0 || pair >= half) {
                        throw std::invalid_argument("8-way fused K-means pair index out of range");
                    }
                    if (seen[pair]) {
                        throw std::invalid_argument("8-way fused K-means pair index appears more than once");
                    }
                    seen[pair] = 1;
                    ++covered;
                }
                if (k_values[stage] <= 0) {
                    throw std::invalid_argument("8-way fused K-means k_values must be positive");
                }
                const int32_t sub_dim = static_cast<int32_t>(pairs.size()) * 2;
                sub_dims[static_cast<size_t>(head) * kStages + stage] = sub_dim;
                max_sub_dim = std::max(max_sub_dim, sub_dim);
                max_k = std::max(max_k, k_values[stage]);
            }
            if (covered != half) {
                throw std::invalid_argument("8-way fused K-means partition must cover every RoPE pair exactly once");
            }
        }

        const int32_t total_tasks = kv_heads * kStages;
        std::vector<int32_t> dim_counts(total_tasks, 0);
        std::vector<int32_t> dim_ids_flat(static_cast<size_t>(total_tasks) * max_sub_dim, 0);
        for (int32_t head = 0; head < kv_heads; ++head) {
            for (int32_t stage = 0; stage < kStages; ++stage) {
                const int32_t task = head * kStages + stage;
                const auto& pairs = head_pair_groups[head][stage];
                int32_t offset = 0;
                int32_t* dst = dim_ids_flat.data() + static_cast<size_t>(task) * max_sub_dim;
                for (int32_t pair : pairs) {
                    dst[offset++] = pair;
                    dst[offset++] = pair + half;
                }
                dim_counts[task] = offset;
            }
        }

        auto result = std::make_unique<KmeansResult>();
        result->kv_heads = kv_heads;
        result->n_tokens = n_tokens;
        result->dim = dim;
        result->subspace_parts = kStages;
        result->per_head_labels_m.resize(kv_heads);
        result->per_head_centroids_m.resize(kv_heads);
        result->per_head_subspace_dims.resize(kv_heads);
        result->per_head_n_clusters.assign(kv_heads, 0);
        for (int32_t head = 0; head < kv_heads; ++head) {
            result->per_head_labels_m[head].resize(kStages);
            result->per_head_centroids_m[head].resize(kStages);
            result->per_head_subspace_dims[head].resize(kStages);
            for (int32_t stage = 0; stage < kStages; ++stage) {
                const int32_t sub_dim = sub_dims[static_cast<size_t>(head) * kStages + stage];
                result->per_head_subspace_dims[head][stage] = sub_dim;
                result->per_head_labels_m[head][stage].resize(n_tokens);
                result->per_head_centroids_m[head][stage].resize(static_cast<size_t>(k_values[stage]) * sub_dim);
            }
        }

        std::vector<int32_t> labels_flat(static_cast<size_t>(total_tasks) * n_tokens, 0);
        std::vector<float> centroids_flat(static_cast<size_t>(total_tasks) * max_k * max_sub_dim, 0.0f);
        const int32_t available_threads = std::max<int32_t>(1, worker_threads);
        const int32_t init_threads = std::max<int32_t>(1, std::min(available_threads, total_tasks));
        #pragma omp parallel for num_threads(init_threads) schedule(dynamic, 1)
        for (int32_t task = 0; task < total_tasks; ++task) {
            const int32_t head = task / kStages;
            const int32_t stage = task % kStages;
            const int32_t sub_dim = dim_counts[task];
            const int32_t k = k_values[stage];
            const int32_t* dim_ids = dim_ids_flat.data() + static_cast<size_t>(task) * max_sub_dim;
            float* centroids = centroids_flat.data() + static_cast<size_t>(task) * max_k * max_sub_dim;

            for (int32_t c = 0; c < k; ++c) {
                const int32_t selected_idx = sample_token_index_cpu(
                    n_tokens, head, stage, c, 0, 42424242ULL
                );
                gather_dim_ids_sample(
                    keys_u16, dtype, kv_heads, n_tokens, dim,
                    head, selected_idx, dim_ids, sub_dim, sequence_major,
                    centroids + static_cast<size_t>(c) * max_sub_dim
                );
            }
        }

        const size_t centroid_stride = static_cast<size_t>(max_k) * max_sub_dim;
        const size_t count_stride = static_cast<size_t>(max_k);
        const size_t total_sum_elems = static_cast<size_t>(total_tasks) * centroid_stride;
        const size_t total_count_elems = static_cast<size_t>(total_tasks) * count_stride;
        const int32_t thread_count = available_threads;
        std::vector<float> thread_sums(static_cast<size_t>(thread_count) * total_sum_elems, 0.0f);
        std::vector<int32_t> thread_counts(static_cast<size_t>(thread_count) * total_count_elems, 0);
        std::vector<float> global_sums(total_sum_elems, 0.0f);
        std::vector<int32_t> global_counts(total_count_elems, 0);

        const int32_t tile_tokens = 32;
        const int32_t tiles_per_head = (n_tokens + tile_tokens - 1) / tile_tokens;
        const int32_t tile_tasks = kv_heads * tiles_per_head;

        for (int32_t iter = 0; iter < runtime_cfg.max_iters; ++iter) {
            std::fill(thread_sums.begin(), thread_sums.end(), 0.0f);
            std::fill(thread_counts.begin(), thread_counts.end(), 0);

            #pragma omp parallel num_threads(thread_count)
            {
                const int32_t tid = omp_get_thread_num();
                float* local_sums = thread_sums.data() + static_cast<size_t>(tid) * total_sum_elems;
                int32_t* local_counts = thread_counts.data() + static_cast<size_t>(tid) * total_count_elems;
                std::vector<float> tile;

                #pragma omp for schedule(static)
                for (int32_t tile_task = 0; tile_task < tile_tasks; ++tile_task) {
                    const int32_t head = tile_task / tiles_per_head;
                    const int32_t tile_idx = tile_task - head * tiles_per_head;
                    const int32_t token_begin = tile_idx * tile_tokens;
                    const int32_t token_end = std::min(n_tokens, token_begin + tile_tokens);
                    load_head_token_tile_fp32(
                        keys_u16, dtype, kv_heads, n_tokens, dim,
                        head, token_begin, token_end, sequence_major, tile
                    );

                    for (int32_t local = 0; local < token_end - token_begin; ++local) {
                        const int32_t token = token_begin + local;
                        const float* token_fp32 = tile.data() + static_cast<size_t>(local) * dim;
                        for (int32_t stage = 0; stage < kStages; ++stage) {
                            const int32_t task = head * kStages + stage;
                            const int32_t sub_dim = dim_counts[task];
                            const int32_t k = k_values[stage];
                            const int32_t* dim_ids = dim_ids_flat.data() + static_cast<size_t>(task) * max_sub_dim;
                            const float* centroids = centroids_flat.data() + static_cast<size_t>(task) * centroid_stride;
                            float best_dist = std::numeric_limits<float>::max();
                            int32_t best_cluster = 0;
                            for (int32_t c = 0; c < k; ++c) {
                                float dist = distance_dim_ids(
                                    token_fp32,
                                    dim_ids,
                                    sub_dim,
                                    centroids + static_cast<size_t>(c) * max_sub_dim
                                );
                                if (dist < best_dist) {
                                    best_dist = dist;
                                    best_cluster = c;
                                }
                            }

                            labels_flat[static_cast<size_t>(task) * n_tokens + token] = best_cluster;
                            local_counts[static_cast<size_t>(task) * count_stride + best_cluster]++;
                            float* sum = local_sums
                                + static_cast<size_t>(task) * centroid_stride
                                + static_cast<size_t>(best_cluster) * max_sub_dim;
                            add_dim_ids_to_sum(token_fp32, dim_ids, sub_dim, sum);
                        }
                    }
                }
            }

            #pragma omp parallel for num_threads(thread_count) schedule(static)
            for (size_t i = 0; i < total_sum_elems; ++i) {
                float value = 0.0f;
                for (int32_t tid = 0; tid < thread_count; ++tid) {
                    value += thread_sums[static_cast<size_t>(tid) * total_sum_elems + i];
                }
                global_sums[i] = value;
            }
            #pragma omp parallel for num_threads(thread_count) schedule(static)
            for (size_t i = 0; i < total_count_elems; ++i) {
                int32_t value = 0;
                for (int32_t tid = 0; tid < thread_count; ++tid) {
                    value += thread_counts[static_cast<size_t>(tid) * total_count_elems + i];
                }
                global_counts[i] = value;
            }

            for (int32_t task = 0; task < total_tasks; ++task) {
                const int32_t head = task / kStages;
                const int32_t stage = task % kStages;
                const int32_t sub_dim = dim_counts[task];
                const int32_t k = k_values[stage];
                const int32_t* dim_ids = dim_ids_flat.data() + static_cast<size_t>(task) * max_sub_dim;
                float* centroids = centroids_flat.data() + static_cast<size_t>(task) * centroid_stride;
                for (int32_t c = 0; c < k; ++c) {
                    const int32_t count = global_counts[static_cast<size_t>(task) * count_stride + c];
                    float* centroid = centroids + static_cast<size_t>(c) * max_sub_dim;
                    if (count > 0) {
                        const float inv = 1.0f / static_cast<float>(count);
                        const float* sum = global_sums.data()
                            + static_cast<size_t>(task) * centroid_stride
                            + static_cast<size_t>(c) * max_sub_dim;
                        for (int32_t d = 0; d < sub_dim; ++d) {
                            centroid[d] = sum[d] * inv;
                        }
                    } else {
                        const int32_t token = sample_token_index_cpu(
                            n_tokens, head, stage, c, iter, 12345ULL
                        );
                        gather_dim_ids_sample(
                            keys_u16, dtype, kv_heads, n_tokens, dim,
                            head, token, dim_ids, sub_dim,
                            sequence_major, centroid
                        );
                    }
                }
            }
        }

        for (int32_t head = 0; head < kv_heads; ++head) {
            std::vector<uint64_t> codes(n_tokens);
            for (int32_t token = 0; token < n_tokens; ++token) {
                uint64_t code = 0;
                uint64_t stride = 1;
                for (int32_t stage = 0; stage < kStages; ++stage) {
                    const int32_t task = head * kStages + stage;
                    const uint64_t k = static_cast<uint64_t>(k_values[stage]);
                    const uint64_t label = static_cast<uint64_t>(
                        labels_flat[static_cast<size_t>(task) * n_tokens + token]
                    );
                    code += label * stride;
                    stride *= k;
                }
                codes[token] = code;
            }
            std::sort(codes.begin(), codes.end());
            result->per_head_n_clusters[head] = static_cast<int32_t>(
                std::unique(codes.begin(), codes.end()) - codes.begin()
            );
        }

        for (int32_t head = 0; head < kv_heads; ++head) {
            for (int32_t stage = 0; stage < kStages; ++stage) {
                const int32_t task = head * kStages + stage;
                const int32_t sub_dim = dim_counts[task];
                const int32_t k = k_values[stage];
                std::memcpy(
                    result->per_head_labels_m[head][stage].data(),
                    labels_flat.data() + static_cast<size_t>(task) * n_tokens,
                    static_cast<size_t>(n_tokens) * sizeof(int32_t)
                );
                const float* src_centroids = centroids_flat.data() + static_cast<size_t>(task) * centroid_stride;
                float* dst_centroids = result->per_head_centroids_m[head][stage].data();
                for (int32_t c = 0; c < k; ++c) {
                    std::memcpy(
                        dst_centroids + static_cast<size_t>(c) * sub_dim,
                        src_centroids + static_cast<size_t>(c) * max_sub_dim,
                        static_cast<size_t>(sub_dim) * sizeof(float)
                    );
                }
            }
        }

        return result;
    }
};

std::unique_ptr<KmeansResult> kmeans_execute_m(
    const void* keys_data,       // 16-bit data (FP16/BF16)
    int32_t kv_heads,
    int32_t n_tokens,
    int32_t dim,
    const std::vector<int32_t>& k_values,
    const std::vector<std::vector<std::vector<int32_t>>>& head_pair_groups,
    torch::Dtype kv_dtype,
    int32_t worker_threads,
    const LayerPipelineRuntimeConfig& runtime_config,
    bool sequence_major
) {
    size_t element_size = c10::elementSize(kv_dtype);
    if (element_size != 2) {
        throw std::invalid_argument("kmeans_execute_m only supports 16-bit inputs (fp16/bf16)");
    }

    if (kv_dtype != torch::kFloat16 && kv_dtype != torch::kBFloat16) {
        throw std::invalid_argument("kmeans_execute_m expects torch.float16 or torch.bfloat16 inputs");
    }

    const uint16_t* keys_u16 = static_cast<const uint16_t*>(keys_data);
    KmeansConcurrencyGuard guard(g_active_kmeans);
    int min_threads_per_phase = runtime_config.enable_omp_nested
        ? runtime_config.min_threads_per_phase
        : 1;
    int effective_threads = worker_threads;
    int effective_min_threads = min_threads_per_phase;
    compute_kmeans_thread_budget(
        worker_threads,
        guard.active(),
        min_threads_per_phase,
        effective_threads,
        effective_min_threads
    );
    auto runtime_cfg = load_lloyd_runtime_config(runtime_config);
    if (static_cast<int32_t>(k_values.size()) == 8) {
        return KmeansExecutor8WayFused::execute(
            keys_u16,
            kv_dtype,
            kv_heads,
            n_tokens,
            dim,
            k_values,
            head_pair_groups,
            effective_threads,
            runtime_cfg,
            sequence_major
        );
    }

    throw std::invalid_argument("learned CPU K-means is 8-way only in hybrid_8way_v2");
}

} // namespace imi
