#include "layer_pipeline.hpp"
#include <omp.h>
#include <immintrin.h>  // AVX2
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <iostream>
#include <iomanip>
#include <atomic>  
#include <c10/core/ScalarType.h>

namespace imi {

// ════════════════════════════════════════════════════════════════════════
// AVX2 Distance Kernels (从kmeans_core.cpp剥离)
// ════════════════════════════════════════════════════════════════════════

static inline float euclidean_distance_avx2(const float* a, const float* b, int32_t dim) {
    __m256 sum = _mm256_setzero_ps();
    int32_t i = 0;

    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    float result[8];
    _mm256_storeu_ps(result, sum);
    float total = result[0] + result[1] + result[2] + result[3] +
                  result[4] + result[5] + result[6] + result[7];

    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }

    return total;
}


struct LloydRuntimeConfig {
    int32_t max_iters = 25;
    float tol = 5e-4f;
    int32_t early_stop_min_iter = 5;
    float early_stop_rel_tol = 5e-4f;
    int32_t fp32_convert_chunk_elems = 1 << 20;
};

static LloydRuntimeConfig load_lloyd_runtime_config(const LayerPipelineRuntimeConfig& runtime_config) {
    LloydRuntimeConfig cfg;
    cfg.max_iters = std::max<int32_t>(1, runtime_config.kmeans_max_iters);
    cfg.tol = std::max(runtime_config.kmeans_tol, 0.0f);
    cfg.early_stop_min_iter = std::max<int32_t>(1, runtime_config.kmeans_early_stop_min_iter);
    cfg.early_stop_rel_tol = std::max(runtime_config.kmeans_early_stop_rel_tol, 0.0f);
    cfg.fp32_convert_chunk_elems = std::max<int32_t>(1, runtime_config.fp32_convert_chunk_elems);
    return cfg;
}

namespace {

std::atomic<int> g_active_kmeans{0};
std::atomic<int> g_kmeans_budget_logs{0};
std::atomic<int> g_kmeans_active_logs{0};

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

void log_kmeans_active(
    int active_kmeans,
    int worker_threads,
    int effective_threads,
    int effective_min_threads,
    int32_t kv_heads,
    int32_t n_tokens
) {
    if (std::getenv("IMI_DEBUG_KMEANS_ACTIVE") == nullptr) {
        return;
    }
    int log_id = g_kmeans_active_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_id >= 128) {
        return;
    }
    std::cout << "[KMeansActive] active=" << active_kmeans
              << ", worker_threads=" << worker_threads
              << ", effective_threads=" << effective_threads
              << ", min_threads_per_phase=" << effective_min_threads
              << ", kv_heads=" << kv_heads
              << ", n_tokens=" << n_tokens
              << "\n";
}



} // namespace

// ════════════════════════════════════════════════════════════════════════
// K-means++ Initialization (in-place, reuse buffers)
// ════════════════════════════════════════════════════════════════════════

static void kmeans_plusplus_init_inplace(
    const float* data,
    int32_t n_samples,
    int32_t dim,
    int32_t k,
    uint64_t seed,
    float* centroids,
    std::vector<float>& min_dists
) {
    if (min_dists.size() < static_cast<size_t>(n_samples)) {
        min_dists.resize(static_cast<size_t>(n_samples));
    }
    std::fill(min_dists.begin(), min_dists.begin() + n_samples, std::numeric_limits<float>::max());

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int32_t> uniform_dist(0, n_samples - 1);

    int32_t first_idx = uniform_dist(rng);
    std::memcpy(&centroids[0], &data[first_idx * dim], dim * sizeof(float));

    for (int32_t c = 1; c < k; c++) {
        for (int32_t i = 0; i < n_samples; i++) {
            float dist = euclidean_distance_avx2(&data[i * dim], &centroids[(c - 1) * dim], dim);
            if (dist < min_dists[i]) {
                min_dists[i] = dist;
            }
        }

        double total_weight = 0.0;
        for (int32_t i = 0; i < n_samples; i++) {
            total_weight += min_dists[i];
        }

        std::uniform_real_distribution<double> real_dist(0.0, total_weight);
        double target = real_dist(rng);

        double cumsum = 0.0;
        int32_t selected_idx = 0;
        for (int32_t i = 0; i < n_samples; i++) {
            cumsum += min_dists[i];
            if (cumsum >= target) {
                selected_idx = i;
                break;
            }
        }

        std::memcpy(&centroids[c * dim], &data[selected_idx * dim], dim * sizeof(float));
    }
}



static void lloyd_iteration(
    const float* data,
    int32_t n_samples,
    int32_t dim,
    int32_t k,
    float* centroids,
    int32_t* labels,
    float& inertia,
    int32_t max_iters,
    float tol,
    int32_t n_threads,
    int32_t early_stop_min_iter,
    float early_stop_rel_tol
) {
    std::vector<float> new_centroids(k * dim, 0.0f);
    std::vector<int32_t> cluster_sizes(k, 0);
    int32_t thread_count = std::max(1, n_threads);

    std::vector<std::vector<float>> thread_centroids(
        static_cast<size_t>(thread_count),
        std::vector<float>(k * dim, 0.0f)
    );
    std::vector<std::vector<int32_t>> thread_sizes(
        static_cast<size_t>(thread_count),
        std::vector<int32_t>(k, 0)
    );
    float prev_inertia = std::numeric_limits<float>::max();

    int32_t actual_iters = 0;
    for (int32_t iter = 0; iter < max_iters; iter++) {
        actual_iters++;

        // E-step: 分配标签
        inertia = 0.0f;

        #pragma omp parallel for num_threads(thread_count) reduction(+:inertia) schedule(static)
        for (int32_t i = 0; i < n_samples; i++) {
            float min_dist = std::numeric_limits<float>::max();
            int32_t best_cluster = 0;

            const float* sample = &data[i * dim];

            for (int32_t c = 0; c < k; c++) {
                float dist = euclidean_distance_avx2(sample, &centroids[c * dim], dim);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }

            labels[i] = best_cluster;
            inertia += min_dist;
        }

        // early stop
        float inertia_change = std::abs(prev_inertia - inertia);
        float relative_change = (prev_inertia > 0) ? (inertia_change / prev_inertia) : 0.0f;

        const bool reached_min_iter = actual_iters >= std::max<int32_t>(1, early_stop_min_iter);
        const bool reach_rel_tol = relative_change <= std::max(0.0f, early_stop_rel_tol);
        const bool reach_abs_tol = inertia_change <= std::max(0.0f, tol);
        if (reached_min_iter && (reach_rel_tol || reach_abs_tol)) {
            static std::atomic<int> convergence_count{0};
            if (convergence_count < 3) {
                std::cout << "[Lloyd] Converged early: iter=" << actual_iters
                          << ", n_samples=" << n_samples
                          << ", k=" << k
                          << ", relative_change=" << relative_change * 100 << "%\n";
                convergence_count.fetch_add(1);
            }
            break;
        }
        prev_inertia = inertia;

        
        // M-step: Parallelized centroid update
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);
        #pragma omp parallel num_threads(thread_count)
        {
            int32_t tid = omp_get_thread_num();
            auto& local_centroids = thread_centroids[tid];
            auto& local_sizes = thread_sizes[tid];
            std::fill(local_centroids.begin(), local_centroids.end(), 0.0f);
            std::fill(local_sizes.begin(), local_sizes.end(), 0);

            #pragma omp for schedule(static)
            for (int32_t i = 0; i < n_samples; i++) {
                int32_t c = labels[i];
                local_sizes[c]++;
                const float* sample = &data[i * dim];
                for (int32_t d = 0; d < dim; d++) {
                    local_centroids[c * dim + d] += sample[d];
                }
            }
        }

        for (int32_t t = 0; t < thread_count; t++) {
            const auto& local_sizes = thread_sizes[t];
            const auto& local_centroids = thread_centroids[t];
            for (int32_t c = 0; c < k; c++) {
                cluster_sizes[c] += local_sizes[c];
                for (int32_t d = 0; d < dim; d++) {
                    new_centroids[c * dim + d] += local_centroids[c * dim + d];
                }
            }
        }

        // Normalization + empty cluster processing
        std::mt19937 rng(iter + 12345);  
        std::uniform_int_distribution<int32_t> sample_dist(0, n_samples - 1);

        for (int32_t c = 0; c < k; c++) {
            if (cluster_sizes[c] > 0) {
                float inv_size = 1.0f / cluster_sizes[c];
                for (int32_t d = 0; d < dim; d++) {
                    centroids[c * dim + d] = new_centroids[c * dim + d] * inv_size;
                }
            } else {
                int32_t random_sample = sample_dist(rng);
                std::memcpy(&centroids[c * dim], &data[random_sample * dim], dim * sizeof(float));
            }
        }
    }
}

static void extract_subvectors(
    const float* full_data,  // [n_tokens, dim]
    float* subvec_front,     // [n_tokens, dim_half]
    float* subvec_back,      // [n_tokens, dim_half]
    int32_t n_tokens,
    int32_t dim
) {
    int32_t dim_half = dim / 2;
    for (int32_t i = 0; i < n_tokens; i++) {
        const float* src = &full_data[i * dim];
        std::memcpy(&subvec_front[i * dim_half], src, dim_half * sizeof(float));
        std::memcpy(&subvec_back[i * dim_half], src + dim_half, dim_half * sizeof(float));
    }
}

static void extract_subvectors_4(
    const float* full_data,  // [n_tokens, dim]
    float* subvec_0,         // [n_tokens, dim_quarter]
    float* subvec_1,         // [n_tokens, dim_quarter]
    float* subvec_2,         // [n_tokens, dim_quarter]
    float* subvec_3,         // [n_tokens, dim_quarter]
    int32_t n_tokens,
    int32_t dim
) {
    int32_t dim_quarter = dim / 4;
    for (int32_t i = 0; i < n_tokens; i++) {
        const float* src = &full_data[i * dim];
        std::memcpy(&subvec_0[i * dim_quarter], src, dim_quarter * sizeof(float));
        std::memcpy(&subvec_1[i * dim_quarter], src + dim_quarter, dim_quarter * sizeof(float));
        std::memcpy(&subvec_2[i * dim_quarter], src + 2 * dim_quarter, dim_quarter * sizeof(float));
        std::memcpy(&subvec_3[i * dim_quarter], src + 3 * dim_quarter, dim_quarter * sizeof(float));
    }
}

static void run_imi_single_phase_from_subvec(
    const float* subvec,     // [n_tokens * dim_half] FP32
    int32_t n_tokens,
    int32_t dim_half,
    int32_t k,
    uint64_t seed,
    int32_t* out_labels,
    float* out_centroids,
    int32_t lloyd_threads,
    const LloydRuntimeConfig& runtime_cfg,
    std::vector<float>& min_dists
) {

    kmeans_plusplus_init_inplace(
        subvec, n_tokens, dim_half, k, seed, out_centroids, min_dists
    );

    float inertia = 0.0f;
    lloyd_iteration(
        subvec, n_tokens, dim_half, k,
        out_centroids, out_labels,
        inertia,
        runtime_cfg.max_iters,
        runtime_cfg.tol,
        lloyd_threads,
        runtime_cfg.early_stop_min_iter,
        runtime_cfg.early_stop_rel_tol
    );
}

// ════════════════════════════════════════════════════════════════════════
// K-means Executor: Parallel execution of K-means with 8 heads
// ════════════════════════════════════════════════════════════════════════

class KmeansExecutor {
public:
    static std::unique_ptr<KmeansResult> execute(
        const float* keys_fp32,     // [kv_heads * n_tokens * dim]
        int32_t kv_heads,
        int32_t n_tokens,
        int32_t dim,
        int32_t k1,
        int32_t k2,
        int32_t worker_threads = 12,
        int32_t min_threads_per_phase = 1,
        const LloydRuntimeConfig& runtime_cfg = LloydRuntimeConfig()
    ) {
        auto result = std::make_unique<KmeansResult>();
        result->kv_heads = kv_heads;
        result->n_tokens = n_tokens;
        result->dim = dim;
        result->k1 = k1;
        result->k2 = k2;
        result->subspace_parts = 2;

        result->per_head_labels1.resize(kv_heads);
        result->per_head_labels2.resize(kv_heads);
        result->per_head_centroids1.resize(kv_heads);
        result->per_head_centroids2.resize(kv_heads);
        result->per_head_n_clusters.resize(kv_heads);

        int32_t dim_half = dim / 2;

        const size_t subvec_size = static_cast<size_t>(n_tokens) * dim_half;
        std::vector<std::vector<float>> subvec_front_per_head(kv_heads);
        std::vector<std::vector<float>> subvec_back_per_head(kv_heads);
        const int32_t phases = 2;
        std::vector<std::vector<float>> min_dists_per_phase(static_cast<size_t>(kv_heads) * phases);
        for (int32_t head = 0; head < kv_heads; head++) {
            subvec_front_per_head[head].resize(subvec_size);
            subvec_back_per_head[head].resize(subvec_size);
            for (int32_t phase = 0; phase < phases; ++phase) {
                min_dists_per_phase[static_cast<size_t>(head) * phases + phase].resize(n_tokens);
            }
        }

        for (int32_t head = 0; head < kv_heads; head++) {
            result->per_head_labels1[head].resize(n_tokens);
            result->per_head_labels2[head].resize(n_tokens);
            result->per_head_centroids1[head].resize(k1 * dim_half);
            result->per_head_centroids2[head].resize(k2 * dim_half);
        }

        int32_t total_tasks = kv_heads * phases;  // head × phase

        int32_t available_threads = std::max<int32_t>(1, worker_threads);
        int32_t min_threads = std::max<int32_t>(1, std::min(min_threads_per_phase, available_threads));
        int32_t max_concurrent_phases = std::max<int32_t>(
            1,
            std::min(total_tasks, available_threads / min_threads)
        );
        int32_t threads_per_phase = std::max<int32_t>(1, available_threads / max_concurrent_phases);
        threads_per_phase = std::max<int32_t>(threads_per_phase, min_threads);
        int32_t extract_threads = std::max<int32_t>(1, std::min(available_threads, kv_heads));
        #pragma omp parallel for num_threads(extract_threads) schedule(static)
        for (int32_t head = 0; head < kv_heads; ++head) {
            const float* head_data = &keys_fp32[head * n_tokens * dim];
            auto& subvec_front = subvec_front_per_head[head];
            auto& subvec_back = subvec_back_per_head[head];
            extract_subvectors(head_data, subvec_front.data(), subvec_back.data(), n_tokens, dim);
        }
        #pragma omp parallel for num_threads(max_concurrent_phases) schedule(dynamic, 1)
        for (int32_t task = 0; task < total_tasks; ++task) {
            int32_t head = task / phases;
            int32_t phase = task % phases;
            auto& min_dists = min_dists_per_phase[static_cast<size_t>(task)];
            if (phase == 0) {
                run_imi_single_phase_from_subvec(
                    subvec_front_per_head[head].data(), n_tokens, dim_half, k1,
                    /*seed=*/42,
                    result->per_head_labels1[head].data(),
                    result->per_head_centroids1[head].data(),
                    threads_per_phase,
                    runtime_cfg,
                    min_dists
                );
            } else {
                run_imi_single_phase_from_subvec(
                    subvec_back_per_head[head].data(), n_tokens, dim_half, k2,
                    /*seed=*/43,
                    result->per_head_labels2[head].data(),
                    result->per_head_centroids2[head].data(),
                    threads_per_phase,
                    runtime_cfg,
                    min_dists
                );
            }
        }

        // Calculate the actual number of activated clusters for each head (after Cartesian product).
        const int32_t total_clusters = k1 * k2;
        std::vector<uint8_t> active_flags(total_clusters, 0);
        for (int32_t head = 0; head < kv_heads; head++) {
            std::memset(active_flags.data(), 0, active_flags.size());
            int32_t active_count = 0;
            for (int32_t i = 0; i < n_tokens; i++) {
                int32_t cluster_id = result->per_head_labels1[head][i] * k2
                                   + result->per_head_labels2[head][i];
                if (cluster_id < 0 || cluster_id >= total_clusters) {
                    continue;
                }
                if (!active_flags[cluster_id]) {
                    active_flags[cluster_id] = 1;
                    ++active_count;
                }
            }
            result->per_head_n_clusters[head] = active_count;
        }

        return result;
    }
};

class KmeansExecutorIVF {
public:
    static std::unique_ptr<KmeansResult> execute(
        const float* keys_fp32,     // [kv_heads * n_tokens * dim]
        int32_t kv_heads,
        int32_t n_tokens,
        int32_t dim,
        int32_t k,
        int32_t worker_threads = 12,
        int32_t min_threads_per_phase = 1,
        const LloydRuntimeConfig& runtime_cfg = LloydRuntimeConfig()
    ) {
        auto result = std::make_unique<KmeansResult>();
        result->kv_heads = kv_heads;
        result->n_tokens = n_tokens;
        result->dim = dim;
        result->subspace_parts = 0;

        // IVF uses a single stage. Store into stage1 for debugging consistency.
        result->per_head_labels1.resize(kv_heads);
        result->per_head_centroids1.resize(kv_heads);
        result->per_head_n_clusters.resize(kv_heads);

        std::vector<std::vector<float>> min_dists_per_head(kv_heads);
        for (int32_t head = 0; head < kv_heads; ++head) {
            result->per_head_labels1[head].resize(n_tokens);
            result->per_head_centroids1[head].resize(static_cast<size_t>(k) * dim);
            min_dists_per_head[head].resize(n_tokens);
        }

        int32_t total_tasks = kv_heads;
        int32_t available_threads = std::max<int32_t>(1, worker_threads);
        int32_t min_threads = std::max<int32_t>(1, std::min(min_threads_per_phase, available_threads));
        int32_t max_concurrent_phases = std::max<int32_t>(
            1,
            std::min(total_tasks, available_threads / min_threads)
        );
        int32_t threads_per_phase = std::max<int32_t>(1, available_threads / max_concurrent_phases);
        threads_per_phase = std::max<int32_t>(threads_per_phase, min_threads);

        #pragma omp parallel for num_threads(max_concurrent_phases) schedule(dynamic, 1)
        for (int32_t head = 0; head < total_tasks; ++head) {
            const float* head_data = &keys_fp32[head * n_tokens * dim];
            auto& min_dists = min_dists_per_head[head];

            kmeans_plusplus_init_inplace(
                head_data,
                n_tokens,
                dim,
                k,
                /*seed=*/42,
                result->per_head_centroids1[head].data(),
                min_dists
            );

            float inertia = 0.0f;
            lloyd_iteration(
                head_data,
                n_tokens,
                dim,
                k,
                result->per_head_centroids1[head].data(),
                result->per_head_labels1[head].data(),
                inertia,
                runtime_cfg.max_iters,
                runtime_cfg.tol,
                threads_per_phase,
                runtime_cfg.early_stop_min_iter,
                runtime_cfg.early_stop_rel_tol
            );
        }

        // Active cluster count (uncompacted; reorganize step will compact centroids)
        std::vector<uint8_t> active_flags(static_cast<size_t>(k), 0);
        for (int32_t head = 0; head < kv_heads; ++head) {
            std::memset(active_flags.data(), 0, active_flags.size());
            int32_t active_count = 0;
            for (int32_t i = 0; i < n_tokens; ++i) {
                int32_t cid = result->per_head_labels1[head][i];
                if (cid < 0 || cid >= k) {
                    continue;
                }
                if (!active_flags[cid]) {
                    active_flags[cid] = 1;
                    ++active_count;
                }
            }
            result->per_head_n_clusters[head] = active_count;
        }

        return result;
    }
};


class KmeansExecutor4 {
public:
    static std::unique_ptr<KmeansResult> execute(
        const float* keys_fp32,     // [kv_heads * n_tokens * dim]
        int32_t kv_heads,
        int32_t n_tokens,
        int32_t dim,
        int32_t k1,
        int32_t k2,
        int32_t k3,
        int32_t k4,
        int32_t worker_threads = 12,
        int32_t min_threads_per_phase = 1,
        const LloydRuntimeConfig& runtime_cfg = LloydRuntimeConfig()
    ) {

        if (dim % 4 != 0) {
            throw std::invalid_argument("KmeansExecutor4 requires dim divisible by 4");
        }

        auto result = std::make_unique<KmeansResult>();
        result->kv_heads = kv_heads;
        result->n_tokens = n_tokens;
        result->dim = dim;
        result->k1 = k1;
        result->k2 = k2;
        result->subspace_parts = 4;

        result->per_head_labels1.resize(kv_heads);
        result->per_head_labels2.resize(kv_heads);
        result->per_head_labels3.resize(kv_heads);
        result->per_head_labels4.resize(kv_heads);
        result->per_head_centroids1.resize(kv_heads);
        result->per_head_centroids2.resize(kv_heads);
        result->per_head_centroids3.resize(kv_heads);
        result->per_head_centroids4.resize(kv_heads);
        result->per_head_n_clusters.resize(kv_heads);

        int32_t dim_quarter = dim / 4;

        const size_t subvec_size = static_cast<size_t>(n_tokens) * dim_quarter;
        std::vector<std::vector<float>> subvec_0_per_head(kv_heads);
        std::vector<std::vector<float>> subvec_1_per_head(kv_heads);
        std::vector<std::vector<float>> subvec_2_per_head(kv_heads);
        std::vector<std::vector<float>> subvec_3_per_head(kv_heads);
        const int32_t phases = 4;
        std::vector<std::vector<float>> min_dists_per_phase(static_cast<size_t>(kv_heads) * phases);
        for (int32_t head = 0; head < kv_heads; head++) {
            subvec_0_per_head[head].resize(subvec_size);
            subvec_1_per_head[head].resize(subvec_size);
            subvec_2_per_head[head].resize(subvec_size);
            subvec_3_per_head[head].resize(subvec_size);
            for (int32_t phase = 0; phase < phases; ++phase) {
                min_dists_per_phase[static_cast<size_t>(head) * phases + phase].resize(n_tokens);
            }
        }

        for (int32_t head = 0; head < kv_heads; head++) {
            result->per_head_labels1[head].resize(n_tokens);
            result->per_head_labels2[head].resize(n_tokens);
            result->per_head_labels3[head].resize(n_tokens);
            result->per_head_labels4[head].resize(n_tokens);
            result->per_head_centroids1[head].resize(k1 * dim_quarter);
            result->per_head_centroids2[head].resize(k2 * dim_quarter);
            result->per_head_centroids3[head].resize(k3 * dim_quarter);
            result->per_head_centroids4[head].resize(k4 * dim_quarter);
        }

        int32_t total_tasks = kv_heads * phases;
        int32_t available_threads = std::max<int32_t>(1, worker_threads);
        int32_t min_threads = std::max<int32_t>(1, std::min(min_threads_per_phase, available_threads));
        int32_t max_concurrent_phases = std::max<int32_t>(
            1,
            std::min(total_tasks, available_threads / min_threads)
        );
        int32_t threads_per_phase = std::max<int32_t>(1, available_threads / max_concurrent_phases);
        threads_per_phase = std::max<int32_t>(threads_per_phase, min_threads);

        int32_t extract_threads = std::max<int32_t>(1, std::min(available_threads, kv_heads));
        #pragma omp parallel for num_threads(extract_threads) schedule(static)
        for (int32_t head = 0; head < kv_heads; head++) {
            const float* head_data = &keys_fp32[head * n_tokens * dim];
            auto& subvec_0 = subvec_0_per_head[head];
            auto& subvec_1 = subvec_1_per_head[head];
            auto& subvec_2 = subvec_2_per_head[head];
            auto& subvec_3 = subvec_3_per_head[head];
            extract_subvectors_4(
                head_data,
                subvec_0.data(),
                subvec_1.data(),
                subvec_2.data(),
                subvec_3.data(),
                n_tokens,
                dim
            );
        }

        #pragma omp parallel for num_threads(max_concurrent_phases) schedule(dynamic, 1)
        for (int32_t task = 0; task < total_tasks; ++task) {
            int32_t head = task / phases;
            int32_t phase = task % phases;
            auto& min_dists = min_dists_per_phase[static_cast<size_t>(task)];
            if (phase == 0) {
                run_imi_single_phase_from_subvec(
                    subvec_0_per_head[head].data(), n_tokens, dim_quarter, k1,
                    /*seed=*/42,
                    result->per_head_labels1[head].data(),
                    result->per_head_centroids1[head].data(),
                    threads_per_phase,
                    runtime_cfg,
                    min_dists
                );
            } else if (phase == 1) {
                run_imi_single_phase_from_subvec(
                    subvec_1_per_head[head].data(), n_tokens, dim_quarter, k2,
                    /*seed=*/43,
                    result->per_head_labels2[head].data(),
                    result->per_head_centroids2[head].data(),
                    threads_per_phase,
                    runtime_cfg,
                    min_dists
                );
            } else if (phase == 2) {
                run_imi_single_phase_from_subvec(
                    subvec_2_per_head[head].data(), n_tokens, dim_quarter, k3,
                    /*seed=*/44,
                    result->per_head_labels3[head].data(),
                    result->per_head_centroids3[head].data(),
                    threads_per_phase,
                    runtime_cfg,
                    min_dists
                );
            } else {
                run_imi_single_phase_from_subvec(
                    subvec_3_per_head[head].data(), n_tokens, dim_quarter, k4,
                    /*seed=*/45,
                    result->per_head_labels4[head].data(),
                    result->per_head_centroids4[head].data(),
                    threads_per_phase,
                    runtime_cfg,
                    min_dists
                );
            }
        }

        const int64_t total_clusters =
            static_cast<int64_t>(k1) * k2 * k3 * k4;
        if (total_clusters > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("IMI 4-way clusters exceed int32 limit");
        }

        std::vector<uint8_t> active_flags(static_cast<size_t>(total_clusters), 0);
        for (int32_t head = 0; head < kv_heads; head++) {
            std::memset(active_flags.data(), 0, active_flags.size());
            int32_t active_count = 0;
            for (int32_t i = 0; i < n_tokens; i++) {
                int64_t cluster_id =
                    (((static_cast<int64_t>(result->per_head_labels1[head][i]) * k2
                     + result->per_head_labels2[head][i]) * k3
                     + result->per_head_labels3[head][i]) * k4
                     + result->per_head_labels4[head][i]);
                if (cluster_id < 0 || cluster_id >= total_clusters) {
                    continue;
                }
                if (!active_flags[cluster_id]) {
                    active_flags[cluster_id] = 1;
                    ++active_count;
                }
            }
            result->per_head_n_clusters[head] = active_count;
        }

        return result;
    }
};

} // namespace imi

// ════════════════════════════════════════════════════════════════════════
// 16-bit→FP32 Conversion Helpers (Thread-Safe, No PyTorch API)
// ════════════════════════════════════════════════════════════════════════

static inline float fp16_to_fp32(uint16_t h) {
    union { uint32_t i; float f; } u;

    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mantissa = (h & 0x03FF) << 13;

    if (exp == 0x1F) {  // Inf or NaN
        u.i = sign | 0x7F800000 | mantissa;
    } else if (exp == 0) {  // Denorm or zero
        if (mantissa == 0) {
            u.i = sign;  // Zero
        } else {
            // Denormalized number
            exp = 0x71;
            while ((mantissa & 0x00400000) == 0) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x007FFFFF;
            u.i = sign | (exp << 23) | mantissa;
        }
    } else {
        // Normalized number
        u.i = sign | ((exp + 112) << 23) | mantissa;
    }

    return u.f;
}

//BF16→FP32 
static inline float bf16_to_fp32(uint16_t h) {
    union { uint32_t i; float f; } u;
    u.i = static_cast<uint32_t>(h) << 16;
    return u.f;
}

// ════════════════════════════════════════════════════════════════════════
// Export wrapper for layer_pipeline.cpp
// ════════════════════════════════════════════════════════════════════════

namespace imi {
std::unique_ptr<KmeansResult> kmeans_execute(
    const void* keys_data,       // 16-bit data (FP16/BF16)
    int32_t kv_heads,
    int32_t n_tokens,
    int32_t dim,
    int32_t k1,
    int32_t k2,
    torch::Dtype kv_dtype,       
    int32_t worker_threads,
    const LayerPipelineRuntimeConfig& runtime_config
) {
    size_t total_elements = static_cast<size_t>(kv_heads) * n_tokens * dim;
    size_t element_size = c10::elementSize(kv_dtype);
    if (element_size != 2) {
        throw std::invalid_argument("kmeans_execute only supports 16-bit inputs (fp16/bf16)");
    }

    if (kv_dtype != torch::kFloat16 && kv_dtype != torch::kBFloat16) {
        throw std::invalid_argument("kmeans_execute expects torch.float16 or torch.bfloat16 inputs");
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
    log_kmeans_active(
        guard.active(),
        worker_threads,
        effective_threads,
        effective_min_threads,
        kv_heads,
        n_tokens
    );


    struct AlignedFloatBuffer {
        float* data = nullptr;
        size_t capacity = 0;

        ~AlignedFloatBuffer() {
            if (data) {
                free(data);
            }
        }

        void ensure(size_t count) {
            if (count <= capacity) {
                return;
            }
            if (data) {
                free(data);
                data = nullptr;
            }
            size_t bytes = count * sizeof(float);
            size_t aligned_bytes = ((bytes + 63) / 64) * 64;
            data = static_cast<float*>(aligned_alloc(64, aligned_bytes));
            if (!data) {
                throw std::runtime_error("Failed to allocate FP32 buffer for K-means");
            }
            capacity = aligned_bytes / sizeof(float);
        }
    };

    static thread_local AlignedFloatBuffer keys_fp32_buffer;
    keys_fp32_buffer.ensure(total_elements);
    float* keys_fp32 = keys_fp32_buffer.data;

    size_t chunk_elems = std::max<size_t>(1, static_cast<size_t>(runtime_config.fp32_convert_chunk_elems));
    chunk_elems = std::min(chunk_elems, total_elements);
    const size_t num_chunks = (total_elements + chunk_elems - 1) / chunk_elems;
    if (kv_dtype == torch::kFloat16) {
        #pragma omp parallel for num_threads(effective_threads) schedule(static)
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const size_t start = chunk_idx * chunk_elems;
            const size_t end = std::min(total_elements, start + chunk_elems);
            for (size_t i = start; i < end; ++i) {
                keys_fp32[i] = fp16_to_fp32(keys_u16[i]);
            }
        }
    } else {  // torch::kBFloat16
        #pragma omp parallel for num_threads(effective_threads) schedule(static)
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const size_t start = chunk_idx * chunk_elems;
            const size_t end = std::min(total_elements, start + chunk_elems);
            for (size_t i = start; i < end; ++i) {
                keys_fp32[i] = bf16_to_fp32(keys_u16[i]);
            }
        }
    }

    auto runtime_cfg = load_lloyd_runtime_config(runtime_config);

    // Execute K-means
    auto result = KmeansExecutor::execute(
        keys_fp32,
        kv_heads,
        n_tokens,
        dim,
        k1,
        k2,
        effective_threads,
        effective_min_threads,
        runtime_cfg
    );

    return result;
}

std::unique_ptr<KmeansResult> kmeans_execute_4(
    const void* keys_data,       // 16-bit data (FP16/BF16)
    int32_t kv_heads,
    int32_t n_tokens,
    int32_t dim,
    int32_t k1,
    int32_t k2,
    int32_t k3,
    int32_t k4,
    torch::Dtype kv_dtype,       
    int32_t worker_threads,
    const LayerPipelineRuntimeConfig& runtime_config
) {
    size_t total_elements = static_cast<size_t>(kv_heads) * n_tokens * dim;
    size_t element_size = c10::elementSize(kv_dtype);
    if (element_size != 2) {
        throw std::invalid_argument("kmeans_execute_4 only supports 16-bit inputs (fp16/bf16)");
    }

    if (kv_dtype != torch::kFloat16 && kv_dtype != torch::kBFloat16) {
        throw std::invalid_argument("kmeans_execute_4 expects torch.float16 or torch.bfloat16 inputs");
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
    log_kmeans_active(
        guard.active(),
        worker_threads,
        effective_threads,
        effective_min_threads,
        kv_heads,
        n_tokens
    );


    struct AlignedFloatBuffer {
        float* data = nullptr;
        size_t capacity = 0;

        ~AlignedFloatBuffer() {
            if (data) {
                free(data);
            }
        }

        void ensure(size_t count) {
            if (count <= capacity) {
                return;
            }
            if (data) {
                free(data);
                data = nullptr;
            }
            size_t bytes = count * sizeof(float);
            size_t aligned_bytes = ((bytes + 63) / 64) * 64;
            data = static_cast<float*>(aligned_alloc(64, aligned_bytes));
            if (!data) {
                throw std::runtime_error("Failed to allocate FP32 buffer for K-means");
            }
            capacity = aligned_bytes / sizeof(float);
        }
    };

    static thread_local AlignedFloatBuffer keys_fp32_buffer;
    keys_fp32_buffer.ensure(total_elements);
    float* keys_fp32 = keys_fp32_buffer.data;

    size_t chunk_elems = std::max<size_t>(1, static_cast<size_t>(runtime_config.fp32_convert_chunk_elems));
    chunk_elems = std::min(chunk_elems, total_elements);
    const size_t num_chunks = (total_elements + chunk_elems - 1) / chunk_elems;

    if (kv_dtype == torch::kFloat16) {
        #pragma omp parallel for num_threads(effective_threads) schedule(static)
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const size_t start = chunk_idx * chunk_elems;
            const size_t end = std::min(total_elements, start + chunk_elems);
            for (size_t i = start; i < end; ++i) {
                keys_fp32[i] = fp16_to_fp32(keys_u16[i]);
            }
        }
    } else {
        #pragma omp parallel for num_threads(effective_threads) schedule(static)
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const size_t start = chunk_idx * chunk_elems;
            const size_t end = std::min(total_elements, start + chunk_elems);
            for (size_t i = start; i < end; ++i) {
                keys_fp32[i] = bf16_to_fp32(keys_u16[i]);
            }
        }
    }

    auto runtime_cfg = load_lloyd_runtime_config(runtime_config);

    auto result = KmeansExecutor4::execute(
        keys_fp32,
        kv_heads,
        n_tokens,
        dim,
        k1,
        k2,
        k3,
        k4,
        effective_threads,
        effective_min_threads,
        runtime_cfg
    );

    return result;
}

std::unique_ptr<KmeansResult> kmeans_execute_ivf(
    const void* keys_data,       // 16-bit data (FP16/BF16)
    int32_t kv_heads,
    int32_t n_tokens,
    int32_t dim,
    int32_t k,
    torch::Dtype kv_dtype,
    int32_t worker_threads,
    const LayerPipelineRuntimeConfig& runtime_config
) {
    size_t total_elements = static_cast<size_t>(kv_heads) * n_tokens * dim;
    size_t element_size = c10::elementSize(kv_dtype);
    if (element_size != 2) {
        throw std::invalid_argument("kmeans_execute_ivf only supports 16-bit inputs (fp16/bf16)");
    }

    if (kv_dtype != torch::kFloat16 && kv_dtype != torch::kBFloat16) {
        throw std::invalid_argument("kmeans_execute_ivf expects torch.float16 or torch.bfloat16 inputs");
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
    log_kmeans_active(
        guard.active(),
        worker_threads,
        effective_threads,
        effective_min_threads,
        kv_heads,
        n_tokens
    );


    struct AlignedFloatBuffer {
        float* data = nullptr;
        size_t capacity = 0;

        ~AlignedFloatBuffer() {
            if (data) {
                free(data);
            }
        }

        void ensure(size_t count) {
            if (count <= capacity) {
                return;
            }
            if (data) {
                free(data);
                data = nullptr;
            }
            size_t bytes = count * sizeof(float);
            size_t aligned_bytes = ((bytes + 63) / 64) * 64;
            data = static_cast<float*>(aligned_alloc(64, aligned_bytes));
            if (!data) {
                throw std::runtime_error("Failed to allocate FP32 buffer for K-means");
            }
            capacity = aligned_bytes / sizeof(float);
        }
    };

    static thread_local AlignedFloatBuffer keys_fp32_buffer;
    keys_fp32_buffer.ensure(total_elements);
    float* keys_fp32 = keys_fp32_buffer.data;

    size_t chunk_elems = std::max<size_t>(1, static_cast<size_t>(runtime_config.fp32_convert_chunk_elems));
    chunk_elems = std::min(chunk_elems, total_elements);
    const size_t num_chunks = (total_elements + chunk_elems - 1) / chunk_elems;

    if (kv_dtype == torch::kFloat16) {
        #pragma omp parallel for num_threads(effective_threads) schedule(static)
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const size_t start = chunk_idx * chunk_elems;
            const size_t end = std::min(total_elements, start + chunk_elems);
            for (size_t i = start; i < end; ++i) {
                keys_fp32[i] = fp16_to_fp32(keys_u16[i]);
            }
        }
    } else {
        #pragma omp parallel for num_threads(effective_threads) schedule(static)
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const size_t start = chunk_idx * chunk_elems;
            const size_t end = std::min(total_elements, start + chunk_elems);
            for (size_t i = start; i < end; ++i) {
                keys_fp32[i] = bf16_to_fp32(keys_u16[i]);
            }
        }
    }

    auto runtime_cfg = load_lloyd_runtime_config(runtime_config);

    return KmeansExecutorIVF::execute(
        keys_fp32,
        kv_heads,
        n_tokens,
        dim,
        k,
        effective_threads,
        effective_min_threads,
        runtime_cfg
    );
}

} // namespace imi
