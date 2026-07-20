// ============================================================================
// layer_pipeline.cpp - Main Implementation
// ============================================================================
#include "layer_pipeline.hpp"
#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>    // ✅ for std::sqrt
#include <array>
#include <algorithm>
#include <limits>
#include <thread>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <omp.h>
#include <cuda_runtime.h>  // ✅ for cudaStreamSynchronize
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <sys/resource.h>

namespace imi {

// ════════════════════════════════════════════════════════════════════════
// Constructor & Destructor
// ════════════════════════════════════════════════════════════════════════

LayerPipeline::LayerPipeline(
    int layer_idx,
    int kv_heads,
    int max_tokens,
    int dim,
    int device_id,
    torch::Dtype kv_dtype,
    bool enable_direct_write,
    int subspace_parts,
    LayerPipelineRuntimeConfig runtime_config
) : layer_idx_(layer_idx),
    kv_heads_(kv_heads),
    max_tokens_(max_tokens),
    dim_(dim),
    kv_dtype_(kv_dtype),
    device_id_(device_id),
    enable_direct_write_(enable_direct_write),
    subspace_parts_(subspace_parts),
    runtime_config_(runtime_config)
{
    if (kv_dtype_ != torch::kFloat16 && kv_dtype_ != torch::kBFloat16) {
        throw std::invalid_argument(
            "[LayerPipeline] Only torch.float16 and torch.bfloat16 are supported for KV dtype");
    }
    if (!runtime_config_.learned_partition_enabled) {
        throw std::invalid_argument("[LayerPipeline] hybrid_8way_v2 requires learned_partition");
    }
    if (runtime_config_.learned_subspace_parts <= 0) {
        runtime_config_.learned_subspace_parts = subspace_parts_;
    }
    if (runtime_config_.learned_subspace_parts != 8) {
        throw std::invalid_argument("[LayerPipeline] hybrid_8way_v2 CPU path supports learned 8-way only");
    }
    subspace_parts_ = runtime_config_.learned_subspace_parts;
    kv_element_size_bytes_ = c10::elementSize(kv_dtype_);

    // ✅ Set CUDA device
    cudaSetDevice(device_id_);

    // ✅ Create independent CUDA stream（用于GPU→CPU拷贝，生命周期与本类一致）
    cudaStreamCreate(&transfer_stream_);

}

LayerPipeline::~LayerPipeline() noexcept {
    try {
        // ✅ 确保后台线程退出
        cancel_pipeline();

        // ✅ 【P0修复】清理预分配的 CPU buffers（在 try 块内）
        // 原因：keys_buffer_ 和 values_buffer_ 是 torch::Tensor，析构可能抛异常
        try {
            keys_buffer_ = torch::Tensor();  // Release tensor
            values_buffer_ = torch::Tensor();
            keys_sequence_cpu_ = torch::Tensor();
            values_sequence_cpu_ = torch::Tensor();
        } catch (const std::exception& e) {
            std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                      << ": keys/values_buffer_ release threw exception: " << e.what() << "\n";
        } catch (...) {
            std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                      << ": keys/values_buffer_ release threw unknown exception!\n";
        }

        // ✅ Free CUDA resources（可能失败，捕获所有异常）
        if (transfer_stream_) {
            cudaError_t err = cudaStreamDestroy(transfer_stream_);
            if (err != cudaSuccess) {
                std::cerr << "[LayerPipeline] Layer " << layer_idx_
                          << ": cudaStreamDestroy failed: "
                          << cudaGetErrorString(err) << "\n";
            }
            transfer_stream_ = nullptr;
        }

    } catch (const std::exception& e) {
        // ✅ 捕获所有 C++ 异常（防止析构抛出异常）
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                  << ": Exception in destructor: " << e.what() << "\n";
    } catch (...) {
        // ✅ 捕获所有其他异常（包括 CUDA 内部异常）
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                  << ": Unknown exception in destructor!\n";
    }
    {
        std::lock_guard<std::mutex> lock(chunk_pool_mutex_);
        chunk_keys_pool_.clear();
        chunk_values_pool_.clear();
        chunk_buffer_in_use_.clear();
        chunk_pool_size_ = 0;
        chunk_buffer_capacity_ = 0;
    }
}

// ════════════════════════════════════════════════════════════════════════
// Wait API
// ════════════════════════════════════════════════════════════════════════

bool LayerPipeline::wait_ready(float timeout_sec) {
    std::unique_lock<std::mutex> lock(ready_mutex_);

    auto maybe_start_deferred_merge = [this]() {
        if (!runtime_config_.defer_merge_until_wait ||
            !transfers_finished_.load(std::memory_order_acquire) ||
            ready_to_merge_.load(std::memory_order_acquire) ||
            merge_done_.load(std::memory_order_acquire) ||
            has_error_.load(std::memory_order_acquire) ||
            stop_requested_.load(std::memory_order_acquire)) {
            return;
        }
        bool has_payload = false;
        {
            std::lock_guard<std::mutex> payload_lock(stats_mutex_);
            has_payload = pending_payload_.has_value();
        }
        if (!has_payload) {
            return;
        }
        ready_to_merge_.store(true, std::memory_order_release);
        merge_cv_.notify_one();
    };

    maybe_start_deferred_merge();
    bool success = ready_cv_.wait_for(
        lock,
        std::chrono::milliseconds(static_cast<int>(timeout_sec * 1000)),
        [this, &maybe_start_deferred_merge] {
            maybe_start_deferred_merge();
            return ready_flag_;
        }
    );

    return success && stage_.load() == PipelineStage::Complete;
}



PipelineStage LayerPipeline::current_stage() const {
    return stage_.load();
}

std::string LayerPipeline::error_message() const {
    return error_message_;
}

namespace {

double get_process_cpu_time_ms() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }
    const double user_ms = static_cast<double>(usage.ru_utime.tv_sec) * 1000.0
        + static_cast<double>(usage.ru_utime.tv_usec) / 1000.0;
    const double system_ms = static_cast<double>(usage.ru_stime.tv_sec) * 1000.0
        + static_cast<double>(usage.ru_stime.tv_usec) / 1000.0;
    return user_ms + system_ms;
}


int determine_worker_threads(const LayerPipelineRuntimeConfig& runtime_config) {
    int total_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (total_threads <= 0) {
        total_threads = 48;  // 常见服务器默认值
    }

    int desired = runtime_config.worker_threads > 0 ? runtime_config.worker_threads : total_threads;
    desired = std::max(1, std::min(desired, total_threads));
    return desired;
}

std::vector<int32_t> calculate_learned_k_values(int seq_len, int target_cluster_size, int subspace_parts) {
    const int M = std::max(subspace_parts, 1);
    const int token_count = std::max(seq_len, 1);
    if (target_cluster_size <= 0) {
        target_cluster_size = 16;
    }

    const double target = static_cast<double>(token_count) / static_cast<double>(target_cluster_size);
    int base = std::min(3, token_count);
    while (true) {
        double product = std::pow(static_cast<double>(base), M);
        if (product > target) {
            return std::vector<int32_t>(static_cast<size_t>(M), base);
        }

        const int high = std::min(base + 1, token_count);
        if (high == base) {
            return std::vector<int32_t>(static_cast<size_t>(M), base);
        }

        for (int high_count = 1; high_count <= M; ++high_count) {
            product = std::pow(static_cast<double>(high), high_count) *
                std::pow(static_cast<double>(base), M - high_count);
            if (product > target) {
                std::vector<int32_t> values;
                values.reserve(static_cast<size_t>(M));
                for (int idx = 0; idx < M; ++idx) {
                    values.push_back(idx < high_count ? high : base);
                }
                return values;
            }
        }

        base = high;
    }
}

} // namespace

void LayerPipeline::mark_failure(const std::string& message) {
    error_message_ = message;
    has_error_ = true;
    stage_.store(PipelineStage::Failed);
}

PipelineStats LayerPipeline::get_pipeline_stats() const {
    std::lock_guard<std::mutex> lock(pipeline_stats_mutex_);
    return pipeline_stats_;
}

void LayerPipeline::reset_pipeline_stats() {
    std::lock_guard<std::mutex> lock(pipeline_stats_mutex_);
    pipeline_stats_ = PipelineStats{};
}

bool LayerPipeline::allocate_chunk_buffers(int capacity_tokens) {
    int required_tokens = std::max(1, capacity_tokens);
    int row_count = row_count_ > 0 ? row_count_ : kv_heads_;
    try {
        std::lock_guard<std::mutex> lock(chunk_pool_mutex_);
        if (chunk_buffer_capacity_ >= required_tokens && !chunk_keys_pool_.empty()) {
            return true;
        }

        int pool_size = static_cast<int>(chunk_pool_size_);
        if (pool_size <= 0) {
            pool_size = runtime_config_.chunk_buffer_count;
        }
        pool_size = std::max(pool_size, 1);

        chunk_keys_pool_.clear();
        chunk_values_pool_.clear();
        chunk_buffer_in_use_.assign(static_cast<size_t>(pool_size), false);
        chunk_keys_pool_.reserve(static_cast<size_t>(pool_size));
        chunk_values_pool_.reserve(static_cast<size_t>(pool_size));

        for (int i = 0; i < pool_size; ++i) {
            if (runtime_config_.learned_partition_enabled) {
                chunk_keys_pool_.push_back(torch::empty(
                    {batch_size_, required_tokens, kv_heads_, dim_},
                    torch::dtype(kv_dtype_).device(torch::kCPU).pinned_memory(true)
                ));
                chunk_values_pool_.push_back(torch::empty(
                    {batch_size_, required_tokens, kv_heads_, dim_},
                    torch::dtype(kv_dtype_).device(torch::kCPU).pinned_memory(true)
                ));
            } else {
                chunk_keys_pool_.push_back(torch::empty(
                    {row_count, required_tokens, dim_},
                    torch::dtype(kv_dtype_).device(torch::kCPU).pinned_memory(true)
                ));
                chunk_values_pool_.push_back(torch::empty(
                    {row_count, required_tokens, dim_},
                    torch::dtype(kv_dtype_).device(torch::kCPU).pinned_memory(true)
                ));
            }
        }

        chunk_buffer_capacity_ = required_tokens;
        chunk_pool_size_ = static_cast<size_t>(pool_size);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                  << ": Failed to allocate pinned staging buffer (" << e.what() << ")\n";
        return false;
    }
}

bool LayerPipeline::prepare_direct_write_buffers(int32_t total_tokens) {
    if (!enable_direct_write_) {
        return true;
    }
    if (direct_write_buffers_ready_) {
        return true;
    }
    if (total_tokens <= 0) {
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                  << ": Invalid token count for direct write buffers\n";
        return false;
    }
    if (!batch_allocate_cpu_buffer_cb_) {
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                  << ": No CPU buffer allocation callback set\n";
        return false;
    }

    static thread_local std::vector<std::pair<int32_t, int32_t>> heads_info;
    heads_info.clear();
    int row_count = row_count_ > 0 ? row_count_ : kv_heads_;
    heads_info.reserve(row_count);
    for (int32_t row = 0; row < row_count; ++row) {
        heads_info.emplace_back(row, total_tokens);
    }

    auto buffers = batch_allocate_cpu_buffer_cb_(layer_idx_, heads_info);
    if (buffers.size() != static_cast<size_t>(row_count)) {
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                  << ": Batch allocate returned wrong size ("
                  << buffers.size() << " != " << row_count << ")\n";
        return false;
    }

    for (int32_t row = 0; row < row_count; ++row) {
        const auto& desc = buffers[row];
        if (desc.key_buffer_ptr == nullptr || desc.value_buffer_ptr == nullptr) {
            std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_ << " Row " << row
                      << ": buffer pointer is null\n";
            return false;
        }
        if (desc.buffer_capacity < static_cast<size_t>(total_tokens)) {
            std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_ << " Row " << row
                      << ": buffer capacity insufficient ("
                      << desc.buffer_capacity << " < " << total_tokens << ")\n";
            return false;
        }
    }

    direct_write_buffers_ = std::move(buffers);
    direct_write_buffers_ready_ = true;
    return true;
}


bool LayerPipeline::write_cpu_buffers() {
    if (!enable_direct_write_) {
        return true;
    }

    if (direct_write_buffers_ready_) {
        return true;
    }

    std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
              << ": Direct write buffers not prepared\n";
    return false;
}

// ════════════════════════════════════════════════════════════════════════
// 🔸 Chunk Pipeline Implementation
// ════════════════════════════════════════════════════════════════════════

bool LayerPipeline::start_chunk_pipeline(int total_middle_tokens, int chunk_size) {
    return start_chunk_pipeline_batch(std::vector<int32_t>{static_cast<int32_t>(total_middle_tokens)}, chunk_size);
}

bool LayerPipeline::start_chunk_pipeline_batch(const std::vector<int32_t>& token_counts, int chunk_size) {
    if (chunk_mode_) {
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                  << ": Chunk pipeline already started!\n";
        return false;
    }

    if (token_counts.empty()) {
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                  << ": token_counts must be non-empty\n";
        return false;
    }
    int max_tokens = 0;
    for (int32_t count : token_counts) {
        if (count <= 0) {
            std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                      << ": token_counts entries must be positive\n";
            return false;
        }
        max_tokens = std::max(max_tokens, static_cast<int>(count));
    }
    for (int32_t count : token_counts) {
        if (count != max_tokens) {
            std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                      << ": CPU batch pipeline currently requires lockstep token_counts\n";
            return false;
        }
    }

    batch_token_counts_ = token_counts;
    batch_size_ = static_cast<int>(token_counts.size());
    row_count_ = batch_size_ * kv_heads_;
    total_middle_tokens_ = max_tokens;

    // ════════════════════════════════════════════════════════════════════
    //配置 OpenMP（reorganize/write 阶段使用）
    // ════════════════════════════════════════════════════════════════════
    worker_threads_ = determine_worker_threads(runtime_config_);
    omp_set_num_threads(worker_threads_);
    omp_set_dynamic(0);
    // ✅ 允许通过环境变量启用嵌套并行（用于加速单层K-means）
    // OpenMP nested parallelism configuration is now runtime-config driven.
    int enable_nested = runtime_config_.enable_omp_nested ? 1 : 0;
    int max_active_levels = runtime_config_.enable_omp_nested
        ? std::max(2, runtime_config_.omp_max_active_levels)
        : std::max(1, runtime_config_.omp_max_active_levels);
    omp_set_nested(enable_nested > 0 ? 1 : 0);
    omp_set_max_active_levels(std::max(1, max_active_levels));

    int min_threads_per_phase = std::max(
        1,
        std::min(runtime_config_.min_threads_per_phase, worker_threads_)
    );
    int expected_kmeans_outer = std::max(
        1,
        std::min(worker_threads_ / min_threads_per_phase, row_count_ * subspace_parts_)
    );
    int expected_kmeans_inner = std::max(1, worker_threads_ / expected_kmeans_outer);
    int expected_kmeans_cores = runtime_config_.enable_omp_nested
        ? expected_kmeans_outer * expected_kmeans_inner
        : expected_kmeans_outer;

    // ✅ 启动时打印 OpenMP 配置（仅一次，便于确认嵌套是否生效）
    static std::atomic<bool> omp_config_logged{false};
    if (!omp_config_logged.exchange(true)) {
        std::cout << "[OpenMP] nested=" << omp_get_nested()
                  << ", max_active_levels=" << omp_get_max_active_levels()
                  << ", max_threads=" << omp_get_max_threads()
                  << ", num_procs=" << omp_get_num_procs()
                  << ", thread_limit=" << omp_get_thread_limit()
                  << ", dynamic=" << omp_get_dynamic()
                  << ", worker_threads=" << worker_threads_
                  << ", min_threads_per_phase=" << min_threads_per_phase
                  << ", kmeans_tasks=" << (row_count_ * subspace_parts_)
                  << ", expected_kmeans_cores=" << expected_kmeans_cores
                  << ", kmeans_outer=" << expected_kmeans_outer
                  << ", kmeans_inner=" << expected_kmeans_inner
                  << "\n";
    }



    // ════════════════════════════════════════════════════════════════════
    // 步骤1：预分配完整 CPU raw buffer。
    // learned/hybrid path keeps raw KV in sequence-major layout so D2H can
    // land directly in the merge source without a CPU-side transpose pass.
    // ════════════════════════════════════════════════════════════════════
    try {
        auto build_opts = torch::dtype(kv_dtype_).device(torch::kCPU).pinned_memory(true);
        if (runtime_config_.learned_partition_enabled) {
            keys_sequence_cpu_ = torch::empty({total_middle_tokens_, row_count_, dim_}, build_opts);
            values_sequence_cpu_ = torch::empty({total_middle_tokens_, row_count_, dim_}, build_opts);
            keys_buffer_ = torch::Tensor();
            values_buffer_ = torch::Tensor();
            sequence_major_cpu_ready_ = true;
        } else {
            keys_buffer_ = torch::empty({row_count_, total_middle_tokens_, dim_}, build_opts);
            values_buffer_ = torch::empty({row_count_, total_middle_tokens_, dim_}, build_opts);
            keys_sequence_cpu_ = torch::Tensor();
            values_sequence_cpu_ = torch::Tensor();
            sequence_major_cpu_ready_ = false;
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                  << ": Failed to allocate CPU buffer: " << e.what() << "\n";
        return false;
    }
    // ════════════════════════════════════════════════════════════════════
    // 步骤2：重置运行时状态
    // ════════════════════════════════════════════════════════════════════
    completed_chunks_ = 0;
    total_chunks_ = 0;
    transfers_finished_ = false;
    ready_to_merge_.store(false, std::memory_order_release);
    max_chunk_end_ = 0;
    submitted_tokens_ = 0;
    last_chunk_seen_ = false;
    pending_payload_.reset();
    has_error_ = false;
    stop_requested_ = false;
    merge_done_ = false;
    ready_flag_ = false;
    error_message_.clear();
    n_tokens_ = 0;
    keys_cpu_ = torch::Tensor();
    values_cpu_ = torch::Tensor();
    merged_centroids_.clear();
    reorganize_results_.clear();
    direct_write_buffers_.clear();
    direct_write_buffers_ready_ = false;
    stage_.store(PipelineStage::Transferring);
    reset_pipeline_stats();

    // 如果上一轮的 merge worker 还未 join，则立即回收
    if (merge_worker_.joinable()) {
        merge_worker_.join();
    }

    if (chunk_worker_.joinable()) {
        chunk_worker_stop_.store(true, std::memory_order_release);
        chunk_queue_cv_.notify_all();
        chunk_pool_cv_.notify_all();
        chunk_worker_.join();
    }
    chunk_worker_stop_.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(chunk_queue_mutex_);
        pending_chunks_.clear();
    }

    // ════════════════════════════════════════════════════════════════════
    // 步骤3：启动 Merge Worker（负责等待所有chunk并执行一次聚类）
    // ════════════════════════════════════════════════════════════════════
    merge_worker_ = std::thread(&LayerPipeline::merge_worker_loop, this);
    chunk_worker_ = std::thread(&LayerPipeline::chunk_worker_loop, this);

    chunk_mode_ = true;
    (void)chunk_size;  // chunk_size 仅用于Python侧日志，此处不再依赖

    return true;
}

void LayerPipeline::set_worker_threads(int32_t worker_threads) {
    worker_threads_ = std::max(1, worker_threads);
    runtime_config_.worker_threads = worker_threads_;
}

void LayerPipeline::chunk_worker_loop() {
    while (true) {
        ChunkTransfer task;
        {
            std::unique_lock<std::mutex> lock(chunk_queue_mutex_);
            chunk_queue_cv_.wait(lock, [this] {
                return chunk_worker_stop_.load(std::memory_order_acquire)
                    || !pending_chunks_.empty();
            });
            if (chunk_worker_stop_.load(std::memory_order_acquire)
                && pending_chunks_.empty()) {
                return;
            }
            task = std::move(pending_chunks_.front());
            pending_chunks_.pop_front();
        }

        if (task.ready_event != nullptr) {
            auto d2h_start = std::chrono::steady_clock::now();
            cudaError_t evt_err = cudaEventSynchronize(task.ready_event);
            cudaEventDestroy(task.ready_event);
            task.ready_event = nullptr;
            auto d2h_end = std::chrono::steady_clock::now();
            double d2h_ms = std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
            {
                std::lock_guard<std::mutex> lock(pipeline_stats_mutex_);
                pipeline_stats_.d2h_ms += d2h_ms;
            }
            if (evt_err != cudaSuccess) {
                mark_failure("Chunk transfer event sync failed");
                merge_cv_.notify_one();
            }
        }

        if (has_error_.load()) {
            if (!task.direct_to_final) {
                std::lock_guard<std::mutex> lock(chunk_pool_mutex_);
                if (task.buffer_index < chunk_buffer_in_use_.size()) {
                    chunk_buffer_in_use_[task.buffer_index] = false;
                }
                chunk_pool_cv_.notify_one();
            }
            continue;
        }

        try {
            auto copy_start = std::chrono::steady_clock::now();
            if (task.batch_sequence_major_staging) {
                const size_t element_size = chunk_keys_pool_[task.buffer_index].element_size();
                const size_t token_bytes = static_cast<size_t>(dim_) * element_size;
                const size_t keys_src_batch_stride =
                    static_cast<size_t>(chunk_keys_pool_[task.buffer_index].stride(0)) * element_size;
                const size_t values_src_batch_stride =
                    static_cast<size_t>(chunk_values_pool_[task.buffer_index].stride(0)) * element_size;
                const size_t keys_src_token_stride =
                    static_cast<size_t>(chunk_keys_pool_[task.buffer_index].stride(1)) * element_size;
                const size_t values_src_token_stride =
                    static_cast<size_t>(chunk_values_pool_[task.buffer_index].stride(1)) * element_size;
                const size_t keys_src_head_stride =
                    static_cast<size_t>(chunk_keys_pool_[task.buffer_index].stride(2)) * element_size;
                const size_t values_src_head_stride =
                    static_cast<size_t>(chunk_values_pool_[task.buffer_index].stride(2)) * element_size;
                const size_t dst_head_stride = static_cast<size_t>(total_middle_tokens_) * token_bytes;
                const size_t dst_offset = static_cast<size_t>(task.start_token) * token_bytes;

                const char* src_keys_base = static_cast<const char*>(chunk_keys_pool_[task.buffer_index].data_ptr());
                const char* src_values_base = static_cast<const char*>(chunk_values_pool_[task.buffer_index].data_ptr());
                char* dst_keys_base = static_cast<char*>(keys_buffer_.data_ptr());
                char* dst_values_base = static_cast<char*>(values_buffer_.data_ptr());

                int batch_rows = std::max(1, task.batch_size * kv_heads_);
                int copy_threads = std::max(1, std::min(worker_threads_, batch_rows));
                #pragma omp parallel for num_threads(copy_threads) schedule(static)
                for (int local_row = 0; local_row < batch_rows; ++local_row) {
                    const int local_batch = local_row / kv_heads_;
                    const int head = local_row % kv_heads_;
                    const int global_row = (task.start_bdx + local_batch) * kv_heads_ + head;
                    const char* src_head_keys = src_keys_base
                        + static_cast<size_t>(local_batch) * keys_src_batch_stride
                        + static_cast<size_t>(head) * keys_src_head_stride;
                    const char* src_head_values = src_values_base
                        + static_cast<size_t>(local_batch) * values_src_batch_stride
                        + static_cast<size_t>(head) * values_src_head_stride;
                    char* dst_head_keys = dst_keys_base + static_cast<size_t>(global_row) * dst_head_stride + dst_offset;
                    char* dst_head_values = dst_values_base + static_cast<size_t>(global_row) * dst_head_stride + dst_offset;
                    for (int token = 0; token < task.chunk_len; ++token) {
                        const size_t keys_src_offset = static_cast<size_t>(token) * keys_src_token_stride;
                        const size_t values_src_offset = static_cast<size_t>(token) * values_src_token_stride;
                        const size_t dst_token_offset = static_cast<size_t>(token) * token_bytes;
                        std::memcpy(dst_head_keys + dst_token_offset, src_head_keys + keys_src_offset, token_bytes);
                        std::memcpy(dst_head_values + dst_token_offset, src_head_values + values_src_offset, token_bytes);
                    }
                }
            } else if (task.sequence_major_staging) {
                const size_t element_size = chunk_keys_pool_[task.buffer_index].element_size();
                const size_t token_bytes = static_cast<size_t>(dim_) * element_size;
                const size_t src_token_stride = static_cast<size_t>(kv_heads_) * token_bytes;
                const size_t dst_head_stride = static_cast<size_t>(total_middle_tokens_) * token_bytes;
                const size_t dst_offset = static_cast<size_t>(task.start_token) * token_bytes;

                const char* src_keys_base = static_cast<const char*>(chunk_keys_pool_[task.buffer_index].data_ptr());
                const char* src_values_base = static_cast<const char*>(chunk_values_pool_[task.buffer_index].data_ptr());
                char* dst_keys_base = static_cast<char*>(keys_buffer_.data_ptr());
                char* dst_values_base = static_cast<char*>(values_buffer_.data_ptr());

                int copy_threads = std::max(1, std::min(worker_threads_, kv_heads_));
                #pragma omp parallel for num_threads(copy_threads) schedule(static)
                for (int head = 0; head < kv_heads_; ++head) {
                    const char* src_head_keys = src_keys_base + static_cast<size_t>(head) * token_bytes;
                    const char* src_head_values = src_values_base + static_cast<size_t>(head) * token_bytes;
                    char* dst_head_keys = dst_keys_base + static_cast<size_t>(head) * dst_head_stride + dst_offset;
                    char* dst_head_values = dst_values_base + static_cast<size_t>(head) * dst_head_stride + dst_offset;
                    for (int token = 0; token < task.chunk_len; ++token) {
                        const size_t src_offset = static_cast<size_t>(token) * src_token_stride;
                        const size_t dst_token_offset = static_cast<size_t>(token) * token_bytes;
                        std::memcpy(dst_head_keys + dst_token_offset, src_head_keys + src_offset, token_bytes);
                        std::memcpy(dst_head_values + dst_token_offset, src_head_values + src_offset, token_bytes);
                    }
                }
            } else if (!task.direct_to_final) {
                const size_t element_size = chunk_keys_pool_[task.buffer_index].element_size();
                const size_t row_bytes = static_cast<size_t>(task.chunk_len) * dim_ * element_size;
                if (row_bytes > 0) {
                    const size_t src_pitch = static_cast<size_t>(chunk_buffer_capacity_) * dim_ * element_size;
                    const size_t dst_pitch = static_cast<size_t>(total_middle_tokens_) * dim_ * element_size;
                    const size_t dst_offset = static_cast<size_t>(task.start_token) * dim_ * element_size;

                    const char* src_keys_base = static_cast<const char*>(chunk_keys_pool_[task.buffer_index].data_ptr());
                    const char* src_values_base = static_cast<const char*>(chunk_values_pool_[task.buffer_index].data_ptr());
                    char* dst_keys_base = static_cast<char*>(keys_buffer_.data_ptr());
                    char* dst_values_base = static_cast<char*>(values_buffer_.data_ptr());

                    int copy_threads = std::max(1, std::min(worker_threads_, kv_heads_));
                    #pragma omp parallel for num_threads(copy_threads) schedule(static)
                    for (int head = 0; head < kv_heads_; ++head) {
                        const size_t src_head_offset = static_cast<size_t>(head) * src_pitch;
                        const size_t dst_head_offset = static_cast<size_t>(head) * dst_pitch + dst_offset;
                        std::memcpy(dst_keys_base + dst_head_offset, src_keys_base + src_head_offset, row_bytes);
                        std::memcpy(dst_values_base + dst_head_offset, src_values_base + src_head_offset, row_bytes);
                    }
                }
            }
            auto copy_end = std::chrono::steady_clock::now();
            double copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
            {
                std::lock_guard<std::mutex> lock(pipeline_stats_mutex_);
                pipeline_stats_.cpu_copy_ms += copy_ms;
                pipeline_stats_.total_chunks += 1;
                pipeline_stats_.total_tokens += static_cast<int64_t>(task.chunk_len) * std::max(1, task.batch_size);
            }
        } catch (const std::exception& exc) {
            std::string msg = "Chunk CPU copy failed: ";
            msg += exc.what();
            mark_failure(msg);
            merge_cv_.notify_one();
        }

        if (!task.direct_to_final) {
            {
                std::lock_guard<std::mutex> lock(chunk_pool_mutex_);
                if (task.buffer_index < chunk_buffer_in_use_.size()) {
                    chunk_buffer_in_use_[task.buffer_index] = false;
                }
            }
            chunk_pool_cv_.notify_one();
        }

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            int chunk_end = task.start_token + task.chunk_len;
            if (chunk_end > max_chunk_end_) {
                max_chunk_end_ = chunk_end;
            }
            if (chunk_end > submitted_tokens_) {
                submitted_tokens_ = chunk_end;
            }
        }

        int completed = ++completed_chunks_;
        int total = total_chunks_.load(std::memory_order_acquire);
        if (total > 0 && completed == total) {
            if (!last_chunk_seen_) {
                std::cerr << "⚠️  [LayerPipeline] Layer " << layer_idx_
                          << ": Completed all submitted chunks but never saw the last chunk flag. "
                          << "Proceeding with " << submitted_tokens_ << " tokens." << std::endl;
            }
            transfers_finished_.store(true, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                pending_payload_ = MergePayload{
                    .total_tokens = submitted_tokens_,
                    .last_chunk_received = last_chunk_seen_
                };
            }
            if (runtime_config_.defer_merge_until_wait) {
                ready_cv_.notify_all();
            } else {
                ready_to_merge_.store(true, std::memory_order_release);
                merge_cv_.notify_one();
            }
        }
    }
}

bool LayerPipeline::submit_raw_sequence_chunk(
    torch::Tensor keys_gpu,
    torch::Tensor values_gpu,
    int chunk_id,
    int source_token,
    int start_token,
    int chunk_len,
    bool is_last
) {
    return submit_raw_sequence_batch_chunk(
        keys_gpu,
        values_gpu,
        chunk_id,
        0,
        source_token,
        start_token,
        chunk_len,
        is_last
    );
}

bool LayerPipeline::submit_raw_sequence_batch_chunk(
    torch::Tensor keys_gpu,
    torch::Tensor values_gpu,
    int chunk_id,
    int start_bdx,
    int source_token,
    int start_token,
    int chunk_len,
    bool is_last
) {
    auto fail_with_message = [&](const std::string& msg) {
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_ << ": " << msg << "\n";
        mark_failure(msg);
        merge_cv_.notify_one();
        return false;
    };

    if (!chunk_mode_) {
        return fail_with_message("Chunk pipeline not started! Call start_chunk_pipeline() first.");
    }
    if (has_error_) {
        return fail_with_message("Previous chunk failed, cannot submit more chunks.");
    }
    if (!keys_gpu.is_cuda() || !values_gpu.is_cuda()) {
        return fail_with_message("Raw sequence chunk requires CUDA tensors.");
    }
    if (keys_gpu.dim() != 4 || values_gpu.dim() != 4) {
        return fail_with_message("Raw sequence chunk tensors must be [bsz, seq_len, kv_heads, dim].");
    }
    for (int dim_idx = 0; dim_idx < keys_gpu.dim(); ++dim_idx) {
        if (keys_gpu.size(dim_idx) != values_gpu.size(dim_idx)) {
            return fail_with_message("Raw sequence keys/values shape mismatch.");
        }
    }
    int submit_bsz = static_cast<int>(keys_gpu.size(0));
    int seq_len = static_cast<int>(keys_gpu.size(1));
    if (submit_bsz <= 0 || start_bdx < 0 || start_bdx + submit_bsz > batch_size_) {
        return fail_with_message("Raw sequence batch range is invalid.");
    }
    if (keys_gpu.size(2) != kv_heads_ || keys_gpu.size(3) != dim_) {
        return fail_with_message("Raw sequence chunk shape mismatch.");
    }
    if (keys_gpu.dtype() != kv_dtype_ || values_gpu.dtype() != kv_dtype_) {
        return fail_with_message("Raw sequence dtype mismatch.");
    }
    if (chunk_len <= 0 || source_token < 0 || source_token + chunk_len > seq_len) {
        return fail_with_message("Raw sequence source token range is invalid.");
    }
    if (start_token < 0 || start_token + chunk_len > total_middle_tokens_) {
        return fail_with_message("Raw sequence destination token range is invalid.");
    }
    if (keys_gpu.stride(3) != 1 || values_gpu.stride(3) != 1) {
        return fail_with_message("Raw sequence last dimension must be contiguous.");
    }

    cudaSetDevice(device_id_);

    if (transfer_stream_ == nullptr) {
        cudaError_t stream_err = cudaStreamCreate(&transfer_stream_);
        if (stream_err != cudaSuccess) {
            return fail_with_message(
                "Chunk " + std::to_string(chunk_id) + ": failed to create transfer stream (" +
                std::string(cudaGetErrorString(stream_err)) + ")"
            );
        }
    } else {
        cudaError_t stream_status = cudaStreamQuery(transfer_stream_);
        if (stream_status == cudaErrorInvalidResourceHandle) {
            cudaGetLastError();
            cudaError_t stream_err = cudaStreamCreate(&transfer_stream_);
            if (stream_err != cudaSuccess) {
                return fail_with_message(
                    "Chunk " + std::to_string(chunk_id) + ": failed to recreate transfer stream (" +
                    std::string(cudaGetErrorString(stream_err)) + ")"
                );
            }
        }
    }

    cudaEvent_t input_ready_event = nullptr;
    cudaError_t cuda_err = cudaEventCreateWithFlags(&input_ready_event, cudaEventDisableTiming);
    if (cuda_err != cudaSuccess) {
        return fail_with_message(
            "Chunk " + std::to_string(chunk_id) + ": cudaEventCreate failed (" +
            cudaGetErrorString(cuda_err) + ")"
        );
    }

    auto current_stream = c10::cuda::getCurrentCUDAStream(device_id_);
    cuda_err = cudaEventRecord(input_ready_event, current_stream.stream());
    if (cuda_err != cudaSuccess) {
        cudaEventDestroy(input_ready_event);
        return fail_with_message(
            "Chunk " + std::to_string(chunk_id) + ": cudaEventRecord failed (" +
            cudaGetErrorString(cuda_err) + ")"
        );
    }
    cuda_err = cudaStreamWaitEvent(transfer_stream_, input_ready_event, 0);
    cudaEventDestroy(input_ready_event);
    if (cuda_err != cudaSuccess) {
        return fail_with_message(
            "Chunk " + std::to_string(chunk_id) + ": cudaStreamWaitEvent failed (" +
            cudaGetErrorString(cuda_err) + ")"
        );
    }

    const size_t element_size = keys_gpu.element_size();
    const size_t keys_src_seq_pitch = static_cast<size_t>(keys_gpu.stride(1)) * element_size;
    const size_t values_src_seq_pitch = static_cast<size_t>(values_gpu.stride(1)) * element_size;
    const size_t src_batch_pitch = static_cast<size_t>(keys_gpu.stride(0)) * element_size;
    const char* keys_src_tensor = static_cast<const char*>(keys_gpu.data_ptr());
    const char* values_src_tensor = static_cast<const char*>(values_gpu.data_ptr());
    const bool sequence_major_contiguous =
        keys_gpu.stride(3) == 1 &&
        values_gpu.stride(3) == 1 &&
        keys_gpu.stride(2) == dim_ &&
        values_gpu.stride(2) == dim_ &&
        keys_gpu.stride(1) == kv_heads_ * dim_ &&
        values_gpu.stride(1) == kv_heads_ * dim_ &&
        keys_gpu.stride(0) == seq_len * kv_heads_ * dim_ &&
        values_gpu.stride(0) == seq_len * kv_heads_ * dim_;
    if (!sequence_major_contiguous) {
        return fail_with_message(
            "Raw sequence bulk D2H requires contiguous [bsz, seq_len, kv_heads, dim] layout; "
            "strided 2D D2H fallback is disabled."
        );
    }

    if (runtime_config_.learned_partition_enabled && sequence_major_cpu_ready_) {
        if (!keys_sequence_cpu_.defined() || !values_sequence_cpu_.defined()) {
            return fail_with_message("Sequence-major CPU raw buffers are not allocated");
        }
        if (keys_sequence_cpu_.size(0) < start_token + chunk_len ||
            values_sequence_cpu_.size(0) < start_token + chunk_len ||
            keys_sequence_cpu_.size(1) != row_count_ ||
            values_sequence_cpu_.size(1) != row_count_) {
            return fail_with_message("Sequence-major CPU raw buffer shape mismatch");
        }

        const size_t dst_row_pitch =
            static_cast<size_t>(row_count_) * static_cast<size_t>(dim_) * element_size;
        const size_t row_width =
            static_cast<size_t>(kv_heads_) * static_cast<size_t>(dim_) * element_size;
        char* keys_seq_base = static_cast<char*>(keys_sequence_cpu_.data_ptr());
        char* values_seq_base = static_cast<char*>(values_sequence_cpu_.data_ptr());

        for (int local_bdx = 0; local_bdx < submit_bsz; ++local_bdx) {
            const int global_bdx = start_bdx + local_bdx;
            const size_t dst_offset =
                (static_cast<size_t>(start_token) * static_cast<size_t>(row_count_) +
                 static_cast<size_t>(global_bdx) * static_cast<size_t>(kv_heads_)) *
                static_cast<size_t>(dim_) * element_size;
            const char* keys_src_base = keys_src_tensor
                + static_cast<size_t>(local_bdx) * src_batch_pitch
                + static_cast<size_t>(source_token) * keys_src_seq_pitch;
            const char* values_src_base = values_src_tensor
                + static_cast<size_t>(local_bdx) * src_batch_pitch
                + static_cast<size_t>(source_token) * values_src_seq_pitch;

            cudaMemcpy2DAsync(
                keys_seq_base + dst_offset,
                dst_row_pitch,
                keys_src_base,
                keys_src_seq_pitch,
                row_width,
                static_cast<size_t>(chunk_len),
                cudaMemcpyDeviceToHost,
                transfer_stream_
            );
            cudaMemcpy2DAsync(
                values_seq_base + dst_offset,
                dst_row_pitch,
                values_src_base,
                values_src_seq_pitch,
                row_width,
                static_cast<size_t>(chunk_len),
                cudaMemcpyDeviceToHost,
                transfer_stream_
            );
        }

        cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            return fail_with_message(
                "Chunk " + std::to_string(chunk_id) + ": sequence-major raw cudaMemcpy2DAsync failed (" +
                cudaGetErrorString(cuda_err) + ")"
            );
        }

        if (is_last) {
            total_chunks_.store(chunk_id + 1, std::memory_order_release);
            last_chunk_seen_ = true;
        }

        cudaEvent_t ready_event = nullptr;
        cuda_err = cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming);
        if (cuda_err != cudaSuccess) {
            return fail_with_message(
                "Chunk " + std::to_string(chunk_id) + ": cudaEventCreate failed (" +
                cudaGetErrorString(cuda_err) + ")"
            );
        }
        cuda_err = cudaEventRecord(ready_event, transfer_stream_);
        if (cuda_err != cudaSuccess) {
            cudaEventDestroy(ready_event);
            return fail_with_message(
                "Chunk " + std::to_string(chunk_id) + ": cudaEventRecord failed (" +
                cudaGetErrorString(cuda_err) + ")"
            );
        }

        {
            std::lock_guard<std::mutex> lock(chunk_queue_mutex_);
            pending_chunks_.push_back(ChunkTransfer{
                .chunk_id = chunk_id,
                .start_token = start_token,
                .chunk_len = chunk_len,
                .start_bdx = start_bdx,
                .batch_size = submit_bsz,
                .is_last = is_last,
                .buffer_index = 0,
                .direct_to_final = true,
                .sequence_major_staging = false,
                .batch_sequence_major_staging = false,
                .ready_event = ready_event,
                .keys_src = keys_gpu,
                .values_src = values_gpu,
                .keys_seq_cpu = keys_sequence_cpu_,
                .values_seq_cpu = values_sequence_cpu_,
            });
        }
        chunk_queue_cv_.notify_one();
        return true;
    }

    if (chunk_len > chunk_buffer_capacity_) {
        bool has_inflight = false;
        {
            std::lock_guard<std::mutex> lock(chunk_pool_mutex_);
            has_inflight = std::any_of(
                chunk_buffer_in_use_.begin(),
                chunk_buffer_in_use_.end(),
                [](bool in_use) { return in_use; }
            );
        }
        if (has_inflight) {
            return fail_with_message(
                "Chunk " + std::to_string(chunk_id) + ": raw staging buffer too small while in-flight"
            );
        }
        int new_capacity = std::max(chunk_len, std::max(chunk_buffer_capacity_ * 2, 1));
        if (!allocate_chunk_buffers(new_capacity)) {
            return fail_with_message(
                "Chunk " + std::to_string(chunk_id) + ": raw staging buffer allocation failed"
            );
        }
    }

    size_t buffer_index = 0;
    bool buffer_acquired = false;
    auto release_buffer = [&]() {
        if (!buffer_acquired) {
            return;
        }
        std::lock_guard<std::mutex> lock(chunk_pool_mutex_);
        if (buffer_index < chunk_buffer_in_use_.size()) {
            chunk_buffer_in_use_[buffer_index] = false;
        }
        buffer_acquired = false;
        chunk_pool_cv_.notify_one();
    };
    {
        std::unique_lock<std::mutex> lock(chunk_pool_mutex_);
        chunk_pool_cv_.wait(lock, [this] {
            if (chunk_worker_stop_.load(std::memory_order_acquire)) {
                return true;
            }
            return std::any_of(
                chunk_buffer_in_use_.begin(),
                chunk_buffer_in_use_.end(),
                [](bool in_use) { return !in_use; }
            );
        });
        if (chunk_worker_stop_.load(std::memory_order_acquire)) {
            return fail_with_message("Chunk worker stopped while waiting for raw staging buffer");
        }
        auto it = std::find(chunk_buffer_in_use_.begin(), chunk_buffer_in_use_.end(), false);
        if (it == chunk_buffer_in_use_.end()) {
            return fail_with_message("No available raw staging buffer");
        }
        buffer_index = static_cast<size_t>(std::distance(chunk_buffer_in_use_.begin(), it));
        chunk_buffer_in_use_[buffer_index] = true;
        buffer_acquired = true;
    }

    const size_t copy_bytes = static_cast<size_t>(chunk_len) * kv_heads_ * dim_ * element_size;
    const size_t keys_dst_batch_pitch =
        static_cast<size_t>(chunk_keys_pool_[buffer_index].stride(0)) * element_size;
    const size_t values_dst_batch_pitch =
        static_cast<size_t>(chunk_values_pool_[buffer_index].stride(0)) * element_size;
    char* keys_dst_tensor = static_cast<char*>(chunk_keys_pool_[buffer_index].data_ptr());
    char* values_dst_tensor = static_cast<char*>(chunk_values_pool_[buffer_index].data_ptr());
    for (int local_bdx = 0; local_bdx < submit_bsz; ++local_bdx) {
        const char* keys_src_base = keys_src_tensor
            + static_cast<size_t>(local_bdx) * src_batch_pitch
            + static_cast<size_t>(source_token) * keys_src_seq_pitch;
        const char* values_src_base = values_src_tensor
            + static_cast<size_t>(local_bdx) * src_batch_pitch
            + static_cast<size_t>(source_token) * values_src_seq_pitch;
        char* keys_dst_base = keys_dst_tensor + static_cast<size_t>(local_bdx) * keys_dst_batch_pitch;
        char* values_dst_base = values_dst_tensor + static_cast<size_t>(local_bdx) * values_dst_batch_pitch;
        cudaMemcpyAsync(
            keys_dst_base,
            keys_src_base,
            copy_bytes,
            cudaMemcpyDeviceToHost,
            transfer_stream_
        );
        cudaMemcpyAsync(
            values_dst_base,
            values_src_base,
            copy_bytes,
            cudaMemcpyDeviceToHost,
            transfer_stream_
        );
    }

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        release_buffer();
        return fail_with_message(
            "Chunk " + std::to_string(chunk_id) + ": raw bulk cudaMemcpyAsync failed (" +
            cudaGetErrorString(cuda_err) + ")"
        );
    }

    if (is_last) {
        total_chunks_.store(chunk_id + 1, std::memory_order_release);
        last_chunk_seen_ = true;
    }

    cudaEvent_t ready_event = nullptr;
    cuda_err = cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming);
    if (cuda_err != cudaSuccess) {
        release_buffer();
        return fail_with_message(
            "Chunk " + std::to_string(chunk_id) + ": cudaEventCreate failed (" +
            cudaGetErrorString(cuda_err) + ")"
        );
    }
    cuda_err = cudaEventRecord(ready_event, transfer_stream_);
    if (cuda_err != cudaSuccess) {
        cudaEventDestroy(ready_event);
        release_buffer();
        return fail_with_message(
            "Chunk " + std::to_string(chunk_id) + ": cudaEventRecord failed (" +
            cudaGetErrorString(cuda_err) + ")"
        );
    }

    {
        std::lock_guard<std::mutex> lock(chunk_queue_mutex_);
        pending_chunks_.push_back(ChunkTransfer{
            .chunk_id = chunk_id,
            .start_token = start_token,
            .chunk_len = chunk_len,
            .start_bdx = start_bdx,
            .batch_size = submit_bsz,
            .is_last = is_last,
            .buffer_index = buffer_index,
            .direct_to_final = false,
            .sequence_major_staging = submit_bsz == 1 && batch_size_ == 1 && start_bdx == 0,
            .batch_sequence_major_staging = !(submit_bsz == 1 && batch_size_ == 1 && start_bdx == 0),
            .ready_event = ready_event,
            .keys_src = keys_gpu,
            .values_src = values_gpu,
        });
    }
    buffer_acquired = false;
    chunk_queue_cv_.notify_one();

    return true;
}

/**
 * 取消 chunk pipeline（用于中途退出）
 *
 * ✅ 【P0修复】移除 chunk_mode_ 检查，无条件 join 线程
 * 原因：成功完成的 pipeline 也需要 join 线程（chunk_mode_=false 但线程仍 joinable）
 */
void LayerPipeline::cancel_pipeline() {
    stop_requested_ = true;
    ready_to_merge_.store(false, std::memory_order_release);
    transfers_finished_.store(false, std::memory_order_release);
    pending_payload_.reset();
    merge_cv_.notify_all();

    chunk_worker_stop_.store(true, std::memory_order_release);
    chunk_queue_cv_.notify_all();
    chunk_pool_cv_.notify_all();

    if (merge_worker_.joinable()) {
        try {
            merge_worker_.join();
        } catch (const std::system_error& e) {
            std::cerr << "[cancel_pipeline] Layer " << layer_idx_
                      << ": merge_worker_.join() failed (expected race): " << e.what() << "\n";
        }
    }

    if (chunk_worker_.joinable()) {
        try {
            chunk_worker_.join();
        } catch (const std::system_error& e) {
            std::cerr << "[cancel_pipeline] Layer " << layer_idx_
                      << ": chunk_worker_.join() failed: " << e.what() << "\n";
        }
    }

    chunk_mode_ = false;
    {
        std::lock_guard<std::mutex> lock(chunk_queue_mutex_);
        pending_chunks_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(chunk_pool_mutex_);
        chunk_keys_pool_.clear();
        chunk_values_pool_.clear();
        chunk_buffer_in_use_.clear();
        chunk_pool_size_ = 0;
        chunk_buffer_capacity_ = 0;
    }
}

/**
 * Merge Worker Loop（单线程，顺序敏感）
 *
 * 说明：等待所有 chunks 完成，合并结果，执行 Stage3/4
 */
void LayerPipeline::merge_worker_loop() {
    std::unique_lock<std::mutex> lock(merge_mutex_);

    // ════════════════════════════════════════════════════════════════════
    // 步骤1：等待所有 chunks 完成或收到取消/错误信号
    // ════════════════════════════════════════════════════════════════════
    merge_cv_.wait(lock, [this] {
        return ready_to_merge_.load() || has_error_.load() || stop_requested_.load();
    });

    // ════════════════════════════════════════════════════════════════════
    // 步骤2：检查是否有错误或取消
    // ════════════════════════════════════════════════════════════════════
    if (has_error_.load()) {
        std::cerr << "❌ [MergeWorker] Layer " << layer_idx_
                  << ": Chunk pipeline reported failure";
        if (!error_message_.empty()) {
            std::cerr << ": " << error_message_;
        }
        std::cerr << ", aborting merge\n";
        merge_done_ = true;
        ready_flag_ = true;
        chunk_mode_ = false;
        ready_cv_.notify_all();
        return;
    }

    if (stop_requested_.load()) {
        ready_to_merge_.store(false, std::memory_order_release);
        transfers_finished_.store(false, std::memory_order_release);
        pending_payload_.reset();
        merge_done_ = true;
        ready_flag_ = true;
        chunk_mode_ = false;
        ready_cv_.notify_all();
        return;
    }

    // ════════════════════════════════════════════════════════════════════
    // 步骤3：合并所有 chunks
    // ════════════════════════════════════════════════════════════════════
    MergePayload payload;
    {
        std::lock_guard<std::mutex> payload_lock(stats_mutex_);
        if (!pending_payload_.has_value()) {
            ready_to_merge_.store(false, std::memory_order_release);
            transfers_finished_.store(false, std::memory_order_release);
            merge_done_ = true;
            ready_flag_ = true;
            chunk_mode_ = false;
            ready_cv_.notify_all();
            return;
        }
        payload = pending_payload_.value();
        pending_payload_.reset();
    }
    ready_to_merge_.store(false, std::memory_order_release);
    transfers_finished_.store(false, std::memory_order_release);

    lock.unlock();  // 释放锁，执行耗时操作

    bool merge_success = merge_all_chunks(payload);

    if (!merge_success) {
        std::cerr << "❌ [MergeWorker] Layer " << layer_idx_ << ": Merge failed";
        if (!error_message_.empty()) {
            std::cerr << ": " << error_message_;
        }
        std::cerr << "\n";
        stage_.store(PipelineStage::Failed);
        if (error_message_.empty()) {
            error_message_ = "Merge failed";
        }
    } else {
        stage_.store(PipelineStage::Complete);
    }

    // ════════════════════════════════════════════════════════════════════
    // 步骤4：通知等待的线程 + ✅ 【P0修复】清除 chunk_mode_ 标志
    // ════════════════════════════════════════════════════════════════════
    {
        std::lock_guard<std::mutex> ready_lock(ready_mutex_);
        merge_done_ = true;
        ready_flag_ = true;
        chunk_mode_ = false;
    }
    ready_cv_.notify_all();

}

/**
 * - 步骤：收集chunks → 全局聚类 → 重组 → 写入
 */
bool LayerPipeline::merge_all_chunks(const MergePayload& payload) {

    // ════════════════════════════════════════════════════════════════════
    // 步骤1：确定需要聚类的 token 数
    // ════════════════════════════════════════════════════════════════════
    int total_tokens = payload.total_tokens;
    if (total_tokens <= 0 || total_tokens > total_middle_tokens_) {
        total_tokens = std::min(max_chunk_end_, total_middle_tokens_);
    }

    if (total_tokens <= 0) {
        std::cerr << "❌ [MergeWorker] Layer " << layer_idx_
                  << ": total_tokens is 0!\n";
        mark_failure("No tokens to cluster");
        return false;
    }
    n_tokens_ = total_tokens;
    {
        std::lock_guard<std::mutex> lock(pipeline_stats_mutex_);
        pipeline_stats_.total_tokens = std::max(
            pipeline_stats_.total_tokens,
            static_cast<int64_t>(total_tokens)
        );
    }



    // ════════════════════════════════════════════════════════════════════
    // 步骤2：获取完整KV数据。learned/hybrid CPU path uses the
    // sequence-major raw buffer written directly by D2H.
    // ════════════════════════════════════════════════════════════════════
    torch::Tensor keys_full;
    torch::Tensor values_full;
    const bool use_sequence_major_source =
        runtime_config_.learned_partition_enabled && sequence_major_cpu_ready_;
    if (use_sequence_major_source) {
        if (!keys_sequence_cpu_.defined() || !values_sequence_cpu_.defined()) {
            mark_failure("Sequence-major raw CPU buffers are missing");
            return false;
        }
        if (total_tokens == total_middle_tokens_) {
            keys_full = keys_sequence_cpu_;
            values_full = values_sequence_cpu_;
        } else {
            keys_full = keys_sequence_cpu_.index({
                torch::indexing::Slice(0, total_tokens),
                torch::indexing::Slice(),
                torch::indexing::Slice()
            });  // [total_tokens, rows, dim]

            values_full = values_sequence_cpu_.index({
                torch::indexing::Slice(0, total_tokens),
                torch::indexing::Slice(),
                torch::indexing::Slice()
            });  // [total_tokens, rows, dim]
        }
    } else if (total_tokens == total_middle_tokens_) {
        keys_full = keys_buffer_;
        values_full = values_buffer_;
    } else {
        keys_full = keys_buffer_.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, total_tokens),
            torch::indexing::Slice()
        });  // [rows, total_tokens, dim]

        values_full = values_buffer_.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, total_tokens),
            torch::indexing::Slice()
        });  // [rows, total_tokens, dim]
    }

    // ════════════════════════════════════════════════════════════════════
    // 步骤3：【关键】全局K-means聚类（per-head并行）
    // ════════════════════════════════════════════════════════════════════
    stage_.store(PipelineStage::Clustering);

    std::vector<int32_t> learned_k_values;
    const int target_cluster_size = runtime_config_.kmeans_target_cluster_size > 0
        ? runtime_config_.kmeans_target_cluster_size
        : 16;

    const int M = runtime_config_.learned_subspace_parts > 0
        ? runtime_config_.learned_subspace_parts
        : subspace_parts_;
    if (M != 8) {
        mark_failure("hybrid_8way_v2 CPU K-means requires M=8");
        return false;
    }
    if (!runtime_config_.learned_k_values.empty()) {
        learned_k_values = runtime_config_.learned_k_values;
    } else {
        learned_k_values = calculate_learned_k_values(total_tokens, target_cluster_size, M);
    }
    if (static_cast<int>(learned_k_values.size()) != M) {
        mark_failure("learned partition k_values length does not match M");
        return false;
    }
    // 确保keys_full连续性
    if (!keys_full.is_contiguous()) {
        keys_full = keys_full.contiguous();
    }
    if (!values_full.is_contiguous()) {
        values_full = values_full.contiguous();
    }

    // 获取16-bit数据指针
    const void* keys_16bit = keys_full.data_ptr();
    auto kmeans_start = std::chrono::steady_clock::now();
    const double kmeans_cpu_start_ms = get_process_cpu_time_ms();
    std::vector<std::vector<std::vector<int32_t>>> row_pair_groups;
    const int row_count = row_count_ > 0 ? row_count_ : kv_heads_;
    if (row_count <= kv_heads_) {
        row_pair_groups = runtime_config_.learned_head_pair_groups;
    } else {
        row_pair_groups.reserve(static_cast<size_t>(row_count));
        for (int row = 0; row < row_count; ++row) {
            row_pair_groups.push_back(runtime_config_.learned_head_pair_groups[row % kv_heads_]);
        }
    }
    std::unique_ptr<KmeansResult> kmeans_result = kmeans_execute_m(
        keys_16bit,
        row_count,
        total_tokens,
        dim_,
        learned_k_values,
        row_pair_groups,
        kv_dtype_,
        worker_threads_,
        runtime_config_,
        use_sequence_major_source
    );
    auto kmeans_end = std::chrono::steady_clock::now();
    const double kmeans_cpu_end_ms = get_process_cpu_time_ms();
    {
        std::lock_guard<std::mutex> lock(pipeline_stats_mutex_);
        const double kmeans_wall_ms = std::chrono::duration<double, std::milli>(
            kmeans_end - kmeans_start
        ).count();
        const double kmeans_cpu_time_ms = std::max(0.0, kmeans_cpu_end_ms - kmeans_cpu_start_ms);
        pipeline_stats_.kmeans_ms += kmeans_wall_ms;
        pipeline_stats_.kmeans_cpu_time_ms += kmeans_cpu_time_ms;
        pipeline_stats_.kmeans_cpu_util_cores = pipeline_stats_.kmeans_ms > 0.0
            ? pipeline_stats_.kmeans_cpu_time_ms / pipeline_stats_.kmeans_ms
            : 0.0;
    }

    if (!kmeans_result) {
        std::cerr << "❌ [MergeWorker] Layer " << layer_idx_
                  << ": kmeans_execute returned null (tokens=" << total_tokens;
        std::cerr << ", learned_M=" << subspace_parts_;
        std::cerr
                  << ", worker_threads=" << worker_threads_ << ")!\n";
        mark_failure("Global K-means failed");
        return false;
    }


    const std::vector<CPUBufferDescriptor>* output_buffers = nullptr;
    if (enable_direct_write_) {
        if (!prepare_direct_write_buffers(total_tokens)) {
            mark_failure("Direct write buffer preparation failed");
            return false;
        }
        output_buffers = &direct_write_buffers_;
    }

    // ════════════════════════════════════════════════════════════════════
    // 步骤4：构建reorganize输入（基于全局聚类结果）
    // ════════════════════════════════════════════════════════════════════
    stage_.store(PipelineStage::Reorganizing);

    auto reorganize_start = std::chrono::steady_clock::now();
    reorganize_from_kmeans_m(
        keys_full,
        values_full,
        *kmeans_result,
        row_pair_groups,
        total_tokens,
        reorganize_results_,
        merged_centroids_,
        output_buffers,
        use_sequence_major_source
    );

    kmeans_result_ = std::move(kmeans_result);
    if (kmeans_result_) {
        // Unified centroid outputs for Python API.
        kmeans_result_->per_head_centroids.clear();
        kmeans_result_->per_head_centroids.resize(row_count);

        for (int32_t head = 0; head < row_count; ++head) {
            if (head >= static_cast<int32_t>(merged_centroids_.size())) {
                continue;
            }
            auto centroids = merged_centroids_[head];
            if (!centroids.defined() || centroids.numel() == 0) {
                continue;
            }
            if (!centroids.is_contiguous()) {
                centroids = centroids.contiguous();
            }
            if (!centroids.device().is_cpu()) {
                centroids = centroids.cpu();
            }
            const auto numel = static_cast<size_t>(centroids.numel());
            kmeans_result_->per_head_centroids[head].resize(numel);
            std::memcpy(
                kmeans_result_->per_head_centroids[head].data(),
                centroids.data_ptr<float>(),
                numel * sizeof(float)
            );
            kmeans_result_->per_head_n_clusters[head] = static_cast<int32_t>(centroids.size(0));
        }
    }
    auto reorganize_end = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(pipeline_stats_mutex_);
        pipeline_stats_.reorganize_ms += std::chrono::duration<double, std::milli>(
            reorganize_end - reorganize_start
        ).count();
    }

    // ════════════════════════════════════════════════════════════════════
    // 步骤5：设置 keys_cpu_ 和 values_cpu_
    // ════════════════════════════════════════════════════════════════════
    if (use_sequence_major_source) {
        keys_cpu_ = torch::Tensor();
        values_cpu_ = torch::Tensor();
    } else if (total_tokens == total_middle_tokens_) {
        keys_cpu_ = keys_buffer_;
        values_cpu_ = values_buffer_;
    } else {
        keys_cpu_ = keys_buffer_.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, total_tokens),
            torch::indexing::Slice()
        });

        values_cpu_ = values_buffer_.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, total_tokens),
            torch::indexing::Slice()
        });
    }

    // ════════════════════════════════════════════════════════════════════
    // 步骤6：写入 CPU buffer
    // ════════════════════════════════════════════════════════════════════
    stage_.store(PipelineStage::Writing);

    auto write_start = std::chrono::steady_clock::now();
    bool write_ok = write_cpu_buffers();
    auto write_end = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(pipeline_stats_mutex_);
        pipeline_stats_.write_ms += std::chrono::duration<double, std::milli>(
            write_end - write_start
        ).count();
    }

    if (!write_ok) {
        std::cerr << "❌ [MergeWorker] Layer " << layer_idx_ << ": CPU write failed\n";
        mark_failure("CPU write failed");
        return false;
    }

    // ✅ 元数据写回完成后，不再需要 chunk staging / 全量 CPU buffer，立即释放内存
    {
        std::lock_guard<std::mutex> lock(chunk_pool_mutex_);
        chunk_keys_pool_.clear();
        chunk_values_pool_.clear();
        chunk_buffer_in_use_.clear();
        chunk_pool_size_ = 0;
        chunk_buffer_capacity_ = 0;
    }
    keys_buffer_ = torch::Tensor();
    values_buffer_ = torch::Tensor();
    keys_cpu_ = torch::Tensor();
    values_cpu_ = torch::Tensor();
    keys_sequence_cpu_ = torch::Tensor();
    values_sequence_cpu_ = torch::Tensor();
    sequence_major_cpu_ready_ = false;
 

    return true;
}

} // namespace imi
