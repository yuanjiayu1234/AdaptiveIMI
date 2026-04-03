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
    void* cache_manager_ptr,
    bool enable_direct_write,
    int subspace_parts,
    LayerPipelineRuntimeConfig runtime_config
) : layer_idx_(layer_idx),
    kv_heads_(kv_heads),
    max_tokens_(max_tokens),
    dim_(dim),
    kv_dtype_(kv_dtype),
    device_id_(device_id),
    cache_manager_ptr_(cache_manager_ptr),
    enable_direct_write_(enable_direct_write),
    subspace_parts_(subspace_parts),
    runtime_config_(runtime_config)
{
    if (kv_dtype_ != torch::kFloat16 && kv_dtype_ != torch::kBFloat16) {
        throw std::invalid_argument(
            "[LayerPipeline] Only torch.float16 and torch.bfloat16 are supported for KV dtype");
    }
    if (subspace_parts_ != 0 && subspace_parts_ != 2 && subspace_parts_ != 4) {
        throw std::invalid_argument("[LayerPipeline] subspace_parts must be 0, 2 or 4");
    }
    if (subspace_parts_ == 4 && dim_ % 4 != 0) {
        throw std::invalid_argument("[LayerPipeline] dim must be divisible by 4 when subspace_parts=4");
    }
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

    bool success = ready_cv_.wait_for(
        lock,
        std::chrono::milliseconds(static_cast<int>(timeout_sec * 1000)),
        [this] { return ready_flag_; }
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

struct AdaptiveKValues {
    int k_ivf = 0;
    int k1 = 0;
    int k2 = 0;
    int k3 = 0;
    int k4 = 0;
};

AdaptiveKValues calculate_adaptive_k_values(int seq_len, int target_cluster_size, int subspace_parts) {
    AdaptiveKValues out;

    if (seq_len <= 0) {
        // Boundary: at least 1 cluster per stage
        if (subspace_parts == 0) {
            out.k_ivf = 1;
        } else if (subspace_parts == 4) {
            out.k1 = 1;
            out.k2 = 1;
            out.k3 = 1;
            out.k4 = 1;
        } else {
            out.k1 = 1;
            out.k2 = 1;
        }
        return out;
    }

    if (target_cluster_size <= 0) {
        target_cluster_size = 16;
    }

    if (subspace_parts == 0) {
        // IVF: k = round(seq_len / target_cluster_size)
        double total_clusters = static_cast<double>(seq_len) / target_cluster_size;
        int k = static_cast<int>(std::round(total_clusters));
        out.k_ivf = std::max(k, 1);
        return out;
    }

    if (subspace_parts == 4) {
        // 4-way IMI: k1=k2=k3=k4=round((seq_len/target_cluster_size)^(1/4)), min 2
        double total_clusters = static_cast<double>(seq_len) / target_cluster_size;
        double k_double = std::pow(total_clusters, 0.25);
        int k = static_cast<int>(std::round(k_double));
        k = std::max(k, 2);
        out.k1 = k;
        out.k2 = k;
        out.k3 = k;
        out.k4 = k;
        return out;
    }

    // 2-way IMI: k1=k2=floor(sqrt(seq_len/target_cluster_size)), min 32
    double total_clusters = static_cast<double>(seq_len) / target_cluster_size;
    int k = static_cast<int>(std::sqrt(total_clusters));
    k = std::max(k, 32);
    out.k1 = k;
    out.k2 = k;
    return out;
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
            chunk_keys_pool_.push_back(torch::empty(
                {kv_heads_, required_tokens, dim_},
                torch::dtype(kv_dtype_).device(torch::kCPU).pinned_memory(true)
            ));
            chunk_values_pool_.push_back(torch::empty(
                {kv_heads_, required_tokens, dim_},
                torch::dtype(kv_dtype_).device(torch::kCPU).pinned_memory(true)
            ));
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
    heads_info.reserve(kv_heads_);
    for (int32_t head = 0; head < kv_heads_; ++head) {
        heads_info.emplace_back(head, total_tokens);
    }

    auto buffers = batch_allocate_cpu_buffer_cb_(layer_idx_, heads_info);
    if (buffers.size() != static_cast<size_t>(kv_heads_)) {
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                  << ": Batch allocate returned wrong size ("
                  << buffers.size() << " != " << kv_heads_ << ")\n";
        return false;
    }

    for (int32_t head = 0; head < kv_heads_; ++head) {
        const auto& desc = buffers[head];
        if (desc.key_buffer_ptr == nullptr || desc.value_buffer_ptr == nullptr) {
            std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_ << " Head " << head
                      << ": buffer pointer is null\n";
            return false;
        }
        if (desc.buffer_capacity < static_cast<size_t>(total_tokens)) {
            std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_ << " Head " << head
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
        try {
            if (!batch_register_from_cpp_cb_) {
                return true;
            }

            std::vector<HeadMetadata> all_heads_data;
            all_heads_data.reserve(kv_heads_);

            for (int32_t head = 0; head < kv_heads_; ++head) {
                const auto& reorg_result = reorganize_results_[head];

                if (reorg_result.cluster_sizes.empty()) {
                    std::string error_msg =
                        "❌ Layer " + std::to_string(layer_idx_) + " Head " + std::to_string(head) +
                        ": cluster_sizes is EMPTY! This indicates reorganize failed or was not executed.";
                    std::cerr << error_msg << "\n";
                    mark_failure(error_msg);
                    throw std::runtime_error(error_msg);
                }

                bool centroids_missing =
                    (head >= static_cast<int32_t>(merged_centroids_.size())) ||
                    !merged_centroids_[head].defined() ||
                    merged_centroids_[head].numel() == 0;
                if (centroids_missing) {
                    int actual_size = (head < static_cast<int32_t>(merged_centroids_.size()) &&
                                       merged_centroids_[head].defined())
                                      ? merged_centroids_[head].numel() : -1;
                    std::string error_msg =
                        "❌ Layer " + std::to_string(layer_idx_) + " Head " + std::to_string(head) +
                        ": merged_centroids is MISSING or EMPTY! " +
                        "(size=" + std::to_string(actual_size) + "). " +
                        "This indicates global K-means failed.";
                    std::cerr << error_msg << "\n";
                    mark_failure(error_msg);
                    throw std::runtime_error(error_msg);
                }

                HeadMetadata head_data;
                head_data.head_idx = head;
                head_data.cluster_sizes = reorg_result.cluster_sizes;
                head_data.cluster_offsets = reorg_result.cluster_offsets;
                head_data.centroids = merged_centroids_[head];
                all_heads_data.push_back(std::move(head_data));
            }

            if (static_cast<int32_t>(all_heads_data.size()) != kv_heads_) {
                std::string error_msg =
                    "❌ Layer " + std::to_string(layer_idx_) +
                    ": Expected " + std::to_string(kv_heads_) + " heads, but only collected " +
                    std::to_string(all_heads_data.size()) + " heads!";
                std::cerr << error_msg << "\n";
                mark_failure(error_msg);
                throw std::runtime_error(error_msg);
            }

            if (!batch_register_from_cpp_cb_(layer_idx_, all_heads_data)) {
                std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                          << ": Batch register metadata failed\n";
                return false;
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                      << " Stage 4 failed: " << e.what() << "\n";
            return false;
        }
    }

    std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
              << ": Direct write buffers not prepared\n";
    return false;

// legacy direct-write path removed
}

// ════════════════════════════════════════════════════════════════════════
// 🔸 Chunk Pipeline Implementation
// ════════════════════════════════════════════════════════════════════════

bool LayerPipeline::start_chunk_pipeline(int total_middle_tokens, int chunk_size) {
    if (chunk_mode_) {
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                  << ": Chunk pipeline already started!\n";
        return false;
    }

    // ✅ 不再接受 k1, k2 参数，由 process_chunk_kmeans() 根据实际 chunk_len 自动计算
    total_middle_tokens_ = total_middle_tokens;

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
        std::min(worker_threads_ / min_threads_per_phase, kv_heads_ * subspace_parts_)
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
                  << ", kmeans_tasks=" << (kv_heads_ * subspace_parts_)
                  << ", expected_kmeans_cores=" << expected_kmeans_cores
                  << ", kmeans_outer=" << expected_kmeans_outer
                  << ", kmeans_inner=" << expected_kmeans_inner
                  << "\n";
    }



    // ════════════════════════════════════════════════════════════════════
    // 步骤1：预分配完整 CPU buffer（⚡ 优化：使用 pinned memory）
    // ════════════════════════════════════════════════════════════════════
    try {
        keys_buffer_ = torch::empty(
            {kv_heads_, total_middle_tokens_, dim_},
            torch::dtype(kv_dtype_).device(torch::kCPU)
        );
        values_buffer_ = torch::empty(
            {kv_heads_, total_middle_tokens_, dim_},
            torch::dtype(kv_dtype_).device(torch::kCPU)
        );
    } catch (const std::exception& e) {
        std::cerr << "❌ [LayerPipeline] Layer " << layer_idx_
                  << ": Failed to allocate CPU buffer: " << e.what() << "\n";
        return false;
    }
    if (!allocate_chunk_buffers(std::max(chunk_size, 1))) {
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
            std::lock_guard<std::mutex> lock(chunk_pool_mutex_);
            if (task.buffer_index < chunk_buffer_in_use_.size()) {
                chunk_buffer_in_use_[task.buffer_index] = false;
            }
            chunk_pool_cv_.notify_one();
            continue;
        }

        try {
            auto copy_start = std::chrono::steady_clock::now();
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
            auto copy_end = std::chrono::steady_clock::now();
            double copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
            {
                std::lock_guard<std::mutex> lock(pipeline_stats_mutex_);
                pipeline_stats_.cpu_copy_ms += copy_ms;
                pipeline_stats_.total_chunks += 1;
                pipeline_stats_.total_tokens += task.chunk_len;
            }
        } catch (const std::exception& exc) {
            std::string msg = "Chunk CPU copy failed: ";
            msg += exc.what();
            mark_failure(msg);
            merge_cv_.notify_one();
        }

        {
            std::lock_guard<std::mutex> lock(chunk_pool_mutex_);
            if (task.buffer_index < chunk_buffer_in_use_.size()) {
                chunk_buffer_in_use_[task.buffer_index] = false;
            }
        }
        chunk_pool_cv_.notify_one();

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
            ready_to_merge_.store(true, std::memory_order_release);
            merge_cv_.notify_one();
        }
    }
}

bool LayerPipeline::submit_chunk(
    torch::Tensor keys_gpu,       // [1, kv_heads, chunk_len, dim]
    torch::Tensor values_gpu,
    int chunk_id,
    int start_token,
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
    int chunk_len = keys_gpu.size(2);  // [1, kv_heads, chunk_len, dim]
    if (start_token < 0 || start_token + chunk_len > total_middle_tokens_) {
        return fail_with_message(
            "Chunk " + std::to_string(chunk_id) + ": invalid token range (" +
            std::to_string(start_token) + "+" + std::to_string(chunk_len) +
            " > " + std::to_string(total_middle_tokens_) + ")"
        );
    }
    auto keys_squeezed = keys_gpu[0];      // [1,8,8192,128] → [8,8192,128]
    auto values_squeezed = values_gpu[0];  // 等价于 squeeze(0) 但更直接

    if (keys_squeezed.stride(2) != 1 || keys_squeezed.stride(1) != dim_) {
        keys_squeezed = keys_squeezed.contiguous();
    }
    if (values_squeezed.stride(2) != 1 || values_squeezed.stride(1) != dim_) {
        values_squeezed = values_squeezed.contiguous();
    }

    // ✅ 【诊断】验证 dtype（首次调试时打印）
    if (keys_squeezed.dtype() != kv_dtype_ || values_squeezed.dtype() != kv_dtype_) {
        std::string msg = "Dtype mismatch! Expected " + std::string(c10::toString(kv_dtype_)) +
                          ", got keys=" + std::string(c10::toString(keys_squeezed.scalar_type())) +
                          ", values=" + std::string(c10::toString(values_squeezed.scalar_type()));
        return fail_with_message(msg);
    }

    cudaSetDevice(device_id_);

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
                "Chunk " + std::to_string(chunk_id) + ": staging buffer too small while in-flight"
            );
        }
        int new_capacity = std::max(chunk_len, std::max(chunk_buffer_capacity_ * 2, 1));
        if (!allocate_chunk_buffers(new_capacity)) {
            return fail_with_message(
                "Chunk " + std::to_string(chunk_id) + ": staging buffer allocation failed"
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
            return fail_with_message("Chunk worker stopped while waiting for buffer");
        }
        auto it = std::find(chunk_buffer_in_use_.begin(), chunk_buffer_in_use_.end(), false);
        if (it == chunk_buffer_in_use_.end()) {
            return fail_with_message("No available staging buffer");
        }
        buffer_index = static_cast<size_t>(std::distance(chunk_buffer_in_use_.begin(), it));
        chunk_buffer_in_use_[buffer_index] = true;
        buffer_acquired = true;
    }

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
            cudaGetLastError();  // clear the error state
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
        release_buffer();
        return fail_with_message(
            "Chunk " + std::to_string(chunk_id) + ": cudaEventCreate failed (" +
            cudaGetErrorString(cuda_err) + ")"
        );
    }

    auto current_stream = c10::cuda::getCurrentCUDAStream(device_id_);
    cuda_err = cudaEventRecord(input_ready_event, current_stream.stream());
    if (cuda_err != cudaSuccess) {
        cudaEventDestroy(input_ready_event);
        release_buffer();
        return fail_with_message(
            "Chunk " + std::to_string(chunk_id) + ": cudaEventRecord failed (" +
            cudaGetErrorString(cuda_err) + ")"
        );
    }
    cuda_err = cudaStreamWaitEvent(transfer_stream_, input_ready_event, 0);
    cudaEventDestroy(input_ready_event);
    if (cuda_err != cudaSuccess) {
        release_buffer();
        return fail_with_message(
            "Chunk " + std::to_string(chunk_id) + ": cudaStreamWaitEvent failed (" +
            cudaGetErrorString(cuda_err) + ")"
        );
    }

    // ✅ 直接获取目标位置（keys_buffer_ 已是 pinned memory，支持高速 DMA）
    // ✅ 计算按 head 的行跨度，使用 cudaMemcpy2DAsync 实现单次批量拷贝
    size_t element_size = keys_squeezed.element_size();  // 2 for FP16/BF16, 4 for FP32
    size_t row_bytes = static_cast<size_t>(chunk_len) * dim_ * element_size;
    size_t dst_pitch = static_cast<size_t>(chunk_buffer_capacity_) * dim_ * element_size;
    size_t keys_src_pitch = static_cast<size_t>(keys_squeezed.stride(0)) * element_size;
    size_t values_src_pitch = static_cast<size_t>(values_squeezed.stride(0)) * element_size;

    char* keys_dst_base = static_cast<char*>(chunk_keys_pool_[buffer_index].data_ptr());
    char* values_dst_base = static_cast<char*>(chunk_values_pool_[buffer_index].data_ptr());
    const char* keys_src_base = static_cast<const char*>(keys_squeezed.data_ptr());
    const char* values_src_base = static_cast<const char*>(values_squeezed.data_ptr());

    // 🚀 单次拷贝：使用2D拷贝同时复制所有 heads
    cudaMemcpy2DAsync(
        keys_dst_base,
        dst_pitch,
        keys_src_base,
        keys_src_pitch,
        row_bytes,
        kv_heads_,
        cudaMemcpyDeviceToHost,
        transfer_stream_
    );
    cudaMemcpy2DAsync(
        values_dst_base,
        dst_pitch,
        values_src_base,
        values_src_pitch,
        row_bytes,
        kv_heads_,
        cudaMemcpyDeviceToHost,
        transfer_stream_
    );

    // ✅ 检查 CUDA 错误
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        release_buffer();
        return fail_with_message(
            "Chunk " + std::to_string(chunk_id) + ": cudaMemcpyAsync failed (" +
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
            .is_last = is_last,
            .buffer_index = buffer_index,
            .ready_event = ready_event,
            .keys_src = keys_squeezed,
            .values_src = values_squeezed,
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
                  << ": Chunk pipeline reported failure, aborting merge\n";
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
        std::cerr << "❌ [MergeWorker] Layer " << layer_idx_ << ": Merge failed\n";
        stage_.store(PipelineStage::Failed);
        error_message_ = "Merge failed";
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
    // 步骤2：获取完整KV数据（已在keys_buffer_/values_buffer_中）
    // ════════════════════════════════════════════════════════════════════
    torch::Tensor keys_full;
    torch::Tensor values_full;
    if (total_tokens == total_middle_tokens_) {
        keys_full = keys_buffer_;
        values_full = values_buffer_;
    } else {
        keys_full = keys_buffer_.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, total_tokens),
            torch::indexing::Slice()
        });  // [kv_heads, total_tokens, dim]

        values_full = values_buffer_.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, total_tokens),
            torch::indexing::Slice()
        });  // [kv_heads, total_tokens, dim]
    }

    // ════════════════════════════════════════════════════════════════════
    // 步骤3：【关键】全局K-means聚类（per-head并行）
    // ════════════════════════════════════════════════════════════════════
    stage_.store(PipelineStage::Clustering);

    // ✅ 自动计算k1/k2（基于完整数据）
    int k1 = 0;
    int k2 = 0;
    int k3 = 0;
    int k4 = 0;
    int k_ivf = 0;
    const bool has_k12 = runtime_config_.kmeans_k1 > 0 && runtime_config_.kmeans_k2 > 0;
    const bool has_k34 = runtime_config_.kmeans_k3 > 0 && runtime_config_.kmeans_k4 > 0;
    const int target_cluster_size = runtime_config_.kmeans_target_cluster_size > 0
        ? runtime_config_.kmeans_target_cluster_size
        : 16;
    if (subspace_parts_ == 0) {
        if (runtime_config_.kmeans_k1 > 0) {
            k_ivf = runtime_config_.kmeans_k1;
        } else {
            auto ks = calculate_adaptive_k_values(total_tokens, target_cluster_size, subspace_parts_);
            k_ivf = ks.k_ivf;
        }
    } else if (subspace_parts_ == 4 && has_k12 && has_k34) {
        k1 = runtime_config_.kmeans_k1;
        k2 = runtime_config_.kmeans_k2;
        k3 = runtime_config_.kmeans_k3;
        k4 = runtime_config_.kmeans_k4;
    } else if (subspace_parts_ == 2 && has_k12) {
        k1 = runtime_config_.kmeans_k1;
        k2 = runtime_config_.kmeans_k2;
    } else {
        auto ks = calculate_adaptive_k_values(total_tokens, target_cluster_size, subspace_parts_);
        k1 = ks.k1;
        k2 = ks.k2;
        k3 = ks.k3;
        k4 = ks.k4;
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
    int dim_half = dim_ / 2;
    int dim_quarter = dim_ / 4;



    auto kmeans_start = std::chrono::steady_clock::now();
    const double kmeans_cpu_start_ms = get_process_cpu_time_ms();
    std::unique_ptr<KmeansResult> kmeans_result;
    if (subspace_parts_ == 0) {
        kmeans_result = kmeans_execute_ivf(
            keys_16bit,
            kv_heads_,
            total_tokens,
            dim_,
            k_ivf,
            kv_dtype_,
            worker_threads_,
            runtime_config_
        );
    } else if (subspace_parts_ == 4) {
        kmeans_result = kmeans_execute_4(
            keys_16bit,
            kv_heads_,
            total_tokens,
            dim_,
            k1,
            k2,
            k3,
            k4,
            kv_dtype_,
            worker_threads_,
            runtime_config_
        );
    } else {
        kmeans_result = kmeans_execute(
            keys_16bit,
            kv_heads_,
            total_tokens,
            dim_,
            k1,
            k2,
            kv_dtype_,
            worker_threads_,
            runtime_config_
        );
    }
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
        if (subspace_parts_ == 0) {
            std::cerr << ", k=" << k_ivf;
        } else {
            std::cerr << ", k1=" << k1 << ", k2=" << k2;
            if (subspace_parts_ == 4) {
                std::cerr << ", k3=" << k3 << ", k4=" << k4;
            }
        }
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
    if (subspace_parts_ == 0) {
        reorganize_from_kmeans_ivf(
            keys_full,
            values_full,
            *kmeans_result,
            total_tokens,
            reorganize_results_,
            merged_centroids_,
            output_buffers
        );
    } else if (subspace_parts_ == 4) {
        reorganize_from_kmeans_4(
            keys_full,
            values_full,
            *kmeans_result,
            total_tokens,
            dim_quarter,
            reorganize_results_,
            merged_centroids_,
            output_buffers
        );
    } else {
        reorganize_from_kmeans(
            keys_full,
            values_full,
            *kmeans_result,
            total_tokens,
            dim_half,
            reorganize_results_,
            merged_centroids_,
            output_buffers
        );
    }

    kmeans_result_ = std::move(kmeans_result);
    if (kmeans_result_) {
        // Unified outputs for Python API
        kmeans_result_->per_head_centroids.clear();
        kmeans_result_->per_head_centroids.resize(kv_heads_);

        // NOTE: per_head_labels is not currently populated here.
        // If needed, it should come from the kmeans stage outputs (before reorg),
        // not reconstructed from the reorganized layout.

        for (int32_t head = 0; head < kv_heads_; ++head) {
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
    if (total_tokens == total_middle_tokens_) {
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
 

    return true;
}

} // namespace imi
