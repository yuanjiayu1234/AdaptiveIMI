// ============================================================================
// layer_pipeline.hpp - End-to-End Layer Pipeline (32-Layer Parallel Safe)
// ============================================================================
#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <deque>
#include <memory>
#include <functional>
#include <optional>
#include "types.hpp"  // ✅ 基础类型别名（fp16等）

namespace imi {

// ════════════════════════════════════════════════════════════════════════
// Forward Declarations
// ════════════════════════════════════════════════════════════════════════
struct PipelineContext;
class KmeansExecutor;
class ReorganizeExecutor;
struct HeadReorgInput;

// ════════════════════════════════════════════════════════════════════════
// Pipeline Stage Enum
// ════════════════════════════════════════════════════════════════════════
enum class PipelineStage : uint8_t {
    Idle = 0,
    Transferring = 1,
    Clustering = 2,
    Reorganizing = 3,
    Writing = 4,
    Complete = 5,
    Failed = 255
};

struct PipelineStats {
    double d2h_ms = 0.0;
    double cpu_copy_ms = 0.0;
    double kmeans_ms = 0.0;
    double kmeans_cpu_time_ms = 0.0;
    double kmeans_cpu_util_cores = 0.0;
    double reorganize_ms = 0.0;
    double write_ms = 0.0;
    int64_t total_chunks = 0;
    int64_t total_tokens = 0;
    int64_t kmeans_gate_wait_ms = 0;
};



// ════════════════════════════════════════════════════════════════════════
// Kmeans Result (Two-stage IMI format - 内部使用)
// ════════════════════════════════════════════════════════════════════════
struct KmeansResult {
    // ✅ 【内部格式】K-means产生两阶段结果（IMI算法本质）
    std::vector<std::vector<int32_t>> per_head_labels1;          // [kv_heads][n_tokens]
    std::vector<std::vector<int32_t>> per_head_labels2;          // [kv_heads][n_tokens]
    std::vector<std::vector<int32_t>> per_head_labels3;          // [kv_heads][n_tokens]
    std::vector<std::vector<int32_t>> per_head_labels4;          // [kv_heads][n_tokens]

    std::vector<std::vector<float>> per_head_centroids1;         // [kv_heads][k1 * dim_half]
    std::vector<std::vector<float>> per_head_centroids2;         // [kv_heads][k2 * dim_half]
    std::vector<std::vector<float>> per_head_centroids3;         // [kv_heads][k3 * dim_quarter]
    std::vector<std::vector<float>> per_head_centroids4;         // [kv_heads][k4 * dim_quarter]

    // ✅ 【统一输出】合并后的labels和centroids（对外接口）
    std::vector<std::vector<int32_t>> per_head_labels;          // [kv_heads][n_tokens]（已合并）
    std::vector<std::vector<float>> per_head_centroids;         // [kv_heads][n_clusters * dim]（已合并）

    // ✅ 实际簇数
    std::vector<int32_t> per_head_n_clusters;                   // [kv_heads]

    int32_t kv_heads = 8;
    int32_t n_tokens = 0;
    int32_t dim = 128;
    int32_t subspace_parts = 2;

    // ⚠️ k1/k2 已废弃（独立 merge 模式）
    int32_t k1 = 0;
    int32_t k2 = 0;
};

// ════════════════════════════════════════════════════════════════════════
// Reorganize Result (Per-Head)
// ════════════════════════════════════════════════════════════════════════
struct ReorganizeResult {
    torch::Tensor reorganized_keys;      // [total_tokens, dim] CPU
    torch::Tensor reorganized_values;    // [total_tokens, dim] CPU
    std::vector<int32_t> cluster_offsets; // [n_clusters+1]
    std::vector<int32_t> cluster_sizes;   // [n_clusters]
};

// ════════════════════════════════════════════════════════════════════════
// CPU Buffer Write Interface (forward declarations)
// ════════════════════════════════════════════════════════════════════════
struct CPUBufferDescriptor {
    void* key_buffer_ptr;       // 16-bit buffer [total_tokens, dim] (FP16/BF16)
    void* value_buffer_ptr;     // 16-bit buffer [total_tokens, dim] (FP16/BF16)
    size_t buffer_capacity;     // Max tokens capacity
    int32_t layer_idx;
    int32_t head_idx;
};

// Callback types (Original - per-head)
    using AllocateCPUBufferCallback = std::function<CPUBufferDescriptor(int32_t layer_idx, int32_t head_idx, int32_t required_tokens)>;
    using RegisterFromCppCallback = std::function<void(int32_t layer_idx, int32_t head_idx,
                                                       const std::vector<int32_t>& cluster_sizes,
                                                       const std::vector<int32_t>& cluster_offsets,
                                                       const torch::Tensor& centroids)>;

// ✅ 【GIL优化】Batch callback types (8 heads in one call)
// Batch allocate: 输入 List[(head_idx, required_tokens)]，输出 List[CPUBufferDescriptor]
using BatchAllocateCPUBufferCallback = std::function<std::vector<CPUBufferDescriptor>(
    int32_t layer_idx,
    const std::vector<std::pair<int32_t, int32_t>>& heads_info  // [(head_idx, required_tokens)]
)>;

// Batch register: 输入所有heads的metadata，输出bool
struct HeadMetadata {
    int32_t head_idx;
    std::vector<int32_t> cluster_sizes;
    std::vector<int32_t> cluster_offsets;
    torch::Tensor centroids;
};

using BatchRegisterFromCppCallback = std::function<bool(
    int32_t layer_idx,
    const std::vector<HeadMetadata>& all_heads_data
)>;



// ════════════════════════════════════════════════════════════════════════
// LayerPipeline - Complete End-to-End Pipeline
// ════════════════════════════════════════════════════════════════════════
struct LayerPipelineRuntimeConfig {
    int32_t worker_threads = 0;
    int32_t chunk_buffer_count = 2;
    int32_t min_threads_per_phase = 1;
    bool enable_omp_nested = false;
    int32_t omp_max_active_levels = 1;
    int32_t kmeans_max_iters = 25;
    float kmeans_tol = 5e-4f;
    int32_t kmeans_early_stop_min_iter = 5;
    float kmeans_early_stop_rel_tol = 5e-4f;
    int32_t fp32_convert_chunk_elems = 1 << 20;
    int32_t kmeans_target_cluster_size = 16;
    int32_t kmeans_k1 = 0;
    int32_t kmeans_k2 = 0;
    int32_t kmeans_k3 = 0;
    int32_t kmeans_k4 = 0;
};

class LayerPipeline {
public:

    struct MergePayload {
        int total_tokens = 0;
        bool last_chunk_received = false;
    };

    LayerPipeline(
        int layer_idx,
        int kv_heads,
        int max_tokens,
        int dim,
        int device_id,
        torch::Dtype kv_dtype = torch::kFloat16,
        void* cache_manager_ptr = nullptr,  // Python ClusterBasedCacheManager*
        bool enable_direct_write = true,  // Enable C++ direct CPU buffer write
        int subspace_parts = 2,
        LayerPipelineRuntimeConfig runtime_config = {}
    );

    ~LayerPipeline() noexcept;

    // ════════════════════════════════════════════════════════════════════════
    // Chunk Pipeline API（统一接口，替代原有 submit_prefill）
    // ════════════════════════════════════════════════════════════════════════

    /**
     * 初始化 chunk pipeline（首次 chunk 提交前必须调用）
     *
     * @param total_middle_tokens middle 区总 token 数（用于预分配 buffer）
     * @param chunk_size Python 侧的 chunk_size（用于容量检查和 pinned buffer 分配）
     * @return 成功返回 true
     *
     * Note: k1, k2 由 C++ 内部根据每个 chunk 的实际大小自动计算
     */
    bool start_chunk_pipeline(int total_middle_tokens, int chunk_size);

    void set_worker_threads(int32_t worker_threads);


    /**
     * 提交单个 chunk（立即返回，GPU→CPU拷贝异步完成）
     *
     * 说明：
     * - 接收 GPU tensors，在独立 transfer_stream_ 上执行异步 GPU→CPU 拷贝
     * - 函数返回时拷贝未必完成，后台线程负责等待并写入 CPU buffer
     * - C++ 会保持对源 tensor 的引用，Python 可提前释放
     *
     * @param keys [1, kv_heads, chunk_len, dim] GPU tensor (支持 FP16/BF16)
     * @param values [1, kv_heads, chunk_len, dim] GPU tensor（与 keys 类型一致）
     * @param chunk_id Chunk 编号（0-based，必须连续递增）
     * @param start_token 该 chunk 在 middle 区的起始位置（全局 offset）
     * @param is_last 是否最后一个 chunk（用于触发 merge worker）
     * @return 成功返回 true，失败返回 false（例如：pipeline 未启动，容量超限）
     */
    bool submit_chunk(
        torch::Tensor keys,       // [1, kv_heads, chunk_len, dim] GPU tensor
        torch::Tensor values,     // [1, kv_heads, chunk_len, dim] GPU tensor
        int chunk_id,
        int start_token,
        bool is_last
    );


    /**
     * 取消 chunk pipeline（用于中途退出）
     *
     * 说明：停止所有 worker 线程，清理资源
     */
    void cancel_pipeline();

    // ════════════════════════════════════════════════════════════════════════
    // Wait for completion (Releases GIL, true parallel)
    // ════════════════════════════════════════════════════════════════════════
    bool wait_ready(float timeout_sec = 60.0f);

    // ════════════════════════════════════════════════════════════════════════
    // Query status
    // ════════════════════════════════════════════════════════════════════════
    PipelineStage current_stage() const;
    std::string error_message() const;

    // ════════════════════════════════════════════════════════════════════════
    // Get results (after wait_ready() returns true)
    // ════════════════════════════════════════════════════════════════════════
    const KmeansResult* get_kmeans_result() const {
        return kmeans_result_.get();
    }

    const std::vector<ReorganizeResult>& get_reorganize_results() const {
        return reorganize_results_;
    }

    torch::Tensor get_keys_cpu() const {
        return keys_cpu_;
    }

    torch::Tensor get_values_cpu() const {
        return values_cpu_;
    }

    PipelineStats get_pipeline_stats() const;

  

   

    // ════════════════════════════════════════════════════════════════════════
    // ✅ 【GIL优化】Set batch callbacks (8 heads in one call)
    // ════════════════════════════════════════════════════════════════════════
    void set_batch_allocate_cpu_buffer_callback(BatchAllocateCPUBufferCallback callback) {
        batch_allocate_cpu_buffer_cb_ = std::move(callback);
    }

    void set_batch_register_from_cpp_callback(BatchRegisterFromCppCallback callback) {
        batch_register_from_cpp_cb_ = std::move(callback);
    }

  

private:
    struct ChunkTransfer {
        int chunk_id = 0;
        int start_token = 0;
        int chunk_len = 0;
        bool is_last = false;
        size_t buffer_index = 0;
        cudaEvent_t ready_event = nullptr;
        torch::Tensor keys_src;
        torch::Tensor values_src;
    };

    // ════════════════════════════════════════════════════════════════════════
    // 🔸 后台流水线
    // ════════════════════════════════════════════════════════════════════════
    void merge_worker_loop();     // Merge worker 线程循环（单线程）
    bool merge_all_chunks(const MergePayload& payload);      // 合并所有 chunk 结果（顺序敏感）
    void mark_failure(const std::string& message);
    bool allocate_chunk_buffers(int capacity_tokens);  // 按需申请/扩展 chunk 级别的 pinned 缓冲
    void chunk_worker_loop();  // D2H 完成后写入 CPU buffer
    void reset_pipeline_stats();

    // ════════════════════════════════════════════════════════════════════════
    // Helper functions (called by merge_all_chunks)
    // ════════════════════════════════════════════════════════════════════════
    bool write_cpu_buffers();          // Write to CPU buffer and register metadata
    bool prepare_direct_write_buffers(int32_t total_tokens);

    // ════════════════════════════════════════════════════════════════════════
    // Core Data
    // ════════════════════════════════════════════════════════════════════════
    int layer_idx_;
    int kv_heads_;
    int max_tokens_;
    int dim_;
    torch::Dtype kv_dtype_ = torch::kFloat16;
    size_t kv_element_size_bytes_ = 2;
    int device_id_;
    int subspace_parts_ = 2;
    LayerPipelineRuntimeConfig runtime_config_{};
    // ✅ k1_, k2_ 已删除：现在在 process_chunk_kmeans() 中根据 chunk_len 自动计算
    int worker_threads_ = 32;

    // Python cache_manager reference (NOT owned)
    void* cache_manager_ptr_;
    bool enable_direct_write_;

    // ════════════════════════════════════════════════════════════════════════
    // CPU Buffer Write Interface
    // ════════════════════════════════════════════════════════════════════════
    // Original per-head callbacks
    AllocateCPUBufferCallback allocate_cpu_buffer_cb_;

    // ✅ 【GIL优化】Batch callbacks (priority over per-head)
    BatchAllocateCPUBufferCallback batch_allocate_cpu_buffer_cb_;
    BatchRegisterFromCppCallback batch_register_from_cpp_cb_;



    // ════════════════════════════════════════════════════════════════════════
    // CUDA Resources (Independent per-layer)
    // ════════════════════════════════════════════════════════════════════════
    cudaStream_t transfer_stream_ = nullptr;

    // ════════════════════════════════════════════════════════════════════════
    // Runtime State
    // ════════════════════════════════════════════════════════════════════════
    int n_tokens_ = 0;  // Actual token count

    // CPU tensors after transfer
    torch::Tensor keys_cpu_;    // [kv_heads, n_tokens, dim]
    torch::Tensor values_cpu_;

    // K-means result
    std::unique_ptr<KmeansResult> kmeans_result_;

    // Reorganize results (per-head)
    std::vector<ReorganizeResult> reorganize_results_;
    std::vector<CPUBufferDescriptor> direct_write_buffers_;
    bool direct_write_buffers_ready_ = false;

    // ✅ 【P0修复】Merged centroids (for CPU write)
    // Shape: [kv_heads] × [n_clusters, dim]
    std::vector<torch::Tensor> merged_centroids_;

    // ════════════════════════════════════════════════════════════════════════
    // 🔸 Chunk Mode State（新架构）
    // ════════════════════════════════════════════════════════════════════════
    bool chunk_mode_ = false;                // 是否启用 chunk 模式
    int total_middle_tokens_ = 0;            // middle 区总 token 数（预分配大小）

    // ✅ 预分配的完整 CPU buffer（零拷贝架构）
    // Shape: [kv_heads, total_middle_tokens, dim]
    torch::Tensor keys_buffer_;              // 16-bit CPU tensor (FP16/BF16)
    torch::Tensor values_buffer_;            // 16-bit CPU tensor (FP16/BF16)
    // ✅ Chunk 级别的 pinned staging buffer 池（避免整段 middle pinned）
    std::vector<torch::Tensor> chunk_keys_pool_;
    std::vector<torch::Tensor> chunk_values_pool_;
    std::vector<bool> chunk_buffer_in_use_;
    size_t chunk_pool_size_ = 0;
    int chunk_buffer_capacity_ = 0;
    std::mutex chunk_pool_mutex_;
    std::condition_variable chunk_pool_cv_;

    std::deque<ChunkTransfer> pending_chunks_;
    std::mutex chunk_queue_mutex_;
    std::condition_variable chunk_queue_cv_;
    std::atomic<bool> chunk_worker_stop_{false};
    std::thread chunk_worker_;

    // ✅ Chunk 统计（简化为一次性聚类）
    std::atomic<int> total_chunks_{0};          // 总 chunk 数（最后一个 chunk 设置）
    std::atomic<int> completed_chunks_{0};      // 已写入的 chunk 数
    std::atomic<bool> transfers_finished_{false};  // 是否已写入全部 chunk 数据
    int max_chunk_end_ = 0;                     // 记录写入的最大 token 位置
    int submitted_tokens_ = 0;                  // 实际提交的middle token数
    bool last_chunk_seen_ = false;              // 是否收到最后一个chunk
    std::mutex stats_mutex_;                    // 保护 max_chunk_end_
    std::optional<MergePayload> pending_payload_;

    // ✅ Merge Worker（仅保留单线程后台）
    std::thread merge_worker_;
    std::condition_variable merge_cv_;
    std::mutex merge_mutex_;
    std::atomic<bool> merge_done_{false};
    std::atomic<bool> stop_requested_{false};   // 终止信号
    std::atomic<bool> has_error_{false};        // 快速检查是否有错误
    std::atomic<bool> ready_to_merge_{false};   // 数据是否到位


    mutable std::mutex pipeline_stats_mutex_;
    PipelineStats pipeline_stats_{};

    // ════════════════════════════════════════════════════════════════════════
    // State Machine
    // ════════════════════════════════════════════════════════════════════════
    std::atomic<PipelineStage> stage_{PipelineStage::Idle};
    std::string error_message_;

    // ════════════════════════════════════════════════════════════════════════
    // Synchronization（被 chunk pipeline 使用）
    // ════════════════════════════════════════════════════════════════════════
    std::mutex ready_mutex_;
    std::condition_variable ready_cv_;
    bool ready_flag_ = false;

    // ════════════════════════════════════════════════════════════════════════
    // Executors (Stateless, can be shared)
    // ════════════════════════════════════════════════════════════════════════
    std::shared_ptr<KmeansExecutor> kmeans_executor_;
    std::shared_ptr<ReorganizeExecutor> reorganize_executor_;
};

// ════════════════════════════════════════════════════════════════════════
// IMI Stage2 Result (每个head独立)
// ════════════════════════════════════════════════════════════════════════
struct HeadReorgInput {
    int32_t n_active = 0;
    std::vector<int32_t> final_labels;         // [T]
    std::vector<int32_t> cluster_offsets;      // [n_active + 1]
};

// ════════════════════════════════════════════════════════════════════════
// Forward Declarations of Wrapper Functions
// ════════════════════════════════════════════════════════════════════════

// K-means executor wrapper (defined in kmeans_core.cpp)
std::unique_ptr<KmeansResult> kmeans_execute(
    const void* keys_data,       // 16-bit data (FP16/BF16)
    int32_t kv_heads,
    int32_t n_tokens,
    int32_t dim,
    int32_t k1,
    int32_t k2,
    torch::Dtype kv_dtype,
    int32_t worker_threads = 12,
    const LayerPipelineRuntimeConfig& runtime_config = {}
);

// IVF (single-space) K-means executor wrapper (defined in kmeans_core.cpp)
// - subspace_parts = 0
// - k clusters on full dim
std::unique_ptr<KmeansResult> kmeans_execute_ivf(
    const void* keys_data,       // 16-bit data (FP16/BF16)
    int32_t kv_heads,
    int32_t n_tokens,
    int32_t dim,
    int32_t k,
    torch::Dtype kv_dtype,
    int32_t worker_threads = 12,
    const LayerPipelineRuntimeConfig& runtime_config = {}
);

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
    int32_t worker_threads = 12,
    const LayerPipelineRuntimeConfig& runtime_config = {}
);





// Reorganize executor wrapper (defined in reorganize_core.cpp)
// IMI-2 compact+reorg
std::pair<HeadReorgInput, std::vector<float>> build_compact_reorg_input(
    const std::vector<int32_t>& labels1,
    const std::vector<int32_t>& labels2,
    const std::vector<float>& centroids1,
    const std::vector<float>& centroids2,
    int32_t T,
    int32_t dim_half
);

// IMI-4 compact+reorg
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
);

// IVF compact+reorg (single-space)
std::pair<HeadReorgInput, std::vector<float>> build_compact_reorg_input_ivf(
    const std::vector<int32_t>& labels,
    const std::vector<float>& centroids,
    int32_t T,
    int32_t dim
);

void reorganize_from_kmeans(
    const torch::Tensor& keys_cpu,
    const torch::Tensor& values_cpu,
    const KmeansResult& kmeans_result,
    int32_t total_tokens,
    int32_t dim_half,
    std::vector<ReorganizeResult>& out_results,
    std::vector<torch::Tensor>& merged_centroids,
    const std::vector<CPUBufferDescriptor>* output_buffers = nullptr
);

void reorganize_from_kmeans_ivf(
    const torch::Tensor& keys_cpu,
    const torch::Tensor& values_cpu,
    const KmeansResult& kmeans_result,
    int32_t total_tokens,
    std::vector<ReorganizeResult>& out_results,
    std::vector<torch::Tensor>& merged_centroids,
    const std::vector<CPUBufferDescriptor>* output_buffers = nullptr
);

void reorganize_from_kmeans_4(
    const torch::Tensor& keys_cpu,
    const torch::Tensor& values_cpu,
    const KmeansResult& kmeans_result,
    int32_t total_tokens,
    int32_t dim_quarter,
    std::vector<ReorganizeResult>& out_results,
    std::vector<torch::Tensor>& merged_centroids,
    const std::vector<CPUBufferDescriptor>* output_buffers = nullptr
);

} // namespace imi
