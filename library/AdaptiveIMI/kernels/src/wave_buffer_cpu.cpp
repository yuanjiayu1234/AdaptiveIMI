#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <iostream>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <cstdlib>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <omp.h>
#include "thread_pool.hpp"

#define PRE_ALLOCATED_NUM 4


class ThreadPool {
    private:
        MyThreadPool* threadpool;
    
    public:
        ThreadPool(int num_threads) {
            threadpool = new MyThreadPool();
            threadpool->Start(static_cast<uint32_t>(num_threads));
        }

        ~ThreadPool() {
            threadpool->Stop();
            delete threadpool;
        }

        MyThreadPool* get() {
            return threadpool;
        }
};


struct ClusterDescriptor {
    bool inBlockCache = false;      // whether the cluster is in the block cache
    int* GPUBlockIDs = nullptr;     // GPU block ids of the cluster, can be incontiguous
    int CPUStartIndex = 0;          // start CPU vector index of the cluster
    int BlockNum = 0;               // number of blocks of the cluster
    int LastBlockSize = 0;          // valid vector number of the last block (only last block may be not full)
    uint16_t freq = 0;
    uint16_t last_epoch = 0;
    int cache_index = -1;
};


class BufferManager {
private:
    const int capacity;             // number of total blocks for cache
    const int block_size;           // full vector number for one block
    const int max_consider_block;   // max consider block number for each group
    ClusterDescriptor* cluster_descriptors; // cluster descriptors
    std::vector<int> free_block_ids;
    std::vector<int64_t> cached_ids;
    struct HeapEntry {
        uint16_t freq;
        uint16_t epoch;
        int64_t cluster_id;
    };

    struct HeapCompare {
        bool operator()(const HeapEntry& a, const HeapEntry& b) const noexcept {
            if (a.freq != b.freq) {
                return a.freq > b.freq;
            }
            if (a.epoch != b.epoch) {
                return a.epoch > b.epoch;
            }
            return a.cluster_id > b.cluster_id;
        }
    };

    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCompare> evict_heap;
    uint16_t epoch = 0;
    uint32_t access_counter = 0;
    uint16_t decay_window = 0;

    int miss_num = 0;                   // number of missing keys
    int hit_num = 0;                    // number of hit keys
    int64_t* _miss_keys = nullptr;      // missing keys
    int64_t* _hit_keys = nullptr;       // hit keys

    static constexpr uint16_t kMaxFreq = 255;
    static constexpr uint16_t kDefaultDecayWindow = 64;
    static constexpr size_t kHeapRebuildFactor = 4;
    static constexpr size_t kHeapRebuildSlack = 64;

    inline uint16_t resolve_decay_window() const noexcept {
        const char* env_value = std::getenv("IMI_LFU_DECAY_WINDOW");
        if (env_value == nullptr || env_value[0] == '\0') {
            return kDefaultDecayWindow;
        }
        char* endptr = nullptr;
        long value = std::strtol(env_value, &endptr, 10);
        if (endptr == env_value || value <= 0) {
            return kDefaultDecayWindow;
        }
        if (value > std::numeric_limits<uint16_t>::max()) {
            return std::numeric_limits<uint16_t>::max();
        }
        return static_cast<uint16_t>(value);
    }

    inline void decay_freq(ClusterDescriptor& desc) noexcept {
        if (epoch == desc.last_epoch) {
            return;
        }
        uint16_t delta = static_cast<uint16_t>(epoch - desc.last_epoch);
        if (delta >= 16) {
            desc.freq = 0;
        } else {
            desc.freq = static_cast<uint16_t>(desc.freq >> delta);
        }
        desc.last_epoch = epoch;
    }

    inline void touch(ClusterDescriptor& desc) noexcept {
        decay_freq(desc);
        if (desc.freq < kMaxFreq) {
            desc.freq += 1;
        }
    }

    inline void record_access(int64_t cluster_id) noexcept {
        auto& desc = cluster_descriptors[cluster_id];
        evict_heap.push({desc.freq, desc.last_epoch, cluster_id});
    }

    inline void maybe_rebuild_heap() noexcept {
        if (cached_ids.empty()) {
            if (!evict_heap.empty()) {
                evict_heap = decltype(evict_heap)();
            }
            return;
        }
        size_t threshold = cached_ids.size() * kHeapRebuildFactor + kHeapRebuildSlack;
        if (evict_heap.size() <= threshold) {
            return;
        }
        decltype(evict_heap) refreshed;
        for (int64_t cluster_id : cached_ids) {
            auto& desc = cluster_descriptors[cluster_id];
            if (!desc.inBlockCache) {
                continue;
            }
            refreshed.push({desc.freq, desc.last_epoch, cluster_id});
        }
        evict_heap.swap(refreshed);
    }

    inline int64_t pick_victim() noexcept {
        while (!evict_heap.empty()) {
            HeapEntry entry = evict_heap.top();
            evict_heap.pop();
            auto& desc = cluster_descriptors[entry.cluster_id];
            if (!desc.inBlockCache) {
                continue;
            }
            decay_freq(desc);
            if (desc.freq != entry.freq || desc.last_epoch != entry.epoch) {
                evict_heap.push({desc.freq, desc.last_epoch, entry.cluster_id});
                continue;
            }
            return entry.cluster_id;
        }
        return -1;
    }

    inline void remove_cached(int64_t cluster_id) noexcept {
        auto& desc = cluster_descriptors[cluster_id];
        if (desc.cache_index < 0 || desc.cache_index >= static_cast<int>(cached_ids.size())) {
            return;
        }
        int idx = desc.cache_index;
        int64_t tail_id = cached_ids.back();
        cached_ids[idx] = tail_id;
        cluster_descriptors[tail_id].cache_index = idx;
        cached_ids.pop_back();
        desc.cache_index = -1;
    }

    inline void evict_cluster(int64_t cluster_id) noexcept {
        if (cluster_id < 0) {
            return;
        }
        auto& desc = cluster_descriptors[cluster_id];
        if (!desc.inBlockCache) {
            return;
        }
        for (int j = 0; j < desc.BlockNum; ++j) {
            int block_id = desc.GPUBlockIDs[j];
            if (block_id >= 0 && block_id < capacity) {
                free_block_ids.push_back(block_id);
            }
        }
        desc.inBlockCache = false;
        desc.freq = 0;
        desc.last_epoch = epoch;
        remove_cached(cluster_id);
    }

public:
    BufferManager(int capacity, int nprobe, int block_size, int max_consider_block, ClusterDescriptor* cluster_descriptors)
     : capacity(capacity), block_size(block_size), max_consider_block(max_consider_block),
     cluster_descriptors(cluster_descriptors) {
        decay_window = resolve_decay_window();
        free_block_ids.reserve(capacity);
        for (int i = 0; i < capacity; ++i) {
            free_block_ids.push_back(i);
        }

        _miss_keys = new int64_t[nprobe];
        _hit_keys = new int64_t[nprobe];
        miss_num = 0;
        hit_num = 0;
    }

    ~BufferManager() {
        free_block_ids.clear();
        cached_ids.clear();
        if (_miss_keys != nullptr) delete[] _miss_keys;
        if (_hit_keys != nullptr) delete[] _hit_keys;
        cluster_descriptors = nullptr;
    }

    inline std::tuple<int, int> batch_update(
        int* update_block_ids, int* update_block_sizes, int* update_block_sizes_cumsum
    ) noexcept {
        access_counter += 1;
        if (decay_window > 0 && access_counter % decay_window == 0 && epoch < std::numeric_limits<uint16_t>::max()) {
            epoch += 1;
        }
        for (int i = hit_num - 1; i >= 0; --i) {
            const int64_t& key = _hit_keys[i];
            auto& cluster_descriptor = cluster_descriptors[key];
            if (cluster_descriptor.BlockNum == 0 || cluster_descriptor.LastBlockSize == 0) {
                continue;
            }
            touch(cluster_descriptor);
            record_access(key);
        }

        int admiss_num = 0;             // number of admissible keys
        int total_blocks_needed = 0;    // total number of blocks needed for cache update
        // iterate sequentially over the miss keys and filter out keys that exceed the capacity.
        for (int i = 0; i < miss_num; ++i) {
            const int64_t& key = _miss_keys[i];
            auto& cluster_descriptor = cluster_descriptors[key];

            if (total_blocks_needed + cluster_descriptor.BlockNum <= capacity) {
                admiss_num++;
                total_blocks_needed += cluster_descriptor.BlockNum;
            } else {
                // exceed capacity, drop the following keys
                break;
            }
        }

        if (admiss_num == 0) {
            // reset
            miss_num = 0;
            hit_num = 0;
            return { 0, 0 };
        }

        // evict to ensure the have enough space for the admissible keys
        while (free_block_ids.size() < static_cast<size_t>(total_blocks_needed)) {
            int64_t victim = pick_victim();
            if (victim < 0) {
                break;
            }
            evict_cluster(victim);
        }

        // insert the admissible keys into the cache
        int update_block_num = 0;
        int update_cumsum = 0;
        for (int i = 0; i < admiss_num; ++i) {
            const int64_t& key = _miss_keys[i];
            auto& cluster_descriptor = cluster_descriptors[key];
            if (free_block_ids.size() < static_cast<size_t>(cluster_descriptor.BlockNum)) {
                continue;
            }

            const int block_num = cluster_descriptor.BlockNum;
            const size_t free_start = free_block_ids.size() - static_cast<size_t>(block_num);
            for (int j = 0; j < block_num; ++j) {
                const int block_id = free_block_ids[free_start + block_num - 1 - j];
                const int block_span = (j + 1 == block_num) ? cluster_descriptor.LastBlockSize : block_size;
                cluster_descriptor.GPUBlockIDs[j] = block_id;
                update_block_ids[update_block_num] = block_id;
                update_block_sizes[update_block_num] = block_span;
                update_block_sizes_cumsum[update_block_num] = update_cumsum;
                update_cumsum += block_span;
                update_block_num++;
            }
            free_block_ids.resize(free_start);

            cluster_descriptor.inBlockCache = true;
            cluster_descriptor.cache_index = static_cast<int>(cached_ids.size());
            cached_ids.push_back(key);
            cluster_descriptor.last_epoch = epoch;
            cluster_descriptor.freq = 1;
            record_access(key);
        }

        // if (update_block_num != total_blocks_needed) {
        //     throw std::runtime_error("Update block ids size mismatch!");
        // }

        // reset
        miss_num = 0;
        hit_num = 0;

        maybe_rebuild_heap();

        return { admiss_num, update_block_num };
    }

    inline std::tuple<int, int, int, int> batch_access(
        const int64_t* keys, const int num, 
        int* hit_block_ids, int* hit_block_sizes, int* hit_block_sizes_cumsum,
        int* miss_block_ids, int* miss_block_sizes, int* miss_block_sizes_cumsum
    ) noexcept {
        if (num == 0) {
            return { 0, 0, 0, 0 };
        }

        miss_num = 0;
        hit_num = 0;
        int hit_block_num = 0;
        int miss_block_num = 0;
        int hit_cumsum = 0;
        int miss_cumsum = 0;
        int consider_block_num = 0;

        for (int i = 0; i < num; ++i) {
            const int64_t& key = keys[i];
            auto& cluster_descriptor = cluster_descriptors[key];
            if (cluster_descriptor.BlockNum == 0 || cluster_descriptor.LastBlockSize == 0) {
                continue;
            }
            
            consider_block_num += cluster_descriptor.BlockNum;
            if (consider_block_num > max_consider_block) {
                throw std::runtime_error("AdaptiveIMI tile exceeds max consider pages; Python tiling should guarantee buffer fit.");
            }
            
            // miss keys, copy from CPU
            if (!cluster_descriptor.inBlockCache) {
                _miss_keys[miss_num++] = key;
                int CPUStartID = cluster_descriptor.CPUStartIndex;
                for (int j = 0; j < cluster_descriptor.BlockNum - 1; ++j) {
                    miss_block_ids[miss_block_num] = CPUStartID;
                    miss_block_sizes[miss_block_num] = block_size;
                    miss_block_sizes_cumsum[miss_block_num] = miss_cumsum;
                    miss_cumsum += block_size;
                    CPUStartID += block_size;
                    miss_block_num++;
                }
                // last block
                miss_block_ids[miss_block_num] = CPUStartID;
                miss_block_sizes[miss_block_num] = cluster_descriptor.LastBlockSize;
                miss_block_sizes_cumsum[miss_block_num] = miss_cumsum;
                miss_cumsum += cluster_descriptor.LastBlockSize;
                miss_block_num++;
            // hit keys, copy from GPU Cache
            } else {
                _hit_keys[hit_num++] = key;

                int* GPUBlockIDs = cluster_descriptor.GPUBlockIDs;
                std::copy(GPUBlockIDs, GPUBlockIDs + cluster_descriptor.BlockNum, hit_block_ids + hit_block_num);
                for (int j = 0; j < cluster_descriptor.BlockNum - 1; ++j) {
                    hit_block_sizes[hit_block_num] = block_size;
                    hit_block_sizes_cumsum[hit_block_num] = hit_cumsum;
                    hit_cumsum += block_size;
                    hit_block_num++;
                }
                // last block
                hit_block_sizes[hit_block_num] = cluster_descriptor.LastBlockSize;
                hit_block_sizes_cumsum[hit_block_num] = hit_cumsum;
                hit_cumsum += cluster_descriptor.LastBlockSize;
                hit_block_num++;
            }
        }

        return { hit_num, miss_num, hit_block_num, miss_block_num };
    }
};


class WaveBufferCPU {
private:
    const int batch_size;   // total batch size
    const int group_num;    // kv_head_num
    int batch_groups;
    const int dim;          // dimension of the vector

    int nprobe;             // searched cluster number
    const int block_size;   // full vector number for one block
    const int final_n_centroids;  // final number of clusters for each group (since index may insert new data during decoding)

    const int buffer_size;  // max consider block number for each group
    const int capacity;     // cache capacity

    int num_threads;        // used thread number
    int group_per_thread;   // groups per thread for decoding and updating
    int construct_groups_per_thread; // groups per thread for construction

    MyThreadPool* pool_;                    // thread pool
    std::vector<BufferManager*> caches;     // Buffer manager

    ClusterDescriptor* cluster_descriptors; // cluster descriptors, (batch_size*group_num, final_n_centroids)

    // pointer to the retrieved clusters, [batch_size*group_num, nprobe]
    int64_t* searched_clusters_ptr;         

    // input data to re-organize keys & values based on the clustering results
    // groups = prefill_bsz*group_num when build index during prefilling
    // groups = batch_size*group_num when update index during decoding
    uint16_t* input_key_ptr;        // (groups, input_seq_length, dim)
    uint16_t* input_value_ptr;      // (groups, input_seq_length, dim)
    int* clusters_ptr;              // key id of each cluster in single batch, (groups, n_centroids, max_cluster_size)
    int* cluster_size_ptr;          // cluster size of each cluster in single batch, (groups, n_centroids)
    int64_t input_seq_length;
    int64_t max_cluster_size;
    int current_start_batch = 0;    // current processing batch index
    int process_groups = 0;         // clusters.shape[0] and cluster_size.shape[0]
    int n_centroids = 0;            // clusters.shape[1] and cluster_size.shape[1]

    // last-round sequence length of each group, [batch_size*group_num]
    int* last_seq_lengths = nullptr;
    // last-round number of clusters
    int last_n_centroids;
    uint64_t last_hit_blocks = 0;
    uint64_t last_miss_blocks = 0;

public:
    // output indices buffer    
    int* hit_block_ids;             // cache block ids of hit keys
    int* hit_block_sizes;           // valid vector number of each hiy blocks
    int* hit_block_sizes_cumsum;    // cumsum of hit_block_sizes
    int* hit_block_nums;            // number of hit block ids for each group
    
    int* miss_block_ids;            // cpu list block ids of missing keys
    int* miss_block_sizes;          // valid vector number of each missing blocks
    int* miss_block_sizes_cumsum;   // cumsum of miss_block_sizes
    int* miss_block_nums;           // number of missing block ids for each group
        
    int* update_buffer_indices;     // buffer start vector position of each update blocks
    int* update_block_sizes;        // valid vector number of each update blocks
    int* update_cache_indices;      // cache update block ids
    int* update_block_nums;         // number of update block ids for each group

    // output kv buffer
    uint16_t* ivf_key_array;        // cluster keys, (batch_size, group_num, output_seq_length, dim)
    uint16_t* ivf_value_array;      // cluster values, (batch_size, group_num, output_seq_length, dim)
    int64_t output_seq_length;


    WaveBufferCPU(int batch_size, int group_num, int dim, int nprobe, int new_nprobe, int block_size, 
        int final_n_centroids, int buffer_size, int capacity, int threads, MyThreadPool* pool)
     : batch_size(batch_size), group_num(group_num), dim(dim), nprobe(nprobe), block_size(block_size),
     final_n_centroids(final_n_centroids), buffer_size(buffer_size), capacity(capacity), pool_(pool) {
        batch_groups = batch_size * group_num;
        // count valid threads and groups per thread
        int min_group_per_thread = 1;
        num_threads = std::min(threads, (batch_groups + min_group_per_thread - 1) / min_group_per_thread);
        group_per_thread = (batch_groups + num_threads - 1) / num_threads;

        ivf_key_array = nullptr;
        ivf_value_array = nullptr;
        
        input_key_ptr = nullptr;
        input_value_ptr = nullptr;
        clusters_ptr = nullptr;
        cluster_size_ptr = nullptr;

        searched_clusters_ptr = nullptr;

        hit_block_ids = nullptr;
        hit_block_sizes = nullptr;
        hit_block_sizes_cumsum = nullptr;
        hit_block_nums = nullptr;

        miss_block_ids = nullptr;
        miss_block_sizes = nullptr;
        miss_block_sizes_cumsum = nullptr;
        miss_block_nums = nullptr;

        update_buffer_indices = nullptr;
        update_block_sizes = nullptr;
        update_cache_indices = nullptr;
        update_block_nums = nullptr;

        last_seq_lengths = new int[batch_groups];
        std::fill(last_seq_lengths, last_seq_lengths + batch_groups, 0);
        last_n_centroids = 0;

        // store descriptor of each cluster
        if (final_n_centroids == 0) {   // will not build index at all
            cluster_descriptors = nullptr;
        } else {
            cluster_descriptors = new ClusterDescriptor[batch_groups * final_n_centroids];
            for (int i = 0; i < batch_groups * final_n_centroids; ++i) {
                cluster_descriptors[i].GPUBlockIDs = new int[PRE_ALLOCATED_NUM];    // pre-allocate memory
            }
        }

        // prepare caches
        caches.resize(batch_groups, nullptr);
        if (final_n_centroids > 0) {
            for (int i = 0; i < batch_groups; ++i) {
                caches[i] = new BufferManager(capacity, nprobe+new_nprobe, block_size, buffer_size,
                                              cluster_descriptors + i * final_n_centroids);
            }
        }
    }

    ~WaveBufferCPU() {
        for (int i = 0; i < batch_groups; ++i) {
            if (caches[i] != nullptr) {
                delete caches[i];
                caches[i] = nullptr;
            }
        }
        caches.clear();
        
        if (cluster_descriptors != nullptr) {
            for (int i = 0; i < batch_groups * final_n_centroids; ++i) {
                if (cluster_descriptors[i].GPUBlockIDs != nullptr) {
                    delete[] cluster_descriptors[i].GPUBlockIDs;
                    cluster_descriptors[i].GPUBlockIDs = nullptr;
                }
            }
            delete[] cluster_descriptors;
            cluster_descriptors = nullptr;
        }

        if (last_seq_lengths != nullptr) {
            delete[] last_seq_lengths;
            last_seq_lengths = nullptr;
        }

        ivf_key_array = nullptr;
        ivf_value_array = nullptr;

        input_key_ptr = nullptr;
        input_value_ptr = nullptr;
        clusters_ptr = nullptr;
        cluster_size_ptr = nullptr;

        searched_clusters_ptr = nullptr;

        hit_block_ids = nullptr;
        hit_block_sizes = nullptr;
        hit_block_sizes_cumsum = nullptr;
        hit_block_nums = nullptr;

        miss_block_ids = nullptr;
        miss_block_sizes = nullptr;
        miss_block_sizes_cumsum = nullptr;
        miss_block_nums = nullptr;

        update_buffer_indices = nullptr;
        update_block_sizes = nullptr;
        update_cache_indices = nullptr;
        update_block_nums = nullptr;
    }

    void set_indices(
        torch::Tensor& hit_block_ids_tensor,
        torch::Tensor& hit_block_sizes_tensor,
        torch::Tensor& hit_block_sizes_cumsum_tensor,
        torch::Tensor& hit_block_nums_tensor,

        torch::Tensor& miss_block_ids_tensor,
        torch::Tensor& miss_block_sizes_tensor,
        torch::Tensor& miss_block_sizes_cumsum_tensor,
        torch::Tensor& miss_block_nums_tensor,

        torch::Tensor& update_buffer_indices_tensor,
        torch::Tensor& update_block_sizes_tensor,
        torch::Tensor& update_cache_indices_tensor,
        torch::Tensor& update_block_nums_tensor,

        torch::Tensor& searched_clusters
    ) {
        hit_block_ids = static_cast<int*>(hit_block_ids_tensor.data_ptr<int32_t>());
        hit_block_sizes = static_cast<int*>(hit_block_sizes_tensor.data_ptr<int32_t>());
        hit_block_sizes_cumsum = static_cast<int*>(hit_block_sizes_cumsum_tensor.data_ptr<int32_t>());
        hit_block_nums = static_cast<int*>(hit_block_nums_tensor.data_ptr<int32_t>());
        // AT_ASSERT(hit_block_ids_tensor.size(-1) == buffer_size, "Wrong hit block ids size.");
        // AT_ASSERT(hit_block_sizes_tensor.size(-1) == buffer_size, "Wrong hit block sizes size.");
        // AT_ASSERT(hit_block_sizes_cumsum_tensor.size(-1) == buffer_size, "Wrong hit block sizes cumsum size.");

        miss_block_ids = static_cast<int*>(miss_block_ids_tensor.data_ptr<int32_t>());
        miss_block_sizes = static_cast<int*>(miss_block_sizes_tensor.data_ptr<int32_t>());
        miss_block_sizes_cumsum = static_cast<int*>(miss_block_sizes_cumsum_tensor.data_ptr<int32_t>());
        miss_block_nums = static_cast<int*>(miss_block_nums_tensor.data_ptr<int32_t>());
        // AT_ASSERT(miss_block_ids_tensor.size(-1) == buffer_size, "Wrong miss block ids size.");
        // AT_ASSERT(miss_block_sizes_tensor.size(-1) == buffer_size, "Wrong miss block sizes size.");
        // AT_ASSERT(miss_block_sizes_cumsum_tensor.size(-1) == buffer_size, "Wrong miss block sizes cumsum size.");

        update_buffer_indices = static_cast<int*>(update_buffer_indices_tensor.data_ptr<int32_t>());
        update_block_sizes = static_cast<int*>(update_block_sizes_tensor.data_ptr<int32_t>()); 
        update_cache_indices = static_cast<int*>(update_cache_indices_tensor.data_ptr<int32_t>());
        update_block_nums = static_cast<int*>(update_block_nums_tensor.data_ptr<int32_t>());
        // AT_ASSERT(update_buffer_indices_tensor.size(-1) == buffer_size, "Wrong update buffer indices size.");
        // AT_ASSERT(update_block_sizes_tensor.size(-1) == buffer_size, "Wrong update block sizes size.");
        // AT_ASSERT(update_cache_indices_tensor.size(-1) == capacity, "Wrong update cache indices size.");

        searched_clusters_ptr = static_cast<int64_t*>(searched_clusters.data_ptr<int64_t>());
        // AT_ASSERT(searched_clusters.size(-1) == nprobe, "Wrong searched clusters size.");
    }

    void set_kv(
        torch::Tensor& ivf_key_tensor,      // (batch_size, group_num, output_seq_len, dim)
        torch::Tensor& ivf_value_tensor,    // (batch_size, group_num, output_seq_len, dim)
        torch::Tensor& input_keys,          // (groups, input_seq_len, dim)
        torch::Tensor& input_values         // (groups, input_seq_len, dim)
    ) {
        if (ivf_key_tensor.dtype() == torch::kFloat16) {     // fp16
            ivf_key_array = reinterpret_cast<uint16_t*>(ivf_key_tensor.data_ptr<at::Half>());
            ivf_value_array = reinterpret_cast<uint16_t*>(ivf_value_tensor.data_ptr<at::Half>());
            input_key_ptr = reinterpret_cast<uint16_t*>(input_keys.data_ptr<at::Half>());
            input_value_ptr = reinterpret_cast<uint16_t*>(input_values.data_ptr<at::Half>());
        } else {    // bf16
            ivf_key_array = reinterpret_cast<uint16_t*>(ivf_key_tensor.data_ptr<at::BFloat16>());
            ivf_value_array = reinterpret_cast<uint16_t*>(ivf_value_tensor.data_ptr<at::BFloat16>());
            input_key_ptr = reinterpret_cast<uint16_t*>(input_keys.data_ptr<at::BFloat16>());
            input_value_ptr = reinterpret_cast<uint16_t*>(input_values.data_ptr<at::BFloat16>());
        }
        output_seq_length = ivf_key_tensor.size(2);
        input_seq_length = input_keys.size(1);
    }



    // organize the KV for each group
    void construct_func(
        void* para
        // int* clusters_ptr,       // (groups, n_centroids, max_cluster_size)
        // int* cluster_size_ptr    // (groups, n_centroids)
    ) {
        int thread_idx = reinterpret_cast<std::intptr_t>(para);
        int _start = thread_idx * construct_groups_per_thread;
        int _end = std::min((thread_idx + 1) * construct_groups_per_thread, process_groups);

        for (int idx = _start; idx < _end; ++idx) {
            auto cluster_descriptors_group = cluster_descriptors + (current_start_batch * group_num + idx) * static_cast<int64_t>(final_n_centroids);
            int* invlists = clusters_ptr + idx * n_centroids * max_cluster_size;
            int* invlists_size = cluster_size_ptr + idx * n_centroids;
            uint16_t* values = input_value_ptr + idx * input_seq_length * dim;
            uint16_t* keys = input_key_ptr + idx * input_seq_length * dim;
            uint16_t* parse_key = ivf_key_array + (current_start_batch * group_num + idx) * output_seq_length * dim;
            uint16_t* parse_value = ivf_value_array + (current_start_batch * group_num + idx) * output_seq_length * dim;

            // organize keys & values by clusters
            int start_idx = 0;
            for (int i = 0; i < n_centroids; ++i) {
                int list_size = invlists_size[i];  // actual list size, count by vectors
                // calculate the number of blocks for this cluster
                const int block_num = (list_size + block_size - 1) / block_size;

                // initialize the cluster descriptor
                cluster_descriptors_group[i].inBlockCache = false;
                cluster_descriptors_group[i].freq = 0;
                cluster_descriptors_group[i].last_epoch = 0;
                cluster_descriptors_group[i].cache_index = -1;
                cluster_descriptors_group[i].freq = 0;
                cluster_descriptors_group[i].last_epoch = 0;
                cluster_descriptors_group[i].cache_index = -1;
                cluster_descriptors_group[i].freq = 0;
                cluster_descriptors_group[i].last_epoch = 0;
                cluster_descriptors_group[i].cache_index = -1;
                cluster_descriptors_group[i].CPUStartIndex = start_idx;
                if (list_size == 0) {
                    cluster_descriptors_group[i].BlockNum = 0;
                    cluster_descriptors_group[i].LastBlockSize = 0;
                    continue;
                }
                if (block_num > PRE_ALLOCATED_NUM) {
                    delete[] cluster_descriptors_group[i].GPUBlockIDs;
                    cluster_descriptors_group[i].GPUBlockIDs = new int[block_num];  // allocate larger memory
                }
                cluster_descriptors_group[i].BlockNum = block_num;
                cluster_descriptors_group[i].LastBlockSize = list_size % block_size == 0 ? block_size : list_size % block_size;

                // copy KV vectors
                for (int j = 0; j < list_size; ++j) {
                    int kv_id = invlists[i * max_cluster_size + j];
                    memcpy(parse_value + (start_idx + j) * dim, values + kv_id * dim, dim * sizeof(uint16_t));
                    memcpy(parse_key + (start_idx + j) * dim, keys + kv_id * dim, dim * sizeof(uint16_t));
                }

                start_idx += list_size;
            }
            last_seq_lengths[current_start_batch * group_num + idx] = start_idx;
        }
    }

    void para_construct() {
        // update the last_n_centroids
        if (last_n_centroids == 0) {
            last_n_centroids = n_centroids;  
        } else {
            AT_ASSERT(last_n_centroids == n_centroids, "n_centroids not equal in one batch.");
        }

        // count valid threads and groups per thread
        int used_threads = std::min(num_threads, process_groups);
        construct_groups_per_thread = (process_groups + used_threads - 1) / used_threads;

        // submit construct job
        pool_->LockQueue();
        for (int i = 0; i < used_threads; ++i) {
            pool_->QueueJobWOLock([this](void* para) { return this->construct_func(para); }, 
                                  reinterpret_cast<void*>(static_cast<std::intptr_t>(i)));
        }
        pool_->AddNumTask(used_threads);
        pool_->UnlockQueue();
        pool_->NotifyAll();
    }

    // organize the keys & values by clustering results
    void async_construction(
        torch::Tensor& clusters,        // key id of each clusters, (groups, n_centroids, max_cluster_size), cpu int tensor
        torch::Tensor& cluster_size,    // cluster size of all clusters list, (groups, n_centroids), cpu int tensor
        int start_bdx
    ) {
        clusters_ptr = static_cast<int*>(clusters.data_ptr<int>());
        cluster_size_ptr = static_cast<int*>(cluster_size.data_ptr<int>());
        process_groups = clusters.size(0);
        n_centroids = clusters.size(1);
        max_cluster_size = clusters.size(2);
        current_start_batch = start_bdx;

        // submit construct job
        pool_->LockQueue();
        pool_->QueueJobWOLock([this](void* para) { return this->para_construct(); }, nullptr);
        pool_->AddNumTask(1);
        pool_->UnlockQueue();
        pool_->NotifyAll();

        return;
    }

    void construction_sync() {
        // wait for construction finish
        pool_->Wait();
        return;
    }



    // organize part of keys & values by clustering results
    void update_kv_func(
        void* para
        // uint16* input_key_ptr,      // (batch_groups, update_seq_len, dim)
        // uint16* input_value_ptr,    // (batch_groups, update_seq_len, dim)
        // int* clusters_ptr,          // (batch_groups, update_n_centroids, max_cluster_size)
        // int* cluster_size_ptr       // (batch_groups, update_n_centroids)
    ) {
        int thread_idx = reinterpret_cast<std::intptr_t>(para);
        int _start = thread_idx * group_per_thread;
        int _end = std::min((thread_idx + 1) * group_per_thread, batch_groups);

        // process each attention group
        for (int idx = _start; idx < _end; ++idx) {
            auto cluster_descriptors_group = cluster_descriptors + idx * static_cast<int64_t>(final_n_centroids) + last_n_centroids;

            int* invlists = clusters_ptr + idx * n_centroids * max_cluster_size;
            int* invlists_size = cluster_size_ptr + idx * n_centroids;

            uint16_t* values = input_value_ptr + idx * input_seq_length * dim;
            uint16_t* keys = input_key_ptr + idx * input_seq_length * dim;
            uint16_t* parse_key = ivf_key_array + idx * output_seq_length * dim;
            uint16_t* parse_value = ivf_value_array + idx * output_seq_length * dim;
            
            // parse clusters
            int start_idx = last_seq_lengths[idx];
            for (int i = 0; i < n_centroids; ++i) {
                int list_size = invlists_size[i];
                // calculate the number of blocks for this cluster
                const int block_num = (list_size + block_size - 1) / block_size;

                // initialize the cluster descriptor
                cluster_descriptors_group[i].inBlockCache = false;
                cluster_descriptors_group[i].CPUStartIndex = start_idx;
                if (list_size == 0) {
                    cluster_descriptors_group[i].BlockNum = 0;
                    cluster_descriptors_group[i].LastBlockSize = 0;
                    continue;
                }
                if (block_num > PRE_ALLOCATED_NUM) {
                    delete[] cluster_descriptors_group[i].GPUBlockIDs;
                    cluster_descriptors_group[i].GPUBlockIDs = new int[block_num];  // allocate larger memory
                }
                cluster_descriptors_group[i].BlockNum = block_num;
                cluster_descriptors_group[i].LastBlockSize = list_size % block_size == 0 ? block_size : list_size % block_size;

                // organize keys & values by clusters
                for (int j = 0; j < list_size; ++j) {
                    int kv_id = invlists[i * max_cluster_size + j];
                    memcpy(parse_value + (start_idx + j) * dim, values + kv_id * dim, dim * sizeof(uint16_t));
                    memcpy(parse_key + (start_idx + j) * dim, keys + kv_id * dim, dim * sizeof(uint16_t));
                }

                start_idx += list_size;
            }
            last_seq_lengths[idx] = start_idx;  // update current sequence length for this group
        }
    }

    // parallel organize batch_groups keys & values by clustering results
    void update_kv(
        torch::Tensor& update_keys,         // (batch_groups, update_seq_len, dim)
        torch::Tensor& update_values,       // (batch_groups, update_seq_len, dim)
        torch::Tensor& clusters,            // key id of each clusters, (batch_groups, update_n_centroids, max_cluster_size), cpu int tensor
        torch::Tensor& cluster_size,        // cluster size of all clusters list, (batch_groups, update_n_centroids), cpu int tensor
        torch::Tensor& searched_clusters    // new searched cluster ids
    ) {
        if (update_keys.dtype() == torch::kFloat16) {     // fp16
            input_key_ptr = reinterpret_cast<uint16_t*>(update_keys.data_ptr<at::Half>());
            input_value_ptr = reinterpret_cast<uint16_t*>(update_values.data_ptr<at::Half>());
        } else {    // bf16
            input_key_ptr = reinterpret_cast<uint16_t*>(update_keys.data_ptr<at::BFloat16>());
            input_value_ptr = reinterpret_cast<uint16_t*>(update_values.data_ptr<at::BFloat16>());
        }
        input_seq_length = update_keys.size(1);

        clusters_ptr = static_cast<int*>(clusters.data_ptr<int>());
        cluster_size_ptr = static_cast<int*>(cluster_size.data_ptr<int>());
        n_centroids = clusters.size(1);
        max_cluster_size = clusters.size(2);

        // submit update tasks
        pool_->LockQueue();
        for (int i = 0; i < num_threads; ++i) {
            pool_->QueueJobWOLock([this](void* para) { return this->update_kv_func(para); }, 
                                  reinterpret_cast<void*>(static_cast<std::intptr_t>(i)));
        }
        pool_->AddNumTask(num_threads);
        pool_->UnlockQueue();
        pool_->NotifyAll();
        pool_->Wait();  // sync

        last_n_centroids += n_centroids;    // update current number of clusters

        // update the searched clusters
        searched_clusters_ptr = static_cast<int64_t*>(searched_clusters.data_ptr<int64_t>());
        nprobe = searched_clusters.size(1);
    }

    void set_cluster_metadata(
        torch::Tensor& cluster_sizes,
        torch::Tensor& cluster_offsets,
        int start_bdx
    ) {
        cluster_size_ptr = static_cast<int*>(cluster_sizes.data_ptr<int>());
        int* cluster_offsets_ptr = static_cast<int*>(cluster_offsets.data_ptr<int>());
        process_groups = cluster_sizes.size(0);
        n_centroids = cluster_sizes.size(1);
        current_start_batch = start_bdx;

        if (last_n_centroids == 0) {
            last_n_centroids = n_centroids;
        } else {
            AT_ASSERT(last_n_centroids == n_centroids, "n_centroids not equal in one batch.");
        }

        for (int idx = 0; idx < process_groups; ++idx) {
            auto cluster_descriptors_group = cluster_descriptors + (current_start_batch * group_num + idx) * static_cast<int64_t>(final_n_centroids);
            int* sizes_group = cluster_size_ptr + idx * n_centroids;
            int* offsets_group = cluster_offsets_ptr + idx * (n_centroids + 1);
            for (int i = 0; i < n_centroids; ++i) {
                int list_size = sizes_group[i];
                cluster_descriptors_group[i].inBlockCache = false;
                cluster_descriptors_group[i].CPUStartIndex = offsets_group[i];
                if (list_size == 0) {
                    cluster_descriptors_group[i].BlockNum = 0;
                    cluster_descriptors_group[i].LastBlockSize = 0;
                    continue;
                }
                int block_num = (list_size + block_size - 1) / block_size;
                if (block_num > PRE_ALLOCATED_NUM) {
                    delete[] cluster_descriptors_group[i].GPUBlockIDs;
                    cluster_descriptors_group[i].GPUBlockIDs = new int[block_num];
                }
                cluster_descriptors_group[i].BlockNum = block_num;
                cluster_descriptors_group[i].LastBlockSize = list_size % block_size == 0 ? block_size : list_size % block_size;
            }
            last_seq_lengths[current_start_batch * group_num + idx] = offsets_group[n_centroids];
        }
    }



    void batch_access(void* para) {
        int thread_idx = reinterpret_cast<std::intptr_t>(para);
        int _start = thread_idx * group_per_thread;
        int _end = std::min((thread_idx + 1) * group_per_thread, batch_groups);

        for (int i = _start; i < _end; ++i) {
            // used for hit keys
            auto hit_block_ids_group = hit_block_ids + i * buffer_size;
            auto hit_block_sizes_group = hit_block_sizes + i * buffer_size;
            auto hit_block_sizes_cumsum_group = hit_block_sizes_cumsum + i * buffer_size;
            // used for missing keys
            auto miss_block_ids_group = miss_block_ids + i * buffer_size;
            auto miss_block_sizes_group = miss_block_sizes + i * buffer_size;
            auto miss_block_sizes_cumsum_group = miss_block_sizes_cumsum + i * buffer_size;
            // access buffer manager
            auto [hit_num, miss_num, hit_block_num, miss_block_num] = caches[i]->batch_access(searched_clusters_ptr + i * nprobe, nprobe, 
                                                                                              hit_block_ids_group,
                                                                                              hit_block_sizes_group, 
                                                                                              hit_block_sizes_cumsum_group,
                                                                                              miss_block_ids_group,
                                                                                              miss_block_sizes_group,
                                                                                              miss_block_sizes_cumsum_group);
            // std::fill(hit_block_ids_group + hit_block_num, hit_block_ids_group + buffer_size, -1);
            // std::fill(hit_block_sizes_group + hit_block_num, hit_block_sizes_group + buffer_size, 0);
            // std::fill(hit_block_sizes_cumsum_group + hit_block_num, hit_block_sizes_cumsum_group + buffer_size, hit_block_sizes_cumsum_group[hit_block_num - 1]);
            // std::fill(miss_block_ids_group + miss_block_num, miss_block_ids_group + buffer_size, -1);
            // std::fill(miss_block_sizes_group + miss_block_num, miss_block_sizes_group + buffer_size, 0);
            // std::fill(miss_block_sizes_cumsum_group + miss_block_num, miss_block_sizes_cumsum_group + buffer_size, miss_block_sizes_cumsum_group[miss_block_num - 1]);

            hit_block_nums[i] = hit_block_num;
            miss_block_nums[i] = miss_block_num;
        }
    }

    void batch_update(void* para) {
        int thread_idx = reinterpret_cast<std::intptr_t>(para);
        int _start = thread_idx * group_per_thread;
        int _end = std::min((thread_idx + 1) * group_per_thread, batch_groups);

        for (int i = _start; i < _end; ++i) {
            auto update_cache_block_ids_group = update_cache_indices + i * buffer_size;
            auto update_block_sizes_group = update_block_sizes + i * buffer_size;
            auto update_buffer_indices_group = update_buffer_indices + i * buffer_size;
            auto [admiss_num, update_block_num] = caches[i]->batch_update(update_cache_block_ids_group,
                                                                          update_block_sizes_group,
                                                                          update_buffer_indices_group);
            // std::fill(update_cache_block_ids_group + update_block_num, update_cache_block_ids_group + buffer_size, -1);
            // std::fill(update_block_sizes_group + update_block_num, update_block_sizes_group + buffer_size, 0);
            // std::fill(update_buffer_indices_group + update_block_num, update_buffer_indices_group + buffer_size, 0);

            update_block_nums[i] = update_block_num;
        }
    }

    void para_batch_updata() {
        pool_->LockQueue();
        for (int i = 0; i < num_threads; ++i) {
            pool_->QueueJobWOLock([this](void* para) { this->batch_update(para); }, 
                                  reinterpret_cast<void*>(static_cast<std::intptr_t>(i)));
        }
        pool_->AddNumTask(num_threads);
        pool_->UnlockQueue();
        pool_->NotifyAll();
    }

    void para_batch_access() {
        // create tasks
        pool_->LockQueue();
        for (int i = 0; i < num_threads; ++i) {
            pool_->QueueJobWOLock([this](void* para) { this->batch_access(para); }, 
                                  reinterpret_cast<void*>(static_cast<std::intptr_t>(i)));
        }
        pool_->AddNumTask(num_threads);
        pool_->UnlockQueue();
        pool_->NotifyAll();
        pool_->Wait();

        uint64_t hit_blocks = 0;
        uint64_t miss_blocks = 0;
        for (int i = 0; i < batch_groups; ++i) {
            hit_blocks += static_cast<uint64_t>(hit_block_nums[i]);
            miss_blocks += static_cast<uint64_t>(miss_block_nums[i]);
        }
        last_hit_blocks = hit_blocks;
        last_miss_blocks = miss_blocks;

        // submit aysnc update tasks
        pool_->LockQueue();
        pool_->QueueJobWOLock([this](void* para) { this->para_batch_updata(); }, nullptr);
        pool_->AddNumTask(1);
        pool_->UnlockQueue();
        pool_->NotifyOne();
        
        return;
    }

    std::tuple<uint64_t, uint64_t> get_last_block_stats() const noexcept {
        return {last_hit_blocks, last_miss_blocks};
    }

    void sync() {
        pool_->Wait();
        return;
    }

};


namespace py = pybind11;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<WaveBufferCPU>(m, "WaveBufferCPU")
        .def(py::init<int, int, int, int, int, int, int, int, int, int, MyThreadPool*>(),
             py::arg("batch_size"), py::arg("group_num"), py::arg("dim"), py::arg("nprobe"), py::arg("new_nprobe"),
             py::arg("block_size"), py::arg("final_n_centroids"), py::arg("buffer_size"), py::arg("capacity"), 
             py::arg("threads"), py::arg("pool"))
        .def("set_indices", &WaveBufferCPU::set_indices, 
            py::arg("hit_block_ids"), py::arg("hit_block_sizes"), py::arg("hit_block_sizes_cumsum"), py::arg("hit_block_nums"),
            py::arg("miss_block_ids"), py::arg("miss_block_sizes"), py::arg("miss_block_sizes_cumsum"), py::arg("miss_block_nums"),
            py::arg("update_buffer_indices"), py::arg("update_block_sizes"), py::arg("update_cache_indices"), py::arg("update_block_nums"), 
            py::arg("searched_clusters"))
        .def("set_kv", &WaveBufferCPU::set_kv, 
            py::arg("ivf_key"), py::arg("ivf_value"), py::arg("input_keys"), py::arg("input_values"))
        .def("set_cluster_metadata", &WaveBufferCPU::set_cluster_metadata,
            py::arg("cluster_sizes"), py::arg("cluster_offsets"), py::arg("start_bdx"))
        .def("async_construction", &WaveBufferCPU::async_construction, 
            py::arg("clusters"), py::arg("cluster_size"), py::arg("start_bdx"))
        .def("construction_sync", &WaveBufferCPU::construction_sync)
        .def("update_kv", &WaveBufferCPU::update_kv, 
            py::arg("update_keys"), py::arg("update_values"), py::arg("clusters"), py::arg("cluster_size"), py::arg("searched_clusters"))
        .def("batch_access", &WaveBufferCPU::para_batch_access, py::call_guard<py::gil_scoped_release>())
        .def("get_last_block_stats", &WaveBufferCPU::get_last_block_stats)
        .def("sync", &WaveBufferCPU::sync, py::call_guard<py::gil_scoped_release>());
    
    py::class_<MyThreadPool>(m, "MyThreadPool")
        .def(py::init<>());
    
    py::class_<ThreadPool>(m, "ThreadPool")
        .def(py::init<int>())
        .def("get", &ThreadPool::get, py::return_value_policy::reference);
}
