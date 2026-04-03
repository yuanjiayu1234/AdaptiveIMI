// ============================================================================
// bindings.cpp - Python bindings for LayerPipeline
// ============================================================================
#include "layer_pipeline.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <algorithm>
#include <cctype>
#include <cstring>

namespace py = pybind11;
using namespace imi;

// Helper: convert Python torch.dtype to torch::Dtype
torch::Dtype py_dtype_to_cpp(py::handle dtype_obj) {
    auto torch_module = py::module_::import("torch");
    auto empty_func = torch_module.attr("empty");
    auto temp_tensor = empty_func(0, py::arg("dtype") = dtype_obj);
    auto cpp_tensor = py::cast<torch::Tensor>(temp_tensor);
    return cpp_tensor.scalar_type();
}

bool py_obj_to_bool(const py::handle& obj, bool fallback) {
    if (obj.is_none()) {
        return fallback;
    }
    if (py::isinstance<py::bool_>(obj)) {
        return py::cast<bool>(obj);
    }
    if (py::isinstance<py::str>(obj)) {
        std::string value = py::cast<std::string>(obj);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (value == "1" || value == "true" || value == "yes" || value == "on") {
            return true;
        }
        if (value == "0" || value == "false" || value == "no" || value == "off") {
            return false;
        }
    }
    return fallback;
}

py::object py_dict_get_value(const py::dict& dict_obj, const char* key) {
    return dict_obj.attr("get")(py::str(key), py::none());
}

int32_t py_dict_get_i32(const py::dict& dict_obj, const char* key, int32_t fallback) {
    py::object value = py_dict_get_value(dict_obj, key);
    if (value.is_none()) {
        return fallback;
    }
    try {
        if (py::isinstance<py::bool_>(value)) {
            return py::cast<bool>(value) ? 1 : 0;
        }
        if (py::isinstance<py::int_>(value)) {
            return py::cast<int32_t>(value);
        }
        if (py::isinstance<py::float_>(value)) {
            return static_cast<int32_t>(py::cast<double>(value));
        }
        if (py::isinstance<py::str>(value)) {
            return static_cast<int32_t>(std::stoll(py::cast<std::string>(value)));
        }
    } catch (...) {
        return fallback;
    }
    return fallback;
}

float py_dict_get_f32(const py::dict& dict_obj, const char* key, float fallback) {
    py::object value = py_dict_get_value(dict_obj, key);
    if (value.is_none()) {
        return fallback;
    }
    try {
        if (py::isinstance<py::bool_>(value)) {
            return py::cast<bool>(value) ? 1.0f : 0.0f;
        }
        if (py::isinstance<py::int_>(value)) {
            return static_cast<float>(py::cast<int64_t>(value));
        }
        if (py::isinstance<py::float_>(value)) {
            return py::cast<float>(value);
        }
        if (py::isinstance<py::str>(value)) {
            return std::stof(py::cast<std::string>(value));
        }
    } catch (...) {
        return fallback;
    }
    return fallback;
}

LayerPipelineRuntimeConfig runtime_config_from_python(const py::object& runtime_config_obj) {
    LayerPipelineRuntimeConfig cfg;
    if (runtime_config_obj.is_none()) {
        return cfg;
    }

    py::dict root = py::cast<py::dict>(runtime_config_obj);
    py::dict pipeline = root.contains("pipeline") && !root[py::str("pipeline")].is_none()
        ? py::cast<py::dict>(root[py::str("pipeline")])
        : py::dict();
    py::dict kmeans = root.contains("kmeans") && !root[py::str("kmeans")].is_none()
        ? py::cast<py::dict>(root[py::str("kmeans")])
        : py::dict();

    cfg.worker_threads = py_dict_get_i32(pipeline, "worker_threads", cfg.worker_threads);
    cfg.chunk_buffer_count = py_dict_get_i32(pipeline, "chunk_buffer_count", cfg.chunk_buffer_count);
    cfg.min_threads_per_phase = py_dict_get_i32(
        pipeline,
        "min_threads_per_phase",
        cfg.min_threads_per_phase
    );
    if (pipeline.contains("enable_omp_nested")) {
        cfg.enable_omp_nested = py_obj_to_bool(pipeline[py::str("enable_omp_nested")], cfg.enable_omp_nested);
    }
    cfg.omp_max_active_levels = py_dict_get_i32(pipeline, "omp_max_active_levels", cfg.omp_max_active_levels);

    cfg.kmeans_max_iters = py_dict_get_i32(kmeans, "max_iters", cfg.kmeans_max_iters);
    cfg.kmeans_tol = py_dict_get_f32(kmeans, "tol", cfg.kmeans_tol);
    cfg.kmeans_early_stop_min_iter = py_dict_get_i32(kmeans, "early_stop_min_iter", cfg.kmeans_early_stop_min_iter);
    cfg.kmeans_early_stop_rel_tol = py_dict_get_f32(kmeans, "early_stop_rel_tol", cfg.kmeans_early_stop_rel_tol);
    cfg.fp32_convert_chunk_elems = py_dict_get_i32(kmeans, "fp32_convert_chunk_elems", cfg.fp32_convert_chunk_elems);
    cfg.kmeans_target_cluster_size = py_dict_get_i32(
        kmeans,
        "target_cluster_size",
        cfg.kmeans_target_cluster_size
    );
    cfg.kmeans_target_cluster_size = py_dict_get_i32(
        kmeans,
        "cluster_size",
        cfg.kmeans_target_cluster_size
    );
    cfg.kmeans_k1 = py_dict_get_i32(kmeans, "k1", cfg.kmeans_k1);
    cfg.kmeans_k2 = py_dict_get_i32(kmeans, "k2", cfg.kmeans_k2);
    cfg.kmeans_k3 = py_dict_get_i32(kmeans, "k3", cfg.kmeans_k3);
    cfg.kmeans_k4 = py_dict_get_i32(kmeans, "k4", cfg.kmeans_k4);
    return cfg;
}

// ════════════════════════════════════════════════════════════════════════
// Helper function to convert KmeansResult to Python dict
// ════════════════════════════════════════════════════════════════════════

py::dict kmeans_result_to_dict(const KmeansResult* result) {
    if (!result) {
        return py::dict();
    }

    py::dict d;
    d["kv_heads"] = result->kv_heads;
    d["n_tokens"] = result->n_tokens;
    d["dim"] = result->dim;
    d["k1"] = result->k1;  // Deprecated (set to 0)
    d["k2"] = result->k2;  // Deprecated (set to 0)

    // ✅ 【架构统一】转换合并后的labels
    py::list per_head_labels_list;
    for (const auto& labels : result->per_head_labels) {
        auto labels_tensor = torch::from_blob(
            const_cast<int32_t*>(labels.data()),
            {static_cast<int64_t>(labels.size())},
            torch::kInt32
        ).clone();
        per_head_labels_list.append(labels_tensor);
    }
    d["per_head_labels"] = per_head_labels_list;

    // ✅ 【架构统一】转换合并后的centroids（完整128维）
    py::list per_head_centroids_list;
    for (size_t head_idx = 0; head_idx < result->per_head_centroids.size(); head_idx++) {
        const auto& centroids = result->per_head_centroids[head_idx];
        int n_clusters = result->per_head_n_clusters[head_idx];
        auto centroids_tensor = torch::from_blob(
            const_cast<float*>(centroids.data()),
            {n_clusters, result->dim},
            torch::kFloat32
        ).clone();
        per_head_centroids_list.append(centroids_tensor);
    }
    d["per_head_centroids"] = per_head_centroids_list;

    d["per_head_n_clusters"] = result->per_head_n_clusters;

    return d;
}

// ════════════════════════════════════════════════════════════════════════
// Helper function to convert ReorganizeResult to Python dict
// ════════════════════════════════════════════════════════════════════════

py::dict reorganize_result_to_dict(const ReorganizeResult& result) {
    py::dict d;
    d["reorganized_keys"] = result.reorganized_keys;
    d["reorganized_values"] = result.reorganized_values;
    d["cluster_offsets"] = result.cluster_offsets;
    d["cluster_sizes"] = result.cluster_sizes;
    return d;
}

py::dict metadata_result_to_dict(LayerPipeline& pipeline) {
    const auto& reorganized = pipeline.get_reorganize_results();
    const KmeansResult* kmeans = pipeline.get_kmeans_result();

    py::dict d;
    if (kmeans == nullptr) {
        return d;
    }

    const int64_t kv_heads = static_cast<int64_t>(reorganized.size());
    int64_t max_clusters = 0;
    for (const auto& result : reorganized) {
        max_clusters = std::max<int64_t>(
            max_clusters,
            static_cast<int64_t>(result.cluster_sizes.size())
        );
    }

    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    auto head_indices = torch::empty({kv_heads}, long_opts);
    auto cluster_counts = torch::zeros({kv_heads}, int_opts);
    auto cluster_sizes = torch::zeros({kv_heads, max_clusters}, int_opts);
    auto cluster_offsets = torch::zeros({kv_heads, max_clusters + 1}, int_opts);
    auto centroids = torch::zeros({kv_heads, max_clusters, kmeans->dim}, float_opts);

    auto* head_indices_ptr = head_indices.data_ptr<int64_t>();
    auto* cluster_counts_ptr = cluster_counts.data_ptr<int32_t>();
    auto* cluster_sizes_ptr = cluster_sizes.data_ptr<int32_t>();
    auto* cluster_offsets_ptr = cluster_offsets.data_ptr<int32_t>();
    auto* centroids_ptr = centroids.data_ptr<float>();

    for (int64_t head = 0; head < kv_heads; ++head) {
        const auto& reorg = reorganized[head];
        const auto& centroid_values = kmeans->per_head_centroids[head];
        const int64_t cluster_count = static_cast<int64_t>(reorg.cluster_sizes.size());
        head_indices_ptr[head] = head;
        cluster_counts_ptr[head] = static_cast<int32_t>(cluster_count);

        if (cluster_count > 0) {
            std::memcpy(
                cluster_sizes_ptr + head * max_clusters,
                reorg.cluster_sizes.data(),
                static_cast<size_t>(cluster_count) * sizeof(int32_t)
            );
            std::memcpy(
                cluster_offsets_ptr + head * (max_clusters + 1),
                reorg.cluster_offsets.data(),
                static_cast<size_t>(cluster_count + 1) * sizeof(int32_t)
            );
            std::memcpy(
                centroids_ptr + head * max_clusters * kmeans->dim,
                centroid_values.data(),
                static_cast<size_t>(cluster_count * kmeans->dim) * sizeof(float)
            );
            const int32_t last_offset = reorg.cluster_offsets[cluster_count];
            for (int64_t idx = cluster_count + 1; idx < max_clusters + 1; ++idx) {
                cluster_offsets_ptr[head * (max_clusters + 1) + idx] = last_offset;
            }
        }
    }

    d["head_indices"] = head_indices;
    d["cluster_counts"] = cluster_counts;
    d["cluster_sizes"] = cluster_sizes;
    d["cluster_offsets"] = cluster_offsets;
    d["centroids"] = centroids;
    return d;
}

py::dict pipeline_stats_to_dict(const PipelineStats& stats) {
    py::dict d;
    d["d2h_ms"] = stats.d2h_ms;
    d["cpu_copy_ms"] = stats.cpu_copy_ms;
    d["kmeans_ms"] = stats.kmeans_ms;
    d["kmeans_cpu_time_ms"] = stats.kmeans_cpu_time_ms;
    d["kmeans_cpu_util_cores"] = stats.kmeans_cpu_util_cores;
    d["reorganize_ms"] = stats.reorganize_ms;
    d["write_ms"] = stats.write_ms;
    d["total_chunks"] = stats.total_chunks;
    d["total_tokens"] = stats.total_tokens;
    d["kmeans_gate_wait_ms"] = stats.kmeans_gate_wait_ms;
    return d;
}

// ════════════════════════════════════════════════════════════════════════
// pybind11 module definition
// ════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(ultra_layer_pipeline_cpp, m) {
    m.doc() = "Ultra-fast end-to-end layer pipeline for IMI cache (32-layer parallel)";

    // ════════════════════════════════════════════════════════════════════════
    // PipelineStage enum
    // ════════════════════════════════════════════════════════════════════════
    py::enum_<PipelineStage>(m, "PipelineStage")
        .value("Idle", PipelineStage::Idle)
        .value("Transferring", PipelineStage::Transferring)
        .value("Clustering", PipelineStage::Clustering)
        .value("Reorganizing", PipelineStage::Reorganizing)
        .value("Writing", PipelineStage::Writing)
        .value("Complete", PipelineStage::Complete)
        .value("Failed", PipelineStage::Failed)
        .export_values();

    // ════════════════════════════════════════════════════════════════════════
    // LayerPipeline class
    // ════════════════════════════════════════════════════════════════════════
    py::class_<LayerPipeline>(m, "LayerPipeline")
        .def(py::init([](int layer_idx,
                         int kv_heads,
                         int max_tokens,
                         int dim,
                         int device_id,
                         py::object kv_dtype_obj,
                         void* cache_manager_ptr,
                         bool enable_direct_write,
                         int subspace_parts,
                         py::object runtime_config_obj) {
                torch::Dtype kv_dtype = torch::kFloat16;
                if (!kv_dtype_obj.is_none()) {
                    kv_dtype = py_dtype_to_cpp(kv_dtype_obj);
                }
                LayerPipelineRuntimeConfig runtime_config = runtime_config_from_python(runtime_config_obj);
                return new LayerPipeline(
                    layer_idx,
                    kv_heads,
                    max_tokens,
                    dim,
                    device_id,
                    kv_dtype,
                    cache_manager_ptr,
                    enable_direct_write,
                    subspace_parts,
                    runtime_config
                );
            }),
             py::arg("layer_idx"),
             py::arg("kv_heads"),
             py::arg("max_tokens"),
             py::arg("dim"),
             py::arg("device_id"),
             py::arg("kv_dtype") = py::none(),
             py::arg("cache_manager_ptr") = nullptr,
             py::arg("enable_direct_write") = true,
             py::arg("subspace_parts") = 2,
             py::arg("runtime_config") = py::none(),
             R"pbdoc(
                Create a LayerPipeline instance.

                Args:
                    layer_idx: Layer index (0-31)
                    kv_heads: Number of KV heads (typically 8)
                    max_tokens: Maximum token capacity
                    dim: Hidden dimension per head (typically 128)
                    device_id: CUDA device ID
                    kv_dtype: torch dtype for KV tensors (torch.float16 or torch.bfloat16)
                    cache_manager_ptr: Optional pointer to cache manager (not used yet)
                    enable_direct_write: Enable C++ direct CPU buffer write (default: True)
                    subspace_parts: Number of subspaces for K-means (2 or 4)

                Returns:
                    LayerPipeline instance with allocated pinned memory and CUDA resources
             )pbdoc")

        // ════════════════════════════════════════════════════════════════════
        // Chunk Pipeline API
        // ════════════════════════════════════════════════════════════════════
        .def("start_chunk_pipeline",
            [](LayerPipeline& self, int total_middle_tokens, int chunk_size) {
                py::gil_scoped_release release;
                return self.start_chunk_pipeline(total_middle_tokens, chunk_size);
            },
            py::arg("total_middle_tokens"),
            py::arg("chunk_size"),
            R"pbdoc(
                Initialize chunk pipeline (must be called before submitting chunks).

                Args:
                    total_middle_tokens: Total middle tokens (for pre-allocating buffer)
                    chunk_size: Python-side chunk_size (for capacity validation)

                Note:
                    k1, k2 are auto-calculated by C++ based on actual chunk_len

                Returns:
                    True if successful
             )pbdoc")


        .def("submit_chunk",
            [](LayerPipeline& self,
               torch::Tensor keys,
               torch::Tensor values,
               int chunk_id,
               int start_token,
               bool is_last) {
                // ✅ Release GIL for true parallelism
                py::gil_scoped_release release;
                return self.submit_chunk(keys, values, chunk_id, start_token, is_last);
            },
            py::arg("keys"),
            py::arg("values"),
            py::arg("chunk_id"),
            py::arg("start_token"),
            py::arg("is_last"),
            R"pbdoc(
                Submit a single chunk (returns after GPU→CPU transfer is queued).

                Args:
                    keys: [1, kv_heads, chunk_len, dim] GPU tensor (torch.float16 or torch.bfloat16)
                    values: [1, kv_heads, chunk_len, dim] GPU tensor (same dtype as keys)
                    chunk_id: Chunk ID (0-based, must be consecutive)
                    start_token: Chunk start position in middle region (global offset)
                    is_last: Whether this is the last chunk

                Returns:
                    True if successful, False if failed (e.g., chunk too large)

                Note:
                    - GPU→CPU transfer runs on independent transfer_stream_
                    - Transfer may still be in-flight when this call returns
                     - Python side does NOT need to keep references to GPU tensors
                     - C++ chunk worker will wait for transfer to complete
             )pbdoc")

        .def("set_worker_threads",
            [](LayerPipeline& self, int32_t worker_threads) {
                self.set_worker_threads(worker_threads);
            },
            py::arg("worker_threads"),
            R"pbdoc(
                Update worker thread budget for CPU stages (kmeans/copy).

                Args:
                    worker_threads: desired thread count (>=1)
             )pbdoc")


        .def("cancel_pipeline",
            [](LayerPipeline& self) {
                py::gil_scoped_release release;
                self.cancel_pipeline();
            },
            R"pbdoc(
                Cancel chunk pipeline (stop all workers).
             )pbdoc")

        // ════════════════════════════════════════════════════════════════════
        // Wait API
        // ════════════════════════════════════════════════════════════════════

        // ════════════════════════════════════════════════════════════════════
        // Wait for completion
        // ════════════════════════════════════════════════════════════════════
        .def("wait_ready",
            [](LayerPipeline& self, float timeout_sec) {
                // ✅ Release GIL for true parallelism
                py::gil_scoped_release release;

                return self.wait_ready(timeout_sec);
            },
            py::arg("timeout_sec") = 60.0f,
            R"pbdoc(
                Wait for pipeline to complete (blocking, GIL-released).

                Args:
                    timeout_sec: Maximum wait time in seconds (default: 60.0)

                Returns:
                    True if completed successfully, False if timeout or failed

                Note:
                    This function releases the GIL, allowing other Python threads
                    to run while waiting. Use this for true 32-layer parallelism.
            )pbdoc")
       

        // ════════════════════════════════════════════════════════════════════
        // Query status
        // ════════════════════════════════════════════════════════════════════
        .def("current_stage", &LayerPipeline::current_stage,
             R"pbdoc(
                Get current pipeline stage (non-blocking).

                Returns:
                    PipelineStage enum value
             )pbdoc")

        .def("error_message", &LayerPipeline::error_message,
             R"pbdoc(
                Get error message if pipeline failed.

                Returns:
                    Error message string (empty if no error)
             )pbdoc")

        // ════════════════════════════════════════════════════════════════════
        // Get results (Python-friendly format)
        // ════════════════════════════════════════════════════════════════════
        .def("get_kmeans_result",
            [](LayerPipeline& self) {
                return kmeans_result_to_dict(self.get_kmeans_result());
            },
            R"pbdoc(
                Get K-means clustering results (call after wait_ready()).

                Returns:
                    Dictionary with keys:
                      - 'per_head_labels': List[torch.Tensor] (int32, shape [n_tokens])
                      - 'per_head_centroids': List[torch.Tensor] (float32, shape [n_clusters, dim])
                      - 'per_head_cluster_offsets': List[List[int]] (CSR format)
                      - 'per_head_n_clusters': List[int]
            )pbdoc")

        .def("get_metadata_result",
            [](LayerPipeline& self) {
                return metadata_result_to_dict(self);
            },
            R"pbdoc(
                Get compact tensor metadata result after wait_ready().

                Returns:
                    Dictionary with CPU tensors:
                      - 'head_indices': int64 [kv_heads]
                      - 'cluster_counts': int32 [kv_heads]
                      - 'cluster_sizes': int32 [kv_heads, max_clusters]
                      - 'cluster_offsets': int32 [kv_heads, max_clusters + 1]
                      - 'centroids': float32 [kv_heads, max_clusters, dim]
            )pbdoc")

        .def("get_pipeline_stats",
            [](LayerPipeline& self) {
                return pipeline_stats_to_dict(self.get_pipeline_stats());
            },
            R"pbdoc(
                Get pipeline timing stats (non-blocking).

                Returns:
                    Dictionary with keys:
                      - d2h_ms
                      - cpu_copy_ms
                      - kmeans_ms
                      - reorganize_ms
                      - write_ms
                      - total_chunks
                      - total_tokens
            )pbdoc")

        .def("get_reorganize_results",
            [](LayerPipeline& self) {
                py::list results_list;
                for (const auto& result : self.get_reorganize_results()) {
                    results_list.append(reorganize_result_to_dict(result));
                }
                return results_list;
            },
            R"pbdoc(
                Get reorganization results (call after wait_ready()).

                Returns:
                    List of dictionaries (one per head), each with keys:
                      - 'reorganized_keys': torch.Tensor (same dtype as kv_dtype, shape [n_tokens, dim])
                      - 'reorganized_values': torch.Tensor (same dtype as kv_dtype, shape [n_tokens, dim])
                      - 'cluster_offsets': List[int] (CSR format)
                      - 'cluster_sizes': List[int]
            )pbdoc")

        .def("get_keys_cpu", &LayerPipeline::get_keys_cpu,
             R"pbdoc(
                Get original CPU keys (after transfer complete).

                Returns:
                    torch.Tensor [kv_heads, n_tokens, dim] CPU tensor (torch.float16/bfloat16)
             )pbdoc")

        .def("get_values_cpu", &LayerPipeline::get_values_cpu,
             R"pbdoc(
                Get original CPU values (after transfer complete).

                Returns:
                    torch.Tensor [kv_heads, n_tokens, dim] CPU tensor (torch.float16/bfloat16)
             )pbdoc")

        // ════════════════════════════════════════════════════════════════════
        // ✅ 【GIL优化】Set batch callbacks (8 heads in one call)
        // ════════════════════════════════════════════════════════════════════
        .def("set_batch_allocate_cpu_buffer_callback",
            [](LayerPipeline& self, py::object callback) {
                self.set_batch_allocate_cpu_buffer_callback(
                    [callback](int32_t layer_idx,
                              const std::vector<std::pair<int32_t, int32_t>>& heads_info)
                              -> std::vector<CPUBufferDescriptor> {
                        py::gil_scoped_acquire acquire;

                        try {
                            // 转换heads_info为Python list
                            py::list py_heads_info;
                            for (const auto& [head_idx, required_tokens] : heads_info) {
                                py_heads_info.append(py::make_tuple(head_idx, required_tokens));
                            }

                            // Call Python: callback(layer_idx, heads_info)
                            py::list result = callback(layer_idx, py_heads_info).cast<py::list>();

                            // 转换返回的list[dict]为vector<CPUBufferDescriptor>
                            std::vector<CPUBufferDescriptor> descriptors;
                            descriptors.reserve(result.size());

                            for (size_t i = 0; i < result.size(); ++i) {
                                py::dict dict_item = result[i].cast<py::dict>();

                                CPUBufferDescriptor desc;
                                desc.layer_idx = layer_idx;
                                desc.head_idx = i;

                                torch::Tensor key_buffer = dict_item["key_buffer"].cast<torch::Tensor>();
                                torch::Tensor value_buffer = dict_item["value_buffer"].cast<torch::Tensor>();

                                desc.key_buffer_ptr = key_buffer.data_ptr();
                                desc.value_buffer_ptr = value_buffer.data_ptr();
                                desc.buffer_capacity = dict_item["buffer_capacity"].cast<size_t>();

                                descriptors.push_back(desc);
                            }

                            return descriptors;

                        } catch (const std::exception& e) {
                            std::cerr << "❌ [BatchAllocateCallback] Layer " << layer_idx
                                      << ": " << e.what() << "\n";
                            return std::vector<CPUBufferDescriptor>();
                        }
                    }
                );
            },
            py::arg("callback"),
            R"pbdoc(
                Set batch callback for allocating CPU buffers (8 heads in one call).

                Args:
                    callback: Python function with signature:
                              (layer_idx: int, heads_info: List[Tuple[int, int]]) -> List[dict]

                              heads_info: [(head_idx, required_tokens), ...]
                              Return: List of dicts, each containing:
                                - 'key_buffer': torch.Tensor
                                - 'value_buffer': torch.Tensor
                                - 'buffer_capacity': int

                Example:
                    def batch_allocate(layer_idx, heads_info):
                        results = []
                        for head_idx, required_tokens in heads_info:
                            capacity = max(required_tokens, 131072)
                            key_buf = torch.zeros(capacity, 128, dtype=torch.float16)
                            val_buf = torch.zeros(capacity, 128, dtype=torch.float16)
                            results.append({
                                'key_buffer': key_buf,
                                'value_buffer': val_buf,
                                'buffer_capacity': capacity
                            })
                        return results

                    pipeline.set_batch_allocate_cpu_buffer_callback(batch_allocate)
            )pbdoc")

        .def("set_batch_register_from_cpp_callback",
            [](LayerPipeline& self, py::object callback) {
                self.set_batch_register_from_cpp_callback(
                    [callback](int32_t layer_idx, const std::vector<HeadMetadata>& all_heads_data) -> bool {
                        py::gil_scoped_acquire acquire;

                        try {
                            // 转换HeadMetadata为Python list
                            py::list py_heads_data;
                            for (const auto& head_data : all_heads_data) {
                                py::tuple item = py::make_tuple(
                                    head_data.head_idx,
                                    head_data.cluster_sizes,
                                    head_data.cluster_offsets,
                                    head_data.centroids
                                );
                                py_heads_data.append(item);
                            }

                            // Call Python: callback(layer_idx, all_heads_data) -> bool
                            bool result = callback(layer_idx, py_heads_data).cast<bool>();
                            return result;

                        } catch (const std::exception& e) {
                            std::cerr << "❌ [BatchRegisterCallback] Layer " << layer_idx
                                      << ": " << e.what() << "\n";
                            return false;
                        }
                    }
                );
            },
            py::arg("callback"),
            R"pbdoc(
                Set batch callback for metadata + centroids registration (8 heads in one call).

                Args:
                    callback: Python function with signature:
                              (layer_idx: int, all_heads_data: List[Tuple]) -> bool

                              all_heads_data: List of (head_idx, cluster_sizes, cluster_offsets, centroids)
                              Return: bool (success/failure)

                Example:
                    def batch_register(layer_idx, all_heads_data):
                        for head_idx, cluster_sizes, cluster_offsets, centroids in all_heads_data:
                            # Register metadata for this head
                            cache_manager.register_metadata(layer_idx, head_idx, ...)
                        return True

                    pipeline.set_batch_register_from_cpp_callback(batch_register)
            )pbdoc");



}
