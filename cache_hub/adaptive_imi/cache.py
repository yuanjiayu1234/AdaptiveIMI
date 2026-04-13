from typing import Dict, List, Optional, Tuple
import json
import math
import os
import queue
import threading
import time

import torch
from library.AdaptiveIMI.cpp_extensions import AdpIMI_ThreadPool, AdpIMI_Index
from library.AdaptiveIMI.cpp_extensions import gather_copy_and_concat, gather_copy_and_concat_retrieval, gather_copy_and_scatter
from library.AdaptiveIMI.imi_adapter import IMIPipeline, IMIRuntimeConfig, get_imi_kernels
from ..base import KV_Cache
from .async_update import AdaptiveIMIAsyncUpdateMixin
from .indexing import AdaptiveIMIIndexingMixin, _KmeansScheduler
from .prefetch import AdaptiveIMIPrefetchMixin
from .retrieval import AdaptiveIMIRetrievalMixin
from .runtime import AdaptiveIMIRuntimeMixin


imi_gpu_kernels = get_imi_kernels()


def _get_available_cpu_cores() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def _read_int_env(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


class adpimi_cache(AdaptiveIMIIndexingMixin, AdaptiveIMIPrefetchMixin, AdaptiveIMIAsyncUpdateMixin, AdaptiveIMIRetrievalMixin, AdaptiveIMIRuntimeMixin, KV_Cache):
    def __init__(
        self,
        valid_start,
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        max_new_length: int,
        input_length: Optional[int],
        static_pattern_start: int,
        static_pattern_end: int,
        core: int,
        pages_per_cluster: int,
        retrieval_budget: float,
        cache_ratio: float,
        buffer_cluster_num: int,
        prefill_bsz: int,
        num_gpus: int,
        model_size: int,
        subspace_parts: int = 4,
        runtime_config: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> None:
        super().__init__(
            layer_num,
            batch_size,
            max_length,
            num_key_value_heads,
            num_heads,
            head_dim,
            dtype,
            layer_mapping,
            prefill_bsz,
            num_gpus,
            model_size,
        )
        if batch_size != 1:
            raise ValueError("IMI cache currently supports batch_size=1.")

        self.device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(":")[-1]))
        self.valid_start_list = valid_start

        self.static_pattern_start = static_pattern_start
        self.static_pattern_end = static_pattern_end
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end

        self.group_size = self.num_heads // self.kv_head
        self.batch_groups = self.batch_size * self.kv_head

        self.page_size = 8
        self.pages_per_cluster = pages_per_cluster
        self.retrieval_budget = retrieval_budget
        self.cache_ratio = cache_ratio
        self.buffer_cluster_num = buffer_cluster_num
        if subspace_parts not in (0, 2, 4):
            raise ValueError("subspace_parts must be 0, 2 or 4")
        self.subspace_parts = subspace_parts
        self._available_cpu_cores = _get_available_cpu_cores()
        runtime_config = runtime_config or {}
        prefill_cfg = runtime_config.get("prefill") or {}
        if not isinstance(prefill_cfg, dict):
            prefill_cfg = {}
        if not prefill_cfg and isinstance(runtime_config.get("streaming"), dict):
            prefill_cfg = {"streaming": runtime_config.get("streaming")}

        kmeans_cfg = dict(runtime_config.get("kmeans") or {})
        per_layer_tasks = self.kv_head * max(self.subspace_parts, 1)
        min_threads_per_phase = max(1, self._available_cpu_cores // max(1, per_layer_tasks))
        env_min_threads = _read_int_env("IMI_MIN_THREADS_PER_PHASE")
        if env_min_threads is not None and env_min_threads > 0:
            min_threads_per_phase = env_min_threads
        pipeline_cfg = {
            "worker_threads": self._available_cpu_cores,
            "chunk_buffer_count": max(2, min(8, self._available_cpu_cores // 8)),
            "min_threads_per_phase": min_threads_per_phase,
            "enable_omp_nested": True,
            "omp_max_active_levels": 2,
        }

        self.runtime_config = {
            "cpu_threads": self._available_cpu_cores,
            "pipeline": pipeline_cfg,
            "kmeans": kmeans_cfg,
            "prefetch": dict(runtime_config.get("prefetch") or {}),
            "streaming": dict(prefill_cfg.get("streaming")or {}),
            "async_update": dict(runtime_config.get("async_update") or {}),
        }

        self.input_length = int(input_length) if input_length is not None else self.max_length - max_new_length
        actual_gen_len = max_new_length - 1
        self.actual_gen_len = actual_gen_len
        self.capacity_input_length = int(self.max_length - max_new_length)
        self.capacity_actual_gen_len = actual_gen_len
        if self.input_length <= 0:
            raise ValueError(f"input length({self.input_length}) should be larger than 0")
        if self.capacity_input_length <= 0:
            raise ValueError(f"capacity input length({self.capacity_input_length}) should be larger than 0")

        self.list_capacity = max(self.capacity_input_length - self.static_pattern_total, 0)
        self.list_stride = max(self.input_length - self.static_pattern_total, 0)
        self.build_index_when_prefilling = self.list_capacity > 0
        if self.build_index_when_prefilling:
            self.static_capacity = self.static_pattern_total + self.capacity_actual_gen_len
            self.static_stride = self.static_pattern_total + actual_gen_len
        else:
            self.static_capacity = self.capacity_input_length + self.capacity_actual_gen_len
            self.static_stride = self.input_length + actual_gen_len
        self.will_update_index = False

        self.adpimi_index_cls = AdpIMI_Index
        self.thread_pool = AdpIMI_ThreadPool(core)
        self.core = core
        self.attn_func = self.dense_attention

        self.n_centroids = 0
        self.nprobe = 0
        self.cache_size = 0
        self.buffer_size = 0
        self.execution_stride = 0
        self.retrieval_execution_stride = 0
        self.max_retrieval_tokens = 0
        self.max_retrieval_pages = 0
        self.cache_budget_pages = 0
        self.buffer_budget_pages = 0

        self.layer_metadata: List[List[Dict[str, object]] | None] = [None] * self.layer_num
        self.layer_ready = [False] * self.layer_num
        self.layer_started = [False] * self.layer_num
        self.cache_prepared = False
        self._index_summary_logged = False
        self._expected_middle_len: List[Optional[int]] = [None] * self.layer_num

        self.steady_zone_keys = []
        self.steady_zone_values = []
        for ldx in range(self.layer_num):
            self.steady_zone_keys.append(
                torch.zeros(
                    (self.batch_size, self.kv_head, self.static_capacity, self.head_dim),
                    dtype=self.dtype,
                    device=self.layer_mapping[str(ldx)],
                ).contiguous()
            )
            self.steady_zone_values.append(
                torch.zeros(
                    (self.batch_size, self.kv_head, self.static_capacity, self.head_dim),
                    dtype=self.dtype,
                    device=self.layer_mapping[str(ldx)],
                ).contiguous()
            )

        self.list_keys = [None] * self.layer_num
        self.list_values = [None] * self.layer_num

        self.imi_pipelines = []
        if self.build_index_when_prefilling:
            enable_direct_write = os.getenv("IMI_DISABLE_DIRECT_WRITE", "0") != "1"
            for ldx in range(self.layer_num):
                device_id = int(self.layer_mapping[str(ldx)].split(":")[-1])
                self.imi_pipelines.append(
                    IMIPipeline(
                        layer_idx=ldx,
                        kv_heads=self.kv_head,
                        head_dim=self.head_dim,
                        dtype=self.dtype,
                        device_id=device_id,
                        max_tokens=self.list_capacity,
                        subspace_parts=self.subspace_parts,
                        enable_direct_write=enable_direct_write,
                        runtime_config=IMIRuntimeConfig(
                            pipeline=self.runtime_config["pipeline"],
                            kmeans=self.runtime_config["kmeans"],
                        ),
                    )
                )
            if enable_direct_write:
                for ldx, pipeline in enumerate(self.imi_pipelines):
                    def batch_allocate(layer_idx, heads_info, ldx=ldx):
                        if layer_idx != ldx:
                            raise ValueError(
                                f"IMI direct write layer mismatch: expected {ldx}, got {layer_idx}"
                            )
                        self._ensure_list_storage(ldx)
                        key_storage = self.list_keys[ldx]
                        value_storage = self.list_values[ldx]
                        results = []
                        for head_idx, required_tokens in heads_info:
                            key_buffer = key_storage[0, head_idx]
                            value_buffer = value_storage[0, head_idx]
                            capacity = int(key_buffer.size(0))
                            if required_tokens > capacity:
                                raise ValueError(
                                    f"IMI direct write buffer too small: {required_tokens} > {capacity}"
                                )
                            results.append({
                                "key_buffer": key_buffer,
                                "value_buffer": value_buffer,
                                "buffer_capacity": capacity,
                            })
                        return results

                    pipeline.set_batch_allocate_cpu_buffer_callback(batch_allocate)
        else:
            self.imi_pipelines = [None] * self.layer_num

        self.adpimi_index = [None] * self.layer_num
        self.cluster_sizes_cpu = [None] * self.layer_num
        self.cluster_offsets_cpu = [None] * self.layer_num
        self.cluster_sizes_gpu = [None] * self.layer_num
        self.centroids = [None] * self.layer_num
        self.centroids_mask = [None] * self.layer_num
        self.centroids_cpu = [None] * self.layer_num
        self.centroids_mask_cpu = [None] * self.layer_num

        self.cache_keys = []
        self.cache_values = []

        self.hit_unit_idices = []
        self.hit_unit_sizes = []
        self.hit_unit_sizes_cumsum = []
        self.hit_num_units = []
        self.miss_unit_idices = []
        self.miss_unit_sizes = []
        self.miss_unit_sizes_cumsum = []
        self.miss_num_units = []
        self.update_buffer_indices = []
        self.update_unit_sizes = []
        self.update_cache_indices = []
        self.update_num_units = []
        self.cluster_ids = []

        self.execution_buffer_keys_dict: Dict[str, torch.Tensor] = {}
        self.execution_buffer_values_dict: Dict[str, torch.Tensor] = {}
        self.valid_lengths_dict: Dict[str, torch.Tensor] = {}
        self.static_len_tensor_dict: Dict[str, torch.Tensor] = {}
        self.static_lengths_dict: Dict[str, torch.Tensor] = {}
        self.similarity_buffer_dict: Dict[str, torch.Tensor] = {}

        self.prefill_chunk_size = max(int(self.runtime_config["streaming"].get("prefill_chunk_size", 32768)), 1)
        self.profile_decode = os.getenv("IMI_PROFILE_DECODE", "0") == "1"
        self.profile_prefill = os.getenv("IMI_PROFILE_PREFILL", "0") == "1"
        self.profile_decode_steps = int(os.getenv("IMI_PROFILE_STEPS", "5"))
        self.profile_decode_count = 0
        self.profile_decode_step_idx = -1
        self.decode_breakdown: Optional[dict] = None
        self.decode_breakdown_accum: dict = {}
        self.prefill_breakdown_rows: List[Dict[str, object]] = []
        self.prefill_gpu_layer_rows: List[Dict[str, object]] = []
        self.profile_block_hit_rate = os.getenv("IMI_PROFILE_HIT_RATE", "0") == "1"
        self.profile_block_hit_every = int(os.getenv("IMI_PROFILE_HIT_EVERY", "1"))
        if self.profile_block_hit_every <= 0:
            self.profile_block_hit_every = 1
        self.profile_block_step = 0
        self.profile_block_hits = 0
        self.profile_block_misses = 0
        self.profile_layer_hit_rate_count = 0
        self.profile_cache_stats = os.getenv("IMI_PROFILE_CACHE", "0") == "1"
        self.decode_debug_step = 0

        async_cfg = self.runtime_config.get("async_update") or {}
        self.async_update_enabled = bool(async_cfg.get("enabled", False))
        self.async_update_batch = max(int(async_cfg.get("batch", 16)), 1)
        self.async_update_threshold = max(int(async_cfg.get("threshold", 16)), 1)
        self.async_update_streams: Dict[str, torch.cuda.Stream] = {}
        self.async_update_buffers: List[Optional[Dict[str, object]]] = [None] * self.layer_num
        self.async_update_delta_tokens = max(int(async_cfg.get("delta_tokens", 512)), 1)
        self.centroids_shadow: List[Optional[torch.Tensor]] = [None] * self.layer_num
        self.centroid_counts: List[Optional[torch.Tensor]] = [None] * self.layer_num
        self.delta_counts: List[Optional[torch.Tensor]] = [None] * self.layer_num
        self.delta_sums: List[Optional[torch.Tensor]] = [None] * self.layer_num
        self.delta_keys: List[Optional[torch.Tensor]] = [None] * self.layer_num
        self.delta_values: List[Optional[torch.Tensor]] = [None] * self.layer_num
        self.delta_cluster_ids: List[Optional[torch.Tensor]] = [None] * self.layer_num
        self.delta_counts_gpu: List[Optional[torch.Tensor]] = [None] * self.layer_num

        prefetch_cfg = (self.runtime_config.get("prefetch") or {}) if self.runtime_config is not None else {}
        self.prefetch_enabled = bool(prefetch_cfg.get("enabled", True))
        self.prefetch_enabled_default = self.prefetch_enabled
        self.prefetch_ratio = float(prefetch_cfg.get("ratio", 0.1))
        self.prefetch_mode = str(prefetch_cfg.get("mode", "random")).lower()
        self.prefetch_seed = int(prefetch_cfg.get("seed", 0))
        self.prefetch_tiles_per_layer = max(int(prefetch_cfg.get("tiles_per_layer", 1)), 1)
        self.prefetch_k = 0
        self.prefetch_done = False
        print(
            "[IMI prefetch] enabled={} ratio={} mode={} seed={} tiles_per_layer={}".format(
                self.prefetch_enabled,
                self.prefetch_ratio,
                self.prefetch_mode,
                self.prefetch_seed,
                self.prefetch_tiles_per_layer,
            )
        )

        self.allocated = self.pre_allocate_decision()
      

        self.prefill_stream_mode = [False] * self.layer_num
        self.prefill_stream_started = [False] * self.layer_num
        self.prefill_stream_chunk_ids = [0] * self.layer_num
        self.prefill_stream_end_submitted = [False] * self.layer_num

        self.prefill_stream_queue_depth = max(int(self.runtime_config["streaming"].get("prefill_queue_depth", 8)), 1)
        self.prefill_stream_stage_slots = max(int(self.runtime_config["streaming"].get("prefill_stage_slots", 8)), 1)

        self.prefill_stream_copy_streams = [None] * self.layer_num
        self.prefill_stream_main_events = [None] * self.layer_num
        self.prefill_stream_copy_events = [None] * self.layer_num
        self.prefill_stream_queues = [None] * self.layer_num
        self.prefill_stream_workers = [None] * self.layer_num
        self.prefill_stream_stop_events = [threading.Event() for _ in range(self.layer_num)]
        self.prefill_stream_done_events = [threading.Event() for _ in range(self.layer_num)]
        self.prefill_stream_errors = [None] * self.layer_num
        self.prefill_stream_expected_chunk_id = [0] * self.layer_num
        self.prefill_stream_expected_token_offset = [0] * self.layer_num
        self.prefill_stream_produced_token_offset = [0] * self.layer_num
        self.prefill_stream_last_chunk_seen = [False] * self.layer_num
        self.prefill_stream_free_slots = [None] * self.layer_num
        self.prefill_stream_stage_keys = [None] * self.layer_num
        self.prefill_stream_stage_values = [None] * self.layer_num

        self._kmeans_job_seq = 0
        self._kmeans_job_lock = threading.Lock()
        self._kmeans_min_threads_per_layer = max(1, self.kv_head * max(self.subspace_parts, 1))
        max_concurrent = _read_int_env("IMI_KMEANS_MAX_CONCURRENT")
        if max_concurrent is None and self.list_capacity >= 65536:
            max_concurrent = 1
        self._kmeans_scheduler = _KmeansScheduler(
            self._available_cpu_cores,
            self._kmeans_min_threads_per_layer,
            max_concurrent=max_concurrent,
        )
        self._metadata_lock = threading.Lock()

    def can_reuse_for_next_sequence(self, valid_start, max_new_length: int, input_length: int) -> bool:
        new_input_length = int(input_length)
        new_actual_gen_len = int(max_new_length) - 1
        new_list_stride = max(new_input_length - self.static_pattern_total, 0)
        if new_list_stride > self.list_capacity:
            return False
        if self.build_index_when_prefilling:
            new_static_stride = self.static_pattern_total + new_actual_gen_len
        else:
            new_static_stride = new_input_length + new_actual_gen_len
        if new_static_stride > self.static_capacity:
            return False
        if len(valid_start) != len(self.valid_start_list):
            return False
        return True

    def reset_for_next_sequence(self, valid_start, max_new_length: int, input_length: int) -> None:
        if not self.can_reuse_for_next_sequence(valid_start, max_new_length, input_length):
            raise ValueError('AdaptiveIMI cache shape mismatch; cannot reuse across samples')

        self.valid_start_list = valid_start
        self.input_length = int(input_length)
        self.actual_gen_len = int(max_new_length) - 1
        self.list_stride = max(self.input_length - self.static_pattern_total, 0)
        if self.build_index_when_prefilling:
            self.static_stride = self.static_pattern_total + self.actual_gen_len
        else:
            self.static_stride = self.input_length + self.actual_gen_len
        self.context = 0
        self.attn_func = self.dense_attention

        self.layer_metadata = [None] * self.layer_num
        self.layer_ready = [False] * self.layer_num
        self.layer_started = [False] * self.layer_num
        self._index_summary_logged = False
        self._expected_middle_len = [None] * self.layer_num

        self.prefetch_enabled = self.prefetch_enabled_default
        self.prefetch_done = False
        self.prefetch_k = 0

        self.profile_decode_count = 0
        self.profile_decode_step_idx = -1
        self.decode_breakdown = None
        self.decode_breakdown_accum = {}
        self.prefill_breakdown_rows = []
        self.prefill_gpu_layer_rows = []
        self.profile_block_step = 0
        self.profile_block_hits = 0
        self.profile_block_misses = 0
        self.profile_layer_hit_rate_count = 0
        self.decode_debug_step = 0

        self.prefill_stream_mode = [False] * self.layer_num
        self.prefill_stream_started = [False] * self.layer_num
        self.prefill_stream_chunk_ids = [0] * self.layer_num
        self.prefill_stream_end_submitted = [False] * self.layer_num
        self.prefill_stream_errors = [None] * self.layer_num
        self.prefill_stream_expected_chunk_id = [0] * self.layer_num
        self.prefill_stream_expected_token_offset = [0] * self.layer_num
        self.prefill_stream_produced_token_offset = [0] * self.layer_num
        self.prefill_stream_last_chunk_seen = [False] * self.layer_num
        for done_event in self.prefill_stream_done_events:
            done_event.clear()

        for layer_idx in range(self.layer_num):
            if self.async_update_buffers[layer_idx] is not None:
                self.async_update_buffers[layer_idx]["count"] = 0
                self.async_update_buffers[layer_idx]["delta_write_pos"].zero_()
            if self.delta_counts[layer_idx] is not None:
                self.delta_counts[layer_idx].zero_()
            if self.delta_sums[layer_idx] is not None:
                self.delta_sums[layer_idx].zero_()
            if self.delta_counts_gpu[layer_idx] is not None:
                self.delta_counts_gpu[layer_idx].zero_()

    def _estimate_cache_parameters(self) -> Tuple[int, int, int, int]:
        if not self.build_index_when_prefilling:
            return 0, 0, 0, 0

        n_tokens = max(self.list_stride, 1)
        total_clusters = max(int(n_tokens / 16), 1)
        if self.subspace_parts == 4:
            total_clusters_f = max(n_tokens / 16.0, 1.0)
            k = max(int(round(total_clusters_f ** 0.25)), 2)
            n_centroids = k ** 4 + 1
        else:
            k = max(int(math.sqrt(total_clusters)), 32)
            n_centroids = k * k + 1

        nprobe = max(round((n_centroids - 1) * self.retrieval_budget), 1)
        nprobe = min(nprobe, n_centroids - 1)

        cache_cluster_num = (
            round(n_centroids * self.cache_ratio) if self.cache_ratio > 0.0 else nprobe * 3
        )
        requested_cache_pages = max(cache_cluster_num * self.pages_per_cluster, 1)
        cache_pages = self._resolve_budget_pages(requested_cache_pages)

        requested_buffer_pages = max(self.buffer_cluster_num, nprobe * 4) * self.pages_per_cluster
        buffer_pages = self._resolve_budget_pages(requested_buffer_pages)

        return n_centroids, nprobe, cache_pages, buffer_pages
    def pre_allocate_decision(self) -> bool:
        """Decide whether to pre-allocate GPU cache and metadata before prefilling."""
        if not self.build_index_when_prefilling:
            return True

        n_centroids, _, cache_pages, buffer_pages = self._estimate_cache_parameters()
        bytes_per_elem = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4

        cache_tokens = cache_pages * self.page_size + self.static_stride
        cache_bytes = (
            2
            * self.layer_num
            * self.batch_size
            * self.kv_head
            * cache_tokens
            * self.head_dim
            * bytes_per_elem
        )
        execution_tokens = buffer_pages * self.page_size + self.static_stride
        execution_bytes = (
            2
            * self.batch_size
            * self.kv_head
            * execution_tokens
            * self.head_dim
            * bytes_per_elem
        )

        centroid_bytes = self.layer_num * self.batch_groups * n_centroids * self.head_dim * 4
        mask_bytes = self.layer_num * self.batch_groups * n_centroids

        estimated_gb = (cache_bytes + execution_bytes + centroid_bytes + mask_bytes) / (1024 ** 3)
        self.esitimate_gpu_memory = estimated_gb
        return self.free_memory > estimated_gb * 1.5
    def _prepare_prefill_chunk(self, states: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return states[:, start:end, :, :].transpose(1, 2).contiguous()
