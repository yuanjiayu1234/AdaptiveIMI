from typing import Dict, List, Optional, Tuple
import os
import queue
import threading
import time

import torch
from library.AdaptiveIMI.imi_adapter import IMIPipeline, IMIRuntimeConfig, get_imi_kernels
from ..base import KV_Cache
from .async_update import AdaptiveIMIAsyncUpdateMixin
from cache_hub.cluster_plan import choose_learned_k_values, product_k_values
from .indexing import AdaptiveIMIIndexingMixin, _BuildJobScheduler
from .partition import build_sequential_hybrid_partition_specs, load_hybrid_partition_specs
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


def _coerce_bool(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off", ""):
            return False
    return default


def _load_learned_partition_specs(
    partition_path: Optional[str],
    *,
    layer_num: int,
    kv_heads: int,
    head_dim: int,
) -> List[Optional[Dict[str, object]]]:
    if not partition_path:
        return build_sequential_hybrid_partition_specs(
            layer_num=layer_num,
            kv_heads=kv_heads,
            head_dim=head_dim,
            required_m=8,
        )

    return load_hybrid_partition_specs(
        partition_path,
        layer_num=layer_num,
        kv_heads=kv_heads,
        head_dim=head_dim,
        required_m=8,
    )


class AdaptiveIMICacheBase(AdaptiveIMIIndexingMixin, AdaptiveIMIAsyncUpdateMixin, AdaptiveIMIRetrievalMixin, AdaptiveIMIRuntimeMixin, KV_Cache):
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
        subspace_parts: int = 8,
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
        if batch_size != 1 and not bool((runtime_config or {}).get("allow_batch_size_gt_one", False)):
            raise ValueError("IMI cache currently supports batch_size=1.")

        self.device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(":")[-1]))
        self.valid_start_list = valid_start

        self.static_pattern_start = static_pattern_start
        self.static_pattern_end = static_pattern_end
        self.base_static_pattern_total = self.static_pattern_start + self.static_pattern_end
        self.static_pattern_total = self.base_static_pattern_total

        self.group_size = self.num_heads // self.kv_head
        self.batch_groups = self.batch_size * self.kv_head

        self.page_size = 8
        self.pages_per_cluster = pages_per_cluster
        self.retrieval_budget = retrieval_budget
        self.cache_ratio = cache_ratio
        self.buffer_cluster_num = buffer_cluster_num
        self._available_cpu_cores = _get_available_cpu_cores()
        runtime_config = runtime_config or {}
        partition_path = runtime_config.get("partition_path")
        self.learned_partition_specs = _load_learned_partition_specs(
            str(partition_path) if partition_path else None,
            layer_num=self.layer_num,
            kv_heads=self.kv_head,
            head_dim=self.head_dim,
        )
        self.learned_partition_enabled = any(spec is not None for spec in self.learned_partition_specs)
        if not self.learned_partition_enabled:
            raise ValueError("AdaptiveIMI base now serves hybrid_8way_v2 only; partition_path is required.")
        self.partition_layout = (
            "sequential_rope_pairs"
            if not partition_path
            else "learned"
        )
        if self.partition_layout == "sequential_rope_pairs":
            print(
                "[AdaptiveIMI] partition_path is empty; using sequential 8-way RoPE-pair partition.",
                flush=True,
            )
        self.subspace_parts = max(
            int(spec.get("subspace_parts", 1))
            for spec in self.learned_partition_specs
            if spec is not None
        )
        prefill_cfg = runtime_config.get("prefill") or {}
        if not isinstance(prefill_cfg, dict):
            prefill_cfg = {}
        streaming_cfg = {}
        if isinstance(runtime_config.get("streaming"), dict):
            streaming_cfg.update(runtime_config.get("streaming") or {})
        if isinstance(prefill_cfg.get("streaming"), dict):
            streaming_cfg.update(prefill_cfg.get("streaming") or {})
        if prefill_cfg.get("prefill_chunk_size") is not None:
            streaming_cfg["prefill_chunk_size"] = prefill_cfg.get("prefill_chunk_size")

        kmeans_cfg = dict(runtime_config.get("kmeans") or {})
        explicit_k_values = kmeans_cfg.get("k_values")
        if explicit_k_values is not None:
            explicit_k_values = [int(value) for value in explicit_k_values]
            if any(value <= 0 for value in explicit_k_values):
                raise ValueError("AdaptiveIMI kmeans.k_values entries must be positive")
        max_partition_m = max(
            [int(spec.get("subspace_parts", 0)) for spec in self.learned_partition_specs if spec is not None]
            or [0]
        )
        if explicit_k_values is not None and len(explicit_k_values) != max_partition_m:
            raise ValueError(
                f"AdaptiveIMI kmeans.k_values must contain {max_partition_m} entries, "
                f"got {len(explicit_k_values)}"
            )
        if explicit_k_values is not None:
            kmeans_cfg["k_values"] = explicit_k_values
        per_layer_tasks = self.kv_head * max(max_partition_m, 1)
        min_threads_per_phase = max(1, self._available_cpu_cores // max(1, per_layer_tasks))
        pipeline_cfg = {
            "worker_threads": self._available_cpu_cores,
            "chunk_buffer_count": max(2, min(8, self._available_cpu_cores // 8)),
            "min_threads_per_phase": min_threads_per_phase,
            "enable_omp_nested": True,
            "omp_max_active_levels": 2,
        }
        user_pipeline_cfg = runtime_config.get("pipeline")
        if isinstance(user_pipeline_cfg, dict):
            pipeline_cfg.update(user_pipeline_cfg)
        pipeline_cfg["enable_direct_write"] = _coerce_bool(pipeline_cfg.get("enable_direct_write"), default=True)
        if pipeline_cfg.get("kmeans_max_concurrent") is not None:
            pipeline_cfg["kmeans_max_concurrent"] = int(pipeline_cfg["kmeans_max_concurrent"])
        cache_cfg = dict(runtime_config.get("cache") or {})
        capacity_by_max_model_len = cache_cfg.get("capacity_by_max_model_len")
        cache_cfg["capacity_by_max_model_len"] = _coerce_bool(capacity_by_max_model_len, default=False)
        multibatch_total_pages = cache_cfg.get("multibatch_total_pages", cache_cfg.get("multibatch_cache_total_pages"))
        if multibatch_total_pages is None:
            multibatch_total_pages = 4096
        cache_cfg["multibatch_total_pages"] = max(int(multibatch_total_pages), 0)

        self.runtime_config = {
            "cpu_threads": self._available_cpu_cores,
            "pipeline": pipeline_cfg,
            "kmeans": kmeans_cfg,
            "streaming": streaming_cfg,
            "cache": cache_cfg,
            "async_update": dict(runtime_config.get("async_update") or {}),
            "partition_path": str(partition_path) if partition_path else None,
        }
        self.multibatch_cache_total_pages = int(cache_cfg["multibatch_total_pages"])
        self.capacity_by_max_model_len = bool(cache_cfg["capacity_by_max_model_len"])
        async_cfg = self.runtime_config.get("async_update") or {}
        self.async_update_enabled = bool(async_cfg.get("enabled", False))
        self.async_update_interval = max(int(async_cfg.get("interval", async_cfg.get("batch", 16))), 1)
        self.async_update_recent_tokens = max(
            int(async_cfg.get("recent_generated_tokens", 512)),
            0,
        )
        self.async_update_start_after_tokens = max(
            int(async_cfg.get("start_after_tokens", self.async_update_recent_tokens)),
            0,
        )
        self.async_update_recent_capacity = max(
            int(async_cfg.get("recent_capacity", self.async_update_recent_tokens + self.async_update_interval)),
            self.async_update_interval,
        )
        self.async_update_delta_nprobe = max(int(async_cfg.get("delta_nprobe", 8)), 0)

        self.input_length = int(input_length) if input_length is not None else self.max_length - max_new_length
        actual_gen_len = max_new_length - 1
        self.actual_gen_len = actual_gen_len
        if self.capacity_by_max_model_len:
            self.capacity_input_length = int(self.max_length - max_new_length)
        else:
            self.capacity_input_length = int(self.input_length)
        self.capacity_actual_gen_len = actual_gen_len
        if self.input_length <= 0:
            raise ValueError(f"input length({self.input_length}) should be larger than 0")
        if self.capacity_input_length <= 0:
            raise ValueError(f"capacity input length({self.capacity_input_length}) should be larger than 0")

        self.list_capacity = max(self.capacity_input_length - self.static_pattern_total, 0)
        self.list_stride = max(self.input_length - self.static_pattern_total, 0)
        self.build_index_when_prefilling = self.list_capacity > 0
        if not self.build_index_when_prefilling:
            self.async_update_enabled = False
        if self.async_update_enabled and self.build_index_when_prefilling:
            static_gen_capacity = min(self.capacity_actual_gen_len, self.async_update_recent_capacity)
        else:
            static_gen_capacity = self.capacity_actual_gen_len
        if self.build_index_when_prefilling:
            self.static_capacity = self.static_pattern_total + static_gen_capacity
            self.static_stride = self.static_capacity
        else:
            self.static_capacity = self.capacity_input_length + self.capacity_actual_gen_len
            self.static_stride = self.input_length + actual_gen_len
        self.will_update_index = False

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
        self.layer_index_stats: List[Dict[str, int] | None] = [None] * self.layer_num
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

        cache_cfg = dict(self.runtime_config.get("cache") or {})
        self.raw_d2h_pool_enabled = bool(cache_cfg.get("raw_d2h_pool_enabled", True))
        self.raw_d2h_pool_group_size = max(1, int(cache_cfg.get("raw_d2h_pool_group_size", 2)))

        self.list_keys = [None] * self.layer_num
        self.list_values = [None] * self.layer_num

        self._raw_d2h_key_slabs = [None] * ((self.layer_num + self.raw_d2h_pool_group_size - 1) // self.raw_d2h_pool_group_size)
        self._raw_d2h_value_slabs = [None] * len(self._raw_d2h_key_slabs)
        self._raw_d2h_group_shapes = [None] * len(self._raw_d2h_key_slabs)
        self._raw_d2h_group_spans = [0] * len(self._raw_d2h_key_slabs)
        self._raw_d2h_group_pending = [0] * len(self._raw_d2h_key_slabs)
        self._raw_d2h_layer_acquired = [False] * self.layer_num
        self._raw_d2h_layer_to_group = [self._layer_group_index(ldx, self.raw_d2h_pool_group_size) for ldx in range(self.layer_num)]

        self.imi_pipelines = []
        if self.build_index_when_prefilling:
            enable_direct_write = bool(self.runtime_config["pipeline"].get("enable_direct_write", True))
            for ldx in range(self.layer_num):
                device_id = int(self.layer_mapping[str(ldx)].split(":")[-1])
                learned_partition = self.learned_partition_specs[ldx]
                if learned_partition is None:
                    raise RuntimeError(f"missing learned partition for layer {ldx}")
                layer_subspace_parts = int(learned_partition["subspace_parts"])
                self.imi_pipelines.append(
                    IMIPipeline(
                        layer_idx=ldx,
                        kv_heads=self.kv_head,
                        head_dim=self.head_dim,
                        dtype=self.dtype,
                        device_id=device_id,
                        max_tokens=self.list_capacity,
                        subspace_parts=layer_subspace_parts,
                        learned_partition=learned_partition,
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
                        key_rows = key_storage.view(self.batch_groups, self.list_capacity, self.head_dim)
                        value_rows = value_storage.view(self.batch_groups, self.list_capacity, self.head_dim)
                        results = []
                        for row_idx, required_tokens in heads_info:
                            key_buffer = key_rows[row_idx]
                            value_buffer = value_rows[row_idx]
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

        self.cluster_sizes_cpu = [None] * self.layer_num
        self.cluster_offsets_cpu = [None] * self.layer_num
        self.cluster_sizes_gpu = [None] * self.layer_num
        self.cluster_offsets_gpu = [None] * self.layer_num
        self.centroids = [None] * self.layer_num
        self.centroids_mask = [None] * self.layer_num
        self.centroids_cpu = [None] * self.layer_num
        self.centroids_mask_cpu = [None] * self.layer_num

        self.cache_keys = []
        self.cache_values = []

        self.gpu_cache_aware_hit_unit_idices = []
        self.gpu_cache_aware_hit_unit_sizes = []
        self.gpu_cache_aware_hit_unit_sizes_cumsum = []
        self.gpu_cache_aware_hit_num_units = []
        self.gpu_cache_aware_miss_unit_idices = []
        self.gpu_cache_aware_miss_unit_sizes = []
        self.gpu_cache_aware_miss_unit_sizes_cumsum = []
        self.gpu_cache_aware_miss_cluster_unit_sizes_cumsum = []
        self.gpu_cache_aware_miss_num_units = []
        self.gpu_cache_aware_miss_copy_unit_idices = []
        self.gpu_cache_aware_miss_copy_unit_sizes = []
        self.gpu_cache_aware_miss_copy_unit_sizes_cumsum = []
        self.gpu_cache_aware_miss_copy_num_units = []
        self.gpu_cache_aware_cluster_copy_unit_idices = []
        self.gpu_cache_aware_cluster_copy_unit_sizes = []
        self.gpu_cache_aware_cluster_copy_unit_sizes_cumsum = []
        self.gpu_cache_aware_cluster_copy_num_units = []
        self.gpu_cache_aware_miss_page_keys = []
        self.gpu_cache_aware_admit_unit_starts = []
        self.gpu_cache_promote_src_offsets = []
        self.gpu_cache_promote_dst_slots = []
        self.gpu_cache_promote_sizes = []
        self.gpu_cache_promote_hash_positions = []
        self.gpu_cache_promote_page_keys = []
        self.gpu_cache_promote_counts = []
        self.gpu_cache_hash_keys = []
        self.gpu_cache_hash_values = []
        self.gpu_cache_slot_keys = []
        self.gpu_cache_cluster_state = []
        self.gpu_cache_cluster_prev = []
        self.gpu_cache_cluster_next = []
        self.gpu_cache_hot_head = []
        self.gpu_cache_hot_tail = []
        self.gpu_cache_cold_head = []
        self.gpu_cache_cold_tail = []
        self.gpu_cache_hot_pages = []
        self.gpu_cache_cold_pages = []
        self.gpu_cache_free_slots = []
        self.gpu_cache_free_counts = []
        self.gpu_cache_hash_size = 0
        self.gpu_cache_aware_ring_size = 2
        self.gpu_cache_aware_active_slots = [-1] * self.layer_num
        self.gpu_cache_promote_slots = [-1] * self.layer_num
        self.gpu_cache_promote_streams: Dict[str, torch.cuda.Stream] = {}
        self.gpu_cache_promote_events = [None] * self.layer_num
        self.gpu_cache_promote_start_events = [None] * self.layer_num
        self.gpu_cache_promotion_decode_step = 0

        self.execution_buffer_keys_dict: Dict[str, torch.Tensor] = {}
        self.execution_buffer_values_dict: Dict[str, torch.Tensor] = {}
        self.execution_buffer_key_ring_dict: Dict[str, List[torch.Tensor]] = {}
        self.execution_buffer_value_ring_dict: Dict[str, List[torch.Tensor]] = {}
        self.execution_buffer_ring_events: Dict[str, List[Optional[torch.cuda.Event]]] = {}
        self.execution_buffer_ring_next: Dict[str, int] = {}
        self.execution_buffer_fallback_keys_dict: Dict[str, torch.Tensor] = {}
        self.execution_buffer_fallback_values_dict: Dict[str, torch.Tensor] = {}
        self.valid_lengths_dict: Dict[str, torch.Tensor] = {}
        self.static_len_tensor_dict: Dict[str, torch.Tensor] = {}
        self.static_lengths_dict: Dict[str, torch.Tensor] = {}
        self.nprobe_tensor_dict: Dict[str, torch.Tensor] = {}
        self.similarity_buffer_dict: Dict[str, torch.Tensor] = {}
        self.selected_cluster_buffer_dict: Dict[str, torch.Tensor] = {}
        self.generated_tokens_total = 0
        self.generated_recent_counts = [0] * self.layer_num
        self.static_layer_lengths = [self.base_static_pattern_total] * self.layer_num
        max_delta_tokens = max(self.capacity_actual_gen_len - self.async_update_start_after_tokens, 0)
        self.async_update_max_delta_chunks = (
            (max_delta_tokens + self.async_update_interval - 1) // self.async_update_interval
            if self.async_update_enabled
            else 0
        )
        self.delta_active_chunks = [0] * self.layer_num
        self.delta_mini_keys: List[Optional[torch.Tensor]] = [None] * self.layer_num
        self.delta_mini_values: List[Optional[torch.Tensor]] = [None] * self.layer_num
        self.delta_mini_centroids: List[Optional[torch.Tensor]] = [None] * self.layer_num

        self.prefill_chunk_size = max(int(self.runtime_config["streaming"].get("prefill_chunk_size", 32768)), 1)
        self.profile_decode = os.getenv("IMI_PROFILE_DECODE", "0") == "1"
        self.profile_prefill = os.getenv("IMI_PROFILE_PREFILL", "0") == "1"
        self.profile_decode_steps = int(os.getenv("IMI_PROFILE_STEPS", "5"))
        self.profile_decode_count = 0
        self.profile_decode_step_idx = -1
        self.profile_token_hit_rate = os.getenv("IMI_PROFILE_TOKEN_HIT_RATE", "0") == "1"
        self.decode_breakdown: Optional[dict] = None
        self.decode_breakdown_accum: dict = {}
        self.decode_profile_rows: List[Dict[str, object]] = []
        self.prefill_breakdown_rows: List[Dict[str, object]] = []
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

        self.allocated = self.pre_allocate_decision()
      

        self.prefill_stream_started = [False] * self.layer_num
        self.prefill_stream_end_submitted = [False] * self.layer_num

        self.prefill_stream_done_events = [threading.Event() for _ in range(self.layer_num)]
        self.prefill_stream_errors = [None] * self.layer_num

        self._build_job_seq = 0
        self._build_job_lock = threading.Lock()
        self._build_min_threads_per_layer = max(1, self.kv_head * max(max_partition_m, self.subspace_parts, 1))
        max_concurrent = self.runtime_config["pipeline"].get("kmeans_max_concurrent")
        if max_concurrent is not None:
            max_concurrent = int(max_concurrent)
        if max_concurrent is None and self.list_capacity >= 65536:
            max_concurrent = 1
        self._build_job_scheduler = _BuildJobScheduler(
            self._available_cpu_cores,
            self._build_min_threads_per_layer,
            max_concurrent=max_concurrent,
        )
        self._metadata_lock = threading.Lock()

    @staticmethod
    def _layer_group_index(layer_idx: int, group_size: int) -> int:
        return int(layer_idx) // max(int(group_size), 1)

    @staticmethod
    def _group_span_for_layer(layer_idx: int, group_size: int, layer_num: int) -> int:
        group_size = max(int(group_size), 1)
        group_start = (int(layer_idx) // group_size) * group_size
        return min(group_size, int(layer_num) - group_start)

    def _ensure_raw_d2h_storage(self, layer_idx: int, raw_shape, dtype: torch.dtype):
        layer_idx = int(layer_idx)
        group_idx = self._raw_d2h_layer_to_group[layer_idx]
        group_start = group_idx * self.raw_d2h_pool_group_size
        span = self._group_span_for_layer(group_start, self.raw_d2h_pool_group_size, self.layer_num)
        expected_shape = (span, *tuple(raw_shape))
        needs_alloc = (
            self._raw_d2h_key_slabs[group_idx] is None
            or tuple(self._raw_d2h_key_slabs[group_idx].shape) != expected_shape
            or self._raw_d2h_key_slabs[group_idx].dtype != dtype
            or self._raw_d2h_value_slabs[group_idx] is None
            or tuple(self._raw_d2h_value_slabs[group_idx].shape) != expected_shape
            or self._raw_d2h_value_slabs[group_idx].dtype != dtype
        )
        if needs_alloc:
            if self._raw_d2h_group_pending[group_idx] > 0:
                raise RuntimeError("cannot resize raw D2H staging slab while the group has active layers")
            self._raw_d2h_key_slabs[group_idx] = torch.empty(
                expected_shape,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            self._raw_d2h_value_slabs[group_idx] = torch.empty(
                expected_shape,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            self._raw_d2h_group_shapes[group_idx] = expected_shape
            self._raw_d2h_group_spans[group_idx] = span
            self._raw_d2h_group_pending[group_idx] = 0
        if not self._raw_d2h_layer_acquired[layer_idx]:
            self._raw_d2h_group_pending[group_idx] += 1
            self._raw_d2h_layer_acquired[layer_idx] = True
        local_idx = layer_idx - group_start
        return self._raw_d2h_key_slabs[group_idx][local_idx], self._raw_d2h_value_slabs[group_idx][local_idx]

    def _release_raw_d2h_storage(self, layer_idx: int) -> None:
        layer_idx = int(layer_idx)
        if not self._raw_d2h_layer_acquired[layer_idx]:
            return
        group_idx = self._raw_d2h_layer_to_group[layer_idx]
        self._raw_d2h_layer_acquired[layer_idx] = False
        self._raw_d2h_group_pending[group_idx] = max(0, self._raw_d2h_group_pending[group_idx] - 1)
        if self._raw_d2h_group_pending[group_idx] > 0:
            return
        self._raw_d2h_key_slabs[group_idx] = None
        self._raw_d2h_value_slabs[group_idx] = None
        self._raw_d2h_group_shapes[group_idx] = None
        self._raw_d2h_group_spans[group_idx] = 0

    def can_reuse_for_next_sequence(self, valid_start, max_new_length: int, input_length: int) -> bool:
        new_input_length = int(input_length)
        new_actual_gen_len = int(max_new_length) - 1
        base_static_pattern_total = int(getattr(self, "base_static_pattern_total", self.static_pattern_start + self.static_pattern_end))
        new_list_stride = max(new_input_length - base_static_pattern_total, 0)
        if new_list_stride > self.list_capacity:
            return False
        if self.async_update_enabled:
            max_delta_tokens = max(new_actual_gen_len - self.async_update_start_after_tokens, 0)
            required_delta_chunks = (max_delta_tokens + self.async_update_interval - 1) // self.async_update_interval
            if required_delta_chunks > self.async_update_max_delta_chunks:
                return False
        if self.build_index_when_prefilling:
            if self.async_update_enabled:
                new_static_stride = base_static_pattern_total + min(new_actual_gen_len, self.async_update_recent_capacity)
            else:
                new_static_stride = base_static_pattern_total + new_actual_gen_len
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
        self.static_pattern_total = int(getattr(self, "base_static_pattern_total", self.static_pattern_start + self.static_pattern_end))
        self.list_stride = max(self.input_length - self.static_pattern_total, 0)
        if self.build_index_when_prefilling:
            if self.async_update_enabled:
                self.static_stride = self.static_pattern_total + min(self.actual_gen_len, self.async_update_recent_capacity)
            else:
                self.static_stride = self.static_pattern_total + self.actual_gen_len
        else:
            self.static_stride = self.input_length + self.actual_gen_len
        self.context = 0
        self.attn_func = self.dense_attention
        self.generated_tokens_total = 0
        self.generated_recent_counts = [0] * self.layer_num
        self.static_layer_lengths = [self.base_static_pattern_total] * self.layer_num
        self.delta_active_chunks = [0] * self.layer_num

        self.layer_metadata = [None] * self.layer_num
        self.layer_index_stats = [None] * self.layer_num
        self.layer_ready = [False] * self.layer_num
        self.layer_started = [False] * self.layer_num
        self.cache_prepared = False
        self._index_summary_logged = False
        self._expected_middle_len = [None] * self.layer_num

        self.profile_decode_count = 0
        self.profile_decode_step_idx = -1
        self.decode_breakdown = None
        self.decode_breakdown_accum = {}
        self.decode_profile_rows = []
        self.prefill_breakdown_rows = []
        self.profile_block_step = 0
        self.profile_block_hits = 0
        self.profile_block_misses = 0
        self.profile_layer_hit_rate_count = 0
        self.decode_debug_step = 0

        self.prefill_stream_started = [False] * self.layer_num
        self.prefill_stream_end_submitted = [False] * self.layer_num
        self.prefill_stream_errors = [None] * self.layer_num
        for done_event in self.prefill_stream_done_events:
            done_event.clear()
        for promote_event in self.gpu_cache_promote_events:
            if promote_event is not None:
                promote_event.synchronize()
        for events in self.execution_buffer_ring_events.values():
            for event in events:
                if event is not None:
                    event.synchronize()
        self.gpu_cache_promote_events = [None] * self.layer_num
        self.gpu_cache_promote_start_events = [None] * self.layer_num
        self.gpu_cache_aware_active_slots = [-1] * self.layer_num
        self.gpu_cache_promote_slots = [-1] * self.layer_num
        for device in list(self.execution_buffer_ring_events):
            self.execution_buffer_ring_events[device] = [None] * len(self.execution_buffer_ring_events[device])
            self.execution_buffer_ring_next[device] = 0
        self.gpu_cache_promotion_decode_step = 0

        for layer_idx in range(self.layer_num):
            if self.cluster_sizes_cpu[layer_idx] is not None:
                self.cluster_sizes_cpu[layer_idx].zero_()
            if self.cluster_offsets_cpu[layer_idx] is not None:
                self.cluster_offsets_cpu[layer_idx].zero_()
            if self.cluster_sizes_gpu[layer_idx] is not None:
                self.cluster_sizes_gpu[layer_idx].zero_()
            if self.cluster_offsets_gpu[layer_idx] is not None:
                self.cluster_offsets_gpu[layer_idx].zero_()
            if layer_idx < len(self.gpu_cache_hash_keys) and self.gpu_cache_hash_keys[layer_idx] is not None:
                self.gpu_cache_hash_keys[layer_idx].fill_(-1)
            if layer_idx < len(self.gpu_cache_hash_values) and self.gpu_cache_hash_values[layer_idx] is not None:
                self.gpu_cache_hash_values[layer_idx].fill_(-1)
            if layer_idx < len(self.gpu_cache_slot_keys) and self.gpu_cache_slot_keys[layer_idx] is not None:
                self.gpu_cache_slot_keys[layer_idx].fill_(-1)
            if layer_idx < len(self.gpu_cache_cluster_state) and self.gpu_cache_cluster_state[layer_idx] is not None:
                self.gpu_cache_cluster_state[layer_idx].zero_()
            if layer_idx < len(self.gpu_cache_cluster_prev) and self.gpu_cache_cluster_prev[layer_idx] is not None:
                self.gpu_cache_cluster_prev[layer_idx].fill_(-1)
            if layer_idx < len(self.gpu_cache_cluster_next) and self.gpu_cache_cluster_next[layer_idx] is not None:
                self.gpu_cache_cluster_next[layer_idx].fill_(-1)
            for queue_tensor in (
                self.gpu_cache_hot_head,
                self.gpu_cache_hot_tail,
                self.gpu_cache_cold_head,
                self.gpu_cache_cold_tail,
            ):
                if layer_idx < len(queue_tensor) and queue_tensor[layer_idx] is not None:
                    queue_tensor[layer_idx].fill_(-1)
            for count_tensor in (
                self.gpu_cache_hot_pages,
                self.gpu_cache_cold_pages,
            ):
                if layer_idx < len(count_tensor) and count_tensor[layer_idx] is not None:
                    count_tensor[layer_idx].zero_()
            if layer_idx < len(self.gpu_cache_free_counts) and self.gpu_cache_free_counts[layer_idx] is not None:
                self.gpu_cache_free_counts[layer_idx].fill_(int(self.cache_size))
            if layer_idx < len(self.gpu_cache_free_slots) and self.gpu_cache_free_slots[layer_idx] is not None:
                free_slots = torch.arange(
                    int(self.cache_size),
                    dtype=torch.int32,
                    device=self.gpu_cache_free_slots[layer_idx].device,
                ).view(1, -1)
                self.gpu_cache_free_slots[layer_idx].copy_(free_slots.expand(self.batch_groups, -1))
            if self.centroids_cpu[layer_idx] is not None:
                self.centroids_cpu[layer_idx].zero_()
            if self.centroids_mask_cpu[layer_idx] is not None:
                self.centroids_mask_cpu[layer_idx].fill_(True)
            if self.centroids[layer_idx] is not None:
                self.centroids[layer_idx].zero_()
            if self.centroids_mask[layer_idx] is not None:
                self.centroids_mask[layer_idx].fill_(True)
            if self.delta_mini_centroids[layer_idx] is not None:
                self.delta_mini_centroids[layer_idx].zero_()

    def _estimate_cache_parameters(self) -> Tuple[int, int, int, int]:
        if not self.build_index_when_prefilling:
            return 0, 0, 0, 0

        n_tokens = max(self.list_capacity, self.list_stride, 1)
        kmeans_cfg = self.runtime_config.get("kmeans") or {}
        target_cluster_size = max(int(kmeans_cfg.get("target_cluster_size", kmeans_cfg.get("cluster_size", 16))), 1)
        max_clusters = 1
        for spec in self.learned_partition_specs:
            if spec is None:
                raise RuntimeError("AdaptiveIMI hybrid_8way_v2 requires learned partition metadata for every layer")
            m = max(int(spec.get("subspace_parts", 1)), 1)
            k_values = kmeans_cfg.get("k_values")
            if k_values is None:
                k_values = choose_learned_k_values(
                    spec,
                    n_tokens,
                    target_cluster_size,
                    default_m=m,
                )
            product = product_k_values(k_values)
            max_clusters = max(max_clusters, min(product, n_tokens))
        n_centroids = int(max_clusters) + 1

        nprobe = max(round((n_centroids - 1) * self.retrieval_budget), 1)
        nprobe = min(nprobe, n_centroids - 1)

        cache_cluster_num = (
            round(n_centroids * self.cache_ratio) if self.cache_ratio > 0.0 else nprobe * 3
        )
        requested_cache_pages = max(cache_cluster_num * self.pages_per_cluster, 1)
        cache_pages = self._resolve_budget_pages(requested_cache_pages)
        if self.batch_size > 1:
            total_cache_pages = int(self.multibatch_cache_total_pages)
            if total_cache_pages > 0:
                cache_pages = min(cache_pages, max(1, total_cache_pages // int(self.batch_size)))

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
        per_layer_cache_bytes = (
            2
            * self.batch_size
            * self.kv_head
            * cache_tokens
            * self.head_dim
            * bytes_per_elem
        )
        execution_tokens = buffer_pages * self.page_size + self.static_stride
        per_execution_buffer_bytes = (
            2
            * self.batch_size
            * self.kv_head
            * execution_tokens
            * self.head_dim
            * bytes_per_elem
        )
        per_layer_execution_bytes = per_execution_buffer_bytes * int(self._execution_buffer_ring_size())
        per_device_fallback_execution_bytes = per_execution_buffer_bytes
        per_layer_centroid_bytes = self.batch_groups * n_centroids * self.head_dim * 4
        per_layer_mask_bytes = self.batch_groups * n_centroids

        estimated_bytes_by_device = {device: 0 for device in self.device_list}
        for ldx in range(self.layer_num):
            device = self.layer_mapping[str(ldx)]
            estimated_bytes_by_device[device] += (
                per_layer_cache_bytes
                + per_layer_execution_bytes
                + per_layer_centroid_bytes
                + per_layer_mask_bytes
            )
        for device in self.device_list:
            estimated_bytes_by_device[device] += per_device_fallback_execution_bytes

        estimated_gb_by_device = {
            device: value / (1024 ** 3)
            for device, value in estimated_bytes_by_device.items()
        }
        estimated_gb = sum(estimated_gb_by_device.values())
        self.estimated_gpu_memory = estimated_gb
        self.estimated_gpu_memory_by_device = estimated_gb_by_device
        self.esitimate_gpu_memory = estimated_gb  # Backward-compatible misspelling.
        for device, estimated_bytes in estimated_bytes_by_device.items():
            free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
            if free_bytes < int(estimated_bytes * 1.5):
                return False
        return True
# Backward-compatible alias for older imports.
adpimi_cache = AdaptiveIMICacheBase
