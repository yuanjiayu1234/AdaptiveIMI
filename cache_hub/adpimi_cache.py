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
from weighted_flash_decoding import weighted_flash_decoding

from library.AdaptiveIMI.imi_adapter import IMIPipeline, IMIRuntimeConfig, get_imi_kernels
from .cache import KV_Cache


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


class _KmeansScheduler:
    def __init__(self, total_cores: int, min_threads_per_job: int, max_concurrent: Optional[int] = None) -> None:
        self._total_cores = max(1, total_cores)
        self._min_threads_per_job = max(1, min_threads_per_job)
        if max_concurrent is None:
            self._max_concurrent = max(1, self._total_cores // self._min_threads_per_job)
        else:
            self._max_concurrent = max(1, max_concurrent)
        self._queue: "queue.PriorityQueue[tuple[tuple[int, int], float, object]]" = queue.PriorityQueue()
        self._active_jobs = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="imi-kmeans-dispatcher",
        )
        self._dispatcher.start()

    def submit(self, priority: tuple[int, int], fn) -> None:
        if self._stop_event.is_set():
            return
        self._queue.put((priority, time.monotonic(), fn))
        with self._cv:
            self._cv.notify_all()

    def shutdown(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._queue.put(((0, 0), 0.0, None))
        with self._cv:
            self._cv.notify_all()
        self._dispatcher.join(timeout=1.0)

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                _, _, fn = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if fn is None:
                return
            with self._cv:
                while self._active_jobs >= self._max_concurrent and not self._stop_event.is_set():
                    self._cv.wait(timeout=0.1)
                if self._stop_event.is_set():
                    return
                self._active_jobs += 1
            worker_threads = max(1, self._total_cores // max(1, self._max_concurrent))
            thread = threading.Thread(
                target=self._run_job,
                args=(fn, worker_threads),
                daemon=True,
                name="imi-kmeans-worker",
            )
            thread.start()

    def _run_job(self, fn, worker_threads: int) -> None:
        try:
            fn(worker_threads)
        finally:
            with self._cv:
                self._active_jobs = max(0, self._active_jobs - 1)
                self._cv.notify_all()


class adpimi_cache(KV_Cache):
    """KV cache using IMI indexing with block cache/gather flow."""

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
        if self.input_length <= 0:
            raise ValueError(f"input length({self.input_length}) should be larger than 0")

        self.list_stride = max(self.input_length - self.static_pattern_total, 0)
        self.build_index_when_prefilling = self.list_stride > 0
        if self.build_index_when_prefilling:
            self.static_stride = self.static_pattern_total + actual_gen_len
        else:
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
                    (self.batch_size, self.kv_head, self.static_stride, self.head_dim),
                    dtype=self.dtype,
                    device=self.layer_mapping[str(ldx)],
                ).contiguous()
            )
            self.steady_zone_values.append(
                torch.zeros(
                    (self.batch_size, self.kv_head, self.static_stride, self.head_dim),
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
                        max_tokens=self.list_stride,
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
        if max_concurrent is None and self.list_stride >= 65536:
            max_concurrent = 1
        self._kmeans_scheduler = _KmeansScheduler(
            self._available_cpu_cores,
            self._kmeans_min_threads_per_layer,
            max_concurrent=max_concurrent,
        )

        self._metadata_lock = threading.Lock()

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


    def _ensure_list_storage(self, layer_idx: int) -> None:
        if not self.build_index_when_prefilling:
            return
        if self.list_keys[layer_idx] is not None and self.list_values[layer_idx] is not None:
            return
        self.list_keys[layer_idx] = torch.empty(
            (self.batch_size, self.kv_head, self.list_stride, self.head_dim),
            dtype=self.dtype,
            pin_memory=True,
        ).contiguous()
        self.list_values[layer_idx] = torch.empty(
            (self.batch_size, self.kv_head, self.list_stride, self.head_dim),
            dtype=self.dtype,
            pin_memory=True,
        ).contiguous()


    def _load_layer_metadata(self, layer_idx: int, metadata: List[Dict[str, object]]):
        self.layer_metadata[layer_idx] = metadata
        self.layer_ready[layer_idx] = True

        cluster_sizes = self.cluster_sizes_cpu[layer_idx]
        cluster_offsets = self.cluster_offsets_cpu[layer_idx]
        centroids_cpu = torch.zeros(
            (self.batch_groups, self.n_centroids, self.head_dim),
            dtype=torch.float32,
        )
        centroids_mask_cpu = torch.ones(
            (self.batch_groups, self.n_centroids),
            dtype=torch.bool,
        )

        for head_meta in metadata:
            head_idx = head_meta["head_idx"]
            sizes_tensor = head_meta["cluster_sizes"]
            offsets_tensor = head_meta["cluster_offsets"]
            centroids_tensor = head_meta["centroids"]
            cluster_count = int(sizes_tensor.shape[0])

            cluster_sizes[head_idx, :cluster_count].copy_(sizes_tensor)
            cluster_offsets[head_idx, : cluster_count + 1].copy_(offsets_tensor)
            if cluster_count < self.n_centroids:
                cluster_offsets[head_idx, cluster_count + 1 :] = int(offsets_tensor[-1].item())
            centroids_cpu[head_idx, :cluster_count, :].copy_(centroids_tensor)
            centroids_mask_cpu[head_idx, :cluster_count] = sizes_tensor == 0
            if cluster_count < self.n_centroids:
                centroids_mask_cpu[head_idx, cluster_count:] = True

        if os.getenv("IMI_DEBUG_INDEX_METADATA", "0") == "1":
            nonzero_heads = 0
            total_clusters = 0
            max_clusters = 0
            sample_counts = []
            for head_meta in metadata[: min(8, len(metadata))]:
                count = int(head_meta["cluster_sizes"].shape[0])
                sample_counts.append(count)
            for head_meta in metadata:
                count = int(head_meta["cluster_sizes"].shape[0])
                if count > 0:
                    nonzero_heads += 1
                    total_clusters += count
                    max_clusters = max(max_clusters, count)
            print(
                json.dumps({
                    "tag": "IMI_META_LOAD",
                    "layer_idx": int(layer_idx),
                    "nonzero_heads": int(nonzero_heads),
                    "total_clusters": int(total_clusters),
                    "max_clusters": int(max_clusters),
                    "sample_cluster_counts": sample_counts,
                }, ensure_ascii=False),
                flush=True,
            )

        self.centroids_cpu[layer_idx] = centroids_cpu
        self.centroids_mask_cpu[layer_idx] = centroids_mask_cpu
        if self.cluster_sizes_gpu[layer_idx] is not None:
            self.cluster_sizes_gpu[layer_idx].copy_(cluster_sizes)
        if self.centroids[layer_idx] is not None:
            self.centroids[layer_idx].copy_(centroids_cpu)
        if self.centroids_mask[layer_idx] is not None:
            self.centroids_mask[layer_idx].copy_(centroids_mask_cpu)
        if self.adpimi_index[layer_idx] is not None:
            self.adpimi_index[layer_idx].set_cluster_metadata(cluster_sizes, cluster_offsets, 0)

    def _ensure_metadata_buffers(self) -> None:
        if self.n_centroids <= 0:
            n_tokens = max(self.list_stride, 1)
            total_clusters = max(int(n_tokens / 16), 1)
            if self.subspace_parts == 0:
                # IVF: n_centroids = round(T / 16)
                self.n_centroids = max(int(round(n_tokens / 16.0)), 1) + 1
            elif self.subspace_parts == 4:
                total_clusters_f = max(n_tokens / 16.0, 1.0)
                k = max(int(round(total_clusters_f ** 0.25)), 2)
                self.n_centroids = k ** 4 + 1
            else:
                # 2-way IMI
                k = max(int(math.sqrt(total_clusters)), 32)
                self.n_centroids = k * k + 1
            self.padding_cluster_id = self.n_centroids - 1
            self.nprobe = max(round((self.n_centroids - 1) * self.retrieval_budget), 1)
            self.nprobe = min(self.nprobe, self.n_centroids - 1)

        for ldx in range(self.layer_num):
            if self.cluster_sizes_cpu[ldx] is not None and self.cluster_offsets_cpu[ldx] is not None:
                continue
            cluster_sizes = torch.zeros(
                (self.batch_groups, self.n_centroids),
                dtype=torch.int32,
                pin_memory=True,
            )
            cluster_offsets = torch.zeros(
                (self.batch_groups, self.n_centroids + 1),
                dtype=torch.int32,
                pin_memory=True,
            )
            self.cluster_sizes_cpu[ldx] = cluster_sizes
            self.cluster_offsets_cpu[ldx] = cluster_offsets
            if self.allocated:
                if self.cluster_sizes_gpu[ldx] is None:
                    self.cluster_sizes_gpu[ldx] = cluster_sizes.to(self.layer_mapping[str(ldx)])
                if self.centroids_mask[ldx] is None:
                    self.centroids_mask[ldx] = torch.ones(
                        (self.batch_groups, self.n_centroids),
                        dtype=torch.bool,
                        device=self.layer_mapping[str(ldx)],
                    )
                if self.centroids[ldx] is None:
                    self.centroids[ldx] = torch.zeros(
                        (self.batch_groups, self.n_centroids, self.head_dim),
                        dtype=torch.float32,
                        device=self.layer_mapping[str(ldx)],
                    )
        
    def _finish_layer_index(self, layer_idx: int):
        progress = os.getenv("IMI_STREAMING_PROGRESS", "0") == "1"
        start_ts = time.perf_counter() if progress else None
        if progress:
            print(f"[IMI INDEX] layer={layer_idx} phase=start", flush=True)
        self._ensure_list_storage(layer_idx)
        metadata = self.imi_pipelines[layer_idx].finish_index(
            self.list_keys[layer_idx],
            self.list_values[layer_idx],
        )
        placeholder = torch.empty((self.batch_groups, 0, self.head_dim), dtype=self.dtype, pin_memory=True)
        self.adpimi_index[layer_idx].set_kv(
            self.list_keys[layer_idx],
            self.list_values[layer_idx],
            placeholder,
            placeholder,
        )
        if os.getenv("IMI_DEBUG_STREAMING_CHECK", "0") == "1":
            expected_len = self._expected_middle_len[layer_idx]
            pipeline_stats = self.imi_pipelines[layer_idx].get_last_pipeline_stats()
            actual_len = None
            if pipeline_stats is not None:
                actual_len = int(pipeline_stats.get("total_tokens", 0))
            print(
                "[IMI STREAM CHECK]"
                f" layer={layer_idx} expected_tokens={expected_len}"
                f" actual_tokens={actual_len}",
                flush=True,
            )
        if self.profile_prefill:
            pipeline_stats = self.imi_pipelines[layer_idx].get_last_pipeline_stats()
            if pipeline_stats is not None:
                self.prefill_breakdown_rows.append({
                    "layer_idx": int(layer_idx),
                    "d2h_ms": round(float(pipeline_stats.get("d2h_ms", 0.0)), 4),
                    "cpu_copy_ms": round(float(pipeline_stats.get("cpu_copy_ms", 0.0)), 4),
                    "kmeans_ms": round(float(pipeline_stats.get("kmeans_ms", 0.0)), 4),
                    "kmeans_cpu_time_ms": round(float(pipeline_stats.get("kmeans_cpu_time_ms", 0.0)), 4),
                    "kmeans_cpu_util_cores": round(float(pipeline_stats.get("kmeans_cpu_util_cores", 0.0)), 4),
                    "reorganize_ms": round(float(pipeline_stats.get("reorganize_ms", 0.0)), 4),
                    "write_ms": round(float(pipeline_stats.get("write_ms", 0.0)), 4),
                    "kmeans_gate_wait_ms": round(float(pipeline_stats.get("kmeans_gate_wait_ms", 0.0)), 4),
                    "total_chunks": int(pipeline_stats.get("total_chunks", 0)),
                    "total_tokens": int(pipeline_stats.get("total_tokens", 0)),
                })
        self._ensure_metadata_buffers()
        with self._metadata_lock:
            self._load_layer_metadata(layer_idx, metadata)
        self._log_index_summary_once(layer_idx)
        if progress and start_ts is not None:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            print(f"[IMI INDEX] layer={layer_idx} phase=done elapsed_ms={elapsed_ms:.2f}", flush=True)

    def _log_index_summary_once(self, layer_idx: int) -> None:
        if os.getenv("IMI_DEBUG_INDEX_SUMMARY", "0") != "1":
            return
        if layer_idx != 0 or self._index_summary_logged:
            return
        self._index_summary_logged = True
        stats = self.get_index_stats()
        print(
            "[IMI INDEX SUMMARY]"
            f" n_centroids={stats.get('n_centroids')}"
            f" ready_layers={stats.get('ready_layers')}"
            f" total_heads={stats.get('total_heads')}"
            f" non_empty_ratio={stats.get('non_empty_ratio'):.4f}",
            flush=True,
        )

    def _schedule_finish_layer_index(self, layer_idx: int) -> None:
        done_event = self.prefill_stream_done_events[layer_idx]

        with self._kmeans_job_lock:
            job_seq = self._kmeans_job_seq
            self._kmeans_job_seq += 1

        def _run(worker_threads: int) -> None:
            try:
                pipeline = self.imi_pipelines[layer_idx]
                if pipeline is None:
                    return
                pipeline.set_worker_threads(worker_threads)
                self._finish_layer_index(layer_idx)
            except Exception as exc:
                print(
                    f"[IMI ERROR] layer={layer_idx} background index failed: {exc}",
                    flush=True,
                )
                if self.prefill_stream_errors[layer_idx] is None:
                    self.prefill_stream_errors[layer_idx] = exc
            finally:
                done_event.set()

        self._kmeans_scheduler.submit((int(layer_idx), job_seq), _run)

    def get_index_stats(self) -> Dict[str, object]:
        ideal_clusters_per_head = max(float(self.list_stride) / 16.0, 0.0)
        total_heads = int(self.layer_num * self.batch_groups)
        ideal_clusters_total = ideal_clusters_per_head * float(total_heads)

        non_empty_clusters_total = 0
        ready_layers = 0
        for metadata in self.layer_metadata:
            if not metadata:
                continue
            ready_layers += 1
            for head_meta in metadata:
                sizes_tensor = head_meta.get("cluster_sizes")
                if sizes_tensor is None:
                    continue
                non_empty_clusters_total += int((sizes_tensor > 0).sum().item())

        centroid_bytes_per_layer = int(self.batch_groups * self.n_centroids * self.head_dim * 4)
        total_centroid_bytes_gpu = int(ready_layers * centroid_bytes_per_layer)
        non_empty_ratio = (
            float(non_empty_clusters_total) / float(ideal_clusters_total)
            if ideal_clusters_total > 0
            else 0.0
        )

        return {
            "n_centroids": int(self.n_centroids),
            "ready_layers": int(ready_layers),
            "total_heads": total_heads,
            "ideal_clusters_per_head": float(ideal_clusters_per_head),
            "ideal_clusters_total": float(ideal_clusters_total),
            "non_empty_clusters_total": int(non_empty_clusters_total),
            "non_empty_ratio": float(non_empty_ratio),
            "total_centroid_bytes_gpu": total_centroid_bytes_gpu,
        }

    def _raise_prefill_stream_error(self, layer_idx: int):
        err = self.prefill_stream_errors[layer_idx]
        if err is not None:
            raise RuntimeError(f"IMI streaming prefill failed on layer {layer_idx}") from err

    def _prefill_stream_worker_loop(self, layer_idx: int):
        worker_queue = self.prefill_stream_queues[layer_idx]
        if worker_queue is None:
            return

        stop_event = self.prefill_stream_stop_events[layer_idx]
        done_event = self.prefill_stream_done_events[layer_idx]

        while True:
            task = worker_queue.get()
            if task is None:
                if stop_event.is_set():
                    return
                continue

            slot_id = task["slot_id"]
            try:
                task["copy_done_event"].synchronize()
                if self.prefill_stream_errors[layer_idx] is not None:
                    continue

                chunk_id = task["chunk_id"]
                token_offset = task["token_offset"]
                chunk_len = task["chunk_len"]
                is_last = task["is_last"]

                expected_chunk_id = self.prefill_stream_expected_chunk_id[layer_idx]
                expected_token_offset = self.prefill_stream_expected_token_offset[layer_idx]
                if chunk_id != expected_chunk_id:
                    raise RuntimeError(
                        f"layer {layer_idx}: chunk_id out of order, expected {expected_chunk_id}, got {chunk_id}"
                    )
                if token_offset != expected_token_offset:
                    raise RuntimeError(
                        f"layer {layer_idx}: token_offset out of order, expected {expected_token_offset}, got {token_offset}"
                    )

                stage_keys = task.get("stage_keys")
                stage_values = task.get("stage_values")
                if stage_keys is None or stage_values is None:
                    stage_keys = self.prefill_stream_stage_keys[layer_idx][slot_id]
                    stage_values = self.prefill_stream_stage_values[layer_idx][slot_id]

                self.imi_pipelines[layer_idx].submit_index_stream_chunk(
                    stage_keys[:, :, :chunk_len, :],
                    stage_values[:, :, :chunk_len, :],
                    chunk_id=chunk_id,
                    token_offset=token_offset,
                    is_last=is_last,
                )

                self.prefill_stream_expected_chunk_id[layer_idx] += 1
                self.prefill_stream_expected_token_offset[layer_idx] += chunk_len

                if is_last:
                    if self.prefill_stream_last_chunk_seen[layer_idx]:
                        raise RuntimeError(f"layer {layer_idx}: duplicate last chunk")
                    self.prefill_stream_last_chunk_seen[layer_idx] = True
                    self.prefill_stream_end_submitted[layer_idx] = True
                    self._finish_layer_index(layer_idx)
                    done_event.set()
            except Exception as exc:
                if self.prefill_stream_errors[layer_idx] is None:
                    self.prefill_stream_errors[layer_idx] = exc
                done_event.set()
            finally:
                free_slots = self.prefill_stream_free_slots[layer_idx]
                if free_slots is not None and slot_id is not None:
                    free_slots.put(slot_id)

    def _ensure_prefill_stream_worker(self, layer_idx: int):
        if self.prefill_stream_queues[layer_idx] is None:
            self.prefill_stream_queues[layer_idx] = queue.SimpleQueue()

        worker = self.prefill_stream_workers[layer_idx]
        if worker is not None and worker.is_alive():
            return

        self.prefill_stream_stop_events[layer_idx].clear()
        worker = threading.Thread(
            target=self._prefill_stream_worker_loop,
            args=(layer_idx,),
            daemon=True,
            name=f"imi-prefill-consumer-{layer_idx}",
        )
        worker.start()
        self.prefill_stream_workers[layer_idx] = worker

    def _shutdown_stream_workers(self):
        for layer_idx in range(self.layer_num):
            worker = self.prefill_stream_workers[layer_idx]
            worker_queue = self.prefill_stream_queues[layer_idx]
            if worker is None or worker_queue is None:
                continue
            if not worker.is_alive():
                continue
            self.prefill_stream_stop_events[layer_idx].set()
            worker_queue.put(None)
            worker.join()
            self.prefill_stream_workers[layer_idx] = None

    def cleanup(self):
        """Explicitly release all resources. Call this before destroying the cache."""
        # 1. Shutdown background threads first (most critical)
        self._shutdown_stream_workers()
        if hasattr(self, "_kmeans_scheduler") and self._kmeans_scheduler is not None:
            self._kmeans_scheduler.shutdown()

        # 2. Clear GPU tensor dicts
        self.execution_buffer_keys_dict.clear()
        self.execution_buffer_values_dict.clear()
        self.valid_lengths_dict.clear()
        self.static_len_tensor_dict.clear()
        self.static_lengths_dict.clear()
        self.similarity_buffer_dict.clear()

        # 3. Clear GPU tensor lists
        self.steady_zone_keys.clear()
        self.steady_zone_values.clear()
        self.cache_keys.clear()
        self.cache_values.clear()
        self.cluster_sizes_gpu.clear()
        self.centroids.clear()
        self.centroids_mask.clear()

        # 4. Clear pinned memory lists
        self.list_keys.clear()
        self.list_values.clear()
        self.hit_unit_idices.clear()
        self.hit_unit_sizes.clear()
        self.hit_unit_sizes_cumsum.clear()
        self.hit_num_units.clear()
        self.miss_unit_idices.clear()
        self.miss_unit_sizes.clear()
        self.miss_unit_sizes_cumsum.clear()
        self.miss_num_units.clear()
        self.update_buffer_indices.clear()
        self.update_unit_sizes.clear()
        self.update_cache_indices.clear()
        self.update_num_units.clear()
        self.cluster_ids.clear()

        # 5. Clear CPU metadata
        self.cluster_sizes_cpu.clear()
        self.cluster_offsets_cpu.clear()
        self.centroids_cpu.clear()
        self.centroids_mask_cpu.clear()

        # 6. Clear streaming prefill resources
        for i in range(self.layer_num):
            self.prefill_stream_stage_keys[i] = None
            self.prefill_stream_stage_values[i] = None
            self.prefill_stream_free_slots[i] = None
            self.prefill_stream_queues[i] = None
            self.prefill_stream_copy_streams[i] = None
            self.prefill_stream_main_events[i] = None
            self.prefill_stream_copy_events[i] = None

        # 7. Clear wave buffers and IMI pipelines
        for i in range(self.layer_num):
            self.adpimi_index[i] = None
            self.imi_pipelines[i] = None

        # 8. Shutdown thread pool
        if hasattr(self, 'thread_pool') and self.thread_pool is not None:
            try:
                del self.thread_pool
            except Exception:
                pass
            self.thread_pool = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, start_bdx):
        bsz, seq_len, _, _ = key_states.shape
        if bsz != 1:
            raise ValueError("IMI cache currently supports batch_size=1 for prefilling.")

        valid_start = self.valid_start_list[start_bdx]

        if self.build_index_when_prefilling:
            middle_start = valid_start + self.static_pattern_start
            middle_end = seq_len - self.static_pattern_end
            middle_len = max(middle_end - middle_start, 0)
            if middle_len <= 0:
                raise ValueError("IMI cache requires a non-empty middle segment.")
            self._expected_middle_len[layer_idx] = int(middle_len)

            if not self.layer_started[layer_idx]:
                if self.prefill_chunk_size > 0 and middle_len > self.prefill_chunk_size:
                    self.imi_pipelines[layer_idx].start_index_chunked(
                        key_states,
                        value_states,
                        middle_start,
                        middle_end,
                        self.prefill_chunk_size,
                    )
                else:
                    middle_keys = self._prepare_prefill_chunk(key_states, middle_start, middle_end)
                    middle_values = self._prepare_prefill_chunk(value_states, middle_start, middle_end)
                    self.imi_pipelines[layer_idx].start_index(
                        middle_keys,
                        middle_values,
                    )
                self.layer_started[layer_idx] = True

            end_bdx = start_bdx + bsz
            self.steady_zone_keys[layer_idx][start_bdx:end_bdx, :, : self.static_pattern_start, :] = (
                key_states[:, valid_start : valid_start + self.static_pattern_start, :, :].transpose(1, 2)
            )
            self.steady_zone_values[layer_idx][start_bdx:end_bdx, :, : self.static_pattern_start, :] = (
                value_states[:, valid_start : valid_start + self.static_pattern_start, :, :].transpose(1, 2)
            )

            self.steady_zone_keys[layer_idx][
                start_bdx:end_bdx, :, self.static_pattern_start : self.static_pattern_total, :
            ] = key_states[:, seq_len - self.static_pattern_end : seq_len, :, :].transpose(1, 2)
            self.steady_zone_values[layer_idx][
                start_bdx:end_bdx, :, self.static_pattern_start : self.static_pattern_total, :
            ] = value_states[:, seq_len - self.static_pattern_end : seq_len, :, :].transpose(1, 2)
        else:
            end_bdx = start_bdx + bsz
            self.steady_zone_keys[layer_idx][start_bdx:end_bdx, :, :seq_len, :].copy_(
                key_states.transpose(1, 2)
            )
            self.steady_zone_values[layer_idx][start_bdx:end_bdx, :, :seq_len, :].copy_(
                value_states.transpose(1, 2)
            )

        if (layer_idx == self.layer_num - 1) and (start_bdx + bsz == self.batch_size):
            self.context += seq_len
            if self.build_index_when_prefilling:
                self.attn_func = self.sparse_attention
            else:
                self.static_pattern_total = seq_len

        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]

    def sync(self, layer_idx, start_bdx):
        if not self.build_index_when_prefilling:
            return None
        if not self.prefill_stream_started[layer_idx]:
            return None

        if not self.allocated:
            copy_stream = self.prefill_stream_copy_streams[layer_idx]
            if copy_stream is not None:
                copy_stream.synchronize()

        done_event = self.prefill_stream_done_events[layer_idx]
        if done_event.is_set():
            self._raise_prefill_stream_error(layer_idx)
            if not self.layer_ready[layer_idx]:
                raise RuntimeError(f"IMI streaming prefill did not finish for layer {layer_idx}")
        return None

    def capture_cuda_graph(self):
        return None

    def prepare_cache(self, skip_prefetch: bool = False):
        if not self.build_index_when_prefilling:
            return
        if self.cache_prepared:
            return

        profile_prepare = os.getenv("IMI_PROFILE_PREPARE_CACHE", "0") == "1"
        if profile_prepare:
            start_ts = time.perf_counter()
            last_ts = start_ts

            def log_step(stage: str) -> None:
                nonlocal last_ts
                now = time.perf_counter()
                elapsed_ms = (now - last_ts) * 1000.0
                print(f"[IMI prepare_cache] {stage}={elapsed_ms:.2f} ms", flush=True)
                last_ts = now

        n_tokens = max(self.list_stride, 1)
        total_clusters = max(int(n_tokens / 16), 1)
        if self.subspace_parts == 4:
            total_clusters_f = max(n_tokens / 16.0, 1.0)
            k = max(int(round(total_clusters_f ** 0.25)), 2)
            self.n_centroids = k ** 4 + 1
        else:
            k = max(int(math.sqrt(total_clusters)), 32)
            self.n_centroids = k * k + 1
        self.padding_cluster_id = self.n_centroids - 1
        self.nprobe = max(round((self.n_centroids - 1) * self.retrieval_budget), 1)
        self.nprobe = min(self.nprobe, self.n_centroids - 1)
        requested_retrieval_tokens = max(int(self.retrieval_budget * self.list_stride), 1)
        self.max_retrieval_tokens = requested_retrieval_tokens
        self.budget_tokens = self.max_retrieval_tokens
        self.max_retrieval_pages = self._tokens_to_pages(self.max_retrieval_tokens)

        cache_cluster_num = (
            round(self.n_centroids * self.cache_ratio) if self.cache_ratio > 0.0 else self.nprobe * 3
        )
        requested_cache_pages = max(cache_cluster_num * self.pages_per_cluster, 1)
        self.cache_size = self._resolve_budget_pages(requested_cache_pages)
        self.cache_budget_pages = self.cache_size

        requested_buffer_pages = max(
            self.buffer_cluster_num,
            self._tokens_to_pages(self.max_retrieval_tokens),
        )
        self.buffer_size = self._resolve_budget_pages(requested_buffer_pages)
        self.buffer_budget_pages = self.buffer_size

        self.execution_stride = self.buffer_size * self.page_size + self.static_stride
        self.retrieval_execution_stride = self.buffer_size * self.page_size

        if profile_prepare:
            log_step("sizes")

        if self.profile_cache_stats or self.profile_block_hit_rate:
            cache_tokens = self.cache_size * self.page_size
            buffer_tokens = self.buffer_size * self.page_size
            print(
                "[IMI cache] "
                f"n_centroids={self.n_centroids} nprobe={self.nprobe} "
                f"list_stride={self.list_stride} static_stride={self.static_stride} "
                f"cache_pages={self.cache_size} buffer_pages={self.buffer_size} "
                f"cache_tokens={cache_tokens} buffer_tokens={buffer_tokens} "
                f"retrieval_execution_tokens={self.retrieval_execution_stride} "
                f"retrieval_tokens={self.max_retrieval_tokens} retrieval_pages={self.max_retrieval_pages}"
            )

        if self.prefetch_enabled and self.prefetch_ratio > 0:
            self.prefetch_k = max(int(self.nprobe * self.prefetch_ratio), 1)
            self.prefetch_k = min(self.prefetch_k, self.n_centroids - 1)
        else:
            self.prefetch_enabled = False

        thread_pool_pointer = self.thread_pool.get()
        for ldx in range(self.layer_num):
            self.adpimi_index[ldx] = self.adpimi_index_cls(
                self.batch_size,
                self.kv_head,
                self.head_dim,
                self.nprobe,
                0,
                self.page_size,
                self.n_centroids,
                self.buffer_size,
                self.cache_size,
                self.core,
                thread_pool_pointer,
            )

        if profile_prepare:
            log_step("adpimi_index")

        if self.async_update_enabled and not self.allocated:
            self.async_update_enabled = False

        if self.async_update_enabled:
            for device in self.device_list:
                if device not in self.async_update_streams:
                    self.async_update_streams[device] = torch.cuda.Stream(device)
            for ldx in range(self.layer_num):
                device = self.layer_mapping[str(ldx)]
                if self.centroids[ldx] is None or self.cluster_sizes_gpu[ldx] is None:
                    continue
                self.centroids_shadow[ldx] = self.centroids[ldx].clone()
                self.centroid_counts[ldx] = self.cluster_sizes_gpu[ldx].to(torch.int32).clone()
                self.delta_counts[ldx] = torch.zeros_like(self.centroid_counts[ldx])
                self.delta_sums[ldx] = torch.zeros(
                    (self.batch_groups, self.n_centroids, self.head_dim),
                    dtype=torch.float32,
                    device=device,
                )
                self.delta_keys[ldx] = torch.zeros(
                    (self.batch_groups, self.async_update_delta_tokens, self.head_dim),
                    dtype=self.dtype,
                    device=device,
                )
                self.delta_values[ldx] = torch.zeros(
                    (self.batch_groups, self.async_update_delta_tokens, self.head_dim),
                    dtype=self.dtype,
                    device=device,
                )
                self.delta_cluster_ids[ldx] = torch.full(
                    (self.batch_groups, self.async_update_delta_tokens),
                    self.padding_cluster_id,
                    dtype=torch.int64,
                    device=device,
                )
                self.delta_counts_gpu[ldx] = torch.zeros(
                    (self.batch_groups),
                    dtype=torch.int32,
                    device=device,
                )
                self.async_update_buffers[ldx] = {
                    "keys": torch.empty(
                        (self.batch_groups, self.async_update_batch, self.head_dim),
                        dtype=self.dtype,
                        device=device,
                    ),
                    "values": torch.empty(
                        (self.batch_groups, self.async_update_batch, self.head_dim),
                        dtype=self.dtype,
                        device=device,
                    ),
                    "count": 0,
                    "event": torch.cuda.Event(),
                    "delta_event": torch.cuda.Event(),
                    "delta_write_pos": torch.zeros(
                        (self.batch_groups),
                        dtype=torch.int64,
                        device=device,
                    ),
                }

        if profile_prepare:
            log_step("async_update")

        for _ in range(self.layer_num):
            self.hit_unit_idices.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.hit_unit_sizes.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.hit_unit_sizes_cumsum.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.hit_num_units.append(
                torch.zeros((self.batch_groups), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.miss_unit_idices.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.miss_unit_sizes.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.miss_unit_sizes_cumsum.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.miss_num_units.append(
                torch.zeros((self.batch_groups), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.update_buffer_indices.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.update_unit_sizes.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.update_cache_indices.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.update_num_units.append(
                torch.zeros((self.batch_groups), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.cluster_ids.append(
                torch.empty((self.batch_groups, self.nprobe), dtype=torch.int64, pin_memory=True).contiguous()
            )

        if profile_prepare:
            log_step("cpu_buffers")

        placeholder = torch.empty((self.batch_groups, 0, self.head_dim), dtype=self.dtype, pin_memory=True)
        placeholder_list = placeholder.unsqueeze(0)

        for ldx in range(self.layer_num):
            self.adpimi_index[ldx].set_indices(
                self.hit_unit_idices[ldx],
                self.hit_unit_sizes[ldx],
                self.hit_unit_sizes_cumsum[ldx],
                self.hit_num_units[ldx],
                self.miss_unit_idices[ldx],
                self.miss_unit_sizes[ldx],
                self.miss_unit_sizes_cumsum[ldx],
                self.miss_num_units[ldx],
                self.update_buffer_indices[ldx],
                self.update_unit_sizes[ldx],
                self.update_cache_indices[ldx],
                self.update_num_units[ldx],
                self.cluster_ids[ldx],
            )
            list_keys = self.list_keys[ldx] if self.list_keys[ldx] is not None else placeholder_list
            list_values = self.list_values[ldx] if self.list_values[ldx] is not None else placeholder_list
            self.adpimi_index[ldx].set_kv(
                list_keys,
                list_values,
                placeholder,
                placeholder,
            )

        if profile_prepare:
            log_step("wave_bind")

        for ldx in range(self.layer_num):
            device = self.layer_mapping[str(ldx)]
            cluster_sizes = self.cluster_sizes_cpu[ldx]
            cluster_offsets = self.cluster_offsets_cpu[ldx]
            if cluster_sizes is None or cluster_offsets is None:
                cluster_sizes = torch.zeros(
                    (self.batch_groups, self.n_centroids),
                    dtype=torch.int32,
                    pin_memory=True,
                )
                cluster_offsets = torch.zeros(
                    (self.batch_groups, self.n_centroids + 1),
                    dtype=torch.int32,
                    pin_memory=True,
                )
                self.cluster_sizes_cpu[ldx] = cluster_sizes
                self.cluster_offsets_cpu[ldx] = cluster_offsets

            if self.cluster_sizes_gpu[ldx] is None:
                self.cluster_sizes_gpu[ldx] = cluster_sizes.to(device)
            if self.centroids_mask[ldx] is None:
                self.centroids_mask[ldx] = torch.ones(
                    (self.batch_groups, self.n_centroids),
                    dtype=torch.bool,
                    device=device,
                )
            if self.centroids[ldx] is None:
                self.centroids[ldx] = torch.zeros(
                    (self.batch_groups, self.n_centroids, self.head_dim),
                    dtype=torch.float32,
                    device=device,
                )

            centroids_cpu = self.centroids_cpu[ldx]
            if centroids_cpu is not None:
                self.centroids[ldx].copy_(centroids_cpu)
            centroids_mask_cpu = self.centroids_mask_cpu[ldx]
            if centroids_mask_cpu is not None:
                self.centroids_mask[ldx].copy_(centroids_mask_cpu)

            if os.getenv("IMI_DEBUG_INDEX_METADATA", "0") == "1":
                nonzero_clusters = int((cluster_sizes > 0).sum().item()) if cluster_sizes is not None else 0
                unmasked_clusters = int((~self.centroids_mask[ldx]).sum().item()) if self.centroids_mask[ldx] is not None else 0
                print(
                    json.dumps({
                        "tag": "IMI_META_PREPARE",
                        "layer_idx": int(ldx),
                        "nonzero_clusters": nonzero_clusters,
                        "unmasked_clusters": unmasked_clusters,
                        "n_centroids": int(self.n_centroids),
                    }, ensure_ascii=False),
                    flush=True,
                )

            if cluster_sizes is not None and self.centroids_mask[ldx] is not None:
                nonzero_clusters = int((cluster_sizes > 0).sum().item())
                unmasked_clusters = int((~self.centroids_mask[ldx]).sum().item())
                if self.layer_metadata[ldx] is not None and len(self.layer_metadata[ldx]) > 0 and nonzero_clusters == 0 and unmasked_clusters == 0:
                    raise RuntimeError(
                        f"AdaptiveIMI invalid decode metadata on layer {ldx}: "
                        "CPU metadata exists but GPU metadata remains empty."
                    )

            self.adpimi_index[ldx].set_cluster_metadata(cluster_sizes, cluster_offsets, 0)

        if profile_prepare:
            log_step("centroids")

        for ldx in range(self.layer_num):
            device = self.layer_mapping[str(ldx)]
            self.cache_keys.append(
                torch.zeros(
                    (self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                    dtype=self.dtype,
                    device=device,
                ).contiguous()
            )
            self.cache_values.append(
                torch.zeros(
                    (self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                    dtype=self.dtype,
                    device=device,
                ).contiguous()
            )

        if profile_prepare:
            log_step("cache_kv")

        for device_idx in self.device_list:
            self.execution_buffer_keys_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.execution_stride, 1, self.head_dim),
                dtype=self.dtype,
                device=device_idx,
            ).contiguous()
            self.execution_buffer_values_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.execution_stride, 1, self.head_dim),
                dtype=self.dtype,
                device=device_idx,
            ).contiguous()
            self.valid_lengths_dict[device_idx] = torch.zeros(
                (self.batch_groups),
                dtype=torch.int32,
                device=device_idx,
            ).contiguous()
            self.static_len_tensor_dict[device_idx] = torch.tensor(
                self.static_pattern_total,
                dtype=torch.int32,
                device=device_idx,
            )
            self.static_lengths_dict[device_idx] = torch.zeros(
                (self.batch_groups),
                dtype=torch.int32,
                device=device_idx,
            ).contiguous()
            self.similarity_buffer_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.n_centroids),
                dtype=torch.float32,
                device=device_idx,
            ).contiguous()

        if profile_prepare:
            log_step("execution_buffers")


        self.attn_func = self.sparse_attention
        self.cache_prepared = True
        if self.prefetch_enabled and not skip_prefetch:
            self._warmup_prefetch()

        if profile_prepare:
            log_step("prefetch")
            total_ms = (time.perf_counter() - start_ts) * 1000.0
            print(f"[IMI prepare_cache] total={total_ms:.2f} ms", flush=True)

    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        self.steady_zone_keys[layer_idx][:, :, self.static_pattern_total, :] = key_states[:, 0, :, :]
        self.steady_zone_values[layer_idx][:, :, self.static_pattern_total, :] = value_states[:, 0, :, :]

        if layer_idx == 0:
            self.decode_debug_step += 1

        if self.async_update_enabled:
            self._schedule_async_update(layer_idx, key_states, value_states)

        if layer_idx == self.layer_num - 1:
            self.context += 1
            self.static_pattern_total += 1

        return None, None

    def dense_attention(self, queries, layer_idx, static_len):
        attn_out = weighted_flash_decoding(
            queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
            self.steady_zone_keys[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
            self.steady_zone_values[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
            previous_out=None,
            previous_lse=None,
            cache_seqlens=static_len,
            return_softmax_lse=False,
        )
        return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)

    def _tokens_to_pages(self, token_budget: int) -> int:
        if token_budget <= 0:
            return 0
        return max((int(token_budget) + self.page_size - 1) // self.page_size, 1)

    def _resolve_budget_pages(self, requested_pages: int, minimum_pages: int = 1) -> int:
        return max(int(requested_pages), minimum_pages)

    def _set_cluster_ids(self, layer_idx: int, cluster_ids: torch.Tensor) -> None:
        target = self.cluster_ids[layer_idx]
        target.fill_(self.padding_cluster_id)
        width = cluster_ids.size(1)
        if width > target.size(1):
            raise ValueError(f"cluster tile width {width} exceeds allocated nprobe {target.size(1)}")
        if width > 0:
            target[:, :width].copy_(cluster_ids)

    def _build_cluster_tiles(self, layer_idx: int, selected: torch.Tensor) -> List[torch.Tensor]:
        if selected.numel() == 0:
            return []

        selected_cpu = selected.to(device="cpu", dtype=torch.int64)

        cluster_sizes = self.cluster_sizes_cpu[layer_idx]
        if cluster_sizes is None:
            return []

        selected_sizes = torch.gather(cluster_sizes, 1, selected_cpu.clamp_max(self.padding_cluster_id))
        selected_sizes = torch.where(selected_cpu == self.padding_cluster_id, 0, selected_sizes)
        selected_pages = torch.div(selected_sizes + self.page_size - 1, self.page_size, rounding_mode="floor")
        selected_pages = torch.where(selected_sizes > 0, torch.clamp(selected_pages, min=1), 0)
        if os.getenv("IMI_DEBUG_OVERSIZED", "0") == "1":
            oversized_mask = (selected_pages > self.buffer_size) & (selected_cpu != self.padding_cluster_id)
            if torch.any(oversized_mask):
                oversized_count = int(oversized_mask.sum().item())
                max_pages = int(selected_pages[oversized_mask].max().item())
                print(
                    "[IMI oversized] "
                    f"layer={layer_idx} count={oversized_count} max_pages={max_pages} "
                    f"buffer_pages={self.buffer_size}"
                )
        prefix_pages = torch.cumsum(selected_pages, dim=1)

        positions = torch.arange(self.nprobe, dtype=torch.int64).unsqueeze(0).expand(self.batch_groups, -1)
        group_offsets = torch.zeros(self.batch_groups, dtype=torch.int64)
        group_counts = (selected_cpu != self.padding_cluster_id).sum(dim=1, dtype=torch.int64)
        tiles: List[torch.Tensor] = []

        while torch.any(group_offsets < group_counts):
            start_valid = group_offsets < group_counts
            start_minus_one = torch.clamp(group_offsets - 1, min=0)
            start_prefix = torch.gather(prefix_pages, 1, start_minus_one.unsqueeze(1)).squeeze(1)
            start_prefix = torch.where(group_offsets > 0, start_prefix, torch.zeros_like(start_prefix))

            relative_pages = prefix_pages - start_prefix.unsqueeze(1)
            candidate_mask = (
                (positions >= group_offsets.unsqueeze(1))
                & (positions < group_counts.unsqueeze(1))
                & (relative_pages <= self.buffer_size)
            )
            take = candidate_mask.sum(dim=1, dtype=torch.int64)

            force_take_mask = start_valid & (take == 0)
            if torch.any(force_take_mask):
                current_positions = group_offsets.unsqueeze(1)
                safe_current = torch.minimum(current_positions, torch.full_like(current_positions, self.nprobe - 1))
                current_pages = torch.gather(selected_pages, 1, safe_current).squeeze(1)

                can_force = current_pages <= self.buffer_size
                force_take_mask = force_take_mask & can_force

                if torch.any(force_take_mask & ~can_force):
                    oversized_count = (force_take_mask & ~can_force).sum().item()
                    if os.getenv("IMI_DEBUG", "0") == "1":
                        print(f"[IMI WARNING] layer={layer_idx} skipping {oversized_count} oversized clusters that exceed buffer_size")

                take = torch.where(force_take_mask, torch.ones_like(take), take)

            take = torch.minimum(take, group_counts - group_offsets)

            max_take = int(take.max().item())
            if max_take <= 0:
                break

            gather_positions = group_offsets.unsqueeze(1) + torch.arange(max_take, dtype=torch.int64).unsqueeze(0)
            safe_positions = torch.minimum(gather_positions, torch.full_like(gather_positions, self.nprobe - 1))
            payload = torch.gather(selected_cpu, 1, safe_positions)
            payload_mask = torch.arange(max_take, dtype=torch.int64).unsqueeze(0) < take.unsqueeze(1)

            tile = torch.full(
                (self.batch_groups, max_take),
                self.padding_cluster_id,
                dtype=torch.int64,
                pin_memory=True,
            )
            tile[payload_mask] = payload[payload_mask]
            tiles.append(tile)
            group_offsets = group_offsets + take

        return tiles

    def _apply_async_centroid_updates(self, layer_idx: int, device: str) -> None:
        if not self.async_update_enabled:
            return
        buffer = self.async_update_buffers[layer_idx]
        shadow = self.centroids_shadow[layer_idx]
        if buffer is None or shadow is None:
            return
        event = buffer.get("event")
        if event is None or not event.query():
            return
        stream = torch.cuda.current_stream(device)
        stream.wait_event(event)
        self.centroids[layer_idx].copy_(shadow)

    def _append_delta_to_execution(
        self,
        layer_idx: int,
        device: str,
        selected: torch.Tensor,
    ) -> None:
        if not self.async_update_enabled:
            return
        buffer = self.async_update_buffers[layer_idx]
        delta_keys = self.delta_keys[layer_idx]
        delta_values = self.delta_values[layer_idx]
        delta_cluster_ids = self.delta_cluster_ids[layer_idx]
        delta_counts = self.delta_counts_gpu[layer_idx]
        if (
            buffer is None
            or delta_keys is None
            or delta_values is None
            or delta_cluster_ids is None
            or delta_counts is None
        ):
            return
        if int(delta_counts.max().item()) <= 0:
            return
        delta_event = buffer.get("delta_event")
        if delta_event is not None and not delta_event.query():
            torch.cuda.current_stream(device).wait_event(delta_event)

        selected_gpu = selected.to(device=device, non_blocking=True)
        capacity = delta_keys.size(1)
        positions = torch.arange(capacity, device=device).unsqueeze(0)
        valid_mask = positions < delta_counts.unsqueeze(1)
        match_mask = (delta_cluster_ids.unsqueeze(-1) == selected_gpu.unsqueeze(1)).any(dim=-1)
        match_mask = match_mask & valid_mask

        exec_keys = self.execution_buffer_keys_dict[device]
        exec_values = self.execution_buffer_values_dict[device]
        valid_lengths = self.valid_lengths_dict[device]
        max_len = int(exec_keys.size(1))

        for group_idx in range(self.batch_groups):
            if not torch.any(match_mask[group_idx]):
                continue
            delta_k = delta_keys[group_idx, match_mask[group_idx], :]
            delta_v = delta_values[group_idx, match_mask[group_idx], :]
            if delta_k.numel() == 0:
                continue
            start = int(valid_lengths[group_idx].item())
            if start >= max_len:
                continue
            available = max_len - start
            if delta_k.size(0) > available:
                delta_k = delta_k[:available]
                delta_v = delta_v[:available]
            end = start + delta_k.size(0)
            exec_keys[group_idx, start:end, 0, :].copy_(delta_k)
            exec_values[group_idx, start:end, 0, :].copy_(delta_v)
            valid_lengths[group_idx] = end

    def _schedule_async_update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        if not self.async_update_enabled:
            return
        buffer = self.async_update_buffers[layer_idx]
        shadow = self.centroids_shadow[layer_idx]
        delta_counts = self.delta_counts[layer_idx]
        delta_sums = self.delta_sums[layer_idx]
        centroid_counts = self.centroid_counts[layer_idx]
        delta_keys = self.delta_keys[layer_idx]
        delta_values = self.delta_values[layer_idx]
        delta_cluster_ids = self.delta_cluster_ids[layer_idx]
        delta_counts_gpu = self.delta_counts_gpu[layer_idx]
        if (
            buffer is None
            or shadow is None
            or delta_counts is None
            or delta_sums is None
            or centroid_counts is None
            or delta_keys is None
            or delta_values is None
            or delta_cluster_ids is None
            or delta_counts_gpu is None
        ):
            return
        event = buffer.get("event")
        if event is not None and buffer["count"] == 0 and not event.query():
            return
        device = self.layer_mapping[str(layer_idx)]
        keys_view = key_states[:, 0, :, :].reshape(self.batch_groups, self.head_dim)
        values_view = value_states[:, 0, :, :].reshape(self.batch_groups, self.head_dim)
        slot = int(buffer["count"])
        buffer["keys"][:, slot, :].copy_(keys_view, non_blocking=True)
        buffer["values"][:, slot, :].copy_(values_view, non_blocking=True)
        buffer["count"] = slot + 1
        if buffer["count"] < self.async_update_batch:
            return

        count = int(buffer["count"])
        update_stream = self.async_update_streams[device]
        update_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(update_stream):
            keys_batch = buffer["keys"][:, :count, :].clone()
            values_batch = buffer["values"][:, :count, :].clone()
            keys_fp32 = keys_batch.float()
            similarities = torch.bmm(keys_fp32, shadow.transpose(1, 2))
            cluster_ids = torch.argmax(similarities, dim=-1)
            ones = torch.ones_like(cluster_ids, dtype=torch.int32)
            delta_counts.scatter_add_(1, cluster_ids, ones)
            delta_sums.scatter_add_(
                1,
                cluster_ids.unsqueeze(-1).expand(-1, -1, self.head_dim),
                keys_fp32,
            )
            update_mask = delta_counts >= self.async_update_threshold
            new_counts = centroid_counts + delta_counts
            new_centroids = (
                shadow * centroid_counts.unsqueeze(-1) + delta_sums
            ) / torch.clamp(new_counts, min=1).unsqueeze(-1)
            shadow.copy_(torch.where(update_mask.unsqueeze(-1), new_centroids, shadow))
            centroid_counts.copy_(torch.where(update_mask, new_counts, centroid_counts))
            delta_counts.masked_fill_(update_mask, 0)
            delta_sums.masked_fill_(update_mask.unsqueeze(-1), 0.0)
            buffer["event"].record(update_stream)

            delta_event = buffer.get("delta_event")
            delta_write_pos = buffer.get("delta_write_pos")
            if delta_event is not None and delta_write_pos is not None:
                capacity = delta_keys.size(1)
                positions = (
                    delta_write_pos.unsqueeze(1)
                    + torch.arange(count, device=device).unsqueeze(0)
                ) % capacity
                for group_idx in range(self.batch_groups):
                    pos = positions[group_idx]
                    delta_keys[group_idx, pos, :].copy_(keys_batch[group_idx])
                    delta_values[group_idx, pos, :].copy_(values_batch[group_idx])
                    delta_cluster_ids[group_idx, pos].copy_(cluster_ids[group_idx])
                delta_write_pos.copy_((delta_write_pos + count) % capacity)
                delta_counts_gpu.copy_(torch.clamp(delta_counts_gpu + count, max=capacity))
                delta_event.record(update_stream)
        buffer["count"] = 0

    def _run_retrieval_tile(
        self,
        layer_idx: int,
        device: str,
        tile_cluster_ids: torch.Tensor,
        profile_decode: bool = False,
    ) -> Optional[dict]:
        if profile_decode:
            torch.cuda.synchronize(device)
            lookup_start = time.perf_counter()

        self._set_cluster_ids(layer_idx, tile_cluster_ids)
        self.adpimi_index[layer_idx].batch_access()

        if profile_decode:
            torch.cuda.synchronize(device)
            lookup_end = time.perf_counter()

        if self.profile_block_hit_rate:
            hit_blocks, miss_blocks = self.adpimi_index[layer_idx].get_last_block_stats()
            self.profile_block_hits += hit_blocks
            self.profile_block_misses += miss_blocks
            self.profile_layer_hit_rate_count += 1

        if profile_decode:
            torch.cuda.synchronize(device)
            gather_start = time.perf_counter()

        gather_copy_and_concat_retrieval(
            self.list_keys[layer_idx],
            self.cache_keys[layer_idx],
            self.execution_buffer_keys_dict[device],
            self.list_values[layer_idx],
            self.cache_values[layer_idx],
            self.execution_buffer_values_dict[device],
            self.miss_unit_idices[layer_idx],
            self.miss_unit_sizes[layer_idx],
            self.miss_unit_sizes_cumsum[layer_idx],
            self.miss_num_units[layer_idx],
            self.hit_unit_idices[layer_idx],
            self.hit_unit_sizes[layer_idx],
            self.hit_unit_sizes_cumsum[layer_idx],
            self.hit_num_units[layer_idx],
            self.valid_lengths_dict[device],
            self.batch_groups,
            self.list_stride,
            self.cache_size,
            self.retrieval_execution_stride,
            self.buffer_size,
        )

        if profile_decode:
            torch.cuda.synchronize(device)
            h2d_start = time.perf_counter()

        self.adpimi_index[layer_idx].sync()

        if profile_decode:
            torch.cuda.synchronize(device)
            h2d_end = time.perf_counter()

        zero_static_len = self.static_len_tensor_dict[device]
        zero_static_len.zero_()
        gather_copy_and_scatter(
            self.execution_buffer_keys_dict[device],
            self.cache_keys[layer_idx],
            self.execution_buffer_values_dict[device],
            self.cache_values[layer_idx],
            self.update_buffer_indices[layer_idx],
            self.update_unit_sizes[layer_idx],
            self.update_cache_indices[layer_idx],
            self.update_num_units[layer_idx],
            self.batch_groups,
            self.retrieval_execution_stride,
            self.cache_size,
            self.buffer_size,
            zero_static_len,
        )

        if not profile_decode:
            return None

        torch.cuda.synchronize(device)
        gather_end = time.perf_counter()

        return {
            "lookup_ms": (lookup_end - lookup_start) * 1000.0,
            "gather_ms": (gather_end - gather_start) * 1000.0,
            "h2d_ms": (h2d_end - h2d_start) * 1000.0,
        }

    def _run_retrieval_full(
        self,
        layer_idx: int,
        device: str,
        cluster_ids: torch.Tensor,
        profile_decode: bool = False,
    ) -> Optional[dict]:
        if profile_decode:
            torch.cuda.synchronize(device)
            lookup_start = time.perf_counter()

        self._set_cluster_ids(layer_idx, cluster_ids)
        self.adpimi_index[layer_idx].batch_access()

        if profile_decode:
            torch.cuda.synchronize(device)
            lookup_end = time.perf_counter()

        if self.profile_block_hit_rate:
            hit_blocks, miss_blocks = self.adpimi_index[layer_idx].get_last_block_stats()
            self.profile_block_hits += hit_blocks
            self.profile_block_misses += miss_blocks
            self.profile_layer_hit_rate_count += 1

        if profile_decode:
            torch.cuda.synchronize(device)
            gather_start = time.perf_counter()

        gather_copy_and_concat(
            self.steady_zone_keys[layer_idx],
            self.list_keys[layer_idx],
            self.cache_keys[layer_idx],
            self.execution_buffer_keys_dict[device],
            self.steady_zone_values[layer_idx],
            self.list_values[layer_idx],
            self.cache_values[layer_idx],
            self.execution_buffer_values_dict[device],
            self.miss_unit_idices[layer_idx],
            self.miss_unit_sizes[layer_idx],
            self.miss_unit_sizes_cumsum[layer_idx],
            self.miss_num_units[layer_idx],
            self.hit_unit_idices[layer_idx],
            self.hit_unit_sizes[layer_idx],
            self.hit_unit_sizes_cumsum[layer_idx],
            self.hit_num_units[layer_idx],
            self.valid_lengths_dict[device],
            self.batch_groups,
            self.static_stride,
            self.list_stride,
            self.cache_size,
            self.execution_stride,
            self.buffer_size,
            self.static_len_tensor_dict[device],
        )

        if profile_decode:
            torch.cuda.synchronize(device)
            h2d_start = time.perf_counter()

        self.adpimi_index[layer_idx].sync()

        if profile_decode:
            torch.cuda.synchronize(device)
            h2d_end = time.perf_counter()

        gather_copy_and_scatter(
            self.execution_buffer_keys_dict[device],
            self.cache_keys[layer_idx],
            self.execution_buffer_values_dict[device],
            self.cache_values[layer_idx],
            self.update_buffer_indices[layer_idx],
            self.update_unit_sizes[layer_idx],
            self.update_cache_indices[layer_idx],
            self.update_num_units[layer_idx],
            self.batch_groups,
            self.execution_stride,
            self.cache_size,
            self.buffer_size,
            self.static_len_tensor_dict[device],
        )

        if not profile_decode:
            return None

        torch.cuda.synchronize(device)
        gather_end = time.perf_counter()

        return {
            "lookup_ms": (lookup_end - lookup_start) * 1000.0,
            "gather_ms": (gather_end - gather_start) * 1000.0,
            "h2d_ms": (h2d_end - h2d_start) * 1000.0,
        }

    def _select_clusters(self, layer_idx, similarities):
        topk = torch.topk(similarities, self.nprobe, dim=-1, largest=True, sorted=True)
        cluster_sizes = self.cluster_sizes_gpu[layer_idx]
        sizes = torch.gather(cluster_sizes, 1, topk.indices)
        valid = sizes > 0
        cumsum = torch.cumsum(sizes, dim=-1, dtype=torch.int32)
        keep = valid & ((cumsum - sizes) < self.max_retrieval_tokens)
        selected = topk.indices.masked_fill(~keep, self.padding_cluster_id)
        return selected

    def _warmup_prefetch(self) -> None:
        if self.prefetch_done or not self.prefetch_enabled or self.prefetch_k <= 0:
            return
        rng = torch.Generator()
        if self.prefetch_seed:
            rng.manual_seed(self.prefetch_seed)
        for layer_idx in range(self.layer_num):
            self.ensure_layer_ready(layer_idx)
            cluster_sizes = self.cluster_sizes_cpu[layer_idx]
            if cluster_sizes is None:
                continue

            if self.prefetch_mode != "random":
                # 目前只支持 random warmup，避免 largest 这类与 query 无关且偏置明显的策略。
                raise ValueError(f"Unsupported prefetch mode: {self.prefetch_mode}")

            prefetch_ids = torch.full(
                (self.batch_groups, self.prefetch_k),
                self.padding_cluster_id,
                dtype=torch.int64,
                pin_memory=True,
            )
            for group_idx in range(self.batch_groups):
                non_empty = torch.nonzero(
                    cluster_sizes[group_idx] > 0,
                    as_tuple=False,
                ).flatten()
                if non_empty.numel() == 0:
                    continue
                if non_empty.numel() <= self.prefetch_k:
                    chosen = non_empty
                else:
                    perm = torch.randperm(non_empty.numel(), generator=rng)
                    chosen = non_empty[perm[: self.prefetch_k]]
                prefetch_ids[group_idx, : chosen.numel()] = chosen

            selected = torch.full(
                (self.batch_groups, self.nprobe),
                self.padding_cluster_id,
                dtype=torch.int64,
                pin_memory=True,
            )
            selected[:, : self.prefetch_k].copy_(prefetch_ids)

            tiles = self._build_cluster_tiles(layer_idx, selected)
            if not tiles:
                continue

            device = self.layer_mapping[str(layer_idx)]
            for tile_cluster_ids in tiles[: self.prefetch_tiles_per_layer]:
                self._run_retrieval_tile(layer_idx, device, tile_cluster_ids)

        self.prefetch_done = True
        self.prefetch_enabled = False

    def ensure_layer_ready(self, layer_idx):
        if self.layer_ready[layer_idx] or not self.build_index_when_prefilling:
            return
        self._raise_prefill_stream_error(layer_idx)
        if not self.layer_started[layer_idx]:
            raise RuntimeError(f"IMI layer {layer_idx} not started")

        if self.prefill_stream_started[layer_idx]:
            done_event = self.prefill_stream_done_events[layer_idx]
            done_event.wait()
            self._raise_prefill_stream_error(layer_idx)
            if self.layer_ready[layer_idx]:
                return
            raise RuntimeError(f"IMI streaming prefill did not finish for layer {layer_idx}")

        self._finish_layer_index(layer_idx)

    def sparse_attention(self, queries, layer_idx, static_len):
        self.ensure_layer_ready(layer_idx)
        device = self.layer_mapping[str(layer_idx)]
        self._apply_async_centroid_updates(layer_idx, device)
        self.static_len_tensor_dict[device].fill_(static_len)
        self.static_lengths_dict[device].fill_(static_len)

        debug_decode = os.getenv("IMI_DEBUG_DECODE", "0") == "1"
        if debug_decode and layer_idx == 0:
            print(
                "[IMI DECODE]"
                f" step={self.decode_debug_step} layer={layer_idx} static_len={static_len}"
                f" list_stride={self.list_stride} static_stride={self.static_stride}"
                f" nprobe={self.nprobe}",
                flush=True,
            )

        profile = self.profile_decode and self.profile_decode_count < self.profile_decode_steps
        if profile and layer_idx == 0:
            self.profile_decode_step_idx += 1
            if self.profile_decode_step_idx < self.profile_decode_steps:
                self._profile_step_breakdown = {
                    "search_ms": 0.0,
                    "gather_ms": 0.0,
                    "attn_ms": 0.0,
                    "total_ms": 0.0,
                }
        if profile:
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()

        query_grouped = queries.view(self.batch_groups, self.group_size, self.head_dim).contiguous()
        similarities = imi_gpu_kernels.fused_query_group_similarities(
            query_grouped,
            self.centroids[layer_idx],
            self.similarity_buffer_dict[device],
        )
        similarities.masked_fill_(self.centroids_mask[layer_idx], float("-inf"))

        if profile:
            torch.cuda.synchronize(device)
            sim_time = time.perf_counter()

        selected = self._select_clusters(layer_idx, similarities)
        if debug_decode and layer_idx == 0:
            selected_count = int((selected != self.padding_cluster_id).sum().item())
            selected_sample = selected[0, : min(16, selected.shape[1])].tolist()
            print(
                "[IMI DECODE]"
                f" step={self.decode_debug_step} layer={layer_idx} selected_clusters={selected_count}"
                f" selected_sample={selected_sample}",
                flush=True,
            )
        max_tiles = int(os.getenv("IMI_MAX_TILES", "0"))
        selected_cpu = None
        tiles = None
        if max_tiles == 0:
            selected_cpu = selected.to(device="cpu", dtype=torch.int64)
            tiles = [selected_cpu]
            use_tiles = False
        else:
            tiles = self._build_cluster_tiles(layer_idx, selected)
            use_tiles = True

        # Debug: 打印 tile 信息
        if os.getenv("IMI_DEBUG_TILES", "0") == "1" and layer_idx == 0:
            base_selected = selected_cpu if selected_cpu is not None else selected
            num_selected = (base_selected != self.padding_cluster_id).sum().item()
            print(f"[IMI tiles] layer={layer_idx} num_tiles={len(tiles)} num_selected_clusters={num_selected} nprobe={self.nprobe}")
            for i, tile in enumerate(tiles):
                tile_size = (tile != self.padding_cluster_id).sum().item()
                print(f"  tile[{i}]: size={tile_size} shape={tile.shape}")

        if profile:
            torch.cuda.synchronize(device)
            select_time = time.perf_counter()

        lookup_ms = 0.0
        h2d_ms = 0.0
        gather_ms = 0.0
        attn_ms = 0.0

        query_view = queries.view(self.batch_groups, 1, self.group_size, self.head_dim)

        if not use_tiles:
            if profile:
                static_time = select_time

            self.valid_lengths_dict[device].zero_()

            retrieval_timing = self._run_retrieval_full(
                layer_idx,
                device,
                selected_cpu,
                profile_decode=profile,
            )

            if debug_decode and layer_idx == 0:
                valid_lengths = self.valid_lengths_dict[device]
                max_len = int(valid_lengths.max().item()) if valid_lengths.numel() else 0
                min_len = int(valid_lengths.min().item()) if valid_lengths.numel() else 0
                mean_len = float(valid_lengths.float().mean().item()) if valid_lengths.numel() else 0.0
                print(
                    "[IMI DECODE]"
                    f" step={self.decode_debug_step} layer={layer_idx} valid_lengths[min,max,mean]=[{min_len},{max_len},{mean_len:.1f}]",
                    flush=True,
                )

            if retrieval_timing is not None:
                lookup_ms += retrieval_timing["lookup_ms"]
                h2d_ms += retrieval_timing["h2d_ms"]
                gather_ms += retrieval_timing["gather_ms"]

            self._append_delta_to_execution(layer_idx, device, selected_cpu)

            if profile:
                torch.cuda.synchronize(device)
                attn_start = time.perf_counter()

            acc_out = weighted_flash_decoding(
                query_view,
                self.execution_buffer_keys_dict[device],
                self.execution_buffer_values_dict[device],
                previous_out=None,
                previous_lse=None,
                cache_seqlens=self.valid_lengths_dict[device],
                return_softmax_lse=False,
            )

            if profile:
                torch.cuda.synchronize(device)
                attn_ms += (time.perf_counter() - attn_start) * 1000.0
        else:
            static_out, static_lse = weighted_flash_decoding(
                queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
                self.steady_zone_keys[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
                self.steady_zone_values[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
                previous_out=None,
                previous_lse=None,
                cache_seqlens=self.static_lengths_dict[device],
                return_softmax_lse=True,
            )

            if profile:
                torch.cuda.synchronize(device)
                static_time = time.perf_counter()

            acc_out, acc_lse = static_out, static_lse

            if debug_decode and layer_idx == 0:
                static_lengths = self.static_lengths_dict[device]
                static_min = int(static_lengths.min().item()) if static_lengths.numel() else 0
                static_max = int(static_lengths.max().item()) if static_lengths.numel() else 0
                print(
                    "[IMI DECODE]"
                    f" step={self.decode_debug_step} layer={layer_idx} static_lengths[min,max]=[{static_min},{static_max}]"
                    f" num_tiles={len(tiles)}",
                    flush=True,
                )

            for tile_idx, tile_cluster_ids in enumerate(tiles):
                # 重置 valid_lengths 为 0，确保每个 tile 独立处理
                self.valid_lengths_dict[device].zero_()

                retrieval_timing = self._run_retrieval_tile(
                    layer_idx,
                    device,
                    tile_cluster_ids,
                    profile_decode=profile,
                )

                if retrieval_timing is not None:
                    lookup_ms += retrieval_timing["lookup_ms"]
                    h2d_ms += retrieval_timing["h2d_ms"]
                    gather_ms += retrieval_timing["gather_ms"]

                if debug_decode and layer_idx == 0:
                    valid_lengths = self.valid_lengths_dict[device]
                    max_len = int(valid_lengths.max().item()) if valid_lengths.numel() else 0
                    min_len = int(valid_lengths.min().item()) if valid_lengths.numel() else 0
                    mean_len = float(valid_lengths.float().mean().item()) if valid_lengths.numel() else 0.0
                    tile_size = int((tile_cluster_ids != self.padding_cluster_id).sum().item())
                    print(
                        "[IMI DECODE]"
                        f" step={self.decode_debug_step} layer={layer_idx} tile={tile_idx} tile_selected={tile_size}"
                        f" valid_lengths[min,max,mean]=[{min_len},{max_len},{mean_len:.1f}]",
                        flush=True,
                    )

                self._append_delta_to_execution(layer_idx, device, tile_cluster_ids)

                if profile:
                    torch.cuda.synchronize(device)
                    attn_start = time.perf_counter()

                acc_out, acc_lse = weighted_flash_decoding(
                    query_view,
                    self.execution_buffer_keys_dict[device],
                    self.execution_buffer_values_dict[device],
                    previous_out=acc_out,
                    previous_lse=acc_lse,
                    cache_seqlens=self.valid_lengths_dict[device],
                    return_softmax_lse=True,
                )

                if profile:
                    torch.cuda.synchronize(device)
                    attn_ms += (time.perf_counter() - attn_start) * 1000.0
        if self.profile_block_hit_rate and layer_idx == self.layer_num - 1:
            self.profile_block_step += 1
            if self.profile_block_step % self.profile_block_hit_every == 0:
                total_blocks = self.profile_block_hits + self.profile_block_misses
                total_hit_rate = self.profile_block_hits / max(total_blocks, 1)
                print(
                    "[IMI hitrate] "
                    f"step={self.profile_block_step} "
                    f"total_hit_rate={total_hit_rate:.4f} "
                    f"hit_blocks={self.profile_block_hits} "
                    f"miss_blocks={self.profile_block_misses}"
                )
            self.profile_block_hits = 0
            self.profile_block_misses = 0
            self.profile_layer_hit_rate_count = 0

        if profile:
            torch.cuda.synchronize(device)
            end_time = time.perf_counter()

            search_ms = (sim_time - start_time) * 1000.0 + (select_time - sim_time) * 1000.0
            gather_total_ms = lookup_ms + h2d_ms + gather_ms
            total_ms = (end_time - start_time) * 1000.0

            self._profile_step_breakdown["search_ms"] += float(search_ms)
            self._profile_step_breakdown["gather_ms"] += float(gather_total_ms)
            self._profile_step_breakdown["attn_ms"] += float(attn_ms)
            self._profile_step_breakdown["total_ms"] += float(total_ms)

            if layer_idx == self.layer_num - 1:
                step_breakdown = dict(self._profile_step_breakdown)
                step_breakdown["other_ms"] = max(
                    step_breakdown["total_ms"]
                    - step_breakdown["search_ms"]
                    - step_breakdown["gather_ms"]
                    - step_breakdown["attn_ms"],
                    0.0,
                )

                for k, v in step_breakdown.items():
                    self.decode_breakdown_accum.setdefault(k, 0.0)
                    self.decode_breakdown_accum[k] += float(v)

                self.profile_decode_count += 1
                if self.profile_decode_count == self.profile_decode_steps:
                    self.decode_breakdown = {
                        k: v / float(self.profile_decode_steps) for k, v in self.decode_breakdown_accum.items()
                    }

                print(
                    "[IMI decode breakdown] step={} search={:.3f}ms gather={:.3f}ms "
                    "attn={:.3f}ms other={:.3f}ms total={:.3f}ms".format(
                        self.profile_decode_count,
                        step_breakdown["search_ms"],
                        step_breakdown["gather_ms"],
                        step_breakdown["attn_ms"],
                        step_breakdown["other_ms"],
                        step_breakdown["total_ms"],
                    )
                )

                if self.profile_decode_count == self.profile_decode_steps:
                    avg = self.decode_breakdown
                    print(
                        "[IMI decode breakdown][avg] search={:.3f}ms gather={:.3f}ms "
                        "attn={:.3f}ms other={:.3f}ms total={:.3f}ms".format(
                            avg["search_ms"],
                            avg["gather_ms"],
                            avg["attn_ms"],
                            avg["other_ms"],
                            avg["total_ms"],
                        )
                    )

        return acc_out.view(self.batch_size, 1, self.num_heads, self.head_dim)
