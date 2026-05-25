from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import math
import os
import threading
import time

import torch

from cache_hub.adaptive_imi.cache import adpimi_cache
from cache_hub.adaptive_imi.indexing import _KmeansScheduler
from library.AdaptiveIMI.imi_adapter import get_imi_kernels
from .gpu_builder import HybridGpuLayerBuildJob
from .layer_scheduler import LayerBuildScheduler
from .partition import load_hybrid_partition_specs


imi_gpu_kernels = get_imi_kernels()


@dataclass
class _PendingGpuBuild:
    layer_idx: int
    keys: Optional[torch.Tensor]
    values: Optional[torch.Tensor]
    token_counts: torch.Tensor
    source_ready_event: torch.cuda.Event
    forced: bool = False
    running: bool = False
    job: Optional[HybridGpuLayerBuildJob] = None
    raw_ready: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None


class Hybrid8WayIMICache(adpimi_cache):
    """Hybrid layer-level 8-way learned IMI cache.

    CPU and GPU layers share the same batch-aware metadata schema. CPU layers
    build while prefill runs; GPU layers enqueue raw KV offload and metadata
    kernels on independent CUDA streams as soon as layer KV is ready. Both
    paths then reuse the existing CPU-list layout and hot/cold retrieval path.
    """

    def __init__(self, *args, **kwargs) -> None:
        runtime_config = dict(kwargs.get("runtime_config") or {})
        partition_path = runtime_config.get("partition_path") or runtime_config.get("learned_partition_path")
        if not partition_path:
            raise ValueError("hybrid_8way_v2 requires AdaptiveIMI.learned_partition_path or partition_path")

        # Strict v2 validation before the legacy cache builds its learned specs.
        load_hybrid_partition_specs(
            partition_path,
            layer_num=int(kwargs["layer_num"]),
            kv_heads=int(kwargs["num_key_value_heads"]),
            head_dim=int(kwargs["head_dim"]),
            required_m=8,
        )

        runtime_config["partition_path"] = str(partition_path)
        runtime_config["allow_batch_size_gt_one"] = True
        pipeline_cfg = dict(runtime_config.get("pipeline") or {})
        configured_core = int(kwargs.get("core") or 0)
        total_cpu_threads = configured_core or int(runtime_config.get("cpu_threads") or 0) or int(os.cpu_count() or 1)
        pipeline_cfg["worker_threads"] = max(int(pipeline_cfg.get("worker_threads", 0) or 0), total_cpu_threads)
        pipeline_cfg["total_cpu_threads"] = max(int(pipeline_cfg.get("total_cpu_threads", 0) or 0), total_cpu_threads)
        pipeline_cfg["enable_omp_nested"] = bool(pipeline_cfg.get("enable_omp_nested", True))
        pipeline_cfg["omp_max_active_levels"] = max(int(pipeline_cfg.get("omp_max_active_levels", 2) or 2), 2)
        pipeline_cfg["defer_merge_until_wait"] = True
        runtime_config["pipeline"] = pipeline_cfg
        runtime_config["cpu_threads"] = max(int(runtime_config.get("cpu_threads") or 0), total_cpu_threads)
        kwargs["runtime_config"] = runtime_config
        kwargs["subspace_parts"] = 8
        super().__init__(*args, **kwargs)
        # v2 builds indexes while prefill is still running. Do not allocate the
        # full decode-side GPU cache before prefill; only CPU/list metadata is
        # produced by background layer builders, and decode buffers are prepared
        # by move()/prepare_cache() after prefill.
        self.allocated = False

        hybrid_cfg = dict(runtime_config.get("hybrid_index") or {})
        self.hybrid_scheduler = LayerBuildScheduler.from_config(self.layer_num, hybrid_cfg)
        gpu_build_during_prefill_env = os.getenv("IMI_HYBRID_GPU_BUILD_DURING_PREFILL")
        if gpu_build_during_prefill_env is not None:
            self.hybrid_gpu_build_during_prefill = gpu_build_during_prefill_env.lower() in ("1", "true", "yes", "on")
        else:
            self.hybrid_gpu_build_during_prefill = bool(hybrid_cfg.get("gpu_build_during_prefill", True))
        self.hybrid_gpu_prefill_layer_gap = max(0, int(hybrid_cfg.get("gpu_prefill_layer_gap", 8)))
        self.hybrid_gpu_prefill_layer_interval = max(1, int(hybrid_cfg.get("gpu_prefill_layer_interval", 8)))
        self.hybrid_gpu_prefill_slow_ms = float(hybrid_cfg.get("gpu_prefill_slow_ms", 1000.0))
        self.hybrid_prefill_current_layer = -1
        self.hybrid_prefill_finished = False
        self.hybrid_gpu_next_prefill_layer = 0
        self.hybrid_gpu_prefill_disabled = False
        self.layer_build_modes = self.hybrid_scheduler.all_modes()
        self.hybrid_gpu_jobs: list[Optional[HybridGpuLayerBuildJob]] = [None] * self.layer_num
        self.hybrid_cpu_source_refs: list[Optional[object]] = [None] * self.layer_num
        self.hybrid_cpu_source_release_started = [False] * self.layer_num
        self.hybrid_gpu_stream_chunk_ids = [0] * self.layer_num
        self.hybrid_gpu_stream_token_offsets = [0] * self.layer_num
        self.hybrid_gpu_stream_token_counts: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_staging_keys: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_staging_values: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_staging_seen: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_staging_token_counts: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_search_metadata_cpu: list[Optional[Dict[str, torch.Tensor]]] = [None] * self.layer_num
        self.hybrid_search_metadata_gpu: list[Optional[Dict[str, torch.Tensor]]] = [None] * self.layer_num
        self.hybrid_gpu_profile_rows = []
        self.hybrid_gpu_pending: list[Optional[_PendingGpuBuild]] = [None] * self.layer_num
        self.hybrid_gpu_pending_order: list[int] = []
        self.hybrid_gpu_cv = threading.Condition()
        self.hybrid_gpu_stop = False
        self.hybrid_max_pending_gpu_builds = max(
            int(os.getenv("IMI_HYBRID_MAX_PENDING_GPU_BUILDS", "4")),
            1,
        )
        self.hybrid_gpu_worker = threading.Thread(
            target=self._hybrid_gpu_build_loop,
            daemon=True,
            name="imi-hybrid-gpu-build",
        )
        self.hybrid_gpu_worker.start()
        self.decode_search_config = dict(runtime_config.get("decode_search") or {})
        pipeline_cfg = dict(runtime_config.get("pipeline") or {})
        env_max_concurrent = os.getenv("IMI_HYBRID_MAX_CONCURRENT")
        if env_max_concurrent:
            default_concurrency = int(env_max_concurrent)
        else:
            default_concurrency = int(pipeline_cfg.get("max_concurrent_prefill", pipeline_cfg.get("kmeans_concurrency", 3)))
        max_concurrent = max(1, min(default_concurrency, self.layer_num))
        env_cpu_threads = os.getenv("IMI_CPU_BUILD_THREADS")
        if env_cpu_threads:
            total_build_threads = int(env_cpu_threads)
        else:
            total_build_threads = int(
                pipeline_cfg.get(
                    "total_cpu_threads",
                    runtime_config.get("cpu_threads") or getattr(self, "core", self._available_cpu_cores),
                )
            )
            total_build_threads = max(
                total_build_threads,
                int(getattr(self, "core", total_build_threads) or total_build_threads),
            )
        total_build_threads = max(1, total_build_threads)
        min_threads_per_job = max(1, total_build_threads // max_concurrent)
        configured_worker_threads = int(pipeline_cfg.get("cpu_worker_threads_per_job", 0) or 0)
        if configured_worker_threads <= 0:
            configured_worker_threads = max(
                min_threads_per_job,
                int(math.ceil(float(total_build_threads) * 0.75)),
            )
        self.hybrid_finish_worker_threads = max(
            min_threads_per_job,
            min(total_build_threads, configured_worker_threads),
        )
        self.hybrid_finish_scheduler = _KmeansScheduler(
            total_build_threads,
            min_threads_per_job,
            max_concurrent=max_concurrent,
            worker_threads_per_job=self.hybrid_finish_worker_threads,
        )

        print(
            "[IMI hybrid v2] layer_build_modes={} cpu_build_concurrency={} cpu_total_threads={} cpu_worker_threads_per_job={}".format(
                ",".join(self.layer_build_modes),
                max_concurrent,
                total_build_threads,
                self.hybrid_finish_worker_threads,
            ),
            flush=True,
        )
        self.index_submit_chunk_size = max(
            int(self.runtime_config["streaming"].get("index_submit_chunk_size", self.prefill_chunk_size)),
            1,
        )

    @staticmethod
    def _storage_ptr(tensor: torch.Tensor) -> int:
        return int(tensor.untyped_storage().data_ptr())

    def prefill_kv_workspace_available(self, keys: torch.Tensor, values: torch.Tensor) -> bool:
        key_ptr = self._storage_ptr(keys)
        value_ptr = self._storage_ptr(values)
        for refs in self.hybrid_cpu_source_refs:
            if not isinstance(refs, tuple) or len(refs) != 2:
                continue
            ref_keys, ref_values = refs
            if ref_keys is None or ref_values is None:
                continue
            if self._storage_ptr(ref_keys) == key_ptr or self._storage_ptr(ref_values) == value_ptr:
                return False
        return True

    def _hold_hybrid_cpu_source(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.hybrid_cpu_source_refs[int(layer_idx)] = (keys, values)
        self.hybrid_cpu_source_release_started[int(layer_idx)] = False

    def _release_hybrid_cpu_source_after_transfer(self, layer_idx: int) -> None:
        layer_idx = int(layer_idx)
        if self.hybrid_cpu_source_release_started[layer_idx]:
            return
        self.hybrid_cpu_source_release_started[layer_idx] = True

        def _release() -> None:
            try:
                pipeline = self.imi_pipelines[layer_idx]
                if pipeline is not None:
                    pipeline.wait_transfers_done()
            except BaseException as exc:
                if self.prefill_stream_errors[layer_idx] is None:
                    self.prefill_stream_errors[layer_idx] = exc
            finally:
                self.hybrid_cpu_source_refs[layer_idx] = None

        threading.Thread(
            target=_release,
            daemon=True,
            name=f"imi-hybrid-cpu-src-release-{layer_idx}",
        ).start()

    def _gpu_kmeans_config(self) -> tuple[int, int]:
        cfg = self.runtime_config.get("kmeans") or {}
        max_iters = int(cfg.get("gpu_max_iters", cfg.get("max_iters", 15)))
        target_cluster_size = int(cfg.get("target_cluster_size", cfg.get("cluster_size", 16)))
        return max_iters, target_cluster_size

    def _gpu_prefill_build_allowed_locked(self, pending: _PendingGpuBuild) -> bool:
        if not self.hybrid_gpu_build_during_prefill or self.hybrid_gpu_prefill_disabled:
            return False
        if not pending.raw_ready.is_set():
            return False
        layer_idx = int(pending.layer_idx)
        if self.hybrid_prefill_current_layer < layer_idx + self.hybrid_gpu_prefill_layer_gap:
            return False
        if layer_idx < self.hybrid_gpu_next_prefill_layer:
            return False
        active_cpu = self.hybrid_finish_scheduler.active_jobs()
        queued_cpu = self.hybrid_finish_scheduler.queued_jobs()
        return active_cpu >= self.hybrid_finish_scheduler.max_concurrent() or queued_cpu > 0

    def _pending_gpu_build_ready_locked(self, pending: _PendingGpuBuild) -> bool:
        return (
            pending.error is not None
            or pending.forced
            or self._gpu_prefill_build_allowed_locked(pending)
        )

    def _select_pending_gpu_build_locked(self) -> Optional[_PendingGpuBuild]:
        for layer_idx in list(self.hybrid_gpu_pending_order):
            pending = self.hybrid_gpu_pending[layer_idx]
            if pending is None or pending.running or pending.done.is_set():
                continue
            if self._pending_gpu_build_ready_locked(pending):
                pending.running = True
                return pending
        return None

    def _pending_gpu_build_count_locked(self) -> int:
        return sum(
            1
            for pending in self.hybrid_gpu_pending
            if pending is not None and not pending.done.is_set()
        )

    def _force_oldest_pending_gpu_build_locked(self) -> None:
        for layer_idx in list(self.hybrid_gpu_pending_order):
            pending = self.hybrid_gpu_pending[layer_idx]
            if pending is None or pending.done.is_set():
                continue
            pending.forced = True
            self.hybrid_gpu_cv.notify_all()
            return

    def _wait_for_pending_gpu_build_capacity_locked(self) -> None:
        while self._pending_gpu_build_count_locked() >= self.hybrid_max_pending_gpu_builds:
            self._force_oldest_pending_gpu_build_locked()
            self.hybrid_gpu_cv.wait(timeout=0.05)

    def _enqueue_pending_gpu_build(self, pending: _PendingGpuBuild) -> None:
        with self.hybrid_gpu_cv:
            self._wait_for_pending_gpu_build_capacity_locked()
            if self.hybrid_gpu_pending[pending.layer_idx] is not None:
                raise RuntimeError(f"duplicate pending GPU build for layer {pending.layer_idx}")
            self.hybrid_gpu_pending[pending.layer_idx] = pending
            self.hybrid_gpu_pending_order.append(pending.layer_idx)
            self.hybrid_gpu_pending_order.sort()
            self.hybrid_gpu_cv.notify_all()
        threading.Thread(
            target=self._prepare_pending_gpu_raw_d2h,
            args=(pending,),
            daemon=True,
            name=f"imi-hybrid-gpu-raw-{pending.layer_idx}",
        ).start()

    def _prepare_pending_gpu_raw_d2h(self, pending: _PendingGpuBuild) -> None:
        layer_idx = int(pending.layer_idx)
        try:
            partition_spec = self.learned_partition_specs[layer_idx]
            if partition_spec is None:
                raise RuntimeError(f"missing learned 8-way partition for layer {layer_idx}")

            max_iters, target_cluster_size = self._gpu_kmeans_config()
            job = HybridGpuLayerBuildJob(
                layer_idx=layer_idx,
                keys=pending.keys,
                values=pending.values,
                partition_spec=partition_spec,
                build_device="gpu",
                token_counts=pending.token_counts,
                target_cluster_size=target_cluster_size,
                max_iters=max_iters,
                source_ready_event=pending.source_ready_event,
                auto_start_raw_d2h=False,
                auto_start_gpu_build=False,
            )
            job.start_raw_d2h(pending.source_ready_event)
            pending.job = job
            self.hybrid_gpu_jobs[layer_idx] = job
        except BaseException as exc:
            pending.error = exc
            if self.prefill_stream_errors[layer_idx] is None:
                self.prefill_stream_errors[layer_idx] = exc
            print(
                f"[IMI ERROR] hybrid layer={layer_idx} raw D2H setup failed: {exc}",
                flush=True,
            )
        finally:
            pending.raw_ready.set()
            with self.hybrid_gpu_cv:
                self.hybrid_gpu_cv.notify_all()

    def _force_pending_gpu_build(self, layer_idx: int) -> Optional[_PendingGpuBuild]:
        with self.hybrid_gpu_cv:
            pending = self.hybrid_gpu_pending[int(layer_idx)]
            if pending is not None and not pending.done.is_set():
                pending.forced = True
                self.hybrid_gpu_cv.notify_all()
            return pending

    def _finish_pending_gpu_build(self, pending: _PendingGpuBuild) -> None:
        layer_idx = int(pending.layer_idx)
        was_forced = bool(pending.forced)
        start_ts = time.perf_counter()
        try:
            pending.raw_ready.wait()
            if pending.error is not None:
                raise pending.error
            job = pending.job
            if job is None:
                raise RuntimeError(f"hybrid GPU build job for layer {layer_idx} was not prepared")
            job.start_gpu_metadata_build(pending.source_ready_event)
            self._finish_layer_index(layer_idx)
        except BaseException as exc:
            pending.error = exc
            if self.prefill_stream_errors[layer_idx] is None:
                self.prefill_stream_errors[layer_idx] = exc
            print(
                f"[IMI ERROR] hybrid layer={layer_idx} GPU build failed: {exc}",
                flush=True,
            )
        finally:
            pending.keys = None
            pending.values = None
            with self.hybrid_gpu_cv:
                if self.hybrid_gpu_pending[layer_idx] is pending:
                    self.hybrid_gpu_pending[layer_idx] = None
                self.hybrid_gpu_pending_order = [
                    idx for idx in self.hybrid_gpu_pending_order if idx != layer_idx
                ]
                if not was_forced:
                    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                    if elapsed_ms > self.hybrid_gpu_prefill_slow_ms:
                        self.hybrid_gpu_prefill_disabled = True
                    else:
                        self.hybrid_gpu_next_prefill_layer = layer_idx + self.hybrid_gpu_prefill_layer_interval
                pending.done.set()
                self.prefill_stream_done_events[layer_idx].set()
                self.hybrid_gpu_cv.notify_all()

    def _hybrid_gpu_build_loop(self) -> None:
        while True:
            with self.hybrid_gpu_cv:
                while not self.hybrid_gpu_stop:
                    pending = self._select_pending_gpu_build_locked()
                    if pending is not None:
                        break
                    self.hybrid_gpu_cv.wait()
                if self.hybrid_gpu_stop:
                    return
            self._finish_pending_gpu_build(pending)

    def _wait_for_pending_gpu_build(self, layer_idx: int) -> None:
        pending = self._force_pending_gpu_build(layer_idx)
        if pending is None:
            return
        pending.done.wait()
        if pending.error is not None:
            raise RuntimeError(f"hybrid GPU build failed on layer {layer_idx}") from pending.error

    def prepare_cache(self, skip_prefetch: bool = False):
        self._wait_for_hybrid_background_finishes()
        return super().prepare_cache(skip_prefetch=True)

    def ensure_layer_ready(self, layer_idx):
        layer_idx = int(layer_idx)
        if self.layer_ready[layer_idx] or not self.build_index_when_prefilling:
            return
        self._raise_prefill_stream_error(layer_idx)
        if not self.layer_started[layer_idx]:
            raise RuntimeError(f"IMI layer {layer_idx} not started")

        pending = self.hybrid_gpu_pending[layer_idx]
        if pending is not None and not pending.done.is_set():
            self._wait_for_pending_gpu_build(layer_idx)
            self._raise_prefill_stream_error(layer_idx)
            if self.layer_ready[layer_idx]:
                return

        if self.prefill_stream_started[layer_idx]:
            done_event = self.prefill_stream_done_events[layer_idx]
            done_event.wait()
            self._raise_prefill_stream_error(layer_idx)
            if self.layer_ready[layer_idx]:
                return
            raise RuntimeError(f"IMI streaming prefill did not finish for layer {layer_idx}")

        self._finish_layer_index(layer_idx)

    def _is_gpu_build_layer(self, layer_idx: int) -> bool:
        return self.layer_build_modes[int(layer_idx)] == "gpu"

    def _decode_search_enabled(self) -> bool:
        mode = str(self.decode_search_config.get("mode", "flat")).lower()
        return mode in ("subspace_rerank", "subspace", "hybrid")

    def can_reuse_for_next_sequence(self, valid_start, max_new_length: int, input_length: int) -> bool:
        allow_reuse = os.getenv("IMI_HYBRID_ALLOW_CACHE_REUSE", "0").lower() in ("1", "true", "yes", "on")
        if not allow_reuse:
            return False
        return super().can_reuse_for_next_sequence(valid_start, max_new_length, input_length)

    def _select_clusters_for_decode(self, layer_idx: int, query_grouped: torch.Tensor, device: str) -> torch.Tensor:
        if not self._decode_search_enabled():
            return self._select_clusters_flat(layer_idx, query_grouped, device)

        metadata = self.hybrid_search_metadata_gpu[layer_idx]
        required = (
            "subspace_centroids",
            "subspace_centroid_masks",
            "subspace_dim_ids",
            "subspace_dim_masks",
            "compact_codes",
        )
        if metadata is None or any(key not in metadata for key in required):
            raise RuntimeError(f"hybrid decode search metadata missing on layer {layer_idx}")

        batch_groups = int(query_grouped.size(0))

        if int(metadata["subspace_dim_ids"].size(0)) != batch_groups:
            raise RuntimeError(
                f"hybrid decode search metadata rows mismatch: "
                f"{metadata['subspace_dim_ids'].size(0)} != {batch_groups}"
            )

        dim_ids = metadata["subspace_dim_ids"].to(device=query_grouped.device)
        dim_masks = metadata["subspace_dim_masks"].to(device=query_grouped.device)
        subspace_centroids = metadata["subspace_centroids"].to(device=query_grouped.device)
        subspace_masks = metadata["subspace_centroid_masks"].to(device=query_grouped.device)
        compact_codes = metadata["compact_codes"].to(device=query_grouped.device)
        active_clusters = int(compact_codes.size(1))
        if active_clusters <= 0:
            return torch.full(
                (batch_groups, self.nprobe),
                self.padding_cluster_id,
                dtype=torch.int64,
                device=query_grouped.device,
            )

        mode = str(self.decode_search_config.get("mode", "flat")).lower()
        default_candidates = max(self.nprobe * 4, 1024)
        candidate_limit = int(self.decode_search_config.get("candidate_clusters", default_candidates))
        candidate_limit = max(1, min(max(candidate_limit, self.nprobe), active_clusters))
        if mode != "subspace":
            fused_rerank = getattr(imi_gpu_kernels, "fused_subspace_rerank_select", None)
            if not callable(fused_rerank):
                raise RuntimeError(
                    "subspace_rerank requires gpu_cluster_manager_cpp.fused_subspace_rerank_select. "
                    "Rebuild library/AdaptiveIMI/cpp_extensions; PyTorch fallback has been removed."
                )
            rerank_limit = int(self.decode_search_config.get("rerank_clusters", self.nprobe))
            return fused_rerank(
                query_grouped.contiguous(),
                subspace_centroids.contiguous(),
                subspace_masks.contiguous(),
                dim_ids.contiguous(),
                dim_masks.contiguous(),
                compact_codes.contiguous(),
                self.centroids[layer_idx].to(device=query_grouped.device).contiguous(),
                self.cluster_sizes_gpu[layer_idx].to(device=query_grouped.device).contiguous(),
                int(self.nprobe),
                int(candidate_limit),
                int(rerank_limit),
                int(self.max_retrieval_tokens),
                int(self.padding_cluster_id),
            )

        query = query_grouped.float().sum(dim=1)
        dim_safe = dim_ids.clamp_min(0)
        query_expanded = query.unsqueeze(1).expand(-1, dim_safe.size(1), -1)
        query_sub = torch.gather(query_expanded, 2, dim_safe)
        query_sub = query_sub.masked_fill(dim_masks, 0.0)
        subspace_scores = torch.einsum("bmd,bmkd->bmk", query_sub, subspace_centroids)
        subspace_scores = subspace_scores.masked_fill(subspace_masks, float("-inf"))

        codes_by_subspace = compact_codes.permute(0, 2, 1).to(torch.long)
        valid_codes = codes_by_subspace >= 0
        code_scores = torch.gather(subspace_scores, 2, codes_by_subspace.clamp_min(0))
        coarse_scores = code_scores.masked_fill(~valid_codes, 0.0).sum(dim=1)
        valid_compact = valid_codes.all(dim=1)
        coarse_scores = coarse_scores.masked_fill(~valid_compact, float("-inf"))

        if mode == "subspace":
            selected_width = min(self.nprobe, active_clusters)
            coarse_topk = torch.topk(coarse_scores, selected_width, dim=-1, largest=True, sorted=True)
            ranked_ids = coarse_topk.indices
            cluster_sizes = self.cluster_sizes_gpu[layer_idx].to(device=query_grouped.device)
            ranked_sizes = torch.gather(cluster_sizes, 1, ranked_ids)
            cumsum = torch.cumsum(ranked_sizes, dim=-1, dtype=torch.int32)
            keep = (ranked_sizes > 0) & ((cumsum - ranked_sizes) < self.max_retrieval_tokens)

            selected = torch.full(
                (batch_groups, self.nprobe),
                self.padding_cluster_id,
                dtype=torch.int64,
                device=query_grouped.device,
            )
            selected[:, :selected_width] = ranked_ids.masked_fill(~keep, self.padding_cluster_id)
            return selected

    def _select_clusters_flat(self, layer_idx: int, query_grouped: torch.Tensor, device: str) -> torch.Tensor:
        similarities = imi_gpu_kernels.fused_query_group_similarities(
            query_grouped,
            self.centroids[layer_idx],
            self.similarity_buffer_dict[device],
        )
        similarities.masked_fill_(self.centroids_mask[layer_idx], float("-inf"))
        selected = self._select_clusters(layer_idx, similarities)
        return selected

    def _valid_starts_for_range(self, start_bdx: int, bsz: int) -> list[int]:
        return [int(value) for value in self.valid_start_list[start_bdx : start_bdx + bsz]]

    def _middle_lengths_for_range(self, seq_len: int, start_bdx: int, bsz: int) -> torch.Tensor:
        lengths = []
        middle_end = int(seq_len) - self.static_pattern_end
        for valid_start in self._valid_starts_for_range(start_bdx, bsz):
            middle_start = valid_start + self.static_pattern_start
            lengths.append(max(middle_end - middle_start, 0))
        return torch.tensor(lengths, dtype=torch.int32)

    def _middle_capacity_for_sequence(self, seq_len: int) -> int:
        middle_end = int(seq_len) - self.static_pattern_end
        return max(
            [max(middle_end - (int(valid_start) + self.static_pattern_start), 0) for valid_start in self.valid_start_list]
            or [0]
        )

    def _prepare_prefill_chunk_ragged(
        self,
        states: torch.Tensor,
        start_bdx: int,
        token_counts: torch.Tensor,
        max_middle_len: int,
    ) -> torch.Tensor:
        bsz = int(states.size(0))
        out = torch.zeros(
            (bsz, self.kv_head, int(max_middle_len), self.head_dim),
            dtype=states.dtype,
            device=states.device,
        )
        if bsz > 1:
            valid_starts = self._valid_starts_for_range(start_bdx, bsz)
            if all(valid_start == valid_starts[0] for valid_start in valid_starts):
                valid_start = valid_starts[0]
                middle_start = valid_start + self.static_pattern_start
                middle_end = middle_start + int(token_counts[0].item())
                out[:, :, : int(token_counts[0].item()), :].copy_(
                    states[:, middle_start:middle_end, :, :].transpose(1, 2)
                )
                return out.contiguous()

        for local_idx, valid_start in enumerate(self._valid_starts_for_range(start_bdx, bsz)):
            token_count = int(token_counts[local_idx].item())
            if token_count <= 0:
                continue
            middle_start = valid_start + self.static_pattern_start
            middle_end = middle_start + token_count
            out[local_idx, :, :token_count, :].copy_(
                states[local_idx, middle_start:middle_end, :, :].transpose(0, 1)
            )
        return out.contiguous()

    def _copy_steady_zone(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        start_bdx: int,
    ) -> None:
        bsz, seq_len, _, _ = key_states.shape
        end_bdx = start_bdx + bsz
        valid_starts = self._valid_starts_for_range(start_bdx, bsz)
        if bsz > 1 and all(valid_start == valid_starts[0] for valid_start in valid_starts):
            valid_start = valid_starts[0]
            self.steady_zone_keys[layer_idx][start_bdx:end_bdx, :, : self.static_pattern_start, :] = (
                key_states[:, valid_start : valid_start + self.static_pattern_start, :, :].transpose(1, 2)
            )
            self.steady_zone_values[layer_idx][start_bdx:end_bdx, :, : self.static_pattern_start, :] = (
                value_states[:, valid_start : valid_start + self.static_pattern_start, :, :].transpose(1, 2)
            )
            if self.static_pattern_end > 0:
                self.steady_zone_keys[layer_idx][
                    start_bdx:end_bdx, :, self.static_pattern_start : self.static_pattern_total, :
                ] = key_states[:, seq_len - self.static_pattern_end : seq_len, :, :].transpose(1, 2)
                self.steady_zone_values[layer_idx][
                    start_bdx:end_bdx, :, self.static_pattern_start : self.static_pattern_total, :
                ] = value_states[:, seq_len - self.static_pattern_end : seq_len, :, :].transpose(1, 2)
            return

        for local_idx, valid_start in enumerate(valid_starts):
            global_idx = start_bdx + local_idx
            if self.static_pattern_start > 0:
                self.steady_zone_keys[layer_idx][global_idx, :, : self.static_pattern_start, :] = (
                    key_states[local_idx, valid_start : valid_start + self.static_pattern_start, :, :].transpose(0, 1)
                )
                self.steady_zone_values[layer_idx][global_idx, :, : self.static_pattern_start, :] = (
                    value_states[local_idx, valid_start : valid_start + self.static_pattern_start, :, :].transpose(0, 1)
                )
            if self.static_pattern_end > 0:
                self.steady_zone_keys[layer_idx][global_idx, :, self.static_pattern_start : self.static_pattern_total, :] = (
                    key_states[local_idx, seq_len - self.static_pattern_end : seq_len, :, :].transpose(0, 1)
                )
                self.steady_zone_values[layer_idx][global_idx, :, self.static_pattern_start : self.static_pattern_total, :] = (
                    value_states[local_idx, seq_len - self.static_pattern_end : seq_len, :, :].transpose(0, 1)
                )

    def _submit_hybrid_layer_build(
        self,
        layer_idx: int,
        middle_keys: torch.Tensor,
        middle_values: torch.Tensor,
        token_counts: torch.Tensor,
    ) -> None:
        if self.layer_started[layer_idx]:
            return
        partition_spec = self.learned_partition_specs[layer_idx]
        if partition_spec is None:
            raise RuntimeError(f"missing learned 8-way partition for layer {layer_idx}")

        build_device = self.layer_build_modes[layer_idx]
        self.layer_started[layer_idx] = True
        self.prefill_stream_started[layer_idx] = True
        self.prefill_stream_done_events[layer_idx].clear()
        self.prefill_stream_errors[layer_idx] = None

        if build_device == "cpu":
            raise RuntimeError(
                "hybrid_8way_v2 CPU build must use raw sequence KV submission; "
                "GPU-contiguous CPU staging is intentionally disabled"
            )

        device_index = int(middle_keys.device.index or 0)
        source_ready_event = torch.cuda.Event()
        source_ready_event.record(torch.cuda.current_stream(device_index))
        self.prefill_stream_end_submitted[layer_idx] = True
        self._enqueue_pending_gpu_build(
            _PendingGpuBuild(
                layer_idx=int(layer_idx),
                keys=middle_keys,
                values=middle_values,
                token_counts=token_counts,
                source_ready_event=source_ready_event,
            )
        )

    def _submit_hybrid_cpu_layer_build_chunked(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        middle_start: int,
        middle_end: int,
    ) -> None:
        if self.layer_started[layer_idx]:
            return
        if self.layer_build_modes[layer_idx] != "cpu":
            raise RuntimeError(
                f"layer {layer_idx} is configured for {self.layer_build_modes[layer_idx]} build, not cpu"
            )
        if self.learned_partition_specs[layer_idx] is None:
            raise RuntimeError(f"missing learned 8-way partition for layer {layer_idx}")

        pipeline = self.imi_pipelines[layer_idx]
        if pipeline is None:
            raise RuntimeError(f"missing IMI pipeline for layer {layer_idx}")

        middle_start = int(middle_start)
        middle_end = int(middle_end)
        pipeline.begin_index_stream(middle_end - middle_start, self.index_submit_chunk_size)
        self._hold_hybrid_cpu_source(layer_idx, key_states, value_states)

        self.layer_started[layer_idx] = True
        self.prefill_stream_started[layer_idx] = True
        self.prefill_stream_end_submitted[layer_idx] = True
        self.prefill_stream_done_events[layer_idx].clear()
        self.prefill_stream_errors[layer_idx] = None

        pipeline.submit_index_stream_raw_sequence_chunk(
            key_states,
            value_states,
            chunk_id=0,
            source_token=middle_start,
            token_offset=0,
            chunk_len=middle_end - middle_start,
            is_last=True,
        )

        self._release_hybrid_cpu_source_after_transfer(layer_idx)
        self._schedule_hybrid_layer_finish(
            layer_idx,
            lambda worker_threads, layer_idx=layer_idx: self._finish_pipeline_layer_index(
                layer_idx,
                worker_threads,
            ),
        )

    def _begin_hybrid_gpu_chunk_stream(
        self,
        layer_idx: int,
        middle_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if self.layer_started[layer_idx]:
            return
        if self.layer_build_modes[layer_idx] != "gpu":
            raise RuntimeError(
                f"layer {layer_idx} is configured for {self.layer_build_modes[layer_idx]} build, not gpu"
            )
        if self.learned_partition_specs[layer_idx] is None:
            raise RuntimeError(f"missing learned 8-way partition for layer {layer_idx}")

        shape = (1, self.kv_head, int(middle_len), self.head_dim)
        self.hybrid_staging_keys[layer_idx] = torch.empty(shape, dtype=dtype, device=device).contiguous()
        self.hybrid_staging_values[layer_idx] = torch.empty(shape, dtype=dtype, device=device).contiguous()
        self.hybrid_gpu_stream_token_counts[layer_idx] = torch.tensor([int(middle_len)], dtype=torch.int32)
        self.hybrid_gpu_stream_chunk_ids[layer_idx] = 0
        self.hybrid_gpu_stream_token_offsets[layer_idx] = 0
        self.layer_started[layer_idx] = True
        self.prefill_stream_started[layer_idx] = True
        self.prefill_stream_end_submitted[layer_idx] = False
        self.prefill_stream_done_events[layer_idx].clear()
        self.prefill_stream_errors[layer_idx] = None

    def _finish_hybrid_gpu_chunk_stream(self, layer_idx: int) -> None:
        middle_keys = self.hybrid_staging_keys[layer_idx]
        middle_values = self.hybrid_staging_values[layer_idx]
        token_counts = self.hybrid_gpu_stream_token_counts[layer_idx]
        if middle_keys is None or middle_values is None or token_counts is None:
            raise RuntimeError(f"hybrid GPU chunk stream for layer {layer_idx} was not initialized")
        if self.hybrid_gpu_jobs[layer_idx] is not None:
            return

        device_index = int(middle_keys.device.index or 0)
        source_ready_event = torch.cuda.Event()
        source_ready_event.record(torch.cuda.current_stream(device_index))
        self.prefill_stream_end_submitted[layer_idx] = True
        self.hybrid_staging_keys[layer_idx] = None
        self.hybrid_staging_values[layer_idx] = None
        self.hybrid_gpu_stream_token_counts[layer_idx] = None
        self._enqueue_pending_gpu_build(
            _PendingGpuBuild(
                layer_idx=int(layer_idx),
                keys=middle_keys,
                values=middle_values,
                token_counts=token_counts,
                source_ready_event=source_ready_event,
            )
        )

    def prefill_submit_kv_chunk(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        start_bdx: int,
        chunk_start: int,
        chunk_end: int,
    ) -> bool:
        if not self.build_index_when_prefilling:
            return False
        layer_idx = int(layer_idx)
        self.hybrid_prefill_current_layer = max(self.hybrid_prefill_current_layer, layer_idx)
        build_device = self.layer_build_modes[layer_idx]
        if build_device not in ("cpu", "gpu"):
            return False
        if int(start_bdx) != 0 or int(key_states.size(0)) != 1 or self.batch_size != 1:
            return False

        seq_len = int(key_states.size(1))
        valid_start = int(self.valid_start_list[int(start_bdx)])
        middle_start = valid_start + self.static_pattern_start
        middle_end = seq_len - self.static_pattern_end
        middle_len = max(middle_end - middle_start, 0)
        if middle_len <= 0:
            raise ValueError("Hybrid 8-way IMI requires a non-empty middle segment")

        submit_start = max(int(chunk_start), middle_start)
        submit_end = min(int(chunk_end), middle_end)
        if submit_start >= submit_end:
            return False

        self._expected_middle_len[layer_idx] = int(middle_len)
        if build_device == "gpu":
            self._begin_hybrid_gpu_chunk_stream(
                layer_idx,
                middle_len,
                dtype=key_states.dtype,
                device=key_states.device,
            )

            staging_keys = self.hybrid_staging_keys[layer_idx]
            staging_values = self.hybrid_staging_values[layer_idx]
            if staging_keys is None or staging_values is None:
                raise RuntimeError(f"hybrid GPU chunk stream for layer {layer_idx} was not initialized")

            split_size = int(self.index_submit_chunk_size)
            for part_start in range(submit_start, submit_end, split_size):
                part_end = min(submit_end, part_start + split_size)
                token_offset = part_start - middle_start
                expected_offset = int(self.hybrid_gpu_stream_token_offsets[layer_idx])
                if token_offset != expected_offset:
                    raise RuntimeError(
                        f"hybrid GPU chunk stream offset mismatch on layer {layer_idx}: "
                        f"expected {expected_offset}, got {token_offset}"
                    )

                token_width = int(part_end - part_start)
                staging_keys[:, :, token_offset : token_offset + token_width, :].copy_(
                    key_states[:, part_start:part_end, :, :].transpose(1, 2)
                )
                staging_values[:, :, token_offset : token_offset + token_width, :].copy_(
                    value_states[:, part_start:part_end, :, :].transpose(1, 2)
                )

                chunk_id = int(self.hybrid_gpu_stream_chunk_ids[layer_idx])
                self.hybrid_gpu_stream_chunk_ids[layer_idx] = chunk_id + 1
                self.hybrid_gpu_stream_token_offsets[layer_idx] = token_offset + token_width
                if part_end == middle_end:
                    self._finish_hybrid_gpu_chunk_stream(layer_idx)
            return True

        # CPU layers are submitted once per layer after prefill has produced the
        # complete KV workspace. That lets the C++ pipeline issue a small number
        # of bulk D2H copies instead of many per-chunk strided transfers.
        if build_device == "cpu":
            return False
        return False

    def _stage_hybrid_layer_batch(
        self,
        layer_idx: int,
        start_bdx: int,
        middle_keys: torch.Tensor,
        middle_values: torch.Tensor,
        token_counts: torch.Tensor,
    ) -> None:
        bsz = int(middle_keys.size(0))
        end_bdx = int(start_bdx) + bsz
        if start_bdx < 0 or end_bdx > self.batch_size:
            raise ValueError(f"invalid prefill batch range [{start_bdx}, {end_bdx})")

        if self.hybrid_staging_keys[layer_idx] is None:
            shape = (self.batch_size, self.kv_head, int(middle_keys.size(2)), self.head_dim)
            self.hybrid_staging_keys[layer_idx] = torch.empty(
                shape,
                dtype=middle_keys.dtype,
                device=middle_keys.device,
            ).contiguous()
            self.hybrid_staging_values[layer_idx] = torch.empty(
                shape,
                dtype=middle_values.dtype,
                device=middle_values.device,
            ).contiguous()
            self.hybrid_staging_seen[layer_idx] = torch.zeros((self.batch_size,), dtype=torch.bool)
            self.hybrid_staging_token_counts[layer_idx] = torch.zeros((self.batch_size,), dtype=torch.int32)

        seen = self.hybrid_staging_seen[layer_idx]
        staging_keys = self.hybrid_staging_keys[layer_idx]
        staging_values = self.hybrid_staging_values[layer_idx]
        staging_token_counts = self.hybrid_staging_token_counts[layer_idx]
        if seen is None or staging_keys is None or staging_values is None or staging_token_counts is None:
            raise RuntimeError(f"hybrid staging buffers for layer {layer_idx} were not initialized")
        if bool(seen[start_bdx:end_bdx].any().item()):
            raise RuntimeError(f"duplicate prefill batch range for layer {layer_idx}: [{start_bdx}, {end_bdx})")

        staging_keys[start_bdx:end_bdx].copy_(middle_keys)
        staging_values[start_bdx:end_bdx].copy_(middle_values)
        staging_token_counts[start_bdx:end_bdx].copy_(token_counts.to(dtype=torch.int32, device="cpu"))
        seen[start_bdx:end_bdx] = True

        if bool(seen.all().item()):
            self._submit_hybrid_layer_build(layer_idx, staging_keys, staging_values, staging_token_counts)
            self.hybrid_staging_keys[layer_idx] = None
            self.hybrid_staging_values[layer_idx] = None
            self.hybrid_staging_seen[layer_idx] = None
            self.hybrid_staging_token_counts[layer_idx] = None

    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, start_bdx):
        bsz, seq_len, _, _ = key_states.shape
        end_bdx = start_bdx + bsz
        if end_bdx > self.batch_size:
            raise ValueError(f"invalid prefill batch range [{start_bdx}, {end_bdx})")

        valid_starts = self._valid_starts_for_range(start_bdx, bsz)
        return_valid_start = valid_starts[0]
        if bsz > 1 and any(valid_start != return_valid_start for valid_start in valid_starts):
            raise ValueError(
                "hybrid_8way_v2 ragged prefill requires prefill_bsz=1 so full prefill attention stays correct"
            )

        if self.build_index_when_prefilling:
            token_counts = self._middle_lengths_for_range(seq_len, start_bdx, bsz)
            max_middle_len = self._middle_capacity_for_sequence(seq_len)
            if max_middle_len <= 0:
                raise ValueError("Hybrid 8-way IMI requires a non-empty middle segment")
            self._expected_middle_len[layer_idx] = int(max_middle_len)
            build_device = self.layer_build_modes[layer_idx]

            if not self.layer_started[layer_idx]:
                self._ensure_list_storage(layer_idx)
                if build_device == "cpu" and bsz == 1 and self.batch_size == 1 and start_bdx == 0:
                    middle_start = int(valid_starts[0]) + self.static_pattern_start
                    middle_end = int(seq_len) - self.static_pattern_end
                    self._submit_hybrid_cpu_layer_build_chunked(
                        layer_idx,
                        key_states,
                        value_states,
                        middle_start,
                        middle_end,
                    )
                else:
                    middle_keys = self._prepare_prefill_chunk_ragged(key_states, start_bdx, token_counts, max_middle_len)
                    middle_values = self._prepare_prefill_chunk_ragged(value_states, start_bdx, token_counts, max_middle_len)
                    if bsz == self.batch_size and start_bdx == 0:
                        self._submit_hybrid_layer_build(layer_idx, middle_keys, middle_values, token_counts)
                    else:
                        self._stage_hybrid_layer_batch(layer_idx, start_bdx, middle_keys, middle_values, token_counts)
            elif (
                build_device == "cpu"
                and bsz == 1
                and self.batch_size == 1
                and start_bdx == 0
                and not self.prefill_stream_end_submitted[layer_idx]
            ):
                raise RuntimeError(
                    f"hybrid CPU chunk stream for layer {layer_idx} did not submit the final middle chunk"
                )

            self._copy_steady_zone(layer_idx, key_states, value_states, start_bdx)
        else:
            self.steady_zone_keys[layer_idx][start_bdx:end_bdx, :, :seq_len, :].copy_(
                key_states.transpose(1, 2)
            )
            self.steady_zone_values[layer_idx][start_bdx:end_bdx, :, :seq_len, :].copy_(
                value_states.transpose(1, 2)
            )

        if (layer_idx == self.layer_num - 1) and (start_bdx + bsz == self.batch_size):
            self.context += seq_len
            self.hybrid_prefill_finished = True
            with self.hybrid_gpu_cv:
                self.hybrid_gpu_cv.notify_all()
            if self.build_index_when_prefilling:
                self.attn_func = self.sparse_attention
            else:
                self.static_pattern_total = seq_len

        return key_states[:, return_valid_start:, :, :], value_states[:, return_valid_start:, :, :]

    def _finish_layer_index(self, layer_idx: int):
        job = self.hybrid_gpu_jobs[layer_idx]
        if job is None:
            raise RuntimeError(f"hybrid build job for layer {layer_idx} was not submitted")

        progress = os.getenv("IMI_STREAMING_PROGRESS", "0") == "1"
        start_ts = time.perf_counter() if progress else None
        build_mode = self.layer_build_modes[layer_idx]
        if progress:
            print(f"[IMI HYBRID INDEX] layer={layer_idx} mode={build_mode} phase=start", flush=True)

        self._ensure_list_storage(layer_idx)
        keep_search_metadata = self._decode_search_enabled()
        result = job.finish(
            self.list_keys[layer_idx],
            self.list_values[layer_idx],
            keep_search_metadata=keep_search_metadata,
        )
        if keep_search_metadata:
            self.hybrid_search_metadata_cpu[layer_idx] = result.search_metadata_cpu
            if result.search_metadata_gpu is None:
                device = self.layer_mapping[str(layer_idx)]
                self.hybrid_search_metadata_gpu[layer_idx] = {
                    key: value.to(device=device, non_blocking=True)
                    for key, value in result.search_metadata_cpu.items()
                }
            else:
                self.hybrid_search_metadata_gpu[layer_idx] = result.search_metadata_gpu
        else:
            self.hybrid_search_metadata_cpu[layer_idx] = None
            self.hybrid_search_metadata_gpu[layer_idx] = None
        self.hybrid_gpu_jobs[layer_idx] = None

        placeholder = torch.empty((self.batch_groups, 0, self.head_dim), dtype=self.dtype, pin_memory=True)
        if self.adpimi_index[layer_idx] is not None:
            self.adpimi_index[layer_idx].set_kv(
                self.list_keys[layer_idx],
                self.list_values[layer_idx],
                placeholder,
                placeholder,
            )

        with self._metadata_lock:
            self._ensure_metadata_buffers()
            self._load_layer_metadata(layer_idx, result.metadata)
        self._log_index_summary_once(layer_idx)

        stats = dict(result.stats)
        stats["layer_idx"] = int(layer_idx)
        self.hybrid_gpu_profile_rows.append(stats)
        if self.profile_prefill:
            kmeans_ms = (
                float(stats.get("gpu_build_ms", 0.0))
                if str(stats.get("build_device", build_mode)) == "gpu"
                else float(stats.get("cpu_build_ms", 0.0))
            )
            self.prefill_breakdown_rows.append({
                "layer_idx": int(layer_idx),
                "d2h_ms": round(float(stats.get("raw_d2h_wait_ms", 0.0)), 4),
                "cpu_copy_ms": 0.0,
                "kmeans_ms": round(kmeans_ms, 4),
                "kmeans_cpu_time_ms": round(float(stats.get("cpu_build_ms", 0.0)), 4),
                "kmeans_cpu_util_cores": 0.0,
                "reorganize_ms": round(float(stats.get("reorganize_ms", 0.0)), 4),
                "write_ms": 0.0,
                "kmeans_gate_wait_ms": 0.0,
                "total_chunks": 1,
                "total_tokens": int(stats.get("total_tokens", 0)),
                "build_device": str(stats.get("build_device", build_mode)),
                "metadata_d2h_ms": round(float(stats.get("metadata_d2h_ms", 0.0)), 4),
            })

        if progress and start_ts is not None:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            print(
                f"[IMI HYBRID INDEX] layer={layer_idx} mode={build_mode} phase=done elapsed_ms={elapsed_ms:.2f}",
            flush=True,
        )

    def _schedule_hybrid_layer_finish(self, layer_idx: int, finish_fn) -> None:
        with self._kmeans_job_lock:
            job_seq = self._kmeans_job_seq
            self._kmeans_job_seq += 1

        def _run(worker_threads: int) -> None:
            try:
                finish_fn(worker_threads)
            except Exception as exc:
                print(
                    f"[IMI ERROR] hybrid layer={layer_idx} background finish failed: {exc}",
                    flush=True,
                )
                if self.prefill_stream_errors[layer_idx] is None:
                    self.prefill_stream_errors[layer_idx] = exc
            finally:
                self.prefill_stream_done_events[layer_idx].set()

        self.hybrid_finish_scheduler.submit((int(layer_idx), int(job_seq)), _run)

    def _finish_pipeline_layer_index(self, layer_idx: int, worker_threads: int) -> None:
        pipeline = self.imi_pipelines[layer_idx]
        if pipeline is None:
            raise RuntimeError(f"missing IMI pipeline for layer {layer_idx}")

        progress = os.getenv("IMI_STREAMING_PROGRESS", "0") == "1"
        start_ts = time.perf_counter() if progress else None
        if progress:
            print(f"[IMI HYBRID INDEX] layer={layer_idx} mode=cpu phase=start", flush=True)

        self._ensure_list_storage(layer_idx)
        pipeline.set_worker_threads(worker_threads)
        metadata = pipeline.finish_index(self.list_keys[layer_idx], self.list_values[layer_idx])
        stats = pipeline.get_last_pipeline_stats() or {}
        search_metadata_cpu = pipeline.get_last_search_metadata_cpu()

        placeholder = torch.empty((self.batch_groups, 0, self.head_dim), dtype=self.dtype, pin_memory=True)
        if self.adpimi_index[layer_idx] is not None:
            self.adpimi_index[layer_idx].set_kv(
                self.list_keys[layer_idx],
                self.list_values[layer_idx],
                placeholder,
                placeholder,
            )

        with self._metadata_lock:
            self._ensure_metadata_buffers()
            self._load_layer_metadata(layer_idx, metadata)
        self._log_index_summary_once(layer_idx)

        stats = dict(stats)
        stats["layer_idx"] = int(layer_idx)
        stats["build_device"] = "cpu_pipeline"
        self.hybrid_gpu_profile_rows.append(stats)
        self.hybrid_cpu_source_refs[layer_idx] = None
        if self._decode_search_enabled() and search_metadata_cpu is not None:
            device = self.layer_mapping[str(layer_idx)]
            self.hybrid_search_metadata_cpu[layer_idx] = search_metadata_cpu
            self.hybrid_search_metadata_gpu[layer_idx] = {
                key: value.to(device=device, non_blocking=True)
                for key, value in search_metadata_cpu.items()
                if key in (
                    "subspace_centroids",
                    "subspace_centroid_masks",
                    "subspace_dim_ids",
                    "subspace_dim_masks",
                    "compact_codes",
                )
            }
        if self.profile_prefill:
            self.prefill_breakdown_rows.append({
                "layer_idx": int(layer_idx),
                "d2h_ms": round(float(stats.get("d2h_ms", 0.0)), 4),
                "cpu_copy_ms": round(float(stats.get("cpu_copy_ms", 0.0)), 4),
                "kmeans_ms": round(float(stats.get("kmeans_ms", 0.0)), 4),
                "kmeans_cpu_time_ms": round(float(stats.get("kmeans_cpu_time_ms", 0.0)), 4),
                "kmeans_cpu_util_cores": round(float(stats.get("kmeans_cpu_util_cores", 0.0)), 4),
                "reorganize_ms": round(float(stats.get("reorganize_ms", 0.0)), 4),
                "write_ms": round(float(stats.get("write_ms", 0.0)), 4),
                "kmeans_gate_wait_ms": round(float(stats.get("kmeans_gate_wait_ms", 0.0)), 4),
                "total_chunks": int(stats.get("total_chunks", 1)),
                "total_tokens": int(stats.get("total_tokens", 0)),
                "build_device": "cpu_pipeline",
                "metadata_d2h_ms": 0.0,
            })

        if progress and start_ts is not None:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            print(
                f"[IMI HYBRID INDEX] layer={layer_idx} mode=cpu phase=done elapsed_ms={elapsed_ms:.2f}",
                flush=True,
            )

    def _wait_for_hybrid_background_finishes(self) -> None:
        if not getattr(self, "build_index_when_prefilling", False):
            return
        for layer_idx in range(self.layer_num):
            self._force_pending_gpu_build(layer_idx)
        for layer_idx in range(self.layer_num):
            if not self.prefill_stream_started[layer_idx] or self.layer_ready[layer_idx]:
                continue
            done_event = self.prefill_stream_done_events[layer_idx]
            done_event.wait()
            self._raise_prefill_stream_error(layer_idx)

    def reset_for_next_sequence(self, valid_start, max_new_length: int, input_length: int) -> None:
        self._wait_for_hybrid_background_finishes()
        super().reset_for_next_sequence(valid_start, max_new_length, input_length)
        self.hybrid_gpu_jobs = [None] * self.layer_num
        self.hybrid_cpu_source_refs = [None] * self.layer_num
        self.hybrid_cpu_source_release_started = [False] * self.layer_num
        self.hybrid_gpu_stream_chunk_ids = [0] * self.layer_num
        self.hybrid_gpu_stream_token_offsets = [0] * self.layer_num
        self.hybrid_gpu_stream_token_counts = [None] * self.layer_num
        self.hybrid_staging_keys = [None] * self.layer_num
        self.hybrid_staging_values = [None] * self.layer_num
        self.hybrid_staging_seen = [None] * self.layer_num
        self.hybrid_staging_token_counts = [None] * self.layer_num
        self.hybrid_search_metadata_cpu = [None] * self.layer_num
        self.hybrid_search_metadata_gpu = [None] * self.layer_num
        self.hybrid_gpu_profile_rows = []
        with self.hybrid_gpu_cv:
            self.hybrid_gpu_pending = [None] * self.layer_num
            self.hybrid_gpu_pending_order = []
            self.hybrid_gpu_cv.notify_all()

    def cleanup(self):
        if hasattr(self, "hybrid_gpu_cv"):
            try:
                self._wait_for_hybrid_background_finishes()
            except Exception:
                pass
            with self.hybrid_gpu_cv:
                self.hybrid_gpu_stop = True
                self.hybrid_gpu_cv.notify_all()
            if hasattr(self, "hybrid_gpu_worker") and self.hybrid_gpu_worker is not None:
                self.hybrid_gpu_worker.join(timeout=1.0)
        if hasattr(self, "hybrid_gpu_jobs"):
            for idx in range(len(self.hybrid_gpu_jobs)):
                self.hybrid_gpu_jobs[idx] = None
                self.hybrid_cpu_source_refs[idx] = None
                self.hybrid_cpu_source_release_started[idx] = False
                self.hybrid_gpu_stream_chunk_ids[idx] = 0
                self.hybrid_gpu_stream_token_offsets[idx] = 0
                self.hybrid_gpu_stream_token_counts[idx] = None
                self.hybrid_staging_keys[idx] = None
                self.hybrid_staging_values[idx] = None
                self.hybrid_staging_seen[idx] = None
                self.hybrid_staging_token_counts[idx] = None
                self.hybrid_search_metadata_cpu[idx] = None
                self.hybrid_search_metadata_gpu[idx] = None
            self.hybrid_gpu_pending = [None] * self.layer_num
            self.hybrid_gpu_pending_order = []
        if hasattr(self, "hybrid_gpu_profile_rows"):
            self.hybrid_gpu_profile_rows = []
        if hasattr(self, "hybrid_finish_scheduler") and self.hybrid_finish_scheduler is not None:
            self.hybrid_finish_scheduler.shutdown()
            self.hybrid_finish_scheduler = None
        super().cleanup()
