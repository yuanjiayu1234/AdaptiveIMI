from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Dict, Optional
import json
import math
import os
import queue as thread_queue
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
class _PendingCpuSubmit:
    kind: str
    layer_idx: int
    keys: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    token_counts: Optional[torch.Tensor] = None
    max_middle_len: int = 0
    chunk_id: int = 0
    start_bdx: int = 0
    source_token: int = 0
    token_offset: int = 0
    chunk_len: int = 0
    is_last: bool = False
    middle_start: int = 0
    middle_end: int = 0
    submit_chunk_size: int = 0
    producer_event: Optional[torch.cuda.Event] = None

class Hybrid8WayIMICache(adpimi_cache):
    """Hybrid layer-level 8-way learned IMI cache.

    CPU and GPU layers share the same batch-aware metadata schema. GPU-only
    prefill follows RetroInfer's bounded ordering: current-layer GPU metadata
    build is synchronous, raw KV D2H is asynchronous until sync(), and CPU list
    organization is allowed to overlap only with the next layer.
    """

    def __init__(self, *args, **kwargs) -> None:
        runtime_config = dict(kwargs.get("runtime_config") or {})
        partition_path = runtime_config.get("partition_path")
        if not partition_path:
            raise ValueError("hybrid_8way_v2 requires AdaptiveIMI.partition_path")

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
        early_hybrid_cfg = dict(runtime_config.get("hybrid_index") or {})
        ordered_cpu_finish = bool(early_hybrid_cfg.get("ordered_cpu_finish", True))
        configured_core = int(kwargs.get("core") or 0)
        configured_cpu_threads = int(runtime_config.get("cpu_threads") or 0)
        total_cpu_threads = configured_cpu_threads or configured_core or int(os.cpu_count() or 1)
        default_worker_threads = max(1, total_cpu_threads)
        if ordered_cpu_finish:
            configured_per_job = int(
                pipeline_cfg.get(
                    "cpu_worker_threads_per_job",
                    early_hybrid_cfg.get("cpu_worker_threads_per_job", 0),
                )
                or 0
            )
            default_worker_threads = configured_per_job if configured_per_job > 0 else max(1, total_cpu_threads // 2)
            default_worker_threads = min(default_worker_threads, total_cpu_threads)
            pipeline_cfg.setdefault("cpu_worker_threads_per_job", default_worker_threads)

        pipeline_cfg["worker_threads"] = int(pipeline_cfg.get("worker_threads", 0) or default_worker_threads)
        pipeline_cfg["total_cpu_threads"] = max(int(pipeline_cfg.get("total_cpu_threads", 0) or 0), total_cpu_threads)
        pipeline_cfg["enable_omp_nested"] = bool(pipeline_cfg.get("enable_omp_nested", not ordered_cpu_finish))
        default_active_levels = 2 if pipeline_cfg["enable_omp_nested"] else 1
        pipeline_cfg["omp_max_active_levels"] = max(
            int(pipeline_cfg.get("omp_max_active_levels", default_active_levels) or default_active_levels),
            default_active_levels,
        )
        pipeline_cfg["defer_merge_until_wait"] = True
        runtime_config["pipeline"] = pipeline_cfg
        runtime_config["cpu_threads"] = total_cpu_threads
        kwargs["runtime_config"] = runtime_config
        kwargs["subspace_parts"] = 8
        super().__init__(*args, **kwargs)
        # v2 builds indexes while prefill is still running. Do not allocate the
        # full decode-side GPU cache before prefill; only CPU/list metadata is
        # produced by background layer builders, and decode buffers are prepared
        # by move()/prepare_cache() after prefill.
        self.allocated = False
        if self.build_index_when_prefilling:
            n_centroids, nprobe, cache_pages, buffer_pages = self._estimate_cache_parameters()
            self.n_centroids = int(n_centroids)
            self.padding_cluster_id = self.n_centroids - 1
            self.nprobe = int(nprobe)
            self.max_retrieval_tokens = max(int(self.retrieval_budget * self.list_capacity), 1)
            self.budget_tokens = self.max_retrieval_tokens
            self.max_retrieval_pages = self._tokens_to_pages(self.max_retrieval_tokens)
            self.cache_size = int(cache_pages)
            self.cache_budget_pages = self.cache_size
            self.buffer_size = int(buffer_pages)
            self.buffer_budget_pages = self.buffer_size
            self.execution_stride = self.buffer_size * self.page_size + self.static_capacity
            self.retrieval_execution_stride = self.buffer_size * self.page_size

        hybrid_cfg = dict(runtime_config.get("hybrid_index") or {})
        self.hybrid_scheduler = LayerBuildScheduler.from_config(self.layer_num, hybrid_cfg)
        self.hybrid_prefill_finished = False
        self.layer_build_modes = self.hybrid_scheduler.all_modes()
        self._init_hybrid_gpu_build_state(hybrid_cfg)
        self.decode_search_config = dict(runtime_config.get("decode_search") or {})
        decode_search_mode = str(self.decode_search_config.get("mode", "flat")).lower()
        if decode_search_mode != "flat":
            raise ValueError(
                "hybrid_8way_v2 decode_search.mode now supports only 'flat'; "
                f"got {decode_search_mode!r}"
            )
        flat_score_mode = str(self.decode_search_config.get("flat_score", "raw_sum")).lower()
        if flat_score_mode not in ("raw_sum", "head_softmax_sum", "per_head_softmax_sum"):
            raise ValueError(
                "hybrid_8way_v2 decode_search.flat_score must be 'raw_sum' or "
                f"'head_softmax_sum'; got {flat_score_mode!r}"
            )
        self.flat_score_mode = "head_softmax_sum" if flat_score_mode == "per_head_softmax_sum" else flat_score_mode
        max_concurrent, total_build_threads = self._init_hybrid_cpu_finish_scheduler(runtime_config)

        print(
            "[IMI hybrid v2] layer_build_modes={} cpu_build_concurrency={} cpu_total_threads={} cpu_worker_threads_per_job={} gpu_build_order=layer_sync".format(
                ",".join(self.layer_build_modes),
                max_concurrent,
                total_build_threads,
                self.hybrid_finish_worker_threads,
            ),
            flush=True,
        )
        self.prefill_index_chunk_size = max(int(self.prefill_chunk_size), 1)

    def _init_hybrid_gpu_build_state(self, hybrid_cfg: dict) -> None:
        self.hybrid_gpu_jobs: list[Optional[HybridGpuLayerBuildJob]] = [None] * self.layer_num
        self.hybrid_cpu_source_refs: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(self.layer_num)
        ]
        self.hybrid_cpu_source_release_started = [False] * self.layer_num
        self.hybrid_cpu_stream_chunk_ids = [0] * self.layer_num
        self.hybrid_cpu_stream_token_offsets: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_cpu_stream_token_counts: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_cpu_stream_seen: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_gpu_stream_chunk_ids = [0] * self.layer_num
        self.hybrid_gpu_stream_token_offsets: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_gpu_stream_token_counts: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_gpu_stream_middle_starts: list[int] = [0] * self.layer_num
        self.hybrid_staging_keys: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_staging_values: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_staging_seen: list[Optional[torch.Tensor]] = [None] * self.layer_num
        self.hybrid_gpu_profile_rows = []
        self.hybrid_gpu_finish_scheduled = [False] * self.layer_num
        self.hybrid_cpu_submit_queue: thread_queue.Queue[Optional[_PendingCpuSubmit]] = thread_queue.Queue()
        self.hybrid_cpu_submit_stop = False
        self.hybrid_cpu_submit_worker = threading.Thread(
            target=self._hybrid_cpu_submit_loop,
            daemon=True,
            name="imi-hybrid-cpu-submit",
        )
        self.hybrid_cpu_submit_worker.start()

    def _init_hybrid_cpu_finish_scheduler(self, runtime_config: dict) -> tuple[int, int]:
        pipeline_cfg = dict(runtime_config.get("pipeline") or {})
        hybrid_cfg = dict(runtime_config.get("hybrid_index") or {})
        default_concurrency = int(
            hybrid_cfg.get(
                "cpu_build_concurrency",
                pipeline_cfg.get("max_concurrent_prefill", pipeline_cfg.get("kmeans_concurrency", 3)),
            )
        )
        if bool(hybrid_cfg.get("ordered_cpu_finish", True)):
            default_concurrency = 1
        ordered_cpu_finish = bool(hybrid_cfg.get("ordered_cpu_finish", True))
        max_concurrent = max(1, min(default_concurrency, self.layer_num))

        total_build_threads = int(
            pipeline_cfg.get(
                "total_cpu_threads",
                runtime_config.get("cpu_threads") or getattr(self, "core", self._available_cpu_cores),
            )
        )
        total_build_threads = max(1, total_build_threads)

        min_threads_per_job = max(1, total_build_threads // max_concurrent)
        configured_worker_threads = int(pipeline_cfg.get("cpu_worker_threads_per_job", 0) or 0)
        if configured_worker_threads <= 0:
            configured_worker_threads = min_threads_per_job
        if ordered_cpu_finish:
            self.hybrid_finish_worker_threads = max(1, min(total_build_threads, configured_worker_threads))
            min_threads_per_job = self.hybrid_finish_worker_threads
        else:
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
        return max_concurrent, total_build_threads

    @staticmethod
    def _storage_ptr(tensor: torch.Tensor) -> int:
        return int(tensor.untyped_storage().data_ptr())

    @staticmethod
    def _emit_pipeline_debug(payload: dict) -> None:
        if os.getenv("IMI_DEBUG_PIPELINE_STATS", "0") == "1":
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def _enqueue_cpu_submit_task(self, task: _PendingCpuSubmit) -> None:
        if not hasattr(self, "hybrid_cpu_submit_queue"):
            raise RuntimeError("hybrid CPU submit worker is not initialized")
        self.hybrid_cpu_submit_queue.put(task)

    @staticmethod
    def _record_cuda_producer_event(tensor: torch.Tensor) -> Optional[torch.cuda.Event]:
        if tensor is None or not tensor.is_cuda:
            return None
        device_index = int(tensor.device.index or 0)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device_index))
        return event

    def _hybrid_cpu_submit_loop(self) -> None:
        while True:
            task = self.hybrid_cpu_submit_queue.get()
            try:
                if task is None:
                    return
                self._run_cpu_submit_task(task)
            finally:
                self.hybrid_cpu_submit_queue.task_done()

    def _run_cpu_submit_task(self, task: _PendingCpuSubmit) -> None:
        layer_idx = int(task.layer_idx)
        if self.prefill_stream_errors[layer_idx] is not None:
            return
        start_ts = time.perf_counter()
        phase = str(task.kind)
        try:
            if task.keys is not None and task.keys.is_cuda:
                torch.cuda.set_device(task.keys.device)
            pipeline = self.imi_pipelines[layer_idx]
            if pipeline is None:
                raise RuntimeError(f"missing IMI pipeline for layer {layer_idx}")

            stream_context = nullcontext()
            if task.producer_event is not None and task.keys is not None and task.keys.is_cuda:
                producer_stream = torch.cuda.Stream(device=task.keys.device)
                producer_stream.wait_event(task.producer_event)
                stream_context = torch.cuda.stream(producer_stream)

            with stream_context:
                if task.kind == "begin_batch":
                    if task.token_counts is None:
                        raise RuntimeError(f"missing token_counts for CPU begin task on layer {layer_idx}")
                    pipeline.begin_index_stream_batch(task.token_counts, task.submit_chunk_size)
                elif task.kind == "batch_chunk":
                    if task.keys is None or task.values is None:
                        raise RuntimeError(f"missing raw tensors for CPU chunk task on layer {layer_idx}")
                    pipeline.submit_index_stream_raw_sequence_batch_chunk(
                        task.keys,
                        task.values,
                        chunk_id=int(task.chunk_id),
                        start_bdx=int(task.start_bdx),
                        source_token=int(task.source_token),
                        token_offset=int(task.token_offset),
                        chunk_len=int(task.chunk_len),
                        is_last=bool(task.is_last),
                    )
                elif task.kind == "finish":
                    self._release_hybrid_cpu_source_after_transfer(layer_idx)
                    self._schedule_hybrid_layer_finish(
                        layer_idx,
                        lambda worker_threads, layer_idx=layer_idx: self._finish_pipeline_layer_index(
                            layer_idx,
                            worker_threads,
                        ),
                    )
                elif task.kind == "full_layer":
                    self._run_cpu_full_layer_submit(task)
                else:
                    raise RuntimeError(f"unknown CPU submit task kind: {task.kind!r}")
        except BaseException as exc:
            if self.prefill_stream_errors[layer_idx] is None:
                self.prefill_stream_errors[layer_idx] = exc
            self.hybrid_cpu_source_refs[layer_idx].clear()
            self.prefill_stream_done_events[layer_idx].set()
            print(
                f"[IMI ERROR] hybrid layer={layer_idx} CPU submit failed during {phase}: {exc}",
                flush=True,
            )
        finally:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            self._emit_pipeline_debug({
                "tag": "IMI_CPU_SUBMIT_WORKER_TIMING",
                "layer_idx": int(layer_idx),
                "phase": phase,
                "elapsed_ms": round(elapsed_ms, 3),
            })

    def _run_cpu_full_layer_submit(self, task: _PendingCpuSubmit) -> None:
        layer_idx = int(task.layer_idx)
        if task.keys is None or task.values is None:
            raise RuntimeError(f"missing raw tensors for full CPU submit task on layer {layer_idx}")
        pipeline = self.imi_pipelines[layer_idx]
        if pipeline is None:
            raise RuntimeError(f"missing IMI pipeline for layer {layer_idx}")

        middle_start = int(task.middle_start)
        middle_end = int(task.middle_end)
        middle_len = middle_end - middle_start
        if middle_len <= 0:
            raise ValueError("Hybrid 8-way IMI requires a non-empty middle segment")

        begin_start = time.perf_counter()
        pipeline.begin_index_stream(middle_len, int(task.submit_chunk_size))
        begin_ms = (time.perf_counter() - begin_start) * 1000.0
        chunk_id = 0
        token_offset = 0
        chunk_start = middle_start
        submit_ms = 0.0
        while chunk_start < middle_end:
            chunk_end = min(middle_end, chunk_start + int(task.submit_chunk_size))
            chunk_len = chunk_end - chunk_start
            submit_start = time.perf_counter()
            pipeline.submit_index_stream_raw_sequence_chunk(
                task.keys,
                task.values,
                chunk_id=chunk_id,
                source_token=chunk_start,
                token_offset=token_offset,
                chunk_len=chunk_len,
                is_last=chunk_end == middle_end,
            )
            submit_ms += (time.perf_counter() - submit_start) * 1000.0
            chunk_id += 1
            token_offset += chunk_len
            chunk_start = chunk_end

        self._emit_pipeline_debug({
            "tag": "IMI_CPU_FULL_LAYER_SUBMIT_TIMING",
            "layer_idx": int(layer_idx),
            "middle_len": int(middle_len),
            "submit_chunk_size": int(task.submit_chunk_size),
            "begin_ms": round(begin_ms, 3),
            "submit_ms": round(submit_ms, 3),
            "chunks": int(chunk_id),
        })
        self._release_hybrid_cpu_source_after_transfer(layer_idx)
        self._schedule_hybrid_layer_finish(
            layer_idx,
            lambda worker_threads, layer_idx=layer_idx: self._finish_pipeline_layer_index(
                layer_idx,
                worker_threads,
            ),
        )

    def prefill_kv_workspace_available(self, keys: torch.Tensor, values: torch.Tensor) -> bool:
        key_ptr = self._storage_ptr(keys)
        value_ptr = self._storage_ptr(values)
        for ref_keys, ref_values in zip(
            getattr(self, "hybrid_staging_keys", []),
            getattr(self, "hybrid_staging_values", []),
        ):
            if ref_keys is None or ref_values is None:
                continue
            if self._storage_ptr(ref_keys) == key_ptr or self._storage_ptr(ref_values) == value_ptr:
                return False
        for refs in self.hybrid_cpu_source_refs:
            if isinstance(refs, tuple) and len(refs) == 2:
                refs = [refs]
            if not isinstance(refs, list):
                continue
            for ref_keys, ref_values in refs:
                if ref_keys is None or ref_values is None:
                    continue
                if self._storage_ptr(ref_keys) == key_ptr or self._storage_ptr(ref_values) == value_ptr:
                    return False
        for job in getattr(self, "hybrid_gpu_jobs", []):
            if job is None:
                continue
            ref_keys = getattr(job, "_keys_ref", None)
            ref_values = getattr(job, "_values_ref", None)
            if ref_keys is not None and self._storage_ptr(ref_keys) == key_ptr:
                return False
            if ref_values is not None and self._storage_ptr(ref_values) == value_ptr:
                return False
        return True

    def _hold_hybrid_cpu_source(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        layer_idx = int(layer_idx)
        refs = self.hybrid_cpu_source_refs[layer_idx]
        key_ptr = self._storage_ptr(keys)
        value_ptr = self._storage_ptr(values)
        for ref_keys, ref_values in refs:
            if self._storage_ptr(ref_keys) == key_ptr and self._storage_ptr(ref_values) == value_ptr:
                return
        refs.append((keys, values))

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
                self.hybrid_cpu_source_refs[layer_idx].clear()

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

    def _wait_previous_layer_finish(self, layer_idx: int) -> None:
        prev_layer = int(layer_idx) - 1
        if prev_layer < 0:
            return
        if not self.prefill_stream_started[prev_layer] or self.layer_ready[prev_layer]:
            return
        self.prefill_stream_done_events[prev_layer].wait()
        self._raise_prefill_stream_error(prev_layer)

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
        return False

    def _token_rerank_enabled(self) -> bool:
        try:
            return float(self.decode_search_config.get("token_rerank_budget_ratio", 0.0) or 0.0) > 0.0
        except (TypeError, ValueError):
            return False

    def can_reuse_for_next_sequence(self, valid_start, max_new_length: int, input_length: int) -> bool:
        return False

    def _select_clusters_for_decode(self, layer_idx: int, query_grouped: torch.Tensor, device: str) -> torch.Tensor:
        return self._select_clusters_flat(layer_idx, query_grouped, device)

    def _select_clusters_flat(self, layer_idx: int, query_grouped: torch.Tensor, device: str) -> torch.Tensor:
        if self.flat_score_mode == "head_softmax_sum":
            similarities = imi_gpu_kernels.fused_query_group_softmax_sum_similarities(
                query_grouped,
                self.centroids[layer_idx],
                self._softmax_logits_buffer(device),
                self.similarity_buffer_dict[device],
            )
            similarities.masked_fill_(self.centroids_mask[layer_idx], float("-inf"))
            selected = self._select_clusters(layer_idx, similarities)
        else:
            selected = self._select_clusters_raw_sum_flat(layer_idx, query_grouped, device)
        return selected

    def _rerank_execution_tokens_for_decode(self, layer_idx: int, device: str, query_grouped: torch.Tensor) -> None:
        if not self._token_rerank_enabled():
            return
        token_budget_ratio = float(self.decode_search_config.get("token_rerank_budget_ratio", 0.0) or 0.0)
        if token_budget_ratio <= 0.0:
            return
        token_budget = max(int(token_budget_ratio * self.list_capacity), 1)
        token_rerank = getattr(imi_gpu_kernels, "token_rerank_compact", None)
        if not callable(token_rerank):
            raise RuntimeError(
                "token rerank requires gpu_cluster_manager_cpp.token_rerank_compact. "
                "Rebuild library/AdaptiveIMI/cpp_extensions."
            )
        token_rerank(
            query_grouped.contiguous(),
            self.execution_buffer_keys_dict[device],
            self.execution_buffer_values_dict[device],
            self.valid_lengths_dict[device],
            self.static_lengths_dict[device],
            int(token_budget),
        )

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
        middle_len = middle_end - middle_start
        if middle_len <= 0:
            raise ValueError("Hybrid 8-way IMI requires a non-empty middle segment")
        self._hold_hybrid_cpu_source(layer_idx, key_states, value_states)

        self.layer_started[layer_idx] = True
        self.prefill_stream_started[layer_idx] = True
        self.prefill_stream_end_submitted[layer_idx] = True
        self.prefill_stream_done_events[layer_idx].clear()
        self.prefill_stream_errors[layer_idx] = None
        self._enqueue_cpu_submit_task(
            _PendingCpuSubmit(
                kind="full_layer",
                layer_idx=int(layer_idx),
                keys=key_states,
                values=value_states,
                middle_start=middle_start,
                middle_end=middle_end,
                submit_chunk_size=max(int(self.prefill_index_chunk_size), 1),
                producer_event=self._record_cuda_producer_event(key_states),
            )
        )

    def _begin_hybrid_cpu_chunk_stream(
        self,
        layer_idx: int,
        max_middle_len: int,
        token_counts: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        total_start = time.perf_counter()
        if self.layer_started[layer_idx]:
            return
        if self.layer_build_modes[layer_idx] != "cpu":
            raise RuntimeError(
                f"layer {layer_idx} is configured for {self.layer_build_modes[layer_idx]} build, not cpu"
            )
        if self.learned_partition_specs[layer_idx] is None:
            raise RuntimeError(f"missing learned 8-way partition for layer {layer_idx}")

        token_counts = token_counts.to(dtype=torch.int32, device="cpu").view(-1).contiguous()
        if int(token_counts.numel()) != self.batch_size:
            raise ValueError("CPU chunk stream token_counts must have one entry per batch item")
        if int(token_counts.min().item()) <= 0 or int(token_counts.max().item()) > int(max_middle_len):
            raise ValueError("invalid CPU chunk stream token_counts")
        if bool((token_counts != int(token_counts[0].item())).any().item()):
            raise ValueError("CPU batch build currently requires lockstep token_counts")

        pipeline = self.imi_pipelines[layer_idx]
        if pipeline is None:
            raise RuntimeError(f"missing IMI pipeline for layer {layer_idx}")
        if not key_states.is_contiguous() or not value_states.is_contiguous():
            raise RuntimeError("CPU raw batch stream requires contiguous [bsz, seq, kv_heads, dim] tensors")

        self._hold_hybrid_cpu_source(layer_idx, key_states, value_states)
        enqueue_start = time.perf_counter()
        self._enqueue_cpu_submit_task(
            _PendingCpuSubmit(
                kind="begin_batch",
                layer_idx=int(layer_idx),
                token_counts=token_counts,
                max_middle_len=int(max_middle_len),
                submit_chunk_size=max(int(self.prefill_index_chunk_size), 1),
            )
        )
        begin_enqueue_ms = (time.perf_counter() - enqueue_start) * 1000.0

        self.layer_started[layer_idx] = True
        self.prefill_stream_started[layer_idx] = True
        self.prefill_stream_end_submitted[layer_idx] = False
        self.prefill_stream_done_events[layer_idx].clear()
        self.prefill_stream_errors[layer_idx] = None
        self.hybrid_cpu_stream_chunk_ids[layer_idx] = 0
        self.hybrid_cpu_stream_token_offsets[layer_idx] = torch.zeros((self.batch_size,), dtype=torch.int32)
        self.hybrid_cpu_stream_token_counts[layer_idx] = token_counts
        self.hybrid_cpu_stream_seen[layer_idx] = torch.zeros((self.batch_size,), dtype=torch.bool)
        self._emit_pipeline_debug({
            "tag": "IMI_CPU_BEGIN_TIMING",
            "layer_idx": int(layer_idx),
            "max_middle_len": int(max_middle_len),
            "prefill_chunk_size": int(self.prefill_index_chunk_size),
            "begin_enqueue_ms": round(begin_enqueue_ms, 3),
            "total_ms": round((time.perf_counter() - total_start) * 1000.0, 3),
        })

    def _finish_hybrid_cpu_chunk_stream(self, layer_idx: int) -> None:
        if self.prefill_stream_end_submitted[layer_idx]:
            return
        self.prefill_stream_end_submitted[layer_idx] = True
        self.hybrid_cpu_stream_token_offsets[layer_idx] = None
        self.hybrid_cpu_stream_token_counts[layer_idx] = None
        self.hybrid_cpu_stream_seen[layer_idx] = None
        self._enqueue_cpu_submit_task(
            _PendingCpuSubmit(
                kind="finish",
                layer_idx=int(layer_idx),
            )
        )

    def _begin_hybrid_gpu_chunk_stream(
        self,
        layer_idx: int,
        max_middle_len: int,
        token_counts: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        middle_start: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        total_start = time.perf_counter()
        if self.hybrid_staging_keys[layer_idx] is not None:
            return True
        if self.layer_build_modes[layer_idx] != "gpu":
            raise RuntimeError(
                f"layer {layer_idx} is configured for {self.layer_build_modes[layer_idx]} build, not gpu"
            )
        if self.learned_partition_specs[layer_idx] is None:
            raise RuntimeError(f"missing learned 8-way partition for layer {layer_idx}")

        max_middle_len = int(max_middle_len)
        if max_middle_len <= 0:
            raise ValueError("Hybrid 8-way IMI requires a non-empty middle segment")
        token_counts = token_counts.to(dtype=torch.int32, device="cpu").view(-1).contiguous()
        if int(token_counts.numel()) != self.batch_size:
            raise ValueError("GPU chunk stream token_counts must have one entry per batch item")

        self.hybrid_staging_keys[layer_idx] = key_states
        self.hybrid_staging_values[layer_idx] = value_states
        self.hybrid_gpu_stream_middle_starts[layer_idx] = int(middle_start)
        self.hybrid_gpu_stream_token_counts[layer_idx] = token_counts
        self.hybrid_gpu_stream_chunk_ids[layer_idx] = 0
        self.hybrid_gpu_stream_token_offsets[layer_idx] = torch.zeros((self.batch_size,), dtype=torch.int32)
        self.hybrid_staging_seen[layer_idx] = torch.zeros((self.batch_size,), dtype=torch.bool)
        self.layer_started[layer_idx] = True
        self.prefill_stream_started[layer_idx] = True
        self.prefill_stream_end_submitted[layer_idx] = False
        self.prefill_stream_done_events[layer_idx].clear()
        self.prefill_stream_errors[layer_idx] = None
        self._emit_pipeline_debug({
            "tag": "IMI_GPU_BEGIN_TIMING",
            "layer_idx": int(layer_idx),
            "staging_mode": "direct",
            "max_middle_len": int(max_middle_len),
            "prefill_chunk_size": int(self.prefill_index_chunk_size),
            "total_ms": round((time.perf_counter() - total_start) * 1000.0, 3),
        })
        return True

    def _finish_hybrid_gpu_chunk_stream(self, layer_idx: int) -> None:
        source_keys = self.hybrid_staging_keys[layer_idx]
        source_values = self.hybrid_staging_values[layer_idx]
        token_counts = self.hybrid_gpu_stream_token_counts[layer_idx]
        if source_keys is None or source_values is None or token_counts is None:
            raise RuntimeError(f"hybrid GPU chunk stream for layer {layer_idx} was not initialized")
        if self.hybrid_gpu_jobs[layer_idx] is not None:
            return
        middle_start = int(self.hybrid_gpu_stream_middle_starts[layer_idx])
        max_tokens = int(token_counts.max().item())
        middle_end = middle_start + max_tokens
        middle_keys = source_keys[:, middle_start:middle_end, :, :].transpose(1, 2)
        middle_values = source_values[:, middle_start:middle_end, :, :].transpose(1, 2)

        device_index = int(source_keys.device.index or 0)
        source_ready_event = torch.cuda.Event()
        source_ready_event.record(torch.cuda.current_stream(device_index))
        max_iters, target_cluster_size = self._gpu_kmeans_config()
        partition_spec = self.learned_partition_specs[layer_idx]
        if partition_spec is None:
            raise RuntimeError(f"missing learned 8-way partition for layer {layer_idx}")

        progress = os.getenv("IMI_STREAMING_PROGRESS", "0") == "1"
        start_ts = time.perf_counter() if progress else None
        if progress:
            print(f"[IMI HYBRID INDEX] layer={layer_idx} mode=gpu phase=build_start", flush=True)
        self._emit_pipeline_debug({
            "tag": "IMI_GPU_BUILD_INPUT",
            "layer_idx": int(layer_idx),
            "staging_mode": "direct",
            "middle_start": int(middle_start),
            "middle_end": int(middle_end),
            "max_tokens": int(max_tokens),
            "source_shape": list(source_keys.shape),
            "build_shape": list(middle_keys.shape),
            "source_contiguous": bool(source_keys.is_contiguous()),
            "build_contiguous": bool(middle_keys.is_contiguous()),
            "source_storage_offset": int(source_keys.storage_offset()),
            "build_storage_offset": int(middle_keys.storage_offset()),
        })
        job = HybridGpuLayerBuildJob(
            layer_idx=int(layer_idx),
            keys=middle_keys,
            values=middle_values,
            partition_spec=partition_spec,
            build_device="gpu",
            token_counts=token_counts,
            target_cluster_size=target_cluster_size,
            max_iters=max_iters,
            source_ready_event=source_ready_event,
            auto_start_raw_d2h=True,
            auto_start_gpu_build=True,
        )
        job.wait_gpu_metadata_build()
        if progress and start_ts is not None:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            print(f"[IMI HYBRID INDEX] layer={layer_idx} mode=gpu phase=build_done elapsed_ms={elapsed_ms:.2f}", flush=True)
        self.hybrid_gpu_jobs[layer_idx] = job
        self.prefill_stream_end_submitted[layer_idx] = True
        self.hybrid_staging_keys[layer_idx] = None
        self.hybrid_staging_values[layer_idx] = None
        self.hybrid_gpu_stream_token_counts[layer_idx] = None
        self.hybrid_gpu_stream_token_offsets[layer_idx] = None
        self.hybrid_staging_seen[layer_idx] = None

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
        total_start = time.perf_counter()
        layer_idx = int(layer_idx)
        build_device = self.layer_build_modes[layer_idx]
        if build_device not in ("cpu", "gpu"):
            return False

        start_bdx = int(start_bdx)
        bsz = int(key_states.size(0))
        end_bdx = start_bdx + bsz
        if start_bdx < 0 or end_bdx > self.batch_size:
            raise ValueError(f"invalid prefill batch range [{start_bdx}, {end_bdx})")
        seq_len = int(key_states.size(1))
        middle_end = seq_len - self.static_pattern_end

        token_counts = self._middle_lengths_for_range(seq_len, 0, self.batch_size)
        max_middle_len = self._middle_capacity_for_sequence(seq_len)
        if max_middle_len <= 0 or int(token_counts.max().item()) <= 0:
            raise ValueError("Hybrid 8-way IMI requires a non-empty middle segment")

        valid_starts = self._valid_starts_for_range(start_bdx, bsz)
        has_overlap = False
        for valid_start in valid_starts:
            middle_start = int(valid_start) + self.static_pattern_start
            if max(int(chunk_start), middle_start) < min(int(chunk_end), middle_end):
                has_overlap = True
                break
        if not has_overlap:
            return False

        split_size = int(self.prefill_index_chunk_size)
        lockstep_middle = all(valid_start == valid_starts[0] for valid_start in valid_starts)
        if build_device == "cpu" and not lockstep_middle:
            raise ValueError("CPU batch build currently requires lockstep valid_start within a submitted group")

        if build_device == "cpu":
            current_offsets = self.hybrid_cpu_stream_token_offsets[layer_idx]
        else:
            current_offsets = self.hybrid_gpu_stream_token_offsets[layer_idx]

        ready_to_submit = False
        if lockstep_middle:
            middle_start = int(valid_starts[0]) + self.static_pattern_start
            middle_len = max(middle_end - middle_start, 0)
            if middle_len <= 0:
                raise ValueError("Hybrid 8-way IMI requires a non-empty middle segment")
            if current_offsets is None:
                token_offset = 0
            else:
                expected_offsets = current_offsets[start_bdx:end_bdx]
                token_offset = int(expected_offsets[0].item())
                if bool((expected_offsets != token_offset).any().item()):
                    raise RuntimeError(
                        f"hybrid {build_device.upper()} chunk stream offset mismatch on layer {layer_idx}: "
                        f"batch_range=[{start_bdx},{end_bdx}) expected lockstep offsets, got {expected_offsets.tolist()}"
                    )
            if token_offset < middle_len:
                token_width = min(split_size, middle_len - token_offset)
                part_end = middle_start + token_offset + token_width
                ready_to_submit = part_end <= int(chunk_end)
        else:
            for local_idx, valid_start in enumerate(valid_starts):
                global_idx = start_bdx + local_idx
                middle_start = int(valid_start) + self.static_pattern_start
                middle_len = max(middle_end - middle_start, 0)
                if middle_len <= 0:
                    continue
                token_offset = 0 if current_offsets is None else int(current_offsets[global_idx].item())
                if token_offset >= middle_len:
                    continue
                token_width = min(split_size, middle_len - token_offset)
                part_end = middle_start + token_offset + token_width
                if part_end <= int(chunk_end):
                    ready_to_submit = True
                    break

        if not ready_to_submit:
            return False

        self._expected_middle_len[layer_idx] = int(max_middle_len)
        if build_device == "cpu":
            begin_start = time.perf_counter()
            self._begin_hybrid_cpu_chunk_stream(
                layer_idx,
                max_middle_len,
                token_counts,
                key_states,
                value_states,
            )
            begin_ms = (time.perf_counter() - begin_start) * 1000.0
            token_offsets = self.hybrid_cpu_stream_token_offsets[layer_idx]
            seen = self.hybrid_cpu_stream_seen[layer_idx]
            if token_offsets is None or seen is None:
                raise RuntimeError(f"hybrid CPU chunk stream for layer {layer_idx} was not initialized")
            self._hold_hybrid_cpu_source(layer_idx, key_states, value_states)

            submit_ms = 0.0
            submitted_parts = 0
            submitted_tokens = 0
            finish_ms = 0.0
            middle_start = int(valid_starts[0]) + self.static_pattern_start
            middle_len = max(middle_end - middle_start, 0)
            if middle_len <= 0:
                raise ValueError("Hybrid 8-way IMI requires a non-empty middle segment")

            expected_offsets = token_offsets[start_bdx:end_bdx]
            token_offset = int(expected_offsets[0].item())
            if bool((expected_offsets != token_offset).any().item()):
                raise RuntimeError(
                    f"hybrid CPU chunk stream offset mismatch on layer {layer_idx}: "
                    f"batch_range=[{start_bdx},{end_bdx}) expected lockstep offsets, got {expected_offsets.tolist()}"
                )
            while token_offset < middle_len:
                token_width = min(split_size, middle_len - token_offset)
                part_start = middle_start + token_offset
                part_end = part_start + token_width
                if part_end > int(chunk_end):
                    break
                expected_offsets = token_offsets[start_bdx:end_bdx]
                if bool((expected_offsets != int(token_offset)).any().item()):
                    raise RuntimeError(
                        f"hybrid CPU chunk stream offset mismatch on layer {layer_idx}: "
                        f"batch_range=[{start_bdx},{end_bdx}) expected {expected_offsets.tolist()}, got {token_offset}"
                    )
                chunk_id = int(self.hybrid_cpu_stream_chunk_ids[layer_idx])
                self.hybrid_cpu_stream_chunk_ids[layer_idx] = chunk_id + 1
                submit_start = time.perf_counter()
                self._hold_hybrid_cpu_source(layer_idx, key_states, value_states)
                self._enqueue_cpu_submit_task(
                    _PendingCpuSubmit(
                        kind="batch_chunk",
                        layer_idx=int(layer_idx),
                        keys=key_states,
                        values=value_states,
                        chunk_id=chunk_id,
                        start_bdx=int(start_bdx),
                        source_token=int(part_start),
                        token_offset=int(token_offset),
                        chunk_len=int(token_width),
                        is_last=token_offset + token_width == middle_len and end_bdx == self.batch_size,
                        producer_event=self._record_cuda_producer_event(key_states),
                    )
                )
                submit_ms += (time.perf_counter() - submit_start) * 1000.0
                submitted_parts += 1
                submitted_tokens += int(token_width) * int(bsz)
                token_offsets[start_bdx:end_bdx] = token_offset + token_width
                token_offset += token_width
                if token_offset == middle_len:
                    seen[start_bdx:end_bdx] = True

            if bool(seen.all().item()) and not self.prefill_stream_end_submitted[layer_idx]:
                finish_start = time.perf_counter()
                self._finish_hybrid_cpu_chunk_stream(layer_idx)
                finish_ms = (time.perf_counter() - finish_start) * 1000.0
            self._emit_pipeline_debug({
                "tag": "IMI_PREFILL_SUBMIT_TIMING",
                "layer_idx": int(layer_idx),
                "build_device": "cpu",
                "start_bdx": int(start_bdx),
                "bsz": int(bsz),
                "chunk_start": int(chunk_start),
                "chunk_end": int(chunk_end),
                "begin_ms": round(begin_ms, 3),
                "cpu_submit_ms": round(submit_ms, 3),
                "finish_enqueue_ms": round(finish_ms, 3),
                "submitted_parts": int(submitted_parts),
                "submitted_tokens": int(submitted_tokens),
                "total_ms": round((time.perf_counter() - total_start) * 1000.0, 3),
            })
            return True

        begin_start = time.perf_counter()
        began_gpu_stream = self._begin_hybrid_gpu_chunk_stream(
            layer_idx,
            max_middle_len,
            token_counts,
            key_states,
            value_states,
            int(valid_starts[0]) + self.static_pattern_start,
            dtype=key_states.dtype,
            device=key_states.device,
        )
        begin_ms = (time.perf_counter() - begin_start) * 1000.0
        if not began_gpu_stream:
            raise RuntimeError(
                f"hybrid GPU chunk stream for layer {layer_idx} could not acquire a raw slot"
            )

        token_offsets = self.hybrid_gpu_stream_token_offsets[layer_idx]
        seen = self.hybrid_staging_seen[layer_idx]
        if token_offsets is None or seen is None:
            raise RuntimeError(f"hybrid GPU chunk stream for layer {layer_idx} was not initialized")
        stage_enqueue_ms = 0.0
        staged_parts = 0
        staged_tokens = 0
        finish_ms = 0.0
        if lockstep_middle:
            middle_start = int(valid_starts[0]) + self.static_pattern_start
            middle_len = max(middle_end - middle_start, 0)
            if middle_len <= 0:
                raise ValueError("Hybrid 8-way IMI requires a non-empty middle segment")

            expected_offsets = token_offsets[start_bdx:end_bdx]
            token_offset = int(expected_offsets[0].item())
            if bool((expected_offsets != token_offset).any().item()):
                raise RuntimeError(
                    f"hybrid GPU chunk stream offset mismatch on layer {layer_idx}: "
                    f"batch_range=[{start_bdx},{end_bdx}) expected lockstep offsets, got {expected_offsets.tolist()}"
                )
            while token_offset < middle_len:
                token_width = min(split_size, middle_len - token_offset)
                part_start = middle_start + token_offset
                part_end = part_start + token_width
                if part_end > int(chunk_end):
                    break
                expected_offsets = token_offsets[start_bdx:end_bdx]
                if bool((expected_offsets != int(token_offset)).any().item()):
                    raise RuntimeError(
                        f"hybrid GPU chunk stream offset mismatch on layer {layer_idx}: "
                        f"batch_range=[{start_bdx},{end_bdx}) expected {expected_offsets.tolist()}, got {token_offset}"
                    )

                staged_parts += 1
                staged_tokens += int(token_width) * int(bsz)

                chunk_id = int(self.hybrid_gpu_stream_chunk_ids[layer_idx])
                self.hybrid_gpu_stream_chunk_ids[layer_idx] = chunk_id + 1
                token_offsets[start_bdx:end_bdx] = token_offset + token_width
                token_offset += token_width
                if token_offset == middle_len:
                    seen[start_bdx:end_bdx] = True

            if bool(seen.all().item()) and not self.prefill_stream_end_submitted[layer_idx]:
                finish_start = time.perf_counter()
                self._finish_hybrid_gpu_chunk_stream(layer_idx)
                finish_ms = (time.perf_counter() - finish_start) * 1000.0
            self._emit_pipeline_debug({
                "tag": "IMI_PREFILL_SUBMIT_TIMING",
                "layer_idx": int(layer_idx),
                "build_device": "gpu",
                "start_bdx": int(start_bdx),
                "bsz": int(bsz),
                "chunk_start": int(chunk_start),
                "chunk_end": int(chunk_end),
                "begin_ms": round(begin_ms, 3),
                "gpu_stage_enqueue_ms": round(stage_enqueue_ms, 3),
                "finish_enqueue_ms": round(finish_ms, 3),
                "staged_parts": int(staged_parts),
                "staged_tokens": int(staged_tokens),
                "total_ms": round((time.perf_counter() - total_start) * 1000.0, 3),
            })
            return True

        for local_idx, valid_start in enumerate(valid_starts):
            global_idx = start_bdx + local_idx
            middle_start = int(valid_start) + self.static_pattern_start
            middle_len = max(middle_end - middle_start, 0)
            if middle_len <= 0:
                raise ValueError("Hybrid 8-way IMI requires a non-empty middle segment")

            token_offset = int(token_offsets[global_idx].item())
            while token_offset < middle_len:
                token_width = min(split_size, middle_len - token_offset)
                part_start = middle_start + token_offset
                part_end = part_start + token_width
                if part_end > int(chunk_end):
                    break
                expected_offset = int(token_offsets[global_idx].item())
                if token_offset != expected_offset:
                    raise RuntimeError(
                        f"hybrid GPU chunk stream offset mismatch on layer {layer_idx}: "
                        f"batch={global_idx} expected {expected_offset}, got {token_offset}"
                    )

                staged_parts += 1
                staged_tokens += int(token_width)

                chunk_id = int(self.hybrid_gpu_stream_chunk_ids[layer_idx])
                self.hybrid_gpu_stream_chunk_ids[layer_idx] = chunk_id + 1
                token_offsets[global_idx] = token_offset + token_width
                token_offset += token_width
                if token_offset == middle_len:
                    seen[global_idx] = True

        if bool(seen.all().item()) and not self.prefill_stream_end_submitted[layer_idx]:
            finish_start = time.perf_counter()
            self._finish_hybrid_gpu_chunk_stream(layer_idx)
            finish_ms = (time.perf_counter() - finish_start) * 1000.0
        self._emit_pipeline_debug({
            "tag": "IMI_PREFILL_SUBMIT_TIMING",
            "layer_idx": int(layer_idx),
            "build_device": "gpu",
            "start_bdx": int(start_bdx),
            "bsz": int(bsz),
            "chunk_start": int(chunk_start),
            "chunk_end": int(chunk_end),
            "begin_ms": round(begin_ms, 3),
            "gpu_stage_enqueue_ms": round(stage_enqueue_ms, 3),
            "finish_enqueue_ms": round(finish_ms, 3),
            "staged_parts": int(staged_parts),
            "staged_tokens": int(staged_tokens),
            "total_ms": round((time.perf_counter() - total_start) * 1000.0, 3),
        })
        return True

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
            max_middle_len = self._middle_capacity_for_sequence(seq_len)
            if max_middle_len <= 0:
                raise ValueError("Hybrid 8-way IMI requires a non-empty middle segment")
            self._expected_middle_len[layer_idx] = int(max_middle_len)
            build_device = self.layer_build_modes[layer_idx]

            if not self.layer_started[layer_idx]:
                if build_device == "gpu":
                    raise RuntimeError(
                        f"hybrid GPU chunk stream for layer {layer_idx} did not start; "
                        "prefill_submit_kv_chunk() must acquire a GPU raw slot before final KV update"
                    )
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
                    raise RuntimeError(
                        f"hybrid chunk stream for layer {layer_idx} was not submitted; "
                        "prefill must use prefill_submit_kv_chunk() before final KV update"
                    )
            elif (
                build_device in ("cpu", "gpu")
                and not self.prefill_stream_end_submitted[layer_idx]
            ):
                raise RuntimeError(
                    f"hybrid {build_device.upper()} chunk stream for layer {layer_idx} "
                    "did not submit the final middle chunk"
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
            if self.build_index_when_prefilling:
                self.attn_func = self.sparse_attention
            else:
                self.static_pattern_total = seq_len

        return key_states[:, return_valid_start:, :, :], value_states[:, return_valid_start:, :, :]

    def sync(self, layer_idx, start_bdx):
        layer_idx = int(layer_idx)
        if not self.build_index_when_prefilling:
            return None
        if self.layer_build_modes[layer_idx] != "gpu":
            return super().sync(layer_idx, start_bdx)
        if not self.prefill_stream_started[layer_idx]:
            return None
        self._raise_prefill_stream_error(layer_idx)
        if self.layer_ready[layer_idx]:
            return None
        if not self.prefill_stream_end_submitted[layer_idx]:
            raise RuntimeError(f"hybrid GPU layer {layer_idx} did not submit the final middle chunk")
        job = self.hybrid_gpu_jobs[layer_idx]
        if job is None:
            raise RuntimeError(f"hybrid GPU build job for layer {layer_idx} was not submitted")

        if not self.hybrid_gpu_finish_scheduled[layer_idx]:
            try:
                # Match RetroInfer's sync point: raw KV must be copied away
                # before this layer workspace can be reused by the next layer.
                job.wait_gpu_sources_consumed()
                # CPU metadata is only needed by the background organizer.  Do
                # not block the prefill path on D2H here; enqueue it and let
                # the finish worker wait before touching CPU metadata.
                job.start_cpu_metadata_d2h()
            except BaseException as exc:
                if self.prefill_stream_errors[layer_idx] is None:
                    self.prefill_stream_errors[layer_idx] = exc
                self.prefill_stream_done_events[layer_idx].set()
                raise
            # Queue CPU metadata/organize in layer order, but do not stall the
            # next prefill layer here. Decode readiness is still gated by the
            # per-layer done event and layer_ready flag.
            self.hybrid_gpu_finish_scheduled[layer_idx] = True
            self._schedule_hybrid_layer_finish(
                layer_idx,
                lambda _worker_threads, layer_idx=layer_idx: self._finish_layer_index(layer_idx),
            )
        return None

    def _install_hybrid_decode_metadata(self, layer_idx: int, metadata_gpu: dict) -> None:
        layer_idx = int(layer_idx)
        if metadata_gpu is None:
            raise RuntimeError(f"hybrid GPU metadata for layer {layer_idx} is not ready")

        centroids_gpu = metadata_gpu.get("centroids")
        cluster_sizes_gpu_src = metadata_gpu.get("cluster_sizes")
        cluster_counts_gpu_src = metadata_gpu.get("cluster_counts")
        if centroids_gpu is None or cluster_sizes_gpu_src is None or cluster_counts_gpu_src is None:
            raise RuntimeError(f"missing GPU decode metadata for hybrid layer {layer_idx}")
        required_centroids = int(centroids_gpu.size(1)) + 1
        if required_centroids > self.n_centroids:
            raise RuntimeError(
                f"hybrid GPU layer {layer_idx} produced {required_centroids - 1} clusters, "
                f"but metadata capacity is {self.n_centroids - 1}; "
                "GPU-only build must not resize decode metadata after allocation"
            )

        if int(centroids_gpu.dim()) != 3:
            raise RuntimeError(f"hybrid layer {layer_idx} centroids must be 3D")
        if int(cluster_sizes_gpu_src.dim()) != 2:
            raise RuntimeError(f"hybrid layer {layer_idx} cluster_sizes must be 2D")
        if int(centroids_gpu.size(0)) != int(self.batch_groups):
            raise RuntimeError(
                f"hybrid layer {layer_idx} centroid rows {int(centroids_gpu.size(0))} "
                f"!= batch_groups {int(self.batch_groups)}"
            )
        if int(cluster_sizes_gpu_src.size(0)) != int(self.batch_groups):
            raise RuntimeError(
                f"hybrid layer {layer_idx} cluster size rows {int(cluster_sizes_gpu_src.size(0))} "
                f"!= batch_groups {int(self.batch_groups)}"
            )
        if int(centroids_gpu.size(2)) != int(self.head_dim):
            raise RuntimeError(
                f"hybrid layer {layer_idx} centroid dim {int(centroids_gpu.size(2))} "
                f"!= head_dim {int(self.head_dim)}"
            )
        if int(cluster_sizes_gpu_src.size(1)) != int(centroids_gpu.size(1)):
            raise RuntimeError(
                f"hybrid layer {layer_idx} cluster_sizes columns {int(cluster_sizes_gpu_src.size(1))} "
                f"!= centroid clusters {int(centroids_gpu.size(1))}"
            )
        if int(cluster_counts_gpu_src.dim()) != 1 or int(cluster_counts_gpu_src.size(0)) != int(self.batch_groups):
            raise RuntimeError(
                f"hybrid layer {layer_idx} cluster_counts shape {tuple(cluster_counts_gpu_src.shape)} "
                f"does not match batch_groups {int(self.batch_groups)}"
            )

        self.centroids_cpu[layer_idx] = None
        self.centroids_mask_cpu[layer_idx] = None
        device = torch.device(self.layer_mapping[str(layer_idx)])
        with torch.cuda.device(device):
            self._materialize_hybrid_decode_metadata_gpu(
                layer_idx,
                centroids_gpu,
                cluster_sizes_gpu_src,
                cluster_counts_gpu_src,
                device,
            )

    def _materialize_hybrid_decode_metadata_gpu(
        self,
        layer_idx: int,
        centroids_compact: torch.Tensor,
        cluster_sizes_compact: torch.Tensor,
        cluster_counts: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Install compact GPU build metadata into the full decode layout.

        Decode, prefetch, and async-update code use the global n_centroids
        contract. The GPU builder emits a compact active-cluster layout, so we
        expand only metadata needed by decode on device and keep centroids off
        the CPU path.
        """
        if centroids_compact.device != device:
            centroids_compact = centroids_compact.to(device=device, non_blocking=True).contiguous()
        if cluster_sizes_compact.device != device:
            cluster_sizes_compact = cluster_sizes_compact.to(device=device, non_blocking=True).contiguous()
        if cluster_counts.device != device:
            cluster_counts = cluster_counts.to(device=device, non_blocking=True).contiguous()
        if not centroids_compact.is_contiguous():
            centroids_compact = centroids_compact.contiguous()
        if not cluster_sizes_compact.is_contiguous():
            cluster_sizes_compact = cluster_sizes_compact.contiguous()
        if not cluster_counts.is_contiguous():
            cluster_counts = cluster_counts.contiguous()

        cluster_sizes_compact = cluster_sizes_compact.to(dtype=torch.int32)
        cluster_counts = cluster_counts.to(dtype=torch.int32)
        compact_clusters = int(cluster_sizes_compact.size(1))
        cluster_ids = torch.arange(
            compact_clusters,
            device=device,
            dtype=torch.int32,
        ).view(1, -1)
        valid_by_count = cluster_ids < cluster_counts.view(-1, 1)
        active_sizes = cluster_sizes_compact.masked_fill(~valid_by_count, 0)
        active_mask = ~valid_by_count | (active_sizes == 0)

        centroids_full = self.centroids[layer_idx]
        cluster_sizes_full = self.cluster_sizes_gpu[layer_idx]
        centroids_mask_full = self.centroids_mask[layer_idx]
        centroid_shape = (self.batch_groups, self.n_centroids, self.head_dim)
        metadata_shape = (self.batch_groups, self.n_centroids)

        if centroids_full is None or tuple(centroids_full.shape) != centroid_shape:
            centroids_full = torch.empty(
                centroid_shape,
                dtype=torch.float32,
                device=device,
            )
        if cluster_sizes_full is None or tuple(cluster_sizes_full.shape) != metadata_shape:
            cluster_sizes_full = torch.empty(
                metadata_shape,
                dtype=torch.int32,
                device=device,
            )
        if centroids_mask_full is None or tuple(centroids_mask_full.shape) != metadata_shape:
            centroids_mask_full = torch.empty(
                metadata_shape,
                dtype=torch.bool,
                device=device,
            )

        cluster_sizes_full.zero_()
        centroids_mask_full.fill_(True)
        centroids_full[:, :compact_clusters, :].copy_(centroids_compact)
        cluster_sizes_full[:, :compact_clusters].copy_(active_sizes)
        centroids_mask_full[:, :compact_clusters].copy_(active_mask)

        self.centroids[layer_idx] = centroids_full
        self.cluster_sizes_gpu[layer_idx] = cluster_sizes_full
        self.centroids_mask[layer_idx] = centroids_mask_full

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
        result = job.finish(
            self.list_keys[layer_idx],
            self.list_values[layer_idx],
        )
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
            self._load_hybrid_gpu_layer_metadata(layer_idx, result)
        self._log_index_summary_once(layer_idx)

        stats = dict(result.stats)
        stats["layer_idx"] = int(layer_idx)
        self.hybrid_gpu_profile_rows.append(stats)
        if os.getenv("IMI_DEBUG_PIPELINE_STATS", "0") == "1":
            print(
                json.dumps({
                    "tag": "IMI_GPU_STATS",
                    "layer_idx": int(layer_idx),
                    "build_device": str(stats.get("build_device", build_mode)),
                    "max_tokens": int(float(stats.get("max_tokens", 0.0))),
                    "total_tokens": int(float(stats.get("total_tokens", 0.0))),
                    "gpu_build_ms": round(float(stats.get("gpu_build_ms", 0.0)), 3),
                    "raw_d2h_wait_ms": round(float(stats.get("raw_d2h_wait_ms", 0.0)), 3),
                    "metadata_d2h_ms": round(float(stats.get("metadata_d2h_ms", 0.0)), 3),
                    "reorganize_ms": round(float(stats.get("reorganize_ms", 0.0)), 3),
                }, ensure_ascii=False),
                flush=True,
            )
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

    def _load_hybrid_gpu_layer_metadata(self, layer_idx: int, result) -> None:
        metadata = result.metadata
        base_metadata_gpu = result.base_metadata_gpu
        if base_metadata_gpu is None:
            raise RuntimeError(f"hybrid GPU layer {layer_idx} did not return decode metadata")
        self.layer_metadata[layer_idx] = metadata

        cluster_sizes = self.cluster_sizes_cpu[layer_idx]
        cluster_offsets = self.cluster_offsets_cpu[layer_idx]
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
            self.cluster_sizes_cpu[layer_idx] = cluster_sizes
            self.cluster_offsets_cpu[layer_idx] = cluster_offsets

        cluster_sizes.zero_()
        cluster_offsets.zero_()
        for row_idx, row_meta in enumerate(metadata):
            cluster_count = int(row_meta.get("cluster_sizes").numel())
            if cluster_count <= 0:
                continue
            if cluster_count >= self.n_centroids:
                raise RuntimeError(
                    f"hybrid GPU layer {layer_idx} produced {cluster_count} clusters, "
                    f"but metadata capacity is {self.n_centroids}"
                )
            sizes_tensor = row_meta["cluster_sizes"].to(dtype=torch.int32, device="cpu")
            offsets_tensor = row_meta["cluster_offsets"].to(dtype=torch.int32, device="cpu")
            cluster_sizes[row_idx, :cluster_count].copy_(sizes_tensor)
            cluster_offsets[row_idx, : cluster_count + 1].copy_(offsets_tensor)
            cluster_offsets[row_idx, cluster_count + 1 :] = int(offsets_tensor[-1].item())

        self._install_hybrid_decode_metadata(layer_idx, base_metadata_gpu)
        if self.cluster_sizes_gpu[layer_idx] is None or self.centroids_mask[layer_idx] is None:
            raise RuntimeError(f"hybrid GPU decode metadata for layer {layer_idx} is missing after install")
        if self.adpimi_index[layer_idx] is not None:
            self.adpimi_index[layer_idx].set_cluster_metadata(cluster_sizes, cluster_offsets, 0)
        self.layer_ready[layer_idx] = True

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
        self.hybrid_cpu_source_refs[layer_idx].clear()
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
        waited_layers = []
        for layer_idx in range(self.layer_num):
            if not self.prefill_stream_started[layer_idx] or self.layer_ready[layer_idx]:
                continue
            done_event = self.prefill_stream_done_events[layer_idx]
            done_event.wait()
            self._raise_prefill_stream_error(layer_idx)
            waited_layers.append(int(layer_idx))
            if not self.layer_ready[layer_idx]:
                raise RuntimeError(f"hybrid layer {layer_idx} finish event completed but layer_ready is false")
        if waited_layers:
            self._emit_pipeline_debug({
                "tag": "IMI_WAIT_ALL_LAYERS_READY",
                "layers": waited_layers,
            })

    def reset_for_next_sequence(self, valid_start, max_new_length: int, input_length: int) -> None:
        self._wait_for_hybrid_background_finishes()
        super().reset_for_next_sequence(valid_start, max_new_length, input_length)
        self.hybrid_gpu_jobs = [None] * self.layer_num
        self.hybrid_cpu_source_refs = [[] for _ in range(self.layer_num)]
        self.hybrid_cpu_source_release_started = [False] * self.layer_num
        self.hybrid_cpu_stream_chunk_ids = [0] * self.layer_num
        self.hybrid_cpu_stream_token_offsets = [None] * self.layer_num
        self.hybrid_cpu_stream_token_counts = [None] * self.layer_num
        self.hybrid_cpu_stream_seen = [None] * self.layer_num
        self.hybrid_gpu_stream_chunk_ids = [0] * self.layer_num
        self.hybrid_gpu_stream_token_offsets = [None] * self.layer_num
        self.hybrid_gpu_stream_token_counts = [None] * self.layer_num
        self.hybrid_gpu_stream_middle_starts = [0] * self.layer_num
        self.hybrid_staging_keys = [None] * self.layer_num
        self.hybrid_staging_values = [None] * self.layer_num
        self.hybrid_staging_seen = [None] * self.layer_num
        self.hybrid_gpu_finish_scheduled = [False] * self.layer_num
        self.hybrid_gpu_profile_rows = []

    def cleanup(self):
        try:
            self._wait_for_hybrid_background_finishes()
        except Exception:
            pass
        if hasattr(self, "hybrid_cpu_submit_queue"):
            self.hybrid_cpu_submit_stop = True
            self.hybrid_cpu_submit_queue.put(None)
            if hasattr(self, "hybrid_cpu_submit_worker") and self.hybrid_cpu_submit_worker is not None:
                self.hybrid_cpu_submit_worker.join(timeout=1.0)
        if hasattr(self, "hybrid_gpu_jobs"):
            for idx in range(len(self.hybrid_gpu_jobs)):
                self.hybrid_gpu_jobs[idx] = None
                self.hybrid_cpu_source_refs[idx].clear()
                self.hybrid_cpu_source_release_started[idx] = False
                self.hybrid_cpu_stream_chunk_ids[idx] = 0
                self.hybrid_cpu_stream_token_offsets[idx] = None
                self.hybrid_cpu_stream_token_counts[idx] = None
                self.hybrid_cpu_stream_seen[idx] = None
                self.hybrid_gpu_stream_chunk_ids[idx] = 0
                self.hybrid_gpu_stream_token_offsets[idx] = None
                self.hybrid_gpu_stream_token_counts[idx] = None
                self.hybrid_gpu_stream_middle_starts[idx] = 0
                self.hybrid_staging_keys[idx] = None
                self.hybrid_staging_values[idx] = None
                self.hybrid_staging_seen[idx] = None
                self.hybrid_gpu_finish_scheduled[idx] = False
        if hasattr(self, "hybrid_gpu_profile_rows"):
            self.hybrid_gpu_profile_rows = []
        if hasattr(self, "hybrid_finish_scheduler") and self.hybrid_finish_scheduler is not None:
            self.hybrid_finish_scheduler.shutdown()
            self.hybrid_finish_scheduler = None
        super().cleanup()
