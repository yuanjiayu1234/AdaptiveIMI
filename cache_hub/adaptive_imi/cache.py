from __future__ import annotations

from typing import Optional
import json
import os
import queue as thread_queue
import threading
import time

import torch

from .core import AdaptiveIMICacheBase
from .cpu_prefill import HybridCpuPrefillMixin, _PendingCpuSubmit
from cache_hub.adaptive_imi.indexing import _BuildJobScheduler
from .gpu_builder import HybridGpuLayerBuildJob
from .gpu_prefill import HybridGpuPrefillMixin
from .layer_scheduler import LayerBuildScheduler
from .metadata import HybridMetadataMixin
from .partition import load_hybrid_partition_specs

class Hybrid8WayIMICache(HybridGpuPrefillMixin, HybridCpuPrefillMixin, HybridMetadataMixin, AdaptiveIMICacheBase):
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

        # Validate partition metadata before the base cache derives learned specs.
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
        prefill_cfg = dict(runtime_config.get("prefill") or {})
        configured_prefill_chunk = prefill_cfg.get("prefill_chunk_size")
        if configured_prefill_chunk is not None:
            self.prefill_chunk_size = max(int(configured_prefill_chunk), 1)
        # Hybrid prefill builds indexes while prefill is still running. Do not allocate the
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
        self.layer_build_modes = self.hybrid_scheduler.all_modes()
        self._init_hybrid_gpu_build_state(hybrid_cfg)
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
        self.hybrid_finish_scheduler = _BuildJobScheduler(
            total_build_threads,
            min_threads_per_job,
            max_concurrent=max_concurrent,
            worker_threads_per_job=self.hybrid_finish_worker_threads,
        )
        return max_concurrent, total_build_threads

    @staticmethod
    def _emit_pipeline_debug(payload: dict) -> None:
        if os.getenv("IMI_DEBUG_PIPELINE_STATS", "0") == "1":
            print(json.dumps(payload, ensure_ascii=False), flush=True)

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

    def can_reuse_for_next_sequence(self, valid_start, max_new_length: int, input_length: int) -> bool:
        return False

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
        if build_device == "gpu":
            if not lockstep_middle:
                raise ValueError("hybrid_8way_v2 GPU build requires lockstep valid_start within the batch")
            if start_bdx != 0 or end_bdx != self.batch_size:
                raise ValueError(
                    "hybrid_8way_v2 GPU build requires a single full-batch prefill range; "
                    f"got batch_range=[{start_bdx},{end_bdx}) for batch_size={self.batch_size}. "
                    "Use prefill_bsz=batch_size for repeat/lockstep batches."
                )
            middle_start = int(valid_starts[0]) + self.static_pattern_start
            middle_len = max(middle_end - middle_start, 0)
            if middle_len <= 0:
                raise ValueError("Hybrid 8-way IMI requires a non-empty middle segment")
            if int(chunk_end) < middle_start + middle_len:
                return False

            self._expected_middle_len[layer_idx] = int(max_middle_len)
            finish_start = time.perf_counter()
            self._submit_hybrid_gpu_layer_job(
                layer_idx,
                start_bdx,
                end_bdx,
                key_states,
                value_states,
                valid_starts,
            )
            finish_ms = (time.perf_counter() - finish_start) * 1000.0
            self._emit_pipeline_debug({
                "tag": "IMI_PREFILL_SUBMIT_TIMING",
                "layer_idx": int(layer_idx),
                "build_device": "gpu",
                "start_bdx": int(start_bdx),
                "bsz": int(bsz),
                "chunk_start": int(chunk_start),
                "chunk_end": int(chunk_end),
                "finish_enqueue_ms": round(finish_ms, 3),
                "submitted_parts": 1,
                "submitted_tokens": int(middle_len) * int(bsz),
                "total_ms": round((time.perf_counter() - total_start) * 1000.0, 3),
            })
            return True

        if not lockstep_middle:
            raise ValueError("CPU batch build currently requires lockstep valid_start within a submitted group")

        current_offsets = self.hybrid_cpu_stream_token_offsets[layer_idx]

        ready_to_submit = False
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
                raise RuntimeError(
                    f"hybrid {build_device.upper()} layer {layer_idx} was not submitted before final KV update; "
                    "prefill must use prefill_submit_kv_chunk()"
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
            if self.build_index_when_prefilling:
                self.attn_func = self.sparse_attention
            else:
                self.static_pattern_total = seq_len

        return key_states[:, return_valid_start:, :, :], value_states[:, return_valid_start:, :, :]

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
        if os.getenv("IMI_DEBUG_STREAMING_CHECK", "0") == "1":
            expected_len = self._expected_middle_len[layer_idx]
            actual_len = int(stats.get("total_tokens", 0)) if stats else None
            print(
                "[IMI STREAM CHECK]"
                f" layer={layer_idx} expected_tokens={expected_len}"
                f" actual_tokens={actual_len}",
                flush=True,
            )

        with self._metadata_lock:
            self._ensure_metadata_buffers()
            self._load_layer_metadata(layer_idx, metadata)
            self.layer_metadata[layer_idx] = None
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
                self.hybrid_gpu_finish_scheduled[idx] = False
        if hasattr(self, "hybrid_gpu_profile_rows"):
            self.hybrid_gpu_profile_rows = []
        if hasattr(self, "hybrid_finish_scheduler") and self.hybrid_finish_scheduler is not None:
            self.hybrid_finish_scheduler.shutdown()
            self.hybrid_finish_scheduler = None
        super().cleanup()
