from __future__ import annotations

import json
import os
import time

import torch

from .gpu_builder import HybridGpuLayerBuildJob


class HybridGpuPrefillMixin:
    def _gpu_kmeans_config(self) -> tuple[int, int]:
        cfg = self.runtime_config.get("kmeans") or {}
        max_iters = int(cfg.get("gpu_max_iters", cfg.get("max_iters", 15)))
        target_cluster_size = int(cfg.get("target_cluster_size", cfg.get("cluster_size", 16)))
        return max_iters, target_cluster_size

    def _submit_hybrid_gpu_layer_job(
        self,
        layer_idx: int,
        start_bdx: int,
        end_bdx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        valid_starts: list[int],
    ) -> None:
        if self.hybrid_gpu_jobs[layer_idx] is not None:
            return
        bsz = int(end_bdx) - int(start_bdx)
        if bsz <= 0:
            raise ValueError(f"invalid hybrid GPU job batch range [{start_bdx}, {end_bdx})")
        if int(start_bdx) != 0 or int(end_bdx) != int(self.batch_size):
            raise ValueError(
                "hybrid_8way_v2 GPU build now supports only one full-batch lockstep job; "
                f"got batch_range=[{start_bdx},{end_bdx}) for batch_size={self.batch_size}. "
                "Use prefill_bsz=batch_size for repeat/lockstep batches."
            )
        if int(key_states.size(0)) != bsz or int(value_states.size(0)) != bsz:
            raise ValueError(
                f"hybrid GPU job source batch {int(key_states.size(0))} does not match range size {bsz}"
            )
        if not valid_starts:
            raise ValueError("hybrid GPU job requires valid_start values")
        if any(int(valid_start) != int(valid_starts[0]) for valid_start in valid_starts):
            raise ValueError("hybrid GPU direct build requires lockstep valid_start within a submitted group")

        seq_len = int(key_states.size(1))
        middle_start = int(valid_starts[0]) + self.static_pattern_start
        token_counts = self._middle_lengths_for_range(seq_len, start_bdx, bsz)
        max_tokens = int(token_counts.max().item())
        if max_tokens <= 0:
            raise RuntimeError(f"hybrid GPU chunk stream for layer {layer_idx} has no middle tokens")
        middle_end = middle_start + max_tokens
        middle_keys = key_states[:, middle_start:middle_end, :, :].transpose(1, 2)
        middle_values = value_states[:, middle_start:middle_end, :, :].transpose(1, 2)

        device_index = int(key_states.device.index or 0)
        source_ready_event = torch.cuda.Event()
        source_ready_event.record(torch.cuda.current_stream(device_index))
        max_iters, target_cluster_size = self._gpu_kmeans_config()
        partition_spec = self.learned_partition_specs[layer_idx]
        if partition_spec is None:
            raise RuntimeError(f"missing learned 8-way partition for layer {layer_idx}")
        partition_spec = dict(partition_spec)
        explicit_k_values = (self.runtime_config.get("kmeans") or {}).get("k_values")
        if explicit_k_values is not None:
            partition_spec["k_values"] = [int(value) for value in explicit_k_values]

        progress = os.getenv("IMI_STREAMING_PROGRESS", "0") == "1"
        start_ts = time.perf_counter() if progress else None
        if progress:
            print(
                f"[IMI HYBRID INDEX] layer={layer_idx} mode=gpu "
                "phase=build_start",
                flush=True,
            )
        self._emit_pipeline_debug({
            "tag": "IMI_GPU_BUILD_INPUT",
            "layer_idx": int(layer_idx),
            "staging_mode": "direct_full_batch",
            "middle_start": int(middle_start),
            "middle_end": int(middle_end),
            "max_tokens": int(max_tokens),
            "source_shape": list(key_states.shape),
            "build_shape": list(middle_keys.shape),
            "source_contiguous": bool(key_states.is_contiguous()),
            "build_contiguous": bool(middle_keys.is_contiguous()),
            "source_storage_offset": int(key_states.storage_offset()),
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
            print(
                f"[IMI HYBRID INDEX] layer={layer_idx} mode=gpu "
                f"phase=build_done elapsed_ms={elapsed_ms:.2f}",
                flush=True,
            )
        self.hybrid_gpu_jobs[layer_idx] = job
        self.layer_started[layer_idx] = True
        self.prefill_stream_started[layer_idx] = True
        self.prefill_stream_end_submitted[layer_idx] = True
        self.prefill_stream_done_events[layer_idx].clear()
        self.prefill_stream_errors[layer_idx] = None

    def sync(self, layer_idx, start_bdx):
        layer_idx = int(layer_idx)
        if not self.build_index_when_prefilling:
            return None
        if self.layer_build_modes[layer_idx] != "gpu":
            return self._sync_cpu_layer_prefill(layer_idx)
        if not self.prefill_stream_started[layer_idx]:
            return None
        self._raise_prefill_stream_error(layer_idx)
        if self.layer_ready[layer_idx]:
            return None
        if not self.prefill_stream_end_submitted[layer_idx]:
            return None
        job = self.hybrid_gpu_jobs[layer_idx]
        if job is None:
            raise RuntimeError(f"hybrid GPU build job for layer {layer_idx} was not submitted")

        if not self.hybrid_gpu_finish_scheduled[layer_idx]:
            try:
                # Raw KV must be copied away before this layer's workspace can be reused.
                job.wait_gpu_sources_consumed()
                # CPU metadata is used only by the background organizer, so enqueue
                # the transfer here and let the finish worker wait on it.
                job.start_cpu_metadata_d2h()
            except BaseException as exc:
                if self.prefill_stream_errors[layer_idx] is None:
                    self.prefill_stream_errors[layer_idx] = exc
                self.prefill_stream_done_events[layer_idx].set()
                raise
            self.hybrid_gpu_finish_scheduled[layer_idx] = True
            self._schedule_hybrid_layer_finish(
                layer_idx,
                lambda _worker_threads, layer_idx=layer_idx: self._finish_layer_index(layer_idx),
            )
        return None

    def _sync_cpu_layer_prefill(self, layer_idx: int):
        if not self.prefill_stream_started[layer_idx]:
            return None
        self._raise_prefill_stream_error(layer_idx)
        if self.layer_ready[layer_idx] or not self.prefill_stream_end_submitted[layer_idx]:
            return None
        done_event = self.prefill_stream_done_events[layer_idx]
        if done_event.is_set() and not self.layer_ready[layer_idx]:
            raise RuntimeError(f"hybrid CPU layer {layer_idx} finish event completed but layer_ready is false")
        return None

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

    def _schedule_hybrid_layer_finish(self, layer_idx: int, finish_fn) -> None:
        with self._build_job_lock:
            job_seq = self._build_job_seq
            self._build_job_seq += 1

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
