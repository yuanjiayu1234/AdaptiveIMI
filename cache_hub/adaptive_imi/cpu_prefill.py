from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional
import threading
import time

import torch


@dataclass
class _PendingCpuSubmit:
    kind: str
    layer_idx: int
    keys: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    token_counts: Optional[torch.Tensor] = None
    chunk_id: int = 0
    start_bdx: int = 0
    source_token: int = 0
    token_offset: int = 0
    chunk_len: int = 0
    is_last: bool = False
    submit_chunk_size: int = 0
    producer_event: Optional[torch.cuda.Event] = None


class HybridCpuPrefillMixin:
    @staticmethod
    def _storage_ptr(tensor: torch.Tensor) -> int:
        return int(tensor.untyped_storage().data_ptr())

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

    def prefill_kv_workspace_available(self, keys: torch.Tensor, values: torch.Tensor) -> bool:
        key_ptr = self._storage_ptr(keys)
        value_ptr = self._storage_ptr(values)
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
