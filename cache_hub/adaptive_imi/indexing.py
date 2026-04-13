from typing import Dict, List, Optional, Tuple
import json
import os
import queue
import threading
import time

import torch


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
        self._workers: set[threading.Thread] = set()
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
            with self._cv:
                while self._active_jobs > 0:
                    self._cv.wait(timeout=0.1)
                workers = list(self._workers)
            for worker in workers:
                worker.join(timeout=1.0)
            return
        self._stop_event.set()
        self._queue.put(((0, 0), 0.0, None))
        with self._cv:
            self._cv.notify_all()
        self._dispatcher.join(timeout=1.0)
        with self._cv:
            while self._active_jobs > 0:
                self._cv.wait(timeout=0.1)
            workers = list(self._workers)
        for worker in workers:
            worker.join(timeout=1.0)
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
            with self._cv:
                self._workers.add(thread)
            thread.start()
    def _run_job(self, fn, worker_threads: int) -> None:
        current = threading.current_thread()
        try:
            fn(worker_threads)
        finally:
            with self._cv:
                self._active_jobs = max(0, self._active_jobs - 1)
                self._workers.discard(current)
                self._cv.notify_all()


class AdaptiveIMIIndexingMixin:
    def _ensure_list_storage(self, layer_idx: int) -> None:
        if not self.build_index_when_prefilling:
            return
        if self.list_keys[layer_idx] is not None and self.list_values[layer_idx] is not None:
            return
        self.list_keys[layer_idx] = torch.empty(
            (self.batch_size, self.kv_head, self.list_capacity, self.head_dim),
            dtype=self.dtype,
            pin_memory=True,
        ).contiguous()
        self.list_values[layer_idx] = torch.empty(
            (self.batch_size, self.kv_head, self.list_capacity, self.head_dim),
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
        ideal_clusters_per_head = max(float(self.list_capacity) / 16.0, 0.0)
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

        # 2. Cancel C++ IMI pipelines before touching Python callbacks/storage.
        for pipeline in getattr(self, "imi_pipelines", []):
            if pipeline is None:
                continue
            try:
                pipeline.cancel_pipeline()
            except Exception:
                pass

        # 3. Wait for scheduled finish jobs to exit after pipeline cancellation.
        if hasattr(self, "_kmeans_scheduler") and self._kmeans_scheduler is not None:
            self._kmeans_scheduler.shutdown()

        # 4. Release pipeline wrappers.
        for i in range(self.layer_num):
            pipeline = self.imi_pipelines[i]
            if pipeline is None:
                continue
            try:
                pipeline.close()
            except Exception:
                pass
            self.imi_pipelines[i] = None

        # 5. Clear GPU tensor dicts
        self.execution_buffer_keys_dict.clear()
        self.execution_buffer_values_dict.clear()
        self.valid_lengths_dict.clear()
        self.static_len_tensor_dict.clear()
        self.static_lengths_dict.clear()
        self.similarity_buffer_dict.clear()

        # 6. Clear GPU tensor lists
        self.steady_zone_keys.clear()
        self.steady_zone_values.clear()
        self.cache_keys.clear()
        self.cache_values.clear()
        self.cluster_sizes_gpu.clear()
        self.centroids.clear()
        self.centroids_mask.clear()

        # 7. Clear pinned memory lists
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

        # 8. Clear CPU metadata
        self.cluster_sizes_cpu.clear()
        self.cluster_offsets_cpu.clear()
        self.centroids_cpu.clear()
        self.centroids_mask_cpu.clear()

        # 9. Clear streaming prefill resources
        for i in range(self.layer_num):
            self.prefill_stream_stage_keys[i] = None
            self.prefill_stream_stage_values[i] = None
            self.prefill_stream_free_slots[i] = None
            self.prefill_stream_queues[i] = None
            self.prefill_stream_copy_streams[i] = None
            self.prefill_stream_main_events[i] = None
            self.prefill_stream_copy_events[i] = None

        # 10. Clear wave buffers and IMI pipelines
        for i in range(self.layer_num):
            self.adpimi_index[i] = None

        # 11. Shutdown thread pool
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
