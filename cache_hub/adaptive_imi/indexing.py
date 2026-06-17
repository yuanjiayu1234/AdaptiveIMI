from typing import Dict, List, Optional, Tuple
import json
import os
import queue
import threading
import time

import torch

from cache_hub.cluster_plan import choose_learned_k_values, product_k_values


class _BuildJobScheduler:
    def __init__(
        self,
        total_cores: int,
        min_threads_per_job: int,
        max_concurrent: Optional[int] = None,
        worker_threads_per_job: Optional[int] = None,
    ) -> None:
        self._total_cores = max(1, total_cores)
        self._min_threads_per_job = max(1, min_threads_per_job)
        self._worker_threads_per_job = max(
            self._min_threads_per_job,
            min(self._total_cores, int(worker_threads_per_job or self._total_cores)),
        )
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
            name="imi-build-dispatcher",
        )
        self._dispatcher.start()
    def submit(self, priority: tuple[int, int], fn) -> None:
        if self._stop_event.is_set():
            return
        self._queue.put((priority, time.monotonic(), fn))
        with self._cv:
            self._cv.notify_all()
    def active_jobs(self) -> int:
        with self._cv:
            return int(self._active_jobs)
    def queued_jobs(self) -> int:
        return int(self._queue.qsize())
    def max_concurrent(self) -> int:
        return int(self._max_concurrent)
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
            # Pass a stable per-job budget. The native K-means code divides
            # this by the true active native jobs; keeping the budget stable
            # avoids queued-job pre-shrinking while still leaving headroom for
            # the prefill path.
            worker_threads = self._worker_threads_per_job
            thread = threading.Thread(
                target=self._run_job,
                args=(fn, worker_threads),
                daemon=True,
                name="imi-build-worker",
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

    def _resize_metadata_capacity(self, required_centroids: int) -> None:
        required_centroids = int(required_centroids)
        if required_centroids <= self.n_centroids:
            return

        old_centroids = int(self.n_centroids)
        self.n_centroids = required_centroids
        self.padding_cluster_id = self.n_centroids - 1
        self.nprobe = max(round((self.n_centroids - 1) * self.retrieval_budget), 1)
        self.nprobe = min(self.nprobe, self.n_centroids - 1)

        def resize_cpu_tensor(tensor, shape, fill_value=0):
            new_tensor = torch.empty(shape, dtype=tensor.dtype, pin_memory=tensor.is_pinned())
            new_tensor.fill_(fill_value)
            slices = tuple(slice(0, min(old, new)) for old, new in zip(tensor.shape, shape))
            new_tensor[slices].copy_(tensor[slices])
            return new_tensor

        for ldx in range(self.layer_num):
            sizes = self.cluster_sizes_cpu[ldx]
            if sizes is not None:
                self.cluster_sizes_cpu[ldx] = resize_cpu_tensor(
                    sizes,
                    (self.batch_groups, self.n_centroids),
                    0,
                )

            offsets = self.cluster_offsets_cpu[ldx]
            if offsets is not None:
                new_offsets = resize_cpu_tensor(
                    offsets,
                    (self.batch_groups, self.n_centroids + 1),
                    0,
                )
                if old_centroids >= 0:
                    tail = offsets[:, min(old_centroids, offsets.shape[1] - 1)].view(self.batch_groups, 1)
                    new_offsets[:, old_centroids + 1 :] = tail
                self.cluster_offsets_cpu[ldx] = new_offsets

            centroids_cpu = self.centroids_cpu[ldx]
            if centroids_cpu is not None:
                self.centroids_cpu[ldx] = resize_cpu_tensor(
                    centroids_cpu,
                    (self.batch_groups, self.n_centroids, self.head_dim),
                    0,
                )

            masks_cpu = self.centroids_mask_cpu[ldx]
            if masks_cpu is not None:
                self.centroids_mask_cpu[ldx] = resize_cpu_tensor(
                    masks_cpu,
                    (self.batch_groups, self.n_centroids),
                    True,
                )

            if self.cluster_sizes_gpu[ldx] is not None:
                device = self.cluster_sizes_gpu[ldx].device
                self.cluster_sizes_gpu[ldx] = self.cluster_sizes_cpu[ldx].to(device) if self.cluster_sizes_cpu[ldx] is not None else None
            if self.cluster_offsets_gpu[ldx] is not None:
                device = self.cluster_offsets_gpu[ldx].device
                self.cluster_offsets_gpu[ldx] = self.cluster_offsets_cpu[ldx].to(device) if self.cluster_offsets_cpu[ldx] is not None else None
            if self.centroids[ldx] is not None:
                device = self.centroids[ldx].device
                resized = torch.zeros(
                    (self.batch_groups, self.n_centroids, self.head_dim),
                    dtype=torch.float32,
                    device=device,
                )
                if self.centroids_cpu[ldx] is not None:
                    resized.copy_(self.centroids_cpu[ldx])
                self.centroids[ldx] = resized
            if self.centroids_mask[ldx] is not None:
                device = self.centroids_mask[ldx].device
                resized = torch.ones(
                    (self.batch_groups, self.n_centroids),
                    dtype=torch.bool,
                    device=device,
                )
                if self.centroids_mask_cpu[ldx] is not None:
                    resized.copy_(self.centroids_mask_cpu[ldx])
                self.centroids_mask[ldx] = resized

    def _load_layer_metadata(self, layer_idx: int, metadata: List[Dict[str, object]]):
        required_centroids = 1
        for head_meta in metadata:
            required_centroids = max(
                required_centroids,
                int(head_meta["cluster_sizes"].shape[0]) + 1,
            )
        self._resize_metadata_capacity(required_centroids)
        self.layer_metadata[layer_idx] = metadata

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

        self.layer_index_stats[layer_idx] = {
            "non_empty_clusters": int((cluster_sizes > 0).sum().item()),
        }

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
        if self.cluster_offsets_gpu[layer_idx] is not None:
            self.cluster_offsets_gpu[layer_idx].copy_(cluster_offsets)
        if self.centroids[layer_idx] is not None:
            self.centroids[layer_idx].copy_(centroids_cpu)
        if self.centroids_mask[layer_idx] is not None:
            self.centroids_mask[layer_idx].copy_(centroids_mask_cpu)
        self.layer_ready[layer_idx] = True
    def _ensure_metadata_buffers(self) -> None:
        if self.n_centroids <= 0:
            n_tokens = max(self.list_capacity, self.list_stride, 1)
            kmeans_cfg = self.runtime_config.get("kmeans") or {}
            target_cluster_size = max(int(kmeans_cfg.get("target_cluster_size", kmeans_cfg.get("cluster_size", 16))), 1)
            max_clusters = 1
            for spec in getattr(self, "learned_partition_specs", []):
                if spec is None:
                    raise RuntimeError("AdaptiveIMI hybrid_8way_v2 requires learned partition metadata for every layer")
                m = max(int(spec.get("subspace_parts", 1)), 1)
                k_values = choose_learned_k_values(
                    spec,
                    n_tokens,
                    target_cluster_size,
                    default_m=m,
                )
                product = product_k_values(k_values)
                max_clusters = max(max_clusters, min(product, n_tokens))
            self.n_centroids = int(max_clusters) + 1
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
                if self.cluster_offsets_gpu[ldx] is None:
                    self.cluster_offsets_gpu[ldx] = cluster_offsets.to(self.layer_mapping[str(ldx)])
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
    def get_index_stats(self) -> Dict[str, object]:
        ideal_clusters_per_head = max(float(self.list_capacity) / 16.0, 0.0)
        total_heads = int(self.layer_num * self.batch_groups)
        ideal_clusters_total = ideal_clusters_per_head * float(total_heads)

        non_empty_clusters_total = 0
        ready_layers = 0
        for layer_idx, stats in enumerate(getattr(self, "layer_index_stats", [])):
            if not stats:
                continue
            ready_layers += 1
            non_empty_clusters_total += int(stats.get("non_empty_clusters", 0))

        if ready_layers == 0:
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
    def cleanup(self):
        """Explicitly release all resources. Call this before destroying the cache."""
        # 1. Cancel C++ IMI pipelines before touching Python callbacks/storage.
        for pipeline in getattr(self, "imi_pipelines", []):
            if pipeline is None:
                continue
            try:
                pipeline.cancel_pipeline()
            except Exception:
                pass

        # 2. Wait for scheduled finish jobs to exit after pipeline cancellation.
        if hasattr(self, "_build_job_scheduler") and self._build_job_scheduler is not None:
            self._build_job_scheduler.shutdown()

        # 3. Release pipeline wrappers.
        for i in range(self.layer_num):
            pipeline = self.imi_pipelines[i]
            if pipeline is None:
                continue
            try:
                pipeline.close()
            except Exception:
                pass
            self.imi_pipelines[i] = None

        # 4. Clear GPU tensor dicts
        self.execution_buffer_keys_dict.clear()
        self.execution_buffer_values_dict.clear()
        self.execution_buffer_key_ring_dict.clear()
        self.execution_buffer_value_ring_dict.clear()
        self.execution_buffer_ring_events.clear()
        self.execution_buffer_ring_next.clear()
        self.execution_buffer_fallback_keys_dict.clear()
        self.execution_buffer_fallback_values_dict.clear()
        self.valid_lengths_dict.clear()
        self.static_len_tensor_dict.clear()
        self.static_lengths_dict.clear()
        self.nprobe_tensor_dict.clear()
        self.similarity_buffer_dict.clear()
        self.selected_cluster_buffer_dict.clear()
        self.gpu_cache_promote_streams.clear()
        self.gpu_cache_promote_start_events = [None] * self.layer_num

        # 5. Clear GPU tensor lists
        self.steady_zone_keys.clear()
        self.steady_zone_values.clear()
        self.cache_keys.clear()
        self.cache_values.clear()
        self.cluster_sizes_gpu.clear()
        self.cluster_offsets_gpu.clear()
        self.centroids.clear()
        self.centroids_mask.clear()

        # 6. Clear pinned memory lists
        self.list_keys.clear()
        self.list_values.clear()
        self.gpu_cache_aware_hit_unit_idices.clear()
        self.gpu_cache_aware_hit_unit_sizes.clear()
        self.gpu_cache_aware_hit_unit_sizes_cumsum.clear()
        self.gpu_cache_aware_hit_num_units.clear()
        self.gpu_cache_aware_miss_unit_idices.clear()
        self.gpu_cache_aware_miss_unit_sizes.clear()
        self.gpu_cache_aware_miss_unit_sizes_cumsum.clear()
        self.gpu_cache_aware_miss_num_units.clear()
        self.gpu_cache_aware_miss_copy_unit_idices.clear()
        self.gpu_cache_aware_miss_copy_unit_sizes.clear()
        self.gpu_cache_aware_miss_copy_unit_sizes_cumsum.clear()
        self.gpu_cache_aware_miss_copy_num_units.clear()
        self.gpu_cache_aware_miss_page_keys.clear()
        self.gpu_cache_aware_admit_unit_starts.clear()
        self.gpu_cache_promote_src_offsets.clear()
        self.gpu_cache_promote_dst_slots.clear()
        self.gpu_cache_promote_sizes.clear()
        self.gpu_cache_promote_hash_positions.clear()
        self.gpu_cache_promote_page_keys.clear()
        self.gpu_cache_promote_counts.clear()
        self.gpu_cache_hash_keys.clear()
        self.gpu_cache_hash_values.clear()
        self.gpu_cache_slot_keys.clear()
        self.gpu_cache_cluster_state.clear()
        self.gpu_cache_cluster_prev.clear()
        self.gpu_cache_cluster_next.clear()
        self.gpu_cache_hot_head.clear()
        self.gpu_cache_hot_tail.clear()
        self.gpu_cache_cold_head.clear()
        self.gpu_cache_cold_tail.clear()
        self.gpu_cache_hot_pages.clear()
        self.gpu_cache_cold_pages.clear()
        self.gpu_cache_free_slots.clear()
        self.gpu_cache_free_counts.clear()

        # 7. Clear CPU metadata
        self.cluster_sizes_cpu.clear()
        self.cluster_offsets_cpu.clear()
        self.centroids_cpu.clear()
        self.centroids_mask_cpu.clear()

        # 8. CPU construction resources are owned by IMIPipeline and released above.
    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
