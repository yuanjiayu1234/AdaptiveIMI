import queue
import sys
import time
import traceback

import torch
from library.AdaptiveIMI.cpp_extensions import (
    gather_selected_cluster_delta_pages,
)
from library.AdaptiveIMI.imi_adapter import get_imi_kernels


imi_gpu_kernels = get_imi_kernels()


def _require_generated_page_extensions() -> None:
    required_gpu = (
        "assign_generated_to_centroids",
        "pack_generated_pages",
        "publish_generated_page_metadata",
        "apply_generated_centroid_updates",
    )
    missing = [name for name in required_gpu if not hasattr(imi_gpu_kernels, name)]
    if gather_selected_cluster_delta_pages is None:
        missing.append("gather_selected_cluster_delta_pages")
    if missing:
        raise RuntimeError(
            "async_update.mode=unified_pages requires rebuilt AdaptiveIMI CUDA extensions; missing "
            + ", ".join(missing)
            + ". Run: cd library/AdaptiveIMI/cpp_extensions && "
            "IMI_BUILD_MODULES=gpu_cluster_manager_cpp,Copy MAX_JOBS=1 python setup.py build_ext --inplace"
        )


class AdaptiveIMIAsyncUpdateMixin:
    def _reset_generated_history(self) -> None:
        if not getattr(self, "async_update_enabled", False):
            return
        if getattr(self, "async_update_pages", False):
            _require_generated_page_extensions()
        worker_queue = getattr(self, "generated_history_worker_queue", None)
        if worker_queue is not None:
            worker_queue.join()
        self._raise_generated_history_worker_error()
        self.generated_history_worker_error_context = ""
        self.generated_history_worker_traceback = ""
        with self.generated_history_lock:
            self.generated_history_keys = [[] for _ in range(self.layer_num)]
            self.generated_history_values = [[] for _ in range(self.layer_num)]
            self.generated_history_centroid_blocks = [[] for _ in range(self.layer_num)]
        while True:
            try:
                self.generated_commit_queue.get_nowait()
            except queue.Empty:
                break
        for commit_queue in self.generated_page_commit_queues:
            while True:
                try:
                    commit_queue.get_nowait()
                except queue.Empty:
                    break
        if self.async_update_pages:
            for layer_idx in range(self.layer_num):
                heads = self.generated_page_cluster_heads_cpu[layer_idx]
                if heads is None:
                    continue
                heads.fill_(-1)
                self.generated_page_cluster_tails_cpu[layer_idx].fill_(-1)
                self.generated_page_prev_cpu[layer_idx].fill_(-1)
                self.generated_page_sizes_cpu[layer_idx].zero_()
                self.generated_page_next_cpu[layer_idx].zero_()
                self.generated_page_cluster_tails_gpu[layer_idx].fill_(-1)
                self.generated_page_prev_gpu[layer_idx].fill_(-1)
                self.generated_page_sizes_gpu[layer_idx].zero_()
                self.generated_cluster_counts_gpu[layer_idx].zero_()
        self.generated_spill_states = [
            [None] * self.async_update_spill_slots for _ in range(self.layer_num)
        ]
        self.generated_spill_next_slots = [0] * self.layer_num

    def _raise_generated_history_worker_error(self) -> None:
        error = getattr(self, "generated_history_worker_error", None)
        if error is not None:
            context = getattr(self, "generated_history_worker_error_context", "")
            detail = f" ({context})" if context else ""
            raise RuntimeError(
                "AdaptiveIMI generated-token background update failed"
                f"{detail}: {type(error).__name__}: {error}"
            ) from error

    def _shutdown_generated_history_worker(self) -> None:
        worker_queue = getattr(self, "generated_history_worker_queue", None)
        worker = getattr(self, "generated_history_worker", None)
        if worker_queue is None or worker is None:
            return
        worker_queue.join()
        worker_queue.put(None)
        worker.join(timeout=1.0)
        self.generated_history_worker = None

    def _generated_history_worker_loop(self) -> None:
        while True:
            task = self.generated_history_worker_queue.get()
            try:
                if task is None:
                    return
                if self.generated_history_worker_error is None:
                    try:
                        self._append_generated_history_chunk(*task)
                    except BaseException as exc:
                        # Keep draining queued tasks, but poison the online
                        # path.  In particular, reset/cleanup must never hang
                        # behind a failed worker with unfinished queue items.
                        if self.generated_history_worker_error is None:
                            layer_idx, chunk_id, spill_slot = (int(task[0]), int(task[1]), int(task[2]))
                            context = (
                                f"layer={layer_idx}, chunk={chunk_id}, "
                                f"spill_slot={spill_slot}"
                            )
                            self.generated_history_worker_error = exc
                            self.generated_history_worker_error_context = context
                            self.generated_history_worker_traceback = traceback.format_exc()
                            print(
                                "[AdaptiveIMI async-update] background worker failed "
                                f"({context}):\n{self.generated_history_worker_traceback}",
                                file=sys.stderr,
                                flush=True,
                            )
            finally:
                self.generated_history_worker_queue.task_done()

    def _append_generated_history_chunk(
        self,
        layer_idx: int,
        chunk_id: int,
        spill_slot: int,
        spill_keys: torch.Tensor,
        spill_values: torch.Tensor,
        producer_event: torch.cuda.Event,
    ) -> None:
        if self.async_update_pages:
            self._append_generated_page_chunk(
                layer_idx, chunk_id, spill_slot, spill_keys, spill_values, producer_event
            )
            return
        block_size = int(self.async_update_cpu_block_chunks)
        block_index = int(chunk_id) // block_size
        block_offset = int(chunk_id) % block_size
        block_shape = (
            self.batch_groups,
            block_size,
            self.async_update_interval,
            self.head_dim,
        )
        with self.generated_history_lock:
            key_blocks = self.generated_history_keys[layer_idx]
            value_blocks = self.generated_history_values[layer_idx]
            while len(key_blocks) <= block_index:
                key_blocks.append(torch.empty(block_shape, dtype=self.dtype, pin_memory=True))
                value_blocks.append(torch.empty(block_shape, dtype=self.dtype, pin_memory=True))
            target_keys = key_blocks[block_index][:, block_offset]
            target_values = value_blocks[block_index][:, block_offset]

        torch.cuda.set_device(spill_keys.device)
        stream_key = str(spill_keys.device)
        stream = self.generated_history_worker_streams.get(stream_key)
        if stream is None:
            stream = torch.cuda.Stream(device=spill_keys.device)
            self.generated_history_worker_streams[stream_key] = stream
        stream.wait_event(producer_event)
        with torch.cuda.stream(stream):
            target_keys.copy_(spill_keys[:, spill_slot], non_blocking=True)
            target_values.copy_(spill_values[:, spill_slot], non_blocking=True)
            copy_event = torch.cuda.Event()
            copy_event.record(stream)
        copy_event.synchronize()
        centroids = torch.empty(
            (self.batch_groups, self.head_dim),
            dtype=torch.float32,
            pin_memory=True,
        )
        centroids.copy_(target_keys.float().mean(dim=1))
        self.generated_commit_queue.put((layer_idx, chunk_id, spill_slot, centroids))

    def _append_generated_page_chunk(
        self,
        layer_idx: int,
        chunk_id: int,
        spill_slot: int,
        spill_keys: torch.Tensor,
        spill_values: torch.Tensor,
        producer_event: torch.cuda.Event,
    ) -> None:
        """D2H once, then pack one immutable page commit in native CPU code."""
        _require_generated_page_extensions()
        torch.cuda.set_device(spill_keys.device)
        stream_key = str(spill_keys.device)
        stream = self.generated_history_worker_streams.get(stream_key)
        if stream is None:
            stream = torch.cuda.Stream(device=spill_keys.device)
            self.generated_history_worker_streams[stream_key] = stream
        staging_keys = self.generated_page_staging_keys[layer_idx][spill_slot]
        staging_values = self.generated_page_staging_values[layer_idx][spill_slot]
        staging_assignments = self.generated_page_staging_assignments[layer_idx][spill_slot]
        stream.wait_event(producer_event)
        with torch.cuda.stream(stream):
            staging_keys.copy_(spill_keys[spill_slot], non_blocking=True)
            staging_values.copy_(spill_values[spill_slot], non_blocking=True)
            staging_assignments.copy_(
                self.generated_page_assignments_gpu[layer_idx][spill_slot], non_blocking=True
            )
            copied = torch.cuda.Event()
            copied.record(stream)
        copied.synchronize()

        # The worker is the sole CPU writer of the append-only page pool.
        # It emits deduplicated changed IDs; no Python loop touches tokens,
        # clusters, or pages.
        imi_gpu_kernels.pack_generated_pages(
            staging_assignments,
            staging_keys,
            staging_values,
            self.generated_page_keys_cpu[layer_idx],
            self.generated_page_values_cpu[layer_idx],
            self.generated_page_cluster_heads_cpu[layer_idx],
            self.generated_page_cluster_tails_cpu[layer_idx],
            self.generated_page_prev_cpu[layer_idx],
            self.generated_page_sizes_cpu[layer_idx],
            self.generated_page_next_cpu[layer_idx],
            self.generated_page_commit_cluster_ids_cpu[layer_idx][spill_slot],
            self.generated_page_commit_cluster_heads_cpu[layer_idx][spill_slot],
            self.generated_page_commit_cluster_tails_cpu[layer_idx][spill_slot],
            self.generated_page_commit_page_ids_cpu[layer_idx][spill_slot],
            self.generated_page_commit_page_prev_cpu[layer_idx][spill_slot],
            self.generated_page_commit_page_sizes_cpu[layer_idx][spill_slot],
            self.generated_page_commit_cluster_sums_cpu[layer_idx][spill_slot],
            self.generated_page_commit_cluster_counts_cpu[layer_idx][spill_slot],
        )
        self.generated_page_commit_queues[layer_idx].put((chunk_id, spill_slot))

    def _apply_async_centroid_updates(self, layer_idx: int, device: str) -> None:
        if not self.async_update_enabled:
            return
        self._raise_generated_history_worker_error()
        if self.async_update_pages:
            self._apply_generated_page_updates(layer_idx, device)
            return
        deferred = []
        while True:
            try:
                commit_layer, chunk_id, spill_slot, centroids = self.generated_commit_queue.get_nowait()
            except queue.Empty:
                break
            if int(commit_layer) != int(layer_idx):
                deferred.append((commit_layer, chunk_id, spill_slot, centroids))
                continue
            block_size = int(self.async_update_cpu_block_chunks)
            block_index = int(chunk_id) // block_size
            block_offset = int(chunk_id) % block_size
            blocks = self.generated_history_centroid_blocks[commit_layer]
            while len(blocks) <= block_index:
                blocks.append(
                    torch.empty(
                        (self.batch_groups, block_size, self.head_dim),
                        dtype=torch.float32,
                        device=device,
                    )
                )
            blocks[block_index][:, block_offset].copy_(centroids, non_blocking=True)
            self.generated_published_chunks[commit_layer] = max(
                int(self.generated_published_chunks[commit_layer]), int(chunk_id) + 1
            )
            self.generated_spill_states[commit_layer][spill_slot] = None
        for commit in deferred:
            self.generated_commit_queue.put(commit)

    def _apply_generated_page_updates(self, layer_idx: int, device: str) -> None:
        """Publish a complete background page on the current decode stream.

        The page contents were finalized before this queue entry existed.  The
        two native kernels therefore make its linked-list metadata and the
        corresponding full centroid visible in exactly the same stream order
        as subsequent centroid scoring and UVA gathering.
        """
        commit_queue = self.generated_page_commit_queues[layer_idx]
        while True:
            try:
                _, spill_slot = commit_queue.get_nowait()
            except queue.Empty:
                break
            ids = self.generated_page_commit_cluster_ids_gpu[layer_idx][spill_slot]
            tails = self.generated_page_commit_cluster_tails_gpu[layer_idx][spill_slot]
            page_ids = self.generated_page_commit_page_ids_gpu[layer_idx][spill_slot]
            page_prev = self.generated_page_commit_page_prev_gpu[layer_idx][spill_slot]
            page_sizes = self.generated_page_commit_page_sizes_gpu[layer_idx][spill_slot]
            sums = self.generated_page_commit_cluster_sums_gpu[layer_idx][spill_slot]
            counts = self.generated_page_commit_cluster_counts_gpu[layer_idx][spill_slot]
            ids.copy_(self.generated_page_commit_cluster_ids_cpu[layer_idx][spill_slot], non_blocking=True)
            tails.copy_(self.generated_page_commit_cluster_tails_cpu[layer_idx][spill_slot], non_blocking=True)
            page_ids.copy_(self.generated_page_commit_page_ids_cpu[layer_idx][spill_slot], non_blocking=True)
            page_prev.copy_(self.generated_page_commit_page_prev_cpu[layer_idx][spill_slot], non_blocking=True)
            page_sizes.copy_(self.generated_page_commit_page_sizes_cpu[layer_idx][spill_slot], non_blocking=True)
            sums.copy_(self.generated_page_commit_cluster_sums_cpu[layer_idx][spill_slot], non_blocking=True)
            counts.copy_(self.generated_page_commit_cluster_counts_cpu[layer_idx][spill_slot], non_blocking=True)
            imi_gpu_kernels.publish_generated_page_metadata(
                ids,
                tails,
                page_ids,
                page_prev,
                page_sizes,
                self.generated_page_cluster_tails_gpu[layer_idx],
                self.generated_page_prev_gpu[layer_idx],
                self.generated_page_sizes_gpu[layer_idx],
            )
            imi_gpu_kernels.apply_generated_centroid_updates(
                ids,
                sums,
                counts,
                self.centroids[layer_idx],
                self.cluster_sizes_gpu[layer_idx],
                self.generated_cluster_counts_gpu[layer_idx],
            )
            self.generated_spill_states[layer_idx][spill_slot] = None

    def current_static_len(self, layer_idx: int) -> int:
        if not self.async_update_enabled:
            return int(self.static_pattern_total)
        return int(self.static_layer_lengths[layer_idx])

    def _acquire_generated_spill_slot(self, layer_idx: int) -> int:
        slots = self.generated_spill_states[layer_idx]
        start = int(self.generated_spill_next_slots[layer_idx])
        while True:
            for offset in range(len(slots)):
                slot = (start + offset) % len(slots)
                if slots[slot] is None:
                    self.generated_spill_next_slots[layer_idx] = (slot + 1) % len(slots)
                    return slot
            self._apply_async_centroid_updates(layer_idx, str(self.layer_mapping[str(layer_idx)]))
            if any(slot is None for slot in slots):
                continue
            time.sleep(0.0005)

    def _flush_delta_chunk(self, layer_idx: int) -> None:
        interval = int(self.async_update_interval)
        recent_count = int(self.generated_recent_counts[layer_idx])
        if recent_count < interval:
            return

        spill_slot = self._acquire_generated_spill_slot(layer_idx)
        base = int(self.base_static_pattern_total)
        key_rows = self.steady_zone_keys[layer_idx].view(self.batch_groups, self.static_capacity, self.head_dim)
        value_rows = self.steady_zone_values[layer_idx].view(self.batch_groups, self.static_capacity, self.head_dim)
        chunk_keys = key_rows[:, base : base + interval, :]
        chunk_values = value_rows[:, base : base + interval, :]
        self.generated_spill_keys[layer_idx][spill_slot].copy_(chunk_keys)
        self.generated_spill_values[layer_idx][spill_slot].copy_(chunk_values)
        if self.async_update_pages:
            imi_gpu_kernels.assign_generated_to_centroids(
                self.generated_spill_keys[layer_idx][spill_slot],
                self.centroids[layer_idx],
                self.centroids_mask[layer_idx],
                self.generated_page_assignments_gpu[layer_idx][spill_slot],
            )
        producer_event = torch.cuda.Event()
        producer_event.record(torch.cuda.current_stream(chunk_keys.device))

        chunk_id = int(self.generated_next_chunk_ids[layer_idx])
        self.generated_next_chunk_ids[layer_idx] = chunk_id + 1
        self.generated_spill_states[layer_idx][spill_slot] = (chunk_id, spill_slot)
        self.generated_history_worker_queue.put(
            (
                layer_idx,
                chunk_id,
                spill_slot,
                self.generated_spill_keys[layer_idx],
                self.generated_spill_values[layer_idx],
                producer_event,
            )
        )

        remaining = recent_count - interval
        if remaining > 0:
            remaining_keys = key_rows[:, base + interval : base + recent_count, :].clone()
            remaining_values = value_rows[:, base + interval : base + recent_count, :].clone()
            key_rows[:, base : base + remaining, :].copy_(remaining_keys)
            value_rows[:, base : base + remaining, :].copy_(remaining_values)
        self.generated_recent_counts[layer_idx] = remaining
        self.static_layer_lengths[layer_idx] = base + remaining

    def _append_generated_history_to_execution(
        self,
        layer_idx: int,
        device: str,
        query_grouped: torch.Tensor,
        execution_keys: torch.Tensor,
        execution_values: torch.Tensor,
    ) -> int:
        published = int(self.generated_published_chunks[layer_idx])
        nprobe = min(int(self.async_update_delta_nprobe), published)
        if nprobe <= 0:
            return 0
        block_size = int(self.async_update_cpu_block_chunks)
        candidates = []
        for block_index, centroids in enumerate(self.generated_history_centroid_blocks[layer_idx]):
            available = min(block_size, published - block_index * block_size)
            if available <= 0:
                break
            scores = torch.bmm(query_grouped.float(), centroids[:, :available].transpose(1, 2)).sum(dim=1)
            local_count = min(nprobe, available)
            local_scores, local_indices = torch.topk(scores, local_count, dim=-1, largest=True, sorted=False)
            candidates.append((local_scores, local_indices + block_index * block_size))
        if not candidates:
            return 0
        candidate_scores = torch.cat([candidate[0] for candidate in candidates], dim=-1)
        candidate_indices = torch.cat([candidate[1] for candidate in candidates], dim=-1)
        _, selected_positions = torch.topk(candidate_scores, nprobe, dim=-1, largest=True, sorted=False)
        selected = torch.gather(candidate_indices, 1, selected_positions)

        interval = int(self.async_update_interval)
        dim = int(self.head_dim)
        gathered_keys = torch.empty((self.batch_groups, nprobe, interval, dim), dtype=self.dtype, pin_memory=True)
        gathered_values = torch.empty(
            (self.batch_groups, nprobe, interval, dim),
            dtype=self.dtype,
            pin_memory=True,
        )
        with self.generated_history_lock:
            for group_idx in range(self.batch_groups):
                for selection_idx, chunk_id in enumerate(selected[group_idx].tolist()):
                    block_index, block_offset = divmod(int(chunk_id), block_size)
                    gathered_keys[group_idx, selection_idx].copy_(
                        self.generated_history_keys[layer_idx][block_index][group_idx, block_offset]
                    )
                    gathered_values[group_idx, selection_idx].copy_(
                        self.generated_history_values[layer_idx][block_index][group_idx, block_offset]
                    )
        gathered_keys = gathered_keys.to(device=device, non_blocking=True).reshape(self.batch_groups, nprobe * interval, dim)
        gathered_values = gathered_values.to(device=device, non_blocking=True).reshape(self.batch_groups, nprobe * interval, dim)
        valid_lengths = self.valid_lengths_dict[device]
        append_tokens = nprobe * interval
        dst_positions = valid_lengths.to(torch.int64).view(self.batch_groups, 1) + torch.arange(
            append_tokens, device=device, dtype=torch.int64
        ).view(1, append_tokens)
        dst_index = dst_positions.unsqueeze(-1).expand(-1, -1, dim)
        execution_keys.view(self.batch_groups, self.execution_stride, dim).scatter_(1, dst_index, gathered_keys)
        execution_values.view(self.batch_groups, self.execution_stride, dim).scatter_(1, dst_index, gathered_values)
        valid_lengths.add_(append_tokens)
        return append_tokens

    def _append_generated_pages_to_execution(
        self,
        layer_idx: int,
        selected_clusters: torch.Tensor,
        execution_keys: torch.Tensor,
        execution_values: torch.Tensor,
    ) -> int:
        _require_generated_page_extensions()
        budget = max(int(self.async_update_total_budget) - int(self.max_retrieval_tokens), 0)
        if budget <= 0:
            return 0
        gather_selected_cluster_delta_pages(
            selected_clusters,
            self.generated_page_cluster_tails_gpu[layer_idx],
            self.generated_page_prev_gpu[layer_idx],
            self.generated_page_sizes_gpu[layer_idx],
            self.generated_page_keys_cpu[layer_idx],
            self.generated_page_values_cpu[layer_idx],
            execution_keys.view(self.batch_groups, self.execution_stride, self.head_dim),
            execution_values.view(self.batch_groups, self.execution_stride, self.head_dim),
            self.valid_lengths_dict[str(self.layer_mapping[str(layer_idx)])],
            int(self.async_update_pages_per_probe),
            budget,
        )
        return budget

    def _append_inflight_spills_to_execution(
        self,
        layer_idx: int,
        device: str,
        execution_keys: torch.Tensor,
        execution_values: torch.Tensor,
    ) -> int:
        active_slots = [slot for slot, state in enumerate(self.generated_spill_states[layer_idx]) if state is not None]
        if not active_slots:
            return 0
        interval = int(self.async_update_interval)
        dim = int(self.head_dim)
        slot_ids = torch.tensor(active_slots, dtype=torch.long, device=device)
        gathered_keys = self.generated_spill_keys[layer_idx].index_select(0, slot_ids).transpose(0, 1).reshape(
            self.batch_groups, len(active_slots) * interval, dim
        )
        gathered_values = self.generated_spill_values[layer_idx].index_select(0, slot_ids).transpose(0, 1).reshape(
            self.batch_groups, len(active_slots) * interval, dim
        )
        valid_lengths = self.valid_lengths_dict[device]
        append_tokens = len(active_slots) * interval
        dst_positions = valid_lengths.to(torch.int64).view(self.batch_groups, 1) + torch.arange(
            append_tokens, device=device, dtype=torch.int64
        ).view(1, append_tokens)
        dst_index = dst_positions.unsqueeze(-1).expand(-1, -1, dim)
        execution_keys.view(self.batch_groups, self.execution_stride, dim).scatter_(1, dst_index, gathered_keys)
        execution_values.view(self.batch_groups, self.execution_stride, dim).scatter_(1, dst_index, gathered_values)
        valid_lengths.add_(append_tokens)
        return append_tokens

    def _schedule_async_update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        if not self.async_update_enabled:
            return
        base = int(self.base_static_pattern_total)
        recent_count = int(self.generated_recent_counts[layer_idx])
        if recent_count >= int(self.async_update_recent_capacity):
            self._flush_delta_chunk(layer_idx)
            recent_count = int(self.generated_recent_counts[layer_idx])

        write_pos = base + recent_count
        self.steady_zone_keys[layer_idx][:, :, write_pos, :].copy_(key_states[:, 0, :, :])
        self.steady_zone_values[layer_idx][:, :, write_pos, :].copy_(value_states[:, 0, :, :])
        recent_count += 1
        self.generated_recent_counts[layer_idx] = recent_count
        self.static_layer_lengths[layer_idx] = base + recent_count

        generated_index = int(self.generated_tokens_total) + 1
        if (
            generated_index > int(self.async_update_start_after_tokens)
            and (generated_index - int(self.async_update_start_after_tokens)) % int(self.async_update_interval) == 0
        ):
            self._flush_delta_chunk(layer_idx)
