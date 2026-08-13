import queue
import time

import torch


class AdaptiveIMIAsyncUpdateMixin:
    def _reset_generated_history(self) -> None:
        if not getattr(self, "async_update_enabled", False):
            return
        worker_queue = getattr(self, "generated_history_worker_queue", None)
        if worker_queue is not None:
            worker_queue.join()
        with self.generated_history_lock:
            self.generated_history_keys = [[] for _ in range(self.layer_num)]
            self.generated_history_values = [[] for _ in range(self.layer_num)]
            self.generated_history_centroid_blocks = [[] for _ in range(self.layer_num)]
        while True:
            try:
                self.generated_commit_queue.get_nowait()
            except queue.Empty:
                break
        self.generated_spill_states = [
            [None] * self.async_update_spill_slots for _ in range(self.layer_num)
        ]
        self.generated_spill_next_slots = [0] * self.layer_num

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
                self._append_generated_history_chunk(*task)
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

    def _apply_async_centroid_updates(self, layer_idx: int, device: str) -> None:
        if not self.async_update_enabled:
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
        self.generated_spill_keys[layer_idx][:, spill_slot].copy_(chunk_keys)
        self.generated_spill_values[layer_idx][:, spill_slot].copy_(chunk_values)
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
        gathered_keys = self.generated_spill_keys[layer_idx][:, active_slots].reshape(
            self.batch_groups, len(active_slots) * interval, dim
        )
        gathered_values = self.generated_spill_values[layer_idx][:, active_slots].reshape(
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
