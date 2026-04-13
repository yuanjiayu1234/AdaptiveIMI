import os

import torch


class AdaptiveIMIAsyncUpdateMixin:
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
