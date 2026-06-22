import torch


class AdaptiveIMIAsyncUpdateMixin:
    def _apply_async_centroid_updates(self, layer_idx: int, device: str) -> None:
        return

    def current_static_len(self, layer_idx: int) -> int:
        if not self.async_update_enabled:
            return int(self.static_pattern_total)
        return int(self.static_layer_lengths[layer_idx])

    def _flush_delta_chunk(self, layer_idx: int) -> None:
        active = int(self.delta_active_chunks[layer_idx])
        max_chunks = int(self.async_update_max_delta_chunks)
        interval = int(self.async_update_interval)
        recent_count = int(self.generated_recent_counts[layer_idx])
        if max_chunks <= 0 or active >= max_chunks or recent_count < interval:
            return

        base = int(self.base_static_pattern_total)
        key_rows = self.steady_zone_keys[layer_idx].view(self.batch_groups, self.static_capacity, self.head_dim)
        value_rows = self.steady_zone_values[layer_idx].view(self.batch_groups, self.static_capacity, self.head_dim)
        chunk_keys = key_rows[:, base : base + interval, :]
        chunk_values = value_rows[:, base : base + interval, :]

        self.delta_mini_keys[layer_idx][:, active, :, :].copy_(chunk_keys)
        self.delta_mini_values[layer_idx][:, active, :, :].copy_(chunk_values)
        self.delta_mini_centroids[layer_idx][:, active, :].copy_(chunk_keys.float().mean(dim=1))
        self.delta_active_chunks[layer_idx] = active + 1

        remaining = recent_count - interval
        if remaining > 0:
            remaining_keys = key_rows[:, base + interval : base + recent_count, :].clone()
            remaining_values = value_rows[:, base + interval : base + recent_count, :].clone()
            key_rows[:, base : base + remaining, :].copy_(remaining_keys)
            value_rows[:, base : base + remaining, :].copy_(remaining_values)
        self.generated_recent_counts[layer_idx] = remaining
        self.static_layer_lengths[layer_idx] = base + remaining

    def _append_delta_to_execution(
        self,
        layer_idx: int,
        device: str,
        query_grouped: torch.Tensor,
        execution_keys: torch.Tensor,
        execution_values: torch.Tensor,
    ) -> int:
        active = int(self.delta_active_chunks[layer_idx]) if self.async_update_enabled else 0
        if active <= 0:
            return 0
        delta_nprobe = min(int(self.async_update_delta_nprobe), active)
        if delta_nprobe <= 0:
            return 0

        centroids = self.delta_mini_centroids[layer_idx][:, :active, :]
        scores = torch.bmm(query_grouped.float(), centroids.transpose(1, 2)).sum(dim=1)
        selected = torch.topk(scores, delta_nprobe, dim=-1, largest=True, sorted=False).indices

        interval = int(self.async_update_interval)
        dim = int(self.head_dim)
        gather_index = selected.view(self.batch_groups, delta_nprobe, 1, 1).expand(-1, -1, interval, dim)
        gathered_keys = torch.gather(
            self.delta_mini_keys[layer_idx][:, :active, :, :],
            1,
            gather_index,
        ).reshape(self.batch_groups, delta_nprobe * interval, dim)
        gathered_values = torch.gather(
            self.delta_mini_values[layer_idx][:, :active, :, :],
            1,
            gather_index,
        ).reshape(self.batch_groups, delta_nprobe * interval, dim)

        valid_lengths = self.valid_lengths_dict[device]
        append_tokens = delta_nprobe * interval
        dst_positions = (
            valid_lengths.to(torch.int64).view(self.batch_groups, 1)
            + torch.arange(append_tokens, device=device, dtype=torch.int64).view(1, append_tokens)
        )
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
