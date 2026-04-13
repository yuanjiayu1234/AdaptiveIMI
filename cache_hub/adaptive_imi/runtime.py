import math
import os
import time

import torch


class AdaptiveIMIRuntimeMixin:
    def capture_cuda_graph(self):
        return None
    def prepare_cache(self, skip_prefetch: bool = False):
        if not self.build_index_when_prefilling:
            return
        if self.cache_prepared:
            if self.prefetch_enabled and not self.prefetch_done and not skip_prefetch:
                self._warmup_prefetch()
            return

        profile_prepare = os.getenv("IMI_PROFILE_PREPARE_CACHE", "0") == "1"
        if profile_prepare:
            start_ts = time.perf_counter()
            last_ts = start_ts

            def log_step(stage: str) -> None:
                nonlocal last_ts
                now = time.perf_counter()
                elapsed_ms = (now - last_ts) * 1000.0
                print(f"[IMI prepare_cache] {stage}={elapsed_ms:.2f} ms", flush=True)
                last_ts = now

        n_tokens = max(self.list_capacity, 1)
        total_clusters = max(int(n_tokens / 16), 1)
        if self.subspace_parts == 4:
            total_clusters_f = max(n_tokens / 16.0, 1.0)
            k = max(int(round(total_clusters_f ** 0.25)), 2)
            self.n_centroids = k ** 4 + 1
        else:
            k = max(int(math.sqrt(total_clusters)), 32)
            self.n_centroids = k * k + 1
        self.padding_cluster_id = self.n_centroids - 1
        self.nprobe = max(round((self.n_centroids - 1) * self.retrieval_budget), 1)
        self.nprobe = min(self.nprobe, self.n_centroids - 1)
        requested_retrieval_tokens = max(int(self.retrieval_budget * self.list_capacity), 1)
        self.max_retrieval_tokens = requested_retrieval_tokens
        self.budget_tokens = self.max_retrieval_tokens
        self.max_retrieval_pages = self._tokens_to_pages(self.max_retrieval_tokens)

        cache_cluster_num = (
            round(self.n_centroids * self.cache_ratio) if self.cache_ratio > 0.0 else self.nprobe * 3
        )
        requested_cache_pages = max(cache_cluster_num * self.pages_per_cluster, 1)
        self.cache_size = self._resolve_budget_pages(requested_cache_pages)
        self.cache_budget_pages = self.cache_size

        requested_buffer_pages = max(
            self.buffer_cluster_num,
            self._tokens_to_pages(self.max_retrieval_tokens),
        )
        self.buffer_size = self._resolve_budget_pages(requested_buffer_pages)
        self.buffer_budget_pages = self.buffer_size

        self.execution_stride = self.buffer_size * self.page_size + self.static_capacity
        self.retrieval_execution_stride = self.buffer_size * self.page_size

        if profile_prepare:
            log_step("sizes")

        if self.profile_cache_stats or self.profile_block_hit_rate:
            cache_tokens = self.cache_size * self.page_size
            buffer_tokens = self.buffer_size * self.page_size
            print(
                "[IMI cache] "
                f"n_centroids={self.n_centroids} nprobe={self.nprobe} "
                f"list_capacity={self.list_capacity} list_stride={self.list_stride} static_capacity={self.static_capacity} static_stride={self.static_stride} "
                f"cache_pages={self.cache_size} buffer_pages={self.buffer_size} "
                f"cache_tokens={cache_tokens} buffer_tokens={buffer_tokens} "
                f"retrieval_execution_tokens={self.retrieval_execution_stride} "
                f"retrieval_tokens={self.max_retrieval_tokens} retrieval_pages={self.max_retrieval_pages}"
            )

        if self.prefetch_enabled and self.prefetch_ratio > 0:
            self.prefetch_k = max(int(self.nprobe * self.prefetch_ratio), 1)
            self.prefetch_k = min(self.prefetch_k, self.n_centroids - 1)
        else:
            self.prefetch_enabled = False

        thread_pool_pointer = self.thread_pool.get()
        for ldx in range(self.layer_num):
            self.adpimi_index[ldx] = self.adpimi_index_cls(
                self.batch_size,
                self.kv_head,
                self.head_dim,
                self.nprobe,
                0,
                self.page_size,
                self.n_centroids,
                self.buffer_size,
                self.cache_size,
                self.core,
                thread_pool_pointer,
            )

        if profile_prepare:
            log_step("adpimi_index")

        if self.async_update_enabled and not self.allocated:
            self.async_update_enabled = False

        if self.async_update_enabled:
            for device in self.device_list:
                if device not in self.async_update_streams:
                    self.async_update_streams[device] = torch.cuda.Stream(device)
            for ldx in range(self.layer_num):
                device = self.layer_mapping[str(ldx)]
                if self.centroids[ldx] is None or self.cluster_sizes_gpu[ldx] is None:
                    continue
                self.centroids_shadow[ldx] = self.centroids[ldx].clone()
                self.centroid_counts[ldx] = self.cluster_sizes_gpu[ldx].to(torch.int32).clone()
                self.delta_counts[ldx] = torch.zeros_like(self.centroid_counts[ldx])
                self.delta_sums[ldx] = torch.zeros(
                    (self.batch_groups, self.n_centroids, self.head_dim),
                    dtype=torch.float32,
                    device=device,
                )
                self.delta_keys[ldx] = torch.zeros(
                    (self.batch_groups, self.async_update_delta_tokens, self.head_dim),
                    dtype=self.dtype,
                    device=device,
                )
                self.delta_values[ldx] = torch.zeros(
                    (self.batch_groups, self.async_update_delta_tokens, self.head_dim),
                    dtype=self.dtype,
                    device=device,
                )
                self.delta_cluster_ids[ldx] = torch.full(
                    (self.batch_groups, self.async_update_delta_tokens),
                    self.padding_cluster_id,
                    dtype=torch.int64,
                    device=device,
                )
                self.delta_counts_gpu[ldx] = torch.zeros(
                    (self.batch_groups),
                    dtype=torch.int32,
                    device=device,
                )
                self.async_update_buffers[ldx] = {
                    "keys": torch.empty(
                        (self.batch_groups, self.async_update_batch, self.head_dim),
                        dtype=self.dtype,
                        device=device,
                    ),
                    "values": torch.empty(
                        (self.batch_groups, self.async_update_batch, self.head_dim),
                        dtype=self.dtype,
                        device=device,
                    ),
                    "count": 0,
                    "event": torch.cuda.Event(),
                    "delta_event": torch.cuda.Event(),
                    "delta_write_pos": torch.zeros(
                        (self.batch_groups),
                        dtype=torch.int64,
                        device=device,
                    ),
                }

        if profile_prepare:
            log_step("async_update")

        for _ in range(self.layer_num):
            self.hit_unit_idices.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.hit_unit_sizes.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.hit_unit_sizes_cumsum.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.hit_num_units.append(
                torch.zeros((self.batch_groups), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.miss_unit_idices.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.miss_unit_sizes.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.miss_unit_sizes_cumsum.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.miss_num_units.append(
                torch.zeros((self.batch_groups), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.update_buffer_indices.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.update_unit_sizes.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.update_cache_indices.append(
                torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.update_num_units.append(
                torch.zeros((self.batch_groups), dtype=torch.int32, pin_memory=True).contiguous()
            )
            self.cluster_ids.append(
                torch.empty((self.batch_groups, self.nprobe), dtype=torch.int64, pin_memory=True).contiguous()
            )

        if profile_prepare:
            log_step("cpu_buffers")

        placeholder = torch.empty((self.batch_groups, 0, self.head_dim), dtype=self.dtype, pin_memory=True)
        placeholder_list = placeholder.unsqueeze(0)

        for ldx in range(self.layer_num):
            self.adpimi_index[ldx].set_indices(
                self.hit_unit_idices[ldx],
                self.hit_unit_sizes[ldx],
                self.hit_unit_sizes_cumsum[ldx],
                self.hit_num_units[ldx],
                self.miss_unit_idices[ldx],
                self.miss_unit_sizes[ldx],
                self.miss_unit_sizes_cumsum[ldx],
                self.miss_num_units[ldx],
                self.update_buffer_indices[ldx],
                self.update_unit_sizes[ldx],
                self.update_cache_indices[ldx],
                self.update_num_units[ldx],
                self.cluster_ids[ldx],
            )
            list_keys = self.list_keys[ldx] if self.list_keys[ldx] is not None else placeholder_list
            list_values = self.list_values[ldx] if self.list_values[ldx] is not None else placeholder_list
            self.adpimi_index[ldx].set_kv(
                list_keys,
                list_values,
                placeholder,
                placeholder,
            )

        if profile_prepare:
            log_step("wave_bind")

        for ldx in range(self.layer_num):
            device = self.layer_mapping[str(ldx)]
            cluster_sizes = self.cluster_sizes_cpu[ldx]
            cluster_offsets = self.cluster_offsets_cpu[ldx]
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
                self.cluster_sizes_cpu[ldx] = cluster_sizes
                self.cluster_offsets_cpu[ldx] = cluster_offsets

            if self.cluster_sizes_gpu[ldx] is None:
                self.cluster_sizes_gpu[ldx] = cluster_sizes.to(device)
            if self.centroids_mask[ldx] is None:
                self.centroids_mask[ldx] = torch.ones(
                    (self.batch_groups, self.n_centroids),
                    dtype=torch.bool,
                    device=device,
                )
            if self.centroids[ldx] is None:
                self.centroids[ldx] = torch.zeros(
                    (self.batch_groups, self.n_centroids, self.head_dim),
                    dtype=torch.float32,
                    device=device,
                )

            centroids_cpu = self.centroids_cpu[ldx]
            if centroids_cpu is not None:
                self.centroids[ldx].copy_(centroids_cpu)
            centroids_mask_cpu = self.centroids_mask_cpu[ldx]
            if centroids_mask_cpu is not None:
                self.centroids_mask[ldx].copy_(centroids_mask_cpu)

            if os.getenv("IMI_DEBUG_INDEX_METADATA", "0") == "1":
                nonzero_clusters = int((cluster_sizes > 0).sum().item()) if cluster_sizes is not None else 0
                unmasked_clusters = int((~self.centroids_mask[ldx]).sum().item()) if self.centroids_mask[ldx] is not None else 0
                print(
                    json.dumps({
                        "tag": "IMI_META_PREPARE",
                        "layer_idx": int(ldx),
                        "nonzero_clusters": nonzero_clusters,
                        "unmasked_clusters": unmasked_clusters,
                        "n_centroids": int(self.n_centroids),
                    }, ensure_ascii=False),
                    flush=True,
                )

            if cluster_sizes is not None and self.centroids_mask[ldx] is not None:
                nonzero_clusters = int((cluster_sizes > 0).sum().item())
                unmasked_clusters = int((~self.centroids_mask[ldx]).sum().item())
                if self.layer_metadata[ldx] is not None and len(self.layer_metadata[ldx]) > 0 and nonzero_clusters == 0 and unmasked_clusters == 0:
                    raise RuntimeError(
                        f"AdaptiveIMI invalid decode metadata on layer {ldx}: "
                        "CPU metadata exists but GPU metadata remains empty."
                    )

            self.adpimi_index[ldx].set_cluster_metadata(cluster_sizes, cluster_offsets, 0)

        if profile_prepare:
            log_step("centroids")

        for ldx in range(self.layer_num):
            device = self.layer_mapping[str(ldx)]
            self.cache_keys.append(
                torch.zeros(
                    (self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                    dtype=self.dtype,
                    device=device,
                ).contiguous()
            )
            self.cache_values.append(
                torch.zeros(
                    (self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                    dtype=self.dtype,
                    device=device,
                ).contiguous()
            )

        if profile_prepare:
            log_step("cache_kv")

        for device_idx in self.device_list:
            self.execution_buffer_keys_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.execution_stride, 1, self.head_dim),
                dtype=self.dtype,
                device=device_idx,
            ).contiguous()
            self.execution_buffer_values_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.execution_stride, 1, self.head_dim),
                dtype=self.dtype,
                device=device_idx,
            ).contiguous()
            self.valid_lengths_dict[device_idx] = torch.zeros(
                (self.batch_groups),
                dtype=torch.int32,
                device=device_idx,
            ).contiguous()
            self.static_len_tensor_dict[device_idx] = torch.tensor(
                self.static_pattern_total,
                dtype=torch.int32,
                device=device_idx,
            )
            self.static_lengths_dict[device_idx] = torch.zeros(
                (self.batch_groups),
                dtype=torch.int32,
                device=device_idx,
            ).contiguous()
            self.similarity_buffer_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.n_centroids),
                dtype=torch.float32,
                device=device_idx,
            ).contiguous()

        if profile_prepare:
            log_step("execution_buffers")


        self.attn_func = self.sparse_attention
        self.cache_prepared = True
        if self.prefetch_enabled and not skip_prefetch:
            self._warmup_prefetch()

        if profile_prepare:
            log_step("prefetch")
            total_ms = (time.perf_counter() - start_ts) * 1000.0
            print(f"[IMI prepare_cache] total={total_ms:.2f} ms", flush=True)
    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        self.steady_zone_keys[layer_idx][:, :, self.static_pattern_total, :] = key_states[:, 0, :, :]
        self.steady_zone_values[layer_idx][:, :, self.static_pattern_total, :] = value_states[:, 0, :, :]

        if layer_idx == 0:
            self.decode_debug_step += 1

        if self.async_update_enabled:
            self._schedule_async_update(layer_idx, key_states, value_states)

        if layer_idx == self.layer_num - 1:
            self.context += 1
            self.static_pattern_total += 1

        return None, None
    def dense_attention(self, queries, layer_idx, static_len):
        attn_out = weighted_flash_decoding(
            queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
            self.steady_zone_keys[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
            self.steady_zone_values[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
            previous_out=None,
            previous_lse=None,
            cache_seqlens=static_len,
            return_softmax_lse=False,
        )
        return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)
