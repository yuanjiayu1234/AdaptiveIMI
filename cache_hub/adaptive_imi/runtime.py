import json
import os
import time

import torch

from cache_hub.cluster_plan import choose_learned_k_values, product_k_values
from .attention_backend import compact_decode_attention


class AdaptiveIMIRuntimeMixin:
    def capture_cuda_graph(self):
        return None
    def prepare_cache(self):
        if not self.build_index_when_prefilling:
            return
        if self.cache_prepared:
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

        def tensor_matches(tensor, shape, dtype, device=None, pin_memory=None) -> bool:
            if tensor is None:
                return False
            if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
                return False
            if device is not None and torch.device(tensor.device) != torch.device(device):
                return False
            if pin_memory is not None and tensor.device.type == "cpu" and tensor.is_pinned() != pin_memory:
                return False
            return True

        def ensure_list_tensor(items, idx, shape, dtype, *, device=None, pin_memory=False, zero_new=True):
            while len(items) <= idx:
                items.append(None)
            if tensor_matches(items[idx], shape, dtype, device=device, pin_memory=pin_memory):
                return items[idx]
            if device is None:
                if zero_new:
                    tensor = torch.zeros(shape, dtype=dtype, pin_memory=pin_memory)
                else:
                    tensor = torch.empty(shape, dtype=dtype, pin_memory=pin_memory)
            else:
                if zero_new:
                    tensor = torch.zeros(shape, dtype=dtype, device=device)
                else:
                    tensor = torch.empty(shape, dtype=dtype, device=device)
            items[idx] = tensor.contiguous()
            return items[idx]

        def ensure_dict_tensor(items, key, shape, dtype, *, device, zero_new=True):
            tensor = items.get(key)
            if tensor_matches(tensor, shape, dtype, device=device):
                return tensor
            if zero_new:
                tensor = torch.zeros(shape, dtype=dtype, device=device)
            else:
                tensor = torch.empty(shape, dtype=dtype, device=device)
            items[key] = tensor.contiguous()
            return items[key]

        def next_power_of_two(value: int) -> int:
            value = max(int(value), 1)
            return 1 << (value - 1).bit_length()

        n_tokens = max(self.list_capacity, 1)
        kmeans_cfg = self.runtime_config.get("kmeans") or {}
        target_cluster_size = max(int(kmeans_cfg.get("target_cluster_size", kmeans_cfg.get("cluster_size", 16))), 1)
        max_clusters = 1
        for spec in getattr(self, "learned_partition_specs", []):
            if spec is None:
                raise RuntimeError("AdaptiveIMI hybrid_8way_v2 requires learned partition metadata for every layer")
            m = max(int(spec.get("subspace_parts", 1)), 1)
            k_values = kmeans_cfg.get("k_values")
            if k_values is None:
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
        requested_retrieval_tokens = max(int(self.retrieval_budget * self.list_capacity), 1)
        self.max_retrieval_tokens = requested_retrieval_tokens
        if self.async_update_enabled and self.async_update_pages:
            # Never take tokens away from the already validated prompt path.
            # The configured budget is a floor; generated leaves receive a
            # reserve in addition to the prompt retrieval allowance.
            self.async_update_total_budget = max(
                int(self.async_update_total_budget),
                int(self.max_retrieval_tokens) + int(self.async_update_generated_reserve),
            )
        self.budget_tokens = self.max_retrieval_tokens
        self.max_retrieval_pages = self._tokens_to_pages(self.max_retrieval_tokens)

        cache_cluster_num = (
            round(self.n_centroids * self.cache_ratio) if self.cache_ratio > 0.0 else self.nprobe * 3
        )
        requested_cache_pages = max(cache_cluster_num * self.pages_per_cluster, 1)
        self.cache_size = self._resolve_budget_pages(requested_cache_pages)
        if self.batch_size > 1:
            total_cache_pages = int(self.multibatch_cache_total_pages)
            if total_cache_pages > 0:
                self.cache_size = min(self.cache_size, max(1, total_cache_pages // int(self.batch_size)))
        self.cache_budget_pages = self.cache_size

        requested_buffer_pages = max(
            self.buffer_cluster_num,
            self._tokens_to_pages(self.max_retrieval_tokens),
        )
        self.buffer_size = self._resolve_budget_pages(requested_buffer_pages)
        self.buffer_budget_pages = self.buffer_size

        if self.async_update_enabled:
            if self.async_update_pages:
                overflow_budget = max(int(self.async_update_total_budget) - int(self.max_retrieval_tokens), 0)
                delta_execution_tokens = overflow_budget + (
                    int(self.async_update_spill_slots) * int(self.async_update_interval)
                )
            else:
                delta_execution_tokens = (
                    int(self.async_update_delta_nprobe) + int(self.async_update_spill_slots)
                ) * int(self.async_update_interval)
        else:
            delta_execution_tokens = 0
        self.execution_stride = self.buffer_size * self.page_size + self.static_capacity + delta_execution_tokens
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

        if self.async_update_enabled:
            for ldx in range(self.layer_num):
                device = self.layer_mapping[str(ldx)]
                self.generated_spill_keys[ldx] = torch.empty(
                    (self.async_update_spill_slots, self.batch_groups, self.async_update_interval, self.head_dim),
                    dtype=self.dtype,
                    device=device,
                ).contiguous()
                self.generated_spill_values[ldx] = torch.empty(
                    (self.async_update_spill_slots, self.batch_groups, self.async_update_interval, self.head_dim),
                    dtype=self.dtype,
                    device=device,
                ).contiguous()
                if self.async_update_pages:
                    # One open page per active base centroid is sufficient in
                    # addition to ceil(history/page_tokens) completed pages.
                    # The entire pool is allocated once; decode never grows or
                    # reallocates it.
                    history_tokens = max(int(self.async_update_page_capacity_tokens), 1)
                    page_tokens = int(self.async_update_page_tokens)
                    total_pages = (history_tokens + page_tokens - 1) // page_tokens + int(self.n_centroids)
                    self.generated_page_total_pages = total_pages
                    spill_shape = (self.async_update_spill_slots, self.batch_groups, self.async_update_interval)
                    spill_kv_shape = spill_shape + (self.head_dim,)
                    page_shape = (self.batch_groups, total_pages, page_tokens, self.head_dim)
                    cluster_shape = (self.batch_groups, self.n_centroids)
                    page_meta_shape = (self.batch_groups, total_pages)
                    self.generated_page_staging_keys[ldx] = torch.empty(spill_kv_shape, dtype=self.dtype, pin_memory=True)
                    self.generated_page_staging_values[ldx] = torch.empty(spill_kv_shape, dtype=self.dtype, pin_memory=True)
                    self.generated_page_staging_assignments[ldx] = torch.empty(spill_shape, dtype=torch.int32, pin_memory=True)
                    self.generated_page_assignments_gpu[ldx] = torch.empty(spill_shape, dtype=torch.int32, device=device)
                    self.generated_page_keys_cpu[ldx] = torch.empty(page_shape, dtype=self.dtype, pin_memory=True)
                    self.generated_page_values_cpu[ldx] = torch.empty(page_shape, dtype=self.dtype, pin_memory=True)
                    self.generated_page_cluster_heads_cpu[ldx] = torch.full(cluster_shape, -1, dtype=torch.int32, pin_memory=True)
                    self.generated_page_cluster_tails_cpu[ldx] = torch.full(cluster_shape, -1, dtype=torch.int32, pin_memory=True)
                    self.generated_page_prev_cpu[ldx] = torch.full(page_meta_shape, -1, dtype=torch.int32, pin_memory=True)
                    self.generated_page_sizes_cpu[ldx] = torch.zeros(page_meta_shape, dtype=torch.int32, pin_memory=True)
                    self.generated_page_next_cpu[ldx] = torch.zeros((self.batch_groups,), dtype=torch.int32, pin_memory=True)
                    self.generated_page_cluster_tails_gpu[ldx] = torch.full(cluster_shape, -1, dtype=torch.int32, device=device)
                    self.generated_page_prev_gpu[ldx] = torch.full(page_meta_shape, -1, dtype=torch.int32, device=device)
                    self.generated_page_sizes_gpu[ldx] = torch.zeros(page_meta_shape, dtype=torch.int32, device=device)
                    self.generated_cluster_counts_gpu[ldx] = torch.zeros(cluster_shape, dtype=torch.int32, device=device)
                    commit_shape = spill_shape
                    self.generated_page_commit_cluster_ids_cpu[ldx] = torch.full(commit_shape, -1, dtype=torch.int32, pin_memory=True)
                    self.generated_page_commit_cluster_heads_cpu[ldx] = torch.full(commit_shape, -1, dtype=torch.int32, pin_memory=True)
                    self.generated_page_commit_cluster_tails_cpu[ldx] = torch.full(commit_shape, -1, dtype=torch.int32, pin_memory=True)
                    self.generated_page_commit_page_ids_cpu[ldx] = torch.full(commit_shape, -1, dtype=torch.int32, pin_memory=True)
                    self.generated_page_commit_page_prev_cpu[ldx] = torch.full(commit_shape, -1, dtype=torch.int32, pin_memory=True)
                    self.generated_page_commit_page_sizes_cpu[ldx] = torch.zeros(commit_shape, dtype=torch.int32, pin_memory=True)
                    self.generated_page_commit_cluster_sums_cpu[ldx] = torch.zeros(spill_kv_shape, dtype=torch.float32, pin_memory=True)
                    self.generated_page_commit_cluster_counts_cpu[ldx] = torch.zeros(commit_shape, dtype=torch.int32, pin_memory=True)
                    self.generated_page_commit_cluster_ids_gpu[ldx] = torch.empty(commit_shape, dtype=torch.int32, device=device)
                    self.generated_page_commit_cluster_tails_gpu[ldx] = torch.empty(commit_shape, dtype=torch.int32, device=device)
                    self.generated_page_commit_page_ids_gpu[ldx] = torch.empty(commit_shape, dtype=torch.int32, device=device)
                    self.generated_page_commit_page_prev_gpu[ldx] = torch.empty(commit_shape, dtype=torch.int32, device=device)
                    self.generated_page_commit_page_sizes_gpu[ldx] = torch.empty(commit_shape, dtype=torch.int32, device=device)
                    self.generated_page_commit_cluster_sums_gpu[ldx] = torch.empty(spill_kv_shape, dtype=torch.float32, device=device)
                    self.generated_page_commit_cluster_counts_gpu[ldx] = torch.empty(commit_shape, dtype=torch.int32, device=device)

        if profile_prepare:
            log_step("async_update")

        ring_buffer_shape = (self.gpu_cache_aware_ring_size, self.batch_groups, self.buffer_size)
        ring_count_shape = (self.gpu_cache_aware_ring_size, self.batch_groups)
        ring_probe_shape = (self.gpu_cache_aware_ring_size, self.batch_groups, self.nprobe)
        count_shape = (self.batch_groups,)
        self.gpu_cache_hash_size = next_power_of_two(max(self.cache_size * 4, 16))
        hash_shape = (self.batch_groups, self.gpu_cache_hash_size)
        cache_slot_shape = (self.batch_groups, self.cache_size)
        cluster_shape = (self.batch_groups, self.n_centroids)
        queue_shape = (self.batch_groups,)
        for ldx in range(self.layer_num):
            device = self.layer_mapping[str(ldx)]
            ensure_list_tensor(self.gpu_cache_aware_hit_unit_idices, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_hit_unit_sizes, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_hit_unit_sizes_cumsum, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_hit_num_units, ldx, ring_count_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_miss_unit_idices, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_miss_unit_sizes, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_miss_unit_sizes_cumsum, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_miss_cluster_unit_sizes_cumsum, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_miss_num_units, ldx, ring_count_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_miss_copy_unit_idices, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_miss_copy_unit_sizes, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_miss_copy_unit_sizes_cumsum, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_miss_copy_num_units, ldx, ring_count_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_cluster_copy_unit_idices, ldx, ring_probe_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_cluster_copy_unit_sizes, ldx, ring_probe_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_cluster_copy_unit_sizes_cumsum, ldx, ring_probe_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_cluster_copy_num_units, ldx, ring_count_shape, torch.int32, device=device, zero_new=False)
            ensure_list_tensor(self.gpu_cache_aware_miss_page_keys, ldx, ring_buffer_shape, torch.int64, device=device, zero_new=False)
            admit_starts = ensure_list_tensor(self.gpu_cache_aware_admit_unit_starts, ldx, ring_probe_shape, torch.int32, device=device, zero_new=False)
            promote_src = ensure_list_tensor(self.gpu_cache_promote_src_offsets, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            promote_dst = ensure_list_tensor(self.gpu_cache_promote_dst_slots, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            promote_sizes = ensure_list_tensor(self.gpu_cache_promote_sizes, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            promote_hash = ensure_list_tensor(self.gpu_cache_promote_hash_positions, ldx, ring_buffer_shape, torch.int32, device=device, zero_new=False)
            promote_keys = ensure_list_tensor(self.gpu_cache_promote_page_keys, ldx, ring_buffer_shape, torch.int64, device=device, zero_new=False)
            promote_counts = ensure_list_tensor(self.gpu_cache_promote_counts, ldx, ring_count_shape, torch.int32, device=device)
            hash_keys = ensure_list_tensor(self.gpu_cache_hash_keys, ldx, hash_shape, torch.int64, device=device, zero_new=False)
            hash_values = ensure_list_tensor(self.gpu_cache_hash_values, ldx, hash_shape, torch.int32, device=device, zero_new=False)
            slot_keys = ensure_list_tensor(self.gpu_cache_slot_keys, ldx, cache_slot_shape, torch.int64, device=device, zero_new=False)
            cluster_state = ensure_list_tensor(self.gpu_cache_cluster_state, ldx, cluster_shape, torch.int8, device=device)
            cluster_prev = ensure_list_tensor(self.gpu_cache_cluster_prev, ldx, cluster_shape, torch.int32, device=device, zero_new=False)
            cluster_next = ensure_list_tensor(self.gpu_cache_cluster_next, ldx, cluster_shape, torch.int32, device=device, zero_new=False)
            hot_head = ensure_list_tensor(self.gpu_cache_hot_head, ldx, queue_shape, torch.int32, device=device, zero_new=False)
            hot_tail = ensure_list_tensor(self.gpu_cache_hot_tail, ldx, queue_shape, torch.int32, device=device, zero_new=False)
            cold_head = ensure_list_tensor(self.gpu_cache_cold_head, ldx, queue_shape, torch.int32, device=device, zero_new=False)
            cold_tail = ensure_list_tensor(self.gpu_cache_cold_tail, ldx, queue_shape, torch.int32, device=device, zero_new=False)
            hot_pages = ensure_list_tensor(self.gpu_cache_hot_pages, ldx, queue_shape, torch.int32, device=device)
            cold_pages = ensure_list_tensor(self.gpu_cache_cold_pages, ldx, queue_shape, torch.int32, device=device)
            free_slots = ensure_list_tensor(self.gpu_cache_free_slots, ldx, cache_slot_shape, torch.int32, device=device, zero_new=False)
            free_counts = ensure_list_tensor(self.gpu_cache_free_counts, ldx, count_shape, torch.int32, device=device)
            hash_keys.fill_(-1)
            hash_values.fill_(-1)
            admit_starts.fill_(-1)
            promote_src.fill_(0)
            promote_dst.fill_(-1)
            promote_sizes.fill_(0)
            promote_hash.fill_(-1)
            promote_keys.fill_(-1)
            promote_counts.zero_()
            slot_keys.fill_(-1)
            cluster_state.zero_()
            cluster_prev.fill_(-1)
            cluster_next.fill_(-1)
            hot_head.fill_(-1)
            hot_tail.fill_(-1)
            cold_head.fill_(-1)
            cold_tail.fill_(-1)
            hot_pages.zero_()
            cold_pages.zero_()
            free_counts.fill_(int(self.cache_size))
            free_slots.copy_(
                torch.arange(int(self.cache_size), dtype=torch.int32, device=device).view(1, -1).expand(self.batch_groups, -1)
            )

        if profile_prepare:
            log_step("gpu_cache_plan_buffers")

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
            if self.cluster_offsets_gpu[ldx] is None:
                self.cluster_offsets_gpu[ldx] = cluster_offsets.to(device)
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

            if self.layer_ready[ldx] and cluster_sizes is not None and self.centroids_mask[ldx] is not None:
                nonzero_clusters = int((cluster_sizes > 0).sum().item())
                unmasked_clusters = int((~self.centroids_mask[ldx]).sum().item())
                if nonzero_clusters == 0 and unmasked_clusters == 0:
                    raise RuntimeError(
                        f"AdaptiveIMI invalid decode metadata on layer {ldx}: "
                        "layer is ready but decode metadata remains empty."
                    )

        if profile_prepare:
            log_step("centroids")

        for ldx in range(self.layer_num):
            device = self.layer_mapping[str(ldx)]
            cache_shape = (self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim)
            ensure_list_tensor(self.cache_keys, ldx, cache_shape, self.dtype, device=device)
            ensure_list_tensor(self.cache_values, ldx, cache_shape, self.dtype, device=device)

        if profile_prepare:
            log_step("cache_kv")

        for ldx in range(self.layer_num):
            self._ensure_execution_buffer_ring(self.layer_mapping[str(ldx)], ldx)
        for device_idx in self.device_list:
            valid_lengths = ensure_dict_tensor(
                self.valid_lengths_dict,
                device_idx,
                (self.batch_groups,),
                torch.int32,
                device=device_idx,
            )
            valid_lengths.zero_()
            static_len = self.static_len_tensor_dict.get(device_idx)
            if not tensor_matches(static_len, (), torch.int32, device=device_idx):
                static_len = torch.tensor(self.static_pattern_total, dtype=torch.int32, device=device_idx)
                self.static_len_tensor_dict[device_idx] = static_len
            else:
                static_len.fill_(self.static_pattern_total)
            static_lengths = ensure_dict_tensor(
                self.static_lengths_dict,
                device_idx,
                (self.batch_groups,),
                torch.int32,
                device=device_idx,
            )
            static_lengths.zero_()
            nprobe_tensor = self.nprobe_tensor_dict.get(device_idx)
            if not tensor_matches(nprobe_tensor, (), torch.int32, device=device_idx):
                nprobe_tensor = torch.tensor(int(self.nprobe), dtype=torch.int32, device=device_idx)
                self.nprobe_tensor_dict[device_idx] = nprobe_tensor
            else:
                nprobe_tensor.fill_(int(self.nprobe))
            ensure_dict_tensor(
                self.similarity_buffer_dict,
                device_idx,
                (self.batch_groups, self.n_centroids),
                torch.float32,
                device=device_idx,
            )
            ensure_dict_tensor(
                self.selected_cluster_buffer_dict,
                device_idx,
                (self.batch_groups, self.nprobe),
                torch.int64,
                device=device_idx,
                zero_new=False,
            )

        if profile_prepare:
            log_step("execution_buffers")

        for device in self.device_list:
            if device not in self.gpu_cache_promote_streams:
                self.gpu_cache_promote_streams[device] = torch.cuda.Stream(device)

        self.attn_func = self.sparse_attention
        self.cache_prepared = True

        if profile_prepare:
            total_ms = (time.perf_counter() - start_ts) * 1000.0
            print(f"[IMI prepare_cache] total={total_ms:.2f} ms", flush=True)
    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        if self.async_update_enabled:
            self._schedule_async_update(layer_idx, key_states, value_states)
        else:
            self.steady_zone_keys[layer_idx][:, :, self.static_pattern_total, :] = key_states[:, 0, :, :]
            self.steady_zone_values[layer_idx][:, :, self.static_pattern_total, :] = value_states[:, 0, :, :]

        if layer_idx == 0:
            self.decode_debug_step += 1

        if layer_idx == self.layer_num - 1:
            self.context += 1
            if self.async_update_enabled:
                self.generated_tokens_total += 1
                self.static_pattern_total = self.current_static_len(layer_idx)
            else:
                self.static_pattern_total += 1

        return None, None
    def dense_attention(self, queries, layer_idx, static_len):
        attn_out = compact_decode_attention(
            queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
            self.steady_zone_keys[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
            self.steady_zone_values[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
            static_len,
        )
        return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)
