from typing import Optional

import os
import torch

from library.AdaptiveIMI.imi_adapter import get_imi_kernels
from library.AdaptiveIMI.cpp_extensions import (
    build_cache_aware_retrieval_plan,
    gather_copy_hit_chunks_profile,
    gather_copy_miss_chunks_profile,
    gather_copy_selected_clusters_profile,
    gather_copy_static_chunks_profile,
    promote_hot_cold_cache_pages_from_execution,
)
from .attention_backend import compact_decode_attention

imi_gpu_kernels = get_imi_kernels()


class AdaptiveIMIRetrievalMixin:
    _CLUSTER_UVA_WARMUP_STEPS = 3
    _GPU_CACHE_PROMOTION_WARMUP_STEPS = 4
    _GPU_CACHE_PROMOTION_STEADY_PAGES_PER_GROUP = 64

    _CACHE_PROFILE_SUM_KEYS = (
        "hit_blocks",
        "miss_blocks",
        "hit_tokens",
        "miss_tokens",
        "update_blocks",
        "miss_pages",
        "miss_copy_runs",
        "miss_runs_1p",
        "miss_runs_2_4p",
        "miss_runs_5_8p",
        "miss_runs_gt8p",
        "promotion_inflight_at_retrieval",
        "evicted_clusters",
        "evicted_blocks",
        "free_blocks_before",
        "free_blocks_after",
        "promotion_submitted",
        "promotion_completed",
        "promotion_ms",
        "promotion_skipped_busy",
        "promotion_skipped_no_buffer",
    )
    _CACHE_PROFILE_MAX_KEYS = (
        "max_group_update_blocks",
        "max_group_evicted_blocks",
        "max_miss_run_pages",
        "promotion_ms_max",
    )

    def _profile_miss_run_histogram(self, copy_sizes: torch.Tensor, copy_counts: torch.Tensor) -> dict:
        max_count = int(copy_counts.max().item()) if copy_counts.numel() else 0
        if max_count <= 0:
            return {
                "miss_runs_1p": 0.0,
                "miss_runs_2_4p": 0.0,
                "miss_runs_5_8p": 0.0,
                "miss_runs_gt8p": 0.0,
                "max_miss_run_pages": 0.0,
            }

        sizes = copy_sizes[:, :max_count]
        counts = copy_counts.view(-1, 1)
        cols = torch.arange(max_count, device=copy_sizes.device, dtype=copy_counts.dtype).view(1, -1)
        run_sizes = sizes.masked_select(cols < counts)
        if run_sizes.numel() == 0:
            return {
                "miss_runs_1p": 0.0,
                "miss_runs_2_4p": 0.0,
                "miss_runs_5_8p": 0.0,
                "miss_runs_gt8p": 0.0,
                "max_miss_run_pages": 0.0,
            }

        run_pages = torch.div(run_sizes + int(self.page_size) - 1, int(self.page_size), rounding_mode="floor")
        return {
            "miss_runs_1p": float((run_pages == 1).sum().item()),
            "miss_runs_2_4p": float(((run_pages >= 2) & (run_pages <= 4)).sum().item()),
            "miss_runs_5_8p": float(((run_pages >= 5) & (run_pages <= 8)).sum().item()),
            "miss_runs_gt8p": float((run_pages > 8).sum().item()),
            "max_miss_run_pages": float(run_pages.max().item()),
        }

    @staticmethod
    def _sum_descriptor_units(sizes: torch.Tensor, counts: torch.Tensor) -> float:
        if sizes.numel() == 0 or counts.numel() == 0:
            return 0.0
        max_count = int(counts.max().item())
        if max_count <= 0:
            return 0.0
        bounded_sizes = sizes[:, :max_count]
        columns = torch.arange(
            max_count,
            device=sizes.device,
            dtype=counts.dtype,
        ).view(1, -1)
        mask = columns < counts.view(-1, 1)
        return float(bounded_sizes.masked_select(mask).sum().item())

    def _cluster_uva_warmup_steps(self) -> int:
        cache_cfg = self.runtime_config.get("cache") or {}
        return max(int(cache_cfg.get("cluster_uva_warmup_steps", self._CLUSTER_UVA_WARMUP_STEPS)), 0)

    def _use_cluster_uva_for_decode_step(self) -> bool:
        warmup_steps = self._cluster_uva_warmup_steps()
        if warmup_steps <= 0:
            return False
        step = int(getattr(self, "gpu_cache_promotion_decode_step", 0))
        return 1 <= step <= warmup_steps

    def _promotion_warmup_steps(self) -> int:
        cache_cfg = self.runtime_config.get("cache") or {}
        return max(int(cache_cfg.get("promotion_warmup_steps", self._GPU_CACHE_PROMOTION_WARMUP_STEPS)), 0)

    def _select_clusters_raw_sum(self, layer_idx: int, query_grouped: torch.Tensor, device: str) -> torch.Tensor:
        similarities = imi_gpu_kernels.fused_query_group_similarities(
            query_grouped,
            self.centroids[layer_idx],
            self.similarity_buffer_dict[device],
        )
        similarities.masked_fill_(self.centroids_mask[layer_idx], float("-inf"))
        return self._select_clusters(layer_idx, similarities)
    def _select_gpu_cache_aware_descriptor_slot(self, layer_idx: int) -> int:
        ring_size = max(int(getattr(self, "gpu_cache_aware_ring_size", 2)), 2)
        active_slots = getattr(self, "gpu_cache_aware_active_slots", None)
        promote_slots = getattr(self, "gpu_cache_promote_slots", None)
        promote_events = getattr(self, "gpu_cache_promote_events", None)
        if active_slots is None or len(active_slots) <= layer_idx:
            return 0

        candidate = (int(active_slots[layer_idx]) + 1) % ring_size
        busy_slot = -1
        if promote_events is not None and len(promote_events) > layer_idx:
            event = promote_events[layer_idx]
            if event is not None and not event.query():
                if promote_slots is not None and len(promote_slots) > layer_idx:
                    busy_slot = int(promote_slots[layer_idx])
        if candidate == busy_slot:
            candidate = (candidate + 1) % ring_size
        active_slots[layer_idx] = candidate
        return candidate
    def _gpu_cache_aware_descriptors(self, layer_idx: int, slot: int):
        return {
            "miss_unit_idices": self.gpu_cache_aware_miss_unit_idices[layer_idx][slot],
            "miss_unit_sizes": self.gpu_cache_aware_miss_unit_sizes[layer_idx][slot],
            "miss_unit_sizes_cumsum": self.gpu_cache_aware_miss_unit_sizes_cumsum[layer_idx][slot],
            "miss_cluster_unit_sizes_cumsum": self.gpu_cache_aware_miss_cluster_unit_sizes_cumsum[layer_idx][slot],
            "miss_num_units": self.gpu_cache_aware_miss_num_units[layer_idx][slot],
            "miss_copy_unit_idices": self.gpu_cache_aware_miss_copy_unit_idices[layer_idx][slot],
            "miss_copy_unit_sizes": self.gpu_cache_aware_miss_copy_unit_sizes[layer_idx][slot],
            "miss_copy_unit_sizes_cumsum": self.gpu_cache_aware_miss_copy_unit_sizes_cumsum[layer_idx][slot],
            "miss_copy_num_units": self.gpu_cache_aware_miss_copy_num_units[layer_idx][slot],
            "cluster_copy_unit_idices": self.gpu_cache_aware_cluster_copy_unit_idices[layer_idx][slot],
            "cluster_copy_unit_sizes": self.gpu_cache_aware_cluster_copy_unit_sizes[layer_idx][slot],
            "cluster_copy_unit_sizes_cumsum": self.gpu_cache_aware_cluster_copy_unit_sizes_cumsum[layer_idx][slot],
            "cluster_copy_num_units": self.gpu_cache_aware_cluster_copy_num_units[layer_idx][slot],
            "miss_page_keys": self.gpu_cache_aware_miss_page_keys[layer_idx][slot],
            "admit_unit_starts": self.gpu_cache_aware_admit_unit_starts[layer_idx][slot],
            "promote_src_offsets": self.gpu_cache_promote_src_offsets[layer_idx][slot],
            "promote_dst_slots": self.gpu_cache_promote_dst_slots[layer_idx][slot],
            "promote_sizes": self.gpu_cache_promote_sizes[layer_idx][slot],
            "promote_hash_positions": self.gpu_cache_promote_hash_positions[layer_idx][slot],
            "promote_page_keys": self.gpu_cache_promote_page_keys[layer_idx][slot],
            "promote_counts": self.gpu_cache_promote_counts[layer_idx][slot],
            "hit_unit_idices": self.gpu_cache_aware_hit_unit_idices[layer_idx][slot],
            "hit_unit_sizes": self.gpu_cache_aware_hit_unit_sizes[layer_idx][slot],
            "hit_unit_sizes_cumsum": self.gpu_cache_aware_hit_unit_sizes_cumsum[layer_idx][slot],
            "hit_num_units": self.gpu_cache_aware_hit_num_units[layer_idx][slot],
        }
    def _gpu_cache_promotion_pages_per_group(self) -> int:
        cache_cfg = self.runtime_config.get("cache") or {}
        if cache_cfg.get("promotion_pages_per_group") is not None:
            return int(cache_cfg["promotion_pages_per_group"])

        hot_ratio = float(cache_cfg.get("hot_ratio", 0.3))
        hot_ratio = min(max(hot_ratio, 0.05), 0.95)
        cold_capacity = max(int(self.cache_size) - int(round(int(self.cache_size) * hot_ratio)), 1)
        step = int(getattr(self, "gpu_cache_promotion_decode_step", 0))
        if 1 <= step <= self._promotion_warmup_steps():
            ratio = float(cache_cfg.get("promotion_initial_fill_ratio", 0.25))
            return max(int(cold_capacity * ratio), self._GPU_CACHE_PROMOTION_STEADY_PAGES_PER_GROUP)
        ratio = float(cache_cfg.get("promotion_steady_fill_ratio", 0.10))
        return max(int(cold_capacity * ratio), self._GPU_CACHE_PROMOTION_STEADY_PAGES_PER_GROUP)
    def _gpu_cache_hot_cold_capacities(self) -> tuple[int, int]:
        cache_cfg = self.runtime_config.get("cache") or {}
        hot_ratio = float(cache_cfg.get("hot_ratio", 0.3))
        hot_ratio = min(max(hot_ratio, 0.05), 0.95)
        hot_capacity = min(max(int(round(int(self.cache_size) * hot_ratio)), 1), max(int(self.cache_size) - 1, 1))
        cold_capacity = max(int(self.cache_size) - hot_capacity, 1)
        return hot_capacity, cold_capacity

    def _tokens_to_pages(self, token_budget: int) -> int:
        if token_budget <= 0:
            return 0
        return max((int(token_budget) + self.page_size - 1) // self.page_size, 1)
    def _resolve_budget_pages(self, requested_pages: int, minimum_pages: int = 1) -> int:
        return max(int(requested_pages), minimum_pages)
    def _empty_cache_profile_stats(self) -> dict:
        stats = {key: 0.0 for key in self._CACHE_PROFILE_SUM_KEYS}
        stats.update({key: 0.0 for key in self._CACHE_PROFILE_MAX_KEYS})
        return stats
    def _merge_cache_profile_stats(self, total: dict, stats: Optional[dict]) -> None:
        if not stats:
            return
        for key in self._CACHE_PROFILE_SUM_KEYS:
            total[key] = float(total.get(key, 0.0)) + float(stats.get(key, 0.0) or 0.0)
        for key in self._CACHE_PROFILE_MAX_KEYS:
            total[key] = max(float(total.get(key, 0.0)), float(stats.get(key, 0.0) or 0.0))
    def _new_timing_event(self) -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)
    def _record_timing_event(self) -> torch.cuda.Event:
        event = self._new_timing_event()
        event.record()
        return event
    def _elapsed_ms(self, start: torch.cuda.Event, end: torch.cuda.Event) -> float:
        return float(start.elapsed_time(end))
    def _execution_buffer_ring_size(self) -> int:
        cache_cfg = self.runtime_config.get("cache") or {}
        raw_size = cache_cfg.get("execution_buffer_ring_size", 2)
        ring_size = int(raw_size)
        return min(max(ring_size, 1), 8)

    def _execution_buffer_ring_key(self, device: str, layer_idx: int) -> str:
        return f"{device}:layer:{int(layer_idx)}"

    def _ensure_execution_buffer_ring(self, device: str, layer_idx: int = 0) -> None:
        shape = (self.batch_groups, self.execution_stride, 1, self.head_dim)
        dev = torch.device(device)
        ring_size = self._execution_buffer_ring_size()
        ring_key = self._execution_buffer_ring_key(device, layer_idx)

        def matches(tensor: torch.Tensor) -> bool:
            return (
                tensor is not None
                and tuple(tensor.shape) == shape
                and tensor.dtype == self.dtype
                and torch.device(tensor.device) == dev
            )

        key_ring = self.execution_buffer_key_ring_dict.get(ring_key)
        value_ring = self.execution_buffer_value_ring_dict.get(ring_key)
        if (
            key_ring is None
            or value_ring is None
            or len(key_ring) != ring_size
            or len(value_ring) != ring_size
            or any(not matches(tensor) for tensor in key_ring)
            or any(not matches(tensor) for tensor in value_ring)
        ):
            self.execution_buffer_key_ring_dict[ring_key] = [
                torch.empty(shape, dtype=self.dtype, device=dev).contiguous()
                for _ in range(ring_size)
            ]
            self.execution_buffer_value_ring_dict[ring_key] = [
                torch.empty(shape, dtype=self.dtype, device=dev).contiguous()
                for _ in range(ring_size)
            ]
            self.execution_buffer_ring_events[ring_key] = [None] * ring_size
            self.execution_buffer_ring_next[ring_key] = 0

        keys = self.execution_buffer_fallback_keys_dict.get(device)
        if not matches(keys):
            self.execution_buffer_fallback_keys_dict[device] = torch.empty(shape, dtype=self.dtype, device=dev).contiguous()
        values = self.execution_buffer_fallback_values_dict.get(device)
        if not matches(values):
            self.execution_buffer_fallback_values_dict[device] = torch.empty(shape, dtype=self.dtype, device=dev).contiguous()

        self.execution_buffer_keys_dict[device] = self.execution_buffer_key_ring_dict[ring_key][0]
        self.execution_buffer_values_dict[device] = self.execution_buffer_value_ring_dict[ring_key][0]

    def _select_execution_buffer(self, layer_idx: int, device: str) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
        self._ensure_execution_buffer_ring(device, layer_idx)
        ring_key = self._execution_buffer_ring_key(device, layer_idx)
        key_ring = self.execution_buffer_key_ring_dict[ring_key]
        value_ring = self.execution_buffer_value_ring_dict[ring_key]
        events = self.execution_buffer_ring_events[ring_key]
        ring_size = len(key_ring)
        for idx, event in enumerate(events):
            if event is not None and event.query():
                events[idx] = None
        start = int(self.execution_buffer_ring_next.get(ring_key, 0)) % ring_size
        for offset in range(ring_size):
            slot = (start + offset) % ring_size
            event = events[slot]
            if event is None:
                self.execution_buffer_ring_next[ring_key] = (slot + 1) % ring_size
                keys = key_ring[slot]
                values = value_ring[slot]
                self.execution_buffer_keys_dict[device] = keys
                self.execution_buffer_values_dict[device] = values
                return keys, values, slot, True

        keys = self.execution_buffer_fallback_keys_dict[device]
        values = self.execution_buffer_fallback_values_dict[device]
        self.execution_buffer_keys_dict[device] = keys
        self.execution_buffer_values_dict[device] = values
        return keys, values, -1, False

    def _copy_static_chunks(
        self,
        layer_idx: int,
        device: str,
        execution_keys: torch.Tensor,
        execution_values: torch.Tensor,
    ) -> None:
        gather_copy_static_chunks_profile(
            self.steady_zone_keys[layer_idx],
            execution_keys,
            self.steady_zone_values[layer_idx],
            execution_values,
            self.valid_lengths_dict[device],
            self.batch_groups,
            self.static_stride,
            self.execution_stride,
            self.static_len_tensor_dict[device],
        )

    def _copy_foreground_chunks(
        self,
        layer_idx: int,
        device: str,
        execution_keys: torch.Tensor,
        execution_values: torch.Tensor,
        desc: dict,
        use_cluster_uva: bool,
    ) -> None:
        if use_cluster_uva:
            gather_copy_selected_clusters_profile(
                self.list_keys[layer_idx],
                execution_keys,
                self.list_values[layer_idx],
                execution_values,
                desc["cluster_copy_unit_idices"],
                desc["cluster_copy_unit_sizes"],
                desc["cluster_copy_unit_sizes_cumsum"],
                desc["cluster_copy_num_units"],
                self.batch_groups,
                self.list_capacity,
                self.execution_stride,
                self.nprobe,
                self.static_len_tensor_dict[device],
            )
            return

        gather_copy_miss_chunks_profile(
            self.list_keys[layer_idx],
            execution_keys,
            self.list_values[layer_idx],
            execution_values,
            desc["miss_copy_unit_idices"],
            desc["miss_copy_unit_sizes"],
            desc["miss_copy_unit_sizes_cumsum"],
            desc["miss_copy_num_units"],
            self.batch_groups,
            self.list_capacity,
            self.execution_stride,
            self.buffer_size,
            self.static_len_tensor_dict[device],
        )

    def _copy_hit_chunks(
        self,
        layer_idx: int,
        device: str,
        execution_keys: torch.Tensor,
        execution_values: torch.Tensor,
        desc: dict,
    ) -> None:
        gather_copy_hit_chunks_profile(
            self.cache_keys[layer_idx],
            execution_keys,
            self.cache_values[layer_idx],
            execution_values,
            desc["miss_unit_sizes"],
            desc["miss_unit_sizes_cumsum"],
            desc["miss_num_units"],
            desc["hit_unit_idices"],
            desc["hit_unit_sizes"],
            desc["hit_unit_sizes_cumsum"],
            desc["hit_num_units"],
            self.batch_groups,
            self.cache_size,
            self.execution_stride,
            self.buffer_size,
            self.static_len_tensor_dict[device],
        )

    def _run_retrieval_full_gpu_cache_aware(
        self,
        layer_idx: int,
        device: str,
        cluster_ids: torch.Tensor,
        execution_keys: torch.Tensor,
        execution_values: torch.Tensor,
        use_cluster_uva: bool,
        profile_decode: bool = False,
    ) -> Optional[dict]:
        descriptor_slot = self._select_gpu_cache_aware_descriptor_slot(layer_idx)
        desc = self._gpu_cache_aware_descriptors(layer_idx, descriptor_slot)
        promotion_inflight_at_retrieval = 0.0
        if profile_decode:
            prior_event = self.gpu_cache_promote_events[layer_idx]
            if prior_event is not None and not prior_event.query():
                promotion_inflight_at_retrieval = 1.0

        if profile_decode:
            lookup_start_event = self._record_timing_event()

        build_cache_aware_retrieval_plan(
            cluster_ids,
            self.cluster_offsets_gpu[layer_idx],
            self.cluster_sizes_gpu[layer_idx],
            self.gpu_cache_hash_keys[layer_idx],
            self.gpu_cache_hash_values[layer_idx],
            desc["miss_unit_idices"],
            desc["miss_unit_sizes"],
            desc["miss_unit_sizes_cumsum"],
            desc["miss_cluster_unit_sizes_cumsum"],
            desc["miss_num_units"],
            desc["miss_copy_unit_idices"],
            desc["miss_copy_unit_sizes"],
            desc["miss_copy_unit_sizes_cumsum"],
            desc["miss_copy_num_units"],
            desc["cluster_copy_unit_idices"],
            desc["cluster_copy_unit_sizes"],
            desc["cluster_copy_unit_sizes_cumsum"],
            desc["cluster_copy_num_units"],
            desc["miss_page_keys"],
            desc["admit_unit_starts"],
            desc["hit_unit_idices"],
            desc["hit_unit_sizes"],
            desc["hit_unit_sizes_cumsum"],
            desc["hit_num_units"],
            self.valid_lengths_dict[device],
            int(self.batch_groups),
            int(self.nprobe),
            int(self.n_centroids),
            int(self.buffer_size),
            int(self.page_size),
            int(self.gpu_cache_hash_size),
            int(self.padding_cluster_id),
        )

        if profile_decode:
            lookup_end_event = self._record_timing_event()
            gather_start_event = self._record_timing_event()

        if profile_decode:
            static_start_event = self._record_timing_event()
            self._copy_static_chunks(layer_idx, device, execution_keys, execution_values)
            static_end_event = self._record_timing_event()
            self._copy_foreground_chunks(layer_idx, device, execution_keys, execution_values, desc, use_cluster_uva)
            miss_end_event = self._record_timing_event()
            if not use_cluster_uva:
                self._copy_hit_chunks(layer_idx, device, execution_keys, execution_values, desc)
                hit_end_event = self._record_timing_event()
            else:
                hit_end_event = miss_end_event
        else:
            self._copy_static_chunks(layer_idx, device, execution_keys, execution_values)
            self._copy_foreground_chunks(layer_idx, device, execution_keys, execution_values, desc, use_cluster_uva)
            if not use_cluster_uva:
                self._copy_hit_chunks(layer_idx, device, execution_keys, execution_values, desc)

        if not profile_decode:
            return None

        gather_end_event = hit_end_event
        promote_pages_per_group = self._gpu_cache_promotion_pages_per_group()
        miss_pages = desc["miss_num_units"].sum()
        if promote_pages_per_group < 0:
            planned_update_blocks = miss_pages
        else:
            planned_update_blocks = torch.clamp(desc["miss_num_units"], max=promote_pages_per_group).sum()
        gpu_cache_stats = {
            "hit_blocks": desc["hit_num_units"].sum(),
            "miss_blocks": miss_pages,
            "update_blocks": planned_update_blocks,
            "miss_pages": miss_pages,
            "miss_copy_runs": desc["miss_copy_num_units"].sum(),
            "promotion_inflight_at_retrieval": promotion_inflight_at_retrieval,
        }
        retrieval_result = {
            "lookup_events": (lookup_start_event, lookup_end_event),
            "gather_events": (gather_start_event, gather_end_event),
            "copy_events": {
                "static": (static_start_event, static_end_event),
                "miss": (static_end_event, miss_end_event),
                "hit": (miss_end_event, hit_end_event),
            },
            "h2d_ms": 0.0,
            "cache_sync_ms": 0.0,
            "scatter_ms": 0.0,
            "cache_stats": gpu_cache_stats,
            "diagnostic_tensors": {
                "miss_copy_sizes": desc["miss_copy_unit_sizes"],
                "miss_copy_counts": desc["miss_copy_num_units"],
            },
        }
        if getattr(self, "profile_token_hit_rate", False):
            retrieval_result["diagnostic_tensors"].update(
                {
                    "hit_unit_sizes": desc["hit_unit_sizes"],
                    "hit_num_units": desc["hit_num_units"],
                    "miss_unit_sizes": desc["miss_unit_sizes"],
                    "miss_num_units": desc["miss_num_units"],
                }
            )
        return retrieval_result
    def _schedule_gpu_cache_promotion(
        self,
        layer_idx: int,
        device: str,
        execution_keys: torch.Tensor,
        execution_values: torch.Tensor,
        execution_slot: int,
        can_admit: bool,
        static_len: int,
        use_cluster_uva: bool,
    ) -> dict:
        descriptor_slot = int(self.gpu_cache_aware_active_slots[layer_idx])
        desc = self._gpu_cache_aware_descriptors(layer_idx, descriptor_slot)
        stats = {
            "promotion_submitted": 0.0,
            "promotion_completed": 0.0,
            "promotion_ms": 0.0,
            "promotion_ms_max": 0.0,
            "promotion_skipped_busy": 0.0,
            "promotion_skipped_no_buffer": 0.0,
        }

        prior_event = self.gpu_cache_promote_events[layer_idx]
        if prior_event is not None:
            if not prior_event.query():
                stats["promotion_skipped_busy"] = 1.0
                return stats
            start_event = self.gpu_cache_promote_start_events[layer_idx]
            if start_event is not None:
                try:
                    elapsed_ms = float(start_event.elapsed_time(prior_event))
                    stats["promotion_completed"] = 1.0
                    stats["promotion_ms"] = elapsed_ms
                    stats["promotion_ms_max"] = elapsed_ms
                except RuntimeError:
                    # Promotion timing is diagnostic only. Do not let a stale or
                    # non-timing event from a previous run break decode.
                    stats["promotion_completed"] = 1.0
            self.gpu_cache_promote_events[layer_idx] = None
            self.gpu_cache_promote_start_events[layer_idx] = None
            self.gpu_cache_promote_slots[layer_idx] = -1

        stream = self.gpu_cache_promote_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device)
            self.gpu_cache_promote_streams[device] = stream

        current_stream = torch.cuda.current_stream(device)
        ready_event = torch.cuda.Event()
        start_event = torch.cuda.Event(enable_timing=True)
        done_event = torch.cuda.Event(enable_timing=True)
        promote_pages_per_group = int(self._gpu_cache_promotion_pages_per_group()) if can_admit else 0
        if not can_admit:
            stats["promotion_skipped_no_buffer"] = 1.0
        miss_dst_offsets = desc["miss_cluster_unit_sizes_cumsum"] if use_cluster_uva else desc["miss_unit_sizes_cumsum"]
        ready_event.record(current_stream)
        with torch.cuda.stream(stream):
            stream.wait_event(ready_event)
            start_event.record(stream)
            hot_capacity, cold_capacity = self._gpu_cache_hot_cold_capacities()
            # Admission reads the already materialized execution buffer. This avoids
            # a second CPU pinned -> GPU transfer for the same miss pages.
            promote_hot_cold_cache_pages_from_execution(
                execution_keys,
                self.cache_keys[layer_idx],
                execution_values,
                self.cache_values[layer_idx],
                miss_dst_offsets,
                desc["miss_unit_sizes"],
                desc["miss_num_units"],
                desc["miss_page_keys"],
                desc["admit_unit_starts"],
                desc["promote_src_offsets"],
                desc["promote_dst_slots"],
                desc["promote_sizes"],
                desc["promote_hash_positions"],
                desc["promote_page_keys"],
                desc["promote_counts"],
                desc["hit_unit_idices"],
                desc["hit_num_units"],
                self.cluster_sizes_gpu[layer_idx],
                self.gpu_cache_hash_keys[layer_idx],
                self.gpu_cache_hash_values[layer_idx],
                self.gpu_cache_slot_keys[layer_idx],
                self.gpu_cache_cluster_state[layer_idx],
                self.gpu_cache_cluster_prev[layer_idx],
                self.gpu_cache_cluster_next[layer_idx],
                self.gpu_cache_hot_head[layer_idx],
                self.gpu_cache_hot_tail[layer_idx],
                self.gpu_cache_cold_head[layer_idx],
                self.gpu_cache_cold_tail[layer_idx],
                self.gpu_cache_hot_pages[layer_idx],
                self.gpu_cache_cold_pages[layer_idx],
                self.gpu_cache_free_slots[layer_idx],
                self.gpu_cache_free_counts[layer_idx],
                int(static_len),
                int(self.batch_groups),
                int(self.execution_stride),
                int(self.cache_size),
                int(self.gpu_cache_hash_size),
                int(self.buffer_size),
                int(self.nprobe),
                int(self.n_centroids),
                int(self.page_size),
                int(hot_capacity),
                int(cold_capacity),
                promote_pages_per_group,
            )
            done_event.record(stream)
        self.gpu_cache_promote_start_events[layer_idx] = start_event
        self.gpu_cache_promote_events[layer_idx] = done_event
        self.gpu_cache_promote_slots[layer_idx] = descriptor_slot
        if execution_slot >= 0:
            ring_key = self._execution_buffer_ring_key(device, layer_idx)
            self.execution_buffer_ring_events[ring_key][execution_slot] = done_event
        stats["promotion_submitted"] = 1.0
        return stats
    def _select_clusters(self, layer_idx, similarities):
        active_clusters = int(similarities.size(-1))
        if active_clusters <= 0:
            return torch.full(
                (similarities.size(0), self.nprobe),
                self.padding_cluster_id,
                dtype=torch.int64,
                device=similarities.device,
            )
        effective_nprobe = min(int(self.nprobe), active_clusters)
        topk = torch.topk(similarities, effective_nprobe, dim=-1, largest=True, sorted=True)
        cluster_sizes = self.cluster_sizes_gpu[layer_idx]
        selected_buffer = self.selected_cluster_buffer_dict.get(str(similarities.device))
        selected = imi_gpu_kernels.apply_budget_after_topk(
            topk.indices.contiguous(),
            cluster_sizes,
            int(self.max_retrieval_tokens),
            int(self.padding_cluster_id),
            selected_buffer,
        )
        if effective_nprobe < int(self.nprobe):
            padded = torch.full(
                (selected.size(0), int(self.nprobe)),
                self.padding_cluster_id,
                dtype=selected.dtype,
                device=selected.device,
            )
            padded[:, :effective_nprobe].copy_(selected)
            selected = padded
        return selected
    def ensure_layer_ready(self, layer_idx):
        if self.layer_ready[layer_idx] or not self.build_index_when_prefilling:
            return
        self._raise_prefill_stream_error(layer_idx)
        if not self.layer_started[layer_idx]:
            raise RuntimeError(f"IMI layer {layer_idx} not started")

        if self.prefill_stream_started[layer_idx]:
            done_event = self.prefill_stream_done_events[layer_idx]
            done_event.wait()
            self._raise_prefill_stream_error(layer_idx)
            if self.layer_ready[layer_idx]:
                return
            raise RuntimeError(f"IMI streaming prefill did not finish for layer {layer_idx}")

        self._finish_layer_index(layer_idx)
    def sparse_attention(self, queries, layer_idx, static_len):
        self.ensure_layer_ready(layer_idx)
        device = self.layer_mapping[str(layer_idx)]
        if layer_idx == 0:
            self.gpu_cache_promotion_decode_step = int(getattr(self, "gpu_cache_promotion_decode_step", 0)) + 1
        self._apply_async_centroid_updates(layer_idx, device)
        self.static_len_tensor_dict[device].fill_(static_len)
        self.static_lengths_dict[device].fill_(static_len)

        debug_decode = os.getenv("IMI_DEBUG_DECODE", "0") == "1"
        if debug_decode and layer_idx == 0:
            print(
                "[IMI DECODE]"
                f" step={self.decode_debug_step} layer={layer_idx} static_len={static_len}"
                f" list_capacity={self.list_capacity} list_stride={self.list_stride} static_capacity={self.static_capacity} static_stride={self.static_stride}"
                f" nprobe={self.nprobe}",
                flush=True,
            )

        profile = self.profile_decode and self.profile_decode_count < self.profile_decode_steps
        if profile and layer_idx == 0:
            self.profile_decode_step_idx += 1
            self._profile_step_use_cluster_uva = self._use_cluster_uva_for_decode_step()
            self._profile_step_breakdown = {
                "search_ms": 0.0,
                "gather_ms": 0.0,
                "selection_d2h_ms": 0.0,
                "gpu_cache_plan_ms": 0.0,
                "concat_copy_ms": 0.0,
                "static_copy_ms": 0.0,
                "miss_copy_ms": 0.0,
                "hit_copy_ms": 0.0,
                "cache_sync_ms": 0.0,
                "scatter_ms": 0.0,
                "delta_append_ms": 0.0,
                "attn_ms": 0.0,
                "total_ms": 0.0,
            }
            self._profile_step_breakdown.update(self._empty_cache_profile_stats())
            self._profile_pending_events = []
        if profile:
            start_event = self._record_timing_event()

        query_grouped = queries.view(self.batch_groups, self.group_size, self.head_dim).contiguous()
        # Fixed flat decode path: score the query group against materialized
        # cluster centroids, then select clusters directly from that score table.
        selected = self._select_clusters_raw_sum(layer_idx, query_grouped, device)

        if profile:
            sim_event = self._record_timing_event()
        if debug_decode and layer_idx == 0:
            selected_count = int((selected != self.padding_cluster_id).sum().item())
            selected_sample = selected[0, : min(16, selected.shape[1])].tolist()
            print(
                "[IMI DECODE]"
                f" step={self.decode_debug_step} layer={layer_idx} selected_clusters={selected_count}"
                f" selected_sample={selected_sample}",
                flush=True,
            )
        lookup_ms = 0.0
        h2d_ms = 0.0
        gather_ms = 0.0
        concat_copy_ms = 0.0
        cache_sync_ms = 0.0
        scatter_ms = 0.0
        delta_append_ms = 0.0
        attn_ms = 0.0
        selection_transfer_ms = 0.0
        cache_profile_stats = self._empty_cache_profile_stats() if profile else None

        query_view = queries.view(self.batch_groups, 1, self.group_size, self.head_dim)

        self.valid_lengths_dict[device].zero_()
        execution_keys, execution_values, execution_slot, can_admit_cache = self._select_execution_buffer(layer_idx, device)
        use_cluster_uva = self._use_cluster_uva_for_decode_step()

        retrieval_timing = self._run_retrieval_full_gpu_cache_aware(
            layer_idx,
            device,
            selected,
            execution_keys,
            execution_values,
            use_cluster_uva=use_cluster_uva,
            profile_decode=profile,
        )
        if profile:
            delta_start_event = self._record_timing_event()
        self._append_delta_to_execution(
            layer_idx,
            device,
            query_grouped,
            execution_keys,
            execution_values,
        )
        if profile:
            delta_end_event = self._record_timing_event()

        if debug_decode and layer_idx == 0:
            valid_lengths = self.valid_lengths_dict[device]
            max_len = int(valid_lengths.max().item()) if valid_lengths.numel() else 0
            min_len = int(valid_lengths.min().item()) if valid_lengths.numel() else 0
            mean_len = float(valid_lengths.float().mean().item()) if valid_lengths.numel() else 0.0
            print(
                "[IMI DECODE]"
                f" step={self.decode_debug_step} layer={layer_idx} valid_lengths[min,max,mean]=[{min_len},{max_len},{mean_len:.1f}]",
                flush=True,
            )

        if retrieval_timing is not None:
            h2d_ms += retrieval_timing["h2d_ms"]
            cache_sync_ms += retrieval_timing.get("cache_sync_ms", retrieval_timing["h2d_ms"])
            scatter_ms += retrieval_timing.get("scatter_ms", 0.0)

        if profile:
            attn_start_event = self._record_timing_event()

        acc_out = compact_decode_attention(
            query_view,
            execution_keys,
            execution_values,
            self.valid_lengths_dict[device],
        )

        if profile:
            attn_end_event = self._record_timing_event()
            total_end_event = self._record_timing_event()

        promotion_stats = self._schedule_gpu_cache_promotion(
            layer_idx,
            device,
            execution_keys,
            execution_values,
            execution_slot,
            can_admit_cache,
            int(static_len),
            use_cluster_uva=use_cluster_uva,
        )
        if profile:
            self._merge_cache_profile_stats(cache_profile_stats, promotion_stats)
        if self.profile_block_hit_rate and layer_idx == self.layer_num - 1:
            self.profile_block_step += 1
            if self.profile_block_step % self.profile_block_hit_every == 0:
                total_blocks = self.profile_block_hits + self.profile_block_misses
                total_hit_rate = self.profile_block_hits / max(total_blocks, 1)
                print(
                    "[IMI hitrate] "
                    f"step={self.profile_block_step} "
                    f"total_hit_rate={total_hit_rate:.4f} "
                    f"hit_blocks={self.profile_block_hits} "
                    f"miss_blocks={self.profile_block_misses}"
                )
            self.profile_block_hits = 0
            self.profile_block_misses = 0
            self.profile_layer_hit_rate_count = 0

        if profile:
            self._profile_pending_events.append(
                {
                    "start": start_event,
                    "sim_end": sim_event,
                    "retrieval": retrieval_timing,
                    "attn_start": attn_start_event,
                    "attn_end": attn_end_event,
                    "total_end": total_end_event,
                    "delta_events": (delta_start_event, delta_end_event),
                    "h2d_ms": h2d_ms,
                    "cache_sync_ms": cache_sync_ms,
                    "scatter_ms": scatter_ms,
                    "cache_profile_stats": cache_profile_stats,
                }
            )

            if layer_idx == self.layer_num - 1:
                for event_record in self._profile_pending_events:
                    event_record["total_end"].synchronize()
                for event_record in self._profile_pending_events:
                    retrieval_events = event_record.get("retrieval") or {}
                    lookup_events = retrieval_events.get("lookup_events")
                    gather_events = retrieval_events.get("gather_events")
                    copy_events = retrieval_events.get("copy_events") or {}
                    lookup_ms = self._elapsed_ms(*lookup_events) if lookup_events else 0.0
                    concat_copy_ms = self._elapsed_ms(*gather_events) if gather_events else 0.0
                    static_copy_ms = self._elapsed_ms(*copy_events["static"]) if "static" in copy_events else 0.0
                    miss_copy_ms = self._elapsed_ms(*copy_events["miss"]) if "miss" in copy_events else 0.0
                    hit_copy_ms = self._elapsed_ms(*copy_events["hit"]) if "hit" in copy_events else 0.0
                    gather_ms = concat_copy_ms
                    h2d_ms = float(event_record.get("h2d_ms", 0.0) or 0.0)
                    cache_sync_ms = float(event_record.get("cache_sync_ms", 0.0) or 0.0)
                    scatter_ms = float(event_record.get("scatter_ms", 0.0) or 0.0)
                    delta_events = event_record.get("delta_events")
                    delta_append_ms = self._elapsed_ms(*delta_events) if delta_events else 0.0
                    selection_transfer_ms = 0.0

                    cache_profile_stats = event_record.get("cache_profile_stats") or {}
                    retrieval_cache_stats = retrieval_events.get("cache_stats") if retrieval_events else None
                    if retrieval_cache_stats:
                        for key, value in retrieval_cache_stats.items():
                            if torch.is_tensor(value):
                                cache_profile_stats[key] = float(value.item())
                            else:
                                cache_profile_stats[key] = float(value or 0.0)
                    diagnostic_tensors = retrieval_events.get("diagnostic_tensors") if retrieval_events else None
                    if diagnostic_tensors:
                        cache_profile_stats.update(
                            self._profile_miss_run_histogram(
                                diagnostic_tensors["miss_copy_sizes"],
                                diagnostic_tensors["miss_copy_counts"],
                            )
                        )
                        cache_profile_stats["hit_tokens"] = self._sum_descriptor_units(
                            diagnostic_tensors["hit_unit_sizes"],
                            diagnostic_tensors["hit_num_units"],
                        )
                        cache_profile_stats["miss_tokens"] = self._sum_descriptor_units(
                            diagnostic_tensors["miss_unit_sizes"],
                            diagnostic_tensors["miss_num_units"],
                        )

                    search_ms = self._elapsed_ms(event_record["start"], event_record["sim_end"])
                    attn_ms = self._elapsed_ms(event_record["attn_start"], event_record["attn_end"])
                    total_ms = self._elapsed_ms(event_record["start"], event_record["total_end"])
                    gather_total_ms = selection_transfer_ms + lookup_ms + h2d_ms + gather_ms + delta_append_ms

                    self._profile_step_breakdown["search_ms"] += float(search_ms)
                    self._profile_step_breakdown["gather_ms"] += float(gather_total_ms)
                    self._profile_step_breakdown["selection_d2h_ms"] += float(selection_transfer_ms)
                    self._profile_step_breakdown["gpu_cache_plan_ms"] += float(lookup_ms)
                    self._profile_step_breakdown["concat_copy_ms"] += float(concat_copy_ms)
                    self._profile_step_breakdown["static_copy_ms"] += float(static_copy_ms)
                    self._profile_step_breakdown["miss_copy_ms"] += float(miss_copy_ms)
                    self._profile_step_breakdown["hit_copy_ms"] += float(hit_copy_ms)
                    self._profile_step_breakdown["cache_sync_ms"] += float(cache_sync_ms)
                    self._profile_step_breakdown["scatter_ms"] += float(scatter_ms)
                    self._profile_step_breakdown["delta_append_ms"] += float(delta_append_ms)
                    self._profile_step_breakdown["attn_ms"] += float(attn_ms)
                    self._profile_step_breakdown["total_ms"] += float(total_ms)
                    if cache_profile_stats is not None:
                        for key in self._CACHE_PROFILE_SUM_KEYS:
                            self._profile_step_breakdown[key] += float(cache_profile_stats.get(key, 0.0) or 0.0)
                        for key in self._CACHE_PROFILE_MAX_KEYS:
                            self._profile_step_breakdown[key] = max(
                                float(self._profile_step_breakdown.get(key, 0.0) or 0.0),
                                float(cache_profile_stats.get(key, 0.0) or 0.0),
                            )

                step_breakdown = dict(self._profile_step_breakdown)
                step_breakdown["other_ms"] = max(
                    step_breakdown["total_ms"]
                    - step_breakdown["search_ms"]
                    - step_breakdown["gather_ms"]
                    - step_breakdown["attn_ms"],
                    0.0,
                )

                for k, v in step_breakdown.items():
                    self.decode_breakdown_accum.setdefault(k, 0.0)
                    self.decode_breakdown_accum[k] += float(v)

                self.profile_decode_count += 1
                hit_blocks = float(step_breakdown.get("hit_blocks", 0.0) or 0.0)
                miss_blocks = float(step_breakdown.get("miss_blocks", 0.0) or 0.0)
                total_blocks = hit_blocks + miss_blocks
                use_cluster_uva = bool(
                    getattr(self, "_profile_step_use_cluster_uva", False)
                )
                self.decode_profile_rows.append(
                    {
                        "profile_step": int(self.profile_decode_count),
                        "use_cluster_uva": use_cluster_uva,
                        "cache_hit_rate_valid": not use_cluster_uva,
                        "block_hit_rate": (
                            hit_blocks / total_blocks if total_blocks > 0 else 0.0
                        ),
                        **{
                            key: float(value)
                            for key, value in step_breakdown.items()
                            if isinstance(value, (int, float))
                        },
                    }
                )
                if self.profile_decode_count >= self.profile_decode_steps:
                    self.decode_breakdown = {
                        k: v / float(self.profile_decode_count)
                        for k, v in self.decode_breakdown_accum.items()
                    }

                print(
                    "[IMI decode breakdown] step={} search={:.3f}ms gather={:.3f}ms "
                    "attn={:.3f}ms other={:.3f}ms total={:.3f}ms".format(
                        self.profile_decode_count,
                        step_breakdown["search_ms"],
                        step_breakdown["gather_ms"],
                        step_breakdown["attn_ms"],
                        step_breakdown["other_ms"],
                        step_breakdown["total_ms"],
                    )
                )
                if os.getenv("IMI_PROFILE_GATHER_DETAIL", "0") == "1":
                    print(
                        "[IMI decode gather] step={} selection_d2h={:.3f}ms gpu_cache_plan={:.3f}ms "
                        "concat_copy={:.3f}ms static_copy={:.3f}ms miss_copy={:.3f}ms hit_copy={:.3f}ms "
                        "cache_sync={:.3f}ms scatter={:.3f}ms "
                        "delta_append={:.3f}ms".format(
                            self.profile_decode_count,
                            step_breakdown.get("selection_d2h_ms", 0.0),
                            step_breakdown.get("gpu_cache_plan_ms", 0.0),
                            step_breakdown.get("concat_copy_ms", 0.0),
                            step_breakdown.get("static_copy_ms", 0.0),
                            step_breakdown.get("miss_copy_ms", 0.0),
                            step_breakdown.get("hit_copy_ms", 0.0),
                            step_breakdown.get("cache_sync_ms", 0.0),
                            step_breakdown.get("scatter_ms", 0.0),
                            step_breakdown.get("delta_append_ms", 0.0),
                        )
                    )
                    if os.getenv("IMI_PROFILE_CACHE_DETAIL", os.getenv("IMI_PROFILE_GATHER_DETAIL", "0")) == "1":
                        miss_pages = float(step_breakdown.get("miss_pages", step_breakdown.get("miss_blocks", 0.0)) or 0.0)
                        miss_copy_runs = float(step_breakdown.get("miss_copy_runs", 0.0) or 0.0)
                        miss_coalescing = miss_pages / max(miss_copy_runs, 1.0)
                        miss_copy_ms = float(step_breakdown.get("miss_copy_ms", 0.0) or 0.0)
                        elem_size = torch.tensor([], dtype=self.dtype).element_size()
                        miss_bytes = miss_pages * float(self.page_size) * float(self.head_dim) * float(elem_size) * 2.0
                        miss_bw_gibs = (miss_bytes / float(1024 ** 3)) / max(miss_copy_ms / 1000.0, 1.0e-9)
                        print(
                            "[IMI cache stats] step={} hit_blk={} miss_blk={} upd_blk={} evict_blk={} "
                            "miss_pages={} miss_runs={} coal={:.2f} miss_bw={:.2f}GiB/s "
                            "run_hist=1p:{}/2-4p:{}/5-8p:{}/gt8p:{} max_run={} "
                            "max_upd_blk={} max_evict_blk={} "
                            "promote_submit={} promote_done={} promote_ms={:.3f} "
                            "promote_max={:.3f} promote_skip_busy={} promote_no_buf={} inflight={}".format(
                                self.profile_decode_count,
                                int(step_breakdown.get("hit_blocks", 0.0)),
                                int(step_breakdown.get("miss_blocks", 0.0)),
                                int(step_breakdown.get("update_blocks", 0.0)),
                                int(step_breakdown.get("evicted_blocks", 0.0)),
                                int(miss_pages),
                                int(miss_copy_runs),
                                miss_coalescing,
                                miss_bw_gibs,
                                int(step_breakdown.get("miss_runs_1p", 0.0)),
                                int(step_breakdown.get("miss_runs_2_4p", 0.0)),
                                int(step_breakdown.get("miss_runs_5_8p", 0.0)),
                                int(step_breakdown.get("miss_runs_gt8p", 0.0)),
                                int(step_breakdown.get("max_miss_run_pages", 0.0)),
                                int(step_breakdown.get("max_group_update_blocks", 0.0)),
                                int(step_breakdown.get("max_group_evicted_blocks", 0.0)),
                                int(step_breakdown.get("promotion_submitted", 0.0)),
                                int(step_breakdown.get("promotion_completed", 0.0)),
                                float(step_breakdown.get("promotion_ms", 0.0)),
                                float(step_breakdown.get("promotion_ms_max", 0.0)),
                                int(step_breakdown.get("promotion_skipped_busy", 0.0)),
                                int(step_breakdown.get("promotion_skipped_no_buffer", 0.0)),
                                int(step_breakdown.get("promotion_inflight_at_retrieval", 0.0)),
                            )
                        )

                if self.profile_decode_count >= self.profile_decode_steps:
                    avg = self.decode_breakdown
                    print(
                        "[IMI decode breakdown][avg] search={:.3f}ms gather={:.3f}ms "
                        "attn={:.3f}ms other={:.3f}ms total={:.3f}ms".format(
                            avg["search_ms"],
                            avg["gather_ms"],
                            avg["attn_ms"],
                            avg["other_ms"],
                            avg["total_ms"],
                        )
                    )
                    if os.getenv("IMI_PROFILE_GATHER_DETAIL", "0") == "1":
                        print(
                            "[IMI decode gather][avg] selection_d2h={:.3f}ms gpu_cache_plan={:.3f}ms "
                            "concat_copy={:.3f}ms static_copy={:.3f}ms miss_copy={:.3f}ms hit_copy={:.3f}ms "
                            "cache_sync={:.3f}ms scatter={:.3f}ms "
                            "delta_append={:.3f}ms".format(
                                avg.get("selection_d2h_ms", 0.0),
                                avg.get("gpu_cache_plan_ms", 0.0),
                                avg.get("concat_copy_ms", 0.0),
                                avg.get("static_copy_ms", 0.0),
                                avg.get("miss_copy_ms", 0.0),
                                avg.get("hit_copy_ms", 0.0),
                                avg.get("cache_sync_ms", 0.0),
                                avg.get("scatter_ms", 0.0),
                                avg.get("delta_append_ms", 0.0),
                            )
                        )
                    if os.getenv("IMI_PROFILE_CACHE_DETAIL", os.getenv("IMI_PROFILE_GATHER_DETAIL", "0")) == "1":
                        miss_pages = float(avg.get("miss_pages", avg.get("miss_blocks", 0.0)) or 0.0)
                        miss_copy_runs = float(avg.get("miss_copy_runs", 0.0) or 0.0)
                        miss_coalescing = miss_pages / max(miss_copy_runs, 1.0)
                        miss_copy_ms = float(avg.get("miss_copy_ms", 0.0) or 0.0)
                        elem_size = torch.tensor([], dtype=self.dtype).element_size()
                        miss_bytes = miss_pages * float(self.page_size) * float(self.head_dim) * float(elem_size) * 2.0
                        miss_bw_gibs = (miss_bytes / float(1024 ** 3)) / max(miss_copy_ms / 1000.0, 1.0e-9)
                        print(
                            "[IMI cache stats][avg] hit_blk={:.1f} miss_blk={:.1f} upd_blk={:.1f} evict_blk={:.1f} "
                            "miss_pages={:.1f} miss_runs={:.1f} coal={:.2f} miss_bw={:.2f}GiB/s "
                            "run_hist=1p:{:.1f}/2-4p:{:.1f}/5-8p:{:.1f}/gt8p:{:.1f} max_run={:.1f} "
                            "max_upd_blk={:.1f} max_evict_blk={:.1f} "
                            "promote_submit={:.1f} promote_done={:.1f} "
                            "promote_ms={:.3f} promote_max={:.3f} promote_skip_busy={:.1f} promote_no_buf={:.1f} "
                            "inflight={:.1f}".format(
                                avg.get("hit_blocks", 0.0),
                                avg.get("miss_blocks", 0.0),
                                avg.get("update_blocks", 0.0),
                                avg.get("evicted_blocks", 0.0),
                                miss_pages,
                                miss_copy_runs,
                                miss_coalescing,
                                miss_bw_gibs,
                                avg.get("miss_runs_1p", 0.0),
                                avg.get("miss_runs_2_4p", 0.0),
                                avg.get("miss_runs_5_8p", 0.0),
                                avg.get("miss_runs_gt8p", 0.0),
                                avg.get("max_miss_run_pages", 0.0),
                                avg.get("max_group_update_blocks", 0.0),
                                avg.get("max_group_evicted_blocks", 0.0),
                                avg.get("promotion_submitted", 0.0),
                                avg.get("promotion_completed", 0.0),
                                avg.get("promotion_ms", 0.0),
                                avg.get("promotion_ms_max", 0.0),
                                avg.get("promotion_skipped_busy", 0.0),
                                avg.get("promotion_skipped_no_buffer", 0.0),
                                avg.get("promotion_inflight_at_retrieval", 0.0),
                            )
                        )

        return acc_out.view(self.batch_size, 1, self.num_heads, self.head_dim)
