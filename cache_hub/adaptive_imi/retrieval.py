from typing import Optional

import os
import torch
from weighted_flash_decoding import weighted_flash_decoding

from library.AdaptiveIMI.imi_adapter import get_imi_kernels
from library.AdaptiveIMI.cpp_extensions import (
    build_cache_aware_retrieval_plan,
    gather_copy_and_concat,
    promote_cache_pages_from_list,
)

imi_gpu_kernels = get_imi_kernels()


class AdaptiveIMIRetrievalMixin:
    _GPU_CACHE_PROMOTION_WARMUP_STEPS = 4
    _GPU_CACHE_PROMOTION_WARMUP_PAGES_PER_GROUP = -1
    _GPU_CACHE_PROMOTION_STEADY_PAGES_PER_GROUP = 64

    _CACHE_PROFILE_SUM_KEYS = (
        "hit_clusters",
        "miss_clusters",
        "hit_blocks",
        "miss_blocks",
        "selected_valid_clusters",
        "considered_blocks",
        "truncated_groups",
        "dropped_clusters_after_budget",
        "admitted_clusters",
        "update_blocks",
        "evicted_clusters",
        "evicted_blocks",
        "free_blocks_before",
        "free_blocks_after",
        "hot_blocks",
        "cold_blocks",
        "hot_promotions",
        "hot_demotions",
        "batch_access_us",
        "batch_update_us",
        "sync_wait_us",
        "access_worker_sum_us",
        "update_worker_sum_us",
        "promotion_submitted",
        "promotion_skipped_busy",
    )
    _CACHE_PROFILE_MAX_KEYS = (
        "access_worker_max_us",
        "update_worker_max_us",
        "max_group_access_us",
        "max_group_update_us",
        "max_group_considered_blocks",
        "max_group_miss_blocks",
        "max_group_update_blocks",
        "max_group_evicted_blocks",
    )

    def _flat_score_mode(self) -> str:
        return str(getattr(self, "flat_score_mode", "raw_sum")).lower()

    def _softmax_logits_buffer(self, device: str) -> torch.Tensor:
        buffer = self.softmax_logits_buffer_dict.get(device)
        shape = (self.batch_groups, self.group_size, self.n_centroids)
        if buffer is None or tuple(buffer.shape) != shape or buffer.device != torch.device(device):
            buffer = torch.empty(shape, dtype=torch.float32, device=device)
            self.softmax_logits_buffer_dict[device] = buffer
        return buffer
    def _select_clusters_raw_sum_flat(self, layer_idx: int, query_grouped: torch.Tensor, device: str) -> torch.Tensor:
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
            "miss_num_units": self.gpu_cache_aware_miss_num_units[layer_idx][slot],
            "miss_page_keys": self.gpu_cache_aware_miss_page_keys[layer_idx][slot],
            "hit_unit_idices": self.gpu_cache_aware_hit_unit_idices[layer_idx][slot],
            "hit_unit_sizes": self.gpu_cache_aware_hit_unit_sizes[layer_idx][slot],
            "hit_unit_sizes_cumsum": self.gpu_cache_aware_hit_unit_sizes_cumsum[layer_idx][slot],
            "hit_num_units": self.gpu_cache_aware_hit_num_units[layer_idx][slot],
        }
    def _gpu_cache_promotion_pages_per_group(self) -> int:
        step = int(getattr(self, "gpu_cache_promotion_decode_step", 0))
        if step <= self._GPU_CACHE_PROMOTION_WARMUP_STEPS:
            return self._GPU_CACHE_PROMOTION_WARMUP_PAGES_PER_GROUP
        return self._GPU_CACHE_PROMOTION_STEADY_PAGES_PER_GROUP

    def _tokens_to_pages(self, token_budget: int) -> int:
        if token_budget <= 0:
            return 0
        return max((int(token_budget) + self.page_size - 1) // self.page_size, 1)
    def _resolve_budget_pages(self, requested_pages: int, minimum_pages: int = 1) -> int:
        return max(int(requested_pages), minimum_pages)
    def _empty_cache_profile_stats(self) -> dict:
        stats = {key: 0.0 for key in self._CACHE_PROFILE_SUM_KEYS}
        stats.update({key: 0.0 for key in self._CACHE_PROFILE_MAX_KEYS})
        stats["max_group_access_id"] = -1
        stats["max_group_update_id"] = -1
        return stats
    def _merge_cache_profile_stats(self, total: dict, stats: Optional[dict]) -> None:
        if not stats:
            return
        for key in self._CACHE_PROFILE_SUM_KEYS:
            total[key] = float(total.get(key, 0.0)) + float(stats.get(key, 0.0) or 0.0)
        for key in self._CACHE_PROFILE_MAX_KEYS:
            total[key] = max(float(total.get(key, 0.0)), float(stats.get(key, 0.0) or 0.0))
        if float(stats.get("max_group_access_us", 0.0) or 0.0) >= float(total.get("max_group_access_us", 0.0)):
            total["max_group_access_id"] = int(stats.get("max_group_access_id", -1))
        if float(stats.get("max_group_update_us", 0.0) or 0.0) >= float(total.get("max_group_update_us", 0.0)):
            total["max_group_update_id"] = int(stats.get("max_group_update_id", -1))
    def _new_timing_event(self) -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)
    def _record_timing_event(self) -> torch.cuda.Event:
        event = self._new_timing_event()
        event.record()
        return event
    def _elapsed_ms(self, start: torch.cuda.Event, end: torch.cuda.Event) -> float:
        return float(start.elapsed_time(end))
    def _run_retrieval_full_gpu_cache_aware(
        self,
        layer_idx: int,
        device: str,
        cluster_ids: torch.Tensor,
        profile_decode: bool = False,
    ) -> Optional[dict]:
        descriptor_slot = self._select_gpu_cache_aware_descriptor_slot(layer_idx)
        desc = self._gpu_cache_aware_descriptors(layer_idx, descriptor_slot)

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
            desc["miss_num_units"],
            desc["miss_page_keys"],
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

        gather_copy_and_concat(
            self.steady_zone_keys[layer_idx],
            self.list_keys[layer_idx],
            self.cache_keys[layer_idx],
            self.execution_buffer_keys_dict[device],
            self.steady_zone_values[layer_idx],
            self.list_values[layer_idx],
            self.cache_values[layer_idx],
            self.execution_buffer_values_dict[device],
            desc["miss_unit_idices"],
            desc["miss_unit_sizes"],
            desc["miss_unit_sizes_cumsum"],
            desc["miss_num_units"],
            desc["hit_unit_idices"],
            desc["hit_unit_sizes"],
            desc["hit_unit_sizes_cumsum"],
            desc["hit_num_units"],
            self.valid_lengths_dict[device],
            self.batch_groups,
            self.static_stride,
            self.list_capacity,
            self.cache_size,
            self.execution_stride,
            self.buffer_size,
            self.static_len_tensor_dict[device],
        )

        if not profile_decode:
            return None

        gather_end_event = self._record_timing_event()
        promote_pages_per_group = self._gpu_cache_promotion_pages_per_group()
        if promote_pages_per_group < 0:
            planned_update_blocks = desc["miss_num_units"].sum()
        else:
            planned_update_blocks = torch.clamp(desc["miss_num_units"], max=promote_pages_per_group).sum()
        gpu_cache_stats = {
            "hit_blocks": desc["hit_num_units"].sum(),
            "miss_blocks": desc["miss_num_units"].sum(),
            "update_blocks": planned_update_blocks,
        }
        return {
            "lookup_events": (lookup_start_event, lookup_end_event),
            "gather_events": (gather_start_event, gather_end_event),
            "h2d_ms": 0.0,
            "cache_sync_ms": 0.0,
            "scatter_ms": 0.0,
            "cache_stats": gpu_cache_stats,
        }
    def _schedule_gpu_cache_promotion(self, layer_idx: int, device: str, static_len: int) -> dict:
        del static_len
        descriptor_slot = int(self.gpu_cache_aware_active_slots[layer_idx])
        desc = self._gpu_cache_aware_descriptors(layer_idx, descriptor_slot)
        stats = {
            "promotion_submitted": 0.0,
            "promotion_skipped_busy": 0.0,
        }

        prior_event = self.gpu_cache_promote_events[layer_idx]
        if prior_event is not None:
            if not prior_event.query():
                stats["promotion_skipped_busy"] = 1.0
                return stats
            self.gpu_cache_promote_events[layer_idx] = None
            self.gpu_cache_promote_slots[layer_idx] = -1

        stream = self.gpu_cache_promote_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device)
            self.gpu_cache_promote_streams[device] = stream

        current_stream = torch.cuda.current_stream(device)
        ready_event = torch.cuda.Event()
        done_event = torch.cuda.Event()
        ready_event.record(current_stream)
        with torch.cuda.stream(stream):
            stream.wait_event(ready_event)
            promote_cache_pages_from_list(
                self.list_keys[layer_idx],
                self.cache_keys[layer_idx],
                self.list_values[layer_idx],
                self.cache_values[layer_idx],
                desc["miss_unit_idices"],
                desc["miss_unit_sizes"],
                desc["miss_num_units"],
                desc["miss_page_keys"],
                self.gpu_cache_hash_keys[layer_idx],
                self.gpu_cache_hash_values[layer_idx],
                self.gpu_cache_slot_keys[layer_idx],
                self.gpu_cache_next_slots[layer_idx],
                int(self.batch_groups),
                int(self.list_capacity),
                int(self.cache_size),
                int(self.gpu_cache_hash_size),
                int(self.buffer_size),
                int(self._gpu_cache_promotion_pages_per_group()),
            )
            done_event.record(stream)
        self.gpu_cache_promote_desc_events[layer_idx] = None
        self.gpu_cache_promote_events[layer_idx] = done_event
        self.gpu_cache_promote_slots[layer_idx] = descriptor_slot
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
        sizes = torch.gather(cluster_sizes, 1, topk.indices)
        valid = sizes > 0
        cumsum = torch.cumsum(sizes, dim=-1, dtype=torch.int32)
        keep = valid & ((cumsum - sizes) < self.max_retrieval_tokens)
        selected = topk.indices.masked_fill(~keep, self.padding_cluster_id)
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
            self._profile_step_breakdown = {
                "search_ms": 0.0,
                "gather_ms": 0.0,
                "selection_d2h_ms": 0.0,
                "cpu_lookup_ms": 0.0,
                "concat_copy_ms": 0.0,
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
        custom_selector = getattr(self, "_select_clusters_for_decode", None)
        if custom_selector is None:
            if self._flat_score_mode() in ("head_softmax_sum", "per_head_softmax_sum"):
                similarities = imi_gpu_kernels.fused_query_group_softmax_sum_similarities(
                    query_grouped,
                    self.centroids[layer_idx],
                    self._softmax_logits_buffer(device),
                    self.similarity_buffer_dict[device],
                )
                similarities.masked_fill_(self.centroids_mask[layer_idx], float("-inf"))
                selected = self._select_clusters(layer_idx, similarities)
            else:
                selected = self._select_clusters_raw_sum_flat(layer_idx, query_grouped, device)
        else:
            selected = custom_selector(layer_idx, query_grouped, device)

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

        retrieval_timing = self._run_retrieval_full_gpu_cache_aware(
            layer_idx,
            device,
            selected,
            profile_decode=profile,
        )

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

        token_reranker = getattr(self, "_rerank_execution_tokens_for_decode", None)
        if token_reranker is not None:
            token_reranker(layer_idx, device, query_grouped)

        if profile:
            attn_start_event = self._record_timing_event()

        acc_out = weighted_flash_decoding(
            query_view,
            self.execution_buffer_keys_dict[device],
            self.execution_buffer_values_dict[device],
            previous_out=None,
            previous_lse=None,
            cache_seqlens=self.valid_lengths_dict[device],
            return_softmax_lse=False,
        )

        if profile:
            attn_end_event = self._record_timing_event()
            total_end_event = self._record_timing_event()

        promotion_stats = self._schedule_gpu_cache_promotion(layer_idx, device, int(static_len))
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
                    "h2d_ms": h2d_ms,
                    "cache_sync_ms": cache_sync_ms,
                    "scatter_ms": scatter_ms,
                    "delta_append_ms": delta_append_ms,
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
                    lookup_ms = self._elapsed_ms(*lookup_events) if lookup_events else 0.0
                    concat_copy_ms = self._elapsed_ms(*gather_events) if gather_events else 0.0
                    gather_ms = concat_copy_ms
                    h2d_ms = float(event_record.get("h2d_ms", 0.0) or 0.0)
                    cache_sync_ms = float(event_record.get("cache_sync_ms", 0.0) or 0.0)
                    scatter_ms = float(event_record.get("scatter_ms", 0.0) or 0.0)
                    delta_append_ms = float(event_record.get("delta_append_ms", 0.0) or 0.0)
                    selection_transfer_ms = 0.0

                    cache_profile_stats = event_record.get("cache_profile_stats") or {}
                    retrieval_cache_stats = retrieval_events.get("cache_stats") if retrieval_events else None
                    if retrieval_cache_stats:
                        for key, value in retrieval_cache_stats.items():
                            if torch.is_tensor(value):
                                cache_profile_stats[key] = float(value.item())
                            else:
                                cache_profile_stats[key] = float(value or 0.0)

                    search_ms = self._elapsed_ms(event_record["start"], event_record["sim_end"])
                    attn_ms = self._elapsed_ms(event_record["attn_start"], event_record["attn_end"])
                    total_ms = self._elapsed_ms(event_record["start"], event_record["total_end"])
                    gather_total_ms = selection_transfer_ms + lookup_ms + h2d_ms + gather_ms + delta_append_ms

                    self._profile_step_breakdown["search_ms"] += float(search_ms)
                    self._profile_step_breakdown["gather_ms"] += float(gather_total_ms)
                    self._profile_step_breakdown["selection_d2h_ms"] += float(selection_transfer_ms)
                    self._profile_step_breakdown["cpu_lookup_ms"] += float(lookup_ms)
                    self._profile_step_breakdown["concat_copy_ms"] += float(concat_copy_ms)
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
                        if float(cache_profile_stats.get("max_group_access_us", 0.0) or 0.0) >= float(
                            self._profile_step_breakdown.get("max_group_access_us", 0.0) or 0.0
                        ):
                            self._profile_step_breakdown["max_group_access_id"] = int(
                                cache_profile_stats.get("max_group_access_id", -1)
                            )
                        if float(cache_profile_stats.get("max_group_update_us", 0.0) or 0.0) >= float(
                            self._profile_step_breakdown.get("max_group_update_us", 0.0) or 0.0
                        ):
                            self._profile_step_breakdown["max_group_update_id"] = int(
                                cache_profile_stats.get("max_group_update_id", -1)
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
                if self.profile_decode_count == self.profile_decode_steps:
                    self.decode_breakdown = {
                        k: v / float(self.profile_decode_steps) for k, v in self.decode_breakdown_accum.items()
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
                if os.getenv("IMI_PROFILE_GATHER_DETAIL", "1") == "1":
                    print(
                        "[IMI decode gather] step={} selection_d2h={:.3f}ms cpu_lookup={:.3f}ms "
                        "concat_copy={:.3f}ms cache_sync={:.3f}ms scatter={:.3f}ms "
                        "delta_append={:.3f}ms".format(
                            self.profile_decode_count,
                            step_breakdown.get("selection_d2h_ms", 0.0),
                            step_breakdown.get("cpu_lookup_ms", 0.0),
                            step_breakdown.get("concat_copy_ms", 0.0),
                            step_breakdown.get("cache_sync_ms", 0.0),
                            step_breakdown.get("scatter_ms", 0.0),
                            step_breakdown.get("delta_append_ms", 0.0),
                        )
                    )
                    if os.getenv("IMI_PROFILE_CACHE_DETAIL", os.getenv("IMI_PROFILE_GATHER_DETAIL", "1")) == "1":
                        print(
                            "[IMI cache stats] step={} hit_blk={} miss_blk={} upd_blk={} evict_blk={} "
                            "valid_clu={} hit_clu={} miss_clu={} consider_blk={} trunc_grp={} drop_clu={} "
                            "access_us={} sync_wait_us={} update_us={} max_access_us={} gid={} "
                            "max_update_us={} gid={} max_upd_blk={} max_evict_blk={} "
                            "promote_submit={} promote_skip_busy={}".format(
                                self.profile_decode_count,
                                int(step_breakdown.get("hit_blocks", 0.0)),
                                int(step_breakdown.get("miss_blocks", 0.0)),
                                int(step_breakdown.get("update_blocks", 0.0)),
                                int(step_breakdown.get("evicted_blocks", 0.0)),
                                int(step_breakdown.get("selected_valid_clusters", 0.0)),
                                int(step_breakdown.get("hit_clusters", 0.0)),
                                int(step_breakdown.get("miss_clusters", 0.0)),
                                int(step_breakdown.get("considered_blocks", 0.0)),
                                int(step_breakdown.get("truncated_groups", 0.0)),
                                int(step_breakdown.get("dropped_clusters_after_budget", 0.0)),
                                int(step_breakdown.get("batch_access_us", 0.0)),
                                int(step_breakdown.get("sync_wait_us", 0.0)),
                                int(step_breakdown.get("batch_update_us", 0.0)),
                                int(step_breakdown.get("max_group_access_us", 0.0)),
                                int(step_breakdown.get("max_group_access_id", -1)),
                                int(step_breakdown.get("max_group_update_us", 0.0)),
                                int(step_breakdown.get("max_group_update_id", -1)),
                                int(step_breakdown.get("max_group_update_blocks", 0.0)),
                                int(step_breakdown.get("max_group_evicted_blocks", 0.0)),
                                int(step_breakdown.get("promotion_submitted", 0.0)),
                                int(step_breakdown.get("promotion_skipped_busy", 0.0)),
                            )
                        )

                if self.profile_decode_count == self.profile_decode_steps:
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
                    if os.getenv("IMI_PROFILE_GATHER_DETAIL", "1") == "1":
                        print(
                            "[IMI decode gather][avg] selection_d2h={:.3f}ms cpu_lookup={:.3f}ms "
                            "concat_copy={:.3f}ms cache_sync={:.3f}ms scatter={:.3f}ms "
                            "delta_append={:.3f}ms".format(
                                avg.get("selection_d2h_ms", 0.0),
                                avg.get("cpu_lookup_ms", 0.0),
                                avg.get("concat_copy_ms", 0.0),
                                avg.get("cache_sync_ms", 0.0),
                                avg.get("scatter_ms", 0.0),
                                avg.get("delta_append_ms", 0.0),
                            )
                        )
                    if os.getenv("IMI_PROFILE_CACHE_DETAIL", os.getenv("IMI_PROFILE_GATHER_DETAIL", "1")) == "1":
                        print(
                            "[IMI cache stats][avg] hit_blk={:.1f} miss_blk={:.1f} upd_blk={:.1f} evict_blk={:.1f} "
                            "valid_clu={:.1f} hit_clu={:.1f} miss_clu={:.1f} consider_blk={:.1f} trunc_grp={:.1f} drop_clu={:.1f} "
                            "access_us={:.1f} sync_wait_us={:.1f} update_us={:.1f} max_access_us={:.1f} "
                            "max_update_us={:.1f} max_upd_blk={:.1f} max_evict_blk={:.1f} "
                            "promote_submit={:.1f} promote_skip_busy={:.1f}".format(
                                avg.get("hit_blocks", 0.0),
                                avg.get("miss_blocks", 0.0),
                                avg.get("update_blocks", 0.0),
                                avg.get("evicted_blocks", 0.0),
                                avg.get("selected_valid_clusters", 0.0),
                                avg.get("hit_clusters", 0.0),
                                avg.get("miss_clusters", 0.0),
                                avg.get("considered_blocks", 0.0),
                                avg.get("truncated_groups", 0.0),
                                avg.get("dropped_clusters_after_budget", 0.0),
                                avg.get("batch_access_us", 0.0),
                                avg.get("sync_wait_us", 0.0),
                                avg.get("batch_update_us", 0.0),
                                avg.get("max_group_access_us", 0.0),
                                avg.get("max_group_update_us", 0.0),
                                avg.get("max_group_update_blocks", 0.0),
                                avg.get("max_group_evicted_blocks", 0.0),
                                avg.get("promotion_submitted", 0.0),
                                avg.get("promotion_skipped_busy", 0.0),
                            )
                        )

        return acc_out.view(self.batch_size, 1, self.num_heads, self.head_dim)
