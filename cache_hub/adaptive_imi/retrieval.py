from typing import List, Optional

import os
import time

import torch
from weighted_flash_decoding import weighted_flash_decoding

from library.AdaptiveIMI.imi_adapter import get_imi_kernels
from library.AdaptiveIMI.cpp_extensions import gather_copy_and_concat, gather_copy_and_concat_retrieval, gather_copy_and_scatter

imi_gpu_kernels = get_imi_kernels()


class AdaptiveIMIRetrievalMixin:
    def _tokens_to_pages(self, token_budget: int) -> int:
        if token_budget <= 0:
            return 0
        return max((int(token_budget) + self.page_size - 1) // self.page_size, 1)
    def _resolve_budget_pages(self, requested_pages: int, minimum_pages: int = 1) -> int:
        return max(int(requested_pages), minimum_pages)
    def _set_cluster_ids(self, layer_idx: int, cluster_ids: torch.Tensor) -> None:
        target = self.cluster_ids[layer_idx]
        target.fill_(self.padding_cluster_id)
        width = cluster_ids.size(1)
        if width > target.size(1):
            raise ValueError(f"cluster tile width {width} exceeds allocated nprobe {target.size(1)}")
        if width > 0:
            target[:, :width].copy_(cluster_ids)
    def _build_cluster_tiles(self, layer_idx: int, selected: torch.Tensor) -> List[torch.Tensor]:
        if selected.numel() == 0:
            return []

        selected_cpu = selected.to(device="cpu", dtype=torch.int64)

        cluster_sizes = self.cluster_sizes_cpu[layer_idx]
        if cluster_sizes is None:
            return []

        selected_sizes = torch.gather(cluster_sizes, 1, selected_cpu.clamp_max(self.padding_cluster_id))
        selected_sizes = torch.where(selected_cpu == self.padding_cluster_id, 0, selected_sizes)
        selected_pages = torch.div(selected_sizes + self.page_size - 1, self.page_size, rounding_mode="floor")
        selected_pages = torch.where(selected_sizes > 0, torch.clamp(selected_pages, min=1), 0)
        if os.getenv("IMI_DEBUG_OVERSIZED", "0") == "1":
            oversized_mask = (selected_pages > self.buffer_size) & (selected_cpu != self.padding_cluster_id)
            if torch.any(oversized_mask):
                oversized_count = int(oversized_mask.sum().item())
                max_pages = int(selected_pages[oversized_mask].max().item())
                print(
                    "[IMI oversized] "
                    f"layer={layer_idx} count={oversized_count} max_pages={max_pages} "
                    f"buffer_pages={self.buffer_size}"
                )
        prefix_pages = torch.cumsum(selected_pages, dim=1)

        positions = torch.arange(self.nprobe, dtype=torch.int64).unsqueeze(0).expand(self.batch_groups, -1)
        group_offsets = torch.zeros(self.batch_groups, dtype=torch.int64)
        group_counts = (selected_cpu != self.padding_cluster_id).sum(dim=1, dtype=torch.int64)
        tiles: List[torch.Tensor] = []

        while torch.any(group_offsets < group_counts):
            start_valid = group_offsets < group_counts
            start_minus_one = torch.clamp(group_offsets - 1, min=0)
            start_prefix = torch.gather(prefix_pages, 1, start_minus_one.unsqueeze(1)).squeeze(1)
            start_prefix = torch.where(group_offsets > 0, start_prefix, torch.zeros_like(start_prefix))

            relative_pages = prefix_pages - start_prefix.unsqueeze(1)
            candidate_mask = (
                (positions >= group_offsets.unsqueeze(1))
                & (positions < group_counts.unsqueeze(1))
                & (relative_pages <= self.buffer_size)
            )
            take = candidate_mask.sum(dim=1, dtype=torch.int64)

            force_take_mask = start_valid & (take == 0)
            if torch.any(force_take_mask):
                current_positions = group_offsets.unsqueeze(1)
                safe_current = torch.minimum(current_positions, torch.full_like(current_positions, self.nprobe - 1))
                current_pages = torch.gather(selected_pages, 1, safe_current).squeeze(1)

                can_force = current_pages <= self.buffer_size
                force_take_mask = force_take_mask & can_force

                if torch.any(force_take_mask & ~can_force):
                    oversized_count = (force_take_mask & ~can_force).sum().item()
                    if os.getenv("IMI_DEBUG", "0") == "1":
                        print(f"[IMI WARNING] layer={layer_idx} skipping {oversized_count} oversized clusters that exceed buffer_size")

                take = torch.where(force_take_mask, torch.ones_like(take), take)

            take = torch.minimum(take, group_counts - group_offsets)

            max_take = int(take.max().item())
            if max_take <= 0:
                break

            gather_positions = group_offsets.unsqueeze(1) + torch.arange(max_take, dtype=torch.int64).unsqueeze(0)
            safe_positions = torch.minimum(gather_positions, torch.full_like(gather_positions, self.nprobe - 1))
            payload = torch.gather(selected_cpu, 1, safe_positions)
            payload_mask = torch.arange(max_take, dtype=torch.int64).unsqueeze(0) < take.unsqueeze(1)

            tile = torch.full(
                (self.batch_groups, max_take),
                self.padding_cluster_id,
                dtype=torch.int64,
                pin_memory=True,
            )
            tile[payload_mask] = payload[payload_mask]
            tiles.append(tile)
            group_offsets = group_offsets + take

        return tiles
    def _run_retrieval_tile(
        self,
        layer_idx: int,
        device: str,
        tile_cluster_ids: torch.Tensor,
        profile_decode: bool = False,
    ) -> Optional[dict]:
        if profile_decode:
            torch.cuda.synchronize(device)
            lookup_start = time.perf_counter()

        self._set_cluster_ids(layer_idx, tile_cluster_ids)
        self.adpimi_index[layer_idx].batch_access()

        if profile_decode:
            torch.cuda.synchronize(device)
            lookup_end = time.perf_counter()

        if self.profile_block_hit_rate:
            hit_blocks, miss_blocks = self.adpimi_index[layer_idx].get_last_block_stats()
            self.profile_block_hits += hit_blocks
            self.profile_block_misses += miss_blocks
            self.profile_layer_hit_rate_count += 1

        if profile_decode:
            torch.cuda.synchronize(device)
            gather_start = time.perf_counter()

        gather_copy_and_concat_retrieval(
            self.list_keys[layer_idx],
            self.cache_keys[layer_idx],
            self.execution_buffer_keys_dict[device],
            self.list_values[layer_idx],
            self.cache_values[layer_idx],
            self.execution_buffer_values_dict[device],
            self.miss_unit_idices[layer_idx],
            self.miss_unit_sizes[layer_idx],
            self.miss_unit_sizes_cumsum[layer_idx],
            self.miss_num_units[layer_idx],
            self.hit_unit_idices[layer_idx],
            self.hit_unit_sizes[layer_idx],
            self.hit_unit_sizes_cumsum[layer_idx],
            self.hit_num_units[layer_idx],
            self.valid_lengths_dict[device],
            self.batch_groups,
            self.list_stride,
            self.cache_size,
            self.retrieval_execution_stride,
            self.buffer_size,
        )

        if profile_decode:
            torch.cuda.synchronize(device)
            h2d_start = time.perf_counter()

        self.adpimi_index[layer_idx].sync()

        if profile_decode:
            torch.cuda.synchronize(device)
            h2d_end = time.perf_counter()

        zero_static_len = self.static_len_tensor_dict[device]
        zero_static_len.zero_()
        gather_copy_and_scatter(
            self.execution_buffer_keys_dict[device],
            self.cache_keys[layer_idx],
            self.execution_buffer_values_dict[device],
            self.cache_values[layer_idx],
            self.update_buffer_indices[layer_idx],
            self.update_unit_sizes[layer_idx],
            self.update_cache_indices[layer_idx],
            self.update_num_units[layer_idx],
            self.batch_groups,
            self.retrieval_execution_stride,
            self.cache_size,
            self.buffer_size,
            zero_static_len,
        )

        if not profile_decode:
            return None

        torch.cuda.synchronize(device)
        gather_end = time.perf_counter()

        return {
            "lookup_ms": (lookup_end - lookup_start) * 1000.0,
            "gather_ms": (gather_end - gather_start) * 1000.0,
            "h2d_ms": (h2d_end - h2d_start) * 1000.0,
        }
    def _run_retrieval_full(
        self,
        layer_idx: int,
        device: str,
        cluster_ids: torch.Tensor,
        profile_decode: bool = False,
    ) -> Optional[dict]:
        if profile_decode:
            torch.cuda.synchronize(device)
            lookup_start = time.perf_counter()

        self._set_cluster_ids(layer_idx, cluster_ids)
        self.adpimi_index[layer_idx].batch_access()

        if profile_decode:
            torch.cuda.synchronize(device)
            lookup_end = time.perf_counter()

        if self.profile_block_hit_rate:
            hit_blocks, miss_blocks = self.adpimi_index[layer_idx].get_last_block_stats()
            self.profile_block_hits += hit_blocks
            self.profile_block_misses += miss_blocks
            self.profile_layer_hit_rate_count += 1

        if profile_decode:
            torch.cuda.synchronize(device)
            gather_start = time.perf_counter()

        gather_copy_and_concat(
            self.steady_zone_keys[layer_idx],
            self.list_keys[layer_idx],
            self.cache_keys[layer_idx],
            self.execution_buffer_keys_dict[device],
            self.steady_zone_values[layer_idx],
            self.list_values[layer_idx],
            self.cache_values[layer_idx],
            self.execution_buffer_values_dict[device],
            self.miss_unit_idices[layer_idx],
            self.miss_unit_sizes[layer_idx],
            self.miss_unit_sizes_cumsum[layer_idx],
            self.miss_num_units[layer_idx],
            self.hit_unit_idices[layer_idx],
            self.hit_unit_sizes[layer_idx],
            self.hit_unit_sizes_cumsum[layer_idx],
            self.hit_num_units[layer_idx],
            self.valid_lengths_dict[device],
            self.batch_groups,
            self.static_stride,
            self.list_stride,
            self.cache_size,
            self.execution_stride,
            self.buffer_size,
            self.static_len_tensor_dict[device],
        )

        if profile_decode:
            torch.cuda.synchronize(device)
            h2d_start = time.perf_counter()

        self.adpimi_index[layer_idx].sync()

        if profile_decode:
            torch.cuda.synchronize(device)
            h2d_end = time.perf_counter()

        gather_copy_and_scatter(
            self.execution_buffer_keys_dict[device],
            self.cache_keys[layer_idx],
            self.execution_buffer_values_dict[device],
            self.cache_values[layer_idx],
            self.update_buffer_indices[layer_idx],
            self.update_unit_sizes[layer_idx],
            self.update_cache_indices[layer_idx],
            self.update_num_units[layer_idx],
            self.batch_groups,
            self.execution_stride,
            self.cache_size,
            self.buffer_size,
            self.static_len_tensor_dict[device],
        )

        if not profile_decode:
            return None

        torch.cuda.synchronize(device)
        gather_end = time.perf_counter()

        return {
            "lookup_ms": (lookup_end - lookup_start) * 1000.0,
            "gather_ms": (gather_end - gather_start) * 1000.0,
            "h2d_ms": (h2d_end - h2d_start) * 1000.0,
        }
    def _select_clusters(self, layer_idx, similarities):
        topk = torch.topk(similarities, self.nprobe, dim=-1, largest=True, sorted=True)
        cluster_sizes = self.cluster_sizes_gpu[layer_idx]
        sizes = torch.gather(cluster_sizes, 1, topk.indices)
        valid = sizes > 0
        cumsum = torch.cumsum(sizes, dim=-1, dtype=torch.int32)
        keep = valid & ((cumsum - sizes) < self.max_retrieval_tokens)
        selected = topk.indices.masked_fill(~keep, self.padding_cluster_id)
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
            if self.profile_decode_step_idx < self.profile_decode_steps:
                self._profile_step_breakdown = {
                    "search_ms": 0.0,
                    "gather_ms": 0.0,
                    "attn_ms": 0.0,
                    "total_ms": 0.0,
                }
        if profile:
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()

        query_grouped = queries.view(self.batch_groups, self.group_size, self.head_dim).contiguous()
        similarities = imi_gpu_kernels.fused_query_group_similarities(
            query_grouped,
            self.centroids[layer_idx],
            self.similarity_buffer_dict[device],
        )
        similarities.masked_fill_(self.centroids_mask[layer_idx], float("-inf"))

        if profile:
            torch.cuda.synchronize(device)
            sim_time = time.perf_counter()

        selected = self._select_clusters(layer_idx, similarities)
        if debug_decode and layer_idx == 0:
            selected_count = int((selected != self.padding_cluster_id).sum().item())
            selected_sample = selected[0, : min(16, selected.shape[1])].tolist()
            print(
                "[IMI DECODE]"
                f" step={self.decode_debug_step} layer={layer_idx} selected_clusters={selected_count}"
                f" selected_sample={selected_sample}",
                flush=True,
            )
        max_tiles = int(os.getenv("IMI_MAX_TILES", "0"))
        selected_cpu = None
        tiles = None
        if max_tiles == 0:
            selected_cpu = selected.to(device="cpu", dtype=torch.int64)
            tiles = [selected_cpu]
            use_tiles = False
        else:
            tiles = self._build_cluster_tiles(layer_idx, selected)
            use_tiles = True

        # Debug: 打印 tile 信息
        if os.getenv("IMI_DEBUG_TILES", "0") == "1" and layer_idx == 0:
            base_selected = selected_cpu if selected_cpu is not None else selected
            num_selected = (base_selected != self.padding_cluster_id).sum().item()
            print(f"[IMI tiles] layer={layer_idx} num_tiles={len(tiles)} num_selected_clusters={num_selected} nprobe={self.nprobe}")
            for i, tile in enumerate(tiles):
                tile_size = (tile != self.padding_cluster_id).sum().item()
                print(f"  tile[{i}]: size={tile_size} shape={tile.shape}")

        if profile:
            torch.cuda.synchronize(device)
            select_time = time.perf_counter()

        lookup_ms = 0.0
        h2d_ms = 0.0
        gather_ms = 0.0
        attn_ms = 0.0

        query_view = queries.view(self.batch_groups, 1, self.group_size, self.head_dim)

        if not use_tiles:
            if profile:
                static_time = select_time

            self.valid_lengths_dict[device].zero_()

            retrieval_timing = self._run_retrieval_full(
                layer_idx,
                device,
                selected_cpu,
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
                lookup_ms += retrieval_timing["lookup_ms"]
                h2d_ms += retrieval_timing["h2d_ms"]
                gather_ms += retrieval_timing["gather_ms"]

            self._append_delta_to_execution(layer_idx, device, selected_cpu)

            if profile:
                torch.cuda.synchronize(device)
                attn_start = time.perf_counter()

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
                torch.cuda.synchronize(device)
                attn_ms += (time.perf_counter() - attn_start) * 1000.0
        else:
            static_out, static_lse = weighted_flash_decoding(
                queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
                self.steady_zone_keys[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
                self.steady_zone_values[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
                previous_out=None,
                previous_lse=None,
                cache_seqlens=self.static_lengths_dict[device],
                return_softmax_lse=True,
            )

            if profile:
                torch.cuda.synchronize(device)
                static_time = time.perf_counter()

            acc_out, acc_lse = static_out, static_lse

            if debug_decode and layer_idx == 0:
                static_lengths = self.static_lengths_dict[device]
                static_min = int(static_lengths.min().item()) if static_lengths.numel() else 0
                static_max = int(static_lengths.max().item()) if static_lengths.numel() else 0
                print(
                    "[IMI DECODE]"
                    f" step={self.decode_debug_step} layer={layer_idx} static_lengths[min,max]=[{static_min},{static_max}]"
                    f" num_tiles={len(tiles)}",
                    flush=True,
                )

            for tile_idx, tile_cluster_ids in enumerate(tiles):
                # 重置 valid_lengths 为 0，确保每个 tile 独立处理
                self.valid_lengths_dict[device].zero_()

                retrieval_timing = self._run_retrieval_tile(
                    layer_idx,
                    device,
                    tile_cluster_ids,
                    profile_decode=profile,
                )

                if retrieval_timing is not None:
                    lookup_ms += retrieval_timing["lookup_ms"]
                    h2d_ms += retrieval_timing["h2d_ms"]
                    gather_ms += retrieval_timing["gather_ms"]

                if debug_decode and layer_idx == 0:
                    valid_lengths = self.valid_lengths_dict[device]
                    max_len = int(valid_lengths.max().item()) if valid_lengths.numel() else 0
                    min_len = int(valid_lengths.min().item()) if valid_lengths.numel() else 0
                    mean_len = float(valid_lengths.float().mean().item()) if valid_lengths.numel() else 0.0
                    tile_size = int((tile_cluster_ids != self.padding_cluster_id).sum().item())
                    print(
                        "[IMI DECODE]"
                        f" step={self.decode_debug_step} layer={layer_idx} tile={tile_idx} tile_selected={tile_size}"
                        f" valid_lengths[min,max,mean]=[{min_len},{max_len},{mean_len:.1f}]",
                        flush=True,
                    )

                self._append_delta_to_execution(layer_idx, device, tile_cluster_ids)

                if profile:
                    torch.cuda.synchronize(device)
                    attn_start = time.perf_counter()

                acc_out, acc_lse = weighted_flash_decoding(
                    query_view,
                    self.execution_buffer_keys_dict[device],
                    self.execution_buffer_values_dict[device],
                    previous_out=acc_out,
                    previous_lse=acc_lse,
                    cache_seqlens=self.valid_lengths_dict[device],
                    return_softmax_lse=True,
                )

                if profile:
                    torch.cuda.synchronize(device)
                    attn_ms += (time.perf_counter() - attn_start) * 1000.0
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
            torch.cuda.synchronize(device)
            end_time = time.perf_counter()

            search_ms = (sim_time - start_time) * 1000.0 + (select_time - sim_time) * 1000.0
            gather_total_ms = lookup_ms + h2d_ms + gather_ms
            total_ms = (end_time - start_time) * 1000.0

            self._profile_step_breakdown["search_ms"] += float(search_ms)
            self._profile_step_breakdown["gather_ms"] += float(gather_total_ms)
            self._profile_step_breakdown["attn_ms"] += float(attn_ms)
            self._profile_step_breakdown["total_ms"] += float(total_ms)

            if layer_idx == self.layer_num - 1:
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

        return acc_out.view(self.batch_size, 1, self.num_heads, self.head_dim)
