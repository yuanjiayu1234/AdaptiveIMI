import os
import torch


class AdaptiveIMIPrefetchMixin:
    def _warmup_prefetch(self) -> None:
        if self.prefetch_done or not self.prefetch_enabled or self.prefetch_k <= 0:
            return
        rng = torch.Generator()
        if self.prefetch_seed:
            rng.manual_seed(self.prefetch_seed)
        for layer_idx in range(self.layer_num):
            self.ensure_layer_ready(layer_idx)
            cluster_sizes = self.cluster_sizes_cpu[layer_idx]
            if cluster_sizes is None:
                continue

            if self.prefetch_mode != "random":
                # 目前只支持 random warmup，避免 largest 这类与 query 无关且偏置明显的策略。
                raise ValueError(f"Unsupported prefetch mode: {self.prefetch_mode}")

            prefetch_ids = torch.full(
                (self.batch_groups, self.prefetch_k),
                self.padding_cluster_id,
                dtype=torch.int64,
                pin_memory=True,
            )
            for group_idx in range(self.batch_groups):
                non_empty = torch.nonzero(
                    cluster_sizes[group_idx] > 0,
                    as_tuple=False,
                ).flatten()
                if non_empty.numel() == 0:
                    continue
                if non_empty.numel() <= self.prefetch_k:
                    chosen = non_empty
                else:
                    perm = torch.randperm(non_empty.numel(), generator=rng)
                    chosen = non_empty[perm[: self.prefetch_k]]
                prefetch_ids[group_idx, : chosen.numel()] = chosen

            selected = torch.full(
                (self.batch_groups, self.nprobe),
                self.padding_cluster_id,
                dtype=torch.int64,
                pin_memory=True,
            )
            selected[:, : self.prefetch_k].copy_(prefetch_ids)

            tiles = self._build_cluster_tiles(layer_idx, selected)
            if not tiles:
                continue

            device = self.layer_mapping[str(layer_idx)]
            for tile_cluster_ids in tiles[: self.prefetch_tiles_per_layer]:
                self._run_retrieval_tile(layer_idx, device, tile_cluster_ids)

        self.prefetch_done = True
        self.prefetch_enabled = False
