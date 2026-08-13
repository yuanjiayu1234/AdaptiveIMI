from __future__ import annotations

import torch

from cache_hub.adaptive_imi.nvtx_utils import nvtx_range


class HybridMetadataMixin:
    def _install_hybrid_decode_metadata(self, layer_idx: int, metadata_gpu: dict) -> None:
        layer_idx = int(layer_idx)
        if metadata_gpu is None:
            raise RuntimeError(f"hybrid GPU metadata for layer {layer_idx} is not ready")

        centroids_gpu = metadata_gpu.get("centroids")
        cluster_sizes_gpu_src = metadata_gpu.get("cluster_sizes")
        cluster_counts_gpu_src = metadata_gpu.get("cluster_counts")
        if centroids_gpu is None or cluster_sizes_gpu_src is None or cluster_counts_gpu_src is None:
            raise RuntimeError(f"missing GPU decode metadata for hybrid layer {layer_idx}")
        required_centroids = int(centroids_gpu.size(1)) + 1
        if required_centroids > self.n_centroids:
            raise RuntimeError(
                f"hybrid GPU layer {layer_idx} produced {required_centroids - 1} clusters, "
                f"but metadata capacity is {self.n_centroids - 1}; "
                "GPU-only build must not resize decode metadata after allocation"
            )

        if int(centroids_gpu.dim()) != 3:
            raise RuntimeError(f"hybrid layer {layer_idx} centroids must be 3D")
        if int(cluster_sizes_gpu_src.dim()) != 2:
            raise RuntimeError(f"hybrid layer {layer_idx} cluster_sizes must be 2D")
        if int(centroids_gpu.size(0)) != int(self.batch_groups):
            raise RuntimeError(
                f"hybrid layer {layer_idx} centroid rows {int(centroids_gpu.size(0))} "
                f"!= batch_groups {int(self.batch_groups)}"
            )
        if int(cluster_sizes_gpu_src.size(0)) != int(self.batch_groups):
            raise RuntimeError(
                f"hybrid layer {layer_idx} cluster size rows {int(cluster_sizes_gpu_src.size(0))} "
                f"!= batch_groups {int(self.batch_groups)}"
            )
        if int(centroids_gpu.size(2)) != int(self.head_dim):
            raise RuntimeError(
                f"hybrid layer {layer_idx} centroid dim {int(centroids_gpu.size(2))} "
                f"!= head_dim {int(self.head_dim)}"
            )
        if int(cluster_sizes_gpu_src.size(1)) != int(centroids_gpu.size(1)):
            raise RuntimeError(
                f"hybrid layer {layer_idx} cluster_sizes columns {int(cluster_sizes_gpu_src.size(1))} "
                f"!= centroid clusters {int(centroids_gpu.size(1))}"
            )
        if int(cluster_counts_gpu_src.dim()) != 1 or int(cluster_counts_gpu_src.size(0)) != int(self.batch_groups):
            raise RuntimeError(
                f"hybrid layer {layer_idx} cluster_counts shape {tuple(cluster_counts_gpu_src.shape)} "
                f"does not match batch_groups {int(self.batch_groups)}"
            )

        self.centroids_cpu[layer_idx] = None
        self.centroids_mask_cpu[layer_idx] = None
        device = torch.device(self.layer_mapping[str(layer_idx)])
        with nvtx_range(f"imi.prefill.L{layer_idx}.load_metadata.install_decode_metadata"):
            with torch.cuda.device(device):
                self._materialize_hybrid_decode_metadata_gpu(
                    layer_idx,
                    centroids_gpu,
                    cluster_sizes_gpu_src,
                    cluster_counts_gpu_src,
                    device,
                )

    def _materialize_hybrid_decode_metadata_gpu(
        self,
        layer_idx: int,
        centroids_compact: torch.Tensor,
        cluster_sizes_compact: torch.Tensor,
        cluster_counts: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Install compact GPU build metadata into the full decode layout."""
        with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.normalize_inputs"):
            if centroids_compact.device != device:
                centroids_compact = centroids_compact.to(device=device, non_blocking=True).contiguous()
            if cluster_sizes_compact.device != device:
                cluster_sizes_compact = cluster_sizes_compact.to(device=device, non_blocking=True).contiguous()
            if cluster_counts.device != device:
                cluster_counts = cluster_counts.to(device=device, non_blocking=True).contiguous()
            if not centroids_compact.is_contiguous():
                centroids_compact = centroids_compact.contiguous()
            if not cluster_sizes_compact.is_contiguous():
                cluster_sizes_compact = cluster_sizes_compact.contiguous()
            if not cluster_counts.is_contiguous():
                cluster_counts = cluster_counts.contiguous()

        with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.cast_inputs"):
            cluster_sizes_compact = cluster_sizes_compact.to(dtype=torch.int32)
            cluster_counts = cluster_counts.to(dtype=torch.int32)
        compact_clusters = int(cluster_sizes_compact.size(1))

        centroids_full = self.centroids[layer_idx]
        cluster_sizes_full = self.cluster_sizes_gpu[layer_idx]
        centroids_mask_full = self.centroids_mask[layer_idx]
        centroid_shape = (self.batch_groups, self.n_centroids, self.head_dim)
        metadata_shape = (self.batch_groups, self.n_centroids)

        with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.ensure_outputs"):
            if centroids_full is None or tuple(centroids_full.shape) != centroid_shape:
                centroids_full = torch.empty(
                    centroid_shape,
                    dtype=torch.float32,
                    device=device,
                )
            if cluster_sizes_full is None or tuple(cluster_sizes_full.shape) != metadata_shape:
                cluster_sizes_full = torch.empty(
                    metadata_shape,
                    dtype=torch.int32,
                    device=device,
                )
            if centroids_mask_full is None or tuple(centroids_mask_full.shape) != metadata_shape:
                centroids_mask_full = torch.empty(
                    metadata_shape,
                    dtype=torch.bool,
                    device=device,
                )

        materialize_fn = None
        try:
            from cache_hub.adaptive_imi.gpu_builder import _load_hybrid_8way_cuda_extension

            materialize_fn = getattr(_load_hybrid_8way_cuda_extension(), "materialize_hybrid_decode_metadata", None)
        except Exception:
            materialize_fn = None

        if materialize_fn is not None:
            with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.fused_kernel"):
                materialize_fn(
                    centroids_compact,
                    cluster_sizes_compact,
                    cluster_counts,
                    centroids_full,
                    cluster_sizes_full,
                    centroids_mask_full,
                )
        else:
            with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.arange_cluster_ids"):
                cluster_ids = torch.arange(
                    compact_clusters,
                    device=device,
                    dtype=torch.int32,
                ).view(1, -1)
            with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.compare_valid_by_count"):
                valid_by_count = cluster_ids < cluster_counts.view(-1, 1)
            with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.mask_active_sizes"):
                active_sizes = cluster_sizes_compact.masked_fill(~valid_by_count, 0)
            with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.build_active_mask"):
                active_mask = ~valid_by_count | (active_sizes == 0)
            with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.fill_full"):
                with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.zero_cluster_sizes_full"):
                    cluster_sizes_full.zero_()
                with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.fill_centroids_mask_full"):
                    centroids_mask_full.fill_(True)
                with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.copy_centroids"):
                    centroids_full[:, :compact_clusters, :].copy_(centroids_compact)
                with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.copy_cluster_sizes"):
                    cluster_sizes_full[:, :compact_clusters].copy_(active_sizes)
                with nvtx_range(f"imi.prefill.L{layer_idx}.materialize_decode_metadata.copy_centroids_mask"):
                    centroids_mask_full[:, :compact_clusters].copy_(active_mask)

        self.centroids[layer_idx] = centroids_full
        self.cluster_sizes_gpu[layer_idx] = cluster_sizes_full
        self.centroids_mask[layer_idx] = centroids_mask_full

    def _load_hybrid_gpu_layer_metadata(self, layer_idx: int, result) -> None:
        metadata = result.metadata
        base_metadata_gpu = result.base_metadata_gpu
        if base_metadata_gpu is None:
            raise RuntimeError(f"hybrid GPU layer {layer_idx} did not return decode metadata")
        if bool((self.runtime_config.get("cache") or {}).get("preallocate_decode_metadata", False)):
            self._ensure_metadata_buffers(preallocate_gpu_only=True)
        cluster_sizes = self.cluster_sizes_cpu[layer_idx]
        cluster_offsets = self.cluster_offsets_cpu[layer_idx]
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
            self.cluster_sizes_cpu[layer_idx] = cluster_sizes
            self.cluster_offsets_cpu[layer_idx] = cluster_offsets

        with nvtx_range(f"imi.prefill.L{layer_idx}.load_metadata.cpu_fill"):
            cluster_sizes.zero_()
            cluster_offsets.zero_()
            for row_idx, row_meta in enumerate(metadata):
                cluster_count = int(row_meta.get("cluster_sizes").numel())
                if cluster_count <= 0:
                    continue
                if cluster_count >= self.n_centroids:
                    raise RuntimeError(
                        f"hybrid GPU layer {layer_idx} produced {cluster_count} clusters, "
                        f"but metadata capacity is {self.n_centroids}"
                    )
                sizes_tensor = row_meta["cluster_sizes"].to(dtype=torch.int32, device="cpu")
                offsets_tensor = row_meta["cluster_offsets"].to(dtype=torch.int32, device="cpu")
                cluster_sizes[row_idx, :cluster_count].copy_(sizes_tensor)
                cluster_offsets[row_idx, : cluster_count + 1].copy_(offsets_tensor)
                cluster_offsets[row_idx, cluster_count + 1 :] = int(offsets_tensor[-1].item())

        # When decode metadata was preallocated, this GPU buffer was created
        # from the initially empty CPU offsets. Publish the populated offsets
        # before decode retrieval consumes them.
        if self.cluster_offsets_gpu[layer_idx] is not None:
            self.cluster_offsets_gpu[layer_idx].copy_(cluster_offsets, non_blocking=True)

        self.layer_metadata[layer_idx] = None
        with nvtx_range(f"imi.prefill.L{layer_idx}.load_metadata.cpu_stats"):
            self.layer_index_stats[layer_idx] = {
                "non_empty_clusters": int((cluster_sizes > 0).sum().item()),
            }
        with nvtx_range(f"imi.prefill.L{layer_idx}.load_metadata.install_gpu"):
            self._install_hybrid_decode_metadata(layer_idx, base_metadata_gpu)
        if self.cluster_sizes_gpu[layer_idx] is None or self.centroids_mask[layer_idx] is None:
            raise RuntimeError(f"hybrid GPU decode metadata for layer {layer_idx} is missing after install")
        self.layer_ready[layer_idx] = True
