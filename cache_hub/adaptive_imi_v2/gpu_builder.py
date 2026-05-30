from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import time
from typing import Dict, List, Optional, Sequence

import torch

from cache_hub.cluster_plan import choose_learned_k_values


_HYBRID_8WAY_CUDA_EXT = None


@dataclass
class HybridBuildResult:
    metadata: List[Dict[str, object]]
    search_metadata_gpu: Optional[Dict[str, torch.Tensor]] = None
    base_metadata_gpu: Optional[Dict[str, torch.Tensor]] = None
    stats: Dict[str, float] = field(default_factory=dict)


def _pair_dim_indices(pairs: List[int], head_dim: int, device: torch.device) -> torch.Tensor:
    half = head_dim // 2
    dims: List[int] = []
    for pair in pairs:
        p = int(pair)
        dims.append(p)
        dims.append(p + half)
    return torch.tensor(dims, dtype=torch.long, device=device)


def _compute_k_values(spec: Dict[str, object], token_count: int, target_cluster_size: int) -> List[int]:
    values = choose_learned_k_values(
        spec,
        token_count,
        target_cluster_size,
        default_m=8,
    )
    if len(values) != 8:
        raise ValueError("hybrid 8-way builder requires exactly 8 k_values")
    return values


def _load_hybrid_8way_cuda_extension():
    global _HYBRID_8WAY_CUDA_EXT
    if _HYBRID_8WAY_CUDA_EXT is not None:
        return _HYBRID_8WAY_CUDA_EXT
    try:
        _HYBRID_8WAY_CUDA_EXT = importlib.import_module(
            "library.AdaptiveIMI.cpp_extensions.hybrid_8way_builder_cpp"
        )
    except Exception as exc:
        raise RuntimeError(
            "hybrid_8way_v2 GPU build requires the custom CUDA extension "
            "`library.AdaptiveIMI.cpp_extensions.hybrid_8way_builder_cpp`. "
            "Build library/AdaptiveIMI/cpp_extensions before using GPU index build."
        ) from exc
    return _HYBRID_8WAY_CUDA_EXT


class HybridLayerBuildJob:
    """One layer build job with batch-aware metadata rows.

    Metadata rows are ordered as `batch_idx * kv_heads + head_idx`, matching the
    existing `batch_groups` layout used by retrieval and hot/cold cache manager.
    """

    def __init__(
        self,
        *,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        partition_spec: Dict[str, object],
        build_device: str,
        token_counts: Optional[Sequence[int] | torch.Tensor] = None,
        target_cluster_size: int = 16,
        max_iters: int = 15,
        source_ready_event: Optional[torch.cuda.Event] = None,
        auto_start_raw_d2h: bool = True,
        auto_start_gpu_build: bool = True,
    ) -> None:
        if keys.dim() != 4 or values.dim() != 4:
            raise ValueError("keys/values must be [batch, kv_heads, tokens, head_dim]")
        if keys.size() != values.size():
            raise ValueError("keys and values must share the same shape")
        if build_device != "gpu":
            raise ValueError("HybridLayerBuildJob only supports GPU metadata build; CPU layers use the C++ pipeline")
        if not keys.is_cuda:
            raise ValueError("GPU build requires CUDA keys")

        self.layer_idx = int(layer_idx)
        self.partition_spec = partition_spec
        self.build_device = build_device
        self.target_cluster_size = int(target_cluster_size)
        self.max_iters = int(max_iters)
        self.batch_size = int(keys.size(0))
        self.kv_heads = int(keys.size(1))
        self.max_token_count = int(keys.size(2))
        self.head_dim = int(keys.size(3))
        if token_counts is None:
            self.token_counts = torch.full((self.batch_size,), self.max_token_count, dtype=torch.int32)
        else:
            self.token_counts = torch.as_tensor(token_counts, dtype=torch.int32, device="cpu").view(-1).contiguous()
            if int(self.token_counts.numel()) != self.batch_size:
                raise ValueError("token_counts must have one entry per batch item")
            if int(self.token_counts.min().item()) < 0 or int(self.token_counts.max().item()) > self.max_token_count:
                raise ValueError("token_counts entries must be in [0, max_tokens]")
        self.source_device = keys.device
        self._keys_ref: Optional[torch.Tensor] = keys
        self._values_ref: Optional[torch.Tensor] = values
        self.source_ready_event = source_ready_event

        self.copy_event: Optional[torch.cuda.Event] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.build_start: Optional[torch.cuda.Event] = None
        self.build_end: Optional[torch.cuda.Event] = None
        self.build_stream: Optional[torch.cuda.Stream] = None
        self._metadata_device: Optional[Dict[str, torch.Tensor]] = None
        self._metadata_cpu: Optional[Dict[str, torch.Tensor]] = None
        self._metadata_cpu_event: Optional[torch.cuda.Event] = None
        self._metadata_cpu_stream: Optional[torch.cuda.Stream] = None
        self._metadata_cpu_started = False
        self._build_ready = False
        self._raw_d2h_started = False
        self._gpu_build_started = False
        self._gpu_build_ms = 0.0
        self._raw_d2h_wait_ms = 0.0
        self._metadata_d2h_ms = 0.0
        self._gpu_sources_released = False

        self.keys_cpu = None
        self.values_cpu = None
        if auto_start_raw_d2h:
            self.start_raw_d2h(source_ready_event)

        if auto_start_gpu_build:
            self.start_gpu_metadata_build(source_ready_event)

    def start_raw_d2h(self, source_ready_event: Optional[torch.cuda.Event] = None) -> None:
        if self._raw_d2h_started:
            return
        keys = self._keys_ref
        values = self._values_ref
        if keys is None or values is None:
            raise RuntimeError("hybrid raw KV source tensors were released before D2H started")
        event = source_ready_event if source_ready_event is not None else self.source_ready_event
        device_index = int(keys.device.index or 0)
        self.copy_stream = torch.cuda.Stream(device=device_index)
        self.copy_event = torch.cuda.Event()
        self.keys_cpu = torch.empty(keys.shape, dtype=keys.dtype, device="cpu", pin_memory=True)
        self.values_cpu = torch.empty(values.shape, dtype=values.dtype, device="cpu", pin_memory=True)
        with torch.cuda.stream(self.copy_stream):
            if event is not None:
                self.copy_stream.wait_event(event)
            else:
                self.copy_stream.wait_stream(torch.cuda.current_stream(device_index))
            self.keys_cpu.copy_(keys, non_blocking=True)
            self.values_cpu.copy_(values, non_blocking=True)
            self.copy_event.record()
        self._raw_d2h_started = True

    def start_gpu_metadata_build(self, source_ready_event: Optional[torch.cuda.Event] = None) -> None:
        if self.build_device != "gpu" or self._gpu_build_started:
            return
        keys = self._keys_ref
        if keys is None:
            raise RuntimeError("hybrid GPU source tensor was released before metadata build started")
        if not keys.is_cuda:
            raise RuntimeError("GPU metadata build requires CUDA keys")

        event = source_ready_event if source_ready_event is not None else self.source_ready_event
        device_index = int(keys.device.index or 0)
        self.build_start = torch.cuda.Event(enable_timing=True)
        self.build_end = torch.cuda.Event(enable_timing=True)
        priority = 0
        get_priority_range = getattr(torch.cuda, "get_stream_priority_range", None)
        if callable(get_priority_range):
            low_priority, _high_priority = get_priority_range()
            priority = int(low_priority)
        self.build_stream = torch.cuda.Stream(device=device_index, priority=priority)
        with torch.cuda.stream(self.build_stream):
            if event is not None:
                self.build_stream.wait_event(event)
            else:
                self.build_stream.wait_stream(torch.cuda.current_stream(device_index))
            self.build_start.record()
            self._metadata_device = self._build_metadata(keys)
            self.build_end.record()
        self._build_ready = True
        self._gpu_build_started = True

    def wait_gpu_sources_consumed(self) -> None:
        if self.build_device != "gpu" or self._gpu_sources_released:
            return
        if not self._raw_d2h_started:
            self.start_raw_d2h()
        if not self._gpu_build_started:
            self.start_gpu_metadata_build()
        self.wait_gpu_metadata_build()
        if self.copy_event is not None:
            copy_wait_start = time.perf_counter()
            self.copy_event.synchronize()
            self._raw_d2h_wait_ms = (time.perf_counter() - copy_wait_start) * 1000.0
        if self.keys_cpu is None or self.values_cpu is None:
            raise RuntimeError("hybrid raw KV D2H did not produce CPU buffers")
        self._keys_ref = None
        self._values_ref = None
        self._gpu_sources_released = True

    def wait_gpu_metadata_build(self) -> None:
        if self.build_device != "gpu":
            return
        if not self._gpu_build_started:
            self.start_gpu_metadata_build()
        if self.build_end is None:
            return
        self.build_end.synchronize()
        self._gpu_build_ms = (
            float(self.build_start.elapsed_time(self.build_end))
            if self.build_start is not None and self.build_end is not None
            else 0.0
        )

    def release_gpu_source_refs(self) -> None:
        if self._gpu_sources_released:
            return
        if self.build_end is not None:
            self.build_end.synchronize()
        if self.copy_event is not None:
            self.copy_event.synchronize()
        self._keys_ref = None
        self._values_ref = None
        self._gpu_sources_released = True

    def _build_metadata(self, keys: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._build_metadata_batched_cuda(keys)

    def _row_token_counts(self, device: torch.device) -> torch.Tensor:
        return self.token_counts.to(device=device, non_blocking=True).repeat_interleave(self.kv_heads)

    def _prepare_partition_tensors(
        self,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        rows = self.batch_size * self.kv_heads
        heads = self.partition_spec["heads"]
        max_pairs = 1
        max_sub_dim = 1
        for head_idx in range(self.kv_heads):
            head_spec = heads[str(head_idx)]
            for pairs in head_spec["groups"]:
                max_pairs = max(max_pairs, len(pairs))
                max_sub_dim = max(max_sub_dim, len(pairs) * 2)

        subspace_dim_ids = torch.zeros((rows, 8, max_sub_dim), dtype=torch.int64, device=device)
        subspace_dim_masks = torch.ones((rows, 8, max_sub_dim), dtype=torch.bool, device=device)
        subspace_pair_ids = torch.full((rows, 8, max_pairs), -1, dtype=torch.int64, device=device)
        subspace_pair_masks = torch.ones((rows, 8, max_pairs), dtype=torch.bool, device=device)

        for batch_idx in range(self.batch_size):
            for head_idx in range(self.kv_heads):
                row_idx = batch_idx * self.kv_heads + head_idx
                head_spec = heads[str(head_idx)]
                groups = head_spec["groups"]
                for subspace_idx, pairs in enumerate(groups):
                    pair_tensor = torch.tensor(pairs, dtype=torch.int64, device=device)
                    dim_tensor = _pair_dim_indices(pairs, self.head_dim, device)
                    subspace_pair_ids[row_idx, subspace_idx, : pair_tensor.numel()].copy_(pair_tensor)
                    subspace_pair_masks[row_idx, subspace_idx, : pair_tensor.numel()] = False
                    subspace_dim_ids[row_idx, subspace_idx, : dim_tensor.numel()].copy_(dim_tensor)
                    subspace_dim_masks[row_idx, subspace_idx, : dim_tensor.numel()] = False

        return (
            subspace_dim_ids,
            subspace_dim_masks,
            subspace_pair_ids,
            subspace_pair_masks,
            max_sub_dim,
            max_pairs,
        )

    def _build_metadata_batched_cuda(self, keys: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, kv_heads, max_token_count, head_dim = keys.shape
        rows = int(batch_size) * int(kv_heads)
        spec = self.partition_spec
        k_values = _compute_k_values(spec, int(max(self.token_counts.max().item(), 1)), self.target_cluster_size)
        max_k = max(k_values)
        packed_code_space = 1
        for value in k_values:
            k_value = max(1, int(value))
            if packed_code_space > ((1 << 64) - 1) // k_value:
                raise ValueError("hybrid 8-way k_values exceed uint64 packed code range")
            packed_code_space *= k_value
        max_compact_clusters = max(1, min(int(packed_code_space), int(max_token_count)))
        ext = _load_hybrid_8way_cuda_extension()

        (
            subspace_dim_ids,
            subspace_dim_masks,
            subspace_pair_ids,
            subspace_pair_masks,
            max_sub_dim,
            max_pairs,
        ) = self._prepare_partition_tensors(device=keys.device)

        row_token_counts = self._row_token_counts(keys.device)
        k_values_tensor = torch.tensor(k_values, dtype=torch.int32, device=keys.device)
        (
            subspace_centroids_cuda,
            subspace_centroid_masks_cuda,
            token_ids_by_cluster,
            compact_codes,
            cluster_sizes,
            cluster_offsets,
            cluster_counts,
            full_centroids,
        ) = ext.run_hybrid_8way_kmeans(
            keys.contiguous(),
            row_token_counts.contiguous(),
            subspace_dim_ids.contiguous(),
            subspace_dim_masks.contiguous(),
            k_values_tensor,
            int(self.max_iters),
            int(max_k),
            int(max_compact_clusters),
        )
        metadata = {
            "token_counts": row_token_counts.to(torch.int32),
            "cluster_counts": cluster_counts.to(torch.int32),
            "cluster_sizes": cluster_sizes.to(torch.int32),
            "cluster_offsets": cluster_offsets.to(torch.int32),
            "centroids": full_centroids.to(torch.float32),
            "token_ids_by_cluster": token_ids_by_cluster.to(torch.int32),
            "compact_codes": compact_codes.to(torch.int32),
            "subspace_centroids": subspace_centroids_cuda.to(torch.float32),
            "subspace_centroid_masks": subspace_centroid_masks_cuda.to(torch.bool),
            "subspace_dims": (~subspace_dim_masks).sum(dim=2).to(torch.int32),
            "subspace_pair_ids": subspace_pair_ids,
            "subspace_pair_masks": subspace_pair_masks,
            "subspace_dim_ids": subspace_dim_ids,
            "subspace_dim_masks": subspace_dim_masks,
            "k_values": k_values_tensor,
        }
        return metadata

    def prepare_cpu_metadata(self) -> None:
        if self._metadata_cpu is not None and self._metadata_cpu_event is None:
            return
        self.start_cpu_metadata_d2h()
        if self._metadata_cpu_event is not None:
            meta_d2h_start = time.perf_counter()
            self._metadata_cpu_event.synchronize()
            self._metadata_d2h_ms = (time.perf_counter() - meta_d2h_start) * 1000.0
            self._metadata_cpu_event = None

    def start_cpu_metadata_d2h(self) -> None:
        if self._metadata_cpu is not None or self._metadata_cpu_started:
            return
        self.wait_gpu_metadata_build()
        if self._metadata_device is None:
            raise RuntimeError("hybrid build metadata was not produced")
        cpu_metadata_keys = [
            "token_counts",
            "cluster_counts",
            "cluster_sizes",
            "cluster_offsets",
            "token_ids_by_cluster",
        ]
        first_tensor = next(
            (self._metadata_device[key] for key in cpu_metadata_keys if key in self._metadata_device),
            None,
        )
        if first_tensor is None:
            raise RuntimeError("hybrid build metadata did not contain CPU metadata tensors")
        device_index = int(first_tensor.device.index or 0)
        self._metadata_cpu_stream = torch.cuda.Stream(device=device_index)
        self._metadata_cpu_event = torch.cuda.Event()
        self._metadata_cpu = {
            key: torch.empty(
                self._metadata_device[key].shape,
                dtype=self._metadata_device[key].dtype,
                device="cpu",
                pin_memory=True,
            )
            for key in cpu_metadata_keys
            if key in self._metadata_device
        }
        with torch.cuda.stream(self._metadata_cpu_stream):
            if self.build_end is not None:
                self._metadata_cpu_stream.wait_event(self.build_end)
            for key, dst in self._metadata_cpu.items():
                dst.copy_(self._metadata_device[key], non_blocking=True)
            self._metadata_cpu_event.record()
        self._metadata_cpu_started = True

    def _validate_cpu_metadata(self, metadata_cpu: Dict[str, torch.Tensor]) -> None:
        required = (
            "token_counts",
            "cluster_counts",
            "cluster_sizes",
            "cluster_offsets",
            "token_ids_by_cluster",
        )
        missing = [key for key in required if key not in metadata_cpu]
        if missing:
            raise RuntimeError(f"hybrid build metadata missing CPU tensors: {missing}")

        token_counts = metadata_cpu["token_counts"].view(-1)
        cluster_counts = metadata_cpu["cluster_counts"].view(-1)
        cluster_sizes = metadata_cpu["cluster_sizes"]
        cluster_offsets = metadata_cpu["cluster_offsets"]
        token_ids = metadata_cpu["token_ids_by_cluster"]
        rows = self.batch_size * self.kv_heads
        if int(token_counts.numel()) != rows or int(cluster_counts.numel()) != rows:
            raise RuntimeError("hybrid build metadata row count mismatch")
        if int(cluster_sizes.size(0)) != rows or int(cluster_offsets.size(0)) != rows or int(token_ids.size(0)) != rows:
            raise RuntimeError("hybrid build metadata tensor row mismatch")
        if int(token_ids.size(1)) < self.max_token_count:
            raise RuntimeError("hybrid token_ids_by_cluster is shorter than source tokens")

        max_clusters = int(cluster_sizes.size(1))
        for row_idx in range(rows):
            token_count = int(token_counts[row_idx].item())
            cluster_count = int(cluster_counts[row_idx].item())
            if token_count < 0 or token_count > self.max_token_count:
                raise RuntimeError(f"hybrid metadata row {row_idx} has invalid token_count={token_count}")
            if cluster_count < 0 or cluster_count > max_clusters:
                raise RuntimeError(f"hybrid metadata row {row_idx} has invalid cluster_count={cluster_count}")
            if cluster_count == 0:
                if token_count != 0:
                    raise RuntimeError(f"hybrid metadata row {row_idx} has tokens but no clusters")
                continue
            row_sizes = cluster_sizes[row_idx, :cluster_count]
            row_offsets = cluster_offsets[row_idx, : cluster_count + 1]
            if int(row_offsets[0].item()) != 0:
                raise RuntimeError(f"hybrid metadata row {row_idx} first cluster offset is not zero")
            if int(row_offsets[-1].item()) != token_count:
                raise RuntimeError(
                    f"hybrid metadata row {row_idx} offset tail {int(row_offsets[-1].item())} "
                    f"!= token_count {token_count}"
                )
            if bool((row_sizes < 0).any().item()):
                raise RuntimeError(f"hybrid metadata row {row_idx} has negative cluster size")
            if int(row_sizes.sum().item()) != token_count:
                raise RuntimeError(
                    f"hybrid metadata row {row_idx} cluster size sum {int(row_sizes.sum().item())} "
                    f"!= token_count {token_count}"
                )

    def finish(
        self,
        list_keys: torch.Tensor,
        list_values: torch.Tensor,
        *,
        keep_search_metadata: bool = True,
        on_sources_consumed=None,
    ) -> HybridBuildResult:
        self.wait_gpu_sources_consumed()
        if on_sources_consumed is not None:
            on_sources_consumed()
        gpu_build_ms = float(self._gpu_build_ms)
        raw_d2h_wait_ms = float(self._raw_d2h_wait_ms)

        self.prepare_cpu_metadata()
        if self._metadata_cpu is None or self._metadata_device is None:
            raise RuntimeError("hybrid build metadata was not produced")
        metadata_cpu = self._metadata_cpu
        self._validate_cpu_metadata(metadata_cpu)
        metadata_d2h_ms = float(self._metadata_d2h_ms)

        reorg_start = time.perf_counter()
        ext = _load_hybrid_8way_cuda_extension()
        ext.reorder_raw_kv_by_token_ids(
            self.keys_cpu.contiguous(),
            self.values_cpu.contiguous(),
            metadata_cpu["token_ids_by_cluster"].contiguous(),
            metadata_cpu["token_counts"].contiguous(),
            list_keys,
            list_values,
        )
        reorg_ms = (time.perf_counter() - reorg_start) * 1000.0
        self.keys_cpu = None
        self.values_cpu = None

        structured: List[Dict[str, object]] = []
        cluster_counts = metadata_cpu["cluster_counts"]
        for row_idx in range(int(cluster_counts.numel())):
            cluster_count = int(cluster_counts[row_idx].item())
            structured.append({
                "head_idx": row_idx,
                "batch_idx": row_idx // self.kv_heads,
                "local_head_idx": row_idx % self.kv_heads,
                "token_count": int(metadata_cpu["token_counts"][row_idx].item()),
                "cluster_sizes": metadata_cpu["cluster_sizes"][row_idx, :cluster_count],
                "cluster_offsets": metadata_cpu["cluster_offsets"][row_idx, : cluster_count + 1],
            })

        stats = {
            "build_device": self.build_device,
            "gpu_build_ms": gpu_build_ms,
            "cpu_build_ms": float(getattr(self, "cpu_build_ms", 0.0)),
            "metadata_d2h_ms": metadata_d2h_ms,
            "raw_d2h_wait_ms": raw_d2h_wait_ms,
            "reorganize_ms": reorg_ms,
            "total_tokens": float(int(self.token_counts.sum().item()) * self.kv_heads),
            "max_tokens": float(self.max_token_count),
            "batch_size": float(self.batch_size),
            "batch_groups": float(self.batch_size * self.kv_heads),
        }

        search_gpu = None
        base_gpu = {
            "centroids": self._metadata_device["centroids"],
            "cluster_sizes": self._metadata_device["cluster_sizes"],
            "cluster_counts": self._metadata_device["cluster_counts"],
        }
        if keep_search_metadata:
            active_clusters = max(int(metadata_cpu["cluster_counts"].max().item()), 1)
            search_keys = (
                "subspace_centroids",
                "subspace_centroid_masks",
                "subspace_dim_ids",
                "subspace_dim_masks",
            )
            search_gpu = {key: self._metadata_device[key] for key in search_keys}
            search_gpu["compact_codes"] = self._metadata_device["compact_codes"][:, :active_clusters, :]
        self._metadata_device = None
        self._metadata_cpu = None
        self._keys_ref = None
        self._values_ref = None
        return HybridBuildResult(
            metadata=structured,
            search_metadata_gpu=search_gpu,
            base_metadata_gpu=base_gpu,
            stats=stats,
        )

# Backward-compatible name used by the v2 cache implementation.
HybridGpuLayerBuildJob = HybridLayerBuildJob
