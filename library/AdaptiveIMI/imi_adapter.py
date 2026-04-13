from __future__ import annotations

from dataclasses import dataclass, field
import ctypes
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import os
import sys

import torch


_CPP_EXT_DIR = Path(__file__).resolve().parent / "cpp_extensions"
if _CPP_EXT_DIR.exists():
    cpp_path = str(_CPP_EXT_DIR)
    if cpp_path not in sys.path:
        sys.path.insert(0, cpp_path)


def _load_shared_library(lib_path: Path) -> bool:
    if not lib_path.exists():
        return False
    try:
        ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def _preload_extension_dependencies() -> None:
    candidate_dirs = []

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib_dir.exists():
        candidate_dirs.append(torch_lib_dir)

    cuda_runtime_spec = importlib.util.find_spec("nvidia.cuda_runtime")
    if cuda_runtime_spec and cuda_runtime_spec.submodule_search_locations:
        for location in cuda_runtime_spec.submodule_search_locations:
            runtime_dir = Path(location) / "lib"
            if runtime_dir.exists():
                candidate_dirs.append(runtime_dir)

    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home:
        for suffix in ("lib64", "targets/x86_64-linux/lib"):
            runtime_dir = Path(cuda_home) / suffix
            if runtime_dir.exists():
                candidate_dirs.append(runtime_dir)

    seen = set()
    unique_dirs = []
    for directory in candidate_dirs:
        resolved = directory.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_dirs.append(resolved)

    preload_order = [
        "libcudart.so.12",
        "libcudart.so.11.0",
        "libc10.so",
        "libtorch.so",
        "libtorch_cpu.so",
        "libtorch_python.so",
        "libc10_cuda.so",
    ]
    for lib_name in preload_order:
        for directory in unique_dirs:
            if _load_shared_library(directory / lib_name):
                break


_preload_extension_dependencies()

try:
    from library.AdaptiveIMI.cpp_extensions import ultra_layer_pipeline_cpp as _pipeline_cpp
except ImportError as exc:  # pragma: no cover - handled at runtime
    _pipeline_cpp = None
    _pipeline_import_error = exc
else:
    _pipeline_import_error = None


class _TorchIMIKernels:
    @staticmethod
    def fused_query_group_similarities(
        query_grouped: torch.Tensor,
        centroids: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query_grouped.dim() != 3 or centroids.dim() != 3:
            raise ValueError("query_grouped and centroids must be 3D tensors")
        if query_grouped.size(0) != centroids.size(0):
            raise ValueError("kv_heads mismatch between query_grouped and centroids")
        if query_grouped.size(2) != centroids.size(2):
            raise ValueError("head_dim mismatch between query_grouped and centroids")

        aggregated = query_grouped.float().sum(dim=1)
        similarities = torch.einsum("hd,hnd->hn", aggregated, centroids.float())

        if out is None:
            return similarities

        kv_heads, n_clusters = similarities.shape
        out_slice = out.narrow(0, 0, kv_heads).narrow(1, 0, n_clusters)
        out_slice.copy_(similarities)
        return out_slice


def get_imi_kernels():
    try:
        from library.AdaptiveIMI.cpp_extensions import gpu_cluster_manager_cpp as imi_gpu_kernels

        return imi_gpu_kernels
    except Exception:  # pragma: no cover - fallback when kernels unavailable
        return _TorchIMIKernels()


@dataclass
class IMIRuntimeConfig:
    pipeline: Dict[str, object] = field(default_factory=dict)
    kmeans: Dict[str, object] = field(default_factory=dict)


@dataclass
class IMIPipeline:
    layer_idx: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device_id: int
    max_tokens: int
    subspace_parts: int = 2
    runtime_config: IMIRuntimeConfig = field(default_factory=IMIRuntimeConfig)
    enable_direct_write: bool = True

    def __post_init__(self) -> None:
        self._pipeline = None
        self._batch_allocate_cpu_buffer_callback = None
        self._last_pipeline_stats: Optional[Dict[str, object]] = None

    def _get_pipeline(self):
        if _pipeline_cpp is None:
            raise ImportError(
                "ultra_layer_pipeline_cpp is required for IMI indexing"
            ) from _pipeline_import_error
        if self._pipeline is None:
            self._pipeline = _pipeline_cpp.LayerPipeline(
                layer_idx=self.layer_idx,
                kv_heads=self.kv_heads,
                max_tokens=self.max_tokens,
                dim=self.head_dim,
                device_id=self.device_id,
                kv_dtype=self.dtype,
                cache_manager_ptr=None,
                enable_direct_write=self.enable_direct_write,
                subspace_parts=self.subspace_parts,
                runtime_config={
                    "pipeline": dict(self.runtime_config.pipeline),
                    "kmeans": dict(self.runtime_config.kmeans),
                },
            )
            if self._batch_allocate_cpu_buffer_callback is not None:
                self._pipeline.set_batch_allocate_cpu_buffer_callback(
                    self._batch_allocate_cpu_buffer_callback
                )
        return self._pipeline

    def set_batch_allocate_cpu_buffer_callback(self, callback) -> None:
        self._batch_allocate_cpu_buffer_callback = callback
        if self._pipeline is not None:
            self._pipeline.set_batch_allocate_cpu_buffer_callback(callback)

    def cancel_pipeline(self) -> None:
        pipeline = self._pipeline
        if pipeline is None:
            return
        pipeline.cancel_pipeline()

    def close(self) -> None:
        pipeline = self._pipeline
        if pipeline is None:
            return
        try:
            pipeline.cancel_pipeline()
        finally:
            self._batch_allocate_cpu_buffer_callback = None
            self._pipeline = None

    def set_worker_threads(self, worker_threads: int) -> None:
        pipeline = self._get_pipeline()
        pipeline.set_worker_threads(int(worker_threads))

    def build_index(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        list_keys: torch.Tensor,
        list_values: torch.Tensor,
    ) -> List[Dict[str, object]]:
        self.start_index(keys, values)
        return self.finish_index(list_keys, list_values)

    def start_index(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.dim() != 4 or values.dim() != 4:
            raise ValueError("keys/values must be [1, kv_heads, tokens, head_dim]")
        if keys.size(0) != 1 or values.size(0) != 1:
            raise ValueError("IMI pipeline supports batch_size=1")
        if keys.size() != values.size():
            raise ValueError("keys and values must share the same shape")
        if keys.size(1) != self.kv_heads:
            raise ValueError("kv_heads mismatch in IMI pipeline")

        token_count = int(keys.size(2))
        if token_count <= 0:
            raise ValueError("IMI pipeline requires non-empty middle tokens")
        if token_count > self.max_tokens:
            raise ValueError(
                f"token_count({token_count}) exceeds max_tokens({self.max_tokens})"
            )

        self.begin_index_stream(token_count, token_count)
        self.submit_index_stream_chunk(
            keys,
            values,
            chunk_id=0,
            token_offset=0,
            is_last=True,
        )

        self._token_count = token_count

    def begin_index_stream(self, token_count: int, submit_chunk_size: int) -> None:
        if token_count <= 0:
            raise ValueError("IMI pipeline requires non-empty middle tokens")
        if token_count > self.max_tokens:
            raise ValueError(
                f"token_count({token_count}) exceeds max_tokens({self.max_tokens})"
            )
        if submit_chunk_size <= 0:
            raise ValueError("submit_chunk_size must be positive")

        pipeline = self._get_pipeline()
        if not pipeline.start_chunk_pipeline(token_count, submit_chunk_size):
            raise RuntimeError("IMI pipeline failed to start")
        self._token_count = token_count

    def submit_index_stream_chunk(
        self,
        chunk_keys: torch.Tensor,
        chunk_values: torch.Tensor,
        chunk_id: int,
        token_offset: int,
        is_last: bool,
    ) -> None:
        if chunk_keys.dim() != 4 or chunk_values.dim() != 4:
            raise ValueError("chunk keys/values must be [1, kv_heads, tokens, head_dim]")
        if chunk_keys.size(0) != 1 or chunk_values.size(0) != 1:
            raise ValueError("IMI pipeline supports batch_size=1")
        if chunk_keys.size() != chunk_values.size():
            raise ValueError("chunk keys and values must share the same shape")
        if chunk_keys.size(1) != self.kv_heads:
            raise ValueError("kv_heads mismatch in IMI pipeline")
        if chunk_id < 0:
            raise ValueError("chunk_id must be non-negative")
        if token_offset < 0:
            raise ValueError("token_offset must be non-negative")

        pipeline = self._get_pipeline()
        if not pipeline.submit_chunk(
            chunk_keys,
            chunk_values,
            chunk_id,
            token_offset,
            is_last,
        ):
            raise RuntimeError("IMI pipeline failed to submit chunk")


    def finish_index(
        self,
        list_keys: torch.Tensor,
        list_values: torch.Tensor,
    ) -> List[Dict[str, object]]:
        token_count = getattr(self, "_token_count", None)
        if token_count is None:
            raise RuntimeError("IMI pipeline finish called before start")

        pipeline = self._get_pipeline()
        if not pipeline.wait_ready(timeout_sec=max(60.0, token_count / 1024.0 * 5.0)):
            raise RuntimeError("IMI pipeline did not finish in time")

        stats = pipeline.get_pipeline_stats()
        self._last_pipeline_stats = dict(stats)
        if os.getenv("IMI_DEBUG_PIPELINE_STATS", "0") == "1":
            stats_payload = {
                "tag": "IMI_STATS",
                "layer_idx": int(self.layer_idx),
                "subspace_parts": int(self.subspace_parts),
                "token_count": int(token_count),
                "d2h_ms": round(float(stats.get("d2h_ms", 0.0)), 3),
                "cpu_copy_ms": round(float(stats.get("cpu_copy_ms", 0.0)), 3),
                "kmeans_ms": round(float(stats.get("kmeans_ms", 0.0)), 3),
                "kmeans_cpu_time_ms": round(float(stats.get("kmeans_cpu_time_ms", 0.0)), 3),
                "kmeans_cpu_util_cores": round(float(stats.get("kmeans_cpu_util_cores", 0.0)), 3),
                "reorganize_ms": round(float(stats.get("reorganize_ms", 0.0)), 3),
                "write_ms": round(float(stats.get("write_ms", 0.0)), 3),
                "kmeans_gate_wait_ms": int(stats.get("kmeans_gate_wait_ms", 0)),
                "total_chunks": int(stats.get("total_chunks", 0)),
                "total_tokens": int(stats.get("total_tokens", 0)),
            }
            print(json.dumps(stats_payload, ensure_ascii=False), flush=True)

        metadata = pipeline.get_metadata_result()
        if os.getenv("IMI_DEBUG_INDEX_METADATA", "0") == "1":
            cluster_counts = metadata.get("cluster_counts")
            head_indices = metadata.get("head_indices")
            if cluster_counts is not None and head_indices is not None:
                cc = cluster_counts.to(dtype=torch.int64, device="cpu")
                hi = head_indices.to(dtype=torch.int64, device="cpu")
                nonzero_heads = int((cc > 0).sum().item())
                total_clusters = int(cc.sum().item())
                max_clusters = int(cc.max().item()) if cc.numel() else 0
                min_clusters = int(cc.min().item()) if cc.numel() else 0
                sample_counts = cc[: min(8, cc.numel())].tolist()
                sample_heads = hi[: min(8, hi.numel())].tolist()
                print(
                    json.dumps({
                        "tag": "IMI_META_FINISH",
                        "layer_idx": int(self.layer_idx),
                        "nonzero_heads": nonzero_heads,
                        "total_clusters": total_clusters,
                        "min_clusters": min_clusters,
                        "max_clusters": max_clusters,
                        "sample_heads": sample_heads,
                        "sample_cluster_counts": sample_counts,
                    }, ensure_ascii=False),
                    flush=True,
                )
        if not self.enable_direct_write:
            reorganized = pipeline.get_reorganize_results()
            for head_idx, head_reorg in enumerate(reorganized):
                head_keys = head_reorg["reorganized_keys"]
                head_values = head_reorg["reorganized_values"]
                head_tokens = int(head_keys.size(0))
                list_keys[0, head_idx, :head_tokens, :].copy_(head_keys)
                list_values[0, head_idx, :head_tokens, :].copy_(head_values)
        return self._build_metadata_from_tensors(metadata)

    def get_last_pipeline_stats(self) -> Optional[Dict[str, object]]:
        if self._last_pipeline_stats is None:
            return None
        return dict(self._last_pipeline_stats)

    def build_index_chunked(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        middle_start: int,
        middle_end: int,
        list_keys: torch.Tensor,
        list_values: torch.Tensor,
        chunk_size: int,
    ) -> List[Dict[str, object]]:
        self.start_index_chunked(
            key_states,
            value_states,
            middle_start,
            middle_end,
            chunk_size,
        )
        return self.finish_index(list_keys, list_values)

    def start_index_chunked(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        middle_start: int,
        middle_end: int,
        chunk_size: int,
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        if key_states.dim() != 4 or value_states.dim() != 4:
            raise ValueError("key_states/value_states must be [1, seq_len, kv_heads, head_dim]")
        if key_states.size(0) != 1 or value_states.size(0) != 1:
            raise ValueError("IMI pipeline supports batch_size=1")
        if key_states.size() != value_states.size():
            raise ValueError("key_states and value_states must share the same shape")
        if key_states.size(2) != self.kv_heads:
            raise ValueError("kv_heads mismatch in IMI pipeline")

        seq_len = int(key_states.size(1))
        if middle_start < 0 or middle_end > seq_len or middle_start >= middle_end:
            raise ValueError("Invalid middle token range for IMI chunked build")

        token_count = middle_end - middle_start
        if token_count > self.max_tokens:
            raise ValueError(
                f"token_count({token_count}) exceeds max_tokens({self.max_tokens})"
            )

        if chunk_size <= 0 or chunk_size >= token_count:
            middle_keys, middle_values = self._prepare_pipeline_chunk(
                key_states,
                value_states,
                middle_start,
                middle_end,
            )
            self.start_index(middle_keys, middle_values)
            return [(middle_keys, middle_values)]

        pipeline = self._get_pipeline()
        self.begin_index_stream(token_count, chunk_size)

        chunk_refs: List[tuple[torch.Tensor, torch.Tensor]] = []
        chunk_id = 0
        token_offset = 0
        for chunk_start in range(middle_start, middle_end, chunk_size):
            chunk_end = min(middle_end, chunk_start + chunk_size)
            chunk_len = chunk_end - chunk_start
            chunk_keys, chunk_values = self._prepare_pipeline_chunk(
                key_states,
                value_states,
                chunk_start,
                chunk_end,
            )
            is_last = chunk_end == middle_end
            self.submit_index_stream_chunk(
                chunk_keys,
                chunk_values,
                chunk_id=chunk_id,
                token_offset=token_offset,
                is_last=is_last,
            )
            chunk_refs.append((chunk_keys, chunk_values))
            chunk_id += 1
            token_offset += chunk_len

        self._token_count = token_count
        return chunk_refs

    def _prepare_pipeline_chunk(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        start: int,
        end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_keys = key_states[:, start:end, :, :].transpose(1, 2).contiguous()
        chunk_values = value_states[:, start:end, :, :].transpose(1, 2).contiguous()
        return chunk_keys, chunk_values

    def _build_metadata_from_tensors(
        self,
        metadata: Dict[str, object],
    ) -> List[Dict[str, object]]:
        head_indices = metadata["head_indices"]
        cluster_sizes = metadata["cluster_sizes"]
        cluster_offsets = metadata["cluster_offsets"]
        centroids = metadata["centroids"]
        cluster_counts = metadata["cluster_counts"]

        structured: List[Dict[str, object]] = []
        num_heads = int(head_indices.numel())
        for row in range(num_heads):
            head_idx = int(head_indices[row].item())
            cluster_count = int(cluster_counts[row].item())
            structured.append(
                {
                    "head_idx": head_idx,
                    "cluster_sizes": cluster_sizes[row, :cluster_count],
                    "cluster_offsets": cluster_offsets[row, : cluster_count + 1],
                    "centroids": centroids[row, :cluster_count],
                }
            )
        return structured
