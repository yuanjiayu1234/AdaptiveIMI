from __future__ import annotations

from dataclasses import dataclass, field
import ctypes
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional
import json
import os
import sys
import time

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

    try:
        cuda_runtime_spec = importlib.util.find_spec("nvidia.cuda_runtime")
    except ModuleNotFoundError:
        cuda_runtime_spec = None
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


def get_imi_kernels():
    try:
        from library.AdaptiveIMI.cpp_extensions import gpu_cluster_manager_cpp as imi_gpu_kernels

        return imi_gpu_kernels
    except Exception as exc:  # pragma: no cover - extension availability is environment-dependent
        raise RuntimeError(
            "gpu_cluster_manager_cpp is required; PyTorch kernel fallback has been removed. "
            "Build library/AdaptiveIMI/cpp_extensions before running AdaptiveIMI."
        ) from exc


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
    subspace_parts: int = 8
    runtime_config: IMIRuntimeConfig = field(default_factory=IMIRuntimeConfig)
    enable_direct_write: bool = True
    learned_partition: Optional[Dict[str, object]] = None

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
            runtime_config = {
                "pipeline": dict(self.runtime_config.pipeline),
                "kmeans": dict(self.runtime_config.kmeans),
            }
            if self.learned_partition is not None:
                runtime_config["learned_partition"] = dict(self.learned_partition)
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
                runtime_config=runtime_config,
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
        worker_threads = max(1, int(worker_threads))
        self.runtime_config.pipeline["worker_threads"] = worker_threads
        if self._pipeline is not None:
            self._pipeline.set_worker_threads(worker_threads)

    def wait_transfers_done(self, timeout_sec: float = 60.0) -> None:
        pipeline = self._get_pipeline()
        deadline = time.monotonic() + float(timeout_sec)
        failed = _pipeline_cpp.PipelineStage.Failed
        while True:
            if pipeline.current_stage() == failed:
                raise RuntimeError(pipeline.error_message() or "IMI pipeline failed during transfer")
            if pipeline.transfers_finished():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("IMI pipeline transfer did not finish in time")
            time.sleep(0.001)

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

    def begin_index_stream_batch(self, token_counts: torch.Tensor | list[int], submit_chunk_size: int) -> None:
        counts = torch.as_tensor(token_counts, dtype=torch.int32, device="cpu").view(-1).contiguous()
        if int(counts.numel()) <= 0:
            raise ValueError("IMI batch pipeline requires non-empty token_counts")
        if int(counts.min().item()) <= 0:
            raise ValueError("IMI batch pipeline requires positive token_counts")
        max_tokens = int(counts.max().item())
        if max_tokens > self.max_tokens:
            raise ValueError(f"token_count({max_tokens}) exceeds max_tokens({self.max_tokens})")
        if submit_chunk_size <= 0:
            raise ValueError("submit_chunk_size must be positive")

        pipeline = self._get_pipeline()
        if not pipeline.start_chunk_pipeline_batch(counts.tolist(), int(submit_chunk_size)):
            raise RuntimeError("IMI batch pipeline failed to start")
        self._token_count = max_tokens
        self._batch_size = int(counts.numel())

    def submit_index_stream_raw_sequence_chunk(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        chunk_id: int,
        source_token: int,
        token_offset: int,
        chunk_len: int,
        is_last: bool,
    ) -> None:
        if key_states.dim() != 4 or value_states.dim() != 4:
            raise ValueError("raw key_states/value_states must be [1, seq_len, kv_heads, head_dim]")
        if key_states.size(0) != 1 or value_states.size(0) != 1:
            raise ValueError("IMI pipeline supports batch_size=1")
        if key_states.size() != value_states.size():
            raise ValueError("raw key_states and value_states must share the same shape")
        if key_states.size(2) != self.kv_heads:
            raise ValueError("kv_heads mismatch in raw IMI pipeline chunk")
        if source_token < 0 or chunk_len <= 0 or source_token + chunk_len > int(key_states.size(1)):
            raise ValueError("invalid raw source token range")
        if token_offset < 0:
            raise ValueError("token_offset must be non-negative")

        pipeline = self._get_pipeline()
        if not pipeline.submit_raw_sequence_chunk(
            key_states,
            value_states,
            chunk_id,
            source_token,
            token_offset,
            chunk_len,
            is_last,
        ):
            raise RuntimeError("IMI pipeline failed to submit raw sequence chunk")

    def submit_index_stream_raw_sequence_batch_chunk(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        chunk_id: int,
        start_bdx: int,
        source_token: int,
        token_offset: int,
        chunk_len: int,
        is_last: bool,
    ) -> None:
        if key_states.dim() != 4 or value_states.dim() != 4:
            raise ValueError("raw key_states/value_states must be [bsz, seq_len, kv_heads, head_dim]")
        if key_states.size() != value_states.size():
            raise ValueError("raw key_states and value_states must share the same shape")
        if key_states.size(2) != self.kv_heads:
            raise ValueError("kv_heads mismatch in raw IMI batch pipeline chunk")
        if source_token < 0 or chunk_len <= 0 or source_token + chunk_len > int(key_states.size(1)):
            raise ValueError("invalid raw source token range")
        if token_offset < 0:
            raise ValueError("token_offset must be non-negative")

        pipeline = self._get_pipeline()
        if not pipeline.submit_raw_sequence_batch_chunk(
            key_states,
            value_states,
            int(chunk_id),
            int(start_bdx),
            int(source_token),
            int(token_offset),
            int(chunk_len),
            bool(is_last),
        ):
            raise RuntimeError("IMI batch pipeline failed to submit raw sequence chunk")

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
            error = pipeline.error_message()
            if error:
                raise RuntimeError(f"IMI pipeline failed: {error}")
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
