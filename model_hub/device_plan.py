from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class DevicePlan:
    mode: str
    root_device: str
    visible_devices: list[str]
    layer_to_device: dict[str, str]
    device_to_layers: dict[str, list[int]]
    boundary_transfers: list[tuple[int, str, str]]
    per_device_budget_bytes: dict[str, int] = field(default_factory=dict)
    per_device_weight_bytes: dict[str, int] = field(default_factory=dict)
    per_device_kv_bytes: dict[str, int] = field(default_factory=dict)
    per_device_runtime_bytes: dict[str, int] = field(default_factory=dict)
    weight_loading: str = "hf"

    def log_summary(self, prefix: str = "[device-plan]") -> None:
        layers = {
            dev: f"{items[0]}-{items[-1]}" if items else ""
            for dev, items in self.device_to_layers.items()
        }
        print(
            f"{prefix} mode={self.mode} root={self.root_device} "
            f"weight_loading={self.weight_loading} visible={self.visible_devices} "
            f"layers={layers} "
            f"boundaries={self.boundary_transfers}",
            flush=True,
        )


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _cuda_devices(requested: str) -> list[str]:
    if requested != "auto":
        return [requested]
    count = torch.cuda.device_count()
    if count <= 0:
        raise RuntimeError("--device auto requires at least one CUDA device")
    return [f"cuda:{idx}" for idx in range(count)]


def _device_total_bytes(device: str) -> int:
    if not str(device).startswith("cuda"):
        return 0
    return int(torch.cuda.get_device_properties(torch.device(device)).total_memory)


def _estimate_layer_weight_bytes(config: Any, dtype: torch.dtype) -> int:
    runtime_itemsize = _dtype_size(dtype)
    fp8_linear = _is_compressed_tensors_fp8_config(config)
    linear_itemsize = 1 if fp8_linear else runtime_itemsize
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    intermediate = int(config.intermediate_size)
    q = heads * head_dim * hidden
    k = kv_heads * head_dim * hidden
    v = kv_heads * head_dim * hidden
    o = hidden * heads * head_dim
    gate = intermediate * hidden
    up = intermediate * hidden
    down = hidden * intermediate
    norms = 2 * hidden
    bias = 0
    if bool(getattr(config, "attention_bias", False)) or str(getattr(config, "model_type", "")).lower() == "qwen2":
        bias += (heads + 2 * kv_heads) * head_dim
    linear_params = q + k + v + o + gate + up + down
    if fp8_linear:
        scale_params = heads * head_dim + 2 * kv_heads * head_dim + heads * head_dim + 3 * intermediate
        return int(
            (
                linear_params * linear_itemsize
                + scale_params * 4
                + (norms + bias) * runtime_itemsize
            )
            * 1.03
        )
    # Keep a small overhead for concatenated qkv/gate_up tensors and allocator rounding.
    return int((linear_params + norms + bias) * runtime_itemsize * 1.03)


def _is_compressed_tensors_fp8_config(config: Any) -> bool:
    quant = getattr(config, "quantization_config", None) or {}
    if not isinstance(quant, dict):
        return False
    if str(quant.get("quant_method", "")).lower() != "compressed-tensors":
        return False
    if str(quant.get("format", "")).lower() != "float-quantized":
        return False
    for group in (quant.get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        if str(weights.get("type", "")).lower() == "float" and int(weights.get("num_bits", 0) or 0) == 8:
            return True
    return False


def _estimate_shared_weight_bytes(config: Any, dtype: torch.dtype) -> int:
    itemsize = _dtype_size(dtype)
    vocab = int(config.vocab_size)
    hidden = int(config.hidden_size)
    tied = bool(getattr(config, "tie_word_embeddings", False))
    total = vocab * hidden
    if not tied:
        total += vocab * hidden
    total += hidden
    return int(total * itemsize * 1.03)


def _contiguous_layer_plan(
    *,
    num_layers: int,
    devices: list[str],
    per_layer_bytes: int,
    shared_bytes: int,
    utilization: float,
) -> tuple[dict[str, str], dict[str, list[int]], dict[str, int], dict[str, int]]:
    budgets = {
        dev: int(_device_total_bytes(dev) * utilization)
        for dev in devices
    }
    weight_bytes = {dev: 0 for dev in devices}
    layer_to_device: dict[str, str] = {}
    device_to_layers = {dev: [] for dev in devices}

    # The root device owns embedding/lm_head in the first implementation.
    weight_bytes[devices[0]] += shared_bytes
    dev_idx = 0
    for layer_idx in range(num_layers):
        dev = devices[dev_idx]
        would_use = weight_bytes[dev] + per_layer_bytes
        if (
            device_to_layers[dev]
            and dev_idx + 1 < len(devices)
            and would_use > budgets[dev]
        ):
            dev_idx += 1
            dev = devices[dev_idx]
            would_use = weight_bytes[dev] + per_layer_bytes
        if would_use > budgets[dev]:
            raise RuntimeError(
                "Unable to place model layer within the configured GPU memory budget: "
                f"layer={layer_idx} device={dev} required={would_use / 1024 ** 3:.2f}GiB "
                f"budget={budgets[dev] / 1024 ** 3:.2f}GiB"
            )
        layer_to_device[str(layer_idx)] = dev
        device_to_layers[dev].append(layer_idx)
        weight_bytes[dev] = would_use

    return layer_to_device, device_to_layers, budgets, weight_bytes


def _balanced_contiguous_layer_plan(
    *,
    num_layers: int,
    devices: list[str],
    per_layer_bytes: int,
    shared_bytes: int,
    utilization: float,
) -> tuple[dict[str, str], dict[str, list[int]], dict[str, int], dict[str, int]]:
    budgets = {
        dev: int(_device_total_bytes(dev) * utilization)
        for dev in devices
    }
    weight_bytes = {dev: 0 for dev in devices}
    layer_to_device: dict[str, str] = {}
    device_to_layers = {dev: [] for dev in devices}
    weight_bytes[devices[0]] += shared_bytes

    base_layers = num_layers // len(devices)
    extra_layers = num_layers % len(devices)
    layer_idx = 0
    for dev_idx, dev in enumerate(devices):
        block = base_layers + (1 if dev_idx < extra_layers else 0)
        for _ in range(block):
            would_use = weight_bytes[dev] + per_layer_bytes
            if would_use > budgets[dev]:
                raise RuntimeError(
                    "Unable to place explicit layer-parallel block within the configured GPU memory budget: "
                    f"layer={layer_idx} device={dev} required={would_use / 1024 ** 3:.2f}GiB "
                    f"budget={budgets[dev] / 1024 ** 3:.2f}GiB"
                )
            layer_to_device[str(layer_idx)] = dev
            device_to_layers[dev].append(layer_idx)
            weight_bytes[dev] = would_use
            layer_idx += 1

    return layer_to_device, device_to_layers, budgets, weight_bytes


def build_device_plan(
    *,
    config: Any,
    dtype: torch.dtype,
    requested_device: str,
    parallel_mode: str = "auto",
    weight_loading: str = "hf",
    gpu_memory_utilization: float = 0.90,
) -> DevicePlan:
    devices = _cuda_devices(requested_device)
    if parallel_mode == "single" and requested_device == "auto":
        devices = [devices[0]]
    if len(devices) == 1:
        layer_to_device = {str(i): devices[0] for i in range(int(config.num_hidden_layers))}
        device_to_layers = {devices[0]: list(range(int(config.num_hidden_layers)))}
        return DevicePlan(
            mode="single",
            root_device=devices[0],
            visible_devices=devices,
            layer_to_device=layer_to_device,
            device_to_layers=device_to_layers,
            boundary_transfers=[],
            per_device_budget_bytes={devices[0]: int(_device_total_bytes(devices[0]) * gpu_memory_utilization)},
            per_device_weight_bytes={devices[0]: _estimate_shared_weight_bytes(config, dtype) + _estimate_layer_weight_bytes(config, dtype) * int(config.num_hidden_layers)},
            weight_loading=weight_loading,
        )

    if parallel_mode == "tp":
        raise NotImplementedError(
            "Tensor parallel execution is not implemented for AdaptiveIMI yet. "
            "Use --parallel-mode layer for whole-layer placement."
        )
    if parallel_mode not in ("auto", "layer"):
        raise ValueError(f"Unsupported parallel_mode={parallel_mode!r}")

    num_layers = int(config.num_hidden_layers)
    per_layer = _estimate_layer_weight_bytes(config, dtype)
    shared = _estimate_shared_weight_bytes(config, dtype)
    planner = _balanced_contiguous_layer_plan if parallel_mode == "layer" else _contiguous_layer_plan
    layer_to_device, device_to_layers, budgets, weight_bytes = planner(
        num_layers=num_layers,
        devices=devices,
        per_layer_bytes=per_layer,
        shared_bytes=shared,
        utilization=float(gpu_memory_utilization),
    )
    boundaries = []
    for layer_idx in range(num_layers - 1):
        src = layer_to_device[str(layer_idx)]
        dst = layer_to_device[str(layer_idx + 1)]
        if src != dst:
            boundaries.append((layer_idx, src, dst))

    return DevicePlan(
        mode="layer_parallel",
        root_device=devices[0],
        visible_devices=devices,
        layer_to_device=layer_to_device,
        device_to_layers=device_to_layers,
        boundary_transfers=boundaries,
        per_device_budget_bytes=budgets,
        per_device_weight_bytes=weight_bytes,
        weight_loading=weight_loading,
    )
