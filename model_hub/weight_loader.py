from __future__ import annotations

import json
import os
from glob import glob
from collections import defaultdict
from typing import Iterable

import torch


class SafetensorsWeightSource:
    """Minimal safetensors streaming reader inspired by nano-vllm's loader.

    It keeps only a parameter-name index in memory. Tensor payloads are read on
    demand and immediately copied into the layer owner device by the caller.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.weight_map = self._build_weight_map(model_path)

    @staticmethod
    def _build_weight_map(model_path: str) -> dict[str, str]:
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"Invalid safetensors index: {index_path}")
            return {
                str(name): os.path.join(model_path, str(filename))
                for name, filename in weight_map.items()
            }

        files = sorted(glob(os.path.join(model_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(
                f"No safetensors files found in {model_path!r}. "
                "Use --weight-loading hf for non-safetensors checkpoints."
            )
        weight_map: dict[str, str] = {}
        from safetensors import safe_open

        for filename in files:
            with safe_open(filename, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_map[str(key)] = filename
        return weight_map

    def has_tensor(self, name: str) -> bool:
        return name in self.weight_map

    def get_tensor(self, name: str) -> torch.Tensor:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"Checkpoint tensor not found: {name}")
        from safetensors import safe_open

        with safe_open(filename, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def get_optional_tensor(self, name: str) -> torch.Tensor | None:
        if not self.has_tensor(name):
            return None
        return self.get_tensor(name)

    def get_tensors(self, names: Iterable[str]) -> dict[str, torch.Tensor]:
        by_file: dict[str, list[str]] = defaultdict(list)
        for name in names:
            filename = self.weight_map.get(name)
            if filename is None:
                raise KeyError(f"Checkpoint tensor not found: {name}")
            by_file[filename].append(name)

        from safetensors import safe_open

        result: dict[str, torch.Tensor] = {}
        for filename, file_names in by_file.items():
            with safe_open(filename, framework="pt", device="cpu") as f:
                for name in file_names:
                    result[name] = f.get_tensor(name)
        return result


def cat_tensors_to_device(
    tensors: Iterable[torch.Tensor],
    *,
    dim: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Concatenate on CPU, then transfer once to the target GPU.

    This avoids multiple H2D copies followed by an extra device-side concat
    allocation during streaming model load.
    """

    cpu_tensor = torch.cat([tensor.to(dtype=dtype, copy=False) for tensor in tensors], dim=dim).contiguous()
    return cpu_tensor.to(device=device, dtype=dtype, non_blocking=True)


def tensor_to_device(tensor: torch.Tensor, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    return tensor.to(device=device, dtype=dtype, non_blocking=True).contiguous()


def zeros_to_device(shape: tuple[int, ...], *, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype, device=device).contiguous()
