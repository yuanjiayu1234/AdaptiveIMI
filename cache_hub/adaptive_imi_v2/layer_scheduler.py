from __future__ import annotations

from typing import Dict, List, Optional


class LayerBuildScheduler:
    """Layer-granularity CPU/GPU index build policy."""

    def __init__(
        self,
        layer_num: int,
        *,
        policy: str = "alternate",
        start: str = "cpu",
        gpu_every: int = 0,
        explicit: Optional[Dict[int, str]] = None,
        allow_gpu: bool = True,
    ) -> None:
        self.layer_num = int(layer_num)
        self.policy = str(policy or "alternate").lower()
        self.start = str(start or "cpu").lower()
        self.gpu_every = max(int(gpu_every or 0), 0)
        self.explicit = {int(k): str(v).lower() for k, v in (explicit or {}).items()}
        self.allow_gpu = bool(allow_gpu)

        if self.start not in ("cpu", "gpu"):
            raise ValueError("hybrid_index.start must be 'cpu' or 'gpu'")
        if self.policy not in ("alternate", "explicit", "cpu_only", "gpu_only", "gpu_every_n"):
            raise ValueError("hybrid_index.policy must be alternate, explicit, cpu_only, gpu_only or gpu_every_n")
        if self.policy == "gpu_every_n" and self.gpu_every <= 0:
            raise ValueError("hybrid_index.gpu_every must be positive when policy is gpu_every_n")

    @classmethod
    def from_config(cls, layer_num: int, cfg: Optional[dict]) -> "LayerBuildScheduler":
        cfg = cfg or {}
        explicit_obj = cfg.get("layers") or cfg.get("explicit") or {}
        explicit: Dict[int, str] = {}
        if isinstance(explicit_obj, dict):
            for key, value in explicit_obj.items():
                explicit[int(key)] = str(value).lower()
        return cls(
            layer_num,
            policy=str(cfg.get("policy", "alternate")),
            start=str(cfg.get("start", "cpu")),
            gpu_every=int(cfg.get("gpu_every", cfg.get("gpu_every_n", 0)) or 0),
            explicit=explicit,
            allow_gpu=bool(cfg.get("allow_gpu", True)),
        )

    def build_device(self, layer_idx: int) -> str:
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= self.layer_num:
            raise IndexError(f"layer_idx out of range: {layer_idx}")

        if self.policy == "cpu_only":
            return "cpu"
        if self.policy == "gpu_only":
            return "gpu" if self.allow_gpu else "cpu"
        if self.policy == "explicit":
            mode = self.explicit.get(layer_idx, "cpu")
            if mode not in ("cpu", "gpu"):
                raise ValueError(f"invalid build mode for layer {layer_idx}: {mode}")
            return mode if mode == "cpu" or self.allow_gpu else "cpu"
        if self.policy == "gpu_every_n":
            if not self.allow_gpu:
                return "cpu"
            offset = 0 if self.start == "gpu" else self.gpu_every - 1
            return "gpu" if layer_idx % self.gpu_every == offset else "cpu"

        # alternate; user-facing "layer1" is zero-based layer_idx 0.
        cpu_on_even = self.start == "cpu"
        mode = "cpu" if (layer_idx % 2 == 0) == cpu_on_even else "gpu"
        return mode if mode == "cpu" or self.allow_gpu else "cpu"

    def all_modes(self) -> List[str]:
        return [self.build_device(layer_idx) for layer_idx in range(self.layer_num)]
