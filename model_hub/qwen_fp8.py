from __future__ import annotations

import torch
import torch.nn.functional as F
import flashinfer

from .fp8_linear import Fp8Linear
from .qwen import QwenModel
from .weight_loader import SafetensorsWeightSource, tensor_to_device


class QwenFp8Layer:
    def __init__(self, layer_idx: int, device: str) -> None:
        self.layer_idx = int(layer_idx)
        self.device = device

    def init_layer_from_tensors(self, tensors: dict[str, torch.Tensor | float | None], dtype: torch.dtype) -> None:
        self.q_proj = Fp8Linear.from_tensors(
            weight=tensors["q_proj.weight"],
            weight_scale=tensors["q_proj.weight_scale"],
            bias=tensors.get("q_proj.bias"),
            device=self.device,
            dtype=dtype,
        )
        self.k_proj = Fp8Linear.from_tensors(
            weight=tensors["k_proj.weight"],
            weight_scale=tensors["k_proj.weight_scale"],
            bias=tensors.get("k_proj.bias"),
            device=self.device,
            dtype=dtype,
        )
        self.v_proj = Fp8Linear.from_tensors(
            weight=tensors["v_proj.weight"],
            weight_scale=tensors["v_proj.weight_scale"],
            bias=tensors.get("v_proj.bias"),
            device=self.device,
            dtype=dtype,
        )
        self.o_proj = Fp8Linear.from_tensors(
            weight=tensors["o_proj.weight"],
            weight_scale=tensors["o_proj.weight_scale"],
            bias=None,
            device=self.device,
            dtype=dtype,
        )
        self.gate_proj = Fp8Linear.from_tensors(
            weight=tensors["gate_proj.weight"],
            weight_scale=tensors["gate_proj.weight_scale"],
            bias=None,
            device=self.device,
            dtype=dtype,
        )
        self.up_proj = Fp8Linear.from_tensors(
            weight=tensors["up_proj.weight"],
            weight_scale=tensors["up_proj.weight_scale"],
            bias=None,
            device=self.device,
            dtype=dtype,
        )
        self.down_proj = Fp8Linear.from_tensors(
            weight=tensors["down_proj.weight"],
            weight_scale=tensors["down_proj.weight_scale"],
            bias=None,
            device=self.device,
            dtype=dtype,
        )

        self.q_norm_weight = None
        self.k_norm_weight = None
        self.q_norm_variance_epsilon = None
        self.k_norm_variance_epsilon = None
        if tensors.get("q_norm.weight") is not None and tensors.get("k_norm.weight") is not None:
            self.q_norm_weight = tensor_to_device(tensors["q_norm.weight"], device=self.device, dtype=dtype)
            self.k_norm_weight = tensor_to_device(tensors["k_norm.weight"], device=self.device, dtype=dtype)
            self.q_norm_variance_epsilon = float(tensors["q_norm.eps"])
            self.k_norm_variance_epsilon = float(tensors["k_norm.eps"])

        self.input_layernorm_weight = tensor_to_device(
            tensors["input_layernorm.weight"],
            device=self.device,
            dtype=dtype,
        )
        self.input_layernorm_variance_epsilon = float(tensors["input_layernorm.eps"])
        self.post_attention_layernorm_weight = tensor_to_device(
            tensors["post_attention_layernorm.weight"],
            device=self.device,
            dtype=dtype,
        )
        self.post_attention_layernorm_variance_epsilon = float(tensors["post_attention_layernorm.eps"])


class QwenFp8Model(QwenModel):
    """Qwen2/Qwen3 compressed-tensors FP8 runtime for AdaptiveIMI."""

    def _infer_model_size_b(self) -> int:
        return self._infer_model_size_from_candidates({"32b": 32, "72b": 72, "8b": 8, "7b": 7})

    def init_model(self):
        self.weight_loading = "compressed_tensors_fp8"
        root_device = self._configure_devices()
        self._init_model_from_safetensors(root_device)

    def _init_model_from_safetensors(self, root_device: str):
        source = SafetensorsWeightSource(self.model_path)
        inv_freq = 1.0 / (
            float(self.base)
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        norm_eps = float(getattr(self.config, "rms_norm_eps", 1e-6))
        shared = source.get_tensors(["model.embed_tokens.weight", "model.norm.weight"])
        self._initialize_shared_runtime_tensors_from_weights(
            embed_weight=shared["model.embed_tokens.weight"],
            lm_head_weight=source.get_optional_tensor("lm_head.weight"),
            norm_weight=shared["model.norm.weight"],
            norm_variance_epsilon=norm_eps,
            inv_freq=inv_freq,
            attention_scaling=1.0,
            root_device=root_device,
        )
        self.layers = []
        for layer_idx in range(self.num_layers):
            prefix = f"model.layers.{layer_idx}"
            required = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_proj.weight_scale",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.k_proj.weight_scale",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_proj.weight_scale",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.o_proj.weight_scale",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.gate_proj.weight_scale",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.up_proj.weight_scale",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.mlp.down_proj.weight_scale",
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
            ]
            optional = [
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
            ]
            tensors = source.get_tensors(required + [name for name in optional if source.has_tensor(name)])
            layer = QwenFp8Layer(layer_idx, device=self.layer_mapping[str(layer_idx)])
            layer.init_layer_from_tensors(
                {
                    "q_proj.weight": tensors[f"{prefix}.self_attn.q_proj.weight"],
                    "q_proj.weight_scale": tensors[f"{prefix}.self_attn.q_proj.weight_scale"],
                    "k_proj.weight": tensors[f"{prefix}.self_attn.k_proj.weight"],
                    "k_proj.weight_scale": tensors[f"{prefix}.self_attn.k_proj.weight_scale"],
                    "v_proj.weight": tensors[f"{prefix}.self_attn.v_proj.weight"],
                    "v_proj.weight_scale": tensors[f"{prefix}.self_attn.v_proj.weight_scale"],
                    "q_proj.bias": tensors.get(f"{prefix}.self_attn.q_proj.bias"),
                    "k_proj.bias": tensors.get(f"{prefix}.self_attn.k_proj.bias"),
                    "v_proj.bias": tensors.get(f"{prefix}.self_attn.v_proj.bias"),
                    "o_proj.weight": tensors[f"{prefix}.self_attn.o_proj.weight"],
                    "o_proj.weight_scale": tensors[f"{prefix}.self_attn.o_proj.weight_scale"],
                    "q_norm.weight": tensors.get(f"{prefix}.self_attn.q_norm.weight"),
                    "k_norm.weight": tensors.get(f"{prefix}.self_attn.k_norm.weight"),
                    "q_norm.eps": norm_eps,
                    "k_norm.eps": norm_eps,
                    "gate_proj.weight": tensors[f"{prefix}.mlp.gate_proj.weight"],
                    "gate_proj.weight_scale": tensors[f"{prefix}.mlp.gate_proj.weight_scale"],
                    "up_proj.weight": tensors[f"{prefix}.mlp.up_proj.weight"],
                    "up_proj.weight_scale": tensors[f"{prefix}.mlp.up_proj.weight_scale"],
                    "down_proj.weight": tensors[f"{prefix}.mlp.down_proj.weight"],
                    "down_proj.weight_scale": tensors[f"{prefix}.mlp.down_proj.weight_scale"],
                    "input_layernorm.weight": tensors[f"{prefix}.input_layernorm.weight"],
                    "input_layernorm.eps": norm_eps,
                    "post_attention_layernorm.weight": tensors[f"{prefix}.post_attention_layernorm.weight"],
                    "post_attention_layernorm.eps": norm_eps,
                },
                self.dtype,
            )
            self.layers.append(layer)
        self._finalize_model_init()

    def wqkv(self, hidden_states, layer):
        query_states = layer.q_proj(hidden_states)
        key_states = layer.k_proj(hidden_states)
        value_states = layer.v_proj(hidden_states)
        if layer.q_norm_weight is not None and layer.k_norm_weight is not None:
            bsz, seq_len, _ = query_states.shape
            query_states = query_states.reshape(bsz * seq_len * self.num_heads, self.head_dim)
            key_states = key_states.reshape(bsz * seq_len * self.num_key_value_heads, self.head_dim)
            query_states = flashinfer.rmsnorm(
                query_states,
                layer.q_norm_weight,
                layer.q_norm_variance_epsilon,
            )
            key_states = flashinfer.rmsnorm(
                key_states,
                layer.k_norm_weight,
                layer.k_norm_variance_epsilon,
            )
            query_states = query_states.view(bsz, seq_len, self.hidden_size)
            key_states = key_states.view(
                bsz,
                seq_len,
                self.hidden_size // self.num_key_value_groups,
            )
        return query_states, key_states, value_states

    def wo(self, hidden_states, layer, bsz, seq_len, dim):
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        return layer.o_proj(hidden_states)

    def mlp(self, hidden_states, layer):
        gate = layer.gate_proj(hidden_states)
        up = layer.up_proj(hidden_states)
        hidden_states = F.silu(gate) * up
        return layer.down_proj(hidden_states)
