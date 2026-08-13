import math
import torch
import torch.nn.functional as F
import flashinfer
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from .base import LLM
from .weight_loader import SafetensorsWeightSource, cat_tensors_to_device, tensor_to_device
from attn_hub import full_decode_attn, imi_decode_attn, \
                     full_prefill_attn, full_decode_attn_offload



class QwenLayer:
    """
    A class representing the Qwen layer.
    """

    def __init__(self, layer_idx, device) -> None:
        self.layer_idx = layer_idx
        self.device = device
    
    def init_layer(self, hf_qwen_layer):
        self.wq = hf_qwen_layer.self_attn.q_proj.weight.detach()
        self.wk = hf_qwen_layer.self_attn.k_proj.weight.detach()
        self.wv = hf_qwen_layer.self_attn.v_proj.weight.detach()
        self.bq = hf_qwen_layer.self_attn.q_proj.bias.detach() if hf_qwen_layer.self_attn.q_proj.bias is not None else torch.zeros(self.wq.size(0), dtype=self.wq.dtype)
        self.bk = hf_qwen_layer.self_attn.k_proj.bias.detach() if hf_qwen_layer.self_attn.k_proj.bias is not None else torch.zeros(self.wk.size(0), dtype=self.wk.dtype)
        self.bv = hf_qwen_layer.self_attn.v_proj.bias.detach() if hf_qwen_layer.self_attn.v_proj.bias is not None else torch.zeros(self.wv.size(0), dtype=self.wv.dtype)
        self.wqkv = torch.cat((self.wq, self.wk, self.wv), dim=0).to(self.device, non_blocking=True)
        self.bqkv = torch.cat((self.bq, self.bk, self.bv), dim=0).to(self.device, non_blocking=True)
        self.wo = hf_qwen_layer.self_attn.o_proj.weight.detach().to(self.device, non_blocking=True)
        self.q_norm_weight = None
        self.k_norm_weight = None
        self.q_norm_variance_epsilon = None
        self.k_norm_variance_epsilon = None
        if hasattr(hf_qwen_layer.self_attn, "q_norm") and hasattr(hf_qwen_layer.self_attn, "k_norm"):
            q_norm = hf_qwen_layer.self_attn.q_norm
            k_norm = hf_qwen_layer.self_attn.k_norm
            self.q_norm_weight = q_norm.weight.detach().to(self.device, non_blocking=True)
            self.k_norm_weight = k_norm.weight.detach().to(self.device, non_blocking=True)
            self.q_norm_variance_epsilon = getattr(q_norm, "variance_epsilon", getattr(q_norm, "eps", 1e-6))
            self.k_norm_variance_epsilon = getattr(k_norm, "variance_epsilon", getattr(k_norm, "eps", 1e-6))
        
        self.gate_proj = hf_qwen_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_qwen_layer.mlp.up_proj.weight.detach()
        self.gate_up_proj = torch.cat((self.gate_proj, self.up_proj), dim=0).to(self.device, non_blocking=True)
        self.down_proj = hf_qwen_layer.mlp.down_proj.weight.detach().to(self.device, non_blocking=True)

        self.input_layernorm_weight = hf_qwen_layer.input_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.input_layernorm_variance_epsilon = hf_qwen_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_qwen_layer.post_attention_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.post_attention_layernorm_variance_epsilon = hf_qwen_layer.post_attention_layernorm.variance_epsilon

        del self.wq, self.wk, self.wv, self.bq, self.bk, self.bv, self.gate_proj, self.up_proj

    def init_layer_from_tensors(self, tensors: dict[str, torch.Tensor | float | None], dtype: torch.dtype):
        q_weight = tensors["q_proj.weight"]
        k_weight = tensors["k_proj.weight"]
        v_weight = tensors["v_proj.weight"]
        self.wqkv = cat_tensors_to_device([q_weight, k_weight, v_weight], dim=0, device=self.device, dtype=dtype)
        q_bias = tensors.get("q_proj.bias")
        k_bias = tensors.get("k_proj.bias")
        v_bias = tensors.get("v_proj.bias")
        if q_bias is None:
            q_bias = torch.zeros(q_weight.size(0), dtype=dtype)
        if k_bias is None:
            k_bias = torch.zeros(k_weight.size(0), dtype=dtype)
        if v_bias is None:
            v_bias = torch.zeros(v_weight.size(0), dtype=dtype)
        self.bqkv = cat_tensors_to_device([q_bias, k_bias, v_bias], dim=0, device=self.device, dtype=dtype)
        self.wo = tensor_to_device(tensors["o_proj.weight"], device=self.device, dtype=dtype)

        self.q_norm_weight = None
        self.k_norm_weight = None
        self.q_norm_variance_epsilon = None
        self.k_norm_variance_epsilon = None
        if tensors.get("q_norm.weight") is not None and tensors.get("k_norm.weight") is not None:
            self.q_norm_weight = tensor_to_device(tensors["q_norm.weight"], device=self.device, dtype=dtype)
            self.k_norm_weight = tensor_to_device(tensors["k_norm.weight"], device=self.device, dtype=dtype)
            self.q_norm_variance_epsilon = float(tensors["q_norm.eps"])
            self.k_norm_variance_epsilon = float(tensors["k_norm.eps"])

        self.gate_up_proj = cat_tensors_to_device(
            [tensors["gate_proj.weight"], tensors["up_proj.weight"]],
            dim=0,
            device=self.device,
            dtype=dtype,
        )
        self.down_proj = tensor_to_device(tensors["down_proj.weight"], device=self.device, dtype=dtype)
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


class QwenModel(LLM):
    """
    A class representing the Qwen model.
    """

    def _infer_model_size_b(self) -> int:
        return self._infer_model_size_from_candidates({"72b": 72, "7b": 7})

    def __init__(
        self,
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str,
        tokenizer: AutoTokenizer = None,
        model_path: str = None,
        parallel_mode: str = "auto",
        weight_loading: str = "hf",
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        super().__init__(
            model_name,
            max_length,
            dtype,
            device_map,
            parallel_mode=parallel_mode,
            weight_loading=weight_loading,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.model_path = self._resolve_model_path(model_path if model_path else model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path) if tokenizer is None else tokenizer
        self.config = AutoConfig.from_pretrained(self.model_path)
        self._populate_common_config_fields()
        self.head_dim = getattr(self.config, "head_dim", self.hidden_size // self.num_heads)
        rope_parameters = getattr(self.config, "rope_parameters", None) or {}
        self.base = getattr(self.config, "rope_theta", None) or rope_parameters.get("rope_theta", 1000000)
        rope_scaling = getattr(self.config, "rope_scaling", None) or {}
        if rope_parameters.get("rope_type") == "yarn":
            rope_scaling = rope_parameters
        rope_type = str(rope_scaling.get("rope_type") or rope_scaling.get("type") or "").lower()
        model_type = str(getattr(self.config, "model_type", "")).lower()
        self.use_yarn_rope = bool(
            rope_type == "yarn"
            or (model_type == "qwen3" and int(self.max_length) > 32768)
        )
        self.yarn_factor = float(rope_scaling.get("factor", 4.0))
        self.original_max_position_embeddings = int(
            rope_scaling.get(
                "original_max_position_embeddings",
                32768 if getattr(self.config, "model_type", "") == "qwen3" else self.max_position_embeddings,
            )
        )
        self.eos_tokens = [self.config.eos_token_id]
        self.init_model()

    
    def _set_cos_sin_cache(self):
        if self.use_yarn_rope:
            def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
                """Inverse dimension formula to find the dimension based on the number of rotations"""
                return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

            def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
                """Find dimension range bounds based on rotations"""
                low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
                high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
                return max(low, 0), min(high, dim - 1)

            def linear_ramp_factor(min, max, dim):
                if min == max:
                    max += 0.001  # Prevent singularity

                linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
                ramp_func = torch.clamp(linear_func, 0, 1)
                return ramp_func
            
            attention_factor = 0.1 * math.log(self.yarn_factor) + 1.0
            beta_fast = 32
            beta_slow = 1

            pos_freqs = self.base ** (torch.arange(0, self.head_dim, 2).float().to(self.inv_freq.device) / self.head_dim)
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (self.yarn_factor * pos_freqs)

            low, high = find_correction_range(
                beta_fast,
                beta_slow,
                self.head_dim,
                self.base,
                self.original_max_position_embeddings,
            )

            inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, self.head_dim // 2).float().to(self.inv_freq.device)
            inv_freq = (
                inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
                + inv_freq_extrapolation * inv_freq_extrapolation_factor
            )

            self.inv_freq = inv_freq
            self.attention_scaling = attention_factor

        t = torch.arange(self.max_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos()*self.attention_scaling, freqs.sin()*self.attention_scaling


    def init_model(self):
        root_device = self._configure_devices()
        if self._should_use_streaming_weight_loader():
            self._init_model_from_safetensors(root_device)
            return
        model_kwargs = {"torch_dtype": self.dtype}
        hf_device_map = self._hf_device_map()
        if hf_device_map is not None:
            model_kwargs.update(device_map=hf_device_map, low_cpu_mem_usage=True)
        hf_qwen = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self._initialize_shared_runtime_tensors(hf_qwen, root_device)
        self._initialize_layers(hf_qwen.model.layers, QwenLayer)
        self._finalize_model_init()

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
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
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
            layer = QwenLayer(layer_idx, device=self.layer_mapping[str(layer_idx)])
            layer.init_layer_from_tensors(
                {
                    "q_proj.weight": tensors[f"{prefix}.self_attn.q_proj.weight"],
                    "k_proj.weight": tensors[f"{prefix}.self_attn.k_proj.weight"],
                    "v_proj.weight": tensors[f"{prefix}.self_attn.v_proj.weight"],
                    "q_proj.bias": tensors.get(f"{prefix}.self_attn.q_proj.bias"),
                    "k_proj.bias": tensors.get(f"{prefix}.self_attn.k_proj.bias"),
                    "v_proj.bias": tensors.get(f"{prefix}.self_attn.v_proj.bias"),
                    "o_proj.weight": tensors[f"{prefix}.self_attn.o_proj.weight"],
                    "q_norm.weight": tensors.get(f"{prefix}.self_attn.q_norm.weight"),
                    "k_norm.weight": tensors.get(f"{prefix}.self_attn.k_norm.weight"),
                    "q_norm.eps": norm_eps,
                    "k_norm.eps": norm_eps,
                    "gate_proj.weight": tensors[f"{prefix}.mlp.gate_proj.weight"],
                    "up_proj.weight": tensors[f"{prefix}.mlp.up_proj.weight"],
                    "down_proj.weight": tensors[f"{prefix}.mlp.down_proj.weight"],
                    "input_layernorm.weight": tensors[f"{prefix}.input_layernorm.weight"],
                    "input_layernorm.eps": norm_eps,
                    "post_attention_layernorm.weight": tensors[f"{prefix}.post_attention_layernorm.weight"],
                    "post_attention_layernorm.eps": norm_eps,
                },
                self.dtype,
            )
            self.layers.append(layer)
        self._finalize_model_init()

    def init_kv_cache(self, valid_start, attn_config):
        self._build_kv_cache(valid_start, attn_config, support_offload=True, adaptive_prefill_bsz=self.prefill_bsz)


    def wqkv(self, hidden_states, layer):
        qkv = F.linear(hidden_states, layer.wqkv, layer.bqkv)
        query_states, key_states, value_states = qkv.split([self.hidden_size, self.hidden_size//self.num_key_value_groups, self.hidden_size//self.num_key_value_groups], dim=-1)
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
        hidden_states = F.linear(hidden_states, layer.wo)
        return hidden_states

    
    def prefill_attention(
        self,
        query_states,
        key_states,
        value_states,
        layer_idx,
        chunk_size=None,
        chunk_callback=None,
    ):
        return full_prefill_attn(
            query_states,
            key_states,
            value_states,
            causal=True,
        )
    

    def decode_attention(self, query_states, key_states, value_states, layer_idx):
        if self.attention_type == 'Full_Flash_Attn':
            attn_out = full_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        elif self.attention_type == 'Full_Flash_Attn_Offload':
            attn_out = full_decode_attn_offload(query_states, layer_idx, self.kv_cache)
        elif self.attention_type == "AdaptiveIMI":
            attn_out = imi_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return attn_out

    
    def mlp(self, hidden_states, layer):
        hidden_states = F.linear(hidden_states, layer.gate_up_proj)
        dim = hidden_states.shape[-1] // 2
        hidden_shape = (hidden_states.shape[:-1] + (dim,))
        out = torch.empty(hidden_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        flashinfer.activation.silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, layer.down_proj)
        return hidden_states 

    
    def parameter_move(self, hidden_states, ldx):
        next_device = self.layer_mapping[str(ldx+1)] if str(ldx+1) in self.layer_mapping else self.layer_mapping[str(0)]
        torch.cuda.set_device(next_device)
        hidden_states = hidden_states.to(next_device, non_blocking=True)
        self._activate_device_runtime_tensors(next_device)
        if self.attention_type == 'Full_Flash_Attn':
            if hidden_states.shape[1] == 1:
                self.kv_cache.batch_indices = self.kv_cache.batch_indices_dict[next_device]
                self.kv_cache.valid_length = self.kv_cache.valid_length_dict[next_device]
        elif self.attention_type == "AdaptiveIMI":
            self.kv_cache.execution_buffer_keys = self.kv_cache.execution_buffer_keys_dict[next_device]
            self.kv_cache.execution_buffer_values = self.kv_cache.execution_buffer_values_dict[next_device]
            self.kv_cache.valid_lengths = self.kv_cache.valid_lengths_dict[next_device]
        return hidden_states

    
    def layernorm(self, hidden_states, epsilon, weight):
        bsz, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.reshape(bsz * seq_len, dim)
        hidden_states = flashinfer.rmsnorm(hidden_states, weight, epsilon)
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        return hidden_states


    def apply_rotary_pos_emb(self, query_states, key_states, position_ids):
        bsz, _, hidden_dim = query_states.shape
        _, _, kv_dim = key_states.shape
        query_states = query_states.view(-1, hidden_dim)
        key_states = key_states.view(-1, kv_dim)
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(position_ids, query_states, key_states, self.head_dim, self.cos_sin_cache, True)
        query_states = query_states.view(bsz, -1, hidden_dim)
        key_states = key_states.view(bsz, -1, kv_dim)
        return query_states, key_states


    def position_embedd(self, query_states, key_states):
        bsz, seq_len, _ = key_states.shape
        position_ids = self.position_ids[self.kv_cache.context:self.kv_cache.context+seq_len].unsqueeze(0).repeat(bsz, 1)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
        return query_states, key_states
