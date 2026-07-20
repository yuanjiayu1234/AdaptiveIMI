import torch
import torch.nn.functional as F
import flashinfer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from .base import LLM
from .weight_loader import SafetensorsWeightSource, cat_tensors_to_device, tensor_to_device
from attn_hub import full_decode_attn, imi_decode_attn, \
                     full_prefill_attn, full_decode_attn_offload


class LlamaLayer:
    """
    A class representing the Llama layer.
    """

    def __init__(self, layer_idx, device) -> None:
        self.layer_idx = layer_idx
        self.device = device
    
    def init_layer(self, hf_llama_layer):
        self.wq = hf_llama_layer.self_attn.q_proj.weight.detach()
        self.wk = hf_llama_layer.self_attn.k_proj.weight.detach()
        self.wv = hf_llama_layer.self_attn.v_proj.weight.detach()
        self.wqkv = torch.cat((self.wq, self.wk, self.wv), dim=0).to(self.device, non_blocking=True)
        self.wo = hf_llama_layer.self_attn.o_proj.weight.detach().to(self.device, non_blocking=True)

        self.gate_proj = hf_llama_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_llama_layer.mlp.up_proj.weight.detach()
        self.gate_up_proj = torch.cat((self.gate_proj, self.up_proj), dim=0).to(self.device, non_blocking=True)
        self.down_proj = hf_llama_layer.mlp.down_proj.weight.detach().to(self.device, non_blocking=True)

        self.input_layernorm_weight = hf_llama_layer.input_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.input_layernorm_variance_epsilon = hf_llama_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_llama_layer.post_attention_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.post_attention_layernorm_variance_epsilon = hf_llama_layer.post_attention_layernorm.variance_epsilon

        del self.wq, self.wk, self.wv, self.gate_proj, self.up_proj

    def init_layer_from_tensors(self, tensors: dict[str, torch.Tensor], dtype: torch.dtype):
        self.wqkv = cat_tensors_to_device(
            [tensors["q_proj.weight"], tensors["k_proj.weight"], tensors["v_proj.weight"]],
            dim=0,
            device=self.device,
            dtype=dtype,
        )
        self.wo = tensor_to_device(tensors["o_proj.weight"], device=self.device, dtype=dtype)
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
        self.input_layernorm_variance_epsilon = tensors["input_layernorm.eps"]
        self.post_attention_layernorm_weight = tensor_to_device(
            tensors["post_attention_layernorm.weight"],
            device=self.device,
            dtype=dtype,
        )
        self.post_attention_layernorm_variance_epsilon = tensors["post_attention_layernorm.eps"]


class LlamaModel(LLM):
    """
    A class representing the Llama model.
    """

    def _infer_model_size_b(self) -> int:
        return self._infer_model_size_from_candidates({"405b": 405, "70b": 70, "8b": 8})

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
        self.config = LlamaConfig.from_pretrained(self.model_path)
        self._populate_common_config_fields()
        self.eos_tokens = [self.config.eos_token_id]
        self.init_model()


    def _set_cos_sin_cache(self):
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
        hf_llama = LlamaForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self._initialize_shared_runtime_tensors(hf_llama, root_device)
        self._initialize_layers(hf_llama.model.layers, LlamaLayer)
        self._finalize_model_init()

    def _init_model_from_safetensors(self, root_device: str):
        source = SafetensorsWeightSource(self.model_path)
        base = f"model.layers"
        rope_scaling = getattr(self.config, "rope_scaling", None)
        rope_type = "default"
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, attention_scaling = rope_init_fn(self.config, device="cpu")
        shared = source.get_tensors(["model.embed_tokens.weight", "model.norm.weight"])
        self._initialize_shared_runtime_tensors_from_weights(
            embed_weight=shared["model.embed_tokens.weight"],
            lm_head_weight=source.get_optional_tensor("lm_head.weight"),
            norm_weight=shared["model.norm.weight"],
            norm_variance_epsilon=float(getattr(self.config, "rms_norm_eps", 1e-6)),
            inv_freq=inv_freq,
            attention_scaling=attention_scaling,
            root_device=root_device,
        )
        self.layers = []
        for layer_idx in range(self.num_layers):
            prefix = f"{base}.{layer_idx}"
            names = [
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
            tensors = source.get_tensors(names)
            layer = LlamaLayer(layer_idx, device=self.layer_mapping[str(layer_idx)])
            layer.init_layer_from_tensors(
                {
                    "q_proj.weight": tensors[f"{prefix}.self_attn.q_proj.weight"],
                    "k_proj.weight": tensors[f"{prefix}.self_attn.k_proj.weight"],
                    "v_proj.weight": tensors[f"{prefix}.self_attn.v_proj.weight"],
                    "o_proj.weight": tensors[f"{prefix}.self_attn.o_proj.weight"],
                    "gate_proj.weight": tensors[f"{prefix}.mlp.gate_proj.weight"],
                    "up_proj.weight": tensors[f"{prefix}.mlp.up_proj.weight"],
                    "down_proj.weight": tensors[f"{prefix}.mlp.down_proj.weight"],
                    "input_layernorm.weight": tensors[f"{prefix}.input_layernorm.weight"],
                    "input_layernorm.eps": float(getattr(self.config, "rms_norm_eps", 1e-6)),
                    "post_attention_layernorm.weight": tensors[f"{prefix}.post_attention_layernorm.weight"],
                    "post_attention_layernorm.eps": float(getattr(self.config, "rms_norm_eps", 1e-6)),
                },
                self.dtype,
            )
            self.layers.append(layer)
        self._finalize_model_init()


    def init_kv_cache(self, valid_start, attn_config):
        self._build_kv_cache(valid_start, attn_config, support_offload=True, adaptive_prefill_bsz=self.prefill_bsz)

    def wqkv(self, hidden_states, layer):
        qkv = F.linear(hidden_states, layer.wqkv)
        query_states, key_states, value_states = qkv.split([self.hidden_size, self.hidden_size//self.num_key_value_groups, self.hidden_size//self.num_key_value_groups], dim=-1)
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
        attn_out = full_prefill_attn(
            query_states,
            key_states,
            value_states,
            causal=True,
        )
        return attn_out
    

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
        if self.attention_type in ('Full_Flash_Attn', 'Full_Flash_Attn_Offload'):
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
