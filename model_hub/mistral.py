import gc
import re
import os
import json
import torch
import torch.nn.functional as F
import flashinfer
from transformers import AutoTokenizer, MistralForCausalLM, MistralConfig, GenerationConfig
from .base import LLM
from attn_hub import full_decode_attn, full_prefill_attn, imi_decode_attn


class MistralLayer:
    """
    A class representing a single Mistral decoder layer.
    """

    def __init__(self, layer_idx, device) -> None:
        self.layer_idx = layer_idx
        self.device = device

    def init_layer(self, hf_layer):
        self.wq = hf_layer.self_attn.q_proj.weight.detach()
        self.wk = hf_layer.self_attn.k_proj.weight.detach()
        self.wv = hf_layer.self_attn.v_proj.weight.detach()
        self.wqkv = torch.cat((self.wq, self.wk, self.wv), dim=0).to(self.device, non_blocking=True)
        self.wo = hf_layer.self_attn.o_proj.weight.detach().to(self.device, non_blocking=True)

        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.gate_up_proj = torch.cat((self.gate_proj, self.up_proj), dim=0).to(self.device, non_blocking=True)
        self.down_proj = hf_layer.mlp.down_proj.weight.detach().to(self.device, non_blocking=True)

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

        del self.wq, self.wk, self.wv, self.gate_proj, self.up_proj


class MistralModel(LLM):
    """
    A class representing the Mistral model, following the same pattern as LlamaModel.
    """

    def _infer_model_size_b(self) -> int:
        return self._infer_model_size_from_candidates({"7b": 7})

    def __init__(
        self,
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str,
        model_path: str = None
    ) -> None:
        super().__init__(model_name, max_length, dtype, device_map)
        self.model_path = model_path if model_path else model_name

        load_path = self.model_path

        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.config = MistralConfig.from_pretrained(load_path)
        if hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None:
            self.config.rope_scaling.setdefault("type", "linear")
        self._populate_common_config_fields()

        try:
            gen_config = GenerationConfig.from_pretrained(load_path)
            eos_ids = gen_config.eos_token_id
        except Exception:
            eos_ids = self.config.eos_token_id
        if isinstance(eos_ids, list):
            self.eos_tokens = eos_ids
        elif eos_ids is not None:
            self.eos_tokens = [eos_ids]
        else:
            self.eos_tokens = [self.tokenizer.eos_token_id]

        self.init_model()

    def _set_cos_sin_cache(self):
        t = torch.arange(self.max_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos() * self.attention_scaling, freqs.sin() * self.attention_scaling

    def init_model(self):
        hf_model = MistralForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            config=self.config,
        )
        root_device = self._configure_devices()
        self._initialize_shared_runtime_tensors(hf_model, root_device)
        self._initialize_layers(hf_model.model.layers, MistralLayer)
        self._finalize_model_init()

    def init_kv_cache(self, valid_start, attn_config=None):
        self._reset_kv_cache_state()
        model_config = self._load_attn_config(attn_config)
        self._build_kv_cache(valid_start, model_config, support_offload=False, adaptive_prefill_bsz=self.batch_size)

    def prefill_attention(
        self,
        query_states,
        key_states,
        value_states,
        layer_idx,
        chunk_size=None,
        chunk_callback=None,
    ):
        if self.attention_type == 'Full_Flash_Attn':
            attn_out = full_prefill_attn(
                query_states,
                key_states,
                value_states,
                causal=True,
            )
        elif self.attention_type == "AdaptiveIMI":
            attn_out = full_prefill_attn(
                query_states,
                key_states,
                value_states,
                causal=True,
            )
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return attn_out

    def decode_attention(self, query_states, key_states, value_states, layer_idx):
        if self.attention_type == 'Full_Flash_Attn':
            attn_out = full_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        elif self.attention_type == "AdaptiveIMI":
            attn_out = imi_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return attn_out

    def wqkv(self, hidden_states, layer):
        qkv = F.linear(hidden_states, layer.wqkv)
        query_states, key_states, value_states = qkv.split([self.hidden_size, self.hidden_size//self.num_key_value_groups, self.hidden_size//self.num_key_value_groups], dim=-1)
        return query_states, key_states, value_states

    def wo(self, hidden_states, layer, bsz, seq_len, dim):
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        hidden_states = F.linear(hidden_states, layer.wo)
        return hidden_states

    def mlp(self, hidden_states, layer):
        hidden_states = F.linear(hidden_states, layer.gate_up_proj)
        dim = hidden_states.shape[-1] // 2
        hidden_shape = (hidden_states.shape[:-1] + (dim,))
        out = torch.empty(hidden_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        flashinfer.activation.silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, layer.down_proj)
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
        position_ids = self.position_ids[self.kv_cache.context:self.kv_cache.context + seq_len].unsqueeze(0).repeat(bsz, 1)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
        return query_states, key_states


    def parameter_move(self, hidden_states, ldx):
        next_device = self.layer_mapping[str(ldx+1)] if str(ldx+1) in self.layer_mapping else self.layer_mapping[str(0)]
        torch.cuda.set_device(next_device)
        hidden_states = hidden_states.to(next_device)
        self.position_ids = self.position_ids.to(next_device)
        self.cos_sin_cache = self.cos_sin_cache.to(next_device)
        if self.attention_type == 'Full_Flash_Attn':
            if hidden_states.shape[1] == 1:
                self.kv_cache.batch_indices = self.kv_cache.batch_indices.to(next_device)
                self.kv_cache.valid_length = self.kv_cache.valid_length.to(next_device)
        elif self.attention_type == "AdaptiveIMI":
            pass
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return hidden_states
