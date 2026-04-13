import gc
import os
import re
import math
import torch
import torch.nn.functional as F
import flashinfer
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Config
from .base import LLM
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
        self.bq = hf_qwen_layer.self_attn.q_proj.bias.detach()
        self.bk = hf_qwen_layer.self_attn.k_proj.bias.detach()
        self.bv = hf_qwen_layer.self_attn.v_proj.bias.detach()
        self.wqkv = torch.cat((self.wq, self.wk, self.wv), dim=0).to(self.device, non_blocking=True)
        self.bqkv = torch.cat((self.bq, self.bk, self.bv), dim=0).to(self.device, non_blocking=True)
        self.wo = hf_qwen_layer.self_attn.o_proj.weight.detach().to(self.device, non_blocking=True)
        
        self.gate_proj = hf_qwen_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_qwen_layer.mlp.up_proj.weight.detach()
        self.gate_up_proj = torch.cat((self.gate_proj, self.up_proj), dim=0).to(self.device, non_blocking=True)
        self.down_proj = hf_qwen_layer.mlp.down_proj.weight.detach().to(self.device, non_blocking=True)

        self.input_layernorm_weight = hf_qwen_layer.input_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.input_layernorm_variance_epsilon = hf_qwen_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_qwen_layer.post_attention_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.post_attention_layernorm_variance_epsilon = hf_qwen_layer.post_attention_layernorm.variance_epsilon

        del self.wq, self.wk, self.wv, self.bq, self.bk, self.bv, self.gate_proj, self.up_proj


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
        model_path: str = None
    ) -> None:
        super().__init__(model_name, max_length, dtype, device_map)
        self.model_path = self._resolve_model_path(model_path if model_path else model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path) if tokenizer is None else tokenizer
        self.config = Qwen2Config.from_pretrained(self.model_path)
        self._populate_common_config_fields()
        self.base = self.config.rope_theta
        self.yarn_factor = 4         # # qwen2.5 use yarn in context length larger than 32768
        self.eos_tokens = [self.config.eos_token_id]
        self.init_model()

    
    def _set_cos_sin_cache(self):
        if self.max_length > 32768:
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

            low, high = find_correction_range(beta_fast, beta_slow, self.head_dim, self.base, self.max_position_embeddings)

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
        hf_qwen = Qwen2ForCausalLM.from_pretrained(self.model_path, torch_dtype=self.dtype)
        root_device = self._configure_devices()
        self._initialize_shared_runtime_tensors(hf_qwen, root_device)
        self._initialize_layers(hf_qwen.model.layers, QwenLayer)
        self._finalize_model_init()


    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        if not model_path or not os.path.isdir(model_path):
            return model_path
        snapshots_dir = os.path.join(model_path, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return model_path
        snapshot_dirs = [
            os.path.join(snapshots_dir, name)
            for name in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, name))
        ]
        if not snapshot_dirs:
            return model_path
        snapshot_dirs.sort(key=os.path.getmtime, reverse=True)
        return snapshot_dirs[0]

    
    def init_kv_cache(self, valid_start, attn_config):
        self._build_kv_cache(valid_start, attn_config, support_offload=True, adaptive_prefill_bsz=self.prefill_bsz)


    def wqkv(self, hidden_states, layer):
        qkv = F.linear(hidden_states, layer.wqkv, layer.bqkv)
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
        hidden_states = hidden_states.to(next_device)
        self.position_ids = self.position_ids.to(next_device)
        self.cos_sin_cache = self.cos_sin_cache.to(next_device)
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
