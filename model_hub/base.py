import gc
import json
import os
import time

import flashinfer
import torch
import torch.nn.functional as F
from termcolor import colored

from config import resolve_config_name

from .prefill import PrefillMixin
from .sampling import SamplingMixin


class LLM(PrefillMixin, SamplingMixin):
    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str
    ) -> None:
        """ Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
            device_map (str): The device for model, suppor 'cuda:x' or 'auto (automatically use all visible GPUs)'.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map
        self.prefill_chunk_size = 65536
        self.prefill_attn_chunk_size = self.prefill_chunk_size
        self.enable_prefill_attn_chunk = True
        self.profile_prefill_gpu = os.getenv("IMI_PROFILE_PREFILL_GPU", "0") == "1"
        self.prefill_gpu_layer_rows = []
        self._prefill_gpu_attn_events = {}
        self._prefill_gpu_total_events = {}
        self._prefill_prefix_kv_workspace = {}

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

    def _infer_model_size_from_candidates(self, fallback_sizes: dict[str, int] | None = None) -> int:
        candidates = [self.model_name, getattr(self, "model_path", None)]
        for item in candidates:
            if not item:
                continue
            match = __import__('re').search(r"(\d+)\s*[Bb]", str(item))
            if match is not None:
                return int(match.group(1))

        lower = (self.model_name or "").lower()
        for key, value in (fallback_sizes or {}).items():
            if key in lower:
                return value

        raise ValueError(
            f"Cannot infer model size from model_name={self.model_name!r}, "
            f"model_path={getattr(self, 'model_path', None)!r}"
        )

    def _populate_common_config_fields(self) -> None:
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.vocab_size = self.config.vocab_size

    def _configure_devices(self) -> str:
        self.num_gpus = torch.cuda.device_count() if self.device_map == 'auto' else 1
        if self.device_map == 'auto' and self.num_gpus == 1:
            self.device_map = 'cuda:0'

        if self.device_map != 'auto':
            self.layer_mapping = {str(layer_idx): self.device_map for layer_idx in range(self.num_layers)}
            return self.device_map

        self.gpu_ids = list(range(self.num_gpus))
        self.layer_interval = (self.num_layers + self.num_gpus - 1) // self.num_gpus
        self.layer_mapping = {
            str(layer_idx): f'cuda:{layer_idx // self.layer_interval}' for layer_idx in range(self.num_layers)
        }
        return f'cuda:{self.gpu_ids[0]}'

    def _initialize_shared_runtime_tensors(self, hf_model, root_device: str) -> None:
        embed_weight = hf_model.model.embed_tokens.weight.detach()
        lm_head_weight = hf_model.lm_head.weight.detach()

        self.embed_tokens = embed_weight.to(root_device, non_blocking=True)
        if embed_weight.data_ptr() == lm_head_weight.data_ptr():
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = lm_head_weight.to(root_device, non_blocking=True)

        self.norm_weight = hf_model.model.norm.weight.detach().to(root_device, non_blocking=True)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon

        self.position_ids = torch.arange(0, self.max_length).to(root_device, non_blocking=True)
        self.inv_freq = hf_model.model.rotary_emb.inv_freq.detach().to(root_device, non_blocking=True)
        self.attention_scaling = hf_model.model.rotary_emb.attention_scaling
        self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
        self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

    def _initialize_layers(self, hf_layers, layer_cls) -> None:
        self.layers = []
        for layer_idx, hf_layer in enumerate(hf_layers):
            layer = layer_cls(layer_idx, device=self.layer_mapping[str(layer_idx)])
            layer.init_layer(hf_layer)
            self.layers.append(layer)
            hf_layers[layer_idx] = None

    def _finalize_model_init(self) -> None:
        del self.inv_freq, self.cos_cache, self.sin_cache
        gc.collect()
        torch.cuda.empty_cache()

    def _clear_prefill_prefix_kv_workspace(self) -> None:
        for workspace in self._prefill_prefix_kv_workspace.values():
            if not workspace:
                continue
            workspace["key"] = None
            workspace["value"] = None
        self._prefill_prefix_kv_workspace.clear()

    def _reset_kv_cache_state(self) -> None:
        if getattr(self, "kv_cache", None) is not None and hasattr(self.kv_cache, "cleanup"):
            self.kv_cache.cleanup()
        self.kv_cache = None
        self._clear_prefill_prefix_kv_workspace()
        gc.collect()
        torch.cuda.empty_cache()

    def _load_attn_config(self, attn_config):
        if attn_config is not None:
            return attn_config
        project_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        config_dir = os.path.join(project_root, "config")
        config_file = os.path.join(config_dir, resolve_config_name(self.model_name))
        with open(config_file, "r") as f:
            return json.load(f)

    def _build_kv_cache(self, valid_start, model_config, *, support_offload: bool, adaptive_prefill_bsz: int | None = None):
        self.apply_prefill_config(model_config)
        common_kwargs = dict(
            valid_start=valid_start,
            layer_num=self.num_layers,
            batch_size=self.batch_size,
            max_length=self.max_length,
            num_key_value_heads=self.num_key_value_heads,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            layer_mapping=self.layer_mapping,
            prefill_bsz=self.prefill_bsz,
            num_gpus=self.num_gpus,
            model_size=self._infer_model_size_b(),
        )

        if self.attention_type == 'Full_Flash_Attn':
            self._reset_kv_cache_state()
            from cache_hub.full_cache import flash_attn_cache
            self.kv_cache = flash_attn_cache(**common_kwargs)
            return

        if self.attention_type == 'Full_Flash_Attn_Offload':
            self._reset_kv_cache_state()
            if not support_offload:
                raise ValueError(f"Unsupported attention type: {self.attention_type}")
            from cache_hub.offload_cache import flash_attn_cache_offload
            self.kv_cache = flash_attn_cache_offload(**common_kwargs)
            return

        if self.attention_type != 'AdaptiveIMI':
            raise ValueError(f"Unsupported attention type: {self.attention_type}")

        imi_config = model_config.get("AdaptiveIMI")
        if imi_config is None:
            raise ValueError("AdaptiveIMI config is missing for this model.")

        streaming_cfg = imi_config.setdefault("streaming", {})
        streaming_cfg["prefill_chunk_size"] = self.prefill_attn_chunk_size

        if imi_config.get("gpu_only", False):
            raise ValueError("AdaptiveIMI gpu_only mode is not supported.")

        from cache_hub.adaptive_imi.cache import adpimi_cache

        existing_cache = getattr(self, "kv_cache", None)
        if isinstance(existing_cache, adpimi_cache) and existing_cache.can_reuse_for_next_sequence(
            valid_start,
            self.max_new_length,
            self.input_length,
        ):
            existing_cache.reset_for_next_sequence(
                valid_start,
                self.max_new_length,
                self.input_length,
            )
            return

        self._reset_kv_cache_state()
        self.kv_cache = adpimi_cache(
            valid_start=valid_start,
            layer_num=self.num_layers,
            batch_size=self.batch_size,
            max_length=self.max_length,
            num_key_value_heads=self.num_key_value_heads,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            layer_mapping=self.layer_mapping,
            max_new_length=self.max_new_length,
            input_length=self.input_length,
            static_pattern_start=imi_config["static_pattern_start"],
            static_pattern_end=imi_config["static_pattern_end"],
            core=imi_config["core"],
            pages_per_cluster=imi_config["pages_per_cluster"],
            retrieval_budget=imi_config["retrieval_budget"],
            cache_ratio=imi_config.get("cache_ratio", 0.0),
            buffer_cluster_num=imi_config["buffer_cluster_num"],
            prefill_bsz=self.prefill_bsz if adaptive_prefill_bsz is None else adaptive_prefill_bsz,
            num_gpus=self.num_gpus,
            model_size=self._infer_model_size_b(),
            subspace_parts=imi_config.get("subspace_parts", 2),
            runtime_config={
                "cpu_threads": imi_config.get("cpu_threads"),
                "pipeline": imi_config.get("pipeline", {}),
                "kmeans": imi_config.get("kmeans", {}),
                "prefetch": imi_config.get("prefetch", {}),
                "prefill": imi_config.get("prefill", {}),
                "streaming": imi_config.get("streaming", {}),
                "async_update": imi_config.get("async_update", {}),
            },
        )

    def apply_prefill_config(self, model_config: dict) -> None:
        if not model_config:
            return
        prefill_cfg = model_config.get("prefill") or {}
        if not prefill_cfg and isinstance(model_config, dict):
            attn_cfg = None
            if hasattr(self, "attention_type") and self.attention_type:
                attn_cfg = model_config.get(self.attention_type)
            if attn_cfg is None:
                attn_cfg = model_config.get("AdaptiveIMI")
            if isinstance(attn_cfg, dict):
                prefill_cfg = attn_cfg.get("prefill") or {}
        if not isinstance(prefill_cfg, dict):
            return

        chunk_size = prefill_cfg.get("prefill_chunk_size")
        if chunk_size is not None:
            chunk_size = int(chunk_size)
            if chunk_size > 0:
                self.prefill_chunk_size = chunk_size
                self.prefill_attn_chunk_size = chunk_size

        if "enable_prefill_attn_chunk" in prefill_cfg:
            self.enable_prefill_attn_chunk = bool(prefill_cfg.get("enable_prefill_attn_chunk"))
    def _reset_prefill_gpu_profile(self):
        self.prefill_gpu_layer_rows = []
        self._prefill_gpu_attn_events = {}
        self._prefill_gpu_total_events = {}
    def _record_prefill_gpu_attn_events(self, layer_idx: int, start_event, end_event) -> None:
        if not self.profile_prefill_gpu:
            return
        self._prefill_gpu_attn_events.setdefault(int(layer_idx), []).append((start_event, end_event))
    def _record_prefill_gpu_total_events(self, layer_idx: int, start_event, end_event) -> None:
        if not self.profile_prefill_gpu:
            return
        self._prefill_gpu_total_events.setdefault(int(layer_idx), []).append((start_event, end_event))
    def _sum_prefill_gpu_events(self, events) -> float:
        total_ms = 0.0
        for start_event, end_event in events:
            end_event.synchronize()
            total_ms += float(start_event.elapsed_time(end_event))
        return total_ms
    def _start_prefill_gpu_segment(self):
        if not self.profile_prefill_gpu:
            return None
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return start_event
    def _finish_prefill_gpu_total_segment(self, layer_idx: int, start_event) -> None:
        if not self.profile_prefill_gpu or start_event is None:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._record_prefill_gpu_total_events(layer_idx, start_event, end_event)
    def _flush_prefill_gpu_layer_events(self, layer_idx: int) -> None:
        if not self.profile_prefill_gpu:
            return
        attn_events = self._prefill_gpu_attn_events.pop(int(layer_idx), [])
        total_events = self._prefill_gpu_total_events.pop(int(layer_idx), [])
        if not attn_events and not total_events:
            return
        self.prefill_gpu_layer_rows.append({
            "layer_idx": int(layer_idx),
            "prefill_gpu_attn_ms": round(self._sum_prefill_gpu_events(attn_events), 4),
            "prefill_gpu_layer_total_ms": round(self._sum_prefill_gpu_events(total_events), 4),
        })
    def layer_decode(self, layer_idx, hidden_states):
        # print(f'Layer = {layer_idx}')

        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        # assert seq_len == 1, f"Error: seq_len should be 1 for decoding, but got {seq_len}."
        layer = self.layers[layer_idx]

        hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
        hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states

        return hidden_states
    def _should_pin_streaming_hidden(self, bsz: int, seq_len: int) -> bool:
        max_pinned_gb = float(os.getenv("IMI_STREAMING_HIDDEN_PINNED_GB", "1.0"))
        required_gb = (
            float(bsz) * float(seq_len) * float(self.hidden_size) * float(torch.tensor([], dtype=self.dtype).element_size())
        ) / (1024 ** 3)
        return required_gb <= max_pinned_gb
    def _streaming_progress_enabled(self) -> bool:
        return os.getenv("IMI_STREAMING_PROGRESS", "0") == "1"
    def _debug_stop_after_prefill(self) -> bool:
        return os.getenv("IMI_DEBUG_STOP_AFTER_PREFILL", "0") == "1"
    def _debug_skip_prefill_kv_update(self) -> bool:
        return os.getenv("IMI_DEBUG_SKIP_PREFILL_KV_UPDATE", "0") == "1"
    def _debug_max_decode_steps(self):
        value = int(os.getenv("IMI_DEBUG_MAX_DECODE_STEPS", "0"))
        return value if value > 0 else None
    def _debug_print_token_ids(self) -> bool:
        return os.getenv("IMI_DEBUG_PRINT_TOKEN_IDS", "0") == "1"
    def _debug_log_token_ids(self, stage: str, token_ids: torch.Tensor) -> None:
        if not self._debug_print_token_ids():
            return
        print(colored(f"IMI debug: {stage}_token_ids = {token_ids.tolist()}", 'yellow'))
    def _log_streaming_progress(self, message: str) -> None:
        if self._streaming_progress_enabled():
            print(message, flush=True)

    def move(self):
        torch.cuda.empty_cache()
        if self.attention_type in ('Full_Flash_Attn', 'Full_Flash_Attn_Offload'):
            self.kv_cache.move_gpu()
        elif self.attention_type == "AdaptiveIMI":
            self.kv_cache.prepare_cache()
        torch.cuda.empty_cache()

    def word_embedding(self, inputs_id):
        return F.embedding(inputs_id, self.embed_tokens)

    def lm(self, hidden_states):
        return F.linear(hidden_states, self.lm_head).float()

    def decode_forward(self, inputs_ids):
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
        
        hidden_states = self.layernorm(hidden_states, self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits
    def inference(self, inputs_ids, do_sample=False, temperature=0.6, top_p=0.95, top_k=20, ignore_eos=True):
        outputs_ids = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request
        
        # Prefilling
        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.time()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = self.sampling(logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
        outputs_ids.append(output_ids)
        self._debug_log_token_ids("prefill", output_ids)

        if self._debug_stop_after_prefill():
            torch.cuda.synchronize()
            prefill_end = time.time()
            self.prefill_latency_s = prefill_end - prefill_start
            print(colored(f"Prefilling latency: {round(self.prefill_latency_s, 4)} s", 'green'))
            return torch.cat(outputs_ids, dim=-1).tolist()

        self.move()

        torch.cuda.synchronize()
        prefill_end = time.time()
        self.prefill_latency_s = prefill_end - prefill_start
        print(colored(f"Prefilling latency: {round(self.prefill_latency_s, 4)} s", 'green'))

        # CUDAGraph Capture (if enabled)
        if self.attention_type == "AdaptiveIMI":
            self.kv_cache.capture_cuda_graph()
        
        stop_tokens = None
        end_of_text = None
        if not ignore_eos:
            stop_token_ids = self._collect_stop_token_ids()
            if stop_token_ids:
                token_id_dtype = torch.int64 if not do_sample else torch.int32
                stop_tokens = torch.tensor(
                    stop_token_ids,
                    device=inputs_ids.device,
                    dtype=token_id_dtype
                ).view(1, 1, -1)
                end_of_text = torch.zeros((self.batch_size, 1), dtype=torch.bool, device=inputs_ids.device)
                end_of_text |= (output_ids.unsqueeze(-1) == stop_tokens).any(-1)
        
        # Decoding
        print("Start decoding ...")
        decode_start = time.time()
        self.decode_latency_s = 0.0
        self.decode_steps = 0
        debug_max_decode_steps = self._debug_max_decode_steps()
        debug_decode_steps = 0

        if end_of_text is not None and end_of_text.all():
            print(colored("All sequences have reached EOS token, stop decoding.", 'yellow'))
        else:
            for _ in range(self.max_new_length-1):
                logits = self.decode_forward(inputs_ids=output_ids)
                output_ids = self.sampling(logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
                debug_decode_steps += 1
                self._debug_log_token_ids(f"decode_step_{debug_decode_steps}", output_ids)
                if end_of_text is not None:
                    end_of_text |= (output_ids.unsqueeze(-1) == stop_tokens).any(-1)
                    if end_of_text.all():
                        print(colored("All sequences have reached EOS token, stop decoding.", 'yellow'))
                        outputs_ids.append(output_ids)
                        break
                outputs_ids.append(output_ids)
                if debug_max_decode_steps is not None and debug_decode_steps >= debug_max_decode_steps:
                    break

        decode_end = time.time()
        self.decode_latency_s = decode_end - decode_start
        decode_steps = max(len(outputs_ids) - 1, 0)
        self.decode_steps = decode_steps
        if decode_steps == 0:
            print(colored(
                f"Decoding latency: {round(self.decode_latency_s, 4)} s (0 ms/step), Throughput: 0 tokens/s",
                'green'
            ))
        else:
            print(colored(
                f"Decoding latency: {round(self.decode_latency_s, 4)} s ({round(self.decode_latency_s * 1000 / decode_steps, 2)} ms/step), "
                f"Throughput: {round(self.batch_size * decode_steps / self.decode_latency_s, 2)} tokens/s",
                'green'
            ))

        self.end2end_latency_s = self.prefill_latency_s + self.decode_latency_s
        print(colored(f"End2End Latency: {round(self.end2end_latency_s, 4)} s\n", 'green'))
        
        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()
        
        return outputs_ids
    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config,
                 do_sample=False, temperature=0.6, top_p=0.95, top_k=20, ignore_eos=True,
                 prefill_bsz=1):
        """ LLM Inference.
        Args:
            attention_type: str, Full_Flash_Attn or AdaptiveIMI
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
            attn_config (dict): The deoding attention configuration.
            do_sample, temperature, top_p, top_k, ignore_eos: The sampling parameters.
            prefill_bsz (int): The batch size for prefill.
        """
        self.attention_type = attention_type

        bs, input_length = inputs_ids.shape
        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        assert self.input_length + self.max_new_length <= self.max_length, \
            f"Error: input_length({self.input_length}) + max_new_length({self.max_new_length}) exceeds max_length({self.max_length})"

        # compute valid start position for each sequence
        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks

        self.prefill_bsz = min(prefill_bsz, self.batch_size)
        # set prefill batch size to 1 if input sequences are not in the same length
        if not (valid_start == 0).all():
            self.prefill_bsz = 1

        print("Allocate GPU buffers and CPU pin memory ...")
        self.init_kv_cache(valid_start, attn_config)

        outputs = self.inference(
            inputs_ids, 
            do_sample=do_sample, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            ignore_eos=ignore_eos
        )

        return outputs
