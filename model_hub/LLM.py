import os
import time
import torch
import flashinfer
from termcolor import colored


class LLM:
    """
    A class representing the LLM (currently support Llama and Qwen).
    """

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
        self.streaming_prefill_threshold = 98304
        self.profile_prefill_gpu = os.getenv("IMI_PROFILE_PREFILL_GPU", "0") == "1"
        self.prefill_gpu_layer_rows = []
        self._prefill_gpu_attn_events = {}
        self._prefill_gpu_total_events = {}

    def apply_prefill_config(self, model_config: dict) -> None:
        if not model_config:
            return
        prefill_cfg = model_config.get("prefill") or {}
        if not prefill_cfg and isinstance(model_config, dict):
            attn_cfg = None
            if hasattr(self, "attention_type") and self.attention_type:
                attn_cfg = model_config.get(self.attention_type)
            if attn_cfg is None:
                attn_cfg = model_config.get("AdaptiveIMI") or model_config.get("RetroInfer")
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

        streaming_prefill_threshold = prefill_cfg.get("streaming_prefill_threshold")
        if streaming_prefill_threshold is not None:
            streaming_prefill_threshold = int(streaming_prefill_threshold)
            if streaming_prefill_threshold >= 0:
                self.streaming_prefill_threshold = streaming_prefill_threshold

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


    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        # print(f'Layer = {layer_idx}, start_bdx = {start_bdx}')

        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]

        pre_attn_total_start = self._start_prefill_gpu_segment()
        temp_hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)

        use_prefill_attn_chunk = (
            self.enable_prefill_attn_chunk
            and self.prefill_method == "full"
        )

        use_imi_streaming_prefill = (
            use_prefill_attn_chunk
            and self.attention_type == "AdaptiveIMI"
            and hasattr(self.kv_cache, "set_prefill_stream_mode")
            and hasattr(self.kv_cache, "begin_prefill_stream")
            and hasattr(self.kv_cache, "submit_prefill_stream_chunk")
        )

        if hasattr(self.kv_cache, "set_prefill_stream_mode"):
            self.kv_cache.set_prefill_stream_mode(layer_idx, use_imi_streaming_prefill)

        attn_chunk_size = self.prefill_attn_chunk_size if use_prefill_attn_chunk else None
        if use_imi_streaming_prefill and attn_chunk_size is not None:
            imi_chunk_size = getattr(self.kv_cache, "prefill_chunk_size", None)
            if imi_chunk_size:
                attn_chunk_size = min(attn_chunk_size, imi_chunk_size)

        if use_imi_streaming_prefill and attn_chunk_size is not None:
            self._finish_prefill_gpu_total_segment(layer_idx, pre_attn_total_start)
            valid_start = self.kv_cache.valid_start_list[start_bdx]
            debug_align = os.getenv("IMI_DEBUG_PREFILL_ALIGN", "0") == "1"
            kv_dtype = temp_hidden_states.dtype
            key_states = torch.empty(
                (bsz, seq_len, self.num_key_value_heads, self.head_dim),
                dtype=kv_dtype,
                device=temp_hidden_states.device,
            )
            value_states = torch.empty_like(key_states)

            self.kv_cache.begin_prefill_stream(
                layer_idx,
                start_bdx,
                key_states,
                value_states,
                attn_chunk_size,
            )

            for chunk_start in range(0, seq_len, attn_chunk_size):
                chunk_end = min(seq_len, chunk_start + attn_chunk_size)
                hidden_chunk = temp_hidden_states[:, chunk_start:chunk_end, :]

                qkv_total_start = self._start_prefill_gpu_segment()
                query_chunk, key_chunk, value_chunk = self.wqkv(hidden_chunk, layer)

                position_ids = self.position_ids[
                    self.kv_cache.context + chunk_start : self.kv_cache.context + chunk_end
                ].unsqueeze(0).repeat(bsz, 1)
                query_chunk, key_chunk = self.apply_rotary_pos_emb(query_chunk, key_chunk, position_ids)

                query_chunk = query_chunk.view(bsz, chunk_end - chunk_start, self.num_heads, self.head_dim)
                key_chunk = key_chunk.view(bsz, chunk_end - chunk_start, self.num_key_value_heads, self.head_dim)
                value_chunk = value_chunk.view(bsz, chunk_end - chunk_start, self.num_key_value_heads, self.head_dim)

                key_states[:, chunk_start:chunk_end, :, :].copy_(key_chunk)
                value_states[:, chunk_start:chunk_end, :, :].copy_(value_chunk)
                self._finish_prefill_gpu_total_segment(layer_idx, qkv_total_start)

                self.kv_cache.submit_prefill_stream_chunk(
                    layer_idx,
                    start_bdx,
                    key_states,
                    value_states,
                    chunk_start,
                    chunk_end,
                    chunk_keys=key_chunk,
                    chunk_values=value_chunk,
                )
                torch.cuda.synchronize()

                q_valid_start = max(valid_start, chunk_start)
                if q_valid_start < chunk_end:
                    if debug_align and layer_idx == 0:
                        q_len = int(chunk_end - q_valid_start)
                        k_len = int(chunk_end - valid_start)
                        q_offset = int(q_valid_start - valid_start)
                        print(
                            "[IMI PREFILL ALIGN]"
                            f" layer={layer_idx} valid_start={valid_start}"
                            f" chunk=[{chunk_start},{chunk_end})"
                            f" q_valid_start={q_valid_start}"
                            f" q_len={q_len} k_len={k_len} q_offset={q_offset}",
                            flush=True,
                        )
                    attn_total_start = self._start_prefill_gpu_segment()
                    q_local_start = q_valid_start - chunk_start
                    q_chunk = query_chunk[:, q_local_start:, :, :]
                    k_cache = key_states[:, valid_start:chunk_end, :, :]
                    v_cache = value_states[:, valid_start:chunk_end, :, :]
                    k_new = key_chunk[:, q_local_start:, :, :]
                    v_new = value_chunk[:, q_local_start:, :, :]
                    if not q_chunk.is_contiguous():
                        q_chunk = q_chunk.contiguous()
                    if not k_cache.is_contiguous():
                        k_cache = k_cache.contiguous()
                    if not v_cache.is_contiguous():
                        v_cache = v_cache.contiguous()
                    if not k_new.is_contiguous():
                        k_new = k_new.contiguous()
                    if not v_new.is_contiguous():
                        v_new = v_new.contiguous()
                    start_event = None
                    end_event = None
                    if self.profile_prefill_gpu:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                    cache_seqlens = torch.full(
                        (bsz,),
                        int(q_valid_start - valid_start),
                        device=q_chunk.device,
                        dtype=torch.int32,
                    )
                    attn_chunk = self.prefill_attention(
                        q_chunk,
                        k_cache,
                        v_cache,
                        layer_idx,
                        k_new=k_new,
                        v_new=v_new,
                        cache_seqlens=cache_seqlens,
                    )
                    if debug_align and layer_idx == 0:
                        q_len = q_chunk.shape[1]
                        k_len = k_cache.shape[1]
                        if k_len > q_len:
                            pad_len = k_len - q_len
                            q_pad = torch.zeros(
                                (bsz, pad_len, q_chunk.shape[2], q_chunk.shape[3]),
                                device=q_chunk.device,
                                dtype=q_chunk.dtype,
                            )
                            q_ref = torch.cat([q_pad, q_chunk], dim=1)
                            ref_out = self.prefill_attention(
                                q_ref,
                                k_cache,
                                v_cache,
                                layer_idx,
                            )
                            ref_tail = ref_out[:, -q_len:, :, :]
                            diff = (ref_tail - attn_chunk).abs()
                            print(
                                "[IMI PREFILL DIFF]"
                                f" layer={layer_idx} chunk_start={chunk_start}"
                                f" max={diff.max().item():.6f} mean={diff.mean().item():.6f}",
                                flush=True,
                            )
                            del q_pad, q_ref, ref_out, ref_tail, diff
                    if self.profile_prefill_gpu:
                        end_event.record()
                        self._record_prefill_gpu_attn_events(layer_idx, start_event, end_event)
                    hidden_states[:, q_valid_start:chunk_end, :] += self.wo(
                        attn_chunk,
                        layer,
                        bsz,
                        chunk_end - q_valid_start,
                        dim,
                    )
                    self._finish_prefill_gpu_total_segment(layer_idx, attn_total_start)
                    del attn_chunk, q_chunk, k_cache, v_cache, k_new, v_new, cache_seqlens

                del query_chunk, key_chunk, value_chunk

            self.kv_cache.prefill_update_kv_cache(key_states, key_states, value_states, layer_idx, start_bdx)
            self.kv_cache.sync(layer_idx, start_bdx)
            del key_states, value_states
        else:
            query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
            query_states, key_states = self.position_embedd(query_states, key_states)

            query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim) # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
            key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
            value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
            self._finish_prefill_gpu_total_segment(layer_idx, pre_attn_total_start)

            key_states, value_states = self.kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)

            attn_total_start = self._start_prefill_gpu_segment()
            if use_prefill_attn_chunk and attn_chunk_size is not None:
                key_seq_len = key_states.shape[1]
                key_offset = max(seq_len - key_seq_len, 0)
                for chunk_start in range(0, seq_len, attn_chunk_size):
                    chunk_end = min(seq_len, chunk_start + attn_chunk_size)
                    q_valid_start = max(chunk_start, key_offset)
                    if q_valid_start >= chunk_end:
                        continue
                    prefix_end = min(key_seq_len, chunk_end - key_offset)
                    if prefix_end <= 0:
                        continue

                    q_chunk = query_states[:, q_valid_start:chunk_end, :, :]
                    k_prefix = key_states[:, :prefix_end, :, :]
                    v_prefix = value_states[:, :prefix_end, :, :]
                    if not q_chunk.is_contiguous():
                        q_chunk = q_chunk.contiguous()
                    if not k_prefix.is_contiguous():
                        k_prefix = k_prefix.contiguous()
                    if not v_prefix.is_contiguous():
                        v_prefix = v_prefix.contiguous()
                    start_event = None
                    end_event = None
                    if self.profile_prefill_gpu:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                    attn_chunk = self.prefill_attention(
                        q_chunk,
                        k_prefix,
                        v_prefix,
                        layer_idx,
                    )
                    if self.profile_prefill_gpu:
                        end_event.record()
                        self._record_prefill_gpu_attn_events(layer_idx, start_event, end_event)
                    hidden_states[:, q_valid_start:chunk_end, :] += self.wo(
                        attn_chunk,
                        layer,
                        bsz,
                        chunk_end - q_valid_start,
                        dim,
                    )
                    del attn_chunk, q_chunk, k_prefix, v_prefix
            else:
                start_event = None
                end_event = None
                if self.profile_prefill_gpu:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                temp_attn_out = self.prefill_attention(
                    query_states,
                    key_states,
                    value_states,
                    layer_idx,
                    chunk_size=attn_chunk_size,
                )
                if self.profile_prefill_gpu:
                    end_event.record()
                    self._record_prefill_gpu_attn_events(layer_idx, start_event, end_event)
                hidden_states += self.wo(temp_attn_out, layer, bsz, seq_len, dim)
                del temp_attn_out
            self._finish_prefill_gpu_total_segment(layer_idx, attn_total_start)

            self.kv_cache.sync(layer_idx, start_bdx)
            del query_states, key_states, value_states

        if hasattr(self.kv_cache, "set_prefill_stream_mode"):
            self.kv_cache.set_prefill_stream_mode(layer_idx, False)
        del temp_hidden_states

        post_attn_total_start = self._start_prefill_gpu_segment()
        hidden_states_norm = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        # faster when split batches
        for batch_idx in range(0, bsz, 1):
            # chunk for lower memory comsumption, especially for 1M context
            for start_idx in range(0, seq_len, self.prefill_chunk_size):
                end_idx = min(seq_len, start_idx + self.prefill_chunk_size)
                residual_chunk = hidden_states[batch_idx:batch_idx+1, start_idx:end_idx, :]
                mlp_out = self.mlp(hidden_states_norm[batch_idx:batch_idx+1, start_idx:end_idx, :], layer)
                hidden_states[batch_idx:batch_idx+1, start_idx:end_idx, :] = residual_chunk + mlp_out
                del mlp_out
        del hidden_states_norm
        self._finish_prefill_gpu_total_segment(layer_idx, post_attn_total_start)
        self._flush_prefill_gpu_layer_events(layer_idx)

        return hidden_states


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


    def _should_stream_prefill(self, seq_len: int, bsz: int) -> bool:
        if bsz != 1:
            return False
        if self.attention_type == "AdaptiveIMI":
            if self.num_gpus > 1:
                return False
            threshold = int(getattr(self, "streaming_prefill_threshold", 98304))
            if seq_len >= threshold:
                return True
            return not getattr(self.kv_cache, "allocated", True)
        # For Full_Flash_Attn: stream prefill when KV cache is not pre-allocated on GPU
        if self.attention_type in ("Full_Flash_Attn", "Full_Flash_Attn_Offload"):
            return not getattr(self.kv_cache, "allocated", True)
        return False


    def layer_prefill_streaming(self, layer_idx, start_bdx, hidden_cpu, output_cpu, seq_len, chunk_size):
        bsz = hidden_cpu.shape[0]
        layer = self.layers[layer_idx]
        device = layer.device
        debug_align = os.getenv("IMI_DEBUG_PREFILL_ALIGN", "0") == "1"

        kv_cache = getattr(self, "kv_cache", None)
        use_imi_streaming_prefill = (
            kv_cache is not None
            and self.attention_type == "AdaptiveIMI"
            and hasattr(kv_cache, "set_prefill_stream_mode")
            and hasattr(kv_cache, "begin_prefill_stream")
            and hasattr(kv_cache, "submit_prefill_stream_chunk")
        )

        if hasattr(kv_cache, "set_prefill_stream_mode"):
            kv_cache.set_prefill_stream_mode(layer_idx, use_imi_streaming_prefill)

        key_states = torch.empty(
            (bsz, seq_len, self.num_key_value_heads, self.head_dim),
            dtype=self.dtype,
            device=device,
        )
        value_states = torch.empty_like(key_states)

        if use_imi_streaming_prefill:
            kv_cache.begin_prefill_stream(layer_idx, start_bdx, key_states, value_states, chunk_size)

        valid_start = 0
        if kv_cache is not None and hasattr(kv_cache, "valid_start_list"):
            valid_start = int(kv_cache.valid_start_list[start_bdx])

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(seq_len, chunk_start + chunk_size)
            chunk_len = chunk_end - chunk_start

            hidden_chunk = hidden_cpu[:, chunk_start:chunk_end, :].to(device, non_blocking=True)
            total_start_event = self._start_prefill_gpu_segment()
            temp_hidden_states = self.layernorm(
                hidden_chunk,
                layer.input_layernorm_variance_epsilon,
                layer.input_layernorm_weight,
            )

            query_chunk, key_chunk, value_chunk = self.wqkv(temp_hidden_states, layer)

            position_ids = self.position_ids[
                kv_cache.context + chunk_start : kv_cache.context + chunk_end
            ].unsqueeze(0).repeat(bsz, 1)
            query_chunk, key_chunk = self.apply_rotary_pos_emb(query_chunk, key_chunk, position_ids)

            query_chunk = query_chunk.view(bsz, chunk_len, self.num_heads, self.head_dim)
            key_chunk = key_chunk.view(bsz, chunk_len, self.num_key_value_heads, self.head_dim)
            value_chunk = value_chunk.view(bsz, chunk_len, self.num_key_value_heads, self.head_dim)

            key_states[:, chunk_start:chunk_end, :, :].copy_(key_chunk)
            value_states[:, chunk_start:chunk_end, :, :].copy_(value_chunk)
            self._finish_prefill_gpu_total_segment(layer_idx, total_start_event)

            if use_imi_streaming_prefill:
                kv_cache.submit_prefill_stream_chunk(
                    layer_idx,
                    start_bdx,
                    key_states,
                    value_states,
                    chunk_start,
                    chunk_end,
                    chunk_keys=key_chunk,
                    chunk_values=value_chunk,
                )

            total_start_event = self._start_prefill_gpu_segment()
            q_valid_start = max(valid_start, chunk_start)
            if q_valid_start < chunk_end:
                if debug_align and layer_idx == 0:
                    q_len = int(chunk_end - q_valid_start)
                    k_len = int(chunk_end - valid_start)
                    q_offset = int(q_valid_start - valid_start)
                    print(
                        "[IMI PREFILL ALIGN]"
                        f" layer={layer_idx} valid_start={valid_start}"
                        f" chunk=[{chunk_start},{chunk_end})"
                        f" q_valid_start={q_valid_start}"
                        f" q_len={q_len} k_len={k_len} q_offset={q_offset}",
                        flush=True,
                    )
                q_local_start = q_valid_start - chunk_start
                q_chunk = query_chunk[:, q_local_start:, :, :]
                k_cache = key_states[:, valid_start:chunk_end, :, :]
                v_cache = value_states[:, valid_start:chunk_end, :, :]
                k_new = key_chunk[:, q_local_start:, :, :]
                v_new = value_chunk[:, q_local_start:, :, :]
                if not q_chunk.is_contiguous():
                    q_chunk = q_chunk.contiguous()
                if not k_cache.is_contiguous():
                    k_cache = k_cache.contiguous()
                if not v_cache.is_contiguous():
                    v_cache = v_cache.contiguous()
                if not k_new.is_contiguous():
                    k_new = k_new.contiguous()
                if not v_new.is_contiguous():
                    v_new = v_new.contiguous()
                start_event = None
                end_event = None
                if self.profile_prefill_gpu:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                cache_seqlens = torch.full(
                    (bsz,),
                    int(q_valid_start - valid_start),
                    device=q_chunk.device,
                    dtype=torch.int32,
                )
                attn_chunk = self.prefill_attention(
                    q_chunk,
                    k_cache,
                    v_cache,
                    layer_idx,
                    k_new=k_new,
                    v_new=v_new,
                    cache_seqlens=cache_seqlens,
                )
                if debug_align and layer_idx == 0:
                    q_len = q_chunk.shape[1]
                    k_len = k_cache.shape[1]
                    if k_len > q_len:
                        pad_len = k_len - q_len
                        q_pad = torch.zeros(
                            (bsz, pad_len, q_chunk.shape[2], q_chunk.shape[3]),
                            device=q_chunk.device,
                            dtype=q_chunk.dtype,
                        )
                        q_ref = torch.cat([q_pad, q_chunk], dim=1)
                        ref_out = self.prefill_attention(
                            q_ref,
                            k_cache,
                            v_cache,
                            layer_idx,
                        )
                        ref_tail = ref_out[:, -q_len:, :, :]
                        diff = (ref_tail - attn_chunk).abs()
                        print(
                            "[IMI PREFILL DIFF]"
                            f" layer={layer_idx} chunk_start={chunk_start}"
                            f" max={diff.max().item():.6f} mean={diff.mean().item():.6f}",
                            flush=True,
                        )
                        del q_pad, q_ref, ref_out, ref_tail, diff
                if self.profile_prefill_gpu:
                    end_event.record()
                    self._record_prefill_gpu_attn_events(layer_idx, start_event, end_event)
                hidden_chunk[:, q_local_start:, :] += self.wo(
                    attn_chunk,
                    layer,
                    bsz,
                    chunk_end - q_valid_start,
                    self.hidden_size,
                )
                del attn_chunk, q_chunk, k_cache, v_cache, k_new, v_new, cache_seqlens

            hidden_states_norm = self.layernorm(
                hidden_chunk,
                layer.post_attention_layernorm_variance_epsilon,
                layer.post_attention_layernorm_weight,
            )
            mlp_out = self.mlp(hidden_states_norm, layer)
            hidden_chunk = hidden_chunk + mlp_out
            self._finish_prefill_gpu_total_segment(layer_idx, total_start_event)
            output_cpu[:, chunk_start:chunk_end, :].copy_(hidden_chunk, non_blocking=True)
            del hidden_chunk, hidden_states_norm, mlp_out, temp_hidden_states

        if kv_cache is not None:
            kv_cache.prefill_update_kv_cache(key_states, key_states, value_states, layer_idx, start_bdx)
            kv_cache.sync(layer_idx, start_bdx)

        if hasattr(kv_cache, "set_prefill_stream_mode"):
            kv_cache.set_prefill_stream_mode(layer_idx, False)

        self._flush_prefill_gpu_layer_events(layer_idx)
        torch.cuda.synchronize(device)
        del key_states, value_states


    def prefill_forward_streaming(self, inputs_ids):
        self._reset_prefill_gpu_profile()
        bsz, seq_len = inputs_ids.shape
        device = self.layers[0].device
        chunk_size = max(int(self.prefill_chunk_size), 1)

        hidden_cpu = torch.empty(
            (bsz, seq_len, self.hidden_size),
            dtype=self.dtype,
            pin_memory=True,
        )
        next_cpu = torch.empty_like(hidden_cpu)

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(seq_len, chunk_start + chunk_size)
            ids_chunk = inputs_ids[:, chunk_start:chunk_end].to(device, non_blocking=True)
            embed_chunk = self.word_embedding(ids_chunk)
            hidden_cpu[:, chunk_start:chunk_end, :].copy_(embed_chunk, non_blocking=True)
        torch.cuda.synchronize(device)

        for ldx in range(self.num_layers):
            self.layer_prefill_streaming(ldx, 0, hidden_cpu, next_cpu, seq_len, chunk_size)
            hidden_cpu, next_cpu = next_cpu, hidden_cpu
            if self.num_gpus > 1:
                next_device = self.layer_mapping[str(ldx + 1)] if str(ldx + 1) in self.layer_mapping else self.layer_mapping[str(0)]
                self.position_ids = self.position_ids.to(next_device)
                self.cos_sin_cache = self.cos_sin_cache.to(next_device)

        last_hidden = hidden_cpu[:, -1:, :].to(device, non_blocking=True)
        last_hidden = self.layernorm(last_hidden, self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden)
        return logits


    def prefill_forward(self, inputs_ids):
        self._reset_prefill_gpu_profile()
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        if self._should_stream_prefill(seq_len, bsz):
            return self.prefill_forward_streaming(inputs_ids)

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device).contiguous()
        for start_bdx in range(0, bsz, self.prefill_bsz):
            end_bdx = min(bsz, start_bdx + self.prefill_bsz)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])  # [prefill_batch_size, seq_len, hidden_size]

            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
        
        last_hidden_states = self.layernorm(last_hidden_states, self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)
        
        return logits
        

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


    def sampling(self, logits, do_sample=False, temperature=0.6, top_p=0.95, top_k=20):
        if not do_sample:
            output_ids = logits.argmax(dim=-1)  # [bsz, 1], torch.int64
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)  # [bsz, 1, vocab_size]
            probs = probs.squeeze(1) # [bsz, vocab_size]
            if top_k != 0:
                output_ids = flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, top_p=top_p, top_k=top_k)
            else:
                output_ids = flashinfer.sampling.top_p_sampling_from_probs(probs, top_p=top_p)
            output_ids = output_ids.unsqueeze(1) # [bsz, 1], torch.int32

        return output_ids


    def _collect_stop_token_ids(self):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return []
        stop_token_ids = []
        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            for token_id in eos_token_id:
                if token_id is not None:
                    stop_token_ids.append(token_id)
        elif eos_token_id is not None:
            stop_token_ids.append(eos_token_id)
        for token in ("<|eot_id|>", "<|end_of_turn|>", "<|eom_id|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None:
                continue
            if tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id:
                continue
            if token_id not in stop_token_ids:
                stop_token_ids.append(token_id)
        return stop_token_ids


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
        self.move()

        torch.cuda.synchronize()
        prefill_end = time.time()
        self.prefill_latency_s = prefill_end - prefill_start
        print(colored(f"Prefilling latency: {round(self.prefill_latency_s, 4)} s", 'green'))

        # CUDAGraph Capture (if enabled)
        if self.attention_type in ("RetroInfer", "AdaptiveIMI"):
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

        if end_of_text is not None and end_of_text.all():
            print(colored("All sequences have reached EOS token, stop decoding.", 'yellow'))
        else:
            for _ in range(self.max_new_length-1):
                logits = self.decode_forward(inputs_ids=output_ids)
                output_ids = self.sampling(logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
                if end_of_text is not None:
                    end_of_text |= (output_ids.unsqueeze(-1) == stop_tokens).any(-1)
                    if end_of_text.all():
                        print(colored("All sequences have reached EOS token, stop decoding.", 'yellow'))
                        break
                outputs_ids.append(output_ids)

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
                 prefill_bsz=1, prefill_method="full"):
        """ LLM Inference.
        Args:
            attention_type: str, Full_Flash_Attn or RetroInfer
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
            attn_config (dict): The deoding attention configuration.
            do_sample, temperature, top_p, top_k, ignore_eos: The sampling parameters.
            prefill_bsz (int): The batch size for prefill.
            prefill_method (str): The method for prefill, support full and xattn.
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
        self.prefill_method = prefill_method
        # set prefill batch size to 1 and prefill method to full attention if input sequences are not in the same length
        if not (valid_start == 0).all():
            self.prefill_bsz = 1
            self.prefill_method = "full"

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
