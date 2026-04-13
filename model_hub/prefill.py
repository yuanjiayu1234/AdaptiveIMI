import os
import time

import torch


class PrefillMixin:
    def _get_prefill_prefix_kv_workspace(self, device: str, bsz: int, seq_len: int):
        workspace = self._prefill_prefix_kv_workspace.get(device)
        needs_alloc = (
            workspace is None
            or int(workspace["batch_size"]) < int(bsz)
            or int(workspace["seq_len"]) < int(seq_len)
        )

        if needs_alloc:
            key_buffer = torch.empty(
                (bsz, seq_len, self.num_key_value_heads, self.head_dim),
                dtype=self.dtype,
                device=device,
            )
            value_buffer = torch.empty_like(key_buffer)
            workspace = {
                "batch_size": int(bsz),
                "seq_len": int(seq_len),
                "key": key_buffer,
                "value": value_buffer,
            }
            self._prefill_prefix_kv_workspace[device] = workspace

        key_states = workspace["key"][:bsz, :seq_len, :, :]
        value_states = workspace["value"][:bsz, :seq_len, :, :]
        return key_states, value_states

    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        # print(f'Layer = {layer_idx}, start_bdx = {start_bdx}')

        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]

        pre_attn_total_start = self._start_prefill_gpu_segment()
        temp_hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)

        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        self._finish_prefill_gpu_total_segment(layer_idx, pre_attn_total_start)

        if self._debug_skip_prefill_kv_update() and self.attention_type == "AdaptiveIMI":
            valid_start = 0
            if hasattr(self.kv_cache, "valid_start_list"):
                valid_start = int(self.kv_cache.valid_start_list[start_bdx])
            key_states = key_states[:, valid_start:, :, :]
            value_states = value_states[:, valid_start:, :, :]
        else:
            key_states, value_states = self.kv_cache.prefill_update_kv_cache(
                query_states,
                key_states,
                value_states,
                layer_idx,
                start_bdx,
            )

        attn_total_start = self._start_prefill_gpu_segment()
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
        )
        if self.profile_prefill_gpu:
            end_event.record()
            self._record_prefill_gpu_attn_events(layer_idx, start_event, end_event)
        hidden_states += self.wo(temp_attn_out, layer, bsz, seq_len, dim)
        del temp_attn_out
        self._finish_prefill_gpu_total_segment(layer_idx, attn_total_start)

        if not (self._debug_skip_prefill_kv_update() and self.attention_type == "AdaptiveIMI"):
            self.kv_cache.sync(layer_idx, start_bdx)
        del query_states, key_states, value_states

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
    def _streaming_build_kv_chunk(
        self,
        hidden_cpu,
        chunk_start,
        chunk_end,
        layer,
        layer_idx,
        key_states,
        value_states,
    ):
        bsz = hidden_cpu.shape[0]
        device = layer.device
        chunk_len = chunk_end - chunk_start

        hidden_chunk = hidden_cpu[:, chunk_start:chunk_end, :].to(device, non_blocking=False)

        pre_attn_total_start = self._start_prefill_gpu_segment()
        temp_hidden_states = self.layernorm(
            hidden_chunk,
            layer.input_layernorm_variance_epsilon,
            layer.input_layernorm_weight,
        )
        query_chunk, key_chunk, value_chunk = self.wqkv(temp_hidden_states, layer)

        position_ids = self.position_ids[
            self.kv_cache.context + chunk_start : self.kv_cache.context + chunk_end
        ].unsqueeze(0).repeat(bsz, 1)
        query_chunk, key_chunk = self.apply_rotary_pos_emb(query_chunk, key_chunk, position_ids)

        query_chunk = query_chunk.view(bsz, chunk_len, self.num_heads, self.head_dim)
        key_chunk = key_chunk.view(bsz, chunk_len, self.num_key_value_heads, self.head_dim)
        value_chunk = value_chunk.view(bsz, chunk_len, self.num_key_value_heads, self.head_dim)

        key_states[:, chunk_start:chunk_end, :, :].copy_(key_chunk)
        value_states[:, chunk_start:chunk_end, :, :].copy_(value_chunk)
        self._finish_prefill_gpu_total_segment(layer_idx, pre_attn_total_start)

        return hidden_chunk, temp_hidden_states, query_chunk, key_chunk, value_chunk
    def _streaming_finish_chunk(
        self,
        layer,
        layer_idx,
        hidden_chunk,
        temp_hidden_states,
        query_chunk,
        key_chunk,
        value_chunk,
        key_states,
        value_states,
        valid_start,
        chunk_start,
        chunk_end,
        output_cpu,
        debug_align,
    ):
        bsz = hidden_chunk.shape[0]

        q_valid_start = max(valid_start, chunk_start)
        if q_valid_start < chunk_end:
            q_local_start = q_valid_start - chunk_start
            q_sub = query_chunk[:, q_local_start:, :, :]
            k_prefix = key_states[:, valid_start:chunk_end, :, :]
            v_prefix = value_states[:, valid_start:chunk_end, :, :]

            if not q_sub.is_contiguous():
                q_sub = q_sub.contiguous()
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
                q_sub,
                k_prefix,
                v_prefix,
                layer_idx,
            )

            if debug_align and layer_idx == 0:
                q_len = int(q_sub.shape[1])
                k_len = int(k_prefix.shape[1])
                q_offset = int(q_valid_start - valid_start)
                print(
                    "[IMI PREFILL ALIGN]"
                    f" layer={layer_idx} valid_start={valid_start}"
                    f" chunk=[{chunk_start},{chunk_end})"
                    f" q_valid_start={q_valid_start}"
                    f" q_len={q_len} k_len={k_len} q_offset={q_offset}",
                    flush=True,
                )

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
            del attn_chunk, q_sub, k_prefix, v_prefix

        post_attn_total_start = self._start_prefill_gpu_segment()
        hidden_states_norm = self.layernorm(
            hidden_chunk,
            layer.post_attention_layernorm_variance_epsilon,
            layer.post_attention_layernorm_weight,
        )
        mlp_out = self.mlp(hidden_states_norm, layer)
        hidden_chunk = hidden_chunk + mlp_out
        self._finish_prefill_gpu_total_segment(layer_idx, post_attn_total_start)

        output_cpu[:, chunk_start:chunk_end, :].copy_(hidden_chunk, non_blocking=False)
        del hidden_chunk, hidden_states_norm, mlp_out, temp_hidden_states, query_chunk, key_chunk, value_chunk
    def layer_prefill_chunked(self, layer_idx, start_bdx, hidden_cpu, output_cpu, seq_len, chunk_size):
        bsz = hidden_cpu.shape[0]
        layer = self.layers[layer_idx]
        device = layer.device
        kv_cache = getattr(self, "kv_cache", None)
        debug_align = os.getenv("IMI_DEBUG_PREFILL_ALIGN", "0") == "1"
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        self._log_streaming_progress(
            f"[IMI STREAM] layer={layer_idx} phase=start seq_len={seq_len} chunk_size={chunk_size} chunks={num_chunks}"
        )

        key_states, value_states = self._get_prefill_prefix_kv_workspace(device, bsz, seq_len)

        valid_start = 0
        if kv_cache is not None and hasattr(kv_cache, "valid_start_list"):
            valid_start = int(kv_cache.valid_start_list[start_bdx])

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(seq_len, chunk_start + chunk_size)
            chunk_idx = chunk_start // chunk_size
            chunk_t0 = time.perf_counter() if self._streaming_progress_enabled() else None
            hidden_chunk, temp_hidden_states, query_chunk, key_chunk, value_chunk = self._streaming_build_kv_chunk(
                hidden_cpu,
                chunk_start,
                chunk_end,
                layer,
                layer_idx,
                key_states,
                value_states,
            )
            self._streaming_finish_chunk(
                layer,
                layer_idx,
                hidden_chunk,
                temp_hidden_states,
                query_chunk,
                key_chunk,
                value_chunk,
                key_states,
                value_states,
                valid_start,
                chunk_start,
                chunk_end,
                output_cpu,
                debug_align,
            )
            if chunk_t0 is not None:
                elapsed_ms = (time.perf_counter() - chunk_t0) * 1000.0
                self._log_streaming_progress(
                    f"[IMI STREAM] layer={layer_idx} chunk={chunk_idx + 1}/{num_chunks} range=[{chunk_start},{chunk_end}) elapsed_ms={elapsed_ms:.2f}"
                )

        if kv_cache is not None and not (self._debug_skip_prefill_kv_update() and self.attention_type == "AdaptiveIMI"):
            finalize_t0 = time.perf_counter() if self._streaming_progress_enabled() else None
            kv_cache.prefill_update_kv_cache(None, key_states, value_states, layer_idx, start_bdx)
            kv_cache.sync(layer_idx, start_bdx)
            if finalize_t0 is not None:
                elapsed_ms = (time.perf_counter() - finalize_t0) * 1000.0
                self._log_streaming_progress(
                    f"[IMI STREAM] layer={layer_idx} phase=finalize_kv elapsed_ms={elapsed_ms:.2f}"
                )

        del key_states, value_states

        self._flush_prefill_gpu_layer_events(layer_idx)
        torch.cuda.synchronize(device)
        self._log_streaming_progress(f"[IMI STREAM] layer={layer_idx} phase=done")
    def prefill_forward_chunked(self, inputs_ids):
        self._reset_prefill_gpu_profile()
        bsz, seq_len = inputs_ids.shape
        device = self.layers[0].device
        chunk_size = max(int(self.prefill_attn_chunk_size), 1)
        pin_hidden = self._should_pin_streaming_hidden(bsz, seq_len)

        self._log_streaming_progress(
            f"[IMI STREAM] prefill_start seq_len={seq_len} chunk_size={chunk_size} pin_hidden={int(pin_hidden)}"
        )

        hidden_cpu = torch.empty(
            (bsz, seq_len, self.hidden_size),
            dtype=self.dtype,
            pin_memory=pin_hidden,
        )
        next_cpu = torch.empty(
            (bsz, seq_len, self.hidden_size),
            dtype=self.dtype,
            pin_memory=pin_hidden,
        )

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(seq_len, chunk_start + chunk_size)
            ids_chunk = inputs_ids[:, chunk_start:chunk_end].to(device, non_blocking=True)
            embed_chunk = self.word_embedding(ids_chunk)
            hidden_cpu[:, chunk_start:chunk_end, :].copy_(embed_chunk, non_blocking=False)
        torch.cuda.synchronize(device)
        self._log_streaming_progress("[IMI STREAM] embedding_done")

        for ldx in range(self.num_layers):
            self.layer_prefill_chunked(ldx, 0, hidden_cpu, next_cpu, seq_len, chunk_size)
            hidden_cpu, next_cpu = next_cpu, hidden_cpu
            if self.num_gpus > 1:
                next_device = self.layer_mapping[str(ldx + 1)] if str(ldx + 1) in self.layer_mapping else self.layer_mapping[str(0)]
                self.position_ids = self.position_ids.to(next_device)
                self.cos_sin_cache = self.cos_sin_cache.to(next_device)

        self._log_streaming_progress("[IMI STREAM] all_layers_done")

        last_hidden = hidden_cpu[:, -1:, :].to(device, non_blocking=False)
        last_hidden = self.layernorm(last_hidden, self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden)
        return logits
    def prefill_forward(self, inputs_ids):
        self._reset_prefill_gpu_profile()
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        if self.enable_prefill_attn_chunk:
            return self.prefill_forward_chunked(inputs_ids)

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
