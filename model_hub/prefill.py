import time

import torch


class PrefillMixin:
    def _prefill_layer_stages(self):
        stages = []
        start = 0
        while start < self.num_layers:
            device = self.layer_mapping[str(start)]
            end = start + 1
            while end < self.num_layers and self.layer_mapping[str(end)] == device:
                end += 1
            stages.append((start, end, device))
            start = end
        return stages

    def _format_prefill_stages(self, stages):
        return ",".join(f"{start}-{end - 1}@{device}" for start, end, device in stages)

    def _stage_hidden_required_bytes(self, bsz: int, seq_len: int) -> int:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return 2 * int(bsz) * int(seq_len) * int(self.hidden_size) * int(element_size)

    def _should_use_stage_gpu_prefill(self, stages, bsz: int, seq_len: int) -> bool:
        mode = str(getattr(self, "streaming_hidden_device", "auto")).lower()
        if mode in ("cpu", "host", "offload"):
            return False
        if not all(str(device).startswith("cuda") for _start, _end, device in stages):
            return False

        required_bytes = self._stage_hidden_required_bytes(bsz, seq_len)
        if mode in ("gpu", "cuda", "device"):
            return True
        if mode != "auto":
            return False

        max_bytes = int(float(getattr(self, "streaming_hidden_gpu_limit_gb", 8.0)) * (1024 ** 3))
        if required_bytes > max_bytes:
            return False

        # Each stage needs source+destination hidden buffers while it is active.
        # Check every distinct layer device up front; if one is tight, keep the
        # older CPU-staged path instead of risking a late OOM.
        for device in sorted({device for _start, _end, device in stages}):
            try:
                free_bytes, _total_bytes = torch.cuda.mem_get_info(torch.device(device))
            except Exception:
                return False
            if required_bytes > int(free_bytes * 0.5):
                return False
        return True

    def _prefill_chunked_batch_ranges(self, bsz: int):
        max_bsz = max(1, min(int(getattr(self, "prefill_bsz", bsz) or bsz), int(bsz)))
        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is None or not hasattr(kv_cache, "valid_start_list"):
            valid_starts = [0] * int(bsz)
        else:
            valid_starts = [int(value) for value in kv_cache.valid_start_list[:bsz]]

        ranges = []
        start = 0
        while start < bsz:
            valid_start = valid_starts[start]
            end_limit = min(bsz, start + max_bsz)
            end = start + 1
            while end < end_limit and valid_starts[end] == valid_start:
                end += 1
            ranges.append((start, end, valid_start))
            start = end
        return ranges

    def _get_prefill_prefix_kv_workspace(self, device: str, bsz: int, seq_len: int):
        pool = self._prefill_prefix_kv_workspace.get(device)
        if pool is None:
            pool = []
            self._prefill_prefix_kv_workspace[device] = pool
        elif isinstance(pool, dict):
            pool = [pool]
            self._prefill_prefix_kv_workspace[device] = pool

        kv_cache = getattr(self, "kv_cache", None)
        available = getattr(kv_cache, "prefill_kv_workspace_available", None)
        for workspace in pool:
            if int(workspace["batch_size"]) < int(bsz) or int(workspace["seq_len"]) < int(seq_len):
                continue
            if available is not None and not available(workspace["key"], workspace["value"]):
                continue
            key_states = workspace["key"][:bsz, :seq_len, :, :]
            value_states = workspace["value"][:bsz, :seq_len, :, :]
            return key_states, value_states

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
        pool.append(workspace)

        key_states = workspace["key"][:bsz, :seq_len, :, :]
        value_states = workspace["value"][:bsz, :seq_len, :, :]
        return key_states, value_states

    def _streaming_build_kv_chunk(
        self,
        hidden_cpu,
        chunk_start,
        chunk_end,
        layer,
        key_states,
        value_states,
    ):
        bsz = hidden_cpu.shape[0]
        device = layer.device
        chunk_len = chunk_end - chunk_start

        hidden_chunk = hidden_cpu[:, chunk_start:chunk_end, :].to(device, non_blocking=True)

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

        return hidden_chunk, query_chunk
    def _streaming_finish_chunk(
        self,
        layer,
        layer_idx,
        hidden_chunk,
        query_chunk,
        key_states,
        value_states,
        valid_start,
        chunk_start,
        chunk_end,
        output_cpu,
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

            attn_chunk = self.prefill_attention(
                q_sub,
                k_prefix,
                v_prefix,
                layer_idx,
            )

            hidden_chunk[:, q_local_start:, :] += self.wo(
                attn_chunk,
                layer,
                bsz,
                chunk_end - q_valid_start,
                self.hidden_size,
            )
            del attn_chunk, q_sub, k_prefix, v_prefix

        hidden_states_norm = self.layernorm(
            hidden_chunk,
            layer.post_attention_layernorm_variance_epsilon,
            layer.post_attention_layernorm_weight,
        )
        mlp_out = self.mlp(hidden_states_norm, layer)
        hidden_chunk = hidden_chunk + mlp_out

        output_cpu[:, chunk_start:chunk_end, :].copy_(hidden_chunk, non_blocking=True)
        del hidden_chunk, hidden_states_norm, mlp_out, query_chunk
    def layer_prefill_chunked(self, layer_idx, start_bdx, hidden_cpu, output_cpu, seq_len, chunk_size):
        bsz = hidden_cpu.shape[0]
        layer = self.layers[layer_idx]
        device = layer.device
        self._activate_device_runtime_tensors(device)
        kv_cache = getattr(self, "kv_cache", None)
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        self._log_streaming_progress(
            f"[IMI STREAM] layer={layer_idx} phase=start seq_len={seq_len} chunk_size={chunk_size} chunks={num_chunks}"
        )

        key_states, value_states = self._get_prefill_prefix_kv_workspace(device, bsz, seq_len)

        valid_start = 0
        if kv_cache is not None and hasattr(kv_cache, "valid_start_list"):
            valid_starts = [int(value) for value in kv_cache.valid_start_list[start_bdx : start_bdx + bsz]]
            valid_start = valid_starts[0]
            if any(value != valid_start for value in valid_starts):
                raise ValueError(
                    "chunked prefill layer received mixed valid_start values; "
                    "prefill_forward_chunked must group ragged batches before attention"
                )

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(seq_len, chunk_start + chunk_size)
            chunk_idx = chunk_start // chunk_size
            breakdown_enabled = getattr(self, "breakdown_enabled", False)
            chunk_t0 = time.perf_counter() if self._streaming_progress_enabled() else None
            build_t0 = time.perf_counter() if breakdown_enabled else None
            hidden_chunk, query_chunk = self._streaming_build_kv_chunk(
                hidden_cpu,
                chunk_start,
                chunk_end,
                layer,
                key_states,
                value_states,
            )
            if build_t0 is not None:
                self.prefill_kv_build_wall_ms += (time.perf_counter() - build_t0) * 1000.0
            if (
                kv_cache is not None
                and hasattr(kv_cache, "prefill_submit_kv_chunk")
                and not (self._debug_skip_prefill_kv_update() and self.attention_type == "AdaptiveIMI")
            ):
                submit_t0 = time.perf_counter() if breakdown_enabled else None
                kv_cache.prefill_submit_kv_chunk(
                    key_states,
                    value_states,
                    layer_idx,
                    start_bdx,
                    chunk_start,
                    chunk_end,
                )
                if submit_t0 is not None:
                    self.prefill_index_submit_block_ms += (time.perf_counter() - submit_t0) * 1000.0
            finish_t0 = time.perf_counter() if breakdown_enabled else None
            self._streaming_finish_chunk(
                layer,
                layer_idx,
                hidden_chunk,
                query_chunk,
                key_states,
                value_states,
                valid_start,
                chunk_start,
                chunk_end,
                output_cpu,
            )
            if finish_t0 is not None:
                self.prefill_layer_compute_wall_ms += (time.perf_counter() - finish_t0) * 1000.0
            if chunk_t0 is not None:
                elapsed_ms = (time.perf_counter() - chunk_t0) * 1000.0
                self._log_streaming_progress(
                    f"[IMI STREAM] layer={layer_idx} chunk={chunk_idx + 1}/{num_chunks} range=[{chunk_start},{chunk_end}) elapsed_ms={elapsed_ms:.2f}"
                )

        if kv_cache is not None and not (self._debug_skip_prefill_kv_update() and self.attention_type == "AdaptiveIMI"):
            finalize_t0 = time.perf_counter() if self._streaming_progress_enabled() else None
            breakdown_finalize_t0 = time.perf_counter() if getattr(self, "breakdown_enabled", False) else None
            kv_cache.prefill_update_kv_cache(None, key_states, value_states, layer_idx, start_bdx)
            kv_cache.sync(layer_idx, start_bdx)
            if breakdown_finalize_t0 is not None:
                self.prefill_index_finalize_block_ms += (time.perf_counter() - breakdown_finalize_t0) * 1000.0
            if finalize_t0 is not None:
                elapsed_ms = (time.perf_counter() - finalize_t0) * 1000.0
                self._log_streaming_progress(
                    f"[IMI STREAM] layer={layer_idx} phase=finalize_kv elapsed_ms={elapsed_ms:.2f}"
                )

        # The next layer may read this layer's CPU-staged hidden output from a
        # different CUDA device. Make the D2H copies explicit before consumers
        # enqueue H2D work, avoiding cross-device timing races with pinned memory.
        if output_cpu.device.type == "cpu":
            torch.cuda.current_stream(torch.device(device)).synchronize()

        del key_states, value_states

        self._log_streaming_progress(f"[IMI STREAM] layer={layer_idx} phase=done")
    def _prefill_embed_to_hidden(self, inputs_ids, hidden_states, device: str, chunk_size: int) -> None:
        for chunk_start in range(0, inputs_ids.shape[1], chunk_size):
            chunk_end = min(inputs_ids.shape[1], chunk_start + chunk_size)
            ids_chunk = inputs_ids[:, chunk_start:chunk_end].to(device, non_blocking=True)
            embed_chunk = self.word_embedding(ids_chunk)
            hidden_states[:, chunk_start:chunk_end, :].copy_(embed_chunk, non_blocking=True)

    def _prefill_forward_cpu_staged(self, inputs_ids, batch_ranges, chunk_size: int, hidden_device, pin_hidden: bool):
        bsz, seq_len = inputs_ids.shape
        device = self.layers[0].device
        hidden_kwargs = {
            "dtype": self.dtype,
            "device": hidden_device,
        }
        if str(hidden_device) == "cpu":
            hidden_kwargs["pin_memory"] = pin_hidden
        hidden_cpu = torch.empty((bsz, seq_len, self.hidden_size), **hidden_kwargs)
        next_cpu = torch.empty_like(hidden_cpu)

        embed_t0 = time.perf_counter() if getattr(self, "breakdown_enabled", False) else None
        self._prefill_embed_to_hidden(inputs_ids, hidden_cpu, device, chunk_size)
        if embed_t0 is not None:
            self.prefill_embedding_wall_ms += (time.perf_counter() - embed_t0) * 1000.0
        self._log_streaming_progress("[IMI STREAM] embedding_done")

        for ldx in range(self.num_layers):
            for start_bdx, end_bdx, _valid_start in batch_ranges:
                self.layer_prefill_chunked(
                    ldx,
                    start_bdx,
                    hidden_cpu[start_bdx:end_bdx],
                    next_cpu[start_bdx:end_bdx],
                    seq_len,
                    chunk_size,
                )
            hidden_cpu, next_cpu = next_cpu, hidden_cpu
            if self.num_gpus > 1:
                next_device = self.layer_mapping[str(ldx + 1)] if str(ldx + 1) in self.layer_mapping else self.layer_mapping[str(0)]
                self._activate_device_runtime_tensors(next_device)

        return hidden_cpu

    def _prefill_forward_stage_gpu(self, inputs_ids, batch_ranges, stages, chunk_size: int):
        bsz, seq_len = inputs_ids.shape
        root_device = self.layers[0].device
        first_device = stages[0][2]
        hidden_states = torch.empty(
            (bsz, seq_len, self.hidden_size),
            dtype=self.dtype,
            device=first_device,
        )

        embed_t0 = time.perf_counter() if getattr(self, "breakdown_enabled", False) else None
        self._prefill_embed_to_hidden(inputs_ids, hidden_states, root_device, chunk_size)
        if embed_t0 is not None:
            self.prefill_embedding_wall_ms += (time.perf_counter() - embed_t0) * 1000.0
        self._log_streaming_progress("[IMI STREAM] embedding_done")

        for stage_idx, (stage_start, stage_end, stage_device) in enumerate(stages):
            if str(hidden_states.device) != str(torch.device(stage_device)):
                torch.cuda.current_stream(hidden_states.device).synchronize()
                hidden_states = hidden_states.to(stage_device, non_blocking=True)

            next_states = torch.empty_like(hidden_states)
            self._log_streaming_progress(
                f"[IMI STREAM] stage={stage_idx} layers={stage_start}-{stage_end - 1} device={stage_device}"
            )
            for ldx in range(stage_start, stage_end):
                for start_bdx, end_bdx, _valid_start in batch_ranges:
                    self.layer_prefill_chunked(
                        ldx,
                        start_bdx,
                        hidden_states[start_bdx:end_bdx],
                        next_states[start_bdx:end_bdx],
                        seq_len,
                        chunk_size,
                    )
                hidden_states, next_states = next_states, hidden_states
            del next_states

        return hidden_states

    def prefill_forward_chunked(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = self.layers[0].device
        chunk_size = max(int(self.prefill_attn_chunk_size), 1)
        stages = self._prefill_layer_stages()
        use_stage_gpu = self._should_use_stage_gpu_prefill(stages, bsz, seq_len)
        keep_hidden_gpu = self._should_keep_streaming_hidden_on_gpu(device, bsz, seq_len)
        pin_hidden = False if keep_hidden_gpu else self._should_pin_streaming_hidden(bsz, seq_len)
        hidden_device = device if keep_hidden_gpu else "cpu"
        path = "stage_gpu" if use_stage_gpu else "cpu_staged"
        log_hidden_device = "stage_gpu" if use_stage_gpu else hidden_device
        log_pin_hidden = 0 if use_stage_gpu else int(pin_hidden)
        stage_desc = self._format_prefill_stages(stages)

        self._log_streaming_progress(
            f"[IMI STREAM] prefill_start path={path} seq_len={seq_len} chunk_size={chunk_size} "
            f"hidden_device={log_hidden_device} pin_hidden={log_pin_hidden} stages={stage_desc}"
        )
        if seq_len >= 32768 and not self._streaming_progress_enabled():
            print(
                "[IMI STREAM] "
                f"prefill_start path={path} seq_len={seq_len} chunk_size={chunk_size} "
                f"hidden_device={log_hidden_device} pin_hidden={log_pin_hidden} "
                f"stages={stage_desc} set IMI_STREAMING_PROGRESS=1 for per-layer/chunk progress",
                flush=True,
            )

        batch_ranges = self._prefill_chunked_batch_ranges(bsz)
        if len(batch_ranges) > 1:
            self._log_streaming_progress(
                "[IMI STREAM] batch_ranges="
                + ",".join(f"[{start},{end})@{valid_start}" for start, end, valid_start in batch_ranges)
            )

        if use_stage_gpu:
            hidden_cpu = self._prefill_forward_stage_gpu(inputs_ids, batch_ranges, stages, chunk_size)
        else:
            hidden_cpu = self._prefill_forward_cpu_staged(
                inputs_ids,
                batch_ranges,
                chunk_size,
                hidden_device,
                pin_hidden,
            )

        self._log_streaming_progress("[IMI STREAM] all_layers_done")

        output_t0 = time.perf_counter() if getattr(self, "breakdown_enabled", False) else None
        if hidden_cpu.device.type == "cuda" and str(hidden_cpu.device) != str(torch.device(device)):
            torch.cuda.current_stream(hidden_cpu.device).synchronize()
        last_hidden = hidden_cpu[:, -1:, :].to(device, non_blocking=True)
        self._activate_device_runtime_tensors(device)
        last_hidden = self.layernorm(last_hidden, self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden)
        if output_t0 is not None:
            self.prefill_output_head_wall_ms += (time.perf_counter() - output_t0) * 1000.0
        return logits
    def prefill_forward(self, inputs_ids):
        return self.prefill_forward_chunked(inputs_ids)
