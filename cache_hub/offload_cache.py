import math
import torch
from flash_attn import flash_attn_with_kvcache
from .base import KV_Cache


class flash_attn_cache_offload(KV_Cache):
    """
    KV Cache with CPU offloading for Full flash-attn.
    KV stays on CPU, loaded to GPU layer-by-layer during decode.

    Notes:
    - Supports multi-GPU via layer_mapping (layers can live on different CUDA devices).
    - KV is stored on CPU pinned memory; per-device GPU staging buffers are allocated on demand.
    """

    def __init__(
        self,
        valid_start,
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        prefill_bsz: int,
        num_gpus: int,
        model_size: int,
    ) -> None:
        super().__init__(
            layer_num,
            batch_size,
            max_length,
            num_key_value_heads,
            num_heads,
            head_dim,
            dtype,
            layer_mapping,
            prefill_bsz,
            num_gpus,
            model_size,
        )

        self.device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(":")[-1]))
        self.valid_start_list = valid_start

        # Keep valid_length on CPU (avoid device-mismatch issues across multi-GPU)
        self.valid_length = None

        # KV cache on CPU pinned memory
        self.key_cache = [
            torch.empty(
                (batch_size, max_length, num_key_value_heads, head_dim),
                device="cpu",
                pin_memory=True,
                dtype=dtype,
            )
            for _ in range(layer_num)
        ]
        self.value_cache = [
            torch.empty(
                (batch_size, max_length, num_key_value_heads, head_dim),
                device="cpu",
                pin_memory=True,
                dtype=dtype,
            )
            for _ in range(layer_num)
        ]

        # Per-device GPU staging buffers (reused)
        self.gpu_key_buffer = {}
        self.gpu_value_buffer = {}
        self.gpu_buffer_capacity = {}

        # Per-device async copy resources
        self.copystream = {}
        self.copyevent = {}
        self.kv_ready_event = {}
        for dev in self.device_list:
            with torch.cuda.device(dev):
                self.copystream[dev] = torch.cuda.Stream()
                self.copyevent[dev] = torch.cuda.Event()
                self.kv_ready_event[dev] = torch.cuda.Event()

    def move_gpu(self):
        # No-op: KV stays on CPU
        pass

    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, start_bdx):
        bsz, seq_len, _, _ = key_states.shape

        if self.valid_length is None:
            # bsz is typically 1 in RULER; store CPU int32 tensor for simplicity.
            self.valid_length = torch.from_numpy(seq_len - self.valid_start_list).to(torch.int32)

        _valid_start = int(self.valid_start_list[start_bdx])
        _valid_length = int(seq_len - _valid_start)

        dev = self.layer_mapping[str(layer_idx)]
        with torch.cuda.device(dev):
            self.kv_ready_event[dev].record()
            with torch.cuda.stream(self.copystream[dev]):
                self.kv_ready_event[dev].wait()
                # Copy GPU -> CPU pinned memory asynchronously without creating an
                # intermediate CPU tensor, so the CPU KV cache is populated directly.
                self.key_cache[layer_idx][start_bdx : start_bdx + bsz, :_valid_length].copy_(
                    key_states[:, _valid_start:],
                    non_blocking=True,
                )
                self.value_cache[layer_idx][start_bdx : start_bdx + bsz, :_valid_length].copy_(
                    value_states[:, _valid_start:],
                    non_blocking=True,
                )
                self.copyevent[dev].record()

        if (layer_idx == self.layer_num - 1) and (start_bdx + bsz == self.batch_size):
            self.context += seq_len

        # Return KV on the same device for attention computation in prefill
        return key_states[:, _valid_start:], value_states[:, _valid_start:]

    def sync(self, layer_idx, start_bdx):
        dev = self.layer_mapping[str(layer_idx)]
        self.copyevent[dev].synchronize()

    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        # Update CPU cache with new token. Keep this copy blocking because decode
        # immediately reloads the CPU KV into GPU staging buffers for attention.
        valid_pos = int(self.valid_length[0].item())  # assume same length for all batches
        self.key_cache[layer_idx][:, valid_pos : valid_pos + 1].copy_(key_states, non_blocking=False)
        self.value_cache[layer_idx][:, valid_pos : valid_pos + 1].copy_(value_states, non_blocking=False)

        if layer_idx == self.layer_num - 1:
            self.context += 1
            self.valid_length += 1

        # Return CPU tensors (will be loaded to GPU in attention)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def load_kv_to_gpu(self, layer_idx, desired_len, device=None):
        """Load KV cache for a layer to the specified GPU device."""
        if device is None:
            device = self.layer_mapping[str(layer_idx)]

        buf_k = self.gpu_key_buffer.get(device)
        buf_v = self.gpu_value_buffer.get(device)

        if buf_k is None or buf_k.shape[1] < desired_len:
            capacity = int(math.ceil(desired_len / 1024.0) * 1024)
            self.gpu_key_buffer[device] = torch.empty(
                (self.batch_size, capacity, self.kv_head, self.head_dim),
                device=device,
                dtype=self.dtype,
            )
            self.gpu_value_buffer[device] = torch.empty(
                (self.batch_size, capacity, self.kv_head, self.head_dim),
                device=device,
                dtype=self.dtype,
            )
            self.gpu_buffer_capacity[device] = capacity
            buf_k = self.gpu_key_buffer[device]
            buf_v = self.gpu_value_buffer[device]

        with torch.cuda.device(device):
            buf_k[:, :desired_len].copy_(self.key_cache[layer_idx][:, :desired_len], non_blocking=True)
            buf_v[:, :desired_len].copy_(self.value_cache[layer_idx][:, :desired_len], non_blocking=True)
            torch.cuda.synchronize(device)

        return buf_k[:, :desired_len], buf_v[:, :desired_len]
