import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache



def _build_cache_seqlens(query_states, key_states):
    q_len = query_states.shape[1]
    k_len = key_states.shape[1]
    if k_len <= q_len:
        return None
    bsz = query_states.shape[0]
    return torch.full(
        (bsz,),
        k_len,
        device=query_states.device,
        dtype=torch.int32,
    )


def full_prefill_attn(
    query_states,
    key_states,
    value_states,
    causal,
    k_new=None,
    v_new=None,
    cache_seqlens=None,
):
    attn_out = flash_attn_func(
        q=query_states,
        k=key_states,
        v=value_states,
        causal=causal,
    )

    return attn_out


def full_prefill_attn_chunked(query_states, key_states, value_states, causal, chunk_size, chunk_callback=None):
    if chunk_size is None or chunk_size <= 0:
        return full_prefill_attn(query_states, key_states, value_states, causal)

    bsz, seq_len, num_heads, head_dim = query_states.shape
    attn_out = torch.empty((bsz, seq_len, num_heads, head_dim), dtype=query_states.dtype, device=query_states.device)

    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        q_chunk = query_states[:, start:end, :, :]
        k_prefix = key_states[:, :end, :, :]
        v_prefix = value_states[:, :end, :, :]

        if chunk_callback is not None:
            chunk_callback(start, end)

        if not q_chunk.is_contiguous():
            q_chunk = q_chunk.contiguous()
        if not k_prefix.is_contiguous():
            k_prefix = k_prefix.contiguous()
        if not v_prefix.is_contiguous():
            v_prefix = v_prefix.contiguous()

        attn_out[:, start:end, :, :] = flash_attn_func(
            q=q_chunk,
            k=k_prefix,
            v=v_prefix,
            causal=causal,
        )

    return attn_out



def full_decode_attn(query_states, key_states, value_states, layer_idx, full_attn_cache):

    valid_len = full_attn_cache.valid_length if layer_idx == full_attn_cache.layer_num-1 else full_attn_cache.valid_length+1

    attn_out = flash_attn_with_kvcache(
        q=query_states,
        k_cache=key_states,
        v_cache=value_states,
        cache_seqlens=valid_len
    )

    return attn_out


def full_decode_attn_offload(query_states, layer_idx, offload_cache):
    """Decode attention with KV offloading from CPU"""
    valid_len = offload_cache.valid_length if layer_idx == offload_cache.layer_num-1 else offload_cache.valid_length+1
    desired_len = int(valid_len[0].item()) if torch.is_tensor(valid_len) else int(valid_len)

    # flash_attn_with_kvcache requires cache_seqlens on CUDA
    if not torch.is_tensor(valid_len):
        cache_seqlens = torch.tensor([desired_len], device=query_states.device, dtype=torch.int32)
    else:
        cache_seqlens = valid_len.to(device=query_states.device, dtype=torch.int32, non_blocking=True).contiguous()

    # Load KV from CPU to the layer's device using the exact same effective length
    k_gpu, v_gpu = offload_cache.load_kv_to_gpu(layer_idx, desired_len=desired_len, device=query_states.device)

    attn_out = flash_attn_with_kvcache(
        q=query_states,
        k_cache=k_gpu,
        v_cache=v_gpu,
        cache_seqlens=cache_seqlens
    )

    return attn_out
