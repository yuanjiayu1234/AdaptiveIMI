from flash_attn import flash_attn_with_kvcache


def compact_decode_attention(q, k_cache, v_cache, cache_seqlens):
    """Decode attention over AdaptiveIMI's compact contiguous execution buffer."""
    return flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
    )
