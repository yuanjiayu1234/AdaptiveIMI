def imi_decode_attn(query_states, key_states, value_states, layer_idx, imi_cache):
    """
    query_states: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
    """
    static_len = (
        imi_cache.static_pattern_total
        if layer_idx == imi_cache.layer_num - 1
        else imi_cache.static_pattern_total + 1
    )
    return imi_cache.attn_func(query_states.contiguous(), layer_idx, static_len)
