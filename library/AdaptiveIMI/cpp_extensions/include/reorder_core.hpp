#pragma once

#include <cstddef>
#include <cstdint>

namespace imi {

void copy_kv_by_token_order_rows_cpu(
    const void* keys_data,
    const void* values_data,
    const int32_t* token_ids_by_row,
    const int32_t* row_token_counts,
    int64_t rows,
    int64_t src_tokens,
    int64_t dst_tokens,
    int64_t dim,
    size_t element_size,
    size_t src_row_stride_bytes,
    size_t dst_row_stride_bytes,
    void* output_keys,
    void* output_values,
    int64_t token_ids_row_stride,
    int64_t tokens_per_task = 32
);

} // namespace imi
