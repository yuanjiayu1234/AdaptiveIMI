#include "reorder_core.hpp"

#include <ATen/Parallel.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

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
    int64_t tokens_per_task
) {
    if (keys_data == nullptr || values_data == nullptr ||
        token_ids_by_row == nullptr || row_token_counts == nullptr ||
        output_keys == nullptr || output_values == nullptr) {
        throw std::runtime_error("copy_kv_by_token_order_rows_cpu received null pointer");
    }
    if (rows < 0 || src_tokens < 0 || dst_tokens < 0 || dim <= 0) {
        throw std::runtime_error("copy_kv_by_token_order_rows_cpu received invalid shape");
    }
    if (token_ids_row_stride < src_tokens) {
        throw std::runtime_error("token id row stride is smaller than source token count");
    }
    if (tokens_per_task <= 0) {
        tokens_per_task = 32;
    }

    const int64_t token_bytes = dim * static_cast<int64_t>(element_size);
    if (src_row_stride_bytes == 0) {
        src_row_stride_bytes = static_cast<size_t>(src_tokens * token_bytes);
    }
    if (dst_row_stride_bytes == 0) {
        dst_row_stride_bytes = static_cast<size_t>(dst_tokens * token_bytes);
    }

    for (int64_t row = 0; row < rows; ++row) {
        const int32_t count = row_token_counts[row];
        if (count < 0 || count > src_tokens || count > dst_tokens) {
            throw std::runtime_error("row token count is out of range");
        }
    }

    const int64_t tasks_per_row = src_tokens == 0
        ? 0
        : (src_tokens + tokens_per_task - 1) / tokens_per_task;
    const int64_t total_tasks = rows * tasks_per_row;
    if (total_tasks == 0) {
        return;
    }

    const char* keys_base = static_cast<const char*>(keys_data);
    const char* values_base = static_cast<const char*>(values_data);
    char* out_keys_base = static_cast<char*>(output_keys);
    char* out_values_base = static_cast<char*>(output_values);

    at::parallel_for(0, total_tasks, 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t row = task / tasks_per_row;
            const int64_t tile = task - row * tasks_per_row;
            const int32_t token_count = row_token_counts[row];
            const int64_t dst_begin = tile * tokens_per_task;
            if (dst_begin >= token_count) {
                continue;
            }
            const int64_t dst_end = std::min<int64_t>(dst_begin + tokens_per_task, token_count);

            const char* row_keys = keys_base + row * src_row_stride_bytes;
            const char* row_values = values_base + row * src_row_stride_bytes;
            char* row_out_keys = out_keys_base + row * dst_row_stride_bytes;
            char* row_out_values = out_values_base + row * dst_row_stride_bytes;
            const int32_t* row_ids = token_ids_by_row + row * token_ids_row_stride;

            for (int64_t dst = dst_begin; dst < dst_end; ++dst) {
                const int32_t src = row_ids[dst];
                if (src < 0 || src >= src_tokens) {
                    throw std::runtime_error("token id is out of source range");
                }
                std::memcpy(
                    row_out_keys + dst * token_bytes,
                    row_keys + static_cast<int64_t>(src) * token_bytes,
                    static_cast<size_t>(token_bytes)
                );
                std::memcpy(
                    row_out_values + dst * token_bytes,
                    row_values + static_cast<int64_t>(src) * token_bytes,
                    static_cast<size_t>(token_bytes)
                );
            }
        }
    });
}

void copy_kv_by_token_order_bthd_cpu(
    const void* keys_data,
    const void* values_data,
    const int32_t* token_ids_by_row,
    const int32_t* row_token_counts,
    int64_t batch,
    int64_t kv_heads,
    int64_t src_tokens,
    int64_t dst_tokens,
    int64_t dim,
    size_t element_size,
    void* output_keys,
    void* output_values,
    int64_t token_ids_row_stride,
    int64_t tokens_per_task
) {
    if (keys_data == nullptr || values_data == nullptr ||
        token_ids_by_row == nullptr || row_token_counts == nullptr ||
        output_keys == nullptr || output_values == nullptr) {
        throw std::runtime_error("copy_kv_by_token_order_bthd_cpu received null pointer");
    }
    if (batch < 0 || kv_heads <= 0 || src_tokens < 0 || dst_tokens < 0 || dim <= 0) {
        throw std::runtime_error("copy_kv_by_token_order_bthd_cpu received invalid shape");
    }
    if (token_ids_row_stride < src_tokens) {
        throw std::runtime_error("token id row stride is smaller than source token count");
    }
    if (tokens_per_task <= 0) {
        tokens_per_task = 32;
    }

    const int64_t rows = batch * kv_heads;
    const int64_t token_bytes = dim * static_cast<int64_t>(element_size);
    const int64_t src_token_stride_bytes = kv_heads * token_bytes;
    const int64_t src_batch_stride_bytes = src_tokens * src_token_stride_bytes;
    const int64_t dst_row_stride_bytes = dst_tokens * token_bytes;

    for (int64_t row = 0; row < rows; ++row) {
        const int32_t count = row_token_counts[row];
        if (count < 0 || count > src_tokens || count > dst_tokens) {
            throw std::runtime_error("row token count is out of range");
        }
    }

    const int64_t tasks_per_row = src_tokens == 0
        ? 0
        : (src_tokens + tokens_per_task - 1) / tokens_per_task;
    const int64_t total_tasks = rows * tasks_per_row;
    if (total_tasks == 0) {
        return;
    }

    const char* keys_base = static_cast<const char*>(keys_data);
    const char* values_base = static_cast<const char*>(values_data);
    char* out_keys_base = static_cast<char*>(output_keys);
    char* out_values_base = static_cast<char*>(output_values);

    at::parallel_for(0, total_tasks, 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t row = task / tasks_per_row;
            const int64_t tile = task - row * tasks_per_row;
            const int64_t batch_idx = row / kv_heads;
            const int64_t head_idx = row - batch_idx * kv_heads;
            const int32_t token_count = row_token_counts[row];
            const int64_t dst_begin = tile * tokens_per_task;
            if (dst_begin >= token_count) {
                continue;
            }
            const int64_t dst_end = std::min<int64_t>(dst_begin + tokens_per_task, token_count);

            const char* batch_keys = keys_base + batch_idx * src_batch_stride_bytes;
            const char* batch_values = values_base + batch_idx * src_batch_stride_bytes;
            char* row_out_keys = out_keys_base + row * dst_row_stride_bytes;
            char* row_out_values = out_values_base + row * dst_row_stride_bytes;
            const int32_t* row_ids = token_ids_by_row + row * token_ids_row_stride;

            for (int64_t dst = dst_begin; dst < dst_end; ++dst) {
                const int32_t src = row_ids[dst];
                if (src < 0 || src >= src_tokens) {
                    throw std::runtime_error("token id is out of source range");
                }
                const int64_t src_offset = static_cast<int64_t>(src) * src_token_stride_bytes
                    + head_idx * token_bytes;
                std::memcpy(
                    row_out_keys + dst * token_bytes,
                    batch_keys + src_offset,
                    static_cast<size_t>(token_bytes)
                );
                std::memcpy(
                    row_out_values + dst * token_bytes,
                    batch_values + src_offset,
                    static_cast<size_t>(token_bytes)
                );
            }
        }
    });
}

} // namespace imi
