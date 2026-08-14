// CPU background packer for AdaptiveIMI generated KV pages.
//
// This runs only on the single generated-history worker after the asynchronous
// D2H event has completed.  Keeping the short (typically 16-token) routing
// loop in native code avoids Python list/dict work and makes publication a
// single, immutable page commit on the decode stream.

#include <torch/extension.h>

#include <algorithm>
#include <cstring>

namespace imi_decode {

namespace {

template <typename Scalar>
void pack_generated_pages_impl(
    const torch::Tensor& assignments,
    const torch::Tensor& staging_keys,
    const torch::Tensor& staging_values,
    const torch::Tensor& page_keys,
    const torch::Tensor& page_values,
    const torch::Tensor& cluster_heads,
    const torch::Tensor& cluster_tails,
    const torch::Tensor& page_prev,
    const torch::Tensor& page_sizes,
    const torch::Tensor& next_pages,
    const torch::Tensor& out_cluster_ids,
    const torch::Tensor& out_cluster_heads,
    const torch::Tensor& out_cluster_tails,
    const torch::Tensor& out_page_ids,
    const torch::Tensor& out_page_prev,
    const torch::Tensor& out_page_sizes,
    const torch::Tensor& out_cluster_sums,
    const torch::Tensor& out_cluster_counts
) {
    const int rows = static_cast<int>(assignments.size(0));
    const int tokens = static_cast<int>(assignments.size(1));
    const int clusters = static_cast<int>(cluster_heads.size(1));
    const int total_pages = static_cast<int>(page_keys.size(1));
    const int page_tokens = static_cast<int>(page_keys.size(2));
    const int dim = static_cast<int>(page_keys.size(3));

    const auto* assignments_ptr = assignments.data_ptr<int32_t>();
    const auto* src_keys = staging_keys.data_ptr<Scalar>();
    const auto* src_values = staging_values.data_ptr<Scalar>();
    auto* dst_keys = page_keys.data_ptr<Scalar>();
    auto* dst_values = page_values.data_ptr<Scalar>();
    auto* heads = cluster_heads.data_ptr<int32_t>();
    auto* tails = cluster_tails.data_ptr<int32_t>();
    auto* prev = page_prev.data_ptr<int32_t>();
    auto* sizes = page_sizes.data_ptr<int32_t>();
    auto* next = next_pages.data_ptr<int32_t>();
    auto* out_ids = out_cluster_ids.data_ptr<int32_t>();
    auto* out_heads = out_cluster_heads.data_ptr<int32_t>();
    auto* out_tails = out_cluster_tails.data_ptr<int32_t>();
    auto* out_pages = out_page_ids.data_ptr<int32_t>();
    auto* out_prev = out_page_prev.data_ptr<int32_t>();
    auto* out_sizes = out_page_sizes.data_ptr<int32_t>();
    auto* out_sums = out_cluster_sums.data_ptr<float>();
    auto* out_counts = out_cluster_counts.data_ptr<int32_t>();

    out_cluster_ids.fill_(-1);
    out_cluster_heads.fill_(-1);
    out_cluster_tails.fill_(-1);
    out_page_ids.fill_(-1);
    out_page_prev.fill_(-1);
    out_page_sizes.zero_();
    out_cluster_sums.zero_();
    out_cluster_counts.zero_();

    for (int row = 0; row < rows; ++row) {
        int cluster_out_count = 0;
        int page_out_count = 0;
        const int64_t cluster_base = static_cast<int64_t>(row) * clusters;
        const int64_t page_base = static_cast<int64_t>(row) * total_pages;
        const int64_t token_base = static_cast<int64_t>(row) * tokens;
        for (int token = 0; token < tokens; ++token) {
            const int cluster = assignments_ptr[token_base + token];
            TORCH_CHECK(cluster >= 0 && cluster < clusters, "generated key was assigned to an invalid centroid");

            int cluster_slot = -1;
            for (int i = 0; i < cluster_out_count; ++i) {
                if (out_ids[token_base + i] == cluster) {
                    cluster_slot = i;
                    break;
                }
            }
            if (cluster_slot < 0) {
                TORCH_CHECK(cluster_out_count < tokens, "generated cluster commit overflow");
                cluster_slot = cluster_out_count++;
                out_ids[token_base + cluster_slot] = cluster;
            }

            int page = tails[cluster_base + cluster];
            if (page < 0 || sizes[page_base + page] >= page_tokens) {
                page = next[row]++;
                TORCH_CHECK(page >= 0 && page < total_pages,
                            "generated delta page capacity exceeded; increase async_update.page_capacity_tokens");
                prev[page_base + page] = tails[cluster_base + cluster];
                sizes[page_base + page] = 0;
                if (heads[cluster_base + cluster] < 0) heads[cluster_base + cluster] = page;
                tails[cluster_base + cluster] = page;
            }

            const int offset = sizes[page_base + page];
            const int64_t source = (token_base + token) * dim;
            const int64_t destination = ((page_base + page) * page_tokens + offset) * dim;
            std::memcpy(dst_keys + destination, src_keys + source, static_cast<size_t>(dim) * sizeof(Scalar));
            std::memcpy(dst_values + destination, src_values + source, static_cast<size_t>(dim) * sizeof(Scalar));
            sizes[page_base + page] = offset + 1;

            int page_slot = -1;
            for (int i = 0; i < page_out_count; ++i) {
                if (out_pages[token_base + i] == page) {
                    page_slot = i;
                    break;
                }
            }
            if (page_slot < 0) {
                TORCH_CHECK(page_out_count < tokens, "generated page commit overflow");
                page_slot = page_out_count++;
                out_pages[token_base + page_slot] = page;
            }
            out_sizes[token_base + page_slot] = sizes[page_base + page];
            out_prev[token_base + page_slot] = prev[page_base + page];

            const int64_t sum_base = (token_base + cluster_slot) * dim;
            for (int d = 0; d < dim; ++d) out_sums[sum_base + d] += static_cast<float>(src_keys[source + d]);
            ++out_counts[token_base + cluster_slot];
        }
        for (int i = 0; i < cluster_out_count; ++i) {
            const int cluster = out_ids[token_base + i];
            out_heads[token_base + i] = heads[cluster_base + cluster];
            out_tails[token_base + i] = tails[cluster_base + cluster];
        }
    }
}

void check_cpu_pinned(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(!tensor.is_cuda() && tensor.is_pinned(), name, " must be pinned CPU memory");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void pack_generated_pages(
    const torch::Tensor& assignments,
    const torch::Tensor& staging_keys,
    const torch::Tensor& staging_values,
    const torch::Tensor& page_keys,
    const torch::Tensor& page_values,
    const torch::Tensor& cluster_heads,
    const torch::Tensor& cluster_tails,
    const torch::Tensor& page_prev,
    const torch::Tensor& page_sizes,
    const torch::Tensor& next_pages,
    const torch::Tensor& out_cluster_ids,
    const torch::Tensor& out_cluster_heads,
    const torch::Tensor& out_cluster_tails,
    const torch::Tensor& out_page_ids,
    const torch::Tensor& out_page_prev,
    const torch::Tensor& out_page_sizes,
    const torch::Tensor& out_cluster_sums,
    const torch::Tensor& out_cluster_counts
) {
    check_cpu_pinned(assignments, "assignments");
    check_cpu_pinned(staging_keys, "staging_keys");
    check_cpu_pinned(staging_values, "staging_values");
    check_cpu_pinned(page_keys, "page_keys");
    check_cpu_pinned(page_values, "page_values");
    check_cpu_pinned(cluster_heads, "cluster_heads");
    check_cpu_pinned(cluster_tails, "cluster_tails");
    check_cpu_pinned(page_prev, "page_prev");
    check_cpu_pinned(page_sizes, "page_sizes");
    check_cpu_pinned(next_pages, "next_pages");
    check_cpu_pinned(out_cluster_ids, "out_cluster_ids");
    check_cpu_pinned(out_cluster_heads, "out_cluster_heads");
    check_cpu_pinned(out_cluster_tails, "out_cluster_tails");
    check_cpu_pinned(out_page_ids, "out_page_ids");
    check_cpu_pinned(out_page_prev, "out_page_prev");
    check_cpu_pinned(out_page_sizes, "out_page_sizes");
    check_cpu_pinned(out_cluster_sums, "out_cluster_sums");
    check_cpu_pinned(out_cluster_counts, "out_cluster_counts");
    TORCH_CHECK(assignments.scalar_type() == torch::kInt32, "assignments must be int32");
    TORCH_CHECK(staging_keys.sizes() == staging_values.sizes() && page_keys.sizes() == page_values.sizes(), "generated KV shape mismatch");
    TORCH_CHECK(staging_keys.dim() == 3 && page_keys.dim() == 4 && assignments.dim() == 2, "invalid generated page rank");
    TORCH_CHECK(staging_keys.size(0) == assignments.size(0) && staging_keys.size(1) == assignments.size(1) && staging_keys.size(2) == page_keys.size(3), "generated assignment/KV shape mismatch");
    TORCH_CHECK(cluster_heads.dim() == 2 && cluster_tails.sizes() == cluster_heads.sizes(), "invalid cluster page state");
    TORCH_CHECK(cluster_heads.size(0) == assignments.size(0), "cluster page row mismatch");
    TORCH_CHECK(page_prev.dim() == 2 && page_sizes.sizes() == page_prev.sizes() && next_pages.dim() == 1, "invalid page state");
    TORCH_CHECK(page_prev.size(0) == assignments.size(0) && page_prev.size(1) == page_keys.size(1) && next_pages.size(0) == assignments.size(0), "page state shape mismatch");
    TORCH_CHECK(out_cluster_ids.sizes() == assignments.sizes() && out_cluster_heads.sizes() == assignments.sizes() && out_cluster_tails.sizes() == assignments.sizes() && out_page_ids.sizes() == assignments.sizes() && out_page_prev.sizes() == assignments.sizes() && out_page_sizes.sizes() == assignments.sizes() && out_cluster_counts.sizes() == assignments.sizes(), "generated commit metadata shape mismatch");
    TORCH_CHECK(out_cluster_sums.size(0) == assignments.size(0) && out_cluster_sums.size(1) == assignments.size(1) && out_cluster_sums.size(2) == page_keys.size(3) && out_cluster_sums.scalar_type() == torch::kFloat32, "generated commit sum shape mismatch");
    TORCH_CHECK(cluster_heads.scalar_type() == torch::kInt32 && cluster_tails.scalar_type() == torch::kInt32 && page_prev.scalar_type() == torch::kInt32 && page_sizes.scalar_type() == torch::kInt32 && next_pages.scalar_type() == torch::kInt32 && out_cluster_ids.scalar_type() == torch::kInt32 && out_cluster_heads.scalar_type() == torch::kInt32 && out_cluster_tails.scalar_type() == torch::kInt32 && out_page_ids.scalar_type() == torch::kInt32 && out_page_prev.scalar_type() == torch::kInt32 && out_page_sizes.scalar_type() == torch::kInt32 && out_cluster_counts.scalar_type() == torch::kInt32, "generated page metadata must be int32");

    if (staging_keys.scalar_type() == torch::kBFloat16) {
        pack_generated_pages_impl<at::BFloat16>(assignments, staging_keys, staging_values, page_keys, page_values, cluster_heads, cluster_tails, page_prev, page_sizes, next_pages, out_cluster_ids, out_cluster_heads, out_cluster_tails, out_page_ids, out_page_prev, out_page_sizes, out_cluster_sums, out_cluster_counts);
    } else if (staging_keys.scalar_type() == torch::kFloat16) {
        pack_generated_pages_impl<at::Half>(assignments, staging_keys, staging_values, page_keys, page_values, cluster_heads, cluster_tails, page_prev, page_sizes, next_pages, out_cluster_ids, out_cluster_heads, out_cluster_tails, out_page_ids, out_page_prev, out_page_sizes, out_cluster_sums, out_cluster_counts);
    } else {
        TORCH_CHECK(false, "generated KV must be BF16 or FP16");
    }
}

}  // namespace imi_decode
