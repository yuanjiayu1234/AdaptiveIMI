# Hybrid 8-Way IMI Current Design

`hybrid_8way_v2` is an 8-subspace learned-partition IMI path. The current implementation is fail-fast: missing search metadata is treated as an index build error, not as a reason to silently fall back to flat search.

## Decode Modes

Only two decode search modes are supported:

- `flat`: score full merged cluster centroids.
- `subspace_rerank`: select candidates with 8-way subspace centroids, then rerank with full centroids.

Removed modes:

- `subspace`
- `hybrid`

## GPU Builder Data Flow

The GPU builder now keeps the compact phase proportional to actual tokens, not to the dense Cartesian product of all `k_values`.

1. Run 8 independent learned-subspace k-means phases on GPU.
2. Pack each token's 8 labels into one `uint64` code.
3. Per row, segmented radix-sort actual token codes.
4. Mark code boundaries and segmented-scan boundary flags.
5. Emit compact metadata:
   - `token_ids_by_cluster`
   - `cluster_counts`
   - `cluster_sizes`
   - `cluster_offsets`
   - `centroids`
   - `compact_codes`
   - `subspace_centroids`
   - `subspace_centroid_masks`
   - `subspace_dim_ids`
   - `subspace_dim_masks`

The old `[rows, dense_code_space]` workspaces are removed. The builder no longer returns `token_labels` or raw per-subspace labels to Python.

## CPU Reorganization

CPU reorganization does not interpret clusters. It only copies raw CPU KV into cluster order using `token_ids_by_cluster`.

The shared C++ primitive is:

- `copy_kv_by_token_order_rows_cpu`

It is used by both:

- GPU build path after metadata D2H.
- CPU build path after labels are compacted.

Parallelism is by row-token tiles, not only by row/head, so it can use CPU cores even when `batch * kv_heads` is small.

## Metadata Ownership

CPU needs:

- `token_counts`
- `cluster_counts`
- `cluster_sizes`
- `cluster_offsets`
- `centroids`
- `token_ids_by_cluster` only during reorganization

GPU `subspace_rerank` needs:

- `subspace_centroids`
- `subspace_centroid_masks`
- `subspace_dim_ids`
- `subspace_dim_masks`
- `compact_codes`

CPU copies of search metadata are disabled by default and should only be kept for explicit debugging.

## CPU 8-Way K-Means

The CPU learned-partition path uses a dedicated 8-way fused executor when `M == 8`.

It does not:

- convert the full layer KV into a persistent fp32 tensor
- build `subvecs[head * 8]`
- run eight independent subspace task streams per head

Instead, it works on CPU data only:

1. Read fp16/bf16 CPU KV.
2. Convert one `(head, token_tile)` into a thread-local fp32 tile.
3. Reuse that tile for all 8 learned subspaces.
4. Accumulate thread-local `[head, 8, k, sub_dim]` sums and counts.
5. Reduce thread-local accumulators and update centroids.

The learned partition is still respected exactly: each subspace uses the RoPE pair ids from the deployment JSON, so subspace dimensions can differ across heads and stages.

The hot path uses flat buffers:

- `dim_ids_flat`: `[heads * 8, max_sub_dim]`
- `labels_flat`: `[heads * 8, tokens]`
- `centroids_flat`: `[heads * 8, max_k, max_sub_dim]`
- `thread_sums`: `[threads, heads * 8, max_k, max_sub_dim]`
- `thread_counts`: `[threads, heads * 8, max_k]`

Distance is computed through `distance_dim_ids(token_fp32, dim_ids, dim_count, centroid)`. AVX2 uses `_mm256_i32gather_ps`; scalar code remains as the fallback.

Tile fp16/bf16 conversion also has CPU SIMD paths:

- fp16 uses F16C when available.
- bf16 uses AVX2 integer unpack/shift when available.
- scalar conversion remains as the fallback.
