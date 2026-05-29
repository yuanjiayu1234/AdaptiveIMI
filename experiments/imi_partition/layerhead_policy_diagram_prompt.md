# IMI Centroid-Structural Partition Training Diagram Prompt

Draw a clear technical pipeline diagram for the current AdaptiveIMI `hybrid_8way_v2`
partition training and validation flow.

The diagram should show these stages from left to right:

1. Calibration data, not test data
   - Use long-context calibration documents from independent LongBench/LongBench-v2 style datasets.
   - Do not use RULER or final evaluation data for training.
   - Prefer long calibration contexts such as 128K tokens when the model context allows it.

2. Training key collection
   - Run prefill forward passes with `logits_to_keep=1` or equivalent memory-saving output.
   - Collect searchable key states `K` after RoPE for selected layers and KV heads.
   - Training does not need query states `Q`, lm-head logits for all tokens, or generated outputs.
   - Slice out static prefix/suffix regions using the same static window convention as the runtime cache.
   - Store collected keys as `[layer, kv_head, token, head_dim]`.

3. Optimization scope
   - The current default is `(layer, KV head)` scope.
   - Each KV head has its own key geometry, so its teacher K-means centroids and IMI compact-cell centroids are computed independently.
   - Layer-group, layer-level, and global scopes are retained only as ablations or faster debugging modes.
   - Each optimized layer-head receives sampled full-dimensional key vectors from that exact `(layer, KV head)`.

4. Candidate partition strategies
   Show these branches:
   - `learned`: balanced RoPE-pair groups optimized by structural centroid error.
   - `pair_sequential`: contiguous RoPE-pair groups, preserving each RoPE pair.
   - `raw sequential`: contiguous raw hidden-dimension groups, represented as `dim_groups` when needed.
   The learned branch should show:
   - start from balanced sequential and random balanced pair groupings.
   - propose balanced swaps between groups.
   - keep the proposal when centroid-structural error decreases.

5. Full-dimensional K-means teacher
   - Build a full-dimensional K-means teacher on sampled calibration keys.
   - Use approximately `tokens / target_cluster_size` clusters, capped by `max_full_clusters`.
   - Let `F_i` be the full-dimensional teacher centroid assigned to key `K_i`.
   - This teacher is the baseline geometry we want IMI cells to preserve.

6. IMI compact-cell construction
   - For a candidate 8-way partition, split each key into 8 subspaces.
   - Choose runtime-compatible `k_values`, with at least 2 clusters per subspace.
   - Cluster each subspace with raw L2 K-means.
   - Combine subspace labels into compact Cartesian cells.
   - For each non-empty compact cell, compute its full-dimensional true centroid:

```text
T_cell = mean(K_i for i in compact cell)
```

7. Structural training objective
   - Compare the full-dimensional teacher centroid assigned to each key with the true full-dimensional centroid of its IMI compact cell.
   - Optimize the partition by minimizing:

```text
Loss = mean_i || F_i - T_cell(i) ||^2
NormalizedLoss = Loss / Var(K)
```

   - This objective measures structural distortion introduced by the subspace product partition.
   - Recall is not the training target; recall is a downstream validation result.

8. Final partition JSON
   - Emit an `adaptive_imi_learned_partition_v1` JSON file.
   - Include `layer_groups`, `M=8`, per-head RoPE-pair `groups`, `k_values`, and structural metrics.
   - Online `hybrid_8way_v2` reads this JSON during prefill and constructs the corresponding compact-cell IMI index.

9. Downstream first-decode validation
   - This is evaluation, not training.
   - Run a real prefill with cache, predict the first generated token, then run cached decode for that token.
   - Collect `Q` from this first generated token's decode forward pass.
   - Do not use the prompt's last-token prefill `Q` as a proxy.
   - Use the prompt's searchable RoPE-applied `K` as the retrieval database.
   - Compute exact top tokens by full-dimensional score:

```text
s_i = Q_first_decode · K_i
ExactTopK = TopK(s)
```

   - Validate `flat` using the same budget semantics as online search:
     - score full-dimensional true compact-cell centroids.
     - select up to `nprobe` cells.
     - stop opening cells when `max_retrieval_tokens` is reached.
     - rerank selected candidate tokens by exact `Q_first_decode · K_i`.
   - `full-dim K-means` baseline uses the same two-stage `flat` budget semantics.
   - `subspace` validation should follow the actual online `subspace_rerank` implementation when enabled.
   - PQ is currently a diagnostic ADC baseline, not a real `hybrid_8way_v2` runtime mode.

Important visual details:

- Clearly separate "training uses K only" from "validation uses first-decode-token Q".
- Mark the old "last-token prefill Q" proxy as obsolete.
- Show the full-dimensional K-means teacher and IMI compact-cell true centroid side by side.
- Show `full_kmeans`, `learned`, `pair_sequential`, `raw sequential`, and `pq` as baselines or comparison branches.
- Distinguish temporary calibration key files from the final persistent partition JSON.

Include these equations in the diagram:

```text
Teacher centroid: F_i = C_full_kmeans(label_full(K_i))
IMI true centroid: T_cell(i) = mean(K_j for K_j in same compact IMI cell as K_i)
Training loss: mean_i ||F_i - T_cell(i)||^2
Flat cell score: Q_first_decode · T_cell
Token rerank score: Q_first_decode · K_i
Recall@k = |RerankedTopK ∩ ExactTopK| / k
```

Emphasize the key idea:

```text
Learn an 8-way RoPE-pair partition that preserves full-dimensional cluster geometry,
freeze it as JSON, and validate retrieval with the first generated token's decode Q.
```
