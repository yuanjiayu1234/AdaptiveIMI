# Dependency-Aware Variable-Rate IMI for RoPE KV Cache

## 1. Motivation

Standard inverted multi-index (IMI) assumes that a key vector can be split into several approximately independent subspaces. For LLM KV cache retrieval, this assumption is fragile for two reasons.

First, Qwen/Llama-style RoPE couples dimensions in pairs. For `head_dim = 128`, the RoPE pairs are:

```text
(0, 64), (1, 65), ..., (63, 127)
```

Splitting dimensions without respecting these pairs breaks the local geometry induced by RoPE. A valid subspace unit should therefore be a RoPE pair, not a single scalar dimension.

Second, RoPE pairs are not equally independent. Some pairs have strong statistical or attention-level dependency. If strongly dependent pairs are assigned to different IMI subspaces, IMI will create many low-probability virtual cells and the product-space approximation becomes poor.

The goal is therefore:

```text
Learn a non-uniform partition of RoPE pairs such that dependency across IMI subspaces is minimized,
then allocate different numbers of centroids to different subspaces according to dimension and importance.
```

This is not a hand-crafted unequal split. The partition is learned from calibration KV data and can vary by model, layer, and KV head.

## 2. Key Representation

For one layer and one KV head, let the RoPE-applied key be:

```text
k_i in R^128
```

where `i` is the token index in the KV cache. We treat the 128 dimensions as 64 RoPE pairs:

```text
p_i,j = [k_i,j, k_i,j+64] in R^2,   j = 0, ..., 63
```

The pair index set is:

```text
P = {0, 1, ..., 63}
```

We want to partition `P` into `M` subspaces:

```text
G = {G_1, G_2, ..., G_M}
G_a ∩ G_b = empty
union_i G_i = P
```

For IMI with 8 subspaces:

```text
M = 8
```

The subspace sizes are learned:

```text
|G_1|, |G_2|, ..., |G_8|
```

They do not need to be equal. A learned result may look like:

```text
[16, 10, 9, 8, 7, 6, 5, 3]
```

but this is only an example. The actual split should come from data.

## 3. What Does "Independent Subspace" Mean?

IMI approximates a high-dimensional distribution with a product of lower-dimensional distributions:

```text
P(k) ≈ product_i P(k_{G_i})
```

This approximation is good only if different subspaces are weakly dependent. Therefore, the partition should minimize dependency leakage across subspaces:

```text
min_G  sum_{a != b} sum_{u in G_a, v in G_b} D(u, v)
```

where `D(u, v)` measures dependency between RoPE pair `u` and RoPE pair `v`.

Equivalently, because total dependency is fixed, this is the same as maximizing within-subspace dependency:

```text
max_G  sum_i sum_{u, v in G_i} D(u, v)
```

This objective has a clear reason: strongly dependent pairs should stay in the same IMI factor, so that the remaining IMI factors are closer to independent.

## 4. Calibration Data

Use a calibration set that matches the intended workload:

```text
long-context prompts
same model
same RoPE scaling mode
same target context length
same dtype path if possible
```

For each calibration prompt:

1. Run prefill.
2. For each layer and KV head, collect RoPE-applied keys:

   ```text
   K_l,h = [seq_len, 128]
   ```

3. Optionally collect decode queries:

   ```text
   Q_l,h_group = [num_decode_steps, group_size, 128]
   ```

4. Exclude sink/local tokens if the index only covers the retrieval zone.

The learned partition can be:

```text
global: shared across all layers and heads
per-layer: one partition per layer
per-layer-group: layers are bucketed into early/mid/late
per-head: one partition per KV head
```

Recommended starting point:

```text
per-layer-group + per-KV-head
```

This captures model structure without making metadata too large.

## 5. Dependency Metric

### 5.1 Basic Pair Correlation

For each RoPE pair `u`, flatten its two coordinates over tokens:

```text
X_u in R^{N x 2}
```

Compute the cross-covariance with pair `v`:

```text
C_uv = Cov(X_u, X_v) in R^{2 x 2}
```

A simple dependency score is:

```text
D_corr(u, v) = || whiten(X_u)^T whiten(X_v) / N ||_F
```

This measures linear dependence between two RoPE pairs.

Pros:

- Simple and stable.
- Does not need query data.

Cons:

- It only measures key distribution dependency, not retrieval importance.

### 5.2 Attention-Weighted Dependency

KV retrieval is driven by QK logits, not just key reconstruction. A better score weights pair dependency by its contribution to attention logits.

For query `q_t` and key `k_i`, the logit decomposes over RoPE pairs:

```text
q_t · k_i = sum_j q_{t,j:j+64} · p_{i,j}
```

Define pair contribution:

```text
s_j(t, i) = q_{t,j:j+64} · p_{i,j}
```

Then define dependency between pairs `u` and `v` as:

```text
D_attn(u, v) = E_{t,i} [ |s_u(t, i) * s_v(t, i)| ]
```

This score is high when two pairs jointly affect the same attention logits. Keeping such pairs in the same subspace reduces error caused by independent product quantization.

Recommended default:

```text
D(u, v) = normalize(D_corr(u, v)) + lambda * normalize(D_attn(u, v))
```

where `lambda` controls how much the partition follows attention behavior.

### 5.3 Output-Aware Dependency

For a stronger but more expensive metric, estimate how separating two pairs affects attention output:

```text
D_out(u, v) = E || Attn(q, K, V) - Attn(q, K with cross-pair approximation removed, V) ||
```

This is more faithful but expensive. It is best used as an evaluation metric or for a small ablation, not as the default learning objective.

## 6. Learning the Non-Uniform Partition

Construct a weighted graph:

```text
nodes: RoPE pairs 0..63
edge weight: D(u, v)
```

The task is graph partitioning into `M = 8` groups:

```text
maximize within-group edge weight
subject to group size constraints
```

### 6.1 Objective

Let `g(u)` be the group assignment of pair `u`.

```text
maximize  sum_{u < v} D(u, v) * 1[g(u) = g(v)]
```

or equivalently:

```text
minimize  sum_{u < v} D(u, v) * 1[g(u) != g(v)]
```

This directly minimizes dependency crossing IMI subspaces.

### 6.2 Constraints

The partition must avoid degenerate groups:

```text
min_pairs <= |G_i| <= max_pairs
sum_i |G_i| = 64
```

Recommended:

```text
min_pairs = 4
max_pairs = 16
M = 8
```

These constraints prevent one huge subspace and many tiny subspaces.

### 6.3 Frequency Locality Regularization

RoPE pairs have ordered frequencies. Adjacent pair indices often correspond to related frequency bands. To avoid arbitrary scattered groups, add a soft regularizer:

```text
R(G_i) = variance({u | u in G_i})
```

The full objective becomes:

```text
maximize  sum_i sum_{u<v, u,v in G_i} D(u,v)
          - gamma * sum_i R(G_i)
```

`gamma` should be small. The dependency matrix should drive the grouping; frequency locality only prevents unstable partitions.

### 6.4 Practical Solver

Exact graph partitioning is not necessary. A practical solver:

1. Initialize groups by RoPE frequency bands or balanced contiguous chunks.
2. Iteratively move one pair from one group to another if it improves the objective.
3. Periodically try pair swaps between groups.
4. Stop when no move or swap improves the score.
5. Run from multiple random seeds and keep the best partition.

Move gain for moving pair `u` from group `A` to group `B`:

```text
gain(u, A -> B)
  = sum_{v in B} D(u,v)
    - sum_{v in A, v != u} D(u,v)
    - gamma * delta_frequency_regularizer
```

Only accept the move if:

```text
|A| - 1 >= min_pairs
|B| + 1 <= max_pairs
gain > 0
```

This solver is simple, deterministic after seed selection, and easy to implement.

## 7. Variable-Rate Centroid Allocation

After learning groups:

```text
G_1, ..., G_8
```

assign a different centroid count to each subspace:

```text
K_1, ..., K_8
```

The reason is that subspaces have different dimensions and different importance. If `G_i` has more pairs, it has higher dimension:

```text
d_i = 2 * |G_i|
```

Using the same `K` for all subspaces would under-quantize large or important groups and over-quantize small or unimportant groups.

### 7.1 Subspace Importance

Define subspace importance as the sum of pair-level attention contribution:

```text
w_i = sum_{u in G_i} w_u
```

where:

```text
w_u = E_{t,i} |s_u(t, i)|
```

Optionally include output sensitivity:

```text
w_u = E |s_u| + eta * E ||output error when pair u is masked||
```

### 7.2 Rate-Distortion Objective

Let `E_i(K_i)` be the quantization error of subspace `i` when using `K_i` centroids.

The allocation objective:

```text
min_{K_1,...,K_M} sum_i w_i * E_i(K_i)
subject to sum_i log2(K_i) = B
```

`B` is the total index bit budget.

For implementation, restrict `K_i` to hardware-friendly choices:

```text
K_i in {16, 32, 64, 128, 256, 512, 1024}
```

Then solve by dynamic programming or enumeration.

### 7.3 Empirical Allocation

A robust practical method:

1. For each group `G_i`, train candidate codebooks for all allowed `K`.
2. Measure validation error:

   ```text
   E_i(K) = mean ||x_{G_i} - centroid_K(x_{G_i})||^2
   ```

3. Weight it by attention importance:

   ```text
   cost_i(K) = w_i * E_i(K)
   ```

4. Choose `K_i` under total bit budget:

   ```text
   minimize sum_i cost_i(K_i)
   subject to sum_i log2(K_i) <= B
   ```

This is easy to justify because it directly optimizes quantization quality under a fixed memory/search budget.

## 8. Building the Variable-Rate IMI

For each layer/head or layer-group/head:

1. Load the learned partition:

   ```text
   G_1, ..., G_8
   ```

2. Load centroid counts:

   ```text
   K_1, ..., K_8
   ```

3. For each retrieval-zone key `k_i`, form subspace vectors:

   ```text
   x_i^{(m)} = gather_rope_pairs(k_i, G_m)
   ```

4. Assign each subspace vector to its nearest centroid:

   ```text
   c_i^{(m)} = argmin_c ||x_i^{(m)} - C_m[c]||^2
   ```

5. Store the code tuple:

   ```text
   code_i = (c_i^{(1)}, ..., c_i^{(8)})
   ```

6. Build inverted lists for product cells:

   ```text
   cell_id = mixed_radix_encode(code_i, K_1,...,K_8)
   inverted_list[cell_id].append(i)
   ```

Because the virtual cell count is:

```text
K_total_virtual = product_i K_i
```

most cells are empty. Store only non-empty cells using a hash map or sorted compact cell table.

## 9. Query-Time Search

For a decode query `q`:

1. Apply the same RoPE path as the model.
2. Gather query subspace vectors:

   ```text
   q^{(m)} = gather_rope_pairs(q, G_m)
   ```

3. Compute centroid scores per subspace:

   ```text
   score_m[c] = q^{(m)} · C_m[c]
   ```

4. Generate top product cells using multi-sequence top-k over:

   ```text
   score(cell) = sum_m score_m[c_m]
   ```

5. Retrieve candidate tokens from the selected cells.
6. Rerank candidates using exact QK or a higher-precision approximation.
7. Compute attention over:

   ```text
   sink tokens + local tokens + retrieved tokens
   ```

The important point is that the IMI search score remains decomposable:

```text
q · k ≈ sum_m q^{(m)} · centroid_m
```

even though the subspaces are non-uniform.

## 10. Why This Improves IMI Quality

Uniform IMI assumes:

```text
all subspaces have similar importance
all subspaces have similar intrinsic dimension
cross-subspace dependency is negligible
```

These assumptions are weak for RoPE KV keys.

The proposed method changes the assumptions:

```text
RoPE pair is the atomic unit
dependent pairs are grouped together
large/important groups receive more centroids
subspaces are optimized to reduce dependency leakage
```

This should improve:

```text
cell occupancy quality
candidate recall
empty-cell ratio
top-cell score calibration
rerank candidate quality
attention output approximation
```

## 11. Evaluation Plan

### 11.1 Partition Quality

Report:

```text
cross-subspace dependency = sum_{u<v, g(u)!=g(v)} D(u,v)
within-subspace dependency = sum_{u<v, g(u)=g(v)} D(u,v)
empty virtual cell ratio
cell occupancy entropy
```

Compare:

```text
uniform contiguous pair split
uniform frequency split
random balanced split
learned dependency-aware split
```

### 11.2 Retrieval Quality

Measure:

```text
Recall@K against full QK top-K
attention output error
final task accuracy
candidate count needed for target recall
```

### 11.3 Ablations

Important ablations:

```text
pair-preserving vs scalar-dim split
uniform groups vs learned non-uniform groups
same K for all groups vs variable K_i
correlation dependency vs attention-weighted dependency
global partition vs per-layer/head partition
RoPE-space key vs inverse-RoPE canonical key
```

## 12. Recommended Initial Configuration

For Qwen3-8B:

```text
head_dim = 128
num_rope_pairs = 64
num_subspaces = 8
min_pairs = 4
max_pairs = 16
allowed K_i = {32, 64, 128, 256, 512}
partition granularity = per layer group + per KV head
dependency metric = D_corr + lambda * D_attn
lambda = 1.0
```

Layer groups:

```text
early:  layers 0..11
middle: layers 12..23
late:   layers 24..35
```

This gives:

```text
3 layer groups * 8 KV heads = 24 learned partitions
```

which is expressive but still manageable.

## 13. Positioning as an Algorithmic Contribution

The contribution should not be described as simply "using unequal subspaces." The stronger framing is:

```text
We identify that IMI's product-space assumption conflicts with RoPE KV geometry.
We define RoPE pairs as atomic indexing units.
We learn a dependency-minimizing non-uniform partition from calibration KV statistics.
We allocate centroid budgets with a rate-distortion objective tied to attention importance.
```

This makes the method model-aware and distribution-aware, rather than a hand-crafted indexing trick.

