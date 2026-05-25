# RoPE-Pair IMI 子空间划分预实验设计

本文档对应新增的独立实验代码：

```text
experiments/imi_partition/
```

目标是先离线找出 Llama3-8B 上更合理的 IMI 子空间数量 `M` 和每个 KV head 的非均匀 RoPE-pair 划分。这个阶段不修改 decode，不修改现有 C++ 的 IVF / 2-way / 4-way 热路径。

## 1. 实验对象

对某个 layer scope 和 KV head，收集 RoPE 之后的 key：

```text
K ∈ R^{N × 128}
```

因为 Llama/Qwen 使用 NeoX-style RoPE，原子单位不是相邻维度，而是：

```text
p_u = (u, u + 64),   u = 0, ..., 63
```

所以任何子空间划分都必须以 RoPE pair 为最小单位。

## 2. 为什么要同时选择 M 和划分

如果 `M` 大：

```text
每个子空间维度更低
KMeans 更快
单个 codebook 更小
```

但代价是：

```text
IMI product-space 独立性假设更强
强依赖 pair 更可能被切到不同子空间
检索 recall 可能下降
```

所以 `M` 不应该手工固定为 `2/4/8`，而应该做一个 constrained optimization：

```text
尽量让 M 大
同时限制跨子空间依赖泄漏不超过阈值
```

推荐选择规则：

```text
M* = max M
s.t. NormalizedLeak(G_M) <= τ
     min_pairs <= |G_m| <= max_pairs
```

其中 `τ` 是可调阈值，例如先试 `0.85`。如果后续有 retrieval recall 验证，可以加入：

```text
RecallDrop(G_M) <= ε
```

## 3. Pair 依赖矩阵

对每个 pair `u`，从所有校准 token 中取二维向量：

```text
X_u ∈ R^{N × 2}
```

先对每个 pair 内部做 whiten：

```text
X̂_u = (X_u - μ_u) Σ_u^{-1/2}
```

然后计算 pair `u` 和 pair `v` 的 cross-correlation：

```text
C_uv = X̂_u^T X̂_v / (N - 1)
```

基础依赖分数：

```text
D_corr(u,v) = ||C_uv||_F
```

这个指标的含义是：两个 RoPE pair 在 key 分布上的线性依赖强度。

## 4. Gaussian MI 近似

为了让依赖指标更接近信息论意义，可以用 whiten 后的 cross-correlation 估计 Gaussian mutual information：

```text
D_mi(u,v) = -1/2 log det(I - C_uv C_uv^T + εI)
```

如果两个 pair 独立，则 `C_uv` 接近 0，`D_mi` 接近 0。

当前实验代码默认组合：

```text
D(u,v) = norm(D_corr(u,v)) + λ_mi · norm(D_mi(u,v))
```

默认：

```text
λ_mi = 0.25
```

后续如果采集 decode query，可以扩展成：

```text
D = norm(D_corr) + λ_attn norm(D_attn) + λ_mi norm(D_mi)
```

其中：

```text
s_u(t,i) = q_{t,u} k_{i,u} + q_{t,u+64} k_{i,u+64}
D_attn(u,v) = E |s_u s_v| / sqrt(E s_u^2 E s_v^2)
```

## 5. 给定 M 的划分目标

设 `g(u)` 是 pair `u` 的 group id。给定 `M`，目标是最大化组内依赖：

```text
max_G  Σ_{u<v} D(u,v) · 1[g(u)=g(v)]
```

等价于最小化跨子空间依赖泄漏：

```text
min_G  Σ_{u<v} D(u,v) · 1[g(u)≠g(v)]
```

加入弱频率连续性正则：

```text
R(G) = Σ_m Var({u | u ∈ G_m})
```

最终局部搜索优化：

```text
max_G  Within(G) - γ R(G)
```

其中：

```text
Within(G) = Σ_m Σ_{u<v, u,v∈G_m} D(u,v)
```

`γ` 只用于避免非常离散的 group，不能太大。默认先用：

```text
γ = 0.01
```

## 6. 跨 M 比较的关键：NormalizedLeak

不同 `M` 的跨组 pair 数量不一样，所以不能直接比较原始 leakage。

定义：

```text
Total = Σ_{u<v} D(u,v)
Within = Σ_m Σ_{u<v, u,v∈G_m} D(u,v)
Leak = Total - Within
LeakRatio = Leak / Total
```

但 `M` 越大，`LeakRatio` 天然越高。为了公平比较，需要除以同尺寸随机划分的期望泄漏。

对于 group sizes `n_m = |G_m|`：

```text
CrossPairFrac = 1 - Σ_m C(n_m,2) / C(64,2)
RandomLeak ≈ Total · CrossPairFrac
NormalizedLeak = Leak / RandomLeak
```

解释：

```text
NormalizedLeak < 1
```

表示该划分比随机划分更好地把强依赖 pair 留在组内。

选择 `M` 时主要看：

```text
M 越大越好
NormalizedLeak 不超过阈值
group size 不退化
```

## 7. 搜索算法

精确求解是二次整数规划，64 个 pair 虽然不大，但为了预实验稳定和易改，当前实现使用多 seed 局部搜索。

对每个候选 `M`：

```text
M ∈ {2,3,4,5,6,8,10,12,16}
```

执行：

1. 初始化：

```text
连续频率 band 初始化
多个随机 balanced 初始化
```

2. 单点 move：

把 pair `u` 从 group `A` 移到 group `B`，收益：

```text
gain(u, A→B)
= objective(A - u, B + u) - objective(A, B)
```

只有满足 size constraints 才接受：

```text
|A|-1 >= min_pairs
|B|+1 <= max_pairs
gain > 0
```

3. 两点 swap：

交换 `u ∈ A` 和 `v ∈ B`，如果目标提升则接受。

4. 多 seed 取最优。

## 8. 推荐 scope

当前先做两个粒度：

```text
global_head:
    每个 KV head 一套划分，跨所有层聚合统计量

layer_group_head:
    early / middle / late × KV head
```

对于 Llama3-8B，`layer_group_head` 输出：

```text
3 × 8 = 24 套划分
```

这是推荐默认。它比 per-layer/head 更稳定，元数据量也更可控。

## 9. 使用流程

### 9.1 采集 RoPE 后 key

可以从本地 JSONL prompt 文件采集：

```bash
python -m experiments.imi_partition.collect_llama3_rope_keys \
  --model llama-3-8b-1048k \
  --jsonl data/longbenchv2_calib.jsonl \
  --text-field input \
  --limit 32 \
  --max-length 32768 \
  --static-start 4 \
  --static-end 1024 \
  --out outputs/imi_partition/llama3_keys.pt
```

也可以使用 HuggingFace dataset：

```bash
python -m experiments.imi_partition.collect_llama3_rope_keys \
  --model llama-3-8b-1048k \
  --hf-dataset <LongBenchV2 dataset id> \
  --split train \
  --text-field input \
  --limit 32 \
  --out outputs/imi_partition/llama3_keys.pt
```

输出 key layout：

```text
[layers, kv_heads, tokens, head_dim]
```

### 9.2 计算依赖矩阵

```bash
python -m experiments.imi_partition.compute_dependency \
  --keys outputs/imi_partition/llama3_keys.pt \
  --layout lhtd \
  --scope layer_group_head \
  --layer-groups 3 \
  --out outputs/imi_partition/llama3_dependency.pt
```

### 9.3 搜索不同 M 的最优划分

```bash
python -m experiments.imi_partition.search_partitions \
  --dependency outputs/imi_partition/llama3_dependency.pt \
  --m-values 2,3,4,5,6,8,10,12,16 \
  --min-pairs 2 \
  --max-pairs 16 \
  --gamma 0.01 \
  --max-normalized-leak 0.85 \
  --out-json outputs/imi_partition/llama3_partitions.json \
  --out-csv outputs/imi_partition/llama3_partition_summary.csv
```

## 10. 如何解读结果

每个 scope 会有多个 candidate：

```text
M
groups
group_sizes
leak_ratio
normalized_leak
locality_penalty
```

优先看：

```text
selected.M
selected.metrics.normalized_leak
selected.metrics.group_sizes
```

如果某个 head 的 `M=12` 满足阈值，而另一个 head 只能到 `M=6`，这说明不同 head 的 pair dependency 结构不同，应该允许 head-wise `M` 和划分不同。

## 11. 后续验证

当前指标只证明“子空间更独立”，还不能完全证明 retrieval 不掉精度。下一步应该增加 offline retrieval proxy：

```text
full QK top-K 作为 oracle
不同 M/partition 构建临时代码本
比较 candidate recall
比较 empty cell ratio / occupancy entropy
比较 attention output error
```

只有当：

```text
NormalizedLeak 好
Recall@K 不明显下降
构建时间下降
```

同时成立，才值得把对应 `M` 和非均匀划分接入 C++ 热路径。

