# hybrid_8way_v2 Decode 搜索模式说明

本文档只描述 `AdaptiveIMI` 的 `hybrid_8way_v2` 路径下，decode 阶段如何从当前 query 选择 IMI cluster，并把这些 cluster 对应的 KV 送进 attention。

重点结论：

- 当前 decode 搜索模式一共有 4 个可见模式：`flat`、`subspace`、`subspace_rerank`、`hybrid`。
- `None` 或 `flat`：走完整 centroid 相似度搜索。
- `subspace`：走 8 路 learned subspace 粗筛，粗筛结果直接给 attention。
- `subspace_rerank`：先用 8 路 learned subspace 粗筛，再用完整 centroid 精搜重排。
- `hybrid`：当前代码里等价于 `subspace_rerank`，只是名字不同。
- 真正进入 attention 的不是 token id，而是 `selected` cluster id 矩阵。后续 retrieval 会根据这些 cluster id 把对应 KV 拷贝到 execution buffer，再由 `weighted_flash_decoding` 做 attention。

## 1. 配置入口

### 1.1 命令行参数

RULER 路径的入口在：

- `benchmark/ruler/pred/call_api.py`

相关参数：

```python
parser.add_argument("--decode_search_mode", type=str, default=None, choices=[None, "flat", "subspace", "subspace_rerank", "hybrid"])
parser.add_argument("--candidate_clusters", type=int, default=None)
parser.add_argument("--rerank_clusters", type=int, default=None)
```

`get_pred()` 中会把命令行参数转换成 `decode_search` 字典：

```python
decode_search = None
if decode_search_mode:
    decode_search = {"mode": decode_search_mode}
    if candidate_clusters is not None:
        decode_search["candidate_clusters"] = int(candidate_clusters)
    if rerank_clusters is not None:
        decode_search["rerank_clusters"] = int(rerank_clusters)
```

因此：

```bash
DECODE_SEARCH_MODE=subspace
```

最终会变成：

```python
decode_search = {"mode": "subspace"}
```

如果额外设置：

```bash
CANDIDATE_CLUSTERS=4096
RERANK_CLUSTERS=1024
```

则传入：

```python
decode_search = {
    "mode": "...",
    "candidate_clusters": 4096,
    "rerank_clusters": 1024,
}
```

### 1.2 config 生成

配置由 `config/config.py::generate_config()` 写入 `AdaptiveIMI` 配置段：

```python
if decode_search is not None:
    base_config["decode_search"] = decode_search
```

同时，`hybrid_8way_v2` 会强制：

```python
subspace_parts = 8
```

并要求提供 `partition_path` 或 `learned_partition_path`。

### 1.3 KV cache 初始化

`model_hub/base.py::_build_kv_cache()` 把 `decode_search` 继续传给 `adpimi_cache`：

```python
runtime_config={
    ...
    "partition_path": imi_config.get("partition_path") or imi_config.get("learned_partition_path"),
    "hybrid_index": imi_config.get("hybrid_index", {}),
    "decode_search": imi_config.get("decode_search", {}),
}
```

`cache_hub/adaptive_imi_v2/cache.py::__init__()` 保存为：

```python
self.decode_search_config = dict(runtime_config.get("decode_search") or {})
```

这个字典是 decode 阶段选择搜索模式和预算的唯一运行时来源。

## 2. Decode 搜索模式总表

| 模式 | `_decode_search_enabled()` | selector | 搜索方式 | 是否需要 search metadata | 是否精搜 |
|---|---:|---|---|---:|---:|
| `None` | 否 | 基类默认路径 | full centroid flat search | 否 | 否 |
| `flat` | 否 | `_select_clusters_flat()` | full centroid flat search | 否 | 否 |
| `subspace` | 是 | `_select_clusters_for_decode()` | subspace 粗筛 top-k | 是 | 否 |
| `subspace_rerank` | 是 | `_select_clusters_for_decode()` | subspace 粗筛 + full centroid rerank | 是 | 是 |
| `hybrid` | 是 | `_select_clusters_for_decode()` | 当前等价于 `subspace_rerank` | 是 | 是 |

判断函数在 `cache_hub/adaptive_imi_v2/cache.py`：

```python
def _decode_search_enabled(self) -> bool:
    mode = str(self.decode_search_config.get("mode", "flat")).lower()
    return mode in ("subspace_rerank", "subspace", "hybrid")
```

也就是说：

- `mode` 缺省是 `"flat"`。
- 只有 `subspace_rerank`、`subspace`、`hybrid` 会启用 hybrid decode search metadata。
- `flat` 不会保留 subspace decode search metadata，直接用普通 centroid 搜索。

## 3. Decode 之前：prefill/build 阶段生成了什么

Decode 搜索不是临时构建索引。它依赖 prefill 阶段已经构建好的以下数据：

### 3.1 KV 分区

输入 prompt 的 KV 被分成两块：

1. static zone
   - 保存 prompt 开头 `static_pattern_start` 个 token。
   - 保存 prompt 结尾 `static_pattern_end` 个 token。
   - decode 新 token 也会追加到 static zone 之后。

2. list zone
   - 保存中间长上下文。
   - IMI 建索引只对这段 middle KV 做。

相关代码：

```python
self.static_pattern_start = static_pattern_start
self.static_pattern_end = static_pattern_end
self.static_pattern_total = self.static_pattern_start + self.static_pattern_end

self.list_capacity = max(self.capacity_input_length - self.static_pattern_total, 0)
self.list_stride = max(self.input_length - self.static_pattern_total, 0)
```

prefill 时：

```python
middle_start = valid_start + self.static_pattern_start
middle_end = seq_len - self.static_pattern_end
```

middle KV 被送入 hybrid_8way_v2 建索引；头尾 KV 被拷贝进：

```python
self.steady_zone_keys[layer_idx]
self.steady_zone_values[layer_idx]
```

### 3.2 list_keys/list_values

每层建索引完成后，重排后的 middle KV 会写入：

```python
self.list_keys[layer_idx]
self.list_values[layer_idx]
```

它们是后续 retrieval gather 的主数据源。

重要形状概念：

```text
list_keys[layer] / list_values[layer]:
  [batch_size, kv_heads, list_capacity, head_dim]
```

`batch_groups = batch_size * kv_heads`，decode 搜索中的每一行对应一个 batch/head group。

### 3.3 普通 metadata：cluster_sizes / cluster_offsets / centroids

无论 flat 还是 subspace 模式，最终都需要普通 cluster metadata：

```python
self.cluster_sizes_cpu[layer_idx]
self.cluster_offsets_cpu[layer_idx]
self.cluster_sizes_gpu[layer_idx]
self.centroids[layer_idx]
self.centroids_mask[layer_idx]
```

这些由 `_load_layer_metadata()` 从 build 结果加载。

每个 head/group 有自己的 cluster 表：

```text
cluster_sizes:
  [batch_groups, n_centroids]

cluster_offsets:
  [batch_groups, n_centroids + 1]

centroids:
  [batch_groups, n_centroids, head_dim]

centroids_mask:
  [batch_groups, n_centroids]
```

`padding_cluster_id = n_centroids - 1`，用于填充无效 cluster。

### 3.4 subspace search metadata

`subspace`、`subspace_rerank`、`hybrid` 需要额外 metadata：

```python
self.hybrid_search_metadata_cpu[layer_idx]
self.hybrid_search_metadata_gpu[layer_idx]
```

必需字段：

```python
required = (
    "subspace_centroids",
    "subspace_centroid_masks",
    "subspace_dim_ids",
    "subspace_dim_masks",
    "compact_codes",
)
```

这些字段的含义：

| 字段 | 位置 | 形状 | 含义 |
|---|---|---|---|
| `subspace_centroids` | GPU | `[batch_groups, 8, max_k, max_sub_dim]` | 每个 head 的 8 个子空间 centroid |
| `subspace_centroid_masks` | GPU | `[batch_groups, 8, max_k]` | 标记无效子空间 centroid |
| `subspace_dim_ids` | GPU | `[batch_groups, 8, max_sub_dim]` | 每个子空间映射回 full head_dim 的维度 id |
| `subspace_dim_masks` | GPU | `[batch_groups, 8, max_sub_dim]` | 标记无效维度 |
| `compact_codes` | GPU | `[batch_groups, active_clusters, 8]` | compact cluster 对应的 8 个子空间 code |

`compact_codes[row, cluster, m] = code` 表示：

```text
第 row 个 batch/head group
第 cluster 个 compact cluster
在第 m 个子空间里对应第 code 个 subspace centroid
```

decode 时正是通过这个表，把 8 个子空间分数合成 compact cluster 粗筛分数。

## 4. CPU build 路径的 metadata 来源

CPU pipeline 路径最终从 C++ pipeline metadata 取数据。

Python 包装在：

```python
library/AdaptiveIMI/imi_adapter.py
```

`finish_index()` 中：

```python
metadata = pipeline.get_metadata_result()
search_keys = (
    "subspace_centroids",
    "subspace_centroid_masks",
    "subspace_dim_ids",
    "subspace_dim_masks",
    "compact_codes",
    "token_ids_by_cluster",
    "cluster_sizes",
    "cluster_offsets",
    "cluster_counts",
)
self._last_search_metadata_cpu = {
    key: metadata[key]
    for key in search_keys
    if key in metadata and torch.is_tensor(metadata[key])
}
```

C++ 端 `pipeline_bindings.cpp` 里 `get_metadata_result()` 返回的关键字段包括：

```text
head_indices:               int64   [kv_heads]
cluster_counts:             int32   [kv_heads]
cluster_sizes:              int32   [kv_heads, max_clusters]
cluster_offsets:            int32   [kv_heads, max_clusters + 1]
centroids:                  float32 [kv_heads, max_clusters, dim]
token_ids_by_cluster:       int32   [kv_heads, n_tokens]
compact_codes:              int32   [kv_heads, max_clusters, subspace_parts]
subspace_centroids:         float32 [kv_heads, subspace_parts, max_k, max_sub_dim]
subspace_centroid_masks:    bool    [kv_heads, subspace_parts, max_k]
subspace_dim_ids:           int64   [kv_heads, subspace_parts, max_sub_dim]
subspace_dim_masks:         bool    [kv_heads, subspace_parts, max_sub_dim]
```

随后 `cache_hub/adaptive_imi_v2/cache.py::_finish_pipeline_layer_index()` 做：

```python
search_metadata_cpu = pipeline.get_last_search_metadata_cpu()
...
if self._decode_search_enabled() and search_metadata_cpu is not None:
    device = self.layer_mapping[str(layer_idx)]
    self.hybrid_search_metadata_cpu[layer_idx] = search_metadata_cpu
    self.hybrid_search_metadata_gpu[layer_idx] = {
        key: value.to(device=device, non_blocking=True)
        for key, value in search_metadata_cpu.items()
        if key in (
            "subspace_centroids",
            "subspace_centroid_masks",
            "subspace_dim_ids",
            "subspace_dim_masks",
            "compact_codes",
        )
    }
```

所以 CPU build 的 subspace decode metadata 起点在 CPU，decode 前被搬到当前 layer 所在 GPU。

## 5. GPU build 路径的 metadata 来源

GPU build 路径在：

```python
cache_hub/adaptive_imi_v2/gpu_builder.py
```

核心函数：

```python
_build_metadata_batched_cuda()
```

它准备 learned partition tensor：

```python
subspace_dim_ids
subspace_dim_masks
subspace_pair_ids
subspace_pair_masks
```

然后调用 CUDA 扩展：

```python
ext.run_hybrid_8way_kmeans(
    keys.contiguous(),
    row_token_counts.contiguous(),
    subspace_dim_ids.contiguous(),
    subspace_dim_masks.contiguous(),
    k_values_tensor,
    int(self.max_iters),
    int(max_k),
    int(dense_code_space),
)
```

CUDA 返回：

```python
(
    _labels_cuda,
    subspace_centroids_cuda,
    subspace_centroid_masks_cuda,
    token_labels,
    token_ids_by_cluster,
    compact_codes,
    cluster_sizes,
    cluster_offsets,
    cluster_counts,
    full_centroids,
)
```

随后生成 metadata：

```python
metadata = {
    "cluster_counts": cluster_counts.to(torch.int32),
    "cluster_sizes": cluster_sizes.to(torch.int32),
    "cluster_offsets": cluster_offsets.to(torch.int32),
    "centroids": full_centroids.to(torch.float32),
    "token_labels": token_labels.to(torch.int32),
    "token_ids_by_cluster": token_ids_by_cluster.to(torch.int32),
    "compact_codes": compact_codes.to(torch.int32),
    "subspace_centroids": subspace_centroids_cuda.to(torch.float32),
    "subspace_centroid_masks": subspace_centroid_masks_cuda.to(torch.bool),
    "subspace_dim_ids": subspace_dim_ids,
    "subspace_dim_masks": subspace_dim_masks,
    ...
}
```

`finish()` 里会构造 `search_cpu` 和 `search_gpu`：

```python
search_cpu = {
    key: metadata_cpu[key]
    for key in search_cpu_keys
    if key in metadata_cpu
}

if self.build_device == "gpu":
    search_gpu = {
        "subspace_centroids": self._metadata_device["subspace_centroids"],
        "subspace_centroid_masks": self._metadata_device["subspace_centroid_masks"],
        "subspace_dim_ids": self._metadata_device["subspace_dim_ids"],
        "subspace_dim_masks": self._metadata_device["subspace_dim_masks"],
    }
    search_gpu["compact_codes"] = self._metadata_device["compact_codes"][:, :active_clusters, :]
```

然后 `_finish_layer_index()` 保存：

```python
self.hybrid_search_metadata_cpu[layer_idx] = result.search_metadata_cpu
self.hybrid_search_metadata_gpu[layer_idx] = result.search_metadata_gpu
```

所以 GPU build 的 subspace decode metadata 主要已经在 GPU 上，CPU 侧只保存一份用于 debug/统计/回退。

## 6. Decode 每一步的调用时机

### 6.1 prefill 结束后切到 sparse attention

prefill 期间 attention 仍然是 dense/full prefill。每层 build 完后，最后一层 prefill 结束时：

```python
if self.build_index_when_prefilling:
    self.attn_func = self.sparse_attention
```

也就是说 decode 阶段才会走 IMI sparse attention。

### 6.2 每个新 token 的 decode loop

`model_hub/base.py::inference()`：

```python
for _ in range(self.max_new_length - 1):
    logits = self.decode_forward(inputs_ids=output_ids)
    output_ids = self.sampling(logits, ...)
```

`decode_forward()` 会逐层调用：

```python
hidden_states = self.layer_decode(ldx, hidden_states)
```

`layer_decode()` 中：

```python
key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
```

`decode_update_kv_cache()` 会把当前 decode token 的 KV 写进 static zone：

```python
self.steady_zone_keys[layer_idx][:, :, self.static_pattern_total, :] = key_states[:, 0, :, :]
self.steady_zone_values[layer_idx][:, :, self.static_pattern_total, :] = value_states[:, 0, :, :]
```

最后一层后更新：

```python
self.context += 1
self.static_pattern_total += 1
```

### 6.3 Attention 调用到 sparse_attention

模型侧 `decode_attention()` 会调用：

```python
imi_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
```

`attn_hub/sparse_imi.py::imi_decode_attn()`：

```python
static_len = (
    imi_cache.static_pattern_total
    if layer_idx == imi_cache.layer_num - 1
    else imi_cache.static_pattern_total + 1
)
return imi_cache.attn_func(query_states.contiguous(), layer_idx, static_len)
```

由于 prefill 后 `attn_func = self.sparse_attention`，所以最终进入：

```python
cache_hub/adaptive_imi/retrieval.py::sparse_attention()
```

这是 decode 搜索和 retrieval 的主执行函数。

## 7. sparse_attention 的真实执行顺序

入口：

```python
def sparse_attention(self, queries, layer_idx, static_len):
```

关键输入：

```text
queries:
  [batch_size, 1, num_heads, head_dim]

layer_idx:
  当前 transformer layer

static_len:
  当前 static zone 中可见 token 长度
```

### 7.1 保证当前层 index ready

```python
self.ensure_layer_ready(layer_idx)
```

如果当前 layer 的 build job 还没 finish，这里会等待并完成 metadata/load。

### 7.2 应用异步 centroid 更新

```python
self._apply_async_centroid_updates(layer_idx, device)
```

如果启用了 async update，并且上一轮 centroid shadow 已经准备好，会把 shadow copy 到：

```python
self.centroids[layer_idx]
```

### 7.3 设置 static length

```python
self.static_len_tensor_dict[device].fill_(static_len)
self.static_lengths_dict[device].fill_(static_len)
```

这些长度会传给 attention kernel，控制 static zone 参与 attention 的 token 数。

### 7.4 query reshape

```python
query_grouped = queries.view(self.batch_groups, self.group_size, self.head_dim).contiguous()
```

含义：

```text
batch_groups = batch_size * kv_heads
group_size = num_heads / kv_heads
```

GQA/MQA 下，一个 KV head 对应多个 query head。这里把 query 按 KV group 聚合。

后续 flat/subspace 两类搜索都基于这个 `query_grouped`。

### 7.5 选择 cluster selector

```python
custom_selector = getattr(self, "_select_clusters_for_decode", None)
if custom_selector is None:
    ...
else:
    selected = custom_selector(layer_idx, query_grouped, device)
```

在 `hybrid_8way_v2` 中，`cache_hub/adaptive_imi_v2/cache.py` 定义了 `_select_clusters_for_decode()`，所以会走 custom selector。

如果 mode 是 `flat` 或未设置，custom selector 内部会转到 `_select_clusters_flat()`。

输出：

```text
selected:
  int64 Tensor [batch_groups, nprobe]
```

每一行是一组要取回的 compact cluster id。无效位置填 `padding_cluster_id`。

## 8. flat 模式

### 8.1 触发条件

以下情况走 flat：

- `DECODE_SEARCH_MODE` 未设置。
- `DECODE_SEARCH_MODE=flat`。
- `subspace/subspace_rerank/hybrid` metadata 不存在，并且 `fallback_flat=True`。
- `subspace_dim_ids` 行数和 `batch_groups` 不匹配，并且 `fallback_flat=True`。

### 8.2 执行函数

```python
_select_clusters_flat(layer_idx, query_grouped, device)
```

核心代码：

```python
similarities = imi_gpu_kernels.fused_query_group_similarities(
    query_grouped,
    self.centroids[layer_idx],
    self.similarity_buffer_dict[device],
)
similarities.masked_fill_(self.centroids_mask[layer_idx], float("-inf"))
selected = self._select_clusters(layer_idx, similarities)
```

### 8.3 输入数据位置

| 数据 | 位置 | 说明 |
|---|---|---|
| `query_grouped` | 当前 GPU | 当前 decode token 的 query，按 KV group reshape |
| `self.centroids[layer_idx]` | 当前 GPU | full centroid，来自 build metadata |
| `self.centroids_mask[layer_idx]` | 当前 GPU | 无效 cluster mask |
| `self.cluster_sizes_gpu[layer_idx]` | 当前 GPU | 每个 cluster 的 token 数 |

### 8.4 算法

`fused_query_group_similarities()` 的 fallback 等价于：

```python
aggregated = query_grouped.float().sum(dim=1)
similarities = torch.einsum("hd,hnd->hn", aggregated, centroids.float())
```

即：

```text
每个 batch/head group:
  1. 把 group_size 个 query head 求和，得到 [head_dim] query
  2. 和所有 full centroids 做点积
  3. 得到 [n_centroids] 相似度
```

然后 `_select_clusters()`：

```python
topk = torch.topk(similarities, self.nprobe, dim=-1, largest=True, sorted=True)
sizes = torch.gather(cluster_sizes, 1, topk.indices)
cumsum = torch.cumsum(sizes, dim=-1, dtype=torch.int32)
keep = valid & ((cumsum - sizes) < self.max_retrieval_tokens)
selected = topk.indices.masked_fill(~keep, self.padding_cluster_id)
```

最终：

```text
候选 cluster 数 = nprobe
token budget = max_retrieval_tokens
超过 token budget 后的 cluster 被置为 padding_cluster_id
```

## 9. subspace 模式

### 9.1 触发条件

```bash
DECODE_SEARCH_MODE=subspace
```

且当前 layer 有完整 search metadata：

```python
subspace_centroids
subspace_centroid_masks
subspace_dim_ids
subspace_dim_masks
compact_codes
```

### 9.2 执行函数

```python
_select_clusters_for_decode(layer_idx, query_grouped, device)
```

`mode == "subspace"` 时，执行粗筛后直接 return，不进入 full centroid rerank。

### 9.3 输入数据位置

| 数据 | 位置 | 说明 |
|---|---|---|
| `query_grouped` | 当前 GPU | 当前 decode query |
| `hybrid_search_metadata_gpu[layer_idx]["subspace_dim_ids"]` | 当前 GPU | 每个子空间对应 full dim 的维度 id |
| `hybrid_search_metadata_gpu[layer_idx]["subspace_dim_masks"]` | 当前 GPU | 子空间维度 mask |
| `hybrid_search_metadata_gpu[layer_idx]["subspace_centroids"]` | 当前 GPU | 子空间 centroid |
| `hybrid_search_metadata_gpu[layer_idx]["subspace_centroid_masks"]` | 当前 GPU | 子空间 centroid mask |
| `hybrid_search_metadata_gpu[layer_idx]["compact_codes"]` | 当前 GPU | compact cluster 到 8 个子空间 code 的映射 |
| `cluster_sizes_gpu[layer_idx]` | 当前 GPU | compact cluster size |

### 9.4 算法步骤

第一步，把 GQA group 内 query 聚合成一个 full query：

```python
query = query_grouped.float().sum(dim=1)
```

形状：

```text
query:
  [batch_groups, head_dim]
```

第二步，根据 learned partition 的维度表，从 full query 中抽取每个子空间 query：

```python
dim_ids = metadata["subspace_dim_ids"].to(device=query_grouped.device)
dim_masks = metadata["subspace_dim_masks"].to(device=query_grouped.device)
dim_safe = dim_ids.clamp_min(0)
query_expanded = query.unsqueeze(1).expand(-1, dim_safe.size(1), -1)
query_sub = torch.gather(query_expanded, 2, dim_safe)
query_sub = query_sub.masked_fill(dim_masks, 0.0)
```

形状：

```text
query_sub:
  [batch_groups, 8, max_sub_dim]
```

第三步，计算每个子空间内 query 到所有子空间 centroid 的分数：

```python
subspace_scores = torch.einsum("bmd,bmkd->bmk", query_sub, subspace_centroids)
subspace_scores = subspace_scores.masked_fill(subspace_masks, float("-inf"))
```

形状：

```text
subspace_scores:
  [batch_groups, 8, max_k]
```

含义：

```text
subspace_scores[row, m, code]
  = 第 row 个 batch/head group
    第 m 个子空间
    第 code 个子空间 centroid 的相似度
```

第四步，用 `compact_codes` 把子空间分数汇总成 compact cluster 粗筛分数：

```python
compact_codes = metadata["compact_codes"].to(device=query_grouped.device)
codes_by_subspace = compact_codes.permute(0, 2, 1).to(torch.long)
valid_codes = codes_by_subspace >= 0
code_scores = torch.gather(subspace_scores, 2, codes_by_subspace.clamp_min(0))
coarse_scores = code_scores.masked_fill(~valid_codes, 0.0).sum(dim=1)
valid_compact = valid_codes.all(dim=1)
coarse_scores = coarse_scores.masked_fill(~valid_compact, float("-inf"))
```

形状：

```text
compact_codes:
  [batch_groups, active_clusters, 8]

codes_by_subspace:
  [batch_groups, 8, active_clusters]

coarse_scores:
  [batch_groups, active_clusters]
```

粗筛分数含义：

```text
coarse_scores[row, cluster]
  = sum_m subspace_scores[row, m, compact_codes[row, cluster, m]]
```

第五步，`subspace` 模式直接取粗筛 top-k：

```python
selected_width = min(self.nprobe, active_clusters)
coarse_topk = torch.topk(coarse_scores, selected_width, dim=-1, largest=True, sorted=True)
ranked_ids = coarse_topk.indices
```

第六步，按 cluster size 和 token budget 做截断：

```python
ranked_sizes = torch.gather(cluster_sizes, 1, ranked_ids)
cumsum = torch.cumsum(ranked_sizes, dim=-1, dtype=torch.int32)
keep = (ranked_sizes > 0) & ((cumsum - ranked_sizes) < self.max_retrieval_tokens)
```

第七步，构造 `selected`：

```python
selected = torch.full(
    (batch_groups, self.nprobe),
    self.padding_cluster_id,
    dtype=torch.int64,
    device=query_grouped.device,
)
selected[:, :selected_width] = ranked_ids.masked_fill(~keep, self.padding_cluster_id)
return selected
```

### 9.5 subspace 模式不会执行的内容

`subspace` 模式不会执行：

```python
candidate_centroids = torch.gather(...)
rerank_scores = torch.einsum(...)
reranked = torch.topk(rerank_scores, ...)
```

因此它是“粗筛后即 attention”的模式。

## 10. subspace_rerank 模式

### 10.1 触发条件

```bash
DECODE_SEARCH_MODE=subspace_rerank
```

### 10.2 执行函数

同样进入：

```python
_select_clusters_for_decode(layer_idx, query_grouped, device)
```

前半段和 `subspace` 完全一样，都会生成：

```python
coarse_scores
```

不同点是：`subspace_rerank` 不直接返回粗筛 top-k，而是先取更大的 candidate set，然后用 full centroid 做精搜。

### 10.3 粗筛 candidate 预算

```python
default_candidates = max(self.nprobe * 4, 1024)
candidate_limit = int(self.decode_search_config.get("candidate_clusters", default_candidates))
candidate_limit = max(1, min(max(candidate_limit, self.nprobe), active_clusters))
candidate_ids = torch.topk(coarse_scores, candidate_limit, dim=-1, largest=True, sorted=False).indices
```

默认：

```text
candidate_limit = max(4 * nprobe, 1024)
```

然后被限制到：

```text
nprobe <= candidate_limit <= active_clusters
```

如果用户设置了 `CANDIDATE_CLUSTERS`，则用用户值，但仍然会被 clamp。

### 10.4 full centroid 精搜

取出粗筛 candidate 对应的 full centroid：

```python
centroids = self.centroids[layer_idx].to(device=query_grouped.device)
candidate_centroids = torch.gather(
    centroids,
    1,
    candidate_ids.unsqueeze(-1).expand(-1, -1, self.head_dim),
)
```

形状：

```text
candidate_centroids:
  [batch_groups, candidate_limit, head_dim]
```

计算 full query 和 full centroid 的点积：

```python
rerank_scores = torch.einsum("bd,bkd->bk", query, candidate_centroids)
```

形状：

```text
rerank_scores:
  [batch_groups, candidate_limit]
```

过滤空 cluster：

```python
cluster_sizes = self.cluster_sizes_gpu[layer_idx].to(device=query_grouped.device)
candidate_sizes = torch.gather(cluster_sizes, 1, candidate_ids)
valid = candidate_sizes > 0
rerank_scores = rerank_scores.masked_fill(~valid, float("-inf"))
```

### 10.5 rerank 预算

```python
rerank_limit = int(self.decode_search_config.get("rerank_clusters", self.nprobe))
rerank_limit = max(1, min(max(rerank_limit, self.nprobe), candidate_limit))
```

默认：

```text
rerank_limit = nprobe
```

如果用户设置了 `RERANK_CLUSTERS`，则用用户值，但仍然会被 clamp：

```text
nprobe <= rerank_limit <= candidate_limit
```

注意：后面真正写入 `selected` 的宽度是：

```python
selected_width = min(self.nprobe, rerank_limit)
```

因为 `rerank_limit >= nprobe`，所以实际最终仍然只写入 `nprobe` 个 cluster。

也就是说，当前代码中 `RERANK_CLUSTERS > nprobe` 只会影响中间 `topk(rerank_scores, rerank_limit)` 的规模，不会增加最终 attention 的 cluster 数。

### 10.6 选择最终 cluster

```python
reranked = torch.topk(rerank_scores, rerank_limit, dim=-1, largest=True, sorted=True)
ranked_ids = torch.gather(candidate_ids, 1, reranked.indices)
ranked_sizes = torch.gather(cluster_sizes, 1, ranked_ids)

selected_width = min(self.nprobe, rerank_limit)
ranked_ids = ranked_ids[:, :selected_width]
ranked_sizes = ranked_sizes[:, :selected_width]
```

再按 token budget 截断：

```python
cumsum = torch.cumsum(ranked_sizes, dim=-1, dtype=torch.int32)
keep = (ranked_sizes > 0) & ((cumsum - ranked_sizes) < self.max_retrieval_tokens)
```

构造最终 `selected`：

```python
selected = torch.full(
    (batch_groups, self.nprobe),
    self.padding_cluster_id,
    dtype=torch.int64,
    device=query_grouped.device,
)
selected[:, :selected_width] = ranked_ids.masked_fill(~keep, self.padding_cluster_id)
return selected
```

## 11. hybrid 模式

### 11.1 当前行为

```bash
DECODE_SEARCH_MODE=hybrid
```

会启用 `_decode_search_enabled()`，因此进入 `_select_clusters_for_decode()`。

但是当前代码只有一个特殊分支：

```python
if mode == "subspace":
    ...
    return selected
```

`hybrid` 不等于 `subspace`，所以会继续执行后面的 two-stage rerank 逻辑。

因此当前：

```text
hybrid == subspace_rerank
```

除非后续代码给 `hybrid` 加新的分支，否则它只是 `subspace_rerank` 的别名。

## 12. selected cluster 如何进入 retrieval

无论哪种搜索模式，最终都会得到：

```text
selected:
  [batch_groups, nprobe]
```

后续在 `sparse_attention()` 中分两种执行路径。

### 12.1 非 tile 路径

默认：

```bash
IMI_MAX_TILES=0
```

代码：

```python
selected_cpu = selected.to(device="cpu", dtype=torch.int64)
tiles = [selected_cpu]
use_tiles = False
```

执行：

```python
self.valid_lengths_dict[device].zero_()
retrieval_timing = self._run_retrieval_full(layer_idx, device, selected_cpu, profile_decode=profile)
self._append_delta_to_execution(layer_idx, device, selected_cpu)
acc_out = weighted_flash_decoding(...)
```

### 12.2 tile 路径

如果设置：

```bash
IMI_MAX_TILES != 0
```

会调用：

```python
tiles = self._build_cluster_tiles(layer_idx, selected)
use_tiles = True
```

`_build_cluster_tiles()` 会根据 `cluster_sizes_cpu`、`page_size`、`buffer_size` 把 selected clusters 切成多个 tile，确保每个 tile 的 page 数不超过 retrieval buffer。

tile 模式会先对 static zone 做一次 attention：

```python
static_out, static_lse = weighted_flash_decoding(
    query,
    self.steady_zone_keys[layer_idx],
    self.steady_zone_values[layer_idx],
    cache_seqlens=self.static_lengths_dict[device],
    return_softmax_lse=True,
)
```

然后每个 tile 重复：

```python
self.valid_lengths_dict[device].zero_()
self._run_retrieval_tile(layer_idx, device, tile_cluster_ids)
self._append_delta_to_execution(layer_idx, device, tile_cluster_ids)
acc_out, acc_lse = weighted_flash_decoding(
    query,
    self.execution_buffer_keys_dict[device],
    self.execution_buffer_values_dict[device],
    previous_out=acc_out,
    previous_lse=acc_lse,
    cache_seqlens=self.valid_lengths_dict[device],
    return_softmax_lse=True,
)
```

因此 tile 模式下 attention 是分块累积的。

## 13. retrieval 的真实执行步骤

### 13.1 `_run_retrieval_full()`

输入：

```python
cluster_ids: CPU int64 Tensor [batch_groups, nprobe]
```

第一步，把 cluster ids 写入 native index 输入 buffer：

```python
self._set_cluster_ids(layer_idx, cluster_ids)
```

第二步，调用 native index 做 cache hit/miss 解析：

```python
self.adpimi_index[layer_idx].batch_access()
```

这里会根据 `cluster_offsets`、cache 状态等信息生成：

```python
self.miss_unit_idices[layer_idx]
self.miss_unit_sizes[layer_idx]
self.miss_unit_sizes_cumsum[layer_idx]
self.miss_num_units[layer_idx]
self.hit_unit_idices[layer_idx]
self.hit_unit_sizes[layer_idx]
self.hit_unit_sizes_cumsum[layer_idx]
self.hit_num_units[layer_idx]
```

第三步，把 static zone、list zone、cache zone 合并到 execution buffer：

```python
gather_copy_and_concat(
    self.steady_zone_keys[layer_idx],
    self.list_keys[layer_idx],
    self.cache_keys[layer_idx],
    self.execution_buffer_keys_dict[device],
    self.steady_zone_values[layer_idx],
    self.list_values[layer_idx],
    self.cache_values[layer_idx],
    self.execution_buffer_values_dict[device],
    ...
    self.static_stride,
    self.list_capacity,
    self.cache_size,
    self.execution_stride,
    self.buffer_size,
    self.static_len_tensor_dict[device],
)
```

注意这里传给 list source 的 stride 是：

```python
self.list_capacity
```

这是因为 `list_keys/list_values` 的物理布局按 `list_capacity` 存储。

第四步，等待 native index 的 H2D/cache 同步：

```python
self.adpimi_index[layer_idx].sync()
```

第五步，把 miss 的内容更新回 GPU cache：

```python
gather_copy_and_scatter(...)
```

此时：

```python
self.execution_buffer_keys_dict[device]
self.execution_buffer_values_dict[device]
self.valid_lengths_dict[device]
```

已经准备好给 attention 使用。

### 13.2 `_run_retrieval_tile()`

tile 路径和 full 路径类似，但区别是：

- tile attention 已经单独处理 static zone。
- retrieval tile 只需要把 list/cache 检索结果放进 retrieval execution buffer。

核心函数：

```python
gather_copy_and_concat_retrieval(
    self.list_keys[layer_idx],
    self.cache_keys[layer_idx],
    self.execution_buffer_keys_dict[device],
    self.list_values[layer_idx],
    self.cache_values[layer_idx],
    self.execution_buffer_values_dict[device],
    ...
    self.list_capacity,
    self.cache_size,
    self.retrieval_execution_stride,
    self.buffer_size,
)
```

随后同样：

```python
self.adpimi_index[layer_idx].sync()
gather_copy_and_scatter(...)
```

## 14. Attention 消费的数据

非 tile 路径：

```python
acc_out = weighted_flash_decoding(
    query_view,
    self.execution_buffer_keys_dict[device],
    self.execution_buffer_values_dict[device],
    previous_out=None,
    previous_lse=None,
    cache_seqlens=self.valid_lengths_dict[device],
    return_softmax_lse=False,
)
```

输入形状概念：

```text
query_view:
  [batch_groups, 1, group_size, head_dim]

execution_buffer_keys / values:
  [batch_groups, execution_stride, 1, head_dim]

valid_lengths:
  [batch_groups]
```

`valid_lengths` 决定每个 batch/head group 实际 attention 的 KV 长度。

最终 `weighted_flash_decoding` 输出当前 layer 的 attention result。

## 15. 预算变量

主要预算在 `prepare_cache()` 中计算：

```python
self.nprobe = max(round((self.n_centroids - 1) * self.retrieval_budget), 1)
self.nprobe = min(self.nprobe, self.n_centroids - 1)

requested_retrieval_tokens = max(int(self.retrieval_budget * self.list_capacity), 1)
self.max_retrieval_tokens = requested_retrieval_tokens
```

含义：

| 变量 | 含义 |
|---|---|
| `retrieval_budget` | 用户传入的比例，比如 `0.1` |
| `list_capacity` | middle KV 最大容量，不含 static start/end 和 generation 预算 |
| `n_centroids` | compact cluster 容量，最后一个是 padding cluster |
| `nprobe` | 每次 decode 最多选择多少个 cluster |
| `max_retrieval_tokens` | 每个 batch/head group 最多检索多少个 middle token |
| `buffer_size` | retrieval buffer page 数 |
| `page_size` | 当前固定为 8 |

`subspace_rerank` 还有两个中间预算：

```python
candidate_limit = max(4 * nprobe, 1024)  # 默认
rerank_limit = nprobe                    # 默认
```

但是最终进入 attention 的 cluster 数仍然是：

```text
<= nprobe
```

## 16. fallback 行为

在 `_select_clusters_for_decode()` 开头：

```python
metadata = self.hybrid_search_metadata_gpu[layer_idx]
required = (...)
if metadata is None or any(key not in metadata for key in required):
    if bool(self.decode_search_config.get("fallback_flat", True)):
        return self._select_clusters_flat(layer_idx, query_grouped, device)
    raise RuntimeError(...)
```

也就是说默认 `fallback_flat=True`。

如果 search metadata 缺失，`subspace` / `subspace_rerank` / `hybrid` 不会报错，而是自动退回 flat。

如果想强制发现 metadata 缺失，可以在 `decode_search` 中传：

```python
{"mode": "subspace", "fallback_flat": False}
```

当前命令行没有直接暴露 `fallback_flat` 参数，需要改 `call_api.py` 或直接改配置。

## 17. 当前各模式的真实步骤对比

### 17.1 `flat`

```text
query_grouped
-> full centroid dot product
-> mask invalid centroids
-> topk nprobe
-> cluster size / token budget 截断
-> selected
-> retrieval gather
-> weighted_flash_decoding
```

### 17.2 `subspace`

```text
query_grouped
-> sum group_size 得到 query
-> 根据 subspace_dim_ids 抽取 8 个 query_sub
-> query_sub 与 subspace_centroids 点积
-> 根据 compact_codes 汇总 coarse_scores
-> topk nprobe
-> cluster size / token budget 截断
-> selected
-> retrieval gather
-> weighted_flash_decoding
```

### 17.3 `subspace_rerank`

```text
query_grouped
-> sum group_size 得到 query
-> 根据 subspace_dim_ids 抽取 8 个 query_sub
-> query_sub 与 subspace_centroids 点积
-> 根据 compact_codes 汇总 coarse_scores
-> topk candidate_limit
-> gather candidate full centroids
-> query 与 candidate full centroids 点积
-> topk rerank_limit
-> 取前 nprobe
-> cluster size / token budget 截断
-> selected
-> retrieval gather
-> weighted_flash_decoding
```

### 17.4 `hybrid`

```text
当前同 subspace_rerank
```

## 18. 读日志时应该看什么

如果开了相关 profile/debug，重点看：

```text
n_centroids=...
nprobe=...
list_capacity=...
list_stride=...
retrieval_tokens=...
retrieval_pages=...
selected_clusters=...
valid_lengths[min,max,mean]=...
```

这些字段可以对应到：

| 日志字段 | 对应变量 |
|---|---|
| `n_centroids` | `self.n_centroids` |
| `nprobe` | `self.nprobe` |
| `list_capacity` | `self.list_capacity` |
| `list_stride` | `self.list_stride` |
| `retrieval_tokens` | `self.max_retrieval_tokens` |
| `selected_clusters` | `selected != padding_cluster_id` 的数量 |
| `valid_lengths` | 最终进入 attention 的 KV 长度 |

如果 `selected_clusters` 正常但 `valid_lengths` 异常，问题通常在 retrieval/gather/cache buffer。

如果 `selected_clusters` 很少或目标 token 所在 cluster rank 很差，问题通常在搜索分数或 metadata。

如果 `subspace_rerank` 比 `flat` 差很多，重点看：

```text
compact_codes 是否正确
subspace_dim_ids 是否和 partition_path 一致
subspace_centroids 是否来自同一次 build
candidate_limit 是否太小导致目标 cluster 没进入候选
```

如果 `subspace` 比 `subspace_rerank` 差很多，说明粗筛分数和 full centroid 分数之间有明显 gap，精搜确实在修正排序。

