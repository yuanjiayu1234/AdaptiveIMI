# Advanced Hybrid 8-Way IMI 实现方案

## 目标

本方案用于实现一个新的 AdaptiveIMI 进阶版本，不和当前 2-way/4-way 基础版本混在同一路径中。

核心目标：

- CPU 和 GPU 可以按 layer 混合构建索引，例如 1-based `layer1` 在 CPU，`layer2` 在 GPU，后续交替。
- 构建索引的调度单位是整层 KV，不再把半层或单 head 当成调度边界。
- CPU 构建路径：整层 middle KV 传输到 CPU 后，CPU 开始构建索引并重组 KV。
- GPU 构建路径：GPU 使用 CUDA 算子直接构建 index metadata；同时原始 KV 仍然开始传输到 CPU。GPU metadata 完成后传到 CPU，再由 CPU 用 CPU KV 和 metadata 进行簇重组。
- CPU/GPU 两条路径最终产出的 metadata 格式必须完全一致。
- 新版本固定面向 learned 8-way，可支持可变子空间维度；子空间不是顺序切分，而是读取训练得到的 JSON。
- 重组逻辑必须按 metadata 的实际 compact cluster 和 token labels 执行，不能沿用只适合 2-way/4-way 顺序切分的逻辑。

## 当前代码结论

当前主路径在这些文件中：

- Python cache 入口：`cache_hub/adaptive_imi/cache.py`
- Python indexing/runtime/retrieval：`cache_hub/adaptive_imi/indexing.py`、`runtime.py`、`retrieval.py`
- Python C++ adapter：`library/AdaptiveIMI/imi_adapter.py`
- C++ pipeline：`library/AdaptiveIMI/cpp_extensions/include/layer_pipeline.hpp`
- CPU kmeans：`library/AdaptiveIMI/cpp_extensions/src/kmeans_core.cpp`
- CPU 重组：`library/AdaptiveIMI/cpp_extensions/src/reorganize_core.cpp`
- pybind：`library/AdaptiveIMI/cpp_extensions/src/pipeline_bindings.cpp`
- decode similarity CUDA：`library/AdaptiveIMI/cpp_extensions/cuda/compute_impl.cu`
- learned partition 训练和聚合：`experiments/imi_partition/*.py`

已有能力：

- 2-way 和 4-way CPU kmeans 路径。
- learned M-way 的 CPU 雏形，当前 JSON 表达的是 RoPE pair 分组，每个 pair 对应维度 `pair` 和 `pair + head_dim/2`。
- 当前 metadata 对外格式是 `head_indices / cluster_counts / cluster_sizes / cluster_offsets / centroids`。
- 当前 decode 检索只依赖 compact 后的 full-dim centroids、cluster sizes、cluster offsets 和重组后的 CPU KV，因此只要新 metadata 保持一致，decode 侧可以最大程度复用。

当前不足：

- 没有 GPU 构建索引路径。
- 没有 CPU/GPU layer 级调度策略。
- 当前 `subspace_parts` 配置仍以 `0/2/4` 为主，learned M-way 是附加路径，不适合作为新 8-way 主路径。
- 当前 M-way CPU 重组会 compact active code，但没有把 token labels 作为统一 metadata 输出；GPU 路径需要这个字段才能让 CPU 重组复用同一套逻辑。
- 当前 pipeline 把 D2H、CPU kmeans、重组写在一个类里。新需求里 GPU 构建时 D2H 和 GPU index build 需要并行，并且 metadata 完成后仍要等 CPU KV 再重组，最好拆开。

## 新路径

新实现建议放在独立路径：

```text
cache_hub/adaptive_imi_v2/
  __init__.py
  cache.py
  config.py
  layer_scheduler.py
  partition.py
  metadata.py
  cpu_builder.py
  gpu_builder.py
  reorganize.py
  runtime.py
  retrieval.py

library/AdaptiveIMI/v2_extensions/
  setup.py
  include/
    hybrid_types.hpp
    hybrid_layer_pipeline.hpp
    gpu_8way_builder.cuh
    cpu_8way_builder.hpp
    reorganize_v2.hpp
  src/
    hybrid_bindings.cpp
    hybrid_layer_pipeline.cpp
    cpu_8way_builder.cpp
    reorganize_v2.cpp
  cuda/
    gpu_8way_builder.cu
```

旧路径只保留兼容：

```text
cache_hub/adaptive_imi/
library/AdaptiveIMI/cpp_extensions/
```

`model_hub/base.py` 中通过配置选择新旧实现：

```json
{
  "AdaptiveIMI": {
    "engine": "hybrid_8way_v2",
    "subspace_parts": 8,
    "learned_partition_path": "outputs/imi_partition/deploy_partition.json",
    "hybrid_index": {
      "policy": "alternate",
      "start": "cpu"
    }
  }
}
```

加载逻辑：

```python
if imi_config.get("engine") == "hybrid_8way_v2":
    from cache_hub.adaptive_imi_v2.cache import Hybrid8WayIMICache
else:
    from cache_hub.adaptive_imi.cache import adpimi_cache
```

## JSON 分区格式

沿用当前 `aggregate_partitions.py` 的输出风格，但新实现严格要求 `M=8`。

推荐 schema：

```json
{
  "metadata": {
    "format": "adaptive_imi_learned_partition_v2",
    "subspace_parts": 8,
    "dimension_unit": "rope_pair"
  },
  "layer_groups": [
    {
      "layer_start": 0,
      "layer_end": 32,
      "M": 8,
      "subspace_parts": 8,
      "k_values": [4, 4, 4, 4, 4, 4, 4, 4],
      "heads": {
        "0": {
          "groups": [[0, 5], [1, 7], [2, 9], [3, 11], [4, 13], [6, 15], [8, 14], [10, 12]]
        }
      }
    }
  ]
}
```

约定：

- `groups` 长度必须为 8。
- 每个 group 是 RoPE pair id 列表。
- 对 head_dim=128，pair id 范围是 `[0, 63]`。
- 一个 pair `p` 对应两个真实维度：`p` 和 `p + head_dim / 2`。
- 每个 head 的 8 个 groups 必须覆盖所有 pair 且不可重复。
- `k_values` 可以是 layer group 级共享，也可以扩展为 head 级。第一版建议 layer group 级共享，降低 GPU kernel 复杂度。

## 统一 Metadata Contract

CPU builder 和 GPU builder 都必须输出同一份逻辑 metadata。建议 C++ 内部统一结构：

```cpp
struct HybridHeadMetadata {
    int32_t batch_idx;
    int32_t local_head_idx;
    int32_t head_idx;
    int32_t token_count;
    int32_t cluster_count;

    // compact cluster id for every token, shape [T]
    std::vector<int32_t> token_labels;

    // CSR over compact clusters
    std::vector<int32_t> cluster_sizes;    // [cluster_count]
    std::vector<int32_t> cluster_offsets;  // [cluster_count + 1]

    // full head_dim centroids after compact Cartesian code merge
    // shape [cluster_count, head_dim], FP32
    torch::Tensor centroids_cpu;
};

struct HybridLayerMetadata {
    int32_t layer_idx;
    int32_t batch_size;
    int32_t kv_heads;
    int32_t token_count;
    int32_t head_dim;
    std::vector<HybridHeadMetadata> heads;
};
```

Python tensor 化后沿用 decode 需要的格式，并额外保留 `token_labels` 给 CPU 重组。底层 row 不再是单纯 `kv_heads`，而是：

```text
row_idx = batch_idx * kv_heads + local_head_idx
batch_groups = batch_size * kv_heads
```

```python
{
    "head_indices": int64[batch_groups],
    "batch_indices": int64[batch_groups],
    "local_head_indices": int64[batch_groups],
    "token_counts": int32[batch_groups],
    "cluster_counts": int32[batch_groups],
    "cluster_sizes": int32[batch_groups, max_clusters],
    "cluster_offsets": int32[batch_groups, max_clusters + 1],
    "centroids": float32[batch_groups, max_clusters, head_dim],
    "token_labels": int32[batch_groups, token_count],
}
```

`token_labels` 是新字段。旧 2-way/4-way 当前重组阶段会从 labels 自己生成 compact labels；新 GPU 路径必须把这个 compact label 传回 CPU，否则 CPU 只有 centroids 和 offsets，无法按簇重排原始 KV。

## Decode 子空间搜索 + 精搜 Metadata

如果 decode 阶段改成“子空间搜索 + 精搜”，metadata 不应该完全替换旧格式，而应该分成两层：

1. 基础 metadata：保持和当前 hot/cold/gather 路径兼容。
2. 搜索 metadata：新增给子空间 coarse search 和 rerank 使用。

基础 metadata 必须继续保留：

```python
{
    "head_indices": int64[batch_groups],
    "batch_indices": int64[batch_groups],
    "local_head_indices": int64[batch_groups],
    "token_counts": int32[batch_groups],
    "cluster_counts": int32[batch_groups],
    "cluster_sizes": int32[batch_groups, max_clusters],
    "cluster_offsets": int32[batch_groups, max_clusters + 1],
    "centroids": float32[batch_groups, max_clusters, head_dim],
    "token_labels": int32[batch_groups, token_count],
}
```

其中 `centroids` 是 compact cluster 的 full-dim centroid。它仍然有两个用途：

- 精搜 rerank：对子空间召回出的候选 compact cluster 做 full-dim 相似度重排。
- 兼容当前 flat centroid search：可以用作 fallback 或 ablation。

新增搜索 metadata：

```python
{
    "M": 8,
    "k_values": int32[8],
    "subspace_dims": int32[batch_groups, 8],
    "subspace_pair_ids": int64[batch_groups, 8, max_pairs_per_subspace],
    "subspace_pair_masks": bool[batch_groups, 8, max_pairs_per_subspace],
    "subspace_dim_ids": int64[batch_groups, 8, max_sub_dim],
    "subspace_dim_masks": bool[batch_groups, 8, max_sub_dim],

    # 每个 head/subspace 的 codebook centroid，按 max_k 和 max_sub_dim padding
    "subspace_centroids": float32[batch_groups, 8, max_k, max_sub_dim],
    "subspace_centroid_masks": bool[batch_groups, 8, max_k],

    # compact cluster -> 8 个子空间 centroid id
    "compact_codes": int32[batch_groups, max_clusters, 8],
}
```

这样 decode 可以走两阶段：

```text
query
  -> 按 JSON pair group 抽取 8 个 query subvector
  -> 每个 subspace 和 subspace_centroids 做 top-k coarse search
  -> 根据 compact_codes 合成候选 compact cluster 分数
  -> 候选 compact clusters 用 full-dim centroids 精搜 rerank
  -> 输出最终 selected compact cluster ids
  -> hot/cold batch_access + gather + attention
```

关键约束：

- 子空间搜索最终输出的必须是 compact cluster id，而不是 8-way 原始笛卡尔积 id。
- `compact_codes[head, compact_cluster]` 是从 compact id 回到 8 个 subspace label 的桥。
- `cluster_sizes/cluster_offsets` 仍然以 compact cluster id 为索引。
- 当前 `AdpIMI_Index.batch_access()` 只接收 compact cluster id，所以只要最后的 selected ids 语义不变，后面的 hot/cold 和 gather 不需要理解 8-way。

第一版 decode 建议实现两个模式：

```json
"decode_search": {
  "mode": "subspace_rerank",
  "subspace_topk": 8,
  "candidate_clusters": 256,
  "rerank_clusters": 64,
  "fallback_flat": true
}
```

`fallback_flat=true` 时，如果搜索 metadata 不存在或 GPU kernel 不可用，就退回当前 full-dim centroid top-k。

## CPU/GPU 数据职责

新版本必须明确哪些数据常驻 CPU，哪些常驻 GPU，避免把 KV 或 metadata 放错位置。

### CPU 常驻

CPU 必须保存：

- `list_keys/list_values`：重组后的 middle KV，shape `[batch_size, kv_heads, T, head_dim]`，pinned CPU。
- `cluster_sizes_cpu/cluster_offsets_cpu`：hot/cold 描述 CPU list 中每个 compact cluster 的位置和长度。
- `token_counts`：每个 `(batch, kv_head)` row 的真实 middle 长度，padding 不参与建索引。
- `token_labels_cpu`：构建和调试用；CPU 重组完成后 decode 不一定需要，但建议保留到本 sequence 结束。
- `compact_codes_cpu` 和搜索 metadata 的 CPU 副本：用于 cache reset、debug、fallback、重新上传 GPU。
- 每层构建 stats。

CPU 不应该保存：

- decode 每步的 query subspace 临时结果。
- GPU coarse search 的临时候选矩阵。
- GPU cache block 内容。

### GPU 常驻

GPU 必须保存：

- `centroids_gpu`：full-dim compact centroids，用于精搜 rerank 或 fallback flat search。
- `centroids_mask_gpu`：padding/empty compact cluster mask。
- `cluster_sizes_gpu`：当前 `_select_clusters` 和预算截断需要用。
- `subspace_centroids_gpu`：子空间 codebook centroids。
- `compact_codes_gpu`：compact cluster 到 8-way code 的映射。
- `subspace_pair_ids_gpu/subspace_pair_masks_gpu`：query subvector gather 所需。
- hot/cold 的 `cache_keys/cache_values`：GPU block cache。
- `execution_buffer_keys/values`：每步 attention 执行 buffer。
- search 临时 buffer：subspace scores、candidate ids、rerank scores、selected compact ids。

GPU 不应该保存：

- 完整 middle KV 的主存副本。原始 middle KV 在 GPU builder 使用完后释放，长期 KV 主副本仍在 CPU `list_keys/list_values`。
- `token_labels` 的长期副本。GPU 构建时需要，构建完成和 metadata D2H 后可以释放；decode 不需要 token -> cluster labels。

### CPU/GPU 构建路径的数据流

CPU layer：

```text
GPU full middle KV
  -> D2H to CPU raw KV
  -> CPU 8-way builder produces base metadata + search metadata
  -> CPU reorganize raw KV into list_keys/list_values
  -> upload decode metadata to GPU
```

GPU layer：

```text
GPU full middle KV
  -> async D2H to CPU raw KV
  -> GPU 8-way builder produces base metadata + search metadata on GPU
  -> metadata D2H to CPU
  -> wait raw KV D2H
  -> CPU reorganize raw KV into list_keys/list_values using token_labels
  -> keep/upload decode metadata on GPU
```

GPU layer 的优化点：如果 GPU builder 已经生成 `centroids_gpu/subspace_centroids_gpu/compact_codes_gpu`，不要先 D2H 再 H2D；可以直接把 GPU metadata attach 到 cache，同时只把一份 CPU metadata 传回用于重组和 fallback。

## Layer 级调度

调度器独立实现：

```python
class LayerBuildScheduler:
    def __init__(self, policy: str, start: str, explicit: dict[int, str] | None):
        ...

    def build_device(self, layer_idx: int) -> Literal["cpu", "gpu"]:
        ...
```

策略：

- `alternate`：交替 CPU/GPU。
- `explicit`：配置中指定每一层。
- `cpu_only`：调试用。
- `gpu_only`：调试 GPU builder 和 metadata。

用户例子中 `layer1 CPU, layer2 GPU` 是 1-based 说法，实现中对应：

```python
if start == "cpu":
    return "cpu" if layer_idx % 2 == 0 else "gpu"
```

配置示例：

```json
"hybrid_index": {
  "policy": "alternate",
  "start": "cpu",
  "gpu_streams": {
    "d2h": true,
    "build": true,
    "metadata_d2h": true
  }
}
```

## 整层构建流程

### Prefill 后每层提交

每层 prefill 结束时，`Hybrid8WayIMICache.prefill_update_kv_cache()` 拿到该层完整 middle KV：

```python
middle_keys = key_states[:, middle_start:middle_end, :, :].transpose(1, 2).contiguous()
middle_values = value_states[:, middle_start:middle_end, :, :].transpose(1, 2).contiguous()
builder = self.layer_builders[layer_idx]
builder.submit_layer(middle_keys, middle_values)
```

输入 shape 统一为：

```text
[1, kv_heads, token_count, head_dim]
```

### CPU layer

CPU layer 的顺序是：

```text
GPU KV -> CPU pinned KV
等待整层 KV D2H 完成
CPU 8-way kmeans
CPU compact metadata
CPU 根据 token_labels 重组 KV
metadata 填入 cache
```

伪代码：

```python
def submit_cpu_layer(layer_idx, keys_gpu, values_gpu):
    d2h_job = copy_full_layer_to_cpu(keys_gpu, values_gpu)
    d2h_job.wait()
    metadata = cpu_8way_build(keys_cpu, partition_spec)
    list_keys, list_values = reorganize_cpu(keys_cpu, values_cpu, metadata)
    return LayerBuildResult(metadata, list_keys, list_values)
```

### GPU layer

GPU layer 的顺序是：

```text
GPU KV -> CPU pinned KV 异步开始
GPU 8-way builder 同时开始
GPU builder 输出 labels / compact map / centroids / offsets
metadata D2H
等待 CPU KV D2H 完成
CPU 根据 GPU metadata 的 token_labels 重组 KV
metadata 填入 cache
```

伪代码：

```python
def submit_gpu_layer(layer_idx, keys_gpu, values_gpu):
    d2h_job = copy_full_layer_to_cpu_async(keys_gpu, values_gpu)
    gpu_meta_job = gpu_8way_build_async(keys_gpu, partition_spec)
    metadata = gpu_meta_job.metadata_to_cpu()
    d2h_job.wait()
    list_keys, list_values = reorganize_cpu(keys_cpu, values_cpu, metadata)
    return LayerBuildResult(metadata, list_keys, list_values)
```

关键点：

- GPU build 不等待 KV D2H。
- CPU 重组必须等待 CPU KV 和 metadata 都 ready。
- metadata 体积远小于 KV，优先传回 CPU。
- CPU/GPU metadata 的 compact cluster id 语义必须相同：`token_labels[token]` 指向 compact 后 `[0, cluster_count)`。

## 8-Way CPU Builder

CPU 端可以借鉴现有 `KmeansExecutorM`，但建议从旧 pipeline 中拆成独立 builder。

输入：

```text
keys_cpu: [batch_size, kv_heads, T, head_dim], FP16/BF16 CPU
partition: per-head 8 groups of RoPE pairs
k_values: [8]
```

步骤：

1. 将 keys 转 FP32。
2. 对每个 batch、每个 head、每个 subspace 抽取子向量。
3. 对 8 个子空间分别跑 kmeans。
4. 对每个 token 收集 8 个 label，形成 code：

```text
code(token) = [label_0, label_1, ..., label_7]
```

5. 不分配完整笛卡尔积数组。用 hash map 只 compact 实际出现过的 code：

```cpp
unordered_map<Code8, int32_t> code_to_compact;
token_labels[token] = code_to_compact.emplace_or_get(code);
```

6. 根据 `token_labels` 统计 `cluster_sizes` 和 `cluster_offsets`。
7. 根据每个 compact code 合成 full-dim centroid。

不能使用完整 `k0*k1*...*k7` 的 dense array。8-way 即使每个 k 很小，笛卡尔积也会快速爆炸；必须只 compact active code。

## 8-Way GPU Builder

GPU builder 第一版建议分成 4 个 CUDA 阶段，不把所有逻辑塞进一个超大 kernel。GPU 构建必须依赖自定义 `.cu` 扩展，不能把 PyTorch CUDA tensor ops 当成最终实现。当前 v2 路径要求编译 `hybrid_8way_builder_cpp`；扩展缺失时 GPU build 直接报错。

当前 CUDA extension 输出完整 GPU build metadata：

```text
labels_m: int32[batch_groups, 8, max_middle_len]
token_labels: int32[batch_groups, max_middle_len]
compact_codes: int32[batch_groups, max_middle_len, 8]
cluster_sizes: int32[batch_groups, max_middle_len]
cluster_offsets: int32[batch_groups, max_middle_len + 1]
cluster_counts: int32[batch_groups]
centroids: float32[batch_groups, max_middle_len, head_dim]
subspace_centroids: float32[batch_groups, 8, max_k, max_sub_dim]
```

Compact active code 在 `.cu` 内完成：8-way labels 被 pack 成 `CodeKey`，按 row 用 Thrust `sort_by_key` 排序，再通过 scan/scatter 生成 compact labels、sizes、offsets 和 compact codebook。

Ragged batch 采用矩形 padded storage：

```text
keys_gpu: [batch_size, kv_heads, max_middle_len, D]
token_counts: [batch_groups]
```

其中 `token_counts[row]` 之外的 padding token 不参与 assignment、cluster size、offset、重组和 decode。

### 阶段 1：Subspace gather

输入 `keys_gpu [batch_size, kv_heads, T, D]` 和 pair groups，输出 8 个子空间 view/buffer：

```text
subvecs[batch][head][m]: [T, sub_dim_m]
```

实现建议：

- 第一版为每个 `(head, m)` 分配连续 GPU buffer，方便后续 kmeans。
- pair group 很小，gather kernel 读 `pair` 和 `pair + D/2` 两个位置。
- 如果 kmeans kernel 能支持 indirect dim index，后续可省掉 materialize buffer。

### 阶段 2：GPU kmeans per subspace

需要一个新的 CUDA kmeans 实现：

```cpp
GpuKmeansResult run_gpu_kmeans_8way(
    const half_or_bf16* keys,
    int batch_size,
    int kv_heads,
    int T,
    int D,
    PartitionSpec partition,
    int k_values[8],
    cudaStream_t stream
);
```

第一版可用 Lloyd kmeans：

- 初始化：先实现 deterministic init，例如均匀采样 token；后续再加 kmeans++。
- assignment kernel：每个 `(batch, head, subspace, token)` 找最近 centroid。
- reduction kernel：按 label 聚合 sum/count。
- update kernel：生成新 centroid。
- 迭代次数使用配置，例如 `max_iters=15/20`。

输出：

```text
labels_m: int32[batch_groups, 8, T]
centroids_m: float32[batch_groups, 8, k_m, sub_dim_m]
```

### 阶段 3：Compact active code

GPU 端根据 8 个 label 生成 compact labels。

第一版推荐做法：

1. 生成 code key：

```cpp
struct Code8 {
    int32_t c[8];
};
```

2. 对每个 `(batch, head)` row 的 `Code8[T]` 排序。
3. unique 后得到 `compact_codes`。
4. 生成 `token_labels`。
5. 统计 cluster sizes 和 offsets。

实现可以使用 Thrust/CUB，避免手写复杂 hash table：

```cpp
thrust::sort_by_key(code_keys, token_ids)
thrust::reduce_by_key(code_keys, ones, unique_codes, cluster_sizes)
scatter compact id back to token_labels[token_id]
exclusive_scan(cluster_sizes, cluster_offsets)
```

注意：

- `Code8` 排序 key 可以先编码成 64-bit/128-bit。若每个 `k <= 255`，可 pack 到 uint64；更通用的实现用两个 uint64 或自定义 comparator。
- 第一版可以限制 `k_values[m] <= 65535`，使用 128-bit equivalent `uint2/ulonglong2`。

### 阶段 4：合成 full-dim centroids

对每个 compact code，根据 8 个子空间 centroid 和 JSON pair group 填回 full dim：

```text
centroids_full[row, compact, pair] = centroids_m[row, m, code[m], offset]
centroids_full[row, compact, pair + D/2] = centroids_m[row, m, code[m], offset + 1]
```

输出 GPU metadata：

```text
token_labels_gpu: int32[batch_groups, T]
cluster_sizes_gpu: int32[batch_groups, max_clusters]
cluster_offsets_gpu: int32[batch_groups, max_clusters + 1]
centroids_gpu: float32[batch_groups, max_clusters, D]
cluster_counts_gpu: int32[batch_groups]
```

然后 metadata D2H。

## CPU 重组

重组必须成为独立函数，不依赖 kmeans 的内部 labels1/labels2/labels4。

统一接口：

```cpp
void reorganize_from_metadata(
    const torch::Tensor& keys_cpu,       // [batch_size, kv_heads, T, D]
    const torch::Tensor& values_cpu,     // [batch_size, kv_heads, T, D]
    const torch::Tensor& token_labels,   // [batch_groups, T] int32 CPU
    const torch::Tensor& cluster_offsets,// [batch_groups, max_clusters + 1] int32 CPU
    const torch::Tensor& cluster_counts, // [batch_groups] int32 CPU
    std::vector<CPUBufferDescriptor>* output_buffers,
    std::vector<ReorganizeResult>& out
);
```

算法：

```text
for each row = batch_idx * kv_heads + head_idx:
  write_positions = cluster_offsets[head][0:cluster_count]
  for token in [0, T):
      cid = token_labels[head][token]
      dst = write_positions[cid]++
      reorder_indices[dst] = token
  parallel for dst in [0, T):
      src = reorder_indices[dst]
      copy key[src] and value[src] to output[dst]
```

这个函数同时服务 CPU builder 和 GPU builder。CPU builder 也应该产出 `token_labels`，然后走同一套重组，避免两套重组逻辑分叉。

## Python Cache 生命周期

`Hybrid8WayIMICache` 的关键状态：

```python
self.layer_build_policy: list[str]            # "cpu" or "gpu"
self.layer_jobs: list[LayerBuildJob | None]
self.layer_metadata: list[LayerMetadata | None]
self.list_keys / self.list_values             # CPU pinned, reorganized KV
self.cluster_sizes_cpu / cluster_offsets_cpu
self.centroids_cpu / centroids_gpu / mask_gpu
```

每层 prefill 结束：

```python
def prefill_update_kv_cache(...):
    middle_keys, middle_values = extract_middle(...)
    self._ensure_list_storage(layer_idx)
    mode = self.scheduler.build_device(layer_idx)
    self.layer_jobs[layer_idx] = self.hybrid_pipeline.submit(
        layer_idx=layer_idx,
        mode=mode,
        keys=middle_keys,
        values=middle_values,
        list_keys=self.list_keys[layer_idx],
        list_values=self.list_values[layer_idx],
    )
    write_static_prefix_suffix(...)
```

decode 前或 `prepare_cache()` 中：

```python
def ensure_layer_ready(layer_idx):
    job = self.layer_jobs[layer_idx]
    result = job.wait()
    self._load_layer_metadata(layer_idx, result.metadata)
    self.adpimi_index[layer_idx].set_kv(...)
```

`_load_layer_metadata()` 可借鉴当前实现，但输入改成统一 tensor metadata。

## C++/CUDA pybind API

建议新扩展只暴露一个主类：

```python
pipeline = hybrid_8way_cpp.HybridLayerPipeline(
    layer_idx=ldx,
    kv_heads=kv_heads,
    max_tokens=list_capacity,
    dim=head_dim,
    device_id=device_id,
    kv_dtype=dtype,
    partition_spec=partition_spec,
    runtime_config=runtime_config,
)

job = pipeline.submit_layer(keys, values, build_device="gpu")
job.wait_ready()
metadata = job.get_metadata_result()
stats = job.get_stats()
```

C++ 内部可以返回 job id，也可以每个 layer pipeline 一次只处理一个 job。第一版为了降低复杂度，每层一个 pipeline、一次一个 job 就够。

metadata pybind 返回：

```python
{
    "head_indices": torch.Tensor,
    "cluster_counts": torch.Tensor,
    "cluster_sizes": torch.Tensor,
    "cluster_offsets": torch.Tensor,
    "centroids": torch.Tensor,
    "token_labels": torch.Tensor,
    "compact_codes": torch.Tensor,
    "subspace_centroids": torch.Tensor,
    "subspace_pair_ids": torch.Tensor,
    "subspace_pair_masks": torch.Tensor,
    "build_device": "cpu" | "gpu"
}
```

对于 GPU layer，pybind 可以额外返回 GPU tensors：

```python
{
    "gpu": {
        "centroids": torch.Tensor,
        "subspace_centroids": torch.Tensor,
        "compact_codes": torch.Tensor,
        "subspace_pair_ids": torch.Tensor,
        "subspace_pair_masks": torch.Tensor,
    }
}
```

CPU layer 没有这些 GPU tensors 时，由 Python cache 在 `_load_layer_metadata()` 里上传。

## Hot/Cold 缓存替换兼容性

可以兼容现有 hot/cold 缓存替换，但有一个硬性条件：decode search 的最终输出必须是 compact cluster id。

当前 `AdpIMI_Index` 的接口和替换逻辑只依赖：

- `set_cluster_metadata(cluster_sizes, cluster_offsets, start_bdx)`
- `set_kv(list_keys, list_values, ...)`
- `searched_clusters`: `[batch_groups, nprobe]` compact cluster ids
- `batch_access()` 根据 compact cluster id 判断 hit/miss
- `gather_copy_and_concat*()` 根据 hit/miss block 描述从 CPU list 和 GPU cache 拼执行 buffer

它不依赖：

- cluster 是 2-way、4-way 还是 8-way。
- cluster id 是怎么搜索出来的。
- 是否使用子空间 coarse search。
- 是否使用 full-dim centroid rerank。

因此 v2 不需要重写 hot/cold，应该继续复用它。需要注意：

- `cluster_offsets[c]` 必须是 `list_keys/list_values` 中 compact cluster `c` 的起始 token。
- `cluster_sizes[c]` 必须是 compact cluster `c` 的 token 数。
- `selected_clusters` 必须使用 compact id，padding 使用 `padding_cluster_id`。
- `n_centroids` 仍然是全局 padded 上限，等于 `max_cluster_capacity + 1`，最后一个是 padding cluster。
- 子空间搜索候选数可以大于最终 `nprobe`，但交给 hot/cold 的只能是 rerank 后的最终 `nprobe` 个 compact id。

如果之后想让 hot/cold 更智能，可以扩展替换权重，例如用 rerank score 或 cluster size 做 admission，但这不是第一版必需项。第一版保持现有 hot/cold 行为，风险最低。

## 需要修改的入口

新路径实现完成后，只改最小入口：

1. `model_hub/base.py`
   - 根据 `AdaptiveIMI.engine` 选择 `Hybrid8WayIMICache`。

2. 配置文件
   - 增加 `engine: hybrid_8way_v2`。
   - 增加 `subspace_parts: 8`。
   - 增加 `learned_partition_path`。
   - 增加 `hybrid_index`。

3. `library/AdaptiveIMI/v2_extensions/setup.py`
   - 独立编译 v2 extension。
   - 不改旧 `cpp_extensions/setup.py`。

## 验证计划

### 单元验证

- JSON parser：
  - M 必须是 8。
  - 每个 head 覆盖所有 RoPE pairs。
  - pair 不重复、不越界。

- CPU builder：
  - 小 T、小 D 下检查 `token_labels` 范围。
  - `sum(cluster_sizes) == T`。
  - `cluster_offsets[-1] == T`。
  - `centroids.shape == [cluster_count, D]`。

- GPU builder：
  - 同样检查 metadata invariant。
  - 固定 seed 和小规模数据下，对比 CPU/GPU cluster_count、offset、sizes 的基本合法性。GPU kmeans 和 CPU kmeans 不必 bitwise 相同。

- 重组：
  - 对每个 cluster，重组后 `[offset[c], offset[c+1])` 里的 token 都来自 `token_labels == c`。
  - key/value 的 token 对齐不变。

### 集成验证

1. `engine=hybrid_8way_v2, policy=cpu_only` 跑通 prefill + 1 step decode。
2. `policy=gpu_only` 跑通 prefill + 1 step decode。
3. `policy=alternate,start=cpu` 跑通完整 decode。
4. 打开 profiling：
   - CPU layer 看到 D2H 后才 kmeans。
   - GPU layer 看到 D2H 和 GPU build 有重叠。
   - CPU/GPU layer metadata load 后 decode 使用同一 sparse path。

### 性能指标

需要记录：

```text
layer_idx
build_device
d2h_ms
gpu_build_ms
metadata_d2h_ms
cpu_kmeans_ms
cpu_reorganize_ms
total_ready_ms
cluster_count_mean
cluster_count_max
```

## 实现顺序

推荐按以下顺序做，避免一次性改太多：

1. 新建 `adaptive_imi_v2` 和 `v2_extensions` 空框架，先能 import。
2. 搬出 learned 8-way JSON parser，做严格校验。
3. 实现 `reorganize_from_metadata`，先用 Python 或 C++ CPU 小规模测试。
4. 实现 CPU 8-way builder，输出统一 metadata 和 token_labels。
5. 用 `cpu_only` 跑通新 cache 的 prefill、prepare_cache、decode。
6. 实现 GPU builder 的 subspace gather 和 kmeans。
7. 实现 GPU compact active code 和 full-dim centroid 合成。
8. 接入 GPU metadata D2H 和 CPU 重组。
9. 接入 `alternate` 调度。
10. 做 profiling 和精度/召回测试。

## 关键风险

- 8-way 完整笛卡尔积不能 dense 展开，必须 active-code compact。
- GPU kmeans 初版如果初始化策略和 CPU 不同，召回可能不同，这是可接受的；但 metadata invariant 必须完全一致。
- GPU builder 生成的是 compact label，不是原始笛卡尔积 id。CPU 重组只认 compact label。
- `n_centroids` 的全局分配仍需要按所有 layer 的最大 active cluster 上限预留。v2 中建议按 `min(T, product(k_values)) + 1` 估计每层上限，再取全局最大。
- learned partition 当前以 RoPE pair 为单位。若未来 JSON 直接给真实 dim id，需要在 `dimension_unit` 中区分，不要混用。
