# hybrid_8way_v2 Decode 搜索模式说明

本文档只描述 `AdaptiveIMI` 的 `hybrid_8way_v2` 路径下，decode 阶段如何选择 IMI cluster。

当前只保留两个可见模式：

- `None` 或 `flat`：走完整 full-dim centroid 相似度搜索。
- `subspace_rerank`：先用 8 路 learned subspace 粗筛 candidate clusters，再用 full-dim centroid 精搜重排。

`subspace` 粗筛后直接进入 attention 的 Python/Torch 路径已经移除。`hybrid` 别名也已经移除；如果配置中继续传 `subspace` 或 `hybrid`，`Hybrid8WayIMICache` 会在初始化阶段报错。

## 配置入口

RULER 参数入口：

```python
parser.add_argument("--decode_search_mode", type=str, default=None, choices=[None, "flat", "subspace_rerank"])
parser.add_argument("--candidate_clusters", type=int, default=None)
parser.add_argument("--rerank_clusters", type=int, default=None)
```

LongBench v2 参数入口：

```python
parser.add_argument("--decode_search_mode", type=str, default=None, choices=("flat", "subspace_rerank"))
parser.add_argument("--candidate_clusters", type=int, default=None)
parser.add_argument("--rerank_clusters", type=int, default=None)
```

`config/config.py::generate_config()` 会把 `decode_search` 写入 `AdaptiveIMI` 配置段；`model_hub/base.py::_build_kv_cache()` 再传给 `Hybrid8WayIMICache`。

## Decode 搜索模式

| 模式 | `_decode_search_enabled()` | selector | 搜索方式 | 需要 search metadata |
|---|---:|---|---|---:|
| `None` | 否 | `_select_clusters_flat()` | full-dim centroid flat search | 否 |
| `flat` | 否 | `_select_clusters_flat()` | full-dim centroid flat search | 否 |
| `subspace_rerank` | 是 | `_select_clusters_for_decode()` | subspace coarse search + full centroid rerank | 是 |

`Hybrid8WayIMICache` 初始化时会校验：

```python
decode_search_mode = str(self.decode_search_config.get("mode", "flat")).lower()
if decode_search_mode not in ("flat", "subspace_rerank"):
    raise ValueError(...)
```

因此旧的 `subspace` 和 `hybrid` 配置不会静默退回其他路径。

## subspace_rerank 执行顺序

prefill/build 阶段必须已经生成 decode search metadata：

```python
required = (
    "subspace_centroids",
    "subspace_centroid_masks",
    "subspace_dim_ids",
    "subspace_dim_masks",
    "compact_codes",
)
```

decode 时执行：

```text
query_grouped
-> fused_subspace_rerank_select
   -> 8-way subspace coarse scoring
   -> top-k candidate_clusters
   -> gather candidate full-dim centroids
   -> full centroid rerank
   -> 按 nprobe 和 max_retrieval_tokens 截断
-> selected compact cluster ids
-> hot/cold batch_access
-> gather selected KV 到 execution buffer
-> weighted_flash_decoding
```

默认 candidate 预算：

```python
candidate_limit = max(nprobe * 4, 1024)
candidate_limit = clamp(candidate_limit, min=nprobe, max=active_clusters)
```

默认 rerank 预算：

```python
rerank_limit = nprobe
rerank_limit = clamp(rerank_limit, min=nprobe, max=candidate_limit)
```

最终交给 retrieval 的 cluster 数仍然最多是 `nprobe`。

## fallback 行为

当前代码没有实现文档旧版本里描述的 `fallback_flat=True` 默认回退。实际行为是：

```python
if metadata is None or any(key not in metadata for key in required):
    raise RuntimeError(f"hybrid decode search metadata missing on layer {layer_idx}")
```

这意味着 `subspace_rerank` 依赖 search metadata 完整存在；metadata 缺失会让 decode 直接失败，而不是自动退回 `flat`。

影响：

- 好处：能尽早暴露 GPU/CPU build metadata 生成、上传或保存链路的问题，不会把一次 `subspace_rerank` 实验悄悄变成 `flat` 实验。
- 风险：如果某层 build 成功但 search metadata 没被保留或上传，整次生成会中断；旧文档预期下用户可能以为会继续生成。
- 实验解释：开启 `DECODE_SEARCH_MODE=subspace_rerank` 后，结果可以认为确实使用了 rerank 路径；不会混入部分 layer 回退 flat 的情况。

如果需要容错，可以后续显式实现 `fallback_flat` 配置，但当前版本保持 fail-fast。
