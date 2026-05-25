 你的判断是对的：工程上不要把粒度做得太细。最佳折中是：

  调度单位：layer
  构建 backend：每层选择 CPU 或 GPU
  划分模板：每个 layer group 一个 M
  head 级别：允许每个 head 有自己的 pair partition，但 M 固定
  metadata：继续沿用当前 compact cluster 格式
  decode：暂时不变

  也就是说：

  不要做：每个 head 一个不同 M、不同 K_i、不同 metadata shape
  可以做：同一层/同一 layer group 固定 M，但每个 head 的 64 个 pair 怎么分可以不同

  现有 Metadata 格式
  当前 C++ reorg 输出的核心格式是：

  ReorganizeResult:
    reorganized_keys    [total_tokens, head_dim]
    reorganized_values  [total_tokens, head_dim]
    cluster_offsets     [n_active_clusters + 1]
    cluster_sizes       [n_active_clusters]

  定义在 library/AdaptiveIMI/cpp_extensions/include/layer_pipeline.hpp:92。

  对 Python 暴露时，会把每个 head 的结果 pad 到同一层的 max_clusters：

  cluster_sizes   [kv_heads, max_clusters]
  cluster_offsets [kv_heads, max_clusters + 1]
  centroids       [kv_heads, max_clusters, head_dim]

  见 library/AdaptiveIMI/cpp_extensions/src/pipeline_bindings.cpp:228。

  Python 侧再拷贝到全局固定大小 buffer：

  cluster_sizes_cpu[layer]  [batch_groups, n_centroids]
  cluster_offsets_cpu[layer][batch_groups, n_centroids + 1]
  centroids_cpu[layer]      [batch_groups, n_centroids, head_dim]
  centroids_mask_cpu[layer] [batch_groups, n_centroids]

  见 cache_hub/adaptive_imi/indexing.py:112。

  所以当前 runtime 有一个很强的隐含约束：

  每层/每个 cache 实例有统一的 n_centroids 上限
  每个 head 可以有不同 active cluster count
  但最终都要 pad 到同一个 n_centroids

  并且底层 AdpIMI_Index::set_cluster_metadata() 也假设传入的 n_centroids 固定，后续不能变：

  AT_ASSERT(last_n_centroids == n_centroids, "n_centroids not equal in one batch.")

  见 library/AdaptiveIMI/cpp_extensions/src/wave_buffer_cpu_hotcold.cpp:861。

  重组逻辑
  当前 reorg 做了三件事：

  1. 把每个 token 的多子空间 label 合成一个 virtual cell id。
  2. 只保留实际出现过的 cell，建立 global_to_compact。
  3. 按 compact cluster id 重排 KV，并生成 full-dim merged centroids。

  2-way 的合并逻辑是：

  combined[i] = labels1[i] * k2 + labels2[i]

  见 library/AdaptiveIMI/cpp_extensions/src/reorganize_core.cpp:32。

  4-way 是：

  combined = (((l1 * k2 + l2) * k3 + l3) * k4 + l4)

  见 library/AdaptiveIMI/cpp_extensions/src/reorganize_core.cpp:112。

  然后生成：

  final_labels[token] -> compact_cluster_id
  cluster_offsets
  cluster_sizes
  merged_centroids [n_active, head_dim]

  真正重排 KV 的地方只关心：

  labels
  cluster_offsets
  n_clusters
  keys_head_ptr
  values_head_ptr

  见 library/AdaptiveIMI/cpp_extensions/src/reorganize_core.cpp:378。

  这点很关键：reorg 不需要知道 M，也不需要知道 pair partition，只需要 final_labels 和 full-dim merged centroids。

  最佳实现方式
  我建议把未来实现分成两层。

  第一层：索引构建内部可以复杂，但输出统一：

  struct BuiltHeadIndex {
      std::vector<int32_t> final_labels;      // [T]
      std::vector<int32_t> cluster_sizes;     // [n_active]
      std::vector<int32_t> cluster_offsets;   // [n_active + 1]
      torch::Tensor merged_centroids;         // [n_active, head_dim]
  }

  不管 CPU 构建还是 GPU 构建，最后都返回这个格式。

  第二层：runtime 继续只吃当前 metadata：

  cluster_sizes
  cluster_offsets
  centroids full head_dim
  reorganized KV

  这样 decode 完全不用知道你是 IVF、2-way、4-way、8-way、非均匀 partition，还是 GPU 构建出来的。

  M 怎么限制
  为了避免工程复杂，我建议：

  M 不做 per-head 自由变化
  M 固定在 layer group 级别

  例如：

  early:  M = 4
  middle: M = 8
  late:   M = 6 或 8

  但每个 head 可以有自己的 partition：

  layer_group = middle, M = 8
  head0: G_0...G_7
  head1: G_0...G_7
  ...
  head7: G_0...G_7

  这样好处是：

  调度简单
  metadata shape 稳定
  KMeans task 数稳定
  GPU kernel launch pattern 稳定
  CPU reorg 不需要处理异构 M
  后续如果实现 GPU 构建，也可以 batch by layer

  实验脚本可以继续输出每个 head 的最优 M_h，但工程落地时做一次聚合：

  每个 layer group 选一个 M_group
  M_group = 满足大多数 head 的最大安全 M

  比如 8 个 head 的实验结果是：

  [8, 8, 6, 8, 10, 8, 6, 8]

  工程上不要真的实现 {6,8,10} 混杂，直接选：

  M_group = 8

  或者保守选：

  M_group = 6

  看 recall 和构建时间权衡。

  CPU/GPU 分层交替构建
  这个想法适合当前结构。建议做成：

  layer_idx % 2 == 0 -> CPU builder
  layer_idx % 2 == 1 -> GPU builder

  但注意，调度单位必须是 layer，不要拆到 head。因为现有 LayerPipeline 的 callback、direct write、metadata register 都是
  layer 级批量处理 8 heads：library/AdaptiveIMI/cpp_extensions/src/layer_pipeline.cpp:354。

  建议抽象成：

  IndexBuildBackend:
    start(layer_idx, keys, values)
    finish() -> List[head_metadata]

  CPU backend 用现有 IMIPipeline。

  GPU backend 即使在 GPU 上完成 KMeans / label assignment，最后也应该返回同样的：

  head_idx
  cluster_sizes
  cluster_offsets
  centroids [n_active, head_dim]

  并且把 list_keys/list_values 写到当前 CPU pinned layout：

  [batch, kv_head, list_capacity, head_dim]

  因为当前检索体系仍然依赖 CPU list + hot/cold cache。

  8-way / 非均匀 reorg 怎么做
  不要继续用 dense used(product(K_i))。8-way 后 virtual cell 空间很容易大，不能分配 product(K_i) 长度数组。

  改成 observed-only compact map：

  for token i:
      code = mixed_radix(labels_m[i], K_m)
      if code not in map:
          map[code] = next_compact_id++
      final_labels[i] = map[code]

  然后：

  counts[compact_id]++
  cluster_offsets = prefix_sum(counts)

  这比当前 used(n_clusters) 更适合 8-way 和 variable-rate。

  如果 code 可能超过 int64，就不要编码成单个整数，直接用 fixed tuple key：

  std::array<int32_t, MAX_M>

  但工程上如果 M <= 8 且 product(K_i) 控制在合理范围，uint64_t mixed_radix 足够。

  最终建议
  最稳的实现路线：

  1. 离线实验继续探索 per-head partition 和候选 M。
  2. 工程配置只采用 layer_group -> M，不采用 per-head 不同 M。
  3. 同一个 layer 内所有 head 使用相同 M 和相同 K_m 模板，但 partition 可以 per-head 不同。
  4. 新增 generic build_compact_reorg_input_m()，输出当前完全相同的 final_labels / cluster_offsets / merged_centroids。
  5. CPU/GPU builder 都必须收敛到同一个 metadata contract。
  6. 现有 decode 暂时不变，仍然对 full-dim compact centroids 做 top-k。

  一句话：让复杂性只存在于离线 partition 和构建阶段内部，不要泄漏到 metadata、CPU reorg、cache manager、decode hot path。