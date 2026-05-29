
• 我读完了 hybrid_8way_v2 的主要实现和实验脚本。核心结论是：你的论文动机应该从“低 k 的近似 top-k 检索”转成“固定 token 预算比例下的高
  召回 sparse attention”。也就是目标不是 recall@32/64 很高，而是在 2.5% 或 5% 上下文预算内，尽量覆盖 full attention 中真正高相关的一
  批 token。

  hybrid_8way_v2 当前算法可以概括为三段：

  1. Prefill 建索引
     每层每个 KV head 使用 learned partition，把 RoPE pair 维度划成 M=8 个子空间。每个子空间独立 k-means，得到 8 个 label；8 个
     label 被打包成一个 compact code，对应一个 active IMI cell/cluster。实现入口主要在 cache_hub/adaptive_imi_v2/gpu_builder.py:260
     和 partition 校验在 cache_hub/adaptive_imi_v2/partition.py:15。
  2. Decode 两阶段选 cluster
     subspace_rerank 先用 8 个子空间 centroid 的得分求和做粗筛，取 candidate_clusters；再 gather 这些候选的 full-dim centroid，用完
     整 q·centroid 重排，取 rerank_clusters/nprobe。最终再按 max_retrieval_tokens 截断 cluster。对应实现是 cache_hub/
     adaptive_imi_v2/cache.py:453 和 CUDA wrapper library/AdaptiveIMI/cpp_extensions/src/compute_wrappers.cpp:187。
  3. 实际 token 预算
     retrieval_budget 同时控制两件事：
     nprobe ≈ active_clusters * retrieval_budget，以及 max_retrieval_tokens ≈ list_capacity * retrieval_budget。也就是说设
     retrieval_budget=0.025 时，系统语义就是最多取约 2.5% middle-context tokens，而不是固定取一个很小的 k。见 cache_hub/
     adaptive_imi/runtime.py:60。

  论文里的关键表述建议这样定：

  > 我们不追求极小 token 数下的 top-k recall，因为即使 exact top-k 在 k 很小时也会丢失大量对 attention 输出有贡献的信息。长上下文
  > attention 压缩的合理目标是在固定比例预算 rho 下选择 token，例如 rho=2.5% 或 5%，使被选集合尽可能覆盖 full attention 中高 q·k 的
  > token，同时保持足够的上下文信息量。

  形式化可以写成：

  - 给定 query q 和上下文 keys K={k_i}_{i=1}^N
  - oracle 集合：T_rho(q) = Top ceil(rho N) tokens by q·k_i
  - 算法输出：S_rho(q)，满足 |S_rho| ≈ rho N
  - 指标：
    Recall@rho = |S_rho(q) ∩ T_rho(q)| / |T_rho(q)|

  这样比 recall@k 更适合你的系统，因为 k 应随上下文长度增长。例如 8K 上下文的 2.5% 是约 200 tokens，128K 上下文的 2.5% 是约 3200
  tokens。固定 k=64 在长上下文下本身就不是合理 attention 近似。

  你现有实验脚本已经在往这个方向走：experiments/imi_partition/run_ruler_recall_budgets.sh 会把 fine_pct=0.025/0.05 转成
  candidate_budget=round(search_tokens * fine_pct)，并把 recall_ks 设成同一个预算。也就是说它评估的是 recall@2.5%/5%，不是传统固定
  recall@k。见 experiments/imi_partition/run_ruler_recall_budgets.sh:28。

  一个需要注意的论文表述点：hybrid_8way_v2 不是直接返回 token-level top-k，而是 cluster-level retrieval with token budget
  truncation。因此它的优势应该描述为：

  - learned 8-way partition 提高 coarse cluster 对真实 q·k 高分 token 的覆盖；
  - subspace coarse search 降低候选 cluster 搜索成本；
  - full centroid rerank 修正子空间近似误差；
  - token-ratio budget 避免低 k exact top-k 的信息缺失问题。

  我建议论文主线定为：

  问题：低预算 sparse attention 如果只强调 tiny top-k，会导致 attention mass 和上下文证据缺失。
  目标：在固定比例 KV 预算下最大化 high-dot-product token 的覆盖。
  方法：learned 8-way IMI + subspace coarse filtering + full-centroid reranking。
  指标：Recall@2.5%/5%，可补充 attention mass retained@rho 和下游 RULER/LongBench 准确率。
  消融：learned partition vs sequential/pair-sequential/kmeans128/PQ；flat centroid vs subspace_rerank；不同 coarse pct 与 fine
  pct。

  当前数据里也能看到方向：在 outputs/new/eval_llama31_online_subspace_rerank_recall_summary.json，8192 长度、candidate_budget=1024、
  recall_k=256 下，learned 的 candidate_recall_mean=0.729，高于 sequential/pair-sequential 约 0.708/0.707，但低于 full 128D kmeans
  和 PQ。这可以作为 early evidence，但论文还需要补齐 2.5%/5% 多长度、多任务的正式表格。