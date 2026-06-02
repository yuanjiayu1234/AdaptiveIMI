
  代码事实是确定的：

  RetroInfer 在 RetrievalAttention/cache_hub/retroinfer_cache.py:764：

  batch_gemm_softmax(...)
  torch.sum(self.softmax_o, dim=1, out=self.dist)

  它的测试文件 RetrievalAttention/library/retroinfer/test/test_batch_gemm_softmax.py:15 明确写了 reference：

  dist = torch.bmm(A, B.transpose(1, 2))
  dist = dist / math.sqrt(A.size(-1))
  dist = torch.softmax(dist, dim=-1)

  所以 RetroInfer 是：

  score(c) = sum_h softmax(q_h · C / sqrt(d))[c]

  AdaptiveIMI 的 flat selector 在 AdaptiveIMI/library/AdaptiveIMI/cpp_extensions/cuda/compute_impl.cu:83 先
  做：

  acc += query_grouped[g, d]
  shared_query[d] = acc

  然后在 AdaptiveIMI/library/AdaptiveIMI/cpp_extensions/cuda/compute_impl.cu:100：

  sim += shared_query[d] * centroid_ptr[d]

  所以 AdaptiveIMI 是：

  score(c) = (sum_h q_h) · c
           = sum_h (q_h · c)

  subspace rerank 路径也是同样先 sum query，再算 centroid score。

  数学上：

  如果 RetroInfer 是：

  sum_h (q_h · c)

  那确实和你的：

  (sum_h q_h) · c

  等价。

  但 RetroInfer 实际多了 softmax：

  sum_h softmax(q_h · C)[c]

  这个和：

  sum_h (q_h · c)

  不等价，因为 softmax 的分母是每个 head 对所有 clusters 的归一化：

  softmax(q_h · C)[c] = exp(q_h · c) / sum_j exp(q_h · c_j)

  所以它不只是对单个 cluster 做单调变换，而是会改变 cluster 之间的相对排名。

  我跑了一个最小例子：

  head logits:
  head1: A=10, B=9,  C=0
  head2: A=0,  B=9,  C=10

  Adaptive raw sum：

  A=10, B=18, C=10
  rank: B > A/C

  RetroInfer sum-softmax：

  A=0.731, B=0.538, C=0.731
  rank: A/C > B

  所以它们连 top cluster 排名都会变。

  更准确的结论是：

  - 等价条件：每个 head 只算 raw dot，然后加起来。
  - 实际 RetroInfer：每个 head 先 softmax 成一个 cluster 概率分布，再加起来。
  - 实际 AdaptiveIMI：先把 query heads 加成一个 query，再 raw dot 排序。
  - 所以两者不是数学等价。

  但我也修正一点：这只能说明 scorer 差异是真实存在的，不能单独证明它就是分数差的唯一原因。要证明它的贡献，需
  要做一个固定 budget、固定 index 的 ablation：把 AdaptiveIMI 的 cluster score 改成 RetroInfer 式 per-head
  softmax + sum，只跑 niah_multikey_3。如果分数明显回升，就能坐实这个原因。