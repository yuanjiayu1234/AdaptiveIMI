• 合理的数据流应该是：GPU 建索引不能作为一个随时抢 SM 的后台任务，而应该是 prefill 流水线里被显式调度的阶段。RetrievalAttention 合理的地方不是它“用 GPU 建索引”，而是它把 GPU
  建索引、D2H copy、CPU 组织索引拆成了清楚的阶段，并且用事件控制它们的依赖。

  当前 hybrid_8way_v2 最大问题是调度点不合理。日志里很明显：

  - /root/AdaptiveIMI/test.log:2315：layer=7 mode=gpu phase=start
  - /root/AdaptiveIMI/test.log:2316：下一层 layer=8 chunk=3 直接变成 829.54ms
  - /root/AdaptiveIMI/test.log:2321：GPU build 结束，耗时 407.97ms

  也就是说，GPU index build 被硬塞到下一层 prefill 的第 3/4 个 chunk 附近执行。正常 chunk 是几十 ms，这里被放大到几百 ms。这个现象和代码一致：cache_hub/adaptive_imi_v2/
  cache.py:749 里 pending GPU build 的 ready 条件是 next_layer 完成 chunk=2 后启动。这是非常危险的固定规则，因为它完全不知道当前 GPU 是否正在跑关键 prefill kernel。

  合理的数据流应该长这样：

  Layer L forward on main stream
          |
          |-- record KV-ready event
          |
          |-- async D2H raw/middle KV staging on copy stream
          |
          +--> CPU route:
          |       CPU worker consumes staged KV
          |       builds list/layout metadata
          |       marks layer ready
          |
          +--> GPU route:
                  GPU index scheduler waits for an explicit safe window
                  runs bounded GPU clustering / labeling kernels
                  emits compact metadata only
                  releases temporary tensors
                  CPU worker does final layout/reorder if needed
                  marks layer ready

  这里的重点有四个。

  第一，prefill 主流必须是最高优先级。GPU index build 不能在“下一层刚开始算大 chunk”的时候突然启动。CUDA stream priority 只能部分缓解，不能抢占已经发出去的长 kernel。所以真正
  合理的设计是：GPU build kernel 要么拆小、受预算控制，要么只在明确窗口运行。

  第二，GPU build 应该只产出紧凑元数据。RetrievalAttention 的思路是 segment_k_means 产出 centroids / clusters / cluster_size 这类紧凑结构，后续 CPU wave buffer 组织再异步做。
  你们现在的 CUDA builder 会分配 full_centroids = [rows, max_tokens, head_dim]，见 library/AdaptiveIMI/cpp_extensions/cuda/hybrid_8way_builder.cu:491，而 decode 选择 cluster
  主要需要 subspace_centroids、compact_codes、mask 之类数据，见 cache_hub/adaptive_imi_v2/cache.py:326。这会让 GPU build 的显存、初始化、copy 和 cache 压力都被放大。

  第三，GPU build 和 CPU 后处理要解耦。现在 _finish_pending_gpu_build() 更像是一个“大包”：GPU build、metadata D2H、CPU organize 一起被一个 pending job 串住。合理做法是 GPU
  worker 只负责发 GPU kernel 并尽快释放大临时；CPU worker pool 消费 labels/codes/raw KV 做最终组织。这样 GPU 不会等 CPU，CPU 也不会因为 GPU job 粒度太大而空转。

  第四，CPU/GPU 分层策略不应该是简单 alternate。CPU/GPU 联合建索引是有价值的：CPU 建索引不会抢 prefill SM，天然适合和 GPU forward 重叠；GPU 建索引适合处理 CPU 队列快要形成
  tail latency 的层，或者 CPU 特别慢的层。但它应该由调度器根据 backlog、剩余 prefill 时间、GPU build 估计耗时来决定，而不是固定奇偶层。

  RetrievalAttention 里值得借鉴的是这个调度形态：

  - 在 /root/RetrievalAttention/model_hub/LLM.py:34 里按层 prefill，层内完成 cache update、attention，然后调用 kv_cache.sync()。
  - 在 /root/RetrievalAttention/cache_hub/retroinfer_cache_gpu.py:427 用 copy stream 做异步 offload。
  - 在 /root/RetrievalAttention/cache_hub/retroinfer_cache_gpu.py:453 直接做 GPU segment_k_means，但产物是紧凑结构。
  - 在 /root/RetrievalAttention/cache_hub/retroinfer_cache_gpu.py:495 等 copy event 后，把 CPU wave buffer construction 异步化。

  我们不需要照抄它的索引结构，但要吸收它的原则：GPU-heavy 阶段是可预测的，CPU-heavy 阶段是异步的，跨阶段靠 event 传递，元数据保持紧凑。

  对 hybrid_8way_v2，我建议的合理版本是：

  1. prefill main stream 只负责模型计算和必要 KV 写入
  2. 每层 KV-ready 后立刻异步 staging，copy stream 只做 D2H/H2D
  3. CPU 层直接进入 CPU pipeline，允许多个 layer job 并发
  4. GPU 层进入 GPU-index scheduler 队列，不立即启动
  5. scheduler 只在以下窗口启动 GPU build:
     - 明确的 layer boundary
     - prefill chunk 间的小预算窗口，前提是 GPU kernel 已切片
     - prefill 尾部，处理剩余 backlog
  6. GPU build 输出 compact labels/codes/subspace centroids
  7. CPU worker 做 reorder/list metadata
  8. decode 前只等待未 ready 的层

  当前最应该避免的是这个数据流：

  Layer L KV ready
    -> pending GPU build
    -> 等到 Layer L+1 chunk 2 后启动
    -> 和 Layer L+1 chunk 3/4 抢 GPU
    -> prefill chunk 延迟暴涨

  这正是日志里 45s/60s 的主要异常来源之一。不过要注意：即使把这些 GPU index 干扰全部移掉，你们现在的 base prefill 仍然偏慢，因为 model_hub/prefill.py:345 是 layer-major +
  chunked，并且每层 chunk 都有 CPU/GPU hidden state 搬运，例如 model_hub/prefill.py:124 和 model_hub/prefill.py:245。所以 20s 目标需要两步：先修 GPU index 调度干扰，再优化
  prefill 本身的数据流。