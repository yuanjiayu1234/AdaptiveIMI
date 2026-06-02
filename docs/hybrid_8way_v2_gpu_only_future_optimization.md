# hybrid_8way_v2 GPU-only 后续优化路线

本文记录当前 `hybrid_8way_v2` GPU-only 路径的状态、仍不够优雅的地方，以及后续优化优先级。背景日志：

```text
/root/AdaptiveIMI/test_gpu_finish_atomic_clean.log
```

当前结论：正确性已经恢复，GPU kmeans 已经不是主瓶颈；后续主要优化空间在 CPU reorder、finish 队列调度，以及 CPU/GPU artifact contract 的进一步收敛。

## 当前正确数据流

现在 GPU-only 路径采用这个 contract：

```text
GPU raw KV
  -> GPU lightweight kmeans
  -> GPU compact metadata:
       centroids / cluster_sizes / cluster_counts / token_ids_by_cluster / offsets

raw KV D2H
metadata D2H:
  token_counts / cluster_counts / cluster_sizes / cluster_offsets / token_ids_by_cluster

CPU finish:
  raw KV + token_ids_by_cluster
  -> CPU reorder
  -> list_keys / list_values / cluster_sizes_cpu / cluster_offsets_cpu

GPU finish:
  compact centroids/sizes/counts
  -> full n_centroids decode metadata on GPU
  -> centroids / cluster_sizes_gpu / centroids_mask

layer_ready:
  CPU list metadata and GPU decode metadata are published together
```

关键约束：

- `centroids` 不 D2H。
- CPU 只接收 reorder 和 wave buffer 必需 metadata。
- Decode 仍使用全局 `n_centroids` contract，而不是 per-layer compact tensor。
- GPU compact metadata 到 full decode metadata 的转换在 GPU 上完成。
- CPU list layout 和 GPU decode metadata 的一致性由同一组 `cluster_counts/cluster_sizes/cluster_offsets` 维护。

## 为什么当前还不够优雅

当前实现为了兼容既有 decode/retrieval/async-update 逻辑，维护了两个视图：

```text
compact GPU build view
full GPU decode view
CPU list/retrieval view
```

这比早期错误路径安全，但还不是最简洁。理想设计应该显式定义一个 layer-level artifact，而不是让 Python 多处代码隐式约定 tensor 语义。

建议未来抽象：

```text
HybridLayerIndexArtifact
  DecodeMetadataGPU
    compact_centroids
    full_centroids
    full_cluster_sizes
    full_centroids_mask

  ListLayoutCPU
    token_ids_by_cluster
    cluster_sizes
    cluster_offsets
    list_keys
    list_values

  SharedShape
    batch_groups
    compact_clusters
    n_centroids
    token_counts
    cluster_counts
```

目标是让每个阶段只消费它需要的 view：

- GPU kmeans 只负责 compact metadata。
- CPU reorder 只负责 list layout。
- Decode 只看 full GPU decode metadata。
- Ready publish 只接受完整 artifact，不接受半成品 tensor。

这样可以减少“某个 tensor 正好 shape 对了但语义不对”的风险。

## 日志耗时分析

`test_gpu_finish_atomic_clean.log` 中有 4 个完整样本：

| sample | prefill | GPU build sum | reorder sum | finish enqueue sum | metadata D2H sum |
|---|---:|---:|---:|---:|---:|
| 1 | 48.11s | 2.35s | 10.24s | 15.89s | 2.1ms |
| 2 | 28.59s | 2.63s | 21.69s | 18.92s | 1.8ms |
| 3 | 26.14s | 2.45s | 17.52s | 19.13s | 2.1ms |
| 4 | 27.10s | 4.42s | 20.62s | 23.13s | 2.1ms |

观察：

- GPU kmeans 已经降到几秒级，通常不是主瓶颈。
- raw D2H wait 基本为 0，说明 raw copy 和主流程重叠得不错。
- metadata D2H 基本可忽略。
- CPU reorder 总和很大，但部分和 prefill 重叠；真正影响尾部的是 finish enqueue / reorder 的排队和长尾。
- 仍有明显抖动：同一层级别的 `reorganize_ms` 可从十几 ms 到 1s 以上。

因此 GPU-only 仍有时间优化空间，但优先级不是继续动 kmeans 初始化，而是压 CPU reorder 和 finish pipeline 的长尾。

## 2026-05 breakdown 暴露的新问题

新的 64k breakdown 对比：

```text
breakdown2: warmup=0, measured prefill ~= 58.7s
breakdown1: warmup=1, measured prefill ~= 30.6s
```

说明有两类问题需要分开看：

1. 冷启动污染测量。
2. warmed path 仍没有把 GPU-only index pipeline 和 prefill forward 充分重叠。

### 测量问题：冷启动不应进入第一条样本

冷启动包含 CUDA context、cuBLAS/FlashInfer、自定义 extension、allocator、pinned memory、CPU threadpool/OMP 初始化等。它会被归因到某个 Python wall timer 上，例如冷跑里 `kv_build_wall_ms` 会异常变大，但这不是 QKV 本身变慢。

要求：

- benchmark 必须先做未计入结果的 warmup/preflight。
- 正式 measurement 前执行 CUDA sync、清理 Python GC、重置 peak memory。
- `repeats=1` 时不能依赖 `drop_first_measurement`，因为 summary 里只有一条 ok record 时不会真正 drop。
- resume 匹配必须包含影响性能的参数，例如 `cache_ratio`、`gpu_only`、`breakdown_mode`、`max_new_tokens`、`input_len`、`batch_size`，避免复用旧实验记录。

推荐口径：

```text
load model
-> build tokenizer/data
-> unmeasured warmup generation, same length and backend config
-> cuda sync + empty_cache + reset peak memory
-> measured generation
```

### 实现问题：submit 不是轻量异步 submit

当前 GPU-only 路径理论上应该接近：

```text
prefill forward/MLP critical path
  + overlapped GPU kmeans
  + overlapped raw D2H / metadata D2H / CPU reorder
```

但现在实际更接近：

```text
prefill layer
  -> prefill_submit_kv_chunk()
     -> HybridGpuLayerBuildJob(...)
        -> allocate pinned raw KV buffers
        -> create streams/events
        -> start raw D2H
        -> prepare partition tensors
        -> keys.transpose(...).contiguous()
        -> launch GPU kmeans
     -> wait_gpu_metadata_build()
  -> continue layer compute
  -> sync()
     -> wait_gpu_sources_consumed()
     -> schedule finish worker
```

breakdown 中 `index_submit_block_ms ~= index_job_create_ms`，而 `metadata_wait_wall_ms` 很小，说明主要卡在 job constructor / eager start，而不是最后 wait。

当前关键阻塞点：

- `HybridGpuLayerBuildJob.__init__` 有副作用，会自动启动 raw D2H 和 GPU build。
- `_finish_hybrid_gpu_chunk_stream()` 立即调用 `job.wait_gpu_metadata_build()`。
- `sync()` 在主 prefill 线程里调用 `job.wait_gpu_sources_consumed()`，会等待 GPU metadata build 和 raw D2H。
- `keys.transpose(1, 2).contiguous()` 复制整层 middle KV，抢 GPU 带宽。
- partition tensors 每层/每样本动态创建，带来 Python loop 和 CUDA allocation。
- raw KV pinned CPU buffer 每层动态分配，可能触发 allocator/NUMA 抖动。
- CPU reorder 在后台执行，但长尾可能没被 prefill 完全盖住，最后压到 `prepare_cache()`。

### 与 RetroInfer 的可借鉴点

`/root/RetrievalAttention/library/retroinfer` 的方法和这里不同：RetroInfer 的 wave buffer / CPU index 逻辑不是这个 GPU kmeans + compact/full metadata contract。但它有几个工程点值得借鉴：

- 用显式 thread pool 提交后台任务，而不是在 Python submit path 内做重工作。
- `async_construction()` 和 `construction_sync()` 分离：提交快，最终只在明确边界等待。
- CPU worker 持有自己的任务队列和工作划分，主线程只做轻量 enqueue。
- sync 语义集中：不是每层 Python submit 都隐式等待。

不能照搬的地方：

- AdaptiveIMI GPU-only 有 GPU compact metadata、full decode metadata、CPU list layout 三个 artifact，必须同时维护一致性。
- GPU centroids 不应该 D2H。
- Decode 仍依赖 full `n_centroids` layout，不能直接消费 compact metadata。

因此这里要借鉴的是“轻量 submit + 后台流水 + 明确 sync 边界”，不是 RetroInfer 的具体 index/layout。

## 详细修改方案

### A. 先修测量口径

目标：冷启动不进入第一条 measurement，breakdown 和 throughput 口径一致。

修改点：

1. breakdown 脚本默认 `WARMUP_RUNS=1`。
2. 正式 measurement 前保留：

```python
gc.collect()
torch.cuda.empty_cache()
reset_cuda_peak_memory()
cuda_sync()
```

3. 对 breakdown 建议默认 `--no-resume`，或增强 `record_matches_run()`，把关键 config 纳入匹配。
4. metadata 里显式写入 warmup policy，例如：

```text
warmup_runs
warmup_same_length
measurement_after_warmup
resume_enabled
```

注意事项：

- warmup 只解决测量公平性，不代表性能优化。
- warmup 不应写入 measurement，但可以写一个 `event=warmup` 的 debug 行，默认关闭。
- `drop_first_measurement` 仍可保留给多 repeat 汇总，但不能替代 warmup。

### B. 让 GPU build job constructor 变纯

目标：`HybridGpuLayerBuildJob(...)` 只创建轻量对象，不启动 GPU/CPU 工作。

当前问题：

```python
if auto_start_raw_d2h:
    self.start_raw_d2h(...)
if auto_start_gpu_build:
    self.start_gpu_metadata_build(...)
```

建议修改：

```text
__init__:
  validate shape
  save refs
  save partition/static metadata handles
  initialize state/events to None
  do not allocate pinned raw KV buffer
  do not create streams
  do not launch kmeans

start_raw_d2h():
  acquire raw CPU buffer from pool
  create/reuse copy stream
  enqueue D2H

start_gpu_metadata_build():
  use cached partition tensors
  enqueue kmeans on low-priority or configured build stream
```

接口变化：

```python
job = HybridGpuLayerBuildJob(..., auto_start_raw_d2h=False, auto_start_gpu_build=False)
gpu_build_scheduler.submit(job)
```

注意事项：

- 构造函数不能再记录 `gpu_build_ms`，计时要迁移到 `start_gpu_metadata_build()`。
- 任何异常都要存入 `prefill_stream_errors[layer_idx]`，不能丢在后台线程。
- job 生命周期必须显式：`created -> submitted -> sources_consumed -> metadata_ready -> finished -> published`。

### C. 去掉 prefill submit path 的立即等待

目标：`prefill_submit_kv_chunk()` 只提交，不等待。

当前问题：

```python
job = HybridGpuLayerBuildJob(...)
job.wait_gpu_metadata_build()
```

建议修改：

```text
_finish_hybrid_gpu_chunk_stream(layer):
  record source_ready_event on current prefill stream
  create job
  store job in hybrid_gpu_jobs[layer]
  mark prefill_stream_end_submitted[layer] = True
  enqueue job to GPU build scheduler
  return immediately
```

注意事项：

- 不要在这里调用 `wait_gpu_metadata_build()`。
- 不要在这里启动 CPU reorder。
- 如果必须限制 in-flight 层数，用 semaphore/slot 控制，而不是直接 synchronize。
- 如果 slot 不够，可以选择：
  - 阻塞等待最早 job source release；
  - 或临时分配额外 workspace；
  - 但要把这类等待单独计时为 `raw_slot_wait_ms`。

### D. 改造 `sync()`：只调度后台 finish，不在主线程等

目标：每层 `prefill_update_kv_cache()` 后的 `kv_cache.sync()` 不应等待 GPU build/raw D2H。

当前问题：

```python
job.wait_gpu_sources_consumed()
job.start_cpu_metadata_d2h()
_schedule_hybrid_layer_finish(...)
```

建议修改：

```text
sync(layer):
  validate job exists
  if finish not scheduled:
      schedule background finish worker:
          wait_gpu_sources_consumed()
          start/finish metadata D2H
          CPU reorder
          install CPU list metadata
          materialize GPU decode metadata
          publish layer_ready
  return immediately
```

`prepare_cache()` 再做最终边界：

```text
prepare_cache():
  wait all unfinished layer_ready events
  super().prepare_cache(skip_prefetch=True)
  materialize optional decode search metadata
```

注意事项：

- 这样做后 `prepare_cache_ms` 可能上升，但如果后台重叠成功，整体 prefill TTFT 应下降。
- 如果 `prepare_cache_ms` 很大，说明后台 finish 没盖住，需要看 finish queue / reorder tail。
- Decode 之前必须保证全部 `layer_ready=True`，否则 sparse attention 会读半成品。

### E. 实现 raw KV workspace / in-flight slot 管理

目标：允许下一层 prefill 继续使用新的 KV workspace，而不是为了保护 source tensor 提前等待。

当前已有 `prefill_kv_workspace_available()` 检查，但缺少真正的 slot/pool 策略。

建议设计：

```text
GpuRawWorkspacePool
  max_inflight_raw_slots
  acquire(shape, dtype, device)
  release_when(job.sources_consumed_event)
```

流程：

```text
layer N writes KV into workspace A
job N holds workspace A
layer N+1 requests workspace
  if A still held, allocate/reuse workspace B
job N raw D2H done -> release A
```

注意事项：

- 默认 `max_inflight_raw_slots` 建议 2 或 3，先 A/B。
- slot 太多会增加显存；slot 太少会回到同步等待。
- 需要统计：
  - `raw_slot_acquire_ms`
  - `raw_slots_in_use_max`
  - `workspace_reuse_count`
  - `workspace_allocate_count`

### F. 去掉 `keys.transpose(...).contiguous()` 的大复制

目标：GPU kmeans 直接消费当前 KV layout 或 stride，不复制整层 middle KV。

当前路径：

```python
middle_keys = source_keys[:, middle_start:middle_end, :, :].transpose(1, 2)
ext.run_hybrid_8way_kmeans(keys.contiguous(), ...)
```

问题：

- `transpose` 后 `.contiguous()` 会复制 `[B, H, T, D]` 的整层 middle KV。
- 这不是 full attention prefill 的必要开销。
- 它会抢 GPU bandwidth，并降低与 prefill forward 的重叠效果。

建议修改：

1. 首选：修改 CUDA extension，支持 `[B, T, H, D]` contiguous layout。
2. 次选：extension 支持 stride 参数。
3. 不建议默认写双 layout，除非确认显存和带宽可接受。

注意事项：

- 修改 extension 后必须逐层校验 kmeans 输出一致性。
- token order、head order、batch group mapping 必须保持：

```text
row_idx = batch_idx * kv_heads + head_idx
```

- 这项会影响 correctness，必须有单元测试和端到端生成一致性测试。

### G. 缓存 partition tensors 和静态 build metadata

目标：不在每层/每样本 build 时重复 Python loop + CUDA allocation。

当前每次 build 都做：

```python
_prepare_partition_tensors(device=keys.device)
torch.tensor(k_values, device=...)
```

建议：

```text
cache init:
  for layer:
    precompute subspace_dim_ids
    precompute subspace_dim_masks
    precompute subspace_pair_ids
    precompute subspace_pair_masks
    precompute k_values_tensor
    precompute max_k / max_pairs / max_sub_dim

job:
  use cached tensors
```

注意事项：

- cache key 至少包含 `layer_idx/device/head_dim/kv_heads/subspace_parts/partition_path`。
- 如果 batch size 改变，rows 可能变，需要区分 per-layer static 和 per-batch expanded tensors。
- 尽量避免每次 `repeat_interleave` 产生新 CUDA tensor；可缓存 row-level mapping。

### H. 池化 pinned CPU raw KV 和 metadata buffers

目标：避免每层动态分配 pinned memory，减少 allocator/NUMA 抖动。

当前：

```python
self.keys_cpu = torch.empty(keys.shape, device="cpu", pin_memory=True)
self.values_cpu = torch.empty(values.shape, device="cpu", pin_memory=True)
```

建议：

```text
PinnedRawKvPool:
  buffers keyed by (shape, dtype)
  acquire/release

PinnedMetadataPool:
  token_ids_by_cluster / cluster_sizes / offsets / counts buffers
```

注意事项：

- pool 需要上限，避免 128k 长度时 pinned memory 无限增长。
- release 必须在 CPU reorder 完成后，而不是 raw D2H 完成后。
- NUMA 绑定如果可控，最好和 reorder worker 所在 CPU socket 一致。

### I. 优化 CPU reorder / finish queue

目标：让 CPU reorder 完全被 prefill/GPU build 盖住，减少 `prepare_cache()` 尾部等待。

建议沿用文档中的 P1/P2，但加更明确 profile：

```text
reorder_internal:
  rows
  tokens
  bytes
  tokens_per_task
  task_count
  thread_count
  elapsed_ms
  GB/s
  p50/p95/p99 task time

finish_queue:
  enqueue_ts
  start_ts
  finish_ts
  queue_wait_ms
  run_ms
```

A/B：

```text
tokens_per_task = 64, 128, 256
finish_concurrency = 1/2/3
threads_per_job = 24, 12, 8
```

注意事项：

- CPU 线程太多可能抢 Python 主线程和 CUDA launch 线程，导致 prefill forward 变慢。
- 并发不是越高越好，判断标准是 end-to-end prefill，而不是 reorder sum。
- `reorganize_ms` 是 per-layer sum，不能直接和 TTFT 相加；重点看 `prepare_cache_ms` 和 queue tail。

## 优化优先级

### P0: 保持 correctness contract 不变

不要再回到 compact GPU metadata 直接给 decode 用的路径。最终 decode metadata 必须满足：

```text
centroids.shape == [batch_groups, n_centroids, head_dim]
cluster_sizes_gpu.shape == [batch_groups, n_centroids]
centroids_mask.shape == [batch_groups, n_centroids]
padding_cluster_id == n_centroids - 1
```

未来任何优化都不能破坏这个 contract，除非同步重写 decode、prefetch、async-update、retrieval 和 padding 逻辑。

### P1: 修测量口径，移出冷启动

正式 benchmark 默认必须先 warmup，再记录 measurement。`repeats=1` 不能靠 drop-first。breakdown 实验建议默认禁用 resume 或严格匹配 run config。

成功标准：

- cold run 和 warmed run 分开记录。
- measurement 第一条不包含 extension/allocator/threadpool 初始化。
- breakdown 和普通 throughput 使用同一 warmup 口径。

### P2: 轻量 submit + 后台 finish

当前最值得优先改。目标是让 prefill 主线程只负责 record event 和 enqueue job，不等待 GPU metadata build、raw D2H、metadata D2H 或 CPU reorder。

关键改动：

- `HybridGpuLayerBuildJob.__init__` 变纯。
- `_finish_hybrid_gpu_chunk_stream()` 去掉 immediate wait。
- `sync()` 只 schedule background finish worker。
- `prepare_cache()` 作为 decode 前唯一强制 wait 边界。
- 增加 raw workspace slot/pool，允许多层 in-flight。

成功标准：

- warmed 64k 下 `index_submit_block_ms` 从约 9-10s 降到接近轻量 enqueue 成本。
- `prepare_cache_ms` 不显著上升；若上升，说明后台 finish 未盖住。
- `gpu_build_ms` 保持几秒级，不回退 correctness。

### P3: 减少 GPU build submit 成本

检查点：

- 去掉 `keys.transpose(...).contiguous()`。
- 缓存 partition tensors / k_values tensors。
- 池化 pinned raw KV 和 metadata buffers。
- 复用 CUDA streams/events。

成功标准：

- `index_job_create_ms` 不再接近 `index_submit_block_ms`。
- GPU build launch 不抢占 prefill forward 太多 bandwidth。

### P4: 优化 CPU reorder 长尾

检查点：

- `reorder_raw_kv_by_token_ids` 是否被 OMP/threadpool 竞争影响。
- `tokens_per_task` 是否太小，导致调度开销过高。
- K/V copy 是否可以更大粒度、向量化、按 cache line 对齐。
- 是否存在 `.contiguous()` 的隐藏 CPU copy。
- pinned CPU output 是否 NUMA/locality 不稳定。

建议实验：

1. 对 `tokens_per_task` 做 A/B：64、128、256。
2. 打印 reorder 内部 thread count、task count、rows、token_bytes、GB/s。
3. 将 K/V copy 保持同一 loop，但按更大 tile copy，减少 `parallel_for` 任务数。
4. 对每层 reorder 加 p50/p95/p99 统计，确认长尾来自 task scheduling 还是内存带宽。

### P5: 调整 finish 队列并发

当前配置里 `cpu_build_concurrency=1`，`cpu_worker_threads_per_job=12`。这对稳定性保守，但可能导致 finish 队列形成 tail。

建议实验：

```text
cpu_build_concurrency=1, threads/job=24
cpu_build_concurrency=2, threads/job=12
cpu_build_concurrency=3, threads/job=8
```

判断指标：

- prefill latency
- finish enqueue sum/max
- reorder sum/max
- CPU memory bandwidth 是否饱和
- GPU forward 是否被 CPU 线程争抢影响

如果并发 2 能降低 tail 且不增加抖动，可以作为默认候选。

### P6: 减少 Python finish 阶段开销

当前 finish 仍在 Python 里逐 row 组装 structured metadata，再写 CPU `cluster_sizes/cluster_offsets`。

未来可以把 CPU metadata install 变成批量 tensor copy：

```text
cluster_sizes_cpu[:, :compact_clusters] = metadata_cpu["cluster_sizes"]
cluster_offsets_cpu[:, :compact_clusters + 1] = metadata_cpu["cluster_offsets"]
tail offsets fill
```

这样可以减少 Python per-row loop。虽然 rows 通常只有 `batch * kv_heads`，不是最大瓶颈，但这是让代码更清晰的方向。

### P7: GPU compact -> full decode metadata 融合

当前在 Python/PyTorch 里执行：

```text
cluster_sizes_full.zero_()
centroids_mask_full.fill_(True)
copy compact centroids
copy sanitized sizes
copy active mask
```

这已经避免 centroids D2H，但仍有几次 GPU tensor op。未来可以写一个小 CUDA kernel 一次完成：

```text
materialize_decode_metadata(
  compact_centroids,
  compact_sizes,
  cluster_counts,
  full_centroids,
  full_sizes,
  full_mask
)
```

这不是当前最大瓶颈，但能让实现更干净，并减少 PyTorch op launch 数。

### P8: 可选 GPU chunk reorder A/B

因为 raw KV 必须 D2H，默认继续 CPU reorder 是合理的。但如果 CPU reorder 长尾无法压下去，可以做一个实验路径：

```text
GPU compact token order
  -> low-priority stream chunked GPU reorder into small staging buffer
  -> ordered KV D2H
```

注意：

- 不能一次 materialize 全层 ordered GPU list，显存不够。
- 必须 chunked staging。
- 必须低优先级 stream。
- 只有当 CPU reorder 明确成为 tail bottleneck 时才值得做。

这个实验需要严格 A/B，因为它可能抢 attention/GPU build 资源。

## 建议下一步

第一阶段先解决测量和流水线阻塞，不动 decode contract：

1. 统一 warmup 口径，冷启动不进入 measurement。
2. 给 `HybridGpuLayerBuildJob` 增加状态机和更细 submit profile。
3. 将 job constructor 改成无副作用。
4. `_finish_hybrid_gpu_chunk_stream()` 只 enqueue，不 wait。
5. `sync()` 只 schedule finish worker，不 wait sources consumed。
6. 实现 raw KV workspace slot/pool，默认 A/B 2 和 3 个 in-flight slots。
7. 再做 CPU reorder 内部 profile 和 `tokens_per_task` / finish concurrency A/B。

成功标准：

- 64k warmed prefill 明显接近 `full prefill attention/MLP + GPU kmeans`，而不是 30s+。
- 96k prefill 稳定低于 25s，而不是偶发低于 25s。
- 结果和 CPU-only / 已知正确 GPU path 一致。
- 默认路径不启用额外 D2H debug。
- GPU centroids 不传 CPU。

## 修改注意事项清单

- 不要破坏 full decode metadata contract。
- 不要让 decode 读到未 `layer_ready` 的层。
- 不要把 compact GPU metadata 直接给 decode 用。
- 不要为了释放 workspace 在 prefill 主线程 `synchronize()`。
- 所有后台异常必须回传到主线程，并在 `prepare_cache()` 或 `ensure_layer_ready()` 抛出。
- 后台线程不能无限吞内存；raw slots、pinned CPU buffers、metadata buffers 都要有上限。
- 任何新增 profile 默认不能引入 CUDA sync；sync-heavy profile 必须放在 detail/debug 模式。
- 判断性能时看 critical path：`prefill_total_ms`、`forward_wall_ms`、`index_submit_block_ms`、`prepare_cache_ms`。`reorganize_ms` 是 layer sum，不能直接和 TTFT 相加。
- 每个优化都要做 correctness A/B：同输入、同 seed、同 decode search mode，对比输出和必要 metadata shape。
