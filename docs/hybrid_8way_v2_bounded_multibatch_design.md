# hybrid_8way_v2 有界多 batch 构建设计

本文档描述 `AdaptiveIMI` 的 `hybrid_8way_v2` 在 prefill 多 batch 场景下应该如何改。目标是谨慎改动，保留现有 CPU 构建索引路径和 GPU 构建索引路径的核心逻辑，只在任务提交、buffer 生命周期和并发控制上补一层有界调度。

## 目标

1. prefill 统一走 chunk path，支持 batch 内多样本。
2. 优先支持 lockstep multi-batch：同一 batch group 内 `valid_start` 必须一致。
3. CPU 和 GPU 都可以构建索引，利用 hybrid 优势，而不是简单固定全 CPU 或全 GPU。
4. 构建任务可以和后续层 prefill overlap，但必须有内存和并发上限，不能无限 pending。
5. 不破坏现有 CPU/GPU 构建路径的先后逻辑：
   - prefill 仍按 layer 顺序执行。
   - build task 仍按 layer ready 顺序提交。
   - `layer_build_modes` 仍是每层 CPU/GPU 归属的来源。
   - `ensure_layer_ready()` / `prepare_cache()` 仍负责在 decode 前等待未完成层。

## 非目标

第一阶段不做复杂重构：

- 不重写 GPU builder 的 kmeans/reorder kernel。
- 不重写 CPU pipeline 的现有单 batch 路径。
- 不引入多层嵌套的 scheduler 类体系。
- 不把 ragged batch 伪装成完整支持。`valid_start` 不一致时先分组或 fail fast。

## 当前数据流

当前 prefill 生产 KV 的顺序是：

```text
layer0 prefill chunk -> layer0 KV ready -> submit index build
layer1 prefill chunk -> layer1 KV ready -> submit index build
...
layer31 prefill chunk -> layer31 KV ready -> submit index build
```

CPU 路径：

```text
prefill_update_kv_cache()
  -> _submit_hybrid_cpu_layer_build_chunked()
  -> IMIPipeline.begin_index_stream()
  -> submit_index_stream_raw_sequence_chunk()
  -> _schedule_hybrid_layer_finish()
  -> _finish_pipeline_layer_index()
  -> _load_layer_metadata()
```

GPU 路径：

```text
prefill_submit_kv_chunk()
  -> fill hybrid_staging_keys/values
  -> _finish_hybrid_gpu_chunk_stream()
  -> _enqueue_pending_gpu_build()
  -> _prepare_pending_gpu_raw_d2h()
  -> HybridGpuLayerBuildJob.start_raw_d2h()
  -> HybridGpuLayerBuildJob.start_gpu_metadata_build()
  -> _finish_layer_index()
  -> _load_layer_metadata()
```

当前问题在 GPU 路径的资源生命周期：

```text
GPU staging/source raw KV
+ pending job 持有 source tensor
+ pinned CPU raw KV copy
+ GPU metadata/kmeans 临时张量
```

如果多个 layer 同时 pending，98k 长上下文下会快速放大内存。单 batch Llama 8 KV heads、128 head_dim、bf16 时，一层 middle keys+values 约 400MB；pending 4 层就可能产生 GB 级 GPU source 和 pinned CPU raw copy。

## 借鉴 ParisKV 和 RetroInfer

ParisKV 的优点：

- 多 batch 第一阶段采用 lockstep：batch 内 `valid_start` 一致。
- 整批处理 `[batch, seq, kv_head, head_dim]`，不逐样本退化。
- 每层 offload/copy 完成后尽快释放该层 inflight buffer。
- 不让很多层 full raw KV 同时挂在后台。

RetroInfer 的优点：

- 把 `batch * group` 作为全局 row。
- `start_bdx + process_groups` 决定当前 micro-batch 写入全局 storage 的位置。
- 构建输入可以是当前 micro-batch，但输出落到全局 batch rows。

AdaptiveIMI 的优势：

- CPU 和 GPU 都能构建索引。
- 因此调度器可以按资源压力分配 CPU/GPU，不必所有层都挤 GPU，也不必所有层都等 CPU。

## 设计原则

### 1. Producer 和 Builder 解耦，但不要过度抽象

prefill 主线程只做两件事：

```text
生成当前 layer 的 KV
把当前 layer 的 build task 交出去
```

不要让 prefill 主线程决定复杂的 CPU/GPU 内部细节。CPU/GPU builder 原有实现继续保留。

### 2. Buffer 必须有状态和上限

不允许每层随手新分配大 buffer。至少要有下面的状态：

```text
FREE -> FILLING -> READY -> BUILDING -> COMMITTED -> FREE
```

第一版不需要新建复杂类，可以在 `Hybrid8WayIMICache` 里用少量字段管理：

```text
gpu_raw_slots
gpu_raw_slot_in_use
gpu_raw_slot_layer
pinned_raw_bytes_inflight
gpu_raw_bytes_inflight
```

只要能表达“当前还有多少 raw KV/pinned copy 被持有”，就足够。

### 3. Backpressure 是正确行为

当 GPU/CPU build 队列或 buffer slot 满了，prefill producer 应该等待一个 slot 释放，而不是继续分配。

```text
if no build slot:
    wait condition variable
else:
    submit build task
```

这会牺牲一部分 overlap，但保证不会被 OOM killer 杀掉。正确的性能优化应该在有界资源内做。

### 4. Layer 顺序不变

prefill 仍然严格按 layer 顺序执行。build 可以后台并行，但提交顺序和 ready 检查保持清晰：

```text
submit order: layer0, layer1, ..., layer31
decode before layer i: ensure_layer_ready(i)
prepare_cache: wait all unfinished layers
```

同一设备队列建议 FIFO，避免高层插队导致低层 decode 等待。

## 建议的数据流

### Prefill layer 完成后

```text
layer_prefill_chunked(layer_idx)
  -> key_states/value_states ready
  -> build_device = layer_build_modes[layer_idx]
  -> submit_build_task(layer_idx, build_device, start_bdx, bsz, token_counts)
```

### CPU task

第一阶段保持现有 batch=1 CPU 路径不变：

```text
CPU task
  -> _submit_hybrid_cpu_layer_build_chunked()
  -> IMIPipeline existing raw sequence stream
  -> _finish_pipeline_layer_index()
```

多 batch CPU 路径作为第二阶段，按 RetroInfer 思路扩展：

```text
keys: [bsz, kv_head, tokens, head_dim]
process_groups = bsz * kv_head
global_row = (start_bdx + local_bdx) * kv_head + head
```

扩展时保留旧 API，新增 batch-aware API，例如：

```text
begin_index_stream_batch(token_counts, start_bdx)
submit_index_stream_batch_chunk(...)
finish_index(...)
```

这样不会破坏现有 CPU 单 batch 路径。

### GPU task

GPU path 保留 `HybridGpuLayerBuildJob` 和 `HybridLayerBuildJob.finish()`，但改变 source buffer 的来源：

```text
旧逻辑:
  每层分配 staging -> pending 持有 -> D2H 分配 pinned copy

新逻辑:
  从有限 gpu_raw_slot 取得 buffer
  prefill chunk 写入 slot
  提交 GPU build
  raw D2H 和 metadata build 完成
  finish/reorder/load metadata
  release slot
```

GPU task 需要两个预算：

```text
max_gpu_builds_inflight
max_gpu_raw_slots
max_pinned_raw_bytes_inflight
```

默认应保守，例如：

```text
max_gpu_builds_inflight = 1
max_gpu_raw_slots = 1 or 2
max_pinned_raw_bytes_inflight = 根据 batch * seq_len 估算
```

之后再通过日志调优。

## 多 batch 策略

第一阶段采用 lockstep group：

```text
batch 内 valid_start 一致 -> 整 batch group 处理
valid_start 不一致 -> 按 valid_start 分组
```

每个 group 的写入规则：

```text
global_batch = start_bdx + local_bdx
global_row = global_batch * kv_head + head_idx
```

这和 RetroInfer 的 `current_start_batch * group_num + idx` 一致。

RULER 正确性测试建议先使用：

```text
NUM_SAMPLES=3
BATCH_SIZE=3
同一任务
确认 tokenized 后同 valid_start 或按 valid_start 分组
```

真正 ragged batch 后续再接 varlen attention，不在第一阶段假装支持。

## 最小修改计划

### Phase 0: 恢复保守行为

- 撤掉激进的 `gpu_only -> interval=1` 默认行为。
- 保留环境变量覆盖能力。
- 默认不要让 GPU-only 在没有预算控制时大量 overlap。

### Phase 1: 补日志和预算

新增轻量日志，不改变 builder：

```text
layer_idx
build_device
source_bytes
pinned_bytes_inflight
gpu_source_bytes_inflight
pending_gpu_builds
pending_cpu_builds
slot_wait_ms
gpu_build_ms
raw_d2h_wait_ms
reorganize_ms
```

没有这些数字，不继续调性能。

### Phase 2: GPU raw slot 有界化

只改 GPU submit 周围逻辑：

- `_begin_hybrid_gpu_chunk_stream()` 不直接无限分配 staging。
- 从有限 slot 获取 source buffer。
- `_finish_hybrid_gpu_chunk_stream()` 提交 task 后不释放 ownership，交给 job 完成后 release slot。
- `_finish_pending_gpu_build()` / `_finish_layer_index()` 完成后归还 slot 并通知等待者。

GPU builder 内部不改。

### Phase 3: CPU/GPU 调度保守 hybrid

先保留 `layer_build_modes` 的显式先后逻辑：

```text
cpu_only / gpu_only / alternate / explicit / gpu_every_n
```

只在资源不足时做 backpressure，不自动改层归属。

稳定后再考虑新增：

```text
policy = dynamic_bounded
```

动态策略必须满足：

```text
GPU 空闲且 slot 足够 -> GPU
CPU 空闲且 CPU buffer 足够 -> CPU
都满 -> 等待
```

### Phase 4: CPU batch-aware 扩展

按 RetroInfer 模型扩展 CPU pipeline，保留旧单 batch API。目标是让 CPU 也能处理：

```text
[bsz, kv_head, tokens, head_dim]
```

并把结果写入全局 rows。这个阶段涉及 C++，必须单独验证。

## 验证顺序

1. `batch_size=1`, `gpu_only`, 8192 context，确认不回退、不 OOM。
2. `batch_size=1`, `gpu_only`, 98304 context，确认接近原 40s 水平，并打印 phase breakdown。
3. `batch_size=3`, lockstep input, 8192 context，确认一次 batch 输出 3 条。
4. `batch_size=3`, lockstep input, 98304 context，确认无 OOM。
5. `alternate` 或显式 CPU/GPU 混合，确认 CPU path 仍按原先逻辑执行。
6. 再做 CPU batch-aware C++ 扩展。

## 关键风险

1. pinned CPU memory 比普通 CPU memory 更容易触发系统压力，必须计入预算。
2. GPU metadata build 会抢 prefill SM，不能只看内存，还要看 prefill chunk 延迟。
3. batch>1 时 raw KV source bytes 线性增长，默认并发必须更保守。
4. 如果 `valid_start` 不一致，矩形 attention 可能错误，必须分组或 fail fast。

## 推荐结论

不要引入复杂多层架构。推荐在 `Hybrid8WayIMICache` 内做一个轻量的有界资源层：

```text
现有 CPU builder: 保留
现有 GPU builder: 保留
新增: source slot / pinned byte budget / pending build budget
新增: submit 时 backpressure
新增: finish 时 release slot
```

这样可以最大程度复用已有 CPU/GPU 逻辑，同时解决多 batch 和长上下文下的 OOM、等待和调度失控问题。
