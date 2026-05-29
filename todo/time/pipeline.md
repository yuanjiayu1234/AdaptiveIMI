我想验证我的流水线设计的高效性，所以我需要测试1、我的GPU/CPU混合构建流水线在不同长度的prefill时延
对比baseline：
1）每一层CPU上串行
2）每一层GPU上串行
3）全CPU上并行
4）不构建索引，纯prefill（full attention）
5）GPU/CPU混合，同时每层并行（我的方法）
测试在8k、16k、32k、64k、96k、128k、256k prefill时延（应该是ttft）这样包含缩影构建时间