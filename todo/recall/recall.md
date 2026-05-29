我想证明我的子空间维度重组方法的有效性

我的算法的核心创新点在于讲key划分成若干个不同的子空间，为了不破坏RoPE特性和维度之间存在的依赖，我首先将128维重组为64个pair，然后在真实数据集上学习得到每一层模型最精确的划分（Query感知），根据这个划分进行子空间划分，然后每个子空间单独聚类。

我想和
1、128维kmeans
2、八个子空间顺序划分0-15、16-31……
3、PQ（参照/root/PQCache的pq_search）
4、我的方法
比较recall（真值为Q与所有key暴力计算距离，求距离最近的前若干个key）
scores = Q @ K.T
gt_topk = topk(scores, K)


但是有几个问题需要解决
一、GQA下怎么测量recall
采用聚合Q的方式


二、这几种方法的decode搜索该怎么做？

1、128维kmeans decode：
query
↓
最近cluster
↓
扫描cluster内部
↓
topk


3、PQ
直接使用：PQ Encode
ADC Search

query
↓
distance table
↓
ADC
↓
topk


2、八个子空间顺序划分0-15、16-31……和4、我的方法deocde：
query
↓
按照学习得到的划分重组
↓
计算每个子空间距离
↓
多路归并
↓
候选集合
↓
rerank