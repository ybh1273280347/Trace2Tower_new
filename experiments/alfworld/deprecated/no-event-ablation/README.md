# 无 ALFWorld 事件消融

此目录保留已经废弃的 ALFWorld 实现，作为机制消融。该实现使用没有
ALFWorld 事件标签的 embedding change-point 分段，因此事件转移矩阵退化为
实例邻接，运行时也使用旧版纯语义 Mid 检索。

这些结果仍可用于证明移除领域事件抽取会损害 Trace2Tower，但不能作为完整算法
的官方证据，也不能描述为 ALFWorld original-concept 结果。

AgentBench 原始 rollout 与共享 manifest 仍保留在权威位置，因为它们同时是消融
管线和修复后官方事件管线的有效输入。
