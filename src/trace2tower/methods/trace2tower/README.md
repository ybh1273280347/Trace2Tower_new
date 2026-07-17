# Trace2Tower Package Layout

本包按 `Trace2Tower原始资料.md` 中的算法阶段组织：

- `core/`: Trace2Tower 配置与领域模型，不包含算法流程。
- `preprocessing/`: 原始轨迹转 transition、通用分段和 segment embedding 编码。
- `adapters/alfworld/`: ALFWorld 动作解析、事件标注和 segment signature。
- `adapters/webshop/`: WebShop 动作解析、事件标注及 WebShop 专属分支图实验。
- `eigen_trace/`: transition-aware EigenTrace 图和 contrastive spectral decomposition。
- `induction/`: Low/Mid/High 技能归纳、High path 与社区发现。
- `rendering/`: 将确定性归纳结果渲染为模型可读技能文本。
- `artifacts/`: Tower snapshot、内容哈希和完整性合同。
- `inference/`: 部署期 High-to-Mid 检索、rewrite 和上下文格式化。

依赖方向应从数据集适配和预处理流向核心算法，再流向归纳、artifact 与 inference。
`eigen_trace/` 不得依赖 benchmark adapter；benchmark 特例只能留在 `adapters/` 或构建脚本。
