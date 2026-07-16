# Trace2Tower 最终实验实施文档

> 面向 Codex 的唯一实施规范
> 文档目标：只规定做什么、怎么做、按什么顺序做。
> 本文不讨论论文写作措辞，不引入原始设计之外的新研究问题。

---

## 0. 实验边界

### 0.1 数据集、主结果与消融

实验只覆盖：

- ALFWorld
- WebShop

主结果方法固定为：

1. No Skill
2. Semantic-Only Clustering
3. Flat Skill Summary
4. SkillX
5. Trace2Tower-Static
6. Trace2Tower-Full

核心消融固定为：

1. No Transition Edge
2. No Outcome Edge
3. No Contrastive Decomposition
4. Trace2Tower-Static，用于衡量部署优化的增益

### 0.2 组件职责

Trace2Tower 的以下步骤由确定性代码完成：

- 原始动作解析
- transition 序列化
- 轨迹切分
- embedding
- 图构造
- 谱分解
- 聚类
- High path 挖掘
- 检索
- refinement proposal 生成

LLM 负责：

- 训练轨迹 rollout
- Flat Skill Summary
- SkillX 官方流程中的 LLM 调用
- Trace2Tower Mid skill card 渲染
- Trace2Tower High skill card 渲染
- 测试阶段 Agent 执行

Mid cluster 的成员、High path 的 Mid ID 顺序和 refinement 的结构关系由确定性代码给出；LLM 只将这些结构渲染为可执行文本。

SkillX 保持官方 extraction、clustering、merger、filter、retrieval 与 Prompt，仅通过公共 LLM runtime 和 benchmark adapter 接入。

测试集只用于最终评测。轨迹池构建、技能构建、超参数确定和部署优化全部在训练阶段完成。


## 1. 仓库与版本锁定

Codex 开始编码前先检查当前仓库，直接复用已经 Git 引入的依赖。

必须找到已经 Git 引入的：

- AgentBench
- SkillX

建议位置如下，但以当前仓库实际路径为准：

```text
third_party/
├── AgentBench/
└── SkillX/
```

开始实验前生成：

```text
artifacts/reproducibility/source-lock.json
```

格式固定为：

```json
{
  "agentbench": {
    "path": "实际相对路径",
    "commit": "实际 git commit"
  },
  "skillx": {
    "path": "实际相对路径",
    "commit": "实际 git commit"
  },
  "main_repository": {
    "commit": "当前主仓库 commit"
  },
  "python": "完整 Python 版本",
  "platform": "操作系统与架构"
}
```

当前参考版本：

- AgentBench：`d1e4a10db08c87075c78972e48ecc182be03e2d5`
- SkillX：`36747f424a17ea041e476adf2ff976a206ec9c30`

实际运行记录本地检出的真实 commit；若与参考版本不同，在日志中明确记录差异。

---

# 第一阶段：以 AgentBench 固定统一实验配置

## 2. 直接复用 AgentBench 任务定义

实验配置、任务索引、环境交互轮次和结果判定直接复用 AgentBench 当前仓库中的：

```text
configs/tasks/alfworld.yaml
configs/tasks/webshop.yaml
src/server/tasks/alfworld/task.py
src/server/tasks/webshop/task.py
```

### 2.1 ALFWorld 固定配置

```yaml
train_split: train_valid
test_split: new_std
max_step: 20
```

执行工具：

```text
take_action(action: string)
```

字段含义：

| 字段 | 含义 |
|---|---|
| `action` | Agent 提交给 ALFWorld 环境的原始动作字符串 |

单任务分数：

```text
primary_score = 1.0  当任务成功
primary_score = 0.0  当任务失败
```

聚合指标：

```text
success_rate = 成功任务数 / 有正式结果的任务数
```

Trace2Tower 在内部将 `action` 解析为 primitive action，但环境调用仍使用 `take_action(action)`。

### 2.2 WebShop 固定配置

```yaml
train_start: 1000
train_end: 12000
test_start: 0
test_end: 200
max_rounds: 20
```

范围采用左闭右开：

```text
train indices = [1000, 12000)
test indices = [0, 200)
```

执行工具只有：

```text
search_action(keywords: string)
click_action(value: string)
```

字段含义：

| 工具 | 参数 | 含义 |
|---|---|---|
| `search_action` | `keywords` | Agent 提交给 WebShop 搜索框的原始关键词 |
| `click_action` | `value` | Agent 从当前页面可点击项中选择的原始值 |

单任务分数：

```text
primary_score = WebShop 环境返回的 final reward
```

聚合指标：

```text
mean_reward = 所有正式任务结果的 primary_score 平均值
```

### 2.3 配置一致性检查

实现：

```text
experiments/common/agentbench_contract.py
```

启动实验时读取本地 AgentBench 配置并与实验 manifest 对照。配置不一致时输出具体差异并终止当前运行，由人工确认后更新 manifest。

## 3. 固定任务 Manifest

所有方法必须消费同一份任务 manifest。

生成：

```text
artifacts/manifests/
├── alfworld_train.jsonl
├── alfworld_test.jsonl
├── webshop_train.jsonl
└── webshop_test.jsonl
```

每行格式：

```json
{
  "benchmark": "alfworld",
  "split": "train",
  "sample_id": "稳定样本 ID",
  "agentbench_index": 0,
  "repeat_id": 0
}
```

生成规则：

1. 调用 AgentBench task 的 `get_indices()`。
2. 按 index 升序固定顺序。
3. 所有方法直接读取同一份 manifest。
4. manifest 生成后保持只读。
5. 失败、超时和 task error 按原任务 ID 写入结果，重试时仍使用同一任务。

---

## 4. 构造共享训练轨迹池

先运行统一 No-Skill Agent，在 AgentBench 训练 manifest 上生成轨迹。

输出：

```text
artifacts/trajectories/
├── alfworld/no_skill_train/
│   ├── shard-00.jsonl
│   ├── ...
│   └── shard-09.jsonl
└── webshop/no_skill_train/
    ├── shard-00.jsonl
    ├── ...
    └── shard-09.jsonl
```

所有技能方法必须使用这一个共享轨迹池：

- Semantic-Only Clustering
- Flat Skill Summary
- SkillX
- Trace2Tower-Static
- Trace2Tower-Full 的初始塔

各方法的初始技能库都从这一个共享轨迹池构建。

### 4.1 轨迹记录的统一语义

轨迹池保留 AgentBench 原始任务输出，并通过 adapter 向 SkillX 和 Trace2Tower 提供统一的读取视图。无需为了统一字段名而重写 AgentBench 的原始日志。

一条轨迹表示：

```text
任务目标
-> 当前环境反馈
-> Agent 动作
-> 动作后的环境反馈
-> 最终 benchmark 结果
```

其中 `observation` 指环境在当前轮提供给 Agent 的文本状态：

- ALFWorld 中是当前家庭环境、可见物体和动作结果的文本反馈。
- WebShop 中是当前页面的文本内容及可执行操作信息。

建议 adapter 提供以下语义字段：

| 字段 | 含义 |
|---|---|
| `sample_id` | AgentBench 中该任务的稳定标识 |
| `task_goal` | 环境提供的原始任务目标 |
| `step_index` | 当前交互轮次，从 0 开始 |
| `observation` | Agent 执行动作前看到的环境文本 |
| `action_name` | 实际调用的工具名 |
| `action_arguments` | 该工具收到的原始参数 |
| `next_observation` | 环境执行动作后返回的文本 |
| `reward` | 该轮环境返回的 reward；环境未提供时为 `null` |
| `done` | 该轮结束后任务是否终止 |
| `primary_score` | 该轨迹最终的 benchmark 单任务分数 |
| `finish_reason` | AgentBench 返回的终止状态 |

动作字段按 benchmark 记录：

| Benchmark | `action_name` | `action_arguments` |
|---|---|---|
| ALFWorld | `take_action` | `{"action": "原始动作字符串"}` |
| WebShop 搜索 | `search_action` | `{"keywords": "原始搜索词"}` |
| WebShop 点击 | `click_action` | `{"value": "原始点击值"}` |

说明性样例：

```json
{
  "sample_id": "123",
  "task_goal": "Find a red cotton shirt under the target price.",
  "steps": [
    {
      "step_index": 0,
      "observation": "当前 WebShop 页面文本",
      "action_name": "search_action",
      "action_arguments": {
        "keywords": "red cotton shirt"
      },
      "next_observation": "搜索结果页面文本",
      "reward": 0.0,
      "done": false
    },
    {
      "step_index": 1,
      "observation": "搜索结果页面文本",
      "action_name": "click_action",
      "action_arguments": {
        "value": "商品标题"
      },
      "next_observation": "商品详情页面文本",
      "reward": 0.0,
      "done": false
    }
  ],
  "primary_score": 0.75,
  "finish_reason": "completed"
}
```

该样例用于说明字段语义。实际持久化优先复用仓库现有模型和 AgentBench 原始输出，由 adapter 完成读取转换。

共享轨迹池同时保留成功和失败轨迹，供 outcome edge 与对比式分解使用。

# 第二阶段：提前压死统一指标和结果脚本

## 5. 公共 Agent 执行器

实现：

```text
experiments/common/agent/
├── evaluator.py
├── prompt.py
├── session.py
└── result_writer.py
```

统一接口：

```python
class AgentEvaluator:
    async def run_episode(
        self,
        *,
        task,
        method_name: str,
        skill_context: str | None,
        repeat_id: int,
    ) -> EpisodeResult:
        ...
```

所有方法共享：

- executor model
- temperature
- thinking 配置
- system prompt
- benchmark action tools
- max steps
- timeout
- retry policy
- conversation history 格式
- observation 格式
- invalid action 处理
- termination rule

主实验固定：

```yaml
thinking: false
```

每种方法只通过 `skill_context` 注入自己的技能内容。

---

## 6. 公共技能注入接口

公共 Agent Prompt 预留统一位置：

```text
# Retrieved Experience

{method_native_skill_context}
```

各方法保留自己的原生技能格式：

- Flat Skill Summary 注入扁平 skill。
- SkillX 注入官方 retrieval 返回的内容。
- Trace2Tower 注入 High skill 及其 child Mid skill。
- No Skill 省略整个 `Retrieved Experience` 区块。

---

## 7. 统一评估口径

### 7.1 主指标

ALFWorld：

```text
episode_primary_score = 0.0 or 1.0
success_rate = 成功任务数 / 有正式结果的任务数
```

WebShop：

```text
episode_primary_score = final reward
mean_reward = 所有正式结果的 episode_primary_score 平均值
```

最终论文表只接受 `official_result_coverage = 100%` 的 run。基础设施错误先按同一任务 ID 重跑，直到每个 manifest 任务都获得正式 benchmark 结果。

### 7.2 执行效率指标

两个 benchmark 统一记录：

- `average_steps`
- `median_steps`
- `invalid_action_rate`
- `average_input_tokens`
- `average_output_tokens`
- `average_billable_tokens`
- `average_latency_ms`
- `total_input_tokens`
- `total_output_tokens`
- `total_billable_tokens`
- `total_latency_ms`

统一定义：

```text
steps = AgentBench 主交互循环实际执行的轮数
```

该口径与 ALFWorld 的 `max_step` 和 WebShop 的 `max_rounds` 一致。没有形成合法工具调用的轮次仍计入 `steps`，并同时计入 `invalid_actions`。

```text
invalid_action_rate
  = 所有任务 invalid_actions 之和
    / 所有任务 steps 之和
```

### 7.3 构建成本

每种技能方法单独记录：

- `construction_llm_calls`
- `construction_input_tokens`
- `construction_output_tokens`
- `construction_billable_tokens`
- `construction_latency_ms`
- `embedding_calls`
- `embedding_input_count`
- `final_skill_count`
- `final_mid_skill_count`
- `final_high_skill_count`

构建成本与测试 episode 的执行成本分开汇总。

---

## 8. 单任务结果格式

每个 episode 写入一行 JSONL。该格式是聚合脚本直接消费的固定协议。

### 8.1 样例

```json
{
  "run_id": "2026-07-13-webshop-trace2tower-static",
  "benchmark": "webshop",
  "split": "test",
  "method": "trace2tower_static",
  "sample_id": "123",
  "repeat_id": 0,
  "shard_id": 3,
  "primary_score": 0.75,
  "success": null,
  "steps": 12,
  "invalid_actions": 0,
  "finish_reason": "completed",
  "input_tokens": 1000,
  "output_tokens": 120,
  "billable_tokens": 1120,
  "latency_ms": 5000,
  "skill_ids": [
    "high_003",
    "mid_014"
  ],
  "skill_context_chars": 2400,
  "error": null
}
```

### 8.2 字段定义

| 字段 | 类型 | 统一口径 |
|---|---|---|
| `run_id` | `string` | 一次完整实验运行的唯一 ID；同一 run 的十个 shard 使用同一值 |
| `benchmark` | `string` | `alfworld` 或 `webshop` |
| `split` | `string` | `train` 或 `test` |
| `method` | `string` | 当前运行配置的唯一名称，见下方方法名称表 |
| `sample_id` | `string` | manifest 中的稳定任务 ID，不同方法对同一任务使用相同值 |
| `repeat_id` | `integer` | 同一任务的重复编号；没有重复时为 `0` |
| `shard_id` | `integer` | 分片编号，范围为 `0` 到 `9` |
| `primary_score` | `number \| null` | ALFWorld 为 `0.0/1.0`；WebShop 为 final reward；尚未得到正式环境结果时为 `null` |
| `success` | `boolean \| null` | ALFWorld 为成功与否；WebShop 为 `null` |
| `steps` | `integer` | AgentBench 主交互循环实际执行的轮数，包括没有产生合法动作的轮次 |
| `invalid_actions` | `integer` | 没有产生合法 benchmark 工具调用的轮数，范围为 `0` 到 `steps` |
| `finish_reason` | `string` | AgentBench 终止状态，见下方状态表 |
| `input_tokens` | `integer \| null` | 本 episode 所有执行 LLM 调用的输入 Token 总数 |
| `output_tokens` | `integer \| null` | 本 episode 所有执行 LLM 调用的输出 Token 总数 |
| `billable_tokens` | `integer \| null` | 公共 LLM runtime 返回的本 episode 计费 Token；provider 不提供时为 `null` |
| `latency_ms` | `integer` | 从 episode 开始到结束的墙钟时间，单位毫秒，包含检索、LLM 请求和环境交互 |
| `skill_ids` | `array[string]` | 最终实际写入 Agent Prompt 的技能 ID，按注入顺序记录；No Skill 为 `[]` |
| `skill_context_chars` | `integer` | 最终注入技能文本的字符数；No Skill 为 `0` |
| `error` | `string \| null` | 正式完成时为 `null`；基础设施异常时写简短错误摘要，完整堆栈写入 shard 日志 |

方法名称固定为：

| 实验配置 | `method` |
|---|---|
| No Skill | `no_skill` |
| Semantic-Only Clustering | `semantic_clustering` |
| Flat Skill Summary | `flat_skill_summary` |
| SkillX | `skillx` |
| Trace2Tower-Static | `trace2tower_static` |
| Trace2Tower-Full | `trace2tower_full` |
| No Transition Edge | `trace2tower_no_transition` |
| No Outcome Edge | `trace2tower_no_outcome` |
| No Contrastive Decomposition | `trace2tower_no_contrastive` |

`finish_reason` 直接使用 AgentBench 的终止语义：

| 值 | 含义 |
|---|---|
| `completed` | 环境正常结束并返回正式结果，包括成功和正常失败 |
| `task_limit_reached` | 达到 AgentBench 最大交互轮次 |
| `agent_validation_failed` | 当前轮未生成可执行工具调用 |
| `agent_invalid_action` | 工具调用或动作无法解析 |
| `task_error` | task、环境或 provider 执行发生异常 |
| `cancelled` | 任务被外部取消 |

字段空值口径：

- `0` 表示已测量且结果确实为零。
- `null` 表示该字段不适用，或尚未取得可靠值。
- `[]` 表示列表字段存在但本次没有元素。

### 8.3 写入和恢复

输出路径：

```text
artifacts/runs/{run_id}/episodes/shard-XX.jsonl
```

每个 episode 完成后原子追加一行。恢复时使用以下联合键识别已完成任务：

```text
benchmark
split
method
sample_id
repeat_id
```

合并脚本检查：

- manifest 中的每个任务都有且只有一条记录；
- 不存在重复联合键；
- `primary_score` 全部为正式 benchmark 结果；
- shard 之间没有遗漏或重复。

基础设施异常可以保留 `error` 记录用于诊断；正式聚合前按同一任务 ID 重跑并替换为正式结果。

---

## 9. 聚合与统计脚本

实现：

```text
scripts/evaluate_results.py
```

输入：

```text
artifacts/runs/{run_id}/episodes/
```

输出：

```text
artifacts/runs/{run_id}/
├── aggregate.json
├── aggregate.md
├── pairwise.json
└── failures.jsonl
```

聚合步骤：

1. 以 manifest 检查样本完整性。
2. 验证每个联合键只有一条记录。
3. 计算 `official_result_coverage`；正式结果要求为 `100%`。
4. ALFWorld 计算 success rate。
5. WebShop 计算 mean reward。
6. 汇总 steps、invalid actions、Token、延迟和构建成本。
7. 按相同 `sample_id` 和 `repeat_id` 做方法间配对。
8. 使用固定随机种子的 paired bootstrap，次数固定为 `10000`。
9. 报告主指标均值差和 95% confidence interval。

# 第三阶段：接入 SkillX 并抽取公共 LLM 执行器

## 10. 公共 LLM Runtime

从 SkillX 的 LLM client 中抽取通用运行能力，SkillX 上层策略保持官方实现。

实现：

```text
experiments/common/llm/
├── client.py
├── models.py
├── retry.py
└── usage.py
```

公共异步接口：

```python
class CommonLLMClient:
    async def ainvoke(
        self,
        messages,
        *,
        regex_pattern=None,
        regex_extractor=None,
        temperature=None,
        max_tokens=None,
        timeout=None,
    ) -> str:
        ...
```

必须支持：

- system/user/assistant/tool messages
- async invocation
- regex_pattern
- regex_extractor
- retry
- timeout
- token usage
- latency
- thinking=false

### 10.1 SkillX LLM Adapter

实现：

```text
experiments/baselines/skillx/llm_adapter.py
```

只做接口转发：

```python
class SkillXLLMAdapter:
    async def ainvoke(
        self,
        messages,
        regex_pattern=None,
        regex_extractor=None,
        **kwargs,
    ) -> str:
        ...
```

适配器原样转发：

- messages
- system prompt
- user prompt
- regex extractor
- output parser

---

## 11. SkillX 上游版本保护

SkillX 官方目录按 source-lock 中记录的 commit 使用。

实现：

```text
scripts/check_skillx_upstream.py
```

检查：

1. 当前 commit 与 source-lock 一致。
2. 官方 Prompt 文件 hash 与该 commit 一致。
3. extraction、clustering、merger、filter、retrieval 文件 hash 与该 commit 一致。
4. benchmark 适配代码位于主工程 adapter 目录。

主实验配置：

```yaml
skill_type: hybrid
plan_strategy: shortest
atomic_mode: omission
enable_clustering: true
filter_timing: pre_merge
num_epochs: 1
expansion_strategy: null
```

SkillX 的 extraction、merge、filter 和 retrieval 流程按官方仓库执行。

## 12. SkillX Benchmark Adapter

实现：

```text
experiments/baselines/skillx/
├── trajectory_adapter.py
├── tool_schema_adapter.py
├── builder.py
├── retriever.py
└── formatter.py
```

adapter 将共享轨迹池转换为 SkillX 官方 pipeline 所需的标准 messages、tool calls、tool schemas 和 outcome。

### 12.1 ALFWorld Adapter

环境工具 schema：

```text
take_action(action: string)
```

轨迹中的每个实际动作转换为：

```json
{
  "name": "take_action",
  "arguments": {
    "action": "原始 ALFWorld 动作字符串"
  }
}
```

Trace2Tower 使用的 primitive action 解析结果可以作为轨迹文本特征，但 SkillX adapter 保留 AgentBench 的真实工具接口。

### 12.2 WebShop Adapter

环境工具 schema 只有：

```text
search_action(keywords: string)
click_action(value: string)
```

转换规则：

```json
{
  "name": "search_action",
  "arguments": {
    "keywords": "原始搜索词"
  }
}
```

```json
{
  "name": "click_action",
  "arguments": {
    "value": "原始点击值"
  }
}
```

adapter 从 AgentBench history 中读取当前 observation、实际 action、reward 和 done，并按 SkillX 官方输入格式组装。

### 12.3 兼容层

优先通过 messages、tool calls 和 tool schemas 的格式转换完成适配。

官方 parser 与 AgentBench 输出存在格式差异时，在 adapter 目录实现兼容 parser，并为以下内容编写单元测试：

- 原始 AgentBench 记录输入
- 转换后的 SkillX 输入
- SkillX 输出字段
- 异常输入处理

# 第四阶段：实现 Trace2Tower 与可插拔消融

## 13. 模块结构

实现：

```text
experiments/methods/trace2tower/
├── config.py
├── models.py
├── action_parser.py
├── transition_encoder.py
├── segmentation.py
├── graph.py
├── spectral.py
├── clustering.py
├── low_skills.py
├── mid_skills.py
├── high_paths.py
├── renderer.py
├── retrieval.py
├── refinement.py
├── lineage.py
└── builder.py
```

所有消融通过同一配置模型中的开关实现，共享同一套算法代码。

固定配置模型：

```python
@dataclass(frozen=True, slots=True)
class Trace2TowerConfig:
    semantic_only: bool = False
    use_transition_edge: bool = True
    use_outcome_edge: bool = True
    use_contrastive_decomposition: bool = True
    enable_refinement: bool = False

    target_segment_length: int = 3
    failure_penalty: float = 1.0
    max_high_path_length: int = 4
    high_min_support_ratio: float = 0.02
    high_top_k: int = 1
    direct_mid_top_k: int = 2
```

这些值在训练阶段冻结，并写入每次 run 的 resolved config。

---

## 14. 数据对象

必须区分以下对象：

1. PrimitiveAction
2. StepTransition
3. SegmentInstance
4. MidCluster
5. HighPath
6. SkillCard
7. TowerSnapshot

### 14.1 StepTransition

```python
@dataclass(frozen=True, slots=True)
class StepTransition:
    transition_id: str
    trajectory_id: str
    step_index: int
    goal: str
    observation_before: str
    raw_action: str
    primitive_action: str
    observation_after: str
    trajectory_score: float
```

### 14.2 SegmentInstance

```python
@dataclass(frozen=True, slots=True)
class SegmentInstance:
    segment_id: str
    trajectory_id: str
    start_step: int
    end_step: int
    transition_ids: tuple[str, ...]
    embedding: tuple[float, ...]
    trajectory_score: float
```

### 14.3 MidCluster

```python
@dataclass(frozen=True, slots=True)
class MidCluster:
    cluster_id: str
    member_segment_ids: tuple[str, ...]
    centroid: tuple[float, ...]
```

### 14.4 HighPath

```python
@dataclass(frozen=True, slots=True)
class HighPath:
    path_id: str
    ordered_mid_ids: tuple[str, ...]
    positive_support: float
    negative_support: float
    contrastive_score: float
```

---

## 15. 原始动作解析

### 15.1 ALFWorld

从 `take_action.action` 的原始字符串中确定性解析 primitive action：

```text
GOTO
PICK
PUT
OPEN
CLOSE
TOGGLE
HEAT
CLEAN
COOL
SLICE
INVENTORY
EXAMINE
LOOK
INVALID
```

解析结果用于 Trace2Tower 的 transition 表示和 Low skill grounding；原始动作字符串同时保留。

### 15.2 WebShop

根据实际工具调用确定性解析：

```text
search_action(keywords) -> SEARCH(keywords)
click_action(value)     -> CLICK(value)
```

工具名或参数无法解析时记为：

```text
INVALID
```

`keywords` 和 `value` 保留 AgentBench 记录中的原始字符串。

## 16. Transition 文本与 embedding

每一步生成稳定文本签名：

```text
Goal:
{goal}

Observation Before:
{observation_before}

Action:
{primitive_action} | {raw_action}

Observation After:
{observation_after}
```

使用冻结的同一 embedding encoder。

要求：

- encoder revision 固定。
- embedding 按内容 hash 缓存。
- Semantic-Only 与 Trace2Tower 使用相同 embedding。
- SkillX 可保留官方 embedding 位置和文本字段，但底层 provider 可走公共客户端。

---

## 17. 确定性轨迹切分

轨迹切分使用确定性 change-point dynamic programming。

对每条轨迹的 transition embedding 序列执行 change-point dynamic programming。

目标函数用代码表达：

```text
objective(partition)
  = sum(within_segment_cost(segment))
  + segmentation_penalty * number_of_segments
```

固定：

```text
target_segment_length = 3
max_segment_length = 6
```

`segmentation_penalty` 通过训练轨迹自动校准，使训练集 segment length 的中位数尽量接近 3。

校准只执行一次，写入：

```text
artifacts/trace2tower/segmentation-calibration.json
```

Semantic-Only 必须复用完全相同的 segment 边界。

---

## 18. 三类边权信息

对 segment 节点计算三类非负分数。

### 18.1 Semantic Similarity

```text
S_uv = max(0, cosine(h_u, h_v))
```

其中 `h_u` 是 segment embedding。

### 18.2 Transition Similarity

Transition 不直接表示两个相邻阶段应该聚在一起。

为每个 segment 构造上下文：

```text
context_u = concat(
  previous_segment_embedding_or_zero,
  next_segment_embedding_or_zero
)
```

然后：

```text
T_uv = max(0, cosine(context_u, context_v))
```

该项衡量两个 segment 是否处于相似的前后结构位置。

### 18.3 Outcome Consistency

先计算 segment 的局部成功倾向。

ALFWorld 的轨迹分数已经位于 `[0, 1]`。

WebShop reward 必须裁剪到 `[0, 1]`。

使用 semantic kNN 对轨迹分数平滑：

```text
rho_u
  = weighted_mean(
      own_trajectory_score,
      semantic_neighbor_trajectory_scores
    )
```

平滑先验固定为 1。

然后：

```text
O_uv = 1 - abs(rho_u - rho_v)
```

结果裁剪到 `[0, 1]`。

---

## 19. 稀疏图构造

候选边来自：

- semantic kNN
- transition kNN

取两者并集。

邻居数固定自动计算：

```text
k = clip(ceil(log2(number_of_segments)), 10, 30)
```

对称化候选 mask：

```text
M_uv = 1
```

当任一方向存在候选边。

完整基础边权使用等权平均，不增加 alpha、beta、gamma 超参数：

```text
B_uv = M_uv * mean(S_uv, T_uv, O_uv)
```

消融时只对仍启用的边权求平均：

No Transition Edge：

```text
B_uv = M_uv * mean(S_uv, O_uv)
```

No Outcome Edge：

```text
B_uv = M_uv * mean(S_uv, T_uv)
```

当 `semantic_only=true` 时完全跳过图构造和谱分解。

---

## 20. 对比式分解

完整方法构造：

```text
W_positive_uv = B_uv * sqrt(rho_u * rho_v)

W_negative_uv
  = B_uv * sqrt((1 - rho_u) * (1 - rho_v))

A_uv
  = W_positive_uv
  - failure_penalty * W_negative_uv
```

固定：

```text
failure_penalty = 1.0
```

对称化：

```text
A = 0.5 * (A + transpose(A))
```

signed degree：

```text
D_abs_uu = sum_v(abs(A_uv))
```

signed normalized Laplacian：

```text
L_signed
  = I
  - inverse_sqrt(D_abs)
    @ A
    @ inverse_sqrt(D_abs)
```

数值要求：

- 零 degree 节点单独处理。
- 对零 degree 节点使用安全逆平方根，输出矩阵保持有限值。
- 固定 eigensolver seed。
- 聚类只使用对特征向量整体符号翻转不敏感的行表示。

### 20.1 No Contrastive Decomposition

保持：

- segment
- S
- T
- O
- B
- spectral clustering

只替换 signed adjacency：

```text
A = B
```

该配置保留 Outcome Edge，只将 signed adjacency 替换为普通 adjacency。

这保证两个消融回答不同问题：

- No Outcome Edge：不在边权中使用结果一致性。
- No Contrastive Decomposition：仍使用结果一致性，但不做正负图相减。

---

## 21. 谱表示与 Mid 聚类

聚类数量通过 eigengap 自动决定。

流程：

```text
1. 计算最小特征值对应的特征向量。
2. 排除常数或数值退化方向。
3. 在允许范围内选择最大 eigengap 对应的 K。
4. 对前 K 个有效特征向量按行归一化。
5. 使用固定 random_state 的 K-Means。
```

`K` 必须写入构建报告。

### 21.1 Semantic-Only Clustering

该 baseline 使用：

```text
segment embedding h_u
```

直接做 K-Means。

为隔离图结构影响：

- 使用与同一运行 Full Trace2Tower 相同的 segment。
- 使用与 Full Trace2Tower 相同的 K。
- 使用相同 random_state。
- 后续 Mid renderer、High path mining 和 retrieval 完全相同。

Semantic-Only 直接复用同一运行中 Full Trace2Tower 得到的 K。

---

## 22. Low、Mid、High 技能

### 22.1 Low Skill

Low skill 是官方 primitive action template。

Low skill：

- 不由 LLM 抽取。
- 不独立检索。
- 只用于 Mid card 的动作 grounding 和合法性检查。

### 22.2 Mid Skill

每个 Mid skill 对应一个固定 MidCluster。

renderer 输入字段：

| 字段 | 含义 |
|---|---|
| `cluster_id` | Mid cluster 的稳定 ID |
| `member_segment_ids` | 该 cluster 的全部 segment ID |
| `member_segment_ids` | 该 cluster 的全部 segment ID |
| `segment_evidence` | 原始 action、observation 片段和轨迹分数 |
| `support_count` | 成员 segment 数量 |
| `primitive_action_distribution` | cluster 内 primitive action 计数 |

LLM 输出固定 schema：

```json
{
  "skill_id": "mid_001",
  "name": "简洁名称",
  "description": "适用条件",
  "procedure": [
    "步骤 1",
    "步骤 2"
  ],
  "constraints": [
    "约束"
  ],
  "grounding_actions": [
    "PICK"
  ]
}
```

`skill_id` 和 `member_segment_ids` 由构建器写入并随 card 一起保存。renderer 只生成 `name`、`description`、`procedure`、`constraints` 和 `grounding_actions`。

### 22.3 High Skill

将每条轨迹映射为有序 Mid ID 序列，并压缩连续重复：

```text
A -> A -> B -> B -> C
变为
A -> B -> C
```

枚举长度 2 到 4 的连续路径。

固定配置：

```text
max_high_path_length = 4
min_high_support_ratio = 0.02
```

路径至少包含两个不同 Mid ID。

路径分数：

```text
contrastive_path_score
  = positive_support
    * log(
        (positive_support + epsilon)
        /
        (negative_support + epsilon)
      )
```

High renderer 输入字段：

| 字段 | 含义 |
|---|---|
| `path_id` | High path 的稳定 ID |
| `ordered_mid_ids` | 已确定的 Mid ID 顺序 |
| `child_mid_cards` | 对应 Mid skill card |
| `positive_support` | 成功轨迹中的支持量 |
| `negative_support` | 失败轨迹中的支持量 |
| `contrastive_path_score` | 上述路径分数 |
| `supporting_trajectory_ids` | 支持该路径的训练轨迹 ID |

renderer 生成 High skill 的名称、适用条件和执行说明；`ordered_mid_ids` 由 HighPath 原样保存。

## 23. Trace2Tower 检索

固定流程：

```text
1. 使用 task goal 检索 Top-1 High。
2. 展开该 High 的全部 child Mid。
3. 使用 task goal + initial observation 直接检索 Top-2 Mid。
4. 对 Mid skill_id 去重。
5. 注入 High card 和最终 Mid cards。
```

固定：

```yaml
high_top_k: 1
direct_mid_top_k: 2
```

没有 High 命中时：

```text
只注入 Top-2 direct Mid
```

Low 只用于 grounding；检索对象为 High 和 Mid。

检索结果按固定 Top-k 完整注入，并记录实际注入 Token。

必须记录实际注入 token。

---

## 24. Flat Skill Summary Baseline

实现：

```text
experiments/baselines/flat_skill_summary/
├── prompt.py
├── builder.py
├── models.py
├── retrieval.py
└── formatter.py
```

构建流程：

```text
1. 读取共享训练轨迹池中的成功轨迹。
2. 每条轨迹调用一次 LLM。
3. 每条轨迹生成一条扁平 skill。
4. 对 flat skill 建立语义索引。
5. 测试时检索 Top-3。
```

输出 schema：

```json
{
  "skill_id": "flat_001",
  "name": "技能名称",
  "description": "适用条件",
  "procedure": [
    "步骤"
  ],
  "constraints": [
    "约束"
  ]
}
```

固定配置：

```yaml
flat_top_k: 3
```

Prompt 只基于输入轨迹总结可复用步骤和约束，生成单层 skill。Prompt 文件写入代码仓库，并将 hash 写入构建报告。

## 25. Deployment Refinement

### 25.1 版本定义

Trace2Tower-Static：

```text
共享 No-Skill 训练轨迹
-> 构建 Tower v0
-> 冻结
-> Test
```

Trace2Tower-Full：

```text
共享 No-Skill 训练轨迹
-> 构建 Tower v0
-> 使用 Tower v0 在训练 manifest 上再 rollout 一轮
-> 累积初始轨迹和技能增强轨迹
-> 计算技能效用
-> 重建候选结构
-> 应用一次 refinement
-> 得到 Tower v1
-> 冻结
-> Test
```

固定：

```yaml
refinement_rounds: 1
```

v0 到 v1 的全部计算使用训练阶段数据；测试集只读取冻结后的 v1。

---

## 26. 四因素效用

对每个实际注入的技能统计四个因素：

1. Success Rate
2. Reward Gain
3. Step Saving
4. Cost Saving

所有因素都使用训练阶段的 skill-conditioned rollout 与相同任务的 No-Skill 训练结果计算。

### 26.1 Success Rate

ALFWorld：

```text
success_rate
  = 成功 exposure 数
    / 总 exposure 数
```

WebShop 没有额外定义二元成功阈值，因此使用：

```text
success_rate
  = 该技能 exposure 对应 episode primary_score 的平均值
```

该字段在两个 benchmark 中都表示越大越好，但计算口径分别遵循各自的官方主分数。

### 26.2 Reward Gain

按相同 `benchmark`、`sample_id` 和 `repeat_id` 配对：

```text
reward_gain
  = skill_episode.primary_score
    - no_skill_episode.primary_score
```

一个 episode 注入多个技能时，该 episode 的 gain 计入每个实际注入技能的 exposure 统计。

### 26.3 Step Saving

本文中的 `steps` 与单任务结果格式一致，表示 AgentBench 主交互循环实际执行的轮数。

```text
raw_step_saving
  = (
      no_skill_episode.steps
      - skill_episode.steps
    )
    / max(no_skill_episode.steps, 1)
```

当 skill episode 的 `primary_score` 低于配对 No-Skill episode 时：

```text
step_saving = 0
```

否则：

```text
step_saving = raw_step_saving
```

### 26.4 Cost Saving

使用单任务结果中的 `billable_tokens`：

```text
cost_saving
  = (
      no_skill_episode.billable_tokens
      - skill_episode.billable_tokens
    )
    / max(no_skill_episode.billable_tokens, 1)
```

只有配对双方的 `billable_tokens` 都有可靠值时才计算；缺失 exposure 不进入该因素的均值，并在 refinement 报告中记录 coverage。

技能构建成本单独报告。

---

## 27. 归一化和等权组合

每个 benchmark、每个 skill level、每轮 refinement 分别归一化。

对同一因素使用 percentile rank：

```text
normalized_value
  = average_rank(value) / number_of_skills
```

规则：

- 按“值越大越好”升序排名。
- 并列值使用 average rank。
- 只有一个技能时取 `0.5`。
- 没有 exposure 的技能取 `0`。

最终效用：

```text
utility
  = (
      normalized_success_rate
      + normalized_reward_gain
      + normalized_step_saving
      + normalized_cost_saving
    )
    / 4
```

四项固定等权，归一化结果和最终 utility 写入 `runtime_states.json`。

## 28. Refinement 的确定性实现

### 28.1 重建候选塔

使用：

```text
initial No-Skill trajectories
+
Tower v0 skill-conditioned train trajectories
```

重新运行：

- transition encoding
- segmentation
- graph construction
- contrastive decomposition
- Mid clustering
- High path mining

得到候选塔。

### 28.2 新旧 Mid lineage

历史 segment ID 保持稳定。

根据共享历史 segment 构造旧 cluster 与新 cluster 的 overlap graph，并识别：

```text
one old -> one new      continuation
one old -> many new     split
many old -> one new     merge
none -> one new         new Mid
one old -> none         disappeared Mid
```

主要判据：

| 指标 | 含义 |
|---|---|
| `shared_member_count` | 新旧 cluster 共享的历史 segment 数 |
| `old_retention` | 旧 cluster 成员进入该新 cluster 的比例 |
| `new_historical_purity` | 新 cluster 的历史成员中来自该旧 cluster 的比例 |

Centroid similarity 只用于 overlap 相同时的 tie-break 和漂移诊断。

完全由新 segment 构成的 cluster 生成新的 Mid ID。

### 28.3 一轮更新预算

每轮只允许：

- Top-1 Split proposal
- Top-1 Merge proposal
- Top-1 Promote proposal
- Top-1 Downweight or Prune proposal

若某类没有合法 proposal，则跳过。

排序规则：

Split：

```text
优先选择 source utility 最低的合法 split
```

Merge：

```text
优先选择历史成员 overlap 最大的合法 merge
并以 source utility 平均值作为 tie-break
```

Promote：

```text
优先选择 contrastive_path_score 最高的新 High path
并以 child Mid utility 平均值作为 tie-break
```

Downweight 或 Prune：

```text
选择 utility 最低的 active skill
```

### 28.4 Downweight 与 Prune

第一轮 refinement 只执行 Downweight，不物理 Prune。

```text
status = downweighted
```

检索 tie-break：

```text
active 优先于 downweighted
```

检索使用原语义分数，并在同分时按 `active` 优先于 `downweighted`。

Prune 代码可以保留接口，但主实验配置关闭：

```yaml
enable_physical_prune: false
```

### 28.5 渲染更新

只重新渲染：

- Split 产生的新 Mid
- Merge 产生的新 Mid
- Promote 产生的新 High
- child path 变化的 High

未改变的 card 直接复用。

### 28.6 Tower 快照

输出：

```text
artifacts/towers/{benchmark}/
├── v0/
└── v1/
```

每个版本包含：

```text
tower.json
low_skills.json
mid_skills.json
high_skills.json
runtime_states.json
lineage.json
retrieval_vectors.npz
build_report.json
```

快照发布后只读。

---

## 29. 消融开关映射

### 29.1 Full Trace2Tower-Static

```yaml
semantic_only: false
use_transition_edge: true
use_outcome_edge: true
use_contrastive_decomposition: true
enable_refinement: false
```

### 29.2 Full Trace2Tower-Full

```yaml
semantic_only: false
use_transition_edge: true
use_outcome_edge: true
use_contrastive_decomposition: true
enable_refinement: true
```

### 29.3 Semantic-Only Clustering

```yaml
semantic_only: true
use_transition_edge: false
use_outcome_edge: false
use_contrastive_decomposition: false
enable_refinement: false
```

### 29.4 No Transition Edge

```yaml
semantic_only: false
use_transition_edge: false
use_outcome_edge: true
use_contrastive_decomposition: true
enable_refinement: false
```

### 29.5 No Outcome Edge

```yaml
semantic_only: false
use_transition_edge: true
use_outcome_edge: false
use_contrastive_decomposition: true
enable_refinement: false
```

No Outcome Edge 中，contrastive decomposition 仍可使用轨迹结果构造正负图。

该消融只移除基础边权中的 `O_uv`。

### 29.6 No Contrastive Decomposition

```yaml
semantic_only: false
use_transition_edge: true
use_outcome_edge: true
use_contrastive_decomposition: false
enable_refinement: false
```

使用：

```text
A = B
```

---

# 第五阶段：实现可复用实验脚本和十分片并发

## 30. 脚本目录

实现：

```text
scripts/experiments/
├── prepare_manifests.py
├── rollout_no_skill_train.py
├── build_flat_skills.py
├── build_skillx.py
├── build_trace2tower.py
├── refine_trace2tower.py
├── run_test.py
├── merge_shards.py
├── evaluate_results.py
├── check_completeness.py
└── run_matrix.py
```

每个脚本必须：

- 支持配置文件。
- 支持 `--benchmark`。
- 支持 `--method`。
- 支持 `--shard-id`。
- 支持 `--num-shards`。
- 支持断点恢复。
- 支持 dry-run。
- 运行前打印完整冻结配置。
- 运行后写入 run metadata。

---

## 31. 十分片规则

固定：

```yaml
num_shards: 10
```

分片规则：

```text
1. 按 manifest 中 sample_id、repeat_id 排序。
2. 第 i 个样本分配到 shard = i mod 10。
3. shard ID 固定为 00 到 09。
```

分片使用排序后的序号取模，保证不同进程和不同机器得到相同结果。

每个 shard：

- 独立输入列表。
- 独立 JSONL 输出。
- 独立错误日志。
- 独立 checkpoint。
- 可以单独重跑。

启动器同时启动十个 shard：

```text
scripts/experiments/run_matrix.py
```

必须使用全局并发 semaphore，避免 provider rate limit。

全局并发上限写入实验配置，并由公共 runner 的 semaphore 统一控制。

AgentBench 原始 task concurrency 只作为参考；公共 runner 的全局并发是唯一生效的 API 并发限制。

---

## 32. 实验配置文件

建立：

```text
configs/experiments/
├── common.yaml
├── alfworld.yaml
├── webshop.yaml
├── no_skill.yaml
├── semantic_only.yaml
├── flat_skill_summary.yaml
├── skillx.yaml
├── trace2tower_static.yaml
├── trace2tower_full.yaml
├── ablation_no_transition.yaml
├── ablation_no_outcome.yaml
└── ablation_no_contrastive.yaml
```

所有方法参数都写入配置文件和 `resolved-config.yaml`。

每个 run 将所有配置合并后写入：

```text
artifacts/runs/{run_id}/resolved-config.yaml
```

---

## 33. 最终执行顺序

严格按以下顺序运行。

### Step 1：锁定来源和环境

```text
1. 记录主仓库 commit。
2. 记录 AgentBench commit。
3. 记录 SkillX commit。
4. 记录 Python、依赖和模型配置。
5. 生成 source-lock.json。
```

### Step 2：生成 AgentBench Manifest

```text
1. 检查 AgentBench 配置。
2. 生成 ALFWorld train/test manifest。
3. 生成 WebShop train/test manifest。
4. 运行 manifest 完整性检查。
```

### Step 3：实现并验证公共执行器和结果管线

```text
1. 公共 LLM client。
2. 公共 AgentEvaluator。
3. 公共 trajectory writer。
4. 公共 episode result writer。
5. 公共 aggregation script。
6. 每个 benchmark 运行两个 smoke-test task。
```

公共结果管线通过 smoke test 后再进入技能算法实现。

### Step 4：生成共享 No-Skill 训练轨迹池

```text
1. ALFWorld 十分片 rollout。
2. WebShop 十分片 rollout。
3. 合并和检查轨迹。
4. 确认成功与失败轨迹均被保留。
```

### Step 5：接入 SkillX

```text
1. 实现 SkillXLLMAdapter。
2. 实现 ALFWorld trajectory/tool adapter。
3. 实现 WebShop trajectory/tool adapter。
4. 保持官方 Prompt 不变。
5. 构建 SkillX skill library。
6. 运行小规模检索和执行 smoke test。
```

### Step 6：实现 Flat Skill Summary

```text
1. 固定 Prompt 和 schema。
2. 从共享成功轨迹生成 flat skill。
3. 建立语义索引。
4. 验证 Top-3 检索。
```

### Step 7：实现 Trace2Tower-Static

```text
1. action parser。
2. transition encoding。
3. deterministic segmentation。
4. S/T/O 边权。
5. sparse graph。
6. contrastive signed decomposition。
7. spectral clustering。
8. Low/Mid/High construction。
9. Mid/High renderer。
10. hierarchical retrieval。
11. 构建 Tower v0。
```

### Step 8：实现开关和消融

按顺序：

```text
1. Semantic-Only
2. No Transition Edge
3. No Outcome Edge
4. No Contrastive Decomposition
```

每个消融必须复用 Full 的其余组件。

### Step 9：实现 Trace2Tower-Full

```text
1. 使用 Tower v0 在 train manifest 再 rollout 一轮。
2. 计算四因素 utility。
3. 重建候选塔。
4. 生成 lineage diff。
5. 每类应用 Top-1 proposal。
6. 只重渲染变化技能。
7. 保存 Tower v1。
```

### Step 10：按固定顺序跑测试集

两个 benchmark 均按以下方法顺序：

```text
1. No Skill
2. Semantic-Only Clustering
3. Flat Skill Summary
4. SkillX
5. Trace2Tower-Static
6. Trace2Tower-Full
7. No Transition Edge
8. No Outcome Edge
9. No Contrastive Decomposition
```

方法顺序只用于调度，不影响任务 manifest。

### Step 11：聚合和检查

```text
1. 合并十个 shard。
2. 检查 manifest 覆盖率。
3. 生成 aggregate.json。
4. 生成 aggregate.md。
5. 生成 pairwise bootstrap。
6. 生成 failures.jsonl。
7. 生成成本报告。
```

---

## 34. 测试要求

### 34.1 单元测试

至少覆盖：

- ALFWorld action parser
- WebShop action parser
- transition serialization
- segmentation 边界
- segment ID 稳定性
- S/T/O 范围
- sparse mask 对称性
- signed adjacency 对称性
- Laplacian 无 NaN/Inf
- eigengap K 稳定性
- Semantic-Only 与 Full 使用相同 K
- High path 连续重复压缩
- High path 最大长度
- retrieval 去重
- percentile rank utility
- lineage continuation/split/merge/new/disappeared
- 十分片无遗漏、无重复
- result writer 断点恢复
- SkillX Prompt 文件 hash 未变化

### 34.2 集成测试

每个 benchmark 至少跑：

```text
2 个 train task
2 个 test task
```

覆盖：

- No Skill
- SkillX
- Trace2Tower-Static
- 一个 edge ablation

### 34.3 完整性门禁

完整实验前必须通过：

```text
pytest
ruff or repository equivalent
type checker
manifest completeness check
upstream integrity check
```

---

## 35. 最终验收标准

Codex 完成任务时必须提供：

1. 所有新增和修改文件列表。
2. AgentBench 与 SkillX commit。
3. 运行命令。
4. smoke test 结果。
5. unit test 结果。
6. 每个方法的 resolved config。
7. 一个完整的两任务端到端示例。
8. 一个十 shard dry-run 示例。
9. 说明 SkillX 哪些文件保持原样。
10. 说明所有消融如何通过开关注入。
11. 说明 test 阶段没有任何技能更新。
12. 说明没有实现 PUE、Choquet、Pareto 或 Raw Trajectory Retrieval。

---

## 36. 协议变更规则

当当前仓库事实与本文档冲突时，Codex 先输出：

```text
1. 当前代码事实
2. 与本文档冲突的具体位置
3. 最小修改方案
4. 对实验口径和可比性的影响
```

实验方法、任务划分、主指标、SkillX 官方策略、Trace2Tower 消融语义和测试集冻结规则的变更，需要人工确认后写入配置与文档。
