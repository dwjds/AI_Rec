# 面向 MOOC 学习场景的 RAG 教育资源推荐与学习规划 Agent

本项目是一个面向 MOOC 学习场景的 RAG 教育 Agent 全栈原型，以 RAG、受控工具调用、用户记忆和可观测 AgentLoop 为核心，支持教育资源推荐、知识点问答、学习路线规划、学习困难诊断、用户反馈调整和执行轨迹追踪。

项目目标不是只做一个“课程搜索器”，而是构建一个工程化 Agent 系统：

- 通过 Router 判断用户任务类型和需要的能力
- 通过 AgentState 统一管理会话状态、用户上下文、证据包和执行结果
- 通过 FunctionCallRuntime 可靠调用工具
- 通过 RAG 检索和 Evidence Builder 约束回答基于证据
- 通过 AgentLoop 完成学习路线和学习诊断的多步执行
- 通过 Trace、RuntimeGuard、Handoff 和 Evaluation 支撑调试、兜底和评估

## 项目定位

核心场景：

```text
用户提出学习需求
  -> Agent 判断任务类型
  -> 读取用户画像和学习状态
  -> 检索 MOOC 资源与知识证据
  -> 推荐资源 / 解释知识点 / 规划路线 / 诊断问题
  -> 生成自然语言回答
  -> 记录 trace、推荐日志、反馈和必要的人工兜底 case
```

当前版本已经打通本地端到端主链路：FastAPI 负责提供 Agent、资源、用户、反馈和笔记接口；前端页面通过 API 调用真实后端；CLI 用于无前端场景下调试 Agent 执行过程。仓库不包含运行数据，完整体验需要在本地构建 MOOPer 数据库和 Chroma 向量库。

## 仓库内容说明

本仓库只上传项目源码、构建脚本、配置样例和前端页面，不上传本地运行数据、测试目录或私有配置。

以下目录已在 `.gitignore` 中忽略，不会提交到 GitHub：

```text
data/                 # 原始 MOOPer 数据、SQLite 数据库、Chroma 向量库、trace、评估报告
tests/                # 本地测试用例
.env                  # 本地密钥配置
```

因此 clone 仓库后需要自行准备 MOOPer 数据集并重新构建数据库和向量库。项目提供了对应脚本：

```text
scripts/build_mooper_db.py
scripts/init_app_db.py
scripts/build_chunks.py
scripts/build_vector_index.py
```


## 总体架构

```text
用户请求
  |
  v
AgentOrchestrator
  |
  +-- MemoryService
  |     -> session memory / user memory / feedback memory / knowledge state
  |
  +-- Router
  |     -> RoutingDecision
  |        task_type
  |        needs_rag
  |        needs_user_profile
  |        needs_agent_loop
  |        needs_clarification
  |        pipeline
  |
  +-- Pipeline Dispatch
        |
        +-- Clarification
        +-- RAG QA
        +-- Recommendation Pipeline
        +-- AgentLoop: Learning Path / Diagnosis
        +-- Feedback Adjustment
        +-- Direct Chat

AgentLoop
  |
  +-- FunctionCallRuntime
  |     -> get_user_context
  |     -> search_courses
  |     -> get_course_detail
  |
  +-- Internal Components
        -> RagRetriever
        -> EvidenceBuilder
        -> LearningPathPlanner / DiagnosisPlanner
        -> ResponseGenerator

Observability & Safety
  |
  +-- AgentTraceRecorder
  +-- RuntimeGuard
  +-- HandoffService
  +-- EvaluationRunner
```

## 目录结构

```text
app/
  agent/
    orchestrator.py            # Agent 总调度器
    router.py                  # 任务路由，输出 RoutingDecision
    agent_loop.py              # 受控工具调用型 AgentLoop
    guard.py                   # 最大轮次、超时、失败分类、人工兜底触发
    state.py                   # 统一会话状态 AgentState
    trace.py                   # Agent 执行轨迹记录

  rag/
    chunker.py                 # 从 MOOPer 资源构建语义 chunk
    embedding_service.py       # embedding 生成
    vector_store.py            # Chroma 向量库
    query_rewriter.py          # LLM 查询改写 + 规则回退
    retriever.py               # 向量召回 + 关键词召回 + 重排
    reranker.py                # Rule-based 粗排 + LLM 精排
    evidence_builder.py        # 按任务组装 evidence package

  recommender/
    recommendation_pipeline.py # 强约束推荐流程
    filters.py                 # 过滤规则
    ranker.py                  # 推荐排序

  planning/
    learning_path_planner.py   # 学习路线规划
    diagnosis_planner.py       # 学习困难诊断

  tools/
    runtime.py                 # Function Calling 可靠执行网关
    registry.py                # 工具注册表
    resource_tool.py           # 资源查询 / 课程详情工具
    user_tool.py               # 用户画像 / 知识状态工具
    feedback_tool.py           # 用户反馈工具

  memory/
    context_builder.py         # 为 Router / Retriever / Planner / Generator 构建阶段上下文
    context_compressor.py      # 会话摘要、列表裁剪、证据裁剪
    memory_service.py          # 分层 memory_context 构建入口
    session_memory.py          # 会话记忆
    user_memory.py             # 用户画像记忆
    knowledge_state.py         # 知识状态记忆
    feedback_memory.py         # 反馈记忆
    resource_memory.py         # 推荐 / 资源行为记忆
    collaborative_memory.py    # 后续协同记忆扩展

  generation/
    prompts.py                 # Prompt 模板
    llm_client.py              # LLM 调用封装
    response_generator.py      # 最终回答生成与 fallback

  services/
    resource_service.py        # 资源服务
    user_service.py            # 用户上下文服务
    feedback_service.py        # 反馈服务
    handoff_service.py         # 人工兜底 case 服务

  stores/
    resource_store.py          # 只读 MOOPer 资源库访问
    user_store.py              # 用户画像、知识状态、反馈、推荐历史
    trace_store.py             # Agent run / step 持久化
    handoff_store.py           # 人工兜底 case 持久化
    tool_call_store.py         # Function Calling 幂等记录
    vector_index_store.py      # chunk 映射读取

  db/
    sqlite.py                  # SQLite 连接
    migrations.py              # app.db 初始化建表

  evaluation/
    datasets.py                # 读取评估用例
    metrics.py                 # 指标计算
    evaluators.py              # 各任务评估器
    judge.py                   # 轻量规则 Judge
    report.py                  # 输出评估报告

scripts/
  build_mooper_db.py           # 构建资源数据库
  init_app_db.py               # 初始化 app.db
  build_chunks.py              # 构建 RAG chunks
  build_vector_index.py        # 构建 Chroma 向量库
  run_agent_cli.py             # 终端交互入口
  run_eval.py                  # 离线评估入口

data/                          # 本地运行数据，已被 .gitignore 忽略，不上传 GitHub
```

## API 与前端

当前提供一个 FastAPI 后端和 API 驱动的前端 MVP。前端不是纯静态展示页，会通过后端接口读取资源、用户画像、笔记、反馈和 Agent 输出：

```text
frontend/
  index.html
  styles.css
  app.js
```

API 入口：

```text
app/api/server.py
app/api/schemas.py
```

主要接口：

- `GET /api/health`：健康检查
- `POST /api/agent`：调用 AgentOrchestrator，返回回答、推荐资源、证据、trace id
- `POST /api/agent/stream`：流式返回 Agent 回答
- `POST /api/auth/register`：注册用户
- `POST /api/auth/login`：登录用户
- `GET /api/resources/search`：从 MOOPer 资源库搜索候选资源
- `GET /api/resources/{resource_id}`：读取资源详情、章节、练习、知识点
- `POST /api/feedback`：记录用户对资源的反馈
- `GET /api/users/{user_id}/context`：查看用户画像、知识状态、反馈和推荐历史
- `PUT /api/users/{user_id}/profile`：更新用户画像
- `GET /api/users/{user_id}/saved-resources`：查看已加入学习的资源
- `POST /api/users/saved-resources`：加入学习资源
- `DELETE /api/users/{user_id}/saved-resources/{resource_id}`：移除已加入资源
- `GET /api/users/{user_id}/notes`：读取学习笔记
- `POST /api/users/notes`：新建学习笔记
- `PUT /api/users/{user_id}/notes/{note_id}`：更新学习笔记
- `DELETE /api/users/{user_id}/notes/{note_id}`：删除学习笔记
- `GET /api/users/{user_id}/settings`：读取前端侧 Agent 设置
- `PUT /api/users/settings`：更新前端侧 Agent 设置

运行：

```bash
python -m uvicorn app.api.server:app --host 127.0.0.1 --port 8010 --reload
```

然后打开：

```text
http://127.0.0.1:8010/
```

## 核心流程

### 1. 请求入口

```text
AgentOrchestrator.run(user_id, query, session_id)
  -> UserService.get_user_context()
  -> MemoryService.build_memory_context()
  -> Router.route()
  -> 根据 RoutingDecision 分流
```

`RoutingDecision` 不只是任务分类，还包含：

- `task_type`：recommend / qa / learning_path / diagnosis / feedback / chat
- `pipeline`：实际执行分支
- `needs_rag`：是否需要 RAG
- `needs_user_profile`：是否需要用户画像
- `needs_agent_loop`：是否需要 AgentLoop
- `information_sufficient`：信息是否足够
- `needs_clarification`：是否需要追问
- `clarification_questions`：追问问题

### 2. 推荐资源流程

推荐是强约束流程，不允许模型决定是否跳过关键步骤：

```text
Router
  -> recommendation_pipeline
  -> RAG 召回候选
  -> ResourceStore 补全真实课程资料
  -> 规则过滤
  -> 排序去重
  -> 记录 recommendation_event
  -> ResponseGenerator 生成回答
```

推荐结果不凭空生成，候选来自：

- Chroma 向量检索
- chunks 关键词召回
- ResourceStore 数据库检索
- EvidenceBuilder 补全课程、章节、练习、知识点等真实资料

### 3. RAG 问答流程

```text
query
  -> QueryRewriter
  -> VectorStore + Keyword Recall
  -> HybridReranker
  -> EvidenceBuilder
  -> ResponseGenerator
```

回答生成时会把 evidence package、用户上下文、任务类型和记忆摘要传给 LLM，并提供确定性 fallback，降低无证据幻觉风险。

### 4. 学习路线 / 学习诊断 AgentLoop

学习路线和诊断采用受控工具调用型 AgentLoop：

```text
read_memory
  -> tool.get_user_context

search_courses
  -> tool.search_courses

inspect_course_detail
  -> tool.get_course_detail

retrieve_evidence
  -> 内部 RagRetriever + EvidenceBuilder

run_learning_path_planner / run_diagnosis_planner
  -> 内部 Planner

generate_response
  -> ResponseGenerator
```

这里采用“工具层 + 内部组件”的混合设计：

```text
工具层 Tool
  负责读取数据、查资源、查课程详情、写反馈

内部组件 Internal Component
  负责检索融合、证据构建、排序、规划、生成和安全策略
```

这样既能体现 Agent 的工具调用能力，又避免 LLM 跳过必须执行的 RAG、Evidence、Planner 和 Guard。

### 5. 反馈调整流程

```text
用户反馈
  -> Router 识别 feedback
  -> Orchestrator 推断反馈对象
  -> FeedbackService 记录反馈
  -> Memory 后续读取反馈摘要
  -> 推荐排序时避让或加权
```

如果无法确定反馈对象，系统会追问用户要反馈哪一个资源。

## 关键设计策略

### Router 策略

Router 采用 LLM 优先、规则兜底，并在后端做策略约束：

- LLM 输出必须通过结构校验
- 不合法任务类型回退到规则路由
- 模糊请求进入 clarification
- 推荐请求如果已有明确主题，可以先推荐，再在回答末尾追加个性化追问

### Function Calling 可靠性

工具不允许裸调用，必须经过 `FunctionCallRuntime.execute()`：

```text
LLM / AgentLoop tool call
  -> ToolRegistry 查找工具
  -> SchemaValidator 校验参数
  -> ArgumentResolver 补全 user_id / source 等系统参数
  -> PermissionPolicy 校验权限
  -> IdempotencyGuard 防止重复写入
  -> tool.execute()
  -> ResultValidator 校验结果结构
  -> AgentTraceRecorder 记录 tool step
```

已实现的可靠性能力：

- Schema 校验：required、type、array、object、minimum、maximum、default
- 参数补全：从 AgentState / ToolCallContext 补齐系统已知参数
- 权限校验：读工具需要 `tool:read`，写工具需要 `tool:write`
- 跨用户保护：禁止访问或修改其他用户数据
- 反馈目标校验：反馈资源必须来自近期推荐历史
- 幂等控制：写工具结果写入 `tool_call_records`
- 结果校验：工具返回必须是 JSON object，并包含关键字段
- Trace：工具调用会写入 `tool.{tool_name}`

### Memory 分层

Memory 不直接替代 Store，而是面向 Agent 组装上下文：

```text
memory_context = {
  raw_memory,
  compressed_raw_memory,
  routing_context,
  retrieval_context,
  ranking_context,
  planning_context,
  generation_context,
  router_prompt_context,
  retriever_prompt_context,
  planner_prompt_context,
  generator_prompt_context
}
```

不同模块只读取自己需要的上下文：

- Router 优先使用 `router_prompt_context`
- Retriever 优先使用 `retriever_prompt_context`
- Ranker 使用 `ranking_context`
- Planner 优先使用 `planner_prompt_context`
- ResponseGenerator 使用 `generator_prompt_context` 和 `generation_context`

上下文压缩策略：

- SessionMemory 保留最近 30 轮原始会话
- ContextCompressor 在 prompt context 中保留最近 8 轮
- 更早的 turns 会压缩为 `turn_summary`
- 用户画像、反馈、知识状态保持结构化字段
- evidence package 默认裁剪到 8 条，每条内容做长度截断
- 当前版本先用字符长度和列表条数控制，不引入 token budget

### RAG 检索策略

检索不是单一向量搜索，而是混合检索：

```text
QueryRewrite
  -> Chroma vector recall
  -> chunks keyword recall
  -> merge candidates
  -> Rule-based Reranker
  -> optional LLM Reranker
  -> EvidenceBuilder
```

当 embedding 或 Chroma 不可用时，Retriever 会记录 fallback reason，并退回关键词召回，避免整个系统直接不可用。

### Evidence Builder 策略

Evidence Builder 按任务组织证据：

- 推荐任务优先课程、章节、练习和资源元数据
- QA 任务优先知识点、章节说明和课程内容
- 学习路线任务优先前置知识、课程结构、练习路径
- 学习诊断任务优先用户薄弱知识点、练习资源和补救课程

它的职责是把检索结果、用户画像、知识状态和资源元数据组装成事实包，约束 LLM 基于证据回答。

### AgentLoop 策略

AgentLoop 不是完全自由 ReAct，而是 Policy-Guided ReAct：

```text
ActionPlanner 选择下一步
Policy 校验动作是否合法
Executor 执行动作
ObservationChecker 检查结果
RuntimeGuard 判断是否需要兜底
Trace 记录全过程
```

这种设计适合教育推荐场景：流程可控、结果可解释、失败可追踪。

### 兜底与人工接管

正常情况下不进入兜底。证据不足或信息不足会优先追问用户。

不可恢复问题才触发 RuntimeGuard：

- AgentLoop 超过最大轮次
- AgentLoop 整体超时
- 单步执行超时
- LLM 输出多次校验失败
- 数据库异常
- 向量库异常
- 检索工具异常
- 未预期异常

触发后会创建 `handoff_cases`：

```text
reason_code
reason_text
trace_run_id
user_id
query
context_json
status
priority
```

同时 trace 中会记录触发原因和 `handoff_case_id`。

## 数据库与数据流

### mooper.db

只读资源库，来自 MOOPer 数据集，包含课程、章节、练习、知识点等资源信息。

由 `ResourceStore` 访问，主要用于：

- 搜索课程
- 获取课程详情
- 补全 RAG 检索结果的真实资源资料

### app.db

应用数据库，包含：

- 用户
- 用户画像
- 知识状态
- 反馈记录
- 推荐历史
- 资源行为事件
- agent_runs
- agent_steps
- handoff_cases
- tool_call_records

由 `UserStore`、`TraceStore`、`HandoffStore`、`ToolCallStore` 等模块访问。

## Trace 与可观测性

Agent 执行过程会写入两类 trace：

```text
app.db:
  agent_runs
  agent_steps

data/traces:
  trace.json
  {run_id}.trace.json
```

Trace 用于检查：

- Router 是否正确分流
- AgentLoop 是否按预期调用工具
- RAG 是否检索到证据
- Planner 是否生成计划
- 是否触发 RuntimeGuard
- handoff case 是否创建

## Evaluation 评估体系

评估模块独立于主流程，用于离线回放和质量检查。

本地评估文件：

```text
data/eval/
  retrieval_cases.jsonl
  recommendation_cases.jsonl
  qa_cases.jsonl
  agent_loop_cases.jsonl
  failure_cases.jsonl
```

评估用例保存在本地 `data/eval/`，默认不随 GitHub 仓库上传。仓库保留评估代码与运行入口，评估数据需要在本地准备或从私有数据目录恢复。

评估维度：

- 检索：Precision@K、Recall@K、MRR、关键词覆盖率、重复率
- 推荐：推荐命中、关键词覆盖、重复推荐率
- QA：关键词覆盖、groundedness、usefulness
- AgentLoop：trace action 是否按预期执行
- Failure：handoff reason_code 是否匹配

评估报告输出：

```text
data/eval_reports/latest_eval.json
data/eval_reports/latest_eval.md
```

## 本地运行准备

因为仓库不包含 `data/`，首次 clone 后需要先准备本地数据。

推荐顺序：

```text
1. 下载并解压 MOOPer 数据集到 data/raw/
2. 构建 data/processed/mooper.db
3. 初始化 data/processed/app.db
4. 构建 data/indexes/chunks.jsonl
5. 构建 data/chroma 向量库
6. 启动 FastAPI 服务或 CLI
```

## 运行方式

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 构建 mooper.db

请先将 MOOPer 原始数据放到 `data/raw/`，再执行：

```powershell
python scripts/build_mooper_db.py
```

### 3. 初始化 app.db

```powershell
python scripts/init_app_db.py
```

### 4. 构建 RAG chunks

```powershell
python scripts/build_chunks.py --types course,chapter,exercise,knowledge_point --output data/indexes/chunks.jsonl
```

### 5. 构建 Chroma 向量库

```powershell
python scripts/build_vector_index.py --reset --batch-size 64
```

`build_vector_index.py` 会读取 `.env` 或系统环境变量中的：

```env
DASHSCOPE_API_KEY=
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus-2025-04-28
EMBEDDING_MODEL=text-embedding-v4
```

### 6. 终端测试 Agent

交互模式：

```powershell
python scripts/run_agent_cli.py --user-id cli_user --session-id test_session
```

单轮测试：

```powershell
python scripts/run_agent_cli.py --query "零基础三个月学习机器学习路线" --no-llm-route --no-llm-rerank --no-llm-generation --debug-json
```

输出完整 JSON：

```powershell
python scripts/run_agent_cli.py --query "推荐人工智能入门课程" --json
```

### 7. 检索用例

```powershell
python scripts/run_retrieval_cases.py --top-k 5
```

### 8. 离线评估

```powershell
python scripts/run_eval.py --suites retrieval,recommendation,qa,agent_loop --top-k 5
python scripts/run_eval.py --suites agent_loop --json
```

## 关键配置

```env
DATA_DIR=data
PROCESSED_DATA_DIR=data/processed
INDEX_DIR=data/indexes

MOOPER_DB_PATH=data/processed/mooper.db
APP_DB_PATH=data/processed/app.db
CHUNKS_PATH=data/indexes/chunks.jsonl
CHROMA_DIR=data/chroma
CHROMA_COLLECTION=mooc_resource_chunks

TRACE_DIR=data/traces
TRACE_JSON_PATH=data/traces/trace.json

DASHSCOPE_API_KEY=
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus-2025-04-28
EMBEDDING_MODEL=text-embedding-v4

AGENT_MAX_ITERATIONS=6
AGENT_LOOP_TIMEOUT_SECONDS=30
AGENT_STEP_TIMEOUT_SECONDS=15
LLM_VALIDATION_FAILURE_THRESHOLD=2
```
