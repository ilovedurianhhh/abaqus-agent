# Abaqus Analysis Agent

基于大语言模型的 Abaqus 有限元分析自动化系统。用户用自然语言描述结构分析问题，Agent 自动生成分析脚本、校验代码正确性、调用 Abaqus CAE 求解、读取后处理结果并返回。

## 系统架构

```
用户自然语言输入
       │
       ▼
┌─────────────────┐
│   Agent 层      │  agent/          ← 自然语言 → Python 代码
│   (Kimi LLM)    │
└────────┬────────┘
         │ ① 关键词路由选择 few-shot 示例
         │ ② Kimi 生成 <plan> + <code>
         │ ③ AST 安全校验（拦截危险调用）
         │ ④ API 签名校验（拦截错误的方法/参数）
         │ ⑤ 校验失败 → 反馈给 Kimi 自动修正（最多 3 轮）
         ▼
┌─────────────────┐
│   API 封装层    │  abaqus_api/     ← Python API → Abaqus 脚本
│   (AbaqusModel) │
└────────┬────────┘
         │ 累积的 Abaqus Python 2.7 脚本
         ▼
┌─────────────────┐
│   通信桥层      │  abaqus_bridge.py ← 脚本 → Abaqus CAE 进程
│   (AbaqusBridge) │
└────────┬────────┘
         │ abaqus cae noGUI=...
         ▼
┌─────────────────┐
│   Abaqus CAE    │  求解器执行 + ODB 结果
└─────────────────┘
```

系统分为三层，各层职责单一、可独立使用：

| 层级 | 目录 | 职责 | 可否独立使用 |
|------|------|------|:---:|
| 通信桥 | `abaqus_bridge.py` | 将 Python 代码发送到 Abaqus CAE 执行，捕获返回结果 | 是 |
| API 封装 | `abaqus_api/` | 提供面向对象的 Python API，累积生成 Abaqus 脚本 | 是 |
| Agent 智能层 | `agent/` | 接收自然语言，调用 LLM 生成代码，校验并执行，返回结果 | 是 |

## 已实现功能

### 12 种分析场景（关键词路由 + Few-shot 示例）

用户输入时，系统根据关键词自动选择最相关的 1-2 个 few-shot 示例注入 prompt，引导 Kimi 生成正确代码。

| 分析类型 | 路由关键词 | 关键 API 方法 |
|---------|-----------|-------------|
| 静力学-压力 | 压力、均布、pressure | `create_static` + `pressure` |
| 静力学-集中力 | 集中力、点力 | `concentrated_force` |
| 重力加载 | 重力、自重、gravity | `create_density` + `gravity` |
| 非线性大变形 | 大变形、塑性、屈服 | `create_static(nlgeom=True)` + `create_plastic` |
| 模态分析 | 模态、频率、振动、固有 | `create_frequency` |
| 动力显式 | 冲击、动力、显式 | `create_dynamic_explicit` + `create_density` |
| 壳结构 | 壳、薄板、薄壁 | `extrude_shell` + `create_shell_section` |
| 位移约束 | 位移约束、强制位移、对称 | `displacement_bc` |
| 旋转体 | 旋转、轴对称、圆柱 | `revolve_solid` + `circle` |
| 多零件装配 | 装配、多零件、组装 | `create_instance` + `translate` + `rotate` |
| 网格细化 | 网格细化、加密、seed | `seed_edge_by_number` + `line` |
| 自定义场输出 | 场输出、PEEQ、应变 | `set_field_output` + `odb.field_output` |

覆盖 `abaqus_api` 全部 33 个公开 API 方法。

### 代码安全校验（exec 前拦截）

生成的代码在执行前经过两层校验：

**第一层：AST 安全校验**
- 语法检查（`ast.parse`）
- 禁止危险导入（`os`、`subprocess`、`socket` 等）
- 禁止危险调用（`eval()`、`exec()`、`__import__()` 等）

**第二层：API 签名校验**
- 方法是否存在（拦截 LLM 臆造的 API，如 `create_cylinder`）
- 必需参数是否齐全（拦截遗漏参数，如 `create_elastic` 缺少 `nu`）
- 是否传了未知参数（拦截参数名拼错）
- 语义约束检查（`pressure` 必须用 `surface`，`fix` 必须用 `set_name`）

校验失败时，精确的错误信息（如"第15行: `create_cylinder()` 方法不存在"）自动反馈给 Kimi 修正，最多 3 轮。

### 日志系统

- 全链路 Python `logging`：LLM 调用耗时、token 用量、代码执行耗时、校验结果
- 日志输出到 `logs/agent.log`
- 日志级别可通过 `.env` 中 `LOG_LEVEL` 配置

### CLI 交互功能

| 命令 | 功能 |
|------|------|
| `/help` | 查看帮助 |
| `/history` | 查看当前对话历史 |
| `/save` | 保存会话到 JSON |
| `/export` | 导出带时间戳的会话文件 |
| `/clear` | 清空对话历史 |
| `quit`/`exit` | 退出并自动保存 |

LLM 生成过程中显示加载动画。

### 配置管理

- API Key 通过 `.env` 文件管理（不再硬编码）
- `.gitignore` 排除敏感文件和运行时目录
- `requirements.txt` 声明 Python 依赖
- `.env.example` 提供配置模板

### 性能优化

- `build_api_reference()` 使用 `@lru_cache` 缓存，API 内省只执行一次
- API 签名校验器的签名表也缓存
- prompt 大小恒定（~7000 字符），不随示例总数增长

## 环境要求

- **Python**: 3.8+
- **Abaqus CAE**: 2020 及以上版本（需在系统 PATH 中可用）
- **操作系统**: Windows（通过 PowerShell 调用 Abaqus）

## 安装

```bash
cd C:\Users\39037\abaqus-agent
pip install -r requirements.txt
```

配置 API Key：

```bash
# 复制模板
copy .env.example .env

# 编辑 .env，填入你的 Kimi API Key
# KIMI_API_KEY=sk-your-api-key-here
```

## 快速开始

### 交互式对话（推荐）

```bash
python -m agent
```

```
Abaqus Analysis Agent
输入分析需求开始，输入 /help 查看帮助

> 分析一个 500x50x30mm 的铝合金悬臂梁，左端固定，右端施加 1000N 向下的集中力

⠹ Kimi 生成中...

## 分析方案
- 几何：500×50×30 mm 长方体
- 材料：铝合金 (E=70000 MPa, ν=0.33)
- 边界条件：左端 (x=0) 全固定
- 载荷：右端 (x=500) 集中力 CF2=-1000 N
- 网格：C3D8R, 种子尺寸 10mm

## 分析完成
- Max von Mises Stress: 42.15 MPa
- Max Displacement: 1.2345 mm
- Max Reaction Force: 1000.00 N

输出文件: output/BeamJob_20260312_143000/

> 把力改成 2000N 重新跑

## 分析完成
- Max von Mises Stress: 84.30 MPa
- Max Displacement: 2.4690 mm
...
```

支持多轮对话，可以直接说"把力改成 2000N"、"换成钢材"、"网格加密到 3mm"等。

### 直接使用 Python API（不经过 LLM）

```python
from abaqus_api import AbaqusModel

m = AbaqusModel("MyBeam")

# 几何
m.part.create_sketch("s", sheet_size=400)
m.part.rectangle(p1=(0, 0), p2=(200, 20))
m.part.extrude_solid("Beam", depth=20)

# 面集合与表面
m.part.create_face_set("Beam", "FixedEnd",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=20.1, zmax=20.1)
m.part.create_surface("Beam", "LoadSurf",
    xmin=199.9, ymin=-0.1, zmin=-0.1, xmax=200.1, ymax=20.1, zmax=20.1)

# 材料
m.material.create_elastic("Steel", E=210000, nu=0.3)
m.material.create_solid_section("Sec", material="Steel")
m.material.assign_section("Beam", "Sec")

# 装配、分析步
m.assembly.create_instance("Beam")
m.step.create_static("Loading")

# 边界条件与载荷
m.load.fix("BC", instance="Beam-1", set_name="FixedEnd")
m.load.pressure("P", step="Loading", instance="Beam-1",
                surface="LoadSurf", magnitude=10.0)

# 网格
m.mesh.seed_part("Beam", size=5.0)
m.mesh.set_element_type("Beam", "C3D8R")
m.mesh.generate("Beam")

# 提交求解
result = m.submit("MyJob", wait=True)

# 读取结果
summary = m.odb.max_values("MyJob.odb", work_dir=result["output_dir"])
print(summary)
```

### 仅使用通信桥

```python
from abaqus_bridge import AbaqusBridge

bridge = AbaqusBridge()

# 检查 Abaqus 是否可用
bridge.ping()

# 执行任意 Abaqus Python 代码
result = bridge.execute("""
from abaqus import mdb
__result__ = {'models': list(mdb.models.keys())}
""")
print(result.result_data)
```

## 项目文件说明

### 通信桥层

| 文件 | 说明 |
|------|------|
| `abaqus_bridge.py` | 通信桥核心。通过 PowerShell 启动 `abaqus cae noGUI` 执行 Python 脚本，自动包装 try/except 和 JSON 结果捕获 |

### API 封装层 (`abaqus_api/`)

| 文件 | 类 | 说明 |
|------|------|------|
| `model.py` | `AbaqusModel` | 顶层入口类，持有所有 builder 子对象，提供 `preview()`、`submit()`、`reset()` |
| `codegen.py` | `CodeBuffer` | 代码累积缓冲区，所有 builder 共享同一个缓冲区 |
| `part.py` | `PartBuilder` | 几何建模：草图（矩形/圆/线段）、拉伸实体、拉伸壳、旋转体、面集合、表面 |
| `material.py` | `MaterialBuilder` | 材料与截面：弹性、密度、塑性、实体截面、壳截面、截面分配 |
| `assembly.py` | `AssemblyBuilder` | 装配：实例化、平移、旋转 |
| `step.py` | `StepBuilder` | 分析步：静力、动力显式、频率提取、场输出控制 |
| `load.py` | `LoadBuilder` | 载荷与边界条件：固支、位移约束、压力、集中力、重力 |
| `mesh.py` | `MeshBuilder` | 网格：全局种子、边种子、单元类型设置、生成网格 |
| `job.py` | `JobBuilder` | 作业：创建、提交等待、写入输入文件、结果捕获 |
| `odb.py` | `OdbReader` | 后处理：最大 Mises 应力/位移/反力、自定义场输出读取 |

### Agent 智能层 (`agent/`)

| 文件 | 类 | 说明 |
|------|------|------|
| `prompts.py` | — | 12 种 few-shot 示例 + 关键词路由选择 + API 内省生成参考文档（带 LRU 缓存） |
| `code_validator.py` | — | API 签名校验器：内省真实 API 签名，AST 解析生成代码，校验方法/参数正确性 |
| `llm.py` | `LLMClient` | Kimi (Moonshot AI) API 封装，从 `.env` 读取 Key，记录 token 用量 |
| `history.py` | `ConversationHistory` | 对话历史管理，保留最近 20 轮 |
| `agent.py` | `AbaqusAgent` | 核心 Agent：自然语言 → LLM 生成 → 安全校验 → API 签名校验 → 执行 → 失败重试 |
| `cli.py` | — | 交互式 REPL：命令系统、加载动画、会话保存/导出 |

### 配置文件

| 文件 | 说明 |
|------|------|
| `.env` | API Key 和日志级别配置（不提交到 Git） |
| `.env.example` | 配置模板 |
| `.gitignore` | 排除 `.env`、运行时目录、Abaqus 临时文件 |
| `requirements.txt` | Python 依赖：`openai`、`python-dotenv` |

### 测试 (`tests/`)

| 文件 | 说明 | 需要 |
|------|------|------|
| `test_agent.py` | 38 个离线测试：路由、解析、安全校验、历史、格式化 | 无 |
| `test_code_validator.py` | 16 个测试：API 签名校验器（方法存在性、参数正确性、约束规则） | 无 |
| `test_routing_demo.py` | 路由效果可视化 + API 覆盖率报告（33/33 = 100%） | 无 |
| `test_kimi_generation.py` | 12 种场景 × Kimi 生成 + 安全校验 + API 签名校验 | Kimi API |
| `test_cantilever.py` | API 层端到端：构建悬臂梁 → 提交 Abaqus → 验证 ODB | Abaqus |

### 运行时目录

| 目录 | 说明 |
|------|------|
| `scripts/` | 临时存放发送给 Abaqus 的 Python 脚本（执行后自动清理） |
| `results/` | 临时存放 Abaqus 返回的 JSON 结果文件 |
| `output/` | 求解输出，按 `<作业名>_<时间戳>/` 组织 |
| `logs/` | 日志文件 |

## 测试方法

```bash
# ① 离线单元测试（不需要 API Key 和 Abaqus，秒级完成）
python tests/test_agent.py            # 38 个测试：路由、解析、安全校验、历史
python tests/test_code_validator.py   # 16 个测试：API 签名校验器
python tests/test_routing_demo.py     # 路由效果 + API 覆盖率可视化

# ② 在线生成测试（需要 Kimi API Key，约 2-3 分钟）
python tests/test_kimi_generation.py  # 12 种场景 × Kimi 生成 + 签名校验

# ③ API 层测试（需要 Abaqus）
python tests/test_cantilever.py       # 完整执行
python tests/test_cantilever.py --preview  # 仅预览脚本

# ④ 端到端测试（需要 Kimi API Key + Abaqus）
python tests/test_agent.py --e2e

# ⑤ 交互式测试
python -m agent
```

## 核心工作流详解

```
用户输入: "分析一个 200x20x20mm 钢制悬臂梁，左端固定，右端面压力 10MPa"
                │
                ▼
        ┌───────────────┐
        │ 关键词路由     │ → 匹配"压力" → 选择 static_pressure 示例
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ Kimi LLM      │ → 返回 <plan> + <code>
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ AST 安全校验  │ → 拦截 os.system/eval/exec 等
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ API 签名校验  │ → 拦截不存在的方法、错误参数、set/surface 混淆
        └───────┬───────┘
                │     ┌──────────────────────────────────┐
                │     │ 任一校验失败：                    │
                │     │ → 精确错误信息反馈给 Kimi         │
                │     │ → Kimi 修正代码（最多 3 轮）     │
                │     └──────────────────────────────────┘
                ▼
        ┌───────────────┐
        │ exec(code)    │ → abaqus_api → bridge → Abaqus CAE
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ 格式化输出    │ → Mises 应力、位移、反力、输出路径
        └───────────────┘
```

## 注意事项

### face set 与 surface 的区别

这是最常见的错误来源（已通过 API 签名校验器自动拦截）：

- **`create_face_set()`** 创建面**集合（Set）** → 用于 `fix()`、`displacement_bc()`、`concentrated_force()`
- **`create_surface()`** 创建**表面（Surface）** → 用于 `pressure()`

### 包围盒选择的容差

使用 `±0.1` 的容差来选择面，确保精确捕获目标几何面：

```python
# 选择 x=0 处的面
m.part.create_face_set("Beam", "FixedEnd",
    xmin=-0.1, ymin=-0.1, zmin=-0.1,   # 目标坐标 - 0.1
    xmax=0.1,  ymax=20.1, zmax=20.1)   # 目标坐标 + 0.1
```

### 实例命名约定

Abaqus 默认的实例名是 `"<PartName>-1"`。例如 Part 名为 `"Beam"`，则实例名为 `"Beam-1"`。

## 理解 Agent：LLM 在本项目中的角色

### LLM（Kimi）到底做了什么

Kimi 在整个系统中只承担一个职责：**把自然语言翻译成 abaqus_api 的 Python 代码**。

Kimi **不执行任何分析，不接触 Abaqus，不读取结果文件**。它的全部工作就是输入一段文字、输出一段代码。真正执行分析的是 `abaqus_api` 和 `abaqus_bridge`，真正编排流程的是 `agent.py` 中的决策循环。

### Agent 的本质

```
Agent = LLM（大脑） + 工具（手脚） + 决策循环（行为模式） + 记忆（经验）
```

| Agent 组件 | 概念 | 本项目对应 | 文件 |
|-----------|------|-----------|------|
| 大脑 | LLM，负责理解需求和生成方案 | Kimi API 调用 | `agent/llm.py` |
| 手脚 | 工具，负责执行具体操作 | abaqus_api + AbaqusBridge | `abaqus_api/` + `abaqus_bridge.py` |
| 行为模式 | 决策循环 | 生成 → 校验 → 执行 → 失败 → 反馈 → 重试 | `agent/agent.py` |
| 经验 | 记忆 | 对话历史（最近 20 轮） | `agent/history.py` |
| 知识 | 指令 | system prompt + 12 种 few-shot 示例 | `agent/prompts.py` |
| 安全 | 代码审查 | AST 安全校验 + API 签名校验 | `agent/agent.py` + `agent/code_validator.py` |

### LLM 是可替换的组件

换掉 LLM 只需修改 `.env` 中的配置：

```bash
# Kimi（当前）
KIMI_API_KEY=sk-your-key
# KIMI_MODEL=moonshot-v1-32k

# 换成其他 OpenAI 兼容 API 只需改 llm.py 中的 base_url
```

## 后续优化方向

- [ ] 支持 2D 平面分析（平面应力/平面应变）
- [ ] 支持接触分析
- [ ] 应力云图导出、路径提取、节点历史输出
- [ ] 支持热分析（需扩展 abaqus_api）
- [ ] 用语义嵌入替代关键词路由，提升匹配准确率
- [ ] 执行前让用户预览/确认生成的代码
