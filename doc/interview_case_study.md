# 代码知识库 RAG 系统 — 检索增强迭代记录

> **面向面试展示**：记录项目开发过程中遇到的实际问题、根因分析、解决方案和决策权衡。
> 更新日期：2026-08-26

---

## 一、项目背景

这是一个**私有代码知识库 RAG 系统**，目标是让用户用自然语言提问，系统检索代码库中的相关片段并生成回答。

### 技术栈

| 层 | 技术 |
|---|---|
| 向量模型 | DashScope `text-embedding-v4` / `qwen3.7-text-embedding`（二选一切换） |
| 向量库 | 自研 `InMemoryVectorStore`（余弦相似度，零依赖） |
| 关键词检索 | `rank_bm25`（BM25Okapi） |
| 重排 | Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| 生成 | DashScope `qwen-max` |
| UI | Streamlit（深色主题 + 自研 CSS） |

### 检索架构（5 步流水线）

```
用户查询
  │
  ├─ ① 代码感知分片（按 def/class 边界切分，每函数独立分片）
  │
  ├─ ② 多路召回
  │     ├─ 向量检索 top-20（语义相似度）
  │     └─ BM25 检索 top-20（关键词匹配）
  │
  ├─ ③ 融合加权
  │     ├─ 分路 min-max 归一化
  │     ├─ 向量分 × 0.3 + BM25 分 × 0.3 + 关键词分 × 0.4
  │     ├─ × 关键词命中倍率 × 源文件倍率
  │     ├─ + 业务关键词路由 boost（+0.15）
  │     └─ × chunk_type 后处理（function/class_def 加分，import/log_call 降分）
  │
  ├─ ④ 邻域扩充（同文件前后 window=2 分片追加）
  │
  ├─ ⑤ Cross-Encoder 精排 + 源文件兜底 → Top 8 → 送入 LLM
  │
  └─ LLM 生成回答
```

---

## 二、问题一：Cross-Encoder 重排后工具函数分片丢失

### 现象

用户问"agent 依靠哪些工具来完成文件操作"，`file_tools.py` 有 9 个工具函数（每个独立分片），但 LLM 输出只列出 6 个，缺失 `copy_file`、`move_file`、`append_file`。

### 根因分析

**重排出口太窄（top_n=5）+ 兜底名额不足（quota=1）**

```
file_tools.py: 9 个函数 → 9 个独立分片
                    ↓
          向量/BM25 召回 top-20（全在候选池）
                    ↓
          Cross-Encoder 精排 top-5 ← 出口太窄！
                    ↓
          ensure_source_file_in_top_n 只保 1 个源文件分片
                    ↓
          copy_file/move_file/append_file 被挤出 top-5
          LLM 看不到 → 回答里缺失
```

Cross-Encoder 是 6 层小模型，对"中文自然语言 vs Python 代码"的语义匹配能力有限。`copy_file` 的 docstring 较短、`move_file` 实现偏技术性，rerank 分数不如 `main.py` 的启动逻辑分片，被挤出 top-5。

### 解决方案

#### 1. 扩宽重排出口：top_n=5 → top_n=8

```python
# app.py / main.py
top_n = reranker.rerank(query, candidates, top_n=8)
```

直接增加送入 LLM 的上下文宽度。Top-8 能容纳至少 4~5 个 `file_tools.py` 的函数分片。

#### 2. 升级兜底机制：quota 1 → quota 2

将 `ensure_source_file_in_top_n` 从"保 1 个源文件分片"升级为"可配置 quota，默认保 2 个"：

```python
# 改前：写死 1 个名额
def ensure_source_file_in_top_n(self, candidates, top_n, query):
    ...
    return top_n[:-1] + [new_item]  # 只替换最后 1 名

# 改后：可配置 quota，保多个名额
def ensure_source_file_in_top_n(self, candidates, top_n, query, quota=2):
    ...
    return top_n[:-len(new_items)] + new_items  # 替换最后 quota 名
```

兜底逻辑分两步：
- **文件名匹配优先**：用户查询里的英文词（如 `tool`、`file`）出现在文件名里时，从候选池拎最多 quota 个该文件分片替换最后 quota 名
- **业务实现兜底**：如果 top_n 里"含关键术语的源文件业务实现分片"不足 quota，按融合分降序补齐

### 效果

改后再问同样问题，LLM 能输出全部 9 个工具函数。

### 面试讲法

> "这个问题的本质是**召回-精排漏斗的出口宽度设计**。当一个代码模块有 N 个独立分片时，top_n 必须能容纳核心分片数量。我做了两件事：① 把 top_n 从 5 扩到 8，这是'宽进宽出'；② 把兜底机制从保 1 个名额升级到保 2 个，防止 rerank 分数偶然波动。这两个改动都是**低成本、可解释、可回退**的——如果以后发现 top_n=8 上下文太长导致 LLM 走神，可以在 8 和 5 之间调。"

---

## 三、问题二：业务关键词路由（轻量意图识别）

### 现象

用户问"agent 依靠哪些工具来完成文件操作"，召回结果全落在 `main.py`、`config.py`、`state.py`，没命中 `registry.py`、`file_tools.py`。

### 根因分析

**现有关键词 boost 是"所有命中关键词的分片都加分"，无法区分"在 main.py 里出现的'工具'"和"在 registry.py 里出现的'工具'"哪个更相关。**

向量和 BM25 检索的语义相似度判断是全局的——`main.py` 作为入口文件，分片里自然出现更多"工具""文件""操作"等词，导致它的得分反而更高。

### 解决方案：轻量业务关键词路由

在融合加权阶段增加一个**关键词→源文件映射表**，给目标文件加一个小 boost。

#### 映射表设计（`multi_retriever.py` 常量）

```python
BUSINESS_KEYWORD_ROUTE = {
    "tool": ["registry.py", "file_tools.py"],
    "工具": ["registry.py", "file_tools.py"],
    "scheduler": ["react_scheduler.py"],
    "调度": ["react_scheduler.py"],
    "llm": ["llm_client.py"],
    "大模型": ["llm_client.py"],
    "state": ["state.py"],
    "沙箱": ["sandbox.py"],
    "agent": ["main.py", "react_scheduler.py"],
    # ... 其他关键词映射
}
```

#### Boost 计算逻辑

```python
def _compute_business_route_boost(query, file_path) -> float:
    # 1. 只给源文件加分，测试文件不加
    # 2. 遍历映射表：查询包含关键词 + 当前文件在目标列表 → +0.15
    # 3. 多个关键词同时命中封顶 0.3
```

#### 注入融合打分（加法，不是乘法）

```python
# 在 final_score 计算之后，用加法叠加
final_score = base_score * kw_multiplier * src_multiplier  # 原有逻辑
route_boost = self._compute_business_route_boost(query_text, file_path)
final_score += route_boost  # 额外加一个小 boost
```

### 关键设计决策

| 决策 | 选择 | 原因 |
|---|---|---|
| Boost 值 | 0.15（加法） | 太大变成硬路由，太小没用。0.15 刚好能在势均力敌时帮目标文件胜出 |
| 加法 vs 乘法 | 加法 | 乘法会把语义差大的分片也放大，加法只是"推一把" |
| 复杂度 | 纯 dict + 遍历 | 不引入 LLM 分类器、不引入多级漏斗，零额外依赖 |
| 测试文件 | 不加 boost | 只引导到源代码，不误导到测试文件 |

### 面试讲法

> "这是一个典型的**检索方向不准**的问题。我没有上 LLM 意图分类器、多级漏斗这些重型方案，而是用了**一张映射表 + 一个加法 boost** 解决。理由有三个：
> 
> 1. **项目规模小**——只有 11 个源文件，一个 dict 足够覆盖所有业务场景
> 2. **零额外成本**——不增加 LLM 调用，不增加推理延迟
> 3. **可解释可维护**——面试时可以直接说'当用户问工具相关问题，我们会优先看 registry.py 和 file_tools.py'，而不是黑箱
> 
> 这在 RAG 工程里叫**路由增强**（Route Enhancement），区别于**检索增强**（Retrieval Enhancement）——前者是决定"去哪里找"，后者是决定"怎么找"。"

---

## 四、问题三：向量模型切换 + 批次大小动态适配

### 现象

将 embedding 模型从 `qwen3.7-text-embedding` 切换到 `text-embedding-v4` 时，报错：

```
InvalidParameter: batch size is invalid, it should not be larger than 10
```

### 根因分析

`text-embedding-v4` 的单批上限是 **10 条**，而代码里 `MAX_BATCH_SIZE` 写死了 **20 条**（是 `qwen3.7-text-embedding` 的上限）。

| 模型 | 单批上限 | 产出维度 |
|---|---|---|
| `qwen3.7-text-embedding` | 20 条 | 1024 |
| `text-embedding-v4` | 10 条 | 1024 |

### 解决方案

将 `MAX_BATCH_SIZE` 从写死的类常量改为**根据模型名动态决定**的实例属性：

```python
_MODEL_BATCH_LIMITS = {
    "qwen3.7-text-embedding": 20,
    "text-embedding-v4": 10,
    "text-embedding-v3": 10,
}
_DEFAULT_BATCH_SIZE = 10

def __init__(self):
    self.MAX_BATCH_SIZE = self._MODEL_BATCH_LIMITS.get(
        EMBEDDING_MODEL, self._DEFAULT_BATCH_SIZE
    )
```

### 面试讲法

> "这个问题的本质是**模型版本差异的兼容性设计**。不同的 embedding 服务批次上限不同，硬编码一个值肯定出问题。我把它改成了**配置驱动**——通过映射表根据当前模型名查对应的上限。这样以后再切模型，只要在映射表里加一行就行，不用改业务逻辑。
> 
> 另外注意：切换 embedding 模型后**必须重建索引**——同一个文本在不同模型下产出的向量不在同一个向量空间，余弦相似度计算毫无意义。我的项目是内存向量库，重启即重建，所以这个问题天然解决了。"

---

## 五、问题四：Streamlit UI 主题 + 源码浏览器

### 问题清单

| 问题 | 解决方案 |
|---|---|
| 默认白色主题刺眼 | `config.toml` 设 `base="dark"`，VS Code Dark+ 深灰基调 `#1e1e1e`（非纯黑，Trae/VS Code 同款） |
| 代码语法高亮丢失 | 全局 CSS 里用 `span[style*="..."]` 选择器 + `!important` 覆盖 Pygments 内联 style，按代码元素类型着色 |
| React removeChild 崩溃 | 删掉 CSS `position:relative;top:-28px` 的 DOM 位移 trick，改用原生按钮 + aria-label 选择器 |
| "查看完整视图"切换卡 bug | 删除 horizontal st.radio（与深色 CSS 兼容性问题），改用下方按钮 + 徽章显示当前模式 |
| 源码浏览器平铺下拉 | 改为树形叠层结构：缩进边框颜色从深到浅递进，每层 `padding-left` 增加，视觉上一层层叠进去 |

### 面试讲法

> "Streamlit 的 UI 定制有两个坑：① 全局 CSS 的 `!important` 容易误伤子组件（比如把代码高亮的 `<span>` 颜色全盖了）；② horizontal radio 在深色主题下有 React DOM 兼容问题。我的经验是：**尽量用 Streamlit 原生组件（type="primary/secondary" 的按钮、原生 tab），少用 CSS 位移 trick**。如果一定要自定义，优先用 aria-label 选择器而不是 DOM 结构选择器，因为 Streamlit 内部的 DOM 结构会随版本变化。"

---

## 六、问题五：rerank top_n 调优（3 → 5 → 8）

### 迭代过程

| 阶段 | top_n | 遇到的问题 | 调整 |
|---|---|---|---|
| v1 | 3 | 核心分片被挤出，回答不完整 | → 5 |
| v2 | 5 | 9 个工具函数的后 3 个被挤出 | → 8 + quota=2 |
| v3 | 8 | 当前稳定，所有工具函数都能进 LLM | 保持 |

### 设计原则

**召回 K ≥ 最终返回 K ≥ 1**，且出口宽度要覆盖目标模块的核心分片数量。

```
代码库规模      建议 top_n
───────────────────────
小（< 50 分片）   5~8
中（50~200 分片）  8~12
大（> 200 分片）   10~20 + 配额保障
```

### 面试讲法

> "Top N 的选择是**召回精度 vs 上下文长度**的 trade-off。Top N 太小，答案不完整；太大，LLM 会被噪声分片干扰、甚至超出 token 限制。我的做法是：先用**召回 K = 20** 拉足够多的候选，再用**重排 Top N = 8** 精挑细选，配合**配额保障**兜底。这是经典的'宽召回、严精排'设计模式。"

---

## 七、核心经验总结（面试金句）

### 检索链路排查方法论

```
召回漏了？
  ├─ 向量没召回 → 调大 VECTOR_CANDIDATE_K / 换 embedding 模型
  ├─ BM25 没召回 → 检查分词 / 关键词覆盖率
  └─ 召回了但被重排挤掉 → 调大 top_n / 加业务路由 boost

方向不对？
  └─ 关键词路由 → 在融合加权阶段给目标文件加 boost（加法，0.15~0.3）

出口太窄？
  └─ 扩 top_n + quota 兜底
```

### 决策原则

1. **先定位瓶颈环节**：召回阶段漏了？融合加权被压了？还是 rerank 挤掉了？不同环节的修复方法完全不同
2. **加法 boost 优于乘法 boost**：加法只是"推一把"，乘法会放大噪声
3. **可解释优于黑箱**：面试项目用 dict 映射表比 LLM 分类器更有说服力
4. **重建索引是切换 embedding 模型的必要步骤**：不同模型的向量空间不兼容

### RAG 检索增强全链路

```
查询理解 → 关键词路由（业务关键词映射）
    ↓
多路召回 → 向量 + BM25（互补关系，K 值相同）
    ↓
融合加权 → 归一化 + 加法 boost + 乘法增强 + chunk_type 后处理
    ↓
邻域扩充 → 同文件前后分片追加（window=2）
    ↓
精排出口 → Cross-Encoder top_n（8）+ 配额兜底（quota=2）
    ↓
LLM 生成 → 结构化 Prompt（来源文件 + 行号 + 代码片段）
```

---

## 附：改动文件清单

| 文件 | 改动内容 |
|---|---|
| `retriever/multi_retriever.py` | 业务关键词路由映射表、boost 计算方法、融合注入、ensure_source_file_in_top_n 升级 quota |
| `embedding_client.py` | 批次大小动态适配（根据模型名查映射表） |
| `app.py` | top_n=8、quota=2、Streamlit 深色主题 CSS、源码浏览器树形 UI |
| `main.py` | top_n=8、quota=2 同步 |
| `.streamlit/config.toml` | 深色主题配置 |
| `.env` | `EMBEDDING_MODEL=text-embedding-v4` |

---

## 快速验证（面试现场演示用）

### 启动命令

```bash
# Streamlit 版（推荐，带 UI 演示）
cd d:\code_RAG
python -m streamlit run app.py

# 命令行版（纯日志输出，看检索分数）
python main.py
```

### 测试查询语句（用于演示各改进点）

| 测试场景 | 查询语句 | 验证点 |
|---|---|---|
| **工具函数全量召回** | "agent 依靠哪些工具来完成文件操作？" | 应输出全部 9 个工具函数（list_dir/create_file/read_file/delete_file/make_folder/search_file/copy_file/move_file/append_file） |
| **业务关键词路由** | "这个 agent 有哪些工具？" | 前几名分片应来自 `registry.py` 而非 `main.py` |
| **文件操作类检索** | "如何读取和写入文件？" | `file_tools.py` 的 `read_file`/`create_file`/`append_file` 应优先出现 |
| **调度器相关** | "react 调度器是怎么工作的？" | `react_scheduler.py` 的分片应排在前面 |
| **源码浏览器功能** | UI 里切到「📂 源码浏览器」Tab → 左侧树形文件树 → 点击文件查看彩色代码 | 展示树形叠层 UI 和语法高亮 |

### 验证模型切换是否生效

```bash
# 看启动日志里的向量化维度
python main.py
# 输出应有：得到 N 个向量，维度=1024（text-embedding-v4）
# 输出应有：MAX_BATCH_SIZE 自动适配为 10（text-embedding-v4 限制）
```

### 看召回详情（命令行版）

`python main.py` 会打印每个候选的融合分、向量分、BM25 分、关键词 boost、业务路由 boost（`route_boost` 字段）。面试时可以展示当查询包含"工具""文件"等业务关键词时，`registry.py` / `file_tools.py` 的分片获得了 `route_boost=0.15`。

