# CodeRAG — 私有代码知识库问答系统

一个面向私有代码库的 RAG（Retrieval-Augmented Generation，检索增强生成）问答系统。把一份 Python 源码工程喂给它，它就能听懂你的中文问题，自动从代码里捞出最相关的几段，交给大模型读完后用人话回答你，并附上代码出处。

**现已支持基于 Streamlit 的 Web 界面**，提供 VS Code Dark+ 风格的深色主题、分层树形源码浏览器、语法高亮、行号跳转以及问答结果一键联动源码查看等功能。

适合场景：接手陌生代码库快速上手、跨文件追查某功能实现、不想一个个翻文件找答案。

---

## 一、核心特性

- **代码感知切片**：按 `def` / `class` / 列表字典元素边界切，不在函数半道或字典中间断开，断点率可压到 0%。
- **双路召回 + 融合**：向量语义召回（阿里云 DashScope）+ BM25 关键词召回（rank_bm25），归一化加权融合，互补盲区。
- **多层启发式加权**：关键词精确命中加分、源文件优先 + 测试文件降权、chunk_type 类型偏好（function/class 加分，import/log 降分），用"加法打基础 + 乘法放大优势"。
- **同文件邻域扩充**：召回后对每个候选分片追加同文件前后 2 个邻居，让重排模型看到完整上下文。
- **Cross-Encoder 精排**：用 `cross-encoder/ms-marco-MiniLM-L-6-v2` 对二三十个候选逐对精排，取 Top 3 喂大模型。
- **源文件名额预留兜底**：重排后若 Top N 里没有目标源文件分片，强制拎一个塞进末位，保证大模型能看到目标代码。
- **混合重排分**：最终分 = 0.6×重排分 + 0.4×融合分，既听模型精排，又保留关键词/源文件启发式话语权。
- **零向量数据库依赖**：向量库用 NumPy 手搓内存版，BM25 用 rank_bm25，无需部署 Milvus/FAISS/ES。
- **断点率可视化调试**：`chunk_viewer.py` 单文件查看切片质量，输出断点率和分片预览。
- **Streamlit Web 界面**：
  - **VS Code Dark+ 深色主题**：精心调配的 Trae 风格配色，代码块使用 Pygments 语法高亮（VS Code 配色方案）。
  - **源码浏览器**：分层树形文件树（每层缩进 + 左边框层级感），支持 Python 语法高亮、行号跳转、完整视图/附近视图切换。
  - **问答 ↔ 源码联动**：问答结果中的代码片段可一键跳转到源码浏览器对应文件与行号。
  - **代码库管理**：支持输入本地路径或上传 ZIP 压缩包动态切换源码目录，无需重启。
  - **React Bug 修复**：采用两阶段渲染机制（进度条 → 结果展示），彻底避免 Streamlit 中的 `removeChild` DOM 错误。

---

## 二、目录结构

```
code_RAG/
├── app.py                    # Streamlit Web UI 主程序
├── main.py                   # 程序入口 + 命令行交互主循环
├── config.py                 # 读取 .env，统一配置项
├── embedding_client.py       # 阿里云 DashScope 向量模型客户端
├── llm_client.py             # 小米 MiMo 大模型客户端
├── chunk_viewer.py           # 切片质量调试工具
├── .env.example              # 环境变量示例
├── requirements.txt          # 依赖清单
├── .streamlit/               # Streamlit 配置目录
│   └── config.toml           # Web 界面全局主题与服务器配置
├── ingestion/                # 数据入库层
│   ├── file_loader.py        # 源码文件加载器（扫描 .py）
│   └── code_splitter.py      # 代码感知切片器
├── retriever/                # 检索层
│   ├── vector_store.py       # 内存向量库（NumPy 手搓）
│   ├── bm25_retriever.py     # BM25 关键词检索
│   ├── multi_retriever.py    # 多路召回融合 + 后处理启发式（核心）
│   └── reranker.py           # Cross-Encoder 重排器
├── test/                     # 单元测试（核心模块均覆盖）
│   ├── test_code_splitter.py
│   ├── test_multi_retriever.py
│   ├── test_reranker.py
│   ├── test_bm25_retriever.py
│   ├── test_vector_store.py
│   └── ...
├── source_code/              # 被索引的源码工程
│   └── file_agent/
│       ├── react_scheduler.py
│       ├── registry.py
│       ├── file_tools.py
│       └── ...
├── doc/
│   ├── README.md             # 本说明文档
│   ├── test_record.md        # 已知短板记录
│   ├── bug_fix.md            # Bug 修复记录
│   ├── interview_case_study.md # 面试案例研究
│   └── leetcode_plan.md      # LeetCode 刷题计划
└── workspace/                # 运行时安全工作目录
```

---

## 三、环境依赖

- **Python** 3.10+
- **核心库**：
  - `streamlit` — Web UI 框架（新增）
  - `requests` — 调大模型 HTTP 接口
  - `python-dotenv` — 加载 `.env` 配置
  - `numpy` — 向量库矩阵运算
  - `rank_bm25` — BM25 关键词检索
  - `dashscope` — 阿里云向量模型 SDK
  - `sentence-transformers` — Cross-Encoder 重排模型
- **外部服务**：
  - 阿里云 DashScope API Key（向量模型，新用户有免费额度）
  - 小米 MiMo API Key（大模型）
  - HuggingFace 镜像（国内下载重排模型用，已配置 `hf-mirror.com`）

安装依赖：

```bash
pip install streamlit requests python-dotenv numpy rank_bm25 dashscope sentence-transformers
```

---

## 四、启动运行步骤

### 方式 A：Streamlit Web 界面（推荐）

1. **进入项目目录并启动**

   ```bash
   cd code_RAG
   python -m streamlit run app.py
   ```

2. **浏览器访问**

   启动后终端会显示类似 `Local URL: http://localhost:8501`，在浏览器中打开即可。

3. **界面功能导航**

   Web 界面包含两个主要标签页：

   - **💬 智能问答**：
     - 点击预设的一键示例问题，或在输入框中键入你的问题，点击「🚀 开始提问」。
     - 系统将展示进度条，依次显示向量化、多路召回、精排、大模型生成等阶段。
     - 回答生成后，下方会列出 Top 代码片段，每个片段提供「📂 在源码浏览器打开」按钮，点击可跳转到源码浏览器查看完整上下文。

   - **📂 源码浏览器**：
     - 左侧为分层树形文件目录结构，点击文件夹图标可展开/折叠，点击 `.py` 文件图标可在右侧查看源码。
     - 右侧代码区支持语法高亮、行号显示。
     - 顶部「跳转到行号」输入框可快速定位到指定行附近（显示目标行前后 30 行，共约 60 行），也可切换到「完整视图」查看整个文件。
     - 侧边栏「📁 代码库管理」支持动态输入本地路径或上传 ZIP 压缩包重建索引。

### 方式 B：命令行交互（CLI）

1. **克隆/进入项目目录**

   ```bash
   cd code_RAG
   ```

2. **安装依赖**（见上一节）

3. **配置环境变量**

   把 `.env.example` 复制为 `.env`，填入你自己的 API Key：

   ```bash
   cp .env.example .env
   ```

   编辑 `.env`，填入 `MIMO_API_KEY` 和 `DASHSCOPE_API_KEY`。

4. **放入要索引的源码**

   把你想问答的 Python 工程放进 `source_code/` 目录（项目已自带一份 `file_agent` 示例工程，可直接用）。

5. **首次启动**（需要联网下载重排模型，约 80MB）

   ```bash
   python main.py
   ```

   启动后会依次执行：扫描源码 → 切片 → 向量化 → 建向量库 → 建 BM25 索引 → 初始化重排器 → 初始化大模型客户端。

6. **开始提问**

   看到 `👉 请输入你的问题:` 提示后输入问题，例如：

   ```
   react_scheduler 的 run 方法是干啥的?
   registry.py 的注册机制怎么实现的?
   这个 agent 依靠哪些工具来完成文件操作?
   ```

   输入 `exit` / `quit` / `q` 退出。

7. **调试切片质量**（可选）

   ```bash
   python chunk_viewer.py source_code/file_agent/registry.py
   ```

   会打印该文件的每个分片起止行、类型、是否断点、断点率。

---

## 五、.env 配置说明

| 变量名 | 说明 | 示例值 |
|---|---|---|
| `MIMO_API_KEY` | 小米 MiMo 大模型 API Key | `your_mimo_api_key` |
| `MIMO_BASE_URL` | MiMo 接口地址 | `https://api.xiaomimimo.com/v1` |
| `MIMO_MODEL` | 模型名 | `mimo-v2.5-pro` |
| `LLM_CLOSE_THINK` | 思考过程开关。`true`=关闭思考省 token，`false`=展示思考过程 | `true` |
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key | `your_dashscope_api_key` |
| `EMBEDDING_MODEL` | 向量模型名 | `qwen3.7-text-embedding` |
| `EMBEDDING_DIMENSIONS` | 向量维度（qwen3.7 支持 1024） | `1024` |
| `HF_ENDPOINT` | HuggingFace 镜像（国内加速） | `https://hf-mirror.com` |
| `HF_HUB_OFFLINE` | 离线模式。模型下载完成后设 `1` 避免每次启动联网检查 | `1` |

---

## 六、Streamlit 主题配置

`.streamlit/config.toml` 已预配置 VS Code Dark+ 风格的深色主题：

```toml
[theme]
base = "dark"
primaryColor = "#0e639c"        # 选中蓝
backgroundColor = "#1e1e1e"      # 编辑器深色底
secondaryBackgroundColor = "#252526"  # 侧边栏/面板深灰
textColor = "#cccccc"            # 文字浅灰
font = "sans serif"
```

如需调整颜色方案，可修改此文件中的颜色值，或在 `app.py` 的 `_inject_theme_css()` 函数中自定义 CSS。

---

## 七、项目亮点

1. **代码结构感知的切片器**：不是按字符数硬切，而是先用 `_parse_line_blocks` 识别 `def`/`class` 边界，过长类按方法切，过长函数兜底滑动窗口 + 括号深度感知（`_find_safe_split_pos`），保证长列表 `TOOL_DEFINITIONS=[...]` 在元素边界切分而非字典中间。实测 `registry.py` / `react_scheduler.py` 断点率从 50% 降到 0%。

2. **多层启发式后处理链**：召回 → 融合加权 → 邻域扩充 → 精排 → 名额预留，五道工序层层补漏。每道工序都解决一个具体问题（关键词稀疏、测试文件噪声、长方法上下文丢失、重排模型误判、目标源码漏召回），不是堆砌概念。

3. **加法 + 乘法混合融合公式**：基础分用加法（向量 0.3 + BM25 0.3 + 关键词 0.4）保证三路投票不被一票否决；乘法增强（关键词乘数 × 源文件乘数 × 类型乘数）让"对的分片"和"错的分片"分数差出几十倍。一个源文件 + 全命中 + function 类型的分片，最终乘数可达 15 倍。

4. **零外部数据库依赖**：向量库用 NumPy 手搓余弦相似度，BM25 用 rank_bm25 内存索引，进程启动即建库，退出即销毁。适合中小规模代码库（几百到几千分片），部署门槛低。

5. **重排分混合策略**：最终重排分 = 0.6×Cross-Encoder 归一分 + 0.4×融合归一分。不全信小模型精排（6 层 MiniLM 对中文问题 vs Python 代码判断力有限），保留 40% 融合分让关键词/源文件启发式继续生效，兜住模型失误。

6. **可观测的调试工具链**：`chunk_viewer.py` 查切片质量，`multi_retriever` 每步打印加权日志，`expand_with_neighbors` 打印邻域扩充新增条数和文件分布，方便定位"为什么没召回"。

7. **专业级 Web 界面**：基于 Streamlit 构建，实现了 VS Code Dark+ 风格主题、分层树形源码浏览、Pygments 语法高亮、问答源码联动跳转等特性。采用两阶段渲染机制解决了 Streamlit 中常见的 React DOM `removeChild` 错误。

---

## 八、当前已知局限

1. **点名提问仍可能漏召回目标分片**：当其他文件的 Cross-Encoder 重排分显著更高时，即使开了源文件名额预留，也无法把目标文件内语义得分劣势的分片送入 Top N。后处理加权只能做偏好引导，不能强行逆转重排模型的相关性判断。
   - 典型案例：问 `registry.py` 注册机制、问 `file_tools.py` 全部工具函数，目标文件后半段核心分片仍可能漏召。

2. **开放式提问（不指明文件名）容易召回"关键词沾边但不含答案"的分片**：点名加权策略不触发，完全依赖原生多路召回 + 重排。`main.py` / `config.py` / `state.py` 这类含高频关键词但非答案所在的文件，容易抢占 Top 3。
   - 典型案例：问 `这个 agent 依靠哪些工具来完成文件操作?`，召回全落在启动/状态/配置文件，没命中 `registry.py` / `file_tools.py`。

3. **重排模型对"中文问题 vs Python 代码"语义匹配能力有限**：默认用的 `cross-encoder/ms-marco-MiniLM-L-6-v2` 是 6 层小模型，训练数据偏英文自然语言问答，对中文 + 代码场景偶有误判（如把测试文件 docstring 误判为比源码更相关）。

4. **向量库纯内存，重启即失**：每次启动都要重新向量化所有分片（调阿里云接口），分片多时启动慢、消耗 API 额度。无持久化。

---

## 九、后续可优化方向

1. **切片器类型识别细化**：当前 chunk_type 把模块级大字典（如 `TOOL_DEFINITIONS=[...]`）标为 `other`，导致注册机制核心代码不被加权。可新增 `module_data` 类型单独加分，从源头解决而非靠后处理兜底。

2. **问题意图识别 + 业务关键词路由**：对高频业务问题（如"注册机制""工具列表"）做定向召回增强，识别到这类意图后直接拉对应文件分片，不依赖文件名提示。

3. **重排模型升级**：换用更大的 Cross-Encoder（如 12 层 MiniLM），或用针对代码训练的重排模型，减少对测试文件 docstring 的误判。

4. **向量库持久化**：把向量和元数据落盘（numpy 数组 `.npy` + JSON），启动时直接加载，避免重复调用向量模型 API。

5. **同文件分片软关联召回**：当目标文件有分片被召回时，把该文件全部分片都纳入候选池（带降权），让重排模型自己挑，彻底解决"目标文件后半段漏召"问题。

6. **残缺分片检测**：切片后加一道校验，若分片开头是缩进的字符串/标量值且上一分片末尾非 `,\n`，回退重切，进一步压低断点率。

7. **Web 界面增强**：
   - 增加代码 diff 对比视图
   - 支持多文件并行问答
   - 记录历史对话，支持会话恢复

---

## 十、技术栈速览

| 模块 | 技术选型 |
|---|---|
| Web 框架 | Streamlit（新增） |
| 主题方案 | VS Code Dark+ 风格（自定义 CSS） |
| 语法高亮 | Pygments（通过 Streamlit 内置代码块） |
| 向量模型 | 阿里云 DashScope `qwen3.7-text-embedding`（1024 维） |
| 关键词检索 | `rank_bm25.BM25Okapi`（中文 2-gram 滑窗分词） |
| 向量库 | NumPy 手搓内存版（余弦相似度） |
| 重排模型 | `cross-encoder/ms-marco-MiniLM-L-6-v2`（Cross-Encoder） |
| 大模型 | 小米 MiMo（OpenAI 协议兼容） |
| 配置管理 | `python-dotenv` + `config.py` |
| 模型镜像 | `hf-mirror.com`（HuggingFace 国内镜像） |
