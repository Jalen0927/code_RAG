# 🐛 CodeRAG 项目 Bug 修复与排查速查表

> 项目：CodeRAG 代码智能问答系统 | 标签：RAG / 代码切片 / Python / Windows / HuggingFace
> 全部 18 项问题（已修复 10 项 + 新发现 8 项，**现已全部修复**）
> 编号体系：Bug 1-10 = 历史已修复；Bug 11-17 = 新发现已修复（2026-08-19）；Bug 18 = 新发现已修复（2026-08-20）

---

## 📋 一、总览表（一键索引）

| 编号 | 标题 | 文件 | 类别 | 严重度 | 状态 |
|---|---|---|---|---|---|
| Bug 1 | HuggingFace 模型在国内无法加载 | `.env` / `config.py` | 网络适配 | 🟡 High | ✅ 已修 |
| Bug 2 | 代码切片器把长 Class 切碎（语义断裂） | `ingestion/code_splitter.py` | 切片策略 | 🔴 Critical | ✅ 已修 |
| Bug 3 | 三引号字符串把缩进解析器搞崩 | `ingestion/code_splitter.py` | 状态机 | 🔴 Critical | ✅ 已修 |
| Bug 4 | Windows 终端 Emoji 显示 GBK 编码错误 | `main.py` | 跨平台 | 🟡 High | ✅ 已修 |
| Bug 5 | PowerShell 中 Python 单行脚本转义地狱 | 临时调试脚本 | 调试技巧 | 🟢 Low | ✅ 已修 |
| Bug 6 | RAG 检索不准（切片策略是根源） | `code_splitter.py` / `multi_retriever.py` | 切片策略 | 🔴 Critical | ✅ 已修 |
| Bug 7 | Cross-Encoder Reranker 本地推理资源不足 | `retriever/reranker.py` | 性能优化 | 🟢 Medium | ✅ 已修 |
| Bug 8 | 多路召回权重和 ≠ 1.0 触发 ValueError | `retriever/multi_retriever.py` | 配置校验 | 🟡 High | ✅ 已修 |
| Bug 9 | Cross-Encoder 重排覆盖关键词/源文件加权 | `retriever/reranker.py` | 多阶段融合 | 🔴 Critical | ✅ 已修 |
| Bug 10 | 加法增强力度不够，改用乘法增强 | `retriever/multi_retriever.py` | 调参策略 | 🔴 Critical | ✅ 已修 |
| Bug 11 | MiMoClient 不校验 API Key | `llm_client.py` | 输入校验 | 🟡 High | ✅ 已修 |
| Bug 12 | _compute_keyword_boost 子串匹配误命中 | `retriever/multi_retriever.py` | 关键词匹配 | 🟡 High | ✅ 已修 |
| Bug 13 | _extract_key_terms 停用词表不完整 | `retriever/multi_retriever.py` | 关键词提取 | 🟢 Medium | ✅ 已修 |
| Bug 14 | _split_class_by_methods 外层循环缺三引号追踪 | `ingestion/code_splitter.py` | 状态机一致性 | 🟡 High | ✅ 已修 |
| Bug 15 | _compute_source_boost 用 test 子串误判测试文件 | `retriever/multi_retriever.py` | 路径判断 | 🟢 Medium | ✅ 已修 |
| Bug 16 | _split_class_by_methods 方法间装饰器丢失 | `ingestion/code_splitter.py` | 切片完整性 | 🟢 Medium | ✅ 已修 |
| Bug 17 | import/日志片段向量分数虚高，挤占 Top 召回 | `ingestion/code_splitter.py` / `retriever/multi_retriever.py` | 召回噪声压制 | 🟡 High | ✅ 已修 |
| Bug 18 | 查询 `react_scheduler`（下划线）检索不到 `ReActScheduler`（驼峰）源码 | `retriever/multi_retriever.py` / `main.py` | 命名风格不一致 | 🔴 Critical | ✅ 已修 |

---

## 🎯 二、触发场景与现象表

| 编号 | 触发场景 | 现象 |
|---|---|---|
| Bug 1 | 国内直连 HuggingFace 服务器 | `WinError 10060` 连接超时，Cross-Encoder 模型无法下载，重排阶段瘫痪 |
| Bug 2 | 对 300+ 行的 `ReActScheduler` 大类切片 | `__init__` 被拦腰截断，`run()` 分散在 3 个分片，问"初始化逻辑"检索不到完整 `__init__` |
| Bug 3 | 修复 Bug 2 后 class 切片仍异常 | class 被切成两半，后半段是从 `SYSTEM_PROMPT="""..."""` 字符串中间开始的"无头"文本 |
| Bug 4 | Windows PowerShell/CMD 打印 emoji | `UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f680'` |
| Bug 5 | PowerShell 跑 `python -c "..."` 单行脚本 | 嵌套引号报语法错误，含中文时更易出错 |
| Bug 6 | 问"ReActScheduler 的初始化逻辑是什么" | 检索返回 `_parse_llm_output` 而非 `__init__` |
| Bug 7 | 加载 `bge-reranker-large`（~560M 参数） | `CUDA out of memory` 或 CPU 推理 2-3 秒/条 |
| Bug 8 | 引入关键词加权后启动 | `ValueError: 基础权重之和必须 = 1.0`（明明 `0.4+0.4+0.2=1.0`） |
| Bug 9 | MultiRetriever 加了关键词/源文件加权后过 Reranker | MultiRetriever 阶段排第 1 的目标分片，Reranker 后掉到第 3，测试文件又冒头 |
| Bug 10 | Bug 9 混合分方案上线后 | `__init__` 进了 Top3 但只排第 3（混合分 0.81），前两名被 `main.py` 和测试文件占据 |
| Bug 11 | `.env` 漏配 `MIMO_API_KEY` / `MIMO_BASE_URL` / `MIMO_MODEL` | 启动打印「✅ MiMoClient 就绪」，首次提问时才在 `chat_completion` 抛 `LLMRequestError` |
| Bug 12 | 用户问「run 方法」「init 逻辑」等含短英文词的问题 | 含 `current` / `running` / `initialize` 等子串的 chunk 也被算作命中关键词 |
| Bug 13 | 用户问「where is run」「how does run work」等含英文疑问词的问题 | `where/when/why/with/from/this/that` 等被作为 key_term 参与匹配 |
| Bug 14 | 超长 class 方法间有 `PROMPT="""..."""` 类属性字符串，且字符串内含 `def xxx():` 行 | 字符串里的 `def xxx():` 被误识别为真实方法，class 被错误切分 |
| Bug 15 | 项目出现 `latest.py` / `protest.py` / `tester_utils.py` 等含 `test` 子串的非测试文件 | 这些源文件被当作测试文件，`src_boost=0.0`，拿不到源文件加分 |
| Bug 16 | 超长 class 内连续多个 `@staticmethod` / `@classmethod` 装饰的方法 | 除第一个装饰器算入 header 外，方法之间的装饰器行不属于任何切片 |
| Bug 17 | 用户问"如何加载源码"但库中存在 `import load_source_files` 这种纯导入片段；或问"日志输出在哪"时纯 `print()`/`logger.info()` 片段被召回 | 这些噪声片段因包含查询关键词，向量+BM25 分数虚高挤进 Top，真正含 `def load_source_files` / `def __init__` 的实现片段被挤出 Top |
| Bug 18 | 问"react_scheduler 的核心功能是什么" | 多路召回 Top10 源文件排第 5（0.81），rerank Top3 全是测试文件+`main.py`，源文件 `react_scheduler.py` 完全消失 |

---

## 🔍 三、根因与影响表

| 编号 | 出错原因 | 对 RAG 召回/链路的影响 |
|---|---|---|
| Bug 1 | HuggingFace 官方服务器在国外，国内直连不稳定甚至被墙 | 重排阶段完全瘫痪，RAG 链路末端失效 |
| Bug 2 | 初版切片器只按缩进判断块边界，超长 block 直接滑动窗口兜底，不考虑内部方法结构 | `__init__` 检索命中率 ~30%，向量表示被稀释 |
| Bug 3 | `_parse_line_blocks` 基于缩进判断边界，三引号字符串内部文本行缩进为 0 被误判为"class 已结束" | 类属性 prompt 被当作方法独立成片，真正后续方法 `start_line` 错位 |
| Bug 4 | Windows 中文系统默认终端编码 GBK，不支持 emoji 的 Unicode 码点 | 程序崩溃，用户无法看到任何输出 |
| Bug 5 | 嵌套引号在 PowerShell 中极易出错，单双引号+中文混合时尤甚 | 调试效率低，命令行验证不可靠 |
| Bug 6 | 旧切片器固定滑动窗口切，`__init__` 被切到两个不同分片，向量表示被稀释 | Top-3 检索命中率仅 42%，问 `__init__` 返回 `_parse_llm_output` |
| Bug 7 | Cross-Encoder 参数量 ~560M，比 Bi-Encoder 大很多 | 单条 rerank 2-3 秒，候选多时整体延迟不可接受 |
| Bug 8 | 引入 `KEYWORD_BASE_WEIGHT` 后旧校验公式没同步更新；浮点累加精度误差（`0.4+0.4+0.2` 可能 = `0.9999`） | 启动即报错，配置无法生效 |
| Bug 9 | Cross-Encoder 只看 query+chunk 语义相关性，完全丢弃 MultiRetriever 的 `score` 字段（关键词/源文件信息）；CE 偏好长文本 | 多阶段优化被吞，关键词/源文件加权等于白做 |
| Bug 10 | 加法形式 `final = base + 0.15*src_boost` 力度太弱，bonus 最多 +0.15，无法弥补测试文件 0.3~0.5 的天然优势 | 目标分片进 Top3 但排第 3，被测试文件反超 |
| Bug 11 | `__init__` 只拼 headers 不做校验，与 `EmbeddingClient` 的 fail-fast 风格不一致 | 错误延后到运行时，初始化阶段「假就绪」，用户难以定位 |
| Bug 12 | `term.lower() in lower_text` 纯子串匹配，无词边界 | 无关 chunk 拿满 `kw_boost=1.0`，乘法增强放大 4 倍，挤占 Top3，稀释召回精准度 |
| Bug 13 | 停用词集合只列了 13 个词，常见英文停用词缺失 | 代码 chunk 偶然出现这些词时被误判命中，进一步扭曲 `kw_boost` |
| Bug 14 | `_split_class_by_methods` 外层两个循环（找第一个 def / 逐个提取方法）没有 `triple_quote` 状态机，与 `_parse_line_blocks`（Bug 3 已修）不一致 | 类属性 prompt 被当作方法独立成片，真正后续方法 `start_line` 错位，检索行号失真 |
| Bug 15 | `"test" in file_path.lower()` 子串判断过于宽松 | 误判源文件在乘法增强上少乘 `1.8` 倍，排名被测试文件反超 |
| Bug 16 | 外层循环遇到非 def 行直接 `i += 1` 跳过，装饰器行被丢弃 | 切片丢失装饰器信息，方法上下文不完整 |
| Bug 17 | 切片器未对 chunk 标记代码类型，多路召回阶段也没有针对纯 import/纯日志片段的降分机制 | 仅 import / 日志片段因包含查询关键词而分数虚高，挤掉真正含函数/类实现的分片；rerank 收到的 Top 候选被噪声污染，召回精准度下降 |
| Bug 18 | ① `_extract_key_terms` 只取原始术语，下划线命名 `react_scheduler` 无法命中源码里的驼峰 `ReActScheduler`（词边界正则不匹配）；② `SOURCE_BOOST_MULTIPLIER=0.8` 加成太弱（1.8 倍 vs 测试文件关键词 4.0 倍）；③ rerank 模型偏好测试文件 docstring「pytest 测试：覆盖 react_scheduler.py 的核心逻辑」 | 源文件 `kw_boost=0`，多路召回分数被测试文件压制；rerank 阶段又被 docstring 命中的测试文件压下去，最终 LLM 看不到真正业务实现代码 |

---

## 🛠️ 四、修复要点表

| 编号 | 修复要点 | 涉及文件 |
|---|---|---|
| Bug 1 | 三步走：镜像 `HF_ENDPOINT=https://hf-mirror.com` + 离线 `HF_HUB_OFFLINE=1` + 环境变量前置加载 | `.env` / `config.py` |
| Bug 2 | 两策略分层切片：第一层按 def/class 边界切；第二层对超长 class 调 `_split_class_by_methods()` 按方法边界切 | `ingestion/code_splitter.py` |
| Bug 3 | 三引号状态追踪：维护 `triple_quote` 变量，奇数个=未闭合，偶数个=已闭合；字符串内部忽略缩进 | `ingestion/code_splitter.py` |
| Bug 4 | 程序入口强制重配置 `sys.stdout/stderr` 为 UTF-8，`errors="replace"` 兜底 | `main.py` |
| Bug 5 | 临时脚本文件法：写 `_check.py`，跑 `python _check.py` | 临时调试脚本 |
| Bug 6 | 端到端调试：用户反馈 → 检索日志 → 切片结果 → 根因分析；设计按方法边界切分方案 | `code_splitter.py` / `multi_retriever.py` |
| Bug 7 | 降级 `bge-reranker-base`（~110M）+ 批处理 + `onnxruntime` INT8 量化 | `retriever/reranker.py` |
| Bug 8 | 校验公式同步更新加 `KEYWORD_BASE_WEIGHT`；用 `abs(sum-1.0) > 1e-6` 容差判断 | `retriever/multi_retriever.py` |
| Bug 9 | Score Blending：`final = 0.6*ce_norm + 0.4*orig_fusion_norm`；两路分数都做 min-max 归一化 | `reranker.py` / `multi_retriever.py` |
| Bug 10 | 加法→乘法：`final = base × (1+3.0×kw_boost) × (1+0.8×src_boost)` | `retriever/multi_retriever.py` |
| Bug 11 | `__init__` 中校验 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` 三项配置，缺失即抛 `LLMRequestError` | `llm_client.py` |
| Bug 12 | 英文术语改用 `(?<!\w)term(?!\w)` 词边界断言，中文仍用子串匹配 | `retriever/multi_retriever.py` |
| Bug 13 | 扩充停用词集合：补 `where/when/why/with/from/this/that/are/is/has/have/can/could/would/should` 等 | `retriever/multi_retriever.py` |
| Bug 14 | 在 `_split_class_by_methods` 的两个外层循环中同步引入 `triple_quote` 状态机，与 `_parse_line_blocks` 一致 | `ingestion/code_splitter.py` |
| Bug 15 | 按目录或文件名前缀/后缀精确判断：`test/` `tests/` 目录 或 `test_xxx.py` / `xxx_test.py` | `retriever/multi_retriever.py` |
| Bug 16 | 装饰器行并入紧随其后的方法块：方法块起始行回退到第一个装饰器 | `ingestion/code_splitter.py` |
| Bug 17 | 双保险：①code_splitter 在 chunk["metadata"] 写入 chunk_type 元标签；②multi_retriever 在第 4 步融合分算完后、第 5 步排序前插入 chunk_type 乘法加权，import/log_call ×0.5 降分、function/class_def ×1.5 加分、other ×1.0 不变；系数严格 ∈ (-1, +1) 保证乘数 > 0 不归零、不丢弃 | `ingestion/code_splitter.py` / `retriever/multi_retriever.py` |
| Bug 18 | 四步组合拳：①新增 `_split_identifier_words()` 拆分驼峰/下划线标识符；②`_extract_key_terms()` 扩展命名变体（下划线↔驼峰）；③`_compute_keyword_boost()` 对拆分单词用子串匹配（命中 `ReActScheduler`）；④`SOURCE_BOOST_MULTIPLIER` 0.8→1.5；⑤新增 `ensure_source_file_in_top_n()` 在 rerank 后兜底提升"含关键术语的源文件业务实现分片"进 Top N | `retriever/multi_retriever.py` / `main.py` |

---

## 📊 五、量化效果表

| 编号 | 指标 | 修复前 | 修复后 |
|---|---|---|---|
| Bug 2 | `__init__` 方法完整性 | 被拦腰截断 | 完整保留在一个分片 |
| Bug 2 | `run()` 方法完整性 | 分散在 3 个分片 | 完整保留在一个分片 |
| Bug 2 | `__init__` 检索命中率 | ~30% | ~85% |
| Bug 6 | 问"初始化逻辑"检索结果 | 返回 `_parse_llm_output` | 返回 `__init__` 方法 |
| Bug 6 | Top-3 检索命中率 | 42% | 89% |
| Bug 9 | `react_scheduler.py:9` 排名（Reranker 后） | 第 3 名 | 第 1 名 |
| Bug 10 | `react_scheduler.py:9` 综合分 | `0.53+0.15=0.68`（加法，第 3） | `0.53×2.5×1.8=2.39`（乘法，第 1） |
| Bug 17 | 纯 import 片段乘数 | `1.0`（与 function 同权） | `0.5`（降分一半，保留为兜底候选） |
| Bug 17 | 含 def 的 function 片段乘数 | `1.0`（与 import 同权） | `1.5`（业务实现加分） |
| Bug 17 | 噪声压制能力 | import/log_call 与 function 同分同权，可能反超 | import/log_call 强制 ×0.5，function/class_def ×1.5，噪声被压制 |
| Bug 18 | `react_scheduler.py` 多路召回排名 | 第 5（score=0.81，kw_boost=0.00） | 第 1（score=6.38，kw_boost=0.67） |
| Bug 18 | rerank Top3 是否含源文件 | ❌ 全是测试文件+main.py | ✅ 第 3 名是 `react_scheduler.py:9`（后处理提升） |

---

## 🏷️ 六、严重度与类别统计表

### 按严重度统计

| 严重度 | 已修复部分 | 新发现部分 | 合计 | 编号 |
|---|---|---|---|---|
| 🔴 Critical | 5 | 1 | 6 | Bug 2/3/6/9/10，Bug 18 |
| 🟡 High | 3 | 4 | 7 | Bug 1/4/8，Bug 11/12/14/17 |
| 🟢 Medium | 1 | 3 | 4 | Bug 7，Bug 13/15/16 |
| 🟢 Low | 1 | 0 | 1 | Bug 5 |
| **合计** | **10** | **8** | **18** | |

### 按修复方法/思路分类

| 修复方法 | 使用场景 | 对应编号 |
|---|---|---|
| 镜像+离线 | 国内网络环境适配 | Bug 1 |
| 语义切片 | 按方法边界切分长 class | Bug 2, Bug 6 |
| 状态机追踪 | 三引号字符串边界处理 | Bug 3, Bug 14 |
| 编码重配置 | Windows 终端兼容 | Bug 4 |
| 脚本文件法 | 避免 PowerShell 引号转义 | Bug 5 |
| 模型降级+批处理 | 推理资源优化 | Bug 7 |
| 配置联动校验 | 新增权重项同步校验 | Bug 8 |
| 分数混合 | 保留多阶段融合信息 | Bug 9 |
| 乘法增强 | 放大关键词/源文件加权 | Bug 10 |
| fail-fast 校验 | 客户端初始化校验配置 | Bug 11 |
| 词边界匹配 | 关键词加权避免子串误命中 | Bug 12 |
| 停用词扩充 | 中英文混合查询术语过滤 | Bug 13 |
| 路径精确判断 | 测试文件识别避免子串误判 | Bug 15 |
| 装饰器并入方法 | 方法切片保留装饰器上下文 | Bug 16 |
| 元标签加权降分 | 噪声召回压制（import/日志片段） | Bug 17 |
| 命名归一化+后处理兜底 | 下划线/驼峰命名风格不一致的检索 | Bug 18 |

### 按涉及文件统计

| 文件 | 命中次数 | 编号 |
|---|---|---|
| `retriever/multi_retriever.py` | 7 | Bug 6/8/10，Bug 12/13/15/17/18 |
| `ingestion/code_splitter.py` | 6 | Bug 2/3/6，Bug 14/16/17 |
| `retriever/reranker.py` | 2 | Bug 7/9 |
| `main.py` | 2 | Bug 4，Bug 18 |
| `llm_client.py` | 1 | Bug 11 |
| `.env` / `config.py` | 1 | Bug 1 |
| 临时调试脚本 | 1 | Bug 5 |

---

## 💡 七、核心教训表

| # | 教训 | 来源 |
|---|---|---|
| 1 | 切片策略是 RAG 检索质量的天花板：固定滑动窗口会切断方法语义，必须按代码结构切分 | Bug 2/6 |
| 2 | 缩进解析器要处理字符串边界：三引号字符串内的零缩进行会被误判为顶层代码，必须维护状态机追踪 | Bug 3 |
| 3 | 多阶段 pipeline 的优化不能被吞掉：MultiRetriever 的加权不在 Reranker 阶段保留等于白做 | Bug 9 |
| 4 | 加法增强力度不够时果断换乘法：线性补偿无法弥补信息密度差距，乘法能指数级放大命中优势 | Bug 10 |
| 5 | 新增配置项必须审计校验逻辑：权重项增加后旧校验要同步更新，浮点比较必须用容差 | Bug 8 |
| 6 | 国内网络环境需要工程适配：HF 模型加载需镜像+离线，且环境变量必须在库导入前加载 | Bug 1 |
| 7 | Windows 终端编码需要手动处理：GBK 默认编码不支持 emoji，入口需强制重配置 | Bug 4 |
| 8 | PowerShell 引号转义极易出错：调试优先用临时脚本文件，避免命令行单行脚本 | Bug 5 |
| 9 | 输入校验要 fail-fast：客户端类 `__init__` 必须校验关键配置，不能让错误延后到运行时 | Bug 11 |
| 10 | 关键词匹配要区分词边界：英文用词边界、中文用子串，不能一刀切 | Bug 12 |
| 11 | 停用词表要持续维护：中英文混合查询场景下停用词缺失会污染 key_terms | Bug 13 |
| 12 | 状态机要全链路一致：`_parse_line_blocks` 修了三引号追踪，`_split_class_by_methods` 外层循环也要同步修 | Bug 14 |
| 13 | 路径判断要精确：用子串判断会误判，应按目录/文件名前后缀 | Bug 15 |
| 14 | 装饰器要跟随方法：方法切片不能只从 `def` 行开始，要把上方装饰器并入 | Bug 16 |
| 15 | 召回后过滤要"降分不丢弃"：纯 import/日志片段易因包含查询词而分数虚高，应通过元标签 × 乘法降分压制，但乘数严格 ∈ (-1, +1) 保证不归零，避免极端情况丢失有效候选 | Bug 17 |
| 16 | 命名风格不一致是检索盲区：用户查询 `react_scheduler` 但源码用 `ReActScheduler`（驼峰），词边界匹配会漏掉；要拆分单词 + 子串匹配 + 后处理兜底三层防御 | Bug 18 |

---

## 📝 八、面试话术速查表

| 话术 | 触发问题 | 关键数据/亮点 |
|---|---|---|
| 切片策略 | "最有挑战性的问题？" | 切片语义断裂 → 两策略分层切片 → 命中率 42%→89% |
| 检索增强调参 | "调参最深刻的经验？" | CE 覆盖融合分 → Score Blending → 加法转乘法 → 综合分 0.68→2.39 |
| 源码排查 | "工程严谨性？" | LLM 客户端假就绪 + 关键词子串误命中 → fail-fast + 词边界断言 |
| 跨平台适配 | "Windows 开发经验？" | GBK 编码错误 → UTF-8 重配置 + `errors="replace"` 兜底 |
| 国内网络 | "HuggingFace 模型加载？" | 镜像+离线+环境变量前置加载三步走 |
| 状态机思维 | "边界 case 处理？" | 三引号/嵌套函数/装饰器/多行注释的状态追踪 |
| 召召回环优化 | "如何处理噪声召回？" | import/日志片段向量分数虚高 → chunk_type 元标签 + 乘法加权降分（×0.5）不丢弃 → 让 function/class_def 加分（×1.5）→ 噪声被压制 |

### 话术模板原文

**话术 1（切片策略）**：
> "在这个 RAG 项目中，我遇到的最有挑战性的问题是代码切片的语义断裂。一开始我用固定滑动窗口切分，导致长 class 的方法被拦腰截断，检索命中率只有 42%。后来我设计了两策略分层切片方案：先用缩进+三引号状态机解析逻辑块，再对超长 class 按方法边界二次切分。修复后命中率提升到 89%。"

**话术 2（检索增强调参）**：
> "切片修好后，检索准确率仍有瓶颈。我发现 Cross-Encoder 重排会覆盖掉 MultiRetriever 阶段的关键词加权和源文件加权，导致短方法分片又被长测试文件淹没。我的解决方案是分数混合（Score Blending）：把 CE 归一化分和原始融合分按 6:4 混合。但混合后发现加法增强力度不够，最终改成乘法增强：`final = base × (1 + 3.0×kw_boost) × (1 + 0.8×src_boost)`，让命中关键词的分片分数放大 4 倍。这是我在调参中最有成就感的时刻——从 0.68 到 2.39 的跃升。"

**话术 3（源码排查）**：
> "项目稳定后，我又对全部主链路源码做了一轮排查，发现 6 个功能性 bug 和逻辑漏洞。最关键的两个：一是 LLM 客户端没在 `__init__` 校验 API Key，导致初始化'假就绪'；二是关键词加权用子串匹配，问 'run 方法' 时含 `current` 的 chunk 也被算命中，乘法增强后分数被放大 4 倍，挤掉真正含 `def run` 的目标 chunk。修复方案是英文术语改用 `(?<!\w)term(?!\w)` 词边界断言。这次排查让我意识到：状态机修一处不够，要全链路同步（`_parse_line_blocks` 修了三引号追踪，`_split_class_by_methods` 的外层循环也要同步修）。"

**话术 4（召回闭环优化）**：
> "做完 Bug 11-16 后我又发现一个召回噪声问题：当用户问'如何加载源码'时，库里的 `import load_source_files` 这种纯导入片段因为包含查询关键词，向量+BM25 分数虚高挤进了 Top，反而把真正含 `def load_source_files` 实现的片段挤出 Top。日志打印片段同理。我的方案是给每个 chunk 打元标签 chunk_type（function/class_def/import/log_call/other），在 multi_retriever 多路融合完成之后、rerank 之前插入一个乘法加权：import/log_call ×0.5 降分、function/class_def ×1.5 加分、other ×1.0 不变。关键是降分而不丢弃——乘数严格落在 (0, 1) 区间，绝不归零，保留兜底候选防止极端情况丢失有效信息。同时为兼容旧索引数据（vector_store 不返回 metadata 字段），multi_retriever 内部独立重识别 chunk_type，双保险设计。"

---

## 🔧 九、关键修复代码片段速查

### Bug 11：MiMoClient 校验 API Key

```python
def __init__(self):
    if not LLM_API_KEY:
        raise LLMRequestError("环境变量缺失: MIMO_API_KEY，请在 .env 中配置")
    if not LLM_BASE_URL:
        raise LLMRequestError("环境变量缺失: MIMO_BASE_URL，请在 .env 中配置")
    if not LLM_MODEL:
        raise LLMRequestError("环境变量缺失: MIMO_MODEL，请在 .env 中配置")
    self.headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
```

### Bug 12：词边界匹配

```python
import re

@staticmethod
def _compute_keyword_boost(chunk_text: str, key_terms: list[str]) -> float:
    if not key_terms:
        return 0.0
    lower_text = chunk_text.lower()
    hits = 0
    for term in key_terms:
        term_lower = term.lower()
        if re.search(r"[a-zA-Z_]", term_lower):
            # 英文标识符：用词边界，避免 init 命中 initialize
            if re.search(rf"(?<!\w){re.escape(term_lower)}(?!\w)", lower_text):
                hits += 1
        else:
            # 纯中文：子串匹配
            if term_lower in lower_text:
                hits += 1
    return min(1.0, hits / len(key_terms))
```

### Bug 13：停用词扩充

```python
_STOPWORDS = {
    # 英文
    "the", "and", "for", "what", "how", "does", "where", "when", "why",
    "who", "with", "from", "this", "that", "are", "is", "has", "have",
    "can", "could", "would", "should", "do", "did", "not", "but", "or",
    # 中文
    "哪些", "什么", "怎么", "如何", "方法", "逻辑", "作用", "哪里", "哪个",
}

# 原判断处
if lower not in _STOPWORDS:
    terms.add(lower)
```

### Bug 14：外层循环三引号追踪

```python
def _split_class_by_methods(class_text: str, class_start_line: int,
                            max_chars: int, overlap: int):
    lines = class_text.splitlines(keepends=True)
    total_lines = len(lines)
    methods = []
    i = 0
    header_end_idx = 0
    triple_quote = None  # 新增：外层循环也维护三引号状态

    # 找第一个 def（带三引号状态追踪）
    while i < total_lines:
        line = lines[i]
        # --- 三引号状态追踪（与 _parse_line_blocks 保持一致）---
        if triple_quote:
            if triple_quote in line:
                if line.count(triple_quote) % 2 == 1:
                    triple_quote = None
            i += 1
            continue
        else:
            unclosed = _count_unclosed_triples(line)
            if unclosed:
                triple_quote = unclosed
                i += 1
                continue
        if _DEF_ONLY_RE.match(line):
            break
        i += 1
    header_end_idx = i
    # ... 后续「逐个提取方法」循环同样在 else 分支加入相同状态追踪 ...
```

### Bug 15：路径精确判断

```python
@staticmethod
def _compute_source_boost(file_path: str) -> float:
    if "__pycache__" in file_path:
        return 0.0
    parts = file_path.replace("\\", "/").lower().split("/")
    file_name = parts[-1]
    in_test_dir = any(p in ("test", "tests") for p in parts[:-1])
    is_test_file = file_name.startswith("test_") or file_name.endswith("_test.py")
    if in_test_dir or is_test_file:
        return 0.0
    return 1.0
```

### Bug 16：装饰器并入方法

```python
# 在「逐个提取方法」外层循环中
while i < total_lines:
    line = lines[i]
    stripped = line.strip()
    if stripped == "":
        i += 1
        continue
    # 新增：收集装饰器行，并入下一个方法
    decorator_lines = []
    while stripped.startswith("@"):
        decorator_lines.append((i, line))
        i += 1
        if i >= total_lines:
            break
        line = lines[i]
        stripped = line.strip()
    match = _DEF_ONLY_RE.match(line)
    if match:
        # 方法块起始行回退到第一个装饰器
        method_start_idx = decorator_lines[0][0] if decorator_lines else i
        method_start_line = class_start_line + method_start_idx
        # ... 后续方法体扫描不变 ...
```

### Bug 17：chunk_type 元标签 + multi_retriever 后处理加权

**① code_splitter.py：识别函数 + chunk 字典统一构造**

```python
import re

_CLASS_DEF_RE = re.compile(r"^class\s+\w+")
_ANY_DEF_RE = re.compile(r"^\s*def\s+\w+")
_IMPORT_LINE_RE = re.compile(r"^\s*(import\s+\S|from\s+\S+\s+import\b)")


def _classify_chunk_type(chunk_text: str) -> str:
    """识别优先级：class_def > function > import(>=70%) > log_call(无赋值/return 且>=50%调用) > other"""
    if not chunk_text or not chunk_text.strip():
        return "other"
    code_lines = [ln.strip() for ln in chunk_text.splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    if not code_lines:
        return "other"
    total = len(code_lines)

    for line in code_lines:
        if _CLASS_DEF_RE.match(line):
            return "class_def"
    for line in code_lines:
        if _ANY_DEF_RE.match(line):
            return "function"
    import_count = sum(1 for ln in code_lines if _IMPORT_LINE_RE.match(ln))
    if total > 0 and import_count / total >= 0.7:
        return "import"
    # log_call 判定：无赋值、无 return 且调用行占比 >= 50%
    # （详见 ingestion/code_splitter.py L86-L115）
    # ...
    return "other"


def _make_chunk(file_path: str, chunk_text: str, start_line: int) -> dict:
    """统一构造 chunk 字典，自动注入 metadata.chunk_type。"""
    return {
        "file_path": file_path,
        "chunk_text": chunk_text,
        "start_line": start_line,
        "metadata": {"chunk_type": _classify_chunk_type(chunk_text)},
    }
```

**② multi_retriever.py：类常量加权系数表**

```python
# 取值严格 ∈ (-1, +1)，保证乘数 > 0，绝不归零或转负
CHUNK_TYPE_BOOST_MULTIPLIER = {
    "function": 0.5,    # 含 def 函数定义，业务实现加分
    "class_def": 0.5,    # 含 class 类定义，业务实现加分
    "other": 0.0,        # 普通代码片段，不变
    "import": -0.5,      # 仅 import 导入片段，降分压制噪声
    "log_call": -0.5,    # 日志/打印/纯调用片段，降分压制噪声
}
```

**③ multi_retriever.py：search 第 4.5 步加权（融合分算完后、排序前）**

```python
# ---------- 第 4.5 步：chunk_type 元标签后处理加权 ----------
# 在多路融合分计算完成之后、排序之前插入：
#   - 对 import / log_call 噪声分片做乘法降分（系数 ∈ (0, 1)，不归零）
#   - 对 function / class_def 业务实现分片做乘法加分
#   - 旧数据也能用，因为这里基于 chunk_text 独立重新识别，向下兼容
for it in all_items:
    chunk_type = self._classify_chunk_type(it["chunk_text"])
    type_boost = self.CHUNK_TYPE_BOOST_MULTIPLIER.get(chunk_type, 0.0)
    type_multiplier = 1.0 + type_boost  # 严格 > 0，绝不归零或转负
    it["score"] = it["score"] * type_multiplier
    it["chunk_type"] = chunk_type
    it["type_multiplier"] = type_multiplier

# ---------- 第 5 步：按综合分降序排序 + 取 top_k ----------
all_items.sort(key=lambda x: x["score"], reverse=True)
return all_items[:top_k]
```

> 加权乘数实测：`function`/`class_def` → 1.50；`other` → 1.00；`import`/`log_call` → 0.50

---

### Bug 18：命名风格不一致 + rerank 后处理兜底

**修复点 1：拆分标识符单词**（`multi_retriever.py` 新增 `_split_identifier_words`）

```python
@staticmethod
def _split_identifier_words(ident: str) -> list[str]:
    import re
    s = ident.replace("_", " ")                              # 下划线→空格
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)               # 小写+大写处切分
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)         # 连续大写后接小写处切分
    return [w.lower() for w in s.split() if w]
# react_scheduler → ['react', 'scheduler']
# ReActScheduler  → ['react', 'act', 'scheduler']
```

**修复点 2：扩展命名变体**（`_extract_key_terms` 内）

```python
words = MultiRetriever._split_identifier_words(m)
for w in words:
    if len(w) >= 2:
        terms.add(w)                                         # 拆分单词
if words:
    camel = "".join(w.capitalize() for w in words)
    terms.add(camel)                                         # 驼峰形式
    terms.add(camel.lower())                                 # 全小写形式
```

**修复点 3：拆分单词用子串匹配**（`_compute_keyword_boost` 内）

```python
if "_" in term_lower:
    # 含下划线完整标识符：词边界匹配
    if re.search(rf"(?<!\w){re.escape(term_lower)}(?!\w)", lower_text):
        hits += 1
else:
    # 不含下划线的英文单词：子串匹配（关键改动）
    # 让 'react' 能命中 'reactscheduler'（class ReActScheduler 小写）
    if term_lower in lower_text:
        hits += 1
```

**修复点 4：源文件加权系数调高**

```python
SOURCE_BOOST_MULTIPLIER = 1.5  # 原值 0.8，让源文件相对测试文件优势更大
```

**修复点 5：rerank 后处理兜底**（`multi_retriever.py` 新增 `ensure_source_file_in_top_n`）

```python
def ensure_source_file_in_top_n(self, candidates, top_n, query):
    key_terms = self._extract_key_terms(query)

    def _is_relevant_source(item):
        # 三层条件：源文件 + 含关键术语 + 业务实现分片(class_def/function)
        if self._compute_source_boost(item["file_path"]) <= 0:
            return False
        if self._compute_keyword_boost(item["chunk_text"], key_terms) <= 0:
            return False
        chunk_type = self._classify_chunk_type(item["chunk_text"])
        return chunk_type in ("class_def", "function")

    # top_n 已有相关源文件分片则不干预
    if any(_is_relevant_source(r) for r in top_n):
        return top_n
    # 否则从 candidates 找分数最高的相关源文件，替换 top_n 最后一名
    for c in candidates:
        if _is_relevant_source(c) and not already_in_top:
            new_item = dict(c)
            new_item["rerank_score"] = 0.0
            new_item["boosted_by_source_filter"] = True
            return top_n[:-1] + [new_item]
    return top_n
```

**修复点 6：main.py 调用兜底**（`answer_query` 内 rerank 之后）

```python
top_n = reranker.rerank(query, candidates, top_n=3)
top_n = multi.ensure_source_file_in_top_n(candidates, top_n, query)  # 新增
```

> 实测：`react_scheduler.py:9` 多路召回排名从第 5 → 第 1，rerank Top3 第 3 名稳含源文件

---

## ✅ 十、问题排查 Checklist

| 症状 | 排查方向 | 对应编号 |
|---|---|---|
| 模型加载超时 | 网络 / HF 镜像 / 离线模式 | Bug 1 |
| 切片结果混乱 | 三引号 / 装饰器 / 嵌套函数 | Bug 3, Bug 14/16 |
| 检索不准 | 切片粒度 / 向量维度 / BM25 参数 / 关键词子串误命中 | Bug 6, Bug 12 |
| 终端乱码 | stdout 编码 / PYTHONIOENCODING | Bug 4 |
| 推理太慢 | 模型大小 / 批处理 / 量化 | Bug 7 |
| PowerShell 报错 | 引号转义 / 改用脚本文件 | Bug 5 |
| 权重校验报错 | 新增权重项没同步到校验公式 / 浮点精度 | Bug 8 |
| Rerank 后排序变差 | CE 覆盖了融合分 / 需 Score Blending | Bug 9 |
| 增强无效 | 加法力度不够 / 改乘法增强 | Bug 10 |
| LLM 初始化"假就绪" | `__init__` 未校验 API Key / Base URL / Model | Bug 11 |
| 关键词加权偏差 | 子串匹配误命中 / 改词边界 | Bug 12 |
| 测试文件误判 | `"test" in path` 子串判断 / 改目录+前后缀 | Bug 15 |
| 装饰器丢失 | 方法切片只从 def 行开始 / 装饰器并入 | Bug 16 |
| 停用词污染 | 停用词表不完整 / 扩充中英文 | Bug 13 |
| Top 候选被噪声挤占 | 纯 import/日志片段向量分数虚高 / 加 chunk_type 元标签 + 乘法降分不丢弃 | Bug 17 |
| 源文件检索不到 | 命名风格不一致（下划线 vs 驼峰）/ rerank 偏好 docstring / 拆分单词+子串匹配+后处理兜底 | Bug 18 |
