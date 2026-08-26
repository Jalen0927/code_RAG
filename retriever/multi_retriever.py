"""
retriever/multi_retriever.py — 多路召回融合器（MultiRetriever）。

职责：
  - 把"向量召回"和"BM25 关键词召回"两路结果合并，输出一份按综合分数排序的候选列表
  - 多层启发式后处理：关键词命中加权、源文件加权、chunk_type 类型加权
  - 同文件邻域扩充：召回后追加每个候选分片的前后邻居，让重排模型看到完整上下文
  - 源文件名额预留：重排后若 Top N 漏掉目标源文件，强制拎一个塞进末位兜底

为什么不能直接把两路分数相加？
  - 向量分数 = 余弦相似度，范围 [-1, 1]
  - BM25 分数 = 词频加权统计值，范围 [0, +∞)，常见 0~10 几十都有可能
  - 直接相加：BM25 永远会压过向量分数，权重形同虚设
  - 解决方案：先各自"归一化"到 [0, 1] 区间，再按权重加权求和
"""

from __future__ import annotations

import os
import sys

# 把项目根目录加入搜索路径，方便直接运行本文件做自测
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.bm25_retriever import Bm25Retriever
from retriever.vector_store import InMemoryVectorStore


class MultiRetriever:
    """
    多路召回融合：向量召回 + BM25 召回 → 去重 → 归一化 → 加权融合。

    使用前提：
        - vector_store 已经 add 过向量
        - bm25_retriever 已经 build_index 过
        - 两者存的 chunk 必须是同一批（file_path + start_line 对得上）
    """

    # ———————— 可调常量 ————————
    # 向量召回召回数量（通常要拉得比 top_k 大，给融合留余量）
    VECTOR_CANDIDATE_K = 20
    # BM25 召回数量
    BM25_CANDIDATE_K = 20
    # 基础融合权重（向量 + BM25 + 关键词基础分 = 1.0）
    VECTOR_WEIGHT = 0.3
    BM25_WEIGHT = 0.3
    KEYWORD_BASE_WEIGHT = 0.4
    # 关键词匹配乘法增强系数（命中越多增强越大，0 = 不增强）
    KEYWORD_BOOST_MULTIPLIER = 3.0
    # 源文件乘法增强系数（非测试文件加分，0 = 不增强）
    # 调高 0.8 → 1.5：让源文件相对测试文件有更大优势，
    # 抵消测试文件靠"文件名包含关键词"产生的虚假高分
    SOURCE_BOOST_MULTIPLIER = 1.5

    # 测试文件降权系数（绝对值，从乘数里减去）
    # 让测试文件乘数从 1.0 降到 0.5，进一步压制 docstring 关键词命中优势
    TEST_PENALTY_MULTIPLIER = 0.5

    # chunk_type 元标签乘法增强系数表：
    #   final_score *= (1.0 + 系数)
    #   正数 = 加分（function / class_def 业务实现片段优先）
    #   负数 = 降分（import / log_call 噪声片段压制，但乘数仍 > 0，不丢弃）
    #   0    = 不变（other 普通片段保持原状）
    # 设计原则：
    #   - 取值严格落在 (-1, +1)，保证乘数 > 0，绝不直接归零或变负
    #   - import/log_call 取 -0.5（降一半，保留为兜底候选）
    #   - function/class_def 取 +0.5（业务实现加分）
    CHUNK_TYPE_BOOST_MULTIPLIER = {
        "function": 0.5,    # 含 def 函数定义，业务实现加分
        "class_def": 0.5,   # 含 class 类定义，业务实现加分
        "other": 0.0,       # 普通代码片段，不变
        "import": -0.5,     # 仅 import 导入片段，降分压制噪声
        "log_call": -0.5,   # 日志/打印/纯调用片段，降分压制噪声
    }
    # top_k 上限
    MAX_ALLOWED_TOP_K = 1000
    # 同文件加权增量（邻域扩充后，对和源分片同文件的候选加轻微偏好）
    # 注意：数值不能过大，防止强行把无关分片顶到前排引入噪声
    SAME_FILE_BONUS = 0.04

    # 业务关键词路由映射：把用户查询中的业务关键词 → 对应源文件
    # 作用：当用户问的是"工具"时，直接给 registry.py / file_tools.py 的分片加一个小 boost，
    #       避免被 main.py / config.py 里的"工具"关键词高频噪声抢占前排。
    # 设计原则：
    #   - 只做"轻推"（boost 0.15），不做"硬路由"，不压制 Cross-Encoder 的语义判断
    #   - 只对源文件加分，测试文件不加
    #   - 关键词用小写做精确子串匹配，覆盖中英文表达差异
    BUSINESS_KEYWORD_ROUTE = {
        # 工具注册 / 文件操作
        "tool": ["registry.py", "file_tools.py"],
        "工具": ["registry.py", "file_tools.py"],
        "注册": ["registry.py"],
        "registry": ["registry.py"],
        "file": ["file_tools.py"],
        "文件": ["file_tools.py"],
        "加载": ["file_tools.py"],
        "load": ["file_tools.py"],
        # 调度 / 执行流程
        "scheduler": ["react_scheduler.py"],
        "调度": ["react_scheduler.py"],
        "react": ["react_scheduler.py"],
        "loop": ["react_scheduler.py"],
        # 大模型调用
        "llm": ["llm_client.py"],
        "大模型": ["llm_client.py"],
        "模型": ["llm_client.py"],
        "prompt": ["llm_client.py"],
        # 状态管理
        "state": ["state.py"],
        "状态": ["state.py"],
        # 沙箱 / 安全
        "sandbox": ["sandbox.py"],
        "沙箱": ["sandbox.py"],
        # 配置
        "config": ["config.py"],
        "配置": ["config.py"],
        # 入口 / 主逻辑
        "main": ["main.py"],
        "入口": ["main.py"],
        "agent": ["main.py", "react_scheduler.py"],
    }

    # 业务路由 boost 系数：小值即可（加法，不是乘法），
    # 足够把目标文件分片推到候选中前排，但不压制 rerank 语义判断
    BUSINESS_ROUTE_BOOST = 0.15
    # ———————————————————————

    def __init__(self, vector_store: InMemoryVectorStore,
                 bm25_retriever: Bm25Retriever,
                 all_chunks: list = None):
        """
        参数:
            vector_store:   已经 add 过向量的 InMemoryVectorStore 实例
            bm25_retriever: 已经 build_index 过的 Bm25Retriever 实例
            all_chunks:     全量分片列表（可选），用于"同文件邻域扩充"功能。
                            传入后会构建 {file_path: [chunk1, chunk2, ...]} 映射，
                            按 start_line 排序，方便查找某个分片的前后邻居。
        """
        # 类型检查（防止用户传错东西）
        if not isinstance(vector_store, InMemoryVectorStore):
            raise TypeError(
                f"vector_store 必须是 InMemoryVectorStore 实例，"
                f"实际 {type(vector_store).__name__}"
            )
        if not isinstance(bm25_retriever, Bm25Retriever):
            raise TypeError(
                f"bm25_retriever 必须是 Bm25Retriever 实例，"
                f"实际 {type(bm25_retriever).__name__}"
            )

        # 校验基础权重之和（向量 + BM25 + 关键词基础分 = 1.0）
        base_total = self.VECTOR_WEIGHT + self.BM25_WEIGHT + self.KEYWORD_BASE_WEIGHT
        if abs(base_total - 1.0) > 1e-6:
            raise ValueError(
                f"基础权重之和必须 = 1.0，当前 VECTOR_WEIGHT={self.VECTOR_WEIGHT} + "
                f"BM25_WEIGHT={self.BM25_WEIGHT} + KEYWORD_BASE_WEIGHT={self.KEYWORD_BASE_WEIGHT} = "
                f"{base_total}"
            )

        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

        # 构建"文件 → 分片列表"映射，用于同文件邻域扩充
        # 结构: {file_path: [chunk1, chunk2, ...]}，每个列表按 start_line 升序排列
        self._file_chunks_map = {}
        if all_chunks:
            for c in all_chunks:
                fp = c["file_path"]
                if fp not in self._file_chunks_map:
                    self._file_chunks_map[fp] = []
                self._file_chunks_map[fp].append(c)
            # 每个文件的分片列表按 start_line 排序，保证"前后邻居"就是相邻分片
            for fp in self._file_chunks_map:
                self._file_chunks_map[fp].sort(key=lambda x: x["start_line"])

    # ============================================================
    #  工具方法
    # ============================================================

    @staticmethod
    def _min_max_normalize(scores: list[float]) -> list[float]:
        """
        最小-最大归一化：把一组分数压到 [0, 1] 区间。

        公式：x' = (x - min) / (max - min)

        边界处理：
          - 空列表 → 返回空列表
          - 全部相等（max == min）→ 全部记为 1.0（避免除零）
            （这种情况说明所有文档相关性一致，给满分让它们都能保留下来）
        """
        if not scores:
            return []

        min_s = min(scores)
        max_s = max(scores)

        if max_s - min_s < 1e-12:
            return [1.0 for _ in scores]

        return [(s - min_s) / (max_s - min_s) for s in scores]

    @staticmethod
    def _make_key(file_path: str, start_line: int) -> str:
        """生成去重用的唯一键：file_path + '|' + start_line。"""
        return f"{file_path}|{start_line}"

    @staticmethod
    def _split_identifier_words(ident: str) -> list[str]:
        """
        把一个标识符拆分为单词列表，同时支持下划线命名和驼峰命名。

        用于解决"查询用下划线、源码用驼峰"导致的命名不一致问题：
          react_scheduler → ['react', 'scheduler']
          ReActScheduler  → ['react', 'act', 'scheduler']
          _build_system_prompt → ['build', 'system', 'prompt']
          __init__        → ['init']

        拆分规则：
          1. 先把下划线作为分隔符（react_scheduler → react scheduler）
          2. 再在大写字母边界插入空格（ReAct → Re Act）
             - 小写+大写：在中间切（aB → a B）
             - 多个大写+大写+小写：在最后两个之间切（HTTPResponse → HTTP Response）
          3. 拆出非空单词，全部小写后返回
        """
        import re
        # 第 1 步：下划线 → 空格
        s = ident.replace("_", " ")
        # 第 2 步：在小写后接大写处插入空格（reactScheduler → react Scheduler）
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
        # 第 3 步：在连续大写后接小写处插入空格（ReAct → Re Act）
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
        # 第 4 步：拆分并过滤
        return [w.lower() for w in s.split() if w]

    @staticmethod
    def _extract_key_terms(query: str) -> list[str]:
        """
        从用户查询中提取代码相关的关键术语。
        包括：dunder 方法名（__init__）、普通方法名、类名、函数名等。

        对每个英文标识符自动扩展命名变体：
          - 拆分出的单词（如 react_scheduler → react, scheduler）
          - 重新组合的驼峰形式（如 react_scheduler → ReactScheduler）
        这样查询 'react_scheduler' 时，也能命中源码中的 'ReActScheduler'。
        """
        import re
        terms = set()

        # 1. 提取 dunder 方法名（如 __init__, __str__）
        for m in re.findall(r"__[a-zA-Z_][a-zA-Z0-9_]*__", query):
            terms.add(m.lower())

        # 2. 提取看起来像 Python 标识符的词
        #    规则：以字母或下划线开头，包含字母/数字/下划线，长度 >= 3
        for m in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", query):
            lower = m.lower()
            # 过滤常见中英文停用词
            if lower in {
                # 英文
                "the", "and", "for", "what", "how", "does", "where", "when", "why",
                "who", "with", "from", "this", "that", "are", "is", "has", "have",
                "can", "could", "would", "should", "do", "did", "not", "but", "or",
                # 中文
                "哪些", "什么", "怎么", "如何", "方法", "逻辑", "作用", "哪里", "哪个",
            }:
                continue

            # 2a. 原始形式（保留小写下划线命名）
            terms.add(lower)

            # 2b. 命名变体：拆分出单词
            words = MultiRetriever._split_identifier_words(m)
            for w in words:
                if len(w) >= 2:
                    terms.add(w)

            # 2c. 重新组合的驼峰形式（用于命中原始驼峰命名的源码）
            if words:
                camel = "".join(w.capitalize() for w in words)
                terms.add(camel)
                terms.add(camel.lower())

        # 3. 提取连续的中文词段（2字以上）
        for m in re.findall(r"[\u4e00-\u9fa5]{2,}", query):
            terms.add(m)

        return list(terms)

    @staticmethod
    def _compute_keyword_boost(chunk_text: str, key_terms: list[str]) -> float:
        """
        计算关键词精确匹配加权分。
        如果分片文本包含查询中的关键术语，给它额外加分。
        分数 = 命中的关键术语数量 / 关键术语总数（上限 1.0）

        匹配策略分级：
          - 含下划线的完整标识符（如 react_scheduler）：用词边界匹配
            （避免 react_scheduler 命中 react_scheduler_xxx）
          - 不含下划线的英文单词/驼峰（如 react, scheduler, reactscheduler）：
            用子串匹配
            （这样 'react' 能命中 'reactscheduler'，解决命名风格不一致问题）
          - 纯中文：子串匹配（中文没有词边界概念）
        """
        if not key_terms:
            return 0.0

        lower_text = chunk_text.lower()
        import re
        hits = 0
        for term in key_terms:
            term_lower = term.lower()
            if not re.search(r"[a-zA-Z_]", term_lower):
                # 纯中文：子串匹配
                if term_lower in lower_text:
                    hits += 1
            elif "_" in term_lower:
                # 含下划线的完整标识符：用词边界断言避免误命中
                if re.search(rf"(?<!\w){re.escape(term_lower)}(?!\w)", lower_text):
                    hits += 1
            else:
                # 不含下划线的英文单词/驼峰：子串匹配
                if term_lower in lower_text:
                    hits += 1

        return min(1.0, hits / len(key_terms))

    @staticmethod
    def _compute_source_boost(file_path: str) -> float:
        """
        计算源文件加权分。
        非测试文件（源代码）给加分，测试文件不给。
        """
        if "__pycache__" in file_path:
            return 0.0
        # 按目录或文件名前缀/后缀精确判断，避免 "test" 子串误判（如 latest.py）
        parts = file_path.replace("\\", "/").lower().split("/")
        file_name = parts[-1]
        in_test_dir = any(p in ("test", "tests") for p in parts[:-1])
        is_test_file = file_name.startswith("test_") or file_name.endswith("_test.py")
        if in_test_dir or is_test_file:
            return 0.0
        return 1.0

    @staticmethod
    def _compute_business_route_boost(query: str, file_path: str) -> float:
        """
        业务关键词路由：根据用户查询中的业务关键词，给对应源文件加一个小 boost。

        原理：
          用户问"工具"时，registry.py 和 file_tools.py 应该比 main.py 更相关。
          但向量/BM25 检索可能让 main.py 的"工具"关键词分片得分更高（因为它是入口）。
          这里用关键词→文件映射做一个"轻推"：给命中业务关键词的源文件分片加 0.15 分。

          这不是"硬路由"——最终排序还是靠 Cross-Encoder 语义判断，
          这里只是确保目标文件的分片不会在召回阶段就被噪声淹没。

        参数:
            query:      用户原始查询
            file_path:  分片所属文件路径

        返回:
            float: 0.0（不命中）或 BUSINESS_ROUTE_BOOST（命中）
        """
        import os
        # 只给源文件加分，测试文件不加
        source_boost = MultiRetriever._compute_source_boost(file_path)
        if source_boost <= 0:
            return 0.0

        file_name = os.path.basename(file_path).lower()
        query_lower = query.lower()

        boost = 0.0
        for keyword, target_files in MultiRetriever.BUSINESS_KEYWORD_ROUTE.items():
            # 查询包含关键词（子串匹配，覆盖中英文混合表达）
            if keyword.lower() not in query_lower:
                continue
            # 当前文件在关键词路由的目标列表里
            if any(file_name == tf.lower() for tf in target_files):
                boost += MultiRetriever.BUSINESS_ROUTE_BOOST

        # 上限封顶：一个分片同时命中多个关键词时最多加 0.3
        return min(boost, 0.3)

    @staticmethod
    def _classify_chunk_type(chunk_text: str) -> str:
        """
        根据分片文本识别其代码类型，用于召回后过滤加权。

        为什么 multi_retriever 内部要重复一份识别逻辑？
          - code_splitter 写入的 chunk["metadata"]["chunk_type"] 在经过
            vector_store.search / bm25_retriever.search 时会被丢弃（这两层
            只返回 file_path / chunk_text / start_line / score 四个字段）。
          - 为保持"只修改 code_splitter 和 multi_retriever 两个文件"的约束、
            不破坏对外接口，且让旧数据（未带 chunk_type 元标签的 chunk）
            也能享受加权逻辑，这里基于 chunk_text 独立重识别。
          - 识别逻辑与 code_splitter._classify_chunk_type 完全一致。

        识别优先级（依次判断，命中即返回）：
          1. 文本中存在 class 定义行 → "class_def"
          2. 文本中存在 def 函数/方法定义行 → "function"
          3. 非 def/class 行中，import 语句占比 >= 70% → "import"
          4. 无赋值、无 return，且日志/print/纯调用行占比 >= 50% → "log_call"
          5. 否则 → "other"

        参数:
            chunk_text: 分片的完整文本

        返回:
            str: "class_def" | "function" | "import" | "log_call" | "other"
        """
        import re

        if not chunk_text or not chunk_text.strip():
            return "other"

        # 提取有效代码行（去空行、纯注释行）
        code_lines = []
        for line in chunk_text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            code_lines.append(s)

        if not code_lines:
            return "other"

        total = len(code_lines)

        # —— 1. 优先级最高：包含 class 定义 → class_def ——
        for line in code_lines:
            if re.match(r"^class\s+\w+", line):
                return "class_def"

        # —— 2. 包含 def 定义 → function ——
        for line in code_lines:
            if re.match(r"^\s*def\s+\w+", line):
                return "function"

        # —— 3. import 语句占比 >= 70% → import ——
        import_count = sum(
            1 for line in code_lines
            if re.match(r"^\s*(import\s+\S|from\s+\S+\s+import\b)", line)
        )
        if total > 0 and import_count / total >= 0.7:
            return "import"

        # —— 4. log_call：无赋值、无 return，且主要是日志/print/纯调用 ——
        def _is_assignment(line: str) -> bool:
            if "import" in line:
                return False
            for op in ("==", "!=", "<=", ">="):
                if op in line:
                    return False
            return "=" in line

        has_assignment = any(_is_assignment(line) for line in code_lines)
        has_return = any(line.startswith("return") for line in code_lines)

        if not has_assignment and not has_return:
            def _is_call_line(line: str) -> bool:
                if (line.startswith("print(") or line.startswith("logger.")
                        or line.startswith("logging.") or line.startswith("log.")):
                    return True
                if line.endswith(")") and "(" in line:
                    return True
                return False

            call_count = sum(1 for line in code_lines if _is_call_line(line))
            if total > 0 and call_count / total >= 0.5:
                return "log_call"

        # —— 5. 其他 → other ——
        return "other"

    # ============================================================
    #  对外 API：多路召回融合
    # ============================================================

    def search(self, query_vector, query_text: str,
               top_k: int = 5) -> list[dict]:
        """
        多路召回 + 融合排序 + 关键词加权 + 源文件加权 + chunk_type 加权。

        参数:
            query_vector: 查询向量，传给 InMemoryVectorStore.search
            query_text:  查询文本，传给 Bm25Retriever.search
            top_k:       最终返回的结果数量

        返回:
            list[dict]，按综合分数从高到低排列
        """
        # ---------- 参数校验 ----------
        if not isinstance(query_text, str):
            raise ValueError(f"query_text 必须是字符串，实际 {type(query_text)}")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k 必须是正整数，实际 {top_k}")
        if top_k > self.MAX_ALLOWED_TOP_K:
            print(f"[警告] top_k={top_k} 超过上限，已截断为 {self.MAX_ALLOWED_TOP_K}")
            top_k = self.MAX_ALLOWED_TOP_K

        # ---------- 第 0 步：从查询中提取关键术语 ----------
        key_terms = self._extract_key_terms(query_text)

        # ---------- 第 1 步：两路召回（每路都多拉一些候选给融合用） ----------
        candidate_k = max(top_k, self.VECTOR_CANDIDATE_K, self.BM25_CANDIDATE_K)

        vector_results = []
        bm25_results = []
        errors = []

        # —— 向量召回 ——
        try:
            vector_results = self.vector_store.search(
                query_vector, top_k=candidate_k
            )
        except Exception as e:
            errors.append(f"vector_search: {e}")
            print(f"[警告] 向量召回失败: {e}")

        # —— BM25 召回 ——
        try:
            bm25_results = self.bm25_retriever.search(
                query_text, top_k=candidate_k
            )
        except Exception as e:
            errors.append(f"bm25_search: {e}")
            print(f"[警告] BM25 召回失败: {e}")

        # 如果两路都失败，直接返回空
        if not vector_results and not bm25_results:
            print(f"[提示] 两路召回均为空，返回空列表。错误: {errors}")
            return []

        # ---------- 第 2 步：去重合并 ----------
        merged: dict[str, dict] = {}

        for r in vector_results:
            key = self._make_key(r["file_path"], r["start_line"])
            merged[key] = {
                "file_path": r["file_path"],
                "chunk_text": r["chunk_text"],
                "start_line": r["start_line"],
                "vector_score": float(r["score"]),
                "bm25_score": 0.0,
                "sources": ["vector"],
            }

        for r in bm25_results:
            key = self._make_key(r["file_path"], r["start_line"])
            if key in merged:
                merged[key]["bm25_score"] = float(r["score"])
                merged[key]["sources"].append("bm25")
            else:
                merged[key] = {
                    "file_path": r["file_path"],
                    "chunk_text": r["chunk_text"],
                    "start_line": r["start_line"],
                    "vector_score": 0.0,
                    "bm25_score": float(r["score"]),
                    "sources": ["bm25"],
                }

        # ---------- 第 3 步：分路归一化 ----------
        all_items = list(merged.values())

        vec_scores_present = [
            it["vector_score"] for it in all_items
            if "vector" in it["sources"]
        ]
        vec_norm_map = {}
        if vec_scores_present:
            vec_normed = self._min_max_normalize(vec_scores_present)
            present_items = [
                it for it in all_items if "vector" in it["sources"]
            ]
            for it, norm_val in zip(present_items, vec_normed):
                key = self._make_key(it["file_path"], it["start_line"])
                vec_norm_map[key] = norm_val

        bm25_scores_present = [
            it["bm25_score"] for it in all_items
            if "bm25" in it["sources"]
        ]
        bm25_norm_map = {}
        if bm25_scores_present:
            bm25_normed = self._min_max_normalize(bm25_scores_present)
            present_items = [
                it for it in all_items if "bm25" in it["sources"]
            ]
            for it, norm_val in zip(present_items, bm25_normed):
                key = self._make_key(it["file_path"], it["start_line"])
                bm25_norm_map[key] = norm_val

        # ---------- 第 4 步：计算综合分（加法基础分 × 乘法增强） ----------
        for it in all_items:
            key = self._make_key(it["file_path"], it["start_line"])
            vec_norm = vec_norm_map.get(key, 0.0)
            bm25_norm = bm25_norm_map.get(key, 0.0)

            # 4a: 关键词精确匹配加权（弥补短方法分片劣势）
            kw_boost = self._compute_keyword_boost(it["chunk_text"], key_terms)

            # 4b: 源文件加权（非测试文件加分）
            src_boost = self._compute_source_boost(it["file_path"])

            # 源文件业务实现分片保底（避免 run 方法中段因关键词稀疏被打入冷宫）
            if src_boost > 0 and kw_boost < 0.15:
                chunk_type = self._classify_chunk_type(it["chunk_text"])
                if chunk_type in ("class_def", "function"):
                    kw_boost = 0.15

            # 4c: 基础分 = 向量分 + BM25分 + 关键词基础分
            base_score = (
                self.VECTOR_WEIGHT * vec_norm +
                self.BM25_WEIGHT * bm25_norm +
                self.KEYWORD_BASE_WEIGHT * kw_boost
            )

            # 4d: 乘法增强 —— 关键词命中越多、源文件越优先，最终分越高
            kw_multiplier = 1.0 + self.KEYWORD_BOOST_MULTIPLIER * kw_boost
            # src_multiplier：源文件加分，测试文件降权
            if src_boost > 0:
                src_multiplier = 1.0 + self.SOURCE_BOOST_MULTIPLIER * src_boost
            else:
                src_multiplier = 1.0 - self.TEST_PENALTY_MULTIPLIER
            final_score = base_score * kw_multiplier * src_multiplier

            # 4e: 业务关键词路由加分（加法，轻推不硬路由）
            route_boost = self._compute_business_route_boost(
                query_text, it["file_path"]
            )
            final_score += route_boost

            it["vector_score_norm"] = vec_norm
            it["bm25_score_norm"] = bm25_norm
            it["keyword_boost"] = kw_boost
            it["source_boost"] = src_boost
            it["kw_multiplier"] = kw_multiplier
            it["src_multiplier"] = src_multiplier
            it["base_score"] = base_score
            it["route_boost"] = route_boost
            it["score"] = final_score

        # ---------- 第 4.5 步：chunk_type 元标签后处理加权 ----------
        # 对 import / log_call 噪声分片做乘法降分（系数 ∈ (0, 1)，不归零）
        # 对 function / class_def 业务实现分片做乘法加分
        # 旧数据（无 chunk_type 元标签）也能用，因为这里基于 chunk_text 独立重新识别
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

    # ============================================================
    #  rerank 前后处理：同文件邻域扩充
    # ============================================================

    def expand_with_neighbors(self, candidates: list[dict],
                               window: int = 2) -> list[dict]:
        """
        同文件邻域扩充：对每个候选分片，把同文件里前后 window 个分片
        追加进候选池，交给 reranker 统一打分排序。

        设计目的：
          - 切片时长方法可能被切成多段，单路召回往往只命中其中一段，
            邻域扩充把相邻分片一起送进 reranker，让重排模型看到完整上下文。
          - 不直接拼接 prompt，只扩充候选池，由 reranker 决定哪些分片有用。

        参数:
            candidates: 多路召回返回的候选分片列表
            window:     往前/往后各取几个邻居分片（默认 2）

        返回:
            扩充后的候选列表（原候选 + 新增邻居分片），不改变原顺序
        """
        # 如果没有构建文件映射（初始化时没传 all_chunks），直接返回不扩充
        if not self._file_chunks_map or not candidates:
            return candidates

        # 用 (file_path, start_line) 做去重 key，避免同一个分片被重复添加
        existing_keys = set()
        for c in candidates:
            existing_keys.add((c["file_path"], c["start_line"]))

        new_items = []  # 存放新增的邻居分片

        # 遍历每个候选分片，找它在同文件分片列表里的位置，取前后邻居
        for c in candidates:
            file_path = c["file_path"]
            # 取出该文件的全部分片（已按 start_line 排序）
            file_chunks = self._file_chunks_map.get(file_path, [])

            # 在 file_chunks 里找到当前候选分片的位置（用 start_line 匹配）
            idx = -1
            for i, fc in enumerate(file_chunks):
                if fc["start_line"] == c["start_line"]:
                    idx = i
                    break
            if idx < 0:
                # 没找到（可能是测试文件等没进索引的分片），跳过
                continue

            # 取 [idx-window, idx+window] 范围的分片作为邻居
            start_idx = max(0, idx - window)
            end_idx = min(len(file_chunks), idx + window + 1)  # +1 因为 range 不含末尾

            for i in range(start_idx, end_idx):
                if i == idx:
                    continue  # 跳过自己（已经在候选列表里了）
                fc = file_chunks[i]
                key = (fc["file_path"], fc["start_line"])
                if key in existing_keys:
                    continue  # 已经在候选列表或已添加过，跳过
                # 标记为新增邻居，记录去重 key
                existing_keys.add(key)
                new_item = dict(fc)  # 复制一份，避免污染原始数据
                new_item["score"] = 0.0             # 邻居分片没有召回分数
                new_item["expanded_by_neighbor"] = True  # 标记来源
                new_item["neighbor_of"] = (
                    c["file_path"], c["start_line"]
                )  # 记录是哪个候选的邻居，方便调试
                new_items.append(new_item)

        # 打印扩充日志，方便调试观察效果
        if new_items:
            # 统计每个文件新增了多少条
            file_counts = {}
            for ni in new_items:
                fp = ni["file_path"].replace("\\", "/").split("/")[-1]
                file_counts[fp] = file_counts.get(fp, 0) + 1
            print(f"       🔄 邻域扩充：新增 {len(new_items)} 条分片")
            for fp, cnt in file_counts.items():
                print(f"          - {fp}: +{cnt} 条")

        # 返回：原候选列表 + 新增邻居分片（顺序在后面，reranker 会重新排序）
        return candidates + new_items

    # ============================================================
    #  rerank 后处理：同文件加权偏好（可选，main.py 未启用）
    # ============================================================

    def apply_same_file_bonus(self, scored_candidates: list[dict],
                               source_files: set) -> list[dict]:
        """
        同文件加权偏好：对和"源分片"同文件的候选，rerank_score 加固定增量。

        设计目的：
          - 邻域扩充会引入邻居分片，其中和源分片同文件的应该有轻微偏好
          - 不强行置顶，只在 rerank 原始分数上 +SAME_FILE_BONUS 微调
          - 让 reranker 的语义判断仍然主导排序，同文件只是轻微加分

        参数:
            scored_candidates: rerank 打分后的候选列表（每条有 rerank_score）
            source_files:      源分片（多路召回直接命中的）的 file_path 集合

        返回:
            加权后按 rerank_score 重新排序的候选列表
        """
        # 空列表或空集合直接返回，不处理
        if not scored_candidates or not source_files:
            return scored_candidates

        bonus_count = 0       # 命中加权的计数器
        file_counts = {}      # 统计每个文件命中多少条 {文件名: 数量}

        # 遍历每条候选，检查它的 file_path 是否在源分片文件集合里
        for item in scored_candidates:
            item["same_file_bonus"] = 0.0
            if item["file_path"] in source_files:
                # 记录原始分数（微调前）
                old_score = item["rerank_score"]
                # 加固定增量（只加一点点，不破坏 rerank 的语义排序）
                item["rerank_score"] = old_score + self.SAME_FILE_BONUS
                item["same_file_bonus"] = self.SAME_FILE_BONUS
                bonus_count += 1
                fp = item["file_path"].replace("\\", "/").split("/")[-1]
                file_counts[fp] = file_counts.get(fp, 0) + 1
                print(f"          - {fp}:{item['start_line']}  "
                      f"原始={old_score:+.4f} → 加权后={item['rerank_score']:+.4f}")

        if bonus_count > 0:
            print(f"       🎯 同文件加权：{bonus_count} 条命中，"
                  f"增量 +{self.SAME_FILE_BONUS}")
            for fp, cnt in file_counts.items():
                print(f"          - {fp}: +{cnt} 条加权")

        # 按 rerank_score 降序重新排序（加权后分数变了，顺序可能变化）
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates

    # ============================================================
    #  rerank 后处理：保证源文件进入最终 Top N
    # ============================================================

    def ensure_source_file_in_top_n(self, candidates: list[dict],
                                    top_n: list[dict],
                                    query: str,
                                    quota: int = 2) -> list[dict]:
        """
        后处理：保证 rerank Top N 中至少包含 quota 个"含关键术语的源文件分片"。

        为什么需要这一步？
          rerank 模型（cross-encoder/ms-marco-MiniLM-L-6-v2）是 6 层小模型，
          对"中文自然语言问题 vs Python 代码"的语义匹配能力有限，常常把
          测试文件的 docstring 误判为比源码本身更相关。本方法在 rerank 完成
          后做一次"兜底提升"，保证至少有 quota 个"含查询关键术语的源文件分片"
          进入最终给 LLM 的 Top N。

        判断标准（同时满足才算"相关的源文件分片"）：
          - 是源文件（非测试文件，src_boost > 0）
          - 分片文本包含查询关键术语（kw_boost > 0）
          - 是业务实现分片（class_def 或 function），排除仅含 import 的 other 分片

        策略：
          1. 文件名匹配优先：若用户问题里的英文词出现在某源文件名里，
             且 Top N 里没有该文件，直接拎 quota 个该文件分片替换最后 quota 名
             （不看 chunk_type，因为注册机制核心可能写在模块级变量里）
          2. 业务实现兜底：若 Top N 里"含关键术语的源文件分片"数量不足 quota，
             从 candidates 里按分降序找对应的源文件分片补齐
          3. 如果已有 quota 个以上，不干预 rerank 判断

        参数:
            candidates: multi.search 返回的完整候选列表（按 score 降序）
            top_n:      reranker.rerank 返回的精排结果（按 rerank_score 降序）
            query:      用户原始查询文本（用于提取关键术语做精准判断）
            quota:      至少要保证的源文件分片数量（默认 2）

        返回:
            list[dict]: 可能被替换了最多 quota 个元素的新 top_n 列表
        """
        if not top_n or quota <= 0:
            return top_n[:]

        # quota 不能超过 top_n 长度
        quota = min(quota, len(top_n))

        # 提前提取关键术语，避免在循环里重复提取
        key_terms = self._extract_key_terms(query)

        def _is_relevant_source(item: dict) -> bool:
            """判断是否为"含关键术语的源文件业务实现分片" """
            if self._compute_source_boost(item["file_path"]) <= 0:
                return False
            if self._compute_keyword_boost(item["chunk_text"], key_terms) <= 0:
                return False
            chunk_type = self._classify_chunk_type(item["chunk_text"])
            return chunk_type in ("class_def", "function")

        # ============ 第 0 步：文件名匹配优先提升 ============
        file_name_terms = [
            t for t in key_terms
            if t.isascii() and len(t) >= 3 and "_" not in t
        ]
        if file_name_terms:
            def _file_name_matches(item: dict) -> bool:
                if self._compute_source_boost(item["file_path"]) <= 0:
                    return False
                file_name = item["file_path"].replace("\\", "/").split("/")[-1].lower()
                return any(term in file_name for term in file_name_terms)

            # top_n 里已有多少个匹配文件名的分片
            existing_name_matches = sum(
                1 for r in top_n if _file_name_matches(r)
            )
            if existing_name_matches < quota:
                need = quota - existing_name_matches
                new_items = []
                for c in candidates:
                    if not _file_name_matches(c):
                        continue
                    already = any(
                        r["file_path"] == c["file_path"]
                        and r["start_line"] == c["start_line"]
                        for r in top_n + new_items
                    )
                    if already:
                        continue
                    new_item = dict(c)
                    new_item["rerank_score"] = 0.0
                    new_item["boosted_by_source_filter"] = True
                    new_item["boost_reason"] = "file_name_match"
                    new_items.append(new_item)
                    if len(new_items) >= need:
                        break
                if new_items:
                    return top_n[:-len(new_items)] + new_items

        # ============ 第 1 步：业务实现兜底 ============
        existing_relevant = [r for r in top_n if _is_relevant_source(r)]
        if len(existing_relevant) >= quota:
            return top_n

        need = quota - len(existing_relevant)
        new_items = []
        seen_keys = {
            (r["file_path"], r["start_line"]) for r in top_n
        }
        for c in candidates:
            if not _is_relevant_source(c):
                continue
            key = (c["file_path"], c["start_line"])
            if key in seen_keys:
                continue
            # 文件名匹配优先：如果文件名匹配到了，同一个文件只取一个
            new_item = dict(c)
            new_item["rerank_score"] = 0.0
            new_item["boosted_by_source_filter"] = True
            new_item["boost_reason"] = "business_fallback"
            new_items.append(new_item)
            seen_keys.add(key)
            if len(new_items) >= need:
                break

        if new_items:
            return top_n[:-len(new_items)] + new_items

        return top_n


# ==================== 自测入口 ====================
if __name__ == "__main__":
    import numpy as np

    # —— 造一份相同的 chunks 数据，让两路都建在同一批分片上 ——
    chunks = [
        {
            "file_path": "file_loader.py",
            "chunk_text": "def load_source_files(source_dir):\n    # 加载所有 Python 源码文件\n    for root, dirs, files in os.walk(source_dir):\n        pass\n    return documents",
            "start_line": 1,
        },
        {
            "file_path": "code_splitter.py",
            "chunk_text": "def split_python_file(doc):\n    # 优先在 def 和 class 边界切分\n    blocks = _parse_line_blocks(doc['content'])\n    return chunks",
            "start_line": 152,
        },
        {
            "file_path": "vector_store.py",
            "chunk_text": "class InMemoryVectorStore:\n    def add(self, vectors, metadatas):\n        # 使用余弦相似度向量检索\n        self.vectors = np.vstack([self.vectors, vec_arr])",
            "start_line": 15,
        },
        {
            "file_path": "bm25_retriever.py",
            "chunk_text": "class Bm25Retriever:\n    def build_index(self, chunks):\n        # 使用 rank_bm25 关键词检索\n        self.bm25 = BM25Okapi(self.tokenized_corpus)",
            "start_line": 15,
        },
    ]

    # —— 1. 建向量库 ——
    vs = InMemoryVectorStore(embedding_dim=4)
    fake_vectors = [
        [0.9, 0.1, 0.0, 0.0],   # file_loader
        [0.1, 0.9, 0.1, 0.0],   # code_splitter
        [0.0, 0.1, 0.9, 0.1],   # vector_store
        [0.0, 0.0, 0.1, 0.9],   # bm25_retriever
    ]
    vs.add(fake_vectors, chunks)

    # —— 2. 建 BM25 索引 ——
    bm = Bm25Retriever()
    bm.build_index(chunks)

    # —— 3. 跑融合检索 ——
    multi = MultiRetriever(vs, bm)

    print("===== 融合检索：query_vector=[0.95,0,0,0], query_text='加载源码文件' =====")
    results = multi.search(
        query_vector=[0.95, 0.0, 0.0, 0.0],
        query_text="加载源码文件",
        top_k=3,
    )
    for r in results:
        print(f"  score={r['score']:.4f}  "
              f"vec={r['vector_score']:.3f}(norm={r['vector_score_norm']:.3f})  "
              f"bm25={r['bm25_score']:.3f}(norm={r['bm25_score_norm']:.3f})  "
              f"sources={r['sources']}  "
              f"file={r['file_path']}:{r['start_line']}")

    print("\n===== 融合检索：query_vector=[0,0,0.95,0], query_text='余弦相似度向量检索' =====")
    results = multi.search(
        query_vector=[0.0, 0.0, 0.95, 0.0],
        query_text="余弦相似度向量检索",
        top_k=3,
    )
    for r in results:
        print(f"  score={r['score']:.4f}  "
              f"vec={r['vector_score']:.3f}(norm={r['vector_score_norm']:.3f})  "
              f"bm25={r['bm25_score']:.3f}(norm={r['bm25_score_norm']:.3f})  "
              f"sources={r['sources']}  "
              f"file={r['file_path']}:{r['start_line']}")
