"""
retriever/bm25_retriever.py — BM25 关键词检索器（使用 rank_bm25 库）。

职责：
  - 对所有分片建 BM25 倒排索引
  - 查询时按关键词命中度打分，取 top_k 个最相关分片
  - 支持中英文混合分词（Python 标识符 + 中文 2-gram 滑窗）

BM25 原理一句话总结：
  查询里每个"词"在文档中出现得越多越相关，但出现太多次会边际效应递减；
  同时，太常见的词（比如"the"、"def"）权重低，稀有的词权重高。
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


class Bm25Retriever:
    """
    BM25 关键词检索。

    内部保存：
        self.chunks: list[dict]
            所有分片的元数据，格式 [{"file_path", "chunk_text", "start_line"}, ...]
        self.tokenized_corpus: list[list[str]]
            每个分片对应的分词列表（tokenized corpus），与 self.chunks 一一对应。
        self.bm25: BM25Okapi | None
            构建索引后生成的 BM25 对象。
    """

    # ———————— 可调常量 ————————
    # top_k 上限，防止用户传离谱的数字
    MAX_ALLOWED_TOP_K = 1000
    # BM25 分数下限，**严格大于**这个值才算命中（所以默认 0.0 = 过滤掉全没命中的、score=0 的文档）
    MIN_SCORE_THRESHOLD = 0.0
    # 中文 n-gram 滑窗大小：2 = 两字一组（bigram），推荐 2~3
    CHINESE_NGRAM = 2
    # ———————————————————————

    # 正则分三段：
    #  1) 双下划线包裹的特殊方法名（如 __init__, __str__）整体作为一个 token
    #  2) 普通 Python 标识符/数字（当成一个完整词）
    #  3) 连续中文字符
    _DUNDER_RE = re.compile(r"__[a-zA-Z_][a-zA-Z0-9_]*__")
    _PY_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
    _CHINESE_RE = re.compile(r"[\u4e00-\u9fa5]+")

    def __init__(self):
        self.chunks: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    # ============================================================
    #  工具方法：分词 + 校验
    # ============================================================

    @classmethod
    def _chinese_ngram(cls, chinese_str: str, n: int) -> list[str]:
        """对连续中文字符串做 n-gram 滑窗切分。

        例："余弦相似度" + n=2  →  ["余弦", "弦相", "相似", "似度"]
        这样不管用户搜"余弦"、"相似度"还是更长的短语，都能有共同的字组命中。
        """
        if len(chinese_str) < n:
            # 字符串比窗口还短，直接原样返回
            return [chinese_str]
        return [chinese_str[i:i + n] for i in range(len(chinese_str) - n + 1)]

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """
        把一段文本切成词（token）列表。

        混合策略：
          1) Python 标识符/数字：按完整单词切（load_source_files 就是一个 token）
          2) 中文字符：先按连续汉字分段，再对每段做 n-gram 滑窗切分
             （没有使用 jieba 等专业中文分词库，以减少依赖）
        """
        if not text:
            return []

        tokens: list[str] = []

        # —— 0. 优先提取双下划线包裹的特殊方法名（如 __init__）——
        # 必须在普通标识符之前处理，否则 __init__ 会被拆成 __, init, __
        dunders = set()
        for t in cls._DUNDER_RE.findall(text):
            dunders.add(t.lower())

        # —— 1. 把"所有的 Python 标识符/数字"切出来，统一小写 ——
        for t in cls._PY_TOKEN_RE.findall(text):
            low = t.lower()
            if low not in dunders:
                tokens.append(low)

        # 补充 dunder 词（确保 __init__ 等作为完整 token）
        tokens.extend(dunders)

        # —— 2. 把"所有的连续中文段"找出来，对每段做 n-gram ——
        n = max(1, int(cls.CHINESE_NGRAM))
        for chinese_seg in cls._CHINESE_RE.findall(text):
            tokens.extend(cls._chinese_ngram(chinese_seg, n))

        return tokens

    def _is_built(self) -> bool:
        """是否已经 build_index。"""
        return self.bm25 is not None and len(self.tokenized_corpus) > 0

    @staticmethod
    def _validate_chunk(chunk, idx: int):
        """检查单个分片是否包含要求的三个字段，不满足就抛 ValueError。"""
        if not isinstance(chunk, dict):
            raise ValueError(f"第 {idx} 条分片不是 dict 类型：{type(chunk)}")
        required = {"file_path", "chunk_text", "start_line"}
        missing = required - chunk.keys()
        if missing:
            raise ValueError(
                f"第 {idx} 条分片缺少字段: {sorted(missing)}，实际字段: {sorted(chunk.keys())}"
            )

    # ============================================================
    #  对外 API
    # ============================================================

    def build_index(self, chunks: list[dict]) -> int:
        """
        构建 BM25 索引。

        参数:
            chunks: list[dict]，每条格式
                    {"file_path": str, "chunk_text": str, "start_line": int}

        返回:
            int: 实际加入索引的分片数量。
        """
        # ---------- 参数校验 ----------
        if not isinstance(chunks, list):
            raise ValueError(
                f"chunks 必须是 list[dict]，实际类型: {type(chunks)}"
            )
        if len(chunks) == 0:
            print("[提示] build_index 收到空列表，未构建任何索引")
            self.chunks = []
            self.tokenized_corpus = []
            self.bm25 = None
            return 0

        for i, c in enumerate(chunks):
            self._validate_chunk(c, i)

        # ---------- 分词 + 建索引 ----------
        self.chunks = list(chunks)  # 存一份副本，外面修改不影响内部
        self.tokenized_corpus = [
            self._tokenize(c["chunk_text"]) for c in self.chunks
        ]

        # 如果分词之后全是空的（比如全部是空格标点），BM25Okapi 会报错，提前兜底
        if all(len(toks) == 0 for toks in self.tokenized_corpus):
            print("[警告] 所有 chunk 分词后都为空，未构建 BM25 索引")
            self.bm25 = None
            return 0

        # 真正构建 BM25 对象（BM25Okapi 是 rank_bm25 库里最经典的 BM25 变种）
        try:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        except Exception as e:
            print(f"[警告] 构建 BM25 索引失败: {e}")
            self.bm25 = None
            return 0

        return len(self.chunks)

    def search(self, query: str, top_k: int = 5,
               min_score: float | None = None) -> list[dict]:
        """
        用关键词检索最相关的 top_k 个分片。

        参数:
            query:     用户查询字符串（原样，内部会做相同规则的分词）
            top_k:     返回结果数量
            min_score: BM25 分数下限，默认使用类常量 MIN_SCORE_THRESHOLD

        返回:
            list[dict]，按 BM25 分数从高到低排列，每条：
            {
                "file_path": str,
                "chunk_text": str,
                "start_line": int,
                "score": float,     # BM25 打分（正数，越大越相关）
            }
        """
        # ---------- 参数校验 ----------
        if not self._is_built():
            print("[提示] BM25 索引未构建（请先调用 build_index），返回空列表")
            return []

        if not isinstance(query, str):
            raise ValueError(f"query 必须是字符串，实际 {type(query)}")
        if not query.strip():
            return []

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k 必须是正整数，实际 {top_k}")
        if top_k > self.MAX_ALLOWED_TOP_K:
            print(
                f"[警告] top_k={top_k} 超过上限 {self.MAX_ALLOWED_TOP_K}，"
                f"已自动截断为上限值"
            )
            top_k = self.MAX_ALLOWED_TOP_K

        threshold = (
            min_score if min_score is not None
            else self.MIN_SCORE_THRESHOLD
        )

        # ---------- 对查询用相同规则分词 ----------
        tokenized_query = self._tokenize(query)
        if len(tokenized_query) == 0:
            # 查询切不出任何 token（比如全是空格和标点），就返回空
            return []

        # ---------- 调 rank_bm25 拿所有 chunk 的分数 ----------
        # get_scores 返回一个 numpy 数组（或 list），长度 = 分片数
        scores = self.bm25.get_scores(tokenized_query)

        # 把 (分数, 下标) 成对凑起来，按分数从高到低排序
        scored = sorted(
            [(float(scores[i]), i) for i in range(len(self.chunks))],
            key=lambda x: x[0],
            reverse=True,
        )

        # ---------- 取 top_k + 过滤阈值 ----------
        # 注意：阈值比较用 "严格大于"，因为 BM25 对完全没命中任何查询词的文档
        # 打分恒为 0，等于阈值时应该视为"不相关"过滤掉
        results = []
        for score, idx in scored:
            if score <= threshold:
                break
            md = self.chunks[idx]
            results.append({
                "file_path": md["file_path"],
                "chunk_text": md["chunk_text"],
                "start_line": md["start_line"],
                "score": score,
            })
            if len(results) >= top_k:
                break

        return results

    def clear(self) -> None:
        """清空所有分片和索引。"""
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None

    # ============================================================
    #  辅助属性
    # ============================================================

    def __len__(self) -> int:
        return len(self.chunks)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0


# ==================== 自测入口 ====================
if __name__ == "__main__":
    retriever = Bm25Retriever()

    fake_chunks = [
        {
            "file_path": "file_loader.py",
            "chunk_text": (
                "def load_source_files(source_dir):\n"
                "    for root, dirs, files in os.walk(source_dir):\n"
                "        if file.endswith('.py'):\n"
                "            with open(full_path, 'r', encoding='utf-8') as f:\n"
                "                content = f.read()\n"
                "    return documents  # 加载所有 Python 源码文件\n"
            ),
            "start_line": 1,
        },
        {
            "file_path": "code_splitter.py",
            "chunk_text": (
                "def split_python_file(doc):\n"
                "    # 优先在 def 和 class 边界切分\n"
                "    blocks = _parse_line_blocks(doc['content'])\n"
                "    # 然后贪心拼接成 chunk\n"
                "    for blk in blocks:\n"
                "        current_blocks.append(blk)\n"
                "    return chunks  # 返回分片后的文本块列表\n"
            ),
            "start_line": 152,
        },
        {
            "file_path": "vector_store.py",
            "chunk_text": (
                "class InMemoryVectorStore:\n"
                "    def add(self, vectors, metadatas):\n"
                "        self.vectors = np.vstack([self.vectors, vec_arr])\n"
                "    def search(self, query_vector, top_k=5):\n"
                "        scores = self._cosine_similarity(query_vector, self.vectors)\n"
                "        return results  # 使用余弦相似度向量检索\n"
            ),
            "start_line": 15,
        },
        {
            "file_path": "main.py",
            "chunk_text": (
                "# 程序主入口\n"
                "if __name__ == '__main__':\n"
                "    user_query = input('请输入您的问题: ')\n"
                "    print('正在为您检索相关文档...')\n"
                "    # 后续集成检索与大模型\n"
            ),
            "start_line": 1,
        },
    ]

    n = retriever.build_index(fake_chunks)
    print(f"build_index 完成，索引 {n} 个分片")

    print("\n===== 搜索：'加载所有 Python 源码文件' top_k=3 =====")
    for r in retriever.search("加载所有 Python 源码文件", top_k=3):
        print(f"  score={r['score']:.3f}  file={r['file_path']}:{r['start_line']}"
              f"  首行='{r['chunk_text'].splitlines()[0]}'")

    print("\n===== 搜索：'余弦相似度 向量检索' top_k=2 =====")
    for r in retriever.search("余弦相似度 向量检索", top_k=2):
        print(f"  score={r['score']:.3f}  file={r['file_path']}:{r['start_line']}"
              f"  首行='{r['chunk_text'].splitlines()[0]}'")

    print("\n===== 搜索：'hello_xyz_不存在' top_k=5 =====")
    res = retriever.search("hello_xyz_不存在", top_k=5)
    print(f"  结果数 = {len(res)}，分数均为 0 或为空")

    retriever.clear()
    print(f"\nclear 后 is_empty = {retriever.is_empty()}")
    print(f"clear 后搜索返回 = {retriever.search('加载文件')}")
