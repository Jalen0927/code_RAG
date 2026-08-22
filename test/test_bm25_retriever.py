"""
bm25_retriever.py 单元测试。

验证 BM25 关键词检索的核心逻辑：
    - 分词器（_tokenize）：Python 标识符、中文 n-gram、双下划线方法名
    - 中文 n-gram 滑窗（_chinese_ngram）
    - 建索引 + 搜索 + 排序
    - 参数校验（空查询、缺字段）

运行方式：
    python -m pytest test/test_bm25_retriever.py -v
"""

import pytest

from retriever.bm25_retriever import Bm25Retriever


# ============================================================
# 一、_chinese_ngram：中文 n-gram 滑窗
# ============================================================

class TestChineseNgram:
    """测中文按 n 字滑窗切分。"""

    def test_normal_bigram(self):
        """n=2：余弦相似度 → [余弦, 弦相, 相似, 似度]。"""
        result = Bm25Retriever._chinese_ngram("余弦相似度", 2)
        assert result == ["余弦", "弦相", "相似", "似度"]

    def test_short_string(self):
        """字符串比窗口短，原样返回。"""
        assert Bm25Retriever._chinese_ngram("余", 2) == ["余"]

    def test_single_char_n1(self):
        """n=1：每个字单独成 token。"""
        assert Bm25Retriever._chinese_ngram("余弦", 1) == ["余", "弦"]


# ============================================================
# 二、_tokenize：混合分词
# ============================================================

class TestTokenize:
    """测混合分词策略。"""

    def test_python_identifier(self):
        """Python 标识符整体作为一个 token。"""
        tokens = Bm25Retriever._tokenize("def load_source_files():")
        assert "load_source_files" in tokens

    def test_dunder_method(self):
        """双下划线方法名 __init__ 应作为整体 token，不被拆散。"""
        tokens = Bm25Retriever._tokenize("def __init__(self):")
        assert "__init__" in tokens

    def test_chinese_ngram(self):
        """中文应做 n-gram 切分。"""
        tokens = Bm25Retriever._tokenize("加载文件")
        assert "加载" in tokens or "文件" in tokens

    def test_empty_text(self):
        """空文本返回空列表。"""
        assert Bm25Retriever._tokenize("") == []

    def test_lowercase(self):
        """所有英文 token 应转小写。"""
        tokens = Bm25Retriever._tokenize("Def FOO Bar")
        for t in tokens:
            if t.isascii() and t.isalpha():
                assert t == t.lower()


# ============================================================
# 三、build_index + search：建索引和搜索
# ============================================================

def _make_fake_chunks():
    """造几个假分片用于测试。"""
    return [
        {
            "file_path": "file_loader.py",
            "chunk_text": "def load_source_files(source_dir):\n    加载所有 Python 源码文件\n    return documents",
            "start_line": 1,
        },
        {
            "file_path": "vector_store.py",
            "chunk_text": "class InMemoryVectorStore:\n    使用余弦相似度向量检索\n    def search(self, query_vector, top_k=5):",
            "start_line": 15,
        },
        {
            "file_path": "main.py",
            "chunk_text": "# 程序主入口\nif __name__ == '__main__':\n    user_query = input('请输入您的问题')",
            "start_line": 1,
        },
    ]


class TestBuildIndexAndSearch:
    """测建索引 + 搜索的完整流程。"""

    def test_build_index_returns_count(self):
        """build_index 返回实际加入的分片数量。"""
        bm25 = Bm25Retriever()
        n = bm25.build_index(_make_fake_chunks())
        assert n == 3
        assert len(bm25) == 3
        assert not bm25.is_empty()

    def test_search_keyword_match(self):
        """搜"加载"应命中 file_loader.py（含"加载"关键词）。"""
        bm25 = Bm25Retriever()
        bm25.build_index(_make_fake_chunks())
        results = bm25.search("加载 Python 文件", top_k=2)
        assert len(results) >= 1
        # file_loader.py 应排第一（含"加载"和"文件"）
        assert results[0]["file_path"] == "file_loader.py"
        assert results[0]["score"] > 0

    def test_search_sorted_by_score_desc(self):
        """结果应按 BM25 分数从高到低排列。"""
        bm25 = Bm25Retriever()
        bm25.build_index(_make_fake_chunks())
        results = bm25.search("余弦相似度", top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_query_returns_empty(self):
        """空查询（纯空格）返回空列表。"""
        bm25 = Bm25Retriever()
        bm25.build_index(_make_fake_chunks())
        assert bm25.search("   ", top_k=3) == []

    def test_search_before_build_returns_empty(self):
        """未建索引时搜索返回空列表。"""
        bm25 = Bm25Retriever()
        assert bm25.search("anything", top_k=3) == []

    def test_build_index_missing_field_raises(self):
        """分片缺字段应抛 ValueError。"""
        bm25 = Bm25Retriever()
        bad_chunks = [{"file_path": "a.py", "chunk_text": "x = 1"}]  # 缺 start_line
        with pytest.raises(ValueError):
            bm25.build_index(bad_chunks)

    def test_build_index_empty_list(self):
        """空列表建索引返回 0。"""
        bm25 = Bm25Retriever()
        assert bm25.build_index([]) == 0
        assert bm25.is_empty()

    def test_clear(self):
        """clear 后库为空。"""
        bm25 = Bm25Retriever()
        bm25.build_index(_make_fake_chunks())
        bm25.clear()
        assert bm25.is_empty()
        assert len(bm25) == 0
