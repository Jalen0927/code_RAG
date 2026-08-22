"""
vector_store.py 单元测试。

验证内存向量库的核心逻辑：
    - 余弦相似度计算（_cosine_similarity）
    - 添加向量 + 搜索排序
    - 参数校验（维度不匹配、缺字段、空库）

运行方式：
    python -m pytest test/test_vector_store.py -v
"""

import numpy as np
import pytest

from retriever.vector_store import InMemoryVectorStore


# ============================================================
# 一、_cosine_similarity：余弦相似度计算
# ============================================================

class TestCosineSimilarity:
    """测手写余弦相似度的正确性。"""

    def test_identical_vectors(self):
        """完全相同的向量相似度应为 1.0。"""
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        sim = InMemoryVectorStore._cosine_similarity(q, store)
        assert abs(float(sim[0]) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """正交向量相似度应为 0。"""
        q = np.array([1.0, 0.0], dtype=np.float32)
        store = np.array([[0.0, 1.0]], dtype=np.float32)
        sim = InMemoryVectorStore._cosine_similarity(q, store)
        assert abs(float(sim[0])) < 1e-6

    def test_opposite_vectors(self):
        """方向相反的向量相似度应为 -1。"""
        q = np.array([1.0, 0.0], dtype=np.float32)
        store = np.array([[-1.0, 0.0]], dtype=np.float32)
        sim = InMemoryVectorStore._cosine_similarity(q, store)
        assert abs(float(sim[0]) - (-1.0)) < 1e-6

    def test_zero_vector_no_nan(self):
        """全 0 向量不应产生 nan（有兜底处理）。"""
        q = np.array([0.0, 0.0], dtype=np.float32)
        store = np.array([[1.0, 0.0]], dtype=np.float32)
        sim = InMemoryVectorStore._cosine_similarity(q, store)
        assert not np.isnan(sim).any()


# ============================================================
# 二、add + search：增删查
# ============================================================

def _make_fake_data():
    """造 4 个假向量和元数据。"""
    vecs = [
        [1.0, 0.0, 0.0, 0.0],   # X 方向
        [0.0, 1.0, 0.0, 0.0],   # Y 方向
        [1.0, 1.0, 0.0, 0.0],   # XY 角平分线
        [0.0, 0.0, 1.0, 0.0],   # Z 方向
    ]
    metas = [
        {"file_path": "a.py", "chunk_text": "def foo(): pass", "start_line": 1},
        {"file_path": "b.py", "chunk_text": "x = 1", "start_line": 10},
        {"file_path": "c.py", "chunk_text": "class Bar: pass", "start_line": 20},
        {"file_path": "d.py", "chunk_text": "import os", "start_line": 30},
    ]
    return vecs, metas


class TestAddAndSearch:
    """测添加 + 搜索。"""

    def test_add_returns_count(self):
        """add 返回实际新增条数。"""
        store = InMemoryVectorStore()
        vecs, metas = _make_fake_data()
        n = store.add(vecs, metas)
        assert n == 4
        assert len(store) == 4

    def test_search_most_similar_first(self):
        """用 [1,0,0,0] 查，最相似应是 a.py（完全相同的向量）。"""
        store = InMemoryVectorStore()
        vecs, metas = _make_fake_data()
        store.add(vecs, metas)
        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) >= 2
        assert results[0]["file_path"] == "a.py"
        assert abs(results[0]["score"] - 1.0) < 1e-5

    def test_search_sorted_desc(self):
        """结果应按相似度从高到低排列。"""
        store = InMemoryVectorStore()
        vecs, metas = _make_fake_data()
        store.add(vecs, metas)
        results = store.search([1.0, 1.0, 0.0, 0.0], top_k=4)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_store_returns_empty(self):
        """空库搜索返回空列表。"""
        store = InMemoryVectorStore()
        assert store.search([1.0, 0.0, 0.0, 0.0]) == []
        assert store.is_empty()

    def test_add_dimension_mismatch_raises(self):
        """维度不匹配应抛 ValueError。"""
        store = InMemoryVectorStore(embedding_dim=4)
        with pytest.raises(ValueError):
            store.add([[1.0, 0.0, 0.0]], [{"file_path": "x", "chunk_text": "x", "start_line": 1}])

    def test_add_missing_metadata_field_raises(self):
        """元数据缺字段应抛 ValueError。"""
        store = InMemoryVectorStore()
        vecs, _ = _make_fake_data()
        bad_metas = [{"file_path": "a.py", "chunk_text": "x"}]  # 缺 start_line
        with pytest.raises(ValueError):
            store.add([vecs[0]], bad_metas)

    def test_add_count_mismatch_raises(self):
        """向量和元数据数量不匹配应抛 ValueError。"""
        store = InMemoryVectorStore()
        vecs, _ = _make_fake_data()
        with pytest.raises(ValueError):
            store.add(vecs, [{"file_path": "a.py", "chunk_text": "x", "start_line": 1}])

    def test_clear(self):
        """clear 后库为空。"""
        store = InMemoryVectorStore()
        vecs, metas = _make_fake_data()
        store.add(vecs, metas)
        store.clear()
        assert store.is_empty()
        assert len(store) == 0
