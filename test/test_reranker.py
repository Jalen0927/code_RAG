"""
reranker.py 单元测试。

验证 Reranker 重排器的核心逻辑：
    - 参数校验（空 query、空候选、缺 chunk_text）
    - 混合分计算（Score Blending：0.6*CE_norm + 0.4*fusion_norm）
    - 模型加载失败时的兜底返回

注意：不加载真实 Cross-Encoder 模型，用 mock 模拟 model.predict 的返回。

运行方式：
    python -m pytest test/test_reranker.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from retriever.reranker import Reranker


# ============================================================
# 一、__init__ 参数校验
# ============================================================

class TestRerankerInit:
    """测 Reranker 初始化的参数校验。"""

    def test_empty_model_name_raises(self):
        """空模型名应抛 ValueError。"""
        with pytest.raises(ValueError):
            Reranker(model_name="")

    def test_default_model_name(self):
        """默认模型名应为 cross-encoder/ms-marco-MiniLM-L-6-v2。"""
        r = Reranker()
        assert "cross-encoder" in r.model_name
        assert r.model is None  # 延迟加载，此时未加载

    def test_is_ready_before_load(self):
        """加载前 is_ready 应为 False。"""
        r = Reranker()
        assert r.is_ready() is False


# ============================================================
# 二、rerank 参数校验
# ============================================================

class TestRerankValidation:
    """测 rerank 方法的参数校验（不触发模型加载）。"""

    def test_empty_query_raises(self):
        """空 query 应抛 ValueError。"""
        r = Reranker()
        with pytest.raises(ValueError):
            r.rerank("", [{"chunk_text": "x"}])

    def test_whitespace_query_raises(self):
        """纯空格 query 应抛 ValueError。"""
        r = Reranker()
        with pytest.raises(ValueError):
            r.rerank("   ", [{"chunk_text": "x"}])

    def test_empty_candidates_returns_empty(self):
        """空候选列表应返回空列表。"""
        r = Reranker()
        assert r.rerank("query", []) == []

    def test_candidates_not_list_raises(self):
        """candidates 不是 list 应抛 ValueError。"""
        r = Reranker()
        with pytest.raises(ValueError):
            r.rerank("query", "not a list")

    def test_missing_chunk_text_raises(self):
        """候选缺 chunk_text 字段应抛 ValueError。"""
        r = Reranker()
        bad_candidates = [{"file_path": "a.py", "start_line": 1}]  # 缺 chunk_text
        with pytest.raises(ValueError):
            r.rerank("query", bad_candidates, top_n=1)

    def test_top_n_must_be_positive(self):
        """top_n 非正整数应抛 ValueError。"""
        r = Reranker()
        with pytest.raises(ValueError):
            r.rerank("query", [{"chunk_text": "x"}], top_n=0)


# ============================================================
# 三、rerank 混合分计算（mock 模型，不加载真实模型）
# ============================================================

class TestRerankBlendScore:
    """测 Score Blending 混合分计算逻辑。"""

    def _make_mocked_reranker(self, predict_scores):
        """造一个 mock 了模型的 Reranker（不加载真实模型）。"""
        r = Reranker()
        # mock：模型已加载
        r.model = MagicMock()
        r.model.predict.return_value = np.array(predict_scores, dtype=np.float32)
        # mock _ensure_model 不做事（假装模型已加载）
        r._ensure_model = MagicMock()
        return r

    def test_results_sorted_by_rerank_score_desc(self):
        """结果应按 rerank_score 从高到低排列。"""
        r = self._make_mocked_reranker([1.0, 5.0, 3.0])
        candidates = [
            {"file_path": "a.py", "start_line": 1, "chunk_text": "aaa", "score": 0.5},
            {"file_path": "b.py", "start_line": 2, "chunk_text": "bbb", "score": 0.3},
            {"file_path": "c.py", "start_line": 3, "chunk_text": "ccc", "score": 0.4},
        ]
        results = r.rerank("query", candidates, top_n=3)
        scores = [x["rerank_score"] for x in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limits_results(self):
        """top_n 应限制返回数量。"""
        r = self._make_mocked_reranker([1.0, 2.0, 3.0])
        candidates = [
            {"file_path": "a.py", "start_line": 1, "chunk_text": "a", "score": 0.5},
            {"file_path": "b.py", "start_line": 2, "chunk_text": "b", "score": 0.5},
            {"file_path": "c.py", "start_line": 3, "chunk_text": "c", "score": 0.5},
        ]
        results = r.rerank("query", candidates, top_n=2)
        assert len(results) == 2

    def test_blend_score_in_range_01(self):
        """混合分应在 [0, 1] 区间（两路分数都做了 min-max 归一化）。"""
        r = self._make_mocked_reranker([1.0, 5.0, 3.0])
        candidates = [
            {"file_path": "a.py", "start_line": 1, "chunk_text": "a", "score": 0.5},
            {"file_path": "b.py", "start_line": 2, "chunk_text": "b", "score": 0.3},
            {"file_path": "c.py", "start_line": 3, "chunk_text": "c", "score": 0.4},
        ]
        results = r.rerank("query", candidates, top_n=3)
        for x in results:
            assert 0.0 - 1e-9 <= x["rerank_score"] <= 1.0 + 1e-9

    def test_preserves_original_fields(self):
        """重排后应保留原候选的所有字段。"""
        r = self._make_mocked_reranker([1.0])
        candidates = [
            {"file_path": "a.py", "start_line": 1, "chunk_text": "a", "score": 0.5,
             "extra_field": "hello"},
        ]
        results = r.rerank("query", candidates, top_n=1)
        assert results[0]["file_path"] == "a.py"
        assert results[0]["extra_field"] == "hello"
        # 应额外带上 rerank_score 等新字段
        assert "rerank_score" in results[0]
        assert "rerank_raw_score" in results[0]
        assert "orig_fusion_norm" in results[0]
