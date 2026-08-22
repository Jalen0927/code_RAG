"""
llm_client.py + embedding_client.py 单元测试。

重点验证 Bug 11 修复：MiMoClient.__init__ 校验 API Key（fail-fast）。
还验证 EmbeddingClient 的输入校验（空文本、空列表）。

这些测试都用 monkeypatch 设置假值，不依赖真实 .env 配置，不调真实 API。

运行方式：
    python -m pytest test/test_clients.py -v
"""

import pytest

import llm_client
import embedding_client
from llm_client import MiMoClient, LLMRequestError
from embedding_client import EmbeddingClient, EmbeddingError


# ============================================================
# 一、MiMoClient 校验（Bug 11 修复验证）
# ============================================================

class TestMiMoClientValidation:
    """Bug 11：__init__ 必须校验 API Key / Base URL / Model 三项配置。"""

    def test_missing_api_key_raises(self, monkeypatch):
        """缺 API Key 应在 __init__ 就抛 LLMRequestError，不假就绪。"""
        monkeypatch.setattr(llm_client, "LLM_API_KEY", "")
        with pytest.raises(LLMRequestError):
            MiMoClient()

    def test_missing_base_url_raises(self, monkeypatch):
        """缺 Base URL 应抛 LLMRequestError。"""
        monkeypatch.setattr(llm_client, "LLM_API_KEY", "fake_key")
        monkeypatch.setattr(llm_client, "LLM_BASE_URL", "")
        with pytest.raises(LLMRequestError):
            MiMoClient()

    def test_missing_model_raises(self, monkeypatch):
        """缺 Model 应抛 LLMRequestError。"""
        monkeypatch.setattr(llm_client, "LLM_API_KEY", "fake_key")
        monkeypatch.setattr(llm_client, "LLM_BASE_URL", "https://fake.url")
        monkeypatch.setattr(llm_client, "LLM_MODEL", "")
        with pytest.raises(LLMRequestError):
            MiMoClient()

    def test_all_config_present_no_init_error(self, monkeypatch):
        """三项配置齐全时，__init__ 不应抛错（构造 headers 成功）。"""
        monkeypatch.setattr(llm_client, "LLM_API_KEY", "fake_key")
        monkeypatch.setattr(llm_client, "LLM_BASE_URL", "https://fake.url")
        monkeypatch.setattr(llm_client, "LLM_MODEL", "fake_model")
        client = MiMoClient()
        assert "Authorization" in client.headers
        assert "Bearer fake_key" in client.headers["Authorization"]


# ============================================================
# 二、EmbeddingClient 校验
# ============================================================

class TestEmbeddingClientValidation:
    """测 EmbeddingClient 的输入校验。"""

    def _make_client_with_fake_key(self, monkeypatch):
        """用假 key 实例化 EmbeddingClient（绕过真实 .env 依赖）。"""
        monkeypatch.setattr(embedding_client, "DASHSCOPE_API_KEY", "fake_key")
        monkeypatch.setattr(embedding_client, "EMBEDDING_MODEL", "fake_model")
        return EmbeddingClient()

    def test_missing_api_key_raises(self, monkeypatch):
        """缺 DASHSCOPE_API_KEY 应抛 EmbeddingError。"""
        monkeypatch.setattr(embedding_client, "DASHSCOPE_API_KEY", "")
        with pytest.raises(EmbeddingError):
            EmbeddingClient()

    def test_missing_model_raises(self, monkeypatch):
        """缺 EMBEDDING_MODEL 应抛 EmbeddingError。"""
        monkeypatch.setattr(embedding_client, "DASHSCOPE_API_KEY", "fake_key")
        monkeypatch.setattr(embedding_client, "EMBEDDING_MODEL", "")
        with pytest.raises(EmbeddingError):
            EmbeddingClient()

    def test_get_embedding_empty_text_raises(self, monkeypatch):
        """空文本应抛 EmbeddingError。"""
        client = self._make_client_with_fake_key(monkeypatch)
        with pytest.raises(EmbeddingError):
            client.get_embedding("")

    def test_get_embedding_whitespace_text_raises(self, monkeypatch):
        """纯空白文本应抛 EmbeddingError。"""
        client = self._make_client_with_fake_key(monkeypatch)
        with pytest.raises(EmbeddingError):
            client.get_embedding("   \n  \t ")

    def test_get_embedding_non_string_raises(self, monkeypatch):
        """非字符串（如 int）应抛 EmbeddingError。"""
        client = self._make_client_with_fake_key(monkeypatch)
        with pytest.raises(EmbeddingError):
            client.get_embedding(123)

    def test_get_batch_embedding_empty_list_raises(self, monkeypatch):
        """空列表应抛 EmbeddingError。"""
        client = self._make_client_with_fake_key(monkeypatch)
        with pytest.raises(EmbeddingError):
            client.get_batch_embedding([])

    def test_get_batch_embedding_non_list_raises(self, monkeypatch):
        """非 list 类型应抛 EmbeddingError。"""
        client = self._make_client_with_fake_key(monkeypatch)
        with pytest.raises(EmbeddingError):
            client.get_batch_embedding("not a list")

    def test_get_batch_embedding_empty_element_raises(self, monkeypatch):
        """列表里有空字符串元素应抛 EmbeddingError。"""
        client = self._make_client_with_fake_key(monkeypatch)
        with pytest.raises(EmbeddingError):
            client.get_batch_embedding(["valid text", ""])
