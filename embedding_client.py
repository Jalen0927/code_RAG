"""
embedding_client.py — 阿里云 DashScope 向量模型客户端。

职责：
  - 封装 DashScope 向量模型 SDK 调用
  - 支持单条 / 批量文本向量化（批量接口自动分批，每批最多 20 条）
  - 初始化时校验关键配置（API Key、模型名）
"""

import dashscope
from dashscope import TextEmbedding

from config import (
    DASHSCOPE_API_KEY,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
)


class EmbeddingError(Exception):
    """向量模型 API 调用异常"""
    pass


class EmbeddingClient:
    """
    阿里云 DashScope（通义千问）向量模型客户端。

    使用 dashscope 官方 SDK 调用，不需要手动拼 URL / Header，
    SDK 内部会自动处理鉴权、重试、网络异常等细节。

    使用前提：
        - pip install dashscope
        - .env 中配置 DASHSCOPE_API_KEY 和 EMBEDDING_MODEL
    """

    # 不同模型的单批上限不同，__init__ 时根据 EMBEDDING_MODEL 动态决定
    #   qwen3.7-text-embedding: 20 条/批
    #   text-embedding-v4 / v3:  10 条/批
    #   默认兜底: 10 条
    _MODEL_BATCH_LIMITS = {
        "qwen3.7-text-embedding": 20,
        "text-embedding-v4": 10,
        "text-embedding-v3": 10,
    }
    _DEFAULT_BATCH_SIZE = 10

    def __init__(self):
        # 校验密钥与模型名
        if not DASHSCOPE_API_KEY:
            raise EmbeddingError(
                "环境变量缺失: DASHSCOPE_API_KEY，请在 .env 中配置"
            )
        if not EMBEDDING_MODEL:
            raise EmbeddingError(
                "环境变量缺失: EMBEDDING_MODEL，请在 .env 中配置"
            )

        # 把 api_key 设置到 dashscope 全局
        # SDK 后续所有调用都会自动用这个 key 鉴权
        dashscope.api_key = DASHSCOPE_API_KEY

        # 根据当前模型动态确定单批上限
        self.MAX_BATCH_SIZE = self._MODEL_BATCH_LIMITS.get(
            EMBEDDING_MODEL, self._DEFAULT_BATCH_SIZE
        )

    def get_embedding(self, text: str) -> list[float]:
        """
        输入单段文本，返回一维浮点向量。

        参数:
            text: 要向量化的文本（不能为空）

        返回:
            list[float]，例如 [0.012, -0.034, 0.078, ...]
        """
        # 空文本校验
        if not isinstance(text, str):
            raise EmbeddingError(f"text 必须是字符串，实际类型 {type(text).__name__}")
        if not text.strip():
            raise EmbeddingError("text 不能为空字符串或纯空白")

        # 复用批量方法，避免重复实现接口调用逻辑
        vectors = self.get_batch_embedding([text])
        return vectors[0]

    def get_batch_embedding(self, text_list: list[str]) -> list[list[float]]:
        """
        批量传入多段文本，一次性调用接口拿到所有向量。

        相比循环调用 get_embedding，批量接口只需一次网络往返，速度大幅提升。

        参数:
            text_list: 文本列表，例如 ["text1", "text2", ...]

        返回:
            list[list[float]]，第 i 个元素是 text_list[i] 对应的向量
        """
        # ---------- 参数校验 ----------
        if not isinstance(text_list, list):
            raise EmbeddingError(
                f"text_list 必须是 list，实际类型 {type(text_list).__name__}"
            )
        if len(text_list) == 0:
            raise EmbeddingError("text_list 不能为空列表")

        # 逐条校验：必须是字符串且不能为空
        for idx, t in enumerate(text_list):
            if not isinstance(t, str):
                raise EmbeddingError(f"第 {idx} 个元素不是字符串: {type(t).__name__}")
            if not t.strip():
                raise EmbeddingError(f"第 {idx} 个元素是空字符串或纯空白")

        # 数量超限时自动分批（避免一次塞太多被接口拒绝）
        all_vectors: list[list[float]] = []
        for start in range(0, len(text_list), self.MAX_BATCH_SIZE):
            batch = text_list[start:start + self.MAX_BATCH_SIZE]
            batch_vectors = self._call_api(batch)
            all_vectors.extend(batch_vectors)

        return all_vectors

    def _call_api(self, batch: list[str]) -> list[list[float]]:
        """
        真正调 dashscope SDK 的那一层（私有方法）。

        dashscope.TextEmbedding.call 的返回结构：
            Response {
                status_code: 200,           # HTTP 状态码（成功时 200）
                request_id: "xxx",           # 请求 ID（排查问题用）
                code: "",                    # 错误码（成功时为空）
                message: "",                 # 错误信息（成功时为空）
                output: {
                    "embeddings": [
                        {"text_index": 0, "embedding": [0.01, 0.02, ...]},
                        {"text_index": 1, "embedding": [0.03, 0.04, ...]},
                    ]
                },
                usage: {"total_tokens": 100}
            }
        """
        # 构造调用参数
        kwargs = {
            "model": EMBEDDING_MODEL,
            "input": batch,
        }
        # 部分模型支持 dimension 参数自定义向量维度
        # qwen3.7-text-embedding / text-embedding-v4 / v3 支持
        if EMBEDDING_DIMENSIONS is not None:
            kwargs["dimension"] = EMBEDDING_DIMENSIONS

        # ---------- 调用 SDK ----------
        try:
            resp = TextEmbedding.call(**kwargs)
        except Exception as e:
            raise EmbeddingError(f"dashscope 调用异常: {e}")

        # ---------- 检查响应状态 ----------
        # 成功时 status_code == 200，失败时 != 200
        if resp.status_code != 200:
            code = getattr(resp, "code", "Unknown")
            message = getattr(resp, "message", "Unknown error")
            raise EmbeddingError(
                f"接口返回错误 (status={resp.status_code}, code={code}): {message}"
            )

        # ---------- 解析响应 ----------
        # resp.output["embeddings"] 是一个列表
        # 每个元素是 {"text_index": 0, "embedding": [0.01, 0.02, ...]}
        embeddings = resp.output["embeddings"]
        if not isinstance(embeddings, list):
            raise EmbeddingError(
                f"响应 embeddings 字段不是列表，实际类型 {type(embeddings).__name__}"
            )
        if len(embeddings) != len(batch):
            raise EmbeddingError(
                f"返回向量数不匹配：请求 {len(batch)} 条，实际返回 {len(embeddings)} 条"
            )

        # 按 text_index 排序后提取向量（保险起见，确保顺序和输入一致）
        embeddings_sorted = sorted(embeddings, key=lambda x: x["text_index"])
        result = []
        for i, item in enumerate(embeddings_sorted):
            if "embedding" not in item:
                raise EmbeddingError(f"第 {i} 条结果缺少 'embedding' 字段: {item}")
            vec = item["embedding"]
            if not isinstance(vec, list) or len(vec) == 0:
                raise EmbeddingError(
                    f"第 {i} 条向量格式异常: 类型={type(vec).__name__}"
                )
            result.append([float(x) for x in vec])

        return result


# ==================== 自测入口 ====================
if __name__ == "__main__":
    try:
        client = EmbeddingClient()
        print(f"已加载配置: model={EMBEDDING_MODEL}, dimensions={EMBEDDING_DIMENSIONS}")

        # 测试 1：单条文本
        print("\n===== 测试 1：单条文本 =====")
        vec = client.get_embedding("如何加载 Python 源码文件？")
        print(f"向量维度: {len(vec)}")
        print(f"前 5 个分量: {vec[:5]}")

        # 测试 2：批量文本
        print("\n===== 测试 2：批量文本 =====")
        texts = [
            "def load_source_files(): 加载所有 Python 文件",
            "class InMemoryVectorStore: 向量存储",
            "BM25 关键词检索",
        ]
        vectors = client.get_batch_embedding(texts)
        print(f"输入 {len(texts)} 条，返回 {len(vectors)} 个向量")
        for i, v in enumerate(vectors):
            print(f"  [{i}] 维度={len(v)}, 前3个值={v[:3]}")

        # 测试 3：空文本校验
        print("\n===== 测试 3：空文本校验 =====")
        try:
            client.get_embedding("")
        except EmbeddingError as e:
            print(f"正确捕获异常: {e}")

        # 测试 4：空列表校验
        print("\n===== 测试 4：空列表校验 =====")
        try:
            client.get_batch_embedding([])
        except EmbeddingError as e:
            print(f"正确捕获异常: {e}")

    except EmbeddingError as e:
        print(f"EmbeddingError: {e}")
    except Exception as e:
        print(f"未预期异常: {e}")
