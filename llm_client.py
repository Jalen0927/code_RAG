"""
llm_client.py — 小米 MiMo 大模型客户端。

职责：
  - 封装 MiMo 大模型的 HTTP 调用（OpenAI 协议兼容）
  - 初始化时校验关键配置（fail-fast，避免启动假就绪）
  - 支持思考过程开关（由 .env 的 LLM_CLOSE_THINK 控制）
"""

import requests

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_CLOSE_THINK,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMP,
    REQUEST_TIMEOUT,
)


class LLMRequestError(Exception):
    """大模型 API 调用异常"""
    pass


class MiMoClient:
    """
    MiMo 大模型客户端。

    思考过程开关：由 .env 里的 LLM_CLOSE_THINK 唯一控制
        - LLM_CLOSE_THINK=false → 开启思考（消耗 token，展示推理过程）
        - LLM_CLOSE_THINK=true  → 关闭思考（省 token，只输出答案）
    """

    def __init__(self):
        # fail-fast：初始化时就校验关键配置
        # 避免"启动显示就绪、提问才报错"的假就绪现象
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

    def chat_completion(self, messages: list) -> str:
        """
        调用大模型生成回答。

        参数:
            messages: OpenAI 协议的消息列表 [{"role": ..., "content": ...}, ...]
        """
        # LLM_CLOSE_THINK=false → enable_thinking=True（开启思考）
        enable_thinking = not LLM_CLOSE_THINK

        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": LLM_TEMP,
            "max_tokens": LLM_MAX_TOKENS,
            "extra_body": {
                "enable_thinking": enable_thinking,
            },
        }

        try:
            resp = requests.post(
                url=f"{LLM_BASE_URL}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            # 解析响应：思考过程 + 正式回答
            msg_data = data["choices"][0]["message"]
            reasoning_text = msg_data.get("reasoning_content", "").strip()
            answer_text = msg_data["content"]

            # 开启思考 + 模型确实返回了思考内容，才打印
            if enable_thinking and reasoning_text:
                print("\n=====模型思考过程=====")
                print(reasoning_text)
                print("======================\n")

            return answer_text

        except Exception as e:
            raise LLMRequestError(f"模型请求异常: {str(e)}") from e
