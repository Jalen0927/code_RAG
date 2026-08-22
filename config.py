"""
config.py — 全局配置加载模块。

职责：
  - 从 .env 文件加载环境变量（API Key、模型名、维度等）
  - 配置 HuggingFace 镜像与离线模式（必须在 import sentence-transformers 之前生效）
  - 统一暴露配置常量供其他模块导入使用
"""

import os

from dotenv import load_dotenv

# 加载环境变量，强制覆盖系统旧缓存
load_dotenv(override=True)

# ============================================================
# HuggingFace 镜像与离线模式配置
# 必须在 import sentence-transformers / reranker 之前生效，
# 否则国内访问 huggingface.co 会被中断
# ============================================================
_hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint

# 离线模式：模型已下载缓存后，设为 1 避免每次启动都联网检查更新
# 注意：第一次使用还没下载模型时，不要开启（设为 0 或在 .env 里注释掉）
_hf_offline = os.getenv("HF_HUB_OFFLINE", "").strip()
if _hf_offline:
    os.environ["HF_HUB_OFFLINE"] = _hf_offline

# ============================================================
# LLM 大模型配置（小米 MiMo，OpenAI 协议兼容）
# ============================================================
LLM_API_KEY = os.getenv("MIMO_API_KEY")
LLM_BASE_URL = os.getenv("MIMO_BASE_URL")
LLM_MODEL = os.getenv("MIMO_MODEL")
LLM_TEMP = float(os.getenv("MIMO_TEMPERATURE", "0.1"))

# 统一小写判断，兼容 TRUE / True / true 写法
_env_close_think = os.getenv("LLM_CLOSE_THINK", "false").strip().lower()
LLM_CLOSE_THINK = _env_close_think == "true"

LLM_MAX_TOKENS = int(os.getenv("MIMO_MAX_TOKENS", "4096"))

# ============================================================
# Embedding 向量模型配置（阿里云 DashScope）
# 只需要 DASHSCOPE_API_KEY，不需要 BASE_URL，SDK 内部自动处理
# ============================================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 模型名统一小写，避免用户大小写不规范被接口拒
_env_embedding_model = os.getenv("EMBEDDING_MODEL")
EMBEDDING_MODEL = _env_embedding_model.lower() if _env_embedding_model else _env_embedding_model

# 向量维度：仅 qwen3.7-text-embedding / text-embedding-v4/v3 支持
# 空字符串或未配置时不传此参数，使用模型默认维度
_env_dimensions = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
EMBEDDING_DIMENSIONS = int(_env_dimensions) if _env_dimensions else None

# ============================================================
# Agent 运行配置
# ============================================================
SAFE_WORKSPACE = os.getenv("SAFE_WORKSPACE", "workspace")
MAX_REACT_LOOP = int(os.getenv("MAX_REACT_LOOP", "10"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# 调试日志开关
DEBUG = os.getenv("DEBUG", "false").strip().lower() == "true"

# 创建安全工作目录
os.makedirs(SAFE_WORKSPACE, exist_ok=True)
