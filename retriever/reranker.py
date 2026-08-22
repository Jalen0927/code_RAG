"""
retriever/reranker.py — Reranker 重排器（基于 Cross-Encoder 模型）。

职责：
  - 对多路召回的候选分片做 Cross-Encoder 精排
  - 把重排分和融合分按 6:4 混合，输出最终排序依据
  - 模型懒加载，首次 rerank 时才下载/加载

什么是重排？为什么需要重排？
  召回阶段（向量/BM25）为了速度，只能用"轻量但粗糙"的方法从海量数据里捞出几十个候选。
  重排阶段用"慢但精准"的大模型对这几十个候选逐个精确打分，重新排序后取最相关的几个。
  简单理解：召回是"海选"，重排是"决赛"。

Cross-Encoder 为什么比召回阶段的打分更准？
  - 双塔模型（向量召回）：query 和 doc 各自独立编码成向量，最后算余弦相似度。
    query 和 doc 在编码时彼此看不到对方 → "瞎子摸象"，语义匹配粗糙。
  - Cross-Encoder：把 query 和 doc 拼在一起送进同一个 Transformer，模型每一层都能
    同时看到 query 和 doc 的所有 token，可以做 token 级别的精细交叉注意力。
    这就像"睁着眼睛对比" → 精度高得多，但速度慢得多（必须每对都过一次完整模型）。
"""

from __future__ import annotations

import os
import sys

# 方便直接运行本文件做自测
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 必须在 import sentence_transformers 之前触发 config.py
# config.py 会读取 .env 里的 HF_ENDPOINT 并设置到 os.environ
# 这样 sentence_transformers 下载模型时就会走 hf-mirror.com 镜像，不会卡在 huggingface.co
import config  # noqa: F401  仅为了副作用，不使用其中的变量

from typing import Optional


class Reranker:
    """
    使用 Cross-Encoder 对多路召回的候选分片做精排。

    使用前提：
        - sentence-transformers 库已安装
        - 首次使用会从 HuggingFace 下载模型（约 80MB），需要网络
        - 后续会缓存在 ~/.cache/huggingface/，不再重复下载
    """

    # ———————— 可调常量 ————————
    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # 候选数量上限，防止用户传太多把模型跑爆
    MAX_CANDIDATES = 200
    # top_n 上限
    MAX_TOP_N = 100
    # 重排分与原始融合分的混合比例：
    #   blend_ratio=0.7 表示最终分 = 0.7 * rerank_score + 0.3 * original_score
    #   这样既保留 Cross-Encoder 的语义精排能力，又不丢失关键词加权的优势
    BLEND_RATIO = 0.6
    # ———————————————————————

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME,
                 device: Optional[str] = None):
        """
        参数:
            model_name: HuggingFace 上的 Cross-Encoder 模型名
                        默认 'cross-encoder/ms-marco-MiniLM-L-6-v2'
                        这是一个轻量模型（~80MB），用 MS MARCO 检索数据训练，专做相关性打分
            device:     'cpu' / 'cuda' / None（让库自动选）
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"model_name 必须是非空字符串，实际 {model_name!r}")

        self.model_name = model_name
        self.device = device
        self.model = None  # 真正的 CrossEncoder 对象，延迟到 _ensure_model 里加载

    # ============================================================
    #  模型加载（延迟加载，避免 import 时就崩）
    # ============================================================

    def _ensure_model(self):
        """
        懒加载模型：第一次 rerank 时才真正加载。
        好处：
          1. 实例化 Reranker 时不卡顿，启动快
          2. 模型下载/加载失败不影响别的功能
          3. 如果从来不调用 rerank，就完全不需要模型
        """
        if self.model is not None:
            return  # 已经加载过，直接返回

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "未安装 sentence-transformers 库，请运行: pip install sentence-transformers"
            ) from e

        try:
            print(f"[Reranker] 正在加载 Cross-Encoder 模型: {self.model_name}")
            # CrossEncoder 内部会自动从 HuggingFace 下载并缓存模型
            # max_length=512 是大多数 MiniLM 模型的标准长度上限
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512,
            )
            print(f"[Reranker] 模型加载完成")
        except Exception as e:
            # 加载失败的常见原因：网络不通下不到模型 / 路径写错 / 内存不够
            self.model = None
            raise RuntimeError(
                f"Cross-Encoder 模型加载失败: {e}\n"
                f"常见原因: 1) 网络不通，无法从 HuggingFace 下载  "
                f"2) 模型名写错  3) 内存不足"
            ) from e

    # ============================================================
    #  对外 API
    # ============================================================

    def rerank(self, query: str, candidates: list[dict],
               top_n: int = 5) -> list[dict]:
        """
        对候选分片做 Cross-Encoder 重排。

        参数:
            query:       用户原始问题文本
            candidates:  多路召回输出的候选列表，每条至少要包含 "chunk_text" 字段
                         （其他字段如 file_path / start_line / 各类分数 都会原样保留）
            top_n:       重排后返回的结果数量

        返回:
            list[dict]，按 rerank_score 降序排列，每条带上：
            {
                ...原有所有字段...,        # file_path / chunk_text / start_line / score / ...
                "rerank_score": float,    # Cross-Encoder 打的相关性分数（越大越相关）
            }
        """
        # ---------- 参数校验 ----------
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"query 必须是非空字符串，实际 {query!r}")

        if not isinstance(candidates, list):
            raise ValueError(
                f"candidates 必须是 list[dict]，实际 {type(candidates).__name__}"
            )
        if len(candidates) == 0:
            return []

        if len(candidates) > self.MAX_CANDIDATES:
            print(
                f"[警告] 候选数量 {len(candidates)} 超过上限 {self.MAX_CANDIDATES}，"
                f"已自动截断。建议召回阶段就控制 top_k。"
            )
            candidates = candidates[:self.MAX_CANDIDATES]

        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError(f"top_n 必须是正整数，实际 {top_n}")
        if top_n > self.MAX_TOP_N:
            print(f"[警告] top_n={top_n} 超过上限，已截断为 {self.MAX_TOP_N}")
            top_n = self.MAX_TOP_N

        # 检查每个候选必须有 chunk_text 字段
        for idx, c in enumerate(candidates):
            if not isinstance(c, dict):
                raise ValueError(f"第 {idx} 条候选不是 dict: {type(c).__name__}")
            if "chunk_text" not in c:
                raise ValueError(f"第 {idx} 条候选缺少 'chunk_text' 字段")

        # ---------- 真正开始重排 ----------
        try:
            self._ensure_model()
        except RuntimeError as e:
            print(f"[警告] 模型加载失败，重排跳过，按原顺序返回前 {top_n} 条: {e}")
            # 兜底：模型加载不上，就把原候选的前 top_n 条原样返回，rerank_score 记 0
            return [
                {**c, "rerank_score": 0.0}
                for c in candidates[:top_n]
            ]

        # ---------- 构造 Cross-Encoder 输入并打分 ----------
        # Cross-Encoder 的输入是 [(text_a, text_b), ...] 的列表
        # 模型会输出每对的"相关性分数"（一个标量，越大越相关）
        pairs = [(query, c["chunk_text"]) for c in candidates]

        # batch_size 控制一次喂多少对，防止内存爆掉
        try:
            scores = self.model.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as e:
            print(f"[警告] Cross-Encoder 推理失败，按原顺序返回前 {top_n} 条: {e}")
            return [
                {**c, "rerank_score": 0.0}
                for c in candidates[:top_n]
            ]

        # ---------- 分数归一化 + 混合 ----------
        # scores 是 numpy 数组，第 i 个元素对应第 i 个候选
        raw_scores = [float(scores[i]) for i in range(len(candidates))]

        # Cross-Encoder 原始分做 min-max 归一化到 [0, 1]
        if raw_scores:
            min_s, max_s = min(raw_scores), max(raw_scores)
            if max_s - min_s > 1e-12:
                norm_rerank = [(s - min_s) / (max_s - min_s) for s in raw_scores]
            else:
                norm_rerank = [1.0 for _ in raw_scores]
        else:
            norm_rerank = raw_scores

        # 取每个候选的原始融合分（来自 MultiRetriever 的 score 字段）
        orig_scores = [float(c.get("score", 0.0)) for c in candidates]

        # 原始融合分也做 min-max 归一化（因为乘法增强后可能 > 1）
        if orig_scores:
            min_os, max_os = min(orig_scores), max(orig_scores)
            if max_os - min_os > 1e-12:
                norm_orig = [(s - min_os) / (max_os - min_os) for s in orig_scores]
            else:
                norm_orig = [1.0 for _ in orig_scores]
        else:
            norm_orig = orig_scores

        # 混合分 = BLEND_RATIO * CE归一化分 + (1 - BLEND_RATIO) * 融合归一化分
        blend = self.BLEND_RATIO
        final_scores = [
            blend * nr + (1 - blend) * no
            for nr, no in zip(norm_rerank, norm_orig)
        ]

        # 组装结果
        scored = []
        for i, c in enumerate(candidates):
            scored.append((
                {
                    **c,
                    "rerank_raw_score": raw_scores[i],
                    "rerank_norm_score": norm_rerank[i],
                    "orig_fusion_score": orig_scores[i],
                    "orig_fusion_norm": norm_orig[i],
                    "rerank_score": final_scores[i],
                },
                i,
            ))

        # 按混合分从高到低排
        scored.sort(key=lambda x: x[0]["rerank_score"], reverse=True)

        # ---------- 取前 top_n ----------
        return [item[0] for item in scored[:top_n]]

    # ============================================================
    #  辅助方法
    # ============================================================

    def is_ready(self) -> bool:
        """模型是否已加载（不触发加载）。"""
        return self.model is not None

    def __len__(self) -> int:
        """返回 0 / 1 表示模型是否就绪（语义上不太合理，但保留以防误用）。"""
        return 1 if self.model is not None else 0


# ==================== 自测入口 ====================
if __name__ == "__main__":
    reranker = Reranker()

    # 模拟"多路召回"返回的结果，每条带 chunk_text 和一些假分数
    fake_candidates = [
        {
            "file_path": "file_loader.py",
            "start_line": 1,
            "chunk_text": (
                "def load_source_files(source_dir):\n"
                "    for root, dirs, files in os.walk(source_dir):\n"
                "        if file.endswith('.py'):\n"
                "            content = open(full_path).read()\n"
                "    return documents  # 加载所有 Python 源码文件\n"
            ),
            "score": 0.95,
            "vector_score": 0.92,
            "bm25_score": 3.5,
            "sources": ["vector", "bm25"],
        },
        {
            "file_path": "main.py",
            "start_line": 10,
            "chunk_text": (
                "# 程序主入口\n"
                "if __name__ == '__main__':\n"
                "    user_query = input('请输入您的问题: ')\n"
                "    print('正在为您检索相关文档...')\n"
            ),
            "score": 0.42,
            "vector_score": 0.30,
            "bm25_score": 0.0,
            "sources": ["vector"],
        },
        {
            "file_path": "code_splitter.py",
            "start_line": 50,
            "chunk_text": (
                "def _parse_line_blocks(content):\n"
                "    # 把代码按 def/class 拆成逻辑块\n"
                "    # 这样切出来的 chunk 不会在函数中间断开\n"
                "    return blocks\n"
            ),
            "score": 0.75,
            "vector_score": 0.65,
            "bm25_score": 2.1,
            "sources": ["vector", "bm25"],
        },
        {
            "file_path": "utils.py",
            "start_line": 8,
            "chunk_text": (
                "def helper():\n"
                "    # 这是一个工具函数\n"
                "    pass\n"
            ),
            "score": 0.10,    # 召回阶段分数很低，可能是噪声候选
            "vector_score": 0.12,
            "bm25_score": 0.0,
            "sources": ["vector"],
        },
    ]

    query = "怎么加载项目里的所有 Python 文件？"

    print(f"查询: {query}\n")
    print("===== 召回阶段排序（按综合分 score 降序）=====")
    for c in sorted(fake_candidates, key=lambda x: x["score"], reverse=True):
        print(f"  score={c['score']:.3f}  {c['file_path']}:{c['start_line']}")

    print("\n===== 重排后排序（按 rerank_score 降序）=====")
    try:
        reranked = reranker.rerank(query, fake_candidates, top_n=3)
        for r in reranked:
            print(
                f"  rerank={r['rerank_score']:+.4f}  "
                f"(原 score={r['score']:.3f})  "
                f"{r['file_path']}:{r['start_line']}"
            )
    except Exception as e:
        print(f"重排失败（可能是网络问题下不到模型）: {e}")
