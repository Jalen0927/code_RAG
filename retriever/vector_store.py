"""
retriever/vector_store.py — 简易内存向量存储（纯 NumPy 手搓版）。

职责：
  - 把向量按行拼成 NumPy 矩阵存储，元数据按相同顺序存为列表
  - 查询时用余弦相似度算查询向量与库内所有向量的相似度，取 top_k
  - 一切只存在内存里，进程退出即销毁，不依赖任何第三方向量数据库
"""

from __future__ import annotations

import numpy as np


class InMemoryVectorStore:
    """
    内存向量存储。

    内部数据结构：
        self.vectors: np.ndarray, shape = (n_total_chunks, embedding_dim)
            每一行对应一个 chunk 的向量（float32）。
        self.metadatas: list[dict]
            第 i 个元素对应 self.vectors[i] 的元数据，格式
            {"file_path": str, "chunk_text": str, "start_line": int}
    """

    # ———————— 可调常量 ————————
    # top_k 的合理范围，防止用户传太夸张的数字
    MAX_ALLOWED_TOP_K = 1000
    # 相似度低于这个阈值的结果会被过滤（防止返回完全不相关的内容）
    MIN_SIMILARITY_THRESHOLD = -1.0  # 余弦相似度范围 [-1, 1]，默认不过滤
    # ———————————————————————

    def __init__(self, embedding_dim: int | None = None):
        """
        参数:
            embedding_dim: 向量维度。可以传 None，在第一次 add 时根据向量自动推断。
        """
        self.embedding_dim = embedding_dim

        # 真正存数据的容器
        self.vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.metadatas: list[dict] = []

    # ============================================================
    #  工具方法：统一校验 + 余弦相似度计算（纯 NumPy 手搓）
    # ============================================================

    def _validate_vectors(self, vectors):
        """把传入的向量统一转成 shape=(N, D) 的 float32 numpy 矩阵。"""
        arr = np.asarray(vectors, dtype=np.float32)

        # 允许用户传单个向量 (D,)，自动包装成 (1, D)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError(
                f"向量维度不对：期望 1D 或 2D，实际 shape = {arr.shape}"
            )

        n, d = arr.shape
        # 如果初始化时没传维度，就用这批向量的维度作为固定维度
        if self.embedding_dim is None:
            self.embedding_dim = d
        if d != self.embedding_dim:
            raise ValueError(
                f"向量维度不一致：本库固定维度 = {self.embedding_dim}，"
                f"传入向量维度 = {d}"
            )
        if n == 0:
            raise ValueError("不能传入空的向量列表")

        return arr

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, store_vecs: np.ndarray) -> np.ndarray:
        """
        手写余弦相似度：一次性计算 query_vec 与 store_vecs 中每行的相似度。

        余弦相似度公式（二维平面理解）:
                          a · b
            sim(a, b) = ────────────
                        ||a|| · ||b||

        也就是"两个向量的点积"除以"各自长度的乘积"。
        结果范围 [-1, 1]：
            = 1   → 两个向量方向完全一致（语义最相似）
            = 0   → 完全正交（没关系）
            = -1  → 方向完全相反
        """
        # —— 1. 算分母：两个向量的"长度"（L2 范数，又叫模长）——
        q_norm = np.linalg.norm(query_vec)
        s_norms = np.linalg.norm(store_vecs, axis=1)

        # 防止除零：如果某个向量长度是 0（全 0 向量），把长度当 1，避免出现 nan
        if q_norm < 1e-12:
            q_norm = 1.0
        s_norms = np.where(s_norms < 1e-12, 1.0, s_norms)

        # —— 2. 算分子：点积（dot product）——
        dot_products = store_vecs @ query_vec

        # —— 3. 分子 / 分母 = 余弦相似度 ——
        similarities = dot_products / q_norm / s_norms

        # 数值稳定性兜底：浮点误差可能让结果超出 [-1, 1] 一点点，强行裁剪
        similarities = np.clip(similarities, -1.0, 1.0)

        return similarities

    # ============================================================
    #  对外 API
    # ============================================================

    def add(self, vectors, metadatas) -> int:
        """
        批量添加向量和对应的 chunk 元数据。

        参数:
            vectors:   list[list[float]] / np.ndarray
                       shape 可以是 (N, D) 或单条 (D,)
            metadatas: list[dict]，每个 dict 至少包含
                       {"file_path", "chunk_text", "start_line"}
                       如果只加 1 条也可以直接传 dict。

        返回:
            int: 本次实际新增的条目数量。
        """
        # --------- 参数校验 ---------
        vec_arr = self._validate_vectors(vectors)
        n_vecs = vec_arr.shape[0]

        # 元数据也允许传单个 dict，自动包装成列表
        if isinstance(metadatas, dict):
            metadatas = [metadatas]
        if not isinstance(metadatas, list):
            raise ValueError(
                f"metadatas 类型不对：期望 list[dict] 或 dict，实际 {type(metadatas)}"
            )
        if len(metadatas) != n_vecs:
            raise ValueError(
                f"数量不匹配：vectors 有 {n_vecs} 条，但 metadatas 有 {len(metadatas)} 条"
            )

        # 检查每条元数据必须包含三个字段
        required_keys = {"file_path", "chunk_text", "start_line"}
        for idx, md in enumerate(metadatas):
            missing = required_keys - md.keys() if isinstance(md, dict) else required_keys
            if missing:
                raise ValueError(
                    f"第 {idx} 条元数据缺少字段: {sorted(missing)}"
                )

        # --------- 真正写入内存 ---------
        # 如果这是第一次写入，初始化 vectors 的列数
        if self.vectors.shape[1] == 0:
            self.vectors = np.empty((0, self.embedding_dim), dtype=np.float32)

        # 按行拼接：把新的向量追加到矩阵底部
        self.vectors = np.vstack([self.vectors, vec_arr])
        # 元数据列表也要按同样顺序追加
        self.metadatas.extend(metadatas)

        return n_vecs

    def search(self, query_vector, top_k: int = 5,
               min_similarity: float | None = None) -> list[dict]:
        """
        查询最相似的 top_k 个 chunk。

        参数:
            query_vector:   list[float] / np.ndarray，shape=(D,) 查询向量
            top_k:          返回结果数量（最大不得超过 MAX_ALLOWED_TOP_K）
            min_similarity: 相似度下限，低于这个分数的结果会被过滤。
                            默认使用类常量 MIN_SIMILARITY_THRESHOLD。

        返回:
            list[dict]，按相似度从高到低排列，每条：
            {
                "file_path": str,
                "chunk_text": str,
                "start_line": int,
                "score": float,   # 余弦相似度分数，范围 [-1, 1]
            }
        """
        # --------- 参数校验 ---------
        if self.vectors.shape[0] == 0:
            print("[提示] 向量库为空，返回空列表")
            return []

        q_vec = np.asarray(query_vector, dtype=np.float32)
        if q_vec.ndim != 1:
            raise ValueError(
                f"查询向量必须是 1D 数组，实际 shape = {q_vec.shape}"
            )
        if q_vec.shape[0] != self.embedding_dim:
            raise ValueError(
                f"查询向量维度不对：期望 {self.embedding_dim}，实际 {q_vec.shape[0]}"
            )

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k 必须是正整数，实际 {top_k}")
        if top_k > self.MAX_ALLOWED_TOP_K:
            print(
                f"[警告] top_k={top_k} 超过上限 {self.MAX_ALLOWED_TOP_K}，"
                f"已自动截断为上限值"
            )
            top_k = self.MAX_ALLOWED_TOP_K

        threshold = (
            min_similarity if min_similarity is not None
            else self.MIN_SIMILARITY_THRESHOLD
        )

        # --------- 计算相似度 + 排序 ---------
        scores = self._cosine_similarity(q_vec, self.vectors)

        # np.argsort 从小到大返回下标；[::-1] 翻转，变成从大到小（相似度从高到低）
        sorted_indices = np.argsort(scores)[::-1]

        # --------- 取 top_k + 过滤阈值 ---------
        results = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if score < threshold:
                break  # 排好序的，第一个低于阈值后面也都低于，直接退出
            md = self.metadatas[idx]
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
        """清空存储的所有向量和元数据。"""
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.metadatas.clear()

    # ============================================================
    #  辅助属性
    # ============================================================

    def __len__(self) -> int:
        return len(self.metadatas)

    def is_empty(self) -> bool:
        return len(self.metadatas) == 0


# ==================== 自测入口 ====================
if __name__ == "__main__":
    store = InMemoryVectorStore()

    # —— 造一些假向量和假元数据做演示 ——
    fake_vecs = [
        [1.0, 0.0, 0.0, 0.0],   # 0号：X方向单位向量
        [0.0, 1.0, 0.0, 0.0],   # 1号：Y方向
        [1.0, 1.0, 0.0, 0.0],   # 2号：XY角平分线（与 0/1 号都相关）
        [0.0, 0.0, 1.0, 0.0],   # 3号：Z方向，与上面都正交
    ]
    fake_meta = [
        {"file_path": "a.py", "chunk_text": "def foo(): pass",     "start_line": 1},
        {"file_path": "b.py", "chunk_text": "x = 1",              "start_line": 10},
        {"file_path": "c.py", "chunk_text": "class Bar: pass",    "start_line": 20},
        {"file_path": "d.py", "chunk_text": "import os",          "start_line": 30},
    ]

    added = store.add(fake_vecs, fake_meta)
    print(f"add 完成，共 {added} 条，当前库总条数 = {len(store)}")

    print("\n===== 查询 [1, 0, 0, 0]，top_k = 3 =====")
    for r in store.search([1.0, 0.0, 0.0, 0.0], top_k=3):
        print(f"  score={r['score']:.4f}  path={r['file_path']}  "
              f"line={r['start_line']}  text='{r['chunk_text']}'")

    print("\n===== 查询 [0, 0, 1, 0]，top_k = 2 =====")
    for r in store.search([0.0, 0.0, 1.0, 0.0], top_k=2):
        print(f"  score={r['score']:.4f}  path={r['file_path']}  "
              f"line={r['start_line']}  text='{r['chunk_text']}'")

    store.clear()
    print(f"\nclear 后 is_empty = {store.is_empty()}，再次查询返回 = {store.search([1,0,0,0])}")
