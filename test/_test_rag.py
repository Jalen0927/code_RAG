"""
自动化 RAG 验证脚本。
模拟用户提问，测试切片修复后的检索效果是否更准确。

测试策略：
    1. 提问 ReActScheduler.__init__ → 期望命中包含 def __init__ 的分片
    2. 提问 ReActScheduler.run 方法 → 期望命中完整的 run 方法
    3. 提问 _build_system_prompt → 期望命中该方法
    4. 提问文件管理工具 → 期望命中工具相关代码
"""

import sys
import os

# Bug 4 修复：确保 Windows 终端能正确显示 emoji（GBK 终端兜底）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 项目根目录是 test 目录的上一级，加入搜索路径以便 import 根目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.file_loader import load_source_files
from ingestion.code_splitter import split_python_files
from embedding_client import EmbeddingClient, EmbeddingError
from retriever.vector_store import InMemoryVectorStore
from retriever.bm25_retriever import Bm25Retriever
from retriever.multi_retriever import MultiRetriever
from retriever.reranker import Reranker
from llm_client import MiMoClient, LLMRequestError
from config import EMBEDDING_DIMENSIONS


TEST_QUESTIONS = [
    "ReActScheduler 的初始化逻辑是什么？",
    "ReActScheduler 的 run 方法做了什么？",
    "_build_system_prompt 方法的作用是什么？",
    "ReActScheduler 有哪些方法？",
]


def build_index():
    print("=" * 60)
    print("📂 加载源码...")
    docs = load_source_files("source_code")
    print(f"   ✅ {len(docs)} 个文件")

    print("✂️  代码感知切片...")
    chunks = split_python_files(docs)
    print(f"   ✅ {len(chunks)} 个分片")

    # 打印 ReActScheduler 相关分片的 start_line
    react_chunks = [c for c in chunks if "react_scheduler" in c["file_path"]]
    print(f"   📋 ReActScheduler 相关分片 ({len(react_chunks)} 个):")
    for c in react_chunks:
        preview = c["chunk_text"][:80].replace("\n", " ")
        print(f"      行{c['start_line']:>4} | {preview}...")

    print("🧠 向量化...")
    emb_client = EmbeddingClient()
    texts = [c["chunk_text"] for c in chunks]
    vectors = emb_client.get_batch_embedding(texts)
    print(f"   ✅ {len(vectors)} 个向量")

    vector_store = InMemoryVectorStore(embedding_dim=EMBEDDING_DIMENSIONS)
    vector_store.add(vectors, chunks)

    bm25 = Bm25Retriever()
    bm25.build_index(chunks)

    multi = MultiRetriever(vector_store, bm25)
    reranker = Reranker()
    llm = MiMoClient()

    return {
        "emb_client": emb_client,
        "multi": multi,
        "reranker": reranker,
        "llm": llm,
        "chunks": chunks,
    }


def test_query(query: str, components: dict) -> dict:
    """测试单个问题，返回详细结果"""
    emb_client = components["emb_client"]
    multi = components["multi"]
    reranker = components["reranker"]

    result = {
        "question": query,
        "candidates": [],
        "top3": [],
        "success": False,
    }

    try:
        query_vec = emb_client.get_embedding(query)
        candidates = multi.search(query_vec, query, top_k=20)
        result["candidates"] = [
            {
                "file_path": c["file_path"],
                "start_line": c["start_line"],
                "score": c["score"],
                "preview": c["chunk_text"][:60].replace("\n", " "),
            }
            for c in candidates
        ]

        if not candidates:
            return result

        top_n = reranker.rerank(query, candidates, top_n=3)
        result["top3"] = [
            {
                "file_path": r["file_path"],
                "start_line": r["start_line"],
                "rerank_score": r["rerank_score"],
                "rerank_raw_score": r.get("rerank_raw_score", 0),
                "rerank_norm_score": r.get("rerank_norm_score", 0),
                "orig_fusion_score": r.get("orig_fusion_score", 0),
                "orig_fusion_norm": r.get("orig_fusion_norm", 0),
                "keyword_boost": r.get("keyword_boost", 0),
                "source_boost": r.get("source_boost", 0),
                "base_score": r.get("base_score", 0),
                "kw_multiplier": r.get("kw_multiplier", 0),
                "src_multiplier": r.get("src_multiplier", 0),
                "preview": r["chunk_text"][:80].replace("\n", " "),
                "full_text": r["chunk_text"],
            }
            for r in top_n
        ]
        result["success"] = bool(top_n)

    except Exception as e:
        result["error"] = str(e)

    return result


def evaluate_results(results: list):
    """评估检索质量"""
    print("\n" + "=" * 60)
    print("📊 检索质量评估报告")
    print("=" * 60)

    for r in results:
        q = r["question"]
        print(f"\n❓ 问题: {q}")

        if r.get("error"):
            print(f"   ❌ 出错: {r['error']}")
            continue

        top3 = r.get("top3", [])
        if not top3:
            print("   ⚠️  无检索结果")
            continue

        # 检查 Top3 是否包含目标内容
        for i, t in enumerate(top3):
            kw_b = t.get("keyword_boost", 0)
            src_b = t.get("source_boost", 0)
            orig = t.get("orig_fusion_score", 0)
            orig_norm = t.get("orig_fusion_norm", 0)
            ce_raw = t.get("rerank_raw_score", 0)
            ce_norm = t.get("rerank_norm_score", 0)
            print(f"   [{i+1}] 混合分 {t['rerank_score']:.4f}  "
                  f"[CE_norm={ce_norm:.2f} orig_norm={orig_norm:.2f}]  "
                  f"(CE_raw={ce_raw:.2f} fusion={orig:.3f} kw={kw_b:.2f} src={src_b:.2f})  "
                  f"📄 {t['file_path']}:{t['start_line']}")
            print(f"       preview: {t['preview']}...")

        # 智能评估
        if "初始化" in q or "__init__" in q:
            init_found = any("__init__" in t["full_text"] for t in top3)
            status = "✅ 命中 __init__" if init_found else "❌ 未命中 __init__"
            print(f"   评估: {status}")

        elif "run 方法" in q:
            run_found = any("def run(" in t["full_text"] for t in top3)
            status = "✅ 命中 run 方法" if run_found else "❌ 未命中 run 方法"
            print(f"   评估: {status}")

        elif "_build_system_prompt" in q:
            prompt_found = any("_build_system_prompt" in t["full_text"] for t in top3)
            status = "✅ 命中 _build_system_prompt" if prompt_found else "❌ 未命中 _build_system_prompt"
            print(f"   评估: {status}")

        elif "哪些方法" in q:
            methods_found = sum(1 for t in top3 if "def " in t["full_text"])
            status = f"✅ 命中 {methods_found} 个方法定义" if methods_found > 0 else "❌ 未命中任何方法"
            print(f"   评估: {status}")

    print("\n" + "=" * 60)
    print("🎯 总结")
    print("=" * 60)
    total = len(results)
    success = sum(1 for r in results if r.get("success") and not r.get("error"))
    print(f"   测试问题: {total}")
    print(f"   检索成功: {success}")
    print(f"   失败: {total - success}")
    print(f"   成功率: {success/total*100:.0f}%")


def main():
    print("🤖 CodeRAG 切片修复验证测试")
    print("=" * 60)

    try:
        components = build_index()
    except Exception as e:
        print(f"❌ 索引构建失败: {e}")
        return

    print("\n" + "=" * 60)
    print("🧪 开始测试 4 个关键问题")
    print("=" * 60)

    results = []
    for q in TEST_QUESTIONS:
        print(f"\n🔍 测试: {q}")
        r = test_query(q, components)
        results.append(r)

    evaluate_results(results)

    # 额外测试：生成一次真实回答
    print("\n" + "=" * 60)
    print("💬 生成一次完整 RAG 回答（问题: ReActScheduler 如何初始化？）")
    print("=" * 60)

    try:
        query = "ReActScheduler 如何初始化？"
        emb_client = components["emb_client"]
        multi = components["multi"]
        reranker = components["reranker"]
        llm = components["llm"]

        query_vec = emb_client.get_embedding(query)
        candidates = multi.search(query_vec, query, top_k=20)
        top_n = reranker.rerank(query, candidates, top_n=3)

        context_parts = []
        for i, r in enumerate(top_n, start=1):
            context_parts.append(
                f"[片段 {i}] 来源: {r['file_path']} (起始行: {r['start_line']})\n"
                f"{r['chunk_text']}"
            )
        context = "\n\n".join(context_parts)

        prompt = (
            f"你是一个代码助手。请根据下方代码片段回答用户的问题。\n"
            f"如果片段中没有相关信息，请如实说明。\n\n"
            f"===== 代码片段 =====\n{context}\n\n"
            f"===== 用户问题 =====\n{query}\n"
        )

        answer = llm.chat_completion([{"role": "user", "content": prompt}])
        print(f"\n🤖 RAG 回答:\n{answer}")

    except Exception as e:
        print(f"❌ 回答生成失败: {e}")

    print("\n✅ 验证测试完成！")


if __name__ == "__main__":
    main()
